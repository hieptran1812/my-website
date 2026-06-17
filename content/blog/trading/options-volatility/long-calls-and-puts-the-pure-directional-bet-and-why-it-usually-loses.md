---
title: "Long Calls and Puts: The Pure Directional Bet (and Why It Usually Loses)"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why buying a single call or put — the beginner's leverage trade — usually loses, the triple hurdle of direction, magnitude, and timing-while-IV-holds, and the rare regimes where a long single is the right tool."
tags: ["options", "volatility", "long-call", "long-put", "directional-trade", "theta", "vol-crush", "breakeven", "vega", "options-strategy"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Buying a single call or put is the most popular options trade and one of the worst, because it only pays when you are right on *direction* AND *magnitude* AND *timing*, all while implied volatility holds up. You are not making a directional bet; you are making a bet on a fast move that beats the premium you paid for time and volatility.
>
> - A long single faces a **triple hurdle**: the move must go the right way, clear breakeven (an OTM call can need **+8%** just to break even at expiry), and arrive before **theta** drains the option — and the **vol crush** can sink you even when all three pass.
> - **Theta and vega are why you lose while right.** A flat stock still drains the premium — an at-the-money call held three weeks loses about **\$0.72** to time alone — and a long-vega position into an event gets crushed when implied vol collapses after the news.
> - A long single is the *right* trade only in a narrow set of regimes: **cheap vol** (implied below the realized you expect), a **dated catalyst** that forces a fast move, a desire for **defined-risk convexity**, or replacing stock with a **deep-ITM call** that tracks the shares for a third of the capital.
> - **The one rule to remember:** a long single's enemy is *time*, not direction. If your edge is "the stock drifts higher over six months," buy the stock or a spread — never a single option, because the clock will beat you to the destination.

In late 2023 a retail trader posted a screenshot that made the rounds on options forums. He had bought call options on a large-cap tech name the afternoon before earnings, convinced the company would beat expectations. He was right. The company beat on revenue, beat on earnings, raised guidance, and the stock gapped up about **3%** in the after-hours session. By every measure his thesis was correct — and when the market opened the next morning, his calls were worth *less* than he paid. He had nailed the direction and still lost money. The replies were full of people who had lived the same story and didn't understand it either.

What happened to him is the single most important lesson in this entire series, and it is the reason this post exists. He did not lose because he was wrong. He lost because a long call is not a bet on direction — it is a bet on a *move that is large enough, fast enough, and arrives while implied volatility stays elevated.* Before earnings, the option's implied volatility was pumped up to roughly **55%** because the market was pricing in a big unknown move. After the report, that uncertainty was resolved, and implied volatility *crushed* down toward its normal **25%**. The 3% move he correctly predicted simply wasn't big enough to overcome the air coming out of the option. We will price this exact trade in a moment and watch the dollars disappear.

This is the post where we dissect the trade almost every options beginner makes first — buying a single call or put for leverage — and explain, with the Black-Scholes model doing the arithmetic, why it usually loses, when it is genuinely the right tool, and how to choose the strike and manage the position so you are not the person in that screenshot. You already know payoff diagrams, the Greeks, and the implied-versus-realized-volatility distinction from earlier in the series, so we will not re-teach those — we will *use* them. Let us begin with the figure that contains the whole problem.

![Long call and long put profit and loss at expiry versus the during-life curve with breakevens and the must-move-past zone marked](/imgs/blogs/long-calls-and-puts-the-pure-directional-bet-and-why-it-usually-loses-1.png)

Look at the left panel, the long call. The solid blue line is the payoff *at expiry* — the hockey stick you already know, kinked at the \$100 strike, with a maximum loss of the **\$5.47** premium and an upside that runs off the top of the chart. But the dashed amber line is the one that matters today: the option's value *during its life*, with three months still to run. Notice that even when the stock sits right at the strike, the dashed line is *below zero* — you are already losing the premium back to time. And notice the shaded red band between today's price (\$100) and the breakeven (\$105.47): that is the "must move past here" zone, the region where the stock can rally and you still lose money at expiry. The whole tragedy of the long single lives in that red band and that gap between the two curves. Everything below is an explanation of those two features.

## Foundations: what you are actually buying when you buy a single option

Before we can explain why long singles lose, we need to be precise about what one *is*. A **long single** is the simplest options position: you buy one option — a call if you are bullish, a put if you are bearish — and you hold nothing else against it. No other leg, no stock, no hedge. "Long" means you are the *buyer*: you paid the premium up front, your maximum loss is that premium, and your upside is the option's payoff. This is the trade newcomers reach for because it is the easiest to understand and because it promises leverage — a small amount of money controlling a large amount of stock.

Let us define that leverage precisely, because it is real and it is the bait. Recall from the [contract-mechanics post](/blog/trading/options-volatility/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement) that one standard equity option controls **100 shares**. So if a \$100 stock has an at-the-money call trading for a **\$5.47** premium, one contract costs \$547 and controls \$10,000 of stock. That is a **0.0547** ratio — your \$547 gives you exposure to \$10,000 of underlying, roughly **18-to-1** notional leverage. If the stock jumps to \$120, the call is worth at least its \$20 intrinsic value, so your \$547 became at least \$2,000 — a **+266%** return on a **+20%** stock move. That asymmetry, capped downside and geared upside, is genuinely attractive and genuinely the reason options exist. The problem is everything that has to happen in between.

The trap hidden in that leverage number is that **notional leverage is not the same as effective leverage**, and the gap is exactly delta. Your \$547 controls \$10,000 of notional stock, but the *at-the-money* call only moves about **\$0.56** for every \$1.00 the stock moves, because its delta is **0.56**. So on the *next* dollar of stock movement, your effective leverage is not 18-to-1 — it is delta times notional leverage, roughly **10-to-1**. For an out-of-the-money call with a delta of 0.40, the effective leverage on the next dollar is lower still. The "18-to-1" figure is only realized if the stock makes a large move that drags the option deep into the money and lifts its delta toward 1.0 — which is the same large-and-fast move every other part of this post says is unlikely. The headline leverage is the leverage you get *after* you have already won; the leverage you start with is materially smaller, and it is paid for with theta and vega. This is the first sleight of hand the long single plays on a beginner: it advertises the leverage of the winning state while charging you in the much-more-likely losing state.

### Why this is the trade everyone makes first

It is worth naming the psychology, because the long single's popularity is not an accident. It is the *only* options trade that maps cleanly onto the mental model a stock investor already has: "I think it goes up, so I buy something that goes up more." There is one decision (call or put), one number to look at (the premium), and a story that fits a tweet. Spreads require holding two legs in your head; selling premium requires being comfortable with a position whose best case is a small gain and whose worst case is alarming; volatility trades require thinking in a dimension — vol itself — that beginners don't yet have intuition for. The long single asks for none of that. It is the path of least cognitive resistance, and brokerages' mobile interfaces, with their one-tap call-buying and their gamified payoff previews, are designed to keep it that way.

The result is a structural mismatch: the easiest trade to understand is one of the hardest to win, and the people most drawn to it are precisely the ones least equipped to see the theta and vega draining it. Everything that follows is an attempt to give you the intuition the interface withholds — so that when you do buy a single option, it is a deliberate choice for a specific job, not a reflex.

### The three inputs that decide your fate

An option's value, from the [five-inputs post](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition), is a function of the spot price, the strike, the time to expiry, the implied volatility, and the interest rate. When you buy and hold a single option, three of those inputs move against you or for you over the life of the trade, and they are the three you must conquer:

- **The spot price (your delta exposure).** If the stock moves your way, your option gains; if it moves against you, it loses. This is the directional bet you *think* you are making.
- **Time to expiry (your theta exposure).** Every day that passes, the option loses a slice of its time value, even if nothing else changes. You are *short time* — the clock is your enemy. We covered this as the [melting ice cube](/blog/trading/options-volatility/time-value-and-theta-why-an-option-is-a-melting-ice-cube).
- **Implied volatility (your vega exposure).** A long option is *long vega* — its value rises when implied vol rises and falls when implied vol falls. Buy when implied vol is high and you are exposed to a crush; that is the [vega lesson](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) made painful.

The directional bet is the only one most beginners are thinking about. The other two are working against them silently, and together they are usually the deciding factor. To win, you need the gain from direction to exceed the combined drag of theta and (often) vega. That is a much harder thing than "the stock goes up."

> [!note]
> **A plain analogy for the drag.** Buying a long call is like renting a fast car to win a race that starts at an unknown future moment. The car (your delta) is genuinely fast, but you pay rent every single day it sits in the garage (theta), and the rental company can hike the price or cut it depending on how exciting they think the race will be (vega). If the race finally happens and you win by a hair, you might still lose money once the rent and the price changes are tallied. To come out ahead you don't just need to win the race — you need to win it *soon* and *decisively*, before the rent eats the prize. Conceptually that is the entire difficulty of the long single, and the next figure quantifies each piece of rent.

### Breakeven: the line you are actually betting to cross

The payoff diagram you already know defines a precise breakeven, and it is worth stating it as a formula because it is the magnitude hurdle in disguise. For a long call, you do not profit at expiry until the stock clears **strike + premium**. For a long put, you do not profit until the stock falls below **strike − premium**. Below (or above) that line you have not made money; you have only earned back what you paid.

For the at-the-money 100 call at a **\$5.47** premium, breakeven is **\$105.47** — the stock must rise **+5.5%** *by expiry* just for you to net zero. For a cheaper out-of-the-money 105 call at a **\$3.35** premium, breakeven is **\$108.35** — a **+8.3%** move required. The cheaper the option, the larger the move you must engineer to break even. This is the first place the trade is harder than it looks: a "bullish" view that turns out to be a modest **+4%** rally produces a *loss* on both of those calls, because neither cleared its breakeven. Being right on direction is necessary but nowhere near sufficient.

And notice that this breakeven is the *at-expiry* line — the easiest version of the hurdle. *Before* expiry the bar is effectively higher, because some of the premium is still time value that you would forfeit by closing early; to get out at break-even *during* the option's life, the stock has to be a little past the at-expiry breakeven to make up for the time value you are surrendering. The breakeven you read off a payoff diagram is therefore a floor on the difficulty, not the whole of it. Every layer we add — time value during life, theta, the vol crush — pushes the real-world breakeven further out than the clean strike-plus-premium formula suggests.

## The triple hurdle, quantified

Here is the core claim of this post, stated as a structure: a long single must clear **three hurdles in sequence**, and failing any one of them turns the trade into a loss. The next figure lays them out as a gauntlet, and then we put a number on each.

![The triple hurdle a long single must clear shown as a left-to-right gauntlet of direction magnitude and timing with fail states](/imgs/blogs/long-calls-and-puts-the-pure-directional-bet-and-why-it-usually-loses-4.png)

Read the gauntlet left to right. You buy the call (the blue start node). To reach the green PROFIT terminal on the right, the trade must pass through three amber gates, and at each gate there is a red trapdoor — a way the trade fails even though it cleared the previous gates.

**Hurdle 1 — Direction.** The stock must move the way you bet. This is the obvious one and the only one beginners plan for. If you buy a call and the stock falls, you lose; the further it falls, the more of the premium you lose, down to a total loss if it expires below the strike. There is no partial credit for being directionally close — a put that should have been a call is just a loss. Empirically, even pure direction is roughly a coin flip over short horizons; the market is close to a random walk on the timescales where most single options live. So before you have even reached the hard hurdles, you have spent your edge on something that is, charitably, 50/50.

**Hurdle 2 — Magnitude past breakeven.** The move must be *big enough* to clear breakeven. As we just computed, an out-of-the-money call can need a **+8%** move at expiry to net zero. The realized move that actually happens has a probability distribution, and most of its mass sits in *small* moves. Let us price exactly how often the move is big enough.

#### Worked example: how often does an OTM call even reach breakeven?

Take the out-of-the-money 105 call on our \$100 stock, three months to expiry, implied vol **25%**, rate **4%**. The Black-Scholes price is **\$3.35** per share, so breakeven at expiry is **\$105 + \$3.35 = \$108.35**, a required move of **+8.3%**.

Now ask the probability the stock actually finishes above \$108.35 in three months. Modeling the stock as geometric Brownian motion with drift equal to the risk-free rate (a neutral assumption — see the [risk-neutral pricing post](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews) for why), the chance of finishing above \$108.35 is only about **27%**. That number should stop you cold. Even if you have *zero* directional edge — even if the stock is a fair coin — the OTM call has roughly a **27%** chance of finishing in profit and a **73%** chance of expiring worthless or below breakeven. You are not buying a 50/50 bet; the magnitude hurdle has already turned it into a roughly 1-in-4 shot. To make the OTM call a positive-expectancy trade, your directional edge has to be large enough to push that 27% meaningfully higher, *and* the payoff in the winning tail has to compensate for the frequent total losses. That is a high bar, and it is why the cheap OTM call is correctly called a *lottery ticket*.

The intuition: the cheapness of an OTM option is not a discount, it is the market's honest assessment that the move it needs is unlikely.

**Hurdle 3 — Timing while IV holds.** The move must arrive *before theta drains the option*, and implied volatility must not collapse underneath you. This is the hurdle that catches the trader who is right on direction *and* magnitude but slow, or who bought into an event. We will spend the next two sections on it, because it is where the surprising losses come from.

The next figure shows hurdles 2 and 3 working together: the P&L of a long at-the-money call as a function of the move size, three weeks after entry, drawn twice — once with implied vol holding at 25%, once with it crushed to 15%.

![Profit and loss of a long call versus stock move size at fixed time with two curves for implied volatility held versus crushed](/imgs/blogs/long-calls-and-puts-the-pure-directional-bet-and-why-it-usually-loses-2.png)

The blue curve (IV holds) crosses zero at about a **+1.3%** move — three weeks after entry, with vol unchanged, you need the stock up a little over one percent just to overcome the theta you have already paid. But the red dashed curve (IV crushed to 15%) crosses zero at about **+3.7%**: the vol crush has shoved your breakeven move more than two full percentage points to the right. Same stock, same time, same directional view — and the vol crush alone has made the hurdle nearly three times higher. The flat amber line near the bottom is the floor you hit if the stock doesn't move at all: pure theta loss. This single chart is the most important one in the post, because it shows that *the hurdle is not fixed* — vega moves it, and vega usually moves it against you.

## Why long singles usually lose, in three mechanisms

We now have the structure. Let us make the three loss mechanisms concrete and watch each one cost real dollars.

### Mechanism 1: theta bleed — the flat-stock tax

Even if you are *exactly* right that the stock won't move much, a long option still loses, because time value decays to zero at expiry. This is theta, and for a long option theta is always working against you. The decay is not linear — it accelerates as expiry approaches, scaling roughly with the square root of remaining time, so the last few weeks bleed fastest. The next figure is the bleed clock: an at-the-money call held at a flat \$100, watched as the days tick by.

![The bleed clock showing the value of a held at-the-money call over time at flat spot decaying to zero fastest near expiry](/imgs/blogs/long-calls-and-puts-the-pure-directional-bet-and-why-it-usually-loses-3.png)

Notice the shape. The call enters at **\$4.48** with 63 days to run. Halfway through the calendar — at 31 days held — it is still worth **\$3.10**, having lost only **31%** of its value. But then the curve bends sharply downward, and the last week erases the remaining time value in a near-vertical drop. Time decay is back-loaded: holding a long option through its final weeks is the most expensive part of the trade. If your catalyst is more than a few weeks out, the option you bought today will have bled a large fraction of its value by the time the catalyst arrives.

#### Worked example: the theta tax on a flat stock

Buy the at-the-money 100 call with 63 days to expiry: implied vol 25%, rate 4%, the Black-Scholes price is **\$4.48** per share, or **\$448** per contract. The stock then does *nothing* — it sits at \$100 for three weeks. With 42 days left, the same call is now worth **\$3.61**. You have lost **\$0.87** per share, **\$87** per contract, to time alone, with the stock exactly where you predicted.

Hold it another three weeks to 21 days left, still flat, and it is worth **\$2.51** — you are now down **\$1.97** per share, **\$197** per contract, about **44%** of your premium gone, purely from the clock. The intuition: being right that "nothing will happen" is a *losing* trade for a long option, because the option is priced for something to happen and charges you rent until it does. If your view is "quiet," you should be *selling* the option, not buying it.

### Mechanism 2: the vol crush — losing while right

The most painful losses come from vega, specifically from buying a long option when its implied volatility is elevated and holding it as that vol collapses. The textbook case is buying an option just before a known event — an earnings report, an FDA decision, a central-bank meeting — when implied volatility is pumped up to price the coming uncertainty. The moment the event resolves, the uncertainty is gone, implied vol crushes back toward normal, and your long-vega position takes a hit that can swamp your directional gain. This is the [vol crush we will study in depth in the events post](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush); here we just need to feel it in dollars.

#### Worked example: nailing the direction and still losing (the hook, priced)

This is the trade from the opening. Two weeks before earnings, the stock is \$100 and the at-the-money 100 call has its implied vol pumped to **55%** because the market is pricing a big earnings move. At 14 days to expiry and 55% vol, the Black-Scholes price is **\$4.37** per share, **\$437** per contract. You buy it, betting the stock rallies on a good report.

The report comes. You were *right*: the stock gaps up **+3%** to \$103. But the uncertainty is now resolved, so implied vol crushes from **55%** down to **25%**, and a day has passed (13 days left). Reprice: at S = \$103, 13 days, 25% vol, the call is worth **\$3.88** per share, **\$388** per contract. You correctly predicted the direction *and* got a 3% move — and you lost **\$0.49** per share, **\$49** per contract, an **11%** loss on the position.

Where did the money go? The +3% move added value through delta, but the vol crush from 55% to 25% subtracted more, and theta took its slice. The intuition: when you buy a long option into an event, you are paying for the *expected* move; if the realized move is smaller than what the elevated implied vol was charging for, you lose even when you are directionally right. This is the single most common way beginners get hurt, and it is why "buy calls before earnings" is, as a blanket strategy, a wealth-transfer mechanism from buyers to sellers.

### Mechanism 3: the move arrives, but too small or too late

The third mechanism blends the first two with a partial directional win. You are right on direction, the move is real, but it is smaller than breakeven *or* it takes so long that theta and a modest vol drift erase the gain. This is the quiet, common loss — no dramatic event, just a slow grind that the option couldn't outrun.

#### Worked example: right direction, real move, still a loss

Buy the at-the-money 100 call at 63 days, 25% vol: price **\$4.48** per share, **\$448** per contract. (Note: this is the same option as the theta-tax example, priced off a 63-calendar-day clock, so the premium is \$4.48 rather than the \$5.47 of the three-month diagram — the contract is the same kind, just a shorter clock.) Now the stock *does* rise **+3%** to \$103 — you were right — but it takes **seven weeks** (49 days) of slow grinding to get there, and over that stretch implied vol drifts down from 25% to 20% as the market calms.

Reprice at S = \$103, 14 days left, 20% vol: the call is worth **\$3.64** per share, **\$364** per contract. You were right on direction, you got a real **+3%** move, and you *still lost* **\$0.84** per share, **\$84** per contract. Decompose it: the slow passage of time cost about **\$2.45** in theta, the +3% move added about **\$1.91** through delta, and the vol drift from 25% to 20% subtracted about **\$0.31** through vega. The directional gain was no match for the theta the slow grind let accumulate, and the vega drift deepened the hole.

The intuition: a long option is a race against the clock, and a *slow* win is often indistinguishable from a loss. If your thesis is "this drifts higher over a couple of months," the long single is the wrong instrument — the theta will eat the drift. This is precisely the regime where you should own the stock or use a spread that finances the theta.

## The base-rate reality

Step back from the individual trades and look at the aggregate. The structural reason long singles lose is the **variance risk premium**: across the market and over time, implied volatility prints *above* the realized volatility that follows. From the [variance-risk-premium post](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt), the long-run gap on the S&P 500 is roughly **+3.7 vol points** — average implied near **19.5%** against average subsequent realized near **15.8%**. Options, on average, are priced for *more* movement than actually shows up.

When you buy a single option, you are systematically paying that premium. You are buying volatility (and time) at a price that, on average, exceeds what the underlying delivers. That does not mean every long option loses — far from it, the convex winners can be enormous — but it means the *base rate* is against you, the way the base rate is against a casino patron. You can absolutely win, especially when you have a genuine view that the realized move will exceed the priced move, but you are fighting a structural headwind, and most buyers who treat the long single as a generic "leverage" tool simply feed the premium to the sellers. The honest framing: a long single is a bet that *realized* volatility (in the direction you chose) will beat *implied* — it is the [implied-versus-realized trade](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) taken from the long side, and you should only take it when that gap is actually in your favor.

### The shape of the distribution, not the average

The reason the long single fools so many people is that its P&L distribution is wildly asymmetric, and humans reason badly about asymmetric distributions. The *most likely* outcome of an OTM long single is a total loss — recall the 105 call's roughly **27%** chance of finishing above breakeven, meaning a **73%** chance of partial or total loss. The *average* outcome, dragged down by the variance risk premium, is a small negative. But the distribution has a long, fat right tail: the rare large-and-fast move that turns \$335 into \$1,800 or more. So the experience of buying long singles is a steady drip of small total losses, punctuated occasionally by a windfall that you remember vividly and that confirms your bias to keep buying.

This is the same cognitive trap that keeps lottery tickets selling: the salience of the jackpot overwhelms the arithmetic of the base rate. Two people running the identical long-single strategy can have opposite stories — the one who hit a winner early "knows it works," the one who hasn't "got unlucky" — even though both face the same negative expectation. The professional's discipline is to ignore the vividness of the right tail and ask the cold question every time: *is the implied vol I am paying lower than the realized move I genuinely expect, on this specific trade?* If the answer isn't a clear yes, the distribution is against you, and the occasional jackpot won't save you. The figure of the triple hurdle is really a map of where the left tail comes from; the right tail is what you are paying all those left-tail losses to reach.

## Common misconceptions

These are the beliefs that keep the wealth flowing from long-single buyers to sellers. Each is corrected with a number.

**Misconception 1: "Calls are cheap leverage."** They are leverage, but they are not cheap — the cheapness is an illusion created by ignoring theta and vega. The \$3.35 OTM 105 call looks like a bargain bullish bet on a \$100 stock. But it has a **27%** chance of finishing in profit at a fair-coin baseline, it bleeds about **\$0.03 per day** in theta near the money, and it needs a **+8.3%** move just to break even. The "cheapness" is the market correctly pricing how unlikely the needed move is. Cheap leverage that pays only 1 time in 4 is not cheap; it is a lottery ticket priced like a lottery ticket.

**Misconception 2: "If I'm right on direction, I make money."** The three worked examples above each got the direction right and *lost*. A +3% rally into earnings lost **\$49** to the vol crush; a +3% rally over seven weeks lost **\$65** to theta and vega drift. Direction is one of three hurdles, and it is the *easiest* one. The gain from being right on direction must exceed the combined drag of theta and vega, and over the short horizons where most options live, it frequently doesn't.

**Misconception 3: "Out-of-the-money options are the best bang for the buck."** The OTM call is cheapest *per contract*, which fools people into thinking it is the most efficient. But it has the lowest delta (**0.40** for our 105 call versus **0.56** for the ATM), so it captures the least of any given move, and it needs the largest percentage move to break even. The OTM call only "wins big" in the rare large-and-fast move; in the far more common small move it loses everything. Cheap per contract is not cheap per unit of expected payoff.

**Misconception 4: "Buying before earnings is a smart way to play the report."** It is one of the worst. The implied vol you pay before the event is, on average, *higher* than the move that follows — that is the whole event-vol-crush phenomenon. You are buying volatility at its most expensive, right before it gets cheaper by construction. The +3% earnings winner that still lost **\$49** is the rule, not the exception. If you want to trade an event with a long option, you must believe the realized move will *exceed* what the elevated implied vol is charging — which is a high and specific bar, not a default.

**Misconception 5: "A long option can't lose more than the premium, so it's low-risk."** The defined, capped loss is real and valuable — your loss is bounded at the premium, unlike the [short option's unlimited downside](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short). But "limited loss" is not "low risk" when the *most likely outcome is losing the entire premium*. A position with a 73% chance of a total loss is high-risk by any sensible definition; it is just high-risk with a known maximum. Defining your risk is about sizing and survival, not about the probability of loss.

## When a long single IS the right trade

We have spent the post explaining why long singles usually lose, but they are not always wrong — there is a narrow set of regimes where the long single is genuinely the best tool, and a serious trader uses it precisely in those regimes. The next figure is the decision map; then we walk each case.

![Decision figure showing when to buy a long single versus when to use a spread or stock with cheap vol catalyst and defined risk pointing to yes](/imgs/blogs/long-calls-and-puts-the-pure-directional-bet-and-why-it-usually-loses-7.png)

The left column (green) is the case *for* a long single; the right column (red) is the case for something else. Three conditions push toward YES, and they tend to come together.

**Cheap vol — implied below the realized you expect.** The variance risk premium is a base rate, not a law; there are regimes where implied vol sits *below* the realized move you have good reason to expect — after a long quiet stretch when the market has gone to sleep, or when you have specific information that a move is coming that the option market hasn't priced. When implied vol is genuinely cheap relative to the move you foresee, the IV-versus-RV gap is in your favor, and the long option is the cleanest way to take the bet. This is the [implied-versus-realized trade](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) from the long side, and it is the *only* condition that flips the structural headwind into a tailwind.

**An asymmetric, dated catalyst.** If there is a specific known date — a binary regulatory decision, a major data release, a court ruling, a product launch — that will force a large move *soon*, the long single's convexity is exactly the right shape. You want a position that loses a little if nothing happens and pays a lot if the move is violent, and you want it to resolve before theta grinds it down. The dated catalyst addresses hurdle 3 (timing) by guaranteeing the move arrives on a known schedule. The danger, as we saw, is that the option's implied vol already prices the catalyst — so this only works when you think the realized move will exceed the priced move, or you express it through structures that neutralize the vol crush (which the [straddle and event-vol posts](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet) cover).

When the big-and-fast move *does* arrive, the convexity is spectacular — and it is the entire reason the long single exists despite its grim base rate. The payoff is convex because the option's delta *grows* as the stock runs in your favor (positive gamma), so a large move pays at an accelerating rate. This is the fat right tail that funds all the small losses, and it is worth pricing so you can see what you are actually buying.

#### Worked example: the convex payoff when the move is big and fast

Buy the out-of-the-money 105 call on the \$100 stock, three months out, 25% vol: premium **\$3.35** per share, **\$335** per contract. Now suppose your catalyst hits and the stock *gaps* **+20%** to \$120 in a single week, with implied vol *rising* to 35% on the violent move (panic lifts vol). Reprice at S = \$120, 12 weeks minus one week of time left, 35% vol: the call is now worth about **\$18.03** per share, **\$1,803** per contract.

Your P&L is **+\$14.68** per share, **+\$1,468** per contract — a **+439%** return. Compare the stock holder, who made **+20%** on the same move. The option captured the move *and* the vol expansion *and* the convexity, turning a 20% underlying move into a fivefold return on premium. The intuition: this single outcome — large, fast, and vol-expanding — is the whole right tail the long single is built to catch, and it is why a string of total losses can still net out to a profit *if* you only buy singles when this kind of move is genuinely plausible. The discipline is buying the lottery ticket only when the jackpot is real, not as a default bullish reflex.

**Defined-risk convexity and tail bets.** Sometimes you *want* the long single precisely because its loss is capped and its payoff is convex. For a tail hedge — a far OTM put bought as insurance against a crash — you are not expecting to win; you are paying a small, defined premium for a large, convex payoff in the rare disaster, exactly the way you buy fire insurance. The expected value can be negative and the trade still be correct, because it pays off when you need it most. The capped loss means a tail bet can never blow up your account, which is the whole point.

**Replacing stock with a deep-ITM call.** This is the most underrated and most boring good use, and it is worth its own worked example. A deep-in-the-money call has a delta near 1.0, almost no time value, and almost no vega — it behaves like the stock itself, but it costs a fraction of the capital and caps your loss. Used this way, the long single is not a leveraged lottery ticket; it is a capital-efficient stock substitute.

![Deep-in-the-money call as a stock replacement showing profit and loss versus 100 shares tracking closely over a spot range](/imgs/blogs/long-calls-and-puts-the-pure-directional-bet-and-why-it-usually-loses-6.png)

The figure overlays the P&L of one deep-ITM 70 call against 100 shares of the \$100 stock, one month forward. The two lines are nearly on top of each other — the call tracks the stock almost dollar-for-dollar because its delta is about **0.99**. The thin amber gap is the small amount of time value you give up (the "rent"). But the call cost about **\$3,147** versus **\$10,000** for the shares — a third of the capital for nearly the same payoff, with your downside capped at the premium instead of running all the way to zero.

#### Worked example: deep-ITM call versus 100 shares

The deep-ITM 70 call on our \$100 stock, six months to expiry, 25% vol, costs **\$31.47** per share — **\$3,147** per contract — and has a delta of **0.99**. Compare two ways to be long 100 shares:

- **Buy 100 shares:** \$10,000 outlay. If the stock rises to \$110 in a month, P&L is **+\$1,000**.
- **Buy 1 deep-ITM 70 call:** \$3,147 outlay. If the stock rises to \$110 in a month, the call (now with five months left) is worth about **\$41.17**, so P&L is **+\$969** — within **\$31** of the stock's gain.

You captured **97%** of the upside for **31%** of the capital, and your worst case is the \$3,147 premium rather than the full \$10,000. The trade-off is the small time value you forfeit (the \$31 gap) and the fact that the option expires while the stock is forever. The intuition: deep in the money, an option is just leveraged stock with a parachute — and that is a perfectly sound, even conservative, use of a long single. The freed-up \$6,853 can sit in T-bills earning the risk-free rate, which often more than pays for the time value you gave up.

The right column of the decision map is the mirror image. If implied vol is *rich* (well above realized), you are overpaying and should sell premium or use a spread. If your view is a *slow grind* with no catalyst, theta will beat you — own the stock or use a spread. And if you "just want leverage," a [debit spread](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk) or a deep-ITM call gives you the gearing without the full theta-and-vega bleed of a naked long single.

## Strike and expiry selection: the delta–gamma–theta trade-off

Suppose you have decided a long single is right. Which strike, and which expiry? The choice is a trade-off among delta (how much of the move you capture), gamma (how fast your delta grows), theta (how fast you bleed), and cost. The next figure lays the three strikes side by side.

![Strike choice comparison showing out-of-the-money at-the-money and in-the-money call delta gamma theta as grouped bars and the cost of each](/imgs/blogs/long-calls-and-puts-the-pure-directional-bet-and-why-it-usually-loses-5.png)

The left panel shows the Greeks (scaled so they fit one axis); the right panel shows the cost. Read across the three strikes for our \$100 stock, three months out, 25% vol:

- **OTM (105 call), \$3.35.** Lowest delta (**0.40**) — captures the least of a given move. Decent gamma. Bleeds about **\$0.03 per day**. This is the lottery ticket: cheap, low probability, explosive only on a large fast move. Choose it only when you expect a *big* move and want maximum convexity per dollar.
- **ATM (100 call), \$5.47.** Highest gamma and the highest theta — the at-the-money option is the most sensitive to the next move and bleeds the fastest. Delta about **0.56**. This is the purest play on an *imminent* move; you pay the most theta for the most responsiveness. Choose it when you expect the move *soon*.
- **ITM (90 call), \$12.03.** Highest delta (**0.84**) — captures the most of a move and behaves most like the stock. Lowest gamma, lowest theta-per-dollar. This is the stock substitute; the deeper you go, the more it is just leveraged stock with a capped loss. Choose it when you want directional exposure with the least time decay.

The expiry choice runs along the same axis. **Short-dated** options are cheaper in dollars but bleed theta fastest and are most exposed to a vol crush — they are for fast catalysts only. **Long-dated** options (LEAPS, a year or more out) bleed theta slowly and give your thesis time to play out, which is why the deep-ITM stock-replacement trade is usually done in long-dated strikes. The general rule: **match the expiry to your catalyst with room to spare.** If you think the move comes in a month, buy three months of time, not one — you do not want to be right on the move but to have it arrive the week after your option expired. Buying *exactly* enough time is a classic way to be right and still lose.

There is a subtle interaction between strike and expiry that the grouped bars don't show but that decides a lot of real outcomes: **gamma and theta trade off against each other, and the at-the-money near-dated option maximizes both at once.** That is why a short-dated ATM call feels so exciting and is so dangerous — its gamma means a fast move pays explosively (your delta ramps from 0.5 toward 1.0 in a hurry), but its theta means that if the move *doesn't* come immediately, the bleed is brutal. You are buying the sharpest possible response to a move at the cost of the fastest possible decay. The OTM near-dated call is the same bargain pushed further: even more leverage to a big move, even less probability of getting it. The deep-ITM long-dated call is the opposite corner — low gamma, low theta, high delta, behaving like the stock. Knowing which corner of the strike-by-expiry grid you are standing in tells you what you are really betting on: the near-ATM corner is a bet on *timing and magnitude together*, the deep-ITM long-dated corner is a bet on *direction over a long horizon*, and they could not be more different despite both being "a long call."

A practical heuristic many traders use: for a directional view with a soft (undated) catalyst, buy a slightly in-the-money call — say a 0.60-to-0.70 delta — with two to three times as much time as your expected holding period. The ITM strike keeps delta high and theta moderate so you are not paying for a lottery ticket; the generous expiry keeps the back-loaded decay far from your holding window. You give up some leverage relative to the OTM strike, but you buy yourself the two things the long single is worst at: a forgiving theta profile and resistance to a vol drift. It is the unglamorous middle of the strike map, and it is where the long single is least likely to embarrass you.

## How it shows up in real markets

**Earnings season, every quarter.** The pattern from the hook repeats thousands of times every earnings season. Implied vol on a single name ramps into the report — often to 50%, 70%, sometimes over 100% for a volatile small-cap — pricing the expected move. Buyers pile into calls or puts the afternoon before. The report lands, the stock moves, and unless the move *exceeds* what the implied vol was pricing (the "expected move," which you can read off the [event-pricing post](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options)), the vol crush eats the buyers. The desks selling that premium know the variance risk premium is on their side; the buyers usually don't. This is the most reliable wealth transfer in the options market.

**The 0DTE explosion.** The growth of zero-days-to-expiry options on the S&P 500 has turned the theta-bleed problem into a same-day phenomenon. A 0DTE option has hours of life, enormous gamma, and theta that drains the entire premium by the close. Buyers can win spectacularly on a fast intraday move, but the base rate is brutal: most 0DTE long singles expire worthless within hours, and the aggregate flow has made dealers' short-gamma hedging a feature of the modern intraday tape. The 0DTE long single is the triple hurdle compressed into a single trading session.

**Meme-stock call buying, 2021.** During the 2021 meme-stock episode, retail traders bought enormous volumes of short-dated OTM calls. When the underlying stocks rocketed, those calls paid off astronomically — the rare large-and-fast move that the OTM lottery ticket is built for — and the buying itself forced dealers to hedge by buying stock, amplifying the rallies (a gamma-squeeze dynamic covered in the [gamma post](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short)). But for every viral screenshot of a life-changing gain, vast numbers of those same calls expired worthless when the move didn't come, didn't come fast enough, or reversed. The episode is the long single's whole distribution made visible: a fat left tail of total losses funding a thin right tail of enormous wins.

**Biotech binary events.** A small-cap drug developer awaiting an FDA decision is the purest live demonstration of the dated-catalyst case — and its dangers. Implied vol on the options can run to 150% or more ahead of the verdict, because the stock will plausibly double on approval or halve on rejection. A long call here is a clean convex bet, but the vol crush after the binary resolves is enormous: even on approval, if the stock rises *less* than the implied vol was pricing, the call can underperform or lose. Biotech is where the lesson "you are buying the expected move, and you must beat it" is least forgiving, because the expected move embedded in the premium is so large. Traders who understand this often prefer to *sell* the inflated pre-event vol, or to structure the bet so the vol crush works for them rather than against them.

**The professional's deep-ITM stock replacement.** Quietly, in the background, sophisticated traders and funds use deep-ITM calls as capital-efficient stock proxies all the time — to free up capital, to define risk on a concentrated position, to gain exposure without tying up cash. It rarely makes the news because it is boring and it works. It is the same instrument as the meme-stock lottery ticket, used at the opposite end of the moneyness spectrum, for the opposite reason. The difference between the trader who blows up on long singles and the one who uses them well is almost never the instrument — it is *which corner of the moneyness-and-vol map they trade it in*, and whether they sized it to survive the base rate.

## The playbook

Here is how to trade a long single without becoming the screenshot.

**The position.** One call (bullish) or one put (bearish), held alone. You are **long delta** (directional), **short theta** (the clock is your enemy), and **long vega** (exposed to a vol crush). Internalize that you are short time and long volatility, not merely "long the stock's direction."

**Entry — only when the regime fits.** Buy a long single only when at least one of these holds: (1) implied vol is *cheap* relative to the realized move you expect — the IV/RV gap is in your favor; (2) there is a *dated catalyst* that forces a fast move and you believe the realized move will beat the priced move; (3) you want *defined-risk convexity* for a tail bet; or (4) you are replacing stock with a *deep-ITM call* for capital efficiency. If none of these holds — if you "just want leverage" on a slow-grind view — use a spread or own the stock instead. Check the implied vol against recent realized *before* you click; never buy elevated event vol unless you specifically expect to beat it.

**Strike and expiry.** Match the instrument to the goal. Big-move convexity → OTM, but size it as a lottery ticket. Imminent move → ATM, accept the theta. Stock substitute → deep-ITM, long-dated. Always buy *more* time than you think you need: match the expiry to the catalyst with weeks to spare, so a correct-but-slow thesis still has room.

**Sizing — the defined loss is the feature, use it.** Because the most likely outcome is a total loss of premium, size each long single as a small fraction of capital — **1–2%** at risk per trade is a sane ceiling for speculative singles, smaller for OTM lottery tickets. The capped loss lets you survive a string of zeros; the [position-sizing post](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) makes the risk-of-ruin math explicit. Never size a long single as if it were a sure thing — it is a bet with a fat left tail.

**Management — avoid the "ride it to zero" trap.** This is where most of the salvageable money is lost. Three rules:

- **Take profit on a fast win.** If the move comes quickly and the option doubles or triples, *take it* — you are racing theta, and the gain you have is real while the gain you are hoping for is exposed to reversal and decay. Pigs get slaughtered by theta.
- **Cut or roll before the bleed accelerates.** From the bleed clock, the last few weeks decay fastest. If your thesis hasn't played out with three or four weeks left, either cut the position (take the partial loss) or *roll* it out to a later expiry — sell the bleeding near-dated option and buy a longer-dated one — rather than holding into the terminal decay. Concretely: if your 14-days-left call has fallen from \$4.48 to \$2.03 with the stock flat, you can sell it for \$2.03 and put the proceeds toward a fresh 63-day call, resetting the clock for the cost of the realized decay rather than donating the remaining time value to a near-certain expiry at zero. The roll is not free — you are crystallizing the loss so far and paying a new spread — but it converts a forced total loss into a financed second chance for a view you still hold. Do not let a directional view you still believe in expire worthless because you held the wrong (too-short) option.
- **Define the exit before you enter.** Decide up front: at what move do I take profit, at what level do I cut, by what date do I roll or abandon. The long single's failure mode is the slow bleed where the trader keeps waiting "one more day" while theta empties the position. A pre-committed exit is the only reliable defense.

**The invalidation.** Your view is wrong, and you should be out, when: the stock moves *against* you past a level you set; the catalyst passes without the move you needed (cut immediately — the vol crush will do the rest); or time runs short and the thesis hasn't played out (roll or abandon, don't ride to zero). The cleanest invalidation is *time itself* — a long single that hasn't worked by its halfway mark is usually a loser already, because the back-loaded theta is about to accelerate.

The deepest lesson of the long single is the one the screenshot trader learned the hard way: **your enemy is time, not direction.** Get that backwards and you will keep being right about the stock and wrong about your account. Get it straight and the long single becomes what it should be — a precise tool for a narrow job, used when cheap vol, a real catalyst, or pure capital efficiency makes it the right shape, and left alone the rest of the time.

## Further reading & cross-links

Within this series:

- [What Is an Option: The Right, Not the Obligation](/blog/trading/options-volatility/what-is-an-option-the-right-not-the-obligation) — the contract this whole post is built on.
- [Calls, Puts, and the Payoff Diagram](/blog/trading/options-volatility/calls-puts-and-the-payoff-diagram-the-language-of-options) — the hockey-stick shapes and breakevens we used throughout.
- [Moneyness and the Strike: ITM, ATM, OTM](/blog/trading/options-volatility/moneyness-and-the-strike-itm-atm-otm-and-what-you-are-really-buying) — the strike-selection trade-off in depth, including the deep-ITM stock proxy.
- [Time Value and Theta: Why an Option Is a Melting Ice Cube](/blog/trading/options-volatility/time-value-and-theta-why-an-option-is-a-melting-ice-cube) — the bleed mechanism behind mechanism 1.
- [Vega: Your Exposure to Implied Volatility](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — why the vol crush hits a long single.
- [Implied vs Realized Volatility: The Trade at the Heart of Options](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) — the cheap-vol condition that flips the long single's odds.
- [The Volatility Smile and Skew: Why OTM Puts Cost More](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) — why the put you buy as a hedge is priced rich.
- [Gamma: The Greek That Bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) — the convexity that makes the rare big-fast move pay, and the short side's danger.
- [The Variance Risk Premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) — the structural reason buyers are fighting a headwind.

Where this leads next (Track D and beyond):

- [Vertical Spreads: Debit and Credit, Defining Your Risk](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk) — the structure that finances the long single's theta and is usually the better leverage trade.
- [Straddles, Strangles, and the Long Volatility Bet](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet) — how to bet on a move without picking a direction, and how to handle the vol crush.
- [Trading Event Vol: Earnings, FOMC, and the Vol Crush](/blog/trading/options-volatility/trading-event-vol-earnings-fomc-and-the-vol-crush) — the event-specific deep dive on what killed the hook trade.
- [Position Sizing and Risk of Ruin in Options Trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — how to size lottery tickets so a string of zeros doesn't end you.

For the pricing theory we deliberately did not re-derive:

- [Black-Scholes: The Pricing Model](/blog/trading/quantitative-finance/black-scholes) — where every premium and Greek in this post comes from.
- [Event Volatility: Implied vs Realized and the Vol Crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) — the cross-asset view of the mechanism that lost the screenshot trader his money while he was right.
