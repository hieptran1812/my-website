---
title: "0DTE and the Rise of Short-Dated Options: The New Market Structure"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "What zero-days-to-expiry options really are, why their Greeks are the most extreme on the board, how their flow reshapes intraday volatility, and how to trade them without blowing up."
tags: ["options", "volatility", "0dte", "gamma", "theta", "dealer-gamma", "market-structure", "pin-risk", "spx", "short-dated-options"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A 0DTE (zero-days-to-expiry) option is an option that expires the same day you trade it. Because it has almost no time left, it carries enormous gamma and theta, almost no vega, and a delta that flips from 0 to 1 across a razor band of price. That makes it the most explosive single instrument on the board, and the flood of 0DTE flow has quietly rewired the intraday volatility of the entire index complex.
>
> - **The Greeks go to extremes.** Our model says an at-the-money 0DTE option with two hours left has gamma around **0.93** versus **0.069** for a 30-day option — roughly **13x** the curvature — while its vega collapses to about a fifth. All direction and time, almost no vol sensitivity.
> - **It is a binary, not a position.** Theta eats an at-the-money 0DTE option from about \$0.22 to near zero over one session, and the decay accelerates into the close. You are right or you are worthless, usually within hours.
> - **The flow moves the market.** 0DTE now makes up a large share of total SPX option volume (north of half on many days). That flow forces dealers to hedge at very high gamma, which can pin the index to a strike or amplify a move once the strike breaks.
> - **The one rule to remember:** trade 0DTE only with defined risk, size it for a full loss, and close before the final-hour gamma cliff — because the theta is brutal and the gamma is binary into the bell.

## A trader up huge at noon, wiped by the close

It is a little after noon on a Wednesday and a retail trader is staring at a number that does not feel real. At 9:45 in the morning he bought ten at-the-money 0DTE calls on a major equity index for about \$0.22 each — \$220 of premium against an index sitting almost exactly on a round strike. By 11:30 the index had drifted up about half a percent, his calls had gone from a few cents of time value to deep in the money, and the screen said his \$220 was now worth a touch under \$2,000. A 9x in two hours. He screenshots it. He does not sell.

He does not sell because the whole appeal of the trade was the asymmetry, and the asymmetry feels like it is just getting started. He has seen the threads: someone turned \$500 into \$40,000 on a Fed day, someone else caught the entire afternoon ramp. The position is "free-rolling" now, he tells himself — he can only lose what he put in, and look how fast it moved. So he holds.

At 1:30 the index gives back the morning's gain. By 2:30 it is slightly red on the day. His calls, which an hour ago were intrinsic-value machines, are now barely out of the money with ninety minutes left — and ninety minutes, for a 0DTE option, is almost no time at all. The bid is \$0.40 and falling. He tells himself it just needs one more push. It does not come. The index drifts sideways into the close, the calls expire a few cents out of the money, and the broker settles them at zero. The \$1,950 of paper profit, and the \$220 he started with, are both gone by 4:00. He was *right about the direction* for most of the day and still lost everything, because with a 0DTE option being right is not enough — you have to be right *at the bell*.

That round trip — euphoria at noon, zero by the close — is not a story about a bad trader. It is a story about what a 0DTE option *is*. Strip out everything else and a same-day option is a leveraged switch on where the index prints in the final minutes, and the switch flips violently right where the trader is standing. This post is about that instrument: what it is, why its Greeks are off the chart, why people trade it anyway, how its sheer volume has reshaped the market's intraday behavior, and how to handle it without becoming the noon screenshot.

![Gamma versus stock price for a 0DTE option and a 30-day option, with the 0DTE curve a tall narrow spike at the strike](/imgs/blogs/0dte-and-the-rise-of-short-dated-options-the-new-market-structure-1.png)

Look at the two curves above. Each is gamma — the curvature of an option, the rate at which its delta changes — plotted against the index price for a strike of \$100. The blue curve is a 30-day option: a low, wide, gentle bell. The red curve is a 0DTE option with about two hours left, and it is a *tower* — many times taller than the blue bell and so narrow it has collapsed to almost nothing a single dollar away from the strike. That red spike is the entire personality of a 0DTE option. Everything below unpacks why it looks like that and what that shape does to your money and to the market.

## Foundations: what a zero-days-to-expiry option actually is

Before the Greeks, the plumbing. If any of the underlying terms feel shaky, this series builds them from scratch elsewhere — [What Is an Option: The Right, Not the Obligation](/blog/trading/options-volatility/what-is-an-option-the-right-not-the-obligation) defines the contract, and [The Black-Scholes pricing derivation](/blog/trading/quantitative-finance/black-scholes) supplies the model that every number in this post is computed from. Here is the part we need.

An **option** is a contract giving its owner the right, but not the obligation, to buy (a **call**) or sell (a **put**) an underlying — here, a stock index — at a fixed **strike** price, on or before an **expiration** date. The buyer pays a **premium** for that right; the seller collects the premium and takes on the obligation. An option's value has two pieces: **intrinsic value** (how far in the money it already is) and **time value** (the extra the market pays for the chance it moves further into the money before expiry). Time value is the whole game in most option trading, because intrinsic value is just arithmetic.

A **0DTE option** — zero days to expiry — is simply an option on the *day it expires*. There is nothing exotic about the contract itself; it is an ordinary listed option that happens to have hours, not weeks, of life left. What makes it special is that one input to its price, time-to-expiry `T`, has shrunk almost to zero, and as you will see, sending `T` toward zero does extreme things to every Greek at once.

### Where they came from: from quarterly to daily expirations

For most of listed-options history, index options expired monthly, then weekly. The structural shift happened gradually and then suddenly. The CBOE listed SPX options with **Monday, Wednesday, and Friday** expirations, then **Tuesday and Thursday**, until by 2022 the S&P 500 index (SPX) had a listed option expiring *every single trading day*. The same daily-expiry buildout happened on SPY (the S&P 500 ETF) and QQQ (the Nasdaq-100 ETF). For the first time, a trader could wake up any morning and buy or sell an option that would live and die before dinner.

The volume response was staggering. By 2023, options expiring the same day they trade grew from a niche to a *majority* of SPX option volume on many days — frequently cited around or above **50%** of total SPX options activity, up from a small minority just a couple of years earlier. This is the structural fact that earns 0DTE its own post: an instrument that barely existed in 2021 had, by 2023–2024, become the single largest slice of the most important equity-index options market in the world. When that much flow concentrates in the most extreme-Greek instrument on the board, it does not just affect the traders using it — it changes the market everyone else trades in.

### Cash settlement and "no overnight"

One mechanical detail matters enormously and is easy to miss. SPX options are **cash-settled** and **European-style**: there is no early exercise, no shares change hands, and at expiry the difference between the settlement price and the strike is simply paid in cash. SPY and QQQ options are American-style and physically settled (you can be assigned shares), which adds [assignment and pin risk](/blog/trading/options-volatility/assignment-pin-risk-and-expiration-day-mechanics) we will return to. But the deeper point applies to both: a 0DTE position has **no overnight**. It is born and it dies inside one session. There is no "I'll hold it and see how it looks tomorrow." Tomorrow does not exist for this contract. The clock that for a normal option ticks down over weeks here runs out in a single afternoon, and the entire risk profile is compressed into those hours.

### Why short-dated options exploded when they did

It is worth pausing on *why* this happened around 2022, because the explanation is itself part of the market-structure story. Three things had to line up. First, the **listings**: the exchange had to actually create a contract expiring every day, which it did by filling in the Tuesday and Thursday expirations to complete the daily roster. Supply came first. Second, **access**: zero-commission retail brokers and slick mobile apps made it trivial and free to trade a single same-day contract, and the gamification of those apps pushed traders toward exactly the kind of cheap, high-variance bet a 0DTE option represents. Third, **a market regime** that rewarded the behavior: the post-2022 environment of strong rallies punctuated by sharp intraday reversals made same-day directional bets feel viable, and the long stretches of low overnight volatility made selling same-day premium feel like free money. Supply, access, and a regime that flattered both sides of the trade arrived together, and the volume went vertical.

There is also a more technical driver behind the *seller* side. Large systematic players — including certain defined-outcome and overwriting funds — sell short-dated index options as a structural strategy, and their flow is mechanical and enormous. When that supply meets the retail demand for cheap lottery tickets, you get a two-sided market with a dealer in the middle, and the dealer's hedging of that flow is the bridge from "lots of people trading 0DTE" to "the index itself moves differently." Hold that thought; it is the heart of the market-structure section later.

### The intraday clock: how a single session decays

For a normal option, the relevant clock is the calendar: theta is quoted per day, and you mentally amortize the premium over the weeks to expiry. For a 0DTE option the clock is the **trading session** — roughly 9:30 to 4:00 Eastern, about 6.5 hours, with the SPX a.m.- or p.m.-settled depending on the contract. Time value does not decay on a wall-clock schedule; it decays on a *volatility* schedule, falling roughly with the square root of the time remaining. That square-root shape is why the morning feels slow and the close feels like a freefall: going from 6 hours left to 5 barely changes the square root, but going from 1 hour left to zero changes it enormously. Every 0DTE trader, long or short, is really trading their view against that square-root clock, and underestimating how fast it runs in the final hour is the single most common way longs get surprised and shorts get complacent.

> **A 0DTE option is an ordinary option with one input — time — collapsed toward zero. Everything strange about it follows from that single fact, because the Greeks are not linear in time; they explode as `T` shrinks.**

## The extreme Greeks: enormous gamma, enormous theta, almost no vega

The Greeks measure how an option's price responds to changes in the world: **delta** to the underlying's price, **gamma** to the *change* in delta, **theta** to the passage of time, and **vega** to changes in implied volatility. For a normal 30-day option these all sit in comfortable ranges. For a 0DTE option they all go to extremes at the same time, and the combination is what makes the instrument behave the way it does. Let us take them one at a time, with numbers from this series' Black-Scholes pricer, all using a \$100 index, a \$100 strike, 20% annualized volatility, and a 4% rate.

### Gamma: the towering spike

Gamma is the curvature of the option — how fast delta changes per \$1 move in the index. It is covered in depth in [Gamma: The Greek That Bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short); the one-line recap is that gamma peaks at the money and grows as expiry approaches, because the option's payoff is bending from a flat line into the sharp hockey-stick kink right at the strike.

For a 0DTE option that bend happens in a vanishingly small region of price. Far from the strike a same-day option is already certain (worthless if out of the money, pure intrinsic if in), so its delta is flat at 0 or 1 and its gamma is zero. But right at the strike, the option has to transition from "worth nothing" to "worth a dollar per dollar in the money" in the few cents of price that remain meaningful in the final hours. That transition is the spike you saw in the cover chart.

#### Worked example: 0DTE gamma versus 30-day gamma

Take our at-the-money setup and compute gamma at the strike for three lives:

- **30 days to expiry:** `gamma(100, 100, 30/365, 0.04, 0.20)` = **0.0693**. A \$1 move changes delta by about 0.069.
- **1 day to expiry:** `gamma(100, 100, 1/365, ...)` = **0.3810**. About **5.5x** the 30-day curvature.
- **2 hours to expiry** (`T = 2/24/365`): `gamma(100, 100, 0.000228, ...)` = **1.3201**. About **19x** the 30-day curvature, and roughly **3.5x** the one-day number.
- **30 minutes to expiry:** gamma = **2.6403** — nearly **38x** the 30-day option.

Read those numbers as physical statements about delta. At 30 days, a \$1 move adds 7 share-equivalents of exposure to a 100-share contract. With 30 minutes left, the *same* \$1 move at the strike adds **264** share-equivalents — your position morphs from flat to enormously long (or short) in the space of a single dollar. The intuition: a 0DTE option does not gradually become a stock position the way a long-dated option does; it *snaps* into one the instant the index crosses the strike, and gamma is the measure of how violent that snap is.

### Theta: the brutal, accelerating clock

Theta is the rate at which an option loses value as time passes, holding everything else constant. A long option is, in the series' running model, a [melting ice cube](/blog/trading/options-volatility/time-value-and-theta-why-an-option-is-a-melting-ice-cube): its time value drips away every day. For a 0DTE option the ice cube is not in a freezer drawer; it is on a hot stove. All of the option's value is time value (at the money it has zero intrinsic), and all of that time value has to reach zero by 4:00. The decay does not run at a steady drip — it *accelerates*, because time value falls roughly with the square root of time remaining, and the square root function falls off a cliff as its argument approaches zero.

![Value of an at-the-money 0DTE option through one trading session, decaying to zero with the decay accelerating into the close](/imgs/blogs/0dte-and-the-rise-of-short-dated-options-the-new-market-structure-2.png)

The chart above prices an at-the-money 0DTE call through a single 6.5-hour session, holding the index pinned at the strike so you see *only* the time decay. At the open, with the full session ahead, the call is worth about \$0.22. By midday it has bled to roughly \$0.13. In the final hour it goes nearly vertical toward zero. That curve — gentle slope at the open, a cliff into the close — is the brutal clock. A long 0DTE holder is fighting that curve every minute the index sits still.

#### Worked example: the intraday melt of an ATM 0DTE call

Using the pricer with the index pinned at \$100, here is the at-the-money call's value at points through the day (time left expressed as a fraction of a year, `T = hours/24/365`):

- **Open, 6.5 hours left:** price = **\$0.2188**.
- **Midday, 3.5 hours left:** price = **\$0.1603**. The morning cost you about **\$0.0585** per share — over a quarter of the premium — for *nothing happening*.
- **Power hour, 1.5 hours left:** price = **\$0.1048**.
- **Final 30 minutes:** price = **\$0.0604**.
- **Last 5 minutes:** price ≈ **\$0.0246**.

Notice the acceleration. The first three hours cost about \$0.0585; the *last* 1.5 hours, a much shorter stretch, cost roughly \$0.0444 — a far higher rate of bleed per minute. The annualized theta confirms it: our pricer reports theta around **−78/year** at one day to expiry but around **−189/year** with four hours left and **−530/year** with 30 minutes left. The intuition: for a long 0DTE holder, every minute the index fails to move is not neutral — it is an actively losing minute, and the losing speeds up as the bell approaches.

### Vega: the one that nearly vanishes

Vega is sensitivity to implied volatility — how much the option's price moves when the market's expectation of future movement changes. It is the central Greek of the whole series, because [implied versus realized volatility](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) is the trade at the heart of options. But vega scales with the *square root of time to expiry*, so as `T` collapses, vega collapses with it.

#### Worked example: vega for a 0DTE versus a 30-day option

At the money, our pricer gives vega (per 1.00 = 100 vol-points of sigma; divide by 100 for a one-vol-point move):

- **30 days:** vega = **11.40**, so a one-vol-point rise in implied vol (say 20% to 21%) adds about **\$0.114** to the option.
- **2 hours (0DTE):** vega = **0.85**. The same one-vol-point change adds only about **\$0.0085** — barely a rounding error.

A 0DTE option has roughly **a thirteenth** the vega of a 30-day option. This is the quietly profound fact about 0DTE: it is almost *immune to implied volatility*. There is no time left for the market's vol forecast to play out, so the vol forecast barely enters the price. A 0DTE option does not really care what the VIX does in the abstract; it cares only about where the index actually prints in the next few hours. The intuition: a normal option is a bet on *volatility and time*; a 0DTE option has had the volatility-and-time engine throttled to almost nothing, leaving a near-pure bet on *direction by the close*.

### Putting it together: the most explosive instrument on the board

Now combine the three. Enormous gamma means the option's delta is hypersensitive to where the index sits. Enormous theta means it is bleeding value violently if it sits still. Tiny vega means none of the usual "the vol came in / the vol popped" cushioning applies. What remains, when you strip away vega and amplify gamma and theta, is a near-**binary** instrument: it resolves to roughly its intrinsic value at the close, which for an at-the-money strike means it goes to a meaningful number if the index finishes on the right side, or to zero if it finishes on the wrong side, with very little in between.

![Profile figure showing 0DTE option has enormous gamma, enormous theta, tiny vega, resolving to a binary win-or-worthless outcome at the close](/imgs/blogs/0dte-and-the-rise-of-short-dated-options-the-new-market-structure-4.png)

The figure traces it: the 0DTE at-the-money option feeds into three boxes — huge gamma, huge theta, tiny vega — and what comes out the bottom is a binary. Right by the close, the option finishes in the money and the move is yours; wrong by the close, it expires worthless and 100% of the premium is gone. There is almost no middle ground, and the whole thing resolves in hours. This is why a 0DTE at-the-money option is, contract for contract, the most explosive thing you can hold: it is leverage with the fuse already lit.

### The delta flip: a switch, not a dial

The gamma spike has a vivid consequence for delta. For a normal option, delta is a smooth dial: it slides gradually from 0 (far out of the money) through 0.5 (at the money) to 1 (deep in the money) as the index rises. For a 0DTE option in the final hour, delta is not a dial — it is a *switch*.

![Call delta versus index price near the strike for a 0DTE option with 30 minutes left and a 30-day option, the 0DTE delta snapping from near 0 to near 1 across a tiny band](/imgs/blogs/0dte-and-the-rise-of-short-dated-options-the-new-market-structure-3.png)

The chart shows call delta against the index for two lives. The blue 30-day curve is the familiar gentle S — at \$99 it is delta 0.17, at \$101 it is 0.83, a smooth ramp. The red 0DTE curve, with 30 minutes left, is a near-vertical step right at the strike.

#### Worked example: the razor band of the delta flip

With 30 minutes left, our pricer gives the call's delta across a tiny window of index price:

- **Index at \$99.75:** delta = **0.049**. Essentially out of the money; behaves like 5 shares.
- **Index at \$100.00:** delta = **0.501**. Exactly at the money; behaves like 50 shares.
- **Index at \$100.25:** delta = **0.951**. Effectively in the money; behaves like 95 shares.

Across a **\$0.50** band of index price, the option's exposure swings from 5 share-equivalents to 95 — almost the entire range from worthless to fully in the money. For comparison, getting that same delta swing on the 30-day option takes a move of several dollars. The intuition: in the last half hour a 0DTE option is a binary switch that the index is standing right on top of, and a fifty-cent wiggle in the index is the difference between a winning lottery ticket and a worthless one.

## Why traders use them anyway

If a 0DTE option is a melting, binary, hyper-levered switch, why is it the most-traded contract on the index? Because each of its extreme properties is exactly what *some* trader wants. There are three broad use cases, and the same instrument serves all three.

### The cheap lottery ticket

The most visible use is the directional punt. A 0DTE at-the-money option costs almost nothing — our open price was \$0.22 per share, \$22 for a 100-multiplier contract on a \$100 index, and on the real SPX an at-the-money 0DTE might run a couple of dollars of index points, a few hundred dollars of premium against a six-figure notional. That cheapness buys enormous leverage: because the option starts near the bottom of its gamma spike, a favorable move multiplies the premium fast, exactly as the noon screenshot showed. For a trader who wants a small, capped-downside bet on "the index rips this afternoon," a long 0DTE call is the purest expression available. The appeal is real. So is the trap: you are buying the instrument with the worst possible theta, and you have until the close — not until you are right, until the *bell* — to be right.

It is worth putting a number on the leverage that draws people in. With the index at \$100 and a \$0.22 at-the-money call, a move to \$101 — a 1% rally — leaves the call worth roughly its \$1.00 of intrinsic plus a sliver of time value, call it \$1.05. That is nearly a **5x** on the premium from a 1% index move, and earlier in the day, with more time value to inflate, the multiple is even larger. No other liquid instrument turns a 1% index move into a 400%-plus gain on capital. But run the same arithmetic the other way: a move to \$99 — a 1% *drop* — leaves the call essentially worthless, a **100% loss**. The leverage is symmetric in the underlying and brutally asymmetric in outcome: you risk all of the premium to make a multiple of it, and because the at-the-money option is a near coin flip on direction *plus* it has to overcome theta, the long buyer's edge is structurally negative. The lottery framing is exact — positive skew, negative expectation — and the same reason most lottery tickets lose is the reason most long 0DTE tickets expire worthless.

### The precise intraday directional tool

The second use is more professional. A trader with a genuine intraday view — the index will hold this support level, the morning gap will fill, the Fed statement will be read as dovish — wants exposure that is responsive *now* and not diluted by weeks of time value or vol noise. A 0DTE option is ideal: its high gamma means it tracks the index move almost dollar-for-dollar once it is in the money, and its tiny vega means a post-event vol crush will not gut the position the way it would gut a longer-dated option after an event (the classic [event vol-crush](/blog/trading/options-volatility/trading-event-vol-earnings-fomc-and-the-vol-crush) that punishes earnings-week call buyers). Used this way, a 0DTE option is a scalpel for a same-day thesis — held for minutes to hours, not into the close, precisely to avoid the binary resolution.

The contrast with a longer-dated bet is instructive. Suppose you want to express "the index goes up 1% by this afternoon." A 30-day call would capture some of that, but a big chunk of your premium is paying for the next 30 days of optionality you do not want, and if the move comes with a vol crush — common after a scheduled event resolves — your 30-day call's \$0.114-per-vol-point vega works against you, giving back gains as implied vol falls. The 0DTE call has almost none of that drag: you pay for today and only today, and the vol crush barely touches you because your vega is a thirteenth the size. For a pure same-day directional view, the 0DTE option is the *cleanest* instrument, not the most reckless one — provided you treat it as a short-fuse scalp and get out before the close turns your scalp into a coin flip. The recklessness is not in the tool; it is in holding it past the point where it stops being a directional bet and becomes a lottery on the final tick.

### Selling premium: the income trade

The third use flips the whole thing around: instead of buying the melting ice cube, *sell* it. The brutal theta that punishes the long is, by definition, a credit to the short. A premium seller writes 0DTE options — usually as defined-risk spreads or [iron condors](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range), occasionally, dangerously, naked — collects the premium at the open, and lets it decay to zero by the close. On a quiet day the index stays inside the sold range, every option the seller wrote expires worthless, and the seller keeps the whole credit. Do that day after day and it feels like a money machine: small, frequent, high-hit-rate wins. This is the [variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) harvested at the highest possible frequency.

And here is the trap of the income trade, which is the mirror of the lottery's trap. The seller is *short* the gamma spike. As long as the index sits still, the short collects theta and looks brilliant. But the moment the index trends through the sold strike, that towering gamma works against the seller with the same violence it works for the buyer. The position that was decaying pleasantly all morning can lurch into a near-maximum loss in the final hour. The income is real most days; the gamma is binary into the close on the day it matters.

#### Worked example: the same-day straddle, both sides

Make the two sides concrete with one trade. At the open, with the index at \$100 and 6.5 hours left, our pricer gives an at-the-money 0DTE straddle (long the call *and* the put) a cost of about call \$0.2188 + put \$0.2159 = **\$0.4347** per share, or **\$43.47** for the 100-multiplier pair. That \$43.47 is the *whole bet* for the buyer and the *whole credit* for the seller.

- **The buyer** needs the index to finish more than \$0.4347 away from \$100 in *either* direction — below \$99.57 or above \$100.43 — just to break even, and further than that to profit. The breakeven move is only about **0.43%**, which sounds easy until you remember the index has to *get there and stay there* by 4:00, fighting accelerating theta the whole way.
- **The seller** keeps the full \$43.47 only if the index finishes inside that ±0.43% band. Step outside it and the loss is unbounded (for the naked straddle): a 1% close move (index at \$101) settles the straddle at \$1.00 intrinsic, so the seller's P&L is (0.4347 − 1.00) × 100 = **−\$56.53**. A 2% move settles at \$2.00 and the loss is **−\$156.53** — over three times the entire day's credit, gone on one trending afternoon.

The intuition: the straddle prices in a roughly 0.43% expected close move, and the buyer and seller are taking opposite sides of whether the *realized* close move beats it — a wager that resolves to a near-binary win or loss in a handful of hours.

## How it shows up in real markets

So far this has been about one trader holding one contract. The reason 0DTE earns a place in a market-structure conversation is that there is not one trader — there are millions of contracts a day, concentrating an enormous amount of gamma at near-the-money strikes that expire *today*. When that much short-dated gamma piles up, the people on the other side of it — the dealers and market makers — have to hedge it, and their hedging spills out into the index itself. This is where 0DTE stops being a personal-risk story and becomes a market-structure story.

The scale is what makes it matter. A single 0DTE contract's hedging is invisible; the aggregate of a market where same-day options are a majority of SPX volume is not. The flow concentrates because 0DTE strikes are listed at fine intervals around the current index level, so on any given morning a few round-number strikes carry an outsized share of the day's open interest — exactly the strikes the index is most likely to be sitting near at the close. That concentration of gamma at a handful of near-the-money strikes is what gives the dealer-hedging loop something to grip, and it is why the effects show up as *strike-specific* phenomena — a pin at one level, an air-pocket through another — rather than a uniform background hum.

### The dealer-hedging loop

When you buy or sell a 0DTE option, a market maker is usually on the other side. They do not want a directional bet; they want to earn the bid-ask spread and stay neutral. So they **delta-hedge**: if they are short a call (because you bought it), they buy index futures or the underlying to offset the delta, and they rebalance that hedge as the index moves. The full mechanics live in [Delta Hedging in Practice](/blog/trading/options-volatility/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral) and the dealer's-eye view in [How an Options Market Maker Thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) and the broader market on [Market Makers and High-Frequency Trading](/blog/trading/finance/market-makers-and-high-frequency-trading). The detail that matters for 0DTE is *how much* and *how often* they have to rebalance, and that is set by gamma.

Recall that gamma is how fast delta changes. A dealer hedging a high-gamma book has to rebalance constantly, because every small index move changes their delta a lot. A 0DTE book near the strike is the highest-gamma book there is. So a dealer who is net long that gamma must, to stay neutral, **sell the index as it rallies and buy it as it dips** — leaning *against* every move. A dealer who is net short that gamma must do the opposite: **buy as it rallies and sell as it dips** — leaning *with* every move, chasing it. Which sign the dealers are on, in aggregate, determines whether their hedging calms the market or feeds it.

![Feedback loop showing 0DTE flow leading to dealer high-gamma hedging that either pins the index or amplifies the move depending on dealer gamma sign](/imgs/blogs/0dte-and-the-rise-of-short-dated-options-the-new-market-structure-6.png)

The loop figure lays it out. Heavy 0DTE flow lands on dealers, who hedge at very high gamma. If they are net long gamma, their hedging sells rallies and buys dips, which **pins** the index near the heavily-traded strike and *dampens* intraday volatility — the tape gets glued to a round number into the close. If they are net short gamma, their hedging buys rallies and sells dips, which **amplifies** a move once a strike breaks, so a small slide can snowball into a fast one. Either way, the outcome moves the spot, which changes the flow on the next bar, and the loop runs all day and tightens into the close. The deep mechanics of this — how gamma, charm, and vanna in dealer books steer the spot — are the subject of the forthcoming [Dealer Gamma, Charm, and Vanna](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot); here the point is just that 0DTE flow is the loudest input into that machine.

#### Worked example: how much a dealer has to hedge a 0DTE book

Make the hedging churn concrete. Suppose dealers are collectively long the gamma of **10,000** at-the-money 0DTE contracts at a strike, with two hours left, where our pricer gives gamma = **1.3201** per share. For 100-multiplier contracts, the book's total gamma is 10,000 × 100 × 1.3201 = **1,320,100** — meaning the book's net delta changes by about 1.32 million share-equivalents for every \$1 the index moves. If the index ticks up \$1, the dealers' delta jumps by roughly 1.32 million long deltas, and to stay neutral they must **sell** about that much index exposure into the rally. Tick back down \$1 and they buy it back. Now compare a 30-day book of the same size: gamma = 0.0693, so the book gamma is only about 69,300 — they would sell only ~5% as much for the same \$1 move. The 0DTE book forces nearly **twenty times** the hedging volume per unit of index move, and it forces it *now*, in the same session. That is the mechanism: a wall of same-day gamma turns dealers into a giant, mechanical buyer-of-dips-and-seller-of-rallies (or the reverse), and the size of their forced flow is set directly by the gamma number — which, as we have seen, is at its absolute maximum for 0DTE near the strike. The intuition: it is not the *number* of 0DTE contracts that moves the index, it is the gamma those contracts carry, and 0DTE carries the most gamma per contract of anything listed.

A caution on reading this: which *sign* the dealers are on in aggregate is genuinely hard to know in real time. The popular "dealers are long gamma so we pin" narrative rests on estimates of net dealer positioning that are noisy and model-dependent. The honest version is conditional — *if* dealers are net long the near-the-money gamma, the hedging dampens; *if* net short, it amplifies — and the sign can flip intraday as flow comes in. Treat dealer-gamma dashboards as a weather forecast, not a law of physics.

### Pinning, and the round-number magnet

The most-discussed visible effect is **pinning**: the tendency of the index to gravitate toward, and stick near, strikes with heavy 0DTE open interest into the close. When dealers are long the gamma at a strike, their hedging is a restoring force — every move away from the strike triggers hedging that pushes back toward it — and the index can spend the last hour drifting in an unnaturally tight range around a round number. Traders who watch dealer-positioning estimates talk about a "call wall" or "put wall" that the index seems reluctant to cross. The noon-screenshot trader from the open lost partly to theta and partly to exactly this: the index that ramped in the morning got magnetized back toward a strike in the afternoon, and his out-of-the-money calls died on the pin.

### The suppress-or-amplify debate

This brings us to the central open question that regulators, exchanges, and researchers actually argue about: **do 0DTE options suppress volatility or amplify it?** The honest answer is *it depends, and on different timescales*.

On a typical calm day, the weight of evidence and the dealer-hedging logic point toward *suppression*. When dealers are net long the 0DTE gamma — which is often the case when the dominant flow is retail buying lottery tickets and systematic sellers harvesting premium leaves dealers long convexity — their rebalancing leans against moves, dampening intraday swings and contributing to the long stretches of eerily low realized volatility the index complex has seen since 2022. Many practitioners credit 0DTE pinning with part of the persistently compressed intraday ranges of recent years.

But the *tail* is the worry. If positioning flips so that dealers are net short the near-the-money gamma, their hedging chases moves, and a shock that would once have been absorbed could instead be amplified within the session — a self-reinforcing slide as hedging sells into weakness. The fear, voiced in regulatory and academic discussions, is that 0DTE has not abolished volatility but *repackaged* it: calmer most days, with the potential for a sharper, faster intraday dislocation on the rare day the loop runs the wrong way. The August 5, 2024 yen-carry-unwind morning, when the VIX spiked to **38.57**, and the broader worry about fragile intraday liquidity are the kinds of events this debate points at. Nobody has the final word yet, which is precisely why it is a live market-structure question and not a settled fact.

A few empirical observations are worth keeping in view, even with the caveats. The *intraday* path of the index has clearly changed shape: a larger fraction of the day's range now tends to print around the open and the close, with a quieter, more pinned middle, which is consistent with dealer-long-gamma hedging compressing the midday. At the same time, the *overnight* gap — the move from one close to the next open — has not gone away and arguably matters more, because 0DTE participants are flat overnight and the repricing happens when they are not there to dampen it. So one plausible reading is that 0DTE pushes volatility *out of the trading session and into the gaps and the close*, rather than reducing it overall. Another worry the exchanges study is whether a 0DTE-driven intraday cascade could feed the overnight VIX print or strain the futures market that dealers hedge in. None of this is settled, and serious people disagree; the responsible stance for a trader is to assume both effects exist — calmer midday, fatter intraday tail risk — and to never size a 0DTE position as if the calm is guaranteed.

There is one more structural wrinkle the debate often skips: the VIX itself is built from roughly 30-day options and is, by construction, *blind* to 0DTE. So an instrument that now dominates SPX volume barely registers in the headline fear gauge. A market can be perfectly placid by the VIX and still be churning violently inside the session at the 0DTE tenor. That gap between what the famous index measures and what is actually happening in the most-traded contracts is itself a reason 0DTE took so many people by surprise.

### The income trade across a year of days

To see why the income trade is so seductive and so dangerous in aggregate, model a full year of a single repeated 0DTE seller.

![Settle P&L of a short 0DTE iron condor versus the index close move, a small plateau of wins and deep capped losses, with a probability bell behind it](/imgs/blogs/0dte-and-the-rise-of-short-dated-options-the-new-market-structure-5.png)

The chart prices a defined-risk version of the income trade: at the open, sell the 0.5%-out-of-the-money 0DTE call and put, buy the 1.5%-out-of-the-money wings for protection — a tight iron condor. The blue line is the settle P&L as a function of where the index closes; the shaded gray bell behind it is the probability of each close move, using a daily volatility of about 1.26% (20% annualized). The shape tells the whole story: a flat green plateau of small wins when the index finishes inside the range, then a cliff down to a capped — but large — red loss once it trends through a wing. And the probability bell shows that *most* days land in the plateau.

#### Worked example: the asymmetry of the income trade

Using the pricer at the open (index \$100, 6.5 hours left), the 0.5%-wide, 1-point-wing condor collects a credit of about **\$0.1049** per share — **\$10.49** per condor. The wings are \$1.00 apart, so the maximum loss is (1.00 − 0.1049) × 100 = **\$89.51**. The risk-to-reward is therefore about **8.5 to 1**: you risk \$89.51 to make \$10.49.

Now weigh it by probability. With a 1.26% daily sigma, the chance the index finishes within ±0.5% of the open — the full-win zone — is only about **31%**, but the chance it finishes within the breakevens (closer to the sold strikes plus credit) keeps the *win rate* well above half; defined-risk 0DTE sellers routinely report win rates of 70–85%. So you win most days and lose rarely. The catch is the **8.5:1** ratio: a single full-loss day erases roughly eight winning days. Win 80% of days and you can *still* lose money over a year if the average loss is 8.5x the average win and the losses cluster on trend days — which they do, because trend days are exactly when premium sellers all get run over together. The intuition: a high win rate on a short-gamma trade is not an edge, it is a *liability disguised as one*, because it lulls you into sizing up right before the day that takes it all back.

## Common misconceptions

0DTE attracts more confident wrong beliefs than almost any instrument, partly because it is new and partly because its extreme Greeks make every intuition built on normal options misfire. Each myth below is corrected with a number from the model, because in 0DTE the numbers are not close to the intuitions — they are off by an order of magnitude in the direction that hurts you.

**"0DTE options are cheap, so the risk is small."** The premium is small; the *risk* is not. A long 0DTE buyer's max loss is the premium — fine. But the typical loss is not "some of the premium," it is **all of it**, and it happens *most days*, because the theta is brutal and the option usually expires worthless. Our intraday-melt example showed an at-the-money call going from \$0.22 to near zero in one session with the index *unchanged*. Cheap per ticket times a high probability of total loss is not cheap; it is a high-frequency way to bleed. And for the seller, "cheap" is irrelevant — the \$10.49 credit in our condor sits in front of an \$89.51 loss.

**"If I'm right about the direction, I make money."** Being right about direction is necessary and not sufficient. The noon-screenshot trader was right — the index was up most of the day — and still lost 100%, because a 0DTE option pays off based on where the index sits *at the bell*, not on whether your view was correct at noon. The delta-flip example makes the math literal: with 30 minutes left, a \$0.50 swing in the index moves the option's delta from 0.05 to 0.95. You can be right by \$0.40 at 3:55 and worthless at 4:00.

**"Selling 0DTE premium is steady income with an 80% win rate."** The 80% win rate is real and it is exactly the problem. The condor example collects \$10.49 against an \$89.51 risk — an **8.5:1** loss-to-win ratio. At that ratio, a 80% win rate is *barely* break-even in expectation before costs, and it is negative once a trend day's loss is larger than the modeled max (gaps, slippage, and getting stuck on the wrong side of a fast move all push realized losses past the tidy \$89.51). The win *rate* is the seductive number; the win-to-loss *ratio* and the clustering of losses are the numbers that decide whether you survive.

**"0DTE options are immune to volatility because their vega is tiny."** They are nearly immune to *implied* volatility — true, vega at two hours is about \$0.0085 per vol point versus \$0.114 for a 30-day option. But they are *hypersensitive* to **realized** volatility, which is a different thing. A 0DTE option lives or dies on how much the index actually moves before the close. Low vega does not mean "safe from volatility"; it means "the bet is on realized moves, not on the market's vol forecast." A 0DTE seller is short realized vol in the most concentrated form there is.

**"0DTE flow has made the market calmer, full stop."** On most days the dealer-long-gamma pinning effect does dampen intraday swings — that part is well supported. But "calmer on average" is not "safer." The same flow that suppresses volatility on quiet days can amplify it on the rare day positioning flips, repackaging steady small swings into the potential for a sharper intraday dislocation. The suppress-versus-amplify debate is unresolved precisely because both effects are real on different days; claiming 0DTE has simply tamed volatility ignores the tail the regulators are actually worried about.

## The playbook: how to trade 0DTE without blowing up

0DTE is not a strategy; it is an *instrument*, and an extreme one. The playbook is therefore not "the best 0DTE setup" — it is a set of survival rules that apply no matter which side you take. The goal is to be able to keep trading it next year, which is a higher bar than most 0DTE traders clear.

![Decision figure for the 0DTE playbook: defined risk only, size for a full loss, close before the gamma cliff, respect the binary](/imgs/blogs/0dte-and-the-rise-of-short-dated-options-the-new-market-structure-7.png)

The decision figure puts the four rules in order. Clear all four and you have a small, defined-risk, intraday bet you can repeat for years. Fail any one and the correct action is to not trade — there is always another open.

**Rule 1 — defined risk only.** Never trade 0DTE with an undefined-loss structure. No naked short options, no naked short straddles, no "I'll just sell a few puts." The gamma spike means a naked short can go from a \$10 winner to a four-figure loser inside the final hour, and the binary nature means you cannot reliably hedge your way out at the close. Trade verticals, [iron condors and credit spreads](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range), or [butterflies and broken-wings](/blog/trading/options-volatility/butterflies-ratio-spreads-and-broken-wings-the-precision-tools) — structures where you know your maximum loss before you click the button. If you are buying, your max loss is the premium by construction; if you are selling, *buy the wings.*

**Rule 2 — size for a full loss, on every ticket.** Assume each trade loses its entire defined risk, and size so that even a string of those losses on consecutive trend days cannot threaten the account. For most retail traders that means risking well under **0.5% to 1% of equity** per 0DTE trade — and remembering, from the income-trade math, that an 8.5:1 short-gamma trade needs even smaller sizing than the win rate suggests, because the losses cluster. This is the same logic as [Position Sizing and Risk of Ruin in Options Trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading): the high win rate of a short-gamma 0DTE trade is exactly the thing that tempts you to over-size right before the day it all comes back.

Put a number on it. Say you have a \$50,000 account and decide to risk 0.5% — \$250 — per 0DTE trade. With our defined-risk condor risking \$89.51 per contract, that is at most **two** contracts a day, not twenty. The seller looking at an 80% win rate and a steady stream of \$10.49 credits feels that two contracts is absurdly timid — at this rate it takes weeks of wins to make a few hundred dollars. That feeling is the trap. The correct response to "this size feels too small" on a short-gamma 0DTE trade is *that is what survival feels like*, because the day the index trends 2% through your wing, your full risk is realized on every contract you held, and the trader who sized for the win rate instead of the loss is the one who does not come back. Size off the \$89.51 you can lose, never off the \$10.49 you usually make, and never let a streak of green days talk you into adding a zero.

**Rule 3 — close before the gamma cliff.** The final hour is where gamma is highest, the delta flip is sharpest, and the pin-or-amplify loop is tightest. That is the worst time to be holding a binary you cannot control. If you are long for an intraday move, take the profit or the loss and flatten well before the bell; do not "let it ride into expiry" hoping the pin breaks your way. If you are short premium, have a plan to close or roll *before* the index is sitting on your strike in the last thirty minutes, because that is precisely when a small wiggle turns a winner into a max loss. Holding a 0DTE coin flip into the close is not a trade, it is a gamble on the final tick.

**Rule 4 — respect the binary.** A 0DTE position resolves to roughly win-big-or-zero, and almost nothing you do at 3:50 changes that. So treat each trade as one ticket with one outcome: no averaging down a losing long (you are buying more of a faster-melting ice cube), no rolling a losing short into the same expiry (you are just re-arming the bomb), no "doubling to recover." Decide your entry, your defined risk, and your exit before you put it on, and then let the outcome happen. The edge in 0DTE, to whatever extent it exists, is not in the premium — the variance risk premium is thin at this tenor and easily eaten by the bid-ask spread and the [hidden tax of getting filled](/blog/trading/options-volatility/liquidity-bid-ask-spreads-and-getting-filled-the-hidden-tax). The edge is in discipline and survival: trading small, defined, repeatable bets and being there next year.

A useful net-Greeks habit ties it together: before you put on any 0DTE position, write down its delta, its gamma, and its theta the way you would for any book — see [The Net Greeks of a Position](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard). For 0DTE the dashboard will scream two things at you: the gamma number will be huge (so a small move is a big P&L swing) and the theta number will be a steady credit or debit (so time is for you or against you every minute). If you are not comfortable with both of those at full size, the position is too big. And if the position is short gamma into the close, the dashboard is telling you the one thing that matters: a gap through your strike is a cliff, and you must know exactly how far you can fall.

The honest summary is the one the noon-screenshot trader learned the hard way and the disciplined seller learns slowly over a hundred quiet days: a 0DTE option is the most explosive instrument on the board *because* its Greeks are at the limit, and the flood of 0DTE flow has rewired the market's intraday behavior around that fact. You can trade it — many do, every day — but only as a small, defined, intraday bet you have already priced as a coin flip you can afford to lose. The theta is brutal, the gamma is binary into the close, and the market structure it created can pin you to a strike or run you over before the bell. Trade it like that and it is a tool. Trade it like income you can count on and it is, eventually, the screenshot.

## Further reading & cross-links

Within this series:

- [Gamma: The Greek That Bites — Curvature, Convexity, and the Toxic Short](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) — the curvature that 0DTE turns up to eleven.
- [Theta: Trading the Clock and the Price of Being Long Options](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options) — the decay that runs at full speed in a single session.
- [Assignment, Pin Risk, and Expiration-Day Mechanics](/blog/trading/options-volatility/assignment-pin-risk-and-expiration-day-mechanics) — the settlement and pinning details behind the close.
- [Iron Condors and Credit Spreads: Selling the Range](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range) — the defined-risk structures the income trade is built from.
- [Butterflies, Ratio Spreads, and Broken Wings: The Precision Tools](/blog/trading/options-volatility/butterflies-ratio-spreads-and-broken-wings-the-precision-tools) — precision structures for a same-day view.
- [Position Sizing and Risk of Ruin in Options Trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — why the high win rate must not set the size.
- [The Net Greeks of a Position: Building Your Risk Dashboard](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) — the dashboard to read before every 0DTE ticket.
- [Dealer Gamma, Charm, and Vanna: How Options Flows Move the Spot](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot) — the deep mechanics of the market-structure loop (forthcoming).

For the pricing theory this post relied on:

- [The Black-Scholes Model](/blog/trading/quantitative-finance/black-scholes) — the pricing model every number here was computed from.
- [Market Makers and High-Frequency Trading](/blog/trading/finance/market-makers-and-high-frequency-trading) — who is on the other side of your 0DTE ticket and how they hedge it.
