---
title: "The Expected Move: How Options Price Event Risk"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How the options market tells you, before any release, how big a move it expects from an event — the expected move — and how to compute it, read it, and trade around it."
tags: ["event-trading", "macro", "expected-move", "options", "implied-volatility", "straddle", "cpi", "vix", "bitcoin", "risk-sizing"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Before any macro release, the options market publishes, in advance, how big a move it expects from the event. That number is the **expected move**, and it is roughly the price of the at-the-money straddle, or equivalently `S × IV × √(t/365)`. It is the single most useful number for an event trader.
>
> - The expected move is a **one-standard-deviation (1σ) band** around the spot price over the window to the event. About **68%** of outcomes are supposed to land inside it; it is *not* a forecast and *not* a cap.
> - To read a print's reaction, compare it to the band. The day before a CPI report the S&P 500 straddle might imply **±1.2%** — that is the bar the print has to clear to genuinely move markets. Inside the band = priced; outside = a real surprise.
> - Crypto prices a far wider move than equities: a typical single-name event might price the S&P at **±1.2%** while Bitcoin prices **±4%** for the same calendar window, because Bitcoin's implied vol is multiples higher.
> - The one number to remember: in September 2022 the S&P straddle priced roughly **±1.2%** into a CPI print, and the actual same-day move was **−4.32%** — more than three times the priced move. The cone is a starting point, not a wall.

The afternoon before a U.S. CPI release, a trader does not stare at a forecast of inflation. She pulls up the option chain on the S&P 500 for the contract that expires the day of the print, finds the strike sitting right at the current index level, and reads off two prices: the call and the put. She adds them together. That sum, divided by the index level, is a percentage — and that percentage is the entire conversation the market is having about tomorrow, compressed into one number.

On one such afternoon in September 2022, that number was about **1.2%**. The S&P 500 was near 4,100, the at-the-money straddle that expired into the print cost roughly the equivalent of a 1.2% move of the index, and so the options market was saying, in effect: *we expect this print to move the S&P about 1.2% in one direction or the other, and we are not sure which.* That ±1.2% was the bar. A print that produced a move inside that band would be, by the market's own pre-stated reckoning, a non-event — already priced. A print that produced a move outside it would be a surprise.

The next morning the August CPI came in hot — 8.3% year-over-year against an expected 8.1%, with core inflation *rising* when the consensus had it falling — and the S&P 500 closed **down 4.32%**, its worst session since June 2020. The realized move was more than three times the priced move. The cone the options had drawn was blown clean through. This post is about that number — what it is, how to compute it two different ways, how to read it, why crypto's version is several times wider, how to size a position to it, and what it means when a print sails straight past the edge of the cone.

![Expected-move cone from spot to the event date with one-sigma band](/imgs/blogs/the-expected-move-pricing-event-risk-with-options-1.png)

The figure above is the whole idea in one picture. Spot today sits on the left. The options market projects a band — a cone — that widens out to the event date on the right. The green core of the cone is what is *priced*: the set of outcomes the market already expects, where the reaction will be small because there is no surprise. The red wings, above and below, are where a print becomes news. Everything in this post is about measuring the width of that cone and trading the relationship between it and what actually happens.

## Foundations: what the expected move actually is

Let us build this from zero, because every term here is going to do real work later.

**An option** is a contract that gives its owner the right — not the obligation — to buy or sell an asset at a fixed price (the *strike*) by a fixed date (the *expiry*). A **call** is the right to buy; a **put** is the right to sell. You pay for that right up front; that payment is the **premium**. If you own a call and the asset finishes above the strike, you exercise and capture the difference; if it finishes below, you let the call expire and lose only the premium. A put is the mirror image: it pays off when the asset falls below the strike.

**At-the-money (ATM)** means the strike equals (or is closest to) the current price of the asset. The ATM call and the ATM put are the most sensitive options to a move in either direction, and they are the building blocks of the expected move.

**Implied volatility (IV)** is the number that falls out of an option's price when you run it backward through a pricing model. An option's premium depends on the strike, the time to expiry, interest rates, and how much the asset is expected to move. Everything but that last quantity is observable, so given the market price of the option you can solve for the one unknown: how much movement the market is pricing in. That solved-for number, expressed as an annualized percentage standard deviation of returns, is the implied volatility. **IV is the market's forecast of future movement, stated in the price of options.** It is forward-looking, unlike *realized* (or *historical*) volatility, which measures how much the asset has actually moved in the past.

Now the central object.

**The expected move** is the size of the move the options market is pricing into a given window of time — say, between now and the day after an event. It is a *one-standard-deviation* band. In a roughly bell-shaped distribution of outcomes, about **68%** of the probability mass sits within one standard deviation of the center. So the expected move marks the band inside which the options market thinks the asset will land about two times out of three, and outside which it will land about one time in three.

There are two equivalent ways to read it off the market, and they should agree.

**Method one — the ATM straddle.** A **straddle** is the combination of buying the ATM call and the ATM put at the same strike and expiry. Its cost — the two premiums added together — is, to a very good approximation, the expected move in dollars over the life of the options. The intuition is clean: the straddle pays off on a move in *either* direction, so what you pay for it is exactly the market's price of "a move of unknown sign." Divide the straddle cost by the spot price and you get the expected move as a percentage.

**Method two — the formula.** The expected move can also be approximated directly from implied volatility and time:

```
expected move ($) ≈ S × IV × sqrt(t / 365)
```

where `S` is the spot price, `IV` is the annualized implied volatility as a decimal, and `t` is the number of calendar days until expiry. The square-root-of-time term is the engine of the whole subject: **volatility scales with the square root of time, not with time itself.** A move expected over four days is twice the size of a move expected over one day, not four times, because random moves partly cancel. We will return to this; it is the reason a one-day event window prices a much smaller move than the same IV would imply over a month.

**The break-even.** Because a straddle costs money, a long straddle only makes money if the asset moves *more* than the expected move. The two **break-even** points sit at the strike plus the straddle cost (on the upside) and the strike minus the straddle cost (on the downside). A move that lands exactly on a break-even returns the premium and nets zero; a move that lands inside the break-evens loses money for the straddle buyer. **The break-evens are the expected move, drawn as the two prices the asset must clear for a straddle buyer to profit.** That is not a coincidence — the expected move *is* the straddle cost, expressed as a distance from the strike.

**Priced versus realized.** Everything above is the *priced* move — what the options say before the event. After the event, the asset makes a *realized* move — what actually happened. The entire craft of event trading lives in the gap between the two: a realized move smaller than priced means the straddle buyer overpaid and the seller won (the "vol crush"); a realized move larger than priced means the buyer was right and the cone was too narrow. We will spend a whole section on that gap, because it is where the money is.

So: the expected move is a 1σ, ~68% band; it equals the ATM straddle cost; it can be cross-checked with `S × IV × √(t/365)`; its edges are the straddle break-evens; and it is a *forecast of magnitude, not direction, and not a limit.* Hold those five facts and the rest of the post is detail.

### Where the 68% comes from, and why it is only an approximation

The 68% figure is the area under one standard deviation of a normal bell curve, and option pricing leans on the assumption that returns are roughly normally distributed in log terms. That assumption is good enough to use the expected move every day, but it is wrong in two ways that matter, and a serious event trader keeps both in mind.

First, real markets have **fat tails**: extreme moves happen far more often than a normal curve predicts. The September 2022 −4.32% S&P day was nominally a 3.6-standard-deviation event against the priced band, and under a strict normal curve a 3.6σ down-day should occur roughly once every few thousand sessions. They happen several times a decade. So the true probability of breaching the cone is higher than the clean 32% the normal curve implies — the tail is heavier than the model. This is precisely why the cone must never be treated as a cap, and why the most disciplined event traders keep their tail risk *defined* (long an out-of-the-money wing, or a hard stop) rather than naked-short the cone.

Second, the distribution around an event is often **bimodal**, not bell-shaped. Before a binary catalyst — a court ruling, a make-or-break earnings number, a referendum, a spot-ETF approval decision — the asset is likely to gap one way or the other and *unlikely* to sit still. The true distribution has two humps, one at "yes" and one at "no," with a valley in the middle. The single ATM expected move still measures the *average* magnitude, but it understates how likely a large move is and overstates how likely a tiny one is. When you see an unusually wide cone before a known binary, that is the options market straining to price a two-humped distribution with a one-number tool.

The practical upshot is the same in both cases: **the expected move is a reliable measure of the typical magnitude and an unreliable measure of the tail.** Use it to size the normal day; never use it to bound the abnormal one.

### Implied volatility is read off the chain, not handed to you

One more foundational point, because beginners often assume IV is a single published number. It is not — there is an implied volatility for *every* strike and *every* expiry, and they differ. The set of all of them is the **volatility surface** (covered in depth in [the volatility surface](/blog/trading/quantitative-finance/volatility-surface)). For the expected move you want the IV of the *at-the-money* options for the *expiry that captures the event* — that single point on the surface. In practice you rarely back it out by hand: you read the straddle price directly, which already bakes in whatever IV the market is using, plus the skew and the event premium the clean formula ignores. That is the deeper reason the straddle is the truth and the formula is the sanity check — the straddle is what trades, so its price is the market's actual, all-in answer.

## Computing the expected move: two roads to the same number

Let us make this concrete with the standard worked inputs, and verify that the two methods agree.

#### Worked example: the expected move from the straddle

Suppose the S&P 500 is at **5,000** and the at-the-money straddle expiring into the event — the ATM call plus the ATM put — costs **\$60** of index points.

- Add the legs: the whole straddle costs **\$60** (this is the call premium plus the put premium combined).
- As a percent of spot: `\$60 / 5,000 = 0.012 = 1.2%`.
- So the expected move is **±1.2%**, i.e. **±\$60** of index level around 5,000.
- Upper break-even = `5,000 + 60 = 5,060`; lower break-even = `5,000 − 60 = 4,940`.

The market is saying it expects the index to finish somewhere between 4,940 and 5,060 about two-thirds of the time. Intuition: the straddle price *is* the expected move; you do not need a model to read it. (Option prices here are illustrative methodology, not a live quote.)

Now the second road, which should land in the same place.

#### Worked example: the expected move from the formula

Take the same index at **5,000**, an annualized implied volatility of **18%**, and a **7-day** window to the event.

- Time factor: `sqrt(7 / 365) = sqrt(0.01918) = 0.1385`.
- Expected move in dollars: `5,000 × 0.18 × 0.1385 = \$124.6`, call it **\$125**.
- As a percent: `\$125 / 5,000 = 2.5%`... over the full seven days.
- But the *event-day* move is the move concentrated in the print. Most of that 7-day variance is spent on the single event day, so the single-session expected move the straddle prices is smaller — around **±1.2%** when the rest of the week is quiet.

The lesson is that the raw formula gives you the move over the whole window; the *event* move depends on how much of that window's variance the market thinks the event itself carries. Intuition: the formula and the straddle agree once you match the time windows — both are just `S × IV × √t` wearing different clothes.

![Expected-move formula with spot, implied vol, and time to event plugged in](/imgs/blogs/the-expected-move-pricing-event-risk-with-options-4.png)

The figure walks the formula left to right: spot, implied vol, and days-to-event flow into the `S × IV × √(t/365)` box, which produces the dollar expected move and then the percent. The single most important behavior to internalize is the **square-root-of-time** term. Because variance (volatility squared) adds linearly across independent days but volatility itself does not, a move expected over four days is `sqrt(4) = 2` times a one-day move, not four times. This is why the expected move *widens as the event approaches gets further away* and *narrows as expiry shortens* — and why a weekly option that straddles a CPI print prices a much bigger move than a weekly option in a quiet week, even at the same headline IV: the event concentrates variance into one session.

Why do the two methods agree? Because the Black-Scholes price of an ATM straddle is, to first order, `≈ 0.8 × S × IV × √(t/365)` — the same formula with a constant of about 0.8 in front. Traders often drop the 0.8 and use the straddle price directly, because the straddle is what they actually trade and it already contains the market's full pricing (skew, kurtosis, the event premium) that the clean formula leaves out. **The straddle is the truth; the formula is the sanity check.**

### Isolating the event from the calendar: the variance-subtraction trick

There is a sharper technique worth teaching, because it answers a real question: *how much of the expected move is the event itself, versus the ordinary noise of the days around it?* The straddle that expires after a CPI print prices the move over its whole life — the event day *plus* the quiet days. If you want the event-day move on its own, you isolate it by subtracting the non-event variance.

The key fact is that **variances add, volatilities do not.** Variance is volatility squared, and over independent days the total variance is the sum of the daily variances. So the variance of a window that contains the event equals the variance of an equivalent window *without* the event, plus the event's own variance. Rearranged, the event's contribution is the difference of two squared expected moves.

#### Worked example: backing out the pure event-day move

Suppose a straddle that spans the CPI print (7 days, covering the event) prices a **±1.8%** expected move, while an otherwise identical 7-day window with *no* event prices **±1.0%** (just ordinary noise).

- Square each to get the variance contributions: event-window `1.8² = 3.24`; quiet-window `1.0² = 1.00`.
- The event's own variance is the difference: `3.24 − 1.00 = 2.24`.
- The pure event-day expected move is the square root: `sqrt(2.24) ≈ 1.50` — so **±1.5%** is what the print itself is pricing.
- On a **\$30,000** position, that pure event move is `\$30,000 × 0.015 = \$450` of the swing attributable to the print alone.

Intuition: subtract the squared "normal" move from the squared "event" move and take the root — what's left is the cleanest read on how big a deal the market thinks the print is, stripped of calendar noise.

This is exactly how desks quote an "implied earnings move" or an "implied CPI move": not the raw straddle, but the straddle's expected move with the background vol subtracted out. It also explains a recurring puzzle — why a weekly option that happens to span an event prices a much larger move than the same-tenor option in a quiet week. The difference *is* the event premium, and the variance-subtraction trick measures it.

The structure of the long straddle — and where its break-evens sit — is worth drawing out, because the break-evens *are* the expected move.

![Long straddle payoff with strike, two break-evens, and max loss labeled](/imgs/blogs/the-expected-move-pricing-event-risk-with-options-2.png)

Read the payoff from the middle outward. At the strike (5,000) the straddle is worth nothing at expiry — both legs expire at-the-money — so the buyer's loss is the full \$60 premium; that is the deepest point of the V. Move the index down to 4,940 and the put has gained \$60, exactly offsetting the premium: lower break-even. Move it up to 5,060 and the call has gained \$60: upper break-even. Beyond either break-even the position profits, dollar for dollar, on every further point of movement. The red middle zone is the "small move" region where the buyer loses; the green wings are where a big move pays. **A straddle is a bet that the realized move beats the priced move — that the asset finishes outside the cone.**

## Reading the number: what a ±1.2% SPX or ±4% BTC move means

A number with no reference is useless. The skill of reading the expected move is in comparing it — to the asset's own history, to the calendar, and across assets.

Start with **what ±1.2% on the S&P 500 feels like**. The index moves about 0.7% on an *average* day. A 1.2% expected move into a single event is therefore the market pricing roughly **1.7× a normal day's move** packed into one session — a clear signal that the event matters. By contrast, on a sleepy week with no top-tier data, a one-day expected move might be 0.5%, *below* the daily average, because the options are pricing a quiet session. **The expected move is most informative as a multiple of the asset's normal daily range.** A 1.2% print-day band on the S&P says "this is a big deal"; a 0.5% band says "the market has already made up its mind."

Now translate the band into money, because percentages hide the stakes.

#### Worked example: turning the band into dollars on a real position

Say you hold **\$25,000** of an S&P 500 index fund into a CPI print that prices a ±1.2% expected move.

- The 1σ band in dollars: `\$25,000 × 0.012 = \$300` either way.
- So two times out of three, the print should move your position less than **\$300**.
- The other one time in three, it moves *more* — and there is no upper limit, because the band is not a cap.
- On the actual hot September-2022 print, the S&P fell 4.32%: `\$25,000 × −0.0432 = −\$1,080` — three and a half times the priced 1σ band.

Intuition: the expected move tells you the *typical* dollar swing to brace for, and the September example tells you the tail can be three or four times larger.

**Bitcoin prices a far wider move.** The same logic runs on crypto, but the numbers are bigger because Bitcoin's implied volatility is multiples of an equity index's. Where the S&P might price ±1.2% into an event, Bitcoin's options might price **±4%** for the same calendar window — and into a crypto-native catalyst (a spot-ETF decision, a major exchange event, a large protocol unlock) the band can be far wider still.

#### Worked example: the expected move on Bitcoin

Suppose Bitcoin is at **\$60,000** and the ATM straddle into an event costs **\$2,400**.

- As a percent of spot: `\$2,400 / 60,000 = 0.04 = 4.0%`.
- So the expected move is **±4%**, i.e. **±\$2,400** around \$60,000.
- Break-evens: `60,000 + 2,400 = \$62,400` up; `60,000 − 2,400 = \$57,600` down.
- The same band on a **\$10,000** BTC position is `\$10,000 × 0.04 = \$400` either way — versus the \$300 the equity band implied on \$25,000.

Intuition: Bitcoin's cone is roughly three times wider than the S&P's, so the bar a catalyst must clear to "surprise" crypto is far higher in percentage terms. (Inputs illustrative.)

![Priced expected move side by side for the S&P 500 and Bitcoin](/imgs/blogs/the-expected-move-pricing-event-risk-with-options-5.png)

The figure puts the two cones side by side: the S&P's ±1.2% against Bitcoin's ±4%. The visual point is not that crypto is "riskier" in some loose sense — it is that the *priced* move, the bar a catalyst must clear to register as a surprise, is several times higher for Bitcoin. A 3% Bitcoin day on a CPI print is *inside* its cone and barely news; a 3% S&P day is more than double its cone and a major event. You cannot read a percentage move without knowing the asset's expected move; the same 3% is routine for one and historic for the other. We unpack why crypto's volatility runs so much hotter in [crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset).

This is also why the expected move is the right denominator for comparing reactions across assets. Saying "Bitcoin fell 9% and the S&P fell 4% on the hot CPI" makes Bitcoin sound far more shaken — but if Bitcoin priced ±9% and the S&P priced ±1.2%, then the S&P's move was the more *surprising* one relative to what was priced. **Always measure a reaction in units of its own expected move, not in raw percent.**

### Reading the calendar: the term structure of expected moves

The expected move is not one number but a *curve* across expiries, and the shape of that curve tells you where the market thinks the risk is concentrated. Line up the straddles for each upcoming weekly and monthly expiry and you get a term structure: the cone for the expiry that captures next Tuesday's CPI, the cone for the one that captures the FOMC two weeks later, the cone for the quiet week in between. The cones that straddle an event bulge outward; the cones over quiet weeks sit lower. Reading that curve is how a desk decides *which* event the market is most braced for.

#### Worked example: comparing two events on the calendar

Say the weekly straddle that captures next week's CPI prices a **±1.3%** expected move, while the one that captures the FOMC the following week prices **±1.7%**, and the in-between quiet week prices **±0.6%**.

- The CPI cone (±1.3%) on a **\$50,000** position is `\$50,000 × 0.013 = \$650` of 1σ swing.
- The FOMC cone (±1.7%) on the same position is `\$50,000 × 0.017 = \$850` — the market is bracing for a bigger move from the Fed than from the print.
- The quiet week (±0.6%) is `\$50,000 × 0.006 = \$300`, below the daily average — the options are pricing a sleepy stretch.
- The *event premium* on the FOMC week is the variance-difference over the quiet week: `sqrt(1.7² − 0.6²) ≈ 1.59` → roughly **±1.6%** attributable to the FOMC alone, or `\$50,000 × 0.016 = \$800`.

Intuition: the term structure ranks your calendar by how much the market fears each event, and the variance-subtraction trick converts each bulge into the pure event premium so you can compare a CPI to an FOMC on equal footing.

The shape carries information beyond size. A cone that bulges sharply at one expiry and collapses right after tells you the market sees a single binary catalyst there. A term structure that is uniformly elevated across many weeks says the market is in a high-volatility *regime*, not bracing for one print. And an *inverted* term structure — near-dated cones higher than far-dated ones — is the classic stress signature: the market is pricing acute near-term danger that it expects to resolve. The August 2024 carry cascade, when the VIX spiked to an intraday 65.73 from a low-20s close just days earlier, was exactly that: the near-dated cone exploded while longer-dated vol barely moved, because the market judged the shock acute but not permanent.

## Priced versus realized: when the print blows through the cone

Here is the heart of it. The expected move is a forecast; the realized move is the outcome; and the gap between them is the event trade.

When the realized move lands *inside* the cone, the options were "too expensive" — the buyer of the straddle paid for a move that did not arrive, and the implied volatility that was elevated into the event collapses the moment the number prints. That collapse is the **volatility crush**: IV deflates as the uncertainty resolves, and option prices fall even if the underlying barely moves. A straddle buyer who is right about direction but wrong about *magnitude* can still lose money, because the vol crush bleeds the premium out of both legs. (We treat the crush in depth in a companion post on implied-versus-realized volatility around events.)

When the realized move lands *outside* the cone, the options were "too cheap" — the buyer was right, the seller is now bleeding, and the move keeps going past the break-even with no cap. This is the regime the September 2022 CPI created. Let us look at the three CPI episodes against the priced cone.

![Priced expected move band versus realized S&P 500 move on three CPI days](/imgs/blogs/the-expected-move-pricing-event-risk-with-options-3.png)

The blue band is the illustrative ±1.2% the S&P straddle priced into a CPI print. The bars are the realized same-day S&P moves on three actual CPI sessions. Two of the three blew straight through the cone:

- **August 2022 CPI (released Sep 13 2022)** — hot, 8.3% versus 8.1% expected, core *rising*. The S&P fell **−4.32%**, more than three times the priced ±1.2% band.
- **October 2022 CPI (released Nov 10 2022)** — cool, 7.7% versus 7.9% expected. The S&P rose **+5.54%**, more than four times the band, its best session of the year.
- **October 2023 CPI (released Nov 14 2023)** — mildly cool, 3.2% versus 3.3% expected. The S&P rose **+1.91%**, modestly past the band — and the rate-sensitive Russell 2000 small-cap index ran **+5.44%**.

Two of three prints realized a move several times the priced one. That is not a failure of the expected move — it is the expected move working exactly as designed. Remember: it is a 1σ band, and a 1σ band is *supposed* to be breached about a third of the time. What these episodes show is that when a CPI surprise is large and lands in a fragile regime, the realized move clusters in the *tail* of the distribution, well past the band. The cone tells you the typical day; the regime tells you whether you are about to get a typical day.

#### Worked example: the straddle buyer's P&L versus the realized move

Take the \$60 straddle on the S&P at 5,000 (expected move ±1.2%, break-evens 4,940 / 5,060) and walk three outcomes.

- **A 1.0% move (inside the cone):** the index moves to 5,050 (or 4,950). The winning leg is worth `5,050 − 5,000 = \$50`; the other leg expires worthless. P&L = `\$50 − \$60 = −\$10`. The buyer *loses* even though he was right about direction, because the move did not clear the cost.
- **A 1.2% move (on the break-even):** the index reaches 5,060. The call is worth \$60; total P&L = `\$60 − \$60 = \$0`. Break-even, exactly.
- **A 4.32% move (the September 2022 case):** the index falls 216 points to 4,784. The put is worth `5,000 − 4,784 = \$216`; P&L = `\$216 − \$60 = +\$156` per straddle — a 2.6× return on the premium.

Intuition: the straddle buyer needs the realized move to beat the priced move; inside the cone he bleeds the premium, outside it he is paid the overshoot.

The asymmetry is stark and worth stating plainly. The straddle *buyer's* worst case is the full premium (−\$60); his best case is unbounded. The straddle *seller's* best case is the full premium (+\$60); his worst case is unbounded. So selling the expected move — collecting premium and betting the print stays inside the cone — wins small and often and loses big and rarely. That is the trade that worked for years of quiet CPIs and then detonated on September 13 2022. **Selling event vol is picking up a known premium in front of an unknown tail; the expected move tells you exactly how far the tail has to reach to hurt you.**

There is a second engine in the seller's favor that the simple payoff diagram hides: the **volatility crush**. Implied volatility ramps *up* into a known event because the options are pricing the uncertainty of the print; the moment the number is released, that uncertainty resolves and IV collapses, often within seconds. Because option prices rise with IV, the crush deflates both legs of the straddle even if the underlying barely moves. So the seller is paid twice when the print lands inside the cone: once because the realized move was small, and again because the IV he was short fell. This is why a straddle *buyer* can pick the direction correctly and still lose — the crush bleeds the unused leg faster than the winning leg gains.

#### Worked example: the straddle seller's P&L with the vol crush

Sell the \$60 straddle on the S&P at 5,000 and walk the same outcomes from the seller's side.

- **A 0.4% move (well inside the cone):** the index moves 20 points to 5,020. The buyer's winning leg is worth \$20; you keep `\$60 − \$20 = +\$40` of the premium. The vol crush helps you close even cheaper.
- **A 1.0% move (inside the cone):** index to 5,050; the buyer's leg is worth \$50; you keep `\$60 − \$50 = +\$10`. Thin, but positive.
- **The 4.32% September-2022 move:** index falls to 4,784; the buyer's put is worth \$216; you lose `\$216 − \$60 = −\$156` per straddle — a single breach erases the premium from more than two dozen quiet prints.

Intuition: the seller collects a small, near-certain edge from the crush and the typical small move, then hands back many prints' worth of premium on the one print that clears the cone — which is why the expected move (the size of the tail) matters far more to a seller than the win rate.

The economics, then, are profoundly asymmetric across the two sides, and the expected move is what makes the asymmetry *measurable*. The seller is collecting the area inside the cone; the buyer is collecting the area in the tails. A market that prices the cone "fairly" gives neither an edge on average — but a market that systematically prices the cone *too wide* (because everyone fears the tail and overpays for protection) hands the seller a structural premium, the well-documented "volatility risk premium." Selling that premium is a real strategy; it is also the strategy that blows up every few years, on exactly the September-13-2022 days when the realized move clears the cone by a multiple. We treat the crush mechanism in full in a companion post in this series on implied versus realized volatility.

## Using the expected move for sizing and stops

The expected move is not just a reading instrument; it is a position-sizing instrument. If you know the 1σ adverse move, you can size a position so that a 1σ move against you costs no more than a fixed budget.

![Sizing a position so a one-sigma adverse move equals the risk budget](/imgs/blogs/the-expected-move-pricing-event-risk-with-options-6.png)

The figure runs the arithmetic: your risk budget divided by the expected move gives the position size at which a 1σ adverse move uses up exactly that budget.

#### Worked example: sizing to a \$500 risk budget

You are willing to lose at most **\$500** on a CPI event, and the S&P prices a **±1.2%** expected move.

- Solve `position × 0.012 = \$500` for the position size.
- `position = \$500 / 0.012 = \$41,667`, call it **≈\$41,000**.
- At that size, a 1σ adverse move (−1.2%) costs `\$41,000 × 0.012 = \$492` — about your full budget.
- A 2σ move (−2.4%) costs `\$41,000 × 0.024 = \$984`; the September 2022 −4.32% would cost `\$41,000 × 0.0432 = \$1,771`.

Intuition: sizing to the 1σ band caps your *typical* loss at the budget, but because the band is breached a third of the time you must add a hard stop for the days the print blows through the cone.

That last line is the whole risk-management discipline. Sizing to the expected move is necessary but not sufficient, because the expected move is a 1σ band and 1σ bands get breached. The September 2022 print would have cost 3.5× the budget at the "correct" size. So the expected move sets your *base* size, and a **hard stop** (or a defined-risk options structure, where the most you can lose is the premium) caps the tail. The two work together: the band tells you how big a normal adverse day is; the stop tells you what to do on the abnormal one.

The expected move also tells you where to place a stop *intelligently* rather than arbitrarily. A stop set inside the expected move — say at −0.5% when the band is ±1.2% — is almost guaranteed to be hit by ordinary event noise, stopping you out of a position that has not actually been proven wrong. A stop set just *outside* the band — at, say, −1.5% — only triggers when the move is genuinely larger than priced, i.e. when the event really did surprise. **Place stops relative to the expected move, not to round numbers.** A trader who fades the knee-jerk reaction, expecting the move to stay inside the cone, puts the stop just outside the band; a trader riding a breakout puts the entry trigger at the band's edge.

The expected move is also the natural input to the consensus framework: if a market has priced a tight expected move into an event, it is telling you the outcome feels well-anticipated; a wide expected move says the market is braced for a surprise. We connect the band to the broader question of what is "priced in" in [consensus, expectations, and 'priced in'](/blog/trading/event-trading/consensus-expectations-and-priced-in).

One number you already know is itself a giant expected move: the **VIX**. The VIX is the 30-day implied volatility of the S&P 500, quoted as an annualized percentage. Divide it by `sqrt(12)` (the square-root-of-time scaling from a year to a month) and you have the market's one-month expected move for the whole index. A VIX of 16 implies a one-month S&P expected move of about `16 / sqrt(12) ≈ 4.6%`; divide instead by `sqrt(252)` for the implied *daily* move, about `16 / 15.9 ≈ 1.0%`. So when the VIX sits at 16, the options market is pricing a typical S&P day of about ±1%, and a CPI day priced at ±1.2% is being marked as slightly bigger than a normal session. The VIX spiking to an intraday 65.73 in August 2024 was the market pricing a one-day expected move of `65.73 / 15.9 ≈ 4.1%` for the *entire index* — an extraordinary cone that, on a **\$100,000** portfolio, meant a 1σ daily swing of `\$100,000 × 0.041 = \$4,100`. **The VIX is the S&P's expected move with the units rearranged; learning to convert it to a daily band makes the "fear gauge" a sizing tool rather than a vibe.**

## Asset differences: why crypto's cone is several times wider

We touched this above; it deserves its own treatment because it is the most common place a beginner misreads a reaction.

The expected move is `S × IV × √(t/365)`, and across assets the only term that changes meaningfully is **IV**. Over the same calendar window the time factor is identical for everything, and spot is just a scaling unit. So the *percentage* expected move of an asset is essentially its implied volatility times the time factor — and implied volatilities differ enormously across asset classes:

- **Major equity indices (S&P 500, Nasdaq):** IV typically runs in the teens to low twenties in calm periods, spiking to 30–80 in crises (the VIX, which is the 30-day S&P implied vol, averaged about 19.5 over the long run and spiked to an intraday 65.73 in the August 2024 cascade). A one-day event prices roughly ±1% to ±1.5%.
- **Single stocks around earnings:** IV is far higher than the index because idiosyncratic risk does not diversify away; expected moves of ±5% to ±10% into an earnings print are routine, and high-beta names price ±15% or more.
- **Bitcoin and major crypto:** annualized IV frequently runs 50–80% even in normal conditions, so a single-event expected move of ±4% is ordinary and catalyst events price far wider. Crypto trades 24/7 with thinner liquidity and higher leverage, which keeps realized vol — and therefore implied vol — structurally elevated.
- **FX (major pairs):** the lowest-vol liquid market; EUR/USD might price a sub-0.5% expected move into a data point, because a currency pair is a ratio of two large, slow economies. (This is exactly why the August 2024 *carry unwind* was so violent — when USD/JPY broke from 161.9 to 141.7 in days, it was a multi-standard-deviation move in a market that normally prices tiny ones.)
- **Bonds (Treasury futures):** moderate vol, but the expected move is often quoted in yield terms (basis points) rather than price, and a DV01 turns the bp move into dollars.

#### Worked example: the same percent move means different things across assets

A print causes a **−3%** day. What does it mean for each asset?

- **S&P 500** (priced ±1.2%): a −3% move is **2.5σ** — a genuine shock, the kind that makes the evening news.
- **Bitcoin** (priced ±4%): a −3% move is **0.75σ** — *inside* the cone, an ordinary session, barely a headline.
- On a **\$20,000** S&P position, −3% = **−\$600**; on a **\$20,000** Bitcoin position, the same −3% = **−\$600** in dollars but a far smaller *surprise* in vol-adjusted terms.

Intuition: the raw percentage is meaningless across assets — only the move measured in units of each asset's expected move tells you how surprised the market should be.

This is the single most useful discipline the expected move teaches: **stop comparing raw percentage moves across assets and start comparing them to each asset's own cone.** A trader who internalizes this stops being impressed by a 5% Bitcoin day and starts being impressed by a 2% S&P day, because the second one is the bigger surprise. The cross-asset reaction always reads cleaner in expected-move units; the raw bars mislead. For the mechanism of how one print propagates across all these cones at once, see the companion post on cross-asset transmission within this series.

### When there is no liquid options market: estimating the cone from history

Most markets a retail trader touches outside the U.S. — Vietnam's VN-Index, many emerging-market indices, smaller single names — have no deep, listed options chain to read an expected move off. You can still build a cone, just from realized volatility instead of implied. Take the asset's recent daily moves, compute their standard deviation, and that one-day realized vol is a usable proxy for the one-day expected move. It is backward-looking rather than forward-looking, so it misses the event premium that implied vol captures, but it gives you a defensible band to size and stop against.

#### Worked example: a realized-vol cone for the VN-Index

Suppose the VN-Index has been moving with a daily standard deviation of about **1.1%** (typical for a calmer stretch) and you hold the equivalent of **\$20,000** of a VN-Index ETF into a State Bank of Vietnam rate decision.

- One-day expected move ≈ the daily realized vol = **±1.1%**, so a 1σ day swings `\$20,000 × 0.011 = \$220`.
- A surprise SBV move — like the +200bp of refinancing-rate hikes in autumn 2022 that helped drag the VN-Index from a ~1,530 peak to a 911 trough — produced index days well beyond ±1.1%, the same "blow through the cone" pattern as a hot CPI.
- Sizing to a **\$300** budget: `\$300 / 0.011 ≈ \$27,000` position, with a hard stop just outside the ±1.1% band.

Intuition: even with no options to read, the standard deviation of recent daily moves gives you a working cone — just remember it cannot see a surprise coming the way an implied cone tries to. The mechanics of SBV policy and the dong are covered in the macro-trading and finance series this post cross-links to.

The same realized-vol substitute is how you sanity-check an implied cone that looks suspicious: if the options are pricing ±0.4% into a CPI print but the asset has realized ±1.0% on recent print days, the implied cone is too cheap and the buyer has an edge. Comparing the implied cone to the realized one is, in miniature, the entire priced-versus-realized trade.

## How it reacted: real episodes

Now put the framework against dated tape. The point of every episode below is the same: the expected move drew a cone, and the print either stayed inside it (and faded) or blew through it (and trended).

**September 13 2022 — the hot CPI that broke the cone.** Going in, the S&P straddle priced roughly ±1.2%. August CPI printed 8.3% against 8.1% expected, with core inflation *rising* to 6.3% when the consensus had it easing — a clear hot surprise in the worst possible regime, where hot inflation meant more Fed tightening and "good news is bad news." The S&P closed **−4.32%**, the Nasdaq **−5.16%**, the Dow **−3.94%**; the 2-year Treasury yield jumped 18bp and the dollar gained 1.4%. Bitcoin fell about **−9.4%** on the day. Every one of those moves was a multiple of its priced cone.

![Cross-asset realized move versus the priced band on the hot Aug-2022 CPI](/imgs/blogs/the-expected-move-pricing-event-risk-with-options-7.png)

The figure lays the cross-asset reaction against the illustrative ±1.2% S&P band. Equities, the dollar, and Bitcoin all sit outside the band — most assets exceeded what the S&P cone priced. The dashed lines mark the cone; almost nothing stayed inside it. This is what a *regime-amplified surprise* looks like: a moderate headline miss (+0.2pp) produced an outsized cross-asset move because the market was fragile and one-directionally positioned.

#### Worked example: the dollar cost of the September 2022 cross-asset move

Run a small multi-asset book through that session.

- **\$25,000** in the S&P: `−4.32% × \$25,000 = −\$1,080`.
- **\$10,000** in Bitcoin: `−9.4% × \$10,000 = −\$940`.
- A **\$1,000,000** 2-year Treasury position, +18bp move, DV01 ≈ \$190 per bp: `−18 × \$190 ≈ −\$3,420` (yields up, price down).
- Against priced ±1.2% S&P, the realized −4.32% was a **3.6σ** event in equities alone.

Intuition: when a print clears every asset's cone at once, a "diversified" book offers no shelter — the surprise hits everything in the same direction.

**November 10 2022 — the cool CPI that broke the cone the other way.** Two months later, October CPI came in *cool* (7.7% versus 7.9% expected). With the market braced for more bad inflation news and positioned short, the relief move was enormous: the S&P **+5.54%**, the Nasdaq **+7.35%**, the 10-year yield down 28bp, the dollar down 2.1%. Again the realized move — more than 4× the priced ±1.2% — sat far outside the cone, this time on the upside. Same lesson, opposite sign: a surprise in a fragile, crowded regime produces a tail move, not a typical one.

**November 14 2023 — the cool CPI that mostly respected the cone.** A year on, October 2023 CPI printed mildly cool (3.2% versus 3.3%). The regime had calmed — inflation was clearly trending down and the Fed was near done. The S&P rose **+1.91%**, only modestly past a ±1.2% cone, while the rate-sensitive Russell 2000 ran **+5.44%** because small caps carry the most floating-rate debt and benefit most from falling rates. The lesson here is about *where* the move concentrates: the index stayed near its cone, but the most rate-sensitive corner blew through its own (wider) one. **The expected move is asset-specific; a print can be a non-event for the index and a huge event for a sub-segment.**

**The 2018 and 2024 Fed episodes — when the priced move and the realized move agreed.** Not every event breaks its cone. On December 19 2018 a hawkish Fed hike produced a −1.54% S&P day; on March 16 2022 the first hike of the tightening cycle produced a +2.24% relief rally; on September 18 2024 the first −50bp cut produced a muted −0.29% on the day. These are the sessions where the realized move sat near the priced cone — the event resolved roughly as the options expected, the vol crushed, and the straddle buyer was the one who paid. **Most events respect the cone; the memorable ones are the minority that don't.** That asymmetry — frequent small payoffs to the vol seller, rare large payoffs to the vol buyer — is the entire economics of event volatility, and it is why the expected move (which sizes the tail) matters more than the average outcome (which flatters the seller). The mechanism of *why* the same number moves differently across these regimes is the subject of the reaction-function post in this series.

## Common misconceptions

**"The expected move is the market's forecast of where the asset will go."** No. The expected move has no direction — it is a symmetric band around spot, equally up and down. The straddle that defines it is a pure bet on *magnitude*, not sign. If the options had a directional view, it would show up as the *skew* (puts priced richer than calls, or vice versa), not in the expected move itself. The expected move says "about this big," never "this way."

**"The expected move is a cap — the asset won't move more than that."** Emphatically no, and this is the most expensive misconception. The expected move is a *1σ* band, which in a normal distribution is breached about **32% of the time** — roughly one event in three. Two of the three CPI episodes above breached it, by 3–5×. A trader who treats the cone as a ceiling and sizes or sells accordingly is short the tail and will eventually meet a September 13 2022. The band is the *typical* move, with a fat tail past it.

**"If I buy the straddle and I'm right about direction, I make money."** Not necessarily — you have to be right about *magnitude*. As the worked example showed, a 1.0% move when the cone is 1.2% loses the straddle buyer \$10 even though the direction was correct, because the move did not clear the premium and the vol crush bled the unused leg. **The straddle buyer is long magnitude, not long direction.** Being right about which way it goes is worthless if it does not go far enough.

**"A bigger expected move means a riskier or worse outcome."** No — it means more *uncertainty*, not more *danger to a given direction*. A wide cone before an event simply means the options market is unsure and the print could land far in either direction. It is the *seller* of a wide cone who is taking the most risk, not the asset. And a wide cone is information: it tells you the market itself does not know, which is often the right reason to size down or stay defined-risk rather than to "expect the worst."

**"IV and the expected move are the same thing."** Closely related, not identical. IV is an annualized rate; the expected move is that rate scaled down to the specific window by the `√(t/365)` factor and multiplied by spot. The same 18% IV implies a tiny expected move over one day and a large one over a year. **IV is the speed; the expected move is the distance you'd travel in *this* much time.**

## The playbook: how to trade the expected move

Here is the if-then map an event trader runs, with the expected move as the central instrument.

**Before the event — read the cone.**
- Pull the ATM straddle for the expiry that captures the event; divide by spot to get the expected move in percent. Cross-check with `S × IV × √(t/365)`.
- Compare the cone to the asset's *normal* daily range. A cone well above the daily average says the event is a big deal and the market knows it; a cone below average says it is mostly priced in.
- Compare the cone across the assets you trade — the S&P's ±1.2% against Bitcoin's ±4% — and decide which asset offers the cleanest expression of your view in expected-move units.

**Form a view — but on the right variable.**
- If you think the realized move will be *larger* than the cone (a fragile regime, crowded positioning, a binary catalyst), you are a **buyer of vol**: long the straddle, defined risk, needing the print to clear a break-even. Your worst case is the premium; your best case is the overshoot.
- If you think the realized move will be *smaller* than the cone (a well-anticipated print, calm regime, no fresh information likely), you are a **seller of vol**: short the straddle or a defined-risk structure, collecting premium and the vol crush. Your best case is the premium; your worst case is the tail — so this is the trade that demands a hard stop and humility.
- If you have a *directional* view, the cone still sets your sizing and stops even though you are not trading the straddle.

**Size and stop to the cone.**
- Base size: solve `position × expected_move = risk_budget` so a 1σ adverse move equals your budget (the \$500 → \$41,000 example).
- Hard stop: just *outside* the cone, so ordinary event noise does not trigger it but a genuine surprise does. Never inside the cone.
- Remember the cone is breached a third of the time — keep the tail defined (a stop, or buying the wing) because the breach is when you lose multiples of the budget.

**After the print — read priced versus realized.**
- Move *inside* the cone → the surprise was small, vol crushes, the knee-jerk often fades back toward the pre-event level; vol sellers win and the fade is the trade.
- Move *outside* the cone → a genuine surprise; vol buyers win, and the move frequently *trends* as positioning unwinds (the September and November 2022 CPIs both trended hard the same session). The break of the cone is itself a signal to ride, not fade.

**The invalidation.** Your view is wrong when the realized move lands on the *opposite* side of your thesis: if you sold the cone and the print blows through it, your loss is uncapped and the stop must fire; if you bought the cone and the print lands dead inside it, the vol crush takes your premium. In both cases the cone you measured before the event is what tells you, in real time, whether you were right — the realized move either cleared the band or it didn't.

The expected move is, in the end, the one number that turns a vague "this print could be big" into a measured bar: *this* big, ±1.2% on the S&P, ±4% on Bitcoin, breached a third of the time, with the tail uncapped. Read it before every event, size to it, stop outside it, and grade every print by whether it cleared it. It will not tell you which way the market goes — nothing does — but it will tell you, with more precision than any forecast, how big a move you are being paid to risk, and how big a move the print has to make to be news.

## Further reading & cross-links

- [Options theory](/blog/trading/quantitative-finance/options-theory) — the mechanics of calls, puts, premium, and payoff that the straddle and the expected move are built from.
- [The volatility surface](/blog/trading/quantitative-finance/volatility-surface) — how implied volatility varies by strike and expiry (skew and term structure), which refines the simple ATM expected move into the full picture.
- [Consensus, expectations, and 'priced in'](/blog/trading/event-trading/consensus-expectations-and-priced-in) — how the market builds the baseline the expected move sits around, and why only the surprise versus that baseline moves price.
- [Crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset) — why Bitcoin's implied volatility, and therefore its expected-move cone, runs several times wider than an equity index's.
- A companion post in this series on **implied versus realized volatility and the vol crush** goes deeper on what happens to option prices *after* the print resolves the uncertainty — the mechanism that decides whether the straddle buyer or seller wins.
