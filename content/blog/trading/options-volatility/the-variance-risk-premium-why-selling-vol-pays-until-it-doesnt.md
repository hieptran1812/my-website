---
title: "The Variance Risk Premium: Why Selling Vol Pays — Until It Doesn't"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why option sellers have a structural edge, the exact shape of its tail, and how to size so the inevitable bad month is a dent and not a death."
tags: ["options", "volatility", "variance-risk-premium", "short-vol", "implied-volatility", "realized-volatility", "tail-risk", "position-sizing", "straddles", "risk-management"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Implied volatility is persistently priced *above* the realized volatility that follows, so the seller of options collects a structural premium — but that premium is the fee for warehousing left-tail risk, not a free lunch.
>
> - The variance risk premium (VRP) is the gap between what the market *charges* for future volatility (implied) and what *actually happens* (realized). Long-run, S&P 500 implied averages about 19.5 vol points against ~15.8 realized — a **+3.7 vol-point** edge that gets much larger when you square it into variance terms.
> - The payoff of harvesting it is brutally lopsided: **positive carry, negative skew, negative convexity**. You win small most months and lose big rarely. It is picking up pennies in front of a steamroller.
> - The headline Sharpe ratio of short-vol is a mirage — it is high *because the worst day hasn't printed yet*. The peso problem hides the tail.
> - The one rule to remember: **size to the crash, not to the carry.** The whole game is making the inevitable bad month a dent, not a death.

A short-vol fund had compounded steadily for years. Every month the same trade: sell a basket of S&P 500 options, collect the premium, watch most of it decay to zero, book the difference. The equity curve was a near-straight line up and to the right, the kind a marketing deck loves. The reported Sharpe ratio was north of 2. Investors saw a smooth, high-return, low-drawdown product and piled in.

On February 5, 2018, the VIX — the market's headline 30-day implied-volatility index — closed at **37.32**, having roughly doubled in a single session from the high teens. Volatility-tracking products that were *short* that move detonated. One exchange-traded note built to profit when volatility fell lost more than 90% of its value overnight and was wound down within days. The trade that had quietly paid pennies for years gave back its decade of gains in an afternoon. Traders later named the episode "Volmageddon."

This is the defining tension of option selling, and the subject of this post. There genuinely *is* a structural edge to selling volatility — it is one of the most robustly documented risk premia in all of finance, and we will measure it precisely. But the edge is shaped exactly like an insurance business: a steady stream of small premiums in the good times, punctuated by the rare, enormous claim that defines whether you survive at all. Understanding the variance risk premium means understanding *both halves* — why the premium exists, and the exact geometry of the catastrophe waiting at the left tail.

![Implied vol versus the realized vol that follows, with the average gap shaded as the seller's edge](/imgs/blogs/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt-1.png)

The chart above is the entire trade in one picture. The blue line is implied volatility — what the option market charged for the next month of movement, year by year, proxied by the annual average of the VIX. The green line is the volatility that actually showed up afterward. The amber band between them is the variance risk premium: the seller's structural edge. Notice two things. First, blue sits above green *almost everywhere* — that persistence is the edge. Second, the gap is far from constant: it widens in calm years and *inverts* (green pokes above blue) around the crisis years like 2008. That second feature is the steamroller.

## Foundations: implied, realized, and the gap between them

Before we can trade the premium, we need to be precise about three words that get used loosely: volatility, implied, and realized. If you have read [the trade at the heart of options](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options), this is the same machinery viewed from the seller's chair; if not, we build it from zero here.

**Volatility** is just the size of an asset's typical move, expressed as an annualized standard deviation of returns. If a stock has a 20% annualized volatility, the everyday meaning is that a one-standard-deviation move over a year is about 20% — and, because volatility scales with the square root of time, the one-standard-deviation move over a *month* is roughly `20% × √(1/12) ≈ 5.8%`. Volatility says nothing about direction. A 20% vol stock is equally likely (in the simplest model) to be 5.8% higher or 5.8% lower a month from now. It is a measure of *how much*, not *which way*.

**Realized volatility** (also called historical or actual volatility) is what you compute *after the fact* from the price path that actually happened. Take a month of daily returns, compute their standard deviation, annualize it by multiplying by `√252` (the number of trading days in a year), and you have the realized vol for that month. It is a backward-looking fact. There is no opinion in it.

The mechanics matter for the trade, so let us be exact. To compute one month of realized volatility you take the daily log-returns — `r_t = ln(P_t / P_{t-1})` — over the 21 or so trading days in the month, compute their standard deviation, and scale up:

```
realized vol (annualized)  =  stdev(daily log-returns) × √252
```

The `√252` is the time-scaling rule and it is worth internalizing, because it governs the entire economics of the trade. Variance is additive over time — the variance of a two-day return is twice the variance of a one-day return, *assuming returns are independent*. Volatility is the square root of variance, so volatility scales with the *square root* of time, not linearly. That is why a 16% annual vol corresponds to a daily move of only about `16% / √252 ≈ 1%`, and a monthly move of `16% × √(21/252) ≈ 4.6%`. The square-root scaling is also why a single explosive day can dominate a whole month's realized vol: one 8% day contributes `8%²= 64` to the variance sum, which can be larger than the contribution of all twenty calm days combined. The realized number is not a smooth average; it is hostage to its worst day. That asymmetry — calm accumulates slowly, a single gap spikes the whole measure — is the same asymmetry that defines the short seller's risk.

**Implied volatility** is the opposite: it is forward-looking and it is an *opinion the market is charging you for*. When you price an option with the Black-Scholes model, volatility is the one input you cannot observe directly — strike, spot, time, and rates are all known. So the market does the reverse: it observes the *price* at which options are trading and backs out the volatility number that would justify that price. That number is the implied volatility. (For the derivation of the pricing model itself, see [Black-Scholes](/blog/trading/quantitative-finance/black-scholes); here we treat the model as a given and focus on the trade.) Implied vol is the market's collective forecast of future realized volatility — *plus a markup*. That markup is the whole story.

The **variance risk premium** is the gap between the two:

```
VRP  =  implied volatility  −  subsequently realized volatility
```

Measured over decades of S&P 500 data, this gap is reliably positive. Using the curated long-run figures from the academic literature (Carr and Wu, 2009; Bollerslev, Tauchen and Zhou, 2009):

- Average implied volatility: **19.5 vol points**
- Average subsequently realized volatility: **15.8 vol points**
- The gap: **+3.7 vol points**

In plain terms, the option market on average charges you for 19.5% of annualized movement and then only 15.8% shows up. The seller pockets the difference. A buyer of that insurance overpays by about 3.7 vol points per year, on average, for the comfort of holding protection.

> [!note]
> **Why "variance" risk premium and not "volatility" risk premium?** The cleanest version of the trade — the variance swap — pays off in *variance* (volatility squared), not volatility. And the academic literature defines the premium in variance terms because variance is additive over time and tractable in the math. We will see in fig 5 that squaring makes the seemingly modest 3.7-point vol gap look much larger. People use "VRP" loosely for both; just know that the cleaner, larger number lives in variance space.

### A first taste of the trade

If implied vol is systematically too high relative to what realizes, the trade writes itself: **sell options, hedge the direction, and pocket the difference between what you were paid (implied) and what it cost you to hold the position (realized).** That is the core of every short-vol strategy — a short straddle, an iron condor, a covered call, a variance swap. They differ in their plumbing, but they are all the same bet: *implied is too rich; I will sell it and collect the gap.*

The reason this is worth a whole post — rather than "just sell options, free money" — is the second half. The premium is not a market inefficiency that a clever quant noticed and arbitraged away. It is *compensation for a real and nasty risk*, and the people paying it are not fools. They are buying insurance, and you, the seller, are the insurance company. Insurance companies are profitable on average and occasionally bankrupt. Everything that follows is about living on the profitable side of that distribution.

## Why the premium exists: you are selling crash insurance

A risk premium that survives for decades, across regimes and markets, is not a free lunch. If it were, capital would flood in and arbitrage it to zero. The VRP persists *because there is a risk attached to harvesting it that most investors do not want to bear* — and so they pay someone else to bear it. There are three reinforcing reasons, and one crucial caveat.

![Insurance demand, crash aversion, and supply imbalance feeding into buyers overpaying and sellers earning the gap, with the crash-risk caveat](/imgs/blogs/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt-3.png)

**1. Insurance demand.** A pension fund, an endowment, a corporate treasury — these are large pools of capital with a mandate to *not blow up*. They hold equities for the long-run return, but a 40% drawdown could breach a funding ratio, trigger a covenant, or force selling at the bottom. So they buy protection: index puts, put spreads, collars. They are structurally, perpetually *long* downside insurance, and they are not especially price-sensitive about it, because the protection is doing a job (sleeping at night, meeting a regulatory test) that is worth more to them than the few vol points of overpayment. That standing demand bids up the price of options — especially downside options.

**2. Crash aversion.** This is the behavioral layer, and it is real even among sophisticated investors. Losses hurt more than equivalent gains feel good — the asymmetry that prospect theory put a number on. A 20% loss is not psychologically offset by a 20% gain; it stings far more. Because crashes are precisely the events that deliver large, sudden losses, investors *over-weight* the probability and severity of crashes when they price protection. This shows up directly in the options market as **skew**: out-of-the-money puts trade at much higher implied vol than out-of-the-money calls, because the demand for downside protection is so much fiercer than the demand for upside speculation. We dissect that asymmetry in [the volatility smile and skew](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more); the punchline for us is that skew is the fingerprint of crash aversion, and it is one of the richest parts of the premium.

**3. Supply imbalance.** For the demand to be satisfied, someone has to be *short* the tail — short the puts, short the variance, on the hook for the claim. That is an uncomfortable position. It has unlimited or near-unlimited downside (a put can pay off enormously), it can blow through your capital in a single gap, and it requires both balance sheet and stomach. Relatively few participants want that job. When demand for protection is high and the supply of willing protection-sellers is scarce, the price of protection rises until it is high enough to *compensate* the sellers for taking it on. That compensation is the premium. It is the same economics as a reinsurer who writes hurricane policies: the price is set where it has to be for someone to be willing to stand behind the catastrophe.

**The leverage effect** is the structural reason the three forces above concentrate on the *downside* rather than spreading evenly. When a company's stock falls, the market value of its equity shrinks while its debt stays roughly fixed — so the firm becomes *more leveraged*, and a more leveraged firm has riskier equity, which means higher volatility. Falling prices and rising volatility are therefore mechanically linked: down moves come *with* a vol spike, while up moves tend to be calmer. This is why equity-index volatility has a persistent *negative* correlation with returns — the market falls and the VIX jumps in the same session, almost every time. For the option seller, the leverage effect is the reason the tail is one-sided: your worst losses do not arrive on big *up* days (those are calm and the realized vol stays low), they arrive on big *down* days, when the move and the vol spike compound. The premium is fat on downside options precisely because the leverage effect guarantees that downside is where the volatility — and the seller's pain — concentrates. It is also why a short-vol book is, in effect, a leveraged *long* position on the market dressed up as a volatility trade: when stocks crash, you lose, and you lose more than the crash because your short gamma is amplifying it.

#### Worked example: pricing the insurance you are selling

Make it concrete. Suppose a stock trades at \$100, one-month options are priced at the long-run *implied* vol of 19.5%, the risk-free rate is 4%, and there are no dividends. Using the Black-Scholes pricer, a one-month at-the-money straddle (long one call + one put at the \$100 strike) costs:

- Call: **\$2.41** per share
- Put: **\$2.08** per share
- Straddle premium: **\$4.49** per share

If you *sell* that straddle, you receive \$4.49 up front. That \$4.49 is the market's price for one month of "I'll cover your move in either direction." It is priced as if the stock will move about \$5.63 in a one-standard-deviation month (`\$100 × 19.5% × √(1/12) ≈ \$5.63`). But the *realized* one-standard-deviation move, on the historical average, is only about \$4.56 (`\$100 × 15.8% × √(1/12)`). You were paid for a \$5.63 world and you live in a \$4.56 world. **The intuition: you sold the insurance at a 19.5% premium rate against a 15.8% actual loss rate — that 3.7-point wedge is your underwriting profit, and it is exactly why the trade has positive expected value.**

The caveat box in the figure is doing the most important work, though. The premium is *compensation for bearing the left tail*. In a crash, the stock does not move \$4.56 — it gaps \$20, the put you sold goes deep in the money, and you pay the insurance claim. The VRP is positive on average *precisely because* the seller occasionally takes that loss. Remove the loss and the premium would not exist; the buyers would have no reason to pay it. **The premium and the tail are two sides of one coin — you cannot collect one without owning the other.**

## The payoff shape: positive carry, negative skew, negative convexity

If you take only one mental model from this post, take this one: short volatility is *long carry and short convexity*. Those four words contain the whole risk profile. Let us unpack each.

**Positive carry** means the position makes money simply by the passage of time when nothing dramatic happens. You sold options; options decay; you keep the decay. The technical name for that decay is **theta**, and a short-option position has positive theta — every day that passes with the stock sitting still, you bank a little of the premium you collected. (We treat theta as the rent the option buyer pays in [theta: trading the clock](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options); as a seller, you are the landlord.) Positive carry is *why the equity curve grinds upward in quiet times.*

**Negative skew** describes the *shape* of the return distribution. A short-vol P&L distribution is not symmetric. It has a hard ceiling — the most you can make is the premium you collected — and a long, heavy left tail of large losses. The mean might be positive, but the distribution leans hard to the downside.

**Negative convexity** is the dynamic, second-order version of the same thing, and it is the most dangerous. A short-option position is **short gamma**: as the underlying moves *against* you, your losses accelerate — they grow faster than linearly. We dig into the mechanics in [gamma: the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short); the short version is that being short gamma means every additional point the market moves against you hurts *more* than the last one. Your delta-hedging has to chase the move, buying high and selling low, locking in losses exactly when you can least afford them. Negative convexity is the engine that turns a bad day into a catastrophic one.

Make the convexity concrete. Suppose you are short a straddle and delta-neutral at \$100. The stock falls to \$95; your short gamma means you are now *long* delta (you have to sell stock to re-hedge), so you sell at \$95. The stock falls again to \$90; you sell more at \$90. Then it rebounds to \$95 and you buy back what you sold — at a higher price than you sold it. Every round trip of the hedge *locks in* a loss: you sold low and bought high, the exact opposite of the long-gamma trader who is selling high and buying low. The faster and larger the swings, the more these forced round trips cost you, and they cost you most precisely on the violent days when the stock is gapping. That is what "realized vol greater than implied" actually *feels* like at the position level: a string of money-losing hedges that, summed up, exceed the premium you were paid. The premium was the budget; realized volatility is the bill; and on a bad enough day the bill is a multiple of the budget.

![Histogram of short straddle profit and loss showing many small wins and rare large losses, a negative skew distribution](/imgs/blogs/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt-2.png)

The histogram above is the short straddle from our worked example, simulated over 60,000 monthly outcomes. The premium received is \$4.49, so that is the hard ceiling — the blue line on the right, the best you can ever do. Most outcomes cluster just below it in green: small wins, month after month, because the realized move is usually smaller than the \$5.63 the premium was priced for. But the left side is a long red tail. The rare large adverse move blows straight through the premium and into deep loss territory. The distribution is *exactly the shape of an insurance book*: a tall stack of small profits and a thin, far-reaching tail of large claims. The arithmetic mean of this distribution is positive — that is the VRP — but the *median* outcome and the *modal* outcome are both small wins, which is precisely what fools people into thinking the trade is safer than it is.

#### Worked example: the short straddle's break-even versus its edge

Stay with the \$4.49 straddle. When you sell a straddle, you keep the full premium only if the stock pins exactly at the \$100 strike at expiry. The position loses money once the stock moves more than the premium in either direction:

- Lower break-even: `\$100 − \$4.49 = \$95.51`
- Upper break-even: `\$100 + \$4.49 = \$104.49`

So you make money as long as the stock finishes between \$95.51 and \$104.49 — a band of `±4.49%` over the month. Now compare that band to the move you actually expect. The *implied* one-sigma move was \$5.63, which is *wider* than your break-even band — at first glance that looks alarming, like you are likely to lose. But you were paid at the implied vol, and the world tends to realize the *lower* 15.8% vol, an expected one-sigma move of only \$4.56. Run the expectation properly: under the realized-vol distribution, the *expected* intrinsic value you have to pay out at expiry is about \$3.64, against the \$4.49 you collected. Your expected profit per month is:

```
expected P&L  =  premium received  −  expected payout
              =  \$4.49  −  \$3.64
              =  +\$0.85  per share per month
```

**The intuition: the edge is real but thin — about 85 cents on a \$100 stock per month — and it is an *average* over a distribution whose left tail can cost you \$15 or \$20 in a single bad month.** You are not collecting \$4.49 of profit; you are collecting \$4.49 of premium against \$3.64 of expected claims, and the 85-cent residual is the variance risk premium showing up in dollars. That thinness is why sizing — covered later — is everything.

### Why the modest vol gap is a bigger trade than it looks

The 3.7 vol-point gap can feel underwhelming. Implied 19.5, realized 15.8 — so what? The "so what" is that the trade does not pay in vol points; it pays in *variance*, and squaring a number magnifies a gap.

![Two bar panels comparing implied and realized in vol points and in variance, with the gap labeled in each unit](/imgs/blogs/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt-5.png)

The left panel is the gap as you quote it: 19.5 versus 15.8, a tidy +3.7. The right panel is the same two numbers *squared* into variance — the unit a variance swap and the deep math actually settle in. `19.5² = 380` against `15.8² = 250`, a gap of **+131 variance points**. The proportional edge is much larger in variance space: the implied is `380 / 250 ≈ 1.5×` the realized variance, a 50% markup, versus the `19.5 / 15.8 ≈ 1.23×` it looked like in vol points. This is not a trick — it is the honest reason the premium is "worth" more than the headline vol gap suggests, and the reason the cleanest expression of the trade (the variance swap) is denominated in variance. When a vol seller says "implied is 25% over realized," they usually mean in *variance* terms, and they are right.

#### Worked example: the VRP in vol points versus variance terms

Suppose you sell a one-year variance swap on an index at an implied (strike) volatility of 19.5%, on a notional of \$10,000 per variance point (a common way to quote them). The swap pays you the difference between the strike *variance* and the realized *variance*:

```
strike variance       =  19.5²  =  380.25  variance points
realized variance     =  15.8²  =  249.64  variance points
gap                   =  380.25 − 249.64  =  130.61 variance points
payout to seller      =  130.61 × \$10,000  =  \$1,306,100
```

Versus the naive vol-point framing, where you might have expected `3.7 × \$10,000 = \$37,000` — off by a factor of roughly 35. **The intuition: because the trade settles in variance, the convex squaring both *enlarges your edge in the average case* and *enlarges your loss in the tail* — a crisis that takes realized vol to 60% delivers a realized variance of 3,600 against your strike of 380, a loss of 3,220 variance points, which is why even a "small" short-variance position can vaporize a fund.** Convexity cuts both ways; in variance space it cuts deeply.

## Picking up pennies in front of a steamroller

The phrase is old trading-desk folklore and it is exactly right. Selling volatility is a strategy of small, frequent wins and rare, enormous losses. The pennies are real — you genuinely pick them up, month after month, for years. The steamroller is also real, and it does not care how many pennies you have collected.

![Cumulative short vol equity curve rising steadily for years then falling off a cliff at a volatility shock](/imgs/blogs/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt-4.png)

The equity curve above is the whole strategy compressed into one path. For roughly six years the account grinds upward — the carry, the pennies, the boring monthly harvest of the VRP. The line is smooth and the slope is consistent, which is exactly what makes it seductive. Then a single tail month — a Volmageddon-style VIX spike — takes the account from \$236 back to \$90 *in days*. Years of patient accumulation are gone in an afternoon, and the account is below where a more cautious strategy would have started. This is not an exaggeration of the shape; it is a stylization of what actually happened to several short-vol vehicles in February 2018 and again, in milder form, in August 2024.

The reason the cliff is so much steeper than the climb is the negative convexity we met earlier. On the way up, you are collecting linear theta — a roughly constant trickle. On the way down, you are short gamma, and your losses compound on themselves: the bigger the move, the faster the next dollar of loss arrives. The climb is arithmetic; the fall is geometric. That asymmetry between how you make money and how you lose it is the signature of every short-convexity trade, from selling earthquake insurance to selling out-of-the-money index puts.

Let us anchor the steamroller in real numbers. These are notable VIX *closing* spikes, the days the steamroller ran:

- **2008-11-20 — 80.86** (Global Financial Crisis peak)
- **2010-05-20 — 45.80** (Flash Crash)
- **2011-08-08 — 48.00** (US credit-rating downgrade)
- **2015-08-24 — 40.74** (China devaluation)
- **2018-02-05 — 37.32** (Volmageddon)
- **2020-03-16 — 82.69** (COVID peak)
- **2024-08-05 — 38.57** (yen carry unwind)

Notice the spacing. These are not annual events; they cluster years apart, and in between, volatility drifts back down and the VRP quietly reasserts itself. That spacing is the trap. A short-vol strategy can run for three, five, seven years and *never* hit one of these — long enough to convince its manager, its investors, and its risk model that the tail is a theoretical curiosity. Then it prints, and the manager who sized to the calm years discovers, all at once, what the premium was compensating them for.

## Harvesting the premium: the menu of short-vol trades

There are many ways to be short volatility, and they trade off how much of the tail you keep versus how much carry you give up. Here is the menu, roughly from most-exposed to most-defended.

**Short straddles and strangles.** The purest expression: sell a call and a put (a straddle at the same strike, a strangle at different strikes) and collect both premiums. Maximum carry, maximum tail. A short straddle is short gamma and short vega with no defined risk — a large move in either direction is an unbounded (puts: large; calls: technically unlimited) loss. This is the highest-octane way to harvest the VRP and the fastest way to blow up. Strangles widen the break-even band by selling further out of the money, trading some premium for a bit more room.

**Iron condors and credit spreads.** Here you sell the straddle/strangle but *also buy* further-out options as wings, capping your maximum loss. You give up some premium to the wings, but in exchange the catastrophe is bounded — the steamroller can only run so far. This defined-risk structure is the workhorse of disciplined retail and many systematic books, and it is the subject of its own deep dive in [iron condors and credit spreads](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range). The trade-off is exact: you are buying back the far left tail at the cost of carry. Whether that is worth it depends entirely on how much you fear the tail — which, after Volmageddon, should be quite a lot.

**Put-writing (the covered put / cash-secured put).** Sell a put, hold the cash to buy the stock if assigned. This is the most "respectable" short-vol trade and is the basis of well-known systematic indices like the CBOE PutWrite (PUT) index, which mechanically sells one-month at-the-money S&P 500 puts and reinvests. Put-writing is short vol with a built-in willingness to own the underlying — your "loss" in a sell-off is that you buy the index lower than the market, which a long-term holder might *want* anyway. It is the gentlest entry into harvesting the VRP, and its long-run record is the cleanest demonstration that the premium is real.

**Variance swaps.** The institutional, over-the-counter version. A variance swap pays the difference between a strike variance (set at trade) and the realized variance over the life of the swap — exactly the VRP, packaged with no strike selection and no delta to hedge. They are the purest harvest, and also the most convex: as we computed above, a crisis that triples realized vol delivers a loss that scales with the *square* of the move. They blew several books up in 2008 for precisely that reason. (For the broader family of these instruments, see [exotic derivatives](/blog/trading/quantitative-finance/exotic-derivatives).)

**Systematic short-vol indices and products.** PUT, the various condor indices, and a generation of exchange-traded products built to short the VIX itself. These productize the harvest. They are convenient and transparent, and they are also where the *crowding* lives — when too much capital sits in the same short-vol trade, the unwind feeds on itself, which is precisely the mechanism that turned February 2018 into Volmageddon. We map that whole ecosystem in [the VIX and vol products](/blog/trading/options-volatility/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll).

#### Worked example: a year of put-writing, then a tail month

Walk a full year of the gentlest harvest — cash-secured put-writing — to see the carry-then-cliff arithmetic with real numbers. You have a \$100,000 account. Each month you sell one-month at-the-money puts on a \$100 index, collecting the implied-vol premium of about \$2.08 per share from our pricer. Size *conservatively*: sell puts covering \$20,000 of notional (200 shares), so a full assignment uses a fifth of your capital and the position is far from all-in.

For eleven quiet months the index drifts and the puts mostly expire worthless or are bought back cheap. Say you net about \$320 per month after the realized moves nibble at the premium — roughly the 85-cent-per-share edge from the straddle example, applied to your 200-share size and the put leg. Over eleven months:

```
quiet-month harvest   =  11 × \$320   =  +\$3,520
account after month 11 =  \$100,000 + \$3,520  =  \$103,520
```

A +3.5% year-to-date, smooth, boring, exactly on plan. Then month twelve is a tail month: the index gaps down 18% in a week (a 2018-scale shock). Your sold \$100-strike puts are now \$18 in the money. On 200 shares the intrinsic loss is `200 × \$18 = \$3,600`, against the \$416 premium (`200 × \$2.08`) you collected for the month, for a net month-twelve loss of about \$3,184:

```
month-12 loss         =  \$416 premium − \$3,600 intrinsic  =  −\$3,184
account after the tail =  \$103,520 − \$3,184  =  \$100,336
```

You finish the year up \$336 — essentially flat. The tail month ate eleven months of harvest, but **because you sized to a fifth of your capital, the cliff was a dent: you are still solvent, still at your starting capital, and still able to keep writing.** Now run the *same* year with a trader who sized ten times larger — 2,000 shares — to chase the carry. Their eleven-month harvest is +\$35,200, a glorious +35% that draws in more capital. Then month twelve costs them `2,000 × \$18 − 2,000 × \$2.08 = \$31,840` — and because that loss likely breaches their margin, they are forced to cover or take assignment at the worst tick, often realizing *more* than the \$31,840 as the gap extends. The intuition: identical edge, identical view, and the only difference — position size — decided whether the year was a flat dent or a near-total wipeout.

### The Sharpe illusion

Here is the most dangerous number in this whole business: the **Sharpe ratio** of a short-vol strategy, measured over a quiet period. The Sharpe ratio divides excess return by volatility of returns. For a strategy that earns a steady ~1% a month with very low month-to-month variation — exactly what short-vol looks like in calm times — the math produces an *enormous* Sharpe.

#### Worked example: the Sharpe ratio that lies

Take the quiet-period short-vol harvest from our equity curve: about +1.1% per month with a monthly standard deviation of returns of only ~0.6% during the calm. The annualized Sharpe ratio is:

```
monthly Sharpe   =  0.011 / 0.006   =  1.83
annualized       =  1.83 × √12      ≈  6.35
```

A Sharpe of 6 is *absurd*. The greatest hedge funds in history run long-run Sharpes around 1 to 2. A genuine 6 would be the discovery of the century. **The intuition: the 6 is not a measure of skill — it is a measure of how long it has been since the tail printed.** The strategy's "volatility" looks tiny because the bad outcomes simply have not happened *yet* in the sample, so the denominator of the Sharpe ratio is artificially small. The instant the tail prints, the realized monthly standard deviation explodes, the Sharpe collapses toward (or below) zero, and the "skill" evaporates. The high Sharpe was always a feature of the *measurement window*, not the strategy.

This is the **peso problem**, named for a 1970s puzzle: the Mexican peso traded at a steady forward discount for years — looking like free money to anyone shorting it — right up until a sudden devaluation wiped out everyone who had been collecting the carry. The realized data, before the devaluation, *understated* the true risk because the disaster was always in the distribution but had not yet appeared in the sample. Short-vol has the same structure. Any backtest or live record that does not contain a real crisis is *systematically overstating* the risk-adjusted return, because the worst draw from the distribution is missing. **When someone shows you a short-vol Sharpe of 4+, the correct response is not "impressive" — it is "this sample has not been through its February 2018 yet."**

## Sizing to survive: the whole game

If the edge is real but thin, and the tail is rare but ruinous, then the entire problem of harvesting the VRP collapses into one question: **how big should the position be?** Get this right and the inevitable bad month is a survivable dent; get it wrong and the inevitable bad month is the end of the strategy. There is no third option, because the bad month is not a possibility — it is a certainty whose *timing* is unknown.

![Two-column comparison of an oversized short vol book that dies on the tail month versus a budgeted book that takes a dent and survives](/imgs/blogs/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt-6.png)

The figure contrasts two books running the *exact same* trade with the *exact same* edge — they differ only in size. The oversized book chases the carry: ten times the lots, no tail budget, +4% a month in the quiet. It looks like the better strategy right up until the tail month, when the same -62% short-vol cliff hits the whole position unhedged and takes the account down 70% in days — a margin call forcing a cover at the worst possible price, which is how a paper loss becomes a permanent one. The budgeted book sizes to survive: a fraction of the lots, defined-risk legs, a deliberate cap on how much any single tail month can cost. Its quiet-month carry is a boring +1%. But when the tail hits, the loss is capped near its budget — the account is down 9%, a bad month rather than a death, and it is *still solvent and still harvesting the premium the next month*.

The asymmetry is the entire lesson. The oversized book and the budgeted book had the same edge and the same view. The only difference was the size, and the size decided whether the strategy compounds for thirty years or dies in year six. **The sizing rule, in one line: set the maximum tail loss you can tolerate first, then back out the position size from it — never size to the carry, because the carry is the most predictable and least informative thing about the trade.** We develop this into a full framework, including the risk-of-ruin math, in [position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading), and the broader sizing logic — fractional Kelly, the geometric-growth penalty of large drawdowns — lives in the [tail-risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) treatment.

There is a second sizing discipline that matters as much as the first: **managing the net Greeks of the whole book.** A single short straddle has a clean Greek profile, but a real short-vol book is dozens of positions, and what matters is their *aggregate* — the net delta, net gamma, net vega, net theta of the entire portfolio. You can be short gamma in three names and accidentally net long gamma across the book; you can think you are vega-neutral and actually carry a large vega exposure concentrated in one expiry. Building the dashboard that tells you your true aggregate exposure is the subject of [the net Greeks of a position](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard); for the short-vol seller, the single most important line on that dashboard is *net gamma*, because net gamma is the multiplier that converts a market gap into a P&L disaster.

### When the premium inverts

So far we have treated the VRP as a positive number you collect. But the gap is *not always positive* — and the months it goes negative are exactly the months the seller bleeds.

![Line of implied minus realized volatility through a crisis, positive then plunging below zero at a shock then recovering](/imgs/blogs/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt-7.png)

The chart plots implied vol *minus* realized vol over time around a volatility shock. For most of the path it hovers around the long-run +3.7 — the green region, the seller earning. Then a shock hits, and the line plunges deep into negative territory (the red region): realized volatility blows straight *past* the implied that was sold the week before. This is the inversion. You sold a month of insurance priced for an 18% world, and the market delivered a 60% week. The realized move overshot the implied, the VRP went sharply negative, and the seller paid out far more than they collected. This is not a tail you can hedge away after the fact — it is the structural cost of the business, paid in concentrated bursts.

Notice the shape *after* the shock, too: the line swings back *above* its baseline, into rich positive territory. After a crisis, options get expensive — implied vol stays elevated and bid for weeks while realized starts to calm down, so the VRP becomes unusually *fat*. This is the regime where disciplined sellers who survived the shock make their best returns, selling richly priced insurance to a still-frightened market. The cruel structure of the trade is that the fattest premiums are available exactly when most short-vol capital has just been wiped out and cannot take advantage. Surviving the inversion is the precondition for harvesting the rich premium that follows it — which loops right back to sizing.

## Common misconceptions

**"Selling volatility is free money — implied is always above realized, so just keep selling."** No. Implied is above realized *on average*, which is a statement about the mean of a violently skewed distribution. The +3.7 vol-point edge is the *average* outcome; the *modal* outcome is a small win and the *tail* outcome is a catastrophic loss. Treating the average as if it were guaranteed is exactly the error that detonated the short-vol products in 2018. The premium is not free; it is the fee for warehousing the left tail, and the bill comes due on a schedule you do not control. Free money does not pay 3.7 points a year for decades — risk premia do, and the word "risk" is load-bearing.

**"A high Sharpe ratio means the strategy is safe."** A high Sharpe ratio over a sample that contains no crisis means the *denominator is too small*, not that the strategy is safe — see the worked example where a perfectly ordinary short-vol harvest produced an annualized Sharpe of 6.35. The Sharpe ratio uses the standard deviation of returns as its risk measure, but standard deviation is the wrong risk measure for a negatively skewed distribution: it captures the small wiggles of the quiet months and completely misses the cliff. A negatively skewed strategy will *always* show an inflated Sharpe until the tail prints. The correct diagnostic for short-vol is not the Sharpe ratio — it is the maximum drawdown in a real crisis and the time it took to recover (if it ever did).

**"I can just hedge the tail away and keep the carry."** You can buy the wings — convert a naked straddle into an iron condor, buy crash puts against your short book — and you absolutely should. But every dollar you spend on tail protection comes *out of the carry*, because that protection is itself priced with the skew premium that makes downside options expensive. There is no free hedge: the same market dynamic that makes selling vol profitable makes buying tail protection costly. A fully tail-hedged short-vol book has a much smaller edge — sometimes none. The honest framing is that you are choosing *how much* tail to keep, paying for the reduction in carry, not eliminating the risk for free. A defined-risk condor keeps less tail and less carry; a naked strangle keeps all of both.

**"The premium exists because the option market is inefficient — eventually it will be arbitraged away."** The VRP is not an inefficiency; it is a *risk premium*, and risk premia do not get arbitraged away because arbitraging them means *bearing the risk*. To "arbitrage" the VRP you would have to be short the tail in size, which is the very position the premium compensates. Capital flows in during calm periods (compressing the premium), gets wiped out in crises (re-widening it), and the cycle repeats. The premium is as durable as crash aversion and insurance demand — which is to say, as durable as human loss-aversion and institutional risk mandates. It has survived every crisis on record and will survive the next one, because the next crisis is exactly what it is paid to cover.

**"Variance swaps are safer than options because there's no strike to pick and no gamma to hedge."** Variance swaps remove the strike-selection and delta-hedging headaches, but they *concentrate* the convexity rather than reduce it. As we computed, a variance swap settles in variance, so a crisis that triples volatility delivers a loss scaling with the square of the move — a 60% realized vol against a 19.5% strike is a loss of over 3,200 variance points. The clean packaging hides a more violent tail, not a gentler one. The 2008 crisis was a graveyard for short-variance positions precisely because their losses were convex in a way that surprised desks who thought they had simplified their risk.

## How it shows up in real markets

**February 2018 — Volmageddon.** The cleanest case study of everything above, and the reason this post exists. By late 2017 the VIX had spent the year at historic lows (the 2017 annual average was **11.1**, the lowest in the index's history), and an enormous amount of capital had crowded into short-vol products betting it would stay there. The carry had been extraordinary; the Sharpe ratios looked superhuman; the tail had not printed in years. On February 5, 2018, a modest equity sell-off triggered a spike in the VIX from the mid-teens to a **37.32** close. Because so many products were *mechanically* short volatility and had to buy it back as it rose, their forced covering pushed volatility higher still — a reflexive feedback loop. One inverse-VIX ETN lost over 90% of its value overnight and was liquidated. The trade that had quietly paid pennies for years was the steamroller's pavement. The whole episode is dissected in [the case study of Volmageddon 2018](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup); the lesson for us is that *crowding plus negative convexity equals a self-amplifying unwind* — the tail is not just large, it is reflexive.

**March 2020 — the COVID crash.** The largest VIX print in the data: an **82.69** close on March 16, 2020, edging past even the 2008 GFC peak of 80.86. This was the inversion of fig 7 in its most violent form. Realized volatility exploded — the S&P 500 had multiple single-day moves above 9%, each one contributing enormously to the realized-variance sum — and it blew straight past any implied vol that had been sold in the calm of January and February, when the 2019 annual VIX average had been a sleepy 15.4. Any short-vol position carried into the crash with no tail budget was destroyed in the first two weeks of the sell-off. But COVID also delivered the textbook *post-shock* regime: for months afterward, implied vol stayed elevated and richly bid while realized began to subside, so the VRP became extraordinarily fat. Sellers who survived March and had the capital to keep writing through April, May, and June harvested one of the best short-vol windows of the decade. The full cycle — inversion, then a gorged premium — played out exactly as the structure predicts, and it rewarded survival above all else.

**August 2024 — the yen carry unwind.** A milder rerun. On August 5, 2024, an unwind of the yen carry trade cascaded into global equities and the VIX spiked to a **38.57** close — briefly touching far higher intraday — before subsiding within days. Short-vol books took a sharp hit, but because the move *reverted* quickly, the books that had *survived* the gap (rather than being forced to cover at the worst tick) recovered much of the loss as implied vol normalized and the post-shock premium fattened. The contrast with 2018 is instructive: the difference between a dent and a death was often not the size of the spike but whether the position was sized to *hold through* it without a margin call. The survivors who could ride out the gap then harvested the rich post-shock VRP; the over-leveraged who were forced to cover locked in the loss at the bottom.

**The long quiet between.** It is worth emphasizing the years that do *not* make the news. Look back at the cover chart: 2013, 2014, 2017, 2019, 2021, 2023, 2024 — long stretches where implied sat comfortably above realized and the VRP harvest was steady and real. The premium genuinely paid for most of the last two decades. A short-vol strategy that was correctly sized would have compounded handsomely across all of it, *including* the 2018 and 2020 hits, precisely because the hits were dents and not deaths. The premium is not a myth and the strategy is not doomed — it is just a business that has to be run like an insurance company: collect steadily, reserve for the catastrophe, and never write more policy than your capital can cover. The VRP as a portfolio building block — treating long-volatility as an asset and short-volatility as a yield source — is explored from the allocator's seat in [volatility as an asset](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear).

## The playbook: how to harvest the VRP without becoming the steamroller's pavement

This is where mechanism becomes practice. If you are going to be short volatility — and the structural premium is a legitimate reason to be — here is the discipline that separates the survivors from the cautionary tales.

**The position.** Start defined-risk, not naked. An iron condor or a put spread, not a naked strangle. You are giving up some carry to the wings, but you are buying a hard floor under the catastrophe, and after Volmageddon the floor is worth far more than the carry you surrender. Graduate to less-defended structures only as your capital, your hedging discipline, and your scar tissue grow. The cash-secured put is the gentlest entry: you are short vol *and* willing to own the underlying lower, so your worst case is a purchase you might have wanted anyway.

**The Greek profile.** Know that you are **short vega** (you profit when implied vol falls or fails to materialize), **short gamma** (your losses accelerate as the market moves), and **long theta** (time decay is your daily income). Watch *net gamma* across the whole book above all else — it is the multiplier on every gap. Manage net vega by expiry so you are not unknowingly concentrated in a single event window. Build the dashboard from [the net Greeks of a position](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) and look at it every single day, because the aggregate exposure is what kills you, not any single leg.

**Entry.** The VRP is richest when implied vol is elevated relative to realized — sell into fear, not into calm. Selling vol when the VIX is at 11 (as in 2017) gives you a thin premium and a fat tail; selling vol when the VIX is at 30 *after* a shock, when implied stays bid while realized calms, gives you the fattest, safest version of the harvest. The post-crisis window in fig 7 — the swing back above baseline — is the prime hunting ground, *if you survived to hunt in it.* Be most aggressive when premiums are richest, which is right after everyone else has been carried out.

**Sizing.** This is the whole game, so it is the longest item. Set your maximum tolerable tail loss *first* — the worst single month you can absorb without a forced cover or a breach — and size the book so that a Volmageddon-scale move stays inside that budget. Never reverse the logic and size to the carry. Assume the tail prints next month, not in five years, and check that you survive it. If the honest answer is "I'd be down 70% and margin-called," you are too big regardless of how good the carry looks. The math of how drawdowns destroy compound growth, and how to set the budget formally, is in [position sizing and risk of ruin](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) and the [tail-risk / extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) treatment.

**The invalidation.** Your view is "implied is rich relative to what will realize." That view is *invalidated* the moment realized starts overshooting implied — the inversion in fig 7. When realized vol is printing above the implied you can sell, the premium has flipped negative and the trade has no edge; selling more is selling insurance at a loss into a fire. The discipline is to *reduce* short-vol exposure precisely when it feels most painful to do so (mid-shock), not to double down to "earn back" the loss. Doubling down into an inversion is the classic terminal mistake — it is selling more policy as the hurricane makes landfall.

**The mindset.** Run it like an insurance company, because it *is* one. Collect steady premiums, reserve explicitly for the catastrophe, diversify across underlyings and expiries, never let one event window dominate the book, and accept — build into the plan — that the bad month is coming. Your job is not to avoid it; the bad month is the price of admission to the premium. Your job is to make sure that when it arrives, it is a dent your book absorbs and recovers from, not the death that ends the strategy. Get the sizing right and the VRP is one of the most durable edges in markets. Get it wrong and you become a name in the next case study.

## Further reading & cross-links

Within this series:

- [Implied versus realized volatility: the trade at the heart of options](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) — the IV-versus-RV gap this whole trade is built on.
- [The volatility smile and skew: why OTM puts cost more](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) — crash aversion's fingerprint and the richest part of the premium.
- [Gamma: the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) — the negative convexity that turns a bad day into a catastrophe.
- [The net Greeks of a position: building your risk dashboard](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) — watching net gamma and net vega across the whole book.
- [The VIX and vol products](/blog/trading/options-volatility/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll) — the products that productized the harvest and crowded the trade.
- [Iron condors and credit spreads: selling the range](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range) — the defined-risk way to sell the premium.
- [Position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — the math of sizing so the tail is a dent.
- [Case study: Volmageddon 2018 and the short-vol blowup](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup) — the steamroller in full detail.

Going further afield:

- [Volatility as an asset: owning fear](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear) — the VRP from the allocator's seat.
- [Tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) — the math of the left tail you are selling.
- [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) and [exotic derivatives](/blog/trading/quantitative-finance/exotic-derivatives) — the pricing model and the variance swap's family tree.
