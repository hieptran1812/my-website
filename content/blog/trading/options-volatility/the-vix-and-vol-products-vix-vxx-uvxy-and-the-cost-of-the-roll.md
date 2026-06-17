---
title: "The VIX and Vol Products: VIX, VXX, UVXY, and the Cost of the Roll"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "What the VIX really is, why you cannot buy it, and how VXX, UVXY, and inverse-vol products bleed or blow up — so you use them as sharp short-term tools and never as a buy-and-hold hedge."
tags: ["options", "volatility", "vix", "vxx", "uvxy", "vix-futures", "contango", "term-structure", "tail-hedge", "volmageddon"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The VIX is a calculation, not a tradable thing; everything you can actually buy holds VIX *futures*, and in a normal upward-sloping futures curve those products lose money every single day they sit still.
>
> - **The VIX is a model-free 30-day implied volatility of the S&P 500**, computed from a whole strip of out-of-the-money SPX option prices (a variance-swap-style sum), annualized into vol points. It is a number, not an asset. There is no spot VIX to hold.
> - **Exposure comes through VIX futures and options on the VIX.** Long-vol products like VXX and VIXY roll a basket of front-month futures to a constant 30-day maturity; UVXY does the same at 1.5x; SVXY is short at -0.5x.
> - **In contango (the futures curve sloping up, the normal state ~80% of the time) the roll is a structural cost.** A long-vol fund buys the richer next-month future and watches it decay toward a lower spot every day. That is why VXX is a melting ice cube over months and years, and a terrible buy-and-hold.
> - **The one rule to remember:** vol products are sharp *short-term* tools — a days-long tail hedge or a tactical long-vol bet — and a guaranteed long-run loser if you hold them. In February 2018, that asymmetry destroyed an inverse-vol note in two sessions, a ~96% loss, when the VIX closed at 37.32.

A retail trader reads that the market looks expensive and decides to buy a crash hedge. Buying put options feels complicated — strikes, expiries, the premium melting away — so the trader reaches for something that sounds simpler: a ticker called VXX, described everywhere as "long volatility," the thing that goes up when the market panics. The trader buys \$10,000 of it in January, satisfied that the portfolio is now protected.

The market does not crash. It grinds quietly higher all year, the VIX drifts in the low teens, and nothing dramatic happens. The trader barely looks at the VXX position because, well, no crash, no payoff — that is how insurance works, right? By December the trader checks the account and finds the \$10,000 has become about \$4,000. The market never fell, the hedge was never needed, and the hedge itself lost 60% of its value doing nothing. It did not expire. Nobody assigned it. It just *melted*.

Now run the opposite trade. In the calm years before 2018, a different crowd noticed that VXX-type products bled steadily, and reasoned: if the long-vol product loses money every day, the *short* of it must make money every day. They bought the inverse — products that profit when volatility stays low — and for years it worked like a money machine. Then on February 5, 2018, the VIX closed at 37.32, more than doubling intraday, and those inverse-vol products lost roughly 96% of their value in a single afternoon. One of the largest was shut down within days. The event got a name: Volmageddon. Both traders — the one who bought the melting hedge and the one who sold it — got hurt by the exact same mechanism, just from opposite sides.

![VIX 2004 to 2024 annual average line with a long-run average and notable closing spikes for the GFC, Volmageddon, COVID, and the yen carry unwind](/imgs/blogs/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll-1.png)

That mechanism — the cost of the roll — is the whole story of this post. To understand it you have to understand three things in order: what the VIX actually measures (and why you cannot buy it), how VIX futures and the term structure work, and how the exchange-traded products that *track* the VIX are built on top of those futures. Get those three right and the melting VXX, the blown-up inverse note, and the right way to use these instruments all fall out of one idea. We will lean on the volatility machinery built up across this series — implied vol as a price in [implied vs realized volatility](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options), the shape of the curve in [the term structure of volatility](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve), and your sensitivity to the level of vol in [vega](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — and tie it all to the practitioner's question: when, if ever, do you actually use this stuff?

## Foundations: what the VIX actually is

Start with the everyday version. Suppose you run a company that sells flood insurance, and you want a single number that summarizes how worried the whole market is about floods over the next month. You could not just look at one policy. Floods come in many sizes — a basement that takes a foot of water, a house washed off its foundation, a whole town underwater — and people buy insurance against all of those scenarios at once. To capture "how worried is everyone," you would have to look at the *prices of the entire menu* of flood policies, from the mild ones to the catastrophic ones, weight them sensibly, and boil them down to one figure. When that figure is high, the market is paying up for protection across the board; when it is low, people are relaxed.

The VIX is exactly that, for the stock market. It is a single number that summarizes how much the options market is paying for protection against moves in the S&P 500 over the next 30 days. When traders are scared, they bid up the price of S&P 500 options (especially the crash-protection puts), and the VIX rises. When they are complacent, option prices sag and the VIX falls. That is why it earned the nickname **the fear gauge**: it is not a forecast that the market *will* fall, it is a read on how much the market is *paying to be protected* if it does.

Now make that precise. An option's market price embeds an assumption about how volatile the underlying will be — the **implied volatility**, the forward-looking vol number you back out of the price using a pricing model (we treated this at length in [implied vs realized volatility](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options); the pricing model itself is derived in [Black-Scholes](/blog/trading/quantitative-finance/black-scholes)). A naive "fear gauge" might just take the implied vol of the single at-the-money S&P 500 option 30 days out. The VIX does something more robust and more interesting: it uses a whole **strip** of out-of-the-money options — puts at strikes below the index and calls at strikes above it — and combines their prices into one number using a variance-swap-style formula.

### The model-free trick

Here is the part that surprises people. The VIX is described as "implied volatility," but it is **not** computed by inverting Black-Scholes on each option and averaging the implied vols. It uses a formula that sums up option *prices* directly, with each price weighted by one over its strike squared, to produce the 30-day **variance** the market is pricing. Then it annualizes that variance and takes the square root to get a volatility in the familiar percentage units. The reason this matters: the variance-swap formula is *model-free*. It does not assume the Black-Scholes world is true; it does not need a single "the" volatility. It is the no-arbitrage price of a contract that pays the realized variance of the index, replicated out of the full menu of option prices. (The mathematics of why a strip of options replicates a variance payoff sits in the quant track — see the [variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) post for the trade that exploits it and the [volatility surface](/blog/trading/quantitative-finance/volatility-surface) for the no-arbitrage object underneath.)

![Pipeline showing a strip of out-of-the-money SPX options weighted by one over strike squared, summed into 30-day variance, blended to constant 30 days, then annualized into the VIX vol number](/imgs/blogs/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll-2.png)

Why use the whole strip instead of just the at-the-money option? Because fear does not live at the money. The most expensive options in a panic are the far out-of-the-money puts — the lottery-ticket crash protection. A measure built only from the at-the-money option would miss exactly the tail the market is most afraid of. By weighting in every OTM strike, the VIX captures the *shape* of demand for protection, not just its center. This is also why the VIX tends to spike harder than at-the-money implied vol alone: in a crash, the deep OTM puts get bid up disproportionately, and the one-over-strike-squared weighting gives those low-strike puts a lot of influence.

A few practical facts to nail down:

- **The units are annualized vol points.** A VIX of 16 means the options market is pricing about a 16% annualized standard deviation of S&P 500 returns over the next 30 days. To get the *monthly* expected move, divide by the square root of 12 (about 3.46): a VIX of 16 implies roughly a 4.6% one-standard-deviation move over the coming month. A VIX of 32 implies about double that.
- **It is a 30-day constant horizon.** The VIX blends the two nearest SPX option expiries to synthesize a clean 30-calendar-day number, so the horizon does not jump around as individual expiries roll off.
- **It is computed continuously** during trading hours from live SPX option quotes by Cboe (which created and maintains it). When you see "the VIX is 18," that is the live output of this calculation, not a quote you could lift and hold.

### The long-run picture

Over its history the VIX has spent most of its time in a fairly tight, low band and occasionally exploded. The long-run average daily close is around 19.5. In calm regimes it sits in the low-to-mid teens; the 2017 *annual average* was about 11.1, an unusually placid year. Then come the spikes: it closed near 80.9 at the depth of the 2008 financial crisis, hit 82.69 on March 16, 2020 in the COVID crash, printed 37.32 on February 5, 2018 (Volmageddon), and 38.57 on August 5, 2024 in the yen-carry unwind. The cover chart above shows the annual averages tracing that calm baseline while the red markers flag the panics.

The *distribution* of those numbers matters as much as the levels. The VIX is sharply right-skewed: most days cluster in the teens, and the rare excursions go enormous. Annual averages tell the regime story — calm years like 2017 (11.1) and 2019 (15.4), elevated crisis years like 2008 (32.7) and 2020 (29.3), and the slow normalization in between (2022 averaged 25.6 in a bear market, 2023 fell back to 16.9). For a trader, two facts fall out of this shape. First, the VIX is *mean-reverting*: a print of 40 is almost never the start of a new normal at 40 — it is a spike that historically decays back toward the high teens over weeks. Second, the asymmetry of the distribution is exactly the asymmetry the products inherit: long the tail you wait a long time for a rare, big payoff; short the tail you collect steadily and then absorb one rare, big loss. The whole VIX-product complex is just a way to take a position on which side of that skewed distribution you want to live on.

#### Worked example: turning a VIX level into an expected move

Suppose the VIX is 20 and you want to know what the market is pricing for the S&P 500 over the next month and the next day. The VIX is an *annualized* one-standard-deviation figure, so to get a horizon-specific move you scale by the square root of time.

For the month: a 30-day window is about 1/12 of a year, so the monthly standard deviation is `20% × sqrt(1/12) = 20% × 0.2887 ≈ 5.77%`. On a \$5,000 S&P 500 index level, that is a one-standard-deviation band of roughly \$5,000 × 5.77% = \$288 over the coming month — so the market is pricing about a two-thirds chance the index lands within ±\$288 of where it is, and a two-thirds chance it stays within roughly ±11.5% (two standard deviations) of where it is.

For the day: there are about 252 trading days in a year, so the daily standard deviation is `20% / sqrt(252) = 20% / 15.87 ≈ 1.26%`. That is the rough size of a "normal" daily move the VIX is implying. Traders use this constantly: VIX 20 means "expect days of about ±1.3%"; VIX 40 means "expect days of about ±2.5%." The key idea is that the VIX is a single annualized number you rescale to whatever horizon you care about by the square root of time, and that is what makes "the fear gauge" actually quantitative rather than vibes.

### The VIX runs hot on purpose

One fact about the VIX explains nearly everything that follows about the products: **the VIX, on average, sits above the volatility that actually shows up.** Over long samples, the VIX (an *implied* vol, forward-looking) averages around 19.5, while the *realized* 30-day volatility that follows averages closer to 15.8 — a persistent gap of roughly +3 to +4 vol points. This is not a measurement error. It is the **variance risk premium**: option buyers are paying a premium for protection above the fair statistical cost, and option sellers collect that premium as compensation for taking on crash risk. The mechanism is the same one that makes most insurance profitable for the insurer over time — the premium exceeds the expected payout, because buyers value the protection more than its actuarial cost and sellers demand to be paid for tail exposure.

Why does this matter for VXX and friends? Because the upward-sloping futures curve *is* the variance risk premium made visible. The reason the one-month VIX future trades above spot (and the two-month above the one-month) is precisely that the market embeds this premium into the term structure — longer horizons price in more of the chance of a future panic, so they cost more. A long-vol product that rolls up that curve is, in effect, *buying* the variance risk premium every day at the price the market sets, and a short-vol product is *selling* it. So the roll cost is not an accident of product design; it is the same structural edge that makes selling options profitable on average, viewed through the futures curve. We trace the trade itself in [the variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) (forward-ref); for now the point is that VXX bleeds because it is the perpetual *buyer* of an insurance premium that, on average, is overpriced.

This also reframes what "the VIX is high" means. A VIX of 25 does not mean realized vol *will* be 25; it means the market is pricing 25 and, historically, the vol that follows tends to come in a few points lower. That gap is why being structurally long vol (VXX) loses over time and being structurally short vol (SVXY) wins over time — until the rare episode where realized vol blows through implied and the seller pays for years of premium in a single week.

## Why you cannot buy the VIX

Here is the fact that trips up almost every newcomer: **there is no such thing as a share of VIX.** The VIX is the *output of a calculation* run on live option prices. It is like an average temperature or a price index — a derived statistic, not a warehouse of something you can take delivery of. You cannot hold it, short it, or arbitrage it directly, because there is nothing to hold.

Compare it to a stock. A share of Apple is a claim on a real company; you can buy it and own a piece. A barrel of oil is a physical thing; you can buy it, store it, and sell it later. The VIX is neither. If the VIX is 18 right now and you somehow "bought it," what exactly would be in your account? Nothing. There is no spot VIX position. The calculation will simply spit out a new number a second later based on fresh option quotes.

So how do people get VIX exposure at all? Through **derivatives that settle to the VIX**:

- **VIX futures.** A VIX future is a contract whose value at expiry equals the VIX level on that expiry date. If you buy a VIX future that expires in 30 days, you are making a bet today on where the VIX calculation will print 30 days from now. Crucially, *before* expiry the future trades at its own price, which can be very different from today's spot VIX — and that gap is the entire engine of the cost of the roll.
- **Options on the VIX.** Calls and puts whose underlying is the VIX (technically, the relevant VIX future). These let you bet on the VIX crossing a level by a date, with the convexity and decay of any option.
- **Exchange-traded products (ETPs)** — VXX, VIXY, UVXY, SVXY and their kin — which package a *rolling basket of VIX futures* into a single ticker you can buy in a normal brokerage account. This is what the retail crowd actually owns, and it is where almost all the pain lives.

Notice what is missing from that list: any way to hold the spot VIX itself. Every single instrument above is built on VIX *futures*, directly or indirectly. So before we can understand why VXX melts, we have to understand the futures and their term structure.

## VIX futures and the term structure

A VIX future is a price, today, for the VIX at some future expiry. Line up the prices of VIX futures for the next several monthly expiries — one month out, two months, three, and so on — and you get the **VIX futures curve**, the term structure of expected volatility. We covered the general shape and its meaning in [the term structure of volatility](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve); here we only need the two regimes and what they do to the products.

**Contango** is when the curve slopes *upward*: longer-dated futures cost more than nearer-dated ones, and all of them sit above the current spot VIX. This is the normal state of the world — the VIX is in contango roughly 75–85% of the time. The intuition: most of the time volatility is low and calm, and the market figures that over a longer horizon *something* could go wrong, so it prices a higher VIX further out. The far months carry a risk premium for the chance of a future panic. So in a calm market you might see spot VIX around 13.9, the one-month future at 14.5, the two-month at 15.5, the six-month at 17.6 — each step up the curve a little richer.

**Backwardation** is the opposite: the curve slopes *downward*, with the front futures *above* the back. This happens during an actual panic. When the VIX has already spiked to 38, the market expects it to *come back down* as the crisis resolves, so the near-month future (which expires while the panic might still be raging) is the most expensive, and the far months price the eventual return to calm. A stress curve might run 38 at the front, 32 two months out, 24 six months out — inverted.

The single most important consequence of the curve is what happens to a future *as it approaches expiry*: a VIX future must converge to the spot VIX at expiry, because at expiry it settles to exactly that. So in contango — where the future sits above spot — the future has to **roll down** toward the lower spot as time passes, even if spot VIX never moves at all. A two-month future at 15.5 must, all else equal, drift down toward the 13.9 spot as it ages into the front of the curve. That downward drift is not a loss caused by the VIX falling; it is a loss caused by *time passing on an upward-sloping curve*. Hold that thought, because it is the whole ballgame for VXX.

#### Worked example: the roll-down on a single VIX future

Take the calm curve above: spot VIX 13.9, one-month future 14.5, two-month future 15.5. You buy one two-month future at 15.5, expecting to hold it for a month. Suppose over that month *nothing happens* — the spot VIX stays pinned at 13.9 and the whole curve keeps its shape.

After a month, your former two-month future is now a one-month future. With the curve unchanged, a one-month future is worth 14.5. You bought at 15.5 and it is now worth 14.5: you have lost `15.5 − 14.5 = 1.0` vol point, or `1.0 / 15.5 ≈ 6.5%` of your money, **purely from the passage of time on an upward curve.** The VIX did not move. You were not wrong about anything. The future simply rolled down the slope toward spot.

If you had then rolled into a fresh two-month future at 15.5 and held *that* for a month with the curve still flat, you would lose another ~6.5%. Do that all year and you compound roughly `(1 − 0.065)` twelve times, ending around 45% of where you started — and that is in a *flat* market. The lesson: in contango, simply *being long a VIX future and rolling it* is a losing position absent a vol spike. The roll-down is the price you pay to hold the protection.

## How the ETPs are built — and why VXX melts

Now we can assemble the product. VXX (and its near-twin VIXY) is an exchange-traded product designed to give you a **constant 30-day** long-VIX-futures exposure. It cannot hold spot VIX (there is none), and it does not want to hold a single fixed future (which would age and eventually expire), so it holds a *blend* of the first- and second-month VIX futures, weighted to synthesize a constant one-month maturity. Every single trading day it rebalances that blend: it sells a little of the expiring front-month future and buys a little of the second-month future to keep the maturity pinned at 30 days.

In contango, look at what that daily rebalance is: it **sells the cheaper, expiring front future and buys the richer, longer second future** — then watches what it just bought roll down the curve toward the lower front over the following days, only to sell it again cheaper and buy the next one richer. It is mechanically buying high and selling low, every day, by construction. This is the **roll cost** or **roll yield** (negative, in contango), and it is not a fee or an expense ratio — it is structural, baked into holding a constant-maturity long-vol position on an upward-sloping curve.

![VIX futures curve in calm contango with the held future rolling down toward spot and the roll cost shaded between the one-month and two-month points](/imgs/blogs/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll-3.png)

The chart above shows it: the fund effectively lives between the one- and two-month points of an upward curve and bleeds the gap between them every roll cycle. Because the loss compounds daily, VXX is a **melting ice cube** over any horizon longer than a few weeks. This is not a defect or a scandal — VXX does exactly what it says, holds a constant-maturity long-vol position. It is just that *holding* a long-vol position costs money in calm markets, the same way holding fire insurance costs premium even when your house does not burn. The product faithfully passes that cost on to you.

#### Worked example: the VXX roll-cost over several months in contango

Let us make the bleed concrete using the calm curve (front 14.5, second 15.5 — a roll cost of `15.5 − 14.5 = 1.0` vol point, about `1.0 / 15.5 ≈ 6.5%` per roughly 21-day roll cycle). Assume a calm, flat market for six months and start with \$10,000 in VXX.

A simple way to model the constant-maturity roll is a steady daily drag. The monthly drag is ~6.5%, so the daily drag is about `6.5% / 21 ≈ 0.31%` per trading day. Over a 21-trading-day month with the VIX flat, the position multiplies by roughly `(1 − 0.0031)^21 ≈ 0.937`. Chain six of those months:

```
    month 0:  $10,000
    month 1:  10,000 x 0.937 = $9,370
    month 2:   9,370 x 0.937 = $8,780
    month 3:   8,780 x 0.937 = $8,227
    month 4:   8,227 x 0.937 = $7,708
    month 5:   7,708 x 0.937 = $7,222
    month 6:   7,222 x 0.937 = $6,767
```

After six months of a flat, calm market, the \$10,000 has eroded to about \$6,767 — a 32% loss with the VIX literally unchanged the entire time. The market never moved, your "hedge" was never needed, and it still cost you a third of your stake. That is the cost of the roll, and it is exactly what happened to our opening trader who held VXX through a quiet year. The intuition: in contango, time is your enemy when you are long vol, the same way time is the long option-holder's enemy through theta (see [theta](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options)) — VXX bundles a slow, relentless theta-like bleed into a single ticker.

### VXX versus the VIX itself

This is where the most damaging misconception comes from. People buy VXX believing it *is* the VIX — that if the VIX goes from 14 to 28, VXX doubles. It does not, and the gap compounds over time. Over a single day VXX moves *with* the front of the futures curve, which is correlated with but not equal to spot VIX — the front future moves less than spot on a given day, and VXX (constant-maturity, between the first two futures) moves even less. Over weeks and months, the roll drag drives VXX *far below* the index. The VIX can end a calm year roughly where it started while VXX is down 50–80%.

![Two lines rebased to 100 over a calm trading year, the VIX index drifting around its level while the long-vol fund decays well below it from daily roll drag](/imgs/blogs/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll-4.png)

The chart above (illustrative, driven by the measured roll cost from a calm contango curve) makes the divergence visible: the index wanders around 100 all year, and the rolling long-vol fund melts to the low forties. Same starting point, same daily index moves — but one is dragged down by the roll and the other is not. This is why issuers have to do reverse splits on VXX every so often: left alone, the share price would grind toward zero. VXX has been reverse-split many times since launch; a single original share's worth has lost effectively all of its value over the long run, punctuated by violent rallies during crashes. **VXX is a vehicle for a short, sharp burst of vol exposure, not a thing you hold.**

#### Worked example: long-term VXX decay versus the index

Stretch the bleed over a multi-year hold to see why "buy and forget" is fatal. Use the same ~6.5% monthly roll drag in steady contango. Suppose the VIX index itself is unchanged over three years — it starts and ends the period at the same level, having wandered in between — so an instrument that *actually tracked the index* would be flat. What does a constant-maturity rolling product do?

Three years is 36 months. Each month the position multiplies by roughly `(1 − 0.065) = 0.935`. Over 36 months:

```
    surviving fraction = 0.935 ^ 36
    0.935 ^ 12 (year 1) = 0.448   -> down ~55% in year one
    0.935 ^ 24 (year 2) = 0.201   -> down ~80% by year two
    0.935 ^ 36 (year 3) = 0.090   -> down ~91% by year three
```

A \$10,000 position becomes about \$4,480 after one year, \$2,010 after two, and \$900 after three — a 91% loss while the index it "tracks" went *nowhere*. The reverse splits hide this from the casual eye (the share price gets reset upward periodically), but the dollar erosion is relentless. The intuition: a constant-maturity long-vol product is a position whose carrying cost is so high that flat markets alone will destroy 90% of it in a few years — it is the single clearest case in liquid markets of an instrument engineered to lose money to anyone who holds it.

## The leverage and the inverse: UVXY and SVXY

VXX and VIXY are the plain 1x long-vol products. The family has two important variants, and the leverage and sign change everything about who the roll helps and who it hurts.

**UVXY** is the *leveraged* long-vol product, currently 1.5x daily (it was originally 2x). It targets 1.5 times the daily return of the same constant-maturity VIX futures index that VXX tracks. On a day the index rises 10%, UVXY aims for about +15%; on a day it falls 10%, about −15%. Two things compound the danger. First, the roll cost is amplified — you are bleeding roughly 1.5x the contango drag, so UVXY melts even faster than VXX. Second, like all daily-rebalanced leveraged products, it suffers **volatility decay** (path dependence): because it resets to 1.5x leverage every day, a choppy sideways path erodes value even if the index ends flat, because the daily compounding of leveraged up-and-down moves does not net to zero. UVXY is a tool measured in *hours to days*, for a sharp spike, in small size. Held for weeks it is a wealth incinerator.

**SVXY** is the *inverse* product, currently -0.5x daily (it was -1x before 2018 — that change is the scar tissue from Volmageddon, which we will get to). It targets minus one-half the daily return of the futures index. When the index falls (calm market, vol bleeding), SVXY rises — it is the product that *harvests* the roll cost that VXX and UVXY pay. This is the **short-vol carry trade** in a box: in contango, being short VIX futures earns the roll-down, day after day, like collecting insurance premiums. For years it was one of the most profitable trades anyone could find. And then, periodically, it detonates, because you are short a thing that can triple overnight.

![Matrix of VXX VIXY, UVXY, SVXY, and the index VIX across exposure, roll behavior in contango, and hold horizon, color coded by who the roll helps or hurts](/imgs/blogs/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll-5.png)

The matrix above lays out the zoo: the +1x and +1.5x long products bleed in contango and are days-or-shorter instruments; the -0.5x inverse earns the roll but lives "until the next spike"; and the index VIX in the bottom row is the reminder that the reference itself is not tradable at all. Read down the "roll in contango" column and the asymmetry jumps out — red bleed for the longs, green carry for the short, and that green is exactly the thing that can turn catastrophic.

#### Worked example: the inverse-vol product and the ~96% Volmageddon loss

Here is the math that wiped out the short-vol crowd. Before Volmageddon, the popular inverse products were -1x — they returned roughly *minus* the daily move of the front VIX futures. Going into February 5, 2018, the market was extraordinarily calm; the front VIX future was trading in the low teens, call it 13. Short-vol holders had been compounding the roll for two years.

Then on February 5, the VIX closed at 37.32, more than doubling on the day, and the front VIX future roughly doubled with it intraday — say from about 13 to about 25–26, a move of roughly +100% in the front future. A -1x product returns approximately *minus* that move: `−1 × (+100%) ≈ −100%`. The note's value is wiped to near zero.

Walk a \$100 note through it:

```
    value before (Feb 2):     $100
    front future move:         roughly +100% over the spike
    inverse note return:       about -1 x (+100%) = -96% to -100%
    value after (Feb 6):       roughly $4
```

The largest -1x ETN lost about 96% of its value in two sessions and was terminated by its issuer within days; the inverse-vol ETF survived but was so battered that the sponsor cut its target leverage from -1x to -0.5x to make a total wipeout harder. The intuition is brutal and worth burning in: **a short-vol position has a tiny, steady gain and an enormous, sudden loss** — you pick up pennies in front of a steamroller, and the steamroller is a VIX spike that the very products feeding it help create. The full anatomy of the feedback loop is its own case study (forward-ref: [case study: Volmageddon 2018 and the short-vol blowup](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup)).

![Bar chart of an inverse-vol note worth one hundred dollars before February 2018 collapsing to about four dollars after the VIX closing print of 37.32, a 96 percent loss](/imgs/blogs/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll-6.png)

### The reflexive danger: when the products move the market

There is a deeper hazard in these products than just the roll, and it is the reason Volmageddon was not merely a few traders' bad day but a market event. Leveraged and inverse VIX ETPs must **rebalance daily to maintain their stated leverage**, and the direction of that rebalance is destabilizing: to keep its leverage after the VIX rises, a leveraged-long product must *buy more* VIX futures, and an inverse product that has lost money must *buy back* the futures it is short — both of which means **buying VIX futures into a rising VIX, near the close.**

On a normal day this is a rounding error. But by early 2018 the short-vol complex had grown so large that the rebalancing flows were big relative to the VIX futures market itself. When the VIX started spiking on February 5, the products faced enormous end-of-day buying needs to rebalance, that buying pushed VIX futures higher, the higher futures meant even larger rebalancing needs, and the loop fed on itself into the close. The product's own mechanical hedging amplified the very spike that was destroying it — a textbook reflexive feedback loop. This is why the asymmetry of short-vol is worse than it looks on paper: in a stampede, your hedging and everyone else's hits the same illiquid market at the same moment, and the price you need is exactly the price that is running away from you.

## Reading the term structure and using VIX options

If the products are blunt instruments, the underlying VIX futures and options are precision ones — and you do not have to trade them to benefit, because the *shape of the curve* is one of the most useful free signals in the market.

The first thing a vol trader checks is the **front-to-back relationship**: is the curve in contango (front below back, calm, roll working against longs) or backwardation (front above back, panic, roll working *for* longs)? A common quick gauge is the ratio of the one-month to the three-month VIX future (or the analogous nine-day-versus-30-day relationship): well above 1.0 means steep backwardation and acute stress; comfortably below 1.0 means calm contango. When the curve flips from contango into backwardation, it is telling you the market has moved from "vol is a cost I'm paying to hold protection" to "vol is so high the market expects it to fall." That regime flip is the difference between VXX melting and VXX rocketing, and it is observable in real time.

The second tool is **options on the VIX itself**. A VIX call is a bet that the VIX will be above a strike at expiry; a VIX put, below. These are genuinely different from VIX futures in two ways. First, they have the convexity of any option — a VIX call's payoff accelerates in a spike, so a small premium can pay off many times over in a panic, which is why some tail-hedge programs prefer VIX calls to VXX (capped, known cost; explosive upside). Second, **VIX options are priced off the corresponding VIX future, not spot VIX** — a subtlety that catches people constantly. If spot VIX is 14 but the relevant future is 16, a "14-strike" VIX call is *out of the money relative to its real underlying*, not at the money. The whole apparatus of moneyness (see [moneyness and the strike](/blog/trading/options-volatility/moneyness-and-the-strike-itm-atm-otm-and-what-you-are-really-buying)) applies, but against the future. This also means VIX options carry their own implied volatility — the "vol of vol" — which spikes in stress and crushes in calm just like equity-index vol; trading them well requires reading that second-order surface, the territory of [vega and the vol of vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol).

For positioning, the practical playbook is: use the curve's *slope* to size the cost of any long-vol trade (steep contango = expensive to be long, cheap to be short; backwardation = the reverse), and use VIX options when you want defined, convex tail exposure rather than the open-ended bleed of a rolling ETP. A trader who wants protection that does not melt buys a VIX call dated past the worry window and accepts a known premium; a trader who wants to harvest carry sells a VIX call spread or holds a small short-vol position and prices the tail. Either way, the curve tells you which side the wind is at your back.

## Common misconceptions

**"VXX tracks the VIX, so if the VIX doubles, VXX doubles."** No. VXX tracks a *constant-maturity basket of VIX futures*, and over any meaningful horizon the roll drives it far below the index. Numerically: in a calm year the VIX can finish roughly flat while VXX is down 50–80% (our illustrative chart had it ending near 40 against an index at 100). Even on a single day VXX moves less than spot VIX, because the front futures move less than spot. If you want the VIX's exact move, there is no product that gives it to you — there is no spot VIX to hold.

**"VIX products are a cheap, set-and-forget crash hedge."** The opposite. They are an *expensive* hedge precisely because they bleed the roll cost continuously. The six-month worked example turned \$10,000 into about \$6,767 in a flat market — that 32% is the carrying cost of the hedge, paid whether or not a crash arrives. A protective put has a known, capped cost (the premium) and a defined expiry; VXX has an open-ended, compounding cost and no payoff unless vol actually spikes while you hold it. For sizing and structuring real portfolio protection, see [hedging a portfolio with options](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) (forward-ref).

**"Shorting VXX (or buying the inverse) is free money because VXX always goes down."** It goes down *most* of the time and then, occasionally, triples in a week. The short-vol carry trade earns small steady gains and is exposed to an unbounded, sudden loss — the literal definition of picking up pennies in front of a steamroller. The -1x notes lost ~96% in two days in February 2018. "Always goes down" is survivorship bias talking right up until the day it does not.

**"UVXY is just a more efficient way to be long volatility — more bang for the buck."** UVXY gives you 1.5x the *daily* return, which means 1.5x the roll bleed *and* volatility decay from daily resetting. Over a choppy month with the index flat, a daily-reset leveraged product can be down meaningfully purely from path dependence — the geometry of leveraged compounding, not any directional view being wrong. UVXY is more bang only over a single sharp move measured in hours to a few days; held longer, the leverage works against you on both the roll and the path.

**"The VIX is the market's prediction of how far stocks will fall."** The VIX is *direction-agnostic*. It measures the *size* of the expected move, not its sign — it is built from both OTM puts and OTM calls. A high VIX says "big moves are being priced," not "down moves are coming." It tends to rise in selloffs because crash protection (puts) gets bid hardest, but the number itself is a width, not a forecast of direction. This is the same point as vega being identical for calls and puts (see [vega](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol)) — volatility is about the size of the cone of outcomes, not which way it points.

## How it shows up in real markets

**February 5, 2018 — Volmageddon.** The canonical example, and the one that named the failure mode. After a year of record-low volatility (2017's VIX averaged ~11.1), the short-vol complex had ballooned. On February 5 the VIX closed at 37.32, more than doubling intraday; the front future roughly doubled; the -1x inverse notes lost ~96%; the largest was terminated within days. The mechanism was the reflexive rebalancing loop — products buying VIX futures into the spike near the close, amplifying it. The market lesson was that a crowded short-vol trade is a hidden source of systemic fragility, because the participants' hedging is destabilizing by construction. SVXY's target leverage was cut from -1x to -0.5x in the aftermath to make a total wipeout structurally harder.

**March 16, 2020 — COVID crash.** The VIX closed at 82.69, eclipsing even the 2008 peak. The futures curve slammed into steep backwardation — the front future printing far above the back as the market expected the panic to eventually subside. For the rare holder of a long-vol product *into* the spike, this was the payoff that justifies the instrument's existence: VXX multiplied several-fold in weeks as both the level rose and the curve inverted (backwardation means the roll briefly works *for* the long, not against it). But it required being positioned *before* the move; chasing VXX after the VIX was already at 60 meant buying a future trading far above the spot it would eventually converge to.

**August 5, 2024 — yen carry unwind.** The VIX closed at 38.57, and intraday it briefly printed far higher in thin early trading. A leveraged unwind of yen-funded positions cascaded into a global volatility spike. Long-vol products surged for a day or two; the spike collapsed almost as fast as it came, and anyone who bought the pop near the highs and held even a week gave most of it back as the curve normalized and the roll resumed its bleed. The episode was a clean reminder that vol spikes are typically *fast and mean-reverting*, which is exactly why these products are timing instruments measured in days, not positions.

**The everyday case — the quiet calendar year.** Most years are not Volmageddon; they are the boring grind of our opening trader. The VIX sits in the teens, contango is steady, and any long-vol product melts a few percent a month. This is the modal outcome, and it is why "I'll just hold some VXX as a hedge" is a slow, near-certain wealth transfer from hedgers to the short-vol carry crowd — until, on one day a few times a decade, the carry crowd pays it all back at once.

The deeper reason all four episodes rhyme is that VIX spikes are not symmetric events. Volatility goes *up like an elevator and down like an escalator*: it jumps in a single session when fear hits and then decays slowly over weeks as calm returns. That asymmetry is structural — fear arrives all at once (a headline, a default, a margin call) but confidence rebuilds gradually. For the products, this shape is everything. It means a long-vol position needs to be *on before* the elevator, because by the time you see the spike the move is largely done; and it means a short-vol position will look brilliant for the long, slow escalator down and then get its face ripped off on the next elevator up. Every one of the dated episodes above is the same elevator-up event punishing whoever was short and rewarding whoever was already long — and then the slow escalator down quietly punishing whoever stayed long too long.

#### Worked example: sizing the short-vol carry against a blowup

Suppose you are tempted by the short-vol carry trade and want to size it honestly. You have \$100,000 and you are looking at SVXY (currently -0.5x). In steady contango, the underlying futures index might fall roughly 5–6% a month from the roll, so a -0.5x product gains about half of that — call it +3% a month, a tempting ~36% annualized before the inevitable drawdowns. The temptation is to put a big slice of capital in.

Now price the tail before you size it. Model a Volmageddon-style event: the front VIX future doubles (+100%) in a session. A -0.5x product returns about `−0.5 × (+100%) = −50%` on that move. So whatever you allocate, you must be willing to lose half of it *overnight*, with no chance to react — and a -1x product (the pre-2018 kind) would lose nearly all of it. Size accordingly:

```
    allocation $A, steady carry:  about +3% / month
    blowup event (-50% on -0.5x):  instant loss of 0.5 x A
    if A = $20,000 (20% of capital): blowup loss = $10,000 (10% of net worth)
    if A = $50,000 (50% of capital): blowup loss = $25,000 (25% of net worth)
```

If you size the carry at 20% of capital, a doubling of the front future costs you 10% of your net worth in an afternoon — survivable, painful. If you let it grow to half your capital because the steady gains felt safe, the same event takes a quarter of everything at once. The discipline is to size the position by its *tail*, not its *carry*: decide how much you can lose in one session to a doubling of vol, and let that — never the seductive monthly drip — set the position. The intuition is the insurer's: you are writing catastrophe coverage, and the only way to survive in that business is to never write more than your capital can absorb when the catastrophe finally comes.

#### Worked example: sizing a VXX tail hedge into a spike

Suppose you run a \$1,000,000 equity portfolio and want a small, tactical tail hedge for the next two weeks because an event worries you. You do *not* want a big standing VXX position (the roll would bleed it). Instead you size a deliberately small, short-dated bet on a vol spike.

You allocate 1% — \$10,000 — to VXX, planning to hold days, not months. Now stress it against a real spike. Take the August 5, 2024 print: the VIX jumped from the mid-teens to a 38.57 close, and the front VIX future rallied sharply — a constant-maturity long-vol product like VXX might rise on the order of +50% to +70% over such a move (it moves less than spot VIX because it sits between the first two futures, and the back of the curve rises less than the front). Call it +60% on your \$10,000:

```
    hedge allocation:        $10,000  (1% of the portfolio)
    VXX move on the spike:    about +60%
    hedge value after:       10,000 x 1.60 = $16,000
    hedge gain:              about +$6,000
```

If the same shock knocked your \$990,000 of equities down, say, 6%, that is a \$59,400 paper loss on the stock (−0.06 × \$990,000) — and the \$6,000 hedge gain offsets about 10% of it. That is a *partial*, cheap, short-duration cushion, not full insurance, and the key discipline is the exit: you take the hedge off within days, because the moment the spike fades the curve re-contangos and your VXX starts melting again. The intuition: a VIX product is a match, not a furnace — you strike it for a brief, bright burst around an event and put it out before it burns your hand.

## The playbook: how to actually use vol products

Everything above collapses into a single decision tree. The question is never "should I own some VIX exposure as a portfolio staple" — the answer to that is always no, because the roll guarantees a long-run loss. The question is "do I have a *specific, time-boxed* reason to want vol exposure right now, and which instrument and size fit it."

![Decision graph for when to use a VIX product, branching from the motivation into a short tail hedge, a tactical long-vol bet, the short-vol carry trade, and the never-buy-and-hold warning](/imgs/blogs/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll-7.png)

Walk the branches:

- **Short-term tail hedge (you fear a near-term crash).** Use VXX in *small* size (1% or so), with a hard time box of days to a couple of weeks, and a pre-committed exit. The position profile is long vega and long the front of the futures curve. It pays off in a fast spike (level up, curve inverting). Invalidation: the event passes without a spike — you take the small, known bleed and exit. Do *not* let it ride; the roll turns a hedge into a slow leak the moment the fear fades. For most portfolios, sized protective puts (see [hedging a portfolio with options](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk)) are a cleaner hedge with a capped, known cost and a defined expiry — reach for VXX only when you specifically want the convex, fast-twitch vol exposure.

- **Tactical long-vol bet (you think vol is too low and about to rise).** This is a directional view on volatility itself — a vega bet via [vega](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol). VXX or, for a sharper move, a *tiny* slug of UVXY measured in hours to days. The entry edge has to beat the roll: if you are paying ~6.5% a month in contango drag, you need the spike to come *soon* and *big* enough to clear that. The discipline is brutal time management — the roll is a clock ticking against you every day you are early. Invalidation: vol keeps grinding lower or stays flat; cut it, because being right on direction but early on timing still loses to the roll.

- **Short-vol carry (you want to harvest the roll).** This is selling insurance: short VIX futures or hold SVXY to collect the contango roll-down. It works most of the time and pays for it with a fat, sudden tail. If you do it: size it *tiny* relative to capital, model the loss against a *doubling* of the front future (a -0.5x product loses ~50%, a -1x ~100% on such a move), and never let the position grow into a crowded, systemically fragile size — that is what turned 2018 into Volmageddon. The honest framing is in [the variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) (forward-ref): selling vol pays a real premium until it doesn't, and the "doesn't" is a cliff.

- **Buy-and-hold "hedge" (you want to park it long-term).** Never. The roll guarantees the position melts toward zero over time; this is the one branch with no legitimate use. If you want standing tail protection, structure it with options that have a known cost and expiry, or hold cash and quality. A long-term VXX position is a fee you pay the short-vol crowd for the privilege of slowly losing money.

Two cross-cutting rules tie the whole playbook together. First, **always know whether the curve is in contango or backwardation before you trade these** — it flips the sign of the roll, and the [term-structure post](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve) is the prerequisite read. Second, **the dealers and market makers on the other side of these products are sizing the very flows that can move the VIX** — understanding how their hedging works (see [market makers and high-frequency trading](/blog/trading/finance/market-makers-and-high-frequency-trading)) is what lets you anticipate the reflexive air-pockets, and the broader case for treating volatility as a tradable asset class in its own right is laid out in [volatility as an asset](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear).

The single sentence to leave with: **the VIX is a number you cannot own, vol products are leveraged or rolling bets on VIX futures that cost the roll every day, and so they are matches to strike for a brief, bright burst — never furnaces to keep lit.** The trader who internalizes that uses VXX for a week around an event and walks away; the trader who forgets it either watches a "hedge" melt 60% in a calm year or gets steamrolled selling vol into the spike the whole complex is creating. Both mistakes are the cost of the roll, paid from opposite sides.

## Further reading & cross-links

- [The term structure of volatility: contango, backwardation, and the VIX curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve) — the prerequisite on the futures curve whose slope sets the sign of the roll.
- [Implied vs realized volatility: the trade at the heart of options](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) — implied vol as a price, which is what the VIX measures.
- [Vega: your exposure to implied volatility and the vol of vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — your dollar sensitivity to the level of vol, which a long-vol product packages.
- [What is an option: the right, not the obligation](/blog/trading/options-volatility/what-is-an-option-the-right-not-the-obligation) — the insurance intuition the whole vol complex is built on.
- [The variance risk premium: why selling vol pays until it doesn't](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) — the structural edge the short-vol carry trade harvests, and its cliff.
- [Case study: Volmageddon 2018 and the short-vol blowup](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup) — the full anatomy of the reflexive feedback loop.
- [Hedging a portfolio with options: protective puts, collars, and tail risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) — the capped-cost alternative to a melting VXX hedge.
- [Volatility as an asset: owning fear](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear) — the allocator's case for treating vol as its own asset class.
- [Market makers and high-frequency trading](/blog/trading/finance/market-makers-and-high-frequency-trading) — the dealers whose hedging flows feed the reflexive VIX spikes.
- [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) and [the volatility surface](/blog/trading/quantitative-finance/volatility-surface) — the pricing theory under the implied vol the VIX is built from.
