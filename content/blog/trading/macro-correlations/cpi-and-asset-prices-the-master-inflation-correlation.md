---
title: "CPI and Asset Prices: The Master Inflation Correlation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why the Consumer Price Index is the single most market-moving number on the calendar, why it is the surprise and not the level that prices assets, and why the same hot print can crush everything in one regime and barely move a needle in another."
tags: ["macro", "correlation", "cpi", "inflation", "surprise", "stocks", "bonds", "gold", "crypto", "regime", "real-yields"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — CPI is the single most market-moving indicator, but asset prices correlate with the CPI *surprise* (actual minus consensus), not the level, and the *sign and size* of that correlation are regime-dependent: in the 2021-23 inflation scare a hot print was bad for almost everything, while in a calm 2% world the same beat barely registers.
>
> - The reaction keys on the **surprise**, not the headline number. A 3.5% print that everyone expected moves nothing; a 3.2% print when 2.9% was expected can move trillions.
> - In the 2022 regime, per +0.1pp core CPI upside surprise: S&P 500 about −0.7%, Nasdaq −1.0%, 10-year yield +7bp, 2-year yield +9bp, the dollar +0.35%, gold −0.8%, Bitcoin −1.6%. Everything risky fell together.
> - The correlation **flips**. Stock–bond correlation runs about −0.45 when inflation sits below 2% and about +0.50 when it runs above 4% — diversification works in calm and fails in the storm.
> - The one number to remember: the 9.06% June 2022 CPI peak was a 40-year high, and that single regime rewrote the correlation map for every asset on Earth.

## The print that moved the world

At 8:30 a.m. Eastern on Tuesday, September 13, 2022, the U.S. Bureau of Labor Statistics released the August Consumer Price Index. Economists had penciled in a small monthly decline in the headline number — gasoline had fallen hard that summer, and the consensus story was that inflation had peaked and was rolling over. Equity futures had rallied into the print on exactly that hope. Then the data hit the tape. Headline CPI rose 0.1% on the month instead of falling, and *core* CPI — the part that strips out food and energy — jumped 0.6%, double what was expected. Inflation was not rolling over. It was broadening.

The market's response was not a polite repricing. It was a stampede. The S&P 500 closed down about 4.3% on the day — its worst single session since June 2020 — the Nasdaq 100 fell more than 5%, the 2-year Treasury yield leapt roughly 17 basis points to a fresh cycle high, the dollar ripped higher, and gold and Bitcoin both slid. There was nowhere to hide. A stock investor, a bond investor, a gold bug, and a crypto trader all lost money on the same morning, for the same reason, off the same number. That is what it means to call CPI the master inflation correlation: a single data point reached into every asset class at once and pulled the same direction.

Now hold that day in your mind and travel forward to a different print. On November 10, 2022 — barely two months later — CPI came in *cooler* than expected: core rose 0.3% against a 0.5% consensus. The reaction was a mirror image. The S&P 500 surged about 5.5% on the day, one of its best sessions of the decade; the Nasdaq jumped more than 7%; the 10-year yield collapsed almost 30 basis points; gold and crypto ripped. Same indicator, same calendar, opposite outcome — because the *surprise* flipped sign. This post is about exactly that machinery: why CPI is tier-one, why the asset price tracks the surprise rather than the level, why the whole map is conditional on the inflation regime, and how you actually read a print.

![CPI surprise transmission fan from rate expectations to every asset](/imgs/blogs/cpi-and-asset-prices-the-master-inflation-correlation-1.png)

The figure above is the mental model for the entire piece. A hot CPI print does not hit each asset independently. It is repriced first into expectations for the Federal Reserve's policy path and into the discount rate the market uses to value every future cash flow. From there it radiates: front-end yields up, the dollar up, stocks down, bonds down, gold down, crypto down. The correlation across assets that looks so dramatic on a CPI day is really one shock travelling down one pipe and splitting at the end.

## Foundations: what CPI is and why it sits at the top of the calendar

Before we can talk about correlation, we have to define the two things being correlated: the indicator and the asset price. Let us build both from zero, because nearly every confusion about CPI-and-markets comes from a fuzzy grip on one of these.

### What the Consumer Price Index actually measures

The Consumer Price Index is the government's attempt to answer one deceptively simple question: how much more (or less) does a typical urban household pay this month for the same basket of goods and services it bought before? The BLS sends data collectors out to price tens of thousands of specific items — a dozen eggs, a gallon of gas, a month's rent on a representative apartment, a doctor's visit, a movie ticket — across the country, every month. It weights each item by how much of the average household budget it represents (shelter is the heavyweight at roughly a third of the index; food and energy are smaller; haircuts and airfares smaller still), and it rolls the whole thing into a single index number.

The number you hear quoted on the news is almost always the *year-over-year* change in that index: "CPI rose 3.2% in the year through March." That is the inflation rate. There is also a *month-over-month* change ("core CPI rose 0.4% on the month") that traders watch even more closely, because it tells you the *current run-rate* of inflation rather than a number polluted by what happened twelve months ago.

Two flavors matter enormously and you must never confuse them:

- **Headline CPI** includes everything, food and energy included. It is the number that affects real wallets, but food and especially energy prices are volatile and driven by global supply shocks (a hurricane, an OPEC decision, a war) that have nothing to do with the underlying inflation trend.
- **Core CPI** strips out food and energy. It is uglier as a measure of cost-of-living but far better as a *signal* of where inflation is heading, because it captures the sticky, broad-based price pressure the Fed actually targets. **When the market reacts to CPI, it reacts mostly to core.** A hot headline driven entirely by a gasoline spike, with a soft core, often gets faded; a hot core with a soft headline gets sold hard. We will see this borne out in the betas later.

### What we mean by "asset price" and "correlation"

The other half of the relationship is the asset price — the level of the S&P 500, the yield on a 10-year Treasury, the price of an ounce of gold, the price of one Bitcoin, the value of the dollar against a basket of currencies (the DXY index). For this series, what we care about is not the *level* of those prices in isolation but how they *move together* — their correlation.

**Correlation, defined from scratch:** if two things tend to go up and down together, they are positively correlated; if one tends to rise when the other falls, they are negatively correlated; if there is no reliable relationship, they are uncorrelated. The standard measure, the Pearson correlation coefficient (written `r`), runs from +1 (they move in perfect lockstep) through 0 (no linear relationship) to −1 (perfect mirror image). A closely related idea is **beta**: if correlation tells you the *direction and reliability* of the relationship, beta tells you the *size* — "for each one-unit move in X, Y moves β units on average." For CPI, the beta we care about is "for each extra 0.1 percentage point of core CPI surprise, the S&P moves X%." We unpack the difference between correlation and beta more carefully in [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta); here we just need the intuition that *sign* and *size* are two separate questions.

### Why CPI is tier-one

There are dozens of economic releases each month. They are not equal. Traders informally rank them in tiers by how much they move markets, and CPI sits in the top tier alongside the jobs report (NFP) and Fed decisions. Three things put it there:

1. **It is timely and it is about prices.** Inflation is the variable the central bank exists to control. CPI is the cleanest, earliest monthly read on it. (Technically the Fed targets the PCE deflator, not CPI — but CPI comes out roughly two weeks earlier and the two move together, so CPI is the *market's* leading read on what PCE will show.)
2. **It directly sets the policy path.** A hot CPI means the Fed will likely keep rates higher for longer; a cool one opens the door to cuts. Because interest rates are the price of money — the discount rate that values every asset — a shift in the expected path of rates reprices everything simultaneously. We do not re-derive that transmission here; it is the subject of [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [how policy moves every asset](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map). Our job is to measure the resulting correlation.
3. **It is a scheduled, binary event.** Everyone knows the exact minute it drops. Positioning crowds into it, options are priced around it, and the entire market holds its breath. That concentration is what produces the violent same-second, cross-asset moves we saw on September 13, 2022.

> [!note]
> This is the *correlation* series, not the *mechanism* or *reaction* series. For why a rate change moves an asset (the causal chain), lean on the macro-trading posts. For the minute-by-minute intraday reaction to the release, lean on [CPI, the report that moves the world](/blog/trading/event-trading/cpi-the-report-that-moves-the-world). Here we are after the measurable statistical fingerprint — the sign, the size, the lead/lag, and the regime in which it flips.

### How the basket is weighted, and why that matters for the reaction

CPI is not one price; it is a weighted average of thousands. The weights are where the signal lives, and a trader who knows them reads the print differently than one who only sees the top-line number. Shelter — the cost of housing, captured through actual rents and a model of what owner-occupied homes would rent for ("owners' equivalent rent") — is by far the largest single component, around a third of the whole index and an even larger share of *core*. Because shelter is measured with long lags (leases reset slowly, and the BLS samples each unit only every six months), it is the *stickiest* part of the index and the part the Fed worries about most: it does not bounce around month to month, so when shelter inflation is high it tends to *stay* high. A hot core print driven by shelter is therefore scarier to the market than a hot core driven by, say, used-car prices, which can reverse next month.

That is the seed of the components story: the market does not weight the CPI surprise uniformly. It up-weights the sticky, persistent categories (shelter, core services ex-housing — the so-called "supercore") and down-weights the volatile ones. A +0.2pp core surprise that came entirely from airfares and hotel rates can be faded; the same +0.2pp from rent and medical services cannot. Two prints with identical headline surprises can produce different reactions because their *composition* differs. The full decomposition — which components actually correlate with the market's reaction — is the subject of [core CPI, shelter, and supercore](/blog/trading/macro-correlations/core-cpi-shelter-and-supercore-what-actually-correlates); for our purposes the takeaway is that "the surprise" is really a *quality-weighted* surprise, not a raw one.

### How the consensus is built, and what "expected" really means

We keep saying the surprise is "actual minus consensus." But where does the consensus come from, and why should we trust that it is "in the price"? The published consensus is a survey: data providers poll dozens of professional economists for their CPI forecasts and report the median. That median is the market's official prior. But there is a subtler, more important prior — the *whisper number* and the *positioning*. Sometimes the published consensus is 0.3% but the trading desks have quietly convinced themselves it will be 0.4%; in that case a 0.4% print, an apparent "upside surprise" versus the published number, actually moves the market *down in surprise terms* because it was the whispered expectation. This is why the realized reaction sometimes contradicts the naive surprise sign: the relevant expectation is what the *market* priced, which can differ from the published survey.

For most purposes, the published consensus is a good-enough proxy, and the surprise computed against it predicts the sign of the reaction well. But it is the reason two analysts can look at the same print, compute the same arithmetic surprise, and disagree about whether the market "should" rally or sell — they are arguing about what was *really* priced. The deeper treatment of how priced expectations, not surveys, drive the reaction is in [the reaction function](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently).

## The surprise, not the level: the single most important idea in this post

Here is the mistake almost every beginner makes. They see CPI come in at 3.5% and reason, "inflation is high, that's bad, stocks should fall." Sometimes the market does fall. Often it rises. The beginner concludes that markets are irrational. They are not. The beginner is correlating the *wrong variable*.

Markets are forward-looking discounting machines. By the time a CPI number is released, the *expected* number is already baked into every price. The consensus forecast — the average of economists' predictions, published on every terminal before the release — is the market's best guess, and prices already reflect it. So a 3.5% print when everyone expected 3.5% contains *zero new information*. Nothing should move, and empirically almost nothing does.

What moves prices is the **surprise**: the gap between the actual number and the consensus.

```
surprise = actual - consensus
```

A 3.2% print when 2.9% was expected is a *hot* surprise (+0.3pp) and sells off risk assets even though 3.2% is a perfectly ordinary inflation rate. A 3.8% print when 4.1% was expected is a *cool* surprise (−0.3pp) and rallies risk assets even though 3.8% is high. The asset price correlates with the surprise, not the level. This is the master key, and it is why the same indicator produced a −4.3% S&P day and a +5.5% S&P day within two months: the *surprises* had opposite signs.

This idea is general — it applies to NFP, retail sales, GDP, every scheduled number — and it is developed in full in the sibling post [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) and in the event-trading primer [why news moves markets](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework). We invoke it here because nothing about CPI-and-asset-prices makes sense without it.

#### Worked example: why a "high" CPI rallied stocks

Suppose CPI prints at 3.7% year-over-year. A novice screams "high inflation, sell!" But the consensus forecast was 4.0%. The surprise is `3.7% − 4.0% = −0.3pp`: a *cool* surprise. Using the 2022-regime equity beta of roughly −0.7% per +0.1pp of core upside surprise (and noting the headline/core surprise here is in the cool direction), a −0.3pp surprise implies an equity move of about `−0.3 / 0.1 × (−0.7%) = +2.1%`. The S&P rallies despite a "high" 3.7% print. **Intuition: the market does not trade the number; it trades the number minus what it already knew.**

## The cross-asset correlation map of a CPI surprise

Now we can draw the centerpiece. When core CPI surprises to the upside, how does each asset respond in the same session? The chart below shows the empirical betas from the 2022-23 inflation-fear regime — the move per +0.1 percentage point of core CPI upside surprise.

![CPI surprise beta bars showing reaction per asset](/imgs/blogs/cpi-and-asset-prices-the-master-inflation-correlation-2.png)

Read it carefully, because every sign on this chart is load-bearing:

- **Stocks fall.** S&P 500 about −0.7%, Nasdaq 100 about −1.0% per +0.1pp surprise. Stocks are claims on future cash flows; a hotter inflation print means a higher discount rate, which lowers the present value of those cash flows. The Nasdaq falls *more* because it is concentrated in long-duration growth companies whose value sits further in the future, making them more sensitive to the discount rate. (Duration is not just a bond concept; an equity whose earnings are decades away behaves like a long bond.)
- **Bonds fall (yields rise).** The 10-year yield rises about +7bp, the 2-year about +9bp. A hot CPI means the Fed stays higher for longer, so yields reprice up. Bond *prices* move inversely to yields, so bonds fall. The 2-year moves *more* than the 10-year because it is closest to the Fed's policy rate — the front end is where rate-hike expectations live.
- **The dollar rises.** DXY about +0.35%. Higher U.S. yields make dollar assets more attractive relative to foreign assets, pulling capital in and lifting the currency. The dollar is the one green bar — the one asset that *benefits* from a hot U.S. inflation surprise.
- **Gold falls.** About −0.8%. This surprises people who think "gold is an inflation hedge, so hot CPI should help gold." It is the opposite, and the reason is real yields, which we devote a whole section to below.
- **Bitcoin falls hardest.** About −1.6%. In the 2022 regime, Bitcoin traded as the highest-beta risk asset on the board — a leveraged bet on liquidity and risk appetite. When a hot CPI tightened expected liquidity, crypto sold off more than anything else.

The single most important takeaway from this chart: **in the inflation-scare regime, a hot CPI was negative for almost every asset at once, and positive only for the dollar.** Diversification offered no protection because the correlation across risk assets went toward +1. That is the cross-asset transmission spelled out in [how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market); the new contribution here is putting numbers on each leg.

### Why one shock hits everything: the discount-rate channel

Step back and ask *why* a single inflation number can pull seven different assets the same way at the same instant. The answer is that almost every asset is priced off the same denominator: the discount rate. The value of any asset is the present value of its future cash flows, and the present value depends on the rate you discount those flows at. Raise the discount rate and you shrink the value of every future dollar — for a stock (future earnings), for a bond (future coupons and principal), for real estate (future rents), even, indirectly, for a non-cash-flow asset like gold or crypto whose appeal is relative to the yield you forgo by holding them.

A hot CPI surprise raises the *expected path of the policy rate*, which is the anchor for the whole discount-rate structure. That single move radiates through the denominator of every valuation at once. This is why the cross-asset correlation on a CPI day is so high: it is not seven independent reactions; it is one shock to the common denominator. The reason the *magnitudes* differ across assets — Nasdaq more than S&P, 2-year yield more than 10-year, Bitcoin most of all — is **duration sensitivity**: the further in the future an asset's value sits, and the more leveraged its exposure to liquidity, the more a discount-rate shock moves it. A long-dated growth stock and a 30-year bond are both "long duration" and both get hammered hardest. The dollar is the exception precisely because it is not a discounted-cash-flow asset; it is a relative-yield asset, and higher U.S. yields make it *more* attractive, so it rises.

Understanding the discount-rate channel is what lets you predict the *ordering* of the betas without memorizing them. Ask "how long-duration is this asset, and how exposed to liquidity?" and you can rank-order the reaction. That ordering is stable even when the magnitudes shift across regimes, which is why it is the most durable thing to internalize from this whole post.

#### Worked example: sizing an implied move from a surprise

A trader sees that core CPI is about to be released with a consensus of +0.3% month-over-month. The actual prints +0.5% — a +0.2pp upside surprise. Using the betas above, the *implied* same-session moves are:

- S&P 500: `+0.2 / 0.1 × (−0.7%) = −1.4%`
- Nasdaq 100: `+0.2 / 0.1 × (−1.0%) = −2.0%`
- 10-year yield: `+0.2 / 0.1 × (+7bp) = +14bp`
- 2-year yield: `+0.2 / 0.1 × (+9bp) = +18bp`
- Gold: `+0.2 / 0.1 × (−0.8%) = −1.6%`
- Bitcoin: `+0.2 / 0.1 × (−1.6%) = −3.2%`

On a \$100,000 60/40 portfolio (\$60k stocks, \$40k bonds), a −1.4% equity leg is about −\$840 and the bond leg falls too (a +14bp move on intermediate Treasuries is roughly −1% in price, about −\$400), for a combined hit near −\$1,240 — the diversifier did not diversify. **Intuition: a beta times a surprise gives you a back-of-envelope expected move, but only within the regime that produced the beta.**

A caution before you bet your account on those numbers: these are *average* reactions over a specific regime, with wide error bars. The actual move on any given day also depends on positioning, the reaction function (covered in [why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently)), and what the *components* of the print said. We will see the regime caveat is the big one.

## The regime that set the table: CPI from 2020 to 2026

Why do the betas above carry "2022-23 regime" stamped all over them? Because the magnitude of every CPI reaction depends on *how worried the market is about inflation in the first place*, and that worry was at a 40-year extreme in 2022. The chart below is the inflation path that set every correlation in this post.

![US CPI year over year path 2020 to 2026 with the 2022 peak](/imgs/blogs/cpi-and-asset-prices-the-master-inflation-correlation-3.png)

Inflation sat near or below the Fed's 2% target through 2020, then exploded through 2021 and into 2022, peaking at **9.06% in June 2022** — the highest reading in four decades. That single spike is the protagonist of the modern macro story. When inflation is running at 9% and the Fed is hiking in 75bp clips, *every* incremental CPI print is a referendum on whether the central bank will have to break the economy to regain control. In that world, a hot print is genuinely terrifying, and the betas are enormous.

Then look at what happened after. Inflation fell back toward 3% through 2023 and hovered in a 2.4%-3.0% band through 2024 and most of 2025 (with a late-2025/early-2026 re-acceleration that is its own live story). In that calmer world, a print that beats by 0.1pp is a curiosity, not a crisis. The market shrugs. The *same surprise* produces a fraction of the move.

This is the heart of the series' thesis — *correlation is a regime, not a constant*, the subject of [correlation is a regime not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant). The CPI betas are not laws of physics. They are conditional on the regime, and the regime is largely set by the inflation level itself.

#### Worked example: the same print, two worlds

Imagine the *identical* CPI release — core +0.5% versus a +0.3% consensus, a +0.2pp surprise — dropping in two different worlds.

- **June 2022 (inflation at 9%, peak scare regime):** equity beta about −0.7% per 0.1pp, so the implied S&P move is `+0.2 / 0.1 × (−0.7%) = −1.4%`, and in practice tail days like September 13, 2022 ran several times the average because positioning amplified the move to about −4.3%.
- **A calm 2.5% world (2024-style benign regime):** the effective equity beta shrinks to roughly −0.1% to −0.2% per 0.1pp, so the implied S&P move is closer to `+0.2 / 0.1 × (−0.15%) = −0.3%`. A rounding error.

Same surprise, roughly a 5- to 10-times difference in the realized move. **Intuition: the print sets the sign of the surprise; the regime sets the size of the reaction.**

## Regime dependence, made concrete

Let us make the conditional nature of the betas precise instead of waving at it. The figure below contrasts the reaction in the scare regime against the benign regime, asset by asset.

![Matrix of CPI reaction in scare regime versus benign 2 percent world](/imgs/blogs/cpi-and-asset-prices-the-master-inflation-correlation-4.png)

Two columns, four assets. In the 2021-23 scare (left, red), a hot core print sends the S&P down sharply, the 2-year yield jumping ~9bp, gold falling, and Bitcoin dropping ~1.6%. In the benign 2% world (middle, amber), the *same* surprise produces muted, sometimes near-zero moves: the S&P barely twitches, the 2-year inches up a couple of basis points, gold is flat, Bitcoin is soft but not crushed. The third column names *why* the size collapses — when inflation is anchored near target, the market reads a CPI beat partly as a *growth* signal rather than a pure rate-fear signal, which we will see can even flip the sign.

The mechanism behind the shrinkage is worth stating plainly. The CPI beta is large precisely when the market believes the Fed's policy response is *highly sensitive* to inflation data — i.e., when inflation is the dominant risk and every print could change the rate path. When inflation is parked at 2.4% and the Fed is on cruise control, a single 0.1pp beat does not change anyone's view of the policy path, so it does not reprice the discount rate, so it does not radiate into assets. The transmission pipe in figure 1 is still there; the *pressure* in it has dropped.

## The correlation that flips: stocks, bonds, and the inflation regime

The most consequential regime effect is not the *size* of the reaction but the *sign* of the cross-asset correlation — specifically the stock–bond correlation, the engine of the entire 60/40 portfolio. The chart below shows it.

![Stock bond correlation by inflation regime and the equity CPI surprise correlation sign flip](/imgs/blogs/cpi-and-asset-prices-the-master-inflation-correlation-5.png)

Read the left panel first. When average core inflation sits below 2%, stocks and bonds are *negatively* correlated (r ≈ −0.45): when stocks fall, bonds rally, so a bond allocation cushions an equity drawdown. This is the "diversifying" property that made the 60/40 portfolio a free lunch for two decades — the negative correlation, the holy grail of [the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine). But as inflation rises through the 3-4% band and above 4%, the correlation climbs and turns *positive* (r ≈ +0.50 above 4%). Now stocks and bonds fall *together*. The diversifier vanishes exactly when you need it most.

Why does CPI sit at the center of this flip? Because the driver of the stock–bond correlation is *what kind of shock dominates the market*. In a low-inflation world, the dominant shock is growth: a recession scare hurts stocks but helps bonds (the Fed cuts, yields fall), so they move oppositely. In a high-inflation world, the dominant shock is *inflation/rates*: a hot CPI hurts stocks (higher discount rate) *and* hurts bonds (higher yields) at the same time, so they move together. CPI is the variable that determines which regime you are in. In 2022, with CPI at 9%, the stock-bond correlation spiked to its most positive in decades and the 60/40 portfolio had one of its worst years on record — both legs down hard, on the same CPI prints.

The right panel makes the point at the print level: the correlation between equity returns and CPI *surprises* is mildly *positive* (r ≈ +0.10) in a low, stable inflation regime — a hot print can be read as a good-growth signal — and strongly *negative* (r ≈ −0.45) in a rising or high regime, where rate-fear dominates. The sign of the relationship between stocks and inflation surprises is not fixed; it inverts with the regime. That inversion is the whole story of the sibling post [inflation and stocks, the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips).

#### Worked example: when the diversifier failed

A retiree holds a classic \$1,000,000 60/40 portfolio: \$600,000 in the S&P 500, \$400,000 in long Treasuries. In a *normal* regime, a bad equity month of −5% (−\$30,000) would typically be partly offset by a bond rally — say bonds up 2% (+\$8,000) — for a net of about −\$22,000. But in 2022's hot-CPI regime, the correlation flipped positive: on the worst CPI days, the −5% equity leg (−\$30,000) was *accompanied* by a −3% bond leg (−\$12,000), for a net of −\$42,000 — nearly double the "normal" loss, and the bonds *added* to the pain instead of cushioning it. **Intuition: a positive stock-bond correlation turns your safe asset into a second risky asset, and high CPI is what turns the correlation positive.**

## The gold puzzle: why "the inflation hedge" falls on hot CPI

The single most counterintuitive sign on the centerpiece chart is gold's. "Gold is the classic inflation hedge," the story goes, "so a hot CPI — proof of more inflation — should send gold up." Empirically, gold *falls* on a hot core CPI surprise (about −0.8% per 0.1pp in the 2022 regime). Why?

Because gold does not correlate with inflation. **Gold correlates, strongly and negatively, with real yields.** A real yield is the nominal yield minus expected inflation — the return you earn *after* inflation eats its share. Gold pays no interest and no dividend; its opportunity cost is whatever you could have earned in a safe, inflation-protected bond (a TIPS). When real yields rise, holding a zero-yield rock becomes more expensive, and gold falls. When real yields fall (or go negative), the rock looks great, and gold rises.

Now connect the wires. A hot CPI surprise makes the market expect *more Fed hikes*. The Fed hikes nominal rates faster than inflation expectations rise, so the *real* yield goes *up*. Higher real yield → gold down. The hot inflation print hurt gold not despite being inflationary, but *because* the policy response to inflation pushed real yields up. The "inflation hedge" framing has the right asset and the wrong mechanism. This is important enough that it gets its own sibling post, [inflation and gold, the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story), and it leans on the macro-trading master signal [real vs nominal, real yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal). Hold the one-liner: *gold hedges real yields, not CPI.*

The corollary is that gold's CPI beta is the *least stable* of any asset, because it depends on whether a hot print pushes real yields up (gold down) or whether the market reads the print as the Fed losing control and demanding a flight to hard assets (gold up). In 2022, real yields dominated and gold fell on hot prints. In other episodes, the "store of value" demand dominated. Of all the betas in this post, treat gold's with the most humility.

## Crypto: the highest-beta read on the same shock

Bitcoin's −1.6% beta is the largest in the chart, and the explanation is that in the 2021-22 regime, Bitcoin traded as a pure, leveraged proxy for risk appetite and liquidity. It had no cash flows to discount and no central bank, so it was not reacting to fundamentals; it was reacting to the *same liquidity signal* that moves the Nasdaq, only with more leverage and more retail reflexivity. When a hot CPI tightened the expected liquidity tide, the highest-beta boat fell furthest.

The crucial nuance — and a theme of this whole series — is that this correlation is *time-varying*. Bitcoin's correlation with the Nasdaq was near zero before 2020, spiked to roughly +0.6 to +0.7 in 2022 as it became a macro-liquidity asset, and then *faded* back toward +0.2 to +0.3 by 2024-25 as the asset matured and developed its own idiosyncratic drivers (ETF flows, halving cycles, regulatory news). So Bitcoin's CPI beta was huge in 2022 and is much smaller now — not because CPI changed, but because crypto's *relationship to the macro* changed. A correlation you measured in 2022 and apply blindly in 2025 will mislead you. That decay is exactly the kind of trap catalogued in [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data).

## The dollar: the one asset on the other side

Every other asset on the centerpiece chart falls on a hot CPI; the dollar rises. It is worth dwelling on *why* the dollar sits on the opposite side, because it makes the dollar the single most useful instrument for hedging inflation-surprise risk, and because it reveals something deep about how the cross-asset map is wired.

A currency is not a discounted-cash-flow asset. You do not value the dollar by discounting future earnings; you value it *relative to other currencies*, primarily through the *interest-rate differential* — how much you earn holding dollars versus holding euros or yen. A hot U.S. CPI raises expected U.S. rates relative to the rest of the world, which widens that differential in the dollar's favor and pulls global capital toward dollar assets. Capital chasing the higher yield bids the dollar up. So the same shock that hurts every cash-flow asset (by raising the discount rate) *helps* the dollar (by raising the relative yield it offers). The dollar is mechanically the mirror image of the risk-asset complex on a rate shock, which is why its hot-CPI correlation is a clean +0.45 while almost everything else is negative.

This is also why the dollar is "cross-asset gravity," the subject of [the dollar, cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity). A stronger dollar — itself often the *product* of a hot CPI — adds a *second* headwind to commodities, emerging-market assets, gold, and crypto, all of which are priced in or sensitive to the dollar. So a hot CPI hits these assets twice: once directly through the rate channel, and again indirectly through the stronger dollar it produces. That double-hit is part of why gold and EM equities carry such reliably negative signs. The practical consequence for a trader is direct: if you want one instrument that profits when a hot CPI hurts your risk book, long dollar exposure is the most reliable single hedge on the board, precisely because it is the only asset structurally on the other side of the shock.

## Lead and lag: CPI as a coincident inflation read

Every correlation in this series has not just a sign and a strength but a *timing* — does the indicator lead the asset, lag it, or move with it? For CPI the answer is mostly *coincident with the policy read but with a few useful leads and lags around it*, and knowing them helps you avoid double-counting information.

CPI is roughly coincident with the Fed's preferred inflation gauge, the PCE deflator, but it arrives about two weeks *earlier*, so for trading purposes CPI is the *leading read on PCE*. When CPI surprises hot, the market immediately marks up its PCE expectation; by the time PCE actually prints, much of the news is stale. This is why CPI moves markets more than PCE despite PCE being the official target — CPI got there first. The lead is small (effectively a couple of weeks of calendar) but it is the difference between trading on news and trading on confirmation.

Upstream of CPI sits the Producer Price Index (PPI), which measures prices at the wholesale/producer level rather than the consumer level. PPI leads the *goods* portion of core CPI by roughly a month: when producers' input and output prices jump, those costs filter into consumer goods prices with a short lag. A trader who watched PPI spike to its 11.7% peak in March 2022 had an early warning that consumer goods inflation would stay hot. PPI is therefore a *leading indicator* for one slice of CPI, and a hot PPI raises the odds of a hot CPI a few weeks later — though the relationship is loose, because services (the bigger and stickier part of CPI) are not well captured by PPI.

Downstream, the shelter component *lags* market rents by roughly a year, because of the slow lease-reset measurement we discussed. This creates one of the most exploited patterns of the 2023-24 disinflation: traders watched real-time rent indices roll over and *knew* that the shelter component of CPI would mechanically cool over the following twelve months, even before it showed up in the official print. The lag made part of future CPI semi-predictable, which dampened the surprise content of those prints. When a component of an indicator is predictable, its surprise — and therefore its market impact — shrinks. The full lead/lag map across all the macro indicators is the subject of [leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators); the CPI-specific lesson is that the *surprise content* of a print depends on how much of it was already forecastable from upstream (PPI) and lagged (shelter) inputs.

#### Worked example: discounting a predictable shelter cooling

In late 2023, market rent indices had been falling for months, implying the shelter component of CPI — running near +0.5% month-over-month — would mechanically decelerate toward +0.3% over the next year. Shelter is about 0.42 of core CPI by weight. If shelter slows from +0.5% to +0.3% (a −0.2pp drop) and nothing else changes, the *mechanical* contribution to core CPI falls by about `0.42 × 0.2pp ≈ 0.08pp` per month. A trader who priced that in *expected* core to drift down, so an in-line print confirming the cooling produced little surprise and little market reaction, while a print that *failed* to cool became a bigger upside surprise than the raw number suggested. **Intuition: the market reacts to the part of a print it could not have forecast, so a predictable, lagged component carries less surprise — and less market impact — than its size alone implies.**

## One row of the master matrix

This whole post is, in effect, a deep zoom on a single row of the series' master correlation map. The full matrix — every macro driver against every asset — is the subject of [the macro asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix). Here is the CPI row, lifted out and laid flat.

![Heatmap row of hot CPI surprise correlation across assets](/imgs/blogs/cpi-and-asset-prices-the-master-inflation-correlation-6.png)

The colors encode sign and strength: red is negative correlation, green is positive, white near zero. A hot CPI surprise is negative for the S&P (−0.55), more negative for the Nasdaq (−0.65), very negative for bonds (−0.75, the bond is most directly a rates instrument), mildly negative for gold (−0.10, weak because the real-yield channel is noisy), negative for Bitcoin (−0.55) and EM equity (−0.45), and *positive* for the dollar (+0.45). One row, one shock, the whole cross-asset fingerprint.

Notice how this matches the beta chart but is not identical to it. The beta chart was the *size* of the same-session move in the 2022 regime; this row is the *correlation* (sign and reliability) across a broader sample. Gold's −0.10 here versus its −0.8% beta in 2022 is the clearest tell: the *sign* is reliably mildly negative, but the *magnitude* is regime-specific and the relationship is the noisiest on the row. Reading the matrix correctly means reading both the color (sign, reliability) and remembering that the size is conditional.

#### Worked example: building a CPI-day hedge from the row

A portfolio manager is long \$500,000 of the Nasdaq 100 into a CPI print and wants to hedge the inflation-surprise risk without selling. The row says the Nasdaq's hot-CPI correlation is −0.65 and the dollar's is +0.45 — they move oppositely on the shock. The manager buys \$200,000 of dollar exposure (long DXY). If a +0.2pp hot surprise lands, the Nasdaq leg loses about `0.2 / 0.1 × (−1.0%) × \$500,000 = −\$10,000`, while the dollar leg gains about `0.2 / 0.1 × (+0.35%) × \$200,000 = +\$1,400` — a partial offset of roughly 14%. To neutralize more, scale the dollar leg up or add a long-2-year-yield (short bond) position, since bonds carry the most negative sign (−0.75). **Intuition: the matrix row tells you which assets move opposite to your risk on the shock, which is exactly the recipe for an event hedge.**

## How you would actually measure the CPI beta

It is worth being concrete about where the betas in this post come from, both so you trust them appropriately and so you can re-measure them when the regime shifts (which, by now, you know you must). The method is an *event study*. You collect every CPI release date over a window, and for each one you record two numbers: the *surprise* (actual core month-over-month minus the published consensus, in percentage points) and the *same-session asset return* (the close-to-close, or release-window, move in each asset). That gives you a scatter of points — surprise on the horizontal axis, asset return on the vertical — one point per CPI day.

You then fit a line through the cloud. The *slope* of that line is the beta: the average asset move per unit of surprise. The *correlation coefficient* of the cloud tells you how tight the relationship is — how reliably the asset follows the surprise versus how much idiosyncratic noise swamps it. A beta of −0.7%/0.1pp with an r of −0.6 is a usable signal; the same beta with an r of −0.2 means the relationship is real on average but drowned in noise on any given day, so you should not bet heavily on a single print. This is exactly the scatter-plus-regression construction that is the signature chart of this whole series, developed in [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises).

Two methodological warnings keep you honest. First, the *window* matters enormously: a beta estimated over 2021-2023 is a scare-regime number; the identical method run over 2015-2019 gives much smaller betas and, for stocks, sometimes a positive sign. There is no single "the CPI beta" — there is the beta in *this* regime, over *this* window. Second, full-sample betas average across regimes and therefore describe *no* regime accurately; they are the statistical equivalent of saying the average temperature of a person with one foot in ice and one in boiling water is comfortable. Always condition on the regime, and always quote the window you measured over. Both pitfalls — window choice and regime mixing — are catalogued in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) and [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data).

## Common misconceptions

**"CPI is high, so I should be bearish stocks."** No — the *level* is in the price already; only the *surprise* moves markets, and the surprise can rally stocks even at a high level. Recall the −0.3pp cool surprise that implied a +2.1% equity move at a 3.7% print. Trade the surprise, not the headline.

**"Gold is an inflation hedge, so hot CPI is good for gold."** No — gold tracks *real yields*, not CPI, with a real-yield correlation around −0.8 over 2007-2021. A hot CPI typically pushes real yields *up* (more expected hikes), which sends gold *down*. The hot 2022 prints sold gold off. The hedge is against negative real yields, not against inflation per se.

**"The stock–bond correlation is negative, so 60/40 always diversifies."** No — it is negative only in low-inflation regimes (about −0.45 below 2%) and turns *positive* (about +0.50 above 4%) when inflation dominates. In 2022 both legs fell on the same CPI prints. Your diversification is conditional on the very regime CPI defines.

**"A bigger CPI number always moves markets more."** No — a hot *headline* driven entirely by a gasoline spike, with a soft *core*, often gets faded, because core is the signal the Fed targets. The market weights the core surprise and the components, not the raw headline. Which components matter is the subject of [core CPI, shelter, and supercore](/blog/trading/macro-correlations/core-cpi-shelter-and-supercore-what-actually-correlates).

**"Bitcoin is uncorrelated, so it diversifies CPI risk."** No — in the 2022 regime Bitcoin had the *highest* CPI beta (about −1.6%) and a +0.6 to +0.7 correlation with the Nasdaq; it was the *most* exposed to the inflation shock, not the least. The correlation has since faded toward +0.25, but "uncorrelated" was never true in the regime that mattered.

## How it shows up in real markets

Let us walk the two anchoring days in detail, plus the regime backdrop, because the dated cases are where the abstractions become money.

**September 13, 2022 — the hot print.** August core CPI rose 0.6% month-over-month against a roughly 0.3% consensus — a large upside surprise, about +0.3pp — and it came when the market had positioned for a peak. The reaction was the full cross-asset fingerprint at maximum amplitude: S&P 500 about −4.3%, Nasdaq 100 worse, the 2-year yield up roughly 17bp to fresh cycle highs, the dollar sharply higher, gold and Bitcoin both down. Every red bar on the beta chart fired at once and several times the average size, because the regime (CPI at ~8%, the Fed mid-hiking-cycle) maximized the betas and the positioning (everyone leaning long into a hoped-for peak) maximized the unwind. This is the canonical "hot CPI is bad for everything" day.

**November 10, 2022 — the cool print.** October core CPI rose 0.3% against a 0.5% consensus — a downside surprise of about −0.2pp — and it landed two months later into a market braced for more pain. The fingerprint inverted at full amplitude: S&P 500 about +5.5%, Nasdaq 100 up more than 7%, the 10-year yield down nearly 30bp, gold and Bitcoin sharply higher. Same indicator, opposite surprise, opposite sign on every asset, similar enormous magnitude — because the regime was still hot enough to make the betas large. This pair, two months apart, is the cleanest natural experiment you will ever see for "the surprise sets the sign."

**The asymmetry.** Notice the cool day (+5.5%) was actually *larger* than the hot day (−4.3%). Reactions to inflation prints are often asymmetric: in a regime where the market desperately wants confirmation that inflation has peaked, a cool surprise (relief) can produce an outsized rally — the "pain trade" of a short-covering, FOMO-fueled squeeze — while hot surprises, though brutal, can be partly anticipated by a nervous, already-defensive market. The asymmetry is not a fixed law; it depends on positioning and on which outcome the market is more "surprised and relieved" by. The practical lesson is to never assume the up-move and down-move are mirror images of equal size.

The mechanism behind that particular asymmetry is positioning. By November 2022 the market was heavily short and defensively positioned after a brutal year; a cool print did not just remove a headwind, it forced a wave of short-covering and chase-the-rally buying on top of the fundamental repricing, which is why the up-move overshot. A symmetric event-study beta — fitted to all CPI days at once — would have *underpredicted* the cool-day rally and roughly matched the hot-day selloff, precisely because the betas average over positioning states they cannot see. This is a recurring limitation of any single-number beta: it captures the average response but misses the state-dependence (how crowded, how defensive, how exhausted the market already is) that determines whether a given print produces an average move or a violent overshoot. When you size a CPI trade, the beta gives you the base case; the positioning tells you which way the tail risk leans.

**The both-fell regime.** Zoom out from single days to the whole of 2022, and you see the regime effect at the portfolio level. As CPI ran at 8-9% all year, the stock-bond correlation sat near its most positive in decades (the +0.60 reading for 2022 in this series' [stock-bond data](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine)). The S&P fell about 19% on the year and long Treasuries fell *more* — both legs of the 60/40 down hard, repeatedly on the same CPI prints. That is the lived experience of a positive stock-bond correlation, and CPI was the variable that produced it. When correlations go to one and the diversifier fails, as catalogued in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis), a hot inflation regime is one of the most reliable causes.

**The fading-beta regime (2023-24).** The most instructive episode for this post's thesis is what happened *after* the scare. As CPI fell from 9% in mid-2022 back toward 3% through 2023 and then drifted in a 2.4%-3.0% band through 2024, the CPI betas visibly shrank. CPI days that would have produced 3-4% S&P swings in 2022 produced fractions of a percent in 2024. A trader who kept applying 2022-sized betas in 2024 would have systematically over-hedged into prints and over-traded the reactions, bleeding on transaction costs and slippage for a signal that had decayed. The *sign map* stayed broadly intact — hot still meant risk-off — but the *size* collapsed as inflation stopped being the dominant risk. This is the single most expensive mistake in practice: assuming a correlation measured in the regime where it was huge still holds in the regime where it has faded. The beta did not "stop working"; it returned to its calmer, structural level, exactly as the regime framework predicts.

**The late-2025/2026 re-acceleration.** And the story is not over: CPI re-accelerated toward and past 4% in early 2026 (the live tail of the path in figure 3). A trader watching that re-acceleration should have done the regime check in reverse — as inflation climbed back above 4%, the betas should *re-inflate* toward their scare-regime sizes, the stock-bond correlation should turn positive again, and CPI days should regain their power to move everything at once. The framework is symmetric: it does not just explain why the beta shrank in 2024; it tells you when to expect it to come roaring back. Treating the betas as conditional, and re-checking the regime on every cycle, is the whole discipline.

## How to read it and use it

Here is the playbook, distilled into a decision flow. The figure below is how a professional reads a CPI print in real time.

![Decision graph for reading a CPI print into a trade](/imgs/blogs/cpi-and-asset-prices-the-master-inflation-correlation-7.png)

Walk it left to right:

1. **Read core, not headline.** The first thing on the tape is the headline number; ignore it until you have the *core* month-over-month figure. The core surprise is what carries the signal. Then glance at the components — was the beat from sticky services/shelter (durable, scary) or from a volatile category (fadeable)?
2. **Compute the surprise.** `actual − consensus`. The consensus is on every terminal before the release. The *sign* of this number is the sign of your expected reaction. A hot surprise (actual above consensus) is risk-off; a cool surprise is risk-on.
3. **Check the regime.** Is inflation running above ~4% (scare regime) or anchored near 2% (benign)? This sets the *size*. In a scare regime, scale the beta up several-fold; in a benign regime, expect a muted move and beware that a hot print might even be read as a positive growth signal, flipping the sign.
4. **Map to assets.** Apply the cross-asset fingerprint: hot surprise → stocks down (Nasdaq most), bonds down (2-year most), dollar up, gold down (via real yields), crypto down most. Cool surprise → the mirror image.

**The signal:** the core CPI surprise, scaled by the regime, predicts the same-session cross-asset move with a known sign map and a regime-dependent size.

**The regime check (the thing that invalidates everything):** the betas in this post are stamped "2022-23 inflation-scare regime." They are *not* constants. Before applying any number here, confirm you are in a comparable regime. If inflation is anchored at 2% and the Fed is on autopilot, the betas shrink toward zero and the equity-vs-inflation correlation can even turn mildly positive. The fastest way to be wrong is to take a 2022 beta and apply it in a 2025 world.

**What invalidates the signal, concretely:**
- *Regime shift.* Inflation falling from 8% to 2.5% collapses the betas. Re-measure on a recent rolling window; do not trust a full-sample number, for the reasons in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters).
- *A bigger event on the same day.* If CPI lands the morning of an FOMC decision or a major geopolitical shock, the CPI signal is swamped. Context dominates.
- *Crowded positioning.* If everyone is already leaned the same way, the surprise can produce an outsized move in the *opposite* direction to the one the beta predicts (a squeeze). The November 2022 +5.5% day was partly that.
- *Component divergence.* A hot headline with a soft core, or vice versa, breaks the simple beta. Always read what drove the print.

The disciplined way to use CPI is therefore not "memorize the betas and bet the print." It is: read the *core surprise* for the sign, check the *regime* for the size, map to the *cross-asset fingerprint*, and respect the *invalidators* — and never forget that the most market-moving number on the calendar is also one whose correlations move under your feet.

## Further reading and cross-links

Within this series:

- [The surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — the general surprise framework this post applies to CPI.
- [Core CPI, shelter, and supercore: what actually correlates](/blog/trading/macro-correlations/core-cpi-shelter-and-supercore-what-actually-correlates) — which *components* of the print carry the signal.
- [Inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips) — the U-shape and the ~3-4% threshold where the sign turns.
- [Inflation and gold: the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story) — why gold hedges real yields, not CPI.
- [The macro asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — the full map this post is one row of.
- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) and [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) — why every number here is conditional.

For the mechanism and the intraday reaction (do not re-derive — cite):

- [Interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [how policy moves every asset](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) — the causal chain from rates to assets.
- [Real vs nominal: inflation and real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — the real-yield channel behind gold and growth equities.
- [CPI: the report that moves the world](/blog/trading/event-trading/cpi-the-report-that-moves-the-world) and [the reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) — the minute-by-minute release reaction.
- [Cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market) — the single-shock, many-assets picture.
- [The 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) and [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — the stock-bond flip at the portfolio level.
