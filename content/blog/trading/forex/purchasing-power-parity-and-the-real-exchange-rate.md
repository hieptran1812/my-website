---
title: "Purchasing Power Parity and the Real Exchange Rate: Why a Big Mac Tells You More Than the Screen Price"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How the law of one price, absolute and relative PPP, and the real exchange rate explain why some currencies are cheap and others dear — and why that anchors decades but never tells you what happens tomorrow."
tags: ["forex", "currencies", "purchasing-power-parity", "ppp", "big-mac-index", "real-exchange-rate", "reer", "valuation", "law-of-one-price", "macro"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Purchasing power parity says an exchange rate should eventually settle where a basket of goods costs the same in both countries; it is the long-run gravity of currencies, but it is useless for timing because over months and years rates are pushed around by interest-rate gaps, flows, and positioning instead.
>
> - The **law of one price** says one good must cost the same everywhere once you convert currencies — because any gap is an arbitrage. Stack the whole shopping basket and you get **PPP**.
> - **Absolute PPP** predicts the exact rate from price levels and almost never holds; **relative PPP** only predicts the *change* from the inflation gap and is a much better compass.
> - The **real exchange rate** strips inflation out of the quoted rate; **PPP is just the claim that the real rate reverts toward a constant.** The trade-weighted version is the **REER** — the single best gauge of how dear a currency is.
> - The one number to remember: in January 2025 a Big Mac said the **Japanese yen was about 41% undervalued** versus the dollar, and the yen *still* kept falling — that is PPP in a sentence.

## A burger, two prices, and a thirty-year-old joke

In 1986 *The Economist* published what it called a "light-hearted" guide to whether currencies were trading at the "right" level. The idea was almost a dare: take one product that is made and sold in roughly the same way all over the world — a McDonald's Big Mac — and compare what it costs, in dollars, in each country. If a Big Mac costs \$5.69 in the United States but only the dollar-equivalent of \$3.36 in Japan, then either Japanese burgers are mysteriously cheap forever, or the yen is *undervalued* — too few dollars buy too many yen, so everything priced in yen looks like a bargain to a visitor holding dollars.

The joke kept being right in spirit and wrong in timing, which is exactly why it became famous. For decades the Big Mac index has flagged the same lineup: the Swiss franc dear, the Scandinavian currencies dear, the yen and most emerging-market currencies cheap. And for decades the cheap currencies have *stayed* cheap — sometimes getting cheaper for years before snapping back. The index is a magnet that everyone can feel but no one can use to set their watch by.

That tension is the whole subject of this post. There is a deep, almost mechanical economic law underneath the burger — the **law of one price** — and a family of theories built on it called **purchasing power parity**, or **PPP**. PPP is one of the four or five ideas you genuinely need to read a currency. But it comes with a giant warning label, and most people who get burned by it ignored the label. An exchange rate is the relative price of two monies, and PPP tells you where that price *belongs* in the long run. It tells you almost nothing about where it will be next week — because in the short run a currency is driven by the gap between two countries' interest rates plus the flow of money across borders, not by the price of lunch.

![Big Mac Index horizontal bar showing which currencies are over and undervalued versus the dollar in January 2025](/imgs/blogs/purchasing-power-parity-and-the-real-exchange-rate-1.png)

Look at that chart and you already have the plot of the entire post. Switzerland sits far to the right — the franc is roughly 39% *dearer* than the dollar on burger-PPP. The yen, the yuan, the rupiah, the rupee, and the Vietnamese dong sit far to the left — 40% to 50% *cheaper*. The euro and the pound hug the middle, within a few percent of the dollar. That spread is real, it is persistent, and by the end of this post you will understand exactly what it measures, why it persists, and — crucially — why a trader who shorted the dollar against the yen in 2021 because "the yen is cheap on PPP" would have lost a fortune over the next three years before being right.

## Foundations: The law of one price and PPP

Everything in this post is built from one stubbornly simple idea, so let's define it from zero and never let go of it.

### The law of one price

The **law of one price** states that an identical good, sold in two places, must sell for the same price once you express both prices in the same currency — otherwise there is free money lying on the table, and someone will pick it up. The "someone" is an **arbitrageur**: a trader who buys where a thing is cheap and sells where it is dear, pocketing the difference. The act of doing that *moves the prices*: buying in the cheap market pushes its price up, selling in the dear market pushes its price down, and they meet in the middle.

Suppose gold trades at \$2,000 per ounce in London and \$2,050 per ounce in New York at the same instant. You buy in London, sell in New York, and make \$50 an ounce minus shipping and financing. So does everyone else, until the London price rises and the New York price falls and the gap closes. The law of one price is not a law of nature; it is a *prediction about what arbitrage does to prices when a good can be moved freely.* For something like gold — dense, valuable, identical, and cheap to ship — it holds to within pennies. For a Big Mac — perishable, made of local beef and local labor and local rent — it holds terribly, and that failure is itself one of the most informative things in this whole subject.

It is worth being precise about the chain of reasoning, because the same chain reappears in every parity relationship in foreign exchange. The law of one price needs three ingredients to bite: the good must be **identical** (a London ounce of gold is the same as a New York ounce); it must be **tradable** at low cost (you can ship gold for a fraction of a percent of its value); and the arbitrage must be **executable** without large frictions (no capital controls, no quotas, no ruinous taxes). Where all three hold — gold, oil, copper, large-cap stocks dual-listed on two exchanges, government bonds — prices line up to within the cost of arbitrage and stay there. Where any one fails, a wedge opens and persists. A Big Mac fails the second and third tests spectacularly: you cannot ship a hot burger from Jakarta to Geneva, and even if you could, the beef is the cheap part — the expensive part is the Geneva real estate the restaurant sits on, which is the most untradable thing in the world. The size of the burger's price gap across countries is therefore a *measure of how untradable a basket really is*, and that, surprisingly, is what makes it informative about whole-economy price levels.

A subtle but important point: the law of one price is a statement about *prices*, but it has a twin that is a statement about *exchange rates*. If the good's local price is sticky (a Tokyo burger doesn't change price just because the yen moved this morning), then the only thing left to adjust is the exchange rate. So a violation of the law of one price can resolve in two ways — the prices converge, or the currency converges — and for currencies it is almost always the second channel that matters over the horizons we care about. The burger price in Tokyo barely budges from year to year; it is the yen that does the moving. This is why PPP is fundamentally an exchange-rate theory dressed up as a price theory: the prices are sticky, so the currency is the variable that absorbs the imbalance.

![Law of one price pipeline showing arbitrage buying cheap, selling dear, and dragging prices together](/imgs/blogs/purchasing-power-parity-and-the-real-exchange-rate-2.png)

Notice the last box in that diagram — the frictions. Shipping costs, import tariffs, the rent on the restaurant, the wage of the person who flips the burger: none of these can be arbitraged across borders. You cannot ship a Tokyo waiter to New York to exploit a wage gap. About half of what you pay for a Big Mac is **non-traded** — local labor and local property — and non-traded things break the law of one price wide open. Hold that thought; it is the single most important reason PPP fails in rich-versus-poor comparisons.

### From one good to the whole basket: absolute PPP

Now stack many goods. Take a representative **basket** — the same bundle of food, clothing, housing, transport, and services that a typical household buys — and ask what it costs in each country. The price of that basket *is* the **price level**, the thing inflation measures. **Absolute purchasing power parity** is the law of one price applied to the entire basket: the exchange rate should equal the ratio of the two countries' price levels.

In symbols, if `S` is the exchange rate (units of home currency per unit of foreign currency), `P` is the home price level, and `P*` is the foreign price level, then absolute PPP says:

```
S  =  P / P*
```

Read it in words: the exchange rate is whatever number makes one basket cost the same in both countries. If a basket costs \$100 in the US and ¥14,000 in Japan, absolute PPP says the "fair" exchange rate is 14,000 ÷ 100 = ¥140 per dollar — because at that rate, \$100 converts to exactly ¥14,000 and buys the same basket. If the market rate is instead ¥157 per dollar, then \$100 converts to ¥15,700, which *overbuys* the Japanese basket — the yen is cheap, Japan is a bargain for dollar-holders, and the Big Mac index lights up red.

Absolute PPP is the strong, clean, beautiful version of the theory. It is also almost always wrong about the *level*, for the non-traded-goods reason above plus quality differences, taxes, and the simple fact that poorer countries have cheaper labor and therefore cheaper services. We'll quantify that failure shortly.

There is a deep reason absolute PPP is the version everyone *learns* and almost no one *uses*. To compute it you need the price of an identical basket in both countries — and the moment you ask "which basket?", the theory wobbles. Americans and Japanese consume different things in different proportions; the statistical agencies that build their price indices use different weights, different sampling, different quality adjustments. The number you get for "the US price level" and "the Japanese price level" depends on choices that have nothing to do with currency markets. The Big Mac index is so beloved precisely because it sidesteps this: a Big Mac is *defined* identically everywhere, so there is exactly one basket and no weighting argument. The price you pay for that cleanliness is that one burger is a laughably small and noisy basket — but at least it is the *same* basket, which is more than the official PPP figures can fully claim.

It also matters that absolute PPP is a statement about a *snapshot*, while currencies are a *flow* of trades. Even if the basket were perfectly comparable and perfectly tradable, the exchange rate is set at the margin by whoever is buying and selling currency *today* — importers, exporters, tourists, but overwhelmingly by financial flows chasing yield and safety. Those financial flows dwarf the trade flows that PPP arbitrage works through. The Bank for International Settlements measures over \$7.5 trillion a day of currency turnover, the vast majority of it financial rather than goods-related. Goods arbitrage is a trickle against that flood, which is the structural reason PPP is a slow tide rather than a fast current.

### The more useful cousin: relative PPP

Because absolute PPP fails on the level, economists lean on a weaker but far more reliable version. **Relative PPP** drops the claim that the rate equals the price ratio *today* and instead claims that the rate *changes* by the difference in the two countries' inflation rates over time.

The intuition: if frictions (shipping, tariffs, the rent gap) are roughly *constant*, they cancel out of the *change* even though they wreck the *level*. So:

```
percent change in S  ≈  home inflation  −  foreign inflation
```

The currency of the higher-inflation country depreciates by roughly the inflation gap. If US prices rise 3% a year and Japanese prices are flat, relative PPP says the dollar should *weaken* about 3% a year against the yen — because each year a dollar buys less stuff at home, so it should buy fewer yen too, keeping their real purchasing power aligned. This is the version that actually anchors decades of currency moves, and it is the version a serious analyst uses.

Why does dropping the level claim rescue the theory? Because the frictions that destroy absolute PPP — the rent gap, the wage gap, the tariff, the non-traded services — are *roughly constant over time*. Geneva will always be more expensive than Jakarta; that gap doesn't arbitrage away, but it also doesn't keep *growing*. When you look at the *change* in the exchange rate rather than its level, those constant frictions cancel out, leaving only the part that *does* move: the difference in inflation. It is the same trick as measuring a child's growth in centimeters per year rather than arguing about how tall they "should" be — the level is contaminated by a thousand factors, but the rate of change is clean. Relative PPP says: forget where the rate "should" be in absolute terms; just expect it to drift at the speed of the inflation gap.

The classic stress test for relative PPP is a high-inflation country. Argentina, Turkey, and Vietnam have all run inflation far above the United States' for years, and in every case the nominal currency depreciated against the dollar at *roughly* the cumulative inflation gap — the peso, the lira, and the dong all weakened in long, grinding slides that tracked their excess inflation. The fit is loose year to year and tight over a decade. That is the signature of relative PPP: invisible in the noise of any single quarter, undeniable across a full cycle. When someone tells you "the lira always goes down," what they are really describing is relative PPP enforced by Turkey's chronic inflation — the currency *has* to fall to keep Turkish goods priced sensibly in dollars.

![Before and after comparison of absolute PPP predicting the level versus relative PPP predicting only the change](/imgs/blogs/purchasing-power-parity-and-the-real-exchange-rate-3.png)

### The real exchange rate: what PPP is really about

Here is the concept that ties everything together and that most beginners skip — to their cost. The number on your screen, ¥157 per dollar, is the **nominal exchange rate**: a raw price quote. It tells you how many yen you get for a dollar, but it says nothing about what those yen *buy*. To know whether Japan is genuinely cheap, you have to adjust the nominal rate for the two countries' price levels. That adjustment gives you the **real exchange rate**.

The real exchange rate `q` is defined as:

```
q  =  S × (P* / P)
```

where `S` is the nominal rate (home per foreign), `P` is the home price level, and `P*` is the foreign price level. In words, the real rate measures **how many foreign baskets one home basket can buy.** It is a pure measure of relative purchasing power, with the currency units divided out. When the real rate is high, the home currency is "strong" in a real sense — your money buys a lot of foreign stuff, your exports look expensive, imports look cheap. When it's low, the home currency is "weak" in real terms — your exports are competitive and tourists flood in.

PPP, stated cleanly, is nothing more than this: **the real exchange rate should be constant** (or revert to a constant) in the long run. Absolute PPP says `q = 1`; relative PPP says `q` doesn't *drift* even if it isn't exactly 1. Every Big Mac headline, every "the yen is undervalued" take, every valuation model in this series is, underneath, a statement about the real exchange rate being away from where it belongs.

The distinction between the nominal and real rate is not academic hair-splitting — it is the difference between a trader who understands currencies and one who is about to lose money. Consider what it means for the real rate to *stay constant while the nominal rate moves*. If Turkey's prices rise 40% in a year and US prices rise 3%, the inflation gap is 37 points. If the lira falls 37% against the dollar over that year, the *nominal* rate has moved a huge amount — your screen shows the lira cratering — but the *real* rate is unchanged. A Turkish exporter is no more competitive than before, because their costs (wages, rent) rose in lockstep with the currency's fall. Nothing real happened; the nominal move just *kept pace* with inflation. Conversely, if the lira falls only 20% against a 37-point inflation gap, the real rate has actually *appreciated* — Turkish goods got *more* expensive in dollar terms despite the visibly falling currency, hurting exporters. The nominal rate can lie to you in both directions. The real rate is the truth serum.

#### Worked example: nominal down, real up

A currency falls and yet its country gets *less* competitive — how? Start with a real exchange rate `q = S × (P* / P)`. Take a base year where everything is indexed to 100, so `S = 100`, `P = 100`, `P* = 100`, and `q = 100`. Now over one year the home currency depreciates 8% (`S` falls to 92 — fewer foreign units per home unit, i.e. the home currency is weaker), but home inflation runs 15% (`P` rises to 115) while foreign inflation is 2% (`P*` rises to 102). The new real rate:

```
q  =  92 × (102 / 115)  =  92 × 0.887  =  81.6
```

Wait — that fell, meaning the home currency got *cheaper* in real terms. Now flip it: the home currency depreciates only 3% (`S` = 97) against the same 15%-vs-2% inflation gap:

```
q  =  97 × (102 / 115)  =  97 × 0.887  =  86.0   (still down, but less)
```

To hold the real rate *constant* at 100, the nominal rate would have had to fall by the full inflation gap of roughly 13%. Any *smaller* nominal fall than the inflation gap is a real *appreciation* — the currency got dearer in real terms even as the screen showed it falling. The takeaway: a nominal devaluation that lags your own inflation makes you *less* competitive, not more, which is the trap that ruins high-inflation exporters who watch the screen instead of the real rate.

![Stack diagram showing the nominal rate deflated by price levels into the real rate and then the trade-weighted REER](/imgs/blogs/purchasing-power-parity-and-the-real-exchange-rate-5.png)

### REER: the real rate against everyone at once

One last definition. A currency doesn't trade against just one partner; it trades against all of them. The **real effective exchange rate**, or **REER**, is the real exchange rate computed against a *trade-weighted basket* of a country's partners, then indexed to 100 in some base year. It answers the question "is this currency, on the whole, dear or cheap right now relative to its own history and its trading partners?" Central banks, the IMF, and the BIS all publish REER series, and a REER reading well above its long-run average is the textbook flag for an overvalued currency that may be losing competitiveness. When you read that "the dollar's REER is at a multi-decade high," that is the real exchange rate, trade-weighted, telling you the dollar is expensive — exactly the burger story, dressed in a suit. The mechanics of how the dollar specifically behaves as a single cross-asset variable live in [the dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity); here we care about the valuation lens itself.

Two things make REER the single most useful PPP number in practice. First, the **trade weighting** stops a single bilateral rate from misleading you. A currency can be falling against the dollar while *rising* against the euro and the yen; the bilateral USD rate alone tells you nothing about overall competitiveness, but the REER nets it all out into one figure. Second, **indexing to history** sidesteps the absolute-level problem entirely. You never have to claim a currency's "true" fair value in some absolute sense — you only ask whether it is dear or cheap *relative to its own ten- or twenty-year average*. That is a far more defensible question, and it is the one allocators actually ask. A REER 15-20% above its long-run mean is the classic setup for the IMF's external-balance assessments to flag a currency as "overvalued," and for a value-style currency manager to start watching for a turn.

But — and this is the recurring theme — even REER overvaluation is a *condition*, not a *trade*. The dollar's REER spent 2022 through 2024 at heights last seen in the mid-1980s, a textbook overvaluation, and the dollar *kept climbing* because the rate differential and safe-haven demand overpowered the valuation drag. REER tells you the spring is coiled. It does not tell you when it releases. We will hammer this point repeatedly, because it is the one professional traders internalize and amateurs never do.

## The Big Mac index: PPP you can taste

The genius of the Big Mac index is that it makes an abstract theory edible. A Big Mac is a near-perfect tiny basket: it bundles traded inputs (beef, wheat, the franchise system) with non-traded inputs (local labor, local rent, local electricity) in roughly the proportions of a real consumption basket. So burger-PPP fails in exactly the same ways and for exactly the same reasons that full-basket PPP fails — which makes it a wonderful teaching tool and a surprisingly decent rough valuation gauge.

The mechanics are simple. You take the local price of a Big Mac, convert it to dollars at the current market rate, and compare it to the US price. If the converted price is *below* the US price, the local currency is **undervalued** — it should buy more dollars than it does. If it's *above*, the currency is **overvalued**.

#### Worked example: the yen on burger-PPP

Take the headline pair. In January 2025 a Big Mac cost **\$5.69** in the United States. In Japan it cost **¥480**. The actual market rate was about **¥157 per dollar**. Convert the Japanese price to dollars:

```
480 yen  ÷  157 yen/dollar  =  $3.06
```

So the same burger costs \$5.69 in America and only \$3.06 in Japan — the Japanese burger is far cheaper, which means the yen is undervalued. By how much? The **implied PPP rate** is the exchange rate that would make the two prices equal:

```
implied rate  =  480 yen  ÷  5.69 dollars  =  84.4 yen per dollar
```

Burger-PPP says the "fair" rate is about ¥84 per dollar, but the market traded at ¥157. The yen's undervaluation is:

```
(84.4  −  157) ÷ 157  =  −46%
```

The index's published figure, after its standard adjustments, is roughly **−41%**. Either way the message is the same: on burger-PPP the yen was somewhere around 40-46% too cheap — and yet it had spent the previous three years getting *cheaper*, not richer. A burger told you the yen was a screaming bargain in 2021, and the bargain got 30% better before it got any better at all.

That last sentence is the whole reason this post exists, and we'll return to it. The intuition is that valuation and timing are different questions, which the figure above makes concrete.

![USD/JPY actual market rate versus OECD purchasing power parity fair value diverging over a decade](/imgs/blogs/purchasing-power-parity-and-the-real-exchange-rate-4.png)

That chart is the same story drawn from a more serious source than burgers: the OECD's full-basket PPP estimate for USD/JPY. The dashed line — fair value — barely moves, hovering near ¥100 per dollar because Japanese and US inflation were similar for years. The solid line — the actual market rate — tears away from it, blowing out from ¥106 in 2020 to ¥157 by 2024. The red gap between them is the deviation: at the extreme, the yen sat roughly 55% below its OECD-PPP fair value. PPP fair value was a flat, patient line; the market was a runaway. If you had traded the gap, you would have been right eventually and broke long before.

### The GDP-adjusted Big Mac: making the burger honest

The Economist itself publishes a second version of the index because the raw one is misleading for exactly the Balassa-Samuelson reason we'll dissect below: poor countries *should* have cheap burgers. The **GDP-adjusted index** plots each country's burger price against its income per head, fits a line through the cloud, and measures how far above or below that line each currency sits. A poor country whose burger is cheap *in line with its income* is fairly valued on this version; only a burger that is cheap *beyond* what its income justifies counts as a truly undervalued currency.

The effect is dramatic. On the raw index, the Chinese yuan looks ~38% undervalued — a number that fed years of "China keeps its currency artificially weak" rhetoric. On the GDP-adjusted version, most of that gap vanishes: a country at China's income level is *expected* to have a cheap burger, so the yuan is only modestly cheap once you account for it. The raw index is a measure of *price level*; the adjusted index is a measure of *currency misalignment*, and they are not the same thing. When a headline tells you a currency is "X% undervalued on the Big Mac index," your first question should always be: raw or adjusted? The raw number for an emerging-market currency is almost always overstated.

This is the single most common way people misread PPP, so it is worth stating as a rule: **never conclude a poor-country currency is a buy from a raw PPP gap.** Half or more of that gap is the country simply being poor, which is not a tradable mispricing — it is a fact about the economy that will still be true in ten years.

#### Worked example: building the "fair" rate from two basket prices

Let's do the general calculation once, cleanly, so you own it. Suppose a fixed consumption basket costs **\$2,400** in the US and **₩3,120,000** (Korean won) in South Korea. Absolute PPP's fair exchange rate is the ratio that equalizes them:

```
fair won/dollar  =  3,120,000 won  ÷  2,400 dollars  =  1,300 won per dollar
```

If the market trades at **1,380 won per dollar**, then \$2,400 converts to 1,380 × 2,400 = ₩3,312,000 — more than enough to buy the ₩3,120,000 basket. The won is undervalued by:

```
(1,300  −  1,380) ÷ 1,380  =  −5.8%
```

A dollar-holder gets about 6% more Korean basket than "fair," so Korea looks cheap to American tourists and Korean exports look competitive abroad. The takeaway is that PPP fair value is always just a price ratio — divide one basket's home-currency price by the other's and you have the rate that would make purchasing power equal.

## Why absolute PPP fails on the level

If the law of one price is so compelling, why does the Big Mac index show the franc 39% dear and the dong 45% cheap *year after year* instead of those gaps arbitraging away? Four reasons, each worth understanding because each one teaches you something real about currencies.

**1. Non-traded goods.** Roughly half of any consumption basket — and about half of a Big Mac — is local services and rent that physically cannot cross borders. You can't import a Geneva haircut or a Hanoi apartment. Where labor is expensive (Switzerland), the non-traded half of the basket is expensive, so the whole basket is expensive, so the currency looks "overvalued" forever. This is not a mispricing to be arbitraged; it is a real feature of a rich economy. Which leads directly to reason two.

**2. The Balassa-Samuelson effect.** This is the most important idea in this section, so slow down. Rich countries are rich because their *traded-goods* sectors (manufacturing, tech) are hugely productive. High productivity means high wages in those sectors. But wages tend to equalize across sectors within a country — a productive factory worker and a barber in the same city earn comparable wages, even though the barber's productivity hasn't changed. So in rich countries, *services are expensive* because they're paying near-manufacturing wages without manufacturing productivity. The result: **rich countries have systematically higher price levels, so their currencies look permanently "overvalued" on PPP, and poor countries look permanently "cheap."** The dong isn't 45% mispriced and waiting to rally; it's cheap because Vietnam's wage and price level are genuinely lower. This is why you should *never* compare a developed and an emerging currency on raw PPP and conclude the EM one is a buy.

**3. Trade barriers and transport.** Tariffs, quotas, shipping costs, and the time-value of moving physical goods all drive a permanent wedge into the law of one price. A 20% tariff lets a price gap of up to 20% persist with no arbitrage available.

**4. Quality and basket differences.** A "Big Mac" is standardized, but a national consumption basket is not. Different countries consume different things, of different qualities, and the statistical agencies measure them differently. Some of the measured PPP gap is just apples-to-oranges.

#### Worked example: how much of the franc's "overvaluation" is real?

Switzerland's burger looks 39% dear. Is the franc a 39% short? Consider that Swiss GDP per capita is roughly **\$90,000** versus about **\$83,000** in the US — Switzerland is genuinely richer and higher-cost. The Balassa-Samuelson logic says a richer country *should* have a higher price level. Empirically, across countries, each \$10,000 of extra GDP per capita is associated with very roughly a **5-10%** higher price level. So a large chunk of the franc's apparent overvaluation is the *fair* price of being a rich, high-wage, high-rent economy — not a tradable mispricing.

```
"raw" overvaluation:        +39%
explained by income level:  ~ +15 to 20% (Balassa-Samuelson, rough)
residual "true" richness:   ~ +20% — still dear, but not 39% dear
```

The lesson: raw PPP overstates how mispriced rich currencies are, because it ignores that being rich *means* being expensive. Adjusted PPP (the index also publishes a GDP-adjusted version) shrinks every gap and is the version a professional actually glances at.

## How slowly is "slowly"? The half-life of a PPP deviation

We keep saying PPP "anchors decades, not days." That phrase deserves a number, because the number is what makes PPP useless for trading and useful for thinking. The empirical literature — decades of studies on real exchange rates — converges on a striking, sobering finding sometimes called the "PPP puzzle": the **half-life of a deviation from PPP is roughly three to five years.** That means if a currency is 40% away from fair value today, you should expect it to still be roughly 20% away three-to-five years from now, and 10% away six-to-ten years out. The pull toward fair value is real, but it is glacially slow.

Sit with what that implies for a trader. A half-life of four years is an annual mean-reversion speed of about 16% — meaning a 40% misalignment closes by only about six percentage points in a typical year, a drift utterly swamped by the 8-15% *annual volatility* of a major currency pair. The signal-to-noise ratio over any horizon shorter than several years is hopeless. You can be completely right that a currency is cheap and watch it get cheaper for years, because the noise is an order of magnitude larger than the drift you are trying to capture.

#### Worked example: how long until the cheap yen gets paid?

Suppose the yen is 40% undervalued and PPP deviations have a four-year half-life. How long until the gap closes to a tradeable 10%? Mean reversion at a constant fractional rate means the deviation `D` after `t` years is:

```
D(t)  =  D(0) × (1/2) ^ (t / half-life)
```

With `D(0) = 40%` and a half-life of 4 years, to reach `D(t) = 10%`:

```
10  =  40 × (0.5) ^ (t / 4)
0.25  =  (0.5) ^ (t / 4)
t / 4  =  log(0.25) / log(0.5)  =  2
t  =  8 years
```

Eight years for a 40% undervaluation to grind down to 10% — *if* nothing pushes it further away in the meantime, which something usually does. Over those eight years a USD/JPY position would swing tens of yen each year on rate moves alone. The takeaway is brutal and clarifying: PPP is a multi-year thesis, and any position sized to "it's cheap, it'll bounce" within a trading horizon is sized against a force a hundred times stronger than the one you're betting on.

## The flow behind the value: why cheap currencies often run surpluses

Here is where PPP stops being a museum piece and starts connecting to the rest of how currencies trade. A currency that is cheap in real terms makes a country's exports cheap and its imports expensive — which tends to produce a **trade surplus**, and a trade surplus is a structural inflow of foreign money that, all else equal, pulls the currency *up* over time. A currency that is dear in real terms does the opposite: expensive exports, cheap imports, a tendency toward **deficits**, financed by capital inflows that can reverse. The current account — the broad measure of a country's trade and income balance — is the slow tide underneath the PPP signal. The full mechanics of that balance live in [the balance of payments and the current account](/blog/trading/forex/the-balance-of-payments-and-the-current-account); here we just want to see the link.

![Current-account balance as a percent of GDP across countries showing surplus and deficit nations](/imgs/blogs/purchasing-power-parity-and-the-real-exchange-rate-6.png)

Look at the pattern. Switzerland — the dearest currency on the burger index — runs a **+6.2%** of GDP current-account *surplus*, which seems to contradict the story until you realize the franc is dear *precisely because* that relentless surplus and safe-haven demand keep bidding it up: the surplus is the cause, the dear currency the effect, and the two reach an uneasy equilibrium. Japan, deeply cheap on PPP, runs a **+3.6%** surplus that, over a long enough horizon, is the gravity pulling the undervalued yen back. The United States, by contrast, runs a **−3.3%** deficit — it consumes more than it produces and finances the gap by selling assets to foreigners — a structural *downward* pull on the dollar that has been overwhelmed for years by the dollar's reserve status and high US interest rates.

#### Worked example: does cheapness actually pay?

This is the question a trader cares about: if you systematically buy the currencies that PPP says are cheap and sell the ones it says are dear, do you make money? Combine the two datasets — the Big Mac valuation and the current-account balance — and a pattern appears.

```
Country        Big Mac valuation   Current account (% GDP)
Switzerland         +38.5%              +6.2%   (dear, but surplus-backed)
Euro area            −2.8%              +2.9%
Japan              −41.0%              +3.6%   (cheap AND surplus)
China              −38.0%              +1.4%   (cheap AND surplus)
Vietnam            −45.0%              +4.8%   (cheap AND surplus)
United States        0.0%              −3.3%   (deficit)
United Kingdom       −3.5%              −2.7%   (deficit)
```

The cheap currencies — yen, yuan, dong — cluster with *surpluses*; the deficit countries — US, UK — sit near "fair" or are propped by other forces. The cheapness has a flow behind it that, over a decade, tends to be the magnet pulling them back. That is why **value (PPP) is a real, documented currency factor** — but, as the scatter below shows, a noisy one with a modest payoff, not a money-printing machine.

![Scatter plot of Big Mac valuation against current-account balance showing cheap currencies cluster with surpluses](/imgs/blogs/purchasing-power-parity-and-the-real-exchange-rate-8.png)

The upward tilt is real but loose — plenty of scatter, no straight line. That looseness is the entire practical message: PPP value works, on average, over long horizons, mixed with a lot of noise. It is one factor in a basket, which is why it shows up as a respectable-but-not-dominant signal in the [FX factor zoo of carry, value, momentum, and dollar](/blog/trading/forex/fx-as-a-factor-zoo-carry-value-momentum-and-dollar). A value-only currency strategy has a Sharpe ratio around 0.35 in the academic literature — positive, persistent, and absolutely capable of drawing down for years while it waits to be right.

## Why PPP fails in the short run

Now we confront the warning label head-on, because this is where careers are made and lost. PPP is a *long-run* statement, and the long run is brutally long — often 5 to 10 years for a deviation to halve. Over any horizon a trader cares about — a day, a month, a quarter — PPP is nearly silent, drowned out by forces that move currencies far harder and far faster.

What moves a currency in the short run? Three things, none of which is the price of lunch.

**Interest-rate differentials.** This is the master variable. Money flows to where it earns the most, so when one central bank hikes rates and another holds, capital pours toward the higher-yielding currency and bids it up — regardless of where PPP says it "should" be. From 2021 to 2024 the US-Japan two-year rate gap blew out from near zero to over four percentage points as the Fed hiked aggressively while the Bank of Japan held rates at zero. That gap, not the burger, is why the yen collapsed from ¥106 to ¥157 even though PPP screamed "undervalued" the whole way down. The mechanism is the subject of its own post, [interest-rate differentials, the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx), and is explored from the policy side in [how monetary policy moves currencies](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials). The one-line version: in the short run the rate gap beats PPP every time.

**Capital flows and the carry trade.** When the rate gap is wide, traders don't just hold the high-yielder — they *borrow* the cheap currency and lever into the expensive one to harvest the difference. This is the **carry trade**, and it pushes undervalued funding currencies (like the yen) even *further* from fair value, because the whole world is short them for yield. The cheapness doesn't self-correct; it self-reinforces, right up until the trade violently unwinds. The August 2024 yen snap-back — USD/JPY from ¥162 to ¥142 in a month — was exactly that.

**Positioning and risk sentiment.** Currencies overshoot because traders pile into the same positions, set stops in the same places, and panic together. A "cheap on PPP" currency that everyone is short can stay cheap, and get cheaper, until the positioning flushes.

![Before and after comparison of short-run drivers of FX versus long-run PPP gravity](/imgs/blogs/purchasing-power-parity-and-the-real-exchange-rate-7.png)

The way to hold both truths at once is the river metaphor made precise: PPP tells you which way the river *flows* — toward fair value — but the boat can be pushed upstream by a strong wind (a wide rate gap, a crowded carry trade) for years. PPP sizes the deviation and tells you the direction of the eventual correction. It does not, ever, tell you when. A 40% undervaluation can become a 55% undervaluation before it becomes a 10% undervaluation, and the trader who sized for "it's 40% cheap, it must bounce" is the trader who gets carried out. The broader framing of how rates, flows, and carry jointly set the rate is laid out in [what moves exchange rates](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry).

## Common misconceptions

**"A cheap currency on PPP is a buy."** No — it is a *valuation observation*, not a trade. The yen was ~40% cheap in 2021 and lost another third of its value before stabilizing. Cheapness is the direction of the long-run pull, not a signal that the pull is about to win. Mixing valuation with timing is the single most expensive mistake in currency trading.

**"PPP holds for traded goods, so it basically works."** It holds *better* for traded goods than for the full basket, but even tradables carry transport, tariffs, and pricing-to-market distortions. And about half of every consumption basket is non-traded services and rent that cannot arbitrage at all. The result is that even relative PPP — the good version — explains currency moves only over multi-year horizons and with large error bands. Over one year, the correlation between the inflation gap and the currency move is close to zero.

**"The Big Mac index is a serious valuation tool."** It is a wonderful *teaching* tool and a decent rough gauge, but a single burger is a tiny, noisy basket loaded with local rent and labor and one company's pricing decisions. The 39% "overvaluation" of the franc shrinks to roughly 20% once you adjust for Switzerland's income level via Balassa-Samuelson. Use the direction; distrust the decimal.

**"Real and nominal are basically the same once you know the rate."** They are not, and the gap between them is the entire point. The nominal rate can be falling while the real rate is *rising* if the foreign country is inflating faster — your screen says the currency is weakening, but its purchasing power is strengthening. A country can devalue its nominal currency 20% and gain *zero* competitiveness if its inflation eats the whole move, leaving the real rate unchanged. Many emerging-market "devaluations" do exactly this.

**"If a currency is overvalued on REER, it will fall soon."** REER overvaluation is a *risk flag*, not a timing signal. The dollar's REER sat at multi-decade highs through 2022-2024 and the dollar kept rising, because a 5-percentage-point rate advantage and global risk-off demand overwhelmed the valuation drag. Overvaluation tells you the *downside is larger when the turn comes* — it does not tell you the turn is coming.

**"PPP says all currencies eventually converge to fair value, so it's a safe long-term bet."** Two problems. First, "eventually" can mean a decade, over which your funding costs and the opportunity cost of the capital can exceed the eventual gain. Second, the fair value *itself moves*: Balassa-Samuelson means a country that grows richer should see its equilibrium real rate *rise*, so a currency that looks cheap today may be converging toward a fair value that is also climbing — a moving target. PPP convergence is not a riskless arbitrage; it is a slow, uncertain pull toward a line that is itself drifting. Treat it as a probabilistic tailwind over years, never as a guaranteed destination.

**"Devaluing the currency always boosts exports."** Only the *real* devaluation helps, and only if domestic costs don't immediately chase the currency down. In economies with high inflation pass-through — where a weaker currency quickly raises import prices, wages, and then everything else — a nominal devaluation can be fully eaten by inflation within a year, leaving the real rate, and competitiveness, exactly where they started. This is the "devaluation treadmill" that traps chronically high-inflation economies; the currency falls and falls and the country never actually gets cheaper.

## How it shows up in real markets

**The yen, 2021-2024 — PPP's most expensive lesson.** Every PPP model from the Big Mac to the OECD's full basket said the yen was deeply undervalued throughout this period — 30%, then 40%, then over 50% below fair value. A naive value trader who went long yen at ¥115 in 2021 "because it's cheap" watched it fall to ¥162 by mid-2024, a 40% loss, before the August 2024 snap-back to ¥142 returned only a fraction. The driver was the rate gap, not the burger. The PPP signal was *correct about direction and useless about timing*, and the cost of confusing the two was a blown-up account. This is also the canonical [carry-trade unwind, in the lineage of 1998, 2008, and 2024](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).

**The dollar's "smile" versus its valuation.** Through 2022-2024 the dollar's REER hit levels not seen since the mid-1980s — clearly "overvalued" by any PPP measure. Yet the dollar climbed, because the Fed had hiked to 5.5% and, in every risk-off scare, the world bought dollars for safety. Valuation was a coiled spring, not a falling rock. When the spring finally released in 2025, the dollar's drop from a DXY of 108 toward 97 was sharper *because* the overvaluation had built up — PPP told you the magnitude of the eventual move, never its timing. The dollar's behavior as a standalone variable is dissected in [trading the dollar via DXY, carry, and the dollar smile](/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile).

The 1985 Plaza Accord is the historical bookend to that dynamic. Through the early 1980s, high US interest rates under Paul Volcker drove the dollar to an extreme overvaluation — the same setup, four decades earlier. PPP screamed that the dollar was far too dear, and PPP was *right* — but it took a coordinated intervention by the five largest economies, the Plaza Accord, to break the move, after which the dollar fell roughly a third over the next two years (DXY from around 123 toward 85). The lesson repeats across forty years: valuation correctly diagnosed an overvalued dollar both times, and both times the correction waited for a *catalyst* — a policy turn, a coordinated intervention, a shift in the rate cycle — rather than arriving on PPP's schedule. Valuation loads the gun; the rate cycle and policy pull the trigger.

**Switzerland 2011-2015 — when a central bank fought its own real rate.** The franc was so persistently dear (safe-haven inflows, a chronic surplus) that the Swiss National Bank tried to *cap* it at 1.20 per euro from 2011, printing francs to hold the floor. By January 2015 the cost of fighting the currency's real strength became unbearable and the SNB abandoned the floor; EUR/CHF collapsed from 1.20 to below parity in minutes. The lesson for PPP: a currency that is "overvalued" on PPP can be overvalued because deep structural flows *want* it strong, and even a central bank with a printing press cannot hold back a real exchange rate indefinitely.

#### Worked example: a nominal devaluation that buys no competitiveness

A finance minister in a high-inflation emerging economy devalues the currency 20% against the dollar, hoping to make exports cheaper. The currency was 4,000 per dollar; now it's 4,800 per dollar. But domestic inflation is running at **18%** a year while US inflation is **3%** — an inflation gap of 15 points. Over the year, relative PPP says the *fair* depreciation was about 15%:

```
nominal devaluation:        −20% (engineered)
inflation gap (fair drift): −15% (home minus foreign)
real depreciation gained:   −20% − (−15%)  =  −5%
```

The country bought only about 5 points of real competitiveness, not 20 — the other 15 were eaten by its own higher inflation. Worse, if the devaluation *itself* feeds inflation (imports cost more, wages chase prices), the real gain can vanish entirely within months. The takeaway: only the *real* exchange-rate move improves competitiveness, and a nominal devaluation in a high-inflation economy is often a treadmill that runs to stand still — the mechanism behind chronic EM weakness covered in [emerging-market and sovereign debt](/blog/trading/fixed-income/emerging-market-and-sovereign-debt-yield-with-country-risk).

**Vietnam — the managed crawl and PPP.** The Vietnamese dong sits ~45% cheap on the Big Mac index, but it does not float freely to "correct." The State Bank of Vietnam runs a managed crawl, letting the dong depreciate slowly and one-directionally — roughly tracking the inflation gap with the US, which is *exactly* relative PPP enforced by policy rather than by markets. The dong's cheapness is structural (Balassa-Samuelson: Vietnam is a lower-wage economy) and policy keeps the *real* rate roughly stable rather than letting the nominal rate snap to some PPP "fair value." This is why you treat EM-versus-DM PPP gaps as information about development stage, not as trades. The USD/VND rate drifted from about 21,340 in 2014 to roughly 26,300 in 2025 — a slow, deliberate slide of a few percent a year that keeps Vietnamese exporters competitive without ever letting the dong "catch up" to its 45% raw-PPP gap, because that gap is mostly Vietnam being a developing economy, not a mispricing waiting to reverse.

**The "law of one price" inside the euro area — where PPP *does* converge.** A useful counter-case sharpens the whole theory. Within the euro area, nineteen-plus countries share *one currency*, so the exchange rate between, say, Germany and Spain is permanently fixed at one. There is no nominal rate to absorb imbalances. What happens? The price levels themselves are forced to do the adjusting — and they do, slowly and painfully. In the 2010s, peripheral economies that had grown uncompetitive (their real rates too high inside the fixed currency) could not devalue, so they ground through years of "internal devaluation" — wage cuts and deflation — to claw real competitiveness back. That episode is PPP's revenge: when you remove the nominal exchange rate as the adjustment valve, the burden falls on prices and wages directly, which is far more brutal. It is the cleanest demonstration that PPP's pull is real — it just chooses whichever channel, currency or prices, is left open to it.

## The takeaway: PPP sizes the gap, the rate gap sets the clock

So how should you actually *use* PPP when you read a currency? Hold three things in your head at once.

First, **PPP is a ruler, not a trigger.** It measures how far a currency sits from where relative prices say it belongs — 40% cheap, 20% dear — and it tells you the *direction* of the eventual correction. That number is genuinely useful: it tells you how much fuel is stored for an eventual move, which sizes the opportunity and the risk. A currency 40% undervalued has more potential upside, and more danger of an explosive snap-back, than one trading at fair value. But the ruler has no clock on it.

Second, **the clock is the rate differential and the flows.** What determines *when* a currency moves toward (or further from) fair value is the gap between two central banks' policy rates, the carry trade built on that gap, and the positioning of everyone crowded into the same trade. In any horizon shorter than several years, those forces dominate PPP completely. The yen falling 40% while "cheap" is not a failure of PPP; it is PPP and the rate gap doing exactly what each does — PPP marking the destination, the rate gap driving in the opposite direction at high speed.

Third, **combine valuation with a catalyst.** The professional way to use PPP is never on its own. You wait until a deeply undervalued currency *also* gets a turning catalyst — its central bank starting to hike, the funding rate gap narrowing, the carry trade beginning to unwind, the current-account surplus finally overwhelming the outflows. PPP tells you which currencies are *loaded*; the rate cycle and flows tell you which loaded ones are about to *fire*. The August 2024 yen unwind worked because deep undervaluation met a Bank of Japan hike and a carry flush *at the same time* — value plus catalyst.

There is a fourth, quieter use that matters even more for most people: **PPP is a risk filter, not just a return signal.** If you are long a currency that is already 20% expensive on REER, you are leaning into a structural headwind — your upside is capped by valuation even if the carry is attractive, and your downside is amplified when the turn finally comes. If you are long a currency that is cheap on PPP, valuation is at your back even while you wait. So the value lens belongs in your *risk* assessment of every currency position, not only in standalone value trades. A carry trade into a wildly overvalued high-yielder is far more dangerous than the same carry into a cheap one, because the eventual valuation correction will land on top of the carry unwind. Reading the real rate tells you not just *whether* to take a trade, but *how much* it can hurt when it goes wrong.

That is the spine of this whole series restated through the valuation lens: an exchange rate is the relative price of two monies, and PPP is the patient claim that this price should equalize purchasing power. It is the long-run gravity of the FX market — real, documented, and the reason cheap surplus currencies eventually grind higher. But gravity acts slowly, and over the horizons traders live in, the gap between two countries' interest rates and the flow of money across borders set the price. Read PPP as the map of where currencies *belong*. Read the rate gap and the flows as the weather that decides where they *are*. Confuse the two, and the most undervalued currency in the world will happily take your money for years before it ever pays you back.

## Further reading & cross-links

- [Interest-rate differentials: the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx) — the short-run force that overwhelms PPP, and why the rate gap, not the burger, drove the yen.
- [FX as a factor zoo: carry, value, momentum, and the dollar](/blog/trading/forex/fx-as-a-factor-zoo-carry-value-momentum-and-dollar) — where PPP value sits among the systematic currency factors, and how it pairs with carry and momentum.
- [The balance of payments and the current account](/blog/trading/forex/the-balance-of-payments-and-the-current-account) — the structural flow underneath cheap-and-surplus versus dear-and-deficit currencies.
- [What moves exchange rates: rates, flows, carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) — the joint framework for how the short-run drivers fit together against the long-run anchor.
- [Trading the dollar: DXY, carry, and the dollar smile](/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile) — how a richly-valued dollar can keep rising, and why REER overvaluation is a risk flag, not a timing signal.
- [The dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity) — the dollar's real exchange rate as a single variable that pulls on every other asset.
