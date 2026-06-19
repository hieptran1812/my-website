---
title: "Oil, CPI, and the Energy-Equity Correlation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why oil is the macro commodity that feeds straight into inflation, why an oil spike is good for energy stocks but bad for airlines and consumers, and why the very same oil move can be risk-on or risk-off depending on whether demand or a supply shock caused it."
tags: ["macro", "correlation", "oil", "cpi", "inflation", "breakevens", "energy", "stagflation", "supply-shock", "sectors", "commodities", "regime"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Oil is the macro commodity that feeds straight into inflation, so it is positively correlated with headline CPI (about +0.55) and with breakevens (about +0.50); that makes an oil *supply* spike a stagflationary shock that is good for energy equities (about +0.75) and bad for oil-consuming sectors like airlines (about −0.45) and, in an inflation regime, bad for the broad market.
>
> - The oil-equity correlation is **regime-dependent and depends on the cause**: a demand-driven oil rise (strong growth) is risk-on, while a supply-shock oil rise (war, OPEC cut) is risk-off — the same price move, opposite equity sign.
> - Oil hits CPI **twice**: directly through the energy line, and indirectly as a cost that bleeds into transport, food, and production over the following months.
> - In 2022, when oil averaged about \$94.53 a barrel, the S&P 500 Energy sector returned **+65.7%** while the index fell 18% and consumer discretionary lost 37% — the cleanest cross-sectional split you will ever see.
> - The one number to remember: across 2020-2025, the correlation between the annual oil price and headline CPI was about **r = +0.86**. Oil really is the inflation engine.

## The barrel that rewrote the inflation map

On the morning of February 24, 2022, Russian forces crossed into Ukraine, and the price of a barrel of crude — already climbing on a tight post-pandemic market — went vertical. Brent crude punched through \$100 a barrel for the first time since 2014, and within ten days it touched roughly \$139. West Texas Intermediate, the U.S. benchmark, rode the same wave. For drivers, it showed up as a gut-punch at the pump: U.S. retail gasoline averaged over \$5.00 a gallon by June, a record. For markets, it showed up as something more systematic. A single commodity reached up the supply chain and into the most-watched number on the economic calendar.

Four months later, on July 13, 2022, the Bureau of Labor Statistics reported that the Consumer Price Index had risen 9.1% over the prior year — a 40-year high. The single biggest contributor was energy, up 41.6% year-over-year, with gasoline alone up nearly 60%. The Federal Reserve, which had spent 2021 insisting inflation was "transitory," was now hiking interest rates at the fastest pace since the early 1980s. And the stock market was in the middle of a brutal year. But not every stock. While the S&P 500 was on its way to an 18% loss and the Nasdaq to a 33% loss, one sector was having the year of its life: energy companies, the businesses that *sell* the barrel, returned a staggering +65.7% on the year. ExxonMobil and Chevron printed record profits. The same oil price that was crushing the broad market was minting money for the people who pump it.

That split — energy soaring while almost everything else sank — is the heart of this post. Oil is unique among macro variables. Most indicators (a hot CPI print, a strong jobs report, a Fed hike) push the *whole* risk complex one direction at once. Oil does something stranger: it feeds inflation, which is bad for the market, and it directly enriches one slice of the market while it taxes the rest. The correlation between oil and "stocks" is not a single number; it is a *split*, and the sign of that split depends on something you have to diagnose yourself — whether the barrel is rising because the world wants more of it, or because the world suddenly has less.

![How an oil spike fans out to inflation and splits the equity market](/imgs/blogs/oil-prices-cpi-and-the-energy-equity-correlation-1.png)

The figure above is the mental model for the whole piece. An oil spike does not hit the market as one shock. It branches. Down one branch it lifts headline CPI and inflation expectations (breakevens), which pressures the consumer and the broad market. Down another branch it pours straight into the revenue of energy producers, who win on every dollar the barrel rises. And it bids up gold as an inflation and real-rate hedge. The job of this post is to build each of those branches from zero, measure the correlations, and then explain the single most important wrinkle: why the *cause* of the move flips the sign for the broad market.

## Foundations: what oil is, why it is everywhere, and what we mean by correlation

Before we can talk about how oil correlates with anything, we have to be precise about three things: what "the oil price" even is, why oil shows up in the inflation number, and what a correlation actually measures. Nearly every confusion about oil-and-markets comes from being fuzzy on one of these.

### What "the oil price" is

When the news says "oil is up," it almost always means the price of a barrel (42 U.S. gallons) of crude oil for delivery next month, quoted in U.S. dollars. There are two benchmark grades you will hear about constantly:

- **WTI (West Texas Intermediate)** is the U.S. benchmark, priced for delivery at a hub in Cushing, Oklahoma. It is the contract behind the headline "U.S. oil price."
- **Brent** is the international benchmark, priced from North Sea crude. It is the reference for most of the world's seaborne oil, and it tends to set the price of the products (gasoline, diesel, jet fuel) that consumers actually buy.

The two usually move within a few dollars of each other, and for our purposes — the macro correlation with inflation and equities — they are interchangeable. In this post we use WTI, because our data series is the EIA's annual-average WTI price. The number that matters is the *direction and size* of the move, and both grades tell the same macro story.

Crude oil is not the finished product. It gets refined into gasoline, diesel, jet fuel, heating oil, and the feedstock for plastics and fertilizer. That refining step is why a crude spike does not hit every consumer equally or instantly — it has to pass through refiners' margins (the "crack spread") and distribution before it reaches the pump and the airline's fuel bill. But the chain is tight enough that a sustained move in crude shows up downstream within weeks. That tightness is exactly what makes oil a macro variable rather than just a commodity-market curiosity.

### Why oil is in the inflation number — twice

The Consumer Price Index, which we cover in depth in [CPI and asset prices, the master inflation correlation](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation), is a weighted basket of the goods and services a typical household buys. Energy is roughly 7% of that basket — gasoline, electricity, natural gas, and heating oil. So when crude rises and gasoline follows, the energy line of CPI rises mechanically. That is the **direct channel**, and it is fast: a gasoline spike shows up in the *next* CPI print.

But energy is far more than 7% of the inflation *story*, because oil is also an *input cost* to almost everything else. The truck that delivers your groceries runs on diesel. The plane that flies your package runs on jet fuel. The factory that makes your furniture buys electricity and petrochemical feedstock. The farmer buys diesel for the tractor and natural-gas-derived fertilizer for the field. When oil rises, all of those costs rise, and businesses pass some of them on to customers over the following months. That is the **indirect channel**, and it is slower and stickier — it leaks into the *core* of the index (the part that strips out food and energy) with a lag.

This is the crucial asymmetry. Headline CPI includes energy, so it moves *with* oil almost immediately — that is the strong +0.55 correlation we will measure. Core CPI excludes energy directly, so the direct channel does not touch it; but the indirect, pass-through channel does, with a delay of a few months. So an oil spike first jolts headline inflation, then slowly seeps into core. A trader who understands this reads an energy-driven headline beat with a hot core very differently from one with a soft core: the first means the pass-through is already broadening, the second means it may still be contained to the pump.

### What "correlation" means, from scratch

The other half of every relationship in this series is a number that tells you *how two things move together*. If two things tend to rise and fall in step, they are positively correlated; if one tends to rise when the other falls, they are negatively correlated; if there is no reliable relationship, they are uncorrelated.

The standard measure is the **Pearson correlation coefficient**, written `r`. It runs from +1 (perfect lockstep) through 0 (no linear relationship) to −1 (perfect mirror image). When we say "oil and headline CPI have a correlation of about +0.55," we mean: across the historical sample, when oil was above its average, CPI tended to be above its average too, but not perfectly — the relationship is strong and reliable, but other things (shelter, wages, supply chains) also move CPI, so it is not a +1.0.

A closely related idea is **beta**. If correlation tells you the *direction and reliability* of a relationship, beta tells you its *size*: "for each one-unit move in X, Y moves β units on average." For oil and CPI, we will compute a beta of roughly 0.108 — meaning each extra \$1 on the barrel adds about 0.108 percentage points to the annual inflation rate, in our sample. Correlation and beta answer two different questions (is the relationship reliable? how big is the move?), and we keep them separate throughout. The full unpacking of `r`, beta, and their cousins lives in [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta); here we just need the two intuitions.

> [!note]
> This is the *correlation* series. For *why* a higher oil price feeds through to inflation and policy (the causal mechanism), lean on [commodities as macro signals: oil, copper, gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold). For the intraday reaction to an oil headline or an inventory report, lean on the event-trading series. Here we are after the measurable statistical fingerprint: the sign, the size, the lead/lag, and the regime in which it flips.

### Why the equity correlation is a split, not a number

Here is the feature that makes oil unusual. A higher oil price is, simultaneously:

- **Revenue** for the companies that produce and sell it — ExxonMobil, Chevron, the whole energy sector. Every dollar on the barrel drops, more or less, to their top line. For them, expensive oil is wonderful.
- **A cost** for the companies that consume it as fuel or feedstock — airlines burn jet fuel, trucking burns diesel, chemicals firms buy crude derivatives, and consumers spend more at the pump and less on everything else (discretionary goods, travel, restaurants). For them, expensive oil is a tax.

So "the correlation between oil and stocks" is the wrong question. The right questions are: *which* stocks, and *why* is oil moving. The energy sector is positively correlated with oil (about +0.75); the oil-consuming sectors are negatively correlated (airlines about −0.45, discretionary about −0.20); and the broad index, which contains both, nets out to a weak and regime-dependent correlation (about +0.15). The whole post is an unpacking of that split.

### Why energy stocks track oil so tightly — the operating-leverage story

It is worth pausing on *why* energy equities have such a strong +0.75 correlation to oil, because the reason is the engine of the whole producer leg. The answer is **operating leverage**. An oil producer's costs — drilling, labor, equipment, lease payments — are largely *fixed* in the short run. The price it sells the barrel for is *variable*. So when the barrel rises, almost the entire increase drops to the bottom line as profit.

Make it concrete. Imagine a producer whose all-in cost to pull a barrel out of the ground is \$45. At \$70 oil, it earns \$25 of margin per barrel. At \$95 oil, it earns \$50 of margin per barrel. The oil price rose 36% (\$70 to \$95), but the producer's *profit per barrel doubled* (\$25 to \$50, +100%). That amplification — a roughly 1.5-to-2x earnings move for each percentage move in oil — is exactly why energy stocks swing harder than the oil price itself, and why a 40% oil rise in 2022 turned into a 65.7% sector return. The stock is a *leveraged* claim on the barrel.

The same operating leverage works in reverse, viciously, when oil falls. At \$45 oil, our producer earns *zero* margin; below \$45 it loses money on every barrel. This is why energy stocks get obliterated in a demand collapse (2020, 2008) far more than the oil price decline alone would suggest — the leverage that magnifies gains on the way up magnifies losses on the way down. The +0.75 correlation is high *and* the beta is high: energy equities are oil with a magnifier bolted on.

## The inflation channel: oil is the engine

Let us start with the cleanest, most reliable relationship in the whole picture: oil and inflation. This is the channel that makes oil a tier-one macro variable rather than just another commodity.

### Measuring the oil-CPI correlation

The correlation we are after is between the oil price and headline CPI. Our curated data files give the WTI annual average and the CPI year-over-year rate over 2020-2025. Plot them on the same axes and the relationship leaps out.

![WTI oil price and headline CPI year over year, 2020 to 2025](/imgs/blogs/oil-prices-cpi-and-the-energy-equity-correlation-3.png)

The chart pairs the WTI annual average (the blue bars, left axis, in dollars per barrel) with headline CPI year-over-year (the amber line, right axis, in percent). The two move together with striking fidelity. Oil collapsed to about \$40 in the demand-destruction of 2020, and CPI averaged a meek 1.3%. Oil rebounded to \$68 in 2021 as the economy reopened, and CPI jumped to 4.4%. Oil peaked at a \$94.53 annual average in 2022 — driven by the war and a tight market — and CPI peaked at a 9.06% reading that June, its highest in four decades. Then oil eased back toward \$65-78 in 2023-2025, and CPI cooled toward the 2.5-3% range. The barrel led; the inflation rate followed.

To put a single number on it, we compute the Pearson correlation between the annual oil price and the annual-average CPI rate across 2020-2025. The signature chart of this series — the scatter with the fitted line — makes both the strength and the slope concrete.

![Oil price versus CPI scatter with regression line and correlation](/imgs/blogs/oil-prices-cpi-and-the-energy-equity-correlation-4.png)

Each dot is one year: oil price on the horizontal axis, CPI on the vertical. The dots march up and to the right, the regression line slopes clearly upward, and the correlation is **r = +0.86** — a strong, reliable positive relationship. The fitted slope (the beta) is about **0.108 percentage points of CPI per \$1 on the barrel**. The 2022 dot sits in the top-right corner, the inflation engine running hot; the 2020 dot sits in the bottom-left, the engine stalled in the pandemic.

A caution that this series hammers on: this is a tiny, six-point sample over an extraordinary period, so r = +0.86 is *illustrative of the regime*, not a law of nature. Over longer, calmer samples the oil-CPI correlation is closer to the +0.55 we quote for headline CPI — still strong, but diluted by years when shelter, wages, or supply chains drove inflation more than oil did. The point is not the exact decimal; it is that oil is one of the two or three biggest swing factors in headline inflation, and you can see it with your own eyes.

### The lead/lag: oil leads inflation, and headline leads core

Every relationship in this series has not just a sign and a strength but a *timing* — a lead or a lag. Oil's timing is one of its most useful features, because oil tends to *lead* the inflation it causes. There are two distinct lags worth separating.

First, **oil leads headline CPI by roughly one month** through the direct gasoline channel. The pump price tracks crude with only a short refining-and-distribution delay, and the energy line of CPI picks it up in the very next monthly print. So a sharp move in the spot oil price this week is a near-real-time forecast of the energy contribution to *next* month's headline CPI — long before the economists' consensus has fully adjusted. A trader watching the barrel in real time has a multi-week lead on the inflation surprise that the rest of the market will react to on release day.

Second, **headline CPI leads core CPI by several months** through the indirect, pass-through channel. The direct gasoline jolt hits headline immediately, but the seepage of higher fuel and feedstock costs into the prices of *other* goods and services — the part that lands in core — takes a quarter or more to show up, as businesses gradually pass along their higher input costs. This is why an oil spike produces a recognizable two-stage inflation signature: headline jumps first and hard, then core grinds higher a few months later if the spike persists. The market's biggest fear with an oil shock is precisely this second stage — that the spike "broadens" from the energy line into sticky core inflation, the kind the central bank cannot dismiss as transitory.

The practical use of the lead: when oil breaks out, you can front-run the inflation print. The energy contribution to next month's headline CPI is nearly knowable today from the spot oil and gasoline price. The *uncertain* part is whether it broadens into core over the following quarter — and that is what determines whether the central bank treats the spike as a passing energy blip or a genuine inflation threat. Reading the lead structure is the difference between trading the obvious headline jump and anticipating the dangerous core spillover.

#### Worked example: front-running the energy contribution to CPI

It is the third week of the month, and you watch spot WTI and retail gasoline jump 12% over four weeks. The CPI report for this month lands in about three weeks. Roughly how much will the energy line add to headline CPI, before the consensus has caught up?

Energy is about 7% of the CPI basket, and gasoline is the most oil-sensitive piece, roughly half of it. A 12% rise in retail gasoline contributes about 0.035 × 12% ≈ **0.42 percentage points** to the *month-over-month* energy push, with the rest of the energy basket (electricity, natural gas) adding a bit more on a lag. Against a consensus that may have penciled in a flat energy contribution, that is a meaningful upside surprise to the headline — and because oil *leads* CPI by about a month, you can see it coming weeks before the print. **The lead structure turns the spot oil price into a free, real-time forecast of the energy line of the next inflation report — the single most front-runnable piece of any CPI release.**

#### Worked example: oil's pass-through to headline CPI

Suppose WTI jumps from \$70 to \$95 a barrel — a \$25 move, roughly what happened into the 2022 peak — and stays there. How much does that add to headline CPI?

Use the beta from the scatter: about 0.108 percentage points of CPI per \$1 on the barrel. A \$25 move implies roughly 25 × 0.108 ≈ **+2.7 percentage points on headline CPI**. That is enormous: it is the difference between a comfortable 2.5% inflation rate and a Fed-panicking 5.2%.

Sanity-check it from the bottom up. Gasoline is roughly 3.5% of the CPI basket. A \$25 move on a \$70 barrel is about a 36% rise in crude; gasoline does not rise one-for-one (refining and taxes are fixed-ish), but say retail gasoline rises about 25%. The direct contribution is then about 0.035 × 25% ≈ **0.9 percentage points** from gasoline alone. Add the rest of the energy basket (electricity, natural gas, heating oil — another roughly 3.5% of CPI, partly oil-linked) and the indirect pass-through into transport and goods over the following months, and you build from the direct 0.9pp toward the 2.7pp the top-down beta implies. The arithmetic and the regression agree: a \$25 oil spike is a multi-point inflation event. **That is why the Fed watches the barrel — oil is the single fastest lever on the headline inflation rate.**

### The breakeven channel: oil moves *expected* inflation too

There is a second, subtler inflation channel, and it is the one that connects oil to gold and to the bond market. It runs through **breakeven inflation**.

A breakeven is the bond market's forecast of average inflation over a horizon. It is computed as the yield on a normal (nominal) Treasury minus the yield on an inflation-protected Treasury (a TIPS) of the same maturity. If the 10-year nominal yield is 4.3% and the 10-year TIPS real yield is 2.0%, the 10-year breakeven is 2.3% — that is what the market expects inflation to average over the next decade. When oil spikes, traders mark up their inflation expectations, and breakevens rise. The data shows it: the 10-year breakeven climbed from 1.79% in 2019 to 2.59% in 2021 and spiked to **3.02% in April 2022**, right alongside the oil surge, before easing back to about 2.3% as oil cooled.

Why does this matter? Because breakevens are the bridge between oil and the assets that price off *expected* inflation and real yields — chiefly gold. The cleanest macro relationship of all, covered in [real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation), is that gold trades inversely to *real* yields (the nominal yield minus expected inflation). An oil spike lifts breakevens, which — holding the nominal yield fixed — *lowers* the real yield, which is bullish for gold. So oil reaches gold not directly but through the inflation-expectations channel. This is one of the prettiest chains in macro, and we will trade it in a worked example below.

There is a reason breakevens, specifically, are so oil-sensitive at the *short* end of the curve. A 5-year breakeven moves much more with oil than a 30-year breakeven, because the market knows an oil spike is unlikely to last 30 years — it will mean-revert. So oil mostly moves *near-term* inflation expectations, leaving long-term expectations (which reflect the market's faith that the central bank will eventually win) more anchored. This is why a sustained oil spike that *un-anchors* even the long-end breakevens is so frightening: it signals the market has lost confidence that the inflation is temporary. In 2022, the 5-year breakeven spiked far more than the 10-year, exactly as the term-structure logic predicts — the bond market believed the oil-driven inflation was real but ultimately containable. The pace and shape of the breakeven response to oil is itself a read on whether the market thinks the central bank is still in control.

#### Worked example: oil to breakevens to gold

You see oil break out from \$80 to \$100 on a supply scare, and you want the cleanest expression of "this is inflationary." The naive trade is to buy energy stocks, but those are already running. Consider instead the chain.

Oil up \$20 lifts the 10-year breakeven by, say, 30 basis points (in 2022 a comparable oil move lifted breakevens from about 2.7% to 3.0%). If the nominal 10-year yield does not rise as much — because the same supply shock raises *recession* fears, capping nominal yields — then the real yield *falls* by the difference. Say the nominal yield rises only 10bp while the breakeven rises 30bp; the real yield falls 20bp.

Gold's beta to real yields is roughly −\$200 per ounce per 1.0 percentage-point move, very roughly, in the modern era. A 20bp (0.20pp) fall in the real yield implies about +\$40 on gold. On a \$2,000 ounce, holding 50 ounces (about \$100,000 of gold), that is a **+\$2,000** gain on the position — and you got there not by betting on oil directly but on the inflation-expectation it created. **The lesson: oil's most elegant expression is sometimes not energy stocks at all, but the breakeven-and-real-yield chain into gold.** The relationship is conditional, though — if the nominal yield rises *more* than the breakeven (a pure rate shock), the real yield rises and gold falls, which is exactly the 2022-24 decoupling that [real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation) dissects.

## The equity split: who wins and who loses

Now the centerpiece. Oil's correlation with the inflation gauges is positive and clean. Its correlation with *equities* is a split, and the split is the whole story.

![Oil correlation bars across energy CPI breakevens and consumer sectors](/imgs/blogs/oil-prices-cpi-and-the-energy-equity-correlation-2.png)

The bar chart ranks each asset and sector by its correlation with the oil price. Read it top to bottom and the logic of the split is laid bare:

- **Energy equities (XLE): +0.75.** The strongest relationship on the board. Energy producers' revenue *is* the oil price, so their stocks track it tightly. When oil rises, ExxonMobil, Chevron, ConocoPhillips, and the rest see margins and earnings balloon. This is the positive leg of the split.
- **Headline CPI: +0.55** and **10-year breakeven: +0.50.** The inflation channel we just built, quantified. Oil pulls actual and expected inflation up with it.
- **Materials / industrials: +0.30.** A mild positive. These sectors include some commodity producers and some firms that benefit from the same global-growth or inflation backdrop, but they also consume energy, so the net is modestly positive.
- **S&P 500: +0.15.** Weak and regime-dependent — this is the headline "oil-stocks correlation," and it is *almost nothing*, because the index nets the energy winners against the consumer losers. Anyone who quotes a single oil-stocks correlation is averaging away the actual signal.
- **Consumer discretionary: −0.20.** Negative. When gas costs more, households have less to spend on everything else — cars, electronics, restaurants, travel. Higher oil is a tax on the discretionary consumer.
- **Airlines / transports: −0.45.** The most negative on the board. Jet fuel is one of an airline's two largest costs (the other is labor), often 20-35% of operating expense. A crude spike hits their P&L directly and brutally. Trucking and shipping are the same story with diesel and bunker fuel.

That single chart is why the question "is oil good or bad for stocks?" has no answer until you say *which* stocks. The energy sector and the airline sector have correlations to oil that are *opposite in sign and similar in magnitude*. They are, in effect, two sides of the same barrel.

It is worth understanding *why* the broad-index correlation is so deceptively weak, because it is a lesson in how aggregation hides structure. Energy is only about 3-5% of the S&P 500 by market weight in a normal year. So even though energy has a powerful +0.75 correlation to oil, its small index weight means the producer leg contributes only a little to the *index's* oil sensitivity. Against it, the much larger consumer-facing and rate-sensitive sectors (technology, discretionary, communications, financials together are well over half the index) carry mildly negative oil correlations. The index sensitivity is the *weighted average* of a small, strongly-positive slice and a large, mildly-negative remainder — and those two roughly cancel, leaving the near-zero +0.15 you see. The signal did not disappear; it was *averaged into invisibility*. This is the deepest reason the index correlation is the wrong unit of analysis: aggregation across sectors with opposite signs destroys exactly the information you want.

#### Worked example: the energy-vs-airlines paired trade on an oil spike

This is the trade the correlation table is begging you to put on. You believe oil is about to spike on a supply disruption. Instead of betting on the broad market (correlation +0.15, basically a coin flip), you isolate the split: go long the energy sector and short the airlines. The two opposite-signed correlations to oil mean the pair is a focused bet on the *barrel* with much of the market-direction risk netted out.

Put \$100,000 long the energy sector ETF (XLE) and \$100,000 short an airline ETF. Now oil rises 20%, the move you forecast. Energy's correlation to oil is +0.75 and its beta is high; in 2022 a roughly 40% average oil rise drove energy equities +65.7%, so call it roughly a 1.5x beta to the oil move. A 20% oil rise then lifts XLE about 30%: your long gains **+\$30,000**. Airlines' correlation is −0.45; their beta to oil is smaller in magnitude (fuel is a big cost but not their whole P&L), say −0.4x; a 20% oil rise pushes airlines down about 8%, so your short (which gains when they fall) earns **+\$8,000**.

Combined: about **+\$38,000** on \$200,000 of gross exposure, and crucially, if the *whole market* sold off 5% on the same day for an unrelated reason, your long lost roughly \$5,000 and your short gained roughly \$5,000 — they cancel. **The paired trade converts a messy market-direction bet into a clean bet on the one variable you actually have a view on: the price of oil.** That is the practical payoff of knowing the correlation is a split.

### The 2022 cross-section: the split made visible

Forget rolling correlations for a moment and just look at one extraordinary year. In 2022, oil averaged \$94.53, inflation hit a 40-year high, the Fed hiked at the fastest pace since the 1980s, and the S&P 500 fell 18%. How did the eleven sectors of the index fare?

![S&P 500 sector returns in 2022 with energy the only winner](/imgs/blogs/oil-prices-cpi-and-the-energy-equity-correlation-6.png)

The chart sorts the eleven S&P 500 sectors by their 2022 total return. The result is the single most vivid illustration of the oil split you will ever see. Energy returned **+65.7%** — not just the best sector, but the *only* sector with a meaningfully positive return (utilities scraped +1.6%, defensive staples and health care were roughly flat). Everything else lost money, and the losses got worse the more rate-sensitive and consumer-facing the sector: financials −10.5%, materials −12.3%, real estate −26.2%, technology −28.2%, **consumer discretionary −37.0%**, and communication services −39.9%.

Read that dispersion. The single best and the single worst sectors of 2022 differed by more than **100 percentage points** of return. Energy, the seller of the barrel, was up 65.7%. Consumer discretionary, where the household spends what is left after the gas tank, was down 37.0%. That is the oil split, written in a single year's returns. An investor who held an equal-weight slice of all eleven sectors had a mediocre year; an investor who understood that an oil-and-inflation regime *rewards the producer and punishes the consumer* could have been long energy and short discretionary and had the year of their life.

#### Worked example: the 2022 sector split in dollars

Take two investors at the start of 2022, each with \$100,000, each convinced of something different.

Investor A reasons: "Inflation is roaring, the Fed is hiking, growth tech is expensive — I'll hide in the broad market." She buys the S&P 500. By year-end her \$100,000 is worth about **\$82,000** (the index's −18% total return), a \$18,000 loss.

Investor B reasons: "This is an oil-and-inflation regime. Oil enriches the producer and taxes the consumer. I'll go long energy and underweight discretionary." She puts \$100,000 into the energy sector. By year-end her stake is worth about **\$165,700** (+65.7%), a \$65,700 gain.

The gap between the two — \$83,700 on a \$100,000 stake, in a single year — was not luck. It was the oil split. Both investors saw the same inflation and the same Fed; B simply knew that "stocks" is the wrong unit of analysis when oil is the driver, and that the right unit is *the sector's relationship to the barrel*. **In an oil-led inflation regime, the cross-sectional dispersion across sectors dwarfs the move in the index itself — the alpha is in the split, not in the direction.**

## The crucial wrinkle: demand-driven versus supply-driven oil

Everything so far treated "oil up" as one thing. It is not. The single most important idea in this post — the thing that separates a novice from someone who actually understands the oil-equity correlation — is that *the same oil move can be risk-on or risk-off for the broad market depending on what caused it*.

![Demand driven versus supply shock oil rise and the opposite equity sign](/imgs/blogs/oil-prices-cpi-and-the-energy-equity-correlation-5.png)

The matrix lays out the two cases side by side. Read across each row.

### Case 1: the demand-driven oil rise (risk-on)

Sometimes oil rises because the global economy is *strong* and demand is pulling prices up. Factories are humming, people are driving and flying, China is growing — and the world wants more barrels than it did. In this case, the oil rise is a *symptom of strength*, not a cause of weakness.

What happens to equities? The energy sector wins, as always (more demand, higher price, higher revenue). But crucially, the *broad market also rises*, because the rising oil price is confirming the good news that is lifting earnings everywhere. The growth read beats the fuel-cost drag. Yields rise too, but on healthy growth, not panic — the bond market is pricing a strong economy, not a stagflationary trap. In this regime, oil and the S&P 500 are *positively* correlated, and the correlation looks benign. This is the world of a normal mid-cycle expansion.

You can confirm the demand story by looking at oil's cousins. If copper is rising with oil, and the copper/gold ratio is rising, and ISM/PMI surveys are strong, the move is demand-led — copper is the purest growth-demand commodity, as we cover in [copper, gold, and the growth-inflation signal](/blog/trading/macro-correlations/copper-gold-and-the-growth-inflation-signal). When *all* the cyclical commodities rise together against a backdrop of strong surveys, the market is telling you the oil rise is about demand.

### Case 2: the supply-shock oil rise (risk-off, stagflationary)

Sometimes oil rises because the *supply* of barrels suddenly shrinks — a war disrupts production, OPEC+ cuts output, a hurricane shuts in Gulf platforms, sanctions remove a producer from the market. In this case the oil rise has *nothing to do with strong demand*. It is a pure cost-push shock: the world is paying more for the same (or less) oil, with no growth behind it.

This is the dangerous case, and it has a name: **stagflation** — stagnant growth *plus* high inflation, the worst macro combination. What happens to equities? The energy sector still wins (it sells the now-more-expensive barrel — it is the *lone* winner). But the broad market *falls*, because higher fuel costs squeeze every other company's margins and tax the consumer, all without the offsetting boost of strong growth. Worse, the central bank is boxed in: inflation is rising (so it cannot cut rates to support the economy) but growth is weakening (so hiking risks a recession). The Fed's least-bad option is usually to keep policy tight, which compounds the equity pain. Yields rise on the inflation, but the rise is *toxic* — it is the wrong kind of yield increase. In this regime, oil and the S&P 500 are *negatively* correlated, and the correlation is vicious.

2022 was a supply-shock regime: a war removed Russian barrels from the Western market, OPEC+ was disciplined, and the price spike was cost-push, not demand-pull. That is exactly why energy was the lone winner and everything else collapsed.

#### Worked example: demand versus supply oil P&L on the same \$10 move

Oil rises \$10, from \$80 to \$90. You are long \$100,000 of the broad S&P 500. What is your P&L? The honest answer is: *it depends entirely on why oil rose.*

**Demand case.** Suppose the move came on a blowout global-PMI print — the world is booming and pulling oil up. The growth read dominates. The S&P, lifted by the same strong economy, rises maybe 1.5% on the week. Your \$100,000 gains **+\$1,500**. Oil and your stock position moved *together*: a positive correlation, and a happy one.

**Supply case.** Now suppose the same \$10 came on news that OPEC+ slashed output and a key pipeline went offline — pure supply shock, no growth behind it. The market reads stagflation: margins squeezed, the consumer taxed, the Fed trapped. The S&P falls maybe 2% on the week. Your \$100,000 loses **−\$2,000**. Identical \$10 oil move, opposite \$3,500 swing in your P&L, because the *cause* was opposite.

**The single most important takeaway of this post: the oil-equity correlation is not a number you can look up — it is conditional on the cause, and your first job when oil moves is to diagnose demand versus supply before you trade the broad market.** The energy-sector leg is robust either way (energy wins in both cases); it is the *broad-market* sign that flips with the cause.

## Common misconceptions

The oil-and-markets relationship is a graveyard of confident, wrong one-liners. Here are the five that cost people the most money, each corrected with a number.

**Myth 1: "Higher oil is good for the stock market."** This is true for *energy* stocks (correlation +0.75) and false for the *broad* market in a supply-shock regime. The S&P 500's correlation to oil is only about +0.15 over the full sample, and it goes negative in a cost-push spike — in 2022 oil soared and the index fell 18%. The truth: oil is good for the *producers* and a tax on the *consumers*, and the index nets the two to roughly nothing on average. There is no single "oil is good/bad for stocks" answer.

**Myth 2: "Oil and the S&P move together, so oil predicts the market."** The full-sample S&P-oil correlation of +0.15 is weak precisely because it averages two opposite regimes: a positive demand-driven world and a negative supply-shock world. Averaging them produces a near-zero number that describes *neither*. This is the classic trap of [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data) — a single coefficient over a regime-switching relationship is a lie that hides the real, conditional structure.

**Myth 3: "An oil spike is automatically bullish for energy stocks no matter what."** Energy equities are bullish on a *price* spike, yes — but only if it sticks. A spike that the market expects to reverse within weeks (a brief geopolitical scare that resolves) often barely moves energy stocks, because equity values discount the *long-run* price, not the spot. And a *demand-collapse* that takes oil down (2020, 2008) is brutal for energy stocks even though it is "lower oil." The correlation is to the *level the market believes is durable*, not to every spot tick.

**Myth 4: "Gold is the oil-inflation hedge."** Gold does benefit from an oil-driven rise in *breakevens* — but only through the *real-yield* channel, and only if nominal yields do not rise even more. In 2022, oil and breakevens spiked, yet gold finished roughly *flat* (−0.3%) because the Fed drove *nominal and real* yields up so hard that the real-yield headwind overwhelmed the inflation-expectation tailwind. "Oil up therefore gold up" skips the step that actually matters. Gold tracks real yields, not oil, as [inflation and gold, the real yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story) details.

**Myth 5: "Inflation is good for stocks because companies raise prices."** Mild inflation is fine for equities; *oil-driven* inflation in a tight-policy regime is poison. The relationship is non-linear and flips with the level — equities deliver their best real returns at 1-3% inflation and *negative* real returns above roughly 4-5%, because at that point the Fed is fighting inflation and the discount rate is climbing. The full U-shape and the sign flip are the subject of [inflation and stocks, the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips). An oil spike that pushes inflation above 4% does not "let companies raise prices" — it puts a hawkish central bank on the broad market's neck.

## How it shows up in real markets

Theory is cheap. Here are the dated episodes where the oil-equity split and the demand-vs-supply distinction paid off — or punished those who ignored them.

### 2022: the supply-shock spike and the great sector split

The defining case. Russia invaded Ukraine in February 2022, removing a major producer's barrels from the Western market just as the post-pandemic economy was already tight. WTI averaged \$94.53 for the year and spiked toward \$120 intraday; Brent briefly touched \$139. This was textbook supply-shock oil: cost-push, no strong demand behind it, with OPEC+ holding discipline.

The fingerprint was perfect. Headline CPI hit 9.1% in June, the 40-year high. The 10-year breakeven spiked to 3.02% in April. The S&P 500 fell 18% on the year and the Nasdaq 33%, as the Fed hiked from near zero to 4.5% in nine months. And the energy sector returned +65.7% — the lone winner in a sea of red, while consumer discretionary lost 37%. Every branch of our opening figure lit up at once: oil → CPI and breakevens up (inflation channel), oil → energy equities up (the producer leg), oil → consumer and airline pain (the consumer leg), and the supply-shock cause → broad market *down*. If you want a single year to memorize the entire oil correlation map, it is 2022.

### The 1970s: the original stagflation oil shock

The template for everything in this post was written in the 1970s, and it is worth knowing because it is the purest supply-shock-stagflation case in modern history. In October 1973, OPEC imposed an oil embargo on countries that had supported Israel in the Yom Kippur War; the price of crude roughly *quadrupled*, from about \$3 to nearly \$12 a barrel, in a matter of months. A second shock followed the 1979 Iranian Revolution, which more than doubled oil again toward \$40.

The macro fingerprint was the stagflation our matrix describes, at maximum intensity. U.S. inflation, dragged up by energy, ran into the double digits — CPI peaked above 14% in 1980. Growth stalled and unemployment rose at the *same time*, the combination that gave "stagflation" its name. And the stock market had a miserable decade in *real* terms: the S&P 500 went roughly nowhere in nominal terms across the 1970s while inflation ate its purchasing power, one of the worst real-return stretches for equities on record. Meanwhile energy stocks and the producers of physical commodities were among the few places to hide. The 1970s are why every macro trader's instinct, when oil spikes on a supply shock, is to fear stagflation and to favor the producer over the consumer — the playbook in this post is, at bottom, the lesson that decade taught.

### 2008: the spike that broke on the recession it helped cause

A subtler case that shows the regime *flipping mid-move*. Through the first half of 2008, oil climbed relentlessly, peaking at about \$147 a barrel in July on a mix of tight supply, a weak dollar, and speculative demand. That spike was a real tax on an economy already wobbling under the housing crisis — high oil helped tip the consumer over the edge. Then, as the financial crisis erupted in the autumn, *demand* collapsed, and oil cratered from \$147 to about \$32 by year-end, a 78% crash in five months.

This is the cautionary tale for the "oil up is good for energy stocks" reflex. Energy equities rode the spike up in the first half, then were devastated in the second half as the demand collapse took oil down 78%. The *same year* contained both a supply-tight spike and a demand-collapse crash, and an investor who failed to re-diagnose the cause as it flipped — who stayed long energy into the autumn because "oil had been going up" — was run over. The lesson reinforces Step 3 of the playbook: a spike can climb so high that it *causes* the recession that then craters it, and you must re-diagnose demand-versus-supply continuously, not once.

### 2020: negative oil and the demand collapse

The mirror image. In the spring of 2020, COVID lockdowns vaporized oil *demand* almost overnight — nobody was driving or flying. With storage filling and producers unable to stop pumping fast enough, the front-month WTI futures contract did something never seen before: on April 20, 2020, it settled at **−\$37.63 a barrel**. Traders were *paying* to have oil taken off their hands because they had nowhere to store it. The annual average still landed around \$39.68, but the intraday print into negative territory is the most extreme demand-collapse signal in the history of the commodity.

This was a *demand-driven* oil move (downward), and it tells you everything about the sign convention. Lower oil from a demand collapse is *not* good for the broad market — it is a recession signal. The S&P 500 had already crashed 34% in March on the same demand collapse that crushed oil. Energy stocks were obliterated (cheap oil destroys producer revenue). And CPI sagged to near zero (the inflation engine stalled). 2020 is the proof that the *cause* matters more than the direction: "lower oil" was catastrophic, because it was lower *for the worst possible reason*.

### OPEC cuts: the recurring supply lever

OPEC+ (the cartel plus Russia and allies) is the single largest deliberate mover of the oil supply curve, and its decisions are pure supply-shock events you can put on a calendar. When OPEC+ *cuts* production — as it did repeatedly in 2023 (a surprise ~1.16 million barrel-per-day cut in April, and further voluntary cuts through the year) — it is engineering exactly the cost-push, stagflationary kind of oil rise our matrix warns about. Energy stocks rally on the cut; consumers and airlines brace for higher fuel; and central banks watch nervously because a cut is *inflationary without being growth-positive*.

The recurring lesson for a trader: an OPEC announcement is a supply event, so it pushes the oil-equity correlation toward the *negative*, risk-off regime for the broad market while keeping the energy leg firmly positive. The 2023 cuts lifted energy shares even in a year when oil's *annual average* fell from 2022 — because the cuts put a floor under the price and rewarded the producers. When OPEC *adds* supply (as in late 2018 or parts of 2024), the signs reverse: cheaper fuel is a tailwind for consumers and airlines, a headwind for energy producers, and a disinflationary gift to the central bank.

### 2024-2025: the disinflation tailwind

The benign side of the same engine. As the war-driven supply premium faded and global growth cooled modestly, oil eased from the \$94.53 average of 2022 to \$77.64 in 2023, \$75.83 in 2024, and \$65.39 in 2025. The inflation engine ran in reverse: headline CPI cooled from 9% toward 2.5-3%, breakevens settled back to about 2.3%, and the Fed was finally able to stop hiking and begin easing. Falling oil was a *disinflationary tailwind* that helped the broad market recover (the S&P made new highs into 2024-2025) — the friendly mirror of 2022. The split persisted in sign but compressed in magnitude: energy stocks gave back some of their windfall as the barrel fell, while consumers and airlines got relief. The same correlation map, running the other direction.

## How to read it and use it

Here is the playbook. The oil-equity correlation is one of the most useful relationships in macro *if* you respect that it is a split conditioned on the cause. Get the cause right and the rest follows.

### Step 1: diagnose demand versus supply *before* you trade the broad market

This is the whole game, and the decision tree captures it.

![Decision tree for diagnosing an oil move as demand or supply and the trade](/imgs/blogs/oil-prices-cpi-and-the-energy-equity-correlation-7.png)

When oil moves sharply, your first question is not "what does this do to stocks?" but "*why* is oil moving?" Run the checklist:

- **Is copper moving with oil?** Copper is the purest demand-growth commodity. If copper and the copper/gold ratio are rising alongside oil, the move is demand-led. If oil is rising *alone* while copper is flat or falling, it is a supply shock.
- **What are the growth surveys saying?** Strong, rising ISM/PMI confirms a demand story. Weakening surveys with rising oil scream stagflation.
- **Is there a named supply event?** A war, an OPEC cut, a refinery outage, a sanctions headline — these are unambiguous supply shocks. The cause is on the wire.

If the answer is **demand-driven**: stay risk-on. The oil rise is confirming strength. Own cyclicals and energy; the broad market can rise with the barrel.

If the answer is **supply-shock**: go risk-off on the broad market. Put on the producer-vs-consumer split (long energy, short airlines and discretionary), and hedge the inflation with breakevens and gold. Expect the Fed to be unhelpful.

### Step 2: trade the split, not the index

Whatever the cause, the *energy-vs-consumer* split is the robust, high-conviction expression. Energy is positively correlated with oil in *every* regime (+0.75); airlines and discretionary are negatively correlated (−0.45 and −0.20). The index correlation (+0.15) is too weak to trade. Express your oil view as a *relative* bet — long the producers, short the consumers — and you isolate the barrel from the market's direction. This is the paired trade we sized earlier, and it is the single most reliable way to monetize an oil view.

The relative trade has a second virtue beyond netting out market direction: it nets out the *uncertainty about the cause*. Recall that the broad-market sign flips between demand and supply regimes, but the energy-vs-consumer split holds in *both* — energy outperforms consumers whenever oil rises, regardless of why. So if you are confident oil will rise but *unsure* whether the move is demand- or supply-driven, the paired trade is the position that does not require you to win the harder diagnosis. You give up the bigger directional payoff of a clean supply-shock call, but you also remove the biggest way to be wrong. A trader who has learned to respect how often the demand-versus-supply diagnosis is genuinely ambiguous in real time will default to the relative expression precisely because it is robust to that ambiguity — it asks only "will oil rise?" and not the much harder "and why, and what will the central bank do about it?"

### Step 3: know what invalidates the signal

The oil-equity correlation breaks, or inverts, in specific, recognizable conditions:

- **When the move is expected to reverse fast.** Equity values discount the durable price, not the spot. A two-day geopolitical scare that resolves moves energy stocks far less than its spot spike implies. Wait for confirmation that the move sticks.
- **When the cause flips mid-move.** A supply shock can morph into a demand collapse (a spike high enough to *cause* a recession, then crater on the recession). The 1970s, 2008, and 2022-into-2023 all carried this risk: oil so high it destroyed the demand that was holding it up. Re-diagnose continuously.
- **When the dollar is the real driver.** Oil is priced in dollars, so a sharp move in the dollar (DXY) moves oil mechanically with no demand or supply story at all. A surging dollar pushes oil *down* (correlation about −0.45) for purely monetary reasons, as covered in [the dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation). Before you read an oil move as demand or supply, check whether it is just the dollar.
- **When you have only the index correlation.** If your only data point is the +0.15 S&P-oil correlation, you have *no* signal — you have an average of two opposite regimes. Always decompose into the sector split and condition on the cause.

### The one-line summary

Oil is the inflation engine: it is strongly, positively correlated with headline CPI (+0.55) and breakevens (+0.50), and that makes a *supply-driven* oil spike a stagflationary shock — wonderful for energy equities (+0.75), a tax on airlines (−0.45) and the consumer, and bad for the broad market when policy is tight. The same oil move driven by *demand* is risk-on. Diagnose the cause first; trade the energy-vs-consumer split, not the index; and remember that the broad-market sign is the only thing that flips — the producer always wins on a higher barrel, and the consumer always pays.

This relationship does not live in isolation. It is one driver in the full [macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix), it interacts with the dollar (oil is priced in dollars), the copper/gold growth signal (which tells you demand from supply), and the real-yield channel into gold. Oil's special role is that it is the one macro variable that is simultaneously an inflation input *and* a direct revenue stream for one slice of the market — which is exactly why it produces the most spectacular cross-sectional dispersion in all of macro.

## Further reading and cross-links

Within this series:

- [CPI and asset prices: the master inflation correlation](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation) — the inflation number oil feeds into, and why assets price the *surprise* not the level.
- [Copper, gold, and the growth-inflation signal](/blog/trading/macro-correlations/copper-gold-and-the-growth-inflation-signal) — how to tell a demand-driven oil rise from a supply shock by watching copper.
- [The dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation) — oil is priced in dollars, so the dollar is the third driver of every oil move.
- [Inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips) — why oil-driven inflation above 4-5% turns the equity correlation negative.
- [The macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — where oil sits in the full map of drivers and assets.

For the mechanism and policy context (cited, not re-derived here):

- [Commodities as macro signals: oil, copper, gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) — the causal chain from a commodity move to the macro economy.
- [How monetary policy moves commodities: real rates and gold](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) — how the Fed's reaction to oil-driven inflation reaches back into commodity and gold prices.
