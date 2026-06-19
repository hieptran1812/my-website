---
title: "Correlation by Regime: The Four Macro Quadrants"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Every cross-asset correlation is conditional on the macro regime; the cleanest map is the 2x2 of growth and inflation, and the quadrant you are in decides which assets lead and the sign of the stock-bond correlation."
tags: ["macro", "correlation", "regime", "quadrants", "stock-bond", "inflation", "growth", "asset-allocation", "goldilocks", "stagflation", "reflation", "deflation"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A cross-asset correlation is never a fixed number; it is conditional on the macro regime, and the cleanest regime map is the 2x2 of GROWTH (up/down) by INFLATION (up/down): Goldilocks, Reflation, Stagflation, Deflation. The quadrant you are in decides which asset leads and — most importantly — the *sign* of the stock-bond correlation, which swings from about −0.40 in Goldilocks to +0.55 in Stagflation to −0.55 in Deflation.
>
> - Growth and inflation are the two master axes because between them they set earnings (the numerator of every asset price) and the discount rate (the denominator). Everything else is downstream.
> - Identify the quadrant *first*, then read the correlations. Stocks lead Goldilocks, commodities and gold lead Stagflation, government bonds lead Deflation. The same pair of assets can be a great hedge in one quadrant and fail completely in another.
> - The stock-bond correlation is the single most important number that flips. When it goes positive (Stagflation), your bonds stop protecting your stocks and the 60/40 portfolio breaks — exactly what happened in 2022.
> - The one number to remember: in the high-inflation quadrant the stock-bond correlation is roughly **+0.55** — diversification fails when you need it most.

In January 2022 a retired schoolteacher in Ohio owned what every textbook calls a "balanced" portfolio: 60% stocks, 40% bonds. For twenty years that mix had behaved like a shock absorber. When stocks dropped, bonds rose, and the combined ride was smooth. Then 2022 arrived. The S&P 500 fell about 18% for the year. Long Treasury bonds — the supposed safety net — fell *more*, roughly 30%. Both legs of the portfolio went down together, and the worst year for the classic balanced portfolio in a century unfolded not because anyone made a bad stock pick, but because a single number nobody on financial television talks about quietly changed sign.

That number is the stock-bond correlation. For the entire 1998–2021 period it had been comfortably negative — stocks and bonds zigged and zagged in opposite directions, which is the whole reason you hold both. In 2022 it flipped to strongly positive. The shock absorber became a second spring pushing the same way. The schoolteacher did nothing wrong. She was simply standing in a different macro quadrant than the one her portfolio was built for, and she did not know the map had changed.

This post is that map. The claim is simple and, once you see it, hard to unsee: **correlation is a regime, not a constant.** And the cleanest way to name the regime is a 2x2 grid built from the only two macro variables that matter at the top level — growth and inflation. Get the quadrant right and the correlations fall out almost for free. Get it wrong and you will keep being surprised by relationships that "always held" right up until the moment you needed them.

![Four macro quadrants from growth and inflation showing the leading asset and stock bond correlation in each cell](/imgs/blogs/correlation-by-regime-the-four-macro-quadrants-1.png)

## Foundations: why growth and inflation are the two master axes

Before any quadrant, we need to agree on what an asset price *is*. Every financial asset is a claim on a future stream of cash. A stock is a claim on future company profits. A bond is a claim on future fixed coupons and the return of principal. Even gold, which pays nothing, is priced against the *opportunity cost* of holding something that pays nothing. So the value of almost any asset can be written, intuitively, as:

> price ≈ (expected future cash) ÷ (the rate you discount it at)

That fraction has a numerator and a denominator. And here is the central fact of all of macro investing: **growth drives the numerator, and inflation drives the denominator.**

- **Growth** — how fast the economy and corporate earnings are expanding — moves the *numerator*. When growth accelerates, companies sell more, earnings rise, defaults fall, and the future cash that every risky asset is a claim on gets bigger. When growth stalls, earnings shrink and defaults climb.
- **Inflation** — how fast prices are rising — moves the *denominator*. Higher inflation forces central banks to raise interest rates, and higher interest rates raise the discount rate applied to every future cash flow. A dollar arriving in ten years is worth less today when rates are 5% than when they are 1%. Higher inflation also erodes the real value of fixed cash flows directly, which is poison for bonds.

So growth and inflation are not just two of many macro variables. They are the two that sit on opposite sides of the pricing equation. Almost everything else — the jobs report, ISM surveys, retail sales, the yield curve, credit spreads, the dollar — is a *messenger* telling you about one of these two master variables. (The mechanism behind how a rate change propagates through every asset is covered in detail in the macro-trading series, especially [interest rates as the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [real versus nominal yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal); here we use the result, not re-derive it.)

An everyday way to feel this: an asset is like a fruit tree you are deciding whether to buy. The *numerator* is how much fruit the tree will bear each year — that is growth: a healthy economy is a well-watered orchard, a recession is a drought. The *denominator* is the interest rate you could earn instead by putting your money in the bank — that is what inflation controls, because central banks raise rates to fight rising prices. When the orchard is thriving and the bank pays nothing (Goldilocks), the tree is worth a fortune. When the orchard is withering *and* the bank suddenly pays 5% (Stagflation), nobody wants the tree at any reasonable price — the fruit is shrinking and the alternative just got attractive. Two dials, fruit and the bank rate, set the value of everything. That is the entire engine behind the quadrants.

Why the *discount rate* matters so much is worth one extra beat, because it is where most beginners' intuition is weakest. A bond paying \$50 a year for thirty years is worth far more when you discount it at 1% than at 5%, because at 5% the far-off coupons are crushed toward nothing in present-value terms. The longer-dated the cash flow, the more violently its present value swings with the discount rate. This is why high-growth tech stocks — whose profits are mostly expected *far* in the future — fall hardest when rates rise: they are "long-duration" in the same sense a 30-year bond is. Inflation, by forcing rates up, hits the longest-duration assets first and hardest. Hold that thought; it is why the Nasdaq, not the broad market, was the worst casualty of the 2022 Stagflation scare.

### What "correlation" means here

A quick primer, since this is a correlation series and we will use the word constantly. The **correlation** between two assets is a single number between −1 and +1 that measures how their returns move together. A value of +1 means they move in lockstep the same direction; −1 means perfectly opposite; 0 means no linear relationship. We usually denote it with the Greek letter rho (ρ).

The reason correlation matters so much for a portfolio is that **risk is not additive**. If you hold two assets with correlation −0.5, their combined volatility is much *lower* than the average of their individual volatilities, because the wobbles partly cancel. That cancellation is the "free lunch" of diversification (see [the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch)). The entire 60/40 portfolio is a bet that stock-bond correlation stays negative. The whole problem this post addresses is that **correlation is not a stable property of two assets — it is a property of the regime they are trading in.**

One refinement that matters for the worked examples below: correlation (ρ) and *beta* (β) are related but different. Correlation tells you the *direction and tightness* of the relationship on a −1 to +1 scale. Beta tells you the *magnitude* of one asset's move per unit move in the other — for example, "the S&P falls 0.7% per +0.1pp core-CPI upside surprise" is a beta. You can think of beta as correlation scaled by the ratio of the two volatilities: a high correlation with a small volatility ratio gives a small beta. For the quadrant framework, the headline news is in the *sign and tightness* of correlation (does the hedge work or not?), but when we price out a specific dollar move we reach for beta. Both are regime-conditional: the same pair can have a +0.10 correlation in one quadrant and a −0.45 correlation in another, and the betas flip with them.

### Two axes, four quadrants

If growth can be "up" or "down" and inflation can be "up" or "down", you get exactly four combinations. Those four boxes are the macro quadrants, and they have names that practitioners have used for decades:

| Quadrant | Growth | Inflation | The vibe |
|---|---|---|---|
| **Goldilocks** | up | down | Not too hot, not too cold: solid growth, cooling prices |
| **Reflation** | up | up | The economy reflates: growth and prices both accelerate |
| **Stagflation** | down | up | The nightmare: stagnant growth *and* rising prices |
| **Deflation** | down | down | The bust: growth collapses and prices fall |

The figure above is this 2x2. Read it the way you would read a map: the vertical axis is growth (up at the top), the horizontal axis is inflation (rising to the right). Each cell tells you the *leader* (the asset class that tends to outperform) and, in the corner, the stock-bond correlation sign. Notice that the four leaders are four *different* assets — stocks, commodities, gold, bonds — and notice that the stock-bond correlation is negative in three cells and strongly positive in one. That single positive cell is where 60/40 portfolios go to die.

### Why the regime, not the asset, owns the correlation

It helps to anchor the whole framework in one piece of history, because it makes the "regime not constant" claim concrete rather than theoretical. The stock-bond correlation has flipped sign *twice* in living memory, and each flip lines up with a change in the inflation regime, not with anything about stocks or bonds themselves:

- **1970s–80s (high inflation):** the correlation was *positive*, around +0.35. This was the original Stagflation era, with the oil shocks and double-digit inflation. Stocks and bonds fell together as the Fed jacked rates toward 20%.
- **1998–2021 (disinflation and QE):** the correlation turned *negative*, around −0.40, and stayed there for two decades. Low, stable inflation gave central banks a free hand; bad-growth news meant rate cuts, so bonds hedged stocks. This is the era that birthed the 60/40 cult.
- **2022 onward (inflation returns):** the correlation flipped *positive* again, around +0.45, the moment inflation became the dominant risk once more.

Three eras, two sign-flips, one driver: the inflation regime. The assets did not change their nature. What changed was the *quadrant the world was sitting in* for years at a time. A reader who learned investing in the middle era could go a full career believing bonds always hedge stocks — and then meet 2022. The quadrant map is the antidote to that kind of regime-blindness: it makes the conditionality of every correlation explicit and visible on a single grid.

## The mechanism: how each quadrant sets the leaders

Let us walk the quadrants one at a time and reason out *why* each leader leads, building from the numerator-denominator logic above. This is the part most coverage skips — it asserts "buy commodities in reflation" without explaining the chain — and it is exactly the part you need so you can spot when a quadrant is bending the usual rules.

### Goldilocks (growth up, inflation down): stocks lead

This is the dream regime for equities. The numerator (earnings) is rising because growth is solid. The denominator (the discount rate) is *falling* or at least stable because inflation is cooling, which lets central banks hold or cut rates. Rising numerator over a steady-to-falling denominator is the perfect recipe for stock prices. Bonds also do fine — falling inflation means falling yields means rising bond prices — but they cannot keep up with equities. Commodities and gold lag, because there is no inflation scare to bid them and no growth shortage to make them scarce.

Crucially, in Goldilocks the **stock-bond correlation is negative** (around −0.40 in our data). Why? Because the dominant risk being priced is *growth* risk. On a day with a growth scare, stocks fall (lower earnings) but bonds rise (the market expects rate cuts to rescue the economy). They move opposite. Bonds hedge stocks. The 60/40 portfolio works exactly as advertised.

The deeper reason the sign is negative is worth spelling out, because it is the mechanism that *reverses* in Stagflation. In a low-inflation world, the central bank has a free hand: if growth weakens, it can cut rates without worrying about prices. So bad-growth news carries an implied promise — "rate cuts are coming" — which is good for bonds even as it is bad for stocks. The bond is, in effect, an option on central-bank rescue. That option is what makes bonds a hedge. The most of the 1998–2021 era lived in this configuration, which is why an entire generation of investors came to believe the negative stock-bond correlation was a permanent law. It was not a law; it was the property of a quadrant they happened to inhabit for two decades.

### Reflation (growth up, inflation up): commodities lead

Now both axes point up. Growth is strong, so the numerator rises — good for stocks. But inflation is also rising, which lifts the denominator and starts to cap how much stocks can gain. The asset that loves this quadrant is **commodities**: rising growth means rising demand for oil, copper, and raw materials, *and* rising inflation means rising commodity prices almost by definition (commodities are a big component of inflation). Gold does well too as an inflation hedge. Stocks still rise but with more volatility. Bonds are the loser here — rising inflation is direct poison for fixed coupons.

The stock-bond correlation in reflation is mildly negative to roughly zero (−0.10 in our data). It is the *transition* quadrant: as long as the growth read dominates the market's attention, stocks and bonds can still move opposite; but as inflation climbs and the market starts keying on rate hikes, the correlation drifts toward positive. Reflation is the antechamber to stagflation.

The second-order story in reflation is about *which* inflation. There is "good" inflation — demand-pull, the kind that comes from a booming economy where everyone has a job and is spending — and "bad" inflation — cost-push, the kind that comes from supply shocks like an oil embargo. Reflation usually starts as the good kind: prices rise because demand is strong, which is fundamentally a growth story, so stocks shrug it off and the stock-bond correlation stays negative. The danger is that good inflation, if it runs hot enough, forces the central bank to lean against it, and once the market believes rate hikes are coming faster than growth can absorb, the *same* inflation prints that were bullish start reading as bearish. That is the moment reflation tips into stagflation, and the stock-bond correlation crosses zero on its way to positive. Watching that tipping point — the shift from "good news is good" to "good news is bad" — is one of the most valuable regime reads there is (the event-trading series covers the intraday version in [cross-asset transmission](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market)).

### Stagflation (growth down, inflation up): gold and commodities lead

This is the dangerous quadrant and the whole reason this post exists. Growth is falling, so the numerator shrinks — bad for stocks. Inflation is rising, so the denominator climbs and central banks are forced to *raise* rates into a slowing economy — which is double poison for stocks (lower earnings *and* higher discount rate) and poison for bonds too (rising rates crush bond prices). When *both* of the things that price every asset are moving the wrong way, almost everything financial falls together.

What survives? **Real assets.** Gold leads — it is the classic store of value when paper money is losing purchasing power and real returns are negative. Commodities hold up because the inflation itself often originates in commodity supply shocks. Cash, earning a high nominal rate, suddenly looks attractive.

And here is the killer fact: in stagflation the **stock-bond correlation flips strongly positive — about +0.55.** Both stocks and bonds are being hammered by the *same* force (rising rates to fight inflation), so they fall *together*. Your bonds no longer hedge your stocks; they add to the loss. This is precisely the 2022 experience. We will quantify it below.

The mechanism is the mirror image of the Goldilocks rescue option. In Goldilocks, bad-growth news implied "rate cuts are coming," which lifted bonds. In Stagflation, the central bank has *no free hand*: inflation is too high to cut rates even though growth is weak. So bad-growth news no longer carries the rate-cut promise — the bond's rescue option is gone, written out of the contract by inflation. Worse, the dominant force in the market is no longer growth risk at all; it is *inflation/rate risk*, and inflation/rate risk hits stocks and bonds the *same* way: both are claims on future dollars, and a higher discount rate shrinks the present value of both. When the single biggest mover is a variable that pushes two assets the same direction, their correlation must turn positive. That is not bad luck or a market malfunction; it is arithmetic. The stagflation positive correlation is the most rational thing in the world once you see that the market switched which risk it was pricing.

A practical tell that you have entered this quadrant: the *bond* market starts leading the equity selloff. In a normal growth scare, equities sell off and bonds rally. In a stagflation scare, you see a "rates tantrum" — yields spike, and *then* equities follow them down. When you notice the bond market driving the stock market lower rather than cushioning it, you are almost certainly in or near the stagflation cell, and your hedges need rethinking immediately.

### Deflation (growth down, inflation down): government bonds lead

The bust. Growth collapses (numerator falls hard) and inflation falls or turns negative. Now the central bank slashes rates to zero to fight the slump. Falling rates send bond prices *soaring* — and because inflation is low, those bonds are real, not eroded. **Government bonds lead** and are the single best asset to own. Stocks fall (earnings collapse), commodities fall (no demand), and gold is roughly flat (the deflation drag on its inflation-hedge role partly offsets the falling-real-rate tailwind).

The stock-bond correlation in deflation is *strongly negative* — about −0.55, the most negative of any quadrant. This is the bond hedge working at full power: stocks crash, bonds rally, the portfolio is partly rescued. This is why long Treasuries are the canonical "risk-off" asset and why they shone in 2008 and March 2020.

But deflation has a vicious second-order effect that the simple "bonds rally" story hides: *credit*. When growth collapses, companies struggle to service their debt, and the market reprices the risk of default. Corporate bond spreads — the extra yield over safe Treasuries that investors demand to hold risky credit — *blow out*. So while the safest government bonds rally, the riskiest corporate bonds can fall almost as hard as stocks, because they share the default risk. This is the deflation cell where the *credit-equity* correlation becomes the canary: widening spreads and falling equities march together toward −1, and credit usually moves *first*, which is why credit spreads are one of the best early warnings of a deflationary bust (the dedicated treatment is in [credit spreads, the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary)). The lesson: "bonds hedge stocks in deflation" is true only for *government* bonds; the corporate-credit part of a bond portfolio is closet equity risk that shows its true colors exactly when you need protection.

![Heatmap of average real returns by macro quadrant and asset class with the leader highlighted per quadrant](/imgs/blogs/correlation-by-regime-the-four-macro-quadrants-2.png)

The heatmap above is the centerpiece. Each row is a quadrant; each column an asset class; each cell the representative average annual real return, with the row's leader boxed. Read across any row and you can see the rotation: stocks dominate Goldilocks (+18%), commodities dominate Reflation (+16%), gold dominates Stagflation (+12% while stocks bleed −12%), and bonds dominate Deflation (+12% while everything else falls). One picture, the whole rotation.

There is a subtler reading available too, and it is worth pausing on, because it explains *why* the quadrant framework is more useful than a list of single-asset rules. Look down any *column* instead of across a row. Stocks go +18, +12, −12, −8 as you move Goldilocks → Reflation → Stagflation → Deflation: they want growth and they tolerate only mild inflation, so they decline monotonically as you walk away from the top-left corner. Gold goes +5, +8, +12, 0: it rises with inflation right up until deflation kills the inflation premium. Bonds go +6, −2, −4, +12: they are the mirror image of stocks, hating inflation and loving the deflationary bust. Each asset has a *signature* across the quadrants, and the portfolio's job is to combine assets whose signatures offset — which is exactly what correlation measures. The quadrant grid is the generator of the correlations; the correlations are the shadow it casts.

### The inflation-equity U-shape that hides inside the quadrants

One relationship deserves special attention because it is the most commonly misread in all of macro: how stocks respond to inflation. Naively you might draw a straight line — more inflation, worse for stocks — and compute a single negative correlation. That line is wrong, and the quadrant map shows why. Equities deliver their *best* real returns in moderate inflation (roughly 1–3%, the Goldilocks and early-Reflation zone) and their *worst* real returns at *both* extremes: in outright deflation (where earnings collapse) and in high inflation above about 4–5% (where the discount rate explodes). Plotted against inflation, real equity returns trace a hump, or inverted-U: rising on the left, peaking in the middle, falling on the right.

This is why a single "stocks versus inflation" correlation is meaningless without the regime. On the left side of the hump (deflation toward moderate inflation), the correlation is *positive* — more inflation means more nominal demand, rising earnings, a healthier economy, so stocks like it. On the right side (moderate toward high inflation), the correlation flips *negative* — more inflation means more rate hikes, a higher discount rate, recession fear, so stocks hate it. The same pair of variables produces opposite-signed correlations on the two sides of the same curve. The quadrant you are in tells you which side of the hump you are standing on, and therefore which sign to expect. This single fact resolves the perennial argument about whether inflation is "good" or "bad" for stocks: it is both, in different quadrants.

#### Worked example: the inflation-equity sign flip in dollars

You hold \$200,000 of the S&P 500 across two different one-year regimes, and you want to know what a +1 percentage-point move in inflation does to your position in each.

In a **low, stable regime** (Goldilocks, inflation drifting from 2% to 3%), the monthly equity-return correlation with inflation surprises is mildly *positive*, about +0.10, because the market reads a hotter print as a growth signal. A representative beta puts a +1pp inflation surprise at roughly +1.5% for the index over the following quarter. On \$200,000 that is about a **+\$3,000** tailwind. Inflation *helped* you.

In a **rising, high regime** (the Stagflation side, inflation climbing from 4% to 5%), the same correlation is strongly *negative*, about −0.45, because now the market reads a hotter print as a rate-hike threat. A representative beta puts the same +1pp surprise at roughly −4% over the following quarter. On \$200,000 that is about a **−\$8,000** drag. Inflation *hurt* you — and the move was nearly three times as large and the opposite sign.

The intuition: the exact same +1pp inflation surprise added \$3,000 in one quadrant and subtracted \$8,000 in another. Anyone running a single full-sample "stocks versus inflation" beta would have averaged these into a small negative number that describes neither regime — which is the whole reason you date the quadrant before you trust the correlation.

#### Worked example: dating the current quadrant from two data points

Suppose it is a given month and you have exactly two fresh readings on your desk. ISM manufacturing PMI just printed **47.0** (below the 50 line that separates expansion from contraction, and down from 48 last month). Core CPI just printed **4.2%** year over year, *up* from 3.9% the month before. Where are you?

- ISM at 47 and falling means **growth is DOWN** (a reading below 50 signals contraction; the downward move confirms direction).
- Core CPI at 4.2% and rising means **inflation is UP.**

Growth down + inflation up = **Stagflation.** That single classification immediately tells you the live correlations: stock-bond is now positive (your bonds will *not* hedge your stocks), gold and commodities are the leaders, and the inflation-equity correlation is strongly negative. A portfolio that was fine in last quarter's Goldilocks regime is now exposed to a correlation flip it was never built for. The intuition: you did not need a single new model — two numbers and a 2x2 told you which world you woke up in, and the world decides the correlations.

## How to date the live quadrant from the data

The worked example above hints at the real skill: turning a stream of macro releases into a quadrant call. You only need to answer two questions — *which way is growth pointing?* and *which way is inflation pointing?* — but each deserves a small toolkit, because no single indicator is perfect and you want confirmation.

**For the growth axis**, the best fast read is the business surveys, especially ISM/PMI new orders, which lead actual GDP and earnings by several months (the [ISM/PMI leading correlation](/blog/trading/macro-correlations/ism-pmi-the-leading-correlation-with-cyclicals) post covers this in depth). A reading above 50 and rising means growth up; below 50 and falling means growth down. Confirm with the direction of *growth surprises* (is data coming in above or below expectations?), the unemployment-claims trend, and the copper/gold ratio, which is a market-priced growth thermometer.

**For the inflation axis**, the headline read is core CPI and core PCE direction — not the level, the *direction and the surprise*. Is inflation accelerating or decelerating relative to what the market expected? Confirm with PPI (which leads goods inflation), 10-year breakevens (the market's forward inflation bet), and oil, which feeds straight into headline CPI.

The trick is that you are dating *direction*, not level. A 3% inflation print that is *falling* from 4% puts you on the cooling side; the same 3% *rising* from 2% puts you on the heating side. The quadrant is about momentum across the two axes.

A few honest cautions about dating, because this is where the framework is hardest to use and where overconfidence is most dangerous:

- **Use surprises, not levels, where you can.** Markets price the *expected* path of growth and inflation already. What moves assets — and therefore what reveals the live correlations — is the *surprise* relative to expectations. A 4% inflation print is bullish for bonds if everyone feared 5%, and bearish if everyone hoped for 3%. The single sharpest input to your quadrant call is the direction of the surprises (the dedicated treatment is in [the surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) and the framework underlying it in [why news moves markets](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework)).
- **Beware the lag.** Growth data is slow and revised. ISM and claims are timely; GDP arrives a quarter late and gets rewritten twice. Lean on the leading indicators for the *direction*, and treat the lagging ones as confirmation, not as your first read. The yield curve, building permits, and ISM new orders all lead actual activity by months (the [lead-lag](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) post quantifies this), which is exactly what you want when dating a *turning* regime.
- **Expect ambiguous prints.** Real data does not arrive labeled "Stagflation." You will get a strong jobs number alongside a soft survey, or a hot headline CPI driven by a one-off energy spike. The discipline is to ask which signal speaks to *growth direction* and which to *inflation direction*, weight the leading indicators, and accept that your quadrant call is a probability, not a certainty. When the call is genuinely 50/50 between two adjacent quadrants, the correct response is usually to reduce the size of any quadrant-dependent bet rather than to pretend you know.
- **Quadrants flip faster than cycles.** A full business cycle takes years, but a *quadrant* can flip in a quarter — 2022 ran through Reflation into a Stagflation scare and back toward disinflation inside about eighteen months. Date the quadrant on a monthly cadence, not an annual one, and be willing to update.

![Decision flow that dates the live quadrant from a growth reading then an inflation reading](/imgs/blogs/correlation-by-regime-the-four-macro-quadrants-3.png)

The decision flow above is the dating checklist as a tree: start with the two readings, branch first on growth direction, then on inflation direction, and you land in exactly one of the four quadrants — each labeled with its leader. It is deliberately mechanical. The discipline of *always answering growth first, inflation second* keeps you from the most common mistake, which is to fixate on inflation headlines and forget that a Stagflation call and a Reflation call differ *only* in the growth axis.

#### Worked example: the stock-bond correlation by quadrant, and what it costs

Let us make the correlation flip concrete in dollars. You run a \$1,000,000 portfolio, 60% stocks (\$600,000) and 40% long Treasury bonds (\$400,000). Stock volatility is 16% a year; bond volatility is 12% a year. The risk of the combined portfolio depends entirely on the stock-bond correlation, through the standard two-asset variance formula:

portfolio variance = (0.6 × 0.16)² + (0.4 × 0.12)² + 2 × (0.6 × 0.16) × (0.4 × 0.12) × ρ

The first two terms are fixed: (0.096)² + (0.048)² = 0.009216 + 0.002304 = 0.01152. The cross term is 2 × 0.096 × 0.048 × ρ = 0.009216 × ρ.

- **Goldilocks (ρ = −0.40):** variance = 0.01152 + 0.009216 × (−0.40) = 0.01152 − 0.003686 = 0.007834. Volatility = √0.007834 ≈ **8.85%**, so a one-standard-deviation bad year is about a **\$88,500** drawdown.
- **Stagflation (ρ = +0.55):** variance = 0.01152 + 0.009216 × (0.55) = 0.01152 + 0.005069 = 0.016589. Volatility = √0.016589 ≈ **12.88%**, so a one-standard-deviation bad year is about a **\$128,800** drawdown.

Same assets, same weights, same individual volatilities. The *only* thing that changed is the regime, and it lifted your expected bad-year loss from roughly \$88,500 to \$128,800 — a 45% increase in risk — without you trading a single share. The intuition: the correlation is not a footnote to your risk; in this example it is the difference between a calm portfolio and a dangerous one, and the quadrant is what sets it.

![Bar chart of the stock bond correlation by macro quadrant flipping from negative to positive](/imgs/blogs/correlation-by-regime-the-four-macro-quadrants-4.png)

The bars above show the flip directly: deeply negative in Deflation (−0.55, the strongest hedge), negative in Goldilocks (−0.40), near zero in Reflation (−0.10), and strongly positive in Stagflation (+0.55, where diversification fails). Green bars are the regimes where bonds protect you; the red bar is the one where they betray you. If you internalize only one chart from this entire series, make it this one. (The deep mechanics of this flip get a dedicated treatment in [the stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) and the case study in [when correlations break: the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip).)

## Which correlations are live in each quadrant

The deepest point of this post is that the quadrant does not just pick a *leader* — it activates an entire *set* of correlations and switches others off. A correlation you studied for months can be completely dormant in the quadrant you are actually in, while a different one suddenly becomes the only thing that matters. Each earlier post in this series describes one correlation; the quadrant tells you *when* to care about it.

- **In Goldilocks**, the live correlations are the gentle ones. Stock-bond is a working hedge. The inflation-equity correlation is mildly positive (the market reads inflation prints as a growth signal, not a rate threat — see [inflation and stocks, the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips)). Credit spreads are tight and dormant. The gold-real-yield correlation is quiet because real yields are not the dominant story.

- **In Reflation**, the inflation-equity correlation starts to bite as prices climb, the oil-CPI link lights up, and the gold-breakeven relationship activates as inflation expectations rise.

- **In Stagflation**, the dangerous set fires all at once: the inflation-equity correlation goes strongly negative, the stock-bond correlation flips positive, and the gold-real-yield correlation becomes the cleanest game in town as gold gets a real-yield and safe-haven bid simultaneously. This is the quadrant where the most correlations are "on" and where they are most painful.

- **In Deflation**, the bond hedge runs at full power (stock-bond strongly negative), and the credit-equity correlation becomes the canary: spreads *blow out* as default risk spikes, and widening spreads and falling equities move together toward −1 (the [credit spreads correlation](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary) post covers this). The dollar's safe-haven bid also activates, pressuring commodities further.

![Matrix of which cross asset correlations are live or dormant in each of the four macro quadrants](/imgs/blogs/correlation-by-regime-the-four-macro-quadrants-5.png)

The matrix above lays it out: rows are the major correlations, columns are the quadrants, and each cell tells you whether that correlation is a friendly hedge (green), a dangerous trap (red), a watch-it transition (amber), or dormant (neutral). The single most striking row is the top one — the stock-bond correlation — which is green-green-RED-green across the quadrants. That red cell in Stagflation is the trapdoor under every "balanced" portfolio.

This is also the deepest answer to the question that titles the whole series — [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant). When you compute a full-sample correlation, you are *averaging across all four quadrants*, which produces a number that describes none of them. The full-sample stock-bond correlation might come out near zero — averaging −0.40 Goldilocks, −0.55 Deflation, and +0.55 Stagflation — and that "zero" is the single most misleading number in macro finance, because in any *actual* quadrant the real correlation is large and the sign is decisive.

## Common misconceptions

**"The stock-bond correlation is negative — bonds always hedge stocks."** No. It is negative in three of the four quadrants but strongly *positive* in Stagflation (about +0.55). The negative correlation everyone treats as a law of nature is a feature of the *low-inflation* regime that happened to dominate from 1998 to 2021. The structural history is positive correlation in the high-inflation 1970s–80s (around +0.35), negative through the 1998–2021 disinflation era (around −0.40), and positive again from 2022 (around +0.45). The hedge is regime-conditional, not permanent.

**"Gold hedges inflation."** Only loosely, and only in the right quadrant. Gold's cleanest correlation is with *real yields* (inflation-adjusted interest rates), not inflation itself — the [real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story) shows roughly −0.82 for 2007–2021. Gold leads in Stagflation not because inflation is high but because *real* yields are negative (central banks cannot raise nominal rates fast enough to outrun inflation). In Reflation, where central banks *can* raise rates and keep real yields positive, gold's performance is more muted than the "inflation hedge" cliché suggests.

**"A single correlation number tells me how two assets relate."** Only if you also tell me the regime. A full-sample correlation averages across quadrants with opposite signs and lands on a number that is true in no quadrant. Always ask "in which regime?" before trusting a correlation. This is also why a [rolling correlation window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters): a 10-year window blends regimes; a 1-year window can isolate the live one.

**"Inflation is bad for stocks."** It depends on the level and the quadrant. Equities deliver their *best* real returns in moderate inflation (roughly 1–3%, the Goldilocks/early-Reflation zone) and their *worst* in both deflation and high inflation. The relationship is a U-shape, not a line, so a single linear "stocks versus inflation" correlation is misleading — the sign flips depending on which side of the U you are on.

**"Diversification protects you in a crisis."** This is the most expensive myth. Diversification protects you in *calm* and in *Deflation* (where bonds rally as stocks fall). It fails precisely in Stagflation and in violent deleveraging events, when average pairwise correlations across risk assets surge toward 0.80. "Diversification fails when you need it most" is not a cynical quip; it is the Stagflation cell of the matrix made literal (see [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis)).

**"The quadrant map and the business-cycle clock are the same thing."** They are cousins, not twins, and confusing them causes timing errors. The business-cycle clock (early → mid → late → recession) is a *time-ordered* path through the economy; the quadrant map is a *state* defined purely by the current direction of growth and inflation. The cycle usually *traces* a loop through the quadrants — recovery tends to be Goldilocks, the late-cycle overheat tends to be Reflation or Stagflation, the recession tends to be Deflation — but the economy can skip cells, reverse, or sit in one for years (the 1998–2021 Goldilocks/Deflation oscillation never visited Stagflation at all). Use the clock to anticipate the *likely next* quadrant; use the quadrant to read *today's* correlations. The clock is the forecast; the quadrant is the nowcast.

**"Cash does nothing in these regimes."** Cash quietly becomes a *leader-adjacent* asset in the two bad quadrants. In Stagflation and Deflation, cash earns a positive (sometimes high) nominal rate while stocks and commodities or bonds are losing money, so its representative real return of about +1% beats most of the risk assets around it. "Cash is trash" is a Goldilocks/Reflation statement; in Stagflation, cash earning 5% while your bonds lose 4% is one of the best risk-adjusted positions on the board.

## How it shows up in real markets

The cleanest way to test this framework is to walk the last few years, because they handed us a rapid tour through three different quadrants — and the stock-bond correlation tracked the inflation regime almost exactly as the model predicts.

![CPI year over year and the rolling stock bond correlation from 2020 to 2025 across reflation stagflation and disinflation](/imgs/blogs/correlation-by-regime-the-four-macro-quadrants-6.png)

The chart above overlays CPI (the inflation axis, amber) with the rolling stock-bond correlation (blue) from 2020 to 2025. Watch the two lines move together: as inflation surged into 2022, the stock-bond correlation climbed from negative to strongly positive; as inflation rolled over into 2023–24, the correlation eased back down. The inflation regime *is* the stock-bond correlation regime.

**2021 — Reflation.** Growth was roaring back from the COVID shock (2021 real GDP grew about 5.9%) and inflation was climbing fast (CPI ran from about 1.4% in January 2021 to 6.8% by November). Growth up, inflation up: textbook Reflation. Commodities led — oil and copper soared — and the stock-bond correlation, per our data, sat around −0.10, the mild-negative transition value. Stocks rose (the S&P gained over 25% on the year) but bonds were the weak leg as yields started to climb. The market was still reading the strong-growth signal more than the inflation threat. This was the antechamber.

**2022 — the Stagflation scare.** Inflation peaked at 9.06% in June 2022 (a 40-year high) while growth visibly decelerated and the Fed hiked aggressively into the slowdown. This was the Stagflation-scare quadrant, and the model's signature prediction came true with brutal precision: the stock-bond correlation flipped to about +0.60. Stocks fell roughly 18% *and* long bonds fell roughly 30% — the diversification that had defined the 60/40 era simply stopped working, because both legs were being crushed by the same force, the repricing of interest rates to fight inflation. Gold, by contrast, held roughly flat in dollar terms (and rose against most other currencies) — the real-asset leadership of the Stagflation cell.

**2023–24 — Goldilocks-ish disinflation.** Inflation rolled over hard — CPI fell from 6.45% at the end of 2022 to about 3.35% by the end of 2023 and toward 2.9% by the end of 2024 — while growth held up better than feared (2023 real GDP about 2.9%, 2024 about 2.8%). Growth holding, inflation cooling: a drift back toward Goldilocks. The stock-bond correlation eased from +0.45 in 2023 toward +0.30 in 2024 — still positive, because inflation remained above the comfortable 2–3% band, but trending back toward the negative hedge regime as the inflation scare receded. Stocks led again, with the S&P closing 2024 near 5,882. The quadrant rotated, and the correlations rotated with it.

**2025–26 — an unsettled boundary.** The most recent data is a useful reminder that quadrants are not always cleanly identifiable. CPI hovered in the high-2s through most of 2025 (about 2.7% mid-year, 3.0% in September) before re-accelerating toward 3.3% in early 2026 and a hot 4.25% print by mid-2026, while the growth read softened (ISM manufacturing sat just below the 50 line — 49.0 in mid-2025). A softening growth read with re-accelerating inflation is precisely the *direction* that points back toward the Stagflation corner, and the stock-bond correlation stayed stubbornly positive (around +0.25–0.30) rather than returning to its old negative hedge value. The lesson of this episode is not which quadrant "won" — that is for the case-study posts to litigate — but that *the correlation refused to go back negative as long as inflation stayed an active threat.* The hedge does not return until the inflation regime fully retreats, and that is exactly what the quadrant map predicts.

The thread running through all four episodes is the same: **the inflation regime owned the stock-bond correlation the entire time.** When you overlay the two series, as in the chart above, you are watching the same hand move two puppets. This is the single most reliable real-world confirmation of the whole framework, and it is why dating the inflation axis is the highest-leverage thing you can do for portfolio construction.

### When the quadrant framework itself breaks

Honesty requires acknowledging the limits. The quadrant map is a powerful default, but it is not a law of physics, and there are three situations where it bends:

- **Liquidity overrides everything.** In a violent deleveraging — a 2008 or a March 2020 — *every* risk asset correlates toward +1 as forced sellers dump whatever they can, regardless of the underlying growth/inflation read. Average pairwise correlations across risk assets surge toward 0.80 in a crisis. In those weeks the quadrant map is temporarily suspended; only the safest government bonds and cash behave. The map reasserts itself once the deleveraging ends, but you must respect that the liquidity regime can briefly trump the macro regime (see [global liquidity and the everything correlation](/blog/trading/macro-correlations/global-liquidity-and-the-everything-correlation)).
- **Policy can decouple an asset.** Gold's textbook negative correlation with real yields broke down in 2022–24 — gold rose even as real yields climbed — because central-bank and official-sector buying overwhelmed the usual real-yield channel. A structural buyer can decouple an asset from its normal driver for years. The quadrant's *leader* can still be right (gold did lead) even when the *mechanism* you expected is overridden.
- **The boundaries are fuzzy and the data is noisy.** Real regimes do not snap cleanly from one cell to the next; they smear across boundaries, and revisions can reclassify a month after the fact. Treat the quadrant as a probabilistic center of gravity, not a hard label, and size your conviction to the clarity of the read.

#### Worked example: the quadrant-rotation trade and its P&L

Now put real money on the framework. You start with \$100,000 and compare two strategies across one year in each quadrant. Strategy A is the static 60/40 (60% stocks, 40% bonds, never changes). Strategy B is a "quadrant-tilted" book that overweights each quadrant's leader. Apply the representative quadrant returns from our data:

- **Goldilocks:** static 60/40 returns about +13.2% → **\$13,200** profit. The tilted book (overweight stocks) returns about +14.2% → **\$14,150**. A small edge — Goldilocks is kind to 60/40 anyway.
- **Reflation:** static 60/40 returns about +6.4% → **\$6,400** (bonds drag). The tilted book (overweight commodities and stocks) returns about +12.0% → **\$12,000**. The tilt nearly doubles the return by sidestepping the bond drag.
- **Stagflation:** static 60/40 *loses* about −8.8% → **−\$8,800** (both legs fall together). The tilted book (overweight gold, commodities, cash) returns about +8.6% → **\$8,550**. This is the whole ballgame: a **\$17,350** swing in a single year, purely from being in the right quadrant.
- **Deflation:** static 60/40 returns about 0% → **\$0** (the bond rally exactly offsets the stock crash). The tilted book (overweight bonds and cash) returns about +6.1% → **\$6,100**.

![Bar chart comparing one year profit on 100000 dollars for a static 60 40 book versus a quadrant tilted book in each quadrant](/imgs/blogs/correlation-by-regime-the-four-macro-quadrants-7.png)

The bars above compare the two books across the four quadrants. The static 60/40 (gray) does fine in Goldilocks and Deflation but bleeds in Reflation and craters in Stagflation; the quadrant-tilted book (blue) is positive in all four. The intuition: 60/40 is implicitly a *bet that you are always in Goldilocks or Deflation* — the two quadrants where the bond hedge works. The moment the regime rotates into Stagflation, that implicit bet costs you nearly \$9,000 on \$100,000, and a portfolio that simply identified the quadrant and leaned into its leader turned that loss into a gain.

A caution before you mortgage the house on this: the returns above are representative averages, not guarantees, and *dating* the quadrant in real time is genuinely hard — the data is noisy, revisions are frequent, and quadrants can flip in a quarter. The edge is not "predict the future perfectly"; it is "stop holding a portfolio that only survives in two of the four quadrants while pretending the other two cannot happen."

## How to read it and use it

Here is the playbook, compressed into the order of operations you should actually follow.

**Step 1 — date the quadrant before you look at any correlation.** Answer two questions in this exact order. Which way is growth pointing (ISM/PMI direction, growth surprises, claims trend, copper/gold)? Which way is inflation pointing (core CPI/PCE direction and surprise, PPI, breakevens, oil)? Those two answers place you in one of four cells. Do this *first*, every time. Most correlation mistakes are really quadrant-misidentification mistakes.

**Step 2 — read the live correlations off the quadrant, not off a full-sample table.** Once you know the cell, the matrix tells you which correlations are on. In Stagflation, expect stock-bond positive, inflation-equity negative, gold-real-yield dominant. In Deflation, expect stock-bond strongly negative, credit-equity tightening toward −1. Do not trust a long-window historical correlation that averages across quadrants — it will tell you the average of −0.55 and +0.55 is zero, which is true and useless.

**Step 3 — size the bond hedge to the quadrant.** This is the single most actionable consequence. Your bonds are a great hedge in Goldilocks and Deflation and a *negative* hedge in Stagflation. If your quadrant call is Stagflation or you are in the Reflation antechamber, your "safe" bond allocation is not safe — it is correlated risk. That is when real assets (gold, commodities, sometimes cash earning a high nominal rate) do the hedging that bonds cannot.

#### Worked example: a quadrant-appropriate allocation P&L

You manage a \$500,000 portfolio and your quadrant call has just shifted from Goldilocks to Stagflation. Compare two responses over the following year, using the representative quadrant returns from our data (Stagflation: stocks −12%, gov bonds −4%, gold +12%, commodities +14%, cash +1%).

**Response A — do nothing (stay 60/40).** \$300,000 in stocks loses 12% → −\$36,000. \$200,000 in bonds loses 4% → −\$8,000. Total: **−\$44,000**, a −8.8% year, with *both* legs red because the stock-bond correlation went positive and the bond "hedge" added to the loss instead of offsetting it.

**Response B — re-allocate to the quadrant's leaders.** Move to \$150,000 stocks, \$50,000 bonds, \$125,000 gold, \$125,000 commodities, \$50,000 cash. Compute each leg: stocks \$150,000 × −12% = −\$18,000; bonds \$50,000 × −4% = −\$2,000; gold \$125,000 × +12% = +\$15,000; commodities \$125,000 × +14% = +\$17,500; cash \$50,000 × +1% = +\$500. Total: **+\$13,000**, a +2.6% year.

The swing from Response A to Response B is **\$57,000** on a \$500,000 book — more than 11% of the portfolio — and it came entirely from recognizing that the stock-bond correlation had flipped and that the quadrant's hedge was now gold and commodities, not bonds. The intuition: in a regime where your traditional hedge has turned into correlated risk, the most valuable trade is not a clever stock pick but simply *re-sizing the hedge to the quadrant you are actually in.*

**Step 4 — know what invalidates the call.** The quadrant call is invalidated when *either* axis reverses direction. A growth re-acceleration flips you from Stagflation back toward Reflation; a sharp disinflation flips you from Stagflation toward Goldilocks; a growth collapse with falling inflation flips you into Deflation, where the bond hedge roars back to life. Watch the *turn* in ISM and the *turn* in core inflation; those turns, not the levels, are your regime-change signals. And remember the U-shape: do not mechanically treat "inflation rising" as bearish for stocks if you are still inside the benign 1–3% band — context is the whole game.

**The one-line discipline:** *identify the quadrant first; the correlations follow.* Every other post in this series describes one correlation in detail; this post is the index that tells you which of them is live today. (For the cycle-timing view of the same idea — the rotation of leadership as the economy moves through phases — see [the business cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) and, on the mechanism side, [the business cycle's four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) and [asset rotation across the business-cycle quadrants](/blog/trading/macro-trading/asset-rotation-across-the-business-cycle-quadrants).)

### How this connects to the rest of the series

It is worth stepping back to see how the quadrant map sits at the center of everything else you have read. The series has been a tour of individual correlations — stock-bond, inflation-equity, gold-real-yield, credit-equity, dollar-commodity, the lead-lag of leading indicators. Each of those posts answers "what is this correlation, how strong, which way does it lead, when does it flip?" The quadrant map answers the meta-question that sits above all of them: *when does each one matter?*

Concretely, the quadrant tells you which earlier post to re-read this month. If your dating exercise lands on Stagflation, the relevant pages are the stock-bond flip, the inflation-equity negative, and the gold-real-yield lead — those are the correlations doing the work. If it lands on Deflation, you re-read the credit-spread canary and the bond-hedge mechanics, because those are the correlations that will define the next few months. The quadrant is a *router*: it takes the firehose of macro data, compresses it to two directional reads, and routes your attention to the three or four correlations that are actually live. That routing is the whole value of the framework, and it is why the [macro-correlation playbook capstone](/blog/trading/macro-correlations/the-macro-correlation-playbook-capstone) uses the quadrant as its first step before anything else.

There is also a humility built into the framework that is easy to miss. Because the quadrant map makes the conditionality of every correlation *explicit*, it also makes you honest about uncertainty. When the growth and inflation reads are clear, you can lean into the quadrant's correlations with conviction. When they are muddy — a soft survey against a strong jobs print, a hot headline against a cool core — the map *tells you* you are between cells, which is itself a signal to size down. A framework that knows when it does not know is worth more than a single full-sample correlation table that is always confidently wrong about the regime you are actually in.

## Further reading and cross-links

Within this series:

- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the thesis this post operationalizes with the quadrant map.
- [The stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) — a deep dive on the single most important correlation that flips.
- [When correlations break: the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip) — the dated case study of the Stagflation-cell flip.
- [Inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips) — the U-shape and why inflation is good for stocks in one quadrant and poison in another.
- [The business cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) — the rotation of leadership over the cycle, the time-axis companion to the quadrant grid.
- [The macro-correlation playbook (capstone)](/blog/trading/macro-correlations/the-macro-correlation-playbook-capstone) — the end-to-end synthesis: read the regime, read the dominant correlations, position.

Mechanism and cross-asset context (don't re-derive — cite):

- [Asset rotation across the business-cycle quadrants](/blog/trading/macro-trading/asset-rotation-across-the-business-cycle-quadrants) and [the business cycle's four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) — the macro-trading mechanism behind the rotation.
- [Correlation by regime: growth and inflation](/blog/trading/cross-asset/correlation-by-regime-growth-and-inflation) — the cross-asset allocator's framing of the same 2x2.
- [Interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [real versus nominal yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — why inflation drives the discount rate.
