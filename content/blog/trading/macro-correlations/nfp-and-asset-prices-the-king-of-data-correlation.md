---
title: "NFP and Asset Prices: The King of Data and Its Flipping Correlation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why the monthly US jobs report is the single highest-variance scheduled release, and why the same strong number can crash stocks in one year and rally them in another."
tags: ["macro", "correlation", "nfp", "nonfarm-payrolls", "jobs-report", "labor-market", "regime", "reaction-function", "equities", "bonds"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Nonfarm payrolls is the highest-variance scheduled data release, but its correlation with stocks has no fixed sign: it depends entirely on the market's reaction function, so a strong jobs print is *bearish* for equities in an inflation-fear regime (2022-23) and *bullish* in a normal expansion.
>
> - The sign of the NFP-equity correlation flips by regime; the bond and dollar reaction does not. A hot print sells the 2-year note and lifts the dollar in *both* regimes — it is only stocks that change their mind.
> - In the 2022-23 "good news is bad news" regime, a +100k upside payroll surprise moved the S&P about −0.50% and the 2Y yield about +6bp. In a benign expansion the same surprise moved the S&P about +0.35% and the 2Y about +4bp.
> - The market does not trade the headline; it trades the component that maps to the Fed path — wages, the unemployment rate, or revisions can override a hot payroll number.
> - The one fact to remember: the jobs report's *direction of effect on stocks is conditional*. Always check the regime before you guess the sign.

On the morning of Friday, 3 February 2023, the US Bureau of Labor Statistics reported that the economy had added **517,000 nonfarm jobs** in January — against a consensus forecast of about 185,000. It was one of the largest upside surprises in the history of the release: nearly three times what economists expected. The unemployment rate fell to 3.4%, the lowest in 53 years. By every plain-language reading, this was spectacular news. The American jobs machine was roaring.

And the stock market sold off. The S&P 500 fell, two-year Treasury yields jumped by roughly fifteen basis points, and the US dollar surged. A beginner watching the tape would have been baffled: the economy is *strong*, why are stocks *falling*? The answer is the single most important idea in this entire post. In February 2023 the market was terrified of inflation and of the Federal Reserve's response to it. A blowout jobs report did not mean "buy stocks, the economy is healthy." It meant "the labor market is too hot, the Fed will have to keep hiking, and higher rates are poison for equity valuations." Good news for Main Street was bad news for Wall Street.

Now rewind to a calmer year — say, a mid-cycle expansion when inflation is near the Fed's 2% target and nobody fears a rate shock. A strong jobs report in *that* world rallies stocks: more people working means more spending, more corporate revenue, more earnings. Same data series. Opposite reaction. This is the defining feature of nonfarm payrolls as a correlation: it is the loudest scheduled number on the calendar, and its relationship with stock prices flips sign depending on what the market is afraid of. This post builds that idea from the ground up — what the jobs report actually contains, how the surprise maps to prices, why bonds and the dollar are honest while stocks are moody, and how to read the regime before you bet on the sign.

![Jobs surprise routed through the reaction function switch into opposite stock signs](/imgs/blogs/nfp-and-asset-prices-the-king-of-data-correlation-1.png)

## Foundations: what the jobs report is and what "correlation" means here

Before we can talk about how the jobs report *moves* asset prices, we need two things on the table: a clear picture of what the report actually contains, and a precise idea of what we mean when we say two things are "correlated." Let us take them in order, building from zero.

### The everyday version of correlation

Imagine you run a small ice-cream stand and you keep two numbers each day: the temperature, and how many cones you sell. Over a summer you would notice that hotter days tend to be bigger sales days. That tendency — *when one number is high, the other tends to be high too* — is **positive correlation**. If instead you tracked the temperature against your sales of hot soup, you would see the opposite: hotter days, fewer bowls of soup. That is **negative correlation**. And if you tracked temperature against the number of even-numbered house addresses you walked past, you would see no relationship at all: **zero correlation**.

Statisticians put a number on this tendency, called the **Pearson correlation coefficient**, written *r*. It runs from −1 to +1. An *r* of +1 means the two numbers move in perfect lockstep up; −1 means perfect lockstep in opposite directions; 0 means no linear relationship. Most real-world relationships sit somewhere in between — temperature and ice cream might be *r* = +0.7, strong but not perfect, because some hot days are rainy and some cool days are festivals. We cover the exact mechanics of computing *r*, and its cousins Spearman and beta, in [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) — for this post you only need the intuition: *r* is a single number summarizing whether two series move together, and which way.

One honesty note before we go further, because it governs everything that follows. The betas in this post — "−0.50% per 100k" and the like — are *averages estimated over many releases*, not laws of physics. Any single jobs report scatters widely around its average beta: the same +100k surprise might move the S&P −0.2% one month and −0.9% the next, depending on what else is going on (a CPI print earlier that week, a Fed meeting around the corner, a thin holiday market). Correlation is also not causation — the jobs report does not *cause* stocks to fall; it shifts the market's expectation of Fed policy, and *that* moves stocks. When we say "NFP correlates −0.50 with the S&P in a fear regime," we mean the historical co-movement averaged that way, with a wide cloud of individual outcomes around it. Treat every beta as a center of gravity, not a guarantee. The traps of reading too much into a single correlation are catalogued in [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data).

There is a closely related idea we will lean on heavily: **beta**. Where correlation asks "do they move together?", beta asks "by *how much*?" If a +100,000 surprise in the jobs number is associated with a −0.5% move in the S&P 500, then the *beta* of the S&P to the jobs surprise is −0.5% per 100k. Beta is the slope of the line you would draw through a scatter plot of (surprise, asset move). Correlation tells you the sign and tightness; beta tells you the magnitude per unit. Both can — and for the jobs report, *do* — flip sign across regimes.

### What is in the jobs report

The release everyone calls "the jobs report" or "NFP" is officially the **Employment Situation** report, published by the Bureau of Labor Statistics (BLS) on the first Friday of most months at 8:30 a.m. Eastern, covering the prior month. It is built from two completely separate surveys, and confusing them is the single most common beginner mistake.

The **establishment survey** (also called the payroll survey or CES) asks roughly 120,000 businesses and government agencies how many people were on their payrolls. From this comes the headline you hear on the news: **nonfarm payrolls**, the net change in the number of jobs, excluding farm work, the military, and a few other categories. "Nonfarm" is a historical quirk — farm employment was seasonal and noisy, so it was carved out. This number is "the king of data" because it is the broadest, most timely, hardest-to-fake read on the real economy that arrives on a fixed schedule.

The **household survey** (CPS) telephones about 60,000 households and asks who is working, who is looking, and who has given up. From this comes the **unemployment rate** — the share of people who want a job and are actively looking but do not have one. The two surveys can disagree in any given month because they measure different things (the household survey counts the self-employed and farm workers; the establishment survey counts jobs, so one person with two jobs is counted twice). That disagreement is itself a signal we will return to.

![Anatomy of the jobs report five components feeding one rate read](/imgs/blogs/nfp-and-asset-prices-the-king-of-data-correlation-7.png)

A worked illustration of why the two surveys can disagree clarifies the whole report. Suppose 100,000 self-employed gig workers lose their contracts and 80,000 of them take payroll jobs at companies. The establishment survey records +80,000 payrolls (they show up on company payrolls now). The household survey, which already counted the self-employed as employed, records a *net loss* of 20,000 employed people (100,000 left self-employment, only 80,000 found payroll work). Same labor market, two surveys, opposite signs — neither is wrong; they are measuring different universes. This is why a sophisticated reader never treats the two as interchangeable, and why a divergence between them is a signal in its own right.

The report bundles five numbers the market cares about:

1. **Headline payrolls** — net jobs added (establishment survey). The marquee number.
2. **Average hourly earnings (AHE)** — wage growth, reported both month-over-month and year-over-year. This is the most direct inflation read in the whole report, because wages are the largest input cost in a services economy.
3. **The unemployment rate** — labor-market slack (household survey).
4. **The participation rate** — the share of working-age people in the labor force. A falling unemployment rate is good for a different reason if it falls because people found jobs versus because they gave up looking.
5. **Revisions** — the BLS revises the prior two months every release. Big downward revisions can quietly flip the whole signal: a "good" headline of +200k looks very different if last month was revised down by 90k.

The market does not weigh these equally, and — this is crucial — *which* one it weighs most heavily changes with the regime.

### Why payrolls is "the king of data": variance and breadth

People call NFP the king of data for two concrete reasons. The first is **breadth**: no other monthly release samples the whole economy as comprehensively. Inflation reports tell you about prices; manufacturing surveys tell you about factories; retail sales tell you about shoppers. The jobs report tells you, in one number, whether the economy as a whole is hiring or firing — and employment is the hinge on which almost everything else turns, because people who have jobs spend money, and people who spend money generate the revenue that becomes corporate earnings.

The second reason is **variance**. Of all the scheduled releases, NFP routinely produces the largest single-day moves across stocks, bonds, and the dollar. Part of this is genuine surprise: payrolls are hard to forecast, so the gap between actual and consensus is often large. Part of it is timing: the report lands at 8:30 a.m. on a Friday, after a week of positioning, and it is frequently the first hard data of the month. And part of it is the seasonal-adjustment machinery itself. The raw, unadjusted payroll figures swing by *millions* of jobs each month — retailers hire armies for the holidays and fire them in January, schools dismiss staff in summer. The BLS applies a large **seasonal adjustment** to strip out these predictable swings, and the size of that adjustment means the final headline can be sensitive to model assumptions. A 0.1% error in a seasonal factor applied to 150 million payroll slots is 150,000 jobs — about the size of an entire month's job growth. This is why payroll prints can be noisy, why the revisions are so important, and why a single month's surprise should never be over-interpreted on its own.

For the trader, the practical upshot is that NFP has the highest **expected move** of any data day. Options markets price this in: implied volatility on the S&P, on Treasury futures, and on the dollar all elevate into the report and collapse afterward. The number is loud precisely because it is both broad and noisy — broad enough to matter, noisy enough to surprise.

### Why higher rates hurt stocks: the discount-rate mechanism

To understand *why* a strong jobs report can be bearish, you need one more building block: how a stock is valued. A share of stock is a claim on a stream of future cash flows — the company's earnings, paid out or reinvested over the years and decades ahead. To value that stream today, you **discount** it: a dollar of earnings ten years from now is worth less than a dollar today, and *how much* less depends on the interest rate you use to discount it. A higher discount rate makes those far-off earnings worth less today, which lowers the stock's present value. This is not a metaphor; it is the arithmetic of present value, and it is the master channel through which interest rates price every asset, developed in the macro-trading post [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

Now connect the chain. A hot jobs report signals a strong labor market. In an inflation-fear regime, a strong labor market means wage pressure, which means more inflation, which means the Fed keeps rates higher for longer. Higher rates mean a higher discount rate. A higher discount rate means lower present value for stocks — *especially* for "long-duration" stocks like high-growth tech, whose value sits mostly in far-future earnings that get discounted the hardest. That is why the Nasdaq typically falls *more* than the S&P on a hawkish jobs print: its cash flows are further out, so they are more sensitive to the discount rate.

This mechanism is what makes "good news is bad news" coherent rather than paradoxical. The good news (strong jobs) is genuinely good for the *economy*. It is bad for *stock prices* because, in a fear regime, it raises the discount rate faster than it raises expected earnings. The two effects — higher earnings outlook (good for stocks) and higher discount rate (bad for stocks) — always both exist; the regime just determines which one wins. In a normal expansion the earnings effect wins and stocks rise; in an inflation-fear regime the discount-rate effect wins and stocks fall. That tug-of-war is the engine of the entire sign flip.

### Why this is a correlation, not a constant

Here is the thesis of this entire series, applied to the jobs report. The relationship between a macro indicator and an asset price has four properties: a **sign** (does the asset go up or down?), a **strength** (how reliably?), a **lead or lag** (does the indicator move first?), and — the property most people forget — it **flips across regimes**. For most indicators the sign is fairly stable. For nonfarm payrolls and stocks, the sign is the *least* stable thing about it. That instability is not a flaw in the data; it is the whole story. The same number is genuinely good news or genuinely bad news depending on what problem the market thinks the economy has.

## The surprise, not the level: how a jobs print becomes a price move

A number on a screen does not move markets. The *difference between that number and what was already expected* moves markets. This is the single most important mechanical idea for trading any data release, and it deserves a careful build.

### Markets price expectations in advance

Before NFP is released, there is a published **consensus** — a survey of economists' forecasts. If the consensus is +185,000 and the print comes in at +185,000, the market barely moves, even if +185,000 is a "strong" number in some absolute sense. Why? Because that strength was already baked into prices. Everyone who wanted to position for a strong number already did. The only thing that creates a fresh move is the gap between the actual print and the consensus — the **surprise**:

```
surprise = actual - consensus
```

So the January 2023 report's market-moving content was not "517,000 jobs." It was "517,000 minus the expected 185,000 = a +332,000 upside surprise" — an enormous shock. The asset's move equals its beta to the surprise, multiplied by the size of the surprise:

```
price move ≈ beta × surprise
```

We develop this surprise framework in depth — including how to standardize a surprise so you can compare a payroll shock to a CPI shock — in [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises). The release-day mechanics of the jobs print specifically, including the first thirty seconds of the tape, are covered in the event-trading companion [the jobs report (NFP): the king of data](/blog/trading/event-trading/the-jobs-report-nfp-the-king-of-data). Here we focus on the part neither of those covers in full: why the *beta itself* changes sign.

#### Worked example: turning a surprise into a dollar P&L

Suppose you hold a \$50,000 position in an S&P 500 index fund, and a jobs report lands with a +100,000 upside surprise during the 2022-23 inflation-fear regime, where the S&P's beta to an NFP surprise is about −0.50% per 100k. The implied index move is:

```
beta × surprise = -0.50% × (100k / 100k) = -0.50%
```

On your \$50,000 position that is a paper loss of:

```
-0.50% × $50,000 = -$250
```

Now run the *same* +100,000 surprise in a normal expansion, where the beta is about +0.35% per 100k:

```
+0.35% × $50,000 = +$175
```

The identical economic event — 100,000 more jobs than expected — costs you \$250 in one regime and earns you \$175 in the other, a \$425 swing on the same position from the same number. The intuition: the jobs surprise did not change, but the *price of that surprise* did, because the market's fear changed.

### The reaction function: the switch that flips the sign

The reason the equity beta flips is something traders call the **reaction function** — the market's working model of how the Federal Reserve will respond to incoming data. When the Fed is in inflation-fighting mode, every data point is read through the lens of "does this push the Fed to hike more or cut less?" A strong jobs report, in that mode, signals an overheating labor market, which signals more wage pressure, which signals a more aggressive Fed, which signals higher discount rates, which crushes the present value of future corporate earnings. Strong jobs → higher-for-longer rates → stocks down. The mechanism by which a single rate read propagates to every asset is laid out in the macro-trading map [how policy moves every asset](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map); the deep dive on why the same print moves differently across regimes is the event-trading piece [the reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently).

When the Fed is *not* fighting inflation — when inflation is near target and the worry, if anything, is that growth is fading — the same strong jobs report is read through a completely different lens: "the economy is healthy, earnings will hold up, recession risk is lower." Strong jobs → growth confirmed → stocks up. The switch between these two readings is the reaction function, and it is why the cover figure above routes one jobs surprise into two opposite equity outcomes while sending the bond and dollar reaction down the same path in both cases.

## The signature chart: the NFP sign-flip

Everything above is theory. Here is the empirical fingerprint of the flip — the centerpiece of this post.

![NFP surprise beta by regime grouped bars showing S and P sign flip](/imgs/blogs/nfp-and-asset-prices-the-king-of-data-correlation-2.png)

The chart shows the same-session reaction of four assets to a +100,000 upside surprise in payrolls, split into two regimes. Read it left to right and the story is unmistakable. The **S&P 500** bar is red (down ~0.50%) in the 2022-23 good-news-is-bad regime and green (up ~0.35%) in a normal expansion — it *flips sign*. The **dollar (DXY)** rises in both regimes (about +0.30% then +0.20%): a stronger labor market means higher US rates, and higher US rates pull capital toward dollars regardless of the equity read. **Gold** falls in both (about −0.50% then −0.20%): gold competes with real yields, and a hot jobs print lifts real yields. And the **US 2-year Treasury yield** — the maturity most sensitive to the expected Fed path — rises in both regimes (about +6bp then +4bp), because in *both* worlds a strong jobs number means the Fed does slightly less cutting or slightly more hiking than previously priced.

This is the asymmetry to burn into memory: **bonds and the dollar are honest; only stocks are moody.** The rate read is unambiguous — strong jobs always nudge the expected policy rate up. What changes is whether the equity market interprets "higher rates" as a threat (discount-rate fear) or shrugs it off because the growth news dominates. The bigger 2Y move in the fear regime (+6bp vs +4bp) reflects a market on a hair trigger for hawkish surprises.

#### Worked example: decomposing the bond-versus-stock split

A trader runs a \$200,000 portfolio: \$100,000 in an S&P fund and \$100,000 short the 2-year note via futures (a bet that yields rise / prices fall, with a position sized so a +1bp move in yield earns about \$185 — roughly the DV01 of \$1mm of 2Y notes scaled to this size). A +100,000 jobs surprise hits in the 2022-23 regime.

The equity leg: −0.50% × \$100,000 = **−\$500**.

The rates leg: the 2Y yield rises +6bp; the short profits +6 × \$185 ≈ **+\$1,110**.

Net: −\$500 + \$1,110 = **+\$610** on the day. The trader was *right about the economy* (jobs were strong) and *right about rates* (yields rose) but *lost money on stocks* — and the rate leg more than paid for it. The intuition: in the fear regime, being long the rate read and underweight stocks is the coherent way to express a hot jobs print, because the rate reaction is the reliable part and the equity reaction is the part that fights you.

### Where this lands in the broader regime story

The NFP sign-flip is one instance of a much bigger pattern: correlations are conditional on the macro regime, never constant. That principle — and the formal way to test whether you are in one regime or another — is the spine of [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant). The jobs report is simply the loudest, clearest example, because its sign flip is so violent and so frequent.

## The labor backdrop: reading the regime from the unemployment rate

You cannot guess the sign of the jobs-stocks correlation without knowing the labor regime, and the cleanest single gauge of that regime is the unemployment rate itself.

![US unemployment rate path 2020 to 2026 with tight labor band shaded](/imgs/blogs/nfp-and-asset-prices-the-king-of-data-correlation-3.png)

The chart traces the US unemployment rate from the COVID spike to 14.8% in April 2020 through the long, remarkable stretch below 4% in 2022-23, then the gentle drift back up toward 4.3% by 2026. The shaded red band marks the "tight labor" zone under 4% — and that band is *exactly* the period in which a hot jobs print was bearish for stocks. When unemployment is at a 50-year low, the labor market is the bottleneck, wages are the inflation risk, and any additional strength is read as a reason for the Fed to lean harder. As unemployment drifts back up toward and above 4%, the bottleneck eases, the inflation fear fades, and the market gradually reverts to the normal reading where strong jobs are simply good news.

This is why the unemployment rate is your **regime dial**. Combined with where inflation sits relative to the Fed's target, it tells you which side of the sign flip you are likely on *before* the next report even prints. When unemployment is very low and inflation is above target, expect the good-news-is-bad regime. When unemployment is normal-to-rising and inflation is near target, expect strong jobs to be read as growth confirmation.

There is a further subtlety in *how* the unemployment rate moves that the market watches closely. The rate is a ratio — unemployed people divided by the labor force — so it can fall for a good reason (people found jobs) or a bad reason (discouraged people stopped looking and left the labor force, shrinking the denominator). This is why the **participation rate** matters: a falling unemployment rate alongside *rising* participation is genuinely strong, while a falling rate alongside *falling* participation can mask a weakening market. Markets also keep an eye on the rate of *change* in unemployment, not just its level — a well-known rule of thumb, the Sahm rule, flags recessions when the three-month-average unemployment rate rises about half a percentage point above its recent low, because unemployment tends to rise gently and then accelerate once a downturn takes hold. The level tells you the regime; the *velocity* tells you whether the regime is about to turn. A trader who watches both is far less likely to be caught on the wrong side of the next sign flip.

#### Worked example: using the regime dial to set your prior

It is the night before a jobs report. Unemployment is 3.5% (deep in the tight band) and core inflation is running at 4.5% year-over-year, well above the 2% target. Consensus payrolls is +180,000. You are long \$80,000 of equities. What is your prior?

The regime dial says: tight labor + above-target inflation = good-news-is-bad. So your prior for a *strong* print (say +280,000, a +100k surprise) is an equity move of about −0.50%, or **−\$400** on your position, with the 2Y up about +6bp. Your prior for a *weak* print (say +80,000, a −100k surprise) is the mirror image: about +0.50%, or **+\$400**, as the market cheers a cooling labor market that takes pressure off the Fed. The intuition: in this regime you are effectively *short* the labor market through your long-equity book — you want the jobs number to *miss*, which feels deeply counterintuitive until you internalize the reaction function.

## Strong jobs as a growth read: the risk-on correlation

To make the "normal expansion" side of the flip concrete, it helps to see what a positive growth surprise correlates with when the market is *not* afraid of inflation. The jobs report, in a benign regime, is essentially a growth surprise, and growth surprises have a well-documented risk-on signature across assets and sectors.

![Asset and sector correlation with a positive growth surprise risk on read](/imgs/blogs/nfp-and-asset-prices-the-king-of-data-correlation-4.png)

When growth surprises to the upside (rising ISM/PMI, of which a strong jobs report is a cousin), the assets that rise are the cyclical, economically-sensitive ones: **cyclical sectors like industrials** (correlation about +0.65), **copper** (+0.60), **small caps** (+0.55), and the **S&P 500** as a whole (+0.55). Yields rise too (+0.45) — that part never changes. And on the other side, the defensive and safe-haven assets fall or stagnate: **long Treasuries** (−0.45) because growth lifts yields, while **consumer staples** (−0.10) and **gold** (−0.05) go roughly nowhere because their appeal is precisely that they do *not* need a strong economy. The relationship between the broader business cycle and which cyclicals lead is developed in the sibling post [ISM/PMI: the leading correlation with cyclicals](/blog/trading/macro-correlations/ism-pmi-the-leading-correlation-with-cyclicals).

This is the picture that *would* govern the jobs report's effect if the market were not in inflation-fear mode. Notice that the S&P correlation here is firmly positive — the opposite of the 2022-23 reading. Same economic signal, different reaction function, different sign.

#### Worked example: the cyclical-versus-defensive spread trade

A trader believes the economy is in a clean expansion (unemployment 4.1%, inflation 2.2%, no Fed fear) and wants to express "strong jobs = risk on" without taking pure market direction. She goes long \$60,000 of an industrials (cyclical) ETF and short \$60,000 of a consumer-staples (defensive) ETF — a market-neutral spread. A jobs report lands +120,000 above consensus.

Using the growth-surprise correlations as rough betas, scale them to this surprise. Cyclicals respond strongly positively, staples barely at all. If the cyclical leg gains about +1.0% and the staples leg gains about +0.1% on the day, the spread earns:

```
long leg:  +1.0% × $60,000 = +$600
short leg: -0.1% × $60,000 = -$60
net:       +$600 - $60     = +$540
```

The intuition: in an expansion, a strong jobs report does not just lift the market, it *rotates* it toward cyclicals — and the cyclical-minus-defensive spread captures that rotation while staying largely immune to whether the index itself happened to be up or down that day.

## The both-fell regime: why bonds stopped diversifying stocks

The NFP sign-flip is intimately tied to one of the most important regime shifts of the last decade: the flip in the **stock-bond correlation**. For most of the post-2000 era, when stocks fell, bonds rose — bonds were the airbag in a 60/40 portfolio. In 2022, that broke.

![Stock bond rolling correlation 1990 to 2025 showing 2022 flip to positive](/imgs/blogs/nfp-and-asset-prices-the-king-of-data-correlation-6.png)

The chart shows the rolling stock-bond correlation from 1990 to 2025. From roughly 2000 to 2021 it sat firmly negative (the shaded green band) — the "bonds diversify stocks" era that made the 60/40 portfolio so comfortable. Then in 2022 it snapped *positive*, to about +0.60 (the shaded red band), and stayed positive through 2023-25. The driver is exactly the same reaction function that flips the NFP sign: when inflation and rates are the dominant risk, a hot data point hurts *both* stocks (higher discount rate) *and* bonds (higher yields = lower bond prices). They fall together. This is why a blowout jobs report in February 2023 sold off equities *and* Treasuries simultaneously — there was no hiding place, because the same force hit both.

This connects two ideas. First, the NFP sign-flip and the stock-bond correlation flip are *the same phenomenon viewed from two angles*: both are symptoms of an inflation-dominated reaction function. Second, it kills the comforting assumption that bonds will cushion an equity drawdown driven by hot data. When the jobs report is the threat, your bonds are not a hedge — they are a second source of the same loss. The mechanics of why the stock-bond correlation lives in inflation regimes are developed in the cross-asset companion [stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).

#### Worked example: the failed 60/40 hedge

An investor holds a classic 60/40 portfolio worth \$100,000: \$60,000 in the S&P and \$40,000 in long Treasuries, expecting the bonds to cushion equity shocks. A blowout jobs report hits in the 2022-23 regime. The S&P drops 0.50% and long Treasuries, with their high duration, drop about 0.80% as yields spike.

```
equity leg: -0.50% × $60,000 = -$300
bond leg:   -0.80% × $40,000 = -$320
total:      -$300 + (-$320)  = -$620
```

The bonds did not cushion anything — they *lost more* than the stocks. The intuition: the diversification you paid for assumed a negative stock-bond correlation, but in an inflation-fear regime that correlation is positive, so a hot jobs print drains both pockets at once.

## The internals-versus-headline trap: which component the market actually trades

So far we have treated "the jobs report" as a single number. In practice the report has five numbers, and the most dangerous beginner mistake is reacting to the headline before checking what the *components* say. The market often trades the internals over the headline.

![Headline versus internals matrix showing which component maps to the Fed path](/imgs/blogs/nfp-and-asset-prices-the-king-of-data-correlation-5.png)

The matrix lays out each component, what it measures, how it maps to the Fed path, and the trap it can spring. The headline payroll number is the loudest, but in an inflation-fear regime **average hourly earnings is often the real driver**, because wages are the most direct read on inflation. A report can show hot payrolls and *cool* wages — and the market will frequently rally bonds and stocks on the cool wage print, treating the strong headline as old news. The opposite also happens: a so-so headline with red-hot wages can sell off, because the inflation read trumped the jobs count.

Three internal traps recur:

1. **Headline hot, wages cool.** Lots of jobs, but at modest pay. The market may shrug off the headline and focus on the benign inflation read — bullish for both stocks and bonds even in a fear regime.
2. **Household survey diverges from establishment.** If payrolls (establishment) are strong but the household survey shows job *losses*, the market's confidence in the headline drops. Persistent divergence often gets read as late-cycle weakness leaking through.
3. **Big downward revisions.** A +200k headline with −90k of revisions to prior months is really a weaker report than it looks. Revisions are easy to miss in the first headline-driven seconds but can dominate within minutes.

The principle: **the market trades the component that maps most directly to the Fed path.** In an inflation regime that is usually wages; in a growth-scare regime it might be the unemployment rate or the household survey. The headline payroll number is the king of data, but the king does not always cast the deciding vote.

#### Worked example: the household-versus-establishment divergence

In a single month, the establishment survey reports +180,000 payrolls (a small upside surprise versus +160,000 consensus), but the household survey shows the number of employed people *fell* by 120,000 and the unemployment rate ticked up from 4.0% to 4.2%. You hold \$50,000 in a small-cap ETF, which is highly sensitive to recession fear. Which survey does the market believe?

The two surveys measure different things and routinely disagree by tens of thousands, but a *persistent* divergence where the household survey weakens while payrolls hold up is a classic late-cycle warning — it often means hiring is concentrating in a few sectors while the broad labor market is cracking. If the market reads it as a soft headline masking real weakness, small caps can fall even on a "beat." Say the small-cap ETF drops 1.2% on the recession read: that is −1.2% × \$50,000 = **−\$600**, despite the headline payroll number being *above* consensus. The intuition: the king of data has a court, and when the unemployment-rate courtier contradicts the payroll king, a nervous market sometimes sides with the courtier — a headline beat can still be a risk-off day.

#### Worked example: the wages-versus-payrolls split

A jobs report prints +250,000 payrolls against a +180,000 consensus — a hot +70k surprise. But average hourly earnings rise just +0.1% month-over-month against a +0.3% consensus — a clear *downside* wage surprise. You hold \$50,000 in the S&P during a fear regime. Which signal wins?

If you traded the headline alone, your prior is bearish: +70k surprise × −0.50% per 100k ≈ −0.35%, or about **−\$175**. But the cool wage print is a *dovish* inflation signal, and in a fear regime the market often weights wages more heavily. The realized move might instead be *positive* — say +0.4%, or **+\$200** — as the market cheers that the labor market grew without adding wage pressure. The intuition: a hot headline and a cool wage print pull in opposite directions, and the inflation-sensitive component usually wins when inflation is the dominant fear, so reading only the headline would have put you on exactly the wrong side.

## Lead, lag, and where NFP sits in the labor data calendar

Every correlation in this series has a fourth property beyond sign, strength, and regime-dependence: a **lead or lag**. Does the indicator move *before* the thing it correlates with, or *after*? For trading, an indicator that leads is far more valuable than one that lags, because it gives you advance warning. Where does the jobs report sit?

Nonfarm payrolls is, for the most part, a **coincident** indicator: it tells you about the state of the economy roughly *now*, not three months from now. It is a snapshot of hiring in the reference month, released a few days into the next month. It confirms the cycle rather than predicting it. That is not a weakness — a clean, broad, timely snapshot of the most important variable in the economy is enormously useful — but it does mean NFP is not where you look for an *early* recession warning.

For that, the labor market offers a faster, more forward-looking cousin: **initial jobless claims**, reported weekly. Claims are higher-frequency (every Thursday rather than once a month) and they tend to turn *before* the unemployment rate does, because layoffs show up in claims immediately while the unemployment rate is a slower-moving stock. A sustained rise in claims is one of the most reliable early signals that the labor market is rolling over, and we devote a full sibling post to it: [unemployment claims and the recession correlation](/blog/trading/macro-correlations/unemployment-claims-and-the-recession-correlation). The practical hierarchy: **claims lead, payrolls confirm, the unemployment rate lags.** A trader watching for a regime change reads claims for the early warning, then waits for payrolls and the unemployment rate to confirm it.

This lead/lag structure also explains a subtle point about the sign flip. Because payrolls is coincident, it tells you about the labor regime you are *already in* — which is exactly the regime that determines its own sign. The unemployment rate, which prints in the same report, is your regime dial precisely because it is a slow-moving stock: it does not whipsaw month to month, so it is a stable read on whether you are in the tight-labor (good-news-is-bad) world or the normal-expansion world. The broader sequencing of which indicators lead, coincide, and lag across the cycle is the subject of [the business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock).

#### Worked example: claims warn, payrolls confirm, and your equity book reacts

A trader runs a \$120,000 equity book. Over six weeks, initial jobless claims climb from 215,000 to 265,000 — a clear upward break — while the most recent payroll print was still a solid +175,000. The claims signal says the labor market is cracking; payrolls has not confirmed yet. The trader trims equity exposure by a third ahead of the next jobs report, moving \$40,000 into cash. The next NFP confirms the weakening: payrolls comes in at +35,000, far below the +150,000 consensus, and the unemployment rate jumps. In a late-cycle growth-scare regime, that weak print *hurts* stocks (recession fear), and the S&P falls 1.5%. On the \$80,000 the trader kept in equities the loss is −1.5% × \$80,000 = **−\$1,200**; had the full \$120,000 stayed invested it would have been −1.5% × \$120,000 = **−\$1,800**. The claims lead saved **\$600**. The intuition: payrolls confirms the regime, but the higher-frequency claims data gives you the head start to position before the confirmation arrives.

## Crypto and the highest-beta version of the flip

Stocks are the moody asset in the jobs-report reaction. Crypto is stocks on amplifier. Bitcoin and the broader crypto complex trade, much of the time, as the highest-beta expression of macro risk appetite — when the reaction function turns hawkish on a hot jobs print, crypto often falls harder than equities, and when the growth read dominates, it rallies harder.

This was most visible in 2022, when Bitcoin's correlation with the Nasdaq 100 spiked to roughly +0.65 as both traded primarily off the same macro-liquidity and rate-fear signals. In that window, a hot jobs report that knocked the Nasdaq down 1% could knock Bitcoin down 2-3%, because crypto carried the same rate-fear sensitivity with a larger beta. As the rate shock faded into 2024-25, that correlation drifted back down toward +0.2 to +0.3 — crypto became somewhat less of a pure macro-rate proxy and traded more on its own idiosyncratic flows (ETF demand, halving cycles, regulation). The lesson mirrors the whole series: even crypto's macro correlation is a regime, not a constant. It was a high-beta rates proxy when rates were the dominant macro force, and it loosened from that grip when rates stopped being the only thing that mattered.

For the trader, the implication is a leverage warning. If you want to express "hot jobs print, hawkish regime, risk-off" and you do it through crypto, you are taking the same directional view as a short-equity position but with roughly double the magnitude — and double the magnitude cuts both ways. A misjudged regime that would cost you 0.5% in equities can cost you 1.5% or more in crypto.

#### Worked example: the crypto beta on a hawkish jobs print

A trader holds \$30,000 of Bitcoin during the 2022-style hawkish regime, when BTC's correlation with the Nasdaq is about +0.65 and its effective beta to a hot jobs print is roughly 3× the S&P's. A +100k upside payroll surprise hits; the S&P falls about 0.50%, so the crypto beta implies roughly −1.5%. The Bitcoin P&L:

```
-1.5% × $30,000 = -$450
```

That is three times the −\$150 a same-sized \$30,000 S&P position would have lost (−0.50% × \$30,000 = −\$150). The intuition: in a rate-fear regime, crypto carries the equity sign but a much larger magnitude, so the jobs-report flip that merely bruises a stock book can carve a real hole in a crypto book — size the position for the beta, not just the direction.

## Common misconceptions

Five myths about the jobs report and its correlations, each corrected with a number.

**Myth 1: "A strong jobs report is bullish for stocks."** Only sometimes. In the 2022-23 inflation-fear regime the S&P's beta to a +100k upside surprise was about **−0.50%** — strong jobs were *bearish*. The sign is conditional on the regime, not fixed. Anyone who tells you NFP is "obviously" good or bad for stocks without naming the regime is guessing.

**Myth 2: "The headline payroll number is what moves the market."** The market often trades the *internals*. A hot payroll headline with a cool average-hourly-earnings print can rally both stocks and bonds, because wages are the more direct inflation read. In an inflation regime, the wage number frequently overrides the job count entirely.

**Myth 3: "Bonds will hedge my stocks when bad data hits."** Not when the data is the threat. In an inflation-fear regime the stock-bond correlation is about **+0.60**, so a hot jobs report drops *both* — a 60/40 portfolio can lose on both legs simultaneously, as the failed-hedge worked example showed.

**Myth 4: "A falling unemployment rate is always good."** It depends on *why* it fell. If unemployment falls because people found jobs, that is healthy. If it falls because discouraged workers gave up and left the labor force (a falling participation rate), the same headline drop masks a weakening labor market. The household survey internals matter as much as the print.

**Myth 5: "The level of the jobs number is what matters."** The *surprise* is what matters — actual minus consensus. A +185,000 print against a +185,000 consensus is a non-event no matter how "strong" 185,000 sounds, because that strength was already priced. Markets trade the gap, not the level. (Same principle as [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises).)

## How it shows up in real markets

Let us ground the sign-flip in dated episodes, with the dollar and bond reactions that accompanied each.

### February 2023: the textbook good-news-is-bad print

The January 2023 report (released 3 February 2023) added 517,000 jobs against a ~185,000 consensus — a roughly +332k upside surprise — with unemployment at a 53-year low of 3.4%. The market was deep in inflation-fear mode: the Fed had been hiking aggressively, and core inflation was still well above target. The reaction was the pure fear-regime template. Stocks fell, the 2-year yield jumped roughly fifteen basis points as the market priced more Fed tightening, and the dollar surged. Strong economy, falling stocks. This is the canonical example of the negative jobs-stocks correlation.

#### Worked example: sizing the February 2023 reaction

Take that +332k surprise and the fear-regime equity beta of −0.50% per 100k. The implied S&P move is:

```
-0.50% × (332k / 100k) = -1.66%
```

On a \$50,000 S&P position that is about **−\$830** of paper loss from a single data release. The 2Y, at roughly +6bp per 100k, implies +6 × 3.32 ≈ +20bp — in the same ballpark as the realized ~15bp move (event-study betas are averages; any single print scatters around them). The intuition: an outsized surprise scales the whole reaction proportionally, which is why the biggest payroll shocks produce the biggest single-day moves on the calendar.

### A strong print in a benign year

Now consider a strong jobs report in a year when inflation is near target and the Fed is on hold or cutting — a mid-cycle expansion. A +100k upside surprise in that environment rallies the S&P by roughly +0.35%, lifts cyclicals and small caps more, pushes the 2Y up a smaller +4bp (the market is not on a hawkish hair-trigger), and nudges the dollar up modestly. Gold dips slightly on the firmer yields. The economy-is-healthy reading dominates. Same +100k surprise, opposite equity sign — the entire thesis of this post in two episodes side by side.

#### Worked example: the dollar leg across both regimes

A currency trader is long \$100,000 notional of the dollar against a basket (a bet that DXY rises). A +100k jobs surprise hits. In the fear regime DXY rises about +0.30%; in the benign regime about +0.20%. The dollar P&L:

```
fear regime:   +0.30% × $100,000 = +$300
benign regime: +0.20% × $100,000 = +$200
```

In *both* regimes the dollar trade made money — the sign never flipped. The intuition: the dollar tracks the rate read, and a strong jobs report always nudges US rates up relative to the rest of the world, so being long the dollar into a strong print is one of the few NFP trades whose sign does not depend on guessing the equity regime.

### The 2024-26 normalization

As unemployment drifted from its sub-4% lows back toward 4.3% and inflation cooled toward target, the market's reaction function gradually softened. Jobs reports through 2024-25 increasingly traded as growth reads rather than inflation threats: a weak print sometimes *hurt* stocks (recession fear creeping in), and a strong print sometimes *helped* them (soft-landing confirmation) — the opposite mapping from 2022-23. This is the regime dial turning in real time, and it is why a static rule like "NFP is bearish for stocks" would have whipsawed a trader who learned it in 2023 and applied it in 2025.

The transition itself is the treacherous part. In a "good news is bad news" regime, you fade strength; in a "good news is good news" regime, you buy it. But the switch does not flip on a known date — it migrates as inflation cools and unemployment normalizes, and for a stretch the market itself is unsure which regime governs, so reactions become inconsistent. A weak print might rally stocks one month (the Fed will cut!) and sink them the next (recession is here!). The only defense is to stop trading a fixed rule and start trading the regime dial: re-read unemployment and inflation before *every* report, and accept that in the transition zone your conviction on the equity sign should be low.

#### Worked example: the whipsaw of trading a stale rule

A trader internalized the 2022-23 rule "strong jobs = sell stocks" and keeps applying it into a 2025 normalization regime. A jobs report beats by +90,000. Following the stale rule, the trader shorts \$40,000 of S&P expecting a fall of about 0.45% (a target gain of about +\$180). But the regime has flipped: inflation is near target, the market reads the strong print as soft-landing confirmation, and the S&P *rises* 0.40%. The short loses −0.40% × \$40,000 = **−\$160**. The same rule that earned money in 2023 now loses it in 2025. The intuition: the sign flip means a profitable jobs-report strategy has a shelf life — when the regime turns, the strategy must turn with it, or the very setup that paid you starts paying the other side.

## How to read it and use it

Here is the playbook — the signal, the regime check, and what invalidates it.

**Step 1 — Identify the regime before the report.** Use the regime dial: the unemployment rate and inflation relative to the Fed's target. Low unemployment (sub-4%) plus above-target inflation → good-news-is-bad regime, expect the negative jobs-stocks sign. Normal-to-rising unemployment plus near-target inflation → growth-read regime, expect the positive sign. The business-cycle phase ties this together; see [the business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock).

**Step 2 — Trade the surprise, not the level.** Have the consensus in hand and compute actual minus consensus the instant the number prints. A strong-sounding number that matches consensus is a non-event.

**Step 3 — Check the internals before reacting to the headline.** In an inflation regime, average hourly earnings can override the payroll count. Look at wages, the unemployment-rate composition (did participation move?), the household-versus-establishment divergence, and the revisions. The headline is the king of data, but the deciding vote often comes from a component.

**Step 4 — Remember which assets are honest.** Bonds and the dollar respond to the rate read and rarely flip sign — a strong print sells the front end and lifts the dollar in essentially every regime. It is only equities (and risk assets like crypto) whose sign depends on the reaction function. If you are unsure of the equity sign, the rates and FX legs are the higher-conviction expressions of a payroll surprise.

**Step 5 — Know what invalidates the signal.** The sign flip is invalidated by a regime change: if the Fed pivots from fighting inflation to supporting growth, the equity beta flips from negative to positive, often before it is obvious in the data. Watch for the unemployment rate crossing 4%, inflation crossing back below target, or the Fed explicitly shifting its language. When the regime turns, last year's "obvious" jobs trade becomes this year's losing trade. And always remember the broader humility of this whole series: correlation is not causation, these betas are noisy averages around which any single print scatters widely, and the most dangerous moment is the one where last year's rule no longer holds. A complementary labor signal — initial jobless claims and its tight link to recessions — is covered in [unemployment claims and the recession correlation](/blog/trading/macro-correlations/unemployment-claims-and-the-recession-correlation).

**A note on position sizing around the report.** Because NFP has the highest expected move of any data day, it is also the day on which an over-sized position does the most damage. The professional discipline is to size *before* the print so that even a beta at the high end of its scatter — say, a −0.9% S&P move on a hot print in a fear regime when your model said −0.5% — is survivable. If you cannot stomach the worst-case move for your position size, you are too big going into the report. Many traders deliberately reduce exposure into NFP and re-establish it afterward, paying a small opportunity cost to avoid being forced out of a good thesis by a single noisy print. The asymmetry to respect: the jobs report rewards a clear regime read and punishes both a wrong sign *and* excessive size, so getting the direction right but the size wrong can still blow up the trade.

**Putting the playbook together.** The full sequence is: read the regime dial (unemployment plus inflation versus target) to set your prior on the equity sign; have consensus in hand to compute the surprise the instant it prints; check the internals — wages first in an inflation regime, the household survey and revisions otherwise — to see whether a component overrides the headline; lean on the honest assets (bonds and the dollar) when you are unsure of the equity sign; size for the worst-case beta, not the average; and re-run the whole check before the *next* report, because the regime that determined the sign this month may have started to turn. Do that consistently and you have converted the loudest, most confusing number on the calendar from a source of whipsaw into a structured, regime-aware edge.

## Further reading & cross-links

Within this series:

- [The surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — the framework for turning any data release into a price move.
- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — why the NFP sign flip is one instance of a universal rule.
- [ISM/PMI: the leading correlation with cyclicals](/blog/trading/macro-correlations/ism-pmi-the-leading-correlation-with-cyclicals) — the growth-surprise sibling of the jobs report.
- [Unemployment claims and the recession correlation](/blog/trading/macro-correlations/unemployment-claims-and-the-recession-correlation) — the higher-frequency labor signal.
- [The business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) — which regime you are in and what rotates.
- [What correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) — the statistics behind every number here.

Mechanism and reaction (other complete series):

- [The jobs report (NFP): the king of data](/blog/trading/event-trading/the-jobs-report-nfp-the-king-of-data) — the release-day mechanics and the first thirty seconds of the tape.
- [The reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) — the deep dive on the switch that flips the sign.
- [How policy moves every asset: the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) — how one rate read propagates to every market.
- [Stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) — why bonds stopped hedging stocks in 2022.
