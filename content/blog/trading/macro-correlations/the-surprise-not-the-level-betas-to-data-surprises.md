---
title: "Correlate the Surprise, Not the Level: Betas to Macro Data Surprises"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Markets price expectations, so an asset barely moves with the level of an indicator and moves instead with the surprise; the right number is the beta of return to that surprise."
tags: ["macro", "correlation", "data-surprise", "beta", "event-study", "cpi", "nfp", "expectations", "regime", "trading"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — An asset price barely correlates with the *level* of a macro indicator because the level is already priced; it correlates with the *surprise* (actual − consensus), and the tradeable number is the **beta** of the asset's return to that surprise, estimated by an event-study regression.
>
> - The level you read in the headline is old news. The market already moved when consensus formed. Only the gap between what was expected and what printed is new information.
> - Surprise = actual − consensus. Standardized surprise = surprise ÷ its historical standard deviation, which lets you compare a CPI miss to a jobs miss on one scale.
> - Beta-to-surprise is the slope of a regression of the same-window asset return on the surprise. For a +0.1pp core-CPI upside surprise in the 2022-23 regime: S&P −0.7%, Nasdaq −1.0%, 10Y +7bp, 2Y +9bp, DXY +0.35%, gold −0.8%, Bitcoin −1.6%.
> - The sign of the beta **flips with the regime**. A strong-jobs surprise was *bad* for stocks in 2022-23 (good-news-is-bad) and *good* in a normal expansion. The beta is a regime-conditional number, not a constant.

## When the number that "matters" didn't matter

On 13 September 2022, the US Bureau of Labor Statistics released the August Consumer Price Index. Headline inflation came in at 8.3% year-over-year. Eight-point-three percent is an enormous, frightening number — four times the Federal Reserve's 2% target, the kind of inflation the United States had not seen in forty years. If asset prices simply tracked the *level* of inflation, that morning should have been roughly as ugly as the month before, when inflation was 8.5%. Inflation had, after all, gone *down*.

Instead, the S&P 500 fell 4.3% in a single session — one of its worst days of the entire cycle. The Nasdaq dropped more than 5%. The two-year Treasury yield, the market's bet on where the Fed funds rate is heading, jumped. The dollar surged. Why? Because economists had penciled in 8.1%, and the core measure (which strips out food and energy) came in hotter than expected too. The *level* fell from 8.5% to 8.3%. The *surprise* was positive — hotter than consensus — and the surprise was what the market traded.

This is the single most important measurement subtlety in all of macro correlation, and almost every beginner gets it backwards. They plot the level of an indicator against an asset price, find a weak or noisy relationship, and conclude either that "fundamentals don't matter" or that "the market is irrational." Both conclusions are wrong. The market is exquisitely rational about data; it simply prices the *expectation* in advance and reacts only to the *error*. Once you measure the right object — the surprise, and the asset's beta to it — the noise resolves into one of the cleanest, most exploitable relationships in finance. This post is about how to measure it.

![Pipeline showing consensus and actual produce a surprise, which through a regression gives a beta and a price move](/imgs/blogs/the-surprise-not-the-level-betas-to-data-surprises-1.png)

This series treats a correlation as a *measurable object* with a sign, a strength, a lead/lag, and a regime in which it holds. This post is the bridge: it is where a vague claim like "inflation is bad for stocks" becomes a concrete, tradeable number like "−0.7% per +0.1pp core-CPI surprise, in the inflation-fear regime." If you only read one post in the series to understand *how* the correlations the rest of the series describes are actually estimated, read this one. (For the separate question of what the correlation *coefficient* itself means — Pearson, Spearman, beta — see [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta).)

## Foundations: why the level is already in the price

Before any math, build the intuition from zero. A market price is not a measurement of the present; it is a *weighted bet about the future*. The price of the S&P 500 today already reflects everyone's best collective guess about earnings, interest rates, inflation, and growth for years to come. That guess incorporates every piece of public information, including the *expected* value of next month's inflation report.

Here is the everyday analogy. Suppose a famous chef is opening a restaurant, and food critics have spent weeks saying "this will be the best restaurant in the city." On opening night, the reviews come out and the food is, indeed, excellent. Does the restaurant's reputation jump? No — everyone already expected excellent. The reputation was *priced in* by the pre-opening hype. Now suppose the food is merely good, not spectacular. The reputation *falls*, even though the food was objectively good, because it fell short of what everyone expected. And if the food is transcendent, beyond even the hype, the reputation soars. The reputation moves on the *gap between expectation and reality*, not on the absolute quality of the meal.

Asset prices work exactly this way. The level of inflation, the level of payrolls, the level of GDP — these are the "expected quality of the meal." The market has already adjusted to them. What moves the price is the *deviation from the expectation*: the surprise.

### The efficient-markets argument, in one breath

Why must the expectation be priced in? Because if it weren't, there would be free money. Imagine everyone *knew* inflation would print at 8.3% next Tuesday and the S&P would fall 1% when it did. Then traders would sell *today* to capture that drop, pushing the price down today. They would keep selling until the expected Tuesday drop disappeared — until the price already reflected the 8.3% they all anticipate. The act of anticipating a move *is* what removes it from the future and folds it into the present. What's left for Tuesday is only the part nobody could anticipate: the error between 8.3% expected and whatever actually prints. This is the efficient-markets logic, and you don't have to believe markets are *perfectly* efficient to accept its core: anticipated information is largely priced, and the residual — the surprise — is what trades.

> [!note]
> This is *not* the same claim as "the news doesn't matter." The news matters enormously. It is the *anticipated* part of the news that doesn't move price, because it was already absorbed. The *unanticipated* part — the surprise — is the whole game. Confusing "the level is priced" with "data is irrelevant" is the classic beginner error.

### Consensus, whisper, and the distribution of expectations

To measure a surprise you need a number for "what was expected." That number is the **consensus**: the median or mean forecast of the economists that data providers (Bloomberg, Reuters, Dow Jones, FactSet) survey before each release. For US CPI, dozens of bank and research-shop economists submit a forecast; the consensus is the middle of that pack. When you read "8.3% vs 8.1% expected," the 8.1% is the consensus.

Two refinements matter in practice. First, the consensus is a *distribution*, not a point. If forecasters are tightly clustered (everyone says 8.1%), then any deviation is a genuine shock. If they are scattered (guesses range 7.8% to 8.5%), the "surprise" relative to the median is less informative, because the market's positioning was already hedged across the range. Second, there is often a **whisper number** — an unofficial expectation that circulates among traders and differs from the published consensus, usually because recent related data (a hot PPI, a strong jobs report) has nudged the real-time expectation away from the stale survey. The market frequently trades against the *whisper*, not the published consensus, which is why a print can sometimes match the official consensus exactly and still move the market: it missed the whisper.

For the rest of this post we will use the published consensus as our expectation, because it is the number you can actually look up and regress against. Just keep in mind that the cleanest surprises are the ones where the consensus was tight and the whisper agreed with it.

### Where the expectation actually lives in the price

It helps to be concrete about *how* an expectation gets embedded in a price, because that is what makes "the level is priced" more than a slogan. Take the 2-year Treasury yield. The 2-year yield is, to a very good approximation, the market's average expected Fed funds rate over the next two years, plus a small term premium. So the 2-year yield is literally a number that *is* an expectation: it encodes the market's forecast of every Fed meeting between now and two years out. When the market expects inflation to stay hot and the Fed to keep hiking, that expectation is already *in* the 2-year yield today. The release of a hot CPI only moves the 2-year if it changes the *expected* Fed path — that is, if the print was hotter than the path already embedded in the curve.

This is why the 2-year repriced *more* than the 10-year on hot CPI surprises (the +9bp versus +7bp betas above). The 2-year is almost a pure expectation of near-term Fed policy, which is exactly what a CPI surprise directly revises. The 10-year blends the near-term path with long-run inflation and growth expectations and a larger term premium, so a single CPI surprise is a smaller fraction of what it prices. The *closer* an instrument is to being a pure expectation of the thing your data informs, the larger its beta to that data's surprise. That single rule explains a surprising amount of the cross-asset beta structure: front-end rates react most to data that revises the near-term Fed path, long-end rates and equities react through the discount-rate channel, and gold reacts through the real-yield slice of the long end.

### Rational expectations does not require everyone to be right

A common objection: "but forecasters are often wrong, so how can the expectation be priced?" The resolution is that the price embeds the *consensus* expectation, and the consensus only has to be *unbiased on average*, not correct on any given print. If forecasters were systematically too low, traders would learn to add a fudge factor, and the bias would disappear from the price. What remains is an expectation that is right on average and wrong by a *random* amount each month — and that random amount is precisely the surprise. The surprise is, by construction, the *unforecastable* residual of the data. That is a deep reason the surprise is the only part that moves price: it is the only part that *couldn't* have been priced, because nobody could have known it. The forecastable part was priced; the unforecastable part is the news.

### Surprise = actual − consensus

With the expectation defined, the surprise is a subtraction:

```
surprise = actual - consensus
```

For the August 2022 CPI, if we use the core month-over-month measure where actual was 0.6% and consensus was 0.3%, the surprise was +0.3pp — a large upside (hotter) surprise. A *positive* surprise on an inflation measure means inflation came in *above* expectations (hotter, more hawkish for the Fed). A *positive* surprise on a growth or jobs measure means the economy was *stronger* than expected. The sign convention is "actual minus consensus," so positive always means "more than expected," whatever the indicator.

![Number line showing consensus already priced and the actual print, with the surprise as the gap that moves price](/imgs/blogs/the-surprise-not-the-level-betas-to-data-surprises-5.png)

The figure above is the whole idea in one picture. The expected band — say core CPI of 0.20% to 0.30% — sits inside the price already; it is gray, inert, old news. The actual print lands at 0.40%. Only the +0.10pp gap between consensus and actual is yellow: that is the news, the part that wasn't already in the price. The price reacts to that gap and nothing else.

#### Worked example: computing a raw surprise

The November 2023 core CPI printed at 0.28% month-over-month against a consensus of 0.30%. The surprise is:

```
surprise = 0.28% - 0.30% = -0.02pp
```

A small *negative* (cooler-than-expected) surprise. Even though core inflation that month was running near 4% year-over-year — a high level by historical standards — the print was a hair *below* what the market expected, so the market read it as good news. The S&P rose that day. The intuition: a high level of inflation can still produce a *positive* market reaction if the print comes in below the (high) consensus. The level was already feared and priced; the cooler-than-feared surprise is what moved the tape.

### Standardizing the surprise: putting CPI and NFP on one scale

A raw surprise has a problem: its units depend on the indicator. A CPI surprise is in percentage points; a payrolls surprise is in thousands of jobs; a GDP surprise is in annualized percent. You cannot compare "+0.1pp on CPI" with "+100k on payrolls" — they live in different worlds. Worse, "big" depends on the indicator's own noisiness. A +0.1pp CPI surprise is large because CPI forecasts are usually accurate to within a few hundredths of a point; a +100k payrolls surprise is roughly normal because the monthly payrolls number routinely beats or misses by that much.

The fix is to **standardize** the surprise by dividing by its own historical standard deviation:

```
standardized surprise (z) = (actual - consensus) / std(historical surprises)
```

This converts every surprise into a *z-score* — a number of standard deviations. A z of +1.0 means "a one-standard-deviation upside surprise," whatever the indicator. Now a CPI shock and a payrolls shock are directly comparable, and you can ask "which print was the bigger shock?" in a single unit.

![Before-after diagram contrasting a raw surprise that cannot be compared with a standardized surprise on one scale](/imgs/blogs/the-surprise-not-the-level-betas-to-data-surprises-4.png)

#### Worked example: standardizing two surprises

Suppose core CPI has a historical surprise standard deviation of about 0.07pp, and payrolls has one of about 70k. A +0.10pp CPI surprise standardizes to:

```
z_CPI = 0.10 / 0.07 = 1.43 sigma
```

A +100k payrolls surprise standardizes to:

```
z_NFP = 100 / 70 = 1.43 sigma
```

These two surprises — utterly different in raw units — are the *same size* once standardized: both are about 1.4-sigma shocks. That is the payoff of standardization: it tells you both prints carried roughly equal informational punch, so you would expect them to move markets by comparable, regime-appropriate amounts. Standardizing turns "+0.1pp" and "+100k" from incomparable headlines into one common currency of surprise.

### Surprise indices: the aggregate of all the surprises

If you can standardize one surprise, you can aggregate many. A **surprise index** rolls up the recent standardized surprises across dozens of indicators into one number that says "is the economy, on net, beating or missing expectations lately?" The most-watched is the **Citi Economic Surprise Index (CESI)**: it rises when data has been coming in above consensus and falls when data has been disappointing.

The crucial property of a surprise index is that it is *mean-reverting by construction*. When data keeps beating, economists raise their forecasts, so the consensus rises to meet reality — and the next beat is harder, so the index rolls over even while the economy is still strong. A surprise index does not measure how *good* the economy is; it measures how good the economy is *relative to a constantly-updating expectation*. That makes it a measure of the *gap*, which is exactly the object that drives asset returns. Risk assets and bond yields often track the CESI more closely than they track the underlying data levels, precisely because the index is built from surprises rather than levels — the same insight as this whole post, scaled up to the whole calendar.

The mean reversion has a tradeable corollary: because a high CESI tends to revert (forecasts catch up), an extremely positive surprise index is often a *contrarian* signal for the data itself — not that the economy will weaken, but that the *rate of positive surprises* is about to slow as the bar rises. Macro desks watch the CESI not as a growth gauge but as a "how much good-news fuel is left" gauge. When the index is pinned high, the easy upside surprises have mostly happened, and the asymmetry tilts toward disappointment even in a healthy economy. This is a subtle but important point: the surprise index can fall while every underlying series is still expanding, simply because expectations have risen to meet them.

#### Worked example: reading a surprise index move

Suppose the CESI for the US is at +60 (strongly positive — data has been beating) and over the next month it falls to +10. Has the economy weakened? Not necessarily. The drop means the *flow* of beats has slowed: prints are now coming in roughly in line with the (now higher) consensus rather than well above it. If the S&P had a beta of about +0.05% per point of CESI change in a growth-led regime, the move from +60 to +10 — a −50 point change — would imply roughly:

```
implied drag = +0.05% per point  x  (-50 points)  =  -2.5%
```

a modest equity headwind, not from the economy contracting but from the *positive-surprise tailwind fading*. The intuition: a surprise index is a momentum-of-expectations gauge, so even a strong economy stops *supplying upside surprises* once forecasts catch up, and that fading tailwind alone can pressure risk assets.

## From "correlation" to a beta: the event-study regression

Now we have the right input — the surprise — we can define the right output: the **beta**. A beta is the slope of a line. Specifically, it is the slope of a regression of the asset's same-window return on the surprise:

```
return = alpha + beta * surprise + error
```

Read that equation in plain English: take many past releases of an indicator; for each one, record the surprise (actual − consensus) and the asset's return over the same window (say the 30 minutes around the release, or the close-to-close day). Plot each release as a dot — surprise on the horizontal axis, return on the vertical axis. Fit the best straight line through the cloud of dots. The slope of that line is the **beta**: the expected asset move *per unit of surprise*. The intercept (alpha) should be near zero — when the surprise is zero, the price shouldn't systematically move. The scatter of dots around the line is the residual: everything *other* than this surprise that moved the price.

![Scatter of releases with a fitted regression line whose slope is the beta of return to surprise](/imgs/blogs/the-surprise-not-the-level-betas-to-data-surprises-7.png)

This is the picture every macro correlation in this series ultimately reduces to. Each dot is one release. A cool surprise (left of zero) sits with a positive return (stocks rally); a hot surprise (right of zero) sits with a negative return (stocks fall). The downward-sloping fitted line says the relationship is negative in this regime, and its slope — about −0.7% per +0.1pp of core-CPI surprise — *is* the beta. The dots scatter around the line because no single data release is the only thing happening on a given day, but the line captures the systematic part. This is what turns a hand-wavy "inflation is bad for stocks" into a number you can trade against. (For the formal mechanics of running this regression in Python — the data alignment, the windowing, the standard errors — see the dedicated toolkit post, [measuring beta to data surprises: an event study in Python](/blog/trading/macro-correlations/measuring-beta-to-data-surprises-an-event-study-in-python).)

### Why an event study, and not a long-horizon correlation

You might ask: why measure the reaction in a *narrow window* around the release? Why not just correlate monthly inflation surprises with monthly stock returns? The answer is *signal isolation*. Over a month, a thousand things move the stock market: earnings, geopolitics, other data, positioning, flows. If you correlate monthly surprises with monthly returns, the inflation signal drowns in that noise and your beta estimate is unstable and tiny. But in the 30 minutes around the CPI release, almost nothing *else* happens — so essentially all of the move in that window is the market repricing on the surprise. Narrowing the window raises the signal-to-noise ratio enormously. This is the entire logic of an **event study**: shrink the window until the event you care about is the dominant thing inside it, then measure the reaction. The narrower the clean window, the sharper the beta.

This is also why the beta is the bridge between this series and the event-trading series. The event-trading series asks *how do I trade the release in real time* — the intraday playbook, the position, the stop. This series asks *what is the measurable relationship* — and the beta is that measurement. Same window, different question: one is the trade, the other is the number behind it. The mechanism of *why* the news moves markets at all is built up in [why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework); here we estimate the slope it implies.

### The beta, the R-squared, and the residual: three numbers, not one

A single regression gives you three numbers, and beginners usually look at only the first. They are:

1. **The beta (the slope).** The expected asset move per unit of surprise. This is the headline — the tradeable magnitude.
2. **The standard error of the beta.** How precisely the slope is estimated. A beta of −0.7 with a standard error of 0.1 is a confident, real relationship; a beta of −0.7 with a standard error of 0.6 might just be noise. You divide the beta by its standard error to get a t-statistic; roughly, a t-stat above 2 means the beta is statistically distinguishable from zero. A beta is only worth trading if it is precisely estimated, and the standard error is how you know.
3. **The R-squared (how much of the move the surprise explains).** This is the fraction of the asset's variance in the window that the surprise accounts for. In a tight intraday window around a major release, the R-squared can be high — the surprise really is most of what's happening. Over a longer window it collapses toward zero, because a thousand other things are moving the asset. The R-squared is your humility number: even a real beta might explain only, say, 40% of the day's move, with the other 60% being everything else.

The relationship between these is the whole reason an event study works. The narrow window *raises the R-squared* (the surprise becomes the dominant force) and *shrinks the standard error* (cleaner data, sharper slope). The same beta estimated over a month would have a tiny R-squared and a huge standard error — the slope might even come out the wrong sign by chance. So when you read "the S&P beta to a CPI surprise is −0.7%," the unstated companions are "with a small standard error and a high R-squared *in a 30-minute window*." Quote the beta without those and you are selling a point estimate as if it were certainty.

### The residual is information, not just error

The scatter of dots around the fitted line — the residual — is not garbage to be ignored. It is *everything other than this surprise* that moved the asset, and it carries two useful lessons. First, its *size* tells you how reliable a single-print trade is: a wide residual cloud means even a big surprise can be swamped by other forces on the day, so you should size accordingly. Second, *which* dots are far off the line often reveals a second event: an outlier release that moved the "wrong" way usually had a confound — a Fed speaker that hour, a geopolitical headline, a quarter-end flow. The residual, read carefully, is a list of the days your event study was contaminated. A disciplined analyst inspects the large residuals before trusting the beta, because they are where the clean "surprise → move" story broke down.

#### Worked example: is a beta real or noise?

Suppose you run the regression and get an S&P beta of −0.7% per +0.1pp CPI surprise with a standard error of 0.15%, over 36 monthly releases. The t-statistic is:

```
t = beta / std_error = -0.7 / 0.15 = -4.67
```

A magnitude well above 2, so the beta is strongly significant — this is a real relationship, not chance. Now suppose a friend regresses the *same* asset on the *level* of inflation instead of the surprise, over the same 36 months, and gets a beta of −0.05% per 1pp of inflation level with a standard error of 0.20% (t ≈ −0.25). That t is far below 2 — indistinguishable from zero. Same asset, same months: the surprise gives a sharp, significant beta; the level gives noise. The intuition: the statistics themselves confirm the thesis — regress on the surprise and the relationship is real and precise; regress on the level and there is essentially nothing to estimate, because the level was already in the price.

### Measurement versus the intraday trade: why this is not the event-trading post

It is worth drawing a sharp line here, because the surprise and the beta show up in two different series and they answer two different questions. The event-trading series is about *the trade*: given a release, how do you position into it, what is your stop, how do you fade or press the first move, how do the first five minutes differ from the close. It is operational and time-of-day specific. This post — and this whole macro-correlations series — is about *the measurement*: what is the slope, how precisely is it estimated, in which regime does it hold, and what does it tell you about the *structure* of the correlation.

The distinction matters because the two can disagree in the short run. The *measured* beta says a hot CPI should drop the S&P 0.7% per 0.1pp; the *intraday trade* might still lose money if the market had positioned heavily for a hot print, so the realized move is muted (a "buy the rumor, sell the fact" dynamic) or even reverses. The beta is the average, structural response across many releases; the trade is one realization, contaminated by positioning, liquidity, and the order of the day's other events. A good macro analyst holds both in mind: the beta tells you the *expected* reaction and its reliability; the intraday playbook tells you how to act on a *specific* print given that day's positioning. This post gives you the first. Treat the beta as the anchor and the realized move as the anchor plus noise — and when the noise is large, that is itself the news.

### The units problem: a beta has a denominator

A beta is meaningless without its units, because it is a *ratio*: move per unit of surprise. "The beta is −0.7" tells you nothing until you specify −0.7 *of what* per *how much* surprise. Is it −0.7% per +0.1pp of CPI surprise? Per 1pp? Per one standardized sigma? These are wildly different claims. Always carry the full denominator. In this post the CPI betas are quoted *per +0.1pp of core-CPI upside surprise*, and — critically — the asset *units differ*: equities, FX, gold, and Bitcoin betas are in percent, while yield betas are in basis points. A beta table without its units is a trap.

## The centerpiece: the CPI surprise beta across assets

Here is the headline result of the post — the beta of seven assets to a single, standardized unit of core-CPI upside surprise, estimated over the 2022-23 inflation-fear regime. This is the inflation report's "reaction fingerprint."

![Bar chart of asset betas to a 0.1pp core CPI upside surprise, percent moves and basis-point yield moves](/imgs/blogs/the-surprise-not-the-level-betas-to-data-surprises-2.png)

Per **+0.1pp** of core-CPI *upside* (hotter) surprise, in the 2022-23 regime:

- **S&P 500: −0.7%.** Higher-than-expected inflation means a more hawkish Fed, higher discount rates, lower equity valuations.
- **Nasdaq 100: −1.0%.** Larger than the S&P because tech is *long-duration* — its value sits further in the future, so it is more sensitive to the discount rate the Fed controls.
- **US 10-year yield: +7bp.** Hotter inflation pushes the whole rate path up.
- **US 2-year yield: +9bp.** The front end repriced *more* than the long end, because the 2-year is almost a pure bet on the next two years of Fed policy, which is what a hot CPI directly changes.
- **DXY (dollar): +0.35%.** Higher US rates pull capital toward dollar assets, strengthening the dollar.
- **Gold: −0.8%.** Gold pays no yield, so when *real* yields rise (which they do on a hawkish surprise), the opportunity cost of holding gold rises and gold falls. Gold is not an inflation hedge here — it is a *real-yield* trade. (More on that misconception below; the mechanism is in [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything).)
- **Bitcoin: −1.6%.** The highest-magnitude beta of the lot, because in this regime Bitcoin traded as a high-beta risk asset, amplifying whatever the Nasdaq did.

Notice the *structure*. The betas are not random; they line up with duration and risk. The most rate-sensitive, longest-duration, highest-beta assets (Nasdaq, Bitcoin) have the largest negative reactions; the dollar and front-end yields move the "right" way for a hawkish shock; gold tracks real yields. Read this way, the beta table is not seven separate facts — it is one coherent transmission map, with each asset's number flowing from its duration and risk character. (The full map of which driver hits which asset is in [how policy moves every asset: the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map).)

#### Worked example: implied move from a surprise and a beta

This is the payoff calculation — the whole reason the beta is worth measuring. Suppose tomorrow's core-CPI print comes in at +0.5% month-over-month against a consensus of +0.3%. The surprise is:

```
surprise = 0.5% - 0.3% = +0.2pp  (an upside, hot surprise)
```

That is *twice* the +0.1pp unit our betas are quoted in. Multiply each beta by 2:

```
S&P 500:   -0.7% per 0.1pp  x 2  =  -1.4%
Nasdaq 100: -1.0% per 0.1pp x 2  =  -2.0%
US 10Y:    +7bp  per 0.1pp  x 2  =  +14bp
US 2Y:     +9bp  per 0.1pp  x 2  =  +18bp
Gold:      -0.8% per 0.1pp  x 2  =  -1.6%
Bitcoin:   -1.6% per 0.1pp  x 2  =  -3.2%
```

So a +0.2pp core-CPI surprise *implies* roughly a −1.4% S&P move, a −2.0% Nasdaq move, a +14bp move in the 10-year, and a −3.2% Bitcoin move — all in the same window, all from one number. The intuition: once you have the beta and the surprise, the cross-asset reaction is not a mystery to be watched; it is an arithmetic to be computed *before the print*, which is precisely what desks do when they pre-trade a scenario.

This is also why the beta is so much more useful than a generic correlation coefficient. A correlation of, say, −0.6 between CPI surprises and S&P returns tells you the *direction and tightness* of the relationship but not the *magnitude* of the move. The beta gives you the magnitude in the asset's own units — exactly what you need to size a position or hedge a book. Correlation is unitless and bounded between −1 and +1; beta has the asset's units and is unbounded. For *measuring the relationship* you want correlation; for *acting on it* you want beta. This post is about the second.

### Why the betas line up: the duration ladder

The beta table is not a list of unrelated facts; it is a *ladder ordered by interest-rate sensitivity*. Walk down it and the logic is mechanical. A hot CPI surprise does one primary thing: it raises the expected path of interest rates. Everything else flows from how exposed each asset is to that rate path.

- **The 2-year yield** is almost a pure expectation of near-term Fed policy, the exact thing a CPI surprise revises, so it moves most in proportion to its job (+9bp).
- **The 10-year yield** prices the rate path plus long-run inflation and growth and a term premium, so a single CPI surprise is a smaller share of it (+7bp).
- **Long-duration equities (Nasdaq)** are valued by discounting cash flows that sit far in the future; raising the discount rate hits far-future cash flows hardest, so the Nasdaq's −1.0% beta exceeds the broad S&P's −0.7%.
- **Gold** has effectively infinite duration (it pays no cash flow ever) but its sensitivity runs through *real* yields, and a hot CPI lifts both nominal yields and inflation expectations, which partly cancel — leaving a moderate −0.8%.
- **Bitcoin**, in the 2022-23 regime, behaved as a leveraged proxy for risk appetite, so it amplified the Nasdaq's move to −1.6%.

Read this way, you don't have to *memorize* seven betas. You can *reconstruct* their relative sizes from one principle: the more an asset's value depends on the future (its duration) and the more leveraged it is to risk appetite, the larger its negative beta to a hawkish surprise. The table is a consequence of the duration ladder, not a coincidence. This is also why, when you encounter a new asset, you can guess its surprise-beta before estimating it: ask how long its duration is and how risk-on it is, and you have the sign and rough magnitude.

#### Worked example: estimating an unlisted asset's beta from the ladder

Suppose you want the CPI-surprise beta of a long-duration growth ETF that is not in the table, but you know it is even more rate-sensitive than the Nasdaq — say its average cash-flow horizon is roughly 1.4 times the Nasdaq's. Using the Nasdaq's −1.0% per +0.1pp as an anchor and scaling by the duration ratio:

```
estimated beta = -1.0% per 0.1pp  x  1.4  =  -1.4% per 0.1pp
```

So you would *expect* this ETF to fall about 1.4% on a +0.1pp core-CPI upside surprise in the inflation-fear regime — before running a single regression. You would then estimate the beta directly to confirm, but the duration ladder gives you a prior to check the estimate against. The intuition: betas are not arbitrary; they scale with duration, so a known beta plus a duration ratio gives you a usable estimate for an asset you have not measured yet.

## The sign flips: the regime is part of the beta

Here is the part that separates a sophisticated read of macro correlation from a naive one, and the reason this whole series insists that "correlation is a regime, not a constant." **The beta's sign depends on the regime.** The same surprise can be good for stocks in one regime and bad in another. If you estimate a beta on the wrong regime's data, you will get the sign exactly backwards and trade into a loss.

The cleanest example is the jobs report. Strong payrolls — a big *positive* surprise — can be either bullish or bearish for stocks, depending entirely on what the market is currently worried about.

![Grouped bar chart showing NFP surprise betas with the S&P sign flipping between the good-news-is-bad regime and a normal expansion](/imgs/blogs/the-surprise-not-the-level-betas-to-data-surprises-3.png)

Per **+100k** of NFP *upside* (stronger jobs) surprise:

- **In the 2022-23 "good-news-is-bad" regime:** S&P **−0.5%**. A strong jobs number meant the labor market was too hot, the Fed would have to hike more, rates would rise, and stocks would fall. Good economic news was *bad* market news, because the binding constraint was inflation, not growth.
- **In a normal expansion:** S&P **+0.35%**. A strong jobs number meant the economy was healthy, earnings would grow, and stocks would rise. Good economic news was *good* market news, because the binding constraint was growth, not inflation.

The S&P beta to the *identical* surprise — a strong jobs report — has the *opposite sign* in the two regimes. The 2-year yield rises in both (more jobs → higher rates either way), and the dollar firms in both, but the equity reaction flips entirely. This is the **reaction function**: the rule the market is currently using to translate data into a Fed path and then into prices. When inflation is the dominant fear, the market reads all data through the lens of "does this make the Fed more hawkish?" and strong data is hawkish, hence bearish for stocks. When growth is the dominant question, the market reads data through "does this confirm the expansion?" and strong data is bullish. The reaction function *is* the regime, and it is baked into the sign of every beta. (The trading playbook for these reaction-function shifts is in [the reaction function](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently); here the point is that the regime is a *measurement* parameter — your beta is only valid for the regime you estimated it in.)

#### Worked example: the same surprise, two implied moves

Suppose payrolls beat by +150k (a +150k upside surprise), which is 1.5 units of our +100k beta. The implied S&P move depends entirely on the regime you are in:

```
Good-news-is-bad regime:  -0.5% per 100k  x 1.5  =  -0.75%   (stocks fall)
Normal expansion:         +0.35% per 100k x 1.5  =  +0.53%   (stocks rise)
```

Same +150k surprise; one regime implies a −0.75% S&P move, the other a +0.53% move — opposite signs, more than a full percentage point apart. The intuition: before you multiply a surprise by a beta, you must first *identify the regime*, because the regime sets the sign. A beta quoted without its regime is not just imprecise — it can point you the wrong way entirely.

### How to tell which regime you are in

If the sign depends on the regime, the practical question is: how do I know which regime I'm in *before* the print, so I apply the right beta? Three reliable tells:

1. **Watch what the market rewards.** In a good-news-is-bad regime, you will see strong data sell stocks and weak data rally them on the *previous* few prints. The market literally shows you its reaction function in recent reactions. If the last three hot CPIs sold off stocks, the regime is inflation-fear.
2. **Watch the Fed's stated focus.** When the Fed says it is "data-dependent" on inflation and the labor market is "too tight," the binding constraint is inflation, and the regime is good-news-is-bad. When the Fed is cutting and watching growth, strong data becomes good news again.
3. **Watch the level of inflation itself.** Empirically the flip happens around the 3-4% inflation band. Below ~3% inflation, growth dominates and good news is good. Above ~4%, inflation dominates and good news is bad. (This same inflation-band logic drives the stock-bond correlation flip — see [stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).)

The regime is observable. You are not guessing; you are reading the same tells the desks read, and updating your beta's sign accordingly.

## The full surprise-beta map

Stepping back, every macro correlation in this series can be arranged as a grid: rows are the *drivers* (each one's surprise or change), columns are the *assets*, and each cell is the sign and strength of that asset's beta to that driver. This is the series in one picture.

![Heatmap of beta sign and strength for seven macro drivers against seven assets](/imgs/blogs/the-surprise-not-the-level-betas-to-data-surprises-6.png)

Read the heatmap by row to see how one driver hits everything, or by column to see what moves one asset. A few patterns leap out:

- **The bond column is almost uniformly red for "rates up" drivers.** Anything that pushes yields up (a 10-year rise, a real-yield rise, a hot CPI) is bad for bond prices — that is nearly mechanical, which is why the bond cells are the strongest (most negative) in the grid.
- **The dollar column is the mirror image.** A stronger dollar is, by definition, +1.0 with itself, and higher US yields strengthen the dollar (+0.40 to +0.45), so the dollar wins when rates rise.
- **Gold's row tells the real-yield story.** Gold is strongly negative to a real-yield rise (−0.80) but only weakly negative to a hot CPI surprise (−0.10) — because a hot CPI's effect on gold runs *through* real yields, and the two forces (higher inflation, higher nominal yields) partly cancel in real terms. Gold cares about real yields, not headline inflation.
- **The ISM/PMI row flips green for risk assets** (S&P +0.55, EM +0.55, Bitcoin +0.40) — a stronger business survey is good for growth-sensitive assets, the opposite sign from the inflation drivers. This is the growth-versus-inflation distinction again, in matrix form.

Every cell in this grid is a beta you could estimate with the event-study regression above. The grid is the destination; the regression is how you get each number. And every number is *regime-conditional* — this grid encodes the documented signs and relative strengths, but the inflation-driver rows in particular would shift sign for risk assets if you re-estimated them in a growth-dominated regime.

## Common misconceptions

**"The level of the indicator is what drives the asset."** No. The level is overwhelmingly priced in advance; only the surprise — the deviation from consensus — moves the price in the reaction window. The August 2022 CPI fell from 8.5% to 8.3% (the level improved) and stocks fell 4.3% (because the surprise was hot). If you regress asset returns on the *level* of an indicator, you will find a weak, unstable relationship and wrongly conclude data doesn't matter. Regress on the *surprise* and the relationship is sharp.

**"A bigger number is a bigger surprise."** No. A surprise is *actual minus consensus*, and its size is best measured in standard deviations, not raw units. A +100k payrolls beat is a routine ~1.4-sigma event; a +0.1pp core-CPI beat is also ~1.4 sigma but in completely different raw units. Standardize before comparing, or you will think the jobs miss was huge and the CPI miss was tiny when they were equally informative.

**"The beta is a fixed property of the asset."** No. The beta's *sign* flips with the regime. A strong-jobs surprise was −0.5% for the S&P in 2022-23 and +0.35% in a normal expansion. The beta is a regime-conditional number; estimating it on the wrong regime's data gives you the wrong sign. Always tag a beta with the regime it was estimated in.

**"Gold hedges inflation, so a hot CPI is good for gold."** No — gold *fell* 0.8% per +0.1pp core-CPI upside surprise in the 2022-23 regime. Gold pays no yield, so it tracks *real* (inflation-adjusted) yields, not headline inflation. A hot CPI raised nominal yields faster than inflation expectations in that regime, so real yields rose and gold fell. Gold is a real-yield trade wearing an inflation-hedge costume. (The full case is in [real vs nominal: real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).)

**"If the print matches consensus, nothing happens."** Usually true, but not always — because the market may have been trading the *whisper* number, not the published consensus. A print can match the official 8.1% consensus exactly and still sell off if the whisper had drifted to 7.9% on a soft PPI the day before. The surprise that matters is relative to the *market's* real-time expectation, which the published consensus only approximates.

## How it shows up in real markets

**September 2022 — the level fell, the surprise was hot, stocks crashed.** As described in the opening, headline CPI *declined* from 8.5% to 8.3% year-over-year, but the print beat the 8.1% consensus and the core measure was hot. The S&P fell 4.3% in a day. The textbook demonstration that the surprise, not the level, drives the move: an improving level produced one of the worst days of the cycle because the surprise was positive. Plug the surprise into the CPI beta and the cross-asset reaction — stocks down hard, yields up, dollar up, gold down — falls right out of the table.

**November 2023 — the cool surprise that turned the cycle.** Core CPI printed at 0.28% month-over-month versus 0.30% expected — a small *negative* surprise of just −0.02pp. The level of inflation was still elevated (core near 4% year-over-year), but the print missed (cooler than) consensus, and the market read it as the first solid evidence that disinflation was real. The S&P rallied, yields fell, and the move kicked off the late-2023 "Fed pivot" rally. A tiny negative surprise against a high level produced a large positive market reaction — exactly the opposite of what a level-based intuition predicts.

**The good-news-is-bad payrolls days of 2022-23.** Repeatedly in that period, a hot jobs report sold off stocks on release: strong payrolls meant the Fed had to keep hiking, so yields jumped and equities fell. The same data — robust hiring — that would have been celebrated in a normal expansion was punished, because the reaction function had inverted. A trader who had estimated the NFP beta on pre-2022 data would have bought the strong print and been run over. The sign of the beta had flipped with the regime.

**The Citi Surprise Index rolling over while the economy stayed strong.** Through several 2023-24 stretches, the US economy kept growing solidly, yet the CESI rolled from strongly positive back toward zero — not because the economy weakened, but because economists raised their forecasts to match the run of beats, so each new print had a higher bar to clear. Risk assets that had rallied with the rising CESI stalled as it flattened. A vivid reminder that markets trade the *gap* to expectations, and the gap shrinks as expectations catch up even when the level is fine.

**The decoupling that broke a "reliable" beta.** Through 2022-24, gold's well-established negative beta to real yields — about −0.8 correlation for the better part of two decades — went haywire. Real yields rose sharply (the 10-year TIPS yield climbed from roughly −1% in 2021 to around +2% by 2024), which by the historical beta should have crushed gold. Instead gold *rose* to record highs, because central-bank buying (a new, large, price-insensitive source of demand) overwhelmed the real-yield driver. An analyst who applied the long-run gold beta mechanically through this period would have been short gold into a record rally. The lesson is not that the beta was "wrong" — it was right for two decades — but that a beta is a *conditional* relationship that can break when a new driver enters. This is why step five of the playbook below insists on carrying the residual and re-checking the regime: even a beta with twenty years of history is a measurement, not a law.

## How to read it and use it

Here is the playbook for turning this measurement subtlety into something you actually do.

**1. Always work with the surprise, never the level.** Before any data release, write down the consensus. After the print, compute actual − consensus. That number — not the headline level — is your input. If you catch yourself reasoning "inflation is high, so stocks should fall," stop: the high level is already priced. Ask instead, "is the print higher or lower than the ~0.3pp the market expected?"

**2. Standardize when comparing across indicators.** Divide each surprise by its historical surprise standard deviation to get a z-score. This lets you rank shocks ("today's CPI was a 2-sigma upside, last week's NFP was only 0.5-sigma") and compare betas across indicators on a per-sigma basis.

**3. Estimate the beta as a regression slope, in a narrow window.** Collect past releases, pair each surprise with the same-window asset return, fit the line, read the slope. Use the tightest clean window you can (intraday is best) to maximize signal-to-noise. The slope is your beta; the intercept should be near zero; the residual scatter tells you how much *else* moves the asset. The Python recipe is the toolkit post linked above.

**4. Tag every beta with its regime — and check the regime before you use it.** The sign is regime-conditional. Identify the regime from recent reactions, the Fed's stated focus, and the level of inflation (the flip lives around 3-4%). Apply the regime-appropriate beta. A beta from the wrong regime can be exactly backwards.

**5. Pre-compute the scenario.** For any plausible surprise, multiply by the beta to get the implied move across assets *before* the print. That converts the release from a thing you watch nervously into an arithmetic you've already done — the difference between reacting and being positioned.

**What invalidates the signal.** The beta breaks when (a) the regime shifts and you haven't re-estimated (the sign flips); (b) the consensus you used was stale and the market was trading a different whisper number (your surprise is mismeasured); (c) a second, larger event lands in the same window (your event study is contaminated — the residual swamps the signal); or (d) the relationship genuinely decouples, as gold did from real yields in 2022-24 when central-bank buying overwhelmed the usual driver. Always carry the residual scatter from your regression as a humility check: a low r means the beta is real but only one of many forces; do not bet the book on a single noisy print.

### A desk workflow: from calendar to position

To make this concrete, here is the loop a macro desk actually runs around a major release, with the beta at its center:

1. **Days before:** pull the consensus and the recent forecast dispersion, and note the whisper if one is circulating. Identify the regime from the last few prints' reactions and the Fed's stated focus. Select the *regime-appropriate* beta vector (the right column of the surprise-beta map) for the assets you care about.
2. **The morning of:** sketch a scenario grid — for a hot surprise of +0.2pp, a cool surprise of −0.1pp, and in-line, multiply each by the beta to get the implied move per asset. You now have a pre-computed reaction table for every plausible outcome, so you are never reacting from scratch when the number hits.
3. **At the print:** compute actual − consensus, standardize it against the historical surprise standard deviation to gauge how big a shock it is, and read off the corresponding row of your scenario grid. If the realized move is far from the implied move, *that gap is the residual* — and it is a signal that something else is in the window (a confound, a positioning squeeze, a second headline), which you investigate before fading or pressing the move.
4. **After:** log the surprise and the realized move as a new dot for your regression, so the beta estimate updates with each release and you can watch for the slope drifting — an early warning that the regime is shifting.

That loop is the entire post in operational form: define the surprise, standardize it, multiply by a regime-conditional beta, and treat the residual as information. None of it works if you anchor on the level instead of the surprise — which is exactly the trap that catches most people.

#### Worked example: sizing a hedge with the beta

Suppose you hold a \$10 million long S&P position into a CPI print, and you judge there is a real chance of a +0.2pp hot surprise. Using the S&P beta of −1.4% for a +0.2pp surprise (computed earlier), the implied loss is:

```
implied loss = $10,000,000  x  -1.4%  =  -$140,000
```

To neutralize that exposure you would short enough S&P futures to offset a 1.4% move on \$10 million — roughly \$10 million notional of futures, since the futures move one-for-one with the index. If instead you only wanted to hedge *half* the risk, you would short \$5 million notional, capping the implied CPI-day loss near \$70,000. The intuition: because the beta gives the move in dollars-per-surprise, it converts directly into a hedge ratio — the beta is not just a description of the correlation, it is the recipe for the offsetting trade.

The deepest lesson is the one this whole series is built on: a correlation is not a constant you look up once. It is a *conditional measurement* — conditional on using the surprise rather than the level, conditional on the units, and above all conditional on the regime. Get those three right and "inflation is bad for stocks" becomes "−0.7% per +0.1pp core-CPI upside surprise, in the inflation-fear regime" — a number you can compute, size, and trade. That is the bridge from a correlation to a tradeable beta.

## Further reading and cross-links

- [What correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) — the statistics behind the coefficient and the slope used here.
- [Measuring beta to data surprises: an event study in Python](/blog/trading/macro-correlations/measuring-beta-to-data-surprises-an-event-study-in-python) — the runnable toolkit that estimates these betas.
- [CPI and asset prices: the master inflation correlation](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation) — the inflation correlation this post measures the surprise-beta for.
- [NFP and asset prices: the king of data correlation](/blog/trading/macro-correlations/nfp-and-asset-prices-the-king-of-data-correlation) — the jobs correlation whose beta flips sign by regime.
- [Why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) — the mechanism behind why only the surprise moves price.
- [The reaction function](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) — the trading playbook for regime-driven sign flips.
- [Real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) — why gold's surprise-beta runs through real yields, not inflation.
- [How policy moves every asset: the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) — the full map the surprise-beta grid summarizes.
