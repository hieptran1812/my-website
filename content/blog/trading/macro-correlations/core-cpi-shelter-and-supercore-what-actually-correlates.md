---
title: "Core, Shelter, and Supercore: Which CPI Components Actually Correlate"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Headline CPI is a blend; the market's reaction correlates with core and supercore, while the most-weighted component, shelter, lags real rents by about a year and is the least timely signal in the report."
tags: ["macro", "correlation", "cpi", "core-cpi", "supercore", "shelter", "inflation", "lead-lag", "regime", "trading"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Headline CPI is a weighted blend, and the market's reaction does not correlate with the blend; it correlates with the *components that predict the Fed* — **core** (ex food and energy) and, in 2023-24, **supercore** (core services ex-housing) — while the single most-weighted component, **shelter**, lags real-time market rents by about a year and is therefore the least timely signal in the whole report.
>
> - The level of headline inflation is mostly noise: food and energy swing it around month to month, and they are not what the Fed is trying to slow. The market strips them out and trades the core.
> - The asset reaction keys on the **core surprise**, not the headline surprise. For a +0.1pp core-CPI upside surprise in the 2022-23 regime: S&P −0.7%, Nasdaq −1.0%, 10Y +7bp, 2Y +9bp, DXY +0.35%, gold −0.8%, Bitcoin −1.6%.
> - **Shelter is about 34% of the basket but lags new-lease market rents by roughly 12 months.** The most-weighted slice of the report is the one that tells you the least about what is happening *now*.
> - The one fact to remember: in 2023 headline fell from 9.1% to 3.0% almost entirely on energy, yet core stayed near 4.8%, the Fed kept hiking, and the tape traded the sticky core — not the falling headline.

## The print that "fell" but felt like a hike

On 13 July 2023, the US Bureau of Labor Statistics released the June Consumer Price Index, and on its face the number was a triumph. Headline inflation printed at 3.0% year-over-year, down from a 9.1% peak almost exactly twelve months earlier. Cable-news chyrons declared inflation beaten. A casual reader, looking only at the headline, would have concluded the inflation war was essentially over and that the Federal Reserve could stop raising interest rates — perhaps even start cutting.

The bond market did not agree. The two-year Treasury yield — the part of the curve that tracks where the Fed funds rate is headed — was sitting near 4.9%, having barely flinched from its highs. Within two weeks, on 26 July, the Fed raised rates again, to a 5.25–5.50% range, the highest in twenty-two years. The headline had fallen by six full percentage points, and yet the central bank kept tightening and the market kept pricing more tightening. Why?

Because the headline had fallen for a reason that told you almost nothing about the underlying trend. Gasoline prices had collapsed year-over-year — energy was a huge *subtraction* from the headline — but the part the Fed actually cares about, **core** inflation (everything except the volatile food and energy items), was still running near 4.8%, more than double the 2% target. And inside core, the piece tied to wages and services — what traders had started calling **supercore** — was hotter still. The headline was a flattering average that hid a stubborn core, and the market, which is not fooled by averages, traded the core. This post is about exactly that: which slices of the most-watched inflation report in the world actually move asset prices, which ones mislead, and why the biggest component of all is the one you should trust least in real time.

![CPI basket decomposed into headline, core, supercore, and shelter with weights and timeliness labels](/imgs/blogs/core-cpi-shelter-and-supercore-what-actually-correlates-1.png)

This series treats every macro-to-asset relationship as a measurable object with a sign, a strength, a lead/lag, and a regime in which it holds (the founding idea is laid out in [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant)). The previous post in this track established that the master inflation correlation runs through CPI and that the market trades the *surprise*, not the level — see [CPI and asset prices, the master inflation correlation](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation) and the dedicated treatment of surprise in [correlate the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises). This post goes one level deeper: it opens the report up, looks at the components inside it, and shows that the correlation is not with "CPI" at all — it is with the specific sub-indices that tell the market what the Fed will do next.

## Foundations: how CPI is built, from zero

Before we can talk about which component correlates with what, we have to know what the components *are*. Most people treat "CPI" as a single number that comes down from the sky each month. It is not. It is a carefully engineered weighted average of thousands of prices, and understanding its construction is the whole key to understanding why the market ignores most of it.

### What the Consumer Price Index actually measures

The Consumer Price Index measures the average change over time in the prices a typical urban household pays for a fixed *basket* of goods and services. The idea is simple: pick a representative shopping cart — rent, groceries, gasoline, a haircut, a doctor's visit, a streaming subscription, a used car — and track how much that exact cart costs month after month. If the cart cost \$100 last year and \$104 this year, prices rose 4%; that is the year-over-year inflation rate.

The everyday analogy is a grocery receipt. Imagine you buy the same fifty items every week. Some weeks the bananas are cheaper and the coffee is dearer; the *total* at the bottom of the receipt blends all those moves into one number. CPI is that bottom-line total for the whole economy's typical receipt. The crucial subtlety — the one that drives everything in this post — is that not every item is weighted equally. You spend far more on rent than on bananas, so rent moves the total far more than bananas do. The weights are what make the headline a *blend*, and a blend can hide as much as it reveals.

There is one more layer to the construction that explains *why* the index has to be a weighted average rather than a simple list. The BLS does not just track headline categories; it collects roughly 80,000 individual price quotes a month, across hundreds of item categories in dozens of geographic areas, and aggregates them up a tree: individual quotes roll into item-area cells, cells roll into item categories, item categories roll into the major groups (food, energy, shelter, and so on), and the major groups roll into the all-items headline. At each level of the tree, the pieces are combined using expenditure weights. The headline you see is the top of that pyramid — a weighted average of weighted averages. When you "strip out energy" to get core, you are simply removing two branches of the tree (food and energy) and re-normalizing the rest. The components are not arbitrary slices; they are the natural branches of the aggregation tree, which is why core, supercore, and shelter are the cuts everyone uses.

### A note on the formula and substitution

It is worth knowing that CPI is, at the lower levels, a *Laspeyres-type* index: it asks what a roughly fixed basket costs over time. A pure fixed-basket index has a known upward bias, because consumers substitute away from things that get expensive (if beef soars, people buy more chicken), so a basket frozen in the past overstates the cost of living. The BLS corrects for this in two ways: it updates the weights regularly (annually in recent years, using the prior year's spending), and it uses a geometric-mean formula within most item categories that captures within-category substitution. None of this changes the components — core is still headline minus food and energy — but it does mean the *level* of CPI carries small methodological choices, which is one more reason the market focuses on the *change* and the *surprise* rather than the absolute level. (For the broader point that you should always correlate the surprise, not the level, see [correlate the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises).)

### Weights: why the basket is not democratic

The Bureau of Labor Statistics assigns each category a **relative importance** weight based on how much of the average household's budget it consumes, measured from the Consumer Expenditure Survey. These weights are the reason headline CPI behaves the way it does. As a rough picture of a recent basket:

- **Shelter** (rent of your home plus the imputed rent homeowners "pay" themselves) is by far the largest single category, around **34%** of the basket.
- **Food** is around **13%**.
- **Energy** (gasoline, electricity, natural gas) is around **7%**.
- **Core goods** (cars, appliances, apparel, furniture) is around **20%**.
- **Core services excluding shelter** — the rest of services: medical care, insurance, recreation, transportation services, dining out — is around **25%**. This is the category that will become "supercore."

A 1% move in shelter therefore moves the headline almost five times as much as a 1% move in energy, simply because shelter is five times the weight. This single fact — that the basket is not democratic — is why you cannot read the headline as if every component were telling you something equally true. A big move in a small-weight component (energy) can dominate the *change* in the headline for a few months even while the broad trend is set by the large, slow-moving components.

### The first cut: headline minus food and energy is "core"

The first thing every serious analyst does to a CPI report is throw away food and energy. The resulting index is called **core CPI**. This is not because food and energy do not matter — you obviously eat and drive — but because their prices are dominated by global commodity swings, weather, and geopolitics that have nothing to do with the underlying domestic inflation trend and everything to do with an OPEC decision or a frost in Brazil. They are *volatile* and *mean-reverting*: a gasoline spike that adds a point to headline this year often subtracts a point next year. For the purpose of reading the *trend* — and for the Fed, which sets policy on the trend — the noise is worse than useless. Stripping it out leaves a cleaner signal.

> [!note]
> "Core" is an old, well-established concept; it dates to the 1970s, when oil shocks made headline inflation gyrate wildly and economists wanted a measure of the persistent underlying rate. It is not a trick to make inflation "look lower." Over long periods, core and headline converge — energy can't fall forever. But over the one-to-two-year horizon that matters for trading and for policy, core is the trend and headline is the trend plus noise.

Core is the most famous "underlying inflation" measure, but it is not the only one, and the alternatives reinforce why the components matter. Stripping out food and energy is a *blunt* way to remove noise — it throws away two whole categories regardless of whether they were actually volatile that month. Statisticians prefer surgical alternatives. The Cleveland Fed publishes **median CPI** (the inflation rate of the single item right in the middle of the distribution, immune to any one category's spike) and **trimmed-mean CPI** (which discards the most extreme price moves in *both* tails each month, whatever category they happen to be in). These robust measures often tell a cleaner story than core: in 2021-22 they confirmed the inflation surge was broad, not just a couple of pandemic categories, which was an early tell that "transitory" was wrong. The deeper lesson is that "underlying inflation" is always a *recipe for which components to keep* — core keeps everything but food and energy, median keeps one item, trimmed-mean keeps the un-extreme middle — and the market watches several recipes at once to triangulate the true trend. Whenever the recipes disagree, the disagreement itself is information.

### The second cut: core services ex-housing is "supercore"

In 2022, as inflation stayed stubbornly high, Federal Reserve Chair Jerome Powell drew analysts' attention to a finer cut still. He split core into three pieces:

1. **Core goods** — physical things ex-energy: cars, furniture, apparel. These had spiked during the pandemic supply-chain mess and were already deflating by 2023 as supply normalized.
2. **Shelter (housing services)** — rent and owners' equivalent rent. Big, slow, and — as we will see — badly lagged.
3. **Core services excluding shelter** — everything else in services: healthcare, insurance, haircuts, airfares, restaurant meals.

That last category, **core services ex-housing**, became known as **supercore**. Powell singled it out because it is the part of inflation most tightly linked to the labor market: services ex-housing are labor-intensive, so their prices track wages, and wages track how tight the jobs market is. If you want to know whether inflation is *embedding* into the wage-price structure — the thing a central bank fears most — supercore is where you look. From late 2022 through 2024, supercore became the single most-watched number in every CPI report, and the market's reaction function re-centered on it.

To make the wage link concrete: think about what you actually pay for in a services-ex-housing item. A haircut, a restaurant meal, a hospital stay, an insurance claim adjuster, a dog-walker, a plumber — the dominant cost in each is *labor*. A barbershop has almost no raw materials; nearly its entire cost is the barber's time. So when wages rise, the price of a haircut has to rise to cover them, with very little a business can do to absorb it through cheaper inputs. Contrast that with a core *good* like a television: its price is mostly components, shipping, and factory automation, with labor a smaller share, so a TV's price can fall even as wages rise (and it did through 2023). That structural difference is why supercore is the inflation category that *embeds* — it is the one whose costs can't be deflated away by a better supply chain. An economy with hot supercore has inflation baked into its labor costs, which is far harder and slower to reverse than a one-off energy or goods shock. This is the deep reason the Fed treats hot supercore as the most dangerous reading in the report, and why the market treats a hot-supercore surprise as the most hawkish.

There is a second-order subtlety worth flagging. Supercore is not perfectly clean either: a few of its components — notably health insurance in CPI and airfares — are measured with quirky methodologies that can add noise to the line month to month. Sophisticated desks therefore watch a *trimmed* or smoothed supercore (three- or six-month annualized run-rates) rather than reacting to one bumpy print. The principle holds — supercore is the timely, wage-driven, embedding signal — but in practice you smooth it to filter the measurement noise, just as you watch core month-over-month rather than a single category.

The figure above lays out the full decomposition: how the 100% headline basket strips down to core (about 80% of the basket), and how core then splits into the lagged shelter slice and the timely supercore slice. Keep that map in mind, because the rest of the post is about which of those boxes the market actually trades.

### Seasonal adjustment, in one paragraph

One more construction detail matters. The numbers the market trades are **seasonally adjusted**. Prices have predictable annual rhythms — airfares rise every summer, apparel discounts every January — and if you didn't remove that rhythm, you would mistake "it's July" for "inflation accelerated." The BLS estimates the typical seasonal pattern and divides it out, leaving the part of the monthly move that is *not* explained by the calendar. This is why traders watch the **month-over-month** core change (e.g., "+0.3% m/m"): it is the cleanest read on the current run-rate, with seasonality and base effects stripped away. The year-over-year number is more familiar to the public but is contaminated by what happened twelve months ago (the "base effect"), which is exactly the trap that made the 2023 headline look so benign.

### Base effects: why year-over-year lies in a turning point

The base-effect trap deserves a full paragraph because it is the mechanical reason the 2023 headline collapse was so misleading. A year-over-year inflation rate compares this month's price level to the level *twelve months ago* — the "base." If the base was unusually high (because something spiked a year ago), then even flat prices today will show a *falling* year-over-year rate, purely because the comparison point is rolling off. In mid-2022, energy prices spiked after the Russian invasion of Ukraine; gasoline hit record highs. Twelve months later, in mid-2023, that high base rolled out of the year-over-year window, so the energy contribution to year-over-year headline went sharply negative *even if gasoline prices were merely stable*. None of that "disinflation" reflected current price behavior — it was arithmetic from the prior year's spike. A trader who reads month-over-month sees through this instantly: the recent *run-rate* of core was still hot, even as the year-over-year headline fell. **The base effect is why you should never read a year-over-year headline at a turning point without checking the month-over-month run-rate of core underneath it.** Conflating "the year-over-year rate fell" with "inflation is cooling now" is the same error as conflating the headline with the core — both are averages that hide the current signal.

## The correlation: it is with the components, not the headline

Now we can state the central claim precisely. When a CPI report drops, the market does not compute "is headline higher or lower than expected" and trade that. It computes a vector of surprises — was core hotter than expected, was supercore hotter, was shelter cooler — and trades the ones that change its forecast of the Federal Reserve's path. The headline surprise correlates with asset moves only to the extent that it carries information about core; when the headline surprise and the core surprise point in opposite directions, the market follows core.

### Why the Fed (and therefore the market) watches core and supercore

The transmission is simple. The Fed has a 2% inflation target, and it has explicitly said it sets policy on the *underlying trend*, not on month-to-month energy noise. Its single preferred gauge is actually core PCE, a close cousin of core CPI (we'll come back to the PCE/CPI distinction). So the Fed's reaction function — the rule by which it decides to hike, hold, or cut — is driven by core and, within core, increasingly by supercore. The market's reaction function is, in turn, an attempt to forecast the Fed's. So:

```
core/supercore surprise
   -> revises the expected Fed path
      -> reprices the 2Y yield (a pure bet on the Fed)
         -> reprices the whole curve, the dollar, and risk assets
```

This is why the asset beta is a *core* beta. The market is not pricing inflation for its own sake; it is pricing the policy response, and the policy response keys on core. For the mechanism of how a rate-path revision then ripples across every asset, see [how policy moves every asset](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map); for why the front end of the curve is essentially a forecast of the Fed, see [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

![Bar chart of CPI component weights and mid-2023 year over year inflation rates](/imgs/blogs/core-cpi-shelter-and-supercore-what-actually-correlates-2.png)

The chart above is the 2023 story in one frame. Energy is a tiny sliver of the basket (about 7%) and its year-over-year rate had gone sharply *negative* (about −12.5%), which is what dragged the headline down. But shelter, at about 34% of the basket, was still running near 7.8%, and supercore was near 4.6%. The big-weight components were hot; the falling component was small. Multiply weight by rate and you see immediately why core barely moved even as the headline collapsed: the falling piece simply didn't have enough weight to pull down the sticky majority.

#### Worked example: re-weighting a headline print into core

Suppose a CPI month-over-month report comes in like this (illustrative, but the arithmetic is exactly how the decomposition works). Headline prints +0.2% m/m. Inside it, using rough weights:

```
energy:        weight 7%,   change -4.0% m/m  -> contribution = 0.07 * (-4.0) = -0.28pp
food:          weight 13%,  change +0.2% m/m  -> contribution = 0.13 * (0.2)  = +0.03pp
core:          weight 80%,  change +0.56% m/m -> contribution = 0.80 * (0.56) = +0.45pp
                                                   headline total ~= +0.20pp
```

The headline of +0.2% looks tame. But notice what produced it: energy *subtracted* 0.28pp, while core *added* 0.45pp. Strip out the volatile energy and you find a core run-rate of +0.56% m/m — which annualizes to nearly 7%. The headline whispered "calm"; the core screamed "hot." A trader who reads only the headline mis-positions; a trader who re-weights into core sees the truth. **The intuition: a benign headline can be hiding a hot core whenever a big-weight volatile component is moving the other way.**

### The asset beta is to the core surprise

So what does a hot core surprise actually do to prices? The cross-asset reaction, estimated from the 2022-23 inflation-fear regime, runs like this for a +0.1pp upside surprise in core CPI:

![Bar chart of asset reactions to a hot core CPI surprise, with percent moves and basis point yield moves](/imgs/blogs/core-cpi-shelter-and-supercore-what-actually-correlates-5.png)

Reading the figure: a hotter-than-expected core print is *hawkish* — it tells the market the Fed will keep rates higher for longer. So yields rise (2Y +9bp, more than 10Y +7bp, because the front end is the purest bet on the Fed), the dollar strengthens (+0.35%, higher US yields pull capital in), and risk assets fall: the S&P −0.7%, the long-duration Nasdaq −1.0% (its far-off earnings are discounted harder when rates rise), gold −0.8% (real yields up hurt the zero-yield metal), and Bitcoin −1.6% (the highest-beta risk asset of all). Every sign and rough magnitude is the same as for the headline surprise *because* the market reads the headline through its core. The key point of this post is that the X-variable in that regression is the **core** surprise; if you ran the same regression on the *headline* surprise, the relationship would be far noisier, because headline surprises are partly energy noise that the market discounts.

> [!note]
> Notice the relative sizes line up with a simple rule: the closer an instrument is to being a pure expectation of the Fed path, the bigger its beta to a core surprise. The 2Y *is* a Fed-path forecast, so it moves most. Equities and gold move through the discount-rate channel, one step removed. This is the same logic developed in [correlate the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — here we are simply pinning down *which* surprise (the core one) carries the signal.

#### Worked example: trading a split print

Imagine a release where the headline *beats* (comes in cool) but the core *misses* (comes in hot): headline +0.1% m/m versus +0.3% expected (a −0.2pp cool surprise), but core +0.4% m/m versus +0.3% expected (a +0.1pp hot surprise). Which way does the market trade?

```
headline surprise: -0.2pp (cool)  -> would suggest risk-on
core surprise:     +0.1pp (hot)   -> hawkish for the Fed
```

The market follows core. Using the core beta above, the +0.1pp core surprise implies roughly S&P −0.7%, 2Y +9bp, dollar +0.35%. On a \$50,000 S&P position, that −0.7% is about −\$350 erased in seconds; the brief headline-driven pop of perhaps +\$150 reverses straight into that \$350 loss as desks read the core line. The cool headline is dismissed as energy noise. In practice you'd often see the S&P pop for a few seconds on the headline beat as algorithms read the top line, then reverse hard within a minute as desks read the core line. **The intuition: when headline and core disagree, the durable move is in the direction of core, because that is what the Fed reacts to.** (For the intraday choreography of that reversal, see the event-trading companion [CPI, the report that moves the world](/blog/trading/event-trading/cpi-the-report-that-moves-the-world).)

## The lead/lag twist: the biggest component is the least timely

Here is the part that surprises even experienced people. The single most-weighted component — shelter, at about a third of the entire basket — is also the *most delayed*. It is a backward-looking measurement masquerading as a current one. To see why, you have to understand how the BLS measures housing costs.

### How shelter is actually measured

You might think CPI measures rent by checking what apartments are leasing for *today*. It does not, and it can't. The shelter index is built from two pieces: **rent of primary residence** (what tenants actually pay) and **owners' equivalent rent**, or OER (an estimate of what homeowners *would* pay to rent their own home, which is the larger piece since most Americans own). The BLS surveys the *existing stock* of rental units — the rents people are actually paying right now under leases signed at various points in the past — not just brand-new leases.

This matters enormously. Most leases are annual. If market rents jump 15% this spring, the typical tenant doesn't feel it until their lease renews, which on average is months away, and the average across all tenants only fully reflects the new market level after roughly a year of renewals roll through. The CPI shelter index, by construction, is a *slow-moving average of past market rents*. Real-time indices of *new-lease* rents — Zillow's Observed Rent Index, Apartment List, CoreLogic's single-family rent index — turn first; the CPI shelter index turns about a year later.

The BLS itself has acknowledged this lag and now publishes a research series, the **New Tenant Rent index**, built only from leases signed by *new* tenants — essentially the BLS's own version of a leading rent measure. Studies from the Cleveland and San Francisco Federal Reserve banks have used exactly this leading-versus-lagging distinction to *forecast* the official shelter index a year ahead, and their models did a creditable job calling the 2024 shelter disinflation well before it showed up in the headline. The point is not that shelter is mismeasured — it is measured correctly *as a stock-of-leases concept*. The point is that the concept it measures (what the average tenant currently pays) is inherently backward-looking relative to the concept the market cares about (where rents are headed). The mismatch between "what the average renter pays now" and "what the marginal renter pays now" *is* the lag.

There is also an asymmetry to the lag that traders exploit. Because the official index is a moving average, it not only turns late, it turns *smoothly* — it doesn't spike or crash, it grinds. That makes its path unusually forecastable: once you've seen the leading indicator roll over, you can be confident the official index will grind down for many months, not whipsaw. A smooth, lagged, predictable component is the easiest kind of series to nowcast, which is exactly why the shelter trade was so well-telegraphed in 2023-24.

![Timeline showing market rents turning before the CPI shelter index follows about a year later](/imgs/blogs/core-cpi-shelter-and-supercore-what-actually-correlates-3.png)

The timeline above traces the 2021-24 episode. New-lease market rents surged through 2021 and peaked near 15% year-over-year in early 2022. CPI shelter, lagging, kept *climbing* through 2022 and only peaked near 8% in early 2023 — about twelve months after the leading indicator turned. By then, real-time rents had already cooled to low single digits, but the official shelter index was still printing hot, holding up the entire core number and giving the Fed (and the market) a reason to stay hawkish. Then, through 2024, shelter finally "caught down," rolling toward 5% as the cool real-time rents of 2022-23 finally rolled through the lease stock.

### The lead/lag, quantified

This is a textbook lead/lag relationship — exactly the kind of cross-correlation the series covers in [leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators). The cross-correlation between new-lease market rents and CPI shelter peaks at a lag of roughly twelve months: market rents *lead* CPI shelter by about a year.

![Bar chart of lead times for market rents to CPI shelter, PPI to core goods CPI, and CPI to PCE](/imgs/blogs/core-cpi-shelter-and-supercore-what-actually-correlates-7.png)

The figure places the shelter lag next to two other pieces of the inflation report's internal plumbing for scale. PPI (producer prices) leads core goods CPI by about a month — upstream factory and wholesale prices feed into consumer goods prices with a short delay. CPI and PCE are essentially coincident (CPI a hair earlier). But shelter's twelve-month lag dwarfs them all. The implication is stark: **the most-weighted component of the most-watched inflation report is the least informative about what inflation is doing right now.** A third of the basket is, in real time, old news.

### The third core component: core goods and the supply-chain echo

The shelter lag and the supercore-wage link are the two stories that defined 2022-24, but the third slice of core — **core goods** — has its own correlation worth understanding, because it ran the entire inflation movie in reverse. Core goods are physical things ex-energy: cars (especially used cars), furniture, appliances, apparel, electronics. During the pandemic, demand shifted violently toward goods (everyone was stuck at home buying exercise bikes and laptops) just as supply chains seized up, and core goods inflation spiked to levels not seen in decades — used-car prices alone rose more than 40% year-over-year at the 2021-22 peak. That goods spike was a huge part of the *initial* inflation surge.

Then it reversed. As supply chains healed and demand rotated back toward services, core goods inflation collapsed and even turned *negative* (outright deflation) through 2023. This is the short lead/lag the figure shows between PPI and core goods: upstream producer prices for goods turned down, and consumer goods prices followed within a month or two, because goods supply chains are fast relative to the year-long lease cycle in shelter. The teaching point for correlation is that core goods is the *fastest* and most *mean-reverting* of the three core slices — it spikes and crashes — while shelter is the slowest and supercore is the stickiest. A single "core" number blends a fast-reverting goods component, a year-lagged shelter component, and a sticky wage-driven supercore component, which is exactly why decomposing core is so much more informative than reading it as one figure. When core fell in 2023, a naive reading credited "disinflation broadly"; the decomposition showed it was goods deflation and (later) shelter catching down doing the work, while the dangerous wage-driven supercore stayed hot. The components told the truth; the blend did not.

#### Worked example: nowcasting shelter from market rents

Suppose it is mid-2022. New-lease market rents (Zillow ORI) are growing about +13% year-over-year, but CPI shelter is printing about +5.8% and rising. A naive forecaster extrapolates the official shelter trend and worries it could keep accelerating toward double digits. A trader who knows the lag does the math instead:

```
CPI shelter today      ~= a 12-month-lagged average of new-lease rents
market rents now        = +13% (but this won't fully hit CPI for ~12 months)
market rents 6mo ago    = ~+15% (the peak; this is what's feeding CPI now)
market rents looking forward (already cooling) = heading toward +3-4%
forecast: CPI shelter keeps rising into early 2023, peaks ~8%, THEN falls through 2024
```

The lag turns shelter from a mystery into a *forecastable* quantity. Because you can see the leading indicator (real-time rents) a year before it shows up in CPI, you can predict the path of a third of the basket with unusual confidence. **The intuition: a lagged component is a gift to a forecaster — you already know its future, because its future is just the leading indicator's present.** This is precisely the trade many macro desks put on in 2022-23: they were short duration on the official sticky shelter print while privately confident that shelter would roll over in 2024 (and it did).

### Why the market re-weights toward the timely components

Put the two facts together and you see why the market's correlation is with core/supercore and *not* with the headline or even with shelter. The headline is contaminated by volatile energy; shelter, a huge chunk of core, is a twelve-month-lagged echo. So the most *information-dense*, *timely* signal in the report is supercore — core services ex-housing — because it is tied to current wages and current demand, with no big lag and no energy noise. From late 2022 the market literally re-centered its reaction function: desks built "supercore" trackers, parsed the services-ex-shelter line first, and traded the deviation of *that* from expectations. The headline became a press-release number; supercore became the trade.

## The asset side: how the components map to the tape

Let's connect the decomposition back to specific asset reactions, because the components don't just differ in timeliness — they differ in *what they imply for policy*, and that is what assets price.

### Hot supercore is the most hawkish read

A hot supercore print is the most hawkish thing a CPI report can contain, because supercore inflation is wage-driven and wage-driven inflation is *sticky* — it doesn't reverse the way an energy spike does. When supercore surprises hot, the market extends its expected Fed path: more hikes, or a longer hold. The reaction is the full hawkish bundle from the beta chart — yields up (front end most), dollar up, equities down, gold down. A hot *energy*-driven headline with cool supercore produces a much weaker, more transient reaction, because the market knows energy mean-reverts and the Fed mostly looks through it.

It helps to trace *why* each asset moves the way it does, because the chain is the same one that runs through every rate-driven correlation in the series. A hot core surprise lifts the expected Fed path, which lifts yields, and a higher discount rate is bad for any asset whose value is the present value of distant cash flows. The Nasdaq falls *more* than the S&P (−1.0% versus −0.7%) because its earnings are weighted further into the future — it is "longer duration" in the equity sense — so a given rise in the discount rate shaves more off its present value. This is the discount-rate channel, the master transmission covered in [real versus nominal: real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal). Gold falls (−0.8%) for a related reason: gold pays no yield, so its appeal moves inversely with the *real* yield, and a hawkish core surprise lifts real yields by pushing nominal yields up faster than it lifts inflation expectations. Bitcoin falls most of all (−1.6%) because in the 2022-23 regime it traded as the highest-beta macro-liquidity asset — a leveraged bet on easy policy — so a hawkish surprise hit it hardest. The dollar rises because higher US yields make dollar assets more attractive relative to the rest of the world, pulling capital in. Every one of these is a *consequence* of the core surprise revising the Fed path; none of them is reacting to the headline directly.

### A "good" shelter print is double-edged

A *cooling* shelter number is, on its surface, disinflationary and dovish. But sophisticated desks discount it, precisely because shelter is lagged and largely *predictable* — the market already knew shelter would cool in 2024 because it could see the leading rent indices. A cooling shelter print that merely confirms what everyone already forecast moves the market little. A shelter print that cools *faster* than the lag model predicted is a genuine dovish surprise; one that stays sticky *longer* than the lag model predicted (as happened repeatedly in 2023) is a genuine hawkish surprise. **The surprise is measured against the lag-model expectation, not against zero.** This is the subtle reason shelter, despite being a third of the basket, often has a *small* same-day beta: most of its move is anticipated by the model.

#### Worked example: shelter's contribution to the 2024 disinflation

Suppose CPI shelter decelerates over a year from 6.0% to 5.0% year-over-year — a one-point fall in the shelter rate. With shelter at about 34% of the headline basket, the mechanical contribution to headline disinflation is:

```
shelter weight  = 0.34
shelter rate change = -1.0pp (from 6.0% to 5.0%)
headline contribution = 0.34 * (-1.0) = -0.34pp of headline disinflation
```

So a one-point drop in shelter alone pulls roughly a third of a point off headline (and an even larger share off core, where shelter is a bigger slice). Across 2024, shelter fell by more than a point, contributing the largest single chunk of the move that took core toward 3% and below. Because the leading rent indices had already cooled in 2022-23, this disinflation was *visible a year early*. **The intuition: the biggest contributor to the 2024 disinflation was the most forecastable component, so the disinflation itself was largely foreseeable — the edge was in believing the lag model when the official print still looked sticky.**

### The components in the broader matrix

Zooming out, the CPI components feed into the master inflation-surprise row of the series' big cross-asset map. A hot core/supercore surprise behaves like a generic hawkish inflation shock: negative for stocks, bonds, gold, and crypto, positive for the dollar. The full indicator-by-asset picture is assembled in [the macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix); here the lesson is just that "CPI surprise" in that matrix really means "core/supercore surprise," because that is the part the matrix's betas are estimating.

### The PCE read-through trade

There is a sophisticated component trade that follows directly from all of this, and it is worth spelling out because it is a daily occupation of inflation desks. The Fed targets core *PCE*, not core CPI, and PCE is released two-plus weeks after CPI. But PCE shares many of the same source data, so a desk that knows the component detail of CPI can *forecast* the upcoming PCE print better than the consensus. The trick is that PCE weights the components very differently — shelter is only about 15% of core PCE versus roughly 42% of core CPI, and health care is much larger in PCE because it counts what insurers and the government pay, not just out-of-pocket costs. So a CPI report that is hot *because of shelter* implies a much *cooler* PCE read-through than a CPI report that is hot because of broad supercore, since PCE down-weights the shelter that drove CPI. A desk that reads the CPI components can therefore position for the PCE surprise before it prints: hot-CPI-on-shelter is a fade for the PCE reaction; hot-CPI-on-supercore is a follow-through. This component-level read-through is one of the cleanest edges the decomposition offers, and it is the bridge to the next post in this track. The PCE/CPI gap, its lower beta, and the full forward-inflation complex (breakevens, 5y5y, TIPS) are developed in [PCE, breakevens, and the forward inflation correlation](/blog/trading/macro-correlations/pce-breakevens-and-the-forward-inflation-correlation) and the release-day mechanics in [PCE, the Fed's preferred inflation gauge](/blog/trading/event-trading/pce-the-feds-preferred-inflation-gauge).

## Common misconceptions

Several beliefs about CPI components are widespread and wrong. Each one costs money.

**Myth 1: "Headline inflation is what matters because it's what people actually pay."** True for your wallet, false for the market. The market trades the *Fed's reaction*, and the Fed sets policy on core. In 2023 the headline fell six points while the Fed kept hiking — if the headline were what mattered, that would have been impossible. Your grocery bill and the bond market are reacting to different objects: you feel the level, the market trades the core surprise.

**Myth 2: "Core inflation is a way to make inflation look lower than it really is."** No. Over long horizons core and headline converge, because energy can't fall forever; core is not systematically lower, it is systematically *less volatile*. In some years (a falling-oil year like 2023) core is *higher* than headline; in a rising-oil year, core is lower. Core is a trend estimate, not a downward fudge.

**Myth 3: "If shelter is a third of the basket, it must be the most important number."** It is the most *weighted* and the *least timely*. Because it lags real-time rents by about a year, its current print is mostly already forecastable, so its *surprise* — the only part the market trades — is usually small. Importance for the trade is about surprise, not weight.

**Myth 4: "Falling headline inflation means falling rates are coming."** Only if the fall is in the sticky core. The 2023 headline fell almost entirely on energy base effects while core stayed near 4.8%; rates went *up*, not down. A disinflation that is all energy is a head-fake; a disinflation that shows up in supercore is the real thing.

**Myth 5: "CPI and the Fed's gauge are the same, so it doesn't matter which you watch."** They're cousins, not twins. The Fed targets core PCE, which weights shelter much less than CPI does (about 15% versus 34%) and uses a different formula, so it typically runs a few tenths *below* core CPI. A core CPI print can look hot while core PCE is closer to target — which is exactly why the market also dissects the components to estimate the PCE read-through. (The PCE/CPI gap, breakevens, and forward inflation get their own treatment in [PCE, breakevens, and the forward inflation correlation](/blog/trading/macro-correlations/pce-breakevens-and-the-forward-inflation-correlation) and the event-trading note [PCE, the Fed's preferred inflation gauge](/blog/trading/event-trading/pce-the-feds-preferred-inflation-gauge).)

## How it shows up in real markets

The components-not-headline story is not a textbook abstraction; it is the explicit driver of several of the biggest macro moves of the cycle.

### 2023: the headline fell, the market traded the sticky core

We opened with this, and it deserves the full numbers. From the June 2022 peak of 9.1% to June 2023's 3.0%, headline CPI fell almost exactly six percentage points. The overwhelming majority of that decline was energy: gasoline prices were down roughly 27% year-over-year by mid-2023, and energy as a whole was deeply negative, mechanically subtracting from the headline. Meanwhile core CPI fell far more slowly — from about 5.9% to 4.8% over the same window — and supercore stayed hot on wages. The Fed delivered hikes in February, March, May, and July 2023, taking the funds rate to 5.25–5.50%. The two-year yield, far from collapsing as a "3% headline" might suggest, climbed to over 5% by October 2023.

![Path chart of headline CPI and core PCE showing the headline falling faster than the sticky core](/imgs/blogs/core-cpi-shelter-and-supercore-what-actually-correlates-4.png)

The chart above shows the divergence with the Fed's own preferred core gauge, core PCE. The grey headline line plunges from 9.1%; the blue core line descends gently and stays well above the 2% target deep into 2023. The red annotation marks the gap that defined the trade: a market reading the headline would have been buying duration and risk; a market reading the core stayed defensive, and the core-readers were right — the Fed wasn't done. This is the cleanest real-world proof that the correlation is with the component, not the blend.

![Before and after comparison of what headline CPI said versus what the market actually traded in 2023](/imgs/blogs/core-cpi-shelter-and-supercore-what-actually-correlates-6.png)

The before/after figure crystallizes the divergence in plain language. On the left, the naive headline narrative: inflation crashing on energy, problem solved, buy everything. On the right, what desks actually did: noted that core and supercore were still hot, that the Fed path keyed on those, and positioned for higher-for-longer. The tape — yields grinding up through 2023, the front end pinned near 5% — followed the right column.

### 2023-24: the shelter-lag debate

Through 2023 there was a genuine, high-profile argument among economists and traders about shelter. One camp pointed at the official CPI shelter index, still printing 6-8%, and warned inflation was sticky and far from beaten. The other camp pointed at real-time rent indices — already cooling to low single digits since late 2022 — and argued the official shelter print was a *lagged artifact* that would inevitably roll over, dragging core down with it in 2024.

The lag-aware camp was largely right. As the cool 2022-23 market rents finally worked through the lease stock, CPI shelter decelerated through 2024, and that deceleration was the single biggest contributor to the fall in core inflation toward 3% and below. Desks that had nowcasted shelter from the leading rent indices were positioned for that disinflation a year early. The episode is the cleanest possible illustration of the series' core lesson: timing the *read* of a lagged component is itself the edge. (The general framework — leading versus coincident versus lagging, and why timing the read matters — is in [leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators).)

### The 2021 head-fake in reverse

The lag cuts both ways, and it fooled the "transitory" camp in 2021. As market rents began surging in early-to-mid 2021, the official CPI shelter index was still subdued, because the surge hadn't rolled through the lease stock yet. Policymakers and many investors, looking at the still-low official shelter number, underestimated how much inflation was already in the pipeline. The leading rent indices were screaming, but the lagged official measure was quiet, and the quiet measure was the one in the headline. A year later, in 2022, that pipeline emptied into CPI shelter and helped push core to multi-decade highs. The lesson is symmetric: a quiet *lagged* component is not reassurance when the *leading* indicator is already hot.

The cost of that mistake was enormous and concrete. Because the most-weighted component looked benign in real time, the Fed and a large fraction of the market accepted the "transitory" framing for too long, kept policy ultra-easy deep into 2021, and then had to tighten with unusual violence in 2022 — the fastest hiking cycle in forty years — once the lagged shelter pipeline (plus everything else) forced its hand. That abrupt tightening is what produced the synchronized 2022 selloff in which stocks *and* bonds fell together. In other words, a single under-appreciated lead/lag relationship inside one component of one report contributed to one of the worst years for diversified portfolios in modern history. It is a vivid reminder that "which component, and how lagged" is not a pedantic detail — it is the difference between reading the inflation trend correctly and being blindsided by it. (For why stocks and bonds fell together in 2022 — the inflation-regime driver of the correlation flip — see [the stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) is part of the broader story, and the cross-asset companion [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).)

#### Worked example: the contribution math behind a sticky core

Let's compute why core stayed near 4.8% in mid-2023 even as the headline crashed. Using rough core sub-weights and approximate mid-2023 year-over-year rates:

```
shelter:        weight ~42% of CORE,  rate ~7.8%  -> contribution = 0.42 * 7.8 = 3.28pp
supercore:      weight ~33% of CORE,  rate ~4.6%  -> contribution = 0.33 * 4.6 = 1.52pp
core goods:     weight ~25% of CORE,  rate ~0.6%  -> contribution = 0.25 * 0.6 = 0.15pp
                                          core total ~= 3.28 + 1.52 + 0.15 ~= 4.95pp
```

(Weights here are shares *within* core and are illustrative; they sum to 100% of core, not of the headline.) The arithmetic shows the lagged shelter component alone contributed about 3.3 points — two-thirds of the entire core rate — purely because of its size and its still-hot lagged print. Energy, sitting *outside* core, did all the work of pulling the headline down but touched core not at all. **The intuition: a sticky core is sticky because its biggest piece (lagged shelter) holds it up, while the falling piece (energy) lives outside core and can't pull it down.** This decomposition is the entire reason a 3% headline coexisted with a hiking Fed.

## How to read it and use it

Here is the practical playbook for trading a CPI release through its components rather than its headline.

### The read order

When the report drops, read the lines in this order, not top-down:

1. **Core month-over-month first.** This is the cleanest run-rate. Compare it to consensus; the surprise here is your primary signal.
2. **Supercore (core services ex-shelter) second.** This is the stickiest, most wage-driven, most hawkish-when-hot piece. A hot supercore is the strongest hawkish signal in the report.
3. **Shelter third, against the lag model.** Don't ask "is shelter high?" (it usually is). Ask "is shelter higher or lower than my lag-model forecast from real-time rents?" Only the deviation from the model is news.
4. **Headline and energy last,** purely to understand the optics and the first-second algorithmic reaction you may need to fade.

### The signal

The tradeable signal is the **core surprise**, with supercore as the highest-conviction sub-signal. Map it to assets through the core beta: a +0.1pp core upside surprise implies roughly S&P −0.7%, Nasdaq −1.0%, 2Y +9bp, 10Y +7bp, DXY +0.35%, gold −0.8%, Bitcoin −1.6%, in an inflation-fear regime. Scale linearly for bigger surprises and remember the front end (2Y) carries the cleanest read because it is the purest Fed-path bet.

### The regime check

The signs above are **regime-conditional**, and this is the single most important caveat. In the 2022-23 inflation-fear regime, hot core was unambiguously bad for risk: the dominant fear was the Fed, so good-on-inflation was good for markets and bad-on-inflation was bad. In a *benign* inflation regime — when inflation is low and stable and the market's attention is on growth — a CPI surprise barely moves anything, because it doesn't change the Fed path. And in a *deflation-scare* regime, a hot core print could even be *risk-positive* (it relieves fears of a deflationary spiral). Always ask first: *what regime are we in?* The beta's magnitude and even its sign depend on it. (The regime-dependence of the whole inflation-equity relationship — including the non-linear U-shape where moderate inflation is best for stocks — is the subject of [inflation and stocks, the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips); the broader principle that correlation is a regime, not a constant, anchors the whole series in [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant).)

The decay of the CPI beta through 2024-25 is a live illustration. As inflation drifted back toward the low-to-mid 3% range and the Fed shifted from hiking to a hold-then-cut stance, the same-day asset reaction to a given core surprise *shrank*. A core print that would have moved the S&P 1% in October 2022 might move it a quarter of that in 2025, because the Fed path is less sensitive to any single inflation print once policy is restrictive and the central bank has signaled it is mostly done. The beta did not disappear — it is still negative, hot core is still hawkish — but its *magnitude* fell as the regime cooled. This is why you must re-estimate the beta on a rolling window rather than trusting a number from the peak-fear period; a stale beta will badly over-predict the reaction in a calmer regime. The mechanics of estimating a time-varying beta — rolling windows, EWMA, regime conditioning — are exactly what the series' measurement track is for, and the rolling-correlation framing is developed in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters).

### What invalidates the read

Three things invalidate the components-not-headline framework and should make you stand down:

1. **A regime shift.** If the Fed has clearly pivoted to a growth/employment focus (a labor-market scare), the market may start trading the *jobs* data and look through CPI entirely. The CPI beta shrinks toward zero.
2. **A measurement change.** The BLS periodically reweights the basket and has changed the OER methodology; a methodology break can make the shelter lag behave differently than the historical model predicts. Re-estimate after any known methodology change.
3. **A consensus that already priced the components.** If a hot PPI two days earlier already moved the whisper number on core, then a "hot" core print that merely matches the whisper won't move the market. The surprise is always relative to the *real-time* expectation, which can drift from the published consensus. (This is the surprise-versus-whisper subtlety covered in [correlate the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises).)

### The one-line takeaway

Don't trade the blend. Strip CPI to core, watch supercore for the sticky wage-driven signal, treat shelter as a lagged-but-forecastable echo of real-time rents, and key your asset beta to the **core surprise** — because that, not the headline, is what the Fed reacts to and therefore what the market trades.

## Further reading and cross-links

Within this series:

- [CPI and asset prices, the master inflation correlation](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation) — the parent post: CPI's surprise-to-asset map across regimes, which this post decomposes into components.
- [Correlate the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — why the market trades actual-minus-consensus, and how the beta is estimated. The core surprise here is a special case.
- [Leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) — the cross-correlation/lead-lag framework that makes the shelter lag a forecasting edge.
- [PCE, breakevens, and the forward inflation correlation](/blog/trading/macro-correlations/pce-breakevens-and-the-forward-inflation-correlation) — the PCE cousin of core CPI, with shelter weighted far less, plus market-implied inflation.
- [Inflation and stocks, the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips) — the regime-dependence and non-linearity of the inflation-equity relationship.
- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the founding thesis of the series.
- [The macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — where the CPI-surprise row sits in the full cross-asset map.

For the mechanism and the release-day reaction (cited, not re-derived here):

- [CPI, the report that moves the world](/blog/trading/event-trading/cpi-the-report-that-moves-the-world) — the intraday choreography of a CPI release, including the headline-then-core reversal.
- [PCE, the Fed's preferred inflation gauge](/blog/trading/event-trading/pce-the-feds-preferred-inflation-gauge) — the release-day reaction to PCE and how it differs from CPI.
- [Interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — why the front-end yield is a forecast of the Fed path.
- [How policy moves every asset](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) — the cross-asset transmission of a rate-path revision.
