---
title: "CPI: The Report That Moves the World"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner's anatomy of the US Consumer Price Index — headline vs core, shelter and the lag, supercore, MoM vs YoY, and how the market decides a print is hot or cool — the single most market-moving scheduled release on Earth."
tags: ["event-trading", "macro", "cpi", "inflation", "core-cpi", "supercore", "shelter", "federal-reserve", "cross-asset", "stocks", "bonds", "crypto"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The US Consumer Price Index is the single most market-moving scheduled release on Earth, because it drives the Fed's interest-rate path — and rates price everything.
>
> - CPI measures how fast the prices of a fixed basket of goods and services are rising. The number on the news is **headline**; traders and the Fed care more about **core** (ex food & energy) and **supercore** (core services ex-housing).
> - One report repriced every market in 2022: on 13 September, a CPI just **+0.2pp** above consensus sent the S&P 500 down **−4.32%**, Bitcoin roughly **−9.4%**, and the dollar up. The reaction runs through the Fed: hotter inflation means higher-for-longer rates.
> - To read a print like a trader, watch the **month-over-month** changes in core and supercore against consensus — not the year-over-year headline. A single tenth of a percent on core can move trillions.
> - The one number to remember: the basket is dominated by **shelter (~34%)**, which is sticky and lagging — so core stays elevated long after headline inflation falls.

At 8:29 a.m. Eastern on a CPI day, the most liquid markets on the planet go quiet. Bid-ask spreads on S&P 500 futures widen. Traders pull resting orders. The whole macro world — every hedge fund, every bank desk, every algorithm, and a surprising number of people who think they don't care about economics — holds its breath for a number that the US Bureau of Labor Statistics will publish at exactly 8:30. The number is the Consumer Price Index, and on the morning it lands, one tenth of one percent on a single line of it can move trillions of dollars of value across stocks, bonds, currencies, gold and crypto in the span of a few minutes.

It sounds absurd when you say it plainly. The CPI is, at bottom, a measurement of how much a typical shopping cart costs this month versus last month. Why should a shopping-cart price index be the most powerful scheduled event in global finance — more reliably market-moving than any earnings report, any election, any geopolitical headline? Because of one chain of causation that this post exists to make obvious: **CPI tells the market what inflation is doing, inflation tells the Federal Reserve what to do with interest rates, and interest rates are the price of money — the single input that sits underneath the valuation of every asset on Earth.** Move the expected path of rates, and you move everything at once.

This is the structural primer for the series. We are not going to trade CPI in this post — the playbook for how to position around a release, scenario by scenario, lives in a sibling post. Here we do something more foundational and more durable: we take the report apart. By the end you will be able to read a CPI release the way a desk trader reads it — knowing the difference between headline and core, why the Fed obsesses over a slice called "supercore," why the shelter component makes inflation sticky, what month-over-month versus year-over-year actually means, and exactly how the market decides, in the first ten seconds, whether a print is hot or cool.

![Layered breakdown of CPI from the whole basket down to core and supercore](/imgs/blogs/cpi-the-report-that-moves-the-world-1.png)

## Foundations: what CPI actually measures

Before any of the trading intuition makes sense, you need to know precisely what the number *is*. If you have no economics background, this section is the load-bearing wall. We will define every term from zero, with an everyday picture before any formula, and then go deep.

### CPI is the price of a fixed shopping cart, tracked over time

The **Consumer Price Index (CPI)** measures the average change over time in the prices that urban consumers pay for a fixed "basket" of goods and services. The agency that builds it is the **Bureau of Labor Statistics (BLS)**, part of the US Department of Labor. Every month, BLS field staff and automated feeds collect roughly **80,000 prices** across about 200 categories — groceries, gasoline, rent, restaurant meals, airline tickets, haircuts, medical visits, cars, clothing — in cities all over the country. They then combine those prices into a single index number.

The cleanest way to hold the idea: say the BLS fills one enormous shopping cart with everything a typical household buys in a month, in the same proportions every month, and rings it up at the register again and again. If the cart cost \$100.00 last month and \$100.30 this month, prices rose 0.3% over the month. That 0.3% is the heart of CPI. Everything else — headline, core, supercore, seasonal adjustment — is a refinement of *which items go in the cart* and *how you express the change*.

The index is anchored to a base period (1982–84 = 100), so the level itself (say, 314) is meaningless in isolation. What matters is the *change*: this month versus last (month-over-month) and this month versus the same month a year ago (year-over-year). We'll come back to that distinction, because the gap between the two is where a lot of market confusion lives.

One subtlety worth flagging up front: CPI is a *fixed-basket* index, which means it can drift away from true cost of living in two ways economists argue about endlessly. People **substitute** — when beef gets expensive they buy chicken — so a rigid basket overstates the pain. And **quality improves** — this year's \$1,000 phone is far better than last year's \$1,000 phone — so a naive price comparison overstates inflation unless you adjust for quality (the BLS does, through "hedonic" adjustments). These adjustments are genuinely contentious, and they're why you'll hear people claim "real inflation is higher than CPI." For a trader, the debate is mostly irrelevant: what moves markets is the *change in the official number versus the consensus*, whatever its philosophical imperfections. The market trades the number the Fed reacts to, and the Fed reacts to official CPI and PCE.

Why does this one statistic carry so much weight, when there are dozens of economic releases each month? Because of where it sits in the chain of causation. Interest rates are the price of money, and the present value of every financial asset — a stock, a bond, a house, a Bitcoin — is its future cash flows discounted back at some rate tied to that price of money. Raise the rate path and you lower the present value of *everything* simultaneously; lower it and you lift everything. CPI is the most timely, highest-frequency read on the one variable (inflation) that most directly determines that rate path. It is, in effect, the market's monthly readout on the discount rate for the entire planet's assets. Nothing else on the scheduled calendar has that combination of frequency, freshness, and direct line to the Fed.

### Headline, core, and supercore: three nested measures

The single most important thing a beginner must learn is that "CPI" is not one number — it is a family of nested measures, each stripping out a noisier layer to reveal the trend underneath.

- **Headline CPI** is the whole basket: every category, food and energy included. This is the number on the evening news and the one politicians quote. It is the truest measure of what a household actually experiences at the register.
- **Core CPI** is headline *minus food and energy*. Food and energy prices are extremely volatile — a frost in Brazil, a heat wave, an OPEC production cut, a pipeline outage — and they swing up and down for reasons that have nothing to do with the broad, persistent inflation trend. Stripping them out gives a cleaner read on where inflation is really heading. Traders and the Fed watch core far more closely than headline.
- **Supercore** (officially "core services excluding housing," sometimes "core services ex-shelter") goes one step further: it takes core and *also* removes housing/shelter. What's left is the price of labor-intensive services — medical care, haircuts, restaurant labor, insurance, transportation services. Because those prices are driven mostly by wages, supercore is the slice the Fed treats as the truest signal of underlying, demand-driven inflation.

Each measure trades breadth for signal. Headline is what people feel; core is the trend; supercore is the part the Fed believes it can actually control with interest rates.

### Food, energy, and why volatility gets stripped

Why strip food and energy at all, when they're a real part of life? Because they are *noise* on a monthly basis. Gasoline can move 10% in a month on a refinery problem and reverse it the next. If you let that swing dominate the inflation read, you'd whipsaw policy — hiking rates because gas spiked, cutting them because it fell back, when nothing about the underlying trend changed. Central bankers around the world strip volatile components for exactly this reason: they want the *persistent* signal, not the weather.

### Shelter, OER, and the lag that makes inflation sticky

The largest single component of CPI is **shelter** — the cost of housing — at roughly **a third of the entire basket**. Within shelter, the biggest piece is something with an awkward name: **Owners' Equivalent Rent (OER)**. Most Americans own their homes rather than rent, but you can't put a house *purchase* in a consumption index (a house is an asset, not a monthly consumption item). So the BLS asks, in effect: *if you rented out your own home, what would it fetch?* That estimated rent is OER, and it stands in for the cost of housing services that homeowners consume.

The crucial trading fact about shelter is that it is **sticky and lagging**. Rents reset slowly — most leases are annual, and the BLS samples a given unit only every six months — so the shelter index reflects rent agreements signed months ago, not today's market rent. When real-time rents surge, shelter CPI keeps climbing for a year afterward; when real-time rents cool, shelter CPI keeps falling long after the turn. This lag is why **core inflation stays elevated after headline inflation has already fallen**: the volatile food-and-energy pieces drop fast, but the giant, slow shelter component is still catching up to where the rental market was a year ago.

There's one more wrinkle that confuses newcomers: shelter CPI does *not* track home prices. When house prices were soaring in 2021, you'd expect CPI to soar with them — but it didn't, because CPI measures the *consumption of housing services* (rent and OER), not the *purchase of a housing asset*. A house you buy is an investment; the service of living in it is consumption. So a housing-price boom shows up in CPI only slowly and indirectly, as higher rents and higher OER eventually feed through. This is a frequent source of "the government is lying about inflation" complaints — house prices were up 20% but CPI shelter was up 5% — and the honest answer is that they measure two genuinely different things.

### Month-over-month vs year-over-year

There are two ways to express the change, and confusing them is the most common beginner mistake.

- **Year-over-year (YoY)** compares this month's index to the same month a year ago. "Inflation is 3.2%" almost always means YoY. It's the headline figure because it's intuitive: prices are 3.2% higher than a year back.
- **Month-over-month (MoM)** compares this month to *last* month. A 0.3% MoM print is the freshest possible read — it tells you what inflation did *right now*, not averaged over twelve months.

Traders care most about **MoM core**, because YoY is a trailing twelve-month average that can be dominated by "base effects" — what happened in the month that just rolled off the back of the window. A scary-looking YoY number can be entirely about a year-ago comparison; the live MoM tells you the current run-rate.

### Seasonal adjustment

Some price changes are perfectly predictable by calendar: airfares jump every summer, retail discounts hit every January. **Seasonal adjustment (SA)** removes those repeating calendar patterns so you can compare months cleanly. The seasonally adjusted MoM number is what markets trade; the non-seasonally-adjusted (NSA) series is used for the YoY headline and for contract escalations. When a desk says "core was 0.3 on the month," they mean SA MoM core.

That's the whole vocabulary. Now we go deep on each piece, and then watch it move markets.

## 1. The components and weights — what's actually in the basket

CPI is a weighted average, and the weights are everything. A 1% rise in airfares barely registers; a 1% rise in shelter is an earthquake. The weights are *expenditure shares* — roughly how much of the typical household's budget goes to each category. The BLS updates them from the Consumer Expenditure Survey, so they track what people actually buy.

The rough breakdown (weights shift a little year to year, but the shape is stable):

- **Shelter ~34%** — by far the largest, dominated by rent and Owners' Equivalent Rent.
- **Food ~13%** — groceries plus food away from home (restaurants).
- **Energy ~7%** — gasoline, electricity, natural gas, fuel oil.
- **Transportation (ex energy) ~8%** — new and used cars, car insurance, repairs, airfares.
- **Medical care ~7%**, **recreation ~5%**, **apparel ~2.5%**, **education & communication ~6%**, and a long tail of everything else.

When you put it that way, two facts jump out. First, **shelter alone is bigger than food and energy combined.** That single category, sticky and lagging, swings the whole index more than any other. Second, **core (ex food & energy) is about 80% of the basket** — so "stripping out food and energy" removes only a fifth of the weight while removing most of the month-to-month noise. That's why core is such a good trade-off.

The weights also explain why the market reacts to *internals* rather than the headline alone. If headline is hot but the heat is all in gasoline (a 7% weight that will reverse), the trade is different than if the heat is in shelter and services (a 60%+ weight that is persistent). Reading CPI means reading *where* the inflation is, not just *how much*.

#### Worked example: how a shelter swing moves the headline

Shelter is roughly **34%** of the CPI basket. Suppose shelter inflation comes in 0.5 percentage points hotter than expected for the month. How much does that push the headline, all else equal?

- Contribution to headline = weight × component change = 0.34 × 0.5pp = **0.17pp**.
- So a half-point shelter surprise moves headline CPI by about **0.17pp** — roughly two of the "tenths" that markets obsess over, from one category.
- By contrast, a 0.5pp surprise in apparel (≈2.5% weight) moves headline by 0.025 × 0.5 = **0.0125pp** — essentially nothing.
- Same component-level surprise, **14× the market impact**, purely because of the weight.

Intuition: in CPI, a category's market power is its weight — which is why a sleepy, slow-moving thing called shelter is the most important number in the report.

### How the number is actually built

It helps to see the assembly line that turns 80,000 price tags into one market-moving figure, because each station along it is a place where the number can surprise you.

![Pipeline of how CPI is built from price collection through weights seasonal adjustment index and reported change](/imgs/blogs/cpi-the-report-that-moves-the-world-5.png)

Walk the stations left to right. First, **collection**: BLS economic assistants visit physical stores and service establishments and pull prices, increasingly supplemented by web-scraped and corporate transaction data. For housing, a separate survey of rental units feeds the rent and OER calculations. Second, **weighting**: each price is weighted by its expenditure share from the Consumer Expenditure Survey, so a 1% move in shelter counts far more than a 1% move in apparel — this is the weight math we just did. Third, **seasonal adjustment**: the BLS strips out repeating calendar patterns so a summer airfare bump doesn't masquerade as inflation. Fourth, the weighted, adjusted prices are **combined into the index level** (anchored to 1982–84 = 100). Fifth and sixth, the BLS reports the **month-over-month** change (the fresh run-rate the market trades) and the **year-over-year** change (the headline figure).

Two practical notes for a trader. The seasonal-adjustment factors are *revised* every February, which can shift the recent monthly history and occasionally rewrite the inflation narrative without any new prices at all. And the BLS publishes not just the headline tenths but a full table of *contributions* — how much each category added to or subtracted from the monthly change. The contributions table is where the real read lives: it tells you whether a 0.3% core was driven by a one-off (used cars, airfares) that will reverse, or by broad-based services that will persist. Amateurs read the headline tenth; professionals read the contributions.

The release itself is a piece of theater worth understanding. The data is produced under tight security — a "lock-up" — and published to everyone at the same instant, 8:30 a.m. Eastern, on a pre-announced schedule. There is no leak, no early look; the entire market gets the number simultaneously, which is exactly why the reaction is so violent and so synchronized. Tens of thousands of participants, all positioned for the consensus, all receive the surprise in the same millisecond and all try to adjust at once. That simultaneity is the engine of the spike.

## 2. Headline vs core — why traders watch core

The public watches headline; the Fed and the trading desks watch core. Here is the full comparison, measure by measure.

![Comparison grid of headline core and supercore measures and what each strips out](/imgs/blogs/cpi-the-report-that-moves-the-world-3.png)

The logic is the **signal-to-noise** trade-off. Headline is the most *complete* picture of cost of living, but it's the noisiest — energy alone can drive a headline beat or miss that says nothing about the trend. Core removes the two noisiest categories and gives a far steadier read on where inflation is actually settling. When the Fed says it is "data dependent," the data it depends on most is core, because core is what its rate policy can actually influence: monetary policy works on demand-driven, broad-based inflation, not on the price of crude oil.

This is why you'll see a CPI release where **headline beats expectations (looks bad) but core is soft (looks good)**, and the market *rallies*. The headline grabbed the gasoline spike; the core told the Fed the underlying trend is cooling; the market trades the Fed's read, not the household's. The reverse happens too: a benign headline masking hot core services will sell off. The professional reflex is to glance at headline and then immediately ask: *what did core do, month over month?*

There is a deeper reason the Fed leans on core that's worth internalizing: the Fed's actual target is **PCE inflation**, a slightly different index built by the Bureau of Economic Analysis, and it targets **2%** over the long run. But CPI comes out first and feeds into PCE, so CPI day is when the market gets its first read on the inflation the Fed will eventually see. Core CPI is the best same-day proxy for the persistent inflation the Fed is trying to steer back to 2%. (For the mechanism connecting inflation to the rate decision, see the macro-trading deep dive on [inflation and the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot).)

It's worth being precise about *why* CPI and PCE differ, because the gap occasionally matters for the trade. PCE uses different weights (it counts spending paid on your behalf, like employer healthcare, so its medical weight is much larger and its shelter weight much smaller — roughly 15% versus CPI's 34%). PCE also updates its weights more frequently, capturing substitution faster. The upshot is that CPI usually runs a few tenths *hotter* than PCE, and the two can tell slightly different stories — CPI more sensitive to shelter, PCE more sensitive to healthcare. A sharp desk knows that a hot CPI driven by shelter may translate into a *milder* PCE surprise three weeks later, because PCE weights shelter so much less. That's a second-order edge, but it's the kind of thing that separates a mechanical read from a real one.

Why does any of this reach beyond the bond market into stocks and crypto? Because of **duration**. A company's value is the present value of its future profits; a fast-growing tech company earns most of its profits far in the future, so its value is highly sensitive to the discount rate — it has long "duration," like a long-dated bond. When a hot core print lifts the expected rate path, the discount rate rises and those far-off profits are worth less *today*, so growth stocks fall hardest. That's why the Nasdaq (−5.16%) fell more than the Dow (−3.94%) on the hot September 2022 print: same news, but the long-duration index took the rate shock harder. Understanding duration is the bridge from "CPI moves rates" to "CPI moves my tech portfolio."

#### Worked example: the duration hit to a growth book

Make the duration idea concrete. Take a **\$30,000** growth-stock book and a **\$30,000** value-stock book into the hot 13 September 2022 print. The Nasdaq fell **−5.16%**; the Dow (a value-tilted index) fell **−3.94%**.

- Growth book P&L = \$30,000 × (−0.0516) = **−\$1,548**.
- Value book P&L = \$30,000 × (−0.0394) = **−\$1,182**.
- The extra pain on the growth book purely from its longer duration = \$1,548 − \$1,182 = **−\$366** on the same \$30,000, the same news, the same day.
- Hold both and the print cost **−\$2,730** combined.

Intuition: the more of your portfolio's value sits in far-future cash flows, the more a CPI-driven rate move hurts you today — duration is the channel that turns an inflation statistic into a tech-stock drawdown.

## 3. Shelter and the lag — why core stays sticky after headline falls

We met shelter in the foundations; now we make its lag the center of attention, because it is the source of one of the most important and most misunderstood dynamics in modern inflation.

![Shelter CPI versus headline CPI showing shelter peaking later](/imgs/blogs/cpi-the-report-that-moves-the-world-4.png)

Look at the two lines. Headline CPI is the dark line: it rockets to a 9% peak in mid-2022 and then collapses, falling back toward 3% within a year. Shelter is the amber line: it keeps climbing *after* headline has already turned, peaking near 8% in early 2023 — months later — and then descends slowly, still elevated in 2025 when headline has long since normalized. That offset is the lag in action.

The mechanism is mechanical, not mysterious. The shelter index is built from rents, and rents in the index are *backward-looking*: a tenant signs a 12-month lease, and that rent sits in the index for a year regardless of what the live rental market does. The BLS visits each housing unit only twice a year. So when market rents surged in 2021–2022, the index kept absorbing those high rents on a delay through 2023. When market rents cooled in 2023–2024, the index kept reflecting the older, higher rents and only gradually came down.

For a trader, this lag has two huge consequences:

1. **Core inflation looks "sticky" even when the inflation problem is largely solved.** Because shelter is ~40% of *core* (it's a bigger share of core than of headline, since core has dropped food and energy), a slow-falling shelter component drags core down slowly even after the rest of the basket has normalized. This is why, through 2023–2024, the Fed kept saying inflation was "too high" while disinflation was clearly underway — the headline measure was stuck on a lagging input.
2. **Smart desks watch real-time rent indices** (private market data from firms that track new leases) to forecast where CPI shelter is *going*, and therefore where core is going, *before* the official number catches up. If live rents have rolled over, you know official shelter — and core — will follow with a lag, even if today's print is still hot. That forward read is an edge.

#### Worked example: real income lost to 9.1% inflation

Inflation isn't a statistic on a screen — it's money out of your pocket. Take the June 2022 headline reading, the 9.1% (the YoY headline that month) peak of the wave, and a worker earning a **\$60,000** salary that did not rise.

- One year of 9.1% inflation means a dollar buys 9.1% less. Purchasing power lost = \$60,000 × 0.091 = **−\$5,460** over the year.
- Put differently, to buy the same basket you bought last year, you now need \$60,000 × 1.091 = **\$65,460** — a \$5,460 shortfall on a flat salary.
- If your raise was 4%, you got \$2,400 more (to \$62,400) but lost \$5,460 of purchasing power — a real income cut of about **−\$3,060**.
- That is why a hot CPI is genuinely bad news for households, not just for stock portfolios: the \$60,000 earner is poorer by thousands of real dollars.

Intuition: inflation is a tax you never voted for, and CPI is the meter that reads it — which is why the politics and the markets both fixate on this one report.

## 4. Supercore — the Fed's favorite slice

If core is the trend, **supercore** is the Fed's stethoscope on the heart of inflation. Recall the definition: supercore is **core services excluding housing** — you take core (ex food and energy) and remove shelter as well. What remains is the price of labor-heavy services: medical care, dining-out labor, car and home insurance, transportation services, recreation services, personal care.

Why does the Fed obsess over this narrow slice? Because it is the most direct read on **wage-driven, demand-side inflation** — exactly the kind of inflation monetary policy can fight. The logic chain runs: a hot labor market pushes wages up → service businesses are labor-intensive, so their costs rise → they raise prices → supercore inflation climbs. When supercore is hot, it tells the Fed that the economy is running too hot and that inflation has become *embedded* in the wage-price loop, not just a passing supply shock. When supercore cools, the Fed gains confidence that the underlying engine of inflation is slowing, even if the lagging shelter component is still keeping core elevated.

Fed Chair Jerome Powell explicitly highlighted this measure in 2022–2023 speeches as the thing he was watching most closely, which instantly made it a market obsession: the moment the chair tells the world which number he's watching, every desk recalculates that number within minutes of each CPI release and trades the surprise in *it*, not the headline. This is the series' core lesson in microcosm — the market trades whatever the rate-setter is reacting to, because the rate path is what reprices every asset.

Practically, supercore is *the* number that separates a sophisticated read from a naive one. A print can have a tame headline and a tame-looking core (because shelter is dragging it down on the lag) while supercore quietly re-accelerates — and that combination is *hawkish*, because it says the demand-driven core of inflation is heating back up. The amateur sees "headline 3%, looks fine"; the professional sees "supercore 0.5% MoM, that's a 6% annualized run-rate in the slice the Fed cares about — this is hot."

The intellectual scaffolding behind supercore is a framework Powell laid out explicitly: think of core inflation as three buckets, each with its own driver and its own policy relevance. **Core goods** (cars, furniture, appliances) are driven by supply chains and were the first to disinflate as 2021–2022 bottlenecks cleared — the Fed largely looks through them. **Housing services** (shelter) are driven by rents and lag by roughly a year — the Fed knows they'll come down mechanically, so a hot shelter print is "in the post" rather than new information. **Core services ex-housing** — supercore — is the bucket driven by the labor market, and it's the only one of the three that monetary policy directly controls by cooling demand. So when the Fed parses a CPI report, it mentally sorts the inflation into these three buckets and asks: how much of today's number is the bucket I can actually do something about? That's supercore.

This three-bucket view also explains the Fed's behavior that frustrated so many people in 2023–2024. Headline and even core were falling, and the public asked why the Fed wouldn't declare victory and cut. The answer: the *fall* was concentrated in core goods (transitory disinflation) and in the *promise* of falling shelter (lagging, already known), while supercore — the bucket that signals embedded, wage-driven inflation — remained stubbornly above target. The Fed wouldn't cut on a falling headline if the one bucket it controls was still hot. Reading CPI like the Fed means decomposing the print into these buckets, not just looking at the top line.

#### Worked example: annualizing a supercore tenth

Annualizing is how a desk converts a monthly tenth into a number it can compare to the 2% target. Suppose supercore prints **0.5%** month-over-month versus a **0.3%** expectation.

- A 0.5% monthly rate compounded twelve times annualizes to (1.005)^12 − 1 ≈ **6.2%**.
- The expected 0.3% monthly annualizes to (1.003)^12 − 1 ≈ **3.7%**.
- So a 0.2pp monthly miss in supercore is the gap between a **3.7%** and a **6.2%** annual run-rate in the Fed's favorite slice — the difference between "near target" and "we have a problem."
- On a \$25,000 equity book, a hawkish reaction of even −2% to that supercore re-acceleration is **−\$500** in a session, before the trend extends.

Intuition: the market doesn't trade the monthly tenth in isolation — it annualizes it in its head and compares to 2%, which is why a 0.5% supercore print, harmless-looking, is read as a 6% siren.

## 5. MoM vs YoY and the rounding that moves markets

We return to the month-over-month versus year-over-year distinction, because it is where the *mechanics of the surprise* actually live.

Markets trade the **seasonally adjusted month-over-month** changes — usually core MoM and supercore MoM — against the consensus forecast. And here is the detail that genuinely moves trillions: these are reported and forecast to **one decimal place**, i.e. in tenths of a percent. A core MoM forecast might be **0.3%**, and the print comes in **0.4%**. That single tenth — 0.1 percentage point — is a massive surprise, because:

- Annualized, the difference between 0.3% and 0.4% per month is the difference between roughly **3.7%** and **4.9%** annual inflation. That's not a rounding error to the Fed; it's the difference between "on track to target" and "we may need another hike."
- The market has *positioned* for the consensus. When the print misses by a tenth, every position built on the consensus must adjust at once — and they all try to adjust in the same direction in the same instant. That synchronized repositioning is the spike.

The **rounding itself** can move markets in a way that feels almost unfair. If the true unrounded core MoM is 0.349%, it prints as 0.3% (cool, market rallies); if it's 0.351%, it prints as 0.4% (hot, market sells off). The economy is identical to four decimal places, but the *reported tenth* flips, and so does the tape. This is why the most experienced desks also read the *internals and the unrounded contributions* the BLS publishes — to see whether a 0.3% was "high 0.3" (one fluke from being 0.4) or "low 0.3," which changes the forward read even when the headline tenth is the same.

**Base effects** are the other reason MoM beats YoY for trading. The YoY number is the sum of the last twelve MoM changes, so when a *high* month from a year ago "rolls off" the back of the window, the YoY can fall even if current inflation is flat or rising. Concretely: if inflation ran 0.9% in a given month last year and 0.3% this year, the YoY drops by 0.6pp just from that one comparison — not because anything got better this month, but because a bad month aged out. The financial press will trumpet "inflation falls!" and the market may shrug, because the desks knew the easy base-effect comparisons were coming and had already priced them. The surprise is always relative to what was *expected*, and the consensus already bakes in the known base effects. So a YoY that "falls" exactly as expected is a non-event; a YoY that falls *less* than the base effects implied is a hawkish surprise, even though the number went down.

This is the single biggest trap for retail traders: reacting to the *direction* of the YoY headline instead of the *surprise* in the MoM. The headline can fall and the market can sell off, because the fall was smaller than the base effects guaranteed — meaning the underlying MoM run-rate actually accelerated. Read the run-rate, not the trailing average.

#### Worked example: the annualized gap behind a single tenth

Put the "one tenth = a lot" claim on a firm footing. A core MoM of **0.3%** versus an expected **0.4%** is a 0.1pp *cool* surprise. What does each imply for the annual run-rate?

- 0.4% per month annualizes to (1.004)^12 − 1 ≈ **4.9%**.
- 0.3% per month annualizes to (1.003)^12 − 1 ≈ **3.7%**.
- That one tenth is a **1.2pp** swing in the implied annual pace — the difference between the Fed staying on hold and the Fed needing to do more.
- For a desk running a \$1,000,000 equity book, even a modest +1.5% relief rally on the cool tenth is **+\$15,000** banked in the session.

Intuition: the market reacts so hard to a single decimal because, annualized, that decimal is more than a full percentage point of inflation — and a full point is the gap between two completely different Fed paths.

#### Worked example: a 0.1pp core surprise on an equity book

Translate that single tenth into money. Take the 13 September 2022 session: August core CPI came in at 6.3% YoY versus 6.1% expected — about a **+0.2pp** core surprise — and the S&P 500 fell **−4.32%** on the day. Apply that to a **\$25,000** equity portfolio that tracks the index.

- Day's P&L = \$25,000 × (−0.0432) = **−\$1,080** in a single session.
- Roughly half that move can be attributed to each tenth of the surprise, so a *single* 0.1pp core beat was worth on the order of **−\$540** on this book that day.
- If you'd held \$100,000 of the index, the same day cost **−\$4,320**.
- And it compounded: that hot print helped push the S&P down further into its October 2022 low.

Intuition: a number you can't see without a magnifying glass — one tenth of one percent on a monthly statistic — is worth real four-figure swings on an ordinary retirement-sized portfolio, because it changes what the Fed will do with rates.

## 6. Hot vs cool — how the surprise is judged vs consensus

Everything converges here: on release, the market does not ask "is inflation high?" It asks "is this print **hot** or **cool** versus what was priced?" The answer is decided in seconds, and it's decided against the **consensus** — the average of economists' forecasts compiled by the data wires before the release. (The full surprise framework is its own foundational post: [why news moves markets](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework).)

The grading, in the order a desk actually reads it:

- **Headline MoM vs consensus** — the first wire flash, often the knee-jerk trigger.
- **Core MoM vs consensus** — the real read; this overrides the headline within seconds.
- **Supercore** — confirms or contradicts the core read; a hot supercore is hawkish even on a soft headline.
- **Shelter trend** — is the big lagging component finally rolling over, or still propping core up?

A print is **hot** when core (and especially supercore) comes in above consensus: it implies higher-for-longer rates, and in the inflation regime of 2022–2023 that meant *stocks down, yields up, dollar up*. A print is **cool** when core lands below consensus: lower rate path, *stocks up, yields down, dollar down*. An **inline** print — matching consensus — usually produces a small relief move and then a fade, because the uncertainty resolved without forcing anyone to reprice the rate path.

![Trader checklist flow from headline core supercore shelter to a hot inline or cool verdict](/imgs/blogs/cpi-the-report-that-moves-the-world-7.png)

Notice the sign convention is regime-dependent, and this is the most important caveat in the entire series. In 2022–2023, inflation was the dominant fear, so the market was in a "good-news-is-bad-news" mode: a *strong* economy or *hot* inflation meant more Fed tightening, which was bad for stocks. In a different regime — say, a growth scare where the fear is recession — a cool inflation print that lets the Fed *cut* is unambiguously good, and the same machinery rallies stocks for the opposite reason. Always say which regime you're in. (The mechanics of why the same number flips sign across regimes are covered in [the reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot), and the way one print propagates to every asset class is in [cross-asset transmission](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market).)

## How it reacted: real episodes

Theory is cheap. Here is CPI moving the actual tape, with real dated numbers.

### June 2022: the 9.1% peak that reset every market

By mid-2022, US headline CPI had climbed to a year-over-year reading that peaked around 9% — a 40-year high. The reconstructed monthly series shows the wave clearly.

![US CPI year over year from 2020 to 2026 with the 2022 peak and the 2 percent target](/imgs/blogs/cpi-the-report-that-moves-the-world-2.png)

That peak did far more than make headlines. It forced the Fed into the most aggressive tightening cycle in decades — the federal funds rate went from near zero to over 5% in about 18 months — and *that* is what reset every market. The 10-year Treasury yield tripled; the S&P 500 fell into a bear market; Bitcoin lost roughly three-quarters of its value from its 2021 high; the dollar surged to multi-decade highs. None of those moves were "about" the CPI number directly. They were about the *rate path* the CPI number forced. The chain — CPI → Fed → rates → every asset — is the whole reason this report sits at the center of global finance.

Notice the right edge of the chart, too: through 2025 and into 2026, CPI is *re-accelerating* off its lows, back above 4% in the reconstructed series. That matters because it kills the lazy assumption that inflation, once tamed, stays tamed. Each monthly CPI is a fresh referendum on whether the disinflation is durable or whether a new wave (tariffs, energy, a hot labor market) is building. A re-acceleration phase is precisely when CPI prints become *most* market-moving, because the market is no longer sure which regime it's in — and uncertainty is what makes a surprise expensive. The report that reset every market in 2022 has lost none of its power; it simply waits for the next inflection.

### 13 September 2022: a +0.2pp surprise, a −4.32% day

The single most instructive CPI session of the era. August CPI was released at 8:30 a.m.: headline 8.3% YoY versus 8.1% expected, core 6.3% versus 6.1% expected — a roughly **+0.2pp hot** surprise. Inflation was already known to be high; the *surprise* was that it wasn't cooling as fast as priced, and that core was re-accelerating. The cross-asset reaction was violent and uniform.

![Cross asset same day reaction to the hot August 2022 CPI release](/imgs/blogs/cpi-the-report-that-moves-the-world-6.png)

Read the bars as a single shockwave from one report. The S&P 500 fell **−4.32%** — its worst day since June 2020. The Nasdaq, heavy with long-duration growth stocks that are most sensitive to rates, fell **−5.16%**. The Dow dropped **−3.94%**. Bitcoin, the supposedly uncorrelated asset, fell roughly **−9.4%** — trading as a high-beta liquidity asset, it took the rate-shock harder than stocks. And the US dollar (DXY) *rose* **+1.4%**: hotter US inflation means a more hawkish Fed, higher US yields, and capital flowing into dollars. Gold dipped slightly as real yields rose. One report, six asset classes, the same instant, the same cause: a +0.2pp surprise that repriced the Fed.

Why did such a small miss do so much? Because the market was positioned for inflation to be *falling*, and the print said it was sticky. Every position built on the disinflation thesis had to unwind at once. That's the surprise framework in its purest form: the level (8.3%) was nearly known; the *surprise* (it didn't cool) was the trade.

#### Worked example: Bitcoin on a hot print

Crypto trades as a macro-liquidity asset, so it takes CPI shocks hard. Take a **\$10,000** Bitcoin position into the 13 September 2022 print, when BTC fell roughly **−9.4%** on the day.

- Day's P&L = \$10,000 × (−0.094) = **−\$940** in a single session.
- That's a steeper hit than the **−\$432** a \$10,000 S&P position took (−4.32%) the same day — crypto's higher beta to liquidity shows up as a bigger move on the same news.
- A \$50,000 BTC position lost **−\$4,700** that day on a number about food, rent and gasoline.
- Hold both: \$10,000 in BTC and \$10,000 in the S&P, and the hot print cost you **−\$1,372** combined.

Intuition: an asset that "doesn't care about governments" cared a great deal about one US government statistic — because that statistic sets the price of money, and crypto is priced off global liquidity. (More on this in [how crypto reacts to macro news](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) and the macro-liquidity framing in [real vs nominal yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).)

### The mirror image: cool prints rally just as hard

The framework is symmetric. One month later, on 10 November 2022, October CPI came in *cool* — 7.7% YoY versus 7.9% expected, with core also undershooting. The S&P 500 rose **+5.54%** that day, its best session in over two years; the Nasdaq jumped **+7.35%**; the 10-year Treasury yield collapsed **−28 basis points**; the dollar fell **−2.1%**. A year later, the cool October 2023 print (3.2% versus 3.3% expected) drove the rate-sensitive Russell 2000 small-cap index up **+5.44%** in a day. Same machinery, opposite sign: a cool surprise lowers the expected rate path, and everything rate-sensitive flies. The bigger the prior fear and the bigger the surprise, the bigger the move in either direction.

The November 2022 session also illustrates *which* assets move most. The Russell 2000 small-cap index and the Nasdaq outran the Dow on cool prints because small caps carry more floating-rate debt (so lower rates directly cut their interest bills) and growth stocks have longer duration (so a lower discount rate lifts their far-future earnings most). The pattern is mechanical and repeatable: on a cool surprise, buy what's most rate-sensitive; on a hot one, that same basket falls hardest. If you only remember one cross-asset rule from this post, make it that the *most rate-sensitive assets give you the most leverage to the CPI surprise* — which is wonderful when you're right about the direction and brutal when you're wrong.

#### Worked example: leveraging the cool print through small caps

On the cool 14 November 2023 print, the S&P 500 rose **+1.91%** while the Russell 2000 rose **+5.44%** — nearly 3× the move, from the same news. Compare two **\$20,000** positions held into that print.

- S&P position P&L = \$20,000 × 0.0191 = **+\$382**.
- Russell 2000 position P&L = \$20,000 × 0.0544 = **+\$1,088**.
- The extra **+\$706** on the small-cap position is the reward for taking the more rate-sensitive expression of the same correct call.
- The catch: had the print been hot, the Russell would have fallen ~3× as hard too — that \$706 of extra upside is \$706+ of extra downside risk.

Intuition: choosing *which* asset to express a CPI view in is itself a sizing decision — rate-sensitivity is leverage on the surprise, in both directions.

### The Vietnam angle: the same engine, a different transmission

For readers watching the VN-Index rather than the S&P 500, the CPI engine still runs — it just reaches you through two extra links. First, **US CPI moves the dollar and US yields**, and a stronger dollar pressures the dong and the State Bank of Vietnam's (SBV) reserves; when US inflation surprised hot through 2022, the SBV had to *raise* its refinancing rate (from 4.0% toward 6.0% that autumn) partly to defend the currency, which tightened domestic liquidity and weighed on the VN-Index as it fell toward its ~911 trough in November 2022. Second, **foreign flows**: global funds rotate out of emerging-market equities like Vietnam's when US rates rise (US cash suddenly pays more, so why hold riskier EM stocks?), and that foreign selling — net outflows on the order of tens of trillions of dong in heavy years — is a direct, observable channel from a US CPI print to HOSE. Vietnam also publishes its *own* CPI (the GSO's, averaging ~3% in recent years), which the SBV watches for domestic policy — but for a VN equity trader, the bigger same-day mover is often the *US* print, because it sets the global rate and dollar backdrop that VN flows ride on. (The SBV mechanism is covered in the macro-trading and finance series; the foreign-flow channel sits in the vietnam-stocks track.)

## Common misconceptions

**"Headline CPI is what matters."** No — core and supercore drive the Fed, and the Fed drives markets. A hot headline that's all gasoline (a 7% weight that reverses) can coincide with a market *rally* if core is soft, because the Fed looks through energy. On 13 September 2022 the headline was actually *down* from the prior month in level terms, yet the market crashed — because *core re-accelerated*. Always read core MoM before reacting to the headline.

**"If inflation is already 8%, another high print shouldn't matter — it's priced in."** The *level* is priced in; the *surprise* is not. The market had priced 8.1% in September 2022; the 8.3% print, a mere 0.2pp miss, erased −4.32% from the S&P precisely because the small deviation forced everyone positioned for disinflation to reprice the Fed path at once. Markets trade the gap between actual and consensus, never the level alone.

**"A falling YoY number means inflation is beaten."** Not necessarily — YoY is a trailing twelve-month average dominated by base effects and the lagging shelter component. A YoY that's falling purely because a high month rolled off the back of the window can mask a *re-accelerating* MoM run-rate. Through 2024, headline YoY hovered near 3% while traders watched MoM core and supercore for the real-time direction. Read MoM for the run-rate; read YoY for the trailing story.

**"Shelter inflation tells me what rents are doing now."** It tells you what rents did up to a year ago. The shelter index lags the live rental market by roughly 9–12 months because of annual leases and twice-a-year sampling. When the index is still printing hot shelter, real-time rents may already have cooled — which is exactly why core stayed sticky into 2024 even as the actual rental market had turned. The official number is a rear-view mirror on housing.

**"Bitcoin and gold are inflation hedges, so they rise when CPI is hot."** Over the very long run, maybe. On the *day* of a hot CPI print, both typically *fall*, because the immediate effect is higher real yields and a stronger dollar, which hurt non-yielding assets. Bitcoin fell −9.4% on the hot September 2022 print; gold dipped too. The same-day reaction is a rates-and-dollar story, not an inflation-hedge story.

## The playbook: reading a CPI print like a trader

This post is the anatomy; the full position-by-position trading plan lives in a companion piece in this series. But here is how to *read* a release in real time, which is the prerequisite for trading it.

**Before the release (the setup).** Write down the consensus for headline MoM, core MoM, and (if available) supercore and shelter. Note the regime: is the market afraid of inflation (good-news-is-bad) or of recession (good-news-is-good)? Note what's *priced* into the rate path — the market may already expect a hot print, in which case only an *even hotter* one is a surprise. Know the expected move from options if you can (covered in the series' expected-move post).

**The first ten seconds (the read).** Glance at the headline, then immediately read **core MoM vs consensus** — this overrides the headline. Then check **supercore**: a hot supercore on a soft headline is hawkish; a cool supercore on a hot headline is dovish. Then ask whether **shelter** is finally rolling over. Grade the print:

- **HOT** (core above consensus, supercore hot): in an inflation regime, expect stocks down, yields up, dollar up, gold and crypto down. The knee-jerk is usually the right *direction*; the question is whether it trends or fades.
- **COOL** (core below consensus): stocks up, yields down, dollar down, rate-sensitive small caps and crypto up hardest.
- **INLINE** (matches consensus): small relief move, then a fade — the event passed without forcing a reprice, and the implied volatility that was priced into options collapses (the "vol crush").

**The microstructure (the move).** The first move is the **knee-jerk** — algorithms and fast money reacting to the headline tenth. Then comes either the **fade** (if the internals contradict the headline, or if positioning was already extreme) or the **trend** (if the internals confirm and the surprise is large enough to genuinely move the rate path). The reaction is the trade, not the number. A hot print that *fades* by lunchtime — because core was actually fine and only gasoline was hot — is a different trade than a hot print that *trends* all day because supercore re-accelerated.

**The invalidation.** Your read is wrong if the internals contradict your grade — e.g. you called it hot off the headline but core and supercore were soft, and the market reverses. Size the position for the expected move, not the move you hope for, and respect that the *same* print can flip sign if the regime is misread. The discipline is: grade the print off core and supercore, decide hot/inline/cool, position for the regime's sign, and define the level that says you got the regime wrong.

A few hard-won notes on the mechanics of the move itself. The **knee-jerk is fast and often headline-driven** — algorithms parse the wire flash in milliseconds and trade the headline tenth before a human has read the contributions table. That means the first 30–60 seconds can move *against* the eventual direction if the headline and internals disagree (headline cool, core hot, for instance). Patient desks let the knee-jerk happen and trade the *correction* once the internals are digested — but that requires conviction in your read of the internals, because sometimes the knee-jerk is right and the "fade" never comes. The skill is distinguishing a print where the headline misleads (fade it) from one where the headline and core agree and the move just trends.

Second, **the volatility was priced in, and it evaporates.** In the days before CPI, options on the S&P and on Bitcoin carry an inflated implied volatility because the market knows a binary event is coming. The instant the number prints and the uncertainty resolves — even on a big directional move — that event premium collapses (the "vol crush"). This is why simply *buying volatility* into CPI is usually a losing trade: you pay for the event premium and it disappears at 8:30:01 regardless of which way the index goes. The expected-move and vol-crush mechanics get a dedicated post in this series; the takeaway here is that *how much* the market is braced for a surprise is itself information, and it's encoded in options prices before the release.

Third, **position around the consensus, not your forecast.** It does not matter whether *you* think inflation is high; it matters whether the print will beat or miss the *consensus the market has priced*. If everyone already expects a hot print and it comes in merely hot-as-expected, the market can rally on relief — the feared even-hotter number didn't show. Trading CPI is trading the second derivative: the surprise versus expectations, filtered through the regime, expressed in the most rate-sensitive asset that fits your conviction and risk budget.

For Vietnamese readers, the same logic applies to local CPI and the State Bank of Vietnam, though the transmission runs partly through foreign flows and the dong; the macro-trading series covers the SBV mechanism, and a dedicated VN event post sits alongside this one.

## Further reading and cross-links

- [Why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) — the founding idea this post rests on: markets trade the surprise, not the level.
- [Cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market) — the channels by which a single CPI number reprices stocks, bonds, FX, gold and crypto at once.
- [Inflation and the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) — the mechanism linking a hot core print to a higher rate path and the dot plot.
- [Real vs nominal inflation and real yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — why the real-yield reaction to CPI is the master signal for gold and crypto.
- [The macro calendar: CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) — where CPI sits in the rhythm of the month and how it interacts with the jobs report and the Fed.

A sibling post in this series builds the full CPI trading playbook — the scenario-by-scenario, asset-by-asset positioning around the release — on top of the anatomy you now understand.
