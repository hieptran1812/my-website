---
title: "PPI: The Upstream Inflation Correlation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Producer prices sit one rung up the price ladder from consumer prices. This post builds the commodities-to-PPI-to-CPI-to-PCE chain from zero, measures how strongly PPI and CPI co-move, and shows why PPI's own market reaction is muted but matters most as a read-through around the CPI print."
tags: ["macro", "correlation", "ppi", "cpi", "pce", "inflation", "producer-prices", "pass-through", "margins", "nowcasting"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — PPI (producer prices) sits *upstream* of CPI and PCE: it is the price firms receive at the factory gate, so it sees a cost shock before the shopper does. PPI and CPI co-move strongly (annual r ≈ 0.90 over 2020–2025), but the pass-through slope is *below one* (about 0.7pp of CPI per 1pp of PPI) because services, rent, imports, and corporate margins absorb part of every shock. The correlation is real but partial, the lead is short (about one month into core goods CPI), and PPI's own market reaction is smaller than CPI's.
>
> - **The number to remember:** PPI final demand peaked at **11.7% year-over-year in March 2022**, well above CPI's 9.06% peak that June — producers felt the shock first and harder.
> - **PPI leads core *goods* CPI by ~1 month**, not a quarter. It is a near-coincident pre-tell, not a long-horizon forecast. The big business-cycle leads (the yield curve, ISM) are measured in quarters; PPI's is measured in weeks.
> - **Desks nowcast core PCE off PPI.** Several PCE components — health-care services, financial services, portfolio management, air transport — are built directly from PPI source data, not from CPI. A PPI surprise mechanically nudges the PCE estimate even before PCE is released.
> - **PPI's market beta is muted.** A PPI surprise moves rates, stocks, and the dollar the *same direction* as a CPI surprise but with a much smaller move, because it is a noisier and only-partial read on the consumer inflation the Fed targets.

In the spring of 2022, the most frightening inflation number in America was not the one that made the front page. The headline Consumer Price Index hit 8.5% year-over-year in March 2022 and that was the figure on every screen. But a day earlier, a quieter release had landed: the Producer Price Index for final demand had jumped **11.7%** year-over-year — the hottest in the history of that series. Producers, the firms that make and ship physical goods, were paying and charging far more than consumers were yet feeling. The shock was already in the system; it simply had not finished traveling down the price ladder to the checkout aisle.

Traders who understood the *order* of that ladder had an edge. They knew that the 11.7% producer print was not a separate story from the 8.5% consumer print — it was the *same* inflation, observed one rung higher up, where it arrives first. They knew that some of that producer heat would pass through to consumer goods over the following months, and some of it would get absorbed by fattened or thinned corporate margins instead. And they knew the Federal Reserve does not target either PPI or CPI directly — it targets a third gauge, core PCE, several of whose ingredients are lifted *straight out of the PPI report*. The producer print was, quietly, a read-through to the one number the central bank actually watches.

This post is about that ladder. It is about the empirical correlation between producer prices and consumer prices — its sign, its strength, its lead, and the regimes where it breaks down. Producer-price inflation and consumer-price inflation are two measurements of the same underlying force, taken at different points in the supply chain, and the relationship between them is one of the cleanest and most useful in all of macro. But it is also subtle: the correlation is strong yet partial, the lead is real yet short, and the market reaction is genuine yet muted. Getting all four of those qualifiers right is what separates someone who trades PPI from someone who just reads the headline.

![Upstream inflation chain from commodities to PPI to core goods CPI to PCE with market reaction at each stage](/imgs/blogs/ppi-the-upstream-inflation-correlation-1.png)

If you have not yet read [CPI and asset prices: the master inflation correlation](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation), start there — it is the anchor of this whole inflation track and it builds the CPI-to-rates-to-everything reaction function that PPI feeds into. This post assumes you know that a hot inflation print is bearish for both stocks and bonds in the inflation-fear regime, and adds the missing upstream link: where that inflation comes from before it reaches the consumer basket.

## Foundations: what PPI actually measures

Let us build the producer price index from absolutely nothing, because almost everyone's first mental model of it is subtly wrong.

The Consumer Price Index, CPI, measures the prices that *households* pay. A statistician fixes a representative shopping basket — groceries, rent, a haircut, a tank of gas, a streaming subscription — and tracks what that exact basket costs month after month. CPI answers the question: how much more expensive has it gotten to *be a consumer*?

The Producer Price Index, PPI, measures something different. It tracks the prices that *domestic producers receive* for their output — the price a steel mill gets for a ton of steel, the price a bakery gets when it sells bread to a grocer, the price a software firm charges another business for a license. PPI answers the question: how much more is it costing the *economy's sellers* to do business, as captured in the prices they charge?

The single most important word in that definition is **receive**. CPI is a *paid* price; PPI is a *received* price. The same transaction has two sides, and the gap between them — the markup, the margin, the layers of distribution and tax — is exactly where the correlation between the two indices leaks. A baker's flour cost (a producer input) can spike without the price of a loaf at the supermarket (a consumer good) moving much at all, if the baker eats the cost in a thinner margin. That gap is not a flaw in the data; it *is* the story.

### Final demand and the stages of processing

Modern PPI is published as **PPI for final demand**, and to read it you have to understand that production happens in stages. Raw materials become intermediate goods become finished products. The Bureau of Labor Statistics, which produces the index, organizes prices along this chain:

- **Crude (or "stage 1") prices** are the rawest inputs: crude oil, iron ore, raw cotton, slaughter livestock. These are the most volatile, because they sit closest to commodity markets where prices swing on supply, weather, and geopolitics.
- **Intermediate prices** are partly-processed goods: refined fuel, steel sheet, milled flour, lumber. A cost shock that began in crude materials shows up here next.
- **Final-demand prices** are the goods and services sold to their *final* purchaser — a finished car, a restaurant meal sold to a diner, a consulting engagement billed to a client. This is the headline PPI that markets watch, and it is meant to be roughly comparable in spirit to CPI: the price of finished output.

The headline "PPI" you see quoted is **PPI final demand**. It is itself split into final-demand *goods*, final-demand *services*, and final-demand *construction*. The goods piece is the one that lines up most directly with consumer goods, and as we will see, it is the part of PPI that leads CPI most cleanly.

The crucial structural point is that a cost shock travels *up* the stages-of-processing ladder over time: crude oil rises this month, refined diesel rises next month, the trucking-and-logistics line of final-demand services rises the month after, and the price of a delivered finished good rises after that. PPI captures different rungs of that ladder; CPI captures only the very top rung, the finished consumer good. The lead of PPI over CPI is, mechanically, the time it takes a shock to climb from the producer's gate to the shopper's receipt.

> [!note]
> A common confusion: PPI is *not* a wholesale price index in the old sense, and it deliberately *excludes imports*. PPI measures prices received by **domestic** producers. So a surge in the price of imported electronics or imported oil shows up in CPI (consumers pay it) but not directly in PPI (no domestic producer received it). This is one of the structural reasons the two indices diverge — and it is why a strong dollar, which cheapens imports, can pull CPI and PPI apart. We will return to this when we discuss why the correlation breaks.

### How PPI is actually collected (and why it is jumpy)

It helps to know where the number comes from, because the collection method explains the volatility. Every month the Bureau of Labor Statistics surveys tens of thousands of establishments and asks a deceptively simple question: what price did you *actually* receive, on a specified day, for a specified good or service, net of discounts and rebates? It is a *transaction* price, not a list price, which is why PPI can move even when no sticker changes — a quietly negotiated discount or a surcharge is a real change in the price received.

Two features of that collection make PPI noisier than CPI. First, PPI includes a line called **trade services**, which does not measure a good's price at all — it measures the *margins* of wholesalers and retailers (the difference between what they buy for and sell for). When retail margins swing, the trade-services line swings, and it drags the PPI headline around for reasons that have nothing to do with goods inflation reaching consumers. Second, PPI includes volatile business-service lines like **portfolio management**, whose "price" is the fee charged on assets under management and therefore tracks the stock market. A 5% rally in equities mechanically lifts that PPI line, again for reasons unrelated to consumer prices. These two quirks — a margin proxy and a market-linked fee proxy living inside the producer-price index — are a big part of why the market discounts PPI surprises relative to CPI: the index has structural noise baked into its construction.

The teaching point is that PPI is *closer to the raw economy* than CPI in a literal sense. It samples the messy transaction prices and margins of the production side directly, rather than the smoothed, shelter-anchored basket a household faces. That rawness is exactly what gives PPI its upstream lead — and exactly what gives it its noise.

### Why "upstream" is the whole point

The word that organizes everything in this post is **upstream**. A river flows from its source to the sea; a price shock flows from raw inputs to the consumer. PPI sits closer to the source. CPI sits closer to the mouth. PCE — the Fed's gauge — sits further still, downstream even of CPI in some of its construction. Watching PPI is like standing at a gauge halfway up the river: when the water rises there, you have a head start on the people downstream.

This single piece of geography explains the sign of the correlation (positive — a producer-price shock pushes consumer prices the same way), the lead (PPI first, CPI later), and the partial strength (water leaks out of the river through margins and services before it reaches the sea). Hold the river picture, and the rest of the post is detail.

It also explains a subtlety that trips up newcomers: causality runs *mostly* downstream, but not entirely. Most of the time the producer-price shock causes the consumer-price move that follows. But there are feedback loops. A consumer-spending boom (a demand shock that shows up in CPI) pulls *up* the prices producers can charge, so CPI strength can feed *back* into PPI rather than the other way around. And both indices are driven by common upstream forces — energy, wages, the exchange rate — so part of their co-movement is not one causing the other at all, but both responding to the same third thing. This matters for trading because it means a high PPI-to-CPI correlation does not prove PPI *causes* CPI; it proves they share drivers and a sequence. When you treat PPI as a leading *indicator*, you are betting on the sequence holding, not on a mechanical law of causation. In hot booms the sequence is reliable; in unusual regimes it can break, which is the entire reason the lead is something to monitor rather than to trust blindly. Keep the river, but remember the river sometimes runs backward and is sometimes fed by rain on both banks at once.

The figure above lays out the full chain: commodities feed PPI, PPI feeds core goods CPI, core goods CPI feeds into PCE, and the market reacts at every stage — but the reaction *shrinks* as you move downstream, because by the time the shock reaches PCE, most of it has already been priced from the commodity and PPI prints that came before.

#### Worked example: a shock entering the top of the chain

Trace a single commodity shock down the ladder to see why the reaction shrinks. Suppose copper, a key industrial input, jumps from \$4.00 to \$4.80 per pound — a 20% move that trades live on the COMEX exchange the instant it happens.

- **Commodity stage.** The \$0.80 move is fully visible to markets *now*, in real time. Anyone tracking copper sees the entire shock immediately. The market reaction is large and immediate, but it is *to copper*, not to any inflation report.
- **PPI stage.** Weeks later, the copper move shows up in the intermediate-goods PPI for copper products, and then in the final-demand PPI for finished metal goods — but diluted, because copper is only a fraction of a finished good's cost, and because the producer may have hedged or absorbed part of it. Maybe \$0.20 of the \$0.80 reaches the producer-price index for the affected goods. By now the market has *already* seen the copper move, so the PPI print is only a partial surprise.
- **CPI stage.** A month or two after that, the move reaches consumer goods that contain copper — appliances, electronics, wiring — diluted again by retail margins and by the small weight of those goods in the consumer basket. Perhaps \$0.10 of the original \$0.80 lands in CPI. The market reaction is now small, because two prior prints (the commodity and the PPI) already told most of the story.
- **PCE stage.** Smaller still, because PCE's weights and its services tilt further dilute the goods shock.

The same \$0.80 copper move arrives as a large live commodity move, a partial PPI surprise, a small CPI surprise, and a tiny PCE nudge. **The reaction shrinks downstream not because the shock weakens but because each stage tells the market something it already half-knew from the stage above.** This is the deepest reason PPI's standalone market reaction is muted: it is sandwiched between the commodity prices that lead it and the CPI print that the market cares about more.

## How PPI differs from CPI: scope, weights, and timing

To trade the correlation you have to know exactly *why* the two indices are not the same number. There are three structural differences, and each one is a reason the correlation is less than perfect.

![PPI versus CPI compared on whose price coverage services weight and timing](/imgs/blogs/ppi-the-upstream-inflation-correlation-6.png)

**Whose price.** As established: PPI is the price the seller *receives*; CPI is the price the buyer *pays*. Between those two prices sit the wholesaler, the retailer, the trucker, sales tax, and the retailer's margin. When any of those layers expands or compresses, the producer price and the consumer price drift apart even though they describe the same good.

**What is in the basket.** CPI includes consumer services that have no PPI counterpart in the goods piece — most importantly **shelter** (the cost of housing, which is roughly a third of the CPI basket and is famously sticky and slow-moving). PPI's goods component has no rent in it. CPI also includes the price of *imported* finished goods; PPI, by design, does not. Conversely, PPI includes prices of business-to-business services and intermediate goods that never appear in a consumer basket. The two baskets overlap heavily in physical consumer goods but diverge sharply everywhere else.

**The weights.** This is the difference that matters most for the correlation's strength. The modern US economy is a *services* economy, and CPI reflects that: services (including shelter) are roughly 60% of the CPI basket. PPI final demand is more goods-weighted, and the *goods* piece is where commodity shocks land first. So when oil spikes, PPI — being goods-heavy — moves a lot, while CPI — being services-heavy and anchored by sticky rent — moves less. The same shock produces a bigger wiggle in PPI than in CPI. That is not a measurement error; it is the two indices honestly reporting their different baskets.

**The timing.** Here is the practical gift. PPI and CPI are released within about a day of each other every month — typically PPI the day before or the day after CPI. That near-simultaneity is what makes PPI a *read-through*: the PPI print is the first hard inflation data of the month, and traders read it as a pre-tell for the CPI number that lands hours later (or, when PPI comes second, as a confirmation of the CPI move that just happened). The day-before / day-after timing is the single most actionable fact in this entire post.

#### Worked example: why the same oil shock moves PPI more than CPI

Suppose crude oil rises 20% in a month. Energy is roughly 12% of the PPI final-demand goods basket but, after you account for services and shelter, energy is only about 7% of the CPI basket. Hold everything else fixed and pass the shock straight through:

- PPI impact ≈ 20% × 12% = **+2.4pp** to producer-goods inflation that month.
- CPI impact ≈ 20% × 7% = **+1.4pp** to consumer inflation that month.

So the identical commodity shock shows up as a \$2.40-per-\$100 jump in producer prices but only a \$1.40-per-\$100 jump in consumer prices, purely because of the different basket weights. **A larger PPI move than CPI move for the same shock is the normal state of the world, not a divergence to fear** — it is the goods-heavy index honestly reflecting that it is more exposed to commodities than the services-heavy one. (Real pass-through is messier than this — margins and lags blur it — but the weight arithmetic is the first-order reason PPI is the more volatile of the two.)

## The empirical correlation: strong, positive, and partial

Now the numbers. We have two annual series — PPI final demand year-over-year and CPI year-over-year — for 2020 through 2025, spanning the calmest deflationary scare (2020), the violent inflation surge (2021–2022), and the disinflation that followed. Plotting them together shows the core of the relationship.

![PPI and CPI year over year inflation both surge in 2021 to 2022 with PPI overshooting](/imgs/blogs/ppi-the-upstream-inflation-correlation-2.png)

The two lines move together. Both were near 1% in the deflationary scare of 2020, both ramped through 2021, both peaked in 2022, and both came down hard in 2023 before settling near the 2% goal. The co-movement is obvious to the eye — but two features are worth circling.

First, **PPI overshoots**. In the surge, PPI ran hotter than CPI (8.0% versus 7.95% on annual averages, and the monthly peaks were 11.7% PPI versus 9.06% CPI). Producer prices, being goods-heavy and upstream, amplified the shock. Second, **PPI undershoots on the way down**. In 2023 PPI collapsed to 1.9% while CPI was still 3.75%, because the services and shelter components of CPI stayed sticky long after goods prices had deflated. The producer index leads in *both* directions: it rises faster and falls faster than the consumer index, because it lacks the sticky services anchor that holds CPI up.

### Measuring it: the scatter and the regression line

The eye says "they move together." Statistics says how much. The signature chart of this whole series is the scatter plot — one variable on each axis, one dot per observation, a fitted line through the cloud — and it is the right tool here. Put PPI on the x-axis and CPI on the y-axis, one dot per year.

![Scatter of PPI versus CPI with regression line showing correlation of 0.90 and slope 0.72](/imgs/blogs/ppi-the-upstream-inflation-correlation-3.png)

The dots line up tightly along an upward-sloping line. The Pearson correlation coefficient — the standard measure of how tightly two series move together, running from −1 (perfect opposite) through 0 (no linear relationship) to +1 (perfect lockstep) — comes out to **r ≈ 0.90** on this annual sample. That is a strong, clean positive correlation. (See [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) for the full definition of r, beta, and what a coefficient of 0.9 does and does not promise.)

But the correlation coefficient is only half the picture. The *slope* of the fitted line is the other half, and it carries the economic content. The regression says:

> CPI ≈ 0.72 × PPI + 1.19

The slope of **0.72** is the empirical pass-through: on this sample, each additional percentage point of producer-price inflation is associated with about 0.72 of a percentage point of consumer-price inflation. The slope is *below one*. Producer-price shocks pass through to consumer prices, but they pass through *attenuated* — roughly 70 cents of every producer-price dollar reaches the consumer, and the other 30 cents gets absorbed by margins, services, and the structural differences between the baskets.

#### Worked example: reading r and the slope correctly

A reader sees "r = 0.90" and "slope = 0.72" and might think they disagree — isn't 0.90 nearly perfect while 0.72 is clearly less than one? They measure different things, and both are right.

- The **correlation r = 0.90** says the two series move together *reliably*: when PPI is high relative to its average, CPI is almost always high relative to its average too. It is about the *consistency* of the co-movement, scaled out of any units. r² ≈ 0.81, so about 81% of the year-to-year variation in CPI inflation lines up with PPI inflation.
- The **slope = 0.72** says the two series move together at a *muted ratio*. For every 1.0pp PPI rises, CPI tends to rise only 0.72pp. It is about the *size* of the response, in real units.

You can have a high r with a low slope (reliable but attenuated — exactly our case) or a low r with a high slope (a big-but-noisy response). For trading you need both numbers: r tells you whether to *trust* the signal, the slope tells you how much to *scale* it. **A \$1.00 producer-price shock reliably becomes a \$0.72 consumer-price shock — reliably, but only seventy-two cents on the dollar.**

### The pass-through chain, mechanically

Why exactly 0.72 and not 1.0? Because the producer-to-consumer journey loses water at every step. Trace it from the commodity to the checkout, and decide where each shock can be diverted.

![Where a PPI shock goes branching into consumer prices or into squeezed equity margins](/imgs/blogs/ppi-the-upstream-inflation-correlation-7.png)

When PPI rises — input costs climb because oil, metals, or freight got more expensive — a producer faces a choice. It can **pass the cost on** by raising its output prices, which pushes core goods CPI up and, eventually, feeds PCE; or it can **absorb the cost** by holding its output prices and eating a thinner margin per unit. In reality every firm does some of both, and the split between "pass on" and "absorb" depends on how much pricing power the firm has — how willing its customers are to tolerate a price increase before they walk away.

In a hot-demand environment (2021–2022), firms had enormous pricing power: customers were flush, supply was scarce, and nearly the whole shock got passed on. Pass-through was high, and CPI followed PPI up tightly. In a soft-demand environment, firms cannot raise prices without losing customers, so more of the shock gets absorbed into margins. Pass-through is low, and CPI barely moves while corporate profits get squeezed. **The pass-through ratio is not a constant; it is a function of pricing power, which is a function of the demand regime.** That is the deepest reason the PPI-to-CPI correlation is regime-dependent rather than fixed.

The two branches of that decision have very different market consequences, and that is where the margin angle becomes a trade — more on that below.

## The lead/lag: PPI is a short-lead pre-tell, not a forecast

Every correlation in this series has a *clock* — does the indicator move before the asset, with it, or after it? (See [lead, lag, or coincident](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) for the full machinery of the cross-correlation function.) For PPI, the honest answer is a useful disappointment: the lead is real, but it is *short*.

![Lead times of inflation chain pairs with PPI leading core goods CPI by about one month](/imgs/blogs/ppi-the-upstream-inflation-correlation-4.png)

When you compute the cross-correlation between PPI and core *goods* CPI — sliding one series past the other to find the lag at which they line up best — the peak sits at about **one month**. PPI leads core goods CPI by roughly a single month. That is far shorter than the headline business-cycle leads: the yield curve leads recessions by about 14 months, ISM new orders lead earnings by about 6 months, but PPI leads its slice of CPI by *weeks*, not quarters.

Two things follow from that short lead, and they pull in opposite directions:

- **It is not a long-horizon forecast.** You cannot look at today's PPI and confidently call CPI two quarters out. The chain from producer gate to consumer checkout for physical goods is short, so PPI's predictive horizon is short too. Anyone selling PPI as a crystal ball for next year's inflation is overpromising.
- **It is a genuine same-week edge.** Because PPI is released a day before CPI in most months, that one-month statistical lead becomes a *one-day* informational lead in practice. The PPI print is real, fresh data about the same inflation that CPI will report hours later. A hot PPI goods number is a real (if noisy) reason to lean toward a hot CPI print the next morning.

#### Worked example: from a PPI surprise to a CPI lean

Suppose PPI final-demand goods comes in at +0.5% month-over-month against a +0.2% expectation — a +0.3pp upside surprise concentrated in goods. How much should that move your expectation for tomorrow's CPI?

Core goods are a much smaller slice of CPI than of PPI (consumer goods ex-food-and-energy are roughly 20% of CPI, versus the goods-heavy PPI), and the pass-through is partial and lagged. A rough same-month read-through might be: a +0.3pp PPI *goods* surprise, applying a partial one-month pass-through of perhaps 0.3–0.4 and the smaller CPI goods weight, nudges your CPI core-goods expectation up by only a few hundredths of a point — call it +0.03 to +0.05pp on *core goods*, which is a rounding error against a core CPI that prints to one decimal. **The honest conclusion: a hot PPI goods print tilts the odds toward a hot CPI, but it rarely moves your CPI point estimate by even a tenth.** It is a thumb on the scale, not a forecast. Treat it as Bayesian evidence that shifts a probability, not as an arithmetic prediction of the next print.

### Why the lead is short, mechanically

The lead is short because the *physical-goods* portion of the supply chain is short. A finished consumer good — a toaster, a t-shirt, a box of cereal — moves from the factory gate to the store shelf in weeks, and its producer price and its consumer price are measured close together in time. The long leads in macro come from chains with many slow steps (a permit becomes a building over a year; an inverted yield curve becomes a recession over many quarters). PPI-to-goods-CPI is a short chain, so it is a short lead. The services and shelter parts of CPI have their *own*, much longer and stickier dynamics that PPI does not lead at all — which is exactly why PPI leads core *goods* CPI cleanly but does not lead *headline* CPI by much.

## The PCE read-through: why desks nowcast PCE off PPI

Here is the part most retail traders miss entirely, and it is the most important institutional reason PPI matters. The Federal Reserve does not target CPI. It targets **PCE** — Personal Consumption Expenditures inflation, and specifically core PCE (excluding food and energy). When Fed officials say "2% inflation," they mean 2% on the PCE deflator. (For the full PCE story and how breakevens price the *forward* path of inflation, see [PCE, breakevens, and the forward inflation correlation](/blog/trading/macro-correlations/pce-breakevens-and-the-forward-inflation-correlation).)

PCE is constructed differently from CPI. It has different weights (less shelter, more health care), it captures spending paid on a consumer's behalf (employer-provided and government-provided health care, which CPI mostly misses), and — critically for us — **several of its components are sourced directly from PPI, not from CPI.** When the Bureau of Economic Analysis builds the PCE price index, it uses PPI as the source price for categories where PPI measures the transaction better than CPI does: portfolio management and investment-advice fees, much of health-care services, air transportation, financial services, and others. Estimates vary, but a meaningful chunk of the core PCE basket — on the order of a quarter — is priced off PPI source data.

This is why the trading desks of every major bank build a **core PCE nowcast** that updates the *moment* the PPI report lands. They take the just-released CPI components, swap in the relevant just-released PPI components for the PCE-specific categories, apply the PCE weights, and produce an estimate of core PCE that is available days or weeks before the official PCE release. The PPI report is not just a pre-tell for CPI — it is a *direct input* into the number the Fed actually watches.

#### Worked example: a PPI surprise nudging the core-PCE nowcast

Suppose the PPI report shows the **portfolio-management and investment-advice** services line jumped +1.5% month-over-month — a category that feeds PCE directly and is notoriously volatile (it tracks the stock market, since fees are charged as a percentage of assets). That line is roughly 1.5% of core PCE by weight. The mechanical contribution to the monthly core-PCE nowcast is:

- contribution ≈ +1.5% × 1.5% weight = **+0.0225pp** to monthly core PCE.

That is more than two hundredths of a point on core PCE from a *single* PPI services line, before any other component moves. Annualized, a +0.02pp monthly surprise is roughly +0.27pp on the annual core-PCE pace — not trivial when the Fed is parsing whether core PCE is running at a 2.6% or a 2.9% pace. **This is the concrete mechanism behind "desks revise their PCE estimate off PPI": the PPI report literally contains the source prices for parts of the Fed's target gauge, so a PPI surprise updates the PCE estimate by arithmetic, not by analogy.** When you hear a strategist say "today's PPI raised our core-PCE tracking estimate by three basis points," this is exactly the calculation they ran.

### Why PPI's own market reaction is muted

Given all that, you would expect PPI to detonate markets the way CPI does. It does not — and the reason is instructive. The asset-price reaction to a PPI surprise is in the *same direction* as the reaction to a CPI surprise (hot PPI → yields up, stocks down, dollar up) but with a noticeably *smaller* beta.

![PPI surprise betas are smaller than CPI surprise betas across stocks bonds dollar and gold](/imgs/blogs/ppi-the-upstream-inflation-correlation-5.png)

The chart anchors on the documented CPI surprise betas — how far each asset moves, in the same session, per +0.1pp of core-CPI upside surprise in the 2022–2023 inflation-fear regime — and shows PPI's reaction as a muted fraction of those. The signs match exactly (an inflation surprise from either index is hawkish), but PPI's bars are roughly a third the height of CPI's. There are three reasons:

1. **PPI is a partial read on the consumer inflation the Fed targets.** It is goods-heavy and excludes shelter and imports — the very components that dominate CPI and PCE. So a PPI surprise contains *less information* about the number the Fed cares about than a CPI surprise does.
2. **PPI is noisier and gets revised.** Volatile lines like trade services (which is essentially a margin measure) and portfolio management swing the headline around, and the market discounts a print it knows is jumpy.
3. **Much of PPI is already priced.** PPI is downstream of commodities, and commodity prices trade in real time. By the time PPI prints, the market has already seen the oil and metals moves that drive much of it, so the PPI report contains less *surprise* than CPI, which has the sticky, hard-to-forecast services and shelter components that genuinely catch the market off guard.

The practical upshot: trade PPI as a *probability-shifter for CPI* and a *PCE-nowcast input*, not as a standalone market-mover. Its own reaction is a faint echo of CPI's. Its value is informational, not directional in its own right.

## When the correlation flips: the regime-dependence of pass-through

The headline r ≈ 0.90 is a *full-sample* number, and full-sample numbers are the great liars of macro. The whole premise of this series is that **a correlation is a regime, not a constant** (see [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant)). The PPI-to-CPI link is no exception: its strength and its slope drift with the demand environment, and understanding *why* is what lets you anticipate when the relationship will tighten or loosen rather than being surprised by it.

The hinge is **pricing power**. Recall the firm's choice when its costs rise: pass it on, or absorb it. The fraction passed on *is* the pass-through slope, and that fraction depends on how much the firm's customers will tolerate. So the PPI-to-CPI slope is high exactly when demand is hot and customers are price-insensitive, and low when demand is soft and customers will walk over a price increase. The correlation is therefore *pro-cyclical in strength*: it tightens in booms and slackens in slowdowns.

This produces a few distinct regimes worth naming:

- **Hot-demand inflation (2021–2022).** Customers flush, supply scarce, pricing power maximal. Nearly the whole PPI shock passes into CPI. The slope approaches one, the correlation is tight, and a hot PPI is a genuine pre-tell of a hot CPI. This is the regime that produced the 0.90 / 0.72 numbers — and it flatters them.
- **Soft-demand cost shock (a "margin recession").** Costs rise (PPI up) but firms cannot raise prices without losing share, so they absorb the shock. The slope collapses, the PPI-to-CPI correlation weakens, and the *same* hot PPI that would have meant hot CPI in a boom now means compressed margins and weak earnings instead. The signal does not vanish — it *changes asset class*, from a rates/inflation trade to an equity-margins trade.
- **Disinflation (2023).** Goods PPI falls fast, but sticky services and shelter hold CPI up. PPI undershoots CPI, the spread goes negative, and the lead reverses in usefulness: PPI tells you goods-disinflation is coming to CPI, while CPI's own stickiness tells you headline inflation will lag.
- **Supply-driven vs demand-driven shocks.** A purely *supply-side* shock (an oil embargo, a port closure) hits PPI first and hard and passes through with whatever pricing power exists. A *demand-side* shock (a stimulus-fueled spending boom) can lift CPI services and shelter — components PPI does not measure — so CPI can run hot with PPI relatively calm. Diagnosing which kind of shock you are in tells you whether to watch PPI as the leader or to ignore it.

#### Worked example: the same PPI shock in two regimes

Take an identical +5% input-cost shock to a goods producer and run it through the two regimes, using a hot-demand pass-through of 0.9 and a soft-demand pass-through of 0.3:

- **Hot-demand regime.** Output price rises 5% × 0.9 = **+4.5%**. On a \$10.00 product with \$7.00 cost: new cost \$7.35, new price \$10.45, new margin \$3.10 — margin *rises* by \$0.10. Here rising PPI is bullish for the producer (it has pricing power) and the shock lands in CPI (+4.5% on this good).
- **Soft-demand regime.** Output price rises 5% × 0.3 = **+1.5%**. New cost \$7.35, new price \$10.15, new margin \$2.80 — margin *falls* by \$0.20. Here the same rising PPI is bearish for the producer (it eats the cost) and barely reaches CPI (+1.5%).

The identical +5% PPI shock produced a +\$0.10 margin and a hot CPI in one regime, and a −\$0.20 margin and a cool CPI in the other. **The shock did not change; the pricing-power regime did — and it flipped the sign of both the equity effect and the CPI read-through.** This is why you must always pair a PPI print with a read on demand before you decide what it means.

## The margin angle: when PPI becomes an equity trade

Return to the decision a firm makes when its costs rise: pass it on, or absorb it. We treated that as the reason pass-through is partial. But the *absorb* branch is itself a tradeable signal, because absorbing a cost shock means a **margin squeeze** — and margins drive equity prices.

When PPI rises faster than the prices firms can charge — when input-cost inflation outruns output-price inflation — corporate margins compress. The firms hit hardest are the ones with the least pricing power and the most input-cost exposure: cyclical, commodity-consuming, competitive industries. Think of a packaged-food company whose grain and packaging costs (PPI inputs) jump while it is reluctant to raise shelf prices for fear of losing share, or an automaker squeezed between steel costs and sticker-price competition. The gap between producer-input inflation and producer-output inflation is, quite literally, a real-time margin gauge for the goods economy.

This is why the PPI report is read by equity desks, not just rates desks. A persistent wedge of "input PPI rising faster than output PPI" is a warning that goods-sector margins are about to compress, which is bearish for exactly the cyclical sectors most exposed to it. (For how and why correlated sectors move together with the macro cycle — and when that co-movement breaks down — see [correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch).)

#### Worked example: a margin squeeze from a PPI-CPI wedge

Take a consumer-goods firm with the following per-unit economics: it sells a product for \$10.00, its cost of goods is \$7.00, so its gross margin is \$3.00 (30%). Now a producer-cost shock raises its input costs by 8% (a PPI input shock), but soft consumer demand means it can only raise its selling price by 3% (the consumer-price pass-through):

- New cost of goods = \$7.00 × 1.08 = **\$7.56**.
- New selling price = \$10.00 × 1.03 = **\$10.30**.
- New gross margin = \$10.30 − \$7.56 = **\$2.74**, or 26.6% of the new price.

The gross margin fell from \$3.00 to \$2.74 — a **−8.7% hit to gross profit per unit** — purely because input PPI (+8%) outran output pass-through (+3%). That \$0.26-per-unit squeeze, multiplied across millions of units, is what shows up as an earnings miss and a stock-price drop. **When PPI inflation runs above the rate at which firms can raise consumer prices, the difference comes out of margins — and a goods company's margin is its stock price's main fuel.** This is the concrete chain from a PPI print to a single-name equity move, and it is why the PPI-CPI *wedge* (not just PPI's level) is a sector signal.

### The wedge as a two-sided indicator

The PPI-minus-CPI spread tells you which way the margin pressure is flowing, and it flips sign across the cycle.

![PPI minus CPI spread showing partial pass through in both directions across years](/imgs/blogs/ppi-the-upstream-inflation-correlation-8.png)

When the spread is **positive** (PPI hotter than CPI, as in 2021–2022), producers face cost inflation faster than they can pass it on — margin pressure is building, and the goods economy is the squeezed party. When the spread is **negative** (PPI cooler than CPI, as in 2023), input costs are deflating faster than consumer prices, which can actually *widen* goods-sector margins as cheaper inputs flow through against still-sticky output prices. The 2023 negative spread is a textbook example: collapsing goods PPI against sticky services-driven CPI handed goods producers a margin tailwind even as consumers kept feeling inflation.

So the same wedge that is bearish for margins on the way up can be bullish on the way down. **A correlation that is partial is not a defect — it is the source of a second, margin-based signal that the perfect-correlation case would not give you.**

## Common misconceptions

Five myths about producer prices, each corrected with a number or a mechanism.

**Myth 1: "PPI is a reliable leading indicator of CPI by several months."** No. PPI leads core *goods* CPI by about **one month**, not a quarter, and it barely leads *headline* CPI at all because headline CPI is dominated by services and shelter that PPI does not touch. The lead is real but short. If you are using last quarter's PPI to forecast this quarter's CPI, you are extrapolating a one-month signal four times too far.

**Myth 2: "A hot PPI means a hot CPI tomorrow, dollar for dollar."** No. The empirical pass-through slope is about **0.72**, and the same-month read-through from a goods-PPI surprise to core-goods CPI is a few hundredths of a point — a thumb on the scale, not an arithmetic prediction. PPI shifts the *probability* of a hot CPI; it does not set its value. Roughly 30 cents of every producer-price dollar never reaches the consumer.

**Myth 3: "PPI doesn't matter for markets because the reaction is small."** No. Its *standalone* reaction is muted (roughly a third of CPI's beta), but it matters for two reasons that do not show up in the same-day price move: it feeds the **core-PCE nowcast** (parts of the Fed's actual target gauge are priced off PPI), and it is the day-before/day-after **read-through** that frames the CPI print. The value is informational, not directional.

**Myth 4: "PPI and CPI measure the same inflation, so they should be the same number."** No. They measure the same *force* but different *baskets*: PPI is goods-heavy, prices received, excludes imports and shelter; CPI is services-heavy, prices paid, includes both. The structural gaps — shelter (~1/3 of CPI, zero in PPI goods), imports, and corporate margins — are precisely why the correlation is r ≈ 0.90 and not 1.0, and why PPI overshoots CPI in surges and undershoots in busts.

**Myth 5: "If PPI is rising, just buy the producers — they have pricing power."** Not necessarily. Rising *input* PPI is only good for a producer if its *output* prices rise at least as fast. When input PPI outruns output pass-through — exactly the soft-demand case — rising PPI is a **margin squeeze**, and it is *bearish* for the most cost-exposed cyclicals. The sign of the equity effect depends on the PPI-CPI wedge, not on PPI's level.

## How it shows up in real markets

Three dated episodes where the producer-to-consumer correlation, the lead, and the muted reaction all showed their hands.

### 2022: the producer print that led the consumer panic

The cleanest illustration of "upstream" was the 2022 surge. PPI final demand peaked at **11.7% year-over-year in March 2022**, three months before CPI peaked at 9.06% in June 2022 and well above it the whole way up. Producers — goods-heavy and sitting on top of an oil price that had spiked after Russia's invasion of Ukraine — felt the shock first and hardest. Anyone watching PPI in late 2021 and early 2022 saw the inflation wave building one rung up the ladder before it crested in the consumer data. The market had largely figured this out, which is *why* the standalone PPI reactions were muted: the producer heat was no surprise by the time it printed, because everyone could see the commodity prices feeding it.

### 2023: the disinflation that PPI called first

On the way down, PPI led again. Goods PPI collapsed through 2023 as supply chains healed and commodities fell, dragging PPI final demand to 1.9% on the annual average while CPI was still 3.75% — held up by sticky shelter and services that PPI does not measure. The PPI-minus-CPI spread went sharply *negative*, the mirror image of 2022. The trade was twofold: it foreshadowed continued goods-disinflation in CPI (the goods piece of CPI did roll over), and it signaled a margin *tailwind* for goods producers whose input costs were falling faster than they were cutting prices — several consumer-goods names posted margin expansion in 2023 for exactly this reason.

### The energy passthrough: where the chain starts

The very top of the upstream chain is energy, and it is the cleanest place to watch the commodity-to-PPI-to-CPI cascade. Energy is a direct PPI input (final-demand energy goods) *and* an indirect input to nearly everything else (freight, plastics, fertilizer, manufacturing). When oil spikes — as it did in early 2022 after Russia's invasion of Ukraine, with WTI averaging \$94.53 for the year against \$68.17 in 2021 — the move shows up in PPI's energy line within the same month, then bleeds into the broader goods PPI as higher freight and input costs over the following months, then reaches CPI energy and core-goods lines after that. Energy is the part of the chain with the *shortest* lead (it is nearly contemporaneous in PPI) and the *fattest* tail (it keeps feeding core goods for months through second-round freight and input effects). This is exactly the dynamic dissected in [oil prices, CPI, and the energy-equity correlation](/blog/trading/macro-correlations/oil-prices-cpi-and-the-energy-equity-correlation) — and PPI is the gauge that catches the energy shock one rung before it reaches the consumer.

### The monthly ritual: PPI as the CPI read-through

Every month, the cleanest live demonstration is the PPI-then-CPI sequence. In a typical month PPI lands the day before CPI. When PPI's goods and PCE-relevant services lines surprise hot, desks immediately mark up both their CPI estimate for the next morning and their core-PCE nowcast, and rates drift in the hawkish direction *into* the CPI print. When PPI surprises cold, the opposite. The PPI report is the warm-up act that sets the crowd's expectations for the CPI headliner — which is precisely the read-through framing developed in the event-trading companion, [PPI: the upstream signal and the CPI read-through](/blog/trading/event-trading/ppi-the-upstream-signal-and-cpi-read-through). That post covers the intraday reaction mechanics; this one covers the correlation structure underneath them.

#### Worked example: the full-sample correlation from the arrays

To make the headline number concrete, compute the correlation directly from the annual series used in the charts above. The PPI final-demand values (2020–2025) are 0.8, 6.5, 8.0, 1.9, 2.4, 2.6 percent; the matching CPI annual averages are 1.28, 4.43, 7.95, 3.75, 2.95, 2.69 percent.

- The means are PPI ≈ 3.7% and CPI ≈ 3.84%.
- The deviations move together: in the two surge years (2021, 2022) both are far above their means; in the calm years both are near or below. There is no year where one is high while the other is low.
- Running Pearson's formula on these six pairs gives **r ≈ 0.90**, and the ordinary-least-squares line is **CPI ≈ 0.72 × PPI + 1.19**.

So on this sample, knowing PPI explains about 81% of the variance in CPI (r² ≈ 0.81), and each extra point of PPI is worth about \$0.72 of CPI. **The single sentence to carry away: producer and consumer inflation move together about as reliably as any pair in macro (r ≈ 0.90), but the consumer side moves only about seventy cents on the producer dollar (slope ≈ 0.72), and the leftover thirty cents is the margin-and-services story.** Be honest about the sample, though: six annual points dominated by one big inflation cycle will *overstate* a correlation; on noisier monthly data with the surge excluded, the relationship is real but considerably looser. Full-sample correlations that lean on a single dramatic episode always flatter the signal.

## How to read it and use it

The playbook for the producer-to-consumer correlation, distilled.

**The signal.** Read PPI as three things, in order of value: (1) a *read-through to CPI* — the day-before/day-after print that frames the consumer number, where a hot goods PPI tilts the odds toward a hot CPI; (2) a *direct input to the core-PCE nowcast* — the PCE-specific lines (portfolio management, health care, air transport, financial services) come straight out of PPI, so a surprise there mechanically moves the Fed's target gauge estimate; and (3) a *margin gauge for the goods economy* via the input-vs-output PPI wedge and the PPI-minus-CPI spread.

**The regime check.** The pass-through is not constant — it rises with pricing power. In a hot-demand regime, expect most of a PPI shock to pass into CPI (high slope, tight correlation, bearish for both bonds and stocks). In a soft-demand regime, expect more of it to be absorbed into margins (low slope, looser correlation, a *margin* trade rather than an *inflation* trade). Check which regime you are in before you decide whether a PPI shock is a CPI story or a margins story.

**What invalidates it.** The correlation breaks when the structural gaps between the baskets dominate: when *shelter* is the story (PPI has no rent, so a shelter-driven CPI move has no PPI counterpart), when *imports* are the story (a strong dollar cheapening imports pulls CPI down without touching domestic PPI — see [the dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity)), or when *margins* are doing all the absorbing (PPI rises but the firm eats it, so CPI does not follow). In any of those cases, a PPI move that "should" have shown up in CPI simply does not — and that non-confirmation is itself information: it tells you the shock is being diverted into shelter, imports, or margins.

**Combining PPI with the rest of the dashboard.** PPI is most powerful when you stop reading it alone and start reading it *against* its neighbors. Three pairings do most of the work. First, **PPI versus commodity prices**: if PPI is rising but the commodities that feed it are flat or falling, the producer-price strength is coming from margins or services, not from input costs — a different and usually stickier kind of inflation. Second, **PPI versus CPI (the wedge)**: a widening positive wedge (PPI hotter) flags building margin pressure on goods producers, while a negative wedge flags a goods-disinflation tailwind, as the 2023 case showed. Third, **PPI's PCE-relevant lines versus the core-PCE nowcast**: the portfolio-management, health-care, air-transport, and financial-services lines are the ones that move the Fed's gauge, so isolate them rather than reacting to the noisy headline. The headline PPI number is the least informative way to use the report; the *composition* — which lines moved, and how they compare to commodities, to CPI, and to the PCE-relevant subset — is where the edge lives.

**The honest caveat.** PPI is a noisy, partial, short-lead read on the consumer inflation the Fed targets. It is genuinely useful — as a probability-shifter, a nowcast input, and a margin gauge — but it is not a forecast, and its standalone market reaction is muted. Trade it as evidence, scaled by the regime, and never as a crystal ball. The correlation is strong, but a strong correlation built on one inflation cycle is a regime, not a law. The single discipline that keeps you honest: before you act on a PPI print, name the regime (hot demand, soft demand, disinflation, supply shock), name the channel (CPI read-through, PCE nowcast, or margins), and only then size the trade. A PPI number without a regime and a channel attached is just a headline, and headlines are the cheapest thing in markets.

## Further reading and cross-links

Within this series:

- [CPI and asset prices: the master inflation correlation](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation) — the downstream anchor; the consumer-price reaction function that PPI feeds into.
- [PCE, breakevens, and the forward inflation correlation](/blog/trading/macro-correlations/pce-breakevens-and-the-forward-inflation-correlation) — the Fed's actual target gauge and how markets price the forward path of inflation.
- [Lead, lag, or coincident: the time axis of every correlation](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) — the cross-correlation machinery behind PPI's short lead.
- [Oil prices, CPI, and the energy-equity correlation](/blog/trading/macro-correlations/oil-prices-cpi-and-the-energy-equity-correlation) — the commodity link at the very top of the upstream chain.
- [What correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) — r, slope, and what a coefficient of 0.9 really tells you.

The release-day reaction (intraday mechanics):

- [PPI: the upstream signal and the CPI read-through](/blog/trading/event-trading/ppi-the-upstream-signal-and-cpi-read-through) — how the PPI print trades intraday and frames the CPI reaction.
- [CPI: the report that moves the world](/blog/trading/event-trading/cpi-the-report-that-moves-the-world) — the consumer-price release this report sets up.

The mechanism (why inflation moves assets):

- [Real vs nominal: real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — why an inflation surprise moves rates and everything priced off them.
