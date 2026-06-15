---
title: "Energy: Oil, Gas, and the Inflation Engine"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Oil and gas are the commodities that feed inflation most directly and spike hardest on supply shocks. This is what actually drives them, why the futures curve is half the story, and the one job energy does in a portfolio."
tags: ["asset-allocation", "cross-asset", "commodities", "oil", "natural-gas", "inflation-hedge", "futures", "contango", "backwardation", "supply-shock", "portfolio-construction"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Energy is the commodity that feeds inflation most directly and spikes hardest on supply shocks (OPEC, wars), so it is the sharpest hedge for exactly the regime that hurts both stocks and bonds. But you cannot buy and hold a barrel — you hold it through *futures*, and the shape of the futures curve decides half your return before the price even moves.
>
> - You **roll futures**, so the curve matters as much as the spot price. An *upward* curve (contango) can bleed you ~10% a year; a *downward* curve (backwardation) can pay you ~10% a year. That single mechanism explains why long-run commodity-index returns diverge so far from the headline oil price.
> - Oil's short-run **supply and demand are both inelastic** around ~100 million barrels a day, so a 2% shortfall can move the price 50%. That is why an OPEC cut or a war is a price *event*, not a gentle adjustment.
> - Energy's correlation with stocks is **positive in a demand boom but negative in a supply shock** — when an oil spike acts as a tax on growth, energy rises while stocks fall. That asymmetry is the whole reason an allocator holds it.
> - The one number to remember: in **2022**, US CPI hit a 40-year high of **9.06%** as WTI spiked to ~\$124, and the Bloomberg Commodity index returned **+16.1%** — the only major asset class that *won* the year stocks fell 18% and bonds fell 13%.

In April 2020, the price of a barrel of oil went *negative*. Not low — negative. On the 20th of April, the front-month West Texas Intermediate (WTI) crude oil futures contract settled at **minus \$37.63** a barrel. For one surreal afternoon, the market was paying *you* almost \$38 to take a barrel of oil off its hands. The reason was brutally physical: the world had stopped driving and flying because of the pandemic, oil kept coming out of the ground because you cannot switch a well off like a light, and every tank, pipeline, and supertanker was filling up. Whoever held a contract that was about to deliver actual barrels had nowhere to put them — and storage you do not have is worth less than nothing.

Two years later, the same commodity did the opposite. Russia invaded Ukraine in February 2022, the West moved to cut off a major oil and gas exporter, and WTI ripped to about **\$124** a barrel by March. Gasoline at the American pump hit records, and US inflation — which had already been climbing — peaked that June at **9.06%** year over year, a number the country had not seen since 1981. The same barrel that nobody wanted in 2020 was, in 2022, the thing eating every household's paycheck.

That is energy in two snapshots: an asset that can crash through zero and spike to record highs inside twenty-four months, and the asset whose price you feel most directly, every week, at the pump and in the grocery aisle. The diagram below is the mental model we will build the whole post around — the reason you can hold oil at all is the *futures curve*, and the shape of that curve decides whether time is working for you or against you.

![Two oil futures curves from the same spot price, one upward contango and one downward backwardation, with roll cost and roll yield labeled](/imgs/blogs/energy-oil-gas-the-inflation-engine-1.png)

This is the deep-dive on energy commodities in the *Cross-Asset Playbook* series — the post that sits alongside [the map of asset classes](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own), where we first laid out what you can own, and next to [gold](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock) and [the industrial metals](/blog/trading/cross-asset/metals-copper-silver-the-economys-pulse), the other two members of the commodity family. Stocks are a claim on corporate profits. Bonds are a claim on a stream of interest. Gold is a near-monetary store of value. Energy is the odd one out: it is a *consumable physical good* that the world burns by the tens of millions of barrels every single day, and you trade it through a contract that promises delivery of something you almost never actually want delivered. Understanding that gap — between the barrel and the contract — is where everything starts.

## Foundations: the barrel, the contract, and why you roll

Let us start from absolute zero, because energy is the asset where the difference between *the physical thing* and *the financial instrument* matters more than anywhere else.

### Oil is a physical good, measured in barrels

Crude oil is a raw material pumped out of the ground, refined into gasoline, diesel, jet fuel, heating oil, and the feedstock for plastics and fertilizer. It is the lifeblood of the physical economy: roughly **100 million barrels a day** (written "100 mb/d") are produced and consumed worldwide. A *barrel* is a unit of volume — 42 US gallons, about 159 liters — and it is the unit everything in oil is quoted in. When you hear "oil is at eighty dollars," that means one barrel of a benchmark crude costs about \$80.

There are two benchmark crudes you will hear about constantly:

- **WTI (West Texas Intermediate)** — the US benchmark, light and "sweet" (low sulfur), priced for delivery at a pipeline hub in Cushing, Oklahoma. When American news quotes "the oil price," it usually means WTI.
- **Brent** — the international benchmark, named after a North Sea oil field, priced for delivery by sea in the North Sea. Most of the world's oil trades at a price referenced to Brent.

The two usually trade within a few dollars of each other — Brent is typically a bit higher because it is the seaborne, globally accessible barrel, while WTI is landlocked in the middle of the United States. The gap between them (the *Brent-WTI spread*) widens when US production is glutted and the barrels are stuck inland, and narrows when the global and US markets are tightly linked. For our purposes, treat them as two slightly different prices for the same global commodity; we will use WTI numbers throughout because the cleanest data series — including the famous negative print — is WTI.

### You cannot buy and hold a barrel

Here is the first thing that makes energy unlike a stock or a bond. If you buy a share of a company, you can put it in a drawer for thirty years; it costs nothing to store and it might pay you dividends along the way. If you buy a barrel of oil, you have a 159-liter problem. You need a tank. The tank costs money. The oil can leak, evaporate, or degrade. And it pays you nothing while it sits there — no coupon, no dividend, just storage bills. *Holding the physical barrel has a negative yield*, the opposite of a bond.

So almost nobody who invests in oil ever touches a barrel. Instead they use a **futures contract** — an agreement, traded on an exchange, to buy or sell a fixed amount of oil at a fixed price on a fixed future date. One WTI futures contract covers 1,000 barrels. If you buy the contract that delivers next month at \$80, you have locked in the right and obligation to receive 1,000 barrels next month for \$80,000. The genius of the futures market is that it lets a financial investor get exposure to the oil price *without* ever arranging a tank — because they sell the contract before it delivers.

> A *futures contract* is just a standardized, exchange-traded promise: deliver X of something, at price P, on date D. It exists so that producers (who want to lock in a selling price) and consumers (an airline wanting to lock in fuel cost) can hedge — and it lets speculators take a price view without handling the physical good.

### The catch: futures expire, so you have to roll

A futures contract has an expiry date. The "front-month" contract — the nearest one — eventually comes due, at which point the holder must either take physical delivery (a tank problem) or close out and move on. A financial investor who wants *continuous* exposure to oil must therefore do something every month: sell the expiring contract and buy the next one out. This is called **rolling** the position.

Rolling is not free, and this is the single most important idea in the entire post. When you sell the expiring contract and buy the next one, you are selling at one price and buying at another. Whether that swap *costs* you money or *pays* you money depends entirely on the **shape of the futures curve** — the set of prices for delivery in 1 month, 2 months, 3 months, and so on out to a year or more.

### Contango vs backwardation: the two shapes of the curve

The futures curve can slope two ways, and which way it slopes decides whether time is your friend or your enemy.

- **Contango** is an *upward*-sloping curve: oil for delivery further out is *more expensive* than oil for delivery soon. The market is saying "barrels later are worth more than barrels now" — usually because storage is full and nobody is desperate for oil today. When you roll in contango, you sell the cheap near contract and buy the more expensive far one, so every roll *costs* you a little. This drip is called **negative roll yield**, and over a year it can quietly bleed a double-digit percentage out of your position even if the spot price never moves.

- **Backwardation** is a *downward*-sloping curve: oil for delivery soon is *more expensive* than oil for delivery later. The market is saying "I need a barrel now and I'll pay up for it" — usually because supply is tight and inventories are low. When you roll in backwardation, you sell the expensive near contract and buy the cheaper far one, so every roll *pays* you a little. This is **positive roll yield**, and over a year it can *add* a double-digit percentage to your return even if spot is flat.

The cover figure shows both: one spot price of \$80, two possible curves. The upper curve (contango) climbs to \$88 a year out — hold through it and you bleed. The lower curve (backwardation) falls to \$72 a year out — hold through it and you earn. The shape, not the spot, is half your return.

This is why "the price of oil went up 50% but my oil fund barely moved," and "oil was flat but my oil fund lost a third," are both real, common experiences. The fund tracks the *rolled futures*, not the *spot barrel*. Internalize this and you will never be surprised by a commodity fund again.

#### Worked example: how the roll bleeds you in contango

Let us make the roll cost concrete, because it is the number that traps unwary investors.

Suppose spot oil is **\$80** a barrel and the curve is in contango: the contract one month out costs about \$80.67, two months out \$81.34, and the contract twelve months out costs **\$88**. You run a simple strategy: always hold the front-month contract, and each month sell it and buy the next one to stay continuously exposed.

Walk one month. You hold the contract that is now expiring at, say, \$80. To stay in the market you must buy next month's contract, which costs more because the curve slopes up. Each monthly roll, you sell low and buy a notch higher. Over twelve rolls, even if the spot price of oil is *exactly* \$80 at the start and \$80 at the end — no net price move at all — you have paid that upward step twelve times.

The math: a curve where the 12-month price is \$88 against an \$80 spot is about 10% higher far-out than near. Rolling monthly through that slope costs you roughly that 10% over the year. So:

- Start: \$10,000 in an oil-futures index, spot \$80, curve in contango (\$88 at 12 months).
- End of year: spot oil is *still* \$80 — zero price change.
- Your index is worth about **\$9,000**. You lost ~\$1,000, about **10%**, purely to roll cost.

The intuition: in contango, you are paying a fee every month to rent exposure to a barrel you will never hold — and a flat oil price is a *losing* year for the futures holder.

#### Worked example: how the roll pays you in backwardation

Now flip the curve. Same spot of **\$80**, but the market is tight: the contract one month out costs \$79.33, two months out \$78.66, and the twelve-month contract costs **\$72**. The curve slopes down — backwardation.

Each month you sell the expensive near contract and buy the cheaper next one. You are rolling *down* the curve, capturing the step each time. Over twelve rolls:

- Start: \$10,000 in the oil-futures index, spot \$80, curve in backwardation (\$72 at 12 months).
- End of year: spot oil is *still* \$80 — again, zero price change.
- Your index is worth about **\$11,000**. You earned ~\$1,000, about **+10%**, purely from roll yield.

A 12-months-out price of \$72 against an \$80 spot is about 10% *below* near, and rolling down that slope earns you roughly that 10% over the year.

The intuition: in backwardation, you are *paid* to provide the market the patience it lacks — the holder of futures earns a positive return on a flat oil price. This is exactly why long-run commodity-index returns diverge so wildly from spot price changes: spot oil in 2024 was lower than in 2008, yet a well-timed backwardated futures position could have earned positive carry across many of those years.

### Why the curve slopes the way it does: storage and convenience

It is worth understanding *why* the curve takes one shape or the other, because the explanation is the bridge between the financial contract and the physical barrel. There are two opposing forces.

The first is the **cost of carry**. If you could store oil cheaply, then in a normal market the futures price *should* equal today's spot price plus the cost of storing the oil until delivery — storage fees, insurance, and the interest you forgo on the cash tied up. That cost-of-carry logic naturally produces a mild *upward* slope (contango): a barrel for delivery in a year "should" cost a bit more because someone had to store it for a year. So contango is the market's resting state when oil is plentiful and storage is available — the curve simply prices in the cost of keeping the barrel around.

The second force pulls the other way: the **convenience yield**. When physical oil is scarce — a war, a refinery outage, a sudden demand surge — there is real value in having a barrel *in your hands right now* rather than a promise of one later. A refinery that runs out of crude has to shut down; an airline that runs out of jet fuel grounds its fleet. That value of immediate availability is the *convenience yield*, and when it is high it *exceeds* the cost of carry and flips the curve into *backwardation*: people will pay a premium for oil today versus oil next year. So backwardation is the market shouting "I need it now," and it is both a fundamental signal of physical tightness and the source of the roll yield you earn. The curve's shape is not arbitrary — it is the physical supply-demand balance, expressed as a slope.

### WTI versus Brent, and why the spread moves

We met WTI and Brent earlier as two prices for the same global commodity. The gap between them — the *Brent-WTI spread* — is itself a small, instructive market. Brent is the seaborne barrel that can sail anywhere in the world, so it reflects the global balance. WTI is landlocked at Cushing, Oklahoma, deep in the US interior, so its price reflects how easily American oil can reach the coast and the wider world. When US shale production booms and the pipelines to the Gulf Coast are full, WTI barrels get "trapped" inland, and WTI trades at a *discount* to Brent — sometimes \$5, \$10, even \$20 below. When the global market is tight and US oil can be exported freely, the spread narrows toward zero.

#### Worked example: trading two benchmarks for the same barrel

Suppose Brent is **\$85** and WTI is **\$78**, a **\$7** spread, because a pipeline bottleneck is trapping US crude inland. You believe new pipeline capacity is about to open, which will let the trapped barrels reach the coast and narrow the spread toward \$3.

- You buy WTI at \$78 and simultaneously sell Brent at \$85 — a *spread trade* that profits if the gap narrows, regardless of whether oil overall goes up or down.
- The pipeline opens; WTI rises to \$83 as the trapped barrels find buyers, while Brent holds at \$86. The spread is now \$3.
- Your WTI leg gained \$83 − \$78 = **+\$5**; your short Brent leg lost \$86 − \$85 = **−\$1**. Net: **+\$4** per barrel, on a 1,000-barrel contract that is **+\$4,000** — earned purely from the spread narrowing, with almost no exposure to the absolute oil price.

The intuition: WTI and Brent are not really one price — they are two prices linked by the physical logistics of getting oil from where it is pumped to where it is burned, and the gap between them is a tradeable read on those logistics.

## What actually drives the oil price

We now have the foundation: you hold oil through rolled futures, and the curve's shape is half the story. The *other* half is the spot price itself — and that is governed by a famously tight tug-of-war between supply and demand. The figure below is the map of what moves it.

![Hub diagram of oil price drivers, with OPEC, shale, and geopolitics on the supply side feeding a thin balance against near-fixed demand](/imgs/blogs/energy-oil-gas-the-inflation-engine-3.png)

### Demand is huge, steady, and short-run inelastic

The world consumes about **100 million barrels a day**, and that number changes *slowly*. Global oil demand grows with the economy — more flights, more trucking, more plastic, more people — at roughly 1% a year in normal times. It does not swing around week to week.

Crucially, oil demand is **inelastic in the short run** — meaning the quantity people consume barely responds to price over a few months. If gasoline doubles tomorrow, you still drive to work, the trucks still deliver food, the planes still fly the routes that are booked. You cannot replace your car, your furnace, or your supply chain overnight. Demand only bends to price over *years*, as people buy more efficient cars and economies electrify. In the short run, the demand curve is nearly vertical.

*Inelasticity* is the single most important property of oil for understanding its violence. When both supply and demand barely respond to price in the short run, a small imbalance has to be resolved entirely by a *large* price move. There is no quantity adjustment available to cushion it — only price can clear the market.

### Supply is the swing factor — and it has three faces

If demand is steady, then most of the action comes from the supply side, which has three main forces:

- **OPEC+ quotas.** OPEC (the Organization of the Petroleum Exporting Countries) plus allies like Russia — together "OPEC+" — control a large share of the world's exportable oil. They meet and set *production quotas*: agreements to pump more or less. When they cut output by 1-2 mb/d to support prices, they are deliberately tightening the balance. They also hold *spare capacity* — wells they could turn on but choose not to — which is the world's main shock absorber. When spare capacity is high, the market feels safe; when it is low, every outage becomes a panic.

- **US shale.** Over the 2010s, the United States became the world's largest oil producer thanks to *shale* — oil locked in rock, freed by hydraulic fracturing ("fracking"). Shale is different from a giant Saudi field: a shale well is cheaper to start but depletes fast, so producers must keep drilling to maintain output. This makes US supply *fast* but *high-cost* — it responds to price within 6-12 months, ramping up when oil is expensive and collapsing when it is cheap. Shale put a rough ceiling and floor on oil that did not exist before: above ~\$80, shale floods in; below ~\$40, it shuts off.

- **Geopolitics.** Oil is concentrated in politically volatile regions, so wars, sanctions, revolutions, and pipeline attacks regularly knock barrels offline. This is the *spike risk* — the source of the sudden, violent moves that make energy a hedge. The 1973 Arab oil embargo, the 1990 Gulf War, the 2022 invasion of Ukraine: each was a supply shock that sent prices vertical.

### Spare capacity: the world's shock absorber

One concept ties the supply side together and deserves its own paragraph: **spare capacity** — the volume of oil production that could be brought online quickly (within about 30-90 days) but is being deliberately held back, almost all of it inside Saudi Arabia and a handful of other Gulf producers. Spare capacity is the market's safety cushion. When it is ample — say 4-5 mb/d of idle, ready production — a disruption somewhere can be offset by turning on idle wells elsewhere, so prices stay calm even when news is bad. When spare capacity is thin — say under 2 mb/d — there is no cushion, and any outage forces the entire adjustment onto price. This is why two identical-looking supply disruptions can produce wildly different price reactions: the one that hits when spare capacity is thin spikes the price; the one that hits when it is ample barely registers. An allocator watching for a supply-shock regime watches spare capacity as closely as the headlines, because it is the variable that turns a piece of bad news into a price event.

This also explains OPEC's enduring power. By choosing how much to produce versus hold back, OPEC effectively *sets* the level of global spare capacity — and therefore the market's sensitivity to shocks. Cutting production tightens the balance directly *and* signals discipline; holding spare capacity back keeps the market on edge. The 2014 decision to abandon that role and pump flat-out, conceding share to shale, is exactly why the price collapsed: OPEC stopped managing the cushion.

### The balance is thin, so the price is violent

Put it together and you have the engine of oil's behavior. Demand is near-fixed at ~100 mb/d. Supply is slow-moving and only partly controllable. The *balance* between them — tracked obsessively through **inventories** (the level of oil sitting in storage tanks worldwide) — is razor-thin. A 2 mb/d shortfall is only 2% of demand, but because nothing else can adjust quickly, it can move the price 30-50%.

#### Worked example: why a 2% shortfall moves the price 40%

Let us make the inelasticity concrete with simple numbers.

The world needs **100 mb/d**. Suppose a war takes **2 mb/d** offline — a 2% supply cut. In a normal good, a 2% shortage would mean a roughly 2% price rise and people would consume a hair less. But oil demand barely bends in the short run.

Imagine demand responds to price only weakly: it takes a **20%** price rise to coax consumers to use just **1%** less oil (an "elasticity" of about 0.05 — for every 1% price move, quantity moves 0.05%). To close a 2% gap purely by suppressing demand, you need:

- Required demand reduction: **2%** (to match the lost 2 mb/d, ignoring the supply side responding).
- Price move needed at 0.05 elasticity: 2% ÷ 0.05 = **40%**.
- So if oil started at **\$80**, it has to rise to about **\$80 × 1.40 = \$112** just to ration 2% of demand away.

A 2% supply loss produced a 40% price spike — and the household sees gasoline jump from \$3.50 to nearly \$5.00 a gallon. The intuition: when neither side of the market can adjust quantity quickly, all the adjustment lands on price, which is why oil makes such large moves on such small physical shocks.

This worked example is a sketch, not a forecast — real elasticities vary and the supply side also responds — but it captures the mechanism that makes energy the most explosive major commodity. (For how these inflation impulses feed into the broader rate and inflation machinery, the macro series' piece on [real vs nominal inflation](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) is the companion read.)

## Natural gas: oil's wilder cousin

Oil gets the headlines, but natural gas is the more *violent* energy commodity, and an allocator should understand why before treating "energy" as one thing.

### Gas is regional, not global

Oil is a global commodity: a barrel in Texas and a barrel in Singapore are linked by tankers, so their prices stay close. Natural gas is different. Gas is mostly moved by *pipeline*, which only connects places that are physically plumbed together. To ship gas across an ocean you must chill it to minus 162°C until it becomes a liquid (*liquefied natural gas*, or LNG), load it onto a specialized tanker, and re-gasify it at the other end — expensive and capacity-limited. So gas markets are *regional*, and the same molecule can cost wildly different amounts in different places:

- **Henry Hub** — the US benchmark, priced at a pipeline hub in Louisiana, quoted in dollars per *MMBtu* (one million British thermal units, a unit of energy content). American gas has been cheap and abundant thanks to shale gas.
- **TTF** — the main European benchmark (the "Title Transfer Facility" in the Netherlands). Europe imports most of its gas, so TTF is far more exposed to supply shocks.

In 2022, when Russian pipeline gas to Europe was cut, European TTF prices exploded to the equivalent of *several hundred dollars* per barrel of oil on an energy-equivalent basis, while US Henry Hub gas, insulated by domestic shale, was a fraction of that. Two prices, same molecule, because the pipelines did not connect.

### Gas is storage-driven and seasonal

Gas demand is heavily *seasonal* — it spikes in winter for heating and (increasingly) in summer for air-conditioning electricity. Because you cannot instantly produce more, the market runs on *storage*: gas is injected into underground caverns in the shoulder seasons and withdrawn in winter. The whole price hinges on whether storage is filling fast enough before the cold. A warm autumn that leaves storage full crashes the price; a cold snap that drains it sends the price vertical. This storage-and-weather dependence, plus the regional fragmentation, makes gas swing far harder than oil. The chart below shows just how hard.

![Henry Hub natural gas price line from 2020 to 2024, swinging from about one and a half dollars to a peak near ten dollars](/imgs/blogs/energy-oil-gas-the-inflation-engine-6.png)

Read it left to right. In 2020, US Henry Hub gas bottomed near **\$1.5** per MMBtu — demand had collapsed and storage was full. By August 2022, with Europe scrambling for every cargo of LNG it could buy after the Russian cutoff, even US gas was dragged up to a peak near **\$9.7**. Then it fell back to roughly **\$2.5** through 2023 and 2024 as the panic faded and supply caught up. That is a **more than six-fold range in three years** — a swing oil rarely matches. Gas is the commodity that most rewards and most punishes timing.

#### Worked example: a cold-winter spike in a gas position

Let us put the volatility in dollar terms.

Suppose you put **\$10,000** into a US natural gas exposure when Henry Hub is at **\$2.50** per MMBtu, a calm summer level. A brutally cold winter arrives, storage drains fast, and the price runs to **\$6.00** — not even the extreme of 2022, just a hard winter.

- Price move: from \$2.50 to \$6.00 is a **+140%** move in the spot.
- Your \$10,000, if it tracked spot, becomes about **\$24,000**.

Now the warning. Gas is almost always in steep *contango* in the off-season (the far months price in the next winter's premium), so a futures-based gas fund bleeds heavy roll cost. The single most notorious example: a popular US natural-gas futures fund lost the overwhelming majority of its value over its life — not because gas fell to zero, but because relentless contango roll cost compounded year after year. So that same \$10,000, held through a futures fund across a flat-but-contangoed stretch, could *halve* even while spot gas went nowhere.

The intuition: gas can double on a cold snap, but its vicious contango means the futures vehicle can destroy capital between the spikes — gas is a trade, not a hold.

### LNG is slowly stitching the gas markets together

One structural shift is worth flagging for the allocator, because it changes gas's behavior over time. Historically the regional gas markets were almost entirely separate — US Henry Hub, European TTF, and Asian LNG prices could diverge by a factor of ten with no mechanism to arbitrage them, because you simply could not move the molecule across oceans at scale. The build-out of *LNG* export and import terminals over the 2010s and 2020s is changing that. As more terminals come online, US gas can increasingly be liquefied, shipped, and sold into Europe or Asia, which links the prices: when European gas spikes, US exporters chase the higher price, dragging US gas up too (exactly what happened in 2022). Gas is on a slow path from a set of isolated regional markets toward a more global one, like oil — but it is decades behind, and for now the regional fragmentation, and the violent dislocations it produces, remain the defining feature. The 2022 European energy crisis was, in part, the painful demonstration that the plumbing to globalize gas was not yet built out enough to rescue Europe quickly.

## How energy behaves: violent spikes and crashes

Now we can describe energy's *behavior* — its volatility, its crashes, its spikes — using the drivers we have built. The defining feature is that energy does not drift; it lurches. The chart below is the WTI price history, and its shape is the whole point.

![WTI crude oil price history line with the 2008 peak, the 2020 negative print, and the 2022 war spike annotated](/imgs/blogs/energy-oil-gas-the-inflation-engine-2.png)

Read it as a series of regime breaks, not a trend:

- **2008: the \$147 peak and the crash to \$44.** In July 2008, with the global economy running hot and supply fears everywhere, WTI touched a record **\$147**. Then the financial crisis hit, demand fell off a cliff, and by December 2008 oil had collapsed to **\$44** — a two-thirds wipeout in five months. That is a demand-driven crash.
- **2011-2014: the \$100 plateau.** Oil sat near **\$99-107** for several years on steady demand and Middle East tension — the calm before a supply break.
- **2014-2016: the shale crash.** US shale flooded the market, OPEC declined to cut, and oil fell from **\$107** (mid-2014) to a low of **\$26** by February 2016 — a supply-driven crash.
- **2020: the negative print.** As described at the top, the pandemic demand collapse plus full storage produced the first-ever negative settlement, **−\$37.63**, on 20 April 2020.
- **2022: the war spike.** Russia's invasion drove WTI to **\$124** by March 2022 before it settled back to **\$80** by year-end — a supply-driven spike.

Look at the magnitudes. Over this stretch oil ranged from *minus \$37* to *plus \$147* — a peak-to-trough spread that no stock index, no bond, and even gold cannot match. Energy is, by a wide margin, the most volatile major asset class an allocator will hold.

### Why the crashes can overshoot through zero

The 2020 negative print deserves its own explanation, because it teaches the deepest lesson about energy. The price went negative not because oil became worthless, but because the *futures contract* obligated its holder to take physical delivery of barrels they had nowhere to store. With every tank in Cushing full, the *cost* of accepting delivery (renting impossible storage, or paying someone to haul it away) exceeded the value of the oil itself. So holders paid to escape. This is the most vivid possible demonstration of the post's core idea: **you are not trading the barrel, you are trading the contract**, and the contract can do things — like go negative — that the underlying physical commodity's "value" never could.

It also explains why energy crashes are so brutal. When supply cannot stop (wells run, production is committed) and demand vanishes, there is no floor until physical storage fills — and once it fills, the price can punch through zero. Stocks have a floor at zero; an over-supplied front-month oil contract does not.

## Common misconceptions

Energy is surrounded by intuitive-but-wrong beliefs. Here are the five that cost investors the most.

**"If oil goes up 50%, my oil fund goes up 50%."** No — your fund holds rolled futures, not spot barrels. In contango it can lag the spot rise badly (the roll bleeds), and in a flat market it can *lose* money even though oil "did nothing." Over the 2009-2020 era of persistent contango, futures-based oil indices dramatically underperformed the spot price. The roll, not the spot, often dominates the multi-year return.

**"Negative oil prices mean oil was worthless."** No — they meant the *front-month futures holder* faced a delivery they could not handle. The physical commodity still had real value at refineries; the negative number was a storage-and-contract artifact, and the very next contract month traded around +\$20 the same day. It was a plumbing crisis, not a value collapse.

**"High oil prices are always good for the economy."** No — for an oil-*importing* economy (most of them), an oil spike is a *tax on growth*. Consumers spend more at the pump and less on everything else; businesses face higher input costs. Most US recessions since 1970 were preceded by an oil spike. High oil is good for oil *producers* and bad for nearly everyone else — which is exactly why energy hedges a stock portfolio in a supply shock.

**"Energy and stocks always move together."** No — and this is the most important nuance. In a *demand* boom (a strong economy lifts both oil and stocks), they move together, around +0.35 correlation. But in a *supply shock* (a war spikes oil while choking growth), they move *opposite*: energy soars while equities fall. The correlation is regime-dependent, and the regime where it flips negative is precisely the regime where you need a hedge.

**"Buy and hold commodities for the long run, like stocks."** No — a barrel of oil throws off no cash flow and physically degrades, so unlike a stock there is no compounding engine inside it. Long-run *spot* commodity returns have roughly tracked inflation, no more. The only structural return from a commodity *index* comes from the roll yield (positive in backwardation) plus the cash collateral's interest — not from the commodity drifting up forever. Energy is a *cyclical* and *regime* allocation, not a buy-and-forget compounder.

**"Owning an oil company is the same as owning oil."** No — an oil-producer's stock and the oil price are related but not the same exposure. A producer's profit depends on its *costs*, its debt, its management discipline, and its tax regime as much as on the oil price, and the share carries broad stock-market beta on top. In a 2022-style supply shock the two can move together (both up), but in a 2008-style demand crash the oil company falls *with* the stock market even though it is "an oil play." If you want the *purest* exposure to the commodity — the cleanest inflation and supply-shock hedge — you want the futures, not the equity; the equity is a hybrid of an oil bet and a stock bet.

## How energy shows up in real markets

Mechanisms become real when you watch them play out. Here are the episodes that an allocator should know cold, each one a driver from this post in action.

### 1973-74: the original supply shock and the birth of "stagflation"

The template for everything energy does as a hedge was set in October 1973. Arab members of OPEC, retaliating for Western support of Israel in the Yom Kippur War, declared an *embargo* — they cut production and refused to sell to certain countries. Oil, which had been a sleepy ~\$3 a barrel, roughly *quadrupled* toward ~\$12 over the following year. The effect on the West was devastating and instructive: it produced *stagflation*, the toxic mix of stagnant growth and high inflation that economists had thought impossible. US consumer prices rose sharply (cumulative inflation over 1973-74 was around 22%), the economy fell into recession, and the S&P 500 dropped roughly **37%** across 1973-74. Yet the assets that *won* were the real ones: oil itself rose several-fold, and gold roughly tripled over the two years. This is the cleanest historical proof of the post's central claim — a supply shock raises inflation and lowers growth at the same time, crushing stocks and bonds while energy and gold soar. Every later episode, including 2022, is a variation on 1973.

### 2008: the demand-driven round trip

Oil rode the pre-crisis global boom to **\$147** in July 2008, with talk of "\$200 oil" everywhere. Then Lehman Brothers failed in September, the global economy seized, and oil demand fell hard. By December, WTI was **\$44** — a ~70% crash in five months. The lesson: when the *demand* side breaks, oil and stocks crash *together* (the S&P 500 fell 37% in 2008, and oil fell more). In a demand-driven downturn, energy is *not* a hedge — it is a high-beta bet on growth. This is the case where the positive energy-equity correlation bites you.

### 2014-2016: shale breaks OPEC

For years OPEC had defended high oil prices by cutting output. In late 2014, facing a flood of US shale oil, OPEC changed strategy: instead of cutting, it kept pumping to defend market share and let the price fall to drive high-cost shale out of business. Oil collapsed from **\$107** to **\$26**. It was a deliberate, supply-driven crash — and it showed that US shale had fundamentally changed the market by adding a fast, price-sensitive source of supply. The new rule: shale floods in above ~\$80 and shuts off below ~\$40, putting soft bounds on the price that did not exist before 2010.

### 2020: the day oil went negative

The pandemic erased ~20-30 mb/d of demand almost overnight as the world locked down. Production could not stop fast enough, and by mid-April every storage tank was filling. On 20 April 2020, holders of the expiring May WTI contract — facing physical delivery into a Cushing hub with no room — paid to get out, and the contract settled at **−\$37.63**. Within months, OPEC+ slashed output by a record ~10 mb/d, demand recovered, and oil was back near **\$48** by December. The episode is the definitive lesson that you trade the contract, not the barrel.

### 2022: the supply-shock hedge that paid off

This is the case study that justifies holding energy at all. In 2022, US stocks fell **−18.1%** and US bonds fell **−13.0%** — the rare year both legs of a balanced portfolio lost double digits, the worst year for a 60/40 portfolio since the 1930s. What worked? Energy and broad commodities. WTI spiked to **\$124** on the Ukraine invasion, US CPI hit a 40-year high of **9.06%**, and the **Bloomberg Commodity index returned +16.1%** for the year — the only major asset class in positive territory. An investor with even a small energy or commodity sleeve cushioned a brutal year. The chart below shows the pattern across a full decade.

![Bloomberg Commodity index annual total returns bar chart with the 2021 and 2022 inflation years highlighted green](/imgs/blogs/energy-oil-gas-the-inflation-engine-4.png)

The bars tell the whole story of *when* commodities win. The two big green bars — **2021 (+27.1%)** and **2022 (+16.1%)** — are the inflation years, exactly when stocks and bonds were under the most pressure. The deep red bars — **2014 (−17.0%)** and **2015 (−24.7%)** — are the shale crash, when oil collapsed. Commodities spent most of the 2010s as a drag (a low-inflation, well-supplied decade) and then earned their keep explosively in the 2021-2022 inflation spike. That is not a smooth diversifier you hold for steady returns; it is a *regime* asset that pays off in the one environment that hurts everything else.

### The pump and the paycheck: oil as the inflation signal

There is a reason central bankers and ordinary households both watch the gas price. Gasoline is the single most *visible* price in the economy — posted in foot-high numbers on every corner, paid weekly, impossible to ignore. When oil spikes, it flows directly into headline inflation through fuel, and indirectly through the cost of everything that must be shipped or manufactured. The chart below shows the linkage.

![Dual axis chart of WTI crude oil and US consumer price inflation from 2020 to 2024 moving together](/imgs/blogs/energy-oil-gas-the-inflation-engine-5.png)

The blue line is WTI; the red dashed line is US CPI year over year. Watch them move together through 2021-2022: oil ran from the 2020 collapse up to about **\$124**, and CPI climbed in lockstep to its **9.06%** peak in June 2022. Then, as oil eased back toward **\$70-80** through 2023, headline inflation fell sharply too. The relationship is not perfect — CPI has a large non-energy core — but energy is the most volatile component and the one that drives the *swings* in headline inflation. This is the mechanical reason energy is the cleanest inflation hedge: when inflation surprises to the upside, energy is very often *the cause*, so owning it is owning the thing that is doing the damage. (For how inflation surprises propagate into the rotation between risk assets, see [risk-on, risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates).)

## How energy correlates with the rest of the portfolio

We have now reached the question that matters most to an allocator: not "is energy a good investment on its own?" but "what does energy do *alongside* my stocks and bonds?" The answer is the most interesting in the cross-asset world, because it is **regime-dependent** — energy changes its relationship to stocks depending on *why* the price is moving.

### Positive with inflation — the core of the hedge

Energy's most reliable relationship is with inflation: it is *positively* correlated, and tightly so, because energy *is* a large input to inflation. When inflation rises, energy almost always rises with it (often leading it). This is the property that makes energy and broad commodities the textbook inflation hedge. Stocks and bonds, by contrast, are *negatively* exposed to an inflation surprise — bonds because higher inflation erodes their fixed coupons and forces rates up (their prices fall), stocks because higher rates compress valuations and squeeze margins. So energy zigs exactly when the rest of a conventional portfolio zags on an inflation shock. That is the entire diversification case in one sentence.

### Positive with stocks in a demand boom

When the economy is booming, growth lifts *both* corporate profits (stocks up) and oil demand (energy up). In these demand-driven moves, energy and equities are positively correlated — roughly **+0.35** for broad commodities versus stocks over a typical period, a touch higher for energy specifically. In a healthy expansion, energy is *not* much of a hedge; it is a pro-cyclical, somewhat redundant bet on growth. This is why energy disappoints in a calm, growing, low-inflation decade like the 2010s.

### Negative with stocks in a supply shock — the payoff

Here is the asymmetry that makes energy special. When the oil price rises because of a *supply* shock — a war, an embargo, an OPEC cut — rather than booming demand, it acts as a **tax on growth**. Every extra dollar at the pump is a dollar consumers cannot spend elsewhere; every higher input cost squeezes corporate margins. So a supply-driven oil spike *lowers* expected growth and *raises* inflation simultaneously — the toxic combination called *stagflation*. In that regime, stocks fall (lower growth, higher discount rates) while energy soars (it is the cause). The correlation flips *negative* exactly when you need it most.

This is the heart of the allocation case, so state it plainly: **energy's correlation with stocks is positive in the good times (when you don't need a hedge) and negative in the supply-shock times (when you desperately do).** A hedge that works precisely in the regime that hurts you most is worth far more than its average correlation suggests. The 1973-74 stagflation (stocks roughly halved, oil quadrupled) and 2022 (stocks −18%, commodities +16%) are the two cleanest examples of this flip in action.

#### Worked example: energy as portfolio insurance in a supply shock

Let us price the hedge in a simple portfolio.

You hold **\$100,000**: **\$90,000** in stocks and **\$10,000** in an energy/commodity sleeve (a 90/10 split, energy as a 10% slice). A supply shock hits — a war spikes oil.

- Stocks fall **20%** in the stagflation hit: \$90,000 × (1 − 0.20) = **\$72,000**. That is a \$18,000 loss on the equity sleeve.
- Energy rises **40%** as the oil spike and backwardation work in your favor: \$10,000 × 1.40 = **\$14,000**. That is a \$4,000 gain.
- Total portfolio: \$72,000 + \$14,000 = **\$86,000**, a **−14%** loss.

Compare to an all-stock investor who lost the full **−20%**. The 10% energy slice cushioned 6 percentage points of the drawdown — and it did so *because* the same shock that crushed stocks lifted energy. Now the honest other side: in a calm growth year, that energy slice might *lag* — energy returns 5% while stocks return 20%, so the 90/10 mix earns about 18.5% versus 20% all-stock, giving up ~1.5 points.

The intuition: energy is insurance, and like all insurance it costs you a little in the good years and pays off in the bad ones — but unlike most insurance, in a backwardated market the *premium can be negative* (you get paid to hold it). That is the rare and valuable property.

## When to own it: the energy allocation playbook

Everything so far has been building to one decision: *when do I hold energy, how much, in what form, and what tells me I am wrong?* The matrix below is the playbook.

![Energy allocation playbook matrix showing lean-in and stand-aside readings across inflation risk, curve shape, roll, equity correlation, and sizing](/imgs/blogs/energy-oil-gas-the-inflation-engine-7.png)

### The two conditions that have to line up

Energy earns its slice when **two** things are true at once:

1. **Inflation and supply-shock risk are rising.** OPEC is cutting, geopolitical tension is high, inventories are tight, or the economy is overheating. This is the *demand* for energy as a hedge — it pays off in this regime.
2. **The futures curve is in backwardation.** A downward-sloping curve means you get *paid to wait* through positive roll yield, and it is also a market signal that physical supply is genuinely tight. Backwardation and supply-shock risk usually arrive together, which is the point — the curve confirms the fundamental story.

When both hold, energy is a high-conviction allocation: it hedges the inflation/supply regime that hurts stocks and bonds, *and* the roll pays you to hold it. When the curve is in **contango** and demand is fading (a recession, a glut), the opposite is true — you bleed roll cost while waiting for a hedge you may not need — so you trim hard, toward zero. Energy is a tail hedge, not a permanent holding.

### The energy-equity vs energy-futures choice

There are two main ways to own energy, and they behave differently:

| | **Energy equities** (oil & gas company stocks / ETF) | **Energy futures** (commodity index / oil futures ETF) |
|---|---|---|
| **What you own** | Shares of producers — a claim on their profits | Rolled exposure to the oil price itself |
| **Pays you** | Dividends, often generous | No yield; roll yield (±) in backwardation/contango |
| **Inflation hedge quality** | Indirect — depends on the company's costs and discipline | Direct — it *is* the commodity price |
| **Main drawback** | Has equity-market beta — falls with stocks in a crash | Roll cost in contango can bleed double digits a year |
| **Best when** | You want income and some hedge with less roll drag | You want the *purest* supply-shock hedge, curve in backwardation |

The trade-off: energy *stocks* give you dividends and a softer, equity-like exposure, but they carry stock-market beta, so in a broad sell-off they fall with everything else — diluting the hedge. Energy *futures* give you the cleanest, most direct exposure to the commodity (the truest hedge), but you pay the roll, so they are only attractive when the curve is in backwardation. A common allocator solution is to use energy *equities* as a long-term, income-paying tilt and *futures* tactically when the curve is backwardated and supply-shock risk is rising.

### Sizing, and what invalidates the case

A practical energy or broad-commodity sleeve is small — typically **5-10%** of a diversified portfolio. The logic mirrors the gold case: energy is so volatile (annualized volatility well above stocks) that even a small slice delivers a meaningful hedge, while a large slice would dominate the portfolio's risk with its lurches. Within that band, you lean toward 10% when both conditions hold (inflation risk rising, curve backwardated) and back toward 5% — or, for futures, toward zero — when the curve is in steep contango and the macro picture is disinflationary.

There is also a quiet free lunch in *how* you hold a volatile asset like energy: rebalancing. Because energy lurches so far from its target weight, a disciplined investor who rebalances back to, say, 7% every quarter is mechanically forced to *sell* energy after a spike (when it has ballooned to 12% of the portfolio) and *buy* it after a crash (when it has shrunk to 3%). Over a full cycle, that systematic "sell high, buy low" can add more to a multi-asset portfolio's return than the energy sleeve's own average return — precisely *because* energy is so volatile and mean-reverting. The volatility that makes energy unpleasant to hold in isolation is, inside a rebalanced portfolio, a source of extra return. This is the deepest reason an allocator tolerates a small energy slice even knowing its long-run spot return barely beats inflation: it is not there to compound on its own, it is there to be volatile in the right direction at the right time and to be harvested through rebalancing.

What invalidates the case? Three signals tell you to trim:

- **The curve flips to deep contango.** You are now paying to hold the hedge; the carry has turned against you.
- **Demand is rolling over into recession** without a supply shock. A demand-driven downturn crushes oil *with* stocks (the 2008 case), so energy stops hedging and starts adding to the loss.
- **Inflation is clearly falling and well-anchored.** The regime energy hedges is receding, so the slice can shrink toward its floor.

And never size energy as if it were a core compounder. It has no internal cash-flow engine; its long-run spot return barely beats inflation. You hold it for *what it does in 2022*, not for what it does on average — a sharp, regime-specific hedge that pays off in exactly the environment that punishes a conventional stock-and-bond portfolio.

## When this matters to you

If you only ever own a stock-and-bond portfolio, the single year that will hurt you most is the one where *both* fall together — and history says that year is almost always an inflation or supply shock. 2022 was that year, and 1973-74 was that year. In both, the asset that paid off was energy, because energy is not just *exposed* to inflation — in a supply shock, it *is* the inflation. That is why an allocator keeps a small, deliberate energy or broad-commodity sleeve even through the long, boring decades when it does nothing: it is the one holding that zigs when everything else zags, in the precise regime that classic diversification fails.

The deeper lesson is the one we opened with: in energy, you never own the thing — you own a contract on the thing, and the *shape of the curve* decides whether time pays you or robs you. Internalize contango and backwardation and you will understand why a commodity fund can lose money while oil rises, why a natural-gas fund can be a wealth incinerator, and why the best time to own energy futures is exactly when the market is in backwardation, paying you to provide the patience it lacks.

From here, the natural next steps in this series are the other two members of the commodity family and the macro signals that govern all of them: [gold, the monetary metal that hedges *real yields* rather than inflation directly](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock); [copper and silver, the industrial metals that read the economy's pulse](/blog/trading/cross-asset/metals-copper-silver-the-economys-pulse); and the master macro signal that ties energy, gold, and rates together, [real vs nominal inflation and real yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal). Energy is the sharpest tool in the inflation-hedge kit — but it is a tool, with a specific job, a real cost, and a curve you must respect.

*This is educational material about how energy markets and asset allocation work, not individualized financial advice. Energy is among the most volatile asset classes there is; positions can lose value rapidly, and futures-based vehicles can lose money even when spot prices rise.*
