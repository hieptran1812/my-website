---
title: "Natural Gas: Henry Hub, TTF, JKM and the Most Seasonal Market"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why natural gas is three regional markets, not one global price — Henry Hub, TTF and JKM — how storage drives its brutal seasonality, what the widow-maker spread is, and how the 2022 crisis blew the three benchmarks apart."
tags: ["commodities", "natural-gas", "henry-hub", "ttf", "jkm", "lng", "seasonality", "storage", "calendar-spread", "energy-crisis", "europe-gas", "amaranth"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Natural gas is the strangest of the big energy markets: it is not one global price but **three regional ones**, because gas is so hard and expensive to move that for decades each continent cleared its own price behind its own pipelines.
>
> - The three benchmarks are **Henry Hub** (the US price, quoted in USD per MMBtu), **TTF** (the European price, quoted in EUR per MWh) and **JKM** (the Asian LNG spot price, in USD per MMBtu). Liquefied natural gas — LNG — is the tanker that is slowly bridging them.
> - Gas is the **most seasonal** of all the majors. Demand triples in winter for heating but supply is steady, so the gap is plugged by **underground storage**: inject in summer, withdraw in winter. Where inventory sits versus the **five-year band** is the single number that moves the price.
> - The seasonality created the **"widow-maker"** — the March/April calendar spread that blew up the Amaranth hedge fund in 2006 for about **\$6 billion** in a matter of weeks.
> - The one fact to remember: in August 2022, after Russia cut supply, European TTF gas spiked to roughly **€339/MWh** — near **\$100/MMBtu** in energy-equivalent terms — while US Henry Hub sat around **\$6**. That is a **15-to-1** gap, and it is exactly the spread that LNG cargoes now race to arbitrage.

On the morning of 26 August 2022, a single number on a screen in Amsterdam told the story of a continent in trouble. The front-month TTF contract — the price of natural gas for next-month delivery in northwest Europe — printed around **€339 per megawatt-hour**. A year earlier the same contract had traded near €46. Three years earlier it had been under €10. Europe's gas had risen roughly **thirty-five-fold**, and with it went the price of electricity, the survival of fertilizer plants, the heating bills of two hundred million households, and the political nerves of every government on the continent.

Now hold that European number next to the American one. On that same day, US natural gas at the Henry Hub in Louisiana was changing hands around **\$9 per MMBtu** at its 2022 peak, and for most of the year it averaged closer to **\$6**. Convert the European price into the same unit and TTF was the equivalent of roughly **\$100 per MMBtu**. Same molecule. Same week. A price gap of more than **fifteen to one** between two continents.

This could never happen in oil. A barrel of Brent in the North Sea and a barrel of WTI in Texas trade within a few dollars of each other, because oil rides a cheap tanker to wherever it is wanted, and that single tanker route is enough to drag every regional price into line. Gas is different, and understanding *why* it is different — why a fuel can cost fifteen times more on one side of an ocean than the other — is the key that unlocks the entire natural-gas market. The answer is one word: **transport**. Gas is a diffuse, low-density vapour. Moving it is hard, slow, and ruinously expensive, and for a century that meant gas was trapped wherever it came out of the ground. This post is about the consequences of that one physical fact.

![Three regional gas markets US Europe Asia bridged by LNG tankers](/imgs/blogs/natural-gas-henry-hub-ttf-jkm-and-the-most-seasonal-market-1.png)

## Foundations: what natural gas is, and why moving it is so hard

Let us build this from nothing, assuming you know only that gas comes out of the ground and ends up heating your home or running a power plant.

**Natural gas** is mostly methane — a small, light molecule that is a gas at ordinary temperature and pressure. That last clause is the whole story. Oil is a liquid you can pour into a barrel, and a barrel into a tank, and a tank onto a ship; a tanker carries a dense, concentrated cargo. Gas, by contrast, is a vapour: at room conditions it occupies hundreds of times the volume of an energy-equivalent amount of oil. You cannot pour a gas into a barrel. To move it at all, you have only two options, and both are expensive.

The first option is a **pipeline**: a sealed steel tube, often a metre or more across, with compressor stations every hundred kilometres or so to keep the gas pressurised and flowing. A pipeline is fabulously efficient once it exists — but it costs billions to build, takes years to permit and lay, and goes from exactly one place to exactly one other place. A pipeline from Russia to Germany delivers Russian gas to Germany and nowhere else. It cannot redirect a cargo to Japan if Japan is paying more. It is a fixed marriage between one seller and one buyer.

The second option is to **liquefy** the gas. Cool methane to about **−162°C** and it condenses into a liquid — **liquefied natural gas**, or **LNG** — that takes up roughly **1/600th** of its gaseous volume. Now it is dense enough to load onto a specialised refrigerated tanker and ship across an ocean, the same way oil moves. But the liquefaction is brutally costly: you must build a multi-billion-dollar plant to chill the gas, a fleet of cryogenic ships to carry it, and a regasification terminal at the other end to warm it back into a vapour. The full chain — liquefy, ship, regasify — adds several dollars per MMBtu to the cost of the molecule and only pays off over decades.

So for most of the twentieth century the second option barely existed, and gas was **pipeline-bound**. And a pipeline-bound commodity cannot have one world price, because there is no cheap way for a surplus in one region to flow to a shortage in another. That single constraint is what splits gas into three separate markets.

One more piece of foundation before we go further: the **units**, because gas is the one commodity where the measurement is genuinely confusing and the confusion hides the story. Oil is simple — it trades in barrels, everywhere, in dollars. Gas trades in a tangle of different units depending on where you are, and you have to be able to translate between them.

- A **cubic foot** (cf) is a unit of *volume* — how much space the vapour takes up. US production and storage are reported in **billions of cubic feet (Bcf)** and **trillions of cubic feet (Tcf)**.
- A **BTU** (British thermal unit) is a unit of *energy* — the heat released when the gas is burned. Because what a buyer actually pays for is the heat, gas prices are quoted per unit of energy, not per unit of volume. One **MMBtu** is one million BTU (the "MM" is the old Roman-numeral convention for "thousand thousand"). Roughly, **1,000 cubic feet of gas ≈ 1.037 MMBtu**, so to a good approximation a thousand cubic feet and one MMBtu are interchangeable.
- A **megawatt-hour (MWh)** is also a unit of energy — the one Europe prefers, because it lets gas and electricity be quoted in the same currency-per-MWh. As we will use repeatedly, **1 MWh ≈ 3.412 MMBtu**.

The reason this matters is that the three benchmarks are quoted in different units on purpose, and you cannot even *see* the price gap between them until you put them in the same unit. Henry Hub in USD/MMBtu and TTF in EUR/MWh look like numbers from different planets — \$3 versus €40 — until you convert, at which point the true gap leaps out. A gas trader does this conversion reflexively; a newcomer who skips it never sees the arbitrage at all.

#### Worked example: why a barrel travels and a cubic foot does not

Picture two cargoes leaving the US Gulf Coast, both worth the same energy: about ten billion BTUs.

As **oil**, that is roughly 1,700 barrels — call it three rail tank-cars, or a sliver of a normal crude tanker that hauls two million barrels at a time. The freight to ship crude from the US Gulf to Europe runs a couple of dollars a barrel. On a \$75 barrel, that is under **3%** of the value. Oil moves almost for free relative to its worth, so it equalises across regions.

As **gas**, that same ten billion BTUs is about ten million cubic feet of vapour — a volume you simply cannot put on a ship without first spending the money to chill it to a liquid. The liquefaction-plus-shipping-plus-regas chain can cost **\$3 to \$5 per MMBtu**. On a \$3 Henry Hub molecule, that is more than **100%** of the value of the gas itself. You are spending more to move the gas than the gas is worth at the wellhead.

The intuition: oil's transport cost is a rounding error on its price, so one tanker route unifies the world; gas's transport cost can exceed the price of the gas, so the molecule stays trapped where it is born — and three islands form.

## The three benchmarks: Henry Hub, TTF and JKM

Because gas is pipeline-bound, each major consuming region grew its own price, set by its own local balance of supply and demand, and quoted at its own delivery point in its own unit. There are three that matter.

**Henry Hub** is the American benchmark. It is a physical place: a pipeline interchange in Erath, Louisiana, where a dozen major interstate and intrastate pipelines meet. Because so much gas physically changes hands there, its price became *the* reference for North American gas, and the NYMEX natural-gas futures contract settles against it. Henry Hub is quoted in **US dollars per MMBtu** — one MMBtu being one million British thermal units, a measure of energy content rather than volume. (Energy units matter here because what a buyer actually wants is the heat, not the cubic feet.) For most of the last decade Henry Hub has been the cheapest gas in the world, because the US shale boom flooded the country with supply.

**TTF** — the *Title Transfer Facility* — is the European benchmark. Unlike Henry Hub it is not a physical hub but a **virtual** one: a notional trading point within the Dutch gas network where ownership of gas can change hands without it physically moving. TTF won out over older hubs (like the UK's NBP) to become the continent's reference price, and it is quoted in **euros per megawatt-hour (EUR/MWh)** — Europe prices its gas in the same energy unit it prices its electricity, which is no accident, because gas and power are joined at the hip. TTF is the price that exploded in 2022.

**JKM** — the *Japan/Korea Marker* — is the Asian benchmark, and it is the youngest and most telling of the three. Northeast Asia (Japan, Korea, China, Taiwan) has almost no domestic gas and almost no pipelines reaching it, so it imports nearly all its gas as **LNG by ship**. JKM, published by the price agency Platts, is the spot price of an LNG cargo delivered into northeast Asia, quoted in **US dollars per MMBtu**. JKM is, in effect, the price of the marginal seaborne cargo in the Pacific — and because it is a *delivered LNG* price rather than a pipeline-hub price, it is the benchmark that most directly competes with TTF for any spare cargo on the water.

There is a fourth detail worth knowing: how each benchmark is *delivered*, because that shapes its behaviour. Henry Hub and TTF are **hub** prices — they reference gas at a fixed point in a pipeline network, so they carry the local pipeline balance directly. JKM is a **delivered-ex-ship (DES)** price — it references an LNG cargo arriving at a northeast-Asian port, so it already bundles in the shipping and the global cargo competition. That difference is why JKM and TTF move together (both compete for the same floating cargoes) while Henry Hub, sitting at the start of the export chain rather than the end, can stay cheap and detached. Henry Hub is the *source* price; JKM is the *destination* price; TTF is somewhere in between, a pipeline hub that has been forced to behave like a destination price ever since Europe lost its own pipeline supply and had to start importing cargoes.

There is also a historical pricing wrinkle that explains why Asia was so exposed. For decades, most Asian LNG was sold on **long-term contracts indexed to the oil price** (a leftover from the era when LNG had no liquid spot market of its own and buyers wanted a familiar benchmark). JKM, the *spot* LNG price, emerged only as a real traded market grew — and in a crisis, it is the spot price, not the old oil-linked contract price, that does the violent moving. So the rise of JKM as a benchmark is itself part of the story of gas globalising: a region that used to price gas off oil now has its own traded marker.

Three benchmarks, three units, three regions. The chart below puts all three into the same unit — US dollars per MMBtu — so you can see them on one axis. The story it tells is the story of this whole post.

![Three regional gas benchmarks Henry Hub TTF JKM in USD per MMBtu 2021 to 2024](/imgs/blogs/natural-gas-henry-hub-ttf-jkm-and-the-most-seasonal-market-2.png)

Notice three things. First, **the levels are wildly different** — Henry Hub sits at the bottom in every year, while TTF and JKM ride far above it. Second, **TTF and JKM track each other closely**: both are seaborne-LNG-linked prices competing for the same cargoes, so when Europe bids up, Asia must match it or lose the ship. Third, and most dramatic, **2022 is a chasm** — US gas barely moved while European and Asian gas quadrupled. That gap is not a market failure; it is the precise, measurable cost of the fact that you cannot instantly ship a US surplus into a European shortage. LNG narrows the gap over time but cannot close it overnight, because there are only so many liquefaction plants and so many ships.

To compare these prices at all, you have to convert between the units, and that conversion is itself a worked example every gas trader does in their head.

#### Worked example: converting TTF in EUR/MWh to USD/MMBtu

Take the August-2022 TTF peak of **€339/MWh** and translate it into the American unit so you can compare it to Henry Hub.

Step one, the energy conversion. One megawatt-hour of energy equals about **3.412 MMBtu** (since 1 MWh = 3,412,000 BTU, and 1 MMBtu = 1,000,000 BTU). So a price *per MWh* must be **divided** by 3.412 to get a price *per MMBtu*:

```
339 EUR/MWh  /  3.412 MMBtu per MWh  =  99.4 EUR/MMBtu
```

Step two, the currency conversion. With the euro around **1.08** US dollars in 2022:

```
99.4 EUR/MMBtu  x  1.08 USD/EUR  =  107.3 USD/MMBtu
```

So Europe's panic price was roughly **\$107 per MMBtu**. On that same day Henry Hub was near **\$9**. The molecule was worth about **twelve times more** in Rotterdam than in Louisiana.

The intuition: the gap between TTF and Henry Hub, once you put them in the same unit, is a giant flashing dollar sign that says *load a US cargo, sail it to Europe, pocket the spread* — and that incentive is the engine of the entire LNG business.

That conversion factor — divide EUR/MWh by 3.412, then multiply by the EUR/USD rate — is worth memorising, because the European and American gas prices are quoted in units that look incomparable until you do it, and the whole arbitrage story is invisible until you can stand them side by side.

## How the US price came to live at the bottom

Before we get to seasonality, look at the long arc of the American price, because it explains why Henry Hub is the cheap anchor at the bottom of every comparison.

![Henry Hub natural gas annual average 2005 to 2025](/imgs/blogs/natural-gas-henry-hub-ttf-jkm-and-the-most-seasonal-market-3.png)

In the mid-2000s, American gas traded around **\$7 to \$9 per MMBtu**, and the US was building *import* terminals because everyone assumed the country was running out of gas and would have to buy LNG from abroad. Then the shale revolution — the same horizontal-drilling-and-hydraulic-fracturing breakthrough that flooded the world with US oil — did the same thing to gas, except even harder. Vast quantities of gas came up alongside the shale oil (so-called "associated gas," produced almost as a by-product), and dedicated gas shales like the Marcellus in Appalachia added a torrent more. Supply exploded, and the price collapsed from roughly **\$8** to roughly **\$2**, where it has mostly stayed.

The 2022 spike to a **\$6.42** annual average is the lone interruption, and even that was modest by global standards — it was the US *exporting* its tightness to the world through LNG, not importing a crisis. By 2024 Henry Hub was back near **\$2.21**, a multi-decade low in real (inflation-adjusted) terms. America is now drowning in gas, which is precisely why it became the world's largest LNG exporter and why the US price anchors the bottom of the global stack.

This is the supply side of the spine of this whole series: a commodity price is a physical thing forced through a financial contract, and when the physical supply of that thing triples, the price falls until the marginal molecule is barely worth pulling out of the ground. Henry Hub at \$2 is the shale glut written as a number. (For the broader story of how shale reshaped the oil market too, see [Crude Oil: WTI vs Brent](/blog/trading/commodities/crude-oil-wti-vs-brent-the-worlds-two-benchmark-barrels) within this series, and for how energy prices feed inflation, [Energy: Oil & Gas, the Inflation Engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine).)

## The most seasonal market on earth

Here is the feature that makes gas unlike every other commodity: its demand is violently seasonal, but its supply is not.

Think about *why* you use gas. In a cold country, the dominant use is **heating** — homes, offices, schools. Heating demand is near zero in July and enormous in January. Add the second big use, **power generation**, which has its own (usually smaller, summer-skewed) peak when air conditioners run hard in a heatwave. The result is a demand curve that can more than **double** between a mild shoulder month and a deep cold snap.

Now look at supply. Gas comes out of wells at a steady, roughly constant rate, because you cannot easily turn a gas field on and off with the seasons — the wells produce what they produce. So you have steady supply meeting demand that lurches up and down by a factor of two or three. Something has to bridge that gap, and that something is **storage**.

![Storage seasonality inject in summer withdraw in winter five-year band](/imgs/blogs/natural-gas-henry-hub-ttf-jkm-and-the-most-seasonal-market-5.png)

Gas is stored underground at enormous scale — in depleted oil and gas reservoirs, in salt caverns washed out of underground salt domes, and in porous aquifers. The annual rhythm is fixed and universal:

- **Injection season (roughly April to October):** demand is mild, so the surplus from steady production gets pumped *into* storage. Inventories climb week after week. The market is loose; prices tend to be soft.
- **Peak stocks (around early November):** storage is near full — typically the US fills to somewhere around 3.5 to 4 trillion cubic feet — and the winter cushion is built.
- **Withdrawal season (roughly November to March):** heating demand spikes, production alone cannot meet it, and gas is drawn *out* of storage to cover the gap. Inventories fall week after week.
- **Trough stocks (around late March):** storage hits its annual low, and the market turns its attention to whether the coming summer's injection will refill it in time.

The number every gas trader watches is the **weekly storage report**. In the US, the Energy Information Administration publishes the change in gas inventories every Thursday at 10:30 a.m. Eastern, and the market often moves violently on the release. But the level alone is meaningless without context, so traders compare it to the **five-year band** — the range (minimum to maximum) of storage at this same week over the previous five years. When inventories sit *above* the five-year band, the market is comfortably supplied and prices are calm. When they fall *below* the band — when a cold winter has drawn storage down faster than usual — the market panics that it might run physically short before spring, and the price spikes.

This is why a cold snap can send gas up double digits in a day while oil shrugs. A forecast of an unusually cold week is a forecast of a faster storage draw, which is a forecast of inventories breaking *below* the band, which is a forecast of physical scarcity. The price reacts to the threat, not the event. And because storage can only release gas so fast (there is a maximum daily withdrawal rate, set by the physical deliverability of the wells and caverns), a deep cold snap colliding with already-low storage is the recipe for a genuine spike.

#### Worked example: a cold-winter storage draw and the price it implies

Suppose the US enters December with storage at **3.2 trillion cubic feet (Tcf)** — already a bit below the five-year average of, say, 3.5 Tcf. A normal winter draws roughly **2.0 Tcf** over the season, which would leave about **1.2 Tcf** at the end of March — a thin but survivable cushion.

Now a polar vortex parks over the country for three weeks. Daily withdrawals run hot — instead of the seasonal-average pace, the market draws an extra **0.4 Tcf** above normal in that stretch. Project the season forward and end-of-March storage now points toward roughly **0.8 Tcf** — uncomfortably close to the physical floor below which pipelines cannot maintain pressure.

What does the price do? It does not wait for March. The moment the forecasts turn cold, the front-month contract reprices to *ration demand now* — to make gas expensive enough that some power generators switch to coal and some industrial users throttle back, slowing the draw. A Henry Hub that was sitting at **\$3** can leap to **\$6 or \$8** on the threat alone. In the extreme — Winter Storm Uri in February 2021 — spot gas at some US hubs briefly traded above **\$100/MMBtu** and even **\$400** at a few constrained points, as wells froze and storage could not deliver fast enough.

The intuition: in gas, the price is not paying for the molecule you burn today; it is paying for the *insurance* that there will still be a molecule in March — and that insurance gets repriced the instant the weather forecast changes.

That seasonality is not noise to be smoothed away. It is the defining structural feature of the market, and it shows up directly in the shape of the forward curve as a **seasonal hump** — winter contracts trade at a premium to summer contracts, every single year, for the obvious physical reason. The illustrative shape below, anchored to the most recent Henry Hub level, shows the pattern.

![Illustrative within-year seasonal price hump for natural gas](/imgs/blogs/natural-gas-henry-hub-ttf-jkm-and-the-most-seasonal-market-6.png)

The winter months sit above the annual-average line (the red region — buyers pay a premium for delivery into the cold), and the spring and early-summer shoulder months sit below it (the green region — gas is cheapest when nobody needs heat and storage is being refilled). This shape is *not* a forecast that prices will rise; it is a stable, recurring premium that the calendar itself puts into the curve. A trader who understands it does not get fooled into thinking "winter gas is expensive, the market expects a shortage" — the winter premium is simply the price of the season. (For the general theory of why curves have these shapes, see [The Forward Curve](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) within this series.)

### The demand side: degree-days, power burn, and switching

To trade the seasonal pattern you have to understand what actually drives the demand swing, and it comes down to three measurable things.

The first is **weather**, quantified by **degree-days**. A *heating-degree-day* (HDD) counts how far the average daily temperature falls below a reference (commonly 65°F / 18°C): a day averaging 45°F is 20 HDD, a mild day near 65°F is zero. Gas heating demand is almost linear in HDD, so the weekly weather forecast translates directly into a gas-demand forecast, which translates into a storage-draw forecast. This is why gas desks employ meteorologists and why the gas price can lurch on a single revision to the ten-day temperature outlook. In summer the mirror image, *cooling-degree-days* (CDD), drives the air-conditioning load that lifts gas-fired power demand. Gas is the commodity most directly priced off the weather map.

The second is **power burn** — the volume of gas consumed by electricity generators. In the US, gas is now the largest single source of power, so a hot summer that runs air conditioners hard pulls a surprising amount of gas into power plants even out of heating season, smoothing the old purely-winter demand shape into something with a secondary summer bump. (The illustrative seasonal chart above shows that mild summer lift.)

The third, and the most important for the *floor* under the price, is **fuel switching**. Power generators that can burn either gas or coal will switch to whichever is cheaper per MWh of electricity. When gas falls cheap relative to coal, generators burn more gas, which lifts gas demand and props up its price — a built-in demand cushion. When gas gets expensive relative to coal, generators switch back to coal, which caps gas demand and pulls the price back down. This switching range — the band of gas prices over which generators flip between fuels — acts like a soft floor and ceiling on the gas price during normal times. It is one reason Henry Hub spends so much time in a \$2-to-\$4 channel: below \$2, switching demand floods in and lifts it; above roughly \$4-\$5, generators flee to coal and cap it. Only when storage anxiety overwhelms switching — a cold snap with low inventories — does the price break out of that band.

#### Worked example: turning a degree-day forecast into a storage draw

Suppose the ten-day forecast suddenly turns colder, adding **80 heating-degree-days** versus the prior outlook for the week. A common rule-of-thumb in the US market is that each incremental HDD adds very roughly **3 to 4 Bcf** of national gas demand for heating.

```
80 extra HDD  x  3.5 Bcf per HDD  =  ~280 Bcf of extra demand in the week
```

That 280 Bcf has to come from somewhere, and with production flat it comes out of storage. A typical mid-winter weekly draw might be **200 Bcf**; this forecast change implies a draw closer to **480 Bcf** — more than double. If storage was already tracking near the bottom of the five-year band, a draw that size threatens to punch *through* the band, and the front-month price reprices upward immediately to choke off demand by pushing generators back to coal.

The intuition: in gas, a weather forecast is a demand forecast is a storage forecast is a price — the chain is short and mechanical, which is why the gas market reacts to the sky faster than any other commodity.

## The widow-maker: when seasonality becomes a weapon

Once you understand that winter gas is structurally dearer than summer gas, you have understood the most famous and most dangerous trade in the entire commodity complex: the **March/April calendar spread**, known on every gas desk as the **"widow-maker."**

A **calendar spread** is the price difference between two delivery months of the same commodity — you buy one month and sell another, betting on how the *gap* between them moves rather than on the outright price. The March/April spread is special because it straddles the single most important seam in the gas calendar: **March is the last contract of winter** (the end of withdrawal season, when storage is at its lowest and a late cold snap can still cause a scramble) and **April is the first contract of summer** (the start of injection season, when the heating crisis is over and the market relaxes). So the March-minus-April spread is, at its core, the market's bet on *how the winter ends*.

In a normal year, March trades at a modest premium to April — winter's last gasp is worth a bit more than spring's first breath. But the spread is explosive, because the two contracts respond to completely different forces. If the winter ends mild and storage finishes the season comfortable, March collapses toward April and the spread shrinks toward zero. But if a brutal late-winter cold snap hits while storage is already drained, March can blow out to a huge premium over April, because there is simply no time left in the season to refill before that March gas must be delivered. The spread can move from \$0.50 to \$5.00 — a tenfold move — in a few weeks, and there is no natural ceiling, because if the physical gas isn't there, the price of the last winter contract is theoretically unbounded.

That asymmetry — bounded on one side, unbounded on the other, with a fuse lit by the weather — is what makes it a widow-maker. And it has the body count to prove it.

#### Worked example: the Amaranth blowup and a calendar-spread P&L

In 2006, a hedge fund called **Amaranth Advisors** had a star natural-gas trader, Brian Hunter, who built an enormous position in calendar spreads — including a massive bet that the March 2007 contract would rise relative to April 2007 (a wager that the winter would end tight). At its peak the fund's gas book was reportedly tens of thousands of contracts, controlling a notional exposure in the tens of billions of dollars.

Walk the math of a single spread to see the leverage. One NYMEX gas contract is **10,000 MMBtu**. Suppose you are long the March/April spread at **\$2.00** (you have bet March will be \$2.00 dearer than April). If the spread *widens* by \$1.00 to \$3.00, you make:

```
spread move  x  contract size  =  P&L per spread
1.00 USD/MMBtu  x  10,000 MMBtu  =  10,000 USD per spread (gain)
```

But the market went the *other* way. A mild autumn and ample storage caused the March/April spread to **collapse**. If your \$2.00 spread falls to \$1.00 — a \$1.00 adverse move — you lose \$10,000 per spread. Now scale it: a fund holding the equivalent of, say, **60,000 spreads** loses

```
1.00 USD/MMBtu  x  10,000 MMBtu  x  60,000 spreads  =  600,000,000 USD
```

— \$600 million on a single dollar of spread movement. The actual collapse was far larger than \$1.00, and Amaranth's position was even bigger and far less liquid than this stylised count. Over a few weeks in September 2006 the fund lost roughly **\$6 billion** — about two-thirds of its capital — and collapsed. It remains one of the largest hedge-fund implosions in history, and it was caused by exactly the seasonal calendar spread we just described.

The intuition: the widow-maker is dangerous not because the *direction* is hard to guess but because the *size* of the move is unbounded and the position is illiquid — when seasonality turns against a large, concentrated book, there is no door wide enough to get out.

The lesson generalises beyond one fund. Seasonal calendar spreads in gas carry a kind of tail risk that outright price bets do not: a position that looks like a calm, range-bound carry trade for months can, in a single cold snap, move further and faster than any stop-loss can catch. Respect the widow-maker.

## The 2022 European crisis: when one region's price broke loose

Everything so far — three benchmarks, brutal seasonality, the storage band — came together in 2022 into the most violent gas price event in modern history, and it is the cleanest illustration of why gas is three markets and not one.

![TTF Dutch gas the 2022 European crisis EUR per MWh](/imgs/blogs/natural-gas-henry-hub-ttf-jkm-and-the-most-seasonal-market-4.png)

For decades, Europe ran on cheap pipeline gas from Russia — at the peak, Russian pipelines supplied roughly **40%** of the EU's gas. That arrangement was the embodiment of the pipeline marriage: a fixed route, a fixed dependency, no easy substitute. When Russia invaded Ukraine in February 2022 and then progressively throttled and finally cut those pipeline flows over the course of the year (the Nord Stream pipelines stopped delivering and were later sabotaged), Europe lost its largest supplier almost overnight — and there was no pipe from anywhere else to replace it.

The only replacement available was **LNG by ship**. But Europe had to *outbid the rest of the world* — primarily Asia — for every spare cargo on the water, because the global LNG fleet and liquefaction capacity are finite and were already spoken for. So TTF did not rise to the cost of producing gas; it rose to whatever price was needed to *pull cargoes away from Asian buyers and toward European terminals*. That is why TTF blew through €100, €200, and at the August peak roughly **€339/MWh** — about **\$100/MMBtu** equivalent. The price was not measuring scarcity of gas in the ground; it was measuring the scarcity of *ships and terminals* able to bring gas to a continent that had just lost its pipeline.

Meanwhile, on the other side of the Atlantic, **Henry Hub barely noticed**. US gas peaked near \$9 and averaged about \$6.42 for the year — elevated for America, but a fraction of Europe's price. The US had its own glut and only limited LNG export capacity, so its surplus *could not physically reach Europe fast enough* to equalise the prices. The gap between TTF and Henry Hub in 2022 was the single most expensive demonstration in history of the central fact of this post: **gas is pipeline-bound, and when one region's pipeline supply breaks, its price can decouple from everyone else's by an order of magnitude.**

#### Worked example: the LNG arbitrage that 2022 set off

Stand in the shoes of a US LNG exporter in mid-2022 and do the arithmetic that the whole industry was doing.

You can buy US gas at Henry Hub for about **\$8/MMBtu** (elevated, mid-2022). Your all-in cost to liquefy it, ship it across the Atlantic, and regasify it in Europe is roughly **\$4/MMBtu**. So your *delivered cost* into Europe is about:

```
8 USD/MMBtu (gas)  +  4 USD/MMBtu (liquefy + ship + regas)  =  12 USD/MMBtu delivered
```

And you can sell it into TTF at the equivalent of roughly **\$70 to \$100/MMBtu**. Your gross margin per MMBtu is:

```
70 USD/MMBtu (TTF sale)  -  12 USD/MMBtu (delivered cost)  =  58 USD/MMBtu
```

On a single standard LNG cargo of about **3.5 trillion BTU** (3.5 million MMBtu), that is a gross spread of roughly:

```
58 USD/MMBtu  x  3,500,000 MMBtu  =  ~200 million USD per cargo
```

A two-hundred-million-dollar margin on one boat. That is why, in 2022, dozens of LNG tankers physically turned around mid-ocean — cargoes originally bound for Asia changed course toward Europe, because Europe was paying more. The arbitrage was so enormous it bent the global shipping map.

The intuition: a price gap between regions is not a curiosity for gas — it is a profit signal that mobilises a fleet of ships, and that fleet is precisely the mechanism slowly turning three island markets into one connected web. (For more on this — the rise of LNG as the great connector — see [LNG and the Globalization of Gas](/blog/trading/commodities/lng-and-the-globalization-of-gas-shipping-the-uncontainable) within this series.)

The crisis did eventually ease. Europe slashed demand, filled its storage to the brim before the winters of 2022–23 and 2023–24, built floating import terminals at record speed, and rode unusually mild weather. By 2024 TTF had fallen back to around €34/MWh — still well above the pre-crisis norm, but a world away from €339. The episode is the case study for how a regional gas market behaves when its pipeline supply is severed and only the slow, capacity-constrained LNG bridge stands between it and a true physical shortage. (It sits alongside the other great supply break of that era; for the metals version, see [The 2022 Energy Crisis and the Nickel Blowup](/blog/trading/commodities/the-2022-energy-crisis-and-the-nickel-blowup-when-supply-broke) within this series.)

## Why gas links to power and fertilizer

Two more connections turn gas from a heating fuel into a systemic price, and both explain why a gas spike ripples far beyond home heating bills.

The first link is **electricity**. In most modern grids, gas-fired power plants are the **marginal** generators — the ones switched on last to meet the final increment of demand, and therefore the ones that *set the wholesale electricity price* for everyone. This is why Europe quotes both gas and power in the same EUR/MWh unit: when TTF gas quadrupled in 2022, European wholesale electricity quadrupled in lockstep, because the gas plant at the margin was setting the price of every kilowatt-hour. A gas crisis is automatically a power crisis. The pass-through is mechanical, not coincidental.

The second link is **fertilizer**. The dominant nitrogen fertilizer, ammonia, is made by combining hydrogen with nitrogen, and the cheapest source of that hydrogen is — natural gas. For an ammonia plant, gas is not just the fuel; it is the **feedstock**, and it can be **70–80%** of the total cost of making the fertilizer. So when European gas spiked in 2022, ammonia plants across the continent simply *shut down* — it was cheaper to stop making fertilizer than to make it at €339 gas — and the price of fertilizer soared worldwide, which fed straight into the cost of growing every crop on earth. A European gas crisis became a global food-cost problem through this single chain.

#### Worked example: how much gas it takes to make a tonne of fertilizer

Making one tonne of ammonia requires roughly **33 to 36 MMBtu** of natural gas as feedstock and fuel. So the gas cost of a tonne of ammonia is just the gas price times that intensity.

At a calm US Henry Hub price of **\$3/MMBtu**:

```
3 USD/MMBtu  x  34 MMBtu per tonne  =  ~102 USD of gas per tonne of ammonia
```

At the 2022 European TTF-equivalent of **\$70/MMBtu**:

```
70 USD/MMBtu  x  34 MMBtu per tonne  =  ~2,380 USD of gas per tonne of ammonia
```

The gas input alone went from about \$100 to about \$2,400 per tonne — a **twenty-three-fold** jump in the single biggest cost of the product. No wonder European plants idled while US plants (paying the \$3 price) kept running and gained a structural advantage.

The intuition: gas is not just a fuel you burn — it is embedded as a raw material in things you would never guess, so a gas price shock propagates into electricity, food, plastics and steel far beyond the gas bill itself.

This embeddedness is the deep reason gas matters to anyone allocating capital across markets. A gas price is an input to the price of power, food, and a large slice of industrial production, which is why it shows up as a macro signal and an inflation driver. (For the cross-asset and macro angle, see [Energy: Oil & Gas, the Inflation Engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) and how unscheduled supply shocks transmit, [Geopolitics, Elections and Unscheduled Shocks](/blog/trading/event-trading/geopolitics-elections-and-unscheduled-shocks).)

## Why one global price never formed — and what LNG changes

Step back and put the structural picture in one frame. The reason gas had three prices, and oil has roughly one, is entirely about the physics of transport, and the reason that is now slowly changing is entirely about LNG.

![Why gas had three prices and how LNG bridges them](/imgs/blogs/natural-gas-henry-hub-ttf-jkm-and-the-most-seasonal-market-7.png)

In the **pipeline era** (the left half of the figure), gas moved only along fixed routes between fixed pairs of buyers and sellers. Each region was an island. A glut in Texas had no path to reach a freezing winter in Germany, so the two prices were free to diverge and *stay* diverged for years. There was no arbitrage mechanism to pull them together, because there was no ship.

In the **LNG era** (the right half), the picture changes — slowly. A cargo of LNG, once liquefied, can in principle sail to whichever region is paying the most. That mobility creates, for the first time, an arbitrage path between the three markets: when TTF rises far above JKM, cargoes divert toward Europe; when Asia pays up, they swing back. Over time, that re-routing pulls the regional prices toward each other, capped by the cost of shipping the LNG. The three benchmarks now *move together* far more than they used to — but only within a band set by liquefaction and shipping capacity, which is why the link is real but loose, and why a sudden shock like 2022 can still blow the prices apart faster than the ships can respond.

The crucial caveat — and the difference from oil — is **capacity**. Oil's tanker fleet is vast, flexible, and cheap to redeploy, so oil equalises almost instantly. LNG's liquefaction plants and specialised tankers are scarce, expensive, and take years to build, so the LNG bridge has a low, fixed bandwidth. When demand for the bridge exceeds its capacity — as it did in 2022 — the prices decouple until new capacity arrives. The world is converging toward a more unified gas price, but it will get there at the speed of concrete and steel, not at the speed of a trader's screen.

There is also an asymmetry in *which way* the bridge flexes. A US export terminal, once built, runs flat out almost regardless of price, because its owner signed long-term, take-or-pay contracts to finance the multi-billion-dollar plant — the gas flows whether the spot arbitrage is wide or narrow. What flexes in the short run is the *destination*: where the cargoes already on the water choose to land. So the LNG fleet does not so much expand and contract with the regional spread as it *re-routes* with it — cargoes already committed swing between the Atlantic basin (toward TTF) and the Pacific basin (toward JKM) depending on which is paying more, with the cost of the longer voyage (and the canal tolls) as the tie-breaker. The total volume of LNG on the water is roughly fixed in any given month; the spread decides who gets it. That re-routing is the fast arbitrage; the slow one — actually building new liquefaction trains — takes the better part of a decade and is what ultimately raises the bridge's bandwidth.

This is also why the US, having gone from essentially zero LNG exports in 2015 to the world's largest exporter by 2023, has quietly become the **swing supplier** to the entire seaborne market. US cargoes, sold on flexible destination terms rather than locked to one buyer, are the ones most free to chase the highest price — which means American gas, the cheap anchor at the bottom of the stack, is increasingly the marginal molecule that *sets* the floor of the global price. The cheapest gas in the world has become the gas that the rest of the world's prices are quietly tethered to, through the cost of liquefying and shipping it. That tether is loose and capacity-limited, but it is real, and it is tightening year by year.

This is the natural-gas version of the series' spine. A gas price is a physical molecule — methane, hard to move — forced through a financial contract. The forward curve's seasonal hump is the calendar's signature; the storage band is the inventory's signature; and the gap between the three benchmarks is the transport constraint's signature. Read those three things and you have read the physical reality of gas off the screen. (Gas is an industrial and consumption commodity, not a monetary one — for the contrast with the one metal that *is* money, see [Is Gold Money?](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything).)

## Common misconceptions

**"There is one world gas price, like oil."** No — and this is the single most important thing to unlearn. In 2022, European TTF gas was the equivalent of roughly **\$100/MMBtu** while US Henry Hub was around **\$6**, a gap of more than fifteen to one. Oil never does this because oil ships cheaply; gas does it routinely because gas does not. There are three benchmarks, and they can diverge by an order of magnitude.

**"A high winter gas price means the market expects a shortage."** Mostly no. Winter contracts trade at a structural premium to summer contracts *every single year* because of heating demand and the storage cycle — that seasonal hump is the calendar's signature, not a forecast. The actual signal of distress is not "winter is dearer than summer" (always true) but "storage is below the five-year band" or "the front month has detached from the seasonal pattern."

**"Henry Hub is the global gas price."** No — it is the *American* price, and it is usually the *cheapest* in the world because the US is awash in shale gas. The price that the rest of the world's traded cargoes actually clear against is JKM (Asia) and TTF (Europe), both of which are LNG-linked and typically far above Henry Hub. Watching only Henry Hub shows you the US market and misses the global one entirely.

**"Calendar spreads are a safe, range-bound carry trade."** Tell that to Amaranth. The March/April widow-maker can move tenfold in weeks and has no natural ceiling, because the price of the last winter contract is unbounded if the physical gas isn't there. A position that behaves like calm carry for months can move further than any stop-loss in a single cold snap. Seasonal spreads in gas carry a hidden tail that outright price bets do not.

**"LNG has made gas a single global commodity now."** Not yet, and maybe not for a long time. LNG creates an arbitrage *path*, but the bridge has a low, fixed bandwidth — liquefaction plants and ships are scarce and take years to build. When demand for the bridge exceeds its capacity, as in 2022, the regional prices decouple anyway. The convergence is real but partial and slow, capped by steel and concrete.

**"A cheap gas price is always good for the producer."** Not for the US gas industry, which spent the 2010s drilling itself into a glut so deep that Henry Hub sank to multi-decade lows in real terms and many pure gas producers struggled to make money at \$2. The escape valve was *exports*: LNG let the cheapest gas in the world find higher-paying buyers abroad, which is the only reason a \$2 domestic price and a healthy producer can coexist. A producer's fortunes depend not just on the local price but on whether there is a pipe — or a tanker — to somewhere that pays more. Cheap gas with no way out is a trap; cheap gas with an export terminal is an arbitrage.

## How it shows up in real markets

A few episodes, with dates and numbers, show the structure in action.

**Winter Storm Uri, February 2021.** A polar vortex plunged into Texas — a state that does not winterise its wells or pipes for deep cold. Wellheads and gathering lines literally froze, cutting production at the exact moment heating demand spiked. Storage could not deliver fast enough through the constrained system. Spot gas at some hubs briefly traded above **\$100/MMBtu**, and at a few constrained points reportedly near **\$400**, while the front-month futures (which average over the month) stayed far lower. The episode is the textbook case of the storage-deliverability limit: when the maximum daily withdrawal rate is the binding constraint, the price has no ceiling.

**The 2022 European energy crisis.** Covered above: TTF to ~€339/MWh in August 2022 (~\$100/MMBtu equivalent) on the Russian supply cut, while Henry Hub averaged ~\$6.42. The defining demonstration that gas is three markets, that the LNG bridge has finite capacity, and that a pipeline-dependent region is acutely vulnerable when its pipe is cut.

**The 2006 Amaranth collapse.** The March/April calendar spread moved against a concentrated, illiquid position, and a fund lost roughly **\$6 billion** in weeks. The case study for the widow-maker's tail risk — the danger is the unbounded, illiquid size of the move, not the direction of the bet.

**The shale crush, 2008–2020.** Henry Hub fell from roughly **\$8** to roughly **\$2** as the shale boom flooded the US with associated and dedicated gas. The supply-side story behind why America is the cheap anchor of the global stack — and why it pivoted from building import terminals to becoming the world's largest LNG *exporter*, going from essentially zero LNG exports in 2015 to the world's largest by 2023.

**Europe's mild-winter recovery, 2023–24.** Having lost Russian pipeline gas, Europe filled storage to the brim, built floating import terminals (FSRUs) in months rather than years, cut demand, and caught two mild winters. TTF fell from €339 to around €34 by 2024. The case study for how a regional market re-equilibrates after a supply break — through demand destruction, storage discipline, and LNG, not through any quick return of the lost pipeline.

## The playbook: the gas trader's dashboard

If you remember nothing else, remember that reading gas comes down to three instruments on one dashboard.

**1. The storage report versus the five-year band.** This is the master gauge. Every Thursday, the EIA prints the weekly change in US gas inventories. The level alone tells you little; the level *relative to the five-year band* tells you everything. Above the band = comfortably supplied, prices calm. Below the band = scarcity fear, prices primed to spike on any cold forecast. In Europe, watch the EU gas-storage fill level the same way, especially heading into winter — the percentage full on 1 November is the continent's single most-watched gas number. Storage is the buffer between steady supply and seasonal demand, so storage *is* the balance.

**2. The calendar spread, especially March/April.** The slope of the curve between two months is the cleanest reading of tightness, and the March/April spread is the concentrated bet on how winter ends. Watch it widen and you are watching the market price in a tight finish to the heating season; watch it collapse and you are watching the all-clear. And respect it: it is the widow-maker for a reason, because its move is unbounded on the upside and the positions are illiquid.

**3. The regional spread — TTF minus Henry Hub, JKM minus TTF.** The gaps between the three benchmarks are the arbitrage signal that drives the LNG fleet. A wide TTF-minus-Henry-Hub spread says cargoes will sail toward Europe; a JKM premium over TTF says they will swing toward Asia. These spreads are bounded above by the cost of shipping LNG (liquefaction + freight + regas, a few dollars per MMBtu) — when a spread blows far past that cost, as in 2022, it is telling you the *bridge itself* is at capacity, not that the gas is gone.

Put the three together and you can read the physical state of the gas market off the screen. Storage tells you how much cushion there is; the calendar spread tells you how the season is resolving; the regional spread tells you which way the ships are pointing. Outright price — the single Henry Hub number that makes the headlines — is the *least* informative of the four, because it blends all three signals into one figure and hides the structure underneath.

And that structure is the lasting insight. Gas is the commodity where the physical reality is least hidden by the financial wrapper. In oil, one cheap tanker erases geography and you get a near-single world price. In gas, geography wins: the molecule is expensive to move, so the map *is* the market, the calendar *is* the curve, and the storage caverns *are* the price. Learn to read the three regional prices, the seasonal storage cycle, and the calendar spread, and you are not reading a chart — you are reading the physical balance of methane across three continents, which is exactly what a gas price has always been.

## Further reading & cross-links

Within this series:

- [The Forward Curve: The Most Important Chart in Commodities](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — why curves have seasonal humps and what contango vs backwardation mean.
- [LNG and the Globalization of Gas: Shipping the Uncontainable](/blog/trading/commodities/lng-and-the-globalization-of-gas-shipping-the-uncontainable) — the tanker bridge that is slowly unifying the three markets.
- [The 2022 Energy Crisis and the Nickel Blowup: When Supply Broke](/blog/trading/commodities/the-2022-energy-crisis-and-the-nickel-blowup-when-supply-broke) — the other great supply break of the same era.
- [Crude Oil: WTI vs Brent, the World's Two Benchmark Barrels](/blog/trading/commodities/crude-oil-wti-vs-brent-the-worlds-two-benchmark-barrels) — the contrast: why oil has one world price and gas has three.

Beyond this series:

- [Energy: Oil & Gas, the Inflation Engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) — how energy prices feed inflation and ripple across assets.
- [Geopolitics, Elections and Unscheduled Shocks](/blog/trading/event-trading/geopolitics-elections-and-unscheduled-shocks) — how an unscheduled supply cut like Russia's transmits through markets.
- [Is Gold Money, a Commodity, or a Currency?](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) — the industrial-vs-monetary contrast: gas is pure consumption, gold is not.
