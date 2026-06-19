---
title: "Weather, the WASDE, and the Supply Shock: Trading the Report"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a grain market can move limit-up in seconds on a government report, why weather is the variable behind it all, and how the stocks-to-use ratio turns a 2% supply miss into a 20% price move."
tags: ["commodities", "agriculture", "grains", "wasde", "usda", "stocks-to-use", "supply-shock", "weather-market", "corn", "soybeans", "wheat", "crop-report"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A grain price is the answer to one question USDA asks every month: after we feed, eat, burn, and export the crop, how much is left over? That leftover, divided by use, is the stocks-to-use ratio, and it maps to price along a steep, convex curve.
>
> - The **WASDE** (World Agricultural Supply and Demand Estimates) is the monthly balance sheet for each crop: **beginning stocks + production − use = ending stocks**, and **ending stocks ÷ annual use = the stocks-to-use ratio**, the single tightness gauge.
> - Grain demand is **inelastic** — people and animals eat about the same amount whatever the price — so a small supply surprise needs a *large* price move to clear. A 3% supply miss can demand a 15-20%+ price move; that is why the report moves the market in seconds.
> - **Weather is the input the report quantifies.** Acres are set at planting; summer heat and rain during the **pollination window** set the yield; the August WASDE just puts a number on weather that already happened. The 2012 US drought ran corn to an intraday **\$8.31**.
> - The one number to remember: **stocks-to-use**. Low (single-digit %) means a high, twitchy, explosive price; high (15%+) means a cheap, calm market. Everything a grain trader does on report day is a bet on which way that ratio moved.

At 11:00 a.m. Central time on August 10, 2012, a few hundred grain traders sat staring at screens that were about to lurch. Corn had already nearly doubled over the summer as a brutal drought baked the US Midwest, the worst since 1988. The only open question was *how bad*, and the answer lived in a single PDF that the US Department of Agriculture was about to release to the entire world at the same wall-clock second. When the file dropped, USDA had slashed its national corn yield estimate to **123.4 bushels per acre**, down from 166 the year before — the lowest in nearly two decades. Corn futures gapped, ran to limit, and the nearby contract touched an intraday high of **\$8.31** a bushel, a price that would have looked insane two summers earlier when corn traded near **\$3.80**.

Nobody on that desk had moved a muscle in the moment of release. There was nothing to do. The number was either above your position or below it, and you found out at the same instant as everyone else. That is the strange, brutal character of agricultural markets: a crop grows silently in a field for five months, the weather quietly decides its fate, and then a government statistician compresses the whole season into one report that the market re-prices in the time it takes to read a headline.

This post is about that machine — how a USDA report can move a market in seconds, why weather sits upstream of everything, and the one ratio that turns a tiny change in a yield estimate into a violent change in price. We will build the WASDE balance sheet from nothing, derive *why* small supply misses cause big price moves, walk through report-day mechanics, and end with how a trader actually prepares for the release.

![The WASDE balance sheet flowing from beginning stocks, production, and use through to stocks-to-use and price](/imgs/blogs/weather-the-wasde-and-the-supply-shock-trading-the-report-1.png)

This is a Track-D post in the series, and it leans on the foundations we have already built. If you have not yet met the three big grains, start with [grains: corn, wheat, and soybeans, the calories that trade](/blog/trading/commodities/grains-corn-wheat-and-soybeans-the-calories-that-trade); for the cast of players who actually take the other side of these report-day moves, see [the four players: producers, consumers, hedgers, and speculators](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators). Here we go deep on the *report* — the scheduled shock that re-prices the whole agricultural complex once a month.

## Foundations: the crop balance sheet, from zero

Let us assume you know nothing about agricultural markets and build the entire idea up. The whole thing rests on one piece of arithmetic that a careful ten-year-old could check.

Think about a single crop — corn — over a single marketing year (for US corn, that year runs September to August, because that is the rhythm of the harvest). At the start of the year there is some corn already sitting in storage from previous harvests: silos on farms, grain elevators by the rivers, terminals at the ports. Call that the **beginning stocks** — the carryover. During the year, farmers grow and harvest a new crop: that is **production**, and it equals the number of **acres harvested** multiplied by the **yield** (bushels per acre). And during the year, the world consumes corn: animals eat it as feed, ethanol plants ferment it into fuel, food and industrial users process it, and foreign buyers import it. Add all of that up and you get **use** (also called *total demand* or *disappearance*).

Whatever is left at the end of the year — what nobody fed, ate, burned, or shipped — gets carried into next year. That is the **ending stocks**, or *carryout*. And it is simply:

> **Ending stocks = beginning stocks + production − use**

That is the entire crop balance sheet. It is not a model or a forecast in disguise; it is an accounting identity. Every bushel that exists is either left over at year end or was consumed, and every bushel consumed came from either the old pile or the new harvest. USDA's job in the WASDE is to put its best monthly estimate on each of those four boxes, for every major crop, for the US and for the world.

### Why ending stocks alone is not enough — the stocks-to-use ratio

Knowing that the US will end the year with, say, 1.4 billion bushels of corn does not tell you whether that is a lot or a little. A big country with huge demand can comfortably carry a billion-bushel pile; a small market would be drowning in it. So you have to **scale** the leftover by how fast the market burns through corn. That gives the single most important number in agricultural trading:

> **Stocks-to-use ratio = ending stocks ÷ total annual use**

It answers the question *"if the harvest stopped tomorrow, how much of a year's worth of demand could we cover from what is in the bins?"* A stocks-to-use of 0.10, or 10%, means the carryout equals about 10% of a year's consumption — roughly five weeks of cover. A ratio of 0.02, or 2%, means the cupboard is nearly bare: about a week. A ratio of 0.20 means a fat, comfortable cushion of more than two months.

That ratio is the **tightness gauge**, and it maps to price. The whole rest of this post is, in one way or another, an unpacking of *why* — why a low ratio means a high, twitchy, dangerous price and a high ratio means a cheap, sleepy one, and why the relationship between the two is a steep curve rather than a straight line.

The figure above traces the flow: acres and yield make production; production plus the old carryover, minus everything the world uses, makes ending stocks; ending stocks over use makes the ratio; and the ratio sets the price. Read it left to right and you have read a WASDE.

#### Worked example: reading a corn balance sheet

Suppose USDA's corn balance sheet for a marketing year looks like this (figures are illustrative but in the right ballpark for a roughly average US corn year, in billions of bushels):

- Beginning stocks: **1.4** bbu
- Production: **15.0** bbu (≈ 90 million harvested acres × ≈ 167 bu/acre)
- Total use: **14.9** bbu (feed **5.7**, ethanol **5.4**, food/seed/industrial **1.4**, exports **2.4**)

Plug into the identity: ending stocks = 1.4 + 15.0 − 14.9 = **1.5** bbu. Now scale it: stocks-to-use = 1.5 ÷ 14.9 = **0.101**, or about **10%**. That is a comfortable-but-not-loose market — roughly five weeks of cover — and historically a 10% corn stocks-to-use is consistent with a season-average price somewhere around \$4.00-\$4.50 a bushel.

The intuition: ending stocks of 1.5 billion bushels sounds enormous in isolation, but against 14.9 billion bushels of annual appetite it is only a five-week buffer — enough to keep the price calm, not enough to make corn cheap.

### Knowing the four boxes well enough to argue with them

Each of the four boxes hides its own story, and a trader who treats them as single numbers misses where the next surprise will come from. Take them in turn.

**Beginning stocks** are not really an estimate at all by the time the year starts — they are last year's ending stocks, carried over, and they are the most settled of the four. But they matter enormously because they are the *buffer* from the previous section. A year that begins with a fat carryover can absorb a bad harvest; a year that begins with a thin one is one drought away from a crisis before the first seed is even planted. The carryover is the financial system's memory of how the last few years went.

**Production** is the most volatile box and the one the reports fight over. It is acres × yield, and the two halves behave completely differently. **Acres** are a human decision — farmers choosing in spring how much corn versus soybeans versus wheat to plant, responding to relative prices, rotation agronomy, and input costs. Acres are surfaced by the Prospective Plantings and Acreage reports and tend to surprise by a percent or two. **Yield** is nature's decision — the bushels each acre actually delivers — and it is set by weather over the summer. Yield is where the *big* surprises live, because a 5% swing in yield is common and, multiplied across 90 million acres, it is a vast quantity of grain.

**Use** splits into four sub-markets that respond to different forces. For corn: **feed** demand tracks the size of the livestock herd and the price of competing feeds; **ethanol** demand is anchored by a fuel-blending mandate but flexes with gasoline demand and ethanol economics; **food, seed, and industrial** is the steadiest slice; and **exports** are the swing factor, lurching with the dollar, foreign harvests, and trade policy. A WASDE can leave production untouched and still re-rate the price purely by revising one use category — most often exports.

**Ending stocks**, the result, is therefore a small number sitting at the end of a chain of much larger ones, which is exactly why it is so sensitive. Ending stocks of 1.5 bbu is the *difference* between supply near 16.4 bbu and use near 14.9 bbu. A 2% error in either of those big numbers is roughly 0.3 bbu — and against a 1.5 bbu carryout, that 0.3 bbu is a 20% swing in ending stocks. **Small percentage errors in the big boxes become large percentage swings in the small box that sets the price.** This is the deep reason the report is so explosive: it is a leverage machine built into the arithmetic itself.

## Why a 2% supply miss moves price 15%: inelastic demand

Here is the fact that makes grain markets so violent, and it is worth slowing down on because it is the deepest point in the post. **Agricultural demand is inelastic.** That word means: the quantity people want barely changes when the price changes. A cow eats the same amount of corn whether corn is \$4 or \$7. A human eats about the same number of calories of wheat whether bread is cheap or dear. An ethanol mandate burns a roughly fixed quantity regardless of price. In the short run — within a marketing year — demand for staple grains is close to a vertical line.

Now picture what that does to a supply shock. (The supply/demand sketch below this paragraph shows the mechanism.) Supply for a given crop year is essentially fixed too: once the crop is harvested, that is all the grain there is until next year — production is also a near-vertical line. So you have a roughly vertical supply curve meeting a roughly vertical demand curve, and the price is just wherever they cross. Knock 3% off supply and the supply line shifts left by 3%. To bring demand down by that same 3% — because the market *must* clear; every bushel must find a buyer — the price has to rise far enough to choke off 3% of consumption. But demand is inelastic, so squeezing out 3% of consumption requires an *enormous* price increase.

How enormous? Economists summarise the responsiveness of demand to price with the **price elasticity of demand**, defined as the percentage change in quantity demanded divided by the percentage change in price. For staple grains in the short run, that elasticity is small — often estimated around **−0.2**, meaning a 10% price rise trims demand by only about 2%. Rearrange the definition and you get the punchline:

> Required price change ≈ (supply shortfall, in %) ÷ |elasticity|

#### Worked example: the inelastic-demand math

Take an elasticity of **−0.2** and a supply shortfall of **3%**. To make the market clear, demand must fall by 3%. Required price change = 3% ÷ 0.2 = **15%**. A mere 3% loss of the crop forces a 15% rally just to ration consumption back into line with the smaller supply.

Make the elasticity even stiffer — say **−0.15**, which is plausible in a genuinely tight year when the easy demand cuts have already happened — and the same 3% shortfall needs 3% ÷ 0.15 = **20%**. And if the shock is bigger, say a 6% production loss in a drought, you get 6% ÷ 0.15 = a **40%** price move. The intuition: when nobody can easily eat less, the only thing that can shrink demand is a price high enough to be painful, so a small dent in supply translates into a brutal move in price.

This is the engine behind every grain blow-off in history. It is *not* that traders are irrational or that speculators are hoarding. It is arithmetic: inelastic demand plus inelastic short-run supply equals enormous price sensitivity to small quantity surprises. And the WASDE is the instrument that *delivers* those quantity surprises, on a schedule, once a month.

It is worth being concrete about *why* grain demand is so inelastic, because the reasons also tell you when elasticity changes. Three forces pin it down. First, **biological necessity**: a population eats roughly fixed calories and a livestock herd eats roughly fixed rations; you cannot tell a feedlot to fatten cattle on 20% less corn because corn got expensive. Second, **mandates and contracts**: the US ethanol blending requirement effectively fixes a large slab of corn demand by law, and food processors lock supply contracts that do not flex week to week. Third, **the small share of the final price**: the corn in a box of cornflakes is a few cents; even a doubling of the corn price barely moves the retail price of the food, so the consumer never feels the signal that would make them eat less. All three forces make the short-run demand curve nearly vertical.

But notice what those forces imply about *when* elasticity rises. Demand becomes more elastic over a longer horizon and at extreme prices, because the slow adjustments finally kick in: feeders substitute wheat or distillers' grains for corn, ethanol plants idle when margins go negative, importers ration or switch origins, and — over years — farmers plant more acres and breeders push yields higher. So the elasticity in the formula is *not a constant*; it is smallest in the heat of a tight year (which is exactly when a shock does the most damage) and larger once the market has had time to adjust. This is why blow-off spikes are sharp and reversion is slow: the spike happens at low elasticity, and the come-down waits for the high-elasticity adjustments to arrive.

### The buffer changes everything: the convex price-vs-stocks-to-use curve

There is a crucial refinement. The elasticity story above assumes the market truly has to clear from current production. But it does not, quite — there is the carryover pile, the ending stocks. That buffer is what lets a comfortable market shrug off a shock. If you have two months of grain sitting in storage and you lose 3% of the harvest, you can just draw the pile down a little; price barely needs to move because the *available* supply (production + stocks) only fell by a sliver. But if the pile is already nearly empty, there is nothing to draw down — the shock hits the price with full force.

![A tight low stocks-to-use crop explodes in price while a comfortable high ratio crop stays calm](/imgs/blogs/weather-the-wasde-and-the-supply-shock-trading-the-report-2.png)

That is why the relationship between price and the stocks-to-use ratio is **convex**, not linear — the figure above contrasts the two regimes. When stocks-to-use is high (say 20%), the price sits low and flat, and even a meaningful shock barely nudges it, because the buffer absorbs the blow. As stocks-to-use falls, the price rises *gently* at first, then more steeply, and once the ratio drops into the single digits — into a genuinely tight market — the curve goes nearly vertical. Each additional percentage point of tightness now demands a violent move, because there is no buffer left and the full force of inelastic demand is exposed.

![Corn price versus stocks-to-use ratio showing a steep convex inverse relationship](/imgs/blogs/weather-the-wasde-and-the-supply-shock-trading-the-report-5.png)

The chart above shows the shape with an illustrative corn dataset — the stocks-to-use values are built to demonstrate the relationship, not pulled from USDA prints (data_commodities.py carries grain *prices* but not a stocks-to-use series). Notice how the points hug a curve that is shallow on the right and explosive on the left. A market sitting at 18-20% stocks-to-use can lose a chunk of its crop and the price hardly stirs; a market sitting at 6% is a coiled spring. This single picture explains why some WASDE reports are non-events and others trigger limit moves: it is not the size of the surprise alone, it is the size of the surprise *times where you are on the curve*.

One more way to feel the convexity: translate the ratio into **weeks of cover**, which is stocks-to-use × 52. A 20% ratio is about ten and a half weeks of grain in the buffer; an 8% ratio is about four weeks; a 4% ratio is about two weeks. The price does not care linearly about weeks of cover — going from ten weeks to nine weeks barely registers, because ten weeks was already plenty, but going from three weeks to two weeks is terrifying, because two weeks is genuinely close to running out before the next harvest. The market's anxiety rises *faster than linearly* as the buffer shrinks, and that accelerating anxiety is the convex curve. It is the difference between a half-full reservoir in a mild drought and a nearly empty one: the same inch of lost water means nothing in the first case and triggers rationing in the second.

#### Worked example: convexity in weeks of cover

Two corn markets each lose **400 million bushels** of crop to a yield cut — the same absolute shock. Market A starts at 16% stocks-to-use (≈ 8.3 weeks of cover, ending stocks 2.3 bbu on 14.4 bbu use); Market B starts at 7% (≈ 3.6 weeks, ending stocks 1.0 bbu on 14.4 bbu use). The shock drops A to about 13.2% and B to about 4.2%. In A, the buffer fell from 8.3 to 6.9 weeks — still comfortable, so price might rise 5-8%. In B, the buffer fell from 3.6 to 2.2 weeks — now genuinely scary, and price can leap 30-40% as the market scrambles to ration the last bushels. **Identical shock, wildly different price response, because the convex curve is flat on the right and vertical on the left.** The intuition: never judge a supply shock in bushels alone; judge it in weeks of cover, because the same bushels mean nothing to a full bin and everything to an empty one.

#### Worked example: a stocks-to-use read

Suppose corn stocks-to-use is **12%** and the season-average price is around **\$4.20**. Now a hot, dry summer cuts the new crop, and the next WASDE drops the ending-stocks estimate so the ratio falls to **6%**. Where does price go?

Reading off the convex curve (and consistent with history — 2012-13 corn stocks-to-use fell toward 7-8% and the season-average price ran to roughly \$6.90), a halving of the ratio from 12% to 6% does *not* halve or double anything linearly; it roughly **doubles the price**, from about \$4.20 to the \$7-8 range, because the move down the curve is into the steep zone. The intuition: the first few percentage points of tightness are cheap, but the last ones are where the price explodes, so the *same* 6-point drop in the ratio matters far more when it lands you at 6% than when it lands you at 16%.

## What the WASDE actually is, and how it is built

The WASDE is published by USDA's **World Agricultural Outlook Board** around the middle of every month, typically at **noon Eastern time**. It is a comprehensive set of balance sheets — supply and use — for the major US and world crops: corn, soybeans, wheat, rice, cotton, plus sugar and livestock products. For each, it gives the four boxes we built above, the resulting ending stocks, and a projected **season-average farm price**.

It is not the only USDA report that matters — and the distinction is worth knowing, because traders position differently around each:

- The **WASDE** itself updates the demand side and the *projected* balance sheet monthly.
- The **Crop Production** report, released alongside the August through November WASDEs, contains USDA's **survey-based yield estimates** — the all-important bushels-per-acre number that drives the production box. The August Crop Production is the first survey yield of the year and is often the single most explosive release on the calendar.
- The **Prospective Plantings** report (late March) gives the first read on how many acres farmers *intend* to plant.
- The **Acreage** report (end of June) gives the actual planted acres — frequently a shock in its own right, as in June 2019 when wet planting weather scrambled corn acres.
- The **Grain Stocks** report (quarterly) is a physical inventory count and can blindside a market that thought it knew demand.
- The weekly **Crop Progress** report (Monday afternoons, in season) tracks planting pace and, critically, the **crop condition** — the percent of the crop rated good-to-excellent — which the market uses to nowcast yield between the big reports.

Not all reports on the calendar carry equal weight, and a trader has to know which dates are dangerous. The **August Crop Production** is usually the year's most explosive because it is the first survey-based yield after the make-or-break July weather — the market has been guessing, and now it gets measured. The **October and November** reports refine that yield as harvest data comes in, and the **January final Crop Production** (released with the January WASDE) is famously volatile despite arriving after harvest, because it makes the *final* yield revision and trues up the quarterly Grain Stocks — January reports have produced some of the biggest single-day grain moves on record precisely because the market assumed the story was settled and it was not. The late-March **Prospective Plantings** and end-June **Acreage** reports move the *new-crop* contracts hardest, since they reset the production box for the year ahead. A grain trader marks these dates on the calendar the way an equity trader marks an FOMC meeting.

The reason these reports move markets at all is **information asymmetry collapse**. In the days before a release, thousands of analysts, farmers, and funds each hold a fragment of a guess about the crop. The WASDE is the moment that USDA — which surveys tens of thousands of farms and, for the survey reports, sends enumerators into fields to count ears and measure plots — replaces all those private guesses with one official number that everyone must now trade against. The price was sitting at the market's *average* guess; the report reveals whether the truth is above or below it, and the price snaps to the new level.

### The whisper number and why surprises, not levels, move price

Here is a point that trips up beginners: a *bullish* WASDE can send prices *down*, and a *bearish* one can send them *up*. The report does not move price by being good or bad news in the abstract. It moves price by being **different from what was already priced in**. Markets are forward-looking machines that bake the consensus expectation into the price *before* the release. The thing that moves on release is the gap between the actual number and the expectation — what traders call the **surprise**.

The consensus expectation has a name on the grain desk: the **whisper number**, or more formally the average of the pre-report analyst estimates that wire services like Reuters and Bloomberg poll and publish a day or two ahead. If the whisper for corn yield is 175 bu/acre and USDA prints 175, the report is a non-event — it confirmed what the price already knew. If the whisper is 175 and USDA prints 170, that is a bullish *surprise* and corn gaps up, even though 170 is a perfectly large crop. If USDA prints 180, corn gaps down despite that also being a big, fine crop. **You are not trading the number; you are trading the number minus the whisper.** This is the same mechanism that governs every scheduled macro release — the surprise framework explained in [why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) applies cleanly to crop reports.

#### Worked example: a yield surprise and the production gap

USDA is about to publish the August corn yield. The whisper number is **177 bu/acre** on **90 million harvested acres**, which the market has already priced as a 90 × 177 = **15.93** billion-bushel crop. The report drops and USDA prints **172** — a downside surprise of 5 bu/acre.

Run the production box: 90 million acres × 5 fewer bushels = **450 million bushels** of corn that the market thought existed and now does not. If the prior ending-stocks estimate was 1.8 bbu against use of 14.5 bbu (a 12.4% stocks-to-use), removing 450 mbu from supply drops ending stocks to about **1.35** bbu and the ratio to roughly **9.3%** — a meaningful tightening that pushes you down the convex curve into a steeper zone. A move from 12.4% to 9.3% is the kind of shift that historically lifts corn 8-12% in the days around the report. The intuition: a 5-bushel cut sounds trivial, but multiplied across 90 million acres it erases nearly half a billion bushels — and against inelastic demand, that has to be rationed by price.

## Report-day mechanics: the lockup, the release, and the gap

Now we get to the part that makes the WASDE physically dangerous to trade through. USDA goes to extraordinary lengths to make sure *nobody* gets the number early, because in a market this sensitive an early peek would be worth a fortune.

![Report day timeline from the lockup through the simultaneous release to the instant price gap](/imgs/blogs/weather-the-wasde-and-the-supply-shock-trading-the-report-4.png)

The timeline above lays out the day. In the hours before release, USDA staff finalise the figures inside a **secure lockup**: a room with no outside communication, windows covered, phones and internet cut, and (historically, in the strictest versions) doors taped shut, so that not a single digit can leak before the official moment. At exactly the scheduled second — noon Eastern for the WASDE — the file is **released to everyone simultaneously**, posted to the USDA website and pushed to the wire services at the same instant. There is no privileged early feed. (USDA tightened these procedures further in the 2010s precisely to remove any micro-timing advantage from co-located machines.)

What happens in the market in the first milliseconds after release is the whole game. Algorithms parse the headline yield and ending-stocks numbers in microseconds and fire orders. If the surprise is large, the price **gaps** — it does not trade smoothly from the old level to the new one; it jumps, because there are no resting orders in between at the old prices. In the grain pits the move can hit the exchange's **daily price limit** (corn, for instance, has a limit of a fixed number of cents per bushel per day, beyond which trading halts or only trades at the limit), and the market can go **limit-up** or **limit-down** and simply stop, leaving anyone on the wrong side unable to get out at all until the next session.

This is why **positioning into a report is so dangerous**. A stop-loss order does not protect you across a gap: if you are long, your stop is at \$4.00, and the market gaps from \$4.10 straight to \$3.70 on a bearish surprise, your stop fills at \$3.70 (or worse), not \$4.00. The 30 cents between your stop and the fill is pure, unhedged loss caused by the gap. On a leveraged futures position, where you might control a \$200,000+ notional of corn (one contract is 5,000 bushels) on a few thousand dollars of margin, that gap can wipe out your margin and trigger a margin call before you have had a coffee.

#### Worked example: the report-gap risk on a leveraged position

You are long **10 corn contracts** into the August report — that is 10 × 5,000 = **50,000 bushels**. Corn is at **\$4.10**, so your notional is 50,000 × \$4.10 = **\$205,000**, held on perhaps **\$25,000** of margin (about 8:1 leverage). You have a stop at \$4.00, comforting yourself that your risk is "only" 10 cents, or \$5,000.

The report prints a bullish-for-no-one, bearish surprise: a record yield. Corn gaps straight to **\$3.78** and trades down to limit. Your stop, useless across the gap, fills around **\$3.78**. Your loss = (4.10 − 3.78) × 50,000 = **\$16,000** — not the \$5,000 you planned for, but more than three times it, and **64% of your margin** gone in one print. The intuition: leverage and gaps are a poisonous pair, because the gap defeats the stop that the leverage made you rely on, so the real risk of holding into a report is not the distance to your stop but the distance the market can *jump past* it.

The professional lesson is blunt: **a report is not a trade, it is a coin flip on an unknown number.** Experienced grain traders rarely hold large directional positions into a major WASDE or Crop Production release. They either flatten down to a size they can afford to be wrong on, hedge with options (defined-risk, no gap surprise beyond the premium), or trade the *reaction* — waiting for the number, letting the gap happen, and then trading the market's digestion of it once liquidity returns. The edge, if there is one, is in reading positioning and the whisper *before*, and in reading the market's response *after* — not in guessing the number itself.

### The tell in the curve: old-crop versus new-crop

A report does not just move the flat price; it reshapes the **forward curve**, and that reshaping carries information you cannot get from the headline. Grain futures trade as a strip of monthly contracts, and the most important divide is between **old-crop** (contracts that deliver from the harvest already in the bin) and **new-crop** (contracts that deliver from the crop still growing in the field). For US corn the old-crop/new-crop boundary sits at the September and December contracts; the July contract is deep old-crop, December is the first new-crop month.

When a balance sheet is tight, the *old-crop* contracts trade at a premium to the new-crop ones — the market is bidding up the scarce grain that exists *now* relative to the harvest that is on the way. That is **backwardation**, and it is the single clearest signal that the current-year stocks-to-use is low: the market is paying up for immediacy because it cannot wait for the new crop. When the balance sheet is comfortable, the curve flattens or tips into **contango**, with deferred contracts at a premium that reflects the cost of storing surplus grain. This is the same forward-curve machinery that runs through the whole series — [contango versus backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — applied to a market where the curve's shape is a direct read on the harvest cycle.

So a WASDE can leave the flat price roughly unchanged and still send a loud signal by *steepening the old-crop/new-crop spread*. If a report cuts old-crop ending stocks but leaves the new-crop outlook intact (a big new crop is coming), the July-December spread can blow out into deep backwardation — old-crop scarcity, new-crop relief — even as the average price barely moves. Traders who only watch the front month miss this entirely. The spread is often where the cleanest expression of a tight-balance-sheet view lives, because it isolates the scarcity story from the broad direction of the whole complex.

## Weather: the input the report quantifies

Step back and ask where the surprise in a Crop Production report actually *comes from*. It comes from **yield**, and yield comes from **weather**. The acres were planted in April and May; by the time the price prints in August, the size of the crop was decided weeks earlier in the field. The report does not *create* the crop; it *measures* one that the weather already grew. This is the single most important mental shift for trading agriculture: **you are not trading the report, you are trading the weather that the report will eventually confirm.**

![The weather market from planting through growing degree days and pollination to the final yield and the WASDE](/imgs/blogs/weather-the-wasde-and-the-supply-shock-trading-the-report-6.png)

The pipeline above shows the chain. **Planting** fixes the acreage — the *width* of the crop — and is largely done by late May. Then the plant grows, and what it needs is heat and water, measured by **growing-degree-days** (GDD): a running sum of how much each day's temperature exceeds a base threshold (around 50°F for corn). Too few GDDs and the crop is behind and vulnerable to an early frost; the right accumulation with adequate rain and the crop thrives. The make-or-break moment is the **pollination window** — for US corn, roughly the back half of July — when the plant sets its kernels. Heat and drought during pollination are catastrophic: the plant aborts kernels and yield collapses, and there is no recovering it later. This is why the corn market lives and dies by the **July weather forecast**, and why the period from roughly June through August is known on the desk as the **weather market**, when prices swing on each new run of the forecast models rather than on any report.

Between the big monthly reports, traders nowcast the crop using two public datasets. The weekly **Crop Progress / Crop Condition** report rates the percentage of the crop **good-to-excellent**; a falling good-to-excellent percentage through July is an early warning that yield — and therefore the next WASDE — is heading down. And the **US Drought Monitor**, a weekly map of drought intensity across the country, tells you *where* the stress is and whether it overlaps the heart of the Corn Belt (Iowa, Illinois, Nebraska, Indiana). A drought map that lights up red over Iowa in mid-July is, in effect, a leaked preview of a bearish-for-supply August WASDE. The traders who position *ahead* of the report are really positioning ahead of the **weather**, using condition ratings and drought maps as the early read on a yield number USDA will not publish for weeks.

### How 2012 actually unfolded

The 2012 drought is the textbook case of weather → yield → report → price, and the data make it vivid.

![Corn annual average price with the 2012 drought intraday peak marked](/imgs/blogs/weather-the-wasde-and-the-supply-shock-trading-the-report-7.png)

The chart above plots corn's annual average and flags the 2012 spike. The crop went into the ground in good shape that spring, and as late as June the market expected a record harvest — early USDA projections had corn yields near 166 bu/acre. Then July turned brutally hot and dry across the Corn Belt, right through the pollination window. The weekly crop-condition ratings cratered: the good-to-excellent percentage fell week after week, and the Drought Monitor map turned a deep red over the western Corn Belt. The market read the weather and rallied hard *before* the reports — corn ran from the high \$5s in June toward \$8 in July and August. Then the August Crop Production report *confirmed* it, cutting the yield estimate to **123.4 bu/acre**, and the September and later reports cut it further to **122.3**. The nearby contract printed its intraday high of **\$8.31**.

The sequence is the whole lesson: the weather did the damage in the field in July, the condition ratings and drought map let the market *see* it in near-real-time, and the WASDE/Crop Production reports simply *ratified* a tightening that the price had already largely moved to. By the time USDA published the number, stocks-to-use had collapsed toward the high single digits and the convex curve did the rest.

There is a second lesson hiding in 2012 about *how the price actually rationed* the smaller crop, because the inelastic-demand story plays out through specific channels. With corn near \$8, the rationing happened in three places. **Ethanol** plants — the most price-sensitive slice of demand — cut runs as their margins turned negative, idling capacity and freeing up corn. **Exports** collapsed as foreign buyers switched to cheaper origins or substituted other feed grains, so US corn priced itself out of the world market. And **feeders** trimmed rations and pulled cattle and hogs forward to slaughter rather than feed them expensive corn, which dumped meat onto the market and quietly shrank the future herd. Each channel is a place where demand finally bent — and the price had to climb to \$8 precisely because none of them bend easily. Watching *which* demand channel cracks first, and how far price has to go to crack it, is the deep read on any tight market: the channel that gives way most cheaply caps the upside, and when even that channel is exhausted, the price has nowhere to go but vertical.

The two grain spikes on the next chart — 2012 (drought) and 2022 (the Russia-Ukraine war shutting in a huge slice of world wheat and corn exports) — show the same machine driven by two different shocks. One was a *yield* shock that the WASDE quantified; the other was a *trade-flow* shock that hit the use/export side of the balance sheet. Both worked through the same arithmetic: a sudden cut to available supply, an inelastic demand curve, a low buffer, and a price that had to move violently to clear.

![Corn wheat and soybean annual average prices with the 2012 drought and 2022 war spikes annotated](/imgs/blogs/weather-the-wasde-and-the-supply-shock-trading-the-report-3.png)

The chart above tracks all three big grains in dollars per bushel. Note that the shocks are not perfectly synchronised: 2012 hit corn and soybeans hardest (a US weather event), while 2022 hit wheat hardest (a Black Sea export event), because each crop's balance sheet has its own geography of supply and its own dominant risk. A corn trader watches Iowa weather; a wheat trader watches the Black Sea, the Russian winter-wheat belt, and the Australian and Argentine harvests; a soybean trader watches both the US Midwest in northern summer and Brazil in the southern-hemisphere summer (the *safrinha* and main crops). The report is global precisely because the supply risk is global, and a tight world balance sheet for one crop will not save you if the local one is loose, or vice versa.

## Common misconceptions

Let us correct the five beliefs that most often get beginners run over by a report.

**"A bullish report means the price goes up."** No — a report moves price by *surprise versus the whisper*, not by being objectively good or bad. In August 2016, USDA printed a then-record corn yield, but the market had braced for an even bigger number, and corn actually firmed on the "less bearish than feared" read. Always anchor to the pre-report consensus, not to the headline.

**"Big ending stocks always mean low prices."** Only relative to *use*. A 2-billion-bushel corn carryout is bearish if use is 13 billion (15% stocks-to-use, cheap) and neutral if use grows to 15 billion (13%, fine). The level of stocks is meaningless without the denominator; **stocks-to-use is the number, not stocks**.

**"Speculators cause the spikes."** The 2012 and 2022 moves were caused by a drought and a war removing physical supply against inelastic demand — the arithmetic of the balance sheet, not a hedge fund's whim. Speculators provide the liquidity that lets producers and consumers hedge, and they get *carried out* on the wrong side as often as the right one. The four-player structure that governs who is actually on the other side is laid out in [the four players](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators).

**"You can trade the report if you guess the number right."** Even a correct guess can lose money if it matched the whisper (priced in already), and a wrong guess can blow through your stop on the gap. The number is necessary but not sufficient; you also have to know what was *expected*, and you have to survive the gap. Most professionals reduce size or use options into the print rather than betting the number.

**"Weather only matters in the US."** Grain is a world market. A perfect US corn crop can still see prices rise if a drought hits Brazil's soybeans or the Black Sea wheat belt, because export demand reroutes to the US and draws down its balance sheet. The report is a *world* supply-and-demand estimate for a reason. For the cross-asset view of how the softs and grains fit the broader commodity complex, see [agriculture and softs: the food and fiber markets](/blog/trading/cross-asset/agriculture-softs-the-food-and-fiber-markets).

## How it shows up in real markets

Three concrete patterns recur, and recognising them is most of the practical skill.

**The pre-report drift and the whisper.** In the days before a major WASDE, the market drifts toward the consensus as positioning builds. Wire services publish the analyst-survey averages — the whisper — and you can see the price gravitate toward what that whisper implies. The tradeable information is in the *dispersion* of the estimates: a tight cluster of analyst guesses means a low-surprise report is likely and the move will be small; a wide spread means the analysts themselves are unsure, the surprise potential is high, and the report is dangerous. Smart desks size their risk to the *dispersion*, not just the level.

There is also a positioning dimension that interacts with the whisper. If the speculative crowd is heavily long going into a report and the number comes in merely *in line*, the price can still fall — because there is no one left to buy and the longs take profits on a non-event. A crowded position is itself a risk that the report can trigger, independent of the surprise: the report becomes the *catalyst* that unwinds an over-extended trade. This is the agricultural version of the same lesson the four-player framework teaches — read positioning before price — and it means two reports with identical surprises can produce opposite moves depending on who was already leaning which way. The Commitments of Traders data and the open-interest trend going into a report tell you how much fuel sits on each side of the gap, which is often more predictive of the *reaction* than the number itself.

**The condition-rating leading indicator.** Through the growing season, the Monday-afternoon crop-condition ratings are a weekly serial on the crop's health. A good-to-excellent percentage that falls three or four weeks in a row during July is a near-certain precursor to a yield cut in the next Crop Production report — the report is *lagging* the field. Traders who watch conditions and the Drought Monitor are effectively front-running USDA's own measurement, which is legal and exactly how the weather market is supposed to work.

**The export-pace surprise.** The use side springs surprises too. USDA publishes a weekly **Export Sales** report, and a string of huge weekly sales (often to China for soybeans) can tighten the balance sheet faster than the monthly WASDE has caught up to. A market that is loose on paper can become tight on the ground if export demand runs hot — and the *next* WASDE then has to revise use upward, cut ending stocks, and re-rate the price. This is the demand-side mirror of a weather shock, and it is why a soybean trader watches Chinese buying as closely as a corn trader watches Iowa rain.

#### Worked example: an export surprise tightens the sheet

Soybeans are sitting comfortably: ending stocks projected at **0.40** bbu against use of **4.40** bbu, a stocks-to-use of **9.1%**, and beans trading near \$10.50. Then a run of weekly Export Sales prints shows China buying aggressively, and the cumulative sales pace runs **120 million bushels** ahead of where USDA's annual export figure implies it should be. The market does the math before USDA does: if those sales are real and shipped, the next WASDE must lift exports by ~120 mbu, which (with production fixed at this stage of the year) comes straight out of ending stocks. Ending stocks drop to **0.28** bbu, and stocks-to-use falls to about **6.2%** — into the steep part of the curve. Beans rally toward \$12-13 in anticipation, *before* the WASDE confirms it. The intuition: on the demand side, the weekly export data is the leading indicator that the monthly report lags, so a soybean desk reads the export pace as a preview of the next ending-stocks revision.

This soybean example also exposes the most important *seasonal* wrinkle in grain trading: the southern hemisphere. US soybeans are harvested in the autumn, but Brazil and Argentina harvest their main crop the following February through May. So a soybean balance sheet has *two* weather markets per calendar year — the US Midwest in northern summer and South America in southern summer — and a US trader who ignores Brazilian rainfall in January is trading half the picture. A drought in Mato Grosso can tighten the *world* soybean balance and pull US export demand even when the US crop was fine, which is the demand-side mirror of a US weather shock. The report is annual and global precisely so it can net these out, but the market trades the pieces as they arrive.

A useful way to hold all three together: the balance sheet has a *supply* side (weather, yield, acres — surfaced by Crop Production and Acreage reports) and a *demand* side (feed, ethanol, exports — surfaced by Export Sales and the WASDE's use revisions). Either side can spring the surprise that re-rates the stocks-to-use ratio, and the convex curve translates that re-rating into price. A scheduled report is simply the moment the official number replaces the market's guess; the *unscheduled* shocks — a sudden export ban, a war, a flash drought — are even more violent because there is no whisper to have pre-priced them. (The mechanics of those unscheduled, off-calendar shocks are covered in [food security, export bans, and when governments hoard](/blog/trading/commodities/food-security-export-bans-and-when-governments-hoard), where a single government decree can do to the balance sheet in a day what a drought does in a month.)

## The playbook: how a trader prepares for a WASDE

Pull it all together into what you actually *do* in the week of a report.

**1. Know the stocks-to-use, and know where you are on the curve.** Before anything else, look up the current ending-stocks and use estimates and compute the stocks-to-use ratio. A market at 6% is a different animal from a market at 18%: at 6% any downside surprise is dynamite, at 18% even a real shock fizzles. Your *whole* risk posture should scale with where you sit on the convex curve, because that determines how much price will move per unit of surprise.

**2. Find the whisper, and weigh the dispersion.** Get the pre-report analyst-survey average — the whisper — for the key numbers (yield, production, ending stocks). Then look at the *range* of estimates. A narrow range means low surprise risk; a wide range means the report is a genuine event and you should cut size or buy optionality. Remember you are trading *surprise versus whisper*, so the whisper is the thing your position is implicitly betting against.

**3. Read the weather and the leading indicators.** During the growing season, the report is downstream of the field. Track the weekly crop-condition (good-to-excellent) trend and the Drought Monitor over the core growing regions. If conditions have been deteriorating, a bearish-for-supply WASDE is likely already baked partway in; if they have been improving, the risk is a bigger crop than the whisper. The forecast for the **pollination window** in July is worth more than any single report.

**4. Respect the gap — size for the print, not the stop.** Going into the release, assume your stop will not hold across the gap. Size the position so that a limit move *against* you is survivable on your margin, or replace the futures with a defined-risk options structure where the most you can lose is the premium. The single most common way to blow up in grains is to carry a leveraged directional bet into a Crop Production report and discover that the gap defeated the stop.

**5. Consider trading the reaction, not the number.** The professional edge is rarely in guessing USDA's figure; it is in reading positioning and the whisper *before*, and in reading the market's digestion *after*. Let the number print, let the gap happen, and then trade the market's overshoot or follow-through once liquidity returns — when you can actually get filled and your stops mean something again.

**6. Watch the curve, not just the front month.** A report's clearest signal often shows up in the old-crop/new-crop spread rather than the flat price. A spread that lurches into deeper backwardation after a report is telling you the *current-year* tightness intensified even if the headline barely moved; a spread that flattens is telling you the scarcity premium is bleeding out as the new crop comes into view. The spread isolates the balance-sheet story from the noise of the whole-complex direction, which is why seasoned grain traders quote each other the spread before they quote the flat price.

A final framing that ties the playbook together: the WASDE is a *forecast that becomes a fact in stages*. In the spring it is mostly a guess about acres and trend yield, with wide error bars; through the summer the weather narrows those bars and the condition ratings let you watch the narrowing happen; by the November final the crop is in the bin and the number is nearly settled. The price's job all season is to keep re-rationing an estimate that keeps getting more certain, and each report is a checkpoint where the market's running guess is forced to reconcile with USDA's. The trader's job is not to out-forecast a survey of tens of thousands of farms — you will lose that fight — but to know where the price sits on the convex curve, what the whisper already assumes, and how much the weather has already told you before the official number lands.

The spine of this whole series is that a commodity price is a physical thing — a crop, a barrel, a tonne — forced through a financial contract, and that the *physical* fundamentals set the boundaries the paper must respect. Grains are the purest expression of that idea. There is no convenience yield or contango story richer than the simple, brutal physical fact that a finite crop was grown in a field by weather you could watch in near-real-time, and that the price is just the market rationing a fixed, inelastic-demand quantity until next year's harvest refills the bins. The WASDE is the scoreboard. The weather is the game. And the stocks-to-use ratio is the single number that tells you how hard the next surprise will hit. Learn to read that ratio, anchor every report to the whisper, watch the July sky over Iowa, and you will understand why a market can move more in one second at noon Eastern than it did in the entire month before.

## Further reading & cross-links

- [Grains: corn, wheat, and soybeans, the calories that trade](/blog/trading/commodities/grains-corn-wheat-and-soybeans-the-calories-that-trade) — the foundations of the three big grain markets this post trades around.
- [Food security, export bans, and when governments hoard](/blog/trading/commodities/food-security-export-bans-and-when-governments-hoard) — the unscheduled, off-calendar supply shocks that hit the demand/export side of the balance sheet.
- [The four players: producers, consumers, hedgers, and speculators](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators) — who is actually on the other side of a report-day move and why.
- [Why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) — the general mechanism (price moves on surprise versus consensus) that governs every scheduled release, including the WASDE.
- [Agriculture and softs: the food and fiber markets](/blog/trading/cross-asset/agriculture-softs-the-food-and-fiber-markets) — how grains and softs fit into the broader cross-asset commodity complex.
