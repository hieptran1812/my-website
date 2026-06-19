---
title: "The Dollar and Commodities: Why a Strong Dollar Weighs on Prices"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a stronger dollar usually pushes commodity prices down — the priced-in-dollars channel that makes a barrel more expensive for every foreign buyer, the two-way causation, and the supply shocks (like 2022) that break the rule."
tags: ["commodities", "dollar", "dxy", "oil", "copper", "real-rates", "fx", "macro", "inverse-correlation", "priced-in-dollars", "supply-shock", "trading"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Almost every globally traded commodity is quoted in **US dollars**, so the dollar's strength is built into every commodity price. When the dollar rises, the same barrel or tonne costs more in euros, yen, and rupees, which softens demand from every non-dollar buyer — and at the same time foreign producers' local-currency costs fall, so they can accept a lower dollar price. Both forces push the dollar price of the commodity **down**. A weaker dollar does the reverse.
>
> - The mechanism is the **priced-in-dollars channel**, not magic: an \$80 barrel costs a euro-zone refiner about **€66.7** at EURUSD 1.20 but **€80.0** at parity — a 20% jump in their real cost with the oil price unchanged.
> - The inverse is **strong but not constant**. It is a default, not a law. It holds cleanly when the dollar is the main thing moving (2014–2016: dollar surges +9.3% in 2015, oil crashes −47.8%), and it **breaks** when a supply shock dominates.
> - **2022 is the textbook break**: the dollar rose **+8.2%** AND oil rose **+39.3%** in the same year, because a war-driven supply shock overrode the currency channel.
> - The one fact to remember: the dollar is **one input** to a commodity view — a headwind gauge — never a mechanical rule. The trader's job is to know which regime they are in.

On 28 September 2022, the US dollar index touched roughly **114.8**, its highest level in two decades. The pound had just cratered after a disastrous UK budget; the euro was below parity for the first time since 2002; the Japanese yen was so weak that Tokyo intervened in the currency market for the first time since 1998. For anyone watching the commodity screens, the textbook said one thing should be happening: with the dollar this strong, oil, copper, and the rest of the complex should be getting crushed. Crude is priced in dollars, the logic goes, so a king dollar makes every barrel more expensive abroad and demand should buckle.

And yet that year, oil did not fall. The annual average price of West Texas Intermediate crude rose to about **\$94.90** a barrel in 2022, up **+39.3%** from 2021 — the same year the dollar rose **+8.2%**. Two things that are "supposed" to move in opposite directions both went up, hard, at the same time. Anyone who had treated the dollar–commodity inverse as a mechanical rule — sell oil because the dollar is strong — got run over by a freight train called Russia's invasion of Ukraine.

That paradox is the whole subject of this post. The inverse relationship between the dollar and commodity prices is one of the most reliable patterns in all of macro — and one of the most misunderstood. It is real, it has a clean mechanism behind it, and it holds far more often than not. But it is a *tendency born of how commodities are priced*, not a physical law, and it shatters precisely when something else — a war, an OPEC cut, a crop failure — becomes the dominant force. By the end of this article you will understand exactly why the inverse exists, how to compute its effect on a foreign buyer to the dollar, when it holds and when it breaks, and how a working macro trader uses the dollar as one input to a commodity view rather than a switch.

![The priced-in-dollars seesaw showing dollar up means commodity down and dollar down means commodity up with the foreign buyer mechanism](/imgs/blogs/the-dollar-and-commodities-why-a-strong-dollar-weighs-on-prices-1.png)

The figure above is the entire argument in one frame. The dollar sits on one end of a seesaw and the commodity price sits on the other. When the dollar strengthens (left), the same barrel costs more in foreign currency and foreign producers' costs fall, so the dollar price tends down. When the dollar weakens (right), the reverse. The rest of this post fills in every box — and then shows you the cases where the seesaw locks up and both ends rise together.

## Foundations: why a commodity price is really a dollar price

Start from zero, because the whole thing rests on one fact that sounds trivial and is not: **almost every globally traded commodity is quoted in US dollars.** A barrel of Brent crude, a tonne of copper on the London Metal Exchange, an ounce of gold, a bushel of Chicago wheat, a metric ton of Arabica coffee — the screen price is a dollar price. Not because anyone legislated it, but because of a deep accident of history and economics: after World War II the dollar became the world's reserve and invoicing currency, the oil trade settled in dollars (the **petrodollar** arrangement), and once a deep, liquid market exists in one currency, everyone converges on it. A single global price in a single currency is enormously convenient — it lets a Chilean miner, a Korean smelter, and a Swiss trading house all transact against the same number. (For the deeper story of how the dollar came to rule global trade and finance, see the cross-links at the end.)

Here is the consequence that drives everything. If oil is quoted in dollars, then the price you actually pay depends on **two** things, not one: the dollar price of the oil, and the exchange rate between your currency and the dollar. A French refiner does not care about "\$80 oil" as a bare number — they care about how many euros they must hand over to buy that \$80 barrel. And that euro cost moves whenever *either* the oil price moves *or* the euro–dollar exchange rate moves. The dollar's strength is, in a very real sense, baked into every commodity price for everyone who does not earn dollars — which is most of the planet.

Let us define the few terms we need, plainly.

- The **US dollar index (DXY)** is a single number that tracks the dollar's value against a basket of six major currencies (the euro dominates the basket at about 58%, then the yen, pound, Canadian dollar, Swedish krona, and Swiss franc). When DXY rises, the dollar has strengthened against that basket; when it falls, the dollar has weakened. It is the market's standard gauge of "is the dollar strong or weak right now." It is not perfect — it is heavily euro-weighted and ignores the Chinese yuan entirely — but it is the number quoted on every screen.
- **EURUSD** is the price of one euro in dollars. At EURUSD 1.20, one euro buys \$1.20; the euro is "strong." At EURUSD 1.00 (parity), one euro buys exactly \$1.00; the euro is weaker and the dollar is correspondingly stronger.
- **A strong dollar** means it takes *fewer* dollars to buy a unit of foreign currency, or equivalently *more* foreign currency to buy one dollar. When the dollar strengthens against the euro, EURUSD *falls* (say from 1.20 to 1.00).

Now the core claim, stated precisely: **a stronger dollar tends to push the dollar price of commodities down, and a weaker dollar tends to push it up.** This is the inverse relationship. It runs through two channels that work in the same direction — a demand channel and a supply channel — and we will build each from scratch.

One more piece of foundation before we do, because it explains a lot of the "why doesn't the rule work?" frustration. The DXY is not a perfect gauge of "the dollar versus the world." It is a fixed basket designed in 1973 and barely changed since, and it is dominated by the euro (about 58% of the weight) with the yen, pound, Canadian dollar, krona, and franc making up the rest. It contains **no** Chinese yuan, no emerging-market currencies, no Indian rupee, no Brazilian real, no Mexican peso. So the DXY really measures the dollar against a handful of rich-world currencies, not against the currencies of the countries that often drive marginal commodity demand. A "strong DXY" can coexist with a dollar that is actually *weakening* against the yuan, which matters enormously if China is the swing buyer of the commodity you care about. Professionals therefore often prefer a **trade-weighted** or **broad** dollar index that includes emerging-market currencies. When you read "the dollar," ask *which* dollar — the rich-world DXY or the broad trade-weighted measure — because the two can diverge for months, and the broad one is usually the better commodity-demand gauge.

## The demand channel: a barrel costs more abroad

The first and most important channel is on the demand side, and it is pure arithmetic. When the dollar strengthens, every buyer who earns a non-dollar currency must spend more of their own money to buy the same dollar-priced barrel. More expensive means less demanded — that is the most basic law in economics — so foreign demand softens. Softer demand at the margin pulls the dollar price down.

The figure below makes the arithmetic concrete: one \$80 barrel, three foreign buyers, before and after a 12% dollar rally. The dollar price never changes. Only the dollar moves. Yet every foreign buyer's real cost jumps.

![An 80 dollar barrel priced in euros yen and rupees before and after a 12 percent dollar rally](/imgs/blogs/the-dollar-and-commodities-why-a-strong-dollar-weighs-on-prices-4.png)

Read the top row. A euro-zone refiner buying an \$80 barrel when EURUSD is 1.20 pays €66.7 (because 80 ÷ 1.20 = 66.7). If the dollar then rallies 12% — EURUSD falls from 1.20 to about 1.07 — that same \$80 barrel now costs €74.7. The oil price on the screen did not move one cent. But the refiner's cost rose 12%, entirely because the dollar got stronger. A Japanese utility and an Indian importer see exactly the same 12% jump in their own currencies. Multiply that across every non-dollar consumer of oil on Earth and you have a meaningful drag on global demand — and therefore on the dollar price.

#### Worked example: a euro buyer's cost at two dollar levels

Take a fixed \$80 barrel and hold the oil price perfectly constant. We will only move the dollar.

- At EURUSD **1.20** (a relatively weak dollar), the euro cost is \$80 ÷ 1.20 = **€66.7** per barrel.
- At EURUSD **1.00** (parity — a much stronger dollar), the euro cost is \$80 ÷ 1.00 = **€80.0** per barrel.

That is a jump from €66.7 to €80.0 — a **+20%** increase in the real cost to the euro-zone buyer — with the oil price unchanged at \$80 the entire time. If the price of anything you buy rises 20%, you buy less of it. So a stronger dollar mechanically suppresses foreign oil demand, and weaker demand pulls the dollar price of oil down until the market re-clears. **The dollar's move alone re-prices the commodity for most of the world, which is why a king dollar is a headwind for commodity prices even when nothing about the physical barrel has changed.**

This is not a small population of buyers. The United States consumes a large share of the world's oil, but the rest of the world — Europe, China, India, Japan, the emerging markets — consumes the majority, and they pay in their own currencies. When the dollar is strong, all of them are facing higher real prices simultaneously. That synchronized demand squeeze is the demand channel.

How big the squeeze is depends on **price elasticity** — how much demand actually falls when the local-currency price rises. For oil and other energy in the short run, demand is fairly *inelastic*: people still have to drive to work and heat their homes, so a 12% price rise does not cut consumption 12%. That inelasticity is why the demand channel, while real, is a marginal effect of a few percent on the price rather than a collapse — it nudges the market rather than breaking it. But elasticity rises over time (people buy more efficient cars, switch fuels, insulate homes) and is higher for discretionary commodities. It is also far higher for the most financially-stretched buyers, which is exactly why the emerging-market amplifier we will meet shortly is so violent: those buyers cannot simply absorb a 25% cost jump, so their demand does not bend, it snaps. The elasticity of the marginal buyer is what turns a gentle currency headwind into, occasionally, a demand cliff.

## The supply channel: a foreign producer's costs fall

The second channel works on the supply side and reinforces the first. Think about a commodity producer outside the United States — a Brazilian iron-ore miner, a Russian oil major, a Chilean copper mine. They *sell* their output in dollars (that is the world price), but they *pay* most of their costs — wages, diesel, electricity, local taxes — in their own local currency. When the dollar strengthens against that local currency, something quietly powerful happens to their economics: their dollar revenue now converts into *more* local currency, while their local-currency costs are unchanged. Their margin, measured in local money, expands.

A producer with a fatter local-currency margin can afford to sell at a *lower* dollar price and still make money. In a competitive global market, that is exactly what tends to happen — producers compete the dollar price down, because each of them is now comfortable accepting less per barrel or per tonne in dollar terms. So the supply channel pushes in the same direction as the demand channel: a stronger dollar lowers the dollar price.

#### Worked example: a foreign producer's margin when the dollar rises

A Brazilian iron-ore producer sells ore at **\$100 per tonne** (the world dollar price) and has all-in cash costs of **R\$300 per tonne** (Brazilian reais), paid locally.

- At an exchange rate of **5.0 reais per dollar**, the \$100 sale converts to R\$500. Margin = R\$500 − R\$300 = **R\$200 per tonne**.
- Now the dollar strengthens to **6.0 reais per dollar** (the real weakened 20%). The same \$100 sale now converts to R\$600. Margin = R\$600 − R\$300 = **R\$400 per tonne** — it doubled, with the dollar price unchanged.

The producer's local-currency profit doubled purely because the dollar got stronger against the real. They could now cut their dollar price toward \$83 a tonne and still earn the same R\$200 margin they started with. **A strong dollar effectively subsidizes foreign producers, giving them room to sell cheaper in dollar terms — so the supply side of the market pushes the dollar price down at the same time the demand side does.** Two channels, one direction. That is why the inverse correlation is as strong as it is.

It is worth being honest about the limits of the supply channel, because it is real but slower than the demand channel. A producer does not cut their dollar price the morning the dollar rallies; the competitive pressure plays out over months as new supply gets committed and contracts reprice. The demand channel — foreign buyers facing higher local-currency costs — bites faster, because purchasing decisions adjust in weeks. So in a sharp, short dollar move, the demand channel does most of the work; the supply channel reinforces it over the following quarters. There is also a feedback loop worth noting: a strong dollar that fattens foreign producers' margins encourages them to *expand* output (drill more, mine more), and the extra supply that eventually arrives pushes the dollar price down further still. The currency move plants a supply response that keeps weighing on the price long after the dollar itself has stopped moving.

There is a quieter financing dimension too, and it connects straight to this series' spine — the cost of carry. Commodities are physical things that must be stored, and storage is financed, overwhelmingly in dollars. When the dollar is strong and dollar interest rates are high (the two usually travel together, since high US rates are what make the dollar strong), the cost of financing inventory rises. Traders and producers respond by drawing down stocks rather than holding them, releasing supply onto the market and pressuring the price. So the same conditions that strengthen the dollar — high US real rates — also raise the carrying cost of the physical commodity, adding yet another downward nudge through the storage-and-financing channel. The forward curve, the cost of carry, and the dollar are not separate stories; they are gears in the same machine.

## Putting it together: the dollar and oil over a decade

Theory is clean; the data is messier — and the mess is the most useful part. The figure below overlays the dollar index against the annual average WTI crude price from 2014 to 2025.

![The US dollar index and WTI crude oil annual averages from 2014 to 2025 showing a broad inverse relationship](/imgs/blogs/the-dollar-and-commodities-why-a-strong-dollar-weighs-on-prices-2.png)

Look at the broad shape first. The mid-2010s tell the textbook story: through 2014–2016 the dollar surged (the index climbed from about 90 to over 100 as the Federal Reserve signaled tightening while the rest of the world eased), and oil collapsed — WTI fell from over \$90 a barrel to the \$40s, a **−47.8%** crash in 2015 alone. Strong dollar, weak oil: the inverse working exactly as the mechanism predicts. (The 2014–2016 oil crash was not *only* a dollar story — the US shale boom flooded the market with supply — but the strong dollar was a genuine part of the squeeze, magnifying the fall for every foreign producer's competitiveness and every foreign buyer's appetite.)

Now find 2022 on the chart, marked in red. Here both lines rise together. The dollar index jumps to its highest level in twenty years *and* oil spikes. The seesaw locked up. We will dissect why in a moment, but notice the lesson already: across a single decade you can see the inverse holding cleanly in some years and breaking spectacularly in others. The relationship is a strong default, not a constant.

#### Worked example: estimating the dollar drag on an oil view

Suppose you have a fundamental view that oil "should" be worth \$80 a barrel based on supply and demand, and today it trades at \$80 with DXY at 100. Now you expect the Fed to out-hike the rest of the world and push DXY up 10% to 110 over the next year, with no change in physical fundamentals.

A common rough rule of thumb among macro desks is that a 10% move in the trade-weighted dollar is associated with something on the order of a **10–20% inverse move** in broad commodity prices, all else equal — though the elasticity varies a lot by commodity and regime. Take the conservative end, ~10%. Your \$80 oil view, adjusted for the dollar headwind alone, becomes roughly **\$72** — a \$8 haircut driven entirely by the currency, not the barrel. **The dollar move is not a side issue you can ignore; for a one-year oil view it can be worth several dollars a barrel, which is why it belongs in the model as an explicit input.** The caveat — and it is a big one — is that "all else equal" almost never holds, which is the whole back half of this post.

## The dollar and copper: the same inverse, noisier

Oil is the cleanest case because it is the most dollar-sensitive commodity and the deepest market. But the channel applies to every dollar-priced commodity. The figure below runs the same overlay for copper — "Dr. Copper," the industrial metal whose price reads the global growth cycle.

![The US dollar index and LME copper annual averages from 2014 to 2025 showing a noisier inverse relationship](/imgs/blogs/the-dollar-and-commodities-why-a-strong-dollar-weighs-on-prices-3.png)

The broad inverse is there — note 2015–2016, where the strong dollar coincided with copper bottoming near \$4,800 a tonne — but it is visibly noisier than oil. The reason is instructive: copper is driven *even more* by a separate force, global growth, and especially Chinese demand. In 2024–2025 copper pushed to record highs (around \$9,150 a tonne on average, with an intraday record near \$11,104 in May 2024) *despite* a firm dollar, because the electrification and supply-gap story — the structural copper bull thesis — simply overwhelmed the currency drag. The dollar was a headwind, but the growth tailwind was stronger.

This is the first hint of a deeper truth we will formalize shortly: the dollar inverse is one factor among several, and whether it dominates depends on what else is happening. For a metal as growth-sensitive as copper, the growth factor often wins. (For the full copper story — the ore-to-cathode chain, China's dominance, and the electrification supply gap — see the within-series Dr. Copper post linked at the end.)

## Not all commodities are equally dollar-sensitive

A crucial refinement: the inverse is much stronger for some commodities than others, and knowing the ranking saves you from misapplying the rule. The dollar-sensitivity of a commodity depends mainly on three things — how globally traded it is, how much of its demand is non-dollar, and how big its own idiosyncratic supply-and-demand swings are relative to the currency effect.

**Gold and oil sit at the high-sensitivity end.** Gold is the most dollar-sensitive of all, because it is a monetary asset bought largely as a store of value, so its price is almost a *pure* function of the dollar and real rates with little independent industrial demand to muddy the signal. Oil is the most dollar-sensitive *consumption* commodity: it is the deepest, most globally traded market on Earth, the majority of demand is non-dollar, and it has no large idiosyncratic non-macro buyer. When the dollar moves and nothing physical is happening, gold and oil respond most cleanly.

**Industrial metals sit in the middle.** Copper, aluminum, and nickel feel the dollar channel, but their own growth-and-China story is so large that it frequently dominates, as we just saw with copper's 2024–2025 records against a firm dollar. The dollar is a real but often secondary factor here.

**Agricultural commodities sit at the low-sensitivity end.** Corn, wheat, soybeans, and the softs are dominated by weather, harvests, planting decisions, and export policy — physical factors that swamp the currency effect most of the time. A drought in the US Midwest or an export ban in India will move wheat far more than any plausible dollar swing. The dollar still matters at the margin (a strong dollar makes US grain exports less competitive against Brazilian or Russian supply), but it is rarely the headline.

#### Worked example: ranking the dollar drag across the complex

Suppose the broad dollar rises 10% in a quiet, monetary-driven year with no major supply shocks anywhere. Estimate the rough dollar drag on each commodity, using the sensitivity ranking:

- **Gold:** highly sensitive — a 10% dollar move might pull gold down on the order of 10–15%, because almost nothing else is driving it.
- **Oil:** highly sensitive — perhaps a 10–20% drag, the cleanest consumption-commodity response.
- **Copper:** moderately sensitive — maybe 5–10%, often offset by its own growth story.
- **Wheat:** weakly sensitive — perhaps 2–5%, easily overwhelmed by the next weather report.

**The same 10% dollar move is worth perhaps three times as much to oil as to wheat, so "the dollar is strong, sell commodities" is far too blunt — you must weight the dollar by how dollar-sensitive the specific commodity actually is.** A macro trader uses the dollar as a strong input for oil and gold, a moderate one for metals, and a minor footnote for grains, where the WASDE report and the weather map are what really move the price. (For the grain markets specifically, the within-series and cross-asset agriculture posts go deep on what actually drives them.)

## The two-way street: commodities can move the dollar too

So far we have treated the dollar as the cause and the commodity as the effect. Reality is a two-way street. A large, sustained move in commodity prices can itself push the dollar around — and ignoring that causality running the other way is a classic mistake.

The channel is called **terms of trade**. A country's terms of trade is the ratio of its export prices to its import prices. When commodity prices boom, commodity-*exporting* countries (Australia, Canada, Saudi Arabia, Brazil, Norway) see their export revenues swell, their currencies tend to strengthen, and money flows toward them. Commodity-*importing* countries (Japan, much of Europe, India) see their import bills balloon and their currencies tend to weaken. The dollar is a peculiar case: the US is a huge oil consumer but, since the shale revolution, also the world's largest oil *producer*, so its net commodity exposure is far smaller than it used to be. That partly explains why, in some commodity booms, the dollar can rise alongside commodities rather than fall.

There is also a recycling channel specific to oil. When oil prices are high, oil-exporting nations earn enormous dollar surpluses (the petrodollar flow) and recycle them into dollar assets — US Treasuries, US deposits — which can support the dollar. So a high oil price can, through this plumbing, actually *bolster* the dollar rather than weaken it. The simple "dollar up → oil down" arrow has a quieter "oil up → dollar up" arrow running underneath it. When the second arrow is strong enough, the headline inverse appears to fail.

There is one more wrinkle that breaks the simple inverse, and it is called the **dollar smile**. The dollar tends to strengthen in two opposite environments: when the US economy is booming and US real rates are high (capital flows in to chase yield), *and* when the world is in panic and everyone scrambles for the safety of dollars (a flight to quality). It is weak only in the comfortable middle — synchronized global growth with calm markets. The two ends of that "smile" have opposite implications for commodities. A boom-driven strong dollar coincides with decent commodity demand (the growth is real), partly offsetting the currency drag. A panic-driven strong dollar coincides with collapsing risk appetite and crashing commodity demand, so both the currency channel *and* the demand collapse push commodities down together — a brutal combination. The same "strong dollar" therefore means very different things for commodities depending on *which side of the smile* produced it. In March 2020 the dollar spiked on panic and commodities cratered; the inverse looked super-charged. In a calm tightening cycle the dollar strengthens on growth and the commodity drag is gentler. (The macro series' "dollar smile" post unpacks this fully.)

The practical takeaway is humility about causation. When you see the dollar and a commodity moving, you cannot always say which is steering. Usually the dollar is the bigger, faster-moving variable and leads — but in a genuine commodity supercycle, the commodity can be the dog and the dollar the tail.

## It is really about real rates and the dollar together

Now we add the layer that separates a sophisticated read from a naive one. The dollar does not move for its own sake. It moves mostly because of **real interest rates** — the interest rate after subtracting expected inflation. When US real rates rise relative to the rest of the world (because the Fed is hiking, or US growth is strong), global capital flows into dollar assets to earn that higher real yield, and the dollar strengthens. So the dollar is, to a large degree, a *messenger* for the real-rate story underneath it.

This matters because real rates hit commodities through a *second*, independent channel beyond the currency one. A commodity sitting in a tank or a warehouse pays no interest and no dividend — it is a zero-yield asset. When real rates are high, the **opportunity cost** of holding a zero-yield asset rises: why tie up capital in a barrel of oil or a bar of metal earning nothing when a Treasury inflation-protected security pays you a real 2%? High real rates also raise the cost of financing inventory, encouraging producers and traders to draw down stocks and discouraging speculative hoarding. Both effects weigh on commodity prices — entirely separately from the exchange-rate channel.

So when you observe a strong dollar weighing on commodities, you are usually watching *two* forces that ride together: the currency channel (foreign buyers pay more, foreign producers can sell cheaper) *and* the real-rate channel (zero-yield assets look worse when real yields are high). They tend to move together because the same thing — rising US real rates — drives both. This is exactly the framework that governs gold, where the real interest rate is the single master variable, and the dollar is its close companion. (For the full real-rate treatment, the gold series' master-variable post is the canonical reference — linked at the end. The logic transfers directly to the broader complex.)

#### Worked example: separating the dollar channel from the real-rate channel

In 2022 the Fed hiked the fed funds rate from near zero to over 4% in a single year — the fastest tightening cycle in four decades. Walk the two channels through that event:

- **Dollar channel:** higher US real rates pulled capital into dollars, DXY rose +8.2% on the year (and intraday hit ~114.8 in September). For every foreign commodity buyer, that 8% dollar move added roughly 8% to the local-currency cost of every dollar-priced commodity — a demand headwind.
- **Real-rate channel:** the 10-year US real yield swung from about **−1%** at the start of 2022 to about **+1.6%** by year-end — a ~2.6 percentage-point jump. That made zero-yield commodities far less attractive to hold and raised the cost of financing inventory — a separate headwind.

Both channels pointed the same way: down. **And yet broad commodities still rose +16% that year**, because a *third* force — the war-driven supply shock — was bigger than both currency and real-rate headwinds combined. The two-channel framework correctly told you commodities faced strong macro headwinds; it just got swamped by a physical-supply earthquake. **The framework is a map of the forces, not a prediction of the outcome — you still have to weigh which force is largest.**

## The emerging-market amplifier

There is a third, more violent channel that deepens the demand effect, and it operates through the emerging markets. Most developing economies are heavy commodity *importers* — they buy oil, gas, grain, and metals priced in dollars — and many of them have also *borrowed* in dollars (because dollar debt is cheaper and deeper than local-currency debt). When the dollar surges, these countries get hit twice at once: their import bill balloons in local-currency terms, *and* the local-currency cost of servicing their dollar debt jumps. That double squeeze forces them to tighten — raise rates, cut spending, ration imports — which crushes their own demand for commodities precisely when the dollar is strong.

So a strong dollar does not just make commodities marginally more expensive abroad; it can trigger genuine demand destruction in the very economies that were the growth engine of commodity consumption. The 2013 "taper tantrum" and the 2018 and 2022 dollar surges all coincided with emerging-market stress — currency crises in Turkey, Argentina, and elsewhere — that fed back into softer commodity demand. This is the demand channel on steroids: not a smooth price elasticity but a discontinuous, crisis-driven collapse in the buying power of a large bloc of consumers.

#### Worked example: an emerging-market importer's double squeeze

Take an emerging-market country that imports 1 million barrels of oil a day at \$80 and has \$50 billion of dollar-denominated debt. Its currency is at 10 units per dollar.

- **Oil import bill:** 1,000,000 × \$80 × 10 = **800 million local units per day**.
- Now the dollar strengthens 25% — the local currency falls to 12.5 per dollar.
- **New oil import bill:** 1,000,000 × \$80 × 12.5 = **1.0 billion local units per day** — up 25%, with the oil price unchanged.
- **Debt servicing:** the local-currency cost of that \$50bn debt also rose 25%, from 500bn to 625bn local units.

Both bills jumped 25% simultaneously, draining the country's reserves and forcing austerity. **A strong dollar can turn a commodity-importing emerging market from a growth engine into a forced seller of demand, which is why dollar surges so often coincide with the worst commodity demand environments — the currency effect and the credit effect compound.** This amplifier is a big part of why the inverse, when it works, can work so violently. (The macro series' "dollar as a wrecking ball for emerging markets" post goes deep on this transmission.)

This is also why the inverse can feel asymmetric in practice. A weak dollar relieves the emerging-market squeeze gradually and lets demand recover, but a strong dollar can trigger the squeeze suddenly, in a crisis-shaped spike. Traders often observe that the dollar–commodity relationship is fiercest on the way *up* in the dollar — when a surge tips a fragile importer into a balance-of-payments crisis — and gentler on the way down. The headwind is not a smooth, symmetric breeze; it gusts hardest precisely when a strong dollar collides with a financially stretched buyer, and that nonlinearity is part of what makes the relationship so hard to capture in a single tidy correlation number.

## When the inverse breaks: the 2022 case

Now we confront the exception head-on, because understanding *when* the rule fails is more valuable than the rule itself. The figure below isolates the year-over-year change in the dollar and in oil, side by side, for 2019 through 2024.

![Year over year change in the dollar index and WTI crude showing 2022 when both rose together](/imgs/blogs/the-dollar-and-commodities-why-a-strong-dollar-weighs-on-prices-5.png)

In a "normal" year the two bars point opposite ways — one up, one down — which is the inverse. But look at 2022: both bars point up, and oil's is enormous. The dollar rose **+8.2%** and WTI rose **+39.3%** in the same year. The seesaw didn't just wobble; it broke.

The reason is simple once you name it: **a supply shock can override the currency channel entirely.** Russia's invasion of Ukraine in February 2022 threatened to remove millions of barrels a day of Russian crude and a huge slice of European natural gas and Ukrainian/Russian wheat from the world market. When the *physical availability* of a commodity is suddenly in question, the price spikes regardless of what the dollar is doing — because no exchange-rate arithmetic can manufacture barrels that aren't there. The demand channel (foreign buyers paying more) and the supply channel (foreign producers' margins) are both *marginal* effects, worth a few percent. A war that threatens 5–10% of global supply is a *first-order* effect worth tens of percent. First-order beats marginal every time.

#### Worked example: the 2022 both-up year, decomposed

Decompose 2022's oil move into the forces pulling each way:

- **Dollar headwind (inverse, pushing oil down):** DXY +8.2% should have shaved roughly 8–16% off the dollar oil price, all else equal — a drag of maybe \$6–\$12 a barrel off a \$80 base.
- **Real-rate headwind (pushing oil down):** the ~2.6-point jump in real yields added more drag on top.
- **Supply shock (pushing oil up):** the threat to Russian supply and the scramble to refill depleted inventories added *tens* of dollars — WTI's average rose from \$68 in 2021 to \$95 in 2022, and Brent spiked to ~\$139 intraday in March.

Net result: oil rose +39% *despite* two macro headwinds, because the supply shock was several times larger than both combined. **The inverse rule did not "fail" — it was simply outvoted by a bigger force, which is exactly what supply shocks do.** A trader who shorted oil in early 2022 "because the dollar is strong" had the macro logic right and the position catastrophically wrong, because they mistook a tendency for a law.

The scatter below makes the breakability undeniable. Each dot is one year from 2015 to 2025: the dollar's change on the horizontal axis, oil's change on the vertical. Green dots are years the inverse held (opposite signs); red dots are years it broke (both moved the same way).

![Scatter of yearly dollar change versus yearly oil change from 2015 to 2025 with a weak trend line](/imgs/blogs/the-dollar-and-commodities-why-a-strong-dollar-weighs-on-prices-7.png)

Here is the honest, slightly uncomfortable result. The fitted trend line through these eleven annual points is **not** cleanly downward-sloping — the correlation is a weak **+0.23**, essentially noise. At the *annual* frequency over this particular window, the dollar–oil inverse barely shows up at all, because the window is dominated by the COVID crash, the 2021 reopening surge, and the 2022 war — three years where *both* the dollar and oil were jerked around by the same enormous global events (a flight to safety, then a supply-and-demand whipsaw). The clean inverse you saw in 2014–2016 is real, but it gets buried at the annual scale by a handful of supply-and-crisis-driven years.

This is the single most important slide in the post, and it is a warning. **The dollar–commodity inverse is robust at high frequency (day-to-day, week-to-week, when the dollar is the main thing moving) and over long, calm windows — but it is weak and unreliable when measured crudely across years dominated by supply shocks.** The pattern is a conditional one: it holds when the dollar is the dominant driver and dissolves when something physical takes over. Anyone who quotes a single "dollar-commodity correlation" number without specifying the frequency and the regime is selling you a false certainty.

### Why the frequency matters so much

It is worth pausing on *why* the inverse is strong at the daily scale and weak at the annual scale, because it is not a contradiction — it is the signature of how the two channels actually operate. On any given trading day, the biggest mover of both the dollar and commodity prices is *news about US monetary policy and risk appetite* — a hot inflation print, a hawkish Fed speaker, a risk-off shock. That news hits the dollar and commodities through the same macro nerve at the same instant, and because the news is monetary, it moves them in opposite directions: the dollar up and commodities down, or vice versa. So at high frequency, in the absence of a fresh physical supply event that day, the monetary channel dominates and the inverse is tight. Studies of daily and weekly data routinely find a negative dollar–commodity correlation of meaningful magnitude.

Stretch the window out to a full year, though, and you are now summing over *every* force that moved oil that year — wars, OPEC decisions, recessions, inventory cycles, refinery outages — and many of those are physical supply-and-demand events that have nothing to do with the dollar and dwarf it in size. The monetary signal that was clean at the daily scale gets buried under physical noise at the annual scale. That is precisely why our 2015–2025 scatter showed a near-zero correlation: those eleven years happened to contain three of the most supply-and-crisis-driven years in modern history (2020, 2021, 2022), and the physical earthquakes drowned out the steady monetary hum.

The practical implication is sharp. **If you are trading the dollar–commodity inverse, trade it at the frequency where it is real — intraday and over weeks, when the dollar is the marginal mover — and do not expect it to "pay off" cleanly over a year studded with supply shocks.** And when you build a fundamental view over months, treat the dollar as one slowly-moving input among several, not as a high-confidence directional signal.

## A longer historical sweep: the inverse across the decades

The pattern is not a recent quirk; it has shaped commodity markets since the dollar was untethered from gold in 1971. A quick tour shows both the durability of the rule and how regime-dependent it is.

**The 1970s — a weak dollar and the great commodity boom.** Through the 1970s the dollar weakened structurally as the US ran inflation and abandoned the gold standard, and commodities had one of their great bull markets — oil quadrupled in 1973–1974 and again in 1979, gold ran from \$35 to over \$800, and the whole complex soared. Part of that was genuine supply shock (the OPEC embargoes), but the weak, inflating dollar was the macro backdrop that let it run. This is the inverse in its grandest form: a debased dollar and a roaring commodity complex, over a full decade.

**The early 1980s — Volcker, a strong dollar, and the bust.** When Paul Volcker crushed inflation with brutally high interest rates, US real rates soared, the dollar surged to extraordinary heights (so high that the 1985 Plaza Accord was needed to coordinate it back down), and commodities entered a long bear market. Strong dollar, high real rates, falling commodities — the textbook regime, playing out over years rather than days. The 1985 Plaza Accord's engineered dollar weakening then helped commodities and exporters recover, a clean demonstration of the causality.

**The 2000s — a weak dollar and the China supercycle.** Through the 2000s the dollar drifted weaker while China's industrialization drove the greatest commodity supercycle in modern history — copper went from under \$2,000 to nearly \$9,000 a tonne, oil from \$25 to \$147. Here the two-way street is visible: the commodity boom was so powerful that it strengthened exporter currencies and arguably contributed to dollar weakness, rather than the dollar driving the commodities. The dog was the commodity; the dollar was partly the tail.

The throughline across fifty years: the inverse is a *real, durable macro relationship* that holds over long monetary regimes — weak-dollar decades are commodity-boom decades and vice versa — but at any given moment it can be overridden by a supply shock or reversed in causation by a supercycle. The decade-scale evidence is far cleaner than the year-to-year noise, which is exactly why our annual scatter looked so weak: eleven years is too short a window, dominated by a few crisis years, to show a relationship that really lives over decades and over high-frequency moves.

## A regime map: when the inverse holds and when it breaks

We can now organize the whole thing into three regimes. The decision grid below is the keeper from this post: identify which regime you are in *first*, then decide how much weight to give the dollar.

![A regime grid showing when the dollar commodity inverse holds weakens or breaks across demand-driven growth-factor and supply-shock cases](/imgs/blogs/the-dollar-and-commodities-why-a-strong-dollar-weighs-on-prices-6.png)

Read it column by column.

- **Demand- or money-driven (the base case).** The dominant force is monetary: Fed policy, real rates, risk appetite. The dollar leads, and the commodity moves opposite. The inverse holds *strongly*. This is the regime where "strong dollar = commodity headwind" is a genuinely useful read. Most ordinary market conditions live here.
- **Common global-growth factor.** A worldwide boom or bust lifts or sinks both the dollar and commodities through the *same* growth channel, so the dollar becomes a sideshow and the measured correlation goes mushy. Here you should trade the growth story directly — watch PMIs, China, global activity — and not lean on the dollar at all.
- **Supply shock dominates.** A war, an OPEC production cut, a drought, a refinery outage. The commodity spikes on its own, and the dollar may even rise *with* it as a safe haven. The inverse breaks outright — 2022 — and the dollar is the wrong thing to be watching. Price the shock; ignore the currency.

The skill is regime identification. Before you let the dollar inform a commodity view, ask: *is the dollar the main thing moving right now, or is something physical?* If a tanker just got torpedoed in the Strait of Hormuz, the dollar is irrelevant to your oil view this week. If the Fed just surprised hawkish on an otherwise quiet day, the dollar is the story and the inverse is your friend.

## Common misconceptions

**"A strong dollar always means lower commodity prices."** No — it means *lower, all else equal*, which is a meaningful tendency but not a guarantee. In 2022 the dollar rose +8.2% and oil rose +39.3% because a supply shock dwarfed the currency channel. The dollar is a headwind, not a verdict.

**"The dollar–commodity correlation is a fixed number you can rely on."** No — it is regime-dependent and frequency-dependent. The clean inverse of 2014–2016 nearly vanishes when you measure 2015–2025 annual data crudely (correlation ~+0.23) because that window is full of supply-and-crisis years. Quote a correlation only with its frequency and regime attached.

**"The dollar causes commodity moves; commodities don't affect the dollar."** No — causation runs both ways. A genuine commodity boom strengthens exporter currencies and, through the petrodollar recycling channel, can support the dollar itself. In a supercycle the commodity can lead and the dollar follow.

**"It's just about the exchange rate."** No — the dollar is largely a messenger for **real interest rates**, which hit commodities through a *second*, independent channel: zero-yield assets look worse and inventory financing gets dearer when real yields are high. You are usually watching two forces ride together, not one.

**"If the dollar weakens, I should automatically buy commodities."** No — a weak dollar removes a headwind, but the commodity still needs a reason of its own (supply tightness, growth, a real-rate tailwind). The dollar is one input to a view, never the entire thesis. In the 2024–2025 stretch copper hit records *despite* a firm dollar because its own growth story was the bigger force.

## How it shows up in real markets

**2014–2016: the textbook inverse.** As the Fed wound down quantitative easing and signaled rate hikes while Europe and Japan eased, the dollar index surged from about 80 in mid-2014 to over 100 by early 2015. Oil, simultaneously hit by the US shale supply glut, collapsed from over \$100 to under \$30 a barrel at the trough. WTI's annual average fell **−47.8%** in 2015 alone. The strong dollar was not the only cause — shale supply was huge — but it magnified the move for every foreign buyer and producer. This is the regime where the inverse is your friend: monetary-driven dollar strength, commodities falling opposite.

**2020–2021: the COVID whipsaw.** When the pandemic hit in March 2020, the dollar first *spiked* in a global dash for safety (everyone wanted dollars at once — the global dollar-shortage dynamic), and commodities crashed on collapsing demand — oil infamously went *negative* on 20 April 2020. Then, as the Fed flooded the world with liquidity and the dollar weakened through late 2020, commodities roared back. Here the inverse worked in the *recovery* phase but was scrambled at the panic peak, when both the dollar and the safe-haven bid moved together. A reminder that crisis-flight regimes can briefly break the rule in either direction.

**2022: the supply-shock break.** Already dissected above. The dollar at a two-decade high *and* oil up +39.3%, because Russia's invasion was a first-order supply shock that overrode the currency and real-rate headwinds. Broad commodities (the Bloomberg index) returned **+16%** in 2022 — the only major asset class besides the dollar itself to gain in a brutal year for stocks and bonds. The inverse didn't fail; it got outvoted.

**2024–2025: the growth factor wins for copper.** The dollar stayed firm (DXY rose to ~108.5 at end-2024) as US growth outpaced the world, yet copper pushed to record highs near \$11,104 intraday in May 2024. The electrification supply-gap thesis — surging demand from EVs and the grid against a decade-long lag in new mine supply — was simply a bigger force than the dollar headwind. The dollar mattered; the growth story mattered more. Regime two: trade the growth, not the FX.

## The playbook: using the dollar as one input, not a switch

Here is how a working macro trader actually uses all this. The dollar is **one input to a commodity view, weighted by regime — never a mechanical rule.**

**Step 1 — identify the regime before anything else.** Ask the single most important question: *is the dollar the dominant force right now, or is something physical?* Scan for active supply shocks — wars, OPEC meetings, sanctions, weather, strikes. If a first-order supply event is live, set the dollar aside entirely and price the shock. If markets are calm and monetary, the dollar inverse is back in play.

**Step 2 — in monetary regimes, treat a strong dollar as a quantifiable headwind.** Use a rough elasticity (a 10% trade-weighted dollar move maps to roughly a 10–20% inverse commodity move, varying by commodity) to haircut or boost your fundamental view. For a one-year oil call, that can be worth several dollars a barrel — material, not noise.

**Step 3 — watch real rates, not just the exchange rate.** Because the dollar is largely a messenger for US real rates, the cleaner causal variable is often the real yield itself. Rising real yields are a double headwind for commodities (currency channel plus opportunity-cost channel). Track the 10-year real yield alongside DXY; when they move together, the signal is strong; when they diverge, dig into why.

**Step 4 — respect the two-way causation.** In a genuine supercycle, don't assume the dollar is steering. If commodities are in a structural bull market on supply scarcity, the dollar may follow the commodity rather than lead it, and the inverse can invert. Know which is the dog and which is the tail.

**Step 5 — never short a commodity "because the dollar is strong" alone.** That is the 2022 mistake. The dollar is a contributing factor, worth a few percent at the margin; it is never a standalone thesis. A short needs its own reason — a supply surplus, a demand cliff, a broken curve — with the dollar as a tailwind, not the whole case.

**Step 6 — size the dollar exposure you are implicitly taking.** Any long commodity position is, hidden inside it, a short-dollar bet. If you are long a broad commodity index and the dollar rallies in a quiet monetary regime, you will lose money even if your supply-and-demand view is dead right. Sophisticated desks measure this with a **beta to the dollar** — roughly, how much the commodity moves for a 1% move in the dollar — and either accept it as part of the trade or hedge it out with an FX position. The worked example below shows how to do the arithmetic.

#### Worked example: a commodity basket's beta to the dollar

You hold a \$10 million long position in a broad commodity index. From history, you estimate the index has a **dollar beta of about −1.5** in normal monetary regimes — meaning a 1% rise in the broad dollar is associated with roughly a 1.5% fall in the index, all else equal.

- The dollar rallies **4%** in a quiet, monetary-driven month with no supply shocks.
- Expected drag on your position: −1.5 × 4% = **−6%**.
- On \$10 million, that is a **−\$600,000** hit from the dollar move alone, before any change in physical fundamentals.

If you did not want that currency exposure, you could have hedged it — for instance, holding a long-dollar position sized to offset the −1.5 beta — leaving you with a cleaner bet on the commodity's own supply and demand. **The point is that you are *always* taking a dollar view when you trade commodities, whether you mean to or not; the professional move is to measure that exposure as a beta and decide deliberately whether to keep it or hedge it, rather than discovering it after a dollar rally has cost you \$600,000.** The beta is regime-dependent — it is sharper in monetary regimes and breaks down in supply shocks — so the hedge is a tool, not a guarantee.

The deeper lesson ties back to this series' spine. A commodity price is a physical thing forced through a financial contract — and that contract is denominated in dollars. The dollar is therefore stitched into every commodity price by construction, which is why the inverse exists at all. But the physical reality — how many barrels actually exist, how much copper the world actually needs — is the *thing itself*, and when the physical reality moves violently, it overrides the financial wrapper every time. The dollar tells you about the wrapper. The supply-and-demand of the physical good tells you about the contents. A complete commodity view needs both, weighted by which one is shouting loudest today.

## Further reading & cross-links

Within this series:

- [Copper: Dr. Copper and the Pulse of the Global Economy](/blog/trading/commodities/copper-doctor-copper-and-the-pulse-of-the-global-economy) — why copper's own growth factor can overwhelm the dollar headwind, the case we saw in 2024–2025.
- [Commodities as an Inflation Hedge — and When They Are Not](/blog/trading/commodities/commodities-as-an-inflation-hedge-and-when-they-are-not) — the related question of how commodities track inflation, where real rates and the dollar reappear.
- [The Commodity Supercycle: Decades-Long Booms and Busts](/blog/trading/commodities/the-commodity-supercycle-decades-long-booms-and-busts) — when the commodity becomes the dog and the dollar the tail.

Going deeper on the dollar and real rates (outside this series):

- [The Dollar System: Why USD Rules Markets (DXY)](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — the full mechanics of dollar dominance and the index.
- [Real Interest Rates: The Master Variable Behind the Gold Price](/blog/trading/gold/real-interest-rates-the-master-variable-behind-the-gold-price) — the real-rate framework that governs the whole complex, transferred from gold.
- [The Petrodollar and Dollar Dominance](/blog/trading/finance/petrodollar-and-dollar-dominance) — why oil settles in dollars in the first place, and the recycling channel that can support the dollar.
- [Real vs Nominal: Inflation, Real Yields, and the Master Signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — the nominal-versus-real distinction underneath every dollar move.
