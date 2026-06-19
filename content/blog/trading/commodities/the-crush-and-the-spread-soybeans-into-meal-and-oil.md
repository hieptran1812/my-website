---
title: "The Crush and the Spread: Soybeans Into Meal and Oil"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a soybean processor is a margin business, how the board crush spread is built from beans, meal and oil in three different units, how a crusher hedges by buying beans and selling products, and how the biodiesel boom re-priced soybean oil."
tags: ["commodities", "soybean-crush", "crush-spread", "soybean-meal", "soybean-oil", "biodiesel", "renewable-diesel", "processing-margin", "grains", "agriculture"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A soybean processor does not make money on the price of beans; it makes money on the *gap* between what it pays for a bushel of soybeans and what it sells the meal and oil crushed out of that bushel for. That gap is the **crush spread**, and it is the soybean world's version of the oil refiner's crack spread.
>
> - A crusher is structurally **long beans and short meal-plus-oil**: it buys the raw bean, sells two finished products, and earns the difference. It hedges by "putting on the crush" — buying bean futures and selling meal and oil futures so it locks the *spread*, not the level.
> - One 60-pound bushel of beans yields roughly **44 pounds of meal (about 80% by weight) and 11 pounds of oil (about 18%)**, with the rest lost as hulls and moisture. Meal is high-protein animal feed; oil is cooking oil and biodiesel feedstock.
> - The arithmetic is genuinely fiddly because the three legs trade in **three different units**: beans in dollars per bushel, meal in dollars per short ton, oil in cents per pound. You have to convert all three onto a per-bushel basis before the spread means anything.
> - The one fact to remember: the **renewable-diesel boom re-priced soybean oil** after about 2021, lifting oil's share of the crush value and pushing the board crush margin to a blowout near **\$2.10 a bushel in 2022** — even as the bean price itself was merely high, not extreme.

In the autumn of 2021, an unusual thing started happening in the back offices of the world's biggest agricultural processors — companies most consumers have never heard of, with names like Archer-Daniels-Midland, Bunge, Cargill, and Louis Dreyfus. The price of soybeans was high but not unprecedented; beans averaged around \$13 a bushel for the year, well off the \$17 spikes the market had occasionally touched. Yet the *margin* these companies earned for crushing a bean into its two products — the protein-rich meal and the golden oil — was climbing toward levels nobody under fifty had seen. By 2022 the board crush margin hit roughly \$2.10 a bushel, more than double its sleepy long-run average near \$0.85. Processors were running their plants flat out and still could not keep up.

The reason was not the bean. It was the *oil*. A wave of new renewable-diesel refineries — plants that turn vegetable oil and animal fat into diesel that burns in an ordinary truck engine — had come online across the United States, and every one of them needed a river of soybean oil to feed it. A product that for a century had been a low-value byproduct of the meal business, used for frying potatoes and making margarine, suddenly had a giant new structural buyer. Soybean oil was re-priced almost overnight, and because oil is one of the two things a crush plant sells, the entire economics of crushing a bean shifted with it.

This is the post that explains that margin from zero. By the end you will understand why a soybean processor is fundamentally a *spread* business and not a *price* business, how the board crush is built from three numbers quoted in three incompatible units, how a processor locks its margin by buying beans and selling products, why meal and oil answer to completely different demand stories, and how to read the crush spread as a live signal of processing demand and the health of the renewable-fuels boom.

![One soybean bushel splitting into meal and oil with the crush margin at the end](/imgs/blogs/the-crush-and-the-spread-soybeans-into-meal-and-oil-1.png)

## Foundations: what a crush plant actually does

Start with the physical thing. A soybean is a small, hard, oily seed — about the size of a pea, packed with two valuable substances locked together: **protein** and **fat (oil)**. On its own a raw soybean is not much use. You cannot feed whole raw beans to most farm animals in quantity (they contain anti-nutritional compounds that have to be cooked out), and you cannot pour a bean into a frying pan. The whole value of the bean is locked up inside it. A **crush plant** is a factory whose only job is to *split the bean apart* and turn it into two things people will pay a great deal for: high-protein meal and refined oil.

The splitting is a genuine industrial process, and you need the rough shape of it to follow everything that comes later.

**First, the beans are cleaned, cracked, and conditioned.** Incoming beans are screened to remove dirt and debris, then run through cracking rolls that break each bean into several pieces and loosen the thin outer **hull** (the skin). The hulls are screened off — they are mostly fibre, low in protein and oil — and are usually ground back into the meal later or sold separately as a fibre source. The cracked pieces are then heated and conditioned so the oil inside becomes easier to extract.

**Second, the conditioned flakes are pressed and solvent-extracted.** The beans are rolled into thin flakes to expose as much surface as possible, and then the oil is pulled out. Modern plants use a chemical solvent — typically **hexane** — that dissolves the oil out of the flakes far more completely than mechanical pressing alone. The hexane-and-oil mixture is then heated to boil off and recover the solvent (it is recycled, not consumed), leaving behind crude soybean oil. The de-oiled flakes that remain are toasted to drive off the last of the solvent and to deactivate the bean's anti-nutritional compounds, and the result is **soybean meal**.

**Third, the two products are finished for their markets.** The crude oil is **refined** — degummed, neutralised, bleached, and deodorised — into clear, food-grade soybean oil, or sold as crude oil to a biodiesel or renewable-diesel plant that will process it further. The toasted meal is ground to a consistent particle size and shipped to feed mills, where it becomes the protein backbone of animal rations.

So a crush plant is a splitting machine: it takes one cheap, locked-up input (the bean) and separates it into two valuable outputs (meal and oil), throwing away or down-cycling the small fibre fraction. The economics are identical to a sawmill that buys logs and sells planks and sawdust, a flour mill that buys wheat and sells flour and bran, or — the closest cousin of all — an **oil refinery** that buys crude and sells gasoline and diesel. In every one of these businesses the model is the same: buy one raw input, split it into a slate of outputs, and earn the difference between the value of the outputs and the cost of the input, minus the cost of doing the splitting. That difference-between-outputs-and-input is, in this industry, literally called the **crush**. (If the refinery comparison appeals to you, the oil-side version of this exact business is laid out in [refining and crack spreads](/blog/trading/commodities/refining-and-crack-spreads-turning-crude-into-products) — the crush is its agricultural twin, and the two posts are deliberately mirror images.)

Like a refinery, a crush plant is capital-heavy and runs continuously. A modern crush plant costs hundreds of millions of dollars and takes years to build, and once built it runs around the clock because the machinery is sunk and the only thing that varies day to day is the *margin per bushel* times the *number of bushels crushed*. This is why crushing is a classic **fixed-cost, high-throughput** business, and why every concept in this post is downstream of one mantra: keep the plant full and crush the best-margin beans.

### The split: where the bushel goes

Here is the single most important physical fact in the whole post, the one number you should tattoo on the inside of your eyelids. A soybean is sold by the **bushel**, which for soybeans is a unit of *weight*: one bushel of soybeans weighs **60 pounds**. When you crush that 60-pound bushel, it comes apart, by weight, roughly like this:

- **About 44 pounds of meal** — roughly 80% of the bushel by weight. This is the big product by mass.
- **About 11 pounds of oil** — roughly 18% of the bushel by weight. This is the smaller product by mass, but pound for pound it is worth far more.
- **About 5 pounds of hulls, moisture, and processing loss** — the remaining ~2-9% that is fibre and water, low or no value.

The exact yields vary a little with the bean and the plant — a common rule-of-thumb pair used across the industry is "44 pounds of meal and 11 pounds of oil per bushel," and that is the convention this post will use throughout. Hold those two numbers — **44 and 11** — because every piece of crush arithmetic is built on them.

The reason the split matters is that meal and oil are worth wildly different amounts per pound, and they answer to completely different markets. Meal is the bulk product — most of the weight — and it is sold as a commodity protein into the global animal-feed business. Oil is the minority product by weight but a much higher value per pound, and it is sold into food and, increasingly, into fuel. The crusher's whole economic life is the arithmetic of buying one bushel of beans and selling that 44-plus-11 split of products — usually for more than it paid.

#### Worked example: the conversion arithmetic, pound by pound

Take one 60-pound bushel of soybeans. After crushing you have, by weight: **44 pounds of meal**, **11 pounds of oil**, and **5 pounds of hulls and loss**. Notice the arithmetic: 44 + 11 + 5 = 60, the weight balances. Now turn the weights into the *quoted units* you will actually see on a screen. Meal is quoted in dollars per **short ton**, and a short ton is **2,000 pounds**, so 44 pounds of meal is 44 ÷ 2,000 = **0.022 short tons** of meal per bushel. Oil is quoted in **cents per pound**, and you have **11 pounds** of it per bushel, so the oil quantity per bushel is simply 11 pounds. Those two conversion factors — multiply the meal price by **0.022** and the oil price (in dollars per pound) by **11** — are the entire bridge between the screen and the spread. **Memorise 0.022 and 11 and you can compute a board crush in your head; everything else is bookkeeping.**

## The crush spread: the processor's gross margin, as one number

The full picture is a little messy — two products, each with its own price, each in its own unit, plus the bean cost in a third unit. Traders wanted a single, simple, tradable number that captures "how good is crushing right now," so they defined one. It is called the **board crush** (because it is computed from the prices on the futures *board* — the exchange screen), and it is the value of the products minus the cost of the beans, expressed per bushel of beans:

$$\text{Board crush} = \underbrace{(0.022 \times P_{\text{meal}})}_{\text{meal value per bushel}} + \underbrace{(0.11 \times P_{\text{oil}})}_{\text{oil value per bushel}} - \underbrace{P_{\text{beans}}}_{\text{bean cost per bushel}}$$

A note on the oil term. Oil is quoted in *cents* per pound, and you have 11 pounds per bushel, so the oil value per bushel is 11 pounds × (cents per pound ÷ 100) = 0.11 × (price in cents). Writing the oil price in cents and multiplying by 0.11 gives dollars per bushel directly. That single fiddly conversion — cents-per-pound times eleven-hundredths — trips up everyone the first time, so we will do it slowly in a worked example below.

The board crush is a *gross* margin: it is the spread between products and beans, before the crusher pays for energy, labour, hexane, maintenance, and freight. The cash cost of actually running the plant — turning beans into finished meal and oil — is on the order of \$0.30 to \$0.60 a bushel depending on the plant, so the *net* crush a processor keeps is the board crush minus that operating cost. The board crush is the headline number traders watch; the net crush is what the plant owner banks.

#### Worked example: computing a board crush from real-ish prices

Suppose the screen shows these three prices, which are roughly representative of a normal year: **soybeans at \$13.00 a bushel**, **soybean meal at \$400 a short ton**, and **soybean oil at 45 cents a pound**. Build the crush leg by leg.

- **Meal value per bushel:** \$400 per short ton × 0.022 short tons per bushel = **\$8.80 per bushel.**
- **Oil value per bushel:** 45 cents per pound × 11 pounds per bushel = 495 cents = **\$4.95 per bushel.**
- **Total product value per bushel:** \$8.80 + \$4.95 = **\$13.75 per bushel.**
- **Bean cost per bushel:** **\$13.00.**
- **Board crush:** \$13.75 − \$13.00 = **\$0.75 per bushel.**

So at these prices the processor earns a gross spread of about 75 cents on every bushel it crushes — before paying the ~40-cent cash cost of running the plant, which would leave a net crush near 35 cents. **The processor does not care much whether beans are \$10 or \$15 a bushel; it cares about whether the meal-plus-oil value runs ahead of the bean cost, and by how much — and that wedge is the entire business.**

That wedge has its own price history, and it is worth seeing that the crush margin is a genuinely *separate* line from the bean price itself.

![Soybean price and board crush margin on a dual axis 2020 to 2024](/imgs/blogs/the-crush-and-the-spread-soybeans-into-meal-and-oil-2.png)

Look carefully at the two lines. The grey line is the bean price; the blue line is the board crush margin. They are *not* the same shape. The bean price peaked in 2022 and then drifted down through 2023 and 2024. The crush margin also peaked in 2022 — but it was already climbing hard in 2021 before beans got expensive, and the *reason* it climbed was nothing to do with the bean. It was the oil. This is the single most important visual in the post: **the crush margin has a life of its own, driven by the relative value of meal and oil, not by the level of the bean.** A processor can have a banner year while the bean price is flat, and a terrible year while beans are expensive, depending entirely on what is happening to the two products.

### Board crush versus physical crush: the basis

There is a subtlety worth pinning down before we go further, because it separates the number traders quote from the margin a plant actually earns. The **board crush** we have been computing uses *futures* prices — the prices on the exchange screen for standardised bean, meal, and oil contracts deliverable at standard locations and dates. The **physical crush** (sometimes called the cash or plant crush) is the margin a specific plant earns from the *actual* beans it buys at its *actual* gate and the *actual* meal and oil it sells to its *actual* customers. These two are not the same, and the gap between them is called the **basis**.

The basis exists because the futures contract is an idealised, standardised thing, while a real plant lives at a real address. A crush plant in central Iowa pays a *local* cash price for beans that differs from the Chicago futures price by a freight-and-handling differential; it sells meal to feed mills at a *local* meal basis; and it sells oil at a *local* oil basis that depends on whether a renewable-diesel refinery happens to be nearby bidding for feedstock. The board crush tells you the *exchange-implied* margin; the physical crush tells you what a particular plant in a particular place actually banks. A plant sited next to a big new renewable-diesel refinery might earn a *better* oil basis — and therefore a fatter physical crush — than the board crush implies, while a plant far from any oil demand might earn worse. Reading the crush well means knowing both: the board crush as the market-wide signal, and your own basis as the local adjustment.

#### Worked example: the gross crush, the operating cost, and the net the plant keeps

Take the \$0.75 board crush from the worked example above and run it all the way down to what the plant owner actually pockets. The board crush is **\$0.75 a bushel** (gross product value minus bean cost). Against that, the plant pays its cash operating costs: energy to run the cracking, heating, and extraction; hexane losses; labour; maintenance; and an allocation for the freight on beans in and products out. Call that **\$0.45 a bushel** for a typical, reasonably efficient plant. The plant also earns a small positive oil basis — say its local renewable-diesel buyer pays **\$0.10 a bushel** above the board — so the physical crush is \$0.75 + \$0.10 = \$0.85 gross. Net of the \$0.45 operating cost, the plant keeps \$0.85 − \$0.45 = **\$0.40 a bushel**. On a plant crushing 150,000 bushels a day that is 150,000 × \$0.40 = **\$60,000 a day** of net margin, or roughly \$22 million a year — from a board crush that looked like a thin 75 cents. **The board crush is the headline; the basis and the operating cost are what turn that headline into the cash a plant actually banks, and a well-sited plant with a good oil basis can earn meaningfully more than the screen implies.**

### Why the products are worth more than the bean

It is worth pausing on *why* the meal and oil together are usually worth more than the bean they came from, because it is not automatic — there are years when the crush is negative and crushing loses money — and it explains where the margin physically comes from.

The first source of value is **transformation**. A whole raw bean is a locked box; the protein and oil inside it are not in a usable form. Animals cannot efficiently digest raw soybeans, and food and fuel applications need clean, refined oil, not the oily seed. Cracking, cooking, extracting, toasting, and refining are *work*, and the market pays for finished work. The crusher is selling not beans but *protein in a form a hog can digest* and *oil in a form a fryer or a diesel engine can use*. The margin is the market's payment for performing that transformation, net of the cost of performing it.

The second source is **separation of two markets that move independently**. Inside the bean, meal and oil are bolted together at a fixed ratio set by biology. Once crushed, they are free to be sold into two completely different markets — feed and food/fuel — that rarely peak and trough at the same time. When meal is in short supply (a livestock boom) the crusher captures that; when oil is in short supply (a biodiesel boom) the crusher captures that too. The crush plant is the only place those two values get *unbundled*, and unbundling has value: a crusher can always lean toward whichever product the market wants more this month.

The third source is **scale and logistics**. Crushing happens in giant plants sited near the beans (in the United States Corn Belt, in Brazil's Mato Grosso, on the Argentine river ports) and near the customers (feed mills, ports, biodiesel refineries). A processor that controls the storage, the rail, and the export terminal can capture extra value that a standalone crusher cannot. This is why the big four — ADM, Bunge, Cargill, Dreyfus — dominate: they are not just crushers, they are integrated logistics machines wrapped around a crush margin. (The wider world of these physical traders is the subject of [commodity trading houses](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura) — the grain majors play the same game with beans that Glencore and Vitol play with metals and oil.)

## Putting on the crush: how a processor locks its margin

Here is the strategic problem at the heart of the business. A crush plant is *naturally exposed* the wrong way around. It buys beans today (a cost, which it would like to be low) and sells meal and oil weeks later (revenues, which it would like to be high). Between the day it buys the beans and the day it sells the products, prices move — and they can move against it. If the bean price jumps after the plant has committed to buying beans, or if meal and oil prices collapse before the plant has sold the products, the margin gets crushed (pun intended). The processor is, in the language of the trade, **long beans and short meal-plus-oil**: it owns the input and owes the outputs.

A processor who wants to bank a known margin does the obvious thing: it *fixes all three prices at once* using futures. The exchange (the CME, which lists soybean, soybean meal, and soybean oil futures) lets it lock each leg.

![Putting on the crush hedge before and after with the reverse crush](/imgs/blogs/the-crush-and-the-spread-soybeans-into-meal-and-oil-3.png)

To **put on the crush** — to lock the margin — the processor does three trades simultaneously:

1. **Buy bean futures.** This fixes the input cost at today's bean price. If beans rise later, the futures gain offsets the higher cash cost of the beans the plant buys.
2. **Sell meal futures.** This fixes the meal revenue at today's meal price. If meal falls later, the short futures position gains to offset the lower cash sale.
3. **Sell oil futures.** Same logic for the oil leg.

Once all three are on, the processor has locked the *spread*. If the whole complex rallies — beans, meal, and oil all up together — the loss on the short product legs is offset by the gain on the long bean leg, and the margin stays put. If the whole complex falls, the same cancellation happens in reverse. **The flat-price level of the soybean complex can swing violently and the hedged crusher does not care, because it has bet on the spread, not the level.** This is the exact same trick a refiner uses to "buy the crack" — fix all three legs, bank the margin, and stop caring about the price of oil.

There is an elegant symmetry here that connects this post to the rest of the series. A crusher locking its margin is doing a **spread trade**: it is long one thing and short another, so its profit and loss depend on the *relationship* between prices, not their absolute level. Spread trades have far smaller risk than outright price bets, because most of the market's volatility — the part that moves all three legs together — cancels out. That property is the whole reason spread trading exists, and it is explored in depth in [calendar spreads and curve trades](/blog/trading/commodities/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level): the crush is a *product* spread (different commodities, same time) rather than a *calendar* spread (same commodity, different times), but the logic — bet the shape, not the level — is identical.

#### Worked example: a processor locks the crush on a day's plant run

A crush plant runs **150,000 bushels of beans a day**. On a Monday morning the board crush is **\$1.20 a bushel** — a healthy margin the plant manager wants to lock for the next month's production, about 30 days × 150,000 = **4.5 million bushels**. The manager puts on the crush: buys bean futures, sells meal futures, sells oil futures, sized so all 4.5 million bushels of next month's run is hedged at \$1.20. The gross margin locked is 4.5 million bushels × \$1.20 = **\$5.4 million** for the month. Now suppose that over the next three weeks the whole soybean complex rallies 20% — beans, meal, and oil all jump together on a South American drought scare. The cash beans the plant buys cost more, but the long bean futures gained to match; the cash meal and oil sell for more, but the short product futures lost to match. **Net of the hedge, the plant still banks its \$5.4 million — the rally that would have terrified an unhedged speculator is a non-event for the hedged processor, because it locked the spread, not the level.**

### The reverse crush

Now flip it. Suppose you are *not* a plant owner — you are a trader, and you think the crush margin is too *low*. Maybe meal and oil look cheap relative to beans, and you expect the margin to widen as processing demand recovers. You can bet on that directly without owning a plant at all, by putting on the **reverse crush**: you do the opposite of the processor. You **sell bean futures** and **buy meal and oil futures**. If the crush margin widens — products rise relative to beans — your long product legs gain more than your short bean leg loses, and you profit on the spread.

The reverse crush is how speculators express a view on processing economics, and it is also what a processor might use to *unwind* a hedge or to bet that a currently-thin margin will fatten. The key thing to see is that putting on the crush and putting on the reverse crush are mirror images: the processor *sells* the crush (locks a good margin so it cannot get worse), and the speculator who thinks margins will improve *buys* the crush (the reverse). The board crush is a tradable instrument in its own right, with its own supply and demand of hedgers and speculators — exactly the cast of characters laid out in [the four players](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators).

## The unit puzzle: three quotes, one spread

We have skated over the single most error-prone part of the whole subject, and it deserves a section of its own because it is where beginners — and not a few professionals — get the crush wrong. The three legs of the crush trade in **three different units**, and you cannot combine them until you have converted all three onto a common per-bushel basis.

![Matrix of beans meal and oil units converted to a common per bushel basis](/imgs/blogs/the-crush-and-the-spread-soybeans-into-meal-and-oil-5.png)

Here is the full reconciliation, the thing the grid above lays out:

- **Soybeans** are quoted in **dollars per bushel**, where a bushel weighs 60 pounds. The bean is the cost leg, and one bushel is already the natural unit, so no conversion is needed: the bean price *is* the per-bushel cost.
- **Soybean meal** is quoted in **dollars per short ton**, where a short ton is 2,000 pounds. You get 44 pounds of meal per bushel, so the conversion factor is 44 ÷ 2,000 = **0.022 short tons per bushel**. Multiply the meal price by 0.022 to get dollars of meal value per bushel.
- **Soybean oil** is quoted in **US cents per pound**. You get 11 pounds of oil per bushel, so multiply the oil price (in cents) by 11 and divide by 100 to get dollars of oil value per bushel — equivalently, multiply the cents price by **0.11**.

The reason the units are different is purely historical: meal trades in tons because it is shipped by the truckload and railcar-load to feed mills, where tonnage is the natural unit; oil trades in cents-per-pound because it is sold to food companies and refiners who buy by weight; and beans trade by the bushel because that is how grain has been measured at the elevator for two centuries. None of these conventions is going to change, so anyone trading the crush simply has to internalise the three conversions. Get one wrong — say, forget the divide-by-100 on the oil leg, or use a metric tonne instead of a short ton for the meal — and your crush will be off by a factor of ten or more, and you will think you have found free money when you have only found an arithmetic mistake.

#### Worked example: rebuilding the same crush from raw screen prices

Let us do the full conversion once more, slowly, from a fresh set of screen prices, to nail the units. The screen shows: **beans \$12.50/bushel**, **meal \$380/short ton**, **oil 50 cents/pound**.

- **Beans:** \$12.50 per bushel — the cost leg, no conversion. **Cost = \$12.50.**
- **Meal:** \$380 per short ton × 0.022 = \$8.36 per bushel. **Meal value = \$8.36.**
- **Oil:** 50 cents per pound × 11 pounds = 550 cents ÷ 100 = \$5.50 per bushel. **Oil value = \$5.50.**
- **Product total:** \$8.36 + \$5.50 = **\$13.86 per bushel.**
- **Board crush:** \$13.86 − \$12.50 = **\$1.36 per bushel.**

So at these prices the gross crush is \$1.36 a bushel — a fat margin, driven here by the high 50-cent oil. **The discipline is always the same: convert meal with 0.022, convert oil with 0.11, leave beans alone, then subtract — and never mix a metric tonne into a short-ton calculation.**

### The oil share: which product is carrying the crush

Once you can build the crush, the next thing a serious analyst looks at is *which leg is doing the work*. Of the total product value per bushel, how much comes from meal and how much from oil? This is called the **oil share** of the crush, and it is one of the most revealing numbers in the whole complex.

Go back to the first worked example: meal value \$8.80, oil value \$4.95, total \$13.75. The oil share is \$4.95 ÷ \$13.75 = **36%**. Historically the oil share has hovered around the low-to-mid 30s — meal, being 80% of the weight, has usually been the bigger contributor to value despite being the cheaper product per pound. But the oil share is not fixed. When soybean oil gets expensive — as it did when renewable diesel arrived — the oil share rises, sometimes past 45%, and the whole character of the crush changes: suddenly the *minority* product by weight is carrying nearly half the value. Watching the oil share tells you, at a glance, whether the crush is a meal story or an oil story this year, and therefore which demand driver to watch.

## Two products, two demand stories

This is the heart of why the soybean is such a fascinating commodity: it is really *two* commodities fused into one seed, and the two halves answer to entirely different parts of the world economy.

![Demand split meal to animal feed oil to food and biodiesel](/imgs/blogs/the-crush-and-the-spread-soybeans-into-meal-and-oil-7.png)

**Meal demand is an animal-protein story.** Soybean meal is the world's dominant high-protein animal feed. It goes into the rations of hogs, poultry, cattle, and farmed fish — anywhere an animal needs protein to grow muscle. The single biggest driver of meal demand is therefore *global meat consumption*, and especially the rise of meat-eating in developing economies. When China's hog herd expands, when poultry production in Southeast Asia grows, when aquaculture booms, meal demand grows with it. Meal demand is steady, structural, and tied to the slow grind of rising global living standards and protein consumption. The big shocks to meal demand are animal-disease events — African swine fever wiped out a huge fraction of China's hog herd in 2018-2019 and gutted meal demand for a time — and the long upward trend of more people eating more meat.

**Oil demand has two engines, and the second one changed everything.** The first engine is *food*: soybean oil is one of the world's most-used cooking oils, the workhorse of frying, baking, margarine, and processed food, especially in the Americas and Asia. Food oil demand is large but slow-growing, tracking population and diet. The second engine — the one that re-priced the entire complex — is *fuel*. Soybean oil is a prime feedstock for **biodiesel** and, more recently, **renewable diesel**.

It is worth being precise about the difference, because it matters. *Biodiesel* (technically FAME, fatty-acid methyl ester) is made by chemically reacting vegetable oil with methanol; it is blended into ordinary diesel at modest percentages. *Renewable diesel* (technically HVO, hydrotreated vegetable oil) is a newer, higher-quality process that hydrogenates the oil into a fuel chemically identical to petroleum diesel — it can be used at 100% with no blending limit, in any diesel engine, with no modification. Renewable diesel is the game-changer: a wave of new and converted refineries on the US West Coast and Gulf Coast, built to satisfy low-carbon-fuel mandates in California and federal blending rules, created enormous new structural demand for soybean oil starting around 2021. (For the broader story of how grains and softs trade as an asset class — including the food-versus-fuel tension — see the allocator's view in [agriculture and softs](/blog/trading/cross-asset/agriculture-softs-the-food-and-fiber-markets).)

The consequence for the crush was profound. For a century, soybean oil was the *junior* product — the meal was the prize, and the oil was a useful byproduct you sold to make margarine. The renewable-diesel boom flipped that. Oil became a strategically scarce feedstock for a fast-growing fuel industry, its price climbed, the oil share of the crush rose, and the whole crush margin widened. The 2021-2022 crush blowout was, at bottom, a *renewable-diesel* event that happened to express itself through the price of a frying oil.

There is a deeper structural reason the oil leg punches above its weight. Soybean oil is only about 18% of the bean, so the world's soybean oil supply is *thin* relative to the meal supply — you cannot make more oil without also making the corresponding (much larger) pile of meal. That means a surge in oil demand cannot be met simply by crushing more beans, because doing so floods the market with four times as much meal, which crashes the meal price and (eventually) the crush. The oil and meal are joined at the hip by the fixed 44-to-11 biology of the bean. So when a giant new oil buyer like renewable diesel arrives, the oil price has to do most of the adjusting — there is no easy way to make oil without making meal — and that rigidity is exactly why a fuel-demand shock shows up as a violent move in the oil leg and the crush, rather than a gentle one. The fixed split that makes the unit arithmetic fiddly is the same fixed split that makes the oil leg so sensitive to a new buyer.

#### Worked example: the oil-share shift when biodiesel demand jumps

Start from a "before biodiesel" world: meal \$400/ton, oil 35 cents/lb. Meal value = \$400 × 0.022 = \$8.80; oil value = 35 × 0.11 = \$3.85; total product value = \$12.65; oil share = \$3.85 ÷ \$12.65 = **30%**. Now a wave of renewable-diesel plants comes online and bids oil up to 65 cents/lb while meal is unchanged at \$400/ton. Meal value is still \$8.80; oil value jumps to 65 × 0.11 = \$7.15; total product value = \$15.95; oil share = \$7.15 ÷ \$15.95 = **45%**. The oil price has nearly doubled, the oil share has leapt from 30% to 45%, and — crucially — if the bean cost did not rise as much as the products did, the entire \$3.30 increase in product value per bushel flows straight into the crush margin. **When a new structural buyer shows up for one of the two products, it does not just raise that product's price — it widens the crush and changes which leg the whole business depends on.**

## How the crush margin has actually behaved

Theory is one thing; let us look at how the crush margin has moved in the real world, because the swings tell the whole demand story.

![Board crush margin bar chart 2019 to 2024 with 2022 blowout annotated](/imgs/blogs/the-crush-and-the-spread-soybeans-into-meal-and-oil-4.png)

The board crush spent the late 2010s in a sleepy range — roughly \$0.85 to \$1.10 a bushel — a comfortable but unremarkable margin for an industry running steadily. Then it took off. Through 2021 and into 2022 the margin climbed to a blowout near **\$2.10 a bushel**, more than double its long-run average, before easing back toward \$1.20 by 2024 as new crush capacity came online to chase the fat margin. That arc — sleepy, blowout, normalisation — is the classic signature of a *demand shock followed by a supply response*, and it is worth understanding each phase.

The **blowout (2021-2022)** was the renewable-diesel demand shock described above. New oil buyers bid up soybean oil faster than the bean or meal could keep up, the oil share rose, and the crush widened. Processors made extraordinary money. The market did what markets do: it signalled, through the fat margin, that the world needed more crushing capacity.

The **supply response (2023 onward)** followed. A fat crush margin is an open invitation to build crush plants, and the industry answered: a wave of new and expanded crush plants was announced and built across the US Midwest specifically to capture the renewable-diesel-driven oil demand. As that new capacity came online, it crushed more beans, produced more oil, and — by competing for the same beans and selling into the same oil market — narrowed the margin back toward normal. This is the iron law of processing margins: a wide margin builds the capacity that eventually closes it. The crush is mean-reverting on a multi-year cycle precisely because capital chases it.

This behaviour makes the crush a genuinely useful *signal*, not just a trade. Because the board crush is the *incentive to crush*, it is a real-time read on processing demand. A high crush margin tells you processors want to run flat out, which means strong demand for meal and oil downstream. A collapsing crush margin tells you the products are getting cheap relative to beans — perhaps a feed-demand slump or an oil glut — and that plants may idle. Watching the crush is watching the demand for the *transformation*, which often leads the demand for the underlying commodities.

### The bean-corn ratio: the acreage fight upstream

There is one more piece of the puzzle that sits *upstream* of the crush, and a complete picture of soybeans requires it: where do the beans come from in the first place? In the US Midwest and in Brazil, soybeans and corn compete for the same acres — a farmer in Iowa or Mato Grosso chooses each spring whether to plant beans or corn on a given field. That choice is driven heavily by the **soybean-to-corn price ratio**: the price of beans per bushel divided by the price of corn per bushel.

![Soybean and corn annual prices with the bean-corn ratio annotated](/imgs/blogs/the-crush-and-the-spread-soybeans-into-meal-and-oil-6.png)

The rough rule of thumb is that a soybean-to-corn ratio **above about 2.5** makes beans the more attractive crop and pulls acres toward soybeans, while a ratio **below about 2.5** favours corn. The exact threshold varies with input costs and rotation agronomy, but the principle is robust: when beans get expensive relative to corn, farmers plant more beans next season, the bean supply expands, and — all else equal — the bean cost leg of the crush eventually softens. The acreage fight upstream feeds the bean supply, which feeds the cost side of the crush. (The corn and grain complex as a whole, including the planting-and-harvest calendar that sets the supply rhythm, is the subject of the companion post on [grains: corn, wheat and soybeans](/blog/trading/commodities/grains-corn-wheat-and-soybeans-the-calories-that-trade).)

For the crush trader, the bean-corn ratio is a slow-moving structural signal about future bean supply, while the crush margin itself is the fast-moving signal about product demand. The two together — supply pressure from the acreage fight, demand pull from feed and fuel — bracket the whole soybean story.

### Where crushing happens: the three great crush regions

The crush is not a single global market; it is three big regional ones that compete and arbitrage against each other, and knowing the geography sharpens how you read the spread.

The **United States** is the original and most financialised crush market — its beans, meal, and oil futures are the ones that define the *board* crush, and its Midwest is dotted with crush plants near both the beans and the feed mills. It is also the home of the renewable-diesel demand that re-priced oil, so the US crush is the one most exposed to fuel policy. **Brazil** is now the world's largest soybean producer, with a vast crush industry in the interior states like Mato Grosso; Brazil's harvest arrives in the Northern-Hemisphere spring, six months out of phase with the US, so the global bean supply has two harvests a year and the crush economics shift as each one lands. **Argentina** is the world's largest *exporter* of soybean meal and oil specifically — it crushes a huge share of its beans rather than exporting them whole, because its export-tax structure historically favoured shipping the higher-value products over raw beans. The cluster of giant crush plants on the Paraná River near Rosario is one of the densest concentrations of crushing capacity on earth.

And then there is **China**, the demand colossus: the world's largest *importer* of soybeans and the world's largest crusher by volume, importing raw beans by the boatload to crush domestically for its enormous hog and poultry herds. China crushes for the *meal* — its driving need is feed protein — and the oil is, for China, more of a byproduct, the mirror image of the renewable-diesel-driven US story where the oil increasingly leads. When Chinese crush demand is strong, it pulls beans out of Brazil and the US and tightens the global bean balance; when an animal-disease shock hits China's herd, the world's single biggest meal buyer pulls back and the whole complex feels it. The crush, in other words, is a global tug-of-war: South America grows and crushes the beans, China imports and crushes for meal, and the US crushes for an oil leg increasingly claimed by fuel — three regions, three different reasons to crush, all reading off versions of the same spread.

## Common misconceptions

**"A crusher wants high soybean prices."** No — a crusher is largely *indifferent* to the bean price level and cares almost entirely about the *spread* between products and beans. A crusher actively dislikes a bean price that rises *faster* than meal and oil, because that compresses the crush. As the dual-axis chart showed, the crush margin and the bean price are different lines: the bean rallied and faded in 2022-2024, but the margin had already blown out in 2021 for oil-side reasons. The crusher's prayer is "let my products run ahead of my beans," not "let beans go up."

**"Meal and oil move together because they come from the same bean."** They share a common origin but answer to completely different demand drivers, and they frequently diverge. Meal tracks animal-protein demand (hogs, poultry, aquaculture); oil tracks food *and* fuel demand. The renewable-diesel boom sent oil sharply higher while meal was comparatively flat — a divergence that lifted the oil share of the crush from the low 30s toward the mid 40s. Treating meal and oil as one market is the fastest way to misread the crush.

**"You can just add the meal and oil prices to the bean price to get the crush."** This is the unit trap, and it is off by an order of magnitude. Meal is in dollars per *short ton* and oil is in *cents per pound* — you must convert meal with the 0.022 factor and oil with the 0.11 factor before any of the three numbers can be combined. Forget the conversions and a real \$1.30 crush will look like a \$385 "crush" (you added \$400 + \$45 − \$13 without converting), and you will think you have discovered a money machine. The units are the whole game.

**"The crush margin is stable because it's a spread."** Spreads are *less* volatile than flat prices, but the crush margin is far from constant: it ranged from sleepy lows near \$0.85 to a blowout near \$2.10 within a few years, a more-than-doubling. What makes the crush *tradable with less risk* than an outright bean bet is that it nets out the part of volatility common to all three legs — but the part that is *specific* to the meal-versus-oil-versus-bean relationship can still swing hard, especially when a demand shock hits one product.

**"Renewable diesel is a niche that doesn't matter for soybeans."** It re-priced the entire complex. A product treated as a low-value byproduct for a century became a strategically scarce fuel feedstock, its price climbed, the oil share of the crush rose toward half the value, and the board crush blew out to record levels. Any analysis of soybeans that ignores the renewable-fuels demand for oil is analysing the 2010 market, not today's.

## How it shows up in real markets

**The 2021-2022 crush blowout.** The cleanest real-world case is the one this post opened with. Through 2021, US renewable-diesel capacity ramped sharply, and projected capacity additions implied an enormous future call on vegetable oil. Soybean oil — a thin market relative to the new fuel demand — was bid up hard, the oil share of the crush rose, and the board crush margin climbed from its sleepy ~\$0.85-\$1.10 range toward roughly \$2.10 a bushel in 2022. Crucially, this happened even though beans themselves were only moderately expensive (averaging around \$13-\$15 a bushel). Processors like ADM and Bunge reported some of their best results ever — not because beans were expensive, but because the *spread* between their products and their beans had widened. The market read the fat margin as a build signal, and a wave of new Midwest crush plants was announced through 2022-2023 specifically to capture renewable-diesel oil demand.

**The 2018-2019 African swine fever shock.** The mirror image hit the *meal* leg. African swine fever devastated China's hog herd — the world's largest, and the single biggest consumer of soybean meal — wiping out a large fraction of the herd over 2018-2019. With far fewer hogs to feed, meal demand sagged, the meal leg of the crush weakened, and (compounded by the US-China trade war that cut Chinese purchases of US beans) the whole complex came under pressure. This is the meal-side analogue of the oil-side boom: a demand shock to *one* product reshaping the crush, in this case downward. It is a reminder that the crush is two demand stories, and either one can move it.

**The policy lever on the oil leg.** Because so much oil demand now comes from fuel, the crush has become sensitive to *policy*. The US Renewable Fuel Standard's blending requirements, the size of the biomass-based-diesel mandate, the value of low-carbon-fuel credits in California, and the rules on which feedstocks qualify for tax credits all move the demand for soybean oil and therefore the crush. A single regulatory announcement — say, a higher-than-expected renewable-volume obligation — can lift soybean oil and widen the crush in a day, the way a refinery's crack spread jumps on a fuel-supply shock. The crush is no longer a purely agricultural number; it is partly an energy-policy number now. (How a single policy print can ripple across every connected market is the general phenomenon mapped in [cross-asset transmission](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market).)

**Reading the crush as a processing-demand gauge.** Putting it together: when you see the board crush widening, ask *which leg* is driving it. If the oil share is rising, it is a fuel/oil-demand story — watch renewable-diesel capacity and policy. If the meal value is leading, it is an animal-protein story — watch the global hog and poultry herd. If the crush is widening because the *bean* is cheap rather than because products are dear, it is an oversupply story — watch the South American harvest and the bean-corn acreage ratio. The crush is not one signal; it is a structured way to ask *what is the world short of right now — feed protein, food and fuel oil, or beans themselves?*

## The takeaway: how to read and trade the crush

Step back and the soybean crush resolves into a handful of durable ideas that will serve you long after the specific numbers in this post are out of date.

**A crusher is a processor, not a speculator.** Its business is the *spread* between products and beans, not the level of any of them. The single mental shift that unlocks the whole subject is to stop thinking "is soybeans going up?" and start thinking "is the meal-plus-oil value running ahead of the bean cost, and which leg is driving it?" Everything else follows from that.

**The board crush is built from three numbers in three units — convert before you combine.** Beans in dollars per bushel (leave alone), meal in dollars per short ton (× 0.022), oil in cents per pound (× 0.11). Memorise 44 pounds of meal and 11 pounds of oil per 60-pound bushel, and the two conversion factors they imply, and you can compute a board crush anywhere, on the back of a napkin.

**To lock a margin, put on the crush; to bet it widens, put on the reverse.** A processor that likes today's margin buys beans and sells meal and oil futures, banking the spread regardless of where flat prices go. A trader who thinks margins are too thin does the opposite. Either way, the bet is on the *relationship*, which is why crush trading carries far less risk than an outright bean position — the common volatility cancels, and only the spread-specific moves remain.

**Watch the oil share to know which story you are in.** When the oil share of the crush is in the low 30s, it is a meal market — watch animal protein. When it climbs toward the mid 40s, it is an oil market — watch food and, above all, renewable diesel. The single biggest structural change in the soybean complex this decade was the renewable-diesel boom re-pricing the oil leg, and the oil share is the number that tells you, at a glance, whether that story is accelerating or fading.

**Read the crush as a processing-demand signal, not just a trade.** Because the board crush *is* the incentive to crush, a widening margin tells you processors want to run flat out — strong downstream demand for meal and oil — and a collapsing margin tells you products are cheap relative to beans and plants may idle. And remember the iron law: a fat crush margin builds the capacity that eventually closes it, so the crush mean-reverts on a multi-year cycle as capital chases it.

The deepest point is the one that ties the soybean back to the spine of this whole series. A commodity price is a physical thing — here, a small oily seed — forced through a financial contract. The crush spread is what you get when you force *two* such contracts (meal and oil) against a *third* (beans) and read the gap. That gap is not noise; it is the price the market pays to perform a transformation, and it carries real information about what the world is short of — feed protein, food and fuel oil, or beans. When you can build a board crush from three screen prices in three units, lock it with three futures trades, and read which leg is carrying it, you are no longer looking at the price of soybeans. You are reading the economics of turning one cheap seed into the protein that feeds the world's livestock and the oil that increasingly fuels its trucks — and that is a far more interesting thing to know.

## Further reading & cross-links

- [Refining and crack spreads: turning crude into products](/blog/trading/commodities/refining-and-crack-spreads-turning-crude-into-products) — the oil-refinery twin of the crush; the crack spread is the crush's exact analogue, same logic, different commodity.
- [Grains: corn, wheat and soybeans, the calories that trade](/blog/trading/commodities/grains-corn-wheat-and-soybeans-the-calories-that-trade) — where the beans come from; the planting/harvest calendar and the bean-corn acreage fight that sets the cost leg of the crush.
- [Calendar spreads and curve trades: trading the shape, not the level](/blog/trading/commodities/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level) — the general theory of spread trading; the crush is a product spread, but the bet-the-relationship logic is identical.
- [The four players: producers, consumers, hedgers and speculators](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators) — who is on each side of the crush, and why the processor hedges while the speculator bets.
- [Agriculture and softs: the food and fiber markets](/blog/trading/cross-asset/agriculture-softs-the-food-and-fiber-markets) — the allocator's view of grains and softs as an asset class, including the food-versus-fuel tension that re-priced soybean oil.
- [Hedging a portfolio with options: protective puts, collars and tail risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) — how a processor can hedge its crush margin with options rather than futures, capping downside while keeping upside.
- [Commodity trading houses: Glencore, Vitol, Trafigura](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura) — the integrated logistics machines wrapped around processing margins like the crush.
- [Cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market) — why a single renewable-fuels policy print can move soybean oil, the crush, and the whole complex in a day.
