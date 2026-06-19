---
title: "Refining and Crack Spreads: Turning Crude Into Products"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a refinery is a margin business, what the 3-2-1 crack spread measures, how a refiner is long products and short crude, and how to read the crack as a real-time refining-profit and demand signal."
tags: ["commodities", "crack-spread", "refining", "crude-oil", "gasoline", "diesel", "3-2-1-crack", "refining-margin", "nelson-complexity", "energy"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A refinery does not make money on the price of oil; it makes money on the *gap* between what it pays for a barrel of crude and what it sells the gasoline and diesel inside that barrel for. That gap is the **crack spread**, and it is the single most useful number in the downstream energy world.
>
> - A refiner is structurally **long products and short crude**: it buys the dirty input, sells the clean outputs, and earns the difference. It hedges by "buying the crack" — selling product futures and buying crude futures so it locks the *spread*, not the level.
> - The industry-standard proxy is the **3-2-1 crack**: 3 barrels of crude in, 2 barrels of gasoline + 1 barrel of distillate out. It is a per-barrel gross margin you can read off a screen.
> - A **complex** refinery (high Nelson complexity) can crack cheap, heavy, high-sulfur crude into the same light products a simple plant can only make from expensive light-sweet crude — so it pockets the heavy-crude discount.
> - The one fact to remember: refining margins can boom while crude is flat. In **2022 the 3-2-1 crack hit roughly \$38 a barrel** — more than double its normal \$18 — even as crude itself rose only modestly. Refiners had their best year in history.

In the summer of 2022, something strange happened to the most boring corner of the oil business. Crude oil itself was high but not historic — West Texas Intermediate (WTI) averaged about \$95 a barrel for the year, below its 2008 peak. Yet the companies that turn that crude into gasoline and diesel — refiners like Valero, Marathon, and Phillips 66 — printed the most profitable quarters in their corporate histories. Valero's refining segment earned more in a single quarter than it had in some entire years. Their stock prices doubled. Pump prices in the United States blew past \$5 a gallon and politicians demanded to know who was getting rich.

The answer was hiding in a number almost no consumer has ever heard of: the **crack spread**. While the crude price was merely elevated, the *margin* between crude and the products refined out of it had exploded. Diesel was catastrophically short across the Atlantic after Russia's invasion of Ukraine scrambled the supply of refined products, refining capacity had been permanently shut during the pandemic, and the few barrels of refining capacity still running could name their price. The refiners were not betting on the price of oil. They were collecting a toll on a bottleneck — and the toll had gone vertical.

This is the post that explains that toll from zero. By the end you will understand why a refinery is fundamentally a *spread* business and not a *price* business, how the 3-2-1 crack is built and traded, why the same barrel of cheap, nasty crude is worthless to one refinery and a goldmine to another, and how to read the crack spread as a live signal of refining profitability and fuel demand.

![One barrel of crude splitting into gasoline diesel jet and fuel oil with the gross margin at the end](/imgs/blogs/refining-and-crack-spreads-turning-crude-into-products-1.png)

## Foundations: what a refinery actually does

Start with the physical thing. Crude oil straight out of the ground is nearly useless. You cannot put it in your car, you cannot fly a plane on it, you cannot heat a house with it. It is a thick, dark, foul-smelling soup of thousands of different **hydrocarbon molecules** — chains of carbon and hydrogen atoms of wildly different sizes, all mixed together. A refinery is a giant factory whose only job is to *sort and rebuild* those molecules into things people will actually pay for.

The sorting happens in two broad stages, and you need both to follow everything else.

**Stage one is distillation.** Crude is heated in a tall steel column — the **atmospheric distillation tower**, often 50 metres tall — until much of it boils into vapour. Different molecules boil at different temperatures: the small, light ones (which make gasoline) boil first and rise to the top of the tower; the medium ones (jet fuel and diesel) condense in the middle; the big, heavy ones (fuel oil, asphalt) stay at the bottom. The tower has trays at different heights, and each tray catches the molecules that condense at that temperature. So with no chemistry at all, just heat and gravity, the tower separates one barrel of crude into a stack of **fractions** sorted by molecular size. This is why the products are called *distillates* — they are what distil out of the crude.

**Stage two is conversion, also called cracking.** Here is the commercial problem distillation alone leaves you with: the molecules nature put in the barrel are not the molecules the market wants. A typical crude has far more heavy, low-value molecules than the market demands, and not enough of the light, high-value gasoline. So refiners *break* the big heavy molecules into smaller light ones. The workhorse unit that does this is the **fluid catalytic cracker (FCC)**, which uses heat and a catalyst to crack heavy gas-oil into gasoline. A **hydrocracker** does something similar under hydrogen pressure to make cleaner diesel and jet. A **coker** takes the very bottom of the barrel — the tar-like residue — and cracks even that into lighter products plus solid petroleum coke. The word "cracking" is literal: you are cracking long carbon chains into short ones. And it is from this verb that the *crack spread* gets its name — it is the margin you earn by cracking crude into products.

So a refinery is a two-step machine: distil the barrel into fractions, then crack the heavy fractions into more of the valuable light ones. The output is the **product slate**: the menu of finished products one barrel becomes.

It helps to hold a simple commercial analogy before any of the numbers. A refinery is the oil world's version of a sawmill, or a flour mill, or a meatpacker. The sawmill buys whole logs (cheap, raw, low-value) and sells planks, beams, and sawdust (more valuable, in graded sizes). The miller buys wheat and sells flour and bran. The packer buys a live animal and sells cuts of meat. In every case the business model is identical: **buy one undifferentiated raw input, split it into a graded slate of outputs, and earn the difference between the value of the outputs and the cost of the input — minus the cost of doing the splitting.** That difference-between-outputs-and-input is, in every one of these industries, called a *processing margin* or a *crush margin* (the soybean processor's version literally is the "crush"). The crack spread is simply the oil refiner's name for the same thing. Once you see that a refiner is a processor and not a speculator, the entire logic of the business falls into place.

The refinery is also genuinely enormous and capital-heavy, which matters for how the margin behaves. A large modern refinery costs many billions of dollars to build, takes years, and once built it is extremely hard to switch off or relocate. It runs flat out, 24 hours a day, because the cost of a "cold start" — shutting down and restarting all those interlinked units — is punishing. This is why refining is a classic **fixed-cost, high-throughput** business: the machinery is sunk, so the only thing that varies day to day is the *margin per barrel* times the *number of barrels run*. A refiner's mantra is "keep the units full and chase the best margin barrels," and every concept in this post is downstream of that single fact.

### The product slate: where the barrel goes

For a typical United States refinery, one barrel of crude (42 gallons) comes apart roughly like this. These are illustrative, rounded shares — every refinery's slate differs, and a plant can shift its mix somewhat — but the pattern is the point.

![Horizontal bar chart of product yield gasoline 45 percent distillate 26 percent jet 9 percent other 12 percent residual 4 percent](/imgs/blogs/refining-and-crack-spreads-turning-crude-into-products-6.png)

- **Gasoline — about 45%.** The single biggest and highest-value cut. The whole United States refining system is tilted toward gasoline because American drivers burn so much of it.
- **Diesel and heating oil (distillate) — about 26%.** The fuel of trucks, trains, ships, farm equipment, and home heating. Globally this is becoming the most strategically important product because it powers freight.
- **Jet fuel / kerosene — about 9%.** Chemically close to diesel; aviation demand swings it.
- **Other — about 12%.** Liquefied petroleum gas (LPG), petrochemical feedstocks like naphtha, lubricants, waxes.
- **Residual fuel oil — about 4%.** The heavy bottom of the barrel: bunker fuel for ships, asphalt, petroleum coke. The lowest-value output, and the thing a complex refinery tries hardest to minimise.

(You may notice these can sum to slightly more than 100% of the input volume. That is real: products are less dense than crude, so a barrel of crude yields slightly *more* than a barrel of liquid product. Refiners call it **processing gain**, and it is a small free lunch worth a percent or two.)

The reason this slate matters is that gasoline and diesel are worth far more per barrel than the crude that went in, and the heavy residue is worth far less. A refiner's whole economic life is the arithmetic of buying one barrel of the cheap input and selling that fan of products — most of it worth more than what it paid.

#### Worked example: the value of the slate vs the cost of the crude

Take one barrel of crude bought at **\$80**. Suppose the products refined out of it sell, on a per-barrel-of-crude basis, for these wholesale values: gasoline portion worth \$45, the distillate portion worth \$30, jet worth \$9, other worth \$10, residual worth \$3 — call it **\$97** of products in total (these are illustrative wholesale equivalents, scaled so the slate adds up). The **gross product worth** of the barrel is \$97; the crude cost \$80; the gross refining margin is \$97 − \$80 = **\$17 a barrel** before the refinery pays for energy, labour, and maintenance. A refinery that processes 200,000 barrels a day at a \$17 gross margin is generating 200,000 × \$17 = **\$3.4 million a day** in gross margin. **A refiner does not care much whether crude is \$40 or \$120 — it cares about the wedge between the products and the crude, and that wedge is the entire business.**

That wedge has a name and a standardised way of being measured, and that is where the crack spread comes in.

### Why the products are worth more than the crude

It is worth pausing on *why* the slate is worth more than the barrel, because it is not obvious and it explains where the margin physically comes from. Crude oil is a raw, unusable mixture; gasoline and diesel are precision products that must meet exacting specifications — vapour pressure, octane rating, sulfur content, cetane number, freezing point for jet fuel. Meeting those specifications is *work*, and the market pays for finished work. The refiner is selling not molecules but *molecules in the exact form and purity an engine needs*. The margin is the market's payment for performing that transformation, net of the cost of performing it.

There is a second, subtler source of value: **optionality**. The same crude can be turned into a slightly different mix of products depending on how the refiner tunes its units, so the refiner can lean toward whatever product is most valuable that week. A plant that can shift a few percentage points of yield from diesel to gasoline (or back) is worth more than a rigid one, because it can always chase the wider crack. This flexibility is itself an asset, and it is one reason complex refineries trade at higher multiples than simple ones — they hold more options on the product mix.

Finally, value comes from **scale and integration**. A refinery sitting next to a petrochemical complex can sell its naphtha and LPG straight into plastics and chemicals production at a premium, rather than blending it down into low-value fuel. Many of the most profitable downstream assets on earth are integrated refining-plus-petrochemical sites for exactly this reason: they squeeze extra value out of the parts of the barrel a standalone fuel refinery would waste.

## The crack spread: the refiner's gross margin, as one number

The full slate is messy — five products, each with its own price, weighted by yield. Traders wanted a single, simple, tradable number that captures "how good is refining right now," so they invented a **proxy**: a fixed recipe of crude in and products out whose margin tracks real refining economics closely enough to trade.

The industry standard is the **3-2-1 crack spread**. Read the name as a recipe:

- **3** barrels of crude in,
- **2** barrels of **gasoline** out,
- **1** barrel of **distillate** (diesel/heating oil) out.

The 3-2-1 ratio is chosen because it roughly matches the gasoline-heavy slate of a typical United States refinery: about two-thirds of the valuable output is gasoline, about one-third is distillate. (Europe, which runs more diesel cars, sometimes uses a more diesel-weighted crack; some traders use a simpler 1-1 "gasoline crack" or "heating-oil crack" for a single product.)

The 3-2-1 crack is then the *value of the products minus the cost of the crude, expressed per barrel of crude*:

$$\text{3-2-1 crack} = \frac{(2 \times P_{\text{gasoline}}) + (1 \times P_{\text{distillate}}) - (3 \times P_{\text{crude}})}{3}$$

There is a unit trap that catches everyone the first time, and it is worth pinning down before any math. Crude trades in **dollars per barrel**. Gasoline and heating-oil futures (on NYMEX) trade in **dollars per gallon**. There are 42 gallons in a barrel. So to put products on the same per-barrel footing as crude, you multiply the per-gallon product price by 42. Get this conversion wrong and your crack is off by a factor of 42 — a classic rookie error.

#### Worked example: computing a 3-2-1 crack from screen prices

Suppose your screen shows: WTI crude at **\$80.00** a barrel; RBOB gasoline at **\$2.50** a gallon; ULSD (ultra-low-sulfur diesel) heating oil at **\$2.80** a gallon.

First convert the products to dollars per barrel: gasoline = \$2.50 × 42 = **\$105.00/bbl**; distillate = \$2.80 × 42 = **\$117.60/bbl**.

Now apply the 3-2-1 recipe. Product value for 3 barrels of crude = (2 × \$105.00) + (1 × \$117.60) = \$210.00 + \$117.60 = **\$327.60**. Crude cost for 3 barrels = 3 × \$80.00 = **\$240.00**. The spread for the 3-barrel basket is \$327.60 − \$240.00 = \$87.60, and per barrel of crude that is \$87.60 ÷ 3 = **\$29.20 a barrel**.

So this refiner's gross margin is **\$29.20 per barrel of crude run** — a healthy, above-average crack. **The crack spread compresses the whole product slate into one number you can read, chart, and trade, and it is the cleanest available gauge of refining profitability.**

### The family of cracks: 3-2-1, 5-3-2, and single-product cracks

The 3-2-1 is the most-quoted crack, but it is one of a family, and which one matters depends on what you are measuring. The recipe always encodes the assumed product slate:

- **3-2-1** — 3 crude, 2 gasoline, 1 distillate. The United States benchmark, matching a gasoline-heavy slate.
- **5-3-2** — 5 crude, 3 gasoline, 2 distillate. A slightly more distillate-weighted recipe some analysts prefer for a national average.
- **2-1-1** — 2 crude, 1 gasoline, 1 distillate. A balanced crack used in regions (like much of Europe and Asia) that consume relatively more diesel.
- **Single-product cracks** — the **gasoline crack** (gasoline price minus crude) and the **heating-oil (distillate) crack** (diesel/heating-oil price minus crude), each on its own. These isolate one product so you can see *which* fuel is driving the margin. In 2022 it was overwhelmingly the diesel crack that blew out; the gasoline crack was strong but the distillate crack was extreme.

None of these is "correct" — they are all proxies, and the right one is whichever best matches the slate you care about. A trader watching United States Gulf Coast economics uses the 3-2-1; an analyst studying European diesel tightness watches the distillate crack alone. The key is to know which recipe a quoted "crack" refers to, because the numbers differ.

#### Worked example: gross margin and daily cash on a 200,000 b/d refinery

Take a refinery running **200,000 barrels a day** when the 3-2-1 crack is **\$22.00 a barrel**. Its **gross refining margin** is the crack times the run rate: \$22.00 × 200,000 = **\$4.4 million a day**, or about \$1.6 billion a year at a constant margin.

But gross is not net. The refinery spends real money to do the cracking. Suppose its **operating costs** — natural gas to fire the units, electricity, hydrogen, catalysts, labour, and routine maintenance — run **\$6.00 a barrel**. The **cash refining margin** (the number the refiner actually banks before depreciation and taxes) is \$22.00 − \$6.00 = **\$16.00 a barrel**, or \$16.00 × 200,000 = **\$3.2 million a day**. Notice how sensitive the bottom line is: if natural gas spikes and operating costs jump to \$9.00 a barrel, the cash margin falls to \$13.00 — a 19% hit to profit from a cost the headline crack does not even show. **The crack is a gross margin; a refiner's real profit is the crack minus its energy and operating costs, which is why a refiner watches its own natural-gas bill almost as closely as it watches the crack.**

### Why a refiner is long products and short crude

Here is the structural insight that makes a refiner different from almost any other commodity participant. A producer (an oil major) is simply **long crude** — they win when oil rises. A consumer (an airline) is simply **short fuel** — they lose when fuel rises. A refiner sits in the middle and is, at the same time, **short crude and long products**: it must *buy* crude (a cost, so effectively short the input) and *sell* products (a revenue, so effectively long the output). Its profit is structurally the *difference* between the two, which is exactly the crack spread.

This is why the crack spread is not just an analyst's metric — it is the refiner's actual position. If you owned a refinery and did nothing, you would already be implicitly **long the crack**: you make more when products rise relative to crude, and less when crude rises relative to products. The crack spread *is* your business, whether you trade it on a screen or not.

![The 3-2-1 crack spread annual average 2018 to 2024 marking the 2022 blowout](/imgs/blogs/refining-and-crack-spreads-turning-crude-into-products-2.png)

Look at how that margin has behaved. In a normal year the 3-2-1 crack sits around \$18 a barrel. In 2020, when the pandemic destroyed fuel demand, it collapsed to roughly \$11 — refineries were running at a loss and some shut for good. Then in 2022 it exploded to about \$38, more than double normal, as the world ran short of refined products. By 2024 it had cooled back toward \$21 as new refining capacity in Asia and the Middle East came online and demand normalised. That single line is a map of the downstream industry's fortunes.

## Hedging: how a refiner "buys the crack" to lock its margin

Being naturally long the crack is great when margins are fat, but it is risky. Between the day a refiner buys crude and the day it sells the finished products, weeks pass — crude must be shipped, stored, run through the units, and the products marketed. In that gap, prices move. If crude jumps or product prices fall, the margin the refiner counted on can evaporate. So refiners *hedge* by locking the spread in the futures market. The trade is called **buying the crack** (when you put the position on) or, from the refiner's natural side, **selling the crack** to lock in a known margin.

The mechanics mirror the physical flow exactly:

- The refiner will *buy* physical crude in the future, so to lock its input cost it **buys crude futures** (goes long crude).
- The refiner will *sell* physical products in the future, so to lock its output revenue it **sells gasoline and distillate futures** (goes short products).

Put on in 3-2-1 proportion (long 3 crude, short 2 gasoline + 1 distillate), this position has a P&L that depends almost entirely on the *spread* between products and crude — not on the *level* of oil. If oil rises \$10 across the board, the loss on the short product legs is roughly offset by the gain on the long crude leg, and the locked margin survives. The refiner has converted a floating margin into a fixed one.

![Before and after diagram of an unhedged refiner versus a refiner selling the crack to lock its margin](/imgs/blogs/refining-and-crack-spreads-turning-crude-into-products-3.png)

#### Worked example: the hedge P&L on a 200,000 b/d refiner

A refiner runs **200,000 barrels a day** and wants to lock a month of margin (call it ~6 million barrels). The 3-2-1 crack is trading at **\$25 a barrel** and management is happy to bank that. It sells the crack: in 3-2-1 proportion that is long crude, short gasoline, short distillate, sized to ~6 million barrels of crude over the month.

Now suppose by delivery the world has changed: crude has *risen* \$12 (bad for the refiner's input cost) but product prices rose less, so the *physical* crack it actually captures on the street falls from \$25 to **\$18** — a \$7-a-barrel margin squeeze worth 6,000,000 × \$7 = **\$42 million** of lost physical margin.

But the hedge moves the other way. The short product / long crude futures position was set to profit when the crack *narrows* — and the crack narrowed by \$7, so the futures position gains roughly \$7 × 6,000,000 = **\$42 million**. The hedge gain offsets the physical loss, and the refiner nets close to the **\$25** it locked in. **By selling the crack, the refiner trades away the upside of a margin boom in exchange for certainty — it banks a known \$25 instead of gambling on what the spread will be when its products hit the market.**

The flip side, of course, is the regret risk: had the crack *widened* to \$40, the refiner's hedge would have lost the difference and it would have "left money on the table." This is the eternal hedger's trade-off — certainty bought at the cost of upside — and it is the same logic an airline faces when it hedges jet fuel. For the consumer's mirror image of this decision, see how options can cap risk without giving up all the upside in [hedging a portfolio with options](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk).

### Paper crack vs physical crack: the basis the refiner cannot hedge away

The hedge above is clean on paper, but a real refiner faces a wrinkle that runs straight through the spine of this whole series — the gap between the *physical* thing and the *paper* contract written on it. The futures the refiner trades are standardised: NYMEX WTI crude delivered at Cushing, Oklahoma; RBOB gasoline and ULSD diesel delivered in New York Harbor. But the refiner's *actual* barrels are a specific crude (maybe a heavy Canadian grade) bought at a specific pipeline point, refined into products sold into a specific local market (maybe the United States Gulf Coast or the Midwest). The price of the refiner's real crude and real products differs from the futures by a **basis** — the local, grade-specific premium or discount to the benchmark.

So when a refiner "sells the crack" with futures, it locks the *benchmark* crack but is still exposed to two basis risks: the difference between its feedstock and benchmark WTI, and the difference between its product realisations and benchmark New York products. These can move independently. A refiner can lock a fat *paper* crack and still see its *physical* crack disappoint if its local product market weakens relative to New York, or if its crude grade richens relative to WTI. This is the universal commodity truth restated for refining: **the paper contract hedges the level of the benchmark spread, but the physical business always carries a residual basis that no standardised future can perfectly offset.** It is why refiners with the most favourable locations — close to cheap crude and strong product demand — earn structurally better margins than the benchmark crack alone implies, and why the crack on your screen is a guide to refining economics, not an exact measure of any one plant's profit.

### Who trades the crack, and why the spread is "safer" than flat price

The crack is not traded only by refiners. The futures market for the spread brings together the same cast that populates every commodity market — natural hedgers on one side, speculators on the other. Refiners are the natural *sellers* of the crack (locking their margin); some integrated oil majors and product marketers take the other side; and speculators (macro funds, commodity trading advisors, the trading houses) trade the spread as a directional view on refining tightness or as a relative-value position. Specialist exchanges even list the 3-2-1 crack as a single tradable instrument so you do not have to leg into three separate contracts. For the full anatomy of who stands on each side of any commodity contract and why, see [the four players: producers, consumers, hedgers, and speculators](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators).

One reason the crack is such a popular thing to trade is that a *spread* is inherently less volatile than an outright price. Because the crack is products *minus* crude, a market-wide move in the price of oil largely cancels out — both legs rise or fall together — leaving only the change in the *relationship* between them. That makes the crack's day-to-day swings far smaller than flat-price WTI's, which means a trader can hold a crack position with much less margin and much less stomach-churning mark-to-market noise than an outright oil bet. It is the same reason a refiner hedges with the crack rather than trying to hedge crude and products separately: the spread isolates the one risk the refiner actually has — its margin — and strips out the flat-price risk it does not care about. This "trade the relationship, not the level" logic is the beating heart of commodity spread trading, and the crack is its cleanest example in the energy patch.

#### Worked example: when the paper crack and the physical crack diverge

A Midwest refiner sells the 3-2-1 crack at a benchmark **\$25.00**. Over the next month the benchmark crack stays at \$25.00, so its futures hedge is flat — no gain, no loss. But its local situation shifts: a competing pipeline reopens and floods the Midwest with gasoline, pushing its *local* gasoline realisation **\$3.00 a barrel below** the New York benchmark it hedged against. At the same time the cheap Canadian crude it runs gets **\$2.00 a barrel cheaper** relative to WTI (a feedstock tailwind). Its physical crack ends up at \$25.00 − \$3.00 (weaker local product) + \$2.00 (cheaper feedstock) = **\$24.00** — \$1.00 below the locked benchmark, despite a "perfect" hedge on paper. **The basis is the part of a refiner's margin that lives in the physical world and refuses to be captured by a standardised futures contract — it is small most of the time, but it is never exactly zero.**

## Not all barrels are equal: feedstock and refinery complexity

So far we have treated "crude" as one thing. It is not. The single biggest reason two refineries side by side can earn wildly different margins on the same day is that they run *different crude* through *different machinery*. This is where the real money in refining is made, and it is the most underappreciated part of the whole business.

Crude oil varies along two axes that matter enormously:

- **Density — "light" vs "heavy."** Light crude (high API gravity) is full of the small molecules that make gasoline and diesel; it is easy to refine. Heavy crude is full of big tar-like molecules; you have to crack a lot of it to get valuable products.
- **Sulfur — "sweet" vs "sour."** Sweet crude has little sulfur and is clean to process. Sour crude is loaded with sulfur, which is corrosive, poisons catalysts, and must be stripped out (and disposed of) before the products can be sold to modern fuel standards.

The premium grades are **light and sweet** (like WTI or Brent): easy to refine, so they command a higher price. The discount grades are **heavy and sour** (like Canadian Western Canadian Select, or much Venezuelan and Middle Eastern crude): hard to refine, so they sell *cheaper* — sometimes \$10 to \$30 a barrel below light-sweet benchmarks. For the full story of why the two benchmark light-sweet barrels themselves diverge, see [crude oil WTI vs Brent](/blog/trading/commodities/crude-oil-wti-vs-brent-the-worlds-two-benchmark-barrels).

Now, the crucial twist: *whether a refinery can profitably run cheap heavy-sour crude depends entirely on how sophisticated its machinery is.* That sophistication has a number.

### Nelson complexity: the refinery's horsepower rating

The **Nelson Complexity Index (NCI)** rates how much conversion machinery a refinery has, relative to its simple distillation capacity. A bare-bones refinery that only distils crude scores around **1 to 5**. A refinery loaded with crackers, cokers, and hydrotreaters — able to upgrade the heavy bottom of the barrel and strip sulfur — scores **10 to 14 or higher**. Some Gulf Coast and Indian "super-refineries" (like Reliance's Jamnagar) score above 12.

Here is why complexity is the whole game. A **simple** refinery has no way to deal with heavy, sour molecules — it would be left with a mountain of worthless high-sulfur residue. So it *must* buy expensive, clean light-sweet crude, and it earns only an ordinary margin because it pays full price for its feedstock. A **complex** refinery, by contrast, can buy the *cheap heavy-sour* crude, crack the heavy molecules into light products, and strip out the sulfur — turning a discounted, despised barrel into the exact same valuable gasoline and diesel. It captures the heavy-crude discount as pure margin.

![Matrix of simple versus complex refinery against light-sweet versus heavy-sour crude showing where margin is earned](/imgs/blogs/refining-and-crack-spreads-turning-crude-into-products-5.png)

The matrix above is the strategic map. The sweet spot — fat margins — sits in one cell only: a *complex* refinery running *heavy-sour* crude. That refiner buys the cheapest barrels on earth and sells the same premium products as everyone else. This is why complex refiners spend billions on conversion units: every dollar of light-heavy price spread they can capture drops to the bottom line. And it explains a counter-intuitive market behaviour — when heavy crude gets *cheaper* relative to light crude (a wider light-heavy differential), complex refiners' margins get *better*, even if the headline crude price is flat.

There is a maintenance and uptime dimension to complexity too. A complex refinery is a chain of interdependent units — distillation feeds the cracker, the cracker feeds the hydrotreater, and so on. If any one link goes down, the whole chain can be throttled. This makes complex refineries more vulnerable to *unplanned outages* (a cracker fire, a coker upset), and an outage at a big complex plant removes a chunk of product supply from the market overnight — which is precisely the kind of shock that sends the crack spiking for the *surviving* refiners. So complexity is double-edged: it lets a plant earn the heavy-crude discount, but it also concentrates risk, and one refiner's outage is another refiner's windfall. This is why crack spreads can jump on a single headline about a fire at a major refinery: the market instantly reprices the value of every still-running plant's margin.

Complexity also opens the door to the **petrochemical pivot**. The most advanced sites do not stop at fuels — they route naphtha and other light streams into steam crackers that make ethylene and propylene, the building blocks of plastics. When fuel cracks are weak but petrochemical margins are strong, an integrated complex can lean its barrel toward chemicals instead of fuels, smoothing its earnings across cycles. A pure fuel refiner has no such escape hatch. This is one more reason the highest-complexity, most-integrated assets are the survivors: they hold the most options on what to do with each part of the barrel, and optionality is worth the most precisely when markets are volatile.

#### Worked example: the complexity advantage on cheap heavy crude

Two refineries face the same product market where the slate is worth **\$100 a barrel**. The simple refinery must buy light-sweet crude at **\$85**, earning a gross margin of \$100 − \$85 = **\$15 a barrel**. The complex refinery buys heavy-sour crude at a \$20 discount — **\$65** a barrel — and, thanks to its crackers and cokers, still turns it into the same \$100 slate (it spends maybe \$4 a barrel more on energy and hydrogen to do so). Its gross margin is \$100 − \$65 − \$4 = **\$31 a barrel** — more than double the simple plant's. **Complexity is a machine for converting the heavy-crude discount into refining margin, which is why the most profitable refineries on earth deliberately run the nastiest, cheapest crude.**

## Seasonality: the refining calendar

Refining margins are not constant through the year, because demand for the *products* swings with the seasons and refiners physically reconfigure their plants to chase it. Understanding the calendar is essential to reading the crack, because a crack that looks "high" in May and "high" in November is telling you two different stories.

![Timeline of the refining year from spring turnaround to summer gasoline to winter distillate](/imgs/blogs/refining-and-crack-spreads-turning-crude-into-products-7.png)

The rhythm runs like this:

- **Late winter / early spring (Feb–Apr): turnaround season.** Refineries take units offline for scheduled maintenance ("turnarounds") and switch their gasoline recipe to the summer blend (a lower-volatility formula required for hot-weather air quality). Capacity tightens, gasoline inventories draw down, and the gasoline crack starts to climb in anticipation of summer.
- **Summer (May–Aug): driving season.** Americans drive most in summer, so gasoline demand peaks. Refiners maximise gasoline yield, and the **gasoline crack is typically widest of the year**. This is the seasonal sweet spot for the gasoline-heavy United States system.
- **Autumn (Sep–Oct): the shoulder + fall turnaround.** Driving demand fades, a second round of maintenance happens, and refiners begin shifting the slate back toward distillate.
- **Winter (Nov–Jan): heating season.** Cold weather lifts demand for heating oil and diesel, so the **distillate crack tends to lead** and refiners tilt their yield toward diesel and heating oil.

So a refinery is not a fixed machine producing a fixed slate — it is a flexible plant that *re-tunes itself* through the year, leaning toward gasoline before summer and distillate before winter, chasing whichever product carries the fattest crack. A trader who watches the crack must always ask: is this strength seasonal and expected, or is it a genuine signal that supply is short?

The seasonality also shows up in **inventories**, which are the early-warning system for the crack. Refiners and the wider system build product stocks ahead of the demand season and draw them down through it. Gasoline inventories typically build over winter and draw hard through summer; distillate stocks build over summer and draw through winter. When the weekly inventory reports (in the United States, the Energy Information Administration's Wednesday data) show stocks drawing down *faster* than the season normally dictates, the crack for that product tends to firm — the market is signalling tightness. When stocks pile up unexpectedly, the crack softens. Reading the crack and the inventory trend together is far more powerful than reading either alone: a rising crack on falling inventories is a genuine tightening, while a rising crack on building inventories is more likely a temporary disruption (a refinery outage) that will mean-revert.

A second-order effect worth knowing is the **gasoline-blend switch** itself. The summer-grade gasoline mandated in much of the United States has lower Reid vapour pressure (it evaporates less, to cut summer smog), and it is more expensive and finicky to make. The spring switchover temporarily reduces effective gasoline supply just as demand ramps — which is a structural reason the gasoline crack so reliably firms in April and May. It is a regulatory feature that creates a recurring, almost calendar-clockwork pattern in the spread. Knowing it exists keeps you from mistaking a predictable spring rally in the gasoline crack for fresh fundamental news.

## Why refining can boom when crude is flat

Now we can answer the puzzle from the introduction. The headline crude price and the refining margin are *different numbers driven by different forces*, and they routinely diverge. Crude is set by the global balance of oil *production* vs *consumption* — OPEC quotas, shale output, demand growth. The crack spread is set by the balance of *refining capacity* vs *product demand* — how much gasoline and diesel the world's refineries can make vs how much the world wants to burn.

![Dual axis chart of WTI crude price and the 3-2-1 crack spread from 2018 to 2024 showing they move apart](/imgs/blogs/refining-and-crack-spreads-turning-crude-into-products-4.png)

The dual-axis chart makes the divergence visible. The crude line (left axis) and the crack line (right axis) sometimes move together and sometimes pull apart. The decisive case is the period after 2022: crude stayed *high* through 2023–24 (around \$77), but the crack *fell back* from \$38 toward \$21 as refining bottlenecks eased and new capacity opened. Conversely, in 2022 the crack roughly doubled while crude rose far less in percentage terms. **The level of oil tells you almost nothing about whether refining is profitable — only the spread does.**

The 2022 blowout is the textbook example of a *refining-capacity* shock independent of a *crude* shock. Three things stacked up:

1. **Capacity had been permanently lost.** During the 2020 demand collapse, refiners worldwide shut roughly 2–3 million barrels a day of capacity for good — older, simpler plants that never came back.
2. **Diesel went catastrophically short.** Europe had relied on Russian diesel; the invasion of Ukraine and the sanctions that followed pulled those barrels out of the Western market, and the world has no spare diesel-refining capacity to replace them quickly.
3. **Demand snapped back** post-pandemic faster than the surviving refineries could supply.

The result: a structural shortage of *refined products* on top of an ordinary crude market. Refiners with running capacity collected enormous tolls. It is the clearest demonstration of the thesis: refining is a capacity-bottleneck business, and bottleneck businesses mint money when the bottleneck binds — regardless of what the underlying commodity costs. For how that fuel-cost surge then transmitted into inflation and equities, see [oil, inflation, and equities](/blog/trading/cross-asset/oil-inflation-and-equities-the-energy-linkage).

### The geography of refining is shifting, and so is the crack

There is a slower, structural story behind the headline cracks that any serious reader should hold. The world's refining capacity is migrating. For decades it sat in the United States, Europe, and Japan, close to the rich consuming markets. Now the centre of gravity is moving toward Asia and the Middle East — new mega-refineries in China, India, Saudi Arabia, Kuwait, and Nigeria (the giant Dangote refinery), often complex and built next to petrochemical plants. Meanwhile older, simpler, sub-scale refineries in the high-cost West keep closing.

This migration has two consequences for the crack. First, **capacity additions are lumpy and slow**: a new mega-refinery takes the better part of a decade and then arrives all at once, adding a million barrels a day of product supply in one step. That lumpiness is why cracks can stay elevated for years (during a capacity drought) and then compress sharply when a wave of new plants starts up — which is roughly the 2022-to-2024 arc, where the crack fell from \$38 back toward \$21 as Asian and Middle Eastern capacity ramped. Second, the **product trade flows are lengthening**: as refining moves away from demand centres, more diesel and gasoline travel by tanker across oceans, which adds freight cost and creates regional crack spreads (a Gulf Coast crack, a Singapore crack, a Northwest Europe crack) that can diverge sharply when shipping is disrupted. The crack you watch is increasingly a *regional* number, and the differences between regions are themselves tradable.

The deeper point ties back to the asymmetry that makes refining such a feast-or-famine business. Capacity can vanish in a quarter (one shutdown) but takes a decade to add (one new build). When demand outruns capacity, there is no quick fix, so the crack can run hot far longer than a textbook supply curve would predict. When capacity finally catches up, it overshoots and the crack collapses. This is the boom-and-bust signature of every capacity-constrained processing industry, and it is why refiner equities are deeply cyclical — you are buying a toll booth whose toll swings from punishing to princely on a multi-year cycle.

## Common misconceptions

**"High oil prices mean fat refining profits."** No — refining profit is the *spread*, not the level. A refiner can lose money with crude at \$60 (if products are weaker still) and mint money with crude at \$60 (if products are far stronger). In 2020 crude was cheap *and* the crack was near \$11, so refiners suffered. In 2022 crude was high *and* the crack was \$38, so they thrived. The level and the margin are independent.

**"A barrel of crude is a barrel of crude."** Light-sweet and heavy-sour crude can trade \$10–30 apart, and that differential is a profit centre for complex refiners and irrelevant to simple ones. Treating "crude" as a single price hides where most refining money is made.

**"The crack spread predicts future oil prices."** It does not. The crack is a *current margin*, the relationship between today's product prices and today's crude. A high crack signals tight *product* supply or strong *fuel demand* right now — useful information, but it is not a forecast of where crude is heading. (The shape of the crude futures curve carries the market's time-pricing; that is the domain of [contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means), not the crack.)

**"Refiners control gasoline prices, so the crack proves they are gouging."** The crack measures gross margin, not net profit — out of it the refiner still pays for the natural gas to run the units (a huge cost), labour, maintenance, and capital. A \$30 crack is a great gross margin, but the cash refining margin after operating costs is meaningfully lower. And the crack is set by the *market* balance of capacity and demand, not by any one refiner's pricing power; when capacity is ample, the crack collapses to single digits no matter how much a refiner might wish otherwise.

**"More refining capacity is always coming, so high cracks can't last."** New refineries take years to build and increasingly few get built in the West due to cost and policy. Capacity can be *lost* in months (a shutdown) but *added* only over years. That asymmetry is exactly why crack spreads can stay elevated far longer than a naive supply-and-demand model would suggest.

**"The crack spread is the refiner's profit, so a \$30 crack means \$30 of profit per barrel."** No — the crack is a *gross* margin. The refiner still pays its energy bill (natural gas to fire the units is often the largest single operating cost, and it can swing the cash margin by several dollars a barrel on its own), labour, maintenance, and the depreciation of the multi-billion-dollar plant. The cash margin a refiner actually banks is meaningfully below the headline crack, which is why a refiner that watches the crack obsessively also watches its own natural-gas cost just as closely — a high crack with high gas prices can be a worse year than a moderate crack with cheap gas.

## How it shows up in real markets

**The 2022 refiner windfall.** As covered above, the 3-2-1 crack averaged around \$38 for the year and spiked far higher intra-year, with United States diesel cracks at points exceeding \$60 a barrel. Refiner equities (Valero, Marathon, Phillips 66) doubled or more. The lesson for a reader: when you see fuel prices spiking and want to know *who profits*, the crack spread tells you it is the refiners with running capacity — not necessarily the oil producers.

**The 2020 negative-margin shutdown.** When the pandemic crushed driving and flying, product demand fell off a cliff while crude was being pumped into a glutted market. The crack collapsed toward \$11 and, for some products in some regions, refining was uneconomic — you would lose money turning crude into gasoline nobody was buying. Refiners cut runs and several older plants closed permanently. That permanent capacity loss is precisely what set up the 2022 boom: the supply side could not snap back. This is the boom-and-bust cycle of a capacity business in miniature.

**The quiet baseline: 2015–2019.** Between the boom years it is worth knowing what "normal" looks like. From 2015 to 2019, with the United States shale boom keeping crude well supplied and global refining capacity adequate, the 3-2-1 crack mostly oscillated in a calm \$15–20 band. Refiners earned a steady, unglamorous gross margin; nobody wrote angry headlines about them. That baseline is the reference point that makes the extremes legible — the 2020 collapse to ~\$11 and the 2022 spike to ~\$38 are dramatic precisely *because* the normal range is so much narrower. When you look at a crack number, the first question is always "where does this sit relative to the \$15–20 normal?"

**Reading the crack as a demand signal.** Because the crack reflects the product-vs-crude balance, a *rising* gasoline crack going into summer that is *stronger than seasonal* is a real-time signal that fuel demand is robust or product inventories are tight — useful well beyond the energy desk, because fuel demand is a clean read on real economic activity (people driving, trucks moving goods). A *collapsing* crack can be an early warning of a demand slowdown. Diesel in particular is the workhorse of the freight economy, so a weakening *distillate* crack is one of the more reliable early tells that industrial activity and goods movement are cooling — it often softens before the broad economic data confirm a slowdown. Commodities like oil, copper, and the products refined from them are watched precisely because they reveal the physical economy in real time; for that broader lens see [commodities as macro signals](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold).

**The light-heavy differential as a complex-refiner tell.** When sanctions or quality shifts make heavy-sour crude unusually cheap relative to light-sweet, complex refiners' margins quietly improve even if the headline crack looks ordinary. Watching the light-heavy spread (e.g. the WCS-WTI or the Brent-Dubai differential) tells you which refiners are about to print, and is a layer of information the simple 3-2-1 crack hides. A vivid recent case: when Western sanctions pushed Russian crude to deep discounts after 2022, the complex refiners (notably in India) that could buy and process that discounted heavy crude earned outsized margins, while their feedstock cost fell relative to the global benchmark even as their product realisations stayed at world prices. The discount on the politically-stranded barrel went straight into their crack.

**The crack tells you who profits when fuel prices spike.** This is the single most practical use for a non-specialist. When pump prices soar and the news asks "who is getting rich," the crack spread is the answer key. If the *crude* price is what rose, the windfall accrues to oil *producers* (and their host governments). If the *crack* is what rose, the windfall accrues to *refiners*. In the 2022 episode both rose, but the crack rose proportionally far more — which is exactly why refiner stocks, not producer stocks, were the standout performers of that year. Decompose any fuel-price spike into "how much is crude, how much is the crack" and you immediately know where the money is flowing.

**The crack and the curve are different signals.** A refiner can hedge its *margin* with the crack and separately face the *shape* of the crude curve when it stores oil or rolls a position. The two are independent: a fat crack tells you product supply is tight today; a backwardated crude curve tells you crude supply is tight today. They often, but not always, point the same way. Keeping them mentally separate prevents the common error of treating "the oil market is tight" as one undifferentiated signal when it is really several, each readable off a different spread.

## The playbook: how to read and use the crack

You now have the toolkit. Here is how a curious investor should actually use it.

- **Separate the level from the margin, always.** When you hear "oil is up," ask the second question that almost nobody asks: *what is the crack doing?* Refiner profitability, fuel-station economics, and the inflation pass-through to consumers all live in the crack, not the crude price. The two can move in opposite directions.

- **Use the 3-2-1 crack as your one refining gauge.** Around \$15–20 is normal; below \$10 means refiners are hurting and may cut runs (which then tightens product supply and pushes the crack back up — a self-correcting cycle); above \$30 means a genuine product shortage and a refiner windfall. Watch where it sits relative to that band, and relative to the season.

- **Mind the calendar.** A high gasoline crack in May is half-expected (summer demand); a high gasoline crack in January is a real anomaly worth investigating. Always seasonally-contextualise the number before you treat it as a signal.

- **Look at complexity and the light-heavy spread to find the winners.** Among refiners, the complex Gulf Coast and Indian plants running discounted heavy-sour crude earn the fattest, most durable margins. When the light-heavy differential widens, those names benefit even when the simple crack is flat. The discount on the ugly barrel is the prize.

- **Treat the crack as a live demand and capacity signal, not a forecast.** A strong, sustained, above-seasonal crack tells you product demand is outrunning refining capacity *right now* — a real-economy read and a refiner tailwind. It does not tell you where crude is going next; that question belongs to the supply-demand balance and the shape of the futures curve.

- **Cross-check the crack against inventories.** A rising crack on *falling* product inventories is a real tightening you can lean on; a rising crack on *building* inventories is more likely a temporary refinery outage that will mean-revert. The two data series together are far more reliable than the crack alone — they let you distinguish genuine demand strength from a transient supply blip.

- **Remember the gross-vs-net gap.** The crack is a gross margin; subtract a refiner's energy and operating costs (often \$5–9 a barrel) to estimate what it actually banks. When natural-gas prices are high, a fat-looking crack can disguise a mediocre cash margin. Always ask what the refiner is paying to do the cracking, not just what the spread says.

Step back to the spine of this whole series: a commodity price is a physical thing forced through a financial contract, and the money is made in the *relationships* between prices — the curve, the basis, the spread — far more than in the level of any single price. The crack spread is the purest example. The refiner does not bet on the price of oil; it collects the difference between the dirty barrel it buys and the clean fuels it sells, and that difference is a contract you can put on a screen, chart through the seasons, and trade. Learn to read the crack and you can see, in one number, who is winning in the downstream energy world and whether the world is short the fuels that actually move it.

## Further reading & cross-links

- [The Oil Value Chain: From Wellhead to Gas Pump](/blog/trading/commodities/the-oil-value-chain-from-wellhead-to-gas-pump) — where refining sits in the upstream–midstream–downstream chain and how a barrel travels.
- [Crude Oil: WTI vs Brent, the World's Two Benchmark Barrels](/blog/trading/commodities/crude-oil-wti-vs-brent-the-worlds-two-benchmark-barrels) — the two reference crudes and the light-sweet benchmarks against which feedstock discounts are measured.
- [Calendar Spreads and Curve Trades: Trading the Shape, Not the Level](/blog/trading/commodities/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level) — the other great "trade the spread, not the level" idea in commodities.
- [Contango vs Backwardation: What the Shape of the Curve Means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — how the crude futures curve, not the crack, carries the market's pricing of time.
- [Oil, Inflation, and Equities: The Energy Linkage](/blog/trading/cross-asset/oil-inflation-and-equities-the-energy-linkage) — how a fuel-price and refining-margin surge transmits into inflation and stock markets.
- [Energy: Oil & Gas, the Inflation Engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) — the energy complex as an asset class and macro driver.
- [Commodities as Macro Signals: Oil, Copper, Gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) — reading the physical economy off commodity and product spreads.
- [Hedging a Portfolio with Options: Protective Puts, Collars, and Tail Risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) — the consumer's mirror of the refiner's hedge: capping risk without surrendering all the upside.
