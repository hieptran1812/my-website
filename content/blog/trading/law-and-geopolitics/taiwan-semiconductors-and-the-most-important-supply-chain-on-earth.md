---
title: "Taiwan, semiconductors, and the most important supply chain on earth"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why one island makes the chips the whole economy runs on, what the silicon shield really is, and how a strait full of geopolitical risk prices into your portfolio."
tags: ["geopolitics", "semiconductors", "taiwan", "supply-chain", "tail-risk", "tsmc", "chip-war", "concentration-risk", "national-security", "hedging"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A single island, through a single company, makes roughly 90% of the world's most advanced chips, which means one 110-mile-wide strait sits underneath the valuation of every technology, auto, and AI company on earth — the largest single-point-of-failure tail risk in markets.
>
> - **The chokepoint is real and narrow.** TSMC holds about 64% of all foundry revenue and roughly 90% of leading-edge (<7nm) capacity, almost all of it in Taiwan; the one machine that makes those chips possible, the EUV scanner, comes from one company (ASML) in the Netherlands.
> - **The cascade is non-linear.** A few dollars of missing chips can halt thousands of dollars of downstream output — a 200 dollar phone or a 40,000 dollar car will not ship for the want of a 10 dollar part — so estimates of a full cutoff run from a deep global recession to a multi-trillion-dollar GDP hit.
> - **The market mostly does not price it.** Geopolitical tail risk shows up in prices in spikes, not steadily; the "Taiwan risk" is dormant in chip multiples until a headline wakes it up, then fades.
> - **The number to remember:** ~90% of the leading edge is made within artillery range of one strait, and the diversification response (CHIPS Act fabs, TSMC Japan and Germany) moves the *trailing* edge first — the leading edge stays in Taiwan into the 2030s.

On August 2, 2022, a US congressional delegation landed in Taipei. Within hours, China announced live-fire military exercises encircling Taiwan, and over the following days launched ballistic missiles over the island and suspended a list of trade flows. It was the sharpest cross-strait escalation in a generation. And yet, if you were watching only the price of TSMC's American depositary receipts, you would have seen something almost anticlimactic: a few percent of wobble, a quick recovery, and within two weeks the stock trading roughly where it started. The most important supply chain on earth had just been rattled, and the market shrugged.

That gap — between the size of the risk and the size of the price reaction — is the whole subject of this post. It is not that the market is stupid. It is that a fat-tail event is, by definition, low-probability in any given month, and markets price the *expected* value of a risk, not its worst case. Most cross-strait scares fade, so the base rate teaches investors to fade them too. The problem is that the *one* time it does not fade, the repricing is not a few percent. It is a generational shock to the entire technology complex and, plausibly, the global economy.

This is a law-and-geopolitics post, so the spine is the same one that runs through this whole series: **geopolitics → policy → macro and flows → asset prices → the trade.** A change in the *rules of the game* — here, the physical and political control of a chokepoint — changes the expected path of supply, and markets discount that change into prices. Your job as an investor is to understand the chokepoint precisely enough to know what reprices if it is ever disrupted, to size that tail rather than pretend it is zero, and to know what would invalidate your view. We will build the whole thing from zero — what a semiconductor supply chain even is, why advanced chips concentrate, what the "silicon shield" means — and end on the playbook.

![Diagram showing Taiwan chip concentration feeding the world economy and a cross-strait tail risk](/imgs/blogs/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth-1.png)

## Foundations: what the chip supply chain is and why it concentrates

Let us start with the thing itself. A semiconductor — a "chip" — is a small slab of silicon with billions of microscopic switches (transistors) etched onto it. Every phone, laptop, car, data center, missile, and washing machine made today contains them. The world makes well over a trillion chips a year. They range from the trivially cheap (a few cents for a basic power-management chip) to the staggeringly expensive (an Nvidia data-center GPU can sell for tens of thousands of dollars).

Here is the first thing a beginner has to internalize: **chips are not made by the companies whose names are on them.** Apple designs the chip in your iPhone but does not manufacture it. Nvidia designs the GPU that trains AI models but does not own a factory that can make it. This split — between companies that *design* chips and companies that *manufacture* them — is the single most important structural fact in the industry, and it is the reason a chokepoint exists at all.

### The four stages of the chain

To see the chokepoints, you have to walk the chain. There are four core stages, and each has a different geography and a different bottleneck.

![Pipeline of the chip supply chain from design through equipment fabrication assembly to end market with choke points marked](/imgs/blogs/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth-2.png)

**1. Design.** A company decides what the chip should do and lays out the billions of transistors. This stage is dominated by US-headquartered firms — Nvidia, Apple, AMD, Qualcomm, Broadcom — plus the software tools they design with (electronic design automation, or EDA, led by Synopsys and Cadence) and the foundational circuit blueprints they license (intellectual property, or IP, led by Arm). A company that designs chips but owns no factory is called **fabless** ("without a fab"). The chokepoint here is concentrated in US firms and US-controlled software.

**2. Equipment.** Before anyone can manufacture a chip, they need the machines. The most extraordinary of these is the **EUV (extreme ultraviolet) lithography scanner** — the machine that prints the finest features onto silicon using light with a wavelength of 13.5 nanometers. It is, plausibly, the most complex machine humans build. Exactly **one** company on earth makes it: **ASML**, headquartered in the Netherlands. There is no second source. This is the deepest chokepoint in the whole chain, and it sits outside Taiwan entirely.

**3. Fabrication ("fab").** This is where the design is physically etched onto silicon wafers. A fabrication plant — a "fab" — is a multi-billion-dollar facility where a state-of-the-art line can cost 20 billion dollars or more to build. A company that manufactures chips for *other* companies' designs is called a **foundry**. The dominant foundry, by an enormous margin, is **TSMC (Taiwan Semiconductor Manufacturing Company)**, headquartered in Hsinchu, Taiwan. This is the chokepoint everyone means when they say "Taiwan risk."

**4. Assembly and test (OSAT).** The finished wafer is cut into individual chips, packaged, and tested. This stage — outsourced assembly and test, or OSAT — is also concentrated in Taiwan and China, though it is the least difficult to relocate.

The chain then feeds **devices** (phones, GPUs, cars, servers) and finally the **end market** — global demand. Read the pipeline left to right and you see the structure: the danger is concentrated at stages 2 and 3, equipment and leading-edge fabbing, and those two chokepoints sit in two different countries.

### The foundry model and why TSMC won

Why did manufacturing concentrate into one company in one place? The answer is the economics of the foundry model and the brutal physics of **Moore's Law**.

In the 1980s, most chip companies made their own chips — they were "integrated device manufacturers" (IDMs), like Intel. Then, in 1987, Morris Chang founded TSMC on a radical idea: a factory that builds *no chips of its own* and *competes with no one*, that simply manufactures whatever its fabless customers design. Because TSMC never competes with Nvidia or Apple, those companies can trust it with their crown-jewel designs. And because every fabless company in the world routes its volume through the same foundries, the leading foundry accumulates more manufacturing volume than any single IDM ever could.

That volume matters because of cost. Each new generation of chip — each new "node" — requires a fab that costs more than the last, and the only way to earn back a 20-billion-dollar fab is to run enormous volume through it. The more volume you have, the more you can spend on the next node; the more you spend, the better your next node; the better your node, the more volume you win. This is a **winner-take-most flywheel**, and TSMC has been spinning it for three decades. The result is the concentration in the chart below.

![Horizontal bar chart of global foundry revenue share with TSMC at 64 percent far ahead of Samsung and others](/imgs/blogs/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth-3.png)

TSMC holds roughly 64% of all foundry revenue. Its nearest competitor, Samsung, holds about 11%. China's SMIC, hobbled by export controls, holds about 6%. No one else is close.

To feel why this concentration is so sticky, sit with the unit economics of a fab for a moment. A leading-edge fab is not a factory you build once and run cheaply forever. It is a continuous, brutal capital treadmill. The building and clean-room shell are a fraction of the cost; the bulk is the tools — hundreds of machines, each a marvel, with the EUV scanners alone running well over \$150 million apiece and the most advanced "High-NA" EUV tools approaching \$400 million each. A single leading-edge line can run \$20 billion or more before it makes its first salable wafer. That capital has to be earned back over a node generation that lasts only a few years before the next, more expensive node makes it obsolete. The only way the arithmetic works is volume — and the only foundry with enough volume is the one that already won the last round. That is the treadmill: you spend more to stay ahead, you stay ahead because you spent more, and a challenger trying to enter has to match a \$20-billion bet against an incumbent who has been climbing a yield-learning curve for years and will simply out-spend and out-yield them. This is why, despite three decades and the deepest pockets in technology trying, the number of companies that can manufacture at the leading edge has *shrunk* from over twenty to effectively one.

Yield is the other half of the moat, and it is the part outsiders underestimate. **Yield** is the fraction of chips on a wafer that come out working. On a brand-new node, yield can start abysmally low — a large share of every wafer is scrap — and it climbs only with cumulative manufacturing experience, as engineers hunt down the thousands of subtle defects that ruin chips. A foundry that has run ten million leading-edge wafers has seen failure modes a newcomer has never encountered. Because TSMC manufactures the most leading-edge volume on earth, it has the best yields on earth, which means the lowest effective cost per *good* chip, which wins it still more volume. Yield is not something a competitor can buy with a subsidy; it is learned, wafer by wafer, over years. That is precisely why a brand-new US or European leading-edge fab, even fully funded, is not yield-competitive on day one — and why the concentration the chart shows is far more durable than the chart alone suggests.

### What "advanced node" means and why it concentrates harder

But raw foundry share understates the concentration, because not all chips are equal. A "node" is, roughly, a measure of how small the transistors are — labeled in nanometers (nm), like "7nm," "5nm," "3nm." Smaller nodes pack more transistors into the same area, which means faster, more power-efficient chips. The frontier — the smallest nodes in mass production — is called the **leading edge**. Everything else is the **trailing edge** or **mature nodes**.

Here is the crucial point: **the more advanced the node, the more concentrated it is in Taiwan.** A trailing-edge 28nm chip (the kind in a car's engine controller or a power-management chip) can be made in dozens of fabs around the world. A leading-edge chip below 7nm — the kind in an iPhone's main processor or an Nvidia AI accelerator — can be made, at scale and at yield, in essentially one place. TSMC manufactures something like **90% of the world's leading-edge capacity**, almost all of it in Taiwan.

![Stacked bar comparing TSMC share of all foundry revenue at 64 percent versus 90 percent of leading edge capacity](/imgs/blogs/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth-4.png)

So the concentration *rises with the node*. TSMC is 64% of the whole market but ~90% of the part that matters most for AI, flagship phones, and the highest-margin silicon on earth. The leading edge is where the single point of failure lives.

Why does the leading edge concentrate harder than the trailing edge? Three reinforcing reasons. First, **cost**: a leading-edge fab is the most expensive industrial facility built today, and only the highest-volume foundry can amortize it. Second, **EUV access**: leading-edge nodes require ASML's EUV scanners, of which a limited number are made each year, and they go to the foundry that can run them most profitably. Third, **yield learning**: making a working leading-edge chip is fiendishly hard, and yield (the percentage of good chips per wafer) improves only with cumulative experience — the foundry that has made the most leading-edge wafers makes the best ones. All three favor the incumbent. The flywheel spins fastest where it is hardest to catch up.

### The silicon shield and the cross-strait status quo

Now we can define the geopolitical layer. The **"silicon shield"** is the thesis that Taiwan's indispensability to the world's chip supply acts as a form of deterrence. The argument runs: because a disruption of TSMC would devastate the economies of the United States, China, Japan, Europe, and China itself (which depends on Taiwanese chips for its own electronics), no major power has an interest in seeing the island's fabs destroyed or stopped. The chips, in this telling, are a shield — they make Taiwan too economically valuable to attack and too valuable to let be attacked.

It is an elegant idea and almost certainly *part* of the deterrence story. But as we will see, it is also routinely overstated, and a serious investor should treat it as one factor among several, not as a guarantee.

The **cross-strait status quo** is the delicate, decades-old arrangement in which the relationship between Taiwan and mainland China is neither formal independence nor unification, held in a deliberate strategic ambiguity that the United States and others have historically supported. The status quo is what keeps the chips flowing. The tail risk is any scenario in which it breaks. And — this is the analytically important part — the ways it could break are *not* a single event. They are a family of distinct scenarios with very different consequences, which we will lay out precisely in the scenario section. Throughout, the point of this post is descriptive and analytical: how these possibilities price into markets, not a forecast of any of them, and certainly not advocacy for any political outcome.

## The concentration risk: a single point of failure for the entire tech complex

Let us make the single-point-of-failure idea concrete, because it is the load-bearing claim of the whole thesis.

When engineers design a system for reliability, they hunt for the **single point of failure** — the one component whose failure takes down everything. A data center with two power feeds, three network paths, and redundant servers, but only one cooling pump, has a single point of failure at the pump. The global electronics economy has spent thirty years building redundancy into almost everything *except* leading-edge chip manufacturing, where it has, if anything, *reduced* redundancy by consolidating into one company in one place.

Look at who depends on TSMC's leading edge. Apple's flagship chips: TSMC. Nvidia's AI accelerators — the silicon underneath the entire generative-AI boom: TSMC. AMD's CPUs and GPUs: TSMC. Qualcomm's premium smartphone chips: TSMC. The list of companies whose most important and highest-margin products *cannot currently be made anywhere else* reads like a roll call of the most valuable corporations on earth. The combined market capitalization of the companies that depend on TSMC's leading edge runs into the many trillions of dollars. Under all of that sits one foundry, in one country, across one strait.

#### Worked example: how much of a designer's revenue rides on one node

Take a stylized fabless designer — call it a flagship-AI-chip company. Suppose its data-center GPU line generates \$80 billion of annual revenue, and that entire line is manufactured on TSMC's leading-edge 4nm and 3nm nodes. Its older, mature-node products (networking chips, embedded parts) generate another \$10 billion and could, in a pinch, be made elsewhere.

What share of revenue is exposed to a single foundry's leading edge?

```
Leading-edge-dependent revenue = 80
Total revenue                  = 80 + 10 = 90
Single-node exposure           = 80 / 90 = 88.9%
```

So **roughly 89% of this company's revenue cannot be produced anywhere but TSMC's Taiwan fabs.** There is no qualified second source for those nodes at the volume and yield required — Samsung's leading edge is not yield-competitive for these parts, and a US fab capable of the same node is years from full volume. Put plainly: when people say a company is "exposed to Taiwan risk," they often mean its supply chain touches Taiwan; the sharper statement is that a supermajority of its *most profitable revenue* has no alternative manufacturer at all.

That is the difference between a *supply-chain dependency* (annoying, reroutable) and a *single point of failure* (existential, not reroutable). The leading edge is the latter.

### It is not only TSMC: the deeper chokepoints

A subtle and important point: even if TSMC's fabs were perfectly safe, the chain would still have chokepoints *upstream* of them. The fabs cannot run without the EUV scanners, and those come from ASML alone. ASML's scanners cannot run without specialized optics from Carl Zeiss (Germany) and light sources from Cymer (US-owned). The whole edifice also runs on a handful of specialty materials — ultra-pure neon gas (historically much of it from Ukraine), photoresist chemicals (concentrated in Japan), and specialty silicon wafers (concentrated in Japan). Japan's Shin-Etsu and SUMCO together make a large share of the world's polished wafers.

So the supply chain is a *chain of chokepoints*, not one. This matters for the misconception we will tackle later — that "only Taiwan matters." A serious map of the tail risk has to flag at least three: leading-edge fabbing (Taiwan), EUV equipment (the Netherlands), and several specialty materials (Japan, and formerly Ukraine for neon). Disrupt any one badly enough and the leading edge stops.

## The cascade: why a few dollars of missing chips halt thousands of dollars of output

Here is where the tail risk gets its teeth. The dangerous feature of a chip shortage is not the dollar value of the missing chips — it is the **multiplier** between the chip and the device it gates.

A car contains hundreds to over a thousand chips. Many are cheap mature-node parts — a microcontroller (MCU) that runs a window motor might cost a dollar or two. But a car cannot be sold incomplete. If one of those microcontrollers is missing, a 40,000-dollar vehicle does not ship. We do not have to theorize about this: in 2021, a relatively *mild* chip shortage — driven by mature-node MCUs, not even the leading edge — idled auto plants worldwide and cut global vehicle production by an estimated several million units, costing the auto industry something like \$200 billion in lost revenue. That shortage never touched TSMC's leading edge. It was a shortage of *cheap* chips.

Now extrapolate to a leading-edge cutoff, and the cascade is far worse, because the leading edge gates the highest-value devices — flagship phones, the servers that run the cloud, and the AI accelerators that an entire wave of corporate capital expenditure depends on.

![Cascade diagram from a leading edge chip cutoff through compute consumer and auto sectors to a global GDP hit](/imgs/blogs/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth-5.png)

Read the cascade left to right: a leading-edge cutoff stops AI and data-center GPUs, idles phone and PC lines, and (combined with mature-node disruption) starves autos of microcontrollers. That freezes technology capital expenditure and idles factories across the economy, which feeds into a global GDP hit.

#### Worked example: the downstream multiplier

Let us put a number on the multiplier. Take a single leading-edge wafer that TSMC sells for, say, \$18,000 (a 3nm wafer is roughly in this range). That one wafer yields hundreds of high-end application processors — suppose 400 good chips after yield. Each of those chips goes into a flagship smartphone with an average selling price of \$900.

```
Value of one wafer to TSMC          = 18,000
Chips per wafer (after yield)       = 400
Devices gated by the wafer          = 400 phones
Downstream device value             = 400 x 900 = 360,000

Downstream multiplier = 360,000 / 18,000 = 20x
```

So **every \$1 of leading-edge wafer revenue gates roughly \$20 of downstream device revenue** in this stylized case — and that is before you count the software, services, and ecosystem revenue those devices unlock over their lives. For AI accelerators the multiplier is different but the logic is the same: a relatively small dollar value of silicon gates an enormous dollar value of cloud and AI infrastructure. The core point: you cannot measure the importance of the chip chokepoint by the size of the chip industry (a few hundred billion dollars of revenue); you have to measure it by the size of everything the chips gate, which is most of the modern economy.

This is why credible estimates of a *complete* TSMC stoppage are so large. Industry and government analyses have floated figures ranging from a deep global recession to **\$1 trillion or more per year in lost output**, with some war-game estimates of a full cross-strait conflict reaching into the low *trillions* once you include the wider disruption to trade, shipping, and financial markets. The exact number is unknowable and the assumptions matter enormously. The point is the order of magnitude: this is not a sector risk. It is a macro risk dressed up as a sector risk.

## The scenarios: a blockade, a quarantine, and an invasion are not the same trade

A serious investor does not price "Taiwan risk" as one binary event. Disruption could take several forms, and they have radically different effects on chip flows and therefore on markets. Collapsing them into one number is the most common analytical error in this whole topic.

![Scenario tree branching cross-strait disruption into quarantine blockade and invasion with different market impacts](/imgs/blogs/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth-6.png)

**Quarantine.** The mildest scenario: China declares some form of customs or inspection regime around the island — stopping and inspecting ships, demanding paperwork — without a full military encirclement. Chips would still flow, but slowly and at higher cost: shipping delays, surging marine insurance premiums, rerouting. Markets would reprice the *friction* — higher costs, lower near-term volumes — but not a full stoppage. This is the scenario closest to the periodic scares we have already lived through.

**Blockade.** A more severe scenario: a naval and air encirclement that chokes off Taiwan's exports. Critically, the fabs themselves could keep running — they have power and inputs for a while — but the *finished chips cannot ship out*, and the inputs (chemicals, wafers, gases) cannot ship in. The world has only weeks of buffer inventory of leading-edge chips in the pipeline. After that buffer drains, the device shortage begins in earnest. A blockade is the scenario that turns the silicon-shield logic on its head: it weaponizes the chokepoint without destroying it.

**Invasion.** The most severe and least likely scenario: a kinetic conflict in which the fabs go offline — through damage, loss of power and skilled staff, or destruction. EUV fabs are extraordinarily fragile; they require ultra-stable power, ultra-pure water and chemicals, and thousands of irreplaceable engineers. A fab does not need to be bombed to stop — cut its power or its people and it halts, and restarting a contaminated leading-edge line can take many months. This is the scenario behind the trillion-dollar estimates: a multi-year loss of the world's leading-edge capacity, a global recession, and a scramble to rebuild that takes the better part of a decade.

The market does not know which of these will happen, or when, or with what probability. So it prices a **probability-weighted blend** of all three (and of the most likely outcome, which is that none happens and the status quo holds). When a headline hits, the blend shifts — the market marks up the probability of the bad branches — and chip stocks move. When the headline fades, the blend reverts. That is the mechanism behind the "beta to strait headlines" we will quantify in the markets section.

The blockade branch deserves a closer look, because it is the one investors most often misprice. It is more probable than a full invasion (it is a step down the escalation ladder, not the top of it) and far more economically devastating than a quarantine, which puts it in an awkward middle of the distribution that the simple "will there be a war or not?" framing misses entirely. A blockade is so dangerous precisely *because* it does not destroy anything. The fabs keep running; the chips keep getting made; they simply cannot leave the island, and the inputs cannot reach it. The world's buffer of finished leading-edge inventory — the chips already in transit, in warehouses, and inside half-built devices — is measured in weeks, not months, for the most advanced parts. Once that buffer drains, you get the cascade without a single fab being touched. And a blockade can be calibrated, extended, tightened, or relaxed as a coercive instrument in a way an invasion cannot, which is exactly why it weaponizes the chokepoint rather than eliminating it. An investor who models only "peace versus war" will be blindsided by the scenario that lives in between.

There is also a subtle point about *who* the disruption hurts, which complicates the silicon-shield story. China itself is a massive consumer of Taiwanese chips — its own electronics and assembly industries depend on them — so a disruption would inflict severe self-harm, which is part of the deterrent. But the calculus is asymmetric across scenarios and across actors: the United States, Europe, and Japan would suffer enormously from a leading-edge cutoff, while a country pursuing a strategic objective might judge the economic cost acceptable relative to the goal. Mutual economic harm deters, but it does not guarantee; the history of the twentieth century is full of belligerents who went to war against their own obvious economic interest. The honest position is that the scenario probabilities are genuinely uncertain and contested, that the silicon shield meaningfully lowers but does not zero the worst branches, and that an investor's job is to size for the distribution rather than to bet the firm on a point estimate.

## The diversification response: real, expensive, and aimed at the wrong node first

Governments understood the concentration risk after the 2021 auto shortage and the pandemic's supply shocks, and they responded with the largest industrial-policy push in a generation. The US **CHIPS and Science Act of 2022** authorized roughly \$52 billion in subsidies and a 25% investment tax credit to build fabs on American soil. The EU passed its own **European Chips Act**. Japan poured subsidies into bringing TSMC to Kumamoto. The map is being redrawn — slowly, and at enormous cost.

But here is the analytical key, and it is the most common misunderstanding among investors who think the risk is "being solved":

![Before and after comparison showing leading edge concentrated in Taiwan today versus partial diversification of trailing edge by 2030](/imgs/blogs/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth-7.png)

**The diversification moves the trailing edge first, and the leading edge stays in Taiwan for years.** TSMC's Arizona fabs are real and are ramping — starting at 4nm-class nodes and moving toward 2nm later this decade — but cutting-edge research, the newest nodes, and the *bulk* of leading-edge volume remain in Taiwan into the 2030s. TSMC's Japan fab targets mature 12-28nm logic (great for autos and industrial, not for flagship AI). Germany's planned fab targets automotive and mature nodes. Intel's foundry ambitions and Samsung's US fabs add capacity but are not yet yield-competitive at the very leading edge for the highest-volume customers.

So the response reduces the *trailing-edge* tail risk meaningfully — the kind of shortage that idled auto plants in 2021 gets less likely as mature-node capacity spreads to Arizona, Japan, Germany, and elsewhere. But the *leading-edge* single point of failure — the one under Apple, Nvidia, and the AI boom — persists. If you are pricing the tail, this is the distinction that matters: the easy part is being diversified; the hard part is not.

#### Worked example: the cost disadvantage of a US fab and what closes it

Why does leading-edge manufacturing not just move? Cost. Building and running a fab in the US is far more expensive than in Taiwan — TSMC's own founder has cited figures suggesting US chip manufacturing costs run perhaps 50% higher, and independent estimates of the *all-in* cost gap over a fab's life often land around 30-40%. Let us size what a subsidy has to close.

Suppose a leading-edge wafer costs \$18,000 to produce in Taiwan, and the same wafer costs 35% more in Arizona:

```
Taiwan wafer cost     = 18,000
Arizona cost premium  = 35%
Arizona wafer cost    = 18,000 x 1.35 = 24,300
Cost disadvantage     = 24,300 - 18,000 = 6,300 per wafer
```

Now suppose the Arizona fab runs 30,000 wafers a month at the leading edge:

```
Wafers per year       = 30,000 x 12 = 360,000
Annual cost gap        = 360,000 x 6,300 = 2,268,000,000 = ~$2.3 billion / year
```

So this single fab carries a **~\$2.3 billion annual cost disadvantage** versus making the same wafers in Taiwan. A one-time CHIPS subsidy of, say, \$6.6 billion (the kind of figure TSMC received for its Arizona project) covers only about *three years* of that gap. What closes the rest? Higher selling prices (customers paying a "made-in-America resilience premium"), continued operating subsidies, yield improvements as the fab matures, and — bluntly — customers and governments deciding the geopolitical insurance is worth the cost. The core point: subsidies can *seed* domestic leading-edge fabs, but they do not erase the structural cost gap, which is exactly why the market keeps the bulk of the leading edge in Taiwan even as new fabs open elsewhere.

## The policy layer: how law turned the chokepoint into a weapon

This is a law-and-geopolitics series, so we have to be precise about the *legal* machinery sitting on top of the physical supply chain — because over the past few years, governments discovered that the chokepoint is not only a vulnerability to defend but a lever to *use*. That discovery is itself a market force.

The key insight is that the chokepoints we mapped — US design software, ASML's EUV monopoly, TSMC's leading edge — are also *control points*. Whoever sits at a chokepoint can decide who is allowed through it. Starting in 2022, the United States used exactly this logic to restrict China's access to advanced chips and the tools to make them, through a body of export-control law. The mechanism that makes it bite is the **Foreign Direct Product Rule (FDPR)**: a US rule that claims jurisdiction over a foreign-made product *if* that product was made using US technology, software, or equipment. Because essentially every leading-edge chip on earth is made with US design software and US-origin equipment components, the FDPR lets US law reach a chip made by a Taiwanese company for a Chinese customer — even though no American company touched the transaction directly. The chokepoint becomes a legal pressure point. (The full mechanics — the Entity List, the FDPR, the October 2022 and 2023 controls — are the subject of [the export controls and chip war post](/blog/trading/law-and-geopolitics/export-controls-and-the-chip-war); here we care about how that policy layer changes the *risk* around Taiwan.)

Why does this matter for the Taiwan tail? Three reasons, each of which an investor should hold in mind.

First, **export controls raise cross-strait tension**, which raises the probability of the bad scenario branches. Restricting a country's access to the chips that power its economy and military is, by design, a coercive act; it tightens the strategic competition that the Taiwan tail risk sits inside. A serious analyst does not treat the export-control track and the conflict track as separate — they are two faces of the same contest over who controls the most important technology on earth.

Second, **the controls reshape the supply map** in ways that change who wins and loses. Cut off from leading-edge tools, China pours national resources into building a domestic supply chain — its own SMIC capacity at mature and near-leading nodes, its own equipment makers, its own design ecosystem. That accelerates a *bifurcation* of the global chip supply chain into two partially separate systems, which over a decade reduces the world's single dependence on Taiwan at the cost of efficiency and higher prices everywhere. Bifurcation is, in a strange way, a slow-motion partial answer to the concentration risk — and a tailwind for equipment makers and foundries on both sides of the divide who get to sell into a duplicated supply chain.

Third, **the silicon shield is increasingly a deliberate policy, not just an accident of economics.** Governments now understand that Taiwan's chip indispensability is a strategic asset, and they shape policy around it — some arguing for keeping the leading edge concentrated (to preserve the deterrent), others for diversifying it (to reduce the vulnerability). The tension between those two goals — deterrence wants concentration, resilience wants dispersion — is unresolved, and it is part of why the diversification response is slower and more conflicted than a pure resilience logic would produce. An investor reading the policy track should watch for which logic is winning at the margin, because it tells you whether the leading edge is being deliberately kept in Taiwan or deliberately moved out.

The bottom line of the policy layer is that the rule-change channel of this series is fully live here: export-control *law* changes the expected path of who can make and buy advanced chips, markets discount that into the relative valuations of TSMC, SMIC, ASML, Nvidia, and the equipment complex, and every tightening of the controls is a discrete, datable event that the chip complex reprices around — exactly the kind of catalyst the regulatory-calendar discipline is built to track.

## How it shows up in real markets

### The market prices the tail in spikes, not steadily

If the Taiwan tail were continuously priced, you would see a persistent risk premium discounting TSMC and its customers all the time. Instead, you see something closer to a dormant volcano: long periods of near-indifference, punctuated by sharp spikes when a headline hits.

![Line chart of the geopolitical risk index spiking around major events and reverting to baseline](/imgs/blogs/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth-8.png)

The Caldara-Iacoviello Geopolitical Risk index captures the pattern for geopolitical risk broadly: it sits near its long-run baseline (indexed to about 100) most of the time and spikes only around acute events — the 9/11 attacks (over 500), the 2022 Russian invasion of Ukraine (~277), the October 2023 Middle East escalation (~235) — before fading back. Cross-strait scares behave the same way. The market does not carry the tail risk as a steady haircut; it repays it in episodic fear and then forgets.

This is rational *and* dangerous. Rational, because the expected value of a low-probability event is low, so pricing it lightly day to day is correct on average. Dangerous, because "low probability" is not "zero probability," and a portfolio that is comfortable precisely because the risk is dormant is a portfolio that has not been stress-tested for the day the volcano wakes up.

#### Worked example: the event-study reaction of a chip basket to a strait headline

Quantify the spike. Take the August 2022 escalation around the Taipei visit. Suppose a chip-heavy basket (a semiconductor index) was trading at 100 the day before the headlines, and over the next three trading sessions, against a market (S&P 500) that was roughly flat, it fell to 95.5 before recovering to 99 within two weeks.

```
Pre-event basket level   = 100
Trough during the scare  = 95.5
Peak drawdown            = (95.5 - 100) / 100 = -4.5%

Market move (control)    = ~0% (flat)
Abnormal return          = -4.5% - 0% = -4.5%

Recovery to             = 99 within ~2 weeks
Residual permanent move  = (99 - 100) / 100 = -1%
```

So the headline produced about a **-4.5% abnormal drawdown** in the chip basket that mostly reversed, leaving a small residual. That is the signature of a geopolitical-headline shock: a sharp, basket-wide repricing of the tail probability, followed by a fade as the status quo reasserts and the probability is marked back down. Put plainly: chip stocks have a real, measurable *beta to strait headlines* — they sell off together on escalation and recover together on de-escalation — but historically the moves have faded, which is exactly what trains investors to fade the next one (and exactly the behavior that would be most painful if a scare ever did not fade).

### The concentration premium and discount

There is a second, subtler way the risk shows up: in *relative* valuations. The market simultaneously rewards and penalizes concentration. TSMC itself often trades at a *discount* to where a company with its dominance and margins would trade if it were located somewhere uncontested — a "Taiwan discount" reflecting the geopolitical overhang. Meanwhile, companies seen as *beneficiaries* of diversification — equipment makers with broad geographic footprints, mature-node foundries outside Taiwan, and firms levered to the reshoring capital-expenditure wave — can earn a relative premium.

#### Worked example: backing out the strait probability the market is pricing

You can flip the logic around and ask: *what probability of a severe disruption is the market implicitly pricing into a chip name's discount?* Take a leading foundry that, on pure fundamentals (growth, margins, dominance), "should" trade at a price-to-earnings multiple of 28x, but actually trades at 22x. Assume the only reason for the gap is the geopolitical overhang, and that in a severe-disruption scenario the equity is worth ~20% of its current value (fabs offline, multi-year rebuild), while in the no-disruption case it is worth the full 28x multiple.

Let `p` be the market's implied probability of the severe scenario over the relevant horizon. The current price is the probability-weighted blend of the two outcomes:

```
Fair multiple (no disruption)   = 28
Multiple in severe disruption   = 28 x 0.20 = 5.6
Observed multiple               = 22

22 = (1 - p) x 28 + p x 5.6
22 = 28 - 28p + 5.6p
22 = 28 - 22.4p
22.4p = 6
p = 6 / 22.4 = 0.268 = ~27%
```

So this stylized discount implies the market is pricing roughly a **27% chance** of a severe disruption over the horizon embedded in the multiple. Is that too high or too low? That is the actual investment question, and reasonable people disagree violently about it. The point of the exercise is that *the discount is not a vague mood — it is a probability you can back out and then argue with.* If you think the real probability is lower than what the discount implies, the name is cheap; if higher, it is expensive even at 22x. Put plainly: a "Taiwan discount" is the market quoting you odds on the strait, and your edge is having a better-calibrated view of those odds than the consensus — or, more honestly, recognizing when nobody can calibrate them and sizing the position so you survive being wrong.

### Defense and chip equities move together on escalation

On an escalation headline, you often see a recognizable cross-asset reaction map: chip stocks and broad equities sell off, safe havens (Treasuries, gold, the dollar, the yen) catch a bid, and *defense* stocks rally. The iShares US Aerospace & Defense complex, for instance, ran up strongly through the period after the 2022 Ukraine invasion as defense-spending expectations rose. A cross-strait escalation tends to trigger the same reflex — long defense, long safe havens, short the most strait-exposed chip names — as a fast, mechanical first-order reaction, regardless of how the scenario ultimately resolves.

### The AI boom is concentrating the risk, not diluting it

There is a powerful second-order trend that an investor must weigh: the AI boom is making the leading-edge chokepoint *more* important, not less. The entire wave of corporate capital expenditure on AI — the data centers, the accelerators, the trillion-dollar infrastructure buildout — rests on the most advanced chips, which are made almost exclusively on TSMC's leading edge. As the world's most valuable computing migrates to the frontier node faster than diversification can spread that node to new geographies, the share of global economic value sitting on top of a single strait is *rising*. This is the uncomfortable arithmetic behind the invalidation conditions in the playbook: the diversification response is a slow tailwind reducing the tail, while AI demand is a fast headwind concentrating it, and over the past few years the headwind has been winning. The result is that the dollar value exposed to a leading-edge disruption today is larger than it was when the CHIPS Act passed, even as more fabs break ground elsewhere.

### Why a chip-complex tail would not stay contained

The first-order reaction map — chips down, defense up, havens bid — understates the danger, because a real leading-edge disruption would not stay a sector event. The cascade figure showed why: missing chips idle factories and freeze technology capital expenditure across the economy. That feeds into the broad equity market (technology is the largest sector weight in most major indices), into inflation (a global goods shortage is inflationary just as a 2021-style chip shortage was), and into the kind of cross-asset correlation breakdown where the usual diversifiers stop diversifying. A portfolio that is "hedged" because its bonds rally when stocks fall may discover that in a genuine chip-complex shock — which is simultaneously a growth shock and an inflation shock — the historical stock-bond relationship does not hold. That is the deeper reason to treat this as a macro tail rather than a sector bet, and it connects directly to the cross-asset behavior covered in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).

## Common misconceptions

### Misconception 1: "Chip production can be quickly reshored"

The belief that subsidies will "bring the chips home" in a few years and dissolve the risk. The reality is on the timescale, the node, and the cost. A leading-edge fab takes roughly **three to five years** to build and ramp to volume, and *competitive yield* on the very newest node takes longer still. As the diversification figure showed, the new fabs in Arizona, Japan, and Germany are heavily weighted toward *mature and trailing-edge* nodes, with leading-edge volume staying in Taiwan into the 2030s. And the worked example put a number on why: a single leading-edge US fab can carry a **~\$2.3 billion-per-year cost disadvantage**, which a one-time subsidy covers for only a few years. Reshoring is real, slow, expensive, and aimed at the easier node first. It reduces the trailing-edge tail; it does not erase the leading-edge one on any near-term horizon.

### Misconception 2: "Only Taiwan matters"

The belief that the entire risk is geographic — that if you watch the strait, you have watched the chokepoint. But the chain is a *chain of chokepoints*. EUV lithography is a literal monopoly held by **one company (ASML) in the Netherlands**; without those scanners, no leading-edge node runs anywhere, including in Taiwan. Specialty materials concentrate elsewhere — Japan makes a large share of the world's photoresist and polished silicon wafers, and a meaningful share of the neon gas critical to chipmaking historically came from Ukraine, which is why the 2022 invasion sent neon prices up several-fold and rattled chip buyers. A complete map of the tail flags at least three nodes — leading-edge fabbing (Taiwan), EUV equipment (the Netherlands), and specialty materials (Japan, and formerly Ukraine) — not one.

### Misconception 3: "The market has already priced the Taiwan risk"

The belief that, because everyone knows about the risk, it must be in the price. But "known" and "priced" are different things, and the GPR chart shows why: geopolitical tail risk is priced **episodically, in spikes, not as a steady premium.** The event-study example quantified a typical scare as a roughly **-4.5% abnormal drawdown that mostly reverses**, leaving little permanent discount. A risk that is repriced sharply on a headline and then faded back to near-baseline is, almost by definition, *not* continuously priced — it is priced only when fear is acute and forgotten the rest of the time. The catastrophic branch — a multi-year loss of leading-edge capacity and a multi-trillion-dollar GDP hit — is essentially *not* in everyday chip multiples, because if it were, those multiples would be far lower than they are.

### Misconception 4: "The silicon shield guarantees nothing happens"

The belief that the economic value of TSMC is itself a sufficient deterrent — that no actor would risk destroying so much value. The silicon shield is plausibly *part* of the deterrence math, but treating it as a guarantee confuses economic logic with strategic logic. A blockade scenario, in particular, *weaponizes* the chokepoint without destroying it — it uses the world's dependence as leverage rather than a deterrent. History is full of episodes where strategic objectives overrode economic self-interest. An investor should treat the silicon shield as a *factor that lowers the probability* of the worst branch, not as a wall that sets it to zero. Pricing the tail at zero because of an elegant deterrence argument is exactly the error that fat-tail risks punish.

## How to trade it: the playbook

The point of this whole series is to land on *so how do you read or trade this?* Here is the framework. Two principles run through all of it. First, **map the chokepoints to your exposure** — know precisely which of your holdings have leading-edge Taiwan single-point-of-failure risk versus reroutable dependency. Second, **treat strait risk as a fat tail to hedge, not a catalyst to time** — you will be wrong about timing, and most scares fade, so the trade is in sizing and insurance, not prediction.

### Step 1: Map the chokepoints to your portfolio

Go through your holdings and classify each by its *kind* of chip exposure, not just its presence:

- **Leading-edge single-point-of-failure**: companies whose most profitable products are made only on TSMC's leading edge (flagship-chip designers, AI-accelerator makers, premium-phone makers). As the first worked example showed, this can be ~90% of a name's revenue. This is the core tail exposure.
- **Reroutable dependency**: companies using mature-node chips that could be sourced elsewhere with effort (industrials, autos for many parts, appliances). Real but survivable.
- **Upstream chokepoint**: the equipment monopoly (ASML) and specialty-materials names — exposed to the *whole-industry* tail but not to Taiwan geography specifically, and beneficiaries of any reshoring capex.
- **Diversification beneficiaries**: equipment makers riding the reshoring capex wave, mature-node foundries outside Taiwan, and the broad industrial supply chain of new-fab construction. These can *gain* from the response to the risk.

### Step 2: Size the tail, do not time it

Because the tail is fat and the timing is unknowable, the right tool is *sizing and hedging*, not a directional bet. Decide how much of a catastrophic chip-complex drawdown your portfolio can absorb, and buy insurance for the rest.

#### Worked example: sizing a tail hedge against a strait scenario

Suppose you run a \$100 million portfolio with \$30 million in chip-complex equities (designers, foundries, equipment) that you judge would fall ~50% in a severe blockade/invasion scenario.

```
Chip-complex exposure        = 30,000,000
Assumed loss in a severe tail = 50%
Tail loss on that sleeve     = 30,000,000 x 0.50 = 15,000,000
Tail loss as % of portfolio  = 15,000,000 / 100,000,000 = 15%
```

Now price the hedge. Long-dated out-of-the-money put options (or put spreads) on a semiconductor index might cost, say, 2% of notional per year for the protected portion. To hedge \$30 million of exposure:

```
Hedge notional               = 30,000,000
Annual hedge cost (2%)        = 30,000,000 x 0.02 = 600,000
Hedge cost as % of portfolio  = 600,000 / 100,000,000 = 0.6% / year
```

So you can cap a ~15%-of-portfolio tail loss for roughly **0.6% per year** in option premium. Is that worth it? That is a judgment about how much you fear the tail and how much carry you can afford. Put plainly: you are not predicting a conflict; you are deciding whether spending 60 basis points a year to convert an unhedgeable 15% drawdown into a known, budgeted cost is good portfolio engineering. For many investors, sizing the chip sleeve smaller is cheaper than hedging it — but if you want the upside of the chip complex *and* protection against the tail, this is roughly what the insurance costs. (For the mechanics of how options price a binary-ish event and how to construct the put spread, see the cross-links below.)

### Step 3: Read the signals and know the invalidation

The signals to watch are the ones that shift the probability of the bad branches:

- **Cross-strait escalation indicators**: military exercise tempo, air-defense-zone incursions, official rhetoric, and the cadence of the political calendar (elections in Taiwan, leadership transitions in Beijing). These move the *blockade/invasion* probability and therefore the headline beta.
- **The export-control and policy track**: tighter US controls on chip equipment and advanced chips going to China raise tension and shift incentives — covered in depth in [the export controls and chip war post](/blog/trading/law-and-geopolitics/export-controls-and-the-chip-war).
- **Diversification milestones**: every leading-edge fab that ramps to volume *outside* Taiwan marginally reduces the tail — watch Arizona's yield reports, not its groundbreaking ceremonies.
- **The insurance/freight market**: marine insurance premiums and shipping rates around the strait are a real-money signal of how professionals price the friction scenario.

A word on discipline, because it is the hardest part. The base rate of cross-strait scares *resolving without disruption* is, historically, very high — every scare so far has faded. That base rate is a genuine guide and you should respect it: most of the time, the right response to a strait headline is to do nothing, or to fade the panic that less-informed money is selling into. The danger is letting a high base rate of nothing-happening collapse into a belief that nothing *can* happen. A risk that is 95% likely to be fine in any given year is still a risk you want insurance against, because the 5% is not a small loss — it is a portfolio-altering one, and 5%-per-year compounds into a meaningful probability over a holding period of a decade. The professional's stance is the union of two ideas that feel contradictory but are not: fade the headlines *and* carry the tail hedge. You fade the noise because the base rate says most scares are noise; you carry the hedge because the one that is not noise is the one that ends careers.

**What invalidates the thesis?** The tail thesis weakens materially if (a) leading-edge volume genuinely diversifies — if, by the late 2020s, a second geography is making a large, yield-competitive share of the leading edge, the single-point-of-failure premium should compress; (b) the cross-strait probability structurally falls (a durable de-escalation or settlement), which would let the dormant risk fade further; or (c) a technology shift — chiplets, advanced packaging, or a node-scaling wall — reduces how much the world depends on the single most advanced node. Conversely, the thesis *strengthens* if AI demand keeps concentrating the world's most valuable computing on the leading edge faster than capacity diversifies, which has been the recent trend. Size accordingly, and revisit when any of those invalidation conditions actually moves — not when the next headline hits.

The deepest lesson is the one we opened with. The most important supply chain on earth runs through one island, one company, and one node, and the market prices that fact in spikes and forgets it in between. The investor's edge is not in predicting the strait. It is in knowing the chokepoints precisely enough to know exactly what reprices if it is ever disrupted, sizing that tail honestly rather than rounding it to zero, and being the one person in the room who has already decided what they will do on the day the volcano wakes up.

## Further reading & cross-links

- [Export controls and the chip war](/blog/trading/law-and-geopolitics/export-controls-and-the-chip-war) — the policy layer on top of this supply chain: the Entity List, the Foreign Direct Product Rule, and how export law reshaped the semiconductor map.
- [Supply-chain geopolitics: reshoring, friendshoring, and critical minerals](/blog/trading/law-and-geopolitics/supply-chain-geopolitics-reshoring-friendshoring-and-critical-minerals) — the broader unbundling of global supply chains, of which the chip chokepoint is the sharpest case.
- [Scenario analysis and war-gaming geopolitical events](/blog/trading/law-and-geopolitics/scenario-analysis-and-war-gaming-geopolitical-events) — the toolkit for turning a scenario tree like the one here into probability-weighted positions.
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the spine of this whole series: how a rule or geopolitical shock prices into assets.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — why a chip-complex tail event would not stay contained: the cross-asset mechanics of a crisis.
- [Economic moats: durable competitive advantage](/blog/trading/equity-research/economic-moats-durable-competitive-advantage) — TSMC's winner-take-most flywheel as a moat, and how concentration cuts both ways.
- [Risk, the pre-mortem, and being wrong well](/blog/trading/equity-research/risk-the-pre-mortem-and-being-wrong-well) — the discipline of sizing a fat tail you cannot time, applied to any concentrated risk.
