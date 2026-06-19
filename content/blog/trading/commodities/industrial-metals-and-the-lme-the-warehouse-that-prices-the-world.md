---
title: "Industrial Metals and the LME: The Warehouse That Prices the World"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How the London Metal Exchange prices physical copper, aluminium, zinc, lead, nickel and tin through a global network of approved warehouses, the warrant, and the famous 3-month forward date — and how a trader reads the cash-to-3-month spread for tightness."
tags: ["commodities", "industrial-metals", "lme", "copper", "aluminium", "nickel", "warehouse", "warrant", "backwardation", "contango", "comex", "shfe"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The London Metal Exchange does not price an abstract number; it prices a *warrant*, a title document to a specific lot of real metal sitting in a specific approved warehouse. That physical anchor is what makes the LME unlike any other exchange, and it is why a metals trader reads warehouse stocks and the cash-to-three-month spread before reading the headline price.
>
> - The base-metals complex — **copper, aluminium, zinc, lead, nickel, tin** — are **consumption / industrial inputs**, not monetary stores of value. Their prices rise and fall with the **global growth cycle**, which is the opposite of how monetary gold behaves.
> - The LME's quirks all trace to physical delivery: the **warrant** (paper title to a named lot of metal), the global **warehouse network**, the open-outcry **ring**, and the **3-month forward date** — a relic of the 1800s when copper from Chile and tin from Malaya took roughly three months to sail to London.
> - The single most useful reading is the **cash-to-3-month spread**: cash *above* three-month means the prompt market is tight (**backwardation**); cash *below* three-month means metal is plentiful (**contango**).
> - The one number to remember: on 8 March 2022, LME nickel printed an intraday **\$101,365/tonne** before the exchange cancelled trades — the day the physical-delivery system everyone trusted simply broke.

In May 2024, the price of copper on the London Metal Exchange touched roughly **\$11,100 a tonne** — an all-time record. The newspapers reached for the obvious story: a clean-energy boom, electric cars and grids and data centres all screaming for copper wire, the metal of electrification. That story is broadly true. But a metals trader watching the screen that month was not looking at the headline price at all. She was looking at two other things: how much copper sat warranted in LME warehouses (not much), and whether cash copper was trading *above* the three-month price (it was, in spikes). Those two numbers told her something the headline could not — that the squeeze was *physical*, that someone, somewhere, could not get their hands on actual metal, and that the warehouses had run thin.

That is the whole secret of the London Metal Exchange. Almost every other market you have met prices a promise. A stock is a claim on future profits. A bond is a stream of future coupons. A stock-index future is a bet on a number. The LME is different in a way that sounds almost old-fashioned: it prices a *thing you can touch*. Behind every LME copper contract is a 25-tonne lot of physical cathode, sitting in a numbered shed in Rotterdam or Busan or Johor, and behind the price is a piece of paper — the **warrant** — that says *you own that exact lot*. The price the world quotes for copper is, at bottom, the price of those warrants changing hands.

This post is about the base-metals complex and the strange, brilliant, occasionally scandalous market that prices it. We will build the LME from nothing: the warehouse, the warrant, the ring, and the famous three-month date. We will see why industrial metals dance to the growth cycle while gold marches to a different drum. We will walk through the 2010s warehouse-queue scandal that turned a storage system into a money machine, and the 2022 nickel day when the physical anchor snapped. And we will end where a trader begins: with the spreads and the stocks that tell you, before any news does, whether metal is scarce or plentiful right now.

![The LME warehouse to warrant to exchange to delivery flow, the four-box pricing system](/imgs/blogs/industrial-metals-and-the-lme-the-warehouse-that-prices-the-world-1.png)

## Foundations: what an industrial metal is, and what the LME actually prices

Let us start with no assumptions at all.

A **commodity** is a physical good that is interchangeable lot-for-lot: one tonne of copper cathode of a given grade is, by agreement, as good as any other tonne of that grade, so they can trade against a single price. (We built this idea up in the series opener, [what is a commodity](/blog/trading/commodities/what-is-a-commodity-the-physical-asset-that-trades-on-paper).) Within commodities, the **base** or **industrial metals** are the workhorses of the physical economy: copper for wire and pipe, aluminium for cans and aircraft and window frames, zinc for galvanising steel against rust, lead for batteries, nickel for stainless steel and now for battery cathodes, and tin for solder. They are called *base* to contrast them with the *precious* metals — gold, silver, platinum, palladium — and the distinction is not snobbery. It is economics.

A base metal is something you *consume*. It gets dug out of the ground, smelted, fabricated into a product, and then it is gone — bound up in a building, a car, a transformer. Demand for it rises when the world is building things and falls when the world stops. That makes the base metals **pro-cyclical**: their prices are a live read on global industrial activity, which is exactly why copper earned the nickname "Doctor Copper," the metal with a PhD in economics. (We give copper its own full treatment in [Doctor Copper](/blog/trading/commodities/copper-doctor-copper-and-the-pulse-of-the-global-economy); here we treat the whole complex and the market that prices it.)

A **monetary metal** behaves in the opposite way. Gold is barely *consumed* at all — almost every ounce ever mined still exists, sitting in vaults and jewellery, and people hold it precisely *because* it does nothing. Its price is driven by real interest rates, the dollar, and fear, not by how many factories are running. That contrast — industrial versus monetary — is the spine of this whole series, and gold is important enough to get its own series. When you want the monetary side of the story, the place to start is [is gold money, a commodity, or a currency](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything). For our purposes the one-line version is: *base metals are a bet on growth; gold is a bet against it.*

It helps to meet the complex as a family, because each member has a different job and therefore a different demand driver, even though they all trade on the same exchange and obey the same warehouse-and-warrant plumbing.

- **Copper** is the conductor: power cables, motor windings, pipes, the wiring of every building and electric car. It is the most economically sensitive of all the metals, and the one whose price the whole world reads as a growth gauge.
- **Aluminium** is the light metal: drink cans, car bodies, aircraft, window frames, transmission lines. It is enormously *energy-intensive* to produce — smelting aluminium is essentially "congealed electricity" — so its price is hostage to power costs as much as to demand, which is why the 2022 European energy crisis sent it spiking.
- **Zinc** is the rust-proofer: most of it goes to *galvanising* steel, coating it so bridges and car panels do not corrode. Its demand is therefore a derivative of construction and autos.
- **Lead** is the battery metal of the old world: the lead-acid battery that starts your petrol car. It is one of the most-recycled metals on earth, so scrap supply matters as much as mine supply.
- **Nickel** wears two hats: the old job is stainless steel (corrosion-resistant cutlery, sinks, chemical plant), and the new job is the cathode of lithium-ion batteries. That second job is why nickel became a battleground for the energy transition — and the stage for 2022's break.
- **Tin** is the smallest and quietest of the six, but it is the metal of *solder* — the dabs that join every electronic circuit — so its demand quietly tracks the whole electronics industry. It is also the most concentrated and supply-fragile of the complex.

Lay their prices side by side over twenty-five years and the family resemblance is obvious: they boom together when the world builds and bust together when it stops.

![LME copper and aluminium annual average price 2000 to 2025 on a dual axis](/imgs/blogs/industrial-metals-and-the-lme-the-warehouse-that-prices-the-world-2.png)

The chart plots the two giants of the complex — copper and aluminium — on one timeline. Both surged through the **2000s China supercycle**, when a continent's worth of construction and grid-building pulled metal out of the ground faster than miners could supply it; copper ran from under \$2,000/tonne to nearly \$9,000 by 2011. Both then slid through the **2011–2016 China slowdown**. Both spiked again in **2021–2022** — copper on the electrification bid, aluminium with an extra kick from Europe's power crisis (the energy-cost story). And copper went on to a fresh **record near \$11,100/tonne in May 2024**. The synchrony is the point: these are six different metals with six different end-uses, yet they move as a *complex* because they share one master driver — the global industrial cycle. That shared heartbeat is what makes them tradable as a group and what separates them, as a class, from monetary gold. (For the allocator's version of this same contrast, see [metals, copper and silver, the economy's pulse](/blog/trading/cross-asset/metals-copper-silver-the-economys-pulse).)

Now, the exchange. The **London Metal Exchange** — the LME — is the world's center for trading these base metals. It was founded in 1877, in the heart of an empire that imported metal from every corner of the globe, and its peculiarities are fossils from that era that turned out to be features. Three of them matter, and all three flow from one fact: **the LME settles in physical metal.**

The first peculiarity is the **warehouse network**. The LME does not own metal and does not store it. Instead it *approves* warehouses — independent storage operators in dozens of locations worldwide (Rotterdam, Antwerp, Busan in South Korea, Johor in Malaysia, Singapore, the US Midwest, and more) — and only metal sitting in an LME-approved shed, of an LME-approved brand and shape, can be delivered against an LME contract. The exchange writes the rulebook; the metal lives in the sheds.

The second peculiarity is the **warrant**. When a lot of metal is placed in an approved warehouse and inspected, the warehouse issues a warrant: an electronic title document that says *this exact lot — say 25 tonnes of Grade-A copper cathode, brand X, in warehouse Y at place Z — belongs to the bearer of this warrant.* The warrant is the bridge between the physical and the paper. When you "buy copper" on the LME and stand for delivery, what you receive is a warrant; whoever holds the warrant can, in principle, drive to the shed and collect the metal.

The third peculiarity is the **date structure** — and especially the famous **3-month** contract — which is strange enough to need its own section below.

#### Worked example: the notional value of one copper lot

Let us make this concrete with the smallest unit the LME deals in. The standard LME copper contract is **25 tonnes**. Suppose copper is trading at **\$9,400/tonne** (its rough 2025 level).

The notional value of one lot is:

```
lot size      = 25 tonnes
copper price  = 9,400 USD/tonne
notional      = 25 x 9,400 = 235,000 USD
```

So a single LME copper contract controls **\$235,000** of metal. A trader who buys ten lots is long 250 tonnes — a quarter of a kilotonne — worth **\$2.35 million**. And crucially, if she holds to delivery, she is not holding a number on a screen; she is holding warrants entitling her to 250 tonnes of physical copper sitting in named warehouses. The intuition: the LME's contract size is small enough that the price stays tethered to real, deliverable, *touchable* metal — which is exactly the discipline that keeps the world's copper price honest.

## How the warrant turns metal into a price

Trace the flow once, slowly, because everything else hangs on it. (The cover figure above lays out these four boxes left to right.) A miner in Chile produces copper concentrate; a smelter refines it into cathode — flat sheets of nearly pure copper of a registered brand. That cathode is shipped to, say, an LME warehouse in Rotterdam. The warehouse weighs it, checks the brand and quality, stacks it, and issues a **warrant**: the electronic title to that specific lot.

Now the metal has become *fungible paper*. The warrant can be bought and sold without anyone touching the cathode. A trader in Singapore can buy that Rotterdam warrant at 2 a.m.; the copper never moves. The LME price is simply the price at which warrants — and the futures that promise to deliver warrants — change hands. When a contract reaches its delivery date and a buyer "takes delivery," the seller hands over a warrant, and *now* the buyer can choose: keep holding the warrant (and pay the warehouse rent), sell it on, or present it at the shed and cancel it to remove the physical metal from the system. Removing metal is called **cancelling a warrant** or putting it "on load-out," and the running total of cancelled-but-not-yet-removed warrants is a number traders watch like hawks, because it signals metal heading *out* of the visible system.

This is the genius and the fragility of the LME in one mechanism. The genius: a physical market with delivery points all over the world is welded to a single, liquid, global price, because the warrant makes location and brand into standardised, tradable title. The fragility: that same warrant can be *rented*, *queued*, and *gamed* — and when it is, the price of paper metal can drift away from the price of metal you can actually obtain. Hold that thought; it is the warehouse scandal, and we will get there.

The headline LME stock number — "LME copper inventories fell to X tonnes" — is simply the total metal currently on warrant in all approved warehouses. Falling stocks mean metal is being pulled out of the visible system (being consumed, or hoarded off-exchange); rising stocks mean metal is piling up unwanted. A trader reads LME stocks the way a doctor reads a blood-pressure cuff: not the absolute number so much as the *trend* and the *rate of change*.

## The 3-month date: a shipping relic that became the benchmark

Here is the single weirdest thing about the LME, and it is worth understanding because it confuses every newcomer. On most futures exchanges you trade *calendar months*: the June contract, the December contract. On the LME, the benchmark everyone quotes is the **3-month** — a date exactly three months from today that rolls forward every single day. Today's "3-month copper" delivers on a date three months out; tomorrow's "3-month copper" delivers one day later than today's did. It is a *rolling* prompt date, not a fixed month.

Why? Because of sailing ships. When the LME was founded in 1877, Britain imported copper from Chile and tin from the Malay peninsula, and those cargoes took roughly **three months** to arrive by sea. A merchant who agreed to sell metal "to arrive" wanted to fix a price for the day the ship docked — about three months out. So the market organised itself around a three-month forward, and even though metal now moves in days and the contract has long since become a financial instrument, the **3-month remains the LME's benchmark date** out of pure inertia. It is a 19th-century logistics fact frozen into 21st-century market plumbing.

Around that benchmark, the LME quotes a ladder of dates:

- **Cash** — settlement in two business days (the LME's version of "spot"; metal you can have almost now).
- **TOM** — "tomorrow," one business day out (a very short prompt date).
- **3-month** — the rolling benchmark, the most liquid point.
- **Longer dates** — out to 15 months for the daily-prompt structure, and to several years for monthly contracts.

The relationship between **cash** and **3-month** is the whole game, and it has a name we have met before in this series: **contango** versus **backwardation**. (We built the general theory in [contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) and [the forward curve](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities); here we put the LME's specific cash-3m spread under the microscope.)

![The LME date structure cash to 3-month and the backwardation versus contango spread](/imgs/blogs/industrial-metals-and-the-lme-the-warehouse-that-prices-the-world-3.png)

- When **cash trades *above* 3-month**, the curve slopes *down* from the prompt date: the market will pay a premium to get metal *now* rather than later. That is **backwardation**, and on the LME it is the unmistakable fingerprint of a *tight* prompt market — low warehouse stocks, a squeeze, real demand outrunning available metal.
- When **cash trades *below* 3-month**, the curve slopes *up*: metal now is cheaper than metal later, because the later price has to cover storage, financing, and rent. That is **contango**, the fingerprint of *ample* supply — full sheds, soft demand, metal happy to sit and wait.

The cash-3m spread is the LME trader's single most important live indicator. It is read in dollars per tonne, and it is quoted as a "backwardation of \$X" or a "contango of \$X." A wide backwardation that suddenly appears is a klaxon: something just got tight.

#### Worked example: reading a cash-3m spread and annualising the backwardation

Suppose LME copper quotes:

```
cash      = 9,520 USD/tonne
3-month   = 9,400 USD/tonne
spread    = cash - 3m = 9,520 - 9,400 = +120 USD/tonne  (backwardation)
```

Cash is **\$120 above** three-month, so the market is in backwardation: prompt copper is scarce. To judge *how* tight, annualise it. The \$120 is earned (by whoever is short the spread, i.e. lends prompt metal and buys it back later) over three months, so:

```
spread as % of price (3 months)  = 120 / 9,400        = 1.277%
annualised (x 4 quarters)        = 1.277% x 4          = 5.1% per year
```

A **5.1% annualised backwardation** is meaningful — it means anyone holding metal and willing to lend it into the prompt is being paid roughly 5% a year to do so, far more than a storage cost would justify. The intuition: a backwardation that annualises well above the cost of money is the market *bidding for physical metal right now*, and it usually precedes a stockdraw and a price spike.

#### Worked example: the same spread in contango

Now flip it. Suppose aluminium quotes:

```
cash      = 2,380 USD/tonne
3-month   = 2,440 USD/tonne
spread    = cash - 3m = 2,380 - 2,440 = -60 USD/tonne   (contango)
```

Cash is **\$60 below** three-month. Annualise the contango:

```
spread as % of price (3 months)  = 60 / 2,440          = 2.46%
annualised (x 4)                 = 2.46% x 4           = 9.8% per year
```

A near-10% annualised contango is a market *paying you to wait* — it more than covers the cost of storing and financing the metal for three months, which is the signal that warehouses are full and nobody needs the aluminium right now. The intuition: steep contango is a glut wearing a price tag, and it is exactly the condition that the warehouse-queue game (next) learned to feed on.

## Trading the spread: borrowing, lending, and the metal carry trade

Once you see the cash-3m spread as the LME's central signal, a second realisation follows: the spread is not just something you *read*, it is something you *trade*. LME professionals spend most of their day trading **spreads** — the price difference between two dates — rather than the outright flat price. The vocabulary is worth learning because it reveals how the physical and the paper are stitched together.

To **borrow** on the LME means to buy the near date and sell the far date — buy cash, sell 3-month. You do this when you need metal *now* (you are short of prompt metal and must cover) or when you expect the prompt to get tighter. To **lend** is the reverse: sell the near date and buy the far date — you have metal now that you do not need, so you deliver it prompt and buy it back later, earning the spread if the market is in contango. In a contango, the *lender* of metal is paid the carry; in a backwardation, the *borrower* pays a premium to pull metal forward in time. The cash-3m spread is therefore literally the price of *time* for physical metal — the rent the market charges to move a tonne of copper from "later" to "now."

This is the **carry trade** in its purest commodity form, and it connects directly to the convenience-yield theory we built earlier in the series ([convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape)). In a normal contango, an arbitrageur can in principle buy cheap cash metal, sell it forward at the dearer 3-month price, take delivery, pay the warehouse to store it for three months, and collect the difference risk-free — *provided* the contango is wider than the all-in cost of storage plus financing. That arbitrage is what *creates* the contango in the first place: it cannot get much steeper than the cost of carry, because if it did, traders would pour metal into storage to harvest the free spread, and the buying of cash plus selling of 3-month would compress the spread back to carry. Backwardation is the anomaly that arbitrage *cannot* fix, because you cannot borrow metal that does not exist — which is exactly why backwardation is such a clean tightness signal.

#### Worked example: the cash-and-carry arbitrage on copper

Suppose copper is in contango and the all-in carry (storage + financing + insurance) is **\$70/tonne over three months**. The market quotes:

```
cash price            = 9,000 USD/tonne
3-month price         = 9,110 USD/tonne
contango spread       = 110 USD/tonne
carry cost (3 months) = 70 USD/tonne
free profit per tonne = 110 - 70 = 40 USD/tonne
```

The arbitrage is alive: buy cash copper at \$9,000, sell it forward at \$9,110, store it for three months at \$70, and bank **\$40/tonne** with no price risk. On one 25-tonne lot that is 25 x \$40 = **\$1,000** locked in. The act of doing this — buying cash, selling 3-month — pushes cash *up* and 3-month *down*, narrowing the contango toward the \$70 carry until the free profit vanishes. The intuition: in contango, storage arbitrage caps how steep the curve can get, which is why a contango wider than carry is rare and fleeting — and why a *backwardation*, which no arbitrage can close, is the market shouting that prompt metal is genuinely scarce.

The spread is also where the **dominant-position risk** lives, and the LME watches it closely. If one player corners a large share of the available warrants *and* the cash positions, they can engineer an artificial backwardation — a "squeeze" or "tom-next squeeze" — by refusing to lend prompt metal, forcing shorts who must deliver to pay a punishing premium to borrow it. The LME's lending rules (which cap how much a dominant holder can charge to lend the prompt) exist precisely to keep this in check. The 2022 nickel episode was a squeeze of this family taken to a system-breaking extreme.

## The ring, the screen, and how an LME price is actually made

A word on the *venue*, because it is as eccentric as the date. For most of its life the LME's official prices were set by **open outcry in "the Ring"** — a small circle of leather benches in a London trading floor where a handful of dealers from the member firms shouted bids and offers at each other in frantic, ritualised five-minute sessions, one metal at a time. The Ring is the last open-outcry floor in Europe, and the prices struck in its sessions become the LME's **official settlement prices**, the references against which a vast web of physical contracts worldwide is invoiced. (The Ring went dark during the 2020 pandemic and the LME debated closing it permanently; after a member backlash it reopened, a rare case of a market choosing tradition over efficiency.)

Today most volume is electronic, traded around the clock on the LMEselect screen, plus an inter-office telephone market that runs 24 hours. But the structure is the same: members make markets in the cash-3m spread and the outright dates, and the warrant sits underneath it all as the deliverable. The point for a reader is not the romance of the floor but *what the floor produces*: a set of official cash and 3-month settlement prices, per metal, per day, that the entire physical metals trade — every smelter invoice, every cable-maker's hedge — references. When a Vietnamese steel mill or a German auto plant agrees a copper price, it agrees a *premium over the LME*, not a number from thin air.

## Who actually uses the LME, and why a price is a hedge

It is easy to picture the LME as a casino of speculators, but its deepest user base is the physical trade — the people who *make* and *use* metal and who need to lock in a price so a swing in the market does not wipe out their business. The exchange exists, fundamentally, to let them transfer that price risk to someone willing to bear it. There are four kinds of player, and the series treats them in full in [the four players](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators); here is how they meet on the LME.

A **producer** — a copper miner — fears the price *falling* between the day it digs the ore and the day it sells the cathode, months later. So it *sells* LME forward: it locks in today's price for metal it will deliver in three months. If the price falls, the loss on the physical copper is offset by the gain on the short futures; if it rises, the miner forgoes some upside but it has secured a known revenue and can plan. A **consumer** — a cable-maker or a carmaker — fears the opposite, the price *rising* before it buys the metal it needs, so it *buys* LME forward to lock in its input cost. Between them sit the **merchants** (the trading houses, who hedge the metal sitting in their warehouses and on their ships) and the **speculators** (funds and traders who take the price view that the hedgers want to offload). The speculator is not a parasite on this system; they are the counterparty that lets the miner and the cable-maker both sleep at night.

This is why the LME's official settlement price matters so far beyond the trading floor. A producer's hedge, a consumer's purchase contract, an insurer's valuation, a bank's loan against metal collateral — all of them are written as "the LME price plus or minus a premium." The exchange is the neutral reference the whole physical economy agrees to invoice against. And it is why a *broken* price — the 2022 nickel cancellation — is so damaging: the moment the reference becomes unreliable, every hedge and every contract built on it is thrown into doubt. The price is not a free-floating number above the metal trade; it is the load-bearing beam the whole structure hangs from.

There is a subtle but important consequence for how the curve behaves. Because producers are structural *sellers* of the forward and consumers are structural *buyers*, the natural hedging flow tilts the curve, and the balance of who needs to hedge more urgently shows up in the cash-3m spread. When producers rush to lock in a high price (selling the forward), they press the back of the curve down, deepening backwardation; when consumers panic-buy forward cover in a shortage, they bid the prompt, doing the same. So the spread is not only a stock-and-flow signal — it is also a *hedging-pressure* signal, a readout of which side of the physical trade is more frightened right now.

## Where the metal comes from: the supply side that sets the floor

A price is a tug-of-war between demand and supply, and on the supply side the base metals share a hard truth: **you cannot conjure a new mine quickly.** A major copper mine takes a decade-plus from discovery to first metal, and the best ore bodies are concentrated in a handful of countries. That geological concentration is why a strike in Chile or a permitting fight in Peru can move the global copper price within hours.

![Copper mine output by country in 2023, horizontal bar chart with the Andean and Congo core highlighted](/imgs/blogs/industrial-metals-and-the-lme-the-warehouse-that-prices-the-world-4.png)

The chart makes the concentration vivid. **Chile alone mines about 5 million tonnes** of contained copper a year — roughly a fifth of the world's mine supply — with **Peru** (~2.6 Mt) and the **Democratic Republic of the Congo** (~2.5 Mt) close behind. China mines a meaningful 1.7 Mt but consumes vastly more than it digs; the United States, Indonesia, Russia and Australia round out the top tier. The takeaway is structural: a few Andean and Central-African deposits, plus the smelting capacity to process their ore (much of which sits in China), set the supply floor for the metal that wires the planet. When a single country holds a fifth of supply, *its* politics, water rights, and ore grades become *everyone's* price risk.

This long, inflexible supply chain is why base-metals prices can stay elevated for years once demand outruns the mines — there is no quick valve to open — and why they can also crash hard when a building boom ends and all that newly-commissioned capacity meets a wall of weak demand. It is the classic commodity supercycle, and copper has lived through two big ones in living memory: the 2000s China boom and the 2020s electrification bid. For the allocator's framing of metals in a portfolio, see [metals, copper and silver, the economy's pulse](/blog/trading/cross-asset/metals-copper-silver-the-economys-pulse).

There is a second chokepoint hiding behind the mine map, and it is just as important: **smelting and refining.** Ore comes out of the ground as concentrate — perhaps 25–30% copper — and must be smelted and refined into the 99.99% cathode that the LME accepts. That conversion is a separate industry with its own geography, and it is even *more* concentrated than mining: China refines well over half the world's copper and a similar share of many other base metals. So the supply chain has two gates, not one — the mine and the smelter — and a problem at either can tighten the market. A smelter outage, a power shortage hitting an aluminium plant, or China restricting concentrate imports can all squeeze refined-metal supply even when ore is plentiful in the ground. The fee that smelters charge to turn concentrate into metal — the **treatment and refining charge (TC/RC)** — is itself a watched indicator: when TC/RCs collapse, it means too many smelters are chasing too little concentrate, a sign the *mine* side is tight; when they balloon, concentrate is abundant. A metals analyst reads the TC/RC the way an oil analyst reads the crack spread ([refining and crack spreads](/blog/trading/commodities/refining-and-crack-spreads-turning-crude-into-products)) — as the margin of the midstream that reveals where the bottleneck really sits.

The demand side has its own structural story, and in the 2020s it acquired a powerful new driver: the **energy transition.** An electric car uses roughly three to four times as much copper as a petrol car; a wind turbine and the grid upgrades to carry its power are copper-and-aluminium-intensive; a battery needs nickel and lithium; solar farms need vast aluminium framing. This is why analysts speak of a structural, multi-decade bid under the base metals that did not exist in the 2000s supercycle, which was a pure construction story. It does not repeal the cycle — metals will still crash in a recession — but it raises the floor and lengthens the boom, because the *trend* demand from electrification compounds on top of the *cyclical* demand from the building economy. Reading a base metal today therefore means holding two clocks at once: the fast cyclical clock (PMIs, Chinese credit, the building cycle) and the slow structural clock (the speed of the transition). The cyclical clock sets the swings; the structural clock sets the floor they swing above.

## Industrial, not monetary: why the price tracks the growth cycle

This is the deepest difference between a base metal and gold, so it deserves a figure of its own. (The chart below traces copper's price against the global cycle.)

![Copper annual average price 2000 to 2025 with growth-cycle phases shaded, showing it tracks the cycle](/imgs/blogs/industrial-metals-and-the-lme-the-warehouse-that-prices-the-world-6.png)

Lay copper's price history against the global growth cycle and the correlation jumps out. The **China boom of the 2000s** dragged copper from under \$2,000/tonne in 2003 to nearly \$9,000 by 2011 as a continent poured concrete and strung wire. The **2008 financial crisis** halved it almost overnight — when factories stop, copper demand evaporates, and the price collapses with it. The long **2011–2016 slide** tracked China's construction slowdown. Then the **2020s electrification story** — EVs, grids, renewables, data centres — bid it back to a fresh record above \$9,000 and an intraday \$11,100 in 2024.

Now contrast gold. Gold barely cares about the factory floor; it cares about real interest rates and fear. During the 2008 crisis, while copper was being cut in half, gold *rose*. That divergence is the whole point: **copper is a bet on growth, gold is a hedge against the absence of it.** One useful lens is the gold-silver ratio, which sits in the data because silver is a hybrid — half industrial, half monetary — and so trades *between* the two worlds (more in [silver, platinum and the gold-silver ratio](/blog/trading/gold/silver-platinum-and-the-gold-silver-ratio-the-rest-of-the-precious-complex)). For the macro mechanics of why real rates drive the monetary metal but not the industrial one, see [how monetary policy moves commodities](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) and [commodities as macro signals](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold).

#### Worked example: the China-share demand sensitivity

China consumes roughly **half** of the world's copper. Treat that as a sensitivity. Suppose global copper demand is **26 million tonnes/year**, of which China is **13 Mt (50%)**, and suppose a Chinese property and grid slowdown trims Chinese copper demand by **6%**:

```
China demand          = 13.0 Mt
China demand drop      = 6% of 13.0 = 0.78 Mt
as a share of world    = 0.78 / 26.0 = 3.0% of global demand
```

A mere **6% wobble in one country's appetite** removes **3% of global demand** — and in a market where supply cannot flex quickly, a 3% demand swing can move the price 15–25%, because the curve has to do all the adjusting through *price*, not *quantity*. The intuition: when one buyer is half the market, that buyer's growth rate *is* the copper price's growth rate, which is why metals traders read Chinese PMIs and credit data as closely as they read warehouse stocks.

## The warehouse-queue scandal: when storage became a money machine

For a few years after the 2008 crisis, the LME's elegant warehouse system was bent into something that looked a lot like a racket — and understanding it teaches you more about how this market really works than any textbook diagram. (The timeline below walks the mechanism step by step.)

![Timeline of the LME warehouse-queue game: metal in, slow load-out, rent accrues, premium inflates, reform](/imgs/blogs/industrial-metals-and-the-lme-the-warehouse-that-prices-the-world-5.png)

Here is the setup. After 2008, demand for aluminium slumped and a glut of metal piled into LME warehouses. Meanwhile, big banks and trading houses had *bought* many of the warehousing companies. A warehouse earns money two ways: a one-time incentive it pays to *attract* metal in, and a daily **rent** it charges for every tonne stored. The more metal in the shed and the longer it stays, the more rent.

Now the game. The LME set only a *minimum* daily load-out rate — the least metal a warehouse had to release per day if owners asked to remove it. It set no *maximum* on how much could pour in. So a warehouse could offer fat incentives to pull in a mountain of aluminium, and then release it back out at the slow minimum rate. Owners who wanted their metal had to *join a queue* — and at one Detroit complex the queue to physically get aluminium out stretched past **a year and a half**. Every tonne waiting in that queue kept paying daily rent to the warehouse owner. The storage system had become a machine for manufacturing rent.

The damage spilled into the real price. A can-maker who needed *physical* aluminium could not wait 18 months, so they paid a fat **physical premium** — a surcharge over the LME price — to get metal from someone who had it. That premium ballooned even as the LME headline price was soft, which meant the *paper* price and the *real, deliverable* price had pulled apart. The LME price said "aluminium is cheap"; the loading dock said "good luck getting any." This is the fragility we flagged earlier: when warrants can be queued and rented, paper metal and obtainable metal diverge.

The scandal drew lawsuits, a US Senate investigation, and finally LME rule changes — most importantly a **load-in/load-out (LILO)** rule forcing warehouses with long queues to ship *out* more than they took *in*, which slowly drained the queues from 2014 on. The episode is the cleanest real-world lesson in why a metals trader watches not just the LME price but the **physical premium** and the **warehouse queues**: the gap between them *is* the market's true tightness, and a system designed to anchor paper to physical can, if the incentives are wrong, do the opposite.

#### Worked example: how warehouse rent turns metal into an annuity

Put numbers on the rent. Suppose a warehouse charges **\$0.50/tonne/day** in rent, and the queue to get your metal out is **400 days**. For one standard 25-tonne copper lot:

```
rent per day      = 25 tonnes x 0.50 USD = 12.50 USD/day
queue length      = 400 days
rent over queue   = 12.50 x 400 = 5,000 USD per lot
```

So the warehouse collects **\$5,000** in rent on a single \$235,000 lot of copper just for the privilege of releasing it slowly — about **2.1%** of the metal's value, skimmed off as storage rent during the wait. Multiply across hundreds of thousands of tonnes and the rent stream is enormous, which is exactly why owning the warehouse, not the metal, was the profitable position. The intuition: rent is a tax on *time in the shed*, so anyone who controls the exit can turn a glut of unwanted metal into a reliable annuity — and that is the perverse incentive the LME had to legislate away.

## When the anchor snaps: the 2022 nickel squeeze

If the warehouse saga showed the system bending, March 2022 showed it breaking. We give this its own full post — [aluminium, nickel, and the 2022 nickel squeeze](/blog/trading/commodities/aluminum-nickel-and-the-2022-nickel-squeeze-when-the-market-broke) — but it belongs in the LME's story because of what it revealed about the limits of a physical exchange.

The short version: a giant Chinese nickel producer held an enormous *short* position on LME nickel — a bet the price would fall — partly as a hedge against its own production. When Russia invaded Ukraine and the market panicked about nickel supply, the price began to rocket. As it climbed, the short position bled losses and triggered margin calls, and the rush to buy back the short fed the rally in a classic **short squeeze**. On **8 March 2022**, LME nickel went vertical: it briefly printed an intraday **\$101,365/tonne**, more than triple where it had started the week.

Then the LME did something extraordinary and deeply controversial: it **cancelled** several hours of trades, rewinding the price to the prior day's close, and suspended the market for a week. It argued the market had become disorderly and that honouring the trades would have bankrupted members and clearing firms. Critics — mostly the funds who had been *long* and watched their winning trades voided — called it a bailout of the short side that broke the cardinal rule of a market: that a trade is a trade. The episode triggered regulatory reviews and lawsuits, and it dented the LME's credibility for years.

For our purposes the lesson is structural. A physical exchange's whole value is that the price is *real* — anchored to deliverable metal and to trades that stand. When the LME cancelled trades, it admitted that under enough stress the anchor could be cut. It is the dark mirror of the warehouse scandal: there, paper drifted *away* from physical and the price was too *low* relative to obtainable metal; here, paper detached *upward* into a squeeze and the exchange chose to sever it. Both are reminders that "the price of nickel" is a human institution, not a law of nature.

#### Worked example: the leverage that made the squeeze lethal

Why does a short squeeze on metal escalate so fast? Leverage. On the LME you post **margin** — a fraction of the contract value — not the full notional. Suppose nickel starts the week near **\$25,000/tonne** and a trader is short with **10% initial margin**:

```
contract notional (6 t nickel)   = 6 x 25,000 = 150,000 USD
initial margin at 10%            = 15,000 USD
price triples to 75,000/tonne    -> new notional = 6 x 75,000 = 450,000 USD
loss on the short                = 450,000 - 150,000 = 300,000 USD
loss vs margin posted            = 300,000 / 15,000 = 20x the margin
```

The short has lost **twenty times** the cash it put up — a wipe-out many times over, and the broker demands the difference *immediately* via a margin call. To stop the bleeding the short must *buy* nickel to close, which pushes the price even higher, which deepens everyone else's losses. The intuition: leverage turns a price move into a solvency event, and a squeeze is leverage running in reverse — the more it moves, the more forced buying it creates, until either the shorts are destroyed or, as in 2022, the exchange pulls the plug.

## LME vs COMEX vs SHFE: three prices for the same metal

A newcomer is often surprised that there is no single "world price" of copper. There are at least three, and they all matter. (The grid below sorts out what each venue anchors.)

![Matrix comparing LME, COMEX and SHFE on what each anchors, currency and delivery, and who reads it](/imgs/blogs/industrial-metals-and-the-lme-the-warehouse-that-prices-the-world-7.png)

- **The LME (London)** is the *global physical* benchmark. It prices the warrant, in dollars, deliverable into approved warehouses on several continents, on the rolling 3-month date. It is the reference the world's producers, merchants and physical traders use. When someone says "the copper price" without qualification, they usually mean the LME.
- **COMEX (New York)**, part of the CME group, is the *dollar paper* benchmark, especially for the US market. It trades calendar-month copper futures (the ticker "HG") with delivery into US warehouses, and it is where many US-based funds and hedgers express a copper view. COMEX is more financialised — heavier with speculative flow — and its price can diverge from the LME when US-specific factors (like tariffs or a regional squeeze) bite. A persistent COMEX-over-LME premium in 2025, for instance, signalled traders pricing in the risk of US copper import tariffs.
- **SHFE (Shanghai)** prices metal *inside China*, in yuan, deliverable only into mainland Chinese warehouses, behind the country's import tariffs and value-added tax. Because China is the dominant consumer, the SHFE price and the **LME-SHFE arbitrage** — adjusted for tax, freight and the exchange rate — are a real-time read on whether China is pulling metal *in* (a premium that sucks imports) or pushing it *out*.

The existence of three venues is not redundancy; it is information. The *spreads between them* tell you where physical metal is wanted. A blown-out COMEX premium says "America is short." A high SHFE-over-LME arb says "China is hungry." The global merchant houses — Glencore, Trafigura and the like, whose business model is exactly this geographic price arbitrage ([commodity trading houses](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura)) — exist to move metal from the cheap venue to the dear one and pocket the difference, and in doing so they knit the three prices back toward each other.

## Common misconceptions

**"The LME price is the price of copper, full stop."** No. The LME price is the price of a *warrant* — title to a specific brand and location of metal on the 3-month date. The price an actual buyer pays is the LME price *plus a physical premium* that depends on where they are, what brand they need, and how tight the local market is. During the warehouse-queue years that premium ballooned while the LME price stayed soft; the headline number and the real cost of metal told opposite stories.

**"Contango means the market expects higher prices."** This is the most common error in all of commodities, and it is wrong. Contango (cash below 3-month) is mostly the *cost of carry* — storage, financing and rent — not a forecast. A steep contango usually signals a *glut* and a *weak* prompt market, not bullishness. Likewise backwardation (cash above 3-month) signals tightness *now*, not a bearish forecast. We hammer this in [contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) and [convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape); on the LME the cash-3m spread *is* this signal in its purest form.

**"Industrial metals are an inflation hedge like gold."** Only loosely, and for a different reason. Gold hedges monetary debasement and fear; base metals "hedge" inflation only insofar as inflation coincides with strong *growth* and rising input costs. In a **stagflation** — high inflation *and* weak growth — gold can soar while copper sags, because the factory demand is missing. (See [the 1970s stagflation case](/blog/trading/cross-asset/case-study-1970s-stagflation-commodities-win) for the messy history.) Lumping them together as "metals that hedge inflation" misses that one is a growth asset and the other is a fear asset.

**"The 3-month date means delivery in exactly three calendar months."** It is a *rolling* date that moves forward one business day each day. Today's 3-month and tomorrow's 3-month are different delivery dates. This trips up everyone moving over from a calendar-month exchange like COMEX, and it is why LME spread trading (cash-3m, 3m-15m) has its own vocabulary.

**"A short squeeze is just a risk for the shorts."** The 2022 nickel day showed it is a risk for the *whole market*, including the exchange. A big enough squeeze can threaten the solvency of clearing members and force the exchange into an impossible choice — let trades stand and risk a default cascade, or cancel them and break the promise that a trade is a trade. The squeeze risk is systemic, not just personal.

**"LME warehouse stocks show all the metal in the world."** They show only the metal *on warrant in approved warehouses* — a sliver of total global inventory. Vast quantities sit off-exchange: in bonded warehouses in China, in producer and consumer stockpiles, in metal "financed" and parked outside the LME system to dodge reporting. So LME stocks can be near zero while plenty of metal exists somewhere — it is just not *available on the exchange right now*. That is why the cash-3m spread, which prices *availability*, often matters more than the headline stock figure, and why a sudden inflow of "invisible" metal onto warrant can flip a tight market loose overnight. Treat LME stocks as a thermometer for the *visible* market, not a census of all metal.

## How it shows up in real markets

**May 2024 — copper's record and the physical tell.** When LME copper hit ~\$11,100, the durable signal was not the headline but the combination of *low visible LME stocks* and *spiking backwardation in the cash-3m spread*: the prompt market was bidding for metal. A trader who watched the spread saw the tightness building before the record print and saw it relax afterward as Chinese scrap and destocking eased the squeeze. The price was the *effect*; the spread and the stocks were the *cause* you could read in advance.

**2013–2014 — the Detroit aluminium queues.** With LME aluminium soft, the *physical premium* paid by real consumers (brewers, carmakers) rose to record highs, and the load-out queue at one Midwest complex passed 600 days. The lesson traders took away — watch the premium and the queue, not just the flat price — outlived the specific scandal and is now standard practice.

**March 2022 — the nickel break.** A short squeeze drove LME nickel to \$101,365 intraday before the exchange cancelled trades. For months afterward LME nickel liquidity was thin and the price unreliable, and a chunk of physical nickel pricing migrated to other references. It was the clearest demonstration in a generation that an exchange price is only as good as the institution standing behind it.

**2025 — the COMEX-LME copper dislocation.** As markets priced in the risk of US copper import tariffs, COMEX copper traded at a persistent premium to the LME, and metal was pulled toward US warehouses to capture it. The *spread between the two exchanges* was the trade and the signal at once — a live example of why three venues beat one.

**2003–2008 — the China supercycle and the great backwardation.** The cleanest demonstration of "industrial metal as growth bet" is the first decade of the 2000s, when China's industrialisation pulled copper from under \$2,000/tonne to nearly \$9,000. Through much of that boom, LME copper sat in *backwardation* — cash above 3-month — for an unusually sustained stretch, and LME warehouse stocks ran down to razor-thin levels. The two signals agreed and reinforced each other: empty sheds and a paid-up prompt both said the same thing, that the world physically could not get enough copper, and the price had to ration it upward. A trader who had only watched the flat price saw a rally; a trader who watched the *spread and the stocks together* understood it was a genuine physical shortage with years left to run, not a speculative froth that would snap back. That is the difference between reading the number and reading the metal, and it is the supercycle's enduring lesson. (For the macro framing of that era, see [the 2000s China commodity supercycle](/blog/trading/cross-asset/case-study-2000s-china-commodity-supercycle).)

## The takeaway: how a metals trader actually reads the LME

Strip away the romance of the Ring and the history of the sailing ships, and a metals trader reads the LME through a short, disciplined checklist — and almost none of it is the headline price.

**Read the cash-3m spread first.** Is the prompt date in backwardation (cash above 3-month, tight) or contango (cash below 3-month, ample)? How wide, and which way is it *moving*? A backwardation that widens fast is the earliest warning of a physical squeeze; a deepening contango says the sheds are filling. Annualise it (as in the worked examples) to judge whether it is a real tightness signal or just the cost of carry.

**Read the stocks and the cancelled warrants.** Falling LME stocks plus rising cancelled warrants means metal is leaving the visible system — bullish for tightness. Rising stocks mean the opposite. The *rate of change* matters more than the level.

**Read the physical premium and the queues.** The gap between the LME price and what a real consumer pays for deliverable metal is the truest measure of tightness, and the warehouse-queue scandal proved that this gap can scream while the flat price whispers.

**Read the venue spreads.** Watch COMEX-over-LME (is America short?) and SHFE-over-LME (is China hungry?). Those arbs are real-time maps of where physical metal is wanted, and the merchant houses move metal to close them.

**Frame it as a growth bet, then watch China.** Above all, remember what you are trading: a *consumption* asset whose demand is the global growth cycle, with one country accounting for half of it. A 6% wobble in Chinese demand is a 3% global shock, and in an inflexible supply market that is a large price move. When you read a metals price, you are reading the world economy's pulse — which is the opposite of reading gold, where you are reading its fears. (For the policy and shock side, see [geopolitics, elections and unscheduled shocks](/blog/trading/event-trading/geopolitics-elections-and-unscheduled-shocks).)

The deepest insight the LME teaches is the one this whole series is built on: a commodity price is a physical thing forced through a financial contract. Nowhere is that more literal than London, where the contract is a warrant to a numbered lot in a numbered shed, the benchmark date is a fossil of 19th-century shipping, and the price you see on the screen is only ever as real as the metal someone can actually walk out of the warehouse. Read the spread and the stocks, and you are reading the metal. Read only the price, and you are reading the shadow.

## Further reading & cross-links

- [Doctor Copper and the pulse of the global economy](/blog/trading/commodities/copper-doctor-copper-and-the-pulse-of-the-global-economy) — copper as the world's leading economic indicator.
- [Aluminium, nickel, and the 2022 nickel squeeze](/blog/trading/commodities/aluminum-nickel-and-the-2022-nickel-squeeze-when-the-market-broke) — the day the LME cancelled trades, in full.
- [The forward curve: the most important chart in commodities](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — how to read the whole strip of dates.
- [Contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) and [convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape) — why the curve has a shape.
- [Is gold money, a commodity, or a currency](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) and [silver, platinum and the gold-silver ratio](/blog/trading/gold/silver-platinum-and-the-gold-silver-ratio-the-rest-of-the-precious-complex) — the monetary side of the metals world.
- [Metals, copper and silver, the economy's pulse](/blog/trading/cross-asset/metals-copper-silver-the-economys-pulse) — base metals in an asset-allocation frame.
- [Commodities as macro signals](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) and [how monetary policy moves commodities](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) — the macro lens.
- [Commodity trading houses: Glencore, Vitol, Trafigura](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura) — the merchants who arbitrage the venue spreads.
