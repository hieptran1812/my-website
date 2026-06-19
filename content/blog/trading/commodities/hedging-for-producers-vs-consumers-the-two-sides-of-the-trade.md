---
title: "Hedging for Producers vs Consumers: The Two Sides of the Trade"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a producer sells the futures and a consumer buys them, how a collar floors revenue or caps cost, and why a hedge that loses money on the derivative leg is still a success."
tags: ["commodities", "hedging", "risk-management", "futures", "options", "collar", "basis-risk", "producers", "consumers", "energy"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A hedge is a risk-management decision, not a market bet: the producer is born long the commodity and sells to floor revenue, the consumer is born short the input and buys to cap cost.
>
> - A **producer** (oil E&P, miner, farmer) will *sell* the commodity, so they are naturally **long**. They hedge by **selling futures**, **buying puts**, or putting on a **producer collar** (buy a put, sell a call to fund it) to set a revenue floor.
> - A **consumer** (airline, food company, utility, refiner) must *buy* the input, so they are naturally **short**. They hedge by **buying futures**, **buying calls**, or a **consumer collar** to set a cost ceiling.
> - A hedge that **loses money on the derivative leg is still a success** — the spot side moved your way, and the leg's "loss" is the premium you paid for certainty. Judge a hedge by variance removed, not by the P&L of one leg.
> - The number to remember: a hedge does not change your *expected* price much — it changes the **spread of outcomes**. An un-hedged producer is a leveraged directional bet on one commodity.

In April 2020 the price of a barrel of WTI crude oil for May delivery did something it had never done in the history of the contract: it went *negative*, settling at **−\$37.63**. Sellers paid buyers \$37 a barrel to take oil off their hands because there was no storage left to put it in. Six months earlier, that same barrel had traded near \$60. Two years later, after Russia invaded Ukraine, it spiked back toward \$130. If you ran an airline, every one of those swings was a multi-billion-dollar question mark hanging over your single largest cost. If you ran a shale driller, the same swings decided whether you could service your debt or filed for bankruptcy.

The people who actually *use* commodities — who pump them, mine them, grow them, refine them, or burn them — cannot live with that kind of uncertainty. A farmer plants corn in April and cannot sell it until October; the price could be anything by then. An airline sells a ticket today for a flight six months out, and the jet fuel for that flight has not been bought yet. These businesses have a *physical* exposure to a price they do not control, and they would very much like to make it go away. The market they use to do that is the futures and options market — the same paper instruments a speculator uses to bet on price, used here for the opposite purpose: to *remove* a bet that already exists.

This is the heart of the matter, and it is the part most people get backwards. A producer who sells futures is not "betting the price will fall." A consumer who buys futures is not "betting the price will rise." They are doing the opposite of betting — they are taking a price risk they were born with and handing it to someone willing to carry it. Once you see hedging as the *transfer* of an existing risk rather than the *creation* of a new one, the whole logic of who-does-what falls into place.

![Producer naturally long sells to floor revenue, consumer naturally short buys to cap cost, both meeting in the futures market](/imgs/blogs/hedging-for-producers-vs-consumers-the-two-sides-of-the-trade-1.png)

This post is about the two sides of that trade. We will build up from the simplest idea — what it means to be "naturally long" or "naturally short" — through the instruments (futures, puts, calls, collars), the worked dollar math of an airline and a farmer and a gold miner, the basis risk that means no hedge is ever perfect, and the accounting and margin mechanics that bite real treasurers. We will end on the one idea that matters most for reading a company's filings: a hedge that *lost* money is usually a hedge that *worked*, and an un-hedged producer is a leveraged directional bet on one commodity whether its management admits it or not.

## Foundations: long, short, and what a hedge actually does

Before any instrument, two words. In markets, you are **long** something if you profit when its price rises and lose when it falls — you own it, or you have a claim that benefits from it going up. You are **short** something if you profit when its price falls and lose when it rises — you owe it, or you will have to buy it later and a higher price hurts you.

Here is the move that confuses everyone, and it is worth slowing down for. Your "long" or "short" position does not have to come from a trade. It can come from your *business*. A copper miner has not bought a single futures contract, and yet the miner is profoundly **long copper**: the company's entire revenue is "however many tonnes we dig up, times whatever copper sells for." If copper doubles, the miner's revenue roughly doubles; if it halves, revenue halves. The miner is long copper the way a homeowner is long their house — without ever placing an order. This is a **natural long**, also called a structural or operational long.

The mirror image is the **natural short**. An airline has not sold any oil futures, and yet it is deeply **short oil**: it will have to *buy* jet fuel, lots of it, at whatever the price turns out to be. If oil doubles, the airline's single biggest cost doubles. A business that must purchase a commodity to operate is short that commodity, because a rising price is a loss to them. The airline is short oil the way a renter is short housing — they owe future payments at an unknown price.

Now the definition of a hedge becomes simple: **a hedge is a financial position that is the opposite of your natural one, sized so the two cancel.** The miner is naturally long copper, so the miner takes a financial *short* (sells futures) to offset it. The airline is naturally short oil, so the airline takes a financial *long* (buys futures) to offset it. After hedging, the gain on one side is paid for by the loss on the other, and the net exposure to the price shrinks toward zero. You have not made the price go away. You have made your *sensitivity* to the price go away.

That last sentence is the whole game, so let us be precise about what changes and what does not. Hedging barely moves your *expected* price — over many years, a hedger and a non-hedger pay roughly the same average price, because futures markets are roughly fair. What hedging crushes is the *variance*: the spread of possible outcomes around that average. The naive view is that a hedge "protects you from bad prices." The accurate view is that a hedge **trades away the good outcomes to be rid of the bad ones**, leaving you with the known middle. A farmer who hedges gives up the windfall of a price spike in exchange for being spared the ruin of a price collapse. Most real businesses take that trade gladly, because they are not in business to gamble on price — they are in business to mine, fly, bake, or refine, and they want the price out of the way so they can do that.

### Why "naturally short" feels strange — and a quick analogy

If you have never traded, "the airline is short oil" sounds bizarre, so here is an everyday version. Suppose you have signed a lease to rent an apartment for the next ten years, but the rent resets every year to whatever the market says. You do not *own* anything, yet you have a giant exposure: if rents soar, you are crushed; if rents crash, you celebrate. You are **short rent**. The way you would hedge is to lock the rent in advance — sign a fixed-rent lease — which is exactly buying a "future" on your own cost. An airline buying oil futures is doing the identical thing: fixing tomorrow's input price today. Hold that picture, because the consumer's hedge is always "lock the cost I will have to pay."

The producer's side is the homeowner who plans to sell. You own a house you intend to sell next year. You are **long the housing market** — a crash before you sell wrecks your plans, a boom is a gift. To hedge, you would want to lock a sale price *now*. That is what selling futures does for a producer: it nails down the price at which next year's output will be sold. Same instrument, opposite direction, opposite natural position.

### Why hedging exists at all: the cost of ruin is not symmetric

If hedging does not change your average price, a sharp reader asks the obvious question: why bother? Why not just ride the swings and come out even over the long run? The answer is that for a real business the *cost* of a bad outcome is not the mirror image of the *benefit* of a good one. A farmer who hits a price spike makes a nice profit; a farmer who hits a price crash *defaults on the operating loan and loses the farm*. Those are not symmetric outcomes — one is a good year, the other is the end of the business. Economists call this **convex costs of distress**: the pain of a 50% adverse move is far more than the pleasure of a 50% favorable one, because past a certain point you hit covenant breaches, forced asset sales, lost credit lines, and bankruptcy. A hedge that flattens the distribution is worth giving up the upside precisely because it removes the *left tail* where the firm dies.

There is a second reason that has nothing to do with the firm's own balance sheet: **planning.** A business that knows its key price can budget, set product prices, sign supply contracts, plan capex, and make promises to investors that it can keep. An airline that knows its fuel cost can price tickets a year out without betting the company on OPEC. A bakery that locks wheat can quote a fixed bread price to a supermarket chain. The certainty itself has operational value far beyond the variance reduction — it lets the business behave like a business instead of a trading desk. This is why the right way to think about a hedge is as the purchase of *predictability*, and predictability is worth paying for even when, on average, it is priced fairly.

### The hedge ratio: how much to hedge

A hedge is not all-or-nothing. The **hedge ratio** is the fraction of your natural exposure you choose to offset — 0% (fully exposed), 100% (fully neutralized), or anything in between. Most real producers and consumers hedge a *rolling fraction* of expected volume: heavily for the near months (which they can forecast confidently), lightly for the far months (which are uncertain), and never beyond the volume they are sure they will produce or consume. A common pattern is "hedge 80% of next quarter, 50% of next year, 20% of the year after" — a declining ladder that keeps the hedge sized to the confidence in the forecast.

Two things push the ratio below 100%. First, you do not want to be *over*-hedged: if you hedge 100% of an *expected* harvest and the harvest comes in short (a drought halves your crop), you now have more short futures than physical to deliver against — you have accidentally become a *speculator*, short the very commodity you produce, exactly when its price is spiking. Producers deliberately under-hedge their forecast for this reason. Second, the ratio reflects how much upside the firm's owners want to keep — the gold-miner lesson, applied as a dial rather than a switch. The hedge ratio is where the art lives: too little and you are exposed, too much and you have either killed the exposure investors wanted or turned a hedge into a bet.

## The producer's toolkit: sell futures, buy puts, or collar

A producer who is naturally long has three standard ways to floor revenue, in increasing order of nuance.

**Sell futures (or forwards).** The bluntest tool. The producer sells futures contracts equal to expected output. If they will dig up 10,000 tonnes of copper next quarter and sell one futures contract per 25 tonnes, they sell 400 contracts. Now whatever copper does, their *net* selling price is locked near today's futures price. If copper falls, the physical sale brings less but the short futures gains exactly as much; if copper rises, the physical sale brings more but the short futures loses exactly as much. The two cancel. This is a **fixed-price hedge** — it nails the price to a single number and gives up *all* upside as well as all downside. It is symmetric: you are equally indifferent to a spike or a crash, because you no longer have a position.

**Buy puts.** A put option is the right, not the obligation, to *sell* at a chosen strike price. A producer who buys puts at, say, a \$70/bbl strike has bought a floor: if oil falls below \$70, the puts pay the difference and revenue cannot fall below \$70 (minus the premium paid); if oil *rises* above \$70, the producer simply lets the puts expire worthless and sells the physical at the high price. Puts are insurance: they cost an upfront premium, and in exchange they protect the downside while leaving the upside intact. The producer keeps the windfall of a price spike and is shielded from a crash — but pays for the privilege. (For the full mechanics of how an option is priced and why that premium is what it is, see the options series on [what sets an option's price](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition).)

**The producer collar.** Puts are nice but the premium stings. The collar is the answer: **buy a put to set the floor, and sell a call to pay for it.** Selling a call obligates the producer to deliver at the call's strike if price rises above it — which means giving up the upside above that strike. In return, the producer collects a premium that funds (often entirely) the put they wanted. The result is a *band*: revenue is floored at the put strike and capped at the call strike, for little or no net cost. A producer collar is "I will give up gains above \$95 so that I do not have to pay anything for protection below \$70."

![Producer collar floors revenue between a put and a call, consumer collar caps cost between a call and a put](/imgs/blogs/hedging-for-producers-vs-consumers-the-two-sides-of-the-trade-2.png)

The figure shows both collars side by side, because they are mirror images. On the producer side (left), revenue tracks the oil price but is floored at the put strike of \$70 and capped at the call strike of \$95: a known band. On the consumer side (right), cost tracks the oil price but is capped at the call strike of \$90 and floored at the short-put strike of \$65: the inverse band. Same shape, flipped — because the producer fears low prices and the consumer fears high ones.

#### Worked example: a producer collar on 10,000 tonnes of copper

A copper miner expects to sell **10,000 tonnes** next quarter. Spot is **\$9,000/tonne**, and the miner is terrified of a repeat of the 2016 slump (copper averaged \$4,868 that year). The treasurer puts on a collar:

- **Buy a put** struck at **\$8,000/tonne**, premium **\$180/tonne**.
- **Sell a call** struck at **\$10,000/tonne**, premium **\$170/tonne**.
- **Net cost** = 180 − 170 = **\$10/tonne**, or 10 × 10,000 = **\$100,000** for the whole position.

Now trace three outcomes at sale:

- **Copper crashes to \$6,000.** Physical sale: 6,000 × 10,000 = **\$60.0m**. The put is \$2,000 in the money: (8,000 − 6,000) × 10,000 = **+\$20.0m**. The call expires worthless. Net = 60.0 + 20.0 − 0.1 (collar cost) = **\$79.9m**, i.e. an effective **\$7,990/tonne** — the floor held.
- **Copper sits at \$9,000.** Physical sale: **\$90.0m**. Both options expire worthless. Net = 90.0 − 0.1 = **\$89.9m**, an effective **\$8,990/tonne** — the band's middle, the premium is the only cost.
- **Copper soars to \$12,000.** Physical sale: **\$120.0m**. The put expires worthless. The short call is \$2,000 in the money *against* the miner: (12,000 − 10,000) × 10,000 = **−\$20.0m**. Net = 120.0 − 20.0 − 0.1 = **\$99.9m**, an effective **\$9,990/tonne** — the cap held.

The miner has locked revenue into a band between **~\$7,990 and ~\$9,990 per tonne** for a net \$10/tonne. The intuition: a collar buys away the catastrophe of a price crash by selling away the lottery ticket of a price spike, leaving a known band you can finance against.

## The consumer's toolkit: buy futures, buy calls, or collar

The consumer's toolkit is the producer's reflected in a mirror. A consumer is naturally short, fears *high* prices, and wants a **cost ceiling**.

**Buy futures (or forwards).** The blunt tool again, flipped. The airline buys oil futures equal to expected fuel burn. Now its net cost is locked near today's futures price: if oil rises, the physical fuel costs more but the long futures gains as much; if oil falls, the fuel costs less but the long futures loses as much. The two cancel; the cost is fixed at a single number, with no exposure to either direction.

**Buy calls.** A call is the right to *buy* at a strike. A consumer who buys calls at a \$90/bbl strike has a ceiling: if oil rises above \$90, the calls pay the difference and the effective cost cannot exceed \$90 (plus premium); if oil *falls* below \$90, the consumer lets the calls expire and simply buys cheap fuel in the spot market. Calls are insurance against a spike while keeping the benefit of a crash — for an upfront premium.

**The consumer collar.** To fund the call, the consumer **sells a put** below the market. Selling the put obligates them to "buy" at the put strike if price falls below it — meaning they give up the savings from a deep crash. In return they collect premium that pays for the call. The result is the inverse band: cost capped at the call strike, but with no benefit below the put strike. A consumer collar is "I will give up the savings from oil below \$65 so that my fuel never costs more than \$90, for nearly nothing."

This is exactly Southwest Airlines' famous playbook, and it is worth pausing on because it is the canonical case of consumer hedging done well — and then done badly.

### Southwest's legendary fuel hedges

For roughly a decade starting in the late 1990s, Southwest Airlines ran the most aggressive and most celebrated fuel-hedging program in the industry. Using a stack of call options, collars, and swaps, Southwest had locked in large fractions of its future jet fuel at prices set years in advance. When crude marched from \$30 toward its July 2008 intraday peak of **\$147.27** a barrel, most airlines were strangled — fuel went from a manageable line item to the single thing that decided survival. Southwest, having bought the right to keep paying old prices, sailed through. By various estimates the hedges saved the airline on the order of **several billion dollars** cumulatively over 1999–2008, and the program was widely credited with funding Southwest's expansion while rivals retrenched. It is the textbook story of a consumer using the paper market to cap a physical cost.

But the same story has a second act that is just as instructive. In the second half of 2008, oil collapsed from \$147 back toward \$30 by December. Now Southwest's hedges were *underwater* — it had locked in fuel at prices well above the new market, and had to mark large losses on the derivative positions (the airline reported roughly **\$100m+** of hedging-related charges in a single quarter as fuel cratered). Headlines that had praised the hedges now mocked them. And here is the lesson that the whole post is building toward: **those "losses" were not a failure.** Southwest had bought certainty, the price moved the favorable way, and so the insurance paid out nothing — exactly as insurance does in a year your house does not burn down. You do not call your fire insurance a failure because the house stood. We will make this rigorous in a moment.

#### Worked example: an airline locking jet fuel on 100 million gallons

A regional airline expects to burn **100 million gallons** of jet fuel next year. Jet fuel tracks crude closely; to keep the arithmetic clean we will hedge with crude futures and assume one barrel of crude (42 gallons) underlies the fuel cost, so the airline's exposure is about **100,000,000 / 42 ≈ 2.38 million barrels** of crude-equivalent. Crude is **\$80/bbl** today, and the airline buys futures to lock it.

- **Hedge:** buy 2.38m barrels of crude-equivalent at **\$80**. Locked fuel-crude cost = 2.38m × 80 = **\$190.5m**.

Now compare hedged vs un-hedged across two worlds at the time the fuel is actually bought:

- **Oil rises to \$110.** Un-hedged, the airline pays 2.38m × 110 = **\$261.9m** for its fuel — a \$71.4m hit it never budgeted. Hedged, the physical fuel still costs \$261.9m, *but* the long futures gained (110 − 80) × 2.38m = **+\$71.4m**, so the net cost is back to **\$190.5m**. The hedge saved \$71.4m.
- **Oil falls to \$60.** Un-hedged, the airline pays 2.38m × 60 = **\$142.9m** — a windfall. Hedged, the physical fuel costs \$142.9m, but the long futures *lost* (60 − 80) × 2.38m = **−\$47.6m**, so net cost is again **\$190.5m**. The hedge "lost" \$47.6m on the derivative leg.

Look closely at the down-move. The hedge *lost money* on the futures: \$47.6m gone. A naive board member shouts that the treasurer blew \$47.6m. But the airline's total fuel cost was **\$190.5m either way** — exactly the number it had budgeted, planned ticket prices around, and promised investors. The intuition: the hedge converted a fuel bill that could have been anywhere from \$143m to \$262m into a flat \$190.5m, and the \$47.6m "loss" is simply the price the airline paid to know that number in advance.

## Why a "losing" hedge is a winning hedge

This is the single most misunderstood idea in corporate risk management, so it earns its own section and its own figure.

![Unhedged outcome is a wide range, hedged outcome is a narrow known band, the losing leg is the premium for certainty](/imgs/blogs/hedging-for-producers-vs-consumers-the-two-sides-of-the-trade-4.png)

The figure contrasts the two worlds. **Un-hedged** (left), the firm faces a wide fan of outcomes — soaring, flat, crashing — and it cannot plan around any of them. The firm is, whether it likes it or not, a leveraged directional trade on one commodity. **Hedged** (right), every one of those same price scenarios lands the firm inside a narrow band: when price moves favorably the derivative leg loses but the physical offsets it, when price moves adversely the derivative leg wins and rescues the physical, and the net result is always the known band. The "loss" on the leg in the favorable scenario is not a mistake; it is the **premium for certainty**, paid in the only currency certainty is ever paid in.

A hedge has exactly one job: **remove variance.** It does this *symmetrically* — it cuts off the good tail and the bad tail together. So in any year, one leg of the hedge will show a profit or a loss depending on which way the price went, but the *purpose* was never to make the leg profitable. Judge a hedge the way you judge a seatbelt. A seatbelt that you wore on a day you did not crash did not "waste" anything; it did its job, which was to be there in case. A fire policy that paid nothing because nothing burned was not a bad purchase. A fuel hedge that "lost" \$47.6m because oil fell did exactly what it was bought to do — it removed the uncertainty, and the firm's cost came in on plan.

There is a corollary that catches even sophisticated managers. Because a hedge is symmetric, **boards that praise a hedge in the year it pays and punish it in the year it doesn't are guaranteeing they will eventually un-hedge at the worst possible time.** The discipline of hedging requires accepting that, by design, roughly half the time the derivative leg will lose money — and that this is not evidence the program is broken. The firms that hedge well are the ones whose boards understand they bought variance reduction, full stop, and do not re-litigate the decision every quarter based on which way the leg happened to break.

### The gold miner's curse: hedging into a bull market

If a losing hedge is still a successful hedge, is there ever a case where a hedge truly is a mistake? Yes — and it is the most painful object lesson in the commodity world: the gold producers who forward-sold their output into the teeth of the 2000s gold bull market.

Through the late 1990s, with gold flat and falling toward \$250/oz, gold miners were urged by banks and shareholders to lock in prices by forward-selling years of future production — selling gold forward at, say, \$300/oz to guarantee revenue. It was textbook producer hedging. Then gold began one of the great bull runs of all time, climbing past \$500, \$1,000, and eventually well over \$1,900/oz by 2011. The miners who had sold forward were now obligated to deliver their gold at \$300 while spot traded at four, five, six times that. Their hedge "book" became a gigantic liability. Barrick Gold — at the time the world's largest gold producer — had one of the most infamous hedge books, and spent years and **billions of dollars** unwinding it, taking enormous charges to buy back the contracts it had sold. By the late 2000s Barrick announced it would eliminate its gold hedges entirely, and other producers followed; the industry's hedge book collapsed from thousands of tonnes to a fraction of that.

So was this a "losing hedge that still worked"? Not quite — and the distinction is the lesson. The problem was not that the hedge lost money on the leg (which by our rule is fine). The problem was that the producers hedged *too much, too far out,* and in doing so they **destroyed the very exposure their shareholders had bought the stock to get.** People buy a gold miner partly *because* they want leverage to the gold price. A miner that forward-sells five years of production has turned itself into a bond — it has hedged away the upside its investors were paying for. The lesson is not "hedging is bad"; it is that a hedge must be sized to the risk you actually want to remove, not to eliminate the entire business. For producers of a commodity that investors hold *for* its upside (gold above all), over-hedging is a strategic error dressed up as prudence. This is why gold gets treated as a special, monetary case — and why the framing of [whether gold is money or a commodity](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) changes everything about how its producers should hedge.

#### Worked example: a farmer's corn hedge (the floor)

Back to the simplest producer of all. A farmer plants in April and expects to harvest **50,000 bushels** of corn in October. In April, the December corn futures trade at **\$4.80/bushel**. The farmer's bank will only extend an operating loan if the farmer can show a guaranteed price, because the farm cannot survive a repeat of the post-2012 collapse (corn fell from a 2012 drought peak near \$8.31 to \$3.36 by 2016). So the farmer **sells 10 futures contracts** (corn contracts are 5,000 bushels each: 10 × 5,000 = 50,000 bushels) at \$4.80.

At harvest, compare:

- **Corn falls to \$3.50.** Physical sale: 50,000 × 3.50 = **\$175,000**. The short futures gained (4.80 − 3.50) × 50,000 = **+\$65,000**. Net = 175,000 + 65,000 = **\$240,000**, an effective **\$4.80/bushel** — the floor held, exactly as the bank required.
- **Corn rises to \$6.00.** Physical sale: 50,000 × 6.00 = **\$300,000**. The short futures *lost* (4.80 − 6.00) × 50,000 = **−\$60,000**. Net = 300,000 − 60,000 = **\$240,000**, again an effective **\$4.80/bushel**.

Either way the farmer realizes **\$240,000**, the number written into the loan agreement back in April. In the up-move the futures lost \$60,000, and a neighbor who did not hedge crows about selling at \$6.00. But the hedged farmer got the loan, planted the crop, and never risked the farm on the weather and the war news between April and October. The intuition: the farmer is not in the business of forecasting corn prices, the farmer is in the business of *growing corn*, and the hedge lets the farm be a farm instead of a leveraged bet on the December futures.

Note the symmetry between the corn farmer and the airline. Both used a flat futures hedge; both locked one price; both saw one leg "lose" in the favorable scenario. The farmer is a *producer* (sold futures, floored revenue); the airline is a *consumer* (bought futures, capped cost). Same machine, opposite gears — which is exactly the spine of this whole series: a physical exposure forced through a financial contract, with the producer and consumer sitting on opposite ends of that contract. (For who *else* sits in the contract — the speculators who take the leftover risk — see [the four players](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators).)

## The price you are hedging: why the swings are this violent

It is worth grounding the abstraction in the actual numbers a hedger stares at, because the violence of commodity prices is what makes hedging a survival decision rather than a finance-team luxury.

![WTI crude annual average 2000 to 2025 with spikes in 2008 and 2022 and the 2020 crash](/imgs/blogs/hedging-for-producers-vs-consumers-the-two-sides-of-the-trade-3.png)

The chart is the airline's nightmare drawn as a line. WTI averaged about **\$30/bbl** in 2000, roared to nearly **\$100** in 2008 (intraday \$147), crashed to **\$39** in 2020 (with the famous negative print), and spiked back to **\$95** in 2022 after the invasion of Ukraine. A cost that can double or halve on a few years' notice is not a cost you can plan around — and for an airline, fuel is typically **20–35% of total operating expense**, so a doubling of oil can wipe out the entire profit margin. That is why fuel hedging is not optional for a well-run carrier; it is the difference between a planned business and a leveraged bet on OPEC and geopolitics. The shaded green annotation marks 2020, the one year an *un-hedged* airline won big — and the reason boards lose their nerve and cut hedges right before the next spike.

The producer side is just as wild. A miner's revenue rides the same kind of curve.

![LME copper annual average 2000 to 2025 with the 2011 peak and the 2016 trough](/imgs/blogs/hedging-for-producers-vs-consumers-the-two-sides-of-the-trade-5.png)

Copper — "Dr. Copper," the metal with a PhD in economics because it tracks global growth — ran from about **\$1,814/tonne** in 2000 to **\$8,828** at the 2011 China-boom peak, slumped to **\$4,868** by 2016, and pushed to a **\$9,150** average in 2024 (with an intraday record near \$11,104). A miner whose all-in cost to produce is, say, \$6,000/tonne is wildly profitable at \$9,000 and underwater at \$4,868. The red annotation marks the 2016 trough — exactly the kind of price a producer's put or forward sale exists to floor. The miner cannot control whether China builds; the miner can control whether a price collapse forces a default on its project finance, and that is what the hedge is for.

The point of both charts is the same: these are not gentle, mean-reverting series you can ride out. They gap, spike, and crater on war, weather, OPEC decisions, and demand shocks. A business with a structural exposure to one of them and no hedge is, quite literally, a directional position on a chart it does not control.

## Basis risk: why no hedge is ever perfect

So far we have pretended the hedge is perfect — that the futures price and the physical price move in lockstep so the offset is exact. In reality they do not, and the gap between them is the most important imperfection in hedging: **basis risk.**

![Basis risk diagram showing physical grade location and timing differ from the standardised futures contract](/imgs/blogs/hedging-for-producers-vs-consumers-the-two-sides-of-the-trade-6.png)

The **basis** is the difference between the price of *your specific physical commodity* and the price of *the standardized futures contract* you hedged with. It exists because a futures contract is a single rigid specification — one grade, one delivery point, one expiry — while your physical barrel, tonne, or bushel is none of those things exactly:

- **Grade.** WTI futures specify light, sweet crude delivered at Cushing, Oklahoma. If you produce a heavier, sourer crude in the Permian or in Canada, it sells at a *discount* to the benchmark, and that discount moves on its own. Your hedge offsets the benchmark, not your grade.
- **Location.** The contract delivers at one hub. Your oil is at the wellhead, your corn is at a Midwest elevator, your gas is at a different pipeline node. The cost and availability of transport to the delivery point is a basis that shifts with pipeline capacity and congestion.
- **Timing.** The contract expires on a fixed date. You sell your physical on *your* schedule, which rarely matches expiry exactly, so the futures you used may have rolled or drifted relative to the spot you actually realize.

Because of all this, a hedge removes the *benchmark* price risk but leaves the **basis** as a residual. The figure traces it: your physical and the futures feed three gaps — grade, location, timing — which combine into the basis, and the hedge lands you at "protected, but not perfect." Basis is usually far smaller and more stable than outright price (a few dollars on a \$80 barrel, versus the \$80 swinging by \$50), which is exactly why hedging is still overwhelmingly worth it — you trade a huge, violent risk for a small, manageable one. But basis is not zero, and in stressed markets it can blow out spectacularly (recall April 2020, when the WTI *contract* went to −\$37 because of a storage crunch *at Cushing specifically* — a basis event, not a global-oil event).

#### Worked example: the basis-risk residual on a crude hedge

A Permian producer expects to sell **100,000 barrels** of its crude next month. Its oil normally sells at a **\$4/bbl discount** to WTI (grade + location basis). WTI futures are **\$80**, so the producer expects to realize \$80 − \$4 = **\$76**. The producer hedges by selling 100 WTI futures contracts (1,000 bbl each) at \$80.

- **Base case — basis stays \$4.** WTI falls to \$70. Physical sale at the local price 70 − 4 = \$66: 100,000 × 66 = **\$6.60m**. Short futures gain: (80 − 70) × 100,000 = **+\$1.00m**. Net = 6.60 + 1.00 = **\$7.60m**, an effective **\$76/bbl** — exactly as planned. The hedge worked.
- **Basis blows out to \$10.** WTI still falls to \$70, but a regional glut widens the local discount to \$10, so the physical sells at 70 − 10 = \$60: 100,000 × 60 = **\$6.00m**. Short futures gain is unchanged at **+\$1.00m** (it only tracks WTI). Net = 6.00 + 1.00 = **\$7.00m**, an effective **\$70/bbl** — \$6 worse than planned.

The hedge removed the \$10 *outright* move in WTI flawlessly, but it could not protect the producer from the \$6 *basis* move, because the futures contract knows nothing about a Permian glut. The intuition: a hedge cancels the contract's price, never your barrel's price, so the leftover basis is the irreducible residual you accept in exchange for killing the much larger benchmark risk.

### Cross-hedging: when no contract for your thing exists

Basis risk has an extreme cousin. Sometimes there is *no futures contract at all* for the exact commodity you are exposed to — and you must hedge with a *related* contract that merely tends to move with yours. This is **cross-hedging**, and the airline is the textbook case: there is no deep, liquid futures market in jet fuel, so airlines hedge their jet-fuel cost using **crude oil** or **heating-oil/diesel** futures, which are highly correlated to jet but not identical. The hedge works because jet fuel and crude move together perhaps 90%+ of the time — but that residual 10% is a permanent cross-hedge basis the airline can never fully remove.

Cross-hedging introduces a new question: *how many* of the proxy contracts do you trade to offset your real exposure? If jet fuel moves only 0.9 dollars for every 1 dollar crude moves, hedging one-for-one *over-hedges* you; you want to scale the hedge down by that ratio. The factor that scales the hedge is the **minimum-variance hedge ratio** — roughly, the historical sensitivity of your physical price to the proxy contract's price (its regression slope, or "beta"). Hedge that fraction and you minimize the variance of the combined position; hedge one-for-one and you leave variance on the table.

#### Worked example: an airline cross-hedging jet fuel with crude

Return to the airline burning **100 million gallons** of jet fuel (~2.38m barrels of crude-equivalent), but be honest that it must use *crude* futures because jet has no liquid market. History says jet-fuel prices move about **0.92 dollars** for every **1 dollar** crude moves — a hedge ratio (beta) of **0.92**. So instead of buying the full 2.38m barrels of crude, the airline buys **0.92 × 2.38m ≈ 2.19m barrels** of crude futures at **\$80**.

- **Crude rises \$20 to \$100; jet rises 0.92 × \$20 = \$18.40.** The airline's *extra* fuel cost is 2.38m × 18.40 = **+\$43.8m**. The crude hedge gains 2.19m × 20 = **+\$43.8m**. The two match almost exactly — the 0.92 scaling lined the hedge up with the jet move rather than the crude move.
- **Had the airline hedged one-for-one (2.38m barrels)**, the hedge would have gained 2.38m × 20 = **+\$47.6m** against a fuel-cost rise of only \$43.8m — an over-hedge that left the airline **+\$3.8m** net long crude, a small *speculative* position it never intended.

The properly scaled cross-hedge neutralizes the fuel cost; the naive one-for-one hedge accidentally turns the treasury into a crude speculator. The intuition: when you hedge with a proxy, you must trade not one contract per unit of exposure but *beta* contracts per unit, or the mismatch quietly becomes a bet.

## Common misconceptions

**"A producer who sells futures is betting the price will fall."** No — the producer is *removing* a bet they already have. The producer is naturally long the commodity through their business; selling futures cancels that to roughly flat. If anything, the un-hedged producer is the one making a giant bet (that price will *rise* or at least hold). Hedging is the opposite of speculation: it takes a position off, it does not put one on.

**"A hedge that lost money was a bad hedge."** The single most expensive myth in corporate finance. A hedge is symmetric variance-reduction; by construction, when the spot price moves your way the derivative leg loses, and that loss is the premium you paid for certainty. Southwest's 2008 hedges "lost" nine figures in Q4 as oil crashed — and they were still a brilliant program, because the airline's fuel cost came in on plan. Judge a hedge by variance removed, not by one leg's P&L.

**"Hedging lowers your average cost / raises your average price."** Mostly false. Futures are roughly fairly priced, so over many cycles a hedger pays about the same average price as a non-hedger. What hedging changes is the *spread* of outcomes, not the center. A treasurer who pitches hedging as a way to "beat the market on price" is selling the wrong thing — and will be fired the first year it doesn't.

**"A hedge perfectly offsets the price."** Only the *benchmark* price. Your physical differs from the contract by grade, location, and timing, so a **basis** residual always remains. The hedge turns a \$50 violent risk into a \$4–6 manageable one — a fantastic trade, but not a perfect one. In a storage or logistics crunch, basis can blow out and surprise a treasurer who assumed the hedge was airtight.

**"Hedging is free / costless."** Never. A flat futures hedge costs you the upside (and ties up cash in *margin* — see below). Options cost a premium. A "zero-cost" collar is zero *cash* cost but very much not free: you pay by surrendering the gains above the call strike. Every hedge has a price; the only question is whether it is paid in premium, in upside, or in margin.

## How it shows up in real markets

We have already walked the canonical cases — Southwest's fuel hedges (consumer, brilliant), Barrick's gold forwards (producer, over-hedged into a bull), the corn farmer and the bank loan (producer, the floor that unlocks credit). Two more mechanics that bite real treasurers are worth making concrete, because they are where good hedges go wrong operationally even when the *strategy* was right.

### Hedge accounting: making the books reflect the economics

Here is a trap. The physical commodity a company will sell or buy is usually *not* on the balance sheet at market value — the farmer's growing corn, the airline's future fuel purchase, the miner's ore in the ground are off-balance-sheet or carried at cost. But the *derivative* hedge **is** marked to market every quarter under accounting rules. So a hedge that is doing its job economically can create wild, lopsided swings in reported earnings: the derivative leg's mark moves every quarter while the physical it offsets sits silent. In the down-move year, the airline's fuel hedge shows a fat *loss* on the income statement while the cheaper fuel it offsets does not show a corresponding gain until later. The economics net to zero; the *accounting* looks like the treasurer lost a fortune.

The fix is **hedge accounting** — a special election (under US GAAP's ASC 815 or IFRS 9) that lets a company defer the derivative's mark-to-market into the same period as the underlying transaction it hedges, so the two finally line up. It is administratively painful (you must formally document the hedge relationship and prove it is "effective"), which is exactly why some firms skip it and then have to *explain* huge non-cash hedging swings to confused investors every quarter. When you read a commodity company's filings and see "unrealized loss on derivatives," do not panic: check whether there is an offsetting physical position the accounting just hasn't recognized yet.

### Margin calls: the hedge that can bankrupt you before it pays off

The nastier operational risk. A futures hedge requires posting **margin** — collateral with the clearinghouse — and that margin is **marked to market daily**. If you are a producer short futures and the price *rises*, your hedge is doing exactly what it should (your physical is now worth more), but your short futures position is *losing* on a mark-to-market basis, and you must wire cash to cover the margin call **today** — long before you sell the physical that offsets it. This is a *timing* mismatch between cash out (margin, now) and cash in (physical sale, later).

In a violent rally this can be lethal. The textbook disaster is **Metallgesellschaft** in 1993: its US subsidiary had sold long-dated fixed-price oil contracts to customers and hedged with short-dated futures it kept rolling. When oil *fell*, the futures hedge (here long, to offset the short physical) generated enormous margin calls in cash, while the offsetting gains on the long-dated customer contracts were locked up and would only be realized over years. The hedge was economically sound but it bled cash *now* faster than the parent could fund, and the company nearly collapsed, taking losses estimated above **\$1 billion** before the position was liquidated — arguably at the worst possible moment. The 2022 European energy crisis produced echoes: utilities that had sold power forward and hedged were hit with tens of *billions* of euros of margin calls as gas and power spiked, and several needed state-backed liquidity lines to avoid being forced out of perfectly sound hedges. The lesson: a hedge removes *price* risk but can *create* **liquidity** risk, and a treasurer must size the position to survive the margin calls on the way to the payoff.

### Reading the crack: the refiner who hedges a margin, not a price

One special consumer-producer hybrid deserves its own mention because it hedges neither a pure cost nor a pure revenue, but a *spread*. A refiner buys crude and sells refined products (gasoline, diesel, jet fuel). It is short crude (an input) *and* long products (its output) at the same time. What it actually cares about is the **gap** between the two — the refining margin, proxied by the **3-2-1 crack spread** (buy 3 barrels of crude, sell 2 of gasoline and 1 of distillate). A refiner can hedge the whole margin in one move by going long the crack: buying crude futures and selling product futures together.

![Three two one crack spread annual average showing refining margin with the 2022 spike](/imgs/blogs/hedging-for-producers-vs-consumers-the-two-sides-of-the-trade-7.png)

The chart shows the crack margin a refiner is trying to lock. It sat near a "normal" **\$18/bbl** through 2018–2021, collapsed to **\$11** in the 2020 demand crash, then exploded to **\$38** in 2022 when war created a shortage of *products* specifically (not just crude). A refiner staring at a \$38 margin — historically extreme — would rush to lock it by selling forward cracks, guaranteeing that fat margin against the near-certain mean-reversion that followed (the crack fell back to \$28 in 2023 and \$21 in 2024). The intuition: a refiner does not hedge the oil price up or down — it hedges the *processing margin*, because that is the actual thing its business earns. (For the full mechanics of how a barrel becomes products and why the crack behaves the way it does, see [refining and crack spreads](/blog/trading/commodities/refining-and-crack-spreads-turning-crude-into-products).)

#### Worked example: a refiner locking the crack on 1 million barrels of throughput

A refiner will process **1,000,000 barrels** of crude next quarter. The 3-2-1 crack is at an unusually rich **\$30/bbl** and the refiner wants to lock it before it mean-reverts. It sells the crack forward (buys crude, sells products) on the full 1m barrels at \$30.

- **Crack collapses to \$15** (products fall faster than crude). The refiner's *physical* margin on the run is only ~\$15/bbl = **\$15m**. But the short crack hedge gained (30 − 15) × 1,000,000 = **+\$15m**. Net realized margin = 15 + 15 = **\$30m**, an effective **\$30/bbl** — locked.
- **Crack widens to \$45** (a product shortage). The physical margin is a gorgeous ~\$45/bbl = **\$45m**, but the short crack hedge *lost* (30 − 45) × 1,000,000 = **−\$15m**. Net = 45 − 15 = **\$30m**, again an effective **\$30/bbl**.

The refiner banked **\$30m** of margin either way — exactly the rich number it spotted and grabbed. In the second world the hedge "lost" \$15m and the refiner watched an un-hedged competitor earn \$45m. But the refiner converted an uncertain, mean-reverting margin into a locked one and could finance its turnaround and capex against it. The intuition: hedging a spread is the same logic as hedging a price, applied to the *difference* a business actually earns rather than the level of any single commodity.

## The takeaway: how to read corporate hedging

You now have the machine. Here is how to *use* it — as an investor reading filings, as an analyst sizing a company's risk, or as anyone trying to understand why two firms in the same business can have wildly different fates in the same price environment.

**Find the natural position first.** Before you read a word about hedges, ask: is this company a *producer* (long the commodity, fears low prices) or a *consumer* (short the input, fears high prices)? An oil E&P, a miner, a farmer, a gas producer is long. An airline, a food company, a utility, a refiner, a packaging firm is short its key input. Everything about how it *should* hedge follows from that one classification — producers sell/floor, consumers buy/cap.

**Read the hedge book as a statement of conviction.** A producer that is heavily hedged for years out has chosen certainty over upside — it is, in effect, a bond on the commodity, with the operational upside stripped away (the Barrick lesson). A producer that is *un-hedged* has chosen — knowingly or not — to be a **leveraged directional bet** on the commodity, and you are buying that bet when you buy the stock. Neither is wrong, but you must know which one you own. Many investors buy a gold miner or a shale driller *for* its leverage to the commodity and would be furious to learn management hedged it away; others want the steady cash flow of a hedged producer. Match the hedge policy to why you are there.

**Treat an un-hedged consumer as a hidden short.** An airline that does not hedge fuel, a chemical company that does not hedge gas, a baker that does not hedge wheat — each is running a large, un-disclosed short position in a commodity it does not control. In a calm year it looks like a great cost-discipline story; in a spike year it is the first to blow up. When oil ran in 2008 and again in 2022, the airlines that survived best were the ones with hedges on; the un-hedged ones took the full hit. "We chose not to hedge" is a *position*, not the absence of one.

**Distinguish a losing leg from a losing program.** When you see a big "loss on derivatives" in a filing, do not flinch — look for the offsetting physical. A producer's short-futures loss in a rally is *paired* with a physical it now sells higher; a consumer's long-futures loss in a crash is *paired* with cheaper input. The program worked; the accounting just shows one half at a time (unless hedge accounting is elected). The genuinely worrying cases are different: hedges that *destroyed the exposure investors wanted* (over-hedging), or hedges whose *margin calls* threatened the firm's liquidity before they paid off (Metallgesellschaft, the 2022 utilities). Those are the failure modes — not a leg that lost money doing its job.

**Watch the line between hedging and speculation.** The bright line is your *natural position*. A producer selling futures up to its expected output is hedging; a producer selling *more* than it produces is now net short and *speculating*. A consumer buying calls on its fuel burn is hedging; buying ten times its burn is a punt. The instruments are identical — futures, puts, calls, collars — and the only thing that separates the hedger from the gambler is whether the paper position *offsets* a physical one or *adds* a new one. The cleanest tell that a "hedging" desk has drifted into speculation is that its derivative book is bigger than its physical exposure, or in the wrong direction.

The deepest point ties back to this series' spine. A commodity price is a physical thing — a barrel, a tonne, a bushel — forced through a financial contract. The producer and the consumer sit on opposite ends of that contract, each handing the other the risk they cannot bear: the producer passes off the fear of low prices, the consumer the fear of high ones, and the curve, the basis, and the margin mechanics decide the exact terms. A hedge is the act of stepping *out* of that price exposure on purpose — trading the wide gamble for the known band — and the firms that understand this build businesses that mine, fly, bake, and refine through any price environment, while the un-hedged ones discover, one spike at a time, that they were running a leveraged commodity trade the whole time. Read a company's hedge book and you are reading whether it knows which business it is actually in.

## Further reading & cross-links

- [The four players: producers, consumers, hedgers, and speculators](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators) — who sits on the other side of the hedge, and who gets paid to carry the risk.
- [Refining and crack spreads: turning crude into products](/blog/trading/commodities/refining-and-crack-spreads-turning-crude-into-products) — the full mechanics behind the refiner-margin hedge in the last worked example.
- [The forward curve: the most important chart in commodities](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — contango vs backwardation, the shape that decides the cost of every roll-based hedge.
- [Contango vs backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — why rolling a hedge can bleed or earn carry depending on the curve.
- [Hedging a portfolio with options: protective puts, collars, and tail risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) — the same put/call/collar machinery applied to a stock portfolio.
- [What sets an option's price: the five inputs and the intuition](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition) — where the premium on the producer's put or consumer's call comes from.
- [Gold futures: COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — how gold producers hedge, and why the monetary metal is a special case.
- [Is gold money, a commodity, or a currency](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) — the framing that explains why over-hedging a gold miner was the Barrick mistake.
