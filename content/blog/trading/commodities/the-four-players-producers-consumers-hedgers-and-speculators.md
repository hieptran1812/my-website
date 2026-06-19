---
title: "The Four Players: Producers, Consumers, Hedgers, and Speculators"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Who is on the other side of every commodity trade and why — producers selling forward to lock revenue, consumers buying forward to cap costs, and the speculators who get paid a risk premium to carry what the hedgers want gone, read through the Commitments of Traders report."
tags: ["commodities", "hedging", "speculation", "producers", "consumers", "commitments-of-traders", "cot-report", "risk-premium", "normal-backwardation", "positioning", "futures", "keynes"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Every commodity futures contract has two sides, and behind those two sides sit four players with completely different motives: producers and consumers who want to *get rid of* price risk, and speculators who are *paid to take it*.
>
> - **Producers** (a miner, an oil major, a farmer) **sell forward** to lock in revenue and de-risk the capital they've already sunk. **Consumers** (an airline, a food company, a utility) **buy forward** to lock in input costs. Together they are the **natural hedgers** — they want certainty, not a bet.
> - The two hedger sides rarely balance perfectly, so **speculators** (macro funds, CTAs, index funds, the retail tail) take up the slack. They are not parasites: they are the *insurance underwriters* of the commodity world, and they are paid a **risk premium** for carrying risk the hedgers shed.
> - The **Commitments of Traders (COT)** report is your weekly window into who is positioned how: *commercials* (hedgers) are usually net short, *managed money* (specs) net long. A crowded spec long is a **contrarian warning**; commercials quietly buying at a low is a **smart-money signal**.
> - The one fact to remember: hedgers and speculators are mirror images of the same trade. Read **positioning before price**, because price tells you what happened and positioning tells you who is *trapped* if the story changes.

In the summer of 2008, the price of crude oil ran from the low \$90s to an intraday high of **\$147.27** a barrel in July. Cable news found a villain fast: *speculators*. Hedge funds, the story went, were hoarding paper barrels and driving the price of gasoline through the roof out of pure greed. Congress held hearings. The phrase "excessive speculation" was repeated like an incantation. And then — within five months — the same oil price collapsed to **\$30.28**. The speculators, it turned out, could not hold the price up any more than they had pushed it there. By December 2008 the very same funds that had been "manipulating" the market to \$147 were being margin-called into oblivion on the way down.

What actually happened that year is the whole subject of this post. The price did not move because villains willed it; it moved because of a *structure* — a market with two natural sides that wanted to offload risk, a third party that was paid to absorb it, and a moment when far too much money piled onto one side of the boat. To understand any commodity price, you have to stop asking "is it going up or down?" and start asking the question every professional asks first: **who is on the other side of this trade, and why are they there?**

That is the question this post answers. We will meet the four players — the producer, the consumer, the hedger they jointly form, and the speculator — figure out exactly what each one wants, prove that they *need each other*, and then learn to read their positions in the one report that lays it all bare each week.

![The four players around a market with risk flowing from hedgers to speculators](/imgs/blogs/the-four-players-producers-consumers-hedgers-and-speculators-1.png)

This is the post that closes the foundations of the series. We have built the [physical-versus-paper idea](/blog/trading/commodities/what-is-a-commodity-the-physical-asset-that-trades-on-paper), the [forward curve](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities), and the cost-of-carry machinery. Now we put *people* into the machine — because a curve does not trade itself. Behind every point on every curve is a producer who wants out, a consumer who wants certainty, and a speculator who is being paid to stand in the gap.

## Foundations: the two sides of a contract, and why they exist

Let us build the whole idea from absolutely nothing, with no finance assumed.

A **futures contract** is a standardised promise to deliver (or receive) a fixed quantity of a commodity at a fixed price on a fixed future date. One CME WTI crude contract is a promise about **1,000 barrels** of oil. One CBOT corn contract is a promise about **5,000 bushels**. The exchange writes thousands of identical copies of each so they can be freely traded. We covered the mechanics in [spot versus futures](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel); here we only need one fact: **every contract has two sides**. For each person who is *long* (agreed to receive the commodity, betting the price goes up), there is exactly one person who is *short* (agreed to deliver it, benefiting if the price goes down). The market is, by construction, a perfect zero-sum tug of war: open interest — the count of live contracts — only exists in matched long/short pairs.

So the real question is not "what is oil worth?" It is: **who chooses to be long, who chooses to be short, and what is each one actually trying to accomplish?** The answer sorts everyone into two camps with opposite goals.

### The first camp: people who want *less* risk

Some participants are in the futures market not to make a bet but to *cancel* one they are already forced to carry. They are the people whose real-world business is exposed to a commodity price whether they like it or not.

- A **producer** owns the commodity, or is committed to producing it. A copper miner has sunk billions into a mine that will spit out copper for twenty years; a farmer has planted a corn crop that will be harvested in October; an oil major has wells already flowing. These people are *long the physical* — they will own a pile of the stuff in the future. Their nightmare is a *falling* price.
- A **consumer** needs the commodity as an input. An airline must buy jet fuel to fly; a cereal company must buy corn and wheat; a power utility must buy natural gas to run its plants. These people are *short the physical* — they will need to buy a pile of the stuff in the future. Their nightmare is a *rising* price.

Notice the beautiful symmetry: the producer fears a fall, the consumer fears a rise. They are afraid of *opposite* things. Each can make its fear go away by taking a futures position — and, crucially, by taking *opposite* positions, so each can be the other's counterparty.

A **hedge** is a futures position taken to *offset* an existing real-world exposure. It is the opposite of a bet. The whole point is to make your future cash flow *certain* — to convert "I'll get whatever the market gives me" into "I locked in a known number." This is the foundation of [hedging for producers versus consumers](/blog/trading/commodities/hedging-for-producers-vs-consumers-the-two-sides-of-the-trade), which goes deep on the mechanics; here we just need the intuition. Together, producers and consumers are the **hedgers** — the people who want to *transfer* price risk away from themselves.

Why does a producer care so much about certainty that it will *give away* upside to get it? The answer is **capital**. A copper mine is not a flexible business; it is a hole in the ground that cost three to ten billion dollars and a decade of permitting to dig, financed with debt that demands fixed interest payments whether copper is at \$4,000 or \$10,000 a tonne. The mine's owner cannot turn it off when prices dip — shutting and restarting a mine is ruinously expensive, so it keeps producing into a falling market. That combination (huge fixed costs, debt service, inflexible output) means a low price isn't merely disappointing; it can be *fatal*. Locking in a price that comfortably covers the cost of capital is what lets the project get financed in the first place. Banks often *require* a producer to hedge a chunk of future output before they'll lend against a new mine or well. So hedging is not a sideshow for producers — it is woven into how the entire extractive industry is funded.

The consumer's logic is the mirror, and it is just as existential. An airline runs on margins of a few percent; fuel is its single largest cost after labour. A doubling of jet fuel doesn't trim profit — it converts the entire airline from profitable to bankrupt, which is precisely what happened to several carriers in 2008. A bakery, a brewer, a fertiliser maker: each lives or dies on the spread between its selling price (often fixed by contract or competition) and its input cost (volatile). Locking the input cost converts a terrifying open-ended risk into a planning number the business can build a budget around. **Neither the producer nor the consumer is in the futures market to express a view on price. They are there to take price *out of the equation* so they can run a real business.** That is the deepest difference between a hedger and a speculator, and everything else follows from it.

### The second camp: people who want *more* risk (for a price)

The trouble is that the hedgers rarely balance. In a given week, the world's oil producers might want to sell far more futures than the world's oil consumers want to buy. Someone has to take the other side of that surplus, or the trade simply cannot happen. That someone is the **speculator**: a trader who has *no* underlying physical exposure and who takes a futures position purely to profit from price moves.

Speculators come in flavours:

- **Macro / discretionary funds** that bet on oil because they have a view on the global cycle.
- **CTAs** (commodity trading advisors), mostly *trend-following* algorithms that buy what's going up and sell what's going down.
- **Index investors** — pensions and ETFs that hold a long-only basket of commodities as an inflation hedge, mechanically rolling futures (more on this in [roll yield](/blog/trading/commodities/backwardation-as-a-structural-return-source-the-carry-of-commodities)).
- **The retail tail** — individuals trading oil or gold futures and ETFs from a screen.

The popular image of the speculator is a greedy gambler who destabilises markets. The reality is closer to the opposite: **the speculator is a risk-absorber**, and a market without them is a market where hedgers cannot find a counterparty. We will see exactly why — and why they get *paid* for the service — in a moment. First, fix the structure in your mind.

It is worth being precise about what each speculator *type* contributes, because they behave very differently and you need to recognise them in the data later:

- A **discretionary macro fund** is a thinking participant: it forms a view ("the global economy is reaccelerating, copper goes up") and takes a position. It can be early, contrarian, or wrong, but it is reasoning about fundamentals.
- A **CTA / trend-follower** is the opposite: it does not have a view, it has a *rule*. When the 50-day moving average crosses above the 200-day, it buys; when momentum reverses, it sells. CTAs are the participants most prone to *crowding* — because they all follow similar trend signals, they tend to be long the same things at the same time, which is exactly how a one-sided positioning extreme builds.
- An **index investor** is the strangest of the lot. A pension fund holds a long-only basket of commodity futures as an inflation hedge and a diversifier, rolling the contracts forward forever with no view at all. It is mechanically, permanently long. After 2004 this group ballooned into hundreds of billions of dollars, and we'll see how it changed the four-player balance.
- The **retail tail** trades oil, gold, and gas through futures and ETFs from a phone. Individually small, collectively a sentiment indicator — and the group most likely to be holding the bag at a top.

The common thread is that *none of them touch the physical commodity*. They are pure financial participants whose only job, from the system's point of view, is to be willing to take a position when a hedger needs a counterparty. That willingness is a service, and like any service it is priced.

The figure at the top of this post lays out the whole arrangement: producers sell into the market, consumers buy from it, and whatever risk the two hedger sides do not cancel between themselves flows out to the speculators, who collect a premium for carrying it. That single flow — *risk moving from those who don't want it to those who are paid to take it* — is the engine of the entire commodity-derivatives world. It is, almost exactly, an insurance market.

### What it actually feels like to hold a hedge

It is easy to say "the farmer sells futures" and skip the lived reality, which matters because it explains why some hedgers panic and unwind at the worst moment. When the farmer sells 10 corn contracts, he does not hand over any corn or receive the full sale price. He posts **initial margin** — a good-faith deposit, perhaps a few thousand dollars per contract — and from then on the position is **marked to market daily**. If corn *rises* the day after he hedges, his short futures position loses money on paper and the exchange pulls **variation margin** from his account that same evening; if corn falls, money flows in.

Here is the trap. The farmer's *physical* corn is gaining value when prices rise, but that gain is unrealised — it sits in the field and won't be cash until harvest. Meanwhile the *futures* leg is bleeding real cash out of his margin account every day prices climb. A producer who hedged the "right" amount can still face a brutal cash-flow squeeze if prices rise hard before delivery, because the hedge's losses are paid in cash *now* while the offsetting physical gain arrives *later*. This is exactly the mechanism that nearly destroyed several European utilities and the 2022 nickel producer we'll meet below: their hedges were economically sound but the daily margin calls demanded billions in cash before the offsetting physical value could be realised. **A hedge transfers price risk but introduces a timing-of-cash-flow risk**, and managing that liquidity is a discipline in itself.

There is one more piece of plumbing: the **basis**. The futures contract specifies a particular grade, at a particular delivery point (corn at certain river terminals; WTI at Cushing, Oklahoma). The farmer's actual corn is a slightly different grade sold at his local elevator. The gap between his local cash price and the futures price is the *basis*, and it wobbles a little even when the flat price is hedged. So a "perfect" hedge locks the *flat price* but leaves the farmer exposed to basis moves — usually small, but the reason hedgers track local basis as closely as the headline futures price.

## The producer's hedge and the consumer's hedge

The cleanest way to understand the two hedgers is to watch them act, side by side. They are mirror images: one sells forward to put a *floor* under revenue, the other buys forward to put a *ceiling* on cost.

![A producer hedge versus a consumer hedge shown side by side](/imgs/blogs/the-four-players-producers-consumers-hedgers-and-speculators-2.png)

The producer **sells** futures. Why selling? Because the producer is already *long* the physical — copper in the ground, a crop in the field. Selling a futures contract is a promise to deliver at a fixed price, so it gains value when the price falls, exactly offsetting the loss on the physical the producer owns. The two legs cancel: whatever the price does, the producer has locked in roughly today's number. That is a **revenue floor**.

The consumer **buys** futures. The consumer is *short* the physical — it will need to buy fuel or grain later. Buying a futures contract gains value when the price rises, offsetting the higher cost of the physical it must eventually purchase. That is a **cost ceiling**.

Let us make this concrete with real units and real dollars.

#### Worked example: a farmer hedging a corn crop

It is May. A Midwest farmer has planted corn that will yield about **50,000 bushels** at the October harvest. The current futures price for December corn is **\$4.20** a bushel. The farmer's break-even cost (seed, fertiliser, land, fuel) is about \$3.60 a bushel, so \$4.20 locks in a comfortable profit — *if* the price holds. But corn is volatile; a big national crop could crush it to \$3.50, wiping out the margin.

One CBOT corn contract covers **5,000 bushels**, so to hedge the full crop the farmer sells `50,000 / 5,000 = 10` December corn contracts at \$4.20. Now run the two scenarios at harvest:

```
Scenario A: price falls to $3.50/bu
  Sell physical corn:  50,000 x $3.50  =  $175,000
  Futures gain:        sold at 4.20, buy back at 3.50
                       (4.20 - 3.50) x 50,000 = +$35,000
  Total:                                  =  $210,000   (= $4.20/bu)

Scenario B: price rises to $5.00/bu
  Sell physical corn:  50,000 x $5.00  =  $250,000
  Futures loss:        sold at 4.20, buy back at 5.00
                       (4.20 - 5.00) x 50,000 = -$40,000
  Total:                                  =  $210,000   (= $4.20/bu)
```

In both worlds the farmer nets **\$210,000**, or \$4.20 a bushel, against a \$180,000 break-even — a locked profit of \$30,000. Unhedged, the farmer would have pocketed \$250,000 in scenario B but only \$175,000 in scenario A (a profit that nearly vanishes once you net out a \$180,000 cost base). The hedge throws away the upside of \$5 corn in exchange for *eliminating* the risk of \$3.50 corn. **The farmer is not trying to beat the market — only to guarantee the business survives a bad year.**

#### Worked example: an airline hedging jet fuel

Now flip to the consumer. A regional airline expects to burn **10 million gallons** of jet fuel over the next year. Jet fuel tracks crude oil closely, so the airline hedges using WTI crude futures as a proxy (this is a *cross-hedge* — the instrument isn't identical to the exposure, but it's correlated enough to take out most of the risk). Crude is at **\$76** a barrel. One contract is **1,000 barrels**. A rough rule of thumb is that one barrel of crude yields enough product to stand in for about 42 gallons of fuel, so 10 million gallons ≈ `10,000,000 / 42 ≈ 238,000` barrels ≈ **238 contracts** bought (long).

The airline's fear is that oil *rises*, because rising oil means rising jet fuel and a shredded operating margin. So it goes long crude futures at \$76:

```
Scenario A: crude rises to $96/bbl (+$20)
  Higher fuel cost hits the airline's P&L on the physical fuel it buys.
  Futures gain:  238 contracts x 1,000 bbl x ($96 - $76)
                 = 238,000 bbl x $20 = +$4,760,000
  The gain offsets the extra fuel cost; the margin is protected.

Scenario B: crude falls to $56/bbl (-$20)
  Cheaper fuel is great for the physical bill...
  Futures loss:  238,000 bbl x ($56 - $76) = -$4,760,000
  ...but the loss on the hedge gives back the windfall.
```

Either way, the airline has locked its effective fuel cost near \$76-equivalent. **It gives up the windfall of cheap oil in exchange for being protected against an oil spike that could bankrupt it** — exactly the symmetric trade-off the farmer made, just from the buying side. Southwest Airlines famously rode aggressive fuel hedges to billions in savings through the 2000s oil run-up; airlines that *didn't* hedge into 2008 got crushed.

Put the two examples together and the magic appears: **the farmer wanted to sell 10 corn contracts; the cereal company that buys his corn wanted to buy them.** When a producer's hedge and a consumer's hedge are on the same commodity, they can be *each other's counterparty* — risk simply cancels between two people who feared opposite outcomes, and no speculator is even needed. The catch is that the two sides almost never match in size or timing. That mismatch is where the speculator earns a living.

## Who hedges what: the natural sides of each market

Before we bring in the speculator, it helps to see that "producer sells, consumer buys" is not a slogan but a map you can draw for the whole commodity complex. Every market has natural sellers and natural buyers, and knowing which corporations sit on which side tells you which way the *commercial* hedging pressure leans by default.

![A grid showing which sectors sell forward and which buy forward across commodities](/imgs/blogs/the-four-players-producers-consumers-hedgers-and-speculators-5.png)

Read the grid as a who's-who of natural hedgers. An **oil major** is a seller across crude, refined products, and increasingly natural gas — it pumps the stuff and wants to lock in revenue against the capex it has sunk into platforms and pipelines. An **airline** is the textbook fuel buyer. A **copper miner** sells copper forward to underwrite a mine that took a decade and billions to build. A **food company** is a buyer of grains. A **corn farmer** is a seller. A **power utility** is a buyer of natural gas. 

The pattern is not random. **Producers tend to outnumber consumers in the futures market for raw extractive commodities** — there are a handful of giant, sophisticated oil and metals producers who hedge aggressively, versus a more fragmented, less-hedged consumer base. That structural imbalance means the *commercial* side of many commodity markets sits **net short** by default: more producers locking in floors than consumers locking in ceilings. Hold that fact — it is the key to reading the COT report later, and it is exactly why those producers exist as long-term clients of the [commodity trading houses](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura) that intermediate physical flows.

The grid also explains why some hedges are *cross-hedges* (the airline using crude as a proxy for jet fuel) and others are direct. Where a clean futures contract exists for your exact exposure, you hedge directly; where it doesn't, you reach for the most-correlated liquid contract and accept some *basis risk* — the gap between your real exposure and the proxy. We'll return to basis risk in the misconceptions section, because it is the single biggest way a "perfect" hedge goes wrong.

#### Worked example: a cereal company locking its margin

Watch a consumer hedge from the angle that actually matters to a business — the *margin*, not the input price in isolation. A cereal maker has signed supermarket contracts to deliver boxed cereal over the next year at prices that are effectively fixed. It will need **2,000,000 bushels** of corn to fill those orders. Corn futures are at **\$4.20** a bushel, and at that input cost the company's blended margin is a healthy 12%. The danger: a drought spikes corn to \$6.00, the input cost balloons, and because the *selling* price is locked by contract, the entire margin evaporates.

So the company buys `2,000,000 / 5,000 = 400` corn contracts (long) at \$4.20:

```
Locked input cost target: $4.20/bu x 2,000,000 = $8,400,000

Scenario A: corn spikes to $6.00/bu (drought)
  Physical corn now costs:  $6.00 x 2,000,000 = $12,000,000
  Extra cost vs plan:                            -$3,600,000
  Futures gain (long 400):  ($6.00 - $4.20) x 2,000,000
                            = +$3,600,000
  Net input cost:           $12,000,000 - $3,600,000 = $8,400,000

Scenario B: corn falls to $3.50/bu (big crop)
  Physical corn now costs:  $3.50 x 2,000,000 = $7,000,000
  Saving vs plan:                                +$1,400,000
  Futures loss (long 400):  ($3.50 - $4.20) x 2,000,000
                            = -$1,400,000
  Net input cost:           $7,000,000 + $1,400,000 = $8,400,000
```

In both worlds the effective corn bill is **\$8,400,000**, and the 12% margin the company promised its lenders and shareholders is intact. The cereal company, like the farmer who *grew* the corn, has chosen certainty over a gamble — and notice that this company is the *natural counterparty to the farmer's hedge*: the farmer sold 10 contracts to lock a floor, the cereal maker bought 400 to lock a ceiling, and risk flows directly between two businesses that feared opposite prices. **A consumer's hedge protects the margin, not the price — the only number that keeps the business alive.**

## Why the speculator gets paid: risk transfer as insurance

Here is the conceptual heart of the post. We have established that hedgers want to *give away* risk. Giving something away usually costs you something. So what do hedgers pay, and to whom?

Think of the futures market as an **insurance market**, because that is functionally what it is.

![Risk transfer shown as an insurance market with hedgers paying a premium](/imgs/blogs/the-four-players-producers-consumers-hedgers-and-speculators-4.png)

When you insure your house, you accept a *known* small loss every year (the premium) to avoid an *unknown* catastrophic one (the house burns down). You are not trying to make money on insurance; you are buying certainty, and you happily accept a slightly negative expected value to get it. The insurance company, on the other side, takes on your fire risk and is *paid* the premium for bearing it. Across thousands of policies, the law of large numbers makes that a profitable business — the company earns the premium as compensation for carrying risk that its individual customers desperately wanted gone.

A commodity hedge is exactly the same trade. The hedger wants certainty and will accept a slightly worse *expected* price to get it. The speculator provides the certainty — takes the open position the hedger needed offloaded — and collects the premium for bearing the price risk. **The premium is the speculator's compensation, just as it is the insurer's.**

This is one of the oldest ideas in financial economics, and it has a name. In *A Treatise on Money* (1930), **John Maynard Keynes** argued that because producers (the dominant hedgers) are net short and need someone to take the long side, they must *bribe* speculators to come in as buyers. The bribe takes the form of a futures price set slightly **below** the expected future spot price. A speculator who goes long at that discounted futures price expects, on average, to see the price drift *up* toward the higher expected spot as delivery approaches — pocketing the difference. Keynes called this condition **normal backwardation**: futures priced below expected spot, so the long speculator earns a positive expected return for providing insurance.

Two things make this subtle and worth slowing down on:

1. **"Normal backwardation" is about the *expected spot*, not the current spot.** It is not the same thing as the *price* backwardation we discussed when reading [the forward curve](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities), where the futures price sits below today's *observed* spot. Keynes's claim is that futures sit below *what spot is expected to be at expiry* — a risk premium, not a curve shape. The two can coincide, but they are different statements.
2. **The premium is small and it is an *average*.** On any single contract the speculator can lose badly — that is the whole point of being the risk-bearer. The pay-off is statistical: across many contracts and over time, the long-only commodity holder has historically earned a modest positive risk premium for being the insurer of last resort. (How big, and how reliably, is genuinely debated — the premium shrank as index funds flooded in after 2004, a story we'll return to.)

#### Worked example: a speculator capturing the risk premium

Suppose oil producers are heavily hedged short, so to clear the market the December crude futures contract trades at **\$74** while the *expected* spot price in December is **\$76** — a \$2 risk premium the hedgers are effectively paying to lay off their risk. A speculator goes long one contract (1,000 barrels) at \$74.

```
If December spot lands at the expected $76:
  Buy at 74, settle at 76:  ($76 - $74) x 1,000 = +$2,000 per contract
  This $2,000 is the captured risk premium -- pay for absorbing risk.

But the speculator is NOT guaranteed this:
  If a demand shock drops spot to $66:
    ($66 - $74) x 1,000 = -$8,000  (the insurer pays a claim)
  If a supply shock lifts spot to $86:
    ($86 - $74) x 1,000 = +$12,000 (a windfall on top of the premium)
```

The \$2,000 is the *expected* edge for standing in as the buyer the hedgers needed. Realised outcomes scatter wildly around it. **The speculator is not predicting the price; he is being paid, on average, to hold a risk that the producer was willing to pay to avoid** — precisely the insurer's bargain. Note the deep connection to commodities as a *carry* asset: this expected drift is the same engine as the roll return in a backwardated market, which is why backwardation is treated as a [structural return source](/blog/trading/commodities/backwardation-as-a-structural-return-source-the-carry-of-commodities).

This reframing matters enormously for how you think about the 2008 oil hearings. The speculators were not "extra" demand bidding up the price of a physical barrel — every long futures contract they held was matched by a short, and most never touched a real barrel. They were providing liquidity and bearing risk. The price went to \$147 because the *physical* market was genuinely tight (global demand had outrun supply) and because *positioning got dangerously one-sided* — which is the next thing we need a tool to measure.

### Why the premium shrinks when too much money wants to be the insurer

Here is a second-order effect that separates the casual reader from someone who actually understands the structure. Keynes's risk premium is *supply and demand for risk-bearing capital*, not a law of nature. If hedgers' demand to offload risk is fixed but more and more speculative capital lines up to take the other side, then — like any market — the price of that service falls. More insurers competing for the same policies means cheaper premiums.

That is precisely what happened after about 2004. The "financialisation of commodities" — the explosion of long-only index funds and commodity ETFs marketed as inflation hedges — poured hundreds of billions of dollars of permanently-long, price-insensitive capital into futures markets. Suddenly there was a glut of willing insurers. The structural risk premium that long-only commodity investors had historically earned (the empirical version of normal backwardation) appears to have *compressed* in the years that followed: several markets that used to be reliably backwardated spent long stretches in contango, and the easy "carry" of being long commodities got harder to capture. The very brochures that sold commodities as a free inflation hedge helped erode the premium they were promising.

The deeper lesson is that **the four players are in a dynamic balance, not a fixed one.** If a wave of new speculative capital arrives, hedging gets cheaper for producers and consumers (good for them) but the speculator's edge thins (bad for the late arrivals). If speculative capital flees in a crisis — as it did in late 2008 when funds were margin-called — hedgers suddenly can't find counterparties at a fair price, bid-ask spreads blow out, and the cost of insurance spikes exactly when it is most needed. A healthy commodity market needs *enough* speculation to absorb the hedgers' risk, but a flood of one-directional money creates the crowded positions that make crashes violent. Speculation is neither hero nor villain; it is a quantity that can be too small *or* too large, and the COT report is how you measure where it sits.

### The speculator's second job: price discovery and liquidity

Risk-bearing is the speculator's headline role, but there are two more services worth naming because they explain why even people who dislike speculators rely on them.

The first is **liquidity**. When the farmer wants to sell 10 contracts on a Tuesday afternoon, there has to be someone willing to buy *right then*, at a price close to the last trade, without the farmer having to wait days for a matching consumer to show up. Speculators — especially market-makers and high-frequency participants — provide that immediacy. They quote a bid and an offer continuously, so any hedger can transact at any moment with a tight spread. Strip the speculators out and the hedger faces a thin, jumpy market where merely *entering* a hedge moves the price against him. The liquidity speculators provide is the difference between a hedge that costs a few cents to put on and one that costs a fortune.

The second is **price discovery**. A commodity's "fair" forward price is not handed down from on high; it is *discovered* through the collision of everyone's information and opinions in the market. Producers know their costs and inventories; consumers know their demand; macro funds know the global cycle; weather traders know the forecast. All of that knowledge gets compressed into a single number — the futures price — through the act of trading. Speculators, by betting on their information, *inject* it into the price. A market with only hedgers would have prices that reflect only physical balances; the speculators add the forward-looking, opinion-rich layer that makes the futures price a genuine forecast rather than a mere reflection of current inventory. When you read a forward curve, you are reading the aggregated, money-weighted opinion of all four players at once — and the speculators are a big part of what makes it informative.

So the speculator wears three hats: **risk-bearer** (paid a premium to insure the hedgers), **liquidity provider** (lets hedgers transact instantly at a tight spread), and **information aggregator** (forces opinions into the price). The popular caricature sees only a gambler. The structure shows a participant doing three jobs the market genuinely needs — which is why the answer to "should we ban the speculators?" has always been, on closer inspection, "then who exactly will the farmer sell to?"

## The COT report: your weekly window into who is positioned how

If the four players are the cast, the **Commitments of Traders (COT)** report is the script that tells you what each one is doing right now. Published every Friday by the U.S. Commodity Futures Trading Commission (CFTC), with data as of the preceding Tuesday, it breaks down the open interest in each major futures market by *type* of trader. It is free, it is public, and it is the single best lens on the structure we have been describing.

The modern "disaggregated" report sorts traders into buckets, but for our purposes two matter most:

- **Commercials** (officially *Producer/Merchant/Processor/User* plus *Swap Dealers*): the **hedgers** — entities that touch the physical commodity. The oil majors, the miners, the farmers' co-ops, the food companies, the banks running hedging desks for them.
- **Managed Money**: the **speculators** — the funds, CTAs, and pools trading futures for profit, with no underlying physical exposure.

(A third bucket, *Non-Reportable*, captures small traders below the reporting threshold — roughly the retail tail.)

Now recall the structural fact from the grid: in most raw-commodity markets, producers outnumber and out-hedge consumers, so **commercials sit net short** and **managed money sits net long**. Because the market is zero-sum, those two positions are roughly mirror images — every contract the hedgers are short, someone (mostly the specs) is long.

![COT-style net positioning showing commercials net short and managed money net long](/imgs/blogs/the-four-players-producers-consumers-hedgers-and-speculators-3.png)

The chart shows the textbook pattern across four markets (the magnitudes are illustrative of how a real COT report reads, not exact prints). In WTI crude, commercials are deeply net *short* (they are the producers locking in floors and the swap dealers facing index longs) while managed money is net *long*. The two bars mirror each other because they *must* — one side's short is the other side's long. The same shape repeats in corn, copper, and gold. **This is the resting state of a healthy commodity market: hedgers short, speculators long, the speculators being paid to carry the producers' risk.**

The information is not in the *direction* — commercials are almost always net short, that's their job. The information is in the **extremes and the changes**.

A useful way to make "extreme" precise is the **COT index** (sometimes called the Williams %R of positioning): take managed money's net position and ask where it sits within its own range over the last, say, three years. If the lowest net long in three years was +50,000 contracts and the highest was +400,000, then this week's +385,000 sits at `(385 - 50) / (400 - 50) = 96%` of the range — almost as crowded-long as it has been in three years. A reading above ~90% is a flashing warning that the long side is nearly maxed out; a reading below ~10% says the shorts are stretched and the squeeze fuel is loaded. Converting the raw contract count into a *percentile of its own history* is what turns the COT from a curiosity ("commercials are short, so what?") into a comparable, cross-market gauge of how one-sided the crowd has become.

#### Worked example: reading a COT extreme as a contrarian signal

Suppose you're watching WTI crude. Over the past year, managed-money net longs have averaged about **+250,000** contracts. This week the report prints managed-money net long at **+385,000** — the highest in five years — while commercials are at a record net short of **-410,000**. Price has rallied 40% over three months. What does this tell you?

```
Managed money net long:   +385,000  (record high, vs +250k average)
Commercials net short:     -410,000  (record short)
Price:                     +40% over three months

Reading:
  - Specs are crowded: nearly everyone who wanted to be long, is.
  - Who is left to BUY and push price higher? Very few.
  - Commercials (the people who know the physical market best)
    are aggressively SELLING into the strength -- hedging more
    barrels because they think the price is good to lock in.
  - If anything spooks the longs, they all sell at once and
    there is no fresh buyer underneath = a sharp reversal.

Signal: crowded spec long => contrarian WARNING (fade, or at
        least stop chasing). Commercials short => smart money
        says the upside is limited here.
```

This is *exactly* the 2008 setup. Managed money was record long into July; commercials were record short; and when the macro picture cracked, the longs had no one to sell to but each other. The price did not fall because speculation is evil — it fell because **positioning was a loaded spring, and the COT report had been flashing the warning for weeks.** The lesson: a crowded one-sided spec position is *fuel*, and fuel burns in whichever direction the crowd is forced to unwind.

The mirror case is just as useful: when managed money is at a record *short* and commercials are quietly *buying*, you have the ingredients for a squeeze higher — the specs are the ones who will be forced to cover. And the subtlest signal of all is **commercials accumulating longs at a price low**: the people who handle the physical barrel, see the real inventories, and know the cost of production are stepping in to buy. They are the closest thing the market has to "smart money," and when they lean *against* their usual short bias, it pays to notice.

## How to read the COT before you trade

Let's turn the report into a decision procedure you can actually run. The mistake beginners make is to look at the *level* ("commercials are short, that's bearish!") when the signal is in the *position relative to its own history* and in *who is doing the unusual thing*.

![A decision tree for reading the Commitments of Traders report](/imgs/blogs/the-four-players-producers-consumers-hedgers-and-speculators-6.png)

The tree walks the read top-down. Start with the weekly report. Split it into commercials and managed money. Then ask two questions:

1. **Is the spec side crowded?** Compare managed-money net positioning to its own range over the last one-to-three years. A reading near a multi-year *extreme* is the warning — not because extremes can't go further (they can, and do, painfully), but because they mark where the trade is most one-sided and a reversal would be most violent. A crowded long is a reason to stop *chasing*; a crowded short is the fuel for a squeeze.
2. **What are the commercials doing that's unusual?** Commercials are always net short, so "they're short" is noise. The signal is a *change*: commercials covering shorts or adding longs at a price low (smart money sniffing value) versus piling on shorts into a rally (hedging because they think the price is rich).

Two practical cautions, because the COT is widely misused:

- **It is a positioning gauge, not a timing trigger.** Crowded can get more crowded. The COT tells you where the *risk* is concentrated, not what happens next week. Use it to size and to avoid chasing, not as a buy/sell button.
- **It lags.** The data is as-of Tuesday, released Friday — three days stale, and it's a weekly snapshot of a market that moves every second. In a fast move it can be badly out of date by the time you read it.

Used correctly, though, the COT answers the foundational question of this whole post — *who is on the other side?* — with actual numbers. Before any commodity trade, the professional checks: where is managed money relative to its history, and what are the commercials doing? That single habit reframes a trade from "I think oil goes up" into "I think oil goes up, *and the people who handle the physical agree, and the spec crowd isn't already all-in on my side.*" The second sentence is a far better trade.

## Common misconceptions

**"Speculators drive prices up and hurt ordinary people."** This was the entire 2008 narrative, and it does not survive contact with the structure. Every long futures contract is matched by a short; speculators as a *group* are not net buyers of physical barrels. When the U.S. CFTC and academic studies dug into the 2008 spike, they repeatedly found that fundamentals — surging Chinese demand against flat non-OPEC supply — drove the move, with positioning amplifying but not causing it. The proof is the aftermath: the same "manipulating" funds could not stop the price collapsing 80% by year-end. **Speculators provide liquidity and bear risk; they cannot levitate a price the physical market won't support.**

**"Hedging means you make money when the market goes your way."** No — a hedge is designed to make your outcome *the same regardless of which way the market goes*. Re-read the farmer example: he nets \$4.20 whether corn is \$3.50 or \$5.00. A producer whose hedge "made a lot of money" *also* saw the physical commodity lose exactly that much. If a hedge has a big standalone P&L, it is doing its job (offsetting the physical), not generating profit. A hedger measuring the futures leg in isolation and cheering or panicking has fundamentally misunderstood the trade.

**"A hedge eliminates all risk."** It eliminates *flat-price* risk but leaves **basis risk** — the gap between the price of your actual exposure and the price of the contract you hedged with. The airline hedging jet fuel with crude is exposed to the crack spread (the [refining margin](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) between crude and products) moving against it; a Cushing-delivered WTI hedge doesn't perfectly cover a barrel sold on the Gulf Coast. Basis risk is usually far smaller than flat-price risk — which is why hedging is still worth it — but it is real, and it is the most common way a "perfectly hedged" company still takes a loss.

**"Commercials are always the smart money, so just follow them."** Commercials are structurally short because they are *hedging*, not because they are bearish — a miner sells forward whether or not he thinks the price will fall. So their *baseline* short position carries little directional information. The signal is only in *deviations* from their normal hedging: commercials buying at a low, or unusually light on shorts. Blindly "fading the specs and following the commercials" ignores that the commercials' default position is a hedge, not a bet.

**"Index funds are speculators piling on the long side."** Index investors (pensions, long-only commodity ETFs) are a strange hybrid. They take the *long* side like speculators, but their motive is closer to a consumer's: they hold commodities for inflation protection and diversification, mechanically, with no view. When index money flooded in after 2004, it arguably *added* to the supply of risk-bearing capital — meaning the hedgers' insurance got cheaper and the historical risk premium (Keynes's normal backwardation) shrank. The "financialisation of commodities" debate is largely about whether that flood changed the four-player balance permanently.

## How it shows up in real markets

**2008 crude: the loaded spring.** We've returned to this all post because it is the cleanest case study. By July 2008, WTI hit \$147.27 intraday with managed money at a record net long and commercials at a record net short — the COT was screaming "one-sided." The physical market was genuinely tight, so the structure wasn't *wrong*, but it had no shock absorber: when the financial crisis cracked global demand, the crowded longs unwound into a vacuum, and oil fell to \$30.28 by December. **Positioning didn't cause the top; it dictated the violence of the fall.**

![WTI crude annual average 2000 to 2025 with the spec-crowded tops marked](/imgs/blogs/the-four-players-producers-consumers-hedgers-and-speculators-7.png)

The long history of WTI marks the moments where the four-player balance tipped. The 2008 doubling-then-crash is the canonical spec-crowding top. The 2020 collapse (annual average \$39, with the infamous *negative* front-month print on 20 April) is the opposite — a demand shock that flushed the longs out entirely. And the 2022 war spike (annual average near \$95) is another run-up into which managed-money longs piled near the high before the price rolled over through 2023-24. In every case the price level is the headline, but the *positioning* is the part that told you how fragile the move was.

**2010-11 cotton and the squeeze that broke the bank.** Cotton ran from about 70 cents a pound to over \$2.00 in 2010-11 — a true backwardated squeeze driven by a Pakistani flood, a Chinese buying spree, and crowded positioning. Textile mills (the consumers) who *hadn't* hedged got annihilated; some defaulted on contracts rather than pay. It is the mirror of the 2008 oil story from the consumer's side: the player who refused to lock a ceiling paid the price when the spike came.

**2022 nickel: when the short hedge became the catastrophe.** In March 2022 the LME nickel price exploded from around \$25,000 to over \$100,000 a tonne in two days. The trigger was a giant Chinese producer (Tsingshan) that was *short* an enormous nickel hedge — a producer's textbook position — caught in a squeeze as the Russia-Ukraine war spiked prices. The "hedge" was so large and so concentrated that when it had to be covered, it moved the whole market, and the LME took the extraordinary step of cancelling trades. The lesson: even a *legitimate producer hedge* becomes a source of systemic risk when one player's position dwarfs the market's capacity to absorb it. The four-player balance assumes no single player is the whole market; nickel showed what happens when that assumption breaks. (We cover the full anatomy in the metals track.)

**Vietnamese coffee: the producer who didn't hedge.** The four players are not an American invention — they shape who wins in every commodity-exporting economy. Vietnam grows roughly 40% of the world's robusta coffee and exported around 1.35 million tonnes in 2024. When global coffee prices roughly doubled in 2024 (robusta ran from about \$2,450 to \$4,100 a tonne; arabica futures hit multi-decade highs), Vietnam's coffee *export value* leapt from about \$4.2 billion in 2023 to \$5.5 billion in 2024 even as volume fell. A windfall — but here is the four-player twist. Many Vietnamese farmers had pre-sold (forward-sold) part of their crop to local traders and international houses at the *old* lower prices, the way a producer hedges. When the price exploded, some refused to deliver and defaulted rather than hand over coffee worth far more than the contract — the same drama as the 2010 cotton mills, from the producer side. The lesson cuts both ways: a forward sale that *floors* your revenue also *caps* it, and a producer who hedges into a once-in-a-decade spike will watch the upside go to whoever took the other side. The four-player structure does not promise anyone gets rich; it promises that risk — and reward — flows to whoever agreed to carry it. (We treat the Vietnamese soft-commodity complex in detail in the [oil and gas](/blog/trading/vietnam-stocks/oil-gas-sector-vietnam-the-pvn-chain) and agriculture tracks.)

**The COT in the gold market.** Gold is the market where commercials-versus-managed-money reading is most popular, partly because gold's positioning is so visibly cyclical. Managed money chases gold momentum hard, pushing net longs to extremes near tops and net shorts near bottoms, while the bullion-bank commercials take the other side. Gold is also the case where you must remember gold is a *monetary* asset, not a consumption one — its hedgers and speculators are driven by [real interest rates and the dollar](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) rather than physical supply and demand, which is exactly why [gold gets its own treatment](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) and we contrast it against the industrial complex throughout this series.

## The takeaway: read positioning before price

Here is the thread pulled tight. A commodity price is a physical thing forced through a financial contract, and that contract always has two sides. Behind those sides sit four players: producers who sell forward to lock revenue, consumers who buy forward to cap cost, the hedgers they jointly form who want risk *gone*, and the speculators who are *paid a premium* to take it. That last transfer — risk flowing from those who fear it to those compensated for bearing it — is an insurance market, and Keynes's normal backwardation is just the insurance premium written in the language of futures prices.

So what do you actually *do* with this?

- **Always ask who is on the other side.** Before any commodity trade, name the natural hedger (is this a producer-heavy or consumer-heavy market?) and the speculative crowd. If you can't say who you're trading against and why, you don't understand the trade.
- **Check the COT before you check your thesis.** Pull the weekly report. Where is managed money relative to its one-to-three-year range? A multi-year extreme on your side is a reason to be *cautious*, not bold — the crowd is already there. Where are the commercials versus their own norm? Commercials buying a low or unusually light on shorts is the closest thing to a smart-money tell.
- **Treat a crowded spec position as fuel, not a forecast.** It doesn't tell you direction or timing; it tells you where the *violence* is stored. Crowded longs make sharp falls possible; crowded shorts make squeezes possible. Size accordingly.
- **Remember the hedger's P&L is supposed to net to zero against the physical.** If you run a business with commodity exposure, the goal of a hedge is *certainty*, not profit. Judge it by whether your overall cash flow got more stable, never by whether the futures leg made money.
- **Respect basis risk and concentration.** A hedge removes flat-price risk, not the basis; and a position large enough to *be* the market (2022 nickel) stops being a hedge and starts being a hazard.

- **Match the lens to the asset.** The four-player frame is built for *consumption and industrial* commodities, where producers and consumers have real physical exposure. It bends for gold and silver, where the "consumers" are partly investors and central banks and the price answers to real rates more than to inventories. When you read positioning in a monetary metal, remember you are reading a different game — which is exactly why we keep contrasting the industrial complex against [gold the monetary asset](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) across this series.

The deepest point is the reframe itself. Most people watch the price and ask "up or down?" The four-player lens teaches you to watch the *positioning* and ask "who is trapped if this moves?" Price tells you what already happened. Positioning tells you what *has* to happen next when the crowded side is forced to unwind. That is why the professional reads the COT first, the news second, and the headline price almost last — and why, when the cable-news anchor blames the speculators, the trader who understands the four players quietly checks whether the speculators are about to be the ones who get hurt. Learn to name the four players in any market, find them in the weekly report, and you will never again watch a commodity move without asking the only question that matters: *who is on the other side, and what do they have to do next?*

## Further reading & cross-links

- [What is a commodity: the physical asset that trades on paper](/blog/trading/commodities/what-is-a-commodity-the-physical-asset-that-trades-on-paper) — the series thesis these four players act inside.
- [The forward curve: the most important chart in commodities](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — what hedging pressure does to the *shape* of the curve.
- [Spot vs futures: the two prices of the same barrel](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel) — the contract mechanics behind every hedge.
- [Hedging for producers vs consumers: the two sides of the trade](/blog/trading/commodities/hedging-for-producers-vs-consumers-the-two-sides-of-the-trade) — the deep mechanics of the producer and consumer hedge.
- [Backwardation as a structural return source: the carry of commodities](/blog/trading/commodities/backwardation-as-a-structural-return-source-the-carry-of-commodities) — how the speculator's risk premium shows up as roll return.
- [Commodity trading houses: Glencore, Vitol, Trafigura](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura) — the merchants who intermediate the producers and consumers physically.
- [Commodities as macro signals: oil, copper, gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) — reading the complex as a barometer of the cycle.
- [How monetary policy moves commodities: real rates and gold](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) — why gold's players answer to a different master variable.
- [Energy, oil and gas: the inflation engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) — the energy sleeve in a cross-asset portfolio.
- [Case study: the 2000s China commodity supercycle](/blog/trading/cross-asset/case-study-2000s-china-commodity-supercycle) — the demand wave behind the 2008 positioning extremes.
- [Is gold money, a commodity, or a currency](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) — the monetary-versus-industrial contrast this series leans on.
- [Geopolitics, elections, and unscheduled shocks](/blog/trading/event-trading/geopolitics-elections-and-unscheduled-shocks) — how a war or outage detonates a crowded position.
