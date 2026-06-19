---
title: "Cash-and-Carry and Storage Arbitrage: Locking In the Curve"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The trade that enforces the forward curve and pays you to store: buy the physical spot, sell a richer future against it, store the barrel, and bank the locked-in difference at delivery. Why this caps contango at full carry, why super-contango sends supertankers to sea, and why the limit is physical tank space."
tags: ["commodities", "cash-and-carry", "storage-arbitrage", "contango", "forward-curve", "floating-storage", "negative-oil", "no-arbitrage", "crude-oil", "trading-houses"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — When the forward curve slopes up more steeply than the cost of holding a barrel, you can buy the physical today, sell a future against it, store it, and pocket the locked-in difference with no price risk. That trade — the **cash-and-carry** — is the machine that enforces the shape of the curve.
>
> - **The trade:** buy spot, sell the future, store, finance, deliver. Your profit is `future − spot − storage − financing`, a number known the day you put it on. No later price move can take it away.
> - **Why it matters:** the trade only pays when contango exceeds the cost of carry, so doing it *caps* contango at full carry. The curve cannot get steeper than the cheapest available storage — the arbitrage drags it back.
> - **When storage gets scarce, the prize explodes.** In April 2020 the contango blew so far past tank rates that traders chartered **supertankers as floating storage** — and when even the sea filled, the front-month WTI contract collapsed to **−\$37.63**, the first negative oil print in history.
> - **The reverse trade is hard.** You can buy and store a barrel, but you cannot easily *borrow and short* a physical barrel — so backwardation has no clean arbitrage floor the way contango has a clean ceiling.
> - The one number to remember: the cash-and-carry caps contango at the **cost of storage**, and the hardest cap of all is **physical tank space**. When Cushing filled in April 2020, the cap broke and oil went below zero.

On the morning of 20 April 2020, a trader watching the May WTI crude futures contract saw something no screen had ever shown: the price was falling toward **zero**, and then it kept going. By the 2:30 p.m. settlement the contract printed **−\$37.63** a barrel. Not a typo, not a glitch — a negative price. For one strange afternoon, the seller of a barrel of oil had to *pay the buyer* nearly forty dollars to take it off their hands. The whole financial press reached for the word "impossible," and yet there it was, settling on the tape.

The thing that broke that day was not supply and demand in the loose way headlines use those words. What broke was a piece of plumbing — a quiet, relentless arbitrage that, in normal times, you never see because it works so smoothly. That arbitrage is the **cash-and-carry trade**, and it is the single mechanism that ties the paper price of a futures contract to the physical reality of a barrel sitting in a tank. It is the reason a forward curve cannot get *too* steep, the reason traders sometimes get *paid* to store oil, and — when it finally runs out of room — the reason a price can fall straight through zero. This post is about that machine: how it works, who runs it, the dollar math of locking in the curve, and the hard physical wall that, on one April afternoon, it slammed into.

![The cash-and-carry loop: buy the physical, sell a future, store, deliver, and bank the locked-in spread](/imgs/blogs/cash-and-carry-and-storage-arbitrage-locking-in-the-curve-1.png)

## Foundations: what the cash-and-carry trade actually is

Let us build the idea from absolute zero, with no finance vocabulary assumed.

You already know two prices for a commodity. The **spot price** is what a barrel costs *right now* — cash today, barrel today. The **futures price** is what someone will pay you for a barrel delivered at a fixed future date — you agree the price today, but the barrel and the money change hands later. A whole strip of these futures prices, one for each delivery month stretching out into the future, is the **forward curve**. When the curve slopes *upward* — barrels for later delivery cost more than barrels today — we call that **contango**. When it slopes *downward* — the prompt barrel is dearer than the distant one — we call it **backwardation**. (Those two shapes get a post of their own in this series, [contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means); here we only need to know which way they point.)

Now suppose the curve is in contango. A barrel costs \$72 today, but a barrel for delivery one year from now is selling for \$78 on the futures market. There is a \$6 gap sitting in plain sight. Can you grab it?

Here is the trade. **Today**, you do three things at once:

1. **Buy** one physical barrel at the \$72 spot price.
2. **Sell** one futures contract at \$78 — that is, you commit to deliver a barrel one year from now and receive \$78 for it. You lock that \$78 in *today*; it cannot change.
3. **Store** the barrel you just bought in a tank for the year, and **finance** the cash you tied up buying it.

**One year later**, you simply take the barrel out of your tank and hand it over to satisfy the futures contract you sold. You receive the \$78 you locked in a year ago. Your profit is the \$78 you collect, minus the \$72 you paid for the barrel, minus whatever the storage and financing cost you over the year.

Notice what is *not* in that calculation: the price of oil a year from now. It does not appear anywhere. Whether oil is at \$40 or \$140 next April, you deliver *your* barrel — the one in your tank — and collect *your* locked \$78. You have no price risk. This is the defining feature of an **arbitrage**: a trade whose profit is fixed at the moment you put it on, regardless of what the market does afterward. The figure above traces the whole loop — buy spot, sell the future, store, finance, deliver, bank the spread — and the reason it deserves a diagram is that the *order* and the *simultaneity* are the whole trick. You lock both legs at once, so the gap between them is yours.

The trade has a name that describes it exactly: **cash-and-carry**. You put up *cash* to buy the physical, and you *carry* it (store it, finance it) to the delivery date. It is one of the oldest trades in commodities, older than the exchanges themselves — a grain merchant in 1850 who bought wheat at harvest, sold it forward to a miller, and stored it in a silo was doing precisely this.

### Why both legs must lock at the same instant

The single most important — and most easily missed — feature of the trade is that the two legs are locked **simultaneously**. You do not buy the barrel today and *hope* to sell it forward at a good price next week; if you did, you would carry naked price risk in the interim, and the moment you carry price risk, it stops being an arbitrage and becomes a bet. The cash-and-carry only deserves the word "arbitrage" because both prices — the \$72 you pay and the \$78 you will receive — are nailed down in the *same breath*. The future you sell is a binding obligation at a price agreed today; the barrel you buy is yours at a price paid today. The \$6 between them is not a forecast, it is a contract.

This is why the trade is sometimes described as buying the **basis**. The basis is just the difference between the futures price and the spot price — here, `78 − 72 = +6`. When you put on a cash-and-carry, you are buying that basis: you own the right to collect the \$6 gap by holding the physical to delivery, and your only job between now and then is to keep the barrel safe and pay the carry. If carry costs you \$6, you collected the basis and paid it straight back out — a wash. If carry costs you \$4, you keep \$2. The trade is a wager that *the basis is wider than your cost of carry*, and nothing else.

### The trade unwinds itself at delivery — no second decision needed

There is a quiet elegance in how the position closes. You never have to *time* an exit or guess when to sell. At delivery, the futures contract you sold simply *demands the barrel*, and you *have the barrel* — so you hand it over and the contract vanishes. The physical you bought and the paper you sold cancel against each other perfectly, by design, because a futures contract on crude is a promise denominated in exactly the thing sitting in your tank. The arbitrage is self-liquidating: it sets itself up with two locked legs and tears itself down with one delivery, and your profit is the difference, banked. This is the opposite of a directional trade, where you must decide *when* to get out and live with the consequences of getting that decision wrong.

#### Worked example: a clean cash-and-carry P&L

Take the numbers above. Spot crude is **\$72** a barrel. The one-year future trades at **\$78**. Storage in a tank costs **\$3** a barrel for the year, and financing the \$72 you tied up — borrowing it at, say, roughly 4% — costs about **\$3** for the year. Put the trade on:

```
Sell the 1-year future, locked today    +78.00   (collected at delivery)
Buy one physical barrel at spot          -72.00
Storage for the year                      -3.00
Financing the cash for the year           -3.00
                                         --------
Locked profit per barrel                   0.00
```

In this case the contango (the \$6 gap) is *exactly* eaten by the \$6 cost of carry, and you lock **\$0** — a wash. Now shift one number: suppose the future is at **\$80** instead. The same trade now locks `80 − 72 − 3 − 3 = `**`+$2`** per barrel, risk-free. On a single 1,000-barrel tank that is \$2,000 for the year; on a 2-million-barrel cargo it is \$4 million. **The intuition: the cash-and-carry only makes money when the contango is steeper than your cost of carry — and that single condition is the hinge on which this entire post turns.**

## The condition: store only when contango beats your cost of carry

Read that worked example again and you will see the rule fall out of it. The trade locks a profit equal to:

```
locked profit = future - spot - storage - financing
```

This is positive only when `future − spot` (the size of the contango) is *bigger than* `storage + financing` (your cost of carry). If the curve is in contango but only *gently* — the gap is \$2 while carry costs \$6 — you lose \$4 doing the trade, so you do not do it. If the curve is in *backwardation* (downward sloping, future *below* spot), the gap is negative and the trade is hopeless from the start. The cash-and-carry is a **contango trade, and only a steep-enough-contango trade.**

There is a deeper reason this works at all, and it is the principle of **convergence**. A futures contract that demands physical delivery must, by the moment it expires, be worth the same as the physical thing it delivers — because at that instant the paper *is* a claim on a real barrel handed over at a real place. If the expiring future and the spot price diverged at delivery, you could buy the cheaper of the two, take or make delivery, sell the dearer, and pocket the gap with no waiting at all. So the two prices are dragged together as expiry approaches, and they meet at the delivery point. The cash-and-carry exploits the *gap before* they converge: today the future is \$78 and spot is \$72, but at delivery they will be the same number, so by buying spot and selling the future you capture the entire \$6 of convergence, minus the carry it cost you to wait for it. Convergence is the guarantee that makes the locked profit real — it is *why* the barrel in your tank will be worth exactly the futures price you sold when the day comes.

It is worth pausing on *why* the cost of carry has exactly those two pieces, because they behave very differently and the whole second half of this post hinges on one of them.

- **Financing** is the cost of the money. To buy the barrel today you either spend your own cash (which could have been earning interest) or you borrow (and pay interest). Either way, tying up \$72 for a year at a 4% rate costs about \$3. Financing is roughly proportional to the spot price and the interest rate, and for a big, well-capitalized trader it is *cheap* — they borrow against the barrel itself at near-wholesale rates. This is one reason the trade is dominated by giants, a point we return to.

- **Storage** is the cost of the *place*. A barrel has to physically sit somewhere — a steel tank at a hub like Cushing, Oklahoma, a salt cavern, or the hold of a ship. The owner of that tank charges rent, plus insurance against leaks and theft. And here is the crucial property: **storage is not a smooth number.** When tanks are half-empty, storage is cheap, because tank owners are desperate to rent out idle space. As tanks fill toward capacity, the *last* available barrel of space becomes precious, and the rental rate climbs steeply. When the tanks are *completely full*, there is no more storage at *any* price — and that is not a high number, it is a hard wall. Hold that thought.

The reason this matters so much is that storage cost is **convex** — it does not rise in a straight line as inventories build, it rises with an accelerating curve and then hits a vertical wall. At 50% full, an extra barrel of space costs almost nothing. At 80% full, it costs more. At 95% full, the few remaining tanks command extortionate rent. At 100%, the cost is no longer a number at all; it is infinity, because the space does not exist. Financing, by contrast, is roughly *linear* and predictable — a steady percentage of the cash tied up. So as a glut develops, the carry stack is dominated more and more by its storage slab, and that slab is the one that can run away from you. The whole drama of super-contango and negative prices is the drama of one convex cost term overwhelming everything else and then going vertical.

The deeper post in this series, [convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape), works through the full `F = S × e^((r + u − y)T)` formula behind all of this. Here we only need the plain-English version: the future *should* sit above spot by exactly the cost of carry, no more. The chart below shows what "no more" means in practice — the actual futures curve riding just above the dashed full-carry line, with the green band between them the gap a storage trader pockets.

![Contango curve sitting above the full cost-of-carry line, with the storage-pays gap shaded](/imgs/blogs/cash-and-carry-and-storage-arbitrage-locking-in-the-curve-2.png)

The dashed amber line is full carry: spot grown forward at roughly 5.5% a year for storage plus financing. The blue line is the quoted futures curve. Wherever blue sits *above* amber, the contango is steeper than carry, and a storage trader can lock the green gap. The instant that gap opens, traders pile in — and their piling in is precisely what closes it, which is the mechanism we turn to next.

## Why the trade enforces the curve: contango is capped at full carry

Here is the part that makes the cash-and-carry more than just a way to make money — it makes it the *enforcement mechanism* for the whole forward curve.

Suppose the one-year future drifts up to \$85 while spot stays at \$72 and carry stays at \$6. Now the trade locks `85 − 72 − 6 = `**`$7`** per barrel, risk-free, on as many barrels as you can store. Every trading desk with a tank and a credit line sees the same \$7 and does the same thing: they **buy spot** (which pushes the spot price *up*) and **sell the future** (which pushes the future *down*). They keep doing it — and more traders join — until the gap shrinks back to the cost of carry and the free \$7 is gone. The arbitrage is self-extinguishing: doing it destroys the very opportunity that motivated it.

The consequence is a hard ceiling. **The contango on a storable commodity cannot durably exceed the cost of the cheapest available storage**, because if it did, arbitrageurs would crush it back. The forward curve is not floating free; it is pinned, at its upper edge, by the price of a tank. When you look at a contango curve, you are looking at the market's estimate of the cost to store the stuff — financing plus the marginal tank rate — written out month by month. That is a profound thing to realize: the *shape* of the paper market is being dictated by the *physical* economics of steel tanks and ships. This is the series spine made literal — a commodity price is a physical thing forced through a financial contract, and the cash-and-carry is the gearwheel where the two mesh.

#### Worked example: the arbitrage dragging an over-rich future back

The market quotes spot **\$72**, carry **\$6** (so full carry is **\$78**), but the one-year future is trading rich at **\$84**. A trading house with spare tank space acts:

```
Borrow $72, buy one barrel at spot          -72.00  (financing already counted in carry)
Sell the 1-year future, locked              +84.00
Storage + financing for the year             -6.00
                                            --------
Locked profit per barrel                     +6.00
```

A \$6 risk-free profit per barrel is enormous in a business that fights over pennies, so the house does it on *millions* of barrels — and so does every competitor. Their collective buying lifts spot from \$72 toward \$75; their collective selling pushes the future from \$84 down toward \$81; the gap narrows from \$12 to the \$6 of carry, and the easy money evaporates. **The intuition: nobody passes a law saying "contango may not exceed storage cost" — the law enforces itself, because the profit from breaking it is exactly what erases it.**

## The carry math, laid out as a bridge

It helps to *see* the accounting as a physical stack, because the cash-and-carry is genuinely just arithmetic with a tank attached. Start at spot, add the cost of the place, add the cost of the money, and you arrive at the breakeven future — the full-carry price. Anything the actual future sits *above* that breakeven is your locked profit; anything *below* it is a loss that tells you to leave the barrel in the ground (or rather, leave it unbought).

![The carry math as a bridge: spot plus storage plus financing equals full carry, compared to the future](/imgs/blogs/cash-and-carry-and-storage-arbitrage-locking-in-the-curve-3.png)

The figure stacks the pieces. The blue **spot** column is the \$72 you pay for the barrel. On top of it the amber **storage** and **financing** slabs add the cost of carrying it — \$3 and \$3 — building up to the lavender **full carry** column at \$78. The final blue column is the **future** you actually sold at. When the future column reaches exactly the full-carry height (\$78 = \$78, as drawn), the trade is a wash. When the future column rises *above* full carry, the overhang is risk-free profit; when it falls below, you would lose money and you simply do not trade. The whole business reduces to a height comparison: *is the future taller than the carry stack?*

This bridge also makes the two cost pieces behave visibly. Financing is a thin, stable slab — it grows slowly with interest rates and barely moves day to day. Storage is the slab that *swells*: in a glut, as we are about to see, it can balloon until it dwarfs everything else, and that is the regime where the trade stops being a quiet utility and becomes the most lucrative thing in the oil market.

## Super-contango: when storing oil becomes the trade of the year

Most of the time the cash-and-carry is dull. The contango is gentle, the locked profit is a few cents a barrel, and the trade is run by a handful of giant houses as a background utility. But every so often the world produces *far more* of a commodity than it can consume *right now*, faster than it can be stored. Supply does not stop on a dime — an oil well cannot be switched off and on like a tap, and a refinery that suddenly has no customers still has crude arriving by pipeline. The surplus has to go *somewhere*, and the only somewhere is storage. When storage starts to fill, the price of a tank skyrockets, and the contango — which is pinned to that storage cost — skyrockets with it. This regime has a name: **super-contango.**

The reason a glut cannot just "clear" the way an ordinary oversupply does is that oil production has enormous inertia. A flowing oil well is not a faucet you turn off when demand dips; shutting in a well can damage the reservoir, and restarting it is slow and costly, so producers keep pumping even into a collapsing market. Refineries running on long pipeline contracts keep taking crude. The result is that when demand falls off a cliff, supply does *not* — the surplus has nowhere to go but into storage, and it goes there fast. That mismatch in speed between how quickly demand can vanish and how slowly supply can be throttled is the engine of every storage crisis. The cash-and-carry is the system's pressure-relief valve, soaking up the surplus into tanks; super-contango is what the gauge reads as the pressure climbs; and a negative price is what happens when the valve is fully open and the tank is still overflowing.

The spring of 2020 produced the most violent super-contango in history. The COVID-19 lockdowns vaporized something like a fifth of global oil demand almost overnight — planes grounded, cars parked, factories shut. But the oil kept pumping. Within weeks the world was producing millions of barrels a day that nobody wanted, and every tank on land began to fill. The forward curve, which had been in mild contango, exploded into a near-vertical contango at the front: the prompt barrel collapsed while barrels for delivery months later, when demand might recover, held up. The chart below is the WTI curve from that moment.

![April 2020 WTI super-contango curve with the spread that funded tanker charters shaded](/imgs/blogs/cash-and-carry-and-storage-arbitrage-locking-in-the-curve-4.png)

Look at the gap. The front of the curve was crushed toward \$20 (and, for one contract, below zero), while six months out it sat near \$34 — a contango of around \$14 a barrel over half a year. No tank on land rented for anything close to \$14 for six months. So the green band in that chart — the spread *above* normal carry — was pure, enormous, available profit for anyone who could find storage. And when every steel tank on land was spoken for, traders did something audacious: they went to **sea**.

### Floating storage: chartering a supertanker as a tank

A Very Large Crude Carrier (VLCC) is a ship that holds about **2 million barrels** of oil. In normal times it exists to *move* oil from the Persian Gulf to Asia. But a ship full of oil sitting *still* is also a tank — a floating one. So in April and May 2020, trading houses chartered VLCCs not to sail anywhere but to fill up, drop anchor off Singapore or the U.S. Gulf Coast, and *wait*. The oil inside was bought at the crushed spot price and sold forward at the fat futures price; the ship was the storage. This is **floating storage**, and at the 2020 peak something on the order of 200 million barrels — months of the entire world's spare oil — was bobbing offshore in idle tankers.

The trade works only when the contango covers the ship's rent. And the ship's rent is *not* cheap, because the moment everyone wants to charter tankers as storage, the day-rate for tankers explodes too — the same scarcity logic that drives the storage cost in the first place. So the floating-storage trade is a race between two soaring numbers: the contango you capture and the charter rate you pay.

#### Worked example: does the floating-storage charter pay?

A trader looks at a VLCC and the April-2020 curve. The ship holds **2,000,000 barrels**. The six-month contango is **\$14** a barrel: buy spot, sell the future six months out, capture \$14 if it costs nothing to store. The gross prize is therefore `2,000,000 × $14 = `**`$28 million`**. Now the costs:

```
Gross contango captured   2,000,000 bbl x $14    = +28,000,000
VLCC charter, ~$150,000/day x 180 days           =  -27,000,000
Financing the ~$40m of oil for 6 months (~3%)    =   -1,200,000
Insurance, port, bunkers (rough)                 =   -1,000,000
                                                   ------------
Net locked profit on the voyage                  =    -1,200,000
```

At a \$150,000/day charter rate this particular voyage *loses* about \$1.2 million — the soaring tanker rate has eaten the soaring contango. But early movers who locked ships at **\$50,000/day** before the stampede paid only `50,000 × 180 = `**`$9 million`** in charter, turning the same \$28 million gross into roughly **\$17 million** of locked profit per ship. **The intuition: floating storage is profitable only for whoever moves *first*, because the act of chartering ships drives the charter rate up to meet the contango — the arbitrage closes on the water exactly as it does on land.**

This is the cash-and-carry's enforcement mechanism operating on the ocean. The contango cannot durably exceed the cost of *the cheapest remaining storage*, and in April 2020 the cheapest remaining storage was a supertanker. As tankers filled and charter rates climbed to meet the contango, the curve's steepness was capped — by the rent of a ship.

## The hard limit: when the tanks fill, the trade breaks

Now we arrive at the wall, and at the explanation for that impossible −\$37.63 print.

Everything above describes a self-correcting system: contango opens, arbitrageurs store, the contango is capped at storage cost, balance is kept. But that system has a silent assumption baked into it — that there is *always more storage to be had at some price*. Usually that is true; you can always rent one more tank, charter one more ship, dig one more cavern. But storage is *finite*. There is a last tank. There is a last ship. And when the surplus is large enough and fast enough, you reach the day when the last cubic foot of storage is full and **there is no more at any price.** On that day the cash-and-carry trade cannot absorb one more barrel, and the mechanism that had been holding the front of the curve up simply switches off.

![Before-after of the storage limit: tanks have room versus tanks full and the prompt price collapsing](/imgs/blogs/cash-and-carry-and-storage-arbitrage-locking-in-the-curve-5.png)

The figure lays out the two regimes side by side. On the left, while tanks have room, the storage bid soaks up surplus barrels: arbitrageurs buy spot, which supports the front-month price, and contango stays capped at full carry — a normal, if oversupplied, market. On the right, the tanks are full. The storage bid vanishes. And here the futures market has a brutal feature that the spot market does not: a futures contract, at expiry, demands *physical delivery*. Whoever is still **long** the expiring contract when it settles is obligated to *take* the barrels — to have a tank at Cushing ready to receive them. If you are a financial player with no tank, you cannot take delivery, so you *must* sell your position before expiry, at any price, to anyone who can.

On 20 April 2020, the day before the May WTI contract expired, that is exactly the corner financial longs found themselves in. The delivery point for WTI is the tank farm at **Cushing, Oklahoma**, and Cushing was filling toward its ~76-million-barrel capacity with no room left to rent. Longs who could not take physical delivery had to dump their contracts, but there were almost no buyers, because anyone who *could* buy already had nowhere to put the oil. The price did not just fall — it fell *through zero*, because at that point a long holder was willing to **pay** someone to take the contract off their hands rather than be stuck owning barrels they physically could not store. That is what a negative price *is*: the price of getting rid of something you cannot hold. The contract settled at **−\$37.63**.

![WTI crude annual averages with the April-2020 negative-price milestone marked](/imgs/blogs/cash-and-carry-and-storage-arbitrage-locking-in-the-curve-6.png)

The chart puts that moment in two decades of context. The blue line is the WTI annual average — note that the *annual* average for 2020 stayed firmly positive near \$39, because the negative print was a single contract's settlement on a single afternoon, not the price of oil for the year. But the red dot below the zero line marks the −\$37.63 settlement: the day the storage trade saturated and the front-month price went where no model thought it could. (The April-2020 episode gets a full forensic treatment elsewhere in this series; here the point is narrower — *the cash-and-carry's physical limit is what made it possible.*)

#### Worked example: the tank-space limit, in barrels

Suppose a hub has **76 million barrels** of total tank capacity and is **90% full** — so **7.6 million barrels** of space remain. The surplus arriving is **2 million barrels a day**. The storage trade can keep absorbing the glut for `7.6m ÷ 2m = `**`3.8 days`**. On day four, the math changes character completely: there is *no* available storage, so the cash-and-carry — which *requires* a place to put the barrel — cannot be put on *at any contango*. The trade does not get less profitable; it becomes **physically impossible**. At that moment the front-month price is no longer anchored by storage cost and is free to fall as far as it must to force consumption or shut-ins — including below zero. **The intuition: every other cost in the cash-and-carry is a number you can pay; tank space is the one input you cannot conjure, which is why the storage wall, not the financing rate, is what ultimately breaks the curve.**

## The reverse trade: why backwardation has no clean floor

We have seen that contango has a tidy ceiling: the cash-and-carry caps it at storage cost. A natural question is whether the *opposite* shape — backwardation, where the front is dearer than the back — has an equally tidy *floor*. The honest answer is no, and the reason is deeply physical.

To arbitrage a contango, you **buy** the cheap physical and **store** it. Anyone with cash and a tank can buy a barrel — there is no obstacle. To arbitrage a *backwardation*, you would have to do the mirror image: **sell** the expensive prompt physical and **buy** the cheap future to cover later. But to sell a barrel today you must *have* a barrel today — and if you do not own one, you would have to **borrow** a barrel, sell it, and return it later, just as a stock short-seller borrows shares. The trouble is that physical commodities are far harder to borrow than shares. There is no deep, cheap lending market for "one barrel of crude" the way there is for shares of a company. The barrel is in someone's tank, committed to a refinery's schedule, expensive to move, and its owner is holding it precisely *because* they want it now. So the reverse cash-and-carry — sometimes called a *reverse carry* — is, for most players, simply not executable.

![Before-after of the forward trade being easy versus the reverse trade being blocked](/imgs/blogs/cash-and-carry-and-storage-arbitrage-locking-in-the-curve-7.png)

The figure contrasts the two. On the left, the forward cash-and-carry: triggered when the future is too *high*, executed by buying physical (anyone can), storing it, and selling forward — so contango is capped near full carry, a firm ceiling. On the right, the reverse: triggered when the future is too *low*, it would require *selling* physical you do not own, which means borrowing a barrel — costly or impossible — so there is **no floor**, and backwardation can run far past full carry. This asymmetry is why oil and other consumable commodities can spend years in deep backwardation that no arbitrage erases, while contango is held on a short leash. The only players who *can* do a reverse carry are those who already *hold* inventory — a refiner or a producer sitting on barrels can sell some prompt physical and buy the future to replace it. But they will only do so if their **convenience yield** (the value of having the stuff on hand) is low enough, and when the market is tight, it rarely is.

#### Worked example: a reverse-carry attempt that cannot get off the ground

The curve is in steep backwardation: spot **\$85**, the one-year future only **\$78**. The gap is \$7 *in your favor* if you could sell high now and rebuy low forward. You try to put on the reverse carry:

```
Sell physical spot now (need a barrel!)   +85.00   <- you do not own one
Buy the 1-year future                     -78.00
                                          --------
Apparent locked profit                     +7.00   <- only if you can source the barrel
```

To actually sell that barrel you must borrow one. But no one will lend you a physical barrel cheaply: the owner wants it for their own refinery run, and the cost to borrow-and-replace (if available at all) easily exceeds the \$7 edge. So the trade dies on the launch pad, and the \$7 backwardation persists, unarbitraged. **The intuition: contango is fenced in by the ease of buying-and-storing, but backwardation roams free because you cannot short a barrel you do not have — the market's two curve shapes are *not* symmetric, and the cash-and-carry is exactly why.**

## Who runs this trade: the trading houses with tanks and ships

A trade that requires buying millions of physical barrels, renting tank farms, chartering supertankers, and financing it all at wholesale rates is not a trade for individuals or even most hedge funds. It is the home turf of the great **physical commodity trading houses** — Glencore, Vitol, Trafigura, Gunvor, Mercuria, Cargill — firms most people have never heard of that move a staggering share of the world's raw materials. (They get a dedicated profile in [commodity trading houses](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura).)

What makes them able to run the cash-and-carry when almost no one else can is that they own — or control — the *physical* leg. They have tank farms at the key hubs, long-term charters on tanker fleets, relationships with port terminals, and the logistics arms to actually move and store a cargo. They finance positions against the barrels themselves at near-wholesale rates, so their *financing* cost is low. And they have the trading desks to lock the *paper* leg on the exchange simultaneously. The combination — physical storage plus cheap money plus a futures desk — is exactly the toolkit the cash-and-carry demands, and it is a toolkit almost impossible to assemble from scratch. In April 2020, it was these houses, not retail traders, that chartered the floating-storage fleet and harvested the super-contango. When you read that "a trader stored oil at sea and made a fortune," the trader was almost always one of these firms.

This is the practical meaning of the trade tying *paper to physical*: the people who enforce the forward curve are the people who can touch the barrel. The futures price is disciplined not by faceless market forces but by specific firms with specific tanks deciding, every day, whether the contango on the screen is worth more than the rent on their storage. The screen and the steel are linked by their balance sheets.

### What can still go wrong: the trade is "risk-free," not "free"

It is worth being precise about the word *risk-free*, because the cash-and-carry is risk-free in its *price* but not free of every hazard, and a careful trader sizes for the differences. The price is locked — that part is genuine. But three frictions sit around the locked spread, and each has bitten someone.

First, **margin risk on the short futures leg.** You sold a future; if the price of oil *rises* sharply before delivery, that short position shows a paper loss and the exchange demands variation margin — cash, posted daily — even though you will ultimately deliver your barrel and be made whole. A trader who locked a fat contango but underestimated the cash needed to fund margin calls during a price rally can be forced to unwind at the worst moment. The profit was locked; the *liquidity to hold it* was not. This is why the trade lives at firms with deep balance sheets.

Second, **quality and location basis.** The barrel in your tank must be deliverable into the specific contract you sold — the right grade, at the right delivery point. WTI futures deliver at Cushing, Oklahoma; if your barrel is the wrong crude or sitting at the wrong terminal, you face the cost of transforming or transporting it, and that cost eats the spread. The "same barrel" on the screen and in the tank must really be the same barrel.

Third, **counterparty and operational risk.** Tanks can leak, ships can be delayed, and the warehouse or terminal you rely on can fail. None of these change the locked *price*, but each can turn a paper profit into a real loss. The trade is an arbitrage in theory and a logistics business in practice — which, again, is why the firms that run it are logistics firms with trading desks bolted on, not trading desks with a spreadsheet.

## Common misconceptions

**"Cash-and-carry is a bet that oil will rise."** No — it is the opposite of a bet. The whole point is that the profit is locked the day you put it on, and the future price of oil never enters the calculation. You deliver your own stored barrel and collect your pre-agreed price. A trader running a pure cash-and-carry is *indifferent* to whether oil triples or halves. If you find yourself rooting for the price to move, you are not running a cash-and-carry; you are running a directional position.

**"If the curve is in contango, the storage trade is free money."** Only if the contango exceeds your cost of carry. A \$2 contango against a \$6 carry cost is a \$4 *loss*, not a gift. Most of the time the contango is gentle and the locked profit is a few cents a barrel — barely worth the effort except at enormous scale. The free-money days are the rare super-contango events, and those are precisely the days when storage is scarcest and hardest to secure.

**"Oil went negative because demand went to zero."** Demand fell hard, but it did not go to zero — people were still burning some oil in April 2020. Oil went *negative* because **storage** went to zero. The front-month contract demanded physical delivery into a Cushing that had no room, and longs who could not store had to pay to escape. A market with collapsed demand but ample storage would have seen a low price, not a negative one. The negative print was a *storage* event, not a demand event — which is the whole reason it belongs in a post about the cash-and-carry's physical limit.

**"You can always just do the reverse trade when oil is in backwardation."** You cannot, unless you already own inventory. The reverse carry requires selling a physical barrel you do not have, which means borrowing one — and there is no cheap, deep market to borrow physical crude. This asymmetry is why backwardation can run far steeper and far longer than contango ever can.

**"Negative prices mean the market is broken."** The −\$37.63 print was the market working *exactly as designed* under an extreme constraint. Physical-delivery futures are *supposed* to force the paper price to converge to physical reality at expiry. When physical reality was "there is nowhere to put this barrel," the paper price converged to "I will pay you to take it." Ugly, yes; broken, no. The lesson traders took was about *their own* exposure to physical delivery, not about a flaw in the exchange.

## How it shows up in real markets

**The contango ceiling, every normal day.** Most of the time you never *see* the cash-and-carry, and that invisibility is the proof it is working. The reason a crude or copper curve in contango rarely slopes up by more than the cost of storage is that the trading houses are silently arbitraging away any excess. When you read a forward curve and back out an implied storage cost from its slope, you are reading the cash-and-carry's fingerprints. The trade is the *enforcement layer* between the paper curve and the physical world, running quietly in the background of every storable-commodity market.

**The 2008–2009 contango and the great floating-storage trade.** April 2020 was not the first time oil went to sea. In the depths of the 2008–2009 financial crisis, demand cratered and the WTI curve fell into a steep contango — the front near \$35 while a year out was near \$50, a \$15 gap. Trading houses chartered tens of VLCCs as floating storage, buying cheap prompt crude and selling the rich future. It was one of the most profitable trades of the crisis for firms like Vitol and Koch, and it is the template the 2020 traders ran again, at larger scale. The same mechanism, the same firms, the same green band of contango above carry. The crucial difference between 2008 and 2020 was the *fill level* of land storage: in 2008 the contango blew out but Cushing never completely filled, so the front, while crushed, stayed positive. In 2020 the land tanks did fill, the floating fleet filled behind them, and that is the line that separates a brutal-but-orderly storage trade from a negative print. Same curve shape, same trade — but one ran into the physical wall and the other did not.

**The roll: paying the storage trader from the other side.** There is a hidden symmetry worth naming. A long-only commodity fund — an oil ETF that simply holds front-month futures and rolls them forward every month — is, in a contango market, the *natural counterparty* to the cash-and-carry trader. Each month the fund sells the cheap expiring front contract and buys the dearer next one, paying away the contango. That payment is the **roll cost**, and it is, dollar for dollar, close to the carry the storage trader collects. One side rents the tank and earns the spread; the other side holds paper and pays the spread. When you read that a long-only oil ETF "bleeds" in contango, the blood is flowing, in part, to the firms running the cash-and-carry. The two are opposite ends of the same physical fact — that holding a barrel through time costs money — which is why the roll problem and the storage trade are best understood together.

**Natural gas and the seasonal cash-and-carry.** Storable gas runs a *seasonal* version of this trade. Gas is cheap in summer (low heating demand) and dear in winter, so the forward curve is in steep seasonal contango from summer into winter. Utilities and traders buy cheap summer gas, inject it into underground storage caverns, and sell the rich winter future against it — a textbook cash-and-carry whose "delivery date" is the cold months. The cap on that seasonal contango is the cost of injecting and holding gas in a cavern, and when caverns fill in autumn, the same saturation dynamics can crush the front, just as Cushing did for oil. The numbers make it concrete: if summer gas sits near \$2.50 per million BTU and the next-winter future trades at \$4.00, the \$1.50 seasonal contango is the prize, and the trade pays as long as injection plus financing plus cavern rent comes in under that \$1.50. The summer-to-winter spread in gas *is* the cash-and-carry, drawn out across the calendar — which is why the gas curve has a repeating sawtooth shape that the oil curve does not. Each winter the stored gas is delivered and the spread is banked; each summer the cycle resets.

**The metals warehouse version.** On the London Metal Exchange, the cash-and-carry runs through the warrant-and-warehouse system: a trader can buy physical metal, store it in an LME-registered warehouse, and sell the three-month forward against it. For years a notorious queue at certain aluminum warehouses let owners earn rent while the metal sat — a storage-arbitrage game so lucrative it drew regulatory scrutiny. The principle is identical to oil; only the steel tank becomes a metal shed and the barrel becomes a tonne. Metals also show the opposite extreme beautifully: because a tonne of copper is dense, durable, and cheap to store, its carry cost is low and its curve sits *close* to full carry almost all the time — there is rarely a storage crisis in copper, because you can always find a shed. The contrast with oil and gas, where the physical is bulky and hard to store, is exactly why those markets, not the metals, produce the violent contango blowouts.

**Gold: the cleanest cap of all.** Push the logic to its limit and you reach the monetary metal. Gold is the most storable commodity there is — dense, imperishable, a fortune in a small vault at trivial cost — so its carry is almost pure financing with a sliver of storage. The cash-and-carry on gold is so frictionless that gold's curve almost never strays from full carry; the arbitrage is too easy to let it. That is why gold trades in near-permanent gentle contango and essentially never goes into the violent backwardation or super-contango that wrack oil. The same paper-versus-physical machinery runs there, just with the storage term turned almost to zero — a useful contrast laid out in the gold series' [gold futures and paper-vs-physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) post. The lesson generalizes: *how easy a commodity is to store determines how tightly the cash-and-carry can cap its curve*, which is the series' spine in one sentence.

## The playbook: how to read and use the storage trade

You will likely never charter a supertanker. But understanding the cash-and-carry changes how you read every commodity market, and here is the concrete toolkit.

**Read the contango as a storage gauge.** When a storable commodity's curve slopes up, measure the slope against a rough cost of carry (financing plus typical storage). If the contango is *near* full carry, the market is comfortably supplied and storage is doing its job. If the contango blows out *far past* carry — a super-contango — it is telling you that storage is *scarce* and filling fast. A widening contango is one of the cleanest real-time signals that a glut is overwhelming the physical system. Watch it the way you would watch a pressure gauge.

**Watch storage levels, not just price.** The headline that matters in a glut is not the price — it is the *fill level* of the key storage hubs. For oil, that means Cushing inventories and global floating-storage counts; for gas, the weekly storage report; for metals, LME warehouse stocks. When you see storage climbing toward capacity, you are watching the cash-and-carry approach its physical limit, and you should treat the front of the curve as fragile. The 2020 lesson is that the danger is not "low price" but "no room."

**Respect the asymmetry.** Expect contango to be capped and well-behaved (the arbitrage is easy), and expect backwardation to be wilder and more persistent (the arbitrage is blocked). If you are tempted to "fade" a steep backwardation on the theory that it must revert, remember there is no clean reverse-carry to force it back — backwardation can stay extreme for as long as the physical market stays tight. The same asymmetry tells you which curve shape carries information. A steep contango mostly tells you about *storage* — it is roughly the rent on a tank, a logistics number. A steep backwardation tells you about *scarcity right now* — it is the market bidding up the privilege of having the physical in hand today, and because no arbitrage caps it, the steepness is a clean read on how tight the spot market truly is. When you want to gauge real physical tightness, read the front of a backwardated curve, not the contango.

**Match the storability to the behavior you expect.** Before you form any view on a commodity's curve, ask how easy the thing is to store. A dense, durable, cheap-to-warehouse good — copper, gold — will hug full carry and rarely surprise you, because the cash-and-carry polices it tightly. A bulky, awkward, or perishable good — crude, natural gas, electricity at the extreme — has a high and convex storage cost, so its curve can swing violently and, in a glut, blow out into super-contango or even break entirely. Storability is the single property that predicts how wild a curve can get, and it is the first thing to look up, not the last.

**If you ever hold a physical-delivery future, know your exit.** The −\$37.63 print was, more than anything, a lesson about *delivery risk*. A financial player who holds an expiring physical-delivery contract with no ability to take delivery is exposed to exactly the corner the 2020 longs fell into. The practical rule that came out of it: if you trade commodity futures and cannot take physical delivery, roll or close your position *well* before expiry, especially when storage is tight. This is also why long-only commodity ETFs roll early and why the roll itself has a cost — a thread picked up in the broader mechanics of commodity index investing.

**Treat a blown-out contango as a stress signal, not an opportunity.** For most readers the practical value of a super-contango is not that you can trade it — you almost certainly cannot, lacking a tank farm — but that it is a flashing warning light about the physical system. A contango that suddenly steepens far past normal carry is the market telling you that supply is overwhelming storage somewhere, and that the front of the curve has become fragile. It often precedes producer shut-ins, refinery run cuts, and sharp moves in related equities and currencies. Reading it as a *diagnostic* — the way a doctor reads a fever — is more useful to most investors than any attempt to capture the spread directly.

**See the trade as the bridge it is.** The deepest takeaway is the one the series keeps returning to. The cash-and-carry is the precise point where the paper market and the physical market are welded together. The futures price is not free to wander away from the spot price plus storage, because a real firm with a real tank will arbitrage the gap — *until* the tanks fill, at which point the weld breaks and the paper price is hurled toward whatever number clears the physical surplus, even a negative one. Every other commodity mechanism — the roll, the convenience yield, the index — sits downstream of this one. When you understand that a forward curve is, at its core, the price of *renting space for a physical thing through time*, and that the renter eventually runs out of space, you understand both why the curve is usually so well-behaved and why, once in a generation, it does something that the morning's models swore was impossible.

## Further reading & cross-links

- [Convenience Yield and the Cost of Carry: Why the Curve Has a Shape](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape) — the full `F = S × e^((r + u − y)T)` model behind the carry math used here.
- [Contango vs Backwardation: What the Shape of the Curve Means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — the two curve shapes the cash-and-carry caps (contango) or cannot reach (backwardation).
- [The Forward Curve: The Most Important Chart in Commodities](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — how to read the strip of futures prices that this trade enforces.
- [Spot vs Futures: The Two Prices of the Same Barrel](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel) — the spot/futures distinction the whole trade exploits, and why most contracts never deliver.
- [The Four Players: Producers, Consumers, Hedgers, and Speculators](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators) — who is on the other side of the storage trade.
- [Commodity Trading Houses: Glencore, Vitol, Trafigura](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura) — the firms with the tanks and tankers that actually run this arbitrage.
- [Gold Futures, COMEX, Contango/Backwardation, and Paper vs Physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — the same paper-vs-physical mechanics on the monetary metal, where storage is cheap and full carry nearly always holds.
- [Commodities as Macro Signals: Oil, Copper, Gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) — reading a blown-out contango as a macro stress gauge.
