---
title: "The Forward Curve: The Most Important Chart in Commodities"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How to read the strip of futures prices across maturities — front month vs back, contango vs backwardation, seasonal humps, and the parallel-shift-vs-twist moves a trader checks before any commodity trade."
tags: ["commodities", "forward-curve", "futures", "contango", "backwardation", "term-structure", "crude-oil", "natural-gas", "calendar-spread", "roll-yield"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The forward curve is the single line that holds a commodity's whole supply-and-demand story: it is the strip of prices, quoted today, for delivery one month out, two months out, twelve months out, and beyond. Read its slope and you have read the market.
>
> - The **front month** is the nearby contract — closest to "the spot price" — and it is the most violent point on the curve. The **back** is the far-deferred, calm end.
> - An **upward** slope (back dearer than front) is **contango**: a sign of ample supply and a storage glut. A **downward** slope (front dearer than back) is **backwardation**: a sign of scarcity and a tight prompt market.
> - The curve moves two ways: a **parallel shift** (the whole level moves, slope unchanged) and a **twist** (the slope rotates, the story flips). The twist is the part that matters.
> - The one number to remember: in April 2020 the front WTI contract traded near **\$20** while a barrel a year out sat near **\$36.50** — a gap of roughly **\$16.50**, the steepest "super-contango" in modern oil, and it was screaming *the tanks are full*.

On 20 April 2020, the price of a barrel of American crude oil for May delivery did something it had never done in the 150-year history of the oil business: it went *negative*. The front-month WTI futures contract settled at **minus \$37.63**. Sellers were paying buyers — handing over money — just to make a barrel of oil someone else's problem.

The headlines screamed "oil is worthless." But a trader who looked at the *whole* curve that afternoon saw something far more precise and far more useful. The negative number was only the *front* of the curve, the contract about to expire with nowhere left to physically store the oil it represented. One year out, that same crude was changing hands around **\$36.50** a barrel. The market was not saying oil had no value. It was saying, in the only language it has — a line of prices across time — *we are out of room to put the stuff this month, but we will badly want it in a year*.

That line of prices is the **forward curve**, and learning to read it is the single highest-leverage skill in commodities. A stock has one price. A bond has a yield. But a barrel of oil, a tonne of copper, a bushel of corn — a physical thing that must be dug up, shipped, and stored — has a *whole spectrum* of prices at once, one for each future delivery date. That spectrum is the chart that every commodity desk on earth looks at before it does anything else. This post teaches you to read it.

Why is it *the* chart? Because a commodity is not a number; it is a thing that exists somewhere, in some quantity, that someone has to keep until it is used. Equities and bonds are claims on the future — abstractions that live happily on a screen. A barrel of oil is a barrel of oil: it has to sit in a tank, and the tank costs money, and if the tanks are full there is nowhere to put the next barrel. All of that physical reality — how much is around, how badly it is wanted right now, how expensive it is to hold — gets compressed into the *shape* of the forward curve. So when you learn to read the curve, you are not learning a charting trick. You are learning to read the physical balance of a real-world market off a single line. That is why the people who trade these markets professionally look at the curve first, the news second, and the headline spot price almost last.

![The forward curve contango versus backwardation, two shapes on the same axes](/imgs/blogs/the-forward-curve-the-most-important-chart-in-commodities-1.png)

## Foundations: What a forward curve actually is

Let us build the idea from absolutely nothing.

You already understand the **spot price**: the price to buy something *right now* and take it home today. A litre of petrol at the pump, a kilo of rice at the market — that is a spot transaction. Money and goods change hands on the spot.

Now add one new ingredient: a **forward agreement to deliver later**. Suppose a bakery knows it will need a tonne of wheat in six months. It can wait and pay whatever the spot price is then — risky, because the price might double. Or it can find a farmer today, agree a price now, and lock in delivery for six months' time. No wheat changes hands today; only a *promise* is exchanged. The agreed price for that future delivery is the **forward price**.

A **futures contract** is just that forward agreement, standardised and made tradable on an exchange. Instead of one bakery negotiating with one farmer, the exchange (NYMEX for oil, CBOT for grains, the LME for metals, ICE for energy and softs) writes thousands of identical contracts — same quantity, same quality, same delivery point — each one for a specific delivery *month*. A trader can buy or sell any of them. So at any instant there is not one oil price but a *menu* of them:

- The **front-month** contract (also called the *prompt* or *nearby*): the one that delivers soonest, often this month or next. Its price is the market's best proxy for "spot."
- The **second month** (M2), the **third** (M3), and so on, each delivering one month later.
- The **back** of the curve: contracts a year, two years, even five years out.

If you write all those prices down — delivery date on the horizontal axis, price on the vertical — and connect the dots, you have drawn the forward curve. It is also called the **term structure** of the commodity, exactly as bond people speak of the term structure of interest rates. Each is the same idea: *the price of the same thing, plotted against how far in the future you take delivery*.

The curve is a live, breathing object. Every one of those contracts trades independently, second by second, so the curve wriggles and reshapes all day. But on any given snapshot it has a *shape*, and that shape is the headline.

One more piece of plumbing makes the whole thing click. Each futures contract has a finite life. A WTI crude contract for, say, June delivery stops trading a few days before the month begins; after that, whoever is still long must either close the position or stand for **physical delivery** of 1,000 barrels of oil at a specific tank farm in Cushing, Oklahoma. Most participants never want the barrels — they are funds, refiners hedging, banks making markets — so they exit before that deadline. But the *threat* of delivery is what keeps the futures price honest. It is the rubber band that ties the paper contract back to the physical thing. As the contract counts down to expiry, that rubber band pulls the futures price toward the spot price of real, deliverable barrels, until on the last day they are essentially equal. This is called **convergence**, and it is the reason the front month is the most physical, and the most violent, point on the whole curve.

So a forward curve is not an abstract forecast hanging in the air. It is a ladder of *contracts*, each anchored to physical delivery on a date, each converging to spot as its date arrives. The shape of the ladder — whether the higher rungs cost more or less than the lower ones — is what the rest of this post decodes.

#### Worked example: reading a snapshot off the curve

Take the illustrative contango curve in the cover chart. Suppose the prices are:

```
month   price (USD/bbl)
M0      72.0      <- front / prompt
M2      73.4
M4      74.5
M6      75.3
M12     76.6      <- one year out
```

To "read" this you ask three questions. **First**, what is the prompt price? About \$72 — that is roughly where the physical barrel trades today. **Second**, which way does the line slope? Upward: each later month costs *more* than the one before. **Third**, how steep? From M0 to M12 the price rises \$76.6 − \$72.0 = **\$4.60**, or 4.6 / 72.0 ≈ **6.4% over a year**. So this market will pay you about 6.4% more for a barrel delivered a year from now than for one delivered today.

The intuition: an upward-sloping curve like this means the market is *happy* to defer — barrels are plentiful now, and someone is being compensated for storing them until later.

That single reading — level, direction, steepness — is the foundation. Everything else is detail layered on top.

## How to read the curve in one pass

Traders do not stare at the curve point by point. They sweep it left to right and ask a fixed sequence of questions. The figure below lays out that sweep.

![How to read a forward curve front month back month and the slope](/imgs/blogs/the-forward-curve-the-most-important-chart-in-commodities-2.png)

**The front month (M1).** This is the prompt contract, the one closest to physical reality. It is also, almost always, the **most volatile point on the curve**, and the reason is structural rather than emotional. As a contract approaches its expiry, it must converge to the physical spot price — there is no more "time" left for storage or financing to smooth things out. Anyone still holding a long front-month future near expiry is effectively committing to *take delivery of real barrels* at a real tank farm, or to close the position. So the front month carries all the prompt-market panic: a pipeline outage, a cold snap, a refinery fire, a tanker stuck in a canal — these slam the front month while the back of the curve barely flinches. (April 2020 was the extreme version: the front month went negative because holders physically could not store the oil, while contracts a year out, where storage was no longer the binding problem, stayed well above \$30.)

**The second month (M2) and the first calendar spread.** The very first thing a desk computes is M2 − M1, the **calendar spread** between the two nearest contracts. That single difference is the local slope of the curve, and it is the cleanest possible reading of "is this market tight or loose right now?" We will use it constantly. Note a subtlety: in the days right around the front month's expiry, that prompt spread gets noisy and unreliable — convergence and roll flow distort it — so practitioners often define their "prompt spread" off the *second* and *third* months once the front is within a week of dying, to keep the reading clean.

**The middle (M3–M6).** This is where most hedging and most liquidity live. A producer locking in next quarter's revenue, an airline hedging jet fuel for the coming season, a fund expressing a multi-month view — they cluster here. The middle is smoother than the front because it is one step removed from prompt panic.

**The back (M12 and beyond).** The far-deferred contracts are the calmest part of the curve. They are anchored less to today's weather and more to the market's belief about the *long-run cost of producing* the commodity — roughly, the price at which a marginal new oil well or copper mine makes sense to build. The back end moves slowly, like a heavy flywheel. There is a practical reason for the calm, too: almost nobody trades five-year-out crude, so the back of the curve is *thin*. Prices there are quoted by a handful of dealers and move in lazy steps, not the tick-by-tick frenzy of the front. When you read a long-dated price, treat it as an estimate of fair value, not a battle-tested market clearing price.

**The slope.** Sweep your eye across all of it and the dominant feature is the slope: does the line go up to the right, or down? That is the story, and it has exactly two names.

A useful habit when you first look at any curve is to mentally split it into two halves: the **prompt structure** (the first three or four months, where physical tightness and storage dominate) and the **deferred structure** (everything past a year, where long-run cost dominates). The two halves can disagree — a curve can be backwardated up front (tight right now) yet drift back up into contango further out (the market expects the tightness to resolve and supply to return). That two-part reading — *tight now, easing later*, or the reverse — is far richer than a single "is it contango or backwardation?" label, and it is exactly the nuance a desk lives on. We will see it explicitly in the gold-versus-oil contrast later: gold's curve is almost all "deferred structure" because nothing makes its prompt tight, while oil's prompt structure swings wildly with the physical balance.

## The two shapes: contango and backwardation

These two words are the entire vocabulary of curve shape, and they are worth burning into memory because every commodity conversation uses them.

**Contango** is an *upward*-sloping curve: each later delivery costs more than the nearer one. Picture the orange line in the cover chart climbing gently from \$72 to nearly \$77 across the year. The word comes from a 19th-century London Stock Exchange term for a fee paid to defer settlement; today it just means "back months dearer than front."

**Backwardation** is a *downward*-sloping curve: the prompt is the dearest barrel, and prices fall the further out you look. That is the green line in the cover chart, sliding from \$85 down toward \$79.

The everyday way to keep them straight: in **contango** you pay a premium for *patience* — the future is expensive, the present is cheap, so the market is rewarding you for waiting. In **backwardation** you pay a premium for *urgency* — the present is expensive, the future is cheap, so the market is punishing you for being in a hurry. Tight markets are impatient markets; they are backwardated. Glutted markets are patient markets; they are in contango.

Why do these shapes exist at all? The short version — and it is only the short version, because the full mechanism is a post of its own — is a tug of war between two forces:

- **The cost of carry** pushes the curve *up* (toward contango). If you buy a barrel today and hold it for a year, you pay for storage (the tank), insurance, and the financing cost of the cash tied up in it. A rational forward price must compensate the holder for all that, so distant barrels "should" cost more. Storage and financing are the spring that lifts the back of the curve. (The full cost-of-carry derivation lives in [Convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape) — read it next for the exact formula.)
- **The convenience yield** pushes the curve *down* (toward backwardation). When inventories are scarce, *physically owning* the commodity right now has value beyond its price: a refiner with crude in the tank can keep running when others are scrambling; a chocolatier with cocoa on hand never misses a production run. That benefit-of-having-it-now is the convenience yield, and when it is large enough it overwhelms the cost of carry and drags the prompt above the deferred — backwardation.

The slope you actually see is the *net* of these two springs. Loose market, cheap storage, low convenience yield → cost of carry wins → contango. Tight market, full pipelines of demand, high convenience yield → convenience yield wins → backwardation. That is the whole intuition; the algebra is in the cost-of-carry post.

![What the slope says contango means supply backwardation means scarcity](/imgs/blogs/the-forward-curve-the-most-important-chart-in-commodities-4.png)

#### Worked example: turning a calendar spread into an annualised carry rate

Reading the slope as a *rate* is what lets you compare one market to another, and it is the single most useful piece of arithmetic on the curve.

Take the contango curve again. The first calendar spread is M2 − M1. Here M1 (the front) is \$72.0 and the M2-ish point is \$73.4, so the spread is:

```
M2 - M1 = 73.4 - 72.0 = +1.40  (USD/bbl, contango: positive)
```

That \$1.40 is the cost of carrying a barrel forward by two months. To compare it against, say, a 5% interest rate or a different commodity, annualise it. The spread covers 2 months, so the per-month carry is 1.40 / 2 = \$0.70, and over twelve months that is \$0.70 × 12 = **\$8.40**. As a percentage of the \$72 front price:

```
annualised carry = (0.70 x 12) / 72.0 = 8.40 / 72.0 = 11.7% per year
```

So this curve is paying roughly **11.7% a year** to hold the back month over the front. If short-term interest rates are 5% and storage-plus-insurance runs maybe 3%, then 11.7% is *more* carry than the physical costs justify — a clue that the front is unusually depressed (a glut) rather than the back being expensive. A backwardated curve runs the same arithmetic with a negative sign and tells you the prompt is bid up by scarcity.

The intuition: converting the spread to an annual percentage lets you ask "is the market paying me *more* or *less* than it actually costs to store this?" — and the answer is the trade.

## The two springs that set the slope

It helps to make the tug of war concrete, because it explains *why* you should never read contango as a forecast. There is a simple, almost mechanical relationship that the forward price must obey, and it is worth seeing once even though the full derivation lives in the cost-of-carry post.

Think of holding a barrel from today until a delivery date a year out. If you *own the physical barrel* over that year, you incur costs and you receive benefits:

- **You pay the financing cost.** The cash you tied up buying the barrel could have earned interest; that forgone interest is a real cost of holding it. Call it the interest rate *r*.
- **You pay storage and insurance.** A tank, the rent on it, insurance against fire and leakage. Call it *u*, the storage cost rate.
- **You receive the convenience yield.** Because you have the physical barrel on hand, you can supply a customer instantly, keep a refinery running through a shortage, or avoid a stockout. That optionality is worth something. Call it *y*, the convenience yield.

For there to be no free lunch, the forward price *F* for one year out has to equal today's spot price *S* grown by the *net* of these:

```
F = S x (1 + r + u - y)
```

Read that equation and the whole behaviour of the curve falls out. The costs *r* and *u* push *F* above *S* — they lift the back of the curve into **contango**. The benefit *y* pushes *F* below *S* — it drags the back of the curve down into **backwardation**. The slope you actually observe is the sign of *(r + u − y)*. Nothing in that equation is a *forecast* of future spot prices; it is purely the arithmetic of carry. That is precisely why "contango means the market expects prices to rise" is wrong — contango is just *r + u* exceeding *y*, the ordinary state of an oversupplied market with idle storage.

#### Worked example: backing the convenience yield out of the curve

The beautiful thing about that equation is that the market hands you *F* and *S* directly, so you can solve for the one thing you cannot see — the convenience yield *y* — and read the market's hidden estimate of scarcity.

Suppose crude spot (the prompt) is *S* = \$85.00 and the one-year future is *F* = \$78.90 — the backwardated curve from the cover chart. Suppose the interest rate is *r* = 5% and storage runs *u* = 4% of the barrel's value per year. Rearrange:

```
F = S x (1 + r + u - y)
78.90 = 85.00 x (1 + 0.05 + 0.04 - y)
78.90 / 85.00 = 1.09 - y
0.9282 = 1.09 - y
y = 1.09 - 0.9282 = 0.1618  ->  about 16.2% per year
```

The market is implicitly valuing the convenience of *having a barrel on hand* at roughly **16% a year** — far above the 9% of financing-plus-storage it costs to hold one. That enormous convenience yield is what bends the curve into backwardation, and it is the market's quantified scream that *prompt barrels are scarce and worth grabbing now*. The intuition: the curve lets you read out a number — the convenience yield — that no inventory report prints directly, and a big convenience yield is the cleanest possible measure of physical tightness.

## Super-contango: when the curve screams

The cleanest demonstration of how loudly the curve can talk is the spring of 2020. As the pandemic erased global oil demand almost overnight, refineries stopped buying and the world's tanks, pipelines, and floating storage filled to the brim. With nowhere to *put* prompt barrels, the front month collapsed — to and through zero — while the back of the curve, where storage would eventually free up, held far higher.

![Super contango the April 2020 oil glut steep upward slope](/imgs/blogs/the-forward-curve-the-most-important-chart-in-commodities-3.png)

That figure plots the illustrative shape of the curve in that moment: a prompt near \$20 (we use the days around the negative print, where the curve had partly stabilised) rising to roughly \$36.50 a year out. A gap of about **\$16.50** between the prompt and the one-year — when a normal contango might be a dollar or two — is what earns the name **super-contango**. The slope was so steep that it became its own trade: charter a tanker, fill it with cheap prompt crude, sell the year-out future against it, and lock in the spread minus your storage cost. That "floating storage" play is the market's self-correcting mechanism — when the curve pays enough to store, traders store, and the glut slowly clears.

#### Worked example: when does the curve pay you to store the barrel?

This is the trade that the super-contango created, reduced to one comparison. Say you can:

- Buy a barrel of prompt crude at the front-month price of **\$20.00**.
- Sell the 12-month future against it at **\$36.50**, locking in your sale price.
- The spread you capture is 36.50 − 20.00 = **\$16.50** per barrel, guaranteed, *if* you can store it.

Now the cost side. Suppose storing a barrel for a year (tank lease or tanker charter, insurance, handling) costs about **\$8.00**, and the financing cost of the \$20 you tied up, at 5% for a year, is 20.00 × 0.05 = **\$1.00**. Total cost to carry = 8.00 + 1.00 = **\$9.00**.

```
locked-in spread        +16.50
storage + insurance      -8.00
financing of the cash    -1.00
                        --------
risk-free profit per bbl +7.50
```

A guaranteed **\$7.50 a barrel** for doing nothing but storing oil you have already sold. That is why, in April–June 2020, every available tanker on earth was hired as floating storage. The intuition: contango is the market *bidding* for storage, and when the bid exceeds the cost of storing, the curve hands a risk-free profit to whoever owns a tank.

## Seasonality: when the curve has a built-in shape

Not every wiggle in a curve is contango or backwardation. Some commodities have a *seasonal* signature baked permanently into their term structure, and natural gas is the textbook case.

![The natural gas forward curve with a winter hump from heating demand](/imgs/blogs/the-forward-curve-the-most-important-chart-in-commodities-5.png)

Demand for natural gas is overwhelmingly about *heating*. In a cold northern winter, households and power plants burn enormous volumes; in the mild shoulder months of spring and autumn, demand sags. The forward curve knows this in advance. Look at the figure: delivery contracts for December, January, and February sit at a visible **premium** — the "winter hump" — while the April–May contracts dip below the anchor. This is not a market opinion that prices will rise; it is the *calendar itself* priced into the strip. A gas contract for January is dearer than one for May for the same reason a beach hotel is dearer in July than in November: predictable, recurring demand on a known date.

Seasonality changes how you read everything. A "contango" in gas between, say, a July and a January contract is not really a storage-glut signal at all — it is mostly the winter premium. To read a seasonal curve properly you compare **the same season across years** (this January vs next January, the *calendar* spread that matters for gas) rather than adjacent months that straddle a seasonal boundary. Grains do this too: a new-crop December corn contract behaves differently from an old-crop July contract because a *harvest* sits between them, flooding the market with supply. The lesson is general: before you call a curve "contango," make sure you are not just looking at the calendar.

The grain example deserves its own beat because the seasonal break is so sharp. A corn crop is planted in spring and harvested in autumn; the December contract is the first "new-crop" month, fat with freshly harvested supply, while the July contract is "old-crop," priced off whatever is left in the bins from last year. In a year when last year's stocks are tight, the old-crop July can trade at a steep *premium* to the new-crop December — an inverted, backwardated look that has nothing to do with storage economics and everything to do with the calendar of the growing season. Reading a grain curve therefore means knowing which months are old-crop and which are new-crop before you say a single word about contango. A trader who buys the cheap new-crop December and sells the dear old-crop July is not betting on storage at all; they are betting that the *transition between crop years* is mispriced.

Electricity sits at the opposite extreme, and it makes the storage point unmistakable: power *cannot be stored* economically at grid scale. With no storage, there is no carry arbitrage to smooth the curve, so the power curve is pure expectation of supply and demand at each delivery moment — and it can be wildly shaped, with summer afternoon peaks priced at multiples of overnight troughs and cold-snap risk premia bolted on. The absence of storage is exactly why electricity behaves so differently from oil: the cost-of-carry spring that lifts oil's curve into gentle contango simply does not exist for power. It is the clearest reminder that the curve's shape is a *physical* fact — about whether and how cheaply the commodity can be stored — not a financial guess about the future.

#### Worked example: the winter premium as a number

Read the gas curve in the figure. The anchor (the flat-season level) is about **\$3.00/MMBtu**. The January contract sits near **\$4.20/MMBtu**, and the shoulder-month April contract sits near **\$2.90/MMBtu**. The winter-to-shoulder premium is:

```
January  4.20
April    2.90
        ------
premium  1.30  (USD/MMBtu)
```

So the curve is pricing a **\$1.30/MMBtu**, or 1.30 / 2.90 ≈ **45%**, premium for a January molecule over an April one. A producer with gas in a salt-cavern storage facility reads that and does the obvious: inject gas in April when it is cheap, withdraw and sell it in January when it is dear, and pocket the 45% spread minus storage cost. The intuition: a seasonal curve is a standing invitation to *time-shift* supply from the cheap season to the expensive one, and the size of the hump is the size of the reward for doing it.

## The front month's special violence

We have called the front month the most volatile point on the curve several times; it is worth understanding *why*, because that violence is both a hazard and a source of the curve's information.

The deep reason is **convergence under a deadline**. Every other point on the curve has time on its side. A contract for delivery in eight months can absorb a surprise — a cold snap, a refinery hiccup — by spreading the shock across the months of storage and financing that still separate it from delivery. The front month has no such cushion. It is days from forcing a physical settlement, so any prompt-market imbalance has nowhere to hide: it lands entirely on the front-month price. When a pipeline trips, the back of the curve shrugs (supply will be fine by the time those contracts deliver) while the front month leaps, because *those specific barrels, at that specific tank, in the next few days* are suddenly hard to get.

There is a second, more mechanical source of front-month violence: the **roll**, and the squeeze it sometimes creates. Most of the money invested in commodities — index funds, ETFs, long-only allocators — holds the front month for exposure but never wants delivery. So in a predictable window each month (for WTI, roughly the 5th to the 9th business day, the "Goldman roll" window for the big indices) all of them must sell the expiring front month and buy the next one *at the same time*. That synchronised selling of the front and buying of the second pushes the front down and the second up — it *steepens contango* purely from flow, independent of any fundamental news. Sharp traders front-run it. And on the rare occasion when the available physical supply at the delivery point is genuinely thin while a large long position is trapped near expiry, you get a **delivery squeeze**: the front month spikes violently as trapped longs scramble either to find barrels or to exit at any price. The 1980 silver squeeze and various corners in thinly-deliverable contracts are the dramatic versions; milder versions happen routinely.

For a reader of the curve, the practical lessons are three. First, *do not over-read a single front-month tick* — it carries convergence noise and roll-flow noise on top of fundamental signal. Second, *the calendar spread filters that noise*: because both legs of a spread share the prompt market, much of the idiosyncratic front-month jumpiness cancels, leaving a cleaner read of structure. Third, *respect expiry*: if you hold the front month and are not set up to make or take delivery, you must roll out before the deadline — and you will pay the contango roll cost to do so, the same cost that quietly bleeds long-only commodity funds.

#### Worked example: the monthly roll cost on a contango position

Put a number on that bleed. Suppose you hold a long crude position via the front month in a steadily contangoed market, and the curve sits like this each month when you roll:

```
front month you sell      72.00
next month you buy        73.40
roll cost per barrel       1.40
```

Every month you sell the expiring contract at \$72.00 and buy the next at \$73.40, paying away \$1.40 per barrel just to *stay invested at the same exposure* — you bought no extra oil; you only moved your position forward one month. Over a year of monthly rolls at that pace, the drag is roughly:

```
monthly roll cost   1.40
months in a year     12
annual roll drag   16.80  (USD/bbl)
as % of 72.00      23.3%
```

A **23% annual headwind** before the spot price even moves. If crude's spot price rises 10% over that year, your long-only position *still loses about 13%*, because the contango roll ate the gain and then some. This is exactly how the USO oil ETF managed to fall while oil "went up" in 2020. The intuition: in a contangoed market, time is not your friend — the curve quietly charges you rent every month you stay long the front, and the steeper the contango, the higher the rent. (The full anatomy of roll yield and how funds fight it is in the [contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) post.)

## Why the price level alone lies to you

It is worth pausing on the most common rookie mistake, because it is exactly the trap that the curve exists to defuse. Beginners watch the front-month price on TV — "oil is at \$80!" — and treat it as *the* price of oil. But the front month is one point on a curve that may be telling a completely different story underneath.

![WTI crude oil annual average price 2000 to 2025](/imgs/blogs/the-forward-curve-the-most-important-chart-in-commodities-6.png)

The chart above is the WTI front-month price over a quarter-century: the 2008 spike to nearly \$100 average, the 2014–16 shale crash, the 2020 collapse, the 2022 war spike. It is the price everyone quotes. But notice the labels: in 2008 and 2022, when prices were *high and rising*, the curve was *backwardated* — the prompt was the dearest barrel because the market was tight. In 2020, when prices were *low and falling*, the curve was in *super-contango*. The headline level and the curve shape are two different pieces of information, and the shape often matters more for what you actually earn.

Why? Because most people who "buy oil" do not take delivery of barrels — they hold futures and must continually **roll** them: as the front month nears expiry, they sell it and buy the next month to stay invested. In contango, you are perpetually selling cheap and buying dear, a small loss every month that compounds — the **roll cost** or negative roll yield. In backwardation, you are selling dear and buying cheap — a steady gain. So a long-only investor can watch the spot price of oil go *up* over a year and still *lose money*, if the curve was steeply enough in contango that the roll bled away the gain. (The famous USO oil ETF did exactly this in 2020.) The full mechanics of the roll, and how it makes or breaks a long-only commodity position, are the subject of [Contango vs backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — that is where the trading implications get the full treatment. Here the point is narrower: *the level is not the story; the shape is.*

## How the curve moves: parallel shifts and twists

Once you can read a static curve, the next skill is reading how it *changes*. There are exactly two pure ways a curve can move, and distinguishing them is the difference between a beginner and a desk trader.

![Curve moves parallel shift versus twist before and after](/imgs/blogs/the-forward-curve-the-most-important-chart-in-commodities-7.png)

A **parallel shift** moves every contract — front, middle, and back — by roughly the same amount. The whole curve floats up or sinks down, but its *slope is unchanged*: contango stays contango, backwardation stays backwardation, and the calendar spreads barely move. Parallel shifts come from broad, level-driving forces: a stronger or weaker dollar repricing all dollar-denominated commodities at once, a global demand shock, a risk-on or risk-off swing across all markets. A parallel shift changes how much money you make if you are *long the whole complex*, but it tells you nothing new about the supply-and-demand balance — the carry story is the same as it was yesterday.

A **twist** is the interesting one. Here the front and back move in *opposite directions*: the curve rotates around a pivot point. A twist *changes the slope* — and that means the supply-and-demand story has changed. The classic trigger is a *prompt-specific* shock: a refinery outage or a sudden inventory draw bids up the front month while leaving the back untouched, rotating a flat curve into backwardation. A surprise glut does the reverse, slamming the front and twisting the curve into contango. Crucially, a twist can leave the *average* price almost unchanged while completely flipping what the curve says — which is exactly why a trader who watches only the headline level misses it.

The practical upshot: **a parallel shift moves your level P&L; a twist moves your carry.** If you hold a calendar spread (long one month, short another), a parallel shift roughly cancels out — both legs move together — and you are exposed almost purely to the *twist*. That is why calendar-spread traders are, in effect, betting on the *shape* of the curve rather than the price of oil. The volatility math behind these curve moves — how the front trades at a different implied volatility than the back, and how that term structure of vol can itself be in contango or backwardation — connects directly to [the term structure of volatility and the VIX curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve).

#### Worked example: telling a shift from a twist with two numbers

You only need the front and one back contract on two days to classify the move. Suppose on Monday and Tuesday the curve reads:

```
            Mon      Tue      change
M1 (front)  80.00    83.00    +3.00
M12 (back)  78.00    81.00    +3.00
```

Both legs rose \$3.00. The slope — M1 minus M12 — was +\$2.00 on Monday (backwardation) and is +\$2.00 on Tuesday: **unchanged**. This is a pure **parallel shift**. The market got more bullish on the *level* of oil, but the supply-demand balance (the slope) did not change. A calendar-spread trader who was long M1 / short M12 made nothing — the two \$3 moves cancelled.

Now a different Tuesday:

```
            Mon      Tue      change
M1 (front)  80.00    84.00    +4.00
M12 (back)  78.00    77.00    -1.00
```

The front jumped \$4 while the back *fell* \$1 — they moved in opposite directions. The slope went from +\$2.00 to +\$7.00: backwardation **steepened sharply**. This is a **twist**, and it is a real signal — something just tightened the prompt market (a draw, an outage, a scramble for barrels). The level barely changed on average, but the *story* changed completely. The intuition: if both ends move the same way it is noise about price level; if they diverge, the curve is telling you the balance of supply and demand just shifted, and that is what you trade.

## Common misconceptions

**"The front-month price is the price of oil."** No — it is *one* point on a curve, and the most jittery one. A market quoted at \$80 front-month can be tight (backwardated, with the back at \$75) or glutted (contango, with the back at \$85), and those are opposite worlds for anyone holding the commodity over time. The front is the headline; the curve is the news.

**"An upward-sloping curve means traders expect prices to rise."** This is the most persistent error in commodities. Contango is *not* a forecast that spot will go up. It is mostly the **cost of carry** — storage plus financing — for a commodity that is currently abundant. In fact a steep contango usually accompanies a *weak, oversupplied* prompt market (April 2020 had the steepest contango in history *because* prompt oil was nearly worthless). The curve is a statement about *storage economics today*, not a prediction of tomorrow's spot. The cleanest proof: the curve was in deep contango in 2020 and oil then *tripled* over the next two years — the opposite of what the "expectations" reading would have told you.

**"Backwardation is bullish and contango is bearish."** Tempting, but backwards-looking. Backwardation describes a market that is *tight right now* (scarce prompt supply), which is often associated with high current prices — but high prices invite new supply that eventually flattens the curve. Contango describes a *loose* market now, which is often near a price bottom. If anything, persistent contango has historically marked oversold conditions and backwardation has marked tight tops. The curve shape describes the *present* balance, not the future direction.

**"All commodity curves slope the same way for the same reason."** Gas curves have a permanent winter hump from heating demand; grain curves break at harvest; metals, which are cheap to store relative to their value, spend most of their time in gentle contango; electricity, which cannot be stored at all, has wild intraday and seasonal shapes with no carry arbitrage to smooth them. The *grammar* (contango/backwardation) is universal; the *vocabulary* is commodity-specific. Read each curve against its own normal.

**"The spread between two months is just a small detail."** The calendar spread *is* the slope, and for many professionals it is the entire trade. A spread trader is long one contract and short another, so a parallel move in the whole market cancels out and only the *shape* change — the twist — pays off. Far from a detail, the spread isolates the one piece of information (the carry) that the noisy outright price buries.

**"Contango can get as steep as it wants."** Not quite — there is a natural ceiling. Because anyone can buy prompt, store, and sell forward, contango cannot persist much wider than the *full cost of storage plus financing* without traders piling into the storage trade and flattening it back down. The cap is set by physical storage capacity: in normal times spare tankage is cheap, so the ceiling sits a dollar or two above the prompt. April 2020 was the exception that proves the rule — contango blew out to roughly \$16.50 over a year precisely because *storage itself ran out*, so the usual arbitrage that caps contango could not function. Backwardation, by contrast, has *no* such ceiling: there is no way to "borrow" a barrel you do not have, so when prompt scarcity bites, the front can spike to whatever level it takes to ration demand. That asymmetry — contango capped by storage cost, backwardation uncapped — is why the most violent curve moves are almost always backwardation spikes.

## How it shows up in real markets

**April 2020 — the negative print and floating storage.** We have used this throughout, but it is worth stating as a clean case. On 20 April 2020 the May WTI contract settled at −\$37.63 while December 2020 sat above \$30 and the one-year-out near \$36.50. The curve was not predicting cheap oil forever; it was pricing a *temporary* inability to store prompt barrels. Traders who read the curve correctly chartered tankers, bought prompt crude, sold the deferred future, and captured the super-contango. Within months the prompt recovered and the curve flattened — the storage trade had done its job. Anyone who read only the front-month number ("oil is negative!") missed the entire mechanism.

**2007–08 and 2022 — backwardation at the tops.** When oil ran to \$147 in mid-2008 and again toward \$130 (Brent) after the 2022 invasion of Ukraine, the curves were *backwardated*: prompt barrels were the dearest, because the physical market was genuinely scrambling for supply. A trader watching the curve in those moments saw the tightness directly in the negative calendar spreads, long before it showed up in lagging inventory reports. The shape was the leading indicator.

**Natural gas, every winter.** The gas curve's winter hump is so reliable that an entire industry — salt-cavern and depleted-reservoir storage operators — exists to arbitrage it, injecting cheap summer gas and withdrawing it into the winter premium. When a forecast turns colder than expected, the *front* of the gas curve spikes (a twist into backwardation) while the back, where the winter is already priced, hardly moves — a textbook prompt-shock twist. The 2021 European gas crisis was this on steroids: the prompt TTF contract exploded while far-dated contracts, expecting eventual normalisation, lagged, producing a violently backwardated curve.

**Copper — the curve as an early warning on Chinese demand.** Industrial metals trade on the LME with their own term structure, and copper's is watched as a real-time gauge of the global economy ("Dr. Copper"). Through the China supercycle, as copper's annual average ran from under \$2,000/tonne in the early 2000s to nearly \$9,000 by 2011 and on to a record above \$11,000 in May 2024, the LME curve repeatedly flashed *backwardation* in the cash-to-three-month spread whenever Chinese restocking drained warehouse inventories — the prompt metal commanding a premium because smelters and fabricators needed cathode *now*. When demand cooled, warehouse stocks rebuilt and the curve slid back into contango. A metals trader reading that cash-to-3-month spread saw the turn in physical demand weeks before it showed up in the headline price or the official trade data. Same grammar as oil — backwardation equals tight, contango equals loose — applied to a different physical market.

**Gold — the curve that almost never backwardates.** Gold is the instructive contrast, and it is the reason gold gets its own series. Because gold is held as a monetary asset (vaults full of it sit idle, with no industrial user desperate for prompt delivery), its convenience yield is essentially zero. With nothing pulling the prompt up, gold's forward curve is in *near-permanent contango*, sloping up at almost exactly the cost of carry (the interest rate, since storage is cheap relative to value). Gold's curve is therefore boringly mechanical — a near-pure interest-rate carry — whereas oil's and gas's curves swing between contango and backwardation as physical tightness ebbs and flows. That difference *is* the line between a consumption commodity and a monetary one. The mechanics of the gold futures curve, COMEX delivery, and paper-vs-physical are covered in [Gold futures: COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical).

#### Worked example: spotting a tightening market before the data confirms it

Suppose you are watching crude and the prompt calendar spread (M1 − M2) over a fortnight moves like this:

```
day   M1 - M2 spread (USD/bbl)
1     -0.40   (contango: M1 below M2)
4     -0.15
7     +0.10   <- flipped positive
10    +0.35
14    +0.60   (backwardation, and steepening)
```

The spread has marched from −\$0.40 (mild contango) to +\$0.60 (clear backwardation) in two weeks, with the front-month outright price barely changed. That steady twist toward backwardation is the curve telling you the *prompt market is tightening* — inventories are drawing, buyers are reaching for nearby barrels — well before the weekly inventory report or the monthly agency data confirm it. A trader who sees this puts on a long-front / short-deferred calendar spread to ride the tightening, rather than betting on the outright price. The intuition: the *change in the spread* is a higher-signal, lower-noise reading of physical balance than the outright price, because it strips out the level moves that drown out the supply-demand information.

## Where the liquidity lives, and why it matters

One last practical layer before the checklist: the curve is not equally *tradable* at every point, and where you can actually transact shapes what you can do with your read.

Liquidity — the ease of buying or selling without moving the price — is concentrated in the front part of the curve and thins out fast toward the back. For crude oil, the first six to twelve months are deeply liquid, with tight bid-ask spreads and enormous volume; by three to five years out, only a handful of dealers quote, spreads widen, and a modest order can move the price. This matters for two reasons. First, a *shape* you spot in the illiquid back of the curve may be partly an artifact of stale or wide quotes rather than a real economic signal — treat far-dated readings with caution. Second, it constrains your trade: a view on the long-run cost of oil is hard to express cheaply because the contracts that would express it barely trade, whereas a view on the *prompt* tightness is easy and cheap to put on. This is one reason calendar-spread trading clusters in the liquid front of the curve, where both legs trade tightly and the spread itself has its own deep market.

It also explains a recurring real-market pattern: prompt shocks produce sharp, liquid, *tradable* twists in the front, while structural shifts in long-run supply (a wave of new shale, a fleet of new copper mines) show up slowly and quietly in the calm, thin back end. The front of the curve is where the market argues loudly about *now*; the back is where it mutters its slow-changing beliefs about the future. A skilled reader weighs them differently — leaning on the front for actionable signal and on the back for context.

A final habit worth building: read the curve in *spreads*, not just levels, even when you are only trying to understand the market rather than trade it. Pull up M1, M3, M6, and M12, and write down the three spreads between them — the prompt spread (M1−M3), the middle (M3−M6), and the deferred (M6−M12). A curve can be in backwardation across the front spread (tight now) yet contango across the deferred spread (loose later), and that "tight now, easing later" profile is a completely different story from a curve that is backwardated all the way out. Three numbers, scribbled in seconds, capture the entire term structure better than staring at the line — and they are exactly the numbers a desk quotes to one another when they describe "how the curve looks today."

## The playbook: how a trader reads the curve before any trade

Here is the checklist a commodities desk runs — explicitly or by reflex — before placing *any* trade. Make it your habit and you will never again be fooled by a headline price.

1. **Read the level, but do not stop there.** Note the front-month price, then immediately ask: what is the *shape*? The level is the headline; the shape is the story underneath it.

2. **Classify the slope: contango or backwardation?** Up to the right means abundant supply and a market paying you to defer. Down to the right means scarcity and a market punishing you for waiting. One glance tells you which world you are in.

3. **Annualise the front spread.** Compute M2 − M1 (or M1 − M12) and turn it into a percent-per-year carry, as in the worked examples. Compare it to interest rates and known storage costs. A carry far *above* what storage justifies flags a depressed front (glut); a deeply negative carry flags a bid-up prompt (scarcity). This one number is your fastest read of physical tightness.

4. **Adjust for seasonality.** If you are in gas, grains, or power, do not mistake the calendar for a signal. Compare the same season across years, not adjacent months that straddle a harvest or a heating season. Strip the predictable hump out before you call the curve tight or loose.

5. **Decide what you are actually betting on.** An *outright* position (just long or short the front) is a bet on the price *level* — you will live and die by parallel shifts. A *calendar spread* is a bet on the *shape* — it cancels the level and isolates the twist. Knowing which exposure you want is the difference between trading the news and trading the noise.

6. **Watch the spread's change, not just its value.** The most reliable early signal in commodities is a calendar spread *trending* — a steady twist from contango toward backwardation (tightening) or the reverse (loosening). That trend usually leads the official inventory data by days or weeks.

7. **Respect the front month's volatility.** Near expiry, the prompt contract converges violently to physical reality. If you are not prepared to take or make delivery, roll out of it in time — and remember that everyone else rolling at once is *why* contango costs long-only holders money.

8. **Cross-check the curve against inventories.** The curve and the storage data should agree: backwardation should coincide with drawing, low inventories, and a tight physical market; contango with building, high inventories, and a glut. When they *disagree* — when, say, the curve is backwardated but reported stocks look comfortable — that divergence is itself a signal. Usually it means the market sees something the lagging inventory report has not yet captured (the curve, priced by people trading physical barrels, tends to lead the weekly statistics). Trust the curve's read of *right now*; use the inventory data to confirm and to understand *why*.

Bring this back to the spine of the whole series. A commodity price is a physical thing — a barrel, a tonne, a bushel — forced through a financial contract. The forward curve is the picture of *that* forcing: it is where the cost of storage, the convenience of having the stuff on hand, and the market's read of scarcity all resolve into a single line of prices across time. Contango and backwardation are not arcane jargon; they are the market telling you, in the only words it has, whether the tanks are full or the shelves are bare. The trader who can read that line before placing a trade is reading the supply-and-demand balance of the physical world itself — and that, more than any forecast, is the edge. (For the two underlying prices the curve connects — the spot barrel and the deferred contract — and how they converge at expiry, see [Spot vs futures: the two prices of the same barrel](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel).)

## Further reading & cross-links

Within this series:

- [Spot vs futures: the two prices of the same barrel](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel) — the two endpoints the curve connects, and how they converge at delivery.
- [Contango vs backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — the full trading implications, the roll, and roll yield in depth.
- [Convenience yield and the cost of carry: why the curve has a shape](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape) — the exact formula behind the two springs that set the slope.

Beyond this series:

- [The term structure of volatility: contango, backwardation, and the VIX curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve) — the same shape language applied to implied volatility.
- [Gold futures: COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — why a monetary metal's curve behaves so differently from oil's.
