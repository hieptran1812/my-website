---
title: "Contango vs Backwardation: What the Shape of the Curve Means"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The two curve shapes in depth — what each says about supply and demand right now, and the punchline that the shape, not the price level, drives long-only commodity returns through the roll."
tags: ["commodities", "contango", "backwardation", "roll-yield", "forward-curve", "futures", "carry", "crude-oil", "natural-gas", "long-only"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A commodity's forward curve has exactly two shapes, and which one you are looking at matters more for a long-only holder than the price itself. **Contango** (upward: later delivery dearer) says supply is ample and the market must pay someone to store the glut. **Backwardation** (downward: prompt dearest) says the commodity is scarce right now and buyers pay a premium to have it today.
>
> - The shape, not the level, drives the return of anyone who holds commodity futures for the long run, because a holder must keep **rolling** — selling the expiring front contract and buying the next one.
> - In **contango** every roll is "sell low, buy high" — a steady **bleed** (negative roll yield). In **backwardation** every roll is "sell high, buy low" — a steady **gain** (positive roll yield).
> - This is why a long-only commodity fund can lose money for years while the spot price goes nowhere — the bleed compounds underneath a flat headline price.
> - The one number to remember: a curve in steep contango can quietly cost a long-only holder on the order of **10-15% a year** in roll yield, and a steeply backwardated one can hand them a similar amount the other way. Check the shape *before* you go long anything.

In the spring of 2009, an investor who believed oil had crashed too far did the obvious thing: they bought the most popular oil fund on the market, an exchange-traded product that held front-month crude futures. Over the next year, the spot price of crude oil rose by roughly half — a huge, correct call. And the fund? It barely moved. By the time the spot price had climbed back toward \$80, the fund's investors had captured almost none of it. They had been *right about oil* and *wrong about the curve*, and the curve quietly ate their profit one month at a time.

How is that possible? How can you be right about the price of a thing and still not make money owning it? The answer is the single most important and least understood idea in commodity investing, and it is not about the price level at all. It is about the *shape* of the forward curve — whether it slopes up (contango) or down (backwardation) — because that shape decides what happens every time the fund rolls from one expiring contract to the next. A flat spot price hides a curve that is bleeding the holder dry. This post is about that shape, why it matters more than the headline price, and how to read it before you put a dollar to work.

We met the forward curve and its two shapes in [The forward curve](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities). There, the curve was the whole map. Here we zoom in on the two shapes themselves, build the full intuition for what each *says about the world today*, and then deliver the punchline that the shape — not the level — is what a long-only holder actually earns or loses. By the end you will never look at a commodity ETF brochure the same way again.

![Contango versus backwardation, the two curve shapes and what each means for supply demand and the roll](/imgs/blogs/contango-vs-backwardation-what-the-shape-of-the-curve-means-1.png)

## Foundations: the two shapes, defined precisely

Before anything else, let us nail down the two words, because the entire post lives or dies on getting them exactly right. They are not jargon for jargon's sake — each describes a precise geometric fact about a line, and that fact maps onto a precise statement about the physical world.

Recall the setup. A commodity does not have one price; it has a *menu* of prices, one for each future delivery month. The **front month** (also called the *prompt* or *nearby*) is the contract that delivers soonest — its price is the market's best proxy for "spot," the price to get the physical thing right now. The **back** of the curve is the far-deferred contracts: a year out, two years out. Plot delivery date on the horizontal axis and price on the vertical, connect the dots, and you have the forward curve. Now the two shapes.

**Contango** is an *upward*-sloping curve. Each later delivery costs **more** than the nearer one. A barrel for delivery a year from now is dearer than a barrel for delivery this month. The line climbs to the right. (The word is a Victorian-era London Stock Exchange term for a fee paid to *defer* settlement — to push a trade into the future — and that is exactly what the shape encodes: the future is dearer.)

**Backwardation** is a *downward*-sloping curve. The prompt is the **dearest** barrel, and prices fall the further out you look. A barrel today costs more than a barrel a year from now. The line slides down to the right. (The word also comes from old LSE settlement slang — a fee paid to *delay delivery the other way* — but you do not need the etymology; you need the picture: prompt high, deferred low.)

That is the whole vocabulary. Two words, two slopes. The chart below shows both on a single axis, using the illustrative WTI-style shapes from our data set so the numbers stay consistent across the series.

![The two curve shapes side by side, contango sloping up and backwardation sloping down](/imgs/blogs/contango-vs-backwardation-what-the-shape-of-the-curve-means-2.png)

The orange line is contango: it starts at a prompt price of about \$72 and climbs to nearly \$77 a year out. The green line is backwardation: it starts high, at about \$85 prompt, and slides down toward \$79 a year out. Notice the contango curve has a *lower* prompt price but a *higher* level a year out, while the backwardation curve is the reverse. That crossing pattern is not a coincidence — it reflects the two different worlds these shapes describe, which is what we turn to next.

One vital clarification before we go on. The *level* of the curve — whether oil is at \$40 or \$120 — and the *shape* of the curve are two completely separate facts. Oil can be expensive and in contango (a high but upward-sloping curve), or cheap and in backwardation (a low but downward-sloping curve), or any combination. People constantly conflate the two and assume "contango means the price will rise" or "backwardation means the price will fall." That is **wrong**, and untangling it is half the job of this post. The shape is a statement about *supply and demand right now relative to storage and time* — not a forecast of where the spot price is heading. Hold that thought; we will hammer it in the misconceptions section.

#### Worked example: telling the two shapes apart from a price list

Suppose a desk hands you two price lists and asks which is contango and which is backwardation.

```
Commodity A            Commodity B
M0 (prompt)  72.0      M0 (prompt)  85.0
M2           73.4      M2           83.2
M4           74.5      M4           81.8
M6           75.3      M6           80.7
M12          76.6      M12          78.9
```

Read each one left to right and ask only: does the price go up or down as the delivery month gets later? Commodity A climbs from \$72.0 to \$76.6 — each later month is dearer — so A is **contango**. Commodity B falls from \$85.0 to \$78.9 — each later month is cheaper — so B is **backwardation**. To put a number on the steepness, compute the front-to-12-month gap: A is +\$4.60 (a contango of about 4.60 / 72 ≈ 6.4% of the prompt price over a year), B is −\$6.10 (a backwardation of about 6.10 / 85 ≈ 7.2% over a year).

The intuition: contango and backwardation are nothing more than the *sign of the slope* — up means later-is-dearer, down means prompt-is-dearest — and the size of that slope, as a percentage of the prompt, is how steep the shape is.

### Two refinements that matter in practice

Two subtleties will save you from rookie mistakes, and both are worth internalizing now.

**A curve can be both shapes at once.** Real curves are rarely a clean straight line. A market can be *backwardated up front* (the first few months slope down, signalling prompt tightness) and yet drift back into *contango further out* (the deferred months slope up, signalling the market expects the tightness to ease and supply to return). Oil does this constantly: a supply scare makes the prompt months scarce and steeply backwardated, while two years out — where storage and long-run cost dominate — the curve curls back up. So the honest reading is often "backwardated now, easing later" rather than a single global label. When a holder asks "what is my roll yield?", the answer comes from the *front* of the curve — the part you actually roll through — so a curve that is backwardated up front pays you even if the far end is in contango. Always read the prompt structure (the first three or four months) separately from the deferred structure (everything past a year).

**The prompt spread gets noisy near expiry.** The single most useful number on the curve is the first calendar spread — the difference between the front month and the next one — because it is the cleanest local reading of "tight or loose right now?" But in the few days around the front contract's expiry, that prompt spread becomes unreliable: convergence to spot and the flood of roll activity distort it. So practitioners often measure their "prompt spread" off the *second* and *third* months once the front is within about a week of dying, to keep the signal clean. This is a small operational detail, but it is exactly the kind of thing that separates a clean read from a misleading one.

## What contango says about the world right now

A shape is not just geometry; it is a message. Contango — later delivery dearer — is the market's way of saying one thing above all: **there is plenty of this commodity around right now, maybe too much, and someone needs to be paid to hold the surplus until it is wanted.**

Walk through the logic physically. Imagine the oil tanks at Cushing, Oklahoma — the delivery point for WTI crude — are filling up. Production keeps coming, but demand has softened. There is a *glut*. Nobody is desperate for a barrel this month; in fact, barrels this month are a nuisance, because there is nowhere left to put them. But a barrel a *year* from now is a perfectly fine thing to own — by then the glut may have cleared and the barrel will be wanted. So the market is happy to defer: it discounts the prompt barrel (it is cheap because it is a burden today) and it values the deferred barrel more (because it has to compensate whoever stores the surplus oil for a year — the tank rental, the insurance, the financing cost of the cash tied up in it).

That is the deep reason contango exists: it is the **cost of carry** showing through. Storing a physical commodity costs money — a tank, a warehouse, a silo, plus insurance, plus the interest on the capital you have sunk into the barrel. A rational forward price has to cover all of that, so distant barrels "should" cost more than prompt ones whenever the market is well-supplied enough that carry, rather than scarcity, dominates. We are deliberately *not* deriving the full cost-of-carry formula here — that is the subject of [Convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape), and it explains exactly *why* the curve takes the shape it does. Here, the takeaway is the reading: **contango = ample supply, cheap-to-hold conditions, the market paying you to store.**

So when you see a commodity in steep contango, your first thought should be *glut*. Tanks filling, warehouses backing up, the prompt market sloppy. The steeper the contango, the worse the glut — because the only way the curve can pay *more* than the actual cost of storage is if the prompt price has been beaten down by an oversupply that nobody can absorb. The most extreme example in modern history was April 2020, when COVID demand collapse and a Saudi-Russian price war filled every tank on Earth. The prompt WTI contract went *negative* — sellers paid buyers to take the oil off their hands — while a barrel a year out still fetched over \$35. That is "super-contango," and it was the market screaming, in the only language it has, *we have run out of room to put the stuff.*

There is a natural ceiling on how steep contango can get, and it is one of the most elegant ideas in the whole subject: contango cannot exceed the **full cost of carry** for very long, because if it did, a riskless profit would appear and traders would arbitrage it away. Suppose the prompt barrel is \$70 and the one-year contract is \$90 — a \$20 contango. If the all-in cost to store a barrel for a year (tank rent, insurance, financing) is only \$8, then anyone can buy the cheap prompt barrel for \$70, *simultaneously* sell the \$90 one-year contract, pay \$8 to store the oil for the year, deliver it into that contract, and bank \$90 − \$70 − \$8 = \$12 of riskless profit. This is the **cash-and-carry** trade. So many traders pile into it that they bid up the prompt (buying barrels) and push down the deferred (selling contracts) until the contango shrinks back to roughly the cost of carry. The implication: a contango steeper than the cost of storage is the market telling you that *storage itself is running out* — there is no more room to do the trade — which is exactly the April-2020 super-contango. The arbitrage that normally caps the contango had broken because there were no tanks left to carry the oil. The full mechanics of this cap live in the cost-of-carry post; for now, hold the picture that contango is *bounded above by storage economics,* and a contango that blows past that bound is a flashing red light about physical storage capacity.

## What backwardation says about the world right now

Backwardation — prompt dearest, later cheaper — says the opposite, and it is the more interesting of the two shapes because it seems, at first, to violate common sense. If storing a commodity costs money, how can a barrel a year out be *cheaper* than a barrel today? Shouldn't the future always cost more, to cover storage?

The answer is the second great force in commodities, the one that makes them different from stocks and bonds: the **convenience yield.** When a commodity is *scarce* — when inventories are low and demand is hot — physically owning the thing *right now* has a value beyond its price. A refiner with crude in the tank can keep its plant running when competitors are scrambling for barrels. A chocolate maker with cocoa on hand never has to halt a production line. A utility with gas in storage keeps the lights on through a cold snap while rivals pay any price for an emergency cargo. That benefit — the value of *having it on hand* when you might otherwise run out — is the convenience yield, and it accrues only to whoever holds the *physical* commodity, not to someone holding a far-dated paper claim.

When the convenience yield is large enough, it overwhelms the cost of carry and flips the curve. The prompt barrel is bid up — because having it *now* is worth a premium — and the deferred barrel is discounted, because by the time it delivers, the scarcity may have eased. The result is backwardation: prompt high, deferred low. So the reading is: **backwardation = scarcity, a tight prompt market, buyers paying a premium for immediate availability.**

The everyday way to keep the two straight: in **contango** you pay a premium for *patience* — the future is expensive, the present is cheap, the market is rewarding you for waiting. In **backwardation** you pay a premium for *urgency* — the present is expensive, the future is cheap, the market is punishing you for being in a hurry. **Tight markets are impatient markets; they are backwardated. Glutted markets are patient markets; they are in contango.** The figure earlier in the post captures exactly this contrast — left side contango and its meaning, right side backwardation and its meaning.

#### Worked example: reading the supply story off the slope

You are looking at copper. The prompt (cash) price on the London Metal Exchange is \$9,400 a tonne, and the three-month forward is \$9,330 a tonne. Three months later you look again: prompt \$9,200, three-month forward \$9,280. What changed?

In the first snapshot the curve is **backwardated**: prompt (\$9,400) is dearer than the three-month (\$9,330), a backwardation of \$70 a tonne, or about 70 / 9,400 ≈ 0.7% over three months (roughly 3% annualized). That tells you LME warehouse stocks are tight — someone wants copper *now* and is paying up for immediate metal. In the second snapshot the curve has flipped to **contango**: prompt (\$9,200) is now *cheaper* than the three-month (\$9,280), a contango of \$80, about 0.9% over three months. Even though the *price fell* (from \$9,400 to \$9,200), the *shape flipped from tight to loose* — the scarcity eased, metal came back into the warehouses, and the market went from paying a premium for prompt copper to paying a premium for patience.

The intuition: the price level and the curve shape are independent — here the price dropped *and* the shape loosened, two separate signals, and the shape told you the physical tightness had resolved.

Notice the asymmetry between the two shapes, because it is a deep point. Contango is *bounded* — it cannot get steeper than the cost of carry without inviting the cash-and-carry arbitrage that flattens it. But backwardation has **no such ceiling.** There is no arbitrage that caps how dear the prompt can get relative to the deferred, because you cannot "store negative inventory" — you cannot borrow a barrel you do not have from the future and deliver it today. If a refiner *must* have crude this week or shut its plant, it will pay almost anything for prompt barrels, and no trader can manufacture more physical oil to satisfy that demand. So backwardation can spike to violent extremes that contango never reaches. This is why the most dramatic curve moves — the squeezes, the spikes — are almost always *backwardation* events: a sudden scarcity with no arbitrage to relieve it. The 2022 LME nickel episode and the periodic blowouts in prompt natural gas during cold snaps are backwardation gone vertical. The lesson for a curve-reader: a steep contango is a glut you can roughly size against storage costs; a steep backwardation is a scarcity that can go anywhere, because nothing physical relieves it.

## The punchline: why shape, not level, drives long-only returns

Now we get to the heart of the post, the thing that separates people who understand commodities from people who merely follow the price. Here it is in one sentence: **a long-only holder of commodity futures earns the spot move *plus or minus* the roll yield, and in many markets the roll yield dominates.**

Let us build that up carefully, because it is the idea everyone gets wrong.

You cannot "buy and hold" a commodity the way you buy and hold a stock. A share of Apple sits in your account forever. A barrel of oil, if you bought the physical, would need a tank, insurance, security — you would be in the storage business. So nobody who wants *price exposure* (rather than the physical barrel) holds the physical. They hold **futures.** And futures *expire.* A crude contract for June delivery stops trading a few days before June; if you still hold it, you must either take delivery of 1,000 real barrels at a tank farm in Oklahoma, or close the position. A fund, an ETF, a speculator — none of them wants the barrels. So before the front contract dies, they **roll**: they sell the expiring front-month contract and buy the next month's contract, to keep their exposure alive without ever touching the physical.

That roll happens every single month, for as long as you hold the position. And here is the punchline: **the price at which you sell the expiring contract and the price at which you buy the next one are not the same — they differ by the slope of the curve.** That difference, repeated every month, is the **roll yield**, and its sign is set entirely by the shape.

- In **contango** (upward curve), the front contract you are *selling* is the *cheap* end, and the next month you are *buying* is the *dearer* end. You sell low and buy high. Every roll loses you a little. That is **negative roll yield** — the bleed.
- In **backwardation** (downward curve), the front contract you are *selling* is the *dear* end, and the next month you are *buying* is the *cheaper* end. You sell high and buy low. Every roll gains you a little. That is **positive roll yield** — the carry.

The figure below shows the mechanic explicitly. Same action — sell the front, buy the next — but in contango it is "sell low, buy high" (a loss) and in backwardation it is "sell high, buy low" (a gain).

![The roll mechanic showing sell low buy high in contango versus sell high buy low in backwardation](/imgs/blogs/contango-vs-backwardation-what-the-shape-of-the-curve-means-5.png)

This is why our 2009 investor was right about oil and still made nothing. Oil's curve was in steep contango through that period. Every month the fund rolled, it sold the cheap expiring contract and bought a dearer one further out. The spot price *did* rise — the investor's directional call was correct — but the roll bleed clawed back almost all of it. The shape ate the level.

#### Worked example: the monthly roll cost in contango

Let us put real numbers on the bleed. Take the contango curve from our data: the front month trades at \$72.0 and the next contract (call it M2, two months out) at \$73.4. A simplified monthly roll moves from the front toward the next contract along that slope.

First, the per-month slope. The front-to-M2 step is \$73.4 − \$72.0 = \$1.40 over two months, so the curve rises about \$0.70 per month. When you roll, you sell at roughly the prompt price (\$72.0) and buy the next month at roughly \$0.70 dearer (\$72.7). The roll costs you:

```
roll cost per month = buy price - sell price
                    = 72.70 - 72.00
                    = 0.70 (USD/bbl)
```

As a percentage of the \$72 you have invested, that is 0.70 / 72.0 ≈ **0.97% per month.** Annualize it (compounding aside, just multiply by 12 for the intuition):

```
annualized roll drag = 0.97% x 12 = approx 11.7% per year
```

So holding this contango position, *before any move in the spot price*, you are bleeding roughly **12% a year** just from rolling. The spot price would have to *rise* about 12% a year for you simply to break even.

The intuition: in contango the curve is an escalator running the wrong way — you have to climb (a rising spot price) just to stand still, and a flat spot price means you slide backward at the roll-yield rate.

#### Worked example: the monthly roll gain in backwardation

Now run the same arithmetic on the backwardated curve, where everything works in your favor. The front month is \$85.0 and the next contract is \$83.2 — a step *down* of \$1.80 over two months, about \$0.90 per month.

When you roll, you sell the dear front contract (\$85.0) and buy the cheaper next month (about \$85.0 − \$0.90 = \$84.1). The roll *gains* you:

```
roll gain per month = sell price - buy price
                    = 85.00 - 84.10
                    = 0.90 (USD/bbl)
```

As a percentage of the \$85 invested, that is 0.90 / 85.0 ≈ **1.06% per month**, or annualized:

```
annualized roll gain = 1.06% x 12 = approx 12.7% per year
```

So in steep backwardation you earn roughly **13% a year** from the roll *even if the spot price never moves.* The escalator now runs your way — a flat spot price hands you the roll yield as a pure tailwind. This positive carry is so important that it gets its own post: [Backwardation as a structural return source](/blog/trading/commodities/backwardation-as-a-structural-return-source-the-carry-of-commodities) explains why a persistently backwardated commodity has historically been a *paid-to-wait* trade.

#### Worked example: the cumulative drag over a year

The single-month numbers are striking, but the real damage is in the *compounding.* Suppose you hold a long-only position in our contango market for a full year, the spot price is exactly flat at \$72 the whole time, and you roll twelve times, each roll costing the 0.97% we computed.

Starting from an index of 100, each month multiplies your value by (1 − 0.0097):

```
year-end value = 100 x (1 - 0.0097)^12
              = 100 x (0.9903)^12
              = 100 x 0.8896
              = approx 88.96
```

So over one year, with the spot price *unchanged*, the contango bleed alone has dragged your investment from 100 down to roughly **89 — a loss of about 11%.** Run that for several years and the divergence between the spot price and what you actually earn becomes enormous. The chart below shows exactly that gap: a spot index that wanders around 100 and goes essentially nowhere over five years, while the long-only total-return index — what the holder actually keeps — bleeds steadily down toward 70.

![Same spot price but a falling investment, the roll bleed in contango over five years](/imgs/blogs/contango-vs-backwardation-what-the-shape-of-the-curve-means-3.png)

The intuition: the bleed is not a one-off cost; it compounds month after month, so a long-only holder in contango can watch a *flat* spot price translate into a 10-30% loss over a few years — which is precisely the divergence that surprises ETF investors. The mechanism behind ETF bleed specifically is the subject of [Roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed).

## Roll yield by regime: putting magnitudes on the shape

How big is the roll yield in practice? It depends on how steep the curve is, which in turn depends on the commodity and the moment. The chart below lays out illustrative annual roll yields across the regimes — from steep contango (the worst bleed, characteristic of chronically over-stored markets like natural gas) through flat to steep backwardation (the best carry, characteristic of tight oil markets).

![Roll yield by curve regime, negative for contango and positive for backwardation](/imgs/blogs/contango-vs-backwardation-what-the-shape-of-the-curve-means-4.png)

Read it from the bottom up. **Steep contango** — say a badly over-supplied natural gas market — can bleed a long-only holder on the order of **−12% a year.** **Mild contango** runs maybe **−4%.** A **flat** curve costs nothing in roll. **Mild backwardation** pays around **+5%,** and **steep backwardation** — a tight oil market in a supply scare — can hand the holder roughly **+12% a year** before the spot price even moves.

These are illustrative magnitudes, but the orders of magnitude are real and they are *large.* A 12% headwind or tailwind, every year, before any directional bet pays off, swamps the kind of edges most investors hope to find elsewhere. This is the empirical heart of the "shape, not level" thesis: across the long history of commodity index investing, the *roll* (the carry from the curve shape) has explained more of the total return than the *spot* (the change in the underlying price) over many multi-year stretches. You are, in large part, being paid or charged for the shape.

It helps to decompose a commodity holder's return into its three honest pieces, because once you see the breakdown you can never un-see how central the roll is:

```
total return  =  spot return  +  roll yield  +  collateral yield
                 (price move)    (the shape)    (interest on cash)
```

The **spot return** is the change in the underlying price — the part everyone focuses on. The **collateral yield** is the interest earned on the cash backing the futures position (futures need only a margin deposit, so the rest sits in T-bills earning the short rate) — a quiet positive in a high-rate world, near zero in a zero-rate one. And the **roll yield** is the carry from the shape, the term this whole post is about. The trap is that the public talks only about the first term, the brochures bury the third, and almost nobody mentions the second — yet over multi-year holds the second term frequently dwarfs the first. When a commodity index "underperforms the spot price," this decomposition is the whole explanation: the roll yield was negative (contango) and quietly subtracted several percent a year. The discipline is to estimate all three before you hold, not to discover the roll term after it has eaten your return.

## Where contango bit hardest: natural gas

The textbook case of a market that punishes long-only holders is U.S. natural gas. Gas has a deep structural reason to live in contango: it is *seasonal and storage-driven.* Demand peaks in winter (heating) and the curve almost always slopes up toward the winter months, so a fund rolling through the calendar is forever selling cheaper near-dated gas and buying dearer deferred gas. Worse, the cost of storing gas (in salt caverns and depleted reservoirs) is real and the convenience yield is usually low outside of cold snaps, so the cost of carry dominates and the curve stays in chronic contango.

The chart below shows the Henry Hub spot price over two decades. Notice how *low* it has often sat — \$2-3 per MMBtu for years at a stretch — punctuated by the 2022 supply spike. But the spot price chart understates the damage to a long-only holder, because beneath that flat-to-falling spot sat a curve that was perpetually sloping up, so the roll bled the holder year after year. A famous natural gas ETF lost the overwhelming majority of its value over its lifetime *not* because gas crashed — gas mostly chopped sideways — but because the contango roll ground it down month after month after month.

![Henry Hub natural gas annual average price, the market where contango bites hardest](/imgs/blogs/contango-vs-backwardation-what-the-shape-of-the-curve-means-6.png)

The lesson generalizes: any commodity that is cheap to store, easy to oversupply, and rarely scarce in the prompt tends to live in contango — and is therefore a poor candidate for a passive long-only position. Gas is the extreme, but heavily-stored industrial commodities share the trait. By contrast, commodities that are expensive or impossible to store, or that swing into prompt scarcity (crude oil in a supply scare, some softs at harvest gaps), spend more time backwardated and have rewarded long holders far better.

## What decides which shape a commodity lives in

If shape drives returns, the natural next question is: what makes one commodity sit in chronic contango while another spends years backwardated? It comes down to a short list of physical properties, and learning to predict the *tendency* from those properties is a genuine edge.

**How cheap is it to store?** This is the single biggest factor. The cheaper and easier the storage, the more the cost of carry shrinks toward just financing, and the easier it is for the market to defer surpluses — which biases the curve toward contango but a *shallow* one. The harder and dearer the storage, the bigger the carry cost and the steeper any contango must be to compensate; and crucially, the easier it is for storage to fill up and tip the market into super-contango or, when it empties, into backwardation. Crude oil and gas are bulky and awkward to store, so their curves swing hard. Gold is trivially cheap to store (a vault holds a fortune in a small space), which is one reason its curve behaves so differently — a contrast we draw out below.

**How perishable or seasonal is it?** A commodity that cannot be stored at all, or only briefly, has a curve dominated by the *production calendar* rather than carry. Livestock cannot be warehoused — a steer eats and ages — so its curve reflects the herd cycle, not storage. Grains are harvested once or twice a year and stored in between, so their curves carry a seasonal signature: cheap and contango-ish right after harvest (silos full), tightening into the next planting gap. Natural gas, as we saw, is seasonal in *demand* (winter heating) and that seasonality is baked into a chronic upward slope toward the cold months.

**How prone is it to prompt scarcity?** A commodity that the world genuinely *needs continuously* and that can suddenly run short in the prompt — crude during an embargo, a battery metal during a supply chokepoint, cocoa after a West-African crop failure — will swing into backwardation whenever that scarcity bites, because the convenience yield of having it on hand spikes. A commodity that is rarely truly scarce stays in contango.

**How easy is it to oversupply?** Commodities with short-cycle, flexible production (U.S. shale oil can ramp in months; a contango that pays for storage encourages it) tend to flood into surplus quickly, reinforcing contango. Commodities with long, lumpy supply (a new copper mine takes a decade) cannot respond to a shortage fast, so scarcity — and backwardation — can persist for years.

#### Worked example: predicting the shape from the physics

You are handed two commodities you have never traded and asked which is *more likely* to be in contango. Commodity P is cheap to store, has flexible short-cycle production, and is rarely scarce in the prompt. Commodity Q is expensive to store, has long-lead supply that cannot respond quickly, and the world consumes it continuously with periodic shortages.

Score them on the four factors. Commodity P: cheap storage (→ contango-friendly), easy to oversupply (→ contango), rarely scarce (→ contango). Three of four point to **contango.** Commodity Q: expensive storage (→ curve swings, prone to backwardation when storage empties), inflexible supply (→ scarcity persists, backwardation), continuous demand with shortages (→ high convenience yield, backwardation). Three of four point to **backwardation.** Without knowing a single price, you can predict that a passive long position in P will probably bleed on the roll, while a long position in Q will more often be paid to wait.

The intuition: you can read a commodity's *likely* curve shape — and therefore whether a long-only holder gets paid or charged — straight off its physical properties, before you ever look at a chart.

## A note on Keynes, "normal backwardation," and the modern view

You will eventually run into a phrase that confuses everyone: **"normal backwardation."** It comes from John Maynard Keynes, and it is worth a paragraph so the terminology does not trip you up.

Keynes's idea (in the 1930s) was about *why* a curve might tend to slope down even absent a glut or scarcity story. His theory: producers — a farmer with a crop coming, a miner with output to sell — are naturally *long* the physical and want to *hedge* by selling futures to lock in a price. To get speculators to take the other side of all that hedging (to buy those futures), the futures price has to be set a little *below* the expected future spot price, so the speculator earns a risk premium for providing the insurance. That downward bias of the futures price relative to expected spot is what Keynes called "normal backwardation." Crucially, it is a statement about the futures price versus *expected future spot* — not the simple "prompt-dearer-than-deferred" slope we have been discussing.

In modern practice, two things are true. First, the *observed* curve shape (contango vs backwardation) is driven far more by the concrete tug-of-war between storage cost and convenience yield than by Keynes's hedging-pressure story — which is why we built the intuition that way. Second, the *spirit* of Keynes's insight survives: in backwardated markets, long-only speculators do tend to earn a structural risk premium for bearing price risk that producers want to shed, and that is precisely the positive roll yield we computed. So do not worry about the historical jargon. Just hold the practical mapping: **down-sloping curve (backwardation) tends to pay the long holder; up-sloping curve (contango) tends to charge them.** The full hedging-pressure story belongs to [The four players](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators), which is the right post for who is on the other side of all this.

## Common misconceptions

**"Contango means the price will go up; backwardation means it will go down."** This is the most common and most expensive error. The slope of the curve is a statement about *storage and scarcity today,* not a forecast. A contango curve does *not* predict a rising spot price — if anything, the steepest contangos coincide with gluts that have just *crushed* the prompt price. And backwardation does not predict a falling price; it just says the prompt is tight now. Plenty of commodities have risen in price while in backwardation and fallen while in contango. Burn this in: **shape ≠ forecast.** The number that proves it: in April 2020 the curve was in record *super*-contango (deeply upward), yet the spot price had just *collapsed* — the opposite of "contango means prices rise."

**"If the spot price is flat, my commodity ETF will be flat too."** No — and this is the trap. In contango, a flat spot price means your ETF *falls* at the roll-yield rate. Our worked example showed a flat \$72 spot translating into an ~11% annual loss for the holder. The spot price and the long-only return diverge by exactly the cumulative roll. The proof is the spot-vs-total-return chart above: spot ends near where it started, the investment ends ~30% lower over five years.

**"Backwardation is rare and weird; contango is the normal state."** It varies wildly by commodity. Heavily-stored, easy-to-oversupply commodities (natural gas, some industrial metals in surplus) live in chronic contango. But crude oil has spent *long* stretches in backwardation — through most tight-supply periods — and some commodities are backwardated more often than not. There is no universal "normal." You have to check each market, each time. The number to remember: a steeply backwardated oil market can pay a long holder on the order of +12% a year — hardly a rare curiosity.

**"Roll yield is a small technical detail."** It is often the *dominant* term in long-only commodity returns over multi-year horizons — frequently larger than the spot move itself. A 12% annual headwind or tailwind is not a rounding error; it is the whole game. Treating it as a footnote is how investors lose money being right about the price.

**"I can avoid the bleed by holding the physical instead of futures."** Only if you are prepared to actually store the commodity — pay for the tank, the insurance, the financing — in which case you are *paying the cost of carry directly* rather than through the roll. There is no free lunch: the contango bleed is the market charging you for storage you would otherwise have to provide yourself. The point of recognizing it is to size the position, choose the commodity, or trade the spread accordingly — not to pretend it can be dodged.

## How it shows up in real markets

**Crude oil, 2009 — the famous ETF gap.** As the post opened, an investor who bought a front-month crude ETF in early 2009 caught a roughly 50% rise in the spot price over the next year and yet barely profited, because the oil curve sat in steep contango (the post-crash glut had tanks brimming). The fund rolled monthly into dearer contracts, and the bleed devoured the directional gain. The episode pushed many such products to switch to "optimized" roll strategies — buying further out the curve where the slope is flatter, or using rules to minimize roll cost — an explicit admission that the *shape* was the problem, not the price.

**Natural gas, the long grind.** A well-known leveraged natural gas product lost the vast majority of its value across its life. Gas spot prices chopped sideways for years; the destruction came almost entirely from the chronic contango roll, compounding month after month. It is the cleanest real-world demonstration that you can be flat on the commodity and still lose nearly everything to the shape.

**April 2020 — super-contango pays the patient.** When the prompt WTI contract went negative and the curve exploded into record contango, the bleed for a *long-only* holder was catastrophic. But the same shape was a *gift* to anyone running a **cash-and-carry** trade: buy the cheap prompt barrel, immediately sell a dear deferred contract, store the oil (in tankers, since onshore tanks were full), and pocket the gap — a near-arbitrage that the steep contango made hugely profitable. The shape that punished the passive long rewarded the storage trader. (That trade, and its limits when storage runs out, is its own subject.)

**Oil in a supply scare — backwardation as a tailwind.** During tight-supply episodes — geopolitical shocks, OPEC+ cuts biting, inventories drawing down — crude has swung into steep backwardation, and long-only holders earned a positive roll *on top of* any price gain. Those are the stretches when commodity indices quietly outperform, because the carry is working for the holder rather than against them. The 2021-2022 energy episode is the textbook case: as the world reopened and supply lagged, the oil curve went deeply backwardated, and a long position collected a fat positive roll month after month while the spot price also climbed — the two effects compounding in the holder's favor. An investor who bought energy in that window did not just catch the price rise; they were *paid to wait* the whole time they held it.

**The same word, two opposite outcomes, in the same year.** It is worth pausing on how completely the shape can dominate. There are years where a commodity's spot price ends roughly where it began, yet a long-only holder either made a double-digit return or lost a double-digit amount — depending entirely on whether the curve was backwardated or in contango through the year. The headline "oil was flat in 20XX" tells you almost nothing about what a futures holder earned; the curve shape tells you almost everything. This is the single most counterintuitive fact in the asset class, and it is why the experienced commodity investor's eye goes to the slope of the strip before it goes to the big number at the top of the screen. The level is the story the news tells; the shape is the story your account statement tells.

#### Worked example: comparing two "correct" bullish bets

You are bullish on two commodities for the next year and expect each spot price to rise 8%. Commodity X is in steep contango (roll yield about −12% a year); commodity Y is in steep backwardation (roll yield about +12% a year). You go long futures on both. What do you actually earn?

```
Commodity X (contango):
  spot move        +8%
  roll yield      -12%
  total return     -4%   <- you were RIGHT and still lost money

Commodity Y (backwardation):
  spot move        +8%
  roll yield      +12%
  total return    +20%   <- the shape doubled your gain (and more)
```

Two identical, correct directional calls; wildly different outcomes — entirely because of the curve shape. On X the roll turned a winning view into a 4% loss; on Y the roll turned the same 8% view into a 20% gain.

The intuition: your directional view is only *half* the trade — the curve shape is the other half, and it can flip a correct call into a loss or amplify it into a windfall, which is why a practitioner checks the shape before, not after, going long.

## The takeaway: check the shape before you go long

Here is the practical rule that falls out of everything above, and it is the one habit that separates people who profit from commodities from people who get quietly ground down: **before you take a long-only position in any commodity, read the shape of the forward curve — not just the price.**

The decision tree below is the whole workflow on one page.

![Decision rule, the curve shape and what it implies for a long-only position](/imgs/blogs/contango-vs-backwardation-what-the-shape-of-the-curve-means-7.png)

Walk it through:

1. **Look at the slope first, the level second.** Pull up the strip of futures prices and ask the one question that matters for a holder: does it slope up (contango) or down (backwardation)? Quantify it as an annualized roll yield using the arithmetic from the worked examples — the front-to-next-month step, scaled to a year, as a percentage of the prompt price.

2. **If it is in contango, respect the bleed.** A long-only position is fighting a headwind. The spot price has to *rise by at least the roll-yield rate* just for you to break even. That does not mean never go long — it means *size down,* hold for a shorter horizon, prefer commodities and points on the curve where the contango is mild, or express the view a different way (a calendar spread, an equity proxy, an optimized-roll product). The one thing you must not do is buy a steep-contango commodity, sit on it for years, and expect to track the spot price. You will not.

3. **If it is in backwardation, the carry is a tailwind.** The roll is *paying* you, and you profit even if the spot price stays flat. This is a *structural* reason to be long — the closest thing to a free lunch in the asset class — and it is why persistently backwardated commodities have rewarded long holders historically. Lean into it, but stay aware that the shape can flip when the prompt scarcity resolves.

There is also a way to *separate* your directional view from the roll, and it is what professionals reach for when they like the price but hate the shape. Instead of a naked long position that pays the full contango bleed, you can hold the exposure at a point on the curve where the slope is *flatter* — buying twelve months out rather than the front, where each roll moves you only a small distance along a gentler part of the curve. Or you can trade the shape *itself*, going long one contract month and short another, so that the flat-price move cancels and you are left exposed only to whether the curve steepens or flattens. That is a **calendar spread**, and it is how a desk expresses a pure "the curve will tighten" view without betting on the level at all — a much smaller, more targeted risk than a flat-price long. The full toolkit for trading the shape rather than the level is its own subject; the point here is simply that recognizing the curve shape opens up *choices* about how to express a view, and the naive front-month long is usually the worst of them in contango.

4. **Match the holding period to the shape.** A short tactical trade can tolerate a contango it would never survive over years, because the bleed is roughly proportional to time held. If your edge is a two-week catalyst, a mild contango barely dents it. If you intend to hold a commodity as a multi-year portfolio sleeve, the shape is everything, because the roll compounds the whole time. The longer the horizon, the more the shape — and the less the level — determines what you keep.

5. **Re-check the shape over time, because it changes.** A backwardated curve can flip to contango when supply returns; a contango can flip to backwardation when a shock tightens the prompt. The roll yield you signed up for is not permanent. The curve is a living thing, and the shape is the part of it you most need to monitor as a holder.

### The contrast that proves the point: gold

Nothing sharpens the "shape, not level" idea like the one commodity that breaks the usual rules — gold. Gold is a *monetary* metal, not a consumption one. Almost nobody burns it, eats it, or feeds it into a factory and uses it up; the gold mined over thousands of years still essentially exists, sitting in vaults and jewelry. That single fact reshapes its curve. Because gold is trivially cheap to store and there is never a *prompt consumption scarcity* the way there is for oil or cocoa, gold's convenience yield is tiny and its curve is almost always in gentle **contango** — the deferred price sits above spot by roughly the financing cost of holding it. There is no glut-versus-scarcity drama in the slope; the slope is essentially just an interest-rate calculation.

So gold's curve is *all deferred structure and no prompt structure.* It does not swing into the violent backwardation of a tight oil market, because there is no physical shortage to create one. The roll yield on a long gold futures position is therefore a small, steady, usually-negative number tied to rates — not the 12% swings of the consumption commodities. This is precisely why gold gets its own treatment, and why we keep insisting that **commodities are consumption and industrial assets, not monetary ones.** When you read a gold curve, you are reading a cost-of-carry-minus-nothing line; when you read an oil curve, you are reading the live physical balance of the world's most important consumption commodity. The exact same words — contango, backwardation — describe both, but they *mean* something completely different in each, and [Gold futures: COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) is where that distinction is drawn out in full.

Step back to the series spine. A commodity price is a physical thing forced through a financial contract, and the forward curve is where that collision is recorded. The *shape* of the curve — contango or backwardation — is the physical balance of the market (glut or scarcity, cheap storage or hot demand) translated into the one thing a paper holder actually experiences: the roll. That is why commodities behave so differently from stocks. A stock has no curve, no roll, no carry tied to storing a physical thing; you buy it and hold it and you get the price. A commodity makes you keep rolling a paper claim on a physical good, and the slope of the curve decides whether that rolling quietly pays you or quietly bleeds you. Get the direction right and the shape wrong, and the shape wins. So read the shape first. It is, more than the price, what you are actually going to earn.

The *why* behind the shape — the exact cost-of-carry equation, storage plus financing minus convenience yield — is the next stop: [Convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape). And the curve shape's most famous victim, the long-only ETF, gets dissected in [Roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed).

## Further reading & cross-links

**Within this series:**
- [The forward curve: the most important chart in commodities](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — the full map of the futures strip, of which contango and backwardation are the two headline shapes.
- [Convenience yield and the cost of carry: why the curve has a shape](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape) — the *why* behind the slope: storage plus financing minus convenience yield, the equation that sets contango vs backwardation.
- [Roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed) — the contango bleed dissected at the product level, with the brochures' fine print.
- [Backwardation as a structural return source: the carry of commodities](/blog/trading/commodities/backwardation-as-a-structural-return-source-the-carry-of-commodities) — why a backwardated curve is a paid-to-wait trade and the historical commodity risk premium.
- [The four players: producers, consumers, hedgers, and speculators](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators) — who is on the other side of the roll, and where Keynes's hedging-pressure story actually lives.

**Out to sibling series:**
- [Gold futures: COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — the same two shapes on the *monetary* metal, where storage and lease rates, not consumption scarcity, set the curve — a sharp contrast to the industrial commodities here.
- [The term structure of volatility: contango, backwardation, and the VIX curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve) — the identical shape language applied to volatility futures, where the contango bleed is the reason short-vol products decay.
- [Energy: oil and gas, the inflation engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) — how the energy complex fits a multi-asset portfolio, where the roll yield becomes a sleeve-level return driver.
- [Commodities as macro signals: oil, copper, gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) — reading the curve shape as a real-time signal of physical tightness across the cycle.
