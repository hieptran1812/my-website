---
title: "Spot vs Futures: The Two Prices of the Same Barrel"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "What a futures contract actually is, how margin and daily mark-to-market work, why 98% of contracts never deliver, and how to read the gap between the spot price and the futures price."
tags: ["commodities", "futures", "spot-price", "margin", "mark-to-market", "clearinghouse", "wti-crude", "basis", "physical-delivery"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A commodity has two prices at once: the *spot* price (cash for a barrel today) and the *futures* price (a standardized, exchange-traded promise to deliver that barrel on a set date), and almost everything in commodities lives in the gap between them.
>
> - A futures contract is not an IOU between two strangers. The exchange standardizes the terms, and a **clearinghouse** steps into the middle so each side faces *it*, not each other.
> - **Margin is a performance bond, not a down-payment.** You post a few thousand dollars to control tens of thousands of dollars of oil, and the clearinghouse re-prices your position and moves cash every single day (mark-to-market).
> - **About 98% of contracts never deliver a physical barrel** — they get offset or rolled before expiry. The 2% that can deliver is what keeps the paper price honest.
> - The one number to remember: one WTI contract = **1,000 barrels**. At \$72 a barrel that is \$72,000 of oil controlled by roughly **\$6,000** of margin.

On 20 April 2020, the price of a barrel of American crude oil fell below zero. Not to a dollar. Not to a dime. The May contract for West Texas Intermediate (WTI) settled at **negative \$37.63** — meaning a seller would pay you \$37,630 to take 1,000 barrels off their hands. For a few surreal hours the most important commodity in the global economy had a price that looked like a typo.

It was not a typo, and understanding *exactly* what broke that day is the fastest way to understand what a futures price even is. Because here is the thing most people miss: the barrel of oil sitting in a tank in Cushing, Oklahoma did *not* have a negative value that afternoon. Crude in the cash market was cheap, maybe \$10–20, but it was positive. The number that went to −\$37.63 was a *futures price* — a price attached to a specific contract with a specific delivery obligation and a specific clock running out. The holders of that contract were trapped between a promise to take delivery of physical oil and a world that had nowhere left to put it.

Two prices, same barrel. That gap — between the physical thing and the paper claim on it — is the engine room of the entire commodity market, and this post takes you inside it from zero.

![Anatomy of one WTI futures contract showing quantity grade delivery point expiry and the clearinghouse between buyer and seller](/imgs/blogs/spot-vs-futures-the-two-prices-of-the-same-barrel-1.png)

## Foundations: what "spot" and "futures" actually mean

Start with the simplest possible transaction. You walk up to an oil producer, hand over cash, and they hand you a barrel of crude right now. The price you pay is the **spot price** (also called the *cash price* or *physical price*). "Spot" means *on the spot* — immediate delivery, immediate payment, the price for a barrel that exists today and changes hands today. When a news anchor says "oil is at \$72," they almost always mean the front-month futures price, but the spot price is the purest thing of all: what a real barrel costs to buy and carry away right now.

Now change one detail. Instead of buying a barrel today, you and the producer agree *today* on a price for a barrel to be delivered *three months from now*. Nothing physical moves yet. You have not paid for oil; you have agreed on the terms of a future exchange. That agreement is a **forward contract**, and the price you locked in is a **forward price**.

A **futures contract** is a forward contract that has been put through an industrial standardizer. The two ideas are cousins:

- A **forward** is a private, custom agreement between two specific parties — you and that one producer — for whatever quantity, quality, place, and date you negotiate. It is bespoke, like a tailored suit.
- A **future** is the same kind of promise, but with every term *standardized by an exchange* and made identical to every other contract of its type, so the contract itself can be bought and sold by anyone, anytime, without the original counterparties ever meeting. It is off-the-rack, in one size, so a million people can trade it.

That standardization is the whole trick, and it is worth being precise about exactly what gets fixed. A single NYMEX WTI crude futures contract specifies:

- **Quantity** — exactly 1,000 U.S. barrels (42,000 gallons). Not 950, not "about a tankerful." Exactly 1,000.
- **Quality / grade** — light, sweet crude meeting a published specification (sulfur content, API gravity, and a list of acceptable domestic streams). Every contract is the *same* oil so nobody argues about what they are getting.
- **Delivery point** — a specific physical location: Cushing, Oklahoma, a pipeline hub where tanks and pipelines meet. The buyer takes title to oil *there*.
- **Delivery period / expiry** — a named calendar month, with the contract ceasing to trade a few business days before the month begins. Time is fixed.

Because all four of those are identical across every contract, the *only* thing left to negotiate is the **price**. And a market where the single open variable is price is a market that can have a screen, a tick, a bid and an offer, and millions of participants who never speak. The exchange manufactured a standardized promise so that the promise itself could be priced and traded like a stock.

So the two prices of the same barrel are:

1. **Spot:** what a real, immediately deliverable barrel costs today.
2. **Futures:** what the standardized, exchange-traded promise to deliver a barrel on a set date costs today.

They are related — tightly, by arbitrage, as we will see — but they are not the same number, and the difference between them carries enormous information.

#### Worked example: spot vs futures on the same day

Suppose on a given morning the cash market for WTI at Cushing is **\$72.00** a barrel — that is the spot price. On the same screen, the futures contract for delivery in six months trades at **\$75.30**. The barrel and the promise are priced \$3.30 apart. That \$3.30 is not a mistake or an inefficiency; it is the market quoting the *cost of waiting* — roughly the cost to store and finance a barrel for six months, net of any benefit of holding the physical now. If you bought spot oil at \$72.00, paid to store it, and sold it forward at \$75.30, the \$3.30 would have to cover your storage and interest or the trade would lose money. **The two prices of the same barrel differ by exactly the economics of holding it through time — that is the seed of the entire forward curve.**

We will not chase the full curve here — the strip of prices across every maturity gets its own treatment in [The Forward Curve: The Most Important Chart in Commodities](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities). For now, hold the simple fact: *spot* and *futures* are two distinct, simultaneously quoted prices, and the gap between them is meaningful.

### Where the futures price comes from: the cost of carry

It is worth slowing down on *why* a futures price exists as a separate number at all, because the answer is the most important idea in the whole series and most explanations skip it. Imagine you could choose between two ways of having a barrel of oil in six months. **Route A:** buy a barrel today at the spot price, pay to store it for six months, pay the interest on the money you tied up, and you have your barrel in six months. **Route B:** do nothing today, and instead agree now (via a futures contract) on a price to buy that barrel in six months. Both routes deliver the same thing — a barrel in six months — so, by a no-free-lunch argument, they should cost about the same. That equivalence pins the futures price to the spot price plus the cost of getting from now to then:

$$F \approx S + \text{storage} + \text{financing} - \text{convenience yield}$$

Read it as plain English. The futures price `F` is roughly the spot price `S`, *plus* what it costs to store the barrel (tank rent, insurance, evaporation), *plus* the interest you forgo by paying cash now, *minus* a fudge term called the **convenience yield** — the hidden benefit of actually holding the physical barrel (a refiner that owns oil can keep running if a pipeline goes down; a paper-contract holder cannot). When storage and financing dominate, futures sit *above* spot (you pay to defer); when convenience yield dominates — because the physical commodity is scarce and valuable to hold *right now* — futures can sit *below* spot. Those two states have names, contango and backwardation, and they are the subject of A3 and A4. For this post, the takeaway is narrower and crucial: **the futures price is not a forecast of where the spot price will be; it is today's spot price adjusted for the arithmetic of carrying a barrel through time.** A six-month future at \$75.30 against a \$72.00 spot is not the market "predicting \$75.30 in six months." It is the market saying carrying a barrel for six months costs about \$3.30, net.

This single equation explains why the two prices must converge at expiry (the carry shrinks to zero as time runs out), why the basis is informative (it *is* the cost of carry), and why a long-only investor who keeps rolling can lose money even when the spot price is flat (they keep paying the carry). Keep it in your pocket; we will cash it out repeatedly.

### Reading a futures quote: ticks, points, and contract size

Before going further, decode what a futures screen is actually telling you, because the numbers are denser than a stock quote. A WTI futures quote like "CLM5 72.45" packs in four things: the product (CL = crude light, the WTI ticker), the delivery month and year (M5 = June 2025, using the exchange's month codes), and the price per barrel (\$72.45). The price is quoted *per unit* — per barrel for oil, per bushel for grain, per pound for copper — but the contract controls a fixed multiple of that unit, the **contract size**. For WTI that multiple is 1,000 barrels, so the quote \$72.45 corresponds to a contract worth \$72,450.

The **minimum price increment**, or **tick**, is the smallest move the price can make: for WTI it is \$0.01 per barrel. Multiply the tick by the contract size and you get the **tick value** — the dollars that change hands per tick: \$0.01 × 1,000 = \$10. This is the unit of P&L. Every penny the price moves is \$10 in or out of your account per contract. A "dollar move" is 100 ticks, \$1,000. Internalize the tick value of any contract you trade and your P&L becomes instantly legible: you do not have to multiply, you just count ticks.

Contract sizes are chosen so the notional is large — that is deliberate, because futures were built for commercial-scale hedging, not retail nibbling. To open the door to smaller players, exchanges list **mini** and **micro** contracts (Micro WTI controls 100 barrels, a tenth of the standard), with proportionally smaller tick values and margins. The mechanics are identical; only the scale changes.

#### Worked example: translating a quote into dollars

You see Micro WTI quoted at **72.45**, up **0.30** on the day. Micro WTI controls **100 barrels**, so its tick value is \$0.01 × 100 = **\$1.00** per penny. The 0.30 (30-cent) rise is 30 ticks, worth 30 × \$1.00 = **\$30** per contract today. The full notional you control is 72.45 × 100 = **\$7,245**, against a micro initial margin of a few hundred dollars. If instead you held the *standard* WTI contract (1,000 barrels), every number is ten times larger: \$10 tick value, \$72,450 notional, \$300 on the day. **A futures quote is per-unit, but your P&L is per-contract — multiply the price move by the contract size, and a wall of numbers turns into a clear dollar figure.**

## The clearinghouse: why you never actually trust the other side

The biggest conceptual leap from "I made a deal with that producer" to "I bought a futures contract" is that in the futures market **you do not know, and do not care, who is on the other side.**

Think about the problem a private forward creates. You agree with a producer in March to buy 1,000 barrels at \$75 for September delivery. By September, oil has crashed to \$50. The producer is thrilled to deliver to you at \$75 — they are getting \$25 over the market. But what if oil had *risen* to \$100? Now *you* want delivery at \$75 (a \$25 gift), and the producer would love to walk away and sell their oil to someone else at \$100. The deal is only as good as the counterparty's willingness and ability to honor it when honoring it hurts. This is **counterparty risk** — the risk that the person who owes you doesn't pay.

The exchange solves this with an institution sitting in the middle: the **clearinghouse**. (For WTI on NYMEX it is CME Clearing.) When your buy order matches someone's sell order, the clearinghouse performs a legal maneuver called **novation**: it tears the single trade in two and inserts itself as the counterparty to *both* sides. After novation, the clearinghouse is the seller to every buyer and the buyer to every seller. You owe the clearinghouse; the clearinghouse owes you. The stranger who took the other side of your trade is now *their* problem, not yours.

That is figure 1's punchline: the buyer and the seller never face each other. They both face the lavender box in the middle. Counterparty risk has been mutualized and transferred to a single, heavily capitalized, tightly regulated institution whose entire job is to guarantee that trades settle.

But a guarantee is only worth something if it is funded. How does the clearinghouse make sure it can always pay the winner even when the loser defaults? Two mechanisms, working together every day: **margin** and **mark-to-market**.

![Margin and daily mark to market flow from posting initial margin through daily settlement to a variation margin call](/imgs/blogs/spot-vs-futures-the-two-prices-of-the-same-barrel-2.png)

## Margin: a performance bond, not a payment

Here is the single most misunderstood word in derivatives, and getting it right will fix half your intuition. In the stock market, "margin" means *borrowing*: you put up \$5,000, the broker lends you another \$5,000, and you buy \$10,000 of shares. That is a loan with interest, and the margin is your down-payment on an asset you now partly own.

**In futures, margin is something completely different.** When you buy a futures contract, you are not buying an asset and you are not borrowing money. You are entering a *promise* — and the margin is the **good-faith deposit that backs your promise**. The technical term is a **performance bond**. It is not a down-payment on the oil (you do not own any oil), and the clearinghouse does not lend you anything (there is nothing to lend — no asset changed hands yet). The margin simply sits there as collateral, proving you can cover the losses your position might generate before the next time accounts are settled.

There are two margin numbers you must know:

- **Initial margin** — the deposit required to *open* a position. For one WTI contract this is on the order of **\$6,000** (the exact figure floats with volatility; the exchange raises it when markets get wild and lowers it when they calm).
- **Maintenance margin** — a slightly lower floor, on the order of **\$5,500**, that your account must stay above while the position is open. Drop below it and you get a **margin call**: a demand to wire fresh cash back up to the initial level.

Notice the size of these numbers next to the thing they control. One contract is 1,000 barrels. At \$72 a barrel, that is **\$72,000** of crude oil. You control all of it with about \$6,000 of bond — roughly **12 to 1 leverage**. This is the defining feature of futures: enormous economic exposure backed by a small fraction of cash.

![Horizontal bar chart comparing a 72000 dollar contract notional against 6000 initial margin and 5500 maintenance margin](/imgs/blogs/spot-vs-futures-the-two-prices-of-the-same-barrel-6.png)

#### Worked example: the leverage in one barrel of paper

You buy one WTI contract at **\$72.00**. The notional value of the oil you now have exposure to is \$72.00 × 1,000 = **\$72,000**. To open the trade you post initial margin of about **\$6,000** — not as a payment, as a bond. Now the price moves. A WTI contract has a **tick value** of \$10 per one-cent move (because 1,000 barrels × \$0.01 = \$10), so a **\$2.00** rise in the price is worth \$2.00 × 1,000 = **\$2,000** to you. Your bond was \$6,000; you just made \$2,000. That is a **33% return on the cash you posted** from a price move of under 3%. The same arithmetic runs in reverse: a \$2.00 *drop* vaporizes \$2,000, a third of your bond, and the clearinghouse will want it back before tomorrow. **Leverage is symmetric — the \$6,000 bond magnifies a 2.8% price move into a 33% swing in your cash either way.**

That magnification is exactly why the clearinghouse cannot just collect a bond once and forget about it. A position that can lose a third of its collateral on a routine daily move would blow through the bond entirely in a bad week. So the clearinghouse settles up not weekly, not monthly — but **every single day.**

### Who sets the margin, and why it changes

The margin number is not arbitrary and it is not set by your broker on a whim. The exchange's clearinghouse calculates it to cover, with high confidence, the *largest one-day loss* a position is likely to suffer — because one day is exactly how long the clearinghouse is on the hook before the next mark-to-market collects the loss in cash. CME uses a risk model called **SPAN** (Standard Portfolio Analysis of Risk) that scans a range of plausible price and volatility moves and sets the margin to the worst plausible daily outcome. The practical consequences are two:

First, **margin tracks volatility, not price.** When markets are calm, a one-day move is small, so the bond can be small. When markets go haywire — a war, a hurricane in the Gulf, a demand collapse — the plausible one-day move balloons, and the clearinghouse *raises* initial margin, sometimes sharply and overnight. This is the cruel feedback loop of the futures market: the moment you most need to hold your position through chaos is the moment the bond required to hold it jumps. A trader who was comfortably margined on Monday can be forced to post much more on Friday for the *same* position, purely because volatility rose. Traders blow up not because they were wrong but because a margin hike forced them out of a position they could no longer fund.

Second, **margin is netted across a portfolio, not charged per contract in isolation.** If you are long one crude contract and short a related one (a calendar spread, say), your true one-day risk is far smaller than two outright positions, and SPAN charges you accordingly — a fraction of the gross. This is why spread trades and hedged books are so much cheaper to carry than naked directional bets: the clearinghouse margins your *net* one-day risk.

#### Worked example: a volatility-driven margin hike

You hold one WTI contract, comfortably margined at **\$6,000** when the market is quiet. Geopolitical news hits and oil's daily swings double. The clearinghouse, seeing a larger plausible one-day move, raises initial margin to **\$11,000** overnight. Your position has not changed and the price might not even have moved against you — but your account, sitting at \$6,200, is now far below the new requirement. You face a **\$4,800** margin call by morning or your broker liquidates. **The bond you must post is a function of how violent the market is, so the riskiest moments are precisely when futures demand the most cash to stay in the game.**

## Mark-to-market: the contract is re-priced every night

At the end of each trading day, the exchange publishes a **settlement price** for every contract — the official closing value. Then it does something that feels almost aggressive in its discipline: it **marks every open position to that settlement price** and moves cash accordingly. Winners get money credited to their accounts; losers get money debited. This daily ritual is called **mark-to-market**, and the cash that flows is called **variation margin**.

This is the loop in figure 2. You post initial margin. The next day the contract is marked to the new settlement. If you are up, cash appears in your account; if you are down, cash is pulled out. If the pull drags your balance below the maintenance floor, you get a margin call and must top back up to the initial level. If you fail to meet the call, your broker liquidates your position — not as punishment, but because an unfunded promise is exactly the thing the whole system exists to prevent.

The beauty of marking to market daily is that the clearinghouse is *never* exposed to more than one day of price movement on any account. Your accumulated profit or loss is not some abstract paper figure waiting until expiry — it has already been *realized in cash*, a little at a time, every night. By the time a contract expires, the gain or loss has been fully paid out along the way. The settlement at the end is almost an afterthought; the real money moved drip by drip.

#### Worked example: five days of variation margin

You go long one WTI contract at the Monday settlement of **\$72.00**, posting **\$6,000** initial margin. Maintenance is **\$5,500**. Watch the daily cash flows (each \$1.00 move = \$1,000):

- **Tue** settles at \$73.50. You are up \$1.50 = **+\$1,500**. Account balance: \$7,500.
- **Wed** settles at \$71.00. Down \$2.50 from Tuesday = **−\$2,500**. Balance: \$5,000. That is below the \$5,500 maintenance floor → **margin call**. You wire \$1,000 to restore the balance to the \$6,000 initial level.
- **Thu** settles at \$70.20. Down \$0.80 = **−\$800**. Balance: \$5,200.
- **Fri** settles at \$72.30. Up \$2.10 = **+\$2,100**. Balance: \$7,300.

Over the week the price went from \$72.00 to \$72.30 — net up \$0.30, worth \$300. And indeed: you started with \$6,000, wired in \$1,000 on the call, and ended with \$7,300, so \$7,300 − (\$6,000 + \$1,000) = **\$300** of profit, exactly the price change times 1,000. **Mark-to-market doesn't change your total P&L; it just collects it in daily installments so no loss is ever allowed to accumulate unfunded.**

This is also why a futures position can be *right* and still wipe you out. If the price ultimately moves your way but takes a deep, scary detour first, the daily margin calls along the route can exceed the cash you have. You are forced to liquidate at the bottom even though your thesis was correct. The clearinghouse does not care about your thesis; it cares about today's settlement. Survival in futures is a function of how much cash you can post against the *path*, not just the *destination*. Options traders manage this differently — buying a put caps your loss at the premium with no margin calls, which is one reason hedgers reach for them; see [Hedging a Portfolio with Options: Protective Puts, Collars, and Tail Risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk).

### What happens when a member actually defaults

We have said the clearinghouse "guarantees" the trade, but a guarantee is only as good as the capital behind it. So what *actually* happens if a clearing member fails to meet a margin call — if the loser truly cannot pay? The clearinghouse has a pre-defined sequence of resources it burns through in order, often called the **default waterfall**:

1. **The defaulter's own margin.** First, the clearinghouse seizes the collateral the failing member already posted. Because margin was sized to cover a one-day move and the position is marked daily, this usually covers the loss by itself. This is the entire reason the daily mark-to-market and the SPAN-sized bond exist: to make step 1 almost always sufficient.
2. **The defaulter's guaranty-fund contribution.** Every clearing member must also pay into a shared **guaranty fund** (or default fund). The failed member's own contribution is consumed next.
3. **"Skin in the game" — the clearinghouse's own capital.** Before reaching for other members' money, the clearinghouse must burn a tranche of its *own* capital, which aligns its incentives to set margins prudently.
4. **The surviving members' guaranty-fund contributions.** Only if the loss exceeds all of the above does the clearinghouse mutualize the remainder across the other members. This is the rare, catastrophic tail.

The reason this matters to *you*, even as a small trader, is that it explains why margin and daily settlement are non-negotiable and occasionally brutal. They are not bureaucratic friction; they are the load-bearing walls of a structure designed so that the chain of promises never breaks even when an individual link fails. Every margin call you grumble about is the system topping up step 1 so it never has to reach step 4.

## Delivery: physical settlement vs cash settlement

Every futures contract has to specify what happens if you hold it all the way to the end. There are two answers, and which one a contract uses changes its character completely.

**Physically settled contracts** end in the actual exchange of the commodity. If you are still long one WTI contract when it expires, you are obligated to *take delivery* of 1,000 barrels of crude oil at Cushing, Oklahoma, and pay the settlement price for them. If you are short, you must *make delivery* — produce the barrels and put them in the buyer's tank. WTI is physically settled. So are most metals on the LME and most agricultural contracts. The physical leg is real: there are tanks, pipelines, warehouses, inspectors, and a delivery process with rules about timing and quality.

**Cash-settled contracts** never involve the commodity at all. At expiry, the exchange looks up a reference price (an index of physical-market prices, usually) and simply pays the difference in cash. Nobody moves a barrel, an ingot, or a bushel. Brent crude is largely cash-settled against an index; many financial and index futures are cash-settled by necessity (you cannot deliver "the S&P 500"). Cash settlement is cleaner and avoids the logistics, but it relies on a trustworthy reference price.

Why does this distinction matter so much? Because **physical settlement is the anchor that ties the paper price to the real world.** A cash-settled contract converges to its reference index by definition. But a physically settled contract converges to the *actual cash market* because, at the very end, the two are interchangeable: if you hold the contract to delivery, you literally get (or give) the physical barrel. That interchangeability is what arbitrage uses to drag the futures price toward spot as expiry approaches — and what removes the floor under the price when the physical side breaks, as it did in April 2020.

#### Worked example: when physical delivery has nowhere to go

Roll the clock to April 2020. COVID lockdowns had crushed oil demand; refineries slowed; and the storage tanks at Cushing — the very delivery point for WTI — were filling toward their physical limit. Now put yourself in the seat of a trader holding **long** May WTI contracts into the final day, with no tank to receive oil and no intent to take 1,000 barrels per contract. Normally you would just sell to close. But on 20 April, almost *everyone* still long was in the same trap, and there were essentially no buyers willing to take delivery into full tanks. To escape the obligation to receive physical oil they could not store, longs had to pay someone — anyone — to take the contracts. The price of the *promise to receive a barrel you cannot store* collapsed through zero to **−\$37.63**. The cash barrel never went negative; the *contract* did, because the obligation attached to it had become a liability. **Physical settlement is what makes a futures price honest — and on rare days it is also what makes it terrifying.** (The full anatomy of that day is a Track H case study; here it is the vivid proof that delivery obligations are real.)

The negative print was a futures-market event, not a "the world will pay you to take oil" event. Knowing the difference is the entire payoff of understanding spot versus futures. We saw the same lesson, less dramatically, in the price history of WTI itself.

![WTI crude oil annual average 2000 to 2025 with the 2008 peak and the April 2020 negative settlement marked](/imgs/blogs/spot-vs-futures-the-two-prices-of-the-same-barrel-4.png)

## Why ~98% of contracts never deliver

If physical delivery is the anchor, you might expect delivery to be common. It is the opposite: the overwhelming majority of futures contracts — on the order of **98%** — are extinguished *before* expiry without a single barrel changing hands. This is not a loophole or a sign that the market is fake. It is by design, and it is the reason the market can be so deep and liquid.

A futures position can end in one of three ways. Figure 5 lays them out.

![Three ways a futures position ends offset roll or physical delivery with most volume offsetting](/imgs/blogs/spot-vs-futures-the-two-prices-of-the-same-barrel-5.png)

**1. Offset (the common case).** You took on an obligation by buying a contract; you discharge it by selling an identical contract before expiry. Because every contract is standardized, your buy and your sell are perfect opposites — the clearinghouse nets them to zero and you are *flat*, with no remaining obligation. Your profit or loss is just the difference between your buy and sell prices, which the daily mark-to-market has already paid you. A speculator betting on the oil price never wants the oil; they want the price change. They offset and walk away. This is most of the volume on any given day.

**2. Roll.** Suppose you want to stay exposed to oil *past* the front contract's expiry — an index fund tracking crude, say, or a producer hedging a long horizon. You "roll": you offset the expiring front-month contract and simultaneously open a position in the next month. You never take delivery; you keep your exposure alive by continuously moving it to a later contract. Rolling is mechanically routine but economically loaded, because the price you sell the old contract for and the price you buy the new one for are usually *different* — and that difference, the roll yield, is where the shape of the forward curve quietly eats or feeds long-only returns. That is the subject of [Contango vs Backwardation: What the Shape of the Curve Means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means).

**3. Deliver (the rare case).** A small minority — commercial players who genuinely want or have the physical commodity — hold to expiry and go through delivery. A refiner that needs crude, a merchant with oil to sell. For them the futures market is what it was originally built to be: a place to lock a price for a real physical transaction. This is the ~2%.

The crucial insight is that the rare 2% *disciplines* the common 98%. Because delivery is always *possible*, the futures price can never wander far from the cash price near expiry — if it did, a commercial player would arbitrage the gap by trading one and delivering into the other. The threat of delivery, even if rarely exercised, is the gravitational pull that keeps paper tethered to physical. A market where delivery were *impossible* would be a pure betting game with no anchor; the deliverable 2% is what makes the other 98% a real reflection of supply and demand.

#### Worked example: offset versus the cost of taking delivery

You are a speculator who bought one WTI contract at **\$72.00** expecting prices to rise. Three weeks later, with the contract near expiry, oil is at **\$76.00**. You have two real choices. **Offset:** sell one contract at \$76.00, netting (\$76.00 − \$72.00) × 1,000 = **+\$4,000** (already in your account via mark-to-market), obligation gone, done. **Take delivery:** hold to expiry, pay \$76.00 × 1,000 = **\$76,000** for 1,000 physical barrels at Cushing, then arrange a truck or pipeline nomination, find storage at maybe \$0.50–1.00 per barrel per month, line up an inspector, and eventually sell the oil into the cash market yourself. The financial result is similar, but the second path requires roughly \$76,000 of cash, a logistics operation, and a buyer for physical crude you have no use for. **For anyone without a tank and a reason, offsetting is not a preference — it is the only sane choice, which is precisely why 98% of contracts are offset or rolled.**

### Who is on the other side: hedgers and speculators

If 98% of contracts are offset and the people behind them mostly do not want oil, who are they and why does the market work? It helps to sort participants into two camps, because the futures market exists at the intersection of their needs.

**Hedgers** have real exposure to the physical commodity and use futures to *transfer* the price risk they do not want. A producer is naturally *long* physical oil (they own barrels coming out of the ground) and will *sell* futures to lock in a selling price. A refiner or an airline is naturally *short* the commodity (they must buy crude or jet fuel) and will *buy* futures to lock in a buying price. Hedgers are not trying to make money on the futures position; they are trying to make their core business predictable. Their futures gain or loss is meant to *offset* the change in the price of the physical barrels they will buy or sell anyway.

**Speculators** have no underlying physical exposure. They take a position purely to profit from the price moving the way they expect. They are often painted as villains, but they perform a genuine economic function: they provide the *liquidity* and the *willingness to take the other side* that lets a hedger transfer risk at a fair price. If a producer wants to sell 5,000 contracts of forward oil and the only natural buyers are a handful of airlines who happen to want exactly that amount on exactly that day, the price would gap around violently. Speculators stand ready to absorb the imbalance, smoothing the price and bearing the risk the hedger is paying to shed.

The elegant part is that **the futures price is the meeting point of these two crowds.** When more hedgers want to sell forward than buyers naturally want to buy, the futures price is bid down until speculators find it attractive enough to step in long; the reverse pushes it up. The price you see on the screen is the level at which the desire to offload risk and the appetite to take it balance. This is also why the basis carries information: a market with desperate physical sellers (a glut) prints a futures price low relative to spot, while a market with anxious physical buyers (a shortage) prints it high — the curve is a readout of which crowd is more desperate.

#### Worked example: a refiner's hedge that "loses" and still wins

An airline expects to buy 1,000,000 gallons of jet fuel (roughly the energy of a few hundred crude contracts; we will keep it to one contract for clarity) and fears prices rising. Today crude is **\$72.00**; the airline *buys* one WTI contract to hedge, posting \$6,000 margin. Three months later crude has risen to **\$82.00**. The airline's real fuel bill went *up* by \$10.00 × 1,000 = **\$10,000** on the physical barrels — bad. But the long futures position *gained* (\$82.00 − \$72.00) × 1,000 = **+\$10,000**, banked in cash via daily mark-to-market. Net effect on the airline: roughly zero — exactly the point. The hedge "made money" because prices rose, but the airline does not celebrate; the gain merely cancelled the higher fuel cost. **A hedge is judged a success not when the futures leg profits, but when the combined position removes the price risk — the futures gain and the higher physical cost are two sides of the same locked-in price.**

## Basis and convergence: the two prices must meet

We now have everything we need to understand the relationship that ties spot and futures together at the finish line. The difference between them has a name: the **basis.**

$$\text{basis} = \text{spot price} - \text{futures price}$$

(Conventions vary — some desks define it the other way around — but the idea is the same: basis is the *gap* between the cash market and a particular futures contract.) Early in a contract's life the basis can be sizeable, positive or negative, reflecting storage costs, financing, and how tight or loose the physical market is. But as the contract approaches its delivery date, something inexorable happens: **the basis shrinks toward zero.** Spot and futures **converge.**

![Basis equals spot minus futures pulled to zero at delivery as the two price lines converge](/imgs/blogs/spot-vs-futures-the-two-prices-of-the-same-barrel-7.png)

Why must they converge? Because at the instant of delivery, the futures contract and the physical barrel are *the same thing*. Holding the contract to expiry gets you (or obliges you to provide) the physical commodity at the delivery point. If, one day before delivery, the futures price sat well above the spot price, an arbitrageur would buy cheap physical oil in the cash market, sell the expensive futures contract, hold the contract to delivery, and hand over the oil they just bought — pocketing the gap for almost no risk. That trade, repeated by everyone who sees the gap, pushes the futures price down and the spot price up until the gap closes. The reverse arbitrage closes a gap in the other direction. **By expiry, the only price that survives is one where spot and futures are essentially equal.** Convergence is not a tendency or a rule of thumb; it is an arbitrage identity enforced by the deliverability we just discussed.

You can see the same convergence from the futures side, walking down the strip. The illustration below takes a contango-shaped strip — futures priced progressively *above* spot for later months — and traces what happens to one contract as it ages.

![Illustrative WTI forward strip with one contract aging down to the spot price at delivery](/imgs/blogs/spot-vs-futures-the-two-prices-of-the-same-barrel-3.png)

#### Worked example: the basis closing over three months

In January, spot WTI is **\$72.00** and the April futures contract trades at **\$75.30**, so the basis is \$72.00 − \$75.30 = **−\$3.30** (futures above spot — a market willing to pay to defer). Nothing fundamental changes about the oil, but the calendar advances. By mid-March, with April delivery a few weeks out, the gap has narrowed: spot \$73.40, April futures \$74.00, basis **−\$0.60**. By the final trading day in April, spot is \$73.90 and the expiring futures settles at \$73.95 — basis **−\$0.05**, a rounding error. The \$3.30 January gap did not vanish because oil rallied or crashed; it vanished because **the cost of carrying a barrel to April shrank to nothing as April arrived.** That is convergence in numbers: the basis is the price of *time*, and time runs out.

For a hedger this convergence is the whole point — it is what makes a futures hedge actually lock in a price. Think about why: a producer who sells futures today and intends to deliver (or to offset and sell physical) at expiry is relying on the futures price and the cash price ending up equal at the finish. If they did *not* converge, the hedge would spring a leak — the producer would be exposed to whatever gap remained. Convergence is the guarantee that the price they locked in is the price they actually receive, give or take a small, predictable local basis between the contract's delivery point and the producer's own location. That residual gap — Cushing versus a wellhead in the Permian, say — is called **basis risk**, and it is the one risk a futures hedge cannot fully erase: you can lock the *benchmark* price, but the difference between the benchmark and your local market still floats.

For a long-only investor convergence is the source of the roll drag (or roll gain): each time you roll, you are selling a near-converged contract and buying a more distant one priced by carry, and that carry leaks out of your returns in contango or pays you in backwardation. And for a trader watching the relationship, the *path* of the basis — whether it is widening or tightening, and how fast — is a live readout of how tight the physical market is right now versus what the paper market expects. A basis tightening faster than the calendar can justify often means the spot market is suddenly hot; a basis blowing out near expiry is the kind of tremor that preceded April 2020.

## Common misconceptions

**"Buying a futures contract means I bought oil."** No. You bought a standardized *promise* — an obligation to transact at a set price on a set date — backed by a performance bond. You own no oil and you paid for no oil. This is why the cash outlay (margin, ~\$6,000) is a fraction of the notional (~\$72,000): there is no asset to pay for, only a bond to post.

**"Margin is a down-payment, like a deposit on a house."** No, and this trips up almost everyone coming from stocks. Margin is **collateral on a promise**, not equity in an asset and not a borrowed sum. The clearinghouse lends you nothing; the margin simply guarantees you can cover daily losses. Calling it a down-payment makes the leverage feel safe — it is not. A \$2 move (under 3% of the price) is a 33% swing on a \$6,000 bond.

**"Oil really went to negative \$37 — they were paying people to take oil."** Half-true and badly misleading. The *futures contract* for May delivery settled negative; the physical cash barrel stayed positive. What went negative was the value of an *obligation to receive* barrels into tanks that were full, on the last day to escape that obligation. It was a delivery-mechanics squeeze, not a statement that crude oil itself was worthless.

**"Most futures end in someone delivering a truckload of stuff."** The opposite: roughly **98%** are offset or rolled before expiry; only ~2% reach delivery. Delivery is rare-but-possible, and the *possibility* is what keeps the price honest, not the frequency.

**"Spot and futures are basically the same number, so the distinction is academic."** They are tightly linked but distinct, and the distinction is where the money is. The gap (basis) encodes storage, financing, and physical tightness; it can be several dollars early in a contract's life and it tells you whether the market is short of barrels now or expects to be later. Treating them as one number throws away the most informative variable in the market.

**"The futures price is the market's forecast of the future spot price."** This is the most seductive error, and the cost-of-carry equation kills it. A six-month future at \$75.30 against a \$72.00 spot does *not* mean the market expects \$75.30 in six months. It means carrying a barrel for six months costs about \$3.30 net. If the market genuinely expected oil to *fall* to \$65 in six months but storage and financing cost \$3.30, the six-month future would still sit near \$75.30, not \$65 — because the arbitrage that pins futures to spot-plus-carry does not care what anyone forecasts. There *is* a forecast buried in the relationship between the curve and people's expectations, but the raw futures price is dominated by carry, not prophecy. Read the curve as a map of storage and scarcity, not as a crystal ball.

## How it shows up in real markets

**Reading the news ticker correctly.** When a headline says "oil jumped 4% today," it is quoting the front-month *futures* contract, not the spot cash price (though near expiry they are nearly identical thanks to convergence). On the day the front month expires and a new month becomes "front," the headline number can jump or drop simply because you are now quoting a different contract on a sloped curve — a "roll" in the index, not a real price move. A reader who knows the difference does not panic at a 2% "move" that is really the index stepping to the next contract.

**The 2008 spike and the path problem.** In July 2008 WTI futures peaked near **\$147** before collapsing to the low \$30s by December — the round trip is visible in the price-history chart above. Anyone who was *correctly* bearish in spring 2008 but short too early faced relentless margin calls as the price climbed another \$30 first. Being right about the destination did not save a position that could not fund the path. Conversely, longs riding the spike up banked daily variation margin in cash the whole way — and then handed it back, in cash, on the way down. Mark-to-market makes the rollercoaster a *cash-flow* event, not just a paper one.

**The producer's hedge.** A shale producer in 2019, watching U.S. output climb toward record levels (it hit ~13 million barrels a day by 2024), might lock in next year's price by *selling* futures against the barrels it will pump. It posts margin, marks to market daily, and at expiry either delivers physical crude or — far more likely — offsets and sells its actual oil in the local cash market, with the futures gain or loss offsetting the cash-price change. The futures market did exactly what it was built for: it let a real producer fix a price for real barrels months in advance. For how these flows interact with the macro picture, see [Commodities as Macro Signals: Oil, Copper, Gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold).

**Brent vs WTI — cash settlement vs physical.** WTI is physically settled at Cushing, an inland hub with finite tank space; Brent settles largely against a published index of North Sea cargoes. That structural difference is part of why the April 2020 negative print happened to *WTI* and not Brent: Brent's cash-settlement mechanism had no "you must receive barrels into a full tank" trap. The contract design is not a detail — it determines how a price behaves under stress. Gold tells the same story from the monetary side, where physical-vs-paper tensions show up at the COMEX delivery window; see [Gold Futures, COMEX, Contango, Backwardation, and Paper vs Physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical).

**The same machinery beyond oil.** Copper on the LME, corn at the CBOT, coffee on ICE — all run the identical engine: standardized contracts, a clearinghouse in the middle, initial and maintenance margin, daily mark-to-market, offset/roll/deliver, and convergence at expiry. Learn it once on a barrel of crude and you can read the wheat market, the copper market, and the cocoa market the same way. The metals complex in particular trades this way as the economy's pulse; see [Metals — Copper, Silver: The Economy's Pulse](/blog/trading/cross-asset/metals-copper-silver-the-economys-pulse).

**The storage glut behind the negative print.** Step back from the −\$37.63 headline and look at *why* the trap formed. In the weeks around April 2020, the WTI forward curve went into what traders called **super-contango**: the front contract was trading near \$20 while contracts a year out were near \$36–37. That enormous gap was the market screaming that physical oil *right now* was nearly worthless because there was nowhere to put it, while oil *later* — once tanks emptied — was worth far more. A gap that wide is itself an arbitrage signal: buy cheap physical oil, store it, and sell the far future for a locked-in \$16 profit per barrel. The catch is that the trade only works *if you can find storage*, and in April 2020 storage was the one thing that had run out. The super-contango was the curve's way of bidding frantically for tank space that did not exist. The negative front-month print was the same shortage seen from the other end: the last holders of expiring contracts had to pay to escape a delivery they could not physically absorb. Same scarcity, two prices. The shape of that curve, and what contango this steep means, is exactly the subject of A4.

**The 2022 nickel squeeze — when the exchange hit the brakes.** In March 2022, LME nickel went vertical, spiking intraday to a reported six-figure level (over \$100,000 a tonne) as a giant short position was caught in a buying panic. Mark-to-market would have generated margin calls in the *billions* against the shorts, threatening to cascade through the clearinghouse's default waterfall. The LME took the extraordinary step of *cancelling* a day of trades and halting the market — a vivid, controversial reminder that the clearinghouse-and-margin system is not a force of nature but an institution, run by people, that can and will intervene when the daily-settlement machine threatens to break the whole exchange. The lesson for a trader: the rules that protect you can also be changed on you in a true crisis. Unscheduled shocks like this are their own genre; see [Geopolitics, Elections, and Unscheduled Shocks](/blog/trading/event-trading/geopolitics-elections-and-unscheduled-shocks).

## The takeaway: how to read spot vs futures as a trader

You now have the mental toolkit. Here is how to *use* it.

**Always know which price you are looking at.** Spot is the cash barrel; the headline number is the front-month future. Near expiry they are nearly the same (convergence); far from expiry they can differ by several dollars (basis). When a price "moves" on a roll day, check whether the front contract just changed before you read meaning into it.

**Treat margin as a path budget, not a ticket price.** The question is never just "am I right about the direction?" It is "can I post enough variation margin to survive the path to being right?" Size positions so a plausible adverse swing — remember, a \$2 move is a third of a \$6,000 bond — does not force you out at the worst moment. The clearinghouse settles daily; plan for daily.

**Watch the basis, not just the level.** The gap between spot and the nearby futures, and whether it is tightening or widening, is a real-time gauge of physical tightness. A basis that snaps in faster than the calendar implies (futures racing down toward a firm spot) often signals a market suddenly short of physical barrels right now. A basis that blows out can signal a storage glut or a delivery squeeze building — the early tremor before something like April 2020.

**Respect physical settlement.** If you ever hold a physically settled contract toward expiry without the ability and intent to deliver or receive, you are standing in the wrong place. Offset or roll well before the delivery window. The 2% who deliver are commercials with tanks and trucks; the 98% who do not are everyone else, and there is no prize for finding out the hard way which group you are in.

Step back and the spine of this whole series comes into focus. **A commodity price is a physical thing forced through a financial contract.** Spot is the physical thing; the futures price is the contract; margin and mark-to-market are the plumbing that lets strangers trade the contract safely; delivery is the rare-but-real event that keeps the contract honest; and basis is the living measure of the gap between the two. Everything else in commodities — the forward curve, contango and backwardation, the convenience yield, who profits from the roll — is built on this foundation. Master the two prices of the same barrel and the rest of the market stops looking like noise and starts looking like a structure you can read.

The next step is to stop looking at *one* futures price and look at *all of them at once* — the strip of prices across every delivery month — because that single line tells you the market's entire story about supply, demand, storage, and time. That is the forward curve, and it is where we go next.

## Further reading & cross-links

- [What Is a Commodity? The Physical Asset That Trades on Paper](/blog/trading/commodities/what-is-a-commodity-the-physical-asset-that-trades-on-paper) — the series opener and the physical-vs-paper thesis this post builds on.
- [The Forward Curve: The Most Important Chart in Commodities](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — from one futures price to the whole strip across maturities.
- [Contango vs Backwardation: What the Shape of the Curve Means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — what the two curve shapes say about supply and demand, and why shape drives the roll.
- [Gold Futures, COMEX, Contango, Backwardation, and Paper vs Physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — the same paper-vs-physical mechanics in the monetary metal (note: gold is a monetary asset, crude an industrial one).
- [What Sets an Option's Price? The Five Inputs and the Intuition](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition) — the other major derivative, and why a futures payoff is linear where an option's is not.
- [Hedging a Portfolio with Options: Protective Puts, Collars, and Tail Risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) — capping losses without margin calls, the alternative to a futures hedge.
- [Commodities as Macro Signals: Oil, Copper, Gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) — what these prices say about the economy.
- [Metals — Copper, Silver: The Economy's Pulse](/blog/trading/cross-asset/metals-copper-silver-the-economys-pulse) — the same futures machinery in the industrial-metals complex.
- [Commodity Trading Houses: Glencore, Vitol, Trafigura](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura) — the merchants who actually take delivery and move the physical barrels.
