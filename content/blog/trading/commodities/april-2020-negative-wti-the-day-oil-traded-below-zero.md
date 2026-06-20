---
title: "April 2020 Negative WTI: The Day Oil Traded Below Zero"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "On 20 April 2020 the front-month WTI crude contract settled at minus 37.63 dollars a barrel, the first negative oil price in history. This is the full mechanics: how a demand collapse plus full storage plus physical settlement trapped longs who had to pay to escape delivery, why Cushing could go negative but seaborne Brent could not, and the playbook lesson that paper is not physical at expiry."
tags: ["commodities", "negative-oil", "wti", "crude-oil", "physical-settlement", "cushing", "contango", "futures-expiry", "uso", "roll-yield"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — On 20 April 2020 the expiring May WTI crude futures contract settled at **−\$37.63** a barrel. It was not a glitch. A physically-settled future is a *claim on a real barrel at a real place* — and when COVID had crushed demand, storage at Cushing, Oklahoma was nearly full, and longs facing delivery had no tank and no buyer, the only way out was to **pay someone to take the obligation**. The price fell through zero to clear.
>
> - **Why it can go negative:** a futures long who holds to expiry must take physical delivery. If you cannot store the barrel and cannot find a buyer, that barrel is a *liability*, not an asset — and a liability has a negative price.
> - **Why Cushing:** WTI delivers into landlocked tanks at Cushing with finite space and no port. When the tanks filled, there was nowhere for the oil to go. Seaborne **Brent** never went negative — a tanker can always sail somewhere.
> - **The setup was super-contango:** the front month traded near zero while distant months were \$30–\$36 higher — the curve was screaming "there is nowhere to put this oil."
> - **The carnage was on the longs who didn't roll:** the **USO** ETF and China's retail **"Crude Oil Treasure"** product held the front contract straight into the collapse. Anyone who had **rolled early** took only the spot-price loss, not the negative print.
> - The one number to remember: **−\$37.63**. The difference between paper and physical is not academic. At expiry, it is everything.

On the morning of 20 April 2020, a junior trader at a London commodity desk refreshed his screen and assumed it had frozen. The May WTI crude futures contract — the most-traded oil instrument on Earth, the number that anchors the price of gasoline from Texas to Tokyo — was quoted at a few dollars and falling fast. Then it touched zero. Then it kept going. By the 2:30 p.m. New York settlement, the contract printed **−\$37.63** a barrel. For one surreal afternoon, the *seller* of a barrel of crude oil had to pay the *buyer* almost forty dollars to take it away.

Every headline reached for the word "impossible." Oil is the most valuable bulk commodity humanity produces; the idea that a barrel could be worth *less than nothing* offended common sense. And yet there it was, settling on the tape, a number the Chicago Mercantile Exchange had to scramble to even allow its software to display. Within hours, ordinary investors halfway around the world — people who had bought what they thought was a simple bet that cheap oil would recover — discovered they had lost not just their stake but, in some cases, *more than their stake*. The number on their statement was negative too.

This post is about exactly what broke that day, and why it was not a glitch but a lesson — the cleanest lesson in this whole series. A commodity future is a *physical thing forced through a financial contract*. Most of the time the financial wrapper is so smooth that you forget there is a real barrel underneath. On 20 April 2020, the wrapper tore and the barrel showed through. We will build the mechanism from zero: what physical settlement means, why a barrel can become a liability, the super-contango that warned of it, why Cushing could go negative when Brent could not, the retail products that got destroyed, and the playbook every reader should take away — the difference between paper and physical, at the one moment it bites.

![How oil traded below zero: a cascade from demand collapse to full storage to a trapped long who must pay to escape the contract](/imgs/blogs/april-2020-negative-wti-the-day-oil-traded-below-zero-1.png)

The figure above traces the whole cascade — demand collapse, storage fills, expiry arrives, the long is trapped, the price goes below zero — and the rest of this post is the dollar-by-dollar story of each link in that chain.

## Foundations: what "physical settlement" actually means

Let us start with no finance vocabulary at all, because the entire event turns on one idea most people never have to think about: what *is* a futures contract a promise to do?

A **futures contract** is a standardized, exchange-traded agreement to buy or sell a fixed quantity of something at a fixed price on a fixed future date. One WTI crude contract is a promise about **1,000 barrels** of a specific grade of light sweet crude oil. If you *buy* a contract — go **long** — you have agreed to *receive* 1,000 barrels and pay the agreed price. If you *sell* a contract — go **short** — you have agreed to *deliver* 1,000 barrels and receive the price. (For a fuller treatment of how the futures price and the cash price relate, see this series' [spot vs futures](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel) post; here we only need the delivery mechanics.)

Now, the crucial fork. When a contract reaches its **expiry**, how does the promise get fulfilled? There are two ways an exchange can design this, and the difference is the whole story:

- **Cash settlement.** At expiry, no physical goods change hands. The exchange looks up a reference price, and longs and shorts simply exchange the cash difference between the contract price and that reference. The stock-index future on the S&P 500 works this way — nobody delivers 500 companies; you just settle the dollar gap. This is the painless kind.
- **Physical settlement.** At expiry, the short must *actually deliver the real commodity* to a real location, and the long must *actually take it*. WTI crude is physically settled. If you are long a May WTI contract at its expiry and you have not closed the position, you are legally obligated to **show up at Cushing, Oklahoma, and accept 1,000 barrels of crude oil into a tank** — and to pay for them, and to have somewhere to put them.

The reason WTI is physically settled at all is worth a sentence, because it is a feature, not a flaw. Physical settlement is what keeps the futures price *honest* — anchored to the real cash market for actual barrels. If a contract could only ever be cash-settled against some reference index, that index itself could drift from physical reality and be gamed. By forcing the contract to terminate in a real delivery of real oil at a real place, the exchange guarantees that the paper price and the physical price must converge at expiry, because anyone could otherwise arbitrage the gap by buying the cheaper one and delivering into the dearer one. That convergence is normally a healthy discipline. April 2020 was the day the discipline turned savage: the paper and the physical converged, all right — they converged at minus thirty-seven dollars, because that was the true clearing price of a physical barrel nobody could store.

That last sentence is the entire event in miniature. Most people who trade oil futures are not refiners; they are speculators or funds who never intend to touch a physical barrel. They expect to **close their position** — sell the contract they bought — before expiry, pocketing or eating the price change in cash, exactly as if it had been cash-settled. And in normal times that works perfectly, because there is always a buyer on the other side: a refiner, a trading house, a merchant who *does* want the oil and is happy to take the long position off your hands.

The negative price happened on the one day when that assumption failed. The speculators wanted out. But the natural buyers — the people who could actually take delivery — had **nowhere to put the oil** and so refused to bid. With no buyer, a long who could not take delivery was trapped against a hard physical obligation. And here is the pivot the whole series rests on: a barrel you are *obligated to receive* but *cannot store* is not an asset worth a positive price. It is a **liability** — a problem you must pay to dispose of. And a liability trades at a *negative* price.

### Why a barrel can become a liability, not an asset

Hold a barrel of oil in your hand (figuratively) and ask: what is it worth? In normal life the answer is obviously positive — oil is energy, it is fungible, somebody will pay you for it. But "somebody will pay you for it" hides an assumption: that you can *deliver it to that somebody*, or *hold it until they want it*. Strip those away and the barrel's value can invert.

Suppose you are contractually handed 1,000 barrels at Cushing tomorrow. To convert that oil into money you must either (a) sell it immediately to a buyer who takes delivery there, or (b) store it in a tank until a buyer appears. If every tank within reach is full and every potential buyer is *also* drowning in oil they cannot store, then neither path exists. Now the oil is just sitting on your account as a thing you legally own, accruing storage and demurrage charges you cannot avoid, with no way to turn it into cash. You would happily pay someone to take it off your hands — and the amount you would pay is exactly how negative the price can go before someone with *one last sliver of tank space* finally steps in.

### Margin, marking-to-market, and the clearing house

There is one more piece of plumbing that makes the negative print bite harder than a simple "the price fell" story suggests, and it is worth building from zero because it explains why nobody could just look away and wait.

When you trade a future, you do not pay the full value of the 1,000 barrels up front. You post **margin** — a good-faith deposit, typically a small fraction of the notional, that the exchange holds as collateral. For oil, margin might be a few thousand dollars to control a contract worth tens of thousands. This is **leverage**: a small deposit controls a large notional, so gains and losses are magnified relative to the cash you put down.

Sitting between every buyer and every seller is a **clearing house** — the exchange's central counterparty. Every open position is **marked to market** every single day: the clearing house computes the change in the contract's value and moves cash between accounts accordingly. If your position lost \$5,000 today, \$5,000 is debited from your margin tonight; if your margin falls below the required level, you get a **margin call** and must wire fresh cash by morning or the broker **liquidates** your position — sells you out at the market, whatever the market is.

Now layer this onto 20 April. As the May contract fell through the afternoon, longs were taking enormous mark-to-market losses in real time. Many would have received margin calls; many would have been *force-liquidated* by their brokers' risk systems — which means their longs were being **dumped into the market** precisely when there was no bid. Forced selling into no bid is how a falling price becomes a vertical one. The clearing mechanism that normally keeps the system safe became, for a few hours, an accelerant: each liquidation pushed the price lower, which triggered more margin calls and more liquidations. The negative settlement was partly the sound of risk systems all hitting the sell button at once into an empty order book.

This also dismantles the comforting thought "I'll just hold and wait for it to recover." You *cannot* hold a leveraged future through a 50-point adverse move and wait — the daily margin mechanics force cash out of your account every night, and once you cannot meet the call, the decision to stay or go is taken away from you. The clearing house does not care about your thesis; it cares about collateral. That is the difference between owning a barrel outright in a tank you control and being long a margined contract: the barrel can sit there for a year while you wait, but the contract marks you to market every night and can eject you at the worst possible moment.

#### Worked example: the negative-settlement P&L on one contract

Make this concrete in dollars. You are long **one** May WTI contract — 1,000 barrels. You bought it a week earlier at, say, \$15 a barrel, expecting a cheap-oil rebound. As expiry approaches on 20 April 2020 you have not closed it. The market gaps lower all afternoon, and the contract **settles at −\$37.63**.

Your profit and loss has two pieces. First, the price moved from your \$15 entry to −\$37.63, a drop of \$52.63 a barrel. On 1,000 barrels that is a loss of:

```
1,000 barrels  x  $52.63 per barrel  =  $52,630 loss on the price move
```

But the settlement price itself is *negative*, which means something stranger than a normal loss. To be flat at a settlement of −\$37.63, you must effectively **pay \$37.63 a barrel** to the party taking the other side — you are paying them to relieve you of the delivery obligation:

```
1,000 barrels  x  $37.63 per barrel  =  $37,630 PAID OUT to escape delivery
```

You did not just lose your investment. You *handed over* \$37,630 on top, simply to not have to receive oil you could not store. A long position you thought capped your loss at "the price goes to zero" instead drained your account *below* zero. The single intuition to carry out of this example: with physical delivery and full storage, a long's loss is not bounded by zero, because the thing you are forced to receive can itself be worth less than nothing.

## The setup: a demand collapse the world had never seen

Negative prices do not come from nowhere. They are the end of a chain, and the chain started with the fastest demand shock in the history of oil.

In the first quarter of 2020, COVID-19 lockdowns spread across the planet. Roads emptied. Airlines parked their fleets — global flights fell by more than half. Factories idled. Oil is consumed when things and people *move*, and almost overnight, things and people stopped moving. Global oil demand, which had been running around **100 million barrels per day**, fell by an estimated **20 to 30 million barrels per day** at the trough — roughly a quarter to a third of all demand, *gone*, in a matter of weeks. Nothing in the modern oil era — not the 2008 crisis, not the 1970s shocks — had erased demand that fast.

Here is the asymmetry that turned a demand shock into a storage crisis: **supply could not fall as fast as demand did.** An oil well is not a faucet. Shutting in a producing well can damage the reservoir, is expensive to reverse, and for many fields is a decision measured in months, not hours. Meanwhile OPEC and Russia spent early March 2020 in a *price war*, briefly *raising* output just as demand was vanishing — the worst possible timing, which they reversed in April, but too late to matter for the May contract. So in the spring of 2020 the world was producing roughly normal volumes of oil into a world that had stopped using a third of it. Every single day, millions of unwanted barrels had to go *somewhere*.

That somewhere is storage. And storage is finite.

![The first negative oil price in history: WTI annual averages with the minus 37.63 settlement of 20 April 2020](/imgs/blogs/april-2020-negative-wti-the-day-oil-traded-below-zero-2.png)

The chart above sets the scale. WTI's annual averages had swung from a \$147 intraday peak in 2008 to the high-90s, low-40s, and back — but the line never came near the zero axis until 2020, when a single contract's settlement punched clean through it to −\$37.63. That red dot is not an annual average; it is one afternoon's settlement, plotted to show how far below every prior data point in oil's history it sat.

### Where the oil goes: the world's tanks fill up

When more oil is produced than consumed, the surplus accumulates in **inventory** — tank farms, pipelines, salt caverns, and, when those run short, ships. Through March and April 2020, inventories everywhere swelled at a record pace. Traders chartered supertankers purely as **floating storage**, paying enormous day-rates just to park oil on the water (a trade this series covers in [cash-and-carry and storage arbitrage](/blog/trading/commodities/cash-and-carry-and-storage-arbitrage-locking-in-the-curve)). The whole world was scrambling for a place to put barrels.

The pinch point for the WTI contract specifically was a small town in Oklahoma.

## Why Cushing: the bathtub that filled

The WTI futures contract does not settle "somewhere in America." It settles into tanks at one specific delivery hub: **Cushing, Oklahoma**, a town of fewer than 10,000 people that happens to sit at the crossroads of a dense web of crude pipelines. Cushing is the official delivery point written into the contract. A short delivers oil there; a long takes oil there.

Two facts about Cushing made it the epicenter:

1. **It is landlocked.** Cushing is in the middle of the continent, hundreds of miles from any coast. Oil arrives and leaves by pipeline and tank, not by ship. There is no port, no fleet of tankers idling offshore to soak up surplus. When Cushing's tanks fill, the oil cannot simply sail away.
2. **Its tank space is finite and was nearly full.** Cushing's working storage capacity is roughly **76 million barrels**. Through early 2020, inventory there climbed relentlessly toward that ceiling. By mid-April, with deliveries still pouring in and almost nobody pulling oil *out*, the practical usable space was nearly exhausted. Worse, much of the remaining nominal capacity was already *leased and committed* — booked by traders who had locked it in weeks earlier — so the space available to a new, desperate buyer on 20 April was effectively zero.

![Why WTI went negative and Brent did not: landlocked Cushing tanks versus a seaborne North Sea benchmark](/imgs/blogs/april-2020-negative-wti-the-day-oil-traded-below-zero-4.png)

The before-and-after figure draws the contrast that explains the entire WTI-versus-Brent divergence. Cushing is a *bathtub* with a fixed volume and one drain; once it fills, a barrel delivered there has nowhere to go. **Brent**, the North Sea benchmark, settles against *seaborne cargoes* loaded onto tankers — and a tanker can sail anywhere a buyer exists on the planet. Brent crashed hard in April 2020, into the teens, but it never went negative, because there was always *somewhere* to physically send the oil. The negative print was a landlocked-delivery problem, not a "the world ran out of oil value" problem.

![The bathtub fills: Cushing inventory climbing toward its tank tops through spring 2020](/imgs/blogs/april-2020-negative-wti-the-day-oil-traded-below-zero-6.png)

The inventory chart (illustrative in its exact weekly figures, but accurate in shape and ceilings) shows the bathtub filling week by week toward the ~76-million-barrel working capacity and the lower practical ceiling. The vertical line marks the 20 April expiry, arriving precisely as the tanks approached their tops. The timing was the trap: the May contract's last gasp coincided with the moment Cushing had the least room it had had in years.

#### Worked example: the storage-full mechanic, in numbers

Put yourself in the seat of a long holding one May contract on the morning of 20 April, deciding whether to take delivery. Taking delivery means receiving 1,000 barrels at Cushing. To make that pay, you need to store the oil until prices recover, then sell it. So you call around for tank space.

Suppose — and this is roughly what happened — that essentially **no uncommitted tank space** is available at any price you can reach, and the few crumbs that exist are quoted at panic rates. Even if you *could* find a tank, the math is brutal. Say a sliver of space is offered at \$2 a barrel per month and you think prices recover in three months:

```
Storage cost  =  1,000 barrels  x  $2/bbl/month  x  3 months  =  $6,000
```

And that is the *optimistic* case where space exists. In the realistic case there is **no tank at all**, so the storage cost is not \$6,000 — it is *infinite*, because the service simply cannot be bought. A barrel you are forced to receive and physically cannot store has, to you, a value below zero by however much it costs to make the problem go away. When the cheapest way to make the problem go away is to *pay another trader to take your long position*, the price of the contract is whatever payment clears that trade. On 20 April, that clearing payment was \$37.63 a barrel. The takeaway: the price did not go negative because oil became worthless; it went negative because *the obligation attached to it* became a cost you would pay to be rid of.

### When the land tanks fill, the trade goes to sea

There is a release valve in the global system that explains both why the rest of the world did not go as negative as Cushing and why the relief came too late for the May contract. When onshore storage fills, traders turn the ocean itself into a tank by chartering **supertankers** — Very Large Crude Carriers, each holding around two million barrels — and parking them, full, offshore. This is **floating storage**, and in April and May 2020 the world hired an enormous fleet of these ships to do nothing but hold oil and wait for prices to recover. Day-rates for the largest tankers spiked from tens of thousands of dollars a day to *hundreds* of thousands, because the demand to store oil at sea was suddenly desperate.

Floating storage is the cash-and-carry trade taken to its physical limit, and it is profitable only when the contango is steep enough to pay for the freakishly high ship rates. In the super-contango of 2020 it was wildly profitable — which is exactly why so many tankers were hired. But here is the trap for the WTI May contract: a barrel that must be *delivered at Cushing* cannot be loaded onto a tanker, because Cushing is hundreds of miles inland with no port. The floating-storage relief valve was available to seaborne grades — to Brent, to crude that could reach a coast — but **not** to oil locked into landlocked delivery. The sea was filling with rescued barrels while the trapped longs at Cushing had no pipeline of escape. The geography that makes WTI a clean domestic benchmark in normal times made it a cage in April 2020.

This is the deeper version of the Cushing-versus-Brent point: it is not merely that Brent settles seaborne and WTI settles inland, but that the *entire global storage system has an offshore overflow* that only seaborne crude can reach. WTI's delivery point sat on the wrong side of that overflow. When the prompt month needed somewhere to put barrels in the final hours, the world's spare storage was floating on the ocean, and there was no way to get a Cushing barrel onto it.

## The mechanism at expiry: the trap closes

We can now assemble the full mechanism with the precision it deserves. The negative print was the product of a **physical-delivery contract**, a **full delivery point**, and a population of longs who **could not take delivery** — all colliding in the final hours before the contract stopped trading.

Walk through the trap as a long experiences it. As expiry nears, you have three theoretical exits.

![The expiry trap: a physical-delivery long runs out of exits as storage fills and buyers vanish](/imgs/blogs/april-2020-negative-wti-the-day-oil-traded-below-zero-5.png)

The decision-tree figure lays out those three branches, and it shows how, on 20 April, two of the three doors were nailed shut:

1. **Roll the position** — sell the prompt contract and buy the next month. This is what almost everyone *should* do and what most professionals *did* do days or weeks earlier. Rolling means you never touch a barrel; you just keep your exposure in a later contract and pay the spread between months. This door was open — but only if you walked through it *early*. By the final day, the cost of rolling had exploded (see the super-contango below), and the latecomers found it ruinous.
2. **Take delivery** — accept the 1,000 barrels at Cushing. This door required a tank. With Cushing full, there was no tank, so for most longs this door was *physically* shut. You cannot accept a barrel you have nowhere to put.
3. **Sell to close before the last trade** — find a buyer and hand them your long. This is the normal escape. But on 20 April everybody holding the May contract wanted to do the same thing at the same time, and the natural buyers (refiners, storage owners) had no room either, so there was *no bid* — or rather, the only bids that appeared were deeply negative, from the handful of players who still had a scrap of tank space and would take the oil *only if paid handsomely to do so*.

With door 2 shut by full tanks and door 3 collapsing for lack of buyers, the trapped longs had to accept whatever price cleared the sells. The price fell, and fell, through zero, and kept falling until it reached the level at which someone with the last available barrel of storage capacity was finally willing to be the buyer — being *paid* \$37.63 a barrel to take oil they could just barely squeeze into a tank. That clearing level was the settlement. The trap had closed.

#### Worked example: what a buyer needed to be paid to say yes

Look at the trade from the *other* side — the rare party who was willing to be the buyer at a negative price — because it shows exactly where −\$37.63 comes from rather than some other number. To be the buyer, you had to be one of the few with a genuine sliver of usable tank space at or near Cushing, and you had to be compensated for everything that came with taking 1,000 unwanted barrels.

Tally what that buyer faced. They must pay to **lease and operate** the tank space for however long until prices recover, pay **pipeline and handling** fees to get the oil into the tank, tie up **capital** in oil they cannot immediately sell, and bear the **risk** that prices fall further or that storage gets even tighter. Roughly:

```
Storage + handling over the hold     :  about $20 per barrel
Capital tie-up + risk premium        :  about $18 per barrel
Total compensation the buyer demands :  about $38 per barrel
```

A buyer who needs about \$38 a barrel of compensation to take the oil will only say yes if they are *paid* about \$38 a barrel — that is, if the price is about **−\$38**. The settlement at −\$37.63 is simply the number at which the marginal buyer's compensation demand met the trapped longs' desperation. The intuition to keep: the depth of a negative price is not arbitrary; it is the *cost of being the last party willing to absorb the physical commodity*, and that cost is exactly what the convenience yield — normally a premium for holding the physical thing — looks like when it inverts into a penalty.

### The super-contango that warned of it

The market had been screaming this warning for weeks, in the shape of the forward curve. **Contango** is when distant-month futures cost *more* than the prompt month — an upward-sloping curve. (Its causes and trading get a dedicated post: [contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means).) A mild contango is normal and reflects the cost of carrying oil through time. What appeared in April 2020 was not mild. It was **super-contango**: the front month near zero while contracts a year out traded \$30 or more higher.

![Super-contango: a front month near zero while the forward strip is worth far more](/imgs/blogs/april-2020-negative-wti-the-day-oil-traded-below-zero-3.png)

The strip chart shows the shape. The prompt contract on its expiry day pierced below zero to −\$37.63, while December delivery sat around \$36 — a spread of more than \$70 across the curve. A forward curve that steep is the market's way of shouting a single sentence: *there is nowhere to put this oil right now, but if you can hold it for a year we will pay you handsomely.* The carry was enormous precisely because storage was the binding constraint. Anyone reading the curve in early April knew the prompt was in mortal danger; the only question was exactly how low it would go.

#### Worked example: rolling early would have saved you

Here is the dollar difference between respecting the curve and ignoring it. Take two longs, each holding one May contract in early April, each with the same view that oil will recover. Both want to stay long oil; neither wants to take physical delivery.

**Trader A — rolls early.** In early April, with the May contract trading around \$25 and the June contract around \$29, Trader A rolls: sells May at \$25, buys June at \$29. The roll costs the \$4 spread:

```
Roll cost  =  1,000 barrels  x  ($29 - $25)  =  $4,000 paid to move to June
```

Trader A is now long June, far from the expiry meltdown. When May later collapses to −\$37.63, Trader A doesn't care — they are not holding May. Their loss is just the ordinary spot-price decline of a far-month contract, a few thousand dollars, fully under control.

**Trader B — waits.** Trader B holds the May contract into the final session, planning to "sell near the close." On 20 April there is no bid, the contract settles at −\$37.63, and Trader B's loss, from the same ~\$25 starting point, is:

```
1,000 barrels  x  ($25 - (-$37.63))  =  1,000  x  $62.63  =  $62,630 loss
```

Same starting position, same market view, same instrument. The only difference is that Trader A paid a known \$4,000 to step out of the delivery month *before* the trap, and Trader B paid \$62,630 by staying in it. The lesson is the spine of this entire series: **the roll exists for a reason.** It is the price you pay to keep paper exposure without ever colliding with the physical barrel. Skipping it to save the roll cost is, at expiry, the most expensive economy in commodities.

### The CME had to let prices go negative

A small technical detail captures how unprecedented the moment was. Most exchange software, for its entire history, assumed commodity prices were positive — many systems simply could not *represent* a negative oil price; the field was unsigned. Just days before expiry, on 15 April 2020, the CME issued a notice that it had tested and was prepared to support **negative prices and negative strikes** in its energy contracts. At the time the notice read as routine risk housekeeping. In hindsight it was the exchange quietly confirming that the thing about to happen was not only possible but *expected* by the people closest to the plumbing.

This matters for how you read the event. A negative price was not a system failing to handle reality; it was a system being *upgraded to handle a reality the fundamentals had already made inevitable*. The exchange could no more stop the price going negative than a thermometer can stop a fever. The job of the settlement is to find the price at which buyers and sellers clear, and on 20 April that price was below zero. The CME's notice simply made sure the screens could show the truth.

### The anatomy of one afternoon

It helps to walk the actual sequence of 20 April 2020, because the negative print was not a single instant but the last few hours of a slow vise closing. The May contract was due to stop trading the next day, 21 April, so 20 April was effectively the last full session for anyone who still needed to get out.

Through the morning the contract was already weak, trading in the low teens and sliding. The fundamental backdrop was fixed and known: Cushing was nearly full, the next several weeks of deliveries were already committed, and the population of longs still holding the May contract was dominated by financial players — index funds, retail products, speculators — who had no intention or ability to take physical oil. The natural buyers who *could* take delivery had already secured the storage they needed and had no reason to bid for more.

As the afternoon wore on, the selling fed on itself. Every long that tried to exit added to the supply of contracts; every buyer that might have stepped in looked at full tanks and stepped back. By early afternoon the price had collapsed to single digits, then to a couple of dollars. Then, in the final stretch before the 2:30 p.m. settlement window, it crossed zero — and instead of stopping, it accelerated, because there was simply no level at which a constrained buyer would willingly absorb more deliverable oil until the price compensated them for the cost and risk of squeezing it into the last sliver of space. The settlement was calculated from trading in that final window, and it landed at **−\$37.63**. The next day the contract expired, and a handful of longs who still had not exited faced the actual delivery obligation.

The shape of the day is the lesson: there was no crash, no single catastrophic headline at 2:00 p.m. There was a *deadline* — expiry — bearing down on a crowd that could not meet it, and a price that fell as far as it had to in order to find the one party who could. The violence came from the *combination* of a hard date and a hard physical limit. Remove either — give the longs another month, or give Cushing another 50 million barrels of empty tanks — and the price never goes negative.

## The carnage: who actually got destroyed

The professionals — refiners, trading houses, sophisticated funds — mostly saw this coming and were out of the May contract well before the close. The people who got destroyed were the ones who had bought *the front-month exposure as if it were a simple, durable bet on the oil price*, without understanding the physical-delivery machinery underneath. Two cases stand out.

### USO: the ETF that was always front-month

The **United States Oil Fund (USO)** is an exchange-traded product that lets ordinary investors "buy oil" through a normal brokerage account. As cheap oil looked like a bargain in spring 2020, retail money poured into USO at record pace — investors who, quite reasonably, thought they were buying a recovery play. The problem is *how* USO got its oil exposure: by holding **front-month WTI futures**, the very contracts at the center of the storm.

A fund that lives in the front month must continuously **roll** — sell the expiring contract and buy the next one — to avoid taking delivery. In normal markets the roll is a quiet cost. In a super-contango it is a bleeding wound: every roll meant selling a cheap prompt contract and buying a much pricier later one, locking in a loss on the spread again and again. This is the structural reason long-only commodity ETFs bleed in contango, which this series treats in full in [roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed). USO did not settle *at* −\$37.63 — its managers scrambled to roll *out* of the May contract before the worst of expiry, and in the days around the event the fund repeatedly changed its strategy, pushing its holdings into later months and spreading across the curve to survive at all.

But the damage was done in the price. Even though USO itself never printed negative, the super-contango and the front-month collapse devastated its net asset value. Investors who bought USO betting on a \$10-to-\$40 oil round trip discovered that even when oil *did* recover, USO recovered far less — the roll cost in contango had quietly eaten the upside. The fund later did a reverse split and changed its mandate. The retail lesson: **"buying oil" through a front-month product is not the same as owning a barrel.** You own a contract that must keep paying the roll, and in a storage crisis that roll is brutal.

#### Worked example: the USO-style roll bleed

Watch the spot price round-trip while the investor still loses. Suppose oil's *spot* price goes from \$20, down to \$15, and back up to \$20 over several months — a full round trip, no net change. A naive investor expects to break even. But a front-month fund in steady contango pays the roll each month. Using the illustrative contango-drag pattern this series tracks, indexed to 100 at the start:

```
Spot price index        :  100 -> 98 -> 103 -> 99 -> 105 -> 101   (roughly flat)
Long-only total return  :  100 -> 92 -> 90 -> 80 ->  78 ->  70   (the roll drag)
```

The spot price ends essentially where it started — barely changed — while the long-only roll-paying position has bled to about **70**, a 30% loss, purely from rolling through contango month after month. No single dramatic crash; just the relentless tax of buying expensive far months and selling cheap prompt ones. The intuition to keep: in a deep-contango regime, you can be *exactly right on the direction of the spot price* and still lose badly, because the instrument you used to express the view is a roll machine, not a barrel.

### "Crude Oil Treasure": the retail product that owed *more* than the stake

The most painful story came from China. The **Bank of China** offered a retail investment product nicknamed **"Crude Oil Treasure"** (原油宝), which let ordinary Chinese savers — tens of thousands of them, many with no derivatives experience — take positions linked to WTI crude futures through their banking app. Crucially, the product tracked the **front-month WTI contract**, and the bank's mechanism rolled it on a schedule that left a large block of investors *holding exposure linked to the May contract right at its expiry settlement*, at the −\$37.63 print.

Because the product was linked to the actual settlement price, and because futures losses are not capped at the amount invested, the retail holders did not merely lose their deposits. The negative settlement meant their positions were marked at a loss *exceeding* their principal — the bank calculated that customers **owed additional money** on top of losing everything they had put in. People who had invested the equivalent of a few thousand dollars received statements showing they were now *in debt* to the bank for thousands more. The episode became a national scandal, triggered lawsuits and regulatory intervention, and forced the bank into negotiated settlements with furious customers.

#### Worked example: how a retail saver ended up owing money

Make the arithmetic explicit, because the "owe more than you invested" outcome is the most counterintuitive part. Suppose a retail saver put in the equivalent of **\$3,000** to take a long position linked to roughly **1,000 barrels** when the contract was around \$20 a barrel — using the product's leverage so that \$3,000 of margin controlled a barrel-scale notional. The contract settles at **−\$37.63**. Their loss on the move from \$20 to −\$37.63 is:

```
1,000 barrels  x  ($20 - (-$37.63))  =  1,000  x  $57.63  =  $57,630 loss
```

Their \$3,000 stake is obliterated in the first \$3-a-barrel of decline. Everything after that is *uncovered* loss. The settlement at a *negative* price means the position cannot be marked at zero — it is marked deeply below zero — so the customer is left owing the difference between what their margin covered and the negative settlement value. The same physics that turned a barrel into a liability turned a small saver's "investment" into a **debt**. The takeaway is blunt: a leveraged long on a physically-settled front-month contract has no natural floor at zero, and marketing it to people who think they are simply "betting oil goes up" hands them a risk they were never shown.

### The aftermath: lawsuits, reforms, and a faster recovery than the price suggested

The negative print did not vanish quietly. Regulators on both sides of the Atlantic opened reviews of the trading around the settlement, asking whether the final-window mechanics had been orderly and whether any participant had acted abusively. The exchange defended the settlement as a faithful reflection of supply, demand, and full storage; critics argued that letting a thin, deadline-driven final window set the price for a contract the whole world references was a structural weakness. Out of the episode came a wave of practical changes: ETFs and structured products that had lived in the front month spread their holdings across the curve or moved to later months; some retail venues restricted or redesigned front-month commodity products; and risk desks everywhere rewrote their playbooks to treat the *delivery month of a physically-settled contract* as a distinct, elevated-risk regime rather than just another trading session.

The cruel irony for the wiped-out longs is that oil recovered quickly. The very next contract — June WTI — never went negative, and by the summer crude was back in the \$40s as demand returned and OPEC+ cut output hard. The investors who had been *right* that oil would recover still lost everything, because they expressed a months-long view through the one instrument with a delivery deadline days away. Being right about the destination is worthless if your vehicle crashes before it leaves the parking lot. The negative print was a *timing-and-instrument* catastrophe layered on top of a correct *direction* call, which is exactly why it is such a clean teaching case: it isolates the cost of ignoring the physical-settlement mechanics from the question of whether your market view was any good.

## Common misconceptions

**"Negative prices mean oil became worthless / the world ran out of oil demand."** No. Crude oil at the wellhead and in distant-month contracts stayed clearly positive throughout — December 2020 WTI was around \$36 the same week the prompt printed −\$37.63. What went negative was *one specific contract*: a claim to receive a barrel *at Cushing, on a specific day, into tanks that were full*. The negative number priced the **cost of the delivery obligation**, not the value of oil in general.

**"It was a glitch or a market manipulation."** It was neither. The CME had to update its systems to even permit a negative quote, which fueled the "glitch" narrative — but the trades were real, cleared, and economically rational given full storage. There were lawsuits and regulatory probes into the trading around the settlement, but the core event was a genuine collision of physical settlement with full tanks, not a software bug.

**"Brent didn't go negative, so WTI must be a worse or more fragile benchmark."** They are *different* benchmarks by design. WTI delivers physically into landlocked Cushing; Brent settles against seaborne North Sea cargoes that can be shipped anywhere. The seaborne escape valve is exactly why Brent held above zero. Neither is "better" — but the episode is a permanent reminder that *where and how a contract settles* is part of its risk, not a footnote. (For more on the two benchmarks, see [WTI vs Brent](/blog/trading/commodities/crude-oil-wti-vs-brent-the-worlds-two-benchmark-barrels).)

**"A long position can only lose 100% — your downside is capped at zero."** This is true for a stock and for a *cash-settled* long, but false for a *physically-settled* long. As the worked examples showed, a physical-delivery long who cannot take delivery can lose *more* than 100%, because the asset can settle below zero. The "downside is capped at zero" rule quietly assumes the thing you own can never become a liability. At expiry, in a storage crunch, that assumption breaks.

**"ETFs like USO let you safely own oil for the long run."** A front-month commodity ETF is a *roll machine*, not a barrel in a vault. In contango it bleeds the roll cost continuously, so it can fall even when spot oil rises. USO recovered far less than crude itself after April 2020 and later changed its strategy and did a reverse split. These products are tools for short-horizon exposure, not buy-and-hold ownership of the commodity.

## How it shows up in real markets

The negative WTI print was a once-in-history event in its magnitude, but the *forces* behind it recur constantly, and reading them is a durable skill.

**Watch the prompt-to-next-month spread into every expiry.** The width and direction of the spread between the expiring contract and the next month is a real-time gauge of physical stress. A spread that blows out into steep contango in the days before expiry is the market telling you storage is tight and the prompt is vulnerable. In April 2020 that spread was a flashing red light for anyone watching. You do not need to predict −\$37.63 to act on it; you just need to be *out of the delivery month* before the spread goes vertical.

**Track delivery-hub inventories against capacity.** For WTI, the weekly EIA Cushing storage number versus the ~76-million-barrel ceiling is the single most informative fundamental for prompt-month risk. When inventories approach the tank tops, the prompt contract is structurally fragile regardless of the headline oil price. The same logic applies to any physically-settled commodity with a finite delivery point — natural gas hubs, metals warehouses, grain terminals. (The series covers an LME analogue, where a different physical-settlement squeeze broke the market, in [the 2022 nickel squeeze](/blog/trading/commodities/aluminum-nickel-and-the-2022-nickel-squeeze-when-the-market-broke).)

**Never confuse "the price of oil" with "the price of the front contract."** Headlines say "oil fell to −\$37." Oil did not. *One contract* did. The distinction matters because your exposure lives in a *specific* contract with *specific* settlement terms, and the front month carries delivery risk that the rest of the curve does not. A trader who internalizes this reads "WTI at −\$37.63" as "the May delivery obligation cleared at −\$37.63," which is a precise, actionable statement, not a doomsday headline.

**Negative prices are rare for oil but not unheard of for other commodities.** Crude going negative was a first, but the underlying phenomenon — a commodity that is *costly to dispose of* trading below zero — appears elsewhere. Regional **natural gas** prices in pipeline-constrained basins (such as the Permian's Waha hub) have repeatedly gone negative when too much gas is produced in a place with nowhere to send it: producers pay to offload gas rather than shut wells or flare. Wholesale **electricity** prices go negative on windy, sunny, low-demand nights, because power must be consumed the instant it is made and some generators find it cheaper to pay to keep running than to switch off. The common thread is always the same: when a commodity *cannot be stored or moved* and *cannot be costlessly stopped*, its local price can fall below zero. Oil at Cushing in April 2020 was the most dramatic member of a family, not a freak of nature. Recognizing the family is the skill — any physically-settled commodity at a capacity-constrained delivery point with inflexible supply is a candidate.

**Respect the difference between physical and cash settlement before you ever hold into expiry.** This is the universal rule. Before holding *any* futures contract toward its last trading day, you must know: is it cash-settled or physically-settled? Where does it deliver? Is that delivery point capacity-constrained? If you cannot take delivery and the contract is physical, you have a hard deadline to be out — and "I'll sell near the close" is not a plan, because the close is exactly when the bid can vanish.

## The playbook: paper is not physical, and at expiry that is everything

Pull it together into rules you can actually use.

![The lesson stack: the longs who treated the contract as pure paper were destroyed; those who respected physical settlement walked away whole](/imgs/blogs/april-2020-negative-wti-the-day-oil-traded-below-zero-7.png)

The final figure stacks the two paths side by side — the trader who treated a delivery contract as a paper bet and the one who respected physical settlement — and every rule below is one row of that contrast:

1. **Know your settlement type before you hold near expiry.** Physically-settled contracts (WTI crude, many grains and metals) carry delivery obligations; cash-settled ones do not. If you cannot or will not take delivery, a physically-settled front-month contract has a hard exit deadline, and missing it can cost more than your entire stake.

2. **Roll early, on a schedule, not at the bell.** The roll is the mechanism that keeps paper exposure clear of the physical barrel. Pay it in calm conditions, days or weeks before expiry. The worked example showed the gap: \$4,000 to roll early versus \$62,630 to be trapped. Never plan to escape "near the close" — the close is when the door slams.

3. **Read the curve as a storage gauge.** Super-contango is not a meaningless squiggle; it is the market pricing the scarcity of a place to put the commodity. A front month collapsing relative to later months is a direct warning that the delivery point is filling. The steeper the contango, the more dangerous it is to hold the prompt.

4. **Distinguish the commodity from the instrument.** "Oil" did not go to −\$37.63; the *May WTI contract at Cushing* did. Front-month ETFs are roll machines that bleed in contango and can fall while spot rises. Leveraged retail products on the front month can leave you owing more than you invested. Match the instrument to your horizon, and never assume "buying oil" means owning a barrel.

5. **Remember that a barrel you cannot store is a liability.** This is the deepest lesson and the spine of the series. A commodity's value depends on being able to *deliver it* or *hold it*. Strip away storage and delivery and the same barrel that is worth \$50 to a refiner is worth *less than zero* to a trapped long. Physical settlement is the moment the financial wrapper meets the physical reality, and the reality wins.

There is a final way to frame why this episode sits at the heart of the whole series. Across these posts, the recurring claim is that a commodity is a *physical thing forced through a financial contract*, and that the forward curve, the cost of storage, and the convenience yield are the gears that connect the two. Most of the time those gears mesh quietly and you trade the paper without ever thinking about the barrel. April 2020 is the day the gears jammed in full view: the curve went to super-contango because storage was the binding constraint, the cost of storage went effectively to infinity because there was no tank, and the convenience yield — the premium for having the physical thing *right now* — inverted into a *penalty* for being handed a barrel you could not store. Every concept this series teaches showed up at once, with a price tag of minus thirty-seven dollars and sixty-three cents attached. If you understand why oil went negative, you understand the machinery of every commodity market in this series, because the negative print is what that machinery looks like when it is pushed to its physical limit.

April 2020 was, in the end, the most expensive teaching aid in the history of commodities. It taught a generation of traders — and tens of thousands of retail savers who learned it the hardest possible way — that a commodity future is not a number on a screen but a *claim on a real thing at a real place on a real day*. Most of the time the financial wrapper is so smooth you can forget that. For one afternoon at Cushing, the wrapper tore open, and the barrel underneath showed its true face: not an asset, but an obligation, priced at minus thirty-seven dollars and sixty-three cents. The difference between paper and physical is not academic. At expiry, it is everything.

## Further reading & cross-links

**Within this series — the mechanics behind the crash:**
- [Cash-and-carry and storage arbitrage: locking in the curve](/blog/trading/commodities/cash-and-carry-and-storage-arbitrage-locking-in-the-curve) — the trade that caps contango at the cost of storage, and what happens when the tanks fill.
- [Spot vs futures: the two prices of the same barrel](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel) — how the cash price and the contract price relate, and why they can diverge violently at expiry.
- [Roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed) — why a front-month fund like USO loses to the roll in contango, even when spot recovers.
- [Contango vs backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — the upward and downward curve shapes and what each says about storage and scarcity.
- [Crude oil: WTI vs Brent, the world's two benchmark barrels](/blog/trading/commodities/crude-oil-wti-vs-brent-the-worlds-two-benchmark-barrels) — why landlocked WTI and seaborne Brent behave differently at the extremes.
- [Aluminum, nickel, and the 2022 nickel squeeze](/blog/trading/commodities/aluminum-nickel-and-the-2022-nickel-squeeze-when-the-market-broke) — another market where physical settlement and a squeeze broke the contract.

**Out to the rest of the blog:**
- [Energy: oil and gas, the inflation engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) — where crude sits in a cross-asset portfolio and how energy shocks transmit to everything else.
- [Gold futures: COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — the same paper-versus-physical tension in the monetary metal, and why gold's storage economics differ from oil's.
- [Commodities as macro signals: oil, copper, gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) — reading oil's price and curve as a real-time signal about the global economy.
