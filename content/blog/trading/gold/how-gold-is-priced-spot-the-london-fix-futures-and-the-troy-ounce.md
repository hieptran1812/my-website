---
title: "How Gold Is Priced: Spot, the London Fix, Futures, and the Troy Ounce"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "There is no single gold price. A beginner's guide to the OTC spot quote, the twice-daily London auction fix, COMEX futures, the troy ounce, and what paper gold versus physical gold really means."
tags: ["gold", "gold-price", "spot-price", "london-fix", "lbma", "comex", "futures", "troy-ounce", "contango", "paper-gold", "physical-gold", "precious-metals"]
category: "trading"
subcategory: "Gold"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — There is no single "gold price"; there is an OTC spot quote, a twice-daily London auction "fix," and COMEX futures, and arbitrage keeps the three glued together within a few dollars.
>
> - Gold is quoted per **troy ounce** (31.103 grams), almost always in **US dollars**, because that is the unit the global wholesale market settled on centuries ago.
> - **Spot** is the headline wholesale quote for immediate delivery; the **London fix** is a transparent auction price banks use to settle contracts; **futures** add the cost of carrying gold to a future date (the *basis*).
> - "Paper gold versus physical gold" is mostly a debate about **counterparty risk** (whether you own an allocated bar or just a claim), not about price.
> - The number to remember: a 1 oz coin at a 5% retail premium costs you about a **6% round-trip** to buy and sell back — that spread is the price of actually *holding* the metal, on top of the wholesale price you see on the screen.

Two people check "the gold price" on the same morning, in the same city, and read off different numbers. One is a treasury manager at a London bank, watching a screen that says \$2,401.50. The second is a futures trader in Chicago, whose December COMEX contract is printing \$2,438. The third walks into a coin shop and is quoted \$2,520 for a single one-ounce coin. Each of them is sure they know "the price of gold." Each of them is right.

This is the part of gold that quietly confuses almost everyone, including people who have bought it for years. They imagine a single number, the way a stock has a single last-traded price. Gold doesn't work like that. What exists instead is a small constellation of related prices — spot, the fix, futures, and a stack of retail premiums on top — held in formation by traders who pounce on any gap between them. Understanding that constellation is the whole game. Once you see how the quotes relate, the "paper gold is fake" arguments, the "the price is rigged" arguments, and the "why is the coin shop charging me so much?" complaints all resolve into plain mechanics.

This post is the plumbing tour. We will define the troy ounce, walk through the spot market, sit inside the London auction, decode a futures quote, explain why gold rises when the dollar falls, and finally separate the wholesale monetary price from what you actually pay to hold insurance in your hand. The spine of this whole series runs underneath: **the price you see quoted is the world's wholesale monetary price for gold; what you pay to truly hold the metal as insurance is always higher.**

![Three gold prices linked by arbitrage between spot, the London fix, and COMEX futures](/imgs/blogs/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce-1.png)

## Foundations: How a gold quote is actually built

Before we can talk about *which* gold price, we have to agree on what a single quote even means. A gold price is three things welded together: a **quantity unit**, a **currency**, and a **delivery promise**. Get any one of them wrong and you are comparing apples to gold bars.

### The troy ounce — the unit that isn't your kitchen ounce

Gold is weighed in **troy ounces**, not the ordinary "avoirdupois" ounce you use for groceries. This trips up everyone at least once. A troy ounce is heavier:

- 1 troy ounce = **31.1035 grams** (call it 31.103 g).
- 1 avoirdupois ounce (the supermarket ounce) = 28.35 g.

So a troy ounce is about 10% heavier than the ounce on a food label. The troy system is medieval — it traces to the French town of Troyes, a major trading fair — and it survived into the modern bullion market simply because that is how the trade always did it. When you see a gold price quoted "per ounce," it is *always* the troy ounce. Silver, platinum, and palladium use it too.

From the troy ounce, every other unit you will meet is a fixed multiple:

- **1 gram** = 0.03215 troy oz (because 1 ÷ 31.103 = 0.03215). Asian retail markets, and increasingly online dealers, quote per gram.
- **1 kilobar** = 1,000 g = **32.151 troy oz**. The kilobar is the workhorse of the Asian wholesale market; a "kilo of gold" is a real, tradeable unit.
- **1 lượng** (also written *cây*, or *tael* in English) = **37.5 grams = 1.20565 troy oz**. This is the Vietnamese and broader East Asian retail unit. Vietnamese gold is priced in *millions of VND per lượng*, not dollars per ounce, which is why a Vietnamese price looks nothing like the world price until you do the conversion. We will do exactly that conversion below, because it is the bridge between this post and the Vietnam track of this series.

The good news: because all of these are *fixed* ratios, converting between them is pure arithmetic. There is no market judgment involved — 37.5 grams is always 37.5 grams. The market judgment is entirely in the *price per unit*. The conversion grid below lays out the four units side by side, with each one's weight, its size in troy ounces, and its dollar value at a \$2,400/oz spot price, so you can see how cleanly any quote translates into any other.

![Units of gold conversion grid for troy ounce gram luong and kilobar at a fixed spot price](/imgs/blogs/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce-5.png)

One subtlety hides inside "purity." A troy ounce of *pure* gold is the unit the price refers to — what the trade calls **fine** gold, meaning the actual gold content, stated as a fineness like 99.99% (often written 9999, "four nines"). A coin or piece of jewelry that weighs one ounce *gross* but is only 22-karat (about 91.7% gold) contains less than one ounce of fine gold, so its bullion value is lower than the headline ounce price. When you see "gold price per ounce," it always means per ounce of *fine* gold; the karat or fineness of a physical item tells you how much fine gold is actually inside it. This is why a 24-karat (≈99.9% pure) one-ounce coin is worth more in metal than a 22-karat one-ounce coin of the same gross weight — the second one is part copper. Vietnam's market draws this line sharply between *9999* bars and lower-purity rings and jewelry, a distinction the practical Vietnam-track post handles in detail.

#### Worked example: converting one quote into every unit

Suppose the world spot price is **\$2,388 per troy ounce** (the 2024 annual average). Watch how cleanly it propagates:

- **Per gram:** \$2,388 ÷ 31.103 = **\$76.78 per gram**.
- **Per lượng (37.5 g):** \$76.78 × 37.5 = **\$2,879 per lượng** — or equivalently \$2,388 × 1.20565 = \$2,879. Same answer, two routes.
- **Into Vietnamese dong:** at an exchange rate of **25,450 VND per USD** (the 2024 year-end rate), \$2,879 × 25,450 ≈ **73.3 million VND per lượng** as the *world-equivalent* price.

That last number matters: in mid-2024 the actual SJC bar in Vietnam was quoted around **92 million VND per lượng** — roughly 19 million dong, about 25%, *above* the world-equivalent we just computed. That gap is the famous Vietnamese gold premium, and it has its own dedicated post in this series. For now, the point is just the mechanics: every gold price in the world is the same metal seen through a different unit and a different currency. *The arithmetic never lies; the premium on top is where all the interesting economics live.*

### The currency — why gold is quoted in dollars

Almost every wholesale gold quote you will ever see is in **US dollars**. This is a historical accident that hardened into a convention. After the Second World War, the dollar was pegged to gold (\$35/oz under Bretton Woods), and London — the center of the bullion trade — settled in dollars. When the gold-dollar peg died in 1971, the habit stayed. The dollar is the world's reserve currency, the deepest and most liquid money on earth, so quoting the most liquid metal in the most liquid currency is natural.

This convention has a profound consequence we will return to: when you watch "the gold price," half of what you are watching is *the dollar*. A rising gold price can mean the market wants more gold, or it can simply mean the dollar is worth less. Holding the metal fixed and changing the measuring stick changes the number. We will make this concrete with a figure later.

It is worth pausing on how *sticky* this convention is. Even as other countries build gold markets — Shanghai runs a large physical exchange quoting in yuan per gram, and India, Turkey, and Vietnam all have deep domestic gold cultures — the *reference* price that those local markets ultimately key off remains the dollar-denominated London/COMEX complex. A Shanghai premium or a Vietnamese premium is, definitionally, a premium *over the dollar world price* converted at the local exchange rate. So the dollar quote is not just one price among equals; it is the *anchor* the others are measured against. The day a non-dollar gold benchmark genuinely set the world price — rather than tracking and adding a local premium to the dollar one — would be a monetary event of the first order, a signal that the dollar's grip on the plumbing of global trade had loosened. That has not happened. For now, the dollar is the measuring stick, the local markets add their wedges on top, and "the gold price" remains, at its core, a number expressed in the world's reserve currency. This is precisely why the petrodollar and dollar-dominance story (`/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy`) sits so close to the gold story: they are two views of the same monetary hierarchy.

### The delivery promise — spot, forward, futures

The third ingredient is *when* you get the gold. This is what separates the three prices in our cover figure:

- **Spot** means "buy now, settle in two business days" (T+2). The spot price is the wholesale price for gold you can take possession of almost immediately.
- A **forward** or **futures** price means "agree the price now, but settle on a future date." Because the seller has to hold (or finance) the gold until then, a futures price is normally a little *higher* than spot — it includes the cost of carrying the metal across time.
- The **fix** is not a different delivery promise at all; it is a different *price-discovery mechanism* for essentially spot/near-spot gold — a transparent auction that produces one published number twice a day.

Hold those three ingredients — unit, currency, delivery — in your head. Every confusing thing about gold pricing is one of them changing while you weren't looking.

## The spot market: the wholesale benchmark

The **spot price** is the closest thing gold has to "the" price. It is the over-the-counter (OTC) wholesale quote for gold delivered in two days, traded between banks, refiners, miners, and large funds, mostly through London.

The key word is **over-the-counter**. Unlike a stock, which trades on a central exchange where every order hits one public order book, gold spot trades *bilaterally* — bank to bank, dealer to client, over phones and electronic dealing systems. There is no single screen that shows "the" spot price the way there is for Apple stock. Instead, dozens of market-makers quote a **bid** (the price they will buy at) and an **ask** (the price they will sell at), and they are all within a hair of each other because they are all watching each other. The "spot price" you see on a financial website is a representative midpoint of these quotes.

This OTC structure is why London dominates. The **London Bullion Market Association (LBMA)** sets the standards — most importantly the **Good Delivery** standard, which defines what a "real" wholesale bar is: a 400-troy-ounce (≈12.4 kg) bar of at least 99.5% purity, from an accredited refiner. When a London bank says it owns "loco London" gold, it means gold sitting in an LBMA vault in London, in Good Delivery form. That is the wholesale unit. The kilobar and the coin are downstream products; the 400-oz bar is the monetary base layer.

A few features of the spot market follow from its OTC nature and are worth pinning down, because they explain why "the" spot price is fuzzier than a stock price:

- **It trades nearly 24 hours.** The gold "day" rolls around the globe — Asian trading (Shanghai, Hong Kong, Singapore, Tokyo) hands off to London, which hands off to New York, which hands back to Asia. There is no single closing bell. The price you see at 3 a.m. your time is a *live* price from whichever center is most active, not a stale last-trade. This is part of why gold is such a useful crisis barometer: when news breaks at any hour, *some* market is open to react.
- **The wholesale bid/ask spread is tiny.** Because dozens of banks make markets in the same fungible metal and watch each other constantly, the wholesale bid/ask is often just a few cents per ounce on 400-oz bars. The metal is about as commoditized as a commodity gets — one Good Delivery ounce is interchangeable with any other — so competition crushes the wholesale spread. The fat spreads, as we will see, appear only downstream, at the retail end.
- **The "price" is a consensus, not a print.** Because there is no central order book, the spot price quoted on a website is a *representative* level synthesized from many dealers' quotes. Two data vendors can show prices a few cents apart at the same instant and both be honest — they are sampling a distributed market, not reading one tape. This is a feature, not a flaw, but it is why pinning "the exact spot price to the cent at this exact second" is a slightly meaningless request.

The deepest consequence of all this is that London's role is *infrastructural*, not merely historical. The LBMA standard defines what counts as monetary gold; the London vaults hold the wholesale stock; the loco-London unallocated balance is the unit traders pass around all day. When we later watch the London-to-COMEX arbitrage break in March 2020, what breaks is precisely the link between this London infrastructure and the New York exchange — and the whole "single gold price" wobbles because of it.

### Allocated versus unallocated — the most important distinction in the whole post

Here is where "paper gold" first appears, and it is not where most people think. In the London market, most gold is held in **unallocated accounts**.

- An **allocated account** holds *specific physical bars* assigned to you, with serial numbers, sitting in a vault. You own the metal outright; it is not the bank's asset and would not be touched if the bank failed. You pay storage for the privilege.
- An **unallocated account** is just a *claim* on the bank for a certain weight of gold. You are an unsecured creditor of the bank, denominated in ounces instead of dollars. The bank owes you gold but doesn't earmark specific bars; it nets your claim against everyone else's. There is no storage fee, because there is, in a sense, nothing specific being stored *for you*.

Unallocated is how the wholesale market actually runs day to day, because it is frictionless — you can trade ounces back and forth without physically moving heavy, expensive-to-insure bars. But it carries **counterparty risk**: if the bank goes under, an unallocated holder stands in line with other creditors, while an allocated holder simply collects their bars.

This is the real meaning of "paper gold versus physical gold." It is not that unallocated gold is *fake*. It is that unallocated gold is a *promise from a counterparty*, and physical/allocated gold is *the thing itself with no promise attached*. The price is the same. The risk is not. Keep this distinction in your back pocket; it is the hinge of the entire "paper gold" debate, and it gets its own deep treatment in the *physical gold* and *gold ETFs* posts later in this series.

It helps to see *why* the market runs mostly on unallocated. Physical gold is heavy, valuable, and expensive to insure and transport — a single 400-oz bar is worth nearly a million dollars and weighs as much as a large bag of cement. If every trade required physically moving bars, the market would grind to a crawl. Unallocated balances let banks net trades against each other the way a bank settles checks: only the *net* flow has to be backed by metal, so the system can support far more *trading* than there is *physical metal changing hands*. That efficiency is exactly what creates the "more claims than bars" property that fuels the paper-gold debate. There is genuinely more *unallocated gold owed* in the London system than there is *physical gold* sitting under it at any instant — not because anyone is lying, but because most claims are never simultaneously converted to metal, just as a bank does not keep enough cash in the vault for every depositor to withdraw at once. The system works on the assumption that not everyone demands delivery on the same day. In a true panic, that assumption is what gets tested — which is precisely why, in a crisis, *allocated* gold (your specific bars, no counterparty) behaves differently from *unallocated* gold (a claim that could face a queue). For the everyday investor the takeaway is blunt: if your reason for owning gold is insurance against a system failure, an unallocated claim on a bank reintroduces the very counterparty risk you were trying to escape.

## The London fix: one number, twice a day

If spot is a fuzzy cloud of bilateral quotes, the **fix** is a single, sharp, published number — the price the market agrees on at a precise moment, through a transparent auction.

The modern version is the **LBMA Gold Price**, run as an electronic auction twice each London day: at **10:30** and **15:00** GMT. (The afternoon auction matters extra because it lands while New York is open, so it sets a reference both London and the US can settle against.) It replaced the century-old "London Gold Fixing," where representatives of a handful of bullion banks literally met — by phone in later years, around a table in the old days — and adjusted a price until buy and sell orders balanced. The mechanism is the same; only the venue (now a regulated electronic platform, overseen since the manipulation scandals of the 2010s) has changed.

Why does the world need a fix when spot already exists? Because thousands of contracts need a single, *referenceable*, *unarguable* price. A mining company selling its monthly output, a jeweler buying in bulk, a fund tracking a benchmark, a central bank valuing its reserves — none of them want to haggle a bilateral price. They want to say "we'll transact at the afternoon fix," knowing that price is public, auditable, and the same for everyone who used it. The fix is the gold market's official photograph of the price, taken at a scheduled time, that everyone can point to.

The auction runs through a set of accredited **direct participants** — large banks and trading firms approved by the administrator — but the orders they bring are largely their *clients'* orders, aggregated. So a refiner who tells its bank "sell my 30,000 oz at the afternoon fix" becomes part of the sell volume the bank submits into the auction. The fix is therefore not a price *imposed* on the market; it is a price *discovered from* the actual buy and sell intentions of everyone who chose to transact at that moment. This is the critical mental correction to the "rigged" narrative: in a properly run auction, the published price is whatever level makes the aggregated client orders balance, and any participant trying to push it away from that level is, by construction, taking the other side and absorbing the imbalance themselves — expensive, and now closely surveilled.

Why *two* fixes a day, and why does the afternoon one carry extra weight? The 10:30 fix gives Asian and European business a morning reference; the 15:00 fix lands while New York is trading, so it is the price both London and the US can settle against simultaneously. Many large institutional benchmarks, valuation marks, and physically-settled products reference the *afternoon* (PM) fix specifically, which is why it is the one most often quoted in financial filings. A central bank marking its reserves to market, an ETF striking its daily net asset value, a structured product paying off against "the gold price on the settlement date" — these typically mean the PM fix, because it is the one public number that the largest, most overlapping pool of global participants agreed on that day.

### How the auction actually finds a price

The fix figure below traces the mechanism. The auction is a clever little machine for finding the price at which supply equals demand:

![The London fix auction iterating a trial price until buy and sell orders balance](/imgs/blogs/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce-2.png)

1. The auction **chair posts an opening price**, near the prevailing spot level.
2. Participating banks (acting for themselves and their clients) **submit the volume** they want to buy or sell at that price.
3. The system checks the **net imbalance** — total buys minus total sells. If the imbalance is within a small tolerance (a few tens of thousands of ounces), the market has cleared and that price becomes the fix.
4. If there is a real imbalance — say far more buyers than sellers — the chair **moves the price**: up if buyers dominate (to coax out more sellers and scare off some buyers), down if sellers dominate.
5. Participants resubmit volumes at the new price. The loop repeats — usually only a handful of rounds — until **buys ≈ sells**.
6. The balanced price is **published as the LBMA Gold Price**, and every party that wanted to transact at the fix does so at that single number.

This is just an auctioneer walking the price to where the room clears. It is the same logic as a fish market auction, except the "fish" is a globally fungible metal and the bidders are some of the largest banks on earth.

#### Worked example: why the fix can differ from spot

Imagine the 15:00 auction opens at \$2,400, in line with spot. At that price, buyers want 250,000 oz and sellers offer 200,000 oz — a **buy imbalance of 50,000 oz**, well outside tolerance. There simply aren't enough sellers at \$2,400.

The chair raises the indicative price to **\$2,403**. At the higher price, some buyers back off (their orders were limit orders capped below \$2,403) and a few more sellers appear (the higher price tempts them). Now buyers want 215,000 oz and sellers offer 212,000 oz — within tolerance. The auction clears at **\$2,403**, and *that* is the published fix.

Notice what happened: the fix printed **\$3 above** where spot was when the auction opened. Nobody rigged anything — a genuine excess of buyers pushed the clearing price up. *The fix can sit a few dollars off the simultaneous spot quote precisely because it is a real auction that reflects the order imbalance at that exact minute, not a passive snapshot.* When you read that "gold fixed at \$2,403 this afternoon," you are reading the output of this little balancing machine.

A historical footnote that matters: the old fixing was rocked by **manipulation findings** in the mid-2010s — banks were fined for nudging the fix to benefit their own positions, especially around the moment of publication. The move to a regulated, electronic, auditable auction (administered by ICE Benchmark Administration from 2015) was the direct response. This is the kernel of truth inside the popular "the gold price is rigged" claim — we will give it the careful, numerate treatment it deserves in the misconceptions section.

## COMEX futures and the basis

The third price on our cover is the loudest one — the one financial TV usually quotes. **COMEX** (part of CME Group in the US) runs the dominant gold **futures** market. A futures contract is a standardized, exchange-traded agreement to deliver a fixed amount of gold (the standard contract is **100 troy ounces**) at a specified future month, at a price agreed today.

Futures exist for two reasons. First, **hedging**: a miner who will produce gold in six months can lock in today's price by selling futures, removing the risk that the price falls before they sell. A jeweler who will need gold can lock in a buy price the same way. Second, **speculation and leverage**: a trader can control 100 oz of gold (over \$240,000 worth) by posting only a few percent in *margin*, so futures let you make a large bet on gold's direction with a small amount of cash. That leverage is why futures volume dwarfs the physical market, and it is why the futures price often *leads* spot intraday — the fastest money trades there.

That leverage cuts both ways, and it is worth understanding the mechanics because they feed straight back into the paper-vs-physical debate. **Margin** is a performance bond, not a down payment: the exchange might require, say, \$11,000 to control a \$240,000 contract, roughly 22-to-1 leverage. Each day the position is **marked to market** — if gold falls and your loss eats into the margin, you get a **margin call** and must post more cash or be liquidated. So a futures holder is not the relaxed owner of a bar in a vault; they are running a leveraged, daily-settled bet that can be force-closed by a price move that a physical holder would simply sit through. This is one reason "paper gold" and "physical gold" behave differently in stress: leveraged futures longs can be *forced* to sell into a falling market (amplifying the drop), while a physical holder feels nothing but a lower mark.

Crucially, **most futures never go to delivery.** The standard contract *can* be settled by delivering 100 oz of approved gold into a COMEX-approved vault, and that delivery option is what anchors the future to the physical metal. But the overwhelming majority of contracts are *closed out* before expiry — a long buys back, a short sells back — or **rolled** forward into a later month. Only a small fraction stand for delivery. This is healthy and normal: the delivery option does its job by *existing* (it makes the cash-and-carry arbitrage enforceable), not by being exercised en masse. The "what if everyone stood for delivery?" thought experiment is exactly the run scenario from the unallocated discussion — there are more open contracts than registered deliverable ounces, so universal delivery is impossible, but it is a theoretical tail, not the day-to-day reality. The full delivery and roll mechanics get their own post (`gold-futures-comex-contango-backwardation-and-paper-vs-physical`).

### The basis: futures price minus spot price

Here is the concept that demystifies the whole futures quote. A futures price is **not a forecast** of where gold will be. It is spot plus the **cost of carry** — the cost of holding gold from today until the delivery date.

What does it cost to carry gold? Mostly the **interest you forgo**. If you buy a bar of gold today instead of leaving the money in a Treasury bill earning, say, 4.5%, you give up that interest while you hold the metal. (There are small storage and insurance costs too, but for gold the dominant term is the interest rate, because gold pays no yield of its own.) So a seller delivering gold in the future must be compensated for that forgone interest, and the futures price embeds it:

```
Futures price = Spot price x (1 + r)^t
```

where `r` is the relevant interest rate and `t` is the fraction of a year until delivery. The **difference** between the futures price and spot is called the **basis**. When futures trade *above* spot — the normal state for gold — the market is in **contango**. (The rare opposite, futures below spot, is **backwardation**, and for gold it usually signals a physical squeeze — people want metal *now* badly enough to pay a premium for immediate delivery.)

The figure below draws the contango ladder: spot at the base, and each later contract stepping up by the accumulating carry cost.

![Gold futures priced as spot plus cost of carry across maturities, the contango ladder](/imgs/blogs/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce-4.png)

#### Worked example: computing the fair futures price

Start with **spot = \$2,400/oz**, an interest rate of **r = 4.5%**, and a **6-month** contract (t = 0.5).

Using the simple-interest approximation (close enough over six months):

```
Fair futures = 2,400 x (1 + 0.045 x 0.5)
             = 2,400 x 1.0225
             = 2,454
```

So the 6-month future should trade around **\$2,454**, a **basis of +\$54**. Stretch it to a full year and the basis roughly doubles to about **+\$108** (futures ≈ \$2,508); shorten it to three months and it halves to about **+\$27** (futures ≈ \$2,427). Those are exactly the rungs on the contango ladder in the figure.

The crucial takeaway: that \$54 is **carry, not a prediction**. The futures market is *not* saying "we think gold will be \$2,454 in six months." It is saying "if you want gold delivered in six months, you must pay for the six months of interest someone gives up holding it for you." *A reader who treats the futures curve as a forecast will misread it every single time; it is a cost structure, not a crystal ball.*

There is one real-world refinement worth naming, because it explains the rare moments when gold flips into backwardation. Gold can be *lent* — central banks and large holders lease metal to the market for a small fee, the **gold lease rate**. The true carry is therefore the interest rate *minus* the lease rate: someone holding gold can earn a little by lending it out, which offsets the interest they forgo. Normally the lease rate is small and the interest rate dominates, so the curve sits in contango. But in a physical squeeze — when borrowers desperately need actual metal *now* — the lease rate can spike above the interest rate, the net carry goes negative, and the front of the futures curve dips *below* spot: **backwardation**. So when you see gold in backwardation, it is almost always a flashing signal that physical demand for immediate metal has overwhelmed the normal cost-of-carry logic — the market is paying you to wait rather than charging you, because everyone wants the bar today. That is a rare and informative state, the opposite of the comfortable contango ladder in the figure.

### Arbitrage: the glue between all three prices

Now we can close the loop on the cover figure. Why do spot, the fix, and futures stay within a few dollars of one another, when they trade in different venues, on different mechanisms, in different time zones?

**Arbitrage.** Suppose 6-month futures traded at \$2,500 while spot was \$2,400 and the fair carry was only \$54. A trader could *buy spot gold at \$2,400, sell the future at \$2,500, store the gold, deliver it in six months, and pocket \$100 minus the \$54 carry = \$46 of risk-free profit per ounce.* That trade is called **cash-and-carry arbitrage**, and the existence of even a small free profit attracts so much of it that the prices snap back into line almost instantly. The same logic links spot to the fix, and London to New York. The specific London-to-COMEX version even has a name — the **Exchange for Physical (EFP)** — which lets a trader swap a futures position for physical London gold (and back), and it is the literal mechanical bridge between the two markets.

#### Worked example: the cash-and-carry arbitrage that closes the gap

Make the trade concrete on one 100-oz COMEX contract. Spot is **\$2,400**, the fair 6-month future (at r = 4.5%) is **\$2,454**, but the market is quoting the future at **\$2,500** — \$46 too rich. The arbitrageur:

```
Buy 100 oz spot now:           100 x 2,400 = 240,000  (cash out)
Sell one 6-month future:       locks a sale at 100 x 2,500 = 250,000
Finance + store the gold 6 mo: cost ~ 100 x 54        =   5,400
At delivery: hand over the 100 oz, collect           = 250,000
```

Profit = 250,000 − 240,000 − 5,400 = **\$4,600 essentially risk-free** on the contract (about \$46/oz), because the sale price was locked the instant the future was sold and the only outflows — the metal and the carry — were both known up front. The moment such a profit exists, every arbitrage desk piles in: they *buy spot* (pushing spot up) and *sell the future* (pushing the future down), and they keep doing it until the gap collapses back to the \$54 fair carry and the free money is gone. *This relentless mechanical force is the reason the three prices in the cover figure are never really independent — any gap is, quite literally, an unfilled order on an arbitrageur's screen, and it does not stay unfilled for long.*

So the three prices are not three independent opinions about gold. They are three faces of one price, kept aligned by an army of traders who treat any gap as money lying on the ground. When that arbitrage *breaks* — when traders physically cannot move gold between London and New York fast enough — the prices can dislocate dramatically. That is exactly what happened in March 2020, and it is our headline case study below.

## Why gold rises when the dollar falls: the numéraire effect

We flagged earlier that watching "the gold price" means half-watching the dollar. Now let's make it precise, because it dissolves one of the most common confusions about gold.

A **numéraire** is just the unit you measure value in — the measuring stick. Because gold is quoted in dollars, the dollar is gold's numéraire. And here is the subtlety: when the *measuring stick itself* shrinks, the *number* you read off gets bigger, even if the thing being measured hasn't changed at all.

Think of it the other way around. The "price of gold in dollars" is mathematically the same statement as the "price of dollars in gold," flipped over. If a dollar buys less gold today than yesterday, then gold "costs more dollars" — those are two descriptions of one event. So a large share of gold's dollar price moves are really *dollar* moves wearing a gold costume. This is why gold and the dollar index (DXY) tend to move in *opposite* directions: a weaker dollar mechanically lifts the dollar price of everything priced in dollars, gold included. The dedicated post `gold-and-the-dollar-the-inverse-relationship-and-when-it-breaks` digs into exactly when that inverse link holds and when it snaps.

The cleanest way to *see* the numéraire effect is to price the same metal in several currencies at once. If gold rose in *every* currency, the move can't be "about" any single currency — it must be about gold's monetary value rising relative to *all* paper. If it rose only in dollars, it was a dollar story.

![Gold indexed to 100 in 2015 priced in four currencies, all rising](/imgs/blogs/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce-6.png)

The figure indexes gold to 100 in 2015 in four currencies. Notice that gold rose in *all* of them — and rose *most* in the yen and the dong, the currencies that lost the most purchasing power over the decade. That is the numéraire effect made visible: the steeper the line, the weaker the currency it is measured in. A Japanese saver and a Vietnamese saver both watched "gold" soar, but a large part of what they watched was their own money depreciating.

#### Worked example: how much "gold price" is really the dollar

Suppose gold rises from \$2,000 to \$2,200 over a year, a **+10%** move on the screen. In the same year the dollar index falls **6%** against a basket of currencies. How much of gold's move was *gold*, and how much was the *measuring stick*?

A rough decomposition: if the dollar lost 6% of its value, then *anything* priced in dollars — oil, copper, imported goods, gold — gets mechanically re-rated upward by roughly that amount, with nothing real changing. So of the +10% gold move, on the order of **6 points** is "the dollar got smaller" and only about **4 points** is "the world wanted gold more." For a European or Japanese holder whose currency *strengthened* against the dollar that year, gold's rise in *their* currency would be smaller than 10% — they captured only the genuine-demand part, because their measuring stick didn't shrink. *The same metal, the same year, produced different returns for different holders purely because they measured it in different money — which is the whole point: a dollar gold quote is a statement about the dollar at least as much as about gold.* This is why serious analysts watch gold priced in a *basket* of currencies, or against a real-asset benchmark, to strip the numéraire noise out and see the underlying monetary signal.

### Real yields: the variable underneath the dollar

Behind the dollar sits an even deeper driver, the one this series treats as the master variable: the **real interest rate** — the interest rate after subtracting inflation. Gold pays no interest. So gold's great rival is any safe asset that *does* pay interest in real terms, like inflation-protected Treasury bonds (TIPS). When real yields are high, holding a barren metal has a steep opportunity cost — why hold gold earning 0% when a TIPS bond pays you 2% above inflation, risk-free? When real yields are low or negative, that opportunity cost vanishes, and gold's lack of yield stops being a handicap. So gold tends to rise when real yields fall and fall when they rise — an inverse relationship.

![Gold price versus the inverted 10-year real yield, two lines mirroring each other](/imgs/blogs/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce-3.png)

The dual-axis figure plots gold against the 10-year TIPS real yield with the yield axis *inverted*, so that "falling real yields" points up. For most of the 2013–2021 stretch the two lines move together strikingly: real yields crater into deeply negative territory in 2020–2021 and gold rips to records. But look at the right-hand side — 2022 onward, real yields climbed back to nearly +2% and yet gold *kept rising to all-time highs*. That decoupling is one of the most important gold stories of the decade: a new buyer (central banks, post-2022) and a debasement/geopolitics bid took over from the pure real-rate trade. The mechanics of *that* are the subject of the drivers track (`real-interest-rates-the-master-variable-behind-the-gold-price` and `central-banks-the-structural-buyer-that-changed-gold-after-2022`). For our purposes here, the lesson is narrower and cleaner: *the dollar gold price you see quoted is anchored, most of the time, by what safe real yields are doing — which is why a "gold price" is never really just about gold.* For the cross-asset framing of why real yields price everything, see `/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything`; for why the dollar sits at the center of all markets, see `/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy`.

## Allocated, unallocated, and where retail premiums come from

We have the wholesale price nailed down. Now we descend from the banks' world to yours — from the 400-oz Good Delivery bar to the one-ounce coin in your hand — and watch the price *grow* on the way down. This is the gap between the screen and the cash register, and it is where "spot is what you pay" goes to die.

The wholesale spot price is for institutions trading 400-oz bars in unallocated form. You are not that. To get from there to a coin you can hold, the metal passes through a chain of intermediaries, each of whom adds a layer of cost. The stack figure lays out the whole markup:

![Premium stack from wholesale spot up to the retail price of a one-ounce coin](/imgs/blogs/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce-7.png)

Reading from the wholesale base upward:

- **Wholesale spot** — the \$2,400/oz headline, available only on 400-oz bars between large players.
- **Dealer bid/ask spread** — the coin shop or online dealer makes its market by buying below and selling above a reference; that spread, often \$10–30/oz on common products, is their cut.
- **Fabrication cost** — turning a big bar into small stamped coins and minted bars costs real money (refining, minting, assaying, packaging), roughly \$30–60/oz depending on the product.
- **Coin/bar premium** — the final markup for scarcity, brand (a Krugerrand or Maple Leaf commands more than a generic round), and the simple fact that small units are in higher retail demand. This is typically **3–8%** over spot for common one-ounce coins, and far more for collectibles.

#### Worked example: the round-trip cost of holding physical

You decide to buy one 1-oz gold coin when spot is **\$2,400**. The dealer charges a **5% premium**, so you pay:

```
Buy price = 2,400 x 1.05 = 2,520
```

Six months later, with spot *unchanged* at \$2,400, you sell that same coin back. Dealers buy back at a discount to spot — say they pay you **1% under spot**:

```
Sell price = 2,400 x 0.99 = 2,376
```

Your round trip: you paid \$2,520 and received \$2,376, a loss of **\$144 on a \$2,520 outlay ≈ 5.7%** — and the gold price never moved. That ~6% round-trip spread is the **cost of holding physical metal**: it is the bid/ask of the retail market, and it is the price of converting an abstract claim into a thing you can bury in the garden. *Physical gold is insurance, and ~6% is roughly the premium on the policy; that is the real meaning of "spot is not what you pay at the coin shop."*

This is also the practical case *for* allocated/unallocated wholesale gold and for ETFs: they let you own gold exposure at close to spot, with spreads measured in basis points rather than percent — at the cost of holding a *claim* rather than a *thing*. The whole "how to own gold" track (`physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk`, `gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal`) is one long meditation on that trade-off: lower cost and convenience versus zero counterparty risk. You cannot have both at once, and the premium stack above is *why*.

## Common misconceptions

Now we can dismantle the myths, each with a number.

**Myth 1: "There is one gold price."** No. There is spot, the morning and afternoon fixes, dozens of futures-month prices, and a stack of retail premiums — and they can legitimately differ by anywhere from a few dollars (spot vs fix) to a few percent (spot vs a coin). They are linked by arbitrage, not identical. The right mental model is *one metal, several quotes*, not *one price*. When someone quotes "the gold price" without saying which, they almost always mean the spot midpoint.

**Myth 2: "Paper gold means there's no real gold behind it."** Mostly false, and confused about what the words mean. "Paper gold" — unallocated accounts, futures, most ETFs — is not counterfeit; it is gold exposure that comes with a *counterparty* (a bank, a clearinghouse, a custodian) rather than a specific bar with your name on it. The distinction that matters is **allocated vs unallocated** (a specific bar you own outright vs a claim on an institution), not paper vs real. The legitimate worry is *counterparty risk in a crisis*, not that the gold is imaginary. Futures *can* stand for delivery, and the major ETFs are backed by audited allocated bars. Where the worry bites is leverage: there are many more *ounces of claims* circulating than physical ounces, so if everyone demanded delivery at once, the claims could not all be met simultaneously — a classic run, not a fraud.

**Myth 3: "The gold price is rigged."** This is the most interesting one because it has a kernel of truth wrapped in an exaggeration. The truth: the *old* London fix was manipulated — banks were fined in the mid-2010s for nudging the fix around the publication moment to favor their own books, and spoofing cases have hit COMEX traders since. The exaggeration: that the *level* of gold — the \$2,400 — is held down or up by a cabal. The arbitrage and numéraire mechanics in this post explain the level perfectly without any conspiracy: gold is high because real yields fell, the dollar weakened, and central banks bought. Manipulation is real but *micro* — it skews the price by cents-to-dollars around specific moments, which is exactly why the fix was rebuilt as a regulated, auditable electronic auction. *Episodic micro-manipulation of the fix: documented. A grand suppression of the price level: not supported by the mechanics or the data.*

**Myth 4: "Spot is what I'll pay."** No — spot is the *wholesale* price for 400-oz bars between banks. As the premium stack showed, a retail one-ounce coin runs roughly 5% over spot to buy and ~1% under spot to sell, a ~6% round trip. If a dealer ever offers you a coin *at* spot, be suspicious; making and distributing small coins costs money, and nobody does it for free.

**Myth 5: "Futures predict where gold is going."** No — the futures curve is spot plus carry, an interest-rate structure, not a forecast. A steep contango means high interest rates, not a bullish view. Reading the curve as a prediction is the single most common futures error.

## How it shows up in real markets

Theory is tidy; markets are not. Two real episodes show the plumbing both working and breaking.

### March 2020: when London and COMEX came apart

In normal times, the London↔COMEX arbitrage we described keeps the two markets within a few dollars. In **March 2020**, as COVID lockdowns hit, it broke spectacularly. Refineries in Switzerland — which turn London's 400-oz bars into the kilobars and 100-oz bars that COMEX accepts for delivery — *shut down*, and passenger flights that normally carry gold in their cargo holds were grounded. Suddenly you physically could not move and re-cast gold from London form into COMEX-deliverable form.

The result: COMEX futures spiked **\$40–70 above** London spot — a basis blowout far larger than any carry cost could justify, because the arbitrage that normally closes it was physically impossible to execute. Traders who were short futures and long London gold (a normally riskless cash-and-carry book) could not deliver and faced huge mark-to-market losses. The dislocation only healed when refineries reopened and logistics restarted, and the EFP — the London↔COMEX swap — normalized over the following weeks.

The lesson is exactly the one this post has been building: *the three prices are one price only as long as arbitrage can physically operate. Remove the ability to move metal between venues, and the "single gold price" splinters into genuinely different prices.* Paper and physical, normally interchangeable, briefly weren't — not because of fraud, but because of grounded planes and shuttered refineries.

### Crisis premiums: when the coin shop runs dry

The second episode is recurring rather than dated: in every fear spike — March 2020 again, but also the 2008 crisis and assorted geopolitical shocks — **retail premiums explode** even as the wholesale spot price wobbles. In March–April 2020, common one-ounce silver and gold coins that normally carried a 4–6% premium briefly traded at **10–25% over spot**, and many dealers simply went "out of stock."

Why? Because the wholesale market and the retail market are connected by that same fragile fabrication-and-distribution chain. When panicked retail buyers all want physical coins at once, mints and dealers cannot stamp and ship coins fast enough; the bottleneck is *manufacturing capacity*, not metal. So the *retail* price of holding a coin detaches upward from spot, even when spot itself is calm or falling. This is the premium stack from earlier, stretched by a demand shock: the top layer (coin premium) balloons because the fabrication layer below it is capacity-constrained. *In a panic, the gap between "the gold price" and "the price to actually hold gold in your hand" widens precisely when you most want to close it — which is the cleanest possible demonstration that spot is the monetary price, and physical possession is a separate, scarcer good.*

Vietnam offers a vivid, structural version of the same phenomenon: the SJC bar trades persistently far above the world-equivalent price because an import ban and a brand monopoly throttle supply. That is a permanent, policy-made premium rather than a temporary panic premium, and it gets the full decomposition in `the-sjc-premium-why-vietnamese-gold-trades-far-above-the-world-price`. The mechanism is the premium stack again, but with the *supply* side clamped by policy instead of the demand side spiking: when the world-equivalent of an SJC lượng was around 73 million VND in mid-2024 (the figure we computed in the opening worked example), the actual SJC quote sat near 92 million — an ~18-million-dong, ~25% wedge that the import restriction wedged open and only narrowed when the central bank stepped in to sell metal directly. Same plumbing, same logic, different valve: in a global panic the *fabrication* layer is the bottleneck; in Vietnam the *import* layer is.

### June 2026: a record print everyone "saw" differently

A final, fresher illustration. By January 2026 spot gold spiked to an all-time high near \$5,400/oz, and the constellation of prices fanned out exactly as this post would predict. The afternoon London fix and spot tracked within a few dollars, as arbitrage demands. COMEX futures sat above both by their carry — a wider basis than usual, because interest rates were still elevated, so the cost of carrying gold to a future month was high. And retail buyers from coin shops in the US to gold shops in Hanoi paid the steepest prices of all, with premiums fattened by a wave of fear-driven demand and, in Vietnam's case, the standing import-driven wedge that pushed the SJC bar past 190 million VND per lượng. Three observers, three different numbers, one metal — and every gap explained by the mechanics in this post, not by anyone's conspiracy or confusion. The record was real; the disagreement about "the price" was an illusion created by not specifying *which* price.

## The takeaway: which price is *your* price?

Step back and the constellation resolves into a single, usable idea. There is no one "gold price"; there is a layered set of prices, each answering a different question:

- **Spot** answers: *what is gold worth, wholesale, right now?* It is the purest read on gold's monetary value — the world's vote, minute by minute, on paper money and real yields. This is the number to watch if you care about gold as a *macro signal*.
- **The fix** answers: *what single, public, auditable price can a contract settle against twice a day?* It is plumbing for institutions; you will rarely transact at it, but it is the reference much of the market is built on.
- **Futures** answer: *what does it cost to get gold delivered later?* — spot plus carry, an interest-rate structure, never a forecast. They are where leverage and price discovery live, and where you trade if you want exposure without metal.
- **The retail premium** answers: *what does it cost to actually hold the insurance in my hand?* — spot plus ~5%, a ~6% round trip. This is the number that matters if you are buying coins, and it is the toll for converting a claim into a thing.

Which one is "your" price depends entirely on what you are doing. A macro trader lives on spot and the futures basis. An institution settles at the fix. A saver in Hanoi or a buyer at a coin shop pays the retail premium and should think hard about that ~6% before treating physical gold as a tradeable position rather than a long-held insurance policy.

And that is the spine of this series, made concrete by the plumbing: **the price you see quoted is the wholesale monetary price of gold; what you pay to truly hold it as insurance is always higher.** The screen shows you the world's no-confidence vote in paper money. The cash register adds the cost of opting out of paper entirely. Knowing the difference — knowing *which* gold price you are looking at, and why the others differ — is the first thing that separates someone who *owns* gold from someone who merely *watches* it.

## Further reading & cross-links

Within this series:

- `gold-futures-comex-contango-backwardation-and-paper-vs-physical` — the futures mechanics, roll, delivery and the paper-vs-physical debate in full.
- `the-sjc-premium-why-vietnamese-gold-trades-far-above-the-world-price` — the math from \$/oz to VND/lượng and why Vietnam's premium blew out and narrowed.
- `physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk` — allocated vs unallocated, storage, and the "your gold isn't your gold" trap.
- `gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal` — how ETFs hold allocated bars and track spot at near-zero spread.

Across the broader trading library:

- [Real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) — why the real rate anchors gold and every other asset.
- [The dollar system: why USD rules markets (DXY)](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — the numéraire effect at the level of the whole market.
- [Gold: money, insurance, or just a rock?](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock) — the portfolio-allocation framing of gold.
- [How monetary policy moves commodities, real rates, and gold](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) — the Fed transmission channel into the gold price.
