---
title: "Commodity regulation: position limits, the CFTC, and the anatomy of a squeeze"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A beginner-friendly deep dive into the rules that govern commodity futures — the CFTC, position limits, margin, and the exchange's emergency powers — and how those rules decide when a corner or squeeze runs and when a regulator halts it."
tags: ["regulation", "commodities", "cftc", "position-limits", "futures", "margin", "short-squeeze", "market-structure", "risk-management", "trading-playbook"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Commodity futures have their own rulebook — the CFTC, position limits, margin, and the exchange's emergency powers — and those rules, not just supply and demand, decide when a corner or squeeze can run and when a regulator or exchange steps in to halt it, force a liquidation, or even cancel trades.
>
> - The CFTC and the exchanges cap how much of one contract a single trader may hold (position limits), with a carve-out — the bona fide hedge exemption — for producers and users who own the physical commodity.
> - A short squeeze is a reflexive loop: a rising price triggers a margin call, the forced buy-to-cover lifts the price again, and only a default or an intervention ends it.
> - In the 1980 Hunt silver corner the exchange's rule change (raising margins and going liquidation-only) broke the longs; in the 2022 LME nickel halt the exchange canceled trades and rescued the shorts. The law chose the winner both times.
> - The one fact to remember: a winning commodity trade can be **canceled** by the exchange after the fact — the LME voided roughly \$3.9 billion of nickel trades in a single morning.

On the morning of March 8, 2022, the price of nickel on the London Metal Exchange did something no liquid futures market is supposed to do: it more than doubled in a matter of hours, blowing through \$50,000 a tonne, then \$60,000, then briefly past \$100,000 — a level it had never approached in the exchange's 145-year history. A large short position, held by a Chinese stainless-steel producer, was being squeezed: as the price rose, the losses on the short ballooned, the margin calls came due, and the only way to stop the bleeding was to buy back the contracts — which pushed the price up again. The market was eating itself.

What happened next was not a market event. It was a legal one. The LME stopped trading in nickel entirely, and then it did something extraordinary: it **canceled every nickel trade that had executed that morning**, rolling the market back to the prior day's close as if the spike had never happened. Around \$3.9 billion of transactions were voided. The traders who had been long nickel — who, on paper, had made fortunes as the price exploded — woke up to find those gains simply deleted by a rule the exchange invoked under its own rulebook. The lawsuits started within weeks.

This post is about the rulebook that made that possible. Commodity markets — oil, gold, wheat, copper, nickel — trade primarily through **futures**, and futures live inside a dense legal structure that most equity investors never have to think about: a federal regulator (the CFTC), hard caps on how much one trader may hold (position limits), a daily cash settlement system (margin and the mark-to-market), and a set of emergency powers that let the exchange halt trading, force liquidations, or void trades outright. Those rules are not background plumbing. They are the difference between a corner that succeeds and one that is broken, between a squeeze that runs to \$100,000 and one that is halted at \$50,000, and — as the nickel longs learned — between a winning trade and a canceled one.

![A futures market sits inside a legal stack where the CFTC, the exchange, and the clearinghouse each hold a lever over a squeeze](/imgs/blogs/commodity-regulation-position-limits-and-the-cftc-1.png)

This is the same idea that runs through the whole series — that a rule changes the *rules of the game*, and the practitioner who reads the rule early prices the consequence before it bites. We develop the general version in [how law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain). In commodities the link is unusually direct: the position limit *is* a ceiling on a corner, the margin formula *is* the engine of a forced liquidation, and the emergency power to cancel trades *is* the exchange deciding, after the fact, who gets paid.

## Foundations: how a commodity futures market actually works

Before we can read a squeeze, we have to build the market from zero. A handful of terms do all the work here, and each one is a rule as much as it is a market mechanism. Let us define them carefully.

### A futures contract and the role of the exchange

A **futures contract** is a standardized, legally binding agreement to buy or sell a fixed quantity of a commodity at a set price on a set future date. "Standardized" is the key word: one COMEX gold futures contract is *always* 100 troy ounces, one CME crude oil contract is *always* 1,000 barrels, one LME nickel contract is *always* 6 tonnes. Because every contract of a given type is identical, they are fungible and can be traded on an **exchange** — a central venue (the CME Group's NYMEX and COMEX in the US, the LME in London, ICE) that lists the contracts, sets their specifications, matches buyers with sellers, and publishes the prices.

A buyer of a futures contract is **long**: they profit if the price rises. A seller is **short**: they profit if the price falls. Crucially, you do not need to own the commodity to sell a futures contract, and you do not need the cash for the full value to buy one — you post only a fraction, called margin, which we define below. This is what lets a futures market carry far more open positions than there is physical commodity to deliver, and it is exactly what makes a corner possible.

### The clearinghouse: why nobody trades with a stranger

Here is the piece that beginners almost always miss, and it is the load-bearing institution in every squeeze story. When you buy a futures contract, you are not legally exposed to the specific person who sold it. Between every buyer and every seller stands a **clearinghouse** — a central counterparty that, the instant a trade is matched, splits it in two: it becomes the seller to every buyer and the buyer to every seller. If your counterparty defaults, the clearinghouse still pays you. This is called **novation**, and it is why a futures market can function at all: you never have to assess the creditworthiness of the anonymous trader on the other side.

The clearinghouse is not doing this for free or out of kindness. It protects itself with **margin** and the **daily mark**, and it backs those with a layered safety net — each member's posted margin first, then a mutualized **default fund** that all clearing members contribute to, then the clearinghouse's own capital. This is the "default waterfall," and it is why a single trader's blow-up does not normally take down the market: the loss is absorbed in layers. But the waterfall is also why an exchange will act forcefully to stop a squeeze before a loss reaches the mutualized layers — once a default is big enough to burn through one member's margin and into the shared fund, *every* member is on the hook, and the exchange's incentive to halt or cancel becomes overwhelming. That incentive is exactly what the LME faced in nickel. Margin and the daily mark are the two mechanisms that turn a price move into a forced cash payment — the engine of every squeeze. Let us trace the full path of a contract before we go deeper.

![A futures contract flows from trader through broker, exchange, and clearinghouse to a daily mark and final delivery or cash settlement](/imgs/blogs/commodity-regulation-position-limits-and-the-cftc-2.png)

### Margin and the daily mark-to-market

**Margin** is the good-faith deposit you post to open and hold a futures position. It is not a down payment on the full value — it is a performance bond, typically 5–15% of the contract's notional value, sized by the exchange to cover roughly one day's worst-case price move. Two flavors matter: **initial margin** (what you post to open the position) and **maintenance margin** (the floor your account must stay above; drop below it and you get a margin call).

The daily mechanism that makes margin work is the **mark-to-market**, or the **daily mark**. At the end of every trading day the clearinghouse revalues every open position at the day's settlement price and moves cash accordingly: if the price moved against you, cash is debited from your account and credited to the winner's, that same day. This is called **variation margin**, and it is settled in cash, daily, no exceptions. A futures loss is not a paper loss you can sit on — it is a cash payment due tomorrow morning. Hold that fact; it is the entire mechanism of a forced liquidation.

### The CFTC: the commodity futures regulator

The **Commodity Futures Trading Commission (CFTC)** is the US federal agency that regulates the commodity futures and options markets, created by the Commodity Futures Trading Commission Act of 1974. It is the commodity-market analog of the SEC (which regulates stocks and bonds — see [securities law 101: the '33 and '34 Acts and the SEC](/blog/trading/law-and-geopolitics/securities-law-101-the-33-and-34-acts-and-the-sec)). The CFTC's mandate is to protect market participants from fraud and manipulation and to ensure the markets are not distorted by excessive speculation. Its core legal tools are antifraud and antimanipulation authority, oversight of the exchanges (which it designates and supervises), and — central to this post — the power to set and enforce **position limits**.

A vital distinction: the CFTC sets the *federal* framework, but the **exchanges** write and enforce most of the operational rules — including their own position limits and, critically, their **emergency powers**. The CFTC approves and supervises; the exchange acts. When a squeeze hits, it is usually the *exchange* that pulls the trigger first, with the CFTC in the background. We will see this in both case studies.

### Position limits and the hedger-versus-speculator distinction

A **position limit** is a hard cap on the number of contracts a single trader (or group acting together) may hold in one commodity. The point is to stop any one player from accumulating a position large enough to distort the price or to corner the market. Limits are tightest in the **spot month** — the contract closest to delivery, where the squeeze risk is highest because the deliverable supply is fixed and known.

But a blunt cap would punish the people the market exists to serve. A wheat farmer who has 5 million bushels in the field needs to *sell* far more futures than a speculator's limit allows, simply to hedge the crop. So the rules carve out an exemption.

![Position limits cap a speculator hard while the bona fide hedge exemption lets a real producer hold a much larger offsetting position](/imgs/blogs/commodity-regulation-position-limits-and-the-cftc-3.png)

The **bona fide hedge exemption** lets a trader who has a genuine, offsetting physical position — a producer with the commodity in the ground, a refiner who must buy crude, an airline that must buy jet fuel — hold futures *above* the speculative limit, because their futures are offsetting real-world risk, not creating new risk. The legal test is that the futures position is "bona fide" — economically appropriate to the reduction of a real commercial risk — and the hedger must apply for and document the exemption. This is the single most important distinction in commodity regulation: a **speculator** is capped hard; a **hedger** with the physical exposure is not. A corner attempt by a speculator runs straight into the limit; a producer's large position is legal because it is hedging.

### A corner, a squeeze, and delivery mechanics

A **corner** is the deliberate accumulation of a dominant long position in both the futures *and* the physical supply of a commodity, so that the shorts — who must deliver the physical at expiry or buy back their contracts — find there is nothing left to buy except from the cornerer, at whatever price the cornerer demands. A **short squeeze** is the price spike that results: the shorts, trapped, bid the price up frantically trying to cover.

The mechanism only works because of **delivery and settlement**. Some futures (gold, oil, grains, metals) are **physically settled**: at expiry, a short who has not closed the position must deliver the actual commodity to a warehouse the exchange recognizes. Others are **cash settled**: no commodity changes hands, the position just settles against a reference price. Physical settlement is what makes a corner possible — if the shorts must deliver real metal and the cornerer owns the warehouses, the shorts are at the cornerer's mercy. We will see exactly this in the Hunt silver story.

### Open interest versus deliverable supply

One more number ties the whole thing together, and it is the one the squeeze hunter watches above all others: **open interest** — the total number of contracts currently outstanding (not yet closed or delivered). Open interest can be many times the physical commodity sitting in the exchange's warehouses, because most futures are closed for cash before expiry and never reach delivery. That is normal and healthy: a market where ten times the deliverable supply trades on paper is just a liquid market. The danger arises only when too many of those open contracts insist on going to *delivery* against a deliverable supply that is fixed and small.

The squeeze condition, then, is arithmetic: when the open interest in the expiring contract that intends to take or make delivery exceeds the deliverable supply in approved warehouses, the shorts as a group cannot satisfy the longs as a group. Someone must buy back at any price. A practitioner reads three numbers together — open interest near expiry, the concentration of that open interest in a few hands, and the warehouse stock — and when all three flash (high, concentrated, low respectively), the contract is squeeze-prone. We return to exactly these three numbers in the playbook.

#### Worked example: when open interest overwhelms the warehouse

Take a copper contract where the exchange-approved warehouses hold **50,000 tonnes** of deliverable metal, and one contract is 25 tonnes — so the deliverable supply is 50,000 ÷ 25 = **2,000 contracts**. Suppose the expiring spot-month contract has open interest of **5,000 contracts**, and a single long holds **1,800** of them and is standing for delivery.

- The shorts owe delivery on 5,000 contracts but the warehouses can only source 2,000. The system is short 3,000 contracts' worth of metal — 75,000 tonnes that does not exist where it is needed.
- A single long demanding delivery on 1,800 contracts (45,000 tonnes) is alone asking for 90% of the warehoused supply. The shorts cannot all deliver; they must buy back from the long.

The intuition: a squeeze is not about how much paper trades — it is about how much paper insists on becoming metal against a warehouse that cannot grow overnight. With these terms defined — futures, exchange, clearinghouse, margin, the daily mark, the CFTC, position limits, the hedge exemption, the corner, delivery, and open interest — we can now go deep on the mechanics that decide who wins.

## How position limits are meant to prevent a corner

Start with the purpose of the cap. A corner requires the cornerer to hold a position large relative to the deliverable supply. If a trader can own contracts representing, say, 5x the silver that physically exists in the exchange's warehouses, then at expiry the shorts cannot all deliver — there is not enough metal — and they must buy back at the cornerer's price. The position limit is the rule designed to make that arithmetic impossible: by capping any one trader well below the deliverable supply, no single player can hold a position the physical market cannot satisfy.

The limit is tightest in the spot month for a precise reason. Months out, the deliverable supply is elastic — more metal can be mined, more wheat planted, inventory shipped in. But in the final days before a physically settled contract expires, the deliverable supply is essentially fixed: it is whatever sits in the exchange-approved warehouses right now. That is the window in which a corner bites, so that is the window the rules guard most tightly. A speculative limit in the spot month might be a few thousand contracts; the same trader might face a far looser "all-months-combined" limit.

#### Worked example: how a position limit caps a corner

Take a silver market where the exchange recognizes **300 million ounces** of deliverable silver in its warehouses. One COMEX-style contract is **5,000 ounces**, so the total deliverable supply is 300,000,000 ÷ 5,000 = **60,000 contracts**. Suppose the spot-month position limit is **1,500 contracts** per trader. The most one trader can legally demand for delivery is 1,500 × 5,000 = **7,500,000 ounces** — which is 7.5M ÷ 300M = **2.5%** of the deliverable supply. At that size, the shorts can comfortably find the metal to deliver; there is no squeeze.

Now suppose there were *no* limit and a single trader amassed **40,000 contracts** — 200,000,000 ounces, or 200M ÷ 300M = **67%** of all deliverable silver. The shorts, needing to deliver against those contracts, would find that two-thirds of the metal is owned by the very person they owe. They would have to buy it back from the cornerer at whatever price was demanded. The position limit, set at 2.5% of supply, is what stands between those two worlds. The takeaway: a position limit is not red tape — it is the numerical ceiling on how big a corner can get, expressed as a fraction of the deliverable supply.

There are, in fact, two layers of limit, and the distinction matters for reading the rules. **Federal position limits** are set by the CFTC under the Commodity Exchange Act and apply across venues to a defined set of "core referenced" physical commodities — historically the agricultural staples (corn, wheat, soybeans, cotton), and after Dodd-Frank expanded to key energy and metals contracts. **Exchange position limits and accountability levels** are set by each exchange on its own contracts; an "accountability level" is softer than a hard cap — cross it and the exchange can demand an explanation and order you to stop adding, rather than forcing an immediate cut. The two layers interlock: the CFTC sets a federal floor on the most sensitive commodities, and the exchanges fill in the rest and police day to day.

The history is worth a sentence, because it explains why the rules look the way they do. Position limits trace to the 1936 Commodity Exchange Act, written after the grain-corner scandals of the 1920s and 1930s. They were tightened after the 1980 Hunt silver corner. And they were expanded again by the 2010 Dodd-Frank Act, which directed the CFTC to set limits across futures *and* the over-the-counter swaps that had grown up alongside them — closing the gap that let a player build an economic position in swaps beyond any single exchange's contract limit. The final CFTC position-limits rule took until 2020 to land, a decade of rulemaking that itself shows how contested the speculation question is (we trace the rulemaking-clock dynamic in [the regulatory calendar: trading the rulemaking clock](/blog/trading/law-and-geopolitics/the-regulatory-calendar-trading-the-rulemaking-clock)).

The limit is a blunt instrument, though, and it has two well-known failure modes. First, a determined player can spread a position across linked instruments — futures, options, over-the-counter swaps — to exceed the *economic* limit while staying under the *contract* limit on any single venue (this is roughly what regulators later accused some players of doing, and exactly what the Dodd-Frank swaps provision was meant to stop). Second, the limit only binds traders the regulator can see and reach; a position built offshore, or through nominees, or through an entity whose true ownership is obscured, can evade it until it is too large to ignore. Both failure modes show up in the case studies. The limit prevents the *naive* corner; the sophisticated one requires the exchange's emergency powers to break.

## The margin spiral: how a squeeze actually forces liquidation

Position limits are the *preventive* rule. The *reactive* mechanism — the one that determines what happens once a squeeze is underway — is margin. To see why a squeeze is so violent, you have to trace the daily-mark loop, because it is reflexive: the price move forces a cash payment, the cash payment forces a trade, and the trade moves the price again.

![A short squeeze is a reflexive loop where margin calls force buying that lifts the price into the next margin call](/imgs/blogs/commodity-regulation-position-limits-and-the-cftc-4.png)

Walk the loop one step at a time for a trapped short. The price rises against the short. At the daily mark, the clearinghouse debits the short's account for the loss — variation margin, due in cash. If the account falls below maintenance margin, the clearinghouse issues a **margin call**: post more cash by morning or your position will be liquidated. The short, often unable or unwilling to post unlimited cash, must reduce the position — which for a short means **buying back** contracts. That buying is fresh demand hitting the order book, so it pushes the price *up* again. The higher price marks the short's *remaining* position to an even bigger loss, triggering the next margin call. The loop accelerates.

This is the same reflexive structure that powers an equity short squeeze — the GameStop episode of January 2021 is the famous recent case — and it is closely related to the volatility-and-margin spiral that circuit breakers are built to interrupt, which we cover in [circuit breakers, halts, and the legal plumbing of a crash](/blog/trading/law-and-geopolitics/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash). What is distinctive about commodities is that the loop has a hard backstop the equity market lacks: at expiry, the short must *physically deliver*, so there is a date on the calendar by which the short must close — and the cornerer knows it.

#### Worked example: the margin spiral forces selling

Take a trader short **100 silver contracts** (5,000 oz each = 500,000 oz), entered at **\$25/oz**, with **\$15,000** initial margin and **\$12,000** maintenance margin per contract — so \$1.5M initial and \$1.2M maintenance on the position, against an account holding **\$2.0M**.

- The price rises \$2 to **\$27**. The short's mark-to-market loss is \$2 × 500,000 = **\$1,000,000**, debited that day. The account falls from \$2.0M to \$1.0M — below the \$1.2M maintenance floor. **Margin call: post \$0.5M** (back to \$1.5M initial) by morning.
- Say the trader can only raise \$0.2M. They are short \$0.3M of the requirement, so the clearinghouse liquidates roughly **20 contracts** to bring the position down. Liquidating a short means *buying* 100,000 oz — fresh demand.
- That buying nudges the price to **\$28**. The trader's *remaining* 80 contracts now mark to a further loss, triggering the next call.

Each turn of the loop forces more buying at a higher price. The intuition: in a squeeze the short is not choosing to buy high — the margin rule is *forcing* them to, and every forced purchase makes the next margin call worse.

The chart below makes the asymmetry visceral. A short's payoff is a straight downhill line: every dollar the price rises is a dollar of loss per ounce, multiplied by the contract size, with no cap. The margin posted — a few dollars an ounce — is a rounding error against the loss when the price doubles.

![A short's P&L slopes straight down as the price rises and a doubling from a 25 dollar entry is a 25 dollar per ounce loss](/imgs/blogs/commodity-regulation-position-limits-and-the-cftc-9.png)

#### Worked example: the short-squeeze P&L when the price doubles

Stay with the trader short **100 silver contracts** (500,000 oz) at **\$25/oz**. The squeeze takes the price to **\$50** — a clean doubling.

- Loss per ounce = entry − price = \$25 − \$50 = **−\$25/oz**.
- Total loss = \$25 × 500,000 oz = **−\$12,500,000**.
- Against initial margin of \$1.5M, the loss is **8.3x** the margin posted. The trader owes \$12.5M in variation margin they never deposited.

If the price had gone to **\$100** (the nickel-style quadrupling), the loss would be \$75 × 500,000 = **−\$37,500,000** — 25x the margin. A short's loss is theoretically unbounded because the price can rise without limit, while the margin is sized for a normal day. The intuition: shorting a squeezable commodity is selling something whose price can run away from you faster than you can post cash, which is precisely why the exchange's emergency powers exist.

## The exchange's emergency powers: the rulebook's kill switches

When a market turns disorderly — a corner forming, a squeeze running, a price that no longer reflects supply and demand — the exchange does not have to sit and watch. Its rulebook, approved by the CFTC, hands it an escalating menu of emergency powers, each of which moves the price or the position *by force* rather than by trading. These are the most consequential rules in this post, because they are where the law openly chooses a winner.

![An exchange facing a disorderly market can raise margin, impose limits, force liquidation, or cancel trades outright](/imgs/blogs/commodity-regulation-position-limits-and-the-cftc-6.png)

The powers, roughly in order of severity:

- **Raise margin requirements.** The exchange can lift initial and maintenance margin overnight, sometimes dramatically. This drains cash from leveraged positions and forces undercapitalized players to reduce. It is the most common first move because it is procedurally light and hits speculators harder than hedgers.
- **Cut position limits or impose them mid-stream.** The exchange can tighten the cap, forcing anyone over the new limit to sell down. This directly targets a cornerer.
- **Liquidation-only orders.** The exchange can declare that traders may *only* close existing positions, not open new ones. This kills a corner by preventing the cornerer from adding, while letting the trapped shorts unwind.
- **Halt trading and cancel trades.** The nuclear option: stop the market entirely, and — in the extreme — void trades that already executed, rolling prices back. This is what the LME did in nickel.

Where does the authority come from? It is written into the exchange's own rulebook, which the CFTC approves — and the CFTC has its *own* emergency powers in reserve. Under Section 8a(9) of the Commodity Exchange Act, the CFTC may direct an exchange to take "such action as in the Commission's judgment is necessary to maintain or restore orderly trading," including ordering position liquidation, setting emergency margin, and suspending trading. So the structure is layered: the exchange acts first under its rulebook, and the CFTC sits behind it with statutory backup. In practice the exchange moves faster, because it does not need a federal vote — its committees can convene and act in hours, as the LME did.

Two things make these powers extraordinary. First, they can be invoked *retroactively* — canceling trades that already happened means erasing realized gains and losses after the fact. That is the feature that makes a commodity gain genuinely contingent in a way an equity gain is not: an equity trade, once printed, is essentially final (busted trades are rare and tightly bounded); a commodity trade on a disorderly day can be unwound wholesale. Second, the powers are exercised under a broad, discretionary standard — "orderly market," "emergency" — that gives the exchange wide latitude and is hard to challenge after the fact, because a court reviewing the decision asks only whether the exchange acted within its rulebook and in good faith, not whether it picked the "right" winner. A trader who does not know these powers exist will assume a winning trade is theirs to keep. It is not. The exchange's emergency powers are the ceiling on every commodity squeeze, and the tail risk on every commodity *position*.

## The 1980 Hunt brothers silver corner

The textbook corner, and the case that defined position-limit policy for a generation, is the Hunt brothers' attempt to corner silver in 1979–80. Nelson Bunker Hunt and William Herbert Hunt — heirs to a Texas oil fortune — began accumulating silver in the mid-1970s, partly as an inflation hedge (this was the late-1970s inflation that Paul Volcker would soon crush; see [Paul Volcker's 1980 rate shock: killing inflation](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation)). By late 1979, the Hunts and their allies controlled an estimated **200 million ounces** of silver — physical bullion plus enormous futures positions — at a time when total above-ground deliverable silver was a small multiple of that. They were taking *delivery* rather than rolling their futures, draining the COMEX and CBOT warehouses of metal.

The price responded exactly as a corner predicts. Silver, which had traded around **\$6/oz** in early 1979, climbed through the autumn and then went vertical: it hit **\$50/oz** in January 1980 — an eightfold rise in a year. The shorts — dealers, jewelers, and speculators who had sold silver futures — were being squeezed brutally, facing delivery obligations against metal the Hunts effectively controlled.

#### Worked example: the economics of the Hunt corner

Take the deliverable silver supply the exchanges recognized in 1979 as roughly **150 million ounces** in approved warehouses (the Hunts also held metal abroad and in physical form). To corner it, you must control a dominant share. Suppose you target **200 million ounces** — more than the warehoused deliverable supply — at an average accumulation cost of **\$10/oz** as you drive the price up. That is a notional outlay of \$10 × 200,000,000 = **\$2.0 billion** of capital deployed (much of it on margin and credit).

If the corner works and you mark that 200M oz at the **\$50** peak, the position is worth \$50 × 200,000,000 = **\$10.0 billion** — a paper gain of **\$8.0 billion**. That is the prize, and it is why a corner is attempted despite the legal risk. But the gain is *paper* until you can sell, and you cannot sell 200M ounces into a market you have hollowed out without collapsing the very price you created. The intuition: a corner's profit exists only as long as you keep holding; the moment you try to monetize it — or the moment the rules force you to — it evaporates.

The rules forced it. In January 1980 the COMEX and the CBOT, watching the corner with alarm, used their emergency powers. They raised margin requirements sharply and then imposed the decisive rule: **"liquidation only"** — traders could close silver positions but not open new ones. That single rule change pulled the floor out from under the corner. The Hunts could no longer add to their position, and the trapped shorts could now unwind. New buying dried up; selling resumed; the price reversed.

The collapse was as violent as the rise. Silver fell from \$50 toward \$10 over the following weeks, culminating in **"Silver Thursday," March 27, 1980**, when the price crashed and the Hunts — who had borrowed heavily and faced their own margin calls on the way down — could not meet a margin call estimated at over **\$100 million** owed to their broker, Bache. Their brokers faced ruin: Bache itself was at risk of failing, and the contagion threatened the brokerage system, because a default that large would cascade through the clearing members. A consortium of banks arranged a **\$1.1 billion** bailout loan to the Hunts — not out of generosity, but to wind the position down in an orderly way and stop the cascade, the same logic that drives every too-big-to-fail rescue.

Notice the symmetry the case exposes. The very mechanism the Hunts used to squeeze the shorts — the margin-and-delivery machinery — turned on them the moment the price reversed: on the way down, *they* were the ones facing margin calls they could not meet. Leverage is direction-agnostic; the daily mark squeezes whoever is offside, cornerer or short alike. The Hunts were later found liable in civil litigation for manipulating the silver market (a jury awarded over \$100 million to a counterparty in 1988), fined by regulators, and effectively wiped out, with Nelson Bunker Hunt filing for personal bankruptcy in 1988.

The regulatory legacy outlived the Hunts. The episode hardened the case for strict spot-month position limits, tightened the rules on financing speculative commodity positions, and became the canonical example in every argument for the CFTC's authority. The lesson policymakers drew — and the reason position limits are written the way they are today — is that the corner was ultimately broken not by the market but by a *rule change*: the exchange raised margins and forbade new positions, and the law decided the longs would lose. The fundamentals of silver had not moved eightfold and back in fifteen months; the rulebook had.

## The 2022 LME nickel short squeeze

Forty-two years later, almost the mirror image played out in nickel — and this time the rule change rescued the *shorts*. The setup: Tsingshan Holding Group, a giant Chinese stainless-steel and nickel producer, held an enormous **short** position in LME nickel futures — by some estimates well over 100,000 tonnes — partly as a hedge against its own nickel production. In early 2022, nickel prices were already rising on the energy transition (nickel is a key battery metal) when Russia's invasion of Ukraine on February 24 added a supply shock: Russia is a major nickel producer, and sanctions risk sent buyers scrambling. The broader 2022 commodity shock is visible in oil, which jumped on the same catalyst.

![Brent crude jumped from about 79 to a peak near 128 USD per barrel around the February 2022 Russia invasion](/imgs/blogs/commodity-regulation-position-limits-and-the-cftc-7.png)

As nickel rose, Tsingshan's short was deep underwater, and the squeeze loop engaged: margin calls forced buying to cover, the buying lifted the price, and on March 8 nickel went parabolic — doubling intraday past \$50,000 and spiking briefly above **\$100,000/tonne**. The LME faced a systemic problem: if Tsingshan and other shorts could not meet their margin calls, the losses would land on the clearinghouse and its members, and several brokers were themselves on the hook.

So the LME used its emergency powers — and used the most extreme one. It suspended nickel trading, and then it **canceled all the trades** executed on March 8, rolling the market back to the March 7 closing price of around **\$48,000**. Roughly **\$3.9 billion** of transactions were voided. The market stayed closed for over a week. The shorts — above all Tsingshan — were rescued: the cancellations erased the catastrophic losses they would have crystallized at \$100,000. The longs, who had bought as the price spiked and were sitting on enormous gains, had those gains deleted.

The 2022 nickel halt and the 1980 silver corner are the two poles of how a rule change can resolve a squeeze.

![In 1980 the rule change broke the long corner while in 2022 the rule change canceled trades and rescued the shorts](/imgs/blogs/commodity-regulation-position-limits-and-the-cftc-5.png)

The lawsuits were immediate and consequential. Hedge funds Elliott Management and Jane Street sued the LME for roughly **\$472 million** combined, arguing the cancellation was an unlawful taking of their property — the exchange had reached into their accounts and deleted realized gains, exceeded its powers, and acted to protect its own clearing members and a favored short rather than to keep the market orderly. In late 2023 the UK High Court ruled in the LME's favor, finding the exchange had acted within its rulebook powers and rationally in the face of a genuine systemic threat; the claimants appealed and the Court of Appeal again sided with the LME in 2024. The precise legal merits matter less, for our purposes, than the structural fact the episode exposed: **an exchange can cancel your winning trade, and a court may well uphold it.**

The episode also drove reform. An independent review (by Oliver Wyman) faulted the LME's market surveillance for missing the build-up of the giant over-the-counter short that did not show in the visible on-exchange positions — the same offshore/OTC blind spot that limits the reach of position limits. The LME responded by introducing daily **price limits** (caps on how far a metal can move in one session, so a runaway move triggers an automatic pause rather than a discretionary cancellation), tighter position reporting including OTC exposure, and stronger surveillance. The deeper point: after every squeeze, the rulebook grows another layer of scar tissue, exactly as circuit breakers did after 1987. But the layer that mattered most in 2022 — the power to cancel trades — remains, and that is a tail risk written into the rulebook of every commodity market, one most traders learned for the first time in March 2022.

#### Worked example: the LME-cancellation counterfactual

Take a hedge fund that bought **200 lots** of LME nickel (each lot = 6 tonnes, so 1,200 tonnes) on the morning of March 8 at an average of **\$60,000/tonne**, and watched the price spike to **\$100,000** before the halt.

- Mark-to-market gain at \$100,000 = (\$100,000 − \$60,000) × 1,200 tonnes = \$40,000 × 1,200 = **\$48,000,000**.
- Even at the \$80,000 level where many trades printed, the gain would be (\$80,000 − \$60,000) × 1,200 = **\$24,000,000**.

When the LME canceled the March 8 trades, *both* the purchase and any intraday sale were voided — the fund's position was rolled back as if the morning never happened, and the **\$24–48 million** of realized or marked gains simply vanished. The intuition: in a normal market a gain that has printed is yours; under the exchange's emergency powers a gain can be unwound retroactively, so a commodity long's "profit" is not final until the trade is unrescindable — and on a wild day, it may never be.

## Financialization and the speculation-versus-fundamentals debate

A recurring policy fight sits underneath all of this: do speculators *set* commodity prices, or merely trade around fundamentals set by supply and demand? The question matters because position limits are justified largely as a curb on "excessive speculation," so how much speculation actually moves prices is the empirical heart of the rulebook.

The backdrop is **financialization** — the post-2004 surge of financial investors into commodities via index funds and ETFs. Pension funds and asset managers began allocating to commodity indices (the S&P GSCI, the Bloomberg Commodity Index) as a diversifier and inflation hedge, pouring tens of billions into long-only positions that simply roll futures forward. Critics argued this wall of passive money inflated the 2007–08 commodity boom — oil to \$147 in mid-2008 — divorcing prices from fundamentals and harming consumers. That argument drove the position-limit provisions of the 2010 Dodd-Frank Act (see [Dodd-Frank: the post-2008 rulebook](/blog/trading/law-and-geopolitics/dodd-frank-the-post-2008-rulebook)).

The loudest version of the critique came from money manager Michael Masters, who testified to Congress in 2008 that "index speculators" were the hidden force behind the spike — the **Masters hypothesis**. It is intuitive: a wall of new buying must lift prices. But the counterargument is just as concrete. Index investors trade *futures*, not the physical commodity; for every long futures contract there is a short, so paper positioning does not, by itself, remove a single barrel from the physical market or change the supply available for consumption. If financial demand truly inflated spot prices above what fundamentals justified, you would expect to see inventories *building* (someone hoarding the physical at the inflated price) — and through the 2008 spike, oil inventories were not building unusually. That absence is the strongest evidence against the pure-speculation story.

The evidence, on balance, is genuinely mixed. Many academic studies — and the CFTC's own staff work — found that speculative flows did not have a robust, systematic effect on price *levels* over time; prices tracked fundamentals (inventories, growth, supply shocks) far more than positioning, and the great 2008 oil round-trip lined up with a demand boom and then a global recession. Other work found speculation amplified short-run volatility, sped up the transmission of news into prices, or mattered at specific high-stress moments even if it did not set the long-run level. The honest summary is that speculators clearly *can* distort a specific contract in a specific squeeze (the Hunts did; that is manipulation, which is illegal regardless), but the broad claim that index investors *set* the price of oil is not well supported by the data. We treat the general "is it fundamentals or flows?" question in [geopolitics as a market factor: the risk-premium channel](/blog/trading/law-and-geopolitics/geopolitics-as-a-market-factor-the-risk-premium-channel) and in the cross-asset work on regimes at [cross-asset](/blog/trading/cross-asset).

What is *not* ambiguous is that geopolitical shocks move commodity prices hard and fast, and that the rulebook gets stress-tested exactly when they do. The geopolitical-risk index spikes on the same events that send oil, gas, and metals lurching.

![Geopolitical risk index spikes on 9/11, the 2003 Iraq war, and the 2022 Ukraine invasion that move commodity prices](/imgs/blogs/commodity-regulation-position-limits-and-the-cftc-8.png)

The takeaway for a reader: do not assume speculators set the price you are trading — the evidence says fundamentals dominate the level — but *do* assume that in a squeeze, positioning and the rulebook dominate the price, because that is precisely when the emergency powers come out.

## The bona fide hedge exemption in practice

Return to the exemption, because it is where the speculation debate meets the operational reality of the market. The whole edifice of position limits would strangle the commercial users the market exists to serve if there were no carve-out. The bona fide hedge exemption is that carve-out, and producers and consumers use it constantly.

A concrete picture: an airline knows it will burn 1 billion gallons of jet fuel next year and wants to lock in the cost. To hedge, it buys crude or heating-oil futures far in excess of any speculative limit — its futures are offsetting a real, quantifiable physical exposure (the fuel it must buy), so the position is bona fide and exempt. An oil producer does the mirror: it sells futures against the barrels it will pump, exempt for the same reason. A grain elevator, a copper smelter, a chocolate maker hedging cocoa — all rely on the exemption to run positions a speculator could never hold.

The legal test has teeth. The futures position must be economically appropriate to the reduction of a genuine commercial risk; it must be tied to actual or anticipated physical exposure; and the hedger must apply for and document it, subject to the exchange's and CFTC's review. The exemption is not a loophole for unlimited speculation — a player who claims a hedge exemption while taking a directional bet beyond its physical needs is misusing it, and that is exactly the kind of conduct regulators probe. But used properly, it is what lets the deliverable supply and the futures market connect: the hedgers, exempt and large, are the natural counterparties to one another and to the speculators who provide liquidity.

#### Worked example: sizing a producer's hedge under the exemption

Take a silver miner expecting to produce **10 million ounces** over the next year, wanting to lock in today's **\$25/oz** price. One contract is 5,000 oz, so a full hedge is 10,000,000 ÷ 5,000 = **2,000 contracts** sold short.

- If the speculative spot-month limit were **1,500 contracts**, the miner's 2,000-contract hedge would *exceed* it — illegal for a speculator.
- Under the bona fide hedge exemption, the miner documents 10M oz of expected production and is permitted the full 2,000 contracts, because each contract offsets real metal coming out of the ground.
- The economics: if silver falls to \$20, the miner loses \$5/oz on the physical (10M oz × \$5 = **−\$50M** on sales) but gains \$5/oz on the short futures (2,000 × 5,000 oz × \$5 = **+\$50M**) — the hedge offsets the loss exactly.

The intuition: the exemption is not a favor to producers — it is the recognition that a hedger's large position *reduces* risk in the system rather than creating it, which is the opposite of a corner.

## Common misconceptions

**"Speculators set commodity prices."** Mostly false, with a real caveat. The bulk of the academic and regulatory evidence finds that index and speculative flows do not robustly move commodity price *levels* over time — fundamentals (inventories, supply shocks, growth) dominate. When oil ran to \$147 in 2008 and back to \$30 within months, the swing tracked a demand collapse far more than positioning. The caveat: a single trader *can* manipulate a single contract in a squeeze (the Hunts moved silver from \$6 to \$50), and speculation can amplify short-run volatility. So "speculators corner specific contracts" is true and illegal; "speculators set the price of oil" is not well supported.

**"An exchange can't cancel your trade — a deal is a deal."** Flatly false. The LME canceled roughly **\$3.9 billion** of nickel trades on March 8, 2022, voiding realized gains of tens of millions for individual funds, and a UK court upheld the action as within the exchange's rulebook powers. Every major commodity exchange has emergency powers to halt, force liquidation, adjust prices, or cancel trades to maintain an orderly market. A printed gain on a wild day is not final until it can no longer be rescinded.

**"Position limits are trivial — just paperwork."** False, and the arithmetic shows why. A spot-month limit set at, say, 2.5% of deliverable supply is the numerical ceiling on a corner: it is the difference between one trader controlling 7.5 million ounces of silver and controlling 200 million. The 1980 silver corner was ultimately broken by the exchange tightening exactly these constraints (margin hikes plus liquidation-only). Limits are the single rule that decides whether the corner arithmetic is possible.

**"Margin is just a deposit you can ignore once you're in."** False, and it is the most dangerous misconception for a short. Margin is settled in cash *daily* via the mark-to-market; a loss is a cash payment due the next morning, not a paper figure you can sit on. In a squeeze, the margin call is what *forces* a trapped short to buy at ever-higher prices — the spiral is a margin mechanism, not a sentiment story.

## How it shows up in real markets

**A short squeeze (LME nickel, March 2022).** Nickel doubled intraday past \$50,000 and spiked above \$100,000/tonne in hours, driven by margin-call-forced covering on a giant short. The exchange halted trading and canceled ~\$3.9 billion of trades, rolling the market to the \$48,000 prior close. The signal that it was coming: an enormous, concentrated short position colliding with a supply shock (Russia/Ukraine), with open interest and inventories flashing that the deliverable supply could not satisfy the shorts. The tail that materialized: the exchange's emergency power to void trades.

**A margin-spike forced liquidation.** When the energy crisis of 2022 sent gas and power prices vertical, European utilities and energy traders with large short hedges faced margin calls running into the tens of billions of euros — not because their hedges were wrong, but because the daily mark demanded cash against positions whose prices had exploded. Several needed state liquidity backstops to meet variation margin. The mechanism was identical to the silver and nickel spirals: a price move forced a cash payment that forced a position change. We trace the energy side of this in [energy geopolitics: OPEC, the oil weapon, and the European gas shock](/blog/trading/law-and-geopolitics/energy-geopolitics-opec-the-oil-weapon-and-the-european-gas-shock).

**A position-limit-driven unwind (Hunt silver, 1980).** Silver ran from \$6 to \$50 as the Hunts cornered the metal, then collapsed back toward \$10 within weeks once COMEX raised margins and imposed liquidation-only. The unwind was not a change in silver's fundamentals — supply and demand had not moved eightfold — it was a *rule* change forcing the corner to unwind. The market structure, not the metal, set the price on the way down.

**A near-corner that the rules contained (cocoa, 2024).** Not every squeeze ends in a halt. In early 2024 cocoa prices tripled to record highs on genuine West African crop failures, and the futures market seized up — margin requirements were hiked repeatedly, open interest collapsed as traders could not fund their positions, and liquidity thinned dramatically. But there was no single cornerer and no trade cancellation; the exchange leaned on rising margin to force orderly deleveraging while the price discovered a new level driven by the real supply shock. It is a useful contrast: when the move is fundamentals-driven and the positioning is not concentrated in one trapped hand, the milder emergency powers (margin hikes) do the work, and the nuclear option stays holstered.

A reader will notice the common thread: in each case the *fundamental* news was real (a supply shock, an inflation hedge, a crop failure), but the *extreme* price move and its resolution were governed by the rulebook — margin, limits, and emergency powers — not by supply and demand alone. The fundamentals set the direction; the rulebook set the magnitude and chose who paid.

## How to trade it: the playbook

The point of all this is to read the rulebook the way a practitioner does — to spot squeeze risk before it runs, to size the tail, and to know what invalidates the view.

**Read the position and inventory data for squeeze risk.** The CFTC publishes the **Commitments of Traders (COT)** report weekly, breaking open interest into commercial (hedger) and non-commercial (speculator) positioning, and the exchanges publish daily warehouse stocks and concentration data (the "concentration ratios" showing what share the largest four traders hold). A massive, concentrated position — especially a crowded short — in a contract whose **deliverable inventories are low and falling** is the classic setup. When open interest near expiry exceeds what the warehouses can deliver, the spot-month contract is squeeze-prone. Watch the **spread** between the spot month and the next month: a violent move to **backwardation** (spot far above the deferred) is the market screaming that physical supply is tight right now — the squeeze signature.

#### Worked example: scoring squeeze risk from the data

Take a tin contract (5 tonnes each). The exchange reports warehouse stock of **3,000 tonnes** — 3,000 ÷ 5 = **600 contracts** of deliverable supply. The expiring spot-month open interest is **2,400 contracts**, and the concentration data show the largest long holds **40%** of it — 960 contracts. The front-to-next-month spread has flipped to **\$2,000/tonne** backwardation from roughly flat a month ago.

- Deliverable supply (600 contracts) is one-quarter of the open interest standing into expiry (2,400) — the shorts as a group cannot all deliver.
- A single long holding 960 contracts is asking for 4,800 tonnes against 3,000 tonnes in the warehouse — 160% of available metal.
- The \$2,000 backwardation says the physical is being bid up *now* relative to next month — the market is pricing immediate scarcity.

All three lights are red, so you size for a squeeze: avoid the short, and if you are long, treat any parabolic gain as un-bankable because intervention is now likely. The intuition: squeeze risk is a number you can compute from public data before the price moves, not a vibe you feel after it has.

**Watch margin as the trigger.** A squeeze does not run on sentiment; it runs on margin calls. When you see an exchange **raise margin requirements** mid-stream, read it as the exchange firing its first emergency power — it will force the weakest hands to reduce, which can accelerate the move before it breaks it. Rising margin is both a warning that the exchange is worried and a catalyst that will move the price.

**Size the exchange-intervention tail — your winning trade can be canceled.** This is the hardest lesson and the one specific to commodities. If you are long a contract that is spiking parabolically, your mark-to-market gain is *not* a safe asset: the exchange can halt, adjust prices, or cancel trades, as the LME did. Do not treat an extreme intraday gain as bankable. If you can, realize gains *before* a price reaches "disorderly" levels where intervention becomes likely, and never bet the firm on a gain that exists only because the market has gone vertical — that is exactly the regime in which the emergency powers come out.

**Position around the asymmetry, not into it.** A short in a squeezable, physically settled commodity has unbounded loss and a hard delivery deadline; the cornerer knows the date the short must close. If you must be short into a tight spot month, define the loss in advance with options (a long call caps the squeeze loss) rather than relying on a futures stop that may not fill in a runaway market. We cover the options side of hedging tail risk in [quantitative finance](/blog/trading/quantitative-finance).

**What invalidates the thesis.** A squeeze view dies when the deliverable supply turns out to be ample (inventories rise, metal ships in, the spread normalizes from backwardation toward contango), when the concentrated position is shown to be a *bona fide hedge* rather than a directional bet (a hedger's large short is not a corner), or when the exchange *removes* a constraint rather than adding one (lowering margin, lifting liquidation-only) — that signals the authorities judge the market orderly again. Conversely, an intervention thesis (betting the exchange will halt or cancel) is invalidated the moment the exchange signals it will let the market clear; do not assume a rescue that the rulebook permits but the exchange chooses not to use.

The synthesis is the series' core claim in its sharpest form. In most markets, the law sets the rules of the game and the players decide the outcome. In a commodity squeeze, the law *is* the outcome: the position limit caps the corner, the margin formula forces the liquidation, and the emergency power chooses who gets paid. Read the rulebook, and you are reading the price. We pull the full toolkit together in [the law, policy, and geopolitics playbook](/blog/trading/law-and-geopolitics/the-law-policy-and-geopolitics-playbook).

## Further reading & cross-links

Within this series:

- [Market-structure law: Reg NMS, PFOF, and short-selling rules](/blog/trading/law-and-geopolitics/market-structure-law-reg-nms-pfof-and-short-selling-rules) — the equity-market analog of the rules that govern who can hold what and how a squeeze is constrained.
- [Circuit breakers, halts, and the legal plumbing of a crash](/blog/trading/law-and-geopolitics/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash) — the margin-and-volatility spiral, and the kill-switches built to interrupt it.
- [Energy geopolitics: OPEC, the oil weapon, and the European gas shock](/blog/trading/law-and-geopolitics/energy-geopolitics-opec-the-oil-weapon-and-the-european-gas-shock) — the commodity supply shocks that stress-test the rulebook.
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the general spine: a rule change reprices an asset before it fully bites.

Cross-asset and quantitative mechanism:

- [Cross-asset](/blog/trading/cross-asset) — how commodity shocks transmit across asset classes and regimes.
- [Quantitative finance](/blog/trading/quantitative-finance) — options, hedging, and capping the unbounded loss of a short into a squeeze.
- [Dodd-Frank: the post-2008 rulebook](/blog/trading/law-and-geopolitics/dodd-frank-the-post-2008-rulebook) — the statute that expanded CFTC position-limit authority over swaps.
- [Paul Volcker's 1980 rate shock: killing inflation](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation) — the macro backdrop to the Hunt silver corner.
