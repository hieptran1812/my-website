---
title: "Physical Gold: Bars, Coins, Allocated vs Unallocated, and Counterparty Risk"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The whole point of gold is to be no one's liability. A beginner's guide to actually owning the metal: bars vs coins, premiums and spreads, allocated vs unallocated vs pooled, storage, and the counterparty traps that quietly turn your gold into someone else's promise."
tags: ["gold", "physical-gold", "bullion", "allocated-gold", "unallocated-gold", "counterparty-risk", "gold-bars", "gold-coins", "gold-storage", "premiums", "rehypothecation", "precious-metals"]
category: "trading"
subcategory: "Gold"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The whole point of gold is to be no one's liability; the moment you hold it *through* someone else, you reintroduce the exact counterparty risk you bought gold to escape.
>
> - **Form is the entire game.** Physical metal in your hand is no one's promise. An "unallocated gold account" at a bank is an *unsecured loan to that bank*, dressed up as gold — in a crisis it is a claim on the very institution you wanted protection from.
> - **You pay a premium to hold the metal**, on top of the wholesale spot price: ~1-2% on a big bar, ~5% on a small coin. That spread is the price of insurance you can touch.
> - **Allocated** = specific numbered bars that are legally *your property*; **unallocated** = a number on the bank's ledger that you are a creditor for; **pooled** = a fractional share of one common heap. Only allocated survives the bank's bankruptcy.
> - The one number to remember: a 1 oz coin bought at a **5% premium** and sold back near spot costs you roughly a **6% round-trip** — the toll for owning gold that no counterparty can default on.

In March 2020, as the pandemic seized the financial system, two people decided to buy gold on the same afternoon. The first opened the trading app for her bank's "gold savings account," tapped to convert \$10,000 of cash into gold, and watched the balance update: *5.20 oz*. Done in fifteen seconds, no shipping, no storage, no fuss. The second drove to a coin dealer, waited in a line out the door, and was quoted a price for one-ounce coins that was nearly 12% above the number flashing on the screens behind the counter — when there were any coins to be had at all. He paid it.

A year later, the first investor's bank quietly amended the terms of its gold account: in "extraordinary market conditions," redemptions in physical metal could be suspended and settled in cash at the bank's discretion. She had never owned an ounce of gold. She had owned a *claim* on her bank that happened to track the gold price — and the bank had just reminded her, in the fine print, that the claim was theirs to define. The second investor had a heavy, awkward, expensive stack of metal in a safe. In a genuine crisis, only one of them held the thing gold is for.

This is the part of gold that almost everyone gets wrong, and it has nothing to do with the price. The screen price — spot, the [London fix](/blog/trading/gold/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce), the futures quote — tells you what an ounce of *wholesale* gold is worth. It tells you nothing about whether the gold you "own" is actually yours, or a promise you'll have to stand in a queue to collect. The spine of this entire series is that **gold is monetary insurance — a bet against paper money, real yields, and political stability.** And here is the cruel twist this post exists to expose: *if you hold that insurance in a form that depends on a counterparty, you have re-bought the very risk gold was supposed to insure you against.* The form of ownership is the whole game.

![The gold ownership ladder by counterparty risk from physical metal to paper claims](/imgs/blogs/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk-1.png)

This post is the practical map. We'll define what physical gold actually is — purity, the troy ounce, the premium and the spread — then walk up the ladder of ownership rung by rung: bars versus coins, allocated versus unallocated versus pooled, the storage choices and their trade-offs, and the failure modes that turn "my gold" into "the bank's gold." By the end you'll be able to look at any gold product and answer the only question that matters in a crisis: *whose liability is this?*

## Foundations: what physical gold actually is

Before we can talk about how to *own* gold, we have to be precise about what the thing is. A lot of confusion in this space comes from people using the word "gold" to mean five different things — a price, a metal, an account, a fund, a contract — without noticing they've switched. Let's nail down the metal first.

### Purity and fineness — what "pure gold" means

Gold in the ground is mixed with other metals. Refining strips those out, and the result is rated by **fineness**: the fraction of the bar or coin that is actually gold, by weight. You'll see it two ways:

- **Fineness in parts per thousand**, written as a number like **999.9** ("four nines") — meaning 999.9 grams of every 1,000 grams are pure gold, i.e. 99.99% pure. Investment bars and modern bullion coins are typically 999.9 or 999.
- **Karat** (spelled *carat* outside the US), an older jewelry scale out of 24. **24 karat** is pure gold (≈99.9%). **22 karat** is 22/24 = 91.7% gold (the rest usually copper, for hardness). **18 karat** is 75%.

This matters because the **price you see quoted is always per ounce of *fine* (pure) gold.** A one-ounce object that is only 22 karat contains about 0.917 oz of actual gold, so its metal value is lower than the headline ounce price. When you buy bullion, you want to know both the *gross* weight and the *fineness*, because only their product — the **fine gold content** — is what the spot price values. A "1 oz" American Gold Eagle, for instance, is 22 karat but is deliberately minted *heavier* than an ounce so it still contains a full troy ounce of fine gold; a Canadian Maple Leaf is 999.9 and weighs exactly an ounce. Same gold content, different gross weight.

### The troy ounce — gold's own unit

Gold is weighed in **troy ounces**, not the kitchen ounce. A troy ounce is **31.1035 grams** (call it 31.103 g), about 10% heavier than the supermarket "avoirdupois" ounce of 28.35 g. Every other bullion unit is a fixed multiple of it: a **gram** is 0.03215 troy oz; a **kilobar** is 1,000 g = 32.151 troy oz; a Vietnamese **lượng** (tael) is 37.5 g = 1.20565 troy oz. These ratios never change — converting between them is pure arithmetic, which is why a price quoted in one unit translates cleanly into any other. (The [pricing post](/blog/trading/gold/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce) does the full conversion grid; we lean on it here rather than re-deriving it.)

One practical consequence is worth internalizing now: because the unit is fixed and purity is fixed, **the only variable in a physical gold transaction is the premium.** Two dealers selling you a 999.9 one-ounce Maple Leaf are selling you the *identical* object — same metal, same weight, same mint. The only thing you're comparing is the markup over spot and the buy-back price. This makes physical gold one of the few markets where shopping on price is purely rational: there is no "quality" difference to pay up for between two pieces of the same product. The whole skill of buying physical metal reduces to *minimizing the premium you pay and maximizing the bid you'll later receive* — which is why we spend so much of this post on premiums and spreads.

### Why gold is "no one's liability" — the idea the whole post turns on

This phrase will recur, so let's make it concrete. Every financial asset other than a physical commodity is a **claim on someone**. A share of stock is a claim on a company's future profits and assets. A bond is a claim on a borrower's promise to pay. Your bank balance is a claim on the bank — legally, the bank *owes* you that money; it isn't sitting in a drawer with your name on it. Even a banknote is, on a central bank's balance sheet, a *liability* — a promise. Every one of these has a counterparty whose failure, dishonesty, or coercion can vaporize the claim.

A bar of gold has none of that. It is a lump of an element that doesn't rust, doesn't decay, isn't issued by anyone, and doesn't depend on any institution's survival to retain its value. Nobody can "default" on a gold bar the way a company defaults on a bond. That property — being an *asset without a corresponding liability* — is gold's entire monetary superpower, and it's why central banks, the most sophisticated money managers on earth, hold thousands of tonnes of a metal that pays no interest. The catch this whole post exists to expose: that superpower lives in the *physical metal*, not in the *price exposure*. The moment you swap the metal for a claim on an institution that tracks the gold price, you've thrown away the one feature you were paying for and kept only the price chart.

### Spot, premium, and the bid/ask spread — defined from zero

Three terms run through this entire post. Let's build them from nothing.

**Spot** is the wholesale price for immediate delivery of a standardized large quantity of gold — the number on the screens, quoted per troy ounce, almost always in US dollars. It's the price at which big institutions trade 400 oz "Good Delivery" bars with each other. It is *not* a price you, a retail buyer, can transact at. Think of it as the factory-gate price of gold.

**Premium** is the amount *above spot* you pay to buy a specific physical product. If spot is \$2,400/oz and a dealer sells you a one-ounce coin for \$2,520, the premium is \$120, or 5%. The premium pays for everything between the wholesale bar and the coin in your hand: refining and minting (fabrication), the dealer's margin, shipping and insurance, and — crucially — *scarcity*, which spikes exactly when everyone wants metal at once.

**Bid/ask spread** is the gap between the price a dealer will *buy* from you (the bid) and the price they'll *sell* to you (the ask). Dealers, like any market maker, make money on this gap. When you buy a coin you pay the ask (spot + premium); when you sell it back you receive the bid (often near spot, or even below). The *round-trip cost* — buy high, sell low — is what physical ownership actually costs you, and it is the single most underappreciated number in this whole subject.

#### Worked example: the round-trip cost of a one-ounce coin

Suppose spot gold is **\$2,400/oz**. You buy one 1 oz bullion coin at a **5% premium**:

- **Buy price (ask):** \$2,400 × 1.05 = **\$2,520**.

The next day — gold hasn't moved — you change your mind and sell it back to the same dealer. Dealers typically buy small coins back at, or just under, spot. Say they pay **\$2,376** (1% *under* spot, a realistic buy-back):

- **Sell price (bid):** \$2,400 × 0.99 = **\$2,376**.
- **Round-trip loss:** \$2,520 − \$2,376 = **\$144**, which is \$144 / \$2,520 = **5.7% of what you paid**.

Gold's price never changed, yet you're down nearly 6%. That ~6% round-trip is the toll for holding insurance you can physically touch — and it tells you immediately that physical gold is a *hold-for-years* instrument, not something you flip. The premium and the spread are the price of the one feature that matters: nobody can default on metal in your safe.

## Bars vs coins: same metal, very different premiums

Once you've decided to own physical gold, the first real choice is *what form* — and it is almost entirely a trade-off between **premium** (cheaper per ounce in big units) and **liquidity/divisibility** (easier to sell and to split in small units). The metal is identical; the packaging is what you're pricing.

### Bars

A **bar** (or **ingot**) is a simple rectangular slab of gold, stamped with its weight, fineness, refiner, and a serial number. They run from tiny **1 gram** bars up through **1 oz**, **10 oz**, **100 g**, **1 kilobar** (32.15 oz), and the wholesale **400 oz Good Delivery bar** (~12.4 kg, the kind central banks hold; worth nearly \$1 million at \$2,400/oz).

The defining feature of bars is **low premium per ounce**, and it gets lower the bigger you go, because the fixed costs of making and handling a bar are spread over more metal. A kilobar might carry a **1-2% premium**; a 400 oz bar trades essentially at spot. The catch: bars are *lumpy*. A kilobar is a single ~\$77,000 object. You can't shave 5% off it to pay for something. And the very biggest bars (400 oz) trade at spot *only inside the wholesale system* — once a bar leaves an LBMA-accredited vault, it loses its "chain of integrity" and may need re-assay before a wholesale buyer will take it at spot, which is a real cost. This is why serious physical buyers often prefer kilobars (small enough to handle, big enough for a thin premium) over either extreme.

That phrase — **chain of integrity** — is worth pausing on, because it explains a subtle premium most people never see. The wholesale market trusts a bar to be exactly what it's stamped *only* if the bar has never left the custody of accredited vaults and carriers since it was poured by an LBMA Good Delivery refiner. That unbroken custody chain is what lets institutions trade 400 oz bars at spot without re-testing each one. The moment a bar leaves that chain — say, you took delivery and kept it in your closet for a decade — a wholesale buyer can no longer be *sure* it wasn't tampered with, so they may demand an assay (a destructive or semi-destructive purity test) before paying full price. The cost and hassle of re-assay is a hidden "exit premium" on large bars held outside the system. Coins and small bars sidestep this differently: a sovereign-minted coin is recognizable enough that dealers will buy it back on sight, no assay needed. You're paying the higher coin premium partly to buy *liquidity on the way out* — the ability to sell without proving the metal all over again.

### Coins

A **coin** is a minted, often government-issued round of bullion — the American Eagle, Canadian Maple Leaf, South African Krugerrand, Austrian Philharmonic, British Britannia. They come mainly in **1 oz** and fractional sizes (1/2, 1/4, 1/10 oz).

Coins carry **higher premiums** than bars — typically **4-8% on a 1 oz coin**, and *much* higher on fractional coins (a 1/10 oz coin can carry a 10-15% premium, because the minting cost is nearly the same as a 1 oz coin but spread over a tenth of the metal). In return you get three things: **recognizability** (a Maple Leaf is instantly accepted by any dealer worldwide, no assay needed), **divisibility** (you can sell three coins instead of cutting up a bar), and **legal-tender status** in some jurisdictions, which can matter for tax. Coins are what you want if you might need to liquidate piecemeal, or sell to a private buyer who trusts a sovereign mint's stamp more than a refinery's.

The fractional-coin premium deserves a hard look, because it's where beginners quietly overpay. The cost of striking a coin — the die work, the handling, the packaging — barely changes between a 1 oz and a 1/10 oz coin. But on the 1/10 oz coin that near-fixed cost is spread over one-tenth of the metal, so it shows up as a far larger *percentage*. A 1/10 oz coin at a 13% premium means you're paying \$13 of premium for every \$100 of gold — versus maybe \$5 on the full ounce. People buy fractionals imagining they'll be handy for "making change" in some collapse scenario, but you pay dearly for that optionality up front, and the round-trip spread on a fractional is brutal. Unless you have a specific reason to need sub-ounce divisibility, the 1 oz coin is the sweet spot: recognizable, liquid, and not paying the fractional surcharge.

There's a sub-trap here worth flagging early: **numismatic** (collectible / "rare") coins are *not* the same as bullion coins, and they are not a store of gold value — more on that in the misconceptions section. The simple tell is the premium: a bullion coin's price moves tick-for-tick with the gold price plus a small, stable premium; a numismatic coin's price is dominated by a large, fashion-driven collector premium that has little to do with the metal inside.

#### Worked example: kilobar vs one-ounce coins for the same \$77,000

You want to put **\$77,000** into physical gold at **\$2,400/oz spot**. Two ways:

- **One 1 kg bar at a 1.5% premium.** Metal content = 32.151 oz. Cost = 32.151 × \$2,400 × 1.015 = \$77,160 × 1.015 ≈ **\$78,318**. Premium paid ≈ **\$1,158**.
- **32 one-ounce coins at a 5% premium.** Cost = 32 × \$2,400 × 1.05 = 32 × \$2,520 = **\$80,640**. Premium paid = **\$3,840**.

The coins cost you **\$2,682 more** for the *same amount of gold* — about 3.4% of the position, purely in extra premium. What did the extra money buy? Divisibility and instant recognizability: you can sell five coins to cover an emergency without disturbing the rest, whereas the bar is all-or-nothing and may need re-assay if it has left a vault. The rule of thumb falls out cleanly: **buy the biggest unit you can imagine selling in one piece** — bars for the bulk of a holding, a handful of coins for the part you might need to liquidate in small amounts.

![The premium stack from spot to retail price for a kilobar versus a one ounce coin](/imgs/blogs/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk-3.png)

The figure above breaks the retail price into its layers. The bottom blue block — spot — is the same for both products; it's the wholesale monetary price every premium sits on top of. What differs is everything stacked above it: a thin amber sliver of fabrication and dealer margin on the kilobar, versus a thick stack on the coin where the same fixed minting cost is spread over a single ounce, plus a scarcity premium that small, popular coins attract. Two products, identical metal, very different totals — and the difference is entirely *form*, not gold.

## Allocated vs unallocated vs pooled: the distinction that decides everything

Here is the most important section in this post, and the one most people never learn until it's too late. When you "own gold" through an institution — a bullion bank, a dealer's vault program, a gold savings account — you are holding it in one of three legal forms, and they are *radically* different in what you actually own. They can quote the same price and the same ounces while meaning entirely different things.

### Allocated gold — your property

**Allocated** means specific, identified, serial-numbered bars (or a precise number of coins) are set aside and recorded as **your legal property**. The vault holds them *for* you, as a bailee — like a coat-check holding your coat. The bars do **not** appear on the custodian's balance sheet, because they were never the custodian's; they're yours. If the bank or vault operator goes bankrupt, your allocated bars are **not** part of the bankruptcy estate. The administrator identifies them by serial number and hands them back. You pay a **storage fee** for this — typically **0.1% to 0.5% of value per year** — precisely *because* the metal is genuinely segregated and yours.

This is the gold equivalent of holding the metal yourself, with the vault providing security and insurance. Counterparty risk is low: the failure you're exposed to is the vault physically losing the metal (theft, fire — covered by insurance), not the institution's solvency.

### Unallocated gold — you are a creditor

**Unallocated** means you do **not** own any specific gold. You own a *claim* — a number of ounces the institution **owes** you, recorded as a liability on **its balance sheet**. The bank holds *some* gold somewhere to back its aggregate unallocated obligations, but no particular bar is yours, and the bank is free to use that gold in its business: lend it, lease it, pledge it. You are, in plain terms, an **unsecured creditor** of the bank for a quantity of gold.

This is the dominant form of "gold accounts," and it's popular for one reason: **it's free or nearly free** — no storage fee, because there's nothing specifically stored for you. That's the lure. But read what it means when the music stops: if the bank becomes insolvent, your unallocated gold is just another unsecured debt. You join the creditor queue alongside bondholders, and you may recover cents on the dollar — or wait years. The cruel irony writes itself: you bought gold to escape the financial system's fragility, and you hold it as a loan to a bank. **An unallocated gold account is a bet on the bank's solvency wearing a gold costume.**

### Pooled / fractional gold — a slice of a common heap

**Pooled** (or fractional) ownership sits in between: you own a *fractional, undivided share* of a common pool of metal, along with other investors. Many online gold platforms and "buy gold by the gram" apps work this way. Whether this is safe depends entirely on the fine print: if the pool is **fully allocated and segregated** from the operator's own assets (a properly structured custodial pool), your slice can be bankruptcy-remote, much like allocated. If it's **unallocated at the pool level** — the operator merely promising that enough metal exists somewhere — you're back to being a creditor, just of the platform instead of a bank. The word "pooled" alone tells you nothing; you have to read whether the pool is segregated and who holds title.

### Why the whole London market runs on unallocated

It would be easy to read the above and conclude that unallocated gold is some kind of scam. It isn't — it's the *plumbing of the entire global bullion market*, and understanding why tells you when it's fine and when it's dangerous. The London market (the LBMA system that sets the price most of the world references) settles enormous daily volumes in **unallocated** balances held at bullion banks. It does this because allocated settlement is clunky: moving specific numbered bars between parties for every trade would be slow and expensive, whereas unallocated balances net out like bank money — debit one account, credit another, no metal physically moves. Unallocated is to gold what a bank deposit is to cash: an efficient *book-entry* claim that makes a market liquid.

And that analogy is exactly the point. A bank deposit is wonderfully convenient and works flawlessly — right up until a bank run, when everyone discovers there isn't enough cash for all the claims at once. Unallocated gold is the same: it works flawlessly as long as not everyone demands physical conversion simultaneously. The system relies on the fact that, in normal times, almost no one converts. **But "works until everyone wants out at once" is the definition of a run — and protection from runs is the exact reason you bought gold.** So unallocated has a perfectly legitimate role for institutions managing liquidity and for short-term trading, where convenience matters and a systemic seizure is not the thing you're hedging. It is the *wrong* form for the one job most retail buyers want gold to do: insurance against a financial-system failure. Using unallocated gold as crisis insurance is like buying flood insurance from a company that keeps its reserves in your own basement.

### How to tell which form you actually have

The labels in marketing material are slippery, so judge by the *terms*, not the name. Three questions cut through almost everything:

- **Is there a storage fee?** If storage is free, you almost certainly hold unallocated or ungated-pooled metal — because nobody segregates and insures real bars for you for free. The fee is the fingerprint of genuine allocation.
- **Are specific bars identified to you?** Allocated programs give you serial numbers, a bar list, weights, and refiner — your specific metal, auditable. If you can't get a bar list, you don't have allocated gold.
- **What happens in the provider's bankruptcy?** Read the actual terms. "Title passes to the customer" / "held as bailee" / "segregated client assets" point to allocated. "The company owes the customer" / "general obligation" / "the customer is a creditor" point to unallocated. If "redemption may be suspended or settled in cash at our discretion," that is a flashing red light: you hold a claim the issuer can redefine, not metal.

When in doubt, assume unallocated — providers advertise allocation loudly when they offer it, precisely because it costs them more to provide.

![Matrix comparing allocated unallocated and pooled gold on legal status balance sheet and bankruptcy](/imgs/blogs/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk-2.png)

The matrix lays the three side by side on the only rows that matter. Notice the pattern: the *cheap* options (unallocated, ungated pooled) are cheap precisely because nothing is set aside for you, which is the same reason they evaporate in a default. The *safe* option (allocated) costs a storage fee precisely because real metal is genuinely segregated as your property. **There is no free lunch — a "free" gold account is telling you, in the price, that you don't own any gold.**

#### Worked example: the storage fee on allocated gold is the price of bankruptcy-remoteness

You hold **\$100,000** of gold. Compare:

- **Unallocated account:** storage fee **\$0/year**. You save \$400/year versus allocated. But you are an unsecured creditor: in the bank's insolvency, a typical unsecured recovery might be, say, **40 cents on the dollar**, putting **\$60,000 at risk** in a true failure.
- **Allocated vault at 0.4%/year:** fee = \$100,000 × 0.004 = **\$400/year**. In the vault operator's insolvency, your serial-numbered bars are returned in full — **\$0 at risk** from insolvency (only the insured physical risks remain).

So the "expensive" option costs **\$400 a year** and the "free" option costs **\$0 a year but up to \$60,000 in the one scenario gold is supposed to protect you from.** Framed that way, the \$400 isn't a fee — it's the **premium on the only insurance that pays out when the financial system is the thing failing.** That is the entire reason allocated exists.

## Storage: home, safe-deposit box, or vault — and the trade-offs

Suppose you've bought physical metal outright. Now you have a problem rich people have had for five thousand years: *where do you put it?* Every option trades **control**, **counterparty risk**, **cost**, **insurance**, and **accessibility** against each other, and there is — again — no free lunch.

### Home storage

Keeping gold at home (a quality safe, ideally bolted down and hidden) gives you **total control and zero counterparty risk**: no third party can freeze it, no institution can fail, you can hold it in your hand tonight. That is its whole appeal, and for a modest holding it's perfectly reasonable. The costs are real but different in kind: **theft and physical loss** (a home safe deters opportunists but not determined burglars), the need to buy your *own* insurance (standard homeowner's policies cap precious-metals coverage very low — often a few thousand dollars — so you need a specific rider, and insurers may require a safe of a certain rating), and the **operational security** burden of not telling people what you keep at home. Home storage is control purchased with personal risk.

### Bank safe-deposit box

A **safe-deposit box** at a bank feels like the safe middle path. It is more dangerous than people assume. First, **the contents are not FDIC-insured** — that famous deposit insurance covers *deposits*, not the random objects in a box; if the bank is robbed or floods, your gold is generally *not* covered unless you bought separate insurance. Second, **access is gated by the bank**: bank hours, bank holidays, and — the part that matters for gold specifically — *bank closures*. In a banking crisis or a bank holiday (exactly the scenario you bought gold for), the branch may be shut and your box inaccessible for days or longer. Third, in some jurisdictions and historical episodes, **box contents have been subject to legal orders** (see the confiscation discussion below). A safe-deposit box reduces home theft risk but reintroduces a bank as a gatekeeper — a partial step back down toward counterparty exposure.

### Allocated vault / depository

A professional **bullion depository** (Brink's, Loomis, the vaults run by mints and exchanges, or specialist precious-metals storage firms) holds your metal **allocated** — segregated, serial-numbered, your property — with **full all-risk insurance** included, audited inventories, and professional security. You pay the **0.1-0.5%/year** storage fee. This is the institutional-grade answer: it gives you the bankruptcy-remoteness of true ownership plus insurance you don't have to arrange yourself. The trade-off is **accessibility and trust**: the metal isn't in your hand, getting it requires shipping or selling through the depository, and you are trusting the operator to actually hold what they say (reputable depositories publish independent audits precisely to address this). For larger holdings, this is usually the right answer; for "grab bag in a true collapse" money, home wins.

Two details separate a *good* depository arrangement from a deceptively-labeled one, and they're worth checking explicitly. First, **allocated must mean allocated**: the contract should state your specific bars are your property held in bailment, segregated from the operator's assets and from other clients' metal, and *not* available to the operator as collateral. Some "storage" programs are quietly *unallocated* — the operator owes you metal but commingles and may lend it — which puts you right back on the creditor ladder despite paying a fee. Demand a bar list and confirm the bankruptcy language. Second, **the insurance should be the vault's, not yours to arrange**: a proper depository carries an all-risk policy (often underwritten at Lloyd's) covering theft, loss, and damage at full value, and an independent auditor periodically counts the bars against the client register. If a program can't show you its insurance terms and its audit cadence, you're trusting a promise — exactly the thing gold is supposed to free you from. Done right, allocated depository storage is the closest you can get to "metal you control" while still outsourcing security and insurance; done carelessly, it's an unallocated claim wearing a vault's uniform.

![Matrix of gold storage options home safe deposit box and allocated depository across control cost and insurance](/imgs/blogs/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk-5.png)

The storage matrix makes the trade-off unavoidable. Read down any column and you'll find a weakness: home storage has theft and self-insurance burden; the safe-deposit box has bank access risk and no automatic insurance; the depository has cost and the "not in your hand" gap. **The right answer is rarely one option — it's a split:** a depository (or allocated) for the bulk, where insurance and bankruptcy-remoteness matter most, and a modest amount of recognizable coins at home for the scenario where the financial system itself is the thing you can't reach.

## Counterparty and custody risk: the "paper claims on metal" trap

Now we get to the heart of it. Everything above has been circling one idea: **gold's unique value is that it is no one's liability.** A stock is a claim on a company. A bond is a claim on a borrower. A bank deposit is a claim on a bank. A dollar bill is, technically, a liability of the central bank. *Every* paper asset is someone's promise, and promises can be broken, frozen, defaulted, or inflated away. Gold, alone among major assets, is *no one's promise* — a bar of gold doesn't depend on anyone's solvency, honesty, or goodwill. That is the entire reason it has served as money and as crisis insurance for five thousand years.

**The trap is that the moment you hold gold through someone else, you bolt a counterparty back onto it.** You take the one asset whose superpower is "no counterparty" and you give it a counterparty anyway. Depending on *how* you hold it, you've reattached a different liability:

- Hold **unallocated**, and your "gold" is a claim on a bank — you've reattached *bank solvency risk*, the thing gold exists to dodge.
- Hold gold via a **dealer's storage program** that turns out to be unallocated or fractional, and you've attached *the dealer's solvency*.
- Hold a **gold ETF**, and you've attached a chain: the trust, the custodian, the sub-custodians (covered in depth in the [ETF post](/blog/trading/gold/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal) — for most investors GLD is fine, but it *is* a chain of counterparties, not metal in your hand).
- Hold **futures**, and you hold a contract — leverage and a clearinghouse, with far more paper claims outstanding than deliverable metal (the [futures post](/blog/trading/gold/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce) covers the paper-vs-physical ratio).

### Rehypothecation — when your metal is lent out behind your back

There's a specific, technical version of this trap with an ugly name: **rehypothecation**. When you hold unallocated gold (or, in some cases, gold in a custody arrangement with loose terms), the institution holding it can **lend or pledge that same gold to someone else** as collateral — and that party may pledge it again. The result is that a single physical ounce can sit behind *multiple* claims: yours, the borrower's, the next party's. As long as everyone's claims aren't called at once, it works fine. In a crisis, when everyone wants their metal simultaneously, **there isn't enough physical gold to satisfy all the claims** — and the people holding mere promises discover that "their" gold was working three jobs. Allocated gold is immune to this by definition: a serial-numbered bar that is legally your property cannot be lent out from under you.

![Diagram of the three failure modes default rehypothecation and confiscation that turn gold claims into losses](/imgs/blogs/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk-6.png)

The diagram traces the three ways "your gold isn't your gold" plays out. Each starts from the same root — *you hold a claim, not the metal* — and each ends in the same place: metal you thought you owned, gone. **Default** drops you into the unsecured creditor queue. **Rehypothecation** means many claims chase one ounce. **Confiscation** (next section, and the [state-seizure post](/blog/trading/gold/when-states-take-gold-confiscation-capital-controls-and-the-1933-order)) means the state simply takes or freezes it. And notice the single green node where all three paths terminate: the *fix* is always the same — hold allocated metal, or metal you physically control, so there's no counterparty to default, no claim to over-issue, and (harder, but better) less of a paper trail to seize. The escape from all three failure modes is the same escape: own the metal, not a promise about it.

## Premiums in a panic: why retail premiums blow out exactly when you want gold

Here's a behavior that surprises first-time buyers and that you must understand before you ever need to buy in a hurry: **retail premiums explode in a crisis** — at the precise moment you most want to buy physical metal.

The mechanism is supply-chain, not monetary. The wholesale market (400 oz bars, traded between institutions at spot) stays liquid and deep even in a panic. But the **retail** market — small coins and bars made by mints and refiners — is a manufacturing pipeline with limited throughput. When fear spikes and a flood of buyers all want a one-ounce coin *this week*, three things happen at once: mints can't ramp production fast enough (it takes weeks to turn bars into coins), dealers run out of inventory, and the few sellers left raise premiums to ration what's available. The spot price might barely move, or even *fall* (in a liquidity squeeze, leveraged holders dump paper gold first — see the [crisis-behavior post](/blog/trading/gold/fear-and-the-safe-haven-trade-how-gold-behaves-in-a-crisis)), while the price of an actual coin in your hand *soars*. The screen and the coin shop tell completely different stories.

This is the single most important practical lesson in physical gold: **the time to buy physical metal is when you don't urgently need it.** Premiums are thin and inventory is plentiful in calm markets. If you wait until the crisis to buy your "crisis insurance," you'll pay a panic premium for the privilege — assuming you can find any at all.

There's a second-order effect worth understanding, because it explains *why* the retail and wholesale prices can diverge so violently. The wholesale market and the retail market are linked by a slow, physical conversion process: turning 400 oz Good Delivery bars into one-ounce coins requires shipping the bars to a mint, melting and re-refining to coin-grade alloy, striking, packaging, and distributing — a pipeline measured in weeks, with finite capacity. In calm times that pipeline keeps retail supply ample and premiums thin, tightly arbitraged to spot. In a panic the pipeline can't speed up, but retail *demand* can spike tenfold overnight. With the conversion rate capped and demand unbounded, the only release valve is price: retail premiums shoot up to ration the limited flow of finished product, while the wholesale price (where bars are plentiful) stays anchored. The premium, in other words, is the *real-time scarcity signal of finished physical metal* — and it screams loudest exactly when the most people are trying to buy insurance at the same moment. Knowing this, the sophisticated move is to hold your physical position *before* you need it and treat the panic as a time to do nothing, not a time to scramble into an empty market.

#### Worked example: the March 2020 premium blowout

In the spring of 2020, the spot gold price was around **\$1,600-1,700/oz**. But retail buyers found something else entirely. As the pandemic panic hit, dealers' premiums on common one-ounce coins, normally ~5%, blew out to **10-15% and higher**, and many sold out completely.

- **Calm-market coin** at spot \$1,650 and a 5% premium: \$1,650 × 1.05 = **\$1,733**.
- **Panic-market coin** at the same \$1,650 spot but a 12% premium: \$1,650 × 1.12 = **\$1,848**.

That's **\$115 more per ounce — a 6.6% surcharge — for the same metal**, driven purely by the supply pipeline seizing up, not by the gold price. Spot itself was *roughly flat to lower* in the worst weeks. A buyer who already owned physical metal paid nothing; a buyer scrambling to buy insurance mid-fire paid the panic premium. The lesson lands hard: insurance bought during the fire is always overpriced.

![Gold silver ratio chart showing the March 2020 stress spike to about 125](/imgs/blogs/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk-7.png)

The chart above isn't the premium itself (no clean retail-premium series exists in the data), but it captures the *same panic*: the gold/silver ratio — how many ounces of silver one ounce of gold buys — spiked to around **125 in March 2020**, far above its long-run average near 47, as the entire precious-metals market seized up. That dislocation in the wholesale ratio is the fingerprint of the same liquidity crunch that detached retail coin premiums from spot. When the ratio goes vertical, the coin shop is empty.

## Buying and verifying physical metal without overpaying or getting faked

If the prescription is "own the metal," then *buying* the metal well is the practical skill that follows. Two risks dominate: paying too much (premium and spread) and getting a fake. Both are manageable once you know what to watch.

### Buy from accredited sources, in recognized products

The single best defense against counterfeits and re-assay hassle is to buy **widely recognized products from reputable dealers**: government-minted bullion coins (Eagle, Maple Leaf, Krugerrand, Britannia, Philharmonic) and bars from **LBMA Good Delivery refiners** (PAMP, Valcambi, Argor-Heraeus, the Royal Canadian Mint, and so on). These are the products dealers worldwide buy back on sight. Avoid no-name bars, "deals" far below the going premium (a price *below* the normal premium is a counterfeit flag, not a bargain), and private sellers you can't verify. A reputable dealer publishes both their **buy and sell prices**, so you can see the spread before you commit.

### Verify the metal

Gold has physical properties that are hard to fake all at once. The cheap, non-destructive checks: **weight and dimensions** (gold is extremely dense — 19.3 g/cm³ — so a fake of the right size is almost always the wrong weight, and a fake of the right weight is the wrong size; calipers and a scale catch most). A **magnet** test (gold is not magnetic; a magnetic "gold" item is fake, though non-magnetic doesn't prove real). The **ping** test (gold rings with a distinctive high tone). For higher-value purchases, dealers use **ultrasonic thickness gauges, XRF analyzers** (which read elemental composition non-destructively), and for bars, tamper-evident **assay cards** (CertiPAMP-style sealed packaging with a serial number matching a certificate). The reason recognized coins and assay-carded bars carry a premium is partly that they bundle this verification in — you're paying a little extra to *not* have to prove the metal yourself, now or when you sell.

### Mind the spread, not just the premium

Beginners fixate on the buy premium and ignore the buy-back. The number that actually determines your cost of ownership is the **round-trip spread** — what you pay to buy minus what you'll receive to sell. A dealer advertising a low 3% buy premium but quoting buy-backs at 4% *under* spot has a 7% round-trip; a dealer at a 5% premium who buys back at spot has a 5% round-trip and is cheaper to actually use. Always ask, before buying, *"what will you pay me to buy this exact product back today?"* The gap between that answer and the asking price is your true transaction cost.

#### Worked example: comparing two dealers by round-trip, not headline premium

Spot is **\$2,400/oz**. You're buying a 1 oz coin and want the cheapest *to own*, not the cheapest *to buy*:

- **Dealer A:** sells at a **3.5% premium** = \$2,400 × 1.035 = **\$2,484**. Buys back at **3% under spot** = \$2,400 × 0.97 = **\$2,328**. Round-trip = \$2,484 − \$2,328 = **\$156 (6.3%)**.
- **Dealer B:** sells at a **5% premium** = \$2,400 × 1.05 = **\$2,520**. Buys back at **spot** = **\$2,400**. Round-trip = \$2,520 − \$2,400 = **\$120 (4.8%)**.

Dealer A has the lower *headline* premium and looks cheaper — but Dealer B is **\$36 cheaper to actually own and exit**, because its buy-back is far stronger. The lesson: a dealer's buy-back quote is half the price and the half most people never check. Judge a dealer on the round-trip, and the "expensive" one is often the bargain.

## Common misconceptions

A handful of beliefs about physical gold are not just wrong but *expensively* wrong. Each is corrected with a number.

### "A gold ETF is the same as owning physical gold."

No — an ETF share is a *claim* in a custody chain, not metal in your hand. For tracking the price and trading cheaply, GLD and its peers are excellent (covered in the [ETF post](/blog/trading/gold/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal)). But if your reason for owning gold is *counterparty-free crisis insurance*, an ETF doesn't deliver it: you depend on the trust, the custodian, the authorized participants, and a functioning market to convert your shares back to value. In the exact scenario gold is insurance against — a systemic financial seizure — that chain is under maximum stress. An ETF is a *price* exposure; physical metal is an *ownership* exposure. They are not substitutes for the insurance use case.

### "An unallocated gold account means I own gold."

It means you own a **claim on a bank** denominated in gold. As shown above, in the bank's insolvency you're an unsecured creditor — the [worked example](#worked-example-the-storage-fee-on-allocated-gold-is-the-price-of-bankruptcy-remoteness) put up to \$60,000 of a \$100,000 holding at risk in a default, versus \$0 for allocated. You bought gold to escape bank risk and then handed your gold to a bank as a loan. The fact that it's "free" is the market screaming at you that nothing is set aside.

### "Numismatic (rare/collectible) coins are a great store of value."

Numismatic coins are priced for their **collectible rarity**, not their gold content — and that premium can be **30%, 50%, or several hundred percent** over melt value. You're buying a thin, illiquid collectibles market on top of the metal. If the numismatic premium compresses (collector fashions change; "rare" turns out to be less rare), you can lose most of that premium even as gold rises. For a *store of monetary value*, you want the **lowest premium over melt** — plain bullion bars and common coins — not numismatics. Collectible coins are a hobby that happens to contain gold, not gold insurance.

### "Bigger premium means better / more 'real' gold."

Premium reflects **fabrication cost, dealer margin, and scarcity**, not gold quality. A 999.9 kilobar at a 1.5% premium and a 999.9 one-ounce coin at a 5% premium contain identical-purity metal. You're paying the extra premium for *divisibility and recognizability*, not for "better" gold. Paying up only makes sense when you actually need those features.

### "Gold in a bank safe-deposit box is insured like my deposits."

It is **not**. FDIC (and equivalent) insurance covers bank *deposits*, not the contents of a box. If the bank is robbed, floods, or fails, your gold is generally uninsured unless you separately arranged a policy. The box is a storage location, not an insured account.

## How it shows up in real markets

Theory is cheap. Here are the episodes where the form of ownership turned out to be the whole story.

### The 2020 retail premium blowout

We did the math above; here's the market texture. Through March-April 2020, the [gold spot price](/blog/trading/gold/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce) was volatile but roughly in the \$1,500-1,700 range, even dipping in the worst liquidity-squeeze days. Yet retail buyers couldn't find coins, and where they could, premiums on Eagles and Maple Leafs ran 10-20% over spot. Major mints suspended or rationed production; some dealers stopped quoting buy-backs entirely. The wholesale market never broke — institutions traded 400 oz bars near spot the whole time — but the *retail* pipeline seized. People who'd planned to "buy gold if things got bad" learned that the metal isn't a faucet you turn on during the fire. The annual chart below shows that 2020 was the year gold's *average* price first pushed decisively above \$1,700 on its way higher; the panic was a spike inside a structural bull market, which is exactly when retail demand floods in.

![Gold spot price 2015 to 2026 the wholesale base under every retail premium](/imgs/blogs/physical-gold-bars-coins-allocated-vs-unallocated-and-counterparty-risk-4.png)

### Bank gold-account episodes and the unallocated lesson

History is littered with cases where "gold accounts" turned out to be claims, not metal. The recurring pattern: an institution offers cheap or free "gold storage," runs it as unallocated (or quietly rehypothecates allocated metal under loose terms), and when stress hits, customers discover their gold is a liability in a queue. The clearest *structural* version is what bullion banks do every day — the London market runs largely on **unallocated** balances precisely because they're efficient and cheap; the system works because not everyone demands physical conversion at once. But "works until everyone wants out at once" is the definition of a run, and gold's whole job is to protect you from runs. This is why central banks — the most sophisticated gold holders on earth — overwhelmingly hold **allocated** metal in their own or trusted vaults, often demanding physical repatriation (Germany's Bundesbank famously brought home hundreds of tonnes from New York and Paris in the 2010s). They understand the apolitical-asset logic better than anyone: the point of gold is that it's no one's liability, so they refuse to hold it *as* a liability. That logic — gold as the only reserve asset that's no one's promise — is the spine of the [apolitical-asset](/blog/trading/gold/the-apolitical-asset-why-central-banks-trust-gold-over-each-others-money) and [sanctions](/blog/trading/gold/sanctions-reserve-freezes-and-the-2022-turning-point-for-gold) posts in this series.

### Confiscation and capital controls — the political failure mode

The third failure mode in our diagram is not market risk but *political* risk, and it's the one people forget. In **1933**, US Executive Order 6102 required citizens to hand in their gold to the government at a fixed price, after which the official price was raised — an effective confiscation-and-devaluation. India has repeatedly used **import duties and curbs** to throttle private gold demand. Various countries impose reporting requirements or capital controls that bite on gold. The lesson for *form*: gold held through an institution — an account, a domestic vault, a box at a bank — is the *easiest* to freeze or seize, because there's a record and a chokepoint. Metal you physically hold is harder for a state to reach, though not impossible. This is the genuinely hard part of the "own the metal" prescription, and it's why the [confiscation post](/blog/trading/gold/when-states-take-gold-confiscation-capital-controls-and-the-1933-order) treats jurisdiction and form as a single linked decision. You can't fully escape political risk, but you can avoid *handing the state a switch* by holding your insurance as a claim it can flip off.

#### Worked example: what the form cost two investors in a true crisis

Take the two investors from the opening, both holding \$50,000 of "gold" into a severe banking crisis where their bank fails and recovers 40% for unsecured creditors:

- **Investor A — unallocated bank gold account.** "Owns" \$50,000 of gold as a claim on the bank. In the failure, recovers 40% = **\$20,000**. The gold "tracked spot perfectly" right up until the bank closed — then it was just a number in a queue. Net: **\$30,000 of insurance evaporated in the one event it was meant for.**
- **Investor B — allocated metal (or coins at home).** Owns \$50,000 of actual gold. The bank's failure is irrelevant to her bars; if anything, gold's *price* rises in the crisis. She still holds 100% of the metal, now worth *more*. Net: **\$0 lost to the failure, full insurance intact.**

Same dollar amount, same gold price, opposite outcomes — entirely because of *form*. That \$30,000 gap is the cash value of understanding allocated vs unallocated. The insurance only pays out if you held it in a form the crisis can't touch.

## The takeaway: how to actually hold gold you control

Step back and the whole post collapses into one principle: **the form of ownership determines whether you own insurance or just a promise that behaves like insurance until you need it.** Gold's entire reason for existing — across five thousand years, every empire, every failed currency — is that it is *no one's liability*. The instant you hold it through a counterparty, you hand that superpower back. So the practical playbook follows directly from that one idea:

- **Default to metal you control.** For the core of a gold holding meant as crisis insurance, own physical bullion outright — allocated in a reputable, audited, insured depository for the bulk (you get bankruptcy-remoteness *and* insurance for ~0.1-0.5%/year), and a modest stack of recognizable common coins where you can physically reach them. That split covers both "systemic failure" and "I can't get to a vault" scenarios.
- **Treat "free" as a warning, not a feature.** A gold account with no storage fee is telling you, in the price, that no specific metal is set aside for you — you're a creditor, not an owner. If you use unallocated balances for liquidity or trading, *know* that's what they are, and don't mistake them for insurance.
- **Buy the biggest unit you'd sell whole, and buy when it's quiet.** Bigger bars carry thinner premiums; the round-trip spread (~6% on a small coin) means physical gold is a multi-year hold, not a trade. And premiums blow out in panics, so the time to buy your fire insurance is before the fire.
- **Know your three failure modes — default, rehypothecation, confiscation — and that all three have the same fix.** Own the metal, hold it where it's genuinely yours, and think about jurisdiction the way you think about purity.

The deepest point is the one the opening promised. Most of finance is about *return* — what an asset earns you. Gold isn't; it pays no yield, and as the [no-yield post](/blog/trading/gold/the-no-yield-problem-how-a-metal-that-pays-nothing-can-be-worth-anything) argues, that's the feature, not the bug. Gold is about *survival of value when the system fails* — and an insurance policy is only worth what it pays out *in the disaster*. An unallocated gold account, an ETF, a futures contract — these track the gold *price* beautifully in normal times and may fail you in precisely the systemic seizure they were bought to insure against, because each is a promise that depends on the system holding together. Physical metal you control is the one form whose payout doesn't depend on anyone keeping their word. That's why how you hold gold isn't a detail. **It's the difference between owning insurance and owning a promise — and you only find out which one you had on the day you need it.**

## Further reading & cross-links

- [Gold ETFs and the GLD machine: how paper gold tracks the metal](/blog/trading/gold/gold-etfs-and-the-gld-machine-how-paper-gold-tracks-the-metal) — the next rung down the ladder: the custody chain behind a gold ETF, why it tracks spot, and the trust-the-custodian question.
- [When states take gold: confiscation, capital controls, and the 1933 order](/blog/trading/gold/when-states-take-gold-confiscation-capital-controls-and-the-1933-order) — the political failure mode in depth: how form and jurisdiction together decide whether your gold can be reached.
- [How gold is priced: spot, the London fix, futures, and the troy ounce](/blog/trading/gold/how-gold-is-priced-spot-the-london-fix-futures-and-the-troy-ounce) — the wholesale price that every retail premium sits on top of, and the paper-vs-physical link.
- [Fear and the safe-haven trade: how gold behaves in a crisis](/blog/trading/gold/fear-and-the-safe-haven-trade-how-gold-behaves-in-a-crisis) — why spot can fall while coin premiums soar, and the liquidity-squeeze exception.
- [The apolitical asset: why central banks trust gold over each other's money](/blog/trading/gold/the-apolitical-asset-why-central-banks-trust-gold-over-each-others-money) — the "no one's liability" logic that makes the world's smartest gold holders insist on allocated metal.
- [Gold: money, insurance, or just a rock?](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock) — the portfolio-allocation framing for *how much* gold to hold, from the cross-asset series.
