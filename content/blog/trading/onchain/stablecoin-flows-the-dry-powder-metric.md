---
title: "Stablecoin Flows: Reading the Market's Dry Powder"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Stablecoins are the cash leg of crypto. Learn to read total supply, mints and burns, the USDT/USDC split, and exchange stablecoin reserves as the on-chain gauge of buying power waiting on the sidelines."
tags: ["onchain", "crypto", "stablecoins", "usdt", "usdc", "tether", "circle", "liquidity", "tron", "defillama", "glassnode", "depeg"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Stablecoins are tokenized dollars, and their total supply is the on-chain gauge of *dry powder*: cash sitting inside crypto, ready to be spent on assets. Growing supply means buying power is entering the system; shrinking supply means it is leaving.
>
> - **What the signal is:** total stablecoin supply, issuer mints and burns, the USDT-vs-USDC split, and how much of that cash is parked on exchanges. Together they measure the buy-side ammunition available to the market.
> - **How to read it:** track total supply on DefiLlama or Glassnode (the slow trend), watch big mints and burns in near real time (the fast flow), and pair exchange stablecoin reserves with the exchange-flow signal you already know.
> - **What you do with it:** rising supply plus rising exchange reserves is a tailwind for risk; falling supply (net redemptions) is a headwind. It is a *liquidity* read, not a timing trigger — correlation, never a guarantee.
> - **The number to remember:** total stablecoin supply went from roughly \$28B in 2020 to roughly \$230B in 2025. That \$230B is the cash float of crypto.

On the weekend of **11 March 2023**, the second-largest stablecoin in the world stopped being worth a dollar. Circle, the issuer of USDC, had disclosed that roughly \$3.3B of its cash reserves were stuck at the collapsing Silicon Valley Bank. The market did the arithmetic in seconds: if those reserves were gone, USDC was not fully backed. USDC, a token whose entire job is to equal exactly one dollar, traded down to about **\$0.88** on exchanges — a 12% gap on a "safe" asset. Traders who held USDC scrambled to swap into USDT or into actual dollars; the ones who could not redeem over the weekend simply watched their "cash" position bleed.

The reason the price could gap so violently is worth pausing on, because it teaches the whole mechanism. USDC's peg holds in normal times because anyone can redeem a token for a dollar with Circle — so if it trades at \$0.99, arbitrageurs buy it and redeem it for \$1.00, instantly closing the gap. But redemptions happen through banks, and over a weekend, with a bank failing and the actual reserve dollars in question, the redemption channel was effectively *frozen*. With no working way to redeem, the arbitrage that normally pins the price could not operate, so the only price was whatever panicked sellers and brave buyers agreed on in the open market — and that price was \$0.88. The peg is not magic; it is arbitrage, and arbitrage needs a functioning redemption door. Close the door, and the peg is only as good as the market's nerve.

Then, on Sunday evening, US regulators announced that all SVB depositors would be made whole. The reserve dollars were safe, the redemption door would reopen, and the arbitrage that pins the peg could function again. By Monday, USDC was back at \$1.00. The depeg was real, terrifying, and — because the collateral turned out to be recoverable — *temporary*. Contrast that with **TerraUSD (UST)** ten months earlier, which fell from \$1.00 toward \$0.10 and never came back, because nothing real was holding it up. Same word, "depeg"; opposite outcomes. The difference was what sat behind the token.

That weekend is the perfect doorway into this post, because it shows the two things you must understand about stablecoins: they are the **cash leg** of the entire crypto market — the dollars parked inside the system waiting to buy something — and the *supply* of that cash, who is minting it, who is redeeming it, and where it is sitting, is one of the most-watched on-chain signals there is. When the cash pile grows, there is fuel for a rally. When it shrinks, buying power is walking out the door.

![Stablecoins as cash on the sideline ready to buy](/imgs/blogs/stablecoin-flows-the-dry-powder-metric-1.png)

## Foundations: what a stablecoin actually is

Let us build this from absolute zero, because the whole post rests on it.

A **stablecoin** is a token that is designed to always be worth one US dollar (some track the euro or gold, but dollar stablecoins dominate, so we will say "dollar" throughout). It lives on a blockchain like any other token — if you have read the companion post on [tokens, on-chain transfers and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals), a stablecoin is exactly that: a smart contract holding a ledger of `address → balance`, where each unit is meant to be redeemable for \$1. The token is not magic and it is not money in the legal sense. It is an **IOU**: a claim on a real dollar held somewhere by the company that issued it.

The two giants you need to know:

- **USDT (Tether)** — the oldest and largest. Issued by Tether Limited, an offshore company. It is the dollar of *offshore* crypto: the default trading pair on most non-US exchanges, the dominant rail in Asia, and — crucially — the king of the cheap **Tron** network used by retail and remittances worldwide. Roughly **\$160B** outstanding by 2025.
- **USDC (Circle)** — the regulated US challenger. Issued by Circle, a US company, with monthly attestations and reserves held mostly in short-term US Treasuries and cash at regulated banks. It is the dollar of *US institutions* and on-chain DeFi. Roughly **\$60B** outstanding by 2025.

Both run on multiple chains: **Ethereum** (the home of DeFi and institutional flow), **Tron** (cheap fees, the retail and remittance workhorse, USDT-dominated), **Solana** (fast, cheap, the memecoin and trading venue), and a dozen others. The *same* logical dollar exists as separate token contracts on each chain. A USDT balance on Ethereum and a USDT balance on Tron are both "Tether dollars," but they are *different tokens on different ledgers* — moving value from one to the other means burning on one chain and minting on the other (or routing through a bridge), which is why cross-chain stablecoin movement is its own subject. When you read "USDT supply," you are reading the *sum* across all those chains, and the per-chain breakdown is itself informative: a chain whose stablecoin balance is swelling is a chain where cash is gathering.

### What actually backs the token: the reserves

The word "stablecoin" hides the single most important question: *what is the dollar made of?* A stablecoin is only as good as the assets the issuer holds against it. There are three broad designs, and they are not equally safe:

- **Fiat-backed (the dominant, sane kind).** Each token is matched by real dollar-equivalent reserves — cash in banks plus short-term US Treasury bills. USDT and USDC are both fiat-backed. The redeemability that holds the peg works because the reserves genuinely exist; an arbitrageur who buys a token at \$0.99 and redeems it for \$1.00 can only do that if the issuer actually has the dollar to pay. Circle publishes monthly attestations of USDC's reserves; Tether publishes quarterly reports. These are *attestations*, not full audits, and the distinction matters — an attestation is a snapshot a third party confirms, not a continuous guarantee — but it is far more than an algorithmic coin offers.
- **Crypto-collateralized (over-collateralized).** Some stablecoins, like DAI, are backed not by dollars but by a *surplus* of volatile crypto locked in smart contracts — say \$150 of ETH locked to mint \$100 of stablecoin, so a price drop still leaves the coin covered. These are transparent (the collateral is on-chain and inspectable) but exposed to the volatility of their backing.
- **Algorithmic (the dangerous kind).** No real reserve at all — the peg is "defended" purely by code and a sister token. This is what UST was, and it is why UST went to zero. An algorithmic stablecoin's "supply" is the least real form of dry powder there is.

For flow-reading, the takeaway is that you should weight supply by *quality*. A billion dollars of attested, fiat-backed supply is a billion dollars of genuine, redeemable ammunition. A billion dollars of algorithmic supply is an IOU from a machine, and it can evaporate in a weekend. Throughout this post, when we say "dry powder," we mean the real, fiat-backed kind — the USDT and USDC float — not the reflexive, self-referential kind that the Terra collapse exposed.

### Why issuers do this: the carry trade

It helps to understand *why* Tether and Circle exist, because their incentive explains the supply dynamics you will read. The issuer's business is gloriously simple: take in a dollar, issue a token that pays the holder 0%, and invest the dollar in Treasury bills paying 5%. The issuer keeps the spread. With \$160B of reserves at a 5% yield, Tether earns on the order of *billions* of dollars a year in interest — a famously profitable operation for a company with a small headcount. This carry-trade incentive is why issuers will happily mint as much as the market demands: every new token is another dollar of float they get to earn yield on. It also explains the rate sensitivity of supply — when T-bill yields are high, holding a 0%-yielding stablecoin is a worse deal *for the holder*, which is part of why supply contracts in high-rate regimes even as it stays wildly profitable for the issuer.

### Issuance (minting) and redemption (burning)

Here is the mechanic that the entire "dry powder" idea depends on. A stablecoin is created and destroyed on demand:

- **Minting (issuance):** someone — usually a large trading firm or exchange — wires real dollars to the issuer. The issuer holds those dollars in reserve and instructs the token contract to **mint** an equal number of new tokens to the depositor's address. \$100M in → 100M new tokens out. Supply goes *up*.
- **Burning (redemption):** the reverse. A holder sends tokens back to the issuer; the contract **burns** (destroys) them; the issuer wires the matching dollars back out. 100M tokens in → \$100M out, and the tokens cease to exist. Supply goes *down*.

![Minting deposits dollars to create tokens and burning destroys tokens to return dollars](/imgs/blogs/stablecoin-flows-the-dry-powder-metric-2.png)

This is why supply is a *flow* signal and not just a static number. Every token in existence was minted because someone, at some point, decided to convert real dollars into on-chain dollars — that is, decided to bring cash *into* crypto. And every burn is someone deciding to take cash *out*. The peg holds in between because anyone (in size) can redeem a token for \$1, so if the token ever trades below a dollar, arbitrageurs buy it cheap and redeem it for full value, dragging the price back to par. The redeemability *is* the peg.

#### Worked example: minting as cash entering the system

A market maker preparing for a busy week wires **\$1B** to Tether and receives **1B fresh USDT** minted to its address. Nothing about Bitcoin's price changed at the moment of the mint — but the system now contains \$1B of on-chain dollars that did not exist yesterday, and that firm minted them *because it intends to deploy them*. If even half of that \$1B becomes buy orders for BTC and ETH over the next week, that is \$500M of fresh demand hitting the order books. **The intuition: a large mint is a firm pre-loading its ammunition — new buying power has entered the system, even before a single trade prints.**

### Market cap of a stablecoin = dollars parked in crypto

One more reframing and the foundations are complete. The "market cap" of a stablecoin is just `supply × \$1` — and because the price is pinned to a dollar, the market cap *is* the supply. So when you read that the total stablecoin market cap is \$230B, do not read it as a speculative valuation the way you would a normal token's market cap. Read it literally: **\$230B of real dollars have been converted into on-chain dollars and are sitting inside the crypto system right now.** That is the cash float of the entire market. That is the dry powder.

## The dry-powder gauge: total stablecoin supply

The single most important reading is the **total outstanding supply across all stablecoins**. It is slow-moving, it is unambiguous, and it answers one question: *is cash entering or leaving crypto?*

![Total stablecoin supply rising from about 28 billion to about 230 billion dollars](/imgs/blogs/stablecoin-flows-the-dry-powder-metric-3.png)

Look at the shape of that trend. From roughly **\$28B at the end of 2020**, supply exploded to about **\$140B by the end of 2021** as the bull market sucked in cash. Then something telling happened: through 2022 and into 2023, total supply *contracted* — from \$140B down toward \$130B — as the bear market, the Terra collapse, the FTX failure, and rising US interest rates pulled cash back out (net redemptions). Then from late 2023 onward it re-expanded hard, blowing past the old high to roughly **\$200B by end-2024 and \$230B by 2025**.

The mechanism behind the contraction is worth dwelling on, because it surprises beginners. When US interest rates rose to 5%+, holding a stablecoin became *expensive* in opportunity-cost terms: a token paying you 0% while a US Treasury bill pays 5% is a bad deal, so capital redeemed stablecoins, took the real dollars, and bought T-bills directly. Supply shrank not because crypto was dying but because the *cash* found a better home. When you read a falling-supply chart, ask *why* it is falling — risk-off flight, or just rate-driven cash management — because the trading implication differs.

#### Worked example: supply growth as fresh dry powder

Say total stablecoin supply climbs from **\$140B to \$200B** over a year. That \$60B did not appear from nothing — it was minted, which means roughly \$60B of real dollars were wired to issuers and converted into on-chain dollars over that period. That is **\$60B of new buying power** that entered the system and is now sitting in wallets and on exchanges. It does not *have* to be spent on BTC — some of it earns yield in DeFi, some just parks — but the ammunition is loaded. **The intuition: a \$60B increase in supply is \$60B of cash that chose to come into crypto, and cash that came in to sit usually came in to buy.**

How to read it in practice: open **DefiLlama's stablecoins dashboard** (or Glassnode's "Stablecoin Supply" metric) and look at the *aggregate* line first, then the trend over months. You are not day-trading this. You are answering: are we in a regime where dry powder is accumulating (supportive of risk) or draining (a headwind)? A multi-month uptrend in total supply has historically coincided with — not caused, *coincided with* — strength in crypto prices, because growing cash-on-the-sideline is the raw material a rally is built from.

### A dated episode: the 2022 drain

The cleanest way to feel the signal is to walk through the great drain of 2022. Coming out of the November 2021 top, total stablecoin supply sat near \$140B. Then the year unfolded as a slow-motion liquidity withdrawal that the supply curve recorded faithfully. In May, **Terra/UST collapsed**, instantly destroying the roughly \$18B of (mostly fake) UST supply and shattering confidence across the whole stablecoin category — holders of *other* coins redeemed in fear, and the aggregate curve dipped. Through the summer, the Three Arrows Capital and Celsius failures forced more deleveraging and more redemptions. Then in **November 2022, FTX collapsed**, and the on-chain record of that week is a master class in flow: enormous stablecoin movement *onto* exchanges as users tried to get cash positioned to withdraw, frantic swaps between coins, and a wave of net burns as real dollars exited the system entirely. Total supply kept grinding lower into 2023, bottoming near \$130B. The supply curve did not *predict* any single event, but it *recorded the regime*: a market from which cash was steadily, visibly leaving. A trader watching that line through 2022 had an unambiguous, lag-free read that the liquidity tide was going out — which is exactly the kind of slow, structural signal stablecoin supply is best at.

The mirror image is the 2024–2025 re-expansion: as the spot Bitcoin ETFs launched and rates began to ease, supply blew past its old high toward \$230B. The cash that had fled in 2022 was coming back, and the supply curve registered the inflow before most narrative-driven commentary caught up. That is the edge: supply is a *measured* quantity off the public ledger, not a survey or an opinion. It does not get revised, it does not lag, and it does not lie about whether cash is entering or leaving — though, as we will see, it can be *misread*.

## Mints and burns: the real-time flow

Total supply is the slow trend. **Mints and burns** are the fast flow — the day-to-day, even minute-to-minute, additions and subtractions that *make up* the trend. This is where on-chain analysis earns its lead time.

Every mint and burn is a public on-chain transaction. When Tether mints \$1B of USDT, that is a single, visible, timestamped event on Ethereum or Tron. Services like **Whale Alert** broadcast these in real time; you can also watch the issuer's treasury contract directly on Etherscan or Tronscan. So you get to see, as it happens, when issuers are expanding or contracting the dollar supply.

The folklore — and you must handle this carefully — is that **big USDT mints often precede rallies**. There is a real, observable tendency for large mints to cluster ahead of or during strong price moves. But the causation is the opposite of what naive readers assume. Issuers do not mint tokens to "pump" the market; they mint **on demand** because a large buyer wired them dollars and *wants* tokens to deploy. The mint is a *symptom* of incoming demand, not its cause. The buyer already decided to buy; the mint is the paperwork. So a cluster of large mints is genuinely informative — it tells you serious money is loading up — but it is a coincident-to-leading indicator of demand that already exists, not a magic "money printer" that levitates prices on its own.

#### Worked example: a \$1B USDT mint as a buy-pressure signal

You see a Whale Alert: **\$1B USDT minted** to a known market-maker address, then bridged to an exchange wallet over the next two hours. Read it like this: a firm converted \$1B of real dollars into on-chain dollars and moved them to a venue where the *only* thing you do with stablecoins is buy crypto. That is up to **\$1B of potential bids** now sitting on the book's cash side. It does not guarantee a rally — they might be providing liquidity, hedging, or sitting — but the base rate says fresh stable deposited to an exchange skews toward buying. **The intuition: a \$1B mint that lands on an exchange is \$1B of ammunition moved to the firing line; it raises the odds of buying, it does not promise it.**

The flip side is just as important and far less discussed: **burns**. When you see large, repeated *burns* — tokens flowing back to the issuer and being destroyed — that is cash *leaving*. A wave of redemptions during a sell-off means holders are not just rotating into other coins; they are taking real dollars off the table entirely. Burns during stress are a sign of genuine deleveraging, not rotation.

There is one nuance that trips up beginners watching Tether's treasury: **"authorized but not issued."** Tether sometimes pre-mints a large batch of USDT into its own treasury wallet *before* it is sold to anyone — this is an inventory operation, like a printer running off banknotes that sit in the vault, not yet in circulation. Those pre-minted tokens are *not* dry powder until they leave the treasury and reach a buyer. So when you see a headline "\$1B USDT minted," the discerning read is: *did it leave the treasury?* A mint that sits in Tether's own inventory wallet is the company restocking the shelf; a mint that moves *out* to a market maker or an exchange is real demand being satisfied. Always trace the *next hop*. Tronscan and Etherscan let you do exactly this — click the recipient address, see whether it is the labeled Tether treasury (inventory) or an exchange hot wallet (deployment).

This is also where you separate signal from theater. Some commentators breathlessly report every treasury pre-mint as "the printer is on," which is misleading because inventory restocking is not new buying power. The honest reader watches *net issued to the market*: tokens that have actually left the issuer's hands and reached someone who intends to use them. That distinction — printed-and-shelved versus printed-and-sold — is the difference between noise and the real flow.

#### Worked example: tracing a mint to tell inventory from demand

You see two \$500M USDT mints in one day. You open the recipient of each on Tronscan. The first lands in Tether's labeled treasury wallet and just *sits* there — that is **\$500M of inventory**, shelved, not yet anyone's buying power. The second moves within an hour to a Binance hot wallet — that is **\$500M of real demand** that has reached a venue where the only use is buying crypto. Same headline ("\$1B minted today"), but only half of it is live dry powder. If you had read the headline alone, you would have double-counted the ammunition by 2×. **The lesson: a mint is only dry powder once it leaves the issuer's vault — trace the next hop before you size the signal.**

## The USDT vs USDC split: what the mix signals

Total supply tells you *how much* dry powder exists. The **split between USDT and USDC** tells you *who* is holding it and *where* — and that is its own signal.

![USDT and USDC supply compared as grouped bars by year](/imgs/blogs/stablecoin-flows-the-dry-powder-metric-4.png)

The two coins are not interchangeable in what they represent:

- **USDT** is the dollar of the *offshore, retail, Asia-centric, Tron-heavy* world. It is the default pair on Binance, OKX, Bybit and the rest of the non-US exchange universe; it is the remittance and savings rail across emerging markets; it is what a trader in Lagos, Manila, or Istanbul actually holds. USDT supply growing tells you the *global retail and offshore* dollar pool is expanding.
- **USDC** is the dollar of *US institutions and on-chain DeFi*. It is regulated, attested monthly, and favored by US-domiciled funds, by Coinbase, and by DeFi protocols that want a "clean" dollar. USDC supply growing tells you *US and institutional* on-chain dollar demand is expanding.

Watch the divergence. After the SVB weekend in March 2023, USDC supply **nearly halved** over the following months — from about \$44B toward \$24B — while USDT *grew*. That was a visible flight from the US-regulated dollar to the offshore one, driven by the depeg scare and by US regulatory pressure on Circle's banking partners. The split moved, and it told a story about *where* trust and capital were concentrating that the aggregate number alone would have hidden.

#### Worked example: a \$10B USDC redemption as buying power exiting

Suppose USDC supply falls from **\$44B to \$34B** in a quarter — a **\$10B** net redemption. That is \$10B of on-chain dollars destroyed and \$10B of real dollars wired back out to bank accounts. If that cash had stayed, it was potential buy-side fuel; now it is gone from the system entirely. Even if USDT grew by \$10B over the same period (so *total* supply was flat), the *composition* shift matters: the US-institutional pool shrank while the offshore-retail pool grew, which is a different market with different behavior. **The intuition: a \$10B USDC burn is \$10B of specifically-institutional dry powder leaving — read the mix, not just the total.**

The practical read: track USDT supply and USDC supply as two separate lines (DefiLlama lets you isolate each issuer). When they move *together*, it is a clean liquidity signal. When they *diverge* — one minting while the other burns — something structural is happening (a regulatory shock, a trust event, a geographic rotation), and the divergence itself is the thing to investigate.

There is also a *dominance* lens. USDT's share of total stablecoin supply has trended steadily upward over the cycle — from roughly half the market in the early days toward roughly two-thirds by 2025. A rising USDT dominance tells you the *offshore, retail, global* dollar is the one growing fastest, which carries its own market-character implications: that pool is more retail-driven, more leverage-prone, and more concentrated in the Asian trading session. A falling USDT dominance (USDC catching up) would signal the opposite — a re-institutionalization of the on-chain dollar, more US-regulated capital coming back. After the SVB scare, USDT dominance jumped precisely because USDC's pool shrank, and it has stayed elevated since. Reading dominance is reading *the character of the marginal dollar*, not just its quantity. It is the stablecoin analogue of watching whether retail or institutions are the marginal buyer in equities.

## Exchange stablecoin reserves: ammunition on the venue

Here is the signal that ties stablecoins directly to price action. Supply tells you how much dry powder *exists*. **Exchange stablecoin reserves** tell you how much of it is *parked on a trading venue, ready to fire*.

![Dry powder moves to an exchange and becomes buy pressure when a holder buys an asset](/imgs/blogs/stablecoin-flows-the-dry-powder-metric-5.png)

Stablecoins sitting in a cold wallet, locked in a DeFi lending pool, or held by a treasury are *not* immediately deployable as bids on Binance. Stablecoins sitting in an exchange wallet **are**. So the amount of USDT and USDC held on exchanges is the most direct proxy for *imminent* buying power — it is the cash that is one click away from becoming a buy order. This is the mirror image of the signal in the companion post on [exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows). There, BTC *flowing onto* an exchange is read as potential *selling* pressure (coins moved to a venue are coins positioned to be sold). Stablecoins flowing onto an exchange are the opposite: potential *buying* pressure. Cash on the venue wants to buy; the asset on the venue wants to sell.

So you read the two together. The cleanest bullish micro-setup on-chain is **stablecoins flowing in while the asset flows out**: cash arriving to buy, coins leaving to be held off-exchange. The cleanest bearish one is the reverse: the asset piling onto exchanges while the stablecoin reserve drains.

#### Worked example: \$30B of stablecoins on exchanges as ready bids

Glassnode shows roughly **\$30B of USDT and USDC** sitting across major exchange wallets. That is \$30B of cash already inside the venues, already past the slow bank-wire and minting step, sitting on the dollar side of the order book. It is not *committed* to buying — some is collateral, some is just parked — but it is the pool from which the next wave of bids will be drawn. If exchange stablecoin reserves *rise* by \$5B over a week while prices are flat, you have \$5B more loaded ammunition than you did seven days ago, and a market that has not yet spent it. **The intuition: \$30B of stablecoins on exchanges is the upper bound on near-term spot buying power — watch whether that pile is filling up or draining.**

A caution that keeps you honest: a rising exchange stablecoin balance is *necessary* but not *sufficient* for a rally. Ammunition on the firing line does not pull the trigger. Plenty of cash has sat on exchanges through grinding sideways markets. The signal raises the *odds* and sizes the *potential*; it does not schedule the move. Treat it as a measure of fuel, never as an ignition.

There is a refinement that experienced readers add: the **stablecoin ratio** or "buy-side dominance." Instead of looking at the raw stablecoin balance, you look at stablecoin reserves *relative to* the coin reserves on the same venue — roughly, how much cash is parked versus how much asset is parked. When the ratio of stablecoins to coins on exchanges is high, there is proportionally more cash sitting ready to buy than asset sitting ready to sell, which is a structurally supportive setup. When it is low, the venue is asset-heavy and cash-light. This ratio is more robust than the raw balance because it normalizes for the overall size of the venue and for price moves — \$30B of stablecoins means something different against \$20B of coins than against \$200B of coins.

A second refinement is *which* chain the exchange-bound stablecoins are arriving on. Stablecoins flooding onto Solana exchange wallets is often a tell for memecoin and high-velocity speculation; stablecoins gathering on Ethereum venues skews toward larger, slower institutional positioning; USDT massing on Tron-linked exchange wallets is frequently the Asian retail on-ramp warming up. The *destination chain* of the dry powder hints at *what kind* of buying it will become. You will not always be right about it, but reading the chain mix alongside the total adds a layer of texture the aggregate number alone cannot.

#### Worked example: the stablecoin ratio as a structural read

Two exchanges each hold \$30B of stablecoins. On Exchange A, coins on the venue are worth \$20B, so the cash-to-coin ratio is `30 / 20 = 1.5` — cash-heavy, structurally primed to buy. On Exchange B, coins on the venue are worth \$120B, so the ratio is `30 / 120 = 0.25` — asset-heavy, more supply than demand sitting ready. The same \$30B of dry powder reads very differently against the two backdrops: on A it is a large fraction of the venue's firepower, on B it is a rounding error against the asset overhang. **The lesson: size the ammunition against the target — \$30B of cash is a cocked hammer on a cash-light venue and a footnote on an asset-heavy one.**

## The Tron-USDT rail: cheap rails carry everything

You cannot understand stablecoin flows without understanding **Tron**, and you cannot understand Tron honestly without the defender's lens.

USDT on Tron (the TRC-20 version) is, by some measures, the single largest pool of stablecoin value on any chain. The reason is brutally simple: **fees and speed**. Sending USDT on Ethereum can cost several dollars in gas when the network is busy; sending the same USDT on Tron costs roughly a dollar or less and settles in seconds. For someone moving \$200 of savings, a \$5 fee is a 2.5% tax and a \$0.50 fee is not. So Tron became the world's retail-dollar rail: remittances from migrant workers, dollar savings for people in countries with collapsing local currencies and no access to a US bank account, and the on/off-ramp of choice for Asian retail trading.

![USDT on the Tron rail serves remittances and retail but also attracts illicit flow](/imgs/blogs/stablecoin-flows-the-dry-powder-metric-6.png)

Now the honest part. The *same* properties that make Tron USDT great for a remittance — cheap, fast, dollar-pegged, globally liquid — also make it attractive to fraud and laundering. Pig-butchering scam proceeds, sanction-evasion attempts, and laundering hops frequently route through TRC-20 USDT, and reputable on-chain analytics firms have repeatedly flagged it as a dominant rail for illicit stablecoin flow. This is not a reason to avoid the topic; it is the reason analysts *study* it. The defender's read is straightforward: because every TRC-20 transfer is public and permanent, investigators cluster scam-linked addresses, trace peel chains across fresh wallets, and flag known-bad addresses — and, importantly, **Tether can freeze tokens**. The issuer holds a centralized freeze function and has frozen hundreds of millions of dollars of USDT tied to thefts and sanctions at the request of law enforcement. That freezability is a double-edged sword (it means USDT is not censorship-resistant the way Bitcoin is), but for a defender it is a lever that does not exist for native crypto.

It is worth pausing on *why* the retail rail matters so much, because it reframes what stablecoins are. For hundreds of millions of people, a dollar stablecoin is not a trading instrument at all — it is the first **dollar bank account** they have ever had. In a country with a currency losing 30% of its value a year and a banking system that will not give an ordinary person dollar access, a USDT balance on a phone is a genuine financial lifeline: a store of value, a way to receive payment for remote work, a way to send money home. That use case has nothing to do with crypto speculation, and it is a large and growing share of total stablecoin demand. The implication for the flow-reader is that *not all stablecoin supply is trading dry powder* — a meaningful chunk is savings and payments that will never become a bid on Binance. The supply number is real cash in the system, but its *velocity toward the order book* varies a lot by who holds it and why.

#### Worked example: why the rail wins on a remittance

A worker sends **\$200** home each month. Through a traditional remittance service charging a 6.5% fee, that costs about **\$13** and can take days. Through USDT on Tron, the network fee is roughly **\$1** and it settles in seconds — a saving of about **\$12 per transfer**, or roughly **\$144 a year** on a \$2,400 annual flow. Multiply that across tens of millions of senders and you see why the rail captured the world's retail-dollar traffic: the fee differential is not a rounding error to someone moving \$200, it is real money. **The lesson: the same cheap rail that makes \$1 transfers viable for remittances is what makes Tron the dominant USDT chain — and most of that flow is payments, not trading ammunition.**

For the flow-reader, the takeaway is calibration: a huge share of *raw* USDT transfer volume on Tron is small-value retail and remittance traffic, not trading dry powder. When you see "USDT on Tron" transaction counts, do not mistake remittance churn for buy-side intent. The dry-powder signal lives in *exchange* stablecoin balances and in *mints to trading firms*, not in the total count of \$50 transfers between retail wallets.

The freeze function deserves one more honest beat, because it cuts both ways for an analyst. Tether's ability to freeze any USDT address means that, for a *defender*, recovered-funds outcomes are genuinely possible — Tether has frozen well over a billion dollars of USDT tied to thefts, scams, and sanctioned entities at law-enforcement request, money that on a freeze-less chain like Bitcoin would simply be gone. For a *holder*, though, the same function means USDT is not censorship-resistant: your "dollars" can be frozen by a centralized issuer. This is a real trade-off, not a flaw to be waved away — the property that lets investigators claw back stolen funds is the same property that makes the token controllable. A clear-eyed reader holds both truths at once and chooses their stablecoin accordingly.

## How to read it: a walkthrough

Let us do a concrete pass, the way you would on a real day, using public tools.

**Step 1 — The slow trend (DefiLlama / Glassnode).** Open the aggregate stablecoin supply chart. Ask one question: over the last three to six months, is total supply rising or falling? Rising = dry powder accumulating, a tailwind. Falling = net redemptions, a headwind. Note the *slope*, not the daily wiggle. In mid-2025 the line is near \$230B and rising — a supportive regime.

**Step 2 — The mix (DefiLlama, isolate issuers).** Split the chart into USDT and USDC. Are they moving together (clean liquidity signal) or diverging (structural story)? If USDC is shrinking while USDT grows, ask whether it is a regulatory or trust event concentrating capital offshore.

**Step 3 — The fast flow (Whale Alert / Etherscan / Tronscan).** Scan the last 24–48 hours of large mints and burns. A cluster of large USDT mints to market-maker addresses says serious cash is loading. A wave of burns during a sell-off says cash is fleeing. Check *where* minted tokens go next — straight to an exchange is the buy-side tell.

**Step 4 — The ammunition (Glassnode / CryptoQuant exchange balances).** Look at the stablecoin balance held on exchanges. Is it filling up (loading) or draining (already deployed, or leaving for yield)? Pair it with the *coin* exchange balance from the [exchange flows post](/blog/trading/onchain/exchange-flows-inflows-and-outflows): stablecoins in + coins out is the bullish micro-pattern.

**Step 5 — The risk check (depeg monitoring).** Glance at the actual *prices* of the major stablecoins. Are they all pinned to \$1.00? Any persistent deviation — even half a cent that will not close — is an early warning that the market is questioning the backing of one of your "cash" instruments. For a deeper read, glance at the big Curve stable-pool balances: a pool tilting heavily toward one coin is the market quietly swapping out of it before the price headline catches up.

Putting the five steps together into a single sentence you could say out loud each morning: *"Total dry powder is [rising / falling] over recent months, the mix is [USDT-led / USDC-led / balanced], the last day's mints [did / did not] reach exchanges, on-venue ammunition is [filling / draining], and the pegs are [intact / wobbling]."* That one sentence is a complete liquidity read of the crypto market, assembled entirely from public on-chain data, costing nothing but the discipline to look. It will not tell you what price does tomorrow. It will tell you whether the *fuel* for a move is accumulating or draining — and over the medium term, that is one of the most reliable structural reads available anywhere in the market.

A final practical note on tooling: you do not need a paid terminal to do most of this. **DefiLlama** is free and covers total supply, the per-issuer split, and per-chain breakdowns. **Etherscan** and **Tronscan** are free and let you watch issuer treasury mints and burns and trace where minted tokens go next. **Whale Alert** surfaces the large flows in real time. **Glassnode** and **CryptoQuant** add the exchange-balance and stablecoin-ratio metrics, with free tiers that cover the basics. The edge here is not expensive data — the data is public and largely free. The edge is knowing *which* number to read, *why* it moves, and the discipline to read it the same way every day.

#### Worked example: a depeg and the paper loss on a "safe" position

You hold **\$50,000 of USDC** as your "cash" position, waiting to deploy. The SVB weekend hits and USDC trades to **\$0.88**. On paper, your \$50,000 of cash is suddenly worth `50,000 × 0.88 = \$44,000` — a **\$6,000** unrealized loss on the one position you thought could never lose money. If you panic-sell into the depeg, you *realize* that \$6,000 loss. If you can verify the collateral is real and recoverable (as it was — SVB depositors were backstopped) and you hold, USDC returns to \$1.00 and your loss evaporates. The trader who understood *what backed the token* held; the one who did not, sold the bottom. **The intuition: a stablecoin is only as safe as its reserves, and a depeg turns your "cash" into a credit bet — \$6,000 on a \$50,000 position, decided entirely by whether the dollars behind it are actually there.**

## Depegs and what they do to flows

A depeg is the stablecoin world's defining stress test, and the two canonical cases teach opposite lessons.

![Depeg lows for USDC during SVB and for UST during the Terra collapse](/imgs/blogs/stablecoin-flows-the-dry-powder-metric-7.png)

**USDC / SVB (March 2023)** was a *backed* depeg. USDC was collateralized by real cash and Treasuries; the problem was that a slice of the cash was momentarily trapped at a failing bank. The market feared a shortfall and the price gapped to \$0.88. But the collateral was genuine, the bank's depositors were backstopped, and the peg snapped back to \$1.00 within days. The lesson: a backed stablecoin's depeg is a *liquidity and confidence* event, and if the reserves are real, it is survivable.

**UST / TerraUSD (May 2022)** was an *algorithmic* depeg, and a fatal one. UST was not backed by dollars at all; it was held to \$1 by a code-driven arbitrage loop with its sister token LUNA. When confidence cracked, the loop ran in reverse — minting more and more LUNA to defend the peg, which crashed LUNA's price, which broke the loop entirely. UST spiraled from \$1.00 toward \$0.10 and *never recovered*, vaporizing tens of billions of dollars. The full anatomy is its own story — see the [Terra/Luna 2022 collapse](/blog/trading/crypto/terra-luna-2022-collapse) deep dive — but the one-line lesson is permanent: **a stablecoin with no real reserve is not a stablecoin; it is a confidence game that works until it doesn't.**

The structural difference is the redemption door again. USDC's holders had a claim on \$1 of real reserves, so once the door reopened, arbitrage repaired the peg automatically. UST's holders had a claim on... more UST and a collapsing sister token. There was no door to a real dollar, so there was nothing for arbitrage to pull the price back *to*. This is the single most important test you can apply to any stablecoin before you park cash in it: *if this token traded at \$0.95 tomorrow, who would buy it, and what could they redeem it for?* If the honest answer is "real dollars from an issuer with the reserves to pay," the peg has a floor. If the answer is "nothing, or another version of the same token," you are not holding cash — you are holding a bet that confidence never breaks, and confidence always breaks eventually.

What a depeg does to *flows* is dramatic and visible on-chain. In both episodes you could watch, in real time, enormous burns as holders raced to redeem the questioned token, frantic swaps from the wobbling coin into a trusted one, and a spike of stablecoin movement onto exchanges as people tried to get out. A depeg is, in flow terms, a bank run rendered on a public ledger — and because it is public, the run is visible block by block.

The on-chain *early-warning* read is worth spelling out, because this is where reading flows pays off as defense rather than offense. Before a depeg shows up loudly in the headline price, it often shows up quietly in two places. First, in the **DeFi pool balances**: a stablecoin under stress gets dumped into automated market-maker pools (like Curve's stable pools), and the pool's composition skews — instead of a balanced 50/50 split with other dollars, it tilts heavily toward the questioned coin as holders swap out of it. A Curve stable pool that goes from balanced to 80% one coin is the market voting with its feet before the price fully cracks. Second, in **redemption queues and burns**: a sudden surge of tokens flowing back to the issuer for redemption is the on-chain signature of a run starting. A reader watching pool balances and redemption flow had a real-time, mechanical early read on both the USDC and UST events — earlier and cleaner than waiting for the price to gap. That is the defensive payoff of reading flows: you see the run forming in the plumbing before it is obvious in the price.

#### Worked example: sizing the dry powder that vanished in a collapse

When UST broke, it had a supply of roughly **\$18B** at its peak. As the peg failed, that \$18B of supposed "dry powder" was revealed to be largely fictitious — not real dollars waiting to buy, but a reflexive token propped up by its own sister asset. The genuine dollars that *had* flowed in were a fraction of the headline. Compare with USDC at the same scale: \$44B of supply backed by attested cash and Treasuries, where a \$10B redemption is \$10B of *real* dollars actually wired out. **The intuition: not all stablecoin "supply" is equal dry powder — \$1B of attested, redeemable supply is real ammunition; \$1B of algorithmic supply is an IOU from a machine that can vanish in a weekend.**

## Stablecoins as a macro-liquidity proxy

Step back to the widest lens. Because total stablecoin supply measures dollars entering and leaving crypto, it doubles as a read on **crypto liquidity conditions** — and liquidity, not earnings, is what drives crypto cycles. Stablecoin supply tends to expand when global financial conditions are easy (cheap money, risk appetite high) and contract when they tighten (high rates pull cash into Treasuries). In that sense it is a crypto-native expression of the same macro-liquidity tide that this [macro-liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset) framing describes, and that the cross-asset view treats as the [new high-beta class](/blog/trading/cross-asset/crypto-digital-assets-the-new-high-beta-class). When the stablecoin float grows, there is more cash chasing a fixed pile of coins; when it shrinks, there is less. It is one of the cleaner liquidity gauges you can read directly off the chain, with no survey lag and no revision.

This connects, too, to *where* that cash can travel. A stablecoin minted on Ethereum that ends up bidding on a Solana exchange had to cross a bridge — the seam where most cross-chain stablecoin movement, and a lot of the risk, lives. The dry powder is not stuck on one chain; it flows to wherever the action is, and following that cross-chain flow is part of reading it.

There is a deeper macro point worth making explicit. The total stablecoin float behaves like a private, crypto-native **money supply** — an M1 for the on-chain economy. In the traditional system, central banks expand and contract the money supply and economists pore over the M2 print weeks after the fact. In crypto, the money supply is the stablecoin float, it is published continuously and transparently on a public ledger, and *you* can read it in real time with no revision and no lag. When that money supply expands, the on-chain economy has more cash chasing roughly the same set of assets, and prices feel the pressure; when it contracts, the opposite. Reading stablecoin supply is, quite literally, watching the money supply of a small open economy update block by block. Few signals in any market are that clean.

The rate sensitivity ties the whole macro story together. When the US Federal Reserve holds rates high, three things happen at once that all drain the float: holding a 0%-yield stablecoin costs you the 5% you could earn in T-bills (so holders redeem), risk appetite across all markets falls (so less cash wants to be in crypto at all), and dollar funding tightens globally (so there is less marginal cash to convert in the first place). When rates fall, all three reverse. That is why stablecoin supply is, at the macro level, a *high-beta read on global dollar liquidity* — it amplifies the same tide that moves every risk asset, expressed in the one number crypto publishes openly.

## Common misconceptions

**"A big USDT mint pumps the market."** No — the causation runs backwards. Issuers mint *on demand* when a buyer wires them dollars; the mint is the receipt for incoming demand, not a money printer that levitates prices. A mint cluster is informative because it reveals that serious money is loading up, but it is a symptom of demand, not its cause. Treat it as correlation, and ask *who* minted and *where the tokens went* before reading anything into it.

**"Stablecoins are risk-free cash."** No. A stablecoin is an IOU backed by reserves you cannot personally inspect in real time. USDC's \$0.88 print and UST's collapse both happened to positions their holders called "cash." A stablecoin is only as safe as its backing, and during a depeg your cash becomes a credit bet on the issuer's reserves. The safe ones are safe *because* the reserves are real, attested, and redeemable — not because of the word "stable" in the name.

**"Total supply growing always means prices are about to rise."** No. Growing supply is *necessary fuel*, not an *ignition trigger*. Cash can sit on the sidelines for months. Supply growth raises the odds and sizes the potential of a move; it does not schedule it. And falling supply does not always mean panic — sometimes it is just rate-driven cash management, with capital redeeming to earn 5% in T-bills rather than fleeing in fear. Always ask *why* the line is moving.

**"USDT and USDC are interchangeable, so only the total matters."** No. The split is a signal in its own right. USDT growing while USDC shrinks (as after SVB) is a visible rotation from the US-regulated dollar to the offshore one — a structural story about where trust and capital are concentrating that the aggregate number hides entirely.

**"High USDT transaction volume on Tron means lots of trading dry powder."** No. Most TRC-20 USDT traffic is small-value retail and remittance flow, not buy-side trading intent. The dry-powder signal lives in exchange stablecoin *balances* and in *mints to trading firms*, not in the raw count of \$50 transfers between retail wallets.

**"A stablecoin's supply going up means the issuer is 'printing money' that dilutes my crypto."** No — this confuses a stablecoin with an inflationary token. Every new stablecoin is matched by a real dollar wired in, so the issuer is not creating value from nothing; it is *tokenizing* dollars that already existed in the banking system. The supply expansion does not dilute Bitcoin any more than someone moving cash from a savings account into a brokerage account dilutes the stocks they buy. It simply moves spendable dollars onto the chain, where they become potential demand. The "money printer" framing is catchy and wrong: the printer only runs when a real dollar is fed into the other end.

## The playbook: what to do with it

The if-then checklist for reading stablecoin flows as a dry-powder gauge:

- **Signal: total supply is in a multi-month uptrend.** → Read: cash is accumulating inside crypto; the liquidity backdrop is supportive. → Action: lean constructive on risk; treat dips as better-supported. → Invalidation: the uptrend stalls or rolls over into net redemptions; or the growth is purely a few mega-mints that never leave treasury wallets.

- **Signal: total supply is contracting (net burns) over months.** → Read: buying power is leaving the system — either risk-off flight or rate-driven cash management. → Action: respect the headwind; size risk down; check whether it is fear or just T-bill arbitrage. → Invalidation: the contraction is shallow and reverses, or it is fully explained by rates rather than risk aversion.

- **Signal: a cluster of large USDT mints to market-maker addresses, bridged to exchanges.** → Read: serious cash is loading the firing line. → Action: note the increased *odds* of near-term buying; do not treat it as a guaranteed pump. → Invalidation: the minted tokens sit in treasury or go to DeFi yield rather than an exchange; or burns offset them within days.

- **Signal: exchange stablecoin reserves rising while coin reserves fall.** → Read: the bullish micro-pattern — cash arriving to buy, coins leaving to be held. → Action: the most direct on-chain green light for spot demand; pair with the [exchange-flow](/blog/trading/onchain/exchange-flows-inflows-and-outflows) read. → Invalidation: the stablecoin inflow is collateral for derivatives, not spot buying; or the pile fills but never gets spent.

- **Signal: USDC shrinking while USDT grows (or vice versa).** → Read: a structural rotation — regulatory shock, trust event, or geographic shift. → Action: investigate the *why*; the divergence itself is the alert, not a directional trade. → Invalidation: the gap closes quickly and both resume moving together.

- **Signal: any major stablecoin trading persistently off \$1.00.** → Read: the market is questioning its backing — a potential depeg. → Action: verify the reserves (attestations, the specific bank or asset at risk); de-risk from the wobbling coin into a trusted one or real dollars; do not assume "it always comes back." → Invalidation: the deviation is a thin-liquidity blip that closes within hours, with reserves confirmed intact.

The through-line: stablecoin flows measure the *fuel*, not the *fire*. Growing, on-venue, attested dry powder is the precondition for a rally and the thing that drains in a bust — but it is a liquidity read, always correlation, never a timing trigger. Read the trend, read the mix, watch the mints, count the ammunition on the venue, and check the pegs. Do that, and you are reading the cash leg of the entire market straight off the public ledger.

## Further reading & cross-links

- [Exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — the mirror signal: coins onto a venue is potential selling, stablecoins onto a venue is potential buying. Read the two together.
- [Tokens, on-chain transfers and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals) — what a token actually is (a contract holding a balance ledger), which is exactly what a stablecoin is at the contract level.
- [Stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) — the issuer-level deep dive on reserves, attestations, and how the shadow-dollar system works.
- [The Terra/Luna 2022 collapse](/blog/trading/crypto/terra-luna-2022-collapse) — the full anatomy of an algorithmic depeg that went to zero and never came back.
- [Crypto as a macro-liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset) — why stablecoin supply doubles as a crypto-native liquidity gauge, and how the macro tide drives the cycle.
