---
title: "What Is On-Chain Analysis? Reading the Public Ledger for an Edge"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A blockchain is a permanent, public, pseudonymous ledger of every transfer ever made. On-chain analysis is the discipline of reading that ledger to see who is moving money, where, when, and why — before it shows up in price."
tags: ["onchain", "crypto", "blockchain", "exchange-flows", "smart-money", "blockchain-forensics", "ethereum", "bitcoin", "etherscan", "due-diligence"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — On-chain analysis is the skill of reading a public blockchain ledger — every transfer is permanent, visible, and pseudonymous — to answer questions price alone cannot: who is moving money, where, when, and why.
>
> - **What it is:** a blockchain is a shared, append-only record of every transaction. Anyone can read it. On-chain analysis turns that raw record into answers — exchange supply, who is accumulating, whether a token is safe, where stolen funds went.
> - **How to read it:** start with a *block explorer* (Etherscan, Solscan) to read a single address, then graduate to *analytics platforms* (Arkham, Nansen, Dune, DeBank) that label addresses and aggregate flow.
> - **What you do with it:** the edge is **lead time** — coins move on-chain before the move shows in price. You trade it, you avoid the rug, or you trace the hack. But the chain lies too — wash trades, bait wallets, survivorship-biased "smart money" — so you *verify*, you never trust the green number.
> - **The number to remember:** an exchange inflow of 12,000 BTC at \$60,000 is \$720M of coins moving toward the order book — potential sell supply you can see before anyone hits "sell."

On 23 March 2022, an attacker drained the Ronin Bridge — the cross-chain bridge that powered the play-to-earn game Axie Infinity — of roughly **\$625M** in ETH and USDC. It was, at the time, the largest crypto theft ever recorded. Here is the part that matters for this series: *the world could watch the money move.* The exploit transactions, the addresses that received the funds, the swaps that converted USDC into ETH, the deposits into the Tornado Cash mixer — all of it was written to the Ethereum ledger in plain sight, permanently, for anyone to read. Within days, investigators had traced the route. Within weeks, the U.S. Treasury and the FBI publicly attributed the theft to the Lazarus Group, a state-sponsored North Korean hacking unit, partly *because* the on-chain trail matched patterns from earlier Lazarus operations.

Now hold that next to a completely ordinary event. On a quiet Tuesday, an old Bitcoin wallet that had sat untouched since 2016 suddenly sends 5,000 BTC to a known exchange deposit address. No press release. No tweet. No news. But anyone watching the chain saw it the moment it confirmed — and a wallet that old, that large, moving onto an exchange, is a classic *distribution* signal: someone who has held for years is positioning to sell. The price had not moved yet. The information had.

That gap — between *what the chain shows* and *what the price has caught up to* — is the entire premise of on-chain analysis. Price is one number, updated after the fact. The chain is the raw event stream that *produces* that number: every transfer, every swap, every deposit and withdrawal, every contract call, recorded as it happens. If you can read it, you are reading the market's plumbing, not its dashboard.

![On-chain analysis loop with the public ledger as a glass box and the steps observe, label, infer, act](/imgs/blogs/what-is-onchain-analysis-1.png)

This post is the front door to a ten-track series on reading that plumbing. By the end of it you will understand exactly what a blockchain ledger is, why "pseudonymous" is not "anonymous," the difference between an exchange's public deposit address and its hidden internal books, what tools turn raw hex into answers, and the four big questions on-chain analysis exists to answer. We will walk through one famous, fully-documented episode — the Ronin trace — to make it concrete, and we will be honest, throughout, about where the chain lies and how it fools careless analysts. Three ideas recur in every single later post, so we name them up front: **(1) the ledger is public, permanent, and pseudonymous; (2) flow leads price; (3) on-chain lies too — always verify.**

## Foundations: what a blockchain ledger actually is

Before any signal, any tool, any trade, you need a clear mental picture of the thing you are reading. We will build it from zero. No prior crypto knowledge assumed; every term is defined the first time it appears.

### A ledger is just a list of transfers

Forget tokens, mining, and "decentralization" for a moment. At its core, a blockchain is a **ledger** — the same word your grandparents' shop used for the notebook where they wrote down who paid what. A ledger is a list of transactions. Each line says *from this account, to that account, this amount, at this time.* Your bank keeps a ledger. It is private: only the bank and the regulator see it. A blockchain keeps a ledger too, with two radical differences.

First, it is **public**. The full list of every transaction since the chain began is downloadable by anyone. There is no login, no permission, no fee to *read* it. You can run software (a "node") that holds the entire history, or — far more commonly — you let a website that already runs one show it to you.

Second, it is **permanent and append-only**. Once a transaction is confirmed and buried under enough later transactions, it cannot be edited or deleted. The ledger only ever grows. There is no "undo," no backspace, no quiet correction. A mistake made in 2017 is still right there in 2026. This is why we keep saying *permanent*: the chain has a memory that never forgets, which is exactly what makes years-later forensics possible.

The "block" in blockchain refers to *how* new transactions get added: they are bundled into a **block** (a batch of transactions) roughly every few seconds (Ethereum) to ten minutes (Bitcoin), and each block carries a cryptographic fingerprint of the block before it, forming a *chain*. That chaining is what makes the history tamper-evident — change one old transaction and every fingerprint after it breaks. You do not need the cryptography to do on-chain analysis; you need to trust the result: **the record is honest, public, and final.**

There is one more piece of plumbing worth naming because it shows up in every transaction you will read: the **fee** (on Ethereum, paid in "gas"). Because block space is scarce, anyone who wants their transaction included must pay the network a fee, and the size of that fee is itself a readable signal. An attacker draining a bridge will often pay an enormous fee to make sure the theft confirms *now*, before anyone can react — an abnormal fee on an abnormal transfer is a flag in its own right. A patient long-term holder consolidating coins might wait for a quiet weekend when fees are cheap. The fee is the price of *urgency*, and urgency is information. You will see it on every line of every explorer: amount, sender, receiver, timestamp, and the fee paid to get there.

Two more terms before we leave the basics, because the series uses them constantly. An **EOA** — *externally owned account* — is a regular wallet controlled by a private key (a person or a bot). A **smart contract** is code that lives at an address and runs when called — an exchange, a lending pool, a token, a bridge are all contracts. The difference matters for reading intent: an EOA *decides* to move money; a contract *executes* rules. When you see funds flow from an EOA into a contract and back out transformed, you are watching a person interact with a protocol, and the protocol's rules constrain what could have happened.

#### Worked example: the size of the thing you are reading

The Ethereum ledger records on the order of one to one-and-a-half million transactions *per day*. Say a transfer moves \$1,000 of stablecoins; in a single busy day the chain might record tens of billions of dollars of such transfers, each one a line you can read: sender, receiver, amount, timestamp, fee. Over a year that is hundreds of millions of lines. The art of on-chain analysis is not reading every line — no human could — it is knowing *which* lines matter and how to query for them. The intuition: you are not reading a book, you are searching a public database that happens to contain every payment ever made on the network.

### Address versus identity: pseudonymous, not anonymous

On a blockchain you do not have a username. You have an **address**: a long string like `0xA11ce…f3` (Ethereum) or `bc1q…` (Bitcoin) derived from a cryptographic key pair. Whoever holds the matching secret key controls whatever sits at that address. An address is a *pseudonym* — a stand-in name — not your real name. Nobody at the network checked your passport.

This is the single most misunderstood property of crypto, so we will be precise. **Anonymous** means *untraceable* — no link, ever, between the action and a person. **Pseudonymous** means *the name is a placeholder, but every action under that name is permanently linked to it and to every other action under it.* A blockchain address is pseudonymous. Every transaction it ever makes is glued to it, forever, in public. You do not start anonymous and stay that way; you start *unlabeled*, and you leak identity with every move.

How does the placeholder name get tied to a real entity? Three forces, which later tracks cover in depth:

- **Funding and cash-out points.** To get coins, most people first buy on a centralized exchange that did collect their passport (this is called KYC — *Know Your Customer*). The moment money flows between that KYC'd exchange account and a personal wallet, the wallet is one hop from a real identity. To spend coins on anything real, the same hop usually happens in reverse.
- **Clustering.** If addresses repeatedly move funds together, are funded from one source, or sign in patterns only one actor would use, analysts *cluster* them as "probably the same entity." On Bitcoin, the classic heuristic is co-spending: if two addresses are used as inputs to the same transaction, the same key-holder likely controls both.
- **Behavior and labels.** Exchanges, large funds, and known bad actors get *labeled* by analytics firms — through subpoenas, leaks, public deposit-address postings, and statistical fingerprinting. Once an address is labeled "Binance hot wallet" or "Lazarus 2022," everything flowing through it is interpretable.

So the honest framing for the whole series: **the chain is pseudonymous, and pseudonymity erodes over time.** A brand-new address tells you little. The same address after a year of activity, clustered and labeled, can tell you a great deal.

### How pseudonymity actually erodes: a clustering sketch

It is worth slowing down on *how* an unlabeled address becomes a named entity, because the mechanism is the foundation of every later forensics post, and because understanding it tells you how to read the limits.

Start with the Bitcoin model. Bitcoin does not have account balances the way your bank does. Instead it tracks **UTXOs** — *unspent transaction outputs*, which behave like individual coins of varying denominations in a wallet. To pay someone, you spend one or more whole UTXOs as *inputs*, and the transaction creates new UTXOs as *outputs* — one to the recipient, and usually one of "change" back to yourself, exactly like handing over a \$20 bill for a \$13 coffee and getting \$7 back. The key forensic consequence: if a single transaction spends two inputs at once, the same private key signed both, so the same entity almost certainly controls both input addresses. This is the **co-spend heuristic**, and run across millions of transactions it collapses thousands of separate-looking addresses into a handful of real wallets. A second heuristic spots the *change* output (often a freshly-created address receiving the leftover) and links it back to the spender. Neither is perfect, but together they let analysts cluster Bitcoin's pseudonyms with surprising reach.

Ethereum uses the simpler **account model**: an address has a running balance, like a bank account, and a transfer just debits one and credits another. There is no change output to chase, but the same de-anonymizing forces apply through *behavior* — an address that always interacts with the same contracts, funds the same downstream wallets, and signs at the same hours of day leaves a fingerprint. Cluster on funding source (who first sent it coins), on co-movement (which addresses move together), and on interaction patterns (which contracts and counterparties recur), and the pseudonym narrows toward an entity.

The last mile is the **label**, attached by analytics firms through public deposit-address disclosures, exchange compliance data, leaks, court records, and statistical matching to known clusters. Once an address carries a trusted label — "Coinbase 10," "Lazarus 2022 staging wallet," "this is the deployer's wallet" — every flow through it becomes interpretable. The honest caveat, again: a label is a probabilistic claim by a private firm, and a wrong label poisons every conclusion built on it. Cluster, then label, then *verify the labels your conclusion depends on* against the raw chain.

### On-chain versus off-chain: the part you can see and the part you cannot

Here is a distinction that trips up almost every beginner and quietly invalidates half of all bad on-chain takes. Not everything that happens in crypto happens *on* the chain.

A **centralized exchange** (CEX) — Binance, Coinbase, Kraka, OKX — is a company that holds your coins for you and lets you trade them on *its* internal system. When you buy 1 BTC from another user inside Binance, no Bitcoin transaction occurs. Binance simply edits two rows in *its own private database*: your balance goes up, the seller's goes down. That trade is **off-chain**. It never touches the public ledger, so on-chain analysis cannot see it directly — the same way you cannot see the trades happening inside a bank by watching the bank's front door.

What you *can* see is the front door: when coins move **onto** an exchange (a deposit) or **off** an exchange (a withdrawal), those are real on-chain transactions to and from the exchange's publicly-known addresses. So the chain shows you the *boundary* of the exchange — money flowing in and out — while the *interior* (who traded with whom, the order book, leverage, customer IOUs) stays hidden.

![On-chain versus off-chain map showing visible public transfers and hidden centralized exchange internal books](/imgs/blogs/what-is-onchain-analysis-3.png)

This boundary is one of the richest signals in the whole discipline, and also one of the most abused. We will return to *exchange flows* below and devote an entire later track to them. For now, lock in the mental model: **on-chain = public and visible; off-chain = private exchange books and hidden.** When someone says "the on-chain data shows Binance is dumping," ask immediately: did they see coins leave Binance's wallets on-chain (real signal), or did they infer it from price action inside the exchange (not on-chain at all)?

### Block explorer versus analytics platform

The last foundational distinction is about *tools*, because "reading the chain" means different things at different altitudes.

A **block explorer** is a free website that shows you the raw ledger one item at a time: a single address and its balance, a single transaction and its details, a single block and the transactions in it. Etherscan (Ethereum), Solscan (Solana), mempool.space and Blockchair (Bitcoin), Tronscan (Tron) are explorers. They are *primary sources* — they show you exactly what the chain says, no interpretation. Every serious analyst keeps an explorer open. It is the equivalent of reading the official court record rather than the newspaper summary.

An **analytics platform** sits on top of the raw chain and adds the two things explorers lack: **labels** and **aggregation.** Arkham, Nansen, DeBank, Dune, Bubblemaps, and Glassnode ingest the whole chain, tag known addresses ("Coinbase," "Wintermute," "Tornado Cash," "this address is a smart-money wallet"), and let you ask aggregate questions — "how much ETH did all exchanges receive this week," "which wallets are accumulating this token," "draw me the cluster of wallets connected to this one." They are *secondary sources*: faster and far more powerful, but they layer judgment on top of the raw data, and that judgment can be wrong. A label is a *claim*, not a fact.

The discipline, then, is a two-tool dance: use the analytics platform to *find* the signal fast, then drop to the block explorer to *verify* it against the primary record. Every later post in this series uses both, and we will keep reminding you which is which.

## Why this is an edge: information asymmetry versus lead time

We keep using the word "edge." It deserves a precise definition, because two very different things hide under it, and confusing them is how people lose money.

**Information asymmetry** is when you know something the market does not — a fact nobody else has. On a *public* ledger, true secret information is rare: everyone can read the same data. You almost never have a fact no one else *could* have.

**Lead time** is different and far more realistic: the data is public, but *you read it before the crowd reacts.* The signal sits in the open; the edge is being early to interpret it. When 5,000 dormant BTC hit an exchange, the transaction is visible to everyone — but most market participants are not watching, do not know the wallet's history, and will only react when the price starts dropping. If you saw it, understood it, and acted in the hours before the rest caught up, you had an edge. Not because the information was secret, but because you were *first to read public information correctly.*

This reframes the entire series. You are not hunting for secrets. You are building a *reading practice* — a routine of watching the right addresses, the right flows, the right contracts — so that when the chain reveals something, you understand it before the price does. That lead time is the prize. And it is fragile: as more people watch the same signals, the lead time shrinks, and the very same flow that was a clean signal becomes crowded noise. We will be honest about that decay everywhere.

The decay is worth dwelling on, because it is the reason naive on-chain trading disappoints people who expected a money printer. When a signal is obscure — a specific flow that only a handful of analysts track — reading it early is a genuine edge. As the signal becomes popular, three things happen in sequence. First, the *lead time compresses*: more eyes mean the price catches up faster, so the window between "you saw it" and "everyone saw it" shrinks from hours to minutes. Second, the signal becomes *gameable*: once a flow reliably moves price, sophisticated actors fake that flow on purpose — staging the deposit, the bait wallet, the wash volume — to trap the people trading it mechanically. Third, the *crowd itself becomes the move*: when ten thousand people all sell on the same exchange-inflow alert, the selling they cause is downstream of the alert, not of any real fundamental, and the signal starts predicting its own followers rather than the market. The practical takeaway is not despair; it is that on-chain analysis is a *living* skill. The edge migrates — from the signal everyone now watches to the next one they do not, from the obvious read to the second-order one (who is *faking* the obvious read, and why). A reader who treats any single metric as a permanent edge will be farmed; a reader who treats the chain as an evolving information landscape, and keeps asking "who else is watching this, and what would they do," stays ahead of the decay.

![The three pillars of on-chain analysis: public and permanent, flow leads price, and the chain lies too](/imgs/blogs/what-is-onchain-analysis-2.png)

The three pillars in the figure are the spine of the whole series. Pillar one — public, permanent, pseudonymous — is *why* the data exists to read. Pillar two — flow leads price — is *why* reading it pays. Pillar three — the chain lies too — is *why* you verify before you trust. Keep all three in your head simultaneously; drop any one and you will either miss the signal, misread it, or get played by it.

## The four big questions on-chain analysis answers

Strip away the jargon and on-chain analysis exists to answer four concrete questions. Almost every signal, tool, and later post maps onto one of them.

### Question 1: How much supply is sitting on exchanges, ready to sell?

Coins on an exchange are coins that *can be sold right now*. Coins in a private wallet have to be moved onto an exchange first, and that move is visible. So the flow of coins onto and off of exchanges is a leading indicator of selling and buying pressure.

The simple read: **coins flowing onto exchanges = potential sell supply arriving; coins flowing off exchanges = coins being taken to self-custody, often a sign of conviction or accumulation.** The aggregate level — total coins held across all exchanges — is the slow-moving version of the same idea: a multi-year decline in exchange balances means the float available to dump is shrinking.

![Bitcoin held on exchanges falling from 2.6M to about 2.3M coins between 2020 and 2022](/imgs/blogs/what-is-onchain-analysis-7.png)

The chart shows aggregate Bitcoin held on exchanges (an estimate — exchanges do not announce their addresses, so analysts infer them, hence "approx"). The level fell from roughly 2.60M coins in 2020 to about 2.30M by 2022 — around **300,000 BTC** leaving exchange wallets over that stretch.

#### Worked example: pricing an exchange-reserve drawdown

Take that 300,000-BTC decline and put a dollar figure on it. At a representative price of \$30,000 per BTC over the period, 300,000 BTC is **\$9B** of coins moved from "sittable on an order book" to "held in private wallets." That is \$9B of potential immediate sell supply removed from the venues where selling actually happens. It does not *guarantee* the price rises — demand still has to show up — but it changes the supply backdrop: every buyer now competes for a thinner available float. The intuition: exchange reserves are the market's "ammunition on the table," and watching that pile shrink or grow is watching the conditions for the next move set up.

The flip side is just as readable. A sudden *inflow* spike is the warning.

#### Worked example: reading an exchange inflow spike

Suppose your dashboard flags 12,000 BTC flowing **into** exchange addresses in a single day, while BTC trades at \$60,000. That is 12,000 × \$60,000 = **\$720M** of Bitcoin arriving at the exact venues where it can be sold — in one day. It does not mean a crash is coming; some of it is market-makers cycling inventory, some is users topping up to buy other coins. But a large, abnormal inflow concentrated from a few old wallets, into spot exchanges, is the classic "supply is positioning to sell" pattern. The action is not to panic-sell; it is to *tighten risk* and watch whether the inflow is followed by realized selling. The intuition: \$720M of coins moving toward the order book is a fact you can see hours before any of it prints as a red candle.

### Question 2: Who is accumulating, and who is distributing?

Not all wallets are equal. Some belong to entities with a track record of being early and right — early investors, profitable funds, addresses that bought the last bottom. Analytics platforms label these "**smart money**" and let you watch what they do. When such wallets are steadily *accumulating* a token (net buying, balances rising) it is a different signal than when they are *distributing* (net selling into strength). The same logic applies to whales (very large holders) and to specific cohorts like a token's team and early-investor wallets, whose unlocks and sells you can watch coming.

This is powerful and dangerous in equal measure, because "smart money" labels suffer brutal **survivorship bias**: a wallet looks smart *because* it happened to win, and a platform that surfaces the winners after the fact will always show you geniuses. We devote a whole later track to doing this honestly. For now: who is moving is a real question with a readable answer, and the answer requires heavy skepticism about the labels.

#### Worked example: reading an early-investor exit

Here is why "who is moving" is worth the skepticism. Suppose an address bought **4,000,000 tokens at \$0.002** very early — a position that cost just `4,000,000 × \$0.002 =` **\$8,000**. Months later the token trades at \$0.05, and the chain shows that same wallet sending its entire balance to an exchange deposit address over three transactions. At \$0.05, that stack is worth `4,000,000 × \$0.05 =` **\$200,000** — a 25× exit, and the wallet is unmistakably *distributing*. An analyst watching saw the sells positioning before the price gave it back. The trap: by the time a platform labels this wallet "smart money" and the crowd piles in to copy it, the smart money is the one *selling to the crowd*. The intuition: the same flow that is a brilliant exit for the insider is a top signal for everyone copying them a beat too late.

### Question 3: Is this token safe, or is it a trap?

Before you buy a new token, the chain can tell you things its website never will. Who holds it (is 90% in five wallets that can dump on you)? Can the contract be changed to freeze your funds or mint unlimited new tokens? Is there real liquidity, or a thin pool the deployer can pull (a "rug pull")? Can you actually *sell* after buying, or is it a "honeypot" contract that only lets the deployer sell? These are **due-diligence** questions, and on-chain they have concrete, checkable answers — token holder distribution, contract permissions, liquidity-lock status, the deployer's history. An entire track is dedicated to this rug-check workflow, framed strictly as *how to protect yourself*, never how to build the trap.

The asymmetry here is what makes the discipline worth it, and it is worth stating plainly because it governs how you weigh the work. A safety check that makes you skip a token that *would* have 10×'d cost you opportunity — painful, but survivable, because there is always another trade. A safety check you *skipped* on a honeypot or a rug costs you principal — the entire position, gone, with no recovery. Those two errors are not symmetric, so the correct bias is not symmetric: when a single on-chain check fails — the top ten wallets can dump the float, the contract has a live mint function, the liquidity is unlocked and pullable, or a test of the sell path shows you cannot exit — you walk, and you do not negotiate with yourself about the upside you might be missing. The chain hands you these answers *before* you commit a dollar; the only mistake is not looking.

The Solana memecoin world makes the survivorship side of this vivid. On launchpads like Pump.fun, on the order of **8 million** tokens have been created cumulatively, and credible estimates put the share that ever reach a meaningful market cap at roughly **1 to 2 percent**. The other ~98% go to zero, fast. So when you see a token that *did* moon, remember the graveyard you are not shown — the base rate for a random new token is near-total loss, and the chain (holder concentration, liquidity, the deployer's prior launches) is the only thing that lets you tell the rare survivor from the overwhelming majority that were designed to extract your money.

### Question 4: Where did the stolen money go?

When a protocol is hacked or a scam cashes out, the funds move on-chain, and — because the ledger is permanent — they leave a trail that can be followed for years. **Blockchain forensics** is the discipline of following that trail: through swaps (changing one token for another), across **bridges** (moving funds from one chain to another), and into **mixers** (services that pool many users' funds to break the link between deposit and withdrawal). This is the defender's craft, and it is the heart of the next section.

![The biggest on-chain hacks leaderboard led by Bybit and Ronin Bridge measured in USD stolen](/imgs/blogs/what-is-onchain-analysis-5.png)

The leaderboard is sobering and instructive. The largest, the **Bybit** exploit of February 2025, moved about **\$1.46B** — the single biggest crypto theft ever, attributed to Lazarus. Ronin sits second at **\$625M**. Notice the outcomes baked into these episodes (covered case-by-case in the forensics track): Poly Network's **\$611M** was *fully returned* by a grey-hat hacker; Euler Finance's \$197M was returned; others were laundered or only partially recovered. The point for an intro: every one of these is a public, traceable event. The size of the theft did not erase the trail — it created the most-watched trail in the world.

## The launder route, and how an analyst reads it: the Ronin trace

Let us make Question 4 concrete with the best-documented episode in the dataset. This is a *forensic case study* — we describe how investigators *detected and traced* the flow, which is exactly the legitimate defender's craft. It is not a how-to for moving stolen funds; the steps below are visible *because* the chain is public, and that visibility is the point.

**The setup.** The Ronin Bridge held the assets backing the tokens used in the Axie Infinity game across two chains. A bridge works by locking your asset on chain A and issuing a claim on chain B; whoever controls the bridge's signing keys controls the locked assets. On 23 March 2022, an attacker who had compromised enough of Ronin's validator keys signed two withdrawals draining roughly **173,600 ETH and 25.5M USDC** — about **\$625M** at the time.

**What an analyst saw, in order.** Because every step is on-chain, an observer with an explorer and a labeling platform could reconstruct it in near real time:

1. **The drain.** Two large withdrawal transactions emptied the bridge into a single attacker-controlled address. Anomaly number one: a bridge's reserves do not normally leave in two giant transactions to a fresh wallet.
2. **The consolidation and conversion.** The stolen USDC was swapped for ETH through decentralized exchanges. Why ETH? Stablecoin issuers like Circle (USDC) and Tether (USDT) can *freeze* tokens at a given address with a single transaction; ETH cannot be frozen by any issuer. Converting frozen-able stablecoins into unfreezable ETH is itself a tell — a defender watching knows the actor is trying to lock in funds that no central party can claw back.
3. **The mixer.** The ETH was then deposited, in chunks, into **Tornado Cash** — an Ethereum mixer that pools deposits so withdrawals cannot be trivially linked back to a specific deposit. This is the obfuscation step. Crucially, mixing breaks the *direct* link but not the *statistical* one: timing, amounts, and the destinations of withdrawals still leak information, and the entry into the mixer is fully public.
4. **The attribution.** Investigators (Chainalysis, Elliptic, and the FBI) matched the on-chain behavior, the wallet-funding patterns, and the laundering route to prior Lazarus Group operations and publicly attributed the theft to North Korea. A portion was later seized as it touched off-ramps.

![The on-chain analysis loop with the public ledger as a glass box and the steps observe, label, infer, act](/imgs/blogs/what-is-onchain-analysis-1.png)

That four-step loop in the cover figure is exactly what the Ronin trace runs: **observe** the abnormal flow (the drain), **label** the addresses (attacker wallet, the DEX routers, the Tornado Cash contracts), **infer** intent (convert to unfreezable ETH, then mix to obfuscate), and **act** (in the defender's case, alert exchanges and law enforcement, flag the addresses so off-ramps refuse the funds). New flow re-opens the loop; investigators watched these clusters for *years*.

#### Worked example: framing the \$625M Ronin trace as money you can follow

Put numbers on why this trail is durable. The \$625M did not vanish — it became `173,600 ETH + 25.5M USDC` sitting in one address that the entire investigative world could see. Converting \$200M-plus of USDC into ETH at, say, \$3,000 per ETH adds roughly 67,000 ETH to the pile — a swap of that size on decentralized exchanges is itself a screaming-loud public event. Even routed through a mixer, the *deposits* are public and the laundering throughput is bounded: pushing hundreds of millions through a mixer takes time and leaves timing fingerprints. The intuition: on a public, permanent ledger, stealing \$625M is the easy part; *spending* it without leaving a years-long, globally-watched trail is the part that has never been fully solved.

The mixer step deserves one honest caveat, which is the whole reason a later track covers it carefully: Tornado Cash over its life processed on the order of **\$7B** in deposits, of which credible estimates tie roughly **30%** to illicit sources — meaning the *majority* of mixer volume was ordinary users seeking privacy, not criminals. The U.S. Treasury sanctioned the Tornado Cash contracts on 8 August 2022, which raised hard questions about sanctioning *code* rather than people. We link out to a dedicated treatment of that below; for this intro, the lesson is narrow: **obfuscation tools make tracing harder, not impossible, and the entry points are always public.**

A word on **bridges**, because they are the cross-chain seam where so much of forensics and so many hacks live. A bridge lets value move between two separate blockchains that cannot natively talk to each other — say from Ethereum to a low-fee network. The standard design *locks* your asset in a contract on the source chain and *issues* a matching claim token on the destination chain; to come back, you burn the claim and unlock the original. The entire security of that arrangement rests on whoever controls the bridge's signing keys, which is precisely what the Ronin attacker compromised. For an analyst, bridges matter for two reasons. First, they are honeypots: a bridge by definition holds a large pool of locked assets, making it the richest single target on-chain — several of the biggest entries on the hacks leaderboard (Ronin, Wormhole, Nomad, Poly Network) are bridge exploits. Second, they are the place a trail *appears to go cold*: funds enter a bridge on one chain and emerge on another, and an analyst who only watches one chain loses them. The craft of cross-chain tracing — matching the deposit on chain A to the withdrawal on chain B by amount and timing — is exactly what the forensics track teaches. The deposit and the withdrawal are both public; the bridge does not erase the trail, it just splits it across two ledgers you have to read together.

## A short, honest tour of the ten tracks ahead

This series is built as ten tracks. Here is the map, so you know where each future post fits and why we are building in this order.

![The on-chain analysis series map of ten tracks from foundations to forensics and the playbook](/imgs/blogs/what-is-onchain-analysis-4.png)

1. **Foundations** — blocks, addresses, transactions, gas, the account model versus the UTXO model. The vocabulary everything else assumes.
2. **The tools** — a hands-on tour of Etherscan, Arkham, Nansen, DeBank, Dune, Solscan, and Bubblemaps: what each is for, what it gets right, and where its labels can mislead.
3. **Exchange flows** — reserves, deposit and withdrawal flows, stablecoin movements onto venues; the supply-on-tap reading from Question 1, done properly.
4. **Smart money** — finding, validating, and following wallets with a track record, while fighting survivorship bias every step.
5. **Stablecoins as dry powder** — minting and redemption of USDT and USDC as a gauge of capital entering and leaving the system.
6. **Reading DeFi** — total value locked (TVL), liquidity-pool flows, and what they say about risk appetite and where money is rotating.
7. **Due diligence** — the rug-check and honeypot-detection workflow, framed entirely as self-defense before you buy.
8. **Market hygiene** — wash trading, MEV (the value extracted by transaction-ordering), and bait wallets: how the chain is gamed and how to not be the mark.
9. **Forensics** — tracing hacks across swaps, bridges, mixers, and peel chains; the Ronin-style investigation, generalized.
10. **The playbook** — turning every signal into the same disciplined loop: signal → read → action → invalidation.

Each track stands on the three pillars and ends, like this post, with a *playbook* — a concrete if-then checklist — so you are never left with a fascinating fact and no idea what to do with it.

## How the chain lies: the limits you must respect

A series that only sold you the upside would be malpractice. The single most valuable habit in on-chain analysis is *skepticism about your own read*. Here are the structural limits, each of which a later post treats in depth.

**CEX internal flows are opaque.** We said it above; we repeat it because it invalidates so many takes. You see deposits and withdrawals at an exchange's border, never the trading inside. "On-chain shows the exchange is selling" is almost always an inference, not an observation. The honest version: "coins flowed onto this exchange's address," full stop.

**Volume can be washed.** **Wash trading** is buying and selling to yourself to fake activity — a deployer trading a token with their own other wallets to make a dead coin look alive, or an NFT flipped between two controlled wallets at rising prices to manufacture a fake floor. On-chain, the volume number looks real; the wallets behind it are the same hand. A green "24h volume" figure is not evidence of genuine demand until you check *who* is trading.

**Wallets can be bait.** Knowing that the crowd follows "smart money," sophisticated actors plant **bait wallets**: addresses that look like an early, profitable insider, seeded to make a token look like it is being accumulated by people who know something. The crowd front-runs the "smart" wallet straight into the exit liquidity the deployer set up. The label "smart money" is a target painted on your back as much as a signal.

**"Smart money" suffers survivorship bias.** A platform shows you the wallets that *won*. For every labeled genius who 100×'d, thousands of similar-looking wallets went to zero and were never surfaced. Copying a wallet *after* it is famous is buying its thesis at the top. The wallet's past returns are not a forward-looking edge.

**Labels are claims, not facts.** Every "Binance hot wallet" or "Jump Trading" tag is a probabilistic attribution by a private firm. They are usually right and occasionally wrong, and a wrong label propagates into wrong conclusions across every dashboard that copied it. Verify the important ones against the primary record on a block explorer.

These are not reasons to abandon on-chain analysis. They are the reasons it is a *skill* and not a button. The data is real; reading it correctly is the hard part.

## On-chain is not "crypto is crime": the illicit-share reality

One more piece of honesty, because the hacks leaderboard above can leave a false impression. Yes, billions are stolen in crypto every year. But that figure has to be read *against the size of the whole system*.

![Value stolen from crypto platforms per year and the tiny illicit share of all on-chain volume](/imgs/blogs/what-is-onchain-analysis-6.png)

The bars show value stolen from platforms per year — roughly \$3.3B in 2021, \$3.8B in 2022, \$1.7B in 2023, \$2.2B in 2024 — with the DPRK/Lazarus-attributed slice broken out (well over \$5B cumulatively across these years). Those are real, large numbers. But the line tells the other half: **illicit transactions are a tiny share of all on-chain volume — on the order of 0.14% to 0.62%**, depending on the year and the methodology. The overwhelming majority of what the chain records is ordinary, legitimate activity: payments, trading, DeFi, savings.

#### Worked example: scaling the crime against the system

Put the two numbers in the same frame. Take 2024: roughly **\$2.2B** stolen, against an illicit share of about **0.14%** of total volume. If illicit flow is 0.14% of the total, then total on-chain volume that year was on the order of \$2.2B ÷ 0.0014 ≈ **\$1.5T** *just from the stolen-funds slice's implied base* — and that undercounts, because not all illicit flow is theft. The real point: even the biggest theft years are a rounding error against the legitimate trillions moving on-chain. The intuition: the chain is not a crime ledger with some legitimate use; it is a legitimate financial ledger with a small, highly-visible, and *traceable* criminal fringe — and that very traceability is why the fringe keeps getting caught.

This matters for your reading practice in a practical way: if you assume every large flow is sinister, you will misread the 99.86% of activity that is mundane. On-chain analysis is mostly about *normal* money — who is buying, who is selling, where liquidity is — not about chasing hackers. The forensics is the dramatic part; the day-to-day edge is in the ordinary flow.

## How to read a single address: the smallest possible walkthrough

Everything above is the why. Here is the smallest possible *how* — the one habit to build first — using a block explorer, the primary source. We will keep it generic (no real personal address) and conceptual, since the dedicated tools track goes deep.

Open a block explorer and paste in an address. You will see, immediately:

- **The balance** — how much of the native coin (ETH, BTC) and which tokens this address holds *right now*. A balance that is 95% one obscure token is a flag; a balance spread across blue-chip assets reads differently.
- **The transaction history** — every transfer in and out, newest first, with timestamps, counterparties, and amounts. This is the address's entire life story. You read it like a bank statement that never lies and never hides a line.
- **The age and first funding** — when the address was created and *where its first coins came from*. An address funded directly from a major exchange's withdrawal wallet is one hop from a KYC'd identity; an address funded out of a mixer is wearing a disguise on purpose.
- **The token approvals (Ethereum)** — which contracts this address has granted permission to move its tokens. A forgotten, unlimited approval to a sketchy contract is how wallets get drained long after the user forgot about it.

The reading practice: do not look at one transaction, look at the *pattern*. Is this address accumulating (balance rising, buys clustered) or distributing (balance falling, sells into strength)? Is it funded from an exchange (likely a person) or from a mixer (likely hiding)? Does its behavior match the label some platform put on it? Then — and this is pillar three in action — drop the convenient dashboard summary and confirm the two or three facts your conclusion depends on against the raw explorer record. The chain is the primary source; everything else is commentary.

Let us walk one realistic pass end to end, using illustrative placeholders so we never point at a real person. Say an analytics platform surfaces a wallet, call it `0xA11ce…`, with a label that reads "early DeFi investor, high win rate." That label is a *hypothesis*, and the two-tool dance turns it into a judgment:

1. **Open it on the explorer.** Drop the address into Etherscan. First look: the balance — is it concentrated in one token (a single conviction bet, or a single bag it is stuck in) or spread across blue chips? Then the age: an address active since 2019 has had time to earn its label; one created last month has not.
2. **Trace the first funding.** Scroll to the oldest transaction. Where did the very first coins come from? If they arrived from a major exchange's withdrawal wallet, this is one hop from a KYC'd human — plausibly a real, accountable investor. If they arrived from a mixer or a chain of fresh wallets, the address is *designed* to look unattached, and "early investor" should drop to "actor hiding its origin."
3. **Read the recent pattern.** Are the last twenty transactions net inflows (accumulating) or net outflows to exchange deposit addresses (distributing)? A wallet *buying* the token you are researching is a very different message than one quietly sending it to Binance to sell.
4. **Check the approvals and the counterparties.** Which contracts can move this wallet's tokens, and which addresses does it repeatedly transact with? If it co-moves with a cluster of look-alike "smart" wallets that all funded from the same source on the same day, you are likely looking at one operator running many wallets — a manufactured "smart money" appearance, not many independent geniuses.
5. **Decide, then verify the load-bearing facts.** Form a read — "this looks like a genuine early holder still accumulating" — and then re-confirm the two facts it hinges on (the funding source and the recent net flow) directly on the explorer, not on the platform's summary card. If those two facts hold, you have a weak-but-real input. If either breaks, you drop the read entirely.

That five-step pass is the whole discipline in miniature: observe, label, infer, act — and verify against the primary source before the conclusion leaves your head. Run it on a handful of wallets a day and the routine becomes second nature; run it on a hundred and you will start to *feel* the difference between a real accumulation and a staged one.

#### Worked example: the lead time on a dormant-wallet move

Quantify why this routine pays. Suppose your watchlist includes ten old, large wallets, and one of them — dormant for four years, holding **2,000 ETH** — suddenly sends the full balance to an exchange deposit address while ETH trades at \$3,000. That is `2,000 × \$3,000 =` **\$6M** of long-held coins arriving where they can be sold, and you saw it the minute it confirmed. The price has not moved. You now have a window — minutes to hours — before the broader market notices the deposit, the selling begins, and the candle turns red. Whether you act (trim risk, hedge) or simply log it, the edge was *lead time*: the same public fact reached you before it reached the price. The intuition: watching the right ten wallets converts a public event into private notice, and that notice is the entire game.

#### Worked example: what one old coin is worth now

Read an address's history and you read time. Say a wallet bought **5 ETH at \$200** in 2016 — a \$1,000 position — and never moved it. On the explorer you would see one inflow in 2016 and then silence. At an ETH price of \$3,000, that dormant \$1,000 is now worth 5 × \$3,000 = **\$15,000** — a 15× return, sitting untouched, fully visible to anyone who looked. The day that wallet finally sends those 5 ETH to an exchange, an attentive analyst sees a long-term holder positioning to sell *the moment it confirms*, while the price has not yet reacted. The intuition: the permanence of the ledger means a wallet's entire conviction — years of holding, then the decision to move — is legible in a way no traditional market ever offers.

## Common misconceptions

**"Crypto is anonymous, so on-chain analysis is impossible."** Backwards. Crypto is *pseudonymous and permanent*, which makes it more analyzable than the traditional financial system, not less. A bank's ledger is private; a blockchain's is public. The Ronin trace happened *because* the chain is transparent. Anonymity is the exception (and even mixers only blur, they do not erase); transparency is the default.

**"If a wallet is labeled 'smart money,' I should copy it."** Survivorship bias plus bait wallets make naive copying a losing strategy. The wallet is famous *because* it already won; its labeled trades are visible to thousands of copiers, and sophisticated actors plant fake-smart wallets precisely to harvest the crowd. A label is a hypothesis to test, not an instruction to follow. A wallet that turned \$8,000 into \$200,000 on one trade tells you it got one trade right, not that its next trade is yours to free-ride.

**"Big exchange inflow means the price will crash."** Inflows are *potential* sell supply, not realized selling, and much exchange flow is market-makers and ordinary users, not whales dumping. An inflow raises the *probability* of nearby selling pressure; it does not predict a crash. Read it as a risk-management input ("supply is positioning"), confirm whether the inflow is actually followed by selling, and never trade a single flow number in isolation.

**"On-chain shows me what's happening inside Binance."** It shows you the *border* of Binance — deposits and withdrawals at its public addresses — never the trading on its internal books. Any claim about what an exchange is "doing" internally, sourced from on-chain data, is an inference dressed as an observation. Demand the on-chain fact: *coins moved to/from this exchange address*, and nothing more.

**"A clean-looking contract is a safe contract."** A token's contract can look fine and still be a honeypot (you can buy but not sell), have a hidden mint function, or be paired with a liquidity pool the deployer can pull at will. "Looks clean" is not "is safe." Safety is a checklist of specific, on-chain-verifiable permissions and distributions — the subject of the due-diligence track — not a vibe.

## The playbook: what to do with on-chain analysis

Every post in this series ends with a concrete if-then loop, because a signal you cannot act on is trivia. Here is the meta-playbook the whole discipline runs on.

**Build the reading routine (always-on).**
- *Signal:* you want lead time, not secrets. *Read:* set up the two-tool dance — an analytics platform (Arkham/Nansen/Dune) to surface flow fast, an explorer (Etherscan/Solscan) to verify it. *Action:* watch a small, deliberate set — major exchange wallets, the tokens you actually hold, a handful of validated entities. *Invalidation:* if you cannot explain *why* an address matters, stop watching it; noise is worse than nothing.

**Exchange flow (supply on tap).**
- *Signal:* an abnormal inflow of coins to spot exchanges, or a sustained drawdown of aggregate reserves. *Read:* inflow = potential sell supply arriving; reserve drawdown = float shrinking. A 12,000-BTC inflow at \$60,000 is \$720M heading toward the order book. *Action:* tighten risk on a large inflow spike; treat a multi-quarter reserve drawdown as a supportive (not sufficient) backdrop. *Invalidation / false positive:* the inflow is from a market-maker or a single internal reshuffle, and no realized selling follows. Confirm with price-and-flow follow-through, not the flow alone.

**Who is moving (accumulation vs distribution).**
- *Signal:* validated entities net-accumulating or net-distributing a token. *Read:* persistent accumulation by independently-verified wallets is more meaningful than one labeled "genius." *Action:* size *small*, use it as a tiebreaker, never as a thesis on its own. *Invalidation / false positive:* the "smart money" is a bait wallet, or you are copying a famous wallet at the top (survivorship bias). If the only reason to buy is "a labeled wallet did," do not buy.

**Is it safe (due diligence before you buy).**
- *Signal:* a new token you are tempted by. *Read:* holder concentration, contract permissions (mint, freeze, pause), liquidity-lock status, deployer history, and whether sells actually execute. *Action:* if any single check fails — top holders can dump, contract can mint, liquidity is unlocked, it is a honeypot — you walk. The asymmetry is brutal: a missed 10× costs opportunity; a rug costs principal. *Invalidation:* none — failing a safety check is a hard stop, not a probability you trade around.

**Where did it go (forensics / defense).**
- *Signal:* a hack, a scam cash-out, funds you need to trace. *Read:* follow the route — swaps to unfreezable assets, bridges across chains, deposits into mixers — and remember the entry points are always public. A \$625M theft becomes the most-watched \$625M in the world. *Action:* flag addresses, alert exchanges and off-ramps, support recovery; for an investor, *avoid* anything touching flagged funds. *Invalidation:* mixing and bridging blur the trail — never present a probabilistic attribution as certainty; say "consistent with," not "proven," until off-ramp evidence confirms it.

The thread through all five: **observe the flow, label the actors, infer the intent, act — then verify.** That is the loop in the cover figure, and it is the loop you will run in every post that follows. The ledger is public and permanent; flow leads price; and the chain lies too — so you read it carefully, and you check your read against the primary record before you ever risk a dollar.

## Further reading & cross-links

Within this series (read in roughly this order):

- [How to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow) — the forensics craft from the Ronin trace, generalized into a repeatable method.
- [What is smart money on-chain](/blog/trading/onchain/what-is-smart-money-onchain) — finding and validating wallets with a track record, and fighting survivorship bias.
- [On-chain due-diligence checklist](/blog/trading/onchain/onchain-due-diligence-checklist) — the rug-check and honeypot workflow, framed as self-defense before you buy.

Foundational background (existing deep-dives that this series builds on rather than re-deriving):

- [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) — what smart contracts and the account model are, the substrate most on-chain tooling reads.
- [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision) — where the public, permanent, pseudonymous ledger idea came from.
- [Tornado Cash and sanctioning code](/blog/trading/crypto/tornado-cash-and-sanctioning-code) — the mixer in the Ronin trace, and the hard legal questions around sanctioning a smart contract.
- [Crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset) — how stablecoin supply and exchange flows fit the bigger liquidity picture.

The discipline is one habit repeated: read the public ledger before the price catches up, label what you see, infer the intent, act — and verify everything, because on a ledger that never forgets, the chain lies too.
