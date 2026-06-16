---
title: "How Stolen Funds Are Laundered — and Where the Trail Leaks"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "An investigator's tour of the crypto laundering pipeline — swap, mix, bridge, peel, off-ramp — and the exact place each step leaks the trail back to an analyst."
tags: ["onchain", "crypto", "money-laundering", "forensics", "lazarus", "tracing", "ethereum", "tornado-cash", "bridges", "kyc"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — After a hack, stolen crypto follows a recognizable laundering pipeline (swap → mix → bridge → peel → dormancy → off-ramp), and every step that tries to break the trail leaves a fresh forensic signature an investigator can anchor on.
>
> - **What it is:** "Laundering" on-chain means breaking the link between stolen funds and a clean cash-out — not deleting the trail, which is impossible on a public ledger, but burying it under so much noise that following it costs more than it is worth.
> - **How to read it:** Walk the pipeline as a defender. The thief swaps to a liquid asset, mixes, bridges across chains, peels into small wallets, and waits — and each step leaks via amounts, timing, or pattern. Tools: Etherscan, Arkham, Chainalysis/TRM, Dune.
> - **What you do with it:** Stop chasing the obfuscation and wait at the chokepoint. Crypto must hit a KYC venue to become spendable fiat, and that is where a name attaches and most cases break.
> - **The number to remember:** Laundering *delays* a funded investigation but rarely *defeats* it — DPRK-linked actors stole ~\$1.34B in 2024 and cumulatively >\$5B, yet a growing share gets frozen or seized at the off-ramp.

On 23 March 2022, an attacker drained roughly \$625M from the Ronin bridge — the network behind the game Axie Infinity — by compromising five of nine validator keys and signing two fraudulent withdrawals. The funds left as ETH and USDC. Within hours, blockchain analytics firms had tagged the thief's addresses, and the FBI later attributed the heist to the Lazarus Group, a state-linked North Korean (DPRK) operation. What happened next was not a clean getaway. It was a months-long, visible, plodding effort to convert those tagged coins into something spendable — swapping, mixing through Tornado Cash, bridging to Bitcoin, peeling into hundreds of wallets — every move of which sat in public view on the chain.

That is the paradox at the heart of crypto laundering, and the reason this post exists. A thief can take the money in a single transaction, but they cannot *spend* it the same way. The instant funds are stolen, they become **tainted**: tagged on every analytics dashboard, watched by every exchange's compliance team, and frozen wherever a centralized issuer can reach them. The entire laundering pipeline is an attempt to scrub that taint — and the entire investigator's job is to recognize that the pipeline, step by step, leaks.

This is a defender's tour. We will follow the canonical DPRK/Lazarus route — swap, consolidate, mix, bridge, peel, sit, cash out — and at every hop we will name the **leak**: the amount that re-links, the timing correlation, the repeating pattern, and finally the KYC chokepoint where the case usually breaks. The goal is not to teach anyone how to launder money; it is to teach you to read the pipeline the way an analyst does, so that the next time you see a headline hack you can picture exactly where the trail will resurface.

![The launder pipeline from hack to off-ramp with leak points annotated at each step](/imgs/blogs/how-stolen-funds-are-laundered-1.png)

## Foundations: why stolen crypto is hard to spend

Before we trace a single hop, we need four ideas from zero. Get these and the whole pipeline reads as one long, leaky story instead of a list of tricks.

### Why a thief can't just spend stolen crypto

Cash is the launderer's dream because it is bearer, fungible, and untraceable: a stolen \$100 bill spends exactly like an honest one. Crypto is almost the opposite. Every coin's entire history is written to a public ledger that anyone can read forever. The moment funds leave a victim's contract in an exploit, three things happen, often within minutes:

- **The address gets tagged.** Analytics firms — Chainalysis, TRM Labs, Elliptic, plus public sleuths like ZachXBT — label the thief's address as `Ronin Exploiter` or `Bybit Hacker`. That label propagates to every dashboard and every exchange's screening system.
- **The funds become traceable.** Because the ledger is a graph of inputs and outputs, anyone can follow the coins forward, hop by hop, indefinitely. There is no "the trail goes cold" on-chain — the data is permanent.
- **The funds can be frozen.** Where a coin is issued by a centralized party (USDT by Tether, USDC by Circle), that issuer can blacklist an address and freeze the balance with a single transaction. Stablecoins are the most liquid escape route *and* the most freezable.

So a thief holding \$100M of stolen tokens holds something closer to a stack of marked, serial-numbered banknotes that every cashier in the world has a photo of. They cannot walk into an exchange and sell it, because the exchange's screening will flag the deposit, freeze it, and file a report. Spending it directly is off the table. That is the problem laundering tries to solve.

### What "laundering" means on-chain

In traditional finance, laundering has three textbook stages: **placement** (get the dirty cash into the financial system), **layering** (move it through enough transactions that the audit trail becomes impractical to follow), and **integration** (bring it back out looking legitimate). On-chain, the funds are already "placed" — they are already digital and in the system the instant they are stolen. So the on-chain game is almost entirely **layering**: pile on so many hops, swaps, mixes, and bridges that an investigator following the trail either runs out of budget or loses the thread in the noise.

The crucial word is *budget*. On a public ledger, laundering can never truly delete the link — the data is permanent and the graph is fully connected. What it can do is **raise the cost of following it** until the trail outlasts the investigator's attention, funding, or jurisdiction. Laundering is a delay tactic dressed up as an escape. An analyst who internalizes that — "they are buying time, not erasing the link" — reads the whole pipeline differently.

### The off-ramp problem: crypto is not fiat

Here is the constraint that shapes everything. The thief's end goal is almost never to hold ETH forever; it is to convert the loot into **fiat** — dollars, won, yuan — that pays for things in the physical world. And crypto-to-fiat conversion has to happen *somewhere*: a centralized exchange, a payment processor, an over-the-counter (OTC) desk, a peer-to-peer trade that eventually touches a bank. Every one of those venues, in any regulated jurisdiction, runs **KYC** — Know Your Customer — identity verification: a name, a government ID, a selfie, a bank account.

That is the **chokepoint**. On-chain, the funds are pseudonymous: addresses, not names. The off-ramp is the one place where the pseudonymous chain meets the identified financial system. Every gram of obfuscation in the pipeline exists to get the funds to an off-ramp *without* the deposit being flagged. And the investigator's whole strategy can be summarized in one line: don't chase the obfuscation, wait at the chokepoint.

### "Tainted" funds and how exchanges flag them

How does an exchange actually "know" a deposit is dirty? Through **taint analysis**. Analytics firms maintain a continuously updated graph of which addresses are associated with hacks, sanctioned entities, scams, and darknet markets. When funds flow out of a tagged address, the taint *propagates* forward along the graph — every downstream address that received those coins inherits some degree of association. When such an address tries to deposit at a screened exchange, the screening system computes the deposit's exposure to flagged sources and, above a threshold, freezes it and files a Suspicious Activity Report.

Taint is not binary and it is not perfect — heavy mixing genuinely dilutes it, and false positives sting innocent users who happened to receive a mixed coin. But the direction of travel matters: every year the labeled set grows, the heuristics improve, and the on-ramp/off-ramp screening tightens. The chain remembers, and the index of who-touched-what only gets better. That is the headwind every launderer runs into.

There are two competing ways analytics firms actually compute taint, and the difference matters for a defender reading a report. **Poison (or "full taint")** says: once an address touches stolen funds, the *entire* address is contaminated, and all downstream funds inherit the taint regardless of how much clean money was mixed in. It is aggressive and catches everything, but it over-flags — a clean exchange that received one tainted deposit would light up. **Haircut taint** is proportional: if a wallet holds 90% clean funds and receives 10% tainted, its outputs carry 10% taint, and that fraction dilutes further with each clean mixing step. Most commercial screening uses a blend, with thresholds tuned so that a deposit only freezes when its tainted exposure exceeds some percentage. The launderer's whole pipeline is, in effect, an attack on the *haircut* model — every clean swap and every mixer hop is an attempt to dilute the tainted fraction below the freeze threshold. The defender's counter is that the *graph* of who-touched-what is preserved perfectly even when the *percentage* gets diluted, so a determined investigator can always re-walk the path the dilution tried to hide.

### The forensic stack: the tools that turn leaks into alerts

The leaks we will catalog are only useful if something is watching for them. In practice that "something" is a stack of tools, and it helps to know which layer catches which leak:

- **Block explorers (Etherscan, Solscan, Blockchair, Tronscan)** — the raw ledger. They show every transaction, every event log, every token transfer, with exact amounts and timestamps. This is where you read a `Swap()` event or confirm a bridge deposit. Free, public, and the ground truth.
- **Attribution and clustering platforms (Arkham, Nansen, Chainalysis Reactor, TRM, Elliptic)** — the *labels*. They cluster addresses into entities using heuristics, tag known exchanges and exploiters, and render a money flow as a graph rather than a list of hashes. This is where "address `0xA11ce…`" becomes "the Ronin exploiter."
- **Query engines (Dune, Flipside)** — the *patterns at scale*. SQL over decoded on-chain data lets an analyst pull "every 100-ETH deposit to this mixer in this window" or "every bridge transfer above \$25M last week" in seconds. This is how a peel chain's regularity or a mixer's deposit burst gets quantified rather than eyeballed.
- **Public investigators (ZachXBT and others)** — the human layer. Independent sleuths often tag and trace hacks faster than anyone, posting the exploiter address within minutes and following the funds in public threads. Their tags frequently seed the commercial labels.

A leak only matters when one of these layers turns it into a signal. The whole point of the pipeline-and-leak framing is that *every* obfuscation step produces a signal in *some* layer — and the analyst who knows which layer catches which leak never has to chase the money blind.

#### Worked example: the marked-bills math

Say a hacker steals \$100M of a project's token. The instant it lands in their wallet, every major analytics provider tags the address. If they try to sell even \$2M of it at a screened exchange, the deposit is frozen on arrival — call it a 100% loss on that tranche. To realize *any* value, they must first spend on average days of effort and a stack of transaction fees per million dollars laundered just to make a deposit that *might* clear screening. The economics are brutal: stealing is one transaction; cashing out is a months-long operating expense with a high failure rate. That asymmetry — instant theft, expensive and leaky exit — is the defender's structural advantage.

## The launder pipeline, step by step — and where each step leaks

Now the route. We will use the canonical DPRK/Lazarus pattern as documented across the Ronin, Harmony, and Bybit post-mortems, because it is the most prolific and the best-studied. Each step has a *purpose* (the link it tries to break) and a *leak* (the signature it leaves). We cover each only to the depth needed to recognize and disrupt it.

### Step 1 — SWAP stolen tokens to a native asset (ETH)

Stolen funds usually arrive as a grab-bag: a project's own token, some stablecoins, some ETH. Two problems with that mix. First, project tokens are **illiquid** — try to sell \$40M of a thin token and the price collapses, and the sale is obvious. Second, stablecoins are **freezable** — Tether and Circle can blacklist the address before it moves. So the first move is to consolidate into ETH (or the chain's native coin), which is deeply liquid and cannot be frozen by a central issuer. The swap happens on a **DEX** — a decentralized exchange like Uniswap, a smart contract that lets anyone trade tokens with no KYC and no account.

**Where it leaks:** every DEX swap emits a public `Swap()` event recording the exact tokens-in, the pool, the amount-out, and the timestamp. There is no privacy here — the DEX is a contract on the same public chain. An analyst reads the swap, notes "the \$40M of token A became 18,000 ETH at 14:07," and keeps following the *same funds* now denominated in ETH. The asset changed; the visibility did not. If anything, dumping a large position into a thin pool is a loud signal — the price impact and the size flag the address to anyone running a large-swap monitor.

There is a second leak hiding in the *economics* of the swap. Stolen project tokens are usually illiquid, so selling tens of millions of dollars of them moves the price against the seller — **slippage**. To avoid one catastrophic price impact, the thief must either split the sale across many transactions (manufacturing the round-number regularity that clustering loves) or accept a large, conspicuous slippage cost that itself flags the trade. A normal trader sizes a swap to minimize cost; a thief dumping a stolen position is forced into either a loud single dump or a regular series of dumps, and *both* shapes are detectable. The DEX charges no KYC, but the pool's liquidity charges a tax in visibility. And because the swap is atomic and on-chain, the analyst sees not just the result but the *full order of operations* — which pool, which slippage, which route — and can often infer the thief was unloading a position rather than trading, just from the shape of the trades. Liquidity, the thing that made ETH attractive, is also the thing that makes the swap loud.

![Stolen tokens swapped to ETH through a DEX leaving public swap events](/imgs/blogs/how-stolen-funds-are-laundered-2.png)

#### Worked example: \$100M swapped to ETH leaves a perfect record

A thief holds \$40M of token A and \$60M of token B. They route both through a DEX over an afternoon, ending with roughly \$100M in ETH (say ~45,000 ETH at \$2,200). To them this feels like progress — they now hold the most liquid, least-freezable asset on the chain. To an analyst it is a gift: the DEX emitted a Swap event for every leg, each stamped with amount and time. The \$40M leg and the \$60M leg are both readable, the resulting ETH is a single consolidated balance, and the trail is unbroken. The swap converted the *form* of the money without converting its *traceability*. Net result for the investigator: one clean ETH balance to follow instead of two messy token positions.

### Step 2 — consolidate, then split

With everything in ETH, the thief often **consolidates** scattered loot into a few control addresses (to manage it), then begins to **split** it into chunks sized for the next steps — typically the fixed denominations a mixer accepts. Consolidation is a convenience; splitting is preparation.

**Where it leaks:** consolidation is the *opposite* of obfuscation — pulling many addresses into one is a giant arrow pointing at the controlling wallet, and it is exactly how analysts perform **address clustering**. The two workhorse heuristics are the **common-input-ownership** heuristic (on UTXO chains like Bitcoin, all the inputs that fund a single transaction are almost certainly controlled by one entity, because you need every input's private key to sign) and the **co-spend / behavioral** heuristics on account chains (addresses that consistently fund each other, share a funding source, or move in lockstep get clustered into one entity). When a thief consolidates ten scattered ETH balances into three control wallets, those three are now provably linked, and any prior attempt to keep them separate is undone in one move.

Splitting into round, identical chunks is also a tell: real economic activity has messy amounts — a Uniswap user does not swap exactly 100.000 ETH ten times in a row — while \$500-ETH-times-100 screams "preparing for a mixer." Regularity is a fingerprint, and this step manufactures regularity. Worse for the launderer, the splitting transactions themselves are timestamped and sequential, so an analyst can reconstruct the *order* of operations and infer intent: "these ten 100-ETH outputs in three minutes are not payments, they are mixer staging." Splitting is meant to look like dispersion; to a clustering tool it looks like a labeled, ordered to-do list.

For the full mechanics of how clustering deanonymizes addresses, see [address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics) — the same techniques that follow smart-money wallets also un-split a launderer's fan-out.

### Step 3 — MIX (Tornado Cash) and where the amount and timing leak

A **mixer** is the laundering tool most people name first. Tornado Cash, the best-known Ethereum mixer, works on a pool model: you deposit a fixed denomination (1, 10, or 100 ETH), and later you withdraw the same denomination to a *fresh* address using a cryptographic proof that you deposited *something* — without revealing *which* deposit. If 500 people have each deposited 100 ETH, your withdrawal could in principle have come from any of them. That crowd is the **anonymity set**, and it is meant to sever the on-chain link between the dirty deposit address and the clean withdrawal address.

This is the strongest single link-breaker in the pipeline, which is exactly why it is the most studied — and the most leaky in practice. We covered the mechanics in depth in [mixers, CoinJoin, and obfuscation](/blog/trading/onchain/mixers-coinjoin-and-obfuscation); here we focus on the leaks.

**Where it leaks — amount and timing:**

- **Amount.** The pool's privacy only holds if your deposit hides among *many identical* deposits. Push an unusual total through and the math gives you away. If you deposit 100 chunks of 100 ETH within a window when only a handful of other 100-ETH deposits exist, the "set" of plausible sources is tiny, and a withdrawal of a matching aggregate to a fresh cluster re-links statistically.
- **Timing.** Deposits and withdrawals are timestamped. A burst of deposits followed shortly by a burst of withdrawals of the same denomination, to addresses that then behave identically, lets an analyst **correlate** the two ends despite the proof. The proof hides *which deposit*, not *that a coordinated batch went in and a coordinated batch came out*.
- **The sanction overlay.** Tornado Cash's smart contracts were sanctioned by the US Treasury's OFAC on 8 August 2022 (see [Tornado Cash and sanctioning code](/blog/trading/crypto/tornado-cash-and-sanctioning-code)). After that, *interacting with the mixer at all* became a flag — and compliant exchanges treat any funds with Tornado exposure as high-risk, even funds with a large anonymity set. The mixer that was supposed to grant privacy now stamps a scarlet letter.

The mixer's anonymity-set mechanics — how a pool of N identical deposits hides one withdrawal, and how the set shrinks under bursty volume — are diagrammed in detail in the dedicated [mixers post](/blog/trading/onchain/mixers-coinjoin-and-obfuscation); we lean on that figure here rather than re-drawing it, and focus on the leaks specific to laundering large stolen sums.

#### Worked example: \$50M through a mixer that timing re-links

A thief pushes \$50M of ETH — about 22,700 ETH at \$2,200 — through a mixer in 227 chunks of 100 ETH over two days. The chunking is meant to hide each 100-ETH deposit in the crowd. But two things leak. First, **volume**: 227 deposits of one denomination from clustered source addresses is a spike far above the pool's organic deposit rate, so the plausible anonymity set for these specific chunks is small. Second, **timing**: a matching burst of 227 withdrawals of 100 ETH lands over the next few days into a set of fresh addresses that then *consolidate again* — and consolidation re-clusters them. An analyst who logs the deposit timestamps and amounts and the withdrawal timestamps and amounts can probabilistically re-link a large fraction of the \$50M. The mixer raised the cost of the trace; it did not erase it. And every withdrawn coin now carries Tornado taint, so the \$50M is *harder* to off-ramp than the clean ETH was.

### Step 4 — BRIDGE across chains, and where amount and timing correlate

A **bridge** moves value from one blockchain to another — say ETH on Ethereum to "wrapped" ETH on another chain, or ETH converted to BTC on Bitcoin. For a launderer the appeal is **discontinuity**: an analyst comfortable with Ethereum's tools now has to follow the money onto a different ledger, with different explorers, different conventions, and (the thief hopes) a different investigator. Lazarus has repeatedly bridged ETH to Bitcoin precisely because Bitcoin's UTXO model and separate tooling add friction to the trace.

We cover the mechanics — lock-mint, burn-mint, and liquidity-network bridges, and the USDT rails on Tron — in [cross-chain tracing: bridges and the USDT rails](/blog/trading/onchain/cross-chain-tracing-bridges-and-the-usdt-rails). Here, the leaks.

**Where it leaks — amount and timing correlation across chains:** a bridge does not make the money disappear; it makes the *same value* reappear on another chain. The deposit on the source chain and the payout on the destination chain are two public events linked by **value** and **time**. If 11,500 ETH locks into a bridge contract at 09:14, and ~11,490-worth of value (minus fees) appears on the destination chain at 09:17 to a fresh address, those two events *match* on amount and clock. The investigator's move is exactly that: join the two sides by amount and timestamp. Bridges that mint a 1-to-1 wrapped asset leave the cleanest match (a Lock event paired with a Mint event); even liquidity-network bridges, which only show "deposit into pool A, payout from pool B," leak the amount-and-time pair. The trail jogs onto a new chain; it does not break.

The friction a bridge adds is real but practical, not cryptographic. It is the friction of an analyst having to switch tooling, learn a second chain's conventions, and — historically — manually stitch the two sides together. That manual step is exactly what modern cross-chain analytics now automate: platforms maintain a continuously updated ledger of bridge deposits and payouts across dozens of chains and surface the matching pairs as candidate links. So the bridge buys the launderer the time it takes an analyst to *notice and join* the two events — minutes to hours on a well-monitored bridge — not permanent escape. The most-monitored bridges (the big lock-mint and burn-mint protocols) are effectively transparent; the harder cases are obscure liquidity-network bridges and chain-hops into ecosystems with thin tooling, which is why Lazarus has favored ETH-to-Bitcoin hops. Even there, the amount-and-time signature survives — it just costs more analyst hours to recover. The [cross-chain tracing post](/blog/trading/onchain/cross-chain-tracing-bridges-and-the-usdt-rails) walks through the matching procedure on the USDT-on-Tron rails, the single highest-volume illicit cash-out corridor.

#### Worked example: \$30M bridged ETH-to-another-chain, matched by amount

A thief bridges \$30M of ETH — about 13,600 ETH — to another chain to shake an Ethereum-focused trace. The bridge takes a small fee, so ~13,580 ETH-equivalent arrives on the destination side a few minutes later. An analyst monitoring the bridge contract sees a 13,600-ETH deposit at 11:02 and, watching the destination chain, sees ~\$29.9M of value credited to a brand-new address at 11:06. The amounts match within the fee, the timestamps are minutes apart, and no other transfer of that size crossed in that window. The two events are joined with high confidence. The \$30M is now on a new chain, but the link is intact — and the destination chain's funds inherit the source taint the moment the match is made.

### Step 5 — PEEL CHAIN into many small wallets

Once the funds are mixed and bridged, the thief often runs a **peel chain**: forward most of a balance to a fresh address while "peeling" off a small slice toward a cash-out, then repeat — hop after hop, sometimes hundreds of times. Visually it looks like the money is dispersing into a fog of wallets, which is the intended impression. The peeled slices drift toward off-ramps in amounts small enough to slip under screening thresholds; the big remainder keeps marching forward.

**Where it leaks — the fingerprint of regularity:** a peel chain is one of the easiest patterns to follow once you recognize it, because it is *regular*. The main trail stays exactly **one address wide** — at each hop a single large output (the remainder) plus a single small output (the peel). That single-heir structure is trivial to walk: just follow the big remainder. And the peel slices are often the *same size* hop after hop, because the thief automates it; that constant slice is a literal fingerprint. Analysts script a "follow the largest output" walk down the chain and collect every peeled cash-out destination along the way. Dispersion that *looks* like chaos is actually a metronome.

The automation cuts both ways, and that is the deeper point. The launderer scripts the peel because doing hundreds of hops by hand is impractical — but a script produces *identical* behavior at every hop: the same peel amount, the same gas settings, the same timing cadence, often funded from the same gas source. Each of those is a feature an analyst can cluster on. A peel chain is therefore self-defeating in a specific way: the very automation that makes it cheap to *run* makes it cheap to *recognize and walk*, because the pattern that an algorithm emits is exactly the kind of low-entropy, repetitive signature that detection algorithms are built to catch. And every peeled cash-out address is a labeled lead — when one of those small slices hits a KYC venue and gets frozen, it confirms the whole chain was a laundering operation and hands the investigator a thread back to the main trail. The fan-out that was meant to scatter the evidence instead multiplies it: forty peel destinations are forty chances to catch the thief at an off-ramp.

![A peel chain forwarding a large remainder and peeling a constant slice at each hop](/imgs/blogs/how-stolen-funds-are-laundered-6.png)

#### Worked example: a peel chain you can walk in an afternoon

Start with 30,000 ETH on a tainted wallet. Hop 1: peel 500 ETH toward a cash-out, forward 29,500 ETH to a fresh address. Hop 2: peel 500 ETH, forward 29,000. Hop 3: peel 500, forward 28,500. The pattern is identical every hop. After 40 hops the thief has shed 20,000 ETH — about \$44M at \$2,200 — into 40 small cash-out-bound addresses, while 10,000 ETH still rides the main trail. An analyst anchors on the 30,000-ETH start, follows the single largest output at each hop (a five-minute scripted walk), and recovers the entire chain plus all 40 peel destinations. The "fog of wallets" turns out to be a single thread with 40 labeled exits hanging off it. The regularity that made it cheap to run is the regularity that makes it cheap to trace.

### Step 6 — DORMANCY: sit for months, but the chain remembers

When the heat is on — fresh headlines, active investigators, frozen tranches — a sophisticated actor will simply **stop** and let the funds sit, sometimes for months or years. The bet is that investigators get reassigned, budgets run out, attention moves on, and by the time the funds move again nobody is watching.

**Where it leaks — the chain never forgets:** dormancy attacks the investigator's attention span, not the data. The stolen balance sits in plain sight the entire time, parked on a still-tagged address that every dashboard still labels and every freeze list still includes. The label does not expire. And the *moment* the funds finally move — even years later — that spend transaction is a fresh, timestamped event that re-arms every alert pointed at the address. Modern analytics watch dormant tagged balances indefinitely; a dormant-balance-just-moved alert is one of the highest-signal events an analyst can get, precisely because it is rare and intentional. Waiting buys time. It does not buy escape.

There is a subtle second-order effect that works *against* the launderer here. While the funds sit, the *defender's* side improves: the labeled set grows, clustering heuristics get better, new sanctioned-address lists are published, the exchange screening tightens, and any co-conspirators or mule accounts that touched the funds earlier may have been caught in the interim. A trail that was 80% resolvable when the hack happened can be 95% resolvable two years later, because two more years of analytics progress have been applied to the same permanent data. Dormancy is one of the few laundering tactics that makes the launderer's position *worse* over time — they freeze their funds while the investigators' tools keep advancing. The only thing waiting reliably accomplishes is to outlast a *poorly funded* investigation; against a well-resourced one with a standing watch list, it is a gift of time to the defender.

It also creates a recovery opportunity. Because the balance is static and visible, law enforcement can pre-stage: line up the freeze requests, brief the likely off-ramp venues, and prepare the seizure paperwork so that when the funds finally twitch, the response is immediate rather than reactive. The dormant period is when the slow machinery of subpoenas and international cooperation can be readied in advance of the spend. The thief thinks they are hiding; they are actually giving the defenders time to set the trap at the chokepoint.

![A dormant stolen balance sits tagged for months then triggers alerts when it finally moves](/imgs/blogs/how-stolen-funds-are-laundered-5.png)

#### Worked example: \$66M parked for eight months wakes the alarms

A thief parks 30,000 ETH — about \$66M — on a single tagged address and waits eight months without moving a coin. To them it feels like the trail has gone cold. In reality, the balance is *more* exposed than active funds: it is a large, static, labeled target that analytics firms keep on a watch list precisely *because* it is dormant. When the thief finally sends the first 5,000 ETH (~\$11M) toward the next laundering hop, the outflow fires an instant alert with a precise timestamp, the taint propagates to the new address, and the trace resumes exactly where it paused — now with the added signal that the actor just tipped their hand by reactivating. The eight-month wait cost the thief eight months and gained them nothing but a louder re-entry.

### Step 7 — OFF-RAMP: the KYC chokepoint where most cases break

Every prior step was layering — buying time, breaking links, adding noise. None of it produces a single dollar the thief can spend. For that, the funds must reach an **off-ramp**: a venue that converts crypto to fiat. The realistic options are a **centralized exchange** (deepest liquidity, strictest KYC), a **P2P trade** (find a counterparty directly — but the counterparty, or their bank, has KYC somewhere downstream), an **OTC desk** (large private trades — regulated desks KYC; unregulated ones are themselves a target), or a **crypto debit card / payment processor** (the issuer is regulated and KYC'd at signup). In every realistic path, the boundary between pseudonymous crypto and identified fiat is crossed at a venue that, in any jurisdiction with functioning regulation, knows who its customer is.

**Where it leaks — identity attaches:** this is *the* chokepoint and the reason most cases ultimately break here. On-chain the funds are addresses; the instant they hit a KYC deposit address, a real-world identity attaches — a name, an ID, a bank account, an IP. The exchange's screening flags the tainted deposit, freezes it, files a report, and (under subpoena or a freeze request) hands investigators the account holder's identity and the destination bank. All the upstream obfuscation collapses into one fact: *someone with a name tried to cash this out here.* That is why the playbook is "wait at the off-ramp" — you let the launderer do all the work of moving the money to the one place it can betray them.

**Why the chokepoint is so hard to avoid.** The launderer's dream is a venue with deep fiat liquidity and no identity check. That venue is increasingly mythical. Walk the realistic options and each one has a KYC gate somewhere:

- **Major centralized exchanges** have the deepest liquidity and the strictest compliance — full KYC, transaction monitoring, and on-chain screening on every deposit. A large tainted deposit here is the *most* likely to freeze. Thieves use them only when the funds look clean enough to slip the threshold, which is exactly the bet that fails when the trail survived.
- **Smaller or offshore exchanges** have looser controls, but they also have thinner liquidity (you cannot dump \$50M without moving the price and drawing attention), worse banking relationships (their fiat partners impose downstream KYC), and a habit of becoming enforcement targets themselves — when a lax exchange gets sanctioned or seized, its records, including the launderer's, become evidence.
- **P2P and OTC** feel KYC-free because you trade with a person, not a platform. But the counterparty has to get fiat from *somewhere* — a bank account, a payment app, a cash pickup — and that endpoint carries identity. A counterparty who knowingly takes hacked funds is a co-conspirator and a future cooperating witness.
- **Crypto cards and payment processors** let you spend crypto "directly," but the card issuer is a regulated financial institution that KYC'd you at signup and monitors transactions. Spending here is just a slower walk to the same identity gate.
- **Nested services and "money mules"** — accounts opened under borrowed or stolen identities at a legitimate exchange — are the launderer's real workaround. But mule accounts are themselves a detectable pattern (sudden large crypto deposits into a freshly KYC'd account with no trading history), and when one is caught it unravels the network behind it.

There is no door out of the maze that does not pass a guard. That is the structural reason laundering delays rather than defeats: the exit is a chokepoint by design of the regulated financial system, not by the cleverness of any one investigator.

![Every off-ramp path converges on a KYC identity gate where the case breaks](/imgs/blogs/how-stolen-funds-are-laundered-3.png)

#### Worked example: a \$2M cash-out that attaches a name and breaks the case

After mixing, bridging, and peeling, a thief routes \$2M of laundered ETH to a deposit address at a mid-tier centralized exchange, hoping it is small enough and "clean" enough to clear screening. The exchange's compliance system computes the deposit's exposure to the tagged hack cluster — even diluted, the Tornado and bridge hops left a measurable trail — and flags it. The \$2M is frozen on arrival. More importantly, the deposit is tied to a verified account: a name, a government ID, a selfie, and a linked bank account already on file from signup. Investigators now have what the entire on-chain trace could never give them — a *person*. One \$2M cash-out attempt converts a months-long pseudonymous chase into a named subject and a frozen balance. This is the single most common way large hacks resolve: not by following every hop, but by being ready at the gate.

## How DPRK/Lazarus launders at scale — the signature

The pattern above is not hypothetical; it is the documented Lazarus signature, repeated across Ronin (\$625M, 2022), Harmony, Atomic Wallet, and the largest hack in history, **Bybit** (\$1.46B, February 2025). The DPRK is by a wide margin the most prolific crypto-stealing actor on Earth. According to Chainalysis, DPRK-attributed groups stole roughly **\$1.34B in 2024** alone, and **cumulatively more than \$5B** over recent years.

![All crypto stolen per year versus the DPRK-attributed share, 2020 to 2024](/imgs/blogs/how-stolen-funds-are-laundered-4.png)

The Lazarus signature is industrial: dump stolen tokens to ETH on a DEX within hours of the hack; run the ETH through a mixer in fixed denominations; bridge a large fraction to Bitcoin to break tooling continuity; peel through long chains of intermediary wallets; let tranches go dormant for weeks; and feed the cleaned output to off-ramps — historically including OTC brokers and over-collateralized cash-out networks in jurisdictions with weaker enforcement. The scale is what makes it both effective and traceable: moving over a billion dollars leaves an enormous, repetitive footprint that analytics firms have now seen enough times to recognize on sight.

#### Worked example: laundering \$1.34B in a single year is a full-time operation

Suppose a state-linked group must launder \$1.34B in one year — the rough 2024 DPRK figure. That is about **\$3.7M every single day**, 365 days a year, through swaps, mixers, bridges, and peel chains, while every analytics firm and several intelligence agencies watch. The sheer throughput forces *repetition*: the same denominations, the same bridge routes, the same peel cadence, the same off-ramp networks — because bespoke laundering of every dollar at that volume is impossible. Repetition is signature. And the off-ramp math is unforgiving: to convert even a fraction of \$1.34B to usable fiat, the group must make thousands of deposits at venues that screen, and a rising share of those — hundreds of millions of dollars — gets frozen or seized. The headline "they stole \$1.34B" obscures the footnote that matters: how much they actually *cashed out* clean is a far smaller, and shrinking, number.

### Reading the outcome column: what actually happens to stolen funds

The most instructive field in any hacks database is not the dollar amount — it is the *outcome*. Look across the biggest exploits and a pattern emerges that confirms the leak thesis:

- **Bybit (\$1.46B, 2025)** — attributed to Lazarus, the largest hack ever; *mostly laundered*, but at a punishing pace under full public surveillance, with significant tranches frozen at venues and a coordinated industry effort to blacklist the addresses in real time.
- **Ronin (\$625M, 2022)** — Lazarus; *partial seizure*. US authorities recovered a meaningful fraction by working the off-ramp chokepoint after the funds were tracked through Tornado Cash and bridges.
- **Poly Network (\$611M, 2021)** — *fully returned*. The attacker discovered, as a defender would predict, that spending \$611M of fully-traced funds was impossible, and negotiated a complete return.
- **Euler Finance (\$197M, 2023)** — *fully returned*. Same story: the exploiter faced an un-launderable sum and a public trace, and gave it back in exchange for a bounty and no prosecution.
- **Mango Markets (\$117M, 2022)** — the perpetrator was *identified, settled, and convicted*, because even a self-described "legal" market manipulation left a fully-named on-chain trail.

The throughline: when funds are fully traced and the off-ramp is watched, the realistic outcomes are *returned*, *frozen*, *seized*, or *stuck* — far more often than *cleanly cashed out*. Laundering succeeds at scale only for actors who (a) accept enormous friction and partial losses, and (b) have access to off-ramps beyond the reach of cooperative enforcement. For everyone else, a public trace plus a watched chokepoint turns a heist into a hostage situation where the thief is holding funds they cannot spend.

#### Worked example: why returning \$611M can be the rational move

Take an attacker sitting on \$611M of fully-traced, freshly-tagged funds (the Poly Network case). Their realistic clean-cash-out rate, after mixing fees, bridge fees, frozen tranches, and off-ramps that reject most of it, might be a small fraction — call it pennies on the dollar, and even those carry prosecution risk. Against that, a negotiated return in exchange for a bounty and no charges can be worth more in *expected, spendable, legal* dollars than the entire \$611M is worth as un-spendable tainted crypto. The math that looks absurd from the outside — "give back \$611M?" — is rational once you price in the leak at every step: traced funds are worth a fraction of their face value, and the cleaner exit is sometimes to hand them back.

## Why laundering delays but rarely defeats a funded investigation

Pull the threads together and a single thesis emerges. Every laundering step is a **link-breaker that leaks**:

| Step | Link it tries to break | The leak |
|---|---|---|
| Swap to ETH | asset identity | public Swap events, amounts match |
| Mix | deposit-to-withdrawal link | amount + timing correlation |
| Bridge | single-chain continuity | value + time match across chains |
| Peel | one fat trail | constant slice, single-heir fingerprint |
| Dormancy | investigator attention | balance is static and tagged; spend re-arms alerts |
| Off-ramp | crypto-to-fiat boundary | identity attaches (the case breaks) |

![A matrix mapping each launder step to what it breaks, where it leaks, and the investigator move](/imgs/blogs/how-stolen-funds-are-laundered-7.png)

The asymmetry runs entirely in the defender's favor on a *funded* case. The thief must succeed at every step and stay clean all the way to fiat; the investigator only needs the trail to survive to the chokepoint, where identity attaches whether or not every intermediate hop was perfectly resolved. Laundering's real product is *delay* — months of cost and friction — not *escape*. The cases that genuinely defeat investigators are not the ones with the cleverest mixing; they are the ones nobody had the budget to pursue, or that cashed out through a jurisdiction beyond reach. Sophistication buys time; lack of enforcement buys escape. That distinction is why "follow the money to the off-ramp" beats "untangle every hop" as a strategy.

It helps to price the asymmetry explicitly. The *cost of stealing* is one transaction and the *cost of tracing* a single hop is, for a competent analyst with the right tools, minutes. But the *cost of laundering* is enormous and front-loaded: mixer fees, bridge fees, the slippage on every dump, the operational overhead of running hundreds of peel hops, the months of dormancy with capital frozen, and — the dominant term — the large fraction of value lost to freezes and rejected deposits at the off-ramp. A launderer who recovers even half of a large stolen sum as clean, spendable fiat is doing well; many recover far less. Meanwhile the defender's marginal cost per additional hop *falls* over time as tooling improves and labels accumulate. Two cost curves diverge: the launderer's rises with every step they take to hide, and the defender's falls with every year of analytics progress. On any timescale long enough to matter, the curves cross in the defender's favor — which is the economic statement of "delays but rarely defeats."

This is also why the *deterrent* value of on-chain forensics exceeds its recovery value. Even when funds are not fully recovered, the near-certainty that a large hack will be traced, that the addresses will be sanctioned, and that the off-ramps will be watched, raises the expected cost of cashing out so high that some attackers return the funds outright (Poly Network, Euler) and many others net only pennies on the dollar. A public ledger does not stop the theft, but it makes the *proceeds* of theft a depreciating, watched, partially-frozen asset — which is a very different thing from cash in a duffel bag.

### The role of stablecoin freezes and sanctioned-address lists

Two tools tilt the field further toward defenders, and both have grown sharply more potent since 2022.

**Stablecoin freezes.** Tether and Circle can freeze USDT and USDC at any address with a single on-chain transaction. When stolen funds — or laundered proceeds — sit in or pass through a frozen-able stablecoin, the issuer can lock them on request from law enforcement. Tether alone has frozen well over a billion dollars across thousands of addresses tied to hacks, sanctions, and fraud. This is why sophisticated thieves swap *out* of stablecoins into ETH or BTC early (Step 1) — but it also means any attempt to use stablecoins as the liquid off-ramp re-exposes them to an instant freeze. See [stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) for how that issuer control works.

**Sanctioned-address lists.** OFAC publishes specific blockchain addresses on its SDN (Specially Designated Nationals) list — it did so for Lazarus addresses and for the Tornado Cash contracts in August 2022. Once an address or contract is sanctioned, every compliant US-touching entity is legally barred from transacting with it, and screening systems hard-block any funds with exposure to it. This converts a technical taint into a legal wall: even funds that are *statistically* clean become commercially radioactive if they carry sanctioned exposure. The launderer's clean-looking ETH is worth far less than its face value if no compliant venue will touch it.

## How to read it: an investigator's walkthrough

Let us run one consolidated trace, naming the leak at each hop, exactly as an analyst would using Etherscan to read raw transactions, Arkham or Chainalysis to read labels and clustering, and a Dune query to pull batch patterns. (For the foundational mechanics of following a single transaction, see [how to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow); for a full end-to-end case, see [tracing a real flow end-to-end](/blog/trading/onchain/case-study-tracing-a-real-flow-end-to-end).)

**Hop 0 — the hack.** \$100M leaves the victim contract to address `0xA11ce…`. On Etherscan you see the exploit transaction; within hours Arkham labels `0xA11ce…` as the exploiter. *Leak: the address is born tagged. We start with a labeled anchor.*

**Hop 1 — the swap.** `0xA11ce…` routes \$40M of token A and \$60M of token B through a DEX, ending with ~45,000 ETH. Etherscan shows the `Swap()` events with exact amounts and timestamps. *Leak: we read the swaps and now follow ~45,000 ETH instead of two token piles. The trail never paused.*

**Hop 2 — consolidate and split.** The ETH consolidates into a few control wallets, then splits into 100-ETH chunks. *Leak: consolidation clusters the control wallets via the co-spend heuristic; the round 100-ETH chunks scream "mixer prep." We pre-position to watch the mixer's deposit feed.*

**Hop 3 — the mixer.** 227 chunks of 100 ETH deposit into the mixer over two days; 227 withdrawals of 100 ETH land over the next few days to fresh addresses. *Leak: the deposit burst is far above the pool's organic rate, the withdrawal burst matches it in size and timing, and the fresh addresses re-consolidate. We probabilistically re-link most of the \$50M and tag every withdrawal address with Tornado exposure.*

**Hop 4 — the bridge.** \$30M of the re-linked ETH bridges to another chain; ~\$29.9M appears on the destination side minutes later. *Leak: the deposit and payout match on amount and clock with no other transfer of that size in the window. We join the two sides and carry the taint onto the new chain.*

**Hop 5 — the peel chain.** 30,000 ETH runs a 40-hop peel, shedding 500 ETH per hop. *Leak: single-heir remainder + constant 500-ETH slice. We script "follow the largest output" and recover the main trail plus all 40 peel destinations in minutes.*

**Hop 6 — dormancy.** A \$66M tranche sits for eight months. *Leak: the tagged balance stays on our watch list; the first 5,000-ETH spend fires an instant alert and the trace resumes.*

**Hop 7 — the off-ramp.** \$2M of the peeled funds hits a KYC exchange's deposit address. *Leak: screening flags the exposure, freezes the \$2M, and the verified account attaches a name, ID, and bank. The case breaks here.*

Read top to bottom, the lesson is unmistakable: the thief did seven sophisticated things, and the trail leaked at all seven. The obfuscation raised our cost and bought them months — but on a funded case, the funds walked themselves to the one venue that could name their owner.

## Common misconceptions

**"Mixers make funds untraceable."** No — they raise the cost and add uncertainty, but they leak through amount and timing, and post-2022 they stamp funds with sanctioned-contract exposure that compliant venues hard-block. A large, bursty, repetitive deposit pattern (exactly what a big hack produces) shrinks the effective anonymity set and is often statistically re-linkable. Heavy mixing of *small, organic* amounts over long periods is genuinely hard; mixing \$50M in two days is not.

**"Once they bridge to another chain, the trail is dead."** No — the bridge re-emits the same value on the new chain, and the deposit/payout pair matches on amount and time. The trail jogs onto different tooling, which adds analyst friction, but it does not break. Cross-chain analytics now stitch these matches automatically.

**"Crypto is mostly used for crime, so laundering is easy."** No — illicit activity is a small share of on-chain volume (Chainalysis estimates on the order of **~0.14%** for 2024, in a roughly 0.1–0.6% historical band). The *vast majority* of flow is legitimate, which is precisely why dirty funds stand out: they are statistical anomalies moving against a sea of normal activity, and the off-ramps that handle the legitimate majority all screen.

**"If they just wait long enough, the trail goes cold."** No — the chain is permanent and the label does not expire. Dormancy attacks the investigator's attention, not the data. A dormant tagged balance is a high-priority watch item, and its first movement is among the loudest signals an analyst can receive.

**"They'll just use a no-KYC exchange and be fine."** Mostly no — truly no-KYC venues with deep fiat liquidity are rare, shrinking, and themselves prime enforcement targets; their banking partners impose KYC downstream; and routing large laundered sums through them is itself a flag. The off-ramp chokepoint is hard to avoid precisely because converting to *spendable fiat* requires a bank somewhere, and banks know their customers.

## The playbook: what to do with it

For an analyst, defender, or investor watching a hack unfold, the pipeline reduces to a small set of if-then moves.

- **Signal: a hack just happened.** *Read:* funds are tainted at birth. *Action:* tag the exploiter address immediately and watch its first outflow; the first swap tells you the asset they are consolidating into. *Invalidation:* if no outflow occurs, the funds may be stuck (frozen stablecoins) — check for issuer freezes.
- **Signal: large round-denomination chunks form.** *Read:* mixer preparation. *Action:* pre-position to monitor the mixer's deposit feed and log the burst's amounts and timing. *False positive:* some protocols batch in round numbers; confirm the source is the tagged cluster.
- **Signal: a mixer deposit burst followed by a matching withdrawal burst.** *Read:* the funds are crossing the mixer. *Action:* correlate by denomination and timing, tag every withdrawal address with mixer exposure, and follow re-consolidation. *Invalidation:* if the anonymity set is genuinely large and organic, treat re-links as probabilistic, not certain.
- **Signal: a bridge deposit matched by a destination-chain payout.** *Read:* cross-chain hop. *Action:* join the two events by amount and clock; continue the trace on the new chain's tooling. *False positive:* unrelated transfers of similar size — verify the timing window is tight and the size match is within fees.
- **Signal: single-heir wallets with a constant peeled slice.** *Read:* peel chain. *Action:* script "follow the largest output" and collect every peel destination as a cash-out candidate. *Invalidation:* if outputs branch genuinely (no dominant remainder), it is dispersion, not a peel — switch heuristics.
- **Signal: a large tagged balance goes dormant.** *Read:* waiting out the heat. *Action:* keep it on a watch list with a dormant-balance-moved alert; do not close the case. *Invalidation:* none — the chain remembers; revisit on first movement.
- **Signal: funds approach a KYC venue.** *Read:* the chokepoint. *Action:* this is where you win — coordinate with the venue's compliance and law enforcement to freeze on deposit and obtain the account identity. *False positive:* deposits to an exchange's hot wallet or internal addresses are not always cash-outs; confirm it is a customer deposit address.

The meta-rule: **don't out-run the launderer, out-wait them.** They have to get the money all the way to fiat; you only have to be watching when it arrives. Laundering is a delay tactic, and a funded investigation that understands the leaks turns every step of that delay into another piece of evidence. For what happens after the trace — freezes, seizures, and the analytics stack that powers all of this — see [freezing, recovery, and chain analytics](/blog/trading/onchain/freezing-recovery-and-chain-analytics).

## Further reading & cross-links

- [Mixers, CoinJoin, and obfuscation](/blog/trading/onchain/mixers-coinjoin-and-obfuscation) — how the anonymity set works and exactly how amount and timing leak it.
- [Cross-chain tracing: bridges and the USDT rails](/blog/trading/onchain/cross-chain-tracing-bridges-and-the-usdt-rails) — the bridge models and how to match a deposit to its destination-chain payout.
- [Tracing stolen funds, step by step](/blog/trading/onchain/how-to-trace-a-transaction-flow) — the foundational walkthrough of following a single transaction's flow.
- [Case study: tracing a real flow end-to-end](/blog/trading/onchain/case-study-tracing-a-real-flow-end-to-end) — a full hack-to-off-ramp trace with real tooling.
- [Freezing, recovery, and chain analytics](/blog/trading/onchain/freezing-recovery-and-chain-analytics) — what happens at the chokepoint: stablecoin freezes, seizures, and the analytics stack.
- [Tornado Cash and sanctioning code](/blog/trading/crypto/tornado-cash-and-sanctioning-code) — the legal overlay that turned mixer exposure into a hard block.
- [Stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) — why issuer freeze power is the defender's sharpest tool.
