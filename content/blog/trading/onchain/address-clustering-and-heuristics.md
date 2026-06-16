---
title: "Address Clustering: How Analysts Link Many Addresses to One Owner"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "One person can control thousands of addresses. Address clustering is the engine that collapses them into a single actor. Learn the classic heuristics — co-spend, change detection, deposit sweeps, behavioral signals — and how they differ on Bitcoin versus Ethereum."
tags: ["onchain", "crypto", "bitcoin", "ethereum", "clustering", "heuristics", "tracing", "attribution", "coinjoin", "arkham"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — One owner almost never uses one address. Address clustering is how analysts collapse a sprawl of addresses back into a single actor — the engine behind every "Binance," "Lazarus," or "Smart Money" label you have ever seen on a dashboard.
>
> - **The signal:** patterns that betray common ownership — coins spent together as inputs (co-spend), the leftover "change" output that returns to a sender, and the deposit addresses an exchange sweeps into one hot wallet.
> - **How to read it:** apply the heuristics by chain. On Bitcoin (a UTXO chain) the **common-input-ownership** rule is near-proof; on Ethereum (an account chain) there is no co-spend, so you lean on funding source, gas, timing and behavior.
> - **What you do with it:** turn an anonymous-looking address into an *entity* — then trace its flows, size its holdings, and judge whether a "whale" is one fund or one address of many. Or, as a defender, follow a hacker's funds past their attempt to fan out.
> - **The number to remember:** a single exchange sweep can link **1,200 deposit addresses** to one operator in one stroke — and a single **CoinJoin** can wrongly merge **50 strangers** into one false cluster if you apply the heuristic naively.

On 2022-03-23, attackers drained roughly **\$625M** from the Ronin Bridge, the cross-chain bridge behind the game Axie Infinity — a theft the FBI later attributed to North Korea's Lazarus Group. The stolen funds did not vanish. Within hours the loot began to *fan out*: a single thief-controlled address split the haul across dozens, then hundreds, of fresh wallets, each one a clean-looking address with no obvious link to the others. To a beginner staring at a block explorer, that fan-out looks like the end of the trail — the money has scattered into a crowd of strangers.

It is not the end of the trail. Investigators at firms like Chainalysis, Elliptic and TRM Labs collapsed that crowd of strangers back into a single actor, because the addresses were *not* strangers — they were all controlled by the same operator, and the chain leaves fingerprints of common control. Two new wallets funded their gas from the same source. A batch of addresses all swept into one consolidation point. Timing clustered into the same hours. That collapse — turning a hundred addresses back into one owner — is **address clustering**, and it is the quiet, foundational skill underneath every "entity" label you will ever see on Arkham, Nansen or a Chainalysis report.

Here is the thing nobody tells a beginner: **an address is not a person, and a person is not an address.** One human, one fund, one exchange routinely controls thousands of addresses — for privacy, for accounting, because a custodian assigns one per customer, or simply because change creates a new address every time a Bitcoin is spent. Clustering is how analysts undo that one-to-many sprawl. Get it right and a fog of pseudonymous addresses resolves into a map of real actors. Get it wrong — merge two owners who were never the same — and you have poisoned every conclusion downstream. This post builds the heuristics from zero, shows you exactly how each one works and how each one can deceive you, and keeps the analyst's humility front and center: a cluster is a *hypothesis*, not a certificate.

![Two coins spent together as inputs to one transaction must share one owner because one signer authorized both](/imgs/blogs/address-clustering-and-heuristics-1.png)

## Foundations: one owner, many addresses, and what a cluster is

Before a single heuristic, we need three ideas pinned down from zero: **why one owner controls many addresses**, what an **address** and a **transaction** actually are on the two chain models, and what we mean by a **cluster** or **entity**. Everything in this post is built on these.

### Why one owner ends up with many addresses

Start with a question that feels like it should have an obvious answer: how many addresses does one person have? On a bank statement you have one account number. On a blockchain, the honest answer is *as many as you want, and usually a lot*. There are four big reasons, and each one is a clustering opportunity later.

**Reason one — privacy.** A blockchain is public and permanent. If you received your salary at one address and everyone knew it, they could watch every coffee you bought for the rest of your life. So privacy-conscious users spread activity across many addresses on purpose, hoping no one links them. (Spoiler: clustering is the discipline that links them anyway.)

**Reason two — change addresses (the Bitcoin quirk).** On Bitcoin, you cannot spend "part of a coin." When you spend a coin worth more than you owe, the leftover comes back to you as a brand-new address called a **change address**. Pay for a \$30 coffee with a \$50 bill and you get \$20 in change; on Bitcoin that \$20 lands at a fresh address you control. So *every* spend tends to mint a new address for the same owner. We will return to this — spotting the change output is a core heuristic.

**Reason three — deposit addresses.** When you send crypto to an exchange like Binance or Coinbase, the exchange typically gives *you* a unique deposit address — one address per customer, sometimes one per customer per asset. Behind the scenes, the exchange periodically **sweeps** all those deposit addresses into a small number of **hot wallets**. So a single exchange controls a galaxy of deposit addresses, all funnelling to a shared sink. This is, as we will see, a clustering goldmine.

**Reason four — operational hygiene.** Funds, market makers, bots and protocols use separate addresses for separate jobs: one for gas, one for trading, one for cold storage, one per strategy. Each separation is sensible operationally — and each leaves a behavioral fingerprint.

The upshot: the mapping from *owners* to *addresses* is one-to-many, often one-to-thousands. The whole job of clustering is to invert that map — to take a pile of addresses and group the ones that share an owner.

> [!note]
> **Pseudonymous, not anonymous.** An address is a pseudonym — a stable handle with no name attached. But behavior, timing, funding sources and these clustering heuristics deanonymize pseudonyms over time. "The chain forgets nothing" cuts both ways: it remembers your privacy mistakes forever. We cover the identity layer in depth in [labeling and attribution](/blog/trading/onchain/labeling-and-attribution) and the address mechanics in [addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts).

### A 60-second refresher on the two chain models

Clustering works differently on the two dominant ways a blockchain stores state, so we need both in our heads. We go deep on this in [how blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account); here is the compressed version.

On a **UTXO chain** (Bitcoin, Litecoin, Bitcoin Cash), there are no balances. The state is a set of **unspent transaction outputs** — discrete coins, each a chunk of value locked to an address. A transaction *consumes* some of these coins as **inputs** and *creates* new coins as **outputs**. Your "balance" is just the sum of the coins you can still spend. Crucially, **a transaction can have many inputs**, and to spend them all in one transaction, the signer must produce a valid signature for each. That single fact powers the strongest clustering heuristic in existence.

On an **account chain** (Ethereum and every EVM L2, plus Solana and Tron), the state is a table of balances, one row per address. A transaction is simpler — `from, to, value` — and almost always has exactly **one sender**. There is no co-spending of inputs from multiple addresses, because there are no inputs. That absence is why clustering on Ethereum is fundamentally harder and softer than on Bitcoin: you lose the near-proof heuristic and must lean on circumstantial evidence.

Hold onto this contrast — it organizes the entire post. **UTXO chains give you co-spend; account chains do not.**

### What is a cluster, and what is an entity?

A **cluster** is a set of addresses an analyst believes share a single owner, grouped by one or more heuristics. An **entity** is a cluster that has been *named* — "this cluster is Binance's hot wallet system," "this cluster is the Bybit exploiter." Clustering produces the groups; labeling/attribution puts names on them.

The distinction matters because the two carry different confidence. A *cluster* is a structural claim ("these addresses move together") that the chain's own data supports. An *entity label* adds a real-world identity, which usually requires off-chain evidence — a known deposit address, a court filing, a leak, a self-doxx. You can be highly confident in a cluster and still be wrong about whose it is. Throughout this post we will keep the two separate: heuristics build clusters; labels turn clusters into entities; and the second step is where most public mistakes happen.

One more vocabulary item, because it is the whole point: a **heuristic** is a rule of thumb that is usually right, not a law that is always right. "Coins spent together share an owner" is a heuristic. It is *extremely* reliable on ordinary transactions and *deliberately wrong* on a CoinJoin. The mark of a good analyst is knowing exactly when each heuristic breaks.

### A short history, and the scale of the problem

It helps to know that clustering is not a recent dashboard gimmick — it is the oldest result in blockchain forensics. Bitcoin launched in 2009 with a stated privacy model: addresses are pseudonyms, so as long as you do not reuse them, your activity is hard to link. That promise survived barely two years. In 2011, Fergal Reid and Martin Harrigan published the first paper showing that the **public transaction graph** could be analyzed to link addresses, using exactly the co-spend idea this post centers on. In 2013, Sarah Meiklejohn and colleagues went further in the now-famous paper *"A Fistful of Bitcoins,"* demonstrating that co-spend clustering plus a handful of known "tagged" addresses (from interacting with exchanges and services themselves) could de-anonymize a large fraction of meaningful Bitcoin activity. The entire commercial blockchain-analytics industry — Chainalysis, Elliptic, TRM Labs, and the entity graphs inside Arkham and Nansen — is the industrialized descendant of those papers.

The scale is worth feeling viscerally. Bitcoin has produced on the order of a billion addresses over its lifetime. Co-spend clustering collapses that down dramatically — many hundreds of millions of those addresses merge into a far smaller set of entities, and the largest few thousand of those entities (big exchanges, miners, custodians, funds) account for a wildly disproportionate share of all value. When you see a clean pie chart of "who holds Bitcoin," you are looking at the *output* of clustering, several inferential steps removed from the raw ledger. The pie is only as honest as the heuristics that drew it — which is exactly why understanding those heuristics, and their failure modes, is not optional for anyone who wants to read on-chain data critically rather than swallow a dashboard's conclusions.

One more vocabulary distinction before we dive in: **on-chain** versus **off-chain** evidence. On-chain evidence is anything the ledger itself proves — a co-spend, a sweep, a funding link. Off-chain evidence is everything else — a KYC record at an exchange, a leaked database, a social-media self-doxx, a court filing. Clustering is almost entirely an *on-chain* discipline: it groups addresses using only the public ledger. Naming the cluster — the attribution step — almost always requires an off-chain anchor. Keeping these two layers separate is the single best defense against the most common analyst error: confidently asserting *whose* a correctly-built cluster is, on evidence that never actually named anyone.

## The strongest UTXO heuristic: common-input-ownership

If you remember one heuristic from this entire post, remember this one. On a UTXO chain, **if two or more coins are spent together as inputs to the same transaction, they were controlled by the same owner.** This is the **common-input-ownership heuristic** (CIOH), and it has been the backbone of Bitcoin clustering since the very first academic deanonymization papers (Reid & Harrigan in 2011, Meiklejohn et al. in 2013).

### Why it works: one transaction, one set of signatures

Recall that to spend a UTXO, you must provide a valid cryptographic signature proving you hold the private key that controls it. A transaction that consumes three inputs must carry valid signatures for all three. In the overwhelmingly common case, the same wallet — the same owner — held the keys for all of them and signed the whole transaction at once.

Put plainly: gathering several coins into one transaction's inputs is something only their common owner can do, because only that owner can sign for all of them. So co-spending is near-direct evidence of common control. The figure at the top of this post is exactly this: two inputs (0.7 and 0.5 BTC) flow into one transaction, one signer authorizes both, and the two source addresses collapse into a single entity.

This is why CIOH is the *strongest* heuristic on Bitcoin — it is close to a proof, not a guess. Every other heuristic in this post is softer. Analysts apply CIOH transitively: if address A and B co-spend in one transaction, and B and C co-spend in another, then A, B and C all belong to one cluster, even though A and C never directly appeared together. Run this transitive merge across the whole Bitcoin ledger and you collapse hundreds of millions of addresses into a few hundred million... and then a few tens of millions of *entities*, with the big exchanges showing up as enormous clusters.

The transitive step is where the power — and the danger — compounds. Formally, clustering is a **union-find** problem: you treat each address as a node, draw an edge between any two addresses that ever co-spend, and then take the **connected components** of that graph. Each connected component is one cluster. The beauty is that the more an owner transacts, the more co-spend edges they create, and the more completely their addresses fuse into one component. A heavy Bitcoin user who has made thousands of transactions has, in effect, signed thousands of small confessions that tie their addresses together. This is why high activity erodes privacy: every additional transaction is another chance to leak a co-spend link.

But notice the flip side, because it is the whole reason the next sections exist. Connected-components clustering has **no notion of confidence** — an edge is an edge. If even *one* of those edges is wrong (a CoinJoin you failed to flag, a custodial co-spend), the two components it bridges fuse permanently into one false super-cluster. One bad edge can merge two real entities forever. The math that makes CIOH powerful is exactly the math that makes a single false-merge catastrophic, because connected components do not "un-merge" on their own. That asymmetry — easy to fuse, impossible to cleanly split — is why disciplined engines flag suspect transactions *before* drawing the edge, not after.

To feel how this plays out in the wild, recall the post-hack fan-out from the opening. When stolen funds split across hundreds of fresh Bitcoin addresses, the thief eventually has to *re-consolidate* some of them to move value efficiently — and the moment two of those fresh addresses appear together as inputs in one transaction, CIOH fuses them. The launderer's own need to spend efficiently betrays them. This is why fan-out alone never defeats a competent tracer on a UTXO chain: scattering coins is cheap, but spending them in bulk later requires co-spending, and co-spending is a confession. Only a deliberate privacy tool that breaks the one-signer assumption — a CoinJoin — actually severs the link, and even that is detectable.

#### Worked example: a co-spend collapses two addresses into one owner

Take a concrete Bitcoin transaction. A wallet spends two coins it controls:

- Input 1: **0.7 BTC** from address A
- Input 2: **0.5 BTC** from address B
- Total spent: **1.2 BTC**, worth about **\$72,000** at \$60,000/BTC

The transaction produces a 1.0 BTC payment and 0.18 BTC of change, paying a 0.02 BTC fee (we will dissect those outputs in the next section). The clustering conclusion comes from the *inputs*, not the outputs: because A and B were spent together, the same private-key holder signed both. Before this transaction, A and B were two anonymous addresses. After it, an analyst merges them: **Owner X = {A, B}**, holding the combined value those coins represented. If A had previously received **\$40,000** from a known exchange withdrawal and B had received **\$32,000** from an unknown source, the co-spend now ties that unknown \$32,000 source to the same person who used that exchange. One transaction, two addresses, one owner — and a new lead on the unknown source. **A co-spend is the closest thing on-chain to a signed confession of common ownership.**

### Why CoinJoin breaks it — on purpose

Here is the catch that makes a thoughtful analyst humble. CIOH assumes a transaction's inputs all have one owner. A **CoinJoin** is a transaction engineered to violate exactly that assumption. In a CoinJoin, many independent users *coordinate* to build one transaction that co-spends all their inputs together, each user signing only their own input. No single key signed the whole thing; the signatures are assembled from many separate signers who never trust each other with funds.

The result is a transaction that *looks* like one owner co-spending many coins but is actually fifty strangers pooling theirs. Apply CIOH naively and you commit a **false merge**: you glue fifty unrelated owners into one bogus cluster. CoinJoin exists precisely to poison this heuristic — it is privacy technology built to defeat clustering. We cover it as a detection-and-defense topic in [mixers, CoinJoin, and obfuscation](/blog/trading/onchain/mixers-coinjoin-and-obfuscation); here the point is narrower: **the strongest heuristic has a documented, deliberate countermeasure, so you must detect the countermeasure before you apply the heuristic.** Good clustering software fingerprints CoinJoin transactions (they have telltale shapes — many equal-sized outputs, specific coordinator patterns) and *refuses* to merge their inputs.

## Change addresses: the leftover that returns to the sender

The co-spend heuristic clusters a transaction's *inputs*. The change heuristic is about its *outputs* — and it is what lets you keep following a single owner forward through the chain, one spend at a time.

### What change is, and why it exists

Recall that on Bitcoin you spend whole coins. If you hold a 1.2 BTC coin and want to pay someone 1.0 BTC, you cannot shave off exactly 1.0 — you spend the entire 1.2 BTC coin and the transaction creates two outputs: 1.0 BTC to the recipient, and the leftover (minus fee) back to *you* at a new address. That return-to-sender output is the **change**. It is yours; it just lives at a fresh address.

So a typical "payment" transaction has two outputs: the **payment** (to the counterparty) and the **change** (back to the sender). If you can tell which output is which, you can do two powerful things: (1) cluster the change address with the sender — same owner — and (2) keep tracing the sender forward by following the change. The merchant's payment is a dead end for tracing the sender; the change is the thread you pull.

![Two outputs leave a transaction; the round-number payment goes to the merchant while the odd leftover returns to the sender as change](/imgs/blogs/address-clustering-and-heuristics-2.png)

### How to spot the change output

You cannot read a label that says "change" — the chain does not mark it. You infer it from a set of tells, none decisive alone, strong in combination:

- **Round-number payment vs. odd change.** People pay round amounts (1.0 BTC, 0.5 BTC, exactly \$500 worth). The change is whatever is left over after the payment and fee — almost always an ugly, non-round number like 0.183947 BTC. The odd output is usually the change.
- **Fresh, never-seen address.** The change address is typically newly generated and has no prior history. The payment often goes to an address that already exists (a merchant or exchange that has received before).
- **Address-type / script clues.** If the inputs are a particular address type (say, native SegWit `bc1q...`) and one output matches that type while the other is an older type, the matching output is often the change — wallets tend to send change to the same script type they spend from.
- **Self-change loop.** Sometimes the change goes right back to one of the *input* addresses, or to an address that later co-spends with the inputs — that ties it directly into the cluster via CIOH.
- **Behavioral consistency.** A given wallet software has habits (change is always the last output, or always uses a specific derivation path). Once you fingerprint a wallet's habit, the change becomes obvious for all its transactions.

#### Worked example: identifying the change in a two-output transaction

Return to the transaction from before. Inputs total 1.2 BTC (**\$72,000** at \$60,000/BTC). Two outputs:

- Output 1: **1.0 BTC** (**\$60,000**) to a fresh address
- Output 2: **0.18 BTC** (**\$10,800**) to a different fresh address
- Fee: inputs − outputs = 1.2 − 1.0 − 0.18 = **0.02 BTC** (about **\$1,200**)

Which output is the payment and which is the change? Output 1 is exactly 1.0 BTC — a round, human-chosen number. Output 2 is 0.18 BTC — an odd leftover, exactly what you would get after a round payment and a fee. The heuristic says **Output 2 (0.18 BTC) is the change, returning to the sender.** So the analyst clusters Output-2's address with the input addresses A and B (all Owner X) and keeps tracing the owner by following that 0.18 BTC forward. Output-1's 1.0 BTC is treated as leaving to a counterparty. If we are wrong — if 1.0 BTC was actually the change and 0.18 the payment — we will follow the wrong thread and corrupt the trace. **Change detection is a high-value guess, which is why analysts cross-check it against several tells before committing.**

There is a deeper level to this, called **wallet fingerprinting**, that turns change detection from a guess into something much sharper. Every wallet software — Bitcoin Core, Electrum, a hardware wallet, an exchange's withdrawal system — has implementation quirks in how it builds transactions: which output it places first, what version and locktime fields it sets, which input-ordering rule it follows (some sort inputs and outputs in a canonical order per BIP-69, some do not), what fee-estimation it uses, what address type it generates change to. These quirks are invisible to a casual reader but are a consistent signature. An analyst who profiles a wallet's fingerprint can then identify *which* output is change across all of that wallet's transactions with high confidence, because the wallet always does it the same way. Researchers have used fingerprinting to distinguish exchanges from one another purely by transaction-construction style. The practical lesson: change detection is weak on a single isolated transaction but becomes strong once you have profiled the wallet that made it — context turns a coin-flip into a near-certainty.

> [!warning]
> **The change heuristic is softer than co-spend.** It is a probabilistic read, not near-proof. A sophisticated user can defeat it by paying odd amounts, sending change to a reused address, or using **PayJoin** (a transaction where the *recipient* also contributes an input, so the usual "one sender, two outputs" shape no longer holds). Treat change detection as a strong lead that you confirm, not a fact you assume.

## Deposit-address clustering: the exchange sweep goldmine

The two heuristics so far are Bitcoin-native. This next one works on *both* chain models and is, in practice, where an enormous share of real-world entity labels come from: the way centralized exchanges manage customer deposits.

### One address per customer, all sweeping to one wallet

When you deposit crypto to a centralized exchange (we cover these in [centralized crypto exchanges: Binance, Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase)), the exchange usually issues you a **unique deposit address** — yours alone, sometimes per-asset. You send your coins there; the exchange credits your account balance internally. But the exchange does not leave thousands of customer deposits scattered across thousands of addresses. To manage funds efficiently, it **sweeps** them: on a schedule, it moves the balance from each deposit address into a small number of consolidated **hot wallets**.

That sweep is the fingerprint. Every deposit address the exchange controls sends to the *same* hot wallet (or a small known set). An analyst who identifies the hot-wallet sink can walk *backwards* and label **every** deposit address that fed it as belonging to the same exchange. One known sink unlocks a whole user set in a single stroke.

![Many unique deposit addresses sweeping into one shared hot wallet reveals they are all controlled by the same exchange operator](/imgs/blogs/address-clustering-and-heuristics-3.png)

### Why this is a clustering goldmine

Deposit-address clustering is potent for three reasons. First, **scale**: a single exchange controls millions of deposit addresses, so one identified sweep pattern labels millions of addresses at once. Second, **stability**: the sweep behavior is operational and repetitive, so it is easy to fingerprint and hard for the exchange to hide without breaking its own accounting. Third, **bridging on-chain to off-chain**: if you can tie even *one* deposit address to a known KYC'd identity (because you control it, or because of a leak, or because law enforcement subpoenas the exchange), that identity now anchors the whole cluster. This is why "your coins touched an exchange" is the moment privacy usually ends — the exchange's clustering plus its KYC records connect a pseudonym to a passport.

#### Worked example: clustering 1,200 deposit addresses via the sweep

Suppose an analyst is studying an exchange and identifies one consolidation hot wallet by watching it receive thousands of small, regular inbound transfers and then forward large amounts to known cold-storage addresses. Walking backward from that hot wallet over a month, they find **1,200 distinct deposit addresses** that each sent their full balance into it on a recurring sweep.

- Average swept balance: about **\$6,667** per deposit address
- Total linked in one cluster: 1,200 × \$6,667 ≈ **\$8,000,000**
- Result: **1,200 addresses** — and the customers behind them — all labeled "this exchange," from a single identified sink.

If law enforcement later subpoenas the exchange for the identity behind just *one* of those 1,200 deposit addresses, every transaction that address ever made is now tied to a real name — and the cluster gives investigators a map of the customer's entire on-chain footprint that fed the exchange. **One hot-wallet sink is a master key: it converts \$8M of scattered deposits into a single labeled entity and a bridge to off-chain identity.**

There is an important nuance that separates a good analyst from a sloppy one here. The deposit addresses and the hot wallet are **all controlled by the exchange** — clustering them together is correct. But the *customers* who fed those deposit addresses are **not** part of the exchange's cluster; each deposit address is a boundary between one customer and the exchange. A common beginner error is to merge a deposit address with the *customer* who funded it and conclude the customer "is" the exchange. They are not — the deposit address is the exchange's, the funds that flowed into it came from a customer who is a separate entity. The correct mental model is a two-sided junction: on one side, the deposit address clusters with the exchange (sweep evidence); on the other, the inbound transaction links it to a customer wallet that belongs to *that customer's* cluster. Getting this boundary right is what lets an investigator say "this customer sent \$50,000 to this exchange" rather than the nonsensical "this customer is the exchange."

This is also where clustering meets the **peel chain**, a laundering pattern we cover in the tracing posts. A launderer trying to cash out through an exchange will often send a large sum to a fresh address, "peel" off a small amount to a deposit address, send the rest to another fresh address, peel again, and repeat — a long chain of hops each shedding a little into different exchange deposits. Deposit-address clustering is the counter-move: because all those small peels land at addresses that sweep into the *same* exchange hot wallets, an analyst can recognize that a dozen seemingly-unrelated peels all terminated at the same one or two exchanges, reconstructing the cash-out even though no single transaction looked suspicious. The sweep fingerprint defeats the peel by revealing the common destination.

## Clustering on account chains: no co-spend, so lean on behavior

Everything above leaned heavily on Bitcoin's UTXO model. Move to Ethereum and the ground shifts. There are no inputs to co-spend, so the single strongest heuristic — CIOH — simply does not exist. Account-chain clustering is therefore softer, more circumstantial, and more dependent on stacking weak signals until they harden. This is the regime that tools like Arkham and Nansen operate in for EVM, and it is worth understanding *why* their EVM entity labels are more of a probabilistic art than their Bitcoin ones.

![Bitcoin clustering rests on the strong co-spend heuristic while account chains have no co-spend and rely on weaker funding and behavior signals](/imgs/blogs/address-clustering-and-heuristics-4.png)

### The funding-source heuristic

The workhorse of EVM clustering is **common funding source**. When someone creates a fresh Ethereum address, it starts empty — zero ETH, so it cannot even pay gas to do anything. To bring it to life, *some other address* has to send it its first funds. That funder is a clue. If two fresh wallets both received their first ETH (their "seed" funding, and the gas to operate) from the *same* address, that is suggestive evidence the same person controls both.

A sharper version is **gas funding**: a user who runs many wallets often tops them all up for gas from one central "gas tank" address. Spot that pattern — one address dripping small, similar ETH amounts to dozens of fresh wallets that then go off and trade — and you have a strong candidate cluster. This is exactly how launderers' fan-out gets un-fanned: the fresh wallets look independent, but they share a gas source.

Be careful, though: this heuristic is **suggestive, not proof.** Centralized exchanges fund withdrawals to *unrelated* customers from the same hot wallet, so "funded by Binance" links nothing — it just means both used Binance. Bridges, relayers, and gas-station services likewise fund thousands of unrelated addresses. The funding-source heuristic only clusters when the funder is a *private* address, not a known shared service. We will see in the false-merge section how badly this bites if you forget it.

A second EVM-native wrinkle is worth flagging because it is increasingly common: **account abstraction and smart-contract wallets.** Classic EVM accounts are externally-owned accounts (EOAs) controlled by one private key — covered in [addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts). But more users now operate through smart-contract wallets (Safe multisigs, ERC-4337 accounts) where transactions are submitted by a **bundler** or relayer on the user's behalf, and gas is paid by a **paymaster**. This scrambles the funding-source heuristic: the address that "funded" or submitted the transaction is infrastructure, not the owner. A naive analyst who clusters everyone sharing a paymaster would merge thousands of unrelated users. The defense is the same as always — recognize the shared-service pattern (one address submitting transactions for a huge, diverse population) and exclude it from clustering. As account abstraction grows, the list of "shared infrastructure to exclude" grows with it, and keeping that exclusion list current is a real part of the job.

### Contract-interaction and approval-graph clustering

A distinctively EVM signal is the **approval graph**. Because using tokens in DeFi requires granting approvals, each wallet accumulates a set of (token, spender-contract) approvals over its life. That set is a behavioral fingerprint: a wallet that has approved Uniswap's router, a specific lending pool, and an obscure new farm has a signature most other wallets do not share. Two wallets with near-identical approval sets — especially approvals to *unusual* contracts granted in the *same order* shortly after creation — are strong cluster candidates, because they reveal the owner ran both through the same sequence of dApps. The same logic extends to contract interactions generally: if two wallets both interacted with the same three rarely-used contracts within minutes of each other, the shared, specific behavior is hard to explain by coincidence. This is why DeFi power-users, ironically, are easier to cluster than passive holders — every protocol they touch adds a distinguishing mark to their profile.

#### Worked example: linking two EVM wallets by a shared gas funder

An analyst is investigating two Ethereum wallets, each holding around **\$150,000** of tokens, total **\$300,000**, that appear unrelated — different tokens, different counterparties. But checking each wallet's *very first* inbound transaction:

- Wallet 1's first-ever funding: **0.2 ETH** (about **\$500** at \$2,500/ETH) from address `0xF11de…`
- Wallet 2's first-ever funding: **0.2 ETH** (about **\$500**) from the *same* `0xF11de…`
- `0xF11de…` is a private address with no exchange or bridge label, and it has funded a dozen other fresh wallets the same way.

That shared private funder is the link. Two wallets controlling **\$300,000** combined, seemingly independent, are now a candidate cluster because one private gas-tank address seeded both. An analyst would then look for corroboration — shared counterparties, synchronized timing, identical approval patterns — before promoting the candidate to a confident cluster. **On EVM you rarely get one decisive clue; you get a shared funder plus three behavioral echoes, and the cluster is the weight of evidence, not a single proof.**

### Stacking behavioral and temporal signals

Because no single EVM signal is conclusive, analysts stack several. Each is a weak ray of light; together they triangulate.

![Account-chain clustering stacks funding source gas timing counterparty and approval signals because no single behavioral clue is conclusive on its own](/imgs/blogs/address-clustering-and-heuristics-5.png)

- **Funding source / gas** — covered above; who brought the wallet to life.
- **Timing.** Addresses controlled by one human (or one bot) tend to act in the same windows — the same hour each day (the owner's timezone), or in synchronized bursts when a script fires. Two wallets that transact within the same few seconds, repeatedly, are likely one automation.
- **Recurring counterparties.** If two wallets both repeatedly interact with the same niche contract or the same handful of unusual addresses, that shared social graph hints at common control.
- **Approval patterns.** On EVM, using a token in DeFi requires an **approval** — granting a contract permission to move your tokens (we cover this in [tokens, on-chain transfers, and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals)). Two wallets that grant the exact same set of approvals to the exact same contracts share a behavioral signature.
- **Nonce / cadence patterns.** Every EVM account has a **nonce** — a counter that increments by one with each transaction it sends, preventing replay. The *style* of activity (batch sizes, round-number transfers, the ordering and pacing of transactions) is a habit. Two wallets with the same cadence and the same round-amount habits may be one operator.

None of these alone would convince a careful analyst. But when a fresh wallet is gas-funded by the same private address as another, acts in the same five-minute windows, approves the same three contracts, and trades the same obscure token — the probability they are independent collapses. **The EVM discipline is evidence-stacking: build the cluster from converging weak signals, and state your confidence honestly.**

#### Worked example: stacking signals to cluster a wash-trading ring

Suppose an analyst suspects a token's volume is fake — wash trading, where one operator trades with themselves across many wallets to inflate apparent activity. They find five wallets that traded the token, generating **\$2,000,000** of reported volume in a week. On the surface, five independent traders. Stacking the signals:

- All five were gas-funded with **0.05 ETH** each (about **\$125** apiece, **\$625** total) from one private address `0xBEEf…`.
- All five approved the same DEX router contract within the same two-hour window.
- Their trades alternate — wallet 1 buys, wallet 2 sells, wallet 3 buys — in a tight loop, each trade roughly **\$8,000**, recycling the same coins back and forth.
- The token's "volume" of \$2M is almost entirely these five wallets trading with each other; net flow to outsiders is near zero.

No single signal proves common control, but four converging signals — one funder, synchronized approvals, alternating self-trades, a closed-loop money flow — push the cluster to high confidence. The read: the \$2M of volume is manufactured by one operator across five wallets, and a trader who saw "rising volume" as a buy signal would be walking into a trap. **On EVM you do not get a confession; you get four weak signals that, stacked, turn a \$2M volume chart into a one-actor wash-trading cluster.**

## A walkthrough: how the tools cluster automatically

You will rarely run these heuristics by hand on a million addresses — that is what clustering engines are for. But knowing the heuristics lets you *read the tools critically* instead of trusting their labels blindly. Here is how a real analyst uses them, across the three classic patterns.

### Pattern 1 — clustering a CEX's deposit addresses (the sweep)

On a tool like Arkham or a Dune query, you start from a suspected exchange hot wallet and look at its inbound transfers. The tell-tale sweep signature: thousands of *distinct* source addresses, each sending its *entire* balance, on a *recurring* schedule, with the hot wallet then forwarding consolidated amounts to cold storage. You verify a handful of those source addresses are single-use deposit addresses (one inbound from a user, one outbound sweep, then dormant). Once confirmed, you label the sink as the exchange's consolidation wallet and inherit the label onto every address that swept into it. In Arkham this is largely pre-computed — the platform shows you "Binance deposit" labels on addresses precisely because it has run this sweep-clustering at scale. Your job is to sanity-check: does the sweep cadence look like one operator, or could two services share a sink?

### Pattern 2 — spotting the change output in a Bitcoin transaction

On a Bitcoin explorer (mempool.space, or a tracing tool like Chainalysis Reactor / Elliptic), open a two-output transaction. Read the inputs: note the address type and total. Read the outputs: one is likely round, one likely odd. Apply the tells — the odd-valued, freshly-generated output matching the input's script type is your change candidate. Cluster that change address with the inputs and follow it forward to the next hop. Tracing tools do this automatically and draw you a flow graph where the sender's owner-thread is highlighted across many hops — but they are applying *this* heuristic internally, and they can be wrong on a PayJoin or an odd-amount payment. Reading the raw transaction yourself is how you catch their mistakes.

### Pattern 3 — linking EVM wallets by common funding source

On Arkham or Nansen, open a wallet and look at its first inbound transfer ("first funded by"). If two wallets you suspect are related were both first funded by the same *private* address, the tool often already groups them under one entity. You verify the funder is not a shared service (check whether it is labeled an exchange, a bridge, or a relayer — if so, the link is worthless). Then you stack: do the wallets share counterparties, timing, approvals? Nansen's "Smart Money" labels and Arkham's entity graph are built on exactly this funding-plus-behavior stacking. The skill is reading their confidence honestly: an EVM entity label is a *hypothesis with evidence*, not a Bitcoin co-spend's near-certainty. We surveyed the full toolset in [the on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) and the broader tracing flow in [how to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow).

## The cost of a wrong heuristic: false merges

Every heuristic in this post can fail, and a failure is not harmless — it is a **false merge**, where you glue two owners who were never the same. False merges are insidious because they look like *success*: your cluster got bigger, your map got richer. But a single wrong merge can chain transitively into a monster cluster that swallows unrelated entities, and every conclusion you draw from it is now wrong. This is the analyst's humility note, and it deserves its own section.

![A CoinJoin deliberately co-spends inputs from many independent owners so a naive co-spend rule wrongly merges strangers into one false cluster](/imgs/blogs/address-clustering-and-heuristics-6.png)

### The classic poisoners

- **CoinJoin (and Wasabi/Whirlpool-style mixers).** As covered, these co-spend inputs from many independent owners *on purpose*. Apply CIOH and you merge strangers. Worse, because clustering is transitive, one un-flagged CoinJoin can merge two previously-separate clusters into one — a cascade.
- **PayJoin.** Here the *recipient* secretly contributes an input to the payment transaction. Now the transaction's inputs have two owners (sender and recipient), and the usual change heuristic mis-reads which output is change. PayJoin is designed specifically to feed clustering software a lie.
- **Shared custody / pooled wallets.** Some custodians and protocols hold many users' funds in one address or co-spend on behalf of many users. Cluster that and you have merged a custodian's entire customer base into "one owner" — technically true that one entity *controls* it, but disastrously wrong if you treat that cluster as a single human actor.
- **Shared services as funders (EVM).** The mirror failure on account chains: treating "funded by the same exchange hot wallet" as evidence of common control. It is not — the exchange funds thousands of unrelated withdrawals from one wallet. Cluster on that and you merge an exchange's entire withdrawing customer base.

#### Worked example: a CoinJoin that breaks the cluster on purpose

Take a CoinJoin transaction with **50 inputs**, each roughly **\$5,000**, contributed by 50 independent users who coordinated through a privacy wallet. The transaction's total input value is 50 × \$5,000 = **\$250,000**, and it produces many equal-sized outputs (a CoinJoin signature) so no one can tell whose output is whose.

A naive analyst applies common-input-ownership: "50 inputs co-spent, therefore one owner controls \$250,000." That conclusion is **completely false** — there are 50 owners, and the transaction was *built* to produce exactly this false merge. If those 50 users had each previously touched different exchanges, the naive cluster now wrongly ties 50 unrelated identities into one entity, and any name later attached to it would smear 49 innocents. The correct move is to **detect the CoinJoin shape** (many equal outputs, known coordinator pattern) and refuse to apply CIOH across it. **A \$250,000 CoinJoin is a heuristic trap: it offers you a fat, wrong cluster, and the whole skill is declining to take it.**

### The discipline that prevents false merges

Serious clustering engines defend against this with a few rules: detect and exclude known mixer/CoinJoin transaction shapes before merging; never treat a *labeled shared service* (exchange, bridge, mixer, relayer) as a clustering link; and propagate a **confidence score** with every cluster rather than a binary "same/not-same." As an analyst reading these tools, your defense is humility: when a cluster looks suspiciously large or merges entities that should not be related, suspect a poisoned heuristic before you trust it. A cluster is a hypothesis; a false merge is what happens when you forget that.

It is worth naming the two directions error can run, because they have opposite consequences. **Over-merging** (a false merge) glues separate owners into one cluster — it inflates how much a single actor appears to control and can wrongly implicate innocents whose funds touched a mixer or a shared service. **Under-merging** (a false split) fails to link addresses that *do* share an owner — it makes one actor look like many, understating concentration and letting a launderer's fan-out look like genuine dispersion. Naive co-spend clustering tends to over-merge on CoinJoins and under-merge on sophisticated users who never co-spend. The two errors call for opposite fixes: over-merging is fixed by *excluding* suspect edges, under-merging by *adding* softer behavioral heuristics to catch links co-spend missed. A mature engine balances both, and reports which way it is likely to err on a given cluster.

This is also why the better tools are moving away from a single deterministic answer toward **probabilistic clustering** — assigning each candidate link a likelihood and letting an analyst set the threshold for the question at hand. A regulator building a criminal case needs near-certainty and will accept only co-spend-grade evidence; a trader sizing a thesis can act on a 70%-confidence behavioral cluster because the cost of being wrong is a bad trade, not a wrongful accusation. The same underlying clustering, read at different confidence thresholds, serves both. The analyst's job is to *know which threshold their question requires* and to refuse to let a 70% behavioral cluster masquerade as a 99% co-spend fact when the stakes demand the latter.

## The full picture: heuristic strength and what breaks each

Putting it together, here is the analyst's mental decision matrix — each heuristic, the chain it works on, what it links, how strong it is, and the documented way it fails. This is the table to keep in your head every time a dashboard hands you an "entity" label.

![Co-spend is near-proof and deposit sweeps are strong while change and behavioral heuristics are softer and each has a documented countermeasure](/imgs/blogs/address-clustering-and-heuristics-7.png)

The hierarchy is worth stating in one breath: **common-input-ownership is near-proof but only on UTXO chains and only outside CoinJoins; deposit-address sweeps are strong on both chains and unlock whole user sets; change detection is a medium-strength forward-tracing tool that round payments and PayJoin can fool; common funding and behavioral stacking are the soft, suggestive backbone of EVM clustering, defeated by shared services and mimicry.** Strength descends as you move from structural proof (one signer signed both) toward circumstantial behavior (these two wallets act alike), and your stated confidence should descend with it.

This is also why a "whale" label deserves scrutiny. A dashboard might show one address holding \$50M and call it a whale. But clustering can reveal that the "whale" is one address of a fund's cluster of forty, or — the reverse error — that what looks like forty separate holders is one entity behind a false merge. Holder-concentration analysis lives or dies on clustering quality, which is why we treat it carefully in [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration). Bad clustering makes concentration look lower (forty addresses) or higher (one false-merged blob) than reality.

To make that concrete: suppose a token's top-10 addresses each hold about **\$5M**, totaling **\$50M**, and a naive "top holders" view reports ten independent whales — comfortably decentralized. Run clustering and find that eight of those ten addresses were gas-funded by one private address, approve the same contracts, and move in lockstep. Now the real picture is **one entity controlling \$40M** (eight of the ten) and only two genuinely independent holders of \$5M each. The token that looked decentralized is in fact controlled by a single actor who could dump \$40M on the market at will. That is the difference clustering makes to a due-diligence read: it converts a reassuring "ten whales" headline into the actionable "one whale who owns 80% of the float." A trader who skipped the clustering step would have badly mispriced the rug risk.

## Common misconceptions

**"One address equals one person."** The foundational error. One owner routinely controls thousands of addresses — for privacy, from change, as exchange deposit addresses, or for operational separation. Reading the chain as if address = identity makes every "this whale just sold" headline unreliable; that "whale" may be one exchange's hot wallet serving a million users. Clustering exists precisely because the address-to-owner map is one-to-many.

**"A cluster is a fact."** A cluster is a *hypothesis* supported by heuristics of varying strength. Co-spend on an ordinary Bitcoin transaction is near-proof; common funding on EVM is a suggestion. Treating an EVM "entity" label with the same certainty as a Bitcoin co-spend cluster is how analysts get burned. Always ask: which heuristic built this, and what breaks it?

**"Mixers make tracing impossible, so clustering is hopeless against them."** Two errors in one. First, mixers don't make tracing *impossible* — they raise its cost and lower confidence, and statistical and timing analysis still recovers many flows; we cover this defensively in [mixers, CoinJoin, and obfuscation](/blog/trading/onchain/mixers-coinjoin-and-obfuscation). Second, the real danger from a mixer like CoinJoin is the *opposite* of hopelessness: it tempts naive clustering into confident *false* merges. The mature stance is to detect the mixer and lower confidence, not to give up and not to merge blindly.

**"Ethereum clustering is just as reliable as Bitcoin's."** No — and the difference is structural, not a tooling gap. Bitcoin has co-spend (near-proof); Ethereum does not, so EVM clustering rests on softer funding-and-behavior signals. An Arkham label on a Bitcoin cluster and the same-looking label on an EVM cluster carry genuinely different epistemic weight. The data model dictates the ceiling on certainty.

**"If a tool shows a clean entity label, I can trust it."** Tools are excellent and run these heuristics at a scale no human can match — but they encode the same failure modes. A shared-service funder, an un-flagged CoinJoin, or a custodian's pooled wallet can produce a confident-looking label that is a false merge. Use the tool's labels as strong leads, then verify the heuristic underneath when the stakes are high.

## The playbook: what to do with clustering

Clustering is a means, not an end. Here is the if-then checklist for putting it to work — and for not fooling yourself.

**As a trader/investor:**

- **Signal:** a dashboard shows a "whale" or "smart money" address making a big move. **Read:** before reacting, check whether that address is part of a cluster — is the "whale" one address of a fund's forty, or an exchange hot wallet (not a real holder at all)? **Action:** size your conviction to the cluster, not the single address. **Invalidation:** if the "whale" turns out to be a labeled exchange or bridge wallet, the "smart money buy" signal is noise — it is a customer's deposit, not an investment thesis.
- **Signal:** token holder-concentration looks reassuringly low (many holders). **Read:** run or check clustering — are those "many holders" actually one entity behind multiple addresses? **Action:** discount apparent decentralization that clustering collapses; a token "held by 500 wallets" that cluster into 3 owners is concentrated. **Invalidation:** if clustering is poisoned by a CoinJoin-style false merge, the concentration could be *overstated* — verify the cluster's heuristic before trusting either direction.

**As an analyst/defender:**

- **Signal:** stolen funds fan out across dozens of fresh addresses after a hack. **Read:** the fan-out is not the end — look for the un-fanning fingerprints: a common gas funder, a later co-spend, synchronized timing. **Action:** cluster the fresh wallets back to the operator and keep tracing; flag the cluster to exchanges so cash-out attempts get frozen. **Invalidation:** if the thief routes through a genuine mixer or CoinJoin, lower your confidence and switch to statistical/timing analysis rather than asserting a false co-spend merge.
- **Signal:** you need to attach a real identity to a cluster. **Read:** find the cluster's contact point with the regulated world — a deposit address at a KYC'd exchange. **Action:** that single off-chain anchor names the whole cluster; subpoena or a known-address match converts pseudonym to person. **Invalidation:** if the only "identity" anchor is a shared service (exchange hot wallet), it names nothing — back off, because everyone who used that service shares it.

**The rule of thumb to carry:** *every cluster is a hypothesis with a confidence level, built by a named heuristic that has a named way to fail.* State the heuristic, state the confidence, know the counterexample. A co-spend on an ordinary Bitcoin transaction is worth near-certainty; a common-funder link on EVM is worth "probably, pending corroboration"; and a cluster that merges strangers through a CoinJoin is worth nothing but a lesson. The analysts who get this right are not the ones with the fanciest tool — they are the ones who never forget that a heuristic is a rule of thumb, and that the chain's most powerful deanonymizers are also its most powerful traps.

## Further reading & cross-links

- [How blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account) — the data-model distinction that decides whether you even *have* the co-spend heuristic.
- [How to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow) — clustering is the engine that keeps a trace alive past fan-out; this is the tracing discipline it feeds.
- [Labeling and attribution](/blog/trading/onchain/labeling-and-attribution) — turning a cluster into a *named* entity, and where the off-chain identity anchors come from.
- [Mixers, CoinJoin, and obfuscation](/blog/trading/onchain/mixers-coinjoin-and-obfuscation) — the privacy techniques built to poison clustering, read as a detect-and-defend topic.
- [Addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts) — what an address actually is, and why one owner holds so many.
- [Supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — the analysis that lives or dies on clustering quality.
- [Centralized crypto exchanges: Binance, Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) — where deposit-address sweeps and the on-chain-to-KYC bridge come from.
