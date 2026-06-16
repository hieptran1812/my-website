---
title: "Mixers, CoinJoin, and Obfuscation: How the Trail Gets Muddied — and Where It Leaks"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How pooled mixers, CoinJoin, and peel chains break the simple follow-the-money trace — and the side channels (amount, timing, gas, the KYC off-ramp) where an investigator re-links the funds."
tags: ["onchain", "crypto", "mixers", "tornado-cash", "coinjoin", "money-laundering", "blockchain-forensics", "ofac", "peel-chain", "ethereum", "bitcoin", "compliance"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A mixer does not erase the money trail; it hides one path inside a crowd of identical-looking paths. The job of an analyst is not to "break the math" but to find the side channels the crowd leaks.
>
> - **What it is:** mixers and privacy tools (Tornado Cash, CoinJoin, peel chains) defeat the simple one-to-one trace by pooling many users so any one withdrawal could plausibly belong to any depositor — the "anonymity set."
> - **How to read it:** the pool hides the *direct* on-chain edge, but it leaks correlatable side channels — matching amounts, deposit-to-withdrawal timing, a reused relayer, gas funded from a linked wallet, and the regulated off-ramp where a real identity finally attaches.
> - **What you do with it:** bracket a withdrawal back to its deposit candidates by amount and timing, walk a peel chain by following the large remainder, and treat the KYC cash-out as the chokepoint that closes the case.
> - **The number to remember:** Tornado Cash moved roughly **\$7B** over its life and Elliptic tied about **30%** of it to illicit sources — most users were not criminals, which is exactly why a thin anonymity set leaks.

On 23 March 2022, attackers drained the Ronin bridge of roughly **\$625M** in ETH and USDC — at the time the largest crypto theft on record. The funds belonged to the Lazarus Group, North Korea's state hacking apparatus, and within weeks they began feeding tranches of the stolen ETH into Tornado Cash, the dominant Ethereum mixer. On paper this should have been the end of the trace: a mixer is supposed to sever the link between a deposit and a withdrawal, leaving investigators staring at a clean, freshly-funded address with no on-chain parent.

It did not end the trace. Chainalysis, Elliptic, and the FBI followed the Ronin funds *through* Tornado, not by breaking any cryptography, but by watching the side channels — the timing of deposits and withdrawals when the pool was thin, the exact denominations, the wallets that paid gas to the "fresh" withdrawal addresses, and ultimately the exchanges where someone tried to convert the laundered ETH into spendable dollars. In August 2022 the US Treasury took the unprecedented step of sanctioning the Tornado Cash *smart contracts themselves* — sanctioning code, not a person or a company. The episode is the cleanest case study in the whole series for one idea: obfuscation buys distance from the source, but it does not buy invisibility, and a defender's edge lives entirely in the gap between those two things.

This post is written from the **analyst and defender's chair**. We will explain how mixers, CoinJoin, and peel chains actually muddy a trace — but only to the depth you need to *recognize and analyze* their on-chain footprint, and the whole second half is about where that footprint **leaks**. This is forensics, not a manual. The mental model below is the one to carry the whole way through.

![A mixer pools N identical deposits then pays out N fresh withdrawals so any one output could be any input](/imgs/blogs/mixers-coinjoin-and-obfuscation-1.png)

## Foundations: why a normal trace works, and what "breaking" it means

Before you can see where obfuscation leaks, you need to be precise about what it is attacking. So let us build up from zero.

**Why a normal trace works at all.** A public blockchain is a permanent ledger of value transfers. On an account-based chain like Ethereum, every transaction says "address A sent X to address B at time T," and that record never disappears. So if you know that a hacker controls address A, you can read off every address A ever paid, then every address *those* paid, and so on, walking the graph outward. The trace works because each hop is a **one-to-one link**: this specific output came from this specific input. The whole discipline of on-chain forensics — covered in detail in [how to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow) — is the art of walking that graph of one-to-one links without losing the thread.

The same is true on Bitcoin, with a twist worth defining now. Bitcoin uses the **UTXO model** (unspent transaction output): instead of an account balance, your wallet holds a set of discrete "coins" of specific amounts, like bills in a wallet, and a transaction consumes some of those bills as inputs and produces new ones as outputs. (We covered this in depth in [how blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account).) The crucial forensic consequence: when a Bitcoin transaction has several inputs, a heuristic called **common-input-ownership** assumes they were all controlled by the same wallet — because to sign a transaction spending three coins, you normally need all three private keys. That single assumption powers most Bitcoin clustering, and as we will see, CoinJoin is a deliberate attack on exactly that assumption.

**What "breaking the trace" means.** An obfuscation tool tries to destroy the one-to-one link at a single hop. It does not delete the funds, and it does not delete the transactions — those are still public and permanent. It just makes the *mapping* from inputs to outputs ambiguous. Where a normal trace says "this 100 ETH that left the pool is the same 100 ETH that entered from the hacker," a mixed hop should only let you say "this 100 ETH could have come from any of the N people who deposited 100 ETH." That is the entire game.

**The anonymity-set concept.** The size of that "any of N" crowd is the **anonymity set**: the number of other users whose funds are indistinguishable from yours at the muddied hop. If 1,000 people deposited an identical amount and you are one of them, your withdrawal hides among 1,000 — the analyst's best guess that any particular deposit is yours is 1-in-1,000. But if only 4 people used the pool that hour and your deposit and withdrawal are minutes apart, your "anonymity set" is effectively 4, or even 1. **The privacy a mixer provides is not a fixed property of the tool; it is a function of how crowded the pool is and how disciplined the user is.** That sentence is the seed of every leak in this post. A mixer with a thin, lazy crowd provides almost no privacy at all.

**Taint and demixing.** Two terms you will see constantly. **Taint** is the property of being connected — directly or through some number of hops — to known-illicit funds; a "tainted" address is one whose money traces back to a hack, a scam, or a sanctioned entity. **Demixing** is the analyst's act of partially or fully reversing a mixer: assigning probabilities to "which deposit funded which withdrawal" using the side channels we will catalog, and in the best cases pinning the link with near-certainty. Demixing rarely means a clean mathematical inversion; it means narrowing the anonymity set from N down to a handful, then to one, using behavior the tool's users leak.

**The three families of obfuscation.** Almost everything in this space is one of three shapes, and the rest of the post takes them in turn:

1. **Pooled mixers** (Tornado Cash). Everyone deposits a *fixed denomination* into one shared smart-contract pool, then later withdraws the same denomination to a fresh address using a cryptographic receipt. The anonymity set is the whole pool.
2. **Collaborative transactions** (CoinJoin, as used by Wasabi and Samourai/Whirlpool on Bitcoin). Several users co-sign *one* Bitcoin transaction with many **equal-sized outputs**, so the common-input-ownership heuristic breaks and no output is provably tied to any one input.
3. **Peel chains and chain-hopping.** Not real mixing at all — a *pseudo-mixer*. Funds are shuffled through a long series of fresh addresses (and sometimes across chains/bridges), peeling off small amounts as they go, hoping the analyst simply gives up walking the chain.

### Pseudonymity is not anonymity — and a mixer attacks the wrong layer

There is a subtlety that explains why mixers are weaker than their reputation. A blockchain address is **pseudonymous**, not anonymous: it is a stable handle (like a pen name) that is not, by itself, tied to your legal identity — but every action that handle ever takes is public and permanent and linked to the same handle. Anonymity would mean *no stable handle at all*; pseudonymity means *one handle whose whole history is an open book*. The reason ordinary wallets get de-anonymized over time is that the pen name accumulates a behavioral signature — funding sources, counterparties, timing habits, gas patterns — until that signature matches a real person.

Here is the catch for mixers. A mixer breaks the link *between two pen names* (the deposit address and the withdrawal address). What it does **not** do is make either pen name anonymous — both still have full, public histories before and after the mixer. So the analyst's job is not "defeat the mixer's cryptography"; it is "re-connect two pen names whose pre- and post-mixer behavior betrays that they belong to the same operator." The mixer attacks the single hop in the middle; the analyst attacks everything *around* that hop, where pseudonymity is doing its usual leaking. This is why so much of demixing is really just clustering applied on both sides of the muddied hop, and why a mixer used by an otherwise-sloppy operator provides far less protection than the marketing implies.

### Quantifying the anonymity set, and how taint propagates

It helps to make the anonymity-set idea numerical, because that is how an analyst actually reasons about it. Suppose a denomination pool currently holds `N` indistinguishable deposits and you are one of them. The naive privacy you get is `1/N`: a uniform guess assigns each deposit a `1/N` chance of being the source of any given withdrawal. With `N = 1,000`, that is a 0.1% prior — strong. With `N = 4`, it is 25% — almost useless. But the naive `1/N` is the *best case*; the analyst's job is to shrink the *effective* `N` below the nominal one using side information. If timing rules out 996 of the 1,000 deposits because they happened days before or after the withdrawal in question, the effective set is 4, and the effective probability jumps from 0.1% to 25%. Every leak in this post is, mathematically, a tool that shrinks the effective `N`. That is the one quantity the whole discipline is fighting over.

There is a second quantity worth naming: **taint propagation**, the model an analytics firm uses to decide *how much* of a downstream coin is "dirty." Two common models matter here. **Poison taint** says: if any tainted satoshi or wei ever touched an address, the entire balance is tainted forever — simple, aggressive, and prone to false positives because it spreads taint to everyone who unknowingly received a sliver. **Haircut taint** is proportional: if 10% of an address's inflows were tainted, then 10% of every outflow carries the taint, and the dirty fraction dilutes as it mixes with clean funds. A third, **FIFO (first-in-first-out)**, treats coins like a queue and is the model several court cases and the UK's *Tulip Trading*-era jurisprudence have leaned on. Why does this matter for mixers? Because a mixer is a deliberate **taint-laundering machine against the haircut model**: by pooling a tainted deposit with thousands of clean ones, it tries to dilute the dirty fraction of each withdrawal toward zero. The defender's counter is to *not* rely on haircut dilution through the mixer at all, and instead re-establish the *specific* deposit→withdrawal link via the side channels — which restores poison-grade certainty to a single thread rather than smearing a diluted percentage across the whole pool. Knowing which taint model an analyst (or a court) is using changes what "the funds are clean" even means.

With the vocabulary fixed, let us go deep on the first and most important family.

## Pooled mixers: Tornado Cash mechanics, at a defender's altitude

Tornado Cash is the canonical pooled mixer, and understanding *just enough* of its mechanics is what lets you read its footprint. We will stay at the altitude an analyst needs — enough to recognize and trace the pattern, not a build guide.

**The fixed-denomination pools.** Tornado did not have one pool; it had separate pools for fixed amounts — on Ethereum, the well-known sizes were **0.1 ETH, 1 ETH, 10 ETH, and 100 ETH** (with equivalent pools for other assets). This is deliberate: if everyone deposits and withdraws *exactly* 100 ETH, then amounts cannot distinguish users. A pool of arbitrary amounts would be trivially demixable (a 73.4182 ETH deposit and a 73.4182 ETH withdrawal are obviously the same money), so fixed denominations are the core privacy mechanism. Keep that fact close — it is also the source of the most important leak.

**Deposit, then withdraw to a fresh address.** At a high level: a user deposits one fixed denomination into the pool and, in doing so, registers a cryptographic commitment. Later — ideally much later, and from a different address — they prove (using a zero-knowledge proof) that they are entitled to withdraw one denomination from the pool, *without revealing which deposit was theirs*. The withdrawal goes to a brand-new address that has no prior on-chain history. The zero-knowledge proof is the clever part: it convinces the contract "I deposited, so let me withdraw" while revealing nothing that links the withdrawal back to a specific deposit. On-chain, you see a deposit transaction from address A, and an unrelated-looking withdrawal transaction to fresh address Z, with no edge between them in the transaction graph.

**The anonymity set is the pool.** Your privacy is the count of *other* deposits sitting in the same denomination pool between your deposit and your withdrawal. The 100-ETH pool, used by a handful of whales, was usually far thinner than the 1-ETH pool — which matters enormously, because a thin pool is a small anonymity set, and a small anonymity set leaks. To genuinely hide, a user needs (a) a crowded pool, (b) a long, randomized delay between deposit and withdrawal, (c) a withdrawal address with no link to the deposit address, and (d) gas for that fresh address that does not come from a traceable place. Most users fail at least one of these. The figure below maps the mechanism and pins the four places it bleeds.

![Tornado deposit pool and withdraw flow with timing amount relayer and gas funding leaks annotated](/imgs/blogs/mixers-coinjoin-and-obfuscation-2.png)

### Where a pooled mixer leaks — the analyst's checklist

This is the heart of the post. A mixer's privacy is only as strong as its weakest-disciplined user, and here is where the discipline breaks. None of these involve breaking cryptography; they are correlation attacks on metadata the chain records for free.

**Leak 1 — timing correlation in a thin pool.** If a 100-ETH deposit lands at 14:02 and a 100-ETH withdrawal to a fresh address fires at 14:11, and *no one else* deposited or withdrew 100 ETH in that window, the link is nearly certain. The zero-knowledge proof hides the mapping cryptographically, but it cannot hide the wall-clock timestamps, and a near-empty pool means the timestamps *are* the mapping. Analysts build a timeline of every deposit and withdrawal per denomination and look for the windows where the set shrinks to one or two.

**Leak 2 — amount correlation, especially with multiple deposits.** The fixed-denomination trick only works if you send round amounts. The moment someone needs to move a non-standard total — say 96.4 ETH — they have to compose it from fixed chunks (some 10s, some 1s, some 0.1s), and that *combination* of denominations and counts can be near-unique. Withdrawing the exact same multiset of denominations you deposited is a glaring tell. Even worse, withdrawing a total that sums to precisely what you deposited (when the world expects round, independent withdrawals) brackets you immediately.

**Leak 3 — the relayer fingerprint.** A fresh withdrawal address has no ETH, so it cannot pay its own gas. Tornado's answer was **relayers**: third parties that submit the withdrawal transaction and take a fee out of the withdrawn amount. Convenient — but a relayer is a fingerprint. If the same launderer routes ten withdrawals through the same relayer at the same fee rate, those ten "unrelated" fresh addresses are now linked to each other through a shared service and fee pattern, collapsing ten separate anonymity sets into one cluster.

**Leak 4 — gas funded from a linked wallet.** When a user does *not* use a relayer, the fresh withdrawal address still needs gas — and that gas has to come from somewhere. If it arrives from the user's main wallet, from the same centralized exchange the deposit funds came from, or from another address in the same cluster, the "fresh" address is fresh in name only. Gas-funding is one of the most reliable de-anonymizers because users forget the new address is a newborn that cannot move without being fed.

**Leak 5 — withdrawing the exact deposited sum, or reuse mistakes.** Beyond amounts, the most common error is simple impatience and reuse: depositing and withdrawing in the same session, sending withdrawn funds straight to a previously-used address, or consolidating several "anonymous" withdrawals back into one wallet (which re-links them all). These are the human errors that turn a strong tool into a weak one — and they are why detection beats obfuscation more often than the public assumes.

**The off-ramp, which deserves its own section, is Leak 6** — and we get there shortly, because it is the leak that closes nearly every case regardless of how disciplined the user was.

### Relayers and gas-funding, in more detail

Leaks 3 and 4 deserve a closer look, because together they are responsible for a large share of real-world mixer de-anonymizations, and understanding *why* they exist tells you where to look first.

The root problem is unavoidable: a withdrawal must land on an address that has no on-chain history, or the history would re-link it to the user. But an address with no history also has **no ETH**, and on Ethereum you cannot move a single token without paying gas. So the "fresh" address faces a chicken-and-egg bind — it needs gas to act, but any gas it receives is a transaction *into* it, which creates a parent edge an analyst can follow. There are only a few ways out of the bind, and each is a tell.

The first is the **relayer**: a service that submits the withdrawal on the user's behalf and deducts its fee from the withdrawn amount, so the fresh address never needs externally-supplied gas. Clean in theory, but the relayer's own address, its fee schedule, and the timing of its submissions form a fingerprint. When an analyst sees a set of withdrawals all submitted by the same relayer at the same fee within a short window, those withdrawals become a *correlation cluster* even if their destination addresses are otherwise unrelated. A single launderer running many withdrawals through one relayer has, in effect, signed all of them with the same pen.

The second way out is to **fund the gas directly** from another wallet — and here the discipline almost always breaks. The launderer needs to seed the fresh address with a little ETH; where does that seed come from? If it comes from their main wallet, the link is immediate. If it comes from the same centralized exchange the original deposit funds came from, the exchange's records tie both ends to one account. If it comes from another address in the same operational cluster, clustering heuristics ([address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics)) merge them. The "fresh" address is a newborn that cannot feed itself, and whoever feeds it leaves a fingerprint on the spoon.

The defender's takeaway is a search order: when you have a candidate withdrawal address, the *first* question is always "who paid its gas, and through what?" That single edge — relayer or direct funding — is more often the de-anonymizer than the timing or the amount, because it is the one step the user cannot skip and most often gets lazy about.

#### Worked example: a \$1M deposit in ten 100-ETH chunks, re-linked by timing

Say a launderer wants to push **\$1,000,000** through Tornado at an ETH price of **\$3,000**, which is about **333 ETH**. The 100-ETH pool is the deepest available, so they round to ten deposits of 100 ETH each — but they are in a hurry. They fire all ten 100-ETH deposits within a 40-minute window, then over the next two hours make ten 100-ETH withdrawals to ten fresh addresses.

Here is the analyst's read. During that afternoon the 100-ETH pool — a thin, whale-only pool — saw essentially *no other activity*. So the analyst has ten deposits and ten withdrawals, all 100 ETH, all clustered in a three-hour window with nobody else in the set. The anonymity set the launderer *thinks* they have is "everyone who ever used the 100-ETH pool"; the anonymity set they *actually* have is "the ten people in this window" — and since all ten are the same person, the set is **1**. A \$1,000,000 movement that should have hidden among thousands instead self-identified by clustering ten round withdrawals into a three-hour box. The lesson: a fixed-denomination pool only protects you if the crowd is real and you wait; batch ten round chunks into one afternoon and you have drawn a box around your own money.

#### Worked example: a 96.4-ETH withdrawal that matches a 96.4-ETH deposit

A second user needs to move an awkward total: **96.4 ETH**, worth about **\$289,200** at \$3,000/ETH. Fixed denominations cannot express 96.4 directly, so they compose it: nine 10-ETH deposits (90), six 1-ETH deposits (6), and four 0.1-ETH deposits (0.4) — a multiset of (9×10, 6×1, 4×0.1). Weeks later, needing the funds, they withdraw the *same total*, 96.4 ETH, reassembled from the same kinds of chunks to fresh addresses.

The analyst does not need timing here; the *amount itself* is the fingerprint. A withdrawal cluster that sums to exactly 96.4 ETH, composed of the same odd mix of denominations as a known deposit cluster, is astronomically unlikely to be coincidence — round numbers are common, but 96.4 assembled from (9, 6, 4) of three denominations is near-unique across the whole pool's history. A **\$289,200** trail that the mixer was supposed to sever is re-tied by arithmetic alone. The intuition: the instant you need a non-round number out of a round-number pool, the *shape* of how you build it becomes a serial number.

### The Tornado Cash sanctions: sanctioning code

On **8 August 2022**, OFAC (the US Treasury's Office of Foreign Assets Control) added Tornado Cash to its sanctions list — not a person, not a company, but a set of immutable Ethereum smart-contract addresses. This was genuinely novel and remains controversial. Sanctioning a *tool* (autonomous code that anyone can call and no one can stop) rather than an actor raised hard questions: code is speech-adjacent; the contracts are immutable and ownerless; and plenty of the volume was legitimate users seeking ordinary financial privacy. We treat the legal and free-speech dimensions in depth in the dedicated post, [Tornado Cash and sanctioning code](/blog/trading/crypto/tornado-cash-and-sanctioning-code) — this post stays on the *forensic* meaning of the event.

For an analyst, the sanction did three concrete things. First, it made *interacting* with the contracts a compliance event in itself: every regulated exchange now screens deposits for any history of touching the Tornado addresses, so even a perfectly-mixed coin carries a "touched a sanctioned contract" flag the moment it tries to cash out. Second, it caused the legitimate crowd to flee, which **shrank the anonymity set** for everyone left behind — fewer honest users means the remaining flow is both more illicit-weighted and thinner, which makes timing/amount correlation *easier*, not harder. Third, it pushed front-end access offline, but the immutable contracts kept running, so the on-chain footprint persisted for analysts to study. The grim irony for a launderer: sanctioning the mixer made the surviving pool a worse place to hide.

#### Worked example: the lifetime numbers — \$7B through the pool, ~30% illicit

Over its life, Tornado Cash processed on the order of **\$7,000,000,000** (\$7B) in deposits, and Elliptic attributed roughly **30%** of that — about **\$2,100,000,000** — to illicit sources (hacks, scams, sanctioned entities). Flip that around: roughly **\$4,900,000,000**, about 70%, came from users who were not criminals — people seeking ordinary financial privacy, traders hiding strategy, donors avoiding retaliation.

That 70% is not a footnote; it is the *mechanism* by which the 30% used to hide. A launderer's privacy was *subsidized* by the honest crowd: the more legitimate 100-ETH deposits sat in the pool, the larger everyone's anonymity set, including the criminal's. Sanctions and screening drove the honest crowd out, which is precisely why the surviving pool leaks more. The chart makes the split concrete.

![Tornado Cash lifetime volume of about 7 billion dollars split into a 30 percent illicit share](/imgs/blogs/mixers-coinjoin-and-obfuscation-6.png)

The intuition an analyst takes from this: **a mixer is a crowd, and you cannot launder in an empty room.** The very thing that makes a mixer ethically defensible — that most of its users are honest — is the thing that, once removed, makes the remaining illicit flow easy to isolate.

### Real episodes: how mixers actually failed their users

Theory convinces no one; dated cases do. Three documented episodes show every leak in this post operating on real money.

**Lazarus through Tornado, after Ronin (2022).** After the **\$625M** Ronin theft, the Lazarus Group moved the stolen ETH into Tornado Cash in tranches over months. Investigators did not break the zero-knowledge proofs; they did exactly what the walkthrough below describes. They watched the thin 100-ETH pool, bracketed deposits and withdrawals by denomination and timing, and tracked the gas that funded the "fresh" withdrawal addresses — which repeatedly traced back to wallets and exchange accounts already in the Lazarus cluster. The OFAC designation of Tornado on 8 August 2022 explicitly cited its use by Lazarus to launder the Ronin proceeds. The forensic point: the mixer did not protect a *disciplined, patient, crowd-hiding* user — it was used at scale, in a thin pool, in a hurry, by an operator who reused infrastructure, and every one of those choices was a leak. Roughly **\$30M** of the Ronin funds were ultimately frozen and clawed back through the off-ramp chokepoint by Chainalysis and law enforcement, precisely because the trail survived the mixer.

**Bitcoin Fog and the decade-late arrest (2021).** Bitcoin Fog was an early Bitcoin mixing service that ran from 2011. Its operator was arrested in 2021 — a decade later — after IRS-CI investigators reconstructed the service's flows using clustering and the permanent public ledger. The lesson that matters for an analyst is the asymmetry of time: **the chain does not forget.** A mixer can delay attribution for years, but the records sit there permanently, and as analytics tooling and cross-referenced KYC data improve, old "anonymous" flows become walkable. Obfuscation is a bet that the trail will go cold before the tooling catches up; that bet has repeatedly lost.

**Wasabi/CoinJoin and exchange screening.** Several large exchanges began, around 2020–2021, flagging or restricting deposits with a CoinJoin history — not because CoinJoin is illegal (it is a legitimate privacy technique with many lawful users), but because the *off-ramp* is where the policy bites. A CoinJoin can genuinely give a Bitcoin user a real anonymity set against amount correlation; what it cannot do is change the fact that the regulated venue at the end screens for the CoinJoin fingerprint and for downstream consolidation. The privacy held on-chain and failed at the edge — which is the recurring shape of every story in this post.

The common thread across all three: **the mixer worked exactly as designed on-chain, and it still leaked**, because the leaks were never in the cryptography. They were in the timing, the reused infrastructure, the consolidation, and the unavoidable cash-out.

## CoinJoin: equal outputs and statistical demixing

Bitcoin's UTXO model invites a different obfuscation shape. There is no shared smart-contract pool (Bitcoin has no general smart contracts of that kind); instead, several users **collaboratively build one transaction**. This is CoinJoin, popularized by the Wasabi and Samourai/Whirlpool wallets.

**How a CoinJoin muddies the trace.** Recall the common-input-ownership heuristic: normally, all inputs to a Bitcoin transaction are assumed to belong to one wallet. A CoinJoin deliberately violates that. Five users each contribute an input, and the transaction produces five **equal-sized outputs** — say five outputs of 0.1 BTC each. Now the common-input heuristic is wrong (the five inputs belong to five different people), and because every output is the same size, no output can be matched to a specific input by amount. An observer sees one transaction with five inputs and five identical outputs and cannot say which input funded which output: with five equal outputs there are 5! = 120 possible mappings, and nothing on-chain prefers one. The figure contrasts the ordinary, trivially-traceable case with the equal-output CoinJoin.

![Before an ordinary Bitcoin tx with distinct amounts and after a CoinJoin with five equal outputs](/imgs/blogs/mixers-coinjoin-and-obfuscation-3.png)

**Where CoinJoin leaks — statistical demixing and clustering.** CoinJoin is genuinely stronger than a thin Tornado pool against *amount* correlation, because the equal outputs are designed to defeat exactly that. But it leaks in other ways, and the leaks are statistical rather than absolute:

- **Sub-set sum analysis.** Real CoinJoins are rarely perfectly equal. There is usually a "change" output (the leftover above the standardized amount), and change outputs are not standardized — so they can be matched back to their input by amount, partially de-anonymizing that participant. Tools search for subsets of inputs that sum (minus fees) to specific outputs.
- **Post-mix consolidation.** The single biggest leak is what users do *after*. If someone receives three 0.1-BTC "mixed" outputs and later spends them together in one transaction, the common-input heuristic re-links all three — the post-mix consolidation undoes the mix. Analysts specifically watch for outputs of a CoinJoin being merged downstream.
- **Wallet-fingerprinting.** Specific CoinJoin implementations have recognizable transaction structures (number of participants, exact standard denominations, coordinator fee outputs). That fingerprint tells an analyst "this was a Wasabi/Whirlpool round," which scopes the analysis and, combined with timing, narrows participants.
- **Re-mixing the same coins.** Repeatedly mixing then partly de-mixing through behavior can, across many rounds, leak more than a single disciplined round would.

So CoinJoin demixing is probabilistic: the analyst assigns likelihoods across the 5! mappings and then uses change outputs, downstream consolidation, and timing to collapse the distribution. It is harder than demixing a thin Tornado pool, but "harder" is not "impossible," and the off-ramp still waits at the end.

**Why the implementation details matter to an analyst.** The two best-known Bitcoin CoinJoin designs leak differently, and recognizing which one you are looking at scopes the investigation. Wasabi-style rounds historically used larger numbers of participants with a coordinator and produced many equal outputs plus change — strong against amount correlation but exposing the coordinator's fee output and the change seam. Whirlpool-style rounds used small, fixed-size "pools" (for example, standardized pool denominations) and a structure designed to be re-mixed repeatedly so that an output of one round becomes an input of the next. That re-mixing is double-edged: each additional round *can* grow the theoretical anonymity set, but it also creates a long behavioral trail of the same coins cycling through the same service, and if the operator ever consolidates across rounds, the whole sequence collapses at once. An analyst who fingerprints the round type immediately knows where to look — change outputs and coordinator fees for one design, cross-round consolidation for the other.

**Anonymity-set decay.** A subtler leak is that a CoinJoin's nominal anonymity set *decays* as the other participants in your round spend their outputs in identifiable ways. If three of your four co-participants later send their mixed coins straight to KYC exchanges that tag them, the analyst can often subtract those three from your set, shrinking your effective anonymity from 5 toward 2 — without ever touching your coin. Your privacy in a CoinJoin is partly hostage to the discipline of strangers, and most strangers are not disciplined. This is the CoinJoin analogue of the thin-pool problem: the crowd that protects you can thin out behind your back.

#### Worked example: a five-party CoinJoin and the change-output leak

Five users join a Whirlpool-style round, each standardizing to a **0.1 BTC** output. One participant, call them user E, actually held **0.137 BTC** and wanted to mix it. The round produces five identical 0.1-BTC outputs plus, for user E, a **0.037 BTC** change output (minus a small fee). At a BTC price of **\$60,000**, the standardized output is **\$6,000** and E's change is about **\$2,220**.

The five 0.1-BTC outputs are genuinely ambiguous — the analyst cannot tell E's from the other four, a real 5-way anonymity set. But the **0.037 BTC change output is not standardized**, so it can be matched by amount straight back to E's original input. E mixed \$6,000 cleanly and simultaneously leaked the *existence and ownership* of the remaining \$2,220 through the change. If E later spends the mixed 0.1 BTC together with that change, the consolidation re-links the whole \$8,220. The intuition: CoinJoin protects the *standardized* slice and quietly betrays the *remainder* — the change output is the seam where equal-output privacy frays.

## Peel chains: pseudo-mixing by chain-hopping

The third family is not real mixing at all. A **peel chain** is a long sequence of transactions where a large balance is moved from one fresh address to the next, "peeling" off a small amount to a cash-out point at each hop, while the bulk — the **remainder** — keeps marching forward. The hope is purely psychological: that an analyst walking the chain will lose patience, lose the thread, or be defeated by sheer length. It provides no real anonymity set; it just adds hops.

![A peel chain forwards a large remainder and peels a small slice to a cash-out at each hop](/imgs/blogs/mixers-coinjoin-and-obfuscation-4.png)

**Why peel chains leak so badly.** A peel chain is the *easiest* of the three to follow once you anchor on it, because it has a glaring regularity: at every hop there is one large output (the remainder, the "heir") and one small output (the peel). An analyst follows the heir. The chain is one address wide — there is no branching crowd to hide in, no equal outputs, no pool. Worse, the peeled amounts are often near-constant (the launderer scripts the same slice each hop), and that cadence is itself a fingerprint that says "this is an automated peel chain, keep walking." Chain-hopping — bouncing the funds across bridges to other chains — adds the bridge seam as another leak point, which we cover in [cross-chain tracing: bridges and the USDT rails](/blog/trading/onchain/cross-chain-tracing-bridges-and-the-usdt-rails); bridges create a deposit-here/withdraw-there pattern that, like a thin mixer, leaks on amount and timing.

The clustering heuristics that make peel chains walkable — common-input ownership, change-address detection, and behavioral fingerprints — are the subject of [address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics); a peel chain is essentially a clustering problem dressed up as obfuscation.

#### Worked example: a peel chain leaking \$50k per hop

A launderer starts with **1,000 ETH** (about **\$3,000,000** at \$3,000/ETH) in a tainted wallet and builds a peel chain: at each hop, forward the remainder to a fresh address and peel **\$50,000** (about 16.7 ETH) to a small cash-out. After hop 1, the heir holds ~983 ETH; after hop 2, ~967 ETH; after hop 3, ~950 ETH — the remainder marches down in a tidy staircase.

The analyst's read is almost insultingly simple: follow the big remainder. The chain is one address wide, so there is no anonymity set to defeat — at each hop you take the larger output and keep going. The constant **\$50,000** peel is the fingerprint that confirms "automated peel chain, single operator," and the *peeled* amounts are themselves leads, because each \$50,000 slice runs off to a cash-out that may attach to an identity. The launderer added fifty hops of busywork and zero real privacy; the analyst added one bookmark per hop. The intuition: length is not anonymity — a peel chain trades the analyst's *time* for nothing, because the remainder is a single thread you simply pull.

## The off-ramp chokepoint: where identity finally attaches

Here is the leak that closes nearly every case, no matter how disciplined the mixing was. **Crypto is not spendable money.** A laundered, perfectly-mixed coin is still just a token sitting in a wallet — it cannot pay rent, buy a car, or settle a debt in the real economy. To become *useful*, it has to be converted to fiat (or a spendable stablecoin), and the overwhelming majority of that conversion happens at **regulated, KYC-ed exchanges** — venues that, by law, collect a real identity (ID, selfie, bank account) before they let you cash out.

That conversion point is the **chokepoint**. The exchange runs every incoming deposit against blockchain-analytics feeds; if your "clean" funds trace back — even through a mixer — to a flagged cluster or a sanctioned contract, the deposit is frozen, a Suspicious Activity Report is filed, and the real identity that just passed KYC is now attached to the tainted money. The figure traces this: the obfuscation layer buys distance from the source, but the regulated cash-out is where a name attaches.

![Mixed funds must pass through a KYC exchange chokepoint where a real identity attaches to the money](/imgs/blogs/mixers-coinjoin-and-obfuscation-5.png)

**This is why obfuscation is a delay, not an escape.** Every dollar of laundered crypto faces the same terminal problem: to spend it, you must touch a regulated edge, and at that edge identity attaches and analytics screen the history. There is a structural tradeoff the launderer cannot win: the *more* hops and mixing they add to put distance between themselves and the source, the *longer* the funds sit unspent and the more time the defender's labeling dataset has to enrich — and the moment they finally cash out, all that delay is converted into a single, screened, identity-bearing deposit. Speed exposes you to thin-pool timing leaks; patience exposes you to an ever-improving dataset. The only escape would be to never convert to spendable money at all, which defeats the entire purpose of stealing it. Laundering is therefore not a problem with a clean solution for the criminal; it is a problem with a built-in terminal leak, and the defender's job is simply to keep the trail alive until that terminal is reached. Launderers respond by seeking *gaps* — no-KYC exchanges, peer-to-peer OTC desks, nested exchanges (accounts at a big exchange resold to others), gambling sites, and lax jurisdictions. But these gaps are exactly what defenders monitor, the off-ramp surfaces are finite and increasingly licensed, and the moment funds re-enter the regulated system the trail resumes. The interplay of freezing, recovery, and the analytics firms that screen these deposits is the subject of [freezing, recovery, and chain analytics](/blog/trading/onchain/freezing-recovery-and-chain-analytics); the broader laundering route — hack, swap, bridge, mix, cash-out — is mapped in [how stolen funds are laundered](/blog/trading/onchain/how-stolen-funds-are-laundered).

#### Worked example: a \$2M cash-out where the identity attaches

A launderer successfully mixes funds and, over weeks, accumulates **\$2,000,000** in fresh, clean-looking wallets — no direct on-chain link to the original hack. Confident, they deposit it to a major regulated exchange in **\$200,000** tranches to convert to dollars and withdraw to a bank.

The exchange's compliance system screens each deposit against analytics feeds. Even though the *immediate* sending address is clean, the funds' history shows contact with a flagged mixer cluster and, in the Tornado case, a *sanctioned contract*. The first **\$200,000** tranche trips the screen; the exchange freezes the account, files a SAR, and hands law enforcement the **KYC identity** — the same name, ID, and bank account the launderer registered. The remaining **\$1,800,000** is now associated with that identity and frozen or flagged across the industry's shared lists. Weeks of careful mixing collapsed at the one point that could not be avoided: turning tokens into money. The intuition: mixing can sever the *on-chain* link, but it cannot sever the *economic* link — to spend laundered crypto, you must walk up to a regulated counter and show your face.

### The gaps launderers seek, and why defenders watch them

Because the regulated off-ramp is so reliably the failure point, sophisticated launderers spend most of their effort trying to find an exit that *isn't* a compliant, screening exchange. It is worth naming these gaps from the defender's side — not as a route, but as a watch-list, because each gap is itself a monitored surface.

**Nested exchanges and instant-swap services.** A "nested" service is an account or sub-business operating *inside* a larger exchange, reselling that exchange's liquidity to its own customers with little or no KYC. Funds flow into the big exchange through the nested service's account, so on the surface the deposit looks like ordinary exchange activity. Defenders counter by clustering deposits behind the nested service's known addresses and pressuring host exchanges to off-board high-risk nested accounts; several large laundering pipelines have collapsed when the host exchange froze the nested account's master wallet.

**OTC desks and peer-to-peer.** Over-the-counter brokers and P2P marketplaces can convert crypto to cash with weaker checks, especially in lax jurisdictions. But OTC desks still need banking relationships somewhere, and those banks sit inside the regulated perimeter; the trail re-attaches at the bank rail even when it briefly escaped the crypto rail.

**High-risk and no-KYC venues.** Gambling sites, no-KYC swap services, and exchanges in non-cooperative jurisdictions are perennial laundering destinations. Analysts maintain attribution lists for these, so a deposit *into* one is itself a risk signal — the funds did not disappear, they moved to a labeled high-risk cluster that compliance systems flag on the next hop.

The pattern is always the same: the gaps are finite, increasingly licensed, and individually monitored, so the funds re-enter the screened system within a hop or two. Obfuscation can change *where* the off-ramp is; it cannot remove the need for one.

## How to read it: an investigator's walkthrough

Let us put it together as a concrete pass, the way an analyst actually works a mixed trail. None of this requires special access — it is reading public chain data with the right correlations in mind.

**Step 1 — confirm the entry and bracket by amount + timing.** You have a known-tainted wallet (say, a hacker's) that deposited into Tornado's 100-ETH pool. Pull every deposit and withdrawal in that pool around that time. For each tainted deposit, list the withdrawals of the same denomination in the following window and rank them by closeness in time. In a thin pool, the candidate set is tiny; sometimes the deposit-and-withdrawal pair is alone in the window, which is a near-certain link. You are not breaking the proof — you are reading the timestamps the proof cannot hide.

**Step 2 — collapse candidates with the relayer and gas-funding side channels.** For each candidate withdrawal address, ask: who paid its gas? If a relayer submitted it, note the relayer and fee — and find every *other* withdrawal that used the same relayer at the same fee, because those are now a cluster. If the fresh address was funded with gas from somewhere, trace that funding: a gas top-up from the hacker's other wallet, or from the same exchange as the deposit, collapses the candidate to a single answer. This is usually the step that converts "probable" into "certain."

**Step 3 — spot a peel chain when the mixer is bypassed.** If the funds did not pool but instead hopped through fresh addresses, check each hop for the peel signature: one large output (the heir) and one small output (the peel). If you see that, stop treating it as mixing — follow the heir hop by hop, bookmark each peel for separate investigation, and watch for the constant peel amount that marks an automated chain. A peel chain is long but linear; patience, not cleverness, walks it.

**Step 4 — read a CoinJoin's equal outputs honestly.** If you hit a CoinJoin, do not over-claim. The equal outputs are a *real* anonymity set; mark them as ambiguous. Then look for the leaks: a non-standard change output you can match by amount, and — most importantly — downstream consolidation where two or more of the round's outputs get spent together, which re-links them. Assign probabilities, not certainties, until a consolidation or change-match collapses them.

**Step 5 — race to the off-ramp.** Regardless of which family you are tracing, identify the exit. Follow each thread to the first regulated, KYC-ed venue it touches. That deposit is the chokepoint where subpoenas and compliance freezes turn pseudonymous addresses into a named person. The whole purpose of steps 1–4 is to keep the thread alive *until* it reaches step 5.

The decision matrix below summarizes which side channel breaks which method — the analyst's cheat sheet for matching the leak to the tool, with the KYC off-ramp as the leak every method shares.

![Decision matrix mapping timing amount structure and off-ramp leaks across mixer CoinJoin and peel chain](/imgs/blogs/mixers-coinjoin-and-obfuscation-7.png)

## How analytics firms operationalize demixing at scale

A lone analyst can bracket one withdrawal by hand, but the reason mixers leak *systematically* is that firms like Chainalysis, Elliptic, and TRM Labs run these correlations across the entire chain, continuously, as a standing dataset. Understanding their pipeline tells you why "I used a mixer" is rarely the end of the story.

First, they **label the edges of the graph**: every known exchange deposit address, every sanctioned contract, every documented hack wallet, every mixer pool, every relayer, and every high-risk venue gets an attribution tag. This labeling is the expensive, proprietary part, and it is what turns a raw transaction graph into a *map with named landmarks*. A withdrawal address means little until you know it was funded with gas from a wallet that the dataset already tags as "Lazarus cluster."

Second, they **precompute the mixer-spanning candidate links**. For each pooled-mixer deposit, the system stores the set of same-denomination withdrawals within plausible time windows, ranked by timing tightness; for each CoinJoin, it flags the non-standard change outputs and tracks downstream consolidation; for peel chains, it follows the heir automatically and bookmarks every peel. None of this is a cryptographic break — it is bulk correlation, run once and queried many times.

Third, they **score taint with a chosen model** (poison, haircut, or FIFO) and expose a risk score on any address, which is what an exchange's compliance system actually queries at deposit time. The exchange does not re-run the investigation; it asks "what is the risk score and attribution of this incoming address?" and acts on the answer. This is the machinery behind the off-ramp chokepoint: by the time funds arrive at the cash-out, the demixing has *already been done* and cached, waiting for the deposit to show up.

The strategic consequence for anyone reading the chain: the asymmetry favors the defender over time. A launderer must get *every* hop right, *forever*, against a dataset that only improves and never forgets; a defender needs the trail to survive to a single off-ramp once. The tooling that powers all of this is surveyed in [the on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape).

#### Worked example: a thin-pool withdrawal scored before it even arrives

Suppose a launderer withdraws **10 ETH** (about **\$30,000** at \$3,000/ETH) from a sparsely-used pool to a fresh address, waits a week, then deposits it to a regulated exchange to cash out. They feel safe — a week of delay, a fresh address, no obvious link.

But the analytics firm precomputed the candidate link the moment the withdrawal happened: in that thin pool, the only same-denomination deposit in the prior window was a tagged hack wallet, so the withdrawal address inherited a high risk score *the day it was created* — long before the exchange deposit. When the **\$30,000** lands at the exchange a week later, the compliance system's query returns "high risk, traces to flagged 10-ETH deposit from a known hack cluster," and the deposit is frozen on arrival. The week of patience bought nothing, because the demixing did not happen *at* cash-out — it happened *at withdrawal* and sat cached. The lesson for an analyst: in a thin pool, the link is often established the instant the withdrawal fires, and delay only gives the defender more time to enrich the label.

## Common misconceptions

**"A mixer makes the money untraceable."** No. A mixer breaks the *direct* on-chain link at one hop, but it leaks side channels — timing, amount, relayer, gas-funding, consolidation — and every path eventually hits the KYC off-ramp. The Ronin funds were followed *through* Tornado Cash. Untraceable is marketing; traceable-with-effort is reality, and the effort is mostly correlation, not cryptography.

**"Tornado's zero-knowledge proof hides everything."** The proof hides *which deposit funded which withdrawal* cryptographically — that part is real. It does **not** hide the wall-clock timestamps, the denominations, who paid gas, or who the relayer was. In a thin pool, those unhidden facts *are* the mapping. The cryptography is strong; the metadata around it is not.

**"A bigger pool is always enough."** Pool size sets the *maximum* anonymity set, but user behavior sets the *actual* one. Deposit and withdraw minutes apart, fund your fresh address's gas from your main wallet, reuse a withdrawal address, or consolidate "anonymous" outputs together, and you collapse a 1,000-deep pool to an anonymity set of one. Privacy tools punish impatience and reuse mercilessly.

**"CoinJoin is unbreakable because the outputs are equal."** The equal outputs are genuinely strong against amount correlation, but CoinJoin leaks through non-standard change outputs, post-mix consolidation, and wallet fingerprinting. Demixing a CoinJoin is statistical and harder than a thin mixer — but "harder" is not "impossible," and the off-ramp still ends the case.

**"Sanctioning Tornado made laundering safer by pushing it underground."** It did the opposite for the on-chain footprint: it drove the *honest* crowd out, shrinking the anonymity set for everyone left, while flagging any coin that ever touched the contract. The immutable code keeps running and keeps leaving a public footprint. For a launderer, a sanctioned mixer with a fleeing crowd is a *worse* hiding place, not a better one.

## The playbook: what to do with it

For the trader/investor and the analyst/defender, here is the if-then checklist.

**If a token or counterparty's funds trace back through a mixer →** treat it as a *risk flag*, not a verdict. Roughly 70% of historical Tornado volume was legitimate, so "touched a mixer" alone does not prove crime. **The read:** look for *corroborating* taint — does the trail go back to a documented hack or a sanctioned cluster? **The action:** if you are an exchange or counterparty, screen and, where it traces to flagged or sanctioned funds, freeze and report; if you are an investor, treat mixer-adjacent project treasuries as elevated risk and size accordingly. **The false positive:** privacy-seeking but lawful users; do not assume guilt from a mixer touch alone.

**If you are tracing a known-illicit deposit into a pooled mixer →** bracket withdrawals by denomination, then by timing in the thin pool, then collapse with relayer and gas-funding links. **The action:** rank candidates by combined timing+funding evidence and follow the strongest to the off-ramp. **The invalidation:** a genuinely deep pool with long, randomized delays and clean gas funding — then the link is probabilistic, and you escalate to the exit rather than over-claiming a specific mapping.

**If the funds hopped through fresh addresses instead of pooling →** identify the peel signature (large heir + small peel) and follow the heir. **The action:** bookmark each peel as a separate cash-out lead; watch for the constant peel amount marking an automated chain. **The invalidation:** genuine branching into many comparable outputs (closer to a real mix) rather than a single heir — then re-assess whether it is a peel chain or a true split.

**If you hit a CoinJoin →** mark the equal outputs as ambiguous and resist over-claiming. **The action:** hunt the non-standard change output and any downstream consolidation that re-links the round's outputs. **The invalidation:** a clean round with no consolidation and standardized change — then it stays a probabilistic set, and you rely on the off-ramp.

**Always, regardless of method →** race the thread to the first KYC off-ramp. **The action:** the regulated cash-out is where identity attaches; keep the trail alive until it gets there. **The rule of thumb to remember:** obfuscation buys *distance from the source*, never *invisibility* — and a defender's entire edge lives in that gap.

## Further reading & cross-links

- [How to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow) — the one-to-one link walking that mixers attack.
- [Address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics) — common-input ownership and change detection, the engine behind demixing and peel-chain walking.
- [How stolen funds are laundered](/blog/trading/onchain/how-stolen-funds-are-laundered) — the full hack → swap → bridge → mix → cash-out route this post zooms into.
- [Cross-chain tracing: bridges and the USDT rails](/blog/trading/onchain/cross-chain-tracing-bridges-and-the-usdt-rails) — chain-hopping as a peel-chain extension and where the bridge seam leaks.
- [Tornado Cash and sanctioning code](/blog/trading/crypto/tornado-cash-and-sanctioning-code) — the legal and free-speech dimensions of the OFAC sanction.
- [Freezing, recovery, and chain analytics](/blog/trading/onchain/freezing-recovery-and-chain-analytics) — what happens at the off-ramp chokepoint once a tainted deposit lands.
