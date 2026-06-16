---
title: "Holder Analysis for Memecoins: Bundles, Snipers, and Fresh-Wallet Clusters"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A memecoin has no fundamentals, so its holder structure is the only real signal. Learn the Solana/pump.fun forensic read — bundle detection, snipers, dev holdings, fresh-wallet clusters, and Bubblemaps — to tell a rigged launch from an organic one before you ape."
tags: ["onchain", "crypto", "memecoins", "solana", "pump-fun", "bubblemaps", "bundle-detection", "snipers", "holder-analysis", "due-diligence", "rug-pull"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A memecoin has no cash flows, no product, and no moat, so the only honest signal you have is its **holder structure** — and that structure is fully public on-chain.
>
> - The four killers to read on every launch: **bundling** (the dev secretly buys its own token across many wallets at launch to fake distribution), **snipers** (bots that grab a large share in block 0-1 before any human can), **dev holdings** (how much the deployer and its linked wallets still hold, and whether they are selling), and **fresh-wallet clusters** (brand-new single-purpose wallets that are one entity wearing a crowd's clothes).
> - The tool: a holder list (Solscan), a cluster map (**Bubblemaps**), and funding-source tracing — you reclassify the top holders into "real crowd" vs "one entity", then ask whether the apparent distribution survives that reclassification.
> - What you DO with it: an organic structure earns the right to a *tiny, -EV-sized bet*; a one-funder cluster holding 30%+ is a skip, full stop. Holder analysis is **risk reduction, not alpha**.
> - The number to remember: roughly **8 million** tokens have launched on pump.fun and only about **1.4%** ever reach a meaningful market cap. You are betting against a brutal base rate; do not pretend otherwise.

On a Solana memecoin launch, the chart that everyone stares at — green candles, volume spiking, holder count ticking up into the thousands — is the least informative thing on the screen. Price is one number. Volume can be two bots passing the same coin back and forth. The holder count is just a row of addresses, and addresses are free. What actually decides whether a launch is a coin or a trap is *who those addresses are and where their money came from* — and that, unlike the marketing, is written permanently into the ledger where you can read it.

Consider the shape of the typical pump.fun rug, which has played out tens of thousands of times. A token deploys. In the same block, before any human has even seen the ticker, a cluster of fresh wallets — all funded minutes earlier from one address — buys 35% of the supply. The chart looks like organic demand. Influencers get paid to post. Real buyers ape in at a price the insiders already set. Twenty minutes later the cluster sells into that demand, the chart goes vertical-down, and the "thousands of holders" turn out to have been a dozen wallets and a crowd of victims. None of this was hidden. Every transfer, every funding link, every block-0 buy was on-chain the whole time. The victims just did not look.

This post teaches you to look. We will build the memecoin-specific forensic read from zero — what bundling, sniping, dev overhang, and fresh-wallet clustering actually *are* on Solana and pump.fun, how to detect each one with real tools, and how to turn the read into a decision before you risk a dollar. The mental model to hold the whole time is the contrast in the figure below: two launches with identical charts and wildly different holder structures.

![Organic launch versus rigged launch shown as two holder structures](/imgs/blogs/holder-analysis-for-memecoins-1.png)

The organic launch on the left is a wide, independent crowd: many funding sources, aged wallets, no single holder dominating, a small or renounced dev. The rigged launch on the right is a dense cluster behind one funder, dressed up to look distributed. Your entire job is to figure out which one you are looking at — and the rest of this post is how.

## Foundations: what a memecoin is and why holder structure is the only real signal

Let us define every term from the ground up, because the whole discipline rests on understanding what a memecoin actually *is* — and, more importantly, what it is not.

**A memecoin is a token with no fundamentals.** A normal company has revenue, assets, employees, and a product; you can value it by discounting its future cash flows. A memecoin has none of that. It is a token — a row in a smart contract's ledger — whose entire value is *attention multiplied by liquidity*. There is no business behind DOGE or PEPE or the thousands of dog-and-frog tokens minted every day. The price is purely a function of how many people want in versus out at any moment, and how much money is pooled to let them trade. That is not a criticism; it is the definition. A memecoin is a coordination game on attention, settled on-chain.

Because there are no fundamentals, the usual analyst toolkit is useless. There is no P/E ratio, no revenue trend, no management to evaluate. The price tells you almost nothing because in the early minutes it is set by whoever controls the most supply, not by any consensus on value. So what is left to analyze? Only one thing: **the holder structure** — the answer to "who owns this, how much, and are they one entity or a real crowd?" That is the only signal that is both hard to fake (because it is recorded on-chain) and genuinely predictive (because a launch where insiders hold the supply is a launch designed to dump on you). Everything in this post flows from that single fact.

**The chain is public, permanent, and pseudonymous.** Every transfer of a Solana token, every wallet that funded another wallet, every buy in the launch block, is recorded forever on the public ledger and visible on a block explorer like Solscan. Wallets are not names — that is the *pseudonymous* part — but they are not anonymous either. A wallet's funding source, its age, the wallets it transfers to, and the exact block it bought in are all public. That is what lets you deanonymize a "distributed" holder list back into the handful of entities that actually control it.

Now the four specific things you are hunting for.

**Bundling.** A "bundle" on Solana is a set of transactions submitted together to be included in the same block, atomically, in a chosen order — a feature originally meant for legitimate uses like arbitrage. The abuse: a dev who has just deployed a token uses a bundle to buy its *own* token across many wallets in the very first block. To the holder list, it looks like dozens of independent buyers piled in at launch. In reality it is one person, funding many wallets from one source, buying their own supply to fake distribution — and holding it ready to dump once real money arrives. Bundling is the single most important thing to detect on a pump.fun launch.

**Snipers.** A "sniper" is a bot that watches for a token to go live and buys in block 0 or block 1 — the instant trading opens, faster than any human can click. Sniping is not always malicious (some snipers are independent traders racing each other), but a launch where snipers grabbed, say, 15% of supply in the first block means anyone buying afterward is buying at a price the snipers set, and is likely to be the snipers' exit liquidity. Block-0/1 buyers are public; you can see exactly how much of the supply they took.

**Dev / team holdings.** The "dev" is the deployer — the wallet that created the token contract and paid the deploy fee. The question is how much supply the dev and its *linked* wallets (ones it funded, or that share a funder with it) still hold, and whether they are selling. A dev sitting on 20% of supply is a permanent overhang: a giant sell order that can land any time. A dev who has already routed tokens to an exchange deposit address is *actively* selling into you.

**Fresh wallets.** A "fresh wallet" is a brand-new address with no prior history — created shortly before the launch, holding only this one token, funded from a single source. One fresh wallet is normal (everyone starts somewhere). A holder list where *dozens* of fresh wallets share the same age, funder, buy size, and timing is not a crowd; it is one entity, and a "fresh-wallet cluster" is one of the clearest tells of a coordinated, insider-controlled launch.

**The pump.fun / Solana context.** Solana is a high-throughput, low-fee blockchain, which makes it the natural home for memecoins — launching a token costs cents and trades settle in under a second. **Pump.fun** is a launchpad on Solana that lets anyone create a token in seconds, with a "bonding curve" that automatically provides liquidity and supposedly makes rugging harder (the dev cannot simply pull the liquidity pool, because the curve holds it until the token "graduates" to a real DEX pool). It removed much of the friction of launching, which is exactly why millions of tokens have been minted on it — and why the survivorship is so brutal. Pump.fun reduced one specific rug vector (liquidity pulls) but did *nothing* about bundling, sniping, dev overhang, or insider clusters. Those are precisely the holder-structure problems this post addresses.

**The bonding curve, in concrete terms.** Because pump.fun's bonding curve shapes every launch you will analyze, it is worth understanding the mechanics, not just the name. When a token is created, it starts with a fixed supply (typically about 1 billion tokens) and no normal liquidity pool. Instead, the contract itself acts as the market maker along a predetermined price curve: the more tokens that have been bought out of the curve, the higher the price of the next token. Early buys are cheap and move the price a lot; later buys are more expensive and move it less. The SOL paid in is locked in the curve, which is why a dev cannot simply withdraw the liquidity in the early phase. When enough has been bought that the token's value crosses a threshold (the "graduation" level, historically around a market cap in the tens of thousands of dollars), the accumulated liquidity is deposited into a real decentralized-exchange pool and the token "graduates" to trading there.

Two consequences of this design drive everything in this post. First, *the earliest buyers get a structurally lower price than everyone after them* — which is precisely why sniping and bundling are so profitable: getting in at block 0 means getting in at the cheapest point on the curve, before the price has climbed. Second, *the vast majority of tokens never graduate* — they are bought a little, dumped, and abandoned far below the graduation threshold, which is the mechanical reason the survivorship number is so low. The curve removed the friction of providing liquidity, which democratized launching; it did not remove the incentive to grab the cheap early supply and sell it to whoever comes next.

**Free float versus controlled supply.** One last piece of vocabulary, because it is where the analysis lands. The *free float* is the supply that genuine, independent holders can sell at any time — the real market. *Controlled supply* is everything in the hands of the dev, the bundle, and the snipers: tokens that look like part of the market but are really one or a few operators' inventory. The entire forensic read reduces to one question: *after you strip out the controlled supply, how much real free float is there, and who holds it?* A launch where 50% of supply is controlled has a free float half the size the chart implies, and a much worse supply-demand balance than it looks. Hold that framing — controlled versus free — through every section that follows.

With the vocabulary in place, let us go deep on each detection, in the order you would actually run them.

## Bundle detection: one funder buying its own launch

Bundling is the deception that makes a rigged launch look organic, so it is where you start. The mechanism is simple and the on-chain footprint is loud once you know to look.

The dev funds a fleet of fresh wallets from one source — sending, say, a few hundred dollars of SOL to each of twenty wallets. Then, in the launch transaction (or a bundle submitted in the same block), the dev buys the token across all twenty wallets at once. The holder list now shows twenty buyers who all got in at the very bottom. To a beginner scrolling Solscan, that looks like twenty smart, early, independent participants. It is one entity holding a large share of supply, spread across wallets specifically to hide the concentration.

![Bundle detection showing one funder seeding many launch-block buyers](/imgs/blogs/holder-analysis-for-memecoins-2.png)

The figure traces the pattern: one dev source funds wallets 1 through 20, all of which buy in the same launch block, and the result is roughly 35% of supply controlled by one entity but spread to look distributed. Your read is at the end — fake distribution, so avoid or size tiny.

**How to detect it.** Two signatures give bundling away, and you want both. First, **common funding**: pull the top holders' wallets and check each one's *first inbound transfer*. If twenty of them were all funded by the same address within the same window, that is one entity. Second, **same-block buys**: check the block (slot, on Solana) in which each of those wallets first bought the token. If they all bought in the launch slot or the next one, in a bundle, that is coordination, not a coincidence. Tools like a bundle-checker (several exist for pump.fun launches), the holder list on Solscan, and a cluster map on Bubblemaps will surface this. The honest version of the check is: *of the top 20 holders, how many trace back to a single funder and bought in the launch block?* If the answer is "most of them", the distribution is fake.

> [!warning]
> A high holder count means nothing on its own. "5,000 holders" can be 4,980 victims and one twenty-wallet cluster that owns a third of the float. Always reduce the holder list to *entities*, not addresses, before you read concentration.

#### Worked example: 20 bundled wallets ready to dump

A token launches at a \$10M fully-diluted market cap. You pull the top 25 holders and trace funding. Twenty of them were each funded with about \$1,500 of SOL — \$30,000 total — from a single address an hour before launch, and all twenty bought the token in the launch slot. Together they hold 35% of supply, which at a \$10M cap is **\$3.5M** of tokens controlled by one entity. Their cost basis is the \$30,000 they spent, so they are sitting on roughly a 100x paper gain and have every incentive to sell. If even half of that 35% hits the market, you are absorbing **\$1.75M** of sell pressure against a thin bonding-curve liquidity that might be \$200,000 deep. The math is not subtle: this is a launch engineered to dump on whoever buys next. The intuition — when one funder controls a third of supply at a 100x markup, the only question is *when* they sell, not *if*.

**Doing the check programmatically.** When you want to scale beyond eyeballing Solscan, the same logic expresses cleanly as a query against an indexed dataset (Dune indexes Solana). The shape of the check is: take the token's first buyers, join each buyer back to the address that first funded it, and count how many distinct buyers collapse onto each funder. A funder that seeded many launch-block buyers is a bundle.

```sql
-- Pseudocode shape of a bundle check (Solana, via an indexed dataset).
-- For one token, find buyers in the launch slot and group them by funder.
with launch_buys as (
    select buyer, amount, slot
    from token_trades
    where token = '<mint address>'
      and slot <= launch_slot + 1          -- block 0 and 1 only
),
funding as (
    select recipient as buyer, sender as funder
    from sol_transfers
    where recipient in (select buyer from launch_buys)
    qualify row_number() over (
        partition by recipient order by block_time asc
    ) = 1                                  -- each buyer's FIRST funder
)
select funder,
       count(distinct buyer)      as wallets_funded,
       sum(amount)                as supply_bought
from launch_buys b
join funding f on b.buyer = f.buyer
group by funder
order by wallets_funded desc;              -- one funder, many wallets = bundle
```

You do not need to write this yourself for every launch — bundle-checker tools and Bubblemaps automate most of it — but understanding the query makes the tools legible. When a dashboard reports a "bundle %", this join is what it is doing under the surface, and knowing that keeps you from treating the number as magic.

**The funder-of-the-funder trap.** A more careful operator does not fund the bundle wallets directly from one obvious address. They route through a layer: one source funds five "hop" wallets, each of which funds four bundle wallets. Now a naive one-hop trace sees five funders, not one, and the concentration looks less alarming. The defense is to trace funding *two or three hops back* and watch for the hops to re-converge — five funders that were themselves all funded by one address is still one entity. This is the memecoin version of the peel-chain idea from laundering analysis: depth in the funding graph is meant to defeat a shallow look, so you go one layer deeper than the obfuscation.

The second-order effect worth internalizing: bundling is *cheaper than ever* and increasingly automated. A dev can spin up the wallets, fund them, and bundle the launch buy with a single script. That means the *base rate* of bundled launches is high — which is exactly why running this check is non-negotiable, and why a *clean* funding trace is genuinely informative (it is rarer than you would hope).

## Sniper detection: who owns the supply before block 2

Snipers are the second supply-grab, and they operate on a different axis than bundling — speed rather than disguise. A sniper bot subscribes to the chain (or the mempool, the pool of pending transactions) and the instant a token's trading opens, it fires a buy in the same block. By the time a human sees the token on a scanner, the snipers have already accumulated.

![Sniper window timeline showing block zero and one buyers taking a large share](/imgs/blogs/holder-analysis-for-memecoins-3.png)

The timeline walks the sequence: block 0 the contract goes live and trading opens; in blocks 0-1 the bots grab their share; from block 2 the first humans ape in at the higher price the snipers set; minutes later the snipers dump into that demand; and your read is that a high block-0/1 share means you are the exit liquidity.

**How to detect it.** This one is concrete and quantitative. For the token, find the buys that occurred in the deploy slot and the one after it, and sum the share of supply they captured. On Solscan you can sort the token's earliest transfers; specialized tools report a "sniper %" directly. The threshold that matters: if block-0/1 buyers hold a *large* share (the rough line is anything in the double digits, and certainly anything north of 15-20%), the price you can buy at is downstream of theirs, and your expected entry is "after the people who got in first and cheapest decide to leave."

Note the important nuance the ethics of this series demand: sniping is *detection*, not a how-to. We are reading the holder list to know whether snipers are present and how much they took, so we can decide *not to be their exit*. We are not building a sniper.

#### Worked example: snipers grab 15% in block 1

A token opens at a \$10M market cap. You check the first two slots and find that sniper bots bought a combined 15% of supply — that is **\$1.5M** of tokens — before block 2. Real buyers, including you if you are not careful, enter from block 2 onward at a price the snipers' \$1.5M of buying already lifted. Say the snipers paid an average price implying an \$8M cap (they bought into the early curve) and you buy at the \$10M cap; their unrealized gain is already about **\$300,000** on their position the moment you arrive. The moment momentum stalls, that 15% is the first to sell. If you put in \$2,000 at the \$10M cap and the snipers' exit knocks the cap to \$6M, you are down to roughly **\$1,200** before any "narrative" even plays out. The intuition — a heavy block-0/1 share tells you the cheapest, fastest money is already in front of you and you are buying its inventory.

**Why the sniper share is so corrosive.** It helps to see the price mechanics. On the bonding curve, the first dollars in are the cheapest, so when snipers take 15% of supply in block 0-1, they take it at the steepest discount the token will ever offer. Every buyer afterward is buying further up the curve, at a price the snipers' buying *created*. That sets up a structural conflict: the snipers' break-even is far below your entry, so they can sell profitably at a price that puts you deeply underwater. The bigger the sniper share, the larger this overhang of cheaply-acquired, sell-ready supply, and the worse your reward-to-risk from the open. A token with 2% sniped and 98% acquired up the curve alongside you has a roughly aligned holder base; a token with 30% sniped has a holder base whose biggest players are incentivized to exit the instant you provide demand.

**The honest counter-case.** Not every sniper is a scam, and pretending so will make you miss the nuance. Some snipers are independent professional traders racing each other for the early discount, with no connection to the dev. A launch can have a meaningful sniper share that is *competitive* rather than *coordinated* — many distinct, well-funded, aged wallets that each grabbed a slice, with no common funder. That is different from a dev sniping its own launch through fresh wallets. The way you tell them apart is, again, funding and history: competitive snipers are distinct entities with their own trading histories and independent funding; a coordinated snipe is fresh wallets sharing a funder. The sniper *percentage* alone is incomplete; you always pair it with the entity question.

#### Worked example: separating competitive snipers from a coordinated snipe

Two launches each show a 15% block-0/1 sniper share at a \$10M cap — **\$1.5M** of supply in both cases. Launch A: the snipe is spread across twelve distinct wallets, each months old, each with its own independent funding from different exchanges, no shared funder. Launch B: the snipe is twelve fresh wallets created the same hour, all funded with \$1,200 each from one address. Same headline number, opposite meaning. In A, you are facing a dozen competing traders who will sell at a dozen different times and prices — annoying overhang, but not a single coordinated dump. In B, you are facing one operator holding \$1.5M who will exit in one move when it suits them. If you must touch either, A is survivable with a tight plan and B is a skip. The intuition — the sniper percentage is a headline; the funding graph is the story, and only the graph tells you whether the early supply is one hand or many.

The second-order point: snipers and bundlers can be the *same* entity (the dev sniping its own launch from "independent" wallets), which is why you do not treat the two checks as separate buckets. You are building one picture of *how much supply is in coordinated, early, motivated-to-sell hands* — and bundling plus sniping are two ways the same hands get there.

## Dev holdings: how much do they hold, and did they sell

The deployer is where the launch came from, so the deployer is where overhang risk lives. Even a launch with a clean-looking holder list can be a trap if the dev quietly controls a large stake through linked wallets and starts feeding it into the market.

![Dev holdings graph showing the deployer linked wallets combined stake and sales](/imgs/blogs/holder-analysis-for-memecoins-4.png)

The figure starts at the deployer, follows funding and transfers to its linked wallets, sums the combined dev stake (about \$400k at this cap across the wallets), flags that about \$100k has already been routed to a CEX deposit address, and lands on the read: a dev selling into you is an overhang you treat as an exit signal.

**How to detect it.** Three moves. First, **find the deployer** — the wallet that created the token contract (the explorer shows it). Second, **find the linked wallets**: wallets the deployer funded, wallets that funded the deployer, and wallets sharing a funder with it. These are the dev's true holdings, not just the single labeled deployer address. Sum the supply across all of them. Third, **check for sales**: look for transfers from those wallets to known exchange deposit addresses, to the bonding-curve/DEX pool (a sell), or to fresh wallets that then sell. A dev that *renounced* the contract and *burned* its tokens is the green case; a dev sitting on supply is overhang; a dev already routing tokens to an exchange is actively dumping.

> [!note]
> "Dev sold" is one of the most reliable bad signals in the whole game. A founder who believes in their own coin does not route their bag to a Binance deposit address in the first hour. On-chain, intent is loud: where the money goes tells you what the dev actually thinks the token is worth.

#### Worked example: a dev holding \$400k who already sold \$100k

You trace the deployer of a token at a \$10M cap and find it is linked — by shared funding — to three other wallets. Combined, they hold 4% of supply, which is **\$400,000** of tokens. Then you check the transfers: one of those wallets has already sent **\$100,000** of tokens to a wallet that immediately deposited them to a centralized exchange. So the dev started with \$400k of paper, has realized \$100k in actual sales, and still holds **\$300,000** of overhang. The reads stack: (1) the dev is a seller, not a holder — they have demonstrated they will dump; (2) there is \$300k of supply still hanging over the price; (3) the "renounced and trustworthy founder" narrative is false. If you were considering a \$1,000 position, the dev's remaining \$300k stake is 300x your size and can erase your trade in one transaction. The intuition — a dev who has already taken \$100k off the table has told you exactly what they plan to do with the other \$300k.

**Finding linked wallets in practice.** The deployer is the seed; from there you build the dev's true footprint by following three kinds of edges. (1) *Outbound funding* — wallets the deployer sent SOL or tokens to before or around launch (it funded its own bundle and its own treasury). (2) *Shared funder* — wallets that, like the deployer, were funded by the same upstream address (often the dev's main wallet or its exchange withdrawal). (3) *Post-launch consolidation* — wallets that, after the launch, send tokens back toward a common address to be sold together. Each edge is a hypothesis, not proof, but when several wallets connect to the deployer by *multiple* edges (it funded them *and* they consolidate back), the linkage is strong. This is the identical heuristic-stacking discipline from general [address clustering](/blog/trading/onchain/address-clustering-and-heuristics), applied to the one wallet you most want to understand.

**The migration and "fresh dev" trick.** Two refinements worth flagging because they defeat lazy checks. First, a dev who knows people check the deployer will sometimes *deploy from a throwaway wallet* and hold the real bag elsewhere — so a clean-looking deployer is not the end of the trace; you still have to ask where the bundle's funding came from, because that often leads to the dev's actual wallet. Second, watch for *re-launch patterns*: the same operator launching token after token from related wallets, rugging each one, then starting over. If the funder behind this launch has funded previous tokens that went to zero, that history is on-chain and it is the single most damning thing you can find — a serial rugger's funding wallet is a fingerprint. Checking the funder's *prior tokens* is an underused, high-signal step.

#### Worked example: a serial rugger's funding fingerprint

You trace the bundle funder for a new launch and, instead of stopping at "one funder, \$30,000, twenty wallets", you look at what *else* that funder has done. Its transaction history shows it funded the bundles for three previous tokens over the past two weeks, each of which spiked and went to zero within an hour, with the funder pulling roughly **\$40,000**, **\$55,000**, and **\$25,000** of profit out to an exchange across the three. That is **\$120,000** extracted from three prior victims by the same hand now launching the token in front of you. No holder-structure snapshot of *this* token is needed to make the call — the operator's track record is the verdict. The intuition — funding wallets persist across launches, so a rugger's history is visible, and the cheapest diligence is sometimes just asking what the funder did last week.

## Fresh-wallet clusters: a crowd, or one entity wearing a crowd's clothes

Bundling and dev overhang are both, underneath, the same phenomenon: *coordinated wallets that look independent*. The general skill is recognizing a fresh-wallet cluster — a set of new, single-purpose wallets that are really one operator. This is where address-clustering heuristics meet memecoins.

![Fresh wallet cluster grid of tells that mark coordinated wallets](/imgs/blogs/holder-analysis-for-memecoins-8.png)

The grid lays out the tells: wallet age (created minutes before launch, zero history), funding source (all topped up from one address in the same hour), single purpose (holds only this token), round amounts (identical buy sizes), lockstep timing (buy in the same block, sell within the same minute), and no independent gas history. No single trait is damning — but a holder list where dozens of wallets share most of them is one entity, not a crowd.

**How to detect it.** You stack weak signals into a strong conclusion, exactly as in general [address clustering](/blog/trading/onchain/address-clustering-and-heuristics). Pull the top holders. For each, ask: How old is the wallet? Where did its first funds come from? Does it hold anything else, or only this token? Are the buy amounts suspiciously identical and round? Did it act in lockstep with others (same block in, same minute out)? Any one of these is suggestive. Three or four pointing the same way, across a dozen wallets, is a high-confidence cluster — and you then collapse those dozen "holders" into the *one* entity they really are before you read concentration.

This is the heart of the memecoin forensic read, and it connects directly to the broader skill of [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration): concentration only means anything once you have de-duplicated the holder list into entities. A token that looks like it has a flat, healthy distribution across 30 wallets can actually have 60% of its float in one cluster's hands.

**Why fresh wallets are the universal disguise — and why they fail.** Wallets are free and instant to create, so the cheapest way to fake a crowd is to make a hundred wallets. That is why fresh-wallet clusters show up everywhere a bad actor needs to look like many people: memecoin bundles, airdrop-farming cohorts, wash-trading rings, vote-buying in governance. The disguise is powerful at the surface and fatally weak underneath, for one reason: *a wallet can be created for free, but it cannot create a* history *for free.* A real holder leaves months of unrelated transactions, funding from a real exchange, gas paid from independent activity, holdings in other tokens. A fresh cluster wallet has none of that — and the moment you check funding and age, the disguise collapses, because the operator cannot fabricate the one thing that takes time and independent activity to accumulate. That asymmetry — cheap to fake the *count*, impossible to fake the *history* — is why holder analysis works at all. The whole pseudonymous-not-anonymous principle of the chain lives in that gap.

The practical consequence: when you scan a holder list, you are not really counting wallets, you are counting *histories*. Two hundred wallets with two hundred independent histories is a crowd; two hundred wallets with one shared history is one operator. Train your eye to see the history behind the address, and a "distributed" rug stops being able to hide from you.

#### Worked example: reclassifying the top holders

A token at a \$10M cap shows a top-30 holder list that, at a glance, looks reasonable — no single wallet over 4%. But you trace funding and timing. Eighteen of those 30 wallets were created within the same ten-minute window before launch, were each funded with exactly \$1,200 of SOL from one address, and all bought in the launch slot. You collapse those 18 into one entity. Their combined holding is 33% of supply — **\$3.3M** at this cap — controlled by a single operator whose total cost was about 18 × \$1,200 = **\$21,600**. The "flat distribution" was a costume. After reclassification, the real top holder owns a third of the supply, against a free float that real buyers are fighting over. The intuition — concentration is only meaningful after you reduce addresses to entities, and a fresh-wallet cluster can hide a controlling stake behind a wall of small, innocent-looking wallets.

## Reading a Bubblemaps cluster map

Doing the funding-and-timing trace by hand for every holder is slow. **Bubblemaps** is the tool that visualizes it. It draws each top holder as a bubble sized by balance, and draws a line between two bubbles whenever they have transferred funds to each other. The result is a picture: independent holders sit alone, and coordinated wallets pull into a tight web of connected bubbles. A dense cluster of linked bubbles is, almost always, one entity.

![Bubblemaps style cluster map where connected bubbles form one entity](/imgs/blogs/holder-analysis-for-memecoins-5.png)

In the figure, holders #2, #4, and #7 are joined by transfer lines and a shared funder — they collapse into one cluster holding about 25% combined. Holders #11 and #15 sit isolated, with no lines and an aged or exchange-sourced history; those are the likely-real holders. Your read: count the connected cluster as ONE holder, not three.

**How to read it, step by step.** Open the token in Bubblemaps. (1) Look for *clusters* — groups of bubbles tied together by lines. A clean, organic token is mostly isolated bubbles; a rigged one has one or more dense webs. (2) Note the *combined size* of the largest cluster — three bubbles at 8% each is a 24% entity, not three small holders. (3) Check whether the cluster includes the *contract/dev* bubble — that tells you the cluster is the team. (4) Watch for *post-launch lines*, where bubbles send to each other after trading starts (consolidating to sell, or shuffling to obscure). (5) Cross-check the cluster's wallets against the bonding curve and exchange deposit addresses to see if it is exiting. Bubblemaps does not *prove* common ownership — two wallets can transfer for innocent reasons — but combined with same-funder and same-block evidence, a tight cluster is as close to proof as memecoin forensics gets.

> [!tip]
> Treat Bubblemaps as a fast first filter, not a verdict. A scary cluster earns a deeper funding trace; a clean map earns the *next* check (snipers, dev sales), not blind trust. No single tool closes the case.

**Where the cluster map misleads.** Two failure modes are worth naming so you do not over-trust a pretty graph. First, *false links*: two wallets can show a transfer line for an innocent reason — one user moving funds between their own wallets, or a holder tipping a friend. A single line is weak evidence; a *web* of mutual links plus shared funding plus same-block buys is strong. Do not condemn a token because two bubbles touched once. Second, *missing links*: Bubblemaps draws lines for *direct* transfers between top holders, but a sophisticated cluster funds its wallets through intermediaries that are *not* on the top-holder list, so the bubbles look isolated even though they share a hidden upstream funder. The map shows you the obvious clusters; it can miss the carefully-routed ones. That is why the map is a first filter and the funding trace is the confirmation — the tool surfaces the candidates, and you do the two-hop funding check to promote a candidate to a verdict.

The connection to the rest of your toolkit is direct: this is the same fan-in/fan-out reading you would use in [early-buyer and insider detection](/blog/trading/onchain/early-buyer-and-insider-detection) and in [rug-pull and honeypot detection](/blog/trading/onchain/rug-pull-and-honeypot-detection) — a cluster of wallets funded by one source, accumulating before everyone else, is the universal shape of a stacked deck, whether the deck is a memecoin, a presale, or an airdrop farm. The same skill that catches an airdrop-farming sybil cohort catches a memecoin bundle, because both are one operator manufacturing the appearance of many independent participants.

## The base rate: you are betting against survivorship

Now the part that no amount of clever holder analysis can rescue you from, and the reason this whole discipline is framed as risk reduction rather than alpha. The base rate for memecoins is catastrophic.

![Pump.fun survivorship bar chart of launches that reach a meaningful cap](/imgs/blogs/holder-analysis-for-memecoins-6.png)

Roughly 8 million tokens have been launched on pump.fun, and only about 1.4% of them ever reach a meaningful market cap. The other ~98.6% effectively go to zero — many within minutes. That is the universe you are picking from. Holder analysis improves your *conditional* odds (a clean structure is genuinely better than a rigged one), but it cannot change the fact that you are fishing in a pond that is overwhelmingly stocked with worthless and engineered-to-fail tokens.

This is *survivorship bias* made concrete. Every screenshot of a memecoin that did 1000x is one survivor pulled from millions of corpses. The trader who shows you their winner is not showing you the forty losers around it. When you read "smart money is in this coin", remember that "smart money" is itself a label assigned *after* the fact to wallets that happened to win — the same survivorship trap covered in [the perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain). The chain shows you flow; it does not show you the future.

#### Worked example: sizing a \$500 bet against the base rate

Suppose you have done everything right: a clean funding trace, no bundle, snipers under 5%, dev renounced and burned, a broad isolated-bubble Bubblemaps map. You have a launch that is in the *better* slice of the distribution. What is a rational bet? Start from the base rate: even conditional on a clean structure, most launches still fail, so treat the position as money you expect to lose. If you allocate **\$500** and the realistic outcome distribution is something like a 90% chance of going to roughly zero and a 10% chance of a 5x to \$2,500, your expected value is about 0.9 × \$0 + 0.1 × \$2,500 = **\$250** — you are paying \$500 for an expected \$250, a deeply negative-EV bet even after good diligence. The honest conclusion is that holder analysis lets you *avoid the worst* launches and size the survivors as entertainment, not as an edge. The intuition — clean structure changes a guaranteed scam into a long-shot lottery ticket; it does not change a lottery ticket into an investment.

This is the brutal honesty the topic demands: **most memecoins are -EV, and holder analysis is a way to lose less, not a way to win.** Anyone selling you a "memecoin alpha system" is selling you survivorship. The genuine value of everything in this post is *defensive* — it turns "I aped because the chart was green" into "I skipped because 35% of supply was one funder's bundle." That skip is the whole game.

## How holder structure evolves after launch

The checks above are a snapshot at the open, but holder structure is not static — it moves, and the *direction* it moves carries as much signal as the starting picture. Reading the evolution is the difference between a one-time screen and an ongoing risk read on a position you already hold.

The healthy direction is *deconcentration*: over the hours and days after launch, the controlled supply bleeds into a wider base. The bundle and snipers sell (taking their profit), and the buyers are many small independent wallets rather than a new cluster. Concentration falls, the holder list genuinely widens, and the free float grows relative to controlled supply. That is what a token finding real demand looks like on-chain — the early operators exit and a crowd absorbs them. It is rare, which is exactly why it is worth recognizing when it happens.

The unhealthy directions are two. *Re-concentration*: the cluster sells to a few new large wallets that turn out to share a funder — the bag is changing hands among insiders, not distributing. Or *quiet accumulation by the dev*: the deployer's linked wallets buying back dips to defend the price, which looks bullish but means the float is getting *more* controlled, not less, setting up a bigger eventual dump. The way you tell distribution from rotation is, once more, the entity question applied to the *new* large holders: are the wallets absorbing the sells independent and aged, or are they fresh and linked? On-chain, "the coin is distributing to a real crowd" and "the bag is rotating among insiders" look identical on the price chart and opposite on the holder graph.

#### Worked example: distribution versus rotation, on the same dip

A token you are watching dipped 30% and the chart looks like a healthy shakeout. You check who bought the dip. Scenario A: the dip was bought by roughly 400 distinct wallets, mostly aged, funded from many different sources, none taking more than 0.5% of supply — that is **\$50,000** of buying spread across a real crowd absorbing the early sellers. Concentration fell; this is genuine distribution. Scenario B: the same 30% dip was bought by four fresh wallets sharing one funder, together taking 12% of supply — about **\$1.2M** consolidated into one new entity. The price recovered identically in both, but A is a token finding a base and B is a bag rotating into one operator's hands for a larger exit later. The intuition — the price chart cannot distinguish a crowd from a cartel buying the dip, but the funding graph of the dip-buyers can, and that distinction is the whole risk read.

## How to read it: a walkthrough on a real-feeling launch

Let us run the full check end-to-end on an illustrative launch — a token we will call SNIPED, at a \$10M fully-diluted cap, the kind of thing that scrolls past on a Solana scanner every few seconds. (Placeholder ticker and addresses; the numbers and procedure are what you would do on a live one.)

**Step 1 — Open the holder list.** On Solscan, pull the token's top 50 holders. You see ~3,800 total holders and a top-25 that, at first glance, looks spread out — biggest wallet at 3.9%. Promising on the surface. We do not trust the surface.

**Step 2 — Bundle check.** Run a bundle checker (or trace by hand): of the top 25, how many were funded by one source and bought in the launch slot? You find 20 wallets each funded with about \$1,500 of SOL from one address (\$30,000 total) an hour before launch, all buying in the deploy slot. That is a bundle. Collapse them: one entity holds 35% of supply — **\$3.5M** at this cap. Red flag, severe.

**Step 3 — Sniper check.** Sum block-0/1 buys. Another 15% of supply — **\$1.5M** — was sniped in the first two slots, partly by wallets that overlap with the bundle. So coordinated, motivated-to-sell hands hold roughly half the supply before any honest buyer arrived.

**Step 4 — Dev check.** Find the deployer; trace its linked wallets. The deployer is one of the bundle funders (no surprise). Its linked wallets hold 4% — **\$400,000** — and one has already routed **\$100,000** to a CEX deposit address. Dev is selling. Red flag.

**Step 5 — Bubblemaps.** Open the cluster map. The top of the list is one dense web of connected bubbles — the bundle — with a few isolated bubbles (real buyers) scattered at the edges. The picture confirms the trace: this is not a crowd, it is one operator plus victims.

**Step 6 — Reclassify and decide.** Reduce the holder list to entities. The "3,800 holders" become: one controlling entity (~35% bundle, overlapping ~15% snipers, 4% dev) and a long tail of small real buyers fighting over a thin free float. The apparent distribution does not survive reclassification. The decision is immediate: **skip.** Not "size small" — skip. When one entity holds a third of supply at a 100x markup and the dev is already selling, there is no price at which you are not the exit.

Now compare to a launch that *passes*: the same six steps, but the funding trace is clean (no common funder across the top holders), snipers are under 5%, the dev renounced the contract and burned its allocation, and Bubblemaps shows mostly isolated bubbles with aged histories. That launch earns the *right* to a tiny, base-rate-aware bet — the \$500-as-entertainment position from earlier, not a conviction trade. The walkthrough's value is the same either way: it replaces a feeling with a structured read.

![Memecoin red-flag matrix mapping each signal to a read severity and action](/imgs/blogs/holder-analysis-for-memecoins-7.png)

The matrix is the whole check on one screen: each row is a holder-structure signal (bundling, snipers, dev stake, fresh-wallet cluster, distribution), and you read left to right — what you see, why it matters, severity, and the action. Bundling and snipers are skip-level; dev stake and fresh-wallet clusters are size-down-and-watch; a genuinely clean distribution is the only row that earns a (still -EV-sized) bet.

## Common misconceptions

**"A high holder count means it is safe."** No. Holder count is the easiest number to fake — a bundle of 20 wallets and a few thousand victims is "5,000 holders" with a third of the supply in one hand. Always reclassify addresses into entities; the holder count after de-duplication is the only one that matters. A token with 200 *genuinely independent* aged holders is far healthier than one with 5,000 addresses dominated by a cluster.

**"Pump.fun's bonding curve means it can't be rugged."** Pump.fun's design makes one *specific* rug — pulling the liquidity pool — harder, because the curve holds liquidity until graduation. It does *nothing* about the rugs that actually dominate: bundling, sniping, dev overhang, and the simple dump where insiders sell their 35% into your buy. "Can't pull liquidity" is a long way from "safe."

**"The dev renounced the contract, so we're good."** Renouncing means the dev can no longer change the contract (mint more, blacklist sellers) — genuinely good and worth checking. But renouncing says *nothing* about how much supply the dev still *holds*. A dev can renounce and still control 30% of the float through linked wallets and dump it tomorrow. Renounce-check and holdings-check are separate questions; pass both, not one.

**"Snipers in a coin are bullish — smart money got in early."** Snipers being present tells you the cheapest, fastest, most-motivated-to-sell money is in front of you. Their presence is not a buy signal; it is a warning about *who you will be selling to and at what price*. The only "smart" thing about a sniper is its latency, and you are not racing it — you are deciding whether to be its exit liquidity. Tie this to [detecting fake volume versus organic demand](/blog/trading/onchain/detecting-fake-volume-vs-organic-demand): early activity that is bots, not humans, is noise dressed as demand.

**"If it's on pump.fun, the holder data is too noisy to trust."** The opposite is true. Pump.fun launches are *more* legible than most tokens, because the bonding curve forces all early trading through one contract you can read precisely — every buy, its slot, its size, and its buyer are right there. The noise is in the *count* of launches (millions, so any single one is a needle in a haystack), not in the *data quality* of a given launch. Once you have picked a token to examine, the on-chain record is clean, complete, and free. The discipline is not "can I trust the data" but "do I bother to look before I ape" — and most people do not, which is exactly where the edge of *avoidance* comes from.

**"A token that already pumped 10x must have organic demand by now."** A 10x can be entirely manufactured: the bundle and snipers buying and re-buying to walk the price up, paid influencers manufacturing the appearance of a crowd, and a thin float that a little real money lifts a long way. The pump is *evidence the trap is working*, not evidence the trap is gone. The holder check on a token that already ran is the same as on a fresh launch — has the controlled supply actually distributed to a real crowd, or is it still concentrated and now sitting on a much larger paper gain? A 10x with concentration intact is a *more* dangerous setup, not a safer one, because the insiders' incentive to sell is now ten times larger.

**"Good holder analysis gives me an edge."** It does not. It reduces your *downside* by helping you skip the rigged launches, and it improves your *conditional* odds on the ones you do touch. But the base rate — ~1.4% survival — is so punishing that even a perfect read leaves memecoin betting negative-EV. Holder analysis is defense. Treat any "edge" framing as the survivorship sales pitch it usually is.

## The playbook: what to do with it

Run this on every launch, in order, before you risk a dollar. Each line is *signal → read → action → what would falsify it*.

- **Bundle present (one funder, many launch-block buyers, large combined share)** → the distribution is fake and that supply is set to dump → **skip**, or if the share is small and everything else is clean, size to money you are fully prepared to lose → *falsifier:* the "cluster" wallets have independent funding sources and aged histories (then it is not a bundle).

- **Sniper share high (double-digit % bought in block 0-1)** → you would be buying inventory the fastest money already holds at a lower basis → **do not buy the open**; if you must, wait for the snipers to visibly exit and re-evaluate → *falsifier:* block-0/1 share is a few percent and those wallets are not selling.

- **Dev stake large and/or dev selling (deployer + linked wallets hold meaningful supply; transfers to CEX/pool)** → overhang and demonstrated intent to sell → **avoid**; exit immediately if you are in and the dev starts routing to an exchange → *falsifier:* dev renounced *and* burned its allocation (verify both on-chain).

- **Fresh-wallet cluster among top holders (dozens of new single-purpose wallets, shared funder, lockstep timing)** → those are one entity; the real concentration is higher than the holder list suggests → **reclassify to entities, then re-read concentration**; treat the cluster as one holder → *falsifier:* the wallets have varied ages, funders, and unrelated histories.

- **Distribution genuinely broad (many independent, aged holders; no dominant entity after reclassification; clean Bubblemaps)** → necessary but *not* sufficient → **a tiny, base-rate-aware bet is permitted**; size it as entertainment, not conviction, and respect the ~1.4% survival rate → *falsifier:* the breadth disappears once you collapse a hidden cluster.

- **Always, regardless of the above** → the base rate is brutal → **never size a memecoin like an investment**; cap total memecoin exposure to money you can lose entirely, and remember that the skip is usually the winning move. The discipline that makes you money here is the diligence that keeps you *out* of the engineered-to-fail launches, which is most of them.

Fold this into your broader [on-chain due-diligence checklist](/blog/trading/onchain/onchain-due-diligence-checklist): for a memecoin, the holder-structure read *is* the due diligence, because there is nothing else to diligence. No revenue, no team to evaluate, no product — only the ledger telling you whether the launch is a crowd or a trap. Learn to read it, and you will skip the traps that take most people's money. That is not alpha. It is survival, and in this corner of the market survival is the whole edge.

## Further reading & cross-links

- [Supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — the general framework for reading who owns a token; memecoin holder analysis is this with the dial turned to maximum.
- [Rug-pull and honeypot detection](/blog/trading/onchain/rug-pull-and-honeypot-detection) — the contract-side checks that pair with the holder-side checks here.
- [Early-buyer and insider detection](/blog/trading/onchain/early-buyer-and-insider-detection) — the funding-source-and-timing trace that underpins bundle and cluster detection.
- [Detecting fake volume versus organic demand](/blog/trading/onchain/detecting-fake-volume-vs-organic-demand) — why the volume on a launch can be bots, not buyers.
- [Address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics) — the general skill of collapsing many addresses into one entity, which is the core move in this post.
- [The on-chain due-diligence checklist](/blog/trading/onchain/onchain-due-diligence-checklist) — where the memecoin read slots into a full pre-trade workflow.
