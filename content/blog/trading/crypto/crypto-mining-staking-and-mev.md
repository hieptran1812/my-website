---
title: "Mining, Staking, and MEV: The Hidden Market Microstructure of Crypto"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A from-zero tour of how crypto blocks actually get built — proof-of-work mining, proof-of-stake staking, and the invisible market in transaction ordering called MEV that quietly taxes every on-chain trade."
tags: ["crypto", "mev", "mining", "staking", "proof-of-work", "proof-of-stake", "ethereum", "flashbots", "front-running", "block-building", "defi", "validators"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Whoever decides the order of transactions inside a block can extract value from everyone else's trades, a hidden tax called MEV; behind crypto's clean front end runs a fierce, mostly invisible market in block space where miners, validators, searchers, and builders compete for the right to reorder reality.
>
> - A blockchain advances by appending **blocks** — bundles of transactions — and whoever produces the next block chooses which transactions go in and **in what order**. That ordering power is worth money.
> - **Mining** (proof-of-work) and **staking** (proof-of-stake) are two ways to win the right to produce a block: one burns electricity, the other locks up capital. Both pay the producer a reward.
> - **MEV** (maximal extractable value) is the profit a block producer or its partners can squeeze out by inserting, reordering, or censoring transactions — arbitraging price gaps, seizing collateral in **liquidations**, or **sandwiching** an ordinary user's trade.
> - A whole supply chain grew up to harvest it: **searchers** find opportunities, **builders** assemble the most profitable block, and **proposers** (validators) sell the right to publish it — most famously through **Flashbots** and **MEV-Boost**.
> - Estimates put extracted MEV in the low billions of dollars cumulatively (roughly \$1.5 billion-plus on Ethereum since 2020, as-of 2026); the deeper worry is **centralization** — a handful of builders now construct the large majority of Ethereum blocks.

The diagram above is the mental model for everything that follows: you sign a swap in your wallet, it lands in a public waiting room, and before it settles, a stranger you will never meet decides exactly where in the next block your trade sits — and profits from that decision. The "exchange" you think you are trading on has no say in the order your orders execute. A block builder does. That single fact — that transaction *ordering* is a scarce, sellable resource — is the hidden microstructure of crypto, and almost nobody using a wallet ever sees it.

![A transaction path from wallet through mempool and builder to a block](/imgs/blogs/crypto-mining-staking-and-mev-1.png)

Most explanations of crypto stop at the front end: you connect a wallet, you click swap, tokens appear. That is the user interface. Underneath it is a market as adversarial as any high-frequency trading pit, except the "exchange floor" is the few hundred milliseconds between when your transaction becomes public and when it gets sealed into a block. In that window, automated bots race to reorder the world's pending transactions for profit. The value they extract comes, ultimately, out of the pockets of ordinary users — it is a tax you pay without an itemized receipt. This post builds the whole machine from zero: what a block is, how mining and staking decide who builds it, where this hidden value comes from, who captures it, and why the people who study Ethereum most closely consider it one of the most important unsolved problems in the system. No prior crypto or finance knowledge is assumed; every term is defined the first time it matters.

## Foundations: how a block actually gets built

Before MEV makes any sense, we need a shared, concrete vocabulary for how a blockchain moves forward. None of this requires prior background. I will define each term the first time it shows up and keep every definition physical.

**A blockchain** is a shared public ledger — a list of who owns what and who sent what to whom — that thousands of independent computers each keep a copy of and agree on. It advances in discrete steps. Every few seconds (about 12 seconds on Ethereum, about 10 minutes on Bitcoin) the network appends a new **block**: a bundle of recent transactions plus a cryptographic link to the previous block. The chain is just blocks linked back to back, which is where the name comes from.

**A transaction** is a signed instruction the network executes — "send 1 ETH from my address to that address," or "swap \$10,000 of USDC for ETH in this pool." ETH is the native currency of Ethereum; USDC is a **stablecoin**, a token designed to stay worth about one US dollar. You create a transaction in your **wallet** (the software plus the secret cryptographic key that proves the funds are yours), sign it with your key, and broadcast it to the network.

**The mempool** (short for "memory pool") is the public waiting room for transactions that have been broadcast but not yet included in a block. It is a holding pen: your signed transaction sits there, visible to everyone running a node, until some block producer scoops it up. The word to underline is *public* — by default, every pending transaction is broadcast to the whole world before it executes. That visibility is the soil in which MEV grows, because anyone watching the pen can see what you are about to do and react before you do it.

**A block producer** is whoever gets to build the next block. They choose which pending transactions to include, which to ignore, and — crucially — **what order to put them in**. Order matters enormously, because transactions execute one after another and earlier ones change the state the later ones see. If two people try to buy the last cheap token, whoever's transaction the producer puts first gets the deal. The producer's job is mechanical, but the *power to order* is the entire subject of this article.

**Gas** is the fee you pay to get a transaction processed. Every computation a transaction performs costs **gas**, measured in tiny units of ETH; you attach a fee, and block producers prefer transactions that pay more. This creates a continuous auction: when blocks are full and demand is high, fees spike, because the limited space in each block goes to the highest bidders. The portion of the fee you set as an incentive to be included quickly is the **priority fee** or **tip**.

Now the two ways the network picks a block producer.

**Proof-of-work (PoW) mining** is the original method, used by Bitcoin. To win the right to produce the next block, computers race to solve a meaningless but hard math puzzle — repeatedly hashing data until they find a number below a target. The only way to find it is brute trial and error, so the chance of winning is proportional to how much computing power, called **hashrate**, you throw at it. The winner is the **miner**; they publish the block and collect a reward. The whole point of burning all that electricity is that rewriting history would require redoing all that work, which is prohibitively expensive — security bought with energy.

**Proof-of-stake (PoS) staking** is the method Ethereum switched to in 2022. Instead of burning electricity, you lock up capital. To become a **validator** you deposit (stake) 32 ETH into a contract. The network then pseudo-randomly selects validators to propose and attest to blocks; your chance of being chosen scales with how much you have staked. The validator who is selected to publish a block is called the **proposer**. Security here is bought with capital at risk rather than energy: misbehave, and the network destroys part of your stake.

**Slashing** is that punishment. If a validator does something provably malicious — signing two conflicting blocks, for instance — the protocol confiscates a chunk of their staked ETH and ejects them. Slashing is what makes staking "skin in the game": the validator has real money on the line that the network can burn.

**The block reward** is what the producer earns: in Bitcoin, newly minted coins (the **subsidy**) plus transaction fees; in Ethereum, newly issued ETH plus the priority fees in the block plus — and here is the twist this whole post is about — any **MEV** they can capture.

It helps to trace the life of one block end to end before adding the economic layer. The producer takes a snapshot of the current ledger state, selects a set of pending transactions from the mempool, decides their order, and executes them one by one against that state — updating balances, running smart-contract code, charging gas. The producer then computes a cryptographic summary (a hash) of the resulting state and the transaction list, packages it with a pointer to the previous block, and broadcasts the finished block. Other nodes re-execute every transaction to check the producer did not cheat; if the result matches, they accept the block and build on top of it. The chain is thus a sequence of agreed-upon state transitions, each one authored by whoever won the right to produce that block. The authorship is the power. Everything a transaction *does* depends on the state it sees, and the state it sees depends on which transactions the producer ran before it — so by choosing the order, the producer chooses outcomes.

**Finality** is the related idea that, after enough subsequent blocks pile on top (or, in proof-of-stake, after enough validators attest), a block becomes practically impossible to reverse. Before finality there is a small window in which the ordering — and therefore who captured what value — could in principle still change if the chain reorganizes ("a reorg"). MEV that relies on deliberately reorganizing already-published blocks to steal value from a past block is the most dangerous variety, called **time-bandit** or reorg MEV, because it threatens the stability of the chain itself rather than merely taxing a user.

That MEV term is the one everything turns on.

**MEV — maximal extractable value** (originally "miner extractable value," renamed when Ethereum left mining) — is the maximum profit a block producer can extract *beyond* the standard block reward and fees, purely by choosing which transactions to include and how to order them. If the producer can insert their own transaction before yours, after yours, or instead of yours to make money, that profit is MEV. It is value that exists *because* one party controls ordering and everyone else's transactions are public.

A few more roles complete the cast, because in practice the proposer no longer does the profitable ordering themselves.

**A searcher** is a person or bot that scans the mempool for MEV opportunities — a price gap, a liquidatable loan, a juicy swap to sandwich — and packages the transactions that capture it into a **bundle**: an ordered list of transactions that must execute together, in that exact order, or not at all.

**A block builder** is a specialist that collects bundles from many searchers plus ordinary transactions and assembles the single most profitable block it can, then bids for the right to have that block published.

**Proposer-builder separation (PBS)** is the architecture that splits these jobs: the **builder** decides the *contents and order* of the block (the profitable, complex part), while the **proposer** (validator) merely picks the most valuable finished block offered to them and signs it. The proposer does not even need to see the block's contents — they choose based on the bid. We will see exactly how this works through a system called **MEV-Boost**.

That is the entire vocabulary. With it, the strange economy on top becomes legible.

## The invisible market: why ordering is worth money

Here is the core idea, stated plainly: **the person who orders the transactions in a block can extract value from everyone else's trades.** Crypto's transparency — every pending transaction visible in the public mempool — combined with the producer's absolute control over ordering, creates profit opportunities that simply do not exist in a traditional market where your order book actions are private until matched.

There are three big families of MEV, and they are worth separating because they differ sharply in who gets hurt.

![A tree of MEV types arbitrage liquidation and sandwich](/imgs/blogs/crypto-mining-staking-and-mev-7.png)

**Arbitrage** is the largest and least harmful category. Two decentralized exchanges (DEXes) — say Uniswap and Sushiswap — can momentarily list ETH at slightly different prices. A searcher buys on the cheaper one and sells on the dearer one in the same block, pocketing the difference. This is value created by *correcting* a price discrepancy; it makes prices across venues consistent, which is genuinely useful market plumbing. The "victim," if there is one, is the liquidity provider who left a stale price, and the arbitrage is what nudges the price back to correct. Most measured MEV is arbitrage.

**Liquidations** are the second family. In a lending protocol like Aave or MakerDAO, you borrow against collateral; if your collateral's value falls too far, your loan becomes undercollateralized and anyone is allowed to repay part of it and seize your collateral at a discount as a reward. Searchers compete to be the one who triggers a profitable **liquidation** the instant a position crosses the threshold. This too is a designed feature — liquidations protect the protocol from bad debt — but the race to capture the liquidation bonus is pure MEV, and being the bot that wins it depends on transaction ordering.

**Sandwiching** is the third family, and the only one that directly preys on an ordinary user. A searcher spots your large swap sitting in the mempool, places a buy *just before* it (pushing the price up), lets your trade execute at the now-worse price, and sells *just after* (pocketing the inflated price). You wanted to swap; the bot made you pay more for the same tokens and kept the difference. This is **front-running** — acting on knowledge of your pending transaction before it executes — turned into a two-sided trap. It is the dark heart of MEV, and the example we will work through in detail.

The reason all three are possible is the same: your intentions are visible (mempool) and someone else controls the sequencing (the producer). Remove either condition and the opportunity vanishes. That is why "private" transaction channels and fairer ordering schemes are such active areas of research — they attack the two preconditions directly.

### How the value flows up the chain

The value extracted does not stay with whoever first spotted it. It gets split up a chain of specialists, each taking a cut for the part of the job they do.

![A stack showing extracted value split from user to searcher builder and proposer](/imgs/blogs/crypto-mining-staking-and-mev-5.png)

At the bottom is the **user**, who unknowingly supplies the value — the slippage on their sandwiched trade, the stale price an arbitrageur corrects, the over-collateral a borrower forfeits. Above them, the **searcher** captures the gross opportunity but must pay a tip to get their bundle included. Above the searcher, the **builder** takes a cut for assembling the winning block. At the top, the **proposer** (the validator) earns the bid that the builder paid for the right to publish. Each layer competes fiercely with peers at the same layer, which drives most of the value up to the proposer — in a competitive auction, searchers bid away almost all their profit to win inclusion, and builders bid away almost all of theirs to win the block. The economics push the surplus toward whoever holds the scarcest resource: the proposer who owns the block space.

This is the same logic that governs any competitive market for a scarce input. The opportunity is the prize; the searchers, builders, and proposers are bidders at successive auctions; and in a perfectly competitive auction the prize accrues to whoever controls the bottleneck, with everyone else competed down to roughly their cost of doing business. The bottleneck here is the right to *order* the block — the proposer's slot. That is why, despite all the sophistication concentrated in searchers and builders, the long-run economic rent of MEV gravitates toward the validators who hold block space, and why MEV shows up as a component of *staking yield*. It also explains the strategic prize of exclusive order flow: a builder that can keep some opportunities away from rival searchers is, in effect, partly removing the competition at the layer below it, so it does not have to bid the full value back up — letting it keep a larger slice for itself. The fight over who captures MEV is, at every layer, a fight over how competitive each auction is allowed to be.

## The MEV supply chain in detail

Let us follow a single opportunity through the machine, because the structure is the whole story.

![A branching graph of the MEV supply chain from searchers to builder to proposer](/imgs/blogs/crypto-mining-staking-and-mev-2.png)

It starts with an **opportunity** appearing on-chain — a large swap hits the mempool, or a loan crosses its liquidation threshold. Multiple **searchers** see it at once, because the mempool is public. Each builds a bundle that captures the opportunity and attaches a tip — effectively a bid — for how much they will pay to have their bundle land in the next block. Searchers are in a sealed-bid race against each other; the one willing to give up the most profit usually wins.

The bundles flow to **builders**. A builder's job is a combinatorial optimization: out of thousands of pending transactions and competing bundles, assemble the single ordered block that maximizes total value (fees plus tips plus the builder's own extracted MEV). The builder that constructs the most valuable block can afford to bid the most for the right to publish it.

Builders submit their finished blocks, along with a bid, to a **relay**. Under **MEV-Boost** — the dominant implementation of proposer-builder separation on Ethereum — the relay runs a blind auction: it holds the full block contents but shows the proposer only the bid amount and a commitment, so the proposer cannot peek at the block, copy the profitable bundles, and steal them. The proposer simply signs the header of the highest-bidding block. The relay then reveals the full block to the network. The proposer never had to find a single MEV opportunity; they just sold their block space to the highest bidder.

This is **proposer-builder separation** made concrete. It exists for a good reason: building maximally profitable blocks is a sophisticated, capital-intensive specialty, and forcing every small validator to do it would push staking toward only the most sophisticated operators. By letting validators outsource block construction to a competitive builder market, PBS lets a hobbyist with 32 ETH earn nearly the same MEV-inclusive reward as a giant — they just auction the work. The catch, which we will return to, is that it concentrates enormous power in a tiny number of builders.

Flashbots, the research-and-software organization that built the first MEV auction in 2020 and later MEV-Boost, did this deliberately. Before Flashbots, searchers competed by spamming the network with high-fee transactions and bidding gas prices into the stratosphere, congesting the chain for everyone in what were called "gas wars" or "priority gas auctions." Flashbots moved that competition into a private, off-chain auction, which reduced the spam and failed transactions — but also formalized MEV extraction into an industrial pipeline.

It is worth being precise about why a *bundle* is the right unit of trade. A searcher's strategy only works if its transactions execute in a specific order with nothing slipped in between — a sandwich is worthless if a third party's trade lands between the front-run and the victim, and an arbitrage is worthless if a rival's identical trade lands first. A bundle is an all-or-nothing, ordered package: the builder must include every transaction in it, in the stated sequence, with no foreign transactions interleaved, or include none of them. This atomicity is exactly the guarantee the public mempool cannot give, which is why searchers route bundles to builders privately rather than broadcasting them. The bundle also lets a searcher bid *conditionally* — "pay the builder X only if this bundle actually captures the opportunity" — so searchers never pay for failed attempts the way they did in the old gas-war days, when a losing bid still burned real gas on a reverted transaction.

The second subtlety is **order flow**. The most valuable input a builder can have is exclusive access to transactions nobody else can see — for example, swaps routed to it privately by a wallet's MEV-protection feature, or trades from an exchange. A builder with exclusive order flow can extract MEV from those transactions (or protect them, and charge for it) that rivals never get a chance at. This is why the competition among builders increasingly turns on *deals for order flow* rather than purely on clever algorithms, and it is the engine of the centralization problem we will reach: whoever accumulates the most exclusive flow builds the most valuable blocks, wins the most auctions, and attracts still more flow.

## Mining economics: what a miner actually earns and spends

To understand staking and MEV, it helps to first understand the older economic engine they replaced: proof-of-work mining. A miner's business is brutally simple. They spend money on specialized machines and electricity; they earn block rewards and fees. Whether mining is profitable is just revenue minus cost, and both sides move constantly.

![A matrix comparing proof-of-work mining and proof-of-stake staking economics](/imgs/blogs/crypto-mining-staking-and-mev-4.png)

Three terms govern the revenue side. **Hashrate** is how many guesses per second a miner's machines make at the puzzle; more hashrate means more chances to win a block. **Difficulty** is a number the network automatically adjusts so that, no matter how much total hashrate joins or leaves, blocks keep arriving at the target interval (about 10 minutes for Bitcoin). If everyone doubles their hashrate, difficulty doubles too, and each miner's *share* of blocks stays the same — you are racing the entire field, not the clock. **The halving** is a scheduled event in Bitcoin where the block subsidy is cut in half roughly every four years; it started at 50 BTC per block in 2009 and, after halvings in 2012, 2016, 2020, and 2024, sits at 3.125 BTC per block as-of 2026. The halving is the mechanism that caps Bitcoin's supply at 21 million coins and steadily shifts miners' revenue from subsidy toward fees.

The cost side is dominated by one thing: **electricity**. Mining machines (ASICs — application-specific integrated circuits built only to hash) run flat out, 24/7, converting power into guesses. A miner's edge is almost entirely the price they pay per kilowatt-hour, which is why mining migrates to wherever power is cheapest — hydro dams, stranded gas, off-peak grids. When the coin price falls or the electricity price rises, the least efficient miners switch off, total hashrate drops, and difficulty adjusts down to compensate. Mining is a constant-margin commodity business where the product (block space) has a wildly volatile price.

Two structural features of mining matter for the centralization story later. The first is **mining pools**. Because a single small miner might wait years to win a block on their own, miners join a pool that combines everyone's hashrate, wins blocks far more regularly, and pays out proportionally to the work each member contributed — smoothing a lottery into a wage. The pool operator, however, is the one who actually decides which transactions go into the pool's blocks and in what order, so the *ordering power* concentrates in the hands of a few pool operators even when the underlying hashrate is widely distributed. A handful of pools have at times controlled a majority of Bitcoin's hashrate, which is uncomfortable precisely because it concentrates ordering.

The second is the **51 percent attack**, the theoretical nightmare proof-of-work is designed to make expensive. If a single entity ever controlled more than half the total hashrate, it could consistently out-mine everyone else, rewrite recent history, and double-spend coins. The defense is purely economic: acquiring that much hashrate would cost so much in hardware and electricity, and the attack would so obviously destroy the coin's value (and thus the attacker's own holdings and reward stream), that it is not worth attempting on a large, valuable chain. Proof-of-work security is, at bottom, the statement "rewriting the ledger costs more than you could steal by doing it." Smaller chains with cheap hashrate have in fact suffered real 51 percent attacks, which is the proof that the economic argument, not magic, is what protects the big ones.

#### Worked example: is a Bitcoin block worth mining?

Suppose a miner runs a fleet drawing 10 megawatts (10,000 kilowatts) of power, and they pay \$0.05 per kilowatt-hour for electricity. Their power bill per hour is 10,000 kW x \$0.05 = \$500 per hour, or \$12,000 per day. Add maintenance, cooling, and amortized hardware, and say their all-in daily cost is \$18,000.

Now the revenue. At a block subsidy of 3.125 BTC plus, say, 0.4 BTC of fees, each block pays the lucky winner about 3.525 BTC. At a BTC price of \$60,000 (illustrative, as-of figure), that is roughly \$211,500 per block. Bitcoin produces about 144 blocks per day, so the *entire network* earns about 144 x \$211,500 = \$30.5 million per day, shared in proportion to hashrate.

If our miner controls 0.1% of total network hashrate, their expected daily revenue is 0.1% x \$30.5 million = \$30,500. Subtract the \$18,000 daily cost and they net about \$12,500 per day — profitable, but thin. Drop the coin price 40% to \$36,000 and revenue falls to about \$18,300, barely covering cost; many less-efficient miners would now be losing money and would power down. **The intuition: a miner's profit is a thin, volatile spread between a fixed electricity bill and a block reward whose dollar value swings with the coin price.**

## Staking economics: yield, slashing, and liquid staking

Proof-of-stake replaces the electricity bill with locked capital. A validator's revenue is a **yield** — an annual percentage return on their staked ETH — and the cost is the opportunity cost of locking the capital plus the tail risk of **slashing**.

The yield has three components. First, **issuance**: the protocol mints new ETH and pays it to validators for proposing and attesting to blocks; the more total ETH staked across the network, the lower this rate per validator (it is spread across more stakers). Second, **priority fees**: the tips users attach to get included. Third, **MEV**: the value flowing up from the supply chain we just traced. As-of 2026, the combined yield on Ethereum staking has run in the neighborhood of 3 to 4 percent annually, with MEV adding perhaps a quarter to a third of the total in active markets — meaning a non-trivial slice of a validator's "yield" is, ultimately, value extracted from other users' trades.

The cost side is **slashing risk** and **lockup**. Your 32 ETH is illiquid while staked, and a serious validator fault can burn part of it. In practice, slashing is rare for honest validators running correct software; the bigger day-to-day risk is downtime penalties (small) and the lockup itself.

It is worth being concrete about what a validator actually *does* to earn the yield, because it shapes both the rewards and the risks. A validator has two recurring jobs. Most of the time it **attests** — votes that it has seen the latest block and agrees on the chain's head; honest, on-time attestations earn a small steady reward, and missing them (downtime) costs a small steady penalty. Occasionally, when the protocol selects it, the validator gets the far more lucrative job of **proposing** a block, which is when it collects the priority fees and MEV. Because proposal slots are randomly assigned and rare for any one validator, most of a small staker's reward is the dull, reliable attestation income, with the occasional proposal — and its MEV windfall — as the upside. The two slashing offenses both involve *equivocating*: signing two different blocks for the same slot, or making contradictory attestations. These only happen if you run two copies of your validator keys at once (a common misconfiguration) or run buggy software, which is why the cardinal rule of staking is "never run your keys in two places."

The lockup is no longer permanent: since the Shanghai upgrade in 2023, validators can withdraw their stake and rewards by exiting the validator set through a queue, so staked ETH is illiquid but not frozen forever. Still, exiting takes time, and during volatile markets the exit queue can stretch to days, which is itself a risk if you need the capital quickly.

That lockup created demand for **liquid staking**. A liquid staking protocol (Lido is the largest) takes your ETH, stakes it on your behalf via professional node operators, and hands you a token (stETH) representing your staked position plus accruing rewards. You can trade or use that token in DeFi while the underlying ETH stays staked — you get the yield without the illiquidity. The convenience is real; the cost is another layer of centralization, because a single liquid staking protocol can come to control a large fraction of all staked ETH.

#### Worked example: staking 32 ETH at 4 percent

You stake exactly 32 ETH. At an ETH price of \$2,500 (illustrative, as-of figure), that is \$80,000 of locked capital. Suppose the all-in staking yield is 4.0 percent annually, of which roughly 3.0 percent is issuance plus base fees and 1.0 percent is MEV.

In one year you earn 4.0% x 32 = 1.28 ETH. At \$2,500, that is \$3,200 of yield. Of that, the MEV slice (1.0% x 32 = 0.32 ETH, about \$800) is value that flowed up from searchers and builders — much of it ultimately from other users' trades. Your base reward (0.96 ETH, about \$2,400) is newly issued ETH and fees.

Now the risk. If you run flawless software, your expected slashing loss is essentially zero. But if you misconfigure your validator and get slashed the minimum penalty (around 1 ETH plus correlated penalties in a bad scenario, say 1.5 ETH total), you lose about \$3,750 — more than a year's yield — in one event. **The intuition: staking turns your capital into a bond-like 3 to 4 percent yield, part of it quietly funded by MEV, with a rare but sharp downside if your node misbehaves.**

## A sandwich attack, step by step

Now the example that makes MEV visceral, because it is the case where an ordinary person is the victim. A sandwich attack works only because your swap is public in the mempool and the attacker controls ordering around it.

![A swap shown with and without a sandwich attack](/imgs/blogs/crypto-mining-staking-and-mev-3.png)

First, a one-paragraph primer on the price mechanism it exploits. A DEX like Uniswap prices a token with a **constant-product** pool: it holds reserves of two tokens, and the product of the reserves stays constant through any trade. The practical consequence is **slippage**: the bigger your swap relative to the pool, the worse the price you get, because each token you buy moves the price against you. Your wallet protects you with a **slippage tolerance** — a maximum acceptable price move, often set to 0.5 percent or 1 percent by default. A sandwich attack lives inside that tolerance: the bot moves the price against you by just under your tolerance, so your trade still executes, just at the worst price you were willing to accept.

#### Worked example: sandwiching a \$10,000 swap

You want to swap \$10,000 of USDC for ETH, and your wallet's slippage tolerance is set to 2 percent (a common default for a volatile pair). Here is the sequence the attacker constructs as a bundle:

1. **The front-run (attacker buys first).** The searcher sees your pending \$10,000 buy. They submit their own buy of, say, \$10,000 of ETH *ordered immediately before yours*. This purchase pushes the pool's ETH price up — let us say it raises the effective price by about 1 percent.
2. **Your trade executes at the worse price.** Your \$10,000 swap now runs against the price the attacker just inflated. Instead of getting \$10,000 worth of ETH at the original price, you get ETH at roughly 1.8 percent worse — just inside your 2 percent tolerance, so your transaction does not revert. In dollar terms you receive about \$9,815 of ETH where a clean trade would have given you roughly \$9,950 after the normal 0.3 percent pool fee and minor natural slippage. You are out about \$135 versus the clean trade, and you have no idea.
3. **The back-run (attacker sells after).** Immediately after your trade — which pushed the price up further — the attacker sells the ETH they bought in step 1, at the now-elevated price. They bought low (before your trade) and sold high (after it), capturing the price impact your trade created.

Tally the attacker's side. They bought \$10,000 of ETH and sold it moments later into a price your buying pressure lifted; the round-trip nets them, in this example, about \$185 gross. From that they pay gas and a priority tip to the builder — say \$50 — leaving roughly \$135 of net profit. Notice the symmetry: **your loss versus a clean trade (~\$135) is almost exactly the attacker's profit.** A sandwich does not create value; it transfers your slippage into someone else's wallet.

Two defenses follow directly from the mechanism. Lower your slippage tolerance and large sandwiches become unprofitable (but your trade may fail in volatile moments). Or route your trade through a **private** channel that never enters the public mempool — many wallets now offer "MEV protection" that submits your transaction directly to builders, hiding it from searchers. The fix attacks the precondition: no public visibility, no sandwich.

There is a subtle arithmetic to why slippage tolerance is the key dial. The attacker's front-run must move the price by less than your tolerance, or your transaction reverts and they earn nothing — but the front-run also costs them gas, and they only profit if the price impact they can safely create exceeds those costs. So the attack is profitable only when your tolerance leaves a band wide enough to fit a price move that more than covers the bot's gas. Tighten the tolerance from 2 percent to 0.3 percent and that band collapses; on most pairs there is simply no longer room for the bot to move the price, get paid, and clear its costs. The trade-off is real: in a fast-moving market the legitimate price may itself drift more than 0.3 percent between when you sign and when you execute, and your transaction will revert, wasting a little gas. The right tolerance is therefore a function of how volatile the pair is and how large your trade is relative to the pool — which is exactly the judgment a good trade-routing tool makes for you automatically.

A third, more structural defense is changing the venue's design so ordering cannot be exploited. **Batch auctions** (used by protocols like CoW Protocol) collect many orders over a short interval and settle them all at a single uniform clearing price, so there is no within-batch ordering to exploit and no "before" or "after" for a bot to occupy. **Encrypted mempools** hide transaction contents until after ordering is fixed, removing the searcher's ability to see what you intend. Both attack the same two preconditions — public visibility and exploitable ordering — at the protocol level rather than asking each user to defend themselves.

## A cross-DEX arbitrage opportunity

Sandwiching is the villainous face of MEV; arbitrage is the mundane, dominant one. It is worth working through because it shows MEV doing something economically useful — keeping prices consistent — while still being a race decided by transaction ordering.

#### Worked example: arbitraging an ETH price gap

Suppose ETH trades at \$2,500 on Uniswap and, for a few seconds, at \$2,520 on Sushiswap, because a large buy just moved the Uniswap price and the two pools have not yet re-equalized. A searcher constructs a single bundle that, in one atomic transaction:

1. **Buys ETH cheap on Uniswap.** Spend \$250,000 to buy 100 ETH at \$2,500. (The exact amount is chosen so the trade does not move Uniswap's price so far that the gap closes against them; this sizing is the searcher's craft.)
2. **Sells the same ETH dear on Sushiswap.** Sell 100 ETH at \$2,520, receiving \$252,000.
3. **Pockets the difference.** Gross profit is \$252,000 - \$250,000 = \$2,000, before costs.

Now the costs. Each leg pays a 0.3 percent pool fee: 0.3% x \$250,000 ≈ \$750 on the buy and ≈ \$756 on the sell, about \$1,506 in fees. That leaves roughly \$494 before the priority tip. In a competitive market, the searcher must bid much of that \$494 to the builder to win inclusion — perhaps \$400 — leaving them a thin \$94. If they bid too little, a rival searcher who bids more lands their bundle first and the opportunity is gone. **The intuition: arbitrage MEV is a real service — it dragged the two ETH prices back together — but competition bids almost all the profit away to whoever controls block ordering.**

Notice what just happened to the prices: the searcher's buying nudged Uniswap up and their selling nudged Sushiswap down, closing the \$20 gap. The arbitrageur was paid to make the two venues agree. That is why most researchers treat arbitrage MEV as benign-to-useful, unlike sandwiching.

## How big is the MEV "tax"?

The natural question is how much of this is happening. The honest answer is that nobody knows exactly, because some MEV is extracted through private channels that public dashboards cannot fully see, and definitions vary. But the measured numbers are large enough to matter.

#### Worked example: estimating annual MEV extracted

Public trackers that reconstruct MEV from on-chain data (the best known is the dataset maintained by the Flashbots research community) have attributed something on the order of \$1.5 billion or more of cumulative MEV on Ethereum since detailed tracking began around 2020 (figures are estimates and as-of 2026; the true total is higher because private flow is undercounted). To get a rough annual sense, spread a conservative \$700 million of clearly-measured extraction across the roughly four most active years and you get on the order of \$175 million per year, with the bulk being arbitrage and liquidations and a smaller, more harmful slice being sandwiching.

Put that against trading volume. If Ethereum DEXes process, very roughly, \$1 trillion of swap volume in an active year, then \$175 million of extraction is about 0.0175 percent of volume — a small per-trade tax on average, but heavily concentrated on large, naive swaps that get sandwiched, where the effective tax can be a full percent or more. **The intuition: averaged over everyone, MEV is a thin tax measured in basis points; concentrated on the unprotected large trader, it can be the single biggest cost of the trade.**

The number to remember is not the precise total — which is genuinely uncertain — but the shape: MEV is a multi-hundred-million-dollar-per-year industry that most users never knew they were funding.

A second way to feel the scale is to look at what validators actually receive. In active markets, the MEV that flows up through MEV-Boost has at times accounted for a meaningful fraction of the *total* reward a validator earns from a block — there are individual blocks where the MEV component dwarfs the base issuance and ordinary fees by an order of magnitude, typically when a single enormous arbitrage or liquidation lands. Averaged over time these spikes smooth out, but they reveal the underlying truth: a non-trivial share of the return that makes staking attractive is, at root, recycled from value extracted out of other people's trades. That is not a moral judgment — most of it is benign arbitrage — but it is a fact worth internalizing before you treat a staking "yield" as if it were a risk-free interest rate. It is partly a fee on the chain's own trading activity, and it rises and falls with how much trading (and mispricing) there is.

The deepest measurement problem is that the most predatory MEV is the hardest to count. Arbitrage and liquidations leave clean, attributable on-chain footprints, but a growing share of flow now moves through private channels precisely to avoid the public mempool, and what happens inside a builder before a block is sealed is opaque. So the published totals almost certainly *undercount* extraction, and the true per-trade tax on an unprotected large swap is higher than the network-wide average suggests.

## How block production evolved: mining, the Merge, and PBS

The architecture we have been describing did not appear fully formed. It evolved as the value at stake grew, and the evolution explains why the system looks the way it does.

![A timeline from Bitcoin mining through the Merge to enshrined PBS](/imgs/blogs/crypto-mining-staking-and-mev-6.png)

In **2009**, Bitcoin launched proof-of-work mining, and for years "MEV" barely existed because there was little on-chain trading to exploit — miners simply ordered transactions by fee. **2015** brought Ethereum, also proof-of-work, and with it programmable smart contracts: DEXes, lending protocols, the whole DeFi machine that *creates* MEV opportunities. As DeFi grew through 2020, so did the chaos of searchers fighting over them in public gas wars.

**2020** is when **Flashbots** formalized the market: a private auction (MEV-Geth, then MEV-Boost) where searchers could submit bundles to miners directly, ending the on-chain spam but industrializing extraction. **2022** brought **the Merge**: Ethereum switched from proof-of-work to proof-of-stake, retiring miners entirely. MEV did not disappear — it moved to **validators**, who now hold the ordering power miners used to. Within weeks of the Merge, **MEV-Boost** became the dominant way validators captured it: by 2023 it was running on roughly 90 percent of Ethereum blocks (as-of figure), meaning the large majority of blocks were built by outsourced builders, not by the validators who proposed them.

That brings us to the present debate. Because MEV-Boost is "out-of-protocol" (bolted on, run by trusted relays rather than enforced by the blockchain itself), researchers are designing **enshrined PBS** — building proposer-builder separation directly into Ethereum's rules so it does not depend on trusting relays. As-of 2026, this remains an active research and governance topic rather than a shipped feature. The arc is clear: ordering power has been progressively split out of the block proposer's hands and into a specialized, competitive — and worryingly concentrated — market.

## The centralization worry

The deepest concern about MEV is not that users pay a small tax. It is that the machinery built to harvest MEV efficiently pushes the whole system toward centralization, which undermines the censorship-resistance that is supposed to be crypto's entire point.

The mechanism is straightforward. Building maximally profitable blocks is a winner-take-most game: the builder with the best algorithms, the most exclusive order flow, and the deepest searcher relationships builds the most valuable blocks, wins the most auctions, and attracts even more order flow — a flywheel. The result, observed repeatedly since the Merge, is that a small handful of builders construct the large majority of Ethereum blocks. If two or three builders order most of the chain's transactions, they collectively hold the power to delay or exclude specific transactions — to censor — even if no single one controls a majority.

The same flywheel operates in staking through liquid staking: the largest liquid staking protocol can come to control a quarter or more of all staked ETH, and in mining through pools, where a few mining pools have at times controlled most of Bitcoin's hashrate. In each case, the economically efficient outcome (let specialists do the hard part) collides with the political goal (no one should be able to control the ledger). This tension — efficiency pulling toward concentration, security demanding decentralization — is the unresolved core problem that enshrined PBS, inclusion lists, encrypted mempools, and a dozen other research directions are all trying to address.

It is worth distinguishing two different failures that "centralization" can cause, because they are often blurred. The first is **censorship**: if a few builders construct most blocks, they can refuse to include particular transactions — say, those touching a sanctioned address — and a transaction that every builder refuses simply never settles. This already happened in practice after the Merge, when a majority of MEV-Boost blocks for a period excluded transactions from a sanctioned protocol; the chain kept running, but its neutrality — the promise that anyone's valid transaction will eventually be included — was visibly dented. The proposed countermeasure is an **inclusion list**: a rule letting the proposer force the builder to include a set of transactions the proposer specifies, so that even a censoring builder cannot keep a transaction out forever.

The second failure is **value capture and rent extraction**: a concentrated builder market can quietly keep more of the MEV for itself rather than passing it up to validators in competitive bids, or can use exclusive order flow to entrench its position so newcomers cannot compete. This is a softer harm than censorship — nobody is locked out of the chain — but over time it determines who earns the rents that ordering generates, and whether staking stays accessible to small participants or drifts toward a few large operators. Enshrined PBS aims at both failures at once by writing the proposer-builder split into the protocol so it does not depend on trusting a small set of off-chain relays, and by pairing it with inclusion lists so censorship has a built-in escape valve.

For a fuller picture of how the underlying programmable-money layer works, see [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money); for the DEXes and lending protocols that *generate* most MEV, see [DeFi protocols: Uniswap, Aave, and MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).

## Common misconceptions

**"MEV is the same as hacking or theft."** Mostly no. Arbitrage and liquidations are designed, legitimate features — the protocols *want* someone to correct prices and clear bad debt, and they pay for it. Sandwiching is predatory and harms users, but it operates entirely within the rules: it exploits public information and ordering power, not a bug. MEV is better understood as the rent that accrues to whoever controls a scarce resource (ordering) than as theft. That distinction matters because you cannot "patch" MEV the way you patch a hack; you have to change the market structure.

**"Proof-of-stake fixed MEV."** No. The Merge moved the ordering power from miners to validators and renamed the term from "miner" to "maximal" extractable value, but the opportunities are identical because they come from DeFi and the public mempool, not from how block producers are chosen. If anything, PoS plus MEV-Boost made the extraction *more* organized.

**"Higher gas fees mean you are protected from MEV."** No — paying more gas just means your transaction is included faster; it does nothing about a searcher sandwiching you within the same block. Protection comes from hiding your transaction from the public mempool (private order flow) or tightening slippage, not from outbidding on gas.

**"Only whales get sandwiched."** Large swaps are juicier, but the threshold is set by profitability, not size alone — in a cheap-gas environment, even mid-sized swaps with loose slippage tolerance are worth sandwiching. The real determinant is whether your slippage tolerance leaves enough room for a profitable attack after the bot's costs.

**"Validators personally hunt for MEV."** Under MEV-Boost, almost never. The whole point of proposer-builder separation is that the validator outsources the profitable, complex block-building to specialist builders and just sells the block space to the highest bidder. A solo staker with 32 ETH captures MEV passively, through the auction, without writing a single arbitrage bot.

**"MEV is an Ethereum-only problem."** No. Any blockchain with a public mempool and on-chain trading has MEV — Solana, BNB Chain, and others all exhibit it, often with different microstructure. Ethereum is simply where it has been studied and instrumented most thoroughly, so its numbers are the ones we can cite.

**"MEV could just be banned or coded away."** Not cleanly. You cannot outlaw arbitrage without breaking the price-correction the system relies on, and you cannot outlaw liquidations without leaving lending protocols full of bad debt. Because most MEV comes from *legitimate* economic activity plus the simple fact that someone must order transactions, the realistic goal is not to abolish it but to *redistribute* it more fairly (back to users or to the protocol) and to *neutralize* its harms (censorship, sandwiching) through better market design. "Make MEV go away" is the wrong frame; "decide who captures the rent, and stop the predatory slice" is the real one.

## How it shows up in real markets

**The 2021–22 "MEV bot wars."** Before private auctions dominated, searchers competed by bidding gas prices into the stratosphere to get their bundle ordered first — so-called priority gas auctions. At peak, these gas wars congested Ethereum and pushed fees painfully high for ordinary users who were merely caught in the crossfire of bots fighting over arbitrage. The original Flashbots research paper, "Flash Boys 2.0" (a nod to Michael Lewis's HFT book), documented this dynamic and named the problem; the move to off-chain auctions was the direct response.

**A real sandwich, in the wild.** On-chain forensic accounts regularly surface specific sandwich attacks where a single large, loosely-protected swap lost a meaningful fraction of its value to a bot — cases of users losing tens of thousands of dollars on a single trade because their slippage tolerance was set high and their transaction sat in the public mempool. These are not hypotheticals; they are reconstructable transaction-by-transaction on a block explorer, which is part of what makes MEV such a vivid object of study.

**The Merge moving MEV to validators (September 2022).** When Ethereum switched to proof-of-stake, the entire MEV apparatus had to migrate from miners to validators essentially overnight. MEV-Boost was ready, and within weeks the large majority of validators were running it — one of the fastest pieces of financial-infrastructure adoption on record, precisely because the money on the table was immediate and obvious.

**Builder centralization after MEV-Boost.** In the year-plus after the Merge, observers tracking block-builder market share repeatedly found that a small number of builders were constructing the majority of Ethereum blocks, and that some builders were excluding transactions from sanctioned addresses — a concrete instance of the censorship concern moving from theoretical to real. This is the episode most often cited by researchers pushing for enshrined, in-protocol PBS and inclusion lists.

**A famous arbitrage.** Atomic arbitrage bundles that net six figures in a single block surface periodically when a large mispricing appears — for example, when a big swap or an oracle update briefly dislocates prices across venues, a single bundle can capture a very large gap in one transaction. These are the benign giants of MEV: a bot made the venues agree on a price and was richly paid for the few seconds of capital and the winning bid.

**The Three Arrows and FTX era stress tests.** During the 2022 crypto deleveraging, cascading liquidations across lending protocols were a feast for liquidation searchers, who competed to seize discounted collateral as positions blew up. The same forced-selling dynamics that hurt leveraged borrowers (see [the 3AC and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion)) generated enormous liquidation MEV — a reminder that MEV volume spikes exactly when markets are most stressed.

## When this matters to you / further reading

If you ever swap tokens on a DEX, MEV is not academic — it is a line item you pay. The practical takeaways are concrete. **Set a tight slippage tolerance** (and accept that very volatile trades may fail rather than overpay). **Use a wallet's MEV-protection / private-transaction feature** for large swaps so your order never hits the public mempool. **Break very large swaps into pieces** or use an aggregator that routes to minimize price impact. None of this is investment advice; it is plumbing hygiene, the crypto equivalent of not announcing your large order to the whole trading floor before you place it.

If you stake — directly or through a liquid staking token — understand that a real slice of your "yield" is MEV flowing up from other users' trades, and that the convenience of liquid staking comes at the cost of concentrating stake in a few operators. If you mine, you already live the thin, volatile spread between an electricity bill and a coin price.

The bigger reason to care is structural. MEV is the clearest worked example of a deep truth about all markets, not just crypto: **whoever controls the sequence of trades holds power, and that power has a price.** Traditional markets learned this the hard way — it is why front-running is illegal and why exchange co-location and order-routing rules exist. Crypto rebuilt the same problem in the open, where you can watch every dollar of it move on a block explorer. For how the analogous game plays out in traditional markets, see [market makers and high-frequency trading](/blog/trading/finance/market-makers-and-high-frequency-trading); for the origins of the trust-minimized vision that MEV complicates, see [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision).

To go deeper: read the original Flashbots "Flash Boys 2.0" paper for the framing of the problem; follow the Flashbots and Ethereum Foundation research on proposer-builder separation, inclusion lists, and encrypted mempools for the proposed fixes; and watch a public MEV dashboard for a few days to see, in real time, the invisible market in block space doing its quiet, relentless work behind every clean-looking swap.
