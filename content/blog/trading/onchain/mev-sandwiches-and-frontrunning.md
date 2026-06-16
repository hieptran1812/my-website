---
title: "MEV, Sandwiches, and Frontrunning: The Tax on Every On-Chain Trade"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero guide to Maximal Extractable Value — what MEV is, how a sandwich attack makes your DEX swap fill worse, how to spot one in your own transaction history, and how to defend with tight slippage, private RPCs, and MEV-aware routers."
tags: ["onchain", "crypto", "mev", "sandwich-attack", "frontrunning", "backrunning", "slippage", "flashbots", "mempool", "dex", "ethereum", "defi"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Maximal Extractable Value (MEV) is the profit bots make by reordering or inserting transactions around yours; the toxic version, the sandwich attack, makes your DEX swap fill at a worse price and quietly taxes you.
>
> - **MEV is value extracted from transaction ordering.** Because the blockchain's pending-transaction waiting room (the mempool) is public, a bot can see your swap before it confirms and decide what trades sit just before and just after it inside the block.
> - **A sandwich is the toxic flavor:** the bot buys right before you to push the price up, lets your swap fill at that worse price, then sells right after — pocketing the spread, which is your lost value.
> - **You can spot it and defend.** In your block, a sandwich looks like one bot address buying just above your tx and selling just below it, with your fill landing near your slippage cap. Defend with tight slippage, a private RPC like Flashbots Protect, an MEV-aware router (CoW Swap, an MEV-blocker RPC), and by avoiding thin pools.
> - **The number to remember:** loose slippage is the open door. A \$50,000 swap at a 2% slippage tolerance can be sandwiched for roughly \$180; tightening that cap to 0.5% can shrink the loss to about \$45.

On a quiet afternoon you decide to buy a token on a decentralized exchange. You type in \$50,000, the interface shows you a quote, you set slippage to the default 2%, and you click swap. Twelve seconds later your transaction confirms. You got your tokens. Nothing looks wrong. The trade "worked".

But if you open that block in an explorer and look at the transaction immediately above yours, you find a buy of the exact same token, in the exact same pool, from an address you have never seen. Look at the transaction immediately below yours, and there is a sell of that token from the same address. The bot bought before you, watched your \$50,000 push the price up, then sold into the price *you* lifted — all in the same block, all in a fraction of a second. The difference between its buy and its sell is real money, and a slice of it came out of your fill. You were sandwiched.

This is not a rare exploit or a clever hack of a buggy contract. It is the steady-state behavior of public blockchains, happening tens of thousands of times a day, and it has a name: **MEV**. The pattern has been documented since at least 2019, when the influential "Flash Boys 2.0" research paper put a name to the on-chain priority-gas auctions bots were already fighting in; by the time Ethereum moved to proof-of-stake in September 2022 and the Flashbots block-building infrastructure became standard, sandwich extraction was an industrialized, multi-billion-dollar business running quietly under every public swap. The mechanism has not changed since: it is the same public mempool and the same ordering auction, just at larger scale.

This post teaches you what MEV is from zero, walks through the sandwich attack step by step, shows you how to find one in your own transaction history with free tools, and gives you a concrete playbook to stop bleeding value on every on-chain trade. For the deeper market-microstructure view of *who* extracts MEV and *how the block-building supply chain works*, this post is the trader-and-defender companion to [Mining, Staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev).

![A five-step pipeline showing a bot buying before a user swap, the user filling at a worse price, and the bot selling after to pocket the spread](/imgs/blogs/mev-sandwiches-and-frontrunning-1.png)

## Foundations: the mempool, transaction order, and where MEV comes from

Before any of the attacks make sense, you need a small, concrete vocabulary for how a transaction actually travels from your wallet into a block. None of this requires prior background. I will define every term the first time it appears and keep each definition physical.

### A transaction is not instant — it waits

When you click "swap" in a wallet, your wallet does one thing locally: it **signs** the transaction with your private key. Signing proves you authorized it; it does not yet touch the chain. The signed transaction is then **broadcast** — handed to a node, which gossips it to other nodes across the network. For a fuller tour of what a single transaction contains and the states it passes through, see [Anatomy of a Transaction](/blog/trading/onchain/anatomy-of-a-transaction).

The signed-but-not-yet-included transaction lands in the **mempool** — short for "memory pool", the public waiting room for pending transactions. It is a holding pen. Your transaction sits there, visible to everyone running a node, until some block producer scoops it up and includes it in a block. The single most important word in that sentence is *public*. By default, every pending transaction is broadcast to the whole world *before* it executes. Anyone watching the waiting room can see what you are about to do and react before you do it. That visibility is the soil in which MEV grows.

### Order inside a block is decided, not random

A block is an ordered list of transactions. The order is not first-come-first-served, and it is not random. Someone *chooses* it. On Ethereum today, that someone is a small supply chain:

- **Searchers** are bots that scan the mempool for profitable opportunities. When a searcher spots one, it assembles a **bundle** — a small ordered package of transactions it wants placed together in a specific sequence.
- **Builders** are specialized parties that take bundles and ordinary transactions and assemble the single most profitable ordering of an entire block.
- **Validators** (also called proposers) are the parties chosen to actually publish the next block. They typically auction off the right to fill their block to the highest-bidding builder.

The key consequence: whoever controls the order can place a transaction *immediately before* or *immediately after* yours. That position — adjacency to your trade, inside the same block — is the lever. Everything in this post is built on it.

### Gas, priority fees, and the ordering auction

To get into a block, a transaction pays a **fee** (denominated in "gas"). Part of that fee is a **priority fee** (a tip) that a transaction attaches to bid for faster, earlier inclusion. Higher tip, better position — roughly. So the ordering of a block is, in practice, the outcome of a continuous, silent **auction**: transactions and bundles bid in priority fees for the slot they want.

A searcher that finds a profitable position next to your trade does not politely ask for it. It *bids* for it — paying a high priority fee, sometimes most of its expected profit, to guarantee its buy lands right before you and its sell right after. The auction is why you cannot simply out-click a bot: it is bidding money, in milliseconds, for the exact ordering it needs.

![A branching graph from a wallet through the public mempool to searcher bots, an MEV bundle, a block builder, a validator, and the final block](/imgs/blogs/mev-sandwiches-and-frontrunning-2.png)

### What MEV actually is

Now the definition lands cleanly. **Maximal Extractable Value (MEV)** is the maximum value that can be extracted from producing a block *beyond the standard block reward and ordinary fees*, by including, excluding, and — above all — *reordering* the transactions in it. It was originally called "Miner Extractable Value" in the proof-of-work era, when miners chose order; after Ethereum moved to proof-of-stake, the "M" was re-read as "Maximal" because validators and builders, not miners, now do the choosing.

MEV is not one thing. It comes in flavors, distinguished by *where the bot's transaction sits relative to yours* and *who pays for it*:

- **Frontrunning** — the bot places its transaction *just before* yours, to beat you to a price or copy a profitable move. You lose because you wanted to be first and a faster bidder got there.
- **Backrunning** — the bot places its transaction *just after* yours, to clean up the state your trade left behind. Most backrunning is benign (arbitrage that closes a price gap, a liquidation that the protocol needs anyway).
- **Sandwiching** — the bot does *both at once*, wrapping your trade in a buy before and a sell after. This is the toxic flavor, and it exists specifically to extract value from *you*.

### Slippage: the door the sandwich walks through

The last foundation is the one that makes you, the ordinary trader, the target. When you swap on a decentralized exchange (a DEX) built on an **automated market maker (AMM)** like Uniswap, you do not trade against a person — you trade against a **liquidity pool**, a smart contract holding two assets. The price is set by a formula on the ratio of the two assets in the pool, so *your own trade moves the price*: a buy pushes it up, a sell pushes it down. The bigger your trade relative to the pool, the more it moves — this is **price impact**. (For the broader mechanics of Uniswap-style pools, see [DeFi Protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).)

Because the price can move between the moment you sign and the moment you execute, every DEX swap carries a **slippage tolerance**: the maximum adverse price change you will accept before the swap reverts. Set it to 2% and you are saying "fill me anywhere up to 2% worse than quoted; beyond that, cancel." That tolerance is meant to protect you from honest price movement. But a sandwich bot reads it as a *budget*: it knows it can push the price up by *almost* your full tolerance and you will still accept the fill, because you told the contract to. Loose slippage is not just a risk setting — it is the size of the meal you are serving the bot.

### Why the price moves: the constant-product curve

It is worth spending one more paragraph on *why* your own trade moves the price, because the whole sandwich rests on it. A classic AMM pool holds two assets and keeps the product of their quantities roughly constant: if the pool has `x` units of token A and `y` units of token B, it tries to keep `x · y = k` fixed as people trade. The price of A in terms of B is just the ratio `y / x`. When you buy token A out of the pool, `x` falls, so to keep `x · y` constant `y` must rise — and the price `y / x` climbs. The more A you pull out in one trade, the further along this curve you walk, and the worse your average price gets. That curvature is the mechanism: a small trade barely moves the ratio, a large trade in a shallow pool swings it hard.

Two consequences fall straight out of the curve, and both feed the sandwich. First, **price impact is convex** — doubling your order size more than doubles the price you move it. Second, the bot can run the curve in its favor on purpose: by buying first it walks the price up the curve, hands you a worse spot on it, then sells back down the curve to collect the difference. The pool does not lose money on net (the constant-product math protects it); the value transferred is purely from your fill to the bot. Understanding the curve is what lets you reason about *which* trades are sandwichable: a \$1,000 swap in a \$50,000,000 pool barely registers, while a \$1,000 swap in a \$30,000 pool is a feast.

#### Worked example: how pool depth decides the size of the meal

Take a token priced at \$1.00. In a deep pool holding \$50,000,000 of liquidity, a \$50,000 buy moves the price by only about 0.1% — there is almost nothing for a bot to skim, maybe a few dollars, and after gas the sandwich is unprofitable. Now run the same \$50,000 buy into a shallow pool holding just \$1,000,000 of liquidity: the same order now moves the price by roughly 5%, blowing straight past a 2% slippage cap. A bot can front-run inside that move and extract a meaningful slice — potentially \$300 to \$500 — before your fill trips the limit. Same dollar order, same token, same price: the *pool depth* alone changed the meal from a few dollars to several hundred. This is why "avoid thin pools" is not folklore — it is the price-impact curve doing arithmetic.

## How a sandwich works, step by step

Now we put the foundations together. A sandwich is three transactions in one block, in this exact order: bot buy, your swap, bot sell. Walk through it slowly.

**Step 1 — The bot sees you.** Your \$50,000 swap is broadcast and sits in the public mempool, unconfirmed. A searcher bot scanning the mempool spots it: a large buy into a pool, with a slippage tolerance it can read directly from the transaction's parameters.

**Step 2 — The bot buys first (the front-run).** The bot submits its own buy of the same token in the same pool and bids a high priority fee to guarantee its buy is ordered *just before* yours. Its buy moves the pool price up. The pool is now more expensive than when you got your quote.

**Step 3 — Your swap fills at the worse price.** Your transaction executes next, at the price the bot just inflated. Because the bot was careful to push the price up by *less* than your slippage tolerance, your swap does not revert — it fills, just at a worse rate than you would have gotten in an empty block. You receive fewer tokens for your \$50,000 than the quote promised.

**Step 4 — The bot sells after (the back-run).** Immediately after your swap, the bot sells the tokens it bought in Step 2, dumping into the price your own buy just lifted further. The bot exits at a higher price than it entered.

**Step 5 — The bot pockets the spread.** The difference between the bot's sell proceeds and its buy cost, minus the gas and priority fees it paid, is its profit. That profit is, almost exactly, the extra value you lost by filling at the inflated price. The pool ends roughly back where it started — the sandwich does not change the long-run price, it just transfers a slice of your trade to the bot.

The detail that makes this nearly risk-free for the bot is **atomic bundling**. The searcher does not fire three independent transactions and hope they land in order — it submits a *bundle* to a builder, with the explicit instruction "include these three transactions, in this exact order, or include none of them". If the builder cannot place the buy directly before your swap and the sell directly after, the whole bundle is dropped and the bot loses nothing but a failed bid. So the bot only ever executes the sandwich when it is guaranteed profitable: it never gets stuck holding the token because its sell failed, and it never buys without your swap landing in between. This asymmetry — guaranteed profit when it works, near-zero cost when it does not — is why sandwich bots can fire on thousands of opportunities a day and why the only durable defense is to remove yourself from the opportunity set entirely (private routing) rather than hope to out-bid them.

![A horizontal timeline of one block showing the pool price at rest, rising on the bot buy, the user fill near the slippage cap, and falling on the bot sell](/imgs/blogs/mev-sandwiches-and-frontrunning-3.png)

The figure above shows the price walk: at rest at \$1.000, lifted to \$1.018 by the bot's front-run, your fill landing there (near a 2% cap), then dropped back to \$1.002 by the bot's back-run. You are left holding tokens bought above their fair price, and the gap is the bot's lunch.

#### Worked example: a \$50,000 swap sandwiched at 2% slippage

Say you swap \$50,000 of a stablecoin for a token in a pool, with slippage set to the default 2%. In an empty block your trade itself has some honest price impact — call the *fair* fill price \$1.000 per token, so you would expect roughly 50,000 tokens.

A sandwich bot front-runs you, pushing the entry price up to about \$1.0036 — just under 0.4% above fair, comfortably inside your 2% cap. Your \$50,000 now buys about 49,820 tokens instead of 50,000. You are short roughly 180 tokens, worth about \$180 at the fair price. The bot then sells its position into your buy, capturing close to that \$180 (minus its fees) as profit. Your loss: about \$180 on a \$50,000 trade, or roughly 0.36%. It feels invisible because the swap "succeeded" — but the bot quietly skimmed \$180 because your 2% tolerance gave it that much room to work.

#### Worked example: tightening slippage to 0.5% caps the loss

Now run the same \$50,000 swap, but set slippage to 0.5% instead of 2%. The bot's whole game is to push the price up by *almost* your full tolerance without tripping the revert. Drop the tolerance and you shrink its working room from 2% to 0.5% — a quarter of the budget. The most a sandwich can now extract before your swap reverts is roughly a quarter as much: instead of about \$180, the worst-case loss drops to around \$45. If the bot's profitable push would need more than 0.5%, your swap simply reverts (you pay a few dollars of gas and try again) and the bot gets nothing from you. By turning one number from 2% to 0.5%, you cut the maximum a sandwich can take by roughly 75% — from about \$180 down to about \$45. The single most powerful defense is the slippage field most people never touch.

## Frontrunning and backrunning: the other two flavors

The sandwich gets the attention, but it is really the union of two simpler moves. Understanding them separately matters because one of them (backrunning) is often *good* — and conflating "MEV" with "theft" leads to bad mental models.

### Frontrunning: racing to be first

Frontrunning is placing your transaction before someone else's to capture value that depends on going first. The classic non-sandwich case: a profitable trade appears in the mempool — say a wallet is about to buy a token right before a known catalyst, or a large arbitrage is sitting unexecuted. A bot copies the trade's logic and bids a higher priority fee to land *its* version first, capturing the gain the original trader was about to make. The original trader either gets a worse fill or misses the opportunity entirely.

This is the same mechanism that makes naive copy-trading dangerous: by the time a profitable wallet's trade is visible on-chain, faster bots have already front-run both the original wallet's follow-up and your manual attempt to mirror it. The lead time you think you have is mostly an illusion — see [The Perils of Copy-Trading On-Chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) for the full latency-gap argument.

#### Worked example: front-running a copy trade into a worse price

Say a wallet you follow buys a token, and you decide to mirror it with a \$10,000 buy. Their buy confirms and becomes visible on explorers at, say, second 12. A copy-bot watching that wallet fires a millisecond mirror, pushing the price up roughly 5% before second 13. You, a human, see the trade and click your \$10,000 order around second 40, by which point the price has drifted up about 9%. Your \$10,000 now buys roughly 9% fewer tokens than the wallet you copied got — about \$900 of value you handed to whoever was faster, before the token has even done anything. The front-run here is not a single bot stealing one slot; it is the entire field of faster participants pricing in the public signal before your manual order can land. The fix is the same family of moves: do not chase visible-on-chain signals with slow manual market orders, and when you must trade, route privately and size sanely.

### Backrunning: cleaning up after the trade

Backrunning places your transaction immediately *after* a target transaction, to act on the state it left behind. Two common, mostly-benign forms:

- **Arbitrage.** Your large swap moved the price in one pool out of line with the price on another exchange. A backrunning bot immediately trades the gap closed, profiting from the difference and — crucially — *restoring price consistency across venues*. This keeps DEX prices honest: arbitrage backrunning is the unglamorous plumbing that keeps the same token priced the same everywhere.
- **Liquidations.** A lending position falls below its collateral threshold. A backrunning bot races to liquidate it the instant it becomes eligible, earning the liquidation bonus. This is the protocol *working as designed*: bad debt is cleared promptly, protecting depositors.

Backrunning does not require pushing your price around first. It does not extract value *from you* — it acts on a public state change you created. That is why the on-chain community broadly considers arbitrage and liquidation MEV "good" or at least neutral: markets need someone to close gaps and clear bad debt, and MEV is the incentive that pays them to.

There is a tricky middle case worth naming so you can recognize it. **JIT (just-in-time) liquidity** is when a sophisticated liquidity provider sees your large swap pending, adds a huge amount of liquidity to the pool *in the same block just before your trade*, captures most of the trading fee your swap pays, and then removes the liquidity immediately after. It does not move your price against you the way a sandwich does — you may even get a *slightly better* fill because the pool is momentarily deeper — but it siphons the fee that would otherwise have gone to the pool's passive liquidity providers. Whether JIT is "toxic" is genuinely debated: the trader is not directly harmed, but the long-term LPs who provide the pool's baseline depth are. The point for you is taxonomy: not every bot transaction adjacent to yours is a sandwich, and learning to tell front-run, back-run, sandwich, and JIT apart is what separates an informed reader of the chain from someone who panics at every neighbor transaction.

![A four-by-three matrix comparing frontrun, backrun, and sandwich across where each sits, what the bot does, who pays, and whether it is harmful](/imgs/blogs/mev-sandwiches-and-frontrunning-4.png)

The matrix above is the clean mental separation: frontrun sits before you and steals your edge; backrun sits after you and (usually) just keeps markets efficient; the sandwich does both and exists only to tax your fill.

#### Worked example: a \$5M whale swap sandwiched for \$8,000

Sandwich profit scales with how far the bot can move the price, which scales with how big your trade is relative to the pool. Take a whale swapping \$5,000,000 into a mid-liquidity pool — a hundred times bigger than the earlier \$50,000 example. A trade that large has serious price impact, and the bot has correspondingly more room to work inside even a modest slippage tolerance.

If the bot can push the entry price up by about 0.16% before the whale's fill, it skims roughly 0.16% of \$5,000,000 ≈ \$8,000 of value across its bracketing buy and sell, net of fees. The whale's swap still "succeeds" and the loss is a small percentage — but \$8,000 from one click is a serious tax, and it is exactly why large on-chain traders almost never route big orders through the public mempool. The lesson scales both ways: the bigger and clumsier the order, the fatter the sandwich.

## How to spot a sandwich in your own transactions

This is the hands-on core: you can *prove* you were sandwiched, for free, in a few minutes. There are two approaches — reading the raw block yourself, and using a dedicated MEV explorer.

### Reading your own block in an explorer

Open your swap transaction in a block explorer (Etherscan on Ethereum, or the equivalent on whatever chain you traded on). Note two things: the **block number** and your transaction's **index** within that block (its position in the ordered list — index 42, say). Then open the full block and look at the transactions immediately around yours. The sandwich signature is a tight, specific pattern:

1. **A buy just above you.** The transaction at index 41 (right before yours) is a buy of the *same token* in the *same pool* you traded, from an address you do not recognize.
2. **A sell just below you.** The transaction at index 43 (right after yours) is a sell of that same token in that same pool, from an address.
3. **The same address on both legs.** Open the "from" address on the buy and on the sell — in a clean sandwich they are the *same wallet* (or two wallets funded from one source). That is the bot.
4. **Your fill near your cap.** Compare the price you actually got to the quote you were shown. If it landed close to your slippage limit rather than near the quoted price, that is the bot eating up your tolerance.
5. **A high priority fee on the bot legs.** The bot's buy usually carries an unusually large priority fee — its bid to win that exact ordering slot from the builder.

When all of these line up, you were sandwiched. The block-order bracketing is the fingerprint: a normal block has unrelated transactions around yours; a sandwiched block has *your own pool* being pumped right before you and dumped right after by one address.

Two refinements make you a sharper reader. First, sophisticated bots sometimes split the two legs across **two different addresses** funded from a common source, to make the bracket look less obvious — but the funding link is itself an on-chain clue (the clustering heuristic in [Address Clustering and Heuristics](/blog/trading/onchain/address-clustering-and-heuristics)), and the *timing and pool match* still give it away. Second, the sandwich's biggest constraint is that it must execute *atomically inside one block* — the bot only profits if its buy, your swap, and its sell all land in the right order in the same block. That is why it lives or dies on winning the ordering auction with a high priority fee, and why a private RPC (which keeps your swap out of the auction entirely) is so effective: there is no pending swap for the bot to wrap. The whole attack is a bet on block ordering, and ordering is exactly what you can take away from the bot.

![A graph showing one block with a bot buy at index 41, the user swap at index 42, a bot sell at index 43, both bot legs tracing to one address, leading to a sandwiched verdict](/imgs/blogs/mev-sandwiches-and-frontrunning-5.png)

#### Worked example: confirming a sandwich from the block view

You swapped at block index 42 and got 49,820 tokens for your \$50,000, when the quote said 50,000. Suspicious, you open the block. At index 41 sits a buy of the same token from `0xBOT…`, paying a priority fee far above the block's median. At index 43 sits a sell of the same token, also from `0xBOT…`. You compute the bot's edge: it bought around \$1.000 and sold around \$1.0036, on roughly the size it needed to move your fill — netting close to the \$180 of value you are missing. The three transactions are adjacent, two of them share one address, and your fill landed near your 2% cap. That is a confirmed sandwich, and you just read it straight off the public ledger — no special access required, because the chain shows everyone the same ordered truth.

### Using an MEV explorer

Reading raw blocks is educational but slow. Purpose-built MEV explorers do the pattern-matching for you. Tools in the style of EigenPhi and libMEV ingest blocks, detect the bracketing buy/sell pattern automatically, and label sandwiches, arbitrage, and liquidations with the extracted value attached. You paste your transaction hash or your wallet address, and the tool tells you whether that trade was sandwiched and by how much. Dune dashboards built by the community aggregate the same data at the protocol level — total MEV extracted per day, per DEX, per searcher. The workflow:

1. Paste your tx hash into the MEV explorer.
2. Read its verdict: sandwiched / arbitraged / clean, with the dollar value extracted.
3. If sandwiched, note the attacking address and the amount — then check whether you were using loose slippage or a public RPC, the two things that made you reachable.

The point of doing this is not to chase the bot — that is hopeless — but to *audit your own behavior*. If your trades keep showing up sandwiched, your slippage is too loose, your orders are too big for the pool, or you are broadcasting through the public mempool when you should not be.

## How to read it: a walkthrough on a single swap

Let me make this fully concrete with a complete, reproducible pass over one swap. Nothing here needs paid access; every step uses public explorers and free dashboards. Suppose you swapped a stablecoin for a token, the quote said 50,000 tokens, and your wallet received 49,820 — you want to know whether a bot ate the difference.

**Step 1 — Find your transaction and its block position.** Paste your wallet address into the explorer, click your swap, and read off two fields: the **block number** and the **transaction index** (its position in the block's ordered list, e.g. index 42). The index is the single most important number — a sandwich is defined by what sits at index 41 and index 43.

**Step 2 — Open the full block and scan the neighbors.** Click the block number to list every transaction in it, in order. Jump to your index and read the transaction directly above (41) and directly below (43). You are looking for the same token, the same pool, on both. The explorer's "Method" column helps: a `swap` or `swapExactTokensForTokens` call into your pool is the tell.

**Step 3 — Decode the two neighbor transactions.** Open the transaction at index 41 and read its Logs / Transfer events (the same receipt-reading covered in [Anatomy of a Transaction](/blog/trading/onchain/anatomy-of-a-transaction)). If it is a *buy* of your token from an address — say `0xBOT…` — note the amount. Open index 43: if it is a *sell* of the same token from the *same* `0xBOT…`, you have the bracket.

**Step 4 — Confirm the address identity and the priority fee.** Click the "from" address on both neighbor transactions. In a clean sandwich they are identical, or two addresses funded from one source (the funding link is a clustering heuristic covered in [Address Clustering and Heuristics](/blog/trading/onchain/address-clustering-and-heuristics)). Check the priority fee on the index-41 buy: an unusually large tip is the bot's bid to win that exact slot from the builder.

**Step 5 — Quantify the damage.** Compare your fill (49,820 tokens) to the quote (50,000). The 180-token gap at \$1.00 is about \$180. Cross-check by reading the bot's buy price and sell price from its two transactions: the spread it captured should be close to your missing \$180, minus its gas. That reconciliation — your loss ≈ the bot's gross profit — is the proof.

**Step 6 — Confirm with an MEV explorer and look at the macro picture.** Paste the same tx hash into an EigenPhi- or libMEV-style MEV explorer; it should independently label the bundle "sandwich" with a dollar figure. Then open a community Dune dashboard on DEX MEV to see the scale: how much sandwich value that DEX's users lose per day, which lets you judge whether your trade was an outlier or business-as-usual.

**Step 7 — Change your settings and re-trade.** The audit's whole purpose is the last step: if you were sandwiched, you were reachable. Tighten slippage, switch to a private RPC, or move size to an MEV-aware venue (all covered next), then re-run the same trade and confirm — by repeating Steps 1–6 — that the bracket is gone. That before/after on your own transactions is the most convincing lesson there is.

This walkthrough is the same skill the rest of the series builds on a different signal each time: the chain shows everyone the same ordered, timestamped truth, and reading it carefully turns an invisible tax into a measured, defensible number.

## Defending: slippage, private RPCs, and MEV-aware routers

You cannot make MEV go away — it is a structural feature of public, ordered ledgers. But you can make *yourself* a bad target. Defense is about shrinking the attack surface, and there are four levers, each closing a different door.

### 1. Tight slippage

As the worked examples showed, your slippage tolerance is the bot's working budget. The tighter you set it, the less a sandwich can extract before your swap reverts. For deep, liquid pairs (a major stablecoin pair, a blue-chip token in a deep pool), 0.1% to 0.5% is usually fine. The catch: set it too tight and ordinary, honest price movement will cause your swap to revert, costing you gas and a retry. The discipline is to use the *tightest tolerance the pool's real volatility allows* — not the lazy 2% default the interface pre-fills.

There is a subtlety worth knowing: some interfaces offer "auto slippage", which tries to pick a tolerance based on the pool. Auto is better than a flat 2%, but it is conservative and will still err loose to avoid reverts. For any trade you care about, set the number manually after glancing at the pool's depth — deep pool, tight number; thin pool, either accept the wider number consciously or, better, do not trade size there at all. The mental check is one sentence: *the slippage I set is the most a sandwich can take, so what loss am I authorizing?*

### 2. A private RPC (Flashbots Protect and friends)

The deepest fix is to never enter the public mempool at all. A **private RPC** (Remote Procedure Call endpoint) like **Flashbots Protect** routes your transaction directly to block builders through a private channel instead of broadcasting it to the public waiting room. If searcher bots never see your pending swap, they cannot bracket it — they cannot front-run what they cannot observe. Many wallets let you switch your RPC endpoint in a single settings field: you replace the default public node URL with the private RPC's URL, and from then on every transaction you sign is submitted privately by default.

Why this is the highest-leverage move: it does not just *shrink* the sandwich, it removes the precondition. A sandwich needs to see your pending transaction to position around it; a private RPC denies that visibility entirely, so the attack has nothing to grab. Some private RPCs go further and run an order-flow auction on your transaction, paying you a rebate if a backrun arbitrage profits from your trade — so you can actually *earn* a small refund from MEV instead of paying it. The catch: you are trusting the relay operator not to leak or misuse your transaction, and confirmation can be marginally slower because you are not bidding in the open auction. For ordinary swaps that tradeoff is overwhelmingly worth it. (Flashbots is the same infrastructure that organizes the builder market described in [Mining, Staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev) — here you are using it defensively.)

### 3. MEV-aware routers and order-flow auctions

Some venues are designed from the ground up to neutralize MEV. **CoW Swap** batches trades and settles them at a uniform clearing price through solvers, removing the intra-block ordering advantage a sandwich relies on. **MEV-blocker RPCs** route your order flow through an auction that either prevents the sandwich or *rebates* the MEV that would have been extracted back to you. The catch: there are fewer such venues, and batch settlement can add a short delay versus an instant AMM swap. But for size, the rebate and protection are real.

### 4. Avoid thin pools and split big orders

A sandwich's profit scales with the price impact of your trade, which scales with your size relative to the pool's depth. Two structural defenses follow: trade in **deep pools** where your order barely moves the price, and **split large orders** into smaller pieces (or use a router that does this for you) so no single fill offers a fat sandwich. The catch: low-liquidity pools are the trap — a big swap into a thin pool is the single juiciest target on the chain. This is also where *fake* depth bites: a pool that looks deep on a dashboard but is propped up by spoofed or wash liquidity will move far more than expected. For how that deception works, see [Fake Depth, Spoofing, and Oracle Attacks](/blog/trading/onchain/fake-depth-spoofing-and-oracle-attacks).

![A four-by-three matrix of sandwich defenses showing how each works, what it stops, and its catch](/imgs/blogs/mev-sandwiches-and-frontrunning-6.png)

The defenses matrix above is the layered view: slippage caps your worst case, a private RPC hides you entirely, an MEV-aware router refunds or blocks the attack, and pool choice removes the bait. Stack them and you go from default prey to a poor target.

#### Worked example: the same \$50,000 swap, defended

Take the earlier \$50,000 swap that bled about \$180 at 2% slippage through the public mempool. Now defend it three ways and add up the change. First, tighten slippage to 0.5%: worst-case loss falls from about \$180 to about \$45. Second, route through a private RPC so the bot never sees the pending swap at all: the sandwich opportunity simply disappears, and your expected MEV loss drops toward \$0 on that trade. Third, split a clumsy \$50,000 market order into smaller fills in a deep pool so price impact stays low. The arithmetic of defense: a trade that handed a bot \$180 for free now hands it roughly nothing, for the price of changing two settings. Over a year of active trading, the difference between defended and undefended swaps is the difference between paying the MEV tax and opting out of most of it.

## MEV as a measurable tax — and why some of it is fine

It is tempting to read all of this as "the chain is rigged against you." That is half right and half a trap. The honest picture has two parts: MEV is a real, measurable cost, *and* most of it is the price of a service the market actually needs.

### The cost is real and adds up

Extracted MEV on Ethereum runs into the billions of dollars cumulatively — roughly \$1.5 billion-plus since 2020 by common estimates, as of 2026 — split across arbitrage, liquidations, and sandwiches. For an individual trader, the sandwich slice is the part that lands on *you*: a fraction of a percent per swap, invisible per trade, but a steady leak across a year of activity. If you make a hundred \$10,000 swaps a year at loose slippage through the public mempool, even an average sandwich tax of 0.2% per swap is \$2,000 a year you handed to bots for nothing. The tax is small per bite and large per appetite.

The reason it stays invisible is psychological as much as technical. A trading fee shows up as a line item; a worse exchange rate does not. You see "swap successful, you received 49,820 tokens" and feel nothing was lost, because you never see the 50,000 you *would* have received in a clean block. The sandwich hides inside the fill price, which is exactly the number most traders never reconcile against the quote. That is why the audit walkthrough above matters more than the abstract billions: until you compare one of your own fills to its quote and read the bracketing transactions, the tax remains a statistic. Once you have done it once, you start setting slippage and routing the way someone who has seen the bill behaves.

![Illustrative grouped bar chart of worst-case sandwich loss in dollars at loose 2 percent versus tight 0.5 percent slippage across four swap sizes on a log scale](/imgs/blogs/mev-sandwiches-and-frontrunning-7.png)

> [!note]
> The bar chart above is an **illustrative teaching sketch**, not measured MEV data. There is no public, audited "sandwich loss by swap size" series, so the dollar figures are derived directly from the slippage worked examples in this post (a representative pool, constant-product price impact) to show the *shape* of the relationship: loss grows with swap size, and a tight cap shrinks the bill at every size. Treat the bars as the mechanism, not as a measurement.

#### Worked example: an illustrative year of the MEV tax

Suppose a DEX pool sees \$1,000,000,000 of swap volume over a year, and suppose — illustratively — that sandwichable order flow loses an average of 0.05% to toxic MEV (much trading is too small, too tight, or routed privately to be worth attacking, which pulls the average well below any single bad fill). That is 0.05% of \$1,000,000,000 = \$500,000 extracted from that one pool's users over the year. Scale that across thousands of pools and you see how cumulative MEV reaches the billions. Now flip it to your seat: if your share of that volume is \$2,000,000 of swaps and you trade undefended, your slice of the tax is roughly 0.05% × \$2,000,000 = \$1,000 a year — money that vanishes one invisible \$180 bite at a time. The figures here are illustrative, but the lesson is exact: the tax is real, it compounds with how much you trade, and almost all of it is avoidable with the four defenses above.

### Not all MEV is theft

The crucial nuance: the *sandwich* is toxic MEV — it extracts value from an unwilling user and gives nothing back. But arbitrage and liquidation MEV are the market doing necessary work. Arbitrage keeps the price of a token consistent across a dozen DEXs and CEXs; without bots racing to close gaps, on-chain prices would drift apart and be exploitable in worse ways. Liquidations clear undercollateralized loans before they become bad debt that depositors eat. MEV is the *incentive* that pays bots to do both. Eliminating all MEV would break the very mechanisms that keep DeFi prices honest and lending solvent. The goal is not to abolish MEV — it is to stop the toxic slice from landing on you.

### The second-order effect: who ends up controlling block order

There is a systemic worry beyond your individual fill, and it is worth understanding because it shapes the defenses available to you. Because MEV is so profitable, the parties who can extract it have an incentive to centralize. Building the single most profitable ordering of a block is a specialized, capital- and compute-intensive job, so a handful of professional **builders** now construct the large majority of Ethereum blocks. That concentration is the price the network pays for outsourcing MEV extraction to experts via the Flashbots-style auction.

This matters to you in two ways. First, **censorship risk**: if a few builders dominate, they can choose to exclude certain transactions (for example, ones interacting with sanctioned contracts), which is a different kind of "ordering power" than the sandwich but flows from the same control over block contents. Second, **the defense landscape depends on this market existing**: private RPCs and order-flow auctions work precisely *because* there is a builder market to submit to privately. The same infrastructure that industrialized the sandwich also gives you the private channel to dodge it. The honest framing is not "MEV good" or "MEV bad" — it is that transaction-ordering power is valuable, someone will always capture it, and your job as a trader is to make sure the toxic slice of that capture does not come from your trades.

### MEV is not just an Ethereum thing

The mechanics above are Ethereum-centric because that is where the tooling and DeFi are deepest, but the principle — whoever orders transactions can extract value — applies to every chain that has a public mempool and an AMM. The details differ enough to change your defenses, so it is worth a quick multi-chain tour.

On **Ethereum Layer-2 rollups** (Arbitrum, Optimism, Base and the like), ordering is usually decided by a single **sequencer** run by the rollup operator, often on a first-come-first-served basis. That structure happens to make classic mempool-sniping sandwiches harder, because there is no open public mempool to watch and no priority-fee auction to win — but it concentrates ordering power in the sequencer, which is its own trust assumption. Many L2s are exploring shared or decentralized sequencing precisely because of this.

On **Solana**, there is no public mempool in the Ethereum sense; transactions are forwarded to the current block leader. That does not eliminate MEV — sandwiching and front-running absolutely happen, especially in the frantic memecoin pools — but it shifts the game toward who has the fastest, most privileged path to the leader. The defensive instinct is the same: tight slippage, deep pools, and skepticism of thin memecoin liquidity, which is where the worst predation concentrates.

On **Bitcoin**, MEV is minimal by comparison — there is no rich AMM ecosystem to sandwich, and the UTXO model plus limited scripting leaves little ordering value to extract beyond simple fee-bidding. The lesson generalizes cleanly: **MEV scales with how much programmable, ordering-sensitive value sits in the mempool.** A chain full of deep DeFi has a lot; a chain that mostly moves coins has little. Wherever you trade, the question to ask is the same — *is my pending transaction visible, and can someone profit by ordering trades around it?* If yes, the four defenses apply.

## Common misconceptions

**"My swap confirmed, so I got a fair price."** No. A sandwich does not make your swap fail — it *relies* on your swap succeeding at a worse price. "It worked" tells you nothing about whether you were taxed. The only way to know is to compare your actual fill to the quote and to check the block order. A clean confirmation is fully compatible with having been sandwiched for \$180.

**"MEV is illegal or a hack."** No. A sandwich exploits no bug. It pays the network's normal fees to acquire a normal ordering position, using information the protocol broadcasts to everyone. It is unfair and predatory, but it is a feature of how public ordered ledgers work, not a breach of one. That is precisely why the fix is on *your* side — change your slippage and routing — not on the protocol prosecuting a criminal. (The legal picture is evolving — there have been cases where MEV operators who went further and *manipulated* a protocol's internal state were charged — but ordinary sandwiching of an honest swap is not, in itself, a hack, and you should not wait for the law to protect your fill.)

**"Slippage tolerance is just about volatility, not bots."** It is about both, and the bot part is the one people miss. Your tolerance protects you from honest price movement *and* simultaneously defines the maximum a sandwich can extract. Leaving it at 2% out of laziness is the equivalent of leaving \$180 on the table per large swap.

**"Big traders are safe because they have more money."** The opposite. Sandwich profit scales with price impact, and price impact scales with order size relative to pool depth. A \$5,000,000 swap through a public mempool into a mid-liquidity pool is one of the fattest targets on the chain — worth \$8,000 to a bot in the earlier example. Size makes you *more* attractive, not less, which is exactly why whales route privately and split orders.

**"All MEV bots are stealing from users."** No — most extracted MEV is arbitrage and liquidation, which keep prices consistent and lending solvent. Conflating the necessary plumbing with the toxic sandwich leads to bad decisions, like avoiding DeFi entirely. The precise target of your defense is *toxic* MEV (the sandwich and predatory frontrunning), not the arbitrage that keeps your pool priced correctly.

**"A private RPC slows me down too much to be worth it."** For the vast majority of swaps, the extra latency is a second or two and the protection is total against public-mempool sandwiches. Weigh it concretely: on a \$50,000 swap, the choice is between waiting perhaps a second longer and handing a bot up to \$180. The only time raw speed beats privacy is a genuine latency race (sniping a brand-new listing), and that is exactly the kind of trade where you are most likely to be the one getting front-run anyway. For ordinary trading, set the private RPC once and forget it.

## The playbook: what to do with it

Translate everything above into an if-then routine you run on every meaningful on-chain trade.

**Before you swap (set the surface):**
- **If** the pool is deep and the pair liquid → **set slippage to 0.1%–0.5%**, not the 2% default. **If** the swap reverts → nudge it up in small steps, not straight to 2%.
- **If** you trade more than occasionally → **switch your wallet's RPC to a private endpoint** (Flashbots Protect or an MEV-blocker RPC) once, and leave it. This is the highest-leverage one-time setting change.
- **If** the order is large relative to the pool → **split it** into smaller fills, or use a router that does, and prefer the deepest available pool. **If** the pool is thin → assume you are bait and either size down or use an MEV-aware venue.
- **If** the venue offers it → **route size through CoW Swap or an order-flow-auction router** that batches or rebates MEV.

**After you swap (audit yourself):**
- **The signal:** compare your actual fill to the quote. **The read:** a fill near your slippage cap, not near the quote, is a sandwich tell. **The action:** open the block and check the transactions at your index ±1.
- **The signal:** a same-pool buy just above you and same-pool sell just below you from one address. **The read:** confirmed sandwich. **The action:** for the next trade, tighten slippage and switch to a private RPC — you were reachable because you were not.
- **The signal:** an MEV explorer labels your tx "sandwiched, \$X extracted". **The read:** quantified tax. **The action:** treat \$X as the price you paid for loose settings, and change them.

**The invalidation / false-positive check:**
- A buy-then-sell pattern around your tx is *not* always a sandwich. **Arbitrage** (a backrun that closes a price gap on another venue) and **liquidations** are benign and will not consistently bracket *both* sides of your trade from one address. **JIT liquidity** adds and removes depth around your trade but does not move your price against you. The toxic-sandwich fingerprint is specifically: *same pool, same token, one address (or one funding source), one buy before + one sell after, your fill near your cap.* If the surrounding trades are in different pools, or only on one side, or from unrelated addresses, you were not sandwiched — you just traded in a busy block.
- Do not over-tighten into constant reverts. If a deep-pool swap keeps reverting at 0.1%, the pool's honest volatility needs a slightly looser cap; the goal is the tightest tolerance that *fills*, not the tightest number possible.
- Do not assume a private RPC makes you invincible. It removes the *public-mempool* sandwich, which is the common case, but you still want sane slippage and deep pools — and you are trusting the relay. Defense is layered, not a single switch.

**The size discipline (for anyone trading real money):**
- **If** a single order would be more than roughly 0.5%–1% of the pool's liquidity → split it into smaller fills or use a router/aggregator that does, so no one fill offers a fat sandwich and your average price impact stays low.
- **If** you are moving genuinely large size → default to an MEV-aware venue (CoW Swap, an order-flow-auction RPC) rather than a raw AMM swap, and treat the public mempool as off-limits for size. This is simply what professional on-chain desks do, and it is the difference between the \$8,000 whale-sandwich earlier and a clean fill.

The throughline of this series holds here too: the ledger is public, and that public ordering is exactly what lets bots tax your trade — and exactly what lets you read, prove, and defend against it. MEV is the price of trading in the open. With four settings, you stop paying most of it.

## Further reading & cross-links

- [Anatomy of a Transaction](/blog/trading/onchain/anatomy-of-a-transaction) — what a single transaction contains and the states it passes through before it is final.
- [DeFi Protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — how AMM pools set price, why your own trade moves it, and how lending positions get liquidated.
- [Address Clustering and Heuristics](/blog/trading/onchain/address-clustering-and-heuristics) — how to link the two legs of a sandwich to one operator even when the bot splits them across addresses.
- [The Perils of Copy-Trading On-Chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) — the latency gap and why front-running makes naive mirroring a losing game.
- [Fake Depth, Spoofing, and Oracle Attacks](/blog/trading/onchain/fake-depth-spoofing-and-oracle-attacks) — when the liquidity you trust is an illusion, and the thin-pool trap that fattens a sandwich.
- [Mining, Staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev) — the market-microstructure view of who builds blocks and how the MEV supply chain (searchers, builders, validators, Flashbots) actually works.
