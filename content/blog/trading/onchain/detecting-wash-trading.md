---
title: "Detecting Wash Trading: Spotting Fake Volume On-Chain"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Wash trading is trading with yourself to fake volume and demand. Learn how on-chain wash trading betrays itself — self-trade loops, circular fund flows, common funders, volume with no price discovery, and the economic tell that wash trading costs more in fees than it can ever profit — and what to do about it as a trader and an analyst."
tags: ["onchain", "crypto", "wash-trading", "fake-volume", "dex", "nft", "ethereum", "solana", "dune", "market-manipulation"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Wash trading is trading with yourself — running both sides of a trade through wallets you control — to fake volume and manufacture the *appearance* of demand. On a public blockchain the counterparties are visible and usually linked, so wash trading is one of the most detectable forms of manipulation there is.
>
> - **What it is:** fake volume. An operator buys from themselves and sells to themselves, printing trades that move money in a circle and create no real demand. They do it to win exchange listings, climb a ranking, trigger FOMO, or farm trading rewards and airdrops.
> - **How to read it:** look for the patterns the chain can't hide — a wallet on *both* sides of a swap, a token that round-trips A to B and back to A, counterparties that all trace to **one funding wallet**, volume with no price discovery, and the economic tell: each wash trade burns gas and LP fees, so sustained wash trading is a *funded loss* — which means there is always an ulterior motive.
> - **What you do with it:** as a trader, never buy fake demand — discount washed volume to zero and check unique buyers before you trust a chart. As an analyst, cluster the counterparties, find the funder, and report the inflated number for what it is.
> - **The number to remember:** a token faking **\$5M of daily volume** can cost roughly **\$15,000 a day** in LP fees and gas. Nobody burns that for fun — find out who is paying, and why.

On 2022-01-19, the NFT marketplace LooksRare launched with a token and a simple promise: trade NFTs here and earn LOOKS rewards proportional to your trading volume. Within days, the platform was reporting *more daily volume than OpenSea* — hundreds of millions of dollars a day. It looked like a giant-killer had arrived. It hadn't. A large share of that "volume" was wallets selling expensive NFTs back and forth to themselves, paying the marketplace fee on each fake sale, and collecting LOOKS rewards worth far more than the fee. The volume was real in the sense that the transactions happened on-chain. It was fake in every sense that mattered: there was no buyer, no seller, no price discovery, no demand. It was one person paying themselves to harvest a reward, and the chain recorded every step of it.

That episode is the whole subject of this post in miniature. **Wash trading** is trading with yourself to fake activity, and it is everywhere in crypto — on decentralized exchanges (DEXes), on NFT markets, and, by inference, on centralized exchanges (CEXes). It exists because volume is a currency of its own: it gets you listed, ranked, trending, and trusted. And it persists because most people read a volume number and stop there. But here is the gift the blockchain gives you that no traditional market ever did: the counterparties are *public*. In a dark pool you can never know whether the other side of a trade was the seller's own desk. On-chain, you can follow the money, see that the "buyer" was funded by the "seller", and prove the volume was theater.

This post builds the whole picture from zero — what wash trading is, why people do it, why it is uniquely detectable on a public ledger, and then the concrete detection toolkit: the patterns, the tools, the economic logic, and the playbook for a trader who refuses to buy fake demand and an analyst who wants to expose it.

![One operator funds two wallets that trade the same token through a pool to inflate reported volume while real demand stays flat](/imgs/blogs/detecting-wash-trading-1.png)

## Foundations: what wash trading is, and why the chain betrays it

Start with the plainest possible definition. A **trade** is an exchange between two different parties: I have an asset, you have money, we swap, and a price is set by our disagreement about what it's worth. **Wash trading** removes the "two different parties" — one party plays both roles. You sell an asset from your left hand to your right hand at a price you choose, and the exchange records a trade and adds it to the day's volume. No ownership really changed (you still control both wallets), no price was discovered (you set both sides), and no demand existed (nobody who wasn't you wanted the asset). The trade is a costume; the volume number is the only thing the audience sees.

Here is an everyday version of the same trick. Say you run a small café and you want it to look busy so passers-by come in. You hand your friend \$50, they buy fifty coffees, hand you back the cups, and you hand them their \$50 again. To anyone watching the till, you sold fifty coffees — your "sales volume" is up. But you sold nothing: the same \$50 went out and came back, the coffees never left, and the only real change is that you paid for fifty cups, fifty lids, and the electricity to make them. That waste *is* the wash. The café looks popular; the books show a loss you chose to take to manufacture the look. Replace the café with a token, the friend with your second wallet, and the wasted cups with gas and LP fees, and you have wash trading exactly. The waste-for-appearance trade is the whole game, and the waste is the evidence.

A few prerequisite terms, defined from zero, because the rest of the post leans on them:

- **EOA (externally owned account):** a normal wallet controlled by a private key — what you'd call "an address." One person can control thousands of EOAs. There is no identity check to open one; the chain doesn't know that 500 addresses are one human.
- **DEX (decentralized exchange):** a smart contract that lets anyone swap one token for another against a shared pool of the two tokens (a **liquidity pool**, or LP). Uniswap is the canonical example. Every swap is a public transaction; there is no hidden order book.
- **LP fee:** the small cut (often 0.30%, sometimes 0.05% or 1%) that a DEX charges on every swap and pays to the people who supplied the pool's liquidity. This fee is the reason wash trading on a DEX is *not* free — every fake swap leaks money to the liquidity providers.
- **Gas:** the fee you pay the network to process any transaction. Even a pointless self-trade costs gas. On a busy chain a single swap can cost a few dollars to tens of dollars; on a cheap chain, cents. Either way it is a non-zero cost per wash.
- **NFT (non-fungible token):** a unique on-chain token representing one specific item (a piece of art, a collectible). NFT markets report a "floor price" — the cheapest one currently for sale — and recent sale prices. Both are easy to fake with self-sales.
- **Volume vs demand:** *volume* is the total value of trades over a period. *Demand* is the existence of people who genuinely want to own the asset more than they want their money. Wash trading manufactures volume without manufacturing demand. The entire detection problem is learning to tell the two apart.

### Why people wash trade

Nobody pays fees to trade with themselves for no reason. The motive is always to convert fake volume into something valuable:

- **Exchange listings.** Many exchanges (and aggregators, and "trending" lists) rank or admit tokens by volume. A project that washes its way to a big number can earn a listing that brings real liquidity and real buyers — at which point the fake volume has paid for itself many times over.
- **Rankings and FOMO.** Humans chase activity. A token sitting at the top of a "most traded" board, or an NFT collection with a soaring floor and brisk "sales," pulls in real buyers who assume the crowd knows something. Wash trading is a way to *buy* the appearance of a crowd.
- **Incentive and reward farming.** This is the LooksRare case: when a platform pays you in its own token for trading, you can wash-trade to harvest rewards that exceed the fees you pay. The volume is a byproduct of farming the reward.
- **Airdrop and points farming.** Many protocols award future tokens (an **airdrop**) based on past activity, sometimes counting volume. Wash trading inflates your apparent activity to maximize the airdrop — which overlaps directly with sybil farming (one operator, many wallets). We'll come back to this.

### Why on-chain wash trading is detectable

The crucial property is that public blockchains are **pseudonymous, not anonymous**. An address is not a name — it doesn't say who you are. But it is a *persistent identifier* that records everything that address has ever done, forever, in public. So while the operator's legal identity stays hidden, their *behavior* is fully exposed: which wallets they funded, when, in what amounts, trading with whom. Wash trading relies on the audience treating two addresses as two independent people. Pseudonymity defeats that, because the chain lets you prove the two addresses are linked even without knowing the human behind them. You don't need to know the washer's name to know the volume is fake — you only need to show the counterparties are one entity.

In a traditional market, the exchange sees the order book and you don't. If a market maker fills both sides of a trade through two of its own accounts, you have no way to know — the accounts are private, the matching is internal. Regulators catch it only with subpoenas and forensic accounting.

On a public blockchain, the asymmetry flips. *Every transaction is public, permanent, and pseudonymous.* You can see that Wallet A sent the token to Wallet B, that Wallet B sent it back, that both wallets were first funded with gas by the same third wallet, and that none of the money ever came from outside the little circle. The operator is hiding in plain sight, and the only thing protecting them is that most observers never click past the volume number. That is the edge this post is about: the willingness to follow the counterparties instead of trusting the total.

The four things the chain gives you that a private order book never could are worth stating explicitly, because every detection technique in this post is built on one of them:

- **Counterparty visibility.** Every swap names its sender. You can see *who* traded, not just that a trade happened. A wash needs the counterparties hidden; the chain refuses to hide them.
- **Funding history.** Every wallet's first transaction is permanent. You can walk backward from any "trader" to the wallet that gave it its first gas, and from there to the operator. There is no statute of limitations on a public ledger.
- **Net-flow accounting.** Because every transfer is recorded, you can compute whether a wallet *accumulated* a token over a period or merely oscillated. Real demand accumulates; wash trading nets to zero. The ledger makes net flow computable for anyone.
- **Cost transparency.** Gas and LP fees are themselves on-chain. You can measure exactly how much a wash operation is bleeding, which — as we'll see — is the single most robust signal of all.

None of this requires a subpoena, a regulator, or insider access. It requires a block explorer, a free Dune account, and the discipline to ask "who, funded by whom, accumulating what, at what cost?" instead of reading a green number and moving on.

### Volume became a currency, so volume got faked

One more piece of foundation, because it explains *why* wash trading is so prevalent rather than a fringe trick. Over the last decade, "volume" stopped being a neutral statistic and became a thing of value in itself — a currency you could spend. Ranking sites sort tokens and exchanges by volume. Aggregators route trades to the venue with the deepest *apparent* liquidity. Listing committees use volume as a proxy for legitimacy. Reward programs pay out in proportion to volume. Airdrop formulas count volume as activity. Every one of those is a place where a bigger volume number converts directly into money, attention, or access.

The economic principle is old and ruthless: **any metric that becomes a target stops being a good measure** (Goodhart's law). The moment volume became the thing that gets you listed, ranked, and rewarded, volume became the thing worth faking — and wash trading is simply the cheapest way to fake it. So you should expect washed volume wherever a volume number pays off, and you should expect it to be most aggressive on exactly the platforms that advertise volume-based rewards. Detection is not paranoia; it is reading the incentives correctly.

## How wash trading looks on a DEX

The simplest DEX wash is a single operator trading a token they control, through a pool they (largely) control, back and forth. Picture the mechanics: the operator holds most of a token's supply and also supplies most of the pool's liquidity. They swap token-for-stablecoin in one transaction (a "sell"), then stablecoin-for-token in the next (a "buy"). Each swap is recorded as volume. Because they own both the token side and the liquidity side, the stablecoin they "spend" buying mostly flows back to them as the liquidity provider. The net economic effect is small — they pay the LP fee and gas — but the *reported volume* climbs by the full notional of each swap.

There are three recurring DEX wash patterns, and they escalate in sophistication:

1. **Self-trade loop (one wallet, both sides).** The crudest form: the same wallet appears as both the swapper and (via its LP position) the counterparty, churning the token through its own pool. This is trivially detectable because one address is on both sides of the economic flow.
2. **Circular A→B→A flow.** Slightly less obvious: the operator uses two wallets. Wallet A sells to Wallet B; later Wallet B sells back to Wallet A. The token ends where it started; only the volume tape grew.
3. **Self-funded counterparties (a ring of wallets).** The mature form: a handful of wallets trade among themselves in a closed loop, so no single wallet is obviously on both sides. The tell is no longer the trading pattern alone — it's that every wallet in the ring traces back to one funding source.

### The circular A→B→A flow

The circular flow is the pattern to internalize first, because it shows the core deceit so cleanly. Follow one token through it.

![A token sold from wallet A to wallet B and back to wallet A leaves the operator holding the same coins minus fees and gas](/imgs/blogs/detecting-wash-trading-2.png)

Wallet A starts holding 1,000,000 of a token worth \$0.05 each — a \$50,000 position. It sells the whole lot to Wallet B, which prints \$50,000 of volume on the tape. A few minutes or hours later, Wallet B sells the same 1,000,000 tokens back to Wallet A, printing *another* \$50,000 of volume. Now look at the end state: Wallet A again holds 1,000,000 tokens, exactly as before. The only thing that changed is that the operator (who controls both A and B) is now poorer by two rounds of LP fees and gas — and the public volume figure for the token shows \$100,000 of "activity" that represents zero change in ownership and zero new demand.

#### Worked example: a token faking \$5M of daily volume

Take a small token and an operator who wants it to *look* like it trades \$5,000,000 a day so it qualifies for a listing on a mid-tier exchange. They round-trip the token through their own pool. On a DEX charging a 0.30% LP fee, \$5,000,000 of swapped notional leaks \$5,000,000 × 0.0030 = \$15,000 a day to liquidity providers. If the operator supplies, say, two-thirds of the pool's liquidity, they get most of that fee back — but the third they don't own is a genuine loss to outside LPs, plus they pay gas. Call the all-in daily bleed \$15,000 in LP fees that's only partly recycled, plus roughly \$1,200 in gas across a few hundred swaps — order \$10,000–\$16,000 a day of real, recurring cost.

Now ask the only question that matters: **who pays \$15,000 a day, and why?** Not a trader hoping to profit on the round-trips — the round-trips are guaranteed losers. The only rational payer is someone for whom the *appearance* of \$5M volume is worth more than \$15k/day: a project trying to earn a listing, pump a chart, or qualify for a reward. The cost itself is the confession. Sustained, expensive volume that produces no price discovery is volume someone is funding for a reason — and that reason is never good for the late buyer.

![Wash trading burns LP fees and gas every day while a real trade of the same size profits nothing](/imgs/blogs/detecting-wash-trading-4.png)

### Why washing through your own pool leaks money — the AMM mechanics

It helps to understand *why* the loss is unavoidable, because the mechanism is what makes the cost a reliable signal. A DEX pool is an **automated market maker** (AMM): it holds a reserve of two tokens and quotes a price from their ratio, charging a fee on every swap. When you swap in one direction, you push the price slightly against yourself (that's **slippage** — the larger your swap relative to the pool, the worse the price you get), and you pay the LP fee. When you immediately swap back, you push the price the other way and pay the fee *again*. You end up roughly where you started in token terms, but you've paid two fees and eaten slippage twice.

If the washer also *owns* the liquidity, they recapture the fee on the share of the pool they supplied — but never the slippage they paid to outside liquidity, and never the gas. The cleaner the wash looks (more swaps, more notional), the more it costs. There is no configuration of an honest AMM in which round-tripping is free; the protocol is built to charge for every swap precisely so that liquidity providers get paid. That structural cost is what converts "is this volume fake?" into "who is paying to keep this volume up, and why?" — a question with a traceable answer.

#### Worked example: \$2M of self-volume on a thin pool, the slippage bite

Take an operator round-tripping \$2,000,000 of notional through a pool that holds only \$500,000 of liquidity — a thin pool, common for small tokens. Because each swap is large relative to the pool, slippage is severe: a swap that moves \$100,000 against a \$500,000 pool might cost 2–4% in price impact. Across \$2,000,000 of round-tripped volume, even a blended 1.5% slippage on the legs that hit *outside* liquidity is \$2,000,000 × 0.015 = \$30,000, on top of \$2,000,000 × 0.003 = \$6,000 in LP fees and gas. The washer is bleeding tens of thousands of dollars to make a thin token look like it trades \$2M. The thinner the real liquidity, the more obvious the wash — because the slippage cost balloons, and nobody pays that to lose money for sport. The lesson: heavy volume on thin liquidity is a wash red flag, not a bullish one.

### Spotting a self-trade loop: a walkthrough

Here is the concrete, click-by-click pass an analyst makes to confirm a DEX wash, using public tools. Suppose a token is trending on a screener with suspiciously round, suspiciously constant daily volume.

1. **Pull the token's swap history.** On a block explorer (Etherscan for Ethereum, Solscan for Solana, BscScan for BNB Chain) open the token contract and view its DEX transactions, or query the swaps directly on **Dune** with a few lines of SQL. You want, for each swap: the sender address, the direction (buy/sell), the size, and the timestamp.

2. **Group swaps by counterparty.** Tally how many swaps came from each address. Genuine volume is spread across many addresses, each trading a few times. Washed volume concentrates: a handful of addresses account for the overwhelming majority of swaps. If 5 addresses produce 90% of the volume, that's your first flag.

3. **Check for both-sides behavior.** For the top addresses, look at whether each one both *buys* and *sells* the token repeatedly, in roughly equal amounts, with no net accumulation. A real buyer accumulates; a real seller distributes. A washer oscillates — buy, sell, buy, sell — because the goal is the volume, not a position.

4. **Trace the funding.** Click each suspect address's earliest transactions. Where did its first ETH/SOL (for gas) and its first stablecoins come from? If several of the "different" counterparties were all initially funded by the same wallet, you've found the operator. This is the step that turns suspicion into proof, and it's the subject of [address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics).

5. **Confirm the absence of price discovery.** Overlay the volume on the price. If \$5M changed hands but the price barely moved all day, no real supply/demand imbalance existed — exactly what you'd expect from trades that net to zero.

A Dune query for step 2 is short enough to read at a glance:

```sql
-- swaps for one token over 24h, counted by trader address
select
    trader,
    count(*)                       as swap_count,
    sum(amount_usd)                as volume_usd
from dex.trades
where token_address = 0xTOKEN     -- the suspect token
  and block_time > now() - interval '24' hour
group by trader
order by volume_usd desc
limit 20;
```

If the top 10 traders are 80–95% of `volume_usd`, and the same addresses appear with near-equal buy and sell counts, you are almost certainly looking at a wash operation rather than a market.

It is just as important to know what *genuine* demand looks like, so you don't cry wolf on a real token. Organic volume has a recognizable shape: it is spread across *many* distinct addresses (hundreds or thousands, with a long tail of one-off buyers), the buyers *net accumulate* over time rather than oscillating, the funding sources are *diverse and external* (money arriving from many independent wallets, exchanges, and bridges, not one faucet), and the price *moves* as the order flow leans one way — that is price discovery doing its job. When you pull a token's swaps and see a broad address distribution, net accumulation, external funding, and a price that responds to flow, you are looking at a market, and the volume number means what it says. Detection is not assuming everything is fake; it is being able to tell the two apart, and the contrast is usually stark once you look past the headline total.

## NFT wash trading

NFT markets are even more exposed to wash trading than fungible-token markets, for a structural reason: an NFT is *unique*, so there is no order book and no continuous price — the "price" of a collection is whatever the last few sales were, and the floor is whatever the cheapest listing is. Both are trivially set by selling to yourself.

The classic NFT wash is a self-sale at an inflated price. You own an NFT. You list it and "buy" it with your *second* wallet for \$200,000 — far above any honest bid. The marketplace now records a \$200,000 sale, the collection's average price jumps, and the chart looks like a hot collection with deep-pocketed buyers. Real buyers, seeing "recent sale: \$200,000," anchor on that number and bid up. But no outside money entered: the \$200,000 went from one of your wallets to the other, and you paid the marketplace fee and gas for the privilege.

![A real NFT market with a honest floor and many distinct buyers versus a wash sale that fakes a high price between two wallets of one owner](/imgs/blogs/detecting-wash-trading-3.png)

#### Worked example: a \$200,000 NFT self-sale to fake a floor

An operator owns an NFT from a thin collection whose honest floor is \$2,000 with a best outside bid of \$1,800. They want the collection to look like it trades at five figures so they can offload their other twelve pieces into the FOMO. From Wallet A they list the NFT and "buy" it with Wallet B for \$200,000 in ETH. The on-chain record now reads: "sold for \$200,000." If the marketplace fee is 2%, the round-trip cost them \$200,000 × 0.02 = \$4,000 in fees (paid to the marketplace), plus gas — money moved from their right pocket to their left, minus the \$4,000 leak.

The detection is almost embarrassingly simple. Open the sale on the marketplace or a block explorer and answer three questions. First: is the buyer's address linked to the seller's? Pull both wallets' funding history — if they were both seeded by the same address, it's a self-sale. Second: did any money enter the collection from *outside*, or did the \$200,000 just circle between two wallets? Third: what is the *real* best bid from an unrelated wallet? Here it's still \$1,800. A \$200,000 "sale" sitting above an \$1,800 honest bid, between two linked wallets, with no outside money, is not a price — it's a prop. The intuition: a price set by the seller paying themselves carries no information at all.

### The LooksRare and Blur incentive-wash episodes

The most instructive NFT wash episodes weren't lone operators faking a floor — they were *whole marketplaces* whose reward designs paid people to wash trade. When [airdrop farming and sybil cohorts](/blog/trading/onchain/airdrop-farming-and-sybil-cohorts) meet an NFT marketplace, you get industrial-scale wash trading.

LooksRare (launched January 2022) paid LOOKS tokens to traders in proportion to their fee-generating volume. The unintended consequence: if the LOOKS you earned for a trade were worth more than the fee you paid, you should trade as much as possible — with yourself. Users bought and sold high-value NFTs between their own wallets, paying the 2% fee, and collected LOOKS rewards that, for a stretch, exceeded the fee. Reported volume rocketed past OpenSea, but a large fraction was this reward-driven self-trading. The "volume" was a farming receipt, not demand.

The detection signature on LooksRare was loud. The wash trading concentrated in a small set of high-value collections traded between linked wallets at prices far above their honest floors, with the same wallets appearing as both buyer and seller across many "sales," and the proceeds and rewards recombining at a few collection addresses. The "average sale price" of some collections detached completely from any honest bid — the canonical sign that the price was being set by sellers paying themselves. Anyone who pulled the per-collection sale history and grouped by counterparty could see that a large slice of the record-breaking volume was a handful of operators farming LOOKS, not a marketplace overtaking OpenSea.

Blur (launched February 2023) ran a token-incentive program tied to *bidding* and trading activity to win NFT liquidity from OpenSea. It largely succeeded at pulling real activity — but it also created strong incentives to bid and trade aggressively to maximize the airdrop, producing a wave of wash-like and reward-optimized activity that inflated headline numbers. Bidding-based rewards in particular produced a strange equilibrium where placing and cycling bids was the rational move regardless of whether you wanted the NFT, because the bids themselves earned points toward the token. The lesson generalizes: **whenever a platform pays for volume or activity, some of the volume it reports is the payment farming itself.** Read any incentivized marketplace's headline numbers with that discount baked in, and ask what fraction of the activity would survive if the rewards stopped tomorrow.

#### Worked example: incentive-wash farming \$50,000 of rewards on \$2M of self-volume

A farmer on a reward-paying marketplace churns \$2,000,000 of NFT self-sales over a campaign — buying and selling their own pieces between two wallets. At a 2% fee, that costs \$2,000,000 × 0.02 = \$40,000 in marketplace fees, plus gas. If the platform's volume-based rewards pay this farmer \$50,000 worth of its token for that activity, the trade nets +\$50,000 − \$40,000 = +\$10,000, before gas. That positive expected value is *exactly* why the wash happens: the reward subsidizes the loss. For an analyst, the math runs in reverse — find the activity whose only profit is the reward, and you've found the wash. For a trader eyeing that token, recognize that \$2M of "volume" represents zero NFT demand and a \$50,000 future-sell-pressure overhang as the farmer dumps their rewards.

## Wash trading across chains: Solana memecoins and Bitcoin

Wash trading is not an Ethereum phenomenon — it follows the incentives onto every chain, and the *shape* of the wash adapts to the chain's mechanics.

**Solana** is the current epicenter of memecoin wash trading, for two reasons. First, fees are tiny — a swap can cost a fraction of a cent — so the gas component of the wash cost nearly vanishes, and a washer can churn thousands of self-trades for the cost of a coffee. Second, the memecoin culture lives and dies on *attention*: a token trending on a screener with brisk volume and a climbing chart pulls in real buyers within minutes. Launchpads have made it trivial to mint a token and a pool in seconds, and a fresh memecoin with no real users can be made to look like a rocket by a single operator round-tripping it through its own pool. The detection patterns are identical — concentration in a few wallets, both-sides oscillation, a common funder, flat-ish price under heavy volume — but you run them on Solscan and Solana-focused Dune dashboards instead of Etherscan. The survivorship math is brutal: of the millions of tokens launched on Solana launchpads, only on the order of one to two percent ever reach a meaningful market cap, and a large share of the rest show exactly this manufactured-volume signature before they go to zero.

#### Worked example: a memecoin faking \$300k of volume for \$200 in fees

A Solana memecoin operator wants their freshly minted token to trend, so they round-trip \$300,000 of self-volume through its pool over a few hours. On a chain where each swap costs well under a cent and the launchpad pool charges a modest swap fee, the all-in cost might be on the order of \$200 — perhaps \$300,000 × 0.0006 in pool fees plus trivial gas. For \$200, the token now shows \$300,000 of "volume," climbs the trending list, and lures real buyers who push the price up — at which point the operator (who holds most of the supply) sells into them. The cheapness of the chain is exactly what makes the wash so prevalent: when faking \$300,000 of demand costs \$200, the only thing stopping it is whether a real buyer shows up to be the exit. Read trending memecoins with the assumption that the early volume is the operator, not the market.

**Bitcoin** is different, because it has no native tokens or DEX in the Ethereum sense — but the **UTXO model** (Bitcoin tracks coins as discrete unspent outputs rather than account balances) makes a *different* kind of fake-activity trace possible and detectable. A common deceptive pattern is generating the appearance of network activity or volume by shuffling coins between one's own addresses: a wallet sends coins to a fresh address it controls, which sends them onward, inflating transaction counts and "transferred value" without any economic exchange. Because each UTXO is traceable to its parent, an analyst can follow the chain of spends and see that the coins never left the operator's control — the "activity" is one entity moving its own money in a circle. The signal generalizes: *value transferred* and *transaction count* are as fakeable as *volume*, and the UTXO graph is what lets you prove the transfers were self-directed. (The mechanics of the model are covered in [how blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account).)

The cross-chain takeaway: the *unit* being faked changes (DEX volume, NFT sale price, memecoin trending volume, Bitcoin transferred value), but the detection discipline is constant — find the counterparties, trace the funding, measure net flow, and price in the cost.

## The economic tell: wash trading is a funded loss

Step back from the patterns to the deepest, most general signal — the one that works even when the trading pattern is cleverly disguised. **Every wash trade costs money.** It burns gas, and on a DEX it leaks LP fees; on a marketplace it leaks the trading fee. A wash trade can never *profit* on its own — by construction it returns the operator to where they started, minus costs. So sustained wash trading is a *funded, recurring loss*. And a rational actor does not fund a recurring loss for nothing.

A subtle but important point: wash trading and *price* manipulation are related but distinct, and reading the price tells you which game is being played. Pure wash trading — round-tripping to fake volume — tends to leave the price roughly *flat*, because the buys and sells offset; the operator wants the *activity* to look real, not necessarily to move the price. A washer who *also* wants to move the price will skew the trades (more buying than selling) and spend more, accepting a larger loss to push the chart up before an exit. So the price pattern is itself a read: heavy volume with a flat price says "faking activity for a listing or ranking"; heavy volume with a steady grind upward on thin real liquidity says "faking activity *and* steering the price toward an exit." Either way, the volume is not demand — it is the operator — and the price tells you what they're trying to achieve with it.

This converts wash detection into a follow-the-money question with only a few possible answers. If you observe volume that:

- produces no price discovery (price flat, or moving only when the operator wants it to),
- concentrates in a few linked wallets,
- and costs real money in fees and gas every day,

then someone is paying to maintain an illusion, and the analyst's job is to find the payoff that makes the loss worth it: a pending listing, a reward program, an airdrop snapshot, an exit they're setting up. The cost is the smoke; the payoff is the fire.

This is also where **MEV** (maximal extractable value — value bots extract by reordering or inserting transactions) intersects wash trading, and it cuts *against* the washer. A washer round-tripping a token through a pool is a sitting duck for sandwich bots, which see the predictable swaps in the public mempool and trade around them, skimming a slice of each wash. So the washer's real cost is often *higher* than the headline fee, because part of every fake swap is being taxed by MEV bots. The deeper the wash, the more it bleeds — which reinforces the tell: only a strong ulterior motive justifies that bleed.

There is a clean quantitative screen that operationalizes the cost tell without any clustering at all: the **volume-to-liquidity ratio**. Divide a token's reported 24-hour volume by its pool's total liquidity. A healthy, organically traded token typically turns over its liquidity a modest number of times a day — a ratio in the low single digits is normal. A washed token shows an absurd ratio, because the operator is round-tripping far more notional than the pool could honestly support: \$5,000,000 of "volume" against \$200,000 of liquidity is a ratio of 25×, which means the pool's entire depth would have to change hands twenty-five times in a day from genuine demand — implausible for a token nobody has heard of. A sky-high volume-to-liquidity ratio doesn't prove a wash by itself, but it is the fastest first-pass filter there is, and it flags exactly the thin-liquidity tokens where washing is cheapest to detect. (The mechanics of reading pool depth are covered across this series' DeFi material.)

#### Worked example: a 25× volume-to-liquidity ratio flags the wash in one division

A screener shows a token with \$5,000,000 of 24-hour volume — impressive for its size. Before reading anything else, pull its pool liquidity: \$200,000. The volume-to-liquidity ratio is \$5,000,000 / \$200,000 = 25. For genuine demand to produce that, the pool's entire \$200,000 of depth would need to fully turn over twenty-five times in a day, with real buyers and sellers, on a token with no narrative and no holders — implausible. Compare a real mid-cap token: \$5,000,000 of volume on \$3,000,000 of liquidity is a ratio of about 1.7×, entirely normal. The single division — volume over liquidity — separated the prop from the market before you traced a single wallet. When the ratio is in the double digits on an obscure token, treat the volume as fake until clustering proves otherwise.

#### Worked example: the \$5M-volume token with only 12 unique buyers

Return to the \$5,000,000-a-day token, and run the single most powerful sanity check there is: **volume divided by unique buyers.** Pull the day's swaps and count *distinct buyer addresses*. A genuinely traded token doing \$5M/day might have 400–800 distinct buyers — call it 640 — averaging roughly \$5,000,000 / 640 ≈ \$7,800 of buying per address, spread across hundreds of independent decisions. The washed token shows \$5,000,000 of volume across just **12 distinct addresses**, averaging \$5,000,000 / 12 ≈ \$417,000 each, oscillating buy-sell-buy-sell with no net accumulation, all funded from one source. Same headline number; completely different reality. The unique-buyer count is the number the volume figure hides, and dividing one by the other is the fastest wash screen you can run.

![Two tokens report the same five million dollars of daily volume but the genuine token has hundreds of buyers and the washed token has twelve](/imgs/blogs/detecting-wash-trading-5.png)

## Confirming it: clustering the counterparties to one funder

Patterns raise suspicion; **clustering** turns suspicion into proof. The defining property of a wash ring is that its "independent" counterparties are not independent — they trace back to one operator. The most reliable link is the **common funding source**: on most chains, a brand-new wallet needs gas before it can do anything, and that first gas almost always comes from somewhere traceable. If Wallets A, B, C, and D all received their first ETH from the same wallet 0xF00…, and then those four wallets only ever trade with each other, you are not looking at four traders — you are looking at one operator wearing four masks.

![Every wallet in a wash ring traces its first gas deposit back to one common funding wallet](/imgs/blogs/detecting-wash-trading-6.png)

This is the same machinery used to unmask sybil farms, and the heuristics are worth naming because each carries a false-positive risk:

- **Common funder.** All counterparties seeded by one wallet. Strong signal — but be careful: centralized-exchange withdrawal hot wallets fund millions of *unrelated* users, so "funded by Binance" links nothing. The signal is a *small, private* funder seeding a *closed* set of wallets, not a giant exchange faucet.
- **Closed trading graph.** The wallets trade overwhelmingly with each other and rarely with the outside world. Real markets are open graphs; wash rings are closed loops.
- **Behavioral fingerprint.** Identical swap sizes, identical timing cadence, identical gas settings, sequential nonce patterns — the wallets *act* like one script because they are one script.
- **Fund recombination.** After the campaign, the wallets' proceeds drain back to one or two collection addresses. Money fanned out from one source and later fanned back in is a near-certain single-operator tell.

No single heuristic is proof — a real trader can happen to be funded by a friend; two strangers can trade with each other. The discipline is to **stack three or four independent signals** before you call it. One link is a coincidence; a common funder plus a closed graph plus identical behavior plus recombination is a fingerprint. Tools like Bubblemaps visualize the funding graph directly, and [labeling and attribution](/blog/trading/onchain/labeling-and-attribution) pipelines automate much of this clustering at scale.

The false-positive risk runs the other way too, and it matters because over-aggressive detection harms real users. The biggest source of false links is **shared infrastructure**: thousands of unrelated users withdraw from the same exchange hot wallet, route through the same bridge, or use the same router contract, and a naive "common funder" rule would lump them all together. A good detector treats funding from a large, public, many-to-many source (an exchange, a bridge, a faucet) as *no signal*, and reserves the common-funder heuristic for *small, private* wallets that seed a *bounded* set of addresses. Likewise, two genuine traders occasionally trade with each other; that is a single weak link, not a ring. The way you avoid punishing real users is the same way you avoid being fooled: never act on one signal, require independent corroboration, and weight a closed trading graph and behavioral identity (which coincidences rarely reproduce) above a lone funding link.

### How a wash tries to hide, and why detection still wins

A sophisticated operator knows the heuristics and tries to defeat them, and it is worth understanding the evasion attempts *as a defender*, because each one leaves its own residue.

- **More wallets.** Instead of two, the operator uses fifty, so no single wallet is obviously on both sides and the trading graph looks busier. But fifty wallets all need funding and all eventually drain somewhere — the funding fan-out and the recombination at the end are *harder* to hide with more wallets, not easier, because there are more edges to trace back to the root.
- **Funding through a hop.** The operator funds the farm wallets from an intermediate wallet, or routes the seed money through a bridge or exchange first, to break the direct funding link. This costs more (extra hops, extra fees) and it is exactly the kind of layered flow that [cross-chain tracing](/blog/trading/onchain/cross-chain-tracing-bridges-and-the-usdt-rails) and peel-chain analysis are built to follow; the link is obscured, not erased.
- **Randomized sizes and timing.** Instead of identical \$50,000 swaps every ten minutes, the operator varies amounts and intervals to defeat the behavioral-fingerprint check. This weakens one signal but cannot touch the others: the net flow still nets to zero, the price still fails to discover, and the cost is still being funded for a reason.
- **Mixing in real trades.** The operator interleaves a few genuine trades to lower the concentration ratio. This raises the cost (the genuine trades carry real risk) and only dilutes the signal rather than removing it — the washed component still shows the closed-loop, zero-net, funded-loss signature underneath.

The reason detection ultimately wins is structural: **the goal of wash trading and the goal of hiding it are in tension.** To fake volume you must trade a lot; to hide that it's fake you must look like many independent traders; but independence costs money (funding many wallets, hopping funds, taking real risk) and the cheapest, highest-volume wash is also the easiest to catch. The harder a wash works to look real, the more it bleeds — and the bleed is the one signal no amount of obfuscation removes. That is the defender's permanent advantage on a public ledger.

## The CEX problem: when you can't see the book

Everything above relies on the counterparties being *on-chain*. Centralized exchanges break that assumption: when you trade on a CEX, the match happens on the exchange's internal database, not the blockchain. You can't see the order book, you can't see who was on the other side, and you can't tell whether the exchange's own market-making desk filled both legs. CEX wash trading — including exchanges inflating their own reported volume to climb ranking sites — has been a documented, large-scale problem; multiple studies have estimated that a substantial fraction of reported CEX spot volume across the industry has historically been fake.

So how do you reason about something you can't directly see? You triangulate from the on-chain edges of the CEX, where it *does* touch the public ledger:

- **Reserve-vs-volume sanity.** A CEX claiming enormous spot volume in a token should hold meaningful on-chain reserves of that token and see on-chain deposits/withdrawals consistent with that activity. If an exchange reports billions in volume for an asset but its visible on-chain wallets hold and move almost none of it, the reported volume can't be backed by real settlement.
- **Withdrawal/deposit flow vs claimed activity.** Real trading volume eventually expresses itself as people depositing to buy and withdrawing what they bought. Volume with no corresponding on-chain deposit/withdrawal footprint is a flag.
- **Proof-of-reserves gaps.** Post-FTX, many exchanges publish on-chain proof-of-reserves. A persistent mismatch between claimed activity, claimed reserves, and verifiable on-chain holdings is a structural warning. (For why this matters, see the [FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried).)

A concrete triangulation runs like this. First, identify the exchange's on-chain wallets for the asset — block explorers and labeling services tag the major exchange hot and cold wallets, and the exchange's own proof-of-reserves disclosures point to them. Second, measure the daily on-chain flow into and out of those wallets: deposits (users funding accounts) and withdrawals (users taking custody). Third, compare that flow and the reserve size to the reported trading volume. A venue genuinely matching buyers and sellers at scale must, over time, see deposits and withdrawals and reserves that *scale with* the volume — money has to enter to be traded and leave when withdrawn. Fourth, look at the *churn ratio*: reported volume divided by on-chain settlement. A normal venue's reported volume can legitimately exceed on-chain settlement by some multiple (lots of trading happens against balances already on the exchange), but when that ratio is in the dozens or hundreds, the reported volume is mostly internal print with no external backing.

You can't *prove* CEX wash trading from the chain alone the way you can prove a DEX wash — but the on-chain footprint constrains what the off-chain claims can credibly be. When reported volume and on-chain settlement diverge by an order of magnitude, treat the reported number as marketing, not measurement. The same logic underlies why post-FTX proof-of-reserves became a baseline expectation: an exchange that won't let you reconcile its claims against the public ledger is asking for a trust you have no way to verify, and unverifiable volume is the first thing to discount.

#### Worked example: a \$1B reported-volume exchange that settles \$30M on-chain

An exchange advertises \$1,000,000,000 of daily volume in a token to top a ranking site and attract listings fees. You check its known on-chain wallets for that token: net deposits and withdrawals total about \$30,000,000 a day, and its reserves of the token sit around \$40,000,000. Real \$1B/day of churn — even with heavy internal netting — would normally leave a far larger on-chain footprint than \$30M of settlement against \$40M of reserves. The \$1B figure and the \$30M of real settlement are off by roughly 33×. You can't subpoena the order book, but the gap tells you the headline volume is mostly internal print, not external demand — and you size your trust accordingly.

## Common misconceptions

**"High volume means high demand."** No — volume is the *value of trades*, and a trade only requires that someone (possibly the same someone) was on both sides. Demand requires net new buyers willing to hold. A token can show \$5M of volume with 12 wallets and zero net accumulation. Always pair a volume number with a unique-buyer count and a net-flow check before you read it as demand.

**"On-chain volume can't be faked because it's all verified."** The transactions are verified; their *meaning* is not. The blockchain confirms that 1,000,000 tokens moved from A to B. It does not confirm that A and B are different people, or that any real economic exchange occurred. Verification of execution is not verification of intent. The chain's gift isn't that fakery is impossible — it's that fakery is *traceable*.

**"If it costs the washer money, they won't do it, so expensive volume must be real."** Backwards. The cost is exactly why expensive, persistent, no-discovery volume is the *strongest* wash signal: a rational actor only funds a recurring loss to obtain something worth more — a listing, a reward, an exit. The fee burn is the confession, not the alibi.

**"A high NFT sale price sets the floor."** Only if the buyer is unrelated and the money came from outside. A \$200,000 sale between two wallets funded by the same address sets nothing — it's the seller paying themselves. Always check whether outside money entered and whether the counterparties are linked before you anchor on a sale price.

**"Wash trading is rare and only on shady tokens."** It's pervasive, and it has appeared on major venues — entire NFT marketplaces' headline volumes were substantially reward-driven self-trading, and industry studies have flagged large fractions of CEX spot volume as fake at times. Treat *every* volume number as a claim to be verified, not a fact to be trusted, especially on incentivized platforms.

## The playbook: what to do with it

Everything above reduces to a small set of signals, each mapping to a read and an action. The matrix below is the field guide: when you see the left column, conclude the middle, and do the right.

![Detection matrix mapping each wash signal to what it means and what a trader or analyst should do](/imgs/blogs/detecting-wash-trading-7.png)

The single most important habit the matrix encodes is at the bottom row: when volume costs more in fees and gas than it could possibly profit, someone is funding a loss, and you must find the motive before you trust the number. Now the if-then checklists.

For the **trader/investor** — don't buy fake demand:

- **Signal:** a token or NFT collection with eye-catching volume relative to its size or age. → **Read:** run the volume-÷-unique-buyers check; pull the top traders; look for both-sides oscillation and a flat price under heavy "volume." → **Action:** if volume concentrates in a few linked wallets with no price discovery, discount the volume to *zero* and do not buy. → **Invalidation:** volume spread across hundreds of independent, externally funded buyers with genuine price movement is real demand — that's a different, tradeable signal.
- **Signal:** an NFT collection with a freshly spiked floor or a headline mega-sale. → **Read:** check whether the buyer and seller wallets share a funder, and whether any outside money entered the collection. → **Action:** if the sale is between linked wallets with no external inflow, treat the "floor" as fake and ignore it. → **Invalidation:** multiple recent sales to distinct, unrelated, externally funded buyers.
- **Signal:** an exchange or token topping a volume ranking. → **Read:** compare reported volume to on-chain settlement and reserves. → **Action:** when reported volume dwarfs the on-chain footprint, treat the ranking as marketing and don't let it drive a thesis. → **Invalidation:** on-chain deposits, withdrawals, and reserves scale with the claimed volume.

For the **analyst/defender** — expose it:

- **Signal:** suspicious volume on a token or collection. → **Read:** pull swaps/sales, group by counterparty, measure concentration, check for both-sides behavior and flat price. → **Action:** trace the top counterparties' funding to a common source; confirm a closed trading graph and behavioral fingerprints; stack three or four signals. → **Invalidation:** counterparties trace to many independent sources and trade openly with the outside world — that's a real market.
- **Signal:** an incentivized marketplace or airdrop with surging activity. → **Read:** estimate the reward-vs-fee math; find activity whose only profit is the reward. → **Action:** discount the platform's reported volume by the farmed share; flag the future sell-pressure as rewards vest. → **Invalidation:** activity persists at similar levels *after* incentives end — then it was real engagement.

The one rule that ties it all together: **volume is a claim, not a fact.** On a public chain you have the rare ability to audit that claim by following the counterparties. A volume number you haven't traced is a number you don't actually know.

## Further reading & cross-links

- [Address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics) — the machinery for tracing wash counterparties back to one operator (common funder, closed graph, behavioral fingerprint).
- [Airdrop farming and sybil cohorts](/blog/trading/onchain/airdrop-farming-and-sybil-cohorts) — the incentive-wash overlap: one operator, many wallets, farming rewards by faking activity.
- [Active addresses and network activity](/blog/trading/onchain/active-addresses-and-network-activity) — why raw activity counts (including unique buyers) are the sanity check that volume numbers hide.
- [Labeling and attribution](/blog/trading/onchain/labeling-and-attribution) — how cluster labels and entity tags are built and where their false positives come from.
- [FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) — why off-chain books can hide everything, and why proof-of-reserves and on-chain footprints matter.
- [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — how DEX liquidity pools and LP fees actually work, the substrate every DEX wash trade runs on.

A note on the patterns and economics that overlap this topic — *fake-volume-vs-organic-demand* screening, *DEX liquidity and pool* mechanics, and *pump-and-dump and coordinated buying* — are treated as their own deep-dives elsewhere in this series; this post focuses specifically on the wash-trading flavor of fake volume, where the counterparties are one operator. The unifying discipline across all of them is the same: never trust a headline number you haven't traced to its source.
