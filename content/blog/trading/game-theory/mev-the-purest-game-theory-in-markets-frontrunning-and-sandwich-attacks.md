---
title: "MEV: The Purest Game Theory in Markets"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "How maximal extractable value turns a public blockchain into the cleanest live game theory anywhere, and exactly how an ordinary swapper protects themselves from frontrunning and sandwich attacks."
tags: ["mev", "game-theory", "defi", "sandwich-attack", "frontrunning", "slippage", "flashbots", "trading", "blockchain", "defense"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — On a public blockchain your pending trade is visible to everyone, and the order it lands in is sold to the highest bidder, so MEV is game theory with the curtain pulled all the way back: you can see exactly who is on the other side and exactly what they will do to you, which means you can defend against it.
>
> - A *sandwich attack* buys right before your swap, lets your buy push the price up, then sells into that push. You eat the difference as *slippage* — the gap between the price you expected and the price you got.
> - The fee race that decides who sits in front of you is a *winner's-curse* auction: more competing bots means the winner pays more, but the loss to you is still bounded by one number you control.
> - That number is your *slippage tolerance*. Set it tight and you put a hard ceiling on what any sandwich can take. Set it loose and you hand the attacker a blank check.
> - The single rule to remember: a sandwich can only steal up to the slippage you let it. Cap the slippage, route privately, or trade in a batch auction, and the attack stops paying.

In the summer of 2021 a trader tried to swap roughly \$200,000 of a token on a decentralized exchange. The trade was simple — sell one asset, buy another, at whatever the pool's price was. Within the same block, a bot bought the token milliseconds *ahead* of them, the trader's own large buy shoved the price up, and the bot sold its position back into that higher price in the very next transaction. The trader paid tens of thousands of dollars more than the screen had quoted. No exchange was hacked. No private key was stolen. Everything happened exactly as the protocol's rules allow. The trader had simply broadcast their intentions to a public waiting room full of opponents and forgotten that someone was reading.

That is *maximal extractable value*, usually shortened to **MEV**: the profit that can be captured by choosing which transactions go into a block and in what order. On a normal stock exchange the order book of resting orders is visible, but the queue of orders *about to arrive* is private — it lives inside the exchange's matching engine for microseconds and then it is gone. On a public blockchain that queue is laid out in the open for anyone to read, and the right to order it is literally auctioned. Strip away the jargon and MEV is the cleanest game of strategy in all of finance: complete information about your opponent's intentions, an explicit auction for advantage, and a payoff you can compute to the dollar. It is also, for an ordinary user, a solvable problem. The diagram below is the mental model for the whole post — a sandwich seen from the victim's seat, and the one setting that caps the damage.

![Sandwich attack timeline showing a victim swap front-run and back-run with a slippage limit capping the loss](/imgs/blogs/mev-the-purest-game-theory-in-markets-frontrunning-and-sandwich-attacks-1.png)

This post builds MEV from zero — the mempool, frontrunning, the sandwich, the priority auction, and proposer-builder separation — and then spends the back half on defense: the concrete settings and tools that take you out of the line of fire. The goal is not to teach anyone to extract MEV. It is to teach you to stop being the one it is extracted from.

## Foundations: the mempool, the auction, and why a swap is a readable signal

Before any of the attacks make sense, you need four primitives: what a blockchain transaction is, where it waits, why waiting in public is dangerous, and how the order it finally lands in gets decided. We will define each from scratch.

### A transaction is a public instruction, broadcast to everyone

When you do anything on a blockchain — send a coin, swap on a decentralized exchange, mint an NFT — you create a *transaction*: a signed instruction that says "do exactly this." You sign it with your private key (a secret number that proves the instruction is really from you) and then you *broadcast* it to the network. Broadcasting means handing it to one of the computers running the blockchain (a *node*), which forwards it to its peers, which forward it again, until thousands of machines around the world all hold a copy.

Here is the part newcomers miss: that broadcast happens *before* the transaction is finalized. It is not yet "on the chain." It is a pending instruction sitting in limbo, waiting for a block producer to pick it up. And while it waits, it is fully readable — the asset, the amount, the exchange, the minimum you are willing to accept, all of it. You have, in effect, announced your trade to the entire market and then asked it to please execute it a moment later.

### The mempool is the public waiting room

That limbo has a name: the **mempool** (short for "memory pool"). Every node keeps a mempool — a list of all the valid transactions it has heard about that have not yet been included in a block. Think of it as the lobby of a bank where everyone's deposit slip is held up over their head for the room to read while they wait for a teller. Anyone can connect to a node, subscribe to its mempool, and watch the flow of pending transactions in real time.

For most transactions this is harmless. A simple coin transfer leaks nothing worth acting on. But a *swap* on a decentralized exchange (a "DEX") is different, because of how DEX prices work.

A DEX like Uniswap does not match buyers to sellers from an order book. It uses an *automated market maker* (AMM) — a pool holding two assets, say ETH and a stablecoin, with a formula that sets the price from the ratio of the two. The canonical formula is the *constant product*: the quantity of ETH in the pool times the quantity of stablecoin stays constant. The practical consequence is the only thing you need to remember: **a buy raises the price and a sell lowers it, and a bigger trade moves the price more.** That movement is called *price impact* or *slippage*.

So a pending DEX swap is not a neutral fact. It is a signal that says "I am about to push this pool's price in this direction by roughly this much." A bot reading the mempool can compute, before your trade executes, exactly how far your buy will move the price — and therefore exactly how much it can profit by trading around you. The swap is readable, and what it reveals is actionable. That is the seed of every MEV attack.

The crucial property is that price impact is *deterministic and public*. The pool's reserves are on-chain for anyone to read; the constant-product formula is public; your trade size is in the mempool. Plug all three into the formula and the bot knows your fill price to many decimal places before you do. There is no estimation, no guessing about your "true intentions" — your intention is the transaction, sitting in plain sight, and the math that turns it into a price move is fixed. This is what makes MEV cleaner than any off-chain game: in a dark pool or a traditional venue, a predator has to *infer* your size from indirect clues; on-chain, your size is literally a field in a public message.

#### Worked example: how price impact grows with trade size

Suppose a pool holds 1,000 ETH and 2,000,000 stablecoins, so the price starts at 2,000,000 ÷ 1,000 = \$2,000 per ETH. The constant product is 1,000 × 2,000,000 = 2,000,000,000. Now you buy 10 ETH. The pool must keep the product constant, so after you remove 10 ETH it holds 990 ETH, and the stablecoin side must rise to 2,000,000,000 ÷ 990 ≈ 2,020,202. You paid about 2,020,202 − 2,000,000 = \$20,202 for 10 ETH — an average price of \$2,020 per ETH, already 1% above the \$2,000 you saw quoted, purely from your own impact.

Now buy 50 ETH instead. The pool holds 950 ETH, the stablecoin side rises to 2,000,000,000 ÷ 950 ≈ 2,105,263, and you paid about \$105,263 for 50 ETH — an average of \$2,105, a 5.3% impact. Five times the size produced more than five times the impact, because the curve steepens as you drain the pool. The intuition: your price impact is a public, computable number that grows faster than your trade size, which is exactly the number a sandwich bot maximizes against — and exactly why splitting a big order into small pieces reduces the signal you leak.

![Mempool as a public waiting room and auction where searchers read a pending swap and bid priority fees to a builder](/imgs/blogs/mev-the-purest-game-theory-in-markets-frontrunning-and-sandwich-attacks-2.png)

### Block ordering is auctioned, so being first is for sale

The last primitive is the most important and the least intuitive. Transactions do not execute in the order you sent them. They execute in the order the block producer *chooses*. A blockchain finalizes in *blocks* — batches of transactions stamped onto the chain every few seconds (about 12 seconds on Ethereum). Whoever produces the block decides which pending transactions to include and in what sequence.

How do they decide? Largely by money. Each transaction can attach a *priority fee* (often still called "gas," in nods to the unit that measures computational work) — a tip to the block producer for including it, and for including it *early*. A block producer, being economically rational, tends to order transactions by how much they pay. Higher tip, earlier slot.

This turns transaction ordering into an *auction*. If two bots both want to sit in front of your swap, they bid against each other in priority fees, and the higher bid wins the position. This is the **priority-gas auction** — the engine room of MEV. The right to be first is not a privilege; it is a thing you buy, and the price floats with demand. The whole game theory of MEV is downstream of this single fact: *order is for sale, and your intention to trade is public, so people will pay to position themselves against you.*

### The three primitives, named

Let us pin the vocabulary, because the rest of the post uses it constantly.

- **Mempool** — the public list of pending, not-yet-finalized transactions. Visible to anyone.
- **Slippage** — the difference between the price you expected and the price you actually got, caused by the pool moving (sometimes because *you* moved it, sometimes because someone moved it on you).
- **Slippage tolerance** — the maximum slippage you tell the exchange you will accept. If the final price is worse than this, your transaction *reverts* — it fails and undoes itself, and you pay only the network fee. This is your single most important defensive control, and we will quantify it later.
- **Priority fee (gas tip)** — the bid in the ordering auction. More tip buys an earlier position.

With these four, every MEV strategy becomes legible. We will now walk through them as games — and for each, the defense.

## Frontrunning: jumping the queue with a higher bid

The simplest MEV play is **frontrunning**: seeing your pending transaction and getting an identical or related one of your own executed *first* by paying a higher priority fee.

The everyday version is a ticket scalper. You post on a forum that you are about to buy concert tickets at face value the moment they release. A scalper reads your post, buys the tickets one second before you, and resells them to you at a markup. They added nothing. They simply got in front of a move you had already signaled, using speed they were willing to pay for.

On-chain it works because the mempool is public and order is for sale. A bot watching the mempool sees a profitable pending action — maybe an arbitrage you spotted, maybe an NFT mint about to sell out, maybe an oracle update that will move a price — and submits its own version with a fatter tip so the block producer slots it ahead of yours. You broadcast the idea; they captured the value of acting on it first.

Frontrunning is a *first-mover* game where the move is bought, not earned. The defining feature is that the front-runner's profit comes from *your* signal. If you had not broadcast, there would have been nothing to front-run. That is why the deepest defenses all reduce to the same idea: *stop broadcasting your intentions to the people who would trade against them.*

#### Worked example: the cost of a front-run mint

Suppose a new NFT collection mints at a fixed \$100 each, and everyone expects them to trade at \$400 on the secondary market the instant minting ends. You spot this and submit a mint transaction with a \$5 priority tip. A bot reading the mempool sees your transaction, infers the \$400 resale, and submits an identical mint with a \$50 tip. The block producer, ordering by tip, puts the bot first.

The bot mints at \$100, you mint at \$100 right behind it (assuming supply lasts), but the *scarce, early* mints — the ones that resell highest — go to the bot. If the collection sells out and only the bot's batch makes it in, your transaction reverts and you are left with nothing but the gas you burned trying. The bot's edge over you was \$50 minus \$5 = \$45 of tip, spent to capture a \$300 resale margin per token. The intuition: frontrunning is just paying for a better seat in a queue you both can see, and the bot will pay any tip up to its expected profit.

The bot's willingness to pay is exactly what turns frontrunning into an auction — which is the next game.

## The priority-gas auction is a winner's-curse auction

When several bots all spot the same opportunity in the mempool, they do not politely take turns. They bid against each other in priority fees for the front position. That bidding war is the **priority-gas auction**, and it has a beautiful, slightly cruel structure that will already be familiar if you have read the sibling post on the [winner's curse in IPOs, treasury auctions, and mints](/blog/trading/game-theory/the-winners-curse-in-ipos-treasury-auctions-and-mints).

Here is why it is a *common-value* auction. In a *private-value* auction, the thing being sold is worth a different amount to each bidder — you value a painting by how much you personally like it. In a *common-value* auction, the thing has one true value that is the same for everyone, but nobody knows it exactly; each bidder only has a noisy estimate. An oil lease is the textbook case: the oil underground is worth the same to whoever wins, but each company's geologists guess the quantity differently.

An MEV opportunity is common-value. The arbitrage in front of your swap is worth, say, \$100 to whoever captures it — that number is fixed by the math of the pool. But each competing bot estimates it slightly differently: gas costs are uncertain, the pool might move, another transaction might interfere. Each bot bids roughly its own estimate. And here is the curse: **the bot that wins is, by definition, the one that estimated highest** — which means the winner is systematically the most over-optimistic, and tends to overpay.

The data model makes this exact. We treat the true opportunity value as \$100, and give each bot a noisy signal around it. For uniform noise of width \$20, the expected winning bid (the highest of the noisy signals) is the true value plus a bias that grows with the number of bidders. The chart below computes it directly.

![Winning bid in a priority-gas auction rising above the true value as the number of competing searchers grows](/imgs/blogs/mev-the-purest-game-theory-in-markets-frontrunning-and-sandwich-attacks-3.png)

#### Worked example: how badly the winning bot overpays

Take an opportunity truly worth \$100, with each bot's estimate off by up to \$20 either way. Using the winner's-curse model from the data toolkit, the expected winning bid climbs with the field:

- With **2 bots**, the expected winning bid is **\$106.67** — an overpayment of \$6.67.
- With **3 bots**, it is **\$110.00** — \$10 overpaid.
- With **5 bots**, **\$113.33** — \$13.33 overpaid.
- With **10 bots**, **\$116.36** — \$16.36 overpaid.

Read the pattern. The opportunity is worth \$100 to whoever wins, yet the winner expects to *bid* \$116 when ten bots compete — paying \$16 more than the prize is worth. Where does that \$16 go? Largely to the block producer, as the tip. The intuition: the more bots crowd a piece of MEV, the more of its value is competed away into fees, and the winner is the one who overestimated the most.

This has two consequences worth sitting with. First, for the bots, MEV is far less profitable than headlines suggest — competition burns most of it into priority fees paid to validators. Second, and more importantly for *you*: a large share of MEV value does not even stay with the attacker. It is dissipated in the auction. That matters for the defense story, because it means protocols that *redirect the auction's proceeds* — to you, or to a shared pool — can recover value that would otherwise vanish into the fee war. Hold that thought; it returns in the playbook.

For the formal treatment of why the winning bidder is cursed and how much to "shave" a bid to break even, the [winner's-curse post](/blog/trading/game-theory/the-winners-curse-in-ipos-treasury-auctions-and-mints) does the derivation. Here the point is narrower: the fee race that decides who frontruns you is a textbook common-value auction, and you can compute the overpayment exactly.

## The sandwich attack: the game that targets ordinary swappers

Frontrunning hurts other bots more than it hurts you. The attack that comes *for you*, the retail swapper, is the **sandwich**. It is the one to understand cold, because it is the most common form of MEV that touches normal users, and it is also the most defensible.

### The mechanics, step by step

A sandwich has three transactions, all in the same block, arranged around yours like bread around a filling:

1. **The front-run (top slice).** The bot sees your pending buy in the mempool. It buys the same token *first*, with a higher priority fee so it lands ahead of you. Its buy nudges the pool price up.
2. **Your swap (the filling).** Your buy executes next, at the now-higher price the bot just created. Your own buy pushes the price up *further*. You receive fewer tokens than you would have, or pay more — that is your slippage, and it is real money out of your pocket.
3. **The back-run (bottom slice).** The bot immediately sells everything it bought in step 1, into the elevated price your buy left behind. It pockets the difference.

The bot bought low (just before you), let your trade lift the price, and sold high (just after you). Your trade was the engine that moved the price in the bot's favor, and you paid for the privilege. The cover figure traces exactly this from your seat.

The asymmetry is the whole point. The bot risks almost nothing: it only acts when it can mathematically guarantee a profit, and if anything goes wrong its transactions revert. You, meanwhile, did nothing wrong except trade in public with a loose enough tolerance to be worth attacking.

#### Worked example: the dollars in a sandwich

You want to buy 10 ETH. The pool's fair mid-price is \$2,000, so you expect to pay \$20,000. You set a *loose* slippage tolerance — say 3% — because the wallet defaulted to it and you did not change it.

A bot reads your pending buy. It front-runs you, pushing the price up to \$2,010 (just under what your tolerance allows). Your buy then fills at about \$2,010 instead of \$2,000. The extra cost to you is:

$$\text{extra cost} = 10 \text{ ETH} \times (\$2{,}010 - \$2{,}000) = \$100$$

That \$100 is \$100 ÷ \$20,000 = **0.5%** of your trade, lost as slippage. The bot then sells its front-run position into the price your buy lifted, and the bulk of that \$100 ends up in the bot's pocket (minus its own fees and the tip it paid to win the front position).

Now the key insight for defense: the bot could only push the price to the edge of *your* tolerance. You set 3%, so it had room to extract up to roughly \$600 on a \$20,000 trade before your transaction would have reverted. It took \$100 here because that was the profitable amount, but your tolerance is the ceiling — the most a sandwich can ever take from a single swap. The intuition: you didn't lose \$100 because the bot was clever; you lost it because you left the ceiling high enough to be worth its while.

That last sentence is the entire defensive thesis, and the next chart turns it into a dial you control.

### Slippage tolerance is a hard cap, and you set it

Here is the defensive heart of the post. A sandwich's profit is bounded above by your slippage tolerance, because the moment the price moves past your tolerance, your transaction reverts and the bot's front-run becomes a loss for the bot. So the bot will only ever push the price *up to* your tolerance, never past it. Your tolerance is, quite literally, a ceiling on the attack.

We can compute the expected cost. Model it as a gamble: with some probability the bot finds your trade worth sandwiching, in which case it extracts up to your tolerance; otherwise you pay nothing extra. Using the expected-value tool from the data toolkit, on a \$10,000 swap with a 40% chance of being targeted, the expected slippage cost is your tolerance times your notional times that probability.

![Expected slippage cost rising linearly with slippage tolerance on a ten thousand dollar swap](/imgs/blogs/mev-the-purest-game-theory-in-markets-frontrunning-and-sandwich-attacks-4.png)

#### Worked example: the slippage dial, in dollars

You are swapping \$10,000. Assume a 40% chance a bot targets the trade. The expected cost is `0.40 × (tolerance × $10,000)`. Run it across settings:

- **0.1% tolerance:** expected cost = 0.40 × (0.001 × \$10,000) = **\$4.00**.
- **0.5% tolerance:** = 0.40 × (0.005 × \$10,000) = **\$20.00**.
- **1.0% tolerance:** = 0.40 × (0.01 × \$10,000) = **\$40.00**.
- **3.0% tolerance:** = 0.40 × (0.03 × \$10,000) = **\$120.00**.
- **5.0% tolerance:** = 0.40 × (0.05 × \$10,000) = **\$200.00**.

The relationship is dead straight: the looser your tolerance, the more you expect to lose, dollar for dollar. Moving from a careless 3% to a careful 0.5% cuts your expected sandwich cost from \$120 to \$20 — a 6× reduction — on the exact same trade. The intuition: slippage tolerance is not a safety net that "lets the trade go through," it is the size of the blank check you are writing to anyone who wants to sandwich you.

There is a real tradeoff, and honesty demands naming it. Set the tolerance *too* tight, and a genuine, honest price move — other people trading the same pool in the same block, with no attacker involved — can trip the revert and make your trade fail. You then have to resubmit, pay gas again, and possibly chase a moving price. So the right setting is "as tight as the pool's natural volatility allows," not "zero." For a deep, liquid pair like ETH/stablecoin, 0.1%–0.5% is usually fine. For a thin, volatile token, you may *need* 1%–3% just to land the trade — which is precisely why thin tokens are sandwiched most. The defense is real, but it is a dial, not a switch.

## Back-running and arbitrage: the MEV that doesn't (directly) hurt you

Not all MEV is predatory toward you. **Back-running** means placing a transaction *immediately after* a target transaction to capture a state it created — without front-running it. The most important and most benign form is **arbitrage**.

When a large swap moves a pool's price out of line with the rest of the market, the pool is now mispriced — say ETH is suddenly \$2,010 in this pool but still \$2,000 everywhere else. A back-running arbitrage bot buys the cheap side and sells the expensive side until the prices align. It profits from the gap, and in doing so it *pushes the pool back toward the true market price*. Your swap caused a temporary dislocation; the arbitrageur corrects it. They did not make your trade worse — your slippage was already incurred — they just cleaned up after it and kept the price.

This is the part of MEV that is arguably good for the system. Arbitrage is the mechanism that keeps prices consistent across dozens of fragmented venues. As the chart later in the post shows, arbitrage is the *largest* category of extracted MEV by value — and it is the one that, on net, makes markets more efficient rather than less. The distinction worth internalizing: **front-running and sandwiching are zero-sum or negative-sum against you; back-running arbitrage is mostly a tax on inefficiency that you were not going to keep anyway.** Defense is about the first kind. The second kind is a feature.

Liquidations are a third flavor: when a borrower's collateral falls below the required threshold, protocols allow anyone to repay the debt and seize the collateral at a discount. Bots compete (again, via the priority-gas auction) for the right to do it. This keeps lending protocols solvent. It is MEV, it is competitive, and it does not target ordinary swappers — it targets under-collateralized loans, which is a service the protocol explicitly wants performed.

The reason the distinction matters for defense is that it tells you where to spend your worry. If you are a *swapper*, the only kind of MEV that reaches into your pocket is the sandwich (and its degenerate cousin, plain frontrunning of your specific trade). Arbitrage and liquidations swirl around you but do not extract from you — they extract from price gaps and from over-leveraged borrowers. So the defensive checklist is short and specific: you are protecting one swap from one attack, and the levers are slippage, privacy, and venue choice. You do not need to defend against the entire MEV supply chain; you need to make your single trade not worth sandwiching.

#### Worked example: when a sandwich is *not* worth it to the attacker

A bot considers sandwiching your swap. Its profit is roughly the slippage it can force on you, minus its own costs: the priority tip it must pay to win the front position, plus the gas for its two extra transactions. Suppose your trade, given your tolerance, lets the bot extract at most \$30 of slippage. To win the front position against competing bots, the winner's-curse auction forces a tip of, say, \$22, and the bot's own gas runs \$10. The bot's expected profit is \$30 − \$22 − \$10 = **−\$2**. It is unprofitable, so the rational bot does not attack — your trade is simply not worth sandwiching.

Now loosen your tolerance so the bot could extract \$120 instead. Its profit becomes \$120 − \$22 − \$10 = **+\$88**, and now it attacks. The intuition: you do not have to make sandwiching impossible, only unprofitable — push the extractable slippage below the bot's cost floor and the rational attacker walks away on its own.

## Proposer-builder separation: how the MEV supply chain actually works

To understand both the threat and the strongest defenses, you need to know who the players are and how the block actually gets assembled today. The naive picture — "the validator reads the mempool and orders transactions greedily" — was roughly true in the early days and led to chaos: bots spamming the public mempool with ever-higher gas bids, congesting the network for everyone, in open priority-gas wars. The ecosystem responded with a structure called **proposer-builder separation (PBS)**, pioneered in practice by **Flashbots**. It splits the job into specialized roles.

- **Searchers** scan for opportunities — arbitrage, liquidations, and yes, sandwiches. A searcher does not produce blocks. It finds a profitable sequence of transactions and packages it as a *bundle*: a set of transactions that must be executed together, in a specified order, or not at all.
- **Builders** assemble full blocks. They take bundles from many competing searchers (each bundle including a tip to the builder), plus ordinary user transactions, and they solve a packing problem: which transactions, in which order, produce the most valuable block? The builder keeps a sliver and passes the rest up.
- **Proposers** are the validators — the parties with the right to add the next block to the chain. Crucially, they do not build the block themselves. They run a *blind auction*: builders submit sealed bids ("I'll pay you X for the right to have my block proposed"), the proposer simply picks the highest bid and signs it, often without even seeing the block's contents until after committing. A neutral *relay* sits between builder and proposer to make this blind auction trustworthy.

The pipeline figure makes the flow concrete.

![Proposer-builder separation pipeline from searcher to bundle to builder to relay to proposer to finalized block](/imgs/blogs/mev-the-purest-game-theory-in-markets-frontrunning-and-sandwich-attacks-5.png)

Why does this structure exist, and why should you care? Three reasons.

First, it *moved the priority-gas war off the public mempool*. Searchers submit bundles privately to builders instead of screaming gas bids into the public mempool. That reduced the network-wide congestion the old open auctions caused. It is, in a sense, a more civilized auction — but it is still an auction, and the winner's curse from earlier still governs who pays what.

Second, it *democratized block-building*. A small validator does not need to be a sophisticated MEV engineer; it can just sell its block-building rights to the highest professional bidder and collect the proceeds. This is efficient, but it concentrates the actual ordering power in a handful of builders — a centralization concern that protocol designers actively worry about.

Third, and most useful to you: **the same private channel that searchers use to avoid the public mempool is available to you, as a defense.** If a searcher can submit a transaction straight to a builder without ever touching the public mempool, so can you — and a transaction the bots cannot see is a transaction they cannot sandwich. That is the mechanism behind "private RPCs" and "MEV-protect" endpoints, which we get to in the playbook. PBS is the attacker's supply chain, but its plumbing is also the defender's best tool.

#### Worked example: who keeps the \$100

Trace a single \$100 arbitrage through the PBS pipeline to see where the value lands. A searcher spots the opportunity, builds a bundle, and to win inclusion offers the builder a tip. With the field of competing searchers from the winner's-curse math, suppose the searcher must tip \$90 to reliably beat the others. The builder keeps a small margin — say \$5 — and bids the remaining \$85 to the proposer to win the blind auction. The proposer (validator) takes the \$85.

So of the \$100 opportunity: the validator captures \$85, the builder \$5, and the searcher keeps just \$10 for actually finding it. The intuition: in a competitive MEV market, almost all the value flows to whoever controls the scarce resource — block space and ordering — which is the validator. The searcher who did the clever work keeps a thin slice. This is exactly the winner's curse in action: competition among searchers bids the opportunity's value up to whoever owns the ordering rights.

That economic fact is the seed of the most elegant defenses, which redirect the proposer's take back toward the users who generated it.

### Why PBS made the game safer for users — and what it concentrated

It is worth being precise about what PBS did and did not fix, because the popular story ("Flashbots fixed MEV") is too rosy. PBS solved a *coordination externality*. In the open priority-gas-war era, every searcher's bid was a public message that congested the network for everyone — bots spamming higher and higher gas prices clogged the mempool, and an ordinary person trying to send their grandmother some money found their unrelated transaction stuck or absurdly expensive. By moving the bidding into private bundle submission, PBS internalized that war: the bidding still happens, but it no longer dumps congestion on bystanders. That is a genuine, large win for ordinary users, even ones who never trade.

What PBS did *not* do is eliminate sandwiching. A searcher can still build a bundle that sandwiches a victim — it just submits that bundle privately to a builder instead of racing it in the public mempool. So PBS made the *side effects* of MEV cheaper for the network while leaving the *core extraction* intact. That is precisely why the user-level defenses in the playbook still matter: PBS protects the bystanders from the war's collateral damage, but it does not protect the *target* of a sandwich. Only routing your own trade out of the readable mempool, or trading where ordering can't be sold, does that.

And there is a cost on the other side of the ledger: PBS concentrated block-building into a small number of professional builders, because building the most valuable block is a sophisticated, capital-intensive optimization. A handful of builders winning most blocks is a centralization pressure that the community watches closely, because whoever controls ordering controls the most powerful lever in the system. The lesson for you is indirect but real — the health of the venue you trade on depends on this builder market staying competitive and honest, which is one more reason the batch-auction and fair-ordering designs in the playbook are attractive: they reduce how much trust you have to place in any single builder.

## Common misconceptions

A handful of beliefs lead people straight into the sandwich. Each is wrong for a reason you can now state precisely.

**"My transaction is private until it's confirmed."** The opposite is true. From the moment you broadcast, until it lands in a block, your transaction sits in the public mempool where anyone can read it. The window is short — seconds — but bots operate in milliseconds, and seconds is an eternity to them. Privacy begins *after* confirmation, not before, on the default path. The whole point of a private RPC is to skip the public mempool entirely so this window never opens.

**"Sandwiching is hacking — it should be illegal/impossible."** It is neither a hack nor a bug. The bot follows the protocol's rules exactly: it pays a higher fee for an earlier slot, which the rules allow. Calling it illegal misunderstands the system; the correct framing is that it is a *design consequence* of public mempools plus ordering-for-sale, and the response is design (private routing, batch auctions, fair ordering), not indignation. You cannot litigate a sandwich, but you can make your trade not worth sandwiching.

**"A higher slippage tolerance just means my trade is more likely to go through."** It means exactly that — and it *also* means you have raised the ceiling on how much a sandwich can take. Wallets often default to a tolerance high enough to guarantee execution, because a reverted trade generates a support ticket and a successful-but-sandwiched trade does not. The default optimizes for "the trade lands," not for "you keep your money." On a liquid pair, the cost of tightening it is usually just an occasional resubmit; the benefit is a hard cap on extraction. Treat the default as a starting point to lower, not a recommendation.

**"Big traders get sandwiched; my small swap is safe."** Profitability, not size, decides. A sandwich pays when the price impact your trade creates, times the amount the bot can ride, exceeds the bot's costs (its tip plus its own gas). A *small* swap into a *thin, illiquid* pool can move the price more than a large swap into a deep pool — and thin pools are where tolerances have to be loose, which raises the ceiling. So a \$500 swap into an obscure token can be a juicier target than a \$50,000 swap into ETH. The relevant question is never "how big is my trade?" but "how much price impact does it create, and how high did I set my ceiling?"

**"MEV is a niche problem for degens; it doesn't affect serious finance."** MEV is just the on-chain, fully-visible version of a force that exists in *every* market: the advantage of seeing order flow and controlling execution order. Traditional markets call its cousins "payment for order flow," "latency arbitrage," and "front-running" — and regulators have fought them for decades precisely because they are hard to see. On-chain, the same game runs in the open where you can measure it to the dollar. That is what makes MEV the *purest* instance of the strategy, not a fringe one. If you trade anywhere, the lesson — *don't broadcast your intentions to people who profit from acting on them first* — is universal.

## How it shows up in real markets

MEV is not a thought experiment. It is a measured, multi-hundred-million-dollar phenomenon with named episodes. Here are concrete cases, with the mechanism from this post visible in each.

**The cumulative scale (Flashbots / EigenPhi, 2020–2023).** Flashbots' MEV-Explore dashboard tracked well over half a billion dollars of extracted MEV on Ethereum in the pre-Merge era, dominated by arbitrage and liquidations, with sandwich attacks a large and growing slice that specifically targets ordinary swappers. Independent analytics firm EigenPhi has reported sandwich extraction running into the hundreds of millions of dollars across the ecosystem. The chart below shows the rough composition — arbitrage largest, sandwiches second, liquidations third — as those dashboards have reported it. These are cited orders of magnitude, not a live reading, but the ordering is robust across sources.

![Bar chart of cumulative extracted MEV on Ethereum split into arbitrage sandwiches and liquidations](/imgs/blogs/mev-the-purest-game-theory-in-markets-frontrunning-and-sandwich-attacks-7.png)

The single most important thing to read off that chart: arbitrage, the largest bar, is mostly the *benign* MEV that keeps prices consistent and that you were never going to keep anyway. The sandwich bar — smaller but very real — is the part that comes directly out of ordinary users' pockets, and it is the part the defenses in this post neutralize.

**The priority-gas-war era (2020–2021).** Before PBS, searchers competed by spamming the *public* mempool with escalating gas bids. During hot opportunities, the network's average gas price would spike as bots outbid each other in the open, and ordinary users found their unrelated transactions suddenly cost a fortune or got stuck. This was the priority-gas auction at its most visible and most destructive — the winner's curse playing out on-chain, with the externalities (congestion, sky-high fees) dumped on everyone. Flashbots' private bundle channel was introduced largely to drag this war off the public mempool, which it substantially did.

**The "salmonella" and counter-attack folklore.** As awareness grew, defenders started building *traps*. Some users deployed tokens or contracts engineered so that a naive sandwich bot trying to attack them would itself lose money — a kind of honeypot. The existence of these counter-attacks is a sign of a maturing game: once the victims understand the attacker's strategy completely (and on a public chain they *can*), they can construct positions where the attacker's best response loses. It is the clearest possible demonstration that MEV is a game of complete information, where reasoning one level deeper flips the outcome.

**The rise of MEV-protected RPCs as a default.** By 2022–2023, the defensive tooling matured from niche to mainstream. Endpoints like Flashbots Protect, MEV Blocker, and similar services let any user route transactions privately — straight to builders, skipping the public mempool — and several wallets began offering this as a one-click or default option. The significance is that the strongest defense stopped requiring technical sophistication. The same private channel that professionals used to *extract* MEV became a consumer product to *avoid* it.

**Batch auctions and CoW Swap.** A different design attacked the root cause rather than the symptom. CoW Protocol (the name stands for "Coincidence of Wants") settles trades in *batches* with a single uniform clearing price per asset per batch, rather than one-at-a-time with per-transaction ordering. If there is no per-transaction ordering to exploit, there is no front position to buy — the sandwich loses its foothold entirely. We will unpack the game theory of this in the playbook, because it is the most complete protocol-level answer.

**The thin-token sandwich pattern.** Analytics dashboards repeatedly find the same fingerprint: the swaps sandwiched most aggressively are mid-sized trades into *low-liquidity* tokens, executed with the wallet's *default* (loose) slippage tolerance. The reason maps exactly onto the worked examples above — a thin pool produces large price impact even from a modest trade, and a loose tolerance leaves a wide ceiling for the bot to extract under. A \$3,000 swap into a freshly listed token with a 5% default tolerance can lose more in absolute dollars than a \$100,000 swap into ETH with a 0.3% tolerance. The episode-level lesson is not "trade small," it is "the two dials that decide your exposure are pool depth and the tolerance you set," and only one of those is under your control on the fly — so set it deliberately.

**The cross-domain frontier.** As the ecosystem spread across many chains and layer-2 networks (cheaper, faster chains built on top of Ethereum), MEV followed. The same readable-mempool, ordering-for-sale game recurs on each venue, with its own builders, its own private channels, and its own defenses. The mechanics are chain-specific enough that the chain-level details belong in their own treatment — see the [on-chain series' deep dive on MEV, sandwiches, and frontrunning](/blog/trading/onchain/mev-sandwiches-and-frontrunning) for the per-chain plumbing. The strategic structure, though, is identical everywhere: public intentions plus auctioned order equals MEV.

## The playbook: how to protect yourself

This is the part that matters. You cannot abolish MEV — it is a structural feature of public, ordered blockchains — but you can take yourself out of the line of fire almost completely. The defenses fall into three tiers, from "free and immediate" to "best but requires the right venue." The comparison matrix lays them side by side.

![Comparison matrix of default RPC versus private RPC versus batch auction across visibility sandwich risk and cost](/imgs/blogs/mev-the-purest-game-theory-in-markets-frontrunning-and-sandwich-attacks-6.png)

### Tier 1 — Set a tight slippage tolerance (free, do it always)

This is the single highest-leverage habit. As the slippage chart proved, your tolerance is a hard ceiling on what any sandwich can take. Lower the ceiling and you lower the maximum loss, linearly.

- **Practical rule:** for deep, liquid pairs (major tokens against ETH or a stablecoin), set 0.1%–0.5%. For thin or volatile tokens, set the lowest value at which your trade still reliably lands — accept that you may need 1%–3% there, and recognize that thinness is *why* those tokens are the most sandwiched.
- **The tradeoff, named:** too tight and honest volatility reverts your trade, costing you a resubmit and gas. Too loose and you write a blank check. Tune to the pool's real volatility, not to the wallet's convenient default.
- **The invalidation:** if a trade keeps reverting at a tolerance you believe is reasonable, the pool may genuinely be that volatile — or thin enough that you should split the order or trade elsewhere. A persistently reverting tight tolerance is information, not just friction.

The execution-sizing logic here rhymes with traditional markets — breaking a large order into pieces to reduce the price impact each piece signals is the same idea as a VWAP/TWAP execution algorithm. The [execution-as-a-game post](/blog/trading/game-theory/execution-as-a-game-vwap-twap-and-hiding-from-predators) develops the off-chain version: hide your size, reduce your footprint, don't telegraph the whole order at once.

### Tier 2 — Route privately (skip the public mempool)

The deeper fix is to never broadcast to the public mempool at all. A *private RPC* or *MEV-protect* endpoint sends your transaction straight to a builder, who can include it in a block without it ever appearing in the public waiting room. A transaction the bots cannot see is a transaction they cannot front-run or sandwich.

- **What it costs you:** you are trusting the private relay/builder not to misbehave with your transaction, and on a bad day a private route might take an extra block or two to land if that builder doesn't win the slot. In practice the major services (the Protect-style endpoints) are reliable and free, and many even *return* to you some of the back-running MEV your trade generates.
- **Why it works, in game terms:** it removes the readable signal. Every attack in this post started with "a bot reads your pending swap in the mempool." Cut the read and the entire attack tree collapses at the root. This is the defensive payoff of the PBS plumbing — the same private channel professionals use to *send* bundles, repurposed so your trade never becomes a signal.
- **The invalidation:** private routing protects against *mempool-based* sandwiches, which is the dominant retail threat. It does not protect you from a malicious *builder* who sees your transaction inside their own block-building — a far rarer and more sophisticated risk, mitigated by using reputable relays and by the batch-auction designs in Tier 3.

### Tier 3 — Trade in a batch auction (remove the ordering advantage entirely)

The most complete answer changes the game rather than hiding from it. In a **batch auction**, many trades are collected over a short window and settled *together* at a single uniform clearing price, instead of one-by-one in a sequence. CoW Protocol is the prominent example.

The game theory is clean. A sandwich requires three things: see the victim's trade, get a transaction *in front of* it, and get one *behind* it. A batch auction destroys the middle requirement. If everyone in the batch settles at the *same* price, there is no "in front" and no "behind" — there is no ordering within the batch to exploit. The front position you would have to buy in a priority-gas auction simply does not exist. *Solvers* compete to find the best settlement for the whole batch (and to source liquidity), and that competition is structured to pass savings to users rather than extract from them. Sandwich MEV, which depends entirely on intra-block ordering, has nowhere to stand.

- **What it costs you:** batch settlement is a little slower than an instant swap (you wait for the batch window), and you rely on solver competition being healthy. For most trades that is a fine price for near-immunity to sandwiching.
- **Why it is the structural fix:** Tier 1 caps the loss, Tier 2 hides the signal, but Tier 3 removes the *mechanism*. It is protocol design doing the defending, which is why it protects every user of the venue by default, including the ones who never read a post like this.

### How protocol design redistributes or minimizes MEV

Beyond what an individual does, the ecosystem is engineering MEV out at the source, and it helps to know the directions because they tell you which venues to prefer.

- **Fair-ordering protocols** try to enforce an ordering rule — often "first-seen, first-served" by consensus, rather than "highest-fee-first" — so that paying for position simply doesn't buy it. If order can't be bought, frontrunning can't be executed. These are harder to build (the network has to agree on what "first" means), but they attack the root.
- **MEV redistribution / MEV-smoothing** schemes accept that MEV exists and route its proceeds back to the people who generate it — rebating the value to the user whose trade created the arbitrage, or socializing it across all validators so no single one is incentivized to reorder aggressively. Recall the winner's-curse result: much of MEV's value is dissipated into the priority auction anyway, so redistributing the proposer's take is recovering value that was never going to stay with the attacker.
- **Encrypted mempools** hide transaction *contents* until after ordering is fixed, so a bot cannot read your swap to act on it. This is the same idea as Tier 2 private routing, generalized to the whole network: if no one can read pending transactions, no one can front-run them.

The throughline of every protocol-level fix is the same as the throughline of every personal defense: **break the link between your visible intention and someone else's ability to act on it first.** Whether you cap the loss (slippage), hide the signal (private RPC), or remove the ordering (batch auction), you are attacking the same chain at a different link.

### A defender's decision tree

To make this operational, here is the order to think in for any swap, from cheapest action to strongest:

1. **Always tighten the slippage.** Free, instant, caps the worst case. Set it as low as the pool's real volatility allows. This alone takes most retail swaps out of the profitable-to-attack zone, because — as the unprofitable-sandwich example showed — pushing the extractable slippage below the bot's tip-plus-gas cost floor makes the rational attacker walk away.
2. **If the trade is large or the token is thin, route privately.** A private/MEV-protect RPC removes the signal entirely, so the size of your impact never becomes a readable target. This is the step that matters most precisely when slippage alone can't be tight (volatile or illiquid tokens, where you're forced to allow more room).
3. **If the venue offers it, prefer a batch auction.** For trades where a small settlement delay is acceptable, settling in a uniform-price batch removes the ordering the sandwich needs to exist. This is the closest thing to immunity, and it protects you by construction rather than by vigilance.
4. **Split large orders.** Independent of routing, breaking a big trade into smaller pieces reduces the price impact each piece signals — the on-chain version of the execution-algorithm logic that off-chain traders have used for decades to hide size from predators.

Notice that this tree never requires you to out-run the bots, out-bid them, or understand their code. It requires you to deny them, one link at a time, the three things every attack needs: a readable signal, room within your tolerance, and an ordering to exploit. That is the whole defense, and it is fully in your hands.

### Who is on the other side, and what game you are in

To close the playbook the way this series always does — name the players and the game. On the other side of your swap is a searcher running a bot that reads the public mempool, estimates the profit in front-running you, and bids in a winner's-curse auction for the right to do it. Its edge is *speed and visibility*: it sees your intention before the chain finalizes it, and it can pay to be first. Your edge is that on a public chain you can see the *entire* game — you know its strategy exactly, which means you can deny it the things it needs. It needs your trade to be readable: route privately and it can't read it. It needs room within your tolerance: tighten the tolerance and the room shrinks to nothing. It needs ordering to exploit: trade in a batch and the ordering disappears.

This is the deepest reason MEV is the purest game theory in markets. In most markets you are reasoning about an opponent you cannot see, with information you do not have. Here, the opponent's strategy is public, its constraints are public, and the counter is computable. Every market is, at bottom, an [auction](/blog/trading/game-theory/every-market-is-an-auction-the-double-auction-of-the-order-book) where someone is on the other side of your order — MEV is just the version where the curtain is gone, the auction is explicit, and, once you understand it, the defense is in your hands.

*This is educational, not financial advice. The numbers here are illustrative models, computed from the series' game-theory toolkit; real on-chain costs vary with the pool, the chain, the venue, and the moment.*

## Further reading & cross-links

- [MEV, sandwiches, and frontrunning (on-chain series)](/blog/trading/onchain/mev-sandwiches-and-frontrunning) — the chain-level mechanics: how the mempool, builders, relays, and per-chain variations actually work in practice.
- [The winner's curse in IPOs, treasury auctions, and mints](/blog/trading/game-theory/the-winners-curse-in-ipos-treasury-auctions-and-mints) — the full derivation of why the bidder who wins a common-value auction (including the priority-gas auction) systematically overpays, and how much to shave a bid.
- [Execution as a game: VWAP, TWAP, and hiding from predators](/blog/trading/game-theory/execution-as-a-game-vwap-twap-and-hiding-from-predators) — the off-chain cousin of slippage defense: how large traders break up orders to avoid telegraphing intent to front-runners.
- [Every market is an auction: the double auction of the order book](/blog/trading/game-theory/every-market-is-an-auction-the-double-auction-of-the-order-book) — why all the games in this series, MEV included, reduce to an auction with someone on the other side of your order.
