---
title: "Anatomy of a Transaction: Reading One End to End on a Block Explorer"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A field-by-field walkthrough of an on-chain transaction on Etherscan — hash, from/to, value, gas, status, logs, internal txns, and input data — so you can open any tx and decode exactly what happened."
tags: ["onchain", "crypto", "etherscan", "transactions", "ethereum", "gas-fees", "block-explorer", "token-transfers", "approvals", "eip-1559"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A blockchain transaction is a small, fixed object — who, to whom, how much, the gas, the data, the result — and a block explorer like Etherscan shows you every field. Learn to read one and you can audit any payment, any swap, any "I got drained" incident on the public ledger yourself.
>
> - **What it is:** every state change on Ethereum is a signed transaction with the same handful of fields (hash, from, to, value, nonce, gas, input data, status) plus a *receipt* (block, gas used, logs, internal transfers) the network generates after it runs.
> - **How to read it:** open the tx on [etherscan.io](https://etherscan.io), then walk the tabs in order — Overview answers *who paid whom*, Logs answers *which tokens moved*, Internal Txns answers *where ETH really went*, State answers *what changed*.
> - **What you do with it:** verify a payment actually landed, decode a swap into its two Transfer legs, spot a dangerous unlimited approval *before* it drains you, and understand why a failed tx still costs you real money.
> - **The one number to remember:** a plain ETH transfer always costs exactly **21,000 gas** — at 25 gwei and \$3,000 ETH that is about **\$1.58**, and that floor is the unit you measure every other fee against.

On the morning of 21 February 2025, the largest theft in the history of money settled in a few seconds and left a perfect paper trail. Attackers tied to North Korea's Lazarus Group tricked the signers of a Bybit cold wallet into approving a malicious upgrade, and roughly **\$1.46 billion** of ETH walked out of the exchange. Within minutes, blockchain investigators on X were posting the exact transaction hashes. Anyone — you, me, a journalist, a thirteen-year-old with a phone — could paste those hashes into Etherscan and watch the funds fan out across hundreds of fresh wallets in real time. There was no subpoena, no press office, no "we are investigating." The evidence was already public, permanent, and free to read.

That is the strange and wonderful thing about on-chain analysis: the ledger is open. Every payment, every swap, every loan, every scam, every bailout is recorded in the same format and visible to everyone forever. But "visible" is not the same as "readable." A first-timer who opens that Bybit transaction sees a wall of hex, a column called *Logs* full of `0x` gibberish, a tab named *Internal Txns*, and a fee denominated in something called *gwei*. It looks like a cockpit. This post is the cockpit checklist.

The transaction is the **atom** of on-chain analysis. Smart-money tracking, exchange-flow monitoring, rug detection, hack forensics — all of it is built out of reading transactions. If you cannot confidently open one transaction and say *who sent what to whom, how much it cost, whether it succeeded, and what it changed*, every fancier dashboard is a black box you have to take on faith. So we are going to start at the atom. By the end of this post you will be able to open **any** transaction on **any** EVM explorer and decode every field, and you will know the small but important differences when you switch to Solana's Solscan, Tron's Tronscan, or Bitcoin's mempool.space.

![Annotated diagram of an Ethereum transaction object with its fields, the state it changes, and the receipt](/imgs/blogs/anatomy-of-a-transaction-1.png)

## Foundations: what a transaction actually is

Before we open an explorer, we need a clean mental model of the thing the explorer is showing. Strip away the jargon and a blockchain is a giant shared spreadsheet of balances and contract storage — what engineers call the **world state**. A **transaction** is the *only* way that spreadsheet ever changes. Nothing on Ethereum happens except as the result of someone submitting a transaction. No transaction, no change.

A transaction is a small, signed message. "Signed" matters: the sender proves they authorized it using their **private key**, a secret number that controls their account. The account itself is identified by an **address** — a 42-character string starting with `0x`, like `0xA11ce…`. There are two kinds of account on Ethereum, and the distinction runs through everything:

- An **EOA** (Externally Owned Account) is a normal wallet controlled by a private key. A human (or a bot) holding the key signs transactions. EOAs are the only things that can *start* a transaction.
- A **contract account** is a program living at an address. It has code and storage but no private key. It cannot start a transaction on its own; it only runs when an EOA (or another contract) calls it.

So every transaction on the chain originates from an EOA pressing "sign." That single signed message then either moves value directly or wakes up a contract, which can in turn call other contracts, move tokens, and emit records. We will unpack that cascade in detail, because it is the single most confusing thing for newcomers: *one* signed transaction can cause *dozens* of downstream effects, and the explorer puts those effects in different tabs.

The reason every transaction on the chain has the *same* fields — and the reason a skill learned on one transaction transfers to all of them — is that the network's rules are uniform. There is no special "swap transaction type" or "NFT transaction type" at the protocol level. There is one transaction format, and the *difference between a payment and a complex DeFi action is entirely in the `to` address and the `input data`.* A payment points `to` an EOA with empty data; a swap points `to` a contract with data that encodes a function call. That uniformity is precisely why the explorer page looks the same every time and why, once you can read one, you can read them all. It is also why on-chain analysis is *possible*: because every action is recorded in one standard format on one shared ledger, a single tool can index and query the entire history of everyone, which no traditional financial system permits an outsider to do.

If you want the deeper "why Ethereum is a programmable computer and not just a payment network" story, the sibling post on [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) builds it from scratch; here we take it as given and focus on reading the output.

### The two flavors of transaction

There are really only two shapes a transaction can take, and telling them apart is the first skill:

1. **A plain value transfer.** The `to` field is an EOA, the `value` field is some amount of ETH, and the `input data` field is empty. This is "send 1 ETH to Bob." Simple, cheap, fixed cost.

2. **A contract call.** The `to` field is a *contract* address, and the `input data` field contains an encoded instruction — *which function to run and with what arguments*. The `value` field may be zero (you are not sending ETH, you are telling a contract to do something) or non-zero (you are sending ETH *and* triggering code, like depositing into a vault). This is "tell the Uniswap router to swap my USDC for ETH," or "approve this DEX to spend my tokens," or "mint this NFT."

The explorer makes this distinction visible the instant you open a transaction. If the `to` address has a little contract icon and the page has a populated **Logs** tab, you are looking at a contract call. If `to` is a plain address and Logs is empty, it is a transfer. Everything else in this post hangs off that fork.

### What a transaction changes, and what a "receipt" is

When you submit a transaction, you author the *request* part — the fields you sign. But there is a second half you do **not** author: the **receipt**, which the network generates after it actually runs your transaction. The request says "I want to do this"; the receipt says "here is what happened."

The request fields (what you sign) are: `nonce`, `to`, `value`, `gas limit`, the fee parameters, and `input data`. The receipt fields (what the network fills in) are: which `block` it landed in, the `timestamp`, the `status` (success or failure), the `gas used`, the **logs** array (events the contracts emitted), and any **internal transactions** (value that contracts moved while running). On Etherscan, the Overview tab blends the two so seamlessly that beginners do not realize half the page is "after the fact." Keeping the request/receipt split in your head is what stops the Logs and Internal Txns tabs from feeling like magic — they are simply *the receipt*, and the receipt is where most of an analyst's signal lives.

### Nonce: the anti-replay counter

Every EOA has a **nonce**: a counter that starts at 0 and increments by 1 with each transaction the account sends. Your first transaction ever has nonce 0, your second has nonce 1, and so on. The nonce does two jobs. First, it stops **replay**: without it, someone who saw your "send 1 ETH to Bob" transaction could rebroadcast the exact same signed message a hundred times and drain you; the nonce makes each transaction one-time-use. Second, it **orders** your transactions: the network will not process nonce 7 until nonce 6 has been included. This is why a single stuck transaction (one with too low a fee) can jam everything behind it — a detail that becomes a real trading problem when you are trying to get an order in during a volatile minute.

### Gas: the four numbers people confuse

"Gas" is where most newcomers' eyes glaze over, so let us be very precise. Running computation on Ethereum costs money because thousands of nodes must execute and store your transaction forever. **Gas** is the unit of computation. Four distinct numbers live under that word, and confusing them is the number-one beginner error:

- **Gas used** — *how much* computation your transaction consumed. A plain ETH transfer always uses exactly **21,000 gas** (it is a fixed-price operation). A token transfer uses ~45,000–65,000. A Uniswap swap uses ~120,000–250,000. A complex DeFi action can use 400,000+. This is the *quantity*.
- **Gas limit** — the *ceiling* you are willing to pay for. You set it above the expected gas used as a safety margin. If execution runs past the limit, the transaction **fails** and you still pay for the gas burned getting there (we will see this cost a person real money below).
- **Gas price** (pre-2021) or **base fee + priority fee** (post-EIP-1559) — the *price per unit* of gas, denominated in **gwei**. One gwei is one-billionth of an ETH (0.000000001 ETH). So a fee of "25 gwei" means you pay 25 billionths of an ETH for each unit of gas.
- **Fee** = `gas used × gas price`. This is the dollars-and-cents number that actually leaves your wallet, on top of any ETH you are sending.

The mental shortcut: **gas used is the quantity, gas price is the unit price, the fee is their product, and the gas limit is just the ceiling.** Memorize the 21,000-gas floor for a transfer and you have a reference point for judging whether any other fee is reasonable.

#### Worked example: the cost of a plain ETH transfer

Say you send 0.5 ETH to a friend. The transfer uses exactly **21,000 gas**. The network's base fee is 23 gwei and you add a 2 gwei tip, so your effective price is 25 gwei per gas. The fee is:

`21,000 gas × 25 gwei = 525,000 gwei = 0.000525 ETH.`

At an ETH price of \$3,000, that fee is `0.000525 × \$3,000 = \$1.58`. So your friend receives 0.5 ETH and you are out 0.500525 ETH — the half-ETH plus about **\$1.58** in fees. If the network were congested and the base fee spiked to 100 gwei, the same transfer would cost `21,000 × 100 gwei = 0.0021 ETH ≈ \$6.30` — same computation, four times the price, because you are buying scarce block space in an auction. *Intuition: a transfer's gas is fixed, so your fee moves only with the gas price, which is just the live auction price of block space.*

### EIP-1559: base fee burned, tip to the validator

In 2021, Ethereum changed how the gas price works (the EIP-1559 upgrade), and the explorer now shows the new structure, so you need to read it. Instead of one "gas price" you bid blindly, the protocol computes a **base fee** automatically for every block based on how full the previous blocks were. The base fee rises when blocks are full and falls when they are empty — it is an algorithmic congestion price. Critically, **the base fee is burned**: it is destroyed, removed from the ETH supply forever, paid to no one.

On top of the base fee you add a **priority fee** (a "tip"), which *does* go to the validator who includes your transaction — it is your incentive for them to pick you. And you set a **max fee** as your absolute ceiling. The protocol charges you `base fee + tip` and refunds the difference between your max fee and what it actually charged. So the explorer shows you four gas-related numbers on a modern tx: the base fee, the priority tip, the total, and "Txn Savings" (your refund).

![Stacked diagram showing the gas fee split into a burned base fee and a priority tip to the validator, with a worked dollar example](/imgs/blogs/anatomy-of-a-transaction-4.png)

#### Worked example: reading the EIP-1559 fee on a transfer

You authorize a max fee of 40 gwei because the network looks jittery. When your transaction lands, the base fee turns out to be 23 gwei and your tip is 2 gwei, so the effective price is 25 gwei. On 21,000 gas:

- Base fee burned: `21,000 × 23 gwei = 0.000483 ETH` — destroyed, paid to no one. At \$3,000 ETH that is `\$1.45` of ETH permanently removed from supply.
- Priority tip to the validator: `21,000 × 2 gwei = 0.000042 ETH ≈ \$0.13`.
- Total you pay: `21,000 × 25 gwei = 0.000525 ETH ≈ \$1.58`.
- Refund (your "Txn Savings"): you authorized 40 gwei but paid 25, so `21,000 × 15 gwei = 0.000315 ETH ≈ \$0.95` is refunded to you — you never lose the headroom you set.

*Intuition: setting a generous max fee costs you nothing extra in calm conditions — you only ever pay base + tip — but it buys you insurance against a sudden base-fee spike, which is exactly why experienced users leave headroom.*

## The lifecycle: from "sign" to "irreversible"

A field on the explorer reads "Confirmed" or shows a green check, and beginners assume that means "done, final, money in the bank." Sometimes it does not. To read the status field correctly you need the lifecycle of a transaction — the five states it passes through between your wallet and irreversibility.

![Pipeline of a transaction moving from signed to mempool to included in a block to confirmed to finalized](/imgs/blogs/anatomy-of-a-transaction-2.png)

1. **Signed.** Your wallet builds the transaction object and signs it with your private key. At this instant it exists only on your device. It has not touched the chain.
2. **Mempool (pending).** Your wallet broadcasts the signed transaction to the network, where it sits in the **mempool** — a waiting room of pending transactions that every node holds. It is now *publicly visible* but in *no block*. This is a crucial and dangerous window: bots watch the mempool, and a juicy pending swap can be **front-run** or **sandwiched** before it ever executes. The mempool is why mempool.space exists for Bitcoin and why MEV searchers exist for Ethereum.
3. **Included.** A validator picks your transaction out of the mempool and packs it into a block. Now it has a **block number** and a **timestamp**. Etherscan will show "1 Block Confirmation."
4. **Confirmed.** As more blocks pile on top, your transaction gets buried deeper. Each additional block makes it exponentially harder to reverse. By convention many exchanges wait for ~12 confirmations before crediting a deposit.
5. **Finalized.** On modern Ethereum, after about two epochs (~13 minutes), the transaction is **finalized** — by the protocol's economic guarantees it cannot be reverted without an attacker burning an astronomical amount of staked ETH. This is true "done."

Why does this matter to an analyst? Because a transaction can be **dropped** from the mempool (if it is replaced or its fee is too low), and in rare network forks an "included" transaction can be **reorged** out of its block before it is finalized. When you see a deposit or a payment, the trader's discipline is: *is it just included, or is it confirmed/finalized?* For small amounts, one confirmation is fine. For the size that moves a market, wait for finality. The explorer's status field and confirmation count are how you check.

## A field-by-field anatomy of an Etherscan transaction page

Now the main event. Open any transaction on [etherscan.io](https://etherscan.io) and you land on the **Overview** tab. Here is every field, top to bottom, and exactly what it tells you.

- **Transaction Hash.** The unique 66-character `0x…` fingerprint of this transaction. It is computed from the transaction's contents, so it is unique and tamper-evident — change any field and the hash changes. This is the "txid" you paste into the explorer or send to someone as proof. *Analyst use: the hash is your permanent citation. When you reference an on-chain event, you cite the hash.*
- **Status.** Either **Success** (a green check) or **Fail** (a red mark). A failed transaction still ran, still consumed gas, and still cost the sender money — it simply reverted its state changes. We dig into this below because it surprises everyone.
- **Block.** The block number it was included in, with the current confirmation count beside it. Click it to see the whole block. *Analyst use: the block number plus timestamp is your "when."*
- **Timestamp.** Wall-clock time, plus "X mins ago." On-chain timing is itself a signal — a wallet that moves seconds before a big announcement is a story.
- **From.** The EOA that signed and sent the transaction (and pays the gas). On Etherscan, known addresses get a **label** here — "Binance 14," "Uniswap Router," "Tornado Cash." Those labels are gold for analysis; we will return to them.
- **To / Interacted With.** The destination. If it is an EOA, this is a plain recipient. If it is a contract, Etherscan writes "Interacted With (To): Uniswap V3: Router" and shows a contract icon. This is your instant transfer-vs-contract-call signal.
- **Value.** The amount of **ETH** moved by the transaction *itself* (the `value` field). Read this carefully: for a token swap, `value` is often **0 ETH**, because the tokens move through the Logs, not the value field. Beginners panic ("it says 0, did my money move?") — no, your *ETH* didn't move directly; your *tokens* did, and those live in Logs.
- **Transaction Fee.** The total fee in ETH and in fiat: `gas used × effective gas price`. This is what you actually paid the network.
- **Gas Price / Base / Priority.** The per-gas price, broken into base fee and tip on a post-1559 tx. Expand "More Details" to see gas limit, gas used (and the % of limit used), the burnt base fee, and your txn savings.
- **Nonce.** The sender's transaction counter for this tx. Useful for tracing the *sequence* of a wallet's actions.
- **Input Data.** The encoded instruction for a contract call (empty `0x` for a plain transfer). This is where the *method* lives, and Etherscan can decode it into a human-readable function name and arguments. More on this below — it is how you know *what a transaction was trying to do*.

That is the Overview. But a contract call's real story is told in the other tabs: **Logs**, **Internal Txns**, and **State**. The matrix below maps each tab to the question it answers — this map is most of the skill.

![Matrix mapping each Etherscan transaction tab to what it shows and the analyst question it answers](/imgs/blogs/anatomy-of-a-transaction-7.png)

## Logs and events: how one transaction moves five tokens

Here is the concept that unlocks on-chain analysis. When a contract runs, it can **emit events** — small records it deliberately writes to the receipt as a kind of structured printout. These records land in the transaction's **logs** array, and the explorer's **Logs** tab decodes them. Events are not the *state change* itself; they are an *announcement* of it, designed to be cheap to write and easy for outside software to read.

Why do events exist at all? Because reading a contract's internal storage from outside is slow and awkward, every well-built contract emits an event whenever something important happens — a token moves, a swap executes, an approval is granted. Wallets, indexers, Dune dashboards, and analysts all reconstruct "what happened" by reading these events rather than re-running the contract. Events are the chain's audit log, and the single most important one is the **Transfer** event.

Every ERC-20 token (USDC, USDT, WETH, every memecoin) emits a `Transfer(from, to, amount)` event whenever its balance moves. So when you read the Logs tab, each `Transfer` row tells you: *this token, moved this many units, from this address, to this address.* A token's actual movement is invisible in the `value` field (which only tracks ETH) — it lives entirely in these Transfer logs. This is why a swap shows "Value: 0 ETH" up top but a flurry of token movement in Logs.

The power move: **a single transaction can emit many events across many contracts.** One signed transaction to a DEX router can fan out into internal calls to a pool and a wrapped-ETH contract, each emitting their own Transfer and Swap events, so one tx legitimately moves five tokens across three contracts — all under one hash, one fee, one signature.

![Graph showing one signed transaction fanning out into internal calls, token transfer logs, and a swap event across three contracts](/imgs/blogs/anatomy-of-a-transaction-3.png)

### Reading a swap's logs, step by step

Let us read the Logs of a Uniswap swap concretely. You swap USDC for ETH. On the Overview tab, Value reads "0 ETH" and the To field says "Uniswap V3: Router." The action is entirely in the Logs tab, which shows (decoded) something like three rows:

1. `Transfer` — from *your wallet* → *the USDC/ETH pool*, amount **5,000 USDC**. (Your stablecoins leaving.)
2. `Transfer` — from *the pool* → *your wallet*, amount **1.66 WETH**. (Wrapped ETH arriving.)
3. `Swap` — emitted by the *pool* contract, recording the trade's exact in/out amounts and the price.

Read top to bottom, those three logs *are* the swap: you paid USDC, you received WETH, the pool announced the trade. No prose needed — the receipt proves it. (You may also see a fourth internal step where WETH is unwrapped to native ETH; that shows in Internal Txns.) The before-and-after of your balances tells the same story from the wallet's point of view.

![Before and after of a swap showing the wallet and pool balances and the Transfer logs that record the move](/imgs/blogs/anatomy-of-a-transaction-5.png)

#### Worked example: a \$5,000 USDC → ETH swap, read from the logs

You swap exactly 5,000 USDC for ETH at a price of \$3,000/ETH. Reading the Logs tab:

- Log 1, `Transfer`: your wallet → pool, **5,000 USDC** out. Since USDC is a dollar-stablecoin, that leg is worth `\$5,000` exactly.
- Log 2, `Transfer`: pool → your wallet, **1.66 WETH** in. At \$3,000/ETH that is `1.66 × \$3,000 = \$4,980`.
- The `\$20` gap (`\$5,000 − \$4,980`) is the pool's fee plus price impact — for a 0.3% pool, `0.3% × \$5,000 = \$15` is the swap fee and the rest is slippage. The Logs prove the math: in 5,000 USDC, out 1.66 WETH, cost ≈ `\$20`.
- The gas on top — say `\$8` at a 150,000-gas swap and 18 gwei — is *separate* and shows in Transaction Fee, not in the Logs.

*Intuition: the two Transfer logs are the receipt of a trade, and the small dollar gap between the legs is exactly the fee plus slippage you paid the pool — read the logs and you never have to trust the dapp's "you received" number.*

### Under the decoded label: topics and data

Etherscan shows logs in a friendly "decoded" view by default, but there is a "Hex" toggle, and understanding the raw shape is what lets you read logs even when a contract is *unverified* and the explorer cannot decode it for you. A raw log has two parts:

- **Topics** — up to four 32-byte slots. Topic 0 is the **event signature hash** (a fingerprint of `Transfer(address,address,uint256)`, identical for every ERC-20 ever made, so you can recognize a Transfer even on an unverified token). Topics 1–3 carry the event's **indexed** parameters — for a Transfer, that is the `from` and `to` addresses, padded to 32 bytes. Indexed parameters are the ones you can **filter and search** on, which is exactly why `from` and `to` are indexed: analysts query "every Transfer to this address" across millions of blocks.
- **Data** — the non-indexed parameters, concatenated. For a Transfer, the `amount` lives here. It is not searchable on its own, but once you have found the event by its topics, the amount is right there in the data.

So a Transfer log, raw, is: topic 0 = the Transfer signature, topic 1 = padded `from`, topic 2 = padded `to`, data = the amount. Read that structure once and you will never be lost in a Hex log again, even on a brand-new token Etherscan hasn't decoded. This topics-vs-data split is also the foundation of every on-chain dashboard: tools like Dune query the logs table by topic 0 to find "all swaps" or "all transfers," then decode the data column. When you build a thesis on a dashboard later in this series, this is the layer it sits on.

A second practical note: the **order** of logs is the order the events were emitted during execution, which is the order things actually happened inside the transaction. For a complex DeFi action — a flash loan that borrows, swaps, repays, and pockets a profit all in one tx — reading the logs top to bottom replays the entire sequence step by step. The log order *is* the execution timeline.

## Internal transactions: value the contract moved

The **Internal Txns** tab confuses people because the name implies "secret transactions." They are not transactions at all in the strict sense — they are **value transfers that a contract made while executing your transaction**. They share your transaction's hash; they are not separately signed and they do not have their own gas. The explorer reconstructs them by tracing the execution.

Why do they exist as a separate tab? Because a contract can move *ETH* (not just tokens) as it runs, and that movement does not appear in the top-level `value` field. The classic case: you swap a token for ETH, the router gets WETH, and an internal step *unwraps* WETH and sends native ETH to your wallet. That final ETH delivery is an **internal transaction** — the contract moved the ETH, not you, so it lives here and not in the Overview value. If you only read the Overview, you would miss where the ETH actually ended up. The Internal Txns tab answers the question *where did the ETH really go?*

For a forensic analyst this tab is essential: when stolen funds are laundered through contracts, much of the value moves as internal transactions, and a tracer who only follows top-level value will lose the trail. The same is true for any DeFi action that routes ETH through intermediary contracts.

There is a subtlety worth internalizing here: a contract can call *another* contract, which calls another, several levels deep, all within your one signed transaction. Each of those calls is a **message call**, and any of them that carries ETH shows up as an internal transaction. This is why a single click on a DEX aggregator (which splits your trade across several pools to get a better price) can produce a dozen internal transactions and a screen full of logs — the aggregator contract called four pool contracts, each called a token contract, and ETH and tokens hopped between all of them. The Internal Txns tab lets you see that whole tree of value movement that the Overview's single `value` field hides. When you read a complex tx and the numbers do not add up from the Overview alone, the missing pieces are almost always in Internal Txns.

One more analyst habit: the `from` of an internal transaction is a *contract*, not an EOA. If you see value arriving at a wallet from a contract address with no obvious signed transaction, do not assume it is a mystery — check whether it is an internal transaction of some other tx, where a contract paid out (a withdrawal from a vault, a claim from an airdrop contract, a payout from a pool). The money has a source; it is just one level down.

## Input data and the method: what the transaction was *trying* to do

The **Input Data** field is the encoded instruction for a contract call. Raw, it is a long hex blob. The first 4 bytes (8 hex characters after the `0x`) are the **method ID** — a fingerprint of the function being called. The rest is the encoded arguments. Etherscan can usually decode this into a readable form when the contract's code is verified.

Click "Decode Input Data" and a swap might read:

```
 Function: swapExactTokensForETH(
   uint256 amountIn,         -- 5000000000  (5,000 USDC, 6 decimals)
   uint256 amountOutMin,     -- 1650000000000000000  (min 1.65 WETH)
   address[] path,           -- [USDC, WETH]
   address to,               -- 0xA11ce...  (your wallet)
   uint256 deadline          -- a unix timestamp
 )
```

Now you know not just *that* tokens moved, but *what the transaction asked for*: swap exactly 5,000 USDC, accept no fewer than 1.65 WETH (that `amountOutMin` is your slippage protection), route through the USDC→WETH path, deliver to your address, expire after the deadline. The decoded input is *intent*; the Logs are *outcome*. Reading both, you can see whether a transaction did what it claimed.

This is also how you spot trouble. A common drainer pattern is a transaction whose decoded method is `setApprovalForAll` or `increaseAllowance` to an unknown address, dressed up as a "claim your airdrop" click. If you had decoded the input *before* signing, you would have seen you were not claiming anything — you were handing over spending rights. Which brings us to the most important risk a beginner can read on-chain.

## Approvals: the most dangerous transaction you will ever sign

To let a DeFi protocol move your tokens — to swap, lend, or stake them — you first send an **approval**: a transaction calling the token's `approve(spender, amount)` function. It does **not** move any tokens. It writes a number into the token contract's storage: "this spender is allowed to move up to *this much* of my balance." Later, when you actually swap, the protocol calls `transferFrom` and pulls tokens up to that allowance — *without you signing again*.

That separation is the risk. An approval is a **standing permission** that outlives the transaction. If you approve an *unlimited* amount (which many dapps request by default, to save you from re-approving each time), the spender can move your *entire* balance, *any time in the future*, with no new signature from you. If that spender contract is malicious, or gets hacked, or was a phishing site all along, your tokens are gone — and the draining transaction will look perfectly valid on-chain, because you did authorize it, weeks ago, in one click.

![Graph showing an approval granting a spender a future allowance and a later transferFrom draining tokens with no new signature](/imgs/blogs/anatomy-of-a-transaction-6.png)

#### Worked example: the dollars at risk in an "unlimited" approval

You hold **\$25,000 of USDC** and you want to swap \$500 of it on a new DEX. The site pops an approval and, by default, requests `amount = unlimited`. You click approve. Reading that transaction on Etherscan:

- Method (decoded): `approve(spender = 0xDEX…, amount = 115792…)` — that giant number is the max uint256, i.e. **unlimited**.
- Tokens moved in this tx: **0**. Value: 0 ETH. It looks harmless; it is not.
- The standing risk: the spender can now `transferFrom` up to your full **\$25,000** at any future moment, not just the \$500 you meant to trade.
- The right move: approve only **\$500** (or a tight buffer), so the most a compromised spender can take is `\$500`, not `\$25,000`. The difference between an unlimited and a capped approval here is exactly **\$24,500** of avoidable exposure.

*Intuition: an approval's danger is not what it moves today (nothing) but what it permits forever — so the dollars at risk equal the allowance you grant, which is why "unlimited" is a \$25,000 decision dressed up as a free click.*

To audit your own standing approvals, open the **Token Approvals** checker on Etherscan (or a tool like Revoke.cash), paste your address, and you will see every spender you have ever granted and the allowance. Revoking sends a tiny transaction that sets the allowance back to 0. A defensive habit worth building: review and revoke stale approvals quarterly. The tooling for this lives across several products — the sibling post on [the on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) maps which tool does what.

## Failed transactions: why you pay for nothing

A transaction can **fail** — revert — and the surprise that catches everyone is that **a failed transaction still costs you gas**. The network ran your transaction up to the point it failed; that computation was real work for thousands of nodes, so you pay for it. The state changes are rolled back (you do not, say, lose the tokens you tried to swap), but the fee is gone.

Why do transactions fail? Common reasons: **slippage** (the price moved past your `amountOutMin` before your tx landed, so the swap reverts to protect you), **out of gas** (your gas limit was too low for what the contract needed), **a require/assert failing** in the contract (a condition wasn't met), or **insufficient allowance** (you tried to swap before approving). On the explorer, a failed tx shows a red "Fail" status and often an error reason like "Out of gas" or a revert message.

#### Worked example: a failed swap that burned \$12 in gas

During a volatile minute you submit a large swap. You set a tight `amountOutMin` and the price moves against you before your transaction is included. The contract checks the price, sees it would give you less than your minimum, and **reverts**.

- Status: **Fail**. Tokens moved: **0** (the swap rolled back — that part protected you).
- Gas used: ~95,000 (it ran most of the way before reverting) at an effective 35 gwei during the congestion.
- Fee burned: `95,000 × 35 gwei = 0.003325 ETH`. At \$3,600 ETH that is `0.003325 × \$3,600 ≈ \$12`.
- Net result: you are out about `\$12` and got nothing — no swap, no tokens, just a fee for the failed attempt. Submit it three more times chasing the price and you have burned `~\$48` for zero fills.

*Intuition: failure is not free because the network did the work either way, so in volatile conditions a string of reverted, fee-burning attempts can cost more than a single successful trade — which is why pros widen slippage tolerance or wait rather than spam-resubmit.*

## From one transaction to a chain of them: the analyst's leap

Everything so far has been a single transaction. On-chain *analysis* begins when you string transactions together — and the explorer is built for exactly that. Every address you click becomes its own page listing all of its transactions, and that is how you trace a story across time.

The `from` and `to` fields are your edges in a graph. The `from` address has a page; click it and you see every transaction that wallet ever sent and received, in order, with running balances. A whale "accumulating" is just a wallet whose transaction list shows repeated inbound Transfers from an exchange over days. A "smart-money" wallet is one whose history shows it bought a token before it ran. A drainer is a fresh address whose first inbound Transfers are other people's funds. The single transaction is the atom; the address page is the molecule; the multi-hop trace is the organism.

Two concepts make multi-transaction tracing precise. The first is **funding source**: the very first inbound transaction to a fresh wallet tells you *where it came from*. If a brand-new wallet's first deposit came from a known exchange withdrawal, you have a thread back toward a real identity (the exchange did KYC). If it came from a mixer, the thread goes cold by design. Investigators chasing the Bybit funds did exactly this — followed each fresh wallet back to its funding source and forward to its next hop.

The second is the **labeled counterparty**. When a transaction's `from` or `to` carries an Etherscan label — "Binance 14," "Coinbase: Hot Wallet," "Tornado Cash: Router" — you have anchored one end of the flow to a known entity. Funds flowing *into* an exchange-labeled address are potential sell pressure (someone is positioning to sell); funds flowing *out* are accumulation or withdrawal to self-custody. This is the raw material of exchange-flow analysis, and it all starts with reading the `from`/`to` labels on individual transactions.

#### Worked example: an exchange inflow read as supply

You are watching a token and you see a single transaction: a wallet sends **2,000,000 of token X** to an address Etherscan labels "Binance 14." The token trades at \$0.40.

- The dollar value moving onto the exchange: `2,000,000 × \$0.40 = \$800,000` of potential sell-side supply arriving where it can be sold.
- This does not *prove* a sale — the holder could be using the exchange as custody — but a large inbound transfer to an exchange-labeled wallet is the on-chain footprint that *precedes* selling, and it is visible to you before any price move shows it.
- Contrast: if the same `\$800,000` had moved *out* of "Binance 14" to a fresh self-custody wallet, that is the opposite signal — supply leaving the exchange, harder to sell quickly.
- The read scales with size: a `\$50,000` inflow is noise; a `\$800,000` inflow into a thin token is a position worth watching.

*Intuition: the same transaction reads as bullish or bearish depending only on the direction relative to the labeled exchange — inflows are latent supply, outflows are accumulation — and that directional read is available the instant the transaction confirms, ahead of the price.*

A caution that this series will repeat until it is reflex: **the chain shows you the move, not the motive.** An exchange inflow might be a sale, a transfer between an entity's own wallets, a market-maker rebalancing, or collateral being posted. The transaction is a *fact*; the interpretation is a *hypothesis*. Good on-chain analysis pairs every read with its alternative explanations and looks for confirming transactions before committing capital. The whole point of learning to read a single transaction this carefully is so that, when you scale up to flows and dashboards, you can always drill back down to the atom and check the claim yourself.

## A guided walkthrough: open these three transactions

Reading about tabs is one thing; the skill is in the fingers. Here is a literal, do-it-now pass through three transaction types. For each, the instruction is the same: paste the hash into Etherscan, then read the tabs in this order.

### (a) A plain ETH transfer

1. **Overview.** `From` is an EOA, `To` is an EOA (no contract icon), `Value` shows the ETH amount (e.g. 0.5 ETH), `Transaction Fee` is tiny (~\$1.58 at calm gas). Status: Success.
2. **Logs.** *Empty.* A plain transfer emits no events — there is no contract to emit them. An empty Logs tab is itself the confirmation that nothing fancy happened.
3. **Internal Txns.** Empty. No contract ran, so no contract moved value.
4. **Input Data.** `0x` (empty). No method, no arguments — there was nothing to instruct.
5. **The read:** "EOA A sent 0.5 ETH to EOA B, paid ~\$1.58, done." That is the entire story, and the empty Logs/Internal/Input tabs *prove* it was a simple transfer and nothing else.

### (b) A Uniswap swap (USDC → ETH)

1. **Overview.** `To` says "Interacted With: Uniswap V… Router" with a contract icon. `Value` is **0 ETH** (do not panic — your tokens move in Logs). `Transaction Fee` is larger (~\$5–15) because a swap is ~120,000–250,000 gas. Status: Success.
2. **Logs.** *Here is the trade.* You see the two `Transfer` events (USDC out, WETH in) and a `Swap` event from the pool, with exact amounts. This is where you read what you actually paid and received — not the dapp's UI, the chain itself.
3. **Internal Txns.** Often shows the WETH→ETH unwrap delivering native ETH to your wallet. This is where the ETH "really arrives," answering a question the Overview's 0-ETH value field couldn't.
4. **Input Data.** Decode it to see the method (`swapExactTokensForETH`) and arguments — your `amountIn`, your `amountOutMin` (slippage floor), the path, the deadline. Intent, in plain function form.
5. **The read:** "I sent 5,000 USDC to the router; the pool gave me 1.66 WETH; it was unwrapped to ETH and delivered to me; I asked for at least 1.65 and the trade respected it; I paid ~\$8 gas plus ~\$20 pool cost." Every claim is backed by a tab.

### (c) An approval

1. **Overview.** `To` is the **token contract** (e.g. USDC), not a DEX. `Value` is 0 ETH. `Transaction Fee` is small (~45,000 gas). Status: Success. *Nothing visibly moved* — and that is the trap.
2. **Logs.** A single `Approval(owner, spender, amount)` event. Read the `amount`: if it is `115792089237…` (max uint256), this is an **unlimited** approval. If it is a specific number, it is capped.
3. **Internal Txns.** Empty — no value moved.
4. **Input Data.** Decode to confirm `approve(spender, amount)` and *who* the spender is. Verify the spender is the protocol you intended, not a look-alike.
5. **The read:** "I granted spender 0xDEX… the right to move up to [amount] of my USDC, indefinitely, until I revoke it." If `amount` is unlimited and the spender is unfamiliar, that is your cue to revoke immediately.

Notice the pattern across all three: **the tabs are a checklist, and the empty ones carry as much information as the full ones.** An empty Logs tab on something that claims to be a swap is a red flag. A populated Approval log on something you thought was a "claim" is a red flag. Reading is partly knowing what *should* be there.

For the deeper mechanics of how tokens, transfers, and approvals work as a standard, the sibling post on [tokens on-chain: transfers and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals) goes under the ERC-20 hood; here we stay at the explorer.

## The other chains: Solana, Tron, and Bitcoin in brief

Etherscan is the template, but you will eventually open a transaction on another chain, and the field names shift. Here is the just-enough translation. The underlying account-vs-UTXO model differences are covered in depth in [how blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account); this is the practical "where do I click" version.

- **EVM chains (BSC, Polygon, Arbitrum, Base, etc.).** Same fields, same tabs, different explorer (BscScan, Polygonscan, Arbiscan, Basescan) — often run by the same company as Etherscan, so the page is nearly identical. The skill transfers one-to-one. Gas is paid in the chain's native token (BNB, MATIC/POL, ETH on L2s) and is usually far cheaper.
- **Tron (Tronscan).** Tron is the dominant rail for retail USDT transfers, especially in emerging markets, and unfortunately for a lot of illicit flow too. A Tron transaction page shows `Owner Address` (the sender), the contract called, and a "Token Transfers" section equivalent to Ethereum's Logs. Fees work differently: Tron uses **Energy** and **Bandwidth** (resources you can freeze TRX to obtain) rather than a gwei gas auction, so a USDT transfer can be near-free if you have staked for resources. The mental model — who, to whom, how much, what method — is identical; only the fee mechanics and the labels change.
- **Solana (Solscan / Solana Explorer).** Solana bundles multiple **instructions** into one transaction and can touch many **accounts** at once, so a Solscan page lists the instructions and the **token balance changes** (before/after per account) rather than an Ethereum-style logs array. For memecoin and smart-money flow, you read the SOL and SPL-token balance deltas. Fees are tiny (fractions of a cent) but you also pay small **rent** to store accounts. Same questions, very different layout.
- **Bitcoin (mempool.space / a block explorer).** Bitcoin has no accounts, no smart contracts in the EVM sense, and no events. A transaction is a set of **inputs** (UTXOs being spent) and **outputs** (new UTXOs created), and you read it by following the inputs and outputs. mempool.space is also the place to watch the **mempool** itself — pending transactions and the fee market — which is its own analytical signal. There is no "Logs" tab because there are no events; the transaction graph *is* the data.

The constant across all of them: a transaction records *who moved what to whom, at what cost, with what result*, and the explorer's job is to show you those facts. Learn to ask those four questions and you can read any chain's transaction page in minutes, even one you have never opened before.

A practical note on *bridges*, which is where most cross-chain confusion lives. When funds move from Ethereum to, say, Arbitrum or Solana, there is no single transaction that spans both chains — there cannot be, because each chain has its own ledger. Instead there are **two** transactions: one on the source chain (you deposit into the bridge contract) and a separate one on the destination chain (the bridge releases or mints funds to you), linked only by the bridge's own logic. To trace a flow across a bridge you read the deposit tx on chain A, note the amount and the destination, then find the corresponding release tx on chain B. This seam is exactly where laundering tries to break the trail and where investigators work hardest to reconnect it — a theme the series returns to, but one that starts with the humble skill of reading each individual bridge transaction on its own explorer.

## Explorer pro-tips that save you hours

A handful of practical habits separate someone who *can* read a transaction from someone who reads them fast and correctly.

- **Use the "Decode Input Data" and "Token Transfers" toggles every time.** The raw view is for when decoding fails; the decoded view is faster. If decoding is unavailable, the contract is **unverified** — itself a yellow flag for a token you are about to trade, because you cannot see what its code does.
- **Read the timestamp, not just the block.** When you are building a timeline of who did what, the wall-clock time and the "X mins ago" are what let you line a transaction up against an off-chain event (an announcement, a liquidation, a news headline). Timing is signal.
- **Click the addresses, don't just read them.** Half the value of the explorer is that every `from`/`to` is a link to that address's full history. The single transaction answers "what happened"; the address page answers "who is this and what else have they done."
- **Trust the math, not the dapp UI.** A dapp's "You will receive ~1.66 ETH" is an estimate before the trade; the **Transfer log after** the trade is what you actually got. When they disagree (the price moved, slippage ate more than shown), the log is the truth.
- **Watch for the difference between `value` and token amounts.** This trips up beginners daily: the big number you care about (the \$5,000 of USDC) is almost never in the `value` field — it is in Logs. Train your eye to skip past `Value: 0 ETH` and go straight to Logs on any contract call.
- **On a deposit you are crediting, wait for confirmations sized to the amount.** A \$50 tip can clear on one confirmation; a \$50,000 settlement should wait for finality. The explorer's confirmation counter is your gauge.

These are small, but together they turn the explorer from a wall of hex into a fast instrument. The goal is to glance at a transaction and, within a few seconds, state the four facts: who, to whom, how much, with what result — and know which tab proved each one.

## Common misconceptions

**"Value: 0 means nothing happened / my money didn't move."** The `value` field tracks *only native ETH*. A swap, an approval, a token transfer can all show `Value: 0 ETH` while moving thousands of dollars of *tokens* — that movement lives in the Logs (Transfer events) and Internal Txns, not in `value`. Always check Logs before concluding nothing moved. On a \$5,000 USDC swap, the Overview reads 0 ETH and the Logs carry the entire \$5,000.

**"A green Success check means it's final and safe."** Success means the transaction *executed without reverting*. It does not mean *finalized* (it may have only 1 confirmation and could, in a rare reorg, be undone), and it certainly does not mean *good for you* — a drainer's `transferFrom` that empties your wallet also shows a green Success. Success is "the code ran"; finality and safety are separate reads.

**"A failed transaction is free since nothing happened."** Wrong, and it is an expensive lesson. A failed tx ran real computation before reverting, so you pay for the gas burned — easily \$12 in a volatile minute, and \$48 if you resubmit four times. The state rolls back; the fee does not.

**"An approval moved my tokens."** An approval moves *zero* tokens. It writes a permission. The danger is precisely that it *doesn't* move anything now but lets someone move everything later — which is why people sign them carelessly and get drained weeks afterward. The dollars at risk equal the allowance, not the trade you intended.

**"The explorer's labels are the truth."** Etherscan's "Binance 14" or "Uniswap Router" labels are extremely useful but they are *Etherscan's attribution*, often community-sourced, and can be incomplete, stale, or occasionally wrong. They are a strong starting hypothesis, not gospel. Cross-check with the address's behavior and other tools before you build a thesis on a single label. For how addresses get attributed in the first place, see [addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts).

## The playbook: what to do with a transaction

When someone hands you a hash — "I think I got scammed," "did this payment land," "what did this whale actually do" — here is the if-then checklist that turns the explorer into answers.

**Signal → read → action → false-positive.**

- **"Did my payment / deposit land?"** → Open the hash, check **Status = Success** and the **confirmation count**. → For small amounts, 1+ confirmation is enough; for size, wait for finality (~13 min). → *False positive:* a green check at 1 confirmation is not yet final — don't release goods on a single-confirmation deposit for a large sum.
- **"What did this transaction actually do?"** → Read **To** (transfer vs contract call), then **Logs** (which tokens moved), then **Internal Txns** (where ETH went), then **decode Input Data** (the intent). → State it as "who moved what to whom, and was that the intent." → *False positive:* a 0-ETH value does not mean nothing happened — the action is in Logs.
- **"Is this approval dangerous?"** → Decode the input; read the `Approval` log's **amount** and **spender**. → If the amount is unlimited and/or the spender is unfamiliar, **revoke** it via Etherscan's Token Approvals or Revoke.cash. → *False positive:* a capped approval to a well-known router (e.g. a \$500 allowance to Uniswap) is normal and fine.
- **"How much did this cost?"** → Read **Transaction Fee** and expand for gas used × effective price; for a swap, add the pool fee + slippage you read out of the Logs. → Compare against the 21,000-gas / ~\$1.58 transfer baseline to judge if you overpaid. → *False positive:* a high fee during congestion is the auction price of block space, not a bug — check the base fee for that block.
- **"Did this transaction fail, and why?"** → Status = **Fail**; read the revert reason (Out of gas, slippage, require failed). → Fix the cause (raise gas limit, widen slippage, approve first) rather than blindly resubmitting and burning more fees. → *False positive:* a "replaced" or "dropped & replaced" tx is not a failure — it means a higher-fee version of the same nonce went through instead.

The meta-skill underneath all of this: **a transaction is a claim, and the explorer lets you verify it independently.** You never have to trust a dapp's confirmation screen, an exchange's email, or a Telegram screenshot. Paste the hash, read the tabs, and the chain tells you the truth. That habit — verify, don't trust the green number — is the foundation the rest of on-chain analysis is built on, and the through-line of this whole series.

## Further reading & cross-links

Within this series:

- [How blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account) — why Bitcoin transactions look so different from Ethereum's, the model under the field names.
- [Addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts) — what the `from` and `to` fields really point to, and how addresses get labeled.
- [Tokens on-chain: transfers and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals) — the ERC-20 standard behind the Transfer and Approval events you read in the Logs tab.
- [The on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — which explorer, indexer, and approval-checker to reach for, and when.

Foundational context:

- [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) — why Ethereum runs contracts at all, the substrate that makes a "contract call" possible.
- [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — the protocols behind the swaps and approvals you decoded above.
- [Stablecoins: Tether, Circle, the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) — the USDC and USDT whose Transfer logs you will read most often.
