---
title: "How Blockchains Store Data: UTXO vs Account, and Why It Changes Your Analysis"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Bitcoin stores spendable coins; Ethereum and Solana store account balances. Learn both data models from zero, and why the difference dictates how you read state, trace flows, and cluster owners."
tags: ["onchain", "crypto", "bitcoin", "ethereum", "solana", "utxo", "account-model", "tracing", "blockchain-state", "etherscan"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A blockchain's data model decides how you read it. Bitcoin stores no balances at all; your "balance" is just the sum of the unspent coins you hold. Ethereum and Solana store one running balance per account. That single difference changes how you trace money on each chain.
>
> - **The signal:** the *shape* of state. UTXO chains (Bitcoin, Litecoin, Bitcoin Cash) track a set of discrete unspent coins; account chains (Ethereum, all EVM L2s, Solana, Tron) track balances attached to addresses.
> - **How to read it:** on a UTXO chain you read a transaction's *inputs and outputs* and follow coins and change; on an account chain you read *from, to, value* and follow value between addresses.
> - **What you do with it:** pick the right tracing method per chain — coin-following and the common-input heuristic on Bitcoin, address-graph following on Ethereum. Using the wrong mental model is the single most common beginner mistake.
> - **The number to remember:** a Bitcoin wallet showing "1.6 BTC" is storing *zero* balance numbers — it is summing three coins of 0.4, 0.5 and 0.7 BTC that you can each only spend whole.

On 2022-03-23, attackers drained the Ronin Bridge — the cross-chain bridge behind the game Axie Infinity — of roughly **\$625M**, one of the largest crypto thefts ever recorded and later attributed by the FBI to North Korea's Lazarus Group. Investigators tracing those funds had to do something most price-chart traders never think about: they had to read the *state* of two completely different kinds of blockchain. The stolen ETH lived on Ethereum, an **account-model** chain where the loot sat as a balance attached to a wallet. As the launderers later peeled funds toward Bitcoin and through mixers, the same investigators had to switch mental models entirely, because Bitcoin does not store balances at all.

The pattern repeats at the very top of the loss table. The largest crypto theft ever recorded — roughly **\$1.46B** stolen from Bybit on 2025-02-21, again attributed to Lazarus — moved ETH and staked-ETH out of an account-model chain, and tracers followed those funds through swaps and bridges by walking an address graph. Had the same crew tried to cash out through Bitcoin, the chase would have flipped to coin-following and change-spotting. Two of the three biggest documented thefts in crypto history demanded fluency in *both* data models from the people chasing the money. That is not a coincidence; it is the job.

Here is the thing nobody tells a beginner: **you cannot trace money you do not understand**, and "understanding" starts one level below price, below even transactions — at how the chain stores who-owns-what. The blockchain is one giant, public, append-only ledger. But there are two dominant ways to organize the "who owns what" inside that ledger, and they are as different as a wallet stuffed with physical banknotes is from a bank statement with a single balance line. Get the model wrong and every downstream skill — tracing a hack, clustering a wallet, reading an exchange flow — quietly breaks.

This post builds both models from zero, with everyday analogies first and the real mechanics second, and then shows you the payoff: how the *same payment* looks different on each chain, and why that dictates how you trace it. By the end you should be able to open any chain's explorer, look at a transaction, and know exactly what you are reading.

![UTXO model holds a wallet of distinct coins while the account model holds one running balance](/imgs/blogs/how-blockchains-store-data-utxo-vs-account-1.png)

## Foundations: ledgers, blocks, transactions, and state

Before we can compare two data models, we need four words pinned down: **ledger**, **block**, **transaction**, and **state**. A beginner can build everything else on these.

### The ledger and the block

A **ledger** is just a record of transactions, in order. A medieval merchant kept one in a leather book: "Tuesday, received 3 florins from Marco; Wednesday, paid 1 florin to the baker." A blockchain's ledger is the same idea, with two superpowers: it is **public** (anyone can download and verify the whole thing) and it is **append-only** (you can add a new page, but you can never tear an old one out without everyone noticing).

The pages of that book are **blocks**. A block is a batch of transactions, bundled together roughly every 10 minutes on Bitcoin and every 12 seconds on Ethereum, stamped with a cryptographic fingerprint (a **hash**) of the block before it. Because each block points back to its parent, rewriting an old transaction would change every fingerprint after it — which is why the chain is tamper-evident. We will go deep on block contents in [the anatomy of a transaction](/blog/trading/onchain/anatomy-of-a-transaction); for now, a block is simply "a page of confirmed transactions, chained to the previous page."

A quick word on that fingerprint, because it is what makes the whole structure trustworthy. A **hash** is a function that takes any data — a transaction, a whole block — and returns a fixed-length string that acts like a tamper-evident seal. Change a single character in the input and the hash changes completely and unpredictably. Each block's header contains the hash of the previous block, so the blocks form a literal *chain*: block 800,001 commits to block 800,000, which commits to 799,999, all the way back to the very first "genesis" block. If a malicious actor wanted to alter a payment buried 100 blocks deep, they would have to recompute that block's hash and every hash after it, *and* outpace the entire honest network doing so in real time. That is economically hopeless on a large chain, and it is the entire security argument in one sentence: history is expensive to rewrite because every page is sealed to the next.

There is one more subtlety worth internalizing early, because it bites every new on-chain analyst: a transaction is only *probably* final, not instantly final. When you broadcast a transaction it first sits in the **mempool** — the network's waiting room of unconfirmed transactions — until a miner or validator includes it in a block. Even after inclusion, a freshly mined block can be displaced if two miners find blocks at nearly the same time (a brief "fork" the network resolves by following the longest chain). This is why exchanges wait for several **confirmations** (additional blocks stacked on top) before crediting a deposit: each new block makes a reversal exponentially less likely. For tracing, the practical rule is that recent transactions are tentative and deep ones are settled — never build a thesis on a single-confirmation transaction.

### The transaction and the state

A **transaction** is an instruction that changes who owns what. "Send 1 BTC from me to the merchant." "Move 5,000 USDC from Alice to Bob." A transaction is a *verb*; it does something.

The result of applying every transaction in order is the **state** — the noun, the current snapshot of who owns what. If the ledger is the history of every move ever made, the state is the scoreboard right now.

This ledger-versus-state distinction is worth slowing down on, because it is the hinge of the whole post. The **ledger** is immutable and grows forever — every transaction that ever happened is recorded and never deleted. The **state** is mutable and roughly bounded — it only needs to describe the present. A new node joining the network downloads the ledger (all of history) and *replays* it from the genesis block to reconstruct the current state. So in principle the state is redundant: you could always rederive it by replaying every transaction. In practice that would be far too slow, so nodes keep a live copy of the state and update it as each block arrives. The two data models in this post are simply two different *data structures* for that live state copy: a set of coins, or a table of balances. The ledger looks similar on both kinds of chain; it is the state representation that diverges.

Why does any of this matter to a trader or an investigator rather than a protocol engineer? Because the questions you ask are state questions and history questions, and the two models answer them differently. "How much does this whale hold right now?" is a state query. "Where did this exchange's outflow go over the last month?" is a history query that walks the ledger. On an account chain, the state query is a one-line lookup. On a UTXO chain, even the simple "how much does this address hold" requires summing every unspent coin tagged to it — there is no single number to read. That asymmetry shows up in every tool you will use.

And here is the fork in the road that this entire post is about: **how do you store that scoreboard?** Two answers dominate, and they are genuinely different.

- The **UTXO model** (Bitcoin's choice): the state is *a set of unspent coins*. There is no "balance" field anywhere. Your balance is something you compute by adding up the coins you can spend.
- The **account model** (Ethereum's and Solana's choice): the state is *a table of balances*, one row per address. Your balance is a number stored directly, that goes up and down.

A useful way to hold the contrast: a UTXO wallet is a coin purse full of distinct, indivisible bills and coins you must spend whole and get change for; an account is a checking account with one balance that rises and falls. Same money, two filing systems. Everything downstream — tracing, clustering, privacy, fees — flows from this choice.

### Why "no balance" sounds impossible (and isn't)

The first time someone hears "Bitcoin doesn't store balances," it sounds absurd. Your wallet app clearly shows a number. But the wallet *computed* that number. The chain itself stores only a giant pile of unspent coins, each tagged with the condition needed to spend it ("whoever can sign with this key"). When your wallet says 1.6 BTC, it scanned the pile, found the coins you can unlock, and added them up. The number lives in your wallet's view, never in the chain's state.

This is not a quirk; it is a deliberate design with real consequences for privacy, parallelism, and — the reason we care — *tracing*. So let us build each model properly.

## How the UTXO model works: coins, inputs, outputs, and change

UTXO stands for **Unspent Transaction Output**. Read it backwards: it is an *output* of some past *transaction* that has not yet been *spent*. Each UTXO is a discrete coin of a specific amount, locked to a spending condition (usually "signed by the holder of this private key").

Take a physical wallet. You have a \$0.40 coin, a \$0.50 coin, and a \$0.70 coin — three distinct pieces of metal. Your "balance" is \$1.60, but that balance is not engraved anywhere; it is just what you get when you count the coins. To pay for a \$1.00 sandwich, you cannot shave 1.00 off a coin. You hand over coins that *cover* the price — say the \$0.50 and \$0.70 — and the cashier gives you \$0.20 in change. Bitcoin works exactly like this, except the "coins" are UTXOs and the "change" is a brand-new UTXO sent back to you.

A Bitcoin transaction therefore has two lists:

- **Inputs:** references to existing UTXOs you are spending. Each input must be spent *whole* — you cannot partially spend a coin. Inputs are *destroyed* (marked spent) when the transaction confirms.
- **Outputs:** new UTXOs the transaction creates. One or more pay your recipient; usually one pays **change** back to an address you control.

The amounts must balance: `sum(inputs) = sum(outputs) + fee`. Whatever you do not send to the recipient and do not explicitly return as change is, by definition, the **miner fee**. (More than one beginner has accidentally paid a five-figure fee by forgetting the change output. The chain does not protect you from that.)

What actually *locks* each coin deserves a sentence, because it explains why a "coin" is more than an amount. Every UTXO carries a small **locking script** — a condition that must be satisfied to spend it. For an ordinary payment the condition is "provide a signature from the private key behind this address." But the script can be richer: "require 2 of these 3 signatures" (a multisig, used by exchanges and custodians), or "cannot be spent until block N" (a timelock). When you spend a UTXO, your input provides the matching **unlocking script** — usually a signature — that satisfies the lock. So a Bitcoin coin is not just "0.5 BTC"; it is "0.5 BTC spendable by whoever can satisfy this condition." That is why the common-input heuristic works later: to spend several coins together, the spender must satisfy *every* lock, which usually means controlling every key.

Two practical consequences of the coins-not-balances design surface immediately. First, **dust**: a UTXO so small that spending it would cost more in fees than it is worth. If you receive 0.00001 BTC (≈ \$0.60 at \$60,000/BTC) but spending it requires \$2 of fees, that coin is economically dead weight — "dust" your wallet may never bother to move. Spammers and de-anonymizers even exploit this with a "dusting attack," sending tiny coins to many addresses hoping the victim later consolidates them and links the addresses. Second, **UTXO-set growth**: because every payment creates new coins, the global set of unspent outputs keeps growing, and every full node must hold it in fast memory to validate new transactions. Managing the size of that set is a real engineering concern that account chains, which store one balance per address, sidestep entirely.

![Two Bitcoin coins are spent whole as inputs and the transaction mints a payment output and a change output](/imgs/blogs/how-blockchains-store-data-utxo-vs-account-2.png)

#### Worked example: spending coins and getting change

You hold three UTXOs: 0.4, 0.5 and 0.7 BTC. At a BTC price of \$60,000, that is 0.4 × \$60,000 = \$24,000, 0.5 × \$60,000 = \$30,000, and 0.7 × \$60,000 = \$42,000, for a total of \$96,000 of value sitting as three separate coins. You want to pay a merchant **1.0 BTC** (= \$60,000) and the network fee is **0.002 BTC** (≈ \$120).

You cannot spend 0.4 + 0.5 = 0.9 BTC — that is short of 1.0. So your wallet selects the 0.5 and 0.7 coins, totalling 1.2 BTC (= \$72,000). The transaction destroys those two inputs and creates two outputs: 1.0 BTC to the merchant, and 1.2 − 1.0 − 0.002 = **0.198 BTC** (≈ \$11,880) of change back to you. Your 0.4 BTC coin is untouched and remains spendable.

After this transaction your wallet shows 0.4 + 0.198 = **0.598 BTC** — but the chain stores no "0.598" anywhere; it stores two unspent coins (0.4 and 0.198) that your wallet sums. *The intuition: a UTXO balance is always a computed total of indivisible coins, never a stored number.*

### Why one transaction can have many inputs and outputs

This is the most important structural fact about UTXO chains, and it is the seed of an entire tracing technique. Because a payment usually requires combining several coins to cover the amount, a single Bitcoin transaction routinely lists **many inputs**. A large payment from a wallet with lots of small coins might consolidate dozens of UTXOs into one transaction. Likewise it can have **many outputs** — an exchange paying 500 withdrawals in one transaction (a "batched payout") has 500 outputs.

Picture how this looks at exchange scale. A custodian like a major exchange receives thousands of small deposits a day, each landing as a separate UTXO at a separate deposit address. To pay out withdrawals efficiently, it periodically *consolidates* hundreds of these small coins into one transaction (saving fees, because one large transaction costs less than many small ones) and *batches* withdrawals so one transaction settles hundreds of customers at once. The result is a transaction with, say, 300 inputs and 200 outputs — a structure you almost never see on an account chain, where one transaction has one sender and one logical recipient. When you open a Bitcoin explorer and see a transaction with dozens of inputs and outputs, you are very likely looking at exchange or custodial plumbing, and that shape itself is a label.

Hold that thought, because it is exactly what makes Bitcoin clustering work. We will return to it after we have the account model in hand.

## How the account model works: balances, nonce, storage, and gas

Now flip to Ethereum. Here the state is literally a giant table: each **account** (identified by an address) has a row, and that row stores a few fields directly. There are two account types:

- **Externally Owned Accounts (EOAs):** controlled by a private key — a normal user wallet. We unpack EOAs vs contracts fully in [addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts).
- **Contract accounts:** controlled by code, not a key — a smart contract.

Every account row stores, at minimum:

- **Balance:** the amount of ETH the account holds, as a single number (in wei, the smallest unit — 1 ETH = 10¹⁸ wei). This is the field Bitcoin does not have. A transfer *subtracts* from one balance and *adds* to another. The number is edited in place.
- **Nonce:** a counter of how many transactions this account has sent. It increments by 1 each time. Its job is to stop **replay** — without it, someone could rebroadcast your "send 1 ETH" forever. Nonce 12 means "this is your 13th transaction (0-indexed)"; the network rejects any transaction whose nonce is not exactly the next expected value.
- **Storage** (contracts only): a key-value store holding the contract's own data — for a token like USDC, this is the table of who-owns-how-many tokens. The contract's logic reads and writes this storage.
- **Code** (contracts only): the contract's bytecode. EOAs have none.

So when you "hold 5,000 USDC," your ETH balance is unchanged — what changed is a *number inside the USDC contract's storage* that says address `0xA11ce…` is owed 5,000 units. USDC is a contract; its balances live in its storage, not in the base-layer balance field. That distinction matters when you trace tokens, and we will use it below.

It helps to see how all these account rows are bound together. Ethereum keeps a single global structure called the **world state** — a giant map from every address to its account row (balance, nonce, storage root, code). That map is itself committed into one cryptographic fingerprint (a Merkle-Patricia trie root) that lives in each block header. The payoff is the same tamper-evidence we saw for blocks, now extended to *balances*: you cannot quietly inflate an account's balance, because doing so would change the world-state root, which is sealed into the block, which is sealed into the chain. So while the account model "edits numbers in place," those edits are still cryptographically committed at every block — the state is mutable in value but immutable in history. For an analyst, the convenient consequence is that you can ask a node "what was address X's balance at block N?" and get a verifiable answer, which is how dashboards reconstruct historical holdings.

The reason this model exists at all is **smart contracts**. Bitcoin's coins-with-locking-scripts can express simple conditions, but they were not designed to hold rich, evolving application state. Ethereum's whole pitch was *programmable money*: accounts that are programs, with their own persistent storage, that can hold balances of other tokens, run lending logic, or operate an exchange. A balance table is the natural fit for that — a contract needs to look up "how much does this user have?" in one step, mutate it, and move on. The account model is, in a sense, the database a smart-contract platform wanted. The tradeoff it accepts is weaker privacy by default, which we return to when we trace.

#### Worked example: native ETH vs token balance

Address `0xA11ce…` shows, on Etherscan, a native balance of 0.8 ETH (= \$2,400 at \$3,000/ETH) and a token balance of 5,000 USDC (= \$5,000). These two numbers live in *different places*: the 0.8 ETH sits in the base-layer balance field of the account row, while the 5,000 USDC is an entry in the USDC contract's storage keyed by this address. If Alice spends gas, only the ETH number drops; if she sends USDC, only the contract storage entry changes. A naive reader who only checks the native balance would value this wallet at \$2,400 and miss the \$5,000 of stablecoins entirely. *The intuition: an account's "holdings" are scattered across the base layer and every token contract it touches, so a full balance read means querying many contracts, not one field.*

![An account transfer subtracts 5,000 USDC from the sender and adds it to the receiver in place](/imgs/blogs/how-blockchains-store-data-utxo-vs-account-3.png)

#### Worked example: an account-model transfer of 5,000 USDC

Alice holds 8,000 USDC (= \$8,000, since USDC targets a \$1 peg) and Bob holds 2,000 USDC (= \$2,000). Alice sends Bob **5,000 USDC** (= \$5,000). Before: Alice 8,000, Bob 2,000. The USDC contract runs its `transfer` logic, which checks Alice's balance ≥ 5,000, then **subtracts** 5,000 from Alice's storage slot and **adds** 5,000 to Bob's.

After: Alice 8,000 − 5,000 = 3,000 USDC (= \$3,000), Bob 2,000 + 5,000 = 7,000 USDC (= \$7,000). No new coins were created and none were destroyed; two numbers were edited. Alice's *nonce* also ticked from 12 to 13, recording that she sent a transaction. *The intuition: an account transfer is two balance edits, the way a bank moves money between two account lines on a statement.*

### Gas: the unit of computation

Bitcoin charges fees roughly by transaction *size* (how many bytes the inputs and outputs take up). Ethereum charges by *computation*, and the unit is **gas**. Every operation a transaction performs — adding two numbers, writing to storage, calling another contract — costs a fixed amount of gas. A plain ETH transfer between two EOAs costs exactly **21,000 gas**. A token transfer or a DeFi swap costs much more because it runs contract code and writes storage.

You pay for gas in ETH, at a **gas price** quoted in *gwei* (1 gwei = 10⁻⁹ ETH). Total fee = `gas used × gas price`. Solana uses an analogous but cheaper model priced in **lamports** (1 SOL = 10⁹ lamports), with an optional priority fee. We dig into fee markets and priority auctions in [crypto mining, staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev); here we just need the unit.

The deeper point is *why the fee unit differs by model*, because it is the same logic that drives the data model. A UTXO chain validates by checking signatures on coins and confirming amounts balance — work that scales with how many bytes the transaction occupies on disk and over the wire. So Bitcoin prices fees by **virtual size** (sat/vB), and a transaction with many inputs is more expensive precisely because each input adds bytes. An account chain runs arbitrary contract code, so its cost is dominated by *computation and storage writes*, not raw size — and gas is just a meter on that work, with writing a new storage slot among the most expensive operations (which is why minting a fresh token balance costs more than updating an existing one). The fee model is downstream of the state model: pay for bytes when state is coins, pay for computation when state is a programmable table.

There is also a third major account-model chain that every tracer must know: **Tron**. Tron is the dominant rail for retail USDT and, unfortunately, a large share of illicit stablecoin flow, because its fees are negligible and its throughput high. Tron uses an account model very similar to Ethereum's, but meters cost in **Energy** and **Bandwidth** rather than gas, and lets users stake the native token to obtain these resources cheaply or for free. For analysis purposes you read Tron the way you read Ethereum — from, to, value, and token-transfer logs — using an explorer like Tronscan; the model is account-based, so you follow value along an address graph, not coins. The reason it matters in this series is volume: an enormous fraction of stablecoin transfers, legitimate and otherwise, settle on Tron, so a complete tracing picture often means following funds across Ethereum *and* Tron.

#### Worked example: the cost of a simple ETH transfer in dollars

A plain ETH transfer uses 21,000 gas. Say the gas price is **20 gwei** and ETH is **\$3,000**. The fee in ETH is 21,000 × 20 gwei = 420,000 gwei = 420,000 × 10⁻⁹ = **0.00042 ETH**. In dollars that is 0.00042 × \$3,000 ≈ **\$1.26**.

Now a token transfer that uses ~65,000 gas at the same 20 gwei costs 65,000 × 20 × 10⁻⁹ × \$3,000 ≈ **\$3.90**, and a busy-day swap burning 200,000 gas at 50 gwei costs 200,000 × 50 × 10⁻⁹ × \$3,000 = **\$30**. *The intuition: on an account chain you are not paying for bytes, you are paying for the computation the contract runs — so the more code your transaction touches, the more it costs.*

## Solana's account model: everything is an account

Solana is also an account-model chain, but it pushes the idea further, and the differences matter for anyone tracing Solana memecoin flows. On Ethereum a contract bundles *code and its data together* in one account. On Solana the two are split:

- **Programs** are accounts that hold executable code but are otherwise **stateless** — they store no balances or app data of their own.
- **Data accounts** hold the state, and each is **owned by** a program. Only the owning program can change a data account's contents. Your token balance on Solana lives in a data account (a "token account") owned by the SPL Token Program, not inside your wallet account.

Every Solana account — wallet, program, or data — carries the same fields: a **lamports** balance, an **owner** (which program controls it), a **data** blob, and an **executable** flag. There is one more twist: **rent**. To stop the chain bloating with abandoned accounts, Solana requires an account to keep a minimum lamport balance (a rent-exempt deposit). Fall below it and the account can be purged. Finally, a Solana transaction must **declare every account it will read or write up front** — which lets the runtime execute non-overlapping transactions in parallel, the root of Solana's speed.

![On Solana wallets code and data are all accounts and programs own the data accounts they mutate](/imgs/blogs/how-blockchains-store-data-utxo-vs-account-5.png)

Why split code from data this way? Two reasons that both pay off for the chain's headline feature, speed. First, because a program is stateless and the same code serves everyone, you do not redeploy logic per user — one Token Program governs every SPL token account in existence. Second, and more importantly, because every transaction declares the exact accounts it will touch, the runtime can look at two transactions, see that their account sets do not overlap, and **execute them at the same time on different cores**. That parallelism is impossible on Ethereum, where a transaction can call any contract and touch any state, so transactions must be processed one after another to be safe. Solana trades the convenience of "just call whatever you want" for the throughput of "tell me up front what you'll touch, and I'll run everyone who doesn't collide in parallel."

For a tracer, the practical upshot is: on Solana you do not just follow "address to address." A token transfer touches the sender's *token account*, the receiver's *token account*, and the Token Program — so reading a Solscan transaction means reading a list of accounts and which program touched each. It is still an account model (follow value between owners), but the owner-of-an-owner structure means you sometimes trace through a token account back to the wallet that controls it. Concretely: a memecoin trade you are following might show the value moving between two token accounts you have never seen, and your job is to resolve each token account to the *wallet* that owns it before you can say whose money moved. We will lean on this when we cover [address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics).

#### Worked example: rent on a Solana token account

To hold a token on Solana, you open a token account, and it must stay rent-exempt — currently around 0.002 SOL of deposit. At a SOL price of \$150, that is 0.002 × \$150 = **\$0.30** locked per token account, refundable when you close the account. A wallet that has interacted with 50 different memecoins might carry 50 token accounts, tying up 50 × \$0.30 = **\$15** in rent deposits — small, but it explains why bots routinely *close* dust token accounts to reclaim the SOL, an action that itself shows up on-chain. *The intuition: on Solana even "holding" a token costs a refundable deposit, so the open-and-close pattern of token accounts is part of what you read when tracing a wallet's activity.*

## The same payment, two chains: a side-by-side

The cleanest way to feel the difference is to send the *same conceptual payment* — "pay a merchant the equivalent of \$60,000, keep the rest" — on each model and watch what the chain records. This is the single exercise that makes the whole distinction click, so it is worth doing slowly: same human intent, same dollar amount, two completely different on-chain footprints.

**On Bitcoin (UTXO).** Your wallet finds coins that cover 1.0 BTC plus fee. It picks the 0.5 and 0.7 BTC coins (1.2 BTC, = \$72,000). The transaction *destroys* those two coins and *creates* two coins: 1.0 BTC to the merchant and 0.198 BTC change back to a fresh address of yours. The chain now has two fewer UTXOs of yours and two new ones in the world. Nobody's "balance number" was edited — the set of coins changed.

**On Ethereum (account).** Your wallet sends a transaction that *subtracts* the value from your balance row and *adds* it to the merchant's balance row (minus gas, which is burned/paid to the validator). Your nonce increments. No coins are created or destroyed; two numbers move and a counter ticks. There is no "change," because there is no coin to break — you simply send the exact amount and your remaining balance stays put.

That asymmetry — *destroy-and-recreate coins* vs *edit two numbers* — is the whole ballgame for analysis. It is why Bitcoin tracing is about **following coins and change through a forest of inputs and outputs**, while Ethereum tracing is about **following value along an address graph**.

The same payment also leaves a different *privacy footprint*, and this is not an accident — it falls out of the model. On Bitcoin, a well-behaved wallet uses a **fresh address for every receive and every change output**, so a casual observer sees a stream of one-off addresses rather than a single reusable identity. There is no global "Alice's balance" to point at; her money is scattered across many coins at many addresses, and linking them requires inference (the heuristics we are about to cover). On Ethereum, by contrast, the natural unit *is* a reusable address — your wallet, your ENS name, the thing you paste to receive funds. Reusing one address is the norm, which makes the account graph far easier to walk: every transaction you ever made hangs off the same node. The UTXO model leans private-by-default-but-inferrable; the account model leans transparent-by-default-but-convenient. Neither is "more anonymous" in an absolute sense — they fail differently, and a skilled tracer beats both — but the *shape* of the privacy is a direct consequence of whether state is coins or balances.

### A deliberate tradeoff, not a winner

It is tempting to rank the two models, but professionals treat them as a genuine tradeoff with no universal winner. The UTXO model buys: cleaner privacy defaults, trivially parallel validation (each coin is independent, so a node can verify many inputs at once), a stateless-ish validation model where you only need the UTXO set rather than full account history, and a smaller surface for subtle state bugs. Its costs: clumsy for rich application state, a UTXO set that grows without bound, and balance reads that require summing coins. The account model buys: natural smart-contract state, one-step balance reads, intuitive "send X to address Y" semantics, and easy reasoning for developers. Its costs: weaker privacy defaults, sequential execution on Ethereum (Solana's account-declaration trick is precisely an attempt to recover the parallelism UTXO chains get for free), and a global mutable state every node must track. Bitcoin keeps UTXO on purpose because it optimizes for being sound money with strong settlement and privacy properties; Ethereum chose accounts because it optimizes for being a world computer. The lesson for an analyst is to stop asking "which is better?" and start asking "what does *this* model make easy or hard to see?"

### Coin selection and the change address

One more UTXO mechanic deserves its own figure, because it is where privacy and tracing collide. When your wallet builds a transaction, it runs **coin selection**: an algorithm that picks which of your UTXOs to spend to cover the target amount while minimizing fee and leftover dust. Then it routes the **change** — almost always to a *brand-new address* your wallet generated, not back to the input address. Good wallets never reuse addresses.

![A wallet selects coins to cover a target then routes the leftover change to a fresh address](/imgs/blogs/how-blockchains-store-data-utxo-vs-account-6.png)

#### Worked example: identifying the change output

You spend the 0.5 and 0.7 BTC coins (1.2 BTC, = \$72,000) to pay 1.0 BTC (= \$60,000) with a 0.002 BTC (≈ \$120) fee. The transaction has two outputs: **1.0 BTC** to one address and **0.198 BTC** (≈ \$11,880) to another. Which is the payment and which is the change?

A tracer reasons: the 0.198 BTC output goes to an address that has never appeared on-chain before and that the spender controls (it pays itself), while 1.0 BTC is a clean round number typical of a deliberate payment. The non-round "leftover" output to a fresh address is the **change heuristic** flag — a strong (not certain) signal that 0.198 BTC stayed with the original owner. *The intuition: change outputs leak ownership, which is why mistaking change for a payment is the classic Bitcoin-tracing error.*

## Why this matters for tracing: clustering and the common-input heuristic

Now we cash in the structural fact from earlier — that a UTXO transaction can have many inputs. This is the engine behind Bitcoin's most powerful, oldest clustering technique: the **common-input-ownership heuristic**.

The logic is simple. To spend several UTXOs in one transaction, *someone has to provide a valid signature for each input*. The cheapest explanation for "all these coins got signed into the same transaction" is "one party controls all of them." So if a transaction spends a coin from address 1, a coin from address 2, and a coin from address 3, a tracer infers that addresses 1, 2 and 3 belong to the **same owner cluster**. Repeat across millions of transactions and you build a graph that collapses thousands of addresses into a handful of entities — exchanges, mixers, merchants. This is the core of how firms like Chainalysis and Elliptic deanonymize Bitcoin flows.

![Three coins from different addresses spent together in one transaction imply one owner cluster](/imgs/blogs/how-blockchains-store-data-utxo-vs-account-4.png)

#### Worked example: clustering by common input

A transaction spends three coins — 0.4, 0.5 and 0.7 BTC (= \$24,000, \$30,000 and \$42,000 at \$60,000/BTC, totalling **\$96,000**) — each previously sitting at a different address. Because all three had to be signed into one transaction, the heuristic clusters those three addresses as one wallet controlling \$96,000. If one of those addresses later receives a deposit from a known exchange withdrawal address, the *whole cluster* inherits the "linked to that exchange account" label — and the owner's other two addresses are now deanonymized too. *The intuition: on UTXO chains, spending coins together is a confession of common ownership, which is why coin consolidation is the tracer's best friend.*

This heuristic has limits and countermeasures — **CoinJoin** transactions deliberately mix many unrelated users' inputs into one transaction precisely to *break* the assumption, producing a transaction where the common-input inference is false. A careful analyst treats common-input clustering as strong evidence, not proof, and checks for CoinJoin structure (many equal-sized outputs) before trusting it. We go deep on these failure modes in [address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics).

It is worth being precise about *why* a CoinJoin defeats the heuristic, because the same logic tells you how to detect it. In a CoinJoin, dozens of strangers each contribute an input and each receive an equal-sized output, all in one transaction. The common-input heuristic would wrongly cluster all those strangers as one owner — but the giveaway is the structure: many inputs *and* a block of identical output amounts (say 30 outputs of exactly 0.1 BTC each). Ordinary payments almost never produce that pattern. So a tracer's rule is: count the inputs, then check whether the outputs cluster around one or two repeated values. Equal-value outputs in bulk → suspect a CoinJoin → do not apply common-input clustering to that transaction. This is the defensive posture the whole series takes: the heuristic is a sharp tool, but you must know the one transaction shape that turns it against you.

#### Worked example: a Bitcoin fee priced by size

Bitcoin fees are paid by virtual size, not computation. Say the network is busy at **50 sat/vB** and your two-input, two-output transaction is about 220 virtual bytes. The fee is 220 × 50 = 11,000 satoshis = 0.00011 BTC, which at \$60,000/BTC is 0.00011 × \$60,000 ≈ **\$6.60**. Now suppose you had instead consolidated *ten* small coins: more inputs means more bytes (≈ 700 vBytes), so the fee jumps to 700 × 50 = 35,000 sat = 0.00035 BTC ≈ **\$21** — over 3× more, for the *same* amount sent. *The intuition: on a UTXO chain you pay for the bytes your coins occupy, so a wallet full of tiny coins is expensive to spend, which is exactly why consolidation (and its clustering exposure) happens.*

### Tracing on the account model

On Ethereum there is no common-input heuristic, because there are no coins to bundle — a transaction comes *from* exactly one account and goes *to* one. Clustering instead relies on different tells: an address that **reuses itself** across many transactions (the norm on Ethereum, unlike Bitcoin's fresh-address habit) is trivially followable; funding patterns (which address first sent gas money to a fresh wallet) link wallets; and behavioral fingerprints (always using the same DeFi router, always at the same time of day) tie addresses to one operator. The trace is a *directed graph of value between addresses*, and because Ethereum users reuse addresses, the graph is often easier to walk than Bitcoin's — at the cost of weaker privacy by default.

A concrete account-model tell worth naming is the **funding heuristic**. Every Ethereum wallet needs ETH to pay gas before it can do anything, and a fresh wallet has none. So someone has to send it that first dollar of gas — and that first funder is a powerful link. If a brand-new wallet receives its first 0.05 ETH (= \$150) from a known exchange withdrawal, the wallet is probably operated by whoever owns that exchange account. If ten fresh wallets all get their initial gas from the same source within minutes, they are very likely one operator running a fleet — a pattern you see constantly in airdrop farming, wash trading, and laundering. None of this exists on Bitcoin in the same form, because there is no separate "gas balance" to seed. The model dictates the heuristic.

The other account-model trap is that the "address" you are following is often a **contract**, not a person. When value flows *to* an exchange's deposit contract, a DEX router, or a bridge, it does not simply sit there — the contract may fan it out internally, swap it, or hold it in a pooled balance shared by thousands of users. Walking "to-address" naively leads you into a contract and stops. The fix is to switch from reading *transfers* to reading the contract's **event logs**, which record what it did with the value (which user it credited, which pool it routed through). Knowing when you have hit a contract versus a person is half the skill of account-model tracing, and we make it concrete in the walkthrough below.

The single most useful table to memorize is the side-by-side of how the model changes each analysis task.

![A six-row comparison of UTXO and account models across balance reading tracing clustering privacy and fees](/imgs/blogs/how-blockchains-store-data-utxo-vs-account-7.png)

## How to read it: a walkthrough on two explorers

Theory is cheap. Let us read the same conceptual transaction on each chain's standard explorer and name every field, so you can do this yourself in five minutes.

### Reading a Bitcoin transaction (a block explorer)

Open any Bitcoin block explorer (mempool.space, Blockchair, blockstream.info) and look at a transaction page. You will see two columns:

- **Inputs (left).** Each input shows an address and an amount — these are the UTXOs being *spent*. If you see three inputs, three coins are being consumed, and (common-input heuristic) one owner likely controls all three. Each input also links back to the previous transaction that created that coin — this back-link is how you walk *upstream* to where the money came from.
- **Outputs (right).** Each output shows a destination address and an amount — these are the *new* UTXOs. One is usually the real payment; one is usually change. Look for the round-number payment vs the odd-leftover change. Each output also shows whether it has been *spent yet* — an unspent output is a live coin sitting in the current state.
- **Fee.** Shown as the difference `sum(inputs) − sum(outputs)`, often quoted in satoshis-per-virtual-byte (sat/vB) — Bitcoin's "gas price" analog.

What you are reading is a *coin-flow*: coins in on the left, coins out on the right. To trace, you click an output to follow the coin forward, or click an input's source to follow it back. There is no "balance" anywhere on the page — only coins and their amounts.

A few signals to read off that page like a professional. If the transaction has **many inputs**, mentally cluster their addresses as one owner (common-input heuristic). If it has **exactly two outputs**, ask which is the change: the one to a never-before-seen address, often a non-round amount, is the likely change that stayed with the sender. If it has **hundreds of inputs and outputs**, you are probably looking at exchange or custodial batching, not a single user's payment. And if you see **many equal-valued outputs alongside many inputs**, pause — that is the signature of a CoinJoin, where the common-input heuristic is deliberately broken and clustering would be wrong. The same page that looks like a wall of hex to a beginner tells a trained reader who owns what and where it is headed.

#### Worked example: reading inputs, outputs, and fee on Bitcoin

A transaction page shows two inputs (0.5 and 0.7 BTC) and two outputs (1.0 BTC and 0.198 BTC). You compute the fee yourself: (0.5 + 0.7) − (1.0 + 0.198) = 1.2 − 1.198 = **0.002 BTC** ≈ \$120 at \$60,000/BTC. The 1.0 BTC output is flagged "unspent" (the merchant has not moved it), the 0.198 BTC change output is flagged "spent in a later transaction" (the owner kept spending). *The intuition: on a Bitcoin explorer you are always reading coins-in vs coins-out, and the fee is the gap you can compute by hand.*

### Reading an Ethereum transaction (Etherscan)

Now open Etherscan and look at a transaction. The layout is completely different, because the state is completely different:

- **From / To.** A single sender address and a single recipient (an EOA or a contract). No "inputs" list — value comes from exactly one account.
- **Value.** The amount of native ETH moved (often `0 ETH` when the transaction is a token transfer or contract call, because the *token* movement lives inside the contract's logs, not in the native value field).
- **Transaction Fee & Gas.** Shows gas used, gas price (gwei), and the total fee in ETH and dollars — exactly the `gas used × gas price` from our worked example.
- **Nonce.** The sender's transaction counter for this transaction.
- **Tokens Transferred / Logs.** For an ERC-20 transfer like USDC, Etherscan parses the contract's event logs and shows you a human line: "5,000 USDC From `0xA11ce…` To `0xB0b…`." *This* is where the token movement actually lives — in the contract's storage edit, surfaced via its log — not in the top-line ETH value.

To trace on Etherscan you click the **To** address, open its page, and read its transaction list and token-transfer history — you are walking the address graph. There are no coins to follow and no change to identify; you follow value from address to address. We turn this into a full step-by-step money-trace in [how to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow).

The professional reflexes here mirror the Bitcoin page but invert the targets. First, check whether **To** is a contract or an EOA — Etherscan flags contracts and shows their code tab; if it is a contract, stop reading transfers and start reading **Logs**, because the value may have been swapped, pooled, or credited to a different user internally. Second, ignore the headline **ETH Value** and go straight to **Tokens Transferred** for the real money on the vast majority of transactions. Third, use the address's **Internal Transactions** tab: when a contract moves ETH on a user's behalf (common in DeFi), that movement does not appear as a normal transaction and is invisible unless you open internal transactions. Fourth, note **address reuse** — an address with thousands of transactions is one persistent identity you can profile (its first funder, its favorite protocols, its active hours), which is exactly the surface Bitcoin's fresh-address habit denies you. The account model gives the tracer a stable handle; the work is deciding when that handle is a person and when it is a contract hiding the next hop in its logs.

#### Worked example: reading value and fee on Etherscan

An Etherscan page shows From `0xA11ce…`, To the USDC contract, Value `0 ETH`, and under "Tokens Transferred": 5,000 USDC (= \$5,000) to `0xB0b…`. The fee line reads gas used 52,000 at 18 gwei = 52,000 × 18 × 10⁻⁹ = 0.000936 ETH ≈ **\$2.81** at \$3,000/ETH. The native Value is `0 ETH` because no ETH moved — the \$5,000 of value is a number edited inside the USDC contract. *The intuition: on Etherscan the real money is often in the token-transfer line, not the ETH value field, so reading only the top number misses the payment entirely.*

## Common misconceptions

**"My Bitcoin balance is stored on the blockchain."** No. The chain stores a set of unspent coins locked to your keys; your wallet sums them to show a balance. There is no balance field for a Bitcoin address anywhere in the protocol. The number you see is always computed — the same wallet imported into two apps with different coin-selection views can even *display* differently while controlling identical coins.

**"On Ethereum, the change output is the leftover, like Bitcoin."** There is no change on Ethereum. You send an exact amount; your balance is decremented by value + gas, and what remains stays in your single balance row. "Change" is a UTXO-only concept. Looking for a change output on Etherscan is a sign you brought the wrong model.

**"More inputs means a richer or more important wallet."** Many inputs usually means the opposite of sophistication — a wallet holding lots of small coins (often from many small deposits) must consolidate them to make a payment, which also *maximally* exposes it to common-input clustering. A privacy-conscious holder avoids needless consolidation. Input count is a structural fact, not a wealth signal.

**"The ETH `Value` field shows how much money moved."** Only for native-ETH transfers. For the overwhelming majority of on-chain activity — stablecoin sends, swaps, NFT trades — `Value` is `0 ETH` and the real value sits in the contract's logs. A \$5M USDC transfer shows `0 ETH` in the value field. Read the token-transfer/logs section, not the headline value, or you will undercount flows by orders of magnitude.

**"UTXO is old and obsolete; account is the modern way."** Both are alive and chosen on purpose. UTXO gives better privacy defaults (fresh addresses, no global balance to link), trivial parallel validation (each coin is independent), and a smaller attack surface for state bugs. The account model gives natural smart-contract state and simpler balance reads. Bitcoin keeps UTXO deliberately; it is a tradeoff, not a relic. The clearest evidence it is no relic: Solana, one of the newest high-throughput chains, leaned *into* the account model's parallel-friendly variant precisely to recover the concurrency that UTXO chains get naturally — modern engineering borrowing from both lineages, not abandoning one.

**"Clustering on Bitcoin gives you a name."** It gives you a *cluster*, not an identity. The common-input heuristic links addresses to one owner, but "owner" is still pseudonymous until something ties the cluster to a real entity — a regulated exchange's know-your-customer record, a public donation address, a leaked database, or an off-chain mistake. On-chain analysis narrows the suspect pool to a single controlling wallet; the last mile to a name almost always comes from outside the chain. Treating a cluster label as a confirmed identity is how analysts produce confident-but-wrong attributions, and it is why this series keeps repeating that on-chain evidence is strong but not omniscient. Frame clusters as leads to verify, never as proof on their own.

## The playbook: what to do with the data model

This is the if-then checklist. Before you trace, cluster, or read any flow, identify the model and switch tools accordingly.

- **Signal:** you are about to read or trace activity on a chain. **Read:** which model is it? Bitcoin / Litecoin / BCH / Dogecoin = **UTXO**; Ethereum + every EVM L2 (Arbitrum, Base, Optimism, Polygon) + Solana + Tron = **account**. **Action:** pick the matching explorer and mental model before clicking anything. **False positive:** assuming "it's crypto, so it's all the same" — that is exactly how beginners misread state.

- **Signal:** a Bitcoin transaction with multiple inputs. **Read:** common-input-ownership — those input addresses are probably one owner. **Action:** cluster them and inherit any label from any one address across the whole cluster. **Invalidation:** the transaction has many equal-sized outputs and many inputs of equal size → likely a **CoinJoin**, where the heuristic is deliberately false; do not cluster.

- **Signal:** a Bitcoin transaction with two outputs, one round and one odd. **Read:** the round number is probably the payment; the odd "leftover" to a fresh address is probably **change** that stayed with the sender. **Action:** follow the change output to keep tracing the original owner. **Invalidation:** both outputs go to previously-seen addresses, or the "odd" one is the round one — the change heuristic is weak; corroborate with address reuse and timing before trusting it.

- **Signal:** an Ethereum/Etherscan transaction showing `0 ETH` value. **Read:** the real movement is a token transfer or contract call inside the logs, not in the value field. **Action:** read "Tokens Transferred" and the event logs to find the actual money; convert token amounts to dollars. **False positive:** treating `0 ETH` as "no money moved" — a \$5M USDC transfer also shows `0 ETH`.

- **Signal:** you want to follow money on an account chain. **Read:** it is an address graph — value flows from one address to one address, and addresses are often reused. **Action:** click **To**, read that address's history, repeat; use funding source and behavioral fingerprints to link wallets. **Invalidation:** the destination is a contract (an exchange's deposit contract, a DEX router, a bridge) — value may fan back out internally; switch to reading the contract's logs rather than its top-line transfers.

- **Signal:** you are estimating a transaction's cost or a flow's friction. **Read:** UTXO fee ≈ size in vBytes × sat/vB; account fee = gas used × gas price (ETH) or compute units × price (Solana lamports). **Action:** quote fees in dollars to compare across chains — a 200,000-gas swap at 50 gwei on \$3,000 ETH is ≈ \$30, versus cents on Solana. **Invalidation:** congestion spikes gas/priority fees 10×+; never assume a stale fee level when timing a move.

## Further reading & cross-links

The data model is the foundation; the rest of this series builds tracing, clustering, and signal-reading on top of it. If you take one habit from this post, make it the reflex you now have: before you read any chain, name its model out loud — coins or balances — and pick your tools to match. Everything that follows assumes you can do that in your sleep.

- [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision) — why Bitcoin chose the UTXO model and a privacy-by-fresh-address ethos.
- [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) — why the account model unlocks smart contracts and on-chain state.
- [Anatomy of a transaction](/blog/trading/onchain/anatomy-of-a-transaction) — inside a block and a transaction, field by field, on both models.
- [Addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts) — EOAs vs contracts, how addresses are derived, and what an address really is.
- [Address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics) — the common-input and change heuristics in depth, plus how CoinJoin and mixers fight them.
- [How to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow) — a hands-on, end-to-end money-trace across both models and a bridge.
