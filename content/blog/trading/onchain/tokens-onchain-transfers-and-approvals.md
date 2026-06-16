---
title: "Tokens On-Chain: Transfers, Approvals, and the Risk Hiding in Allowances"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "What a token actually is on-chain — a contract holding a balance ledger — and how to read Transfer and Approval events, spot a hidden-mint rug, verify a token is real, and revoke the allowances that drain wallets."
tags: ["onchain", "crypto", "tokens", "erc-20", "erc-721", "approvals", "allowances", "etherscan", "wallet-security", "solana", "tron"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Almost everything worth analyzing on-chain is *token* activity, and a token is not a coin in your wallet: it is a row in a smart contract's balance ledger. Learn to read the Transfer event (the backbone of all flow analysis) and the Approval event (the #1 wallet-drain surface), and you can analyze flows, vet a token, and defend a wallet.
>
> - **What it is**: an ERC-20 token is one smart contract storing a map from address to balance. Your wallet holds a key, not coins; the contract's ledger says how much you "have."
> - **How to read it**: every transfer emits a `Transfer(from, to, value)` log and every permission grant emits an `Approval(owner, spender, value)` log. Explorers, Dune, and analytics tools are just indexes over those two events.
> - **What you DO with it**: trace flows from Transfer logs; verify a token by its *contract address* (name and symbol are free to fake); check your outstanding allowances and revoke the dangerous ones.
> - **The number to remember**: an *unlimited* approval (`2^256 − 1`) to a malicious contract puts your **entire** balance of that token at risk — not the amount you swapped, all of it.

In February 2025, attackers tied to North Korea's Lazarus Group took roughly \$1.46 billion out of the exchange Bybit in a single operation — the largest crypto theft ever recorded. But here is the detail that matters for this post: the headline number wasn't "Bitcoin." A large chunk of what moved, and of what gets stolen across the industry every year, is **tokens** — staked ETH derivatives, stablecoins, governance tokens — assets that don't exist as coins at all. They exist as entries in smart contracts. When you read that an address "holds \$50,000 of USDC," there is no folder of dollar-coins anywhere. There is a contract on Ethereum, and inside it a ledger, and in that ledger a row that says your address is owed 50,000 units. That's the whole thing.

This distinction is not pedantic. It is the difference between understanding on-chain activity and being fooled by it. The most common way ordinary people lose money on-chain in 2024–2025 was not a clever exploit of a billion-dollar protocol. It was an *approval* — a permission the victim themselves signed, often months earlier, granting a contract the right to move their tokens, which a drainer then quietly used. To read the chain for an edge *and* to keep your own wallet safe, you have to understand tokens at the level of the contract: how balances are stored, how transfers actually work, and — above all — how approvals turn a convenience into a liability.

It's worth pausing on how the Bybit funds were tracked, because it shows the whole toolkit in miniature. Within hours of the theft, blockchain analytics firms and independent on-chain sleuths had a near-complete map of where the money went — not because they had special access, but because every movement was a `Transfer` event sitting in public logs. The attackers consolidated the stolen tokens, swapped some into ETH and native coins to shed the freezable ones (a centralized stablecoin issuer can blacklist a thief's address; ETH cannot be frozen), then began routing funds across bridges and mixers. Each of those steps is a chain of `Transfer` logs, and following them hop by hop is the bread-and-butter of on-chain forensics. The lesson cuts both ways: the same public ledger that lets a thief move \$1.46 billion in minutes also makes every one of those minutes permanently visible to anyone who knows how to read tokens.

This post builds that understanding from zero. We start with what a token *is*, then read the two events you will see thousands of times (`Transfer` and `Approval`), then go deep on the approval mechanism that hides most of the real risk, and finish with a hands-on Etherscan walkthrough and a defender's playbook.

![Token contract balance ledger mental model with wallets and explorer](/imgs/blogs/tokens-onchain-transfers-and-approvals-1.png)

## Foundations: a token is a contract, not a coin

Before any event logs or hex, fix the core idea, because every later confusion dissolves once it clicks.

### Native coin vs. token

A blockchain has a **native coin**. On Ethereum it's ETH; on Solana it's SOL; on Tron it's TRX; on Bitcoin it's BTC. The native coin is special: the protocol itself tracks every address's balance of it directly, in the chain's base state, and you pay transaction fees ("gas") in it. When you send ETH, the protocol debits your account and credits the recipient's — it's built into the rules of the network.

A **token** is different. A token is *not* part of the base protocol. A token is a **smart contract** — a program deployed to the chain — that keeps its own internal ledger of who owns how much. USDC, USDT, UNI, LINK, PEPE, WBTC: every one of these is a contract sitting at some address, with code and a stored mapping. The network has no idea, at the protocol level, that "USDC" exists. It only knows there is a contract at address `0xA0b8...6eB48` whose internal state happens to encode dollar balances. The "token" is a social and economic agreement layered on top of a contract that follows a standard interface.

So when you "own" 2,500 USDC, here is the literal situation: the USDC contract has a variable, conventionally called `balanceOf`, that maps addresses to numbers, and `balanceOf[yourAddress]` equals 2,500 (in its smallest units — we'll get to that). Your wallet software reads that number from the contract and displays it. Your wallet itself stores nothing but a private key. The key proves you control your address; the *balance* lives in the contract. Take that case seriously and the rest of this post is downhill.

The cover figure above shows exactly this: two wallets (each just a key), one token contract, and the ledger of address→balance rows inside it. The block explorer doesn't reach into anyone's wallet to find a balance — it reads the contract's rows.

### The ERC-20 interface

On Ethereum, the agreement that makes a contract "a fungible token" is a standard called **ERC-20** (Ethereum Request for Comments, number 20). Fungible means every unit is interchangeable — one USDC is exactly like any other USDC, the way one dollar bill is like another (as opposed to non-fungible, where each item is unique, like a deed to a specific house). ERC-20 says: if you want to be a token, your contract must expose this set of functions and emit these events. Wallets, exchanges, and DeFi protocols can then treat *any* ERC-20 the same way, because they all speak the same interface. That standardization is why a brand-new token can appear on Uniswap within minutes — the infrastructure already knows how to talk to it.

The power of a shared interface is hard to overstate. Before ERC-20, every project that wanted a token would have invented its own incompatible scheme, and every wallet and exchange would have needed custom code to support each one. ERC-20 collapsed that into a single contract shape, so the *entire* ecosystem of wallets, explorers, DEXes, lending markets, and analytics tools works with any conforming token automatically. It is the USB port of crypto: define the plug once, and everything that follows the spec interoperates. The flip side — and it's the recurring theme of this post — is that the standard defines the *interface*, not the *behavior* behind it. Two contracts can both be valid ERC-20s while one is the genuine, reserve-backed USDC and the other is a honeypot that lets you buy but never sell. The interface is a promise about which functions exist, not a promise that the code behind them is honest.

The core ERC-20 functions, stated plainly:

- **`balanceOf(address)`** — read-only. Returns how many units the given address owns. This is the ledger lookup.
- **`transfer(to, amount)`** — moves `amount` from *the caller's* balance to `to`. You're spending your own tokens.
- **`approve(spender, amount)`** — grants `spender` permission to move up to `amount` of *your* tokens later. This is the dangerous one; we'll spend a third of this post on it.
- **`transferFrom(from, to, amount)`** — moves `amount` from `from` to `to`, but the *caller* is some third party that `from` previously approved. This is how a DEX, a lending pool, or a drainer pulls tokens.
- **`allowance(owner, spender)`** — read-only. Returns how much `spender` is still permitted to pull from `owner`.
- **`totalSupply()`** — read-only. The sum of all balances; the size of the float.

Two of these are read-only views (`balanceOf`, `allowance`, `totalSupply`) — calling them costs nothing and changes nothing. The other three (`transfer`, `approve`, `transferFrom`) are *state-changing transactions* that cost gas and rewrite the ledger. Hold onto that split: views are free reads; the three writes are where money and risk live.

### What an "event" is

When a contract function changes state, it can **emit an event** — a structured record written into the transaction's *logs*. Events are not part of the contract's storage; they're a cheap, append-only side channel designed to be *read by the outside world*. Block explorers, indexers, Dune queries, your wallet's activity feed — all of them are built by reading events.

ERC-20 mandates two events:

- **`Transfer(from, to, value)`** — emitted on every movement of tokens, including mint and burn (more soon).
- **`Approval(owner, spender, value)`** — emitted whenever an allowance is set or changed.

If you remember one thing about reading the chain: **the `Transfer` event is the atom of all flow analysis, and the `Approval` event is the atom of all allowance risk.** Exchange-flow dashboards, smart-money trackers, whale alerts, stablecoin supply charts — every one of them is, underneath, a query over `Transfer` logs. Wallet-drain investigations and the "revoke your approvals" advice you've heard — those are queries over `Approval` logs. The entire toolkit you'll build in this series sits on top of these two records.

### Decimals: why 1 USDC is 1,000,000 units

Here's a foundation that trips up every beginner the first time they read raw on-chain data. Smart contracts can't store fractions — the EVM works in integers only. So a token that wants to represent cents, or eighteen decimal places, stores everything as a big whole number of its *smallest unit*, and declares how many decimal places to shift when displaying.

ERC-20 tokens have a `decimals()` value:

- **USDC and USDT use 6 decimals.** So 1 USDC is stored as `1,000,000` base units. \$2,500 of USDC is stored as `2,500,000,000`.
- **Most other ERC-20s (UNI, LINK, DAI, WETH) use 18 decimals**, matching ETH itself. So 1 DAI is stored as `1,000,000,000,000,000,000` (1 followed by 18 zeros).
- **WBTC uses 8 decimals**, matching Bitcoin's satoshis.

When you look at a raw `Transfer` log on Etherscan and the `value` field reads `2500000000`, that is not 2.5 billion dollars — it's \$2,500 of a 6-decimal token. Misreading decimals is the single most common rookie error in on-chain analysis, and it's also how some scam tokens try to confuse you. Always check `decimals()` before you interpret a raw value.

#### Worked example: sending \$2,500 of USDC in base units

You send a friend \$2,500 of USDC. USDC has 6 decimals, so the contract stores everything in millionths of a dollar.

- Amount to move = \$2,500.
- Base units = \$2,500 × 10^6 = `2,500,000,000` units.
- The `transfer(friend, 2500000000)` call subtracts `2,500,000,000` from your `balanceOf` row and adds it to your friend's row.
- The emitted log is `Transfer(you, friend, 2500000000)`.

If you'd misread that `value` as a token count, you'd think 2.5 billion of something moved. The intuition: raw on-chain numbers are always in the token's smallest unit, so a \$2,500 transfer of a 6-decimal token shows up as the integer `2,500,000,000`, and you only get dollars back by dividing by 10 to the power of `decimals`.

### Mint and burn

Two more foundations and we're ready to go deep. **Minting** creates new tokens out of nothing: the contract adds units to some address's balance and increases `totalSupply`. **Burning** destroys tokens: it removes units from a balance and decreases `totalSupply`. Both are represented, by convention, as `Transfer` events with a special party — minting is a `Transfer` *from* the zero address `0x000...000`, and burning is a `Transfer` *to* the zero address (or a dead address like `0x000...dEaD`). That's why you'll see transfers "from" an address that nobody controls: it's the protocol's way of saying "these tokens were just created."

For a legitimate token, mint and burn follow a documented policy: USDC mints when Circle receives a real dollar and burns when it pays one out; a governance token might mint on a fixed emission schedule. For a *scam* token, an unrestricted mint function is a trapdoor — the kind we'll learn to spot. We'll return to this, because "who can mint, and how much?" is one of the highest-signal questions in token due diligence.

### NFTs: ERC-721 and ERC-1155

Everything so far described *fungible* tokens, where every unit is interchangeable. There is a second family — **non-fungible tokens (NFTs)** — where each token is unique, and they follow different standards that you'll read constantly in any NFT or gaming analysis.

**ERC-721** is the NFT standard. Instead of a `balanceOf` mapping address to a *quantity*, an ERC-721 contract maintains an `ownerOf` mapping that ties each unique token *ID* to a single owner address. Token #1234 of a collection has exactly one owner; it isn't divisible and isn't interchangeable with token #1235. The same `Transfer` event exists — `Transfer(from, to, tokenId)` — but now the third field is the *ID* of the specific item, not an amount. So the entire ownership history of any single NFT is reconstructable from that collection's `Transfer` logs, filtered to one `tokenId`. That's how marketplaces show you provenance: every prior owner is a row in the log.

**ERC-1155** is a hybrid "multi-token" standard, popular in gaming, that lets one contract manage *many* token types at once — some fungible (10,000 identical gold coins), some unique (one legendary sword) — with batch transfers for efficiency. It emits `TransferSingle` and `TransferBatch` events rather than the plain `Transfer`. The mental model is unchanged: a contract holds a ledger, movements emit logs.

Crucially for safety, **NFTs have their own approval functions** — `approve(spender, tokenId)` for a single item and `setApprovalForAll(operator, true)` to grant a spender control over your *entire collection*. That last one is the NFT equivalent of an unlimited approval, and it is exactly how NFT-drainer scams work: a malicious "mint" or "claim" site asks you to `setApprovalForAll`, and once you sign, the operator can transfer out every NFT in that collection. When you audit approvals, the `setApprovalForAll` grants matter as much as the unlimited ERC-20 ones — a single careless signature can hand over a whole collection.

## The Transfer event: the backbone of flow analysis

Now we go deep on the first of the two events. Everything you've heard about "tracking smart money," "watching exchange inflows," or "following the stolen funds" is, mechanically, the analysis of `Transfer` logs.

A `transfer` is not a coin physically moving. It is the contract performing three steps inside one transaction: check that the sender has enough, subtract from the sender's ledger row, add to the recipient's ledger row, and then emit the `Transfer(from, to, value)` log so the world can see it happened. The figure below traces that sequence end to end.

![Transfer function pipeline from caller through ledger edit to emitted event](/imgs/blogs/tokens-onchain-transfers-and-approvals-2.png)

### Why analysts live in the Transfer log

Because the log carries exactly three fields — `from`, `to`, `value` — and because *every* token movement emits one, you can reconstruct the full flow graph of any token from its Transfer history alone. Concretely:

- **Exchange-flow analysis**: label the deposit and withdrawal addresses of major exchanges, then sum `Transfer` values flowing into vs. out of them. Net inflows of a coin often precede selling (supply arriving on the order book); net outflows often signal accumulation into self-custody. The entire "exchange reserves" genre of dashboards is a `Transfer`-log aggregation against a set of labeled addresses.
- **Smart-money tracking**: pick a set of historically-profitable addresses, then watch their `Transfer` activity — what they're accumulating, what they're dumping. Tools like Nansen and Arkham are, at heart, a labeled address book plus a `Transfer` index.
- **Holder concentration**: group `Transfer` history by current holder and you can see whether a token is held by ten wallets or ten thousand. Concentration is a rug-risk signal.
- **Forensics**: when funds are stolen, investigators follow the chain of `Transfer` events hop by hop — into a swap, across a bridge, through a mixer — to map where the money went. The Bybit funds were traced this way in near-real-time.

#### Worked example: reading a whale's exchange inflow as potential supply

You're watching a labeled whale address. Over one morning its `Transfer` logs show 4,000,000 UNI moving *to* a Binance deposit address. UNI is trading at \$8.

- Tokens moved = 4,000,000 UNI (an 18-decimal token, so the raw `value` would read `4000000` followed by 18 zeros).
- Dollar value arriving on the exchange = 4,000,000 × \$8 = \$32,000,000.
- Read: \$32,000,000 of potential sell-side supply just landed where it can be sold. It is *not* a guarantee of selling — the whale might be posting collateral or rebalancing — but it's pre-positioned supply you couldn't have seen from price alone.
- The invalidation: if a matching `Transfer` log moves it back *out* to a self-custody address within hours, the "inflow" was a transit, not a sale.

The intuition: a single `Transfer` log, valued in dollars and read against a labeled address, turns an abstract token count into a concrete "\$32,000,000 of supply just moved to where it gets sold" — that dollar framing is the whole edge.

### How the Transfer log can deceive you

The chain doesn't lie about *what* happened, but it's easy to misread *why*. Three traps:

1. **Wash trading**: an actor moves tokens between wallets it controls to manufacture "volume" or "holders." The `Transfer` logs are real, but the economic activity is fake. NFT and low-cap token "volume" is heavily washed. A cluster of addresses funded from one source, trading in circles, is the tell.
2. **Internal vs. economic movement**: a protocol rebalancing its own treasury, or a bridge moving its reserves, generates large `Transfer` logs that mean nothing about market sentiment. Label the address before you interpret the flow.
3. **Decimals and fakes**: a scam token can mint itself a huge `value` and airdrop it into your wallet to create a `Transfer` log that *looks* like you received something valuable — bait to lure you to a phishing site. Never interact with a token that simply appeared; check its contract first.

These are not reasons to distrust the data — they're reasons to *label and verify* before you read meaning into it. That discipline is the difference between an analyst and a dashboard tourist.

### Anatomy of a raw Transfer log

When you click into a single token transfer on an explorer's logs view, the `Transfer` event has a specific structure worth knowing, because it explains why indexing is fast and why some fields are searchable. An event log has *topics* and *data*. The first topic is the event's signature hash (a fixed fingerprint of `Transfer(address,address,uint256)`); the next topics are the *indexed* parameters — for ERC-20's Transfer, that's `from` and `to`; and the un-indexed `value` lives in the data field. "Indexed" means an indexer can filter on it efficiently, which is exactly why you can ask an explorer "show me every transfer to this address" and get an instant answer: `to` is an indexed topic. This is the plumbing under every "transfers in/out of address X" view you'll ever use.

A practical consequence: the analytics tools differ mostly in *labeling and aggregation*, not in the underlying data. Etherscan, Dune, Nansen, and Arkham are all reading the same `Transfer` topics. What you pay for with Nansen or Arkham is the curated *address book* — the human work of tagging which address is Binance's hot wallet, which is a market maker, which is a known scammer — layered on top of the free, public log stream. Knowing this keeps you honest: a fancy dashboard's "smart money is buying" is only as good as its address labels, and those labels carry survivorship bias and error. The raw `Transfer` log is the ground truth; the labels are an interpretation.

### Gas, and why moving a token costs ETH

A subtle point that confuses newcomers: to send a token like USDC, you pay the transaction fee ("gas") in **ETH**, not in USDC. This follows directly from the contract model. Moving USDC means calling the USDC contract's `transfer` function, and *any* call to the chain — running any contract code — is paid for in the network's native coin. The USDC contract is just code the network executes on your behalf; the network charges ETH for that execution. This is why a wallet can be "stuck" holding tokens it can't move: if it has \$5,000 of USDC but \$0 of ETH, there's no gas to pay for the `transfer` call, and the tokens sit frozen until ETH arrives. It's also why an `approve` is a real, gas-costing transaction — it executes contract code to write the allowance into storage — while *reading* `allowance` or `balanceOf` is free, because reads don't change state and don't need to be mined into a block.

## Approvals: the convenience that drains wallets

Here is the heart of the post. The `transfer` function lets *you* move *your own* tokens. But DeFi needs something more: it needs a contract — a DEX router, a lending pool, an NFT marketplace — to move *your* tokens *on your behalf*, as part of executing a swap or a deposit. ERC-20 solves this with a two-step pattern: **approve, then transferFrom**.

- **Step 1 — `approve(spender, amount)`**: you sign a transaction that records, in the token contract, `allowance[you][spender] = amount`. You haven't moved anything. You've created a *standing permission* — a note in the contract that says "this spender may pull up to this many of my tokens, whenever it likes."
- **Step 2 — `transferFrom(you, destination, amount)`**: later, the spender calls this. The contract checks your allowance, and if it's sufficient, moves the tokens and reduces the allowance.

The figure below shows the two steps and — crucially — the time gap between them, which is where the danger lives.

![Approve then transferFrom flow with allowance stored and a risk time gap](/imgs/blogs/tokens-onchain-transfers-and-approvals-3.png)

### Why a DEX needs your approval

Say you want to swap \$5,000 of USDC for ETH on Uniswap. Uniswap's router contract has to *take* your USDC to give you ETH. But a contract can't reach into your balance uninvited — ERC-20 has no "pull without permission." So the flow is: first you `approve` the Uniswap router to spend your USDC, then you call the swap, and *inside* that swap the router calls `transferFrom` to pull the USDC it's now allowed to take. That's why the first time you trade a new token on any DEX, you sign **two** transactions: an approval, then the swap. Every subsequent swap of that token (up to the approved amount) needs only the swap, because the standing permission is already recorded.

This is genuinely useful. It's also exactly the mechanism attackers abuse, because the permission *persists* and can be for *any amount*, including unlimited.

#### Worked example: the approval that precedes a \$5,000 DEX swap

You swap \$5,000 of USDC for ETH on Uniswap for the first time.

- Transaction 1: `approve(uniswapRouter, 5000000000)` — you grant the router permission for `5,000,000,000` base units (\$5,000 of 6-decimal USDC). This costs gas (say \$3 at the time) and moves no tokens.
- Transaction 2: the swap. Inside it, the router calls `transferFrom(you, pool, 5000000000)`, pulling exactly \$5,000 of USDC, and sends you the ETH.
- Result: you spent \$5,000 of USDC plus ~\$3 of gas on the approval and ~\$6 on the swap, and the allowance is now back to \$0 because the router pulled exactly what it was permitted.

The intuition: an *exact* approval is self-extinguishing — it permits precisely the \$5,000 the swap needs and nothing remains afterward, so even a compromised router could never pull a second dollar.

### The unlimited-approval trap

Many wallet interfaces and dApps, for "convenience," default the approval `amount` to the maximum a 256-bit integer can hold: `2^256 − 1`, an effectively infinite number. The pitch is that you'll never have to approve that token again — one approval covers all future swaps. The cost is that you've handed the spender a standing permission to drain your *entire current and future balance* of that token, with no second signature required, forever, until you revoke it.

If the spender is the real Uniswap router, fine — it only ever pulls what your swaps require. But if the spender is a malicious contract — because you got phished, because the dApp was compromised, because the "spender" was a drainer disguised as a swap — then that unlimited allowance is a live wire. The attacker can call `transferFrom` at any later moment and take everything. The figure below contrasts the two approvals on the same wallet.

![Before and after comparison of unlimited versus exact token approval loss](/imgs/blogs/tokens-onchain-transfers-and-approvals-4.png)

#### Worked example: an unlimited approval that drains \$18,000 months later

In January you connect your wallet to a slick-looking yield site and approve its contract for "unlimited" USDC so you "won't have to approve again." It pulls nothing at the time. Your wallet then grows to \$18,000 of USDC over the next few months.

- The allowance recorded back in January is `2^256 − 1` — effectively infinite.
- In April, the site's contract (a drainer all along, or since compromised) calls `transferFrom(you, attacker, 18000000000)`.
- It pulls all `18,000,000,000` base units — your entire \$18,000 — in one transaction. You signed nothing in April; the standing permission from January was all it needed.
- Had you approved an *exact* \$200 back in January for the single action you actually took, the most it could ever have taken was \$200.

The intuition: an unlimited approval converts a one-time interaction into a permanent, balance-tracking liability — the drain can take \$18,000 it couldn't even see when you signed, because the permission is for "everything," not "the amount I'm using today."

### Permit and Permit2: signatures instead of transactions

A newer pattern deserves a flag because it changes what a "signature" can do. Classic `approve` is an on-chain transaction. **Permit** (EIP-2612) and Uniswap's **Permit2** let you grant an allowance with an *off-chain signature* — you sign a message in your wallet, no gas, and the dApp submits it. This is more gas-efficient and increasingly common. But it means a malicious site can present a "sign this message" prompt that, if you approve it, grants a sweeping allowance *without a normal transaction confirmation*. The defense is the same discipline: read what you're signing. A signature request that mentions a token, a spender, and a large or unlimited amount is an approval, even if no gas is involved and even if it doesn't look like a "transaction."

### How approval-drain campaigns actually run

It helps to see the full lifecycle of a drain so the defensive habits make sense rather than feeling like superstition. Approval-based theft is an *industry*, run with off-the-shelf "drainer kits" that affiliates rent and point at victims. The pattern, observed across countless 2023–2025 campaigns, runs like this:

1. **Lure.** A fake airdrop site, a cloned exchange page, a compromised project Discord, or a phishing email points the victim at a site that looks legitimate — often an exact visual copy of a real dApp.
2. **The signature, not the send.** The site doesn't ask for the victim's seed phrase (that's old-school and people are wary now). It asks them to "connect wallet" and then to *approve* a token or *sign a permit* — framed as "enable trading," "claim your airdrop," or "verify your wallet." The amount is unlimited; the spender is the drainer's contract.
3. **The quiet wait, or the instant sweep.** Sometimes the contract pulls immediately. Sometimes it waits — the drainer monitors the approved wallet and fires `transferFrom` only when the balance is worth taking, which is why a drain can land months after the signature.
4. **The cash-out.** Stolen tokens are swapped to native coins and routed through bridges and mixers, the same Transfer-log trail investigators follow.

Every defense in this post targets one of those steps: verifying the contract address defeats the lure; reading what you sign defeats the malicious approval; approving exact amounts caps step three's payoff; auditing and revoking allowances closes the standing permissions before the quiet sweep ever fires. The reason "revoke your approvals" is repeated so often is that step three — the wait — means a victim's *current* safety says nothing about an approval they signed and forgot.

#### Worked example: the cost of one careless signature vs. the cost of revoking

Compare the two outcomes on a wallet holding \$7,500 of a token.

- **Path A — sign the unlimited approval, do nothing.** The drainer waits, then calls `transferFrom` and sweeps all \$7,500 in one transaction. Loss: \$7,500, unrecoverable.
- **Path B — audit and revoke.** The next time you review approvals, you spot the unlimited grant to an unknown spender and send `approve(spender, 0)`. Cost: roughly \$2 of gas. The standing permission is gone before the sweep fires.

The intuition: the entire \$7,500 of downside is closed by a \$2 transaction, so the expected value of a quarterly approval audit dwarfs its cost — you are paying single-digit dollars to remove four-figure live wires.

## Mint, burn, and spotting a hidden-mint rug

Back to supply. We said minting adds tokens and burning removes them, both showing up as `Transfer` events to or from the zero address. For analysis and for safety, the question that matters is **who is allowed to mint, and is there a cap?**

![Mint increases supply and burn decreases it while a hidden mint prints and dumps](/imgs/blogs/tokens-onchain-transfers-and-approvals-5.png)

A legitimate token answers this transparently. USDC's mint is controlled by Circle and backed 1:1 by reserves; its supply expands and contracts with real dollar flows. A fixed-supply token has *no* mint function reachable after deployment. A token with on-chain governance mints on a schedule the community can audit. In every legitimate case, you can read the rules from the verified contract code and the on-chain mint history.

A scam token hides a mint function — often named something innocuous, or gated to an "owner" address the deployer controls — that lets the deployer create unlimited new tokens at will. The play is brutal and common: the token launches, attracts liquidity, the price rises, and then the owner mints an enormous quantity to themselves and sells it all into the liquidity pool, crashing the price to zero and walking away with the pool's real assets. This is one flavor of **rug pull**, and the hidden mint is its loaded gun.

#### Worked example: a hidden mint that drains \$300,000 of liquidity

A new token "MOON" launches with a 1,000,000 supply and attracts a liquidity pool holding \$300,000 of ETH paired against it. The contract has an owner-only `mint` the marketing never mentioned.

- The owner calls `mint(ownerWallet, 100000000)` — printing 100,000,000 new MOON, 100× the original float, for free.
- The owner sells that 100,000,000 MOON into the pool. The pool's pricing curve hands over almost all of its \$300,000 of ETH in exchange for the freshly-printed, worthless tokens.
- Holders are left with MOON whose price has collapsed to ~\$0; the owner walks away with ~\$300,000 of ETH.
- The on-chain tell, visible *before* the rug: a `mint` function callable by a single owner with no cap, plus an owner address holding the keys.

The intuition: an uncapped owner-mint means the token's supply is whatever the deployer wants it to be at any moment, so the \$300,000 of "value" in the pool was never yours to keep — it was collateral waiting to be drained by a printing press.

So the due-diligence question is sharp and checkable: pull up the verified contract, search for `mint`, and ask *who can call it and is it capped?* If the answer is "a single owner address, uncapped," treat the token as a rug in waiting regardless of how good the chart looks. We go deeper on the full rug-and-honeypot checklist in [Rug pulls and honeypots: detecting the trap before you buy](/blog/trading/onchain/rug-pull-and-honeypot-detection); the hidden mint is one entry on that list.

## Fakes and metadata: verify the contract address, not the name

A token's **name** ("USD Coin") and **symbol** ("USDC") are just strings stored in the contract. *Anyone* can deploy a contract that calls itself "USD Coin" with the symbol "USDC." There is nothing on-chain stopping a scammer from deploying a perfect-looking impostor. The only thing that distinguishes the real USDC from a fake is the **contract address** — the unique, unforgeable identity of the actual Circle-controlled contract.

![Matrix comparing real and fake tokens by name symbol contract address and verdict](/imgs/blogs/tokens-onchain-transfers-and-approvals-6.png)

This is why every reputable source publishes the *canonical contract address* of a token, and why "verify the contract address" is the first rule of token safety. A few practical signals that separate real from fake:

- **The contract address matches the official one.** Get it from the project's verified site, CoinGecko/CoinMarketCap's contract field, or the token's official docs — never from a random link, a DM, or a tweet reply.
- **The contract is "verified" on the explorer** (its source code is published and matches the deployed bytecode). Unverified code on a token claiming to be a major asset is a red flag, though verification alone doesn't make a token safe.
- **Liquidity and holder depth match the claim.** Real USDC has tens of billions in supply and millions of holders; a fake "USDC" has a thin pool and a handful of addresses. A "major token" with \$4,000 of liquidity is a fake.
- **The token didn't just appear in your wallet.** Unsolicited airdrops of valuable-looking tokens are almost always phishing bait. The token's transfer-in is real; the trap is the website it lures you to.

The discipline generalizes: on-chain, identity is an address, and a name is decoration. Before you value, trade, or interact with any token, confirm you're looking at the contract you think you are.

## SPL on Solana and TRC-20 on Tron: same idea, different machinery

ERC-20 is Ethereum's standard, but the *concept* — a token is a ledger maintained by on-chain code — is universal. Two other ecosystems matter enough to know the differences.

### SPL tokens on Solana

Solana's token standard is **SPL** (Solana Program Library). The big architectural difference: on Ethereum, each ERC-20 is its own separate contract with its own internal `balanceOf` map. On Solana, there is a *single* shared **Token Program**, and each token is a "mint" account, while each holder has a separate **token account** that records their balance of that mint. So a holder's balance isn't a row inside the token's contract — it's its own little account that the shared program manages. Practically, this changes how you read balances (you look up a wallet's token accounts), and it's why Solana wallets sometimes show "rent" and account-creation quirks that Ethereum users never see. But the mental model holds: a token is still a ledger maintained by a program, and movements still emit traceable records. Solana is the home of high-velocity memecoin trading, so SPL `Transfer`-equivalent data is where most Solana smart-money analysis happens.

### TRC-20 on Tron

Tron's token standard, **TRC-20**, is deliberately near-identical to ERC-20 — same `transfer`, `approve`, `transferFrom`, `balanceOf`, same Transfer and Approval events. If you can read an ERC-20, you can read a TRC-20. Tron matters disproportionately for one reason: it is the **dominant rail for USDT in retail and high-risk flows**, especially across Asia, because Tron's fees are tiny and its USDT throughput is enormous. A large share of the world's stablecoin *transactions* (as opposed to dollar value) happen as TRC-20 USDT on Tron. For an analyst tracing retail or illicit stablecoin flow, Tron is often where the trail leads, and the approval mechanics are exactly the same as on Ethereum.

The takeaway across all three: the *interface* differs in details, but `Transfer`-and-`Approval`-style records exist everywhere, and the contract-is-the-ledger model is the same. Learn it once on ERC-20 and you can read tokens on any chain.

### The second-order effects of "a token is a contract"

Once you internalize that a token is a contract with a ledger, a series of otherwise-puzzling facts about crypto fall into place:

- **A token is exactly as trustworthy as its code and its admin keys.** Because the ledger is whatever the contract says it is, a token's real risk profile is "what can the contract do, and who controls it?" — not the logo or the marketing. A token whose owner can mint, pause transfers, or blacklist holders is one decision away from changing your balance. This is why reading the verified contract is non-optional due diligence, and why "is it a contract I can audit?" matters more than any price chart.
- **Failed transfers and weird tokens exist because each token is its own program.** Some tokens charge a fee on every transfer (a cut routed to the deployer), some rebase (your balance changes without a transfer you made), some block selling entirely (a honeypot). Native ETH does none of this because it isn't a contract. Every quirk you hit with a token traces back to "this is custom code, not the base protocol."
- **Composability and contagion share the same root.** Because any contract can hold and move any ERC-20 via the same interface, tokens slot into lending pools, DEXes, and other protocols like Lego — that's DeFi's superpower. But it also means one token's failure (a depeg, a hidden mint, an exploit) propagates through every protocol that accepted it as collateral. The 2022 collapses showed how fast a single broken token's ledger can cascade. The approve/transferFrom rail is what wires it all together — which is precisely why the standing permissions you grant are a systemic surface, not just a personal one.
- **"Your keys, your coins" is really "your keys, your ledger rows."** Self-custody protects the *key* that authorizes moving your rows. It does not protect you from a malicious token contract, a hidden mint, or an approval you signed. Custody and contract-risk are different threats; holding your own keys is necessary but not sufficient.

## The scale of tokenized value: stablecoins

To anchor *why* tokens dominate on-chain analysis, look at stablecoins — tokens that each represent one dollar. They are the clearest example of "value that exists only as a contract ledger," and they are enormous. The chart below shows the combined outstanding supply of the two largest, Tether (USDT) and Circle (USDC), growing from roughly \$25 billion to well over \$200 billion.

![Stacked area chart of USDT and USDC outstanding supply from 2020 to 2025](/imgs/blogs/tokens-onchain-transfers-and-approvals-7.png)

Every dollar of that supply is a number in a contract's ledger. When you read that "\$220 billion of stablecoins" exist, there is no vault of digital dollar-coins — there are two big contracts (plus smaller ones), each with a `balanceOf` map summing to those totals, each emitting a `Transfer` log every time a dollar moves. Stablecoins are also the on-chain market's **dry powder**: a rising stablecoin supply is capital sitting on-chain ready to buy, and analysts watch it as a liquidity gauge. We dig into how the issuers actually work — reserves, minting, redemption, and the systemic questions — in [Stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar). For this post, the point is narrower and foundational: the single largest category of on-chain value is *tokens*, and a token is a contract ledger. That's why almost everything you'll analyze is token activity.

## How to read it: a walkthrough on Etherscan

Theory is cheap; let's read real token data. Etherscan (the Ethereum block explorer) exposes everything we've discussed. Pick any token and open its page — the URL pattern is `etherscan.io/token/<contract-address>`. Here's the tour, tab by tab, with what each one is *for*.

### The token overview

At the top you'll see the token's name, symbol, and — read this first — its **contract address**, plus its `decimals`, total supply, and number of holders. This is your verification checkpoint: does the contract address match the canonical one you got from a trusted source? Does the holder count and supply match a real, deep asset, or a thin fake? For a major token, expect millions of holders and a verified contract; for a fake impersonator, expect a handful of holders and a thin pool.

### The Holders tab

This lists every address holding the token, ranked by balance, with each holder's percentage of supply. This single view answers a high-signal question: **how concentrated is ownership?** If the top 10 addresses hold 90% of supply, a few wallets can crater the price at will — a major risk for a small token. Filter out known contracts (the liquidity pool, the staking contract, burn addresses) and look at how much *real, individual* holders control. For new tokens, concentration in a few fresh wallets funded from one source is a classic rug setup.

### The Transfers tab

This is the live feed of every `Transfer` event for the token — each row showing `from`, `to`, `value`, and the transaction. This is where flow analysis happens by hand. You can:

- Spot large transfers into known exchange deposit addresses (potential sell pressure).
- See mints (transfers *from* `0x000...000`) and burns (transfers *to* the dead address) and judge whether supply is being inflated.
- Follow a specific address's history — click any address and see every token it has sent and received.

For serious flow work you'd graduate to Dune (SQL queries over the Transfer logs) or Arkham/Nansen (labeled addresses on top of the same data), but the raw Transfers tab is the ground truth they're all built on.

A concrete reading exercise: open the Transfers tab of a mid-cap token and scan the most recent rows. Ask of each large one — is the `to` a known exchange (sell pressure incoming), a known contract (a deposit into a protocol, not a sale), the zero address (a burn, reducing supply), or a fresh wallet with no history (worth watching)? Then click a single large recipient and look at *its* full history: was it funded from an exchange withdrawal an hour earlier, then it bought this token — a possible insider front-running news? Or is it an address that has held for two years — a long-term holder adding? None of these reads is certain, but each is a hypothesis the `Transfer` log lets you form *before* the move shows up in price or in a news headline. That lead time, read carefully and skeptically, is the entire premise of on-chain analysis.

### The Contract tab

Click "Contract." If the code is **verified**, you'll see the actual Solidity source. This is where due diligence gets real. Search the source for `mint` — who can call it, and is the amount capped? Search for `owner` and `onlyOwner` — what privileged powers does the deployer retain? Look for the ability to pause transfers, blacklist addresses, or set arbitrary fees — all of which can be legitimate (USDC can freeze addresses by regulation) or predatory (a honeypot that blocks *you* from selling). If the contract is *unverified* on a token claiming to be a serious asset, that alone is a reason to walk away. The "Read Contract" and "Write Contract" sub-tabs let you call view functions directly — including `allowance(owner, spender)`, which brings us to the most important check of all.

### Checking your own outstanding allowances

Here's the defender's core move. For any address, you can enumerate the approvals it has granted — every `Approval(owner, spender, value)` event where `owner` is that address, netted against any later changes, gives the current outstanding allowances. Etherscan has a "Token Approvals" tool (`etherscan.io/tokenapprovalchecker`) that does exactly this: enter an address and it lists, per token, every spender you've approved and *how much* — flagging the unlimited ones.

Read that list as a risk inventory. Each row is a contract that can pull up to that amount of that token from you, right now, with no further signature. An unlimited approval to a contract you don't recognize, or to a dApp you used once and forgot, is exactly the live wire from the \$18,000 worked example. The action is to revoke it.

#### Worked example: pricing your outstanding-allowance risk

You run the approval checker on your wallet and find three live approvals:

- Unlimited USDC approved to the real Uniswap router. Your USDC balance is \$12,000. Risk *if the router were ever compromised*: up to \$12,000 — but it's the canonical, audited router, so this is low-probability, high-trust.
- Unlimited USDC approved to a yield site you used once last year and can't quite remember vetting. Same \$12,000 at risk, but to an unknown, unaudited contract — high-probability-of-trouble exposure.
- An exact \$50 approval to an NFT mint contract. Maximum loss: \$50.
- Total *unlimited* exposure: \$12,000 to a sketchy spender you'd revoke immediately, plus \$12,000 to a trusted one you might keep.

The intuition: your allowance list is a literal dollar-denominated risk ledger — each unlimited row to an untrusted spender is your *entire balance* of that token exposed, so you triage by "how much, to whom, how trusted" and revoke the worst rows first.

## The defender's move: revoking approvals

Revoking an approval is mechanically simple: it's an `approve(spender, 0)` transaction. Setting the allowance back to zero means the spender can no longer call `transferFrom` on your tokens — the standing permission is gone. The figure below shows the full defensive loop.

![Revocation pipeline from listing approvals to setting allowance to zero](/imgs/blogs/tokens-onchain-transfers-and-approvals-8.png)

In practice, you don't hand-craft these transactions. The standard tool is **[revoke.cash](https://revoke.cash)** — connect a wallet (or just paste an address to view, read-only) and it shows every token approval across many chains, sorted with the dangerous unlimited ones surfaced, and lets you revoke each with one click. Etherscan's Token Approvals tool does the same on Ethereum. Other wallet dashboards (DeBank, the wallet's own settings) increasingly bundle approval management too.

A few realities to internalize:

- **Revoking costs gas, not tokens.** Each revocation is an on-chain transaction, so you pay a small fee. Revoking a stale unlimited approval for a few dollars of gas is cheap insurance against losing your whole balance.
- **Revoking does not undo a drain that already happened.** It's prophylactic. If a malicious spender already pulled your tokens, revoking afterward only stops *further* pulls — which is still worth doing immediately if you suspect compromise.
- **A revoked approval can be re-granted.** Revocation isn't permanent protection; it's hygiene. The durable habit is to *approve exact amounts* when a dApp lets you, and to periodically audit and prune your approval list — quarterly, or any time after using an unfamiliar site.
- **Watch what you sign, not just what you send.** With Permit2 and off-chain signature approvals, a sweeping allowance can be granted by signing a *message*, with no transaction and no gas. Treat any signature that names a token, a spender, and an amount as the approval it is.

This defensive discipline connects directly to how DeFi works — every Uniswap swap, every Aave deposit, every protocol interaction runs on the approve/transferFrom rail. Understanding that rail is what lets you use DeFi without leaving live wires behind; see [DeFi protocols: Uniswap, Aave, and MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) for how those protocols use your approvals, and [Reading DEX liquidity and pools](/blog/trading/onchain/reading-dex-liquidity-and-pools) for how the swap that consumes your approval actually executes against a pool.

## Common misconceptions

**"My tokens are stored in my wallet."** No — your wallet stores a private key. Your token balances are rows in each token's contract ledger. This is why losing your key loses your tokens (you can't prove ownership of the rows) but why a wallet "holding" a token it never interacted with is meaningless until you check the contract. The 12-word seed phrase reconstructs the key, which is the only thing that was ever yours.

**"Sending tokens and sending ETH work the same way."** They don't. Sending ETH is a native protocol operation the network handles directly. Sending a token is a *call to the token contract's `transfer` function* — which is why a token transfer can fail even when you have ETH for gas (the contract might reject it), why you pay gas in ETH to move a non-ETH token, and why "approving" exists for tokens but never for ETH (ETH has no `approve` — you can't grant a standing permission over your ETH the way you can over a token).

**"If I revoke an approval, my tokens are safe."** Only going forward. Revocation sets a future permission to zero; it does nothing about tokens already pulled. And it's per-spender — revoking one approval leaves every other live. Safety is the *ongoing* practice of approving exact amounts and auditing the full list, not a one-time button.

**"A token with the right name and symbol is the real one."** Name and symbol are free-to-copy strings. Only the contract address is the real identity. A perfect-looking "USDC" with a thin pool and ten holders is a fake — and interacting with it (especially approving it) is how the trap springs. Verify the address against a trusted source every time.

**"Unlimited approval is fine because I trust the dApp."** Maybe today. But an unlimited approval outlives your trust: the dApp can be compromised, the contract can have a latent bug, the team can turn malicious, and the approval keeps standing the whole time, tracking your growing balance. Trust is not a property you can pin to a permanent on-chain permission — which is why "exact amount, then revoke" beats "unlimited because I trust them."

## The playbook: what to do with it

The if-then checklist for reading tokens as a trader *and* defending a wallet.

**As an analyst / trader:**

- **Signal**: a large `Transfer` into a labeled exchange deposit address. **Read**: potential sell-side supply arriving. **Action**: size the inflow in dollars (tokens × price) and weigh it against context. **Invalidation**: a matching transfer back out to self-custody within hours — it was transit, not a sale.
- **Signal**: a token's Holders tab shows the top 10 addresses holding most of supply. **Read**: concentrated ownership; a few wallets can crater price. **Action**: treat as high-risk for a small token; size positions accordingly. **False positive**: the "holders" are the liquidity pool, staking, or burn contracts — label them out before judging.
- **Signal**: rising aggregate stablecoin supply. **Read**: dry powder accumulating on-chain. **Action**: read as a medium-term liquidity tailwind, not a timing signal. **Invalidation**: supply contracting (net redemptions) is the opposite read.

**As a defender / investor doing due diligence:**

- **Signal**: a token's verified contract has an owner-callable, uncapped `mint`. **Read**: the deployer can print and dump at will — a loaded rug. **Action**: do not buy; if holding, exit. **False positive**: a regulated stablecoin's controlled mint with public reserves — legitimate, because the policy and reserves are auditable.
- **Signal**: a token claiming to be a major asset has an *unverified* contract, a thin pool, or a contract address that doesn't match the canonical one. **Read**: likely a fake impersonator. **Action**: do not interact, do not approve. **Invalidation**: address matches the trusted source and depth is real — it's the genuine token.
- **Signal**: a valuable-looking token simply appeared in your wallet unsolicited. **Read**: phishing bait. **Action**: ignore it; never click through to any site it points to, never approve it. **Why**: the transfer is real but its only purpose is to lure you into signing an approval.
- **Signal**: your approval checker lists an unlimited allowance to a spender you don't recognize or a dApp you used once. **Read**: your entire balance of that token is one `transferFrom` from gone. **Action**: revoke it now (a few dollars of gas). **Habit**: approve exact amounts when possible; audit the full list quarterly and after any unfamiliar site.
- **Signal**: a dApp asks you to *sign a message* (no gas) that names a token, a spender, and a large or unlimited amount. **Read**: that's a Permit/Permit2 approval, equal in power to an on-chain `approve`. **Action**: read it; reject anything unlimited or to an unknown spender.

The one rule that ties it all together: on-chain, a token is a contract ledger, identity is a contract address, and an approval is a standing permission. Read the `Transfer` log for flow, read the `Approval` log and your allowance list for risk, verify the address before you value or trust anything, and never let an unlimited permission outlive the reason you granted it.

## Further reading & cross-links

- [Anatomy of a transaction](/blog/trading/onchain/anatomy-of-a-transaction) — what a single on-chain transaction contains, including the logs that carry Transfer and Approval events.
- [Addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts) — why your wallet holds a key, not coins, and how EOAs differ from contract accounts.
- [Rug pulls and honeypots: detecting the trap](/blog/trading/onchain/rug-pull-and-honeypot-detection) — the full checklist, of which the hidden mint is one entry.
- [Reading DEX liquidity and pools](/blog/trading/onchain/reading-dex-liquidity-and-pools) — how the swap that consumes your token approval executes against a pool.
- [Stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) — how the largest tokens by value actually mint, redeem, and hold reserves.
- [DeFi protocols: Uniswap, Aave, and MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — the protocols that run on the approve/transferFrom rail this post explains.
