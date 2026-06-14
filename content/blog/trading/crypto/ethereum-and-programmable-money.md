---
title: "Ethereum and Programmable Money: The World Computer Behind Decentralized Finance"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How Ethereum turned a blockchain from a ledger of money into a global programmable computer where self-executing smart contracts became the foundation of a parallel financial system."
tags: ["ethereum", "smart-contracts", "blockchain", "defi", "proof-of-stake", "staking", "layer-2", "crypto", "evm", "tokens"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Ethereum took the one trick Bitcoin proved (a shared ledger nobody controls) and made it programmable, so that instead of just recording who owns what, the network can run code that moves money automatically when conditions are met.
>
> - A **smart contract** is a small program that lives on the blockchain. Once deployed, it runs exactly as written, with no company able to pause, edit, or reverse it. That is the whole revolution: rules that enforce themselves.
> - Every contract runs on the **Ethereum Virtual Machine (EVM)**, a single shared computer replicated across roughly a million validators. You pay for the computation in **gas**, a metered fee that stops anyone from running the world's computer for free.
> - In 2022 Ethereum swapped its engine from energy-hungry **proof-of-work** mining to **proof-of-stake**, in an event called the Merge that cut the network's electricity use by roughly 99.9 percent overnight.
> - **Composability** — contracts freely calling other contracts — turned a pile of separate apps into "money legos," and produced an entire parallel financial system (DeFi) of lending, trading, and stablecoins.
> - It is not free of risk: a single bad line of code can drain millions (the 2016 DAO hack split the chain in two), fees can spike to absurd levels, and most users now transact on **layer-2 rollups** that add their own trust assumptions.

The diagram above is the mental model for this entire post: Ethereum is a stack. At the bottom sits one slow, expensive, maximally secure base chain that settles everything. On top sit faster, cheaper layers and the apps people actually touch — and the whole tower only works because each layer can trust the math of the layer beneath it.

If [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision) showed that a group of strangers could agree on a ledger of money without a bank in the middle, Ethereum asked a bigger question: what if the ledger could also run *programs*? What if, instead of only recording "Alice sent Bob 1 coin," the network could record and enforce "Alice sends Bob 1 coin *automatically, the instant a shipment is confirmed*"? That single shift — from a ledger of money to a ledger of *programmable* money — is why Ethereum became the foundation of decentralized finance, NFTs, stablecoins, and a sprawling experiment in rebuilding financial plumbing in open source.

![The Ethereum stack from settlement layer up to apps](/imgs/blogs/ethereum-and-programmable-money-1.png)

This post builds the whole idea from zero. We will define every term — smart contract, the EVM, gas, ETH versus tokens, an account, a dApp, composability, proof-of-stake, staking, and rollups — before we go deep on how they fit together, how Ethereum wields power and where it makes (and loses) money, the famous episodes that tested it, and the misconceptions that trip up newcomers. No prior crypto knowledge assumed. No price predictions, no advice — just the mechanics, with the risks named beside every upside.

## First principles: the building blocks

Let us nail down the vocabulary first. Everything else is a recombination of these pieces.

**A blockchain** is a shared database that thousands of independent computers each keep a full copy of, and that they update by agreeing — through a defined set of rules — on a new "block" of changes every so often. Because everyone holds the same copy and the rules decide what counts as valid, there is no central server to hack, bribe, or shut down to rewrite history. Bitcoin was the first blockchain that worked at scale, and it stored one kind of data: who owns how many bitcoins.

**On-chain** simply means "recorded on the blockchain, visible to anyone, and enforced by the network's rules." Its opposite, **off-chain**, means anything that happens in the ordinary world or on a private server — a promise, a bank balance, a Google spreadsheet. A central idea throughout this post is that being on-chain makes a rule *self-enforcing*: nobody has to be trusted to honor it, because the network simply will not process an action that breaks it.

**A smart contract** is the heart of Ethereum. It is a small program — typically written in a language called Solidity — that you deploy onto the blockchain. Once it is there, it has its own address, it can hold money, and it runs *exactly* as written whenever someone sends it a transaction. The word "contract" is a useful analogy and a slightly misleading one. It is like a contract in that it encodes "if this, then that." It is unlike a paper contract in that there is no court, no lawyer, and no enforcement gap: the code *is* the enforcement. If the contract says "release these funds when three of five people sign," then the funds are released exactly when three of five sign, no sooner, no later, and no human can override it. The promise and the keeping of the promise are the same object.

**The Ethereum Virtual Machine (the EVM)** is the shared computer that runs all those contracts. Picture a single global processor whose memory and state are replicated across every node in the network. When a contract runs, every validator runs the exact same steps and must arrive at the exact same result, or the block is rejected. This is what makes a contract trustworthy: it does not run on one company's server that could be tampered with — it runs on a machine that thousands of strangers are simultaneously checking. The trade-off is that this shared computer is staggeringly slow and expensive compared to your laptop, because it is doing the same work a million times over for the sake of agreement.

**Gas** is how Ethereum prices that shared computation. Because the EVM is a public resource, you cannot let people run unlimited code for free — someone would write an infinite loop and freeze the network. So every individual operation the EVM performs (adding two numbers, storing a value, calling another contract) costs a fixed number of **gas units**. The total fee you pay is `gas units used x gas price`, where the gas price (quoted in *gwei*, a billionth of an ETH) is set by supply and demand for blockspace. When the network is busy, the gas price rises; when it is quiet, it falls. Gas is the metering that keeps the world computer from being abused, and the fee market that decides whose transaction gets included next.

**ETH** is the native asset of the network — the coin you pay gas in, and the thing validators earn for securing the chain. It is to Ethereum what bitcoin is to Bitcoin. But Ethereum also lets *anyone* create their own tokens by deploying a contract, and those are called **ERC-20 tokens** when they are fungible (interchangeable, like dollars) or **ERC-721 tokens** when they are non-fungible (unique, like a deed). ERC-20 is just a standard — a list of functions a token contract must implement (`transfer`, `balanceOf`, and so on) so that every wallet and exchange knows how to talk to it. A stablecoin like USDC, a governance token like UNI, a meme coin — all of them are ERC-20 contracts. ETH itself is special (it is the native gas currency), but the thousands of *other* tokens are just programs following a shared interface.

**An account (or address)** is your identity on Ethereum. It is a long string like `0x71C7...976F`, derived from a cryptographic key pair you control. There are two kinds: *externally owned accounts*, controlled by a private key held in a wallet (this is "you"), and *contract accounts*, which are smart contracts that hold funds and run code. When people say their **wallet** — MetaMask, a hardware wallet, a phone app — they mean the software that stores their private key and signs transactions on behalf of their address. Lose the key and you lose the account; there is no password reset, because there is no company that holds your funds.

**A decentralized application (a dApp)** is an app whose backend logic lives in smart contracts on Ethereum rather than on a company's private servers. The front end (the website you click) can still be ordinary code, but the part that holds money and enforces rules is on-chain. Uniswap is a dApp: its trading logic is a set of contracts anyone can call directly, even if Uniswap's website disappeared tomorrow.

**Composability** — often nicknamed "money legos" — is the property that any contract can call any other contract, permissionlessly, in the same transaction. Because every dApp speaks the same EVM language and tokens follow shared standards, a new protocol can plug into existing ones like Lego bricks: a lending app can route through a trading app, which routes through a stablecoin, all atomically. This is the single most important reason an entire financial system grew on Ethereum rather than a thousand disconnected apps. We will spend a whole section on it.

**Proof-of-work and proof-of-stake** are two ways a blockchain decides *who* gets to add the next block and how it stops cheaters. **Proof-of-work** (Bitcoin's method, and Ethereum's until 2022) makes computers race to solve a pointless math puzzle; the winner adds the block and earns new coins. It is secure because attacking the chain would require out-spending everyone else on electricity and hardware — but that same electricity is the cost. **Proof-of-stake** (Ethereum's method since 2022) replaces the puzzle with *bonded capital*: validators lock up coins as collateral, are chosen to propose blocks roughly in proportion to their stake, and lose part of their stake (they are "slashed") if they cheat. Security now comes from money at risk, not energy burned.

**Staking** is the act of locking up ETH to become (or back) a validator and earn rewards for helping secure the chain. On Ethereum it takes exactly 32 ETH to run your own validator. Most people who stake smaller amounts do so through pools.

**A layer-2 rollup** is a separate, faster chain that does its computation off the main Ethereum chain but periodically posts a compressed proof or summary back to it — "rolling up" thousands of transactions into one. The rollup borrows Ethereum's security (the base chain is the final judge) while charging a fraction of the fees, because the expensive base-chain blockspace is shared across all those bundled transactions. Arbitrum, Base, and Optimism are rollups. The stack in figure 1 is exactly this: base chain at the bottom, rollups in the middle, apps on top.

With that vocabulary in hand, here is the thesis stated plainly: **Bitcoin proved a shared ledger could exist without a trusted middleman; Ethereum made that ledger programmable, and programmability is what let an entire parallel financial system grow on top of it.** The rest of the post is the elaboration.

## Vitalik Buterin and the 2015 launch

Ethereum was first described in a 2013 whitepaper by Vitalik Buterin, then a teenage programmer and Bitcoin writer who had grown frustrated that Bitcoin's scripting language was deliberately limited. Bitcoin can express "pay this address" and a handful of conditions, but it cannot easily express arbitrary programs — by design, for safety. Buterin's insight was that if you gave the blockchain a *Turing-complete* programming language (one that can express any computation), you could build not just a currency but a platform: a single world computer that anyone could deploy applications onto.

The project raised funds in a 2014 token sale (one of the first large "ICOs," or initial coin offerings) and launched its live network — *mainnet* — in July 2015, in a release named Frontier. A nonprofit, the Ethereum Foundation, coordinates research and the reference software, but it does not own or control the network; the chain runs on independent nodes worldwide. From the first day, Ethereum did the one thing Bitcoin could not: it let strangers deploy and run code that held and moved real value.

It is worth being precise about what "launched" means here, because it is unusual. There was no company with a server you logged into. There was a piece of open-source software that, when enough people ran it and pointed it at the same genesis block, *became* the Ethereum network. The network is the agreement among everyone running the software. That is why no government has been able to "shut down" Ethereum: there is no headquarters to raid, only software running on tens of thousands of machines in dozens of countries.

The way this turned into a financial platform is best read as three pivots over the decade that followed, which the timeline below lays out and which the rest of this post unpacks in order.

![Timeline of Ethereum milestones from 2015 to 2024](/imgs/blogs/ethereum-and-programmable-money-5.png)

The chain *launched* in 2015 with the bare ability to run contracts. In 2016 it nearly died and then split in two over the DAO hack, settling the question of who really holds authority. Through 2017 to 2021 it hosted the ICO boom and NFT mania, which proved demand but congested the network and spiked fees. In 2022 it changed its own engine in the Merge, swapping mining for staking. And in 2024 the Dencun upgrade made layer-2 transactions cheap, pushing most activity off the base chain. Three of those are deep sections of their own below; the timeline is the skeleton they hang on.

#### Worked example: the cost of the world computer's slowness

To feel why programmability is both powerful and expensive, compare two ways to enforce the same rule: "pay a freelancer \$1,000 when a client approves the work."

The traditional way: the client wires money to an escrow company, the escrow company holds it (charging perhaps \$50 to \$200), and a human releases it on approval. The cost is the fee plus the trust that the escrow company will not abscond or go bankrupt.

The Ethereum way: someone deploys an escrow smart contract once. The client sends \$1,000 of a stablecoin into the contract. The contract holds it with no human custodian. When the approval condition is met, the contract releases the \$1,000 to the freelancer automatically. The marginal cost is just the gas to deploy and call the contract — say \$5 to \$40 depending on network congestion — and the trust required is zero, because the code cannot abscond.

The intuition: Ethereum makes each individual computation far more expensive than a normal server would, but it removes the trusted middleman entirely, which is worth paying for whenever the middleman's fee or risk was large.

## What smart contracts make possible

The leap from "a ledger of money" to "programmable money" sounds abstract until you see what it unlocks. A smart contract can do four things an ordinary database entry cannot:

1. **Hold and move value on its own.** A contract has an address and can own ETH and tokens. It can receive a deposit and send a payment without any human pressing a button. This is what makes contracts *custodial without a custodian* — they hold your money, but the rules of who can withdraw it are fixed in code, not in a company's terms of service.

2. **Enforce conditions automatically.** "Release these funds only if X" is enforced by the network, not by a person who might be bribed, asleep, or insolvent. The earlier escrow example is the simplest case; the same pattern powers lending (return collateral when the loan is repaid), insurance (pay out when an oracle reports a flight was delayed), and auctions (transfer the asset to the highest bidder when the timer ends).

3. **Compose with other contracts.** Because every contract can call every other contract, a new app does not have to rebuild trading, lending, or a stablecoin from scratch — it calls the existing ones. This is the property that turned isolated apps into a financial system, covered next.

4. **Be permanent and permissionless.** Once deployed, a contract is there for anyone to use, forever, without asking permission and without the original author being able to revoke access (unless they deliberately coded in an off switch, which many do, with its own trade-offs). A market that no single party can take down behaves very differently from one that depends on a company staying in business.

The catch — and it is a large one — is that *permanent and self-enforcing* also means *permanently broken if it is broken*. If a contract has a bug that lets an attacker drain it, the network will faithfully execute the drain, because the code is the law. There is no fraud department to call. The 2016 DAO hack, which we will dissect, is the canonical lesson in this double edge.

This is also where the parallel to traditional finance sharpens. In [the world of regulated financial institutions](/blog/trading/finance/field-guide-to-financial-institutions), a bank, a clearinghouse, or an exchange is a trusted intermediary whose job is partly to *be* the trusted party — to hold assets, enforce settlement, and absorb the cost of mistakes through capital and insurance. Ethereum's bet is that for a large class of these jobs, you can replace the trusted institution with a transparent, auditable program, and pay for security with cryptography and economic incentives rather than with capital buffers and legal recourse. It is a real bet with real failure modes, not a free lunch.

## The EVM and the gas market

Let us go under the hood of how a transaction actually executes and gets priced, because gas is where most newcomer confusion lives.

![A smart-contract transaction metered by gas](/imgs/blogs/ethereum-and-programmable-money-2.png)

The diagram traces one transaction. Your wallet builds and **signs** a transaction, attaching a *gas limit* (the most gas you will allow it to consume, so a buggy contract cannot drain your wallet) and a *gas price* (what you will pay per unit). The signed transaction enters the **mempool**, a holding area of pending transactions that every node sees. A validator picks transactions from the mempool — generally favoring those offering higher fees — and the **EVM executes** them step by step, *metering* the gas as it goes. When execution finishes, the **fee is charged**: gas units used times the price. Finally the **block is finalized** and the network's state is updated for everyone.

Two refinements matter. First, since the 2021 "London" upgrade, the fee splits into a *base fee* that is **burned** (destroyed, removing ETH from supply) and a *priority tip* that goes to the validator. The base fee adjusts automatically block by block: if blocks are full, it rises; if they are empty, it falls. This is a thermostat for congestion. Second, if your gas limit is too low for the work the contract needs, the transaction *reverts* — it fails and undoes its effects — but you still pay for the gas consumed up to the failure, because the validators did the work. New users lose money this way constantly.

There is a third subtlety that practitioners care about deeply: *ordering*. The validator who builds a block chooses not just which transactions to include but in what order, and on a financial network order is worth money. If your transaction is about to push a token's price up, someone who sees it sitting in the mempool can pay a higher tip to be placed *just before* you (buying first, so you push the price into their position) and *just after* you (selling into the price you moved). This extractable value — called *maximal extractable value*, or MEV — is a real, ongoing tax on naive transactions, and an entire ecosystem of block-builders and "private" transaction routes has grown up to manage it. It is the on-chain echo of front-running on a stock exchange, except the order book and the pending orders are visible to everyone. The takeaway for a beginner is simply that the mempool is public, ordering is auctioned, and a large or price-moving transaction sent carelessly can be sandwiched for a real loss. Gas pays for *computation*; the tip-and-ordering game decides *sequence*, and both are priced markets.

#### Worked example: the gas fee on a token swap

Suppose you want to swap \$2,000 of ETH for USDC on a decentralized exchange. A simple swap on the base chain typically consumes around 150,000 gas units (a plain ETH transfer is 21,000; a swap touches more contracts and storage). Say the gas price at that moment is 30 gwei.

The fee in ETH is `150,000 units x 30 gwei = 4,500,000 gwei`. Since 1 ETH is 1,000,000,000 gwei, that is `4,500,000 / 1,000,000,000 = 0.0045 ETH`.

If ETH is worth \$3,000, the fee is `0.0045 x \$3,000 = \$13.50`. You pay \$13.50 to move \$2,000 — about 0.7 percent.

Now imagine the network is congested and the gas price jumps to 200 gwei (this happens during manias). The same swap now costs `150,000 x 200 = 30,000,000 gwei = 0.03 ETH`, or `0.03 x \$3,000 = \$90`. The *dollar value* you are moving did not change, but the fee jumped almost 7x because blockspace got scarce.

The intuition: a gas fee is rent on a shared computer's scarce blockspace, so it scales with network congestion and the ETH price, not with the size of your transaction.

This is exactly why layer-2 rollups exist and why the 2024 Dencun upgrade mattered: by bundling thousands of transactions and posting cheap "blob" data to the base chain, a rollup can drop that \$13.50 swap to a few cents. We will return to it.

## Token standards: ERC-20, ERC-721, and the rest

One reason Ethereum became a financial platform rather than a single app is that it standardized what a *token* is, so that any new token works everywhere automatically.

![Token standards on Ethereum as a tree](/imgs/blogs/ethereum-and-programmable-money-7.png)

The tree shows the family. An **ERC-20** is a *fungible* token: every unit is identical and interchangeable, like dollars or shares. The standard is just an agreed-upon list of functions — `totalSupply`, `balanceOf`, `transfer`, `approve`, `transferFrom` — that a token contract must implement. Because wallets and exchanges are written against that interface, the moment you deploy a conforming contract, MetaMask can show its balance and Uniswap can trade it, with zero custom work on their part. Stablecoins (USDC, USDT), governance tokens (UNI, AAVE), and the vast majority of crypto assets are ERC-20 contracts. (Stablecoins specifically — dollar-pegged tokens backed by reserves — are their own deep topic; see [stablecoins and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar).)

An **ERC-721** is a *non-fungible* token, an NFT: each one carries a unique ID, so no two are interchangeable. This is the standard behind digital art, collectibles, in-game items, and tokenized deeds — anything where "this specific one" matters. CryptoPunks and Bored Apes are ERC-721 collections. The contract tracks which address owns which unique ID.

An **ERC-1155** is a *multi-token* standard that can manage many token types — some fungible, some unique — in a single contract, which is efficient for things like game inventories where you might hold 500 identical gold coins and 1 unique sword.

#### Worked example: minting an ERC-20 token

Suppose a project wants to create a governance token called GOV with a fixed supply of 1,000,000 units.

The developer writes a contract that imports a standard ERC-20 template, sets the name to "Governance Token," the symbol to "GOV," and the constructor *mints* 1,000,000 GOV to the deployer's address — meaning it sets `balanceOf(deployer) = 1,000,000` and `totalSupply = 1,000,000`. Deploying this contract costs gas, say roughly 1,200,000 gas units at 30 gwei: `1,200,000 x 30 gwei = 0.036 ETH`, or about \$108 at \$3,000 per ETH. That one-time payment creates the token.

Now the project distributes it. To send 10,000 GOV to a contributor, the deployer calls `transfer(contributorAddress, 10000)`, which costs about 50,000 gas — `50,000 x 30 = 0.0015 ETH`, roughly \$4.50. The contract's internal accounting updates: the deployer's balance drops by 10,000, the contributor's rises by 10,000, and the total supply stays at 1,000,000.

The intuition: a token is not a coin sitting in a wallet — it is a row in a contract's ledger, and "owning" tokens means that contract records your address with a balance.

This standardization is quietly one of the most consequential design choices in crypto. It is the reason a stablecoin issued by one company, a lending market built by another, and a wallet built by a third all interoperate without ever coordinating — they all agree on the shape of a token.

## Composability: the money legos that built a financial system

Now we reach the property that turned a collection of apps into an economy.

![Composability shown as dApps calling each other in one transaction](/imgs/blogs/ethereum-and-programmable-money-3.png)

The way this works is that any contract can call any other contract within a single transaction, and the whole sequence either succeeds together or reverts together (it is *atomic*). The graph above traces a real pattern called a *flash-loan arbitrage*. In one transaction, a user borrows \$1,000,000 from a lending protocol (Aave) with no collateral — because the loan must be repaid by the end of the same transaction or the entire thing reverts as if it never happened. The borrowed funds are routed through two trading venues (Uniswap and Curve) that momentarily disagree on a price. The user buys low on one and sells high on the other, captures the price gap as profit (say \$3,000), repays the \$1,000,000 loan plus a small fee (Aave charges around 0.09 percent), and keeps the difference. If at any step the numbers do not work out — if the profit would not cover the repayment — the transaction reverts and the user is out only the gas.

Read that again, because it is genuinely strange and only possible with composability: you can borrow a million dollars with no collateral, use it, and repay it, all in a few seconds, because the *atomicity* of the transaction guarantees the loan is repaid or nothing happened at all. No bank could offer this. It exists only because the lending contract, the two trading contracts, and the token contracts all speak the same language and execute in one indivisible step.

Flash loans are a vivid example, but the everyday version of composability is more important. A yield aggregator deposits your stablecoins into whichever lending protocol pays the most, automatically. A "vault" borrows against your ETH, swaps the borrowed funds, and stakes them, all in one click, because each step is just a contract call to an existing protocol. This is the substance of **DeFi** (decentralized finance): the lending, trading, and derivatives protocols of [Uniswap, Aave, and MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) are not isolated apps but interlocking pieces that anyone can wire together.

The dark side of composability is *contagion*. If protocols are legos, a defect in one brick can crack everything built on top of it. When a widely used stablecoin de-pegs, every lending market that accepted it as collateral, every pool that paired against it, and every vault that routed through it can break at once — the same plumbing that lets value flow freely also lets failure flow freely. Composability multiplies both the upside and the systemic risk, and serious DeFi risk analysis is largely about mapping which bricks depend on which.

#### Worked example: a smart-contract escrow releasing \$1,000 on a condition

Let us make the self-enforcing rule fully concrete. A freelancer agrees to build a website for \$1,000, paid in USDC.

Step 1 — deploy. An escrow contract is written with three roles: the *buyer* (client), the *seller* (freelancer), and an *arbiter* (a neutral third party, optional). It holds funds and exposes a `release()` function that pays the seller, callable only when the release condition is met.

Step 2 — fund. The client sends 1,000 USDC into the contract. Anyone can verify on-chain that exactly 1,000 USDC sits in the contract's address. The freelancer can see the money is real and locked, not just promised.

Step 3 — work and approve. The freelancer delivers. The client calls `approve()`, which flips an internal boolean `approved = true`. (A more elaborate version requires two of the three roles to sign, protecting against a client who refuses to approve good work — the arbiter can break the tie.)

Step 4 — release. With `approved = true`, calling `release()` transfers the 1,000 USDC to the freelancer's address. The transfer costs perhaps \$5 in gas. If the condition is never met, a `refund()` path after a deadline can return the funds to the client.

At no point did a bank, a lawyer, or an escrow company hold the money. The 1,000 USDC sat in code that could only pay out the way it was written.

The intuition: a smart-contract escrow replaces a trusted custodian with a transparent program — the money is visibly locked, and the only way out is the path written into the code.

## The Merge: from proof-of-work to proof-of-stake

For its first seven years, Ethereum secured itself the way Bitcoin does — with proof-of-work mining. Miners worldwide ran specialized hardware racing to solve a cryptographic puzzle; the winner added the next block and earned newly issued ETH plus fees. It worked, and it was secure, but it consumed electricity on the order of a mid-sized country's usage, and it issued a large amount of new ETH every day to pay the miners.

In September 2022, Ethereum executed *the Merge*: it swapped that mining engine for proof-of-stake while the network stayed live, a feat often compared to changing a plane's engine mid-flight. It is arguably the most consequential planned upgrade in crypto history, and it changed three things at once.

![Proof-of-work versus proof-of-stake before and after the Merge](/imgs/blogs/ethereum-and-programmable-money-4.png)

The before-and-after captures it. Under **proof-of-work**, security came from miners burning electricity — roughly 78 terawatt-hours per year at the peak, comparable to a small country — and the network issued on the order of 13,000 ETH per day to reward them. Under **proof-of-stake**, security comes from validators who lock up 32 ETH each as collateral; there is no puzzle to solve, so energy use collapsed to around 0.01 terawatt-hours per year — roughly a 99.9 percent reduction — and issuance dropped to on the order of 1,700 ETH per day, since you no longer need to pay miners' electricity bills. (All these figures are approximate and as of the early-2020s data behind the Merge.)

Three consequences followed:

1. **Energy.** The most visible change. Ethereum went from one of the most energy-intensive blockchains to one of the most efficient, essentially overnight, because proof-of-stake does not require burning electricity to compete for blocks.

2. **Issuance and supply.** Far less new ETH is created. Combined with the fee-burning introduced in 2021, Ethereum's net issuance can even go *negative* during busy periods — more ETH burned in fees than minted in rewards — making the supply mildly deflationary at times. Whether that matters for price is not something this post will speculate on; the mechanical fact is that the supply schedule changed dramatically.

3. **Security model.** Attacking the chain no longer means out-spending everyone on hardware and power. It means acquiring and risking an enormous amount of staked ETH, which the network can *slash* (confiscate) if you misbehave. The cost of attack shifted from an ongoing electricity bill to bonded capital that you stand to lose.

The trade-offs are real and debated. Critics argue proof-of-stake can favor those who already hold the most ETH ("the rich get richer" through staking rewards) and that it is more complex and less battle-tested than proof-of-work. Supporters counter that slashing makes attacks economically self-punishing and that the energy savings are decisive. Both are fair points; the Merge resolved the engineering question (it worked) while leaving the philosophical one open.

#### Worked example: the Merge cutting energy use ~99.9 percent

Put the energy claim in numbers. Before the Merge, estimates put Ethereum's annual electricity consumption around 78 terawatt-hours (78,000,000,000 kilowatt-hours) — in the ballpark of a country like Chile or Austria.

After the Merge, proof-of-stake validators are essentially ordinary computers; estimates put the network's draw around 0.01 terawatt-hours per year (10,000,000 kWh).

The reduction is `1 - (0.01 / 78) = 1 - 0.000128 = 0.99987`, or about 99.99 percent — usually rounded to "99.9 percent." Put differently, post-Merge Ethereum uses roughly `78 / 0.01 = 7,800` times less energy than before.

If you priced the *saved* electricity at \$0.10 per kWh, the annual saving is about `(78,000,000,000 - 10,000,000) x \$0.10 = ~\$7.8 billion` worth of electricity no longer burned.

The intuition: proof-of-stake secures the chain with money at risk instead of energy spent, so the same security no longer costs a country's worth of power.

## Staking and Lido

Proof-of-stake created a new role anyone with ETH can play: validator, or backer of one. To run your own validator you must stake exactly 32 ETH (locking it as collateral) and run validator software with near-perfect uptime. In return you earn staking rewards — newly issued ETH plus a share of transaction tips — for proposing and attesting to blocks honestly. Misbehave or go offline at the wrong time and you are *slashed*, losing part of your stake. Staking is how the network pays for its own security.

Two problems made raw staking impractical for most people. First, 32 ETH is a lot of money (tens of thousands of dollars). Second, your staked ETH was *locked* and illiquid — you could not sell it or use it elsewhere while it secured the chain. These frictions created an opening, and it was filled largely by **Lido**, a *liquid staking* protocol.

Lido works like a pool. You deposit any amount of ETH — 0.1, 5, whatever — and Lido aggregates deposits into 32-ETH validator units run by professional operators. In return, you receive a token called **stETH** that represents your staked deposit plus accruing rewards. The key word is *liquid*: stETH is an ERC-20 token, so you can sell it, lend it, or use it as collateral in other DeFi protocols while your underlying ETH keeps earning staking rewards. You get the yield without the lockup. Because of this convenience, Lido grew to control a very large share of all staked ETH — at times around 30 percent — which is itself a debated risk, because too much stake concentrated in one protocol could threaten the network's decentralization. The convenience and the concentration are two faces of the same coin.

#### Worked example: staking 32 ETH and the annual yield

Suppose you stake exactly 32 ETH to run a solo validator, and the network's staking reward rate is about 3.5 percent per year (this rate floats with how much total ETH is staked — more stakers means a lower rate per validator).

Your annual reward is `32 x 3.5% = 1.12 ETH`. At an ETH price of \$3,000, that is `1.12 x \$3,000 = \$3,360` per year, on a staked position worth `32 x \$3,000 = \$96,000`.

But the yield is denominated in *ETH*, not dollars. If ETH's price falls 20 percent to \$2,400, your 32 ETH is now worth \$76,800 and your 1.12 ETH reward is worth `1.12 x \$2,400 = \$2,688`. You earned the same 1.12 ETH, but its dollar value and your principal's dollar value both dropped. Staking yield does not protect you from ETH price moves; it only adds ETH on top.

Through Lido, you skip the 32-ETH minimum and the lockup: deposit 5 ETH, receive stETH, earn the same percentage (minus Lido's fee, around 10 percent of rewards), and keep the stETH liquid. Your gross reward on 5 ETH at 3.5 percent is `5 x 3.5% = 0.175 ETH`; after a 10 percent fee you net about `0.175 x 0.90 = 0.1575 ETH` per year.

The intuition: staking pays a yield *in ETH* for locking capital to secure the chain, but it is not a risk-free deposit — the principal still rides the ETH price, and slashing can take a slice.

## Scaling via rollups and layer-2s

Ethereum's base chain is deliberately slow: it processes only about 15 transactions per second, because every one of its roughly one million validators must process every transaction for security. When demand for that scarce blockspace spikes, gas fees soar — sometimes to \$50 or \$100 for a single swap, as happened repeatedly during the 2020–2021 booms. A world computer that costs \$100 to use is not a world computer for most people.

The dominant scaling answer is the **layer-2 rollup**. The idea: do the heavy computation on a separate, faster chain (the layer-2), but periodically post a compressed record back to Ethereum's base chain (the layer-1), which remains the ultimate source of truth. Thousands of layer-2 transactions get "rolled up" into a single batch settled on layer-1, so the expensive base-chain cost is amortized across all of them. The rollup inherits the base chain's security — if the rollup operator tries to cheat, the layer-1 can catch and reject it — while charging users a small fraction of layer-1 fees.

There are two main flavors. *Optimistic rollups* (Arbitrum, Optimism, Base) assume batches are valid and allow a challenge window during which anyone can submit a fraud proof; *zero-knowledge rollups* (zkSync, Starknet, Linea) post a cryptographic proof that the batch is valid, which is more complex but settles faster. The 2024 Dencun upgrade added "blobs" — a cheap, temporary data lane on layer-1 specifically for rollup data — which cut layer-2 fees by roughly another order of magnitude.

The trade-off, named honestly: a rollup is not the base chain. It adds assumptions — that its operators (or "sequencers") will not censor or reorder your transactions unfairly, that bridges moving assets between layers are not hacked (bridge hacks have been among the largest in crypto), and that fraud proofs or validity proofs actually work as designed. You gain cheap, fast transactions; you take on the rollup's specific risks. For most users most of the time this trade is worth it, which is why the great majority of Ethereum activity has migrated to layer-2s — but "cheaper and faster, with extra trust assumptions" is the accurate summary, not "free and trustless."

## The DAO hack and the fork: Ethereum versus Ethereum Classic

If the escrow and flash-loan examples show the upside of self-enforcing code, the 2016 DAO hack shows the downside in its purest form — and it produced the most important philosophical split in Ethereum's history.

The DAO ("Decentralized Autonomous Organization") was an ambitious 2016 project: a smart contract that pooled investors' ETH into a venture fund governed entirely by token-holder votes, with no managers. It raised an enormous sum for the time — over \$150 million worth of ETH, around 14 percent of all ETH then in existence. Then, in June 2016, an attacker found a flaw.

The bug was a *reentrancy* vulnerability. The DAO's withdrawal function sent ETH to the caller *before* updating its internal record of the caller's balance. The attacker's contract exploited the ordering: when it received the ETH, it immediately called back into the withdrawal function *again*, before the balance had been zeroed out — and again, and again, recursively draining the DAO in a loop. The contract did exactly what its code said; the code just said the wrong thing. Roughly 3.6 million ETH (about \$60 million at the time) was siphoned out.

This created an agonizing dilemma. The whole premise of Ethereum was that *code is law* — contracts run as written, immutably, with no human override. But here the code's literal execution had stolen \$60 million through an obvious bug. Should the community honor immutability and let the theft stand? Or should it change the rules to claw the money back?

The community split, and the resolution was a **hard fork**. The majority chose to alter the blockchain's history at the protocol level to move the stolen funds back to a recovery contract, effectively undoing the hack. This forked chain is the **Ethereum** we use today. A minority refused on principle — "code is law, even when it hurts" — and kept running the original, un-reverted chain, which became **Ethereum Classic (ETC)**. From that block on, two chains exist: the same history up to the fork, then divergent.

#### Worked example: the DAO drain and the fork math

The DAO held roughly 3.6 million ETH stolen by the attacker. At the ETH price of around \$20 in mid-2016, that was about `3.6M x \$20 = \$72 million` (often cited as ~\$50–60M as the price moved during the event).

Consider what the fork did to a holder. Before the fork, you owned, say, 100 ETH on one chain. After the fork, the history duplicated: you now held 100 ETH on the new forked Ethereum *and* 100 ETC on Ethereum Classic — the same balance copied onto both chains, because they shared all history up to the split. The two coins then traded independently. If ETH was worth \$13 and ETC settled around \$1 shortly after, your 100 ETH + 100 ETC was worth `100 x \$13 + 100 x \$1 = \$1,400`, versus \$1,300 if only the new chain had value.

The intuition: a hard fork does not destroy coins — it clones the entire ledger, so every holder ends up with a balance on both resulting chains, and the market decides what each is worth.

The lasting lessons are two. First, smart-contract security is brutally unforgiving: a single mis-ordered line cost \$60 million, and reentrancy bugs remain a top cause of hacks years later. Second, "code is law" is an ideal, not a guarantee — when the stakes were high enough, a human community chose to override the code, which tells you something honest about where the ultimate authority lies. The split into Ethereum and Ethereum Classic is the permanent monument to that choice.

## Ethereum as a financial operating system

Step back and the pieces assemble into something coherent: Ethereum behaves like an *operating system* for finance. An operating system provides shared services — storage, identity, a way for programs to call each other — so that applications do not each reinvent the basics. Ethereum provides exactly that for money.

- **Settlement** is the base chain: the final, authoritative record of who owns what.
- **Native currency and a fee market** is ETH and gas: the unit you pay for computation and the mechanism that prices scarce blockspace.
- **A standard for assets** is the ERC token standards: any program can issue dollars (stablecoins), shares, deeds, or points and have them work everywhere.
- **Interoperability** is composability: programs call each other, so a new app inherits the entire existing ecosystem.
- **Identity and custody** is the account model: your address is your login, your wallet holds your keys, and contracts can custody assets under fixed rules.
- **A faster execution tier** is layer-2: where most users actually transact cheaply, settling back to the base chain.

On this operating system, an entire parallel financial system has grown: lending and borrowing markets, decentralized exchanges, derivatives, stablecoins that move trillions of dollars in volume, prediction markets, and more. It runs 24/7, is open to anyone with an internet connection, and exposes its rules and balances publicly on-chain. The promise is a financial system that is transparent and permissionless by default rather than gated and opaque.

The honest counterweight: an operating system for finance with *no off switch and no fraud department* is as dangerous as it is powerful. The same openness that lets a developer in any country deploy a lending market also lets a scammer deploy a rug-pull; the same composability that builds money legos also propagates failures; the same immutability that prevents censorship also prevents fixing a drained contract. Ethereum did not abolish financial risk — it *relocated* it from institutions and regulators into code and economic incentives, where it takes new and often sharper forms.

## Competitors: Solana and the others

Ethereum is not the only programmable blockchain, and its design involves trade-offs that competitors attack from different angles.

![Ethereum versus Solana and other base chains compared](/imgs/blogs/ethereum-and-programmable-money-6.png)

The matrix sketches the core tension. **Ethereum's base layer** is deliberately slow (around 15 transactions per second) and can be expensive, but it is the most decentralized — roughly a million validators, runnable on modest hardware — which is its security argument. **Solana** takes the opposite bet: it processes thousands of transactions per second at sub-cent fees by requiring beefy, expensive validator hardware, which means fewer (on the order of a couple thousand) validators and a more centralized network that has suffered several full outages. **Ethereum plus layer-2s** tries to get the best of both: cheap, fast transactions on rollups that still settle to Ethereum's heavily decentralized base chain, at the cost of the bridge and sequencer trust assumptions discussed earlier.

Other notable platforms each make their own trade: some emphasize developer ergonomics or formal verification, some optimize for specific app categories, and some are "Ethereum-compatible" chains that run the same EVM so existing contracts port over easily. The recurring theme is the *blockchain trilemma* — the rough observation that it is hard to maximize decentralization, security, and scalability all at once, so each chain sacrifices something. Ethereum sacrifices base-layer speed for decentralization and pushes speed up to layer-2; Solana sacrifices some decentralization for raw throughput. There is no free lunch, and "which is best" depends entirely on what you are willing to give up.

A note on neutrality: this is a comparison of design choices, not an endorsement. Throughput numbers, fees, and validator counts move constantly and the figures here are approximate and as-of the mid-2020s; treat them as orders of magnitude, not quotes.

## Common misconceptions

**"Ethereum is just another cryptocurrency, like a competitor to Bitcoin."** ETH is a cryptocurrency, but Ethereum is a *platform*. Bitcoin is, by design, mostly one application: a digital money ledger. Ethereum is a general-purpose computer that can run any program, of which a currency is just one. Comparing them as "two coins" misses that one is a money and the other is a computing platform that happens to have a money inside it. They are answering different questions.

**"Smart contracts are legally binding contracts."** They are not, and the name causes endless confusion. A smart contract is *code that self-executes* — it enforces its own rules mechanically. It is not a legal agreement, it does not reference any jurisdiction, and a court will not interpret its "intent." If the code does something the parties did not want (as in the DAO hack), the code wins on-chain, regardless of what anyone "meant." Some real-world arrangements pair a legal contract *with* a smart contract, but the two are different objects.

**"Gas fees are Ethereum's profit, like a company's revenue."** No single company collects gas fees. Since 2021, the *base fee* is **burned** — destroyed, benefiting no one directly — and only the smaller *priority tip* goes to whichever validator includes your transaction. Ethereum is not a corporation with a P&L; it is a protocol, and "fees" are partly a congestion price and partly a payment to the decentralized set of validators, not earnings flowing to a headquarters.

**"Proof-of-stake means whoever has the most ETH controls Ethereum."** Stake influences *who proposes blocks*, roughly in proportion to stake, but it does not give anyone the power to rewrite the rules or steal funds. Validators who break the protocol get *slashed* — they lose stake. Changing the actual rules of Ethereum requires the broad community of node operators to voluntarily run new software, which a large staker cannot force. Concentration of stake (e.g. in Lido) is a genuine decentralization concern, but it is not the same as "control."

**"If a smart contract has a bug, you can just call support and reverse it."** There is no support line and, normally, no reversals. A deployed contract executes as written; a drained contract stays drained. The DAO fork is the famous exception that *proves the rule* — it required a contentious, community-wide hard fork that split the chain in two, precisely because reversing on-chain actions is supposed to be impossible. Treat on-chain transactions as final.

**"Layer-2s are just Ethereum, so they carry no extra risk."** Layer-2s inherit Ethereum's settlement security but add their own: sequencer centralization (one operator orders transactions), bridge risk (the contracts moving assets between layers have been hacked for hundreds of millions of dollars), and the soundness of their fraud or validity proofs. Cheaper and faster, yes — but with strictly more moving parts than transacting on the base chain.

## How it shows up in real markets

**The DAO hack and the Ethereum / Ethereum Classic fork (2016).** Roughly \$60 million in ETH drained through a reentrancy bug, and the community's decision to hard-fork and reverse it split the chain permanently into Ethereum and Ethereum Classic. It remains the defining case study in smart-contract risk and in the limits of "code is law." Every serious Solidity audit since has reentrancy at the top of its checklist precisely because of this episode.

**DeFi summer and the gas-fee spikes (2020).** As lending and yield-farming protocols exploded in mid-2020, demand for blockspace overwhelmed the base chain. Gas prices routinely hit levels where a single swap or deposit cost \$20, \$50, even over \$100. It was the clearest demonstration that the world computer's blockspace is a scarce, auction-priced resource — and the urgent motivation for the entire layer-2 scaling effort.

**NFT mania (2021).** ERC-721 collections went mainstream, with profile-picture projects and digital art selling for staggering sums and a handful of pieces fetching tens of millions of dollars. It validated the non-fungible token standard as a real primitive — and also produced a speculative bubble, a wave of low-effort and fraudulent mints, and yet more gas congestion. Both the genuine innovation and the froth were real, and the prices of most collections later collapsed.

**The Merge (September 2022).** Ethereum switched from proof-of-work to proof-of-stake live, without halting, cutting energy use roughly 99.9 percent and slashing new ETH issuance. It is the largest planned change to a major blockchain's core mechanism ever executed successfully, and it reset Ethereum's environmental profile and supply economics in a single step.

**The layer-2 explosion (2023–2024).** Following the Merge and especially the 2024 Dencun upgrade (which added cheap "blob" data for rollups), the majority of Ethereum activity migrated to layer-2 rollups like Arbitrum, Base, and Optimism, with fees dropping to cents. It is the live, ongoing answer to the 2020 fee crisis — and it shifted where users actually transact away from the base chain, along with the new trust assumptions that move entails.

**Bridge and contract exploits (ongoing).** Some of the largest crypto thefts have been cross-chain *bridge* hacks — the contracts that move assets between Ethereum and other chains or layer-2s — with individual incidents exceeding hundreds of millions of dollars. They are the recurring reminder that composability and cross-chain plumbing concentrate value in code that, if flawed, fails catastrophically and irreversibly. The DAO's lesson never stopped being relevant; it just got more expensive.

## When this matters to you, and further reading

You do not need to use Ethereum to benefit from understanding it, but here is when the mental model pays off:

- **When you read about "DeFi," a "hack," or a "rug pull."** The headline almost always reduces to one of the mechanics here: a contract bug (the DAO pattern), a composability-driven contagion, a bridge exploit, or a gas/congestion story. Knowing the building blocks turns a scary, opaque headline into a comprehensible failure mode.
- **When someone pitches you a token with a guaranteed yield.** Now you can ask the right questions: where does the yield come from — real staking rewards and fees, or new deposits paying old depositors (a Ponzi)? Is your capital locked or liquid? What happens to the token's dollar value if the underlying asset falls? A 20 percent "APY" with no clear source is a red flag, not an opportunity.
- **When you compare blockchains.** The trilemma framing (decentralization vs security vs scalability) lets you see past marketing: a chain bragging about speed is usually trading away decentralization, and a chain bragging about decentralization is usually slow at its base. Ask what each one *gave up*.
- **When you think about where finance is going.** Whether or not crypto "wins," the idea of programmable, composable, self-settling money is now out of the box, and regulated institutions are experimenting with the same primitives (tokenized assets, on-chain settlement). Understanding Ethereum is understanding the prototype.

The risks deserve the last word, plainly. Smart-contract code can be exploited and the loss is usually permanent. Keys, once lost or stolen, mean funds gone with no recourse. Tokens and ETH are volatile, and a staking yield does not protect the principal's dollar value. Layer-2s and bridges add trust assumptions on top of the base chain's. Nothing here is investment advice, and nothing here predicts a price — the goal is only that you understand the machine well enough to evaluate any specific claim about it for yourself.

For the companion pieces that go deeper into the parts: [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision) for where the idea of trustless money began; [DeFi protocols: Uniswap, Aave, and MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) for the lending and trading apps built on the composability described here; [stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) for the dollar-pegged tokens that became DeFi's base money; and [a field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) for the traditional intermediaries Ethereum is, in part, trying to replace with code.
