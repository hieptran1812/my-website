---
title: "Addresses, Wallets, and Contracts: What You're Actually Looking At"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-to-deep guide to what a blockchain address really is, why an EOA is not a smart contract, why a wallet is rarely one address, and how to read all of it on an explorer."
tags: ["onchain", "crypto", "addresses", "wallets", "smart-contracts", "eoa", "multisig", "account-abstraction", "ethereum", "etherscan"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — An "address" on a blockchain is not a person, and a "wallet" is almost never one address; before you can read flows, cluster entities, or trace stolen funds, you have to know exactly what you're looking at.
>
> - An **address** is a short public string derived from a key or from code. There are two kinds of account behind one: an **EOA** (externally-owned account, controlled by a private key) and a **smart contract** (an address whose owner is code on the chain).
> - **How to read it**: on an explorer like Etherscan, a contract shows a **Contract** tab with code; an EOA does not. The address format also tells you the chain — `0x…` is EVM, `bc1…` is Bitcoin, a long Base58 string is Solana, a `T…` string is Tron.
> - **What you do with it**: stop treating one address as one person. A real-world entity (an exchange, a fund, a treasury) spans **many** addresses — hot, cold, and one deposit address per user. Counting its balance, or attributing a trade, means summing across the cluster.
> - The one rule to remember: **whoever controls the key controls the address.** Custodial exchange = the exchange holds the key. Non-custodial wallet = you do. A multisig means *several* keys must agree.

On 21 February 2025, the crypto exchange Bybit signed what its operators believed was a routine transfer between its own wallets. Within minutes, roughly **\$1.46 billion** of Ether had left its cold storage and landed in addresses controlled by the Lazarus Group, North Korea's state hacking operation — the largest single theft in the history of money, by some accounting. The attackers had compromised the *signing process*, not the chain itself. The Ethereum network did exactly what it was told: it moved the funds because the transaction carried valid signatures.

Here is the detail that matters for this post. Within hours, blockchain investigators were publicly tracing where the money went — not because they had subpoenaed anyone, but because every address that touched the stolen funds was sitting in plain sight on the public ledger. They could see the cold wallet that was drained, the intermediary addresses the funds were split across, and the contracts the thieves used to swap and obfuscate. They argued (correctly, as it turned out) about which addresses were the attacker's, which belonged to exchanges, and which were just routing hops. That entire investigation is a single skill applied at scale: **reading addresses, wallets, and contracts for what they actually are.** Get that wrong and you mislabel a victim as a thief, or a routing contract as a person. Get it right and the chain reads like a confession.

This is the bedrock chapter. Almost everything later in this series — clustering, attribution, exchange-flow analysis, tracing a hack — is built on the distinctions below. So we start from zero: what a key is, what an address is, why a wallet is not one address, and how to tell a person-controlled account from a piece of code.

![EOA vs contract account mental model showing a private key controlling an EOA and code controlling a contract](/imgs/blogs/addresses-wallets-and-contracts-1.png)

## Foundations: keys, addresses, and what "controlling" an account means

Before any of the on-chain tooling makes sense, four ideas have to be solid. They are not crypto-specific magic; they are the same public-key cryptography that secures your bank login and the padlock in your browser. We will define each from the ground up.

### A private key is just a very large secret number

A **private key** is a randomly chosen number. On Ethereum and Bitcoin it is 256 bits long — a number with about 78 digits, drawn from a space so large (roughly 10⁷⁷ possibilities) that no one can guess yours and no two users will ever collide by accident. That is the whole secret. Everything you own on a blockchain is "owned" in exactly one sense: **you know a number that no one else knows.**

From that private key, a one-way mathematical function (elliptic-curve multiplication) produces a **public key**. "One-way" is the load-bearing word: it is cheap to go from private key to public key, and computationally hopeless to go backward. You can hand out your public key freely; no one can run the math in reverse to recover the secret.

From the public key, another one-way function (a cryptographic hash) produces your **address**. On Ethereum the address is the last 20 bytes of the Keccak-256 hash of the public key, written in hexadecimal with a `0x` prefix — something shaped like `0xA11ce…42f`. The address is what you share to receive funds; it is the public face of the secret number you keep.

So the chain of derivation is strictly one-directional:

> private key → public key → address

You can always move *right* along that arrow. You can never move *left*. That asymmetry is the entire security model. Let me make the pipeline concrete.

![Pipeline from private key to public key to hashed address with signing step](/imgs/blogs/addresses-wallets-and-contracts-2.png)

The figure above shows why an address is **pseudonymous, not anonymous**. The address itself contains no name, no email, no country. But it is also permanent and public: every transaction it ever signs is recorded forever, visible to everyone. Anonymity would mean no one can ever connect the address to you. Pseudonymity means the address is a *pen name* — and pen names get unmasked. The moment you withdraw to it from an exchange that knows your identity, fund it from a wallet that's already labeled, or reuse it across services, the behavior starts pointing back to a real person. That tension — public-but-pseudonymous — is the reason this whole field of on-chain analysis exists.

A reasonable beginner question at this point is: if anyone can generate keys for free, why can't someone just generate keys until they land on *my* address and steal my funds? The answer is the sheer size of the keyspace. A 256-bit private key is one of roughly 2²⁵⁶ ≈ 10⁷⁷ possibilities. That number is not just "big"; it is comparable to the number of atoms in the observable universe. Even if every computer on Earth tried trillions of keys per second for the entire age of the universe, the fraction of the space they'd cover would round to zero. This is why "ownership = knowing a secret number" works at all: the secret is drawn from a space so vast that finding a *specific* occupied address by brute force is not improbable, it is physically impossible. The practical risks are never "someone guessed my key" — they are that you leaked it, stored it badly, signed something malicious, or trusted a custodian who lost it. Every real-world crypto loss traces back to one of those, not to the math failing.

One more consequence of free key generation deserves a flag, because it surfaces constantly in analysis: **addresses do not have to be "created" before they can receive funds.** Any of the 10⁷⁷ possible addresses can be sent to right now, whether or not anyone holds its key. This is why you can send funds to a typo'd address (the funds sit there, permanently unspendable, because no one knows the key), and it's part of why address poisoning works — the attacker's lookalike is a perfectly valid destination the instant they grind it. An address only "appears" on the chain when it first transacts; until then it is a latent slot in an unimaginably large space.

### "Controlling" an address means being able to sign for it

When you spend from an address, you don't "send a coin." You broadcast a **transaction** — a message that says "move X from this address to that one" — and you attach a **digital signature** produced with the private key. The network verifies that the signature matches the address's public key. If it does, the transaction is valid. If not, it is rejected.

Crucially, **you never reveal the private key to do this.** The signature proves you know the key without disclosing it (this is the magic of asymmetric cryptography). So "controlling" an address is precise: it means being able to produce valid signatures for it, which means knowing the private key. Whoever knows the key controls the address — full stop. There is no password reset, no support line, no "I am the rightful owner" appeal. The key *is* ownership.

This is also why a leaked key is catastrophic and irreversible. If an attacker learns your private key, they can sign transactions indistinguishable from yours, and the chain will honor them. The Bybit theft was, at its root, a signing-process compromise: the attackers got a valid signature produced for a malicious transaction.

> [!note]
> **Seed phrase vs private key.** When a wallet app shows you 12 or 24 English words to write down, that "seed phrase" (or "recovery phrase") is a human-friendly encoding of a *master secret* from which many private keys are derived. Lose the seed and you lose every key it generates. Anyone who reads your seed controls every address under it. Treat those words exactly like the keys they unlock.

### Two kinds of account live at one kind of address

Here is the distinction that trips up almost every beginner, and the spine of this entire post. On Ethereum and EVM chains, there are **two types of account**, and they share the same `0x…` address format:

- An **EOA** — *Externally-Owned Account* — is an account controlled by a private key. "Externally owned" means the controller is *external* to the chain: a human (or a script) holding a key, outside the blockchain's own logic. An EOA has **no code**. It can hold funds and tokens, and it can *initiate* transactions by signing them.
- A **smart contract** (a *contract account*) is an address whose controller is **code deployed on the chain itself**. A contract has a program attached to its address. It cannot initiate a transaction on its own — something has to call it — but when called, it runs its code: it can move funds, call other contracts, mint tokens, enforce rules. A Uniswap pool, a USDC token, a Gnosis Safe, an NFT collection — all of these are contract accounts.

The single most useful sentence to memorize: **an EOA is controlled by a key; a contract is controlled by its code.** The cover figure at the top of this post draws exactly this fork. A key derives an EOA; deployed code becomes a contract; both live in the same `0x…` address space, but what *controls* them is completely different.

Why does this matter for on-chain analysis? Because a contract has no private key — there is no person who "is" a Uniswap pool. If you see \$50 million flow into a contract address, that is not someone's net worth; it is liquidity sitting in a program. If you see funds move *out* of a contract, no human signed for it directly — a function was called under rules you can read. Confusing the two is how amateurs mislabel a routing contract as a "whale" or a token contract's balance as someone's holdings.

### A wallet is software, an account is a key, an address is a string

These three words get used interchangeably, and the sloppiness causes real errors. Pin them down:

- A **wallet** (like MetaMask, Phantom, Ledger, or a Gnosis Safe) is the *thing that manages your keys and signs transactions for you*. A wallet app is not on the chain at all — it's software (or hardware) on your side. One wallet app typically manages **many** accounts.
- An **account** is one key pair (one private key + its public key). One wallet can hold dozens of accounts.
- An **address** is the public string for one account. One account = one address (on EVM). One wallet = many accounts = many addresses.

So when someone says "my wallet," they might mean their MetaMask app (software), one account inside it, or a single address. In on-chain analysis we are almost always talking about **addresses** (the on-chain object) and **entities** (the real-world person or organization behind a cluster of addresses). The "wallet" — the app — is invisible on-chain. You never see "MetaMask" on Etherscan; you see the addresses MetaMask signs for.

A concrete way to feel this: install MetaMask and it asks you to back up a 12-word seed phrase. From that single seed, it derives "Account 1" with one address. Click "Add account," and MetaMask derives "Account 2" — a *different* address, with its own private key, but generated deterministically from the *same* seed. You can keep going: Account 3, 4, 5, each a distinct on-chain address, all recoverable from those same 12 words. This is the **hierarchical-deterministic (HD) wallet** standard (BIP-32/39/44), and it means a "wallet" in the everyday sense is one seed fanning out into an unlimited tree of addresses. To an outside observer on Etherscan, those five accounts look like five unrelated addresses — there is nothing on-chain that says "we share a seed." They only become linkable when their *behavior* connects them: one funds another, they transact with the same counterparties, or they move in correlated patterns. This is the same many-addresses-one-entity problem the whole series circles back to, now visible inside a single person's everyday wallet.

The reason this matters so much for analysis is that it cuts both ways. A privacy-conscious user can deliberately spread activity across many HD-derived addresses to fragment their on-chain footprint — and a naive analyst will count them as many separate small players. An investigator, conversely, looks for the behavioral seams that re-link them. Neither the privacy nor the deanonymization is about the cryptography of the seed; it is about whether the *addresses transact in ways that reveal common control*. Knowing that one wallet routinely means many addresses is the mental correction that prevents the most common counting error in the field.

### Custodial vs non-custodial: who holds the key?

The last foundational split. When you buy crypto on Coinbase or Binance and leave it there, **the exchange holds the private keys.** Your "balance" on the exchange is a number in their database — an IOU. You log in with a password (which the exchange can reset), not a private key. This is **custodial**: a third party custodies your keys, and therefore controls your coins. You are trusting them not to lose, freeze, or steal them. (FTX held its customers' keys; when it collapsed in 2022, customers discovered exactly what "the exchange holds your keys" means.)

When you withdraw to your own MetaMask or hardware wallet, **you** hold the key. No one can freeze, reverse, or seize those funds without the key. This is **non-custodial**: "not your keys, not your coins." The flip side is brutal: lose the key (or the seed phrase), and the funds are gone forever — there is no reset.

![Custodial versus non-custodial comparison showing who holds the private key in each case](/imgs/blogs/addresses-wallets-and-contracts-7.png)

This distinction governs *how addresses behave on-chain*, which is why it belongs in the foundations. An exchange's addresses are a handful of company-controlled wallets servicing millions of users (we'll dissect this below). A non-custodial user's address is theirs alone. When you see funds move from a labeled exchange address to an unlabeled one, you're very likely watching a withdrawal — a user taking custody. When you see funds move *to* an exchange, you're watching a deposit — a user (or a thief) handing over custody, often to sell. Reading that direction correctly is the foundation of exchange-flow analysis later in the series.

#### Worked example: an exchange's one-database-row balances

Say an exchange has 1,200,000 users, and the average balance is \$1,800. The exchange's database shows total customer liabilities of 1,200,000 × \$1,800 = \$2.16 billion. But on-chain, those balances do **not** sit in 1.2 million separate user-controlled addresses. They are pooled in a small number of exchange-controlled wallets — perhaps one big cold wallet holding ~85% (\$1.836 billion) and a hot wallet holding the rest (\$324 million) for day-to-day withdrawals. Each user sees "\$1,800" in the app, but the keys to that \$1,800 are held by the exchange. If the exchange's cold-wallet key is compromised — as Bybit's effectively was — \$1.836 billion can leave in one signed transaction, while every user's app still shows their familiar balance until the music stops. **The lesson: a custodial balance is an IOU against a key you don't hold.**

## EOA vs contract: how to tell them apart on an explorer

You now know the conceptual difference. Here is how you actually *see* it, because this is the single most common thing you'll do on a block explorer.

Open [Etherscan](https://etherscan.io) and paste any `0x…` address into the search bar. The address page loads. Now look for these tells:

1. **The Contract tab.** A contract account shows a tab labeled **Contract** (often with a green checkmark if the source code is verified, or a red warning if it has special features). An EOA has no such tab — just Transactions, Token Transfers, and so on. This is the fastest, most reliable check: *Contract tab present → it's a contract; absent → it's an EOA.*
2. **The label under the address.** Etherscan tags many known contracts and entities: you'll see chips like `Uniswap V3: Router`, `Tether: USDT Stablecoin`, `Binance: Hot Wallet`. These are Etherscan's labels, crowd-sourced and curated — useful, but *not* authoritative (we'll return to the danger of trusting labels).
3. **"Contract Creator" line.** A contract page shows who *deployed* it and in which transaction ("Contract Creator: 0x… at txn 0x…"). An EOA has no creator — it springs into existence the first time it receives funds or is derived from a key; it is never "deployed."
4. **Code.** Click the Contract tab and you can read the contract's bytecode, and if verified, its Solidity source. An EOA has nothing to show — there is no code at the address.

Let me put the practical heuristic in one box.

> [!tip]
> **The 5-second EOA-vs-contract check on Etherscan.** Does the address page have a **Contract** tab? Yes → smart contract (it has code; look for the verified-source green check and the "Contract Creator" line). No → EOA (controlled by a private key; look at funding source and first/last activity instead). When in doubt, the presence of code is the definition.

This matters constantly in practice. Suppose you're investigating a token and you see a big address holding 12% of supply. Is that a whale to worry about, or the liquidity pool? Check for a Contract tab. If it's `Uniswap V3: USDC/WETH Pool`, that 12% is shared liquidity, not one person's stash. If it's an EOA with no label, that *might* be a whale — or a deployer wallet, or an exchange's omnibus address. The Contract tab is the first fork in every investigation.

There's also a subtle behavioral tell that survives even when labels are missing. **An EOA can be the `from` of a transaction (it can initiate, because a key signed it). A contract can never be the top-level `from`** — it only acts when called. So if you see an address appear as the originating sender of transactions, it's an EOA. If it only ever appears as a `to` (a call target) or as an internal actor moving funds *within* a transaction another address initiated, it's a contract. This is sometimes the only way to classify an unverified address whose code is just bytecode.

This `from`-versus-called distinction connects to a second tab you should learn to read: **Internal Transactions**. When a contract moves funds during execution — a router forwarding ETH to a pool, a Safe paying out to a recipient — those movements are *not* separate signed transactions; they are internal calls triggered inside the one transaction that an EOA originated. Etherscan surfaces them on the "Internal Txns" tab. The practical upshot: if you only look at the normal Transactions tab, you will *miss* funds that flowed into or out of an address via contract calls, because those funds never appear as a top-level transfer. When tracing where money actually went — especially through DeFi protocols or a hack's laundering route — the internal-transactions view is where the truth lives. A whale that "received nothing" on the normal tab may have received millions via internal transfers from a contract it interacted with.

There is also a funding-and-gas dimension that helps you read an address's nature. Every transaction on Ethereum costs **gas**, paid in ETH by the EOA that signs it. A contract cannot pay its own gas to *start* something — it has no key to sign with — so something must always poke it. When you see a fresh EOA that has *only* ever received a small amount of ETH and then immediately started signing transactions, you are often looking at a wallet that was funded specifically to act — the gas had to come from somewhere, and that funding source is a clustering clue. Conversely, a contract that holds large balances but has never "spent gas" of its own is behaving exactly as a contract should. Reading who pays the gas, and where the first ETH came from, is one of the oldest tricks for connecting a new address back to an entity that funded it.

#### Worked example: is this 12%-of-supply holder a whale?

A new token has a 100,000,000 total supply trading at \$0.40, so a fully-diluted value of \$40 million. Etherscan shows address `0xC0ffee…` holding 12,000,000 tokens — 12% of supply, worth \$4.8 million. A naive analyst flags a "whale" who could dump and crash the price. You open the address: it has a **Contract** tab labeled `Uniswap V3: Pool`. That 12% is not one person — it's the liquidity pool, jointly owned by dozens of liquidity providers, and it's *supposed* to hold the token (it's how trading happens). The real concentration risk is elsewhere. Had you instead found an unlabeled EOA holding the same 12,000,000 tokens (\$4.8 million) that received them in a single transfer from the deployer at launch, *that* is a genuine red flag — a team or insider wallet that can sell into your bids. **Same 12%, same \$4.8 million, opposite meaning — the Contract tab is what tells them apart.**

## Address formats across chains, and the checksums that protect you

So far everything has been Ethereum's `0x…` world. But on-chain analysis is multi-chain, and the *shape* of an address is the first clue to which chain you're even on. You don't need to memorize encodings; you need to recognize formats at a glance and understand why they're built the way they are.

![Matrix of address formats across Ethereum Bitcoin Solana and Tron with prefixes and checksums](/imgs/blogs/addresses-wallets-and-contracts-5.png)

Walk through the rows of that matrix:

- **Ethereum / EVM (`0x…`)**: 40 hexadecimal characters after the `0x`, e.g. `0xA11ce…42f`. All EVM chains (Ethereum, Arbitrum, Optimism, Base, Polygon, BNB Chain) share this exact format — *the same address works on all of them*, which is itself a source of confusion and risk (sending to the right address on the wrong chain is a classic way to lose funds). The checksum is **EIP-55**: the *capitalization* of the hex letters encodes a checksum. `0xab…` lowercased is "no checksum"; the mixed-case version `0xAb…` lets a wallet detect a typo. If you mistype one character, the case pattern no longer matches and a good wallet warns you.
- **Bitcoin, modern (`bc1…`)**: native SegWit addresses use **Bech32** encoding — lowercase letters and digits, starting `bc1q` (or `bc1p` for the newer Taproot). Bech32 has a strong built-in checksum that can not only *detect* but help *locate* errors, which is why these addresses are considered the safest to copy by hand.
- **Bitcoin, legacy (`1…` or `3…`)**: older addresses use **Base58Check** — Base58 (which omits look-alike characters like `0`, `O`, `l`, `I`) with a 4-byte hash appended as a checksum. `1…` addresses are the original pay-to-public-key-hash; `3…` are pay-to-script-hash (often multisig). You still see these everywhere.
- **Solana**: a Solana address is a **Base58-encoded 32-byte public key**, typically 32–44 characters, with no fixed prefix, e.g. `7Eq9b…Wm3`. Notably, Solana addresses have **no built-in checksum** — the address *is* the public key, so a mistyped Solana address may still be a structurally valid (just wrong) address. Extra care when copying.
- **Tron (`T…`)**: Tron addresses start with `T` and use Base58Check like legacy Bitcoin. Tron is worth flagging specifically because it has become the dominant rail for USDT (Tether) in retail and, unfortunately, illicit flows — so you'll encounter `T…` addresses constantly when tracing stablecoin movement.

The deep point: **the address encodes both an identifier and, usually, a self-check.** A checksum is an internal consistency test — a few extra bits derived from the rest of the address so that a single typo produces an address that "doesn't add up," and a wallet refuses to send. This is your last line of defense against fat-fingering a transfer into the void. It is *not*, however, defense against sending to a *valid-but-wrong* address — and that gap is exactly what the address-poisoning scam exploits.

#### Worked example: the cost of one mistyped character

You intend to send \$25,000 of USDC to a friend's Ethereum address. You mistype one character. Because Ethereum uses EIP-55 mixed-case checksums, your wallet detects the inconsistency and blocks the send before it broadcasts — your \$25,000 is safe; you fix the typo and resend. Now run the same scenario on Solana, which has no built-in checksum: the same single-character slip produces a *different but structurally valid* address, the wallet sees nothing wrong, and your \$25,000 of USDC is sent to an address whose private key nobody controls — gone forever, frozen on-chain for all time. **The checksum is invisible until the day it saves you \$25,000; on chains that lack one, copy-paste and triple-check.**

## A wallet is rarely one address: multisigs and smart-contract wallets

We have been quietly assuming "one key, one address, one owner." Real custody for anyone serious — exchanges, DAOs, funds, treasuries — looks nothing like that. The most important upgrade is the **multisig**.

### Multisig: M-of-N keys must agree

A **multisig** (multi-signature) wallet requires **M out of N** keys to sign before a transaction executes. A 3-of-5 multisig has five designated signer keys; any three of them must approve a transaction for it to go through. The most widely used implementation on Ethereum is the **Gnosis Safe** (now just "Safe") — and critically, *a Safe is a smart contract, not an EOA.* The Safe contract holds the funds; its code enforces the threshold. When enough signers have approved, the contract executes the transfer.

Why do serious operators use this? Because single-key custody is a single point of failure. One leaked key, one compromised laptop, one rogue employee, and everything is gone. A 3-of-5 requires an attacker to compromise *three* independent keys — held by different people, on different devices, ideally in different places — before they can move a cent. It converts catastrophic single failures into survivable ones.

![A 3-of-5 multisig flow where three of five signers approve before the Safe contract executes](/imgs/blogs/addresses-wallets-and-contracts-4.png)

That figure shows the flow: a signer *proposes* a transaction, others *confirm* it, and once the Safe contract counts at least three valid signatures (the threshold), it executes. Signers 4 and 5 aren't needed for this particular transaction. And the crucial security claim is the red node: **one leaked key is one signature — below the threshold — so it can't move funds.** The thief who steals a single signer's key has accomplished nothing on its own.

#### Worked example: a 3-of-5 Safe holding a \$40M treasury

A DAO keeps its treasury — \$40 million in USDC and Ether — in a 3-of-5 Gnosis Safe. The five keys are held by the founder, the CFO, two independent board members, and a security firm, on five different hardware wallets. An attacker phishes the CFO and steals one key. With single-key custody, the attacker would now control the full \$40 million. With the 3-of-5 Safe, the stolen key is worth exactly one signature; the attacker needs two more, held by people who haven't been compromised, and any attempt to move funds is a *proposed* transaction the other four signers can see and refuse. The \$40 million doesn't move. To drain it, the attacker would have to compromise three of the five independent keyholders at once — orders of magnitude harder. **A multisig turns "one mistake = total loss" into "you'd have to break three independent people at once," which is why every credible treasury uses one.**

### How a multisig looks on-chain (and how to read a Safe)

On Etherscan, a Safe address has a **Contract** tab (it's a contract) and usually a label like `Safe: …` or `Gnosis Safe`. But the richer view is the **Safe app itself** (`app.safe.global`), which shows you the two facts you most want when assessing a treasury:

- **The signers (owners)**: the list of N addresses that can sign.
- **The threshold (M)**: how many of them must approve.

Together, "M of N" tells you how decentralized and how robust the custody is. A treasury in a 1-of-1 Safe is barely better than an EOA — one key still moves everything. A 2-of-3 is decent for a small team. A large protocol treasury sitting in a 4-of-7 with reputable, independent signers is genuinely hard to drain. When you're doing due diligence on a project, *reading the Safe configuration is reading the project's risk of an inside job or a single-key hack.* A "\$200 million treasury" in a 1-of-2 multisig where both keys are on the founder's two laptops is a different risk than the same \$200 million in a 5-of-9.

On-chain, you can also watch a Safe transaction assemble: the proposal transaction, then a series of confirmation transactions from different signer addresses, then the execution. Investigators use this to see *who* approved a controversial transfer and *when* — the multisig's transparency cuts both ways.

A second, easy-to-miss detail about multisigs is that the *signer keys themselves are usually EOAs* (or sometimes other contracts, including hardware-wallet-backed accounts). So a Safe's security is only as good as the independence and protection of those underlying keys. When you read a 4-of-7, the natural follow-up is: are those seven signer addresses genuinely separate parties, or were five of them funded from the same source, active only minutes apart, and behaving like one person's seven laptops? On-chain you can often answer this. If all the "independent" signers were funded from one address and have no other activity, the multisig is theater — a 4-of-7 that an attacker can satisfy by compromising the single human behind all seven. This is precisely the kind of due diligence that separates a real on-chain analyst from someone reading a dashboard's "multisig ✓" badge. The label says "decentralized custody"; the funding graph of the signers tells you whether that's true.

This also reframes what some of crypto's biggest thefts really were. Several major exchange and bridge hacks — including the Bybit and Ronin events — were not breaks of the blockchain or even of the multisig logic; they were compromises of *enough signer keys* (or of the signing interface the signers trusted) to clear the threshold. Reading "this treasury is a 5-of-9" is necessary but not sufficient; you also have to ask how those nine keys are stored, who holds them, and whether the signing process itself can be tricked into producing valid signatures for a malicious transaction. The chain will always honor a transaction that meets the threshold with valid signatures — even when those signatures were obtained by deceiving the humans who hold the keys.

### Smart-contract wallets and the ERC-4337 smart-account standard

A multisig is one example of a broader idea: a **smart-contract wallet**, where the "account" is a contract with custom logic rather than a bare key. The general framework on Ethereum is the **smart-account standard ERC-4337** (the industry term for it is "account-level programmability" — the account's rules live in code rather than in a single raw key). The motivation is that plain EOAs are rigid: one key, no recovery, no spending limits, you must hold ETH to pay gas, and a single key leak is fatal. A smart-contract wallet can encode richer rules:

- **Social recovery**: if you lose your key, a set of "guardians" you chose earlier can together help you recover access — no seed phrase to lose.
- **Spending limits and session keys**: allow small daily spends with a hot key but require extra approval for large ones; grant a game or app a limited key that expires.
- **Gas sponsorship (paymasters)**: someone else (an app, the protocol) can pay the gas fee, so a user can transact without first owning ETH — a big onboarding win.
- **Batched transactions**: approve and swap in one atomic action instead of two.

ERC-4337 implements all this *without* changing Ethereum's core protocol, by introducing a parallel system: users submit "UserOperations" to a separate mempool, "bundlers" package them, and a global `EntryPoint` contract orchestrates execution. You don't need the plumbing to do analysis; you need to recognize that **an ERC-4337 smart-account wallet is a contract, and its behavior is defined by code, not just a key.** On an explorer it shows a Contract tab; its transactions may originate through the bundler/EntryPoint machinery rather than as a simple signed EOA transfer, which can look unusual if you expect plain transfers. As these wallets proliferate, "this address is a contract" stops implying "this is a protocol, not a user" — increasingly, real end-users *are* contracts.

> [!note]
> **EIP-7702 (Pectra, 2025).** A newer upgrade lets an ordinary EOA *temporarily* take on smart-contract code for the duration of a transaction — blurring the EOA/contract line further. The takeaway for analysts: the binary "key-controlled vs code-controlled" is becoming a spectrum, and you should rely on *behavior* (what the address does, what it's bound to) as much as the static "has a Contract tab" check.

## One entity, many addresses: the preview of clustering

Now we arrive at the idea that powers the rest of this series. The naive mental model — *one address = one person* — is wrong in both directions:

- **One person controls many addresses.** A single user might have a MetaMask account, a hardware wallet, a Safe for savings, and several throwaway addresses. A trader might split holdings across dozens of addresses deliberately, to avoid being tracked.
- **One organization controls a *huge* number of addresses.** This is most extreme for exchanges, and it's the single most important structural fact about on-chain flow analysis.

### Why an exchange has so many addresses

A custodial exchange like Binance or Coinbase manages funds for millions of users. It does not give each user a self-controlled wallet (that would defeat custody). Instead it operates a layered address structure:

- **Cold wallets**: the bulk of reserves (often 85–95%), held in offline, heavily-secured, frequently multisig addresses that rarely move. This is the vault.
- **Hot wallets**: a smaller working balance in online addresses, used to fund customer withdrawals on demand. These move constantly.
- **Deposit addresses**: here's the key one. To credit deposits correctly, exchanges typically generate **a unique deposit address for each user** (or each user-asset pair). When you "deposit BTC to Binance," Binance shows *you* a dedicated address; when funds arrive there, their system knows it's *your* deposit and credits your account.

![Graph of one exchange entity controlling cold hot and many per-user deposit addresses](/imgs/blogs/addresses-wallets-and-contracts-3.png)

The figure shows the structure: one entity, fanning out into a cold wallet, a hot wallet, and a vast pool of per-user deposit addresses. And it marks the analytically golden link in red — the **sweep**. Deposit addresses don't keep funds; periodically the exchange *sweeps* the balances from thousands of deposit addresses into its hot or cold wallet to consolidate. That sweep transaction is a giant tell: it visibly connects thousands of deposit addresses to one central address, *proving* they're all controlled by the same entity. This is the bedrock heuristic of clustering — the "common-input" or "common-destination" pattern — which a later post unpacks in full.

Note the crucial property of a deposit address that makes it so useful in tracing: **it is a one-way funnel tied to one user, even though the exchange controls its key.** When investigators trace stolen funds that get "deposited to an exchange," what they often find first is a deposit address — and because that deposit address maps one-to-one to a specific KYC'd customer in the exchange's records, identifying it is the moment an anonymous on-chain trail becomes a name a court can subpoena. This is exactly how a large share of hacked funds eventually get attributed: the thief, needing to cash out, sends to an exchange deposit address, and that single hop bridges the pseudonymous chain to a real, identified account. The deposit-address structure that exists for the exchange's *accounting* convenience is the same structure that makes exchanges the chokepoint where investigations succeed.

It also clarifies what an exchange's reserves really are. Because user funds are pooled (this is **omnibus custody** — many users' assets commingled in shared addresses, rather than segregated one-address-per-user), the on-chain balance of an exchange's cold and hot wallets is the relevant number for solvency, not anything a single user can see. When analysts talk about "proof of reserves," they mean: can we attribute a set of addresses to the exchange, sum their balances, and check that the total covers what the exchange owes its users? That entire exercise depends on first correctly clustering the exchange's addresses — which is, again, this post's core skill applied to a multi-billion-dollar question. And reading the *direction* of the flows across those wallets is the basis of exchange-flow analysis: net coins moving *onto* exchanges historically signals selling pressure (users preparing to sell), while net coins moving *off* exchanges into self-custody signals accumulation. You cannot read that signal at all until you know which addresses belong to the exchange.

#### Worked example: 1.2 million deposit addresses and what the sweep reveals

An exchange with 1,200,000 users generates roughly one deposit address per user — call it 1.2 million addresses (often more, since many users get a separate address per supported asset). Each might hold a small, variable balance — say an average of \$300 in pending deposits, so 1,200,000 × \$300 = \$360 million scattered across the deposit layer at any moment. To a naive observer, that's 1.2 million separate "wallets" holding \$360 million. But every few hours the exchange sweeps them: tens of thousands of these addresses send their balances into a single consolidation address in batched transactions. The instant an analyst sees thousands of addresses all feeding one destination, the cluster collapses — those 1.2 million "wallets" are revealed as **one entity**. **The exchange's privacy was never in having many addresses; it dissolves the moment the addresses transact together, which is the core of address clustering.**

### Counting an entity's true balance

A direct consequence: to know what an entity actually holds, you must sum across *all* the addresses you can attribute to it — not read one address and call it a day. This is why headline claims like "this whale holds \$X" are so often wrong: they read one address and miss the other seven.

#### Worked example: an entity's real balance across 8 addresses

You're tracking a fund you suspect is one entity. Address-by-address, the explorer shows: a main Safe with \$1,200,000; a hot EOA with \$400,000; a trading address with \$250,000; a staking deposit with \$180,000; an LP position worth \$120,000; two cold addresses with \$90,000 and \$40,000; and a fresh address that just received \$20,000. Read in isolation, none looks like more than \$1.2 million, and a careless analyst reports the entity as "a \$1.2M player." Sum the cluster — \$1,200,000 + \$400,000 + \$250,000 + \$180,000 + \$120,000 + \$90,000 + \$40,000 + \$20,000 = **\$2,300,000** — and the true picture is nearly double. **An entity's balance is the sum over its cluster; quoting a single address systematically understates whales and overstates how fragmented the market is.**

## A defender's note: vanity addresses and address-poisoning scams

Two related phenomena round out the picture of what addresses really are — one mostly benign, one a live threat you must understand to protect yourself and to recognize in an investigation.

### Vanity addresses

Because an address is derived from a key, you can keep generating random keys until the resulting address *happens* to start (or end) with characters you like — a **vanity address** like `0xDEAD…` or one ending in your initials. This is just brute-force search: you grind through keys until one produces the pattern, and the more characters you want to fix, the exponentially more attempts it takes. Vanity addresses are legitimately used for branding (a protocol's address starting with a recognizable string) and convenience. The key point for analysis: a "pretty" address is not inherently trustworthy — anyone can grind one. And the same grinding technique enables an attack.

### Address poisoning

**Address poisoning** weaponizes a habit nearly everyone has: when sending crypto, you glance at the *first and last few characters* of an address and assume the middle matches. Attackers exploit this directly.

![Address poisoning scam where an attacker plants a lookalike address to bait a copy-paste](/imgs/blogs/addresses-wallets-and-contracts-6.png)

The attack, as a defender needs to recognize it, runs as in the figure:

1. The attacker watches your transaction history and sees an address you regularly send to — say `0xA11ce…f29d`.
2. They grind a **vanity lookalike** that matches the *visible* parts — same first few and last few characters (`0xA11ce…f29d`) — even though the hidden middle is entirely different. The full strings are not equal; only the bits a human eyeballs match.
3. They send a tiny "dust" transaction (often a worthless token, or a \$0-value transfer) *from* the lookalike *to* you. This **plants the lookalike in your transaction history**, sitting right next to the real address you use.
4. Next time you go to pay your real counterparty, you copy the address from your history — and you grab the *poisoned* row, because it looks identical at a glance and may even sort to the top as the "most recent" match.
5. Funds go to the attacker. The transaction is valid and irreversible. There is no recovery.

The defense is the success-colored node: **verify the full address, never just the first and last four characters.** Use an address book / saved contacts in your wallet, scan a QR code from the source rather than copying from history, and confirm the *entire* string (or at least many more characters than the visible ends) before signing a large transfer. Hardware wallets that display the full address on a separate screen exist precisely for this.

#### Worked example: a \$50,000 address-poisoning loss

A treasury manager regularly pays a supplier at `0xA11ce…f29d`. An attacker grinds a lookalike `0xA11ce…f29d` (identical visible ends, different middle) and sends a \$0 dust transfer to the treasury so the lookalike appears in its history. The next month, the manager opens the wallet, copies "the address we always use" from recent activity, and sends the monthly \$50,000 payment — to the attacker's lookalike. The supplier never receives the \$50,000; the funds are irreversibly gone; the only forensic trace is that the destination differs from the real address in the middle characters everyone skips. **Address poisoning costs nothing to set up and routinely steals five- and six-figure sums precisely because it attacks human eyeballs, not cryptography — defend by verifying full addresses, not the ends.**

## How to read it: a walkthrough on Etherscan

Let's make all of this concrete with the tool you'll use most. Here is a step-by-step pass through an address page on Etherscan, the canonical EVM explorer, classifying what you're looking at and extracting the facts that matter. (Block explorers for other chains — Blockchair for Bitcoin, Solscan for Solana, Tronscan for Tron — follow the same logic with different layouts.)

**Step 1 — Paste the address and read the top of the page.** The header shows the address, its current ETH balance, and its total value in tokens. Immediately scan for a **label chip** (e.g. `Binance 14`, `Uniswap V3: Router`) and for the **Contract** tab.

**Step 2 — Classify it: EOA or contract?**
- No Contract tab → EOA. Treat it as key-controlled — a user, a trader, an exchange hot wallet, a deployer. Your next questions are about *behavior*: who funded it, what does it do, where does its money go.
- Contract tab present → contract. Click it. If the source is verified (green check), read what it is (a token, a router, a Safe, a vault). If only bytecode is shown (unverified), be cautious — unverified contracts holding user funds are a classic rug/honeypot warning sign you'll learn to flag in a later post.

**Step 3 — Read the activity timeline.** Under Transactions, look at:
- **First activity** ("First seen" / the earliest transaction): a freshly created address that suddenly holds millions is suspicious; an address active for years reads differently.
- **Last activity**: dormant vs active. A long-dormant whale waking up is itself a signal.
- **Who funded it first**: the very first inbound transaction often reveals the address's origin — funded *from* a known exchange (so a real person KYC'd to that exchange likely controls it), or *from* another address in a cluster.

**Step 4 — Read the token holdings.** The token dropdown lists every ERC-20 and NFT the address holds, with USD values. This is how you assess "what is this entity actually holding," for a single address — remembering Worked Example above, that the *entity's* balance may be spread across many such addresses.

**Step 5 — For a contract, read its role.** If it's a token contract, the page shows total supply, holders, and transfers. If it's a Safe, jump to `app.safe.global` and read the **owners (signers)** and **threshold (M-of-N)** — the custody risk in two numbers. If it's a DeFi contract, the verified source and the "Read/Write Contract" tabs let you see its functions and current state.

**Step 6 — Distrust the label, verify the behavior.** Etherscan labels are helpful but crowd-sourced and occasionally wrong or stale. An address labeled "Binance" that's behaving nothing like an exchange (or a label absent on an address that clearly is one) should send you to corroborate with on-chain behavior — funding source, counterparties, transaction patterns — and with a second tool (Arkham, Nansen, or a Dune query). **The label is a hypothesis; the behavior is the evidence.**

Run those six steps and you've extracted, from a single explorer page, the facts that every downstream technique in this series consumes: is this code or a person, how old and how active, funded from where, holding what, and — if a contract — governed by what rules and signers.

> [!tip]
> **One-glance triage.** Contract tab? → it's code (read its role and signers). No tab? → it's a key-controlled account (read its funding source and counterparties). Then *always* ask: is this one address part of a bigger entity? The single page is never the whole story.

## Common misconceptions

**"An address is a person."** No — an address is a public string controlled by a key (EOA) or by code (contract). One person can hold dozens of addresses, and one *entity* (an exchange) can control millions. Reading "address X did Y" as "person X did Y" is the most common rookie error, and it's how routing contracts get mislabeled as whales. The correct unit of analysis is the **entity** (a cluster of addresses), recovered by clustering — not the lone address.

**"A wallet is one address."** No — a wallet app (MetaMask, Phantom) manages *many* accounts, each with its own address; a "wallet" in the custody sense (a Safe) is a contract that may control funds for a whole organization. When someone shares "their wallet address," that's one of potentially many. Summing an entity's balance means summing across its addresses — a single address routinely shows a fraction of the truth (recall the \$1.2M-looking entity that actually held \$2.3M).

**"A smart contract holds someone's money like a bank account."** No — a contract holds funds under *code*, with no private key and no single owner. \$50 million in a Uniswap pool is shared liquidity governed by the AMM's math; \$40 million in a Safe moves only when M-of-N signers approve. There's no human whose "balance" that is in the EOA sense. Treating a contract's balance as a person's net worth misreads the chain.

**"If the first and last characters match, it's the right address."** Dangerously no — address poisoning is built entirely on this assumption. Attackers grind lookalikes with identical visible ends and a different middle, plant them in your history with dust transfers, and wait for you to copy the wrong one. A \$50,000 transfer to the right-looking-but-wrong address is irreversible. Verify the *full* string, every time, for anything that matters.

**"A contract address means it's a protocol, not a real user."** Increasingly no — with the ERC-4337 smart-account standard and EIP-7702, ordinary end-users are now using smart-contract wallets. "Has a Contract tab" no longer cleanly separates "protocol/infrastructure" from "person." Classify by behavior — what the address does and who it's bound to — not just by the static presence of code.

## The playbook: what to do with it

This is the foundation post, so the playbook is less "place a trade" and more "never misread the board again." Use it as a checklist whenever you open an address.

**Signal → Read → Action.**

1. **You see an address you want to understand.**
   - *Read*: Contract tab present? → contract (read role, verified source, and if a Safe, the M-of-N signers). Absent? → EOA (read funding source, first/last activity, counterparties).
   - *Action*: classify it correctly *before* you attribute any behavior or value to it. Code is not a person; a person is not their one address.
   - *Invalidation / false positive*: with ERC-4337 smart accounts, a contract may be an end-user; an Etherscan label may be wrong or stale. Corroborate with behavior and a second tool before you trust it.

2. **You want an entity's true balance or position.**
   - *Read*: don't read one address — find the cluster (sweeps, common funding sources, repeated counterparties), then *sum* across it.
   - *Action*: quote the entity's aggregate, not the single address. Recall the \$2.3M entity that looked like \$1.2M from its main wallet alone.
   - *Invalidation*: over-clustering is a real risk — don't merge addresses on a single weak heuristic; one shared counterparty isn't proof of common control.

3. **You're assessing a project's or treasury's custody risk.**
   - *Read*: find the treasury address, confirm it's a Safe, read the threshold and signers (M-of-N) on `app.safe.global`.
   - *Action*: a \$40M treasury in a 3-of-5 with independent signers is robust; the same \$40M in a 1-of-2 on one person's two laptops is a single-key risk dressed up as a multisig. Size your trust accordingly.
   - *Invalidation*: signers can be sham (all controlled by one person), and a high threshold with colluding signers is theater. Check whether the signer addresses are themselves independent.

4. **You're about to send funds (or you're reviewing someone who did).**
   - *Read*: verify the **full** destination address — not the first/last four. Check it against a saved contact or a freshly scanned QR, never a copy from transaction history.
   - *Action*: for large transfers, confirm on a hardware-wallet screen and on the correct chain (the same `0x…` is valid on every EVM chain — right address, wrong chain still loses funds).
   - *Invalidation / false positive*: a lookalike in your history with matching ends is the address-poisoning trap — its middle differs. A single dust transfer from an unfamiliar lookalike is the tell.

The throughline of the entire playbook: **identify the object before you interpret it.** Key-controlled or code-controlled? One address or an entity's cluster? Real custody or theater? Get the object right, and every later technique — flow analysis, attribution, rug-checking, hack tracing — has something true to stand on. Get it wrong, and you're building analysis on a misread.

## Further reading & cross-links

- [How blockchains store data: UTXO vs account model](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account) — *why* a Bitcoin "address" behaves differently from an Ethereum account, and what an account's balance really represents at the protocol level.
- [Anatomy of a transaction](/blog/trading/onchain/anatomy-of-a-transaction) — the signed message that an EOA broadcasts and a contract executes; the `from`/`to`/internal-transfer distinction that classifies addresses.
- [Address clustering and heuristics](/blog/trading/onchain/address-clustering-and-heuristics) — the full treatment of how the many-addresses-one-entity problem is solved (common-input, sweeps, behavioral fingerprints).
- [Labeling and attribution](/blog/trading/onchain/labeling-and-attribution) — how an address gets a name, why Etherscan labels are a hypothesis not a fact, and how to corroborate one.
- [Centralized crypto exchanges: Binance, Coinbase, and the on-chain footprint](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) — the custodial model that produces hot/cold/deposit address structures, and what their flows mean.
- [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) — background on the EVM, accounts, and why contracts are first-class on Ethereum.
- [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision) — the origin of pseudonymous, key-controlled money that everything here builds on.
