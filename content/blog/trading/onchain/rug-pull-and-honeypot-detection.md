---
title: "Rug Pull and Honeypot Detection: A Pre-Buy Safety Checklist"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Before you buy any low-cap token, six repeatable on-chain checks catch the most common ways you get robbed — a liquidity rug, a mint-and-dump, or a honeypot you can buy but never sell."
tags: ["onchain", "crypto", "rug-pull", "honeypot", "token-due-diligence", "defi", "ethereum", "solana", "token-sniffer", "etherscan", "goplus", "safety"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Before you buy any low-cap token, a short on-chain checklist catches the two ways you get robbed: a *rug pull* (the team pulls the liquidity or mints and dumps) and a *honeypot* (a contract that lets you buy but reverts your sell).
>
> - **The signal:** the token contract is code with owner powers, and the liquidity pool is a balance someone can withdraw. Both are public — you can read them before you spend a dollar.
> - **How to read it:** run six checks with free tools — is the LP locked or burned? is mint authority renounced? what owner functions exist? does a simulated sell go through? how concentrated are the holders? is the contract verified? Token Sniffer, Honeypot.is, GoPlus, DEXScreener's audit tab, and Etherscan's read/write tabs do most of it.
> - **What you do with it:** any single hard red flag is a *stop*, not a thing you average against a pretty chart. All clear means you may size small — never "this is safe."
> - **The one rule:** a token that passes every safety check can still dump 90% on its own tokenomics. **Safe-to-trade is not the same as good-to-own.**

On 2024-05-31, a single attacker drained roughly **\$305 million** out of DMM Bitcoin in a key-and-infrastructure compromise later attributed to the Lazarus Group. That is the kind of headline that makes the news. But it is not how most retail crypto users actually lose money. The quiet, grinding losses happen one \$200, \$2,000, or \$20,000 buy at a time, in tokens nobody will ever write a post-mortem about — tokens that were *designed* to take your money the moment you sent it.

Two designs do almost all of that damage. The first is the **rug pull**: a team launches a token, lets a crowd pile in, then withdraws the liquidity (or prints new supply and dumps it), leaving holders with a coin that has nothing to sell into. The second is the **honeypot**: a contract that happily accepts your buy — your transaction confirms, your wallet shows a balance, the chart even ticks up — but reverts every sell *except* the owner's. You can get in. You can never get out.

Here is the part that should change how you behave: both of these are *visible on-chain before you buy*. The contract's owner powers are public bytecode you can read on Etherscan. The liquidity pool's ownership is a token balance you can look up in two clicks. A simulated sell can be run against the live contract for free. This post turns those facts into a concrete, repeatable **pre-buy checklist** — the six checks below — and shows you exactly which tool answers each one.

![Pre-buy safety checklist with six on-chain checks feeding a buy or avoid verdict](/imgs/blogs/rug-pull-and-honeypot-detection-1.png)

This is a defender's guide. We cover how these traps are built only to the depth you need to *recognize and avoid* them — not as a manual for building one. The goal is simple and entirely self-interested: stop sending money to contracts that were written to keep it.

One reframe worth internalizing before the mechanics: in traditional finance, a lot of safety is provided *for* you — an exchange vets listings, a regulator polices fraud, a custodian holds the asset. In permissionless crypto, almost none of that exists at the low-cap level, and the responsibility shifts entirely onto the buyer at the moment of purchase. That sounds like a bug, and for the careless it is. But it's also the source of the edge: because the rules are public code and the money flows are a public ledger, a buyer who *does* look has more information about a token's safety than they'd ever have about a private company's. The trap is fully visible in advance. The asymmetry only hurts the people who don't read it. This post is about being on the right side of that asymmetry.

## Foundations: tokens, liquidity, and owner power from zero

Before the checklist makes sense, you need four building blocks. None of them require code; all of them are things you can verify yourself.

### What a token actually is

On Ethereum and EVM chains (Base, BNB Chain, Arbitrum, Polygon, and the rest), a "token" is not a coin sitting in a vault. It is a **smart contract** — a program deployed to an address — that keeps an internal ledger of who owns how much. When you "buy 1,000,000 TOKEN," what happens is that the contract updates a number: `balanceOf[yourAddress] = 1000000`. When you sell or send, the contract decrements your number and increments someone else's. We covered the mechanics in [tokens, on-chain transfers and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals); the one fact that matters here is that **the rules for those updates are written by whoever deployed the contract.** Most tokens follow the standard ERC-20 template, but the deployer can add whatever extra logic they want — including logic that decides whether *you specifically* are allowed to sell.

An **EOA** (externally owned account) is a normal wallet controlled by a private key — a person. A **contract account** is code. The "owner" of a token contract is usually an EOA the deployer controls, and many contracts grant that owner special abilities the rest of us don't have. Those abilities are the whole game.

### What liquidity and an LP token are

You don't buy a brand-new token from the team directly. You buy it from a **liquidity pool** on a decentralized exchange (DEX) like Uniswap or PancakeSwap. A pool for, say, TOKEN/ETH holds a balance of both assets — some TOKEN and some ETH — and a formula sets the price from their ratio. When you add ETH and take TOKEN out, the ratio shifts and the price rises; when you sell TOKEN back, you take ETH out and the price falls. (The full mechanism — constant-product pricing, slippage, impermanent loss — lives in our piece on [DeFi protocols Uniswap, Aave, and MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).)

Here is the load-bearing detail. Whoever deposits both sides of the pool receives an **LP token** (liquidity-provider token) — a receipt that represents their claim on the pooled assets. Whoever holds that LP token can **withdraw the pool** by burning the receipt and reclaiming the underlying ETH and TOKEN. So the question "who holds the LP token?" is identical to the question "who can pull the money out of the pool?" That single question is the first and most important check on the list.

### What "renouncing ownership" means

A token contract can have its owner address set to the **zero address** (`0x000…000`) — a do-nothing address whose private key nobody holds. Calling the function that does this is called **renouncing ownership**. After it's renounced, the privileged owner functions can never be called again, because no one controls the owner address. Renouncing is a one-way, permanent, on-chain action you can verify. A token whose owner is still a live EOA has live owner powers; a token that has renounced does not.

### What "locking" or "burning" liquidity means

Two ways to take the rug button away from the team, both verifiable:

- **Locking the LP** means sending the LP token to a *time-lock contract* (Unicrypt/UNCX, Team Finance, PinkLock, and similar) that holds it and refuses to release it until a date. Until that date, nobody — not even the team — can withdraw the pool.
- **Burning the LP** means sending the LP token to a burn address (`0x000…dead`), permanently. Nobody holds the receipt, so the pool can never be withdrawn by anyone. This is stronger than a lock because there's no unlock date to worry about.

So the four foundations are: a token is owner-written code; the LP token is the withdraw key to the pool; renouncing kills owner powers; locking/burning kills the rug button. Every check below is just *reading these facts off the chain.*

### Why these traps are even possible

It's worth pausing on the deeper reason rugs and honeypots exist at all, because it tells you why no exchange or regulator catches them for you. Deploying a token on a public chain is **permissionless**: anyone with a wallet and a few dollars of gas can deploy a contract — there is no listing committee, no KYC, no code review, no minimum standard. A DEX like Uniswap is equally permissionless: anyone can create a TOKEN/ETH pool by depositing both sides, and that pool is instantly tradable by the whole world. Nobody approved it. Nobody checked it.

That openness is the same property that makes the system useful — a legitimate builder can ship without asking permission — and the property that lets a scammer ship a honeypot in the same five minutes. The chain doesn't distinguish intent. It just executes the code as written. So the *only* gatekeeper in the entire pipeline is **you, the buyer, reading the contract before you send money.** There is no one else in the loop. That single sentence is the reason this checklist is worth committing to muscle memory: on a permissionless chain, due diligence isn't outsourced to anyone — it's the price of admission, and skipping it is the most common way people lose money in crypto.

This also means the contract's *behavior is deterministic and public.* A bank can quietly change your account terms in a back office; a token contract cannot do anything that isn't written in its bytecode and visible on-chain. That's the asymmetry the checklist exploits: every power the team has over your money has to be encoded as a function you can read in advance. The trap is visible *before* it springs — if you look.

### A note on other chains: Solana and the rest

The mechanics above are described in EVM (Ethereum/Base/BNB/Arbitrum) terms because that's where the richest tooling lives, but the same questions apply everywhere — only the vocabulary changes. On **Solana**, where most of the memecoin volume now sits, a token (an SPL token) has a **mint authority** and a **freeze authority** instead of an `owner()`. A live mint authority is the same risk as a live EVM `mint()` — the team can print more supply; a renounced (null) mint authority is the equivalent of a renounced owner. A live **freeze authority** is even more direct than a blacklist: the team can freeze *your* token account so you can't move it at all. Solana tools like **RugCheck** and **Solscan** report whether mint and freeze authority are still set, and whether the LP is burned, exactly as Token Sniffer and Etherscan do on EVM. The six checks below translate one-to-one: liquidity, mint, owner/authorities, sell test, holders, verification. Learn the EVM version and you can read any chain.

## What a rug pull is: three flavors

"Rug pull" is a loose term for "the team took your money." There are three distinct mechanisms, and the checklist defends against each differently.

### Flavor 1 — the liquidity rug

The cleanest and most common. The team provides the initial liquidity (so the token has a price and is buyable), lets buyers add ETH to the pool, then — because they still hold the LP token — **withdraws the entire pool**, taking the ETH that buyers put in and leaving the pool with almost nothing on the ETH side. Your tokens still exist in your wallet, but there's nothing to sell them into, so the price collapses toward zero in one block.

![Liquidity rug before and after the team withdraws the pooled ETH](/imgs/blogs/rug-pull-and-honeypot-detection-2.png)

#### Worked example: the \$500k liquidity rug

A token launches with a pool holding 500,000 TOKEN and **\$500,000 of ETH**. The price per token is set by the ratio — roughly \$1.00 each at launch. Over a day, FOMO buyers add another **\$500,000 of ETH** to the pool chasing the chart, so the pool now holds about \$1,000,000 of ETH against the remaining tokens, and the price has roughly doubled. The team has held the LP token the whole time — it was never locked.

In one transaction, the team burns the LP token and **withdraws the full ~\$1,000,000 of ETH**. The pool is now essentially all TOKEN and ~\$0 of ETH. A holder trying to sell 10,000 tokens that showed as "worth \$20,000" a minute ago now finds the pool will give them a few dollars of dust, because there's no ETH left on the other side of the ratio. The team walked away with the **\$1,000,000** that buyers deposited; the holders are left with a token that has no liquidity. The lesson: the price you see is only real if the ETH backing it can't be withdrawn.

### Flavor 2 — the mint-and-dump

Here the team doesn't touch the pool. Instead, the contract retains a **mint function** the owner can call to print new tokens into their own wallet, with no cap. The owner mints a huge new tranche — sometimes 10× the circulating supply — and sells it into the pool's bids. Your tokens aren't withdrawn; they're *diluted* into worthlessness while the pool's ETH flows to the minter.

#### Worked example: an \$80k mint-and-dump

A token has a circulating supply of 1,000,000 tokens trading at **\$0.10**, so a market cap near \$100,000, against a pool holding **\$80,000 of ETH**. The owner calls `mint()` and prints **9,000,000 new tokens** to their own wallet — instantly 10× the supply — then sells them into the pool. As they dump, they drain the pool's ETH: the first chunks sell near \$0.10, later chunks at \$0.03, \$0.01, then dust. By the time the pool is empty, the owner has extracted roughly the **\$80,000 of ETH** that was in it, and the existing holders' 1,000,000 tokens are now 10% of a 10,000,000 supply chasing an empty pool. The takeaway: an uncapped mint function is a withdraw button on the pool wearing a different hat.

### Flavor 3 — the slow rug

The patient version. No single dramatic transaction; instead the team bleeds the project over weeks — selling their allocation steadily into every bounce, abandoning development, quietly moving treasury funds — until the token is dead. There's no one block to point at, which is exactly why it's harder to catch and why the holder-concentration check (who holds what, and is the team's wallet still loaded?) matters even when liquidity looks locked.

The slow rug is the hardest to defend against with a single snapshot, because nothing about the *contract* is wrong — it can be renounced, the LP can be burned, the sell-sim passes. The entire risk lives in the **holder distribution** and the team's *intent*, which the chain reveals only over time. The defensive read is a one-time proxy for that intent: how much supply does the deployer (and wallets it funded) still control, and is any of it unlocked? A deployer sitting on 30–50% of supply with no vesting is a fuse, regardless of how clean the contract is. The slow rug is the clearest example of why Check 5 (holders) is independent of Checks 1–4: you can pass every contract check and still be the exit liquidity for a loaded insider.

This is also where on-chain *monitoring* beats a one-time check. The same Etherscan Holders tab read days apart shows whether the deployer wallet is shrinking — a deployer steadily moving tokens to exchanges is selling, and that's the slow rug in motion. For a position you hold, re-running the holder check weekly is cheap insurance; the trap that takes weeks to spring can be seen weeks in advance if you look.

## What a honeypot is: buy works, sell reverts

A honeypot is a different category of trap. The liquidity might be genuinely locked; the team might never pull anything. The trick is that **the contract's transfer logic treats buys and sells differently.** A buy (ETH in, token out) is allowed and confirms normally — that's the bait, because every buy with no sells produces a clean, pumping chart that lures more buyers. But when you try to sell, the contract **reverts the transaction** (transfer logic throws an error for non-whitelisted addresses) or applies a **100% sell tax** so the entire proceeds are confiscated. Either way you cannot exit, while the owner — who *is* whitelisted — sells freely and takes the ETH every buyer deposited.

The reason this works is the same permissionless property from the foundations: the ERC-20 `transfer` function is just code, and the deployer can put a condition inside it — `if (from != owner && isSell) revert;` is a one-line difference from an honest token. Nothing about the buy experience hints at it. Your wallet shows the balance, the block explorer shows your tokens, the chart shows green. The asymmetry only surfaces the instant you try to leave, which is precisely the moment it's too late to act on. That's why a *pre-buy* simulated sell — testing the exit before you take the entry — is the only check that catches it in time. By the time you discover the revert by hitting sell with real money, the trap has already closed.

![Honeypot contract where buys succeed but sells revert and only the owner exits](/imgs/blogs/rug-pull-and-honeypot-detection-3.png)

#### Worked example: the \$2,000 honeypot you can never exit

You see a new token up 60% in an hour, a chart that only goes up, a few hundred holders. You buy **\$2,000 of ETH** worth; the transaction confirms, your wallet shows 4,000,000 TOKEN, the DEX shows your position "worth \$2,180." An hour later you decide to take profit. You hit sell — the wallet estimates gas, you confirm — and the transaction **reverts**. You try a smaller size. Reverts. You raise the slippage to 49%. Reverts. The reason is in the contract: the transfer function reverts for any address that isn't on the owner's whitelist. Your **\$2,000** is not "down"; it is *inaccessible at any price.* Meanwhile the owner, whitelisted, sells their own bag into the bids that you and others created. The chart that looked like demand was a one-way trap: every buy fed the owner's exit, and no buy could ever come back out.

The reason "the chart only goes up" should *raise* your suspicion, not lower it: a token where nobody can sell will, by construction, never print a red candle until the owner pulls the plug. A suspiciously clean uptrend in a brand-new low-cap is a honeypot tell, not a momentum signal.

## The pattern in the wild: documented rugs as forensic cases

These aren't theoretical. A handful of well-documented episodes show every flavor above playing out at scale, and reading them as forensic cases — what the chain showed *before* and *after* — sharpens the checklist.

The most cited honeypot is the **Squid Game token (SQUID)** of late 2021. Riding the Netflix show's name, it rocketed from cents to a reported ~\$2,800 per token in days, drawing a flood of buyers chasing the chart. The catch was in the transfer logic: ordinary buyers could not sell. The contract restricted selling, so the "price" was a one-way ramp built entirely from buys with no offsetting sells — the textbook honeypot signature. When the developers cashed out the liquidity, the token went to effectively zero and the team disappeared with an estimated **\$3.3 million** of buyer money. Every red flag was on-chain: an unverified-at-first contract, restricted transfers a sell-simulation would have caught instantly, and liquidity the team controlled. The chart looked like the trade of the year; the contract said "you can check in but you can't leave."

The classic liquidity rug is **AnubisDAO (2021)**, where roughly **\$60 million** of ETH raised in a liquidity bootstrapping event vanished within about a day — the funds were moved out of the project's control almost immediately after the raise, before holders had any real chance to react. And the slow rug has a thousand unnamed examples: a team launches with a locked LP (so it passes the first check), then bleeds its own large allocation into every bounce over weeks while quietly abandoning the project — no single dramatic block, just a holder-concentration fuse burning down. The checklist's value is that the SQUID honeypot dies at Check 4 (the simulated sell reverts), the AnubisDAO-style rug dies at Check 1 (the raised funds were never locked), and the slow rug shows up at Check 5 (a loaded deployer wallet). Different mechanisms, different checks, same five-minute pass.

The broader number frames the stakes: Chainalysis attributes roughly **\$1.7 billion** stolen from crypto platforms in 2023 and about **\$2.2 billion** in 2024 — and that's only the *hacks* big enough to be tracked. The retail rug-and-honeypot losses, spread across millions of tiny tokens, are largely uncounted on top of that. None of it reaches you if you don't buy the trap.

## The dangerous owner functions, and how to read them

The honeypot and the mint-and-dump both come from the same source: **privileged owner functions** baked into the contract. These are not bugs or hacks. They are legal, ordinary Solidity that the deployer chose to include and that the owner can call at will. A token that keeps them is one transaction away from a trap, even if it looks clean today.

![Matrix of dangerous owner functions mint pause blacklist set-fee and how each one robs holders](/imgs/blogs/rug-pull-and-honeypot-detection-4.png)

The four to fear, and what each lets the owner do:

- **`mint()`** — print new tokens to any address, usually their own, with no supply cap. This is the mint-and-dump button. A capped or renounced mint is fine; an uncapped, still-callable mint is a live dilution risk.
- **`pause()` / `setTradingEnabled(false)`** — freeze all transfers across the whole token. The owner can pause selling right before their own exit, so only insiders get out at the top while everyone else is frozen.
- **`blacklist()` / `setBlacklist()`** — flag specific addresses as unable to send or receive the token. The owner can blacklist *you* the instant you try to sell, leaving you holding a permanently dead bag while others still trade.
- **`setFee()` / `setTaxes()`** — change the buy/sell tax after launch, often up to 100%. Set the sell tax to 99–100% and every exit attempt is confiscated — a honeypot you can flip on after people have bought.

The mental shift that makes these click: in a normal company, the "owner can do X" risk is buried in legal documents and back-office systems you can't audit. In a token, every power the owner has over your money is a *named function in public code.* That's the whole reason the checklist is even possible — the power surface is enumerable. You're not guessing at the team's intentions; you're reading the exact list of things they're technically able to do to you, and then asking whether ownership is live or dead. A token where ownership is renounced has an empty power surface no matter how scary the function names look, because none of them can be called. A token where ownership is live has a power surface equal to the union of those functions.

You read these on **Etherscan** (or the chain's equivalent: BscScan, Basescan, Arbiscan). On a verified contract, the **Contract → Read Contract** and **Contract → Write Contract** tabs list every public function by name. You're scanning the function list for those names and, critically, for an `owner()` that *isn't* the zero address. If `owner()` returns a live wallet and the write tab shows a `setFee` or `blacklist` you can call, the team can call it too. (If the contract is *not* verified — no published source — that itself is a red flag; you can't read what you can't see, and a hidden contract on a token asking for your money is a no.)

A nuance that separates a careful read from a panicked one: *not every owner function is malicious.* Honest tokens routinely keep a modest, **capped** tax (say, 5% hard-coded as a maximum in the contract, used to fund liquidity or the treasury), a `pause` for the first hours of launch to stop bots, or an owner that will be renounced after the launch period. The skill is reading the *cap* and the *trajectory*, not the function name. A `setFee` capped at 5% in code is a different animal from a `setFee` that accepts any value up to 100%. A `pause` on a contract that's about to renounce is harmless; a `pause` on a live-owner contract with no renounce plan is a loaded weapon. When the function list looks scary, drop into the actual Solidity (it's right there on the verified contract) and check whether the dangerous value is bounded — the difference between a 5% cap and an unbounded tax is the difference between a fee and a confiscation.

#### Worked example: a 100% sell tax that traps \$80k of holders

A token launches verified, LP locked for a year, mint renounced — three green checks. But the owner did *not* renounce ownership, and the contract has a `setSellTax()` function with no hard cap in code. For two weeks everything works: buys and sells both go through at a 5% tax, the project looks alive, **\$80,000 of buyer money** accumulates in the pool. Then the owner calls `setSellTax(99)`. Now every sell routes 99% of its proceeds to the owner's wallet. A holder selling **\$1,000** of tokens receives **\$10** and the owner pockets **\$990**; repeat across the holder base and the owner siphons most of the **\$80,000** in the pool, one taxed sell at a time, without ever pulling liquidity or minting. The lesson: a *locked LP and renounced mint do not protect you* if ownership is live and a `setFee`-class function exists. Renounce-ownership is its own separate check.

## How to read it: a walkthrough of the pre-buy checklist

Here is the actual pass you make before buying a low-cap. It takes about five minutes once it's habit, and the tools are free. Use a token's contract address (from DEXScreener or the project, never from a random Telegram link) as your starting point.

One discipline before the checks: **get the right contract address.** A huge fraction of losses come not from a clever honeypot but from buying an *impostor* token — a scammer deploys a contract with the same name and ticker as a legitimate project and seeds a pool, then promotes the fake address in replies and Telegram groups. Always source the contract address from the project's official site or a trusted aggregator, then confirm the *exact* address matches on DEXScreener. The name and ticker mean nothing on-chain; only the address is identity. A "TOKEN" at `0xA11ce…` and a "TOKEN" at `0xB0b…` are entirely different contracts, and the scam version is often the one being pushed at you.

### Check 1 — Liquidity: is the LP locked or burned, and for how long?

This is the single highest-value check. The mechanism is the question "who holds the LP token?"

![LP token locked or burned versus held by the deployer wallet, before and after](/imgs/blogs/rug-pull-and-honeypot-detection-5.png)

How to read it:

- On **DEXScreener**, open the token's pair page. It often shows the liquidity in USD and, in the info/audit panel, a "liquidity locked" badge with a percentage and an unlock date from lockers it recognizes.
- To verify it yourself, find the **LP token's contract** (the pair address on Uniswap/PancakeSwap) on Etherscan and open its **Holders** tab. Look at where the LP tokens sit. If the top holder is a **burn address** (`0x…dead`) holding the bulk of the LP, the pool is burned — strong. If the top holder is a known **locker contract** (UNCX, Team Finance, PinkLock), the pool is locked — check the unlock *date* by reading the locker. If the top LP holder is a fresh EOA — the deployer — the pool can be pulled any second.
- **"Locked" without a date is meaningless.** A 3-day lock is not a lock. Read the actual unlock timestamp; anything under ~30 days, or a lock that releases right after a hype window, is a red flag, not a green one.

The read: LP burned → liquidity-rug risk is structurally off the table. LP locked for a real horizon → low rug risk for that window. LP in the deployer's wallet, or locked for days → **stop.**

A second number to read alongside the lock: *how deep is the liquidity relative to the market cap?* A token with a \$5,000,000 fully-diluted valuation sitting on a \$40,000 pool is fragile no matter who holds the LP — even an honest holder selling a modest position moves the price violently, and there isn't enough ETH in the pool to absorb a real exit. DEXScreener shows liquidity in USD right next to the market cap; a liquidity-to-mcap ratio in the low single-digit percent is a thin, easily-crashed market. Locked-but-thin is still a stop for a position you'd want to be able to exit.

#### Worked example: a "locked" pool that's really 3 days deep

A token's DEXScreener page shows a green "liquidity locked 100%" badge, so a casual check waves it through. You buy **\$5,000**. Reading the locker contract on Etherscan, the unlock timestamp is **72 hours away**. Three days later the lock releases, the team withdraws the **\$200,000 pool**, and the price falls to dust. Your \$5,000 is gone — not because the badge lied, but because "locked" without a date is a countdown, not a guarantee. Had you read the unlock timestamp (one click into the locker), the 72-hour window was the whole story. The rule: a lock is only as good as its remaining duration, and you must read the duration, never the badge.

### Check 2 — Mint authority: can the owner print more supply?

Open the contract on Etherscan's **Read/Write Contract** tabs and scan for a `mint` function (or `_mint` exposed through some owner-only wrapper). Then check `owner()`:

- `owner()` returns the **zero address** → ownership renounced → `mint` (and every other owner-gated function) is permanently uncallable. Safe on this axis.
- `owner()` returns a **live wallet** *and* a `mint`-class function exists with no cap → the owner can dilute you at will. This is the mint-and-dump risk; treat it as a stop for a low-cap with anonymous founders.

Many honest tokens have a fixed supply set in the constructor and no mint function at all — that's the cleanest case. **GoPlus** and **Token Sniffer** both report "mintable: yes/no" so you don't have to read bytecode, but spot-check the contract yourself when real money is involved.

The exact click sequence on Etherscan: paste the token contract address, open the **Contract** tab (it shows a green check if verified), then **Read Contract**. Scroll the alphabetized function list. `owner()` returns the current owner — click it and read the address; if it's `0x0000000000000000000000000000000000000000`, ownership is renounced and every owner-gated function is dead. `totalSupply()` plus `maxSupply()` or a cap constant tells you whether supply is fixed. Then open **Write Contract**; the presence of a `mint`, `setMaxSupply`, or owner-only `_mint` wrapper here is the live dilution surface. Two minutes, no Solidity knowledge required beyond reading function names and one address.

The subtle case to watch: a contract whose owner *looks* renounced but has a **hidden re-claim**. Some malicious contracts include a function that lets a specific address (hardcoded in the bytecode) take ownership back even after a public "renounce." GoPlus flags this as "can take back ownership" / "hidden owner" — it's the reason you don't stop at "owner() is the zero address" on an anonymous low-cap; the renounce can be theater.

### Check 3 — Owner privileges: pause, blacklist, set-fee, modify-balances

Still on the Write Contract tab, scan the function names for the dangerous four from the section above: `pause`/`setTradingEnabled`, `blacklist`/`setBlacklist`, `setFee`/`setTaxes`, and anything that looks like it can **modify balances** directly (`setBalance`, `updateBalance` — rare, and a glaring red flag if present). **GoPlus** summarizes most of this as flags: "can take back ownership," "owner can change balance," "trading cooldown," "transfer pausable," "is blacklisted." A wall of red flags here means the owner can reshape the rules of *your* position after you're in. Cross-reference with `owner()`: if ownership is renounced, these functions are dead weight; if it's live, they're loaded.

### Check 4 — The honeypot test: does a simulated sell go through?

You don't have to risk real money to test whether you can sell. **Honeypot.is** and **Token Sniffer** run a *simulation*: they execute a buy and then a sell against the live contract in a forked environment and report whether the sell succeeds, plus the actual buy tax and sell tax the contract charges. **GoPlus** flags "is_honeypot," "cannot sell all," "sell tax," and "buy tax" the same way.

- Simulated sell **succeeds** with a sane tax (say, both taxes under ~10%) → not a honeypot on current logic.
- Simulated sell **reverts**, or sell tax shows ~100%, or "cannot sell all" is flagged → **honeypot. Stop.**

How the simulators actually do this is worth understanding, because it tells you both why they work and where they fail. A tool like Honeypot.is **forks the chain** — it spins up a local copy of the current blockchain state — and then, in that sandbox, sends a buy transaction from a throwaway address followed by a sell. Because it's a fork, no real money moves; but the contract executes its real logic, so if the transfer function reverts for a non-whitelisted seller, the simulated sell reverts too, and the tool reports it. It also reads back how many tokens the buy produced versus how much ETH the sell returned, which is how it derives the *effective* buy tax and sell tax — including hidden taxes that don't appear as a labeled fee but show up as "you got back less than you should have."

This catches three distinct traps in one pass: the hard honeypot (sell reverts), the soft honeypot (sell succeeds but a 90–100% tax confiscates the proceeds), and the **max-transaction trap** (the contract caps how much you can sell in one transaction at some tiny amount, so you can technically sell but never enough to meaningfully exit). A good simulator flags "cannot sell all" for that last one.

Where the simulation *fails* is the limit you must respect: it tests the contract **as it is right now, from a generic address.** Three things can defeat it. First, the owner can flip a tax or trading switch later (Check 3), so a clean sim today says nothing about tomorrow. Second, the owner can upgrade the logic through a proxy (Check 6). Third — the nastiest — some honeypots only spring the trap **after a delay or a threshold** (e.g., transfers are fine until N blocks pass, or until liquidity crosses some size), so an early simulation passes and the trap arms once enough victims are in. The defense against all three is the same: a passing sell-sim is necessary but not sufficient. Pair it with "is ownership renounced and the contract immutable?" — if yes, the sim result is durable; if no, treat live owner power as the thing that re-arms the trap, and re-run the sim immediately before you actually buy.

### Check 5 — Holder concentration: who's holding the bag-to-be?

Open the token's **Holders** tab on Etherscan (or use **Bubblemaps**, which draws wallets as bubbles sized by holdings and links wallets funded from the same source). You're answering:

- **How concentrated is the top of the book?** If the top 1–10 wallets (excluding the locked LP and the burn address) hold a huge share — say one wallet with 30%+ of supply — a single sell craters the price. We go deep on the thresholds and the dust-wallet tricks in [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration).
- **Is the deployer still loaded?** If the wallet that deployed the token still holds a large tranche, that's a slow-rug fuse — they can bleed it into every bounce.
- **Are the "holders" really one entity?** Bubblemaps clusters wallets funded by the same address. A token bragging "5,000 holders" that's actually 30 clusters of dust wallets feeding 5 real ones is manufactured distribution, not organic demand. Wash-traded volume and Sybil holder counts are designed to look like the safety signal you're checking for; verify clustering rather than trusting the holder count on the dashboard.

Reading the Holders tab well means *labeling the top rows before you judge them.* The biggest holder is often the locked-LP or burn address — that's a feature, not a whale. A known CEX hot wallet (Etherscan labels many) isn't a single seller either. What you're hunting for is the *unlabeled fresh EOA* holding 20–40% of supply with no lock — that's a loaded gun pointed at the price. Bubblemaps makes the funding links visual: if ten of the "top holders" all trace their first deposit back to the deployer's wallet, they're one actor wearing ten hats, and the real concentration is far worse than the flat holder list suggests.

#### Worked example: a 35%-deployer wallet that caps your upside

A token passes Checks 1–4 cleanly: LP burned, mint renounced, no dangerous owner functions, sell-sim succeeds at 3% tax. But the Holders tab shows the deployer wallet still holds **35% of supply** — about **\$700,000** worth at the current **\$2,000,000 market cap** — unlocked and free to move. You buy **\$4,000**. There was never a malicious transaction; the contract is exactly as safe as it claimed. But over the next three weeks the deployer sells that 35% into every green candle, and with only **\$90,000 of liquidity** in the pool, each tranche knocks the price down hard. By the time they're done, the price is off 75% and your \$4,000 is worth about **\$1,000**. No rug, no honeypot — just a holder-concentration fuse the first four checks don't measure. The lesson: concentration is a fifth, independent risk, and a loaded deployer caps your upside even when the contract is honest.

### Check 6 — Verified contract and the proxy trap

Finally, confirm the contract is **verified** (source published) on Etherscan — an unverified contract on a token soliciting buys is a no. Then check whether it's a **proxy**. A proxy contract holds your balances but delegates its logic to a *separate* logic address, and an **upgradeable proxy** lets the owner point it at *new* logic later via an `upgradeTo()`-style call.

![Proxy contract delegating to logic v1 today and swapped to a malicious logic v2 later](/imgs/blogs/rug-pull-and-honeypot-detection-6.png)

This is the rug you can't see with a one-time check. The token can pass every simulation today because logic v1 is genuinely clean — and then the owner swaps in logic v2 that adds a sell block or a 100% tax after you've bought. Etherscan labels a proxy ("Read as Proxy"/"Write as Proxy" tabs appear; the contract often shows an "implementation" address). The defender's read: **proxy + a live admin key = a trapdoor.** A renounced/immutable contract can't do this; an upgradeable one with an active owner can. For a small anonymous-team token, an upgradeable proxy with a live admin is a stop on its own.

How to tell on Etherscan: a proxy contract shows extra tabs — **Read as Proxy** and **Write as Proxy** — and a banner like "This contract may be a proxy contract" with a linked **implementation** address (the logic v1). Click through to the implementation to read the *real* functions, because the proxy's own code is just delegation plumbing. Then ask the decisive question: *who can call the upgrade function, and is it timelocked?* If the proxy admin is a live single EOA on an anonymous launch, the upgrade can happen in one transaction with no warning — a trapdoor. If the admin is a multisig behind a public timelock (so any upgrade is announced days in advance and visible on-chain before it executes), the risk is far lower because you'd see the malicious upgrade queued before it lands. Reputable DeFi protocols use exactly that pattern; anonymous memecoins almost never do.

This is also why "I read the contract and it was clean" is not a permanent statement for a proxy token. You read logic v1. The owner owns the right to make logic v2 say anything. On an upgradeable contract, your due diligence has a shelf life measured by the owner's restraint — which, for an anonymous team, is no guarantee at all.

### Putting the pass together — the decision matrix

Run all six, then turn each into a binary read. The discipline that saves you money is treating *any single hard red flag as a stop*, not as a number to average against an exciting chart.

![Decision matrix mapping each check to its hard red flag and a buy or avoid verdict](/imgs/blogs/rug-pull-and-honeypot-detection-7.png)

The named tools, and what each is best for:

- **DEXScreener** — find the pair, see liquidity in USD, the audit/info tab's lock badge, and the holder/volume snapshot.
- **Token Sniffer** — an automated score plus simulated buy/sell, taxes, mintable/ownership flags, and clone detection. Good first pass; never the only pass.
- **Honeypot.is** — the focused sell-simulation tool: can you sell, and at what tax.
- **GoPlus** (token security API / app) — the broadest flag set: honeypot, taxes, mintable, pausable, blacklist, can-take-back-ownership, hidden-owner, proxy.
- **Etherscan / BscScan / Basescan** — the ground truth: Read/Write Contract tabs for `owner()` and the function list, the LP token's Holders tab, the proxy label, contract verification.
- **Bubblemaps** — holder clustering, to catch manufactured distribution.

Automated scores are a screen, not a verdict. They can be gamed (a token can pass an automated check and still be upgradeable), and they can false-positive on legitimate tokens with high-but-temporary launch taxes. Use them to *flag*, then confirm the high-stakes items on Etherscan yourself.

A practical division of labor: let the automated tools (Token Sniffer, GoPlus) run all six checks in seconds and give you a fast yes/no. If *anything* is flagged, or if you're about to commit a meaningful sum, drop to Etherscan and confirm the two checks that can't be faked — the LP token's actual holder and unlock date (Check 1), and `owner()` plus the proxy/implementation status (Checks 2, 6). Those are the two reads that decide whether the team can take your money, and they're the two an automated score is most often wrong about (it may not know the locker, or it may miss a hidden re-claim). The tools tell you where to look; Etherscan tells you the truth.

#### Worked example: the same \$10k buy through a green pass and a red pass

You're sizing a **\$10,000** position in two different low-caps. Token A: Token Sniffer green, GoPlus no flags; you confirm on Etherscan that the LP is burned, `owner()` is the zero address, no proxy, and Honeypot.is sells clean at a 3% tax. Six green checks. You buy \$10,000 — and now do the *real* work (tokenomics, unlocks, demand) and keep the size small, because green means "not built to rob me," not "will go up." Token B: GoPlus flags "owner can change balance" and "proxy contract"; Etherscan shows a live EOA admin on an upgradeable proxy. That's two hard red flags. You do **not** buy, no matter how good Token B's chart looks — because a single \$10,000 buy into a live-admin proxy can be turned into a \$0 bag in one upgrade transaction, and you'd have no way to exit. The discipline isn't "weigh the green against the red"; it's "any red is a stop." Token A earned five minutes of further analysis; Token B earned a hard pass.

## The limits: a clean checklist is not a green light

This is the most important section, because the checklist's biggest failure mode is *overconfidence.* Everything above protects you from a contract that is built to rob you. None of it tells you the token is a good investment, or even that you won't lose 90% honestly.

![Base rate showing about 98 percent of launched tokens go to zero](/imgs/blogs/rug-pull-and-honeypot-detection-8.png)

The base rate for low-cap tokens is brutal: of the millions launched on permissionless launchpads, only on the order of **1–2%** ever reach a meaningful market cap. The other ~98% go to zero — some via the traps above, but plenty via nothing more sinister than no demand, a top-heavy emission schedule, or the team's vesting unlocking into a thin market. A safety check filters out the *built-to-rob* tokens; it does nothing about the *honestly-dies* tokens, which are the majority.

#### Worked example: a renounced, LP-burned token that still dumps 90%

A token does everything right on the checklist: contract verified, ownership renounced (no live owner functions), LP **burned** (rug button gone), simulated sell passes at a 2% tax, no proxy. Every box is green. You buy **\$3,000** at a fully-diluted valuation of **\$2,000,000**. What the checklist *didn't* show you is the emission schedule: 70% of supply is locked in a "rewards" contract that vests linearly and starts unlocking next week. Over the following month, **\$1,400,000** of token value (70% of the \$2M FDV) unlocks and is sold into a pool with only **\$120,000** of liquidity. The price falls 90%; your **\$3,000** is now worth about **\$300**. Nobody pulled the LP. Nobody minted. Nobody set a 100% tax. The contract was exactly as safe as it claimed — and you still lost 90%, because *safe-to-trade is not the same as good-to-own.* Tokenomics, supply unlocks, real demand, and valuation are a second, separate analysis the safety check never touches.

So hold two ideas at once: the checklist is a **necessary** filter (skip it and you'll eventually buy a honeypot), but it is nowhere near **sufficient** (pass it and you've only ruled out the malicious-by-construction failures). Treat a fully-green checklist as permission to do the *real* work — read the tokenomics, the unlock schedule, the holder distribution, whether the volume is organic — and to size the position small, not as a reason to size up.

It's worth naming, explicitly, what a clean safety pass does *not* protect against, because each is a real way to lose money that lives outside the six checks:

- **Emission and unlock schedules.** A renounced, locked token can still have 70% of supply vesting into the market over months. The contract is safe; the supply overhang is not. Read the vesting and treasury contracts separately.
- **No real demand.** Most tokens go to zero simply because nobody wants them — no users, no product, no reason to hold. Safety says nothing about demand.
- **Manufactured volume.** A token can wash-trade its volume to look liquid and active. The pool can be real and the volume fake at the same time; verify whether trading is organic, not just whether you can sell.
- **Off-chain risk.** A team can locked-and-renounce everything on-chain and still run a coordinated social pump-and-dump off-chain, or simply abandon the project. The chain can't see the Telegram.
- **Your own behavior.** The most common loss isn't a contract trap at all — it's buying the top of a hyped chart, with or without a clean contract, and holding through the round-trip. The checklist can't fix position sizing or FOMO.

The honest framing: the six checks move you from "the token might be *engineered* to take my money" to "the token is *allowed* to lose my money the ordinary way." That's a real and valuable step — it eliminates the worst, most asymmetric outcomes (you can't even sell) — but it's the floor of due diligence, not the ceiling.

## Common misconceptions

**"It's locked, so it's safe."** A lock has a *duration* and a *percentage*. A 100% lock for one year is meaningfully different from a 40% lock for 3 days. And a locked LP says nothing about mint authority, owner functions, or an upgradeable proxy. In the \$80k-tax example, the LP was locked for a year and holders still got drained 99% on a sell-tax flip — because *ownership was live.* "Locked" answers one of six checks, not all six.

**"The contract is verified and audited, so I'm fine."** Verified means the source is published so you can read it — it is a prerequisite for checking, not a clean bill of health. "Audited" is even weaker: many "audits" on low-caps are unbranded PDFs the team paid for or wrote, and a real audit of v1 logic says nothing about an upgrade to v2. Read what the verified source actually *does* (the owner functions); don't stop at the green "verified" badge.

**"It has 5,000 holders, so it must be legit."** Holder counts are trivially manufactured by spraying tokens to fresh wallets, and volume is trivially washed by trading with yourself across wallets. Both are *designed* to spoof the exact safety signal you're checking. Bubblemaps clustering and an honest look at whether the volume is real cut through it; the raw count on a dashboard does not.

**"The chart only goes up, that's bullish."** In a brand-new low-cap, a chart with buys and no sells is the *signature of a honeypot*, because a contract where nobody can sell cannot print a red candle. A clean parabolic uptrend in a token you haven't sell-tested should *increase* your suspicion. Run the simulated sell before you read anything into the chart.

**"I'll just sell fast if it looks like a rug."** Against a liquidity rug or a tax-flip, the malicious transaction and your escape are in the same block or seconds apart, and the owner is whitelisted while you're not. You will not out-click a contract that's built to revert your sell. The only reliable defense is *not buying the trap in the first place* — which is exactly what the pre-buy checklist is for.

**"It passed Token Sniffer, so it's safe."** An automated score is a fast screen, not the truth. It can miss an upgradeable proxy, a hidden owner re-claim, or a delayed-trap honeypot, and it can false-positive on an honest token with a temporary launch tax. A green score earns the token a manual look at the two reads that can't be faked — the LP holder/unlock and `owner()`/proxy status on Etherscan — not your money. Treat the tool as the metal detector and Etherscan as the bag check.

**"A new token, but the influencer holds it, so it's fine."** A paid promoter holding a bag tells you they were paid or pre-loaded, not that the contract is safe — in fact, a known wallet loudly buying a low-float token is a classic setup to lure copiers before a dump. The contract checks are independent of who's shilling it; run them regardless, and treat a loud, well-funded "endorsement" of a brand-new low-cap as a reason for *more* scrutiny, not less.

## The playbook: what to do with it

The signal → the read → the action → the false-positive, for each check. Run them in order; one hard stop ends the pass.

- **LP ownership.** Signal: where the LP token sits (Etherscan Holders tab / DEXScreener lock badge). Read: burn address or long-dated locker = safe; deployer wallet or sub-30-day lock = rug-able. Action: deployer-held or short lock → **avoid**. Burned/long-locked → proceed to the next check. False-positive: a "locked" badge with no readable date — verify the unlock timestamp before trusting it.

- **Mint authority.** Signal: `mint`-class function + `owner()` on Etherscan; "mintable" flag on GoPlus/Token Sniffer. Read: renounced owner or no mint = safe; live owner + uncapped mint = dilution risk. Action: live uncapped mint on an anonymous-team low-cap → **avoid**. False-positive: a *capped* mint or a renounced owner is fine — read the cap, don't just see the word `mint`.

- **Owner functions.** Signal: `pause`/`blacklist`/`setFee`/balance-edit functions + `owner()`. Read: renounced = dead; live = loaded. Action: live owner with a `setFee`/`blacklist`/`pause` and an anonymous team → **avoid**. False-positive: many real tokens have a modest, *capped* tax and pause for launch — a 5% capped sell tax is not a 100% trap; read the cap and whether ownership will be renounced.

- **Sell test.** Signal: Honeypot.is / Token Sniffer / GoPlus simulated sell + taxes. Read: sell succeeds at sane tax = not a honeypot *now*; revert or ~100% tax = honeypot. Action: revert or near-100% sell tax → **avoid, hard stop**. False-positive: a one-time launch tax can read high for a few minutes — re-run the sim; persistent revert is the real signal.

- **Holder concentration.** Signal: Etherscan Holders / Bubblemaps clusters. Read: a single non-LP wallet with a dominant share, or a loaded deployer, is a slow-rug fuse; clustered "holders" are manufactured. Action: extreme concentration or obvious clustering → **avoid or size tiny**. False-positive: a CEX hot wallet or the locked-LP/burn address looks like a whale but isn't a seller — label the top holders before judging.

- **Verified + proxy.** Signal: contract verification + proxy label on Etherscan; "proxy"/"hidden owner" on GoPlus. Read: unverified = no; immutable verified = good; upgradeable proxy + live admin = trapdoor. Action: unverified, or upgradeable with a live anonymous admin → **avoid**. False-positive: some reputable projects use audited, time-locked, multisig-governed proxies — proxy alone isn't damning, but on a small anonymous launch it's a stop.

The meta-rule that ties it together: **any single hard red flag is a stop.** You do not get to average a honeypot sell-revert against a great chart, or a deployer-held LP against a big holder count. And the rule beyond the rule: a fully-green checklist earns you the right to do the *real* due diligence (tokenomics, unlocks, demand, valuation) and to size small — it never earns the word "safe."

## Further reading & cross-links

- [Tokens, on-chain transfers and approvals](/blog/trading/onchain/tokens-onchain-transfers-and-approvals) — what a token contract is, how balances and approvals work, and why an approval you sign is its own attack surface.
- [Supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — the thresholds, dust-wallet tricks, and clustering behind Check 5.
- [DeFi protocols: Uniswap, Aave, and MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — how AMM pools price tokens and how liquidity provision (and the LP token) actually works.
- [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) — why a token is owner-written code in the first place, and what "smart contract" really means.
- [Tornado Cash and sanctioning code](/blog/trading/crypto/tornado-cash-and-sanctioning-code) — where rug and honeypot proceeds often head next, and the defender's view of obfuscation.
