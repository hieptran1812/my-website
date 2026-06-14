---
title: "DeFi Protocols: How Uniswap, Aave, and MakerDAO Rebuilt Banking Without Banks"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A from-zero tour of decentralized finance: how automated market makers, overcollateralized lending, and crypto-backed stablecoins rebuild trading, borrowing, and money itself as open code."
tags: ["defi", "uniswap", "aave", "makerdao", "stablecoins", "amm", "liquidity-pool", "lending", "flash-loan", "ethereum", "crypto", "smart-contracts"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Decentralized finance (DeFi) rebuilds trading, lending, and borrowing as open smart contracts with no bank in the middle; Uniswap, Aave, and MakerDAO show both the elegance of composable "money legos" and the new risks of turning code into a bank.
>
> - A **smart contract** is a program that holds money and runs by itself on a blockchain — no branch, no loan officer, no business hours. DeFi is just a stack of these programs.
> - Uniswap replaced the order book with a **liquidity pool** priced by a single formula, `x * y = k`; Aave replaced the loan officer with an **overcollateralized** vault; MakerDAO replaced the central bank with a crypto-backed stablecoin, **DAI**.
> - The killer feature is **composability**: any contract can call any other, so one transaction can borrow, swap, and stake across four protocols at once — including a **flash loan** that borrows millions and repays it in the same transaction.
> - The killer bug is also composability: a smart-contract flaw, a manipulated **oracle**, or a collateral **de-peg** propagates instantly, and hacks have drained hundreds of millions (Euler, ~\$197M; Mango, ~\$117M; the Terra de-peg, ~\$40B of value erased).
> - DeFi does not abolish risk — it relocates it from a regulated balance sheet you can sue into open code you cannot. That trade is the whole story.

The diagram above is the mental model for everything that follows: a trader sends \$10,000 into a pool of two tokens, a formula reprices the pool, and the trader walks away with the other token — no exchange, no market maker quoting prices, no counterparty agreeing to the trade. The "bank" is a few hundred lines of code that anyone can read and no one can switch off. That is the strange, powerful, and occasionally catastrophic idea at the center of DeFi.

![An AMM swap routed through a liquidity pool](/imgs/blogs/defi-protocols-uniswap-aave-makerdao-1.png)

Traditional finance runs on institutions you trust to hold the ledger: a bank records your deposit, an exchange matches your buy order against someone's sell order, a clearinghouse guarantees the trade settles. DeFi removes those institutions and replaces them with programs that hold the ledger themselves. The promise is openness — anyone with a wallet and an internet connection can use the same financial rails as a hedge fund, at the same fees, with the same code. The price is that the code *is* the institution: if it has a bug, there is no FDIC, no chargeback, no regulator to call. This post builds the whole thing from zero — what a smart contract is, how a pool sets a price, how a loan with no credit check can possibly work — and then walks through the three protocols that defined the category and the failures that taught the category its limits.

## Foundations: every term this post turns on

Before any of the protocols make sense, we need a shared vocabulary. None of these terms require prior finance knowledge. I will define each one the first time it matters and keep the definitions concrete.

**A blockchain** is a shared public ledger that thousands of independent computers maintain a copy of, agreeing on its contents through a consensus process so that no single party can rewrite history. Bitcoin's blockchain mostly records "address A sent X coins to address B." Ethereum's blockchain does that too, but it also records *programs* and the results of running them. DeFi lives almost entirely on Ethereum and chains like it (Arbitrum, Base, Polygon) that copy Ethereum's programming model.

**A token** is an entry in a ledger that some contract maintains, representing a unit of value — a coin (ETH, the native currency of Ethereum), a stablecoin (USDC, DAI), or a share in some pool. "You own 3 ETH" means the ledger has your address credited with 3 ETH; there is no physical coin.

**A wallet** is the software (and the cryptographic keys) that lets you sign transactions — instructions the blockchain executes, like "send 1 ETH to this contract." Your wallet *is* your identity in DeFi; there is no username, no password reset. Whoever holds the keys controls the funds. This is called **self-custody**: you hold your own assets directly, rather than a bank or exchange (a **custodian**) holding them for you.

**On-chain** means "recorded and executed on the blockchain, visible to everyone." When a DeFi protocol does something on-chain, the action and its result are public and permanent. This is the opposite of a bank's internal ledger, which only the bank can see.

**A smart contract** is a program deployed to the blockchain that holds funds and runs exactly as written whenever someone calls it. "Smart" oversells it — it is not intelligent, it is *automatic and unstoppable*. Once deployed, it does precisely what its code says, every time, for everyone, with no human in the loop. A vending machine is the standard analogy: put in the right input, get the defined output, and the machine does not care who you are or whether it is a holiday. A DeFi protocol is a set of these contracts.

**A liquidity pool** is a smart contract holding a reserve of two (or more) tokens that anyone can trade against. Instead of matching your buy order to someone's sell order, you trade directly with the pool: you put one token in, the pool sends the other token out, and the pool's reserves change. The people who deposited the tokens into the pool are **liquidity providers (LPs)**, and they earn a cut of every trade.

**An automated market maker (AMM)** is the rule a pool uses to decide how much of token B to give you for the token A you deposit. The most famous rule is the **constant-product formula**, written `x * y = k`: if the pool holds `x` of one token and `y` of the other, their product `k` must stay the same after your trade. We will work this out with real numbers in a moment — it is the single most important equation in DeFi.

**Impermanent loss** is the cost a liquidity provider quietly pays when the two tokens in the pool drift apart in price. It is the gap between "I provided liquidity" and "I just held the two tokens in my own wallet." It is real money, it is frequently misunderstood, and we will compute it.

**Overcollateralized lending** is how DeFi makes loans without a credit check. To borrow \$100, you must first lock up *more* than \$100 of some other asset — say \$150 of ETH — as **collateral**. The protocol never trusts you; it trusts the collateral. If the collateral falls in value, the protocol sells it.

**Liquidation** is that forced sale. When your collateral drops close to the value of your loan, the protocol lets anyone repay your debt in exchange for your collateral at a discount. You lose the collateral; the loan is cleared; the lender is made whole. No phone call, no grace period — just code.

**A stablecoin** is a token engineered to hold a steady value, usually \$1. Some are backed by real dollars in a bank (USDC, USDT — see the companion piece on [Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar)). One, **DAI**, is backed by *crypto* locked in smart contracts. A **de-peg** is when a stablecoin loses its anchor — a "\$1" coin trading at \$0.90 or \$0.00.

**An oracle** is a service that reports off-chain facts (mostly prices) onto the blockchain so contracts can use them. A lending protocol needs to know "what is ETH worth right now?" to decide whether to liquidate you. It cannot look at a stock ticker; it reads an oracle. Oracles are a notorious weak point: if you can fool the oracle, you can fool every contract that trusts it.

**Yield** and **APY** describe what you earn. Yield is the return on capital you put to work — fees from a pool, interest from a lending market. **APY (annual percentage yield)** annualizes it and includes compounding, so a "5% APY" means \$10,000 grows to about \$10,500 over a year if rates hold. Beware: a 1,000% APY almost always means either fleeting token incentives or an outright scam.

**Total value locked (TVL)** is the headline metric of DeFi — the dollar value of all assets currently deposited in a protocol's contracts. A protocol with \$10B TVL is holding \$10B of user funds. It is a measure of scale and trust, and it is also exactly the number an attacker wants to drain.

**Composability** is the property that makes DeFi more than the sum of its parts: because every protocol is open code on the same chain, any contract can call any other. Developers call this "money legos" — you snap protocols together. One transaction can mint a stablecoin in MakerDAO, lend it on Aave, and use the receipt as collateral somewhere else.

**A flash loan** is composability taken to its logical extreme: a loan of any size, with no collateral, that must be borrowed and repaid *within a single transaction*. If you cannot pay it back by the end of the transaction, the entire transaction reverts as if it never happened — so the lender is never at risk. It sounds impossible until you see how a blockchain transaction works, which we will.

### A map of the four categories

Almost everything in DeFi is one of four primitives, or a wrapper around them: an **exchange** (a place to swap one token for another), **lending** (a place to deposit and earn, or post collateral and borrow), a **stablecoin** (a token that holds a fixed value), and **staking** (locking an asset to secure a network and earn a yield). The four protocols this post centers on map one-to-one onto those categories — Uniswap is the exchange, Aave the lending market, MakerDAO the stablecoin issuer, Lido the staking layer. Hold this taxonomy in mind; it keeps the protocols from blurring together.

![The four DeFi categories with their flagship protocols](/imgs/blogs/defi-protocols-uniswap-aave-makerdao-7.png)

Before the deep dive, one comparison is worth fixing in place: each of the four protocols owns exactly one of these primitives, earns money in a different way, and carries its own characteristic risk. They do not really compete with one another — they compose. The matrix below is the cheat sheet for the rest of the article; every cell is unpacked in the section that follows.

![Four protocols compared by function, revenue, and risk](/imgs/blogs/defi-protocols-uniswap-aave-makerdao-5.png)

Uniswap earns a 0.30% fee on every swap and its providers carry impermanent loss; Aave earns the spread between borrow and supply rates and carries bad-debt risk when liquidations fail; MakerDAO earns a stability fee on minted DAI and carries the risk that its collateral de-pegs; Lido takes a cut of staking rewards and carries the risk that its stETH token slips from its peg. Same shape, different primitive. With that vocabulary and map in hand, the rest of the post is a tour of how these pieces fit together — and where they break.

## How a swap with no exchange actually works: Uniswap and the AMM

Start with the simplest financial act: trading one asset for another. On a normal exchange this needs an **order book** — a list of people willing to buy at various prices and people willing to sell at various prices — and a matching engine that pairs a buyer with a seller. Order books need market makers constantly posting quotes, which needs fast servers, which needs a company running them. None of that survives the move on-chain, where every action costs gas and takes seconds.

Uniswap, launched in 2018, threw out the order book entirely. Its insight: you do not need a counterparty if you have a *pool* and a *formula*. The way this works is that a liquidity pool holds reserves of two tokens, and a constant-product rule decides the exchange rate from those reserves alone. Trade against the pool, and the pool reprices itself.

![An AMM swap routed through a liquidity pool](/imgs/blogs/defi-protocols-uniswap-aave-makerdao-1.png)

Here is the rule, stated plainly. Let the pool hold `x` units of token A and `y` units of token B. Define their product:

```
k = x * y
```

The pool promises to keep `k` constant across any trade (before fees). So if you put some token A in, the pool's `x` goes up, and to keep `k` the same, its `y` must go down — the amount it goes down by is exactly the token B you receive. The *price* you got is just the ratio of what you put in to what you took out. There is no quote, no spread set by a human; the price is a mathematical consequence of the pool's current balances.

This has a beautiful property and an ugly one. The beautiful one: the pool can never run out and never refuses a trade, because as `x` grows, each additional unit of A buys less and less B — the price moves against you, so it would take infinite A to drain all the B. The ugly one: big trades move the price a lot. That price movement is called **slippage** or **price impact**, and it is the AMM's version of a bid-ask spread.

#### Worked example: a \$10,000 swap and its price impact

Take a pool holding 100 ETH and 300,000 USDC (USDC is a \$1 stablecoin, so a USDC is a dollar here). The starting price is implied by the reserve ratio: 300,000 USDC / 100 ETH = \$3,000 per ETH. The constant is:

```
k = 100 * 300,000 = 30,000,000
```

You want to buy ETH with \$10,000. Ignore fees for a second. You add 10,000 USDC, so the USDC reserve becomes 310,000. To keep `k` fixed, the ETH reserve must fall to:

```
new ETH = k / new USDC = 30,000,000 / 310,000 = 96.774 ETH
```

So the pool now holds 96.774 ETH, down from 100. You receive the difference:

```
ETH out = 100 - 96.774 = 3.226 ETH
```

You paid \$10,000 for 3.226 ETH, an effective price of \$10,000 / 3.226 = \$3,100 per ETH. But the pool's price *started* at \$3,000. You paid about 3.3% more than the starting price — that is your price impact. And it gets worse fast: a \$100,000 order against this same pool would push the effective price to roughly \$3,333, an 11% premium, because you are walking further up the curve. Now add Uniswap's 0.30% fee, charged on the input: about \$30 of your \$10,000 is skimmed off and left in the pool for the liquidity providers, so you actually get a hair less ETH.

The one-sentence intuition: in an AMM there is no fixed price — every trade pays a price set by how much it itself shifts the pool, so small trades are cheap and large trades are expensive, by design.

### Who fills the pool, and what they earn

The pool's reserves do not appear from nowhere. **Liquidity providers** deposit equal *value* of both tokens — for our pool, someone supplying 1 ETH must also supply 3,000 USDC at the starting price — and in return receive **LP tokens** representing their share of the pool. Every swap pays the 0.30% fee into the pool, growing the reserves, so the LP tokens slowly become redeemable for more than was deposited. That fee income is the LP's yield. A pool doing \$10 million of daily volume at 0.30% generates \$30,000 a day in fees, split among LPs by share. Supply 1% of the pool and you earn \$300 a day, or about \$109,500 a year on your slice — an APY that depends entirely on volume.

So far this sounds like free money: deposit two tokens, collect fees forever. It is not, and the reason is impermanent loss.

### Impermanent loss: the LP's hidden cost

When the two tokens drift in price, arbitrageurs trade against the pool to bring its implied price back in line with the rest of the market. Every such trade nudges the pool's composition. The result is that an LP ends up holding *more of the token that fell* and *less of the token that rose* — the pool automatically sells you out of the winner and into the loser. Compared with simply holding the two tokens in your wallet, you come out behind. The loss is called "impermanent" because it reverses if prices return to where they started; if they do not, it becomes permanent the moment you withdraw.

#### Worked example: impermanent loss on an ETH/USDC position

You deposit into the 100-ETH / 300,000-USDC pool when ETH is \$3,000. Say you supply 1 ETH and 3,000 USDC — \$6,000 total, and you own 1% of the pool. Now ETH doubles to \$6,000 on outside exchanges.

Arbitrageurs buy ETH from your pool (it is cheap relative to the market) until the pool's price catches up to \$6,000. The pool's reserve ratio must satisfy two things at once: the new price (USDC / ETH = 6,000) and the constant (ETH * USDC = 30,000,000). Solving, the whole pool rebalances to about 70.71 ETH and 424,264 USDC. Your 1% share is now 0.7071 ETH and 4,242.64 USDC.

Value of your share at the new price: 0.7071 ETH * \$6,000 + \$4,242.64 = \$4,242.60 + \$4,242.64 = **\$8,485.24**.

Now compare with just holding: 1 ETH * \$6,000 + \$3,000 = **\$9,000**.

You are behind by \$9,000 - \$8,485.24 = **\$514.76**, about 5.7% of the hold value. That gap is impermanent loss. The fees you earned while providing liquidity offset some of it — if the pool paid you more than \$515 in fees over the period, you still came out ahead. That is the LP's real bet: fees versus divergence.

The one-sentence intuition: providing liquidity is a bet that trading fees will out-earn the cost of the pool quietly selling your winners — when prices move a lot in one direction, that bet often loses.

### Why this matters beyond Uniswap

The AMM model spread across all of DeFi because it needs no off-chain infrastructure: the pool *is* the market, the formula *is* the market maker. Uniswap alone has processed well over \$2 trillion in cumulative volume since launch (as of early 2026, approximate). Its later versions sharpened the economics without abandoning the core idea.

The big upgrade was **concentrated liquidity**, introduced in Uniswap v3. In the original design, an LP's capital is spread across every possible price from zero to infinity, even though almost all trading happens near the current price — so most of the capital sits idle. Concentrated liquidity lets an LP say "deploy my money only between \$2,800 and \$3,200 of ETH." Inside that band, the same dollar of capital provides far deeper liquidity and earns far more fees — sometimes a dollar of v3 liquidity does the work of \$20 or more of the old kind. The trade-off is sharper: if ETH leaves the band, the LP earns nothing and is fully converted into the losing asset, a more severe form of impermanent loss. The provider has effectively become an active market maker who must manage a range, not a passive depositor.

Consider the stakes with a number. An LP supplies \$50,000 concentrated in a tight \$2,900–\$3,100 band on a high-volume pool and, while price stays inside, earns fees that annualize to perhaps 40% APY — \$20,000 a year — because the capital is doing 20x the work. But if ETH gaps to \$3,500 and the LP does not move the range, the position stops earning and sits entirely in the asset that fell relative to the move. Concentrated liquidity does not abolish the fee-versus-divergence bet from the impermanent-loss example; it amplifies both sides of it. The core never changed: price comes from reserves, and big trades pay for the privilege. This same machinery underpins Ethereum's role as a settlement layer for programmable value, a foundation explored in [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money).

## Lending without a loan officer: Aave and Compound

A bank loan is an act of trust extended to a *person*. The bank checks your credit, verifies your income, and lends you more than you have posted — that is the entire point of a mortgage or a credit line. A blockchain knows none of this. It does not know who you are, whether you have a job, or whether you will pay it back. So DeFi lending inverts the model: it trusts no one and demands collateral worth more than the loan.

![A bank loan versus a DeFi overcollateralized loan](/imgs/blogs/defi-protocols-uniswap-aave-makerdao-3.png)

Aave (launched 2017 as ETHLend, rebuilt as Aave in 2020) and Compound (2018) are the two flagship lending protocols, and they work almost identically. Each runs a set of **pooled money markets**: suppliers deposit an asset (USDC, ETH, DAI) into a shared contract and earn interest; borrowers take that same asset out, paying interest, but only after locking up collateral. There is no matching of a specific lender to a specific borrower — everyone supplies into one pool and borrows from the same pool, and an interest-rate formula keeps the two sides in balance.

### Where the interest rate comes from

The rate is not set by a committee. It is a function of **utilization** — the fraction of the pool that is currently borrowed. When utilization is low (lots of idle supply), borrowing is cheap to attract demand. As utilization rises toward 100%, the rate climbs steeply to attract new suppliers and discourage further borrowing, protecting the pool from being fully drained (suppliers need some liquidity to withdraw). These are **variable rates**: they update every block as the pool's utilization changes. Aave also offers a stable-rate option, but the variable rate is the heart of the system.

The rate curve has a deliberate **kink**. Below a target utilization (say 80–90%) the rate rises gently; above it, the slope turns steep — sometimes the rate doubles over the last few percent of utilization. The kink is the protocol's pressure valve: it makes the last sliver of available liquidity very expensive to borrow, so there is always *some* idle balance for suppliers who want to withdraw. A bank manages this with reserve requirements and a central bank standing behind it; Aave manages it with a piecewise function that nobody can override.

The protocol takes a slice of the interest — the **reserve factor** — as its revenue, building up a safety buffer and paying the rest to suppliers. The gap between what borrowers pay and what suppliers earn is the protocol's spread, the same way a bank's net interest margin is its spread. Newer versions add finer controls — **isolation mode** caps how much can be borrowed against a riskier collateral asset and restricts what it can be used to borrow, and **efficiency mode** allows higher leverage between assets that track each other closely (like two dollar stablecoins, or ETH and stETH). Each lever is a parameter the protocol's governance sets and can change by vote — a reminder that "no loan officer" does not mean "no decisions," only that the decisions are made collectively and enforced by code.

One feature is easy to miss: Aave is itself a flash-loan lender. Anyone can borrow from its pools with no collateral inside a single transaction for a 0.09% fee, which is exactly the instrument the flash-loan example below uses. The same pooled liquidity that funds ordinary loans also funds atomic, zero-collateral loans — a clean illustration of how one protocol's plumbing becomes another transaction's raw material.

#### Worked example: the APY on \$10,000 supplied to a lending pool

You supply \$10,000 of USDC to Aave. Suppose the USDC market is 80% utilized and the borrow rate at that utilization is 6% APY. Borrowers are paying 6% on the borrowed 80% of the pool. The interest those borrowers generate, spread across *all* suppliers (including the idle 20%), and net of, say, a 10% reserve factor, lands the supply APY at roughly:

```
supply APY ~= borrow rate * utilization * (1 - reserve factor)
            = 6% * 0.80 * 0.90
            = 4.32%
```

On your \$10,000, that is about \$432 over a year, compounding continuously as interest accrues every block. If borrowing demand surges and utilization hits 95%, the borrow rate might jump to 12% and your supply APY to about 10.3% — \$1,030 a year. If demand collapses and utilization falls to 30%, your APY might be under 1.5%. The one-sentence intuition: in a DeFi money market your yield is not promised by anyone — it is borrowers' demand for the asset, divided across all suppliers, updating block by block.

### Borrowing, the health factor, and automatic liquidation

Now the borrower's side. You deposit \$150 of ETH and borrow \$100 of USDC. The protocol tracks a **health factor**, essentially "how much cushion is left before your collateral can no longer cover your debt." Each collateral asset has a **liquidation threshold** — for ETH on Aave, around 80%, meaning your debt may not exceed 80% of your ETH's value. Cross that line and you are liquidatable.

#### Worked example: the price at which a \$100 loan gets liquidated

You deposit ETH worth \$150 (say 0.05 ETH at \$3,000) and borrow \$100 of DAI. ETH's liquidation threshold is 80%. You are safe as long as:

```
debt <= 0.80 * collateral value
$100 <= 0.80 * collateral value
collateral value >= $125
```

Your collateral is 0.05 ETH. It is worth \$125 when ETH = \$125 / 0.05 = **\$2,500**. So the moment ETH falls from \$3,000 to \$2,500 — a 16.7% drop — your position becomes liquidatable. A **liquidator** (any bot watching the chain) repays your \$100 of DAI and seizes a chunk of your ETH plus a **liquidation penalty** (commonly 5–10%) as their reward. You keep whatever collateral is left after the debt and penalty are taken, but you have lost the penalty and the upside on the seized ETH. There was no warning, no margin call you could answer — the liquidation transaction simply lands.

The one-sentence intuition: a DeFi loan has no human grace period; the instant your collateral ratio breaks, code sells your collateral at a discount to whoever calls the function first.

### How the protocols make money and wield power

Aave and Compound earn the reserve-factor spread on every market, which on billions of TVL is a substantial revenue stream. But the deeper form of power is **governance**: each protocol issues a governance token (AAVE, COMP) whose holders vote on parameters — which assets to list, what collateral factors to set, how big the reserve factor is. Whoever accumulates enough tokens can steer the protocol. That is a feature (no central operator) and a risk (a **governance attack**, which we will see is not hypothetical).

The risk that defines lending protocols is **bad debt**. If a collateral asset crashes faster than liquidators can act — a thin market, a chain congested with transactions, a price gap overnight — the protocol can end up holding collateral worth less than the loan it backed. That shortfall is a loss the protocol's reserves (or, ultimately, its token holders) must eat. We will see exactly this in the real-markets section.

## Programmable money itself: MakerDAO and DAI

Uniswap rebuilt the exchange; Aave rebuilt the bank loan. MakerDAO, launched in 2017, attempted something more audacious: to rebuild *money* — a stablecoin that holds \$1 without a single dollar in a bank account, backed instead by crypto locked in smart contracts.

The token is **DAI**. Unlike USDC or Tether, which hold reserves of real dollars and Treasuries (the model dissected in the [stablecoins explainer](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar)), early DAI was backed entirely by crypto collateral — chiefly ETH — locked in contracts called **vaults** (originally "CDPs," collateralized debt positions). The mechanism is the same overcollateralized loan we just met, used as a printing press.

### How DAI is minted and why it holds \$1

To create DAI, you lock collateral in a vault and *mint DAI as a loan against it*. The system demands heavy overcollateralization — historically a minimum **collateralization ratio** of 150% for ETH, meaning to mint \$100 of DAI you lock at least \$150 of ETH. You now hold \$100 of freshly created DAI, and you owe \$100 of DAI plus a **stability fee** (interest) to unlock your ETH later. To get your collateral back, you repay the DAI (which is then *burned*, destroyed) plus the fee.

#### Worked example: minting DAI and the liquidation price

You lock \$300 of ETH (0.1 ETH at \$3,000) and mint the maximum DAI at a 150% minimum ratio:

```
max DAI = collateral / 1.50 = $300 / 1.50 = $200 DAI
```

You walk away with 200 DAI — \$200 of spendable stable money — while still being exposed to ETH's price (you get your ETH back when you repay). But if ETH falls, your ratio tightens. Your vault is liquidated when its collateral value drops to 150% of your \$200 debt, i.e. \$300 of collateral value. Wait — you started at exactly \$300, so you have no room? Right: minting the absolute maximum leaves zero buffer. Prudent users mint far less. If you instead minted only \$100 DAI against the \$300 of ETH (a 300% ratio), you would be liquidated only when collateral fell to 150% of \$100 = \$150, i.e. when ETH dropped from \$3,000 to \$1,500 — a 50% crash. The one-sentence intuition: DAI is not "printed from nothing"; every DAI in existence is a loan someone took against over-150%-valued crypto they could lose to liquidation.

That liquidation machinery is what holds the peg from below. If ETH crashes and a vault becomes undercollateralized, the system **auctions** the collateral for DAI, using the proceeds to cancel the debt. The peg is also defended by two levers Maker's governance raises or lowers like interest rates: the **stability fee** (the interest minters pay) and the **DAI Savings Rate** (the yield earned by simply holding DAI in a savings contract). Raise the savings rate and demand to hold DAI rises, pushing a sagging price back toward \$1; raise the stability fee and minting new DAI gets costlier, throttling supply when DAI trades above \$1. This is, quite literally, monetary policy — open-market operations and a policy rate — run by token-holder vote rather than a central bank board. It is a striking mirror of how real money is created through lending, a process traced in [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).

There is also a blunt instrument for the peg's upper and lower edges: the **Peg Stability Module (PSM)**, which lets anyone swap USDC for DAI one-for-one (and back) directly with the protocol, no slippage. When DAI drifts above \$1, arbitrageurs mint DAI through the PSM with USDC and sell it, pushing the price down; when it drifts below, they buy cheap DAI and redeem it for USDC, pushing it up. The PSM is enormously effective at pinning the peg — and it is also the clearest example of Maker's centralization drift, because it means a large share of DAI is, in practice, a claim on USDC, which is itself a claim on dollars at a regulated company. A "decentralized" stablecoin partly anchored to a centralized one is a compromise the white paper never imagined.

### The drift toward real-world backing

Pure crypto backing has a problem: in a crash, *all* the collateral falls at once, and the system can be stressed faster than auctions clear (exactly what happened on Black Thursday, below). Over time Maker diversified — accepting USDC and other stablecoins as collateral, then real-world assets like short-term US Treasuries, which now back a large share of DAI. This made DAI far more stable but also more *centralized*: a stablecoin partly backed by US Treasuries held by off-chain partners is no longer the trustless crypto-only money of the white paper. That tension — decentralization versus stability — is the central trade-off of the whole protocol, and Maker's 2024 rebrand toward "Sky" and a new token reflects its ongoing attempt to navigate it.

## Liquid staking: Lido and the productive collateral problem

There is one more primitive to meet, because it ties the others together. On Ethereum, you can **stake** ETH — lock it up to help secure the network — and earn a yield (roughly 3–4% as of 2026, approximate) for doing so. The catch: staked ETH is locked and illiquid. You cannot trade it, lend it, or use it as collateral while it is staked.

**Lido** solves this with **liquid staking**. You deposit ETH with Lido; it stakes the ETH on your behalf and hands you back a token, **stETH**, that represents your staked ETH plus accruing rewards. stETH trades freely, can be supplied to Aave, swapped on Uniswap, or used to mint DAI — so you earn the staking yield *and* keep your capital productive. At its peak Lido held tens of billions in TVL, making it one of the largest DeFi protocols by that measure.

The risk is twofold. First, **peg risk**: stETH is supposed to track ETH one-for-one, but in stressed markets it can trade at a discount (it fell to about \$0.94 per \$1 of ETH during the 2022 turmoil), which cascades into every protocol that accepted it as collateral. Second, **concentration**: when one liquid-staking provider holds a large share of all staked ETH, it becomes a systemic node — a bug or a governance failure there reverberates through the network's security itself. Liquid staking is enormously useful and quietly one of DeFi's biggest concentration risks at once.

## Composability and flash loans: the money legos

We now have the four primitives — swap, lend, stable money, staking. The reason DeFi feels less like four separate apps and more like one programmable financial system is **composability**: every protocol is open code on the same chain, so any contract can call any other inside a single transaction. Developers snap them together like Lego bricks. Conceptually, one transaction can route a single pile of capital through four protocols and come out the other side earning stacked yield.

![Composability: protocols calling each other in one transaction](/imgs/blogs/defi-protocols-uniswap-aave-makerdao-2.png)

Consider a single transaction that locks ETH in MakerDAO to mint DAI, supplies that DAI to Aave to earn interest, swaps some on Uniswap into more ETH, and stakes that ETH through Lido for stETH — all atomically, with no human approving each step. This is what "money legos" means in practice: the output of one protocol is the input of the next, and the composition is itself a new product. Entire protocols (yield aggregators like Yearn) exist only to automate these compositions, moving user funds to wherever the yield is highest. The whole arrangement sits in a layered stack, each layer depending on the one below it.

![The four layers of the DeFi stack](/imgs/blogs/defi-protocols-uniswap-aave-makerdao-4.png)

At the bottom is **settlement** — Ethereum itself, which orders transactions and secures roughly tens of billions of dollars of value. Above it sit the **assets** (ETH, USDC, DAI, stETH). Above those, the **protocols** (Uniswap, Aave, Maker, Lido). At the top, **aggregators and interfaces** (1inch for best-price routing, wallets, dashboards) that bundle the protocols into one-click products. Each layer trusts the one below; a flaw at the settlement or asset layer propagates up through everything built on it.

### Flash loans: a loan that exists for one transaction

Composability enables the single most counterintuitive instrument in finance: the **flash loan**. You can borrow an enormous sum — millions of dollars — with *zero collateral*, on one condition: you must repay it (plus a small fee) before the transaction ends. If you cannot, the entire transaction reverts, unwinding every step as if it never happened, and the lender's funds are never at risk.

This only works because a blockchain transaction is **atomic**: all of its steps either succeed together or fail together. The protocol lends you the money, runs your code, and checks at the end, "is the loan repaid?" If yes, the transaction commits. If no, the chain throws everything away. There is no moment in real time where the money is "out" and unrepaid — it is all one indivisible step.

![Composability lets a flash loan chain protocols in one transaction](/imgs/blogs/defi-protocols-uniswap-aave-makerdao-2.png)

#### Worked example: a flash loan borrowed and repaid in one transaction

You spot that ETH is \$3,000 on Uniswap but \$3,030 on another exchange (Sushiswap). With no money of your own, you do this in one transaction:

1. **Flash-borrow** \$3,000,000 of USDC from Aave. Fee is 0.09%, so you owe back \$3,002,700.
2. **Buy ETH on Uniswap** at \$3,000: \$3,000,000 / \$3,000 = 1,000 ETH (ignoring the price impact for clarity; a real arbitrageur sizes the trade so the impact stays below the spread).
3. **Sell those 1,000 ETH on Sushiswap** at \$3,030: 1,000 * \$3,030 = \$3,030,000.
4. **Repay Aave** \$3,002,700.
5. **Keep the rest:** \$3,030,000 - \$3,002,700 = **\$27,300** profit, with no starting capital.

If step 3 had not produced enough to repay step 4 — say the price gap had closed before your transaction landed — the repayment would fail, and the chain would revert the entire sequence; you would lose only the gas fee, not the loan. The one-sentence intuition: a flash loan turns capital into a non-issue for arbitrage and refinancing — but the same instrument hands an attacker millions of dollars of temporary firepower to bend any protocol that misprices something within a single transaction.

That dual nature — a legitimate tool for arbitrage, refinancing, and collateral swaps, *and* the favorite weapon of attackers — is why flash loans are the perfect bridge to the risks.

## The risks: when code is the bank, the bugs are the bank runs

Every upside above has a matching risk, and DeFi's openness makes the risks fast and final. There is no fraud department, no reversal, no insurer of last resort. Four categories cover almost every disaster.

**Smart-contract bugs.** The code is the institution, so a flaw in the code is a flaw in the institution. A single mistaken line — a missing access check, a reentrancy hole (where a contract calls out to another that calls back in before the first finishes updating its books), a botched upgrade — can let an attacker drain the entire TVL in one transaction. Audits help but do not guarantee safety; many of the largest hacks hit audited code. Once funds leave, they are gone: pseudonymous, often laundered through mixers within minutes.

**Oracle manipulation.** A lending protocol decides liquidations from a price it reads off an oracle. If that oracle takes its price from a market the attacker can move — say a thin on-chain pool — then a flash loan can momentarily distort the price, trick the protocol into thinking some collateral is worth far more (or some debt far less) than it is, and extract the difference. To make the mechanism concrete: suppose a protocol prices a small token at whatever a thin \$200,000 pool says, and lets you borrow against it. An attacker flash-borrows millions, dumps it into that pool to spike the token's quoted price from \$1 to \$10, posts a tiny amount of the now-"\$10" token as collateral, borrows out far more real value than the token is truly worth, and repays the flash loan — all in one transaction, leaving the protocol holding worthless collateral against a real debt. That is, in outline, exactly the Mango and bZx playbooks. The fix is robust oracles (time-weighted averages that a single transaction cannot move, multiple independent sources like Chainlink), but every protocol that cuts that corner is a target.

**Governance attacks.** If protocol parameters are set by token-holder vote, then concentrating enough tokens — sometimes via a flash loan that borrows the governance token, votes, and returns it in one transaction — lets an attacker pass a malicious proposal: drain the treasury, change a critical parameter, mint themselves tokens. Defenses include timelocks (a delay between a vote passing and it taking effect, so the community can react) and quorum rules, but governance remains a live attack surface.

**De-pegs and collateral cascades.** Stablecoins and liquid-staking tokens are supposed to hold a fixed value; when they slip, every protocol that treated them as worth \$1 is suddenly undercollateralized at once. A de-peg can trigger mass liquidations, which dump collateral, which pushes prices down further, which triggers more liquidations — a reflexive spiral that the Terra collapse turned into a \$40 billion crater.

These are not separate risks so much as a chain reaction waiting for a trigger, and composability is the wiring that carries the spark from one protocol to the next.

## DeFi versus traditional banks

It helps to lay the systems side by side, because the differences are not cosmetic — they are a transfer of *who bears which risk*.

| Dimension | Traditional bank | DeFi protocol |
| --- | --- | --- |
| Who holds your money | The bank (custodian) | You (self-custody) or a contract |
| Who you trust | The institution + regulator | The code + the chain |
| Access | KYC, credit check, hours | Anyone, any time, with a wallet |
| A loan requires | Your creditworthiness | Collateral worth more than the loan |
| If it fails | Deposit insurance, courts, bailouts | Nothing — funds may be unrecoverable |
| Reversibility | Chargebacks, fraud reversal | Irreversible once confirmed |
| Transparency | Private internal ledger | Fully public on-chain |
| Speed of failure | Days (a bank run unfolds) | Seconds (a hack drains in one block) |

The honest summary: DeFi is more open, more transparent, faster, and permissionless — and it removes exactly the safety nets that make traditional banking forgiving of error. A bank that fails has the FDIC, the central bank as lender of last resort, and a legal system behind it (the machinery surveyed in the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions)). A protocol that fails has a Discord server and a post-mortem. Neither system is strictly safer; they fail *differently*, and they relocate risk to different parties.

There is a subtler asymmetry in *recourse*. When a bank wrongs you — a fraudulent charge, an error, an insolvency — you have layers of recourse: the bank's own dispute process, a regulator to complain to, a court to sue in, and ultimately a government that may backstop deposits to prevent a panic. The cost of those safety nets is everything DeFi dislikes: gatekeeping, surveillance (the credit checks and identity verification of "know your customer" rules), slowness, and the implicit subsidy that bailouts hand to those who took the risk. DeFi flips the bargain. It gives you full control and full transparency, and in exchange it gives you full responsibility: there is no one to call, and "the code did what it said" is a complete and final answer even when the code did something catastrophic. Whether that trade is liberating or terrifying depends entirely on who you are and where you live — which is precisely why DeFi has found its strongest adoption among people whom the traditional system serves worst.

## Common misconceptions

**"DeFi is decentralized, so no one controls it."** Mostly aspirational. Many protocols have admin keys, upgradeable contracts, or governance tokens concentrated among founders and venture funds. A multisig of a few people can sometimes pause or upgrade a contract. "Decentralized" is a spectrum, and the marketing usually overstates where a given protocol sits on it. Always ask: who holds the keys, and what can they change?

**"Overcollateralized lending is pointless — why lock \$150 to borrow \$100?"** Because the borrower usually wants leverage or liquidity *without selling*. If you believe ETH will rise, you can lock ETH, borrow stablecoins against it, and spend the stablecoins while keeping your ETH exposure — and repay later to reclaim the ETH. It is a way to get cash out of an asset you do not want to sell, or to loop into leverage, not a substitute for an unsecured consumer loan.

**"A high APY means a great opportunity."** A sustainable yield comes from real economic activity — trading fees, borrower interest. A 1,000% APY almost always comes from a protocol printing its own token to bribe depositors, which dilutes holders and collapses when incentives stop, or from an outright Ponzi paying early entrants with later entrants' money. The number alone tells you nothing; the *source* of the yield is everything.

**"Stablecoins are all the same — they hold \$1."** They hold \$1 by very different mechanisms, with very different failure modes. USDC is backed by dollars and Treasuries at a regulated company. DAI is backed by overcollateralized crypto and real-world assets in smart contracts. Terra's UST was "backed" by an algorithm and a sister token — and that one went to zero. The dollar sign hides enormous variation in what actually stands behind the peg.

**"My funds are safe because the contract was audited."** An audit is a snapshot review, not a guarantee. Auditors miss bugs; code gets upgraded after the audit; the economic design can be exploitable even when the code is technically correct (an oracle that *works as written* can still be manipulated). Many nine-figure hacks hit audited protocols. An audit lowers risk; it never removes it.

**"Flash loans are a hacking tool that should be banned."** Flash loans do not *create* vulnerabilities — they *reveal* them. An attack that succeeds with a flash loan would also succeed for anyone with enough capital; the flash loan just removes the capital requirement, so the protocol was always exploitable. Banning flash loans would hide the symptom and leave the disease. The fix is robust oracles and sound economic design, not removing the instrument.

## How it shows up in real markets

DeFi's history is best read through its crises and its booms — each one taught the category something it had assumed away.

**MakerDAO's Black Thursday (March 12–13, 2020).** As COVID panic crashed every market, ETH fell about 50% in a day. Maker's vaults flooded into liquidation, but Ethereum was so congested that gas fees spiked and price feeds lagged. In the chaos, some collateral auctions received bids of essentially **\$0** — liquidators won ETH for nothing because no one else could get a transaction through. Maker was left with roughly **\$4 million** of bad debt (DAI that existed without collateral behind it). The protocol covered the shortfall by auctioning newly minted MKR governance tokens, diluting holders to recapitalize. The lesson: liquidation systems that work in calm markets can break exactly when you need them, and chain congestion is a systemic risk, not a footnote.

**The bZx flash-loan attacks (February 2020).** Among the first flash-loan exploits, the bZx lending protocol was drained in two attacks (about \$350,000 and \$650,000) where the attacker used a flash loan to manipulate the price oracle bZx relied on, opening a position the protocol mispriced and pocketing the difference — all in single transactions. Small by later standards, but they announced the entire genre of attack: flash loan plus weak oracle equals theft.

**DeFi Summer (mid-2020).** The boom that named the category. Compound launched its COMP governance token and distributed it to users for borrowing and lending — **liquidity mining**. Suddenly you could earn the token on top of interest, and capital poured in chasing "yield farming." TVL across DeFi went from roughly \$1 billion to over \$15 billion in months and would later peak near \$180 billion in 2021 (approximate). It proved DeFi could attract real capital — and it also birthed the unsustainable-APY casino that produced countless rug pulls.

![DeFi from Black Thursday to today](/imgs/blogs/defi-protocols-uniswap-aave-makerdao-6.png)

**The Curve wars (2021 onward).** Curve, an AMM specialized in trading stablecoins against each other with minimal slippage, paid rewards in its CRV token, and crucially let CRV holders direct *where* those rewards flowed. So protocols began competing to accumulate CRV (and voting power) to steer rewards toward their own pools — a meta-game of bribes and token accumulation dubbed the "Curve wars," with platforms like Convex built entirely to win it. It is the clearest illustration that in DeFi, governance tokens are not just votes — they are control of cash flows, and control is worth fighting (and paying) for.

**The Mango Markets exploit (October 2022).** An attacker, Avraham Eisenberg, used a relatively modest amount of capital to ramp the price of the MNGO token on Mango's own oracle, used the inflated "collateral" to borrow out the protocol's entire treasury — about **\$117 million** — and then, extraordinarily, negotiated to return part of it while keeping a "bounty," arguing publicly it was a legal trading strategy. (He was later arrested and convicted.) It is the textbook oracle-manipulation case and a landmark in the legal question of whether draining a protocol "as designed" is theft.

**The Euler Finance hack (March 2023).** A lending protocol, Euler, lost about **\$197 million** to a flash-loan-assisted attack exploiting a flaw in its donation and liquidation logic — at the time one of the largest DeFi hacks ever. In a rare twist, the attacker returned nearly all the funds weeks later after negotiation. It underlined that even audited, well-regarded protocols carry tail risk in the tens of millions, and that on-chain transparency occasionally helps recover funds — but cannot be relied on.

**The Terra/UST collapse (May 2022).** Not a hack but a design failure, and the largest of all. Terra's UST was an *algorithmic* stablecoin: it held \$1 not with reserves but by letting holders always swap 1 UST for \$1 of its sister token LUNA. When confidence cracked and UST slipped below \$1, the swap minted ever more LUNA, crashing LUNA's price, which destroyed the only thing backing UST — a reflexive death spiral. Roughly **\$40 billion** of value evaporated in days, and the contagion bankrupted lenders and funds across crypto. The lesson DeFi keeps relearning: a peg backed by your own token is backed by nothing when you need it most.

**The DAO hack (June 2016) — the original.** The cautionary tale that predates all of these. "The DAO" was an early investment fund as a smart contract; a reentrancy bug let an attacker drain about \$60 million of ETH (a third of its funds). Ethereum's community ultimately chose to *fork* the chain to reverse it — splitting into Ethereum and Ethereum Classic — a decision still debated today as either a necessary rescue or a betrayal of "code is law." It established, at the very birth of programmable finance, that the gap between what code says and what people want it to do is where the hardest problems live.

## When this matters to you, and where to read next

You do not need to ever touch DeFi for it to matter to you. It now intermediates hundreds of billions of dollars, increasingly touches traditional finance through tokenized Treasuries and stablecoins, and is a live laboratory for what "money as software" looks like — which means its successes and its blow-ups preview questions every financial system will face as it digitizes.

If you do interact with it, the foundations above are the survival kit. Before supplying to a pool, ask what the yield's *source* is and price in impermanent loss. Before borrowing, know your exact liquidation price and leave a real buffer. Before trusting a stablecoin, ask what actually backs the peg. Before trusting a protocol, ask who holds the keys and what an audit did and did not cover. And treat every "this can't fail" claim as the precise place the next failure will start.

The deeper point is the one in the TL;DR: DeFi does not abolish the risks of banking and trading — it *relocates* them. It moves custody from an institution to you, trust from a regulator to code, and recourse from a courtroom to a Discord. For some people, in some places, that is a strictly better deal — open, fast, and theirs. For others it removes safety nets they did not know they were relying on. The technology is genuinely elegant; the constant-product pool, the overcollateralized vault, the atomic flash loan are beautiful pieces of engineering. The risk is the same elegance that lets four protocols compose into one system also lets one bug cascade through all four. Understanding both halves — that is the whole point.

For the layers beneath and beside this story: [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) explains the settlement layer DeFi is built on; [stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) goes deep on the dollar-backed coins DAI competes with; the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) and [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) show the traditional system DeFi is trying to route around — and reveal how much of "decentralized finance" is, mechanically, the same old banking with the institution swapped for code.

(All market figures — TVL, prices, hack sizes, yields — are approximate and as of early 2026; on-chain numbers move constantly, and historical figures are drawn from widely reported post-mortems.)
