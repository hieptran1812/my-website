---
title: "Analyzing Lending Protocols and Liquidations: Utilization, Health Factors, and Cascades"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "On-chain lending is fully transparent — every loan, its collateral, and its liquidation price are public. Learn to read utilization, health factors, the liquidation mechanic, and how a cascade forms."
tags: ["onchain", "crypto", "defi", "lending", "liquidations", "aave", "compound", "health-factor", "utilization", "risk", "dune", "ethereum"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — An on-chain loan is a glass box: you can see the collateral, the debt, and the exact price at which the position gets liquidated, for every borrower on the chain. That transparency is both a risk gauge and a tradeable signal.
>
> - **The concepts**: a loan is *overcollateralized* (you post more than you borrow); the **health factor** measures how far you are from being liquidated (below 1 = liquidatable); **utilization** (borrowed ÷ supplied) sets the interest rate via a "kink" curve.
> - **How to read it**: pull a market's utilization and rates on Aave or DeFiLlama; look up a whale's position and its **liquidation price**; build an aggregate **liquidation map** on Dune to see how much collateral liquidates at −10%, −20%.
> - **What you do with it**: treat dense liquidation clusters as magnets and stop-hunt zones, size down before a cascade, and never run a health factor so thin that one bad candle wipes you.
> - **The number to remember**: liquidators earn a **5–8% bonus** on the debt they repay — that bonus is the borrower's loss, and it is exactly why forced selling clusters and accelerates.

On 18 November 2022, two weeks after FTX imploded, a single address parked roughly **\$1.6 billion of borrowed and lent positions** on Aave and tried to push the price of a low-liquidity token, CRV, in a way that would saddle the protocol with bad debt. It half-worked: the attacker's own position got liquidated, Aave was left holding about **\$1.6 million of bad debt** that the borrower's collateral could not cover, and for a few hours everyone watching could see the whole thing unfold *transaction by transaction*. Nobody needed a leak or an insider. The collateral, the debt, the falling health factor, the keeper bots racing to liquidate — all of it was on the public ledger in real time.

That is the strange superpower of on-chain lending. In traditional finance, a bank's loan book is a secret; you find out a borrower was overlevered only after they blow up. On Aave, Compound, Morpho and their kin, **every loan is public, and every loan tells you its own liquidation price**. You can stand over the shoulder of the largest borrowers in the market and watch exactly how close they are to the edge — and, crucially, the market can *see the same thing* and push price toward those edges on purpose.

This post teaches you to read that book. We will build from zero: what overcollateralized lending even is, how utilization sets rates, what the health factor measures, and what a liquidation actually does to your money. Then we go deep on the parts that matter for a trader, a risk manager, or an analyst: how to read a whale's liquidation price, how to map aggregate at-risk collateral, and how a **liquidation cascade** turns one price drop into a self-reinforcing spiral of forced selling.

![An on-chain loan with collateral, debt, health factor and liquidation price feeding keeper bots](/imgs/blogs/analyzing-lending-and-liquidations-1.png)

## Foundations: how on-chain lending works

Start with the everyday version. You want to borrow money but you do not want to sell the asset you already own — maybe you hold a house, or stock, or in this case ETH, and you would rather keep it than dump it. So you pledge it as **collateral** and borrow against it. A pawnshop does this with your watch; a brokerage does it with a *margin loan* against your stock; a bank does it with a mortgage against your house. On-chain lending is the same idea, automated by a smart contract, with no loan officer and no credit check.

The defining feature of on-chain lending is that it is **overcollateralized**. There is no way to chase a borrower in court — addresses are pseudonymous and global — so the protocol cannot extend a loan it could not instantly recover. Instead it demands that you post *more* value than you borrow, and it keeps the right to seize your collateral the moment your loan gets too close to its value. If you want to read the protocol mechanics from first principles, the sibling post on [DeFi protocols (Uniswap, Aave, MakerDAO)](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) walks through the contract design; here we focus on *reading* the loans those contracts hold.

A few terms you need before anything else makes sense. Define them once and the rest of the post is just arithmetic.

- **Collateral** — the asset you deposit and pledge. On Aave this earns a small yield while it sits there, because other people can borrow against the pool of collateral.
- **Debt** — the amount you have borrowed, usually a stablecoin like USDC, but it can be any asset the protocol supports.
- **Loan-to-value (LTV)** — your debt divided by your collateral, as a percent. Borrow \$600k against \$1M of ETH and your LTV is 60%.
- **Collateral factor / max LTV** — the *most* you are allowed to borrow against a given asset, set by the protocol. ETH might have a max LTV of 80%, meaning \$1M of ETH lets you borrow up to \$800k.
- **Liquidation threshold** — a slightly higher line, e.g. 82.5%, at which the protocol declares you *liquidatable*. The gap between max LTV (where you can borrow) and the liquidation threshold (where you get seized) is a deliberate safety buffer.
- **Liquidation** — when your position crosses the threshold, anyone may repay part of your debt and take an equivalent slice of your collateral *plus a bonus*. More on this below.
- **Keeper / liquidator bot** — an automated program that watches every position and fires a liquidation the instant one becomes profitable. There is no human in the loop; this is a competitive, sub-second business.

### The health factor: how close you are to the edge

The single most important number on an on-chain loan is the **health factor (HF)**. It compresses everything above into one figure. The formula Aave uses is:

```
health factor = (collateral value × liquidation threshold) / total debt
```

When HF is above 1, you are safe. When HF drops to 1 or below, you are liquidatable. The beauty of it is that it folds price, collateral, debt, and the protocol's risk parameters into a single dimensionless number you can watch tick down in real time. A health factor of 2.0 means your collateral could lose half its (threshold-adjusted) value before you are in trouble; a health factor of 1.05 means you are one bad candle away from being seized.

Because the formula is public and deterministic, you can run it *backwards* to find the exact price at which any position gets liquidated — the **liquidation price**. That is the number the whole market can see and, as we will discover, sometimes push toward.

#### Worked example: a \$1M ETH loan and its liquidation price

Say you deposit ETH worth **\$1,000,000** when ETH is at **\$3,000** (so you hold about 333.3 ETH). Aave's ETH liquidation threshold is **82.5%**. You borrow **\$600,000** of USDC. Your starting health factor is:

```
HF = (1,000,000 × 0.825) / 600,000 = 825,000 / 600,000 = 1.375
```

A health factor of **1.375** is comfortable but not bulletproof. Now solve for the liquidation price. You get liquidated when collateral value × 0.825 = debt, i.e. when your collateral falls to \$600,000 / 0.825 = **\$727,273**. Since you hold 333.3 ETH, that collateral value corresponds to an ETH price of \$727,273 / 333.3 = **\$2,182**. So this loan liquidates at **ETH = \$2,182**, a **27% drop** from the \$3,000 entry. That \$2,182 is not a guess or a private risk-model output — it is a public number anyone can compute from the on-chain position, and it becomes a line on the chart that the next sections will teach you to read.

The lesson: borrowing 60% of a \$1M position does not feel reckless, but it puts your liquidation price only 27% below spot. In crypto, a 27% drawdown is a routine month. The first thing an analyst checks on any leveraged on-chain position is not the size — it is the *distance to liquidation*.

It is worth dwelling on *why* the formula uses a liquidation threshold higher than the borrow cap, because the gap is doing real work. When you open the loan, the protocol limits how much you may borrow to the **max LTV** (say 80% of ETH). But it does not liquidate you the instant your LTV touches 80% — it waits until the slightly higher **liquidation threshold** (82.5%). That 2.5-percentage-point gap is a buffer that absorbs the small price moves between blocks and gives keepers a window to act before the position goes underwater. Without it, every position that borrowed the maximum would be liquidatable on the very next tick. The buffer is also why your *practical* safe leverage is lower than the max: borrowing right up to the cap means you are liquidatable after a move of just a couple of percent. Experienced borrowers leave a wide margin — they target a health factor of 2 or more, not 1.05 — precisely because the threshold is a hard, automated line with no grace period and no phone call.

One more subtlety the health factor hides: it moves when *either* side of the loan moves. Your HF falls when collateral price drops, but it *also* falls as your debt grows from accrued interest. A borrower who never touches their position can still be liquidated months later purely because interest compounded the debt upward while collateral stayed flat. This is why a position you opened safely is not safe forever — the health factor is a live number, and on a high-utilization market the interest leg can erode it surprisingly fast even in a sideways market.

### Utilization and the interest rate

Where does the interest rate come from? Not a central bank — from supply and demand inside each individual pool, expressed through **utilization**.

A lending pool has two sides. Suppliers deposit assets (and earn yield); borrowers take assets out (and pay interest). Utilization is simply how much of the supplied pool is currently borrowed:

```
utilization = total borrowed / total supplied
```

If \$100M is supplied and \$70M is borrowed, utilization is 70%. The protocol uses utilization to set the borrow rate through a piecewise "**kink**" model: rates rise gently while utilization is below an optimal point, then spike steeply above it. The steep part is a deliberate pressure valve. If too much of the pool is borrowed, suppliers cannot withdraw (there is no idle cash left), so the protocol jacks the rate up to *bribe* borrowers to repay and suppliers to deposit, dragging utilization back toward the comfortable zone. We will read this curve in detail in the next section.

### Why people borrow against crypto at all

A reasonable beginner asks: if I have to post *more* than I borrow, what is the point? Three reasons explain almost every on-chain loan, and knowing which one a borrower is pursuing tells you how they will behave under stress.

The first is **leverage**. Deposit ETH, borrow a stablecoin, buy more ETH, deposit that, borrow again — each loop multiplies your exposure to ETH. This is the classic "looping" or "folding" trade, and it is the source of most cascade risk, because looped positions are thin by construction and all liquidate in the same direction at once. The second is **liquidity without selling**: you hold an asset you do not want to sell (for tax reasons, conviction, or because selling would trigger a taxable event or signal weakness), so you borrow stablecoins against it to fund spending or other trades while keeping the upside. The third is **yield arbitrage**: borrow an asset at a low rate and deploy it somewhere paying more — including the looping version where you supply a staking derivative and borrow against it, which the liquid-staking sibling post explores in depth.

The behavioral read matters. A *leverage looper* will be liquidated en masse in a downturn and is the raw fuel of a cascade. A *liquidity borrower* who took a small, conservative loan against a large holding rarely gets liquidated and is not a risk signal. When you see a large position, ask which kind it is: a 60%-LTV loop is fragile; a 25%-LTV liquidity loan with a health factor of 3 is not. The size alone tells you nothing; the *purpose and the buffer* tell you everything.

That is the whole foundation: post collateral, borrow less than it is worth, watch your health factor, and know that utilization is quietly setting the price of your loan. Everything from here is reading these signals on real markets and understanding how they break.

## Reading utilization and the rate curve

The first thing to pull on any lending market is its **utilization and rate curve**, because it tells you three things at once: how much room suppliers have to withdraw, where the rate is going if demand keeps rising, and whether the market is under stress.

The canonical shape is the **kink model** (Aave calls its parameters the "interest rate strategy"; Compound calls it the "jump rate model" — same idea). Below the optimal utilization point, say 90% for a stablecoin market, the borrow rate climbs slowly. Above it, the rate slope steepens dramatically, so a few extra percent of utilization can take the borrow rate from 4% to 40%.

![Utilization sets the borrow rate with a gentle slope below the kink and a steep slope above it](/imgs/blogs/analyzing-lending-and-liquidations-3.png)

Read the curve like a fuel gauge with a cliff at the end. When utilization sits comfortably below the kink, the market is calm: suppliers can withdraw freely and rates are cheap. When utilization is pinned near 100%, two things are true and both are signals. First, suppliers may be *unable to withdraw* — there is no idle liquidity left in the pool, so a depositor trying to pull out has to wait for borrowers to repay or new deposits to arrive. Second, the rate has spiked, which is the protocol screaming at borrowers to repay and at suppliers to deposit.

The shape of the curve is a deliberate policy choice by the protocol, and it is worth understanding what the parameters do because they are what you are actually reading. The **base rate** is where the curve starts at zero utilization — usually 0% for a stablecoin, so an empty pool costs nothing to borrow from. The **first slope** governs how fast the rate rises up to the kink; a gentle slope keeps borrowing cheap through the normal operating range. The **optimal utilization** (the kink point) is where the protocol *wants* the pool to sit — high enough that suppliers earn a decent yield, low enough that there is always some liquidity for withdrawals. And the **second slope** is the punishing one above the kink, often steep enough to take the rate from single digits to 50% or more in the last sliver of utilization. That steepness is intentional cruelty: it makes borrowing so expensive at the top that borrowers are economically forced to repay, which is the only mechanism the protocol has to keep the pool from being fully drained. There is no central banker deciding rates here — the curve *is* the policy, and it runs automatically every block.

Different markets get different curves. A blue-chip stablecoin market might have a kink at 90% and a mild second slope, because its supply and demand are stable and deep. A volatile-collateral market or a thin token might have a kink at 45% and a brutal second slope, because the protocol wants to discourage heavy borrowing of an asset it cannot safely liquidate. When you read a market, the curve's parameters tell you how the protocol's risk team rates that asset: a low kink and a steep slope is the protocol saying "we do not trust this market to stay liquid under stress."

#### Worked example: lending into a utilization spike

Suppose a USDC market has **\$200,000,000** supplied and **\$150,000,000** borrowed, so utilization is 75% and the supply rate is a sleepy **3%** APR. A wave of leverage demand pushes borrowing to **\$190,000,000** — utilization is now 95%, just above the 90% kink. On the illustrative curve above, the borrow rate jumps from roughly 4% to perhaps **25%** APR, and because suppliers earn (borrow rate × utilization × (1 − reserve factor)), the supply rate leaps too. A supplier who deposits **\$1,000,000** of fresh USDC into that 95%-utilization pool might suddenly earn on the order of **\$200,000** annualized instead of \$30,000 — until utilization normalizes. That is a real, if fleeting, edge: utilization spikes are short windows where suppliers get paid unusually well to provide the liquidity the market is begging for.

The trap on the other side: if you are the *borrower* when utilization pins at 100%, your interest cost can quintuple overnight, quietly eroding your collateral cushion and dragging your health factor down even if price never moves. And if you are a *supplier* who needs your money back, a pinned pool means you are stuck until utilization eases. Always check whether a high yield is the kink model rewarding you for scarce liquidity (a transient bonus) or a sign the pool is jammed (a liquidity trap). The difference is whether utilization is climbing into the kink or stuck above it.

### Where to read it

On **Aave's** own app, each market shows utilization, the supply and borrow APR, and the total supplied and borrowed in dollars. **DeFiLlama** (under "Yields" and "Lending") aggregates the same across protocols and chains so you can compare the USDC market on Aave Ethereum against Aave Arbitrum, Compound, Morpho and Spark side by side. **Dune** dashboards let you chart utilization over time, which is where you spot the slow creep toward a kink before it spikes.

### Reading the rate as a stress gauge over time

A single snapshot of utilization is useful; the *time series* is better. What you want to watch is the trajectory relative to the kink. A market that has spent weeks creeping from 75% to 88% utilization is telling you demand is building and the kink is approaching — the next wave of borrowing tips it over and rates jump. A market that *repeatedly* slams into 99% and bounces back is one where suppliers keep getting trapped and rushing back in once they can withdraw; that oscillation is itself a sign of structural undersupply.

There is also a clean way the rate curve interacts with liquidations that beginners miss. When utilization is high and the borrow rate is, say, 40% APR, the debt of every borrower in that pool is growing at 40% a year — over 3% a month. For a borrower already near the threshold, that compounding debt is dragging their health factor down even in a flat market, which can *trigger* liquidations that then dump collateral, which lowers the price, which liquidates more positions. A high-rate environment is quietly cascade-prone for a reason that has nothing to do with a price catalyst: the interest leg alone is enough to push thin positions underwater over time.

### Morpho and the move toward isolated, peer-matched lending

A quick note on the newer architecture, because it changes how you read the risk. Classic Aave and Compound use big shared pools: everyone's collateral and everyone's debt sit together, so a bad-debt event in one corner can socialize losses across all suppliers. **Morpho** (and isolated-market designs generally) breaks lending into many small, isolated markets — one collateral, one borrow asset, its own parameters — so a blowup in a risky market does not bleed into a safe one. For an analyst this means you must read risk *market by market* rather than protocol-wide: a protocol's headline TVL can look healthy while one isolated market inside it is a powder keg of looped leverage on thin collateral. Isolation contains contagion, but it also means the at-risk-collateral map you build has to be per-market, not aggregated blindly.

## Health factors at the position level: the whales are public

Here is the part that has no equivalent in traditional finance. On-chain, you can look up *any individual borrower's position* — their collateral, their debt, their health factor, and the price at which they get liquidated. The big positions matter most, and the biggest ones (the "whales") are watched obsessively.

This is not theoretical. There are public dashboards (on Dune, DeBank, DeFiLlama, and dedicated risk tools like the now-defunct but widely-cloned liquidation trackers) that list the largest borrowers on Aave and Compound, sorted by debt size, each annotated with its health factor and liquidation price. When a single address has a nine-figure position with a health factor of 1.1, that is front-page news in crypto risk circles, because everyone can see that a modest price drop will trigger a very large forced sale.

#### Worked example: a whale's known \$2M liquidation level

Say an address posts **\$30,000,000** of ETH collateral and borrows **\$18,000,000** of stablecoins, with a liquidation threshold of 82.5%. Its health factor is (30,000,000 × 0.825) / 18,000,000 = **1.375**, and it liquidates when ETH falls about 27%, at a specific dollar price — call it **\$2,182** as in our earlier example. Now suppose that within the whale's collateral there is a particular tranche — a **\$2,000,000** slice of a thinner alt-collateral — whose liquidation would dump onto a shallow order book. The market can see that **\$2M of forced selling sits waiting at ETH ≈ \$2,182**. A well-capitalized trader can ask: if I push price down into that level, the whale's liquidation fires, the keeper bot dumps \$2M of collateral, price gaps lower, and I cover my short into the air pocket. The liquidation level becomes a *target*. This is the on-chain version of a stop-hunt, except the stops are public and the size is known to the dollar.

That is the double edge of transparency. As a borrower, your liquidation price is a billboard advertising exactly where you are vulnerable. As a trader, other people's billboards are a map of where forced sellers live. We will turn that map into a strategy in the "liquidation levels as a map" section below — but first, what actually *happens* at the moment of liquidation.

### Watching a whale defend a position in real time

Because positions are public, you can also watch borrowers *react*. When ETH starts dropping toward a whale's liquidation price, you frequently see the address top up: a fresh deposit of collateral, or a repayment of debt, both of which lift the health factor and push the liquidation price lower. These defensive transactions are themselves a signal. A whale that keeps adding collateral as price falls is fighting to hold a conviction position — and is also concentrating more and more of their net worth into a single trade, which makes their eventual capitulation (if it comes) far larger. A whale that does *nothing* as their health factor slides toward 1 is either asleep, out of dry powder, or has decided to let the position go.

The famous on-chain risk stories are mostly stories of whales who ran out of room to defend. The 2022 CRV-on-Aave episode in our hook was one address whose position grew too large to defend once the market turned against it; the Mango Markets exploit was an attacker who *built* a position specifically so its liquidation would do damage. In both cases, observers could watch the health factor in real time and see the danger building blocks before the resolution. The actionable version of this for an ordinary trader: pick the two or three largest, thinnest positions in a market you trade, set an alert on their health factor, and you have a free early-warning system for forced selling — one that no traditional market gives you.

## The liquidation mechanic: keepers, bonuses, partial vs full

When a position's health factor drops to 1 or below, it becomes liquidatable, and a competitive swarm of bots — **keepers** — pounces. The mechanic is the same across major protocols, with parameter differences.

![Liquidation pipeline from price drop to keeper repaying debt and seizing collateral at a discount](/imgs/blogs/analyzing-lending-and-liquidations-2.png)

A liquidation is a trade the protocol offers to anyone: *repay some of this borrower's debt, and in exchange take an equal value of their collateral plus a bonus.* The **liquidation bonus** (Aave) or **liquidation incentive** (Compound) is typically **5–8%**, sometimes higher for riskier collateral. That bonus is what pays the keeper for the gas, the capital, and the speed required to win the race. It is also, dollar for dollar, *the borrower's loss* — the borrower's collateral is sold at a discount to its market value, and the discount goes to the liquidator.

Most modern protocols use **partial liquidation**: a keeper can repay only up to a fraction of the debt (Aave's "close factor" is typically 50%) in a single liquidation, enough to drag the health factor back above 1 without wiping the whole position. If price keeps falling, the position becomes liquidatable again and another partial liquidation fires. Older or stressed designs sometimes allow *full* liquidation, and when collateral is volatile and thin, partial liquidations can chain rapidly into something that looks like a full one.

#### Worked example: a liquidator earning the bonus on a \$200k liquidation

A borrower's position drops below HF = 1. Their debt is **\$400,000** and the close factor is 50%, so a keeper may repay up to **\$200,000**. The liquidation bonus on this collateral is **8%**. The keeper repays **\$200,000** of the borrower's stablecoin debt and in return receives collateral worth **\$200,000 × 1.08 = \$216,000**. After paying perhaps **\$3,000** in gas and priority fees to win the race, the keeper nets about **\$13,000** of profit on the trade — a clean ~6.5% on the capital deployed, captured in one block. Repeat that across hundreds of positions during a sell-off and you see why liquidations are a multi-million-dollar bot industry, and why the bonus is so reliably paid: there is always a keeper willing to do it for the spread. The borrower, meanwhile, just paid \$16,000 to have their collateral force-sold.

The keeper economics are why liquidations are *fast and reliable* in normal conditions and *dangerous* in abnormal ones. In a calm market, the 5–8% bonus is more than enough to make liquidating profitable, so positions get cleared promptly and the protocol stays solvent. But the bonus is only worth taking if the keeper can *re-sell the seized collateral* for close to the price the protocol assumed. When collateral is thin or price is in free fall, the keeper might not be able to dump the collateral without crashing it further, the bonus stops covering the risk, and liquidations *stall* — which is precisely how bad debt is born.

Keepers themselves are a fascinating, brutal corner of on-chain markets. Winning a liquidation is a race: the moment a position crosses HF = 1, dozens of bots see it in the mempool and compete to land the liquidation transaction in the next block. They bid up gas (and, on Ethereum, priority fees and MEV bundles) to get their transaction ordered first, because only the winner collects the bonus. This is a form of MEV (maximal extractable value) — value extracted purely from transaction ordering — and it overlaps heavily with the sandwich and front-running world. The competition is so fierce that in calm markets the *effective* bonus a keeper keeps after gas wars is far below the headline 5–8%; the protocol sets the bonus high enough that even after the bots compete it away, the position still clears. The borrower pays the full headline discount regardless of how the keepers split it among themselves.

There is also a smarter, capital-light version of liquidation that matters for understanding cascades: the **flash-loan liquidation**. A keeper with no capital of their own can flash-borrow the stablecoins needed to repay the debt, use them to liquidate the position, sell the seized collateral, repay the flash loan, and pocket the bonus — all in a single atomic transaction. This means liquidation capacity is effectively *unlimited* in normal conditions; a keeper does not need to hold \$200,000 to do a \$200,000 liquidation. The catch, again, is the *sell* step: the flash-loan liquidation only works if the seized collateral can be sold for enough to repay the flash loan plus capture the bonus. When the collateral is crashing, even infinite borrowing capacity does not help, because the sell leg fails. So the very mechanism that makes liquidations frictionless in calm markets evaporates exactly when a cascade is underway.

### How bad debt forms

If collateral falls *faster* than keepers can liquidate it — a sharp gap-down, a thin alt-collateral with no buyers, or both — a position can blow through HF = 1, through the bonus buffer, and into territory where the collateral is worth *less* than the debt. Now liquidating is unprofitable (the keeper would pay \$X to seize collateral worth less than \$X), so the bots stand aside, and the protocol is left holding a loan it cannot fully recover. That shortfall is **bad debt**, and it is socialized — eaten by the protocol's reserves or, in the worst case, by suppliers. The November 2022 CRV episode on Aave left roughly **\$1.6 million** of exactly this kind of bad debt. It is the on-chain analog of a margin call that can't be filled.

#### Worked example: when a liquidation goes underwater into bad debt

A position holds **\$1,000,000** of a thin alt-token as collateral against **\$820,000** of stablecoin debt, right at the 82.5% threshold, HF = 1.0. The token gaps down 15% in one block — there were no buyers between \$1.00 and \$0.85. The collateral is now worth **\$850,000**. A keeper who liquidates must repay debt and take collateral *plus the 8% bonus*, so to clear the full \$820,000 of debt they would need to seize \$820,000 × 1.08 = **\$885,600** of collateral — but only **\$850,000** exists. There is no profitable liquidation: any keeper who steps in pays \$820,000 to receive at most \$850,000 of a token they then have to sell into a market with no bids, likely netting a *loss*. So the keepers wait. If the token keeps sliding to \$0.70, the collateral is worth **\$700,000** against \$820,000 of debt — the protocol now has **\$120,000** of bad debt that no one will ever repay, and it comes out of reserves or supplier funds. The lesson: bad debt is not a liquidation that was too slow; it is a liquidation that became *unprofitable*, which happens precisely when collateral gaps faster than the bonus buffer can absorb.

## Large-position risk: a single whale as a known trigger

Step back and combine two facts we have established: liquidation prices are public, and large positions force large sales. Put them together and you get the central risk read of on-chain lending — **a single large borrower near their liquidation price is a known trigger level that the whole market can see and act on.**

In traditional markets, a fund's margin level is private. The market might *suspect* a big player is offside, but it cannot prove it or know the exact price. On-chain, the suspicion is replaced by certainty: the position is right there, the liquidation price is computable, and the size of the forced sale is known to the dollar. This changes behavior in two ways.

First, **defensively**, an analyst monitors the top borrowers on each major market and treats a large position with a thin health factor as systemic risk. If one address holds \$200M of debt with HF = 1.08, a 7–8% adverse move triggers a \$200M-collateral liquidation event — and depending on what the collateral is, that can move the whole market. You do not need to know who the borrower is; you need to know the level and the size.

Second, **offensively** (and this is the dark side), a well-capitalized actor can *aim* at the level. This is the Mango Markets and CRV-on-Aave playbook: identify a large position whose liquidation will dump illiquid collateral, push the price toward the trigger, let the cascade do the selling, and profit from the move you induced. When the trigger price is public and the collateral is thin, the attack is almost telegraphed. The defense is also on-chain: you can watch the would-be attacker accumulate the position, watch the targeted whale's health factor, and get out of the way. For the price-manipulation half of this — how an attacker moves an oracle to *force* a liquidation that should not happen — see the companion post on [fake depth, spoofing and oracle attacks](/blog/trading/onchain/fake-depth-spoofing-and-oracle-attacks); lending protocols read price from oracles, and a manipulated oracle is the cleanest way to trigger a liquidation on demand. The oracle design itself is covered in [Chainlink and blockchain oracles](/blog/trading/crypto/chainlink-and-blockchain-oracles).

#### Worked example: a −20% ETH move liquidating \$500M of collateral into a cascade

Suppose the aggregate liquidation map (next section) shows that across all the major lending markets, a **−20%** move in ETH would push roughly **\$500,000,000** of collateral into liquidatable territory. ETH starts at **\$3,000**. A broad market sell-off drags it to **\$2,400** (−20%). Now \$500M of collateral must be sold by keepers. If the order books can absorb only, say, **\$150,000,000** of selling without slipping 5%, the remaining \$350M of forced sales pushes ETH down *further* — to \$2,250, then \$2,100 — into the *next* cluster of liquidation levels, which triggers *another* wave. Each \$100,000,000 of forced selling that the book cannot absorb is another leg down. This is the cascade: the price drop causes liquidations, the liquidations cause forced selling, and the forced selling causes the next price drop. The \$500M number was never going to liquidate all at once at a clean −20% — it liquidates in a *spiral*, and the spiral overshoots.

## Liquidation cascades: the reflexive spiral

A **cascade** is what happens when liquidations stop being independent events and start feeding each other. It is the single most important second-order effect in on-chain lending, and it is the reason a market can fall 30% in an afternoon when the "fundamental" news was only a 10% catalyst.

![Liquidation cascade compared to a calm market: forced selling drives price into the next cluster](/imgs/blogs/analyzing-lending-and-liquidations-4.png)

The mechanism is reflexive — the output (lower price) becomes the input (more liquidations). In a calm market, each leveraged position sits independently above its liquidation price; a sell here, a buy there, all absorbed by ordinary order flow. The risk is *latent*: the cluster of liquidation levels exists, everyone can see it, but price has not touched it yet. Then a catalyst pushes price into the first cluster. Keepers liquidate, the seized collateral is force-sold onto the order book, and that selling *is itself* the next downward push. If the next cluster of liquidation levels sits just below, price falls into it, more positions go underwater, and the loop tightens. The faster price falls, the more positions cross their thresholds simultaneously, and the more keepers are dumping collateral at once.

Two things determine whether a dip becomes a cascade. The first is **clustering**: if liquidation levels are spread thinly across a wide price range, each wave of forced selling is small and the book absorbs it. If they are bunched in a dense band — because many borrowers entered at similar prices with similar leverage — then crossing into the band triggers a wall of selling at once. The second is **collateral liquidity**: deep collateral (ETH, BTC) can absorb large forced sales with modest slippage, so cascades are damped; thin collateral (a small-cap token) gaps down on the first liquidation, instantly pushing every other position using it underwater. The worst cascades combine a dense cluster with thin collateral, which is why the most violent on-chain deleveraging events happen in alt-collateral markets, not in the deep ETH and BTC pools.

This is also where the relationship to perpetual-futures deleveraging matters. Lending-market cascades and perp-DEX liquidation cascades often fire *together* during a crash, because the same leveraged traders are offside in both places at once; the forced selling from one venue is the catalyst that tips the other. The dynamics of on-chain perp liquidations — funding, mark price, auto-deleveraging — are the natural companion to this post; once a perp-DEX deep-dive sibling is published it pairs directly with everything here, and the shared lesson is the same: **leverage plus public liquidation levels plus thin liquidity equals cascade risk.**

### Two real cascades worth studying

The cleanest way to internalize cascade mechanics is to look at documented episodes. The **Terra/LUNA collapse of May 2022** was the largest reflexive deleveraging crypto has seen: as the UST stablecoin lost its peg, the mint-and-burn mechanism printed enormous quantities of LUNA, the price collapsed, and every leveraged position and lending market touching the Terra ecosystem cascaded at once. The whole event — peg break, forced selling, contagion into adjacent protocols — is dissected in the [Terra/LUNA 2022 collapse](/blog/trading/crypto/terra-luna-2022-collapse) post; the lending-and-liquidation lens here is one slice of that larger reflexive failure. The point for our purposes: cascades are not exotic edge cases. The biggest single value-destruction events in crypto history are cascades.

The **Mango Markets exploit of October 2022** is the manipulation version. An attacker took a large position in a thinly traded token, then deliberately pushed its price up to inflate the value of their own collateral, borrowed against the inflated value, and walked away with roughly **\$117,000,000** before the price collapsed back. It is the mirror image of a forced-liquidation attack: instead of pushing price *down* to trigger someone else's liquidation, the attacker pushed price *up* to over-borrow against their own position. Either direction, the lesson is the same — when a protocol reads price from a manipulable source and lets you borrow against it, the liquidation and borrowing logic becomes a weapon. The exploit mechanics belong to the oracle-attack and DeFi-hack posts; the relevance here is that *both* the defender and the attacker are reading the same public position data, and the attacker is just acting on it first.

#### Worked example: from a 10% dip to a 30% cascade

ETH is at **\$3,000**. A dense cluster of leveraged longs entered near \$3,300 and liquidates around **\$2,700** (a −10% level), totalling **\$300,000,000** of collateral. A 10% dip to \$2,700 fires that cluster. Keepers force-sell \$300M; the order book absorbs maybe \$120M cleanly, and the remaining **\$180,000,000** of selling pushes ETH to **\$2,450** — into the *next* cluster at the −18% level holding another **\$250,000,000**. That fires too, force-selling collateral that pushes price to **\$2,100** (−30%). What started as a 10% news-driven dip became a 30% cascade, with two-thirds of the move manufactured entirely by forced liquidations selling into each other. None of it required new bad news — it required only that the price touch the first cluster. The intuition: a cascade is leverage unwinding at the speed of code, and the depth of the move is set by how the liquidation levels are stacked, not by the size of the original catalyst.

## How to read a lending market: a walkthrough

Theory is fine; here is the hands-on pass. The goal is to answer three questions about any lending market in fifteen minutes: *Is the pool stressed? Where are the big positions vulnerable? How much liquidates if price drops?* Use Aave's app, DeFiLlama, and a Dune dashboard together.

**Step 1 — Pull utilization and rates.** Open the market on the **Aave** app or on **DeFiLlama** (Lending). Read four numbers: total supplied (\$), total borrowed (\$), utilization (%), and the borrow/supply APR. Compute or confirm utilization = borrowed ÷ supplied. Ask: is utilization below the kink (calm, suppliers can exit) or pinned near 100% (stressed, suppliers stuck, rates spiking)? A USDC market at 78% utilization with a 5% borrow rate is healthy; the same market at 99% with a 40% borrow rate is jammed.

**Step 2 — Find the large borrowers and their health factors.** On **Dune** (search the protocol's risk dashboards) or **DeBank** (look up known whale addresses), list the largest positions by debt. For each, read collateral, debt, health factor, and liquidation price. You are hunting for the combination that matters: *large debt + thin health factor + thin or correlated collateral.* A \$5M position at HF = 2.5 is irrelevant; a \$200M position at HF = 1.08 is the whole market's risk concentrated in one address.

**Step 3 — Compute or read the liquidation price.** For any position, liquidation price (for a single-collateral, single-debt loan) is approximately the current price × (current debt ÷ (collateral value × liquidation threshold)). Or just read it off the dashboard. Mark these levels on your chart — they are where forced sellers live.

**Step 4 — Build the aggregate liquidation map.** This is the highest-value, lowest-effort analyst move. A Dune query (or a risk dashboard like the ones DeFi risk teams publish) sums, across all positions, how much collateral becomes liquidatable at each price level below spot.

![Aggregate at-risk collateral rising from tens of millions at minus 5 percent to over a billion at minus 30 percent](/imgs/blogs/analyzing-lending-and-liquidations-5.png)

The map above is the single most useful picture in on-chain lending risk. Read it as: "if price drops by X%, roughly \$Y of collateral gets force-sold." A market where −10% liquidates \$220M but −20% liquidates \$900M has a *dense band* in the −15% to −20% zone — that is exactly the kind of clustering that turns a dip into a cascade. A market where at-risk collateral rises smoothly and stays small is well-distributed and cascade-resistant. You are not predicting price; you are mapping where the forced sellers are stacked so you know which dips are dangerous.

**Step 5 — Cross-check the oracle.** Lending protocols liquidate based on the *oracle* price, not the price on any one exchange. If the collateral's oracle reads from a manipulable source, the real risk is not a fair price drop but an *induced* one — see the [oracle attacks post](/blog/trading/onchain/fake-depth-spoofing-and-oracle-attacks). Before trusting any liquidation level, know where the protocol gets its price.

Run those five steps and you have done what a professional on-chain risk desk does every morning: gauged pool stress, located the vulnerable whales, mapped the at-risk collateral, and flagged the oracle exposure. The same map that tells a risk manager "we are exposed to a cascade below \$2,700" tells a trader "there is a wall of forced sellers at \$2,700, plan accordingly."

## Liquidation levels as a map: support, resistance, and stop-hunts

The analyst's edge is to stop looking at the price chart alone and start overlaying the **liquidation map**. A naive technical trader draws support and resistance from candles and round numbers. An on-chain trader knows *where the forced sellers actually are* — and forced sellers behave very differently from voluntary ones.

![Naive price chart compared with an on-chain liquidation map showing whale clusters as magnets](/imgs/blogs/analyzing-lending-and-liquidations-6.png)

Two behaviors make liquidation clusters tradeable. First, they act as **magnets**. Market makers and predatory traders know that touching a dense liquidation band releases a burst of forced selling (a guaranteed seller), so price is often *drawn toward* the band — the liquidity is there to be harvested. This is the "stop-hunt" pattern: price wicks down to exactly the level where a big cluster sits, triggers the liquidations, and then snaps back once the forced selling is exhausted. On a naive chart it looks like a random spike; on the liquidation map it was the obvious target all along.

Second, once a cluster is *consumed*, that selling pressure is gone. A level that has already been liquidated through is no longer a forced-seller zone — the leverage there has been flushed. So the map is dynamic: you watch clusters build up (as new leverage enters) and get consumed (as price sweeps through them). A market that has just cascaded and flushed its clusters is, counterintuitively, *safer* in the short term than one with fat untouched clusters just below spot.

The practical read: when you see a dense liquidation band a few percent below spot, treat it as a likely target, not as solid support. Either stand aside through the band, or plan to *fade the wick* — buy the over-extended flush once the forced selling is spent — rather than getting stopped out by the very move you could see coming. This is the same liquidity-hunting logic that drives MEV and sandwich attacks at the transaction level; the difference is the scale and the visibility. On-chain, the forced sellers announce themselves in advance.

### What a borrower can actually do about it

If you are the one holding the loan, the defenses are simple and worth stating plainly, because most liquidations happen to people who understood the risk in theory but did nothing concrete about it. First, **run a wide health factor**. The single biggest determinant of whether you survive a normal drawdown is your starting buffer; an HF of 2.0 survives a 40%+ collateral drop, an HF of 1.1 does not survive a bad afternoon. Second, **match your collateral to your debt where possible** — borrowing a stablecoin against ETH means your liquidation price moves only with ETH; borrowing one volatile asset against another means *both* legs move and your health factor can swing on news that touches either side. Third, **watch the interest leg on high-utilization markets**, because a 40% borrow rate is quietly compounding your debt upward and dragging your HF down even in a flat market. Fourth, **set your own alerts below your liquidation price plus a margin**, so you have time to add collateral or repay before a keeper does it for you at an 8% discount.

There is also a structural defense the best borrowers use: **never be the thinnest large position in a market.** If your liquidation price is the highest (closest to spot) among the big positions, you are the first domino, the first forced seller, and the most attractive stop-hunt target. Keeping your liquidation price *below* the crowd's means someone else cascades first and you get a warning shot. Reading the same liquidation map an attacker reads is how a careful borrower stays off the front line.

## Common misconceptions

**"DeFi loans are safe because they're overcollateralized."** Overcollateralization protects the *protocol*, not *you*. If you borrow against volatile collateral, you can be liquidated at a 5–8% loss in a routine drawdown — the protocol stays solvent precisely *because* it took your collateral cheaply. And when collateral falls faster than keepers can act, overcollateralization fails at the system level too: that is bad debt, and it has happened repeatedly (the \$1.6M Aave CRV episode, among others).

**"A high supply APR means a good, safe yield."** A high supply rate usually means *high utilization*, which means the pool is near the kink and possibly jammed. The yield is the kink model paying you for scarce liquidity — and the flip side of scarce liquidity is that you may not be able to *withdraw* until utilization eases. Always check utilization before chasing a lending yield. A 20% USDC yield at 99% utilization is a liquidity trap wearing a yield's clothing.

**"My liquidation price is private — only I know it."** The opposite. Your collateral, debt, and liquidation price are all public, and dashboards specifically surface the largest, thinnest positions. If you are a whale running thin leverage, the market knows your liquidation price to the dollar and may push toward it. Privacy is not a feature of on-chain lending; transparency is the whole design.

**"Liquidations are an exchange problem, not a price-action problem."** Liquidations *are* price action. Forced selling from liquidations is real selling that moves real markets — in a cascade, most of the move can be manufactured by liquidations rather than by any fundamental flow. Ignoring the liquidation map means ignoring a large fraction of the actual order flow.

**"Liquidation keepers always step in, so positions always clear cleanly."** Keepers step in *only while it is profitable*. The 5–8% bonus has to cover the cost of re-selling the seized collateral. In a fast crash or with thin collateral, that bonus is not enough, keepers stand aside, and the position blows through into bad debt. Reliable liquidation is a property of *normal* markets; it is exactly when you most need it (a violent sell-off) that it can fail.

## The playbook: what to do with it

The signal → the read → the action → the invalidation, for a trader, a borrower, and a risk analyst.

**Signal: utilization pinned near 100%.**
- *Read*: pool is stressed; rates spiking; suppliers may be unable to withdraw.
- *Action (supplier)*: a transient spike is a chance to earn an unusually high yield — but size it as a short-term trade, not a parked position. *Action (borrower)*: your interest cost just jumped; repay or add collateral before the rate erodes your health factor.
- *Invalidation / false positive*: if utilization is climbing into the kink (demand-driven, transient), the high yield is real; if it is stuck above the kink for days, it is a liquidity trap — do not deposit money you might need back.

**Signal: a large borrower with a thin health factor near a public liquidation price.**
- *Read*: a known, sized forced-seller level the whole market can see.
- *Action (trader)*: mark the level; treat it as a magnet and a likely stop-hunt target, not as support. Either stand aside through it or plan to fade the post-flush wick. *Action (risk)*: flag the concentration; a single thin nine-figure position is systemic risk regardless of who holds it.
- *Invalidation*: if the whale adds collateral or repays, the level moves — re-read it. Liquidation prices are not static; they drift as the position changes.

**Signal: a dense band on the aggregate liquidation map just below spot.**
- *Read*: high cascade risk; crossing into the band releases a wall of forced selling.
- *Action*: size down before the band, keep dry powder, and expect overshoot if price touches it. Do not buy "support" inside a fat liquidation cluster. *Action (long-term buyer)*: a cascade flush is where the best entries appear — the forced selling overshoots fair value — but wait for the cluster to be *consumed* before stepping in.
- *Invalidation*: if the band has already been liquidated through, the leverage is flushed and the level is no longer a forced-seller zone. The map is dynamic; re-pull it after any large move.

**Signal: a manipulable oracle on the collateral.**
- *Read*: liquidations can be *induced* rather than driven by fair price; the real trigger is the oracle, not the market.
- *Action*: discount the apparent safety of the liquidation level; assume an attacker can push the oracle to force a liquidation. Avoid borrowing against, or supplying, collateral that prices off a thin manipulable pool. See the [oracle attacks](/blog/trading/onchain/fake-depth-spoofing-and-oracle-attacks) and [anatomy of a DeFi hack](/blog/trading/onchain/anatomy-of-a-defi-hack) posts for the exploit mechanics.
- *Invalidation*: a robust oracle (deep aggregate of many venues, with sanity bounds and delays) removes most of this risk — but verify the oracle source before trusting it.

![Lending-risk decision matrix mapping signals to healthy ranges, warning thresholds and actions](/imgs/blogs/analyzing-lending-and-liquidations-7.png)

The decision matrix above is the one-page version: four signals, each with a healthy range, a warning threshold, and a concrete action. Keep it next to the chart. The whole discipline of on-chain lending analysis reduces to checking those four rows on every market you touch — because the ledger is showing you, in advance and to the dollar, exactly where the leverage is and where it breaks.

The deeper point that runs through this entire series: **flow beats price, and on-chain you can see the flow before it hits the tape.** A liquidation map is forced flow you can read *before* it sells. That lead time is the edge. It is also the trap — the same map that helps you dodge a cascade is the map an attacker uses to *cause* one. Read it as both a defender and an opportunist, and never run leverage so thin that you become a level on someone else's map.

## Further reading & cross-links

- [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — how the lending contracts themselves are built, from first principles.
- [Fake depth, spoofing and oracle attacks](/blog/trading/onchain/fake-depth-spoofing-and-oracle-attacks) — how a manipulated price feed can *force* a liquidation that fair pricing would never have triggered.
- [Chainlink and blockchain oracles](/blog/trading/crypto/chainlink-and-blockchain-oracles) — where lending protocols get the price that drives every liquidation, and why oracle design is the real attack surface.
- [Anatomy of a DeFi hack](/blog/trading/onchain/anatomy-of-a-defi-hack) — oracle-manipulation and flash-loan liquidations as documented exploits.
- Companion reads in this series cover **on-chain perp DEXs and derivatives** (perp liquidations and funding, which cascade alongside lending liquidations), **reading DeFi TVL honestly** (where lending TVL sits in the bigger picture), and **liquid staking and restaking yield** (the collateral that increasingly backs these loans) — pair them with this post once they are live for the full leverage-and-liquidity stack.
