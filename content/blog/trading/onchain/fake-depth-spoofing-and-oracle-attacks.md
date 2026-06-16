---
title: "Fake Depth, Spoofed Liquidity, and Oracle Attacks: When the Price Itself Is the Lie"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A defender's guide to the manipulations where the liquidity or the price feed itself is fake — spoofed depth, oracle attacks, and the flash-loan version of both — and how to spot a fragile protocol before you deposit."
tags: ["onchain", "crypto", "oracle-manipulation", "flash-loan", "defi", "liquidity", "twap", "chainlink", "mango-markets", "amm", "lending", "defi-security"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The most dangerous on-chain manipulation isn't a sneaky trade against you; it's when the **liquidity** you're trading into or the **price** a protocol reads is itself fake.
>
> - Three related deceptions: **spoofed / thin liquidity** (a pool that displays \$2M of depth but is one-sided or yanked the instant you sell), **oracle manipulation** (ramming a thinly traded price so a protocol mints, borrows, or liquidates against a lie), and the **flash-loan-powered** version that does all of it inside one transaction with borrowed millions.
> - How to read it: check what oracle a lending market actually uses (a deep aggregated feed, or the spot price of one shallow pool?), and check whether the priced asset's pool is deep enough that a single large trade can't move it.
> - What you do with it: **avoid protocols that price off thin pools**, size your swaps to real depth (not the displayed TVL), and after a hack, read the attacker's transaction to confirm the price-manipulation loop.
> - The number to remember: **Mango Markets, ~\$117M, October 2022** — one attacker rammed a thinly traded perp, borrowed against the inflated mark, and walked out with the treasury.

On 11 October 2022, a trader named Avraham Eisenberg deposited collateral into Mango Markets, a Solana lending-and-perps protocol, across two accounts. Then he did something that looks, on the chart, like an ordinary aggressive buy: he bought the MNGO perpetual future, hard, in a market so thinly traded that his own orders moved the price. The mark on MNGO rocketed several-fold in minutes. Mango's risk engine, reading that mark, decided his other account was now sitting on an enormous unrealized profit — collateral it would happily lend against. So he borrowed. He borrowed out roughly **\$117M** of Bitcoin, Ether, Solana, and stablecoins against a number that existed only because he had just manufactured it. Then he let the price fall back, leaving the protocol holding the bad debt.

Eisenberg later called it "a highly profitable trading strategy," posted about it on Twitter, and even negotiated to return some funds. A US court disagreed about the "trading strategy" framing; he was convicted of fraud and market manipulation. But strip away the legal aftermath and the mechanism is the thing worth studying, because it is not exotic. It is the cleanest example of a whole family of attacks where **the price itself is the lie** — where an attacker does not beat the market, they *redefine* it for one block, and a protocol that trusted the wrong number pays out against a hallucination.

This post is about that family. Most on-chain manipulation guides teach you about trades *against* you — a sandwich that front-runs your swap, a wash trade that fakes volume. This is different. Here the manipulation is one layer deeper: the **liquidity** you see on the dashboard isn't real, or the **price feed** a protocol relies on can be pushed. If you trade, provide liquidity, or deposit into DeFi, this is the attack surface that decides whether your money is safe — and you can read most of it before you commit a dollar.

![Thin pool feeding an oracle that a protocol reads as truth, with an honest feed and a poisoned feed as the two outcomes](/imgs/blogs/fake-depth-spoofing-and-oracle-attacks-1.png)

The mental model above is the whole post in one figure. A protocol — a lending market, a perps exchange, a stablecoin — does not magically know what an asset is worth. It reads a **price feed**. That feed is a bridge from some market price into an on-chain number the protocol can act on. If the feed is wired to a deep, aggregated source, the price tracks reality and you are safe. If it is wired to a thin, shallow pool that one trade can move, then an attacker who controls that pool for one block controls the protocol's idea of truth. The lie propagates straight into your collateral. Everything below is about recognizing which side of that divide a protocol is on.

## Foundations: oracles, depth, price impact, and flash loans

Before the attacks make sense, four ideas have to be solid. None of them is hard, but each one is a place where a beginner's intuition is subtly wrong, and the attacks live precisely in those wrong intuitions.

### What an oracle is, and why protocols need one

A blockchain is a closed world. A smart contract can see everything that has ever happened *on its own chain* — every transfer, every balance, every swap in every pool — but it cannot see anything off-chain. It does not know the price of ETH on Coinbase, the USD/EUR rate, the score of a football match, or the weather. It has no internet connection and no senses. This is by design: every node must reach the same answer when it re-runs the code, so the code can only depend on data already on the chain.

That creates a problem the moment you want to build anything financial. A lending market needs to know "how much is the collateral worth?" to decide how much you can borrow. A perps exchange needs a price to mark positions and trigger liquidations. A stablecoin needs to know whether it is holding \$1 of backing per token. None of those numbers live on-chain natively. So you need a **bridge** — a mechanism that takes an off-chain or off-protocol price and writes it onto the chain in a form the contract can read. That bridge is an **oracle**.

There are two broad ways to build one, and the difference between them is the entire security story of this post:

- **Read a price already on-chain.** The cheapest oracle just looks at an automated market maker (AMM) pool — say, the ETH/USDC pool on a DEX — and uses its current ratio as "the price." No external data needed; it is already on the chain. The catch: that price is whatever the last trade made it, and anyone can make a trade.
- **Bring an external price on-chain.** A network of independent reporters (the canonical example is **Chainlink**) each fetch the price from many exchanges, agree on a median, and post that aggregated number on-chain at intervals. This is more robust but more complex, and it costs gas to keep updating.

We covered how decentralized oracle networks actually achieve agreement, and why aggregation across many sources is the security property, in [Chainlink and blockchain oracles](/blog/trading/crypto/chainlink-and-blockchain-oracles). For this post, the one sentence to carry forward is: **an oracle is only as honest as its source.** A protocol that prices off a single shallow pool has, in effect, outsourced its idea of truth to whoever can afford to move that pool.

### What "liquidity depth" means (and why TVL lies)

When people say a market is "liquid," they mean you can trade size without moving the price much. On an order-book exchange, that's the stack of resting buy and sell orders near the current price. On an AMM — the dominant DEX design — there is no order book. Instead, liquidity providers (LPs) deposit pairs of tokens into a **pool**, and a formula sets the price from the ratio of the two tokens in the pool.

The classic formula is the **constant product**: `x * y = k`, where `x` and `y` are the quantities of the two tokens and `k` is fixed. The price of token X in terms of token Y is just `y / x`. When you buy token X, you add Y to the pool and remove X, which lowers `x` and raises `y` — so the price goes up as you buy. The more you buy, the worse your average price. That movement is **price impact** (also called slippage), and how much you suffer depends entirely on how much is in the pool.

Here is the trap. The number a dashboard usually shows you is **Total Value Locked (TVL)** — the dollar value of everything in the pool. TVL is *not the same as depth near the current price*. A pool can show a large TVL while almost all of that value sits far from where you'd actually trade, leaving the price razor-thin to small orders. In modern "concentrated liquidity" AMMs, LPs choose a price *range* to provide in; a pool can be deep at one price band and empty a few percent away. So "the pool has \$2M of TVL" tells you very little about what your \$100k sell will actually cost. We unpack pool mechanics and how to read real depth in [reading DEX liquidity and pools](/blog/trading/onchain/reading-dex-liquidity-and-pools); the point here is that **displayed TVL is a headline, not a quote.**

#### Worked example: a "deep-looking" pool fills you 30% worse

Two pools both display **\$2M TVL**. Pool A is a broad, balanced full-range pool; pool B has 85% of its value concentrated just *above* the current price (so it's deep if you're buying, shallow if you're selling) plus a thin tail on the sell side — effectively only about **\$300k** of real depth where you'd be selling.

You want to sell **\$100k** of the token into each.

- In pool A, a \$100k sell against genuine broad depth moves the price about **0.5%**. You receive roughly **\$99,500**. Cost to price impact: about **\$500**.
- In pool B, your \$100k sell is one-third of the *real* sell-side depth. It walks the price down hard — call it a **30%** average slippage on the marginal token sold. You receive roughly **\$70,000**. Cost: about **\$30,000**.

Same TVL on the dashboard, same trade size, a **\$29,500** difference in outcome. The intuition: depth near the price — not the TVL headline — is what your money actually trades against, so a one-sided pool can fill you 30% worse than a balanced one that looks identical on the surface.

![Same hundred thousand dollar swap filling 30 percent worse in a thin one-sided pool than in a deep balanced pool](/imgs/blogs/fake-depth-spoofing-and-oracle-attacks-2.png)

### What price impact looks like, formally enough to reason about

You do not need calculus to reason about this, only one fact: on a constant-product AMM, price impact grows roughly with the *square* of how big your trade is relative to the pool. Doubling your order roughly quadruples the slippage; trading 10% of a pool's reserve moves the price by something like 20%+. That convexity is why a pool that's fine for \$5k trades can be brutal for \$100k ones, and why an attacker with a flash loan — who can trade *enormous* size for one block — can move a thin pool's price by multiples, not percentages.

It's worth seeing the arithmetic once, because it makes the attack's economics obvious. Take a pool with `x = 1,000,000` USDC and `y = 1,000,000` TOKN, so `k = 10^12` and the price is \$1.00 per TOKN. Now an attacker pushes 1,000,000 USDC into it (doubling the USDC side). The pool must keep `x * y = k`, so the new TOKN reserve is `k / 2,000,000 = 500,000`. The attacker took out `1,000,000 − 500,000 = 500,000` TOKN, and the new price is `2,000,000 / 500,000 = \$4.00` — a **4× move** from a single trade equal to the pool's own size. If an oracle was reading that pool's spot price, it now believes TOKN is worth \$4.00. The attacker spent borrowed money they'll repay in the same block; the only thing they actually *bought* was four seconds of a false price, which is all they need.

That same arithmetic shows why depth is the only thing that protects a price. To move the price only 1%, the attacker would need to push in roughly 0.5% of the pool — for a \$10M pool that's only \$50k, but for a \$10B aggregate it's \$50M, and to move it the multiples an exploit needs you'd have to deploy capital comparable to the entire market. Depth converts "a flash loan can do it for free" into "you'd have to move the whole world." There is no clever code in between; it is pure reserve size.

The mirror image of "you suffer slippage when you trade size" is "an attacker can *manufacture* a price by trading size on purpose." The same convexity that punishes a large honest sell is the lever a manipulator pulls to ram a price up before the protocol reads it. Keep that symmetry in mind: **price impact is a cost to you and a tool to an attacker.**

### What a flash loan is

A flash loan is the piece that turns these manipulations from "rich-attacker-only" into "anyone-with-a-script." Normally, to borrow money you must post collateral, because the lender needs assurance you'll repay. A **flash loan** removes the collateral requirement by adding a different assurance: you must repay *within the same transaction*. The protocol lends you the money at the start of the transaction, runs your code, and at the end checks whether the money (plus a small fee) came back. If it didn't, the entire transaction reverts as if it never happened — the loan, your trades, everything is unwound. Because a blockchain transaction is **atomic** (all of it happens or none of it does), the lender takes no risk: either you repay or the loan was never really made.

The consequence is staggering and easy to under-appreciate: **for the duration of one transaction, anyone can wield tens of millions of dollars with no capital.** You can borrow \$50M, do whatever you want with it across a dozen protocols, and repay it microseconds later — all in one block. Flash loans have entirely legitimate uses (arbitrage, collateral swaps, refinancing in one step). But they also mean the attacker's bankroll is no longer a defense. If your protocol could be exploited "if only someone had \$50M to move a pool," then flash loans guarantee someone effectively *does*.

### What a TWAP is (the first real defense)

The simplest fix for "a single trade can move my spot price" is to stop reading the spot price and read an **average over time** instead. A **TWAP** — time-weighted average price — is the average price over the last N blocks (or minutes). Many AMMs expose one natively by recording a running price accumulator.

Why does averaging help? Because to move a TWAP, an attacker can't just spike the price for one block; they have to *hold* the manipulated price across the whole averaging window, and every block they hold it, arbitrageurs are trading against them, bleeding their capital. Moving a spot price for one block is nearly free with a flash loan (you unwind in the same transaction). Moving a 30-minute TWAP means defending a fake price against the whole market for 30 minutes — which a flash loan can't do, because flash loans last one transaction. TWAP converts a cheap one-block attack into an expensive multi-block one. It is not bulletproof (a thin enough pool can still be held, and TWAP lags real moves), but it is the first serious wall.

The most robust setups go further and read an **aggregated off-chain feed** (Chainlink-style): the median of prices from many exchanges, where moving the on-chain number means moving the *entire global market* for the asset. That's prohibitively expensive for all but the most thinly traded assets. The whole spectrum — spot, TWAP, aggregated — is the design space, and where a protocol sits on it determines whether it's a target.

It's worth being precise about *why* a median across venues is so much stronger than a single pool, because it's the crux of the whole defense. A single pool is one number that one trade can set. A median across, say, ten major exchanges is a number that only moves if you move *most* of those exchanges at once — and each of those exchanges has its own deep order book, its own arbitrageurs, and its own market makers who will instantly trade against any price that's out of line. To push the median, you'd have to simultaneously overpower the deepest liquidity in the entire market, on every venue, faster than the world's arbitrage bots can correct you. That's not a flash-loan-sized problem; that's a "you'd need to be the market" problem. The cost scales with the *total* liquidity of the asset everywhere, not the liquidity of one pool — which is exactly the property you want backstopping your collateral.

With those five ideas in hand — oracle, depth, price impact, flash loan, TWAP — every attack below is just a recombination.

## Spoofed and thin liquidity: when the depth itself is fake

Start with the simplest deception, because it costs you money even when there is no "hack" at all. The liquidity you see is not the liquidity you get.

### One-sided pools: deep until you trade the wrong way

A constant-product pool is symmetric, but a concentrated-liquidity pool is not. An LP can place all their capital in a narrow band *above* the current price — meaning there's plenty for buyers but almost nothing for sellers — and the dashboard will still total it all up as TVL. To a buyer, the pool looks and behaves deep. To a seller, it's a cliff. This isn't always malicious; it's often just how LPs position. But a token's promoters can deliberately stack one-sided liquidity to make the chart look healthy and the buy side smooth, while anyone trying to *exit* discovers the floor isn't there.

The tell is asymmetry. If you simulate a \$50k buy and it shows 0.4% impact, then simulate a \$50k sell and it shows 18% impact, the pool is one-sided. The displayed TVL is the same in both directions; only the depth differs. A defender always checks **both sides** before sizing a position, because the side that matters — the exit — is usually the thin one.

Why does one-sided depth form in the first place? Sometimes it's innocent: in a strongly trending token, LPs in a concentrated-liquidity pool naturally end up holding mostly one asset as the price moves through their range, so the live depth skews. But it's also a deliberate launch tactic. A team that wants their token's chart to look healthy will seed liquidity tightly above the launch price — so early buyers get smooth fills and a rising line — while leaving the sell side thin, so that when those buyers try to take profit, they crater their own exit and the line "holds." The chart tells a story of strength; the order book tells a story of a trap. The only way to know which you're in is to simulate the exit, not admire the chart. This is the same survivorship-and-appearance problem that runs through token due diligence generally — a green dashboard is a marketing surface, not a guarantee, and the discipline is to test the claim the number is making rather than trust the number.

### Removable liquidity: deep until the block before you trade

The more aggressive version is liquidity that's real right now but can vanish on command. On most AMMs, an LP can withdraw their entire position in a single transaction. If one wallet provides most of a pool's depth, that wallet can watch the public mempool — where your pending sell sits, visible, before it's mined — and **front-run** you: pull the liquidity in the block *before* your sell, let your order crater down the now-empty book, then re-add the liquidity afterward. We cover the mempool-visibility mechanic that makes this possible in [MEV, sandwiches, and front-running](/blog/trading/onchain/mev-sandwiches-and-frontrunning); here the point is narrower: **a pool's depth is a removable snapshot, not a promise.** What you saw on the dashboard a minute ago says nothing about what's there when your transaction actually executes.

![Pool showing two million TVL on the dashboard whose concentrated liquidity is pulled the block before a sell, collapsing the fill](/imgs/blogs/fake-depth-spoofing-and-oracle-attacks-3.png)

The figure shows the gap between the dashboard view and the execution view. At rest, the pool reads \$2M TVL with a tight spread and a deep-looking chart. But the depth is concentrated and controlled by a single wallet, and your sell is visible in the mempool. The instant it's profitable, that wallet pulls its liquidity, your order walks down a thin book, you fill far below quote, and the liquidity returns once you've been harvested. The defender's read — the green node — is the only durable defense: **check LP concentration and removability before you trust the depth.** A pool whose liquidity is one wallet that can leave is a pool that will leave when it hurts most.

#### Worked example: a one-sided pool and a removed-liquidity sell

You hold tokens you bought for **\$40,000** and they now show a paper value of **\$120,000** at the quoted price. The pool displays **\$1.8M TVL**, so a \$120k exit "should" be a few percent of the pool — comfortable.

But the pool is provided almost entirely by one wallet, concentrated just above the current price. You submit a market sell of your full **\$120,000**. The LP wallet, seeing it in the mempool, withdraws **\$1.5M** of its position one block ahead. Now your sell hits a pool with maybe **\$300k** of real depth. Your \$120k order is **40%** of what's left; it walks the price down and you net roughly **\$78,000** instead of \$120,000 — a **\$42,000** haircut. Right after, the wallet re-adds liquidity and the quoted price snaps back, so the chart barely shows what happened to you.

The intuition: a paper gain priced against removable, one-sided liquidity is not a realized gain — it's a number that exists only until you try to collect it.

This is the retail-facing edge of the problem. No protocol was hacked; you simply trusted a displayed number that wasn't binding. But the *same* fake-depth mechanism, pointed at a protocol's oracle instead of at your sell order, becomes a multi-million-dollar exploit. That's next.

## Oracle manipulation: ramming a thin price into a protocol's brain

Now connect the two foundations. A protocol reads a price from somewhere. If that "somewhere" is a thin pool, and a thin pool's price can be rammed by a large trade, then an attacker can *write* a false price into the protocol's brain and act against it before anyone can correct it.

### The mechanism, step by step

The attack has a fixed shape, regardless of which protocol is the victim:

1. **Find the dependency.** Identify a protocol that prices an asset off a manipulable source — usually the spot price of a single, shallow pool, or a perp mark on a thinly traded market. This is reconnaissance you can do too, as a defender, *before* depositing.
2. **Get the capital.** Take a flash loan for as much as you need — \$10M, \$50M, \$100M — with no collateral, to be repaid in the same transaction.
3. **Ram the pool.** Trade the borrowed capital into the thin pool, hard, in the direction that moves the priced asset's price the way you want — up if you'll borrow against it, down if you'll liquidate someone or mint cheaply.
4. **Exploit the false price.** While the oracle reads the manipulated price, do the thing the protocol lets you do at that price: borrow against inflated "collateral," mint stablecoins against an inflated asset, or trigger liquidations and scoop the collateral.
5. **Unwind.** Sell back out of the pool. The pool's price returns to normal. You keep whatever the protocol paid you at the false price.
6. **Repay the flash loan.** Return the borrowed capital plus fee. Net the difference.

Every step happens **inside one transaction**, so there is no window for arbitrageurs or the protocol's team to react. By the time a human sees it, the money is gone and the pool price looks normal again.

The atomicity is what makes this fundamentally different from old-world market manipulation. A trader cornering a commodity market in the physical world has to hold their position over days, exposed the whole time to the price moving against them, to regulators, to margin calls. The flash-loan version compresses all of that into a single block where *nothing can move against the attacker*, because the entire sequence either completes profitably or reverts to never having happened. There is no holding period, no exposure, no risk of the price recovering before they exit — the recovery is *also* inside their transaction, executed by them, on purpose. That's why the capital requirement collapses to zero and the risk collapses to "the gas fee if it reverts." It is manipulation with the time dimension removed, which is precisely the dimension that used to make manipulation dangerous to attempt.

![Six-stage flash-loan oracle attack from borrowing capital through ramming a pool to repaying the loan, all in one transaction](/imgs/blogs/fake-depth-spoofing-and-oracle-attacks-4.png)

The pipeline above is the canonical loop. Notice the colors: the flash loan and pool-ram (amber and red) are where the lie is manufactured, the oracle read and exploit (red) are where it's monetized, and the unwind-and-repay (blue) are where the evidence is cleaned up. The reason this is an oracle attack and not just "a big trade" is step 3-to-4: the protocol *believed* the rammed price was the real price of the asset, because that's the only price it could see.

#### Worked example: a \$50M flash loan doubling a thin pool's oracle price

A lending protocol prices a token called XYZ off the spot price of a single pool that holds **\$5M of XYZ and \$5M of USDC** (so XYZ trades at \$1.00, and the pool's TVL is \$10M). XYZ has almost no other liquidity anywhere, so this pool *is* the price.

The attacker flash-borrows **\$50M** of USDC and buys XYZ in that pool. On a constant-product curve, pushing \$50M of USDC into a \$5M-per-side pool moves the price dramatically — the price of XYZ roughly **doubles to ~\$2.00** as the curve steepens (the exact figure depends on the curve, but doubling a pool this size with 10× its depth is easy). The protocol's oracle, reading spot, now believes XYZ is worth **\$2.00**.

The attacker had earlier deposited a small amount of XYZ as collateral — say **5M XYZ**, which the protocol now values at **5M × \$2.00 = \$10M** instead of its true \$5M. At a 75% loan-to-value ratio, the protocol lets them borrow **\$7.5M** of *real* stablecoins against that fake \$10M. They take the **\$7.5M**, sell their XYZ back to unwind the pool, repay the \$50M flash loan plus fee, and walk away with roughly **\$7.5M** minus costs.

The intuition: the attacker never *had* \$50M — they rented it for a microsecond, used it only to bend the one number the protocol trusted, and converted that bent number into real borrowed dollars the protocol can never recover.

### Why the protocol can't tell the difference

The maddening part, from the protocol's side, is that *every individual step is a legal operation.* Taking a flash loan is allowed. Trading in a pool is allowed. Borrowing against collateral at the current oracle price is allowed. The protocol has no rule that's being broken — it's faithfully executing its own logic against a price it has no reason to distrust. The vulnerability isn't a bug in the code; it's a **bug in the assumption** that the price source is honest. That's why fixes are about the oracle design, not about patching a function: you can't out-code a bad price source, you can only stop trusting it.

This is also why oracle manipulation is so hard to fix reactively. A reentrancy bug or a signature flaw can be patched once it's found, and the same code is then safe. But an oracle dependency on a thin pool isn't a defect that gets "fixed" — it's a design choice that has to be *replaced*, often by integrating a whole external oracle network or building and battle-testing a TWAP. In the gap between "we realize our oracle is fragile" and "we've migrated to a robust one," the protocol is sitting under a known sword. And because the dependency is publicly readable on-chain, attackers can find these fragile setups proactively, scanning for protocols whose price source is a shallow pool — exactly the reconnaissance step 1 of the attack describes, which is also exactly what a defender should be doing first.

There's a deeper lesson here about composability, the property that lets DeFi protocols call into each other freely. Composability is what makes a flash loan possible at all: the attacker stitches a lender, a DEX, and a victim protocol into one transaction precisely because they can all be called from a single contract. The same openness that makes DeFi powerful — anyone can build on top of anyone — means that a protocol's security depends not just on its own code but on every price source it reads and every protocol that can be wedged between its calls. You are never auditing one contract; you are auditing a graph of trust, and the oracle is usually the weakest edge in that graph.

## The canonical case: Mango Markets, ~\$117M

Mango Markets is the example everyone reaches for because it's the cleanest. Let's walk it as a forensic case study, the way you'd reconstruct it after the fact.

### What happened

Mango was a Solana-based protocol offering spot, margin, and perpetual futures, with a shared cross-collateral risk engine: your various positions all contributed to one health number that determined how much you could borrow. The price used to mark positions and value collateral came, for the thinly traded MNGO token, from a source that reflected MNGO's own (very shallow) market.

On 11 October 2022, the attacker funded two accounts with stablecoins. From account A, he bought the **MNGO perpetual** aggressively; from account B, he took the *other side* (so he was, in aggregate, just moving the price, not taking directional risk against himself). Because MNGO's market was so thin, his buying spiked the MNGO mark by roughly **5–10×** in minutes. Now account B — which was long MNGO via spot/perp exposure valued at the inflated mark — showed a massive unrealized profit. Mango's risk engine counted that inflated value as borrowing power.

So he borrowed against it. He withdrew essentially everything the protocol held — **BTC, ETH, SOL, USDC, and more, totaling roughly \$117M** — as loans against collateral that was only worth that much because he had just manufactured the price. The price later fell back toward reality, leaving Mango with a hole: the loans were real and outgoing, the collateral backing them was a mirage.

![Six-stage Mango Markets attack from funding two accounts through ramming the MNGO perp to draining one hundred seventeen million dollars](/imgs/blogs/fake-depth-spoofing-and-oracle-attacks-5.png)

The pipeline maps the mechanism: fund two accounts, ram the thin perp from one, watch the mark spike, see the inflated unrealized PnL counted as collateral, borrow out the treasury, and leave the bad debt behind. The neutral first node and the cascade of red after it tell the story — the only "normal" step was the deposit; everything after the ram was the protocol faithfully lending against a price the attacker controlled.

#### Worked example: the Mango \$117M, sized

Put rough numbers on it. MNGO was trading near **\$0.03** before the attack. The attacker's buying pushed the mark to roughly **\$0.15** — a **5×** move — in a market where that took only a few hundred thousand dollars of buying because the order book was so thin.

Account B's MNGO-linked exposure, say a notional position that was worth about **\$20M** at \$0.03, was now marked at about **\$100M+** at \$0.15. Against that inflated collateral, Mango's risk engine extended borrowing power that let him withdraw roughly **\$117M** of genuine, liquid assets — real BTC and ETH and stablecoins that left the protocol and never came back as anything but bad debt. The capital he risked to move the price was a tiny fraction of the **\$117M** he extracted.

The intuition: the leverage here wasn't financial leverage on a position — it was leverage on *the price itself*, where a few hundred thousand dollars of buying in a thin market unlocked nine figures of borrowing against the lie.

### Where it sits among the big thefts

Mango's \$117M is large, but in the grim leaderboard of on-chain thefts it's mid-pack — and that's worth seeing, because it tells you oracle manipulation is one repeatable category among several, not a freak event.

![Horizontal bar chart of the biggest on-chain thefts with Mango Markets at one hundred seventeen million highlighted in amber for oracle manipulation](/imgs/blogs/fake-depth-spoofing-and-oracle-attacks-6.png)

The amber bar is Mango — the canonical oracle-manipulation case — sitting among bridge hacks, key compromises, and signature bugs that dwarf it (Bybit at \$1.46B, Ronin at \$625M). The lesson isn't that oracle attacks are the biggest; it's that they're a *standing* category. Every time a new protocol prices a borrowable or mintable asset off a thin source, the Mango template is sitting there waiting. We catalog how these post-mortems are read and how the stolen funds are traced afterward in [the anatomy of a DeFi hack](/blog/trading/onchain/anatomy-of-a-defi-hack).

### The legal twist that matters for analysts

Eisenberg argued in public that this was legitimate trading — he "simply" used the protocol as designed and the price moved because he bought. The court rejected that; he was convicted of fraud and market manipulation. For an on-chain analyst, the takeaway isn't the law per se — it's that **"every step was an allowed operation" is not a defense, and it's also not a safety guarantee.** A protocol can be drained entirely through legal operations if its price assumption is broken. When you evaluate where to put money, "the code has no bugs" is not the same as "the code can't be manipulated."

The Mango case is also instructive about what *recovery* looks like, because it's rare. Eisenberg returned a portion of the funds in exchange for the protocol's DAO agreeing not to pursue charges — a deal that the criminal justice system later disregarded entirely. Compare that to the broader leaderboard: the Poly Network attacker returned everything, Euler's attacker returned everything, but the Lazarus-attributed thefts (Bybit, Ronin, DMM) were laundered and gone. The pattern is that grey-hat exploiters — those who attack to prove a point or negotiate a bounty — sometimes return funds; state-sponsored and purely criminal actors do not. For a depositor, this means you cannot price in recovery as a backstop. If a protocol you're in gets oracle-drained, the realistic base case is that your share of the loss is permanent. That asymmetry — small yield upside, total-loss downside, no recovery — is exactly why the fragility check belongs *before* the deposit, not after the headline.

## Spot vs TWAP vs aggregated feeds: why robust oracles resist this

Everything above hinges on the oracle reading a manipulable price. So the defense is entirely about *what the oracle reads*. There's a clean spectrum, and knowing where a protocol sits on it tells you almost everything about its oracle risk.

![Matrix comparing raw spot, TWAP, and aggregated feeds across what they read, cost to manipulate, weakness, and best use](/imgs/blogs/fake-depth-spoofing-and-oracle-attacks-7.png)

Read the matrix top to bottom as a ladder of increasing manipulation cost:

- **Raw spot (one pool).** Reads the pool's price right now, this block. Cost to manipulate: cheap — one flash loan, one block, unwind in the same transaction. This is the Mango/flash-loan setup. A raw spot read should *never* be a protocol's sole oracle for anything borrowable or mintable. If you see it, that's a red flag, full stop.
- **TWAP (time-averaged).** Reads the average price over the last N blocks. Cost to manipulate: now you have to *hold* the manipulated price across the whole window, defending it against arbitrageurs the entire time — which a flash loan (one transaction) structurally cannot do. The weakness is that it lags fast real moves and is still an on-chain-only source, so a thin enough pool can in principle be held by a determined, well-capitalized attacker across multiple blocks. Good for on-chain assets with *deep* pools.
- **Aggregated off-chain feed (Chainlink-style).** Reads a median of prices from many independent exchanges. To move the on-chain number you'd have to move the *global market* for the asset across all those venues at once — prohibitively expensive for any liquid asset. The remaining weaknesses are operational, not manipulation-based: update latency (the feed only refreshes periodically) and oracle downtime (what if the feed stalls?). This is what serious lending markets, perps, and stablecoins use.

The single most important question you can ask about any DeFi protocol you're considering is therefore: **which row of this matrix is its oracle in, and how deep is the asset it's pricing?** A robust aggregated feed on a deep asset is genuinely hard to attack. A raw spot read on a thin token is Mango waiting to happen.

#### Worked example: a lending market priced off a \$300k pool

A new lending market lets you borrow against a token, TOKN, and prices TOKN off the spot price of a pool holding about **\$300k** total. You're tempted by a high yield. Before depositing, you do the fragility check yourself.

You simulate the impact of a **\$500k** trade in that pool. A \$500k order against ~\$150k-per-side depth doesn't just move the price — it *dominates* the curve, moving TOKN's price by roughly **40%+** in one trade. So a flash-borrowed \$500k (trivial to obtain) can swing the oracle 40% at will. Now ask: what can someone do with a 40% price swing? If they've deposited TOKN as collateral, a +40% pump lets them borrow ~40% more real stablecoins than their collateral is worth, then unwind and walk. The protocol's entire borrowable pool is at risk to a **\$500k** flash loan — capital anyone can rent for a few dollars of fee.

The intuition: when the asset's whole on-chain price lives in a pool a single rentable trade can move 40%, the oracle isn't a price feed — it's an open door, and the yield you were chasing is the bait in front of it.

## How to read it: checking a protocol's oracle before you deposit

This is the hands-on part — the "how do I actually do this with tools" pass. None of it requires you to be a developer; it requires you to be willing to read.

### Step 1: find out what oracle the protocol uses

Most reputable protocols document their oracle in their docs, and you can verify it on-chain. The workflow:

- **Read the docs first.** Search the protocol's documentation for "oracle." Reputable lending markets (Aave, Compound, and their forks) and perps name their price source explicitly — usually Chainlink for major assets, sometimes a TWAP for long-tail ones. If the docs are silent or vague about pricing, treat that as a warning by itself.
- **Verify on-chain via the block explorer.** On Etherscan (or the chain's explorer), find the protocol's price-oracle contract and read its configuration. You're looking for whether it points at a Chainlink aggregator address, an AMM pool's TWAP, or a raw `getReserves()`/spot read. A contract that calls a single pool's current reserves to derive a price is reading raw spot — the fragile row.
- **Check per-asset, not just per-protocol.** A protocol can use a rock-solid Chainlink feed for ETH and BTC, then a raw spot read for some obscure listed token because no Chainlink feed exists for it. The blue-chip assets can be safe while the long-tail asset is a Mango waiting to happen. *The risk lives in the weakest-priced asset you're exposed to,* including any asset you'd be liquidated against.

You can use a transaction explorer to confirm this by reading the protocol's actual price-update or borrow transactions and seeing which oracle contract they call — the same tracing technique we walk through in [how to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow).

### Step 2: check whether the priced source is manipulable

Knowing the oracle reads "pool P's spot price" only matters if pool P is thin. So:

- **Find the pool's real depth, not its TVL.** On a DEX analytics dashboard or directly via the pool contract, look at the liquidity *near the current price* — for concentrated-liquidity pools, the depth within ±2% of spot. A pool can show \$5M TVL with only \$300k of usable depth at the price.
- **Simulate a large trade both directions.** Most DEX front-ends and aggregators let you enter a trade size and show the expected price impact. Enter a size comparable to what a flash loan could deploy (millions). If a \$5M simulated trade moves the price 30%+, the pool is rammable, and any oracle reading its spot price is exploitable.
- **Check who provides the liquidity.** If one or two wallets provide most of the pool, the depth is removable on command, and even the depth you measured is conditional. Bubblemaps-style holder/LP views and the pool's mint/burn event history show concentration.

#### Worked example: pricing the cost of an attack before you deposit

You're considering a lending market that lets you supply USDC to earn yield, but the same market lets others borrow against a token, GMBL, that's priced off a single pool. Your USDC is what gets borrowed, so GMBL's fragility is *your* risk. You do the math an attacker would do.

The GMBL pool holds about **\$2M** total (\$1M per side), GMBL trades at \$1.00, and the market's GMBL collateral factor is 70%. You simulate a flash-loaned **\$4M** buy. Pushing \$4M into the \$1M USDC side takes the USDC reserve to \$5M; keeping the product constant, the GMBL reserve falls from 1M to `5,000,000,000,000 / 5,000,000 = 1,000,000`... checking with the per-side product `1M × 1M = 10^12`, after adding \$4M the new GMBL reserve is `10^12 / 5,000,000 = 200,000`, so the attacker pulled out 800,000 GMBL and the new price is `5,000,000 / 200,000 = \$25` — a **25× spike**. At that price, even a modest GMBL deposit becomes enormous borrowing power: 100,000 GMBL deposited earlier, worth \$100,000 honestly, is now valued at **\$2.5M**, against which the attacker can borrow **\$1.75M** of your USDC. Cost to the attacker: a \$4M flash loan fee of roughly **\$2,000–\$3,600**. Return: **\$1.75M** of real stablecoins.

The intuition: when the attacker's cost to drain a market is three thousand dollars and the prize is your seven-figure deposit, the only thing standing between you and the loss is whether anyone's noticed yet — which is not a risk you're being paid enough to take.

### Step 3: read the attacker's transaction after a hack

When a protocol does get hit, the transaction is public and permanent, and reading it confirms the mechanism. After a suspected oracle exploit, pull the attacker's transaction on the explorer and look at the **internal transaction trace** — the ordered list of calls the transaction made. The signature of a flash-loan oracle attack is unmistakable in that trace:

- A **flash-loan borrow** call near the top (from Aave, Balancer, dYdX, or a DEX flash-swap).
- A **large swap into a thin pool** right after — the ram.
- A **borrow / mint / liquidate** call against the victim protocol while the rammed price is live.
- A **reverse swap** unwinding the pool.
- A **flash-loan repay** at the bottom, with the net difference staying in the attacker's wallet.

Seeing those five in order, in a single transaction, *is* the proof that the price was the weapon. It also tells you exactly which pool and which oracle to flag so the same template can't be reused elsewhere.

A concrete reading aid: when you open the attacker's transaction on a tracing front-end (Etherscan's "Internal Txns" tab, or a dedicated tool like Tenderly or Phalcon that renders the full call tree), you're not trying to understand every line of bytecode. You're scanning for *shape*. The flash-loan borrow shows up as a transfer of a very large sum into the attacker's contract with nothing posted as collateral. The ram shows up as a swap whose output token spikes the pool's reserve ratio. The exploit shows up as a `borrow`, `mint`, or `liquidate` call to the victim protocol, sandwiched in time between the ram and the unwind. And the repay closes the loop with the same large sum (plus a fee, typically 0.05–0.09% on Aave) flowing back to the lender. If you can label those five calls in the trace, you have fully reconstructed the attack — and you can hand that reconstruction to the protocol, an exchange compliance team, or a forensic firm who will use it to trace where the proceeds went next.

One more thing the trace tells you that the headline number doesn't: *whether anyone else can copy it.* Once an oracle dependency is public knowledge — and the attacker's transaction makes it public the moment it lands — every other protocol pricing off the same thin pool is now a known target. This is why oracle exploits often come in clusters: the first attacker proves the pool is rammable, and copycats race to hit every other protocol that trusts it before the pools are drained or the oracles are patched. Reading one attack teaches you which *other* deposits to pull.

### The fragility check, as one matrix

Put the two questions together — *what does the oracle read?* and *how deep is the source?* — and you get a single decision grid you can apply to any protocol in about five minutes.

![Matrix of oracle type against pool depth showing which combinations are danger, risky, caution, ok, and safe](/imgs/blogs/fake-depth-spoofing-and-oracle-attacks-8.png)

The grid is read by finding your protocol's square. Raw spot on a thin pool (top-left) is the danger zone — the Mango template, exploitable by one cheap flash loan. As you move down (spot → TWAP → aggregated feed) the cost of manipulation climbs; as you move right (thin → medium → deep) the source gets harder to move. Only the bottom-right corner — a robust aggregated or time-weighted feed sitting on genuinely deep liquidity — is *safe*. Everything in between is a spectrum of "how much are you betting nobody bothers." The practical discipline is simple: locate your square before you deposit, and if it isn't green, either size down to a loss you'd accept or walk away. The yield is never worth the top-left corner.

## Common misconceptions

**"High TVL means the pool is safe to trade and safe to price off."** No. TVL is the dollar total of everything in the pool; depth near the price is what your trade — and an oracle — actually interacts with. A \$5M-TVL pool can have \$300k of usable depth, fill a \$100k sell 30% worse than expected, and be rammable by a flash loan. Always measure depth near price and simulate a large trade both directions; the TVL headline is the least informative number on the page.

**"If the protocol's code has no bugs, my deposit is safe."** No. The Mango drain used zero code exploits — every operation was allowed. The vulnerability was the *assumption* that the price source was honest. A bug-free protocol that prices a borrowable asset off a thin pool can be drained entirely through legal operations. "Audited, no bugs" and "manipulation-resistant oracle" are different properties; you need the second one.

**"Chainlink (or any oracle) makes a protocol immune."** No — it depends on *which* assets use it and how the protocol consumes it. A protocol might use Chainlink for ETH but a raw spot read for a long-tail token, and the long-tail token is the soft underbelly. Even with a good feed, a protocol that uses a stale price, ignores the feed's freshness checks, or lets you act between updates can still be gamed. Robust source *and* careful consumption are both required.

**"Flash loans are the vulnerability, so banning them fixes it."** No. Flash loans only *amplify* the real flaw — a manipulable price source. The same attack is possible (just costlier) for any whale who already holds the capital. Banning flash loans would slow attackers, not stop them, and would break legitimate arbitrage that keeps prices honest. The fix is a manipulation-resistant oracle, which closes the door regardless of where the capital comes from.

**"If the chart snapped right back, nothing happened."** No — that's the *signature* of the attack, not evidence of innocence. A flash-loan ram-and-unwind leaves the pool price looking normal seconds later precisely because the attacker reversed their pool trade in the same transaction. The damage is in the protocol's books (bad debt) and in your fill (if you sold into a yanked pool), not in the lingering chart. To see what happened you have to read the transaction, not the price line.

## The playbook: what to do with it

Here is the if-then checklist for a trader, an LP, and an analyst. Each item is **signal → read → action → what would make it a false alarm.**

### As someone about to deposit into a DeFi protocol

- **Signal:** the protocol prices a borrowable or mintable asset off a single pool's spot price (raw `getReserves` / current ratio). **Read:** this is the fragile top row of the oracle matrix — the Mango template. **Action:** do not deposit, or only with size you'd accept losing; prefer protocols on aggregated feeds or deep-pool TWAPs. **False alarm:** the asset is itself so deep across many venues that no rentable trade moves it (rare for long-tail tokens; check by simulating a multi-million-dollar trade).
- **Signal:** a long-tail asset on the protocol uses a different, weaker oracle than the blue-chips. **Read:** the risk lives in the weakest-priced asset you're exposed to, including liquidation paths. **Action:** avoid exposure to the weakly-priced asset even if the rest of the protocol is fine. **False alarm:** you have zero exposure to that asset and can't be liquidated against it.

### As someone about to trade or provide liquidity

- **Signal:** displayed TVL is high but simulated price impact is large in one direction. **Read:** the pool is one-sided; the exit is thin. **Action:** size your trade to the *thin* side's real depth, split into smaller orders, or use an aggregator that routes across pools. **False alarm:** impact is small in both directions — the depth is genuinely balanced.
- **Signal:** most of a pool's liquidity is one or two wallets. **Read:** the depth is removable; it can be front-run away the block before your sell. **Action:** assume the depth isn't there for your exit; use limit-style routing or trade where depth is diffuse. **False alarm:** liquidity is spread across many LPs who can't coordinate a withdrawal.

### As an analyst or defender after an incident

- **Signal:** a protocol reports a sudden loss with the asset's chart looking normal again. **Read:** likely a flash-loan ram-and-unwind, not a slow drain. **Action:** pull the attacker's transaction, read the internal trace for the borrow → ram → exploit → unwind → repay sequence, and flag the manipulated pool and the consumed oracle. **False alarm:** the trace shows a key compromise or signature bug instead (the loss came from access, not price) — a different category entirely.
- **Signal:** a new high-yield protocol launches pricing a freshly listed token off its own launch pool. **Read:** the oracle and the only liquidity are the same thin pool — a closed loop an attacker can fully control. **Action:** treat the yield as bait; stay out until an independent, deep price source exists. **False alarm:** the token has substantial liquidity on multiple independent venues feeding an aggregated feed.

The single rule that ties it all together: **don't trust the number, trust the source of the number.** A price on a screen, a depth on a dashboard, a collateral value in a risk engine — each is only as honest as where it came from. The chain lets you check the source directly. The whole skill is the willingness to look one layer deeper than the displayed figure, every time, before your money is the one resting on the lie.

## Further reading & cross-links

- [Reading DEX liquidity and pools](/blog/trading/onchain/reading-dex-liquidity-and-pools) — how AMM pools work, how to measure real depth near price, and why concentrated liquidity makes TVL such a misleading headline.
- [MEV, sandwiches, and front-running](/blog/trading/onchain/mev-sandwiches-and-frontrunning) — the public-mempool mechanic that lets a liquidity provider pull depth the block before your sell, and how ordering inside a block is monetized.
- [The anatomy of a DeFi hack](/blog/trading/onchain/anatomy-of-a-defi-hack) — how post-mortems are read, how exploit categories are classified, and how stolen funds are traced after the fact.
- [Analyzing lending and liquidations](/blog/trading/onchain/analyzing-lending-and-liquidations) — how lending markets value collateral and trigger liquidations, the exact machinery an oracle attack abuses.
- [Chainlink and blockchain oracles](/blog/trading/crypto/chainlink-and-blockchain-oracles) — how decentralized oracle networks aggregate many sources into a manipulation-resistant feed, and why aggregation is the security property.
- [How to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow) — the hop-by-hop tracing technique you use to read an attacker's transaction and confirm which oracle and pool were the weapon.
