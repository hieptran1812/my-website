---
title: "MEV: The Invisible Tax on Every Trade"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "How public transaction ordering lets searchers, builders, and validators extract value from on-chain trades—and the practical defenses that reduce the toll."
tags: ["crypto", "mev", "ethereum", "defi", "mempool", "sandwich-attacks", "arbitrage", "liquidations", "market-structure", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 31
---

> [!important]
> **TL;DR** — MEV is the value created or captured by choosing which transactions enter a block, and in what order. Sometimes that ordering repairs a broken market; sometimes it quietly worsens your fill.
>
> - A public-mempool swap advertises its size, route, and slippage boundary before execution. Searchers can react, builders can order, and a proposer can sell the block-space decision.
> - Sandwiches are the clearest retail harm: a searcher buys before your swap, lets your buying move the pool, then sells after it. Slippage tolerance is the ceiling on the damage, not a promise of a fair fill.
> - Arbitrage and liquidations are often useful MEV. They align stale prices and close unsafe loans, but the cost is paid by liquidity providers, borrowers, or traders who created the opportunity.
> - Ethereum’s PBS/MEV-Boost supply chain separates searchers, builders, relays, and proposers; it redistributes revenue but does not make public order flow private.
> - The practical defense is layered: use a sensible slippage limit, prefer private/protected order flow, and use batch auctions or intent/RFQ systems when the trade is large or price-sensitive.
> - All arithmetic in this post is illustrative unless a source and an as-of date are stated. The named criminal case is discussed as an indictment and allegation, not as a conviction.

If you have ever swapped tokens on a decentralized exchange and wondered why the quoted output was better than the final output—or why a trade failed even though the market had not visibly moved—the answer may be hiding in the ordering of transactions around yours.

The front end shows you a quote. The chain settles a sequence. Between those two moments, another participant may observe your signed instruction, simulate its effect, submit related transactions, and pay for a preferred position in the sequence. That participant is not necessarily stealing your tokens. More often, it is collecting a small difference between the price you expected and the state it helped create.

That difference is part of maximal extractable value, usually shortened to MEV. The phrase sounds abstract, but the mechanism is concrete: transaction order is a scarce resource, and scarce resources acquire prices.

![The MEV supply chain from wallet and mempool to searcher, builder, relay, proposer, and final block](/imgs/blogs/mev-the-invisible-tax-on-every-trade-1.webp)

This is the Wave 7 post about the player who sits beneath every other player. Venture funds, market makers, foundations, exchanges, and whales can move the market by trading. MEV participants can move the outcome of a trade by deciding where that trade sits in a block. Read it alongside [Mining, Staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev), which covers the block producer’s economics from the consensus side. Here the focus is the market structure around ordering: who sees what, who bids for what, who pays whom, and what a retail trader can do.

## Foundations: the building blocks from zero

### A transaction is a signed state change

A blockchain is a replicated database. It stores balances and contract state, and many computers agree on the next valid update. A transaction is a signed instruction to make such an update: transfer ETH, call a lending contract, or exchange one token for another.

Your wallet creates and signs the instruction. An RPC endpoint—remote procedure call—is the service that carries it to a node. The node checks the signature and basic validity, then gossips the transaction to peers. Before inclusion, the transaction is pending. The set of pending transactions visible to a node is its mempool.

![Traditional broker orders are private before execution; a public-mempool swap is visible before it fills](/imgs/blogs/mev-the-invisible-tax-on-every-trade-2.webp)

The mempool is not one globally synchronized queue. Different nodes hear transactions at slightly different times, and a transaction sent to a private endpoint may not enter the public gossip network at all. That distinction matters. “The mempool” is useful shorthand for a changing set of pending observations, not a single universal screen that every participant sees at the same instant.

### A block is an ordered program

A block is a bundle of transactions plus metadata linking it to the prior chain. Its important property for MEV is that transactions execute sequentially. Transaction B reads the state left by transaction A. If A changes a pool’s reserves, B sees the new reserves. If A repays a loan, B cannot liquidate that loan afterward. If A consumes the last cheap price on one venue, B may profit from selling on another.

The block producer chooses inclusion and ordering subject to protocol rules. “Producer” is a role, not always a single business. Under proof of work it was a miner. Under proof of stake it is a proposer, usually a validator selected for a slot. In Ethereum’s current out-of-protocol proposer-builder separation, specialized builders assemble candidate blocks and a proposer selects a bid for one of them through middleware such as MEV-Boost.

Ethereum’s own PBS overview, updated June 24, 2026, describes the design as separating block building from block proposal and notes that the proposer can choose a profitable block without seeing its contents in advance. The distinction between the design goal and the deployed implementation matters: Ethereum’s enshrined PBS roadmap is still a protocol design topic, while MEV-Boost is deployed middleware. See [Ethereum’s PBS overview](https://ethereum.org/roadmap/pbs/) and the [MEV-Boost documentation](https://boost.flashbots.net/), both checked July 31, 2026.

### Gas, priority, and ordering

Gas measures the computational work a transaction asks the chain to perform. The fee attached to that work is an economic signal. In a crowded block, a transaction that pays a larger priority fee can be more attractive to include quickly. But gas price is only one ordering input. A builder may prefer a bundle with a direct payment, an arbitrage profit, or a liquidation reward even if the individual transactions are not the highest-fee transactions in the public pool.

This is why “front-running” is an imperfect umbrella. A searcher may submit a higher-fee transaction to win an earlier position in a public fee auction. A builder may receive a private bundle containing a complete sequence. A proposer may select the candidate with the highest bid. The economic objective is similar—obtain a profitable ordering—but the mechanism and the parties differ.

### What MEV means

MEV originally meant miner extractable value. After Ethereum moved from proof of work to proof of stake in September 2022, “maximal extractable value” became the more general term. It is the value available because a block producer or its supply-chain partners can include, exclude, or reorder transactions.

The word “maximal” is aspirational, not a guarantee that anyone found the mathematically best block. A searcher’s simulation can be wrong. Another searcher can bid away the opportunity. A transaction can revert. A builder can reject a bundle. A proposer can miss the slot. MEV is an opportunity set and an auction, not a free money machine.

## The players: searchers, builders, relays, and proposers

MEV is easier to understand as a supply chain than as a bot.

### Searchers find state-dependent opportunities

Searchers monitor pending transactions, on-chain state, price differences, lending positions, and protocol events. They simulate candidate sequences and submit transactions or bundles that attempt to capture a spread. A searcher might be an independent trading firm, an exchange, a protocol-owned service, or a builder with its own internal strategy.

The searcher’s essential input is not merely speed. It is a model of how a sequence changes state. A profitable arbitrage that exists now may vanish after one unrelated trade. A liquidation may be available only until a borrower adds collateral. A sandwich may work only if the victim’s maximum slippage leaves enough room after pool fees and gas.

This is an analytical description, not an invitation to run an extraction system. The same mechanics that make MEV measurable also make it adversarial. The defensive question is always: what information about my transaction is exposed, and how much of the possible surplus is someone else allowed to take?

### Builders assemble candidate blocks

Builders collect ordinary transactions, searcher bundles, and their own opportunities. They simulate the combined block and rank candidates by the payment they can deliver to the proposer while satisfying validity constraints. A builder is therefore an auctioneer and a compiler: it turns many possible sequences into one executable block.

The builder wants the highest value block it can safely construct. That value can come from ordinary priority fees, arbitrage, liquidations, bundle payments, or other state-dependent transactions. The builder may keep a margin, pay searchers for order flow, and bid the rest to the proposer.

### Relays provide a trust boundary

In the MEV-Boost design, a relay sits between a builder and a proposer. The relay checks that a builder’s payload is valid and communicates a bid and a blinded block header to the proposer. The proposer can choose the highest bid without receiving the full transaction list before signing the header. The relay then reveals the payload.

The relay is not a magical privacy layer for retail users. It is a coordination and verification layer in the builder-proposer market. Private order flow may enter through other paths, and a builder can still learn what it needs to build its block. PBS reduces the amount of specialist machinery every validator must operate; it does not eliminate the economic value of ordering.

### Proposers and validators finalize the choice

A proposer is the validator selected for a particular slot. It chooses a candidate block, publishes it, and is later checked and attested by other validators. Under MEV-Boost, the proposer receives the builder’s bid, which is a payment for the right to have the candidate block included.

The proposer’s role is often described as passive, but it is economically important. The proposer controls the final acceptance decision among available candidates. A proposer can also choose not to use a builder market, include its own transactions, or use a different relay configuration. The constraints are protocol validity, timing, and the validator’s incentives.

![The 12-second slot as a PBS sequence: builders bid, the proposer signs a blinded header, and validators attest](/imgs/blogs/mev-the-invisible-tax-on-every-trade-6.webp)

### Worked example: searcher, builder, and proposer payment (illustrative)

Suppose a searcher finds an executable opportunity worth $10,000 before costs. It submits a two-transaction bundle and offers the builder a $6,500 payment. The builder spends $300 in gas and engineering/relay overhead for this illustrative block, then bids $5,800 to the proposer. The numbers are invented arithmetic, not a historical payment.

1. Searcher gross opportunity: $10,000.
2. Searcher payment to builder: $6,500.
3. Searcher remaining before its own gas: $3,500.
4. Builder’s gross margin after paying the proposer: $700.
5. Builder’s margin after the assumed $300 cost: $400.
6. Proposer receives $5,800, in addition to whatever ordinary block rewards and fees apply.

The $10,000 did not appear from nowhere. It is the difference between two states: the state before the sequence and the state after it. The auction decides how that difference is divided among the searcher, builder, and proposer.

**Intuition:** MEV is a stack of bids. The party that discovers the opportunity is not automatically the party that keeps most of it.

## The most visible extraction: a sandwich

A sandwich is a three-part sequence around a victim’s swap:

1. The searcher buys the asset immediately before the victim.
2. The victim’s purchase moves the automated-market-maker price upward.
3. The searcher sells immediately after the victim at the higher pool price.

The searcher has not changed the victim’s signed minimum output. It has changed the state in which the victim executes, so the victim receives less output than it would have received absent the surrounding trades. The searcher’s two trades are the bread; the victim’s trade is the filling.

The risk is not only moral or legal language. It is arithmetic. Automated market makers have a predictable curve, and the victim’s slippage boundary tells an observer how far along that curve the victim is willing to go.

### Constant-product math

Consider a pool with reserves (x) ETH and (y) USDC. A constant-product market maker keeps (x y = k), ignoring the small fee for the first intuition. If a trader adds (Delta y) USDC, the pool’s new ETH reserve is approximately (k/(y+Delta y)), so the trader receives:

\[
\Delta x = x - \frac{k}{y + \Delta y}.
\]

With a fee, only (0.997\Delta y) enters a pool charging 0.30%, so the exact Uniswap-v2-style formula is:

\[
\Delta x = \frac{x(0.997\Delta y)}{y + 0.997\Delta y}.
\]

The formula does not say that every swap is attackable. It says that size changes price. A sandwich is profitable only if the price movement created by the victim is large enough to cover the attacker’s fees, gas, competition, and payment to the builder.

#### Worked example: a constant-product sandwich (illustrative arithmetic)

Assume an ETH/USDC pool starts with 1,000 ETH and 3,000,000 USDC. The spot price is therefore $3,000 per ETH. A victim wants to swap $50,000 USDC for ETH and sets a 1.0% minimum-output tolerance. The 0.30% pool fee applies to each swap. Every dollar here is illustrative.

**Step 1 — the victim without a sandwich.** The fee-adjusted input is (50,000\times0.997=49,850). The victim receives:

\[
\frac{1,000\times49,850}{3,000,000+49,850}=16.616\text{ ETH, approximately}.
\]

The average execution price is about $3,009.10 per ETH, already above the $3,000 spot because the trade moves the curve and pays the fee.

**Step 2 — the searcher buys first.** Suppose the searcher puts $10,000 USDC into the pool. Its fee-adjusted input is $9,970, and it receives:

\[
\frac{1,000\times9,970}{3,000,000+9,970}=3.312\text{ ETH, approximately}.
\]

The pool now holds approximately 996.688 ETH and 3,009,970 USDC.

**Step 3 — the victim trades in the changed state.** The victim’s $49,850 effective input now produces approximately:

\[
\frac{996.688\times49,850}{3,009,970+49,850}=16.235\text{ ETH}.
\]

The victim receives about 0.381 ETH less than in the no-sandwich path. At the initial $3,000 reference price, that is about $1,143 of gross output difference. It is not all attacker profit: the pool charged fees, the searcher paid gas, and the searcher must unwind.

**Step 4 — the searcher sells after the victim.** After the victim, the pool contains approximately 980.453 ETH and 3,059,820 USDC. The searcher sells its 3.312 ETH back. The fee-adjusted ETH input is (3.312\times0.997=3.302) ETH. The constant-product output is approximately:

\[
3,059,820 - \frac{k}{980.453+3.302} \approx 10,834\text{ USDC},
\]

where (k) is the post-victim reserve product. The searcher spent $10,000 and receives roughly $10,834, for gross revenue around $834 before gas and builder payment. If the combined gas and payment are $900, the sequence loses about $66 and should not be submitted. This is why the attacker’s gross opportunity and net profit are different numbers.

The victim’s quoted minimum output is the boundary that determines whether the sandwich can fit. A tighter limit may make the victim revert. It does not make the public transaction private, and it does not guarantee that the victim receives the no-sandwich output.

**Intuition:** constant-product math turns “a bot moved the price” into a balance-sheet statement: the victim’s trade moves the curve, and the attacker tries to buy that movement before and sell it after.

![Illustrative sandwich outcomes for a $50,000 swap under different slippage limits and pool depths](/imgs/blogs/mev-the-invisible-tax-on-every-trade-4.webp)

### Why slippage is not the same as MEV

Slippage is the difference between a reference quote and actual execution caused by trade size, fees, latency, and changing prices. MEV is value captured because someone can condition their action on transaction ordering or state transitions. A large swap can have slippage without a sandwich. A sandwich can be small even when the pool has significant depth. The two interact, but they are not synonyms.

The honest mental model is: your slippage tolerance is a maximum acceptable execution loss relative to the quote, not an amount a searcher is entitled to collect. If a route is public and the pool is shallow, however, the tolerance can become a very visible target.

## Backruns and arbitrage: extraction that can repair a market

The word MEV includes activities that are not obviously harmful to the initiating trader. Arbitrage is the simplest case. If ETH is priced at $3,000 in one pool and $3,030 in another, a trader can buy in the cheaper pool and sell in the more expensive one. The trades push the cheap venue upward and the expensive venue downward, reducing the gap.

A backrun is a transaction deliberately placed after a known transaction because the first transaction creates a predictable state change. If a large swap moves a pool away from the price on another venue, a backrunner can trade after it to restore parity. The large trader may pay the cost through price impact, but the backrunner’s activity can improve the next quote for everyone else.

### Worked example: cross-venue arbitrage (illustrative)

Suppose a token trades at $100.00 on Venue A and $100.80 on Venue B. An arbitrageur buys 1,000 tokens on A and sells 1,000 on B.

1. Gross purchase cost on A: (1,000\times100.00=\$100,000).
2. Gross sale proceeds on B: (1,000\times100.80=\$100,800).
3. Gross spread: $800.
4. Assume A and B together charge 0.20% of notional: (0.002\times200,800=\$401.60).
5. Assume gas and builder payment total $180.
6. Net illustrative profit: (800-401.60-180=\$218.40).

The arbitrageur earns $218.40, but the market receives a service: the next buyer sees a smaller cross-venue discrepancy. If the arbitrageur must pay $700 for priority, the trade becomes unprofitable and disappears. Competition compresses the opportunity.

**Intuition:** useful MEV is often a paid repair job. The repair is valuable, but the fee for doing it still comes from somebody’s price impact or liquidity provision.

### Extractive versus useful MEV

The categories are not morally perfect, but they are analytically helpful.

**Extractive MEV** worsens a user’s outcome without providing a corresponding market function. A sandwich is the canonical example: the victim pays a worse price so the searcher can round-trip around the victim. A malicious censoring or time-bandit reorganization would be more severe because it attacks settlement rather than merely execution.

**Useful or protective MEV** can maintain a protocol’s solvency or price consistency. Arbitrage aligns venues. Liquidations close loans whose collateral no longer covers debt. Some backruns absorb a predictable imbalance after a trade. The opportunity is still extractable value, but removing it entirely could leave pools stale or lending protocols with bad debt.

The boundary can be contested. An arbitrageur may improve the next quote while charging liquidity providers for being slow. A liquidation bonus keeps keepers available but makes the borrower’s loss larger. A backrun may be benign in one protocol and harmful in another. MEV is not a moral label; it is a map of who can change the state and who pays for the change.

![A matrix separating sandwiches, arbitrage, liquidations, backruns, and reorg attempts by payer and system usefulness](/imgs/blogs/mev-the-invisible-tax-on-every-trade-3.webp)

## Liquidations: the bounty that prevents bad debt

Lending protocols let a borrower deposit collateral and borrow another asset. The protocol needs a rule for when the collateral is no longer worth enough. A liquidation repays some or all of the debt and sells or claims collateral, usually giving the liquidator a bonus.

The bonus is an MEV opportunity because the right to liquidate is time-sensitive. Many keepers watch health factors, submit a transaction, and compete for priority. The borrower pays through the bonus and sometimes through a penalty. The protocol receives a solvent position instead of a loan whose collateral is worth less than its debt.

### Worked example: liquidation bonus (illustrative)

Assume a borrower owes 10,000 USDC and has ETH collateral. The protocol permits liquidation when the position breaches its threshold and pays a 5% liquidation bonus on the debt repaid. A keeper repays the full 10,000 USDC and receives collateral worth (10,000\times1.05=\$10,500) at the protocol’s execution price.

1. Keeper advances: $10,000 USDC.
2. Collateral received: $10,500 market value, illustratively.
3. Gross liquidation incentive: $500.
4. Gas, price impact, and builder payment: assume $180.
5. Net keeper surplus: $320.
6. Borrower’s collateral shortfall relative to debt is reduced by the liquidation process; the borrower’s economic cost includes the $500 bonus and any remaining deficit under the protocol’s rules.

If the keeper payment rises to $600 because several keepers compete for priority, this particular liquidation is no longer profitable. If the collateral price falls another 4% before settlement, the keeper may lose money. The bonus is a risk budget, not a guaranteed return.

**Intuition:** a liquidation bounty is an insurance premium paid by an unhealthy borrower to keep a protocol from socializing bad debt across everyone else.

Liquidation MEV can still become predatory when protocols expose large, predictable positions, when oracle updates arrive in bursts, or when a keeper’s transaction crowds out a user’s attempt to add collateral. The defense is mostly protocol design: robust oracles, partial liquidation, auction formats, and limits on how much collateral can be seized at once.

## Priority ordering and the old priority-gas auction

Before private bundles became common, searchers often competed by repeatedly increasing gas prices. This was called a priority gas auction, or PGA. The 2019 paper [Flash Boys 2.0](https://arxiv.org/abs/1904.05234) documented arbitrage bots and described PGAs as a continuous game for priority ordering. Its title and findings are historical research, not a claim that every modern block uses the same mechanism.

The basic arithmetic is a first-price auction. Suppose an opportunity is worth $1,000 to a searcher. Searcher A bids $700 in fees, Searcher B bids $850, and Searcher C bids $950. If C wins and no other costs apply, C keeps $50. If the proposer receives only the fee and the winner also pays $100 of gas overhead, C loses $50. A rational searcher bids up to its private value, but mistakes and latency make overbidding common.

Under builder markets, the auction can move one level outward. Searchers bid to builders; builders bid to proposers. The transaction’s visible gas fee is not necessarily the complete payment for preferred ordering. A bundle can include an explicit payment transaction, a bribe-like transfer, or a high-value state change whose surplus is shared with the builder.

### Worked example: priority ordering (illustrative)

Suppose three candidate bundles all fit in the next block:

| Bundle | Searcher opportunity | Payment to builder | Builder costs | Bid to proposer |
| --- | ---: | ---: | ---: | ---: |
| A | $1,000 | $650 | $50 | $500 |
| B | $900 | $700 | $40 | $620 |
| C | $1,300 | $800 | $100 | $650 |

The builder compares its own residual after the proposer bid and costs. A leaves (650-50-500=\$100). B leaves (700-40-620=\$40). C leaves (800-100-650=\$50). A builder maximizing its own margin chooses A, while a builder maximizing the proposer’s bid chooses C. Real builders optimize a more complicated objective involving reliability, block fullness, and relationships. The example shows why “highest opportunity” and “highest proposer payment” are not the same.

**Intuition:** ordering is a multi-stage auction; the visible gas price is only one bid in a chain of bids.

## PBS and MEV-Boost: what changed, what did not

Proposer-builder separation addresses a structural problem. If every validator had to run the most sophisticated search and block-construction system, economies of scale could push staking toward a small number of professional operators. PBS lets specialized builders compete to construct blocks while many validators retain the proposal and verification roles.

MEV-Boost is Flashbots’ implementation of this idea for proof-of-stake Ethereum. Its documentation describes validators selling block space to an open builder market to maximize staking reward. The proposer receives bids, not necessarily the builder’s complete transaction list before signing. This is an important operational distinction from a proposer directly running every searcher strategy.

![PBS does not remove ordering power; it routes it through searchers, builders, relays, and proposers](/imgs/blogs/mev-the-invisible-tax-on-every-trade-5.webp)

PBS changes the distribution and concentration of MEV:

- Searchers specialize in discovery and simulation.
- Builders specialize in block assembly and bidding.
- Relays mediate delivery and validity checks.
- Proposers sell the final slot decision.
- Stakers receive the proposer-side reward, directly or through a pool.

It does not make a public swap private, guarantee fair ordering, or prevent a builder from learning the transaction’s economic consequences. It also introduces new concentration questions. If a small group of builders wins most valuable blocks, they may gain informational and political power even if proposers remain numerous. The Ethereum PBS roadmap explicitly discusses the tradeoff between specialized block building and decentralized validation; the roadmap page was checked July 31, 2026.

### Why builders may accept or reject sandwiches

A builder has a policy choice. It can accept all valid profitable bundles, reject clearly harmful patterns, or participate in a protected-order-flow ecosystem. Its choice is constrained by competition: rejecting a profitable bundle can lower its bid and lose the slot. A builder may also receive private order flow from wallets, RPCs, wallets, or protocols that expect certain treatment.

This is why the phrase “Flashbots protects users” needs precision. Flashbots offers products and infrastructure, but a builder market is not identical to a universal user-protection rule. Read the specific endpoint’s guarantees, refund policy, chain coverage, and failure behavior. Do not infer a property from a brand name.

## A named case study: the Peraire-Bueno indictment

On May 15, 2024, the United States Department of Justice announced charges against Anton Peraire-Bueno and James Peraire-Bueno and described an alleged theft of approximately $25 million in cryptocurrency through a technologically sophisticated Ethereum scheme. The DOJ release says the brothers were indicted for conspiracy to commit wire fraud, wire fraud, and money laundering. The [DOJ announcement](https://www.justice.gov/archives/opa/pr/two-brothers-arrested-attacking-ethereum-blockchain-and-stealing-25m-cryptocurrency) and the indictment are the primary sources for what prosecutors alleged.

The alleged mechanism is relevant because it was not a normal sandwich. Prosecutors alleged that the defendants studied MEV bots, prepared lure transactions, induced victim bots to construct bundles, and then used validator access and transaction ordering to manipulate what those bundles executed. The central lesson is structural: when someone can control the transaction sequence at the moment of proposal, a bot’s assumptions about atomic execution can become a vulnerability.

Three cautions matter.

First, an indictment is an allegation, not evidence of guilt beyond a reasonable doubt and not a conviction. Second, the case illustrates a category of ordering abuse; it does not prove that ordinary arbitrage or every private bundle is criminal. Third, this post is frozen as of July 31, 2026 and makes no claim about a final judgment. Readers should consult the federal docket for later procedural developments.

The case also shows why “MEV” is too broad to be a legal conclusion. Searchers routinely submit bundles that depend on ordering. Builders routinely select among bundles. The alleged conduct, if proven, concerns deception and manipulation of another system’s assumptions—not the mere existence of a profit-seeking transaction sequence.

## How it shows up in price

MEV is often invisible in a chart because the chart shows the final price, not the counterfactual price without the ordering event. Look for symptoms rather than a single magic signature.

### A worse-than-quote swap

Your wallet quotes 10.000 units of an asset for $30,000. The transaction settles with 9.850 units while the visible market price appears almost unchanged a second later. Possible causes include pool fees, normal price movement, route differences, stale quotes, or a sandwich. A worse fill is evidence of execution friction, not proof of extraction.

### A brief price spike around a large swap

The price jumps immediately before a large public swap and retraces immediately after it. This is consistent with a sandwich pattern, but it can also be ordinary order-flow response, an oracle update, or a large arbitrage. Attribution requires transaction-level analysis and a counterfactual.

### Cross-venue convergence after a gap

A token trades at $100 on one venue and $100.80 on another, then both converge near $100.30 after a few blocks. That is the signature of arbitrage doing useful work: the spread is consumed and liquidity providers absorb the inventory change.

### A loan closes at a sharp discount

During a fast market, a lending protocol’s liquidation transaction sells collateral below the last displayed price. The gap may reflect a liquidation bonus, auction clearing, price impact, or an oracle lag. The borrower’s loss is not automatically abusive; the protocol may have designed the bonus to attract a keeper precisely when risk is highest.

### Priority fees jump without a broad market move

If a narrow set of transactions suddenly pays unusually high fees while a particular pool or contract changes state, there may be a time-sensitive opportunity. Researchers can investigate; retail traders should not interpret the fee spike alone as a buy signal.

<figure class="blog-anim">
<svg viewBox="0 0 900 260" role="img" aria-label="A victim transaction is placed between a searcher buy and searcher sell inside one block" style="width:100%;height:auto;max-width:900px">
<style>
.m1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.m1-search{fill:#ffc9c9;stroke:#b91c1c;stroke-width:2}.m1-victim{fill:#ffec99;stroke:#92400e;stroke-width:2}.m1-block{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:8 8}.m1-lbl{font:600 19px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.m1-small{font:500 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.m1-dot{fill:var(--accent,#6366f1)}
@keyframes m1-run{0%,12%{transform:translateX(0);opacity:0}22%{opacity:1}42%,100%{transform:translateX(690px);opacity:1}}@keyframes m1-pulse{0%,35%{opacity:.25}45%,70%{opacity:1}80%,100%{opacity:.25}}.m1-move{animation:m1-run 8s ease-in-out infinite}.m1-highlight{animation:m1-pulse 8s ease-in-out infinite}@media (prefers-reduced-motion:reduce){.m1-move{animation:none;transform:translateX(690px);opacity:1}.m1-highlight{animation:none;opacity:1}}
</style>
<rect class="m1-block" x="25" y="25" width="850" height="210" rx="14"/><text class="m1-small" x="450" y="52">one block · ordered execution</text><rect class="m1-search" x="70" y="90" width="200" height="82" rx="12"/><text class="m1-lbl" x="170" y="124">Searcher buy</text><text class="m1-small" x="170" y="151">before victim</text><rect class="m1-victim m1-highlight" x="350" y="90" width="200" height="82" rx="12"/><text class="m1-lbl" x="450" y="124">Victim swap</text><text class="m1-small" x="450" y="151">moves the pool price</text><rect class="m1-search" x="630" y="90" width="200" height="82" rx="12"/><text class="m1-lbl" x="730" y="124">Searcher sell</text><text class="m1-small" x="730" y="151">after victim</text><circle class="m1-dot m1-move" cx="80" cy="205" r="9"/><text class="m1-small" x="450" y="220">the highlighted victim is bracketed by two ordered trades</text>
</svg>
<figcaption>The same block order that settles the victim can place a searcher’s buy before it and sell after it.</figcaption>
</figure>

The animation is intentionally schematic. It teaches the sequence, not a recipe for constructing an attack.

## Defensive tools: reduce exposure without pretending risk disappears

There is no universal “MEV-proof” switch. Each defense changes the information available, the execution venue, the latency, or the counterparty risk. Choose based on trade size and sensitivity.

![A defense ladder comparing slippage, protected RPC, batch auctions, and intent-based execution](/imgs/blogs/mev-the-invisible-tax-on-every-trade-8.webp)

### 1. Set a deliberate slippage limit

Slippage tolerance is the most basic control. It says: do not execute if the output falls below this boundary. For a liquid pair, a very wide default can donate unnecessary room to a sandwich. For an illiquid token, an extremely tight limit may cause frequent reverts.

Use the quoted route, pool depth, fee tier, and current volatility to choose a boundary. A limit is not a forecast and not a guarantee. It protects the worst acceptable output, while a private or auction-based path addresses the visibility problem.

### 2. Prefer private or protected order flow when appropriate

A private RPC sends a transaction to a selected endpoint or builder path instead of broadcasting it to the ordinary public gossip pool. Flashbots Protect describes its service as allowing users to control transaction processing and choose preferences around speed, privacy, and refunds; see [Flashbots Protect](https://protect.flashbots.net/start), checked July 31, 2026.

The tradeoff is trust and coverage. You rely on the endpoint and downstream builders not to expose or misuse the order, and a private transaction may be delayed, dropped, or behave differently across chains. A private endpoint also does not protect against every form of adverse execution. Read its policies.

### 3. Use batch auctions for orders that can wait

In a batch auction, orders arriving during a window are cleared together, often at a uniform or jointly determined price. This reduces the advantage of being first within the batch. CoW Protocol describes itself as using fair combinatorial batch auctions for price discovery; its documentation explains that off-chain intents are aggregated and auctioned to solvers. See [CoW Protocol documentation](https://docs.cow.fi/), checked July 31, 2026.

The cost is latency and design complexity. A batch can miss a rapidly moving price, and solver competition introduces a different trust and execution model. For a $200 swap, waiting may not matter. For a $200,000 swap, avoiding a predictable public-mempool exposure may matter a great deal.

### 4. RFQ and intent-based execution

An intent is a signed statement of desired outcome rather than a fully specified public transaction sequence. “Sell up to this amount for at least this amount before expiry” gives a filler or solver room to find liquidity. An RFQ—request for quote—asks designated counterparties to compete on a price.

Intent systems move the auction earlier. Instead of showing every observer a transaction that will hit a particular pool, the user shows a set of constraints and lets fillers compete to satisfy them. The filler can earn a spread, but it competes on the user’s outcome.

UniswapX’s official overview says swappers create orders defining auction parameters and price tolerance; its order messages are broadcast and can be filled by fillers. The [UniswapX overview](https://developers.uniswap.org/docs/liquidity/uniswapx/overview) and [filler documentation](https://developers.uniswap.org/docs/liquidity/uniswapx/filling/overview) were checked July 31, 2026. This is not “no MEV.” It is a different auction in which the desired output and execution rules are the object of competition.

### 5. Do not confuse a private path with a better price

Privacy and price improvement are separate axes. A private RPC may prevent a public sandwich while still routing you through a costly pool. A batch auction may provide a uniform price while a fast-moving market moves against you. An intent filler may protect your minimum output while earning a spread.

Ask four questions before approving a route:

- Is my signed order public before it fills?
- Who can fill or reorder it?
- What is the minimum guaranteed output, and when does it expire?
- Who receives any surplus: me, a solver, a liquidity provider, or a protocol treasury?

### Worked example: choosing between public and protected execution (illustrative)

Suppose a $25,000 swap has a quoted expected output of $25,000 in the destination asset. The public route has a 0.30% pool fee and an illustrative expected MEV loss of 0.40%, while the protected route charges a 0.10% routing fee and has a 0.15% probability of missing the intended block. These probabilities are invented for arithmetic, not measurements.

Public expected friction: (0.30\%+0.40\%=0.70\%), or (\$175).

Protected expected fee: (0.10\%\times\$25,000=\$25). If a missed block costs an illustrative $120 in adverse movement and the miss probability is 0.15%, expected miss cost is (0.0015\times120=\$0.18). Total expected protected friction is about $25.18.

The protected route looks better in this toy model, but the conclusion depends entirely on the assumed MEV loss, miss cost, chain, endpoint, and route. The useful calculation is to compare expected outcomes, not to treat “private” as a magic adjective.

**Intuition:** a defense is worth its fee when the execution risk it removes is larger than the new risk it introduces.

## What protocols can do

Retail settings matter, but protocol architecture determines the size of the tax.

### Hide information until commitment

Commit-reveal designs, encrypted mempools, threshold decryption, and batch settlement can keep a transaction’s details hidden until the ordering commitment is fixed. Each adds timing, liveness, cryptographic, or censorship-resistance tradeoffs. “Hide everything” is not free: validators still need enough information to verify and execute in time.

### Make ordering less valuable

Uniform-price clearing, frequent batch auctions, and solver competition reduce the value of being the first transaction in a sequence. If orders in the same batch share a clearing price, the sandwich pattern loses its most important advantage: controlling the exact before-and-after state around one victim.

### Build safer AMM interfaces

Protocols can limit toxic flow with dynamic fees, concentrated-liquidity design, trade-size warnings, oracle checks, and route-level simulation. None is perfect. A dynamic fee that rises during volatility may protect LPs but increase a user’s cost. A route simulator can be stale. A warning can be ignored.

### Design liquidations as auctions, not races

Dutch auctions, partial liquidations, and backstop liquidity can reduce the advantage of the fastest keeper. The goal is to pay enough to attract execution while giving multiple participants a fair chance. The right parameter depends on collateral volatility, oracle latency, and how much bad debt the protocol can tolerate.

### Treat builder concentration as a security question

MEV is not only a trader-protection issue. If builders become highly concentrated, they can gain censorship leverage, privileged information, and influence over which application designs survive. PBS is intended to separate specialized block construction from broad validation, but the concentration of builders, relays, and private order flow remains a live design concern. Ethereum’s PBS roadmap and EIP-7732 draft discuss these tradeoffs; [EIP-7732](https://eips.ethereum.org/EIPS/eip-7732) is a draft, not a finalized protocol rule, as checked July 31, 2026.

## Common misconceptions

### “All MEV is theft”

No. A sandwich is a compelling example of value taken from a trader without a useful balancing service. Arbitrage can restore price parity. Liquidation can protect lenders and depositors from bad debt. The right question is who pays, what state improvement occurs, and whether the extraction is necessary for the market to function.

### “The highest gas price always wins”

No. Builders evaluate complete candidate blocks and bundles. A lower visible gas transaction can be part of a more valuable sequence. Private order flow and explicit builder payments can matter. A high gas bid can still lose because the transaction arrives late, reverts, conflicts with another bundle, or does not fit.

### “A failed transaction means the bot stole my funds”

Usually not. A reverted transaction generally does not apply its intended state change, though the user may still pay gas for computation. Failure can result from stale quotes, nonce conflicts, contract conditions, gas limits, or a competitor changing state first. Inspect the receipt and trace before assigning blame.

### “Private RPC means guaranteed protection”

No. It reduces exposure to the public mempool. It introduces trust in the endpoint and its connected builders, and it may not cover every chain or route. A private path can also fail to land promptly.

### “A solver is just a benevolent router”

No. A solver or filler is a market participant. It competes to satisfy your constraints and may earn a spread, a fee, or surplus. That can still be a good trade if the outcome is better than the alternatives. Read the surplus and fee rules.

### “MEV only exists on Ethereum”

No. Any system with ordered state transitions, scarce block space, and participants able to influence inclusion can produce analogous extraction. The roles and terminology differ across chains, rollups, order books, and app-specific sequencing systems.

### “A chart proves a sandwich”

No. A sandwich is a transaction-level hypothesis. You need the victim transaction, the before and after trades, pool reserves, fees, ordering, and a plausible counterfactual. A wick on a chart is not a forensic conclusion.

## A retail checklist

Before approving an on-chain swap, especially one large relative to pool depth:

1. Check the route and the pool fee.
2. Set a minimum output you can defend, not an arbitrary wide default.
3. Ask whether the order enters a public mempool.
4. Prefer a protected or private route for sensitive swaps when the endpoint’s policy is acceptable.
5. Consider a batch auction or RFQ/intent route for larger trades.
6. Compare the quoted output with the guaranteed output and the expiry.
7. After execution, inspect the transaction’s position and surrounding state changes if the fill looks unusual.
8. Do not paste seed phrases or private keys into any “MEV bot” website. The defensive tool is an execution path, never a program that asks for custody.

This checklist does not turn a volatile market into a safe one. It changes the information and ordering game in your favor. The goal is not to eliminate every fee or every adverse price move. The goal is to stop donating a predictable, avoidable margin to a participant who saw your order before it settled.

## Looking ahead: the next layer of the machine

Wave 8 will move from mechanism to forensic practice: tracing wallet clusters, distinguishing observed flow from inferred intent, and identifying where public data stops being enough. That work needs a higher evidentiary standard than a chart screenshot. A wallet can be labelled, but ownership can be uncertain; a repeated pattern can be observed, but motive can remain contested.

The broader series continues with [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move), [cross-exchange arbitrage and the latency game](/blog/trading/crypto-players/cross-exchange-arbitrage-and-the-latency-game), [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does), [liquidations and lending mechanics](/blog/trading/onchain/analyzing-lending-and-liquidations), [AMMs versus arbitrageurs](/blog/trading/game-theory/amms-vs-arbitrageurs-toxic-flow-and-loss-versus-rebalancing), and [every market is an auction](/blog/trading/game-theory/every-market-is-an-auction-the-double-auction-of-the-order-book). For the consensus-side primer, return to [Mining, Staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev).

## Sources & further reading

- [Flash Boys 2.0: Frontrunning, Transaction Reordering, and Consensus Instability in Decentralized Exchanges](https://arxiv.org/abs/1904.05234), Daian et al., April 10, 2019. Historical research on DEX arbitrage bots and priority gas auctions.
- [Maximal extractable value](https://ethereum.org/developers/docs/mev/), Ethereum.org, checked July 31, 2026. Protocol vocabulary and MEV context.
- [Proposer-builder separation](https://ethereum.org/roadmap/pbs/), Ethereum.org, page updated June 24, 2026 and checked July 31, 2026. PBS design, benefits, and centralization tradeoffs.
- [MEV-Boost in a Nutshell](https://boost.flashbots.net/), Flashbots, checked July 31, 2026. Deployed PBS middleware and builder/proposer roles.
- [MEV-Boost repository](https://github.com/flashbots/mev-boost), Flashbots, checked July 31, 2026. Open-source implementation context.
- [EIP-7732: Enshrined Proposer-Builder Separation](https://eips.ethereum.org/EIPS/eip-7732), draft, checked July 31, 2026. Future protocol design; not treated here as finalized behavior.
- [Two Brothers Arrested for Attacking Ethereum Blockchain and Stealing $25M](https://www.justice.gov/archives/opa/pr/two-brothers-arrested-attacking-ethereum-blockchain-and-stealing-25m-cryptocurrency), U.S. Department of Justice, May 15, 2024. Primary source for the Peraire-Bueno indictment and prosecution allegations; charges are not convictions.
- [Flashbots Protect](https://protect.flashbots.net/start), Flashbots, checked July 31, 2026. User-facing protected transaction-order-flow documentation.
- [CoW Protocol documentation](https://docs.cow.fi/), checked July 31, 2026. Fair combinatorial batch auctions and solver-based execution.
- [UniswapX overview](https://developers.uniswap.org/docs/liquidity/uniswapx/overview) and [filler documentation](https://developers.uniswap.org/docs/liquidity/uniswapx/filling/overview), Uniswap Developers, checked July 31, 2026. Intents, Dutch auctions, and filler roles.
- [Uniswap v2 whitepaper](https://uniswap.org/whitepaper.pdf), Uniswap, 2020. Constant-product exchange formula and fee-adjusted swap mechanics used for the illustrative walkthrough.
