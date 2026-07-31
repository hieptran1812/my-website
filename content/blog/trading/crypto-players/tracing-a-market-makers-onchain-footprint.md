---
title: "Tracing a Market Maker's On-Chain Footprint"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "A defensive, build-from-zero method for labeling a market maker's CEX deposits, DEX liquidity, hot wallets and settlement flows, then reading cadence and inventory without pretending the chain reveals intent."
tags: ["crypto", "market-makers", "onchain-analysis", "wallet-labeling", "cex", "dex", "liquidity", "market-microstructure", "crypto-players", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — A market maker is rarely one wallet. The useful on-chain question is not “which address belongs to the firm?” but “which addresses behave like one operating system, and what does each role imply?”
>
> - Start with a candidate label, then verify it against contract behavior, funding links, repeated routes, and exchange-side evidence.
> - Separate CEX deposit addresses, DEX LP positions, hot wallets, and settlement wallets before adding balances. The same token in each bucket means something different.
> - Deposit/withdrawal cadence can reveal rebalancing, inventory transfers, or distribution pressure. It cannot, by itself, prove a directional bet, common control, or manipulation.
> - Use Arkham, Nansen, Etherscan, Dune, Bubblemaps, and DefiLlama as dated evidence tools—not as magic ownership or intent detectors.
> - The number to remember is not a market-maker balance. It is **one timestamp**: every inventory estimate must say when it was measured, on which chain, and from which source.

You see a token move from a project treasury to a wallet that an explorer labels “Binance deposit.” A few hours later, the same token appears in a concentrated-liquidity pool. A week after that, part of it returns to a wallet that never touches the public order book. Is the market maker selling? Hedging? Rebalancing? Holding project inventory? Moving a client’s assets? Or are you looking at three unrelated addresses that happen to share a token?

That ambiguity is the point of this article. On-chain data is transparent about transactions and remarkably opaque about human organization. A blockchain can show that address A sent 60,000 tokens to address B at a particular block. It cannot, by itself, show whether A and B have the same owner, whether the transfer was a loan or a sale, whether the tokens were sold on a centralized exchange, or whether anyone intended to move price.

The method below is therefore a tracing method, not a doxxing method. We build a defensible evidence table, classify wallet roles, reconcile token and stablecoin flows, read cadence, and state confidence with caveats. The mental model is simple: a professional market maker’s footprint is a set of operational surfaces connected by inventory plumbing.

![A market maker's footprint is a connected operating system of treasury allocations, hot-wallet activity, DEX liquidity, CEX deposits and settlement flows—not one address.](/imgs/blogs/tracing-a-market-makers-onchain-footprint-1.webp)

The diagram above is the map for the rest of the post. A token can arrive from a project treasury, pass through a hot wallet, enter a DEX LP position, reach a centralized-exchange deposit address, and later be netted through a settlement wallet. Each hop answers a different question. The route tells us more than the largest balance.

This post enriches [Wintermute: Inside Crypto's Algorithmic Powerhouse](/blog/trading/crypto-players/wintermute-the-algorithmic-powerhouse) and [DWF Labs: The Controversial Newcomer](/blog/trading/crypto-players/dwf-labs-the-controversial-newcomer). Those profiles explain why the firms matter. This one is the field notebook for reading a footprint without overclaiming.

## Foundations: the building blocks

### What a wallet is—and what it is not

A **wallet address** is a public identifier on a blockchain. It can be controlled by a person, a company, a smart contract, a custodian, a bridge, an exchange, or nobody in the usual sense. A **wallet label** is a tag supplied by an explorer, analytics company, exchange disclosure, research team, or community. Labels are useful indexes, not deeds of ownership.

An **externally owned account** (EOA) signs transactions with a private key. A **smart contract address** executes code when called. A DEX pool is usually a contract, so a transfer into a pool is not the same thing as a transfer to a trader. A multisig adds another layer: several keys may be required to approve a transaction, and those signers may not all be employees of the same firm.

An **address cluster** is a group of addresses that an analyst believes may share an operator or operating purpose. The belief can come from common funding, identical transaction patterns, a public disclosure, a tagged exchange relationship, or a combination. A cluster is a hypothesis that should become more or less credible as new evidence arrives.

### What a market maker does

A **market maker** is a trading firm that continuously offers to buy and sell an asset. The buyer pays the **ask**, the seller receives the **bid**, and the gap between them is the **bid-ask spread**. A market maker provides immediacy: another trader can transact now instead of waiting for a natural counterparty.

The maker is not supposed to earn only from the token going up. It tries to earn spread and fee income while managing **inventory**, the net position left after customers trade against its quotes. If customers buy tokens from the maker, the maker becomes shorter in tokens or less long. If customers sell tokens to the maker, it accumulates tokens. The desk may hedge that exposure elsewhere, move assets between venues, or widen its quotes.

In crypto, the same organization may provide liquidity on a centralized exchange, seed a DEX pool, execute OTC trades, receive a token loan, and settle with a project. Those are different activities with different observable traces. A CEX deposit often means “make assets available to a venue,” not “sell immediately.” A DEX LP position is a paired inventory strategy, not a simple wallet balance. A settlement wallet may be deliberately quiet between netting cycles.

### The units that make a trace readable

We need four quantities.

- **Token balance:** units of the asset at an address or contract position.
- **Stablecoin balance:** a dollar-like crypto asset such as USDC or USDT. It is a rough accounting unit, not a guarantee of one dollar in every market or jurisdiction.
- **Notional value:** token units multiplied by a chosen price at a chosen time. It is an estimate of exposure, not cash realized.
- **Net flow:** inflows minus outflows over a stated interval. Positive net flow means more arrived than left; negative net flow means more left than arrived.

The illustrative phrase “the firm owns $5 million of tokens” hides at least four choices: which addresses count, whether LP positions are included, which price is used, and what time the snapshot represents. A good trace makes those choices visible.

#### Worked example: a one-wallet balance is not inventory

*Illustrative numbers.* Suppose a candidate address contains 100,000 tokens. The token’s chosen snapshot price is $2.00. The raw mark is:

1. Token units: 100,000.
2. Snapshot price: $2.00 per token.
3. Raw notional: 100,000 × $2.00 = **$200,000**.

That arithmetic is correct but incomplete. Suppose the address sends 60,000 tokens to a labeled CEX deposit address, 25,000 tokens to a DEX pool, and retains 15,000 tokens. The same 100,000 tokens now have three operational destinations. The retained 15,000 are not automatically “the firm’s directional bet”; they may be gas-funded float, an unfilled allocation, or a buffer for the next transfer.

The intuition: a balance is a photograph. Inventory is a dated, role-adjusted interpretation of a sequence.

## 1. Start with a candidate, not a conclusion

The first mistake in wallet tracing is beginning with a story: “this is the market maker’s dump wallet.” Start with a candidate address and a falsifiable question instead. For example: “Did this address repeatedly move the project token into a venue-linked deposit address after receiving inventory from the project?” That question can be answered with blocks and transaction hashes. “Is the firm bearish?” cannot be answered from a transfer alone.

### Discovery tools and what each one contributes

**Arkham** is useful for entity labels, address relationships, and cross-chain visualization. Treat the label as a lead. Open the underlying transactions and record the chain, block time, asset, amount, destination, and source shown by the interface.

**Nansen** is useful for wallet labels, token flows, smart-money dashboards, and portfolio history. Its convenience is valuable, but the label is still an attribution model. Check the methodology and the underlying transaction before writing “owned by.”

**Etherscan** is the primary inspection surface for Ethereum transactions and contract calls. Read the token-transfer tab, the internal transactions, the contract address, and the event logs. For a DEX LP position, inspect the pool contract and position manager rather than counting only the wallet’s visible token balance.

**Dune** is useful when you need a reproducible SQL query over decoded blockchain tables. Save the query, chain, query date, and filters. A dashboard screenshot without its query is a weak citation because the reader cannot tell whether the result includes bridges, routers, failed transactions, or proxy contracts.

**Bubblemaps** helps visualize clusters and token-holder relationships. It is particularly useful for seeing whether many wallets received funds from a common source or moved together. It does not establish legal ownership; it highlights relationships worth checking.

**DefiLlama** is useful for protocol, chain, and pool context. Use its dated data to understand whether a token’s liquidity is in a DEX pool, whether a protocol is the relevant contract, and how a move compares with the broader venue context. Do not silently treat a live dashboard number as a historical number.

The right mental model is a toolbox with different lenses. Arkham and Nansen may help you discover a relationship. Etherscan lets you inspect the transaction. Dune lets you reproduce a count. Bubblemaps helps you see a cluster. DefiLlama supplies protocol context. None of them can read an off-chain exchange account or prove intent.

![Attribution is an evidence ladder: a candidate address becomes more useful as labels, clustering, cadence and inventory reconciliation agree, while intent remains an inference.](/imgs/blogs/tracing-a-market-makers-onchain-footprint-2.webp)

### The evidence ladder

Use five levels of language:

1. **Observed:** “Address A sent 60,000 tokens to address B at block X.”
2. **Labeled:** “Etherscan or an analytics service labels B as a CEX deposit address.”
3. **Clustered:** “A and B share funding, gas, or repeated routing patterns.”
4. **Reconciled:** “The token and stablecoin movements across the candidate cluster fit a venue-rebalancing pattern.”
5. **Inferred:** “The pattern is consistent with market-making inventory management.”

Notice what is missing: “therefore the firm sold,” “therefore the firm manipulated,” or “therefore price must fall.” Those statements require evidence outside a basic chain trace, including exchange-side fills, account ownership, contracts, and sometimes legal discovery.

#### Worked example: confidence without false precision

*Illustrative numbers.* Imagine three observations about candidate address A:

1. A receives 100,000 project tokens from a treasury.
2. A sends 60,000 tokens to a venue-linked deposit address and 25,000 to a DEX pool.
3. The route repeats twice, with transfers of 50,000 and 40,000 tokens, each followed by a return of a smaller token amount to a settlement address.

You may write: “The repeated route is consistent with an operational market-making cluster.” You may not write: “The firm sold 155,000 tokens,” because the chain shows deposits, not fills. You may not write: “The firm owns 165,000 tokens,” because the DEX position, venue account, and settlement wallet may overlap economically.

The intuition: confidence should rise with independent evidence, but the sentence should never become stronger than the evidence.

## 2. Label wallet roles before adding balances

The phrase “the market maker’s wallet” is usually a category error. Different addresses are built for different jobs, and the same asset behaves differently in each job.

### CEX deposit addresses

A **centralized exchange** (CEX) maintains an off-chain trading ledger. When you deposit tokens, the blockchain sees a transfer into a deposit address, but your subsequent trades may not appear on-chain. The deposit address is therefore an entrance or exit to a private accounting system.

A transfer into a labeled deposit address can mean inventory was made available for quoting, collateral was posted, a client deposit was processed, or an operator was preparing to sell. The chain alone does not distinguish those cases. You need timing, return flows, token/stablecoin reconciliation, and—if available—exchange data.

Do not count an exchange deposit as an executed sale. It is a venue inflow. If a token later returns to the same cluster, the flow may be rebalancing. If the token never returns, the address may have distributed it inside the exchange, but that remains an inference unless the exchange ledger or a public disclosure confirms the fill.

### DEX LP positions

A **decentralized exchange** (DEX) uses smart contracts rather than a conventional central order book. In an automated market maker pool, liquidity providers deposit a pair of assets, such as a token and a stablecoin. They receive fees when trades use the pool, but their inventory changes as the pool price moves.

A simple constant-product pool maintains a relationship often written as $x \times y = k$, where $x$ and $y$ are the quantities of the two assets and $k$ is the pool’s invariant before fees. When traders buy the project token, the pool sends out some tokens and receives the paired asset. The LP’s position becomes less token-heavy. This is why a DEX LP position is not a passive token balance.

For concentrated liquidity, the position may be active only inside a chosen price range. A position can show a token balance in one contract view, a collectible position NFT in the wallet, and accrued fees elsewhere. Inspect the protocol’s position manager and pool events.

### Hot wallets

A **hot wallet** is an operational signing wallet connected to systems that need to move funds quickly. It tends to show short holding periods, many counterparties, gas top-ups, router calls, and transfers into venue addresses or pools. That pattern is compatible with a market maker, but it is also compatible with a treasury operations team, a custody service, or a protocol’s distribution wallet.

### Settlement wallets

A **settlement wallet** is used to net obligations over a slower rhythm. It may receive assets from several hot wallets, consolidate stablecoins, fund a new venue, or send a project’s tokens back after a term ends. It may be quieter and larger than the hot wallets because speed is less important than accounting.

![CEX deposits, DEX LP positions, hot wallets and settlement wallets have different signals, horizons and false positives.](/imgs/blogs/tracing-a-market-makers-onchain-footprint-4.webp)

The matrix is a reminder to keep role in the data model. A CEX deposit is not interchangeable with an LP position, even if both contain the same token. A settlement transfer is not a trade print. A hot wallet’s short residence time does not prove that every outgoing transfer was a sale.

#### Worked example: the same token, four meanings

*Illustrative numbers.* At a snapshot price of $2.00, suppose a candidate cluster contains:

- 60,000 tokens at a CEX deposit address: raw mark **$120,000**.
- 25,000 tokens in a DEX LP position: raw token-side mark **$50,000**, plus the paired asset and fees.
- 15,000 tokens in a hot wallet: raw mark **$30,000**.
- 100,000 tokens shown by a settlement report that may already include the first three buckets.

The naive sum is $120,000 + $50,000 + $30,000 + $200,000 = **$400,000**, but that can double-count the same 100,000 tokens. A role-adjusted report would state the buckets separately and say whether the settlement figure is an independent balance or a consolidated total.

The intuition: classification prevents arithmetic from turning one asset into several imaginary assets.

## 3. Read transfers as a cadence, not a headline

One transfer is an event. A **cadence** is a repeated rhythm: how often funds move, how long they stay, which destinations recur, what arrives afterward, and whether the route changes when price or liquidity changes.

The safest cadence table has one row per transaction and columns for:

| Field | What to record | Why it matters |
|---|---|---|
| Timestamp | UTC date and block time | Makes the claim reproducible |
| Chain and block | Network and block number | Prevents cross-chain mix-ups |
| Asset and units | Token contract, decimals, quantity | Avoids ticker collisions |
| From / to | Full addresses and labels | Keeps labels separate from facts |
| Function | Transfer, swap, add liquidity, bridge | Shows what actually happened |
| Venue | CEX deposit, DEX pool, router | Separates operational surfaces |
| Price source | Chosen price and timestamp | Makes notional calculation auditable |
| Return leg | Token or stablecoin back to cluster | Helps identify netting patterns |

The table should include failed transactions when they explain gas behavior or a broken route, but do not count failed transfers as successful inventory movement. Bridges require special care: the source-chain burn or lock and destination-chain mint or release can make one economic movement look like two transfers.

![A recurring deposit-then-withdrawal rhythm can reveal venue rebalancing without proving the trader's directional view.](/imgs/blogs/tracing-a-market-makers-onchain-footprint-3.webp)

### Cadence patterns worth testing

**Short residence:** tokens arrive in a hot wallet and leave within a short interval. This is compatible with operational routing or pre-positioning. It is weak evidence of a sale.

**Periodic consolidation:** many smaller transfers arrive at a settlement wallet, followed by one larger move to a venue or project. This may reflect netting, gas management, or a scheduled obligation.

**Two-way venue flow:** tokens go to a CEX and later some tokens return while stablecoins move elsewhere. This is compatible with inventory rebalancing, but it does not tell you whether the trader made or lost money.

**LP drift:** the token side of a pool position falls while the paired stablecoin rises. That can mean traders bought the token from the pool. It does not identify who bought, and it does not prove the LP operator wanted the token price lower.

**Unlock-linked flow:** transfers cluster around a dated token unlock or treasury event. This raises a relevant question about supply, but timing alone cannot prove that a particular firm sold the unlocked tokens.

#### Worked example: net flow versus gross flow

*Illustrative numbers.* In one seven-day window, a cluster receives 100,000 tokens, sends 60,000 to a CEX, receives 20,000 back, sends 25,000 into a DEX LP position, and ends with 35,000 in the hot wallet.

The gross outgoing token flow is 60,000 + 25,000 = **85,000 tokens**. The gross incoming flow is 100,000 + 20,000 = **120,000 tokens**. The simple net flow across the cluster is 120,000 − 85,000 = **+35,000 tokens**.

If the starting balance was zero and the LP position is included in the cluster, the arithmetic reconciles. But “+35,000 net flow” does not mean “bought 35,000 tokens.” It means the selected addresses held 35,000 more tokens at the end of the window after the selected transfers. Off-chain CEX trades, fees, LP withdrawals, and unobserved addresses can change the economic position.

The intuition: gross flows describe activity; net flows describe what remains; neither alone describes intent.

## 4. Infer inventory with a reconciliation sheet

The phrase **inventory estimate** should mean a reproducible accounting view, not a dramatic number copied from a dashboard. Choose a cutoff time and build four ledgers: token units, paired assets, venue deposits, and obligations.

### Token ledger

For every address or position, record the token contract, units, role, chain, and cutoff timestamp. Do not merge a bridged representation with the native asset without documenting the bridge. Do not merge two assets with the same ticker.

### Stablecoin and cash-like ledger

Record USDC, USDT, and other stablecoins separately. A market maker may hold stablecoins as quote inventory, collateral, or settlement proceeds. A stablecoin sent to a project treasury is not automatically the proceeds of a token sale; it may be collateral for a loan, a redemption, or payment for an OTC block.

### Venue ledger

A venue deposit is an on-chain claim that assets entered the venue’s custody system. Unless the venue publishes account-level data, the blockchain cannot show the exact order fills afterward. Mark this bucket as “deposited / off-chain state unknown,” not “sold.”

### Obligation ledger

Add terms that affect economic exposure: token loans, options, lockups, LP ranges, collateral, and client assets. A market maker that holds borrowed tokens is not economically identical to a market maker that owns them free and clear. A project’s loan agreement may also explain why a large token transfer should not be interpreted as a discretionary purchase.

![A raw sum overstates the picture; a role-adjusted estimate separates venue inventory, LP inventory and hot-wallet float at one timestamp.](/imgs/blogs/tracing-a-market-makers-onchain-footprint-5.webp)

### A practical inventory formula

For a deliberately narrow snapshot, write:

$$\text{role-adjusted token units} = \text{CEX deposit units} + \text{DEX position units} + \text{hot-wallet units} + \text{settlement units} - \text{known overlap}.$$

The words “known overlap” do a lot of work. If a settlement report already consolidates the CEX and hot-wallet balances, subtract those duplicates rather than adding them. If an LP position is represented by a position NFT, use the protocol’s position data at the cutoff instead of the wallet’s visible token balance.

#### Worked example: a dated inventory snapshot

*Illustrative numbers.* At 12:00 UTC on a chosen date, your evidence table contains:

1. CEX deposit bucket: 60,000 tokens.
2. DEX LP token-side position: 25,000 tokens.
3. Hot-wallet balance: 15,000 tokens.
4. Settlement balance: 40,000 tokens.
5. Documented overlap between the settlement report and hot wallet: 10,000 tokens.

Role-adjusted units = 60,000 + 25,000 + 15,000 + 40,000 − 10,000 = **130,000 tokens**.

At a hypothetical snapshot price of $2.00, the marked token-side notional is 130,000 × $2.00 = **$260,000**. That is not realized profit, not a claim about legal ownership, and not the full economic value of the LP because the paired asset and fees are separate. It is a dated, labeled estimate using stated inclusion rules.

The intuition: an inventory number is only as honest as its cutoff, role definitions, overlap treatment and price source.

![Confidence rises from an observed transaction through venue metadata, clustering and inventory reconciliation, but the final conclusion must still state uncertainty about intent.](/imgs/blogs/tracing-a-market-makers-onchain-footprint-6.webp)

### The confidence stack in practice

Think of the investigation as a stack in which each layer answers a different objection.

The bottom layer is the raw transaction: a block, an asset contract, a quantity, a sender, a recipient and a timestamp. This is the part a chain explorer can show directly. It is often the most reliable layer, but it is also the least interpretive. A raw transfer does not tell you why it happened.

The next layer is protocol and venue metadata. Is the recipient a known pool? Is the sender calling a position manager? Does an explorer identify the destination as a CEX deposit address? Does the contract emit a swap, mint, burn, or liquidity event? Metadata narrows the interpretation, but it can be stale, incomplete or wrong.

The third layer is clustering. Do the candidate addresses share a funder? Do they receive gas from the same source? Do they route the same tokens through the same contracts? Do their timing patterns recur? Clustering is powerful because operating systems leave habits. It is also dangerous because service providers create shared habits for many clients.

The fourth layer is reconciliation. Token inflows, token outflows, stablecoin flows, LP position changes, and known obligations should fit together at a stated cutoff. A proposed cluster that cannot balance its own visible movements is not ready for a strong attribution claim. Reconciliation does not require the numbers to be identical; it requires the differences to have named explanations such as fees, bridges, or off-chain settlement.

The top layer is a defensible inference. “Consistent with a market-making cluster” is often a good conclusion. “This firm intentionally sold into retail” is a much stronger claim. The higher the claim climbs from observation toward motive, the more independent evidence it needs.

#### Worked example: a missing layer changes the conclusion

*Illustrative numbers.* Suppose an analyst sees 100,000 tokens leave a project treasury and 100,000 tokens arrive at a wallet labeled with a market-maker name. The analyst calls this “the maker’s inventory.” Later, Etherscan shows that the address is a multisig whose signers also approve transfers for a custody service. A second label shows that the address is used for several projects.

The transaction layer is unchanged: 100,000 tokens arrived. The metadata layer changes the role from “firm hot wallet” to “shared or custody-controlled wallet.” The clustering layer becomes ambiguous. The inventory estimate must now say “tokens in a wallet associated with the arrangement,” not “tokens owned by the market maker.”

The intuition: a label can increase confidence, but a better label can also reduce it. Good research is allowed to move backward.

### Data quality traps that look like market signals

**Decimals:** Token contracts store integer units and expose a decimals field. A display of 100,000 tokens can be wrong by a factor of ten or more if the analyst reads raw units as human units. Always record the contract address and decimals, not just the ticker.

**Proxy contracts:** A proxy may delegate execution to an implementation contract. If you identify a proxy as a simple wallet, you may miss the function that moved the funds. Read the verified source where available and inspect the call trace.

**Routers:** A router can receive tokens temporarily and forward them to a pool. Counting router inflow as final inventory will double-count the same asset. Follow the event logs to the terminal contract.

**Omnibus exchange addresses:** A CEX may consolidate user deposits into a hot wallet. The hot wallet’s balance can represent thousands of customers. A large outflow from that wallet does not identify a market maker.

**Bridges:** A bridge locks or burns assets on one chain and releases or mints a representation on another. If the analyst adds both sides, one economic movement appears twice. Treat the source and destination as linked legs.

**Rebases and wrappers:** Wrapped tokens, rebasing assets and vault shares can change displayed balances without a simple transfer. Use protocol-specific accounting when a position is not a plain ERC-20 balance.

**Price timestamps:** Marking a historical token balance at today’s price creates a false historical notional. The arithmetic may be impeccable and the claim still misleading. Use the price at the observation timestamp or say explicitly that the mark is a current snapshot.

**Block versus wall-clock time:** Two explorers may display local time differently. Store UTC, block number and transaction hash. If the question is about a fast move, block ordering is the safer primary sequence.

**Failed transactions:** A failed transaction can appear in an address history and may consume gas, but it did not complete the intended transfer. Do not count it as flow.

Each trap is a reason to slow down before narrating a price move. The public chain is exact about what it records; the analyst is responsible for not asking it to record something it does not.

### A minimum reproducibility packet

For every published trace, keep a small packet alongside the prose: the address list, token contract list, chain list, UTC cutoff, source links, query text, price source, role assignments, overlap decisions, and a CSV or table of transactions. If the dashboard changes tomorrow, the reader should still be able to reconstruct the result.

This packet also prevents a subtle error: changing the address list after seeing the answer. Discovery and measurement should be separate. First define the candidate set and why each address is included. Then run the flow calculation. If a new address is discovered later, add it as a new version rather than silently rewriting history.

The workflow is particularly important when a claim could damage a real firm’s reputation. Use a neutral identifier while the evidence is exploratory. Name a firm only when the association is public, relevant, and sourced. Report the firm’s response when a claim is contested. Avoid putting an employee’s name, location, or private identity into a wallet analysis unless the information is already public and material to the documented case.

### What a good negative result looks like

Sometimes the correct outcome is that the trace cannot distinguish the competing explanations. That is not a failed investigation. It is a useful result when the uncertainty is explained precisely.

For example, suppose an address receives project tokens, sends them to a CEX deposit, and never moves them on-chain again. The analysis may establish that the tokens entered exchange custody. It may not establish whether they were sold, used as collateral, placed into an internal market-making account, or transferred to another customer. The negative result is: “public chain data cannot resolve the off-chain state after deposit.” That sentence tells a retail reader not to treat a deposit alert as a confirmed dump.

Another negative result is an attribution that remains one clue short. Suppose several addresses share a funding source but interact with multiple projects and a known custodian. The cluster may be operationally related, but the same facts fit a custody-service explanation. The responsible report says: “shared funding is observed; common beneficial control is unconfirmed.” A later public disclosure, contract, or exchange record can change the result.

The discipline of negative results also improves the positive ones. If your method sometimes concludes “unknown,” readers can distinguish a measured inference from a predetermined narrative. That matters in crypto because the most shared wallet stories are often the least complete: they compress a multi-chain, partly off-chain operation into one address and one arrow.

When the data is insufficient, preserve the transaction table, list the missing evidence, and stop. Do not fill the gap with a price prediction. Do not infer intent from a color-coded dashboard. The value of an on-chain trace is not that it always produces a trade; it is that it tells you which parts of the story are public, which are private, and which remain hypotheses.

That restraint is especially valuable around launches, unlocks, listings and sudden volatility, when many unrelated wallets move at once. A dated route that survives those noisy periods is more informative than a single dramatic transfer found after a chart has already moved.

It also makes updates safer: append a new dated snapshot instead of silently replacing an old one, so readers can see whether the conclusion changed because the market changed or because the evidence improved.

## 5. Distinguish inventory from positioning

**Inventory** is what the operation holds or owes. **Positioning** is an interpretation of how that inventory may behave under price moves. A market maker can hold tokens and still hedge them with perps, options, OTC forwards, or offsetting balances on an exchange. A wallet trace that sees only the token side is missing the hedge.

### The three-layer view

1. **Physical or on-chain layer:** tokens, stablecoins, LP positions, collateral.
2. **Venue layer:** CEX order-book inventory, fills, borrow, margin and derivatives—often off-chain.
3. **Contract layer:** token loans, options, lockups, and client agreements.

Only the first layer is usually fully visible. The second is partially visible through deposits and withdrawals. The third may be visible only through governance proposals, filings, or public announcements. Any inference that leaps from layer one to a claim about the whole firm should be labeled as partial.

### The inventory-pressure heuristic

Suppose a market maker receives a large token allocation, deposits part of it to a venue, and the token’s circulating supply is small. That may create potential sell pressure. It is not proof of realized sell pressure. The deposit could be collateral or inventory for two-sided quoting. The firm may sell tokens and hedge by buying elsewhere. The venue may match the firm’s sales with demand that would have arrived anyway.

The useful defensive question is not “will this wallet dump?” It is: “what supply is potentially available, over what horizon, under what obligation, and what independent evidence would confirm that it actually traded?”

#### Worked example: why a deposit is not a sale

*Illustrative numbers.* A candidate deposits 60,000 tokens into a CEX. The snapshot price is $2.00, so the deposit has a gross notional of 60,000 × $2.00 = **$120,000**.

Scenario A: the venue account uses the tokens to quote both sides, sells 30,000, buys 25,000, and withdraws 20,000 later. The on-chain record shows the original 60,000 deposit and a 20,000 return, but not the exact fills.

Scenario B: the venue holds the tokens as collateral, and no token sale occurs.

Scenario C: the firm sells all 60,000 off-chain and later withdraws a different asset.

The same initial on-chain event is compatible with all three scenarios. Only venue-side records, public disclosures, or a robust reconciliation with subsequent transfers can narrow the possibilities.

The intuition: a CEX deposit proves custody movement, not execution direction.

## 6. Use tools as a repeatable investigation notebook

This section is a defensive, step-by-step workflow for analyzing public blockchain data. It is not a recipe for manipulating markets, evading surveillance, or targeting individuals. The goal is to make a claim auditable and to stop when the evidence stops.

### Step 1: define the claim and the cutoff

Write one sentence. “Between 2026-07-01 and 2026-07-15, did the candidate cluster’s net token flow to labeled venue addresses increase?” Specify chain, token contract, time zone, and cutoff. A moving dashboard number without a date is not a durable fact.

### Step 2: discover candidate addresses

Search Arkham, Nansen, public project disclosures, governance forums, token-holder pages, and prior research. Capture the source and the exact label. If a project names a market maker publicly, treat that as a starting point for the operational addresses, not as a complete list.

### Step 3: verify the address and contract

Open the address in Etherscan or the relevant chain explorer. Confirm the token contract, decimals, chain, and whether the destination is an EOA, router, pool, bridge, multisig, or exchange-labeled address. Inspect function calls and event logs. If the address is a proxy or a contract, trace the implementation or protocol documentation before assigning a wallet role.

### Step 4: build the route graph

Use Dune for reproducible flows and Bubblemaps for cluster visualization. Start with direct transfers, then add shared funders, gas sources, repeated counterparties, and bridge hops. Do not add every interacting address to one cluster just because it touched the same token.

### Step 5: classify roles

Assign each address one primary role and, if necessary, a secondary role: CEX deposit, DEX LP, hot wallet, settlement, treasury, bridge, router, or unknown. “Unknown” is a valid result. A precise unknown is better than a wrong label.

### Step 6: reconcile token and quote assets

For each route, track both the project token and paired assets. A DEX pool requires both sides. A CEX deposit may be followed by an off-chain trade that never appears on-chain. A stablecoin transfer back to a treasury may be collateral rather than sale proceeds.

### Step 7: calculate dated snapshots

Use a price source and timestamp. For historical analysis, save the source URL, query, and date. DefiLlama can provide protocol context; a chain explorer supplies transaction facts; an exchange’s historical market data is needed for venue-side price or fill claims. Never use today’s token price to describe an older notional without saying so.

### Step 8: stress-test alternative explanations

Before publishing “distribution,” test client custody, airdrop allocation, bridge migration, LP rebalancing, collateral, treasury management, and exchange hot-wallet sweeping. If the alternative remains plausible, keep the claim at “consistent with,” not “proves.”

### Step 9: write observation and inference in separate columns

The final table should contain: observation, source, timestamp, role, inference, confidence, alternative explanation, and what evidence would change the conclusion. This format makes a reader less vulnerable to a screenshot that compresses uncertainty into a red arrow.

![A defensive trace separates discovery, verification, clustering, reconciliation, timestamping, stress-testing and reporting.](/imgs/blogs/tracing-a-market-makers-onchain-footprint-7.webp)

### A small pseudocode sketch

```pseudocode
for each transaction in the dated window:
    normalize(chain, block_time, token_contract, units, from, to)
    classify(from, to, contract_function)
    append_to_ledger(role, direction, units, venue, source)

reconcile token_in - token_out by role
flag venue deposits as "off-chain state unknown"
subtract documented overlap
report observation separately from inference
```

This is deliberately boring. A robust trace should be reproducible by another analyst using the same address list, date window, token contracts and role rules.

## 7. A complete worked trace

We can now walk through a fictional case without pretending it is a live firm or a current wallet. The numbers are illustrative arithmetic. The method is the point.

### The question

“Did Candidate Cluster K make 100,000 project tokens potentially available to venues, and did its net token inventory rise or fall during the observation window?”

### The evidence table

At the chosen cutoff, the analyst records these hypothetical observations:

| Event | Units | Role | Interpretation boundary |
|---|---:|---|---|
| Treasury → hot wallet | +100,000 tokens | Allocation | Inventory arrived; legal terms unknown |
| Hot wallet → CEX deposit | −60,000 tokens | Venue inflow | Off-chain trading state unknown |
| Hot wallet → DEX LP | −25,000 tokens | LP position | Token-side position; paired asset separate |
| CEX-linked address → settlement | +20,000 tokens | Return flow | Compatible with rebalancing, not proof of profit |
| Hot wallet ending balance | +15,000 tokens | Operational float | May include gas or future routing |

Gross venue-directed flow is 60,000 + 25,000 = **85,000 tokens**. Net cluster movement from the listed transfers is 100,000 + 20,000 − 60,000 − 25,000 = **+35,000 tokens**. The role-adjusted ending inventory represented by the listed balances is 60,000 at the CEX, 25,000 in the LP, 15,000 in the hot wallet, and 20,000 in settlement, or **120,000 token units** if all buckets are independent. If the 20,000 settlement balance is part of the original 100,000 allocation rather than a separate asset, the economic result is different. The analyst must resolve that overlap before claiming a total.

### What can be said

The route is consistent with a market-making operation that allocated tokens across a venue deposit, a DEX LP position, a hot wallet and a settlement flow. The cadence suggests active management rather than a single passive holder. The deposit makes 60,000 tokens potentially available inside the CEX’s private ledger. The LP position exposes the operation to changing token/stablecoin composition.

### What cannot be said

We cannot say 60,000 tokens were sold. We cannot say the market maker was bearish. We cannot say the addresses share legal ownership without stronger evidence. We cannot say the project token price was manipulated. We cannot infer realized P&L from token transfers without entry prices, fills, fees, hedges and obligations.

### What would strengthen the trace

Public confirmation of the address, exchange-side fill data, a dated market-making agreement, a matching stablecoin settlement, repeated routes across independent venues, and reconciliation against the project’s reported allocation would all strengthen the analysis. A single new transfer from a different cluster can weaken it by showing that the supposed “settlement wallet” is a shared custodian.

The intuition: the best trace ends with a bounded statement and a list of unknowns, not a dramatic verdict.

## 8. Named case study: Wintermute and the Optimism OP mishap

Wintermute is a useful case because the public record makes a market-maker relationship and a wallet-control failure unusually visible. In June 2022, the Optimism Foundation published a message explaining that Wintermute had received a **20 million OP token loan** to provide liquidity around the token’s centralized-exchange launch. The official account also described an additional **20 million OP** arrangement with **$50 million USDC collateral** while the parties worked through the problem. Those figures and the sequence come from Optimism’s public governance communication, not from an inferred wallet balance. [Optimism’s official community message](https://gov.optimism.io/t/message-to-optimism-community-from-wintermute/2595) is the primary source.

The reported failure involved a mismatch between the Ethereum and Optimism deployments of a multisig address. The important tracing lesson is not that “20 million tokens means a dump.” It is that a market maker’s operational address can be a contract-control dependency. A transfer to an address that looks correct at the text level may still fail to give the intended operator control on the destination chain.

The public record also shows why role labels matter. Wintermute was acting as a market maker and was trading OP across multiple centralized exchanges, according to the discussion attached to the official post. The same economic arrangement therefore touched an on-chain loan, a multisig, CEX inventory, and launch-liquidity obligations. A trace that looked only at a token-holder page would miss the contract and cross-chain layer.

The case is also a warning about attribution. The fact that tokens were sent to a market maker’s intended address did not mean the intended operator controlled them. Address equality across chains is not control equality. For a tracer, the right checks are chain, code, deployment state, signer set, and transaction outcome.

In September 2022, Reuters reporting carried by Euronews described a separate **$160 million** hack affecting Wintermute’s DeFi operations, dated **20 September 2022**. The amount and date are reported figures, not a balance inferred from a wallet screenshot. [The Reuters report as republished by Euronews](https://www.euronews.com/next/2022/09/20/crypto-currency-wintermute) describes the affected DeFi accounts. Again, the lesson is operational surface separation: a firm can have CEX, OTC, and DeFi activities with different custody and risk controls.

The defensive takeaway is clear. Wallet traces are most valuable when they reveal plumbing and failure boundaries: which chain, which contract, which venue, which wallet role, which obligation. They are least reliable when they are used to leap from a token transfer to a claim about motive.

## 9. DWF Labs, contested claims, and the limits of inference

DWF Labs is useful as a second case because its public identity combines investment, OTC activity and market making. The company’s role bundle creates exactly the attribution problem this method is designed to handle: a token transfer may be a purchase, a market-making allocation, a loan, a settlement, or some combination.

In 2023, CoinDesk reported that DWF had announced more than **$200 million** of deals by **29 March 2023**, while other reporting discussed individual announced transactions. The number describes announced deals reported by the outlet at that date; it is not an independently verified estimate of cash invested or inventory held. DWF’s own framing and the interpretation of those transactions are contested. The [existing DWF profile](/blog/trading/crypto-players/dwf-labs-the-controversial-newcomer) discusses the distinction between a discounted token purchase and conventional venture equity.

The more serious allegation belongs in careful language. The Wall Street Journal reported in **May 2024** that Binance’s internal surveillance team had alleged DWF manipulated prices and engaged in more than **$300 million** of wash trades during **2023**. DWF denied the claims, and Binance said there was insufficient evidence of market abuse; the details and intent remain contested. A report of an internal allegation is not a regulatory finding. The numbers should therefore be written as “reported allegations,” attributed to the reporting, with the denials attached—not as established fact.

What can on-chain data contribute? It can test whether a token route appears to move from a project or treasury to a cluster, whether the cluster deposits to a venue, whether the same timing repeats across tokens, and whether transfers are round-tripped. It cannot establish that two exchange accounts were controlled by the same party, that a matched trade was wash trading, or that a transfer caused a price move. Those questions require off-chain account records, surveillance data, contracts, and legal process.

This is where the defensive line matters. A retail reader can record a suspicious-looking route, reduce confidence in a token’s liquidity story, and avoid treating announced deal size as equivalent to committed long-term capital. The reader should not turn a public wallet label into an accusation against a person or firm.

## How it shows up in price

On-chain traces matter to price through four channels.

### Potential float

If a large token allocation moves from a treasury into a venue-linked address, the market may treat some portion as potentially sellable. The effect is larger when the token’s circulating float is small or the order book is thin. The trace is a supply-risk signal, not a guaranteed sale.

### Liquidity shape

When an LP position holds less of the token and more of the paired asset, the pool may have absorbed buys. When it holds more of the token, the pool may have absorbed sells. A pool can quote continuously while becoming more exposed to one side. That exposure can change how aggressively a liquidity provider rebalances.

### Inventory hedging

A market maker may sell tokens on one venue and buy perps or spot elsewhere. The on-chain footprint may show a deposit and a withdrawal, while the hedge remains off-chain. Price can therefore react to the visible flow without the firm having a simple one-way view.

### Narrative reflexivity

A wallet label can itself move attention. Screenshots of a “market maker deposit” may prompt traders to front-run an assumed sale. The resulting price move can then make the original inference look correct even when the wallet was only preparing inventory. This is why a defensive analyst should publish timestamps, alternatives and confidence rather than a directional headline.

#### Worked example: a thin-book impact estimate is not a forecast

*Illustrative numbers.* Suppose a token trades at $2.00 and the visible bids within a chosen price band sum to 40,000 tokens. A suspected venue deposit is 60,000 tokens. The deposit is larger than the visible bid depth, but that does not mean the price must fall. The tokens may be posted on both sides, sold in smaller clips, hedged elsewhere, or never sold.

The only arithmetic we can safely state from the hypothetical inputs is the ratio: 60,000 ÷ 40,000 = **1.5×** the selected visible bid depth. That is a stress flag for liquidity, not a price target. The band, time window and depth source must be stated because order books change.

The intuition: compare potential flow with available liquidity, but do not convert a capacity comparison into a deterministic price prediction.

## Common misconceptions

### “The biggest holder is the market maker.”

It may be a treasury, custodian, bridge, pool, vesting contract or exchange omnibus address. A holder ranking is discovery evidence. Role classification and transaction history are required before attribution.

### “A transfer to Binance means the tokens were dumped.”

It means assets entered a venue-linked custody address. The subsequent ledger is usually off-chain. Call it a deposit unless fills or a later reconciliation prove more.

### “A DEX LP balance is a long token bet.”

An LP position contains paired assets and changes composition as trades move the pool price. It can earn fees while losing token units. Read the position contract and both sides of the pool.

### “A wallet cluster proves common ownership.”

Shared funding, gas, timing and routes can suggest common control, but a custodian, service provider or client operation can create the same pattern. Use “attributed,” “clustered,” or “consistent with” when ownership is not independently confirmed.

### “A large inventory means the firm is bullish.”

Inventory may be borrowed, hedged, collateralized, or required to quote two sides. Directional exposure is an economic quantity that may not be visible on-chain.

### “On-chain transparency makes manipulation obvious.”

Blockchains expose transactions, not exchange account ownership, order intent, fills, contracts or motive. A suspicious pattern can justify investigation; it is not automatically proof of manipulation.

## Retail defensive takeaway

If you are trading or researching a token, use wallet traces to improve your questions, not to outsource your judgment.

First, record the token contract and the date. A ticker is not a unique asset, and a live dashboard changes. Second, ask whether the observed address is a treasury, pool, bridge, venue deposit, hot wallet or settlement wallet. Third, distinguish a deposit from a fill and a token balance from an economic position. Fourth, look for paired-asset flows and obligations such as loans or unlocks. Fifth, compare potential flow with actual liquidity, but avoid false precision about price impact.

For a simple monitoring sheet, keep these columns:

| Question | Safe wording |
|---|---|
| What happened? | “Address A transferred X units at block Y.” |
| What is the address? | “Explorer labels it as…” |
| What does the role imply? | “Consistent with venue inventory…” |
| What remains unknown? | “CEX fills and account ownership are not public.” |
| What would change the view? | “A public allocation, return flow, or venue record.” |

Do not copy a whale-alert screenshot into a trade thesis. Do not publish an allegation as a fact. Do not try to identify a private individual from a wallet. The defensive edge is better measurement: dated evidence, independent sources, explicit alternatives, and a willingness to conclude “unknown.”

## When this matters to you

Market makers are part of the price you receive even when you never interact with them directly. Their quotes affect spread and slippage. Their inventory and hedging affect how a token absorbs a large order. Their agreements with projects can affect the amount of supply that may become available. Their wallet operations can reveal when a liquidity arrangement is being funded, rebalanced or unwound—but only if the trace is read as a system.

The method is reusable beyond market makers. The same role-first approach helps analyze token treasuries, venture allocations, bridge migrations, lending liquidations and exchange solvency claims. In every case, start with what the chain proves, then add labels, context and inference one layer at a time.

The central discipline is simple:

> A wallet is an address. A footprint is a hypothesis about a workflow. A hypothesis earns trust by surviving alternative explanations.

This is educational analysis, not individualized financial advice. On-chain evidence is incomplete, and market-making arrangements can include off-chain accounts and confidential contracts that no public dashboard can recover.

## Sources & further reading

- [Message to the Optimism community from Wintermute](https://gov.optimism.io/t/message-to-optimism-community-from-wintermute/2595) — Optimism Collective, June 2022; primary source for the reported 20 million OP loan and the additional 20 million OP / $50 million USDC collateral arrangement.
- [Major crypto trader Wintermute hit by $160 million hack](https://www.euronews.com/next/2022/09/20/crypto-currency-wintermute) — Reuters report republished by Euronews, 20 September 2022; source for the dated reported DeFi-loss figure.
- [DWF Labs denies report that it did $300 million of wash trading on Binance last year](https://www.theblock.co/post/293429/dwf-labs-denies-report-that-it-did-300-million-of-wash-trading-on-binance-in-2023) — The Block, May 2024; source for the reported allegation and denials. The claim is contested, not presented here as an adjudicated fact.
- [DWF Labs emerges as top crypto investor](https://www.coindesk.com/business/2023/03/29/market-maker-dwf-labs-emerges-as-top-crypto-investor/) — CoinDesk, 29 March 2023; source for dated reporting on announced deal activity.
- [Etherscan](https://etherscan.io/) — explorer for transaction, contract and event-log inspection; use the relevant chain and save the query date.
- [Arkham](https://www.arkhamintelligence.com/), [Nansen](https://www.nansen.ai/), [Dune](https://dune.com/), [Bubblemaps](https://bubblemaps.io/), and [DefiLlama](https://defillama.com/) — complementary discovery, clustering, reproducible-query and protocol-context tools; labels and dashboard values should be checked against dated underlying data.
- [What a Crypto Market Maker Actually Does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) — the series foundation on quoting, spread and inventory.
- [Inventory Risk, Hedging and Delta Neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality) — why visible token balances do not equal unhedged direction.
- [Reading the Tape: Defending Yourself as Retail](/blog/trading/crypto-players/reading-the-tape-defending-yourself-as-retail) — a companion guide to interpreting market activity without turning signals into certainty.
- [Wash Trading, Spoofing and Manufactured Volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) — the manipulation mechanics that require exchange-side evidence beyond a public transfer graph.
