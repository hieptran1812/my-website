---
title: "The On-Chain Tooling Landscape: Explorers, Analytics, and Where the Data Comes From"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A practical map of every on-chain analyst's toolbox — what each tool is for, when to reach for which, and where the data actually comes from, from nodes to the dashboards you click."
tags: ["onchain", "crypto", "blockchain-tools", "etherscan", "dune", "nansen", "arkham", "defillama", "glassnode", "chainalysis", "blockchain-data", "indexer"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — On-chain analysis is not one tool; it is a toolbox, and the whole skill is matching the question you have to the tool that answers it. Every dashboard you click is the polished far end of the same pipeline: a node holds the raw chain, an RPC endpoint serves it, an indexer turns it into queryable tables, and an API feeds the screen you read.
>
> - **The pipeline:** node → RPC → indexer → API → dashboard. You almost never run your own node; you rent the output of one through a tool.
> - **The five jobs:** block explorers read one transaction, entity platforms tell you *who* an address is, metric dashboards size a market, query platforms answer any custom question, and forensics suites trace stolen funds. Pick by the shape of your question.
> - **What you DO:** route the question by its shape — one tx → Etherscan; who is buying → Arkham/Nansen; how big is the market → DefiLlama/Glassnode; a custom slice → Dune; where did the hack go → Chainalysis.
> - **The one limit to remember:** no tool on earth can see inside a centralized exchange. The CEX internal ledger is an off-chain private database. On-chain data ends at the deposit address and resumes at the withdrawal — the matching in between is invisible.

On 21 February 2025, a routine-looking transfer left one of Bybit's Ethereum cold wallets. Within minutes it was clear that something was very wrong: roughly **\$1.46 billion** of ETH and staked-ETH derivatives had drained out in a single coordinated sweep — the largest crypto theft ever recorded. Here is the remarkable part for our purposes. Nobody had to wait for Bybit to issue a press release to know the money was gone. Anyone watching the right tools watched it happen *live*. Arkham and Nansen lit up with the outflow inside minutes. Etherscan showed the raw transactions. Within an hour, forensics firms like Elliptic and TRM had begun tagging the destination addresses, and over the following days they published interactive maps of the funds fanning out across hundreds of wallets, getting swapped to native assets, and bridging toward laundering rails.

That is the whole promise of on-chain analysis in one episode: the ledger is public, so the theft was public, and a chain of free and paid tools turned an unreadable stream of bytes into a story a human could follow. But it also shows why the toolbox matters. Etherscan could show you *that* a transaction happened; it could not tell you the destination wallets belonged to a North Korean laundering operation. Arkham could *label* the entity but could not, on its own, freeze anything. And the moment those funds touched a centralized exchange's internal books, even the best forensics suite went briefly blind. Different tools, different jobs, hard limits on each.

This post is the map of that toolbox. It is the reference the rest of this series links back to whenever a post says "open Etherscan" or "check it on Dune." We will build from the ground up: what a node even *is*, why the data has to be processed before you can read it, and then a category-by-category tour of the explorers, intelligence platforms, dashboards, query tools, and forensics suites — what each is for, what it costs, and where it goes blind.

![Data pipeline from a blockchain node through RPC, an indexer, and an API to the dashboard you read](/imgs/blogs/the-onchain-tooling-landscape-1.png)

## Foundations: where on-chain data actually comes from

Before we name a single tool, you need the pipeline in the figure above in your head, because *every* tool — free or \$500-a-month — is just a different slice of it. The data does not start in a dashboard. It starts in a database that thousands of computers around the world keep in sync. Let us walk it from the source.

### A node is a computer that holds the chain

A **blockchain** is a shared, append-only ledger: an ordered list of blocks, each block a batch of transactions, each transaction signed by the address that sent it. A **node** is simply a computer running the blockchain's software (for Ethereum that is a client like Geth, Nethermind, or Reth) that downloads, validates, and stores that ledger. There is no central server; the "blockchain" is the agreement among all these nodes about which blocks are valid and in what order.

There are two flavors of node you will hear about, and the difference matters for tooling:

- A **full node** holds the current *state* (every account balance, every contract's storage right now) plus enough recent history to validate new blocks. It can answer "what is this address's balance *today*?" but it prunes old intermediate states to save space.
- An **archive node** keeps *every* historical state — the balance of every address at *every* past block. This is what lets a tool answer "what was this wallet holding the day before the crash?" An Ethereum archive node is enormous: roughly **2 terabytes** and growing, and it takes days to sync from scratch.

The first thing to internalize: on-chain analysis is fundamentally *replaying* and *querying* this data. Every chart of exchange reserves, every "smart money bought" alert, every traced laundering route is, underneath, a question asked of node data.

### RPC is the doorway into a node

You do not talk to a node by reading its hard drive. You talk to it through an **RPC endpoint** — Remote Procedure Call, a standardized request-response interface (JSON-RPC for Ethereum). You send a request like "get the balance of address X" or "give me the receipt for transaction Y" and the node answers. RPC is the universal doorway: wallets like MetaMask use it to show your balance, explorers use it, indexers use it. It is the lowest level most builders ever touch.

The catch is that raw RPC is *granular and dumb*. You can ask for one block, one transaction, one address's balance. You cannot ask "show me every wallet that bought this token in the last week" — RPC has no concept of "this token" as a queryable table; it only knows blocks and accounts and raw event logs. To get a *question-answering* layer, you need the next stage.

#### Worked example: why you do not run your own node

Say you want to track exchange flows yourself. To do it from first principles you would run an Ethereum archive node. The hardware: a machine with a fast NVMe SSD of at least **4 TB** (headroom over the ~2 TB archive), 32 GB of RAM, and a reliable connection. Buy it outright and you are looking at roughly **\$1,500–\$2,500** in hardware, plus the electricity and the days of sync time, plus the ongoing maintenance whenever the client updates. Rent equivalent cloud capacity and a beefy instance with the storage runs on the order of **\$300–\$600 per month**.

Now compare: a managed RPC provider (Alchemy, Infura, QuickNode) will give you a hosted archive endpoint with a **free tier** that covers millions of requests a month, and paid tiers starting around **\$49–\$199 per month** for serious volume. You get the node's output without ever touching the hardware. Unless you are building infrastructure that needs custom node behavior, paying \$0–\$199 to rent beats spending \$2,000 up front and babysitting a server. *The intuition: you are almost never paying for the chain data itself — it is public and free — you are paying someone to run the annoying machine that serves it.*

### An indexer turns raw blocks into tables you can query

Here is the stage that makes everything readable. An **indexer** is a service that connects to a node (over RPC), reads every block as it arrives, *decodes* the raw data, and writes it into a structured, queryable database — tables of transfers, swaps, mints, liquidations, with human-readable columns.

Why is decoding necessary? Because what the node stores for a smart-contract event is a blob of hexadecimal. A Uniswap swap, on the wire, is an event log with a cryptic `topic0` hash identifying the event type and a `data` field of packed hex numbers. It is correct, complete, and totally unreadable to a human. The indexer holds the contract's **ABI** (Application Binary Interface — the schema that says "this event is called Swap and its fields are amount0, amount1, …") and uses it to translate the hex into a row that says, in effect, *"address 0xA11ce sold 5,000 USDC for 1.6 ETH in the USDC/ETH pool."*

![Before and after of an indexer decoding a raw hex event log into a readable swap row](/imgs/blogs/the-onchain-tooling-landscape-4.png)

The figure above is the single most important idea in this whole post. The left column is what comes off the node: opaque hex. The right column is what every dashboard shows you. **The entire industry of on-chain tooling exists to perform that transformation at scale**, across every contract, every chain, and then to let you ask questions over the result. When Dune lets you write `SELECT * FROM uniswap_v3.swaps`, that table exists because an indexer decoded billions of those hex logs into rows first.

A specific, popular flavor of indexer is **The Graph**, an open protocol where developers publish **subgraphs** — small, open-source indexing definitions that say "watch this contract, decode these events, expose this GraphQL API." When a DeFi app shows you your position history, it is very often reading a subgraph. The Graph matters because it decentralizes and standardizes the decoding step that used to be every team's private plumbing.

### Why the chain matters: account vs UTXO

One more foundation before the tools, because it explains why you cannot use one explorer for everything. Blockchains do not all model "who owns what" the same way, and the data model drives the tooling.

**Account-based chains** — Ethereum and every EVM chain (Arbitrum, Base, BNB, Polygon), plus Solana and Tron — keep a running balance for each account, like a bank ledger: address A has 3 ETH, address B has 1,200 USDC. A transaction debits one balance and credits another. This is intuitive, and it is why Etherscan can show you "this address's balance" as a single number. Most of DeFi lives here because smart contracts (programs that hold and move funds) fit naturally on an account model.

**UTXO-based chains** — Bitcoin, and Bitcoin-derived chains — have *no* running balances at all. Instead, your "balance" is the sum of a set of **unspent transaction outputs** (UTXOs): discrete chunks of coin, each one a leftover from a previous transaction, like a wallet full of specific bills and coins rather than a single account number. To spend, you consume whole UTXOs as *inputs* and create new UTXOs as *outputs* (including change back to yourself). This is why a Bitcoin explorer like mempool.space shows a transaction as a fan of inputs flowing into a fan of outputs, not a simple "from → to." It is also why tracing Bitcoin is its own discipline: you follow the *graph of outputs*, and a single payment can fragment your coins across many UTXOs that later get recombined in revealing ways.

The practical takeaway for the toolbox: an EVM explorer cannot read Solana or Bitcoin, a Solana explorer cannot read Bitcoin, and tracing heuristics differ by model. When a post in this series says "open the explorer," the *right* explorer depends on the chain — Etherscan-family for EVM, Solscan/SolanaFM for Solana, Tronscan for Tron, mempool.space for Bitcoin. The pipeline (node → RPC → indexer → API → dashboard) is identical in *shape* on every chain; only the contents of each stage change.

### The difference between raw data and curated analytics

This gives us the crucial distinction that organizes the whole toolbox. There are two fundamentally different things a tool can hand you:

1. **Raw (or lightly-decoded) data** — the transactions, balances, and decoded events, presented faithfully and neutrally. A block explorer is this. It shows you *exactly* what is on the chain and editorializes very little. Truthful, complete, but it makes you do the interpretation.
2. **Curated analytics** — somebody has run the raw data through their own logic, labels, and assumptions to produce a *signal*: "this address is smart money," "exchange reserves fell 8% this month," "this token's TVL is \$40M." Useful, fast, but you are now trusting *their* methodology, not just the chain.

Both are valid. The danger is forgetting which one you are looking at. A block explorer's "balance" is a fact. A dashboard's "smart money is accumulating" is a *claim* built on a label set that has its own biases and survivorship problems. Throughout this series the refrain is the same: **on-chain data is trustworthy; on-chain interpretations are opinions wearing data's clothes.** Know which layer you are standing on.

With the pipeline in hand, here is the whole toolbox at a glance — five categories, each answering a different shape of question.

![Matrix of on-chain tool categories with the best question for each and its pricing tier](/imgs/blogs/the-onchain-tooling-landscape-2.png)

The rest of this post walks each row of that matrix in turn. Then we will do the reverse: a list of concrete questions and which tool to reach for, the decision flow you will actually use in practice.

## Category 1: block explorers — read any single transaction

A **block explorer** is the closest thing to a neutral window onto the chain. Type in an address, a transaction hash, or a contract, and it shows you exactly what is there: balances, the full transaction history, the contract code, the token transfers inside a transaction. It is free, it is canonical, and it is where every investigation either starts or ends.

The names you need:

- **Etherscan** for Ethereum — the reference explorer, copied everywhere. Its siblings run the same software for other EVM chains: **Arbiscan** (Arbitrum), **Basescan** (Base), **Polygonscan**, **BscScan** (BNB Chain), **Optimistic Etherscan**. If you know Etherscan you know all of them.
- **Solscan** and **SolanaFM** for Solana — Solana's account model and transaction format are different enough that you need a Solana-native explorer; the EVM explorers will not work.
- **Tronscan** for Tron — important because Tron carries an enormous share of retail USDT transfers (and, correspondingly, a large share of illicit stablecoin flow), so a lot of stablecoin tracing happens here.
- **mempool.space** for Bitcoin — Bitcoin's UTXO model is different again, and mempool.space is the standard for watching pending transactions, fee rates, and tracing Bitcoin's input/output graph.

### How to read it: a walkthrough on Etherscan

Say you have a transaction hash from a friend who claims they got an airdrop. Paste it into Etherscan and you get a structured page. The fields that matter:

- **Status** — Success or Failed. A failed transaction still costs gas and still appears on-chain; people miss this constantly.
- **From / To** — the sending address and the immediate recipient (often a contract, not a person).
- **Value** — the native-token (ETH) amount moved directly. Frequently `0`, because the real action is in token transfers and internal calls, not the headline value.
- **Tokens Transferred** — the decoded ERC-20/ERC-721 movements. *This* is usually where the substance is: "1,000 USDC from A to B," "5 NFTs from C to D."
- **Internal Transactions** — calls a contract made to other contracts. A single user click can trigger a dozen internal calls; this tab is how you see the cascade.
- **Logs** — the raw decoded events. This is the hex-to-readable layer from our indexer figure, exposed directly.
- **Gas** — what the transaction cost to execute, in gas units and in ETH (and the dollar equivalent at the time).

For the deeper anatomy of every field — nonce, gas limit versus gas used, the difference between value and internal transfers — see [the anatomy of a transaction](/blog/trading/onchain/anatomy-of-a-transaction), which dissects a single transaction field by field. For what an *address* is versus a *contract* (the From/To distinction above), see [addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts).

#### Worked example: spotting a honeypot token on Etherscan

A token is trading and you are tempted. Before you buy, the explorer answers the most important question for free: *can I sell it?* Here is the pass.

Open the token's contract on Etherscan and check **Holders**. Suppose the page shows the top holder — usually the deployer or a "team" wallet — controls **62%** of the supply. That alone is a red flag: one wallet can crush the price at will. Next, check **Contract → Read/Write** and look at whether the source is **verified** (if it is not even verified, walk away — you cannot see what it does). If verified, scan for functions that let an owner change transfer rules, blacklist sellers, or set a sell tax to 99%. Those are the mechanics of a honeypot, a contract you can buy but not sell.

Now the money. Say the token has a **\$2,000,000** fully-diluted "market cap" on the screen, but the actual liquidity pool — visible in the token's transactions — holds only **\$8,000** of paired ETH. That \$2M is fiction; you could realistically extract a few hundred dollars before the price collapses to zero. If you had put in **\$5,000**, the honeypot or the thin liquidity means your realistic recoverable value is closer to **\$0**. *The intuition: the explorer will not say "scam," but it shows you concentration, contract powers, and real liquidity — three facts that turn "looks like a 10x" into "looks like a \$5,000 donation."* The full rug-detection routine — every check, in order — is its own post; this is the explorer's slice of it.

### Beyond the basics: what else an explorer gives you for free

The transaction page is the headline, but explorers quietly pack in tools that replace whole paid features elsewhere:

- **Token approvals.** Etherscan's "Token Approvals" view lists every contract you have granted permission to move your tokens — the single most useful security check there is, because an old, forgotten, unlimited approval to a since-compromised contract is how a huge share of "I got drained" stories happen. Revoking stale approvals costs only gas and closes the door.
- **Contract verification and read/write.** For a verified contract you can read its current state (who is the owner, what is the fee, is trading enabled) and even call functions directly from the explorer, no code required. This is how you confirm — not assume — that a token's owner cannot freeze your sells.
- **Label clouds and tags.** Etherscan crowd-sources labels: many exchange wallets, bridges, and known scam addresses carry a tag. Treat these as helpful hints, not authority — they are incomplete and can lag.
- **The "Analytics" and "Holders" tabs.** Holder distribution, transfer counts, and a balance-over-time chart for any address, free. For a quick "is this concentrated?" read it beats opening a paid tool.

The skill with explorers is knowing they answer *factual, local* questions extremely well and *interpretive, global* questions not at all. "What did this transaction do?" — perfect. "Is this wallet a fund?" — not their job; that is the next category.

### How explorers can deceive you

Explorers are honest, but they are not *interpreted*. A transaction page does not tell you the To address is a Lazarus wallet, that the "1,000 USDC transfer" is the eighth hop in a peel chain, or that the token's "volume" is wash trading between two addresses the same actor controls. The explorer shows the *facts*; the *meaning* is up to you or up to the next category of tool. And the labels explorers *do* show (Etherscan tags some addresses as "Binance," "Tornado Cash," etc.) are crowd-sourced and incomplete — useful hints, never gospel.

## Category 2: entity and intelligence platforms — find out *who*

A block explorer shows you address `0x7a16…`. It will never tell you that `0x7a16…` is a market maker, a hedge fund's trading wallet, or the same person who deployed three rug-pulls last month. That is the job of **entity/intelligence platforms**: they cluster pseudonymous addresses into named actors and attach behavior labels, so you can ask *who* rather than just *what*.

![Tree showing many pseudonymous addresses clustered into one labeled entity by behavioral heuristics](/imgs/blogs/the-onchain-tooling-landscape-6.png)

How do they do it? The chain is pseudonymous, not anonymous. As the figure shows, platforms run **clustering heuristics**: addresses funded by the same source, addresses that move in tight time-sync, addresses that deposit to the same exchange account, addresses that interact with the same private contracts. Stack enough of these signals and a swarm of addresses resolves into one entity. This is exactly how [smart money is identified on-chain](/blog/trading/onchain/what-is-smart-money-onchain) — and exactly why those labels are fallible, since a wrong cluster mislabels a stranger as your "smart money."

The platforms:

- **Arkham** — entity labels and money-flow visualization. Its strength is the graph: pick a wallet, see who funds it and where its money goes, with entities (exchanges, funds, individuals) named. Arkham famously runs an "intel exchange" where users bid for de-anonymization work. Freemium: a lot is free, deeper features and alerts are paid.
- **Nansen** — the "smart money" platform. Nansen's edge is its labels: it tags wallets as "Smart Money," "Fund," "Whale," "First Mover," and lets you see what those cohorts are buying and selling in near-real-time. Its **Wallet Profiler** and token-god-mode dashboards are the canonical "is smart money buying this?" tool. Subscription, and not cheap.
- **DeBank** and **Zerion** — portfolio view across chains. Paste any address and see its entire DeFi position — tokens, LP positions, lending deposits, debts — aggregated across dozens of chains in one net-worth number. Free, and the fastest way to "read" a wallet's holdings.
- **Bubblemaps** — holder-cluster maps. It draws token holders as bubbles, with lines connecting wallets that have transacted, instantly revealing when "500 holders" are really 5 clusters controlled by one deployer. The killer free tool for rug due-diligence.

### How to read it: is smart money buying?

The canonical workflow. You see a token pumping and want to know if it is real demand or a pump-and-dump. On Nansen, open the token's page and filter holders to the **Smart Money** label. If 30 smart-money wallets have *added* in the last 24 hours and few have sold, that is a genuine accumulation signal. If instead the "buyers" are a cloud of fresh wallets all funded from one source an hour ago — Bubblemaps or Arkham will show that cluster — it is manufactured.

#### Worked example: when a Nansen subscription pays for itself

Nansen's paid tiers run on the order of **\$100–\$150 per month** for an individual plan (pricing changes; treat this as the right order of magnitude). Is it worth it? Run the math from a single trade.

Suppose Nansen's smart-money flow flags an early accumulation in a token: 25 high-quality wallets adding over two days while the price is flat. You buy **\$5,000** worth. Over the next three weeks the token 3×'s as the move plays out, and you exit at **\$15,000** — a **\$10,000** gain. Against that, the cost of the tool for the month was **\$150**. The subscription paid for itself **66 times over** on one trade.

But run the *honest* version too. Smart-money labels suffer survivorship bias — you see the wallets that *were* right, named "smart" *because* they were right. For every flagged accumulation that 3×'s there are several that go nowhere or get dumped on you. If you blindly followed **10** such signals at **\$5,000** each (**\$50,000** deployed) and the real hit rate were, say, 3 winners averaging +\$10,000 and 7 losers averaging −\$2,500, your net would be **+\$30,000 − \$17,500 = +\$12,500** — still positive, but a far cry from "66×," and easily negative if your discipline slips. *The intuition: a \$150/month tool is cheap relative to one good trade, but it sells you a labeled signal, not a guarantee — the edge is in how you size and verify, not in the green "Smart Money" tag.*

#### Worked example: a full token due-diligence pass on free tools alone

The strongest argument for the free tier is that a serious token check needs *zero* paid tools. Run the pass on a token someone is shilling, before you risk a dollar.

Start on **Etherscan**: open the contract, confirm the source is verified, and check holders — say the top wallet holds **48%** and the top ten hold **81%**. That concentration alone caps how much you would commit. Next, **Bubblemaps**: the holder map shows those top-ten wallets are connected by transfers into **two** clusters, not ten independent holders — so the real float in strangers' hands is tiny. Then the **liquidity**: the token shows a **\$3,000,000** fully-diluted cap on the price screen, but the paired liquidity in its main pool is only **\$22,000**. Finally, **DeBank**: paste the deployer wallet and you see it has launched and abandoned three other tokens in the past month.

Tally the cost: Etherscan **\$0**, Bubblemaps **\$0**, DeBank **\$0**. Total spent on tools to learn this is a serial rug with \$22,000 of real liquidity behind a \$3M paper cap: **\$0**. Had you skipped the pass and bought **\$4,000**, the thin liquidity and 81% insider control mean your realistic exit value is near **\$0** the moment they sell. *The intuition: the free stack answers the only question that matters before buying — can insiders crush this, and is the liquidity real — and it answers it for nothing; paying for a tool would not have made this check any better.*

### The deception to watch

Entity platforms are the most *seductive* category because a confident label feels like truth. Three traps: **survivorship bias** in "smart money" (defined by past wins), **stale or wrong clusters** (a heuristic that wrongly merges two unrelated actors, or misses that one actor split across fresh wallets specifically to dodge labels), and **bait wallets** (sophisticated actors who know they are watched and deliberately create misleading flows). Treat a label as a *lead to verify on the raw chain*, never as a verdict.

## Category 3: metric and research dashboards — size the market

The first two categories are address-level. This one zooms out to the whole market or a whole protocol and gives you aggregate metrics: how much value is in DeFi, how much Bitcoin sits on exchanges, whether the average coin is held at a profit or a loss. These are the dashboards that turn the chain into macro-style indicators.

- **Glassnode** and **CryptoQuant** — the Bitcoin/Ethereum market-metrics platforms. They compute and chart on-chain indicators: **MVRV** (market value over realized value — a cost-basis valuation gauge; high = the average holder is deep in profit and may sell, low = capitulation), **SOPR** (spent-output profit ratio — are coins moving at a profit or loss), **exchange reserves** (how much supply sits on exchanges ready to be sold), realized cap, HODL waves. Free tiers show some; the good stuff is paid.
- **DefiLlama** — the free, neutral standard for DeFi aggregates. **TVL** (total value locked — the dollar value deposited in a protocol), fees, revenue, stablecoin supply, yields, chain rankings. It is community-run, ad-free, and the first place to size a protocol or a chain.
- **Token Terminal** — protocol *fundamentals* framed like a stock screener: revenue, fees, price-to-sales, treasury, active users, presented for comparison across protocols. The tool for "is this protocol's valuation justified by its actual cash flows?"

### How to read it: what DefiLlama shows

DefiLlama's home screen is a chart of total DeFi TVL over time, and that single chart tells the cycle story better than any price chart.

![Line chart of DeFi total value locked rising to a 2021 peak then collapsing post-FTX and recovering](/imgs/blogs/the-onchain-tooling-landscape-8.png)

The shape above is the real history (rounded, from DefiLlama): TVL climbed from about **\$16B** in late 2020 to a euphoric peak near **\$180B** in November 2021, collapsed to a post-FTX trough around **\$39B** by the end of 2022, and recovered to roughly **\$135B** by mid-2025. Each turn maps to a known regime: the DeFi-summer mania, the LUNA/FTX deleveraging, the slow rebuild. When you read a single protocol's TVL, you read it *against* this backdrop — a protocol bleeding TVL while the whole market rises is in real trouble; one falling alongside a market-wide drawdown may just be beta.

#### Worked example: reading a protocol's TVL and revenue

You are evaluating a lending protocol. DefiLlama shows **\$135B** total DeFi TVL across the market and, say, **\$2.7B** of that sitting in this one protocol — about **2%** of all of DeFi, which already tells you it is a top-tier, not a fringe, venue. Now flip to Token Terminal for the fundamentals. Suppose it shows **annualized fees of \$120M** and **annualized protocol revenue of \$30M** (the share that accrues to the protocol/token rather than to lenders).

Put a valuation on it. If the token's fully-diluted market cap is **\$900M**, then price-to-sales (on revenue) is **\$900M / \$30M = 30×**. Is 30× expensive? For a growing protocol with real, recurring revenue, maybe defensible; for a flat one, rich. Compare it to a peer at **15×** on the same screen and you have a relative-value read in two minutes — the kind of comparison that, in equities, takes a Bloomberg terminal and a quarterly filing. *The intuition: metric dashboards let you size and value a protocol like a business — TVL is its deposit base, fees are its top line, revenue is what reaches the owner, and price-to-sales is the multiple you are paying for it.*

For exchange reserves specifically — the supply-on-tap gauge — there is a dedicated post on [exchange flows, inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) that shows how to read a reserve drawdown as a supply signal. And for the broader question of why crypto trades on the same liquidity tides as everything else, see [crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset).

### The three metrics you will see everywhere

Because the rest of this series leans on these, here is the plain version of the three on-chain market metrics you will meet most often on Glassnode and CryptoQuant:

- **MVRV (market value / realized value).** The numerator is the price today times coins outstanding. The denominator values each coin at the price it *last moved* — its on-chain cost basis. So MVRV is, roughly, "how far is the average holder in profit?" Above ~3.5 historically meant euphoria and a top zone (the average coin is held at a fat gain, inviting profit-taking); below ~1 meant capitulation (the average holder is underwater). It does not predict *when* — it tells you the *temperature*.
- **SOPR (spent-output profit ratio).** Of the coins that *moved* on a given day, were they sold at a profit or a loss on average? SOPR above 1 means coins are realizing gains; below 1 means holders are selling at a loss (often capitulation). It reads the *flow* of realized profit, where MVRV reads the *stock* of unrealized profit.
- **Exchange reserves.** The total coin balance across addresses an analyst believes belong to exchanges. Falling reserves are read as bullish (coins leaving exchanges to self-custody = less immediate sell pressure); rising reserves as supply arriving to be sold. The caveat from the off-chain wall applies: these addresses are *identified heuristically* and are estimates.

None of these is a buy/sell button. They are *context* — gauges of where the market sits in its own cycle. The reason they belong to the dashboard category and not the explorer category is that every one of them embeds a *choice* (which addresses are exchanges? what counts as "moved"?), which is exactly why two dashboards can publish different values for the "same" metric.

### Where the dashboards deceive

Two pitfalls. First, **methodology differences**: two dashboards can report different TVL or different "exchange reserves" for the same thing because they count different addresses or handle double-counting (a token deposited, then re-lent) differently. Always know whether a number is gross or net. Second, **metrics that lag or that everyone watches**: an indicator like MVRV is genuinely informative, but it is also watched by millions, so it is partly priced in and prone to crowded positioning. A dashboard metric is an input, not a trade.

## Category 4: query and build platforms — answer *any* question

The first three categories answer the questions their designers anticipated. Eventually you will have a question nobody built a dashboard for: "which wallets bought token X *and* token Y in the same week," "how much did this specific contract pay out in the last 30 days," "what's the median size of a swap on this pool." For those, you need to query the decoded data yourself.

- **Dune** — the dominant one. Dune maintains decoded, queryable tables of on-chain data and lets you write **SQL** over them, then publish the results as charts and dashboards. Tens of thousands of community dashboards already exist for popular protocols, and you can fork and adapt any of them. The free tier is generous for learning; paid tiers add private queries, more compute, and API access.
- **Flipside** — a similar SQL-over-decoded-data platform, historically strong on non-EVM chains (Solana, Near, Flow) and known for its analyst bounty programs.
- **The Graph subgraphs** — for *application* developers who need a live, typed API of one protocol's data inside their app, rather than ad-hoc analysis. You query a subgraph with GraphQL.

The mental shift here is the biggest in the toolbox: you stop being a *consumer* of someone's dashboard and become the analyst. The flip side is responsibility — your query is only as correct as your understanding of the tables. A subtle join error or a forgotten decimal-scaling can produce a confident, beautiful, *wrong* chart.

### How to read it: a first Dune query

The first thing to understand about Dune is that the hard work — decoding — is already done. The `uniswap_v3.swaps` table exists because Dune's indexers translated billions of hex logs into rows (exactly the transformation in our before/after figure). Your job is just to ask. A first query to count daily swaps on a pool looks like ordinary SQL:

```sql
select
  date_trunc('day', block_time) as day,
  count(*)                      as swaps,
  sum(amount_usd)               as volume_usd
from uniswap_v3.swaps
where pool = 0xUSDC_ETH_pool_address
  and block_time > now() - interval '30' day
group by 1
order by 1
```

Run it, get a table, click "New visualization," and you have a chart of daily volume on that pool. That is the whole loop. The full hands-on — picking tables, handling token decimals, building a dashboard — is its own walkthrough in [writing on-chain queries with Dune](/blog/trading/onchain/writing-onchain-queries-with-dune); this is the map-level view of where Dune sits in the toolbox.

#### Worked example: the cost of running queries at scale

Dune is free to learn on. The cost shows up when you go from clicking around to building something that runs queries *programmatically* — say a bot or a backend that hits the data on a schedule. Two ways that bites you.

On Dune, paid plans (on the order of **\$349–\$399 per month** for a "Plus"-style tier) give you API access and a monthly allotment of query *credits*; heavy automated use can push you to higher tiers. Separately, if you build directly on a managed RPC provider instead, you pay per request. Suppose your indexing job makes **1,000,000 RPC calls** in a month. On a provider that prices archive requests at roughly **\$2 per million compute units** (and a heavy call can cost several units), a million calls land somewhere around **\$20–\$60** for that month — cheap. But scale to a service making **50 million** calls a month and you are at **\$1,000–\$3,000 monthly**, at which point running your own node (that \$300–\$600/month box from earlier) suddenly *does* pencil out. *The intuition: querying is free until it is automated and high-volume; the crossover where renting a node beats paying per-call is real, and it arrives faster than beginners expect.*

### What lives in the decoded tables

It helps to know roughly what is on the shelf before you go shopping. On Dune, the data is organized into namespaced schemas. The *raw* layer mirrors the chain almost one-to-one: `ethereum.transactions`, `ethereum.logs`, `ethereum.traces` (internal calls). On top of that sit *decoded* tables, one set per protocol: `uniswap_v3.swaps`, `aave_v3.borrow`, `erc20_ethereum.evt_Transfer`. And above *those* sit *spell* tables — curated, cross-protocol views the community maintains, like a unified `dex.trades` that stitches every DEX's swaps into one table so you do not have to union twelve protocol schemas yourself.

The progression matters because it mirrors the raw-vs-curated distinction from the Foundations: the lower you query, the more faithful and the more work; the higher you query, the more convenient and the more you are trusting someone's stitching logic. A beginner should start at the decoded-protocol layer (faithful enough, readable) and only drop to raw `logs` when a protocol you need is not decoded yet — which, when it happens, is itself a signal that the protocol is new or obscure.

The other thing to internalize: **decimals and prices are where queries go wrong.** A token amount on-chain is an integer in the token's smallest unit; USDC has 6 decimals, ETH has 18. Forget to divide and your "5,000 USDC swap" reads as 5,000,000,000. And turning token amounts into dollars requires a *price at that block*, which the spell tables provide but which a naive raw query does not. Most embarrassing on-chain charts trace to one of these two slips.

### How query platforms deceive you

The deception here is self-inflicted and the most dangerous, because the output *looks* authoritative. A query that double-counts a token because of an unhandled proxy contract, or that misses a chunk of volume routed through an aggregator, will still produce a clean chart with a precise number. Community dashboards you fork inherit their authors' errors. The defense is the same as a spreadsheet: spot-check the totals against a neutral source (does your Dune volume roughly match DefiLlama's for the same protocol?), and never trust a single query you have not sanity-checked against the raw chain.

## Category 5: forensics suites — trace stolen funds

The final category is the one most readers will never log into, because it is enterprise-only and priced for institutions: **blockchain forensics and compliance**. These are the tools that exchanges, banks, and law enforcement use to trace illicit funds, screen addresses for sanctions exposure, and build the evidence that ends up in court.

- **Chainalysis** — the market leader; its *Reactor* (investigations) and *KYT* (Know Your Transaction, real-time screening) products are the de-facto standard for exchange compliance and government investigations. Its annual Crypto Crime Report is the most-cited source for stolen-fund and illicit-share numbers.
- **TRM Labs** and **Elliptic** — direct competitors with overlapping capabilities: entity attribution at scale, sanctions and risk scoring, cross-chain tracing through bridges and mixers.

What do they do that Arkham or Nansen do not? Three things. First, **proprietary attribution at scale and depth** — they run large clustering operations, ingest off-chain intelligence (law-enforcement tips, undercover data, leaked records), and maintain attribution that retail tools cannot match, especially for older or obfuscated flows. Second, **cross-chain and through-mixer tracing** — they specialize in following funds across bridges, through services like Tornado Cash, and out the other side, which is precisely where retail tools lose the thread. Third, **compliance-grade outputs** — risk scores and audit trails built to satisfy regulators and stand up as evidence. (For the legal seam around mixers specifically, see the case of [Tornado Cash and sanctioning code](/blog/trading/crypto/tornado-cash-and-sanctioning-code).)

When a forensics firm ranks the biggest thefts, this is the kind of leaderboard it produces — the same data that drives every "funds traced to…" headline.

![Horizontal bar chart of the biggest crypto hacks ranked by USD stolen, Bybit at the top](/imgs/blogs/the-onchain-tooling-landscape-7.png)

#### Worked example: a forensics tool ranking the Bybit hack

Put the leaderboard above into money terms. The **Bybit** theft of **\$1,460M** (\$1.46 billion) on 21 February 2025 is, by itself, larger than the next two combined: **Ronin** at **\$625M** (March 2022) and **Poly Network** at **\$611M** (August 2021). It is *more than double* Ronin. And it dwarfs the rest of the top eight — **DMM Bitcoin \$305M**, **Wormhole \$326M**, **WazirX \$230M**, **Cetus \$220M**, **Mixin \$200M** — each a major incident in its own right, several attributed to the same Lazarus Group that took Bybit.

A forensics suite does not just rank these; it follows them. In the Bybit case, within hours the funds began fanning out across hundreds of intermediary wallets, getting swapped from staked-ETH derivatives into native ETH (which is harder to freeze than a centrally-issued token), and bridging toward laundering rails. Where a retail explorer would show you a wall of transactions, a Chainalysis or Elliptic graph collapses them into the *route*. *The intuition: the gap between "a \$1.46B transaction left a wallet" and "here is the laundering route and which exchange deposit addresses the funds hit" is exactly the gap between a free explorer and an enterprise forensics suite — and it is why those tools cost what they cost.*

### What a forensics tool sees that you cannot

The retail-vs-enterprise gap is clearest in the patterns these tools are built to recognize on sight. When stolen funds move, they tend to follow a recognizable shape — and a forensics suite collapses that shape into a labeled route where an explorer would only show you transactions:

- **Fan-out (layering).** One wallet splits funds across hundreds of fresh addresses to break the single-thread trail. An explorer shows you 300 transfers; a forensics graph shows you *one source* fanning into 300 nodes — and keeps them grouped as one cluster.
- **Peel chains.** A large balance moves through a long sequence of addresses, "peeling off" a small amount to a cash-out at each hop while the bulk continues. Hard to follow by hand; trivial for a tool that tracks the main thread.
- **Bridge hopping.** Funds cross from Ethereum to another chain through a bridge, which is exactly where retail tools lose the thread (the destination-chain transaction is a *different* transaction with no on-chain link to the source). Forensics suites maintain cross-chain attribution to stitch the two sides back together.
- **Mixers.** Services like Tornado Cash deliberately break the link between deposit and withdrawal. Even here, forensics firms use timing, amount, and gas-funding correlations to attach *probabilistic* attribution that retail tools do not attempt.

This is purely the *defender's* lens: recognizing these patterns is how investigators and exchanges catch laundering and freeze funds, and how an investor avoids touching tainted coins. Knowing the shapes exists so you can *detect* them, not reproduce them — the mechanics here are deliberately at the recognition level, not an operational guide.

For the hands-on method of following money yourself with *free* tools — the peel-chain pattern, fan-out, bridge hops — see [how to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow). You will not match Chainalysis's attribution, but you can follow a surprising amount of the way.

## The pricing reality: free vs paid tiers

A practical map of what costs what, because it shapes which tools you actually reach for:

- **Free, always:** block explorers (Etherscan and family, Solscan, mempool.space), DefiLlama, DeBank/Zerion portfolio views, Bubblemaps' basic maps, the learning tier of Dune. You can do an enormous amount of serious analysis without paying a cent.
- **Freemium / subscription (roughly \$50–\$200/month for individuals):** Arkham's deeper features, Nansen, Glassnode/CryptoQuant pro tiers, Token Terminal, Dune's higher tiers. This is the band where you pay for *labels, alerts, and convenience* — curated signal on top of free raw data.
- **Enterprise (four to six figures a year):** Chainalysis, TRM, Elliptic, and high-volume RPC/indexing infrastructure. Priced for institutions, sold with contracts, not credit cards.

The honest guidance for a beginner: live entirely in the free tier until a *specific* recurring question justifies a subscription. Most people pay for Nansen or Glassnode long before they have a question those tools uniquely answer — and then they treat the green numbers as truth, which is worse than not having them. Buy a tool when you can name the exact trade or investigation it unblocks.

There is also a *layering* logic to spending that follows the pipeline. The further down the pipeline a tool sits, the more it costs and the more opinionated it is. Raw access (explorers, your own RPC) is cheapest and most faithful. The indexer/query layer (Dune, Flipside) is mid-priced and gives you faithful tables plus the freedom to ask anything. The curated-analytics layer (Nansen labels, Glassnode metrics) is the most expensive per insight because you are buying someone's *interpretation*, not just their data — and interpretation is where the value, and the bias, both live. A sensible progression for most analysts is: master the free explorers, then learn Dune (still mostly free, and it makes you a real analyst), and only *then* consider a curated subscription for the one signal you have proven you cannot reconstruct yourself.

## The decision flow: which tool for which question

Here is the reverse lookup — the flow you will actually run in your head. Start from the *shape* of your question and the tool follows.

![Decision flow routing an on-chain question to the right tool category by its shape](/imgs/blogs/the-onchain-tooling-landscape-3.png)

To make it concrete, the questions this series keeps asking and where each one goes:

- **"What exactly happened in this transaction?"** → a **block explorer** (Etherscan/Solscan). Read the status, token transfers, internal calls, and logs.
- **"Is this token a rug?"** → **Etherscan** (holder concentration, verified source, owner powers) plus **Bubblemaps** (are the "holders" one cluster?) plus the real liquidity. Free, every time.
- **"Where did the hacked money go?"** → **free explorers + tracing** for the path you can follow ([how to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow)); **forensics suites** (Chainalysis/TRM/Elliptic) for attribution and through-mixer tracing you cannot.
- **"Is smart money buying this?"** → **Nansen** (smart-money labels and flows), cross-checked against **Arkham** entity flows and the raw chain.
- **"What is this protocol's revenue / is it overvalued?"** → **Token Terminal** (fees, revenue, P/S) and **DefiLlama** (TVL, fees).
- **"What's Bitcoin's exchange reserve doing?"** → **Glassnode / CryptoQuant** (aggregate exchange balances, MVRV, SOPR).
- **"Who are the top holders of this token?"** → **Etherscan** Holders tab and **Bubblemaps** for clustering; **Arkham** to name the big ones.
- **"I have a custom question no dashboard answers."** → **Dune** (SQL over decoded tables) or **Flipside**.

Memorize the five categories and this routing and you can place any on-chain question in seconds. That is the actual skill the rest of the series builds on.

## The universal limitation: what no tool can see

Now the most important caveat in on-chain analysis, the one that humbles every tool in this post. **On-chain tools can only see what is on-chain.** The single biggest blind spot is the **centralized exchange internal ledger**.

![Pipeline showing on-chain visibility breaking at a centralized exchange's off-chain internal ledger](/imgs/blogs/the-onchain-tooling-landscape-5.png)

When you deposit to Binance or Coinbase, your coins go to an exchange-controlled deposit address — *visible on-chain*. But once they are credited, your balance and all your trades live in the exchange's **private internal database**, which is **off-chain**. If you trade, send funds to another user, or hold, none of it touches the blockchain. When you withdraw, coins leave an exchange hot wallet to your address — *visible again*. As the figure shows, the chain goes dark in the middle: an analyst sees money go *in* and money come *out*, but the matching between deposits and withdrawals is invisible.

This is why so much tracing "ends at an exchange." When stolen funds hit a CEX deposit address, the on-chain trail stops; from there only the exchange (via subpoena or its own compliance team using Chainalysis-style KYT) can connect the deposit to an account and a withdrawal. It is also why exchange reserve metrics are *estimates* — analysts identify exchange-controlled addresses heuristically, and they can be wrong or incomplete. The same blindness applies to anything off-chain: a deal negotiated in a Telegram chat, a wallet's real-world owner, an OTC trade settled bank-to-bank. The chain records *settlement*, not *intent* — and not the books of the intermediaries that sit on top of it.

## Common misconceptions

**"On-chain tools show you everything that happens in crypto."** No — they show you what *settles on a public blockchain*. Everything inside a centralized exchange (the majority of retail trading volume) is off-chain and invisible. A tool that shows "Binance reserves" is showing the addresses it *believes* Binance controls, not Binance's actual books.

**"A dashboard's number is a fact."** Only the raw-data layer is fact. The moment a number involves a *label* ("smart money"), a *cluster* ("this entity"), or an *aggregate that requires choices* ("TVL," "exchange reserves"), you are looking at a methodology, and two reputable tools will disagree. Always ask: is this a transaction (fact) or an interpretation (opinion)?

**"Paid tools are categorically better."** They are better at *specific jobs* — labels, alerts, attribution, scale. For most beginner questions (what's in this tx, is this a rug, how big is this protocol), the **free** tools are not just adequate, they are canonical: Etherscan *is* the reference, DefiLlama *is* the standard. Paying does not improve those; it adds curation you may not yet need.

**"If a tool labels a wallet 'smart money,' following it is an edge."** Smart-money labels are defined by *past* success and so suffer survivorship bias; the cohort is also widely watched, so its moves are partly front-run and priced in. The label is a lead to verify, not a signal to copy.

**"More tools = more edge."** More tools means more *surface area to be confidently wrong*. The edge is in asking the right question, picking the *one* tool that answers it, and verifying the answer against the raw chain — not in stacking ten dashboards whose green numbers you do not interrogate.

## The playbook: what to do with the toolbox

The if-then checklist for actually using this map.

- **Signal: you have any on-chain question at all.** → Read: classify it by shape — one tx, one entity, a market, a custom slice, or a hack. → Action: route it through the five categories (explorer → entity → metric → query → forensics). → Invalidation: if no category fits, your question is probably *off-chain* (about a CEX's internal books, an OTC deal, or someone's real identity) — no tool will answer it.

- **Signal: a dashboard shows a clean, precise number.** → Read: identify whether it is a *transaction* (fact) or an *interpretation* (label/cluster/aggregate). → Action: trust facts directly; for interpretations, check the methodology and cross-reference a second source. → False positive: a beautiful Dune chart can be precisely wrong — sanity-check its totals against a neutral aggregator before you act.

- **Signal: a tool labels a wallet "smart money" or names an entity.** → Read: treat it as a *lead*, not a verdict. → Action: verify the underlying flows on the raw chain (Etherscan/Arkham), check for survivorship bias and bait, and size accordingly. → Invalidation: if the "holders" or "smart wallets" turn out to be one cluster funded from a single source (Bubblemaps), the signal is manufactured — stand down.

- **Signal: you are about to pay for a subscription.** → Read: name the *specific recurring question* it uniquely unblocks and the trade or investigation that depends on it. → Action: if you can name it and the expected value clears the ~\$50–\$200/month cost, subscribe; otherwise stay on the free tier. → False positive: "everyone uses Nansen" is not a reason — pay for a tool only after the free tools have failed you on a real question.

- **Signal: a trace ends at an exchange deposit address.** → Read: you have hit the off-chain wall; the chain goes dark inside the CEX. → Action: note the exchange and the deposit address, and recognize that only the exchange (or law enforcement via subpoena and KYT) can go further. → Invalidation: do not invent a continuation of the trail past the deposit — fabricated "it went to person X" claims are how on-chain sleuthing gets people sued and gets the wrong person doxxed.

- **Signal: you need a question no dashboard answers.** → Read: this is a query-platform job. → Action: write it on Dune/Flipside over the decoded tables, *and* validate the result against a known-good aggregate before trusting it. → False positive: an unhandled proxy, missing decimals, or a forked dashboard's inherited bug will produce a confident wrong chart — your query is only as correct as your understanding of the schema.

The throughline of this entire series sits in one sentence: **the data is public and almost always trustworthy; the interpretations layered on top of it are opinions, and your edge is knowing the difference.** Match the question to the tool, read the raw layer when it matters, and never let a green number do your thinking for you.

## Further reading & cross-links

- [Anatomy of a transaction](/blog/trading/onchain/anatomy-of-a-transaction) — every field on an explorer's transaction page, dissected.
- [Addresses, wallets, and contracts](/blog/trading/onchain/addresses-wallets-and-contracts) — the From/To distinction and what an address really is.
- [How to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow) — following money hop-by-hop with free tools.
- [What is smart money on-chain](/blog/trading/onchain/what-is-smart-money-onchain) — how labels are built and why they mislead.
- [Writing on-chain queries with Dune](/blog/trading/onchain/writing-onchain-queries-with-dune) — the hands-on SQL walkthrough.
- [Exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — reading reserves as a supply signal.
- [Centralized crypto exchanges: Binance, Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) — why the internal ledger is the wall every tool hits.
- [Crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset) — why on-chain metrics move with global liquidity.
