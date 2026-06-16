---
title: "Writing On-Chain Queries with Dune: SQL From Zero to Your First Dashboard"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Learn to query decoded blockchain data with SQL on Dune — from your first SELECT to a live dashboard tracking exchange flows in dollars."
tags: ["onchain", "crypto", "dune", "sql", "ethereum", "dashboards", "exchange-flows", "data-analysis", "defi", "queries"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Dune is a SQL interface over indexed, decoded blockchain data with a charting and dashboard layer on top; learning to write a query is the single most empowering on-chain skill, because it turns "I wonder if money is leaving this exchange" into a number you can chart in minutes.
>
> - **What it is:** Dune indexes every block, decodes the raw events into named tables, and lets you query them with ordinary SQL — then chart the result and pin it to a dashboard.
> - **How to read it:** you describe *what you want* (SELECT these columns FROM this table WHERE this is true GROUP BY this), and Dune scans the chain history for you in seconds.
> - **What you DO with it:** build the metric *you* care about — daily exchange inflow, unique buyers, stablecoin mints — instead of trusting someone else's green number on a screenshot.
> - **The one rule:** reach for the **spellbook** tables (`tokens.transfers`, `dex.trades`, `prices.usd`) first; they already did the hard joining, so you write five lines instead of fifty.

On the morning of 21 February 2025, a routine-looking transaction left a Bybit cold wallet. Within hours, roughly **\$1.46 billion** of staked Ether and tokens had been drained — the largest crypto theft ever recorded, later attributed to North Korea's Lazarus group. The price of ETH barely twitched at first. But on dashboards built by anonymous analysts, the story was already legible: a cluster of fresh wallets receiving enormous transfers, then fanning the funds out across dozens of addresses and into decentralized exchanges. None of those analysts had inside access. They had a free account on a website and the ability to write a few lines of SQL.

That is the quiet superpower this post is about. The blockchain is a public ledger — every transfer, every swap, every contract call is recorded forever and visible to anyone. But "visible" is not the same as "readable." Raw blocks are a wall of hex. To turn that wall into an answer — *how much money flowed into Binance yesterday? how many unique wallets bought this token? did the team just mint \$50 million of a new stablecoin?* — you need a tool that has already indexed and decoded the chain, and a language to ask it questions. Dune is that tool, and SQL is that language.

This is a hands-on build guide. By the end you will understand the table landscape, write a first query, filter by address, aggregate flows into a daily series, join token transfers to prices to get dollar values, and turn the result into a chart you pin to a dashboard. We build one real metric end to end: **daily net exchange flow for a token, in USD.** No prior SQL and no crypto background assumed — we define every term the first time it appears.

![Dune pipeline from blockchain through indexed tables and SQL to a chart and dashboard](/imgs/blogs/writing-onchain-queries-with-dune-1.png)

## Foundations: what Dune is and why SQL

Let's build the picture from the ground up, because every later section leans on it.

### What a blockchain actually stores

A blockchain like Ethereum is, at heart, a giant append-only list of **transactions**. A transaction is a signed instruction: "address A sends 1 ETH to address B," or "address A calls function `swap` on contract C with these inputs." Transactions are grouped into **blocks** (roughly one block every 12 seconds on Ethereum), and each block points back to the previous one — that chain of pointers is what makes the history tamper-evident.

Two terms you need from the start. An **EOA** (externally owned account) is a normal wallet controlled by a private key — a person or a bot. A **contract** is code deployed at an address that runs when called. When a contract does something noteworthy — a token moves, a swap executes, a loan is liquidated — it emits an **event**, also called a **log**: a small record written into the block saying "this happened, with these values." Events are how the outside world learns what a contract did, and they are the raw material of almost all on-chain analysis.

The catch: on the raw chain, an event is stored as **encoded hex** — a topic hash and a blob of bytes. To know that a particular log means "Alice transferred 4,000 USDC to Binance," you have to know the contract's **ABI** (application binary interface — the schema that says how to decode its events) and apply it. Doing that yourself, block by block, for the entire history of a chain, is the indexing problem. It is a lot of engineering.

To make this concrete, a raw `Transfer` log on Ethereum looks roughly like this before decoding: a `topic0` equal to the Keccak hash of the event signature `Transfer(address,address,uint256)`, two more topics holding the indexed `from` and `to` addresses (left-padded to 32 bytes), and a `data` field holding the amount as a 32-byte hex number. Nothing in that blob says "USDC" or "4,000" — you have to know that this contract is USDC, that USDC has six decimals, and how to slice the bytes. Multiply that decoding by billions of logs across years of blocks and dozens of chains, and you have the reason most people never look at the chain directly. They look at a tool that has already done the slicing.

It also matters that the chain only stores *events the contracts chose to emit*. There is no global "balance changed" feed; a token contract emits a `Transfer` event by convention (the ERC-20 standard tells it to), and analysts rely on that convention. When a contract is non-standard or emits incomplete events, the data is genuinely harder to read — a real limit you will occasionally hit and should know is the contract's fault, not yours.

### What Dune does

Dune solves the indexing problem once, for everyone. It runs **indexers** that ingest every block of many chains (Ethereum, Bitcoin, Solana, Polygon, Arbitrum, Optimism, Base, and more), and **decoders** that use known ABIs to turn encoded logs into human-readable rows. The output is a warehouse of **tables** you can query with SQL, plus a layer for **charting** a query's result and assembling charts into a **dashboard** that anyone can open by URL.

So Dune is really three things stacked:

1. **An indexed, decoded copy of the chain**, organized as SQL tables.
2. **A query engine** (Dune runs on Trino/DuckDB-style SQL — the dialect is close to standard SQL with a few functions of its own).
3. **A visualization and dashboard layer** that takes the rows your query returns and draws them.

The free tier is generous enough to learn on and to build real dashboards: you can write queries, save them, chart them, and publish dashboards publicly. Heavy or scheduled use moves you onto paid plans that buy more compute, faster refreshes, and private queries — but nothing in this guide needs a paid plan.

One mental adjustment matters here. Dune is **not** a node you run, and it is **not** a live connection to a wallet. It is a data warehouse: a snapshot of the chain's history, refreshed continuously as new blocks settle, that you query like any other database. You are not "talking to Ethereum" when you run a query; you are reading a meticulously maintained copy of what Ethereum already did. That distinction explains both Dune's power (it has the whole history pre-decoded, so a question about three years of data is as easy as a question about today) and its limits (it lags the chain tip slightly and cannot see anything that has not been mined yet). We come back to those limits at the end, but plant the idea now: a warehouse of settled history, not a real-time tap.

A second piece of vocabulary you will see everywhere on Dune: a **query** is a saved piece of SQL with a name and a URL; a **visualization** is a chart attached to that query; a **dashboard** is a page of pinned visualizations. Every query on Dune is, by default, public and forkable — which is itself a superpower. When you find a great dashboard, you can open any of its queries, read the exact SQL behind the number, and **fork** it (copy it into your own account to modify). Most analysts learn faster by forking a working query and changing one filter than by writing from a blank editor. Treat the entire public library of Dune queries as your textbook.

### Why SQL — the declarative idea

If you have only ever written normal step-by-step code, SQL feels strange at first, and then it feels like a gift. The key idea is that SQL is **declarative**: you describe *what you want*, not *how to get it*. You do not write a loop that walks every transaction. You write a sentence — "give me the daily sum of inflows to this address" — and the engine figures out how to scan the data efficiently.

#### Worked example: the question behind a query

Say you want to know how much money flowed into a specific exchange wallet yesterday. In step-by-step code you would: open the chain, loop over every block in the last 24 hours, for each block loop over every transaction, decode each log, check whether it is a transfer to your address, and if so add the amount to a running total. That is hundreds of lines and it scans millions of rows by hand.

In SQL it is one sentence: select the sum of `amount` from the transfers table where the recipient is your address and the day is yesterday. If yesterday's inflows were 4,000 ETH and ETH was about \$3,000, the query returns a single number — roughly **\$12 million** — and you got there by describing the answer, not coding the search. That shift, from *how* to *what*, is why SQL is the right language for a public ledger.

### The query → visualization → dashboard pipeline

Here is the workflow you will repeat for the rest of your on-chain life. You write a **query** and run it; Dune returns a small **result table** (rows and columns). You attach a **visualization** to that query — pick a chart type (bar, line, counter, table) and map columns to axes. You then **pin** that visualization onto a **dashboard**, a page that can hold many panels from many queries. The dashboard re-runs its queries on a schedule, so it stays current, and you share it with a single link. We will walk this pipeline end to end in the final sections; for now, hold the shape: *one query is one answer; a dashboard is a collection of answers that updates itself.*

## The table landscape: raw, decoded, and spellbook

The hardest part of Dune for a beginner is not the SQL — it is knowing **which table to read**. Get the table right and the query is short; get it wrong and you fight the data. Dune's tables come in three tiers, and the whole game is to reach for the highest tier that answers your question.

![Matrix comparing raw, decoded, and spellbook tables by example, contents, effort, and use](/imgs/blogs/writing-onchain-queries-with-dune-2.png)

### Tier 1 — raw tables (the chain as-is)

The raw tables are the chain exactly as recorded, per chain. The two you will meet first are:

- `ethereum.transactions` — one row per transaction: sender, recipient, value, gas used, gas price, block time, and the input data (still encoded).
- `ethereum.logs` — one row per emitted event: the contract that emitted it, the topics, and the data — **still hex-encoded**.

Raw tables are complete but unfriendly. To read a `Transfer` event out of `ethereum.logs` you would have to know its topic hash and manually slice the data field. You reach for raw tables only when nothing higher exists — for example, to count *all* transactions in a block regardless of type, or to inspect gas behavior.

### Tier 2 — decoded tables (events with names)

When a contract's ABI is submitted to Dune (anyone can submit popular ones), Dune generates **decoded tables**: one table per contract, per event, with named columns. The naming convention is roughly `<project>_<chain>.<contract>_evt_<EventName>`. The most useful general one is the standard token-transfer table:

- `erc20_ethereum.evt_Transfer` — every ERC-20 token transfer on Ethereum, with columns `contract_address` (which token), `from`, `to`, `value` (the raw amount), and `evt_block_time`.

Decoded tables are named and typed, so the SQL reads like English. The cost is that they are *per contract*: amounts are in the token's smallest unit (you still divide by `10^decimals` to get human units), and to value a flow in dollars you must join to a price table yourself.

### Tier 3 — spellbook tables (the pre-joined shortcuts)

The top tier is the **Spellbook** — a community-maintained set of curated, cross-chain, pre-joined tables that already did the annoying work. These are the tables you should reach for **first**. The headline three:

- `tokens.transfers` — token transfers across many chains, already normalized, with the amount in human units and the token symbol attached. (One table for everything, instead of one per contract.)
- `dex.trades` — every decentralized-exchange swap across Uniswap, Curve, PancakeSwap, and dozens more, **already priced in USD** via the `amount_usd` column. This single table answers most "DEX volume" questions in two lines.
- `prices.usd` — the USD price of thousands of tokens, by hour. This is what you join transfers to in order to value a flow.

The rule of thumb that will save you the most time: **search the Spellbook before you touch a decoded or raw table.** If `dex.trades` already has `amount_usd`, you do not write a join to `prices.usd` — the spell did it for you.

### Reading a single transfer row

Before aggregating millions of rows, it pays to understand what *one* row looks like, because every later query is just a filter and a sum over rows of this shape. A single row of `tokens.transfers` carries these columns:

```sql
-- The columns of one tokens.transfers row (a single token movement)
-- block_time          -- when it happened (a timestamp)
-- blockchain          -- 'ethereum', 'arbitrum', 'base', ...
-- contract_address    -- which token moved
-- symbol              -- 'USDC', 'WETH', ... (human label)
-- "from"              -- the sender address
-- "to"                -- the recipient address
-- amount              -- the amount in HUMAN units (decimals already applied)
-- amount_raw          -- the amount in the token's smallest unit
-- tx_hash             -- the transaction this transfer belongs to
```

This is the unit of on-chain accounting: one party sent some amount of one token to another party at a point in time, inside a transaction. A "flow" is nothing more than a `sum(amount)` over many such rows that share a property — same recipient, same token, same day. An "active address count" is a `count(distinct "from")` over rows in a window. Once you see that every metric is a filter-then-aggregate over rows of this shape, the whole field of on-chain analysis stops being a pile of jargon and becomes a small number of SQL patterns applied to one table shape. The rest of this post is those patterns.

A note on why the spellbook's `amount` (human units) versus the decoded table's `value` (raw units) matters so much: the decoded `erc20_ethereum.evt_Transfer` table gives you `value` in the token's smallest unit, so a transfer of 4,000 USDC shows up as `4000000000` (4,000 × 10⁶). The spellbook `tokens.transfers` table already divided by the decimals, so the same transfer shows up as `4000`. Mixing the two up — summing raw `value` and thinking it is dollars — produces numbers off by six or eighteen orders of magnitude. When a result looks absurdly large, suspect un-applied decimals first.

## Your first query: SELECT and WHERE

Enough landscape. Let's write SQL. The smallest useful query has two clauses: `SELECT` (which columns you want) and `FROM` (which table). A third clause, `WHERE`, filters which rows you keep. Here is a first query that counts how many transactions happened on Ethereum today.

```sql
-- Count today's Ethereum transactions
SELECT count(*) AS tx_count
FROM ethereum.transactions
WHERE block_time >= date_trunc('day', now())
```

Read it as a sentence. *Select the count of all rows, call it `tx_count`, from the transactions table, keeping only rows whose `block_time` is on or after the start of today.* `count(*)` counts rows; `AS tx_count` names the output column; `date_trunc('day', now())` chops the current timestamp down to midnight, so the filter means "since midnight." Run it and you get one number — typically over a million on a busy day.

Two beginner notes. First, `--` starts a comment in SQL; everything after it on the line is ignored, which is how you annotate queries. Second, **always filter on `block_time` early.** The transactions table holds years of history; without a time filter you ask the engine to scan everything, which is slow and burns compute. A time filter is the single most important habit in on-chain SQL.

The Dune editor itself is friendly to a beginner. You write SQL in a panel, hit run, and the result table appears below; a schema browser on the side lets you search table and column names without leaving the page, and an autocomplete suggests columns as you type. When something breaks, the error message points at the line — usually a typo, a missing quote on `"to"`, or a column that does not exist on the table you chose. The fastest way to learn the available columns is to run a tiny exploratory query first: `SELECT * FROM tokens.transfers LIMIT 10` shows you ten real rows with every column, so you see exactly what you are working with before you write the real aggregation. Get in the habit of peeking at ten rows of any unfamiliar table before you trust a sum over it.

### Anatomy of a query

Every query you ever write is built from the same four clauses, asked in a fixed logical order. Internalize this shape and SQL stops being mysterious.

![Grid showing the four SQL clauses SELECT FROM WHERE GROUP BY and the result table](/imgs/blogs/writing-onchain-queries-with-dune-3.png)

- **`SELECT`** — which columns (or aggregates) to keep in the answer.
- **`FROM`** — which table to read.
- **`WHERE`** — which rows to keep (the filter).
- **`GROUP BY`** — how to roll many rows up into summary rows (we get to this next).

The engine logically reads `FROM` first (get the table), then `WHERE` (drop rows), then `GROUP BY` (bucket what's left), then `SELECT` (compute the output columns). You *write* them in SELECT-FROM-WHERE-GROUP BY order, but it helps to *think* in FROM-WHERE-GROUP-SELECT order.

## Filtering by address: an exchange's inflows

Now we ask a real question: *how much of a token is flowing into a particular exchange wallet?* This is the heart of **exchange-flow analysis** — coins moving onto an exchange are coins positioned to be sold, so a surge of inflows is a supply-side warning. (We treat the trading meaning of this in depth in [exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows); here we just learn to compute it.)

Exchanges hold their reserves in known, labeled wallets. Using the decoded transfer table, an inflow is any `Transfer` whose `to` field is the exchange wallet. Here is the query for USDC inflows to one wallet over the last day.

```sql
-- USDC transferred INTO one exchange wallet in the last 24h
SELECT count(*)        AS transfer_count,
       sum(value) / 1e6 AS usdc_in        -- USDC has 6 decimals
FROM erc20_ethereum.evt_Transfer
WHERE contract_address = 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48  -- USDC token
  AND "to"             = 0xAAAA000000000000000000000000000000000001  -- exchange wallet (illustrative)
  AND evt_block_time  >= now() - interval '1' day
```

A few things to notice. We filter on **two** addresses: `contract_address` picks the token (this is the real USDC contract address on Ethereum), and `"to"` picks the recipient. The recipient here is an **illustrative placeholder** — in a real query you would paste the exchange's actual labeled wallet, which Dune and labeling services publish. We divide `value` by `1e6` because USDC stores amounts in units of `0.000001` (six decimals); dividing converts to whole USDC. And `now() - interval '1' day` is the rolling 24-hour window.

#### Worked example: a \$12M inflow read off one filter

Suppose this query returns `transfer_count = 318` and `usdc_in = 12,400,000`. That single `WHERE` clause just told you that **\$12.4 million of USDC moved into this one exchange wallet in a day**, across 318 separate transfers. If this wallet's normal daily inflow is around \$3 million, a jump to \$12.4 million is a 4× spike — the kind of supply build-up that often precedes selling pressure. The point is how little work it took: two filters and a sum turned the raw chain into a dollar figure and a signal.

A subtlety worth flagging early: the address literals on Dune are written as bare hex (`0xa0b8…`), not quoted strings, and column names that are SQL keywords — like `to` and `from` — must be wrapped in double quotes (`"to"`, `"from"`). Forgetting the quotes is the single most common first-query error.

## Aggregation: SUM, GROUP BY, and COUNT DISTINCT

A single sum is useful, but the real power shows up when you **aggregate** — collapse many rows into summary rows. The two workhorses are `sum()` (add up a column) and `count(distinct …)` (count unique values). The clause that turns them from "one number" into "a number per bucket" is `GROUP BY`.

### Daily netflow with GROUP BY

So far we got one number for the whole window. To get a **daily series** — one row per day — we bucket the rows by day and sum within each bucket. `date_trunc('day', evt_block_time)` chops each timestamp to its date; grouping by that date gives one row per day.

```sql
-- Daily USDC inflow to one exchange wallet, last 30 days
SELECT date_trunc('day', evt_block_time) AS day,
       sum(value) / 1e6                   AS usdc_in
FROM erc20_ethereum.evt_Transfer
WHERE contract_address = 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
  AND "to"             = 0xAAAA000000000000000000000000000000000001
  AND evt_block_time  >= now() - interval '30' day
GROUP BY 1                 -- group by the first SELECT column (day)
ORDER BY 1
```

`GROUP BY 1` is shorthand for "group by the first column in the SELECT list," which is `day`. `ORDER BY 1` sorts the output chronologically. The result is exactly the shape a line chart wants: a date column and a value column. This is the **time-bucketing** pattern, and you will use it in almost every dashboard query.

### COUNT DISTINCT: unique buyers behind the volume

Volume can be one whale trading with itself, or it can be a thousand real buyers. `count(distinct …)` tells them apart by counting *unique* values rather than rows. To count how many distinct wallets bought a token, you count distinct buyer addresses.

```sql
-- Unique buyers of a token on DEXes in the last day
SELECT count(distinct taker) AS unique_buyers,
       sum(amount_usd)       AS volume_usd
FROM dex.trades
WHERE token_bought_address = 0xBBBB000000000000000000000000000000000002  -- the token (illustrative)
  AND block_time          >= now() - interval '1' day
```

Notice we switched tables to `dex.trades` (a spellbook table) because it already carries `amount_usd` and a `taker` (the buyer) column — no join needed. `count(distinct taker)` counts unique buyers; `sum(amount_usd)` adds up the dollar volume.

#### Worked example: 800 buyers behind \$5M of volume

Say the query returns `unique_buyers = 800` and `volume_usd = 5,000,000`. That is **\$5 million of buying spread across 800 distinct wallets**, an average of about \$6,250 per wallet. Broad and granular — the profile of organic retail demand. Contrast a token that shows the same **\$5 million** of volume but only `unique_buyers = 12`: that is a dozen wallets, possibly one entity wash-trading to fake activity. Same dollar headline, opposite meaning — and one `count(distinct)` is what separates them. (We dig into telling real demand from fake in [detecting fake volume vs organic demand](/blog/trading/onchain/detecting-fake-volume-vs-organic-demand).)

### Ranking: ORDER BY, LIMIT, and top-N wallets

Aggregation also powers **leaderboards** — "who are the biggest buyers?", "which tokens saw the most volume?" The shape is: group by the entity, sum the metric, sort descending, keep the top N. Sorting is `ORDER BY <column> DESC`; keeping the top few is `LIMIT N`.

```sql
-- Top 20 buyers of a token by USD bought, last 7 days
SELECT taker                AS buyer,
       sum(amount_usd)      AS bought_usd,
       count(*)             AS trades
FROM dex.trades
WHERE token_bought_address = 0xBBBB000000000000000000000000000000000002
  AND block_time          >= now() - interval '7' day
GROUP BY 1
ORDER BY 2 DESC            -- sort by bought_usd, largest first
LIMIT 20
```

`GROUP BY 1` buckets by the buyer; `sum(amount_usd)` totals each buyer's spend; `ORDER BY 2 DESC` sorts by the second output column (`bought_usd`) largest-first; `LIMIT 20` keeps the top twenty. The result is a ranked table — exactly what you display as a "top buyers" panel or feed into wallet research. If you want to keep only buyers above a threshold, you filter on the *aggregate* with `HAVING` (the post-aggregation cousin of `WHERE`): adding `HAVING sum(amount_usd) > 100000` would keep only wallets that bought more than \$100,000. `WHERE` filters rows *before* grouping; `HAVING` filters groups *after* — a distinction that confuses every beginner exactly once.

## The decoded and spellbook tables in action

The two queries above already used both worlds: a decoded per-contract table (`erc20_ethereum.evt_Transfer`) and a spellbook table (`dex.trades`). It's worth seeing why the spellbook is so much less work.

### DEX volume in two lines

To compute total daily DEX volume for a token, the spellbook hands you everything pre-priced.

```sql
-- Daily DEX volume (USD) for one token
SELECT date_trunc('day', block_time) AS day,
       sum(amount_usd)               AS volume_usd
FROM dex.trades
WHERE token_bought_address = 0xBBBB000000000000000000000000000000000002
  AND block_time          >= now() - interval '90' day
GROUP BY 1
ORDER BY 1
```

That is the entire query. Without the spellbook you would query each DEX's own swap events separately, normalize amounts across token decimals, join to a price source, and union it all together — fifty lines instead of five. `dex.trades` already aggregated every major DEX and priced each trade, which is exactly the kind of leverage the spellbook gives you. (For the trading read on this data, see [analyzing DEX and AMM activity](/blog/trading/onchain/analyzing-dex-and-amm-activity).)

### prices.usd: valuing anything

When a table is *not* pre-priced — like the raw transfer tables — you value it by joining to `prices.usd`, which carries a token's USD price per hour. That join is important enough to get its own section, because it is the move that turns token counts into money.

## JOINs: turning token amounts into dollars

A token amount is not money. "4,000 ETH moved" means very different things at \$1,500 and at \$3,000. To make a flow comparable and chartable in dollars, you attach a price to each transfer — and the SQL mechanism for attaching one table's data to another is the **JOIN**.

![Graph showing transfers and prices tables joining to produce USD values that get summed](/imgs/blogs/writing-onchain-queries-with-dune-4.png)

A JOIN matches rows from two tables on a shared key. To value transfers, the shared keys are the **token** and the **time bucket**: for each transfer of token X at hour H, find the row in `prices.usd` for token X at hour H, and multiply the amount by that price. Here is the pattern using the spellbook transfer table.

```sql
-- Value daily inflows of a token in USD, by joining to prices
SELECT date_trunc('day', t.block_time)     AS day,
       sum(t.amount * p.price)             AS inflow_usd
FROM tokens.transfers AS t
JOIN prices.usd       AS p
  ON  p.contract_address = t.contract_address
  AND p.blockchain       = t.blockchain
  AND p.minute           = date_trunc('minute', t.block_time)
WHERE t."to"        = 0xAAAA000000000000000000000000000000000001   -- exchange wallet
  AND t.block_time >= now() - interval '30' day
GROUP BY 1
ORDER BY 1
```

Read the JOIN as a lookup. For each transfer row `t`, the `ON` clause finds the matching price row `p` — same token (`contract_address`), same chain (`blockchain`), same minute bucket. Then `t.amount * p.price` is that transfer's dollar value, and `sum(...)` adds them up per day. We alias the tables (`t` and `p`) so we can say which table a column comes from; `t."to"` is the transfer's recipient, `p.price` is the matched price.

#### Worked example: 4,000 ETH valued at \$12M through one join

Suppose on a given day this exchange wallet received transfers totaling **4,000 ETH**, and `prices.usd` shows ETH averaging about **\$3,000** that day. The join attaches \$3,000 to each ETH transfer; the multiplication and sum return roughly **\$12 million** of inflow for the day. Without the join, your chart would show "4,000" on the y-axis — a number that means nothing across time as ETH's price swings. With the join, it shows **\$12 million**, directly comparable to last week's **\$8 million** and last month's **\$20 million**. The JOIN is what makes a flow series honest.

One practical caution: a JOIN can *multiply* rows if the keys are not unique, and it can *drop* rows if a price is missing for some hour (a left join versus an inner join changes which). For a first dashboard, prefer spellbook tables that are already priced (`dex.trades.amount_usd`) and reserve manual price joins for tokens or flows the spellbook does not cover.

## Time-bucketing and the shape of a series

We have used `date_trunc('day', …)` several times — it is the engine of every time series. `date_trunc(unit, timestamp)` rounds a timestamp down to the start of the given unit: `'day'` gives midnight, `'hour'` gives the top of the hour, `'week'` gives the week's start, `'month'` gives the first of the month. Grouping by a truncated timestamp is how "millions of individual transfers" becomes "one row per day."

The choice of bucket is a real decision. Too fine (per-minute) and your chart is noise; too coarse (per-month) and you miss the spike that *was* the signal. For exchange-flow dashboards, daily buckets are the default — granular enough to catch a one-day surge, smooth enough to read a trend. For intraday event analysis (a hack unfolding, a depeg), drop to hourly.

A second-order point that trips up beginners: **a missing bucket is not a zero.** If no transfers happened on a given day, `GROUP BY day` simply produces *no row* for that day, leaving a gap in your chart rather than a zero. When you need a continuous daily axis (so the chart doesn't compress gaps), you generate a full sequence of days and **left join** your data onto it — a "calendar spine." It's an intermediate trick, but worth knowing it exists the first time your chart looks suspiciously gappy.

There is also a subtle timezone trap. `date_trunc('day', …)` buckets in UTC, which is the chain's native clock — every block timestamp is UTC. That is the *right* choice for on-chain work (it matches how the chain records time and how every other analyst buckets), but it means "today" on your chart ends at midnight UTC, not midnight in your local timezone. When you compare an on-chain daily figure to a number from a centralized exchange that reports in, say, US Eastern time, the day boundaries will not line up and the totals will differ slightly. Keep everything in UTC and the on-chain numbers stay internally consistent.

Finally, the bucket interacts with the **noise versus signal** tradeoff in a way worth internalizing. A per-minute exchange-flow chart is almost unreadable: every block adds a spike, and a single \$2 million transfer dominates the frame even though it is routine. A daily chart smooths those individual transfers into a trend you can actually act on — you see that *this week* averaged \$15 million of net inflow against last week's \$4 million, which is a real shift, rather than fixating on one large transfer that happened to land at 14:32. Pick the coarsest bucket that still resolves the signal you care about; for most flow and supply metrics, that is one day.

## Parameters: making a query reusable

You do not want to hard-code one exchange wallet and one token forever. Dune supports **parameters** — placeholders in the SQL that become input boxes on the dashboard, so a viewer can swap the address or token without editing SQL. A parameter is written `{{name}}`.

```sql
-- Daily inflow for ANY token to ANY wallet (parameterized)
SELECT date_trunc('day', block_time) AS day,
       sum(amount)                    AS token_in
FROM tokens.transfers
WHERE "to"                = {{wallet}}
  AND contract_address    = {{token}}
  AND block_time         >= now() - interval '{{days}}' day
GROUP BY 1
ORDER BY 1
```

Now the same query powers a panel where a viewer types a wallet, a token, and a number of days. One query, infinite questions. Parameters are what turn a one-off query into a reusable **tool** — and they are how a good dashboard lets non-SQL colleagues explore the data themselves. Default values you set make the query run sensibly out of the box.

Dune supports a few parameter *types*. A **text** parameter (the default) drops the typed value straight into the SQL — good for addresses and token symbols. A **number** parameter constrains input to digits — good for the `days` window above. And a **dropdown** (list) parameter restricts the choices to a fixed menu — good when you want a viewer to pick from "Binance / Coinbase / Kraken" rather than paste a raw address. The dropdown is the safest for a public dashboard, because it stops a confused viewer from typing something that breaks the query.

There is one real caution with text parameters: because the value is substituted directly into the SQL, a parameter is a place where a careless query *could* be manipulated. On Dune this is contained — queries are read-only against the warehouse, so the worst case is a viewer seeing data, not changing it — but the habit of constraining inputs (numbers as numbers, choices as dropdowns) is the right one to build, and it carries over to every other database you will ever touch. For the on-chain analyst, the everyday payoff is simpler: a well-parameterized query is one you write *once* and reuse for every token and every exchange for the rest of the cycle, instead of editing SQL each time the question changes slightly.

## Turning a query into a chart, and pinning it to a dashboard

A query returns a table of rows. A **visualization** turns that table into a picture. On Dune, after a query runs you add a visualization to it, choose a chart type, and map columns to axes:

- **Counter** — a single big number. Map it to a one-row, one-column result (today's total inflow). Great for "the headline number" panels.
- **Bar chart** — categories on one axis, a value on the other. Map `day` to x and `volume_usd` to y for daily DEX volume.
- **Line / area chart** — a value over time. Map `day` to x and `inflow_usd` to y for a flow trend. This is the default for any `date_trunc` series.
- **Table** — the raw rows, sortable. Useful for "top 20 wallets" lists.

![Pipeline from a saved query to a visualization to a dashboard panel that refreshes and shares](/imgs/blogs/writing-onchain-queries-with-dune-5.png)

Once a visualization exists, you **pin** it onto a **dashboard** — a page that holds many panels. You arrange panels in a grid, give the dashboard a title, and publish. Dune re-runs the underlying queries on a schedule (the cadence depends on your plan), so the dashboard stays live without you touching it. You share it with one URL; anyone who opens it sees the current numbers and, if you added parameters, can re-run with their own inputs.

#### Worked example: a dashboard panel tracking a \$50M stablecoin mint

A **mint** is the creation of new tokens; on-chain it appears as a `Transfer` event whose `from` address is the zero address (`0x0000…0000`) — tokens coming from "nowhere." Tracking mints of a stablecoin is the **dry-powder** signal: freshly minted stablecoins are dollars about to enter the market. (We treat this signal fully in [stablecoin flows: the dry-powder metric](/blog/trading/onchain/stablecoin-flows-the-dry-powder-metric).)

```sql
-- Daily stablecoin mints (from the zero address)
SELECT date_trunc('day', evt_block_time) AS day,
       sum(value) / 1e6                   AS minted     -- 6-decimal stablecoin
FROM erc20_ethereum.evt_Transfer
WHERE contract_address = {{stablecoin}}
  AND "from"           = 0x0000000000000000000000000000000000000000   -- mint
  AND evt_block_time  >= now() - interval '90' day
GROUP BY 1
ORDER BY 1
```

Suppose one day's row reads `minted = 50,000,000`. That panel just flagged a **\$50 million mint** of the stablecoin in a single day. Pin it next to a redemptions panel (mints where `"to"` is the zero address — tokens burned) and a cumulative-supply line, and you have a live **dry-powder dashboard**: on a day the panel jumps to **\$50 million** of net new supply, you know dollars are being staged to buy. The whole panel is one query, one chart, one pin — and it updates itself.

## How to read it: building daily net exchange flow end to end

Let's tie everything together by building one genuinely useful metric from scratch: **daily net exchange flow for a token, in USD.** Net flow is inflows minus outflows: positive net flow means coins are piling onto exchanges (supply-side pressure), negative means coins are leaving for self-custody (often a bullish accumulation signal). This is one of the most-watched on-chain metrics, and now you can build it yourself.

The logic in plain words: inflows are transfers *to* the exchange's wallets; outflows are transfers *from* them. We compute both per day, value them in USD, and subtract. We use the spellbook `tokens.transfers` (human-unit amounts, cross-chain) and join to `prices.usd`. To handle "many exchange wallets," we test membership with `IN (...)`.

Why this metric is worth the effort: it is one of the few on-chain signals with a genuine *lead* over price. When a holder wants to sell on a centralized exchange, the coins have to *arrive there first* — a transfer onto the exchange precedes the sell order, sometimes by hours or days. So a build-up of exchange inflows is supply being positioned ahead of selling, visible to you before it hits the order book. The reverse — coins leaving exchanges for self-custody — is supply being taken off the table, the on-chain fingerprint of holders who do not intend to sell soon. Neither is a crystal ball (a transfer onto an exchange can be a deposit for *buying* with, or an internal reshuffle), but in aggregate, across an asset's whole exchange footprint, net flow has repeatedly led price at major turns. The reason most people cannot use it is that computing it requires exactly the SQL you are about to write. That asymmetry — a public signal that takes a query to extract — is the whole thesis of on-chain analysis.

```sql
-- Daily NET exchange flow for one token, in USD
WITH ex_wallets AS (
    -- the exchange's known deposit + hot wallets (illustrative placeholders)
    SELECT addr FROM (VALUES
        (0xAAAA000000000000000000000000000000000001),
        (0xAAAA000000000000000000000000000000000002)
    ) AS w(addr)
),
priced AS (
    SELECT t.block_time,
           t."from"            AS sender,
           t."to"              AS recipient,
           t.amount * p.price  AS usd
    FROM tokens.transfers AS t
    JOIN prices.usd       AS p
      ON  p.contract_address = t.contract_address
      AND p.blockchain       = t.blockchain
      AND p.minute           = date_trunc('minute', t.block_time)
    WHERE t.contract_address = {{token}}
      AND t.block_time      >= now() - interval '{{days}}' day
)
SELECT date_trunc('day', block_time) AS day,
       sum(CASE WHEN recipient IN (SELECT addr FROM ex_wallets) THEN usd ELSE 0 END) AS inflow_usd,
       sum(CASE WHEN sender    IN (SELECT addr FROM ex_wallets) THEN usd ELSE 0 END) AS outflow_usd,
       sum(CASE WHEN recipient IN (SELECT addr FROM ex_wallets) THEN usd ELSE 0 END)
         - sum(CASE WHEN sender IN (SELECT addr FROM ex_wallets) THEN usd ELSE 0 END) AS net_flow_usd
FROM priced
GROUP BY 1
ORDER BY 1
```

This is the most advanced query in the post, so let's unpack the new pieces. `WITH name AS (...)` defines a **CTE** (common table expression) — a named sub-result you can reference later, which keeps complex queries readable. We define two: `ex_wallets` (the list of exchange addresses, built from literal `VALUES`) and `priced` (every relevant transfer, already valued in USD by the price join). The final `SELECT` then computes, per day, a `CASE WHEN … THEN usd ELSE 0 END` sum — `CASE` is an inline if/then, so this adds up only the dollars whose recipient is an exchange wallet (inflow), only those whose sender is one (outflow), and subtracts to get net.

The steps you would take in the Dune editor:

1. **Find the wallets.** Replace the placeholder `ex_wallets` with the exchange's real labeled addresses (from Dune's labels, a labeling service, or the exchange's published proof-of-reserves wallets). Getting this list right is the whole accuracy of the metric.
2. **Run it on a short window first.** Set `{{days}}` to `7` and run. Confirm the numbers look sane and the query finishes fast before you ask for a year.
3. **Chart it.** Add a line visualization: `day` on x, `net_flow_usd` on y. Add a zero reference line so positive (inflow-heavy) and negative (outflow-heavy) days are visually obvious.
4. **Add the components.** Optionally add `inflow_usd` and `outflow_usd` as two more lines so a viewer sees what drives the net.
5. **Pin and publish.** Pin the chart to a new dashboard titled e.g. "Token X exchange flow," set the parameters' defaults, and publish. Share the URL.

#### Worked example: reading a net-flow chart

Suppose your finished panel shows, over a week: net flow of **+\$20 million**, **+\$15 million**, **+\$30 million**, then **−\$5 million**, **−\$25 million**. The first three days, coins worth tens of millions piled onto the exchange — say **\$65 million** of net inflow over three days, classic supply build-up that often precedes selling. Then the last two days flip to **−\$30 million** of net outflow: coins leaving for self-custody, which historically marks accumulation. You built that read — the lead indicator that does not show up in price until later — out of one query and one chart. That is the edge the public ledger gives anyone willing to write SQL.

## Performance and cost: querying without burning credits

Dune's compute is metered, and even on the free tier, a sloppy query is slow. A few habits keep queries fast and cheap, and they are the same habits that make your numbers correct.

- **Filter on time first, always.** A `WHERE block_time >= …` clause lets the engine skip whole partitions of history. A query without a time filter scans years of data to answer a question about yesterday.
- **Filter on the indexed columns** (token address, time) before computing aggregates. The engine drops rows cheaply at the `WHERE` stage; an aggregate over fewer rows is faster.
- **Prefer spellbook tables.** `dex.trades` is already aggregated and priced; reading it is far cheaper than unioning a dozen raw DEX tables and joining a price source yourself.
- **Test on a small window.** Develop on `interval '7' day`, then widen to `'90' day` or `'365' day` only once the query is correct. Iterating on a year of data wastes minutes and credits per run.
- **Avoid `SELECT *` on big tables.** Name the columns you need; pulling every column of `ethereum.logs` is far heavier than pulling three.

#### Worked example: the cost of a missing time filter

Say `ethereum.transactions` holds roughly **2 billion** rows of history, and the day you care about is **1.5 million** of them. A query with `WHERE block_time >= date_trunc('day', now())` scans about 1.5 million rows; the same query without the time filter scans all 2 billion — over a thousand times more work for the identical one-number answer. On a metered plan that is the difference between a free, instant query and one that eats a noticeable slice of your monthly compute. The time filter is free correctness *and* free speed.

A related habit that compounds with the above: when a dashboard has several panels that all read the same heavy table for the same window, lift the shared work into a single base query and have the panels read *that*, rather than each panel re-scanning the raw table. Dune lets one query reference another (you can query a query's result), so a "load and price the last 90 days of this token's transfers once" base query can feed an inflow panel, an outflow panel, a net-flow panel, and a top-wallets panel — four charts, one expensive scan instead of four. On a flow dashboard refreshing every few hours, that is the difference between a dashboard that stays comfortably inside the free tier and one that constantly bumps its compute ceiling. Treat compute the way you treat correctness: something you design in from the first line, not something you bolt on after the query is slow.

## Common misconceptions

A few myths trip up new on-chain analysts. Each is worth correcting with a concrete mechanism, because believing them leads to wrong reads.

**"The dashboard shows what's happening *right now*."** It does not. Dune queries **settled history** — blocks that have already been indexed and decoded. There is **indexing lag**: the most recent minutes (sometimes the last hour, more on busy chains) may not be in the tables yet. And Dune has *no view of the mempool* — the pending transactions waiting to be included in a block. If you need pre-confirmation visibility (front-running, a hack mid-flight), Dune is the wrong tool; you need a mempool-watching service. Treat the latest bucket on any chart as provisional, and compare *yesterday's* complete day, not the last ten minutes.

**"If the number is on a chart, it's true."** A query is only as honest as its table choice, its filters, and its address list. A net-flow chart built on an incomplete exchange-wallet list will understate flows; a "volume" chart on `dex.trades` includes wash trades unless you filter them out. The chart faithfully reports what you asked — which is not always what you meant. Always sanity-check a new query against a known number (does total supply match the issuer's published figure?) before you trust it.

**"More token volume always means more demand."** Volume counts trades, not traders. A single wallet trading with itself can manufacture millions in volume. `count(distinct buyer)` is the antidote: \$5 million of volume across 800 wallets is demand; \$5 million across 12 wallets is suspicious. Never read a volume number without its distinct-address companion.

**"Raw tables are the 'real' data, so I should use them."** Raw tables are *complete*, but the spellbook tables are *correct and convenient* — they are built from the raw data by people who handled the decimals, the cross-chain normalization, and the pricing. Reaching for raw `ethereum.logs` when `tokens.transfers` answers your question is more work *and* more error-prone. Use the highest tier that answers the question.

**"Amounts are in dollars."** Decoded tables store token amounts in the token's smallest unit, and you must divide by `10^decimals` (six for USDC, eighteen for ETH and most ERC-20s) to get human units — and then join to `prices.usd` to get dollars. Forgetting the decimals gives you numbers off by a factor of a million or a billion; it is the second most common beginner error after the unquoted `"to"`.

**"An outflow from an exchange means coins are leaving the exchange."** Not always. Exchanges constantly shuffle funds *between their own wallets* — moving coins from a deposit address to a hot wallet to cold storage — and each of those internal moves looks like an outflow from one wallet and an inflow to another. If your wallet list is incomplete, an internal transfer registers as a real outflow when no coin actually left the exchange's control. The fix is to maintain the *full* set of an exchange's known wallets and exclude transfers where both the sender and recipient are inside that set. The lesson generalizes: an on-chain metric is only as good as the labels behind it, and label quality is where most "surprising" numbers turn out to be mistakes.

## The playbook: what to do with it

Here is the if-then checklist for turning Dune from a toy into an instrument. Each line is a signal, the read, the action, and the false-positive to guard against.

**Recipe lookup — match the question to the table.** Before writing SQL, name the metric and pick the tier.

![Matrix of common query recipes mapping metric to table, filter, aggregation, and result shape](/imgs/blogs/writing-onchain-queries-with-dune-6.png)

- **Signal: a token's net exchange flow turns sharply positive.** Read: coins are piling onto exchanges, supply-side pressure building. Action: tighten risk, watch for distribution. Invalidation: the inflow is one wallet repositioning, or your exchange-wallet list is wrong — check `count(distinct sender)` and the wallet labels.
- **Signal: net exchange flow turns sharply negative for days.** Read: coins leaving for self-custody, often accumulation. Action: a constructive backdrop for that asset. Invalidation: an exchange is just rotating reserves between its own wallets (an internal transfer that looks like an outflow) — exclude intra-exchange transfers.
- **Signal: DEX volume rises *and* `count(distinct taker)` rises with it.** Read: broad organic demand. Action: a real demand signal worth weighting. Invalidation: distinct buyers flat while volume spikes — likely wash trading, ignore the volume.
- **Signal: a stablecoin mint panel jumps tens of millions in a day.** Read: dry powder being staged. Action: watch for it to flow toward exchanges or DeFi next. Invalidation: a mint that is immediately offset by a redemption (net supply unchanged) — track *net* mints, not gross.
- **Signal: your chart suddenly looks gappy or flat-lines.** Read: probably indexing lag or a broken join, not a real on-chain change. Action: re-check the latest bucket (provisional), the price join (missing prices drop rows), and the decimals. Invalidation: it really did go quiet — confirm against a second table.

### The build checklist

When you build any new metric, run this loop:

1. **Name the metric and the shape** you want (one number? a daily line?).
2. **Pick the table** — spellbook first, decoded next, raw last.
3. **Filter on time and the key entity** (token, wallet) in `WHERE`.
4. **Aggregate** with `sum` / `count(distinct)` and `GROUP BY` the time bucket.
5. **Value in USD** via `amount_usd` if present, else a `prices.usd` join.
6. **Test on a short window**, sanity-check against a known number.
7. **Parameterize** the entity so the query is reusable.
8. **Chart and pin**, add a zero reference line where sign matters, publish.

### The gotcha guard

Finally, keep the three failure modes in view every time — they are the difference between a query that informs and one that quietly lies.

![Before and after of three query gotchas lag cost and wrong table with their fixes](/imgs/blogs/writing-onchain-queries-with-dune-7.png)

Lag means the freshest data is incomplete, so read settled days. Cost means an unfiltered query scans everything, so filter on time first and test small. The wrong table means you fought raw hex when a spellbook table had the answer, so search the spellbook before anything else. And remember the structural limit: **Dune shows you the chain's settled history, not its pending future.** That is enormous — it is more than the chain ever showed before tools like this — but it is not the mempool and it is not real time. Read it for what it is: the most powerful lens ever pointed at the public ledger, available free to anyone who learns to write the query.

It is worth naming why this skill compounds. Once you can write the net-flow query, the same five moves — pick a spellbook table, filter on time and an entity, aggregate with `sum` or `count(distinct)`, value in USD, bucket by day — produce nearly every other on-chain metric in this series. Active addresses is a `count(distinct "from")`. DEX volume is a `sum(amount_usd)` on `dex.trades`. Smart-money tracking is the top-N wallet query pointed at a watchlist. Stablecoin dry powder is a mint query. You are not learning seven hundred separate techniques; you are learning one query template and a handful of variations, and then *every* signal becomes reachable. That is the leverage of SQL over a public ledger: a small grammar applied to an enormous, honest dataset.

The first query you write yourself — the one where *your* filter returns *your* number off the live chain — is the moment on-chain analysis stops being something you read about and becomes something you do. From here, every signal in this series is a query you can build.

## Further reading & cross-links

- [The on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — where Dune fits among Etherscan, Nansen, Arkham, and the rest, and when to reach for each.
- [Exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — the trading meaning of the net-flow metric we built here.
- [Active addresses and network activity](/blog/trading/onchain/active-addresses-and-network-activity) — another count-distinct metric and what it really measures.
- [Analyzing DEX and AMM activity](/blog/trading/onchain/analyzing-dex-and-amm-activity) — going deeper on `dex.trades` and what swap data tells you.
- [Stablecoin flows: the dry-powder metric](/blog/trading/onchain/stablecoin-flows-the-dry-powder-metric) — the mint/redeem signal our dashboard panel tracked.
- [Detecting fake volume vs organic demand](/blog/trading/onchain/detecting-fake-volume-vs-organic-demand) — why `count(distinct)` is the antidote to a fake volume number.
- [Following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets) — pointing the same query toolkit at specific wallets.
- For the underlying mechanics, see [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money), [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao), and [stablecoins: Tether, Circle, and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar).
