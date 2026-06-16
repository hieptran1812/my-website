---
title: "On-Chain Alerts and Monitoring Bots: Building Your Own Real-Time Watchers"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A hands-on guide to building real-time on-chain watcher bots — subscribing to events, watching the mempool, polling balances, decoding logs, and pushing alerts to Telegram or Discord — plus the hosted shortcuts for non-coders."
tags: ["onchain", "crypto", "alerts", "monitoring-bot", "web3py", "ethersjs", "mempool", "telegram", "ethereum", "rpc", "websocket", "automation"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A monitoring bot is just a small program that listens to the blockchain and pings you the moment something you care about happens; this post wires one up end-to-end.
>
> - A watcher is a five-stage assembly line: a **provider** (RPC or websocket) feeds raw chain data in, a **filter** keeps only the events that matter, a **decoder** turns hex into dollars, a **channel** (Telegram/Discord/webhook) pushes it to you, and you **act**.
> - There are three ways to listen: **event subscription** (fast, cheap, for confirmed events), **polling** (a timer, for slow state with no event), and **mempool watching** (the earliest signal, but a firehose and expensive).
> - The hard part is not the happy path — it is **reliability**: reorgs, duplicate alerts, missed blocks, rate limits, and crashes each need a specific guard, or your bot quietly lies to you.
> - The rule of thumb: **buy** a hosted tracker (Cielo, Nansen, DeBank, Arkham) if your rule is standard and you do not code; **build** a web3.py/ethers.js bot when the logic is custom — and the core watcher fits in about 20 lines.

On 21 February 2025, an address controlled by the North Korean Lazarus Group drained roughly **\$1.46B** from a Bybit cold wallet — the largest crypto theft ever recorded. Within minutes, on-chain sleuths were posting the exploiter's address, the token breakdown, and the first hops of the laundering route. They were not psychic. They were running **watchers**: bots that had been subscribed to those wallets and contracts for months, primed to fire the instant an abnormal outflow appeared. The chain is public, so anyone with a listener saw the same blocks Bybit's own team saw.

That is the asymmetry this post is about. The information is free and arrives the same second for everyone. The edge — for a trader bracing for a sell-off, or a defender tracing stolen funds — goes to whoever has *already wired up the alert*. By the time a human refreshes Etherscan, the move is over. A watcher that was listening the whole time hands you the event while it still matters.

The [anomaly-detection companion post](/blog/trading/onchain/anomaly-detection-building-the-alerts) explained *what* to alert on and *why* — the concept of an on-chain alert as a signal. This is the build-it sibling: we actually wire up the watchers. We will go from "what is a bot?" to a working "large transfer to an exchange" alert in about twenty lines of code, then climb through mempool watching, decoding, the notify layer, and the unglamorous reliability work that separates a toy from something you can trust. Here is the whole machine on one page.

![Watcher bot architecture pipeline from provider to filter to decode to notify to act](/imgs/blogs/onchain-alerts-and-monitoring-bots-1.png)

## Foundations: what a watcher actually is

Strip away the jargon and a monitoring bot is embarrassingly simple: **a process that listens to the chain and notifies you when a rule matches.** It runs forever, watches a data source, checks each new piece of data against your rule, and on a match it sends a message. That is the whole idea. Everything else in this post is making that loop fast, accurate, and trustworthy.

Let me define every moving part from zero, because each one is a place beginners get stuck.

**The chain as a data source.** A blockchain like Ethereum produces a new **block** every ~12 seconds. Each block contains a list of **transactions** (transfers, swaps, contract calls), and each transaction that touches a contract can emit **event logs** — small structured records the contract writes to say "this happened." When you send USDC, the USDC contract emits a `Transfer` log saying *from X, to Y, amount Z*. Logs are the chain's built-in notification system, and they are what most watchers listen for.

**EOA vs contract.** An **EOA** (externally owned account) is a normal wallet controlled by a private key — a person or a bot. A **contract** is code deployed at an address that runs when called. You can watch both: an EOA's balance changing, or a contract emitting a specific event. We will use placeholder addresses like `0xA11ce…` throughout — never treat an invented address as a real one.

**The provider: RPC and websockets.** Your bot does not run its own copy of Ethereum (that is a heavy node sync). Instead it talks to a **provider** — a company that runs nodes and exposes them over the internet. You call it two ways:

- **RPC (HTTP):** request-response. You ask "what is the latest block?" or "give me the logs matching this filter," and it answers once. Good for polling and one-off queries.
- **WebSocket:** a persistent open connection. You **subscribe** once ("push me every new `Transfer` log for this contract") and the provider streams events to you as they happen. Good for real-time alerts.

The big providers are **Alchemy, Infura, and QuickNode** (paid, reliable, generous free tiers), plus **public RPCs** (free, rate-limited, flaky — fine for learning, risky in production). They sell you **credits** or **compute units**: every call costs a little, and websocket events count too. We will cost this out later.

It helps to picture the provider as a translator standing between you and a machine that speaks only a terse binary dialect. The blockchain itself is a peer-to-peer network of nodes gossiping blocks; to ask it anything you must run software that speaks the protocol, stay synced to the chain tip, and store the state. A provider runs that software at scale — hundreds of nodes, load-balanced, geographically spread — and rents you a clean HTTP/websocket endpoint. You get the chain's answers without the chain's operational burden. The price you pay for that convenience is twofold: you trust the provider to answer truthfully (a centralization most analysts accept, and a few mitigate by cross-checking two providers), and you live within their rate limits and credit budget. For a watcher, that trade is almost always worth it — your job is to *listen*, not to operate infrastructure.

There is also a free, self-hosted middle path worth naming: running your own node. If you sync an Ethereum execution client (Geth, Nethermind, Reth) plus a consensus client, you get an unlimited, private, rate-limit-free RPC/websocket endpoint on your own hardware — no per-event billing, no provider seeing your watchlist, and a full mempool view if you want it. The cost is a one-time ~1TB+ disk, an always-on machine, and the ongoing chore of keeping it synced and patched. Most individual watchers never need this; a provider's free tier covers a hobby bot for years. But once you are watching the mempool seriously, or you care that no third party sees which wallets you track, a home node stops being overkill and starts being the cheapest option per event. Knowing the option exists keeps you from over-paying a provider for volume a Raspberry-Pi-class node could serve for free.

**The library.** You do not hand-write JSON-RPC. In Python you use **web3.py**; in JavaScript/TypeScript you use **ethers.js** (or viem). They wrap the provider, handle the connection, and — crucially — **decode** raw data for you using an **ABI**.

**The ABI and decoding.** An event log on the wire is *hex*: a few `topics` (indexed fields, like the from/to addresses) and a blob of `data` (the amounts). It is unreadable. The **ABI** (Application Binary Interface) is a JSON description of a contract's functions and events — the schema. Feed the ABI and the raw log to your library and it hands back named fields: `from = 0xA11ce…`, `to = 0xBinance…`, `value = 5_000_000_000000` (USDC has 6 decimals, so that is 5,000,000 USDC). Decoding is the step that turns hex into a sentence a human can act on.

**The notify channel.** Once you have a human-readable event, you push it somewhere you will actually see it. The two staples are a **Telegram bot** (you message a bot token's API and it posts to your chat) and a **Discord webhook** (you POST JSON to a URL and it appears in a channel). Both are free, instant, and take five minutes to set up. A **webhook** in general is just "an HTTP URL you POST to, and something happens on the other end" — the universal glue of the notify layer.

That is the entire vocabulary. A watcher is: *provider → filter → decode → notify*. Now let us look at the three fundamentally different ways to do the "listen" part, because choosing wrong is the most common beginner mistake.

### Three ways to listen: subscribe, poll, watch the mempool

There are exactly three places a watcher can get its data, and they trade off latency, cost, and complexity in different directions.

![Matrix comparing event subscription polling and mempool watching across latency cost and best use](/imgs/blogs/onchain-alerts-and-monitoring-bots-2.png)

**Event subscription** is the default and the one you will use 90% of the time. You open a websocket, subscribe to a **log filter** (this contract, this event, optionally these indexed values), and the provider pushes you every matching log the instant it is mined into a block. It is **low-latency** (you hear about a confirmed event within a second of the block), **cheap** (one open socket, no wasted calls), and it covers everything that emits a log: transfers, swaps, mints, approvals, liquidations. If an event exists on-chain, subscribe to it.

**Polling** is asking on a timer. Some things do *not* emit a clean event you can subscribe to — a wallet's total ETH balance, a protocol's TVL, an account's health factor in a lending market. For those you call the chain on a schedule ("every 30 seconds, read this balance; alert if it crossed a threshold"). Polling **lags** by up to one interval and **burns credits** even when nothing changed, but it is the only option for state that has no event. Use it for slow-moving numbers, not for time-critical alerts.

**Mempool watching** is the exotic one: you subscribe to **pending** transactions — the ones broadcast to the network but *not yet mined into a block*. This lets you see a large swap or an exploit transaction *before it confirms*, buying seconds of lead time. The catch: the mempool is a **firehose** (thousands of pending txns), most never confirm (replaced, dropped, or front-run), and you generally need a premium provider to even get the stream. It is the highest-skill, highest-cost option, and we will give it its own section.

A good mental rule: **subscribe for confirmed events, poll for stateful numbers, watch the mempool only when the seconds before confirmation are the whole point.**

These three modes are not mutually exclusive — a serious watcher often runs all three at once for different jobs. A single bot might subscribe to a token's `Transfer` logs (confirmed events), poll a lending position's health factor every minute (stateful number with no clean event), and tap the mempool only for the one contract whose exploit you genuinely need to catch pre-confirmation. The skill is matching each *question* to its cheapest sufficient mode rather than forcing everything through one. Beginners tend to reach for polling because it is the easiest to reason about — a loop with a `sleep` — and end up with a sluggish, credit-hungry bot that learns about events a full interval late. The fix is almost always "find the event and subscribe to it." Most of what you want to know *does* emit a log; the chain was designed that way precisely so listeners could react.

The chain you are watching also shapes the mode. On Ethereum a 12-second block time makes polling tolerable for slow state and confirmations cheap. On a fast L2 or Solana, blocks come sub-second, so polling lags less but the event firehose is much heavier and reorg/confirmation semantics differ — you lean even harder on subscription and on the provider's indexing. On Bitcoin, with its 10-minute blocks and UTXO model, there are no smart-contract event logs at all; a "watcher" there means polling addresses or subscribing to a provider's address-activity stream, a different shape covered in [how-blockchains-store-data-utxo-vs-account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account). The architecture in this post is EVM-centric because that is where the richest event model lives, but the *provider → filter → decode → notify* skeleton transfers to any chain — only the listening mechanics underneath change.

## Event subscription: the workhorse watcher

Let us build the core listener. The pattern is identical across libraries: connect over websocket, create a filter, register a callback, loop forever. Here is the minimal web3.py version that watches a single ERC-20 token's `Transfer` events.

```python
from web3 import Web3

    # Connect over websocket so the provider can PUSH events to us.
    # wss:// is the secure websocket scheme; swap in your provider URL.
w3 = Web3(Web3.LegacyWebSocketProvider("wss://eth-mainnet.example/v2/KEY"))

    # The Transfer event signature is standard across all ERC-20 tokens.
    # web3.py needs only this slice of the ABI to decode the log.
TRANSFER_ABI = [{
    "anonymous": False,
    "name": "Transfer",
    "type": "event",
    "inputs": [
        {"indexed": True,  "name": "from",  "type": "address"},
        {"indexed": True,  "name": "to",    "type": "address"},
        {"indexed": False, "name": "value", "type": "uint256"},
    ],
}]

USDC = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
token = w3.eth.contract(address=USDC, abi=TRANSFER_ABI)

    # Create a filter: only Transfer logs from the USDC contract.
event_filter = token.events.Transfer.create_filter(from_block="latest")

while True:
        # Ask the filter for new matching logs since we last checked.
    for log in event_filter.get_new_entries():
        args = log["args"]              # decoded: from, to, value
        amount = args["value"] / 1e6    # USDC has 6 decimals
        print(f"{args['from']} -> {args['to']}: {amount:,.0f} USDC")
```

A few things to notice. We connected over `wss://` (websocket), not `https://`, so the provider can push. We supplied only the `Transfer` slice of the ABI — you do not need a contract's whole ABI to decode one event, just the event's definition, which is identical for every ERC-20. The filter narrows the firehose to one contract's transfers before anything reaches our code, so we are not decoding the whole chain. And `log["args"]` is already **decoded** — web3.py matched the log against the ABI and handed us named fields. The raw hex never touched our logic.

In ethers.js the same watcher is even terser, because `Contract` objects expose events as listeners:

```javascript
import { ethers } from "ethers";

// Persistent websocket connection to the provider.
const provider = new ethers.WebSocketProvider("wss://eth-mainnet.example/v2/KEY");

// Minimal ABI fragment: just the Transfer event.
const abi = ["event Transfer(address indexed from, address indexed to, uint256 value)"];
const USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48";
const token = new ethers.Contract(USDC, abi, provider);

// Register a callback; ethers decodes the log and calls us with named args.
token.on("Transfer", (from, to, value) => {
  const amount = Number(ethers.formatUnits(value, 6)); // 6 decimals
  console.log(`${from} -> ${to}: ${amount.toLocaleString()} USDC`);
});
```

Both versions are "tell me every transfer" — far too noisy to be useful. The art is the **filter**. You almost never want every event; you want every event *that matters*, and "matters" is a rule you bolt on. The two most useful narrowing dimensions are **which addresses** (watch only transfers *to* a set of exchange deposit wallets) and **what size** (only transfers above a dollar threshold).

You can push the address filter all the way down to the provider so it never even sends you irrelevant logs. Indexed event fields (`from` and `to` are indexed in `Transfer`) can be matched server-side:

```python
    # Only Transfers whose `to` is one of our watched exchange wallets.
    # Matching on the indexed field happens at the provider — far cheaper
    # than streaming every transfer and filtering in Python.
exchange_wallets = [
    Web3.to_checksum_address("0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8"),  # an exchange hot wallet (illustrative)
]
event_filter = token.events.Transfer.create_filter(
    from_block="latest",
    argument_filters={"to": exchange_wallets},
)
```

Now the provider only sends logs already matching your address set. The size filter (the dollar threshold) you apply in your own code after decoding, because amounts are not indexed and cannot be matched server-side. That two-layer approach — **narrow at the provider on indexed fields, then threshold in your code** — is the single most important efficiency pattern in this whole post. It is the difference between decoding a handful of relevant logs and choking on every transfer on Ethereum.

#### Worked example: an alert firing on a \$5M exchange inflow

A watcher subscribed to USDC `Transfer` logs with `to` filtered to a known exchange hot wallet sees a log: `from = 0xA11ce…`, `to = 0xExchange…`, `value = 5_200_000_000000`. Decode the value: USDC has 6 decimals, so that is `5,200,000` USDC. At a \$1.00 peg that is a **\$5.2M** inflow to the exchange. Your size filter is "alert above \$1M," so it clears, and the bot fires: *"0xA11ce… moved \$5.2M USDC to Exchange."* Why this matters: stablecoins generally flow *to* an exchange right before someone wants to buy, and a token flows to an exchange right before someone wants to sell. A single \$5.2M stablecoin inflow is one data point — but if you see five of them in an hour, totalling **\$5.2M + \$3.1M + \$8.4M + \$2.0M + \$6.3M = \$25M** of fresh buying power staged on one venue, that is a meaningful demand signal that hit your phone before it hit the price. The intuition: the alert converts a hex log into a dollar figure you can size a trade against, the same second the block confirmed.

The companion concept post, [anomaly-detection-building-the-alerts](/blog/trading/onchain/anomaly-detection-building-the-alerts), catalogs which events map to which threats — large CEX inflows, liquidity removals, mint calls, unlimited approvals. This post is where you turn each of those into a live `create_filter` call.

## Decoding: turning hex into a sentence

Decoding deserves its own pass, because it is where a beginner's bot most often goes silently wrong. A raw log has two parts: **topics** (an array of 32-byte hex words — `topics[0]` is the event signature hash, and the rest are the *indexed* parameters) and **data** (a hex blob holding the *non-indexed* parameters, packed end to end). Without the ABI it is meaningless. With the ABI, your library knows the event's shape and unpacks each field into a typed Python/JS value.

![Flow graph from raw log through ABI decode and enrichment to a formatted alert pushed to Telegram and Discord](/imgs/blogs/onchain-alerts-and-monitoring-bots-3.png)

Three decoding traps catch everyone at least once:

**Decimals.** On-chain amounts are integers in the token's smallest unit. USDC has 6 decimals, so `5_200_000_000000` is 5.2M USDC, not 5.2 trillion. ETH and most ERC-20s have 18. WBTC has 8. **Always divide by `10 ** decimals`**, and read the decimals from the contract (`token.decimals()`) rather than assuming — a bot that hard-codes 18 will report a USDC transfer as a comically large number and either spam you or, worse, look plausible while being a million times off.

**Address vs amount confusion in topics.** Indexed addresses arrive in `topics` left-padded to 32 bytes; your library strips the padding for you, but if you ever parse raw topics by hand, you must slice the last 20 bytes. Use the library's decoder; do not roll your own hex slicing for production alerts.

**Enrichment.** A decoded transfer says "5.2M USDC from A to B." To make it *actionable* you enrich it: multiply by a USD price (for a non-stable token you need a price feed — an oracle, a DEX quote, or a price API), and look up whether `to` is a labeled address (an exchange? a bridge? a known smart-money wallet?). The decode step gives you the *what*; enrichment gives you the *so what*. The flow graph above shows decode feeding two enrichment branches — price and labels — that join back at the threshold check before a formatted message goes out.

Here is decoding plus enrichment in one handler, the heart of a real alert:

```python
def handle_transfer(log, decimals, usd_price, labels):
        # log["args"] is already ABI-decoded into named fields.
    a = log["args"]
    amount = a["value"] / (10 ** decimals)       # raw int -> human units
    usd = amount * usd_price                       # enrich: dollars
    dest = labels.get(a["to"].lower(), "unknown")  # enrich: who is `to`?

        # Threshold AFTER decoding (amounts are not indexed / filterable).
    if usd < 1_000_000:
        return None

    return (f"{a['from'][:8]}... sent {amount:,.0f} tokens "
            f"(~{usd:,.0f} USD) to {dest}")
```

Notice the threshold lives *after* the decode, in dollars, because dollars are the unit you actually reason about. A "1,000 token" threshold is meaningless across tokens; a "\$1M" threshold is comparable everywhere. A token count is not a signal until you price it.

The price feed itself deserves a word of caution, because enrichment is only as trustworthy as the price you multiply by. There are three common sources, in descending order of safety. A **price oracle** like Chainlink publishes an aggregated, manipulation-resistant price on-chain — slowest to update but hardest to fool, and the right choice when a wrong price would trigger a wrong action. A **price API** (CoinGecko, a centralized exchange's REST endpoint) is convenient and broad but off-chain, so it can lag, rate-limit you, or simply be down at 3am. A **live DEX quote** — reading the spot price straight from the pool you are watching — is the freshest but the most dangerous, because the very swap your alert is reacting to may have *moved* that pool's price, and for thinly-traded tokens the pool price can be manipulated outright by the same actor you are trying to catch. A practical rule: use an oracle or API for the *enrichment* (turning a token amount into a rough dollar figure for the alert), and never let a single DEX-quoted price drive an automated action without a sanity bound. The [chainlink-and-blockchain-oracles](/blog/trading/crypto/chainlink-and-blockchain-oracles) post explains why on-chain price feeds are a hard problem and how aggregation defends against exactly the manipulation a naive DEX quote invites.

#### Worked example: a wallet-watch bot flagging a smart wallet's \$200k buy

You maintain a watchlist of "smart money" addresses — wallets with a documented record of buying early. You subscribe to `Transfer` logs with `to` filtered to those wallets (tokens flowing *in* means they are accumulating). A log fires: a watched wallet received `100,000,000` units of a token with 18 decimals — wait, the bot divides by `10 ** 18` and gets a tiny number, which is the trap; the real fields say the wallet *swapped* and the inbound token amount, priced at \$0.04, equals a **\$200,000** position. The bot enriches: `to` is labeled "SmartMoney-07," the token is two days old. Alert: *"SmartMoney-07 bought ~\$200,000 of NEWTOKEN."* You do not blindly copy it — survivorship bias means yesterday's genius is tomorrow's bagholder, a trap the [following-smart-money-wallets](/blog/trading/onchain/following-smart-money-wallets) post hammers — but a \$200k conviction buy by a wallet that nailed three of its last five calls is worth a look. The intuition: filtering on the *destination* address turns a generic transfer firehose into a private feed of one cohort's accumulation.

## Polling: when there is no event to subscribe to

Subscription is beautiful when an event exists. But plenty of the most important on-chain state emits *no event when it crosses a line you care about*. A lending position's **health factor** is computed from prices and balances; it does not fire a log the moment it dips below 1.05 (the danger zone before liquidation). A wallet's total **ETH balance** changes via transactions, but there is no "balance crossed \$1M" event. A protocol's **TVL** is a derived aggregate. For these you **poll**: read the number on a timer, compare to the last reading (and to your threshold), alert on a crossing.

```python
import time

THRESHOLD_USD = 1_000_000
last_alerted = False

while True:
        # Read state directly: a contract view call or a balance lookup.
    bal_wei = w3.eth.get_balance(WATCHED_WALLET)
    usd = (bal_wei / 1e18) * eth_price()          # enrich to dollars

        # Alert only on the CROSSING, not every loop, to avoid spam.
    if usd >= THRESHOLD_USD and not last_alerted:
        notify(f"{WATCHED_WALLET[:8]}... balance crossed ~{usd:,.0f} USD")
        last_alerted = True
    elif usd < THRESHOLD_USD:
        last_alerted = False

    time.sleep(30)        # poll cadence: tune for cost vs latency
```

Two design points. First, **alert on the crossing, not the state.** If you alert whenever `usd >= THRESHOLD`, you fire every 30 seconds forever once the line is crossed — instant mute-the-bot territory. The `last_alerted` flag (a one-bit memory of "did I already say this?") makes the alert edge-triggered. Second, **the poll cadence is a cost-vs-latency dial.** A 30-second loop means up to 30 seconds of lag and 2,880 reads a day; a 5-minute loop is 288 reads a day but you might learn about a near-liquidation five minutes late. For a health factor that can go from safe to liquidated in one block, polling is the wrong tool — you would subscribe to the price-update or the lending pool's events instead. Polling is for genuinely slow state.

The [analyzing-lending-and-liquidations](/blog/trading/onchain/analyzing-lending-and-liquidations) post goes deep on health factors and what a wave of near-liquidations signals; here the point is just that *some* watchers must poll because the chain offers no event to hook.

#### Worked example: the RPC cost of a chatty polling bot

Say you poll 40 wallets' balances every 15 seconds. That is `40 wallets x (60 / 15) per minute x 60 minutes x 24 hours = 230,400` reads a day, or about **6.9M reads a month**. If your provider charges roughly \$0.20 per million compute units and a balance read is ~1 unit, that polling alone costs about **\$1.38/mo** — trivial. But add a price lookup and a health-factor view call per wallet per loop and you triple the unit count: now you are at ~20M units/mo, about **\$4/mo**, and on a free tier capped at, say, 10M units/mo you would blow the cap halfway through the month and the bot would start getting 429-throttled — silently seeing nothing. The intuition: polling is cheap per call but the calls multiply by wallets × frequency, so the bill (and the rate-limit risk) is set by cadence, not by any single read — tighten the loop only where the latency is worth it.

## Mempool watching: seeing it before it confirms

This is the section people get excited about and then humbled by. Every transaction is first **broadcast** to the network and sits in the **mempool** — a holding area of pending, unconfirmed transactions — before a validator includes it in a block. If you subscribe to *pending* transactions instead of confirmed logs, you can see a large swap, a liquidation, or an exploit transaction *seconds before it lands*. That lead time is real, and for a handful of use cases it is decisive.

![Before and after panels showing a thirty million dollar swap seen in the mempool versus only after confirmation](/imgs/blogs/onchain-alerts-and-monitoring-bots-4.png)

The mechanic in web3.py is a different subscription:

```python
    # Subscribe to the stream of PENDING transaction hashes.
pending = w3.eth.filter("pending")

while True:
    for txhash in pending.get_new_entries():
            # A pending hash is just an id; fetch the full txn to inspect it.
        tx = w3.eth.get_transaction(txhash)
        if tx is None:           # may have been dropped/replaced already
            continue
            # Decode tx.input against the router ABI to read the swap intent.
        intent = decode_swap_input(tx)
        if intent and intent["usd"] > 10_000_000:
            notify(f"PENDING: ~{intent['usd']:,.0f} USD swap incoming")
```

Now meet the four walls, because the brief and the kit both insist you understand the limits, not just the thrill:

**It is a firehose.** Ethereum's public mempool holds thousands of pending transactions at any moment. You are fetching and decoding a huge volume to find the few that matter, which is why mempool watching is the heaviest of the three modes by far.

**Most pending txns never confirm as-seen.** They get replaced (a higher-fee version), dropped, or reordered. A pending swap you alerted on might get **front-run** by an MEV bot that saw the same mempool — the very dynamic the [mev-sandwiches-and-frontrunning](/blog/trading/onchain/mev-sandwiches-and-frontrunning) post dissects. So a mempool alert is *probabilistic*: "this is likely about to happen," not "this happened."

**You need a premium provider.** Many cheap and public RPCs do not expose a full pending-txn stream at all, or give a thin partial view. A real mempool feed (Alchemy's `alchemy_pendingTransactions`, a dedicated mempool service like BloXroute, or your own node) is a premium product, because the bandwidth is enormous.

**Private mempools hide the juicy ones.** Increasingly, large trades and MEV-sensitive transactions are submitted through *private* channels (Flashbots, MEV-Boost relays) that never hit the public mempool — precisely so they cannot be front-run. The biggest swaps you most want to see early are often the ones you cannot.

Given all that, mempool watching earns its place only when the seconds before confirmation are the entire value: detecting an exploit transaction as it is broadcast so a protocol team can try to pause, or bracing for a known large swap into a thin pool. For ordinary "who moved what" alerts, confirmed-event subscription is faster to build, cheaper to run, and more reliable. The [detecting-an-exploit-in-progress](/blog/trading/onchain/detecting-an-exploit-in-progress) post is the use case where mempool latency genuinely changes the outcome.

#### Worked example: a mempool watcher catching a \$30M swap before confirmation

Your bot subscribes to pending transactions and decodes any whose `to` is a known DEX router. A pending txn appears: a swap of stablecoins for a mid-cap token, sized at **\$30M**, routed through a pool that holds only ~\$12M of that token's liquidity. The bot decodes the input, prices it, and fires: *"PENDING: \$30M swap into a \$12M pool — expect a violent price move."* The transaction is still ~3 to 9 seconds from confirmation. A risk manager watching that token sees the alert, and rather than getting caught mid-position, pulls a resting order or hedges. Now compare the same event on a confirmed-only watcher: the alert arrives *after* the block lands, by which point the \$30M swap has already moved the price 15–25% and the lead time — the only thing mempool watching buys you — is gone. The intuition: a \$30M order into a \$12M pool is a guaranteed dislocation; mempool watching is the difference between trading *into* it and trading *after* it.

## The notify layer: getting it to your phone

A perfect alert that prints to a server log nobody reads is worthless. The notify layer is how the event reaches a human, and it is the easiest part to build — five minutes per channel.

**Telegram** is the on-chain analyst's default. You create a bot via @BotFather, get a **bot token**, find your **chat id**, and then sending a message is one HTTP POST:

```python
import requests

def notify_telegram(text):
        # BotFather gives you the token; chat_id is your channel/DM.
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": text})
```

**Discord** is even simpler because it has dedicated **webhooks** — you create a webhook URL on a channel (Server Settings → Integrations) and POST JSON to it; no bot, no token management:

```javascript
// A Discord webhook is just a URL you POST a JSON body to.
async function notifyDiscord(text) {
  await fetch(WEBHOOK_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: text }),
  });
}
```

The deeper idea is that `notify()` should be an **interface**, not a hard-coded Telegram call. Wrap your channels behind one function so the watcher logic does not care where the message goes — today Telegram, tomorrow also a Discord channel for the team and a webhook into your trading dashboard. A clean notify layer also lets you **route by severity**: a routine \$1M inflow goes to a noisy "flow" channel, while a \$50M exchange inflow or a detected exploit pings a dedicated "urgent" channel that you actually have notifications on for. Severity routing is what keeps a useful bot from becoming a muted one.

Two more notify-layer details that beginners learn the hard way. First, **format for glanceability**: an alert you read on a phone lock screen in two seconds beats a wall of hex. Lead with the dollar figure and the verb (*"\$5.2M USDC → Exchange A"*), put the addresses and a block-explorer link below, and use the channel's formatting (Telegram and Discord both support bold and links) so the number that matters is the first thing your eye lands on. An alert is a piece of UI, and the most common reason a technically-correct bot gets ignored is that its messages are unreadable. Second, **make the notify call non-blocking and failure-tolerant.** If your Telegram POST hangs because their API is slow, a naive bot blocks its whole watch loop waiting — and misses the next ten events. Fire the notification on a separate thread or queue, wrap it in a try/except so a failed send logs an error instead of crashing the watcher, and consider a tiny retry. The alerting channel is the one part of the system *outside* your control (it is someone else's API), so it is exactly where you must assume failure and contain it. A watcher that dies because Telegram had a bad minute is a watcher that was never reliable.

One safety note in keeping with this series' ethics framing: a notify layer is for *defensive* awareness — flagging risk, bracing for moves, tracing stolen funds. The same plumbing should never be pointed at, say, automatically front-running other users' pending transactions for profit; that crosses from monitoring into the predatory MEV behavior the [mev-sandwiches-and-frontrunning](/blog/trading/onchain/mev-sandwiches-and-frontrunning) post warns about. Build watchers to *understand* the chain, not to prey on the people transacting on it.

## The walkthrough: a large-transfer-to-exchange alert, end to end

Now we assemble everything into the canonical first bot: **alert me when more than \$1M of a token moves to an exchange.** This is the most useful single watcher a trader can run, because tokens flowing *to* an exchange precede selling, and it ties together every piece — subscribe, filter, decode, threshold, notify. Here it is in about twenty lines of web3.py:

```python
from web3 import Web3
import requests

w3 = Web3(Web3.LegacyWebSocketProvider("wss://eth-mainnet.example/v2/KEY"))

TOKEN = Web3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")  # USDC
EXCHANGES = {                                  # to-address -> label (illustrative)
    "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8": "Exchange A hot wallet",
}
ABI = [{"name": "Transfer", "type": "event", "anonymous": False, "inputs": [
    {"indexed": True, "name": "from", "type": "address"},
    {"indexed": True, "name": "to", "type": "address"},
    {"indexed": False, "name": "value", "type": "uint256"}]}]

token = w3.eth.contract(address=TOKEN, abi=ABI)
flt = token.events.Transfer.create_filter(            # narrow at the provider:
    from_block="latest",
    argument_filters={"to": [Web3.to_checksum_address(a) for a in EXCHANGES]})

def notify(text):
    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                  json={"chat_id": CHAT_ID, "text": text})

while True:
    for log in flt.get_new_entries():
        a = log["args"]
        usd = a["value"] / 1e6                          # USDC: 6 decimals, ~$1 peg
        if usd >= 1_000_000:                            # threshold in dollars
            dest = EXCHANGES[a["to"].lower()]
            notify(f"ALERT: {a['from'][:8]}... sent {usd:,.0f} USD to {dest}")
```

Walk it once more, because every line earns its place. We connect over **websocket** so events are pushed. We define the **token** (USDC) and a dict of **exchange wallets to labels** — using the dict both to filter at the provider *and* to label the alert. The **ABI** is just the `Transfer` slice. `create_filter` with `argument_filters={"to": [...]}` does the heavy lifting: the provider only sends us transfers already destined for an exchange, so our loop decodes a trickle, not the firehose. Inside the loop we **decode** (divide by `1e6`), **threshold** in dollars (`>= 1,000,000`), **label** the destination, and **notify** via Telegram. That is a genuinely useful production-shaped bot in twenty lines. The [exchange-flows-inflows-and-outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) post explains why this specific signal — flow *to* exchanges — is worth watching above almost any other.

But notice what this version does *not* handle: what if the websocket drops? What if the provider re-delivers a log after a reconnect? What if a block you saw gets reorged away? That gap between "works in a demo" and "works at 3am while you sleep" is the rest of the post.

## Reliability: the part that separates a toy from a tool

A demo bot has one job: print on the happy path. A production watcher has a harder job: **never lie to you.** It must not miss the event you built it for, must not spam you with duplicates, and must not alert on something that got rolled back. There are five failure modes, and each has a specific, standard guard.

![Matrix of five reliability failures reorg duplicate missed block rate limit and crash with guards](/imgs/blogs/onchain-alerts-and-monitoring-bots-5.png)

**Chain reorgs.** Ethereum's chain tip can be **reorged**: a block you saw gets orphaned and replaced, so an event "happened" and then un-happened. If you alerted the instant a transfer appeared and it gets reorged away, you sent a false alert. The guard is **confirmations**: for high-stakes alerts, wait N blocks (commonly 2–12 on Ethereum, more on faster chains) before firing, so the event is buried deep enough to be effectively final. The tradeoff is latency: confirmations cost you ~12 seconds each. Tune N to the stakes — a routine flow alert can fire at 1 confirmation; an alert that triggers an automated action should wait for finality.

**Duplicate alerts.** Reconnects, overlapping filters, and backfills all re-deliver logs. The guard is **deduplication**: every log has a unique `(transactionHash, logIndex)` pair; keep a **seen-set** of those you have already alerted on, check before sending, and you will never double-fire. A seen-set is the single highest-leverage reliability feature — it is what lets you reconnect freely without spamming.

**Missed blocks.** Websockets drop. While your socket was dead, blocks were mined, and any matching event in that gap is *gone* from a pure subscription — you cannot un-miss it. The guard is a **block cursor**: persist the last block number you fully processed; on reconnect, **backfill** the gap with a historical `get_logs(from_block=cursor, to_block=latest)` call before resuming the live subscription. Subscription gives you real-time; `getLogs` backfill gives you completeness; you need both.

**Rate limits.** Under load, providers throttle you with HTTP 429s, and a naive bot treats the failed call as "nothing happened" — it looks alive while seeing nothing, the worst failure because it is silent. The guards are **exponential backoff with retry** (on a 429, wait, then wait longer, then alert yourself if it persists), **batching** (combine many `getLogs` ranges into fewer calls), and a **dead-man's switch** (a heartbeat that pings you if the bot has not processed a block in X minutes, so silence itself becomes an alert).

**Crash and restart.** Processes die — out-of-memory, a deploy, a server reboot. If your cursor and seen-set live only in memory, a restart either re-alerts old events or silently skips the gap. The guard is **persistence**: write the cursor and seen-set to disk (or Redis/SQLite), and on boot resume from exactly where you left off. A watcher that cannot survive a restart is not a watcher; it is a script you have to babysit.

The composite pattern that handles all five looks like this:

```python
seen = load_seen_set()          # persisted (tx_hash, log_index) pairs
cursor = load_cursor()          # last fully-processed block number

def process_logs(logs):
    for log in logs:
        key = (log["transactionHash"].hex(), log["logIndex"])
        if key in seen:                       # dedup guard
            continue
        if confirmations(log) < MIN_CONF:     # reorg guard
            continue
        alert_from(log)
        seen.add(key)
    persist(seen, cursor)                     # survive a restart

def run():
    while True:
        try:
                # Backfill any gap since the cursor, THEN go live.
            backfill = w3.eth.get_logs({"fromBlock": cursor, "toBlock": "latest",
                                        "address": TOKEN})
            process_logs(backfill)
            cursor = w3.eth.block_number
            subscribe_and_stream(process_logs)   # live websocket loop
        except (ConnectionError, TimeoutError):
            backoff_sleep()                       # rate-limit / reconnect guard
```

This is the unglamorous code, and it is exactly the code that makes the difference. A trader who trusts a bot that has not been reorg-hardened will eventually trade on a false alert; a defender who trusts a bot without a dead-man's switch will eventually be blind during the one incident that mattered. The [combining on-chain with off-chain signals](/blog/trading/onchain/anomaly-detection-building-the-alerts) discipline — cross-checking a single alert against a second source before acting — is the human-side guard that sits on top of all five technical ones.

#### Worked example: how a missing dedup guard turns one event into a \$0 alert disaster

Your exchange-inflow bot has no seen-set. Overnight your websocket drops and reconnects four times. Each reconnect re-streams recent logs, so a single **\$5M** inflow that happened once gets re-delivered and re-alerted **five times**: *"\$5M to Exchange A"* at 02:14, 02:31, 02:47, 03:02, 03:20. You wake to five identical alarms summing to a phantom **\$25M** of inflow that was really just **\$5M** seen five times. You either panic-sell on a \$25M signal that does not exist, or — having learned the bot cries wolf — you mute it, and the next night you sleep through a real \$50M inflow. The fix costs nothing: a seen-set keyed on `(txHash, logIndex)` collapses all five deliveries to one alert. The intuition: a duplicate alert does not just annoy — it corrupts the *quantity* you are reasoning about and trains you to ignore the bot, so dedup is a correctness feature, not a politeness feature.

## Cost and scaling: what running this actually costs

A single watcher is nearly free; a fleet is where the bill shows up. Providers price in **credits / compute units**, and each operation has a unit cost: a websocket subscription holds an open connection (cheap to maintain, but events you receive count), `getLogs` backfills cost more for wider ranges, and mempool streams are the most expensive of all. Knowing the rough shape lets you build something you can afford.

The big levers:

- **Subscription beats polling for cost.** One open socket pushing only matching events is far cheaper than thousands of timer-driven reads, most of which return "nothing changed." Whenever an event exists, subscribe.
- **Filter at the provider, not in your code.** Every log the provider does *not* send you is decoding (and often credits) you do not pay for. Push address/topic filters server-side.
- **Backfill ranges cost by width.** A 10,000-block `getLogs` is much pricier than a 50-block one. Keep your cursor current so backfills are small.
- **Mempool is a different budget.** A full pending-txn stream can be 10–100× the data volume of a confirmed-event subscription, which is why it usually requires a premium plan.

#### Worked example: the RPC cost of 1M websocket events a month

Suppose your watcher cluster receives **1,000,000 websocket events** in a month (each matching log the provider pushes you). On a typical plan a pushed event costs on the order of a fraction of a compute unit; call it ~1 unit each for round numbers, so 1M events ≈ **1M units**. At roughly **\$0.20 per million units**, that streaming load costs about **\$0.20/mo** — effectively free. Now add the *backfills*: say 30 reconnects a month, each backfilling a 500-block range at ~75 units per range, that is `30 x 75 = 2,250` units — still pennies. Where it actually bites is if you switch that same 1M-event need to **polling**: to catch those events on a 12-second timer across the contracts involved you might make 5M+ reads/mo at ~1 unit each, **\$1+/mo** for *worse* latency. And a **mempool** stream feeding the same logic could pull 50M+ units/mo (~\$10+/mo) just in bandwidth. The intuition: at hobby scale every approach is cheap, but the *ratio* is brutal — event subscription delivers the best latency at the lowest cost, so reach for polling or mempool only when subscription genuinely cannot answer the question.

For most individuals, a single watcher on a provider's free tier and a \$5/mo cloud VM (or even a Raspberry Pi at home) is the entire infrastructure. You graduate to paid tiers only when you are watching dozens of contracts, backfilling aggressively after frequent reconnects, or tapping the mempool. The [the-onchain-tooling-landscape](/blog/trading/onchain/the-onchain-tooling-landscape) post maps where providers sit in the broader stack of nodes, indexers, and analytics platforms.

## The hosted shortcuts: alerts without writing code

Not everyone wants to run a server. The good news is that for the most common watchers — track a wallet, flag a big swap, watch a token — **hosted services do it for you in minutes**, with labeled data and multi-chain coverage you would spend weeks building yourself. The tradeoff is custom logic and control. Here is the comparison.

![Matrix comparing hosted alert services and a DIY bot across setup logic coverage cost and control](/imgs/blogs/onchain-alerts-and-monitoring-bots-6.png)

The notable hosted options, by job:

- **Cielo** (formerly Cielo Finance) is the go-to **wallet-tracker**: paste a list of addresses, and it pushes you their swaps and transfers to a Telegram bot in near-real-time across many chains. It is the fastest way for a non-coder to run a "smart money" feed — the exact use case the [following-smart-money-wallets](/blog/trading/onchain/following-smart-money-wallets) post is about, minus the code.
- **Nansen** offers **Smart Alerts** layered on its labeled-wallet database: alert when a labeled cohort (e.g., "Smart Money") buys a token, when a fund moves funds, or on custom wallet/token conditions. You are paying for the *labels*, which are the hard part to build yourself.
- **Arkham** provides entity-level **alerts** on its attribution graph — alert when a tracked *entity* (an exchange, a fund, a person Arkham has de-anonymized) transacts. Strong for investigation and following institutions.
- **DeBank** is portfolio-first but offers wallet **stream** following and webhooks, good for watching a handful of wallets' DeFi positions across chains.
- **DexScreener** (and similar) covers the **token / DEX** side: price, volume, liquidity alerts on new and trending pairs, useful for memecoin and new-listing watching.

When does hosted win? When your rule is *standard* (track these wallets, flag swaps over \$X, watch this token) and you would rather pay a subscription than maintain code. When does DIY win? When the rule is *custom* (a multi-step condition, an odd threshold, a bespoke decode no template offers), when you want full **control and privacy** (a hosted service sees your watchlist and whom you track), or when at scale a per-seat subscription costs more than your own RPC credits. Many serious analysts run *both*: hosted trackers for broad coverage and a couple of bespoke DIY watchers for the edge cases that matter most to them.

#### Worked example: hosted vs DIY as a dollars-and-hours decision

You want to track 50 smart-money wallets and get pinged on every buy. **Hosted (Cielo-style):** paste the 50 addresses, connect Telegram, done in 20 minutes; cost maybe **\$0 to \$30/mo** depending on tier and wallet count. **DIY:** write the subscription + decode + dedup + persistence + Telegram code, host it on a **\$5/mo** VM with maybe **\$2/mo** of RPC credits, and spend perhaps **8 hours** building and hardening it. If your time is worth even \$50/hr, that build is a **\$400** up-front cost versus the hosted service's \$30/mo — so for a *standard* wallet feed, hosted is the obvious buy: it would take **\$400 / \$30 ≈ 13 months** of subscription to match the build cost, and you would still be maintaining the code. DIY only pays off when you need a rule no hosted menu offers — at which point the \$400 buys you something money cannot rent. The intuition: build-vs-buy is a question of *standard vs custom*, not of *cheap vs expensive*; pay for time on the standard stuff and spend your hours only where the logic is genuinely yours.

![Decision flow graph for choosing a hosted tracker versus building a DIY watcher bot](/imgs/blogs/onchain-alerts-and-monitoring-bots-7.png)

The decision flow above is the one-glance version: start from the use case, branch on *standard vs custom rule*, and only the custom branch — if you are comfortable hosting code — leads to a DIY build. Everything else is a hosted tracker, and there is no shame in that; the goal is to *see the chain in time to act*, not to prove you can write a websocket loop.

## Common misconceptions

**"A watcher needs me to run a full Ethereum node."** No. A full node is a heavy, ~1TB+ sync you almost never need. A provider (Alchemy/Infura/QuickNode) runs the node and you talk to it over RPC/websocket. You run a 50-line script on a \$5 VM. People conflate "watching the chain" with "being the chain" — you only need the former.

**"Polling and subscribing are interchangeable."** They are not. Polling re-asks on a timer and lags by the interval; a subscription is pushed the instant a matching log is mined. For a time-critical alert (a drainer approval, an exploit), polling's lag can cost you the entire lead time. Subscribe when an event exists; poll only for stateful numbers that emit no event.

**"My demo bot is production-ready."** A bot that prints on the happy path will, left running, eventually fire a duplicate (no dedup), miss the one event that mattered (a dropped socket with no backfill), or alert on a reorged-away event (no confirmations). The 20-line version teaches the flow; the reliability section is what makes it trustworthy. Roughly, the happy path is 20% of the real code.

**"Mempool watching lets me front-run everyone."** Two problems. First, the biggest, most front-runnable trades increasingly go through *private* mempools (Flashbots) you cannot see. Second, the public mempool is already crawling with professional MEV bots with faster infrastructure than yours — you are not the only one looking, and you are probably the slowest. Mempool watching is genuinely useful for *bracing* and *exploit detection*, not for out-running searchers at their own game. And turning a watcher into a predatory front-runner is exactly the line this series tells you not to cross.

**"More alerts mean a better bot."** The opposite. An alert you receive 200 times a day is an alert you mute, and a muted bot is a blind bot. The skill is *narrowing* — tight filters, dollar thresholds, severity routing — so that when your phone buzzes, it is because something genuinely happened. A good watcher is mostly silent.

## The playbook: what to do with it

Here is the build-and-operate checklist, in the series' signal → read → action → invalidation shape.

**Pick the listen mode from the question.**
- *Signal:* "I need to know when X happens."
- *Read:* Does X emit an event? → **subscribe**. Is X a slow stateful number? → **poll**. Do I need it *before* it confirms? → **mempool** (premium, last resort).
- *Action:* Build the narrowest filter that answers the question; threshold in dollars, not tokens.
- *Invalidation:* If you are polling for something time-critical, you chose wrong — find the event to subscribe to instead.

**Filter at the provider, threshold in your code.**
- *Signal:* The firehose is too loud.
- *Read:* Narrow on indexed fields (address, topic) server-side; apply the dollar threshold after decoding.
- *Action:* Decode → enrich with price + labels → threshold in USD → notify.
- *Invalidation:* If your bot decodes thousands of irrelevant logs, your provider-side filter is too wide.

**Harden before you trust.**
- *Signal:* You are about to act on an alert.
- *Read:* Does the bot dedup `(txHash, logIndex)`? Wait for confirmations on high-stakes events? Backfill the cursor on reconnect? Back off on 429s? Persist across restarts? Have a dead-man's switch?
- *Action:* No → it is a demo; add the missing guard before you size a trade on it.
- *Invalidation:* A single false or duplicate alert that you acted on means a guard is missing — patch it before re-trusting the channel.

**Route by severity, stay mostly silent.**
- *Signal:* You are tuning the bot.
- *Read:* Are routine and urgent alerts in the same channel? Are you getting buzzed dozens of times a day?
- *Action:* Split channels by severity; tighten thresholds until the bot is quiet by default and loud only when it matters.
- *Invalidation:* If you have muted the bot, the thresholds are too loose — narrow them, do not turn it off.

**Build vs buy by standard-vs-custom.**
- *Signal:* You want an alert and you are deciding how.
- *Read:* Is the rule standard (wallet tracking, big swaps, token watch)? → **buy** (Cielo/Nansen/DeBank/Arkham/DexScreener). Is it custom, or do you need control/privacy/scale? → **build** (web3.py/ethers.js).
- *Action:* Pay for time on the standard stuff; spend your hours only on the logic that is genuinely yours.
- *Invalidation:* If you are spending weekends maintaining a bot that does what a \$30/mo service does, you should have bought.

**Cross-check before you act.**
- *Signal:* One alert fired.
- *Read:* A single watcher is one data point and on-chain data lies too — washed volume, bait wallets, mislabeled "smart money."
- *Action:* Confirm against a second source (a block explorer, a hosted tool, a different metric) before trading or raising an alarm.
- *Invalidation:* If you traded on one unconfirmed alert and got faded, the discipline — not the bot — failed.

A watcher is the cheapest edge in crypto: the data is free, the code is short, and the only scarce thing is having wired it up *before* the moment you needed it. Build the boring twenty-line version this week, harden one failure mode a day, and route it to a channel you will actually look at. The chain has been broadcasting the whole time. A watcher is just you, finally listening.

## Further reading & cross-links

- [Anomaly detection: building the alerts](/blog/trading/onchain/anomaly-detection-building-the-alerts) — the concept companion to this post: *what* to alert on and *why*, mapping each event to the threat it catches.
- [Following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets) — the wallet-tracking use case your watcher feeds, plus the survivorship-bias trap in "smart money" labels.
- [Detecting an exploit in progress](/blog/trading/onchain/detecting-an-exploit-in-progress) — where mempool latency genuinely changes the outcome, and how a watcher buys a protocol team seconds to react.
- [Exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — why "flow to an exchange" is the single most valuable thing the large-transfer watcher catches.
- [MEV, sandwiches, and front-running](/blog/trading/onchain/mev-sandwiches-and-frontrunning) — the predatory side of the mempool, and the line between a defensive watcher and a front-runner.
- [The on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — where providers, indexers, and analytics platforms sit in the broader stack your bot plugs into.
- [Ethereum and programmable money](/blog/trading/crypto/ethereum-and-programmable-money) — the contracts-and-events model your watcher listens to, from first principles.
- [Centralized crypto exchanges (Binance, Coinbase)](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) — what the exchange wallets your inflow alert watches actually are.

*This post is part of the **On-Chain Analysis** series. Related craft posts on writing chain queries, building dashboards, and fusing on-chain with off-chain signals extend the builder track once those siblings publish — until then, the [anomaly-detection post](/blog/trading/onchain/anomaly-detection-building-the-alerts) is the closest companion.*
