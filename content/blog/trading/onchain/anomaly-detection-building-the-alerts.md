---
title: "Anomaly Detection: Building the Alerts That Catch It in Real Time"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Turn the on-chain patterns you already know — rugs, drainers, sell-walls, insider moves — into automated alerts that fire the moment they happen, not in hindsight."
tags: ["onchain", "crypto", "anomaly-detection", "alerts", "monitoring", "rug-pull", "exchange-flows", "web3py", "dune", "ethereum", "defi-security"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — An on-chain alert is a rule that watches the public ledger and pings you the moment a specific event crosses a threshold, so you catch a rug, a drainer, or a sell-wall as it happens instead of reading about it after your money is gone.
>
> - An alert has four parts: **watch** (an address, contract, or event type) → **threshold** (how big before it matters) → **context filter** (cut the false positives) → **notify** (a channel you actually read).
> - The taxonomy maps one-to-one onto threats you already know: large CEX inflow = sell-watch, liquidity removal = rug-watch, mint event = dilution-watch, large approval = drainer-watch, dormant-wallet wake = insider-watch, volume spike = pump-watch, whale move = flow-watch.
> - A good rule cuts noise by **90%+**: don't watch "every Transfer," watch "Transfers over \$100k to a known exchange wallet, once per 30 minutes." The fix for alert fatigue is always *more context, not more alerts*.
> - The one rule to remember: **page yourself only when severity (the dollars at stake) AND confidence (the true-positive rate) are both high.** Everything else gets queued or ignored.

In the early hours of one trading day, a liquidity pool for a small token quietly drained. The deployer wallet called `removeLiquidity`, pulled roughly \$500,000 of paired ETH out of the automated market maker, and the token's price collapsed to near-zero within a single block. Holders woke up to a chart that looked like a cliff. The on-chain record of the rug was perfectly clear, public, and permanent — the `removeLiquidity` transaction sat right there in the block explorer for anyone to read. The problem was *timing*: by the time a human noticed the price drop and went to investigate, the deployer's ETH had already been swapped and was on its way to a mixer.

Here is the uncomfortable truth that every earlier post in this series has been circling: **reading the chain after the fact is forensics; reading it as it happens is defense.** The previous Track-E posts taught you to recognize the patterns — what a wash trade looks like, how a pump-and-dump unfolds, the signature of a rug pull, the shape of a sandwich attack, the fingerprint of an oracle manipulation. That knowledge is necessary but not sufficient. A pattern you can only recognize in a post-mortem protects no one. The same `removeLiquidity` call that took you twenty minutes to find in hindsight could have pinged your phone *thirty seconds after it landed* — early enough, in many cases, to sell into the last drops of liquidity, revoke an approval, or simply not be the holder who bought the dip on a dead token.

This post is about closing that gap. It is a build-and-usage guide: how to take the threats you already understand and wire them into **automated alerts** that watch the chain for you, around the clock, and notify you the instant a condition is met. We will build the mental model of what an alert *is*, lay out the full taxonomy of alert types mapped to the threats they catch, learn how to design a rule that doesn't drown you in false positives, survey the tooling spectrum from click-and-go hosted services to a fifty-line bot you run yourself, and walk through setting up three concrete alerts end to end. The figure below is the spine of the whole post — the five stages every alert passes through.

![Five-stage alert pipeline from chain event to action](/imgs/blogs/anomaly-detection-building-the-alerts-1.png)

## Foundations: what an alert actually is

Before any code or any tool, get the concept clean. An **alert** is a small, persistent rule that does three things forever: it *watches* some part of the blockchain, it *tests* each new thing it sees against a *condition*, and when the condition is true it *notifies* you. That is the entire idea. Everything else — the tooling, the thresholds, the dashboards — is machinery built around those three verbs.

It helps to name the everyday version first. A bank text that says "a \$2,000 charge was just made on your card" is an alert: the bank watches your account (the *watch*), tests each transaction against a rule you set like "over \$500" or "in a foreign country" (the *condition*), and texts you when it fires (the *notify*). An on-chain alert is exactly this, except the ledger it watches is public, so *anyone* can build the watcher — you don't need to be the bank. That is the superpower of a public ledger: the surveillance infrastructure that a bank keeps private is, on a blockchain, available to every participant. The flip side is that you have to build (or rent) the watcher yourself, because no one is watching *for* you by default.

Let me define the prerequisite terms from zero, because the rest of the post leans on them.

- **EOA (Externally Owned Account):** a normal wallet controlled by a private key — a person or a bot. It can send transactions.
- **Contract:** a program living at an address. It can't initiate transactions on its own; it runs when called. A token, a DEX pool, and a lending market are all contracts.
- **Transaction (tx):** a single signed action recorded on-chain — sending coins, calling a contract function. Permanent and public.
- **Event / log:** when a contract does something noteworthy, it *emits* a log — a structured record like `Transfer(from, to, amount)` or `Mint(to, amount)`. Logs are the backbone of alerts: they are the chain *announcing* what just happened, in a machine-readable form. A token transfer emits a `Transfer` log; adding liquidity emits a `Mint` log on the pool; approving a spender emits an `Approval` log.
- **Block:** a batch of transactions confirmed together, roughly every 12 seconds on Ethereum, faster on most L2s, ~10 minutes on Bitcoin. New blocks are the heartbeat that drives polling-style alerts.
- **Mempool:** the waiting room of transactions that have been broadcast but not yet included in a block. Watching the mempool lets you see things *before* they are final — the edge that MEV bots and the fastest defenders exploit.
- **LP (liquidity provider) / LP token:** in an automated market maker, liquidity providers deposit a pair of tokens into a pool and receive LP tokens representing their share. Removing liquidity burns LP tokens and returns the underlying — and is the exact action a rugging deployer takes.
- **CEX / DEX:** a centralized exchange (Binance, Coinbase — custodial, off-chain order book) versus a decentralized exchange (Uniswap, an on-chain AMM). Coins moving *to* a CEX are potential sell supply; this is the basis of exchange-flow alerts.
- **Approval:** an ERC-20 token requires you to `approve` a contract to spend your tokens before it can move them. A *malicious* unlimited approval to a drainer contract is how most wallet-drain scams actually steal funds — not by stealing your key, but by tricking you into approving them.

With those in hand, the anatomy of an alert becomes concrete. The **watch** is "which address, which contract, which event type do I subscribe to?" The **threshold** is "how big does the value have to be before I care?" The **context filter** is "what else has to be true for this to be a real signal and not noise?" And the **notify** is "where does the message land — Telegram, Discord, email, a dashboard tile?" Hold those four words; the entire taxonomy is just different settings of them.

### Event-driven vs polling: how an alert actually watches

There are two fundamentally different ways to watch the chain, and choosing the wrong one is the most common beginner mistake.

**Polling** means asking the chain on a timer: every 30 seconds, your script wakes up and asks an RPC node "have there been any new transfers to this address since I last checked?" It's simple to write and works against any data source — including aggregated metrics like exchange reserves or total value locked that only update slowly. Its weakness is latency and waste: between checks you are blind, and most checks return nothing, so you burn API calls finding out that nothing happened.

**Event-driven** (or *subscription*) means telling the node "push me a message the instant a log matching this filter appears." You subscribe once to, say, `Transfer` events on a specific token, and the node streams matches to you in real time. No timer, no wasted empty calls, and latency measured in the block time rather than your polling interval. This is the right architecture for anything time-critical — a rug, a live drainer, a sandwich setup — where seconds matter.

![Event-driven subscription versus timer-based polling architecture](/imgs/blogs/anomaly-detection-building-the-alerts-4.png)

The rule of thumb: **use event-driven subscriptions for anything where reaction speed is the point, and polling for slow aggregate metrics where a 5-minute lag is irrelevant.** A rug-watch must be event-driven — thirty seconds of lag is the difference between selling into liquidity and holding a dead token. An exchange-reserve trend, which updates meaningfully over days, can poll once an hour and lose nothing.

### Signal vs noise: what makes an alert *good*

An alert is only as useful as its **true-positive rate** — the fraction of firings that are actually worth your attention. This is the single most important idea in the whole post, so let me state it bluntly: *a noisy alert is worse than no alert*, because a channel that pings 400 times a day gets muted, and a muted channel misses the one firing that mattered. Every defender who has built alerts has lived this: the rug they missed wasn't because they had no alert, it was because the alert was buried in dust transfers.

A good rule has a high signal-to-noise ratio, and it gets there through three levers, which we'll dissect in detail later: a **threshold** (only fire on size that matters), an **entity context filter** (only fire when the counterparty is the kind that makes it meaningful), and a **cooldown** (don't fire ten times for one ongoing event). A bare event watch — "tell me about every `Transfer` of this token" — has a true-positive rate near zero, because the overwhelming majority of transfers are tiny, routine, or internal. Add a \$100k threshold and a "destination is a known exchange" filter, and the same watch fires a handful of times a day, nearly all of them genuinely worth a look. That is the craft of alerting: not catching *more*, but catching *only what matters*.

There's a quantitative way to think about this that's worth carrying around. Borrow two words from medical testing. **Sensitivity** is the fraction of real events your alert catches (you don't want to miss the rug). **Precision** is the fraction of firings that are real events (you don't want to be pinged by noise). The two trade off against each other: loosen the threshold and you catch more real events but also more noise; tighten it and you cut noise but risk missing a real-but-small event. There is no setting that maximizes both — you choose where to sit on the curve based on the *cost of a miss* versus the *cost of a false alarm*. For a drainer watch on your own \$40,000, a miss is catastrophic and a false alarm costs you ten seconds, so you tune for high sensitivity and tolerate noise. For a "whale is selling" watch that's merely informational, a miss costs you nothing irreversible, so you tune for high precision and accept that you'll miss the occasional smaller move. Naming which axis matters for a given alert is the first design decision, before you ever pick a number.

### Where alerts run: hosted, self-hosted, or hybrid

The last foundational question is *where the watcher physically lives*, because it shapes everything downstream — your latency, your costs, and how much you can customize. There are three homes.

A **hosted service** runs the watcher on someone else's infrastructure. You log into a website, click a wallet or contract, pick a rule, and the service watches and notifies for you. You never see a server. This is the path of least resistance and the right answer for most people most of the time — you trade control for speed-to-live and for the service's pre-built entity labels.

A **self-hosted bot** runs on a machine you control: a cheap cloud server, a Raspberry Pi at home, even a laptop that stays on. It connects to an RPC endpoint, runs your filter logic, and posts to your channel. You own the latency budget and the logic, and at high volume it's far cheaper than per-seat hosted pricing — but you have to build it, keep it running, and watch *it* for failures.

A **hybrid** uses a hosted RPC provider (Alchemy, Infura, QuickNode) as the data feed but runs your own logic on top. This is the sweet spot for most builders: you skip the genuine pain of running a full node — syncing, disk, uptime — while keeping full control of the rule logic and the notification path. The `web3.py` snippet later in this post is a hybrid: a hosted RPC endpoint feeds a watcher you wrote and run.

The choice is not permanent. A common path is to start hosted (validate that an alert is useful at all), then graduate the rules you care most about to a self-hosted or hybrid bot once you know exactly what you want them to do and need the latency or the cost savings.

## The alert taxonomy: seven watches mapped to seven threats

Here is the heart of the build guide. The earlier posts in this series taught patterns; each pattern has a corresponding alert, defined by which event it watches and which threat it catches. Internalize this table and you have the menu from which you'll assemble your monitoring stack.

![Seven alert types matched to the events they watch and threats they catch](/imgs/blogs/anomaly-detection-building-the-alerts-2.png)

Walk through them one by one.

**(1) Large exchange inflow — the sell-pressure watch.** You watch for a large `Transfer` whose destination is a known centralized-exchange deposit address. The threat: someone — often a whale or an early investor — is moving coins to a venue where they can be sold. Coins sitting in a self-custody wallet can't hit the order book; coins that just landed in a Binance hot wallet can be sold in one click. A sudden inflow is a leading indicator of potential supply hitting the market. The action: a sell-watch — tighten longs, don't add, watch for follow-through. We cover the mechanics of why deposits arm the order book in the [exchange-flows post](/blog/trading/onchain/exchange-flows-inflows-and-outflows).

**(2) Liquidity removal / LP unlock — the rug watch.** You watch a DEX pool for a `removeLiquidity` call or a `Burn` log on the LP token, especially from the deployer or a wallet holding a large LP-token share. The threat: a rug pull, where the team pulls the paired asset (the ETH or stablecoin side) out of the pool, leaving holders with a token they can't sell at any meaningful price. Of all the alerts here, this is the most time-critical — the value evaporates within a block or two. The action: exit immediately if you're in, and treat the token as dead. A related variant watches for an LP-lock *expiry*: if a project's liquidity was time-locked (a common "trust us" signal), the moment the lock unlocks is the moment a rug becomes possible.

**(3) Mint event / supply change — the dilution watch.** You watch the token contract for a `Mint` log or any call that increases `totalSupply`. The threat: hidden inflation or dilution — a contract owner minting new tokens (to dump, to fund themselves, or as a backdoor that the audit missed). A legitimate token with a fixed supply should *never* mint after launch; one that does has an owner power you need to understand. The action: re-check the contract's owner privileges and supply schedule; a surprise mint is often the prelude to a sell.

**(4) Large new approval — the drainer watch.** You watch *your own* wallet (or a wallet you're protecting) for `Approval` events, especially `approve` calls for an unlimited or very large amount to an unfamiliar contract. The threat: a wallet-drainer. Most wallet thefts don't steal your private key — they trick you into signing an `approve` that lets a malicious contract move your tokens whenever it likes. The approval is the loaded gun; the drain is pulling the trigger, which can come minutes or weeks later. The action: revoke the approval *immediately* (via a tool like Revoke.cash) before the drainer fires. This is the one alert on the list that protects you from your own clicks.

**(5) Dormant-wallet wake — the insider / old-coins watch.** You watch a set of long-dormant addresses (early investors, team wallets, ancient Bitcoin coins) for *any* outgoing transaction after a long sleep. The threat: an insider or early holder moving coins they've sat on for years — often the prelude to selling, or a sign that someone with information is repositioning. When a wallet that received tokens at the seed round and hasn't moved in two years suddenly sends them to an exchange, that's a story. The action: watch where the coins go; if it's to a CEX, fold it into your sell-watch.

**(6) Sudden volume / holder spike — the pump watch.** You watch a token for an abnormal jump in trading volume, transaction count, or holder count in a short window. The threat: an organic spike *or* a coordinated pump-and-dump where promoters inflate activity before dumping on the buyers they attracted. The action: don't chase the candle; check *who* is buying (a flood of fresh wallets funded from one source is a sybil/wash signature, covered in the wash-trading material) and whether the volume is real or washed.

**(7) Whale accumulation / distribution — the flow watch.** You watch a set of labeled large wallets ("smart money," known funds, market makers) for buys and sells. The threat — or opportunity — is that informed players are accumulating (bullish) or distributing (bearish) ahead of the crowd. The action: position *with* the flow when the wallet has a credible track record, but remember the survivorship-bias trap: "smart money" labels are assigned to winners after the fact, so a label is a hypothesis, not a guarantee.

Notice the structure: every alert is the same four-part machine (watch → threshold → context → notify) pointed at a different event. Once you can build one, you can build all seven by swapping the event filter and the threshold. That's why this post is a build guide and not seven separate recipes.

### The same taxonomy across chains

The taxonomy is chain-agnostic in spirit, but the *event you watch* differs by chain's data model, so a quick orientation. On **Ethereum and EVM L2s** (the spine of DeFi and tooling), everything above is a log: `Transfer`, `Approval`, `Mint`, `Burn` are standardized ERC-20/ERC-721 events you subscribe to directly. This is the easiest environment to alert on, which is why most examples here are EVM. On **Solana**, there are no Ethereum-style logs; you watch program instructions and account changes (a token balance jumping, a liquidity-pool account draining), and the memecoin-heavy ecosystem makes the pump-watch and rug-watch especially active — Solana launchpads spawn enormous volumes of tokens, the vast majority of which go to zero. On **Bitcoin**, the UTXO model means there are no smart-contract events at all; the alerts that apply are the dormant-wallet wake (old coins moving) and the large exchange inflow (a big UTXO consolidated and sent to an exchange cluster) — there's no mint, no approval, no liquidity to remove. On **Tron**, which carries an outsized share of retail USDT, the relevant alerts are large stablecoin transfers to exchanges and dormant-wallet moves. The four-part machine is identical everywhere; only the *event primitive* you subscribe to changes with the chain's data model, and **bridges** are the seam where a flow can hop chains and your single-chain alert goes blind — a reason cross-chain watchers exist.

### Why the approval watch deserves special attention

Of the seven, the large-approval (drainer) watch is the one most worth building first, because it's the only one that protects you from a loss you *cause yourself* by signing. The mechanic is worth spelling out because it's so widely misunderstood. An ERC-20 token tracks, for every (owner, spender) pair, an *allowance* — how much of the owner's balance the spender is permitted to move via `transferFrom`. A normal interaction (swapping on a DEX, depositing to a lending market) requires you to `approve` the protocol's contract for some amount first. Scam sites exploit this by getting you to sign an `approve` for an *unlimited* amount (`type(uint256).max`) to a contract they control. Crucially, the approval *moves no money* — your balance is untouched, your wallet still shows the tokens — so nothing looks wrong. The theft happens later, when the attacker's bot calls `transferFrom` and sweeps everything you approved, possibly hours or days after you forgot about the transaction. The approval is the loaded gun; the `transferFrom` is the trigger. An alert on the `Approval` event catches the gun being loaded, which is the only moment you can still act — once `transferFrom` fires, the money is gone. This is why the drainer watch is uniquely valuable: it's a window between setup and execution that no other alert offers.

## Designing a rule that doesn't drown you

The difference between a useful alerting setup and one you mute within a week is almost entirely about **rule design**. A beginner's instinct is to watch broadly — "alert me on everything this contract does" — and then add more alerts when the first ones don't catch what they wanted. That is exactly backwards. The fix for a missed signal is rarely *another* alert; it's a *better-tuned* version of the one you have.

![Naive alert versus a tuned rule with threshold, context, and cooldown](/imgs/blogs/anomaly-detection-building-the-alerts-3.png)

A good rule is built from three levers, layered on top of the raw event watch.

**Lever 1 — the threshold.** Most on-chain activity is small. Dust transfers, test transactions, routine micro-movements: they vastly outnumber the moves that matter. A size threshold is the cheapest, highest-leverage filter you can add. "Transfers to this exchange" might be 400 events a day; "transfers *over \$100,000* to this exchange" might be three. The threshold should be set in *dollars* (or the token's market value), not raw token count — a "1 million token" threshold is meaningless when the token trades at a fraction of a cent, and far too high when it trades at \$50. Always denominate the threshold in money.

**Lever 2 — entity context.** Size alone isn't enough, because a large transfer between two wallets owned by the same person (an exchange shuffling its own hot and cold wallets, a fund rebalancing internally) is not the signal you want. The context filter asks: *is the counterparty the kind that makes this meaningful?* For a sell-watch, the destination must be a genuine exchange *deposit* address — a cluster where users send coins to sell — not an internal CEX wallet reshuffle and not an ETF custodian move. This is where entity labels (from a tool like Nansen or Arkham, or your own address book) earn their keep: they turn "a large transfer" into "a large transfer *to Binance's user-deposit cluster*," which is a far stronger signal. The labeling and attribution problem is deep enough that the series devotes a whole post to it; the short version is that good context filtering depends on good labels.

**Lever 3 — the cooldown.** One real-world event often emits many on-chain events. A whale selling \$5M might split it across twenty transactions; a draining contract might fire repeatedly. Without a cooldown, your alert pings twenty times for one story, and the spam is indistinguishable from twenty separate stories. A cooldown — "fire at most once per address per 30 minutes" — collapses a burst into a single notification. It's the difference between "Whale X is selling" (one useful ping) and twenty identical messages that train you to ignore the channel.

Layer all three and a rule that would have fired 400 times a day fires three times, with a true-positive rate near 100%. That is a ~99% reduction in volume with *zero* loss of the signals that matter — because the 397 you dropped were noise by construction. The craft is choosing the threshold and context so that the things you drop are genuinely things you'd never have acted on.

#### Worked example: tuning a threshold to cut noise 90%

Say you set a raw alert on a popular token: "notify me on every `Transfer`." Over a day it fires 1,000 times. You inspect them: roughly 900 are under \$1,000 (dust, micro-trades, bot activity), 80 are between \$1,000 and \$100,000 (retail and small players), and 20 are over \$100,000 (the whales and institutions you actually care about). At 1,000 pings a day, you mute the channel by lunchtime and the system is worthless.

Now add a single threshold: fire only on transfers over \$100,000. The volume drops from 1,000 to 20 — a **98% reduction**. But suppose you find 20 is still a touch noisy because some are internal exchange shuffles. Add the context filter "destination is a user-deposit cluster," and you're down to maybe 8 genuinely actionable firings. From 1,000 → 8 is a **99.2% cut**, and you didn't lose a single signal you'd have acted on, because nothing under \$100,000 between non-exchange wallets was ever going to change your position. The lesson: **the threshold is where you buy back your attention.** A \$100k floor instead of a \$1k floor isn't 100× stricter for no reason — it's the line between the moves that move markets and the dust that doesn't.

### Setting the threshold from data, not from a guess

Where does "\$100,000" come from? The single biggest mistake in rule design is picking a threshold out of the air. The right number comes from the token's own history. The chain is a complete record, so you can *backtest* a threshold before you ever go live: pull the last 30 days of `Transfer` events to exchange addresses for your token, sort them by dollar value, and look at the distribution. You'll almost always see a long tail — thousands of tiny transfers, then a sharp drop-off to a few dozen large ones. The threshold belongs at the knee of that distribution: high enough to cut the tail of dust, low enough to keep the moves that historically preceded price action.

A practical procedure: set the threshold at roughly the 95th percentile of recent transfer sizes, then sanity-check it against the events you *wish* you'd caught. Take the three biggest dumps in the token's recent history; would your threshold have fired on them? If yes, and the day-to-day firing count is in single digits, you've found the number. If a real dump slipped under your floor, lower it until it catches — then re-check that you haven't reopened the noise. This is the same discipline as backtesting a trading rule: you're validating against history before risking it live, and you're explicitly checking both the false positives (too many firings) and the false negatives (the real event you'd have missed).

#### Worked example: backtesting a threshold against a real dump

You're protecting a position in a token and want a sell-watch. Pulling 30 days of exchange-bound transfers, you find the median is \$420, the 95th percentile is \$95,000, and the largest single transfer was a \$5,000,000 dump that crashed the price 18% the next day. You set the threshold at \$90,000 — just under the 95th percentile. Backtesting that floor over the 30 days: it would have fired 11 times total, including on the \$5,000,000 dump (the one you most needed to catch) and on two other \$200,000+ moves that also preceded down-days, while ignoring the ~2,000 sub-\$90,000 transfers entirely. Eleven firings in 30 days is roughly one every three days — a channel you'll actually read. The \$90,000 floor is *earned* from the data: it sits above the dust and below every move that historically mattered. Had you guessed \$1,000,000 instead, you'd have caught the \$5M dump but missed the two \$200k moves that also lost money; had you guessed \$5,000, you'd have been pinged hundreds of times and muted the channel before the dump ever came.

## The tooling spectrum: hosted vs build-your-own

You don't have to write a single line of code to run useful alerts — and you also might want to, for control and cost. The tooling lives on a spectrum, and the right choice depends on how fast you need to be live, how custom your logic is, and how much you'll run at scale.

![Hosted alert tools versus a build-your-own bot compared across six dimensions](/imgs/blogs/anomaly-detection-building-the-alerts-6.png)

### The hosted end: click, set, done

**Hosted tools** let you point at a wallet or contract, pick a rule from a menu, and choose a delivery channel — all in a web UI, live in minutes. Their killer feature is **pre-built entity labels**: the hard problem of knowing that `0x28C6…` is a Binance deposit address is already solved for you.

- **Nansen** and **Arkham** offer wallet/entity alerts on top of large labeled datasets — alert when a labeled "smart money" wallet buys, when a tagged fund moves, when a specific address transacts. Arkham's alert engine in particular lets you build conditions on labeled entities.
- **DeBank** and **Cielo** (and similar "wallet tracker" bots) specialize in following specific wallets and pushing their activity to Telegram or Discord in real time — ideal for the whale-watch and smart-money-watch alerts.
- **DEXScreener** and similar token-data sites offer price, volume, and liquidity alerts on individual pairs — good for the volume-spike (pump) watch and for watching a pool's liquidity.
- **Block-explorer alerts** (Etherscan's address watch, plus its API) can email you when a watched address transacts — the simplest possible alert, free, and a fine starting point.
- **Security-focused monitors** (the kind security firms run) watch for malicious approvals and known drainer signatures across many wallets — the productized version of the drainer watch.

The trade-offs: hosted tools are fast and label-rich, but you're limited to their rule menu, you don't control their latency, and at scale the per-seat or per-alert pricing adds up. For most traders and for fast due diligence, that's a fine trade — the speed and labels are worth more than the control.

### The build-your-own end: your node, your logic

When you need a condition the menu doesn't offer, lower latency, or volume that makes a subscription expensive, you build your own. The stack is four parts.

![DIY alert stack: RPC node, log filter, rule logic, webhook, channel](/imgs/blogs/anomaly-detection-building-the-alerts-5.png)

1. **A node / RPC endpoint** — your data feed. Either run your own node (maximum control and privacy) or use a hosted RPC provider like Alchemy or Infura (no infrastructure to manage). This is where logs come from.
2. **A log filter / watcher** — a script (commonly `web3.py` in Python or `ethers.js` in JavaScript) that subscribes to specific events on specific contracts, or polls for new ones.
3. **Rule logic** — code that decodes each matched log, applies your threshold and context check, and decides whether to fire.
4. **A webhook + channel** — when the rule passes, an HTTP POST sends the message to wherever you read alerts: a Telegram bot, a Discord channel, Slack, or your own dashboard.

Here is the core of an event-driven watcher in `web3.py`. It subscribes to `Transfer` events on a token and fires when one crosses a dollar threshold to a known exchange. Note the comments are indented (never in column 0), and the snippet is illustrative — the threat-detection framing is *defensive*: this is how you watch your own or a public address, not how you attack one.

```python
import json
import requests
from web3 import Web3

    # connect to an RPC endpoint (your node, or Alchemy/Infura)
w3 = Web3(Web3.WebsocketProvider("wss://your-rpc-endpoint"))

    # the ERC-20 Transfer event signature, and a token + an exchange-deposit address
TOKEN = Web3.to_checksum_address("0xTokenContractAddressHere")
EXCHANGE_DEPOSIT = Web3.to_checksum_address("0xExchangeDepositClusterHere")
TRANSFER_TOPIC = w3.keccak(text="Transfer(address,address,uint256)").hex()

    # rule parameters: only fire above a dollar threshold, dedupe with a cooldown
USD_THRESHOLD = 100_000
TOKEN_PRICE_USD = 2.50        # pull live from a price feed in production
DECIMALS = 18
last_fired = {}               # address -> timestamp, for the cooldown
COOLDOWN_SECONDS = 1800

def usd_value(raw_amount):
    tokens = raw_amount / (10 ** DECIMALS)
    return tokens * TOKEN_PRICE_USD

def notify(text):
    requests.post(
        "https://your-webhook-url",   # Telegram/Discord/Slack incoming webhook
        json={"text": text},
        timeout=5,
    )

    # subscribe to Transfer logs on the token, filtered to the exchange destination
event_filter = w3.eth.filter({
    "address": TOKEN,
    "topics": [TRANSFER_TOPIC, None, "0x" + EXCHANGE_DEPOSIT[2:].rjust(64, "0")],
})

while True:
    for log in event_filter.get_new_entries():
        raw = int(log["data"], 16)
        value = usd_value(raw)
        if value < USD_THRESHOLD:
            continue                  # threshold lever: skip dust
        sender = "0x" + log["topics"][1].hex()[26:]
        notify(f"Large CEX inflow: ${value:,.0f} from {sender[:10]}...")
```

A few things to read out of that snippet. The `topics` filter does the **watch** at the node level — the node only sends you `Transfer` logs *to the exchange deposit address*, so you're not decoding millions of irrelevant transfers client-side. The `USD_THRESHOLD` check is the **threshold lever**, denominated in dollars via a live price. The `last_fired`/`COOLDOWN_SECONDS` scaffolding (sketched, not fully wired) is the **cooldown lever**. And `notify()` is the **delivery** — a single HTTP POST to a webhook. That's the whole machine: subscribe, decode, filter, post.

For wallets where speed is everything (rug and drainer watches), the most aggressive version watches the **mempool** — pending transactions before they're mined — so you can react in the seconds *before* a `removeLiquidity` is even confirmed. That's the same architecture with a `pending` subscription instead of a confirmed-log filter; it's how the fastest defenders (and, in the offensive world, MEV bots) get their lead time.

### The query end: Dune-style scheduled alerts

Between hosted and fully-DIY sits a third option: a **scheduled query**. Tools like Dune let you write SQL against decoded on-chain data and run it on a schedule, alerting when the result crosses a condition. This is polling, not event-driven, so it's right for slower metrics — but it's enormously powerful for conditions that are awkward to express as a single event filter. A liquidity-removal watch, written as a query, looks like this:

```sql
-- conceptual: flag large liquidity removals on watched pools in the last hour
select
    block_time,
    pool,
    tx_hash,
    amount_usd
from dex.liquidity_removals
where block_time > now() - interval '1' hour
    and amount_usd > 100000           -- the dollar threshold lever
    and pool in (select pool from my_watchlist)   -- the context filter
order by amount_usd desc
```

You schedule that to run every few minutes; when it returns rows, the alert fires. The strength is expressiveness — you can join against your watchlist, against label tables, against historical baselines to define "abnormal" — at the cost of the polling lag. For a rug watch where seconds matter you'd prefer the event-driven bot; for "any pool on my watchlist had a large removal in the last hour," the scheduled query is clean and maintainable.

### Combining alerts into a dashboard

Individual alerts are points; a **dashboard** is the picture. As your setup grows past a handful of rules, you'll want a single surface — a Telegram channel, a Discord server with one channel per alert type, or a custom web page — where every firing lands, tagged by type and severity. The dashboard does three jobs: it deduplicates (one cooldown across the whole system, not per-rule), it ranks (severity-sorted, so the \$40k drainer floats above the \$1k volume blip), and it gives you history (you can see that this wallet has been distributing for three days, not just today). Hosted tools give you a basic version of this for free; a DIY stack routes all its webhooks into one channel and adds a severity tag in the message.

The choice of **notification channel** is itself a design decision, not an afterthought, because the channel sets the *interrupt level*. A rule that lands in the page-me quadrant should arrive somewhere loud and immediate — a phone push, a Telegram message with sound, a phone call for the truly critical drainer watch — because its whole value is the seconds of lead time, and a notification you read three hours later is worthless. A rule that merely logs should land somewhere quiet — an append-only Discord channel or a database row you review on your own schedule — so it doesn't compete for your attention with the firings that need it. A common, robust setup uses three channels: a silent `#log` for low-severity confirmed events, a normal `#watch` for things worth a daily review, and a noisy `#page` (with @mention and sound) reserved exclusively for the top-right quadrant. Routing each rule to the right channel *is* the prioritization made physical — the severity-confidence grid isn't an abstract idea, it's literally which channel the webhook posts to.

One more dashboard discipline that separates a setup you trust from one you abandon: **a feed of what your rules are seeing even when nothing fires.** A channel that's been silent for two days might mean a calm market — or a dead RPC connection, an expired API key, or a contract that migrated out from under your filter. You cannot tell silence-because-calm from silence-because-broken without a heartbeat. The cheapest version is a once-a-day "still alive, processed N events, 0 fired" summary; the moment that summary stops arriving, you know the monitor itself died, not the market.

## Alert fatigue and prioritization

Build enough alerts and you create a new problem: there are now too many of them. **Alert fatigue** is the failure mode where so many notifications fire that you stop reading any of them — and the one that mattered scrolls past unseen. Every defender hits this wall. The cure is not fewer alerts (you'd lose coverage); it's **prioritization** — deciding, automatically, which firings deserve to interrupt you and which just get logged.

The cleanest way to prioritize is along two axes: **severity** (how much money is at stake) and **confidence** (how likely the firing is a true positive, not noise). Plot every alert on that grid and the policy writes itself.

![Severity versus confidence grid showing act, queue, and ignore zones](/imgs/blogs/anomaly-detection-building-the-alerts-7.png)

- **High severity + high confidence → page me now.** A \$40,000 unlimited approval to a known-drainer-pattern contract, or a confirmed \$500k liquidity removal on a token you hold. These interrupt you immediately, by the loudest channel you have.
- **High severity + low confidence → queue and verify.** A large transfer to an address you *think* might be an exchange but isn't sure. Worth a look, but verify the entity before acting — don't trade on an unconfirmed label.
- **Low severity + high confidence → log, don't page.** A confirmed but small move from a watched wallet. Track it (it might be the start of a pattern) but don't interrupt your day.
- **Low severity + low confidence → ignore.** This is the noise to mute. If your alerts are mostly landing here, your thresholds are too loose — tighten them.

The single rule that makes this work: **page yourself only when both axes are high.** Severity alone (a big move you can't confirm) gets verified first; confidence alone (a certain but tiny move) gets logged. This is exactly why the threshold and context levers from earlier matter so much — they're what *raise* an alert's severity and confidence enough to clear the "page me" bar. Tuning a rule isn't just about cutting volume; it's about ensuring that when a rule does fire, it lands in the top-right quadrant where action is warranted.

Prioritization is also where you decide what to do when alerts *correlate*. Real events rarely fire one rule in isolation — a rug often shows up as a liquidity-removal alert *and* a large exchange inflow (the deployer cashing out) *and* a volume spike (panic selling) within the same few minutes. A naive system treats these as three separate firings and triples the noise; a smart one recognizes the cluster as *one story* and raises its confidence precisely because multiple independent rules agree. Correlated firings are the strongest signal you can get, because a single rule can be fooled but three rules firing on the same address in the same window is hard to fake. The practical implementation is a short correlation window — "if two or more rules fire on the same entity within five minutes, escalate the whole cluster to the page-me channel and tag it as a multi-signal event." This is the on-chain analog of how a security operations center treats correlated log entries: any one is interesting, but the correlation is the alarm.

A final, humbling point about prioritization: your thresholds and quadrant boundaries are not set-and-forget. As a token's price moves, a fixed dollar threshold drifts in token terms; as a market heats up, the baseline rate of large transfers rises and yesterday's "abnormal" becomes today's normal. The same \$100,000 floor that fired 8 times a day in a quiet market might fire 80 times a day in a frenzy — pushing perfectly good signals back down into the noise. Periodically re-backtest your thresholds against the recent distribution, the same way you set them in the first place. A monitoring setup is a garden, not a statue: it needs pruning as the conditions it watches evolve.

## How to set up three alerts end to end

Theory is cheap; let's wire three concrete alerts the way you actually would. Each follows the same four-part recipe — watch, threshold, context, notify — and each maps to a worked dollar example so you can see the money the alert is protecting.

### Alert A — large exchange inflow (sell-pressure watch)

**Watch:** `Transfer` events on the token you hold (or BTC/ETH movements), with destination filtered to a known exchange deposit cluster. **Threshold:** dollar value over a floor you set from the token's typical large-trade size — say \$100,000 for a mid-cap, higher for BTC. **Context:** the destination must be a *user-deposit* address (where coins go to be sold), not an internal exchange wallet shuffle and not an ETF custodian — this is where labels matter. **Notify:** a "sell-watch" channel.

**Tool choice:** for most people, a hosted wallet-tracker (Cielo, Nansen, Arkham) watching the relevant whale wallets is the fastest path; for a custom token or lower latency, the `web3.py` snippet above.

#### Worked example: a \$5M exchange inflow fires the sell-watch

A wallet holding 2,000,000 of a token trading at \$2.50 sends the whole position to a Binance deposit address. The alert decodes the `Transfer` log, prices it at 2,000,000 × \$2.50 = **\$5,000,000**, sees that \$5M is well over the \$100,000 threshold, confirms the destination is Binance's user-deposit cluster (context check passes), and pings the sell-watch channel: "Large CEX inflow: \$5,000,000 to Binance from 0xA11ce…". You now know, *minutes* after it landed and likely before any price impact, that \$5M of this token is sitting one click from the order book. If you're long, that's your cue to tighten stops or trim — \$5M of fresh sellable supply on a thinly traded token is the kind of overhang that caps a rally. The alert didn't tell you the price would fall; it told you the *supply to make it fall* just arrived.

### Alert B — liquidity removal (rug watch)

**Watch:** the DEX pool contract for your token — specifically `Burn` events on the LP token or `removeLiquidity` calls, weighted toward the deployer or large LP holders. **Threshold:** any removal over a fraction of the pool (e.g. >20% of pooled liquidity) or over a dollar floor. **Context:** is the remover the deployer or an insider? Removal *by the team* is the rug signature; removal by a random LP rebalancing is routine. **Notify:** a high-priority "rug" channel — this one pages you.

**Tool choice:** event-driven is mandatory here. A `web3.py`/`ethers.js` watcher on the pool, or a hosted security monitor that flags LP removals. DEXScreener can alert on a liquidity *drop* as a coarse backstop.

#### Worked example: an LP-removal alert catches a \$500k rug 30 seconds in

A token's pool holds \$500,000 of paired ETH. The deployer wallet calls `removeLiquidity`, burning its LP tokens and pulling the ETH out. Your event-driven watcher sees the `Burn` log *in the same block it's mined* — roughly **30 seconds** after the transaction broadcast — decodes that \$500,000 of liquidity just left, confirms the remover is the deployer address (context check: this is the rug signature, not a routine rebalance), and pages your rug channel. In the seconds that follow, the token's price collapses as the pool empties. If you held \$8,000 of the token, that 30-second head start might let you sell into the last \$50,000 of remaining liquidity and recover a meaningful fraction; without the alert, you'd find out when you next checked the chart and the token was already at zero. The alert converts a \$500k on-chain event into your personal 30-second escape window.

### Alert C — large approval (drainer watch)

**Watch:** `Approval` events emitted *by your own wallet* (the address you're protecting), filtered to large or unlimited approvals to contracts not on your allowlist. **Threshold:** unlimited approvals (`type(uint256).max`) or approvals exceeding a dollar value of your holdings, to an unfamiliar spender. **Context:** is the spender a known-good contract (Uniswap router, a reputable protocol) or an unrecognized address — possibly a drainer? **Notify:** an immediate, high-priority channel; this is the one alert that protects you from your own click.

**Tool choice:** a security-focused monitor that knows drainer signatures, plus a simple `Approval` watch on your own address, plus Revoke.cash bookmarked for the one-click revoke.

#### Worked example: an approval alert flags an unlimited approval risking \$40k

You sign what you think is a routine transaction on an unfamiliar site. It emits an `Approval` giving an unknown contract an *unlimited* allowance over a token you hold \$40,000 of. Your approval-watch fires within the block: "Unlimited approval to 0xDr4in… for TOKEN — \$40,000 at risk." Because nothing has *moved* yet — the approval is the loaded gun, not the trigger — you have a window, often minutes to hours, before the drainer's bot sweeps the funds. You open Revoke.cash, revoke the approval, and the \$40,000 is safe. The drainer never gets to call `transferFrom`. This is the highest-value alert on the list precisely because it catches the threat *before* a single dollar leaves your wallet — the approval is the early-warning that the theft has been *set up* but not yet executed.

Three alerts, three threats, one recipe. Notice that A and C are about *your own* exposure (supply against your position, an approval against your wallet) while B is about a token's *health* — but all three are the same machine pointed at different events, with the threshold and context tuned to the threat.

## Common misconceptions

**"More alerts means better coverage."** No — more *noise* means worse coverage, because the alert that matters drowns in the ones that don't. The team that misses the rug usually has *too many* alerts, not too few. Coverage comes from well-tuned rules in the right quadrant of the severity-confidence grid, not from volume. If you're adding a third redundant alert because the first two are noisy, fix the first two.

**"On-chain alerts are real-time and instant."** Only event-driven subscriptions approach real-time, and even those are bounded by block time — roughly 12 seconds on Ethereum, faster on L2s, ~10 minutes on Bitcoin. A polling alert can lag by its whole interval. And confirmation isn't instant either: a transaction in the mempool can be dropped or reordered. If you need to beat the block, you watch the mempool — and even then you're racing MEV bots. "Real-time" on-chain means "within a block or two," not "the same millisecond."

**"A green dashboard number means the signal is real."** This is the series' central warning, and it applies doubly to alerts. A volume spike can be washed; a "smart money" label is a survivorship-biased hypothesis; a transfer to an "exchange" address might be an internal reshuffle. An alert tells you an *event* happened; it cannot tell you the event *means* what you assume. That's what the context filter is for — and why high-severity, low-confidence firings get *verified*, not acted on.

**"If I build it once, it runs forever."** Contracts get upgraded, exchanges rotate deposit addresses, tokens migrate, RPC endpoints rate-limit you, and price feeds drift. An alert is a living thing: a stale label turns a good rule into a blind spot, and a dead RPC connection turns your whole stack silent without telling you. Monitor your monitors — a heartbeat alert that pings you if the bot *stops* reporting is the alert that saves all the others.

**"Watching the chain lets me front-run anything."** The chain's lead time is real but bounded and contested. By the time a `Transfer` to an exchange is confirmed and your alert fires, faster bots have seen the same mempool transaction. The edge from alerts is mostly *defensive* (don't be the one holding the rug) and *positional* (size and timing around flow), not a license to front-run. Treat the lead time as risk management, not a printing press.

## The playbook: what to do with it

Here is the if-then checklist that turns this whole post into a routine. For each alert: the signal it fires on → the read → the action → the false-positive to rule out.

- **Large CEX inflow fires** → *read:* potential sell supply just armed the order book → *action:* sell-watch — tighten longs, don't add, watch for follow-through selling → *false positive:* internal exchange wallet reshuffle or ETF custody move; confirm it's a user-deposit cluster before reacting.
- **Liquidity removal fires** → *read:* a rug may be in progress → *action:* exit immediately if you hold; treat the token as dead → *false positive:* a routine LP rebalance by a non-team wallet, or a migration to a new pool; confirm the remover is the deployer/insider.
- **Mint event fires** → *read:* supply just expanded — dilution or a hidden owner power → *action:* re-read the contract's owner privileges and supply schedule; reduce exposure if the mint is unexplained → *false positive:* a documented, scheduled emission (staking rewards, vesting) that you already knew about.
- **Large approval fires (your wallet)** → *read:* a drainer may be set up → *action:* revoke the approval *now* via Revoke.cash, before funds move → *false positive:* an approval you just made intentionally to a reputable protocol; confirm the spender against your allowlist.
- **Dormant-wallet wake fires** → *read:* an insider or early holder is repositioning → *action:* watch where the coins go; if to a CEX, fold into the sell-watch → *false positive:* a custody migration or a wallet consolidation with no sell intent.
- **Volume/holder spike fires** → *read:* organic interest or a coordinated pump → *action:* don't chase; check whether the buyers are fresh sybil wallets and whether the volume is washed → *false positive:* a genuine catalyst (listing, partnership) driving real new holders.
- **Whale move fires** → *read:* informed flow, accumulation or distribution → *action:* consider positioning *with* a credible wallet; size for the survivorship-bias risk → *false positive:* a "smart money" label that's just a lucky past winner; the label is a hypothesis, not a guarantee.

And three meta-rules that sit above the table:

1. **Denominate every threshold in dollars, not tokens.** A \$100k floor is meaningful; a "1M token" floor is meaningless across price regimes.
2. **Tune for the top-right quadrant.** A rule that doesn't fire in the high-severity, high-confidence corner is either too loose (noise) or too tight (silent). Adjust threshold and context until firings land where action is warranted.
3. **Monitor your monitors.** A heartbeat alert that catches a dead bot or a stale label is the one alert that protects all the others. Silence is not safety.

The arc of this series has been: the ledger is public, so learn to read it. This post is the operational endpoint of that arc. You no longer have to *be reading* the chain at the moment something happens — you build the rules once, and they read it for you, around the clock, and tap you on the shoulder only when something crosses a line you've drawn in dollars and context. That's the difference between forensics and defense, between the post-mortem and the save.

## Further reading & cross-links

- [Exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — the mechanics behind the sell-pressure alert: why deposits arm the order book and withdrawals disarm it.
- [The on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — the full survey of explorers, dashboards, and analytics platforms the hosted alerts sit on top of.
- [What is on-chain analysis](/blog/trading/onchain/what-is-onchain-analysis) — the foundation: why the public ledger gives you a readable, alertable edge.
- [How to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow) — when an alert fires, this is how you follow the money it points at.
- [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) — how AMM pools and liquidity work, the substrate of the rug-watch alert.
- [Centralized crypto exchanges: Binance, Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) — what an exchange deposit address is and why coins landing there matter.
- [Crypto mining, staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev) — the mempool and MEV context behind why "real-time" on-chain has a hard floor.
