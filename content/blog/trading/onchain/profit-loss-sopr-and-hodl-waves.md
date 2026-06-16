---
title: "SOPR and HODL Waves: Are Holders Selling at a Profit or a Loss?"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Because every coin's last-moved price is written on-chain, you can tell — in aggregate — whether the coins moving today are selling at a profit or a loss, and how old they are. SOPR, realized profit/loss, coin-days-destroyed and HODL waves turn that into a readable map of market psychology."
tags: ["onchain", "crypto", "sopr", "hodl-waves", "coin-days-destroyed", "realized-profit", "bitcoin", "glassnode", "market-psychology", "cost-basis", "capitulation"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Every coin carries its last-moved price on-chain, so the ledger tells you whether the coins being *sold today* are moving at a profit or a loss, and how *old* they are. That is enough to read market psychology directly from the chain.
>
> - **What it is:** **SOPR** (Spent Output Profit Ratio) = value sold ÷ value originally paid. Above 1 = aggregate profit, below 1 = aggregate loss, exactly 1 = break-even. **Coin-days-destroyed** (CDD) and **HODL waves** tell you whether the coins moving are young weak hands or old strong hands waking up.
> - **How to read it:** open Glassnode (or a free Dune board). Watch where SOPR sits relative to the **1.0 line** — the psychological pivot. Split it into short-term-holder and long-term-holder cohorts. Watch CDD for old-coin spikes and the HODL-waves "rainbow" for the accumulation/distribution shift.
> - **What you do with it:** SOPR persistently below 1 = capitulation, a place to accumulate; SOPR far above 1 with a CDD spike from old coins = distribution near a top, a place to trim. But verify — an exchange or custody migration can fake an old-coin spike with no real sale behind it.
> - **The number to remember:** a coin bought at \$20,000 and sold at \$60,000 is a SOPR of 3.0 and **\$40,000 of realized profit per coin** — money that just left "paper gains" and became real, recorded forever on the ledger.

On a quiet day in mid-2020, a Bitcoin wallet that had not touched its coins since 2011 suddenly moved roughly **50 BTC**. The coins were nine years old. At the price of the day, that stack was worth a few hundred thousand dollars, but the cost basis — whatever those coins were worth in 2011 — was on the order of a few hundred *dollars* total. The internet lit up. Was Satoshi moving? (Almost certainly not.) Was an early miner finally cashing out at the top of a decade-long hold? The price had not moved. There was no announcement. But on-chain, the event was unmistakable, because the chain records one thing that no price chart can: **how long each coin had been sitting still, and what it last moved at.**

That single move is a microcosm of this entire post. Price tells you what a coin is worth *now*. The chain tells you what each coin was worth when it *last moved* — its cost basis — and how long ago that was. Put those two facts together across every coin that changes hands on a given day and you can answer a question that no order-book, no candlestick, no sentiment survey can answer directly: **are the people selling today doing it at a profit or at a loss, and are they recent buyers or seasoned holders?** That answer is market psychology, read straight off the ledger.

This is one of the most powerful and most *misread* corners of on-chain analysis. Done well, it lets you see capitulation while the crowd is still calling for lower prices, and lets you see distribution while the crowd is still euphoric. Done carelessly, it turns a routine exchange wallet reshuffle into a fake "whale dumping" headline. We will build every concept from zero — cost basis, spent outputs, SOPR, realized profit and loss, coin age, dormancy, coin-days-destroyed, and HODL waves — then walk through reading them on real tools, and finish with an honest list of the ways they lie.

![SOPR mental model: a coin moving today was bought higher meaning loss or lower meaning profit, around the 1.0 pivot line](/imgs/blogs/profit-loss-sopr-and-hodl-waves-1.png)

Three ideas carry through everything below, so we name them now. **(1)** A coin's cost basis is fixed at its *last move* and lives on-chain forever. **(2)** When a coin moves, the gap between its cost basis and the current price is *realized* — turned from paper into fact — and SOPR is just that gap expressed as a ratio. **(3)** The *age* of the coins moving matters as much as the profit on them: young coins moving is the crowd reacting; old coins moving is conviction breaking, and that is the rarer, louder signal.

## Foundations: cost basis, spent outputs, and what "realized" means

Before SOPR, before any ratio, you need a crisp picture of three plain ideas: where a coin's "cost basis" comes from, what it means for a coin to "move," and the difference between a gain that exists only on paper and one that has been *realized*. No prior crypto knowledge is assumed; every term is defined the first time it appears.

### A coin's cost basis is its last-moved price

Your **cost basis** in any asset is simply what you paid for it. If you buy a share at \$100, your cost basis is \$100; if it rises to \$150, you have a \$50 *unrealized* gain — real on paper, but you have not sold, so you have not locked anything in. Crypto is the same, with one beautiful twist: the blockchain *records* the cost basis for you, in public, because of how coins move.

On Bitcoin, coins live in **UTXOs** — *unspent transaction outputs*. Picture each UTXO as a labeled bill in your wallet: "0.5 BTC, created in block X, on date Y." When that block was mined, BTC had some price — call it the **creation price** of that output. The chain does not literally store a dollar figure, but because every output has a timestamp, an analyst can look up the price on that date and assign the output a cost basis: *the market value of the coins at the moment they last moved.* (On account-based chains like Ethereum the bookkeeping differs, but the principle — value at last movement — still applies, which is why SOPR-style metrics are cleanest on Bitcoin's UTXO model. We will come back to why BTC is the home turf for these metrics.)

So the chain hands you, for free, something a stock investor would kill for: a public record of *what every unit of supply last changed hands at.* That is the raw material for everything in this post.

### "Spending" an output is the moment of truth

When you send Bitcoin, you **spend** one or more UTXOs — you consume those labeled bills and create new ones for the recipient (and change back to yourself). The output you consumed is now a **spent output**. This is the single moment that matters for our metrics, because spending an output is the only time its cost basis gets *compared* to a current price.

Here is the mechanic in one sentence: **when an output is spent, you compare the price the day it was created (its cost basis) to the price the day it is spent (its realized value), and the difference is the realized profit or loss on those coins.** An output created at \$20,000 and spent at \$60,000 realizes a \$40,000-per-coin gain. An output created at \$60,000 and spent at \$51,000 realizes a \$9,000-per-coin loss. Nothing is realized while the output just sits there; the act of *spending* is what turns paper into fact.

A crucial honesty note up front: "spending" an output is not always "selling." You spend an output when you send to an exchange to sell, yes — but also when you move coins between your own wallets, consolidate dust, or migrate to a new custody setup. The chain sees a spend; it does *not* see your intent. Holding that distinction firmly is what separates a careful analyst from a headline-chaser, and we will hammer it repeatedly.

There is a second subtlety worth naming because it trips people up. When you spend an output and get *change* back, that change is a brand-new output with a brand-new cost basis: its creation timestamp is *today*, not the day you originally bought. So a single self-send can quietly "reset the clock" on a chunk of your coins, making old supply look young in the age bands even though the same person still controls it. Good metric providers correct for this (they try to recognize change outputs and self-transfers and not treat them as fresh purchases), but it is a reminder that none of these numbers are handed to you clean — they are *estimates* built on heuristics about what a spend meant. Treat every SOPR, CDD, or HODL-waves figure as a well-reasoned approximation, never as gospel precision.

### Why the chain can compute all this and your bank cannot

Step back and appreciate the strangeness of what we just described. No traditional market can produce a SOPR. Your stockbroker knows *your* cost basis, and the exchange knows the order book, but nobody on earth has a public, per-unit record of what *every share of Apple* last traded at and how long each has been held in which account. That information is private, fragmented across custodians, and never aggregated. On Bitcoin it is the *default*: every output, public, timestamped, with an inferable cost basis, queryable by anyone. The entire toolkit of this post exists *only* because the ledger is public and permanent — the precondition the whole on-chain field is built on, and the property the early cypherpunks fought to make real. The metrics are downstream of that one radical design choice.

#### Worked example: an output's full lifecycle

Follow a single 0.5 BTC output. It is **created on a day BTC trades at \$30,000**, so its cost basis is 0.5 × \$30,000 = **\$15,000**. It sits unspent for 400 days, banking 0.5 × 400 = **200 coin-days** and carrying only unrealized P/L. BTC rises to \$50,000; the output now has an unrealized gain of 0.5 × (\$50,000 − \$30,000) = **\$10,000**, but nothing is locked in. Then the holder spends it at \$50,000: the chain records SOPR = 50,000 ÷ 30,000 = **1.67**, a **realized profit of \$10,000**, and destroys the **200 coin-days** it had accumulated. The intuition: one output silently carries four readable facts at once — a cost basis, an age, an unrealized P/L while it waits, and a realized P/L plus a CDD contribution the instant it moves.

### Realized vs unrealized: paper gains become real money

Lay the two side by side. **Unrealized profit/loss** is the gap between current price and cost basis on coins that are *just sitting there* — it is the market's mood, the paper wealth, the number that makes people feel rich or poor but has not changed hands. **Realized profit/loss** is that same gap, but *only measured on coins that actually moved* — it is real money that just got locked in, recorded on-chain.

The whole field of profit/loss on-chain metrics is the study of *realized* flows, because those are the actions, not the feelings. When a market is full of unrealized profit (everyone is up on paper) but realized profit is low (nobody is actually selling), you have a coiled spring. When realized profit suddenly floods out — lots of coins moving at big gains — you are watching people convert paper wealth into cash, which is exactly what tops are made of.

#### Worked example: paper gain vs realized gain

Say you bought **1 BTC at \$20,000** in 2020. By a 2021 peak it is worth \$60,000 — you have a **\$40,000 unrealized gain**, real on paper, locked to nothing. You feel rich; your net worth on a spreadsheet is up \$40,000. Now you actually send that coin to an exchange and sell it at \$60,000. The instant that output is spent, the chain records a **realized gain of \$40,000** — your cost basis was \$20,000, your realized value is \$60,000, and SOPR on that spend is 60,000 ÷ 20,000 = **3.0**. The intuition: unrealized gains are the market's *temperature*; realized gains are its *pulse*, the actual money in motion that you can see on-chain.

## SOPR: the spent output profit ratio

Now the headline metric. SOPR — **Spent Output Profit Ratio** — is almost insultingly simple once you have the foundations, and that simplicity is its power.

### The formula and the 1.0 line

For a single spent output:

```
SOPR = (price when spent) / (price when created)
     = realized value / cost basis
```

Aggregate it across *every* output spent in a time window (say, a day), value-weighted, and you get the network-wide SOPR. The reading:

- **SOPR > 1** — the coins that moved today are, in aggregate, in **profit**. Realized value exceeds cost basis. Sellers are sitting on gains.
- **SOPR = 1** — the coins moved at **break-even**. On average, sellers got out at exactly what they paid.
- **SOPR < 1** — the coins moved at a **loss**. Realized value is below cost basis. Sellers are eating losses to get out.

That `1.0` value is not just a number; it is a *psychological pivot*, and it behaves like support and resistance for sentiment.

![SOPR plotted over a cycle bouncing off the 1.0 line in a bull market and rejecting at it in a bear market](/imgs/blogs/profit-loss-sopr-and-hodl-waves-2.png)

Here is why the 1.0 line matters so much. In a **bull market**, the average holder is in profit, so SOPR naturally sits above 1. When price dips, recent buyers who are now slightly underwater would rather *not* sell at a loss — so selling pressure dries up exactly as SOPR approaches 1 from above, and dip-buyers step in. The result is a pattern you can see again and again: in healthy uptrends, SOPR repeatedly dips *toward* 1.0 and bounces, as if the line were a trampoline. People refuse to realize losses, so the line holds.

In a **bear market**, the polarity flips. The average holder is underwater, SOPR sits below 1, and every rally toward break-even brings out sellers who are desperate to get out at "even" — so the 1.0 line becomes a *ceiling* that rejects rallies. SOPR pushes up to 1, fails, and rolls over. The transition from "1.0 is a floor" to "1.0 is a ceiling" is itself one of the cleaner regime-change signals in on-chain analysis.

#### Worked example: a capitulation day at SOPR 0.85

Suppose on a single ugly day, the average coin that moves was last bought at \$40,000 and is being spent at roughly \$34,000. SOPR that day = 34,000 ÷ 40,000 = **0.85** — the coins are moving **15% below cost**. That is capitulation: holders are not selling because they want to, they are selling because they can no longer stand the pain, eating a loss to exit. A trader holding a **\$10,000 position** that is now underwater is exactly the kind of weak hand getting flushed. The intuition: a SOPR reading well under 1, especially after a long decline, means the marginal seller is *exhausted* — and exhaustion of sellers is the raw material of a bottom, not a reason to panic-sell alongside them.

### Adjusted SOPR (aSOPR): filtering out the noise

Raw SOPR has a problem: a huge fraction of on-chain "spends" are coins moving between addresses controlled by the *same* entity — exchange wallet shuffles, consolidations, internal transfers — often at a holding period of less than an hour. These have a SOPR of essentially 1.0 (price barely changed in an hour) and they swamp the signal with meaningless noise.

**Adjusted SOPR (aSOPR)** fixes this by *excluding outputs younger than ~1 hour* (sometimes a longer threshold). The idea: a coin that moved an hour ago and moves again now almost certainly did not represent a real economic decision — it is plumbing, not a trade. Strip those out and what is left is a cleaner read on genuine profit-taking and loss-realization. When analysts say "SOPR" for market-timing, they usually mean aSOPR. We will keep saying SOPR for readability, but assume the adjusted version when we talk about reading the 1.0 line.

This filtering idea generalizes. The reason cohort variants exist at all — aSOPR, STH-SOPR, LTH-SOPR — is that *raw aggregate SOPR is a blunt instrument* that mixes plumbing, panic, and profit-taking into one line. Each refinement removes a source of noise or isolates a behavior: aSOPR removes the sub-hour plumbing, STH-SOPR isolates the new money's emotion, LTH-SOPR isolates the old money's intent. The skill is choosing the *right* cut for the question you are asking. "Is the dip being bought?" is a short-term-holder question. "Is the cycle topping?" is a long-term-holder question. Reading aggregate SOPR for either is like taking the average temperature of a building that is on fire in one room and freezing in another — technically a number, practically useless.

### Realized profit and loss in dollars: the actual money

SOPR is a *ratio*, which is great for spotting the 1.0 pivot but hides the *scale*. A SOPR of 1.5 on a day when almost nothing moved is trivia; a SOPR of 1.5 on a day when \$2B of coins changed hands is a flood of profit-taking. So analysts pair SOPR with the dollar series: **realized profit** (total USD gains locked in by coins moving at a profit) and **realized loss** (total USD losses locked in by coins moving at a loss), often netted into **net realized profit/loss**.

This is the metric that translates psychology into money. Net realized profit spiking to multi-billion-dollar days is the signature of a top — that is paper wealth being converted to cash at scale. Net realized *loss* spiking is the signature of capitulation — that is real money being burned to escape, the financial equivalent of throwing in the towel.

![Pipeline from a coin's cost basis through a spend to realized profit in dollars](/imgs/blogs/profit-loss-sopr-and-hodl-waves-3.png)

The figure traces it end to end: a coin's cost basis is fixed at its last move (here \$20,000), it sits dormant carrying only *unrealized* P/L, then it is spent at \$60,000, the chain compares the two to get SOPR 3.0, and the \$40,000 difference is *realized* — converted from paper into a permanent on-chain fact. Realized P/L is just this, summed across every coin that moved, in dollars.

#### Worked example: net realized profit at a market top

Take a single day near a cycle peak where coins worth a total of **\$5B** change hands. Of those, suppose coins with an aggregate cost basis of **\$1.5B** are spent at a realized value of **\$4B** — a realized profit of **\$2.5B** locked in that day. Meanwhile, only a sliver moves at a loss, say \$50M of realized loss. Net realized profit for the day is roughly **\$2.45B**. That number is not a feeling; it is \$2.45B of paper wealth that *became cash* in 24 hours. The intuition: when net realized profit prints multi-billion-dollar days repeatedly, the market is distributing — early holders are handing coins to new buyers at a profit — and that hand-off, sustained, is what tops are physically made of.

## Coin age, dormancy, and coin-days-destroyed

SOPR tells you *whether* the coins moving are in profit. It does not tell you *who* is moving them — a panicking buyer from last week, or a holder who has sat still since 2017. For that you need the second axis: **age**.

### Coin age and dormancy

The **age** of a coin (more precisely, of a UTXO) is simply how long it has been since it last moved. A coin spent an hour after it was received is one hour old; a coin untouched since 2016 is years old. **Dormancy** is the same idea at the aggregate level: a measure of the *average age of the coins moving on a given day.* When dormancy is low, the coins changing hands are mostly young — recent buyers trading among themselves. When dormancy spikes, *old* coins are moving, and old coins moving is a fundamentally different event from young coins trading.

Why does age matter so much? Because age is a proxy for *conviction*. Someone who has held through a 70% drawdown and three years of boredom has demonstrated low time-preference and strong hands. When *that* person finally moves coins, it is information — either they have decided the cycle is over (distribution), or they are migrating custody, or, in the rare and dramatic case, an early entity is finally cashing out a life-changing position. Young coins moving is the crowd's reflex; old coins moving is conviction changing its mind.

### Coin-days-destroyed: weighting moves by how long they sat still

Here is the elegant metric that captures all of this in one number. A coin accumulates one **coin-day** for every day it sits unspent: 1 BTC held for 100 days has accumulated 100 coin-days. When that coin finally moves, those accumulated coin-days are **destroyed** — reset to zero. **Coin-days-destroyed (CDD)** for a day is the total coin-days wiped out by all the coins that moved:

```
CDD = sum over moved coins of (coins moved x days held)
```

The genius of CDD is the *weighting*. A day where a million fresh coins (each held one day) trade hands destroys very few coin-days. A single ancient wallet of old coins moving destroys an enormous number — because each of those coins had been silently banking coin-days for years. CDD makes the loud signal — old money moving — *loud*, and makes the routine churn of young coins — quiet — quiet. It is a volume metric that automatically up-weights the moves that matter.

![Coin-days-destroyed chart with a flat young-coin baseline and one tall spike where an old coin wakes](/imgs/blogs/profit-loss-sopr-and-hodl-waves-4.png)

The picture is the whole point: a quiet baseline of young coins trading, then one giant bar where an old-coin wallet wakes up. The math is in the box — a 1,000-BTC wallet held for 2,500 days destroys 2.5 million coin-days in a single transaction, dwarfing a busy day of fresh coins.

#### Worked example: a 2017-vintage whale wakes up

Take a wallet that bought **1,000 BTC in 2017 at roughly \$2,000 each** — a **\$2M** cost basis. Seven years later, in 2024, it moves all of it at **\$60,000 per coin**, a market value of **\$60M**. Two things light up at once. First, **realized profit** = (60,000 − 2,000) × 1,000 = **\$58M** locked in — a single transaction realizing fifty-eight million dollars of gains, SOPR = 60,000 ÷ 2,000 = **30**. Second, **CDD** = 1,000 coins × roughly 2,500 days dormant = **2.5 million coin-days destroyed** in one move, a vertical spike on the CDD chart. The intuition: this one transaction is simultaneously a screaming SOPR/realized-profit event *and* a screaming CDD event — which is the textbook fingerprint of old, early money taking life-changing profit. (Whether it is a sale or a custody migration is the next question, and we will not skip it.)

### Smoothing CDD: dormancy and the supply-adjusted variants

Raw daily CDD is spiky and scale-dependent — a single big transaction can dominate, and the absolute number grows as the coin supply grows, making years hard to compare. So analysts use a family of derived metrics. **Dormancy** divides CDD by the number of coins that moved that day, giving the *average age* of the coins in motion (in days) rather than a raw total — a cleaner read on "how old is the typical coin moving right now." **Supply-adjusted CDD** divides by total supply to make different eras comparable. **Binary CDD** and **90-day-average CDD** smooth the spikes into a trend you can actually read. The point of all of them is the same: separate the routine churn of young coins from the rare, loud event of old coins waking. When dormancy trends *up* over weeks — not just one spike — the *average* coin moving is getting older, which means seasoned holders are steadily becoming net sellers. That sustained drift matters more than any single CDD bar, because one bar can be a custody migration, but a multi-week rise in dormancy is a behavioral shift in the holder base.

#### Worked example: dormancy as average age

On a quiet day, 5,000 BTC move with a total of 250,000 coin-days destroyed; dormancy = 250,000 ÷ 5,000 = **50 days** — the typical coin moving is fresh, young hands trading. On a distribution day, 5,000 BTC move but 3,500,000 coin-days are destroyed; dormancy = 3,500,000 ÷ 5,000 = **700 days** — nearly two years, the typical coin moving is now an old holder's coin. If those 5,000 BTC at **\$50,000** are heading to exchanges, that is **\$250M** of old-holder supply hitting the market in a day. The intuition: dormancy turns a raw CDD number into a single intuitive figure — the *age of the average coin selling* — and a jump from 50 to 700 days is the on-chain sound of strong hands turning into sellers.

### Reading CDD spikes carefully

A CDD spike says "old coins moved." It does *not* say "old coins sold." This is the single most important caveat in the whole post, because it is where careless analysts manufacture false signals. When a large, dormant entity simply *moves its coins to a new wallet* — upgrading custody, rotating to a new cold-storage setup, splitting an estate, an exchange migrating its reserves — CDD spikes hard, dormancy spikes hard, and *nothing has been sold.* The coins did not hit an order book. There is no distribution.

The discipline: when you see a CDD/dormancy spike, do not stop at "old coins moved." Trace the destination. Did the coins land on a known exchange deposit address (a genuine sell signal)? Or did they land on a fresh cold wallet that then sits dormant again (a custody migration, no sale)? The chain shows you the destination; read it before you tweet "whale dumping." We will build the explicit decision rule for this shortly.

## HODL waves: the supply, sliced by age

CDD gives you the *flow* of old coins on a given day. **HODL waves** give you the *stock* — a picture of how the entire supply is distributed across age bands at every point in time, which is the single best map of accumulation versus distribution across a whole cycle.

### What a HODL-waves chart is

Take the entire circulating supply and bucket every coin by how long it has been since it last moved: under 1 month, 1–3 months, 3–6 months, 6–12 months, 1–2 years, 2–3 years, 3–5 years, 5+ years, and so on. Stack those bands as percentages of total supply and plot them over time, and you get the famous "rainbow" HODL-waves chart — a layered area chart where warm colors (young coins) sit at the bottom and cool colors (old coins) sit at the top, each band breathing as coins age into older bands or get spent back into younger ones.

![HODL waves stack splitting total supply into age bands from under one month up to five years and older](/imgs/blogs/profit-loss-sopr-and-hodl-waves-5.png)

Reading the bands from young to old gives you a complete behavioral map: the freshest band swells when new buyers are absorbing supply (typical near tops), the oldest bands shrink only when the strongest hands finally move, and the maturing middle records coins quietly crossing from short-term into long-term-holder status.

### Reading the rainbow: accumulation vs distribution

The signal lives in *which bands are swelling*:

- **Young-coin bands swelling (and old-coin bands shrinking)** = **distribution**. Coins are moving from long-dormant wallets into fresh hands. This is the on-chain signature of a market top: old holders are selling to a flood of new buyers, so supply is "getting younger." The under-1-month band ballooning while the 2-year+ bands deflate is a classic top warning.
- **Old-coin bands swelling (and young-coin bands shrinking)** = **accumulation**. Coins are going dormant — new buyers from the last cycle are now holding through time and aging into the older bands, while few coins move. Supply is "getting older." This is the signature of a bear-market base and the long, boring middle of a cycle: maturation, not movement.

The beauty of HODL waves is that it shows this as a *gradual tide*, not a single day's noise. You can literally watch supply mature through a bear market (old bands fattening) and then watch it get spent into new hands through a bull market (young bands fattening). It is the accumulation/distribution cycle made visible.

A practical reading trick: focus on a *single threshold band* rather than the whole rainbow. The share of supply that has *not moved in at least one year* (often charted on its own as "1-year+ HODL supply" or "long-term holder supply") is one of the most-watched lines in on-chain analysis precisely because it strips the noise. When that line is *rising*, coins are going dormant faster than old coins are waking — accumulation, the base of a cycle. When it is *falling*, old coins are being spent faster than new ones mature — distribution, the late stage of a cycle. It tends to peak in the depths of bear markets (everyone who is going to hold is holding) and trough near euphoric tops (old hands have handed off). Watching that one line cross from rising to falling is a cleaner regime signal than trying to read every band at once.

#### Worked example: distribution near a top, in dollars

Suppose near a cycle peak, the share of supply held less than 3 months climbs from roughly 15% to 25% of the ~19.7M BTC supply over a couple of months — about **2M BTC changing into young hands**. If the average price over that window is **\$60,000**, that is on the order of **\$120B** of coins moving from older holders into fresh buyers. You do not need the exact figure to read it: old hands handed a *nine-figure-times-a-thousand* mountain of coins to new buyers, and the HODL-waves young bands ballooned to record it. The intuition: a HODL-waves chart with the young bands swelling at the top of a parabolic move is the market quietly telling you that the people who held longest are the ones now selling — the textbook end-of-cycle hand-off.

## Cohorts: short-term vs long-term holders

So far we have treated "the holders" as one crowd. They are not. The most useful refinement in this whole toolkit is splitting every metric by **cohort** — and the cleanest split is **short-term holders (STH)** versus **long-term holders (LTH)**, divided at roughly **155 days** of holding. (That threshold is empirical: statistically, once a coin has been held ~155 days, the probability it moves on any given day drops sharply — it has "graduated" to strong-hand status.)

![Before and after comparison of short-term holder SOPR as a panic gauge and long-term holder SOPR as a distribution gauge](/imgs/blogs/profit-loss-sopr-and-hodl-waves-7.png)

### STH-SOPR: the panic gauge

Apply SOPR only to coins younger than 155 days and you get **STH-SOPR** — the realized profit/loss of *recent* buyers. Recent buyers have a thin cost-basis cushion (they bought near current prices), so their SOPR is volatile and emotionally driven. **STH-SOPR dropping below 1** means recent buyers are selling at a loss — panic. In bull-market pullbacks, STH-SOPR dips below 1 briefly and snaps back fast as the dip gets bought; sustained STH-SOPR below 1 is the new money capitulating, and bottoms frequently print exactly there. STH-SOPR is the gauge of the *newest, weakest* money's emotional state.

### LTH-SOPR: the distribution gauge

Apply SOPR only to coins older than 155 days and you get **LTH-SOPR** — the realized profit/loss of *seasoned* holders. Long-term holders sit on enormous unrealized profit (they bought far lower), so when *they* move, LTH-SOPR can spike to large values. **LTH-SOPR spiking and staying elevated** means old, strong hands are realizing large gains — i.e. distributing — and that is one of the more reliable late-cycle top warnings. The smartest, longest-conviction money is handing off to the crowd. LTH-SOPR is the gauge of the *oldest, strongest* money's *intent*.

#### Worked example: the two cohorts disagreeing

Take a day where **STH-SOPR = 0.95** (recent buyers are slightly underwater, mildly panicking) while **LTH-SOPR = 8** (long-term holders moving coins are realizing 8× their cost basis). On the surface, aggregate SOPR might read a healthy 1.3 and look benign. Split it, and the story is sharp: the new money is scared and selling at a small loss, while the old money is *gleefully cashing out* at huge profit. Concretely, an LTH spending a coin bought at **\$7,500** and sold at **\$60,000** realizes **\$52,500** per coin at SOPR 8. The intuition: the aggregate number averaged two opposite emotions into a bland middle; the cohort split revealed that strong hands are distributing into weak-hand fear — a far more bearish picture than the blended SOPR suggested.

## How MVRV connects: the aggregate cost-basis cousin

SOPR measures realized profit on coins that *move*. Its close cousin **MVRV** — *market value to realized value* — measures the *unrealized* profit of the *whole* market at once. Realized value is the sum of every coin valued at its last-moved price (the aggregate cost basis of all supply); market value is every coin valued at today's price. MVRV = market value ÷ realized value. When MVRV is high, the average coin is deep in profit, so any coin that moves is *likely* to realize a gain (pushing SOPR above 1). When MVRV is below 1, the average coin is underwater, so moves tend to realize losses (SOPR below 1). They are two views of the same cost-basis machinery — SOPR on the coins in motion, MVRV on the whole stack.

![BTC MVRV ratio over time with the euphoria zone above 3.5 and capitulation zone below 1](/imgs/blogs/profit-loss-sopr-and-hodl-waves-8.png)

The chart is real BTC MVRV data, and it frames SOPR perfectly: MVRV peaked near 3.9 in April 2021 (deep euphoria, the average coin far in profit, the regime where SOPR sits comfortably above 1 and profit-taking floods out) and bottomed near 0.75 in November 2022 (capitulation, the average coin underwater, the regime where SOPR struggles to hold 1). When you read SOPR, glance at MVRV to know which *regime* you are in: SOPR dipping to 1 means something very different at MVRV 0.8 (a beaten-down market, dips are loss-realization) than at MVRV 3.5 (a euphoric market, dips are profit-takers pausing). We cover MVRV and realized cap in depth in [realized cap, MVRV, and cost basis](/blog/trading/onchain/realized-cap-mvrv-and-cost-basis); here it is just the regime backdrop for SOPR.

## How to read it: a walkthrough on Glassnode

Enough theory. Here is how you actually pull these signals up and read them, step by step. The reference tool is **Glassnode** (the most complete on-chain metrics platform), but the same series exist on **CryptoQuant**, and free approximations live on **Dune** dashboards and **Bitcoin Magazine Pro**. You do not need a paid plan to learn the patterns — many SOPR and HODL-waves charts are free to view.

### Step 1 — Find SOPR and anchor on the 1.0 line

Open the SOPR (or aSOPR) chart. The very first thing your eye should do is find the **horizontal line at 1.0** and ask: is the series mostly above it or below it right now? Above = the market is in a profit regime; below = a loss regime. Then look at the *recent behavior near the line*:

- If SOPR has been **dipping to 1.0 and bouncing** repeatedly, you are likely in an uptrend where dips are bought — holders refuse to realize losses, so the line holds as support.
- If SOPR has been **pushing up to 1.0 and failing**, you are likely in a downtrend where every approach to break-even brings out sellers — the line acts as resistance.
- A **reset to exactly 1.0 during a bull-market pullback** is the classic "healthy correction" tell: the froth (high SOPR) cooled off, weak hands who bought the local top got shaken out at break-even, and the metric is now coiled to resume higher. Many traders treat an aSOPR reset to ~1.0 in an established uptrend as a buy-the-dip confirmation.

### Step 2 — Add the cohort split (STH-SOPR and LTH-SOPR)

Switch on STH-SOPR and LTH-SOPR. Now you are reading *who* is in profit or pain. STH-SOPR below 1 = recent buyers panicking (often a local-bottom tell in an uptrend). LTH-SOPR spiking and staying high = old holders distributing (a top tell). When the two diverge — STH scared, LTH greedy — trust the LTH read for the bigger picture, because old money has the better track record.

### Step 3 — Check CDD and dormancy for old-coin moves

Pull up **coin-days-destroyed** (and its smoothed cousins, like the 90-day-average CDD or "dormancy"). Scan for *spikes* against the baseline. A spike means old coins moved. Then — this is the non-negotiable step — **ask where they went.** Glassnode and CryptoQuant publish exchange-inflow metrics; cross-reference. If the CDD spike coincides with a spike in *exchange inflows*, old coins are heading to order books → real distribution. If exchange inflows are flat while CDD spiked, the old coins moved wallet-to-wallet → likely a custody migration, *not* a sale. This cross-check is what separates signal from noise, and we cover the inflow side in [exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows).

### Step 4 — Read the HODL-waves rainbow for the regime

Finally, open the HODL-waves chart and step *back* — this one is about the slow tide, not the daily wiggle. Are the young bands (warm colors, bottom of the stack) swelling? That is distribution; supply is getting younger; you are likely late in a cycle. Are the old bands (cool colors, top of the stack) swelling? That is accumulation; supply is getting older; you are likely in a base or mid-cycle. The HODL-waves chart sets your *strategic* posture; SOPR and CDD give you the *tactical* timing within it.

### Step 5 — Pair with MVRV for the regime backdrop

Glance at MVRV (Step covered above) to know whether you are in a euphoric (MVRV > 3.5) or capitulated (MVRV < 1) regime, because the same SOPR reading means opposite things in each. Now you have the full stack: MVRV for regime, HODL waves for strategic accumulation/distribution, SOPR for the profit/loss pivot, cohort-SOPR for who is acting, and CDD for old-money moves — cross-checked against exchange flows for intent.

## A real episode: the 2021 top and the 2022 capitulation

The clearest way to internalize these signals is to walk through a cycle where they all fired, with real (rounded, approximate) numbers. The 2020–2022 Bitcoin cycle is the textbook case, and the MVRV chart above is the spine of it.

**The 2021 distribution.** Through late 2020 and into the spring of 2021, BTC ran from roughly \$10,000 to a peak near \$64,000 in April 2021. MVRV climbed to about **3.9** — deep in the euphoria zone above 3.5, meaning the average coin was nearly four times its cost basis. In that environment, SOPR sat persistently and comfortably above 1: of course it did, the average holder was massively in profit. The *signal* was not "SOPR > 1" (that was the baseline); it was the *behavior of the cohorts and the age bands.* Long-term-holder supply began *declining* — coins that had aged into the 1-year-plus bands started moving, and LTH-SOPR registered old holders realizing enormous gains. On the HODL-waves chart, the young bands swelled as new buyers absorbed the supply that strong hands were handing them. CDD spiked repeatedly as older coins woke up. Net realized profit printed massive days. Every signal in this post pointed the same way: **old, strong hands were distributing to a flood of new buyers near a euphoric top.** Price topped, chopped, and then began the long fall.

**The mid-2021 reset (not the top yet).** Between the April peak and a second high in November 2021, BTC fell sharply to about \$30,000 mid-year. This is a beautiful teaching moment, because the on-chain signals *distinguished a correction from the end.* SOPR reset down toward 1.0 and bounced; STH-SOPR dipped below 1 as recent buyers panicked and capitulated at small losses; but LTH-SOPR cooled and old hands largely *stopped* distributing, and HODL-waves maturation resumed. MVRV fell from ~3.9 toward ~1.0 but did not collapse below it. The read: a violent but *healthy* bull-market pullback — weak hands flushed, strong hands holding — not the cycle top. Price duly recovered to a new high near \$69,000 in November 2021.

**The 2022 capitulation.** The real top was the November 2021 high, and the descent through 2022 was brutal: a cascade of failures — a major algorithmic stablecoin and its sister token imploding in May 2022, a string of crypto lenders going insolvent through the summer, and a large exchange collapsing in November 2022 — drove BTC under \$16,000. On-chain, this was capitulation in its purest form. SOPR plunged and *held below 1* for extended stretches — coins were moving at sustained losses. STH-SOPR went deeply negative in spirit (well below 1) as recent buyers dumped underwater. Realized *losses* spiked to some of the largest dollar figures in BTC's history; holders were burning real money to escape. MVRV bottomed around **0.75** in November 2022 — below 1, meaning the average coin was now *underwater*, the capitulation zone. Then, as it always does eventually, the selling exhausted itself: there were no weak hands left to flush, SOPR clawed back above 1, supply began maturing again (old bands swelling on the HODL-waves chart), and a new accumulation base formed. The bottom was, in hindsight, exactly where the on-chain capitulation signals screamed loudest.

#### Worked example: sizing the 2022 capitulation read

Suppose during the worst of late 2022 you were tracking a position you had bought at the November 2021 high near **\$60,000**, now worth about **\$16,000** — a **73% unrealized loss**, roughly **\$44,000 underwater per coin**. The naive move is to capitulate alongside the crowd at SOPR < 1. The on-chain read says the opposite: SOPR holding below 1 with realized losses spiking and MVRV at 0.75 is *seller exhaustion*, historically the best accumulation zone in a cycle. A patient buyer adding into that fear — say averaging in another **\$10,000** across the capitulation months near \$16,000–\$20,000 — was buying exactly when the chain showed weak hands had been wrung out. The intuition: the dollar size of your paper loss is a measure of your *pain*, not of *opportunity*; the on-chain capitulation signals measure the opportunity, and they peak precisely when the pain does.

## The decision matrix: capitulation vs profit-taking vs distribution

Put the pieces together and you can classify almost any "the holders are selling" moment into one of a few regimes, each with a different action. This is the practical payoff of the whole post.

![Decision matrix separating capitulation profit-taking old-coin distribution and custody migration by SOPR coin age and CDD](/imgs/blogs/profit-loss-sopr-and-hodl-waves-6.png)

Read the matrix row by row. **Capitulation** — SOPR below 1, mostly young coins, low CDD — is weak hands flushed out at a loss; the action is to *accumulate into the fear*, because seller exhaustion builds bottoms. **Profit-taking** — SOPR steadily above 1, mixed age, moderate CDD — is a healthy rally where holders bank some gains; the action is to *hold and trail the trend*. **Old-coin distribution** — SOPR far above 1, with a big CDD spike from genuinely old coins heading to exchanges — is early money distributing near a top; the action is to *trim risk and tighten stops*. And the trap row, **custody migration** — a big CDD spike from old coins but *no exchange inflow and no real sale* — looks identical to distribution on a CDD chart alone; the action is to *verify the destination first* before doing anything. The whole skill is telling row 3 from row 4.

## The limits: where these signals lie and break

Every metric in this post is powerful *and* fragile, and a serious analyst spends as much time on the failure modes as on the signals. Here are the structural limits, in rough order of how often they bite.

**Custody and exchange migrations fake old-coin spikes.** We have said it twice; it earns a third. The largest single source of false CDD/dormancy spikes is entities reorganizing their own coins. When a major exchange migrates cold-storage reserves, or a custodian rotates wallets, or an early holder splits a position across new addresses, CDD and dormancy spike violently while *zero* coins reach an order book. These events are routine and frequent. The Mt. Gox estate distributions, exchange proof-of-reserves reshuffles, and ETF custodian setups have all produced enormous CDD spikes that headline-writers mislabeled as "whales dumping." The only defense is to trace the destination — and even then, a transfer to an exchange's *deposit* address is a strong sell signal, while a transfer between an exchange's *internal* wallets is not. You need labeled addresses to tell them apart, which is why these metrics are read alongside entity-labeling platforms.

**Cohort aggregation hides as much as it reveals.** SOPR, even split into STH and LTH, is still an *average* over millions of coins. An average can be a bland 1.2 while underneath it a terrified cohort sells at 0.9 and a greedy cohort sells at 5.0 — two opposite stories blended into a meaningless middle. The cohort split helps, but even "long-term holders" is a coarse bucket containing both a 2017 whale and someone who bought 200 days ago. Whenever a metric looks suspiciously calm, suspect that aggregation is hiding a divergence, and drill into finer cohorts or age bands.

**The 155-day STH/LTH line is a convention, not a law.** The split exists because the *statistical* probability of a coin moving drops sharply around five months of holding — but it is a smoothed average, not a hard switch. A coin held 150 days is not categorically different from one held 160 days. Treat the cohort boundary as a useful fiction that captures a real behavioral gradient, not as a wall.

**It works best on Bitcoin, and degrades elsewhere.** Everything here is cleanest on BTC's UTXO model, where each output has one unambiguous creation time and cost basis. On Ethereum and other account-based chains, balances commingle, smart contracts custody huge fractions of supply (staking, bridges, DeFi pools), and a single address pools many cost bases — so SOPR-style metrics are noisier and used with more caution. Worse, a large share of "supply" on smart-contract chains is locked in protocols where the concept of a "holder selling at a profit" barely applies. Read these metrics as BTC-native unless a provider has done the (hard) work to adapt them.

**Lost coins pollute the oldest bands.** A meaningful chunk of the 5-year-plus HODL-waves band is coins that are *lost forever* — keys thrown away, drives destroyed, early holders deceased. These coins will never move, never sell, and never realize anything, yet they sit in the oldest band inflating it. So "ancient supply" is partly a graveyard, not a coiled spring of future selling. When the oldest band shrinks, that *is* meaningful (genuinely old coins moved); but its sheer size overstates how much old supply could ever actually come to market.

**These are context, not triggers.** The deepest limit is conceptual: SOPR, CDD, and HODL waves describe what holders *just did*. They tilt probabilities and frame regimes; they do not predict price. Capitulation "often" precedes bottoms and distribution "often" precedes tops — but the gap between "often" and "always" is where accounts blow up. Use them to size conviction and set posture, combined with price structure, exchange flows, and macro liquidity — never as a standalone signal you trade with leverage.

## Common misconceptions

**"A SOPR spike above 1 means a top is in."** No — SOPR is above 1 for most of every bull market, because the average holder is in profit. A high SOPR is normal and healthy in an uptrend. What flags a *top* is not "SOPR > 1," it is *LTH-SOPR spiking and staying elevated* (old money distributing) together with the HODL-waves young bands swelling. Aggregate SOPR above 1 is the baseline, not the alarm.

**"A CDD spike means a whale is dumping."** This is the single most common error, and it manufactures false headlines daily. A CDD spike means *old coins moved* — and a vast amount of old-coin movement is custody migrations, exchange wallet reshuffles, estate settlements, and internal transfers, *with no sale behind them.* On chains and exchanges, large entities reorganize cold storage routinely; each reshuffle spikes CDD and dormancy while nothing hits an order book. Always trace the destination before reading a spike as selling.

**"SOPR works the same on every chain."** It works *best on Bitcoin.* SOPR's clean cost-basis accounting depends on the UTXO model, where each output has an unambiguous creation timestamp. On account-based chains like Ethereum, the bookkeeping is murkier (balances commingle, smart contracts hold supply, and a single address can pool many cost bases), so SOPR-style metrics are noisier and used more cautiously. When you read a SOPR chart, assume BTC unless told otherwise. The UTXO-vs-account distinction is covered in [how blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account).

**"Below 1 means sell, above 1 means buy."** Backwards. SOPR persistently *below* 1 (capitulation) is historically where you *accumulate*, not sell — sellers are exhausted and realizing losses, which builds bottoms. SOPR far above 1 *with old-coin distribution* is where you *trim*, not pile in. SOPR is a sentiment gauge, not a literal buy/sell trigger, and the contrarian reading (buy fear, sell greed) usually beats the naive one.

**"Realized profit/loss tells you about future price."** It tells you what holders *just did*, not what price *will do*. A capitulation flush of realized losses often precedes a bottom, but "often" is not "always," and these are *probabilistic context*, not a crystal ball. Treat SOPR, CDD, and HODL waves as evidence that *tilts the odds and frames the regime*, to be combined with price structure and other flows — never as a standalone signal you trade blind.

## The playbook: what to do with it

The if-then checklist. Each line is **signal → read → action → what would invalidate it.**

- **SOPR resets to ~1.0 in an established uptrend (with MVRV still elevated, STH-SOPR snapping back fast).** → Read: a healthy bull-market pullback; weak hands shaken out at break-even, froth cooled. → Action: a candidate buy-the-dip *confirmation* alongside your other signals. → Invalidation: if SOPR keeps falling and *holds below 1* while LTH-SOPR turns down too, the regime may be flipping bull→bear; stand aside.

- **SOPR persistently below 1 after a long decline, STH-SOPR deep below 1, realized losses spiking, MVRV < 1.** → Read: capitulation; sellers exhausted, real money burned to escape. → Action: scale *into* the fear over time (not all at once); this is a high-quality accumulation zone historically. → Invalidation: a structural break (a major insolvency, a stablecoin depeg, a chain-level failure) can make "cheap" cheaper; size for the chance you are early.

- **LTH-SOPR spiking and staying elevated + HODL-waves young bands swelling + MVRV in the euphoria zone (>3.5).** → Read: old, strong hands distributing to new buyers near a top. → Action: trim risk, tighten stops, stop adding leverage; respect that the smartest money is selling. → Invalidation: if LTH-SOPR cools back to normal *without* a price break and young bands stop swelling, distribution may have been a pause, not the top.

- **A CDD / dormancy spike from genuinely old coins.** → Read: *old coins moved* — but you do not yet know if it is a sale. → Action: **trace the destination first.** If it lands on exchange deposit addresses with a matching exchange-inflow spike → treat as distribution (bearish). If it lands on fresh cold storage with no inflow → custody migration, *ignore the "dump" narrative.* → Invalidation: the destination *is* the answer; do not act on the spike alone.

- **STH-SOPR below 1 in an otherwise intact uptrend (a brief dip).** → Read: recent buyers panicking at a local low; often a local bottom. → Action: a candidate add point if your higher-timeframe thesis is intact. → Invalidation: if it is the *start* of a broader breakdown (LTH-SOPR also rolling, price losing key structure), it is not a dip to buy.

The meta-rule that ties it together: **SOPR and HODL waves are best as a contrarian sentiment gauge, not a literal signal.** Buy when the chain shows capitulation (SOPR < 1, losses realized, supply maturing); get cautious when the chain shows distribution (old coins moving to exchanges at huge profit, supply getting younger). And on every old-coin spike, *verify the destination before you believe the dump.* The edge is reading the crowd's realized profit and loss directly off the ledger — the trap is mistaking plumbing for a trade.

Who exactly the "old, strong hands" tend to be — early miners, long-term funds, seasoned individuals — and how to identify and follow specific smart-money wallets is its own discipline, and later posts in this series take it up directly. For how this supply-distribution picture connects to concentration risk, see [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration). And for the philosophical root of *why* every coin's history is public in the first place — the property that makes all of this possible — see [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision).

## Further reading & cross-links

- [Realized cap, MVRV, and cost basis](/blog/trading/onchain/realized-cap-mvrv-and-cost-basis) — the aggregate cost-basis machinery SOPR sits on top of, and the regime backdrop (euphoria vs capitulation) for reading any SOPR print.
- [Supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — who holds the float, and how HODL-waves age bands relate to concentration and distribution risk.
- [Exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — the inflow cross-check that tells real distribution (coins to exchanges) from a harmless custody migration.
- [How blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account) — why SOPR's clean cost-basis accounting works best on Bitcoin's UTXO model.
- [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision) — why the ledger is public and permanent in the first place, the precondition for every metric in this post.
