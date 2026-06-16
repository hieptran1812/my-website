---
title: "Realized Cap, MVRV, and Cost Basis: Valuing a Coin by What Holders Actually Paid"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Market cap prices every coin at today's number — but most coins never traded there. Realized cap values each coin at the price it last moved on-chain, MVRV compares the two, and together they form one of the best cycle-timing tools on-chain."
tags: ["onchain", "crypto", "mvrv", "realized-cap", "cost-basis", "nupl", "bitcoin", "glassnode", "valuation", "cycle-timing", "realized-price", "on-chain-metrics"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Market cap (price × supply) is a fiction: it values every coin at today's price, but most coins never traded there. *Realized cap* values each coin at the price it last moved on-chain — the market's true aggregate cost basis — and *MVRV* (market value ÷ realized value) tells you whether holders are sitting on big paper gains or are underwater.
>
> - **What it is:** realized cap sums every coin at its last-moved price, so it measures *money actually committed*, not today's marginal price applied to all coins. Realized price = realized cap ÷ supply is the whole market's break-even.
> - **How to read it:** MVRV above ~3.5 has historically marked euphoria and cycle tops; MVRV near 1 is fair value; MVRV below 1 means the average holder is underwater — capitulation and cycle bottoms. NUPL turns the same idea into a six-zone sentiment ladder.
> - **What you do with it:** these are *cycle-timing context*, not a precise trigger. High MVRV = de-risk and trim; MVRV under 1 = the market is at a discount to its own cost basis, historically a generational accumulation window. Short-term-holder cost basis is a tactical support/resistance level you trade around.
> - **The number to remember:** if Bitcoin trades at \$60,000 and its realized price is \$30,000, MVRV is 2.0 — the average coin is sitting on a 100% paper gain, which is the middle of the band, not the top.

On 22 November 2022, three weeks after FTX collapsed and dragged the whole market into its worst drawdown of the cycle, Bitcoin's price sat around \$16,000 and almost nobody wanted to touch it. The mood was not "buy the dip" — it was "is this thing going to zero." And yet, on that exact day, one specific on-chain number was screaming the opposite of the headlines. Bitcoin's **MVRV ratio** had fallen to roughly **0.75**. Put concretely, the entire market was valued *25% below the average price every holder had actually paid for their coins*. The aggregate cost basis of all Bitcoin in existence was around \$21,000, and the price was \$16,000. The market, as a whole, was underwater.

Here is why that number mattered more than the headlines. In the entire history of Bitcoin, MVRV below 1 had appeared only a handful of times — late 2018, the March 2020 COVID crash, and a few brief dips — and *every single one of them* had marked a major cycle bottom within months. MVRV was not telling you the price would bounce tomorrow. It was telling you that, on a cost-basis-weighted measure, you were being offered the asset below what the people holding it had paid — a condition that historically does not last, because at some point holders stop selling at a loss and the marginal seller is exhausted. By the time anyone could read about "the bottom" in the financial press, MVRV had already quietly recovered above 1.

That is the power of cost-basis metrics, and it is the subject of this post. Market cap — the number every website leads with — is a poor measure of how much money is actually *in* an asset, because it multiplies one number (today's price) by every coin ever minted, including coins that last changed hands at \$200 in 2015 and coins that haven't moved in a decade. Realized cap fixes that by valuing each coin at the price it *last actually moved* on-chain. From that one repair flows an entire toolkit: realized price (the aggregate break-even), MVRV (are holders in profit or pain?), NUPL (the same thing as a sentiment ladder), and the cost basis of specific cohorts that act as support and resistance.

![Market cap values every coin at today's price while realized cap values each coin at its last-moved price](/imgs/blogs/realized-cap-mvrv-and-cost-basis-1.png)

By the end of this post you will understand exactly what realized cap is and why it is "money actually committed," how to compute it from a coin's last-moved price, how MVRV maps to the phases of a market cycle, how NUPL turns paper gains into a sentiment spectrum, and how the short-term and long-term holder cohorts give you cost-basis levels you can trade around. We will be honest, throughout, about the limits: aggregation hides distribution, these metrics work far better for Bitcoin and Ethereum than for low-float alts, and none of them is a precise timer. Three ideas recur: **(1) realized cap measures committed capital, not marginal price; (2) MVRV is a band, not a buy button; (3) the chain shows the cost basis, but the cost basis is an average that hides who is where.**

## Foundations: market cap, cost basis, and the realized stack

Before any ratio or any chart, we build the vocabulary from zero. Every term is defined the first time it appears. If you have never owned a single coin, you will still be able to follow every step.

### Market cap, and why it overstates "money in"

**Market capitalization** ("market cap") is the headline number for any asset: it is the current price of one unit multiplied by the total number of units in existence.

`market cap = price × circulating supply`

For Bitcoin, with roughly 19.8 million coins in circulation, a price of \$60,000 gives a market cap of about \$1.2 trillion. This number is useful for one thing: ranking assets by size. It is dangerously misleading for almost everything else, and the reason is subtle but crucial.

Market cap implicitly claims that *all* the coins are worth today's price. But the price is set at the **margin** — it is whatever the last buyer and the last seller agreed on for the last small batch of coins that traded. The vast majority of coins did not trade at that price. They are sitting in wallets, last moved months or years ago, at prices that might be a tenth or ten times today's level. To say a coin bought at \$1,000 in 2017 and never sold is "worth \$60,000" is true only in the sense that you *could* sell it for that — if the market could absorb it, which it usually cannot.

So market cap massively overstates the amount of money that has actually flowed into an asset. If everyone tried to sell at once, the realized proceeds would be a fraction of the market cap, because selling pressure would collapse the price long before all the coins cleared. Market cap is a *marginal* number masquerading as a *total* number.

#### Worked example: market cap overstates committed capital

Take a tiny toy coin with exactly four coins in existence. One was last bought at \$1,000, one at \$50,000, one at \$59,000, and the most recent trade — the marginal price — was \$60,000. The market cap is `4 × \$60,000 = \$240,000`. But the actual money committed by the holders, summed up, is `\$1,000 + \$50,000 + \$59,000 + \$60,000 = \$170,000`. Market cap claims \$240,000 of value; only \$170,000 was ever actually paid in. That \$70,000 gap is unrealized paper profit that exists only if the price holds — and it is exactly the gap that realized cap is built to measure.

### A coin's cost basis: the price it last moved

In ordinary investing, your **cost basis** is the price you paid for an asset. If you bought a share at \$40, your cost basis is \$40, and your profit or loss is measured from there. On a blockchain, every coin has a kind of cost basis too — and remarkably, *it is public*.

Here is the key fact that makes all of this possible. On a blockchain, every coin (or, more precisely, every unit of value) carries a timestamp of when it last moved, and the network knows the price at that moment. The last time a coin was sent in a transaction, the market had a price. We treat *that price* — the price when the coin last changed hands on-chain — as the coin's on-chain cost basis. It is an approximation of "what the current holder paid," and across millions of coins, the approximations average out into something very robust.

Why "last moved" and not "originally bought"? Because the chain only records movements, not intentions. When you buy a coin on an exchange and withdraw it to your wallet, that withdrawal is an on-chain movement at that day's price. When a coin sits untouched, its cost basis stays frozen at its last move. This is a feature, not a bug: a coin that hasn't moved in five years carries a five-year-old cost basis, which is exactly the information we want — it tells us that whoever holds it is sitting on an enormous unrealized gain and has shown no inclination to sell.

![Each coin carries its last-moved price as a cost basis and realized price averages them](/imgs/blogs/realized-cap-mvrv-and-cost-basis-2.png)

### Realized cap: summing every coin at its last-moved price

Now the central definition. **Realized capitalization** ("realized cap") is the sum, over every coin in existence, of that coin's value *at the price it last moved*.

`realized cap = Σ (each coin × its last-moved price)`

Compare that to market cap, which is `Σ (each coin × today's price)`. The only difference is the price you apply to each coin: market cap uses one price (today's) for all of them; realized cap uses each coin's own last-moved price. Realized cap is therefore a far better proxy for the **total capital actually committed** to the asset — the aggregate of every dollar that was ever spent acquiring coins, valued at the moment of acquisition.

Realized cap was introduced by the analysts Nic Carter and Antoine Le Calvez (the team behind Coinmetrics) in 2018, and it has since become the foundation of an entire family of cost-basis metrics popularized by **Glassnode**, the leading on-chain data provider. It is sometimes called the "thermo-cap of the holders" — the heat of capital that has actually entered, rather than the mirage of marginal price applied to all supply.

A property worth internalizing: realized cap moves *slowly*. Market cap can drop 30% in a day because price is marginal and volatile. Realized cap barely flinches on a price crash, because the coins that already exist still carry their old cost basis — a price drop doesn't change what anyone *paid*. Realized cap only changes when coins *move* (re-pricing them to the new level) or when new coins are minted. This slow-moving quality is precisely why it makes such a good baseline: it is the stable "what was paid" line against which the volatile "what it's worth now" line can be measured.

### Realized price: the aggregate break-even

If you divide realized cap by the circulating supply, you get the **realized price**:

`realized price = realized cap ÷ circulating supply`

This is the single most intuitive number in the whole toolkit. Realized price is the **average cost basis of the entire market** — the supply-weighted average price at which all coins last moved. It answers a beautifully concrete question: *on average, what did the market pay for its coins?*

When the live price is **above** realized price, the average holder is in profit. When the live price is **below** realized price, the average holder is underwater — they are holding coins worth less than they paid. That is why realized price has historically acted as a kind of dynamic support: when price falls to the aggregate cost basis, the average participant is at break-even, and the marginal seller (the person selling at a loss) tends to dry up around there.

#### Worked example: realized price as the market's break-even

Suppose Bitcoin's realized cap is \$600 billion and there are 19.8 million coins. The realized price is `\$600,000,000,000 ÷ 19,800,000 ≈ \$30,300`. That means the average Bitcoin holder paid about \$30,000 for their coins. If the live price is \$60,000, the market is trading at roughly *double* its own cost basis — the average holder is up about 100%. If the live price instead fell to \$25,000, the market would be trading *below* its \$30,000 average cost basis: the typical holder would be sitting at a loss, a condition that historically marks the deep-discount zone of a bear market.

### MVRV: the ratio that ties it together

Finally, the metric that gives this post its name. **MVRV** stands for **Market Value to Realized Value**, and it is simply the ratio of the two caps:

`MVRV = market cap ÷ realized cap`

Equivalently — and this is the more intuitive form — it is the ratio of live price to realized price:

`MVRV = price ÷ realized price`

MVRV measures, in a single number, *how far above (or below) its aggregate cost basis the market is trading.* An MVRV of 1.0 means price equals realized price: the average holder is exactly at break-even. An MVRV of 2.0 means price is double the average cost basis: the average holder is up 100%. An MVRV of 0.75 means price is 25% below the average cost basis: the average holder is down 25%, underwater.

#### Worked example: turning price and realized price into MVRV

A coin trades at \$60,000 and its realized price is \$30,000. Then `MVRV = \$60,000 ÷ \$30,000 = 2.0`. The average holder is sitting on a 100% paper gain. Now express the same thing in caps: if market cap is \$1.2 trillion and realized cap is \$600 billion, then `MVRV = \$1,200B ÷ \$600B = 2.0` — identical, as it must be, because dividing both caps by the same supply leaves the ratio unchanged. The single number 2.0 tells you the market is trading at twice what it cost, which historically sits in the middle of the cycle, not at the euphoric top.

That is the entire foundation. Realized cap is the slow-moving "what was paid" baseline; realized price is its per-coin form, the aggregate break-even; MVRV is the live distance from that break-even. Everything else in this post — the cycle bands, NUPL, the cohort cost-basis levels — is built from these three pieces.

## How realized cap is actually computed

It is worth one section to make the computation concrete, because the mechanics differ between Bitcoin's and Ethereum's data models and because understanding the mechanics is what protects you from misreading the metric.

### On Bitcoin: the UTXO last-moved price

Bitcoin does not have "accounts with balances." It has **UTXOs** — Unspent Transaction Outputs. A UTXO is a discrete chunk of bitcoin, created by a transaction, that sits unspent until it is used as the input to a future transaction. Your wallet's balance is just the sum of all the UTXOs it controls. (If this is new, the sibling post on [how blockchains store data](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account) walks through the UTXO model from scratch.)

The crucial property for our purposes: **every UTXO has a creation timestamp**, and the network knows the market price at that timestamp. When realized cap is computed, each UTXO is valued at the price on the day it was created — i.e. the day those specific coins last moved. Sum that across every unspent UTXO and you have the realized cap. When a UTXO is finally spent, those coins get "re-priced" to the new transaction's date — their contribution to realized cap updates from the old price to the current price.

This is why realized cap is sometimes described as valuing coins at "the price they last touched the chain." A coin that has sat in a 2013-era UTXO contributes its 2013 price to realized cap, year after year, until the day someone finally moves it.

#### Worked example: an old coin re-pricing on a spend

Say a single Bitcoin sits in a UTXO created in 2015, when the price was \$300. For nine years it contributes \$300 to realized cap. Then in 2024 the holder moves it on-chain at a price of \$60,000. The moment that transaction confirms, that coin's contribution to realized cap jumps from \$300 to \$60,000 — a re-pricing of \$59,700 in committed-capital terms for that one coin. Multiply such re-pricings across millions of coins during a bull market, when old holders finally take profit, and you see why realized cap *rises* in bull markets: dormant low-cost coins move and get re-priced up to current levels.

### On Ethereum: balances and a chosen accounting

Ethereum uses the **account model** — addresses with running balances, not discrete UTXOs. There is no native "this exact coin last moved on this date." So realized cap on account-based chains is computed with an accounting convention applied to each address's inflows and outflows (a first-in-first-out or average-cost treatment of when value entered an address at what price). The number is therefore slightly more *model-dependent* on Ethereum than on Bitcoin, where the UTXO gives an exact last-moved price. The interpretation is the same; just know that the Ethereum figure rests on an accounting choice, and different providers may differ at the margin.

### Why realized cap rises in a bull market

A subtle but important behavior: realized cap *grows* during bull markets, and contracts only slowly. This trips people up, because "what was paid" sounds like it should be fixed. It isn't, and the reason is the re-pricing mechanic above. In a bull market, dormant low-cost coins finally move — old holders take profit, coins migrate from cold storage to exchanges and into new hands — and each of those moves re-prices a coin from its old, low cost basis to today's much higher level. Millions of such re-pricings *add* committed capital to the realized cap. So realized cap is a running record of capital *rotating in at ever-higher prices*: it ratchets up as the cycle matures and new, higher cost bases get locked in. After a top, it plateaus and then drifts down only gradually as some coins re-price lower — which is exactly why the realized price (the average cost basis) acts as such durable support on the way down. The market has to fall below years of accumulated, rising cost basis before the average holder is underwater.

### Why this is "money actually committed"

Step back and appreciate what realized cap captures. Every time someone acquires a coin on-chain at some price, that price becomes the coin's cost basis and enters realized cap. Realized cap is therefore the running tally of **all the capital that has been spent acquiring coins, valued at the moment of spending** — net of coins that have since been re-priced by moving again. It is the closest on-chain analogue to "how much money is actually invested in this asset," as opposed to market cap's "what would all the coins be worth if today's marginal price magically applied to every single one." That distinction is the whole reason the cost-basis metrics work.

## MVRV across a cycle: the bands that have timed tops and bottoms

Now we put MVRV to work. The reason MVRV is one of the most-watched on-chain metrics is that, for Bitcoin, it has spent the better part of a decade oscillating within a remarkably consistent band — and the extremes of that band have lined up with cycle tops and bottoms with uncanny regularity.

![Bitcoin MVRV across a cycle with the top zone above 3.5 and the bottom zone below 1 shaded](/imgs/blogs/realized-cap-mvrv-and-cost-basis-3.png)

Read the chart from left to right and the pattern is hard to miss. Every time MVRV pushed up into the shaded red band above roughly **3.5**, the market was at or near a major top — the average holder was sitting on a 250%+ paper gain, euphoria was rampant, and the conditions for a top (everyone in profit, no one left to buy, every old holder tempted to sell into strength) were maximally present. The April 2021 reading near **3.9** is the clearest example: a few weeks later, the first leg of the 2021 top was in.

Every time MVRV fell into the shaded green band below **1.0**, the market was at or near a major bottom — the average holder was underwater, capitulation was setting in, and the marginal seller (selling at a loss) was running out. The late-2018 reading near **0.70**, the March-2020 COVID crash to **0.85**, and the post-FTX November-2022 reading near **0.75** all marked generational accumulation zones. An MVRV near **1.0** — as in mid-2022 — is "fair value," price sitting right at the market's average cost basis.

The historical band reads, for Bitcoin, are roughly:

- **MVRV > 3.5** — euphoria / top-risk zone. Average holder up 250%+. Historically a signal to de-risk, not to add.
- **MVRV ≈ 1.5–3.5** — the normal trending range of a bull market. Healthy, not extreme.
- **MVRV ≈ 1.0** — fair value. Price equals the average cost basis.
- **MVRV < 1.0** — capitulation / bottom zone. Average holder underwater. Historically a generational accumulation window.

A useful way to internalize the bands is to translate them into average holder profit and loss, which is exactly `(MVRV − 1)` expressed as a percentage.

![Average holder profit and loss across the cycle derived from MVRV](/imgs/blogs/realized-cap-mvrv-and-cost-basis-7.png)

At the April-2021 peak, with MVRV near 3.9, the average coin was up about **290%**. At the November-2022 trough, with MVRV near 0.75, the average coin was *down* about **25%**. The chart makes the asymmetry vivid: tops are zones where the average participant has a huge, fragile paper gain that they are itching to protect; bottoms are zones where the average participant is sitting at a loss and has mostly already sold what they were going to sell.

#### Worked example: buying when MVRV is below 1

Suppose Bitcoin's realized price (the aggregate cost basis) is \$30,000 and the live price has fallen to \$24,000 in a brutal bear market. Then `MVRV = \$24,000 ÷ \$30,000 = 0.80` — the market is trading 20% below its own cost basis, and the average holder is underwater. Historically, every time MVRV has dropped this far for Bitcoin, the eventual recovery to even just realized price (\$30,000 from \$24,000) was a 25% move, and the recovery to the next cycle's euphoria zone (MVRV 3.5, i.e. \$105,000 against a \$30,000 cost basis) was multiples higher. A \$10,000 position deployed at \$24,000 that merely returned to the \$30,000 aggregate cost basis would be worth \$12,500 — and that is the *conservative* target, just getting back to break-even for the average holder.

A vital caveat before you treat this as a money machine: **MVRV is a band, not a button.** It told you November 2022 was a deep-value zone, but the exact bottom (\$16,000) came weeks later, and MVRV stayed below 1 for a couple of months. It told you April 2021 was top-risk, but the absolute peak of that cycle came months afterward in a second leg. MVRV identifies *zones of asymmetric risk and reward*, not precise turning points. Used as a precise timer it will whipsaw you; used as a context gauge that says "the deck is now stacked for/against further gains," it is one of the most reliable tools on-chain.

### Sorting the readings into zones

It helps to see every reading sorted into its zone, stripped of the time axis, so the clustering is obvious.

![Every dated MVRV reading sorted into its cycle zone from underwater to top-risk](/imgs/blogs/realized-cap-mvrv-and-cost-basis-4.png)

Sorted from lowest to highest, the readings fall into three clean buckets. The green bars (MVRV below 1) are the underwater bottoms: late 2018, March 2020, November 2022. The amber bars (roughly 1 to 3.5) are the broad fair-to-bullish middle, where most of a cycle is actually spent. The single red bar (April 2021, MVRV 3.9) is the lone top-zone reading in this sample — a reminder of how *rare* the extreme top zone is. MVRV spends very little time above 3.5; when it gets there, it is telling you something genuinely unusual is happening.

### MVRV-Z: normalizing for a maturing asset

There is a refinement worth knowing because it appears on every serious dashboard: the **MVRV Z-score**. As Bitcoin matures, its volatility falls, and a raw MVRV of 3.5 means something different in 2024 than it did in 2013. The Z-score normalizes MVRV by its own historical standard deviation:

`MVRV-Z = (market cap − realized cap) ÷ stdev(market cap)`

This produces a score that has historically peaked in a consistent zone (roughly 6–8) at tops and bottomed near or below 0 at major lows, *adjusting for the asset's changing volatility regime.* When you read a Glassnode or analyst chart that talks about "MVRV-Z entering the red band," they mean this normalized version. The idea is identical to raw MVRV — distance from cost basis — but expressed in standard-deviation units so the bands stay meaningful as the asset ages.

### A dated walk through one full cycle in MVRV

To make the bands concrete, walk the 2018–2022 cycle the way the metric actually moved, because seeing it as a *story* is what builds the instinct.

In **December 2018**, after a year-long bear market that had taken Bitcoin from roughly \$20,000 down toward \$3,200, MVRV sat near **0.70**. Read that number plainly: the market was trading 30% below the average price every holder had paid. The aggregate cost basis was around \$4,500 and price was \$3,200. Sentiment was funereal — "Bitcoin is dead" was its own running joke — and yet the cost-basis metric was flashing the deepest value it had shown in years. Anyone who accumulated in that MVRV-under-1 window, even badly, even early, was buying below the market's own break-even.

By **June 2019**, price had rallied to around \$13,000 and MVRV had climbed to roughly **2.1** — the average holder back up about 110%, a healthy mid-cycle reading, no longer cheap but nowhere near euphoric. Then came the **March 2020** COVID crash: a violent, liquidity-driven flush that briefly knocked price to ~\$3,800 and MVRV back down to **0.85**, under 1 again. That sub-1 print lasted only weeks — one of the shortest deep-value windows on record — and it again marked a generational entry, with price never revisiting those levels.

The bull market that followed drove MVRV to its cycle peak near **3.9 in April 2021**, squarely in the red euphoria zone, with the average holder up nearly 290%. That was the warning. Price chopped, MVRV cooled to around **2.8 by November 2021** even as price made a marginally higher high — a classic *bearish divergence* where price pushed up but the cost-basis premium did not, a sign the rally was running on thinner committed capital. Through 2022 the metric ground down: **1.0 in June 2022** (fair value, price at the average cost basis) and then **0.75 in November 2022** after FTX, back into the deep-value green zone where this post began.

#### Worked example: the round trip in dollars

Trace one notional coin through that cycle. Accumulated near the December-2018 MVRV-0.70 window at \$3,200, it rode to the April-2021 MVRV-3.9 euphoria near \$60,000 — a move from \$3,200 to \$60,000, or roughly 18×, on a \$5,000 stake growing to about \$93,000. A holder who used MVRV as *context* and trimmed into the euphoria zone locked in much of that. A holder who ignored it and round-tripped through the 2022 bear watched the same position fall back toward the \$16,000 MVRV-0.75 zone — about \$25,000 left of the \$93,000 peak. The metric did not call the exact tick either way, but it told you, in real time, when the deck was stacked for you (MVRV under 1) and against you (MVRV over 3.5).

## NUPL: the same idea as a sentiment ladder

MVRV is a ratio. **NUPL** — Net Unrealized Profit/Loss — is the *same underlying information* expressed as a fraction of market cap, which makes it map beautifully onto market psychology. The definition:

`NUPL = (market cap − realized cap) ÷ market cap`

The numerator, `market cap − realized cap`, is the total **unrealized profit** sitting in the market — the paper gains that exist but haven't been sold. Dividing by market cap expresses it as a share: *what fraction of the asset's value is unrealized paper profit?* When NUPL is 0.5, half of the market cap is paper profit. When NUPL is negative, the market is collectively at a loss — realized cap (what was paid) exceeds market cap (what it's worth).

NUPL ranges from below 0 (net loss) up toward 1 (almost all paper profit), and its zones have been given evocative names that track the emotional arc of a market cycle.

![NUPL emotional zone ladder from capitulation at the bottom to euphoria at the top](/imgs/blogs/realized-cap-mvrv-and-cost-basis-5.png)

Reading the ladder from bottom to top:

- **Deep capitulation (NUPL < −0.25):** maximum pain, the market deeply underwater, historic bottoms.
- **Capitulation (−0.25 to 0):** net loss across the market, holders underwater — the accumulation zone.
- **Hope / Fear (0 to 0.25):** small net profit, fragile, the early-recovery zone where sentiment flips.
- **Optimism / Anxiety (0.25 to 0.5):** trending up, gains building, the meat of a bull market.
- **Belief / Denial (0.5 to 0.75):** strong bull, late-cycle conviction, paper profits large.
- **Euphoria / Greed (NUPL > 0.75):** top-risk, distribution by smart holders, the take-profit zone.

The reason NUPL is worth knowing *in addition* to MVRV is that the zone names are a built-in narrative for where the crowd's head is. When NUPL crosses up out of "Hope/Fear" into "Optimism," it is a different statement than a bare MVRV number — it says the market has decisively flipped from a fragile net-small-profit state into a confident uptrend. And when NUPL pushes above 0.75 into "Euphoria," it is the same warning MVRV > 3.5 gives, dressed in the language of crowd psychology: most of the value in the asset is now paper profit that people will rush to protect at the first crack.

#### Worked example: NUPL from the caps

Market cap is \$1.2 trillion and realized cap is \$600 billion. Unrealized profit is `\$1,200B − \$600B = \$600B`. Then `NUPL = \$600B ÷ \$1,200B = 0.5` — exactly half the market's value is paper profit. That puts the market in the "Belief/Denial" zone: a strong bull, late-cycle, plenty of conviction but also plenty of fragile gains. Notice this is the same scenario as our MVRV = 2.0 example (price double the cost basis), just viewed through the NUPL lens: MVRV 2.0 and NUPL 0.5 are two descriptions of one state, because `NUPL = 1 − 1/MVRV` (here `1 − 1/2 = 0.5`). They are the same coin, two faces.

That identity, `NUPL = 1 − 1/MVRV`, is worth holding onto. MVRV and NUPL are not independent signals to be confirmed against each other — they are algebraic transforms of the same two caps. Reading both is reading one thing twice. The value of having both is *communication*: MVRV's ratio is cleaner for the "how many times cost basis" question, and NUPL's zones are cleaner for the "where is sentiment" question.

## The wider cost-basis family

MVRV and NUPL are the headline two, but the realized-cap foundation spawns a whole family of metrics you will meet on any serious dashboard. They are all variations on the same question — *what did holders pay, and where does price sit relative to that?* — and knowing the family keeps you from being confused by jargon.

**Percent Supply in Profit.** Instead of an aggregate ratio, this counts the *fraction of coins* whose last-moved price is below the current price (i.e. coins held at a profit). It rises toward 100% near tops (almost everyone is in the green) and collapses toward 0% near bottoms (almost everyone is underwater). It is the distributional sibling of MVRV: where MVRV gives the average, Percent Supply in Profit gives the *headcount*. The two can diverge usefully — a market can have a high average gain concentrated in a few old whales while a majority of recent coins are underwater, and Percent Supply in Profit catches that split where the MVRV average hides it.

**Realized Profit / Loss and the realized cap's rate of change.** When coins move, they realize their gain or loss — the difference between the new price and their old cost basis. Aggregating those realizations gives daily *Realized Profit* and *Realized Loss*. Heavy realized profit (old coins moving at a gain) is distribution; heavy realized loss (coins moving below cost) is capitulation. This is the spending-behavior layer — the engine behind SOPR — and it is covered in depth in the sibling post on [profit/loss, SOPR, and HODL waves](/blog/trading/onchain/profit-loss-sopr-and-hodl-waves).

**MVRV by cohort.** Just as cost basis can be split into STH and LTH, MVRV itself can be computed per cohort. STH-MVRV (price ÷ short-term-holder cost basis) is a fast, tactical oscillator that swings above and below 1 within a trend; LTH-MVRV is a slow, cycle-scale gauge that only reaches extremes at major tops and bottoms. When STH-MVRV dips below 1 (recent buyers underwater) inside an otherwise healthy uptrend, it often marks a local bottom — a dip where the weak hands are flushed but the long-term holders are still deeply in profit.

#### Worked example: percent-in-profit vs the average

Suppose the average MVRV is 2.0 — the market is, on average, up 100%. You might assume "everyone is winning." But say Percent Supply in Profit reads only 70%. That means 30% of all coins are *underwater* even though the average is a healthy 2× — those are the recent buyers who bought a local top. Concretely, if 30% of a 19.8-million-coin supply (about 5.9 million coins) sits at a loss with an average underwater amount of, say, \$8,000 per coin, that is roughly `5,900,000 × \$8,000 ≈ \$47B` of coins held at a loss — a pool of potential capitulation supply sitting under the market even while the *average* holder is comfortably in profit. The average and the distribution tell different, complementary stories.

### How realized cap behaves when price crashes

One behavior is worth isolating because it surprises newcomers and is the key to reading these charts in a panic. **When price crashes, market cap craters but realized cap barely moves.** Market cap is `price × supply`, so a 50% price drop halves it instantly. Realized cap is `Σ each coin × its last-moved price`, and a price crash does *not* change what any coin last moved at — so realized cap holds roughly flat, drifting only as coins actually transact at the new lower levels (which adds *some* low-cost-basis coins, nudging it down slowly).

The practical consequence: in a crash, MVRV falls fast because its numerator (market cap) is collapsing while its denominator (realized cap) is sticky. That is *why* MVRV plunges under 1 at bottoms — not because committed capital vanished, but because marginal price fell below the sticky average of what was paid. Reading it the other way: MVRV under 1 is the market telling you price has fallen below the durable, slow-moving cost basis of everyone holding, a dislocation that historically corrects upward.

#### Worked example: the sticky denominator in a 50% crash

Market cap is \$1.2 trillion and realized cap is \$600 billion, so MVRV is 2.0. Now price halves overnight in a liquidation cascade. Market cap drops to about \$600 billion; realized cap stays near \$600 billion because no one's cost basis changed (coins didn't suddenly un-buy themselves). New MVRV: `\$600B ÷ \$600B ≈ 1.0`. A 50% price crash took MVRV from 2.0 to 1.0 — from "average holder up 100%" to "average holder at break-even" — without a single dollar of realized capital leaving. If price fell a further 25% to roughly \$450 billion market cap, MVRV would print `\$450B ÷ \$600B = 0.75`, the deep-value zone, purely because marginal price undershot the sticky cost basis. The denominator's stickiness is the whole reason the bands mean anything.

## Cohort cost basis: STH vs LTH as support and resistance

So far we have treated the whole market as one blob with one cost basis. The real power for tactical trading comes from splitting holders into **cohorts** and computing each cohort's realized price separately. The standard split is by holding age.

- **Short-Term Holders (STH):** coins that last moved *within the last ~155 days*. These are the recent buyers, the speculators, the weak hands. Their cost basis is close to recent prices and moves quickly.
- **Long-Term Holders (LTH):** coins that have *not moved for more than ~155 days*. These are the conviction holders, the accumulators, the strong hands. Their cost basis is much lower and moves slowly.

The 155-day threshold isn't arbitrary: empirically, once a coin has been held past roughly that age, the probability it moves (and gets spent) drops sharply — the holder has demonstrated diamond hands. Glassnode and others use it as the dividing line between "still likely to sell" and "committed."

Each cohort has its own realized price — its own aggregate cost basis. And those two cost-basis lines turn out to be *levels that price respects.*

![Short-term and long-term holder cost basis lines acting as support and resistance against price](/imgs/blogs/realized-cap-mvrv-and-cost-basis-6.png)

The **STH cost basis** is the more tactical line. Because short-term holders are the marginal, reactive participants, their break-even tends to act as **support in a bull market and resistance in a bear market**. The logic: in an uptrend, when price dips back to where recent buyers got in, those buyers (now at break-even) tend not to sell, and dip-buyers step in around the same level — so the STH cost basis holds as support. In a downtrend, when a bounce carries price back up to the STH cost basis, recent buyers who were underwater take the chance to exit at break-even, creating supply — so the same line acts as resistance from below. Watching price reclaim or lose the STH cost basis is one of the cleaner regime signals on-chain.

The **LTH cost basis** is the deeper, slower line. It sits well below price for most of a cycle and represents the floor of conviction capital. When price falls all the way to the LTH cost basis — as it did near the 2022 bottom — you are at a genuinely rare extreme: even the strongest, lowest-cost cohort is approaching break-even. That has historically been the deepest of deep-value zones.

#### Worked example: trading the STH cost basis level

Suppose the live price is \$58,000 and the STH cost basis (recent buyers' average) is also \$58,000 — price is sitting right on it. The LTH cost basis is far below at \$40,000. A trader treats the \$58,000 STH line as the pivot: as long as price holds *above* \$58,000, recent buyers are in profit and the path of least resistance is up, so a \$5,000 tactical long with a stop just under \$58,000 risks about \$200–\$300 if the level breaks, against multiples of that to the upside if the uptrend resumes. If price decisively *loses* \$58,000, recent buyers flip underwater and tend to sell, so the same trader flattens or flips — and the next real support is the \$40,000 LTH floor, an \$18,000 air-pocket below. The cohort lines turn a vague "support somewhere down there" into two concrete, on-chain-derived levels with a clear invalidation.

This is the single most *actionable* use of cost-basis metrics. MVRV and NUPL give you slow, cycle-scale context. The STH cost basis gives you a specific price level, updated daily, that the market demonstrably reacts to — a level you can build a trade and a stop around. (For the related read on *who* is holding those coins — concentration, whales, and distribution — see the sibling post on [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration), and for the spending-behavior side, [profit/loss, SOPR, and HODL waves](/blog/trading/onchain/profit-loss-sopr-and-hodl-waves).)

## Why cost-basis levels are self-reinforcing

There is a deeper reason the cost-basis lines work as support and resistance, and it is worth a section because it is the kind of second-order effect that separates understanding a metric from merely plotting it. Cost-basis levels are **reflexive**: they work partly *because so many people watch them.*

Start with the mechanical reason. A holder's willingness to sell is heavily anchored to their entry price — this is the well-documented *disposition effect*, the tendency to hold losers (refusing to "lock in" a loss) and sell winners. When price approaches a cohort's aggregate cost basis from above, the holders in that cohort are nearing break-even; the ones who bought near the top and have been underwater see a chance to exit "even" and take it, which creates supply right at the line (resistance). When price approaches from below in an uptrend, those same holders are back in profit and reluctant to sell, while dip-buyers see a familiar level and step in (support). The cost-basis line is, in effect, the price at which the largest cluster of holders changes its mind about selling.

Now layer on the reflexive part. Glassnode, CryptoQuant, and a hundred analysts publish these exact lines. Traders place orders around them, set alerts on them, and write commentary about them. So the line acquires a second source of power: it becomes a *Schelling point*, a level the market coordinates on because everyone knows everyone else is watching it. The realized price and STH cost basis hold not only because of where holders actually paid, but because a large, attentive slice of the market has agreed to treat those levels as meaningful. That makes them more reliable than a randomly chosen number — and also means they can fail abruptly when the coordination breaks (a hard capitulation slices straight through, because the agreement to defend the level collapses all at once).

#### Worked example: the air-pocket below a broken level

Say the STH cost basis is \$58,000 and it has held as support on three prior dips, each time bouncing a few thousand dollars off the line. A trader who knows the LTH cost basis sits far below at \$40,000 understands the *shape of the risk*: while \$58,000 holds, dips are shallow and buyable; but if \$58,000 breaks decisively, there is no comparable cluster of cost basis until \$40,000 — an \$18,000 air-pocket where support is thin because few coins were acquired in that range. A \$20,000 position held through a clean break of \$58,000 could see an outsized, fast drawdown toward \$40,000 precisely because the level everyone was defending gave way. Knowing where the *next* cost-basis cluster sits is knowing where the floor is, and where it isn't.

This reflexivity is also why these metrics degrade as more capital games them and why they work best on the most-watched, deepest assets. It is the same lesson that runs through the series: a signal that everyone can see gets partly priced in, so the edge is in reading it *in context* — against flows, structure, and distribution — not in mechanically obeying a line on a chart.

## How to read it: a walkthrough on a Glassnode-style dashboard

Let us put the whole toolkit together in the way you would actually use it, reading a cost-basis dashboard (Glassnode, CryptoQuant, or a free Dune board) the way a practitioner does.

**Step 1 — Start with realized price as the baseline.** Pull up the realized price line overlaid on the spot price. The single most important read is simply: *is price above or below realized price?* Above, the average holder is in profit and the market is in a "normal" or bullish posture. Below — price under the aggregate cost basis — and you are in a bear-market deep zone where, historically, you want to be a buyer, not a seller. In our running numbers, realized price \$30,000 with spot \$60,000 says the market is comfortably above cost basis; spot \$25,000 against the same \$30,000 realized price says the market has dipped under its own cost basis.

**Step 2 — Read MVRV (or MVRV-Z) for the band.** Now look at where MVRV sits in its historical band. Is it pressing toward the red 3.5+ euphoria zone (de-risk), sitting in the 1.5–3.5 normal range (hold/trend), at fair value near 1 (neutral), or under 1 in the green capitulation zone (accumulate)? Use the Z-score version if you want the volatility-normalized read that stays comparable across cycles. This is your *cycle-scale* posture.

**Step 3 — Cross-check with NUPL's zone.** Glance at NUPL to confirm the *sentiment* read matches. If MVRV says top-risk and NUPL is in "Euphoria/Greed," the two agree (they must, algebraically) and the message is unambiguous. The NUPL zone name is what you write in your notes: "market entered Euphoria on [date]" is a more memorable journal entry than "MVRV crossed 3.5."

**Step 4 — Drop to the cohort lines for the tactical level.** Finally, pull up the STH and LTH cost-basis lines for your near-term levels. Mark the STH cost basis as your tactical pivot and the LTH cost basis as your deep floor. This is where the slow cycle context (steps 1–3) becomes an actionable price (a level to lean on, with a stop).

**Step 5 — Sanity-check against flows and price action.** No on-chain metric lives alone. Cross-reference what MVRV is telling you with [exchange inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) (is supply moving to exchanges to be sold?) and basic price structure. If MVRV says "deep value" *and* exchange balances are falling (coins leaving exchanges, i.e. accumulation), the signals reinforce. If they conflict, you have found exactly the kind of nuance that separates a careful analyst from someone trading a single green number.

The discipline here is the same one that runs through this whole series: a metric is *context*, and you confirm it against other independent reads before you act. The [on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) post covers which platforms surface these cost-basis charts and how to find them.

## Common misconceptions

**"MVRV below 1 means buy right now / above 3.5 means sell right now."** No — MVRV identifies *zones*, not triggers. In November 2022, MVRV under 1 correctly flagged a deep-value zone, but it stayed under 1 for months and the exact bottom was weeks away. In April 2021, MVRV above 3.5 correctly flagged top-risk, but the cycle's absolute price peak came in a later leg. Treat the extremes as "the deck is now stacked," then time your entries and exits with price action and flows. As a precise timer, MVRV will whipsaw you.

**"Realized cap is the 'true' market cap."** Realized cap measures *committed capital*, which is genuinely more meaningful than market cap for "how much money is in," but it is not "the real value." It is one of two useful numbers. Market cap tells you marginal value and size; realized cap tells you committed cost basis. You need both — their *ratio* (MVRV) is the insight, not either one alone.

**"MVRV and NUPL are two confirming signals."** They are algebraic transforms of the same two caps — `NUPL = 1 − 1/MVRV`. When they "agree," that is arithmetic, not corroboration. Reading both is reading one signal in two dialects. Genuine confirmation comes from *independent* metrics: exchange flows, SOPR, supply distribution, derivatives funding.

**"These metrics work for any coin."** They work *beautifully* for Bitcoin and reasonably for Ethereum, because those assets have deep history, broad distribution, and reliable on-chain price data. For a low-float altcoin where a handful of wallets hold most of the supply, realized cap is dominated by those few wallets' cost basis and the aggregate is nearly meaningless — the "average holder" is a fiction when three wallets are 80% of supply. Cost-basis metrics assume a broad, deep holder base. Apply them to a thin alt and you are measuring noise.

#### Worked example: why cost-basis metrics break for a thin alt

Take an altcoin trading at \$0.50 with a realized price of \$0.80 — MVRV `\$0.50 ÷ \$0.80 = 0.625`, so on paper the "average holder" is down about 38% and the metric flashes "deep value." But suppose 80% of the supply sits in three insider wallets that received their tokens at \$0.001 in a private round. Those insiders are up 500×, not down 38% — the \$0.80 realized price is an artifact of a small float of public buyers who bought the top, dominated by however the accounting treats the locked insider supply. A \$1,000 position bought here on the strength of "MVRV says deep value" could be exit liquidity for insiders sitting on enormous gains. The lesson: before trusting MVRV on any asset, check the *distribution* — cost-basis metrics are only as honest as the holder base is broad.

**"Realized price is a hard floor that price can't break."** Realized price acts as support *statistically and often*, not mechanically. Price has spent months below realized price in deep bears (2018–19, mid-2022). "Dynamic support" means "a level the marginal seller tends to exhaust near," not "a wall." In a genuine capitulation, price slices through realized price and trades below the average cost basis until forced selling is done.

## The playbook: what to do with it

The if-then checklist for using cost-basis metrics as a trader, investor, or analyst. The signal → the read → the action → the invalidation.

- **Signal: MVRV / MVRV-Z pushes into the euphoria zone (MVRV > 3.5).**
  Read: average holder up 250%+, late-cycle, fragile paper gains everywhere.
  Action: de-risk — trim, take profit on a plan, tighten stops, stop adding leverage. Do *not* short blindly; tops extend.
  Invalidation / false positive: a maturing asset can top at a lower MVRV each cycle (the Z-score helps). MVRV high *and still rising* with price is not yet a top — it's the warning, not the event.

- **Signal: MVRV falls below 1 (price under the aggregate cost basis).**
  Read: average holder underwater, capitulation, marginal seller exhausting.
  Action: accumulate on a plan (dollar-cost in, don't lump-sum the falling knife), extend time horizon, stop using leverage that can liquidate you out of the bottom.
  Invalidation / false positive: MVRV can stay below 1 for months; the deep COVID-style flush can briefly hit 0.85 and the post-FTX low 0.75 — "below 1" is the zone, not the bottom tick.

- **Signal: price reclaims / loses the STH cost basis.**
  Read: recent buyers flipping into profit (bullish) or into loss (bearish); a regime pivot for the marginal participant.
  Action: use it as a tactical pivot — long bias above with a stop just under the line; flatten or flip on a decisive loss; next real support is the LTH cost basis below.
  Invalidation / false positive: whipsaws around the exact line are common; require a *decisive* close beyond it (and ideally confirming volume), not a single wick.

- **Signal: NUPL crosses a zone boundary (e.g. into Euphoria, or up out of Capitulation).**
  Read: a sentiment-regime change in the crowd's aggregate paper P/L.
  Action: log it as a context flag and adjust posture (more cautious entering Euphoria, more constructive exiting Capitulation). It is the *narrative* layer over MVRV.
  Invalidation / false positive: it's the same data as MVRV — don't double-count it as independent confirmation.

- **Signal: cost-basis metric flashes extreme on a low-float altcoin.**
  Read: probably an artifact of concentrated supply, not a real "average holder."
  Action: check the holder distribution first; if a few wallets dominate supply, *discard* the MVRV read and value the asset another way.
  Invalidation / false positive: the metric itself — on a thin alt, treat any cost-basis extreme as noise until the distribution proves otherwise.

The throughline: cost-basis metrics give you a *cost-basis-anchored map of the cycle* — where the average holder stands, how far from break-even the market is trading, and which specific levels the crowd reacts to. They are among the best slow-context tools on-chain. They are not a precise timer, they assume a broad holder base, and MVRV and NUPL are the same signal twice. Read them as the *context layer* under your flows and price action, never as a single green or red number to obey.

For the broader frame — why these on-chain valuation tools matter for the asset class as a whole — the macro view in [crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset) and the origin story in [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision) put cost-basis analysis in its place: a way of reading committed capital on a transparent ledger that no traditional asset can offer.

## Further reading & cross-links

- [What is on-chain analysis?](/blog/trading/onchain/what-is-onchain-analysis) — the front door to reading the ledger for an edge.
- [How blockchains store data: UTXO vs account](/blog/trading/onchain/how-blockchains-store-data-utxo-vs-account) — the UTXO model that makes a coin's last-moved price computable.
- [Profit/loss, SOPR, and HODL waves](/blog/trading/onchain/profit-loss-sopr-and-hodl-waves) — the spending-behavior cousins of cost-basis metrics.
- [Supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — who holds the coins, and why distribution decides whether MVRV is meaningful.
- [Exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — the flow signal you cross-check MVRV against.
- [The on-chain tooling landscape](/blog/trading/onchain/the-onchain-tooling-landscape) — where to find Glassnode-style cost-basis charts.
- [Crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset) — the macro frame around on-chain valuation.
- [Bitcoin and the cypherpunk vision](/blog/trading/crypto/bitcoin-and-the-cypherpunk-vision) — the transparent-ledger origin that makes any of this possible.
