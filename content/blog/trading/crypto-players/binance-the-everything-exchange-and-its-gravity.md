---
title: "Binance: The Everything Exchange and Its Gravity"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A build-from-zero profile of Binance as a market participant rather than a marketplace — the share of global volume it clears, the BNB burn flywheel, Launchpad and Launchpool as a distribution machine, the listing and delisting levers, the venture arm that invests in tokens the exchange lists, the stablecoins it depends on, the November 2023 guilty plea, and why one venue this large bends price discovery even when everyone inside it behaves."
tags: ["crypto", "binance", "bnb", "crypto-players", "centralized-exchange", "listings", "launchpool", "token-burn", "market-structure", "stablecoins", "price-discovery"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 52
---

> [!important]
> **TL;DR** — Binance is not a neutral marketplace that crypto happens to trade on. It is the largest single participant in its own market, and a token can be downstream of it in seven different ways at once.
>
> - **The scale:** Binance's share of centralized spot volume peaked at **55.2% in January 2023**, fell to **30.1% by December 2023** and a four-year low near **27% in October 2024**, then settled around a third — **34.0% in January 2026** and **35.4% in March 2026** (CCData). On derivatives it ran roughly **35% of a ~\$4.9 trillion quarter** in Q1 2026. Binance's own ninth-anniversary release (14 July 2026) claims **323 million-plus registered users** and **\$156 trillion** of cumulative volume.
> - **The flywheel:** BNB started at **200 million** tokens with a whitepaper promise to burn half. As of the 36th quarterly burn (15 July 2026) supply is **133,166,127.91 BNB** — and the burn is sized by BNB Smart Chain's block count and BNB's price, not by Binance's profits.
> - **The distribution machine:** Launchpad and Launchpool hand out a few percent of a new token's supply to people who lock BNB. That is not a giveaway; it is a way to manufacture holders, volume and a price on day one.
> - **The levers:** a listing, a Monitoring Tag, a delist vote and a Launchpool schedule are each a supply-or-liquidity event that Binance controls the timing of. Binance Labs — rebranded **YZi Labs in January 2025** — invests in tokens the exchange also lists.
> - **The public record:** on **21 November 2023** Binance pleaded guilty and the Justice Department announced a resolution of **\$4,316,126,163** (DOJ press release, 21 November 2023); Changpeng Zhao pleaded guilty to one Bank Secrecy Act count, was fined \$50 million personally, stepped down as CEO, and was sentenced on **30 April 2024** to four months. He was pardoned on **23 October 2025**. These are settled matters of public record and nothing in this article alleges conduct beyond it.
> - **The number to remember:** in the nine months to 27 July 2026, burns removed **3.3%** of BNB's supply while the price fell **58%** from its all-time high (Binance Square burn announcements; CoinMarketCap, as of 27 July 2026). If you own the token for the burn, you own it for the smallest force acting on it.

There is a question that sounds naive and is actually the whole subject: *who is the counterparty when you trade on the world's largest crypto exchange?*

On the New York Stock Exchange the answer is boring, and boring is the point. Some other investor is on the other side. The exchange matches you, takes a fee measured in fractions of a cent, and has no view on whether Apple goes up. It does not own Apple shares. It did not seed-fund Apple. It does not issue the dollars you paid with. It does not run the system that settles the trade. Those functions live in different companies, and the separation is so complete that nobody thinks about it.

Now ask the same question on Binance. You buy a token. The order matches in Binance's engine. The coin sits in Binance's wallet. The pair you traded is quoted in a stablecoin Binance partnered to launch. If the token is recent, it may have been distributed to the market through Binance's own Launchpool, to people who locked Binance's own token to qualify. The venture fund that led its seed round may be the firm that used to be called Binance Labs. It settles, if it settles on-chain at all, on a blockchain Binance's founder created. And whether the market gets to keep trading it at all depends on a risk label Binance applies and a delisting vote Binance runs.

None of that requires anyone at Binance to do anything wrong. That is what makes it interesting. The diagram above is the mental model for this entire article: **the concentration itself is the mechanism**. When one firm occupies seven roles that are normally held by seven firms, prices stop being a clean signal about what an asset is worth and start being partly a signal about what one company decided this week.

![Binance sits at the center of seven roles — venue, token, chain, launch platform, venture arm, quote asset and risk label — that are separate firms in traditional markets](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-1.webp)

This post builds the whole picture from zero. If you have never placed a trade, start at the beginning and nothing will be assumed. If you already know what a maker fee is, skim the foundations and start at the scale section. Either way, the destination is the same: a working method for reading Binance-related news as what it actually is — a supply or liquidity event with a timestamp on it.

This is educational writing about market structure, not investment advice.

## Foundations: what a centralized exchange actually is

Before we can talk about Binance's gravity we need four ideas, defined from nothing.

### An exchange, an order book, and the two fees

An **exchange** is a place where buyers and sellers meet. A **centralized exchange** — often shortened to *CEX* — is one run by a single company that stands in the middle of everything.

The matching happens in an **order book**, which is just two sorted lists. One list holds every offer to buy, ranked by price, highest first. The other holds every offer to sell, ranked lowest first. The highest bid and the lowest ask are the two prices you see quoted, and the gap between them is the **bid-ask spread** — the small toll you pay for the privilege of trading right now instead of waiting.

There are two ways to trade against that book, and they are charged differently:

- A **maker** posts an order that *sits* in the book waiting — "I'll buy at \$99." Makers add liquidity, so exchanges charge them less.
- A **taker** hits an order that is already there — "I'll buy at whatever the current ask is." Takers remove liquidity, so exchanges charge them more.

Binance's published standard is **0.100% maker and 0.100% taker** on spot trading, dropping to **0.075%** on both sides if you pay fees in BNB, which applies a 25% discount (Binance fee schedule, accessed 27 July 2026). Large traders pay far less through **VIP tiers**, a volume ladder: VIP 1 requires at least \$1 million of 30-day volume and 5 BNB held; VIP 3 requires \$20 million and 100 BNB and pays 0.040% maker / 0.060% taker; the top rung, VIP 9, requires \$4 billion of monthly volume and 5,500 BNB and pays **0.011% maker / 0.023% taker** (Binance fee schedule, accessed 27 July 2026).

Hold onto that spread between 0.100% and 0.011%. Almost everything about who really pays for a crypto exchange is contained in it.

### What you actually own when you deposit

This is the part beginners consistently get wrong, and it matters more than any fee.

When you send a coin to Binance, the coin moves on the blockchain into a wallet Binance controls. Binance then credits your account balance. From that moment until you withdraw, **your balance is a row in Binance's database.** When you trade, no blockchain is involved. Binance decrements one row and increments another. The coin does not move. It cannot move — it is not yours to move.

![On a centralized exchange, deposit and withdrawal touch a blockchain; everything in between is an internal ledger entry](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-2.webp)

This is why the phrase "not your keys, not your coins" exists, and why the collapse of FTX mattered so much: an exchange's internal ledger can say anything until someone tries to withdraw. If you want the full mechanics of custody, commingling and what proof-of-reserves does and does not prove, [the companion post on centralized exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) covers that ground, and [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) is the case study.

For our purposes the relevant consequence is narrower: **because trading is internal, the exchange sees everything.** It knows every order, every cancellation, every account's position, in real time, before anyone else. That is not an accusation. It is an architectural fact, and it is true of every centralized exchange on earth.

### Spot versus perpetual futures

**Spot** trading is what it sounds like: you exchange dollars for a coin and you own the coin.

A **perpetual future** — a *perp* — is a contract that tracks a coin's price without ever expiring and without you owning anything. You post **margin** (collateral) and take a leveraged position. To keep the contract's price glued to the spot price, the exchange charges a **funding rate**: a small payment made every few hours from longs to shorts, or shorts to longs, depending on which side is more crowded.

Perps matter here for one reason: they are where the volume is. CCData put total centralized derivatives volume at **\$85.7 trillion for 2025** (CCData full-year 2025 data) — many multiples of spot. Fees are thinner per trade, but the notional is enormous.

### Float, FDV, and why the two diverge

A crypto token has a **total supply** (every token that will ever exist) and a **circulating supply**, usually called the **float** — the tokens actually free to trade right now. The rest sit locked: with the team, with investors, in a treasury, on a vesting schedule.

Two valuations follow:

- **Market cap** = float × price. What the tradable tokens are worth.
- **Fully diluted valuation (FDV)** = total supply × price. What *all* tokens would be worth at today's price.

When a token lists with 10% of its supply floating, the market has only ever priced 10% of it, and FDV is a number no buyer has ever had to defend. The gap between those two numbers is where most of the drama in this article lives, and [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) walks the full schedule.

### The Binance group, in plain terms

Finally, "Binance" is not one thing. The pieces that matter here:

| Piece | What it is |
|---|---|
| Binance.com | The global exchange — spot, margin, perpetual futures |
| BNB | The exchange's token: fee discount, Launchpool ticket, gas on BNB Chain |
| BNB Smart Chain | A blockchain created by Binance's founder; BNB is its gas token |
| Launchpad / Launchpool | Token distribution platforms that require holding BNB |
| Binance Labs → YZi Labs | The venture arm, rebranded in January 2025 |
| Binance Alpha / Binance Wallet | A pre-listing venue for early-stage tokens |
| Binance.US | A separate US entity, small by comparison |

That is the entire vocabulary. Everything from here builds on it.

## The scale: what "largest" actually means

Numbers first, interpretation second.

**Spot market share.** Binance's share of centralized spot trading volume, as measured by CCData, ran at **55.2% in January 2023** — a majority of all centralized spot trading on the planet, in one company. It fell hard through that year to **30.1% by December 2023** (CCData, reported by CoinDesk on 11 December 2023), a decline that coincided with the loss of a zero-fee promotion and the year's regulatory pressure. Bloomberg reported a four-year low near **27% in October 2024**. It then recovered: **35.09% in Q3 2025**, **34.0% in January 2026**, **35.4% in March 2026** (CCData Exchange Review).

![Binance's share of centralized spot volume: 55.2% in January 2023, 30.1% by December, a 27% low in October 2024, and a recovery to roughly a third through 2026](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-3.webp)

Read that chart the way a market-structure person reads it. The interesting fact is not the decline. It is that **after losing nearly half its share, Binance still clears about one in three of all centralized spot trades.** The second-largest venue is not close. A "collapse in dominance" left it with a share that would be a monopoly finding in most other industries.

**Derivatives.** CCData data reported in Q1 2026 put Binance at roughly **35% of the derivatives market on about \$4.9 trillion of volume for the quarter** (CCData, Q1 2026). In its May 2026 Exchange Review, CCData described Binance as holding the largest share without publishing a specific percentage.

**Volume and users, per Binance itself.** In a ninth-anniversary release on 14 July 2026, Binance claimed **more than 323 million registered users across 100-plus countries** and **\$156 trillion in cumulative all-time trading volume** (Binance via PR Newswire, 14 July 2026). These are the company's own figures and are not independently audited. In January 2026, CCData put Binance's monthly spot volume at **\$407 billion**, up 10.8% on the month (CCData Exchange Review, report dated 16 February 2026).

**Revenue.** Binance is private and publishes no audited financial statements. Forbes reported on 10 March 2026 an estimated **\$16–17 billion of revenue for 2024–2025**, roughly 2.5x Coinbase's \$6.6 billion, and an estimated company valuation near \$100 billion with CZ owning about 90% (Forbes, 10 March 2026). Every one of those is a press estimate, not a filed number.

Let's see whether the estimate is even plausible.

#### Worked example: what a third of the market is worth in fees

We know one real number — Binance's January 2026 spot volume of \$407 billion — and one real fee schedule. Let's build up.

**Step 1 — the naive ceiling.** If every trade paid the standard 0.100% on both sides, \$407 billion of notional would generate:

\$407bn × 0.10% × 2 sides = **\$814 million** in one month, from spot alone.

**Step 2 — kill the naive number.** That is far too high, and the fee schedule tells us why. The top VIP tier pays 0.011% maker and 0.023% taker — roughly *one-tenth* the standard rate — and the largest traders do most of the volume. Add the 25% BNB discount and periodic zero-fee promotions. So assume a **blended 0.03% across both sides**, which is a guess we will immediately test:

\$407bn × 0.03% = **\$122 million per month** from spot.

**Step 3 — add derivatives.** Binance's Q1 2026 derivatives volume of ~\$4.9 trillion is about **\$1.63 trillion per month**. Perp fees are thinner; assume a blended 0.015%:

\$1.63tn × 0.015% = **\$245 million per month**.

**Step 4 — total and annualize.** \$122M + \$245M = **\$367 million per month**, or roughly **\$4.4 billion a year**.

**Step 5 — check against the estimate.** Forbes's \$16–17 billion is nearly four times our figure. Both can be true: our blended rates may be too conservative, and exchange revenue includes far more than trading fees — withdrawal fees, listing-adjacent revenue, staking and "Earn" products, custody, card programs, and BNB Chain activity.

The more useful output of this exercise is the **sensitivity**. One single basis point — 0.01% — of blended take rate is worth about **\$41 million a month on spot** and **\$163 million a month on derivatives**. That is why the VIP ladder is designed with such precision, and why fee promotions are strategic weapons rather than marketing.

*The intuition: at this scale, market share is not a vanity metric. Each percentage point of it is a recurring annuity measured in hundreds of millions of dollars, which tells you how hard the firm will fight to defend it.*

## BNB: the token the exchange manufactures demand for

Every exchange would like its users to hold its own token. Binance is the case study in what happens when that works.

### What BNB is, and what it is not

BNB is a token that does three concrete things:

1. **Cuts your trading fees** by 25% on spot and margin if you elect to pay fees with it (Binance fee schedule, accessed 27 July 2026).
2. **Buys you access** — Launchpad allocations, Launchpool farming and HODLer airdrops are all sized by how much BNB you hold.
3. **Pays for gas** on BNB Smart Chain, the blockchain where BNB is the native fee token.

What BNB is *not*, in Binance's own positioning, is equity. It carries no shareholder rights, no dividend, and no claim on Binance's profits. In July 2024, a US court ruled in the SEC's case against Binance that secondary sales of BNB were not securities transactions. If you want the general principle — why owning a token is structurally different from owning a share — [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) is the dedicated treatment.

BNB launched in an ICO from 26 June to 3 July 2017 with a total supply of **200,000,000** and a whitepaper commitment to eventually destroy half of it (BNB whitepaper, 2017). Binance's own account is that it sold 100 million BNB and raised about \$15 million. Forbes reported on 5 October 2023 that only about **10.78 million BNB were actually distributed** and the raise was under \$5 million. Both accounts are on the record; the discrepancy is unresolved publicly and worth knowing about.

As of 27 July 2026, BNB traded around **\$572** with a market capitalization near **\$76 billion** (CoinMarketCap), against an all-time high of **\$1,370.55** set on **13 October 2025**.

### The burn, and why it changed

For years, Binance burned BNB quarterly in an amount tied to **20% of that quarter's exchange profits** (the pre-2022 rule, per Binance Academy). That created an obvious problem: the burn was a number only Binance could compute, from financials only Binance could see.

On **22 December 2021**, Binance replaced it with **Auto-Burn**. The published formula takes two inputs — the number of blocks produced on BNB Smart Chain during the quarter, and the quarter's average BNB price — and divides their product by a fixed constant. The burn halts permanently when supply reaches 100 million.

The algebra matters less than one property: **Binance's profits do not appear in it anywhere.** The burn became a function of public, verifiable, on-chain data. That was the point.

Running alongside it is **BEP-95**, a mechanism announced by BNB Chain on 22 October 2021 that burns a portion of every block's gas fees in real time — initially 10%, adjustable by governance (BNB Chain blog, 22 October 2021). So BNB has two supply sinks: a large quarterly event and a continuous drip.

![The BNB flywheel: volume produces fees, blocks and price produce burns, scarcity and fee discounts produce demand for BNB, which produces volume](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-4.webp)

Now look at that wheel and notice what is unusual about it. In a normal buyback, a company converts *profit* into *scarcity* — shareholders get a claim on real cash flow. Here, the exchange's activity produces blocks on a chain the same founder created, and those blocks produce scarcity in a token whose primary utility is a discount at the same exchange. Every arrow starts and ends inside the same building. It is a genuine mechanism, verifiable on-chain, and it is also entirely self-referential.

#### Worked example: the burn math, honestly

Let's use two real, primary-sourced data points.

**The inputs.** Binance's 33rd quarterly burn, announced on **27 October 2025**, destroyed **1,441,281.413 BNB** worth roughly **\$1.208 billion**, leaving total supply at **137,738,379.26 BNB** (Binance Square, 27 October 2025). The 36th burn, on **15 July 2026**, destroyed **1,615,827.795 BNB** worth roughly **\$931.7 million**, leaving **133,166,127.91 BNB** (burn announcement, 15 July 2026).

**Step 1 — how much supply went away.**

137,738,379.26 − 133,166,127.91 = **4,572,251.35 BNB** over three quarters, or 261 days.

As a percentage of the starting supply: 4,572,251.35 / 137,738,379.26 = **3.32%**.

Annualized: 3.32% × (365 / 261) = **about 4.6% a year**.

**Step 2 — how long to the floor.** 133,166,127.91 − 100,000,000 = **33,166,127.91 BNB** left to burn, about 24.9% of current supply. The three burns averaged 1,524,084 BNB each. At that pace:

33,166,128 / 1,524,084 = **21.8 quarters ≈ 5.4 years**, landing somewhere around late 2031.

**Step 3 — the part that actually matters.** Over roughly the same window, BNB went from its 13 October 2025 all-time high of **\$1,370.55** to **\$572.37** on 27 July 2026 — a decline of **58.2%** (CoinMarketCap, as of 27 July 2026).

So: supply fell 3.3%. Price fell 58.2%. **Demand moved the price roughly seventeen times as far as supply did.**

**Step 4 — the detail people miss.** Burn #36 destroyed *more coins* than burn #33 — 1.616 million versus 1.441 million — but only \$931.7 million of value versus \$1.208 billion. The quantity went up and the dollar value went down, because the price fell in between. A burn denominated in coins is not a buyback denominated in dollars, and headlines that report the dollar figure are reporting a number that mostly moved for a different reason.

*The intuition: the burn is real, mechanical and verifiable, and it is a slow ~4-5% annual supply reduction. If you hold BNB for the burn, you are holding it for the smallest of the forces acting on its price.*

![BNB supply from 200 million at genesis to 133.17 million after the 36th burn, against a 100 million floor — with the price down 58% over the same recent window](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-5.webp)

### What the fee discount is actually worth — and what it costs

The discount is not a gimmick. At scale it is a large, calculable number. So is the risk of the inventory you must hold to get it.

#### Worked example: the VIP 3 trader

**The setup.** You trade **\$20 million of spot per month**, all as a taker. That qualifies you for VIP 3, which requires at least \$20 million of 30-day volume *and* at least 100 BNB held.

**Step 1 — fees without any optimization.** At the standard 0.100% taker rate:

\$20,000,000 × 0.100% = \$20,000 per month = **\$240,000 a year**.

**Step 2 — fees at VIP 3, paying in BNB.** VIP 3 taker is 0.060%; electing to pay fees in BNB takes another 25% off:

0.060% × 0.75 = **0.045%** → \$20,000,000 × 0.045% = \$9,000 per month = **\$108,000 a year**.

Annual saving: **\$132,000**.

**Step 3 — what the ticket cost.** You must hold 100 BNB. At \$572.37 on 27 July 2026 that is **\$57,237** of capital tied up. But capital tied up is not the real cost — *price risk* is. Those same 100 BNB were worth **\$137,055** at the 13 October 2025 all-time high. Between then and 27 July 2026, holding them cost:

\$137,055 − \$57,237 = **\$79,818** of mark-to-market loss.

**Step 4 — net it out.** The discount saved you \$132,000 a year. The inventory required to unlock it lost \$79,818 in about nine months — roughly **seven months' worth of the saving it bought**.

*The intuition: the fee discount is real money, and the collateral it demands is an unhedged position in the exchange's own token. You are paid a certain, modest sum to take an uncertain, large risk. That is the trade, and it is worth stating explicitly rather than absorbing by accident.*

## The distribution machine: Launchpad, Launchpool and Alpha

Here is where Binance stops being a venue and becomes something closer to an underwriter.

### Launchpad: the allocation auction

**Launchpad** is a token sale run inside the exchange. Binance takes a multi-day snapshot of your average BNB balance, which sets the maximum you may commit. You commit BNB during a subscription window, and your allocation is:

> your committed BNB ÷ total committed BNB × tokens for sale

Unused BNB is refunded. CryptoRank's tracker counts **85 launches since 2017 raising roughly \$133.14 million** in total (CryptoRank Launchpad tracker, accessed 2026) — a small number that tells you the raise is not the point.

The returns that made Launchpad famous were extraordinary, and they need careful framing. CoinGecko Research's tally reports Axie Infinity (AXS) at a **1,649x** peak from its 2020 launch price of \$0.10 to its \$164.90 all-time high in November 2021; Polygon (MATIC) at **1,110x** from \$0.00263 in 2019 to \$2.92 in December 2021; The Sandbox (SAND) at **1,008x** (CoinGecko Research, peak-multiple tally, 2024).

Read those with two caveats. First, they are **peak-to-launch multiples**, not returns anyone realized — they assume selling at the exact all-time high. Second, they are survivors: the same list does not report the projects that went to zero. A 1,649x headline is a fact about one price on one day, not about what participating in Launchpads returns.

### Launchpool: locking BNB to farm a new token

**Launchpool** is the mechanism that runs most often now. Binance allocates a slice of a new token's supply — typically a few percent — to a farm. You deposit BNB (and often FDUSD or USDC) into pools for a fixed number of days; rewards accrue hourly, pro rata to your share of the pool; the token lists on Binance at the end.

Real, published examples of the allocation:

| Token | Launchpool allocation | Pool split |
|---|---|---|
| Space and Time (SXT) | 125,000,000 of 5,000,000,000 supply (2.5%) | BNB 85% / USDC 10% / FDUSD 5% |
| CyberConnect (CYBER) | 3,000,000 of 100,000,000 supply (3%) | — |
| Vana (VANA) | 4,800,000 of max supply (4%) | BNB 85% / FDUSD 15% |

*Allocation and pool-split figures come from the individual Binance Launchpool announcements as compiled by CoinCodex (2023–2025); Binance does not publish a consolidated table.*

Now think about what this accomplishes for everyone except the farmer. On listing day the token has thousands of holders who did not buy it, a distribution that looks decentralized, a Binance spot listing, guaranteed opening volume, and a price. The project got all of that for a few percent of a supply it printed. This is the same economics [the launchpad and points meta](/blog/trading/crypto-players/launchpads-airdrops-and-the-points-meta) covers in general form.

And what does the farmer get?

#### Worked example: the Launchpool APR, and what it hides

**The setup.** Take SXT's real parameters: 125,000,000 SXT allocated, 85% of it to the BNB pool = **106,250,000 SXT**. Suppose — and these two inputs are *illustrative*, since Binance does not publish a fixed pool size and the listing price is unknown in advance — that **20,000,000 BNB** are committed to the BNB pool over a **3-day** farm, and the token opens at **\$0.50**.

**Step 1 — your share.** You commit 100 BNB:

100 ÷ 20,000,000 = **0.0005%** of the pool.

**Step 2 — your reward.**

0.0005% × 106,250,000 SXT = **531.25 SXT**.

At \$0.50: **\$266**.

**Step 3 — the headline APR.** Your locked capital is 100 BNB × \$572.37 = **\$57,237**. The return over three days:

\$266 ÷ \$57,237 = **0.465%**, or 0.155% per day.

Annualized: (1.00155)³⁶⁵ − 1 = **about 76%**.

Seventy-six percent. That is the number that gets screenshotted.

**Step 4 — what the number hides.** You did not earn 76% a year. You earned \$266 once, for locking \$57,237 of BNB for three days. And over those three days you carried the full price risk of that BNB. A **1% move in BNB is \$572** — more than **twice** your entire reward. BNB routinely moves more than 1% in a day.

**Step 5 — the second-order case.** You could hedge: short 100 BNB of perpetual futures against the locked spot, capturing the reward with no price exposure. Now you pay the funding rate, post margin on the short, and take the risk that the perp and the spot diverge. And the perp you would use to hedge is, most likely, listed on Binance — so the hedge does not escape the gravity, it just moves you to a different desk inside the same building. [Inventory risk and delta neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality) covers how professional desks actually run this.

*The intuition: the farm pays you a fixed quantity of a token you cannot value yet, in exchange for carrying an unhedged position whose daily noise is twice the payout. The APR is arithmetically correct and analytically useless.*

![A Launchpool reward of \$266 earned by locking \$57,200 of BNB for three days, against price risk of \$572 per 1% move](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-6.webp)

### Binance Alpha: the funnel before the funnel

Since 2025, Binance has run **Binance Alpha** inside Binance Wallet — a pre-listing venue for early-stage tokens. Users accumulate **Alpha Points** through activity and spend them to enter pre-token-generation-event sales; allocation is pro rata to deposit; participants receive a non-tradable on-chain "key" as proof, then an airdrop, then the ability to trade the token inside Alpha. Strong performers *may* be promoted to Binance spot or futures, though Binance does not guarantee it.

Structurally this creates a ladder — Alpha pre-sale → Alpha trading → possible spot listing — where each rung is operated by the same firm and the next rung's existence is the reward. That is a well-designed funnel. It is also one more surface on which the exchange decides which assets get liquidity.

## The listing decision: the single largest lever

A Binance spot listing is the most valuable thing a token can receive, and the mechanics of *why* are worth being precise about.

### Why a listing moves price so much

It is not primarily reputational. It is mechanical, and it comes down to float.

#### Worked example: float, FDV, and the unlock that needs no seller

**The setup.** A token has a **total supply of 1,000,000,000**. At listing, **100,000,000 tokens (10%)** are floating; the rest are locked with the team, investors and treasury.

**Step 1 — the listing.** Suppose net demand of **\$100,000,000** arrives to buy that float. Price settles at:

\$100,000,000 ÷ 100,000,000 tokens = **\$1.00**.

**Step 2 — the two valuations.**

- Market cap = 100,000,000 × \$1.00 = **\$100 million** — the value the market actually tested.
- FDV = 1,000,000,000 × \$1.00 = **\$1 billion** — a number no buyer ever had to defend.

**Step 3 — the leverage of a small float.** A \$1 million buy order is **1% of the entire float**. In a listed equity with a \$1 billion market cap, a \$1 million order is 0.1% of the shares outstanding. The identical dollar order is **ten times more powerful** against a 10% float. This is the whole reason listing-day charts look the way they do — not enthusiasm, arithmetic.

**Step 4 — the unlock.** Now 100,000,000 more tokens vest. The float doubles to 200,000,000. If demand is unchanged at \$100 million:

\$100,000,000 ÷ 200,000,000 tokens = **\$0.50**.

The price halved and **nobody sold a single token to make it happen.** The price is a ratio; the denominator changed. Actual selling by unlocked holders is an additional, separate force on top of this.

*The intuition: a listing does not reveal what a token is worth. It reveals what a small, deliberately constrained float clears at. Every subsequent unlock re-tests that price against a supply the original buyers never had to absorb.*

![The same demand meeting a doubled float halves the price, with fully diluted valuation unchanged on paper](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-7.webp)

[How VCs move price through listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) traces the same arithmetic from the seed investor's side of the table.

### The "Binance effect" — and the evidence that it inverted

For years the received wisdom was that a Binance listing was a guaranteed repricing upward. One frequently repeated figure, an average 80% increase, traces back to an analysis of just **12 listings using 2018 data** — a sample far too small to lean on.

The recent evidence points the other way, and it is worth laying out with its provenance because none of it is a peer-reviewed study:

- An independent researcher publishing as "Flow" analyzed 31 tokens listed on Binance in the preceding six months and found only **5 (16%) were trading above their listing price**, with an average drawdown greater than 18% at six months (FXStreet, 20 May 2024). This is one researcher's dataset, not an institutional study.
- BeInCrypto's review of 2024 Binance listings reported that **29 of 30 tokens** showed significant losses, ranging from −23% to worse than −95% (BeInCrypto, 2024 listings analysis).
- A broader cross-exchange analysis of **389 tokens listed across six major exchanges in 2024** (data gathered 2–4 February 2025) found an average **54% surge at launch** followed by **89% of tokens declining**, averaging **−52%**.
- For 2025, BeInCrypto reported only **11.1%** of Binance-listed tokens posted a positive return (BeInCrypto, 2025 listings analysis).

Kaiko Research's framing is the most useful one: it treats the "Binance effect" primarily as a **liquidity-concentration** phenomenon rather than a price-prediction one. When Binance listed USD1 on 22 May 2025, Kaiko observed PancakeSwap V3 trade counts rise from about 28,000 to 283,000 within four days (Kaiko Research, May 2025). The listing did not necessarily make the asset go up; it made the asset *tradable at scale*, which is a different and more durable effect.

That reframing is the correct one. **A listing is a liquidity event, not a quality signal.** It reliably produces volume. It does not reliably produce returns, and since 2024 it has mostly produced the opposite.

### The listing-fee controversy

Whether Binance charges for listings has been publicly contested. Everything in this subsection is **reported or alleged**, with denials noted, and none of it is an established fact:

- In November 2024, Moonrock Capital's chief executive Simon Dedic **alleged** that Binance had requested 15% of a tier-one project's token supply — a figure he put at \$50–100 million — for a listing (Bitcoinist, November 2024). Binance co-founder **Yi He denied** the claim, called it FUD, and said Binance's listing rules are transparent; CZ responded publicly that Bitcoin never paid a listing fee.
- Tron's Justin Sun and Sonic's Andre Cronje have both **stated** that Binance charged them nothing, contrasting it with much larger sums they said Coinbase requested — claims Coinbase has denied, with its chief executive maintaining that Coinbase listings are free.
- In March 2026, Limitless Labs founder CJ Hetherington **accused** Binance of requesting roughly 8% of a token's airdrop plus a \$250,000 fiat deposit (reported March 2026). Binance called the accusation **"false and defamatory,"** said it does not profit from listings, and threatened legal action.
- Binance's standing public position, per Yi He, is that projects failing its review are not listed regardless of what they offer.

**No regulator has made a public finding on Binance listing fees as of 27 July 2026.** These are competing public statements, and the honest summary is that outsiders cannot verify the answer.

## The other lever: the Monitoring Tag and delisting

The power to list implies the power to unlist, and Binance formalized the intermediate step.

On **26 July 2023**, Binance introduced **Seed Tags** and **Monitoring Tags**, replacing the earlier "Innovation Zone." A Seed Tag marks an early-stage, innovative and volatile project. A **Monitoring Tag** marks an already-listed token showing notably higher volatility or risk — the initial batch of 26 tokens included FTT, TORN and MULTI. To trade a tagged token, users must pass a quiz **every 90 days** and re-accept the terms of use, and risk banners appear throughout the interface.

Binance publishes its review criteria: team commitment, development activity, trading volume, network stability, public communication, and evidence of unethical conduct. Tags are added and removed on review — in April 2026, for instance, Binance added seven tokens including FARM to the Monitoring Tag and lifted the Seed Tag from XAUT.

Binance also runs a **"Vote to Delist"** process alongside "Vote to List." Users with a verified account holding at least 0.01 BNB throughout the voting period get up to five votes, one per project. Binance is explicit that the community vote is *not* the sole determinant; the final decision comes from internal review.

The price effects reported around these events are substantial, though the figures below come from aggregator roundups rather than primary market data:

- **BLZ** fell about **45%** in 24 hours after a delisting announcement (25 December 2024).
- **BADGER** fell about **74%** over three months after being included in the delisting round (April 2025).
- **LINA**'s market capitalization reportedly fell about **65%** after its delisting announcement (early 2025).

On **16 April 2025**, Binance delisted 14 pairs in a single batch, including BADGER, BAL, BETA, CREAM, CTXC, ELF, FIRO, HARD, NULS, PROS, SNT, TROY, UFT and VIDT.

Mechanically this is the float argument in reverse. Delisting from the venue that clears a third of global spot volume does not merely remove a listing; it removes most of the liquidity that made the token's price meaningful. The remaining venues cannot absorb the exiting flow at anything like the old price. The tag is a genuine, well-intentioned consumer-protection tool **and** an announcement that reprices an asset. Both things are true at once.

## The venture arm: Binance Labs, now YZi Labs

Binance Labs was Binance's venture and incubation arm. In **January 2025** — YZi Labs' own announcement is dated 27 January 2025 — it was rebranded **YZi Labs** and repositioned as the family office of CZ and co-founder Yi He rather than the exchange's dedicated fund, with an expanded mandate covering AI and biotechnology. Ella Zhang, a Binance Labs co-founder, was named to lead it from Hong Kong.

Reported figures put the portfolio at **over \$10 billion across 250-plus projects** (single-source roundup, 2025), including Sky Mavis, LayerZero, Aptos Labs and Polygon. That figure comes from a single-source roundup and is not independently verified.

The structural point is simple and does not require anyone to have misbehaved. **A fund closely associated with the exchange invests in tokens, and those tokens can subsequently be listed on that exchange.** In equity markets those two functions live in different companies for reasons everyone understands.

The most-cited data point on this comes from BeInCrypto's 2024 analysis, which noted that a set of Binance Labs-backed tokens listed on Binance — AI (Sleepless AI), MANTA, AXL, ENA, REZ, BB and LISTA — were subsequently **down between 44% and 90%** (BeInCrypto, 2024). BeInCrypto's interpretation, that listings provided exit liquidity for venture backers, is **an attributed opinion, not an established fact**, and the same period saw nearly all new listings fall regardless of who backed them.

What can be said without inference is narrower and still significant: an investor whose fund holds a token, and whose affiliated exchange decides whether that token gets the single most liquidity-generative event available to it, holds two positions that a traditional market structure would not permit in one place. [Cui bono — the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) maps this class of overlap across the industry.

## The quote asset: BUSD, FDUSD and the stablecoin dependency

Almost nothing on Binance is priced in dollars. It is priced in **stablecoins** — tokens that aim to hold a value of one dollar, issued by a company that claims to hold real dollars against them. Which stablecoin is the quote asset is therefore an enormous decision, and Binance has had to make it twice under pressure.

**BUSD** was a Binance-branded dollar stablecoin issued by Paxos. It peaked at a market capitalization of about **\$23.36 billion on 12 November 2022**, the day after FTX's bankruptcy filing, as capital fled to perceived safety. Three months later it was over: the **SEC issued Paxos a Wells notice on 3 February 2023** alleging BUSD was an unregistered security, and on **13 February 2023 the New York Department of Financial Services ordered Paxos to stop minting new BUSD**, effective 21 February. A stablecoin that cannot be minted can only shrink.

Binance's largest quote asset had been legislated out of existence in a fortnight, and the replacement arrived quickly. **FDUSD**, issued by Hong Kong-based First Digital, launched on Binance on **26 July 2023**, and from **4 August 2023** Binance ran zero-fee promotions on BTC/FDUSD and ETH/FDUSD — a direct subsidy to move the order book onto the new quote asset.

That dependency was tested on **2 April 2025**, when Justin Sun publicly alleged that First Digital Trust was "effectively insolvent" and unable to meet redemptions. FDUSD depegged the same day to roughly **\$0.87–0.91**. First Digital Trust **denied the allegations**, called them "completely false," and filed a defamation claim (CoinDesk, 9 April 2025). The peg recovered.

Sit with what that day looked like from inside the order book. A large share of the exchange's pairs were quoted in an asset that briefly stopped being worth a dollar. Every price on every one of those pairs was, for a few hours, denominated in a unit of uncertain value — while the assets themselves had not changed at all.

Binance has since diversified: on **11 December 2024** Circle and Binance announced a strategic partnership to expand USDC availability on the platform and adopt USDC in Binance's corporate treasury (Circle press release, 11 December 2024). And on **12 March 2025**, Abu Dhabi state-backed fund MGX announced a **\$2 billion investment in Binance** — the first institutional investment in the company's history — which was confirmed at Token2049 Dubai on 1 May 2025 to have been **settled in USD1**, the stablecoin of World Liberty Financial. That transaction drew political scrutiny in the United States, including letters from members of the Senate Banking Committee, reported alongside coverage of the pardon discussed below. The equity stake acquired has not been publicly disclosed.

For the general mechanics of how stablecoins hold their peg and what breaks them, [the stablecoin deep-dive](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar) is the companion piece.

## The public record

This section states settled matters and court records as such. **Nothing here alleges conduct beyond the public record**, and where allegations were contested or dismissed, that is stated.

![Timeline of the public record from the 2017 BNB ICO through the November 2023 resolution to the October 2025 pardon](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-9.webp)

**21 November 2023 — the resolution.** The US Department of Justice announced that Binance had pleaded guilty and agreed to a resolution totaling **\$4,316,126,163**, comprising **\$2,510,650,588 in forfeiture** and a **\$1,805,475,575 criminal fine** (DOJ, 21 November 2023). Binance pleaded guilty to failing to maintain an effective anti-money-laundering program under the Bank Secrecy Act, to conducting an unlicensed money transmitting business, and to violating the International Emergency Economic Powers Act. The resolution included a **three-year independent compliance monitor**.

The same day:

- **FinCEN** assessed a **\$3.4 billion** civil penalty, with \$150 million suspended pending compliance, and imposed a **five-year monitorship** (FinCEN, 21 November 2023).
- **OFAC** assessed **\$968,618,825** for **1,667,153 apparent sanctions violations** between August 2017 and October 2022, involving Iran, North Korea, Syria and Crimea (OFAC enforcement release, 21 November 2023).
- The **CFTC** settlement required **\$2.7 billion** from Binance (\$1.35 billion disgorgement plus \$1.35 billion penalty) and **\$150 million** from CZ personally (CFTC consent order, entered December 2023).

A note on arithmetic that many summaries get wrong: these figures are **not simply additive**. The commonly quoted "\$4.3 billion" is the DOJ figure, and the Treasury and CFTC amounts run partly concurrent with or are credited against the DOJ forfeiture. Adding them to about \$11 billion, as some coverage did, overstates the total.

**Changpeng Zhao.** CZ pleaded guilty to a single Bank Secrecy Act count, was fined **\$50 million** personally by DOJ (DOJ, 2023) plus the \$150 million CFTC amount,, and stepped down as chief executive the same day; **Richard Teng**, previously head of global markets, was named CEO. On **30 April 2024**, Judge Richard Jones in Seattle sentenced him to **four months** — the Justice Department had sought 36 — and he began serving on 31 May 2024. On **23 October 2025**, President Trump granted CZ a full and unconditional pardon.

**The SEC case.** The SEC sued Binance, its US affiliates and CZ on **5 June 2023**, alleging operation of an unregistered exchange, broker and clearing agency, commingling of customer funds, misleading investors, and inflated trading volumes. On **29 May 2025** the case was **dismissed with prejudice** by joint stipulation (SEC Litigation Release LR-26316). A dismissal with prejudice means the claims cannot be refiled; it is not a finding that they were true.

**Elsewhere.** The Netherlands' central bank fined Binance €3.3 million for operating without registration (De Nederlandsche Bank, 18 July 2022). France escalated a preliminary probe into a formal JUNALCO judicial investigation covering 2019–2024 conduct, reported on 28 January 2025 — **an open investigation, not a finding**. In Nigeria, Binance compliance executive Tigran Gambaryan was detained in February 2024 and released in October 2024 with charges dropped. Meanwhile Binance accumulated licenses: a full VASP license from Dubai's VARA in April 2024, a full license from Kazakhstan's AFSA on 6 September 2024, and three ADGM licenses in Abu Dhabi effective 5 January 2026.

#### Worked example: putting \$4.3 billion in proportion

A number this large is meaningless without a denominator, so let's compute three.

**Denominator 1 — one month of spot volume.** Binance's January 2026 spot volume was \$407 billion.

\$4,316,126,163 ÷ \$407,000,000,000 = **1.06%** of a single month's spot notional.

**Denominator 2 — our conservative fee estimate.** Earlier we built a deliberately low estimate of \$367 million a month in trading fees.

\$4,316,126,163 ÷ \$367,000,000 = **11.8 months** of trading-fee revenue.

**Denominator 3 — the press revenue estimate.** Forbes's estimate was \$16–17 billion a year; take \$16.5 billion.

\$4,316,126,163 ÷ \$16,500,000,000 = **0.26 years ≈ 3.1 months** of estimated total revenue.

So the largest crypto enforcement resolution in history equals somewhere between **three months and one year** of the company's revenue, depending on which estimate you trust — plus a criminal conviction, two monitorships, and a founder who served a prison sentence.

*The intuition: whether a penalty is "large" is a question about the denominator, and reasonable people reading the same public record land in different places. The non-monetary terms — the guilty plea, the monitors, the change of chief executive — are the parts that actually constrain behavior going forward, and they do not have a dollar figure.*

## Why the largest venue's gravity distorts price discovery

Now we can state the thesis precisely.

**Price discovery** is the process by which many independent participants, each acting on their own information, produce a price that aggregates what they collectively know. It works when the participants are actually independent. It degrades when they are not.

Here is the problem, and it does not require anyone to cheat.

![Functions carried out by separate firms in an equity market — matching, custody, listing, investing, issuing the quote asset, settlement, risk labelling — sit inside one firm on Binance](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-8.webp)

For a token that came through the full funnel, the following can all be true simultaneously:

1. Its **liquidity** is concentrated on Binance, because a third of global spot volume is.
2. Its **listing** was Binance's decision.
3. Its **initial holders** were manufactured by Binance's Launchpool, and they hold it because they locked Binance's token.
4. Its **quote asset** is a stablecoin Binance partnered to promote and subsidized with zero-fee trading.
5. Its **settlement layer**, if it is a BNB Chain token, is a chain the same founder created.
6. An early **investor** was the fund formerly called Binance Labs.
7. Its **risk label** and its continued listing are Binance's decisions.

Every one of those relationships is disclosed and legal. And yet the price that emerges is not an independent aggregation of many views. It is substantially the output of one firm's decisions about liquidity provision, timing and access.

Three concrete distortions follow.

**Concentration converts an operational event into a price event.** When one venue holds a third of liquidity, its maintenance windows, its API outages, its margin-parameter changes and its collateral-haircut updates all move price. A risk-parameter change is an internal operations decision; on a venue this size it is a market-wide event. This is a general property of concentrated markets, not a criticism of any particular decision.

**The reference price becomes partly endogenous.** Index providers, lending protocols and other exchanges reference the deepest book, which is often Binance's. So a liquidation cascade on Binance propagates into oracle prices, which trigger liquidations elsewhere, which trade back onto Binance. The largest book stops being *a* measurement of the price and starts being *the* price — including for participants who never touch it. [How crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) traces that transmission in detail.

**Announcement timing becomes a tradable asset in itself.** Because listings, tags, delistings and burn dates all move price and all originate inside one organization, knowledge of the calendar has value independent of any view on the assets. Binance publishes its announcements and has tightened market-maker disclosure rules over time. The structural point is that the *category* of information exists at all, at a scale where no equivalent exists in equity markets.

The honest summary: **you do not need bad actors to get distorted prices. You only need one participant large enough that its ordinary business decisions are indistinguishable from market events.** That is the gravity in the title, and it is why [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues) and [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) treat venues as participants rather than infrastructure.

## Common misconceptions

**"A Binance listing means the token is good."** A listing is a liquidity event, not a quality certificate. The 2024–2025 data points the opposite way: BeInCrypto reported 29 of 30 tokens listed in 2024 finished with significant losses, and only 11.1% of 2025 listings posted a positive return (BeInCrypto, 2024 and 2025 analyses). What a listing reliably delivers is *tradability at scale* — which is valuable to the project and neutral for the buyer.

**"The BNB burn is like a stock buyback."** A buyback converts a company's actual profits into a shareholder claim, and shareholders own the residual cash flows. The BNB Auto-Burn is a function of BNB Smart Chain's block count and BNB's price — **Binance's profits are not an input**. BNB holders own no claim on Binance's earnings. The scarcity is real; the analogy to equity is not.

**"Proof of reserves means Binance is solvent."** Proof of reserves shows *assets* at a snapshot. Solvency is assets *minus liabilities*. Binance's system uses a Merkle tree and, since February 2023, zk-SNARK verification, which proves your balance was included without revealing it — a genuine cryptographic improvement over trusting a screenshot. It is still not an audit. Mazars, which produced Binance's early reports, **suspended all crypto proof-of-reserves work on 16 December 2022**, and no Big Four firm had taken on the work at that time. Reading the assets page and concluding "solvent" skips the entire liability side.

**"Volume equals liquidity."** Reported volume tells you how many trades printed. Liquidity tells you how much you can actually sell without moving the price. They diverge badly on thin, newly listed tokens, where enormous volume can coexist with a book that a \$500,000 order would walk through. [Wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) covers how to tell the difference.

**"The \$4.3 billion penalty was really \$11 billion."** Adding the DOJ, FinCEN, OFAC and CFTC figures together double-counts. The Treasury and CFTC amounts run partly concurrent with or credited against the DOJ forfeiture. The DOJ's announced figure — \$4,316,126,163 — is the right headline number.

**"CZ's pardon means the conviction was overturned."** A pardon is executive clemency. It does not vacate a guilty plea or erase the record of the conduct that led to it. Binance's own corporate guilty plea and the associated obligations are separate from any relief granted to an individual.

**"If I hold BNB I'm exposed to Binance's growth."** You are exposed to *demand for BNB*, which correlates with Binance's activity but confers no ownership. In the nine months to 27 July 2026, BNB fell 58% from its high while Binance's spot market share *rose* from about 34% to 38% (CoinMarketCap and CCData, 2026). The token and the business can move in opposite directions because there is no contractual link between them.

## How it shows up in real markets

**The 2023 share collapse — and why it barely mattered.** Binance ran zero-fee BTC trading through much of 2022, and its spot share hit 55.2% in January 2023. When the promotion ended and regulatory pressure mounted, share fell to 30.1% by December 2023 and to about 27% by October 2024. On any normal reading that is a catastrophic loss of position. And yet by March 2026 it was back to 35.4% (CCData). The lesson for anyone modeling competitive dynamics in this market: **liquidity is sticky in a way that fees are not.** Traders route to the deepest book because slippage costs more than fees, so a venue can lose a price war and keep the network effect.

**BUSD, February 2023 — a quote asset legislated away.** The SEC's Wells notice to Paxos on 3 February 2023 and NYDFS's mint-stop order on 13 February 2023 ended a stablecoin with a \$23.36 billion peak market capitalization. Note what was and was not at risk: BUSD did not break its peg or fail to redeem. A regulator simply forbade the creation of new units, and a stablecoin that cannot grow cannot serve as a quote asset. Binance had to migrate its order books to FDUSD within months — and subsidized the migration with zero-fee promotions from 4 August 2023. **Quote-asset risk is a distinct category of risk**, and it is regulatory before it is financial.

**FDUSD, 2 April 2025 — the dependency tested.** Justin Sun's public allegation that First Digital Trust was "effectively insolvent" — which First Digital **denied as "completely false"** and answered with a defamation claim — knocked FDUSD to roughly \$0.87–0.91 within a day (reported by Blockworks and CoinDesk, April 2025). For those hours, prices on a large share of Binance's pairs were quoted in a unit that was not worth a dollar. The peg recovered and no redemption failure was demonstrated. The episode is instructive precisely *because* nothing broke: the quote asset is a single point of failure whose risk is invisible until the day it is not.

**The BSC Token Hub hack, October 2022.** An exploit of the bridge between BNB Beacon Chain and BNB Smart Chain allowed an attacker to forge roughly **2 million BNB, about \$570 million at the time** (CNBC, October 2022). Validators halted the chain to contain it, freezing an estimated \$7 million. Consider the two facts side by side: a chain that could be stopped by its validators, and a token whose supply had just been inflated by 1% of its total through a bug. The halt was almost certainly the right operational call. It also demonstrated that BNB Chain's decentralization is a design choice with a coordination override — relevant if you are pricing the chain's TVL, which DefiLlama put at **\$4.878 billion**, second only to Ethereum (DefiLlama, as of 27 July 2026).

**October 2025 to July 2026 — the burn against the tape.** BNB set an all-time high of \$1,370.55 (CoinMarketCap, 13 October 2025). Over the following nine months Binance executed three quarterly burns removing 4,572,251.35 BNB — 3.3% of supply — and the price finished at \$572.37 on 27 July 2026, down 58%. This is as clean a natural experiment as this market offers: a mechanical, on-chain-verifiable, unambiguously deflationary supply program, running at full speed, into a 58% drawdown. **Supply mechanics are real and they are second-order.**

**The 2025 delisting rounds.** On 16 April 2025 Binance removed 14 pairs in one batch. Around such announcements, BLZ reportedly fell about 45% in 24 hours (25 December 2024) and BADGER about 74% over three months. These figures come from aggregator roundups rather than primary market data, so treat the magnitudes as indicative. The mechanism, though, is not in doubt and is the float argument in reverse: removing the venue that clears a third of global volume removes most of the liquidity that made the price meaningful, and the remaining venues cannot absorb the exit at the old price.

**Alameda and the counterparty lesson.** The reason any of this matters practically is that concentration and opacity compound. When one firm is simultaneously a venue's largest counterparty, its liquidity provider and an affiliate of its owner, ordinary business decisions stop being separable from market manipulation risk — not because anyone necessarily intends harm, but because nobody outside can distinguish the two. [Alameda Research remains the cautionary tale](/blog/trading/crypto-players/alameda-research-the-cautionary-tale) for exactly this structure, and it is the reason the separations described in this article exist in older markets at all.

## How to read Binance signals

Here is the practical method. Every Binance announcement is a supply or liquidity event, and each kind has one question that matters more than the rest.

![Every Binance announcement is a supply or liquidity event: the question is always whose supply, against what float](/imgs/blogs/binance-the-everything-exchange-and-its-gravity-10.webp)

**A spot listing announcement.** Ask: *what percentage of total supply is actually floating?* Then look up the vesting schedule and find the next unlock date. A listing with a 10% float is a listing where the price is set by a tenth of the token and every future unlock re-tests it against buyers who never had to absorb it. Also check whether a Binance-affiliated fund appears on the cap table — not because that implies wrongdoing, but because it tells you who has an exit that the float will have to clear. [Following the money through a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) is the method for that.

**A Launchpool announcement.** Ask: *what percentage of total supply is being handed out, and over how many days?* Two to four percent of supply, distributed to thousands of people whose cost basis is zero, arriving in wallets on the same day the token starts trading, is a supply event with a known timestamp. Also compute the honest yield rather than the headline APR — the reward against the price risk of the BNB you must lock, not the annualized rate.

**A Monitoring Tag.** Ask: *is a delist vote the next step?* The tag is a published risk signal from the venue that holds most of the token's liquidity, and it carries a real procedural consequence — the 90-day quiz requirement adds friction to every new buyer.

**A quarterly burn.** Ask: *how large is the supply change compared with the demand change?* Compute the percentage of supply removed and compare it against the price move over the same window. In the most recent full example, the answer was 3.3% against 58%. Report the coin count, not the dollar value, since the dollar value moves mostly for a different reason.

**The proof-of-reserves page.** Read it as *an assets-only snapshot on a date you chose*, not an audit. The zk-SNARK verification genuinely proves your balance was included in the tree. It proves nothing about liabilities.

**Any exchange-wide parameter change.** Margin requirements, collateral haircuts, tier thresholds, funding-rate caps: on a venue with a third of the market, these are macro events for the assets they touch.

One more discipline that costs nothing: **write down the date next to every number you use.** Everything in this article was true as of 27 July 2026 and some of it will not be by the time you read it. Market shares move quarterly. Prices move hourly. The burn count increments every three months. A figure without a date is not a fact, it is a memory.

## When this matters to you

If you never trade a token, the honest answer is that this affects you mostly through the second-order channel: Binance's book is a reference price for lending protocols, index products and other exchanges, so its liquidity conditions propagate to instruments you might hold without ever visiting the site.

If you do trade, three things change once you internalize this article. You stop reading a listing as an endorsement and start reading it as a liquidity event with a float behind it. You stop reading an APR as a return and start pricing the risk you had to carry to earn it. And you stop treating the venue as neutral infrastructure — because a firm that clears a third of the market, issues the token you pay fees in, runs the chain the asset settles on, and decides whether it stays listed is a participant with the largest position in the room.

None of that means the largest venue is acting badly. The public record contains what it contains — a corporate guilty plea, a resolution of \$4,316,126,163 announced on 21 November 2023, a four-month sentence for its founder, a subsequent pardon, a dismissed SEC case, and a growing list of jurisdictions that have granted it licenses. Beyond that record, this article claims nothing.

What it does claim is structural: **concentration is a mechanism, not a moral failing, and it changes prices whether or not anyone intends it to.** That is the thing worth carrying to the next announcement.

For the next steps, [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) explains the transmission from order flow to price, [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) covers the supply schedule you now know to look up, and [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues) generalizes this profile to the rest of the industry.

## Sources & further reading

**Primary legal and regulatory sources**

- US Department of Justice, "Binance and CEO Plead Guilty to Federal Charges in \$4B Resolution," 21 November 2023 — the \$4,316,126,163 total, the forfeiture and fine breakdown, the guilty-plea counts and the three-year monitorship.
- FinCEN, settlement announcement, 21 November 2023 — the \$3.4 billion civil penalty and the five-year monitorship.
- OFAC, enforcement release, 21 November 2023 — \$968,618,825 for 1,667,153 apparent violations, August 2017 to October 2022.
- CFTC, complaint filed 27 March 2023 and consent order entered December 2023 — the \$2.7 billion settlement and CZ's \$150 million.
- SEC, complaint in *SEC v. Binance Holdings Ltd.*, filed 5 June 2023; Litigation Release LR-26316, dismissal with prejudice, 29 May 2025.
- New York Department of Financial Services, consumer alert on Paxos and Binance, 13 February 2023 — the BUSD mint-stop order.
- Executive pardon of Changpeng Zhao, 23 October 2025.

**Binance's own documentation**

- Binance fee schedule (`binance.com/en/fee/trading`), accessed 27 July 2026 — the 0.100%/0.100% spot standard, the 25% BNB discount, and the VIP 1 through VIP 9 ladder.
- Binance Support, "Introducing BNB Auto-Burn," 22 December 2021 — the formula's inputs and the 100 million floor.
- BNB Chain Blog, "Introducing BEP-95," 22 October 2021 — the real-time gas-fee burn.
- Binance Support, "Introducing Seed Tags & Monitoring Tags," 26 July 2023 — criteria, the 90-day quiz requirement and the initial tagged batch.
- Binance Square, 33rd quarterly burn announcement, 27 October 2025, and 36th quarterly burn, 15 July 2026 — the burn quantities and post-burn supply figures.
- Binance ninth-anniversary release via PR Newswire, 14 July 2026 — 323 million-plus registered users and \$156 trillion cumulative volume.
- Binance proof-of-reserves page, accessed 27 July 2026 — the Merkle-tree and zk-SNARK methodology.

**Market data and research**

- CCData Exchange Review (January 2026 report dated 16 February 2026; May 2026 edition) — spot share, monthly volumes and derivatives share.
- CoinDesk, "Binance's Market Share of Crypto Trading Tumbled to 30% in 2023," 11 December 2023 — the 55.2% to 30.1% decline.
- Bloomberg, "Binance's Crypto Market Share Drops to Lowest Level in Four Years," 3 October 2024.
- Kaiko Research, "The Binance Effect" — the liquidity-concentration framing and the USD1 listing data from May 2025.
- CoinGecko Research, Binance Launchpad project returns — the AXS, MATIC and SAND peak multiples.
- DefiLlama, BNB Chain TVL, accessed 27 July 2026.
- CoinMarketCap, BNB price, supply and all-time-high data, accessed 27 July 2026.

**Journalism and analysis (attributed claims)**

- Forbes, 5 October 2023 — the disputed account of the 2017 BNB ICO distribution.
- Forbes, 10 March 2026 — the \$16–17 billion revenue estimate and ~\$100 billion valuation estimate.
- FXStreet, 20 May 2024 — coverage of the independent researcher "Flow" analysis of 31 Binance listings.
- BeInCrypto, 2024 and 2025 Binance listing performance analyses.
- Bitcoinist and CryptoBriefing, November 2024 — the Moonrock Capital listing-fee allegation and Binance's denial.
- CoinDesk, 25 March 2026 — Binance's market-maker disclosure rules and its response to the Limitless Labs allegation.
- CoinDesk, 9 April 2025 — First Digital Trust's denial and defamation claim following the FDUSD depeg.
- Circle press release, 11 December 2024 — the Binance USDC partnership.
- CoinDesk and Bloomberg, 12 March and 1 May 2025 — the MGX \$2 billion investment and its settlement in USD1.
- CoinDesk, 16 December 2022 — Mazars suspending crypto proof-of-reserves work.
- CNBC and Halborn, October 2022 — the BSC Token Hub exploit and the chain halt.
- CryptoSlate and YZi Labs, 27 January 2025 — the Binance Labs rebrand.

Figures attributed to aggregators — the delisting price moves, the Launchpool allocation splits, the YZi Labs portfolio size — are indicative rather than primary and are labeled as such in the text where they appear.
