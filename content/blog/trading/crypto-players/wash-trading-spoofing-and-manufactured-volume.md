---
title: "Wash Trading, Spoofing, and Manufactured Volume: The Dark Side of Liquidity"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "A build-from-zero guide to how fake volume and fake liquidity are manufactured in crypto — wash trading, spoofing, layering, and painting the tape — why exchanges and projects tolerate it, and the statistical and on-chain fingerprints researchers use to detect it."
tags: ["crypto", "wash-trading", "spoofing", "market-manipulation", "trading-volume", "liquidity", "order-book", "market-microstructure", "on-chain-analysis", "crypto-players", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 38
---

> [!important]
> **TL;DR** — The two numbers a token page shows you — its **volume** and its **order book** — are the two most easily faked signals in crypto. Wash trading manufactures fake *volume*; spoofing manufactures a fake *price*. Neither requires a real buyer or a real seller.
>
> - A **wash trade** is a trade with yourself. One entity controlling two accounts can print unlimited "volume" while its net position and the real order book never change. Volume can be typed into existence; liquidity cannot.
> - **Spoofing** and **layering** post large orders the trader never intends to fill, to nudge the price, then cancel them. The canonical case — the 2010 "flash crash" trader — used sell orders that were, at their peak, roughly **20–29% of the visible sell side** (CFTC) and were never executed.
> - Exchanges and projects tolerate manufactured volume because volume drives **rankings and listings** — fake activity buys visibility, which buys real buyers, who become the exit liquidity.
> - Landmark studies put numbers on it: **Bitwise told the SEC in 2019 that ~95%** of reported bitcoin spot volume looked non-economic, and a peer-reviewed academic study (Cong, Li, Tang & Yang) estimated wash trading **averaged over 70%** of reported volume on unregulated exchanges. Both are *research estimates*, not universal facts.
> - Detection is statistical and on-chain: **volume-vs-liquidity divergence**, **Benford's-law digit tests**, round-number clustering, and **self-funded wallet loops**. The one habit to keep: divide reported volume by real order-book depth before you trust it.

Open any token's page and two numbers jump out: a big **24-hour volume** figure and a live **order book** with a last price. Both feel like measurements — like a thermometer reading. They are not. They are the two easiest things in all of crypto to fabricate, and a surprising share of the market's headline activity is, by multiple independent estimates, exactly that: fabricated.

This is not a conspiracy claim. It is plumbing. Once you understand what a "trade" actually is and how thin most order books really are, you can see precisely how a single actor with two accounts can print a \$10-million "volume" number that corresponds to zero real economic activity — and how another actor can post a giant buy order it has no intention of filling, watch the price twitch upward, sell into that twitch, and cancel the order before anyone touches it. Then you can learn to *see* the fakes, because they leave fingerprints.

The diagram below is the mental model for the whole article. Everything downstream is a drill-down into one of its four leaves.

![A branching map: the token page splits into a 24h volume number and an order book, and each splits into a real version and a manufactured version — wash trades fake volume, spoof orders fake the book.](/imgs/blogs/wash-trading-spoofing-and-manufactured-volume-1.webp)

The screen hands you two signals. The volume number can be real economic trades or **wash trades**; the order book can be real resting orders or **spoof orders**. This post is about the two right-hand branches — how they are built, why they persist, and how they are caught. It is strictly analytical: the goal is to make you a better *reader* of a market, not a better manipulator of one. Manipulating markets this way is illegal in regulated venues and fraudulent everywhere; every mechanic here is presented so you can detect it from the other side of the trade.

## Foundations: volume, liquidity, and the difference that makes faking possible

Before any of the tricks make sense, four ideas have to be rock-solid. If you have read [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move), some of this is review; if not, start here.

### Volume vs liquidity — the one distinction that matters

**Volume** is a count of what *has already traded* over some window — "\$50 million traded in the last 24 hours." It is a number about the past, and it is reported by the venue.

**Liquidity** is the market's *capacity to absorb a trade right now* without the price moving much — how much you could actually buy or sell this second before you move the price 1% or 2%. It lives in the **order book**: the stack of resting buy and sell orders waiting to be filled.

Here is the entire vulnerability in one sentence: **volume is a claim; liquidity is a commitment.** To add real liquidity you must post real capital that anyone can trade against and take from you. To add "volume," you need only... trade. Even with yourself. The reported volume number is generated by the venue from its own trade log, and a trade log does not know or care whether the two sides were strangers or the same person wearing two hats.

> Volume tells you what a venue *says* happened. Liquidity tells you what you could *actually do* if you showed up with money. Manipulators love the first number precisely because it costs almost nothing to inflate.

### What a trade actually is — and what a "wash trade" is

A **trade** is a matched pair of orders: a buyer willing to pay price *P* meets a seller willing to accept *P*, ownership of the asset moves one way, cash moves the other, and the venue records one print of size *Q* at price *P*. That print adds *Q × P* to the day's volume.

A **wash trade** is a trade where the buyer and the seller are *the same beneficial owner* — the same person, fund, or bot, often using two accounts or two wallets. (The name comes from the trade "washing out": you end where you started.) A **self-trade** is the narrowest version: the same account's own buy order crossing its own sell order. Because ownership does not really change, a wash trade transfers nothing of substance — but the venue still records a print and still adds *Q × P* to volume. That gap between "recorded" and "transferred" is the raw material of every fake-volume scheme.

Your **net position** is simply what you own after everything nets out. The defining feature of wash trading is that you can run the volume counter to any number you like while your net position stays flat. You are moving money from your left pocket to your right pocket and back, and calling the motion "activity."

### The order book, spread, and depth — in one minute

The **order book** has two sides. **Bids** are resting buy orders (the best bid is the highest price someone will pay). **Asks** (or offers) are resting sell orders (the best ask is the lowest price someone will accept). The gap between best bid and best ask is the **spread**. **Depth** is how much size is resting near the current price — often measured as the dollar value of orders within ±2% of the mid-price. A **thin** book has little depth: a small order eats through several price levels and moves the price a lot. **Resting liquidity** is just another name for those waiting orders.

Two terms you will meet later: a **market maker (MM)** is a firm that continuously posts both bids and asks to provide liquidity and earn the spread — the subject of [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does). And **slippage** is the difference between the price you expected and the price you actually got because your order walked up or down the book. Thin books produce big slippage, which is exactly why thin books are so easy to push around.

### Where the volume number comes from

One more thing has to be clear before the tricks make sense: **who produces the volume number, and how.** On a regulated stock exchange, trades are reported to a consolidated tape under legal supervision, and a listed company's activity is audited. In crypto, the 24-hour volume you see on an aggregator is, in most cases, **self-reported by the exchange** — the venue publishes its own trade log or exposes an API, and aggregators sum it up. There is no independent auditor standing between the exchange's claim and your screen. A regulated venue has strong reasons not to lie, because it can be fined or shut down; an unregulated offshore venue competing for a league-table ranking has strong reasons to inflate. The number is only as trustworthy as the incentives of whoever generated it — and for much of the market, those incentives point the wrong way. This is the gap every technique in this article lives inside: **a self-reported metric, with a fee-paying beneficiary, and no auditor.**

#### Worked example: how self-trades print volume from nothing

Suppose you control two accounts, A and B, on the same exchange. Account A holds 100 tokens ("TKN"); account B holds \$100 of cash. You place a sell order on A and a matching buy on B at \$1.00.

![One owner controls Account A and Account B; the token is sold back and forth, each round-trip printing \$200 of volume, while net position stays zero and the only real cost is fees.](/imgs/blogs/wash-trading-spoofing-and-manufactured-volume-2.webp)

- **Round-trip 1:** A sells 100 TKN to B for \$100 (one \$100 print), then B sells the 100 TKN back to A for \$100 (a second \$100 print). Two trades, **\$200 of reported volume.**
- You repeat this **50 times.** Total reported 24-hour volume: 50 × \$200 = **\$10,000.**
- Your net position after all 100 trades: account A still holds 100 TKN, account B still holds \$100 cash. **Net change: zero.** No outsider ever traded. The order book underneath is exactly as thin as when you started.

The only real cost is **fees.** If the venue charges 0.1% per side, each \$100 trade costs \$0.10, so 100 trades cost about \$10 to print \$10,000 of volume — a 0.1% "tax" to manufacture a headline. And on many venues that tax is smaller than it looks: high-volume accounts get **fee rebates**, and on a venue the manipulator effectively controls, the fee can round to zero. The intuition to carry forward: **fake volume is cheap, and the thing it fakes — the impression of a busy, liquid market — is worth far more than the fee.**

## 1. Wash trading: manufacturing volume at scale

The two-account toy above is the whole idea; real operations just industrialize it. Instead of one person and 100 trades, it is a bot cycling dozens of wallets, randomizing sizes and timing so the pattern looks organic, running 24/7. The output is the same: a volume number decoupled from any real transfer of ownership.

Two flavors are worth separating:

- **Self-trading / circular wash trading** — the same owner on both sides, either directly (an account crossing its own order) or through a ring of wallets that pass the token around a loop and back to the start.
- **Collusive / reciprocal wash trading** — two *different* parties agree to trade back and forth with each other. Neither ends up with a net position, but because the accounts belong to different legal entities it is harder to prove they are coordinating. This is the version that shows up in the most contested real-world allegations.

At industrial scale, a wash operation is a piece of software. It controls dozens or hundreds of wallets, funds them from a common treasury through intermediary hops to blur the link, and runs a matching engine that crosses its own orders thousands of times a day — randomizing sizes, jittering the timing, even leaving small real-looking spreads so the pattern reads as organic two-sided flow rather than a metronome. The economics scale exactly like the toy example: the marginal cost per printed dollar of volume is just the fee, so the operator optimizes to minimize it — trading on venues with maker rebates, on its own book, or on chains with negligible gas. The output is a volume figure that can be dialed up to whatever ranking or listing threshold the operator is targeting. And none of it adds a single dollar of *real* depth: the book an honest buyer has to trade against is precisely as thin as before the bot switched on.

#### Worked example: reciprocal wash trading between two "independent" firms

Self-trading through a single account is the easiest version to catch — most exchange matching engines can block an account from crossing its own resting order. So the more durable scheme uses two *legally distinct* parties. Suppose firm X and firm Y agree to trade a token back and forth at \$1.00: X sells 100,000 tokens to Y for \$100,000, then Y sells the same 100,000 back to X for \$100,000, and they repeat the loop all day.

- Each pass prints **\$200,000** of real volume on the tape (two \$100,000 trades).
- After each loop, **both firms hold exactly what they started with** — X has its tokens back, Y has its cash back. Net position change for each: zero.
- Because the two accounts belong to different entities, the venue's self-trade filter never fires. Proving the coordination requires showing the firms acted in concert — usually only visible by tracing that both wallets were funded from a common source, or via the statistical fingerprints below.

This is the structure at the heart of the most contested real-world allegations: a market maker and a project, or two desks, generating volume *for* a token that neither is really taking a position in. It is also why detection so often has to be statistical or on-chain rather than a simple same-account check.

### Reported volume vs real volume

If wash trading were rare, it would be a curiosity. The uncomfortable finding of the last several years of research is that, across the long tail of exchanges, it has been *large*. The most-cited early estimate came from the asset manager **Bitwise**, which analyzed 81 exchanges and told the U.S. Securities and Exchange Commission in **March 2019** that about **95% of reported bitcoin spot volume was non-economic** — wash-traded or otherwise fabricated.

![A tall red bar of ~\$6bn reported daily BTC spot volume next to a tiny green bar of ~\$273m estimated real volume, with the 95% gap shaded amber; a note attributes the figures to Bitwise's 2019 SEC filing.](/imgs/blogs/wash-trading-spoofing-and-manufactured-volume-3.webp)

In Bitwise's sample, aggregators showed roughly **\$6 billion/day** of reported bitcoin volume, but only about **\$273 million/day** — near **4.5%** — looked economically real, concentrated on the ~10 exchanges whose trade data behaved like genuine markets. (This was a 2019 research estimate submitted in a spot-bitcoin-ETF application; it was contested at the time — Alameda Research, then a major trading firm, publicly disputed the methodology. Treat it as a well-sourced *estimate of a moment*, not a permanent constant.)

A peer-reviewed academic study reached a similar conclusion for the broader exchange landscape. In **Crypto Wash Trading** (Cong, Li, Tang & Yang, published in *Management Science* in 2023), the authors examined transaction-level data across major centralized exchanges and estimated that on **unregulated** venues, wash trading **averaged over 70% of reported volume** — implying trillions of dollars of fabricated volume a year — while **regulated** exchanges showed the statistical patterns you expect from real markets. Their detection method matters as much as the number, and we will use it in the detection section.

#### Worked example: estimating real vs reported volume

You do not need a research lab to sanity-check a volume number. You can bound it from the order book. The logic: real volume is real trades, and real trades have to be *absorbed by the book*. If the book is small, the volume can only be large if that small book "turns over" an implausible number of times a day.

Take a token that reports **\$5 billion/day** of volume. Pull its real depth: suppose across all venues there is about **\$10 million** of orders resting within ±2% of the mid-price. Ask how many times that book would have to completely turn over to produce \$5bn of genuine trading:

$$\text{implied turns} = \frac{\$5{,}000{,}000{,}000}{\$10{,}000{,}000} = 500 \text{ times per day.}$$

A real, liquid market might turn its near-touch book on the order of 10–50 times a day. **500 turns is not trading; it is printing.** Even generously assuming 20 turns/day, the *plausible* real volume is about \$10m × 20 = **\$200 million/day**, meaning roughly **96% of the reported \$5bn is unexplained** by the visible liquidity. That residual is your fake-volume estimate. This is the back-of-the-envelope version of what Bitwise and the academics did with far more data — and it is the single most useful reflex a retail reader can build.

## 2. Why anyone bothers: the incentive to fake

If wash trading transfers nothing, why is it everywhere? Because the *number* is valuable even when the *trading* is not. Reported volume is an input to almost every ranking, filter, and listing decision in crypto, and those decisions route real money.

![A causal loop: a project pays for volume, the token looks liquid, it ranks higher on aggregators and the exchange earns fees, it wins a listing, retail discovers it, and real buyers become insider exit liquidity.](/imgs/blogs/wash-trading-spoofing-and-manufactured-volume-4.webp)

Follow the loop:

- **Rankings.** For years, aggregators like CoinMarketCap and CoinGecko sorted assets and exchanges partly by reported volume. A token that "trades \$50m/day" ranks above one that "trades \$500k/day," gets more page views, more screener hits, more attention. Fake volume buys rank; rank buys eyeballs.
- **Listings.** Exchanges want to list assets that will generate real trading fees. A token that *looks* liquid and in-demand is an easier "yes." Manufactured activity is, in effect, an audition — and a listing on a major venue is itself a price event (a dynamic explored in [how VCs move price](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative)).
- **The appearance of a healthy market.** Retail traders avoid ghost towns. A token with visible depth and a busy tape *feels* safe to enter — which is exactly the feeling a manipulator wants to sell, because the real buyers who arrive become the **exit liquidity** for insiders holding cheap early supply.
- **The venue's own incentive to look away.** An exchange's league-table position and marketing lean on aggregate volume. Cracking down on a high-"volume" client can mean cutting your own reported numbers and a fee-paying relationship. That conflict is why detection so often has to come from *outside* the venue.

There is a specific contractual channel worth naming. When a project launches a token, it usually signs a **market-making agreement** with one or more MM firms, sometimes structured as a loan of tokens plus call options (the mechanic covered in the [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) post). Some of these agreements have historically specified **volume or "liquidity" targets** — the MM is expected to make the token *look* active. An honest version of that target is met by genuinely tightening spreads and quoting real two-sided size. A dishonest version is met by wash trading, because printing volume is far cheaper than committing real capital to a deep book. The line between "providing liquidity" and "manufacturing the appearance of it" is exactly where the abuse lives, and it is often invisible from outside the contract — which is why the allegations in this space so frequently involve a project's own paid market maker.

This is the structural reason manufactured volume is tolerated rather than stamped out: at each step, someone in the loop benefits. It is the same conflict-of-interest map that runs through the whole [crypto power structure](/blog/trading/crypto/crypto-vc-and-market-makers) — incentives that are "aligned" right up until they collide with the retail buyer at the end of the chain.

## 3. Spoofing and layering: faking the price, not the volume

Wash trading fakes the *volume* number. **Spoofing** fakes the *price* — and it does so without any completed trade at all. It works entirely in the order book, on the resting orders that have not yet filled.

**Spoofing** is placing an order you intend to cancel before it executes, in order to create a false impression of supply or demand and move the price. **Layering** is spoofing with multiple orders stacked at several price levels, to make the fake pressure look deeper and more convincing. Both exploit a simple behavioral fact: other traders and trading bots read the book as information. A big resting bid looks like real demand; algorithms and momentum traders lean toward it. The spoofer's whole trick is to *look like* demand or supply just long enough to move the last price, then vanish.

![A two-panel order book: on the left, a spoofer posts a 200k fake bid wall at \$0.98 below a thin real book; on the right, after the last price is nudged to \$1.01 and the spoofer's real 50k sell is filled, the wall is canceled with zero traded.](/imgs/blogs/wash-trading-spoofing-and-manufactured-volume-5.webp)

Read the two panels left to right. On the left, the real book is thin — a few thousand tokens on each side around a \$1.00 last price — and the spoofer is sitting on a real **50,000-token sell order** at \$1.01 that it actually wants to fill. To attract buyers up to it, the spoofer posts a **200,000-token bid at \$0.98**: a wall far bigger than anything real in the book. That wall is never meant to trade; it exists to make the book look one-sided and bullish. Momentum traders and bots see "huge demand just below," and buy — lifting the last price toward \$1.01. The spoofer's real sell fills into that manufactured demand. Then, on the right, the wall is **canceled**: zero tokens ever traded at \$0.98. The displayed demand was a prop.

#### Worked example: a spoof on a thin book

Put numbers on the spoofer's edge. The spoofer wants to sell 50,000 tokens.

- **Without the spoof:** the real book is thin. Selling 50,000 tokens into a few thousand tokens of resting bids walks the price *down* — say the average fill comes in around \$0.99 because the sell pressure eats through weak bids. Proceeds ≈ 50,000 × \$0.99 = **\$49,500.**
- **With the spoof:** the 200,000-token wall makes the book look like demand is stacking up. Buyers chase; the last price ticks up to \$1.01, and the spoofer's resting offer gets lifted there. Proceeds ≈ 50,000 × \$1.01 = **\$50,500.**
- **Edge from the spoof:** \$50,500 − \$49,500 = **\$1,000** on this one clip, from an order the spoofer never intended to honor. Scale that across thousands of clips a day and the fake wall is a money machine — one that dissolves the instant anyone tries to trade against it.

Notice what did *not* happen: no wash trade, no fake volume. Spoofing manufactures a **price signal**, not a volume number. That is why it is a different animal from wash trading, and why detecting it means watching *order placement and cancellation*, not just completed prints.

#### Worked example: layering the book to make the fake look deep

A single giant wall can look suspicious precisely because it is a single giant wall. **Layering** makes the fake pressure look like organic depth by spreading it across several price levels. Suppose the real book under \$1.00 is thin — a few thousand tokens at \$0.99, \$0.98, and \$0.97. The layerer adds fake bids at *each* level: **40,000 at \$0.99, 60,000 at \$0.98, 80,000 at \$0.97.** Now the book reads as a steadily thickening wall of demand — exactly the shape a genuine accumulation would make — and it is far more convincing than one lone order. Momentum algorithms that weight *cumulative* depth near the touch see a strongly bid book and lean long. The layerer sells its real inventory into that lean, then cancels all three layers together. The tell is in the cancellations: legitimate resting orders get *filled* as price moves through them; layered orders get *pulled* in a coordinated burst the instant price approaches, having transferred nothing.

This points at the cleanest spoofing signature there is: the **order-to-trade ratio** (or cancel-to-fill ratio). A genuine market maker posts orders it is willing to have filled, so a healthy fraction of them execute. A spoofer posts orders it needs *not* to fill, so its cancellation rate is extreme and its executions rare — and, tellingly, the few executions that do happen sit on the *opposite* side from the big canceled orders (the spoofer sells while walling the bid). Regulated venues and surveillance teams monitor exactly this ratio; it is one of the few tells that catches spoofing even when each individual order, viewed alone, looks perfectly innocent.

### Painting the tape and momentum ignition

Two close cousins round out the family:

- **Painting the tape** is executing a series of small trades — often wash trades — specifically to *create the appearance of price movement or activity* on the chart, "painting" a picture for anyone watching the tape. A manipulator can ratchet the last price up with tiny self-trades: sell 10 tokens to yourself at \$1.01, then \$1.02, then \$1.03, each a trivially small print, and the chart now shows a rising price on rising "volume." The candles look bullish; the moves cost almost nothing and transferred almost nothing.
- **Momentum ignition** is a deliberate burst of orders (real or spoofed) designed to *kick off* a fast move and trigger other traders' stops and momentum algorithms, so the crowd carries the price the rest of the way. The igniter lights the fuse and steps back to let reflexivity do the work.

All four techniques — wash trading, spoofing, layering, painting the tape — share one DNA: they generate a **signal** (volume, depth, or price action) that is disconnected from real supply and demand, in order to make someone else act on it.

## 4. Detecting the fakes: statistical and on-chain fingerprints

Here is the good news, and the reason this article can be defensive rather than cynical: manufactured activity is *detectable*, because faking a market and running a real one leave different statistical and on-chain signatures. You will not catch everything, but you can catch a lot with public tools.

### Volume-vs-liquidity divergence

Start with the single most powerful tell, the one from the worked example above generalized into a habit. Real volume must be absorbed by real liquidity, so **reported volume and order-book depth should move together.** When volume is enormous but depth is tiny, the volume is suspect.

![A three-by-three matrix comparing a healthy token (\$5M volume, \$500k depth, ~10 turns/day) with a suspicious token (\$50M volume, \$80k depth, ~625 turns/day) and reading the divergence as a red flag.](/imgs/blogs/wash-trading-spoofing-and-manufactured-volume-6.webp)

#### Worked example: the divergence red-flag calc

Two tokens:

- **Healthy token:** reports **\$5m/day** of volume and shows **\$500k** of depth within ±2% of mid. Turnover = \$5,000,000 ÷ \$500,000 = **10 turns/day.** Plausible — a real market can turn its near book ten times in a day.
- **Suspicious token:** reports **\$50m/day** — ten times more — but shows only **\$80k** of depth. Turnover = \$50,000,000 ÷ \$80,000 = **625 turns/day.** For that to be real, the entire near-touch book would have to be completely traded and replaced 625 times in 24 hours, roughly every two minutes, all day, with no lasting price impact. That is physically absurd. The volume is almost certainly manufactured.

The tell is not the size of either number alone — it is the **divergence** between them. A big volume number sitting on a thin book is the loudest red flag in the market, and you can compute it in ten seconds from any depth chart.

### Benford's law and round-number clustering

Real trade sizes are generated by countless independent decisions — different traders buying different amounts for different reasons. Numbers produced that way obey **Benford's law**, a statistical regularity in which the leading digit is a **1** about **30.1%** of the time, a **2** about **17.6%**, and so on down to **4.6%** for a **9**. It shows up in naturally occurring data across accounting, economics, and science, and it is a classic fraud-detection tool precisely because *fabricated* numbers usually fail it.

![A grouped bar chart contrasting the Benford distribution of leading digits in real markets (a tall 30.1% bar for digit 1, tapering to 4.6% for 9) with a flat ~11% distribution for wash-traded data.](/imgs/blogs/wash-trading-spoofing-and-manufactured-volume-7.webp)

Wash-trading bots, generating sizes algorithmically, tend to spread their leading digits **too evenly** — flattening toward ~11% each instead of Benford's steep taper. That is exactly the test the *Crypto Wash Trading* authors used: they compared each exchange's first-significant-digit distribution against Benford's law and found that unregulated venues **violated it**, while regulated ones matched it. A related tell is **round-number clustering** — real order sizes are messy (0.4173 BTC), while fabricated ones pile up on suspiciously round lots (exactly 1.0, 5.0, 10.0), because a script writing round numbers is easier than a script imitating human mess. Neither test is proof on its own, but a venue that fails Benford *and* clusters on round lots *and* shows volume-vs-depth divergence is telling you something.

### On-chain: self-funded wallet loops and circular flows

Crypto has a detection advantage traditional markets lack: the settlement layer is **public**. On decentralized exchanges (DEXs) and for on-chain transfers, you can often see the wallets themselves. That turns "who is really on both sides of this trade?" from a guess into a graph you can trace.

The signature of on-chain wash trading is a **self-funded loop**: a cluster of wallets that were all originally funded *by each other* (or by one common source) trading back and forth among themselves. **Chainalysis**, in its 2022 crime report, formalized one version for NFTs — it flagged **262 addresses** that had each sold NFTs to **self-financed** addresses more than **25 times**, a "25-transaction threshold" it used to call an address a habitual wash trader. The same idea generalizes: if wallet A sells to wallet B, and B was funded by A (or both were funded by wallet C), the "trade" between them is not price discovery — it is one owner talking to themselves on a public ledger. Tools like **Arkham**, **Nansen**, **Etherscan**, **Dune**, and **Bubblemaps** exist to make these funding relationships and circular flows visible. This is the hands-on forensic track the later posts in this series pick up — see the forthcoming [detecting-manipulation-onchain-red-flags](/blog/trading/crypto-players/detecting-manipulation-onchain-red-flags) and [the-price-manipulation-playbook](/blog/trading/crypto-players/the-price-manipulation-playbook) for the step-by-step workflows.

Two related on-chain patterns deserve names. A **round-trip wallet** is one that ends a sequence of trades holding almost exactly what it started with — tokens out, tokens back, net near zero — which is the on-chain fingerprint of a wash loop. A **circular flow** is the same idea across a group: value that leaves a cluster of wallets and returns to it, sometimes through several hops meant to disguise the loop. The disguise is rarely perfect, because every hop is permanently recorded; a graph tool can collapse the intermediary wallets and reveal that the "market" for a token was, on inspection, a handful of addresses passing it in a circle. On a decentralized exchange this is especially visible: the trades, the liquidity-pool interactions, and the wallet funding all sit on the same public ledger.

### CEX reported volume vs on-chain reality

Centralized-exchange (CEX) trades happen *inside* the exchange's own database — they do not each hit the blockchain — so you cannot see individual CeFi trades on-chain. But you *can* see the exchange's wallets: the deposits flowing in and the withdrawals flowing out. That gives you a sanity ceiling, because real trading ultimately requires assets to arrive at and leave the venue.

#### Worked example: the settlement-ceiling check

Suppose a venue reports **\$2 billion/day** of volume for a token, but its publicly labeled wallets show only a **few million dollars** of that token ever moving in or out over several weeks. Genuine two-sided trading of \$2bn/day, sustained, would demand large and persistent on-chain flows to and from the exchange as traders fund positions and withdraw proceeds. When the reported number is three orders of magnitude larger than any observable settlement, the trades — if they happened at all — were the same coins looping inside the venue's internal ledger: the CeFi equivalent of the two-account wash from the very first example. Analysts build exactly this comparison with **DefiLlama**, **Dune**, **Arkham**, and **Nansen**. A large, persistent gap between reported volume and observable on-chain support is one of the strongest available signals that a venue's numbers are inflated — and, importantly, it works even though you cannot see the individual CeFi trades themselves.

### The red-flag checklist

Put the fingerprints together into one scan you can actually run.

![A five-row checklist matrix: volume larger than depth, round-number/digit anomalies, self-funded wallet loops, CEX volume larger than on-chain reality, and volume with no price discovery — each with how to check it and which public tools to use.](/imgs/blogs/wash-trading-spoofing-and-manufactured-volume-8.webp)

| Red flag | What it catches | How to check it |
|---|---|---|
| Volume ≫ order-book depth | fake volume on a thin book | divide reported 24h volume by ±2% depth (turns/day) |
| Round-number / digit anomalies | synthetic, non-natural trade sizes | Benford digit test; look for round-lot clustering |
| Self-funded wallet loops | trades between wallets one owner controls | trace which wallets funded each other (Arkham/Nansen/Etherscan) |
| CEX volume ≫ on-chain settlement | inflated exchange print | compare a venue's reported volume to on-chain deposits/withdrawals |
| Volume with no price discovery | volume that is all noise | volume spikes while price stays flat and the spread stays wide |

None of these is a courtroom-grade verdict. But any two of them together should move a token from "interesting" to "prove it to me." The aggregators learned this the hard way: after the 2019 fake-volume revelations, **CoinGecko introduced its "Trust Score"** later that year, deliberately ranking exchanges by **liquidity, order-book depth, and volume consistency** rather than raw reported volume — an explicit admission that the volume number alone was too easy to fake to be trusted.

### What detection cannot see

Honesty about the method matters: on-chain and statistical detection is powerful but not complete. **OTC block trades** are negotiated off the order book and often settled privately, so they never touch the public tape at all — real size can move invisibly, which is the subject of [OTC desks and moving size without moving price](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price). **CeFi internalization** hides individual trades inside an exchange's database, so you see only the aggregate the venue chooses to report. **Mixers and fresh wallets** break the funding trail that self-funded-loop detection relies on. And a sufficiently sophisticated operation can *deliberately* imitate Benford's law and human-messy sizes to pass the statistical tests. The right posture is not "the tests catch everything" but "the tests raise the cost and catch the crude majority" — and combining several weak signals (divergence *and* digits *and* wallet loops) is far harder to defeat than any one alone. Where the trail genuinely goes dark, that itself is information: opacity concentrated around a single token is a reason for more caution, not less.

## How it shows up in price

Manufactured activity does not stay in the abstract; it changes the price you pay and the exit you get.

- **You buy a "liquid" token that is not liquid.** The tape and the ranking said it was busy; the book was thin the whole time. Your entry order slips more than you expected, and when you try to exit, the "volume" evaporates — because it was never there. You provided the real liquidity; the fake volume just lured you in.
- **You chase a spoofed move.** The price ticks up on what looks like stacking demand, you buy the breakout, and the wall cancels. The demand you chased was a prop, and now you are holding at a price no real buyer supports.
- **You misprice risk from the chart.** Painted candles show a smooth, rising trend on "rising volume." You size up because it looks healthy. The trend was manufactured, and the reversal — when the manipulator stops painting — is faster and deeper than a real trend's.
- **You trust a ranking.** The token was near the top of a screener sorted by volume. Sorting by a fakeable number selected *for* the tokens most willing to fake it. The ranking was, in part, a filter for manipulation.

There is a second-order effect worth naming: **reflexivity**. Manufactured signals do not just deceive; they can *bootstrap* the very reality they fake. Fake volume attracts real attention, which attracts real buyers, whose real buying moves the price for real — briefly making the manipulation look like a correct call. A painted uptrend can trigger genuine momentum; a spoofed breakout can start a real one. The danger is that the manufactured foundation is hollow: when the operator stops printing, stops painting, or pulls the walls, the real demand it summoned has nothing underneath it, and the reversal is faster and deeper than an organically-built move would be.

The through-line: every one of these puts *you* on the other side of an insider's trade at a price the real supply-and-demand of the market does not support. Manufactured volume and price are, functionally, a mechanism for transferring cheap early supply to late retail buyers at a good price for the insider.

## Common misconceptions

**"High volume means a token is liquid and safe."** Volume is a claim about the past; liquidity is capacity in the present. A token can report enormous volume on a paper-thin book. Always check depth, not just volume — the divergence between them is the whole game.

**"Wash trading is basically harmless — no one really loses if it nets to zero for the washer."** It nets to zero *for the washer*. It does not net to zero for the retail buyer who entered because the token looked busy and liquid, or for the honest project outranked by a competitor buying fake volume. The loss is transferred, not erased.

**"If the exchange shows the volume, it must be real."** Reported volume is generated by the venue from its own trade log, and the venue often has an incentive to *want* the number high. Regulated venues are policed for this; on the long tail of unregulated ones, the reported number and the real number have, by multiple studies, diverged enormously.

**"Spoofing must involve fake trades."** Spoofing involves fake *orders* — placed and canceled, never executed. There is often no trade at all in the spoof itself. That is why you cannot catch it by looking only at volume or completed prints; you have to watch placements and cancellations in the book.

**"Detecting this requires insider access."** Much of it is checkable from public data: depth charts, trade-size distributions, and — on-chain — the actual wallets. The volume-vs-depth ratio alone, which anyone can compute in seconds, catches a large share of the crudest fakes.

**"A big buy wall on the book is bullish."** A large resting order is *information only if it is real*. The bigger and rounder the wall, and the further it sits from the touch, the more you should suspect it is there to be seen, not filled — and to disappear the moment price approaches it.

**"This only happens on shady micro-cap tokens."** The *rate* of manipulation is highest on small, unregulated venues and thin tokens, but the *techniques* are venue-agnostic — spoofing was defined and prosecuted in the deepest, most regulated futures market in the world (E-mini S&P 500). Size and prestige reduce the odds; they do not make a market immune. Verify, don't assume.

## How it shows up in real markets

These are documented episodes and research findings. Where a specific firm's *intent* is contested, the claim is presented strictly as **reported or alleged**, attributed to its source, with the firm's response noted. Nothing here should be read as a finding of fact against a named company; several of these are estimates or allegations, not adjudicated conclusions.

### 1. The Bitwise "95%" filing (2019)

In March 2019, **Bitwise Asset Management** submitted analysis to the SEC, as part of a spot-bitcoin-ETF application, arguing that roughly **95% of reported bitcoin spot volume** was fake or non-economic — that of 81 exchanges examined, only about 10 showed volume that behaved like a real market, and that genuine daily volume was closer to **\$273 million** versus the **~\$6 billion** aggregators displayed. The claim was influential in how regulators and data providers thought about crypto volume, and it was also **contested** — Alameda Research publicly disputed the methodology. Its lasting importance is less the exact percentage than the demonstration that *reported volume and real volume could differ by more than an order of magnitude.* (Sources: Bitwise's SEC submission; contemporaneous coverage in Forbes and CNBC, 2019.)

### 2. The academic estimate: over 70% on unregulated venues

The peer-reviewed study **Crypto Wash Trading** (Cong, Li, Tang & Yang, *Management Science*, 2023; NBER working paper 30783) analyzed transaction data across major centralized exchanges and estimated wash trading **averaged more than 70% of reported volume on unregulated exchanges**, implying trillions of dollars of fabricated volume annually, while regulated venues showed normal patterns. Crucially, the authors did not just assert it — they demonstrated it statistically, via **Benford's-law** violations, **round-number clustering**, and abnormal **tail distributions**, and showed the fake volume measurably improved exchange rankings. It is the most rigorous public evidence that manufactured volume has been systemic, not incidental, on the unregulated tail of the market.

### 3. The DWF Labs allegations (reported, 2024 — firm denies)

In **May 2024**, *The Wall Street Journal* **reported** that Binance's internal market-surveillance team had, in an internal report, **alleged** that the market-making and investment firm **DWF Labs** engaged in roughly **\$300 million of wash trading in 2023** and manipulated the price of the YGG token and, per the report, at least six other cryptocurrencies. The *Journal* further reported that Binance did **not** act on the recommendation to remove DWF as a client, deemed the evidence insufficient, and dismissed the head of the surveillance team. **DWF Labs publicly denied the allegations**, characterizing them as unfounded and, in its own words, "competitor-driven FUD" that distorted the facts; **Binance** stated that it maintains strict market-surveillance standards. These are **reported allegations and denials**, not established facts or findings by a court or regulator — the episode is included here as an illustration of how such allegations surface, how they are framed, and how a named firm responds, and it is exactly the kind of case the on-chain forensic track later in this series examines from public data. (Source: *The Wall Street Journal*, reported May 9, 2024; DWF Labs and Binance statements in response.)

### 4. Chainalysis and NFT wash trading

In its 2022 crime report, the blockchain-analytics firm **Chainalysis** used on-chain data to flag wash trading in NFT markets, identifying **262 addresses** that had each sold NFTs to **self-funded** addresses more than **25 times** — a threshold it used to label habitual wash traders. Because NFT trades settle on public chains, the funding relationships between the "buyer" and "seller" wallets were directly observable, turning a suspicion into a traceable graph. It is the clearest public demonstration that the on-chain detection method actually works at scale.

### 5. The 2010 "flash crash" and spoofing (the canonical case)

The definitive spoofing prosecution comes from traditional markets, and it is the clearest illustration of the mechanic. **Navinder Singh Sarao**, charged by the **CFTC** and the U.S. Department of Justice in **2015**, ran a "layering algorithm" on E-mini S&P 500 futures that placed large sell orders with a **"cancel if close"** rule so they would never actually execute. On **May 6, 2010** — the day of the "flash crash" — his resting spoof orders reached roughly **\$170–200 million** in notional and, at their peak, were about **20–29% of the entire visible sell side** of the order book before being canceled at 1:40 p.m. CT. The CFTC said he made roughly **\$40 million** over about five years of this conduct; he pleaded guilty in 2016 and was ordered to pay more than **\$38 million** in monetary sanctions. Nearly all of the spoof orders were canceled without ever trading — the textbook definition of the technique. (Sources: CFTC press releases 7156-15 and 7486-16; DOJ.)

### 6. Spoofing is illegal — and prosecuted in crypto too

Spoofing is not a gray area in regulated markets. The **Dodd-Frank Act of 2010** explicitly defines it as bidding or offering "with the intent to cancel the bid or offer before execution," and criminal spoofing can carry up to **10 years in prison per count**. The hard part of any spoofing case is proving *intent* — that the trader placed the orders specifically meaning to cancel them — which is why prosecutors lean on exactly the fingerprints in this article: extreme cancel-to-fill ratios, orders sized far larger than anything the trader ever lets execute, and executions that consistently sit opposite the canceled side. That authority reaches crypto derivatives: the **CFTC** has charged traders with spoofing **bitcoin futures** on the CME, establishing that spoof patterns in crypto's regulated venues are detectable and prosecutable. Spot crypto on unregulated offshore venues is a much murkier enforcement environment — no consolidated tape, no mandatory surveillance, jurisdictions that are hard to reach — which is precisely why so much of the manufactured activity concentrates there, and why outside, independent detection matters so much.

### 7. The aggregators' response: from volume to "trust"

The market's own infrastructure eventually adapted. After the 2019 fake-volume findings, **CoinGecko** rolled out its **Trust Score** — later refined into a version that grades exchanges on **liquidity, order-book depth, and volume consistency against a benchmark**, rather than raw reported volume — and **CoinMarketCap** moved toward liquidity- and confidence-adjusted metrics. The reform is itself evidence of the problem: the industry's leading data providers concluded that the headline volume number was too easily manufactured to rank markets by, and rebuilt their rankings around signals that are harder to fake.

Step back and the pattern across all seven episodes is consistent. In *regulated* venues — CME bitcoin futures, the E-mini — manipulation is defined by statute, surveilled, and prosecuted, so it is rarer and, when it happens, it is named and punished. In the *unregulated* long tail — offshore spot exchanges, thin tokens, NFT marketplaces — the same techniques run with far less friction, which is exactly why independent researchers, blockchain-analytics firms, and data aggregators, rather than a regulator, have done most of the catching. The lesson for a reader is not "everything is fake" — the serious, liquid markets are mostly real — but "the trustworthiness of the number depends entirely on who is allowed to check it," and on the unregulated tail, that job falls to you.

## When this matters to you

You are unlikely to spoof a futures book. But you are very likely to *read* a token page, and manufactured volume and price are aimed squarely at readers. Picture the ordinary situation: you find a token near the top of a screener sorted by 24-hour volume, its chart is trending up on rising "volume," and there is a big buy wall just under the price. Every one of those three signals is one of the fakeable ones in this article — the ranking, the painted tape, and the wall. None of them is proof of anything wrong; all three are reasons to run the checks before you commit money, not after. Here is the defensive checklist — the same one the [reading-the-tape retail-defense](/blog/trading/crypto-players/how-crypto-prices-actually-move) mindset runs on:

- **Divide volume by depth before you trust either.** A big volume number on a thin book is the loudest red flag in crypto. Turns-per-day above a few dozen deserves suspicion; hundreds means printing.
- **Watch whether volume moves price.** Genuine volume produces price discovery. Volume that spikes while the price stays flat and the spread stays wide is likely wash-traded noise.
- **Distrust big, round, distant walls.** A giant resting order far from the touch that keeps reappearing and vanishing is behaving like a spoof, not like real demand.
- **Check the venue, not just the token.** On an unregulated exchange with no surveillance, the reported numbers carry far less weight. Use aggregators' liquidity/trust scores rather than raw volume ranks.
- **Follow the wallets on-chain when you can.** For DEX activity, trace whether the "traders" funded each other. Self-funded loops are wash trading in plain sight. The forensic posts later in this series — [tracing a market maker's on-chain footprint](/blog/trading/crypto-players/tracing-a-market-makers-onchain-footprint) and [case forensics](/blog/trading/crypto-players/case-forensics-reconstructing-a-token-episode) — turn this into a repeatable workflow.
- **Remember who benefits.** Fake volume and fake liquidity exist to make a market look worth entering. If you cannot see who is providing the *real* liquidity, assume it might be you.

This is educational, not financial advice. The point is not to make you paranoid — real, liquid, honestly-traded markets exist and are the majority of the serious ones. The point is to give you the one reflex that separates a reader from a mark: **treat the volume number and the order book as claims to be verified, not facts to be trusted, and know exactly how to verify them.**

## Sources & further reading

- **Bitwise Asset Management**, presentation to the SEC on real vs reported bitcoin volume (March 2019); contemporaneous coverage: Forbes, "95% Of Reported Bitcoin Trading Volume Is Fake, Says Bitwise" (2019) and CNBC (March 22, 2019). The ~95% figure was a contested research estimate; Alameda Research publicly disputed it.
- **Lin William Cong, Xi Li, Ke Tang, Yang Yang**, "Crypto Wash Trading," *Management Science* (2023); NBER Working Paper No. 30783. Source of the ">70% of reported volume on unregulated exchanges" estimate and the Benford's-law / round-number / tail-distribution detection methods.
- **Chainalysis**, 2022 Crypto Crime Report — NFT wash trading section (the 262-address, 25-transaction finding).
- **The Wall Street Journal**, reporting on Binance's internal surveillance report and the DWF Labs wash-trading allegations (reported May 9, 2024). DWF Labs and Binance both publicly responded; the allegations are reported, not adjudicated.
- **U.S. Commodity Futures Trading Commission**, press releases 7156-15 and 7486-16 (Navinder Sarao spoofing / price manipulation charges and sanctions); the Dodd-Frank Act's statutory definition of spoofing.
- **CoinGecko**, "Trust Score" methodology (introduced 2019); **CoinMarketCap** liquidity/confidence metrics — the data providers' response to fakeable volume.
- On this blog: [crypto VCs and market makers — the series hub](/blog/trading/crypto/crypto-vc-and-market-makers) · [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) · [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) · [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table).
