---
title: "Real vs Fake Volume: Telling Organic Demand From Manufactured Hype"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Volume is the most faked number in crypto. Learn the on-chain tests — unique buyers, funding diversity, holder growth — that separate real demand from manufactured churn before you buy."
tags: ["onchain", "crypto", "fake-volume", "wash-trading", "organic-demand", "holder-growth", "dexscreener", "dune", "etherscan", "trading", "memecoins"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Reported volume is the easiest metric in crypto to fake, but on-chain you can check whether that volume came from a real crowd of new buyers or from a handful of recycling wallets — and only the first is demand worth buying.
>
> - **What it is:** "Volume" is just the dollar value of trades; it says money moved, not that *new* people committed *new* capital. Wash trades, bot churn, and market-maker round-trips all print as volume while adding zero demand.
> - **How to read it:** Run four cheap on-chain checks — the **unique-buyer test** (distinct wallets, not trade count), **buyer-funding diversity** (many independent sources vs one seed wallet), **holder growth vs volume**, and **net new buying** — using DEXScreener, Dune, and a block explorer.
> - **What you DO with it:** Never buy a token because it has "high volume." Buy only when volume, price, and a *rising holder count* all agree. If volume is high but holders are flat, that is distribution into you — avoid.
> - **The number to remember:** 1,000 trades from 12 wallets is fake; 1,000 trades from 800 wallets is real. Count the *buyers*, not the trades.

On 21 February 2025, the Bybit exchange lost roughly \$1.46 billion to a single cold-wallet exploit — the largest crypto theft ever recorded. In the days that followed, dozens of obscure tokens lit up on the trackers with suspiciously round, suspiciously enormous volume numbers, riding the attention. Traders who sorted their screeners by "24h volume" and bought the top of the list were not buying demand. They were buying a number that someone had manufactured precisely so it would appear at the top of their screen.

This is the oldest trick in the market, and crypto makes it almost free to pull off. In a traditional exchange, faking volume means you have to actually trade against a counterparty and pay a real spread to a real broker. On a decentralized exchange you can be both the buyer and the seller, control both wallets, and round-trip your own money through a pool you partly own. The "volume" you generate is real in the narrow sense that the transactions happened and the gas was paid — and completely fake in the sense that matters: no new person decided this token was worth owning.

The good news, and the reason this post exists, is that the blockchain records *who* did each trade, *where their money came from*, and *whether the holder base actually grew*. Reported volume is one lonely number that anyone can inflate. On-chain demand is a fingerprint with many independent ridges, and faking all of them at once is expensive, slow, and detectable. The same volume number can describe two completely different markets — and the figure below is the whole post in one picture.

![Same five million dollar volume split into a fake column with twelve recycling wallets and an organic column with eight hundred independent buyers](/imgs/blogs/detecting-fake-volume-vs-organic-demand-1.png)

This article teaches you to tell those two columns apart on-chain, with real tools, before the buy decision. We build it from zero — what volume even measures and why it is gameable — then work through each test with dollar-denominated examples, and finish with a checklist you can run on any "high volume" token in about ten minutes.

## Foundations: what volume measures, and why it lies

Let us start with the most basic question, because almost everyone who loses money to fake volume skipped it: **what is volume, exactly?**

**Volume is the total value of trades over a period.** If in one hour ten people each buy \$1,000 of a token and ten people each sell \$1,000, the hour's volume is \$20,000 — the sum of all the trades, buys and sells together. That is the entire definition. Notice what it does *not* contain: it does not say how many distinct people traded, whether they were the same people trading repeatedly, where their money came from, or whether anybody is holding more of the token at the end than at the start. Volume is a flow number. It measures motion, not arrival.

**Demand is something else entirely.** Demand is *new people committing new capital and keeping the token.* When demand is real, two things happen that volume alone never captures: the number of distinct holders goes up, and the *net* flow — buys minus sells — is positive, because more money is entering than leaving. Volume can be enormous while demand is exactly zero. That gap is the whole game.

To make this concrete, here is the everyday version. Say a shopkeeper wants their store to look busy so a landlord will rent them a bigger unit. They could hire one friend to walk in the front door, buy a candy bar, walk out the back, walk around the block, and do it again — a thousand times in a day. The cash register rings a thousand times. "Foot traffic" looks incredible. But there is exactly one customer, and the store sold one candy bar a thousand times to the same person who took it home and brought it back. The *volume* of transactions is huge; the *demand* for the store is one bored friend. That is wash trading, and on a blockchain the friend is a second wallet you control.

### The three machines that print volume without demand

Volume can be inflated by three distinct mechanisms, and a good analyst keeps them separate because they look slightly different on-chain.

**1. Wash trading.** One operator controls two or more wallets and trades the token back and forth between them, often through a pool where the operator is also a liquidity provider. Each round-trip prints volume on both legs. The operator's net position barely changes — they end the day holding roughly what they started with — but the tape shows a torrent of activity. This is covered in depth in the companion post on [detecting wash trading](/blog/trading/onchain/detecting-wash-trading); here we care about it only as a *source of fake volume that fools the buy decision.*

**2. Bot churn.** Automated bots — sometimes the project's own, sometimes third-party "volume bots" you can literally rent — fire rapid small buys and sells to keep the token on screeners' "trending" and "most active" lists. The bots are usually delta-neutral by design: they buy and sell in roughly equal measure so the operator does not accumulate a position they would have to dump later. The point is purely to occupy a slot on a tracker so real humans see the token and FOMO in.

**3. Market-maker (MM) round-trips.** A legitimate market maker quotes both sides of the book and earns the spread; their job is to provide liquidity, not to express a directional view. Their trading generates real, useful volume — but it is *churn*, not demand. An MM that does \$10M of volume a day on a token has not bought \$10M of the token; it has bought \$5M and sold \$5M and pocketed the spread. Confusing MM churn for demand is the most respectable-looking version of this mistake, because the MM is not doing anything wrong — you are simply misreading what their volume means.

All three machines share one tell: **they move tokens without growing the set of people who own them.** That is what the rest of this post is built to detect.

### Reported volume vs on-chain volume

There is a second foundational distinction that trips up beginners: *where the volume number comes from.*

**Reported (CEX) volume** is the number a centralized exchange — Binance, Coinbase, an obscure tier-3 venue — publishes for a trading pair. You cannot independently verify it. The exchange's matching engine is a private database; you see only what they choose to display. Historically this has been the most abused number in the entire industry. A landmark 2019 report by Bitwise, submitted to the U.S. SEC, found that roughly **95% of the reported Bitcoin spot volume** on the unregulated exchanges of the day was fake — wash-traded to manufacture the appearance of liquidity and game ranking sites. The lesson generalizes: a CEX volume number is a marketing claim until proven otherwise.

**On-chain (settled) volume** is different in kind. It is the value of swaps that actually executed against a smart-contract pool — a Uniswap pool on Ethereum, a Raydium pool on Solana — and *every one of those swaps is a public, permanent transaction you can audit.* You can see the wallet, the amount, the counterparty pool, the timestamp, and the gas paid. On-chain volume can still be *washed* (that is what the wallets-and-funding tests are for), but it cannot be simply *invented* the way a CEX can invent a number in a database. This is the core advantage the chain gives you: even fake on-chain volume leaves real, traceable evidence of *how* it was faked.

The mental shorthand: **a CEX volume number is a claim; an on-chain volume number is a receipt.** You still have to read the receipt carefully — but at least there is one.

### Why crypto makes faking volume nearly free

It is worth dwelling on *why* volume is faked more in crypto than anywhere else, because understanding the economics tells you exactly which fakes are cheap (and therefore common) and which are expensive (and therefore rare). Three properties of crypto markets collapse the cost of manufacturing volume toward zero.

**You can be both sides of the trade.** In a stock market, if you want to wash-trade, you have to route an order to an exchange and hope to match against *yourself* — which most venues actively police as a "self-trade prevention" violation, and which still costs you the bid-ask spread to a real market maker in the middle. On a decentralized exchange, there is no matching engine and no counterparty to police anything: you swap against a *pool*, and you can own both the wallet doing the buying and the wallet doing the selling and even part of the pool itself. The "counterparty" is a smart contract that does not care who you are. The only cost is gas and the pool's swap fee — and if you are the liquidity provider, the swap fee partly comes back to you.

**There is no gatekeeper to fake your way past.** Listing a token on a DEX requires no approval, no audit, no minimum float. Anyone can deploy a token contract, seed a pool, and start "trading" against themselves within minutes. The permissionlessness that makes DeFi powerful also means there is no exchange compliance desk asking "is this volume real?" The only gatekeeper is *you*, the reader, doing the checks in this post.

**Trackers reward raw volume mechanically.** DEXScreener's trending list, ranking sites, "most active" feeds — they sort by volume because it is the cheapest signal to compute, and they do it automatically. That creates a direct, mechanical payoff for faking: spend a few hundred dollars in fees to manufacture enough volume to land on a trending list, and thousands of real humans see your token and some fraction FOMO in. The fake volume is an *advertising spend*, and a cheap one. This is the engine behind most of what you will see at the top of a trending list — which is why "it's trending" is the *weakest* possible reason to buy.

Put together: faking *gross volume* and *trade count* is nearly free, so it is everywhere. Faking *unique buyers* costs gas per wallet. Faking *funding diversity* costs laundering effort per wallet. Faking a *growing, sticky holder base* requires genuinely giving away tokens — which defeats the purpose. The cost ladder is the whole strategy of this post: each test up the ladder is one an operator is less willing to pay for, so each test catches a larger share of fakes.

### The volume-bot rental market

To make the "nearly free" point concrete: there is an open market for *renting* volume. Services advertise "volume bots" and "trending packages" that, for a fee, will round-trip a target token thousands of times to inflate its 24-hour volume and push it onto trending lists. The going rate varies, but a package that manufactures, say, \$500,000 of daily volume might cost the project a few hundred to a couple thousand dollars in fees and service charges — a rounding error against the retail capital it can attract.

These bots are deliberately *delta-neutral*: they buy and sell in near-equal measure so the operator running them does not accumulate inventory they would later have to dump. On-chain this leaves a very specific fingerprint — a small set of wallets, enormous trade count, net flow hovering near zero, and a holder count that does not move — which is exactly the signature the net-flow and holder-growth tests are built to catch. When you internalize that the volume might literally have been *purchased as a service*, you stop treating a big volume number as evidence of anything and start treating it as a question.

### Unique buyers vs trade count

The single most important distinction in this whole topic, and the one the cover figure is built around: **the number of trades is not the number of buyers.**

A "trade count" of 1,000 buys could be 1,000 different people each buying once — a broad, healthy crowd — or 12 wallets each buying 83 times — one operator playing pinball. The dollar volume is identical. The trade count is identical. The *unique buyer count* is 800 in the first case and 12 in the second, and that single number is the difference between demand and a mirage.

This is why every serious on-chain check starts by **deduplicating to distinct wallet addresses** before doing anything else. We will formalize this as the unique-buyer test in the next section, but internalize the reflex now: when you see a big volume or trade-count number, your first question is never "how big?" It is always "**from how many distinct wallets?**"

#### Worked example: \$5M of volume, twelve wallets vs eight hundred

Take two tokens, each printing \$5,000,000 of buy volume in a day across 1,000 buy trades.

- **Token A (fake):** Group the 1,000 trades by wallet. Twelve wallets account for all of them — about 83 trades each. The average trade size is \$5,000,000 ÷ 1,000 = \$5,000, and a single wallet doing 83 trades cycled roughly 83 × \$5,000 = \$415,000 of buys (and a similar amount of sells, because it is round-tripping). Twelve such wallets explain the entire \$5M. There is no crowd here — there is a dozen recycling accounts.
- **Token B (real):** Group the same 1,000 trades by wallet. You find roughly 800 distinct wallets, most of them buying once or twice, average buy ≈ \$5,000,000 ÷ 800 = \$6,250 of net new exposure each. These 800 wallets did not exist as holders yesterday; today they each committed real capital and kept the token.

Same \$5M, same 1,000 trades. Token A is twelve wallets manufacturing a number; Token B is 800 people deciding the token is worth owning. Volume could not tell them apart; the unique-buyer count told you instantly.

## The unique-buyer test

The unique-buyer test is the first and cheapest filter, and it disqualifies most fake-volume tokens before you spend another minute on them.

**The procedure, in one sentence:** pull the list of buy transactions for the token over your window, deduplicate by the `from` address (the buyer's wallet), and compare the count of *distinct buyer wallets* to the trade count and to the dollar volume. The figure below is the decision in its purest form.

![Unique buyer test flow grouping one thousand trades into either twelve wallets verdict churn or eight hundred wallets verdict real demand](/imgs/blogs/detecting-fake-volume-vs-organic-demand-2.png)

What you are looking for is the *ratio* of trades to unique buyers and the *shape* of the distribution. A few rules of thumb that hold up across thousands of tokens:

- **Trades-per-buyer near 1–3 is healthy.** Real people buy a token once, maybe add once. A broad base of wallets each trading a handful of times is what organic accumulation looks like.
- **Trades-per-buyer in the dozens is a red flag.** If 1,000 trades come from 30 wallets (33 trades each), that is not a crowd; it is a small ring recycling.
- **A handful of wallets producing the majority of volume is the strongest tell.** If you sort buyers by volume and the top 5 wallets are 80%+ of all buy volume, you are looking at a manufactured market regardless of how big the headline number is.

The reason this works is structural, and it is worth understanding rather than memorizing. **Faking the unique-buyer count is expensive in a way that faking the trade count is not.** To inflate trade count, an operator just makes their existing wallets trade more — free, instant, unlimited. To inflate *unique buyers*, the operator must create and fund many fresh wallets, each needing gas, each needing to be seeded with capital, each leaving a funding trail. The cost and the trail both scale with the number of fake "buyers." So an operator who wants the volume number to look big will almost always take the cheap path — a few wallets trading a lot — and that is exactly the signature the unique-buyer test catches.

### Reading it on DEXScreener and Dune

On **DEXScreener**, open the token's pair page and look at the "Makers" or "Traders" count alongside the "Txns" count for your window. A token with 5,000 transactions but only 40 makers is screaming at you. DEXScreener's maker count is exactly the deduplicated distinct-wallet number; the gap between txns and makers is the trades-per-buyer ratio doing the work for you.

For a rigorous pass, **Dune** is the right tool. The query you want, in plain language, is: *over the last 24 hours, for swaps that bought this token, count distinct `taker` addresses, count total swaps, and sum USD volume.* The skeleton looks like this (Dune's `dex.trades` table already normalizes swaps across chains and DEXes):

```sql
-- distinct buyers vs trade count vs volume, last 24h
select
    count(*)                         as trades,
    count(distinct taker)            as unique_buyers,
    sum(amount_usd)                  as buy_volume_usd,
    count(*) * 1.0 / count(distinct taker) as trades_per_buyer
from dex.trades
where token_bought_address = 0xYOUR_TOKEN_HERE
  and block_time > now() - interval '24' hour
```

The four numbers that fall out — trades, unique buyers, volume, and trades-per-buyer — are your first verdict. A `trades_per_buyer` of 1.4 with thousands of unique buyers is a real crowd. A `trades_per_buyer` of 40 with a unique-buyer count you can fit on a sticky note is manufactured volume.

#### Worked example: \$10M of volume churning while holders stay flat

A token is showing \$10,000,000 of daily volume — a big, attention-grabbing number — and a trader is tempted. Before buying, run the test over ten days. The volume bars stay high the whole window, but the holder count never moves off ~4,100.

![Bar chart of ten million dollar daily volume with a flat holder line at about four thousand one hundred showing churn not demand](/imgs/blogs/detecting-fake-volume-vs-organic-demand-3.png)

Do the arithmetic on what \$10M/day *should* do if it were demand. If even 20% of that volume — \$2,000,000 — were net new buyers at an average position of \$2,000 each, that is 1,000 new holders a day. Over ten days the holder count should have climbed by roughly 10,000, from 4,100 toward 14,000. Instead it sat at 4,100. The conclusion is forced: essentially none of the \$10M/day was net new buying. It was the *same* wallets cycling \$10M of round-trips, holders flat the entire time. That is \$100M of cumulative "volume" across ten days with zero demand behind it — pure churn. You walk away, and you do not feel clever for it; you feel like someone who read the receipt.

## Buyer-funding diversity

The unique-buyer test can be beaten by a determined, well-funded operator who creates hundreds of fresh wallets. It is expensive — but for a token where the payoff is a multi-million-dollar exit, it can be worth it. So the second test goes one level deeper and asks a question that is far harder and costlier to fake: **where did each buyer's money come from?**

This is the funding-diversity test, and it rests on a simple truth about how real and fake crowds are funded. The figure shows both shapes side by side.

![Funding diversity graph contrasting organic buyers funded from Binance Coinbase and peers with a manufactured crowd all funded by one seed wallet](/imgs/blogs/detecting-fake-volume-vs-organic-demand-4.png)

**Real buyers arrive from everywhere.** A genuine crowd of 800 people funded their wallets independently, over time, from many different places: this one withdrew from Binance last month, that one from Coinbase last week, a third received a peer-to-peer transfer from a friend, a fourth bridged from another chain. There is no common origin because there is no common operator. The funding graph fans *in* from a wide, diffuse set of sources.

**A manufactured crowd traces back to one place.** When an operator spins up 200 wallets to fake a broad buyer base, they have to *fund* those 200 wallets — at minimum with gas, usually with the capital to buy. That money comes from the operator's own treasury. So if you trace each "buyer" back to its first incoming transaction — the transaction that funded the wallet — they all converge on a single seed wallet, or a small handful of seed wallets, or a freshly-withdrawn exchange account whose own deposit came from the operator. The funding graph collapses to a point.

**The reflex:** for the top buyers by volume, look at each wallet's *first* funding transaction. If they fan out to many independent exchanges and peers, that is consistent with a real crowd. If they all lead back to one wallet — the way the wash-trading post's cluster all led back to `0xF00...` — you have caught a manufactured market no matter how many wallets it spans.

### Reading it on a block explorer and Arkham

This test is hands-on. On **Etherscan** (or the relevant chain's explorer), take a buyer wallet, sort its transactions to the oldest, and find the first incoming transfer — the one that funded it. Note the sender. Repeat for the top 10–20 buyers. You are building, by hand, the left half of the figure above. **Arkham** and **Nansen** automate the unpleasant part: they label known entities (so the source shows up as "Binance 14" instead of a raw hex string) and let you visualize the funding graph directly, which is why professionals reach for them when the manual trace gets deep. The mechanics of tracing a wallet back through its funding history are covered step by step in [how to trace a transaction flow](/blog/trading/onchain/how-to-trace-a-transaction-flow); here the point is the *pattern*: convergence to one funder is the fingerprint of a fake crowd.

A subtlety worth flagging, in the spirit of "on-chain lies too": a *sophisticated* operator can launder the funding trail — routing each wallet's seed through a mixer, a centralized exchange, or a bridge so the common origin is obscured. This is real, and it is why funding diversity is a *strong* signal, not a *proof*. But it is also expensive and slow, and it interacts badly with the other tests: an operator who carefully launders the funding for 200 wallets has spent real money and time and *still* has to make the holder count grow and stay grown, which is the test that no amount of wallet-spinning can fake.

#### Worked example: the cost of faking a 200-wallet crowd

Suppose an operator wants to fake the funding-diversity test by routing 200 wallets through a mixer so they do not obviously share an origin. On Ethereum at a busy moment, a single mixer deposit plus the wallet seeding can run on the order of \$30–\$60 in gas per wallet once you count the deposit, the withdrawal, and the buy. Take \$45 as a round figure: 200 wallets × \$45 = \$9,000 in pure gas, before a single dollar of the actual buying capital. To make each of the 200 wallets look like a \$5,000 buyer, the operator must also commit 200 × \$5,000 = \$1,000,000 of real capital to the buys. So manufacturing a *plausible* 200-buyer crowd costs roughly \$1,009,000 up front — and the operator still has to eventually sell that \$1M back out without crashing the price they manufactured. Faking the *shape* of demand costs almost as much as real demand would, which is exactly why most fakes don't bother and get caught by the cheaper unique-buyer test.

## Holder growth vs volume — the test that can't be faked cheaply

If you remember one section from this post, make it this one. The unique-buyer and funding tests detect *how* volume was faked; the holder-growth test detects the thing fake volume can never produce: **a permanently larger set of people who own the token.**

Here is the principle. **Real demand is conservation of holders going up.** When a new person buys and keeps a token, the holder count increases by one and stays increased. Wash trades, bot churn, and MM round-trips all *move* tokens between existing wallets without creating new holders — by construction, because the operator does not want to give the tokens away to strangers; they want to keep them to dump later. So the holder count is the one number that a fake-volume operation structurally cannot grow, because growing it would mean genuinely distributing the supply to people who might never sell back.

This gives you a clean divergence test:

- **Volume up + holders up together → real demand.** New money is arriving and staying. The volume is the *footprint* of genuine buying.
- **Volume up + holders flat → manufactured.** The same wallets are cycling. The volume is noise. This is the most common fake-volume signature, and the [\$10M-churn worked example above](#the-unique-buyer-test) is exactly it.
- **Volume up + holders *down* → distribution / a top.** Insiders or early buyers are selling into the volume; the crowd is shrinking even as the tape looks busy. This is the listing-pump and pump-and-dump signature, covered next.

The reason this is the hard test to beat: to grow the holder count, the operator must actually hand tokens to wallets they do not control and that *keep* the tokens. The moment they do that, they have lost the tokens — they have done the opposite of accumulation. There is no cheap way to fake a genuinely growing, *sticky* holder base, because a sticky holder base is the literal definition of the thing you are trying to fake. (You can airdrop tokens to 10,000 wallets to inflate the *holder count* directly — but that shows up as a single distribution event with no buys behind it, and those wallets typically dump or sit dead, which the net-buys check catches.)

### Reading holder growth

**DEXScreener** and **Birdeye** (for Solana) show a holder count and often a holder chart. **Dune** dashboards for most major tokens track holders over time. The reflex is to overlay the holder line on the volume bars, exactly as the figure above does, and ask: *do they move together?* A holder line that climbs with the volume is the green light. A holder line that flatlines under a forest of volume bars is the red one.

For memecoins specifically — where this matters most because the fakes are most aggressive — pair this with concentration: a "growing" holder count is worthless if the top 10 wallets still hold 90% of the supply, because those 10 can dump on the new holders at will. The companion post on [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) goes deep on reading the holder distribution itself; here, holder *growth* and holder *concentration* are two halves of the same demand picture.

#### Worked example: \$1M of genuine buying that grew holders 40%

Now the positive case, so the test is not purely a way to say no. A token has a holder count of 5,000 and a trader is watching to see whether a real demand wave is forming. Over a week, on-chain net buying — buys minus sells — totals \$1,000,000, and the holder count climbs from 5,000 to 7,000, a 40% increase.

Check that the numbers cohere. \$1,000,000 of net new buying spread across 2,000 new holders is an average of \$1,000,000 ÷ 2,000 = \$500 of net new exposure per new holder — a believable retail buy size. The funding test shows the 2,000 new wallets funded from dozens of different exchanges and peers, not one seed. The unique-buyer test shows a low trades-per-buyer ratio. And critically the holder count *stayed* at 7,000 — the new buyers held rather than round-tripping out. All four signals agree: this is \$1M of real demand, and the 40% holder growth is the receipt. *This* is what you are willing to buy. Not the \$10M-volume churn from earlier — this.

### How these tests can deceive you

In the spirit of the series — *on-chain lies too* — no single test is a proof, and a disciplined analyst knows each one's failure mode before relying on it. Here is the honest accounting.

**The unique-buyer test has false positives at the extremes.** A genuinely new token in its first hours may legitimately have only a handful of buyers — early adopters who found it before the crowd. A low buyer count there is not necessarily fake; it may just be early. The disambiguator is the *other* tests: if those few buyers are funded independently and are actually holding (positive net flow, no round-trips), it is early organic interest, not a wash ring. Conversely, the test has false *negatives* against a well-funded operator who spins up hundreds of genuinely distinct wallets — which is why funding diversity exists as the next rung.

**Funding diversity can be laundered.** A determined operator can route each fake wallet's seed through a centralized exchange, a mixer, or a bridge so the common origin is obscured, defeating a naive funding trace. This is real, but it is expensive and slow, and it does not solve the operator's deeper problem: they still have to make the holder base grow and *stay* grown. Laundered funding buys past the funding test and crashes straight into the holder-growth test.

**Holder count can be inflated by airdrops.** An operator can airdrop a token to 10,000 wallets to make the holder count jump, faking the one signal that is supposedly hardest to fake. But an airdrop has a distinct on-chain shape: a single distribution transaction (or batch) with *no buys behind it*, and recipient wallets that typically either dump immediately or sit dead forever. The net-buys check catches it — a holder count that grew without corresponding net new *buying* is a giveaway, not demand. So even the holder test is read alongside net flow, never alone.

**Net flow can be gamed over short windows.** Over a single hour an operator can show positive net flow simply by buying and not yet selling — the selling comes later, in the fade. This is why net flow is read over a *meaningful window* (days, not minutes) and cross-checked against whether the holders who produced that net buying are still holding. A net-positive hour followed by insider exchange-inflows is the listing-pump signature, not accumulation.

The robust posture, then, is not "find the one magic test" but **demand convergence.** A real demand wave passes all four tests *and* triangulates clean (price, volume, holders agreeing) *and* survives a holder-distribution check. A fake fails at least one — usually the cheapest one the operator declined to pay for. Build your verdict from the agreement of independent signals, exactly the way a forensic analyst builds a case from corroborating evidence rather than a single witness. The same discipline underlies reading [active addresses and network activity](/blog/trading/onchain/active-addresses-and-network-activity): distinct *participants*, corroborated across signals, beat any single inflated count.

## CEX-listing pump anatomy

A centralized-exchange listing is the most respectable-looking fake-demand event in crypto, because the volume spike is *real* — real people really do pile in — and yet buying it is usually a fast way to lose money. Understanding the anatomy is the difference between selling the listing and being sold to.

![Timeline of a centralized exchange listing pump from pre listing accumulation to a three times spike to insiders selling two million dollars to a seventy percent fade](/imgs/blogs/detecting-fake-volume-vs-organic-demand-5.png)

The four phases play out almost identically every time:

**Phase 1 — Pre-listing accumulation.** Before the listing is announced, insiders — the team, early investors, the market maker engaged for the listing — quietly accumulate the token cheaply, often on-chain via a DEX or OTC. On-chain you can sometimes see this: a cluster of wallets steadily buying weeks before any news, frequently funded from the project treasury.

**Phase 2 — The listing spike.** The listing goes live. The headline ("Now trading on [major exchange]") triggers a wave of retail FOMO. Price jumps — 2×, 3×, sometimes 10× in minutes. Volume explodes, and *this volume is genuine* in the sense that real new buyers are hitting the market. This is precisely what makes it dangerous: the unique-buyer test *passes* during the spike. There really are hundreds of new buyers.

**Phase 3 — Insiders sell into the spike.** Here is the catch. The new retail demand is the *exit liquidity* for the Phase-1 accumulators. Insiders and the market maker sell their cheap inventory into the FOMO. On-chain — for tokens that settle on-chain — you can watch the Phase-1 wallets sending tokens to the exchange (an inflow), the on-chain analog of "they're about to sell." This is where exchange-flow reading earns its keep; the mechanics are in [exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows).

**Phase 4 — The fade.** Once the insiders have distributed their inventory, the buying that was holding the price up vanishes. There were never enough *sticky* new holders to support the elevated price — the buyers were chasing a spike, not accumulating a position — so price fades, often 50–80% off the listing high over the following days. Volume dries up. And the holder count, if you check it a week later, is barely above where it started: the spike created transactions, not lasting demand.

The on-chain read that saves you: **during a listing spike, watch the net flow and the insider wallets, not the price.** If the Phase-1 accumulation wallets are sending to exchanges while retail buys, that is distribution dressed as demand — the holder count will not grow, and the fade is coming. This is the same coordinated-distribution pattern dissected in [pump-and-dump and coordinated buying](/blog/trading/onchain/pump-and-dump-and-coordinated-buying); a listing is just a pump with a press release.

#### Worked example: a 3× listing spike that fades 70% as insiders sell \$2M

A token lists at \$1.00 and spikes 3× to \$3.00 in the first hour on heavy volume. It looks like demand — and in the narrow sense, it is: real buyers pushed it up. But trace the Phase-1 wallets. Across the spike, a cluster of insider wallets that accumulated near \$0.50 sends tokens worth \$2,000,000 to centralized exchanges — the on-chain footprint of \$2M of selling into the FOMO. Those insiders bought near \$0.50 and sold near \$3.00: on the slice they distributed, that is roughly a 6× exit, turning perhaps \$330,000 of cost basis into \$2,000,000.

Now the holder check. A day after the spike, the holder count is up only marginally — the buyers were chasers, not accumulators. With insider selling exhausted and no sticky demand underneath, price fades from \$3.00 to \$0.90, a 70% drop from the high. A trader who bought the spike at \$3.00 and held is down 70%, to \$0.90. A trader who *read the exchange inflows* saw \$2M of insider distribution against a flat holder count and stayed out. The listing's volume was real; its demand was a press release.

## Market-maker churn vs accumulation

This is the most subtle of the fakes, because nobody is doing anything dishonest. A market maker's job is to quote both sides and provide liquidity, and the volume they generate is genuine, useful trading. The error is purely in the *reader*: mistaking the MM's two-sided churn for one-sided demand.

The key fact: **a market maker is delta-neutral by design.** They aim to end each period holding roughly the same inventory they started with, earning the spread between their bid and ask on the flow that passes through. If an MM does \$10M of volume on a token in a day, they did not *accumulate* \$10M of the token — they bought about \$5M and sold about \$5M, and their net change in holdings is near zero. Their volume is a *byproduct* of providing liquidity, not an expression of demand.

So when you see a token with steady, high, smooth volume and you discover that one or two wallets (the MM's) are on both sides of most of it, the correct read is: *there is a market maker here providing liquidity.* That is genuinely useful — it means you can probably get in and out without enormous slippage — but it is **not** a buy signal. The MM is not betting the token goes up; they are agnostic. Reading their churn as demand is like concluding a casino is bullish on its customers because so much money crosses the table.

How to tell MM churn from accumulation on-chain:

- **MM churn:** one or two wallets, high volume, *near-zero net flow* (buys ≈ sells), holder count flat. The wallets are usually labeled or identifiable as MMs via their funding (often from the exchange or the project treasury) and their mechanical, two-sided behavior.
- **Accumulation:** wallets that buy and *do not sell back* — net flow strongly positive, holdings growing, often spread across many independent wallets. This is the smart-money signature covered in [following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets).

The single discriminator is **net flow**. Churn round-trips to ≈ zero net; accumulation leaves a positive net position behind. Always look at net buys, never gross volume.

#### Worked example: \$8M of MM volume that is \$0 of net demand

A token shows \$8,000,000 of daily volume, comfortably enough to top a screener. Pull the trades and group by wallet: two wallets account for \$7,200,000 of the \$8M — 90% of it. Now compute their *net* flow. Wallet 1 bought \$3,600,000 and sold \$3,560,000, net +\$40,000. Wallet 2 bought \$3,560,000 and sold \$3,600,000, net −\$40,000. Together: +\$40,000 − \$40,000 = \$0 net flow across \$7.2M of gross volume. These are market makers round-tripping the spread, not buyers accumulating. The \$8M headline contained essentially \$0 of net new demand. The remaining \$800,000 of volume from other wallets might contain real buyers — *that* is the slice worth analyzing — but the \$7.2M MM churn is liquidity, not a reason to buy.

## Where this matters most: Solana memecoins

Everything so far applies to any token on any chain, but there is one arena where fake volume is so dense that these tests are not optional — they are survival. That arena is the Solana memecoin launchpad ecosystem, where tools like Pump.fun let anyone create a token in seconds for a few dollars.

The scale is staggering and the survivorship brutal. Cumulatively on the order of **8 million tokens** have been launched through Solana launchpads, and by reasonable estimates only around **1.4% ever reach a meaningful market cap** — the rest go to zero, most within hours. (These figures are approximate, drawn from public launchpad dashboards; treat them as orders of magnitude, not precise truth.) When 98-plus percent of launches are dead on arrival, the *entire game* of the survivors is to manufacture enough apparent activity to attract the next buyer before the music stops. Fake volume is not an edge case here; it is the default state of the screener.

The economics make memecoins the perfect fake-volume habitat. Gas on Solana is fractions of a cent, so round-tripping a token thousands of times costs almost nothing — the volume-bot rental market is cheapest exactly where it does the most damage. A freshly launched memecoin with no organic interest can be made to *look* like the hottest token on the chain for a few dollars, land on a trending list, and harvest real buyers who never run a single check. The unique-buyer test, run on these tokens, routinely returns single-digit or low-double-digit buyer counts behind six- and seven-figure "volume."

This is also where holder *concentration* compounds the danger. A memecoin can show a "growing" holder count while the deployer and a handful of insider wallets still hold the overwhelming majority of supply — so even genuine new buyers are buying into a setup where ten wallets can dump the entire float on them at any moment. Reading the holder distribution, not just the count, is the companion check; it is covered in depth in [supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration).

#### Worked example: a memecoin's \$600k volume from 9 wallets

A Solana memecoin is trending with \$600,000 of 24-hour volume and a market cap of \$2,000,000 — exciting numbers for a token a day old. Run the unique-buyer test on a Solana explorer or Birdeye: the buy side traces to **9 distinct wallets**, with the top 3 accounting for \$520,000 of the \$600,000 (87%). The holder count is 140 and has not moved in twelve hours despite the "volume." Trace the 9 buyers' funding: all 9 received their first SOL from the same wallet, which itself was funded by the token's deployer. Net flow across the \$600,000 of gross volume is roughly +\$8,000 — essentially round-trips. The verdict writes itself: \$600,000 of manufactured volume from 9 deployer-funded wallets, 140 holders, no demand. A buyer who saw "\$600k volume, trending" and aped in is exit liquidity for the deployer; a buyer who ran the four tests spent two minutes and kept their money. With ~1.4% of these tokens ever amounting to anything, the default action on an unverified memecoin is *avoid*, and the tests are how you find the rare exception.

## How to read it: a walkthrough on a "high volume" token

Let us run the whole battery on a hypothetical token — call it `TOKEN` — that just appeared at the top of a DEXScreener trending list with \$5,000,000 of 24-hour volume. We will use DEXScreener for the first glance, Dune for the rigorous counts, and a block explorer for the funding trace. The goal is a buy / avoid / investigate verdict in about ten minutes.

**Step 1 — DEXScreener first glance (30 seconds).** Open the pair page. The header shows \$5M volume, 2,400 transactions, and — the number that matters — **38 makers**. Right away the trades-per-buyer ratio is 2,400 ÷ 38 ≈ 63. Sixty-three trades per distinct wallet. That is not a crowd; that is a small group recycling. Provisional verdict: suspicious, keep going.

**Step 2 — Dune unique-buyer test (2 minutes).** Run the distinct-buyer query against `dex.trades`:

```sql
-- TOKEN buyers, last 24h
select
    count(*)                              as trades,
    count(distinct taker)                 as unique_buyers,
    sum(amount_usd)                        as buy_volume_usd,
    count(*) * 1.0 / count(distinct taker) as trades_per_buyer
from dex.trades
where token_bought_address = 0xTOKEN
  and block_time > now() - interval '24' hour
```

It returns: 1,400 buy trades, **22 unique buyers**, \$2,600,000 buy volume, 63.6 trades per buyer. Now sort buyers by volume — the top 5 wallets are \$2,300,000 of the \$2,600,000, **88% of all buying from 5 wallets.** This is a manufactured market. But finish the battery, because a clean kill needs all the evidence.

**Step 3 — Holder growth (2 minutes).** Pull the holder count over the last week from a Dune dashboard. It has been flat at ~1,900 the entire week despite the volume. No net new holders. Confirmed: the \$5M of "volume" produced zero growth in ownership.

**Step 4 — Funding trace (4 minutes).** Take the top 5 buyer wallets to the block explorer. Each one's first funding transaction traces back to the *same* wallet `0xSEED...`, which itself was funded from a single exchange withdrawal a few days before the volume started. Five "buyers," one funder. The funding graph collapses to a point.

**Step 5 — Net flow (1 minute).** Net buys for the token over 24h: +\$60,000 against \$5M of gross volume. The buying and selling nearly cancel — round-trips, not accumulation.

**The verdict:** every test fails the same way. 22 buyers (not a crowd), 88% from 5 wallets, all funded by one seed, holders flat, net flow ≈ zero. The \$5M volume is manufactured. **Action: avoid.** Total time: under ten minutes. The decision matrix below is the version of this you tape to your monitor.

![Decision matrix mapping unique buyers funding holder growth and net buys to fake readings organic readings and a buy avoid or investigate action](/imgs/blogs/detecting-fake-volume-vs-organic-demand-6.png)

The deeper point of the walkthrough is that **no single test is the answer; the convergence is.** Each test alone has a failure mode — a determined operator can beat the unique-buyer test, can launder the funding, can even airdrop to inflate raw holders. What no operator can beat is *all four at once while also growing a sticky holder base and leaving positive net flow.* Demand is over-determined; fakery is brittle. Make the token pass every check, not just the cheap one.

## The price-volume-holders triangulation

Step back from the individual tests to the framework that organizes them. The cleanest mental model for "is this demand real?" is a **triangulation of three independent signals: price, volume, and holders.** Real demand lights up all three together; any fake lights up at most two, and the mismatch is the tell.

![Matrix of real demand wash churn and listing pump scored across volume price holder count and a verdict column](/imgs/blogs/detecting-fake-volume-vs-organic-demand-7.png)

Read the matrix row by row, because each row is a *named pattern* you will see repeatedly:

- **Real demand:** volume rises (driven by broad buyers), price rises *and holds*, holder count grows with positive net buys. All three agree → genuine demand. This is the only row you buy.
- **Wash / bot churn:** volume is high and constant, but price is flat (no real directional pressure) and holders are flat (no new owners). Volume alone fired; price and holders did not follow → churn, no demand.
- **Listing pump:** volume spikes then collapses, price goes 3× then −70%, and the holder count is flat after the spike. Volume and price fired briefly; holders never did → distribution, not demand.

The discipline this enforces is the entire thesis of the post in one rule: **never act on one signal.** A trader who buys on "high volume" is acting on one corner of the triangle. A trader who buys on "price is pumping" is acting on another. Real demand requires all three corners — and the third corner, holders, is the one almost nobody checks and the one that is hardest to fake. The holder count is your tiebreaker: when volume and price disagree about whether something is real, the holder line settles it. This same "triangulate independent signals, distrust any single green number" discipline is what keeps you from being the exit liquidity in [the perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain), where a single shiny metric — a "smart" wallet's buy — substitutes for the whole picture.

It helps to walk the *disagreements* explicitly, because that is where the framework earns its keep. If **volume is high but price is flat**, the volume is two-sided churn — wash bots or a market maker round-tripping — and there is no directional demand; you do not buy. If **price is up but volume is thin**, a small amount of buying moved an illiquid token, which says more about the lack of sellers than the presence of demand; that is a low-float setup, not a demand wave, and it reverses just as easily on the first real seller. If **price and volume are both up but holders are flat**, that is the most seductive trap — a coordinated pump where insiders are marking the price up while distributing to a small recycling set; the flat holder line is the giveaway that no durable demand is forming under the move. Only when **all three rise together** — broad volume, holding price, growing holder count on positive net buys — do the corners corroborate one another, and corroboration is the whole point. A single corner is a claim; three corroborating corners are a case.

A practical note on *which* corner to weight when you are short on time: lead with **holders and net flow**, because they are the corners hardest to fake and the ones almost everyone skips. Price and volume are the two numbers every chaser is already staring at, which is precisely why they are the two most manufactured. The edge is not in seeing the volume — everyone sees the volume. The edge is in being the one who checked whether anyone actually bought and kept the thing.

## Common misconceptions

**"High volume means high demand."** This is the entire mistake the post exists to correct. Volume measures motion, not arrival. Twelve wallets round-tripping \$5M a day produce identical volume to 800 people buying \$5M, and only the second is demand. Volume is necessary for a liquid market but not remotely sufficient evidence of demand. Always ask "from how many buyers, funded from where, growing the holder count by how much?" — not "how big?"

**"It's on-chain, so the volume is real."** On-chain volume is *settled* — the transactions genuinely happened — but "real transactions" is not "real demand." Wash trades are real transactions between wallets the same operator controls. The on-chain advantage is not that the volume is automatically honest; it is that even *dishonest* volume leaves a traceable trail (the wallets, the funding, the flat holder count) you can audit. On-chain doesn't make volume trustworthy; it makes it *checkable*.

**"A market maker's volume is bullish."** A market maker is delta-neutral by design — they end the day holding roughly what they started with, earning the spread on the flow. Their \$8M of volume can be \$0 of net demand. MM volume signals *liquidity* (good — you can get in and out), not *demand* (a reason to buy). The discriminator is net flow: churn round-trips to ≈ zero; accumulation leaves a positive net position.

**"The listing spike proves people want it."** During a CEX-listing spike, real buyers really do pile in — so the unique-buyer test passes. But the spike's buyers are usually exit liquidity for insiders who accumulated cheaply pre-listing and are selling into the FOMO. The buyers are chasers, not accumulators; the holder count barely grows; and once insiders finish distributing, price fades 50–80%. A spike with insider exchange-inflows and a flat holder count is distribution wearing a demand costume.

**"More holders is always better."** Raw holder *count* can be inflated by airdropping tokens to thousands of dead or mercenary wallets — that bumps the count without any buying behind it. What matters is holder *growth driven by net new buying* (a distribution event with zero buys is not demand) and holder *quality* (a growing count is worthless if the top 10 wallets still hold 90% and can dump at will). Pair holder growth with net flow and concentration, never read it alone.

## The playbook: what to do with it

The whole post compresses into one rule and one checklist. The rule: **do not buy "high volume" — buy confirmed organic demand.** Volume gets you to look; demand gets you to buy. The checklist turns a faked headline number into a defensible decision.

**The four-test battery (run before any buy on a "high volume" token):**

1. **Unique-buyer test.** Pull buys over your window; count *distinct wallets*, not trades. Compute trades-per-buyer and the top-5-wallet share of volume.
   - *Signal:* trades-per-buyer in the dozens, or top-5 wallets > 80% of volume.
   - *Read:* a few wallets are manufacturing the volume.
   - *Action:* **avoid.** *False positive:* a brand-new token where a couple of large genuine buyers dominate early — check funding diversity to disambiguate.

2. **Buyer-funding diversity.** Trace the top buyers to their first funding transaction.
   - *Signal:* all buyers trace back to one seed wallet or one fresh exchange account.
   - *Read:* one operator wearing many wallets.
   - *Action:* **avoid.** *False positive:* a legitimate launchpad or a single large fund deploying through sub-wallets — context (a known entity label) resolves it.

3. **Holder growth vs volume.** Overlay the holder count on the volume.
   - *Signal:* volume high, holder count flat (or falling).
   - *Read:* churn (flat) or distribution (falling), not demand.
   - *Action:* **avoid.** Demand requires the holder line to *climb with* the volume. *False positive:* an airdrop event spiking the count — confirm with net buys, not raw count.

4. **Net new buying.** Compute buys minus sells.
   - *Signal:* net flow ≈ zero against large gross volume.
   - *Read:* wash/MM round-trips; the position isn't growing.
   - *Action:* **avoid.** Real accumulation leaves a *positive* net position. *False positive:* none worth worrying about — positive net flow with broad buyers and growing holders is the green light.

**The triangulation rule (the synthesis):** buy only when **price, volume, and holders all agree** — volume up, price up *and holding*, holder count growing on positive net flow, funded from many independent sources. Any single signal alone is noise. All four tests passing plus the triangle lighting up green is the only configuration that earns a buy, and even then it earns "investigate further," not "all in."

**The one-line discipline:** *count the buyers, trace their money, and watch the holder line — the volume number is the bait, not the catch.*

## Further reading & cross-links

- [Detecting wash trading](/blog/trading/onchain/detecting-wash-trading) — the deep dive on the mechanism behind most fake volume: how one operator round-trips through two wallets, and how to cluster a wash ring back to its funder.
- [Pump-and-dump and coordinated buying](/blog/trading/onchain/pump-and-dump-and-coordinated-buying) — the coordinated-distribution playbook the CEX-listing pump is a special case of.
- [Active addresses and network activity](/blog/trading/onchain/active-addresses-and-network-activity) — the network-wide version of the unique-buyer idea: counting distinct participants instead of raw transaction volume.
- [Supply distribution and holder concentration](/blog/trading/onchain/supply-distribution-and-holder-concentration) — reading the holder *distribution*: why a growing holder count is worthless if the top wallets still hold most of the supply.
- [Following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets) — the accumulation signature (positive net flow, sticky holdings) that is the opposite of MM churn.
- [Exchange flows: inflows and outflows](/blog/trading/onchain/exchange-flows-inflows-and-outflows) — reading the insider-selling leg of a listing pump as exchange inflows.
- [The perils of copy-trading on-chain](/blog/trading/onchain/the-perils-of-copy-trading-onchain) — why a single shiny on-chain metric (a "smart" buy) is the same trap as a single big volume number.
