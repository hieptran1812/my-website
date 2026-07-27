---
title: "Wintermute: Inside Crypto's Algorithmic Powerhouse"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A build-from-zero profile of Wintermute — how one London algorithmic trading firm came to quote thousands of tokens across CeFi order books, OTC blocks, DeFi pools and options, what its footprint looks like in the book you trade, how to trace it on-chain without overclaiming, and what the September 2022 vault drain taught everyone about hot-wallet hygiene."
tags: ["crypto", "market-makers", "wintermute", "crypto-players", "otc-trading", "defi-liquidity", "on-chain-analysis", "algorithmic-trading", "order-book", "inventory-risk"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 48
---

> [!important]
> **TL;DR** — Wintermute is a London-founded algorithmic trading firm that makes markets in crypto on four surfaces at once — public exchange order books, bilateral OTC blocks, on-chain DeFi pools, and options — and nets all of it into a single inventory book. It is a volume business, not a directional one, and its scale is the whole point.
>
> - **Who:** founded in July 2017 by Evgeny Gaevoy, who previously ran Optiver's European ETF business. Headquartered in London with offices in Singapore and New York.
> - **How big:** in its *OTC Markets 2025* report (13 January 2026) the firm described itself as running **over \$15 billion in average daily trading volume** across **60+ centralized and decentralized exchanges**. A year earlier the same series put it at 50+ venues, 1,000+ assets and daily volume "frequently exceeding \$5 billion."
> - **Independent confirmation exists:** Robinhood's SEC Form 10-Q for the first quarter of 2025 names Wintermute as accounting for **11% of its transaction-based revenues** — Robinhood only names counterparties at or above 10%.
> - **How it earns:** the bid-ask spread repeated millions of times, maker rebates, OTC block markups, and — for token projects — a *loan-plus-call-option* deal that transfers real upside to the desk. That last structure is the one worth understanding before you read any "market maker dumped my token" thread.
> - **The scar:** on **20 September 2022** its DeFi vault was drained of about **\$160 million** after the private key to a "vanity" hot wallet was reconstructed. The reported cause was a flaw in the open-source `Profanity` address generator; Wintermute said publicly it remained solvent with roughly twice that amount left in equity.
> - **The number to remember:** a market maker's on-chain deposit is a *custody* event, not a *trade*. Everything you can prove by watching Wintermute's wallets stops at "these coins moved" — and most of the internet does not stop there.

Somewhere in the last month you almost certainly traded against Wintermute and never knew it. If you bought a mid-cap token on a centralized exchange, if you swapped through a DEX aggregator, if you tapped "buy" in a retail app — the firm on the other side, or one layer behind it, was plausibly this one. It has no retail product, no consumer brand, and almost no reason to talk to you. It just quotes.

This post is a profile of that firm, written for someone with no finance background. We will build every term from zero — bid, ask, spread, inventory, RFQ, notional, delta, basis point — before we lean on it, and we will ground every mechanism in a worked example with round numbers you can check in your head. By the end you should be able to look at a token's order book, or at a labelled wallet on a block explorer, and reason carefully about what a firm like Wintermute is doing and — just as importantly — about what you *cannot* conclude.

The diagram below is the mental model for everything that follows. Four different trading surfaces, four different kinds of counterparty, four different risk profiles — all feeding one net position that has to be managed as a single thing.

![Wintermute quotes on four surfaces — CeFi order books, OTC blocks, on-chain pools and options — and nets every fill into a single inventory and risk book hedged with perpetual futures.](/imgs/blogs/wintermute-the-algorithmic-powerhouse-1.webp)

This is a case study companion to the series hub, [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers), and it sits directly on top of [What a Crypto Market Maker Actually Does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does), which builds the quoting business from first principles. If the words "spread" and "inventory" are new to you, skim that post first; here we assume the mechanics and climb up to a specific, named, real firm. This is educational, not financial advice, and nothing here is a claim about what Wintermute will do next.

## Foundations: the vocabulary you need

Before we can say anything precise about a market maker, we need about a dozen words. Let us get them all out of the way, in plain language, with a concrete number attached to each.

### The order book, in one paragraph

An **order book** is a public list of everyone's standing offers to trade a particular asset on a particular exchange. Offers to buy are **bids**; offers to sell are **asks** (or **offers**). The highest bid and the lowest ask are the **best bid** and **best ask** — together, "the top of the book." The gap between them is the **spread**. If the best bid is \$99.50 and the best ask is \$100.50, the spread is \$1.00 and the **mid-price** — the average of the two — is \$100.00.

The amount available at each price level is called **depth**. A book with 50,000 tokens resting within a cent of the mid is "deep"; a book with 300 tokens is "thin." Depth is what determines whether a big order can execute without moving the price.

Two units you will see everywhere. A **basis point** (bp) is one hundredth of one percent — 0.01%. So 5 bps is 0.05%, and 100 bps is 1%. And **notional** is the dollar value of a trade rather than its unit count: buying 20,000,000 tokens at \$1.00 is \$20 million of notional.

### Market maker, maker, taker

A **market maker** is a firm that continuously posts both a bid and an ask on the same asset at the same time and earns the spread between them. It is not betting on direction. It is renting out *immediacy*: you get to trade right now, at a known price, and you pay a sliver for the privilege.

Two ways to place an order, and the difference decides who pays whom:

- A **limit order** names a price and waits in the book. You control the price, not whether you trade. This *provides* liquidity, and you are the **maker**.
- A **market order** names a size and takes whatever is available now. You are guaranteed to trade but you pay the prevailing price and cross the spread. This *consumes* liquidity, and you are the **taker**.

Most exchanges charge takers more than makers, and at the top volume tiers some venues pay makers a small **rebate** — a negative fee — for leaving resting liquidity on the book.

### Inventory and delta

Every fill leaves the maker holding something it did not choose to own. If its bid gets hit, it now owns tokens. If its ask gets lifted, it is now **short** — it owes tokens it must buy back. The running net position is its **inventory**, and it is the residue of whoever happened to trade against it.

Inventory is dangerous because the price moves while you hold it. The sensitivity has a name from the options world: **delta**, or directional exposure, which is simply how much you make or lose per \$1 move in the price. A maker long 5,000 tokens has a delta of +5,000: a \$1 rise earns \$5,000, a \$1 fall costs \$5,000. A maker with zero inventory is **flat** — zero delta — and genuinely does not care which way the price goes. Staying near flat is the entire craft, and the mechanics are the subject of [Inventory Risk, Hedging, and Delta Neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality).

The usual tool for getting flat without selling the tokens is a **perpetual futures contract** — a "perp." It is a derivative that tracks the token's price with no expiry date, so a maker who is long 5,000 tokens on the spot market can short 5,000 tokens' worth of perp and carry roughly zero net exposure while still holding the coins.

### OTC, RFQ, and "block"

Not all trading happens on a public order book. **OTC** — over the counter — means a bilateral trade agreed directly between two parties, with no exchange in the middle and no print on the public tape. The usual mechanism is **RFQ**, request for quote: you tell a desk what you want to trade and in what size, and it answers with a single firm price, usually good for a few seconds. A **block** is simply a large trade done in one go rather than sliced up.

The reason this exists is the reason most of this post exists: on a public book, a large order *moves the price against itself*. RFQ lets you find out the price *before* you commit, and lets the desk price the risk of absorbing your whole size in one hit.

### CeFi, DeFi, AMM, and MEV

**CeFi** — centralized finance — means exchanges that hold your money and run a matching engine: Binance, Coinbase, OKX, Bybit. **DeFi** — decentralized finance — means smart contracts on a public blockchain that anyone can trade against directly from their own wallet.

The dominant DeFi trading design is the **automated market maker (AMM)**, of which Uniswap is the canonical example. Instead of an order book, there is a **pool** holding two tokens, and a formula that quotes a price against the pool's balances: as buyers drain one token, the formula raises its price. Anyone who deposits into the pool becomes a **liquidity provider (LP)** and earns a share of the trading fees. It is the same spread-inventory-adverse-selection business as a human maker, expressed as a curve. The DeFi-native name for the loss an LP takes to better-informed traders is **impermanent loss**.

One DeFi-only hazard: **MEV**, maximal extractable value. Because pending transactions on a public blockchain are visible before they are confirmed, specialised bots ("searchers") can reorder, front-run or sandwich them for profit. A market maker quoting on-chain is quoting into a crowd that can see its orders before they execute.

### Adverse selection: the fear underneath everything

The last piece. **Adverse selection** is the risk that the person taking your quote knows something you don't. Most people who trade against a resting quote are trading for reasons unrelated to the next tick — rebalancing, paying for something, acting on a hunch. Some are trading *precisely because* they know the price is about to move, and they will always pick the side that hurts you.

The maker cannot tell them apart at the moment of the trade. All it can do is price the spread wide enough that the harmless majority pays for the dangerous minority. This is why thin, obscure tokens have wide spreads and deep, liquid ones have razor-thin ones — the spread is a break-even cost, not a profit margin.

That is the whole vocabulary. Now the firm.

## 1. Who Wintermute is

Wintermute was founded in **July 2017** in London by **Evgeny Gaevoy**, along with co-founders from a similar background. Gaevoy's résumé is the tell: before crypto, he built and ran **Optiver's European ETF business** — one of the largest in the EU. Optiver is a Dutch high-frequency proprietary trading firm, and an ETF desk is, structurally, exactly what Wintermute became: a machine for quoting two-sided prices in thousands of instruments simultaneously, hedging the resulting inventory in a correlated market, and living off a spread measured in fractions of a basis point.

That heritage matters more than the crypto part. Nothing in Wintermute's business model was invented for crypto. What crypto supplied was a market where the same machinery could be pointed at assets nobody was making markets in yet, on venues with no incumbent, twenty-four hours a day.

![Wintermute's arc from a 2017 London startup to a firm reporting over $15B of average daily volume, with the June and September 2022 security failures in the middle.](/imgs/blogs/wintermute-the-algorithmic-powerhouse-2.webp)

### Scale, with dates attached

Scale figures for private trading firms are self-reported, so the honest way to state them is with the source and the date. Here is what the record actually says.

| As of | Claim | Source |
|---|---|---|
| Jan 2025 | 50+ centralized and decentralized venues; 1,000+ digital assets quoted; daily volumes "frequently exceeding \$5 billion"; record single-day OTC spot volume of about \$2.24bn in Nov 2024 | Wintermute, *OTC 2024 in review & 2025 outlook* |
| May 2025 | 11% of Robinhood's transaction-based revenues in Q1 2025 | Robinhood Markets, SEC Form 10-Q |
| Jan 2026 | "Over \$15 billion in average daily trading volume"; liquidity provider across "60+ centralized and decentralized exchanges" | Wintermute, *OTC Markets 2025* report / press release, 13 Jan 2026 |
| May 2026 | "Over \$3.5 trillion in annual trading volume"; begins quoting prediction markets | Wintermute announcement, 29 May 2026 |

Two things are worth noticing. First, the numbers moved a lot in eighteen months — from "frequently exceeding \$5 billion" a day to "over \$15 billion" average — so any single figure you see quoted without a date is probably stale. Second, the venue count is quoted as 50+, 60+ and 70+ in different materials from different months; these are marketing-page figures with no audited definition of "venue." Treat the order of magnitude as the information and the precise integer as noise.

The Robinhood filing is the most interesting line in that table, because it is the only one Wintermute did not write. A US-listed company's 10-Q is a legal document, and Robinhood's policy is to name counterparties that represent 10% or more of its transaction-based volumes. In the first quarter of 2025 it named three: Citadel Securities at 12%, B2C2 at 12%, and Wintermute at 11%. That is external, auditable confirmation that a London prop shop most retail traders have never heard of was, at that moment, one of the three largest sources of a major US brokerage's trading revenue.

### The other arms

Two adjacent businesses are worth knowing about because they change how you should read the firm's incentives.

**Wintermute Ventures**, launched in 2020, is the firm's venture arm; by its own description it has backed more than 100 projects, with third-party trackers counting somewhere between 105 and 123 investments depending on methodology. A market maker that also holds equity and tokens in the projects it quotes is a structurally different animal from one that only quotes — a tension we come back to in the misconceptions section, and one explored properly in [Designated versus Principal Market Making](/blog/trading/crypto-players/designated-versus-principal-market-making).

**The options desk.** Wintermute quotes derivatives from vanilla swaps through to exotic structures on long-tail altcoins and crypto indices, distributed partly through **Paradigm**, the largest multi-dealer OTC options platform in crypto. In its 2025 OTC review the firm reported that options notional volumes ran roughly four times higher at year-end than year-start, with trade counts more than doubling — and, notably, that flow was for the first time dominated by systematic yield and risk-management strategies rather than one-off directional bets.

## 2. The operating model: four surfaces, one book

Here is the thing that makes a firm like this hard to reason about from the outside: it is not doing one business four times. It is doing four *different* businesses whose risks partially cancel, and treating the residue as a single position.

### Surface one — CeFi order books

This is the classic job. The firm rests bids and asks on public exchange order books across a very large number of venues and tokens, refreshes them continuously as prices move, and collects the spread when both sides get filled. Revenue per round trip is tiny; the number of round trips is astronomical.

The economics are covered in depth in the [mechanics companion](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does), but the shape is worth restating: gross spread income is roughly `N × s`, the number of completed round trips times the spread captured on each. The art is making `N` enormous while keeping `s` small enough that you win the queue against every other maker quoting the same pair.

#### Worked example: what half a basis point on \$15 billion looks like

Let us take the firm's own January 2026 figure — over \$15 billion of average daily trading volume — and ask what a plausible spread capture does to it. *These capture rates are illustrative, not reported figures; Wintermute does not publish margins.*

Suppose the blended effective capture across all that flow is **0.5 basis points** — half of one hundredth of a percent. That is a deliberately brutal assumption: on the most liquid pairs, real capture is often thinner than that, and competition grinds it down constantly.

- Daily gross: 0.00005 × \$15,000,000,000 = **\$750,000 per day**.
- Over a 365-day year (crypto does not close): 365 × \$750,000 ≈ **\$274 million per year** of gross spread income.

Now halve the capture to **0.25 bps** — 0.000025 as a decimal — which is entirely plausible for a book dominated by BTC and ETH:

- Daily gross: 0.000025 × \$15,000,000,000 = **\$375,000 per day**, or about **\$137 million per year**.

The intuition this teaches: at this scale the *spread* is almost an irrelevance as a number — nobody would build a company to earn a quarter of a basis point — but as a *rate applied to fifteen billion dollars a day*, it is a large business. And it also means the firm's revenue is far more sensitive to volume than to skill at any individual quote. When crypto volumes halve, so does the top line, regardless of how good the models are.

### Surface two — the OTC desk

The OTC desk is a different product sold to a different customer. A fund that needs to move \$20 million of a mid-cap token does not want to walk the public book; it wants one price, for the whole size, now, without telling the market. It sends an RFQ and Wintermute answers with a firm quote, usually automated, sometimes with a human trader on it for the biggest or weirdest tickets. Per the firm's own materials the desk covers spot and derivatives on more than 250 digital assets.

The desk's margin is the difference between the price it quotes the client and the price at which it can lay the risk off. Everything difficult is in that second half.

#### Worked example: an OTC block, priced and hedged

*Illustrative numbers throughout.* A fund wants to **sell** 20,000,000 units of a token that is trading at a \$1.000 mid on the public book.

1. **The desk quotes.** It offers \$0.992 for the whole size — 80 bps below mid. The fund accepts. The desk has now bought 20,000,000 tokens for \$19,840,000 and is suddenly **long 20 million tokens** it did not want.
2. **Immediately hedge the direction.** The desk shorts \$20 million notional of the token's perpetual future. Its net delta goes to roughly zero: if the token falls 5%, the spot inventory loses about \$1,000,000 and the perp short gains about \$1,000,000.
3. **Work out of the inventory.** Over the next several hours it feeds the 20 million tokens into the public book as passive resting sell orders, buying back the perp short as it goes. Say it achieves an average exit of \$0.998 — 20 bps below where the mid started, because its own selling pushed the price down a little.
4. **Tally.** Bought at \$0.992, sold at an average of \$0.998. Gross: 20,000,000 × \$0.006 = **\$120,000**. Subtract exchange fees, the perp's periodic **funding rate** (the small payment longs and shorts exchange to keep the perp tethered to spot), and the slippage on putting the hedge on — call it \$35,000 all-in. Net: about **\$85,000** on a \$20 million ticket, roughly 4 basis points.

The intuition: the OTC desk is not paid for having a view on the token — it is paid for being willing to own \$20 million of it for six hours while it finds the other side, and its entire skill is in guessing, at quote time, how expensive those six hours will be.

Notice what happens if it guesses wrong. If the token drops 3% during the unwind for reasons unrelated to the desk's own selling, the perp hedge covers the directional loss, but the *execution* loss — selling into a falling book at worse and worse prices — is real, and it can dwarf the \$120,000 of gross spread. This is why quotes get wider before a scheduled unlock, an exchange listing, or a macro data release: the desk is pricing the difficulty of the exit, not the current price.

### Surface three — on-chain and DeFi

Wintermute was unusually early to trade on-chain, and this is where its footprint is most visible to you, because everything settles on a public ledger.

There are three distinct on-chain activities and they get conflated constantly:

- **Providing liquidity to AMM pools.** Depositing token pairs into Uniswap-style pools and earning fees. This is passive quoting: the curve does the work, and the LP eats impermanent loss when arbitrageurs pick the pool off.
- **Taking liquidity on-chain.** Trading *against* pools, usually to arbitrage a price difference between a DEX and a centralized exchange, or to rebalance inventory that ended up on the wrong chain.
- **Quoting on-chain RFQ.** Newer DEX aggregator designs let professional makers stream firm quotes that a user's swap can fill against directly, rather than routing through a pool. This is the OTC model wearing a DeFi costume, and it lets the maker price each trade instead of committing a curve.

All three have to be hedged against the same central book, which is a genuinely hard engineering problem: a fill on Ethereum settles in seconds to minutes, and the hedge has to go on before that, against a CeFi venue on a different clock.

### Surface four — options and derivatives

The last surface changes the risk vocabulary. On a spot book you manage **delta** — exposure to price. On an options book you also manage **vega** (exposure to changes in implied volatility, i.e. how much the market expects the price to swing) and **gamma** (how fast your delta changes as the price moves). A desk can be perfectly delta-neutral and still lose a great deal of money because volatility repriced.

This is why the options business is structurally different from the spot business rather than an extension of it, and why relatively few crypto firms do both at scale.

![The four surfaces differ in counterparty, price-setting mechanism, public visibility and dominant risk — which is why the same firm looks completely different depending on where you meet it.](/imgs/blogs/wintermute-the-algorithmic-powerhouse-3.webp)

The matrix above is the single most useful thing to internalise about a firm like this. When someone says "Wintermute is doing X," the first question is always: *on which surface?* A claim that is trivially observable on-chain is invisible on an OTC desk, and a quoting behaviour that is obvious in a CeFi ladder leaves no trace anywhere else.

## 3. How the money actually arrives

There are four revenue lines, and they are not equally understood.

### Line one: the spread

Covered above. Small per trade, enormous in aggregate, competed down relentlessly. The most honest way to think about it is as a *fee for immediacy* charged to whoever is impatient.

### Line two: maker rebates

Exchanges want deep books, so they tilt fee schedules to reward resting orders. At the top volume tiers some derivatives venues run maker fees that are actually negative — the exchange pays you to post. Rates change constantly and vary by venue and tier, so treat the structure rather than any specific number as the durable fact.

The strategic effect is subtle and important: a rebate **lowers the maker's break-even spread**. If a venue pays you 0.5 bps to post, you can quote half a basis point tighter than a desk that pays to trade, and still make the same money. That is how a large maker wins the queue — not by being cleverer about fair value, but by having a cost base that lets it quote inside everyone else.

### Line three: the OTC markup

Covered in the worked example above. Note it is a genuinely different revenue *shape*: lumpy, relationship-driven, and dependent on being able to warehouse risk. A firm with a big balance sheet can quote tighter on a \$20 million block than one without, because it can afford to hold the position longer and unwind it more patiently.

### Line four: the token-project deal

This is the one that generates all the drama, and it deserves a careful, non-hysterical explanation.

A new token project has a problem: on day one there is no liquidity, so the spread is enormous, every trade lurches the price, and the token looks broken. The project needs a professional maker to stand there and quote. But the maker needs *tokens* to quote with — you cannot post a two-sided market without inventory on both sides.

So the project lends the maker tokens. The dominant structure in crypto is a **loan plus call option**:

- The project lends the maker a block of tokens for a fixed term, at zero or near-zero interest. Wintermute has publicly described using interest-free loans, and in its August 2023 governance proposal to Yearn Finance it asked for 350 YFI on a 12-month loan at a nominal **0.1%** — a rate Gaevoy said existed for legal and accounting reasons and should be read as effectively zero.
- Alongside the loan, the maker receives a **call option** on those tokens: the right, but not the obligation, to buy them at a pre-agreed **strike price** at the end of the term.

That second leg is where the economics live.

![Below the strike the market maker's option leg is worth nothing and only spread income remains; above it, each dollar of token price is a dollar of upside on the full option size.](/imgs/blogs/wintermute-the-algorithmic-powerhouse-4.webp)

#### Worked example: the loan-plus-call-option deal

*Illustrative numbers.* A project engages a maker for twelve months.

- **Loan leg:** the project lends **10,000,000 tokens** at 0% interest. The maker uses them as inventory to quote a two-sided market. At the end of the term it must return 10,000,000 tokens (or their agreed cash equivalent).
- **Option leg:** the maker also gets a call option on those same 10,000,000 tokens, struck at **\$0.50** — roughly the token's price when the deal was signed.

Now run three scenarios at the end of the twelve months:

| Token price at term end | Option value | What the maker does |
|---|---|---|
| \$0.30 | 10,000,000 × \$0 = **\$0** | Option expires worthless; returns the tokens; keeps only its spread income |
| \$0.50 | 10,000,000 × \$0 = **\$0** | Exactly at the strike; no intrinsic value |
| \$1.50 | 10,000,000 × (\$1.50 − \$0.50) = **\$10,000,000** | Exercises: buys 10M tokens at \$0.50, sells at \$1.50 |

The intuition: the option leg is **free upside with no downside**. The maker paid nothing for it, so at worst it is worth zero. That is not a scandal by itself — it is compensation, and the project chose to pay in optionality rather than in cash, which is often exactly what a cash-poor project wants.

But look at where the \$10,000,000 comes from. It comes from the project's own token supply. Every dollar the maker makes on the option leg is a dollar of dilution borne by everyone else holding the token. And that creates the incentive problem the crypto community argues about endlessly: a maker holding a free call option is *not indifferent to price* in the way a pure spread business is. Its position has positive delta by construction.

There is a second, sharper worry. If the maker also *borrowed* 10 million tokens it can sell, and holds a call struck at \$0.50, then selling the borrowed tokens now and buying them back cheaper later is profitable — and so is exercising the option if the price runs. A structure that pays off in both directions is a structure whose holder has no strong reason to want a stable price. Whether any given firm behaves that way is a separate, evidentiary question; the point is that the *structure* creates the incentive, which is why an increasing number of projects now push for a plain **retainer** — a flat monthly fee, no tokens, no options, no dilution — instead. Wintermute itself has publicly argued for more transparency around these arrangements.

The full anatomy of these deals, including how to read one in a project's disclosures, is in [The Loan-Plus-Options Deal: How Market Makers Get Paid](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid).

### An aside on the Yearn episode

In August 2023 Wintermute posted a proposal to Yearn Finance's governance forum asking to borrow 350 YFI tokens — worth roughly \$2.1–2.2 million at the time — for twelve months at 0.1% interest, offering in exchange to deposit CRV with the protocol and to make markets in YFI. The community response was hostile; accusations of an intended pump-and-dump circulated, Gaevoy publicly called the accusations "flattering," Wintermute revised the proposal, and the vote failed at the end of the month.

It is a useful episode precisely because nothing illegal or even unusual happened. A market maker proposed a standard structure; a token community read the structure, understood who bore the downside, and said no. That is what an informed counterparty looks like, and it is the outcome this whole post is trying to make possible for you.

## 4. How this shows up in the book you trade

Everything above happens off-screen. What you see is the shadow.

### Reading the ladder

When a professional maker is quoting a pair, three things are true at once: the spread is tight, the depth near the top of the book is substantial, and — the part most people miss — the depth *refreshes*. Take out the top level and a new one appears within milliseconds. That refresh is the signature of an algorithmic quoting engine rather than a pile of static retail limit orders.

When the maker steps back, all three degrade together. The spread gaps, the top-of-book size shrinks, and — critically — the replenishment stops. A ladder that looks thin but refills instantly is being made; a ladder that looks thin and *stays* thin is not.

And the maker can flip between those states in milliseconds, because its orders are resting limit orders and cancelling them is free. It does exactly that when the risk of being picked off spikes: before a scheduled announcement, during a volatility burst, when its models detect informed flow arriving. The liquidity evaporates precisely when you would most want it. That is not misconduct; it is the rational response of a firm that would otherwise be the last standing offer in front of a freight train.

### Which tokens get supported at all

A maker quotes a token only if the expected spread income exceeds the expected cost of adverse selection plus inventory carry plus the operational cost of listing it. That calculation is why coverage is so lopsided. Wintermute's own 2025 OTC review is unusually candid about the concentration: BTC and ETH accounted for **49% of total notional in 2025**, down from **54% in 2023**, with blue-chip tokens outside those two picking up about 8 percentage points of share over the two years — and the rest of the long tail failing to hold any of it. In the same report, the average altcoin rally lasted **19 days in 2025, down from 61 days in 2024**, which the firm attributed to there being insufficient liquidity to carry a narrative.

Read that as a market maker's confession about its own coverage. The tail is quoted, but it is quoted thinly and defensively, because the tail is where adverse selection is worst and the exit is narrowest.

### Sweeping versus asking

The most practical consequence of a maker's existence is that you usually have a choice about *how* you access it.

![Buying $20M by sweeping the book costs about $310,000 against the starting mid and prints publicly; an RFQ block at a quoted $1.008 costs about $160,000 and never touches the tape.](/imgs/blogs/wintermute-the-algorithmic-powerhouse-5.webp)

#### Worked example: sweeping the book versus asking for a quote

*Illustrative numbers.* You want to buy \$20 million of a mid-cap token trading at a \$1.000 mid.

**Route A — sweep the public book.** You want 20,000,000 tokens, so you send a market buy for that size. It consumes the resting asks level by level: some at \$1.001, more at \$1.004, more at \$1.010, and the last tranche at \$1.031. Suppose the volume-weighted average fill comes out at **\$1.0155**.

- Total paid: 20,000,000 × \$1.0155 = \$20,310,000.
- Cost against the starting mid: 20,000,000 × (\$1.0155 − \$1.0000) = **\$310,000**, or 1.55%.
- Every one of those fills printed on the public tape, so anyone watching now knows a large buyer just arrived.

**Route B — request a quote.** You ask an OTC desk for a price on 20,000,000 tokens. It answers **\$1.008**, firm for about five seconds. You accept.

- Total paid: 20,000,000 × \$1.008 = \$20,160,000.
- Cost against the mid: 20,000,000 × \$0.008 = **\$160,000**, or 0.80%.
- Nothing printed. The desk now owns the problem of getting flat.

The intuition: **you paid the desk \$160,000 to take a \$310,000 problem off your hands, and both of you came out ahead.** The desk profits if it can unwind for less than \$160,000 of cost; you saved \$150,000 versus doing it yourself, plus the information leakage you avoided.

And now the honest caveat, because this is where beginners get it wrong: Route B is only cheaper because the desk is *better at the exit than you are*. It can spread the unwind across sixty venues, net it against opposite flow from other clients, and hedge the direction with perps while it works. If you had those tools, you would not need the desk. The \$160,000 is the price of not having them — and if the desk misprices its exit, that \$160,000 is not profit, it is a loss.

## 5. How to trace Wintermute on-chain

Here is the section this post exists for. Because a large share of Wintermute's activity settles on public blockchains, you can watch a meaningful slice of it — and because you can watch it, an entire genre of misleading commentary has grown up around doing so badly.

What follows is a defender's method: how to build a bounded, defensible read of a market maker's on-chain footprint, and — more importantly — where to stop.

![Six steps take you from a public entity label to a bounded claim about inventory; the last two steps, bounding the inference and cross-checking the tape, are the ones most commentary skips.](/imgs/blogs/wintermute-the-algorithmic-powerhouse-6.webp)

### Step 1 — Start from a public entity label

Blockchain analytics platforms maintain human-readable labels mapping wallet clusters to real-world entities. **Arkham Intelligence** publishes a `Wintermute` entity page and labels individual addresses (it also maintains a separate `Wintermute Hacker` entity for the 2022 exploit addresses). **Nansen** maintains a `Wintermute Trading` entity in its wallet profiler.

Two caveats before you build anything on top of a label:

- **Labels are inferences, not disclosures.** They are built from funding patterns, transaction graph clustering, exchange deposit-address attribution and occasional public confirmation. They are usually right and occasionally wrong, and they are rarely complete — a firm running hundreds of addresses across a dozen chains will have some that no one has labelled.
- **Check for conflicts.** Wintermute has been publicly reported as a partner and investor in Arkham's ecosystem, which does not make Arkham's Wintermute labels wrong but is exactly the kind of thing you should know before treating one vendor's labelling as ground truth. Cross-check across at least two providers.

### Step 2 — Enumerate the wallets and split them by function

A market maker's address set is not homogeneous. Sort what you find into at least four buckets, because they mean different things:

| Wallet type | What it does | What a movement means |
|---|---|---|
| Exchange deposit addresses | Unique per-venue addresses that route funds into a CEX account | Custody moving *into* trading availability |
| Hot / settlement wallets | Operational wallets that pay out, receive, and rebalance across chains | Plumbing; usually says nothing directional |
| DEX LP positions | Tokens committed to AMM pools | Passive quoting capital, genuinely deployed |
| Treasury / cold storage | Long-dated holdings, rarely moved | A move here is a real signal because it is rare |

The distinction that matters most is the first versus the third. A transfer into an exchange deposit address changes *where the coins can be traded*. A change in an LP position changes *how much liquidity the firm is providing on-chain*. Those are different claims and they are constantly conflated.

### Step 3 — Read cadence, not single transactions

A single large transfer is almost never informative. Market makers move inventory around continuously; a 400 BTC hop between two of their own wallets is Tuesday.

What is informative is **net flow measured over a window**, and its change relative to the firm's own baseline. Compute, per day: total in-flow to exchange deposit addresses, minus total withdrawals back out, per asset. Then compare that net number to a trailing average — say the previous thirty days. A day that is two or three times the normal net direction is worth looking at. A day that is inside the normal range is noise no matter how big the absolute number sounds in a headline.

#### Worked example: reading a week of exchange deposits

Take a real, dated episode, and then read it properly.

Over **31 December 2025**, wallets labelled by Arkham as Wintermute's sent **1,518.6 BTC** to Binance and withdrew **305.5 BTC**, for a net deposit of **1,213 BTC** — about **\$107 million** at prices that day. Deposits continued into 1–2 January 2026, for a further net of roughly 1,441 BTC across the two days (source: CryptoSlate, Jan 2026, citing Arkham data).

Now let us do the arithmetic that the headlines skip.

1. **Net, not gross.** The gross deposit was 1,518.6 BTC. The number that matters is 1,518.6 − 305.5 = **1,213 BTC**. Reporting the gross figure inflates the story by 25%.
2. **Size it against the firm's own scale.** CryptoSlate put the 1,213 BTC at about \$107 million, which implies a bitcoin price around \$88,000 that day. Against a firm reporting **over \$15 billion of average daily trading volume**, \$107 million is about **0.7% of one day's volume**. It is a large number to you and a rounding error to the desk.
3. **Ask what it could mean.** At least four readings fit the same data equally well: (a) the desk is selling BTC; (b) the desk is positioning inventory to *make markets* over a low-liquidity holiday period, when its quotes are the only ones left; (c) it is settling an OTC block for a client who sold, and the deposit is the hedge leg; (d) it is rebalancing between venues.
4. **Note the conditioning variable.** 31 December is one of the thinnest liquidity days of the year. Moving inventory *onto* a venue before a thin session is exactly what you would do if you intended to *provide* liquidity, not consume it.

The intuition: **the on-chain record told you 1,213 BTC changed custody. It did not tell you a single trade happened.** As the CryptoSlate piece itself put it, on-chain transfers timestamp custody changes, not trades — a deposit can sit untraded for days or execute instantly, and nothing on the blockchain distinguishes the two.

### Step 4 — Bound the inference explicitly

Write down, in words, the strongest claim your data actually supports. For the example above it is something like:

> Between 31 December 2025 and 2 January 2026, wallet clusters labelled as Wintermute by Arkham moved a net ~2,654 BTC into Binance deposit addresses, concentrated in a low-liquidity holiday window. This is consistent with inventory positioning, with hedging an OTC flow, and with distribution; the on-chain data alone does not distinguish between them.

Now compare that to the shape of headline this genre of data reliably produces — some variant of *"market maker dumps \$X hundred million: what do they know?"*, typically quoting the gross inflow rather than the net, and treating a custody transfer as a completed sale. Both readings are built on the same transfers. Only one of them is a claim you could defend if someone pushed back.

If you want the sentence to be stronger than the one above, you have to go and *get* more data — the trailing baseline for that cluster, the same cluster's behaviour before previous holiday sessions, the exchange volume that followed. Strengthening the claim by adjective rather than by evidence is the failure mode.

The specific errors to avoid, in order of how often they appear:

- **Custody-equals-trade.** The single most common. A deposit is an option to trade, not a trade.
- **Gross-for-net.** Quoting inflows while ignoring the withdrawals going the other way in the same window.
- **Agency confusion.** A market maker moving a client's hedge is not expressing its own view. Much of what a desk does on-chain is somebody else's position wearing the desk's wallet.
- **Selection bias.** You notice the deposits before a fall and forget the identical deposits before a rally. Build the full time series before you build the narrative.
- **Label creep.** One unlabelled address adjacent to a labelled cluster becomes "Wintermute" in a thread, and by the third retweet it is a fact.

### Step 5 — Cross-check against the tape

The final discipline: if your on-chain story is true, it should leave fingerprints elsewhere. Ask whether centralized-exchange volume, spread and perpetual funding rates moved the way your story predicts.

If you believe a desk distributed \$100 million of BTC into a thin market, you should be able to find the volume spike, the spread widening, and probably a shift in perp funding. If none of that happened, your story is wrong — the coins moved, but they did not do what you said they did.

And if you want to build this into a repeatable workflow rather than a one-off, the mechanics get a full treatment later in this series: [Tracing a Market Maker's On-Chain Footprint](/blog/trading/crypto-players/tracing-a-market-makers-onchain-footprint) for the wallet-clustering and cadence methodology, [Following Token Flows from Insiders to Exit Liquidity](/blog/trading/crypto-players/following-token-flows-from-insiders-to-exit-liquidity) for the supply-side version of the same technique, and [Detecting Manipulation: On-Chain Red Flags](/blog/trading/crypto-players/detecting-manipulation-onchain-red-flags) for the patterns that genuinely do distinguish market making from manipulation. The complementary CeFi-side skill — reading spoofing and manufactured volume off the tape — is in [Wash Trading, Spoofing, and Manufactured Volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume).

## 6. Case study: the September 2022 vault drain

On **20 September 2022**, Wintermute's DeFi operations were drained of about **\$160 million**. Gaevoy put the figure at roughly \$162.5 million in his public statements the same day; press coverage settled on "about \$160 million." It remains one of the largest single losses suffered by a crypto trading firm, and — unusually for this genre — the technical cause is well documented and genuinely instructive.

The account below is what was reported at the time by security firms and press. It is the version Wintermute's own statements are consistent with, but it is a reconstruction from on-chain evidence, not an audited finding.

![A gas optimisation plus a missed permission revocation combined into a single point of failure: the vanity hot wallet's key was reconstructed and it was still an admin on the DeFi vault.](/imgs/blogs/wintermute-the-algorithmic-powerhouse-7.webp)

### What a vanity address is, and why a trading desk wanted one

Ethereum addresses are 40 hexadecimal characters of essentially random-looking data. A **vanity address** is one that has been searched for until it starts with a chosen pattern — for example, seven leading zeros: `0x0000000...`.

Why would a trading firm want that? Because of a quirk of Ethereum's fee mechanics. Transaction cost — **gas** — depends partly on the number of zero bytes in the transaction data, with zero bytes charged at a lower rate than non-zero ones. An address with many leading zeros therefore costs marginally less gas every single time it appears in a transaction. For a firm doing enormous on-chain volume, "marginally less, millions of times" is a real optimisation. It was a sensible engineering decision made for a sound reason.

### The flaw in the generator

The tool commonly used to search for these addresses was an open-source utility called **Profanity**. To find an address matching a pattern, it generates enormous numbers of candidate private keys and checks the resulting addresses.

The problem was in how it generated those candidates. As reported by multiple security firms, Profanity seeded its random number generator with a **32-bit** value. Thirty-two bits is about 4.3 billion possibilities. A properly generated Ethereum private key has 256 bits of entropy — a number so large it is not searchable by anything. Four billion is searchable by a laptop.

The most vivid demonstration came from **Amber Group**, which published a reproduction of the attack. By their account, a precomputation phase took **under 10 hours**, after which cracking a specific seven-leading-zero address took about **40 minutes** — the entire exercise completed in **under 48 hours** on a MacBook M1 with 16GB of RAM. The precomputed dataset only had to be built once and could then be reused against other Profanity addresses.

So: an address that looked random, and that was generated by a widely used community tool, was in fact drawn from a pool small enough to enumerate on a consumer laptop.

### The second failure: the permission that was not revoked

Here is the part that turns a bad tool into a \$160 million loss, and the part every engineering team should sit with.

The Profanity vulnerability became **public on 15 September 2022**, when the 1inch Network disclosed it — five days before the drain. Wintermute responded, and moved ether out of the affected hot wallet.

But according to the post-mortems, the compromised address was not only a wallet holding funds. It was also registered as an **admin** on Wintermute's DeFi vault smart contract — the account authorised to move assets on the vault's behalf. Emptying the wallet removed the funds *in* it. It did not remove its *authority over* the vault. That permission was left in place.

On 20 September the attacker reconstructed the private key, took control of the address, and called the vault as its administrator.

### What was taken

Reported compositions differ slightly between post-mortems, which is normal for an event reconstructed from on-chain flows across many tokens. The broad picture, as reported at the time:

- Roughly **\$118 million** in stablecoins (predominantly USDC and USDT) per Halborn's itemisation;
- **671 WBTC** (wrapped bitcoin), valued at roughly \$13 million at the time;
- **6,928 ETH**, valued at roughly \$9.4 million;
- the balance in a long tail of smaller tokens.

The Block and Forbes reported a similar shape at coarser resolution: around \$120 million of stablecoins, about \$20 million of BTC and ETH, and about \$20 million of lesser-known tokens.

### The response

Gaevoy went public within hours. His statements that day, as widely reported: the hack was confined to the DeFi operations; **CeFi and OTC services were unaffected**; and the firm was "**solvent with twice over that amount in equity left**" — which, against a \$160 million loss, implies roughly \$320 million of remaining equity. He also said the firm was "open to treat this as a white hat" and subsequently offered the attacker a **10% bounty** for returning the remainder. CoinDesk reported separately that Wintermute had around **\$200 million in outstanding DeFi debt** at the time, which is why the solvency question was not academic.

The firm continued operating. Three and a bit years later it was reporting record volumes and being named in a US brokerage's SEC filings. Whatever else the episode demonstrates, it demonstrates that a well-capitalised trading firm can absorb a nine-figure operational loss without failing — which is itself worth knowing when you are assessing counterparty risk.

Some observers speculated at the time about an inside job. No such claim was ever substantiated, and the technical reconstruction that the independent security firms converged on — Profanity key recovery plus an unrevoked vault permission — accounts for the observed on-chain behaviour without needing one. It should be treated as speculation, not as a finding.

### The other 2022 incident

Three months earlier, in **June 2022**, a separate and much smaller failure: the Optimism Foundation sent **20 million OP tokens** to an address Wintermute had supplied for its market-making engagement. The address was a multisig wallet that existed on Ethereum mainnet but had not been deployed on the Optimism chain, so the tokens landed at an address nobody controlled. An attacker was able to take control of the address on Optimism via transaction replay and moved the tokens. **17 million OP** were subsequently returned; the attacker also sent 1 million OP to Vitalik Buterin's address, recovered separately.

Two incidents, three months apart, both fundamentally about address handling rather than about trading. That is the honest lesson: for a firm like this, the trading models are not usually the fragile part. The plumbing is.

## Common misconceptions

**"Wintermute is betting against retail."** A well-run market maker is deliberately delta-neutral: it hedges the position you hand it within seconds and earns from the volume of your trading, not the direction. That said, the caveat is real and specific — a desk holding *free call options* from a token deal, or a firm whose venture arm holds a position in a project it also quotes, is not delta-neutral on that name. The correct mental model is not "they're on your side" or "they're against you," it is: *find out what position the structure gives them, then assume they will behave consistently with it.*

**"A big wallet deposit means they're dumping."** This is the single most common on-chain error and section 5 exists to correct it. A deposit into an exchange moves custody. It creates the *ability* to sell; it is not a sale. It is equally consistent with positioning inventory to provide liquidity, hedging a client's OTC block, or rebalancing across venues. If someone shows you a deposit and tells you what it means, ask them how they eliminated the other explanations.

**"They got hacked, so they must be badly run."** The 2022 drain was caused by a widely used community tool with a subtle entropy bug, compounded by a missed permission revocation during an incident response. Both failures are ordinary engineering failures of a kind that has happened at very well-run organisations. The relevant question for a counterparty is not "did something go wrong" but "were they capitalised enough to absorb it, and did they disclose it fast?" On both counts the record is reasonably good: public disclosure within hours, and a solvency claim that subsequent years of continued operation have not contradicted.

**"A wide spread on a small token means the maker is gouging."** The spread is a break-even price, not a margin. It has to cover expected adverse-selection losses, the cost of carrying inventory that cannot be cleared quickly, and operating cost, before a cent of profit is left. A thin token has all three costs high. If the spread were pure profit, a competitor would undercut it; that it stays wide means the costs are real.

**"Market makers create the volume, so the volume is fake."** Legitimate two-sided quoting is the opposite of manufactured volume: it dampens volatility by absorbing imbalances and narrows spreads by competing. Wash trading — trading with yourself to inflate the tape — is a distinct and in many jurisdictions illegal activity, and conflating the two makes you worse at spotting the real thing. The distinguishing patterns are covered in [Wash Trading, Spoofing, and Manufactured Volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume).

**"If I can see the wallets, I can front-run them."** Even setting aside that most of the firm's activity never touches a chain, the on-chain record is a *lagging* signal — you learn a transfer happened after it confirmed, by which time any associated trading is done. The information asymmetry runs the other way: the desk sees your order before you see its wallet.

## How it shows up in real markets

### 1. The Robinhood filing, May 2025

The cleanest illustration of scale is the one Wintermute did not author. In its Form 10-Q covering the first quarter of 2025, Robinhood disclosed the market makers responsible for 10% or more of its transaction-based revenues, and named Wintermute at **11%**, alongside B2C2 at 12% and Citadel Securities at 12%. Robinhood is one of the largest retail brokerages in the United States. The mechanism from section 1 — `N × s`, a fraction of a basis point applied to an enormous count — is what puts a private London firm into a US-listed company's revenue-concentration disclosure. It is also a reminder that "crypto liquidity" and "the retail app you use" are not separate worlds; they are the same pipe with a UI on one end.

### 2. New Year's Eve 2025, and how a story gets built

The Dec 31 2025 – Jan 2 2026 BTC deposits are a small masterclass in how on-chain narratives form. The observable facts were modest: net ~1,213 BTC in on 31 December, ~1,441 BTC net over the following two days, all into Binance, all from Arkham-labelled Wintermute clusters. Note the verb in the headline the reporting itself used — the coins were "secretly offloaded," which asserts both concealment and a sale, neither of which is in the data. Public wallets are not secret, and a deposit is not an offload.

Nothing in the on-chain record supports the second framing over the first. The deposits landed into one of the thinnest liquidity windows of the calendar year — which is precisely when a market maker most needs inventory *on venue* in order to keep quoting. The custody moved; the trades, if any, are invisible. Both the sober reading and the dramatic reading are consistent with the data, and that is exactly the point: when two incompatible stories fit equally well, you do not have evidence, you have a Rorschach test.

### 3. The 2025 liquidity concentration, read as a coverage decision

Wintermute's *OTC Markets 2025* report contains a finding that is easy to skim past: BTC and ETH fell from 54% of notional in 2023 to 49% in 2025, blue chips outside those two gained about 8 points, and the average altcoin rally collapsed from 61 days in 2024 to 19 days in 2025. The firm's own explanation was that there was insufficient liquidity to carry narratives further down the curve.

Turn that around and it is a statement about where professional quoting capital went. Rallies get shorter when nobody is willing to warehouse the inventory required to sustain them. If you have ever wondered why a token can rip 200% in a week and then bleed out over the next month with no news, this is a large part of the answer, and it is a liquidity-supply story rather than a sentiment story.

### 4. Prediction markets, May 2026

In May 2026 Wintermute announced it had begun providing two-sided liquidity on prediction-market event contracts, citing more than \$60 billion of event-contract trading volume in 2026 and describing the segment's order books as shallower and spreads wider than in comparable futures or options markets. Jake Ostrovskis, the firm's head of OTC trading, framed the opportunity in exactly the terms of section 4 of this post: sustained two-sided liquidity narrows spreads, supports larger trades, and improves how much information the market price actually carries.

It is a clean illustration of the business model's portability. Nothing about the machinery is crypto-specific. Point it at any market with wide spreads, adequate volume and a hedging instrument, and the same `N × s` engine applies.

### 5. The Yearn vote, August 2023

Already described above, but worth reading as a market event rather than a governance one. A maker proposed a standard loan-plus-terms structure; a token community priced the structure, concluded that the dilution risk was not worth the liquidity, and rejected it. The token did not need Wintermute; Wintermute did not need the token. Both walked away. That is what a functioning negotiation between a project and a liquidity provider looks like, and it happens far more often than the pump-and-dump narrative allows.

### 6. The vanity-address bug as an industry-wide event

The Profanity flaw did not only affect Wintermute. Once the vulnerability was public, addresses generated with the tool were, collectively, a pot of money sitting behind a lock anyone could pick, and other users were drained too. The industry response — audit every address for provenance, rotate anything generated by a community vanity tool, and, above all, *revoke permissions rather than merely emptying wallets* — is now standard practice.

The generalisable lesson has nothing to do with crypto: an optimisation that touches your key material is a security decision wearing a performance costume. Wintermute chose leading zeros to save gas. The saving was real. The cost, in the specific case where the generator was flawed, was \$160 million.

## What it means if you're trading against them

You will not out-trade a firm like this on speed, infrastructure or information, and nothing in this post is a strategy. What understanding it buys you is a set of better default assumptions.

- **Read the spread and the refresh rate before you size a trade.** A tight spread with depth that replenishes instantly means a professional is quoting and confident. A wide spread that stays wide means nobody wants to stand behind this token at a tight price, and your exit will cost you what your entry did.
- **For real size, ask for a price instead of taking one.** The \$150,000 difference in the worked example above is not exotic; it is the routine gap between sweeping a book and requesting a quote. Any desk will quote a size you can actually move.
- **Treat a sudden spread gap as information, not opportunity.** When quotes thin out on a token you hold, the makers are pricing danger they can see and you cannot. That is a moment to be careful, not to buy the dip in liquidity.
- **Find out how the maker is paid before you judge its behaviour.** A retainer-paid maker and a loan-plus-call-option maker have different exposures to the token price, and the second one's incentives are not neutral. Projects increasingly disclose this; ask if they don't.
- **When you read an on-chain thread, run the five checks from section 5.** Net or gross? Sized against what baseline? What are the alternative explanations? Whose position is it? Does the tape corroborate? Most viral on-chain claims fail at least two of those, and knowing which two is most of the skill.
- **Hold both facts about the firm at once.** It provides a genuine service — the tight spreads and smooth fills you take for granted on liquid pairs exist because someone is standing there — and its liquidity is conditional, its interests are its own, and it can step aside in milliseconds. Neither romanticise it nor demonise it; model it.

Wintermute is the clearest available specimen of a species that shapes almost every price you see in crypto and is almost never named in the reporting about those prices. Learning to read one firm properly — its surfaces, its incentives, its footprint, and the hard limits of what its footprint can tell you — is the transferable skill. There are perhaps a dozen firms that matter this much, and they all work roughly the same way.

For where market makers sit in the wider hierarchy of who moves crypto, see [The Hidden Power Structure of Crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto); for the order-book layer underneath everything here, [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move); and for the venue-to-venue arbitrage that keeps all sixty of those order books roughly agreeing with each other, [Cross-Exchange Arbitrage and the Latency Game](/blog/trading/crypto-players/cross-exchange-arbitrage-and-the-latency-game).

## Sources & further reading

**Company and scale**

- Wintermute, [*OTC 2024 in review & 2025 outlook*](https://www.wintermute.com/insights/market-color/reports/wintermute-otc-2024-in-review-2025-outlook) (Jan 2025) — 50+ venues, 1,000+ assets, daily volumes frequently exceeding \$5bn, and the ~\$2.24bn single-day OTC spot record of Nov 2024.
- Wintermute / PR Newswire, [*OTC Markets 2025 report*](https://www.prnewswire.com/news-releases/wintermutes-otc-markets-2025-report-shows-cryptos-upper-tier-becoming-an-established-asset-class-as-liquidity-concentrates-302659976.html) (13 Jan 2026) — "over \$15 billion in average daily trading volume," 60+ venues, BTC/ETH share 54% (2023) → 49% (2025), altcoin rally duration 61 days (2024) → 19 days (2025), options notional ~4x by year-end.
- Robinhood Markets, [Form 10-Q, Q1 2025](https://investors.robinhood.com/static-files/5b0eff4a-59b7-4f67-be5e-917fc00f4bd2) — market-maker revenue concentration; reported by The Block, [*Robinhood lists B2C2 and Wintermute as largest crypto market makers*](https://www.theblock.co/post/352819/robinhood-lists-b2c2-and-wintermute-as-market-makers-for-the-first-time-in-latest-sec-filing) (May 2025).
- Wintermute, [*Enters prediction markets as a liquidity provider*](https://www.wintermute.com/insights/news/announcements/wintermute-enters-prediction-markets-as-a-liquidity-provider-as-event-contract-trading-surpasses-60-billion-in-2026) (29 May 2026) — the \$3.5tn annual volume figure and the \$60bn event-contract market size.
- Wintermute, [Company](https://www.wintermute.com/company) and [OTC](https://www.wintermute.com/otc) pages — business lines, 250+ assets on the OTC desk, offices.

**The September 2022 hack**

- Halborn, [*Explained: The Wintermute Hack (September 2022)*](https://www.halborn.com/blog/post/explained-the-wintermute-hack-september-2022) — the Profanity 32-bit seed, the unrevoked vault admin, and the itemised losses (\$118.4m stablecoins, 671 WBTC, 6,928 ETH).
- 1inch Network, [*A vulnerability disclosed in Profanity, an Ethereum vanity address tool*](https://blog.1inch.com/a-vulnerability-disclosed-in-profanity-an-ethereum-vanity-address-tool/) (15 Sep 2022) — the original disclosure, five days before the Wintermute drain.
- Amber Group, [*Exploiting the Profanity Flaw*](https://medium.com/amber-group/exploiting-the-profanity-flaw-e986576de7ab) (Sep 2022) — the reproduction: under 10 hours of precomputation, ~40 minutes per seven-leading-zero address, under 48 hours total on a MacBook M1 with 16GB RAM.
- The Block, [*Experts blame a 'vanity address' bug for Wintermute's \$160 million hack*](https://www.theblock.co/post/171192/experts-blame-a-vanity-address-bug-for-wintermutes-160-million-hack) (Sep 2022) — the asset breakdown as reported at the time.
- Forbes, [*How Crypto Trading Firm Wintermute Was Hacked For \$160 Million*](https://www.forbes.com/sites/jeffkauflin/2022/09/20/profanity-may-be-the-cause-of-crypto-trading-firm-wintermutes-160-million-hack/) (20 Sep 2022).
- CoinDesk, [*Crypto Market Maker Wintermute Hacked for \$160M, OTC Services Unaffected*](https://www.coindesk.com/business/2022/09/20/crypto-market-maker-wintermute-hacked-for-160m-says-ceo) and [*Hacked Crypto Market Maker Wintermute Has \$200M in Outstanding DeFi Debt*](https://www.coindesk.com/business/2022/09/20/hacked-crypto-market-maker-wintermute-has-200m-in-outstanding-defi-debt) (Sep 2022) — Gaevoy's solvency statement and the outstanding DeFi debt figure.

**The June 2022 Optimism incident**

- CoinDesk, [*\$15M of Optimism Tokens Stolen After Wintermute Sent Wrong Wallet Address*](https://www.coindesk.com/tech/2022/06/09/15m-of-optimism-tokens-stolen-by-an-attacker-after-wintermute-sent-wrong-wallet-address) (Jun 2022); Decrypt, [*Optimism Hacker Returns 17 Million Tokens After Airdrop Blunder*](https://decrypt.co/102541/optimism-hacker-returns-17-million-tokens-airdrop).
- SlowMist, [*Key to the Theft of 20 Million OP Tokens — Transaction Replay*](https://slowmist.medium.com/slowmist-key-to-the-theft-of-20-million-op-tokens-transaction-replay-490baaf45f26) — the technical mechanism.

**On-chain tracing**

- Arkham Intelligence, [Wintermute entity page](https://intel.arkm.com/explorer/entity/wintermute) and the separate [Wintermute Hacker](https://intel.arkm.com/explorer/entity/wintermute-hacker) entity; Nansen, [Wintermute Trading profiler](https://app.nansen.ai/profiler?entity=Wintermute+Trading&chain=ethereum&tab=overview).
- CoinDesk, [*Wintermute, a Major Trader, Is a Key Player in Arkham's Controversial Dox-to-Earn Platform*](https://www.coindesk.com/business/2023/08/02/wintermute-a-major-trader-is-a-key-player-in-arkhams-controversial-dox-to-earn-platform) (Aug 2023) — the conflict disclosure that matters when you use Arkham labels on Wintermute.
- CryptoSlate, [*Major market maker secretly offloaded 1,213 BTC onto Binance during New Year's Eve thin liquidity*](https://cryptoslate.com/major-market-maker-secretly-offloaded-1213-btc-onto-binance-during-new-years-eve-thin-liquidity/) (Jan 2026) — the 31 Dec 2025 flows, and the article's own caveat that on-chain transfers timestamp custody changes, not trades.

**Market-maker deal structures**

- CoinDesk, [*Yearn Finance Voters to Wintermute: Drop Dead*](https://www.coindesk.com/business/2023/08/29/yearn-finance-voters-to-wintermute-drop-dead) (29 Aug 2023) — the 350 YFI / 0.1% / 12-month proposal and its rejection.
- DL News, [*Wintermute CEO says pump-and-dump accusations are 'flattering'*](https://www.dlnews.com/articles/people-culture/wintermute-ceo-says-pump-and-dump-accusations-are-flattering/) — Gaevoy on the loan and call-option structures.
- DL News, [*Crypto market makers rake in cash shorting their customers' tokens. One firm is calling for more transparency*](https://www.dlnews.com/articles/markets/market-makers-short-tokens-but-one-firm-wants-transparency/) — the loan-plus-option versus retainer debate.

**This blog**

[Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers) (series hub) · [What a Crypto Market Maker Actually Does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) (the mechanics, from zero) · [The Loan-Plus-Options Deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid) · [Inventory Risk, Hedging, and Delta Neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality) · [Designated versus Principal Market Making](/blog/trading/crypto-players/designated-versus-principal-market-making) · [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move) · [Wash Trading, Spoofing, and Manufactured Volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) · [Cross-Exchange Arbitrage and the Latency Game](/blog/trading/crypto-players/cross-exchange-arbitrage-and-the-latency-game) · [The Hidden Power Structure of Crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto)
