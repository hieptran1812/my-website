---
title: "Exchanges Are Players, Not Just Venues: The Conflict Map Behind Every Listing"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A build-from-zero guide to what a crypto exchange actually is — a venue, a venture investor, a listing gatekeeper, a possible counterparty, a token issuer, and your custodian, all inside one company — and what that combination does to the price of the thing you are buying."
tags: ["crypto", "exchanges", "listings", "conflicts-of-interest", "market-structure", "binance", "coinbase", "exchange-tokens", "crypto-players", "custody", "regulation"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 47
---

> [!important]
> **TL;DR** — A crypto exchange is not a neutral pipe that trades pass through. It is a commercial firm that simultaneously runs the venue, invests in tokens, decides which tokens list, may be a counterparty to your trade, issues its own token, and holds your coins. Those six roles are separately licensed businesses in traditional finance.
>
> - **A listing is a price event, not an administrative one.** Nothing about the token changes; access changes. A fixed float meets one-sided demand, and the arithmetic does the rest.
> - **The exchange usually knows first.** In a typical listing workflow the decision is inside five organisations for weeks before the announcement. You are the last node on that chain, by construction.
> - **The pop is a queue, not a return.** In this post's illustrative worked example, a \$2.0M market buy pays a volume-weighted average of **\$0.72** while printing a last price of **\$1.10** — a +175% chart on a +79.6% average fill.
> - **The venture arm's return is the vesting integral, not the listing print.** In the same illustrative arithmetic, a \$2.0M seed cheque worth **\$32.0M** on listing day realises about **\$9.7M** through a 12-month cliff and 24-month vest — 4.8×, not 16×.
> - **The exchange's own token is a claim you can divide.** Annual burn ÷ market cap is the whole thesis, and it is usually a fraction of a percent.
> - **The one habit to keep:** size to the venue's daily volume, not to your conviction — and assume every listing headline reached someone else first.

Nothing about a token changes on the day it lists.

The code is the same code. The team is the same team. The supply schedule was fixed months earlier. Not one line of the whitepaper is different at 14:00 than it was at 13:59. And yet at 14:00, when a large exchange opens the pair, the price of that unchanged thing routinely doubles — and then, over the following weeks, routinely gives most of it back.

That gap between "nothing changed" and "the price doubled" is the subject of this post. It has a mundane explanation, and the mundane explanation is more useful than the conspiratorial one: **an exchange listing is not information about the token. It is a change in who is allowed to buy it, and with how much friction.** The exchange controls that gate. It also, in most cases, owns some of the token, employs the people who decided to open the gate, runs the venue where the resulting trades happen, may be quoting some of those trades, issues a token whose value rises with the fees those trades generate, and is holding the coins you use to pay for them.

That is six businesses. In equity markets those six businesses require six different licences from at least three different regulators, and several of the combinations are simply illegal. In crypto they are one login.

![The six commercial roles a single crypto exchange entity holds at the same time, and the conflict each one creates.](/imgs/blogs/exchanges-are-players-not-just-venues-1.webp)

The diagram above is the mental model for everything that follows: one entity, six hats, and a set of incentives that do not point the same way. This is not an accusation. Most of what follows describes structure, not misconduct — the point of a conflict map is that it tells you what *could* happen and what you should therefore check, entirely independently of whether any particular firm has done anything wrong. Where real allegations exist, this post reports them as allegations, names the source and the date, and includes the denial. Nothing here asserts wrongdoing beyond what is on the public record.

> [!note]
> **How to read the numbers in this post.** Two kinds appear, and they are kept strictly apart. **Dated, attributed facts** — court filings, settlements, sentences, regulations — carry the date and the source inline, and are listed again under *Sources & further reading*. **Illustrative arithmetic** — every worked example, every figure built on one — uses round invented numbers for an imaginary token, is labelled *illustrative* where it appears, and describes no real listing, deal, or firm. Nothing in this post reports a listing-pump percentage, a fund size, or a fee as measured fact unless it is dated and attributed.

If you have read [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto), you already know the cast: funds, market makers, foundations, exchanges. This post takes the last of those and pulls it apart.

## Foundations: what an exchange actually is

Before any of the conflict argument works, you need the machinery. If you already know what a taker fee and a maker rebate are, skim to the next section. If you don't, none of this is obvious and you should not pretend otherwise — start here.

### The matching engine and the order book

An exchange is, at its core, one piece of software: a **matching engine**. It keeps a list of everyone who wants to buy and everyone who wants to sell, sorted by price, and it pairs them off.

That list is the **order book**. It has two sides:

- **Bids** — resting buy orders, the highest-priced ones first. The best bid is the most you could sell into right now.
- **Asks** (or offers) — resting sell orders, the lowest-priced ones first. The best ask is the least you could buy at right now.

The gap between the best bid and the best ask is the **bid-ask spread** — a real cost. If the best bid is \$0.99 and the best ask is \$1.01, buying and immediately selling loses you \$0.02 per token, or about 2% of the \$1.00 mid price. That 2% goes to whoever was quoting both sides.

Two order types matter:

- A **limit order** names a price and waits. It joins the book and *provides* liquidity. You control your price but not whether you trade at all.
- A **market order** names a size and takes whatever prices the book currently offers. It *consumes* liquidity. You are guaranteed to trade but not at what price — the engine fills you against the best resting order, then the next, then the next, walking the price against you until your size is done.

That walking-up behaviour is the single most important mechanical fact in this post. We will come back to it with numbers.

### How the venue gets paid

Exchanges charge a **trading fee**, quoted in basis points. A *basis point* (bps) is one hundredth of one percent — 0.01%. Ten basis points is 0.10%.

Most venues use a **maker-taker** schedule:

- The **taker** — whoever sends the market order that consumes resting liquidity — pays the higher fee, typically a single-digit-to-low-double-digit number of basis points at retail tiers, falling sharply at high-volume tiers. Every venue publishes its own schedule; look yours up rather than trusting a number in an article.
- The **maker** — whoever posted the resting order that got consumed — pays less, sometimes zero, sometimes a negative fee (a *rebate*, i.e. the venue pays them).

Why pay the maker? Because resting orders are the product. An exchange with no resting orders is a website. Paying market makers to sit in the book is how a venue manufactures the thing it sells.

This immediately tells you what an exchange's business model is: **it is paid on volume, not on outcomes.** It does not care whether you make money. It cares that you trade, that you trade often, and that you trade in size. Hold that thought — it explains an enormous amount of behaviour that otherwise looks strange.

### Float, FDV, and why "market cap" lies

Three definitions you must have:

- **Circulating supply** (or **float**) — the tokens that actually exist in tradeable hands right now. Not locked, not vesting, not in a treasury multisig.
- **Total supply** — every token that will ever exist.
- **Fully diluted valuation (FDV)** — price × total supply. What the project would be "worth" if every future token existed today.

A token can have a \$20M float and a \$250M FDV. Those are wildly different claims about the same asset, and both get printed on the same page. The float is what sets the price; the FDV is what sets the future selling pressure. [Why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) goes deeper here, and [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) walks the whole schedule.

For our purposes: **a listing pours demand onto a float, and the float is usually tiny.** Small denominator, large numerator, big number.

### Centralised versus decentralised, in one paragraph

A **centralised exchange** (CEX) — Binance, Coinbase, OKX, Bybit, Upbit — takes custody of your assets, runs its matching engine on private servers, and settles internally. You have an IOU from the company. A **decentralised exchange** (DEX) — Uniswap, Curve, Hyperliquid's on-chain book — settles on a blockchain, and you keep your own keys. Almost every conflict in this post is a CEX conflict, because almost every one of them requires a company with a balance sheet, a listing committee, and a venture arm. A DEX has none of those things; it has different problems, which is a different post.

### What "listing" means here, and why it is not an IPO

In equities, a company that wants to be publicly traded goes through an underwriting process: an investment bank prices the offering, a regulator reviews a registration statement, and a set of rules governs who can sell what, when. The exchange that admits the shares is not the bank that priced them, is not an investor in the company, and does not trade the shares for its own account.

In crypto, "listing" means one thing: **an exchange decides to open a trading pair.** There is no prospectus requirement, no statutory quiet period, no underwriter, no mandated lockup, and — critically — no rule preventing the venue from also being an investor in the thing it is listing.

That is the whole story compressed into one sentence. The rest of this post is that sentence in detail.

## The six hats

Here is the map, hat by hat, before we take each one apart.

| Hat | What the exchange does | Whose interest it can cut against |
|---|---|---|
| 1. The venue | Runs the matching engine, sees every order, every cancel, every wallet | Traders whose intentions are visible to the operator |
| 2. The ventures arm | Buys tokens in private rounds at private-round prices | Anyone buying the same token at public prices later |
| 3. The listing desk | Decides which tokens trade, when, and on which pairs | Projects that are not selected; buyers who arrive after those who knew |
| 4. Prop / market-making ties | May trade on its own platform, or have close relationships with firms that do | Anyone on the other side of those trades |
| 5. Its own token | Issues a token whose value is a claim on venue fees | Holders, who own an unsecured claim with no legal enforcement |
| 6. The custodian | Holds customer coins on its own balance sheet | Customers, who are unsecured creditors in a bankruptcy |

No single hat is scandalous. A venue running a matching engine is a venue. A fund buying seed rounds is a fund. The problem is arithmetic: the conflicts **compose**. Hat 2 gives the firm a position. Hat 3 gives it the ability to create demand for that position. Hat 1 tells it how much demand already exists. Hat 5 pays it a second time on the fees generated by the resulting volume. Hat 6 means you cannot leave while it happens.

Now, one at a time.

## Hat 1 — the venue that sees everything

An exchange operator sees the complete order book, including the parts you cannot: every resting order, every cancellation, the identity behind each account, the deposit and withdrawal flows on both sides, and the historical behaviour of every trader on the platform. You see the aggregated book. The operator sees the ledger.

This is not sinister by itself — you cannot run a matching engine without seeing the orders in it. In traditional markets the equivalent information exists too. What differs is the *rulebook around it*. A registered US national securities exchange operates as a **self-regulatory organisation** (SRO) under Section 6 of the Securities Exchange Act of 1934, must file its rules — including its listing standards — with the SEC under Section 19(b), and has statutory obligations to police its own members. When an alternative trading system operates in US equities, Form ATS-N requires it to disclose, publicly, what the operator's own affiliates do on the venue.

The crypto equivalent of Form ATS-N is: a blog post, if the exchange chooses to write one.

The practical consequence for you is narrower than "they are front-running you," which is a serious allegation requiring serious evidence. The practical consequence is that **the operator's information advantage is structural and permanent**, and the only mitigations are the ones the operator voluntarily adopts and voluntarily audits. When you are told "we have internal information barriers," you have been told about a policy, not a rule. Policies are not filed with anyone. They can be changed on a Tuesday.

Two habits follow. First, assume any large resting order you leave on a venue is visible information, not a secret. Second, when you see a size that would matter, ask why it is sitting on a public book at all rather than being worked through an OTC desk — a subject we take up in [OTC desks and moving size without moving price](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price).

## Hat 2 — the ventures arm

Every large exchange has, or has had, an investment arm: Coinbase Ventures, Binance Labs (later rebranded YZi Labs), OKX Ventures, Bybit's affiliated Mirana Ventures, KuCoin Labs, Crypto.com Capital, Gate Ventures. The pattern is universal, because it is a good business.

A caveat that is itself part of the story: **these vehicles do not report like funds.** They are not registered investment advisers filing public disclosures, they do not publish audited assets under management, and the relationship between the venture arm and the exchange — same balance sheet, affiliated entity, or founder's personal vehicle — varies by firm and by year, and has been restructured more than once. Any specific figure you see quoted for one of their fund sizes is a press estimate or a self-reported number, not a filing. That opacity is the first thing to notice about the hat.

Here is why it is a good business, and why it is a conflict at the same time.

A venture arm buys tokens or equity in a private round, at a private-round price, before the token trades anywhere. It is the same trade a crypto fund makes — [the crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model) describes it in full. The difference is that this particular investor also happens to own the largest liquidity venue in the asset class, and the largest liquidity venue in the asset class is the single most valuable thing a pre-launch token can be given.

Nobody has to do anything improper for this to matter. A venture arm that invests in projects it believes in, and a listing desk that lists projects that meet its standards, will naturally overlap, because both are applying a quality filter to the same universe. The overlap is not evidence. But the overlap is also not *nothing* — it means the firm holds a position whose value moves with a decision the same firm makes.

The honest framing is this: **the exchange's venture arm is long the exact thing the exchange's listing desk can create demand for.** Whether any given firm manages that well is a question about that firm's controls. Whether the structure creates the incentive is not a question at all.

### What the venture arm actually earns

The naive version of this trade — "buy at seed, sell into the listing pop" — is not what happens, and understanding why makes the incentive clearer, not weaker.

#### Worked example 1: the ventures-arm return

Suppose an exchange's venture arm writes a \$2.0M cheque into a seed round. (Illustrative numbers throughout this example — not a real deal.)

**Step 1 — what \$2.0M buys.** The round is priced at a \$25M fully diluted valuation on a 1,000,000,000-token supply. Price per token:

- \$25,000,000 ÷ 1,000,000,000 = **\$0.025 per token**

So \$2,000,000 ÷ \$0.025 = **80,000,000 tokens**, which is 8.0% of total supply.

**Step 2 — the listing print.** Twelve months later the token lists and opens at **\$0.40**. On paper:

- 80,000,000 × \$0.40 = **\$32,000,000**

That is a 16× on a screenshot. It is also entirely fictional, because none of those tokens can be sold.

**Step 3 — the lockup.** The round carried a standard schedule: a **12-month cliff** (nothing releases at all for a year) and then **24 months of linear vesting** (equal monthly releases). So from month 13 to month 36:

- 80,000,000 ÷ 24 = **3,333,333 tokens per month**

**Step 4 — the price path.** Assume the token loses 5% of its value each month from the listing print — a mild decline by the standards of a small-float launch, and again purely illustrative. By month 13, when the first tranche unlocks:

- \$0.40 × 0.95¹³ = **\$0.2053**

By month 36, the final tranche:

- \$0.40 × 0.95³⁶ = **\$0.0631**

**Step 5 — the realised total.** Summing 3,333,333 tokens sold at the price of each month from 13 to 36:

- Realised proceeds ≈ **\$9,692,000**

**Step 6 — the two numbers.** The paper peak was \$32.0M. The bank balance is \$9.7M. Against a \$2.0M cost that is **4.8×**, not 16×. The lockup ate **\$22.3M** of paper value.

![A ventures-arm position priced at three moments: the cheque, the paper peak on listing day, and what the vesting schedule actually delivers.](/imgs/blogs/exchanges-are-players-not-just-venues-7.webp)

**The intuition:** a venture arm's return is the *integral of the price over its vesting window*, not the height of the listing candle. Which is exactly why a venue-affiliated investor's interest is not "one big pop." It is **sustained liquidity and sustained attention for three years** — continued listings on more pairs, continued marketing slots, continued inclusion in campaigns. That is a longer, quieter, and more consequential form of influence than a single day's price spike, and it is the one worth watching.

If you want the counterpart from the fund side, [how VCs move price through listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) covers the same mechanism from the investor's chair.

## Hat 3 — the listing decision as a price event

This is the load-bearing hat. Everything else amplifies it.

### The information ladder

A listing is not a decision made and announced in the same minute. It is a workflow, and workflows have participants.

![The typical path from a listing application to the public announcement, and where retail sits on it.](/imgs/blogs/exchanges-are-players-not-just-venues-3.webp)

A composite of how this generally runs — no single exchange's disclosed workflow — looks roughly like:

1. **T−90d** — the token team applies, submits documentation, legal opinions, tokenomics.
2. **T−60d** — a listing committee reviews. Multiple people, multiple functions.
3. **T−30d** — legal and compliance sign off, jurisdiction by jurisdiction.
4. **T−14d** — market makers are onboarded for the pair. This is where an inventory loan is typically struck: the project lends tokens to a market maker so there is something to quote against. [The loan-plus-options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid) explains that contract in detail.
5. **T−7d** — engineering schedules the pair, the pair is configured, deposits are enabled.
6. **T−0** — the public announcement.
7. **T+0** — you can buy.

By the time step 6 happens, the information has lived inside roughly five organisations — the project, the exchange, the exchange's legal counsel, at least one market maker, and usually a communications agency — for about three months. Dozens of people knew. Some of them are paid in the token.

That is not an allegation of leaking. It is a description of the surface area. And the surface area is why the price often starts moving *before* the announcement: the T−3d "rumour" leg on the chart is a recurring feature, not a coincidence.

It is also why the one criminal case in this area is instructive. In July 2022 the US Department of Justice charged Ishan Wahi, a Coinbase product manager, along with his brother and an associate, in what prosecutors described as the first insider-trading case involving cryptocurrency; the SEC filed a parallel civil action. Ishan Wahi pleaded guilty and was sentenced to two years in prison in May 2023; his brother Nikhil was sentenced to ten months in January 2023. Coinbase said it had referred the matter itself. The point is not that exchanges are criminal enterprises — the point is that the information asymmetry is real enough that someone went to prison for monetising it.

### Why the listing moves the price even with zero misconduct

Now the mechanism, with no bad actors at all.

A listing does two things. It removes friction — suddenly millions of accounts that already have funded balances can buy the token in one click, without bridging, without a new wallet, without a hardware key. And it confers a **certification signal** — a large venue's compliance process is a filter, and passing it means something to buyers who cannot evaluate the token themselves.

It does *not* create supply. The float is the float. So the demand curve shifts right against a fixed, and usually very small, quantity available at any given price.

![A \$2.0M market buy walking a thin listing-day ask ladder from \$0.40 to \$1.10.](/imgs/blogs/exchanges-are-players-not-just-venues-4.webp)

#### Worked example 2: listing-day float math

An imaginary token — call it ILLUS — opens at **\$0.40** with a book five levels deep. (Illustrative; not a real listing.) Resting asks:

| Price | Size (tokens) | Value at that level | Cumulative \$ |
|---|---|---|---|
| \$0.40 | 450,000 | \$180,000 | \$180,000 |
| \$0.46 | 350,000 | \$161,000 | \$341,000 |
| \$0.55 | 400,000 | \$220,000 | \$561,000 |
| \$0.68 | 456,000 | \$310,080 | \$871,080 |
| \$0.88 | 511,000 | \$449,680 | \$1,320,760 |

A single **\$2.0M market buy** arrives. Walk it:

**Step 1 — consume the visible book.** All five levels are eaten. That is 2,167,000 tokens for **\$1,320,760**.

**Step 2 — the remainder.** \$2,000,000 − \$1,320,760 = **\$679,240** still needs filling. The next resting size sits at **\$1.10**:

- \$679,240 ÷ \$1.10 = **617,491 tokens**

**Step 3 — the last print.** The final fill happens at **\$1.10**. Against a \$0.40 open, that is:

- (1.10 ÷ 0.40) − 1 = **+175%**

This is the number that appears on every chart, every screenshot, every "up 175% since listing" post.

**Step 4 — what the buyer actually paid.** Total tokens received: 2,167,000 + 617,491 = **2,784,491**. Volume-weighted average price:

- \$2,000,000 ÷ 2,784,491 = **\$0.7183**, call it **\$0.72**

So the buyer's real cost basis is \$0.72 — already **+79.6%** above the open — while the chart says +175%.

**The intuition:** on a thin book, the printed price and the price anyone actually paid are two different numbers, and the gap between them *is* the pump. The chart is a record of the last trade, not of the average trade.

#### Worked example 3: sizing with the square-root law

How much should a \$2.0M order have been expected to move this market? There is a standard model for that, and it is worth knowing because it is the only sizing rule that generalises.

Empirical market-impact research — Almgren and colleagues, and the Bouchaud school of market microstructure — finds that the price impact of an order scales roughly with the **square root** of the order's size relative to daily volume:

$$\text{impact} \approx Y \cdot \sigma \cdot \sqrt{\frac{Q}{V}}$$

where $Q$ is your order size, $V$ is average daily volume in the same units, $\sigma$ is the asset's daily volatility, and $Y$ is a constant of order 1 that depends on the market.

Take an imaginary token with daily volatility $\sigma$ = 6% and suppose you want to buy 25% of a day's volume ($Q/V$ = 0.25), with $Y$ = 1. (Illustrative parameters — the model is real, the inputs are invented.)

- impact ≈ 1.0 × 6% × √0.25 = 1.0 × 6% × 0.5 = **3.0%**

Now the same token, but you size to 2.8% of daily volume:

- impact ≈ 0.5 × 6% × √0.0278 = 0.5 × 6% × 0.1667 ≈ **0.5%**

**The intuition:** impact is driven by your size *relative to the market's*, and because the relationship is a square root rather than a straight line, going from a small order to a large one costs you far more than proportionally. On listing day, $V$ for the first hour is essentially undefined — the market has no history — which is precisely why listing-day slippage is so violent. You are sizing against a denominator that does not exist yet.

### The anatomy of a listing pump

Put the pieces together and you get a shape that recurs.

![The five phases of a listing round trip, from pre-announcement rumour to the thirty-day price.](/imgs/blogs/exchanges-are-players-not-just-venues-5.webp)

The illustrative path in the figure: a rumour leg at T−3d, the announcement at T−0 with the token at \$0.40, a peak of \$0.62 twenty minutes after the open, then \$0.53 at one hour, \$0.44 at 24 hours, \$0.31 at seven days, \$0.22 at thirty days.

Two structural forces make the shape:

- **Supply arrives late.** Everyone who held the token cheaply — seed investors past their cliff, airdrop recipients, the market maker holding an inventory loan — now has a deep book to sell into for the first time. Listing day is the first day their position is actually liquid. Liquidity is what a seller needs.
- **Demand arrives all at once and then stops.** The listing is a single event. The buyers it summons are the buyers who were waiting for it. Once they have bought, the flow is gone, and the sellers are still there.

#### Worked example 4: what the retail buyer actually earns

Take the illustrative path above — again, an invented price series for an imaginary token, not a measured average across real listings — and put a person on it. You see the announcement, you decide to buy, and by the time your order lands the token is running.

**Step 1 — your fill.** You send a market order during the first twenty minutes. You do not get \$0.40. You get filled around **\$0.55** — between the open and the \$0.62 peak, which is where most of the first-hour volume trades.

**Step 2 — fees.** A 0.10% taker fee on top:

- \$0.55 × 1.001 = **\$0.5505** effective cost per token

**Step 3 — mark to market.** Using the path above and taking 0.10% off on the way out:

| When | Price | Your P&L |
|---|---|---|
| T+24h | \$0.44 | **−20.2%** |
| T+7d | \$0.31 | **−43.7%** |
| T+30d | \$0.22 | **−60.1%** |

On a \$5,000 stake, the seven-day mark is about **\$2,813**.

**Step 4 — the counterfactual.** The same \$5,000 deployed at T+30d buys at \$0.22 rather than \$0.5505 — 2.5× as many tokens for identical cash.

**The intuition:** the listing pop is not a return available to the public; it is a transfer, and the public is usually on the paying side of it. The variable that determined your outcome was not whether the project was good. It was your position in the queue.

None of this requires anyone to have cheated. It is what happens when a fixed float meets a scheduled demand event.

### Listing fees: the contested part

Here the record gets genuinely disputed, and it is worth being careful.

The public argument broke open at the end of October 2024, on X, over several days. As reported at the time: Simon Dedic of Moonrock Capital alleged that a top-tier exchange had asked a project for a very large share of its token supply in exchange for a listing. Andre Cronje, the developer behind Fantom and Sonic, responded that exchanges had quoted his projects listing costs ranging from the high six figures into the tens of millions, and said that Coinbase had never asked him for a fee. Coinbase's chief executive Brian Armstrong then stated publicly that Coinbase does not charge listing fees. Binance publicly disputed the characterisation and has maintained that it does not charge listing fees, and that any token deposit associated with a listing is not a fee to the exchange.

**These are competing public statements, not findings.** No regulator has adjudicated them; no exchange has been found to have charged what was alleged; the individuals named made their statements publicly and are reported here as making them. This post asserts neither side. And note what nobody produced during the argument: a document. What is worth extracting, then, is not who is right but *why the number would matter if a fee did exist*.

#### Worked example 5: what a supply-denominated fee is worth

Fees in crypto listings are alleged to be quoted in *supply*, not dollars — a percentage of the token's total supply. That framing is what makes the numbers explosive. The three cases below are illustrative arithmetic on invented percentages and valuations; none is a reported fee for any real listing.

- A fee of **2% of supply** on a token that lists at a \$1.0B FDV = 0.02 × \$1,000,000,000 = **\$20,000,000**
- A fee of **5% of supply** on a token that lists at a \$400M FDV = 0.05 × \$400,000,000 = **\$20,000,000**
- A fee of **15% of supply** on a token that lists at a \$1.0B FDV = 0.15 × \$1,000,000,000 = **\$150,000,000**

**The intuition:** a supply-denominated fee is not a payment, it is a position — and a position vests, unlocks, and eventually sells. A cash fee of \$20M is a cost to the project. A 2%-of-supply fee is \$20M of *future selling pressure sitting above every buyer*, on a schedule the buyer cannot see. The two look identical on an invoice and are completely different on a chart. That is why the question "does this exchange take supply?" belongs on your checklist, whatever the answer turns out to be.

The general lesson is the one from [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table): every entity holding tokens below your price is a future seller, and the identity of the entity tells you when.

## Hat 4 — the exchange as counterparty

Can the venue be on the other side of your trade?

This is where the public record is richest, because regulators have made specific allegations. Handle them precisely.

- **The CFTC's complaint against Binance**, filed 27 March 2023 in the Northern District of Illinois, alleged among other things that Binance operated trading accounts of its own on its own platform and did not adequately disclose that fact or police the resulting conflicts. Binance settled with the CFTC as part of the broader November 2023 resolution.
- **The SEC's complaint against Binance**, filed 5 June 2023 in the District of Columbia, alleged that entities controlled by Binance's founder — named in the complaint as **Sigma Chain AG** and **Merit Peak Ltd** — engaged in wash trading that inflated reported volume on the Binance.US platform, and that Merit Peak acted as an undisclosed market maker. The SEC subsequently moved to dismiss this case, and it was dismissed with prejudice in 2025 — a dismissal, note, reflects the agency's enforcement posture, not a factual finding that the allegations were false.
- **The Department of Justice resolution announced 21 November 2023** saw Binance agree to pay approximately **\$4.3 billion** across DOJ, FinCEN and OFAC components, and founder Changpeng Zhao plead guilty to failing to maintain an effective anti-money-laundering programme. He was sentenced to **four months in prison on 30 April 2024**. The AML charges are not conflict-of-interest charges — but the resolution establishes that the entity in question was, on the public record, running a venue without the controls a regulated venue is required to have.
- **Reported, not adjudicated:** in May 2024 the Wall Street Journal reported that Binance's internal monitoring team had flagged suspected wash trading by the market-making firm DWF Labs and that the investigation was not pursued as the team recommended. Binance disputed the characterisation and DWF Labs denied wrongdoing. [Wash trading, spoofing, and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) covers the detection methods that make such claims testable.

On the other side of the ledger, exchanges do publish policies. Coinbase has long stated that it does not trade for its own account in the proprietary, directional sense, while acknowledging it holds crypto on its balance sheet and conducts certain hedging and operational transactions; the SEC's own June 2023 case against Coinbase — which concerned whether listed tokens were securities, not conflict of interest — was dismissed with prejudice in February 2025.

The synthesis a careful reader should take away:

1. **Being the counterparty is not inherently improper.** Many regulated venues have affiliated liquidity providers. What regulation demands is *disclosure and separation*, not abstinence.
2. **The disclosure is what is missing, not the activity.** In US equities, an ATS operator's own trading is a Form ATS-N line item. In crypto, it is whatever the operator chooses to tell you.
3. **You can partly test it yourself.** The manipulation fingerprints in [detecting manipulation: on-chain red flags](/blog/trading/crypto-players/detecting-manipulation-onchain-red-flags) and the volume-quality checks in the wash-trading post are the closest thing retail has to an audit.

And the thing to hold onto from [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does): the firm quoting your token has a contract with the project, sometimes an option struck at a price you cannot see, and a relationship with the venue. You are trading against a network, not against a crowd.

## Hat 5 — the venue's own token

Most large exchanges issue a token: BNB, OKB, KCS, CRO, and — until it took the whole company down — FTT. Understanding what these are is the single most under-taught thing in retail crypto.

### What an exchange token actually is

An exchange token is typically **not** equity. It carries no legal claim on profits, no vote on the board, no residual claim in liquidation, no dividend. What it usually carries is:

- **Fee discounts** on the venue.
- **Access rights** — allocation in launchpad sales, priority in campaigns.
- Sometimes **gas utility**, if the exchange runs a chain.
- And a **buyback-and-burn** commitment: the exchange uses some portion of its revenue or profit to buy the token on the open market and destroy it, shrinking supply.

That last mechanism is doing all the valuation work, and it is an *engineered* analogue of a buyback. In equities, a buyback shrinks the share count so each remaining share owns a larger slice of a legally enforceable claim. In crypto, a burn shrinks the token count so each remaining token owns a larger slice of… a promise. The promise can be changed by the issuer, and has been: Crypto.com burned a very large quantity of CRO in 2021 and then, in 2025, put a re-issuance to a community vote — an episode worth reading about precisely because it demonstrates that "burned forever" is a policy, not a property of the asset.

BNB is the most developed version. Binance's original whitepaper committed to eventually removing 100 million of the initial 200 million BNB from supply. That is now executed through two channels: a quarterly **Auto-Burn**, whose size is determined by a published formula driven by BNB Chain block production and the average BNB price, and **BEP-95**, a real-time burn of a portion of gas fees on BNB Chain live since November 2021.

### Divide the burn by the market cap

Whatever the mechanism, the valuation question is a division problem, and you can do it yourself.

![Turning venue volume into a burn yield: the one division that tells you what an exchange token's buyback is actually worth.](/imgs/blogs/exchanges-are-players-not-just-venues-6.webp)

#### Worked example 6: valuing an exchange token's burn

Illustrative figures, chosen to be round; substitute the real ones for whatever token you are looking at.

**Step 1 — venue revenue.** The exchange does **\$10 billion of spot volume per day**. Blended realised fee across all tiers — remember most volume comes from VIP tiers paying far less than retail — call it **0.04%** (4 bps):

- \$10,000,000,000 × 0.0004 = **\$4,000,000 per day**
- × 365 = **\$1.46 billion per year**

**Step 2 — profit.** At a 60% operating margin:

- \$1,460,000,000 × 0.60 = **\$876,000,000 per year**

**Step 3 — the burn.** The commitment is 20% of profit to buyback-and-burn:

- \$876,000,000 × 0.20 = **\$175,200,000 per year burned**

That is a genuinely large number. Now do the division.

**Step 4 — the burn yield.** If the token's market capitalisation is **\$80 billion**:

- \$175,200,000 ÷ \$80,000,000,000 = **0.219%**, call it **0.22% per year**

**Step 5 — compare.** The S&P 500's aggregate buyback yield — total buybacks divided by index market capitalisation, as tracked by S&P Dow Jones Indices — has run in the low single digits, broadly the 1.5–2.5% range across the past decade. A 0.22% burn yield is a fraction of that, for an instrument with no legal claim, no audited accounts, and full issuer discretion over the policy.

**Step 6 — where it would be interesting.** Hold the burn constant and shrink the market cap to \$8 billion:

- \$175,200,000 ÷ \$8,000,000,000 = **2.19%**

**The intuition:** the burn is real, and the yield on it is almost always a rounding error at the valuations these tokens trade at. When someone tells you an exchange token is "backed by burns," they have named the numerator. Ask for the denominator.

### The FTT lesson: a token backed by your own promise

The most expensive demonstration of what an exchange token is came in November 2022.

FTX's token, FTT, was designed with a buyback-and-burn funded by a share of exchange fees. Its market value was therefore a function of FTX's volume — that is, of FTX's continued health. On **2 November 2022**, CoinDesk published a report on a leaked balance sheet from Alameda Research, the trading firm affiliated with FTX, which reportedly showed that a very large share of Alameda's assets consisted of FTT — the token issued by its sister exchange — including a line held as collateral.

The circularity is the whole story: a trading firm's balance sheet was supported by a token whose value depended on the exchange that the same people ran, and which could not be sold in size without collapsing its own price. On **6 November 2022** Binance's founder announced publicly that Binance would liquidate its FTT holdings. FTT collapsed. Customers withdrew. FTX filed for Chapter 11 on **11 November 2022**. Sam Bankman-Fried was convicted on seven counts on **2 November 2023** and sentenced to **25 years in prison on 28 March 2024**.

[The FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) and [Alameda Research: the cautionary tale](/blog/trading/crypto-players/alameda-research-the-cautionary-tale) go through the mechanics. For this post, the extracted lesson is narrow and permanent: **an exchange token is a leveraged claim on the exchange, and it is worth the most exactly when you least need it and zero exactly when you most do.** Its correlation with your other risk is 1 at the worst possible moment.

## Hat 6 — the custodian

The last hat is the quietest and the one that determines whether the other five can hurt you.

When you hold coins on a centralised exchange, you do not hold coins. You hold a database entry: a claim on the company. The coins, if they exist, are in the company's wallets, commingled with everyone else's, on the company's balance sheet. In a bankruptcy you are, in most jurisdictions and absent a specific statutory or contractual trust arrangement, an **unsecured creditor** — behind secured lenders, in line with everyone else, waiting years.

This is exactly what US broker-dealer rules exist to prevent. **Rule 15c3-3** under the Exchange Act — the customer protection rule — requires a broker-dealer to segregate customer securities and to maintain a reserve of cash or qualified securities for customer credits, so that the firm cannot finance itself with your assets. Futures commission merchants face parallel segregation requirements under CFTC rules. In both cases the point is that customer property is *not* the firm's property, and cannot become part of the estate.

Crypto's answer to this has been **proof of reserves**: a periodic cryptographic attestation, usually a Merkle-tree construction, showing that the exchange's on-chain holdings cover the sum of customer balances. It is a genuine improvement and it has a genuine limitation — it proves assets at a moment in time, and says much less about **liabilities**. An exchange can prove it holds the coins and still be insolvent if it owes more elsewhere, and a snapshot can be taken on a day chosen by the firm. Read attestations as evidence, not proof.

Custody also concentrates. When US spot bitcoin exchange-traded funds began trading in January 2024, Coinbase's institutional custody arm was named as custodian by a large share of the issuers — a fact you can check yourself, because each fund names its custodian in its own prospectus and continues to do so in its periodic reports. The consequence is that a substantial portion of institutional bitcoin exposure runs through one operator's infrastructure. That is a different risk from the conflicts above — it is single-point-of-failure risk rather than conflict risk — but it belongs on the same map, because it is the same firm.

The practical rule is old and boring and correct: **the amount you keep on an exchange should be the amount you need to trade, and no more.**

## The other direction: tags, delistings, and the withdrawal of attention

Everything so far has been about the exchange granting access. The mirror image is at least as powerful and gets a fraction of the attention: the exchange can *withdraw* it, in stages, without ever delisting anything.

Binance operates label systems — a **Seed Tag** for early-stage or higher-volatility assets and a **Monitoring Tag** for tokens under heightened review, both of which surface a warning to the user and require an explicit risk acknowledgement before trading. Coinbase has used trading-suspension and "experimental" labels. Korean exchanges including Upbit apply investment-warning designations under the local framework. In every case the label is not a statement that the token is bad. It is a change in the token's *distribution*.

![Four escalating exchange actions, what each one changes, and what it costs a holder.](/imgs/blogs/exchanges-are-players-not-just-venues-8.webp)

The escalation ladder runs roughly: a tag → a reduction in available leverage and futures limits → removal of quote-asset pairs → full delisting. Each rung reduces the token's reachable buyer base and its available liquidity, and liquidity is a cost you pay every time you transact.

#### Worked example 7: what losing a venue costs you

Illustrative. You hold a **\$200,000** position in a token with **\$12M** of daily volume across five venues, **60%** of it (\$7.2M) on the exchange that is about to tag it. Daily volatility is 6%. Use the square-root impact model from earlier with $Y$ = 0.5.

**Step 1 — round-trip cost today.** Half-spread is 7.5 bps (a 15 bps quoted spread). Your size relative to on-venue volume:

- \$200,000 ÷ \$7,200,000 = **0.0278**
- impact = 0.5 × 6% × √0.0278 = 0.5 × 6% × 0.1667 = **0.50%**
- round trip = 2 × (0.075% + 0.50%) = **1.15%** → **\$2,300**

**Step 2 — after the tag.** Volume on the venue falls 45% to \$3.96M as mandate-constrained funds step back and leverage limits force levered holders out. The quoted spread widens from 15 bps to 70 bps, so half-spread is 35 bps:

- \$200,000 ÷ \$3,960,000 = **0.0505**
- impact = 0.5 × 6% × √0.0505 = 0.5 × 6% × 0.2247 = **0.674%**
- round trip = 2 × (0.35% + 0.674%) = **2.05%** → **\$4,097**

**Step 3 — the difference.** \$4,097 − \$2,300 = **\$1,797**, or **0.9% of the position**, purely from a label. Nothing about the token's code, team, or revenue changed.

**The intuition:** liquidity is a position you hold without realising it, and the exchange can mark it down unilaterally. This is why a delisting announcement produces an immediate price gap that has nothing to do with fundamentals — holders are repricing their exit cost, all at once.

## What TradFi forbids, and why

It is easy to read the six hats and conclude that traditional finance is simply better behaved. That is the wrong lesson. Traditional finance is not better behaved; it is more *constrained*, and every constraint was written after somebody did the thing the constraint now forbids.

![The functions traditional markets split across separately regulated firms, and the single entity that holds all of them in crypto.](/imgs/blogs/exchanges-are-players-not-just-venues-2.webp)

Run the map against the rulebook:

**The venue must file its rules.** A US national securities exchange registers under **Section 6** of the Exchange Act, and under **Section 19(b)** any change to its rules — including listing standards and fee schedules — must be filed with the SEC and is subject to public comment and Commission review. A crypto exchange changes its listing standards by editing a page.

**An exchange that lists its own stock needs special machinery.** This is the closest direct analogue to Hat 5 and it is instructive. When Nasdaq and NYSE became publicly traded and listed their own shares — Nasdaq through the 2000s, NYSE after the 2006 Archipelago merger — the SEC did not simply allow it. Each had to put in place conflict-mitigation arrangements so that the exchange was not the sole judge of its own compliance with its own listing standards: independent regulatory oversight, delegation of the surveillance and enforcement function to an independent regulator, and periodic reporting to the Commission. The principle is explicit: *an issuer may not also be the unsupervised regulator of its own listing.* No equivalent constraint exists on an exchange token.

**Proprietary trading by deposit-taking institutions is restricted.** The **Volcker Rule**, Section 619 of the Dodd-Frank Act, restricts banking entities from short-term proprietary trading and from certain fund relationships. It exists because the 2008 crisis demonstrated what happens when the institution holding customer money also runs a book. It does not apply to crypto exchanges.

**Research and underwriting are walled off from trading.** After the dot-com era's conflicted equity research, the April 2003 **Global Research Analyst Settlement** — roughly \$1.4 billion across the major banks — imposed structural separation between research and investment banking, later codified in FINRA's research rules. The reason is exactly Hat 2: an institution that owns a position should not also be the institution telling the public what the position is worth.

**Customer assets are segregated.** Rule 15c3-3, as above.

**Operator trading on an ATS must be disclosed.** Form ATS-N, adopted in 2018 for NMS-stock alternative trading systems, requires public disclosure of the broker-dealer operator's activities on its own venue and the conflicts that arise. It exists because "we have information barriers" was not considered a sufficient answer.

The picture in crypto is changing, unevenly. The EU's **Markets in Crypto-Assets Regulation** (Regulation (EU) 2023/1114, with the crypto-asset service provider regime applying from 30 December 2024) imposes conflict-of-interest obligations on service providers and constrains a platform operator from trading on its own account on the platform it operates. IOSCO's November 2023 policy recommendations for crypto and digital asset markets put conflicts arising from vertically integrated business models near the top of the list, precisely because one entity performing exchange, broker, custodian and proprietary-trading functions is the defining structural feature of the industry. Hong Kong's virtual-asset trading platform licensing regime and Singapore's framework take related approaches. The United States, as of mid-2026, has been working through market-structure legislation and a rewritten regulatory posture; the shape of the eventual rule on exchange conflicts was not settled at the time of writing.

**None of that helps you today.** The reason to know the TradFi rulebook is not to demand it, but to use it as a checklist: every separation on the left column of the diagram is a question you can ask about the venue in front of you.

## Common misconceptions

**"A listing means the exchange has vetted the token."** It means the token passed that exchange's listing process, which is designed primarily around legal, operational and compliance risk *to the exchange*. It is not an assessment that the token is a good investment, will hold value, or is fairly priced at the listing print. Certification and endorsement are different things, and exchanges are careful to say so in their own disclaimers.

**"The exchange makes money when the price goes up."** The exchange makes money when *volume* goes up. Volume rises in crashes too — often more than in rallies, because liquidations are volume. The venue's revenue is closer to a claim on volatility than a claim on price. This is why "the exchange is pumping it" is usually the wrong model; the right model is "the exchange benefits from anything that makes you trade."

**"Delisting means the project failed."** Delisting decisions turn on regulatory exposure, volume thresholds, engineering burden, and the exchange's own risk appetite in a given jurisdiction. Perfectly functional projects get delisted for reasons that are about the exchange, not about them. What is true is that once delisted, the token's exit cost rises immediately and permanently — which is why the *price* reaction is real even when the *judgement* embedded in the decision is not about quality.

**"Proof of reserves means my funds are safe."** It means the exchange demonstrated control of certain assets at a certain moment. It typically says much less about liabilities, about off-balance-sheet obligations, or about what happens between attestations. A solvency statement needs both sides of the balance sheet, audited by someone with something to lose.

**"Exchange tokens are like exchange stock."** Almost never. Equity carries a legal residual claim, enforceable in court, with audited financial statements behind it. An exchange token typically carries a discount, some access rights, and a burn policy the issuer controls. The economics can superficially resemble a buyback; the legal position does not resemble a share at all.

**"If the exchange's venture arm invested, the token must be good."** It means a professional investor bought at a price you cannot get, on terms you cannot see, with an unlock schedule that sits above your entry. That is information — but the direction it points is not the one most readers assume. See [worked example 1](#worked-example-1-the-ventures-arm-return): the investor's exit is a three-year selling programme.

## How it shows up in real markets

**FTX and FTT (2022).** The canonical case of the hats composing. One group ran the exchange, an affiliated trading firm, and the token that both depended on. When CoinDesk reported on 2 November 2022 that Alameda's balance sheet was heavily composed of FTT, the circularity became visible; Binance announced on 6 November that it would sell its FTT; FTX filed for bankruptcy on 11 November. Customers discovered what "unsecured creditor" means. Sam Bankman-Fried was convicted on 2 November 2023 and sentenced to 25 years on 28 March 2024.

**Binance's 2023 resolutions.** Three separate US actions in one year — the CFTC complaint of 27 March 2023, the SEC complaint of 5 June 2023, and the roughly \$4.3 billion DOJ/FinCEN/OFAC resolution announced 21 November 2023 at which the founder pleaded guilty to an anti-money-laundering failure and was later sentenced to four months. The conflict-relevant allegations — house accounts trading on the platform, an affiliated undisclosed market maker, wash trading inflating reported volume — appear in the CFTC and SEC complaints. The SEC's case was dismissed with prejudice in 2025; the allegations were never adjudicated.

**The Coinbase insider-trading case (2022–2023).** A product manager with advance knowledge of listings was prosecuted for trading on it; he was sentenced to two years in May 2023, his brother to ten months in January 2023. The case is the clearest public evidence that the listing information ladder is monetisable, and that the people on it know it.

**The October–November 2024 listing-fee argument.** A public dispute between fund managers, a prominent developer, and the two largest exchanges over whether Tier-1 listings carry fees denominated in token supply. Claims and denials were exchanged on X within days. Nothing was adjudicated. What it demonstrated is the absence of a disclosure regime: in equity markets, listing fees are published in an exchange's rule filings, and the question could be answered by reading a document rather than by watching an argument.

**The DWF Labs reporting (2024).** In May 2024 the Wall Street Journal reported that Binance's internal surveillance team had flagged suspected wash trading by a market-making firm and that its recommendations were not acted on; Binance disputed the account and DWF Labs denied wrongdoing. It is reported, not proven. Its relevance here is that the surveillance function and the commercial relationship sat inside the same company — which is precisely the separation that a self-regulatory structure with an independent regulator is designed to enforce.

**Exchange-token supply policy changes.** Crypto.com's large 2021 burn of CRO, followed in 2025 by a widely reported proposal and community vote to re-issue tokens, is the cleanest demonstration that a burn is a policy rather than a property of the asset. Check the exact quantities and the vote outcome at the source before relying on them; the point that survives any correction to the figures is structural. Holders who had modelled a permanently declining supply were, in effect, modelling a decision that the issuer retained the power to revisit.

**Any small-cap listing you can find, this week.** The most useful case study is the one you run yourself. Pick a token listed on a major venue in the past six months. Chart the price from three days before the announcement to thirty days after. Then look up the seed round, the FDV, the unlock schedule, and whether the exchange's venture arm is on the cap table. You will have reproduced this post from primary data in about twenty minutes.

## What to actually do with this map

You cannot fix the structure. You can decide where in it you are standing. This is educational material, not investment advice — but the checklist below is mechanical, and mechanical things are checkable.

![Four questions that determine where a retail buyer sits in the listing queue.](/imgs/blogs/exchanges-are-players-not-just-venues-9.webp)

**1. Ask how you learned about it.** If the source is the exchange's own announcement, you are at the end of the information chain by construction, not by bad luck. Everything from the T−90d application onward already happened. The pop, if there is one, is being sold to you by people who have been positioned for months.

**2. Never send a market order into the first hour.** This is the single highest-value habit in this post. The first hour of a listing has no meaningful volume history, so you cannot size against it, and the book is at its thinnest. Worked example 2 is what that costs: a \$0.72 average fill on a \$1.10 print. If you must participate, use limit orders and accept that you may not be filled.

**3. Size to daily volume, not to your wallet.** Use the square-root rule. If your intended order is more than a low single-digit percentage of the token's genuine daily volume — genuine, after you have discounted for the wash-trading fingerprints — expect meaningful impact and split the order across time.

**4. Read the cap table before the chart.** Who bought at seed? Is the exchange's venture arm among them? What is the cliff and the vesting schedule? Every locked token below your price is a scheduled seller, and you can read the schedule. The methodology is in [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table).

**5. Do the burn division yourself.** For any exchange token: annual burn in dollars ÷ market cap. If the answer is a fraction of a percent, you are buying a narrative and an access right, which may be fine — but know which one you are buying.

**6. Treat the venue as a counterparty, not a bank.** Keep on-exchange only what you are actively trading. Read proof-of-reserves attestations as evidence about assets and be explicit with yourself that they say little about liabilities.

**7. Watch the tags, not just the delistings.** A monitoring tag is an early, public, machine-readable signal that the venue's own risk team has downgraded the asset. It is one of the few pieces of exchange-internal judgement they publish. Worked example 7 is what ignoring it costs.

**8. Ask the six questions.** For any venue you use: Does it run a venture arm? Does that arm hold tokens listed here? Does it publish its listing criteria? Does it disclose affiliated trading? Does it issue a token? Where are customer assets held, and under what legal arrangement? You will often not get answers. *The absence of an answer is itself an answer*, and it is the cheapest research you will ever do.

The reason to build this map is not cynicism. Exchanges do genuinely useful things — they aggregate liquidity, they filter out a great deal of fraud, they make an asset class usable by people who should not be managing their own keys. The map exists so you can use them for what they are good at while knowing, precisely, which of your interests they are not structurally obliged to protect.

For where this fits in the wider hierarchy of who actually moves crypto prices, [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) is the companion piece, and [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) is the mechanics underneath all of it. The two exchange profiles in this series — [Binance: the everything exchange and its gravity](/blog/trading/crypto-players/binance-the-everything-exchange-and-its-gravity) and [Coinbase: the compliant giant](/blog/trading/crypto-players/coinbase-the-compliant-giant) — take these six hats and fit them to two very different firms.

## Sources & further reading

Where a claim in this post is contested, it is described above as reported or alleged, with the source and the response. Nothing here asserts wrongdoing beyond the public record, and dismissals and denials are noted where they exist.

- **U.S. Commodity Futures Trading Commission**, complaint against Binance Holdings Ltd., Changpeng Zhao and Samuel Lim, filed 27 March 2023 (N.D. Ill.); CFTC press release 8680-23 (27 March 2023) and the November 2023 consent order.
- **U.S. Securities and Exchange Commission**, complaint against Binance Holdings Ltd. et al., filed 5 June 2023 (D.D.C.) — the allegations regarding Sigma Chain AG and Merit Peak Ltd. Case dismissed with prejudice by joint stipulation in 2025.
- **U.S. Department of Justice**, "Binance and CEO Plead Guilty to Federal Charges in \$4B Resolution" (21 November 2023); sentencing of Changpeng Zhao, 30 April 2024 (W.D. Wash.).
- **U.S. Securities and Exchange Commission**, complaint against Coinbase, Inc. (6 June 2023, S.D.N.Y.); dismissed with prejudice February 2025.
- **U.S. Department of Justice / SEC**, *United States v. Ishan Wahi* and *SEC v. Wahi* (July 2022); sentencing reported May 2023 (Ishan Wahi) and January 2023 (Nikhil Wahi).
- **CoinDesk**, Ian Allison, "Divisions in Sam Bankman-Fried's Crypto Empire Blur on His Trading Titan Alameda's Balance Sheet" (2 November 2022) — the report that began the FTX collapse; FTX Chapter 11 petition, District of Delaware, 11 November 2022; *United States v. Bankman-Fried* verdict 2 November 2023, sentencing 28 March 2024 (S.D.N.Y.).
- **The Wall Street Journal**, reporting on Binance's internal surveillance findings regarding DWF Labs (May 2024). Binance disputed the characterisation; DWF Labs denied wrongdoing.
- **Securities Exchange Act of 1934**, Sections 6 and 19(b); **Rule 15c3-3** (customer protection); **Regulation ATS / Form ATS-N** (adopted 2018) — operator-conflict disclosure for NMS-stock ATSs.
- **Dodd-Frank Wall Street Reform and Consumer Protection Act**, Section 619 (the Volcker Rule).
- **Global Research Analyst Settlement** (SEC, NASD, NYSE and state regulators, April 2003) — the structural separation of research from investment banking; subsequently codified in FINRA research-conflict rules.
- **Regulation (EU) 2023/1114** (Markets in Crypto-Assets, "MiCA") — conflict-of-interest obligations for crypto-asset service providers and constraints on a trading-platform operator dealing on its own account; CASP regime applying from 30 December 2024.
- **IOSCO**, *Policy Recommendations for Crypto and Digital Asset Markets*, Final Report (November 2023) — recommendations addressing conflicts arising from vertically integrated business models.
- **Binance**, BNB whitepaper (the original 100M-of-200M burn commitment), the quarterly Auto-Burn methodology, and BEP-95 (real-time burn, live since November 2021); Binance's Seed Tag and Monitoring Tag documentation.
- **Market-impact literature**: Robert Almgren et al., "Direct Estimation of Equity Market Impact" (2005); Jean-Philippe Bouchaud et al., *Trades, Quotes and Prices* (Cambridge University Press, 2018) — the square-root impact law used in worked examples 3 and 7.
- On this blog: [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) · [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) · [the lifecycle of a token: seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) · [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) · [centralized crypto exchanges: Binance and Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) · [crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers).
