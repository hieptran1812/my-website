---
title: "Coinbase: The Compliant Giant"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A build-from-zero profile of Coinbase as a crypto player — how a public company's audited disclosures make it the only major exchange you can actually check, where its money really comes from (retail take rate, institutional volume, and an interest-rate bet dressed as a stablecoin), how the listing machine works and why a Coinbase listing once moved price, what Base and ETF custody turned it into, and how the SEC case that was supposed to define crypto law ended without a ruling."
tags: ["crypto", "coinbase", "crypto-players", "exchanges", "stablecoins", "usdc", "bitcoin-etf", "regulation", "sec", "layer-2", "custody", "market-structure"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Coinbase is the only large crypto venue whose books you can actually read, and that single fact — being a US-listed public company with audited filings — is simultaneously its product, its moat, and the source of its most interesting conflicts.
>
> - **The model:** Coinbase sells *legitimacy*. Institutions, ETF issuers and regulated funds use it not because it is cheapest or deepest, but because it is the counterparty a compliance officer can sign off on.
> - **The money:** roughly two-thirds of revenue is trading fees and one-third is "subscription and services" — and the largest piece of that third is interest income on USDC reserves. A big chunk of a crypto company's profit is really a bet on short-term interest rates.
> - **The asymmetry:** retail volume pays roughly a 1%-ish take rate, institutional volume pays single-digit basis points. One retail dollar of volume is worth something like fifty institutional dollars.
> - **The four seats:** Coinbase is a venue, a custodian, an investor (Coinbase Ventures) *and* — since Base — an issuer of blockspace. The same token can pass through all four.
> - **The legal turn:** the SEC sued Coinbase on **6 June 2023** and the case was dismissed **with prejudice on 27 February 2025**, with no penalty and no ruling on the merits. Two years of existential risk ended in a stipulation, not a doctrine.
> - **The number to remember:** **8 of the 11** US spot bitcoin ETFs approved in January 2024 named Coinbase Custody as custodian. Diversifying issuers concentrated on one point of failure.

Every other profile in this series has the same problem. You want to know how much money Wintermute makes, what Jump's crypto book looked like before Terra, whether DWF Labs is a market maker or a prop desk with a marketing department — and you cannot know, because none of them have to tell you. You are reading tea leaves: on-chain wallet clusters, leaked term sheets, a founder's podcast boast, a bankruptcy filing that accidentally reveals a counterparty.

Then there is Coinbase, which files a 10-K.

That is the whole story in one sentence. Coinbase made a bet, very early and very expensively, that in crypto the scarce resource would not be liquidity or technology but *permission* — and that the firm which could be legally boring would eventually be handed the parts of the market that institutions are allowed to touch. The bet has been ridiculed in every bull market and vindicated in every bust. It has also produced a company with a genuinely strange income statement, a set of conflicts that no amount of disclosure fully resolves, and a pile of public documents that happen to be the best free dataset about the crypto industry that exists.

![The four seats Coinbase occupies in the crypto market: venue, custodian, issuer of blockspace, and investor](/imgs/blogs/coinbase-the-compliant-giant-1.webp)

The diagram above is the mental model for this entire piece. Most firms in crypto hold one of those four seats. Coinbase holds all four, and the interesting questions all live where they touch.

A note on numbers before we start, because it matters for how you should read what follows. Coinbase's headline financials are cited to its own filings and shareholder letters and are given as approximate, dated figures. Where a precise line item could not be re-verified at the time of writing, I have used an **explicitly labelled illustrative figure** and shown the mechanism instead — those are marked in the text. Nothing here is a reported number dressed up as precision it does not have. This is educational writing, not investment advice.

## Foundations: the words you need before any of this makes sense

If you have never read a company's financial statements, and you are not sure what a "take rate" or a "custodian" is, this section is for you. If you already know, skim to the next heading.

**An exchange** is a place where buyers and sellers meet. In crypto, a *centralized* exchange like Coinbase is much more than a meeting place: it holds your money, holds your coins, matches your orders in its own internal database, and settles the trade by editing two rows in that database. No blockchain is involved in a trade between two Coinbase users. This matters enormously and we will come back to it.

**Custody** means holding an asset on someone else's behalf. If you own bitcoin "on Coinbase", what you actually own is a *claim* against Coinbase, recorded in Coinbase's ledger. The bitcoin itself sits in wallets Coinbase controls. A **custodian** is a firm whose job is holding assets safely for others — a boring, licensed, insured business that predates crypto by a century.

**A take rate** is the fraction of the money that flows through a platform that the platform keeps. If you trade \$1,000 of bitcoin and the platform earns \$15, the take rate is 1.5%. It is the single most important number in any marketplace business, and it is usually not disclosed directly — you compute it by dividing revenue by volume.

**A basis point** ("bp" or "bip") is one hundredth of one percent — 0.01%. Institutional trading fees are quoted in basis points because they are so small. Three basis points is 0.03%.

**A 10-K** is the annual report every US-listed public company must file with the Securities and Exchange Commission (the SEC, the US markets regulator). It is audited by an independent accounting firm, it must disclose revenue by segment, related-party transactions, risk factors and every material legal proceeding, and if it contains a knowing falsehood, executives go to prison. A **10-Q** is the quarterly version. A **shareholder letter** is the friendlier document a company publishes alongside its results — not audited, but still subject to securities-fraud liability if it lies.

**A stablecoin** is a token designed to always be worth one dollar. The honest kind works like a money-market fund: someone gives the issuer \$1, the issuer holds that \$1 in short-term US government debt, and issues one token. The issuer keeps the interest. We cover the species in detail in [stablecoins, Tether, Circle and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar).

**A Layer 2** ("L2") is a separate blockchain that borrows Ethereum's security. Transactions happen cheaply on the L2, and the L2 periodically posts a compressed summary back to Ethereum. The software that orders transactions on the L2 is called the **sequencer**, and whoever runs the sequencer collects the fees.

**A Wells notice** is a letter from the SEC telling a company that the enforcement staff intends to recommend charges, and inviting it to argue why they shouldn't. It is the last stop before a lawsuit.

That is the vocabulary. Now the interesting part.

## What it actually means that you can read the books

![Comparison of what a US-listed exchange must disclose versus what an offshore exchange chooses to disclose](/imgs/blogs/coinbase-the-compliant-giant-2.webp)

Coinbase went public on **14 April 2021** through a direct listing on Nasdaq under the ticker **COIN** — not a traditional IPO, but the same disclosure obligations either way. In **May 2025** it was added to the S&P 500 index, the first crypto-native company to join, which mechanically forced every S&P 500 index fund in the world to buy some.

The consequence people underrate is not prestige. It is *legal exposure to its own statements*.

When Binance said it held customer assets one-to-one, that was a claim. When a smaller exchange published a "proof of reserves" attestation, that was a snapshot at a moment of the exchange's choosing, produced by an accountant the exchange hired, with no obligation to show liabilities. Neither is worthless. Neither is an audit. The whole architecture of pre-2023 crypto — and the reason FTX could be a \$32 billion company with, functionally, no accounting department — rested on the fact that nobody could compel disclosure.

A 10-K compels it. And the specific things it compels are the things that killed everyone else:

- **Segregation of customer assets.** A 10-K must describe how customer crypto is held, whether it is commingled, and what happens to it in a bankruptcy. Coinbase's own disclosure on this point caused a small panic: in a quarterly filing in 2022, prompted by a new SEC accounting bulletin, Coinbase disclosed that in the event of bankruptcy, custodially held crypto assets *could* be treated as property of the bankruptcy estate and customers *could* be treated as general unsecured creditors. The stock fell; the CEO said publicly there was no risk of bankruptcy and that the disclosure was a required new risk factor, not a change in how assets were held. It was, in retrospect, an advertisement for the system working. Nobody had to disclose that at FTX, and so nobody did, and so nobody knew.
- **Related-party transactions.** Loans between the exchange and an affiliated trading firm — the exact mechanism that destroyed FTX and Alameda, which we cover in [Alameda Research: the cautionary tale](/blog/trading/crypto-players/alameda-research-the-cautionary-tale) — must be disclosed in detail. This is the single most valuable line in any exchange's filings.
- **Legal proceedings.** Every material investigation, subpoena and lawsuit, itemized, quarterly.
- **Revenue by segment,** audited, so you can see whether the business is what management says it is.

The trade Coinbase made is now legible: it accepted higher costs, slower product launches, fewer listings, and no ability to offer the leverage and perpetual futures that drove offshore volume — in exchange for being the one venue a pension fund's compliance department could approve. For most of the last decade that looked like leaving money on the table. Then the money left the table anyway, along with the tables.

### The cost of being boring

It is easy, from 2026, to describe this as a brilliant long game. It did not look like one for most of its duration, and the strategy is only interesting if you understand how much it cost along the way.

Coinbase was founded in 2012, when the regulated-exchange thesis was close to absurd — the entire market was hobbyists and a Japanese venue that would shortly lose several hundred thousand bitcoin. Through the 2017 mania, its product was conspicuously worse than the competition: fewer assets, no leverage, higher fees, and outages under load. Through the 2020–2021 boom, offshore venues offering 100× leverage and instant listings ran circles around it on volume, and the industry consensus was that Coinbase had chosen a niche while everyone else took the market.

Then 2022 happened. Terra collapsed in May, taking a market maker's balance sheet with it — the story we tell in [Jump Crypto and the Terra entanglement](/blog/trading/crypto-players/jump-crypto-and-the-terra-entanglement). Celsius froze withdrawals in June. FTX failed in November, and a very large fraction of the industry discovered it had been an unsecured creditor of a company with no meaningful books.

Coinbase's customers had a bad year in the sense that their portfolios fell. They did not have a bad year in the sense of not being able to get their money.

That distinction is the entire return on a decade of accepting worse unit economics. It is also why the compliant model is so hard to copy: the asset being accumulated was not a licence or a technology, it was a *track record of surviving a specific kind of failure*, and there is no way to buy that in a hurry. A competitor can obtain the same licences in a year. It cannot obtain ten years of not having blown up.

The corollary is uncomfortable for the industry's self-image. The thing that made Coinbase safe was not decentralization, cryptography, or open-source software. It was ordinary financial regulation — audits, capital requirements, segregation rules, and the threat of prison for lying in a filing — applied to a custodial intermediary that, functionally, is a bank.

This is the sense in which **legitimacy is a product**. Coinbase does not merely comply; it *sells* compliance to counterparties who cannot buy it anywhere else. When an ETF issuer needs a custodian, when a public company wants to put bitcoin on its balance sheet, when a bank wants crypto exposure without a headline — the shortlist is short, and Coinbase is on it. That is a different kind of gravity from the volume gravity we describe in [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues) and the everything-everywhere gravity of [Binance](/blog/trading/crypto-players/binance-the-everything-exchange-and-its-gravity). It is narrower. It is also much harder to replicate, because it is made of years of not doing things.

## The revenue mix: where the money actually comes from

![The revenue stack: transaction revenue split between consumer and institutional, and subscription and services split across stablecoin, staking, custody and interest income](/imgs/blogs/coinbase-the-compliant-giant-3.webp)

Coinbase reports revenue in two big buckets, and the split is roughly two-thirds / one-third.

**Transaction revenue** is trading fees. In its FY2024 results (reported in the Q4 2024 shareholder letter, February 2025), Coinbase reported total revenue of approximately **\$6.6 billion**, up from approximately **\$3.1 billion** in FY2023 — a reminder that this is a violently cyclical business. Roughly two-thirds of the 2024 figure was transaction revenue.

**Subscription and services** is everything else: stablecoin revenue, blockchain rewards (staking), custodial fees, interest and finance fee income, and subscription products like Coinbase One. This bucket was built deliberately, after 2022, to answer the question "what does this company earn when nobody is trading?" The answer turned out to be: quite a lot, as long as interest rates are high — which is a different vulnerability, not an absence of one.

### The take-rate asymmetry

The most important structural fact about Coinbase's trading business is that its two customer types are economically unrelated.

![Worked example: the same volume earns radically different revenue from a retail user versus an institution](/imgs/blogs/coinbase-the-compliant-giant-4.webp)

#### Worked example 1: what a dollar of volume is worth

Coinbase does not publish a line called "take rate". You compute it by dividing transaction revenue by trading volume, both of which it does disclose, separately for consumer and institutional. Do that for any recent period and you find two numbers that are almost comically far apart. Let's walk it with round, illustrative rates that sit in the disclosed range.

**The retail side.** You open the Coinbase app and buy \$1,000 of bitcoin.

- Trade size: \$1,000
- Effective take rate: 1.50%
- Coinbase revenue: \$1,000 × 0.0150 = **\$15**

**The institutional side.** A hedge fund executes \$1,000,000 through Coinbase Prime, the institutional platform.

- Trade size: \$1,000,000
- Effective take rate: 0.03% (3 basis points)
- Coinbase revenue: \$1,000,000 × 0.0003 = **\$300**

Now the comparison that matters. How much institutional volume does it take to earn the same \$15 as one retail \$1,000 trade?

$$\text{institutional volume needed} = \frac{\$15}{0.0003} = \$50{,}000$$

**Fifty thousand dollars** of institutional flow earns what one thousand dollars of retail flow earns. A single retail dollar is worth roughly fifty institutional dollars.

Scale it up to a quarter:

| Segment | Volume | Take rate | Revenue |
| --- | --- | --- | --- |
| Consumer | \$10bn | 1.50% | \$150m |
| Institutional | \$100bn | 0.03% | \$30m |

The institutional business is ten times the volume and one fifth the revenue.

*The intuition: headline trading volume tells you almost nothing about an exchange's revenue unless you know the mix. An exchange bragging about institutional volume is bragging about a low-margin business.*

This asymmetry explains an enormous amount of behaviour you observe from the outside. It explains why Coinbase's app is designed the way it is, why the fee schedule is opaque to retail and razor-transparent to institutions, why revenue collapses so violently when retail loses interest even if institutional volume holds up, and why "Coinbase is expensive" is both true and beside the point — the retail customer is not buying execution quality, they are buying a US-regulated on-ramp with a phone number.

It also explains why competitors attack from the retail side. Zero-fee brokerages adding crypto, and offshore venues offering better prices, are aiming at exactly the slice of volume that carries all the margin. The long-run direction of retail take rates in every marketplace business ever studied is *down*.

### The rest of the stack

Within subscription and services, the components are worth separating because they respond to completely different forces:

| Line | What it actually is | What drives it |
| --- | --- | --- |
| Stablecoin revenue | Interest on USDC reserves, shared from Circle | Short-term interest rates × USDC balances |
| Blockchain rewards | Commission on staking customer assets | Amount staked × network staking yield |
| Custodial fees | Basis points on assets held for institutions and ETFs | Crypto prices × assets under custody |
| Interest & finance fee income | Interest on customer fiat balances, financing fees | Interest rates |
| Subscriptions | Coinbase One and similar | Retail engagement |

Notice how many of those rows say "interest rates". That is the punchline of the next section.

#### Worked example 6: why the earnings swing so violently

Revenue roughly doubled between FY2023 and FY2024 — from approximately \$3.1 billion to approximately \$6.6 billion. Understanding *why* the profit line moves far more than the revenue line is the key to reading any exchange's results.

The reason is **operating leverage**: most of an exchange's costs are fixed. Engineers, compliance staff, legal, security, offices and cloud infrastructure cost roughly the same whether volume triples or halves. Matching one extra trade costs almost nothing.

Work it with illustrative round numbers to see the mechanic clearly:

- **Year 1.** Revenue \$3,000m. Fixed costs \$2,500m. Variable costs 10% of revenue = \$300m.
  Profit = \$3,000m − \$2,500m − \$300m = **\$200m**.
- **Year 2.** Revenue doubles to \$6,000m. Fixed costs rise modestly to \$2,800m. Variable costs = \$600m.
  Profit = \$6,000m − \$2,800m − \$600m = **\$2,600m**.

Revenue went up 100%. Profit went up

$$\frac{2{,}600 - 200}{200} = 1{,}200\%$$

Thirteen times. And the mechanism runs identically in reverse: a 50% revenue fall in the following year would wipe the profit out entirely and push it negative, without a single thing going wrong operationally.

*The intuition: exchange earnings are a leveraged derivative on trading activity. Never annualize a good quarter, and never extrapolate a bad one — you are looking at the same cost base under two different weather conditions.*

This is also why the post-2022 push into subscription revenue was strategically necessary rather than cosmetic. Fixed costs need a revenue floor underneath them, and fee income from a cyclical retail audience does not provide one.

## The interest-rate business hiding inside a crypto company

![How USDC reserve income flows from Treasuries through Circle to Coinbase and out to users as rewards](/imgs/blogs/coinbase-the-compliant-giant-5.webp)

USDC is issued by Circle, not by Coinbase. But the two companies are joined at the hip in a way that took years to become visible.

The history: Coinbase and Circle founded the **Centre Consortium** in 2018 as the joint governance body for USDC. On **21 August 2023** they announced a restructuring — Centre was dissolved, Circle took full control of issuance, and Coinbase took an equity stake in Circle (the size of that stake was not disclosed at the time, and no cash changed hands; it was a stock-for-stock arrangement). Circle went public on the NYSE under the ticker **CRCL**, pricing its IPO at **\$31.00 per share on 5 June 2025**. Circle's registration statement, filed on **1 April 2025** ahead of that listing, is where the economics finally became legible to outsiders, because a company going public has to explain who it pays and why.

The mechanism is this. When you hold USDC, Circle holds roughly a dollar of short-dated US government debt against it. That reserve earns the Treasury bill yield. Circle keeps some and pays a large share away as "distribution and transaction costs" to the partners who put USDC in front of users. Coinbase is by far the largest such partner. The arrangement disclosed in Circle's S-1 gives Coinbase **100% of the residual reserve income on USDC held on Coinbase's own platform**, and **50% of the residual reserve income on USDC held everywhere else**.

Sit with that second clause for a moment. Coinbase earns half the interest on USDC that has nothing to do with Coinbase — USDC sitting in a wallet in Singapore, in a DeFi pool, on a competing exchange. It is paid for having been present at the creation.

The magnitudes are large. Per Circle's S-1 disclosures, of Circle's roughly **\$1.011 billion** of total distribution and transaction costs in FY2024, approximately **\$908 million** — around 90% — went to Coinbase. And the share of USDC actually sitting on Coinbase's platform grew from roughly **5% in 2022 to about 20% in 2024 and around 22% by Q1 2025**, which mechanically shifts more of the reserve income into the 100% bucket rather than the 50% one. Coinbase's own Q2 2025 shareholder letter put average USDC balances held in Coinbase products at **\$13.8 billion**, up 13% quarter over quarter; its Q4 2025 letter reported average USDC market capitalization rising about \$8.4 billion in the quarter to **\$76.2 billion**.

Read that twice, because it is the single most counterintuitive fact about Coinbase's business: **Coinbase earns interest on a product it does not issue, in proportion to how much of it sits on its platform, and the interest comes from US Treasuries.** It is a distribution deal on a money-market fund wearing a blockchain costume.

Coinbase then pays a portion back to users as "USDC Rewards" — an advertised yield on idle USDC balances, tiered by product and subscription status. Coinbase has advertised different rates across its surfaces; as one dated example, it introduced a 3.5% APY tier for Coinbase One subscribers effective **19 February 2026**, and has separately advertised a higher rate for onchain balances held in Coinbase Wallet. Check the current published rate rather than any figure quoted in an article, including this one — these move with the policy rate.

Structurally, that payout is a customer-acquisition cost funded out of the spread, exactly like a bank paying deposit interest out of its lending margin. Coinbase is running a deposit franchise.

### Worked example 2: how much a rate cut costs

![Reserve income on a fixed stablecoin balance scales linearly with the short-term interest rate](/imgs/blogs/coinbase-the-compliant-giant-6.webp)

Take a round **\$10 billion** balance of USDC on Coinbase's platform, earning the prevailing short-term rate. The \$10 billion is a **round illustrative figure chosen for legible arithmetic**, not a reported number — though it is the right order of magnitude: Coinbase's Q2 2025 shareholder letter put average USDC balances in Coinbase products at \$13.8 billion.

At a 5% short-term rate:

$$\$10{,}000{,}000{,}000 \times 0.05 = \$500{,}000{,}000 \text{ per year}$$

At 3%:

$$\$10{,}000{,}000{,}000 \times 0.03 = \$300{,}000{,}000 \text{ per year}$$

At 1%:

$$\$10{,}000{,}000{,}000 \times 0.01 = \$100{,}000{,}000 \text{ per year}$$

The relationship is a straight line through the origin. Every 100 basis points of policy rate is worth \$100 million a year on a \$10 billion balance. A central bank cutting from 5% to 3% removes 40% of that revenue line, and it does so without a single customer leaving, without a single token falling, and with essentially no offsetting reduction in costs.

Now the second-order version, which is the one that actually matters. Rewards are paid *out of* that income, so the net is what you care about:

- Reserve income at 5%: **\$500m**
- Rewards paid to users at 4.0%: \$10bn × 0.040 = **\$400m**
- Net to Coinbase: **\$100m**

Cut rates to 3% and suppose the advertised reward falls to 2.5%:

- Reserve income: **\$300m**
- Rewards paid: \$10bn × 0.025 = **\$250m**
- Net: **\$50m**

The gross income fell 40%; the *net* fell 50%. The spread business compresses faster than the gross, because you cannot cut the customer's rate as fast as the market cuts yours without losing the balances. This is the oldest problem in banking and it has been imported into crypto wholesale.

*The intuition: a large slice of Coinbase's most-praised "diversified, non-trading" revenue is a leveraged bet on the front end of the yield curve. It is genuinely uncorrelated with crypto prices — which is the point — but it is not uncorrelated with anything.*

### The regulatory overhang on the rewards half

This is the live risk to the business line, and it is worth stating precisely because it is widely garbled.

The **GENIUS Act** (Public Law 119-27) was signed into law on **18 July 2025**, establishing a federal framework for payment stablecoins. It prohibits payment stablecoin *issuers* from paying holders interest or yield — in cash, tokens, or other consideration — solely for holding, using or retaining the stablecoin.

Read literally, that constrains Circle, not Coinbase: it is Coinbase, not the issuer, that pays USDC Rewards. Whether that distinction survives is exactly the contested question. In an **notice of proposed rulemaking dated 19 September 2025**, the Office of the Comptroller of the Currency proposed extending the interest-and-yield prohibition beyond issuers to *affiliates and third parties* — which would reach the exchange-paid rewards model directly. Separately, legal commentators have argued that the existing Circle-to-Coinbase reserve-income sharing may already sit uncomfortably with the statute, on the theory that Coinbase is the "holder" of custodially-held USDC. That is contested legal analysis, not a regulatory finding, and Coinbase and Circle dispute the framing.

The banking lobby wants the gap closed; the crypto industry wants it open. The stakes are the customer-facing half of the arithmetic in Worked Example 2: if Coinbase cannot pay rewards, it keeps the whole spread but likely loses balances to platforms in jurisdictions that permit it. If it can, the model continues. As of this writing the rules were still moving — date-stamp anything you read on it.

## The listing machine

![The Coinbase asset listing pipeline from application through legal, security and market review to the listing decision](/imgs/blogs/coinbase-the-compliant-giant-7.webp)

Here is where Coinbase stops being a company and starts being an institution: it decides which assets get access to the largest regulated retail crypto audience in the United States.

The process is real and bureaucratic. Coinbase publishes a digital asset listing framework, and internally the decision sits with a committee — the **Digital Asset Listing Group** — which weighs the outputs of several independent reviews:

- **Legal and regulatory review.** The central question: does this asset look like a security under US law? This is not a formality. It is the reason Coinbase's US listings lagged offshore venues by months or years, and the reason certain large-cap tokens were simply never available to American users.
- **Technical and security review.** Does the code do what it says? Can Coinbase custody it safely? Are there admin keys, upgrade backdoors, mint functions?
- **Market review.** Is there real liquidity and real demand, or is the float controlled by four wallets? We cover how to check this yourself in [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table).
- **Compliance and sanctions screening.**

The output is a decision, and — since a 2020 process change — a *public roadmap post* announcing that an asset is under consideration, before trading opens.

That roadmap post is the interesting object.

### The "Coinbase effect" and its decay

For several years, "listed on Coinbase" functioned as a legitimacy stamp with a measurable price. Research firms documented an announcement effect: tokens added to the Coinbase roadmap tended to rise sharply in the hours and days after the post, before trading had even opened. In the 2019–2021 era the reported average effects were large — the phenomenon was widely enough known to be nicknamed the "Coinbase effect" and traded systematically.

I want to be careful here, because this is exactly the sort of claim that gets repeated with a spuriously precise number attached. The honest summary is: **the direction is well documented and the magnitude is not well pinned down by rigorous, replicated research.** Different studies use different windows, different samples, and different treatments of the announcement-versus-listing distinction, and they disagree by a lot. Treat any single "average listing pop is X%" figure you see — including ones from reputable research shops — as an estimate over one particular sample, not a constant of nature.

What *is* clear is the mechanism, and the mechanism explains the decay.

#### Worked example 3: anatomy of a listing trade (illustrative case)

This is a **labelled illustrative case**, not a specific token. The shape is representative of what traders described repeatedly through the 2020–2021 cycle.

Token XYZ trades at **\$2.00**. Coinbase posts it to the listing roadmap.

1. **Announcement.** Within 48 hours the price is **\$2.60**. That is a gain of

   $$\frac{2.60 - 2.00}{2.00} = 30.0\%$$

   Nothing has changed about the token. No new users can buy it on Coinbase yet. The move is entirely anticipation.

2. **Listing day.** Trading opens on Coinbase and the price prints **\$2.75** — up 37.5% from the pre-announcement level. This is where the retail flow finally arrives.

3. **Three weeks later.** The price is **\$2.10**, up 5% from where it started.

Now compute two traders' outcomes.

**Trader A** buys the announcement at \$2.20 and sells into listing-day liquidity at \$2.75:

$$\frac{2.75 - 2.20}{2.20} = +25.0\%$$

**Trader B** — the retail user who learned about the token *because* Coinbase listed it — buys on listing day at \$2.75 and holds:

$$\frac{2.10 - 2.75}{2.75} = -23.6\%$$

Same token, same three weeks, opposite outcomes. Trader B provided the exit liquidity for Trader A.

*The intuition: the listing does not create value; it creates a moment of concentrated attention and a new pool of buyers. Whoever is positioned before that moment sells to whoever arrives at it.*

Why did the effect decay? Four reasons, each of which you can reason about from first principles:

1. **It became known.** Any published, systematic, free money trade gets arbitraged. Once traders ran screens on roadmap posts, the move front-ran the announcement rather than following it.
2. **Coinbase listed far more assets.** Scarcity was the whole mechanism. A stamp given to a handful of assets a year is a signal; a stamp given to hundreds is a category.
3. **The competitive set changed.** By the mid-2020s, a serious token was already liquid on many venues before Coinbase looked at it. Coinbase became a late confirmation rather than a first access point.
4. **The regulatory chill.** During the Wells-notice and litigation period from 2023, Coinbase's US listing pace for anything with securities-law ambiguity slowed sharply. The assets it *did* list skewed toward the ones with the least to prove, which are also the ones with the least room to re-rate.

The fourth reason deserves emphasis because it is the clearest example of regulation reshaping price formation. When the gatekeeper's legal risk rises, the gate narrows, and the value of getting through it changes character — it stops being "access to buyers" and becomes "a public assertion that our lawyers think you're not a security." That is a different signal, and the market prices it differently. For how listings interact with the rest of a token's lifecycle, see [how VCs move price: listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) and [the lifecycle of a token: seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock).

### The listing desk is also an insider-trading surface

If a roadmap post moves price, then knowing about the roadmap post before it publishes is worth money. In **July 2022** the US Department of Justice charged a former Coinbase product manager, **Ishan Wahi**, along with two others, in what prosecutors described as the first cryptocurrency insider-trading case. Wahi had access to listing information; the allegation was that he tipped it. He pleaded guilty and was **sentenced in May 2023 to two years in prison**. The SEC brought a parallel civil case.

Two things are worth taking from this. First, it is direct evidence that the listing signal was economically real — nobody goes to prison over a signal worth nothing. Second, it is another instance of the disclosure asymmetry: this is knowable because a US public company operating under US law generated a US criminal case with a public docket. Equivalent conduct at a venue outside any enforcement perimeter produces no case, no docket, and no knowledge.

## Coinbase Ventures: the investor who is also the gatekeeper

![The conflict structure: Coinbase Ventures holds a token, the listing group decides, the listing moves the price, the stake marks up](/imgs/blogs/coinbase-the-compliant-giant-8.webp)

Coinbase Ventures, launched in 2018, is Coinbase's investment arm. It has backed hundreds of companies and protocols — one of the most prolific investors in the sector by deal count, writing small early cheques across an extremely wide surface.

The structural problem writes itself. Coinbase Ventures takes a position in a token or in the company that issues it. Later, that token seeks a Coinbase listing. The Digital Asset Listing Group decides. If it lists, the token typically gets attention, liquidity and — historically — a price move. The Ventures stake marks up.

Every element of that chain is ordinary on its own. Together they describe a firm that can hold an asset, decide whether to grant it access to the largest US retail audience, and book the gain that access creates.

Coinbase's stated answer is disclosure plus internal separation: Ventures is described as operating independently of the listing process, listing decisions are said to be made without reference to Ventures' holdings, information walls are maintained, and the relevant relationships are disclosed. As a public company, related-party and investment disclosures are also compelled in its filings in a way they are not for private venture arms.

It is worth being precise about what that does and does not achieve. Information walls address the *narrow* conflict — a specific person trading on specific knowledge. They do not address the *structural* one: an institution whose balance sheet benefits from listing decisions its employees make, over years, across hundreds of positions, with no single decision ever being provably tainted. There is no allegation required here, and I am making none. The point is that this is a conflict managed by policy rather than eliminated by structure, and a reader should know the difference. The same logic applies across the industry — the venture arms of exchanges, the market makers who are also investors, the funds who are also advisors. We map the whole web in [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) and [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto).

What makes Coinbase's version *better* than the industry norm is not that the conflict is smaller. It is that you can see it. An offshore exchange with a venture arm and a listing desk has exactly the same conflict, plus no filings, plus no jurisdiction.

## Base: when the exchange becomes a landlord

In **August 2023** Coinbase launched **Base**, its own Layer 2 blockchain, built on the OP Stack — the open-source codebase developed by the Optimism ecosystem — and part of the "Superchain" of related chains. On **24 August 2023** Base and Optimism announced a shared governance and revenue-sharing framework. Under the terms Optimism has published for OP Stack chains, each chain pays the Optimism Collective the **greater of 2.5% of chain (sequencer) revenue or 15% of onchain profit** — profit meaning fee revenue minus the L1 gas and data-availability costs. Base pays this on the same terms as every other Superchain member.

Strategically this is the most underrated move in the company's history, and it is worth being clear about what changed. An exchange is a *venue*: it rents access to a market. A chain is *infrastructure*: it rents access to computation and settlement. By launching Base, Coinbase stopped being only a toll booth on trades and became a landlord of blockspace — earning fees from activity that has nothing to do with anyone trading on Coinbase at all.

Two features of Base are distinctive.

**It has no network token — and the story there has moved.** For its first two years Base's position was flatly that there would be no token. That changed: at BaseCamp in **September 2025**, Base creator Jesse Pollak said publicly that the team was "beginning to explore" a network token, adding that they were "in the early phases of exploration, and don't have any specifics to share around timing, design, or governance." It was the first acknowledgment that the no-token stance was not permanent.

As of **27 July 2026, no Base network token has launched.** Be careful reading headlines here, because there is a confusable adjacent fact: Base activated a token *standard* called B20 on **8 July 2026**, aimed at stablecoins and real-world assets on the chain. That is a technical specification for assets issued *on* Base, not a native chain token, and conflating the two is an easy mistake.

The no-token position was unusual to the point of being strange, and worth understanding on its merits: essentially every other L2 launched a token, used it to bootstrap incentives, and captured value through it. Base instead captured value *directly, in fees, in dollars, on a public company's income statement*. The reason is the same reason as everything else in this article — a US-listed company issuing a token to the public has a securities-law problem that a Cayman foundation does not. Compliance removed the standard playbook and forced a business model that is, arguably, cleaner: revenue you can audit instead of a token whose price is the marketing budget.

Which is exactly why the exploration announcement matters. If Base does eventually issue a token, it will be the clearest test yet of whether the post-2025 US regulatory environment really has changed — because this is precisely the transaction the compliant model spent a decade refusing to do. This is a fast-moving fact; check the current position rather than trusting this paragraph.

**Its costs collapsed.** An L2's main cost is posting data back to Ethereum. Before March 2024, rollups posted this data as ordinary transaction "calldata" at a cost on the order of **\$1,000 per megabyte**. On **13 March 2024** Ethereum's Dencun upgrade introduced "blobs" (EIP-4844) — a cheap, dedicated, temporary data channel built for exactly this purpose. Reported L2 transaction fees fell by roughly **100–200×**, and because blob supply has generally exceeded demand, the blob base fee has spent much of the period since sitting at or near its one-wei floor. In other words, an L2's single largest input cost went to approximately zero, by a decision made in a protocol community that Coinbase does not control. Every L2's margin structure changed overnight.

### Worked example 4: sequencer margin

![Waterfall: user fees in, Ethereum data costs and the Optimism revenue share out, net margin to Coinbase](/imgs/blogs/coinbase-the-compliant-giant-9.webp)

These are **illustrative volumes** chosen to make the arithmetic legible; treat the structure as the lesson, not the level.

Suppose Base processes **5,000,000 transactions** in a day at an average user fee of **\$0.01**.

**Step 1 — gross sequencer revenue:**

$$5{,}000{,}000 \times \$0.01 = \$50{,}000 \text{ per day}$$

**Step 2 — subtract the Ethereum data cost.** Post-blobs, suppose posting the day's data costs **\$2,000**:

$$\$50{,}000 - \$2{,}000 = \$48{,}000 \text{ gross profit}$$

Note what just happened: the cost of goods sold is **4% of revenue**. Before blobs, this line could plausibly have consumed most of the gross. A protocol upgrade on someone else's chain turned a marginal business into a very good one.

**Step 3 — subtract the Optimism Collective share,** defined as the greater of two formulas:

- 2.5% of revenue: \$50,000 × 0.025 = **\$1,250**
- 15% of profit: \$48,000 × 0.15 = **\$7,200**

The greater is **\$7,200**.

**Step 4 — net:**

$$\$48{,}000 - \$7{,}200 = \$40{,}800 \text{ per day}$$

Annualized: \$40,800 × 365 ≈ **\$14.9 million per year**, at an operating margin north of 80% on the marginal transaction.

*The intuition: a sequencer is a software business with near-zero marginal cost, and after blobs its main input became almost free. The revenue is small relative to trading fees, but it does not care whether anyone is trading.*

**A reality check on those illustrative numbers.** They are close to the right order of magnitude, which is why I chose them, but the real figures move a lot. Per the public chain dashboard growthepie, Base was running about **5.2 million transactions per day** and roughly **\$35,200 per day** of chain revenue as of **26 July 2026** — with both figures down sharply week over week (transactions −26%, revenue −39%), which tells you how volatile this line is. Base's total value locked was about **\$4.6 billion** on **27 July 2026** per DefiLlama, and its on-chain stablecoin supply about \$4.67 billion.

One important caveat about all Base figures: **Coinbase does not break Base out as a separate reportable segment** in its financial statements. Sequencer income is folded into broader revenue lines. Every Base-specific revenue number you will read — including analyst estimates of net sequencer profit — is derived from public chain data or estimated, not lifted from a filing. That is a notable gap in the "you can audit this company" story, and worth holding in mind given how much strategic weight Base carries.

That aside, the strategic point stands. Base revenue is uncorrelated with Coinbase's trading revenue in a way that stablecoin revenue is not uncorrelated with interest rates. It is the closest thing in the portfolio to a genuinely independent line.

The uncomfortable part is centralization. Base's sequencer has been operated by Coinbase. That means one company orders the transactions on a chain that settles billions in value — it can, in principle, censor or reorder — and decentralizing the sequencer has been a stated roadmap item rather than a shipped fact. An exchange that got rich on the premise that intermediaries should be trustworthy-because-audited has built a chain whose neutrality currently rests on the same premise. That is coherent. It is not what most people mean by decentralization.

## The derivatives gap, and buying the way out of it

There is one category where the compliant model straightforwardly lost, and it happens to be the largest category in crypto by volume: **derivatives**.

A derivative is a contract whose value depends on something else — here, a bet on a token's future price rather than ownership of the token itself. Crypto's dominant instrument is the *perpetual future*, a leveraged contract with no expiry date, invented offshore precisely because it fit no existing regulatory box. For most of the last decade, perpetuals have accounted for the large majority of all crypto trading volume, and essentially none of it was available to US retail users on a US-regulated venue.

This is the clean counterfactual for the whole "legitimacy as a product" thesis. Compliance is not free, and here is the bill: Coinbase spent years locked out of the biggest product in its own industry, watching volume it could measure but not serve route to venues it could not follow. The leverage that made offshore exchanges dangerous was the same leverage that made them liquid.

The response was to buy in rather than build in. In **May 2025** Coinbase announced an agreement to acquire **Deribit**, the dominant crypto options venue, in a cash-and-stock deal reported at approximately **\$2.9 billion** — the largest acquisition in the company's history at the time. Buying an established derivatives venue solves in one transaction what regulatory-perimeter constraints had made unsolvable organically: it brings a mature options book, its institutional client base, and its market-maker relationships inside a US-listed entity.

Two observations follow. First, this is what having a public stock is *for*. A listed company can pay for acquisitions in its own shares — a currency an offshore competitor simply does not have — so the compliance premium that constrained the product line also funded the fix. Second, derivatives revenue behaves differently from spot: options volume is driven by hedging and volatility demand, which persists in flat markets where spot trading dries up. Whether it becomes a third genuinely independent revenue leg alongside custody and Base is the open question in the business.

The general lesson generalizes past Coinbase. When a regulated firm cannot build a product, it usually ends up buying one, and the acquisition price is a reasonable estimate of what the regulatory constraint cost.

## Custody: the plumbing under the ETFs

![Grid of the eleven US spot bitcoin ETFs approved in January 2024, with eight naming Coinbase Custody](/imgs/blogs/coinbase-the-compliant-giant-10.webp)

On **10 January 2024** the SEC approved the first US spot bitcoin exchange-traded products, and they began trading on **11 January 2024**. Eleven funds launched. **Eight of the eleven named Coinbase Custody Trust Company as their custodian**, including the largest.

Coinbase Custody Trust Company is a New York limited purpose trust company chartered by the New York Department of Financial Services. That charter is the reason it won this business: an ETF issuer needs a custodian that a US regulator has examined, and the population of such entities that can hold bitcoin is small.

Consider what this arrangement means. The entire promise of a spot bitcoin ETF is that ordinary investors get bitcoin exposure through a brokerage account without touching a wallet. Investors chose between issuers on fee and brand — BlackRock versus Fidelity versus Grayscale — believing they were diversifying. Underneath, most of them were pointing at the same vault.

#### Worked example 5: sizing the concentration

Take the launch cohort. If 8 of 11 funds use one custodian, then by simple count:

$$\frac{8}{11} = 72.7\% \text{ of the funds}$$

But funds are not equal in size, and the largest funds were in the eight. Weighted by assets rather than count, the share was higher than the headline fraction — which is the opposite of how concentration usually gets understated.

Now the revenue side. Custody is priced in basis points on assets held. Coinbase has not, to my knowledge, publicly disclosed the specific rate it charges ETF issuers, so the following is **explicitly illustrative arithmetic**, not a reported figure. If a custodian charged **10 basis points** on **\$50 billion** of custodied ETF bitcoin:

$$\$50{,}000{,}000{,}000 \times 0.0010 = \$50{,}000{,}000 \text{ per year}$$

And here is why this line is so different from every other line in the business: it requires no trading, no volatility, and no user engagement. It scales with the *price of bitcoin* times the *quantity held*. If bitcoin doubles and nobody trades a single share, this revenue doubles.

*The intuition: custody converts Coinbase's revenue from a bet on activity into a bet on asset prices. It is the most passive, highest-margin, and most concentrated business the company has.*

The risk is the mirror image. A single operational failure, security incident, or legal seizure at one custodian would simultaneously affect most of the US spot bitcoin ETF complex. The ETF prospectuses disclose custodian risk as a risk factor precisely because it is one, and commentators — including some at competing issuers — have made the concentration argument publicly since launch. Issuers have since begun adding secondary custodians, which is the market's own answer to the problem; I would encourage checking the current custodian arrangements in any specific fund's latest prospectus rather than relying on the launch-day picture, because this is actively changing.

There is a subtler version of the risk too. Coinbase is, at once, custodian to the ETFs, a large holder of crypto on its own balance sheet, the operator of a trading venue where ETF authorized participants transact, and a public company whose stock is itself a crypto proxy. In traditional markets those functions are frequently required to sit in separate legal entities with separate regulators for exactly this reason. We look at the whole ETF-to-TradFi bridge in [bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge).

## The SEC years

![Timeline of the SEC action against Coinbase from Wells notice to dismissal with prejudice](/imgs/blogs/coinbase-the-compliant-giant-11.webp)

The central irony of Coinbase's story is that the company which spent a decade trying to be legal was sued by its own regulator for being illegal, and then had the case dropped.

The sequence, with dates:

- **4 January 2023** — the New York Department of Financial Services announced a consent order with Coinbase over compliance and anti-money-laundering program failures, with a **\$50 million penalty** and a commitment to invest a further **\$50 million** in its compliance program.
- **22 March 2023** — Coinbase disclosed it had received a **Wells notice** from the SEC.
- **6 June 2023** — the SEC filed suit in the Southern District of New York (*SEC v. Coinbase, Inc. and Coinbase Global, Inc.*), alleging Coinbase operated as an unregistered securities exchange, broker and clearing agency, and separately that its staking-as-a-service program was an unregistered securities offering. The complaint identified a list of tokens the SEC contended were crypto asset securities. The case was assigned to Judge Katherine Polk Failla.
- **27 March 2024** — the court largely denied Coinbase's motion to dismiss, allowing the core claims to proceed. One claim, relating to the self-custody Wallet product, was dismissed.
- **7 January 2025** — the court certified an interlocutory appeal, sending the underlying legal question toward the Second Circuit before trial.
- **21 February 2025** — Coinbase announced the SEC had agreed in principle to dismiss.
- **27 February 2025** — the parties filed a joint stipulation and the case was **dismissed with prejudice**, with no penalty and no changes required to Coinbase's business.

"With prejudice" means it cannot be refiled. "No ruling on the merits" means the question everyone was waiting for — *which* tokens are securities, under *what* test — was never answered by a court. The case that was supposed to produce doctrine produced a dismissal.

Three things follow, and it is worth separating them carefully because they are routinely conflated.

**First: this was a change in enforcement posture, not a change in law.** The dismissal came amid a broad shift in US crypto policy in early 2025, including the formation of an SEC crypto task force and the dropping or pausing of several other crypto enforcement actions. A different administration with a different view can adopt a different posture. Statutory clarity — an actual market-structure law defining when a token is a security and which regulator supervises it — was still working through Congress rather than settled as of this writing. Anything you read about the state of US crypto legislation should be checked against a current date.

**Second: the litigation period had real, measurable effects.** For roughly two years, a US-listed company operated under an existential claim that its core business was illegal. That is not a background condition; it is a constraint on every decision. It slowed listings, particularly for anything with securities ambiguity. It affected which products could launch in the US. It pushed activity offshore, to exactly the venues with fewer disclosures. Whatever one thinks of the merits, "regulation by enforcement" had the observable effect of making the most transparent venue the most constrained one.

**Third: Coinbase's response was itself a strategy.** It fought publicly and loudly, funded advocacy, and made the argument that it had repeatedly asked for a registration path and been refused. A private company would have settled quietly. A public company with a legitimacy-based business model could not — settling would have damaged the exact asset it sells. The incentive to fight was structural.

## Common misconceptions

**"Coinbase is just an expensive Binance."** They are different businesses that happen to share a category. Binance's model is scale, product breadth, derivatives and global reach; Coinbase's is regulatory access sold at a premium to counterparties who require it. Compare the revenue mix and it is obvious — a large slice of Coinbase's income has nothing to do with trading at all. The head-to-head is in [centralized crypto exchanges: Binance and Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase).

**"Being public means Coinbase can't have conflicts of interest."** Being public means conflicts must be *disclosed*, not that they cannot exist. Coinbase Ventures investing in assets Coinbase may list is a live structural conflict managed by internal policy. Disclosure is a genuine improvement over opacity. It is not the same as separation.

**"A Coinbase listing means an asset is safe or approved."** It means Coinbase's internal review concluded the asset met its criteria, most importantly that its lawyers did not think it was a security. That is a legal-risk judgment about Coinbase, not a quality judgment about the token, and certainly not an endorsement of its price. Assets have been listed and subsequently collapsed.

**"Coinbase's stablecoin revenue proves it isn't dependent on crypto prices."** It proves it isn't dependent on crypto *volatility*. It is instead dependent on short-term interest rates, which is a real diversification but not an absence of risk. As Worked Example 2 shows, the net spread compresses faster than the gross when rates fall.

**"If you hold crypto on Coinbase, you own that crypto."** You own a claim against Coinbase, recorded in Coinbase's ledger. Coinbase's own filings disclose the bankruptcy risk explicitly. That disclosure is a feature — it exists because a public company must make it — but the underlying fact is true of every custodial exchange, disclosed or not.

**"The SEC case being dismissed means US courts decided tokens aren't securities."** No court decided anything. The case ended with prejudice by stipulation. The substantive question remains open, and the earlier motion-to-dismiss ruling actually went largely *against* Coinbase before the posture changed.

## How it shows up in real markets

**The 2022 bankruptcy-disclosure scare.** A required new risk factor about how custodial assets would be treated in a hypothetical bankruptcy caused a sharp negative reaction, and the company had to publicly clarify. The episode is a clean illustration of the trade Coinbase made: mandatory disclosure creates short-term pain and long-term trust, and it is the single clearest reason its customers did not experience what FTX customers experienced.

**January 2024, the ETF launch.** Eleven funds launched, eight pointed at one custodian, and a business line that barely existed a few years earlier became structural infrastructure for US retail bitcoin exposure. Watch what this did to Coinbase's revenue *character*: from cyclical fee income to price-linked custodial income.

**The Wells-notice era listings chill, 2023–2024.** With the SEC alleging its listings were unregistered securities transactions, Coinbase's incentive to list anything ambiguous went to roughly zero. The observable market consequence was that new assets found liquidity offshore first, and the "Coinbase listing" signal weakened — partly because the gate narrowed and partly because it stopped being the first gate.

**February 2025, the dismissal.** The removal of a two-year existential legal claim, with no penalty, was a genuine regime change for the company's risk profile. It is also a case study in how much of crypto's "regulatory clarity" has come from enforcement discretion rather than from law.

**The Circle listing, June 2025.** Circle going public forced disclosure of the USDC distribution economics, which is how outsiders learned how much of Coinbase's stablecoin income depends on an arrangement with a third party. Two public companies in a revenue-sharing relationship produce far more information than either would alone. This is the compounding effect of the compliant model: transparency in one place creates transparency in its counterparties.

**Base after the Dencun upgrade, March 2024.** An improvement to Ethereum's data availability — a decision made by a protocol community Coinbase does not control — materially improved the margin of a Coinbase business line overnight. It is a reminder that even the most compliant, most vertically integrated crypto company sits on infrastructure it does not own.

## Using COIN filings as a crypto-market data source

![What each COIN disclosure tells you about the wider crypto market](/imgs/blogs/coinbase-the-compliant-giant-12.webp)

This is the practical payoff, and the reason this post exists in a series about *players*. Coinbase's filings are the only audited, quarterly, legally enforced window into crypto market activity that anyone publishes. You do not have to care about the stock to use them.

Here is what to read and what it proxies for:

| Read this | It tells you |
| --- | --- |
| **Consumer transaction revenue and volume** | Retail risk appetite, denominated in dollars, quarter over quarter. This is the cleanest public retail-participation series in crypto. |
| **The implied consumer take rate** (revenue ÷ volume) | Competitive pressure on retail fees. A falling take rate means someone is undercutting. |
| **Institutional volume** | Whether professional flow is present even when retail is asleep — the two decouple often. |
| **Stablecoin revenue** | The size of the rate-linked income pool across the industry, and by extension how much of "crypto profitability" is really Treasury yield. |
| **Assets on platform / assets under custody** | Custody concentration, and a decent proxy for institutional commitment (assets are stickier than volume). |
| **Blockchain rewards revenue** | Staking economics at institutional scale — hard to observe anywhere else. |
| **The legal proceedings section** | The actual regulatory temperature, not the Twitter version. Every material investigation, itemized. |
| **Risk factors** | What the company's own lawyers think could kill it, updated quarterly and written under liability. |

A few practical notes on reading them. Compare quarters, not years, because crypto cycles are faster than annual reporting. Always compute the take rate yourself rather than trusting a summary. Read the *changes* in the risk factors between filings — a newly added risk factor is a signal, and the diff is where the information is. And remember the sample bias: Coinbase's customers skew US, retail-heavy and compliance-constrained, so its numbers describe *one* segment of the market well and the offshore, high-leverage segment not at all.

## When this matters to you

If you hold crypto anywhere, the Coinbase story is a working demonstration of what disclosure is worth. Not because Coinbase is virtuous — it is a company optimizing for profit like any other — but because the obligation to file changes what is knowable, and what is knowable changes what can go wrong quietly. Every crisis in this industry, from Mt. Gox to Celsius to FTX, was a crisis of information asymmetry before it was a crisis of solvency.

If you are trying to understand who moves crypto prices — the question this whole series is about — Coinbase is the player whose incentives you can actually verify instead of infer. Read one 10-K carefully and you will have a better map of the industry's economics than a year of following analysts. Then go back to the opaque players in [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) and notice how much of what you "know" about them rests on their own say-so.

And if you take one analytical habit from this piece, take this one: when a firm sits in several seats at once, do not ask whether it behaves badly. Ask what the structure would let it do, what disclosure would reveal if it did, and whether anyone is required to look. For Coinbase, the answers are unusually specific — which is exactly the point.

## Sources & further reading

Primary sources, which are where every figure in this post should be checked against a current date:

- **Coinbase Global investor relations** — quarterly shareholder letters, 10-K and 10-Q filings: [investor.coinbase.com](https://investor.coinbase.com/). The shareholder letters are the fastest route to the revenue split; the 10-K legal proceedings and risk factors sections are the most information-dense pages in crypto.
- **SEC EDGAR** — Coinbase Global, Inc. filings: [sec.gov/edgar](https://www.sec.gov/edgar/searchedgar/companysearch). Search for CIK 0001679788.
- **SEC v. Coinbase, Inc. and Coinbase Global, Inc.**, S.D.N.Y., complaint filed 6 June 2023; case dismissed with prejudice 27 February 2025. SEC litigation materials: [sec.gov/litigation](https://www.sec.gov/litigation/litreleases). Docket materials are available via [CourtListener](https://www.courtlistener.com/).
- **New York Department of Financial Services** — consent order with Coinbase, Inc., announced 4 January 2023: [dfs.ny.gov](https://www.dfs.ny.gov/).
- **US Department of Justice, S.D.N.Y.** — press releases on *United States v. Ishan Wahi* (charged July 2022; sentenced May 2023): [justice.gov/usao-sdny](https://www.justice.gov/usao-sdny).
- **Circle Internet Group (CRCL)** — the S-1 registration statement filed 1 April 2025 and subsequent filings, which disclose the USDC reserve-income distribution arrangements with Coinbase (the 100% / 50% residual split, and the FY2024 distribution-cost figures): [investor.circle.com](https://investor.circle.com/). IPO pricing announcement, 5 June 2025: [circle.com/pressroom](https://www.circle.com/pressroom/circle-announces-pricing-of-upsized-initial-public-offering).
- **GENIUS Act**, Public Law 119-27, signed 18 July 2025 — the statutory text of the payment-stablecoin framework and the interest/yield prohibition: [congress.gov](https://www.congress.gov/119/plaws/publ27/PLAW-119publ27.pdf).
- **Office of the Comptroller of the Currency** — GENIUS Act implementation notice of proposed rulemaking, 19 September 2025, proposing to extend the interest/yield prohibition to affiliates and third parties: [federalregister.gov](https://www.federalregister.gov/documents/2025/09/19/2025-18226/genius-act-implementation).
- **Coinbase asset listing information** — the published listing framework and current listed-asset roadmap: [coinbase.com/listings](https://www.coinbase.com/listings).
- **Base documentation** — chain architecture, OP Stack and sequencer design: [docs.base.org](https://docs.base.org/). The Optimism Collective revenue-share formula (greater of 2.5% of revenue or 15% of profit) is set out by Optimism: [optimism.io/blog](https://www.optimism.io/blog/how-(and-why)-the-superchain-drives-fees-to-the-optimism-collective).
- **L2BEAT**, **growthepie** and **DefiLlama** — independent L2 activity, cost, revenue and TVL data for Base and its peers: [l2beat.com](https://l2beat.com/) · [growthepie.com/chains/base](https://www.growthepie.com/chains/base) · [defillama.com/chain/Base](https://defillama.com/chain/Base). Use these rather than any quoted Base revenue figure, since Coinbase does not report Base as a segment.
- **Ethereum EIP-4844 (proto-danksharding)**, activated with the Dencun upgrade on 13 March 2024: [eips.ethereum.org/EIPS/eip-4844](https://eips.ethereum.org/EIPS/eip-4844).
- **ETF prospectuses** — for current custodian arrangements, read the latest prospectus of the specific fund (e.g. iShares Bitcoin Trust, Fidelity Wise Origin Bitcoin Fund, Grayscale Bitcoin Trust) on EDGAR rather than relying on launch-day reporting.

Secondary reporting used for context and cross-checking — Bloomberg, Reuters, the Financial Times, CoinDesk and The Block all covered the litigation timeline, the ETF launch and the Circle listing contemporaneously, and are the right places to verify a specific date or figure.

Related posts in this series: [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) · [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues) · [Binance: the everything exchange and its gravity](/blog/trading/crypto-players/binance-the-everything-exchange-and-its-gravity) · [how VCs move price: listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) · [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) · [the lifecycle of a token: seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) · [centralized crypto exchanges: Binance and Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) · [bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge) · [stablecoins, Tether, Circle and the shadow dollar](/blog/trading/crypto/stablecoins-tether-circle-shadow-dollar).
