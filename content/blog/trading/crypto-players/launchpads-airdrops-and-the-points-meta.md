---
title: "Launchpads, Airdrops and the Points Meta: How Token Supply Actually Reaches You"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "Every way a new crypto token gets distributed, from the 2017 ICO to today's off-chain points programs, and the honest answer to who captures the value and who provides the exit liquidity."
tags: ["crypto", "airdrops", "points", "launchpad", "ico", "ieo", "tokenomics", "fdv", "sybil", "token-distribution", "crypto-players"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 43
---

> [!important]
> **TL;DR** — Token distribution has evolved through five generations, and every step moved the cash payment later and the risk further onto the user, while the issuer's obligation shrank toward zero.
>
> - **ICO (2017)** you wired cash for a whitepaper. **IEO/launchpad (2019)** the exchange gated it and made you hold its own token. **IDO (2020)** you bought on a public curve. **Retroactive airdrop (from Uniswap's UNI, 16 September 2020)** you paid nothing and got tokens for past usage. **Points (from late 2023)** you deposit real capital for an off-chain scoreboard that promises nothing.
> - The points meta is the issuer's best deal ever invented: it buys total value locked, user counts and narrative *before* committing to any tokenomics, and it can still decide the conversion rate, the vesting, the sybil rules and whether to launch at all.
> - The structural asymmetry is timing. **You commit first and irreversibly; the issuer picks the snapshot, the rate, the vesting and the blocked jurisdictions after watching what you did.**
> - Low float plus high fully diluted valuation is the arithmetic that makes a launch look bigger than the money in it. Tokens launched in 2024 averaged a market-cap-to-FDV ratio near 12 percent (Binance Research, May 2024) — meaning roughly 88 percent of the headline valuation was supply that had never had to find a buyer.
> - Farming is rational mainly when the capital had nowhere better to be. Once you price the opportunity cost of locked capital honestly, the expected value of the median farm goes negative. Distrust any "average airdrop return" figure: the outcome distribution is so skewed that the mean describes nobody's actual experience.

Somewhere right now, a person is bridging \$8,000 of stablecoins to a chain that has no token, to earn "points" that have no defined value, under rules that have not been written, from a team that has promised nothing. They are not being stupid. Two years ago the same behaviour turned into a five-figure payday for a lot of people, and the memory of that is doing all the work.

This post is about the machine that produces those moments. Not the hype cycle around it — the actual plumbing. How does new token supply get from the people who created it into the hands of the public? There have been five distinct answers to that question since 2017, each one a response to the way the previous one broke, and each one shifting a little more of the cost and risk from the issuer onto you.

![The five generations of token distribution on a timeline, from ICO in 2017 to the points meta in late 2023](/imgs/blogs/launchpads-airdrops-and-the-points-meta-1.webp)

The diagram above is the mental model for the whole post. Read it left to right and watch two things move in opposite directions: the moment you have to hand over money moves *later* (from "before there is a product" all the way to "never"), while the amount of your own capital and behaviour committed before you know the terms moves *up*. That is not an accident and it is not a conspiracy. It is what happens when a market repeatedly discovers that the previous method was either illegal, unfair, or too expensive for the issuer — and each fix happens to be cheaper for the issuer than the thing it replaced.

By the end you should be able to look at any token launch and answer four questions in order: who is paying cash, who is receiving tokens, what is the receiver actually risking, and what did the issuer get for free.

## The foundations: what it means to "distribute" a token

Before any of the mechanisms make sense, you need six words defined properly. If you already know them, skim; if you don't, nothing later will land.

**A token is a ledger entry with a supply schedule.** When a team "creates" a token, they deploy a smart contract — a program on a blockchain — that says how many units exist and who is allowed to move them. That is it. There is no company, no share register, no legal claim on anything, unless a separate legal document creates one. This is the single biggest difference from equity, and it is worth reading [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) if you want the full version.

**Total supply is every unit that will ever exist. Circulating supply is the subset that can be sold right now.** The gap between them is tokens that exist in the contract but sit locked — held by the team, by investors, by a foundation treasury, or by an unclaimed airdrop contract. The circulating supply, expressed as a fraction of total, is called the **float**. A "low float" launch is one where only a small slice of the supply can trade on day one.

**TGE means token generation event** — the moment the token first exists and, usually, first trades. Everything before TGE is private: private rounds, private terms, private valuations. Everything after is public and visible on-chain.

**Market cap is circulating supply times price. Fully diluted valuation (FDV) is total supply times price.** These two numbers can differ by a factor of ten, and which one gets quoted in the headline is a marketing decision.

**Vesting is a lock with a schedule; a cliff is the day a big tranche of that lock opens at once.** A typical insider allocation has a one-year cliff and then linear monthly unlocks over two or three years. The whole schedule is usually public before you buy. [The lifecycle of a token from seed round to unlock cliff](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) walks the full pipeline.

**A snapshot is the moment the issuer freezes the ledger and decides who was doing what.** Airdrop eligibility is computed from a snapshot. The critical property — and this will come up again and again — is that the snapshot date is chosen by the issuer, and it can be chosen *retroactively*, after the behaviour has already happened.

**A sybil is one person operating many wallets to look like many people.** The name comes from a 2002 computer-science paper about identity forgery in peer-to-peer networks. Because a blockchain address costs nothing to create, any rule of the form "each address gets X" is an invitation to create a lot of addresses.

**Total value locked (TVL) is the dollar value of assets deposited into a protocol's contracts.** It is the headline metric of decentralised finance, it is the thing points programs are designed to buy, and — as we will see — it is much less informative than it looks.

One more piece of context. Two categories of player recur throughout: the venture funds who bought supply years before you could, and the exchanges and market makers who control where and how it trades. Their incentives run underneath every mechanism in this post. [Cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) is the map; [how VCs move price through listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) is the specific mechanism on the fund side.

## Generation one: the ICO, or wiring cash to a whitepaper

The initial coin offering was the purest version of the trade: you send cryptocurrency to an address, and at some point in the future you receive tokens. There was no product, no revenue, and frequently no code. There was a PDF.

The mechanics were trivial, which was the point. A team published an Ethereum address and a document, people sent ether, and a contract distributed tokens pro rata. No exchange, no gatekeeper, no jurisdiction check. The whole apparatus that normally stands between a company and public money — underwriters, prospectuses, regulators, listing standards — was simply absent.

The scale got large fast. Block.one's EOS sale ran for a full year, from June 2017 to June 2018, and is widely reported to have taken in something on the order of four billion dollars; the SEC settled with Block.one on 30 September 2019 over an unregistered securities offering, with a \$24 million civil penalty that pointedly did not require the tokens to be returned or registered. Filecoin, Tezos and Telegram each raised sums that would have been respectable IPOs.

Then the legal system arrived. The SEC's investigative report on The DAO, published 25 July 2017 (Release No. 34-81207), applied the *Howey* test — from *SEC v. W.J. Howey Co.*, 328 U.S. 293 (1946) — to a token sale and concluded that yes, this can be an investment contract: an investment of money, in a common enterprise, with an expectation of profit, derived from the efforts of others. Two enforcement actions made the point concrete. In *SEC v. Telegram*, Judge Castel of the Southern District of New York granted a preliminary injunction on 24 March 2020 blocking the TON distribution; Telegram subsequently returned roughly \$1.2 billion to purchasers and paid an \$18.5 million penalty. In *SEC v. Kik Interactive*, the same court granted summary judgment to the SEC on 30 September 2020, and Kik settled for a \$5 million penalty.

Note what the enforcement actually targeted: not fraud, in these two cases, but *selling unregistered securities to the American public*. That distinction shaped everything that came next. Every subsequent generation of distribution is, in part, an attempt to reach users without executing a sale to a US person.

Here is the honest accounting for generation one. **You paid** cash, up front, in full, with no product. **You risked** the entire amount, with no recourse and no information rights. **The issuer received** unrestricted cash and committed to a document. That asymmetry was so extreme that it could not persist — not because anyone reformed it, but because the legal exposure became intolerable and because retail participation collapsed after the 2018 drawdown.

I want to flag something about the aggregate story here. You will see confident statistics quoted about ICO outcomes — "X percent of 2017 ICOs were dead within a year", "Y percent traded below their sale price". Those numbers come from studies with wildly different definitions of "dead", "ICO" and "return", built on samples of self-reported projects, and they disagree with each other. I am not going to quote one as fact. The qualitative claim that survives every methodology is narrow and sufficient: **the distribution of outcomes was extremely skewed, most projects did not produce a working product, and a small number produced enormous returns.** That shape — skew, not average — is the single most important statistical fact in this entire post, and it is going to recur.

## Generation two: the IEO and the exchange launchpad

The fix for "anyone can sell anything to anyone" was to put a gatekeeper in the middle. In an **initial exchange offering**, the exchange runs the sale on its own platform. It does the know-your-customer checks, it enforces the geographic restrictions, it lists the token immediately afterward, and — critically — it stakes its reputation on the project not being an outright fraud.

Binance launched the model. Its first launchpad sale, BitTorrent's BTT token, ran on 28 January 2019 and sold out in minutes. That speed was the problem. A first-come-first-served sale on a centralised exchange is a race decided by network latency and server luck, which produces exactly the bot-versus-human dynamic that made ICO gas auctions miserable, just moved indoors.

So the model mutated, and the mutation is the interesting part. Rather than sell to the fastest, the exchange started selling to the **most committed holders of its own token**. The current shape — you will find the live rules on the exchange's own launchpad and launchpool pages, and they change — works roughly like this:

1. The exchange announces a sale and a snapshot window, typically several days long.
2. Your *average* balance of the exchange's native token across that window determines your entitlement, either as lottery tickets or as a pro-rata share of the sale.
3. Winners (or all participants, in the pro-rata version) get a capped allocation at a fixed sale price.
4. The token lists on that exchange within days, usually at a large multiple of the sale price.

![The launchpad lottery pipeline: holding the exchange token through a snapshot, earning tickets, and the small expected value against the large position risk](/imgs/blogs/launchpads-airdrops-and-the-points-meta-5.webp)

Read that list again and notice what you actually bought. You did not buy an allocation. You bought a **lottery ticket, paid for by holding a large position in the exchange's token through a multi-day snapshot window.** The allocation is the prize; the position is the price. And the exchange collects a listing relationship, a marketing event, and — most valuably — persistent structural demand for its own token from everyone who wants to be eligible for the *next* sale. This is one of the mechanisms that gives an exchange token its gravity, and it is a good illustration of why [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues) and why [Binance's everything-exchange model](/blog/trading/crypto-players/binance-the-everything-exchange-and-its-gravity) compounds the way it does.

Let us put numbers on it.

#### Worked example: what a launchpad lottery is really worth

All numbers here are round and illustrative — plug in the real ones from whichever sale you are looking at.

You hold 100 units of the exchange token, and it trades at \$600. That is a **\$60,000 position** you must maintain through a seven-day average-balance snapshot.

- Ticket formula: one ticket per 50 units held, so you get **2 tickets**.
- Across all participants, 600,000 tickets are entered and **30,000 win** — a **5 percent** hit rate per ticket.
- Each winning ticket buys **\$200** of the token at the sale price.
- The token lists at **5×** the sale price, so a winning ticket is worth \$200 × 5 = \$1,000, a profit of **\$800**.

Your expected profit:

$$
\text{EV} = 2 \text{ tickets} \times 0.05 \times \$800 = \$80
$$

Eighty dollars. Now price the other side. You held \$60,000 of a volatile asset for seven days. If that token has a weekly standard deviation of about 4 percent — unremarkable for a large-cap crypto asset — then a one-standard-deviation move is **\$2,400**. Your \$80 of expected launchpad profit is **3.3 percent of a routine weekly wiggle in the position you were forced to hold**.

Now run it backwards and ask what listing multiple would make the lottery matter. Expected profit is 2 × 0.05 × \$200 × (m − 1) = \$20(m − 1), where m is the listing multiple. To generate even \$300 — a mere 0.5 percent adverse move on your \$60,000 — you need \$20(m − 1) = \$300, so **m = 16**. The sale would have to list at sixteen times its price for the lottery to cover a half-percent hiccup in the token you had to hold.

**The intuition: a launchpad allocation is a small option bolted onto a large directional bet. If you would not hold the exchange's token anyway, the lottery is not the trade — the token is, and the lottery is a rounding error on it.**

The **launchpool** variant softens this. Instead of a lottery you stake the exchange token (or a stablecoin) into a pool for a few days and farm the new token pro rata, with no purchase and nothing at risk beyond the price of the asset you staked and the yield you gave up. It is a better deal for the user precisely because it asks for less: no capital committed to the new token at all. It is also a smaller prize.

## Generation three: the IDO and the bootstrapped curve

The decentralised answer to the launchpad was to skip the exchange. An **initial DEX offering** sells the token directly through an automated market maker — a smart contract that quotes a price from a formula rather than an order book, which is the core machinery described in [DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).

The naive version — dump tokens into a pool and let people buy — fails immediately, because whoever transacts first buys at the lowest price on the curve, so the sale becomes a bot race again, this time settled by transaction ordering.

The clever version is the **liquidity bootstrapping pool**. You launch the pool with weights deliberately skewed so the starting price is far *above* what anyone thinks the token is worth, then let the weights drift over hours or days so the quoted price falls continuously unless buyers push it up. Anyone who front-runs the launch overpays. Anyone who waits gets a lower price. The bot advantage evaporates because there is no single advantageous block to win — the price discovery is spread across a window.

Here the accounting is: **you pay** cash on a public curve at a price you choose, **you risk** having bought too early on the descent, and **the issuer receives** cash plus an instantly liquid float and a public price. It is the fairest of the paid mechanisms, and it is also the one where nobody can pretend a sale did not occur — which is exactly why the industry's centre of gravity kept moving.

## Generation four: the retroactive airdrop

On 16 September 2020, Uniswap announced UNI. The structure is worth stating precisely because it became the template that everything since has copied or reacted against.

Per Uniswap's own announcement that day: a genesis supply of **1,000,000,000 UNI**, with **60 percent allocated to the community**, roughly 21.5 percent to team and future employees, about 17.8 percent to investors and a fraction to advisors, with the team and investor allocations vesting over four years. **15 percent of the total supply was made immediately claimable by historical users** — and every address that had ever interacted with the protocol could claim **400 UNI**, regardless of how much or how little it had done. Something on the order of a quarter of a million addresses qualified.

Three things about that design were genuinely new.

**It was retroactive.** Nobody farmed for it, because nobody knew it was coming. The snapshot looked backwards at behaviour that had already happened for its own reasons. That is the only version of this mechanism that unambiguously rewards genuine usage, and it is the only version that can never be repeated by a protocol that has already done it once — because from that moment on, everyone knows.

**It was flat.** 400 UNI to a wallet that had swapped \$50 once, 400 UNI to a wallet that had swapped ten million dollars. A flat per-address grant is maximally egalitarian per person and maximally exploitable per sybil, and the industry has been arguing about that tradeoff ever since.

**It was a gift, legally framed as a distribution rather than a sale.** No money changed hands in the primary distribution. That is the property that made it attractive to lawyers, and it is the reason the mechanism spread so fast.

The template propagated quickly. 1inch airdropped on Christmas Day 2020. dYdX ran a retroactive distribution in September 2021 — and excluded US persons entirely, which is the first widely-noticed instance of the geographic exclusions that are now standard. ENS distributed a quarter of its supply to .eth name holders in November 2021, with an allocation formula that weighted how long you had held the name. Optimism's first airdrop, on 31 May 2022, distributed 214,748,364 OP tokens (a deliberately chosen number: it is 2³¹ divided by ten) to roughly a quarter of a million addresses, and the claim traffic congested the chain it was launched on. Arbitrum's ARB distribution on 23 March 2023 sent about 11.6 percent of its ten-billion total supply — roughly 1.162 billion ARB — to on the order of 625,000 addresses, and it used a **points-based eligibility rubric** in which specific actions earned specific scores.

That rubric is the hinge. The moment eligibility is scored rather than binary, and the moment everyone knows an airdrop is *coming*, the behaviour being measured stops being behaviour and starts being performance. Which brings us to the fifth generation.

## Generation five: the points meta

Somewhere in late 2023, a protocol figured out that it did not need to promise a token at all.

A **points program** works like this. The protocol runs an off-chain scoreboard. You do things — deposit, borrow, refer, hold — and a server owned by the protocol increments a number attached to your wallet. The number lives in the protocol's database, not on a blockchain. It is not a token. It cannot be sold on an exchange. It confers no rights. The terms of service typically say, in one form or another, that points are non-transferable, have no monetary value, do not represent any claim, and may be modified or discontinued at any time.

And people deposit billions of dollars for them.

![Total value locked through a points campaign, rising into the token generation event and falling sharply after](/imgs/blogs/launchpads-airdrops-and-the-points-meta-8.webp)

That curve is the signature of the whole mechanism, and it is worth naming its points because we will use the same illustrative campaign twice more. Deposits build slowly from about **\$0.1 billion** eight months before launch, cross **\$1.2 billion** around the four-month mark as the program gets noticed, and peak at **\$2.0 billion** on the day the token generates. Then the reason to be there expires. Two months later the protocol holds **\$0.7 billion**; four months later, **\$0.5 billion** — a **75 percent** decline from the peak. Nothing broke. The lease simply ended.

Understand what the issuer just bought. Before points, a protocol wanting deposits had to pay for them: emit a token, or pay interest, both of which require having a token or having revenue, and both of which are *contractual*. With points, the protocol gets:

- **Deposits, immediately.** TVL is the metric every dashboard ranks on, every journalist quotes, and every venture fund uses to justify a markup.
- **A user count and a usage graph** that make the next fundraise easier.
- **Free marketing**, because participants become evangelists — a farmer who has deposited real money has an incentive to talk the program up, since a bigger program means a bigger token.
- **Total optionality on the tokenomics.** The conversion rate from points to tokens does not exist yet. Neither does the vesting on your claim, the sybil policy, the excluded countries, or the decision to launch at all.
- **A behavioural dataset** showing exactly how sensitive its users are to incentives — which tells the issuer precisely how little it can pay and still keep them.

There were named examples of every part of this. Blast, launched on 20 November 2023 by the team behind Blur, ran a bridge that accepted deposits before its chain existed and did not permit withdrawals until the network launched the following February; it accumulated deposits in the billions and drew heavy criticism for exactly that one-way structure. EigenLayer's restaking points program accumulated one of the largest TVL figures in the sector during 2024, and when EIGEN launched that October the tokens were initially non-transferable and US and Canadian users were excluded, which produced a second round of criticism. Ethena ran "shards" and then "sats". Linea had LXP, Scroll had Marks, Kamino had Points, Blur ran numbered seasons. For the current TVL series on any of these, DefiLlama's per-protocol pages are the right source rather than my memory; I am not going to quote peak figures I cannot verify as I write.

The one that pays for the whole meta is Hyperliquid. Its points seasons ran through 2024, and at its genesis distribution on 29 November 2024 it allocated **31 percent of its one-billion total supply** to users — on the order of 94,000 addresses — with no venture round preceding it. It was, by a wide margin, the most generous major distribution to actual users that the industry had produced. Every points program launched since has been marketed against the memory of it.

That is the mechanism at the heart of this: **a heavily skewed outcome distribution, plus one enormous visible winner, produces mass participation at negative expected value.** It is the same shape as a lottery, and it works for the same reason.

#### Worked example: what a points program costs the issuer per dollar of TVL

Take an illustrative eight-month campaign.

- At the token generation event the protocol allocates **7 percent of a 1,000,000,000 supply = 70,000,000 tokens** to points holders.
- The token opens at **\$1.50**, so the distribution is worth 70,000,000 × \$1.50 = **\$105,000,000**.
- Average TVL across the campaign was **\$1.2 billion** over **8 months**, which is 8/12 = 0.667 years. So the campaign bought \$1.2B × 0.667 = **\$800 million of "TVL-years"**.

Cost per dollar of TVL per year:

$$
\frac{\$105{,}000{,}000}{\$800{,}000{,}000} = 0.131 = 13.1\%
$$

Now the counterfactual. Attracting the same deposits by simply paying interest at 8 percent would have cost \$1.2B × 8% × 0.667 = **\$64 million in cash**.

So the points route cost about **1.6 times** the cash route. Why would any issuer choose it? Because of *what* it paid with and *when*. The \$64 million would have been real money leaving the treasury every month, contractually owed. The \$105 million was paid in a token the protocol printed, valued at a price the campaign itself manufactured, handed over only at the end, at a rate the protocol chose after the fact, to a set of recipients the protocol also chose after the fact. And it bought a user count and a narrative that interest payments do not buy.

Now flip to your side of the table. Your realised yield was that same 13.1 percent annualised — **if the token holds \$1.50**. If it halves within a month of listing, which is a completely ordinary outcome, your realised yield was 6.5 percent, for eight months of capital lockup and smart-contract risk.

**The intuition: a points program is a variable-rate deposit account where the rate is announced after the term ends, in a currency the borrower prints.**

![A matrix comparing the five distribution methods on who pays, what the user risks, what the issuer receives and what the issuer commits to](/imgs/blogs/launchpads-airdrops-and-the-points-meta-2.webp)

That matrix is the whole tour in one frame. Track the last column — what the issuer commits to — down the rows: a whitepaper, a listing date, nothing, nothing, nothing at all. That column is the history of token distribution.

## The mechanics that decide the outcome

Knowing which generation you are in tells you the shape of the deal. Four pieces of machinery decide how it actually turns out.

### Initial float versus fully diluted valuation

This is the arithmetic that makes a launch look bigger than the money in it, and almost everyone gets it backwards on first encounter.

![A twelve percent float supporting a two billion dollar fully diluted valuation on two hundred forty million dollars of real buying](/imgs/blogs/launchpads-airdrops-and-the-points-meta-3.webp)

#### Worked example: how \$240 million of buying supports a \$2 billion headline

Illustrative numbers throughout — no real token is being described here, only the arithmetic that applies to all of them.

A token launches with a total supply of **1,000,000,000** and a day-one circulating supply of **120,000,000** — a **12 percent float**. It opens at **\$2.00**.

- **Market cap** = 120,000,000 × \$2.00 = **\$240,000,000**
- **FDV** = 1,000,000,000 × \$2.00 = **\$2,000,000,000**

Both numbers are true. Only one of them corresponds to money that has been committed. Now ask the question that makes the difference visible: *what price would clear if all one billion tokens had to find a holder at the same total dollar demand?*

$$
\frac{\$240{,}000{,}000}{1{,}000{,}000{,}000} = \$0.24
$$

The \$2.00 price is **8.3 times** the price at which the entire supply would clear against the demand that actually showed up. The other 88 percent of the supply has never had to find a buyer. It is not that FDV is a lie — it is that FDV is the price at which a small number of buyers agreed to value a very large number of tokens that nobody has yet been forced to sell.

Now add the unlock. In month thirteen a cliff releases **100,000,000 tokens** — 10 percent of the supply, but **83 percent of the existing 120,000,000 float** — in a single day. For the price to stay at \$2.00, roughly \$200,000,000 of genuinely new demand has to appear that day. It very rarely does.

And from the issuer's chair, the same numbers read completely differently: **you sold 12 percent of the supply for real money, and marked the other 88 percent at the price the 12 percent cleared at.**

**The intuition: a low float is not a technicality about supply, it is a lever that converts a modest amount of real buying into a large headline number, and every unlock afterwards is that lever being released.**

This is why Binance Research's May 2024 finding that tokens launched in 2024 averaged a market-cap-to-FDV ratio near 12 percent is the single most useful number in this whole area. It says the *typical* 2024 launch had roughly seven-eighths of its valuation sitting in supply that had never traded. If you want the cap-table mechanics behind those locked slices, [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) is the companion piece.

### The claim-day sell pressure curve

An airdrop creates, in one instant, tens or hundreds of thousands of holders whose cost basis is zero. There is no other event in finance quite like it.

![Airdropped tokens splitting into instant sellers, gradual sellers, holders and unclaimed, converging on a single exchange order book](/imgs/blogs/launchpads-airdrops-and-the-points-meta-4.webp)

Before the numbers, a warning about what you can and cannot know here. You will see claims like "X percent of airdrops are dumped within 24 hours." Those figures come from on-chain dashboards whose methodology varies enormously — what counts as a sell, whether transfers to exchanges count, whether unclaimed tokens are in the denominator, whether one wallet's behaviour is weighted by size. **I am not going to quote an aggregate as fact, because public reporting does not pin it down and the dashboards disagree.** What I will do is show you the arithmetic with clearly labelled illustrative numbers, so you can substitute the real ones from a dashboard you have actually read. The right places to look are DefiLlama's airdrops section and the Dune Analytics dashboards published per-token at claim time — and when you read one, check the methodology note before the headline.

#### Worked example: the claim-day overhang ratio

An airdrop distributes **100,000,000 tokens**. The token opens at **\$1.20**, so \$120,000,000 of notional value is handed to the public in one moment.

Suppose the recipients split like this (illustrative, not a reported figure):

| Cohort | Tokens | Share |
| --- | --- | --- |
| Sell within 24 hours | 40,000,000 | 40% |
| Sell within 30 days | 25,000,000 | 25% |
| Hold | 25,000,000 | 25% |
| Never claim | 10,000,000 | 10% |

Day-one sell pressure is 40,000,000 × \$1.20 = **\$48,000,000** of market sell orders.

Against that, suppose the order book holds **\$12,000,000** of bids within 20 percent of the opening price. The ratio that matters:

$$
\text{overhang ratio} = \frac{\$48{,}000{,}000}{\$12{,}000{,}000} = 4.0
$$

Four dollars of supply for every dollar of nearby demand. Any ratio above 1.0 means **the opening price is not the clearing price** — it is just the last price before the supply arrived. The book gets walked down until enough bids accumulate to absorb the flow.

Now look at the two levers that change the number, because they explain almost every design choice you see in modern airdrops:

- **Vest the airdrop.** Spread the same 40,000,000 tokens over six months and monthly pressure becomes about 6,700,000 tokens, or \$8 million — an overhang ratio of **0.67**, and it now sits below 1.0. The airdrop stops being an event and becomes a drip. This is why vesting on airdrop claims went from unheard-of to routine.
- **Buy the other side of the book.** A market maker paid to quote tightly deepens the bids and lowers the ratio directly. That deal has its own economics, covered in [the loan-plus-options deal: how market makers get paid](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid) and in [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers).

**The intuition: the supply side of an airdrop is a crowd of a hundred thousand independent sellers, and the demand side is one order book. Unless the issuer does something about the ratio between them, the price on the screen at the open is a fiction with a very short life.**

### Sybil farming and the filter arms race

Every airdrop rule is a specification, and a specification with money attached will be optimised against. The question is only whether it is cheaper to satisfy the spirit of the rule or to satisfy its letter many times over.

![A sybil farm branching from one funding wallet into five hundred wallets, and the five heuristics a filter uses to detect it](/imgs/blogs/launchpads-airdrops-and-the-points-meta-6.webp)

#### Worked example: why a per-wallet floor is the most expensive line in an airdrop design

Illustrative figures again — the point is the ratio, not the levels. Take an allocation formula with a **minimum grant of 300 tokens** for any qualifying address, and a heavily sublinear scale above that — so a wallet with a hundred times more activity gets maybe eight times more tokens. This is not a strawman; flat and heavily-capped formulas are common precisely because they read as egalitarian.

**The honest whale.** One wallet, \$100,000 deployed for six months, doing everything genuinely. Under the sublinear scale it earns **2,500 tokens**.

**The farm.** The same \$100,000, split across **500 wallets at \$200 each**, each doing the twelve transactions needed to clear the minimum. Each clears the 300-token floor:

$$
500 \times 300 = 150{,}000 \text{ tokens}
$$

Costs: 500 wallets × 12 transactions × \$0.60 of gas = \$3,600, plus funding and defunding each wallet at roughly \$1.20 a round trip = \$600. Call it **\$5,000** all in with bridging.

At \$1.00 per token, the farm collects \$150,000 against the whale's \$2,500. Even after costs, **the farm did sixty times better with identical capital.**

Now suppose the filter is good and catches 60 percent of the farm's wallets. The farm keeps 200 × 300 = 60,000 tokens — still **twenty-four times** the whale. A filter has to be extraordinarily good to change the conclusion, which is why the real fix is not detection but formula design: make the per-wallet grant scale with something expensive to fake, and remove the floor.

**The intuition: a per-address minimum converts capital into addresses at a fixed exchange rate, and addresses are nearly free. The filter is a patch on an economic hole in the formula.**

The detection side has become genuinely sophisticated. The standard heuristics look for the things a farm cannot avoid: **funding-graph clustering** (many wallets funded from one source, or draining to one destination), **timing correlation** (wallets acting in the same block or on the same cadence), **amount fingerprints** (identical or suspiciously round transfer sizes), **behavioural uniformity** (the same twelve actions in the same order), and **nonce patterns** (sequential, scripted transaction histories). Every one of those is a consequence of the farm being *efficient*. A farm that randomises everything enough to defeat all five costs so much in gas and human attention that it stops beating the honest whale.

Real airdrops have leaned on this hard. Arbitrum's distribution used a sybil analysis by the on-chain analytics firm Nansen to strip flagged addresses before the snapshot. Earlier, Hop Protocol ran a community-driven sybil hunt in 2022 in which participants were paid to submit evidence against clusters — turning the crowd's own knowledge into the filter. Connext ran a similar exclusion process. Reported counts of removed addresses vary between the projects' own posts and third-party analyses, and I would rather point you at the projects' published sybil lists than repeat a figure from memory.

LayerZero's approach in mid-2024 was the most game-theoretically interesting one anyone has tried. It offered self-identified sybils an amnesty: **self-report by a deadline and keep 15 percent of your allocation; get reported by someone else and receive nothing**, with a bounty paid to the reporter out of the forfeited allocation. That is a prisoner's dilemma deliberately constructed to make farms defect on themselves. It also required claimants to make a donation per token claimed as a "proof of donation" step, which added a real cost to claiming a large number of small allocations. Whether it worked is genuinely debated; that it changed the incentive structure rather than just the detection technology is not.

One framing worth keeping. Sybil accusations against named projects and named clusters are, in the general case, **allegations supported by on-chain heuristics, not proof of identity.** A cluster of wallets funded from one exchange withdrawal might be one farmer or might be twelve friends splitting a withdrawal. Filters have false positives, and people with legitimate claims have been excluded by them. Every project that publishes a sybil list also publishes an appeals process, and both facts matter.

### The carry trade: farming with borrowed capital

Once points scale linearly with deposit size, someone will lever up. The loop is mechanical: deposit a yield-bearing asset, borrow stablecoins against it, buy more of the asset, redeposit, repeat.

#### Worked example: the looped points position

Illustrative rates and levels; substitute the live ones from whichever lending market you are looking at.

You start with **\$10,000** of a liquid-staking token that earns **3.5 percent** staking yield and accrues points at one point per dollar per day.

You borrow at **60 percent loan-to-value** and re-deposit, three times:

| Loop | Deposit added | Cumulative position | Cumulative debt |
| --- | --- | --- | --- |
| 0 | \$10,000 | \$10,000 | \$0 |
| 1 | \$6,000 | \$16,000 | \$6,000 |
| 2 | \$3,600 | \$19,600 | \$9,600 |
| 3 | \$2,160 | \$21,760 | \$11,760 |

Your position is now **\$21,760** on \$10,000 of your own money — **2.18× the points** of the unlevered version.

Now the carry. Staking yield on the full position: \$21,760 × 3.5% = **\$762 a year**. Borrow cost at 7 percent on the debt: \$11,760 × 7% = **\$823 a year**.

$$
\text{net carry} = \$762 - \$823 = -\$61 \text{ per year}
$$

You are **paying \$61 a year, plus the opportunity cost of your \$10,000, to earn 2.18× of a points balance whose conversion rate does not exist yet.**

And now the part people skip. Your loan-to-value is \$11,760 / \$21,760 = **54 percent**. If liquidation triggers at 75 percent, solve for the collateral drawdown x that gets you there:

$$
\frac{\$11{,}760}{\$21{,}760 \times (1-x)} = 0.75 \implies 1-x = 0.7203 \implies x = 28\%
$$

A **28 percent** fall in your collateral's value liquidates the position. If the collateral is a liquid-staking token trading against its own underlying, 28 percent is a catastrophic depeg and unlikely. If the collateral is the volatile asset itself, 28 percent is a bad fortnight.

**The intuition: leverage multiplies your points linearly and your liquidation probability non-linearly. You are underwriting a tail event in order to buy a coupon that has not been declared.**

There is a second-order effect worth naming: **looped deposits inflate TVL by counting the same original dollar several times.** In the table above, \$10,000 of genuinely new capital produced \$21,760 of TVL. When a protocol's headline TVL is largely looped, the number describes leverage, not adoption. Hold that thought — it is the buyer's problem, and we come back to it at the end.

### Why the median airdrop disappoints and a handful are transformative

Put the four mechanics together and the outcome distribution follows almost deterministically.

The claim-day overhang ratio is above 1.0 for most launches, so the opening price is not a clearing price. The float is small, so the headline valuation is levered on a thin base of real buying. The unlock schedule adds supply on known dates against demand that has to be freshly created. And the airdrop recipients themselves have a zero cost basis, which makes them the most price-insensitive sellers in the market.

Against that, one thing occasionally goes right: a protocol has genuine, fee-paying usage that survives the end of the incentive, its float is large enough that the price is real, and the token accrues something. Then the distribution is a windfall that lasts.

The result is a distribution with a long left mass and a very long right tail — most outcomes clustered at "small and fading", a few at "life-changing". **The mistake almost everyone makes is to reason about it with an average.** In a skewed distribution the mean is dragged upward by the tail, so "the average airdrop returned X" is simultaneously true and useless: you will almost certainly not receive the average. The median is the number that describes your likely experience, and the median is much, much lower than the mean.

I want to be explicit about the epistemics here rather than dress this up with a statistic. **No public aggregate of airdrop returns is stable enough across methodologies to state as a general law.** Samples differ, price references differ, survivorship bias is enormous (dead tokens fall out of datasets), and the "return" depends entirely on an assumed sell date.

The closest thing to a defensible aggregate I am aware of is Keyrock's 2024 study *Airdrops in the Barren Desert*, which examined 62 airdrops across six chains and reported that the large majority — on the order of 88 percent — of airdropped tokens declined in price within months of distribution. Carry two caveats with that number. Its scope is 62 airdrops in a single year, not airdrops in general. And "declined" is doing heavy lifting: declined from *what* reference price, measured over *what* window, is precisely the definitional choice that makes studies in this area disagree with one another. I am citing it as reported through secondary coverage, having been unable to reach Keyrock's own publication while writing this. Treat it as one carefully-scoped study pointing in the same direction as the mechanics above — not as a measured constant, and not as a prediction about the specific airdrop in front of you.

What is robust is the *shape*: heavy skew, with a small number of outcomes doing all the work. Plan around the shape, not around a number someone put in a thread.

## The asymmetry: the issuer sees your hand before choosing the rules

Everything above is mechanics. This section is the actual thesis.

![Two columns: the user commits capital and irreversible on-chain actions early, while the issuer chooses snapshot, rate, vesting, sybil rules and jurisdiction later](/imgs/blogs/launchpads-airdrops-and-the-points-meta-9.webp)

In an ordinary financial contract, the terms are fixed before you commit. You know the coupon before you buy the bond, the strike before you buy the option, the price before you buy the share. Whatever happens afterwards, both sides agreed to the same document.

In a points program, the sequence is inverted. You commit capital and take irreversible on-chain actions for months. **Then** the issuer decides:

- **When to snapshot.** Choosing a date is choosing a winner. A snapshot on the campaign's first day rewards the earliest believers; a snapshot on the last day rewards whoever showed up last week with the most money.
- **The conversion rate.** Points-to-tokens is a free parameter, set with full knowledge of how many points exist and what the token might be worth.
- **Who counts as a sybil.** The filter's threshold is a dial. Turning it up saves supply and excludes some genuine users; turning it down does the opposite. Either choice is defensible and neither is auditable by you.
- **Whether your claim vests.** Handing you tokens over twelve months instead of instantly cuts day-one sell pressure, which is good for the price and bad for your optionality — and it is decided after your capital is already deployed.
- **Which jurisdictions are excluded.** More on this below, but note the timing: you can farm for eight months and find out at the claim page that your country is blocked.
- **Whether to launch at all.** The ultimate option. A protocol can run a points campaign, collect the TVL and the users and the fundraise, and simply never ship a token.

Every one of those is an option held by the issuer and written by you. In options language, you sold a strip of options for a premium that has not been fixed, on an underlying that does not exist, with the issuer choosing the settlement terms after expiry. That is not a criticism of any particular team — many teams have converted points generously and on time. It is a structural description that stays true regardless of intent.

The one clean defence against it is also simple, and it will be the backbone of the framework at the end: **be in a position where you are fine if the answer to all six of those questions is the worst one.**

## The part that is never in the announcement thread: KYC, geo-blocks and tax

Three practical realities that turn a paper gain into a smaller, later, or non-existent one.

**Geographic exclusion is normal now, and it is decided late.** dYdX's September 2021 distribution excluded US persons. EigenLayer's EIGEN claim in October 2024 excluded users in the United States and Canada. Exchange launchpads and launchpools maintain their own restricted-jurisdiction lists and change them as regulation moves. The pattern to internalise: **the exclusion list is published at claim time, not at farm time.** If you are in a jurisdiction that projects commonly block, you should assume a meaningful probability that eight months of farming ends at a page saying your region is not eligible — and attempting to route around a geo-block typically breaches the terms you are claiming under.

**Identity verification increasingly gates the claim.** Larger distributions, and anything routed through a centralised exchange, increasingly require know-your-customer verification before tokens move. This is the quiet re-centralisation of a mechanism whose original appeal was permissionlessness, and it is a direct consequence of the enforcement history in generation one.

**Receiving tokens is very often a taxable event, at a value you did not choose.** In the United States, Revenue Ruling 2019-24 (issued 9 October 2019) addressed cryptocurrency received from an airdrop following a hard fork, and held that the taxpayer has **ordinary income equal to the fair market value of the tokens at the moment they gain dominion and control** over them. The ruling's specific facts are about hard forks, and its application to modern protocol airdrops is a matter of professional interpretation rather than settled law — but the principle practitioners generally work from is that tokens you can move are income when you can move them. In the United Kingdom, HMRC's Cryptoassets Manual (the airdrops section, CRYPTO21250) draws a line by whether you did something in return: an airdrop received without any service or expectation is generally not income, while one received in return for a service is.

The practical bite of this is the timing mismatch, and it is worth one line of arithmetic. **Suppose you claim 10,000 tokens on the day they open at \$3.00. That is \$30,000 of income recognised at claim, and a tax bill computed on \$30,000. If the token is at \$0.60 three months later and you have not sold, you owe tax on \$30,000 while holding \$6,000 of assets.** People have been genuinely ruined by that gap in previous cycles. This is general information about how the rules are structured, not tax advice for your situation — the right move is to talk to someone who does this professionally in your jurisdiction, before you claim rather than after.

## Common misconceptions

**"An airdrop is free money."** The tokens are free; the position is not. You paid in capital lockup, gas, time, smart-contract exposure and — in most jurisdictions — a tax liability that crystallises at claim regardless of what you do next. "Free" describes the purchase price, not the cost.

**"A big FDV means the project is worth a lot."** FDV is a price multiplied by a supply, and on a 12 percent float the price was set by a small amount of buying. As the worked example showed, the same \$240 million of demand supports either a \$240 million market cap or a \$2 billion FDV depending only on how much supply is locked. The FDV is not evidence of value; it is evidence of the float.

**"High TVL means the protocol is working."** TVL bought with points is rented capital. It also double-counts looped positions — our looped farmer turned \$10,000 into \$21,760 of TVL without adding \$11,760 of new money to the ecosystem. The number that matters is TVL that stays after the incentive ends and pays fees while it is there.

**"Farming more wallets is just being efficient."** It is, right up until the filter. But it also directly changes the design of the next airdrop: every successful farm makes the next formula stingier for everyone, which is why per-address floors keep disappearing.

**"The team will be fair because it is bad for their reputation not to be."** Sometimes true, and there are teams with genuinely good records here. But reputation is a repeated-game argument, and a token launch is often close to a one-shot game for the entity that matters. Structure your exposure so you do not need the counterparty to be generous.

**"Points are basically a token."** They are a number in someone else's database, and the terms of service usually say so explicitly. The distance between a point and a token is the entire set of issuer options listed in the asymmetry section.

## How it shows up in real markets

**Uniswap, 16 September 2020.** The template. 400 UNI to every historical user, 15 percent of supply distributed at once, a genuinely retroactive snapshot nobody could have farmed. It worked precisely because it was a surprise — and by working, it guaranteed nothing would ever be a surprise again. Every points program running today is a descendant of that one announcement.

**dYdX, September 2021.** The first widely-noticed geographic exclusion, with US persons cut out entirely. It established that a "permissionless" distribution mechanism would be routed around national regulation by excluding nations, and that the exclusion would be a product decision made by the issuer.

**Arbitrum, 23 March 2023.** Roughly 1.162 billion ARB, about 11.6 percent of a ten-billion supply, to on the order of 625,000 addresses, allocated by a published points rubric with sybil filtering by Nansen applied beforehand. This is the moment eligibility became a scored, gameable specification rather than a binary fact, and the industry's farming infrastructure organised itself around it within weeks.

**Blast, from 20 November 2023.** The purest expression of the points structure: a bridge that took deposits before the chain it was bridging to existed, with withdrawals disabled until launch. It drew billions and heavy criticism simultaneously, and it demonstrated that "deposit now, terms later" was not a bug in the model — it *was* the model.

**EigenLayer, October 2024.** A restaking points program that accumulated one of the sector's largest TVL figures, followed by an EIGEN launch where the tokens were initially non-transferable and US and Canadian users were excluded. Two of the issuer's six options — vesting-equivalent restrictions and jurisdiction — exercised in one announcement, after the capital was already committed. Both were later revised under pressure, which is itself informative about where the leverage sits.

**LayerZero, mid-2024.** The self-report amnesty (keep 15 percent if you confess, get nothing if someone else reports you, with a bounty to the reporter) plus a required donation per token claimed. The most creative attempt yet to attack farming with incentives rather than heuristics. Debated in its effectiveness, undebated in its ingenuity.

**Hyperliquid, 29 November 2024.** 31 percent of supply to roughly 94,000 users at genesis, with no venture round ahead of it. The counterexample that keeps the whole meta alive. It is worth being precise about why it is the exception: the protocol had genuine trading revenue, a large distribution, and no cap table of early investors waiting to sell into the users. Most points programs share none of those three properties.

## A farmer's framework: when the expected value is actually positive

Now let us make the whole thing operational, starting with the calculation almost nobody does.

![The farmer's expected-value equation decomposed into three probabilities and two certain costs](/imgs/blogs/launchpads-airdrops-and-the-points-meta-7.webp)

#### Worked example: your actual expected value over six months

Every input below is an illustrative assumption, not a measured statistic — the value of the exercise is that it forces you to write down numbers you are usually allowed to leave vague.

You deploy **\$10,000** into a protocol with a points program for **six months**.

**The certain costs first**, because they are the part people leave out:

- **Opportunity cost.** If the safe alternative for that capital pays 4 percent a year, six months costs you \$10,000 × 4% × 0.5 = **\$200**.
- **Transaction costs.** Fifteen transactions a month for six months is 90 transactions at roughly \$0.60 each = \$54, plus about \$26 to bridge in and out. Call it **\$80**.
- **Total certain cost: \$280.**

**Now the uncertain payoff**, decomposed honestly:

- **P(the protocol ships a token within twelve months) = 0.50.** Many do not.
- **P(you qualify, given it ships) = 0.70.** Snapshot timing, minimum thresholds, sybil false positives, geo-blocks.
- **E[allocation value, given you qualify]**, modelled as the skewed distribution it actually is:

| Outcome | Probability | Value |
| --- | --- | --- |
| Small allocation | 60% | \$150 |
| Decent allocation | 30% | \$800 |
| Large allocation | 10% | \$4,000 |

$$
E[\text{allocation}] = 0.6 \times \$150 + 0.3 \times \$800 + 0.1 \times \$4{,}000 = \$90 + \$240 + \$400 = \$730
$$

Gross expected value:

$$
0.50 \times 0.70 \times \$730 = \$255.50
$$

Net expected value: \$255.50 − \$280 = **−\$24.50.**

Slightly negative — and notice how *reasonable* every input was. Nothing in that table is pessimistic. The median farm loses a small amount of money quietly.

**Now the two sensitivities that matter most.**

**Scale up the capital.** Deploy \$50,000 instead of \$10,000. Opportunity cost rises **linearly** to \$1,000. But allocation formulas are deliberately **sublinear** in capital — if allocation scales roughly with the square root, five times the capital earns about 2.24 times the allocation, so E[allocation] becomes \$1,635 and gross EV becomes 0.50 × 0.70 × \$1,635 = \$572. Net: \$572 − \$1,080 = **−\$508**. *More capital made it worse.* This is the single most counter-intuitive result in farming, and it follows directly from the fact that the cost is linear and the reward is not.

**Zero out the opportunity cost.** Suppose you were going to hold this asset in this protocol anyway — you use it, you want the exposure, the capital had nowhere better to be. Then the \$200 disappears and net EV is \$255.50 − \$80 = **+\$175.50**.

**The intuition: farming has positive expected value almost exclusively for the person who was going to be there anyway. For everyone else it is a small negative carry bought in exchange for a lottery ticket — which is a fine thing to buy, as long as you know that is what you bought.**

That result generalises into four questions, asked in order.

![A decision tree with four sequential questions, any 'no' leading to negative expected value](/imgs/blogs/launchpads-airdrops-and-the-points-meta-10.webp)

1. **Would this capital be sitting idle anyway?** If no, the opportunity cost has probably already eaten the expected value — as the sensitivity above showed, it does so faster as size grows.
2. **Can you lose 100 percent of the deposit and be fine?** New protocols with points programs are new, unaudited-in-production code holding large balances, which is the highest-value target class in the industry. Points do not compensate you for smart-contract risk because points are not priced.
3. **Is the allocation formula sublinear per wallet, with no minimum floor?** If there is a per-address floor, a farm will out-earn you by an order of magnitude and your share will be diluted accordingly. That is a reason to expect a smaller allocation, not a reason to build a farm.
4. **Are you in a jurisdiction that is likely to remain eligible?** Ask before you deploy, not at the claim page.

If all four are yes, farm it — and **size it as a lottery ticket rather than as a position**, because the outcome distribution is a lottery's, not an investment's. If any one is no, the expected value was already negative before you started.

And the clean case for *not* farming: when the only reason the capital is there is the points. That is the definition of mercenary capital, and mercenary capital is precisely what the sublinear formula, the sybil filter and the vesting schedule are all designed to pay as little as possible.

## A buyer's framework: what a points-driven TVL number is worth

The other side of this trade is the person deciding whether to buy the token once it lists — and the metric they are usually shown is TVL.

![A waterfall discounting headline TVL for capital that leaves at launch, looped double-counting and non-fee-paying deposits](/imgs/blogs/launchpads-airdrops-and-the-points-meta-11.webp)

Apply three haircuts, in this order.

**Haircut one: how much of it is renting?** Ask what fraction of deposits arrived after the points program was announced. Capital that showed up for points leaves when points convert — that is not cynicism, it is the contract. The TVL chart around a token generation event tells you this directly; DefiLlama's per-protocol pages carry the series, and the shape you are looking for is a peak at TGE and a cliff after it.

**Haircut two: how much of it is the same dollar counted twice?** Looping inflates TVL mechanically. If the protocol supports using its own deposit receipt as collateral to borrow and redeposit, some meaningful share of the headline is leverage rather than capital. Our worked example turned \$10,000 into \$21,760 of TVL single-handedly.

**Haircut three: how much of it pays a fee?** Deposits that generate no revenue are a cost centre wearing a growth metric's clothes. The question is not "how much is deposited" but "how much revenue did the deposits produce last month, and would that revenue survive the incentive ending?"

Run the illustrative version: **\$2.00 billion headline, minus \$1.30 billion that leaves at TGE, minus \$0.30 billion of looped double-counting, minus \$0.20 billion that pays no fee, equals \$0.20 billion of sticky fee-paying TVL — 10 percent of the headline.** Whether the real numbers are better or worse than that for any given protocol is exactly the research question, and the answer is knowable from public on-chain data. The mistake is to skip the exercise because the headline number was printed in a large font.

Then ask the question that the whole post has been building toward: **at what float did this list, and what is the unlock schedule?** A protocol with genuinely sticky TVL and a 12 percent float still has 88 percent of its supply arriving on a published calendar. Those are two separate assessments — is the business real, and is the supply schedule survivable — and a token can fail the second while passing the first.

## When this matters to you

If you never farm an airdrop and never buy a newly-listed token, this still matters, because the same structure keeps appearing wherever a platform can pay in something it prints. The pattern — commit capital now against a scoreboard whose conversion rate the issuer sets later — is not unique to crypto. It is a loyalty program with a bigger denominator.

If you do participate, the practical takeaways are narrow and boring, which is usually a good sign:

- **Price the opportunity cost of locked capital every time.** It is the term everyone omits, it is certain, and it scales linearly while your reward does not.
- **Assume every issuer option resolves against you** — late snapshot, stingy rate, vesting on the claim, your country blocked — and check that you are still fine.
- **Read the float and the unlock schedule before the FDV.** Those two numbers tell you more about the next twelve months of price than any narrative will.
- **Treat the tail as a tail.** One person's transformative airdrop is not evidence about your expected value; it is the reason the sample you see is biased.
- **Find out how tokens are taxed where you live before you claim,** because the taxable moment is often the claim, not the sale.

For the next layers of this: [the lifecycle of a token from seed round to unlock cliff](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) covers the private pipeline that runs before any of this becomes visible to you; [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) covers what you actually own at the end of it; and [centralized crypto exchanges: Binance and Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) covers the venue that decides whether any of it trades at all.

This is educational material about how these mechanisms work, not advice to participate in any of them.

## Sources & further reading

Primary sources for the headline figures in this post, plus the dashboards worth reading before you trust any aggregate:

**Distribution announcements and project documentation**

- Uniswap, "Introducing UNI", 16 September 2020 — the genesis supply, the 60 percent community allocation, the 15 percent historical-user distribution and the 400 UNI per address. `blog.uniswap.org/UNI`
- Arbitrum Foundation, airdrop eligibility and distribution documentation, March 2023 — the points rubric and the token allocation. `docs.arbitrum.foundation`
- Optimism Collective, Airdrop #1 announcement, 31 May 2022 — the 214,748,364 OP distribution.
- LayerZero, sybil policy and Proof-of-Donation announcements, May–June 2024 — the self-report terms and the claim requirements.
- Hyperliquid documentation, genesis distribution, November 2024 — the 31 percent genesis allocation to users.
- Binance, Launchpad and Launchpool rules pages — the current snapshot windows, ticket formulas and restricted jurisdictions. These change; read the live page for the sale in front of you.

**Research and data**

- Binance Research, low-float / high-FDV analysis, May 2024 — the finding that 2024 token launches averaged a market-cap-to-FDV ratio near 12 percent.
- DefiLlama — per-protocol TVL series (the shape around a token generation event) and the airdrops section. `defillama.com`
- Dune Analytics — per-token claim and sell-through dashboards, published at claim time. Read the query, not just the headline; methodology varies enormously between dashboards.
- Nansen — sybil analysis published around the Arbitrum distribution, March 2023.
- Keyrock, *Airdrops in the Barren Desert*, 2024 — 62 airdrops across six chains; reported that roughly 88 percent of airdropped tokens declined within months. Cited here as reported via secondary coverage; Keyrock's own page was unreachable at the time of writing, so read the study directly before leaning on the figure.

**Legal and tax primary text**

- SEC, Report of Investigation Pursuant to Section 21(a): The DAO, Release No. 34-81207, 25 July 2017.
- *SEC v. W.J. Howey Co.*, 328 U.S. 293 (1946) — the investment-contract test.
- *SEC v. Telegram Group Inc.*, S.D.N.Y., preliminary injunction 24 March 2020; settlement June 2020 (approximately \$1.2 billion returned, \$18.5 million penalty).
- *SEC v. Kik Interactive Inc.*, S.D.N.Y., summary judgment 30 September 2020; \$5 million penalty.
- SEC administrative proceeding against Block.one, settled 30 September 2019, \$24 million civil penalty.
- IRS Revenue Ruling 2019-24, issued 9 October 2019 — ordinary income at fair market value upon dominion and control over airdropped units following a hard fork. `irs.gov/pub/irs-drop/rr-19-24.pdf`
- HMRC Cryptoassets Manual, airdrops (CRYPTO21250) — the "something in return" test. `gov.uk/hmrc-internal-manuals/cryptoassets-manual`

**A note on what is deliberately absent.** This post states no aggregate as a general law about airdrops. It contains no "average airdrop return", no percentage of airdropped tokens sold within a fixed window, and no count of how many launches traded below their listing price. Those figures circulate widely and are not supported by any methodologically stable public source I would stand behind. The one aggregate that appears — Keyrock's 88 percent — is presented with its scope, its definitional weakness, and the fact that I reached it through secondary coverage rather than the study itself. Everywhere else the argument needed a number of that kind, I used an explicitly labelled illustrative example instead, so you can substitute the real figures from a dashboard whose methodology you have personally checked.
