---
title: "Token Foundations and Treasuries: Crypto's On-Chain Central Banks"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "How Swiss foundations and DAO treasuries came to hold billions of tokens they never bought, why selling them is the job rather than a betrayal, and how to read their wallets without fooling yourself."
tags: ["crypto", "tokenomics", "dao-treasury", "ethereum-foundation", "on-chain-analysis", "governance", "token-supply", "treasury-management", "grants", "crypto-players"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 54
---

> [!important]
> **TL;DR** — Token foundations and DAO treasuries are the largest price-insensitive holders in crypto: permanent, legally insulated from token holders, and funded by a genesis allocation they never bought. That combination makes them the closest thing the asset class has to a central bank, and their wallets are unusually readable.
>
> - A foundation is a legal structure with **no owners** — a Swiss *Stiftung* or a Cayman foundation company answers to a supervisory authority and a founding deed, not to the people holding its token.
> - Its balance came from a **genesis allocation**, so its cost basis is effectively zero and it has no profit motive, no redemption pressure, and no fiduciary duty to a shareholder.
> - Its job is to fund research and grants by **selling the token it stewards**. That is the design, not a scandal — but it makes the entity a structural seller for as long as it exists.
> - The Ethereum Foundation's published treasury was **\$970.2 million as of 31 October 2024** (Ethereum Foundation Report, November 2024), 81% of it crypto and 99.45% of that crypto in ETH. Its [June 2025 treasury policy](https://blog.ethereum.org/2025/06/04/ef-treasury-policy) caps annual spending at **15% of the treasury** with a **2.5-year fiat buffer**.
> - A "\$2 billion" DAO treasury that is 90% its own illiquid token is not \$2 billion. Mark-to-market assumes a buyer who does not exist at that size.
> - The number to remember: a **10,000 ETH (~\$42.7M) foundation-linked deposit to Kraken reported by on-chain trackers in September 2025** was roughly **0.3% of one ordinary day's ETH volume**. It could not have moved the price. It was still news — and understanding why is the whole point of this post.

In September 2025, on-chain trackers reported that a wallet linked to the Ethereum Foundation deposited 10,000 ETH — about \$42.7 million at the time, per that reporting — into Kraken. Crypto Twitter did what crypto Twitter does. *They're dumping on us. They know something. This is why ETH underperforms.*

Here is the awkward arithmetic. Ether trades tens of billions of dollars a day across spot venues. Against a genuinely conservative \$15 billion day, \$42.7 million is **0.28%** of the volume. Against a quiet \$10 billion day it is 0.43%. If that flow were worked through an execution desk over five sessions — which is exactly what the Foundation's own published policy says it does, splitting sales "into multiple smaller orders to reduce market impact" — it would be **0.06%** of daily volume. You could not detect it in the tape with a microscope.

So the deposit could not have moved the price. And yet it was, correctly, news. Not because of the size. Because of *who* was selling, what that seller's constraints are, and what those constraints imply about the next four quarters.

That gap — between what a treasury transaction mechanically does to price and what it *tells you* — is the most under-exploited edge in crypto analysis. Almost everyone reads these wallets wrong in one of two directions: either they scream "dump" at a rounding error, or they ignore the entities entirely because "foundations don't trade." Both are mistakes, and they are opposite mistakes.

This post builds the whole thing from zero: what a foundation legally *is*, where its tokens came from, what its treasury actually holds, how a payment gets authorised, why grants are supply, why staking income does not save it, why buybacks are the same operation with the sign flipped — and finally, a monitoring checklist that tells you which signals are real and which are noise.

## The map: who is actually holding the supply

Before anything else, we need one piece of vocabulary, because almost every argument about token supply is really an argument about definitions.

![A tree diagram splitting total token supply into the tradable float and the permanent non-float holders, with foundation and DAO treasuries highlighted as the largest permanent holders](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-1.webp)

**Total supply** is every token that exists. **Circulating supply** is the subset that is not contractually locked. **Float** — the concept that actually matters for price — is the subset that is genuinely available to trade at something near the current price. Float is always smaller than circulating supply, because circulating supply counts tokens sitting in wallets whose owner has no intention of selling at any price you would recognise.

The diagram above is the mental model for this entire post. Supply splits into two worlds. On the left, the **tradable float**: exchange order books, market makers' inventory, the funds and individuals who are actually price-sensitive. This is where price is discovered, and it is usually a shockingly small fraction of total supply.

On the right sit the **permanent holders** — wallets that are outside the float, sometimes for years, sometimes forever:

- **Foundation treasuries.** A genesis grant, held by a legal entity whose purpose is to spend it slowly on ecosystem development.
- **DAO treasuries.** Tokens controlled by an on-chain governance process, usually denominated in the DAO's own token.
- **Team and investor allocations still vesting.** These are genuinely economic sellers — they bought (or earned) at a low price and want out. Covered in detail in [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock).
- **Burned or protocol-owned supply.** Gone, or locked by a contract with no key.

The first three all *can* re-enter the float. The difference between them is **motive**, and motive is everything. A VC fund holds tokens to sell them at a profit — its behaviour is a function of price, its lockup, and its LPs' patience ([how VCs move price through listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) covers that machinery). A foundation holds tokens to *fund a budget*. It sells because payroll is due in March, not because ETH broke a moving average.

That is the property that makes foundations analytically special, and it is the property this post keeps returning to: **they are price-insensitive**. Their selling is driven by a cost structure and a written policy, not by a view on price. In traditional markets, we have a name for a large, permanent, price-insensitive balance sheet with a published operating rule. We call it a central bank.

For scale: the Ethereum Foundation held roughly **0.26% of all ether** as of end-October 2024, per its own report. The Optimism Collective earmarked **850 million OP — 20% of the token's initial supply** — for retroactive public goods funding alone, [per Optimism's own posts](https://www.optimism.io/blog/optimism-2024-year-in-review). These are not rounding errors in the cap table. They are among the largest single positions in their respective assets, and unlike a whale's wallet, they publish their intentions.

## Foundations, from zero: what the thing legally is

Start with the word itself, because it is doing a lot of quiet work.

A **foundation**, in the European legal tradition, is not a company. A company has shareholders who own it and can vote to change what it does. A foundation has *no owners at all*. It is created by taking a pile of assets and legally dedicating them to a stated purpose. Once dedicated, the assets belong to the purpose. Nobody can take them back — not the founder, not the donors, not the people who later hold the token.

If that sounds strange, the closest everyday analogy is a university endowment or a family charitable trust. Someone gives money, writes down what it is for, appoints a board to spend it on that thing, and then loses the ability to change their mind. That is a feature. It is exactly why the structure was chosen.

![A comparison matrix of Swiss Stiftung, Cayman foundation company and unwrapped DAO across owners, supervision, who sets the purpose, and whether token-holder votes bind](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-2.webp)

### The Swiss Stiftung

The dominant structure in crypto is the Swiss *Stiftung*, governed by **Article 80 and following of the Swiss Civil Code (ZGB)**. A Stiftung is created by dedicating assets to a particular purpose; it has no members and no shareholders; its purpose is fixed in a founding deed; and it is supervised by a state authority — the cantonal foundation supervisory authority, or the federal authority (the *Eidgenössische Stiftungsaufsicht*, ESA) for foundations operating nationally or internationally.

**Stiftung Ethereum** is registered in Zug. Its stated purpose is the promotion of new open decentralised software architectures, with a focus on the Ethereum protocol. The **Solana Foundation** is likewise a Zug *Stiftung*. So are the Cardano Foundation, the Web3 Foundation (Polkadot) and the DFINITY Foundation. This is not a coincidence of geography. Zug's "Crypto Valley" clustering matters, but the legal properties matter more:

1. **No shareholders.** There is nobody with an equity claim on the treasury, which helps a protocol argue it is neutral infrastructure rather than a company's product.
2. **Purpose is sticky.** Changing a Stiftung's purpose is hard and requires supervisory approval. A board cannot simply pivot the money into something else.
3. **Assets are legally separated from the founder.** The people who wrote the protocol do not own the pile.
4. **State supervision exists but is thin.** The supervisory authority checks that the foundation spends on its stated purpose and files accounts. It does not run the treasury, opine on token sales, or represent token holders.

Read point four again, because it is where most people's intuition breaks. There *is* an overseer. It is a Swiss regulator interested in whether charitable purpose is being honoured. It is emphatically **not** an advocate for the people who bought the token on an exchange.

### The Cayman foundation company

The newer structure — and now the default for DAOs — is the **Cayman Islands foundation company**, created under the Foundation Companies Act. It is a deliberate hybrid: it has the separate legal personality and limited liability of a company, and the purpose-driven, beneficiary-free character of a trust.

Its two defining features for crypto:

- **It may be ownerless.** A foundation company can exist with no members and no shareholders, which is the closest legal analogue to "a DAO" that a court will actually recognise.
- **It is steered by a supervisor.** The structure requires at least one supervisor, who has *no ownership and no economic entitlement*, and whose role is simply to ensure the directors honour the foundation's governing documents. Supervision without ownership is precisely the DAO-shaped hole the structure was designed to fill.

Why bother wrapping a DAO at all, if the whole point is decentralisation? Because a DAO has no legal personality. It cannot sign a contract, open a bank account, hold an exchange account, employ anyone, buy insurance, or be sued — and the last one matters more than people expect, since in many jurisdictions an unwrapped DAO risks being treated as a **general partnership**, in which every participant is jointly and severally liable for everything the collective does. The wrapper converts unlimited personal exposure into a bounded entity. By-laws can also be kept private, which is either prudent operational security or a transparency problem depending on your priors.

### Who it actually answers to

Put the three structures side by side, as in the figure above, and one column tells the whole story: **token-holder vote binding?**

- **Swiss Stiftung:** No. Advisory only. A token vote is input to a board that is legally accountable elsewhere.
- **Cayman foundation company:** Only if the by-laws explicitly bind the directors to on-chain outcomes — and you often cannot read the by-laws.
- **Unwrapped DAO:** Binding on-chain, unenforceable off-chain. The vote moves the tokens because a contract executes; it cannot compel a human to do anything.

This is not a cynical reading. It is the design. Foundations were chosen *because* they insulate a protocol's funding from the churn of token-holder sentiment — the same way a central bank's independence is a feature until the day you disagree with it.

The canonical demonstration is the **Tezos Foundation**, 2017–18. Tezos raised roughly \$232 million in a July 2017 token sale, per contemporaneous reporting. The Swiss foundation controlled the funds; the project's creators, Arthur and Kathleen Breitman, controlled the code. When a dispute erupted between foundation president Johann Gevers and the Breitmans — reportedly beginning with a compensation package the Breitmans considered excessive and inadequately disclosed — the contributors who had sent money discovered they had no lever at all. The dispute stalled the project for months. It was resolved in February 2018 when Gevers stepped down and was replaced by Ryan Jesperson, following board turnover, not a token vote.

The lesson generalises: **the wrapper decides who controls the money, and the wrapper is chosen before you ever see the token.** For the broader version of this argument, see [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto).

## Where the balance came from: a gift, not a purchase

Now the second load-bearing fact, and the one that most changes how you should model these entities.

**The foundation did not buy its tokens.**

At genesis, a fixed supply is created and allocated by a spreadsheet: some to the public sale, some to the team, some to investors, some to an ecosystem fund, and some to the foundation. No money changed hands for the foundation's slice. It was minted and assigned. (Where a foundation *did* receive sale proceeds — as with Ethereum's 2014 sale or Tezos's 2017 raise — the proceeds were the raise, and the token position was still an allocation rather than an open-market purchase.)

Three consequences follow, and they are the reasons a foundation behaves unlike every other holder you track:

**1. Cost basis is effectively zero.** There is no "underwater" for a foundation. A VC that entered at \$0.10 has a psychological and reporting anchor at \$0.10; below it, they are down, and their LPs can see it. A foundation has no such anchor. Price is an input to a budget, not to a P&L.

**2. There is no redemption pressure.** A hedge fund can be forced to sell by investor withdrawals. A market maker can be forced to sell by a risk limit or a margin call — the mechanics of that are in [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does). A foundation cannot be redeemed. Its liabilities are salaries and grant commitments, which are slow, known in advance, and denominated in fiat.

**3. There is no shareholder to answer to for underperformance.** Nobody fires the board for holding through a 70% drawdown. The horizon is genuinely permanent.

Put those together and you get a holder with an infinite horizon, a zero cost basis, no forced-selling channel, and a fiat cost structure. That is a *very* unusual object. The only comparable entities in traditional markets are sovereign wealth funds, central banks, and permanent university endowments — and all three are famous for being the participants whose flows are worth tracking precisely because they are not trying to be clever.

The corollary is uncomfortable and worth stating plainly: because their cost basis is zero, **foundations are never sellers of last resort and always sellers of first resort.** Any budget they fund by selling is funded at whatever price exists. The seller's indifference to price is exactly what makes the flow predictable — and predictability is the raw material of edge. For how a token's initial allocation determines everything downstream, see [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table).

## What is actually inside a foundation treasury

Enough theory. What does one of these balance sheets look like?

The Ethereum Foundation is the right specimen, because it publishes. Its first-ever public report, released in April 2022, disclosed a treasury of **\$1.6 billion as of 31 March 2022**, of which **80.5% was ETH** — representing **0.297% of total ether supply** — with the remaining 19.5% in non-crypto assets and other cryptocurrencies. Spending in 2021 was approximately **\$48 million**. (All figures per that April 2022 Ethereum Foundation Report.)

Its November 2024 report told a different story:

| Line item | Value | As of |
|---|---|---|
| Total treasury | \$970.2M | 31 Oct 2024 |
| Crypto assets | \$788.7M (81.3%) | 31 Oct 2024 |
| — of which ETH | 99.45% of crypto | 31 Oct 2024 |
| Non-crypto investments and assets | \$181.5M (18.7%) | 31 Oct 2024 |
| Share of total ETH supply | ~0.26% | 31 Oct 2024 |
| 2022 expenditure | \$105.4M | full year 2022 |
| 2023 expenditure | \$134.9M | full year 2023 |

*Source: Ethereum Foundation Report, November 2024, as reported by [The Block](https://www.theblock.co/post/325166/ethereum-foundation-holds-788-million-in-crypto) and [CoinDesk](https://www.coindesk.com/tech/2024/11/08/ethereum-foundations-treasury-shrunk-39-over-2-12-years-to-970m). All figures as of the dates shown; balances move, so check current values on-chain and in the latest report before relying on any of them.*

The treasury fell **39%** between those two snapshots, from \$1.6 billion to \$970.2 million. The Foundation attributed this to roughly \$240 million of spending across 2022–23 combined with an ETH price decline of about 22%, from roughly \$3,300 to roughly \$2,600.

Stare at that for a moment. A 39% decline in the endowment of the entity funding Ethereum's core research, over two and a half years, during which the protocol shipped the Merge. Nothing went wrong. This is simply what happens when you fund a fiat cost structure out of a volatile asset without a spending rule.

Which is why, on **4 June 2025**, the Foundation published its first formal **Treasury Policy**. Two parameters do most of the work:

- **A = 15% of treasury** as the annual operating expenditure cap.
- **B = 2.5 years** of that opex held as a buffer in fiat-denominated assets.

The policy states an intention to "reduce annual opex roughly linearly over the next five years, ending at a long-term 5% baseline that is common for endowment-based organizations." And it specifies the selling mechanism explicitly: the Foundation "will periodically calculate the deviation of the treasury's fiat-denominated assets from the Opex Buffer (B) target and determine how much, if any, Ether will be sold over the next three months," executed "via fiat off-ramps or onchain swaps."

That is a published reaction function. Central banks have those too.

![A two-column balance sheet showing the Ethereum Foundation's held assets against the fiat buffer its own policy requires, with the resulting shortfall marked in red](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-3.webp)

### Worked example 1: runway, and why the budget is a price bet

Let's apply the published rule to the published balance sheet. Every input below is a real, dated figure; the *combination* is an illustration of how the rule works, not a claim about what the Foundation actually did.

**Step 1 — the naive runway.** Treasury \$970.2M, 2023 spend \$134.9M.

\$970.2M ÷ \$134.9M = **7.2 years**

Comfortable. But this assumes the treasury's value is fixed, which it emphatically is not, and it assumes spending is fixed, which the policy says it is not either.

**Step 2 — the policy budget.** The rule caps annual opex at 15% of treasury:

15% × \$970.2M = **\$145.5M**

Slightly above the 2023 actual. Fine so far.

**Step 3 — the buffer requirement.** The rule wants 2.5 years of that budget held in fiat-denominated assets:

2.5 × \$145.5M = **\$363.8M**

**Step 4 — the shortfall.** How much fiat-denominated stuff was actually held? \$181.5M in non-crypto assets.

\$363.8M − \$181.5M = **\$182.3M short**

To close it, you sell ETH. At the roughly \$2,600 price cited for that snapshot:

\$182.3M ÷ \$2,600 = **~70,100 ETH**

So a treasury policy, applied to a real balance sheet, generates a mechanical multi-quarter selling programme of roughly seventy thousand ETH — without anyone forming a single opinion about the price of ether. That is what "price-insensitive seller" means in practice.

**Step 5 — the reflexive part.** Now halve the ETH price. The crypto block goes from \$788.7M to \$394.4M; the non-crypto block stays at \$181.5M.

New treasury: \$394.4M + \$181.5M = **\$575.9M**
New budget: 15% × \$575.9M = **\$86.4M**

That is a **41% cut to the research budget** from a 50% price decline. Not because anyone panicked — because the rule says so.

![An XY chart showing treasury value and the 15% annual budget as functions of the ETH price, with the region below roughly $2,370 shaded red where the rule funds less than 2023 actual spending](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-4.webp)

**Step 6 — the crossover.** At what ETH price does the 15% rule fund less than the Foundation actually spent in 2023? Let *P* be the ETH price. The crypto block scales as \$788.7M × (P ÷ \$2,600) = 0.3033 × P (in millions):

0.15 × (0.3033P + 181.5) = 134.9
0.0455P + 27.2 = 134.9
P = **~\$2,370**

Below roughly \$2,370 per ETH, a disciplined 15% rule funds less than the 2023 run-rate. The shaded red region in the chart is that zone.

> **The intuition:** a percentage-of-treasury spending rule converts a volatile balance sheet into a volatile *research roadmap*. It protects the foundation from ever going broke, and it guarantees that the amount of work funded is a leveraged bet on the token price. There is no version of this that is both stable and honest.

This is precisely the trade-off endowments made a century ago and resolved with smoothing rules — spending a fixed percentage of a *trailing three-year average* rather than the spot value. Whether crypto foundations adopt that refinement is one of the more consequential and least-discussed questions in the space.

## The job: funding public goods by selling the token you steward

Now the part that generates the most heat and the least light.

A foundation's operating model is: hold token → sell some token → pay researchers, auditors, conference organisers, grant recipients and lawyers in fiat → repeat. There is no other model available to it. It has no revenue. It has no customers. It has one asset.

![A cash-flow timeline showing the genesis allocation inflow at t=0, quarterly token sales and grant outflows, and staking and DeFi yield inflows from 2026](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-5.webp)

The timeline above is the shape of it. At t=0, a large green inflow that was never purchased. Then, forever: red outflows for grants and payroll, and periodic red outflows of token into the market to fund them. The two green inflows on the right — DeFi yield and staking yield — are recent additions we will size later, and the punchline is that they are small.

**A foundation is a structural seller by construction.** Not by choice, not by malice, not by loss of faith. If it never sold, it would never pay anyone, and the protocol it exists to fund would go unfunded.

Once you accept that, the emotional charge around "the foundation is selling" mostly evaporates — and the *analytical* content becomes visible. The question is never "are they selling?" (yes, always, it is the job). The questions are: **how much, on what schedule, into what liquidity, and does the published rule tell me what happens next?**

### Why an EF sale becomes a news event anyway

Three reasons, and only one of them is legitimate.

**Reason one, illegitimate: the size fallacy.** People see a large absolute dollar figure and assume large price impact. This is the single most common error in on-chain analysis and it is worth killing properly.

#### Worked example 2: the sale that cannot move the price

Take the widely-reported September 2025 deposit: **10,000 ETH, about \$42.7 million** (implying roughly \$4,270/ETH at the time), moved from an Ethereum Foundation-linked wallet to Kraken.

**Step 1 — establish the denominator.** Ether spot volume across major venues routinely runs in the tens of billions of dollars per day. Take a deliberately conservative \$15 billion.

**Step 2 — the ratio.**

\$42.7M ÷ \$15,000M = **0.28% of one day's volume**

Even on a quiet \$10 billion day: \$42.7M ÷ \$10,000M = **0.43%**.

**Step 3 — the execution adjustment.** The Foundation's policy says sales are split into multiple smaller orders. Spread over five sessions:

\$42.7M ÷ 5 = \$8.5M/day → \$8.5M ÷ \$15,000M = **0.057% of daily volume**

**Step 4 — the comparison.** A mid-sized market maker will turn over more than \$42.7 million of ETH inventory before lunch. A single institutional block through an OTC desk — see [OTC desks and moving size without moving price](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price) — can be several multiples of it and never touch a public order book at all.

> **The intuition:** at this size, the trade is invisible. Any price move that day was caused by something else, and attributing it to the foundation is storytelling. Mechanical impact requires a meaningful fraction of daily volume; 0.3% is not that.

**Reason two, illegitimate: the "they know something" fallacy.** A price-insensitive seller executing a published buffer rule has no informational content about price. That is what price-insensitive means. If the Foundation sells in a quarter when the fiat buffer is short, it will sell whether ETH is at \$1,500 or \$5,000. Reading a directional signal into it is reading a coin flip.

**Reason three, and this one is legitimate: the sale reveals the state of the balance sheet.** *That* is real information. A sale tells you the buffer was below target, which tells you something about the previous quarter's prices and spending, which — combined with the published rule — lets you forecast the *next* few quarters of supply. Not the price. The supply. Those are different things, and the second is forecastable.

There is also an unavoidable optics problem. The entity that stewards the protocol and shapes its roadmap is simultaneously a large seller of its token. In an equity market, an insider selling triggers mandatory disclosure precisely because the conflict is presumed. Crypto foundations sit in a strange place: the conflict is structurally identical, the disclosure regime is voluntary, and the entity's defence — "we are spending on the ecosystem, not enriching ourselves" — is usually true and completely unverifiable from the outside. Publishing a treasury policy in advance is fundamentally an attempt to defuse this by making the selling boring and pre-announced. It is the same move a central bank makes with forward guidance, and for the same reason.

## What the on-chain evidence actually shows

If you want to check any of this yourself, here is how the plumbing works — and where it lies to you.

**Address labels.** Block explorers and intelligence platforms attach human-readable names to addresses. On Etherscan, the address `0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe` carries the label **"EthDev"** and has been associated with Ethereum Foundation development activity since the network's earliest days. Arkham Intelligence maintains an "Ethereum Foundation" *entity* that clusters multiple addresses. Nansen does something similar with its wallet labels. All three are useful. None is authoritative.

Labels come from three sources: self-disclosure by the entity, clustering heuristics (addresses that transact together are inferred to share an owner), and manual research. Heuristics are wrong sometimes. Entities open new addresses. A "foundation-linked" wallet in a news headline may be one hop removed from anything the foundation actually controls.

**The labelled-address gap.** This is the trap that catches even careful analysts. Arkham's Ethereum Foundation entity has been reported at roughly **102,400 ETH (~\$211 million) across 14 addresses**, with total tracked assets around **\$270.9 million**, in its 2026 top-holders research. The Foundation's own report put the treasury at **\$970.2 million as of 31 October 2024**.

Those are not contradictory. They are measuring different things:

- Staked ETH sits in validator deposits and liquid-staking positions that an entity page may not attribute back to the wallet.
- Non-crypto assets — \$181.5 million of them in the 2024 report — have no on-chain footprint at all.
- Addresses the entity has never publicly linked are, by construction, unlabelled.
- The snapshots are eighteen months apart, across which the Foundation both spent and sold.

**The rule: labelled-address totals are a lower bound on what an entity holds, and an unreliable one.** Use them for *flows* — direction, size, timing, counterparty — and use published reports for *stocks*.

**What a transfer actually means.** A transfer to an exchange address is one of at least four things:

1. A sale — the tokens are about to hit an order book.
2. An OTC settlement — the exchange is acting as a venue for a pre-negotiated block trade that never touches the book.
3. Custody rotation — moving between the entity's own hot and cold storage, or onboarding to a new custodian.
4. Collateral — posting the asset against a loan or a derivatives position.

You usually cannot distinguish these from the transfer alone. Sometimes you can infer it later: if a deposit is followed by a stablecoin withdrawal to the same entity's wallets, a sale is a reasonable inference. Reporting around the September 2025 episode noted a subsequent withdrawal of roughly 3.39 million DAI to Foundation-linked addresses, which is the pattern you would expect from a sale — but the honest framing remains "consistent with a sale," never "the foundation dumped."

That distinction matters more than pedantry. **"Dumped" asserts intent.** It says the seller chose the moment to inflict maximum damage. Almost every foundation sale that has been described that way was, on the evidence, a scheduled buffer top-up. Where you must characterise a sale as opportunistic, cite the reporting and the entity's stated reason, and label it as an allegation. The mechanism is interesting enough without the melodrama.

## Diversification: stablecoins, RWAs, and the DeFi-deployment debate

If holding 99% of your endowment in one volatile asset is obviously imprudent, why does everyone do it?

Three reasons, in decreasing order of respectability:

1. **Signalling.** The Foundation's 2024 report put it directly: "We choose to hold the majority of our treasury in ETH. The EF believes in Ethereum's potential, and our ETH holdings represent that long-term perspective." A steward that diversifies out of its own asset says something loud about its conviction, and markets will hear it.
2. **Alignment.** If the foundation's resources rise and fall with the network's success, its incentives point the same way as everyone else's.
3. **Inertia and optics.** Selling is a headline. Not selling is not a headline. The path of least resistance is to hold.

Against that sits arithmetic. An entity with fiat liabilities and a single volatile asset is running an unhedged currency mismatch — the same structure that destroys emerging-market borrowers who fund in dollars and earn in local currency. Every foundation that has thought carefully about this has moved in the same direction: **more stablecoins, more real-world assets, more yield-bearing deployment.**

### The Ethereum Foundation's DeFi turn

On **13 February 2025**, the Ethereum Foundation deployed **45,000 ETH** into lending protocols: 10,000 into Spark, 10,000 into Aave Prime, 20,800 into Aave Core and 4,200 into Compound, [as reported by Cointelegraph](https://cointelegraph.com/news/ethereum-foundation-120-million-aave-spark-compound). At roughly \$2,600/ETH that was about **\$120 million**. On **29 May 2025** it borrowed \$2 million of GHO — Aave's stablecoin — against its position, which is a materially different act from selling: it converts the ETH into spendable dollars *without* transferring the ETH to anyone. In **March 2026** it extended the programme with roughly 3,400 ETH into Morpho vaults, including 1,000 ETH into Morpho Vaults V2.

The June 2025 treasury policy formalised this with a set of criteria the Foundation calls **"defipunk"** — a deployment must be permissionless (a binary requirement), default to self-custody, use free/libre open-source licensing rather than source-available, support privacy, keep core logic trustless with minimised admin keys, and offer distributed front-ends with direct contract access. Whatever one makes of the framing, it is a real, published, checkable standard.

The debate this triggered is genuine, and both sides are serious:

**For deployment.** Yield reduces the need to sell. Borrowing stablecoins against ETH funds operations without adding a single token to the float. And a foundation that will not use its own ecosystem's financial infrastructure is making an implicit statement about that infrastructure's safety.

**Against deployment.** Smart-contract risk is real and the tail is fat — a treasury that funds core protocol development is exactly the treasury you least want exposed to an oracle failure. Borrowing against ETH introduces liquidation risk, which converts a permanent holder into a forced seller in precisely the market conditions where forced selling hurts most. And a foundation's deployment is an implicit endorsement that moves real capital, which makes protocol selection a political act. For the mechanics of the protocols involved, see [DeFi protocols: Uniswap, Aave and MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).

### The DAO version: real-world assets

DAO treasuries have moved further and faster, because they have less signalling to protect and more governance-driven pressure to show responsible stewardship.

The **Arbitrum DAO** is the clearest case. Its **Stable Treasury Endowment Program (STEP)** allocated **35 million ARB** to purchase tokenised US Treasury products from managers including Franklin Templeton, Spiko and WisdomTree ([The Block](https://www.theblock.co/post/353631/arbitrum-dao-us-treasurys)) — an explicit conversion of governance-token exposure into an asset whose yield is uncorrelated with crypto. A second tranche, STEP 2.0, followed at around \$15 million, and a treasury management committee — GFXlabs, Northlakes Legal, Nethermind, Entropy and Karpatkey — was constituted to run the mandate.

The **ENS DAO** went furthest. It appointed the professional treasury manager **karpatkey** in late 2022 and initiated an *endowment* on **7 March 2023**, funded with a first tranche of **16,000 ETH**. The explicit goal is an endowment that funds ENS operations from investment returns rather than from selling the governance token. Per karpatkey's reporting, the endowment has generated roughly **\$2.92 million** in net DeFi results since inception, and in 2024 those results represented about **12% of ENS's total revenue**.

Twelve percent is a modest number, and that modesty is the honest lesson: professional treasury management is real and worth doing, and it does not remotely replace the need to either sell tokens or earn protocol revenue.

## The reflexivity trap: a \$2 billion treasury that isn't

Here is the single most consequential accounting error in crypto, and it is committed daily by dashboards, journalists and DAO contributors alike.

Treasury trackers report the value of a DAO's holdings by multiplying token count by last traded price. For a treasury holding stablecoins and ETH, that is fine. For a treasury holding its own governance token — which describes the overwhelming majority of them — it is close to meaningless.

The dated evidence is stark. According to **DeepDAO's tracker, as of 31 March 2023**, aggregate DAO treasuries totalled **\$25.1 billion**, of which roughly \$22 billion was classified as liquid. The largest individual treasuries at that snapshot:

| DAO | Reported treasury (31 Mar 2023) |
|---|---|
| Optimism Collective | \$5.5B |
| Arbitrum | \$4.4B |
| BitDAO | \$2.6B |
| Uniswap | \$2.5B |
| Polygon | \$1.5B |

Aggregate DAO treasury value has since fluctuated with token prices — DeepDAO's tracker has shown totals in the \$20–30 billion range across 2024–26 — and the composition problem has not changed at all: protocol DAOs typically hold the large majority of their treasury in their own governance token.

Why is that a problem? Because the reported number answers a question nobody asked. "What are these tokens worth at the last price?" is not the same question as "what could this treasury actually spend?"

![A before-and-after comparison showing a treasury reported at $200 million realizing $147.5 million across four quarters of selling, a 26% haircut](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-6.webp)

### Worked example 3: the own-token haircut

Everything in this example is **illustrative** — a made-up DAO with round numbers, to isolate the mechanism.

**The setup.** Meridian DAO holds **100 million MRD**. MRD last traded at **\$2.00**. Every dashboard reports a **\$200 million treasury**.

Two other facts, which the dashboards do not show:

- **Float: 300 million MRD.** The treasury is therefore **33% of the tradable float**.
- **Average daily volume: \$8 million**, which at \$2.00 is **4 million MRD per day**.

**Step 1 — how fast can it sell?** A widely used rule of thumb is that you can execute up to about **10% of average daily volume** without leaving obvious footprints. That is 400,000 MRD per day. To sell 25 million MRD:

25,000,000 ÷ 400,000 = **62.5 trading days ≈ one quarter**

So liquidating a quarter of the treasury takes a quarter of a year, at the *disciplined* pace. Liquidating all of it takes a year — and that is the optimistic case, because it assumes volume stays constant while you are selling, which it will not.

**Step 2 — the reflexive spiral.** Selling pushes the price down. A lower price reduces volume, which reduces how much you can sell per day, which extends the programme, which extends the selling. Assume — and this is the assumption doing the work — that the DAO's own supply pushes the realised average price down by roughly 12–18% each quarter:

| Quarter | Tokens sold | Avg realised price | Proceeds |
|---|---|---|---|
| Q1 | 25M MRD | \$1.85 | \$46.3M |
| Q2 | 25M MRD | \$1.60 | \$40.0M |
| Q3 | 25M MRD | \$1.35 | \$33.8M |
| Q4 | 25M MRD | \$1.10 | \$27.5M |
| **Total** | **100M MRD** | — | **\$147.5M** |

**Step 3 — the haircut.**

\$147.5M ÷ \$200M = 73.75% → a **26% haircut** on the reported value.

And 26% is the *good* case. It assumes no front-running of a publicly telegraphed governance-approved sale programme, no panic among other holders watching the treasury address, and no correlated market drawdown.

**Step 4 — scale it up.** Apply the same logic to a headline "\$2 billion treasury, 90% in its own token." The \$1.8 billion own-token slice takes the haircut; the \$200 million of stablecoins does not:

(\$1.8B × 0.74) + \$0.2B = \$1.33B + \$0.2B = **~\$1.53 billion**

And if that treasury is a larger share of its float than Meridian's 33%, the realistic haircut is worse — realisable value in the 50–60% range is entirely plausible for a treasury that dominates its own float.

> **The intuition:** a treasury denominated in its own token is not an asset in the ordinary sense. It is an option on the token that the DAO can only exercise by damaging the thing the option is written on.

### The deeper problem: correlation equals one

There is a second-order version of this that matters even more.

A DAO treasury holding its own token is **long its own equity with a liability structure denominated in dollars**. The correlation between the value of the asset and the DAO's ability to fund itself is exactly 1 — and, worse, it is *negatively* correlated with need. When the token is falling, the ecosystem needs funding most (developers leaving, incentives required, sentiment poor) and the treasury is worth least. When the token is ripping, the treasury is enormous and the ecosystem needs nothing.

This is the argument for diversification stated without any moralising. It is not about prudence or conviction. It is about **not concentrating your funding capacity in the asset that fails at the same time your funding need spikes.** Every institution that has learned this lesson learned it the expensive way.

## Treasury governance: multisigs, delegates and the quorum problem

A treasury is not just a balance. It is a *permission system*, and the permission system is where most of the real behaviour lives.

![A branching governance flow from forum temperature check through Snapshot vote, on-chain proposal, quorum check, timelock and multisig execution, with the failure branch marked in red](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-7.webp)

### The plumbing

A **multisig** is a wallet requiring *m* of *n* signatures to move funds — a 5-of-9 multisig needs five of nine designated key-holders to approve each transaction. Almost every foundation treasury and a large share of DAO treasuries sit behind one. This creates a small, named, humanly-identifiable group with de facto control of a very large pot, which is either reassuring or alarming depending on the day.

A **delegate** is someone to whom token holders assign their voting power. Because most holders never vote, delegates accumulate outsized influence — in most large DAOs, a handful of delegates can decide any given proposal between them.

A **timelock** is a mandatory delay between a vote passing and the transaction executing. Uniswap's UNIfication proposal, for instance, entered a **two-day timelock** after passing. The delay exists so that if a malicious proposal slips through, there is a window to react.

**Quorum** is the minimum participation required for a vote to count at all.

### Where it actually breaks

Follow the diagram left to right and count the chokepoints: forum temperature check, off-chain Snapshot vote, on-chain proposal, quorum, timelock, multisig execution. Six stages, each with its own failure mode. The most common outcome by a wide margin is not a bad decision. It is **no decision** — a proposal that never reaches quorum, and a treasury that does nothing for another quarter.

This has a real analytical consequence. **A DAO treasury's default state is paralysis, which makes it more price-insensitive than a foundation, not less.** A foundation with a published policy will sell on schedule. A DAO treasury that cannot assemble a quorum will sit on its own token through an entire cycle. If you are modelling supply, the DAO's tokens are closer to "permanently locked" than most models assume — right up until the quarter when a well-organised delegate coalition decides otherwise.

### Case study: Arbitrum's AIP-1

The definitive demonstration ran in **March–April 2023**.

Arbitrum's first governance proposal, AIP-1, asked token holders to approve the allocation of **750 million ARB** — roughly \$1 billion at the time — to the Arbitrum Foundation, to fund "special grants," service-provider reimbursements and operating costs without requiring a full on-chain proposal for each disbursement.

Token holders reacted badly. More than **78%** of votes cast went against it. Then the situation deteriorated: it emerged that nearly all 750 million ARB had **already been moved** to a Foundation-controlled address before the vote concluded, that **40 million ARB** had been lent to what was described as a sophisticated financial-market actor, and that **10 million ARB** had already been converted to fiat for operating costs. A Foundation representative characterised AIP-1 as informing the community of decisions already made rather than requesting permission — which was, if anything, more inflammatory than the proposal.

The Foundation subsequently pledged no near-term ARB sales, committed to splitting AIP-1 into separate votes, and rebuilt the process. But the episode established the durable lesson:

> **Ratification is not authorisation.** A vote held after the tokens have moved is a press release with a progress bar.

It also demonstrated the structural point from earlier in this post. The Foundation was legally entitled to do what it did. The token vote was advisory. The outrage was about the gap between what holders believed governance meant and what the legal wrapper actually provided — and that gap exists, to some degree, in every one of these structures. [Cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) works through the general version of this asymmetry.

## Grants are supply

Here is a mechanism that is systematically underweighted because it does not look like selling.

A foundation awards a grant. The grant is denominated in the token. The recipient is a four-person team in Berlin who need to pay rent in euros. So the recipient sells. The tokens move from the treasury to the float, one grant at a time, without a single "foundation sells" headline.

**Every grant programme is a token distribution programme.** The only question is how big it is relative to the market that has to absorb it.

![A grouped bar chart comparing daily sell pressure as a percentage of average daily volume for Optimism's 2024 retro funding versus a program shipping 5% of float in one quarter](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-8.webp)

### Worked example 4: grants are supply, but rarely the supply that matters

The Optimism Collective runs the most transparent large grants programme in crypto, so we can use real figures.

**The programme.** The Collective has allocated **850 million OP** to Retro Funding from the initial token supply — **20% of the initial supply**. In **2024**, Retro Funding distributed **20.4 million OP** through three rounds to **374 projects** ([Optimism, "2024 year in review"](https://www.optimism.io/blog/optimism-2024-year-in-review)).

**Step 1 — average grant size.**

20,400,000 OP ÷ 374 projects = **~54,500 OP per project**

**Step 2 — convert to sell pressure.** Assume — and these are assumptions, clearly labelled — OP at \$1.50 and 60% of grants sold within 90 days to cover costs:

20.4M OP × 60% = 12.24M OP
12.24M OP × \$1.50 = **\$18.4 million**
\$18.4M ÷ 90 days = **~\$204,000 per day**

**Step 3 — compare to volume.** Against an assumed \$80 million average daily volume:

\$204,000 ÷ \$80,000,000 = **0.26% of daily volume**

An entire year of the flagship public-goods programme of a major L2 produces sell pressure of roughly a quarter of one percent of daily volume. It is *below the noise floor.* Nobody could detect it.

**Step 4 — now the case that does matter.** Consider a programme that ships **5% of float in a single quarter** — the shape of an aggressive liquidity-mining or incentive campaign. Assume the token's daily volume is about 1% of float, which is typical for a mid-cap:

5% of float ÷ 60 trading days = **0.083% of float per day**
0.083% ÷ 1% = **8.3% of average daily volume, every day, for a quarter**

That is thirty times the visibility threshold. That is a programme you can see in the chart.

> **The intuition:** the size of a grants programme relative to the treasury tells you nothing. The size relative to *daily volume* tells you everything. A \$50 million grants budget is invisible in a liquid token and catastrophic in an illiquid one.

The practical corollary: when you evaluate a token's forward supply, retroactive grants and research funding are usually not where the pressure is. It lives in **unlock cliffs** and **short-dated incentive campaigns**. The unlock calendar is where you should be looking — see [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) — and points programmes carry the same structure, examined in [launchpads, airdrops and the points meta](/blog/trading/crypto-players/launchpads-airdrops-and-the-points-meta).

## Income: staking, fees and the endowment dream

If selling the token is the problem, the obvious solution is income. Every large foundation has reached for it. It is worth sizing honestly.

The Ethereum Foundation began staking treasury ETH on **24 February 2026**, with an initial deposit of **2,016 ETH** and a stated target of about **70,000 ETH**. It reached that target on **3 April 2026**, per CoinDesk's reporting, having staked roughly \$143 million worth of ether. Staking rewards flow back to the treasury, funding research without selling principal. The Foundation used a multi-jurisdictional validator setup with open-source tooling to avoid single points of failure, and by late April 2026 was reported to be unwinding a liquid-staking position — roughly 21,270 ETH via batched wstETH withdrawals — in favour of its own validators.

### Worked example 5: can staking fund the foundation?

**Step 1 — the yield.** Take a net staking yield of about **3.0%** on 70,000 ETH:

70,000 ETH × 3.0% = **2,100 ETH per year**

**Step 2 — convert to dollars.** The reported \$143 million for 70,000 ETH implies roughly **\$2,043 per ETH**:

2,100 ETH × \$2,043 = **~\$4.3 million per year**

That sits inside the \$3.9–5.4 million annual range CoinDesk estimated. Good — our arithmetic and the reporting agree.

**Step 3 — compare to the cost base.** The same reporting put the Foundation's annual expenses at roughly **\$100 million**:

\$4.3M ÷ \$100M = **4.3% of annual expenses**

Staking the whole 70,000 ETH covers about two and a half weeks of operations.

**Step 4 — what would full funding require?** To fund \$100 million a year entirely from a 3% net yield, you need principal of:

\$100M ÷ 0.03 = **\$3.33 billion**
\$3.33B ÷ \$2,043 = **~1.63 million ETH**

Against a total ether supply of roughly 120 million, that is about **1.35% of all ETH** — more than five times what the Foundation's 2024 report indicated it held, and far more than any labelled-address tracker attributes to it today.

**Step 5 — read the policy backwards.** This is the step worth doing. The treasury policy's long-term target is a **5% annual opex** baseline. A 5% spending rate implies a treasury **20 times** annual spending:

20 × \$100M = **\$2 billion**

The Foundation's published treasury was \$970.2 million in October 2024. So the glide path from 15% to 5% over five years is not, arithmetically, a plan to double the endowment. It is a plan to **reduce spending until it fits the endowment that exists.**

> **The intuition:** staking is a real improvement and a rounding error. It converts a fraction of the treasury from dead weight into income, and it does not change the fundamental fact that a foundation funded by a token allocation must eventually shrink its spending to a sustainable percentage of a volatile pile.

### Staking as policy, not income

There is a second use of staking that has nothing to do with revenue, and the Solana Foundation is the clearest example.

In **November 2020**, the Solana Foundation [announced a delegation strategy](https://solana.com/news/announcing-the-solana-foundation-delegation-strategy) committing **100,000,000 SOL — described at the time as over 80% of the Foundation's treasury** — to be delegated to validators. The stated goals were not financial. They were to improve censorship resistance by spreading stake more evenly, and to encourage validator growth by giving smaller nodes a baseline delegation that makes running a node economically viable. The algorithm explicitly "dynamically and uniformly divides and delegates" the pool to **maximise the minimum number of unique nodes constituting 33% of global stake** — which is a direct attack on the network's Nakamoto coefficient, the number of entities that would have to collude to halt the chain.

That is a foundation using its balance sheet as a **policy instrument** rather than an investment. It is the single closest analogue in crypto to a central bank conducting open-market operations to hit a target that is not profit. The mechanics of staking itself are covered in [crypto mining, staking and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev).

## Buybacks and burns: the mirror image

Now flip the sign, because the two operations are the same accounting entry in opposite directions.

![Two parallel pipelines showing a foundation converting genesis tokens into fiat for grants versus a protocol converting trading fees into token burns](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-9.webp)

A foundation converts **token → market → fiat → salaries**: value leaves the token and becomes payroll, and float increases. A protocol buyback-and-burn converts **fees → protocol → token purchase → destruction**: value flows into the token, and float decreases.

Same pipe. Opposite direction.

The defining recent example is Uniswap's [**UNIfication**](https://blog.uniswap.org/unification), proposed jointly by Uniswap Labs and the Uniswap Foundation on **10 November 2025**. It passed in December 2025 with roughly **125 million UNI in favour against 742 opposed**, per CoinDesk's reporting, then entered a two-day timelock. Its components:

- A **one-time burn of 100 million UNI from the treasury** — reported at roughly \$940 million at then-prevailing prices — sized as an estimate of what would have been burned had the protocol fee been active since launch.
- Activation of the **protocol fee switch**: for v2 pools, liquidity providers keep 0.25% of the 0.30% swap fee and the protocol retains **0.05%**, directed to burns. For v3, protocol fees are set at one quarter of LP fees for the 0.01% and 0.05% tiers and one sixth for the 0.30% and 1% tiers.
- A structural change to the Foundation itself: most Uniswap Foundation employees transition to Uniswap Labs, leaving a small grants-focused team behind.

### Worked example 6: the burn as sign-flipped treasury policy

**Step 1 — the one-time burn as a share of supply.** UNI launched with a genesis supply of 1 billion tokens:

100,000,000 ÷ 1,000,000,000 = **10% of the genesis supply, retired permanently**

**Step 2 — the implied price.** A reported value of roughly \$940 million for 100 million UNI implies:

\$940M ÷ 100M UNI = **~\$9.40 per UNI**

**Step 3 — the run-rate.** Reporting estimated the fee switch could direct roughly **\$130 million a year** into burns at then-current volumes:

\$130M ÷ \$9.40 = **~13.8 million UNI per year**
13.8M ÷ 1,000,000,000 = **~1.4% of genesis supply retired annually**

**Step 4 — the symmetry.** Compare that with a foundation selling \$130 million of its token a year to fund operations. Identical magnitude, identical mechanism, opposite sign. One adds roughly 1.4% of supply to the float annually and produces research; the other removes roughly 1.4% annually and produces nothing but scarcity.

> **The intuition:** a burn is a distribution to every holder, paid in scarcity instead of cash. A treasury sale is a capital raise, paid for by every holder through dilution of the float. Which one a protocol can afford depends entirely on whether it has revenue — and only a handful do.

That last clause is the part that gets lost in burn enthusiasm. A protocol can only burn what it earns. A foundation sells precisely *because* it earns nothing. The move from "sell tokens to fund development" to "burn tokens funded by fees" is not a change of policy preference; it is a change of business model, and it is only available to protocols that have found one. The Uniswap Foundation's simultaneous shrinkage is the tell: when the protocol develops revenue, the foundation's reason to hold a large token treasury weakens.

## The analytical core: why these really are crypto's central banks

We can now state the thesis precisely.

A **central bank** — to define the term, since this post assumes no background — is an institution that holds a very large balance sheet in the asset it also governs, that acts for policy reasons rather than profit, that publishes a reaction function so markets can anticipate it, and whose announcements move prices more than its transactions do.

Token foundations and DAO treasuries match that description on four of five counts:

| Property | Central bank | Token foundation / DAO treasury |
|---|---|---|
| Very large holder of the asset it governs | Yes | Yes — often the single largest identifiable holder |
| Motivated by policy, not profit | Yes | Yes — funds a purpose, has no P&L |
| Permanent horizon, no redemption risk | Yes | Yes — cannot be redeemed or margin-called |
| Publishes a reaction function | Yes | Increasingly yes — the EF's 15%/2.5-year rule is exactly this |
| Legal monopoly and lender-of-last-resort role | Yes | **No** — this is where the analogy stops |

The last row matters. A foundation cannot create tokens, cannot set an interest rate, cannot backstop anyone, and has no legal authority over anything. Do not stretch the metaphor into a claim that foundations control token prices. They do not.

But the first four rows are enough to generate something genuinely useful, and here is why reading these wallets is tractable edge rather than noise:

**1. They telegraph.** A published treasury policy is a forward supply schedule. Once you know that a foundation targets a 2.5-year fiat buffer and spends 15% of treasury annually, you can compute — not guess — the conditions under which it must sell. That is a better forward-supply model than exists for almost any other holder class.

**2. They are slow.** Governance timelocks, quarterly review cycles, multisig coordination and quorum failure mean these entities move on a timescale of months. You are never racing them. You have time to read the forum post, do the arithmetic, and act.

**3. They are labelled.** Not perfectly, but far better than any other large holder. A hedge fund's positions are invisible; a foundation's are on a block explorer with a name attached.

**4. Their constraints are arithmetic.** They cannot spend what they do not have, cannot sell faster than liquidity allows, and cannot escape the correlation between their asset and their funding need. Everything in this post has been arithmetic, not psychology — which is precisely why it is forecastable.

**5. The market misreads them predictably.** The 0.3%-of-volume deposit that gets called a dump; the \$2 billion treasury that is really \$1.5 billion; the grants programme that panics a chat group and never touches the tape. Predictable misreading by others is the only durable source of edge there is.

What this does *not* give you is a price signal. It gives you a **supply model** and a **narrative-risk model** — you know roughly how much token will reach the float over the next year, and you know which announcements will be misinterpreted. How that translates into price runs through the machinery described in [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move): thin float, reflexive positioning, and the fact that in crypto, information about flow moves price far more than the flow itself does.

## Common misconceptions

**"A transfer to an exchange is a sale."**
It is one of at least four things: a sale, an OTC settlement, a custody rotation, or collateral posting. Exchange deposit addresses are used for all of them. The sale inference becomes reasonable only when you observe the return leg — stablecoins coming back to the entity's wallets — and even then it is an inference. Treat "deposited to exchange" as *inventory positioned for possible sale*, not as a completed transaction.

**"A \$2 billion DAO treasury means \$2 billion of buying power."**
It means \$2 billion at the last printed price, on the assumption that a buyer exists for the whole position at that price. For a treasury that is 90% its own token and 30%+ of its own float, realistic realisable value after a disciplined liquidation is materially lower — our worked example produced a 26% haircut on generous assumptions. Read every treasury number as "own-token slice, at risk of a large haircut" plus "stablecoin slice, real."

**"The foundation dumping is why the price fell."**
Check the denominator before you accept this. A \$42.7 million sale into a \$15 billion daily market is 0.3% of volume, and worked over five sessions it is under 0.06%. Flows of that size are mechanically undetectable. What *can* move price is the narrative around a sale — which is a statement about other traders' reactions, not about the seller's impact.

**"Token holders control the treasury."**
For a Swiss Stiftung, no — the vote is advisory and the board answers to a supervisory authority. For a Cayman foundation company, only insofar as the by-laws bind the directors, and the by-laws are often private. For an unwrapped DAO, the vote binds on-chain and nothing off-chain. Arbitrum's AIP-1 was the public demonstration: the tokens had already moved before the vote closed, and 78% opposition changed the process but not the underlying legal reality.

**"Selling the token is a betrayal of the ecosystem."**
Selling is the operating model. A foundation with no revenue and one asset either sells that asset or funds nothing. The legitimate criticisms are about *how*: whether the schedule is published in advance, whether execution minimises impact, whether the spending rule is disciplined, and whether the entity's dual role as steward and seller is disclosed. "They sold" is not a criticism; "they sold without a published rule" is.

**"A burn creates value out of nothing."**
A burn is funded by revenue, and revenue comes from users paying fees. It is a distribution of real earnings, paid in scarcity rather than cash. If a protocol burns tokens it holds but did not earn — a treasury burn rather than a fee burn — it has changed the supply schedule and destroyed its own future funding capacity, which may be a good trade or a bad one, but is definitely not free.

## How it shows up in real markets

**The Ethereum Foundation's drawdown, 2022–2024.** The treasury fell from \$1.6 billion (31 March 2022) to \$970.2 million (31 October 2024), a 39% decline the Foundation attributed to roughly \$240 million of spending in 2022–23 alongside an ETH decline of about 22%. This is the mechanism from Worked Example 1 running in the wild: an unruled spending programme against a volatile asset compounds two headwinds. The response — the June 2025 treasury policy with its 15% cap and 2.5-year buffer — is the institutional lesson every endowment eventually learns, arriving in crypto about a decade after the asset class needed it.

**The Foundation's DeFi turn, 2025–2026.** Forty-five thousand ETH into Spark, Aave and Compound in February 2025; \$2 million of GHO borrowed against the position in May 2025; Morpho vaults added in March 2026; a published "defipunk" standard governing which protocols qualify. Read as treasury management, it is a modest yield programme. Read as policy, it is a large institution putting its own balance sheet behind a specific vision of what DeFi should be — permissionless, self-custodial, open-source, private. When an entity of this size deploys, the deployment is an endorsement, and endorsements move capital.

**Arbitrum's AIP-1, March–April 2023.** Seven hundred and fifty million ARB, moved before the vote concluded; 78% opposition; 40 million ARB lent to a market-making counterparty; 10 million already converted to fiat. The Foundation retreated, split the proposal and pledged no near-term sales. The market lesson was not "foundations are bad." It was that **the gap between advisory governance and legal control is invisible until it is tested**, and it is always tested on the first large allocation.

**ENS's endowment, from March 2023.** ENS did the boring, correct thing: appointed a professional manager (karpatkey), seeded an endowment with 16,000 ETH, published monthly reports, and set out to fund operations from investment returns rather than token sales. Roughly \$2.92 million of net DeFi results since inception, contributing about 12% of ENS's 2024 revenue. It is the most institutionally mature treasury in crypto, and its honest lesson is how *modest* the contribution is. Good treasury management buys you a tenth of your budget, not your budget.

**Uniswap's UNIfication, November–December 2025.** One hundred million UNI burned — 10% of genesis supply — a fee switch activated at 0.05% of the 0.30% v2 swap fee, and the Foundation's staff largely folded into Labs. This is what the end state looks like when a protocol develops real revenue: the treasury stops being the funding mechanism, fees take over, and the foundation shrinks toward a grants desk. Whether \$130 million a year of burns is a good use of that revenue versus funding development is a genuine debate; the structural shift is not in doubt.

**The Tezos Foundation, 2017–2018.** A \$232 million raise, a Swiss foundation controlling the funds, the creators controlling the code, and a compensation dispute that froze the project for months. The people who sent the money had no mechanism to intervene. It ended with board turnover — Johann Gevers stepping down in February 2018, replaced by Ryan Jesperson — because board turnover was the only mechanism that existed. Every foundation structure since has been designed with this episode in mind, and none has eliminated the underlying issue.

**The Solana Foundation's delegation programme, from November 2020.** One hundred million SOL — over 80% of the Foundation's treasury at the time — algorithmically delegated to maximise the number of independent validators making up 33% of stake. Zero of it motivated by yield. This is the purest example of a treasury used as a policy tool: the Foundation spent its balance-sheet capacity to buy decentralisation, which is not an asset that appears on any dashboard.

## A practical monitoring checklist

Here is the routine, and — equally important — the list of things that look like signals and are not.

![A three-column grid listing what to watch, what each source tells you, and what it does not tell you, across labelled addresses, exchange deposits, treasury policies, governance forums and treasury dashboards](/imgs/blogs/token-foundations-and-treasuries-the-on-chain-central-banks-10.webp)

### What to watch

**Addresses.** Bookmark the entity pages, not individual addresses: Arkham's entity view, Nansen's labels, and the Etherscan label for the addresses you care about (`0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe`, labelled "EthDev", is the canonical Ethereum example). Watch *flows*, and remember the labelled set is a lower bound on holdings.

**Reports.** The primary documents are worth more than all the on-chain analysis combined, and almost nobody reads them: foundation annual and treasury reports, published treasury policies, and professional-manager reports such as karpatkey's monthly ENS endowment updates. A treasury policy is a forward supply schedule written by the seller.

**Governance.** Forums and voting front-ends — Snapshot, Tally, Agora and each DAO's own discussion board — show committed future spending weeks or months before any token moves. A passed proposal with a timelock is the highest-confidence supply signal available in crypto, because the execution is contractually scheduled.

**Dashboards.** DeepDAO and DefiLlama's treasury pages for composition and own-token share. Use them for *what a treasury is made of*, never for *what it is worth*.

### The monthly routine

1. **Composition check.** For each treasury you track: what share is own-token, what share is stablecoin, what share is other-crypto? Only the stablecoin slice is reliably spendable. Recompute the own-token slice as a percentage of float, not of supply.
2. **Policy check.** Has a treasury policy been published or amended? Compute the implied sale requirement at current prices, as in Worked Example 1. This is your forward supply estimate.
3. **Governance queue.** What has passed and is sitting in a timelock? What is in temperature check? Committed spend is the only supply you can date with confidence.
4. **Flow check.** Any exchange deposits since last month? Size them **as a percentage of daily volume** before forming any view. If it is under 1%, it is invisible to price and interesting only as balance-sheet information.
5. **Runway check.** Treasury value divided by last disclosed annual spend. When runway drops below roughly three years, expect either a selling programme or a spending cut. Both are forecastable months ahead.

### Signals that mean nothing

- **A single large transfer between an entity's own wallets.** Custody rotation is routine and produces alarming headlines. Check whether the destination has prior history with the entity.
- **A treasury's headline dollar value rising or falling.** If it is denominated in its own token, that number is just the token price wearing a costume. It contains no information about the treasury's decisions.
- **"Foundation wallet activity detected" alerts.** Foundations pay grants, rotate custody, rebalance and test contracts constantly. Activity is not signal; *size relative to volume* is signal.
- **A grants programme's total budget.** Meaningless without the daily-volume denominator and the disbursement schedule. Worked Example 4 showed a flagship annual programme landing at 0.26% of daily volume.
- **A burn announcement, absent revenue.** A burn funded by treasury tokens changes the supply schedule and reduces future funding capacity. A burn funded by fees is a distribution of earnings. They look identical in a headline and are entirely different in substance.

## When this matters to you

If you hold a token, you are a residual claimant on a supply schedule that a foundation or a DAO treasury partly controls. You do not get a vote that binds them. What you do get — and this is genuinely unusual in finance — is the ability to read their balance sheet, their policy and their transactions directly, before anyone tells you what they mean.

That access is worth using properly. The discipline is small: find the denominator before you form a view, distinguish stock from flow, read the treasury policy rather than the tweet about it, and never let a number denominated in the token be mistaken for a number denominated in money.

None of this is investment advice, and none of it predicts price. It predicts *supply*, which is a smaller claim and a much more defensible one.

For the next layer, [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) maps how foundations sit alongside the other entities that shape a token's life, and [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) covers the two holder classes whose motives are the exact opposite of a foundation's — which is precisely why the contrast is instructive.

## Sources & further reading

**Primary — foundation and protocol documents**

- Ethereum Foundation, ["Ethereum Foundation Treasury Policy"](https://blog.ethereum.org/2025/06/04/ef-treasury-policy), 4 June 2025 — the 15% opex cap, the 2.5-year buffer, the five-year glide to 5%, the sale mechanism and the "defipunk" deployment criteria.
- Ethereum Foundation Report, April 2022 — treasury of \$1.6bn as of 31 March 2022, 80.5% ETH, 0.297% of ETH supply, ~\$48m of 2021 spending.
- Ethereum Foundation Report, November 2024 — treasury of \$970.2m as of 31 October 2024; \$788.7m crypto (99.45% ETH), \$181.5m non-crypto; 2022 spend \$105.4m, 2023 spend \$134.9m.
- Uniswap, ["UNIfication"](https://blog.uniswap.org/unification), 10 November 2025 — the 100m UNI burn, the v2 and v3 protocol-fee parameters, and the Foundation/Labs restructuring.
- Solana Foundation, ["Announcing the Solana Foundation Delegation Strategy"](https://solana.com/news/announcing-the-solana-foundation-delegation-strategy), November 2020 — the 100,000,000 SOL delegation, described as over 80% of the Foundation's treasury.
- ENS DAO governance forum and karpatkey endowment monthly reports — endowment initiation 7 March 2023 with a 16,000 ETH first tranche; DeFi results and their share of ENS revenue.
- Arbitrum DAO governance (Tally) — the Stable Treasury Endowment Program allocating 35m ARB to tokenised Treasuries, and STEP 2.0.

**Securities and financial press**

- CoinDesk, ["Ethereum Foundation's Treasury Shrunk 39% Over 2 1/2 Years to \$970M"](https://www.coindesk.com/tech/2024/11/08/ethereum-foundations-treasury-shrunk-39-over-2-12-years-to-970m), 8 November 2024.
- CoinDesk, ["Ethereum Foundation Unveils New Treasury Policy With 15% Opex Cap"](https://www.coindesk.com/tech/2025/06/05/ethereum-foundation-unveils-new-treasury-policy-with-15-opex-cap), 5 June 2025.
- CoinDesk, ["Ethereum Foundation stakes another \$93 million ether, reaching its 70,000 ETH target"](https://www.coindesk.com/markets/2026/04/03/ethereum-foundation-stakes-another-usd93-million-ether-reaching-its-70-000-eth-target), 3 April 2026 — the staking target, the ~\$100m annual expense figure and the \$3.9–5.4m estimated annual yield.
- The Block, ["Ethereum Foundation reports crypto holdings of \$788M"](https://www.theblock.co/post/325166/ethereum-foundation-holds-788-million-in-crypto), November 2024 — including the 0.26%-of-supply figure.
- The Block, ["Ethereum Foundation begins staking part of ether treasury"](https://www.theblock.co/post/390993/ethereum-foundation-begins-staking-part-of-ether-treasury-plans-to-deploy-about-70000-eth-to-generate-yield), February 2026.
- The Block, ["Arbitrum DAO approves 35 million ARB allocation to tokenized US Treasurys"](https://www.theblock.co/post/353631/arbitrum-dao-us-treasurys).
- Cointelegraph, ["Ethereum Foundation deploys \$120M to DeFi apps"](https://cointelegraph.com/news/ethereum-foundation-120-million-aave-spark-compound), February 2025 — the 45,000 ETH split across Spark, Aave Prime, Aave Core and Compound.
- CoinDesk, ["Arbitrum Foundation Scraps Vote, Pledges Redo After ARB Tokenholders Revolt"](https://www.coindesk.com/business/2023/04/02/arbitrum-foundation-scraps-vote-pledges-redo-after-arb-tokenholders-revolt), 2 April 2023; and Blockworks, ["Arbitrum Walks Back \$1B Proposal — But It Already Used Some of It"](https://blockworks.co/news/arbitrum-walks-back-proposal).
- CoinDesk, ["Uniswap token burn moves closer to reality as 99% of voters back the fee switch"](https://www.coindesk.com/markets/2025/12/22/uniswap-token-burn-moves-closer-to-reality-as-99-of-voters-in-favor-of-fee-switch-proposal), December 2025; and DL News on the estimated annual fee run-rate.
- SWI swissinfo.ch and Reuters coverage of the Tezos Foundation dispute, 2017–2018, including Gevers' February 2018 departure.

**Data and tooling**

- [Etherscan](https://etherscan.io) — address labels, including "EthDev" for `0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe`.
- [Arkham Intelligence](https://intel.arkm.com/explorer/entity/ethereum-foundation) — the Ethereum Foundation entity page and its 2026 top-ETH-holders research.
- Nansen — wallet labelling and entity flows.
- [DeepDAO](https://deepdao.io) — DAO treasury aggregates; the \$25.1bn total and the largest-treasury table cited here are its 31 March 2023 snapshot.
- DefiLlama treasury dashboards, and Optimism's own ["2024 year in review"](https://www.optimism.io/blog/optimism-2024-year-in-review) and Retro Funding posts for the 850m OP allocation and the 2024 distribution of 20.4m OP across three rounds to 374 projects.

**On the legal structures**

- Swiss Civil Code, Article 80 et seq. (ZGB) — the *Stiftung*; and Swiss cantonal/federal foundation supervisory authority guidance.
- Cayman Islands Foundation Companies Act — the ownerless foundation company, the supervisor role, and its use as a DAO wrapper (Mourant, Carey Olsen and Maples client briefings are the standard practitioner references).

*This post is educational and is not investment advice. Every dated figure is attributed to its source above; balances and prices move, so verify current values before relying on any number here.*
