---
title: "Following Token Flows From Insiders to Exit Liquidity"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "A defensive, hop-by-hop method for tracing treasury tokens, market-maker loans, VC unlocks, OTC transfers, exchange deposits, and the retail liquidity that absorbs them."
tags: ["crypto", "on-chain-analysis", "token-flows", "market-makers", "token-unlocks", "venture-capital", "exchange-flows", "retail-defense", "crypto-players"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 28
---

> [!important]
> **TL;DR** — On-chain forensics is a custody map, not a crystal ball: follow the token from a treasury, investor, or market-maker wallet to a known exchange deposit, then compare the amount with float, unlocks, and actual liquidity.
>
> - A transfer to a centralized exchange (CEX) is observable positioning; the sale itself usually happens inside the exchange's private order book.
> - “This wallet belongs to a fund” is an attribution hypothesis. Etherscan proves a transaction; Arkham and Nansen add labels; Dune lets you aggregate; none proves motive by itself.
> - A market-maker loan and a VC unlock are different supply events. The first can be inventory management; the second is a change in selling rights. Do not collapse them into “insider selling.”
> - A defensible thesis has four timestamps: unlock or loan date, source transfer, exchange deposit, and the market's response.
> - In the SEC's April 2023 Terraform complaint, LUNA was described as moving from under $1 in early 2021 to around $119.18 in April 2022 before falling below a penny in May 2022; that is a dated case record, not a prediction. [SEC complaint](https://www.sec.gov/files/terraform-labs-pte-ltd-amended-complaint.pdf)

The most useful on-chain question is not “who is smart money?” It is narrower: **which holder acquired the right to sell, where did the tokens go next, and what could the market absorb?** That question turns a dramatic wallet screenshot into a sequence you can audit.

The distinction matters because crypto has two ledgers. The blockchain records token custody changes. A CEX records matching-engine fills in its own database. When a wallet sends tokens to a Binance, Coinbase, or OKX deposit cluster, you can observe a likely preparation for trading. You cannot observe the exact customer, side of every fill, or whether the exchange later swept those tokens into a cold wallet without additional evidence.

This article builds a forensic workflow around that limitation. It is not a manual for spoofing, wash trading, evading detection, or coordinating a dump. It is a defensive reading method for a reader deciding whether a visible rally is meeting organic demand or a predictable supply release.

![The public money map runs from a project treasury through a market-maker loan to a known CEX deposit; the final retail fill occurs inside the venue's private ledger.](/imgs/blogs/following-token-flows-from-insiders-to-exit-liquidity-1.webp)

The diagram's arrows are deliberately modest. “Treasury → market maker → CEX” does not prove “market maker sold.” It says that custody moved along a path that makes selling possible. The rest of the post is about adding enough independent evidence to distinguish possibility from pressure.

## Foundations: the building blocks

### Token supply, float, and the right to sell

**Total supply** is the number of tokens currently created or intended under a project's stated supply policy. **Circulating supply** is the portion a data provider counts as available to the public. **Float** is the more practical idea: tokens that can actually meet a buyer now, excluding tokens locked by a vesting contract, held in a treasury, or otherwise not available for ordinary trading. Providers can disagree about what “circulating” means, so treat the number as a dated estimate unless the issuer and chain data reconcile.

**Fully diluted valuation (FDV)** is price multiplied by total supply. Market capitalization is price multiplied by circulating supply. Neither is cash in a bank account. Both are a price multiplied by a supply convention. If a token trades at $2, has 10 million circulating tokens, and a stated total supply of 100 million, its market cap is $20 million and its FDV is $200 million. The $180 million gap is not a guaranteed future sale; it is a claim on future supply at today's price.

### Wallets and transfers

A **wallet address** is a public account identifier. A **transfer** is a token balance change recorded by a smart contract event. A **cluster** is a group of addresses an analytics provider believes is controlled by one entity. A cluster is useful because large organizations rarely use one address forever, but the label remains an inference unless the entity or a strong primary source confirms it.

An **EOA** is an externally owned account controlled by a private key. A **smart contract** can hold tokens and release them according to code. A vesting contract is stronger evidence of a time lock than an ordinary wallet described as “locked” in a slide deck. A **CEX deposit address** is an exchange-controlled address that credits a user's internal account. The address is public; the internal account and subsequent fills generally are not.

### Market makers, loans, and OTC

A **market maker (MM)** quotes bids and asks, holding inventory so buyers and sellers can trade. A **token loan** gives the MM tokens that it promises to return, often with separate compensation. The loan can be used to make a market, hedge, lend onward, or sell and later repurchase; the chain alone does not tell you which. The obligation to return the same token amount creates economic exposure, but it is not proof of abusive intent.

**OTC**, or over-the-counter, is a negotiated transfer away from the visible exchange order book. An investor may sell tokens to another buyer under a contract. That buyer may later deposit them on an exchange. Looking only at the final deposit can falsely assign the sale to the original investor. “Source → intermediary → venue” is therefore more useful than a one-wallet blacklist.

### The four questions every hop should answer

1. **Who controlled the source?** Issuer, treasury, investor, MM, vesting contract, bridge, or unknown?
2. **What changed?** Unlock, loan, transfer, sale, repayment, bridge movement, or exchange sweep?
3. **Where did it go?** Another internal wallet, OTC counterparty, DEX pool, CEX deposit, or burn address?
4. **What did price and volume do afterward?** A deposit is a risk signal; a market reaction is a separate observation.

#### Worked example 1: market cap versus future supply

Suppose a hypothetical token trades at **$2**. There are **10,000,000** circulating tokens and **100,000,000** total tokens.

1. Market cap = $2 × 10,000,000 = **$20,000,000**.
2. FDV = $2 × 100,000,000 = **$200,000,000**.
3. Locked or not-yet-circulating supply = 100,000,000 − 10,000,000 = **90,000,000 tokens**.
4. If only 1,000,000 tokens arrive at an exchange, that is 10% of the current float (1,000,000 ÷ 10,000,000), not 1% of total supply.

The intuition: pressure is measured against the supply that actually trades, not the larger supply number that makes the FDV look tidy.

## 1. Start with the document, then find the wallet

The fastest way to make an on-chain investigation unreliable is to begin with a wallet screenshot. Begin with the project's tokenomics page, token-generation announcement, vesting contract, investor disclosure, and market-maker agreement if one is public. Build a table containing allocation bucket, recipient type, unlock rule, custody mechanism, and confidence.

The relevant distinction is **right to sell** versus **intention to sell**. A cliff date can establish that a VC allocation became transferable. It cannot establish that the fund sold. A transaction from a treasury to a market maker can establish custody movement. It cannot establish whether the MM sold, hedged, or returned the same tokens.

Use Etherscan or the relevant chain explorer to verify the transaction hash, token contract, block timestamp, amount, sender, recipient, and whether the event is a direct transfer or a contract interaction. Save the link and the chain name. Do not rely on a cropped image where the token contract is hidden; fake-token homonyms and wrapped assets make symbol-only searches dangerous.

![The forensic workflow branches from a question into transaction evidence, attribution, unlock rights, destination evidence, and market context before reaching a conditional thesis.](/imgs/blogs/following-token-flows-from-insiders-to-exit-liquidity-2.webp)

### Why attribution is probabilistic

Arkham and Nansen can be excellent starting points for labels and alerts. Their labels may come from public disclosures, known exchange infrastructure, behavioral clustering, or research. The correct language is “labeled as,” “attributed by,” or “consistent with,” unless there is a primary confirmation.

Bubblemaps is useful for visualizing connected wallets and transfers, especially when an allocation is split into many addresses. A cluster is not a conviction. Related wallets can be service providers, custodians, treasury subaccounts, or unrelated users who interacted with the same contract. The visual is a lead generator.

Dune is useful when you need a repeatable query: daily net transfers to known exchange clusters, token movements from a set of vesting contracts, or the balance of a labeled address over time. The query's assumptions matter. If the destination labels are incomplete, the output is incomplete. DefiLlama supplies protocol and liquidity context, but its TVL is not the same thing as spot order-book depth.

#### Worked example 2: turning a transfer into a ratio

Imagine a hypothetical token with **50,000,000** circulating units. A labeled treasury sends **2,000,000** units to a market-maker wallet. Two days later, the MM sends **1,200,000** units to a known CEX deposit cluster. The token's reported spot volume for that day is **$3,000,000**, and the token is trading at **$0.50**.

1. Treasury-to-MM transfer = 2,000,000 ÷ 50,000,000 = **4% of float**.
2. CEX deposit = 1,200,000 × $0.50 = **$600,000 of token value at the observed price**.
3. Deposit relative to reported daily dollar volume = $600,000 ÷ $3,000,000 = **20%**.
4. The correct conclusion is “a potentially material deposit relative to reported volume,” not “the MM dumped $600,000.” Execution may occur later, across venues, or not at all.

The intuition: ratios tell you why a hop matters; they do not upgrade an observable transfer into an unobserved fill.

## 2. Separate treasury → MM loan from investor → OTC → exchange

The brief visual story “insider sells to retail” hides at least two different paths.

In the first, a project treasury lends tokens to an MM. The MM receives inventory to quote with and may return the same quantity at maturity. The loan can sit in the MM's wallet, move to a subaccount, be posted to a venue, or be used in hedging. A deposit is a change in venue custody, not an invoice showing the MM's P&L.

In the second, a VC allocation unlocks. The investor may transfer tokens to a custodian, sell OTC to a market participant, split the tokens across addresses, or deposit directly at a CEX. The buyer now owns the tokens and may be the party that eventually sells to the public. The original VC's unlock is evidence of available supply; it is not evidence of a completed sale.

The key forensic test is the **intermediate hop**. If the investor wallet sends to a new wallet and that wallet immediately sends to a CEX, the story is stronger than if the investor wallet sends to a known custody provider. If the tokens move to a market maker and later return to the treasury, the loan interpretation gains support. If they go to a CEX and the balance stays there, the evidence remains ambiguous.

![A CEX deposit can receive treasury, unlocked VC, market-maker, airdrop, or OTC-buyer supply; the destination alone cannot identify the seller.](/imgs/blogs/following-token-flows-from-insiders-to-exit-liquidity-5.webp)

#### Worked example 3: an unlock is not the same as a sale

Suppose an investor has **5,000,000** tokens subject to a cliff. On unlock day, **5,000,000** become transferable. The investor moves **3,000,000** to a custody wallet and leaves **2,000,000** in the vesting contract's recipient address.

1. Transferable supply increases by 5,000,000, even though only 3,000,000 moved.
2. The visible new CEX supply is **0** at that moment if no exchange deposit occurred.
3. If 3,000,000 later reach a CEX, the CEX-facing amount is 3,000,000, while the total newly sellable allocation remains 5,000,000.
4. If the token's float before unlock was **25,000,000**, the unlocked allocation is 5,000,000 ÷ 25,000,000 = **20% of the pre-unlock float**.

The intuition: an unlock changes optionality; a venue deposit changes location; only a fill changes executed ownership.

## 3. Read time, amount, and destination together

Single-hop analysis fails because blockchains contain many innocent-looking movements. A treasury may distribute grants. An exchange may sweep deposits to cold storage. An MM may rebalance between chains. A bridge may create a burn-and-mint pair that looks like a sale if you inspect only one side.

Build a hop ledger with one row per transaction:

| Timestamp | Source label | Destination label | Amount | Supply status | What it proves |
|---|---|---|---:|---|---|
| block time | vesting contract | investor wallet | token amount | unlocked | transfer right changed |
| block time | investor wallet | custody wallet | token amount | transferable | custody changed |
| block time | custody wallet | CEX deposit | token amount | venue-facing | deposit occurred |
| later | CEX internal book | unknown buyer | not public | executed | requires venue data |

The last row is intentionally not visible on-chain. Do not fill it with a guess.

#### Worked example 4: timing windows and alternative explanations

Consider a hypothetical **400,000-token** CEX deposit after a **1,000,000-token** unlock. The deposit occurs **two days** after the unlock, while price falls from **$1.20** to **$1.00** over that window. Daily reported volume is **$2,000,000** at the time, and the deposit's notional at the starting price is **$480,000**.

1. Deposit share of the unlocked amount = 400,000 ÷ 1,000,000 = **40%**.
2. Deposit notional at the starting price = 400,000 × $1.20 = **$480,000**.
3. Deposit-to-volume ratio = $480,000 ÷ $2,000,000 = **24%**.
4. The two-day price change is ($1.00 − $1.20) ÷ $1.20 = **−16.67%**, rounded to two decimals.

This is evidence consistent with selling pressure, but three alternatives remain: the deposit may be collateral, the venue may have swept it without a sale, or broader market risk may explain the decline. A cautious report records all three and looks for repeated deposits, net exchange balances, and order-book response.

The intuition: correlation becomes useful only when you preserve the competing explanations.

## 4. Estimate absorption without pretending to know hidden liquidity

The market's capacity to absorb tokens is not the same as a site's 24-hour volume number. Volume can include multiple counting conventions, derivatives, wash trades, or activity on venues that cannot accept the token deposit. Order-book depth is dynamic. A market can print $10 million of volume and still move sharply when a concentrated seller arrives.

Use several imperfect views:

- **Reported spot volume**, dated and named by provider, is a coarse turnover denominator.
- **Visible depth** near the best bid estimates how much can be sold before the quoted price moves, but it changes as orders are canceled.
- **CEX deposit flows** show venue-facing supply, not fills.
- **DEX pool liquidity** is not equivalent to executable depth at a fixed price because an AMM's reserves move along a curve.
- **DefiLlama liquidity and TVL** are context, not proof that a token can absorb an unlock without slippage.

![The tool matrix separates transaction proof, attribution, custom aggregation, and market context; each tool sees only one layer of the map.](/imgs/blogs/following-token-flows-from-insiders-to-exit-liquidity-4.webp)

#### Worked example 5: a simple absorption stress test

Suppose a hypothetical token has **$5,000,000** of usable spot volume over a dated day. A source cluster deposits **$750,000** worth of tokens. Assume, only for this illustration, that the seller's execution is spread evenly across the day's volume.

1. Deposit-to-volume ratio = $750,000 ÷ $5,000,000 = **15%**.
2. If only half of reported volume is genuine two-way liquidity available to absorb this seller, the assumed absorption base is $5,000,000 × 50% = **$2,500,000**.
3. The deposit is then $750,000 ÷ $2,500,000 = **30%** of that assumed base.
4. A 30% share does not forecast a 30% price fall. It says the supply is large enough to deserve stress testing rather than a casual “volume is high” dismissal.

The intuition: treat volume as a denominator with assumptions, not a magical shield.

## 5. The numbers that make a money map honest

### Cost basis is a clue, not a motive

Private-round price can help explain why a holder may be economically able to sell. A token bought at a lower price has more room before the sale becomes unprofitable. But cost basis does not reveal a fund's liquidity needs, lock agreement, hedging, tax position, or mandate. Never convert “large unrealized gain” into “will sell.”

### Market cap and FDV can conceal the same supply twice

If a wallet's tokens are already counted in circulating supply, do not add them to an “unlocked overhang” number again. If a treasury is excluded from circulating supply, document that convention before computing the percentage. A sound report names its denominator every time: total supply, circulating supply, free float, or daily volume.

### Loans can be economically important without being net selling

Loaned tokens may be used to quote asks and bids. The MM can have short inventory, long options, stablecoin hedges, or offsetting positions elsewhere. The project may require return of the same token quantity. A reader should ask for contract terms: amount, term, permitted use, collateral, return mechanics, and whether options or fees exist. Publicly available information may be incomplete.

### Why the exchange boundary matters

Before the CEX boundary, you can inspect token transfers. After it, you need exchange data, filings, or a later withdrawal trail. The internal account may net several users. A deposit can be sold, borrowed, pledged, or swept. The correct label is “exchange-facing supply,” not “retail sold to.”

![The timeline separates a 1,000,000-token unlock, a 400,000-token CEX deposit, and the later private execution question.](/imgs/blogs/following-token-flows-from-insiders-to-exit-liquidity-3.webp)

## 6. A worked forensic example: ACME, a deliberately hypothetical token

To make the workflow concrete, consider ACME. This is a fictional case study for arithmetic, not a claim about a real project.

ACME has **1,000,000,000** total tokens and **230,000,000** circulating at launch. A published allocation assigns **180,000,000** to team, **220,000,000** to investors, **250,000,000** to treasury, **120,000,000** to ecosystem rewards, **150,000,000** to community, and **80,000,000** to liquidity. The bucket totals equal 1,000,000,000.

The team and investors together hold 400,000,000 tokens, or 400,000,000 ÷ 1,000,000,000 = **40% of total supply**. The community and liquidity buckets sum to 230,000,000, matching the stated launch float. The first observation is therefore structural: insiders have a future claim larger than the current float.

At month 12, **50,000,000** investor tokens unlock. The allocation is now transferable, but nothing has reached an exchange. On day 2, **12,000,000** move to a new wallet; on day 4, **8,000,000** move to a labeled custody cluster; on day 7, **5,000,000** reach a known CEX deposit cluster.

1. The unlock is 50,000,000 ÷ 230,000,000 = **21.74% of the launch float**.
2. The first wallet move is 12,000,000 ÷ 50,000,000 = **24% of the unlocked tranche**.
3. The CEX-facing amount is 5,000,000 ÷ 230,000,000 = **2.17% of the original float**.
4. If ACME trades at a hypothetical **$0.40**, the CEX-facing notional is 5,000,000 × $0.40 = **$2,000,000**.
5. If dated spot volume is **$8,000,000**, that notional is $2,000,000 ÷ $8,000,000 = **25% of reported volume**.

The conclusion is conditional: “An investor-linked cluster became transferable, moved through custody, and placed 5,000,000 ACME at a known CEX deposit. At $0.40, that is $2,000,000 or 25% of the dated reported spot volume. The chain does not prove how much was sold.” That sentence is useful. “The VC dumped on retail” is stronger than the evidence.

![The defensive upgrade replaces a chart-first reaction with a cap-table, unlock, wallet, liquidity, and invalidation checklist.](/imgs/blogs/following-token-flows-from-insiders-to-exit-liquidity-7.webp)

## 7. What a professional investigation keeps in its notebook

The difference between a useful investigation and a thread of screenshots is reproducibility. A second analyst should be able to start with the same token contract, repeat the query, and understand where judgment entered the chain of reasoning.

### Preserve the raw evidence

Save the token contract address, chain, transaction hash, block number, UTC timestamp, sender, recipient, token amount, and the explorer URL. If a dashboard supplies a label, save the dashboard URL and the date observed. Labels change. An address once called “fund wallet” can later be reclassified as a custodian, bridge, or exchange sweep. Your note should preserve both the original label and the current label.

Do not round the raw token amount before analysis. Store the integer token units and the token's decimals. A display reading of 1.25 can represent a different raw quantity on a token with six decimals than on one with eighteen. The calculation layer can round for readability; the evidence layer should not.

Record time zones explicitly. Block timestamps are generally expressed as Unix time and displayed by a site in a chosen time zone. An unlock at 00:00 UTC can appear on the prior local calendar date. A two-day window should mean two stated calendar intervals, not “a little while later.”

### Track confidence in layers

Use a simple confidence vocabulary:

- **Confirmed by chain:** the token transfer, contract call, or balance change is directly verifiable.
- **Issuer-stated:** the project or legal filing identifies an address, allocation, or schedule.
- **Vendor-attributed:** Arkham, Nansen, Bubblemaps, or another analytics provider labels a cluster.
- **Consistent with:** timing and amount fit a proposed explanation but alternatives remain.
- **Unknown:** the data cannot distinguish the alternatives.

This vocabulary prevents a common rhetorical slide. “The wallet received tokens from the vesting contract” is confirmed by chain. “The wallet belongs to Fund X” may be vendor-attributed. “Fund X sold to retail” may be unknown. Each sentence can be true at its own confidence level without the first two magically proving the third.

![A raw transaction becomes a cautious thesis only after source, timing, rights, destination, market response, and alternative explanations are recorded separately.](/imgs/blogs/following-token-flows-from-insiders-to-exit-liquidity-6.webp)

### Test the obvious false positives

Before publishing an alert, test at least these alternatives:

1. **Exchange sweep:** a deposit address forwarded the tokens to a known cold wallet. The first hop looked bearish only because the exchange's operational architecture is visible in stages.
2. **Bridge movement:** the source chain burned or locked tokens while a destination chain minted a representation. Search both chains and the bridge contract.
3. **Liquidity provision:** tokens went to a DEX pool, not a seller. A pool deposit can increase available liquidity and has a different interpretation from a CEX deposit.
4. **Collateral:** a lending protocol or prime broker received tokens. The holder may face liquidation risk, but custody movement is not a voluntary market sale.
5. **Loan return:** an MM sent inventory back to the project. If the flow is reversed and the contract term has ended, the same addresses can tell a completely different story.
6. **Airdrop distribution:** a distributor contract sent small amounts to many recipients. A large aggregate outflow can be community distribution rather than a single seller.

The test is not bureaucratic. Each false positive has a distinct market implication. A bridge transfer may move sellable supply to a new venue; a liquidity deposit may improve execution; collateral may create liquidation risk later; a sweep may make the initial deposit irrelevant. The same number needs different prose depending on its mechanism.

### A compact evidence schema

For repeatable research, think of each flow as an event with fields:

`time | chain | asset | source | source-confidence | destination | destination-confidence | amount | supply-denominator | event-type | evidence | alternatives`

The schema is intentionally boring. It keeps “source confidence” separate from “destination confidence,” and it makes the denominator impossible to forget. A spreadsheet works. A Dune query works. A small notebook works. The tool matters less than keeping the raw observation distinct from the interpretation.

### Why intent should stay out of the first draft

Intent is the most tempting word in on-chain analysis because it makes a story feel complete. A treasury sends tokens to a market maker, so someone “wanted to dump.” A VC moves tokens to an exchange, so the fund “exited.” The chain usually cannot observe the private contract, hedges, internal approvals, or a later return. Use intent only when a primary record, admission, or adjudicated finding supports it. Otherwise write what happened and what it is consistent with.

That discipline is especially important for real firms. Allegations can be serious and well sourced while still being allegations. A complaint may describe conduct the regulator says occurred; a settlement order may describe findings; a criminal indictment states charges, not a conviction. The reader deserves the legal status as well as the wallet story.

## 8. Second-order effects: why the visible seller is not always the first seller

Token flows can create pressure before a token is sold. A market maker may hedge a loan with perpetual futures. An investor may sell a call option to another party. A treasury may borrow stablecoins against tokens. A VC may transfer an economic interest in an OTC contract while the tokens remain in custody. These arrangements can change who bears price exposure without changing the token holder on-chain.

This is why the strongest defensive question is not “which address is selling?” but “which entity has an incentive to reduce token exposure, and which entity can deliver tokens to a venue?” Those are different roles. A borrower can be economically short while a lender remains the legal token holder. An OTC buyer can become the eventual seller while the original holder appears motionless.

### Worked example 6: the same token amount, different economic meaning

Imagine **1,000,000** hypothetical tokens at **$1** each move from a project treasury.

- In case A, the treasury lends them to an MM for a stated **30-day** term and the MM returns 1,000,000 tokens. The temporary movement is **$1,000,000 of notional inventory**, but the chain alone does not establish net disposal.
- In case B, the treasury sends 1,000,000 tokens to a CEX deposit and the balance later disappears into the exchange's internal system. This is **$1,000,000 of exchange-facing notional** at the observed price, but the executed sale remains unobserved.
- In case C, the treasury sends 1,000,000 tokens to a DEX pool and receives stablecoins and a liquidity-position token. That is an asset swap and liquidity action, not automatically a market sell.

The token count and displayed price are identical in all three cases. The event type changes the interpretation. A forensic workflow that stores only `amount` loses the information that matters.

### Worked example 7: price can rise while a holder exits

Suppose a hypothetical seller owns **2,000,000** tokens. During a two-hour window, the seller transfers **200,000** tokens to a CEX each hour, while new buyers purchase **250,000** tokens each hour at rising prices.

1. The seller has placed 400,000 tokens at the venue over two hours.
2. Buyers have demanded 500,000 tokens over the same window.
3. Net demand exceeds the seller's visible supply by **100,000 tokens**.
4. The chart can rise even as the seller reduces exposure by 400,000 tokens.

The intuition: a bullish chart does not prove that large holders are holding. It may show that new demand is temporarily larger than their exit flow.

## 9. How to write the final alert without overclaiming

A useful alert has five sentences. First, identify the source and confidence. Second, state the exact event and date. Third, quantify the amount against a named denominator. Fourth, say what the destination does and does not prove. Fifth, name the invalidation or alternative.

For the hypothetical ACME case, the alert would read: “An investor-linked cluster, labeled by the analytics provider and funded by the dated unlock, transferred 5,000,000 ACME to a known CEX deposit cluster on the recorded date. At the observed $0.40 price, that represented $2,000,000, or 25% of the dated reported spot volume. The transfer is consistent with exchange-facing sell capacity, but the exchange's internal fills are not public. The thesis weakens if the tokens are withdrawn, returned, or absorbed without a deterioration in depth.”

Notice what is absent: “dump,” “guaranteed crash,” “insider,” and “retail exit liquidity” as asserted facts. The alert is still actionable as risk awareness because it describes a measurable change in supply access and a condition to watch.

## Named case study: Terraform, LUNA, UST, and the limits of inference

Terraform is a useful case because primary records let us separate observed market history, regulatory allegations, and later findings. The SEC's amended complaint filed in April 2023 alleged that LUNA's market price rose from under $1 in early 2021 to around $119.18 in April 2022, then fell below a penny in May 2022. The complaint also described efforts to protect LUNA's trading price by limiting resale amounts during periods of the project’s growth. Those are allegations in a civil pleading, not neutral on-chain truth.

The SEC's later matter concerning Tai Mo Shan, a Jump Crypto subsidiary, is more specific about regulatory disposition: the Commission's January 31, 2025 page says the matter was settled in December 2024 and describes findings concerning offers and sales of LUNA and conduct around UST's $1 peg. The order is a primary enforcement record; it does not mean every wallet flow involving Jump was a sale or that every market move had one cause.

What does the forensic framework add? It tells us to keep four layers apart:

1. **Market outcome:** the price path recorded in the SEC complaint.
2. **Reported conduct:** what the complaint alleges about Terraform and counterparties.
3. **Regulatory finding:** what the Tai Mo Shan order says the Commission found and settled.
4. **Wallet evidence:** transfers that would still need chain-level transaction links, attribution, and timing analysis.

This is the right way to discuss contested real-firm claims: name the source, use “alleged,” “reported,” or “found” precisely, and say what remains unknown. Intent is contested unless established by a finding or admission.

## Named case study: Coinbase listing information and a different kind of flow

The SEC's July 21, 2022 release about Ishan Wahi, Nikhil Wahi, and Sameer Ramani describes an alleged scheme ahead of Coinbase listing announcements. The release says the defendants allegedly traded at least 25 crypto assets, at least 9 of which the SEC characterized as securities, and generated more than $1.1 million in alleged illicit profits. It also says the alleged conduct ran from at least June 2021 to April 2022.

This case is not a treasury-to-CEX dump. It is a reminder that “flow” can mean information flow rather than token custody flow. The market signal was a private listing schedule allegedly moving from an employee to traders. A defensive analyst should not use an on-chain wallet label to imply insider knowledge. The evidence required is different: access, confidentiality, timing, trades, and enforcement records.

The lesson is broader than Coinbase: map the type of edge before choosing the tool. Etherscan can show a transfer. It cannot prove who knew a listing date. Arkham can label a wallet. It cannot prove the label holder received confidential information.

## Named case study: FTX and the danger of treating exchange custody as a black box

The DOJ's December 13, 2022 announcement said the indictment alleged that Sam Bankman-Fried misappropriated billions of dollars of FTX customer funds and used them for investments, political contributions, and repayment of loans owed by Alameda Research. DOJ's later material also says FTX claimed approximately **$15 billion in daily trading volume** in early 2022. This is a dated claim about what the platform claimed, not an independent measure of executable liquidity.

FTX demonstrates why a deposit address is not the whole story. A blockchain observer could see funds move among addresses, but the economic ownership and liabilities lived partly inside exchange systems and corporate entities. The visible chain was one layer of a larger balance-sheet problem. The intended retail defense is therefore not “watch one wallet and you are safe.” It is “ask who owes what, who controls the keys, and which parts of the ledger you cannot independently reconcile.”

## How it shows up in price

The usual sequence is quieter than a single red candle:

1. Price rises while float is small and attention expands.
2. A known unlock makes more supply transferable.
3. Tokens move through one or more intermediary wallets.
4. Deposits to one or several CEX clusters increase.
5. The order book absorbs some supply; spreads widen or bids step down if demand is insufficient.
6. Price may drift lower, chop sideways, or continue higher if new buyers exceed the seller.

The same chain path can produce different price outcomes. If a deposit is collateral and no sale occurs, price may do nothing. If the deposit is sold gradually while a narrative brings in new buyers, the chart can rise while the seller exits. If the market is thin, a smaller flow can move price more than a larger flow in a deep market. “Tokens to exchange” is therefore a risk flag, not a directional trade signal.

### A compact price-response checklist

Compare the flow against: current float; the specific tranche unlocked; dated spot volume; venue concentration; visible bid depth; funding or derivatives context; and broader market direction. Then state an invalidation: for example, “the pressure thesis weakens if the cluster withdraws the tokens, returns them to the treasury, or the market absorbs repeated deposits without deteriorating depth.” An invalidation keeps analysis falsifiable.

## Common misconceptions

### “A whale transfer means a whale sold.”

No. A transfer is a custody event. It may be a sale, loan, collateral posting, internal rebalance, bridge move, or exchange sweep. Use the strongest language the evidence supports.

### “Unlock equals circulating supply.”

Unlock means a holder can transfer under the relevant rules. Some unlocked tokens remain in cold custody or a vesting recipient address. Circulating-supply providers may use a different convention. Reconcile the definition before computing percentages.

### “A labeled wallet is a confirmed identity.”

Labels are often probabilistic. A self-published treasury address is stronger than a vendor's “smart money” tag. Even a confirmed entity may use custodians or several wallets.

### “High volume means an unlock is harmless.”

Volume is turnover, not guaranteed absorption. It can include multiple venues, derivatives, internal activity, and low-quality trades. Compare the flow with relevant spot depth and use a dated denominator.

### “OTC removes supply pressure.”

OTC can reduce immediate visible slippage, but it transfers ownership. The buyer may later sell into an exchange. OTC changes the path and timing; it does not make supply disappear.

### “The price fell after the deposit, so the deposit caused it.”

Timing is evidence of a possible relationship, not proof of causality. Check broader market moves, other deposits, liquidations, listings, unlocks, and the exact venue window.

## Retail defensive takeaway

Before treating a token rally as proof of demand, create a one-page evidence sheet:

- Copy the allocation and vesting source, with its publication date.
- Record total supply, circulating supply, and your denominator.
- Identify vesting contracts and treasury addresses on the correct chain.
- Treat Arkham, Nansen, Bubblemaps, and similar labels as hypotheses to corroborate.
- Use Etherscan for transaction hashes and Dune for reproducible aggregates.
- Check DefiLlama for protocol context, while keeping TVL separate from executable depth.
- Mark the unlock date, loan term, intermediate hops, CEX deposit time, and price/volume window.
- Write the strongest alternative explanation and the observation that would weaken your thesis.

This process will not predict every move. It can prevent a more basic mistake: buying a thinly floated token without noticing that a much larger group has just acquired the right to sell it.

## When this matters to you

The practical question is not whether you can identify every insider. You cannot. It is whether your position depends on a market absorbing supply that is visible, scheduled, and concentrated. If it does, the risk belongs in your mental model before the trade, not as a surprise after the chart turns.

The method is educational, not individualized financial advice. On-chain data can be incomplete, labels can be wrong, exchange books are partly private, and a correct observation can still produce an incorrect forecast. Use the workflow to improve questions and risk awareness, not to manufacture certainty.

## Sources & further reading

- [Terraform Labs amended complaint](https://www.sec.gov/files/terraform-labs-pte-ltd-amended-complaint.pdf), SEC, filed April 3, 2023; price figures and allegations are attributed to the pleading.
- [Tai Mo Shan matter](https://www.sec.gov/enforcement-litigation/distributions-harmed-investors/tai-mo-shan-limited), SEC, page dated January 31, 2025; settled matter concerning LUNA and UST conduct.
- [Coinbase listing-insider release](https://www.sec.gov/newsroom/press-releases/2022-127), SEC, July 21, 2022; alleged trades, asset counts, dates, and alleged profits.
- [FTX founder indictment announcement](https://www.justice.gov/archives/opa/pr/ftx-founder-indicted-fraud-money-laundering-and-campaign-finance-offenses), DOJ, December 13, 2022; allegations concerning customer funds and Alameda.
- [United States v. Ryan Salame filing](https://www.justice.gov/usao-sdny/media/1313336/dl), DOJ/SDNY, filed material describing FTX's claimed early-2022 daily volume.
- Continue with [reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table), [unlock cliffs and supply overhang](/blog/trading/crypto-players/unlock-cliffs-and-the-supply-overhang-trade), and [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does).
