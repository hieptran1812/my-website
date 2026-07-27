---
title: "DWF Labs: The Controversial Newcomer Who Rewrote the Market-Maker Sales Playbook"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A build-from-zero profile of DWF Labs, the self-described venture market maker that fused token investing with market making and turned the deal announcement itself into a product, plus a careful walk through the contested wash-trading allegations, the mechanics that make them worth investigating, and how to trace a bundled deal on-chain yourself."
tags: ["crypto", "market-makers", "dwf-labs", "crypto-players", "onchain-analysis", "wash-trading", "token-liquidity", "otc-trading", "conflicts-of-interest", "market-microstructure", "crypto-vc"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 46
---

> [!important]
> **TL;DR** — DWF Labs is a crypto trading firm that packaged three separate businesses into one product: buying a project's tokens at a discount, quoting markets on those same tokens, and announcing the whole thing as an "investment." That bundle is legal, common, and structurally conflicted at the same time — which is exactly why it draws allegations, and why the allegations are hard to settle.
>
> - **The model:** DWF Labs launched in September 2022 as the investment arm of Digital Wave Finance, a high-frequency trading firm. Instead of writing venture cheques into funding rounds, it typically bought liquid tokens straight from a project's treasury at a discount to the market price, often bundled with market making, listing help and marketing (CoinDesk, 14 April 2023).
> - **The sales innovation:** the *announcement* became the product. A headline "\$40 million investment" — the size reported for the Fetch.ai and Tomi deals (CoinDesk, 29 March 2023) — is a number the market reads as validation, even when the cash is a discounted block purchase rather than equity into a round.
> - **The conflict:** an investor wants the price up. A market maker is supposed to be indifferent to direction. A dealer selling treasury supply wants the price up *while it sells*. One firm holding all three roles on one token has an incentive map that a regulator would find uncomfortable in equities.
> - **The contested case:** the *Wall Street Journal* reported on 9 May 2024 that Binance's internal investigations team concluded DWF had manipulated the price of YGG and at least six other tokens and made over \$300 million in wash trades in 2023, and that the team's head was fired about a week after filing the report. Binance said there was "insufficient evidence" of market abuse. DWF called the claims "unfounded" and said they "distort the facts." **These are reported allegations that the firm denies. No public regulatory finding of manipulation against DWF Labs is known as of 2026-07-27.**
> - **What you can actually check:** you cannot read intent off a blockchain, but you can read the *route, the timing and the size* of every token transfer. Treasury to maker wallet to exchange deposit is a public, timestamped trail — and the last section of this post is the method for reading it.
> - The number to hold onto: DWF Labs' own site advertised a ticket range of **\$10 million to \$50 million** per project for its \$250 million Liquid Fund (announced 24 March 2025). That is the size of cheque that now comes bundled with a quoting mandate.

In March 2023, a firm almost nobody in crypto had heard of a year earlier announced a \$40 million investment in Fetch.ai. A few weeks later, \$40 million into Tomi. Then \$10 million into CryptoGPT. Then Synthetix. Then dozens more. By April 2023, CoinDesk counted more than \$200 million of announced deals from a company that had existed for roughly seven months (CoinDesk, 29 March 2023 and 14 April 2023).

Traditional venture funds do not deploy like that. They take months of diligence per deal, sit on a board seat, and wait five years for an exit. DWF Labs was signing deals in days and, in several documented cases, moving the tokens to an exchange within a week.

That is not a criticism by itself. It is a description of a *different business*, and the first job of this post is to explain precisely what that different business is — because most of the argument about DWF Labs is really an argument about vocabulary. Is a discounted block purchase of already-trading tokens an "investment"? Is a firm that both buys a token and quotes it a "market maker" or something else? Once you can answer those two questions cleanly, the controversy stops being noise and becomes a specific, checkable set of claims.

![One firm can hold three roles on the same token at once, and the three roles want different things from the price.](/imgs/blogs/dwf-labs-the-controversial-newcomer-1.webp)

The figure above is the mental model for everything that follows. Three hats — investor, market maker, OTC dealer — sit on one counterparty. Two of the three have a strong preference about where the price goes. One is supposed to have none. Every allegation you will read about later in this post is, at bottom, a claim that the hats got mixed up.

This post is a companion to the series hub, [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers), and it assumes the mechanics laid out in [What a Crypto Market Maker Actually Does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does). You do not need to have read either — everything gets defined from zero here — but they are the natural next stops.

**A note on how the contested parts are written.** Every allegation in this article is reported as an allegation, attributed to the outlet that reported it, and paired with the denial that followed. Nothing here asserts that DWF Labs manipulated any market. The section on manipulation mechanics exists so that a reader can understand *why* particular patterns get investigated and *how to look for them defensively* — not as instructions for producing them. This is educational writing, not financial advice, and not a legal conclusion about anybody.

## Foundations: the words you need first

Skip this section if you already trade. If you do not, read it — everything after it leans on these ten terms, and none of them is hard once someone actually defines it.

### The order book, the spread, and liquidity

A **token** is a unit of a crypto project, tradeable on exchanges much like a share is tradeable on a stock exchange. An **exchange** matches buyers and sellers. On most exchanges, that matching happens through an **order book**: a live list of every standing offer to buy and every standing offer to sell.

- A **bid** is a standing order to buy: "I will pay \$0.99 for this token."
- An **ask** (or **offer**) is a standing order to sell: "I will sell at \$1.01."
- The **spread** is the gap between the best bid and the best ask. Here, \$1.01 − \$0.99 = **\$0.02**.
- The **mid-price** is the average of the two: **\$1.00**. It is a convenient shorthand for "where the market is," even though you cannot trade at it.
- **Depth** is how much money is resting in the book near the mid. A book with \$2 million of bids within 1% of the mid is *deep*. A book with \$40,000 is *thin*.

**Liquidity** is the combination of a tight spread and real depth: the property that you can buy or sell a meaningful amount without moving the price much. **Slippage** is what it costs you when liquidity is missing — the gap between the price you saw and the average price you actually got.

### The market maker

A **market maker** (MM) is a firm that continuously posts *both* a bid and an ask on the same token at the same time, and earns the spread between them across thousands of trades. It is a volume business, not a directional bet. The maker's defining property is **indifference to direction**: it does not care whether the token goes up or down, only that people keep trading, and that it does not get stuck holding a position while the price moves against it.

That leftover position has a name. **Inventory** is the net tokens the maker is long or short as a residue of whoever happened to trade against it. A maker with zero inventory is **flat**, and a flat maker genuinely has no view. A maker sitting on ten million tokens is not flat, and it is no longer indifferent — it now profits when the price rises. Hold onto that: *inventory is what converts a neutral market maker into an interested party.* The mechanics of managing it are the subject of [Inventory Risk, Hedging and Delta Neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality).

### Token treasuries, float, and unlocks

When a crypto project launches, it typically creates a fixed supply of tokens and keeps a large share of them in a **treasury** — a wallet controlled by the project or its foundation, used to pay contributors, fund grants, and, crucially, raise money by selling tokens.

The **circulating supply** (or **free float**) is the portion of tokens actually available to trade. Everything sitting in a treasury, or held by an investor under a lockup, is *not* float. This matters enormously: a token with a \$100 million "market capitalisation" (price times total supply) but only 10% of supply actually circulating is a \$10 million market wearing a \$100 million costume.

A **lockup** (or **vesting** period) is a contractual promise not to sell for a stated time — a year is common. An **unlock** is the moment that promise expires and the tokens become sellable. Unlocks are usually published in advance, which means the market can anticipate them. The full arc is covered in [The Lifecycle of a Token: Seed to Unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock).

### OTC, discounts, and the dealer

**OTC** stands for **over the counter** — a trade negotiated privately between two parties rather than matched on a public exchange. If a project wants to sell \$10 million of its own tokens and does it on the open market, it will crush the price. If it sells the block privately at a negotiated price, the market never sees the order.

The buyer of that block will insist on a **discount to spot** — a price below the current market price — as compensation for the risk of holding a large, hard-to-exit position. A 10% discount on a token trading at \$1.00 means an entry at \$0.90.

A **dealer** in this context is a firm that buys blocks and then works them out into the market over time. That is a legitimate, ancient business. What makes it interesting in crypto is that the same firm is often *also* the market maker quoting the book those tokens will eventually be sold into.

### Wash trading

**Wash trading** is buying and selling the same asset with yourself — or with a coordinated partner — so that trades print on the tape without any genuine change of ownership. The purpose is to manufacture the *appearance* of volume and interest.

It is illegal in regulated securities markets in most jurisdictions, and it is prohibited by the terms of service of major crypto exchanges even where the local securities law does not clearly reach the asset. The reason it is banned is not moralistic: volume is a signal that other traders and ranking sites rely on, and fake volume corrupts that signal for everybody. The general mechanics are unpacked in [Wash Trading, Spoofing and Manufactured Volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume).

Two things about wash trading are worth internalising now, because they explain why the DWF story stayed unresolved:

1. **It is hard to prove from outside.** Two accounts trading with each other look, on the tape, exactly like two strangers trading with each other. Distinguishing them usually requires exchange-internal account data — who owns which account, funded from where.
2. **A market maker's legitimate activity looks superficially similar.** A maker holding inventory on two venues and rebalancing between them generates a lot of self-directed flow that is entirely normal. This is precisely why allegations against market makers are contested rather than obvious.

That is the whole vocabulary. Now the firm.

## 1. Who DWF Labs is

DWF Labs launched in **September 2022** as the investment arm of **Digital Wave Finance** — that is what the initials stand for — a proprietary high-frequency trading firm. Its managing partner and public face is **Andrei Grachev**, who had previously run Huobi's Russia office in 2018 and who has publicly denied that the firm received Russian funding (CoinDesk, 14 April 2023). The firm is headquartered in Dubai.

The order of operations matters. DWF Labs did not start as a venture fund that later added trading. It started as a *trading firm* that later added a venture-shaped wrapper. Grachev's framing at the time was that the firm had "accumulated enough funds from our profits to invest" and that a bear market was "the best time to join the investment space" (CoinDesk, 29 March 2023). The capital came off a trading P&L, and the deals were priced by traders.

Scale, dated and attributed, because the numbers here move and several of them are self-reported:

| Claim | Value | Source and date |
|---|---|---|
| Venue coverage of the parent trading firm | 40+ exchanges | CoinDesk, 14 April 2023 |
| Venue coverage claimed by DWF Labs | 60+ exchanges | dwf-labs.com, checked 2026-07-25 |
| Portfolio size claimed by DWF Labs | "800+ projects" on one page; "1000+" on another | dwf-labs.com, checked 2026-07-25 |
| Deals led, Q3 2022 to Q3 2023 | 39 deals, \$324 million | Binance Research, as reported 2023 |
| Announced deals by April 2023 | more than \$200 million | CoinDesk, 14 April 2023 |
| Reported monthly trading volume | more than \$4 billion | CoinDesk, 9 May 2024, summarising the WSJ report |
| Liquid Fund | \$250 million, tickets \$10–50 million | dwf-labs.com announcement, 24 March 2025 |

Note the second and third rows. DWF Labs' own website advertised two different portfolio counts on two different pages when I checked it on 2026-07-25. That is not a scandal — marketing pages drift — but it is a useful calibration exercise. **Self-reported scale figures from any crypto firm are marketing copy, not audited disclosures.** Treat "1000+ projects" the way you would treat a restaurant's claim to be "world famous."

The firm has kept expanding its surface area. On 24 March 2025 it announced a \$250 million **Liquid Fund** aimed at mid- and large-cap projects, writing tickets of \$10 million to \$50 million, and said it had deployed more than \$11 million in the preceding two weeks with deals of \$25 million and \$10 million pending (DWF Labs announcement, 24 March 2025). In April 2025 it was reported to have bought \$25 million of World Liberty Financial's WLFI token in a private transaction. It also incubated **Falcon Finance**, a synthetic-dollar protocol co-founded by Grachev, launched in early 2025.

The shape to notice: the firm keeps moving *toward* the assets it also trades and quotes. That is the structural feature the rest of this post is about.

## 2. The operating model: what a "venture market maker" actually sells

DWF Labs describes itself as a "new generation Web3 investor and market maker." The compound term is not marketing fluff — it is an accurate description of a bundle. Here is what is inside it.

![A single announced number wraps a discounted token purchase plus quoting, listings, marketing and treasury services.](/imgs/blogs/dwf-labs-the-controversial-newcomer-2.webp)

According to internal messages reviewed by CoinDesk (14 April 2023), DWF presented projects with a menu rather than a single product. Two structures recurred:

- **Buy liquid tokens straight from the project treasury at a discount to spot, with no lockup.** The project gets cash today; DWF gets tokens it can move immediately.
- **Buy a lump sum at a steeper discount with a one-year lockup, bundled with market-making services.** The project gets cash *and* a quoted market; DWF gets more tokens per dollar but cannot sell them for a year.

Around either core, the deal could include liquidity provision, exchange listings, media coverage and treasury-management help. Grachev described the range plainly: "we have pure investments without market making, we have market-making [agreements] without investment, and we have [them] combined" (CoinDesk, 14 April 2023).

The key structural fact is in that last clause. When capital and quoting are combined, the firm is simultaneously long the token and responsible for the price discovery on it.

### Why a project says yes

It is worth understanding the demand side, because the model did not spread by accident. Put yourself in the seat of a founder whose token launched eighteen months ago.

Your treasury is denominated in your own token. You need dollars to pay engineers. You cannot sell into the open market without visibly tanking your own price and enraging your community. You want a Tier-1 exchange listing, and exchanges want to see that a professional maker will support the book. And your token's chart has been flat for six months, so nobody is writing about you.

A single counterparty walks in and offers: dollars today, a maker on the book tomorrow, an introduction to a listing desk, coverage, and a press release with a large number in it. Compared to running four separate processes with four separate counterparties over six months, that is an extremely attractive package. The bundling is the innovation.

#### Worked example 1: what the discount actually buys

Suppose a token trades at exactly \$1.00. A project treasury wants \$10,000,000 of cash and is offered the two structures above. All numbers here are illustrative round figures chosen so you can check the arithmetic in your head.

![A ten percent discount without a lockup and a thirty percent discount with one are different instruments, not different prices.](/imgs/blogs/dwf-labs-the-controversial-newcomer-3.webp)

**Option A — liquid OTC at a 10% discount, no lockup.**

- Entry price: \$1.00 × (1 − 0.10) = **\$0.90**
- Tokens received: \$10,000,000 ÷ \$0.90 = **11,111,111 tokens** (call it 11.11 million)
- The maker can move them from day one. Its economics are essentially settled at signing: it holds roughly \$11.1 million of notional value against \$10 million paid, an instant paper gain of about **\$1.1 million** — which it only realises if it can actually sell without moving the price.

**Option B — 30% discount with a 12-month lockup, bundled with market making.**

- Entry price: \$1.00 × (1 − 0.30) = **\$0.70**
- Tokens received: \$10,000,000 ÷ \$0.70 = **14,285,714 tokens** (call it 14.29 million)
- The maker cannot sell for a year. Its payoff is 14.29 million × whatever the price is in twelve months.

**Where is the break-even between them?** Option B is worth more than Option A at the end of the year if:

$$14{,}285{,}714 \times P_1 > 11{,}111{,}111 \times P_0$$

where $P_1$ is the price in twelve months and $P_0$ is today's \$1.00. Solving:

$$P_1 > \frac{11{,}111{,}111}{14{,}285{,}714} \times 1.00 = 0.778 \text{ USD}$$

So Option B beats Option A as long as the token has fallen by less than about **22%** over the year.

Two things fall out of that arithmetic, and they are the whole reason this deal shape is contentious.

First, **the extra 20 points of discount is the price of the lockup.** That is ordinary finance: you pay less for something you cannot sell. Nobody should be scandalised by it.

Second — and this is the part that matters — **Option B converts a purchase into a twelve-month directional position, held by the same firm that is quoting the token's order book for those twelve months.** The maker's payoff is now $14.29\text{M} \times P$, strictly increasing in the price, on an asset whose visible price it helps set. That is not evidence of wrongdoing. It is a description of an incentive.

*The intuition to take away: the discount is not the interesting number. The lockup is, because the lockup is what gives the market maker a year-long reason to care which way the price goes.*

For a longer treatment of the standard alternative structure — where the maker borrows tokens and is paid in call options rather than buying them outright — see [The Loan-Plus-Options Deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid).

## 3. The sales innovation: the announcement as a product

Plenty of firms did bundled deals before DWF Labs. What DWF did differently, and what genuinely changed the market, was the *marketing* of them.

Traditional crypto market makers are near-invisible by design. Their client agreements are confidential, their P&L comes from spread capture, and public attention is a liability. Wintermute, GSR, Cumberland: you will find their names in exchange rankings and not much else.

DWF inverted that. Deals were announced loudly, frequently, and with the largest defensible number attached. The announcement did four things at once:

1. **It marketed to the next project.** Every founder watching saw a peer get a headline. The announcement *was* the sales pitch, and it scaled better than any business-development team.
2. **It marketed to the token's holders.** A large number from a named institution reads as third-party validation, which is scarce and valuable for a mid-cap token.
3. **It created a public event with a timestamp.** Attention concentrates around events. An announcement is an event a firm can schedule.
4. **It made the firm itself a brand.** By early 2023 "DWF invested in X" was a recognisable market phrase — which is remarkable for a company that was six months old.

Compare that to how a16z crypto or Paradigm operate, which I cover in [a16z Crypto: The Institutional Giant](/blog/trading/crypto-players/a16z-crypto-the-institutional-giant): equity-and-token positions in early rounds, multi-year holds, heavy public research output, and a deliberate distance from the trading desk. DWF's model is closer to a dealer's, wearing a venture fund's clothing.

#### Worked example 2: unpacking a "\$40 million investment"

Take an announced headline of **\$40,000,000** — the size reported for both the Fetch.ai and Tomi deals (CoinDesk, 29 March 2023). What can that number legitimately mean? All of the following are consistent with the same press release. The figures are illustrative.

**Reading 1 — cash into a funding round.** The firm wires \$40,000,000 to the company in exchange for equity or newly issued tokens with a multi-year vesting schedule. Money at risk today: \$40,000,000. This is what most readers assume.

**Reading 2 — a discounted treasury block.** The firm buys \$40,000,000 of existing tokens from the treasury at a 25% discount, receiving \$40,000,000 ÷ 0.75 = **53.3 million tokens** worth \$53,300,000 at spot. Money at risk today: \$40,000,000, but against an immediately marked paper gain of \$13,300,000. Nothing new was issued; the project's float just grew by 53.3 million tokens.

**Reading 3 — a commitment, drawn down over time.** The firm commits \$40,000,000 to be deployed in tranches "over the next 12 months," subject to milestones. Money at risk today: possibly \$2,000,000. The headline is a ceiling, not a cheque.

**Reading 4 — a facility with a lock and a claw-back.** \$40,000,000 nominal, of which \$30,000,000 is a token loan the firm must return, and the economics are a fee plus options. Actual net capital committed: perhaps \$10,000,000.

Every one of these can be announced as "a \$40 million investment," and it is often impossible for an outsider to tell which one happened. In November 2023, for one example, a \$10 million TON ecosystem commitment was reported alongside "50 seed investments scheduled over the next 12 months" — a structure much closer to Reading 3 than to Reading 1.

*The intuition: an announced dollar number in crypto is a claim about size, not a claim about structure. Two deals with identical headlines can have a fourfold difference in capital actually at risk. Until you know the discount, the lockup and the drawdown schedule, you do not know what happened.*

This is the single most useful habit this post can give you. When you see a deal announcement, the question is never "how big is the number." It is **"what did each side actually give up, and when can either side sell?"**

### The critique that followed

The model drew pointed criticism almost immediately, and the criticism was about vocabulary as much as conduct.

An executive at a rival market-making firm told CoinDesk the arrangements were "poorly disguised agency OTC trades" — meaning DWF was helping projects offload treasury tokens without disclosing that this was what was happening. The founder of a crypto analytics firm said DWF would "market them as an investment, and then claim to do 'market making' so they can keep funds on exchanges and just dump," and called keeping the tokens on exchanges "a red flag."

Walter Teng of Fundstrat put the structural point most cleanly: **"If you invest, you want the token's price to go up. If you market make, you can manipulate the price."**

DWF pushed back on the framing rather than the facts. Partner Stefano Virgilli argued: "if we're purchasing the tokens and they're using the funds to further develop, that's an investment." Grachev said the firm keeps inventory on exchanges for operational reasons, not because it intends to sell, and later acknowledged that his "biggest mistake" was failing to explain the firm's operating philosophy well enough (all quotes: CoinDesk, 14 April 2023).

Both positions are defensible on their own terms. A block purchase whose proceeds fund development genuinely is capital into a business. And a maker genuinely does need inventory sitting on exchanges to quote both sides. The disagreement is not really about whether either statement is true; it is about whether a reader of the press release could tell.

## 4. How the model touches price

Here is where the analysis has to get precise, because "market makers move prices" is the kind of sentence that is either trivially true or seriously wrong depending on what you mean.

A bundled deal touches the price through three distinct channels. They have different mechanics, different visibility, and very different ethical status. Conflating them is the main reason public argument about market makers goes nowhere.

![A bundled deal reaches the price through three separate channels, and only one of them is actual liquidity provision.](/imgs/blogs/dwf-labs-the-controversial-newcomer-4.webp)

### Channel 1: the announcement

The announcement moves price through attention. A token that nobody was looking at is suddenly in feeds, on listing-site "trending" tabs, and in group chats. New buyers arrive. On a thin float, a small amount of new buying moves the price a lot.

This channel involves no trading by the maker at all. It is a communications effect, and it is entirely legal. It is also the channel with the shortest half-life: attention decays in days.

The CoinDesk reporting on the So-Col deal is the documented example. DWF announced a \$1.5 million investment in SIMP tokens on 28 March 2023 (CoinDesk, 14 April 2023). SIMP roughly doubled to about 3.4 cents within a week — then fell back toward 1 cent by 4 April 2023. That round trip took eight days.

### Channel 2: the quoting

This is the real market-making channel, and it is the one that unambiguously helps. A maker posting genuine two-sided quotes tightens the spread and deepens the book. Every trader on that pair gets a better price. This is a public good and it should be said plainly, because it is the honest answer to "why do projects hire these firms at all."

#### Worked example 3: what a maker is actually worth on a thin book

Suppose a token trades at \$1.00 and you want to sell **\$500,000** of it in one market order. Two scenarios; the depth figures here are invented for the walkthrough, but the arithmetic is exactly how execution cost works.

**Scenario A — the thin book.** No dedicated maker. Resting bids amount to roughly **\$75,000 for every 1% you walk down** from the mid. Selling \$500,000 means eating through:

$$\frac{500{,}000}{75{,}000 \text{ per } 1\%} = 6.67\%\ \text{of price depth}$$

You do not pay 6.67% on the whole order — you pay progressively worse prices on the way down, so your *average* fill sits roughly halfway: **3.33% below the mid**. In cash:

$$500{,}000 \times 3.33\% \approx 16{,}700\ \text{USD of slippage}$$

**Scenario B — the made book.** A maker is quoting, and depth is ten times better: **\$750,000 per 1% band**. The same order walks down 0.67%, so the average fill is about **0.33% below the mid**:

$$500{,}000 \times 0.33\% \approx 1{,}650\ \text{USD of slippage}$$

The maker's presence saved you **\$15,050 on a single \$500,000 trade** — roughly a tenfold reduction in execution cost.

![A dedicated maker cuts the slippage bill on the same order by roughly ten times at every order size.](/imgs/blogs/dwf-labs-the-controversial-newcomer-5.webp)

The chart shows the same comparison at four order sizes. Notice that the gap *widens* with size: at \$150,000 the thin book costs 1.00% against 0.10%, but at \$750,000 it costs 5.00% against 0.50%. Liquidity is not a luxury good for large traders — it is a tax on them, and the maker is the thing that removes the tax.

*The intuition: whatever else is true about bundled deals, the quoting leg is genuinely valuable, and a token with no maker is a token you cannot exit at a sane price.*

### Channel 3: the supply

This is the channel almost nobody prices correctly, and it is the one that quietly does the most damage.

When a project sells \$10 million of treasury tokens to a maker, those tokens move from a wallet that was never going to sell into a wallet whose business is transacting. The *circulating supply* just increased, without any of the public choreography of a scheduled unlock. There was no unlock date on a calendar, no community announcement, no chart annotation on a token-unlock tracker. The float simply got bigger, quietly.

#### Worked example 4: sizing the overhang

Take our Option B deal from earlier. Illustrative numbers again.

- Token price: **\$1.00**
- Circulating supply before the deal: **100,000,000 tokens**, so a circulating market cap of **\$100,000,000**
- Maker receives **14,285,714 tokens** at \$0.70, locked for twelve months

When the lockup expires, the potential new float is:

$$\frac{14{,}285{,}714}{100{,}000{,}000} = 14.3\%\ \text{of circulating supply}$$

Now ask how long the market would need to absorb it. Suppose genuine net buying — real demand, not churn — runs at **\$250,000 per day**. At \$1.00 per token, absorbing the position without a price decline requires:

$$\frac{14{,}285{,}714 \text{ tokens} \times 1.00}{250{,}000 \text{ per day}} = 57\ \text{trading days}$$

Roughly **three months of every single dollar of genuine net demand** doing nothing but absorbing one counterparty's position. In practice it never works that cleanly: the moment the market suspects the selling has begun, buyers step back, daily net demand falls, and the denominator shrinks while the numerator stays put.

*The intuition: the supply channel is a slow leak, not an event. A scheduled unlock is priced in advance because everyone can see the date. A treasury block sale is not on anyone's calendar, which is exactly why it surprises people three months later.*

This is why the third channel is the one to watch, and why the on-chain method later in this post focuses on it. You cannot observe intent. You *can* observe that 14.29 million tokens left a treasury on a Tuesday.

## 5. The contested case: what was reported, and what was denied

Now the part that requires the most care. Read the framing rules before the facts: everything below is a *report* of an allegation, attributed to the outlet that published it, paired with the response. None of it is a finding of fact by a court or a regulator, and as of 2026-07-27 no public regulatory enforcement action or court finding of manipulation against DWF Labs is known to me.

![The public record is a sequence of reported allegations, each paired with a denial and no regulatory finding.](/imgs/blogs/dwf-labs-the-controversial-newcomer-6.webp)

### Summer 2023: the community criticism

Through the middle of 2023, DWF Labs drew sustained criticism in crypto communities over its secondary-market activity in tokens including **YGG, DODO, C98 and CYBER** (as reported by CoinDesk). The criticism was of the general shape described above: that price strength around announcement dates, followed by weakness, was consistent with distribution rather than support.

Grachev denied the characterisation directly. In a September 2023 interview with Blockbeats he said: **"We are not involved in any manipulation."**

It is worth being honest about the epistemic situation here. Community criticism based on chart shapes is *weak evidence*. Tokens go up on announcements and fade afterwards constantly, for entirely mundane reasons — announcement-driven attention decays, and a token that ran 100% in a week attracts profit-taking regardless of who is quoting it. A chart that fits the accusation also fits several innocent explanations. Chart-shape arguments are a reason to look closer, not a conclusion.

### 9 May 2024: the Wall Street Journal report

This is the substantive allegation, and it is substantive precisely because it is not chart-reading — it is a claim about what an exchange's own surveillance team found in its own data.

The *Wall Street Journal* reported on 9 May 2024, sourced to a former Binance insider, that:

- Binance's internal investigations team — staffed with people from traditional finance — reviewed DWF's activity after complaints from **competing market makers**.
- The team concluded that DWF had **"manipulated the price of YGG and at least six other tokens, and made over \$300 million in wash trades in 2023,"** in violation of Binance's terms of service.
- The **head of the investigations team was fired about a week after submitting the findings**, following a complaint to leadership from the head of Binance's VIP client department.
- Binance subsequently determined there was **"insufficient evidence"** that DWF had engaged in market abuse, citing concerns that the investigators were biased toward the competing market makers who raised the original complaint.

For scale: DWF was reported to be doing more than **\$4 billion of trades a month**, and Binance's VIP clients traded more than \$100 million monthly (CoinDesk, 9 May 2024, summarising the WSJ report). The alleged \$300 million of wash trades is a figure covering the full year 2023.

### The denials

Both named parties responded publicly the same day.

**DWF Labs** said the allegations were **"unfounded and distort the facts"** (statement posted to X, 9 May 2024). A founding partner separately characterised the accusations as **"competitor-driven FUD"** — fear, uncertainty and doubt originating from rival firms (DL News). DWF also described the WSJ's claims as "misinterpretations without supporting evidence" and said it would cooperate fully with any investigation.

**Binance** said: **"We do not tolerate market abuse. Over the last three years, we have offboarded nearly 355,000 users with a transaction volume of more than \$2.5 trillion for violating our terms of use."** On the firing, Binance said the dismissal followed an inquiry that found the allegations were not "fully substantiated."

### How to hold this honestly

There are three separate propositions in the reporting, and they have different evidentiary weight. Keeping them apart is the whole discipline.

| Proposition | What supports it | Status |
|---|---|---|
| Binance's investigations team produced a report making these findings | A named outlet's reporting from a former insider; Binance's own response implicitly acknowledges an inquiry existed | Reported, and not denied in substance by either party |
| The head of that team was fired shortly afterwards | WSJ reporting; Binance confirmed a dismissal but disputed the framing | Reported, with a competing explanation on the record |
| DWF Labs manipulated markets | The investigations team's internal conclusion, as reported | **Alleged.** Denied by DWF. Binance itself concluded the evidence was insufficient. No regulatory or judicial finding is known as of 2026-07-27 |

The second row is the one that made the story travel. Even people with no view on DWF found the sequence uncomfortable: a surveillance team files an adverse report on a large client, the client-relationship side objects, and the surveillance head is gone within a week. That is a *governance* story about exchange conflicts of interest, and it is arguably the more durable takeaway. An exchange earns fees from the volume it is also supposed to police. When surveillance and revenue collide inside one company, the reader learns something regardless of whether DWF did anything wrong.

Note also who initiated the complaint: **rival market makers**. That cuts in both directions, and you should hold both. Competitors have the best data and the strongest motive to notice genuine misconduct — they lose money to it directly. Competitors also have the strongest motive to file complaints about a firm winning their mandates. Binance itself cited the second possibility as a reason to discount the report. Neither reading is provably correct from the outside.

## 6. Why the allegations are plausible to investigate

The purpose of this section is narrow and worth stating up front: to explain *why* a surveillance team looks at deals of this shape, and how the arithmetic of detection works — so that you can read manipulation claims critically. It is not a description of how to do anything, and everything below is a detection method, not a technique.

### The structural reason

Return to the three hats. A bundled deal creates a firm that is (a) long a large token position, (b) in control of the visible quoting on that token, and (c) frequently under an obligation to distribute treasury supply. In regulated equity markets, this combination is precisely what information barriers exist to prevent — the prop desk holding a position, the market-making desk quoting it, and the syndicate desk distributing it are separated by policy, by systems, and by law.

Crypto has, for the most part, no such requirement. The separation, where it exists, is voluntary. The difference between designated and principal market making, and why the distinction matters legally, is covered in [Designated Versus Principal Market Making](/blog/trading/crypto-players/designated-versus-principal-market-making).

That is why the *structure* draws scrutiny even when the *conduct* is clean. A surveillance analyst does not need evidence of wrongdoing to open a file on an entity that combines all three roles on a thin-float token. The combination itself is the trigger.

### Why volume specifically

Volume is the metric that manufacturing distorts most efficiently, for three reasons:

1. **Ranking sites sort by it.** Appearing in a "top gainers by volume" list is free distribution.
2. **Exchange listing committees look at it.** Demonstrated liquidity is an input to listing decisions.
3. **Traders treat it as confirmation.** A price move on high volume is read as more real than the same move on low volume, which is a reasonable heuristic that becomes exploitable the moment volume can be fabricated.

#### Worked example 5: the arithmetic of a volume-integrity check

This is the analysis a surveillance team, an exchange listing committee, or you with a spreadsheet would run. Illustrative numbers throughout.

Take a token with:

- Circulating market cap: **\$50,000,000**
- Reported daily volume: **\$40,000,000**
- Number of distinct on-chain holders: **12,000**

**Test 1 — the turnover ratio.** Daily volume divided by circulating market cap:

$$\frac{40{,}000{,}000}{50{,}000{,}000} = 0.80,\ \text{or } 80\%\ \text{of the float changing hands every day}$$

For context, a heavily traded large-cap equity turns over roughly 1% of its float per day; a very active crypto major might do 10–20%. A sustained 80% daily turnover on a \$50 million token is not impossible — it happens during genuine mania — but it is a strong prompt to check whether the volume is real. If the same ratio persists for weeks *without* price volatility, the prompt becomes a red flag: real turnover that heavy almost always comes with violent price movement.

**Test 2 — volume per holder.** Divide daily volume by holders:

$$\frac{40{,}000{,}000}{12{,}000\ \text{holders}} = 3{,}333\ \text{USD of daily volume per holder}$$

Every single holder, including the thousands with dust balances, would need to trade \$3,333 per day. Compare that with the number you would expect from a retail base: perhaps \$50–200 per active holder per day, with most holders trading nothing on any given day. A 17-to-67-times gap says the volume is not coming from the holder base. It is coming from a small number of very active accounts — which may be entirely legitimate makers and arbitrageurs, or may not be.

**Test 3 — scale the reported allegation.** The WSJ report describes over \$300 million of alleged wash trades across YGG and at least six other tokens during 2023. Spread evenly across seven tokens and 365 days:

$$\frac{300{,}000{,}000}{7\ \text{tokens} \times 365\ \text{days}} \approx 117{,}000\ \text{USD per token per day}$$

That is a genuinely useful sanity check, and it cuts *against* the most sensational reading of the headline. On a token doing \$40 million of daily volume, \$117,000 per day is roughly **0.3%** of the tape — invisible without account-level data, and far too small to be "the reason" for any large price move by itself. The \$300 million number sounds enormous as a lump sum and is small as a daily rate. Both statements are true, and the second one is the more informative.

*The intuition: aggregate figures in manipulation reporting are almost always quoted as annual totals because that maximises the number. Always divide by the time period and the number of assets before deciding how big the alleged conduct actually was.*

### What outsiders genuinely cannot determine

Be clear about the limits, because this is where confident commentary usually goes wrong:

- **You cannot identify wash trading from public tape.** Two accounts trading with each other are indistinguishable from two strangers trading with each other unless you know who owns the accounts. Only the exchange knows that.
- **You cannot distinguish inventory from intent.** Tokens sitting on an exchange might be quoting inventory or might be pre-positioned for sale. The same wallet balance is consistent with both.
- **You cannot infer causation from a chart.** A price that rises after an announcement and falls afterwards is the single most common pattern in crypto. It is what attention decay looks like.

What you *can* do is establish the factual skeleton: what moved, when, how much, and to where. That is the next section.

## How to trace DWF-style deals on-chain

This is the defender's method: how to establish the observable facts of a bundled deal from public data, so that when you read a claim about one, you can check the parts that are checkable. Everything here is read-only analysis of public blockchain records.

![You cannot read intent from a transfer, but you can read the route, the timing and the size, and all three are public.](/imgs/blogs/dwf-labs-the-controversial-newcomer-7.webp)

Before the steps, the governing principle, because it is the thing most on-chain commentary gets wrong:

> A blockchain records **what moved, when, and between which addresses**. It does not record why. Every conclusion you draw beyond route, timing and size is inference, and you should label it as such — to yourself first.

### Step 1: fix the deal date and the counterparties

Start with the announcement. Note the exact date the deal was made public and, if disclosed, the date it was signed. These are frequently different, and the gap is informative.

Then identify the wallets. Two labelling services do most of the work:

- **[Arkham Intelligence](https://intel.arkm.com)** maintains entity labels that map addresses to named organisations — exchanges, funds, market makers, project treasuries. Search the project name and the maker's name; you are looking for a wallet tagged as the project's treasury or foundation.
- **[Nansen](https://www.nansen.ai)** maintains its own wallet labels with a different methodology and different coverage. Where the two disagree, that disagreement is itself worth noting.

Labels are inferences, not facts. Both services build them from heuristics plus manual research, and both get things wrong. Treat a label as a strong hint that you then confirm against the transaction history. The general methodology and its failure modes are covered in [Labeling and Attribution](/blog/trading/onchain/labeling-and-attribution).

### Step 2: find the transfer at the deal date

Open the treasury wallet on the relevant block explorer — **[Etherscan](https://etherscan.io)** for Ethereum, or its equivalent on whichever chain the token lives — and filter the token-transfer tab to the window around the deal date.

You are looking for one or a small cluster of large outbound transfers of the project's own token. Three properties to record:

- **Size**, in tokens and in dollars at the price that day.
- **Timing** relative to the announcement. Did the tokens move *before* the public announcement, on the day, or after?
- **Destination**, and whether that destination carries a label.

The CoinDesk reporting on the Synthetix deal is the canonical worked case, and it is worth walking because every element is publicly verifiable. Blockchain data showed DWF received **5.3 million SNX** directly from Synthetix's treasury between **14 and 16 March 2023**, and then transferred all of those tokens to Binance between **16 and 20 March 2023** (CoinDesk, 14 April 2023). The announced headline for that deal was reported in the \$15–20 million range across CoinDesk's 29 March and 14 April 2023 pieces.

Note what that establishes and what it does not. It establishes that tokens moved from treasury to maker to exchange within about six days. It does **not** establish that they were sold: DWF's stated position is that it keeps inventory on exchanges for market-making operations, which is a genuine operational requirement. A deposit is not a sale. It is a *precondition* for a sale, which is a different and much weaker claim.

### Step 3: follow the tokens to their destination

From the maker wallet, the tokens go somewhere. Three destinations, three very different readings.

**3a — a centralised exchange deposit address.** This is the ambiguous one. The tokens are now available to be quoted *or* sold, and you cannot tell which from the chain. What you *can* do is size it: if the deposit is a meaningful fraction of the exchange's typical daily volume in that token, it is large relative to what quoting inventory would require. Quoting a book needs enough inventory to fill the orders that actually arrive, not the entire treasury block.

**3b — a decentralised exchange liquidity pool.** This is the cleanest positive signal. Adding tokens to a DEX liquidity pool is *provable* liquidity provision: the position is on-chain, the depth it creates is measurable, and it cannot be simultaneously used to sell into the same pool. If a maker's deal genuinely was about liquidity, you should find LP positions. Their absence is not proof of anything, but their presence is real evidence for the benign reading.

**3c — a hop to an unlabelled fresh wallet.** Chain of custody breaks here. A transfer into an address with no history and no label is where confident analysis has to stop. Note it, do not extrapolate from it. Fresh wallets are used for a hundred mundane operational reasons — hot/cold segregation, per-venue accounting, key rotation — and reading intent into one is how on-chain analysis gets a bad reputation.

**Cluster visualisation.** [Bubblemaps](https://bubblemaps.io) renders a token's holder set as connected bubbles, sized by balance and linked by transfer history. It is the fastest way to see whether the top holders form one connected cluster or genuinely independent groups. A tight cluster of large holders that all funded from one source is a structural fact worth knowing before you take a position — and it is visible in about thirty seconds.

### Step 4: measure the market effect, do not assume it

Now go back to market data and check whether the liquidity you were promised actually appeared. This is where [Dune](https://dune.com) earns its keep, because it lets you query on-chain data with SQL and chart the result. If you have not written a Dune query before, [Writing On-Chain Queries with Dune](/blog/trading/onchain/writing-onchain-queries-with-dune) walks through the mechanics.

Four measurements, before and after the deal date:

1. **DEX pool depth** for the token's main pairs. Did total value locked in the pools rise around the deal date? If a maker was hired to provide liquidity and the on-chain pools are unchanged, the liquidity is either entirely on centralised venues or it is not there.
2. **Realised spread**, approximated from trade prices. Did the average gap between consecutive buy and sell prints narrow?
3. **Volume composition.** What share of volume is on-chain versus centralised, and did the mix shift?
4. **Holder count and distribution.** Did the number of holders grow — which suggests genuine distribution to new participants — or did volume rise while the holder count stayed flat?

That fourth measurement is the sharpest of the four. **Volume that rises while the holder count is flat is volume that is not reaching new people.** It has innocent explanations (active traders churning, arbitrage between venues) and less innocent ones. It is a question, not an answer — but it is the right question.

### Step 5: write down what you concluded and how confident you are

Force yourself to separate the tiers explicitly. For any deal you trace, you should be able to fill in:

- **Observed (high confidence):** N tokens moved from address A to address B on date D. Address B deposited M tokens to exchange E on date D+k.
- **Inferred (medium confidence):** address B is the maker, based on Arkham and Nansen both labelling it so, and on a transaction history consistent with market-making operations.
- **Speculative (low confidence, label it):** the deposit was for selling rather than quoting.

If your public conclusion lives in the third tier, say so in the same sentence. The reason on-chain analysis has a credibility problem is that too much of it presents tier-three inference in tier-one language.

### Where this goes next in the series

This section is the compressed version. Four later posts in this series take each piece apart properly:

- [Tracing a Market Maker's On-Chain Footprint](/blog/trading/crypto-players/tracing-a-market-makers-onchain-footprint) — the full wallet-identification methodology, including how to separate quoting inventory from directional position.
- [Following Token Flows from Insiders to Exit Liquidity](/blog/trading/crypto-players/following-token-flows-from-insiders-to-exit-liquidity) — tracking supply from treasury and insider allocations all the way to the retail bid.
- [The Price Manipulation Playbook](/blog/trading/crypto-players/the-price-manipulation-playbook) — a defensive taxonomy of the patterns that surveillance teams look for and why each one leaves a signature.
- [Detecting Manipulation On-Chain: Red Flags](/blog/trading/crypto-players/detecting-manipulation-onchain-red-flags) — the concrete checklist, with the false-positive rate of each signal.

## Common misconceptions

**"A market maker buying the token it quotes is obviously improper."** It is not obviously anything. A maker needs inventory to quote; inventory means holding the asset. What matters is *scale and disclosure*: enough inventory to fill arriving orders is operational, and a position several times larger than the daily volume of the pair is something else. The line is real but it is a line of degree, not of kind.

**"If the price pumped after the announcement, that proves manipulation."** No. Announcement-driven attention moves thin-float tokens routinely, with no trading by the announcer at all. The So-Col round trip — up to roughly 3.4 cents within a week of the 28 March 2023 announcement, back toward 1 cent by 4 April 2023 — is exactly what a decayed attention spike looks like. It is *also* consistent with distribution. A chart cannot distinguish them, and anybody who says it can is selling something.

**"Wash trading is easy to spot from public data."** It is close to impossible. Public tape shows trades, not identities. This is why the DWF allegations came from *inside* an exchange, where account ownership is visible, and why they could not be independently confirmed from outside.

**"The \$300 million figure means DWF made \$300 million."** Wash-trade totals are *notional volume* — the sum of the trade sizes — not profit and not net position. A single \$1 million round trip repeated 300 times produces \$300 million of notional with no change in holdings. Notional and profit are different units, and conflating them inflates the story by orders of magnitude.

**"Binance clearing DWF settles the matter."** It settles very little. Binance concluded there was insufficient evidence *for its own internal purposes*, while also being the venue earning fees on the volume in question. That is not an independent adjudication in either direction. Equally, an internal investigations team's conclusion is not a judicial finding. Both institutional facts are weak evidence, and the honest position is that the question remains open.

**"Venture market making is a DWF invention."** The bundling is not new — banks have combined underwriting, market making and proprietary positions for centuries, which is precisely why the separation rules exist. What DWF changed was the *marketing*: making the deal announcement itself a public product. The structure is old; the loudness is new.

## How it shows up in real markets

### The Synthetix flow, March 2023

The clearest documented case of the mechanics, because the on-chain leg is fully public. DWF received 5.3 million SNX directly from Synthetix's treasury between 14 and 16 March 2023 and transferred all of them to Binance between 16 and 20 March 2023 (CoinDesk, 14 April 2023). Headline size was reported in the \$15–20 million range.

Read the two competing interpretations side by side, because this is the template for every case that followed. The **benign** reading: a maker took delivery of inventory and moved it to the venue where it would quote, which is the normal operating sequence and takes about a week. The **critical** reading: tokens acquired at a discount went straight to the venue with the deepest bid. Both readings fit the same public data exactly. The chain records the route; it is silent on the reason. What the episode genuinely establishes is the *speed*: the round trip from treasury to exchange took under a week, which is a different business rhythm from anything a venture fund does.

### So-Col and SIMP, March–April 2023

DWF invested \$1.5 million in SIMP tokens with one-year vesting to February 2024. On-chain records showed DWF received 3.3 million SIMP between 6 and 24 March 2023, transferred 2.6 million to KuCoin, and moved the remainder to an unidentified wallet on 30 March. After the 28 March announcement, SIMP roughly doubled to about 3.4 cents within a week and then fell back toward 1 cent by 4 April 2023 (CoinDesk, 14 April 2023).

The lesson is about *sequence reading*. The tokens arrived across a three-week window; the announcement landed on 28 March; the price round trip completed by 4 April. Anyone holding SIMP who saw only the announcement had a very different picture from someone who had also watched the treasury wallet. The information asymmetry was not secret — it was public on the chain, and almost nobody looked.

### The YGG allegation and its aftermath, 2023–2024

Yield Guild Games is the token named specifically in the WSJ's May 2024 report, which described Binance's investigators concluding that DWF manipulated its price alongside at least six unnamed others. DWF denies this. Binance found the evidence insufficient.

The durable market consequence had nothing to do with YGG's price. It was reputational and structural: after May 2024, "who is your market maker, and what is the deal structure" became a question that serious token holders started asking out loud. Several projects began disclosing MM arrangements voluntarily. That is a real improvement in market hygiene, and it arrived because of a contested allegation rather than a proven one.

### The Liquid Fund and the institutional turn, 2025

In March 2025 DWF announced a \$250 million Liquid Fund for mid- and large-cap projects with \$10–50 million tickets, saying it had deployed over \$11 million in the previous fortnight with \$25 million and \$10 million deals pending. In April 2025 it was reported to have bought \$25 million of World Liberty Financial's WLFI token privately. It also stood behind Falcon Finance, a synthetic-dollar protocol co-founded by Grachev and launched in early 2025.

The pattern to notice is the move up the market-cap curve. Larger, more liquid tokens are structurally harder for any single participant to move — the float is deeper and the holder base is wider. Whether that is a deliberate response to the criticism or simply where the capital naturally went as the firm grew, the effect is the same: the bigger the token, the weaker any single counterparty's price influence.

### The governance story, which outlasted the trading story

Strip the DWF specifics away and the most transferable lesson is about exchanges. A venue that earns fees on volume is a poor policeman of that volume. The reported sequence at Binance — surveillance files an adverse report on a large client, the client-relationship side objects, the surveillance head is dismissed within a week — is the exact conflict that regulated exchanges resolve by making surveillance report outside the revenue line, and by handing enforcement to an external regulator entirely.

That is worth carrying into every venue you use, independent of whether DWF did anything. When you read "the exchange investigated and found nothing," ask who inside the exchange did the investigating, and who they reported to. The incentive map for the whole industry is laid out in [Cui Bono: The Incentive Map of Crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto).

## When this matters to you

If you never touch a mid-cap token, the DWF story is a curiosity. If you do, it changes four concrete habits.

**Before you buy, find out who makes the market.** For any token below roughly \$500 million of circulating market cap, the identity and deal structure of the market maker is a first-order fact about the asset — comparable in importance to the token's unlock schedule. Many projects now disclose it. If yours does not, that silence is information.

**Read the float, not the market cap.** A \$100 million "market cap" with 10% circulating is a \$10 million market. Every deal structure in this post becomes more powerful as the float shrinks, because the same dollar of buying or selling moves a smaller float further. [Follow the Money: Reading a Token's Cap Table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) is the practical version of this check.

**Treat an announced number as a headline, not a fact.** Worked example 2 exists for this. Until you know the discount, the lockup and the drawdown schedule, "a \$40 million investment" tells you almost nothing about capital actually at risk. Ask the founder. Genuinely — projects answer this question far more often than you would expect.

**Check the treasury wallet before you check the chart.** Ten minutes on Arkham and Etherscan around any deal date will tell you more about a token's near-term supply than a week of price analysis. Tokens leaving a treasury are the most reliable early signal of future float that exists, and it is free and public.

And hold the ambiguity honestly. The single most useful skill this whole story teaches is the ability to say: *the structure creates an incentive, the incentive makes the allegation worth investigating, the investigation was contested, and it was never resolved.* That sentence is unsatisfying, and it is true. Most of what you will read about market makers replaces one of those four clauses with a certainty that nobody actually has.

The natural next reads in this series are [How VCs Move Price: Listings, Unlocks and Narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative), which covers the same mechanics from the fund side, and [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move), which builds the order book from scratch if any of section 4 felt fast.

## Sources & further reading

Primary reporting behind the figures and quotations in this post. Where a claim is contested, both the allegation and the response are listed.

- **CoinDesk, "Crypto Market Maker DWF Labs' More Than \$200M in Deals Blur What 'Investing' Means" (14 April 2023).** The foundational reporting on the deal structures, the internal messages describing the OTC menu, the Synthetix and So-Col on-chain flows, and the Virgilli, Grachev and Teng quotations. [coindesk.com](https://www.coindesk.com/business/2023/04/14/market-maker-dwf-labs-more-than-200m-in-deals-blur-what-investing-means)
- **CoinDesk, "Market Maker DWF Labs Emerges as Top Crypto Investor" (29 March 2023).** Early deal list, including the Fetch.ai, Tomi and Synthetix headline numbers, and Grachev on funding the deals from trading profits. [coindesk.com](https://www.coindesk.com/business/2023/03/29/market-maker-dwf-labs-emerges-as-top-crypto-investor)
- **The Wall Street Journal (9 May 2024).** The originating report on Binance's internal investigations team, the \$300 million wash-trade allegation, the YGG finding, and the dismissal of the investigations head. Summarised with the Binance and DWF statements by CoinDesk. [coindesk.com summary](https://www.coindesk.com/business/2024/05/09/binance-fired-investigator-who-uncovered-market-manipulation-at-client-dwf-labs-wsj)
- **DL News, "DWF Labs calls market manipulation claims 'competitor-driven FUD'" (May 2024).** DWF's response, the "competitor-driven FUD" characterisation, and Binance's "insufficient evidence" conclusion. [dlnews.com](https://www.dlnews.com/articles/people-culture/dwf-denies-market-manipulation-wash-trading-on-binance/)
- **DWF Labs statement on X (9 May 2024).** "Unfounded and distort the facts." [x.com/DWFLabs](https://x.com/DWFLabs/status/1788507756326269293)
- **Binance statement on X (9 May 2024).** "We do not tolerate market abuse… nearly 355,000 users… more than \$2.5 trillion." [x.com/binance](https://x.com/binance/status/1788523509209051315)
- **DWF Labs, Liquid Fund launch announcement (24 March 2025).** The \$250 million fund size, the \$10–50 million ticket range, and the deployment figures. [dwf-labs.com](https://www.dwf-labs.com/news/530-liquid-fund-launch-announcement)
- **DWF Labs corporate site**, checked 2026-07-25, for self-reported venue coverage (60+ exchanges) and the two differing portfolio counts. [dwf-labs.com](https://www.dwf-labs.com/)
- **Wikipedia, "DWF Labs."** Founding date, parent company, headquarters, and a summary of the 2024 allegations. Useful as an index to primary sources, not as a source itself. [en.wikipedia.org](https://en.wikipedia.org/wiki/DWF_Labs)

Tools referenced in the tracing section: [Arkham Intelligence](https://intel.arkm.com) and [Nansen](https://www.nansen.ai) for entity labels, [Etherscan](https://etherscan.io) for raw transfers, [Bubblemaps](https://bubblemaps.io) for holder-cluster visualisation, and [Dune](https://dune.com) for querying on-chain data at scale.

*Nothing in this article is a finding of fact about any person or firm, and nothing in it is financial advice. The allegations described are reported allegations that DWF Labs disputes; no public regulatory or judicial finding of manipulation against the firm is known to the author as of 2026-07-27.*
