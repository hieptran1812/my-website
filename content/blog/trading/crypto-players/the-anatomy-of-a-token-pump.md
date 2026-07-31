---
title: "The Anatomy of a Token Pump: How a Coordinated Move Is Actually Assembled"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "A pump is not one act by one villain — it is five ordinary market-structure choices that compose. This is the full mechanical walkthrough: thin float, a market-maker loan-plus-call package, a scheduled listing, a seeded narrative, and an unlock calendar, worked end to end on a hypothetical token, so you can look at a chart and name which part produced each leg."
tags: ["crypto", "crypto-players", "market-structure", "token-unlocks", "float", "fdv", "market-making", "order-book", "slippage", "manipulation", "pump-and-dump", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 58
---

> [!important]
> **TL;DR** — A token pump is not one act by one villain. It is five ordinary market-structure choices — a thin float, a market-maker package, a scheduled listing, a seeded narrative, and an unlock calendar — that compose into a price path with a predictable shape.
>
> - **Float is the denominator.** If only 10% of supply trades, the fully diluted valuation is ten times the money anyone actually paid. In the illustrative token we build below, **\$2.5 million of net buying doubles the price** — half of one percent of its \$500 million FDV headline.
> - **The market maker is paid by the price, not by the quoting.** The standard loan-plus-call package pays a desk a few hundred thousand dollars a year for making markets, and tens of millions if the token triples. That is a description of incentives, not an accusation of intent.
> - **The unlock is the deadline everything else is scheduled against.** A twelve-month cliff releasing 112.5 million tokens drops **\$101 million of newly sellable supply** into a market trading roughly \$3 million a day — thirty-four days of the entire market's turnover, arriving in one morning.
> - **Most "proof of manipulation" proves nothing.** A 300% move, concentrated volume, and a cluster of posts in one week are all equally consistent with genuine demand. Only a handful of signals actually discriminate, and even those need subpoena power, not a chart.
> - **The one number to keep:** in the composite walkthrough, the buyer whose entry was the top lost **74.9%** including slippage while the chart showed **−73.1%**. The gap between those two numbers is the part nobody screenshots.

There is a particular chart shape that anyone who has spent a year in crypto can draw from memory. A new token lists. It doubles in an hour, gives half of it back, then grinds sideways for a fortnight while the chat rooms lose interest. Then it lists somewhere bigger and doubles again. Six weeks later it is up 300%, the timeline is full of people explaining why the protocol is generationally important, and the funding rate on the perpetual future is punitive. Then it stops going up. It fades for months on declining volume, always making lower highs, until one specific morning it gaps down 60% and never really recovers.

The shape is so consistent that people assume it must be one thing — a scheme, a cabal, a room where somebody presses a button. It is more interesting than that, and more useful to understand. Almost every leg of that chart is produced by a separate, individually defensible decision, made by a different party, for a reason that would sound completely reasonable if you heard it in isolation. The founders wanted a low initial float so the token would not be dumped on day one. The market maker wanted to be compensated for taking inventory risk. The exchange wanted to list something with momentum. The fund wanted its research to reach an audience. The vesting schedule was set two years earlier by a lawyer copying a template.

None of those is a crime. Assembled in the right order, with the right timing, they produce the chart anyway.

This post is the assembly manual. We are going to build the whole machine from parts, using a hypothetical token so that every number can be worked out in full without accusing anyone of anything, and then we are going to look at the real public record — actual charged cases, with actual court documents — to see which parts of the machine regulators have successfully proven were operated deliberately, and which parts are simply how the market is built.

By the end you should be able to look at one of these charts and say, out loud, which component produced which leg. That is a genuinely useful skill, and it is mostly arithmetic.

![The five parts of a token pump assembling into one price path](/imgs/blogs/the-anatomy-of-a-token-pump-1.webp)

The diagram above is the mental model for the whole post: five inputs, one output, and two very different sets of people on either end of it. Everything that follows is an elaboration of that picture.

A note on what this post is and is not. It is a mechanical and defensive explanation: how the parts fit, and how to spot the assembly from the outside. It is not a recipe, and where it touches real firms it sticks to what has been formally charged, adjudicated, or reported by name — with the denials attached. It is also not investment advice. It is an attempt to make one specific chart shape legible.

## Foundations: the vocabulary, from zero

If you already know what float, FDV, depth, slippage, and a cliff unlock are, skim this section. If you do not, none of the rest will land, so let us build it properly. Every term here gets used later with real numbers attached.

### Supply, float, and the two different "market caps"

A token has a **total supply** — the number of units that exist or will ever exist. Our hypothetical token, which we will call **NOVA**, has a total supply of **1,000,000,000** (one billion) units. This is written into the contract at launch.

Almost none of those units can be sold on day one. They are distributed across a schedule: some to early investors, some to the team, some held by a foundation, some set aside to pay users for using the protocol. Most of those allocations are **locked** — the tokens exist, but code or contract prevents the holder from moving them until a date arrives.

The portion that *is* freely sellable right now is the **circulating supply**, or the **float**. This is the single most important number in the whole post, and it is the one most often glossed over.

For NOVA, the launch allocation looks like this — a completely ordinary structure you would find in dozens of real projects:

| Allocation | Tokens | Share | Lock |
| --- | --- | --- | --- |
| Public float at launch (airdrop, exchange liquidity, market-maker loan) | 100,000,000 | 10% | none |
| Team | 200,000,000 | 20% | 12-month cliff, then 24-month linear |
| Investors (seed and Series A) | 250,000,000 | 25% | 12-month cliff, then 24-month linear |
| Foundation / treasury | 300,000,000 | 30% | discretionary, governance-controlled |
| Ecosystem incentives | 150,000,000 | 15% | emitted over 48 months |
| **Total** | **1,000,000,000** | **100%** | |

So on launch day, **10% of NOVA trades and 90% does not.**

Now, the two market caps. Suppose NOVA's first traded price is **\$0.50**.

- **Market capitalisation** is price times *circulating* supply: 100,000,000 × \$0.50 = **\$50 million**. This is roughly the amount of money that has actually changed hands to establish this price. It is a receipt.
- **Fully diluted valuation (FDV)** is price times *total* supply: 1,000,000,000 × \$0.50 = **\$500 million**. This is what the entire supply *would* be worth if every locked token could be sold at today's price, right now, without moving it. It is a promise, and it is a promise about a thing that is arithmetically impossible.

The ratio of the two — here **10×** — tells you how much of the valuation is claim rather than cash.

![Market cap versus fully diluted valuation for a token with a ten percent float](/imgs/blogs/the-anatomy-of-a-token-pump-2.webp)

#### Worked example: what FDV actually promises

Take the \$500 million FDV seriously for a moment and ask what would have to be true for it to be realised.

The 900 million locked tokens would each have to find a buyer at \$0.50 or better. That requires **\$450 million of new money** to arrive and buy them — on top of the money already in the float. And that assumes the price stays at \$0.50 while 900 million tokens are sold into the market, which is exactly the assumption that never survives contact with an order book.

Compare that to the actual cash committed: something on the order of the \$50 million market cap, and in practice far less, because most of the float was airdropped rather than purchased.

The intuition this teaches: **FDV is a valuation of supply that does not trade, computed at a price set by supply that does. The smaller the float, the more the headline number is a statement about the future rather than the present.**

If you want the longer version of this argument, [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) works through the ways equity intuitions mislead here, and [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) covers how to actually pull these numbers for a live token.

### The order book, depth, and slippage

A **central limit order book** is just a list. On one side are **bids** — standing offers to buy, each with a price and a size. On the other are **asks** (or offers) — standing offers to sell. The highest bid and the lowest ask are the **best bid and offer**; the gap between them is the **spread**.

When you place a **market order** — "buy \$250,000 of NOVA, now, whatever it costs" — the exchange walks your order up the ask side, filling against the cheapest offers first, then the next cheapest, until your dollars run out. Every price level you consume is a level that no longer exists, which is why the price is higher when you finish than when you started.

Two consequences, and they are the mechanical heart of this entire post:

- **Depth** is how many dollars of resting orders sit within some distance of the current price. "Two-percent depth of \$400,000" means there is \$400,000 of asks between the current price and 2% above it. Depth is a completely different quantity from market cap, and it is the one that actually determines what happens when someone buys.
- **Slippage** is the difference between the price you saw and the average price you got. If NOVA is quoted at \$0.50 and your \$250,000 order fills at an average of \$0.5307, your slippage is 6.1%. That is a real, permanent cost, paid the instant you press the button.

For a much deeper treatment of how the book itself works, [inside an exchange: the matching engine and the order book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book) is the mechanical version, and [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) is the crypto-specific one.

### What a market maker is, and what one is paid

A **market maker** is a firm that continuously quotes both a bid and an ask, so that anyone who wants to trade can. They earn the spread — buying slightly below the mid-price, selling slightly above — and in exchange they take **inventory risk**: at any moment they are holding some quantity of a token they did not want to own, because someone sold it to them.

In equities, a designated market maker is typically paid in rebates and fee discounts. In crypto, the dominant structure for a *new* token is different, and it matters enormously: the issuer **lends** the market maker a block of tokens, and the market maker's compensation includes **call options** on those tokens.

We will work this out in full later. For now, hold the shape: the desk borrows inventory it did not pay for, and holds the right — not the obligation — to buy that inventory at a fixed price later. [What a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) and [the loan-plus-options deal: how market makers get paid](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid) go through the contract mechanics; [designated versus principal market making](/blog/trading/crypto-players/designated-versus-principal-market-making) covers why the distinction between "quoting for a fee" and "trading your own book" is doing so much work.

### Perpetual futures and funding

A **perpetual future** ("perp") is a derivative that tracks a token's price but never expires. Because it never expires, there is nothing forcing it to converge to the spot price, so exchanges bolt on a mechanism: the **funding rate**. Every eight hours, if the perp trades above spot, longs pay shorts a small percentage of their position; if it trades below, shorts pay longs.

Funding is therefore a direct readout of crowding. A funding rate of **+0.25% per eight-hour period** means longs are paying 0.75% a day — about 274% a year if it persisted — for the privilege of being long. Nobody pays that voluntarily unless they expect a very fast move, which is another way of saying: extreme positive funding is the market telling you that the marginal buyer is leveraged and impatient.

### Cliffs, vesting, and the unlock calendar

Locked tokens do not stay locked forever. Two mechanisms release them:

- A **cliff** is a date before which nothing unlocks and on which a chunk unlocks at once. A twelve-month cliff means the holder can sell nothing for a year, and then can sell a large block on the anniversary.
- **Linear vesting** releases a steady stream after the cliff — typically monthly, over one to three years.

The combination — a cliff followed by a linear tail — is the near-universal template. It exists for a good reason: it stops insiders from selling into the launch. But it has a mechanical consequence that is impossible to avoid, which is that it guarantees a **known date on which the float steps up sharply**. Everyone can read that date. It is published. The [lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) walks the whole calendar; here we care about one thing, which is that the date exists and everybody upstream of it knows it.

That is the whole vocabulary. Now let us build the machine.

## Part 1 — Thin float: the fuel-to-air ratio

Every other component in this post is a multiplier on this one. A thin float is what makes small amounts of money produce large price moves, and it is the reason the same nominal buying pressure can do nothing to one token and 300% to another.

The mechanism, stated plainly: **price is set at the margin.** The last trade sets the printed price, and that price is then multiplied by the entire supply to produce a headline number. If the supply that can actually reach the market is small, the amount of money required to set the marginal price is small too — but the headline number it produces is not.

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="The same one million dollars of net buying produces a one percent price move against a forty percent float, two point seven percent against a fifteen percent float, and eight percent against a five percent float" style="width:100%;height:auto;max-width:820px">
<style>
.fd-t{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.fd-s{font:500 11.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.fd-h{font:700 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.fd-v{font:800 17px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.fd-base{stroke:var(--border,#d1d5db);stroke-width:1.5}
.fd-pill{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.2}
.fd-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:50% 100%}
@keyframes fd-grow{0%{transform:scaleY(0.02)}40%,85%{transform:scaleY(1)}100%{transform:scaleY(0.02)}}
@keyframes fd-show{0%,12%{opacity:0}40%,85%{opacity:1}100%{opacity:0}}
.fd-g{animation:fd-grow 7s cubic-bezier(.35,.7,.3,1) infinite}
.fd-f{opacity:0;animation:fd-show 7s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.fd-g{animation:none}.fd-f{animation:none;opacity:1}}
</style>
<text class="fd-t" x="360" y="26">The same &#36;1,000,000 of net buying, three different floats</text>
<text class="fd-s" x="360" y="46">Illustrative toy model: 1,000M supply at &#36;0.50, and eating 1% of the tradeable float value moves the price 2%.</text>
<rect class="fd-pill" x="52" y="66" width="176" height="42" rx="8"/>
<text class="fd-h" x="140" y="84">40% float — 400M NOVA</text>
<text class="fd-s" x="140" y="100">&#36;200M tradeable</text>
<rect class="fd-pill" x="272" y="66" width="176" height="42" rx="8"/>
<text class="fd-h" x="360" y="84">15% float — 150M NOVA</text>
<text class="fd-s" x="360" y="100">&#36;75M tradeable</text>
<rect class="fd-pill" x="492" y="66" width="176" height="42" rx="8"/>
<text class="fd-h" x="580" y="84">5% float — 50M NOVA</text>
<text class="fd-s" x="580" y="100">&#36;25M tradeable</text>
<line class="fd-base" x1="40" y1="272" x2="680" y2="272"/>
<rect class="fd-bar fd-g" x="112" y="252" width="56" height="20" rx="3"/>
<rect class="fd-bar fd-g" x="332" y="218" width="56" height="54" rx="3"/>
<rect class="fd-bar fd-g" x="552" y="112" width="56" height="160" rx="3"/>
<text class="fd-v fd-f" x="140" y="243">+1.0%</text>
<text class="fd-v fd-f" x="360" y="209">+2.7%</text>
<text class="fd-v fd-f" x="580" y="103">+8.0%</text>
<text class="fd-s" x="140" y="292">&#36;1M = 0.5% of the float</text>
<text class="fd-s" x="360" y="292">&#36;1M = 1.3% of the float</text>
<text class="fd-s" x="580" y="292">&#36;1M = 4.0% of the float</text>
<text class="fd-s" x="360" y="312">Headline market cap is identical in all three cases only if you quote FDV. What differs is how much supply can actually be sold.</text>
</svg>
<figcaption>Float is the denominator. The buying is the same in all three columns; the only thing that changes is how much supply is standing there to absorb it. Numbers are illustrative.</figcaption>
</figure>

The animation above uses a deliberately crude toy model — assume the resting depth scales with the float's dollar value, and that consuming 1% of it moves the price 2% — because the point is the *ratio*, not the level. Halve the float and you roughly double the sensitivity. Cut it to an eighth and a million dollars does eight times as much.

Why do issuers choose thin floats? The honest reasons are real. A large day-one float means the airdrop recipients — who paid nothing — can dump the entire supply into a market with no natural buyers, and the token's first week is a straight line down. Locking supply protects the price *and* protects the people who bought at launch. Exchanges also prefer to list tokens whose supply will not be swamped immediately.

The less-discussed reason is that a thin float produces a large FDV, and a large FDV is a marketing asset. It shows up on aggregator pages, in "top 100 by FDV" lists, in the comparison tables in a fundraising deck. It is a number that makes a project look like it is already winning.

Both reasons can be true simultaneously. That is the theme of this post.

#### Worked example: how much money it takes to double NOVA

Here is the concrete version. On listing day, this is the resting sell-side liquidity above \$0.50 — the cumulative dollars of asks you would have to consume to reach each price level:

| Price band | Cumulative asks consumed to reach the top of the band |
| --- | --- |
| \$0.50 → \$0.55 | \$200,000 |
| \$0.55 → \$0.60 | \$450,000 |
| \$0.60 → \$0.70 | \$900,000 |
| \$0.70 → \$0.85 | \$1,600,000 |
| \$0.85 → \$1.00 | \$2,500,000 |

Read the last row again. **To take NOVA from \$0.50 to \$1.00 — to double it — someone has to buy \$2.5 million of it.**

Now compare that \$2.5 million to the numbers the token is described by:

- It is **5%** of the \$50 million market cap.
- It is **0.5%** of the \$500 million FDV.
- It is roughly the size of a single mid-sized allocation in the seed round.

And what does the doubling produce? At \$1.00, NOVA's FDV is now **\$1 billion**. So \$2.5 million of buying — which one participant could do alone, without borrowing, without coordinating with anyone — has manufactured a **\$500 million increase in the headline valuation**.

The intuition this teaches: **on a thin float, the headline valuation is a lever with a five-hundred-to-one ratio. You do not need a conspiracy to move it. You need a few million dollars and a book this thin.**

Read that last sentence carefully, because it cuts both ways. It means a coordinated actor does not need much capital. It also means an entirely genuine buyer — a fund taking a real position, a wave of real users — produces exactly the same chart. We will come back to this.

## Part 2 — The market maker's book: who is actually quoting

There is a widespread and wrong mental model in which the market maker is a neutral utility, like a plumber. In crypto, for a newly launched token, the market maker is a **principal** with a large, asymmetric, and entirely disclosed-to-the-issuer position in the outcome. Understanding that position is the single highest-leverage thing in this post, because it explains why the book behaves the way it does in each phase.

### The loan-plus-call structure

The issuer has a problem: on day one nobody will quote a two-sided market in a token that has never traded, because the risk of being the only buyer is unbounded. The issuer also has an asset: tokens, which cost nothing to create.

So the deal is: **the issuer lends the market maker tokens, and pays the market maker in call options on those same tokens.**

For NOVA, an illustrative deal — and these terms are in the range of what has been publicly described for real token launches, though every contract differs:

- **The loan:** 20,000,000 NOVA (2% of total supply), for twelve months. The desk can trade this inventory freely; it must return 20,000,000 NOVA (or settle in cash) at the end.
- **The obligations:** quote both sides on three named venues, spread no wider than 50 basis points (0.50%), at least \$100,000 of depth on each side within 2% of mid, with 95% uptime.
- **The compensation:** call options on the borrowed tokens, in three tranches:
  - **Tranche A** — 7,000,000 tokens, strike **\$0.75**
  - **Tranche B** — 7,000,000 tokens, strike **\$1.20**
  - **Tranche C** — 6,000,000 tokens, strike **\$2.00**

A **call option** gives its holder the right, but not the obligation, to buy at the strike price. If NOVA ends at \$1.50, Tranche A's holder can buy 7 million tokens for \$0.75 each and immediately sell them for \$1.50 — a profit of \$0.75 per token. If NOVA ends at \$0.40, all three tranches are worthless and simply expire. If you have not met options before, [calls, puts and the payoff diagram](/blog/trading/options-volatility/calls-puts-and-the-payoff-diagram-the-language-of-options) is the ground-up version.

![The market maker's option payoff against the token price, with three strikes marked](/imgs/blogs/the-anatomy-of-a-token-pump-4.webp)

#### Worked example: what the desk earns at each price

Let us price the package at expiry. The payoff on each tranche is (number of tokens) × (price − strike), floored at zero.

| NOVA at expiry | Tranche A (7M @ \$0.75) | Tranche B (7M @ \$1.20) | Tranche C (6M @ \$2.00) | **Total option payoff** |
| --- | --- | --- | --- | --- |
| \$0.40 | \$0 | \$0 | \$0 | **\$0** |
| \$0.75 | \$0 | \$0 | \$0 | **\$0** |
| \$1.00 | 7M × \$0.25 = \$1,750,000 | \$0 | \$0 | **\$1,750,000** |
| \$1.20 | 7M × \$0.45 = \$3,150,000 | \$0 | \$0 | **\$3,150,000** |
| \$1.50 | 7M × \$0.75 = \$5,250,000 | 7M × \$0.30 = \$2,100,000 | \$0 | **\$7,350,000** |
| \$2.00 | 7M × \$1.25 = \$8,750,000 | 7M × \$0.80 = \$5,600,000 | \$0 | **\$14,350,000** |
| \$3.00 | 7M × \$2.25 = \$15,750,000 | 7M × \$1.80 = \$12,600,000 | 6M × \$1.00 = \$6,000,000 | **\$34,350,000** |

Now compare that column to the desk's income from *actually doing the job*. Suppose NOVA trades \$3 million a day, the desk is on one side of 30% of it, and captures an average of 15 basis points net of hedging. That is \$3,000,000 × 0.30 × 0.0015 ≈ **\$1,350 a day**, or roughly **\$0.5 million a year** — and that is a generous assumption; on a quieter token it is a fraction of that.

So the quoting income is **half a million dollars**, and the option leg is worth **\$7.35 million if NOVA reaches \$1.50** and **\$34.35 million if it reaches \$3.00**. The compensation for the service is somewhere between one and seven percent of the compensation for the price going up.

The intuition this teaches: **the market maker's contract makes them long the outcome, not neutral to it. Everything the desk is nominally paid for is a rounding error next to the thing it is not nominally paid for.**

### What this does and does not imply

Be precise here, because this is exactly where careless writing turns market structure into an accusation.

What it **does** imply, mechanically:

- The desk's quoting behaviour is not symmetric in its own interest. Providing thick support on the bid — absorbing sellers — protects a position it wants to appreciate. Providing thick offers on the ask — capping rallies — works against it.
- The desk knows the unlock calendar, the listing pipeline, and often the announcement schedule, because it needs them to manage inventory. That is legitimately necessary information which is also, unavoidably, valuable information.
- Because the desk borrowed the tokens rather than buying them, its downside is capped in a way an ordinary holder's is not. It can be wrong about the price and still return the loan.

What it **does not** imply:

- That any particular desk trades on that information, or quotes asymmetrically, or does anything other than run a delta-hedged book. A desk can and often does hedge the option leg — selling perps against it — which neutralises much of the directional exposure. Reputable firms maintain information barriers between the market-making desk and any proprietary book.
- That options-based compensation is improper. It is a rational solution to a real problem: nobody will provide liquidity in an untraded asset for a flat fee, and an option is the natural instrument for paying someone whose value to you is contingent.

The honest statement is about incentives, not conduct: **this contract structure pays the liquidity provider more when the price is higher, and the liquidity provider is the party that decides how much liquidity there is.** What any individual firm does with that is a question for evidence, not inference — and, as we will see, sometimes evidence exists.

For how this plays out across the actual named firms, [Wintermute: the algorithmic powerhouse](/blog/trading/crypto-players/wintermute-the-algorithmic-powerhouse), [GSR, Cumberland and the established OTC desks](/blog/trading/crypto-players/gsr-cumberland-and-the-established-otc-desks) and [DWF Labs: the controversial newcomer](/blog/trading/crypto-players/dwf-labs-the-controversial-newcomer) cover the range of models and the range of controversy. [Inventory risk, hedging and delta neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality) is the technical counterweight to the cynical reading.

## Part 3 — The listing: a scheduled attention event with a thin book

A listing is the only component of this machine that is a *scheduled public event*. That is what makes it structurally special: it is a moment when attention and order flow arrive at a predictable time, into a book that is at its thinnest.

Three things happen at once when a token lists on a venue with real users:

1. **Discovery.** Hundreds of thousands of people who could not previously buy the token now see it on a screen they already look at. Many exchanges surface new listings prominently.
2. **Access.** Buying goes from "bridge to a chain, get a wallet, find the pool, accept the slippage" to "tap the button". The activation energy collapses.
3. **Legitimacy.** A listing on a large, compliance-heavy venue is read by many participants as a due-diligence signal, whether or not the exchange intends it that way.

Meanwhile the book on day one is thin, because there has been no time for a natural distribution of holders to build up limit orders, and because a large fraction of the sellable supply is either in the market maker's hands or in the hands of airdrop recipients who have already sold.

Attention at its maximum, depth at its minimum. That is the listing.

![The listing-day ask ladder and where a quarter-million-dollar market buy stops](/imgs/blogs/the-anatomy-of-a-token-pump-3.webp)

#### Worked example: what a \$250,000 market buy actually does on day one

Take the NOVA ask ladder from Part 1 and send a \$250,000 market buy into it.

**Step 1 — the first band.** The \$0.50–\$0.55 band holds \$200,000 of asks. Your order consumes all of it. Filling uniformly across that band, your average price in it is about **\$0.525**, so you receive:

\$200,000 ÷ \$0.525 = **380,952 NOVA**

**Step 2 — the second band.** You have \$50,000 left. The \$0.55–\$0.60 band holds \$250,000 (that is \$450,000 cumulative minus the \$200,000 already gone). Your \$50,000 is 20% of it, so you eat one-fifth of the way through, taking the price from \$0.55 to about **\$0.56**. Your average price in this band is about **\$0.555**:

\$50,000 ÷ \$0.555 = **90,090 NOVA**

**Step 3 — the result.**

- Tokens received: 380,952 + 90,090 = **471,042 NOVA**
- Dollars spent: **\$250,000**
- Average price paid: \$250,000 ÷ 471,042 = **\$0.5307**
- **Slippage versus the \$0.50 you saw: 6.1%**
- **Last printed price: \$0.56 — the chart now shows +12%**

Two things fall out of this that are worth stopping on.

First, **the chart moved twice as much as your cost.** The screen says the token is up 12%. You are up 5.5% (you paid \$0.5307 and it is now \$0.56). Everyone who watched from outside sees a 12% move; only you know it cost 6.1% to make it happen.

Second, **you now own 471,042 NOVA and you cannot sell it back for anything like \$250,000.** The bid side is typically thinner than the ask side on a new listing — the market maker is more willing to offer inventory it borrowed than to bid for inventory it would have to fund. Selling the position straight back would walk down the bid ladder and realise a loss well beyond the spread. Round-tripping a position on a thin book is expensive in both directions, and the second direction is the one nobody models.

#### Worked example: the cost of moving the price 1%

The most useful single statistic for a token, and one almost nobody quotes, is: **how much net buying does it take to move this 1%?**

For NOVA on listing day, the first band gives us the answer directly. \$200,000 of asks span a 10% move, so — spread evenly — roughly **\$20,000 moves the price 1%.**

![What it costs to move the price one percent, across four levels of depth](/imgs/blogs/the-anatomy-of-a-token-pump-5.webp)

Set that next to tokens at other points on the liquidity spectrum. The figures below are illustrative and chosen to show the shape of the distribution, not to describe any specific asset:

| Token | Net buying to move price 1% |
| --- | --- |
| NOVA, listing day | \$20,000 |
| NOVA, month 2 (book has widened) | \$180,000 |
| An established mid-cap token | \$1,500,000 |
| A top-ten token | \$12,000,000 |

The ratio between the first and last rows is **600×**. Two tokens can carry a similar FDV on a screen and differ by nearly three orders of magnitude in what it costs to move them. Market cap tells you nothing about this. Only depth does.

The intuition this teaches: **"market cap" measures the size of the claim; depth measures the size of the market. A pump is what happens when the first number is large and the second is small.**

This is also, incidentally, why the same mechanic produces profitable arbitrage rather than pumps when the book is deep — see [cross-exchange arbitrage and the latency game](/blog/trading/crypto-players/cross-exchange-arbitrage-and-the-latency-game) for the other side of the coin, and [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues) for why the listing decision itself is a commercial act rather than a neutral one.

## Part 4 — The narrative: manufacturing a reason to buy

Thin float and a listing explain *how* a price can move. They do not explain *why anyone shows up*. That is the narrative's job, and it is the least mechanical and most misunderstood component.

A token has no cash flows to discount. There is no earnings number, no dividend, no book value — [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) works through the consequences of that in detail. What a token has instead is a **story about future adoption**, and the price is whatever the marginal buyer's belief in that story supports. This is not unique to crypto — early-stage equity works similarly — but crypto is unusual in that the story is priced continuously, in public, by anyone with a phone.

That makes attention a genuine input to price, which in turn makes attention a purchasable input to price.

![How a thesis travels from an issuer to a retail timeline through four intermediaries](/imgs/blogs/the-anatomy-of-a-token-pump-6.webp)

### The distribution chain and where disclosure evaporates

The chain has four hops, and the crucial property is that **provenance degrades at every one of them**:

1. **The issuer or its investor publishes a thesis.** A fund writes up why it led the round; the project publishes a roadmap. This is disclosed by construction — the fund's name is on it, and everyone understands they own the token.
2. **Tier-one accounts amplify it.** Large accounts with real followings post threads. Some disclose a paid relationship or an allocation. Some do not. In the United States the relevant rule is the anti-touting provision, [Section 17(b) of the Securities Act](https://www.law.cornell.edu/uscode/text/15/77q) (15 U.S.C. § 77q(b)), and it is stricter than most people assume: it makes it unlawful to publish anything describing a security "for a consideration received or to be received, directly or indirectly, from an issuer, underwriter, or dealer, without fully disclosing the receipt … **and the amount thereof**." Not merely "I was paid" — how much. That provision only bites where the asset is a security and the promoter is within reach of a US court, which between them exclude a great deal of what actually happens.
3. **Tier-two accounts and group chats relay it.** By this point the thread is being quoted by people who genuinely believe it, alongside people who were paid to relay it, and the two are indistinguishable from outside.
4. **Aggregators and "trending" surfaces pick it up.** A token appears on a trending list because of engagement volume. Engagement volume is a function of steps 2 and 3. The trending list then generates more engagement.

By the time a thesis reaches an ordinary timeline, **whoever paid for it is four hops behind and structurally invisible.** The reader sees consensus. What produced the consensus was distribution.

None of this requires anybody to lie. The most effective version of this chain runs on people who sincerely believe the thing they are amplifying — belief is more persuasive than performance, and it is free. [Influencers, KOLs and the narrative-for-hire machine](/blog/trading/crypto-players/influencers-kols-and-the-narrative-for-hire-machine) is the full treatment of how that market is actually priced and structured; [how VCs move price: listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative) covers the fund side of it. If you want the theory of why a crowd converges on a story that nobody independently verified, [information cascades and herding](/blog/trading/game-theory/information-cascades-and-herding-when-rational-traders-follow-the-crowd) and [reflexivity: markets that watch themselves](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves) are the two pieces of game theory that explain it best.

### Why narrative timing matters more than narrative content

The content of the story is almost irrelevant to the mechanics. What matters is *when* it arrives.

A story that lands while the float is thin and the book is shallow converts a modest amount of buying into a large price move — and the large price move then becomes evidence for the story. That is the reflexive loop, and it runs in both directions. The same story landing six months later, into a float three times larger and a book ten times deeper, does very little.

This is why the sequencing in the composite walkthrough below is not arbitrary. Every component is timed relative to every other one, and the whole schedule points at one date.

## Part 5 — The clock: the unlock is the deadline

Here is the component that turns a set of separate decisions into a schedule.

Locked tokens become sellable on a published date. That date is not a risk in the ordinary sense — it is not uncertain. Everyone can look it up. What makes it powerful is not surprise but **magnitude**: the amount of supply arriving relative to the amount of buying the market can generate.

![The NOVA unlock calendar and the size of the cliff relative to daily volume](/imgs/blogs/the-anatomy-of-a-token-pump-7.webp)

#### Worked example: how big is the NOVA cliff, really

NOVA's team and investors together hold **450,000,000 tokens** (200M + 250M) under a twelve-month cliff followed by twenty-four months of linear vesting. Take a common structure: **25% of the locked amount releases at the cliff, and the remaining 75% drips monthly over the following two years.**

- **At the cliff:** 450,000,000 × 25% = **112,500,000 NOVA** becomes sellable in one day.
- **Monthly thereafter:** (450,000,000 − 112,500,000) ÷ 24 = **14,062,500 NOVA per month**, every month, for two years.

Now put that in dollars, at an illustrative month-twelve price of **\$0.90**:

- **Cliff value:** 112,500,000 × \$0.90 = **\$101,250,000**
- **Monthly drip value:** 14,062,500 × \$0.90 = **\$12,656,250 per month**

And now the comparison that actually matters. Suppose NOVA's reported volume at month twelve is \$8 million a day, of which — being conservative about how much reported crypto volume is real — perhaps **\$3 million a day is genuine two-sided flow.**

**Days to absorb the cliff, if every single dollar of real volume were a buyer taking supply:**

\$101,250,000 ÷ \$3,000,000 = **33.75, call it 34 days**

Thirty-four days of the *entire market's turnover*, every trade a buy, just to clear the cliff. And of course a market where 100% of flow is buying does not exist. If newly unlocked supply can be absorbed at 10–20% of daily volume without pushing the price down, the realistic absorption window is:

\$101,250,000 ÷ (\$3,000,000 × 0.20) = **169 days** at the optimistic end
\$101,250,000 ÷ (\$3,000,000 × 0.10) = **338 days** at the pessimistic end

**Between six months and a year of continuous overhang, from one morning's unlock.**

And the drip that follows is not small either. \$12.66 million a month against roughly \$90 million of monthly volume is **14% of all turnover**, arriving as supply, every month, for twenty-four months.

There is one more number. The float before the cliff has grown from 100 million to roughly **160 million** through ecosystem emissions. The cliff takes it to **272,500,000** — an increase of **70% in a single day**.

The intuition this teaches: **an unlock is not a sentiment event, it is an arithmetic one. The float steps up by a known percentage on a known date, and the price has to find a new level at which the larger float clears. Everything upstream — the listing, the narrative, the market-making support — happens in the window before that date.**

That is worth stating without euphemism. It is not that a cliff *causes* the earlier legs. It is that **the cliff defines the interval in which anything the earlier legs achieve can still be converted into cash by the people who hold locked supply.** A holder who cannot sell for twelve months has a twelve-month problem: they need liquidity and a price to exist on the day their tokens arrive. Every component in Parts 1 through 4 helps produce both.

Nothing about that requires anyone to break a rule. It requires only that people respond to a deadline they can all read.

For the deeper version of this, the sibling post [unlock cliffs and the supply-overhang trade](/blog/trading/crypto-players/unlock-cliffs-and-the-supply-overhang-trade) works through how the overhang is traded rather than just measured, and [the low float, high FDV game](/blog/trading/crypto-players/the-low-float-high-fdv-game) covers the launch-design side.

## The composite walkthrough: eighteen months of NOVA

Now we assemble all five components on one timeline. Everything below is **hypothetical and illustrative** — NOVA is not a real token, no real firm is described, and every number is arithmetic rather than reportage. The point is that each step is individually ordinary.

<figure class="blog-anim">
<svg viewBox="0 0 760 360" role="img" aria-label="An illustrative token price path assembles in five phases: a listing pop from a thin float, a grind higher on market-maker quotes, a narrative-driven blow-off top, a fade as supply is distributed into strength, and a gap down when the cliff unlock lands" style="width:100%;height:auto;max-width:860px">
<style>
.pa-bg{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.pa-grid{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:3 5;opacity:.7}
.pa-t{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.pa-s{font:500 11.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.pa-sm{font:500 11.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.pa-cap{font:700 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.pa-leg{fill:none;stroke:var(--accent,#6366f1);stroke-width:3;stroke-linecap:round;stroke-linejoin:round}
.pa-leg5{fill:none;stroke:#dc2626;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}
.pa-chip{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.2}
.pa-litbox{fill:var(--accent,#6366f1);opacity:0}
.pa-chiptx{font:600 11.5px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.pa-l1{stroke-dasharray:145 145;stroke-dashoffset:145;animation:pa-d1 15s linear infinite}
.pa-l2{stroke-dasharray:148 148;stroke-dashoffset:148;animation:pa-d2 15s linear infinite}
.pa-l3{stroke-dasharray:172 172;stroke-dashoffset:172;animation:pa-d3 15s linear infinite}
.pa-l4{stroke-dasharray:136 136;stroke-dashoffset:136;animation:pa-d4 15s linear infinite}
.pa-l5{stroke-dasharray:158 158;stroke-dashoffset:158;animation:pa-d5 15s linear infinite}
@keyframes pa-d1{0%{stroke-dashoffset:145}18%,97%{stroke-dashoffset:0}100%{stroke-dashoffset:145}}
@keyframes pa-d2{0%,20%{stroke-dashoffset:148}38%,97%{stroke-dashoffset:0}100%{stroke-dashoffset:148}}
@keyframes pa-d3{0%,40%{stroke-dashoffset:172}58%,97%{stroke-dashoffset:0}100%{stroke-dashoffset:172}}
@keyframes pa-d4{0%,60%{stroke-dashoffset:136}78%,97%{stroke-dashoffset:0}100%{stroke-dashoffset:136}}
@keyframes pa-d5{0%,80%{stroke-dashoffset:158}96%,97%{stroke-dashoffset:0}100%{stroke-dashoffset:158}}
@keyframes pa-k1{0%,19%{opacity:.20}21%,100%{opacity:0}}
@keyframes pa-k2{0%,19%{opacity:0}21%,39%{opacity:.20}41%,100%{opacity:0}}
@keyframes pa-k3{0%,39%{opacity:0}41%,59%{opacity:.20}61%,100%{opacity:0}}
@keyframes pa-k4{0%,59%{opacity:0}61%,79%{opacity:.20}81%,100%{opacity:0}}
@keyframes pa-k5{0%,79%{opacity:0}81%,99%{opacity:.20}100%{opacity:0}}
@keyframes pa-c1{0%,19%{opacity:1}21%,100%{opacity:0}}
@keyframes pa-c2{0%,19%{opacity:0}21%,39%{opacity:1}41%,100%{opacity:0}}
@keyframes pa-c3{0%,39%{opacity:0}41%,59%{opacity:1}61%,100%{opacity:0}}
@keyframes pa-c4{0%,59%{opacity:0}61%,79%{opacity:1}81%,100%{opacity:0}}
@keyframes pa-c5{0%,79%{opacity:0}81%,99%{opacity:1}100%{opacity:0}}
.pa-b1{animation:pa-k1 15s linear infinite}
.pa-b2{animation:pa-k2 15s linear infinite}
.pa-b3{animation:pa-k3 15s linear infinite}
.pa-b4{animation:pa-k4 15s linear infinite}
.pa-b5{animation:pa-k5 15s linear infinite}
.pa-x1{opacity:0;animation:pa-c1 15s linear infinite}
.pa-x2{opacity:0;animation:pa-c2 15s linear infinite}
.pa-x3{opacity:0;animation:pa-c3 15s linear infinite}
.pa-x4{opacity:0;animation:pa-c4 15s linear infinite}
.pa-x5{opacity:0;animation:pa-c5 15s linear infinite}
@media (prefers-reduced-motion:reduce){.pa-l1,.pa-l2,.pa-l3,.pa-l4,.pa-l5{animation:none;stroke-dashoffset:0}.pa-b1,.pa-b2,.pa-b3,.pa-b4,.pa-b5{animation:none;opacity:.14}.pa-x1,.pa-x2,.pa-x3,.pa-x4,.pa-x5{animation:none;opacity:0}.pa-x3{opacity:1}}
</style>
<text class="pa-t" x="380" y="24">NOVA (illustrative) — the price path assembles one phase at a time</text>
<line class="pa-grid" x1="70" y1="76" x2="710" y2="76"/>
<line class="pa-grid" x1="70" y1="210" x2="710" y2="210"/>
<text class="pa-s" x="716" y="80">&#36;2.20</text>
<text class="pa-s" x="716" y="214">&#36;0.50</text>
<line class="pa-bg" x1="70" y1="258" x2="710" y2="258"/>
<line class="pa-bg" x1="70" y1="52" x2="70" y2="258"/>
<polyline class="pa-leg pa-l1" points="70,210 110,177 150,196 196,190"/>
<polyline class="pa-leg pa-l2" points="196,190 250,183 280,159 322,127"/>
<polyline class="pa-leg pa-l3" points="322,127 380,76 410,110 448,139"/>
<polyline class="pa-leg pa-l4" points="448,139 500,151 560,167 574,179"/>
<polyline class="pa-leg5 pa-l5" points="574,179 600,179 615,222 700,220"/>
<text class="pa-cap pa-x1" x="380" y="286">Phase 1 — the listing pop: a thin book meets scheduled attention</text>
<text class="pa-cap pa-x2" x="380" y="286">Phase 2 — the grind: spreads tighten, the story gets refreshed</text>
<text class="pa-cap pa-x3" x="380" y="286">Phase 3 — the blow-off: retail chases, perp funding turns punitive</text>
<text class="pa-cap pa-x4" x="380" y="286">Phase 4 — the fade: early supply is distributed into strength</text>
<text class="pa-cap pa-x5" x="380" y="286">Phase 5 — the cliff: locked supply unlocks, price re-rates to the real float</text>
<rect class="pa-chip" x="46" y="308" width="128" height="34" rx="8"/>
<rect class="pa-litbox pa-b1" x="46" y="308" width="128" height="34" rx="8"/>
<text class="pa-chiptx" x="110" y="330">Thin float + listing</text>
<rect class="pa-chip" x="184" y="308" width="128" height="34" rx="8"/>
<rect class="pa-litbox pa-b2" x="184" y="308" width="128" height="34" rx="8"/>
<text class="pa-chiptx" x="248" y="330">Market-maker quotes</text>
<rect class="pa-chip" x="322" y="308" width="128" height="34" rx="8"/>
<rect class="pa-litbox pa-b3" x="322" y="308" width="128" height="34" rx="8"/>
<text class="pa-chiptx" x="386" y="330">Seeded narrative</text>
<rect class="pa-chip" x="460" y="308" width="128" height="34" rx="8"/>
<rect class="pa-litbox pa-b4" x="460" y="308" width="128" height="34" rx="8"/>
<text class="pa-chiptx" x="524" y="330">Distribution</text>
<rect class="pa-chip" x="598" y="308" width="128" height="34" rx="8"/>
<rect class="pa-litbox pa-b5" x="598" y="308" width="128" height="34" rx="8"/>
<text class="pa-chiptx" x="662" y="330">Unlock cliff</text>
<text class="pa-sm" x="380" y="272">Phases are evenly spaced for legibility; the real elapsed time is roughly twelve months.</text>
</svg>
<figcaption>The same chart, assembled in order. Each leg is drawn by a different part of the machine — and the part that draws it is highlighted below. Numbers are illustrative.</figcaption>
</figure>

### Month −6 to 0: the parts are ordered

The private rounds close. Seed investors buy at **\$0.02**, a Series A at **\$0.09**. Both allocations vest on the twelve-month cliff described above. The valuation in the Series A round implies an FDV of \$90 million — a number that will be quoted repeatedly as evidence that a \$500 million listing valuation is a bargain, which is a comparison between two things that are not comparable.

The market-making agreement is signed: 20 million tokens lent, three option tranches struck at \$0.75, \$1.20 and \$2.00. Nobody involved is doing anything unusual; this is roughly the standard package.

Listing conversations begin with two tier-two exchanges for day zero and one tier-one exchange for a few weeks later. Staggering listings is normal practice — it gives the token a second scheduled catalyst, and it lets the larger venue see how the asset trades before committing.

Research goes out. The lead investor publishes its thesis. This is disclosed, signed, and entirely proper.

### Day 0: the listing pop

NOVA opens at **\$0.50** on the tier-two venues. Float is 100 million; market cap \$50 million; FDV \$500 million.

Within forty minutes, roughly **\$2.0 million** of net buying arrives — airdrop farmers who want more, funds that missed the round, and the ordinary flow that any new listing generates. Against the ladder in Part 3, \$2.0 million consumes everything up to about **\$0.92** (the \$1.6 million band takes you to \$0.85; the next \$0.4 million carries you roughly half-way through the band above it).

Then it settles back to **\$0.68**, because the first wave of airdrop recipients sells into the strength. This is the healthiest thing that happens in the whole eighteen months — real supply meeting real demand and finding a level.

**Component responsible: thin float + fresh listing.**

### Weeks 1–3: the grind

Volume falls by 80%. The market maker's quotes tighten from an initial 200 basis points to the contracted 50. Price drifts from \$0.68 to **\$0.85** on genuinely small volume, because a tighter spread and a slightly thicker bid mean the same small buy flow does more.

To a chart reader this looks like accumulation, and in a sense it is — it is just that most of what is being accumulated is patience.

**Component responsible: market-maker support.**

### Week 3: the tier-one listing

The larger exchange lists. Price goes from \$0.85 to **\$1.15** within the hour, on flow that is genuinely new — an entirely different set of users can now buy with one tap.

Note where \$1.15 sits: comfortably above Tranche A's \$0.75 strike. The desk's first option tranche is now roughly \$2.8 million in the money.

**Component responsible: fresh listing (again) + thin float.**

### Weeks 3–7: the narrative phase

The story compounds. A partnership is announced. Protocol metrics — which are real, and which are also heavily influenced by the ecosystem incentives being emitted — are posted weekly. Tier-one accounts write threads. Tier-two accounts relay them.

Price grinds \$1.15 → **\$1.55**. Perpetual funding turns persistently positive, running around **+0.10% per eight hours** — roughly 0.30% a day, or about 110% annualised. Leverage is building.

**Component responsible: seeded narrative.**

### Week 8: the blow-off

On a Friday, NOVA goes parabolic to **\$2.20**. Funding hits **+0.25% per eight hours** — 0.75% a day, an annualised 274%. Everyone who is long is paying a great deal to stay long, which is a precise statement that the marginal buyer has run out of patience.

Then it reverses 30% intraday and closes near **\$1.40**.

At \$2.20 all three of the desk's option tranches are in the money, the third only barely. Extending the table above: 7M × \$1.45 + 7M × \$1.00 + 6M × \$0.20 = **\$18.35 million** on paper.

FDV is now **\$2.2 billion**, up 4.4× from the listing — produced by a float that has grown by less than a fifth and a book that was never deep enough to require much absorbing. The first double alone, remember, cost \$2.5 million.

**Component responsible: narrative + leverage, on a float that has barely grown.**

### Months 2–11: the fade

Lower highs, on declining volume, for nine months. \$1.40 → \$1.25 → \$1.05 → **\$0.90**.

This is the phase people find hardest to read, because nothing dramatic happens in it. What is happening is that the ecosystem emissions are steadily adding float — 100 million to roughly 160 million — while the buying that arrived during the narrative phase is not being replenished. Supply grows; demand does not. The price does the only thing it can.

**Component responsible: supply distribution meeting exhausted demand.**

### Month 12: the cliff

The unlock lands. 112.5 million tokens become sellable in one morning. Float goes from 160 million to **272.5 million — up 70%.**

Price gaps from **\$0.90 to \$0.35** and does not recover. That is **−61% in a day**, and it is not a crash in any meaningful sense. It is a repricing: the market discovers what the token is worth when the float is what it actually is, rather than what the lock schedule was pretending it was.

Note that not one seller needs to be malicious for this to happen. If holders of 112.5 million tokens sell even 15% of their position in the first week, that is 16.9 million tokens — \$15 million of supply against a market that trades \$3 million a day. The arithmetic does the rest.

**Component responsible: the unlock clock.**

#### Worked example: the eighteen-month P&L, by participant

Who ends up with what? Take the same 100 tokens' worth of exposure for each participant and follow it through.

| Participant | Entry | Exit | Return |
| --- | --- | --- | --- |
| Seed investor | \$0.02 (locked 12 months) | sells the cliff tranche into \$0.35–0.90 | **17× to 45×** |
| Series A investor | \$0.09 | same window | **4× to 10×** |
| Market maker | borrowed at \$0, options struck \$0.75/\$1.20/\$2.00 | exercises what is in the money | **\$0 to \$18.35M**, entirely depending on the price path |
| Airdrop recipient who sold day one | \$0 | \$0.68 | **all upside, no capital at risk** |
| Buyer at the listing pop | \$0.92 | \$0.35 | **−62%** |
| Buyer at the blow-off | \$1.30 | \$0.35 | **−73%** |

Look at the top rows and the bottom rows. Both are real returns on real positions. The difference between them is not skill, information, or conviction. It is **cost basis and the ability to sell.**

The intuition this teaches: **in a structure where 90% of the supply is issued at or near zero cost and released on a schedule, the distribution of outcomes is decided before the token trades. The chart is where that distribution gets expressed, not where it gets determined.**

## How it shows up in price: naming every leg

Now put the whole thing on one chart and label it.

![The eighteen-month NOVA price path with each leg annotated to its mechanism](/imgs/blogs/the-anatomy-of-a-token-pump-8.webp)

Here is the reading key, which is the practical takeaway of the post:

| What you see | What produced it | The tell |
| --- | --- | --- |
| **Leg 1** — a violent move in the first hours, then a 25% retrace | Thin book meeting scheduled attention; airdrop supply selling into it | The retrace is as fast as the move. Volume is enormous relative to everything that follows. |
| **Leg 2** — a quiet drift up on falling volume | Spreads tightening; the same small flow doing more work | Price rises while volume falls. This is the signature. |
| **Leg 3** — a parabolic move with a fast reversal | Leveraged narrative-chasing on a float that has not grown | Perp funding goes extreme; open interest rises faster than spot volume. |
| **Leg 4** — months of lower highs on declining volume | Emissions adding float; earlier demand not replenished | Each rally reaches a lower price on less volume than the last. |
| **Leg 5** — a single-day gap down that never recovers | The cliff unlock; float steps up by a known percentage | It happens on a date you could have read a year in advance. |

Two of these are worth extra attention, because they are the ones most commonly misread.

**Leg 2 is the most misread leg on any token chart.** Price rising on falling volume feels bullish — it looks like supply has been absorbed and the token is "coiling". Mechanically it often means the opposite: the book has thinned to the point where small flow moves it, and the participants doing the moving are the ones contractually obliged to be there. It is not evidence of demand. It is evidence of *low resistance*, which is a different thing and sometimes the reverse.

**Leg 5 is the most predictable event in crypto and is still repeatedly treated as a surprise.** Unlock dates are published. Unlock sizes are published. The float step-up is arithmetic. Whether the market has priced it in ahead of time is a genuinely open question — sometimes it clearly has, sometimes it clearly has not — but the *event* is never a surprise, and treating it as one is a choice.

#### Worked example: the retail round trip, including the part nobody screenshots

Finally, the number that matters most to an individual. Suppose you bought \$10,000 of NOVA during the blow-off, at a quoted \$1.30. Because you were chasing into a fast market, you paid about 3% slippage, so your average fill was **\$1.339**:

\$10,000 ÷ \$1.339 = **7,468 NOVA**

Now it is month twelve, the cliff has landed, and NOVA is \$0.35. On paper your position is worth 7,468 × \$0.35 = **\$2,614**. But selling it is not free. The bid side after an unlock is thin — the market maker's loan has been returned, the narrative is gone, and the participants who would normally bid are the ones holding newly unlocked supply. Assume a modest 4% slippage on the way out:

\$2,614 × 0.96 = **\$2,509**

Your actual return: (\$2,509 − \$10,000) ÷ \$10,000 = **−74.9%**

The chart, meanwhile, shows \$1.30 → \$0.35, which is **−73.1%**.

The intuition this teaches: **the chart understates the loss at both ends. You bought above the quoted price and you sold below it, and on a thin book those two costs are large enough to matter. Every screenshot you have ever seen of a token's drawdown is the optimistic version.**

## What is coordination, and what is just a market?

This is the section that makes the difference between analysis and conspiracy theory, and it is the one most crypto commentary skips.

Everything described above can happen with **no coordination whatsoever.** A thin float is a design choice made months before launch. A market-making contract is signed with a firm that will never speak to the KOLs. An exchange lists a token on its own commercial judgment. A fund publishes research because that is what funds do. An unlock happens because a smart contract counts to twelve.

Assemble those five independent decisions and you get the chart. Add genuine enthusiasm — a real product, real users, a real bull market — and you get a *bigger* version of the same chart, faster.

So the honest question is not "is this shape suspicious?" It is: **which observations actually distinguish a coordinated move from a crowded one?** And the answer is: fewer than people think.

![Which observable signals actually distinguish coordination from genuine demand](/imgs/blogs/the-anatomy-of-a-token-pump-9.webp)

| Observation | Consistent with real demand? | Consistent with coordination? | Does it distinguish? |
| --- | --- | --- | --- |
| Price up 300% in six weeks | Yes | Yes | **No** |
| Volume concentrated on two venues | Yes — that is where the liquidity is | Yes | **No** |
| Top ten wallets hold 60% of the float | Yes — exchanges and custodians hold in omnibus wallets | Yes | **No** |
| Twelve large accounts post within 48 hours | Yes — news is news | Yes | **Weakly** |
| Orders repeatedly self-matched at identical size | Rarely — it costs fees for no economic purpose | Yes | **Yes** |
| The buying wallets are all funded from one address | Rarely | Yes | **Yes** |

The top three rows are where most retail "manipulation analysis" lives, and all three are worthless as evidence. A 300% move is what a thin float does when anyone buys. Concentrated volume is what liquidity fragmentation looks like from outside. Concentrated wallets are, more often than not, exchange omnibus addresses holding thousands of customers' balances — the single most common error in amateur on-chain analysis.

The bottom two rows are different in kind. **Self-matched trades** — the same beneficial owner on both sides — have no economic purpose. You pay fees to move an asset from your left hand to your right. The only product is a print on the tape: a volume number and a price. That is what **wash trading** means, and it is prohibited in essentially every regulated market precisely because it manufactures the appearance of a market where none exists. [Wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) covers the taxonomy, and [detecting wash trading](/blog/trading/onchain/detecting-wash-trading) covers the on-chain forensics.

Even then — and this is the part that matters — **the on-chain and tape evidence establishes a pattern, not a purpose.** Proving that a pattern was intentional generally requires the things a chart cannot give you: internal messages, contracts, testimony, the ability to compel a firm to explain its own logs. That is why the cases below matter. They are the situations where somebody with subpoena power went and got that evidence.

Two other framings worth holding onto:

- **Describe incentives, not intent.** "This contract pays the desk more when the price is higher" is a fact about a document. "The desk pumped the token" is an allegation about a person. The first is analysable from outside; the second is not.
- **Absence of enforcement is not absence of conduct, and presence of enforcement is not proof of guilt.** A charge is an allegation. A consent judgment is a negotiated resolution, not a jury's finding. And a docket that says nothing but "terminated" is genuinely ambiguous: a case can end in a guilty plea, in a settlement, in a dismissal on the merits, or in a dismissal because an agency's enforcement priorities changed — and you cannot tell which from the fact of termination alone. US crypto enforcement between 2018 and 2026 contains examples of all four, which is precisely why the honest move is to read the operative order rather than the outcome column.

## How it shows up in the public record

Almost everything in Parts 1 through 5 is lawful structure. Two things in the neighbourhood are not: **fabricating trading activity**, and **selling fabricated trading activity as a service**. Those are the parts that generate court files, and court files are the only place where the question of intent gets answered by evidence rather than inference.

A note on method, because it matters for how much weight you should give what follows. The SEC's own document server (`sec.gov`) blocks automated retrieval, and the Justice Department's press pages are now behind an edge filter that does the same. So the facts below are taken from the **federal courts' own dockets** via the public RECAP archive rather than from either agency's press release. That is a stronger source for dates, case numbers, parties and judgment amounts — it is the court's record, not a party's characterisation of it — and a weaker source for narrative, because a docket tells you what was filed and decided, not what the complaint argued. Where I could not reach the underlying pleading, I say so instead of paraphrasing from memory.

### Case one: a market-making bot with a dollar figure attached to it

On **28 September 2022** the SEC filed *U.S. Securities and Exchange Commission v. The Hydrogen Technology Corporation*, No. **1:22-cv-08284**, in the Southern District of New York, before Judge **Lewis A. Kaplan**. The defendants were the company, its founder **Michael Ross Kane**, and **Tyler Ostern**. The case concerned trading in the company's own token, and the mechanism at issue was automated market-making activity — the widely reported account is that a trading bot was used to generate apparent volume and price. I could not retrieve the complaint itself, so treat the mechanism as reported rather than as something I have read.

What the docket does establish precisely is how it ended, and that part is unusually legible:

- **6 April 2023** — the SEC moved for approval of a **consent judgment** as to Hydrogen and Kane, with each defendant's signed consent attached. This was a negotiated resolution, not a contested trial verdict.
- **20 April 2023** — the court entered **final judgment** as to both. Hydrogen was ordered liable for **disgorgement of \$1,516,703.53**, described in the judgment as "representing net profits gained as a result of the conduct alleged in the Complaint", plus **prejudgment interest of \$244,531.98**, plus a **civil penalty of \$1,035,000** under Section 20(d) of the Securities Act (15 U.S.C. § 77t(d)) and Section 21(d)(3) of the Exchange Act.

That totals **\$2,796,235.51**.

And on the market-making side, the individual was charged criminally. On **19 April 2023** — one day before the civil judgment — the government filed a one-count **information** against Ostern in the Southern District of Florida (*United States v. Ostern*, No. **1:23-cr-20165**). A charge brought by information rather than indictment almost always signals a negotiated disposition, and the docket confirms it: a **plea agreement was filed on 30 May 2023**, sentencing memoranda followed in August, and the case closed on **16 August 2023**. The sentence imposed is not in the public RECAP record, so I am not going to state one.

Two things are worth extracting from this. First, **a court put a number on it**: "net profits gained" from the conduct alleged came to roughly \$1.5 million. Second, **the person on the liquidity-provision side went to criminal court and resolved it by plea.** That is about as close as the public record gets to establishing that manufactured trading was deliberate rather than emergent.

### Case two: the October 2024 cluster, where the defendants were the market makers

In October 2024 something structurally new appeared on the docket in the **District of Massachusetts**: not token issuers charged for lying about a product, but **market-making firms charged as corporate defendants alongside their principals.**

The criminal cases, with charging dates that show the indictments sat sealed for months before being unsealed on **9 October 2024**:

| Case | No. | Charged | Corporate defendant |
| --- | --- | --- | --- |
| *United States v. ZM Quant Investment Ltd.* | 1:24-cr-10187 | 2024-06-27 | ZM Quant Investment Ltd (with Baijun Ou, Ruiqi Liu) |
| *United States v. Kohli* | 1:24-cr-10189 | 2024-06-27 | — (Manpreet Kohli, Haroon Mohsini, Nam Tran) |
| *United States v. Gotbit Consulting LLC* | 1:24-cr-10190 | 2024-06-27 | Gotbit Consulting LLC (with Aleksei Andriunin, Qawi Jalili, Fedor Kedrov) |
| *United States v. CLS Global FZC LLC* | 1:24-cr-10293 | 2024-09-19 | CLS Global FZC LLC (with Andrey Zhorzhes) |
| *United States v. Zhou* | 1:24-cr-10312 | 2024-10-07 | — (Liu Zhou) |

Five parallel **SEC civil complaints** were filed in the same district on **9 October 2024**: *SEC v. Kohli* (1:24-cv-12586), *SEC v. ZM Quant Investment Ltd.* (1:24-cv-12587), *SEC v. Pham* (1:24-cv-12588), *SEC v. Gotbit Consulting LLC* (1:24-cv-12589), and *SEC v. CLS Global FZC LLC* (1:24-cv-12590). Between them the civil cases named several further individuals, including Vy Pham, Maxwell Hernandez and Russell Armand.

Where things stand, from the dockets as of **31 July 2026**:

- ***United States v. CLS Global FZC LLC*** closed **7 April 2025**.
- ***United States v. Gotbit Consulting LLC*** closed **18 June 2025** as to both the company and Aleksei Andriunin.
- *SEC v. Pham* closed **25 March 2025**; *SEC v. Kohli* closed **29 April 2025**; *SEC v. CLS Global FZC LLC* closed **2 April 2026**.
- **Still open:** the criminal cases against ZM Quant, Kohli and Zhou, and the civil cases against ZM Quant and Gotbit.

I want to be exact about what that list is and is not. Those are **termination dates, not verdicts.** A closed docket can mean a guilty plea and sentence, a negotiated settlement, or a dismissal — and the entries stating which are not in the portion of the record I could retrieve. So: **these are charges. Several defendants have not been adjudicated at all, and I am not characterising any firm named above as a manipulator.** What the record establishes is that charges were brought and that some matters have concluded; it does not, on what I can see, establish how.

What it *does* establish, and this is the analytically useful part, is **what draws a charge.** Nobody in this cluster was charged for launching with a 10% float, or for signing a loan-plus-call market-making agreement, or for listing on two venues three weeks apart, or for paying for research distribution. Those things happened at enormous scale in 2024 and generated no dockets at all. The alleged conduct that produced criminal exposure was of a different kind: creating trading activity that had no economic purpose other than to look like trading activity, and selling that as a product to token issuers.

Which is exactly the bottom two rows of the discrimination table above. The line the law draws is not "the price went up too much." It is "you manufactured the tape."

### What the two cases teach about the machine

**Enforcement attaches to fabrication, not to structure.** Every component in Parts 1 through 5 remains, as far as these records show, entirely legal. If you were hoping this post would end with the discovery that thin floats are illegal, they are not, and no case here suggests otherwise.

**The evidence that mattered was never chart-shaped.** A consent judgment with a disgorgement figure computed to the cent, a one-count information, a plea agreement — these come from bank records, internal messages, and the ability to compel a firm to explain its own order flow. No amount of staring at a candlestick chart produces any of them. That is the epistemics lesson of the whole section: the question you can answer from outside is *what is the structure*, and the question you cannot is *what did they intend*.

**And the amounts are small relative to the prices.** Hydrogen's disgorgement — the court's own figure for net profits gained — was about **\$1.5 million**. Set that beside the arithmetic from Part 1, where **\$2.5 million of net buying doubles a token and adds \$500 million to its headline valuation.** The input needed to manufacture the *appearance* of a market is roughly the same order of magnitude as the input needed to move a thin one. That is not a coincidence; it is the same fact viewed twice. On a book this shallow, both the real thing and the fake thing are cheap.

## Common misconceptions

**"A pump means someone illegally manipulated the price."** Usually not, in the legal sense. Most of the price action described in this post is produced by lawful structure: lock-ups, liquidity agreements, listing decisions, marketing. Manipulation is a specific thing — trading designed to create a false or misleading appearance of market activity, or to deceive others about the price. It requires conduct and, generally, intent. The chart on its own establishes neither. Conversely, "it was legal" does not mean "it was fair to the person who bought the top", and those are separate questions.

**"Fully diluted valuation is the real valuation."** It is the opposite: it is the least real of the two numbers, because it prices supply that cannot be sold at a price set by supply that can. FDV is useful for one thing — comparing the *eventual* dilution of two projects — and misleading for almost everything else. When a token's FDV is ten times its market cap, nine-tenths of the headline is a claim on the future.

**"High volume means the market is liquid."** Volume and depth are different quantities and can diverge enormously. Volume is how much traded; depth is how much is *standing there ready to trade*. A token can print large volume from a small amount of capital turning over rapidly — or, in the pathological case, from the same capital trading with itself. Depth is what determines your slippage, and depth is what disappears exactly when you need it.

**"The market maker's job is to keep the price stable."** No — a market maker's contractual job is to keep a *quote* available at a bounded spread. That is a statement about the gap between bid and ask, not about the level of the price. A desk can honour every term of its agreement while the price falls 80%, because the spread stayed tight the whole way down.

**"If I can see the unlock schedule, it must be priced in."** Sometimes; often not, and the reason is structural rather than about information. Everyone can see the date, but the people who could most efficiently price it — those who would short the overhang — frequently cannot: borrow is unavailable or ruinously expensive on small tokens, perp open interest is too thin to carry size, and the position has a negative carry while you wait. Information being public is not sufficient for it to be in the price; someone has to be *able* to trade it.

**"Concentrated holdings prove insiders are about to dump."** The top wallets on most tokens are exchange omnibus addresses, bridge contracts, staking contracts and treasuries — none of which are a person about to sell. Reading a rich-list without labelling the addresses produces confident, wrong conclusions almost every time. [Whales, smart money and on-chain wallet-watching](/blog/trading/crypto-players/whales-smart-money-and-on-chain-wallet-watching) covers how to label them properly.

**"This only happens to obvious scams."** The mechanics described here are structural, not moral. A genuinely excellent project with a real product, an honest team and a thin float and a twelve-month cliff will produce the same chart shape, because the chart shape is a consequence of the supply schedule and the depth, not of the quality of the software.

## Defending yourself as a retail participant

Nothing here is investment advice — it is a set of things you can *measure* before deciding anything. The common thread is that all of them are checkable in a few minutes and almost none of them are about the price.

**1. Compute the float ratio before you look at the chart.** Circulating supply divided by total supply. Under 15% at listing means the price you see is being set by a small tail wagging a very large dog. Then compute market cap and FDV yourself rather than reading either off a card — aggregators disagree, and some quote FDV where you expect market cap.

**2. Read the unlock calendar first, not last.** Find the next cliff date and the tokens releasing on it. Multiply by the current price. Divide by the token's genuine daily volume. If the answer is more than a few days of total turnover, you are looking at a supply event that the market will need months to absorb. That single ratio is the most useful number you can compute about a young token, and it takes ninety seconds.

**3. Measure depth, not volume.** Open the order book on the venue where you would actually trade and add up the bids within 5% of the price. That number is what your position is worth in a hurry — not the mid-price times your size. If the bids within 5% total \$80,000 and you are thinking about a \$20,000 position, you are a quarter of the exit.

**4. Price your round trip before you enter.** Estimate slippage in *and* out, and add them. On a thin book that can be 8–12% before you have been right or wrong about anything. If your thesis needs a 15% move to break even, it is a different thesis than you thought.

**5. Treat extreme funding as a position readout, not a signal.** Persistent funding above roughly 0.1% per eight hours means longs are paying over 100% annualised. That does not predict direction. It does tell you that the marginal holder is leveraged, which means the next move down will be amplified by liquidations. Size accordingly.

**6. Trace the story back one hop.** When a thesis reaches you, ask who published it first and what they hold. Usually one click. If the original source is an investor in the round, that is not disqualifying — it is the most normal thing in the world — but you now know you are reading marketing rather than research, and you can weight it accordingly.

**7. Distrust your own pattern recognition on Leg 2.** Rising price on falling volume is the leg that produces the most confident retail entries and is the most mechanically ambiguous. If you cannot say who is buying, you do not know that anyone is.

**8. Separate "this is rigged" from "this is structured against me".** The first is usually unprovable and often false. The second is frequently, demonstrably true and is far more actionable — because it points at things you can measure (float, unlocks, depth, cost basis) instead of things you can only speculate about (intent).

The sibling post [reading the tape: defending yourself as retail](/blog/trading/crypto-players/reading-the-tape-defending-yourself-as-retail) goes considerably deeper on the practical toolkit, and the later parts of this series cover the forensic side directly — a full manipulation playbook and the on-chain red flags that actually discriminate are both covered later in the series.

## When this matters to you

If you never buy a newly listed token, most of this is spectator knowledge — though it is worth having, because the same structure shows up wherever a small tradeable float sits on top of a large locked claim. That is not a crypto-specific arrangement. Lock-up expiries after an IPO work the same way, with more disclosure and slower clocks.

If you do buy newly listed tokens, the practical shift this post is arguing for is small and specific: **stop reading the price first.** The price is the output. Float, depth, and the unlock calendar are the inputs, they are all public, and they take about five minutes to look up. Someone who checks those three numbers before looking at a chart is playing a materially different game from someone who does not — not because they will pick better tokens, but because they will know what they are holding and what it costs to stop holding it.

And the meta-lesson is worth stating on its own. The most useful posture towards this machine is neither credulity nor conspiracy. It is mechanical curiosity: *which part of the structure produced this leg, and can I measure it?* Most of the time you can. When you cannot — when the question genuinely turns on what someone intended — the honest answer is that you do not know, and that the people who find out have subpoenas.

For the wider map of who the participants are and what each of them is optimising for, [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) is the series overview and [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) is the one that most directly generalises the argument here. [Crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) is the hub for the whole series.

## Sources & further reading

**A note on the numbers in this post.** Everything concerning NOVA — the supply table, the ask ladder, the market-making terms, the option strikes, the price path, the unlock sizes, the participant P&L — is **illustrative arithmetic on a hypothetical token**. NOVA does not exist. Those numbers are chosen to be internally consistent and recomputable, not to describe any real asset, and no real firm is depicted anywhere in the composite walkthrough. The comparison figures for "cost to move the price 1%" across tokens of different sizes are likewise illustrative and are there to show a ratio, not a level. Every claim about the real world is separated out below.

**Statutory text**

- Securities Act of 1933, § 17(b) — the anti-touting provision, codified at 15 U.S.C. § 77q(b). Full text via the [Cornell Legal Information Institute](https://www.law.cornell.edu/uscode/text/15/77q). This is the provision requiring disclosure of consideration received for describing a security, *including the amount*.

**Court records**

All docket metadata below was verified against the public federal dockets via [CourtListener / RECAP](https://www.courtlistener.com/), accessed 2026-07-31. Docket numbers, courts and filing dates are primary-source facts; charges are allegations unless a judgment is noted.

United States District Court for the District of Massachusetts — criminal cases unsealed on 2024-10-09:

- *United States v. ZM Quant Investment Ltd.*, No. 1:24-cr-10187 (filed 2024-06-27)
- *United States v. Kohli*, No. 1:24-cr-10189 (filed 2024-06-27)
- *United States v. Gotbit Consulting LLC*, No. 1:24-cr-10190 (filed 2024-06-27)
- *United States v. CLS Global FZC LLC*, No. 1:24-cr-10293 (filed 2024-09-19)
- *United States v. Zhou*, No. 1:24-cr-10312 (filed 2024-10-07)

Parallel SEC civil complaints, all filed in the same district on 2024-10-09:

- *SEC v. Kohli*, No. 1:24-cv-12586
- *SEC v. ZM Quant Investment Ltd.*, No. 1:24-cv-12587
- *SEC v. Pham*, No. 1:24-cv-12588
- *SEC v. Gotbit Consulting LLC*, No. 1:24-cv-12589
- *SEC v. CLS Global FZC LLC*, No. 1:24-cv-12590

United States District Court for the Southern District of Florida:

- *United States v. Ostern*, No. 1:23-cr-20165 (filed 2023-04-19) — the criminal case against the Moonwalkers Trading principal in the HYDRO matter.

**Where to read further in this series**

The rest of this series takes each component apart on its own terms: [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) and [the loan-plus-options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid) for Part 2; [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) and [unlock cliffs and the supply-overhang trade](/blog/trading/crypto-players/unlock-cliffs-and-the-supply-overhang-trade) for Part 5; [wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) for the conduct that is actually prohibited. A dedicated manipulation playbook and an on-chain red-flag guide come later in the series.

*This post is educational. It explains mechanisms and measurable quantities; it is not investment advice and does not recommend buying or selling anything.*
