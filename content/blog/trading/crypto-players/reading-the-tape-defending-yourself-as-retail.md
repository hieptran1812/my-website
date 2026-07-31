---
title: "Reading the Tape: Defending Yourself as Retail"
date: "2026-07-31"
publishDate: "2026-07-31"
description: "The seven questions to answer before you buy a token  -  cap table, float versus FDV, market-maker deal terms, unlock calendar, where size actually trades, listing incentives, and who started the story  -  with what to look up, where to look it up, and what a bad answer looks like."
tags: ["crypto", "crypto-players", "retail-investing", "tokenomics", "float", "fdv", "unlocks", "market-makers", "due-diligence", "market-structure", "risk-management"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR**  -  Every token trade has someone on the other side of it. In crypto, that someone is unusually likely to be a professional with a cost basis near zero, a contractual reason to sell on a specific date, and better information than you. The defence is not a better prediction. It is a checklist.
>
> - **You cannot beat information asymmetry, but you can measure it.** Seven public facts  -  cap table, float, market-maker terms, unlock calendar, venue depth, listing incentives, narrative origin  -  tell you most of what you need. All seven are free to look up. Most people look up none.
> - **Float is the number that does the damage.** A token with 12% of its supply circulating is not cheap because its market cap is small; it is expensive because the other 88% is a queue of sellers with a schedule. Fully diluted valuation is the price you are actually paying.
> - **The unlock calendar is the only part of the future that is written down.** A cliff that releases 11 days of average daily volume in a single block is not a risk you forecast. It is a date you read.
> - **Depth, not price, decides what you can actually do.** If a token's order book holds \$70,400 within 2% of the mid, a \$250,000 position is not a position  -  it is a hostage situation. Size against exit liquidity, never against entry price.
> - **The one number to remember:** if you cannot name who is selling to you and roughly what they paid, you are not the buyer of the trade. You are the exit.

Every few months a token goes vertical and a version of the same conversation happens. Someone you know buys it near the top, watches it fall 70%, and asks what went wrong. The honest answer is almost never "the technology failed" or "the market turned." It is usually much more boring: the buying was real, the selling was scheduled, and only one side of that knew both facts.

This post is the capstone of a series about who actually moves crypto prices  -  the venture funds, the market makers, the exchanges, the foundations, the influencers, and the handful of people whose balance sheets are large enough that their portfolio decisions *are* the market. If you have read the rest, you know how each of those players works. This post is the one that turns all of it into something you can run in twenty minutes before you press buy.

![Who is on the other side of your trade  -  a retail market buy at the centre, surrounded by the five parties most likely to be filling it](/imgs/blogs/reading-the-tape-defending-yourself-as-retail-1.webp)

Figure 1 is the concrete map to keep in mind, and it is worth sitting with for a moment before we go anywhere else. When you place a market buy, a matching engine pairs you with whoever has an order resting on the other side. That counterparty is not a placeholder. It is a specific entity with a specific cost basis and a specific reason to be selling at this price, right now. In a mature equity market that counterparty is usually another investor with roughly your information and roughly your cost basis. In crypto it very often is not. It is a seed fund that paid two cents. It is a market maker returning borrowed inventory. It is a foundation converting tokens into payroll. It is an advisor whose allocation vested last Tuesday.

None of that is illegal, hidden, or even unusual. Most of it is disclosed somewhere. The asymmetry is not that the information does not exist  -  it is that one side reads it as a matter of professional routine and the other side has never been told it exists.

Let us fix that.

## The foundations: the words you need before the checklist makes sense

Skip this section if you already trade. If you do not, read it once and the rest of the post will be legible.

**A token** is an entry in a database that a blockchain keeps. Owning one means the database says a key you control is associated with it. That is the whole thing. Whether the token also entitles you to anything  -  revenue, votes, a service  -  depends entirely on what its issuer chose to build, and in most cases the answer is "governance votes and nothing else." This is the first and largest departure from equities, and the series covers it in [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock).

**The order book** is the list of unfilled orders at a venue  -  the structure is the same in every modern market, and [inside an exchange: the matching engine and the order book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book) is the general version. Buy orders (*bids*) sit below the current price; sell orders (*asks*) sit above it. The highest bid and the lowest ask are the *best bid* and *best ask*, and the gap between them is the *bid-ask spread*  -  the round-trip cost of entering and immediately exiting. A **market order** says "fill me now at whatever price it takes," so it walks up the asks, consuming them cheapest-first, until it is filled. A **limit order** says "fill me only at this price or better," so it rests in the book and waits. This bid/ask and market/limit-order structure is also described in [Binance Academy's order-book guide](https://academy.binance.com/en/articles/what-is-an-order-book-and-how-does-it-work) (updated June 25, 2026).

**Depth** is how much money sits in the book near the current price. It is usually quoted as *±2% depth*: the total dollar value of orders within 2% of the midpoint. Depth is the single most under-appreciated number in retail crypto, because price tells you what one token costs and depth tells you what a *position* costs. They are not the same, and the difference gets larger the more you buy.

**Volume** is how much traded over a period. It is also the number most easily faked, which is why the series devotes a whole post to [wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume). Treat reported volume as a claim, not a measurement.

**Circulating supply** is the number of tokens that exist and are free to trade. **Total supply** is every token that exists now. **Maximum supply** is every token that will ever exist. **Float** is circulating supply as a percentage of total  -  the fraction of the project that is actually in the market.

**Market capitalisation** is price multiplied by circulating supply  -  what the tradeable slice is worth. **Fully diluted valuation (FDV)** is price multiplied by total or maximum supply  -  what the *whole project* is worth at today's price. This distinction and the caveat that FDV is theoretical are set out in [CoinGecko's market-cap and FDV explanation](https://www.coingecko.com/learn/what-is-market-cap-in-crypto) (updated April 10, 2026). When these two numbers are far apart, the gap is a promise that a great deal of supply is coming, and that someone will need to buy it.

**An unlock** (or *vest*) is the scheduled moment when previously restricted tokens become transferable. A **cliff** is an unlock that happens all at once, typically at the 12-month mark. **Linear vesting** releases a fixed amount each month or block after the cliff. These are implementation concepts rather than universal legal definitions; the Ethereum [ERC-5725 vesting specification](https://eips.ethereum.org/EIPS/eip-5725) is a useful technical reference for vested, claimable, locked, and timestamp-based payouts. **TGE** is the *token generation event*  -  the moment the token first exists and usually first trades.

**A market maker (MM)** is a firm that quotes both a bid and an ask continuously, earning the spread and holding inventory risk in between. The general economics of that role are covered in [market makers and the spread](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity). In crypto they are usually paid by the issuer, in tokens, under a contract  -  which is a genuinely different arrangement from equities and is the subject of [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) and [designated versus principal market making](/blog/trading/crypto-players/designated-versus-principal-market-making).

**OTC** means over-the-counter: a trade negotiated bilaterally and settled off the public order book. It exists precisely so that large sellers can move size without the book seeing it, which is exactly why it matters to you. [OTC desks and moving size without moving price](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price) is the long version.

**The cap table** is the list of who owns what and at what price. In equities it is a regulated filing. In crypto it is a blog post, if it exists at all  -  which is the whole problem, and the subject of [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table).

That is the vocabulary. Everything below is built from it.

## The one question underneath all seven

Before the checklist, the principle that generates it.

Markets are not a machine that converts good projects into rising prices. They are a mechanism for transferring assets between people who disagree about what those assets are worth. Every completed trade is a disagreement resolved by an exchange of money for a thing. For you to make money buying a token at \$2.00, someone has to be willing to sell it to you at \$2.00 and then be wrong about that decision.

So the question that generates every other question is: **who is selling to me, and why are they willing?**

Economists have a name for the situation where one side of a trade systematically knows more than the other: it is the *lemons problem*, and [asymmetric information: the lemons problem in markets](/blog/trading/game-theory/asymmetric-information-the-lemons-problem-in-markets) is the general treatment. Its central result is uncomfortable and useful  -  when buyers cannot tell good from bad, they rationally discount everything, and the best sellers leave the market. Crypto's version is more tractable than the classic one, because a surprising amount of the information *is* public. It just is not summarised anywhere.

There are only a few good answers. They need liquidity for reasons unrelated to the token. They have a shorter time horizon than you. They are rebalancing a portfolio. They genuinely think it is worth less and might be wrong. Those are the trades worth taking  -  you are being paid to bear a risk somebody else does not want.

And there are bad answers. They paid \$0.02 and you are paying \$2.00. Their tokens unlocked yesterday and their fund's mandate requires distribution. They are contractually obliged to deliver tokens to a counterparty this week. They wrote the thread that made you want to buy it. In those trades you are not being paid for bearing risk. You are the liquidity event.

The series calls this frame *cui bono*  -  who benefits  -  and maps it across every player type in [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto). The checklist below is that map turned into seven lookups.

## The seven questions

![The seven questions in order, each with the answer that should stop you](/imgs/blogs/reading-the-tape-defending-yourself-as-retail-2.webp)

Each question below has the same four parts: **what to look up**, **where to look it up**, **what a bad answer looks like**, and **which post in the series explains the mechanism**. Run them in order. The order matters  -  question 2 is meaningless if you skipped question 1, and question 5 will change how you act on all of them.

Throughout, I will run a single illustrative token through the checklist so the arithmetic stays concrete. Call it **NOVA**. NOVA is not a real token and none of its numbers are real  -  every figure attached to it is invented for teaching, and I will say so again wherever the arithmetic appears. Real, sourced cases come later in the post.

### Question 1  -  Who owns it?

**What to look up.** The allocation table: what percentage went to seed investors, later rounds, the team, the foundation or treasury, the community, advisors, and the market maker. Then, for each bucket, the price paid and the vesting terms.

**Where to look it up.** The project's own documentation first  -  a tokenomics page, a launch blog post, a whitepaper appendix. Then the funding databases that track private rounds, which will often name the funds and the round sizes even when the project is vague. Then the block explorer for the token's contract, which will show you the largest holding addresses whether or not anyone wanted you to see them.

**What a bad answer looks like.** There are three tiers of bad. The worst is *no allocation table at all*  -  if a project will not tell you who owns it, that is the answer to the question. The second is a table with percentages but **no prices and no vesting terms**, which tells you the shape of the cap table but not the pressure in it. The third and subtlest is a table that adds up to less than 100%, or that has a large bucket called "ecosystem" or "reserve" with no further breakdown. Undefined buckets are where discretionary selling lives.

#### Worked example: reading NOVA's cap table (illustrative)

Here is NOVA's allocation, on a total supply of 1,000,000,000 tokens:

| Bucket | Share | Tokens | Price paid | Vesting |
| --- | --- | --- | --- | --- |
| Seed round | 18% | 180,000,000 | \$0.02 | 12-month cliff, then 24 months linear |
| Series A | 12% | 120,000,000 | \$0.10 | 12-month cliff, then 24 months linear |
| Team | 20% | 200,000,000 | \$0 | 12-month cliff, then 24 months linear |
| Foundation / treasury | 25% | 250,000,000 |  -  | No lock; discretionary |
| Community / airdrop | 15% | 150,000,000 | \$0 | 90M claimable at TGE, 60M staged |
| Market maker loan | 3% | 30,000,000 | Loaned | Returnable or purchased at strike |
| Advisors | 7% | 70,000,000 | \$0 | 6-month cliff, then 18 months linear |

Two things jump out of that table before any price is mentioned.

First, the **cost bases**. Seed paid \$0.02. If NOVA lists at \$2.00, seed is up 100× before a single unit of the product ships. A holder at 100× does not need the price to keep rising to be delighted with selling. They need it to still exist. This is the mechanic behind [the crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model) and it is why the phrase "the VCs are dumping" is usually less a moral accusation than a description of a fund returning capital to its own investors on schedule.

Second, the **25% foundation bucket has no lock**. Nothing in any unlock calendar will ever flag it, because contractually it is already free. If the foundation decides to fund three years of payroll by selling into a rally, that supply arrives with no warning event at all. [Token foundations and treasuries: the on-chain central banks](/blog/trading/crypto-players/token-foundations-and-treasuries-the-on-chain-central-banks) is the post on how those decisions actually get made.

*The intuition: the cap table tells you the cost basis of everyone who might sell to you, and cost basis determines how far a price can fall before selling stops.*

**Where this is explained:** [follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) and [the lifecycle of a token: seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock).

### Question 2  -  How much of it actually floats?

**What to look up.** Circulating supply, total supply, market cap, and FDV. Then the ratio of FDV to market cap, which is simply total supply divided by circulating supply. Then  -  and this is the step almost everyone skips  -  check whether the "circulating supply" number is *true*.

**Where to look it up.** The major price aggregators publish circulating supply, market cap and FDV on every token page, and they are the fastest first stop. But circulating supply on an aggregator is a number the *project submits* and the aggregator verifies to varying degrees. Cross-check it against the block explorer's holder list: if the top ten addresses hold 70% of what is supposedly circulating and several of them are labelled as team or foundation wallets, the real float is much smaller than the published one.

**What a bad answer looks like.** For this screen, I use a float below roughly 15% at listing or an FDV-to-market-cap ratio above about 5× as warning heuristics, not universal market standards; the underlying supply distinction is the one defined by [CoinGecko](https://www.coingecko.com/learn/what-is-circulating-supply-crypto) (updated April 28, 2026). Worst of all is a published circulating supply that the explorer contradicts. Any one of those means the tradeable slice is small relative to the supply queued behind it.

![Three illustrative tokens screened on float and FDV  -  the same market cap can hide very different amounts of future supply](/imgs/blogs/reading-the-tape-defending-yourself-as-retail-3.webp)

#### Worked example: the float / FDV screen (illustrative)

Three tokens, all invented, all trading today:

| | NOVA | MERIDIAN | ORBIT |
| --- | --- | --- | --- |
| Price | \$2.00 | \$1.00 | \$0.20 |
| Circulating | 120,000,000 | 450,000,000 | 600,000,000 |
| Total supply | 1,000,000,000 | 1,000,000,000 | 10,000,000,000 |
| **Float** | **12%** | **45%** | **6%** |
| Market cap | \$240,000,000 | \$450,000,000 | \$120,000,000 |
| FDV | \$2,000,000,000 | \$1,000,000,000 | \$2,000,000,000 |
| **FDV / MC** | **8.3×** | **2.2×** | **16.7×** |

A retail screen that sorts by market cap puts ORBIT at the top of the "cheap" list: \$120 million, the smallest of the three. That screen is backwards.

Work out what each ratio means in dollars. ORBIT's locked supply is 94% of 10,000,000,000 tokens, which is 9,400,000,000 tokens. At \$0.20 that is **\$1,880,000,000 of supply still to come**. For ORBIT's price to be flat in three years, the market has to absorb \$1.88 billion of new selling  -  not raise \$1.88 billion of new investment, but find \$1.88 billion of *net new buying that does not currently exist* just to stand still.

NOVA's locked supply is 880,000,000 tokens, worth **\$1,760,000,000** at \$2.00. MERIDIAN's is 550,000,000 tokens, worth **\$550,000,000**. So MERIDIAN  -  the one with the *largest* market cap of the three, the one a naive screen calls expensive  -  has by far the least supply pressure ahead of it. It has already been diluted. The dilution is in the price.

The single sentence to carry out of this: **market cap tells you what you are buying; FDV tells you what you are eventually competing with.** And when the two diverge by more than about 5×, you are not really buying a token at a \$240 million valuation. You are buying an option on a \$2 billion valuation, and paying for a great deal of supply that has not arrived yet.

*The intuition: a small market cap next to a large FDV is not a discount, it is a queue.*

**Where this is explained:** [the low float, high FDV game](/blog/trading/crypto-players/the-low-float-high-fdv-game) goes deep on why this structure became the default and who chose it.

### Question 3  -  Who market-makes it, and on what terms?

**What to look up.** Which firm or firms are contracted to make markets in the token, how many tokens they were loaned, and  -  the part that actually matters  -  the strike prices and expiry of any call options attached to the loan.

**Where to look it up.** Almost never in one place, which is itself informative. Look for a foundation blog post announcing a market-making partnership; occasionally the terms are disclosed voluntarily, more often only the name is. Governance forums are the richest source, because a DAO that has to *vote* on a market-making agreement will usually publish it. Failing that, the block explorer: find the token's largest non-team holders and check whether they are labelled as a known trading firm, then look at whether that address moves inventory to exchanges in the pattern a market maker's would.

**What a bad answer looks like.** "We have a market maker" with no name. A named firm with no terms. Or terms that are disclosed but structured so the maker's payoff rises with price  -  because that tells you their interest and yours align on the way up and diverge sharply at the strike.

The dominant structure in crypto is a **token loan plus call options**. The issuer lends the maker tokens to quote with. The maker can either return the tokens at the end of the term or buy them at pre-agreed strike prices. That structure is covered in full in [the loan plus options deal: how market makers get paid](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid); here is what it does to you as a buyer.

#### Worked example: what the market maker's option is actually worth (illustrative)

NOVA lends its market maker 30,000,000 tokens for 24 months, with call options in three tranches:

| Tranche | Tokens | Strike |
| --- | --- | --- |
| 1 | 10,000,000 | \$2.40 |
| 2 | 10,000,000 | \$3.20 |
| 3 | 10,000,000 | \$4.00 |

Suppose NOVA trades at \$4.00 at expiry. Work each tranche:

- Tranche 1: (\$4.00 − \$2.40) × 10,000,000 = **\$16,000,000**
- Tranche 2: (\$4.00 − \$3.20) × 10,000,000 = **\$8,000,000**
- Tranche 3: (\$4.00 − \$4.00) × 10,000,000 = **\$0**

Total intrinsic value: **\$24,000,000**.

Now suppose NOVA trades at \$2.00  -  exactly its listing price  -  at expiry. Every strike is above spot, every option expires worthless, and the maker's entire compensation is whatever spread it earned quoting. Their payoff looks like the hockey-stick of any long call position: flat below the strike, rising one-for-one above it. If that shape is unfamiliar, [calls, puts and the payoff diagram](/blog/trading/options-volatility/calls-puts-and-the-payoff-diagram-the-language-of-options) builds it from scratch.

Read that payoff as an incentive map rather than an accusation. A firm holding \$24 million of upside at \$4.00 and \$0 at \$2.00 has an obvious preference about which of those happens, and it also holds 30,000,000 borrowed tokens  -  25% of the entire circulating float  -  with which to express that preference. It is worth being precise about what this does and does not imply. It does **not** imply the firm manipulates the price; market makers have legitimate, well-understood reasons to quote tightly and hold inventory, and [inventory risk, hedging and delta neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality) explains why a well-run book is usually hedged rather than directional. What it *does* imply is that a very large block of the tradeable supply sits with a party whose payoff is convex in price, on a known timetable, and that both the tightness of the market before expiry and the depth of it afterwards may have more to do with that contract than with anything about the project.

*The intuition: when you can see the strike prices, you can see the price levels at which somebody large stops caring.*

**Where this is explained:** [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does), [the loan plus options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid), and the firm profiles  -  [Wintermute](/blog/trading/crypto-players/wintermute-the-algorithmic-powerhouse), [GSR and Cumberland](/blog/trading/crypto-players/gsr-cumberland-and-the-established-otc-desks), [Amber, Galaxy and the Asia MM landscape](/blog/trading/crypto-players/amber-galaxy-and-the-asia-mm-landscape), and [DWF Labs](/blog/trading/crypto-players/dwf-labs-the-controversial-newcomer).

### Question 4  -  When does supply land?

This is the highest-value question on the list, because it is the only one whose answer is a **date**. Everything else on this checklist tells you about a state of the world. The unlock calendar tells you about the future, and it is written down.

**What to look up.** Every unlock event for the next 24 months: the date, the token quantity, and which bucket it belongs to. Then convert each one into two ratios  -  unlock size as a percentage of current circulating supply, and unlock size as a multiple of average daily volume.

**Where to look it up.** Dedicated unlock trackers maintain calendars per token, and the large DeFi data aggregators publish emissions and unlock schedules too. Always reconcile whatever the tracker says against the project's own vesting documentation, because trackers are built from those documents and inherit their errors. Where the vesting is enforced on-chain by a contract, the contract is the authority  -  read it, or read a block explorer's decoded view of it.

**What a bad answer looks like.** A cliff inside the next 90 days. Any single unlock worth more than roughly 5 - 10 days of genuine daily volume. A schedule you cannot find at all. And the special case worth naming: a schedule that ends, on paper, but whose "ecosystem" bucket is unlocked and undated  -  which means the calendar is describing only the part of the supply that happens to be documented.

![Reading an unlock calendar  -  the cliff at month 12 and the monthly vest that follows, both measured against daily volume](/imgs/blogs/reading-the-tape-defending-yourself-as-retail-4.webp)

#### Worked example: reading NOVA's unlock calendar (illustrative)

NOVA's contractually locked buckets  -  seed (180M), Series A (120M) and team (200M)  -  total 500,000,000 tokens. The terms are a 12-month cliff releasing 10% of that, then the remaining 90% linearly over 24 months.

Step one, the cliff. 10% of 500,000,000 = **50,000,000 NOVA** arriving on a single day.

Step two, put it in context. NOVA's circulating supply is 120,000,000. So the cliff is:

```
50,000,000 / 120,000,000 = 41.7% increase in float, overnight
```

Step three, measure it against liquidity rather than supply. NOVA's genuine daily volume is 4,500,000 tokens (\$9,000,000 at \$2.00). So:

```
50,000,000 / 4,500,000 = 11.1 days of total market volume
```

That second number is the one that should stop you. It does not say "the price will fall 41%." It says that if even a fraction of that block wants out, it has to be absorbed by a market that trades 4.5 million tokens a day in total  -  and that total includes all the buying *and* selling that would have happened anyway.

Step four, the drip after the cliff. The remaining 450,000,000 tokens vest over 24 months:

```
450,000,000 / 24 = 18,750,000 NOVA per month
18,750,000 / 30 = 625,000 NOVA per day
625,000 / 4,500,000 = 13.9% of daily volume, every single day, for two years
```

The cliff is the event people talk about. The drip is what actually determines the two-year chart. A market that must absorb an extra 13.9% of its own daily volume in structural selling every day does not need bad news to grind lower; it needs *unusually good* news just to stay flat.

The animation below is the same schedule seen as a mechanism rather than a table.

<figure class="blog-anim">
<svg viewBox="0 0 760 420" role="img" aria-label="A price line stays flat for twelve months, drops sharply when a fifty-million-token cliff lands, then drifts lower as monthly vests keep arriving" style="width:100%;height:auto;max-width:780px">
<title>An unlock cliff landing in a thin market, then the monthly drip</title>
<style>
.uc-axis{stroke:var(--border,#adb5bd);stroke-width:2}
.uc-tick{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.uc-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.uc-note{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.uc-red{font:700 14px ui-sans-serif,system-ui;fill:#e03131}
.uc-line{fill:none;stroke:#1971c2;stroke-width:3.5;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:730;stroke-dashoffset:730;animation:uc-draw 13s linear infinite}
.uc-big{fill:#e03131;opacity:0;animation:uc-drop 13s linear infinite}
.uc-s1{fill:#e03131;opacity:0;animation:uc-d1 13s linear infinite}
.uc-s2{fill:#e03131;opacity:0;animation:uc-d2 13s linear infinite}
.uc-s3{fill:#e03131;opacity:0;animation:uc-d3 13s linear infinite}
.uc-s4{fill:#e03131;opacity:0;animation:uc-d4 13s linear infinite}
.uc-cliffnote{opacity:0;animation:uc-note 13s linear infinite}
@keyframes uc-draw{0%{stroke-dashoffset:730}92%,100%{stroke-dashoffset:0}}
@keyframes uc-drop{0%,28%{opacity:0;transform:translateY(-120px)}34%,100%{opacity:.9;transform:translateY(0)}}
@keyframes uc-note{0%,32%{opacity:0}38%,100%{opacity:1}}
@keyframes uc-d1{0%,44%{opacity:0;transform:translateY(-90px)}49%,100%{opacity:.8;transform:translateY(0)}}
@keyframes uc-d2{0%,56%{opacity:0;transform:translateY(-90px)}61%,100%{opacity:.8;transform:translateY(0)}}
@keyframes uc-d3{0%,68%{opacity:0;transform:translateY(-90px)}73%,100%{opacity:.8;transform:translateY(0)}}
@keyframes uc-d4{0%,80%{opacity:0;transform:translateY(-90px)}85%,100%{opacity:.8;transform:translateY(0)}}
@media (prefers-reduced-motion:reduce){.uc-line{animation:none;stroke-dashoffset:0}.uc-big,.uc-s1,.uc-s2,.uc-s3,.uc-s4{animation:none;opacity:.85;transform:none}.uc-cliffnote{animation:none;opacity:1}}
</style>
<text class="uc-lbl" x="40" y="30">NOVA price (illustrative)</text>
<text class="uc-note" x="40" y="52">locked supply arriving as red blocks</text>
<line class="uc-axis" x1="80" y1="360" x2="720" y2="360"/>
<line class="uc-axis" x1="80" y1="90" x2="80" y2="360"/>
<text class="uc-tick" x="52" y="125">$2.00</text>
<text class="uc-tick" x="52" y="255">$1.40</text>
<text class="uc-tick" x="52" y="335">$1.05</text>
<text class="uc-tick" x="66" y="384">Month 0</text>
<text class="uc-tick" x="262" y="384">Month 12</text>
<text class="uc-tick" x="468" y="384">Month 24</text>
<text class="uc-tick" x="673" y="384">Month 36</text>
<rect class="uc-big" x="266" y="150" width="48" height="80" rx="5"/>
<text class="uc-red uc-cliffnote" x="330" y="176">50,000,000 at once</text>
<text class="uc-note uc-cliffnote" x="330" y="197">= 11.1 days of volume</text>
<rect class="uc-s1" x="380" y="286" width="20" height="34" rx="4"/>
<rect class="uc-s2" x="450" y="292" width="20" height="34" rx="4"/>
<rect class="uc-s3" x="530" y="298" width="20" height="34" rx="4"/>
<rect class="uc-s4" x="610" y="304" width="20" height="34" rx="4"/>
<text class="uc-note" x="380" y="348">monthly vest: 18.75M, every month, for 24 months</text>
<polyline class="uc-line" points="80,120 290,120 340,250 700,310"/>
</svg>
<figcaption>Illustrative. The flat stretch is the twelve months during which the schedule was public and nothing happened  -  which is exactly when it was cheapest to read. The cliff is one event; the four small blocks are the drip that follows it, and the drip is what shapes the second half of the chart.</figcaption>
</figure>

*The intuition: an unlock calendar converts an unknowable future into a dated supply schedule, and the only number that matters is the unlock divided by daily volume.*

**Where this is explained:** [unlock cliffs and the supply overhang trade](/blog/trading/crypto-players/unlock-cliffs-and-the-supply-overhang-trade) and [how VCs move price: listings, unlocks and narrative](/blog/trading/crypto-players/how-vcs-move-price-listings-unlocks-and-narrative).

### Question 5  -  Where does size actually trade?

**What to look up.** Which venues list the token, how much genuine volume each does, and the ±2% order-book depth on the deepest one. Then compare the depth number to the position size you were planning. The NOVA book below is invented for arithmetic; an actual depth snapshot must be dated because the book changes continuously.

**Where to look it up.** Aggregator exchange pages list every venue and pair with reported volume, and several publish a liquidity or confidence score per pair that is a rough proxy for depth. The exchange's own public API will give you the raw order book for free if you want the real number rather than a proxy. For perpetual futures, the funding rate and open interest tell you how much leveraged positioning sits on top of the spot market.

**What a bad answer looks like.** One venue carrying the overwhelming majority of volume. Reported volume that is large relative to a visibly thin book  -  the classic signature discussed in [wash trading, spoofing and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume). Or open interest in perpetuals that dwarfs spot depth, which means the price is set by leverage and can gap when that leverage is liquidated.

![The depth check before you size  -  a thin ask ladder, and what a quarter-million-dollar order actually pays](/imgs/blogs/reading-the-tape-defending-yourself-as-retail-5.webp)

#### Worked example: the depth check before you size (illustrative)

Here is NOVA's ask side on its deepest venue:

| Price | Tokens offered | Cumulative cost |
| --- | --- | --- |
| \$2.00 | 15,000 | \$30,000 |
| \$2.02 | 20,000 | \$70,400 |
| \$2.05 | 25,000 | \$121,650 |
| \$2.10 | 40,000 | \$205,650 |
| \$2.20 | 60,000 | \$337,650 |

First, the depth number. Two percent above \$2.00 is \$2.04, so the orders within 2% of the top of the book are the \$2.00 and \$2.02 levels: **\$70,400 of ask-side depth**. That is the entire market, at any price you would call "the current price."

Now suppose you want a \$250,000 position and you send a market order. Walk the book:

1. 15,000 tokens at \$2.00 = \$30,000. Spent: \$30,000. Tokens: 15,000.
2. 20,000 tokens at \$2.02 = \$40,400. Spent: \$70,400. Tokens: 35,000.
3. 25,000 tokens at \$2.05 = \$51,250. Spent: \$121,650. Tokens: 60,000.
4. 40,000 tokens at \$2.10 = \$84,000. Spent: \$205,650. Tokens: 100,000.
5. Remaining budget: \$250,000 − \$205,650 = \$44,350, at \$2.20 → 20,159 tokens. Tokens: 120,159.

Your average fill price:

```
$250,000 / 120,159 = $2.0806
```

That is **4.03% above the \$2.00 you saw on the screen**, paid before the position has done anything. And it is the *optimistic* half of the round trip: the bid side is usually thinner than the ask side in a token people are excited about, so exiting the same size in a hurry costs more, and exiting it during the unlock cliff you found in question 4 costs a great deal more.

The practical rule that falls out of this: **size against depth, not against price.** A common desk convention is to keep a single clip to something like 10% of the ±2% depth, so the order is absorbed rather than announced. For NOVA that is:

```
10% × $70,400 = $7,040
```

So a \$250,000 NOVA position is roughly 35× a comfortable clip and 3.5× the entire two-percent book. That does not make it impossible. It makes it a *project*, executed over days in small pieces, or negotiated off-book through an OTC desk  -  which is what the professionals on the other side of your trade are already doing, and why [cross-exchange arbitrage and the latency game](/blog/trading/crypto-players/cross-exchange-arbitrage-and-the-latency-game) and [OTC desks](/blog/trading/crypto-players/otc-desks-and-moving-size-without-moving-price) exist as businesses.

There is one more consequence worth stating plainly, because it is the most common way retail gets hurt without anyone doing anything wrong. **Depth is not constant.** The book you measured on a calm Tuesday is not the book that exists during a liquidation cascade, when market makers widen their quotes precisely because inventory risk has spiked. The depth you can rely on in the moment you most want to sell is a fraction of the depth you measured when you bought.

*The intuition: price is what one token costs; depth is what your position costs, and only one of them is on the screen.*

**Where this is explained:** [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) and [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues).

### Question 6  -  Who wanted this listing?

**What to look up.** How the token came to be listed where it is listed. Whether the exchange, its venture arm, or its launchpad holds tokens. Whether the listing came with a marketing campaign, a points programme, or an airdrop to the exchange's own users.

**Where to look it up.** The exchange's own announcement post, which will usually say more than you expect if you read it as a disclosure rather than as marketing. The exchange's venture arm publishes its portfolio. Launchpad and launchpool mechanics are documented on the venue itself. And funding databases will tell you whether the exchange's investment arm participated in a private round.

**What a bad answer looks like.** The venue that lists the token also invested in it, and does not say so prominently. A listing bundled with a launchpad sale in which the venue's users bought at a set price shortly before the open market did. A "listing fee" structure the venue will not describe.

The reason this question is on the list is that **an exchange is not a neutral piece of infrastructure**. It chooses what to list, when, with what fanfare, against which quote asset, and with what leverage available on day one. Each of those choices affects price. When the venue also holds a position, the choices and the position point the same direction. That is not a conspiracy; it is the ordinary conflict of interest that regulated venues in other asset classes manage with disclosure rules and that crypto venues largely manage by not having them. The series covers the structure in [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues), the two dominant venue models in [Binance: the everything exchange and its gravity](/blog/trading/crypto-players/binance-the-everything-exchange-and-its-gravity) and [Coinbase: the compliant giant](/blog/trading/crypto-players/coinbase-the-compliant-giant), and the distribution mechanics in [launchpads, airdrops and the points meta](/blog/trading/crypto-players/launchpads-airdrops-and-the-points-meta).

There is a second-order effect here worth internalising. A large listing is itself a liquidity event *for existing holders*. The day a token gains access to a major venue's user base is the first day a seed investor can realistically distribute size. So "listed on a top exchange" is simultaneously the bullish headline and the mechanism that makes the bearish supply real. Both are true at once.

*The intuition: ask what the venue owns before you ask what the venue thinks.*

### Question 7  -  Who started the story?

**What to look up.** The origin of the thesis you are acting on. Not where you heard it  -  where it *started*. Then whether the people who spread it hold tokens, and whether they were paid. The SEC's [statement on potentially unlawful promotion of ICOs](https://www.sec.gov/newsroom/speeches-statements/statement-potentially-unlawful-promotion-icos) (November 1, 2017) explains why compensation and conflicts matter when investment products are promoted publicly.

**Where to look it up.** Search for the earliest instance of the specific framing you are repeating, sorted by date rather than relevance. Check whether the accounts amplifying it disclose a position. Check whether the "independent research" was commissioned. Check the timestamps against the price chart: a narrative that appears *after* a 40% move is usually an explanation manufactured to justify buying that already happened.

**What a bad answer looks like.** You cannot find the origin. The origin is a paid promotion without a visible disclosure. Or  -  the most common and least examined case  -  the origin is you, reconstructing a reason for a price move you already saw. That last one has a name in behavioural finance and is covered from the market-mechanics side in [narrative cycles and who sets the story](/blog/trading/crypto-players/narrative-cycles-and-who-sets-the-story).

The relay below is the shape this usually takes.

<figure class="blog-anim">
<svg viewBox="0 0 760 400" role="img" aria-label="A narrative travels along five stages from foundation to insider selling, each lighting in turn, while a price line rises and then falls" style="width:100%;height:auto;max-width:780px">
<title>How a narrative reaches you, and what is waiting at the end of it</title>
<style>
.nr-box{fill:var(--surface,#f8f9fa);stroke:var(--border,#adb5bd);stroke-width:2}
.nr-fill1{fill:#f08c00;opacity:0;animation:nr-a1 14s linear infinite}
.nr-fill2{fill:#f08c00;opacity:0;animation:nr-a2 14s linear infinite}
.nr-fill3{fill:#f08c00;opacity:0;animation:nr-a3 14s linear infinite}
.nr-fill4{fill:#2f9e44;opacity:0;animation:nr-a4 14s linear infinite}
.nr-fill5{fill:#e03131;opacity:0;animation:nr-a5 14s linear infinite}
.nr-t{font:700 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.nr-s{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.nr-h{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.nr-arrow{stroke:var(--border,#adb5bd);stroke-width:2;fill:none;marker-end:url(#nrhead)}
.nr-price{fill:none;stroke:#1971c2;stroke-width:3;stroke-dasharray:700;stroke-dashoffset:700;animation:nr-draw 14s linear infinite}
@keyframes nr-a1{0%,3%{opacity:0}8%,100%{opacity:.28}}
@keyframes nr-a2{0%,19%{opacity:0}24%,100%{opacity:.28}}
@keyframes nr-a3{0%,35%{opacity:0}40%,100%{opacity:.28}}
@keyframes nr-a4{0%,51%{opacity:0}56%,100%{opacity:.32}}
@keyframes nr-a5{0%,67%{opacity:0}72%,100%{opacity:.32}}
@keyframes nr-draw{0%{stroke-dashoffset:700}90%,100%{stroke-dashoffset:0}}
@media (prefers-reduced-motion:reduce){.nr-fill1,.nr-fill2,.nr-fill3{animation:none;opacity:.28}.nr-fill4,.nr-fill5{animation:none;opacity:.32}.nr-price{animation:none;stroke-dashoffset:0}}
</style>
<defs><marker id="nrhead" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 z" fill="#adb5bd"/></marker></defs>
<text class="nr-h" x="30" y="28">the story moves left to right; the tokens move the other way</text>
<rect class="nr-box" x="24" y="60" width="126" height="76" rx="8"/>
<rect class="nr-fill1" x="24" y="60" width="126" height="76" rx="8"/>
<text class="nr-t" x="38" y="92">Foundation</text><text class="nr-s" x="38" y="113">writes the thesis</text>
<rect class="nr-box" x="176" y="60" width="126" height="76" rx="8"/>
<rect class="nr-fill2" x="176" y="60" width="126" height="76" rx="8"/>
<text class="nr-t" x="190" y="92">Research</text><text class="nr-s" x="190" y="113">often commissioned</text>
<rect class="nr-box" x="328" y="60" width="126" height="76" rx="8"/>
<rect class="nr-fill3" x="328" y="60" width="126" height="76" rx="8"/>
<text class="nr-t" x="342" y="92">KOL threads</text><text class="nr-s" x="342" y="113">disclosure optional</text>
<rect class="nr-box" x="480" y="60" width="126" height="76" rx="8"/>
<rect class="nr-fill4" x="480" y="60" width="126" height="76" rx="8"/>
<text class="nr-t" x="494" y="92">Retail bid</text><text class="nr-s" x="494" y="113">you arrive here</text>
<rect class="nr-box" x="632" y="60" width="112" height="76" rx="8"/>
<rect class="nr-fill5" x="632" y="60" width="112" height="76" rx="8"/>
<text class="nr-t" x="646" y="92">Insiders</text><text class="nr-s" x="646" y="113">sell into it</text>
<path class="nr-arrow" d="M152 98 L172 98"/>
<path class="nr-arrow" d="M304 98 L324 98"/>
<path class="nr-arrow" d="M456 98 L476 98"/>
<path class="nr-arrow" d="M608 98 L628 98"/>
<text class="nr-h" x="30" y="192">price, over the same period</text>
<polyline class="nr-price" points="40,330 200,312 360,268 520,214 660,232 730,318"/>
<text class="nr-s" x="470" y="200">retail volume peaks here</text>
<text class="nr-s" x="600" y="360">and this is the exit</text>
</svg>
<figcaption>The story travels one way and the tokens travel the other. Note where the retail bid sits in the sequence: not at the start of the narrative, and not at the end of the price move, but exactly at the point where the two intersect.</figcaption>
</figure>

None of the five stages is inherently improper. Foundations *should* explain their theses. Research desks are entitled to be paid. Influencers with disclosed positions are doing nothing wrong. The problem is structural rather than individual: the sequence reliably delivers a story to the largest, least-informed pool of buyers at precisely the moment when the best-informed holders most want liquidity, and the disclosure that would let you see it is voluntary. The mechanism by which each participant rationally passes the story on  -  each one updating on the fact that others already believed it  -  is an [information cascade](/blog/trading/game-theory/information-cascades-and-herding-when-rational-traders-follow-the-crowd), and once the price itself becomes the evidence for the thesis you have [reflexivity](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves).

*The intuition: find the first person who said it and check what they own.*

**Where this is explained:** [influencers, KOLs and the narrative-for-hire machine](/blog/trading/crypto-players/influencers-kols-and-the-narrative-for-hire-machine), [narrative cycles and who sets the story](/blog/trading/crypto-players/narrative-cycles-and-who-sets-the-story), and [the anatomy of a token pump](/blog/trading/crypto-players/the-anatomy-of-a-token-pump).

## Running the whole checklist end to end

Now put the seven together on NOVA. Everything below is illustrative arithmetic on an invented token; the point is the shape of the reasoning, not the numbers.

![Can you name the seller  -  a decision tree from the answer to your position size](/imgs/blogs/reading-the-tape-defending-yourself-as-retail-6.webp)

**Q1  -  Who owns it?** Seed at \$0.02 holds 18%. Team holds 20% at zero. The foundation holds 25% with no lock at all. Combined, 63% of the supply is held by parties with a cost basis at or near zero. *Answer: red.*

**Q2  -  How much floats?** 12% float, \$240 million market cap against \$2.0 billion FDV, a ratio of 8.3×. \$1.76 billion of locked supply sits behind a market that has absorbed \$240 million. *Answer: red.*

**Q3  -  Who market-makes it?** One firm, 30,000,000 tokens on loan  -  25% of the entire float  -  with calls struck at \$2.40, \$3.20 and \$4.00. The terms happen to be public here, which is better than most. *Answer: amber; the disclosure is good, the concentration is not.*

**Q4  -  When does supply land?** A 50,000,000-token cliff at month 12 (11.1 days of volume; +41.7% float), then 18,750,000 per month for 24 months (13.9% of daily volume, daily). *Answer: red, and it is a date rather than an opinion.*

**Q5  -  Where does size trade?** \$70,400 of depth within 2%. A comfortable clip is about \$7,040. *Answer: red for any position above roughly five figures.*

**Q6  -  Who wanted the listing?** Suppose the venue's venture arm participated in the Series A. Then the listing decision and the position point the same way. *Answer: amber to red depending on disclosure.*

**Q7  -  Who started the story?** Suppose the earliest instance of the thesis traces to a research note commissioned by the foundation, amplified by three accounts that did not disclose positions. *Answer: red.*

Six reds and an amber does not mean "NOVA goes to zero." Plenty of tokens with worse scorecards have gone up a great deal, and for a while. What the scorecard tells you is something more useful and more actionable: **the structure of this trade is that you are providing exit liquidity on a published schedule, and you are being compensated for it only if the narrative outruns the supply.** That is a bet you are allowed to make. It is simply a completely different bet from "I think this project will succeed," and the whole value of the checklist is that it stops those two from being confused.

The decision tree above compresses the point into one question. If you can name who is selling to you and what they paid, you have a thesis  -  you are taking the other side of a specific person's specific reason. If you can only name a *class* of holder, you have a partial thesis and should size accordingly. If you cannot name a seller at all, you have not found a trade where nobody is selling. You have found one where you did not look.

## The scorecard

![The red-flag scorecard  -  seven questions, where to look, and the answer that should stop you](/imgs/blogs/reading-the-tape-defending-yourself-as-retail-7.webp)

Printed form, for reuse:

| # | Question | What to look up | Where | Red flag |
| --- | --- | --- | --- | --- |
| 1 | Who owns it? | Allocation %, price paid, vesting per bucket | Project docs, funding databases, block explorer holder list | No table; no prices; a large undefined "ecosystem" bucket |
| 2 | How much floats? | Circulating, total, MC, FDV, FDV/MC | Price aggregators, cross-checked against explorer | Float under ~15%; FDV/MC above ~5×; explorer contradicts the published supply |
| 3 | Who market-makes it? | Firm name, loan size, option strikes, term | Foundation posts, governance forums, explorer labels | Unnamed maker; undisclosed terms; loan is a large share of float |
| 4 | When does supply land? | Every unlock: date, size, % of float, × daily volume | Unlock trackers, DeFi aggregators, the vesting contract | Cliff within 90 days; any unlock above ~5 - 10 days of volume; no schedule |
| 5 | Where does size trade? | Venue list, genuine volume, ±2% depth, perp OI | Aggregator exchange pages, exchange APIs, derivatives dashboards | One venue; volume large relative to a thin book; OI dwarfing spot depth |
| 6 | Who wanted the listing? | Venue's position, launchpad terms, campaign | Exchange announcement, venture-arm portfolio, funding databases | Venue invested and did not say so; bundled launchpad sale |
| 7 | Who started the story? | Earliest instance, who amplified, who was paid | Date-sorted search, disclosure checks, timestamps vs chart | Origin untraceable; undisclosed paid promotion; narrative postdates the move |

A practical note on how to use it: **the checklist is not a scoring system that produces a buy or sell.** It is a way of converting a vague feeling ("this seems risky") into specific, falsifiable statements ("a block worth eleven days of volume unlocks on this date"). Specific statements can be sized around, hedged, timed, or declined. Vague feelings can only be ignored or obeyed.

## Common misconceptions

**"Low market cap means there is room to grow."** Market cap measures only the circulating slice. A \$120 million market cap sitting under a \$2 billion FDV is not a small company; it is a large company with a small proportion of its shares released. The room to grow is real, but so is the 94% of supply that has to be sold to somebody, and the price you pay reflects only the first of those. Compare FDV to FDV across candidates, never market cap to market cap.

**"The unlock is priced in."** Sometimes it genuinely is  -  an unlock that has been discussed for months, in a token with deep derivatives markets where the overhang can be hedged, may well be reflected in the price before it happens. But "priced in" is a claim about a specific market's ability to express a view in advance, and that ability requires borrow to short, depth to trade, and participants paying attention. In a token with a 12% float, no borrow, and one venue, there is no mechanism by which the market could price it in even if everyone agreed about it. Ask *through what instrument* the pricing-in was supposed to have happened.

**"The team is doxxed and the VCs are top-tier, so it is safe."** Reputable backers change the distribution of outcomes; they do not change the mechanics of supply. A top-tier fund still has a fund life, still has limited partners expecting distributions, and still bought at a price you did not get. Its presence is evidence about quality and simultaneously evidence about future selling. Both readings are correct  -  see [a16z Crypto](/blog/trading/crypto-players/a16z-crypto-the-institutional-giant), [Paradigm](/blog/trading/crypto-players/paradigm-and-the-research-driven-fund) and [Polychain and Multicoin](/blog/trading/crypto-players/polychain-multicoin-and-the-thesis-funds) for how these firms actually operate.

**"High volume means it is liquid."** Volume and depth measure different things, and only one of them is hard to fake. A pair can print enormous volume through self-trading while holding almost nothing in the book. Depth is the number that determines your fill; volume is the number that determines whether you noticed the token. Always check the book  -  and if you want the statistical tells that separate real prints from manufactured ones, [detecting wash trading](/blog/trading/onchain/detecting-wash-trading) is the hands-on version.

**"Market makers are the price-support team."** This one is worth correcting carefully because it is half true. A designated market maker's contractual obligation is typically to quote continuously within a spread and size  -  which does stabilise a market in normal conditions. What it is not is a commitment to buy your tokens when everyone sells at once. A maker running a hedged book widens or steps away as inventory risk rises, and that is the *correct* behaviour for the firm and the worst possible moment for you. The distinction is the whole subject of [designated versus principal market making](/blog/trading/crypto-players/designated-versus-principal-market-making), and the underlying economics  -  why a maker's spread is really a fear-of-the-informed premium  -  are in [the market maker's game](/blog/trading/game-theory/the-market-makers-game-inventory-the-spread-and-fear-of-the-informed).

**"If it were manipulation, someone would have stopped it."** Enforcement in this market is slow, jurisdictionally fragmented, and has changed direction more than once. Cases take years to reach a resolution, and the resolution frequently arrives long after the tokens involved have become worthless. Planning around enforcement is planning around a process whose timescale is measured in years and whose posture is set by policy rather than by the facts of your particular loss.

**"I will just get out before the unlock."** Everyone reading the same public calendar has the same plan. The exit is a door whose width you measured in question 5  -  and it narrows exactly when the crowd arrives at it, because that is when makers widen. If your plan requires selling \$250,000 into \$70,400 of depth on a specific known date alongside everyone else with the same idea, it is not a plan. This is a textbook crowded trade, and [crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game) works through why the crowd's shared plan is the thing that makes the plan fail.

<!-- REAL-MARKETS-SECTION -->

## What the checklist cannot tell you

An honest checklist has to say where it stops.

**It cannot tell you whether the project is good.** Every question above is about market structure  -  supply, incentives, liquidity, disclosure. None of them evaluates whether the technology works or whether anyone wants it. A structurally ugly token can belong to an excellent project, and a structurally clean token can belong to a worthless one. The checklist tells you what kind of trade you are in, not whether the thing is worth owning.

**It cannot see private agreements.** The most consequential documents in a token's life  -  market-making contracts, OTC sale agreements, exchange listing terms, side letters with early investors  -  are private by default. You can sometimes infer their existence from on-chain flows  -  [following smart money wallets](/blog/trading/onchain/following-smart-money-wallets) and [a case study in smart money front-running a listing](/blog/trading/onchain/case-study-smart-money-front-ran-a-listing) show what that inference looks like in practice. But inference is not disclosure, and a checklist built on public information will systematically miss the things that were deliberately kept off it. This is the single largest limitation, and it is worth holding in mind whenever a scorecard comes back clean: *clean may mean clean, or it may mean well-concealed.*

**It cannot time anything.** Knowing that 50,000,000 tokens unlock on a date tells you when supply arrives. It does not tell you whether the price falls before, on, or after that date, or whether it falls at all. Supply overhang is a structural headwind, not a trading signal, and treating it as the latter is how people end up short a token that triples.

**It cannot price the tail.** Exchange insolvency, bridge exploits, contract bugs, and outright fraud are not on this list because they are not visible in the tape. The history of this market  -  [FTX](/blog/trading/crypto/ftx-collapse-sam-bankman-fried), [Terra-Luna](/blog/trading/crypto/terra-luna-2022-collapse), [Three Arrows](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion)  -  is largely a history of risks that no amount of order-book reading would have surfaced. The defence against those is not analysis; it is position sizing and custody.

**And it cannot make an illiquid asset liquid.** This is worth ending the section on because it is the failure mode that turns a bad trade into a life event. Everything above helps you decide what to buy and how much. Nothing above helps you sell something for which there is no bid.

## The defence stack: what to do with the answers

![The retail defence stack  -  four layers, each of which survives being wrong about the token](/imgs/blogs/reading-the-tape-defending-yourself-as-retail-8.webp)

A checklist that ends in "so be careful" has not done its job. Here is what the answers actually change, from the foundation upward.

**Layer 1  -  a position size you can be completely wrong about.** This is the base of the stack because it is the only layer that works when every other layer fails. Size is the one variable entirely under your control, and it is the only defence that does not depend on your analysis being right. The checklist informs it: a token scoring six reds gets a fraction of the size a token scoring two ambers gets, not because the reds predict a loss but because they widen the distribution of outcomes.

**Layer 2  -  exit liquidity, not entry price.** Question 5 gives you a number. Use it as a constraint rather than a curiosity: decide the maximum position you could exit inside your tolerance for slippage, in the market conditions you would be exiting in  -  which are worse than the ones you measured. If that maximum is smaller than the position you wanted, the position you wanted was never available. This is also the layer where venue choice matters: a token that trades meaningfully on more than one venue gives you more than one door.

**Layer 3  -  a horizon shorter than the cliff, or long enough to survive it.** Question 4 gives you dates. There are two coherent responses and one incoherent one. Coherent: hold a position sized to survive the full vesting schedule, accepting two years of structural selling as the cost of the thesis. Also coherent: hold a position with an explicit horizon that ends before the cliff, taken as a trade rather than an investment. Incoherent  -  and overwhelmingly the most common  -  is to buy as an investment, discover the calendar afterwards, and convert into a trade under duress.

**Layer 4  -  a written sell rule, decided before you buy.** Written down, with a number in it, before the position exists. Not because rules are magic, but because the moment you most need one is the moment you are least able to make one. Every mechanism in this post  -  the narrative relay, the unlock cliff, the depth that evaporates  -  does its damage during the specific window when your judgement is worst. A rule written in a calm hour is a message to yourself in a panicked one. The market-side machinery that makes that hour arrive  -  forced selling feeding forced selling  -  is in [stop hunts, liquidation cascades and the predator](/blog/trading/game-theory/stop-hunts-liquidation-cascades-and-the-predator), and the on-chain view of who is moving before you are is in [whales, smart money and on-chain wallet watching](/blog/trading/crypto-players/whales-smart-money-and-on-chain-wallet-watching).

Notice what all four layers have in common: **none of them requires you to be right about the token.** That is the design goal. Retail's structural disadvantage is informational, and you cannot out-inform a firm that holds the private agreements. What you can do is build a position whose survival does not depend on winning the information contest  -  which is a strictly easier problem, and the only one available to you.

One line of genuine caution before moving on: everything here is a description of mechanisms, not a recommendation about any asset. Nothing in this post is financial advice, and the right position size in every one of these examples may well be zero.

## Where this goes next

Everything in this post is done from public data, with a browser, in about twenty minutes. That is deliberate  -  a defence you will not actually run is not a defence.

But the checklist has an obvious frontier. Questions 1, 3 and 6 all bottom out in the same limitation: the agreements that matter most are private, and the checklist can only see their shadows. That shadow is on-chain, and it can be read.

Wave 8 of this series takes the checklist from browser tabs to hands-on tooling: tracing a market maker's on-chain footprint to see when inventory actually moves to an exchange, following token flows from a foundation treasury through intermediaries to a venue deposit address, the manipulation playbook as a set of patterns to *recognise* rather than run, the specific on-chain signatures that distinguish organic volume from manufactured volume, forensic reconstruction of cases after the fact, and  -  importantly  -  the limits of tracing, because chain analysis produces far more confident-looking conclusions than it produces correct ones. Those posts do not exist yet, so there is nothing to link; they are where this checklist stops being a reading exercise and starts being a research one.

## When this matters to you

It matters at exactly one moment: the twenty minutes before you buy something, when you are excited and the chart is going up and the checklist feels like an obstacle between you and a decision you have already made emotionally. That is not a coincidence. The moment the checklist is most annoying is the moment it is most valuable, because the conditions that make it annoying  -  urgency, social proof, a rising price, a compelling story  -  are precisely the conditions the machinery described in this series is built to manufacture.

You will not run all seven questions every time, and you do not need to. If you only ever run two, run **question 2 and question 4**: the float and the unlock calendar. They take five minutes between them, they are the two most reliably available pieces of public information in this entire market, and together they answer the question that generates all the others  -  how much supply is coming, and when.

And if you only ever remember one sentence: *if you cannot name who is selling to you, you are not the buyer.*

For the players themselves, start with [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) and the series hub, [crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers). For the mechanics of what those players do to price, [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) is the foundation, and the rest of this wave  -  [the anatomy of a token pump](/blog/trading/crypto-players/the-anatomy-of-a-token-pump), [unlock cliffs and the supply overhang trade](/blog/trading/crypto-players/unlock-cliffs-and-the-supply-overhang-trade), [the low float, high FDV game](/blog/trading/crypto-players/the-low-float-high-fdv-game), [narrative cycles and who sets the story](/blog/trading/crypto-players/narrative-cycles-and-who-sets-the-story), and [MEV: the invisible tax on every trade](/blog/trading/crypto-players/mev-the-invisible-tax-on-every-trade)  -  is each of the seven questions at full depth.

<!-- SOURCES-SECTION -->

## Sources & further reading

The NOVA token, its allocation table, prices, unlocks, market-maker terms, volume, and order book are invented examples, not live market data. The percentages and cutoffs in the checklist are screening heuristics, not universal thresholds. Where a real market snapshot is used in a future application of this checklist, record the venue and timestamp; order-book depth and circulating supply can change.

- [CoinGecko: What Is Market Cap in Crypto and How Is It Calculated?](https://www.coingecko.com/learn/what-is-market-cap-in-crypto)  -  circulating supply, market capitalisation, FDV, and why FDV is theoretical (updated April 10, 2026).
- [CoinGecko: What Is Circulating Supply and Why It Matters](https://www.coingecko.com/learn/what-is-circulating-supply-crypto)  -  circulating, total, and maximum supply and the effect of vesting and unlocks (updated April 28, 2026).
- [Binance Academy: What Is an Order Book and How Does It Work?](https://academy.binance.com/en/articles/what-is-an-order-book-and-how-does-it-work)  -  bids, asks, market orders, limit orders, and market depth (updated June 25, 2026).
- [Ethereum Improvement Proposal 5725: Transferable Vesting NFT](https://eips.ethereum.org/EIPS/eip-5725)  -  technical vocabulary for vesting, cliffs, claimable payouts, and timestamps.
- [SEC: Statement Urging Caution Around Celebrity-Backed ICOs](https://www.sec.gov/newsroom/speeches-statements/statement-potentially-unlawful-promotion-icos)  -  disclosure and conflict-of-interest concerns in paid public promotion (November 1, 2017).
