---
title: "The Lifecycle of a Token: From Seed Round to Unlock Cliff"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "The full pipeline every crypto token walks, from a private seed round years before you can buy to the public unlock cliff, and exactly where each player enters and at what price."
tags: ["crypto", "tokenomics", "vesting", "token-unlocks", "seed-round", "saft", "fdv", "circulating-supply", "cliff", "crypto-players"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Every token you can buy on an exchange already walked a long private pipeline (seed round to private to strategic to TGE to listing to cliff to linear unlocks), and by the time it reaches you, insiders own a big, cheap chunk of the supply on a public unlock schedule you can read in advance.
>
> - Insiders enter at private prices like \$0.02 to \$0.10 while the public buys near the \$0.50 listing price, so the earliest money can be up 5x to 25x on paper before your first order fills.
> - "Circulating supply" is not "total supply." At launch a token often floats only 10 to 20 percent of its tokens; the rest is locked and lands later on a *known* schedule.
> - A single unlock cliff can add more tokens to the tradeable float in one day than everything that was trading the day before. Arbitrum's March 2024 cliff released about 1.1 billion ARB, roughly 87 percent of its circulating supply at the time.
> - Fully diluted valuation (FDV) prices *all* supply, so on a thin float most of the headline valuation is overhang. Tokens launched in 2024 averaged a market-cap-to-FDV ratio near 12 percent (Binance Research, May 2024).
> - The one habit that protects you: read the vesting schedule *before* you buy. Token Unlocks and DefiLlama publish the calendar for free. If the next unlock is a huge fraction of the float, someone is about to become exit liquidity — make sure it is not you.

Here is a question almost nobody asks before they buy a token: *who is selling it to me, when did they buy it, and how much did they pay?*

In the stock market you can at least guess. A company IPOs, insiders are locked up for a defined window enforced by the exchange, and every quarter the firm files a disclosure of who owns what. In crypto there is no such gate. The price you see on listing day is not the beginning of the story — it is the *end* of a long private chain that started years earlier, in rooms you were never in, at prices you never got. By the time a token trades publicly, venture funds already hold a cheap slice of the supply, a market maker is already quoting both sides of the book, and a schedule already exists that says exactly when the locked insider tokens will hit the market and dilute you.

The diagram below is the mental model for this entire post. Read it left to right: an idea becomes a seed round, then private and strategic rounds, then a token generation event and a public listing, and then — on dates fixed in a smart contract — the cliff and the linear unlocks. Each stage has a different buyer paying a different price under a different lock. The whole game of who profits and who provides the exit is decided by *where you enter this pipeline*.

![The lifecycle of a token: a horizontal timeline from seed round through private and strategic rounds, TGE, listing, the month-12 cliff unlock, and linear monthly unlocks, each stage labelled with who buys, at what price, and under what lock.](/imgs/blogs/the-lifecycle-of-a-token-seed-to-unlock-1.webp)

This is not a conspiracy piece and it is not mostly illegal. It is a *plumbing* story, and the plumbing is largely public if you know where to look. Equity markets have venture capitalists and lockups too. What makes crypto different is that the same machinery runs faster, with far weaker disclosure, and lets the earliest buyers exit on a public market years sooner than an equity investor ever could. This post builds that machinery from zero. We define every term, walk a full cap table with real dollar arithmetic, trace circulating supply month by month, size a single unlock against the float, and then ground all of it in a real token — Arbitrum — whose vesting schedule is public and whose famous 2024 unlock cliff is exactly the event this post is named after. No predictions, no advice, no shilling. Just the pipeline.

This is the third post in the [Crypto Players and Power Structure](/blog/trading/crypto/crypto-vc-and-market-makers) series. If you want the overview of *who* the players are — the funds, the market makers, the exchanges — start with [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers). This post zooms into one thing: the *pipeline* every token walks, and where each of those players plugs in.

## Foundations: the building blocks

Before we name a single price, we need a shared vocabulary. Crypto borrows half its words from finance and invents the other half, and the whole story collapses into confusion if any of these is fuzzy. Read this section even if some of it feels familiar — the conflicts later hinge on the precise meaning of "float," "vesting," and "FDV."

### The players and the rounds

A crypto project raises money in *rounds*, just like a startup, except that instead of (or in addition to) selling equity, it sells rights to a future token. From earliest and cheapest to latest and most expensive:

**Pre-seed / idea stage.** The founders and a few angels put in the first money when the project is barely a whitepaper. They pay the lowest price of anyone because they take the most risk — most projects at this stage never ship a token at all.

**Seed round.** The first institutional money. Dedicated crypto venture funds (the kind profiled in [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers)) buy a stake in the project — and rights to a slice of its future tokens — at a very low price per token. A seed price of a cent or two per token is normal.

**Private round.** A larger raise from more funds at a higher price than seed. The project now has a product or a testnet, so the risk is lower and the price is higher. Think a few cents per token.

**Strategic round.** A round aimed at investors who bring something beyond money — an exchange's venture arm, a market maker that will later quote the token, a big ecosystem partner. Strategic money often pays a premium to seed and private (it is buying access and a relationship), but still a deep discount to the eventual public price.

**Public sale.** The first time ordinary buyers can purchase the token directly from the project, often through a launchpad on an exchange. This price is far higher than the private rounds — it is the last discount before the open market.

Two contracts glue these rounds together:

**SAFT (Simple Agreement for Future Tokens).** In the early rounds the token usually does not exist yet. So investors sign a *SAFT*: they pay now for tokens that will be *delivered later*, when the network launches. It is a promissory note for tokens. The SAFT is where the price and the lock terms are written down — and it is private, so you rarely see the exact numbers.

**Token warrant / equity with token rights.** Sometimes investors buy equity in the company that builds the protocol, plus a *right* to a portion of the tokens. This matters because "investors got X percent of the supply" can mean they bought tokens directly *or* that they bought equity and the token allocation reflects that equity stake. Arbitrum, which we will meet later, is the second kind.

### From token birth to public trading

**Token Generation Event (TGE).** The moment the token is actually created (minted) on a blockchain. This is when SAFTs settle — the promised tokens become real — and when any airdrop goes live. TGE is the token's birthday. It is *not* necessarily the same day the token starts trading on a big exchange, though the two are often close.

**Listing.** When the token becomes buyable and sellable on an exchange. A *centralized exchange* (CEX) like Binance or Coinbase lists it on its order book; a *decentralized exchange* (DEX) like Uniswap gets a trading pool. A listing is a price event in itself — being listed on a major venue is a stamp of legitimacy and a burst of new buyers, which is why projects fight for it. We go deep on exchanges as players in [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase).

**Liquidity and the float.** *Liquidity* is how easily you can buy or sell without moving the price. It is *provided* by someone — usually a market maker who is loaned tokens by the project to quote both sides of the order book. The *float* (or free float) is the number of tokens actually free to trade right now. A thin float means a small pool of tradeable tokens, so even modest buying or selling swings the price hard.

### Locks: cliff, vesting, and the unlock

Insiders' tokens are almost never all free at launch. They are *locked* by a smart contract and released on a schedule. Three words describe that schedule:

**Vesting.** The gradual release of locked tokens over time. If 360 million tokens vest linearly over 36 months, then 10 million become free each month.

**Cliff.** A date before which *nothing* unlocks. A one-year cliff means an investor gets zero tokens for twelve months, and then — on the cliff date — a chunk unlocks at once (typically the amount that "would have" vested during the cliff), after which the rest vests gradually. The cliff exists to stop insiders dumping on day one. Its side effect is that the cliff *date itself* becomes a large, sudden, and completely predictable increase in supply.

**Unlock.** Any scheduled date when previously locked tokens become tradeable, increasing the circulating supply. Unlocks are public: they are written into the token contract and republished by trackers like Token Unlocks and DefiLlama. Everyone can see them coming.

### The three supplies

This is the distinction that trips up the most people, so we will be pedantic about it:

- **Circulating supply** — the tokens free to trade *right now*. This is what "market cap" is computed from.
- **Total supply** — every token that currently exists, including locked insider tokens that have not unlocked yet.
- **Max supply** — the most that will *ever* exist (for a fixed-supply token, total will rise toward max as the last tokens are minted; for our examples we treat total = max = a fixed cap).

The gap between circulating and total is the **overhang**: supply that *will* hit the market on a known schedule. A token can show a small, healthy-looking circulating market cap while a mountain of locked supply waits above it.

### The allocation buckets

When a project designs its "tokenomics," it splits a fixed supply into *buckets*. The exact names vary, but almost every token has some version of these six:

| Bucket | Who holds it | Typical lock |
|---|---|---|
| **Team & advisors** | founders, employees, advisors | long lock, long cliff |
| **Investors** (seed / private / strategic) | the venture funds | long lock, cliff at ~1 year |
| **Public sale** | launchpad / IDO buyers | usually unlocked at TGE |
| **Community / airdrop** | early users, given free | often partly unlocked at TGE |
| **Treasury / foundation** | the DAO or foundation | gated by governance |
| **Liquidity / market maker** | loaned to the MM to quote the book | available to make markets |

The picture below shows how a fixed supply might split across those buckets. Notice the colour coding: the amber buckets — seed, private, strategic, and team — are *locked at launch*. Together they are more than a third of all supply, and they are the overhang that lands on the market later.

![Allocation buckets shown as a ranked horizontal bar chart: treasury 26 percent, team 20 percent, community airdrop 18 percent, public sale 10 percent, liquidity and market maker 10 percent, private round 8 percent, strategic round 4 percent, seed round 4 percent, with locked insider buckets highlighted as 36 percent of all supply.](/imgs/blogs/the-lifecycle-of-a-token-seed-to-unlock-2.webp)

And the same six buckets look completely different depending on *which column you read*. The matrix below lays the buckets against price paid, how much unlocks at TGE, the cliff, and full vesting. The red band across the insider rows is the point: the people who paid the least (or nothing) are the ones whose tokens are locked, so they become sellers later, on a schedule.

![A comparison matrix of six allocation buckets against price paid, unlocked at TGE, cliff, and full vesting: team and investors pay little or nothing with a 12-month cliff and 36-month vest, while the public sale pays the most but is fully liquid on day zero.](/imgs/blogs/the-lifecycle-of-a-token-seed-to-unlock-3.webp)

#### Worked example: the simplest possible case

Before the full cap table, here is the one-line version. Suppose a token has a fixed supply of 1,000 tokens. A seed investor buys 40 of them for \$0.02 each — a total of \$0.80. A year later the token lists at \$0.50. The seed investor's 40 tokens are now worth 40 x \$0.50 = \$20. They put in \$0.80 and hold \$20 on paper: a 25x paper gain, before they have sold a single token. That multiple — the ratio of the listing price to the price an insider paid — is the engine that drives everything else in this post. Hold onto it.

Now let us scale that intuition up to a realistic token and put real dollar figures on every stage.

## 1. The rounds: who enters, and at what price

Let us build one hypothetical token and follow it all the way through. We will call it **NOVA**. The numbers are round and illustrative — the point is the *structure*, not the specific token — but they are chosen to mirror how real launches actually look.

NOVA has a fixed maximum supply of **1,000,000,000 (1 billion) tokens**. Here is its full cap table across every round:

| Bucket | Allocation | Tokens | Price / token | Raised |
|---|---|---|---|---|
| Seed round | 4% | 40M | \$0.02 | \$0.8M |
| Private round | 8% | 80M | \$0.05 | \$4.0M |
| Strategic round | 4% | 40M | \$0.10 | \$4.0M |
| Public sale | 10% | 100M | \$0.20 | \$20.0M |
| Team & advisors | 20% | 200M | grant (\$0) | — |
| Community airdrop | 18% | 180M | free (\$0) | — |
| Treasury / foundation | 26% | 260M | — | — |
| Liquidity / market maker | 10% | 100M | loaned | — |
| **Total** | **100%** | **1,000M** | | **\$28.8M** |

Read the price column top to bottom. The seed round paid \$0.02. The private round paid \$0.05 — 2.5 times more — because it entered later, when the project was further along. The strategic round paid \$0.10. The public paid \$0.20 in the sale. Every round buys the *same token*, but each one pays more than the last, because each enters after more risk has been removed. The chart below is that price ladder.

![A column chart of price paid per round: seed \$0.02, private \$0.05, strategic \$0.10, public sale \$0.20, and a dashed reference line at the \$0.50 listing price, annotated with the paper multiple each round holds at listing.](/imgs/blogs/the-lifecycle-of-a-token-seed-to-unlock-4.webp)

Then the token *lists* — starts trading on the open market — at **\$0.50**. Now the ladder pays off. Here is the paper multiple each insider holds the moment the token lists, computed as the listing price divided by what they paid:

#### Worked example: insider paper multiples at launch

- **Seed** paid \$0.02. At \$0.50 that is \$0.50 / \$0.02 = **25x**. The seed fund put in \$0.8M for 40M tokens; at \$0.50 those tokens are worth 40M x \$0.50 = **\$20M** on paper.
- **Private** paid \$0.05. At \$0.50 that is **10x**. Their \$4M is worth \$40M on paper (80M x \$0.50).
- **Strategic** paid \$0.10. At \$0.50 that is **5x**. Their \$4M is worth \$20M on paper (40M x \$0.50).
- **Public sale** paid \$0.20. At \$0.50 that is **2.5x**. The best deal the public got is still a fraction of what seed got.
- **Team** paid nothing — the tokens were a grant. Their paper multiple is, mathematically, infinite. 200M tokens x \$0.50 = **\$100M** of value at zero cost basis.

The single sentence to take from this: **by the time you can place your first order, the insiders are already up somewhere between 2.5x and infinity, and they are up on tokens they have not sold yet.** Whether that paper gain becomes a real gain depends entirely on the next part of the story — the locks — because a 25x on tokens you cannot sell is just a number on a screen.

There is a subtle but important honesty point here. In the early rounds, investors often did not buy tokens at a published price at all — they bought *equity* in the company plus a *right* to tokens. We will see exactly this with Arbitrum, where the "investors" bucket reflects venture funds that bought equity in Offchain Labs, not a public token sale at a disclosed price. So the "price per token" for insiders is frequently *inferred* from the round valuation, not printed on a receipt. That opacity is itself part of the structure: you can see *how many* tokens insiders hold far more easily than you can see *what they paid*.

### What this costs and when it breaks

The price ladder is not a scandal on its own — early investors take real risk and deserve a return, exactly as in equity venture capital. The problem is what happens when the ladder meets a *public* market with a *thin* float and *weak* disclosure. In equity, a seed investor waits five to ten years for an IPO or acquisition, and even then faces an exchange-enforced lockup. In crypto, the same investor can be selling on a liquid exchange **12 months after the token generation event**. That compression — venture-scale entry prices with public-market exits on a one-year horizon — is where the interests of insiders and launch-day buyers collide. The rest of this post is about the mechanics of that collision.

## 2. TGE, the SAFT settlement, and the listing

Between the private rounds and the moment you can buy, three things happen in quick succession, and it is worth slowing down to see them clearly because the marketing blurs them together.

**First, the Token Generation Event.** The token is minted. Every SAFT signed in the seed, private, and strategic rounds now settles: the investors' tokens are created and assigned to their wallets — but *locked*, according to the vesting terms in their contract. The community airdrop also goes live at TGE, and a portion of it is usually immediately claimable. The public sale tokens are delivered, fully unlocked. So at the instant of TGE, the *total* supply is 1 billion tokens (they all exist now), but the *circulating* supply is tiny — only the unlocked pieces.

**Second, liquidity is seeded.** The project loans its market maker the liquidity/MM bucket (100M NOVA in our example) so the MM can quote both sides of the order book from the first second of trading. Without this, the token would have no liquidity and any order would move the price violently. The MM's deal — often a token loan plus call options — is one of the most important and least understood mechanics in all of crypto, and it is the subject of its own post later in this series. For now, just note that the tokens sitting behind the order book on day one are *borrowed*, not sold.

**Third, the listing.** The token starts trading on a CEX order book, a DEX pool, or both. This is the price event the whole pipeline was building toward, and it is where the public arrives. The listing price is set by where the first buyers meet the first sellers — and on a thin float, it can spike well above the public-sale price. In our example NOVA sold to the public at \$0.20 and *lists* at \$0.50, a 2.5x "listing pop" that looks great in a headline and means the public sale buyers are already up — but also means everyone now anchors on \$0.50, the highest price in the token's short life.

Here is the trap hidden in that sequence. At the listing, the token has a **fully diluted valuation** of \$0.50 x 1 billion = **\$500 million**. That number goes in every headline: "NOVA launches at a \$500M valuation." But almost none of that supply is actually trading. We will quantify exactly how little in the next two sections.

### What this costs and when it breaks

The listing pop is the moment retail is most likely to buy and least likely to check the supply schedule. The story is loud ("new L1, backed by top funds, up 2.5x on day one"), the float is thin, and the FDV headline sounds enormous and therefore legitimate. Everything about the moment is engineered to make you anchor on the price and ignore the calendar. The calendar is where the risk lives.

## 3. Circulating vs total supply over the first 24 months

Now we make the supply concrete. The single most useful habit you can build as a token buyer is to stop looking at *price* alone and start looking at *how many tokens will exist and trade over time*. Let us trace NOVA's circulating supply month by month.

At the **listing (month 0)**, only two buckets are actually liquid:

- The **public sale**: 100M tokens, fully unlocked.
- The **early portion of the community airdrop**: roughly 30% of the 180M airdrop is claimable at TGE, about 50M tokens.

Everything else is locked: the team (200M), the investors (seed + private + strategic = 160M), the treasury (260M, gated by governance), the market-maker loan (100M, behind the book), and the rest of the airdrop (130M, vesting over the first year). So the starting **float is about 150M tokens** — public sale plus early airdrop — out of a billion. That is a **15% float**. For the first twelve months, roughly that same thin slice is what trades, because the big insider buckets are behind a one-year cliff.

Then comes **month 12: the cliff.** The team and investor tokens have a 12-month cliff and then vest linearly over the following 24 months (a 36-month total schedule). On the cliff date, the twelve months of vesting that were "held back" all release at once. With 360M insider tokens over 36 months, the monthly rate is 10M tokens, so the cliff releases 12 x 10M = **120M tokens in a single day.** After that, another 10M unlock every month for 24 more months.

The chart below draws exactly this. Blue is the float that actually trades — a thin ~150M base for the whole first year. Gray is the locked overhang towering above it. And amber is the cliff: a 120M block that lands in one day at month 12 and roughly doubles the float.

![Circulating vs total supply over 24 months: a thin blue float of about 150 million tokens for the first year, a large gray locked region above it, and an amber 120-million cliff that lands at month 12, with total supply fixed at 1 billion.](/imgs/blogs/the-lifecycle-of-a-token-seed-to-unlock-5.webp)

#### Worked example: the supply schedule month by month

- **Month 0 (listing):** ~150M circulating (15% of supply). Total supply 1,000M. FDV \$500M at \$0.50.
- **Months 1 to 11:** float stays near 150M. The dominant buckets are all behind the cliff, so despite the \$500M headline, only ~15% of the token is in play.
- **Month 12 (cliff):** +120M unlock in one day. Float jumps from ~150M to ~270M — an **80% increase overnight**.
- **Months 13 to 36:** +10M unlock every month. By month 24, circulating supply is roughly 270M + (12 x 10M) = **~390M**, and it keeps climbing toward the full billion as the schedule runs its course.

Sit with the shape of that curve. For a full year, a buyer sees a token with a \$500M "valuation" and assumes it is a large, established asset. In reality only 15% of it trades, and a schedule already exists that will more than double the tradeable float within twelve months and keep diluting for two years after that. None of this is hidden. It is in the token contract and republished on free dashboards. It is simply *ignored*, because the price chart is more exciting than the supply chart.

### What this costs and when it breaks

The break happens when new demand fails to keep pace with new supply. Every unlock adds tokens that *want to be sold* — insiders realizing a 25x, a team funding operations, a fund returning capital to its own investors. If the number of new buyers does not grow at least as fast as the circulating supply, the price has to fall to clear the market. That is not a manipulation; it is arithmetic. Supply goes up on a schedule; if demand does not, price goes down. The next two sections put numbers on that pressure.

## 4. The cliff: when one unlock is bigger than the whole float

The word in this post's title is *cliff*, and it deserves its own section because it is the single most violent moment in a token's supply life.

Recall the setup at month 11, the day before NOVA's cliff. The float is ~150M tokens: the public sale plus the early airdrop. The insiders have sold *nothing* — their tokens are still locked. The order book has been quoting against that thin 150M float for a year.

Then the cliff hits and **120M tokens unlock in a single day.** That is not 120M added to a billion-token ocean — it is 120M added to a *150M* tradeable float. The comparison that matters is not "unlock vs total supply" (120M / 1,000M = 12%, which sounds small); it is "unlock vs *circulating float*" (120M / 150M = **80%**). In one day, the amount of newly tradeable supply is 80% of everything that was trading the day before.

![A before-and-after comparison of the cliff: the day before, a ~150 million token float made of public sale plus early airdrop with insiders having sold nothing; on cliff day, +120 million unlocks in one day, the float jumps to ~270 million (up 80 percent overnight), and sellers outnumber buyers on a thin book.](/imgs/blogs/the-lifecycle-of-a-token-seed-to-unlock-6.webp)

#### Worked example: sizing a single unlock against the float

The right way to size any unlock is a two-step calculation you can do in your head:

1. **Unlock size:** 120M tokens.
2. **Circulating float before the unlock:** ~150M tokens.
3. **Unlock as a percentage of float:** 120M / 150M = **80%.**

An unlock worth 80% of the float is enormous. Compare it to the same unlock measured against total supply — 120M / 1,000M = 12% — and you can see why projects and their supporters prefer the second framing. "Only 12% of supply unlocks" sounds routine. "The tradeable float grows 80% in a day" sounds like what it is: a wall of potential sellers arriving into a small pool of buyers. Both numbers are true. Only one tells you about the pressure on *price*, because price is set at the margin, in the float, not against the total supply.

Not every unlock is sold, of course. Some insiders hold. Some tokens go to a foundation that stakes rather than sells. An unlock can even be neutral or bullish if the recipients are long-term aligned. But the *base case* for a cliff that hands cheap tokens to funds sitting on a 10x to 25x paper gain is straightforward: a meaningful fraction gets sold, into a float that is far too thin to absorb it without the price moving. That is why cliffs are traded, feared, and — as we will see with Arbitrum — endlessly discussed months before they arrive.

### What this costs and when it breaks

The cliff is the cleanest illustration of the whole post's thesis. The people selling into your buy orders on cliff day are insiders whose cost basis is a fraction of the current price, whose tokens you did not know were about to unlock unless you checked, and whose selling was scheduled a year in advance in a public contract. If you bought in the twelve months before the cliff without reading the schedule, you were buying into a known future flood of supply. The defense is not sophisticated — it is *reading the calendar* — which is why the last section of this post is about exactly that.

## 5. FDV vs float: the low-float, high-FDV illusion

We keep mentioning that \$500M "valuation." Now let us take it apart, because the gap between what a token is *quoted* at and what it would be *worth if all its supply traded* is one of the most important and most misunderstood ideas in crypto.

There are two ways to measure a token's size:

- **Circulating market cap** = price x *circulating* supply. This is the value of the tokens that actually trade. For NOVA at listing: \$0.50 x 150M = **\$75M.**
- **Fully diluted valuation (FDV)** = price x *total* supply. This is what the token would be worth if *every* token, including all the locked ones, traded at the current price. For NOVA: \$0.50 x 1,000M = **\$500M.**

The ratio between them, **MC / FDV**, tells you how much of the "valuation" is real, trading value versus locked overhang. For NOVA that ratio is \$75M / \$500M = **15%**. In plain words: 85% of the headline valuation is supply that has not yet hit the market.

![FDV vs circulating market cap shown as one stacked column: a small blue circulating market cap of \$75 million (15 percent) at the bottom and a large amber locked value of \$425 million (85 percent) above it, with the formulas for FDV and market cap and a note that 2024 launches averaged a 12.3 percent market-cap-to-FDV ratio.](/imgs/blogs/the-lifecycle-of-a-token-seed-to-unlock-7.webp)

#### Worked example: why a low MC/FDV ratio is a warning

Two tokens both trade at \$0.50 and both have a \$75M circulating market cap. Token A has 150M of its 1,000M supply circulating (15% float, \$500M FDV). Token B has 800M of its 1,000M circulating (80% float, and therefore a \$625M... let us keep it simple: Token B has 750M circulating so its market cap and FDV are close). For Token A, there are 850M locked tokens waiting to be sold at some price over the next few years; for Token B, there are only 250M. Same price today, wildly different future supply pressure. **A low MC/FDV ratio means most of the dilution is still ahead of you.** All else equal, you would rather buy the token whose supply is already mostly in the market, because there is less overhang left to fall on you.

This is not a fringe concern. In May 2024, Binance Research published a study titled "Low Float and High FDV: How Did We Get Here?" documenting that tokens launched in 2024 had an average MC/FDV ratio of about **12.3%** — meaning the typical 2024 launch had roughly seven-eighths of its valuation locked up — and that these tokens often had under 20% of supply circulating. The same report estimated that roughly **\$155 billion** of tokens would unlock between 2024 and 2030, and noted that well over 80% of tokens newly listed on Binance had declined in value. The low-float, high-FDV launch is not an accident of a few projects; it was the dominant launch structure of the era, and it has a specific loser: whoever buys the thin float at the high FDV and then holds through the unlocks.

### What this costs and when it breaks

FDV is a seductive number because it is big, and big sounds legitimate. But FDV is a *promise about the future*, not a fact about the present. It says "if everyone who will ever own this token valued it at today's price, the whole thing would be worth \$500M." The moment the locked supply starts unlocking, that promise gets tested against real demand — and if the demand is not there, the price falls to a level where the *diluted* supply clears. A token with a small float and a huge FDV is a coiled spring of future selling. Reading the MC/FDV ratio is how you see the spring before it releases.

## 6. How it shows up in price: the overhang trade

We have built every piece. Now let us put them together into the mechanism by which all of this reaches the price on your screen — the thing the series calls "how it shows up in price."

The core insight is that the selling pressure from unlocks is *predictable*, because the unlock dates are public. That predictability changes how the price behaves. Sophisticated traders do not wait for the cliff; they anticipate it. If everyone can see that 120M tokens — 80% of the float — will hit the market on a fixed date, then rational holders sell *before* the date to avoid the flood, and the price tends to drift *down into* the unlock rather than crashing precisely on it. The unlock is often partly "priced in" ahead of time. Sometimes the actual unlock day is even a relief ("sell the rumor, the news is already out"), which is why unlock trades are subtle rather than mechanical.

The chain below is the mechanism. A public unlock date leads to insiders receiving a block of tokens, which they sell into a thin float, which pushes the price down into the unlock — and the retail buyers who bought the story and did not read the calendar are the ones providing the exit liquidity.

![How the overhang shows up in price: a five-step chain from a public unlock date to insiders receiving 120 million tokens, selling into a thin 150 million float, price drifting down into the unlock, and retail buying the dip as exit liquidity, with a retail-defense strip advising readers to read the vesting schedule before buying.](/imgs/blogs/the-lifecycle-of-a-token-seed-to-unlock-8.webp)

"Exit liquidity" is the blunt term for the last part of that chain. It means the buyers whose orders let earlier holders sell without crashing the price. When a fund sitting on a 25x wants to realize gains on tokens it received at a cliff, it needs someone on the other side of the trade. On a thin float, that someone is disproportionately retail buying because the token is in the headlines — often *because* the unlock and the surrounding narrative put it there. This is the uncomfortable core of the whole series: at each step of the pipeline, someone profits, and the structure quietly arranges for the last, least-informed buyer to be on the other side of the most-informed seller.

#### Worked example: the overhang as predictable pressure

Suppose NOVA trades at \$0.50 for its first year on a 150M float, and a 120M cliff is scheduled for month 12. A trader who understands the pipeline reasons like this: "In month 12, tokens worth 80% of the current float unlock, held mostly by funds up more than 10x. Even if only a third of that gets sold, that is 40M tokens hitting a 150M float — a ~27% increase in effective sell-side supply. Buyers are unlikely to grow that fast in a month. So the risk is asymmetric to the downside into the unlock." That reasoning, multiplied across many holders, is *why* tokens with heavy near-term unlocks so often bleed lower in the weeks before a cliff. The selling does not wait for permission; it front-runs the calendar everyone can read.

### What this costs and when it breaks

The mechanism breaks — in retail's favor — only when genuine demand outpaces the scheduled supply. That does happen: a token with real usage, growing revenue, and a narrative that pulls in new capital faster than it unlocks can absorb its overhang and rise anyway. But that is the exception, and it is knowable in advance only by comparing the demand story to the supply schedule. Which brings us to the one habit that turns this entire post into a usable defense.

## Common misconceptions

**"The market cap is small, so there is lots of room to grow."** The market cap you see is *circulating* market cap — price times the thin float. The number that bounds "room to grow" against future dilution is the FDV. A token with a \$75M market cap and a \$500M FDV is not a small token; it is a token whose price is being held up by a tiny float while 85% of the supply waits to be sold. Always check the MC/FDV ratio before you conclude anything from "small market cap."

**"Insiders are locked, so they can't dump on me."** Locks are temporary, and the *unlock dates are public and knowable*. "Locked" does not mean "gone"; it means "arriving later, on a schedule you can read." The lock is a delay, not a cancellation, and the whole overhang trade exists precisely because the delay ends on a known date.

**"A token unlock is bad, so the price always crashes on the unlock day."** Not necessarily. Because unlocks are predictable, the selling is often spread out and partly priced in *before* the date. Prices frequently drift down *into* an unlock and can even bounce on the day itself. The unlock matters, but its effect is anticipatory, not a mechanical crash you can trade by simply shorting the exact date.

**"FDV is just a theoretical number, so I can ignore it."** FDV is theoretical in the sense that not all supply trades today — but every locked token in that FDV is a real claim that will become tradeable on a schedule. Ignoring FDV is ignoring the future supply that will dilute you. It is the single most important number for judging how much overhang sits above the current price.

**"A token is basically a stock, so the same rules protect me."** It is not. There is no IPO gate, no exchange-enforced lockup, no mandatory quarterly disclosure of who owns what, and insiders can exit on a public market on a one-year horizon instead of waiting for an acquisition. The structural gap between a token and a share is the subject of its own post in this series, [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) — and that gap is what makes the entire pipeline in this post possible.

**"If the project is backed by top funds, the tokenomics must be fair."** Top-tier backing tells you the project cleared a diligence bar; it tells you nothing about whether *your* entry price and the unlock schedule are favorable to you. In fact, heavy VC backing often correlates with a lower float and a higher FDV, because more private capital was bid in before launch — exactly the structure the Binance Research report flagged. Backing is a signal about the project; the cap table is the signal about your trade.

## How it shows up in real markets

The hypothetical NOVA made the arithmetic clean. Now let us ground it in real tokens with public, documented schedules. Everything below is sourced and dated; the running example's round numbers were illustrative, but these are real.

### 1. Arbitrum's tokenomics: the buckets, from the docs

Arbitrum (ARB) is one of the largest Ethereum layer-2 networks, and its tokenomics are unusually well documented, which is why it is our anchor case. ARB has a fixed supply of **10 billion tokens**. According to the Arbitrum Foundation's own distribution documentation, the allocation is: **DAO treasury 42.78%**, **Offchain Labs team and advisors 26.94%**, **investors 17.53%**, **user airdrop 11.62%**, and **airdrop to DAOs 1.13%**. Add the airdrops and the treasury together and about 55.5% went to the "community" side; the team and investors together hold about **44.5%** — the locked insider block, the real-world version of NOVA's amber buckets.

Notice the honesty point from Section 1 in action. ARB's "investors" did not buy tokens at a public price. They were the venture funds that backed **Offchain Labs**, the company behind Arbitrum, which raised a **\$120 million Series B in August 2021 led by Lightspeed at a \$1.2 billion valuation**, with participants reportedly including Polychain, Pantera, Ribbit, Redpoint, Mark Cuban, and Alameda Research (TechCrunch, Cointelegraph, September 2021). Their 17.53% token allocation reflects that *equity* stake, not a disclosed per-token sale price — a clean example of why insider "prices" in crypto are so often inferred rather than printed.

### 2. Arbitrum's cliff: the event this post is named after

Here is the case study that makes the whole "cliff" concept concrete. ARB's token generation event was **March 16, 2023** (the community airdrop became claimable about a week later, on March 23, 2023). All team and investor tokens were subject to a **four-year lockup with a one-year cliff** measured from the March 16, 2023 TGE, then monthly vesting for the remaining three years — precisely the cliff-then-linear structure we built for NOVA, just at a larger scale and a 48-month total.

That one-year cliff fell on **March 16, 2024**. On that single day, roughly **1.11 billion ARB** — the team and investor tokens that had accrued during the cliff year — unlocked at once. The math mirrors NOVA exactly: team plus investors held about 4.447 billion ARB (26.94% + 17.53% of 10 billion); over a 48-month schedule with a 12-month cliff, the cliff releases 12/48 = 25% = about 1.11 billion tokens in one shot, after which roughly **92.6 million ARB unlock every month** from April 2024 through March 2027 (the remaining 3.34 billion over 36 months). Those monthly figures are published on unlock trackers like DefiLlama and Token Unlocks / Tokenomist.

Now size the cliff against the float, the way this post teaches. At the time, ARB's circulating supply was about **1.275 billion tokens** — which is almost exactly the airdrop allocation (11.62% + 1.13% = 12.75% = 1.275 billion), because for its first year the tradeable float was essentially *just the airdrop*, with the big insider buckets locked. So the March 2024 cliff of ~1.11 billion ARB was about **87% of the circulating supply** (CoinDesk reported the ~\$1.2 billion unlock in August 2023; the ratio was widely discussed as the date approached). At the going price near \$1.12 that day, the unlock was worth roughly **\$1.2 to 1.24 billion**. That is the textbook cliff: a single, scheduled, publicly known day on which the tradeable supply nearly doubled. It was on every unlock calendar for a year in advance — which is exactly the point. The information was free; using it was optional.

### 3. Optimism: the same structure, a different schedule

Arbitrum's great rival, Optimism (OP), shows that the pattern is a genre, not a one-off. OP's allocation (per its published tokenomics and community docs) puts about **19% with core contributors** and **17% with investors**, alongside large ecosystem, public-goods, and airdrop buckets. Core contributors and investors were subject to a one-year full lock from the May 2022 launch, then multi-year vesting — the same "lock, cliff, then vest" grammar. The specific percentages and dates differ from Arbitrum, but the structure is identical: a big insider block, locked at launch, released on a schedule that any buyer can read on DefiLlama or Token Unlocks before deciding to buy.

### 4. The 2024 low-float, high-FDV wave

Zoom out from any single token and you get the macro version of Section 5. The Binance Research report of May 2024 quantified the era: 2024's token launches averaged a market-cap-to-FDV ratio of about **12.3%**, meaning the typical new token had roughly seven-eighths of its valuation locked; circulating supply was frequently under 20%; and an estimated **\$155 billion** of tokens were scheduled to unlock between 2024 and 2030. The report's blunt observation — that well over 80% of newly listed tokens on Binance had fallen — is the aggregate footprint of the overhang trade playing out across hundreds of tokens at once. When an entire cohort launches at a thin float and a high FDV, and then unlocks on schedule into demand that cannot keep pace, the average outcome is not a mystery. It is Section 3's arithmetic, repeated.

### 5. When the overhang is absorbed

For balance: the overhang does not doom every token, and pretending it does is as lazy as ignoring it. Tokens with real, growing usage can out-run their unlock schedules. The test is always the same comparison — is new demand growing at least as fast as new supply? A protocol whose revenue, users, and narrative pull in fresh capital faster than its tokens unlock can rise straight through its cliffs; a protocol coasting on a launch narrative cannot. The unlock schedule tells you the supply side of that equation with near-certainty. Your job as a buyer is to form an honest view of the demand side and compare the two. Most of the time, the supply side is the part people never bother to look up — which is the entire reason the schedule is such an edge.

## When this matters to you

If you ever buy a token — or even just want to understand why the ones you watch behave the way they do — the pipeline in this post is a checklist you can run in about ten minutes, for free, before you commit a dollar. This is educational, not financial advice; the point is not to tell you what to buy, but to make sure you can see who is on the other side of the trade.

**Read the vesting schedule before you buy.** This is the whole defense in one sentence. Pull up the token on **Token Unlocks (tokenomist.ai)** or **DefiLlama's unlocks page**, both free, and look at the calendar of upcoming unlocks. If there is a large cliff in the next few months, you are considering buying into a scheduled wave of supply.

**Size the next unlock as a percentage of float, not of total supply.** "12% of supply unlocks" and "the float grows 80%" can be the same event. Always divide the unlock by the *circulating* supply, because that is where price is set. The float-relative number is the one that tells you about pressure.

**Check the MC/FDV ratio.** A ratio in the low teens — the 2024 average was about 12% — means most of the dilution is still ahead of you. It is not automatically disqualifying, but it tells you how much overhang you are buying above.

**Find out who holds the locked supply and what they paid.** The cap table (allocation percentages) is usually public even when the round *prices* are not. Big team and investor buckets mean a big future overhang from holders whose cost basis is far below the current price. On-chain tools like Etherscan and Arkham can show you the insider wallets and, sometimes, the flows out of them around unlock dates — the subject of the series post on [reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table).

**Compare the supply schedule to the demand story.** The unlock schedule is a near-certain forecast of future supply. Ask honestly whether the project's usage and inflows are growing fast enough to absorb it. If you cannot make that case, you are betting that new buyers will keep arriving faster than insiders sell — which is a bet on the crowd, not on the token.

The tokens in this post were mostly a fictional example built for clean arithmetic, but the structure is real, the tools are free, and Arbitrum's cliff was a genuine, scheduled, billion-dollar event that anyone could see coming a year out. The pipeline is not hidden. It is published, on dashboards, in contracts, in foundation docs. The edge does not come from secret information; it comes from bothering to read the information that is already public — the same edge that runs through the rest of this series, from [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) to the mechanics of [how the market makers who quote these tokens actually get paid](/blog/trading/crypto/crypto-vc-and-market-makers).

## Sources & further reading

Primary sources behind the headline figures in this post:

- **Arbitrum Foundation — ARB airdrop eligibility and distribution** (docs.arbitrum.foundation): the official 10 billion supply and allocation breakdown (DAO treasury 42.78%, team 26.94%, investors 17.53%, user airdrop 11.62%, DAO airdrop 1.13%) and the four-year lockup with one-year cliff for team and investor tokens.
- **DefiLlama — Arbitrum unlocks** and **Token Unlocks / Tokenomist.ai — Arbitrum**: the vesting schedule, the ~1.11 billion March 16, 2024 cliff, and the ~92.6 million ARB monthly unlocks from April 2024 through March 2027.
- **CoinDesk, "Arbitrum Will Unlock \$1.2B ARB in March 2024" (August 16, 2023)**: the size and timing of the cliff, and the circulating-supply context (~1.275 billion) that made it roughly 87% of float.
- **TechCrunch / Cointelegraph (August–September 2021)**: Offchain Labs' \$120 million Series B led by Lightspeed at a \$1.2 billion valuation, with Polychain, Pantera, and others participating.
- **Binance Research, "Low Float and High FDV: How Did We Get Here?" (May 2024)**: the ~12.3% average MC/FDV for 2024 launches, sub-20% circulating floats, the ~\$155 billion of 2024–2030 unlocks, and the observation that over 80% of newly listed tokens on Binance had declined.
- **Optimism tokenomics** (gov.optimism.io and Tokenomist.ai): the OP allocation (core contributors ~19%, investors ~17%) and the one-year lock plus multi-year vesting structure.

Tools you can use for free to run the checklist:

- **Token Unlocks (tokenomist.ai)** and **DefiLlama Unlocks** — vesting schedules and upcoming unlock calendars.
- **CoinGecko / CoinMarketCap** — circulating vs total vs max supply, market cap, and FDV for any listed token.
- **Etherscan / Arkham / Nansen** — on-chain wallet forensics to see insider allocations and flows around unlock dates.

Related posts in this series and on this blog:

- [Crypto VC and market makers: the real power structure behind the tokens](/blog/trading/crypto/crypto-vc-and-market-makers) — the series hub: who the funds and trading firms are.
- [Why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) — the structural gap that makes this whole pipeline possible.
- [Follow the money: reading a token's cap table](/blog/trading/crypto-players/follow-the-money-reading-a-tokens-cap-table) — the on-chain forensics companion to this post.
- [The FTX collapse and Sam Bankman-Fried](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) and [the Terra-Luna 2022 collapse](/blog/trading/crypto/terra-luna-2022-collapse) — what happens when the conflicts in this structure are taken to their extreme.
- [Three Arrows Capital and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion) — how leverage on top of illiquid, locked tokens cascaded through the market.
