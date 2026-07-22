---
title: "Polychain and Multicoin: The Thesis Funds That Bet Big and Concentrated"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "How high-conviction, concentrated crypto funds like Polychain and Multicoin make money by betting big on a single thesis — and how that model produces returns and drawdowns that swing an order of magnitude in a single year."
tags: ["crypto", "venture-capital", "hedge-funds", "polychain", "multicoin", "solana", "concentration", "drawdown", "reflexivity", "crypto-players", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 38
---

> [!important]
> **TL;DR** — A *thesis fund* concentrates most of its money into a few high-conviction bets — often one or two layer-1 blockchains — instead of spreading it thin. That is why Polychain and Multicoin could compound thousands of percent on the way up, and why the same funds lost most of their assets in a single bad year.
>
> - **Concentration is the whole strategy and the whole risk.** A 30% single-token position turns a token's 3x into a +60% fund year — and its −70% into a −21% fund year. One bet decides the result.
> - **These funds run two books at once.** A *liquid* token book they can sell on an exchange any day, and a *locked* venture book that vests over years and cannot be sold. The same thesis behaves oppositely in the two books when the price turns.
> - **Multicoin's Solana bet is the archetype.** It reportedly bought SOL in the 2018 seed round near \$0.04; SOL later peaked near \$260 (Nov 2021), then fell to roughly \$8 (Dec 2022) — a full round-trip in one cycle.
> - **The volatility is brutal by design.** Multicoin's hedge fund was reported up 20,287% since inception (to investors, late 2021), then lost 91.4% in 2022 (investor letter, per CoinDesk). Polychain's hedge fund returned +2,278.8% in 2017, then −60.4% in 2018 (per CoinDesk).
> - **The one number to remember:** a −91% loss needs a +1,011% gain just to break even. Deep drawdowns are near-fatal, and concentration is what makes them possible.

Why would a professional investor put a third of a billion-dollar fund into *one* cryptocurrency, when every textbook ever written says to diversify? And why would sophisticated limited partners — pensions, endowments, family offices — hand their money to a manager who does exactly that?

The answer is the subject of this post. There is a species of crypto fund built on the opposite of the diversify-everything instinct: it makes a small number of enormous, high-conviction bets and holds them through drawdowns that would get an ordinary manager fired. When it works, the results are among the best in the history of investing. When it fails, the fund can lose most of its value in a matter of weeks. Two firms are the archetypes — [Polychain Capital](/blog/trading/crypto/crypto-vc-and-market-makers), founded by Olaf Carlson-Wee, and Multicoin Capital, founded by Kyle Samani and Tushar Jain — and their story is the clearest way to understand how conviction becomes a business model, and how a fund's conviction can both *move* a token and get *moved* by it.

![A left-to-right flywheel: a high-conviction thesis leads to a concentrated position held in a liquid book and a locked venture book, the fund accumulates size into a thin float, the token price re-rates up, the fund's NAV compounds, and louder conviction pulls in new capital.](/imgs/blogs/polychain-multicoin-and-the-thesis-funds-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. Read it left to right. A thesis fund starts with a *conviction* — a belief that one blockchain, one design, one bet will win big. It concentrates a large share of the fund into that bet, held partly in a liquid token book and partly in a locked venture book. Because the position is large relative to the token's available supply, the fund's own buying helps lift the price. When the price rises, the fund's net asset value compounds — and the louder the fund's public conviction becomes, the more new capital it pulls in, some of which flows back into the same bet. That feedback loop is the source of both the spectacular returns and the spectacular fragility. By the end you will be able to look at a chart and see the fund inside it.

## Foundations: the building blocks

Before we can talk about Polychain and Multicoin specifically, we need a shared vocabulary. Crypto borrows words from traditional finance and bends them, so the argument turns to mush if any term is fuzzy. None of this assumes a finance background — read it even if the words look familiar, because the later mechanics hinge on their precise meaning.

### A fund, an LP, NAV, and AUM

A **fund** is a pool of other people's money that a manager invests on their behalf. The people who put money in are **limited partners** (LPs) — "limited" because their downside is limited to what they invested and they don't run the day-to-day. The manager is the **general partner** (GP). The GP typically earns a **management fee** (often around 2% of assets per year, to keep the lights on) and **carried interest**, or "carry" (often around 20% of the profits, which is where the real money is). This is the classic "2 and 20" structure, and it means the GP's biggest reward comes from making the fund's value go *up*.

Two numbers describe a fund's size and value:

- **AUM** — *assets under management* — is the total dollar value of everything the fund holds right now. If the fund manages \$1 billion, its AUM is \$1 billion.
- **NAV** — *net asset value* — is the value of one investor's share of the fund, or equivalently the fund's total value after fees. When people say a fund "was up 300%," they mean its NAV grew 300%.

The key thing to internalize: in crypto, AUM and NAV move *with the market*, violently. A fund holding tokens that triple sees its AUM triple without raising a dollar of new money. A fund holding tokens that fall 80% sees its AUM collapse the same way. This is not a bug in the reporting; it is the nature of holding volatile assets.

### Thesis and conviction

A **thesis** is a specific, falsifiable belief about the future that an investment is built to express. Not "crypto will go up" — that's a mood, not a thesis. A real thesis is "this particular blockchain's design (fast, cheap, one global state) will win the majority of on-chain activity, and its token will capture that value." **Conviction** is how strongly you hold that thesis — specifically, how much of your fund you're willing to bet on it and how long you'll hold through pain. A *high-conviction, thesis-driven fund* is one that translates a strong belief into a large, concentrated position and refuses to sell it just because the price fell.

> Conviction is not the same as being right. It is a willingness to be *very* wrong or *very* right, with little in between.

### Concentration vs diversification

**Diversification** is spreading your money across many bets so that no single one can sink you. If you hold 50 tokens at 2% each and one goes to zero, you lose 2%. **Concentration** is the opposite: putting a large share into a few positions so that a single bet can *make* the fund — or break it.

![A two-column comparison: a concentrated thesis fund with Token A at 35% of NAV, Token B at 20%, Token C at 15%, where one bet can make the year; versus a diversified fund whose largest position is 6%, top ten are about 45%, holding 50+ positions, where no single token can sink it.](/imgs/blogs/polychain-multicoin-and-the-thesis-funds-2.webp)

The figure makes the trade-off concrete. On the left, a concentrated fund's biggest position is 35% of NAV — a single token's move dominates the entire result. On the right, a diversified fund's biggest position is 6%; the outcome is an average of many bets, and no one of them can ruin the year. Neither is "correct." Diversification buys you survival at the cost of ever having an extraordinary year. Concentration buys you the chance at an extraordinary year at the cost of survival. Thesis funds choose the second deal on purpose, and the arithmetic of that choice is the first worked example below.

### The liquid book and the venture book

Here is the wrinkle that makes crypto funds different from both traditional hedge funds and traditional venture funds: a single crypto thesis fund often runs *two books at once*.

- A **liquid token book** holds tokens that already trade on exchanges. The fund can buy and sell them any day, and their value is **marked to market** — updated to the current trading price continuously. This behaves like a **hedge fund**: nimble, priced daily, redeemable.
- A **locked venture book** holds early-stage positions — tokens bought in a seed or private round before they trade, or equity in a startup — that are subject to a **vesting schedule** or **lock-up**: a contractual period (often one to four years) during which they cannot be sold. This behaves like a **venture-capital fund**: illiquid, long-horizon, marked at the last funding round rather than a live price.

![A two-row matrix comparing the liquid token book and the locked venture book across four questions: can you sell today, how it is valued, the lock-up, and what happens in a crash. The liquid book can sell on-exchange, is marked daily, has no lock-up, but sells into weak bids in a crash; the locked venture book cannot sell, is held at its last-round mark, vests over one to four years, and is stuck at a paper loss in a crash.](/imgs/blogs/polychain-multicoin-and-the-thesis-funds-3.webp)

The two books diverge exactly when it matters most. In a crash, the liquid book *can* be sold — but only into falling bids, so selling itself pushes the price down. The venture book *cannot* be sold at all; the fund watches the mark drop and can do nothing, holding a paper loss it cannot realize. A thesis fund's fate in a downturn is decided by the mix of these two books and by how correlated they are — and in a fund concentrated on one ecosystem, they are almost perfectly correlated, because the liquid token and the venture positions are all bets on the *same thesis*.

This dual nature is what makes a crypto thesis fund a genuinely new animal. In traditional finance the two jobs are done by two different kinds of fund, run by different people, with different rules. Crypto smashed them together, because a single blockchain thesis can be expressed both as a locked seed investment *and* as a liquid token you trade — often at the same time, in the same fund. The table makes the three models legible:

| | Traditional hedge fund | Traditional VC fund | Crypto thesis fund |
|---|---|---|---|
| **What it holds** | Liquid stocks, bonds, derivatives | Illiquid startup equity | Liquid tokens **and** locked early positions |
| **Liquidity** | Sells any day | Locked ~7–10 years | Split: one book liquid, one locked |
| **Valuation** | Marked to market, daily | Last funding round | Both — a live price and a stale mark |
| **Time to exit** | Minutes to days | A decade | Days for tokens; years for the venture book |
| **Volatility of NAV** | Moderate | Smoothed (rarely re-marked) | Extreme — repriced live on volatile tokens |
| **Public exit** | Already public | IPO/acquisition, years out | Token listing, often within *months* |

Look at the bottom-right cell of that last row: the crypto-specific twist is that an early token investment can reach a *public* market — a listing on an exchange — years earlier than a startup equity stake could ever IPO. That early liquidity is the superpower and the curse of this model. It lets a fund realize venture-style gains far sooner than a normal VC could; it also means the fund's private, high-conviction bet gets a live, twitchy public price stamped on it every second, which is what makes the NAV — and the headlines — swing so hard. (The mechanics of how a fund's LPs, fees, carry, and this liquid-vs-locked split actually work are the subject of [the crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model).)

### Drawdown and mark-to-market volatility

A **drawdown** is the peak-to-trough fall in a fund's value — how far it dropped from its high before recovering. **Mark-to-market volatility** is how much the *reported* value swings day to day because the holdings are repriced to live market prices. A concentrated crypto fund has enormous mark-to-market volatility: its NAV can move 20% in a day because one token did. The single most important fact about drawdowns — the one that makes concentration so dangerous — is that they are *asymmetric*, and we'll prove that with arithmetic shortly.

#### Worked example 1: what a 30% position does to a fund

Let's make concentration concrete. Suppose a fund has **\$1,000 million** (\$1 billion) in AUM. It holds **30%** — \$300 million — in a single token, and the other 70% (\$700 million) in things that happen to stay flat this year. (Round numbers; this is illustrative arithmetic, not a real fund's book.)

**Case A — the token triples (a 3x, i.e. +200%).**

- The \$300 million position becomes \$300 million × 3 = **\$900 million**.
- The other \$700 million is unchanged.
- New NAV = \$700m + \$900m = **\$1,600 million**.
- That's a **+60%** fund year — from one 30% position.

**Case B — the token falls 70%.**

- The \$300 million position becomes \$300 million × 0.30 = **\$90 million**.
- The other \$700 million is unchanged.
- New NAV = \$700m + \$90m = **\$790 million**.
- That's a **−21%** fund year.

![A bar chart of fund NAV under three scenarios for a \$1,000M fund with 30% in one token: base NAV \$1,000M, the token tripling lifts NAV to \$1,600M (+60%), and the token falling 70% drops NAV to \$790M (−21%).](/imgs/blogs/polychain-multicoin-and-the-thesis-funds-4.webp)

The chart shows all three states side by side. Notice how much of the fund's result comes from one position: a single 30% bet swung the fund from +60% to −21% depending on what that one token did. Now suppose the concentration is higher — 40%, 50% — and you can see how a thesis fund's annual return becomes, essentially, *a leveraged bet on one asset*. **The intuition to keep: in a concentrated fund, the fund's return is not an average of many bets; it is mostly the story of one.**

## 1. Polychain: the fund that made concentration respectable

Polychain Capital is where the crypto thesis-fund model went institutional. It was founded in **September 2016** by **Olaf Carlson-Wee**, who had been the *first employee* of Coinbase and who, famously, once put most of his savings into Bitcoin (Forbes, March 2017). What makes Polychain notable is not just its bets but its backers: the firm raised capital from blue-chip venture names including **Andreessen Horowitz (a16z), Sequoia Capital, Union Square Ventures, and Founders Fund** (per its Wikipedia profile and contemporaneous press). When Sequoia backs a crypto fund, it signals to the rest of the market that this asset class is investable — the backing itself becomes a kind of price signal, a theme we return to later.

### The structure: a hedge fund wrapped around venture bets

Polychain runs both books we described. Its flagship is a **liquid token hedge fund** that trades protocol tokens, and alongside it the firm raised a series of **venture funds** that take locked, early-stage positions (by 2023 it had raised multiple venture funds and, per Fortune, a fourth crypto venture fund of about \$200 million in July 2023). The hedge fund is the part with a public track record, and that track record is a perfect illustration of the concentrated model's shape.

### The track record: the shape of concentration

According to an investor breakdown reported by CoinDesk (March 2020), Polychain's hedge fund posted these annual returns:

| Year | Polychain hedge fund return |
|---|---|
| 2016 (partial) | −2.7% |
| 2017 | **+2,278.8%** |
| 2018 | **−60.4%** |
| 2019 | +56.1% |
| Cumulative (inception → Nov 2019) | **+1,332.3%** |

Read those numbers as a single sentence: the fund made more than twenty-two times its money in one year, gave back well over half of it the next, and *still* finished the period up more than thirteen-fold. Its worst six-month stretch (July–December 2018) was a **−47.6%** drawdown, and it carried a minimum six-month lock-up so investors couldn't flee at the bottom. On the AUM side, Polychain reportedly held **more than \$1 billion** in early 2018 and saw that fall to about **\$591.5 million by the end of 2018** — a drop driven by the *market*, not by investors redeeming (its Q4 2018 AUM fell about 40%, per Finance Magnates). The fund later grew again to a reported **\$4 billion (April 2021)** and around **\$5 billion (2022)** as the market recovered, illustrating the round-trips these funds live through.

The lesson of Polychain is that the concentrated model, run by a credentialed manager with blue-chip backers, could survive a −60% year and come back — *if* the manager and the LPs had the stomach (and the lock-ups) to hold. That "if" is doing a lot of work, and it's exactly what the next fund tested to the limit.

#### Worked example 2: why the liquid book and the venture book behave oppositely

Suppose a thesis fund puts **\$10 million** into a young layer-1 ecosystem, split evenly: **\$5 million** buys the token on an exchange (the liquid book) and **\$5 million** goes into a seed round of the same project at a fixed price, with a **3-year vesting** lock-up (the venture book). A year later the token is up **10x**.

- **Liquid book:** \$5 million × 10 = **\$50 million**, marked to market. The fund *can* sell — it can take profits, trim the position, and turn paper gains into cash. If it wanted, it could realize \$50 million today.
- **Venture book:** \$5 million × 10 = **\$50 million** on paper too — but it is *locked*. The fund cannot sell a single token for two more years. The \$50 million is a mark, not money.

Now the cycle turns and the token falls **80%** from that peak.

- **Liquid book:** if the fund didn't sell, its \$50 million becomes **\$10 million**. It *could* have locked in gains and didn't — the flexibility was real but only useful if exercised.
- **Venture book:** the \$50 million mark becomes **\$10 million** too, and the fund still can't sell. It rode the entire round-trip on paper, unable to act, and its LPs watched a 10x mark evaporate before a single token unlocked.

**The intuition: the venture book's paper gains are not spendable, and its paper losses are not avoidable.** The same thesis, expressed in two books, gives the fund flexibility in one and total helplessness in the other — and the more concentrated the fund, the more both books move together, so there's no diversification benefit between them.

## 2. Multicoin: conviction as a business model

If Polychain made concentration respectable, **Multicoin Capital** made it a religion. Founded in **2017** by **Kyle Samani** and **Tushar Jain** and based in Austin, Texas, Multicoin built its reputation on a handful of enormous, non-consensus bets — most famously **Solana**, but also **Helium**, **The Graph**, and **Arweave**. The firm has been described (by outlets covering crypto venture) as possibly the best-performing venture fund of all time, and that reputation rests almost entirely on one thing: it identified a thesis early, bet enormously on it, and refused to flinch.

### The Solana thesis and the entry price

Multicoin's central thesis was about *performance*: it bet that a blockchain optimized for raw speed and low cost — one global, fast state machine — would win developers and users, and that most competing designs were solving the wrong problem. It expressed that thesis on Solana as early and as hard as it could. Multicoin backed Solana from the seed stage and **led Solana's roughly \$20 million Series A in July 2019** (per The Block and contemporaneous reports). The entry price is the part that sounds made up but isn't: Solana's **seed sale, on 5 April 2018, priced SOL at about \$0.04** (per token-sale trackers such as ICO Drops). When SOL later ran to a peak near **\$260 in November 2021**, an entry near \$0.04 was up on the order of *thousands of times* on paper.

![A timeline of Solana's price journey: April 2018 seed sale near \$0.04 per SOL, March 2020 CoinList auction at \$0.22, November 2021 all-time high near \$260, December 2022 FTX-crash low near \$8, and a 2023-24 recovery back to three figures.](/imgs/blogs/polychain-multicoin-and-the-thesis-funds-5.webp)

The timeline shows why this one bet defines the fund. Note the milestones the market itself provides as reference points: the public **CoinList auction in March 2020 sold SOL at \$0.22**, so even the ordinary public entry was a small fraction of the eventual peak. And note the right half of the chart — the same asset that ran to \$260 fell to roughly **\$8 by December 2022** as the FTX collapse (more on that shortly) erased almost its entire value, before recovering into three figures over 2023–24. A single bet delivered both the best trade of a generation and a near-total round-trip, within one cycle.

#### Worked example 3: an early-L1 entry, from seed price to peak to round-trip

Let's walk the paper-versus-realized distinction with Solana's real, sourced price points, using a hypothetical position size to keep the arithmetic clean. Suppose a fund put **\$1 million** into SOL at the **\$0.04** seed price. (The dollar figure is illustrative; the SOL prices are sourced and dated.)

- **Tokens acquired:** \$1,000,000 ÷ \$0.04 = **25,000,000 SOL**.
- **At the November 2021 peak (~\$260):** 25,000,000 × \$260 = **\$6.5 billion** on paper. That is a **6,500x** paper return.
- **But most of it is locked and unsellable at the top.** Early-round tokens vest over years; a fund holding 25 million SOL cannot dump them at the peak without collapsing the price (and much of the stake may still be under lock-up). The \$6.5 billion is a mark, not a realization.
- **At the December 2022 low (~\$8):** 25,000,000 × \$8 = **\$200 million**. Still a **200x** return on the \$1 million cost — extraordinary — but a **97% fall from the paper peak**.

**The intuition: the headline "6,500x" and the money actually banked are different universes.** Concentration plus lock-ups means the paper peak is largely theoretical; what a fund keeps depends entirely on how much it could sell, and when, on the way up — and a concentrated fund that is *the* backer of a token often cannot sell much without cratering the very price its NAV depends on. That is the reflexive trap we build up to in section 4.

### The returns, and the fall

Multicoin's reported hedge-fund performance is the most vivid illustration in this whole series of what concentration does. The numbers, each sourced and dated:

- The fund told investors in **late 2021** that it was up **20,287% since its October 2017 inception** (as reported by Axios, December 2021). That is not a typo: roughly a 200-fold gain, gross, driven by Solana and a few other bets.
- Then, in **2022**, the same hedge fund **lost 91.4%** (per an investor letter reported by CoinDesk, March 2023). Its assets under management, per one tally, fell from about **\$8.9 billion (2021) to \$1.4 billion (2022)** — a roughly **85%** collapse (Newcomer, 2023).
- It was not the end. The fund reportedly gained **100.9% in January 2023** alone, and finished **2023 up about 537%** as crypto rebounded (per CoinDesk and crypto press). By a 2024 investor note, the hedge fund was reported up **9,281% since inception** (The Block) — still a huge number, but a fraction of the 2021 peak.

Sit with that sequence: **+20,287%, then −91.4%, then +537%.** A fund's headline return swung by orders of magnitude in consecutive years. That is not mismanagement; it is *the concentrated model working as designed*, in both directions. The next section explains why the −91.4% is so much more dangerous than it looks.

## 3. The volatility a concentrated model produces

The reason deep drawdowns are so dangerous is arithmetic, and it is worth burning into memory because it governs the entire risk of concentration. When you lose a percentage of your money, the *gain* required to get back to even is not the same percentage — it is larger, and it grows explosively as losses deepen.

The formula for the break-even gain $g$ after a drawdown of fraction $d$ is:

$$g = \frac{1}{1-d} - 1$$

where $d$ is the loss as a decimal (a 50% loss is $d = 0.5$) and $g$ is the gain needed to recover, also as a decimal. The intuition: if you have \$100 and lose 50%, you have \$50, and \$50 must *double* (+100%) to get back to \$100. Losing half means you must then make double.

#### Worked example 4: the recovery math behind a −91% year

Let's tabulate the break-even gain for a range of drawdowns, then apply it to the real funds:

| Drawdown | Money left (from \$100) | Gain needed to break even |
|---|---|---|
| −20% | \$80 | +25% (1.25x) |
| −50% | \$50 | +100% (2x) |
| −70% | \$30 | +233% (3.3x) |
| −91% | \$9 | **+1,011% (11.1x)** |

![A bar chart showing the break-even gain required to recover from each drawdown level: a −20% drawdown needs +25%, a −50% needs +100%, a −70% needs +233%, and a −91% needs +1,011%, with the final bar towering over the others to show the convex escalation.](/imgs/blogs/polychain-multicoin-and-the-thesis-funds-6.webp)

The chart shows the convex explosion. The first three bars rise gently; the −91% bar towers over them, because recovering from a 91% loss requires making **eleven times your money**. Apply this to the real cases:

- **Multicoin's −91.4% (2022)** left roughly 8.6 cents on the dollar and required about **+1,063%** just to reach its old high. It got a chunk of the way there with +100.9% in January 2023 and +537% for the full year, but "up 537%" after "down 91.4%" still leaves an investor who entered at the 2021 peak deep underwater — because 0.086 × 6.37 ≈ 0.55, i.e. still about 45% below the peak.
- **Polychain's −60.4% (2018)** needed about **+153%** to recover, which its subsequent years delivered.

**The intuition: concentration doesn't just make big losses possible — it makes them near-permanent, because the recovery math is stacked against you.** A diversified fund rarely draws down 90%; a concentrated one can, and once it does, only another extraordinary bull market can undo it. This is why lock-ups matter so much to these funds: without them, LPs redeem at the −91% bottom, forcing the fund to sell into weakness and turning a recoverable drawdown into a permanent loss. When a downturn hits and money is trapped in a mark that only shrinks, funds sometimes create a **side pocket** — a walled-off portion of the fund holding the impaired or illiquid assets — precisely so the mess in one bet doesn't force the liquidation of everything else. Multicoin created exactly such a side pocket for its FTX-affected assets in November 2022 (per Blockworks and Fortune reporting).

## 4. The reflexive risk: the fund and the token that move each other

Now we arrive at the deepest idea in this post, and the reason a concentrated thesis fund is not just *exposed* to a token's price but *entangled* with it. A fund that is publicly, famously the biggest backer of a token holds a position so large relative to the token's tradable supply that its own actions move the price — which means the fund is on both sides of a feedback loop. This is a specific example of **reflexivity** — a market that watches and reacts to itself, so that prices and fundamentals influence each other rather than one simply reflecting the other. (We have a whole post on [reflexivity: markets that watch themselves](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves) if you want the general theory.)

To see why this is dangerous, we need one more term. A token's **float** is the portion of its total supply that is actually available to trade right now — not locked in vesting, not sitting in a foundation treasury, but liquid on exchanges. Many young tokens have a *thin* float: only a small slice of supply trades, so it doesn't take much buying or selling to move the price a lot. **Slippage** is the price impact of your own order: when you try to sell more than the order book can absorb at the current price, you push the price down as you sell, and get filled at progressively worse prices. A large holder selling into a thin float suffers heavy slippage — and, crucially, *drives down the price everyone else sees, including the price used to mark the seller's own remaining position.*

![A serpentine loop: a shock or redemption call forces the fund to sell its big token, the thin float absorbs the selling poorly, the price drops sharply, the fund's NAV falls because the token is 30%+ of it, which forces still more selling — the loop feeds back on itself.](/imgs/blogs/polychain-multicoin-and-the-thesis-funds-7.webp)

The figure traces the doom loop. Start at the top-left: a market shock or an LP redemption request forces the fund to raise cash, so it must sell its big token. The token's thin float absorbs the selling poorly, so the price drops sharply. Because that token is 30%+ of the fund, the fund's NAV falls — which triggers *more* redemptions and, if the fund used any leverage, margin calls, which force *more* selling. Each turn of the loop makes the next turn worse. A fund whose conviction *lifted* a token on the way up finds that same entanglement *amplifying* the collapse on the way down.

#### Worked example 5: the reflexive cost of being the token's biggest holder

Suppose a fund holds **\$400 million** of a token that makes up **40%** of its \$1 billion NAV, and that this position is **20% of the token's entire tradable float** (a plausibly large stake for a token's lead backer). A shock forces the fund to raise \$200 million of cash quickly, so it needs to sell half its position.

- **Selling 10% of a token's float into the market is enormous.** Order books for young tokens are thin; dumping that much supply might push the price down **30–50%** as it fills (heavy slippage — an illustrative range, since exact impact depends on the book).
- Say the sale realizes an average price **35% below** the pre-sale mark. The fund wanted \$200 million; the market impact means it destroys value getting there, and the *remaining* \$200 million position is now marked at the new, lower price too.
- **The remaining position re-marks down.** If the price is now 35% lower, the fund's *unsold* \$200 million stake is worth about **\$130 million**. So raising \$200 million of cash cost the fund not just slippage on what it sold, but a **\$70 million paper hit on what it kept** — and NAV falls further, potentially triggering the next redemption.

**The intuition: when you are the token, you cannot exit the token.** A position large enough to move the price on the way up is large enough to move it against you on the way down, and the fund's own selling becomes the thing that craters its NAV. This is the reflexive core of the concentrated model — the same entanglement that reads as genius in a bull market reads as a liquidation spiral in a bear. It is also why these funds and the tokens they back are so tightly linked in the public mind: the market knows the fund is the price, and the fund knows it too. For the crowd-behavior half of this dynamic — everyone piling into the same "smart-money" bet and then rushing the same exit — see [information cascades and herding](/blog/trading/game-theory/information-cascades-and-herding-when-rational-traders-follow-the-crowd).

## How it shows up in the price you trade

Step back from the two firms and look at the mechanism as a retail trader would experience it. A concentrated thesis fund shows up in the price you see in five recognizable ways:

1. **The narrative premium.** When a famous fund is publicly the biggest backer of a token, that association is itself a reason people buy — "Multicoin is behind it" becomes a bullish talking point. Part of the token's price is a premium on the fund's conviction, not on the protocol's usage. That premium can evaporate faster than fundamentals, because it was never about fundamentals.
2. **Thin-float amplification.** Because these funds and other insiders hold so much locked supply, the tradable float is small, so relatively modest buying (including the fund's own) produces outsized price moves. The chart looks more dramatic than the actual demand, in both directions.
3. **The reflexive round-trip.** The same entanglement that lifts a token when the fund is buying and marketing it drags the token down when the fund is forced to sell — so tokens tied to one big backer tend to have violent, correlated round-trips rather than steady grinds.
4. **The unlock overhang.** The fund's venture book eventually vests. When those locked tokens unlock, the fund *can* finally sell, which adds supply to the market on a schedule that is often public — a persistent source of downward pressure that has nothing to do with the protocol's health.
5. **The correlation trap.** A fund concentrated on one ecosystem holds the L1 token, the DeFi tokens on that chain, and equity in startups building on it — all bets on the same thesis. When the thesis wobbles, *everything* the fund touches falls together, and because the fund is a large holder across all of them, its distress transmits across the whole ecosystem at once.

None of this requires anyone to do anything improper. It is the natural physics of a large, concentrated, publicly-known position in a thinly-traded asset. Once you can see it, a token that is "a fund's flagship bet" stops looking like a safe endorsement and starts looking like what it is: a highly reflexive instrument whose price is partly a bet on one fund's ability to keep the faith and keep the float thin.

## Why sophisticated investors sign up for the volatility

A fair question at this point: who willingly gives money to a manager who can lose 91% in a year? The answer reveals something important about how professional allocation works, and it is not "gamblers." The LPs in these funds are often endowments, family offices, funds-of-funds, and wealthy individuals — people who allocate across dozens of strategies at once. For an investor holding a large, diversified portfolio, a concentrated crypto fund is a *small slice* deliberately chosen to be high-variance.

The logic is the **barbell**. An allocator might put the vast majority of a portfolio in safe, boring assets (government bonds, index funds) and a small sliver — say 1% to 3% — in things that can either go to zero or return 50x. The safe majority guarantees survival; the tiny risky sliver provides the *convexity*, the chance at an outsized payoff that moves the whole portfolio if it hits. A fund that returns 200x on a 2% allocation adds four times the entire portfolio's starting value; a fund that goes to zero on that same 2% costs almost nothing by comparison. When your position in the fund is small enough, its −91% year is survivable and its +20,000% run is life-changing. The concentrated manager's job is not to be safe; it is to be *the convex sliver* in someone else's diversified book.

Three features that look like bugs to a retail observer are, to these LPs, the whole point:

- **The lock-up is a feature, not a restriction.** A minimum lock-up (Polychain's six months, or multi-year commitments in the venture funds) stops LPs from panic-redeeming at the bottom — which is exactly what would force the fund to sell into weakness and turn a recoverable drawdown into a permanent one. Sophisticated LPs *want* to be protected from their own worst instincts, and from other LPs' redemptions dragging the fund down.
- **The concentration is the product.** An LP who wanted diversification would buy an index. They are paying a concentrated manager precisely for the differentiated, non-consensus bet they cannot get elsewhere. A thesis fund that quietly diversified to reduce volatility would be failing to deliver what it was hired to do.
- **The manager eats their own cooking.** Because the GP earns most of their money from carry (a share of the profits), their incentive is aligned with the upside — they get rich when the fund does, and the "2 and 20" management fee merely keeps the lights on through the drawdowns. The flip side, as we noted, is that this same incentive rewards public conviction that lifts the fund's own marks.

This is why judging these funds by a retail lens — "how could anyone hold through −91%?" — misses the structure. The people holding through it sized their position so they *could*, and they are being paid, in expectation, for their willingness to endure the variance that scares everyone else away. That said, the arrangement only works if the LP truly sized it as a small sliver; an investor who bet too much of their wealth on one concentrated fund experiences the −91% the way the fund does, with none of the barbell's protection.

## Common misconceptions

**"A fund up 20,000% is a genius; a fund down 91% is an idiot."** They are usually the *same fund in consecutive years*. Multicoin was both. Judging a concentrated manager by a single year's return tells you almost nothing, because the strategy is *built* to produce extreme years in both directions. The right question is whether the process — the thesis, the sizing, the ability to hold — is sound over a full cycle, not what the last print was.

**"Concentration is just recklessness."** It is a deliberate, coherent strategy with a real edge: if you have genuine, differentiated conviction, concentrating on it is the only way to earn an outsized return, because a 2% position in a 100x winner barely moves a diversified fund. The recklessness isn't concentration itself; it's concentration *without* the conviction, the research, the lock-ups, and the risk controls (like side pockets) that let a fund survive being wrong for a while. Concentration is a bet on your own edge — dangerous precisely because most people overestimate their edge.

**"These funds got rich by trading skillfully."** Mostly they got rich by being *early and patient* on a venture bet, not by nimble trading. Multicoin's fortune was made buying SOL near \$0.04 in 2018 and holding, not by timing trades. The hedge-fund wrapper makes it look like trading; the returns came from venture-style conviction expressed years before the token was liquid. This matters because it means the model isn't replicable by watching charts — the edge was in the private-round access and the willingness to hold, neither of which a retail trader has.

**"If a top fund is buying it, it's safe for me to buy it."** This is exactly backwards. A fund that entered a token's seed round near \$0.04 is a happy holder at \$50; a retail buyer entering at \$50 needs the price to keep rising just to break even. You are not on the same side of the trade as the fund even when you hold the same token, because your cost basis and your lock-up are completely different. (This is the core argument of [cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto).)

**"The AUM number tells you how much money they raised."** For these funds, AUM is mostly *marks*, not deposits. Multicoin's AUM went from \$8.9 billion to \$1.4 billion in 2022 without investors necessarily pulling \$7.5 billion out — the holdings simply repriced. AUM in a concentrated crypto fund is a live market quote on a portfolio, not a measure of capital raised, and it can round-trip as violently as any token.

**"A fund's public conviction is just honest belief."** It may be sincere *and* self-serving at once. A fund that is the largest holder of a token benefits directly when more people believe its thesis and buy — the louder conviction pulls in capital and lifts the mark on the fund's own position. That doesn't make the thesis wrong, but it means you should weigh a fund's public cheerleading for a token it holds the way you'd weigh any stakeholder talking their own book.

## How it shows up in real markets

![A grid summarizing Multicoin's reported roller-coaster: 2021 peak AUM about \$8.9B with a hedge fund up 20,287% since 2017; 2022 AUM about \$1.4B (down 85%) with the fund down 91.4% amid the FTX and Solana collapse; 2023 recovering with a +537% rebound; 2024 AUM about \$600M with AUM up about 50% year over year.](/imgs/blogs/polychain-multicoin-and-the-thesis-funds-8.webp)

The grid above is the whole thesis in one table (the asterisk on "+20,287% since '17" marks it as the figure reported to investors in late 2021, gross, since the October 2017 inception). Below, four episodes show the mechanics of this post in action, with sourced figures.

### 1. Multicoin's 2021 → 2022 round-trip

Multicoin is the textbook case of the concentrated model's full arc. On the way up, its Solana-heavy book drove a reported **+20,287% since inception** by late 2021 (Axios) and an AUM near **\$8.9 billion**. On the way down, the fund had roughly **10% of its assets stuck on FTX** when the exchange collapsed in November 2022, plus heavy exposure to **FTT, SOL, and SRM** — all tokens that cratered together (per Fortune and CNBC reporting). In about two weeks, Multicoin reportedly lost around **55% of its fund's capital**, finished 2022 down **91.4%** (CoinDesk), and created a **side pocket** to wall off the FTX-affected assets. The mechanism is every idea in this post at once: concentration, correlated exposure, a reflexive collapse, and a drawdown so deep the recovery math became a mountain to climb.

### 2. Polychain's 2018 drawdown and recovery

Polychain's flagship hedge fund lived the same shape one cycle earlier. After a **+2,278.8%** year in 2017, it fell **−60.4%** in 2018, with a worst six-month drawdown of **−47.6%** (July–December 2018), and its AUM dropped from over **\$1 billion** to about **\$591.5 million** — driven by the market, not redemptions, because the six-month lock-up kept investors from fleeing (CoinDesk, Finance Magnates). The fund survived and later grew to a reported **\$4–5 billion** as crypto recovered. Polychain is the proof that the concentrated model *can* survive a −60% year — but only with the right investor base and lock-ups, and only if the next bull market shows up.

### 3. Solana's round-trip as a proxy for the model

Solana's own chart is the clearest external evidence of the reflexive round-trip. A seed price near **\$0.04 (April 2018)**, a public auction at **\$0.22 (March 2020)**, a peak near **\$260 (November 2021)**, a low near **\$8 (December 2022)**, and a recovery into three figures over 2023–24 (per token-sale trackers and market data). No protocol's *usage* swings by 97% and back; that violence is the signature of a thinly-floated asset tied to a handful of large, concentrated backers whose conviction — and forced selling — moved the price as much as any fundamental did.

### 4. The backing-as-signal effect

When Polychain raised from **a16z, Sequoia, Union Square Ventures, and Founders Fund**, the news itself moved sentiment: blue-chip venture capital validating a crypto fund told the market the asset class was investable. This is the up-cycle version of reflexivity — a fund's *pedigree* becomes a price signal, drawing capital that lifts the very assets the fund holds. The same mechanism runs in reverse when a marquee backer is in distress, which is why the market watches these funds' fortunes so closely.

### 5. The people behind the conviction move on

A final, human data point on how concentrated these funds are around a person and a thesis: in **February 2026**, Kyle Samani stepped back from his managing-director role at Multicoin to pursue other areas of tech, with day-to-day operations continuing under Tushar Jain and Brian Smith while Samani stayed involved in the Solana ecosystem (per CoinDesk and The Block, February 2026). When a fund's identity is fused to one founder's conviction and one chain's thesis, the founder's attention is itself a fundamental — another way the "fund" and the "bet" are harder to separate than they look.

## When this matters to you

If you never invest a dollar in crypto, this still matters, because it changes how you *read* a token's price. When you see a chart that ran 100x and then round-tripped, you now know to ask: is this a thinly-floated token whose price is entangled with one big backer's conviction and forced selling? If so, the chart is telling you about a fund's position and psychology as much as about the protocol.

If you do trade, here is the retail-defense takeaway, and it is not advice to buy or sell anything — it is a way to see the table you're sitting at:

- **Find out who the big backers are and what they paid.** A fund that entered near \$0.04 is a happy seller at almost any price you can buy at. Its cost basis is public in a way its intentions are not; assume it can profitably sell where you're buying.
- **Check the float, not just the price.** A low tradable float means the price is easy to move and easy to crater. A token that is 40% of one fund's NAV and 20% of its own float is a reflexive instrument, not a stable store of value. (Our post on [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) walks the float-and-slippage mechanics.)
- **Read the unlock calendar.** The venture book vests eventually. Known unlock dates are known future selling; treat them as scheduled headwinds.
- **Distrust the narrative premium.** "A top fund backs it" is a reason the price is *high*, not a reason it will *stay* high. The premium is on conviction, and conviction is the first thing to leave in a downturn.
- **Respect the drawdown math on your own book.** If concentration can take a professional fund down 91%, it can do the same to you — and you don't have a lock-up protecting you from panic-selling the bottom. The recovery from a −91% loss is +1,011%; size your bets so no single one can put you in that hole.

Concentrated thesis funds are neither villains nor geniuses; they are a specific machine with a specific physics. They can produce the best returns in the market and the worst drawdowns, often in the same two years, and their entanglement with the tokens they back means their conviction is written into the price you trade. Seeing that clearly is worth more than any single trade.

*This post is educational, not financial advice. It explains how a category of fund works and how its behavior shows up in prices; it is not a recommendation to buy or sell any token or to invest in any fund.*

## Sources & further reading

Primary and reputable-press sources behind the headline figures (with as-of dates):

- **Polychain hedge-fund returns and drawdown** (2016 −2.7%, 2017 +2,278.8%, 2018 −60.4%, 2019 +56.1%, cumulative +1,332.3% to Nov 2019, −47.6% worst six months, six-month lock-up): *"Investors in Polychain Capital's Crypto Hedge Fund Saw 1,332% Gains,"* CoinDesk, 30 March 2020.
- **Polychain AUM (>\$1B early 2018 → ~\$591.5M end 2018; −40% Q4 2018)**: Finance Magnates and The Block, 2019; *Polychain Capital*, Wikipedia (accessed 2026).
- **Polychain later AUM (~\$4B April 2021; ~\$5B 2022; ~\$200M Fund IV July 2023)**: *Polychain Capital*, Wikipedia; *"Polychain raises \$200 million for fourth crypto VC fund,"* Fortune, 18 July 2023.
- **Polychain founding, Olaf Carlson-Wee (first Coinbase employee), backers (a16z, Sequoia, USV, Founders Fund)**: *Olaf Carlson-Wee* and *Polychain Capital*, Wikipedia; Forbes, 7 March 2017.
- **Multicoin founding (2017, Samani & Jain), Solana Series A (~\$20M, July 2019)**: *Kyle Samani* profile, The Block; contemporaneous reporting.
- **Multicoin hedge-fund returns (+20,287% since inception, reported late 2021)**: *"Multicoin Capital posts mega returns,"* Axios, 29 December 2021.
- **Multicoin −91.4% in 2022; ~55% of capital lost in two weeks; ~10% of assets on FTX; FTT/SOL/SRM exposure; side pocket**: *"Multicoin Capital's Hedge Fund Lost 91.4% Last Year,"* CoinDesk, 4 March 2023; *"Crypto VC Multicoin Capital has assets frozen with FTX bankruptcy pending,"* Fortune, 15 November 2022; Blockworks and CNBC, November 2022.
- **Multicoin AUM (~\$8.9B 2021 → ~\$1.4B 2022, −85%; ~\$600M 2024, +50%+ YoY)**: Newcomer, 2023; *"Crypto venture firms… Multicoin posts 56% annual growth,"* Fortune, 7 April 2025.
- **Multicoin 2023 rebound (+100.9% January 2023; +537% for 2023; +9,281% since inception by 2024)**: CoinDesk, March 2023; The Block, 2024.
- **Solana prices (seed ~\$0.04, 5 April 2018; CoinList auction \$0.22, March 2020; ATH ~\$260, November 2021; low ~\$8, December 2022)**: ICO Drops and CoinList (token-sale data); CoinMarketCap historical data.
- **Kyle Samani steps back from managing-director role (February 2026)**: CoinDesk and The Block, February 2026.

Related posts on this blog:

- [Crypto VC and market makers: who really moves prices](/blog/trading/crypto/crypto-vc-and-market-makers) — the series hub.
- [The crypto VC operating model](/blog/trading/crypto-players/the-crypto-vc-operating-model) — how a fund's LPs, fees, carry, and liquid-vs-locked book actually work.
- [Paradigm and the research-driven fund](/blog/trading/crypto-players/paradigm-and-the-research-driven-fund) — a different flavor of crypto fund, where research and capital are one weapon.
- [Cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) — why the fund and the launch-day retail buyer are on opposite sides.
- [Reflexivity: markets that watch themselves](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves) — the general theory behind the doom loop.
- [Three Arrows Capital and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion) and [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) — what happens when concentration and leverage meet a shock.
