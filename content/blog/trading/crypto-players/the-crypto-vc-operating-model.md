---
title: "The Crypto VC Operating Model: How Token Funds Really Make Money"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "A plain-English, worked-arithmetic tour of how a crypto venture fund is built and paid — LPs and GPs, fees and carry, the liquid-versus-locked token book, and the crypto twist that lets a fund exit on a public market years before an equity investor could."
tags: ["crypto", "venture-capital", "token-fund", "carry", "management-fee", "tokenomics", "saft", "vesting", "fund-economics", "crypto-players", "retail-defense"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 38
---

> [!important]
> **TL;DR** — A crypto venture fund is an ordinary money-management business — pooled capital, a 2%-and-20% fee model, a ten-year clock — bolted onto one extraordinary advantage: it can sell its winners on a public market years before an equity fund could.
>
> - A fund is money from **limited partners (LPs)** run by a **general partner (GP)**. The GP earns a **management fee** (~2% a year of committed capital) to operate, and **carried interest** ("carry", ~20% of the profits) as its real payday — but only after LPs get their capital back.
> - The crypto twist is the **exit**. An equity VC waits 7–10 years for an IPO or acquisition; a token fund can often start selling a portfolio token on a public exchange **1–2 years** after it invests, braked only by a vesting schedule the project set for itself.
> - That advantage hides a trap for the GP too: most of a young fund's headline value is **locked, marked-to-model paper**, not cash. The gap between the paper multiple (**TVPI**) and the cash actually returned (**DPI**) is where crypto fund reporting can flatter reality.
> - Real anchors, sourced below: **a16z crypto** raised a **$4.5B** fund in May 2022 (its largest crypto fund reportedly fell ~40% in value in the first half of that year); **Paradigm** raised **$2.5B** in 2021; **Pantera** launched the first US institutional Bitcoin fund in 2013 with just **$13M**. The specific fund sizes here are real and dated; the return arithmetic uses round hypothetical numbers.
> - The one habit that protects you: when a fund-backed token unlocks, someone has to buy what the fund sells. **Read the vesting calendar** — a "known" unlock is a scheduled transfer of supply from insiders to whoever is bidding.

Here is a number that should stop you: a fund can buy a token for **two cents**, and roughly a year later that same token can be trading on a public exchange for **two dollars** — a hundred times its cost — while an equity investor who wrote a check into a startup the same week will wait the better part of a *decade* before they can sell a single share. Same money-management business on paper. Wildly different clock in practice. That difference is not a detail. It is the whole reason a crypto venture fund is one of the most powerful — and least understood — players in the market.

Most people picture a crypto VC as a rich firm that "invests in coins." That is not wrong, but it misses the machine. A fund is a *business* with its own customers (the people whose money it manages), its own product (returns), its own paycheck (fees and a share of the profit), and its own life cycle (a clock that starts the day it opens and forces it to close about ten years later). Understanding that machine — who pays whom, in what order, and when the cash actually shows up — tells you more about why token prices move the way they do than any chart pattern ever will.

The diagram below is the mental model for this whole piece: money flows *in* from limited partners, a manager deploys it into a book of tokens and equity, and — this is the crypto-specific part — it can reach a *public* exit door years before the equity door ever opens. Everything that follows is a tour of this picture, built from zero, with the arithmetic shown at every step.

![The crypto VC machine: LPs and a GP fund a manager who deploys into tokens and equity, then exits tokens on a public market years before equity](/imgs/blogs/the-crypto-vc-operating-model-1.webp)

This post is a companion to the broader map in [crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers), which introduces the whole cast of funds and trading firms. Here we zoom all the way into one of them — the venture fund — and answer the most basic question about it: *how does it actually make money, and how does the way it makes money end up in the price you pay?* We will use real funds (a16z, Paradigm, Polychain, Pantera) as illustrations, with every specific figure sourced and dated. The step-by-step dollar walkthroughs use round, made-up numbers on purpose, so the mechanics stay clean.

## Foundations: the building blocks

Before we can talk about how a fund gets paid, we need a shared vocabulary. Read this section even if a few terms feel familiar, because the entire argument later hinges on the *precise* meaning of "committed capital", "carry", and "mark". A pro can skim; a beginner should not skip.

### Who's who: LPs, the GP, and the fund

A venture **fund** is not a company you can buy shares of. It is a pool of money with a legal wrapper (usually a *limited partnership*) and a job: invest that money, grow it, and hand it back bigger. Two kinds of people stand around that pool.

- The **limited partners (LPs)** are the people and institutions who *provide* the money. Think pension funds, university endowments, family offices (the private investment arms of wealthy families), sovereign wealth funds, funds-of-funds (funds that invest in other funds), and some wealthy individuals. "Limited" refers to *limited liability* — an LP can lose the money it put in, but no more, and it has no say in the day-to-day investing. LPs are the fund's customers.
- The **general partner (GP)** is the firm that *runs* the pool — the investment team that sources deals, decides what to buy, negotiates terms, manages the positions, and eventually sells. When you hear "a16z crypto" or "Paradigm", you are hearing the GP. The GP makes every investment decision and, in exchange, takes a slice of the economics (the fees and carry we will get to). "General" partner because, legally, it carries the general management responsibility.

So the sentence "Paradigm raised a $2.5 billion fund" means: the GP (Paradigm) persuaded a set of LPs to *commit* $2.5 billion of their money to a pool that Paradigm will invest on their behalf. In November 2021, Paradigm did exactly that — $2.5 billion, at the time the largest crypto-focused venture fund ever raised ([Blockworks](https://blockworks.com/news/paradigm-launches-2-5-billion-crypto-fund), 2021; [Paradigm on Wikipedia](https://en.wikipedia.org/wiki/Paradigm_(venture_capital_firm))).

### Committed vs. deployed capital, and the "vintage"

When an LP agrees to put money in, it does not wire the whole sum on day one. It signs up for a **commitment** — a promise to provide up to a certain amount when the GP asks. The GP then **calls** the capital in pieces (a "capital call") as it finds deals to fund. Two terms fall out of this:

- **Committed capital** is the total the LPs have promised. A "$500M fund" means $500M committed.
- **Deployed (or invested) capital** is how much has actually been called and put into investments so far.

Early in a fund's life, committed is high and deployed is low — the GP is still hunting for deals. A fund's **vintage** is simply the year it started investing (its "birth year"); funds are compared to other funds of the same vintage because they lived through the same market. A "2022-vintage" fund and a "2021-vintage" fund faced very different worlds even though they are otherwise similar.

### The fund's clock: a ten-year life

A venture fund is not forever. The standard structure gives it a **term of about ten years**, split into two phases ([Carta](https://carta.com/learn/private-funds/management/fund-performance/)):

- An **investment period** (roughly the first 3–5 years) when the GP is actively making *new* investments.
- A **harvest period** (the back half) when the GP stops buying, nurtures what it owns, and works to *exit* — turn positions into cash and return it to LPs. The fund then winds down and closes.

This clock matters enormously, and it is where crypto breaks the mold, as we will see. Hold the idea: the GP is on a timer, and it must eventually turn its holdings into cash it can hand back.

### The two ways a GP gets paid: management fee and carry

The GP earns money two ways, and telling them apart is the single most important thing in this whole business.

- The **management fee** is an annual operating stipend. The market standard is around **2% per year of committed capital** during the investment period, typically stepping down to a smaller percentage (often of *invested* capital or net asset value) once the investment period ends ([VC Beast](https://vcbeast.com/2-and-20-fee-structure-explained)). On a $500M fund, 2% is $10M a year. This pays salaries, rent, travel, legal bills, and research. It is *not* the GP's profit; it is the fuel that keeps the lights on. Crucially, the GP collects it **whether or not the investments work.**
- **Carried interest** — "carry" — is the GP's share of the *profits*. The standard is **20% of the fund's gains** ([VC Beast](https://vcbeast.com/venture-capital-glossary/2-and-20)). This is where a GP actually gets rich. But — and this is the load-bearing rule — the GP only earns carry *after* the LPs have their original capital back. First the LPs are made whole; only then does the GP take its cut of what is left.

Put "2% and 20%" together and you get the famous shorthand **"2 and 20."** Not every fund uses exactly these numbers — a 2023 analysis found real terms vary ([TechCrunch](https://techcrunch.com/2023/09/27/venture-fund-2-and-20/)) — but 2-and-20 is the reference point the whole industry is measured against.

Two more terms complete the fee picture:

- A **hurdle rate** (or "preferred return") is a minimum annual return the LPs must receive *before* the GP earns any carry — often set around **8% a year** in private equity. Interestingly, early-stage venture funds frequently *omit* the hurdle, on the logic that VC returns are binary and long-dated, so an annual floor is less meaningful ([VC Beast](https://vcbeast.com/venture-capital-glossary/2-and-20)). Many crypto funds follow that VC convention and skip it.
- The **GP catch-up** is a clause that, once the hurdle is cleared, lets the GP "catch up" by taking a larger share of the next dollars until it has reached its full 20% of total profit. It only exists when there is a hurdle.

There is also a quieter third piece of the economics: the **GP commitment**. To prove it believes in its own bets, the GP typically invests *its own* money into the fund alongside the LPs — commonly on the order of 1–5% of the fund's size. On a $500M fund, a 2% GP commitment is $10M of the partners' personal capital at risk. It is the "skin in the game" that reassures LPs the GP eats its own cooking: if the fund loses money, the GP loses real principal, not just its future carry. When you evaluate a fund, the GP commitment is one of the most honest signals of alignment — a manager who barely invests in its own fund is telling you something.

#### Worked example: the simplest possible fee

Start tiny. You (an LP) commit **$1,000** to a fund charging 2 and 20.

```
Management fee, year 1  = 2% × $1,000        = $20
```

That $20 comes out to run the fund. Now suppose, years later, your $1,000 has grown to **$1,600** and is returned to you.

```
Capital returned to you (LP)  = $1,000   (your money back first)
Profit                        = $600
GP carry (20% of profit)      = 20% × $600  = $120  -> to the GP
LP profit share (80%)         = 80% × $600  = $480  -> to you
Your total back               = $1,000 + $480       = $1,480
```

You turned $1,000 into $1,480 net; the GP earned $120 of carry plus the small management fees it collected along the way. **The intuition: the GP is paid a steady wage to operate (the fee) and a big bonus only if it makes you money (the carry) — and you always get your principal back before the GP shares in the gains.** Every number later in this post is just this same waterfall scaled up.

### Mark-to-market, and "liquid" vs "illiquid"

An investment is **liquid** if you can sell it quickly at a knowable price (a token trading on a big exchange). It is **illiquid** if you cannot (a stake in a private startup, or tokens still locked by a vesting schedule).

For anything it cannot sell yet, the fund still has to *state a value* in its books — that is **marking**, and the value it writes down is the **mark**. **Mark-to-market** means valuing a position at the current live market price. When there is no live price (a private position), the fund marks to a *model* or to the price of the last funding round instead. Keep this distinction close: a mark is an *estimate of what something is worth*, not *cash in hand*. The whole "paper vs cash" problem later lives in that gap.

### How a crypto fund even buys a token: SAFT and the token warrant

An equity VC buys **shares**. A crypto fund often cannot, because at the moment it invests, the token frequently *does not exist yet* — the project is still being built. So the industry invented instruments to sell a claim on *future* tokens:

- A **SAFT** — *Simple Agreement for Future Tokens* — is a contract in which an investor pays now for the right to receive tokens later, once the network launches. It was pioneered by Protocol Labs (with AngelList and the law firm Cooley) and first used for the **Filecoin** sale, which completed on 2017-09-07 and raised over **$205 million** ([Protocol Labs](https://www.protocol.ai/blog/filecoin-sale-completed/); [Protocol Labs, "Announcing the SAFT Project"](https://www.protocol.ai/blog/announcing-saft-project/)). A SAFT is treated as a security in the US and sold privately to accredited (wealthy/institutional) investors.
- A **token warrant** (or **token side letter**) is the modern default, usually bolted onto a **SAFE** (*Simple Agreement for Future Equity*, the standard startup investment contract). The fund buys equity in the company *and* gets a warrant — an option — to receive a slice of the project's tokens if and when it launches ([Pulley](https://pulley.com/guides/token-warrant); [DAOSPV](https://blog.daospv.com/token-warrants-safes-and-safts-a-founders-guide-to-web3-fundraising-instruments/)). This hedges the fund against two risks at once: the company pivoting its business, and the token never shipping.

The point for us: whatever the paperwork, the fund ends up holding a *right to tokens at a very low price*, tokens that will (if all goes well) one day trade on a public market. That right is the seed of everything.

### The scorecards: DPI, TVPI, IRR

Finally, three ratios that LPs use to grade a fund. You will meet them again in the "paper vs cash" section, so just get the shapes now.

- **DPI — Distributed to Paid-In.** Cash actually *returned* to LPs, divided by cash they *paid in*. DPI of 1.0x means "you have your money back"; 1.5x means "you have 1.5× your money back, in real dollars." DPI is the honest one — it is cash that has left the fund and hit LP bank accounts ([Carta](https://carta.com/learn/private-funds/management/fund-performance/)).
- **TVPI — Total Value to Paid-In.** *Everything* the fund is worth (cash already distributed **plus** the current marked value of what it still holds), divided by cash paid in. TVPI includes paper marks, so it is always ≥ DPI, and for a young fund it is *mostly* paper.
- **IRR — Internal Rate of Return.** The annualized percentage return, accounting for *when* the cash flowed (time value of money). A 3x that took 3 years has a far higher IRR than a 3x that took 10.

Early in a fund's life, TVPI can look spectacular while DPI is near zero — lots of paper gains, little cash returned. That is normal, and it is also exactly where crypto fund marketing can get slippery.

## 1. Fees and carry: how the fund gets paid

Now we can build the fund's actual business model. The plain-English version: **the GP runs on the fee and gets rich on the carry, and the carry only pays after the LPs are whole.** Let's put real dollars on it.

### The management fee stream

Take a **$500M fund** charging a 2% management fee during a five-year investment period, stepping to (say) 1.5% for the remaining five years. The fee is a *cost* to LPs and a *revenue* to the GP, paid every year regardless of performance:

```
Years 1-5:  2.0% × $500M = $10.0M/yr  × 5 = $50.0M
Years 6-10: 1.5% × $500M = $7.5M/yr   × 5 = $37.5M
Total management fees over the fund's life ≈ $87.5M
```

That is real money — roughly $87.5M to operate the fund over a decade, *before a single investment pays off.* It is why raising a large fund is itself a business goal: a bigger fund means a bigger fee base. It is also why LPs care about fees: every dollar of fee is a dollar not invested and not compounding for them.

But nobody builds a venture firm to collect the fee. The fee keeps the lights on; the carry is the prize.

### The carry: the GP's real payday

Carry is 20% of the *profit*, and profit is measured only after LPs get their committed capital back. The order of operations is a **distribution waterfall** — cash flows down a series of priorities, and each level fills before the next gets a drop. The figure shows the standard four steps.

![The distribution waterfall: LP capital returns first, then any hurdle, then the GP catch-up, then an 80/20 split of the rest](/imgs/blogs/the-crypto-vc-operating-model-3.webp)

Reading it top to bottom: **(1)** every dollar of LP paid-in capital comes back first; **(2)** if there is a hurdle, LPs get their preferred return (often ~8%/yr, but frequently *waived* in early-stage crypto funds); **(3)** a GP catch-up, if any; and **(4)** everything above that splits **80% to LPs, 20% to the GP.** The GP's 20% is the *last, most leveraged slice* — it exists only if there is genuine profit.

#### Worked example: fee + carry economics on a $500M fund

Suppose our $500M fund is a success and, over its life, turns the $500M into **$1.5 billion** of realized value — a 3x. Walk the waterfall (we'll assume no hurdle, the common early-stage crypto case):

```
Total realized value                 = $1,500M
- Return LP paid-in capital first    = $500M   -> back to LPs
= Profit to split                    = $1,000M

GP carry  (20% × $1,000M)            = $200M   -> to the GP
LP profit share (80% × $1,000M)      = $800M   -> to LPs

LPs receive:  $500M capital + $800M profit = $1,300M  (a 2.6x on their money)
GP receives:  $200M carry + ~$87.5M fees   ≈ $287.5M over the fund's life
```

Two things jump out. First, the LPs earned a 2.6x *net* even though the fund made 3x *gross* — the ~0.4x gap is the fee-and-carry drag, the price of hiring the manager. Second, the GP's carry ($200M) dwarfs its fees ($87.5M): on a *winning* fund, carry is the story. **The intuition: fees are a salary that is paid no matter what; carry is a performance bonus that only exists when LPs win big — which is exactly why GPs are so motivated to produce a home run.**

Now redo it with a wrinkle. Suppose instead the fund only returns **$500M** — it broke even, no profit.

```
Total realized value  = $500M
- Return LP capital    = $500M   -> back to LPs
= Profit               = $0
GP carry (20% × $0)    = $0
```

The GP collected ~$87.5M in fees over the decade and earned **zero** carry. This is the honest asymmetry of the model: the GP always eats (the fee), but only feasts (the carry) if the LPs feast first. Critics of the fee model point exactly here — a large fund can be a comfortable business for the GP on fees alone, even if it never produces a carry-worthy return. That tension shows up across the whole [incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto).

#### Worked example: carry on a single realized token exit

Zoom into one deal. The fund bought a token position for a **cost basis** (what it paid) of **$10M**. Years later it sells the vested portion for **$60M**. On a deal-by-deal ("deal-by-deal carry") basis — some funds compute carry per exit, others only across the whole fund — the split is:

```
Sale proceeds     = $60M
- Cost basis      = $10M   (returned toward LP capital)
= Gain            = $50M
GP carry (20%)    = $10M   -> to the GP
LP share (80%)    = $40M   -> to LPs
```

A single 6x exit threw off **$10M of carry** to the GP. String together three or four of those across a fund and you see how a top-tier crypto GP earns hundreds of millions on a single vintage. Note the subtlety, though: this carry is *earned on cash the fund actually received.* When a fund instead marks an unsold, locked position up to $60M, no carry is truly banked — it is paper. That gap is section 4.

### What this costs, and when it breaks

The fee model quietly reshapes incentives. Because the fee scales with fund *size*, a GP has a reason to raise the biggest fund it can — even if a smaller fund might post a better *percentage* return. And because carry rewards big multiples, a GP is pulled toward swing-for-the-fences bets and, in crypto specifically, toward *liquid* tokens that can be marked up (and sometimes sold) fast. Keep that pull in mind; it explains a lot of behavior later.

There is also a subtler cost the arithmetic makes plain: **fee and carry drag bite hardest when returns are mediocre.** In the 3x example the LPs' net was 2.6x — a ~0.4x haircut off the gross. Now take a modest winner: the fund returns **1.5x gross**, or $750M on $500M.

```
Realized value        = $750M
- Return LP capital    = $500M
= Profit               = $250M
GP carry (20%)         = $50M   -> to the GP
LP profit share (80%)  = $200M  -> to LPs
LPs get (before fees)  = $500M + $200M = $700M  ≈ 1.4x
```

So a "1.5x fund" hands LPs about **1.4x** before the ~$87.5M of management fees they also paid over the decade — and *below* 1.4x once those fees are counted. The carry alone shaved 0.1x; the fees shave more. **The intuition: fees and carry are a fixed toll on every dollar the fund touches, so the manager's cut eats a larger *share* of a thin return than of a fat one.** For an LP, that is the whole case for caring about fund size, fee terms, and — above all — a GP that can actually produce the home run that makes the toll worth paying.

## 2. The liquid and the locked book

An equity VC's holdings are almost entirely illiquid until an IPO — a startup share has no public market. A crypto fund's book is stranger: it is split into a part it can (eventually) *sell on a public exchange* and a part still frozen by vesting. Understanding that split is essential, because the fund's *stated* value and its *sellable* value can be miles apart.

![The locked book vs the liquid book: most of a young fund's value sits in locked, model-marked positions, not sellable tokens](/imgs/blogs/the-crypto-vc-operating-model-4.webp)

The figure contrasts the two sides. The **locked / vesting book** is seed tokens still on a **cliff** (a date before which *nothing* unlocks) or a vesting schedule (a gradual release over months or years). The fund holds them, but cannot sell them; it marks them to the last round or a model. Those are **paper gains**. The **liquid / unlocked book** is tokens that have vested and trade freely; the fund marks them at the live market price and — critically — can actually sell them to raise **cash**, which is what becomes DPI for LPs.

### Why the split exists — and why it's dangerous

When a project launches its token (its **TGE**, or *Token Generation Event*), insiders like the fund almost never get all their tokens at once. A typical schedule might be: a **one-year cliff** (nothing for 12 months), then **linear vesting** over the next two to three years. So even after a token is public and liquid *in the market*, the fund's own stake is mostly still locked. We trace this full pipeline in [the lifecycle of a token from seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock).

Here is the danger. Suppose a token launches and trades at a price that values the fund's *entire* stake — locked and unlocked — at $300M. The fund can write "$300M" in its books (mark-to-market on the live price). But if only 10% has vested, the fund could only *realize* $30M if it tried to sell today — and even that assumes the market could absorb the sale without the price collapsing (it usually cannot; see section 5). The other $270M is a number on a screen that depends on a price holding steady for years while supply unlocks. **A fund's headline value is a promise about the future; its liquid book is the only part that is money.**

This is not hypothetical fragility. When crypto prices fell hard in the first half of 2022, a16z's largest crypto fund reportedly *dropped about 40% in value* in that stretch, according to a report on figures shared with investors ([CoinDesk](https://www.coindesk.com/business/2022/10/26/a16zs-largest-crypto-fund-loses-40-value-in-first-half-of-2022-report), 2022-10-26). Much of that swing was marks moving, not cash lost — which is the whole point: a mark can round-trip up and back down before it is ever realized.

## 3. The crypto twist: a public exit years early

We now reach the mechanism that makes crypto venture a fundamentally different game from equity venture. It comes down to one word: **exit**.

An equity VC has only two ways to turn a startup stake into cash: the company goes public (**IPO**) or gets bought (**M&A**). Both are rare and late — commonly **7 to 10 years** after the seed investment. Until one happens, the VC's money is locked in an illiquid share with no buyer.

A crypto fund has a third door, and it opens early: the **public token market.** Once a portfolio project launches its token and the fund's vesting begins releasing, the fund can sell tokens on a public exchange — often just **1 to 2 years** after investing. No IPO gate, no underwriter, no S-1 registration, no acquirer required. The token *is* the liquidity event, and it arrives while an equity fund of the same vintage is still years from any exit at all. This is the structural gap we dissect in [why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock).

The timeline makes the contrast physical.

![Life of a $500M fund: token exits can begin around years 2–6, long before an equity IPO would arrive near the fund's end](/imgs/blogs/the-crypto-vc-operating-model-2.webp)

The fund closes at Year 0, invests through Years 1–3, sees a first token list publicly around Year 2, watches cliffs vest across Years 3–6, and can be *distributing cash to LPs* by Years 5–8 — all before Year 10. An equity fund's IPO, if it ever comes, lands at the far right of that same chart. The crypto fund gets to feed its LPs cash *mid-life*.

### Two kinds of crypto fund: liquid and venture

Not every crypto fund is the ten-year, illiquid venture vehicle we have been describing. The industry runs two archetypes, and the biggest firms often run both under one roof:

- A **venture fund** invests in early private rounds (SAFTs, token warrants, equity), holds illiquid and locked positions, and lives on the ~10-year clock — the model at the center of this post.
- A **liquid fund** (closer to a crypto *hedge fund*) trades tokens that are *already public*, marks its book to market daily, and offers LPs periodic redemption — say, quarterly — rather than a decade-long lockup. Its fee model can differ too: sometimes an annual performance fee on yearly gains rather than an end-of-life carry.

The distinction blurs in crypto in a way it never does in equity. A venture fund's locked tokens *become* liquid the instant they vest — so a crypto venture fund gradually mutates into a partly-liquid book whether it wants to or not. Several of the largest players run both strategies at once: Pantera has historically managed passive, hedge, and venture strategies side by side, and Polychain has run both liquid and venture funds ([Pantera](https://panteracapital.com/firm/); [Polychain on Wikipedia](https://en.wikipedia.org/wiki/Polychain_Capital)). So when a headline says a firm "manages $5 billion," that number can blend daily-liquid trading capital with decade-locked venture marks — two very different kinds of dollar wearing the same label.

### The economics of an early, cheap entry

Why is this such an advantage? Because the fund enters at a *tiny* price and the public enters at a large one. The picture below traces a single token through its funding rounds.

![One token, four prices: a fund can buy near $0.02 at seed while the public buys near $2.00 at launch — a 100x paper gap](/imgs/blogs/the-crypto-vc-operating-model-5.webp)

Each private round reprices the token upward — seed, then private, then strategic — and by the public **launch** the price can be a hundred times the fund's seed cost. The fund's *cost basis* stays near $0.02; only the market price climbs. The retail buyer's entry, at launch, is often the *top* of that curve.

#### Worked example: a seed token at $0.02 vs a public launch at $2.00

The fund buys **10 million tokens** at the seed price of **$0.02** each:

```
Cost of the position  = 10,000,000 × $0.02 = $200,000
```

At the public launch, the token trades at **$2.00**:

```
Market value at launch = 10,000,000 × $2.00 = $20,000,000
Return multiple        = $2.00 / $0.02      = 100x  (on paper)
Paper gain             = $20,000,000 - $200,000 = $19,800,000
```

A 100x. On a $200k check. This is the headline that makes crypto venture intoxicating — and it is *real* arithmetic. But notice the two words doing all the work: **on paper.** The fund cannot sell 10 million tokens at $2.00. Its own vesting locks most of the stake, and even the unlocked slice, dumped into the market at once, would crater the price. What the fund *realizes* is a different, smaller number — and that is section 4.

Still, even a fraction of a 100x is spectacular, and it arrives *years* before an equity fund sees a dime. That asymmetry — cheap entry, early public exit, minimal disclosure — is why token funds became kingmakers. The comparison table lays the two models side by side.

![Crypto token fund vs equity VC: same fee model, but the token fund reaches a public market years sooner with far less disclosure](/imgs/blogs/the-crypto-vc-operating-model-6.webp)

Same 2-and-20 fee model on both sides. Opposite exit reality: the equity fund enters through a priced equity round, waits 7–10 years for an IPO or M&A, sits under a ~180-day post-IPO lockup, and files mandated disclosures (an S-1 to go public, 10-Ks thereafter). The token fund enters through a SAFT or token warrant, exits on a public listing in ~1–2 years, is braked only by a *self-set* vesting schedule, and discloses rarely and voluntarily. The fee model is identical; the plumbing underneath is a different universe.

### When the early exit becomes a weapon

There is a sharp edge here. Because the fund can sell into the public market while retail is buying, the fund's *interests can diverge from the token holders'* the moment vesting begins. The fund is structurally an early, cheap seller; the public is structurally a late, expensive buyer. That is not a moral claim — it is arithmetic and incentives. It becomes a price event in section 5, and it is the reason the whole [Crypto Players series](/blog/trading/crypto/crypto-vc-and-market-makers) exists.

## 4. Paper vs cash: TVPI, DPI, and the mark

We keep circling the same crack, so let's stare straight at it. A crypto fund's *stated* performance and its *realized* performance can diverge dramatically, and the divergence lives in the marks. This is the most important section for anyone trying to judge whether a fund is actually good or just *marked* good.

Recall the two scorecards. **TVPI** counts everything the fund is worth including paper marks; **DPI** counts only cash returned. The chart shows a fund at two moments in its life.

![Paper gains vs cash returned: a young fund's value is mostly unrealized marks (TVPI), while cash actually returned (DPI) lags for years](/imgs/blogs/the-crypto-vc-operating-model-7.webp)

At **Year 3**, the fund reports a glittering **TVPI of 3.0x** — but its **DPI is 0.3x**. Almost all of that 3.0x is paper: locked tokens marked to a hot market price. Only 0.3x is cash LPs can spend. By **Year 8**, the picture has matured: TVPI has actually *drifted down* to 2.5x (some marks came back to earth), but DPI has climbed to **1.6x** as the fund vested, sold, and distributed real money. The dashed line at 1.0x is break-even — the moment LPs have their capital back.

#### Worked example: the paper-vs-cash gap

Concrete dollars on the Year-3 snapshot. LPs paid in **$500M**. The fund's positions are marked at **$1.5B** total, but it has only distributed **$150M** of cash so far.

```
Paid-in capital        = $500M
Current total value    = $1,500M   (mostly locked marks)
TVPI = 1,500 / 500     = 3.0x      <- the number in the pitch deck
Cash distributed       = $150M
DPI  = 150 / 500       = 0.3x      <- the number in LP bank accounts
```

The fund can truthfully say "we're up 3x." An LP who needs *money* has received 0.3x. Now let the market fall 50% before the locked tokens vest. The marks reprice:

```
New total value (marks halved on the locked book) ≈ $825M
New TVPI = 825 / 500  ≈ 1.65x
DPI still               = 0.3x   (the distributed cash didn't change)
```

The "3.0x fund" is now a "1.65x fund" without a single new transaction — just marks moving. **The intuition: TVPI is an opinion until it becomes DPI; a crypto fund's headline multiple is a mark that can evaporate, and only the cash it has actually returned is real.** This is why sophisticated LPs increasingly push on DPI — "show me the cash" — rather than celebrating a paper TVPI. It is also why a fund is motivated to *realize* gains during a strong market, which brings us to the part you feel directly.

### The unrealized-gain tax problem (a real wrinkle)

There is a nasty second-order effect worth naming. In some structures a GP can earn carry on *marked* (unrealized) gains during good years, then face **clawback** — returning previously paid carry — if those marks later collapse and the fund ends below the threshold. Clawbacks are legally messy and sometimes uncollectable. This is one more reason the paper-vs-cash gap is not an accounting nicety: it decides who actually keeps the money when a cycle turns.

## 5. How it shows up in your price

Everything above has been about the fund. This section is about *you* — the person trading the token — because the fund's operating model reaches directly into the order book you are staring at.

The mechanism is the **unlock**. A fund's tokens vest on a schedule that is, by design, *public and known* — it is written in the tokenomics. On each unlock date, a fresh tranche of tokens the fund could not previously sell lands in its wallet, now sellable. And a fund that is up 50x on paper, watching a mark it fears might fall, has every incentive to convert some of that paper into cash. The figure traces what happens next.

![How a fund's selling shows up in your price: a scheduled unlock routes fresh supply into a thin order book and retail becomes the exit liquidity](/imgs/blogs/the-crypto-vc-operating-model-8.webp)

A known unlock cliff hits; vested tokens land in the fund's wallet; the fund sells into a **float** (the freely-trading supply) that is often thin; the order book has to *absorb* that selling; and the price a retail buyer sees drops through **slippage** (the price moving against a large order as it eats through the resting bids). The retail buyer, in aggregate, is the one *on the other side* — the exit liquidity for the fund's realized gain.

#### Worked example: selling into a thin float

Say a token has a **circulating float** worth **$50M** of resting buy orders near the current price, and an unlock hands the fund **$10M** of tokens it decides to sell over a day.

```
Sell order size        = $10M
Nearby resting bids    = $50M
Fraction of the book   = 10 / 50 = 20%
```

Eating 20% of the visible bids in a session pushes the price down materially — a **supply overhang**. And because the unlock date was *known in advance*, sophisticated traders often *front-run* it: they sell (or short) ahead of the unlock, anticipating the fund's supply, which pushes the price down *before* the unlock even arrives. Either way, the fund's operating model — cheap entry, locked stake, scheduled release — becomes a scheduled headwind in the price. We cover the price plumbing itself in [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move).

### The retail-defense takeaway

Here is the single habit that protects you: **before you buy a token, read its vesting and unlock calendar.** It is usually public (project docs, tokenomics dashboards, on-chain vesting contracts). Ask three questions:

1. **Who holds the locked supply, and at what cost basis?** If funds bought at $0.02 and the token trades at $2.00, they are up 100x and have every reason to sell when they can.
2. **What percentage of supply unlocks, and when?** A large unlock relative to the float is a scheduled wave of selling. A token that is "only 10% circulating" is 90% overhang.
3. **Am I the early buyer or the late one?** If you are buying at a public launch while insiders are still locked, you are — structurally — the exit liquidity being set up. That does not mean *don't buy*; it means buy with your eyes open about who is on the other side. (This is educational, not financial advice.)

None of this requires believing in a conspiracy. It is just the arithmetic of the operating model, read forward.

## Common misconceptions

**"A crypto VC makes its money picking good projects."** Partly. But the fee model means a GP earns ~2% of committed capital *every year regardless of picks*. On a $500M fund that is ~$10M a year to operate before any project succeeds. Good picks drive the *carry*; the *fee* rewards asset-gathering. Both are real; conflating them hides the incentive to simply raise a bigger fund.

**"If a fund reports a 5x, the LPs made 5x."** No. A reported multiple is almost always **TVPI** — total value including unrealized marks. What LPs can *spend* is **DPI** — cash returned. A young fund can show 5x TVPI and 0.2x DPI. Until marks become distributions, "5x" is an opinion.

**"The fund and the token holders want the same thing."** Only until vesting starts. The fund entered at a fraction of a cent and holds a locked stake it is structurally motivated to sell; the public entered at the market price and holds a liquid stake. Their *time horizons and cost bases* diverge, which is why unlocks are contentious. Marketing calls it "aligned incentives"; the arithmetic frequently disagrees.

**"Carry means the GP gets 20% of everything."** No — 20% of the *profit*, and only after LPs get their capital back (and clear any hurdle). On a fund that merely returns capital, carry is *zero*. The GP's 20% is the last slice of the waterfall, not the first.

**"Crypto funds are just faster equity VCs."** The speed is real but it is not the deepest difference. The deepest difference is the **public** exit with **self-set** lockups and **minimal disclosure**. An equity insider must clear an IPO gate, an underwriter, a registration statement, and a ~180-day lockup. A token insider is braked mainly by a vesting schedule the project wrote for itself. Same fee model, categorically different rules.

**"A fund always dumps its whole stake the instant it unlocks."** Often, but not always. A fund that wants to keep raising money from LPs and keep getting invited into the best deals has a *reputation* to protect, and visibly crashing its own portfolio tokens is bad for that business. Some funds sell gradually, route size through OTC desks to avoid moving the price on-screen, stake tokens for yield, or hold high-conviction winners for years. The unlock is the *moment selling becomes possible*, not proof it happened — which is why watching on-chain flows around an unlock, not just the calendar, is the real skill.

**"A big-name fund's backing guarantees the token is a good investment."** A fund's investment tells you the fund believes in the *project*, at the fund's *cost basis*, with the fund's *exit timeline*. None of those are yours. The fund can win handsomely on a token that later falls 80% from its launch price, because the fund's entry was 100x lower than yours.

## How it shows up in real markets

Named, dated episodes where the operating model above played out. Every figure here is sourced; where a number is an estimate or a report, it is labeled as such.

### 1. a16z crypto and the $4.5B mega-fund

In May 2022 — *during* a savage market downturn — Andreessen Horowitz's crypto arm announced a **$4.5 billion** fund, at the time the largest single crypto fund ever raised in venture, split roughly **$1.5B for seed** investments and **$3B for later-stage** deals ([Forbes](https://www.forbes.com/sites/alexkonrad/2022/05/25/a16z-crypto-record-4th-fund-doubles-down-on-web3-amid-market-crash/); [TechCrunch](https://techcrunch.com/2022/05/25/amid-crypto-downturn-a16z-debuts-4-5-billion-web3-mega-fund/)). It followed a **$2.2B** third fund from June 2021, bringing a16z's total crypto capital raised past **$7.6 billion**. The mechanics of this post are all visible in that one raise: a bigger fund is a bigger *fee base* (2% of $4.5B is $90M a year), and the timing — raising into a crash — reflects the ten-year clock, which lets a GP deploy patiently across a cycle. Notably, that largest fund reportedly fell **~40% in the first half of 2022** on marks ([CoinDesk](https://www.coindesk.com/business/2022/10/26/a16zs-largest-crypto-fund-loses-40-value-in-first-half-of-2022-report)) — paper moving, not cash lost — and by May 2026 a16z had returned to raise a smaller **$2.2B** fifth fund ([Fortune](https://fortune.com/2026/05/05/a16z-crypto-andreessen-horowitz-fifth-fund-2-2-billion/)), a right-sizing after the boom.

### 2. Paradigm: from a record $2.5B to a disciplined $1.2B

Paradigm raised **$2.5 billion** in November 2021 — then the largest crypto venture fund in history, reportedly twice its initial target ([Blockworks](https://blockworks.com/news/paradigm-launches-2-5-billion-crypto-fund)). As the cycle cooled it raised a smaller **$850M** early-stage fund in 2024, and in July 2026 a **$1.2 billion** fourth fund that explicitly broadened beyond crypto into AI and robotics ([TechCrunch](https://techcrunch.com/2026/07/08/crypto-vc-firm-paradigm-raises-1-2b-to-invest-in-technical-frontier-startups/)). The arc — $2.5B at the peak, then right-sizing — is the fund-size-follows-the-cycle pattern the fee model encourages: raise big when LP appetite is hot, shrink when it cools.

### 3. Polychain and the long road from mark to cash

Olaf Carlson-Wee founded Polychain in 2016 with roughly **$5 million**; the firm's assets swung from about **$1 billion** in 2017 down to **$592 million** at the end of 2018, then to around **$5 billion** by 2022 (per various reports) — a vivid illustration of how much a crypto fund's stated AUM is *marks* that breathe with the market ([Polychain on Wikipedia](https://en.wikipedia.org/wiki/Polychain_Capital)). Polychain raised roughly **$750M** for a third venture fund in 2022 and began a fourth around 2023 ([Fortune](https://fortune.com/crypto/2023/07/18/crypto-vc-polychain-200-million-staff-shakeup/)). Crucially, in May 2024 Bloomberg reported Polychain **made distributions to LPs** in two of its funds ([Bloomberg](https://www.bloomberg.com/news/articles/2024-05-10/crypto-vc-firm-polychain-makes-payouts-to-investors-in-two-funds)) — the moment paper (TVPI) finally became cash (DPI). That transition, years after the marks first spiked, is exactly the paper-vs-cash lag from section 4.

### 4. Pantera and the $13M that started an industry

In 2013, Pantera launched the **first US institutional Bitcoin fund** with just **$13 million**, when Bitcoin traded around **$65** ([Pantera](https://panteracapital.com/firm/); [Pantera on Wikipedia](https://en.wikipedia.org/wiki/Pantera_Capital)). By the 2020s Pantera reportedly managed on the order of several billion dollars across passive, hedge, and venture strategies (reported figures vary, roughly **$4.8–5.6 billion**). The lesson is the early-entry advantage taken to its extreme: a fund that bought Bitcoin near $65 has a cost basis so low that almost any later price is a monumental multiple — the same "cheap entry, public exit" arithmetic as our $0.02 token, just with the longest possible time horizon.

### 5. The SAFT that launched a template: Filecoin

When Protocol Labs sold Filecoin in 2017, it needed a legal way to sell a token that *did not exist yet.* Its answer — the **SAFT**, built with AngelList and Cooley — let it raise over **$205 million** by 2017-09-07 from accredited investors buying a right to *future* tokens ([Protocol Labs](https://www.protocol.ai/blog/filecoin-sale-completed/)). Funds bought SAFTs at private prices; the tokens later traded publicly. The SAFT (and its successor, the SAFE-plus-token-warrant) is the contractual seed of every "fund buys at $0.02, public buys at $2.00" story in this post — the instrument that lets a fund acquire a cheap claim on a token years before anyone else can touch it.

### 6. The AUM game and why fund sizes ballooned

Trace the fund sizes in this section — a16z at $4.5B, Paradigm at $2.5B, Polychain and Pantera each around several billion — and a pattern emerges that the fee model predicts perfectly. Because the management fee is a percentage of *committed capital*, a larger fund is a larger, guaranteed revenue stream independent of performance: 2% of a16z's $4.5B fund is roughly **$90 million a year** to operate, regardless of whether the investments work. That is a powerful gravitational pull toward raising ever-bigger funds during a bull market, when LP appetite is hottest. The correction is just as telling: when the cycle cooled, both a16z (to a $2.2B fifth fund by 2026) and Paradigm (to a $1.2B fourth fund by 2026) *right-sized downward* ([Fortune](https://fortune.com/2026/05/05/a16z-crypto-andreessen-horowitz-fifth-fund-2-2-billion/); [TechCrunch](https://techcrunch.com/2026/07/08/crypto-vc-firm-paradigm-raises-1-2b-to-invest-in-technical-frontier-startups/)). Fund size, in other words, is a barometer of LP enthusiasm as much as of conviction — and a fund raised at the top of a cycle carries a fee base built for a market that may not return for years.

## When this matters to you

If you never buy an individual token, this is spectator knowledge — useful for understanding headlines about "a16z's new fund" without mystique. But the moment you consider buying a token, the operating model above is *directly* about your money, because it tells you who is likely on the other side of your trade and when they are motivated to sell.

Three practical reflexes fall out of it, none of which is advice to buy or avoid anything — just ways to see clearly:

- **Translate every "up 10x" into "TVPI or DPI?"** A fund up 10x on marks and a fund that has *returned* 10x in cash are different animals. The same discipline applies to the tokens you hold: a paper gain is not a realized one until you (or the insiders) can actually sell.
- **Treat the vesting calendar as a price forecast, not a footnote.** A large scheduled unlock relative to the float is a known future supply of selling. It is one of the few genuinely *predictable* things in crypto.
- **Ask where you sit on the entry curve.** If insiders entered at a fraction of a cent and you are buying at the public launch, you are the expensive, late buyer by construction. That can still be a fine trade — but only if you have priced in that the cheap, early holders can sell to you on a schedule.

The crypto VC is not a villain in this story. It is a rational business running a well-understood model — pooled capital, fees, carry, a ten-year clock — that happens to sit on top of an instrument (the token) whose public liquidity arrives absurdly early and whose disclosure arrives barely at all. Learn the model, and the price action stops looking like magic and starts looking like arithmetic.

## Sources & further reading

Primary and reported sources behind the figures in this post:

- a16z crypto Fund IV ($4.5B, May 2022; $1.5B seed / $3B venture; >$7.6B total): [Forbes](https://www.forbes.com/sites/alexkonrad/2022/05/25/a16z-crypto-record-4th-fund-doubles-down-on-web3-amid-market-crash/), [TechCrunch](https://techcrunch.com/2022/05/25/amid-crypto-downturn-a16z-debuts-4-5-billion-web3-mega-fund/). Fifth fund ($2.2B, May 2026): [Fortune](https://fortune.com/2026/05/05/a16z-crypto-andreessen-horowitz-fifth-fund-2-2-billion/). Largest fund down ~40% in H1 2022 (reported): [CoinDesk](https://www.coindesk.com/business/2022/10/26/a16zs-largest-crypto-fund-loses-40-value-in-first-half-of-2022-report).
- Paradigm One ($2.5B, Nov 2021): [Blockworks](https://blockworks.com/news/paradigm-launches-2-5-billion-crypto-fund). Fourth fund ($1.2B, July 2026): [TechCrunch](https://techcrunch.com/2026/07/08/crypto-vc-firm-paradigm-raises-1-2b-to-invest-in-technical-frontier-startups/). Firm history: [Wikipedia](https://en.wikipedia.org/wiki/Paradigm_(venture_capital_firm)).
- Polychain Capital (founding, AUM swings, funds, 2024 LP distributions): [Wikipedia](https://en.wikipedia.org/wiki/Polychain_Capital), [Fortune](https://fortune.com/crypto/2023/07/18/crypto-vc-polychain-200-million-staff-shakeup/), [Bloomberg](https://www.bloomberg.com/news/articles/2024-05-10/crypto-vc-firm-polychain-makes-payouts-to-investors-in-two-funds).
- Pantera Capital (first US institutional Bitcoin fund, 2013, $13M at ~$65 BTC): [Pantera](https://panteracapital.com/firm/), [Wikipedia](https://en.wikipedia.org/wiki/Pantera_Capital).
- SAFT and the Filecoin sale (>$205M, 2017-09-07): [Protocol Labs — sale completed](https://www.protocol.ai/blog/filecoin-sale-completed/), [Protocol Labs — the SAFT Project](https://www.protocol.ai/blog/announcing-saft-project/).
- Fund economics — 2-and-20, hurdles, DPI/TVPI/IRR, the ten-year term and J-curve: [VC Beast (2 and 20)](https://vcbeast.com/2-and-20-fee-structure-explained), [TechCrunch (fee variation)](https://techcrunch.com/2023/09/27/venture-fund-2-and-20/), [Carta (fund performance metrics)](https://carta.com/learn/private-funds/management/fund-performance/).
- Token warrants and SAFEs: [Pulley](https://pulley.com/guides/token-warrant), [DAOSPV](https://blog.daospv.com/token-warrants-safes-and-safts-a-founders-guide-to-web3-fundraising-instruments/).

Sibling posts on this blog:

- [Crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) — the series hub: the whole cast of funds and trading firms.
- [Why a token is not a stock](/blog/trading/crypto-players/why-a-token-is-not-a-stock) — the legal gap behind the early public exit.
- [The lifecycle of a token: seed to unlock](/blog/trading/crypto-players/the-lifecycle-of-a-token-seed-to-unlock) — the pipeline every token walks, and where each player enters.
- [Cui bono: the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) — who profits at each step, and where insiders and retail collide.
- [How crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) — the order-book mechanics behind the unlock overhang.

*Educational, not investment advice. Every fund size and dated figure above is sourced; the step-by-step dollar walkthroughs use round hypothetical numbers to keep the arithmetic clean.*
