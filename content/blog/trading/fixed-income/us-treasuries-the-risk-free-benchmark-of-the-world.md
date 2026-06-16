---
title: "US Treasuries: the risk-free benchmark of the world"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into US Treasury securities — bills, notes, bonds, FRNs and TIPS — how they are auctioned, why they are treated as the risk-free benchmark, and how the 10-year yield prices almost everything else on earth."
tags: ["fixed-income", "bonds", "treasuries", "risk-free-rate", "auctions", "repo", "collateral", "10-year-yield", "safe-assets", "benchmark"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — A US Treasury is a loan to the US government, and because that borrower is treated as the safest on earth, the yield on Treasuries has become the *benchmark* — the baseline interest rate that every other price in finance is measured against.
>
> - The Treasury family is one issuer with five products sorted by maturity: **bills** (one year or less, sold at a discount with no coupon), **notes** (2–10 years), **bonds** (20–30 years), **FRNs** (floating-rate notes), and **TIPS** (inflation-protected). Same credit, different shapes.
> - Treasuries are sold at **auctions** run on a published calendar. Big banks called **primary dealers** are required to bid; the **bid-to-cover ratio** — total bids divided by the amount sold — measures how hungry the buyers were. A 2.5× cover means \$2.50 was offered for every \$1.00 sold.
> - The newest issue of each maturity is **on-the-run** — the most liquid, slightly lower-yielding bond everyone trades; older ones are **off-the-run**, a touch cheaper to own. That tiny gap is the price of liquidity.
> - The **10-year Treasury yield** is the single most important number in global finance. Mortgages, corporate bonds, emerging-market debt, and the value of stocks are all priced as "the 10-year, plus a spread for extra risk." Move the 10-year and you move the world.
> - "Risk-free" means **free of default risk**, not free of *all* risk. A Treasury can still lose value if rates rise (the SVB story) or if a self-inflicted **debt-ceiling** standoff threatens a technical default. The benchmark is sturdy, not magic.

Pick any interest rate in the world — the rate on a 30-year mortgage in Ohio, the yield a company in Germany pays to borrow, the cost for Brazil to issue debt in dollars — and trace it back far enough, and you will arrive at the same place: the yield on a US Treasury. Everything in finance is priced as "the risk-free rate, plus something extra for the risk we are actually taking." And the thing the whole planet has agreed to treat as the risk-free rate is the IOU of the United States government.

This is a strange and powerful fact. The United States is roughly \$36 trillion in debt. And yet its debt is considered the *safest* asset on earth — so safe that banks hold it as their emergency cash, central banks from Tokyo to Riyadh stockpile it as reserves, and traders use it as the universal collateral they pledge to borrow from each other. The market for US Treasuries is the deepest, most liquid pool of any asset anywhere: on a normal day, hundreds of billions of dollars of Treasuries change hands, more than the entire US stock market.

How did one country's IOUs become the unit everyone else measures themselves against? Not by decree, but by a slow accretion of trust: a borrower that always paid, in a currency the world wanted, traded in a market so deep you could never get stuck holding it. Over decades, that combination — never defaulting, borrowing in the global reserve currency, and offering a market too big to break — turned US debt from "an investment" into "the reference point." When a trader anywhere on earth wants to know what money costs, they look at the Treasury yield first and adjust from there. That is what it means to be the benchmark: not the best return, but the universally agreed *starting point* for every other return.

![A grid of the five US Treasury security types sorted by maturity, from short-term bills to long bonds, with FRNs and TIPS](/imgs/blogs/us-treasuries-the-risk-free-benchmark-of-the-world-1.png)

The figure above is the mental model for this whole post: one issuer — the US Treasury — selling a small family of products that differ mostly in *how long* they last and *how* they pay. Master that family, understand how they are sold at auction, and grasp why the 10-year yield sits at the center of everything, and you will understand the bedrock that the rest of the bond market — indeed the rest of finance — is built on. Let us build it from zero.

## Foundations: what a Treasury actually is

Before we meet the products, we need a shared vocabulary. A reader with no finance background needs five ideas in place: what a bond is, what par and coupon mean, what yield means, what "discount" pricing is, and why we call the US government "risk-free." A practitioner can skim; a beginner cannot proceed without these.

### A bond is a tradable loan

A **bond is a tradable loan.** When you buy one, you are lending money to whoever issued it. For a Treasury, the borrower is the US federal government, which raises money this way to fund the gap between what it spends and what it collects in taxes. In return, the government promises to pay back the amount borrowed on a fixed future date, and (for most Treasuries) to pay interest along the way.

The word that matters is *tradable.* Unlike the loan you take from a bank, a Treasury can be sold to someone else at any time before it matures. That is what makes it an *asset* — something you can own, value, and resell in seconds — rather than a private IOU. The Treasury market is simply the giant arena where these loans-to-the-government change hands.

### Par, coupon, and maturity

Three words describe almost any bond. **Par** (or *face value*) is the amount printed on the bond — what the government pays you back at the end. The standard unit is \$1,000 of par, though Treasuries are issued in increments of \$100. The **coupon** is the fixed annual interest rate the bond pays, quoted as a percentage of par: a 4% coupon on \$1,000 of par pays \$40 a year (usually split into two \$20 payments, six months apart). The **maturity** is the date the loan ends and par is repaid.

We will use one recurring example throughout: a **\$10,000 face-value 6-month Treasury bill**, and alongside it a **\$1,000 par, 4% coupon, 10-year Treasury note**. Keeping the same numbers lets the ideas compound. We will also occasionally compare a Treasury to a bond from a fictional company, **Northwind Corp**, to make the meaning of "risk-free" concrete.

### Yield: the return you actually earn

The **coupon** is fixed forever, but the **yield** is the return you actually earn given the price you pay *today.* If you buy a \$1,000 bond with a \$40 coupon for exactly \$1,000, your yield is \$40 ÷ \$1,000 = 4%. But bond prices move, and here is the seesaw that runs through all of fixed income: **when the price changes, the yield changes the opposite way.** Pay only \$950 for that same \$40-coupon bond and your yield rises to roughly \$40 ÷ \$950 ≈ 4.2%, because you are getting the same fixed payments for less money. Price down, yield up; price up, yield down. (The dedicated post [price and yield, the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds) covers this in full.)

One unit of jargon you will meet constantly: a **basis point** is one hundredth of a percentage point — 0.01%. "The 10-year yield rose 40 basis points" means it went up 0.40%, say from 4.10% to 4.50%. Traders quote bond moves in basis points (written "bps") because the numbers are small and precision matters: on a multi-trillion-dollar market, a single basis point is a lot of money.

### "Discount" pricing: how a bill works without a coupon

Most Treasuries pay coupons. The shortest ones — **Treasury bills** — do not. Instead they are sold at a **discount**: you pay *less* than face value today, and the government pays you the *full* face value at maturity. Your return is the gap between what you paid and what you get back. Buy a \$10,000 bill for \$9,800 and hold it six months; you receive \$10,000, and the extra \$200 is your interest. There is no coupon to wait for — the whole return is baked into the discounted purchase price. We will price exactly this bill in a moment.

### Why the US government is the benchmark for "risk-free"

When we call a borrower "risk-free," we mean something narrow and specific: the chance it fails to pay you back, *in full and on time*, is treated as effectively zero. This is called freedom from **credit risk** or **default risk** — the risk that a borrower cannot or will not pay.

The US government is treated this way for two reasons. First, it has never failed to pay its dollar debt. Second — and this is the deeper reason — it borrows in *its own currency*, the dollar, which it alone can create. A company like Northwind Corp can run out of dollars and default; the US Treasury, in the last resort, can have the Federal Reserve create the dollars it owes. That does not make Treasuries free of *every* risk (we will see they are not), but it does mean the one risk that usually scares lenders — that you simply do not get paid — is, for Treasuries, off the table. That single property is what lets the whole world use the Treasury yield as its baseline.

## The Treasury security family

The US Treasury is one borrower, but it sells a small family of products. They share the same credit (the full faith and credit of the United States) and differ mostly in two ways: *how long* the loan lasts and *how* it pays you. Knowing the family by heart is the first step to reading any financial headline.

| Product | Maturity | How it pays | Key feature |
|---|---|---|---|
| **Bills (T-bills)** | 4, 8, 13, 17, 26, 52 weeks | Sold at a discount, no coupon | The market's cash equivalent |
| **Notes (T-notes)** | 2, 3, 5, 7, 10 years | Semi-annual coupon | The core of the market; the 10-year is the benchmark |
| **Bonds (T-bonds)** | 20, 30 years | Semi-annual coupon | The "long end" — most sensitive to rate moves |
| **FRNs** | 2 years | Floating coupon (resets weekly) | Pays more when short rates rise |
| **TIPS** | 5, 10, 30 years | Coupon on an inflation-adjusted principal | Protects against inflation |

A few notes on the family. **Bills** are the workhorses of cash management: corporations, money-market funds, and banks park spare cash in them because they mature so soon that their price barely moves. **Notes and bonds** are where investors take *duration* — exposure to interest-rate moves — in exchange for higher yield; the 2-year note tracks what the market thinks the Fed will do, and the 10-year note is the global benchmark we will keep returning to. **FRNs** (floating-rate notes) have a coupon that resets every week to track the bill rate, so they barely lose value when rates rise — useful when you fear higher rates. **TIPS** (Treasury Inflation-Protected Securities) adjust their principal up with the consumer price index, so they pay a *real* (after-inflation) return; we will not go deep on them here, but they are the instrument that reveals what the market expects inflation to be.

#### Worked example: reading the family from a single yield headline

Suppose a news ticker says "the 2-year yields 4.2%, the 10-year yields 4.5%, the 30-year yields 4.7%." What is it telling you? Three different loans to the *same* borrower — the US government — at three different lengths. The credit risk is identical (zero, by assumption); the only thing changing is maturity. The fact that the 10-year pays \$0.30 more per \$100 of yield than the 2-year, and the 30-year more still, is the **yield curve** sloping upward — investors demanding extra compensation to lock their money up for longer. Plot yield against maturity for all of these at once and you get [the yield curve](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance), the single most-watched chart in finance.

*Intuition: the Treasury family is one borrower seen at many time horizons; the differences between the products are about time and inflation, never about whether you get paid.*

### STRIPS: taking a Treasury apart

There is a sixth member of the family worth knowing, because it reveals what a coupon bond *really* is underneath. A coupon Treasury is, mechanically, a bundle of separate promises: "I will pay you \$20 in six months, \$20 in a year, \$20 in eighteen months … and \$1,000 at the end." Each of those promises is a standalone cash flow. **STRIPS** — Separate Trading of Registered Interest and Principal of Securities — is the Treasury program that lets dealers literally split a coupon bond into its individual pieces and trade each one separately.

When you strip a 10-year note, you get twenty little **zero-coupon** bonds (one for each semi-annual coupon, each paying \$20 on a single future date and nothing before) plus one **principal STRIP** (the \$1,000 at year ten). A zero-coupon bond has no coupon — like a bill, it is sold at a discount and pays a single lump sum at maturity, except it can be decades away. Strips matter for two reasons. First, they let pension funds and insurers buy a precise cash flow on a precise future date to match a precise liability — exactly the *immunization* logic in [immunization and duration matching](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge). Second, the prices of strips are how analysts read the **spot rate curve** — the pure interest rate for each single future date — which is the foundation of all bond pricing (see [spot rates, the zero curve, and bootstrapping](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping)).

The deep point is that a Treasury note is not an atom; it is a molecule of dated cash flows, and the market can price each atom on its own. Everything we say about discounting one bill applies, twenty times over, to one stripped note.

## Pricing a T-bill: the discount math

The bill is the simplest Treasury to price because it has no coupon to discount — the entire return is the gap between purchase price and face value. But the way bills are *quoted* trips up almost every beginner, so let us be careful.

![A before-and-after diagram of a Treasury bill, paying less than face value today and receiving full face value at maturity](/imgs/blogs/us-treasuries-the-risk-free-benchmark-of-the-world-2.png)

There are two numbers people quote for a bill, and they are not the same:

1. The **discount rate** (also "bank discount yield"): the discount expressed as a fraction of *face value*, annualized using a 360-day year. This is the convention the Treasury and the news use, and it is slightly misleading because it divides by face value (what you get) rather than price (what you pay).
2. The **investment yield** (also "bond-equivalent yield" or "coupon-equivalent yield"): the actual return on the money you put in, divided by *price* and annualized over 365 days. This is the honest number — the one you can compare to a bank deposit.

Let us price our recurring bill and compute both.

#### Worked example: pricing a 6-month T-bill quoted at a discount

A 26-week (roughly 182-day) bill is quoted at a **discount rate of 4.00%**. We hold \$10,000 of face value. How much do we pay, and what do we actually earn?

**Step 1 — the dollar discount.** The discount rate uses a 360-day year and is applied to face value:

$$ \text{discount} = \text{face} \times \text{rate} \times \frac{\text{days}}{360} = \$10{,}000 \times 0.04 \times \frac{182}{360} = \$202.22 $$

**Step 2 — the price you pay.** You pay face value minus the discount:

$$ \text{price} = \$10{,}000 - \$202.22 = \$9{,}797.78 $$

So you hand over \$9,797.78 today and receive \$10,000 in 182 days. Your dollar profit is \$202.22.

**Step 3 — the investment (true) yield.** The honest return divides the profit by what you actually *paid* and annualizes over a 365-day year:

$$ \text{investment yield} = \frac{\text{face} - \text{price}}{\text{price}} \times \frac{365}{\text{days}} = \frac{\$202.22}{\$9{,}797.78} \times \frac{365}{182} = 4.14\% $$

Notice the gap: the headline "discount rate" was 4.00%, but the yield you actually earn on your money is **4.14%**. The discount rate *understates* your real return, because it divides by the larger number (face) and uses the shorter 360-day year. Whenever you compare a bill to a savings account, use the investment yield — it is the apples-to-apples number.

*Intuition: a bill's "discount rate" is a quoting convention that flatters the borrower; the investment yield is what actually lands in your pocket, and it is always a bit higher.*

In each symbol: $\text{face}$ is the \$10,000 you get back, $\text{price}$ is the \$9,797.78 you pay, $\text{days}$ is the 182-day holding period, and the 360 vs 365 difference is purely a calendar convention that the two formulas disagree on.

To see how the price and the two yields move as the discount rate changes, it helps to lay several bills side by side. The table figure below prices our \$10,000, 182-day bill at four different quoted discount rates, and shows the price you pay, the dollar profit, and the honest investment yield in each case.

![A table pricing a six-month Treasury bill at four discount rates, showing the price paid the dollar profit and the true investment yield](/imgs/blogs/us-treasuries-the-risk-free-benchmark-of-the-world-7.png)

Two patterns jump out of the table. First, a higher discount rate means a lower price — you pay less today for the same \$10,000 at the end, which is just the price-yield seesaw again. Second, the investment yield is *always* a little above the quoted discount rate, and the gap *widens* as rates rise (about 14 bps at a 4% discount, more at 8%), because the discount-rate convention divides by the larger face value and uses the shorter 360-day year. When rates are near zero the two numbers nearly coincide; when rates are high the difference is meaningful.

#### Worked example: comparing a bill to a bank deposit

Your bank offers a 6-month certificate of deposit (CD) at **4.05%**. A 6-month T-bill is quoted at a **4.00% discount rate**. Which pays more? The naive comparison says the CD wins, 4.05% vs 4.00%. But that compares the CD's *true* yield to the bill's *misleading* discount rate. Convert the bill to its investment yield — **4.14%**, as we computed — and the bill actually pays more: 4.14% vs 4.05%. And the bill has two extra advantages a CD lacks: its interest is exempt from state and local income tax (a real edge in high-tax states), and you can sell it any day in the world's most liquid market, whereas breaking a CD early usually costs a penalty. The lesson is that the *quoting convention matters*: always convert a bill to its investment yield before comparing it to anything else.

*Intuition: never compare a bill's headline discount rate to a deposit rate — convert it to the investment yield first, or you will systematically underrate the bill.*

## How Treasuries are sold: the auction

The government does not phone investors one at a time. It sells new Treasuries through **auctions** — competitive sales run on a published calendar, week after week, year after year. The auction is the beating heart of the market: it is where new debt is born and where the world signals, in real money, how much it wants to lend to the United States.

![A pipeline of the Treasury auction process from announcement to bidding to settlement](/imgs/blogs/us-treasuries-the-risk-free-benchmark-of-the-world-4.png)

The process runs in five steps:

1. **Announcement.** A few days ahead, the Treasury announces the size of the auction (e.g. "\$42 billion of 10-year notes") and the date.
2. **Bidding.** On auction day, buyers submit bids electronically. There are two kinds. A **competitive bid** specifies the *yield* the bidder is willing to accept; these come from big institutions. A **non-competitive bid** says "I will take whatever yield the auction clears at, just give me the bonds" — this is how smaller investors and the public participate, guaranteeing they get filled.
3. **The clearing yield.** The Treasury fills non-competitive bids first, then accepts competitive bids from the lowest yield upward (lowest yield = highest price = best deal for the government) until the whole amount is sold. The yield of the *last* bid accepted — the **high yield** or **stop-out yield** — becomes the yield *everyone* in that auction pays. This is a *single-price* (Dutch) auction: even the bidder who asked for a lower yield gets the higher clearing yield. That design encourages aggressive, honest bidding.
4. **The metrics.** The auction publishes results minutes later, and traders pounce on them: the high yield versus where the bond was trading just before (a "tail" if it cleared cheap, a sign of weak demand), and the **bid-to-cover ratio** (next section).
5. **Settlement.** A day or two later, the bonds are delivered and the cash is paid. The new bond becomes the **on-the-run** issue and starts trading.

### Primary dealers: the bidders who must show up

The auction works because a set of large banks — currently about two dozen — are designated **primary dealers**. In exchange for the privilege of trading directly with the Federal Reserve, they are *obligated* to bid in every auction and to make markets in Treasuries afterward. They are the backstop that guarantees every auction gets covered: even if no one else shows up, the dealers must bid. This is a quiet but crucial piece of plumbing — it is *why* the US has never failed to sell its debt at auction.

#### Worked example: reading a bid-to-cover ratio

The Treasury auctions **\$42 billion** of 10-year notes. When bidding closes, it has received **\$105 billion** of total bids. The **bid-to-cover ratio** is:

$$ \text{bid-to-cover} = \frac{\text{total bids received}}{\text{amount sold}} = \frac{\$105\text{B}}{\$42\text{B}} = 2.50 $$

A bid-to-cover of **2.50** means \$2.50 was offered for every \$1.00 the Treasury actually sold. That is healthy demand — comfortably above the rough 2.2–2.5 range that 10-year auctions tend to print. If the next month's auction came in at **2.10**, with the high yield clearing several basis points *above* where the bond traded beforehand (a "tail"), traders would read it as soft demand: the world wanted this debt a little less, the government had to pay up to sell it, and yields across the market would likely drift higher in response. A *strong* cover (say 2.7) with the auction clearing *through* the pre-auction level signals hunger for safe US debt and tends to nudge yields down.

*Intuition: the bid-to-cover ratio is a real-money applause meter for US credit — high means the world is eager to lend, low means the government had to sweeten the deal.*

### Why a single-price auction is clever

It is worth pausing on the *design* of the auction, because it is a quiet masterpiece of mechanism design. The Treasury could run a **multiple-price** auction, where each winning bidder pays the price implied by *their own* bid. That sounds fair, but it has a perverse effect: it punishes you for bidding aggressively. If you bid a low yield (high price) to make sure you win, you end up paying that high price even if everyone else got the bonds cheaper. Knowing this, bidders *shade* their bids — they bid less aggressively to avoid overpaying, which means weaker demand and a worse deal for the government. This is the "winner's curse."

The **single-price** (uniform-price, or Dutch) auction the Treasury actually uses fixes this. Everyone who wins pays the *same* clearing yield — the stop-out yield of the last accepted bid. Because you know you will pay the clearing price no matter what, you can bid your *true* value without fear of overpaying. Honest, aggressive bidding becomes the dominant strategy, demand is stronger, and the government borrows more cheaply. The US switched to single-price auctions for all maturities in the late 1990s after experiments showed it lowered borrowing costs. It is a beautiful example of how the *rules* of a market shape the prices that come out of it.

### What a "tail" tells you

Traders watch one more number obsessively: the **tail**. Just before an auction closes, the not-yet-issued bond ("when-issued") trades in the market at some yield — the market's best guess of where the auction will clear. The tail is the gap between that pre-auction yield and the actual stop-out yield. A *positive* tail (the auction cleared at a *higher* yield than expected) means the government had to pay up — demand was softer than the market thought, and yields across the curve often jump in the seconds after. A *through* result (cleared at a *lower* yield than expected, a negative tail) signals hunger and tends to rally bonds. The auction is, in effect, a giant real-money referendum held dozens of times a month on how much the world wants to lend to the United States — and the tail is the margin of the vote.

## On-the-run versus off-the-run

Here is a subtlety that reveals how much the market values *liquidity* — the ease of trading large size without moving the price. Treasuries of the same maturity, issued by the same government, with nearly identical cash flows, can trade at slightly different yields depending purely on how *recently* they were issued.

![A before-and-after comparison of an on-the-run Treasury versus an off-the-run Treasury and the small yield gap between them](/imgs/blogs/us-treasuries-the-risk-free-benchmark-of-the-world-5.png)

The **on-the-run** issue is the most recently auctioned security of a given maturity — today's 10-year note, for example. It is where almost all the trading volume concentrates: it is the bond dealers quote, the one used as a benchmark, the one easiest to buy or sell \$1 billion of in seconds. Once the *next* 10-year is auctioned a few weeks later, the old one becomes **off-the-run** — still a perfectly good 10-ish-year Treasury, but now slightly less traded.

Because traders pay a premium for the liquidity of the on-the-run bond, it trades at a slightly *higher* price and therefore a slightly *lower* yield than its nearly-identical off-the-run cousin. The gap — the **on-the-run/off-the-run spread** — is usually just a few basis points, but it is real, and it is one of the purest measures of how much the market is willing to pay for liquidity at any moment. The spread *widens* in a crisis, when everyone crowds into the single most-liquid bond and abandons the rest.

#### Worked example: the cost of liquidity in basis points

Two Treasuries both mature in about ten years. The **on-the-run** 10-year yields **4.50%**. A nearly identical **off-the-run** note — issued three months earlier, maturing a quarter-year sooner — yields **4.54%**. The gap is **4 basis points**.

What does 4 bps mean in dollars? On a \$1,000 par note with roughly 9 years of *duration* (a measure of price sensitivity we cover in [duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income)), 4 bps of extra yield is worth roughly \$1,000 × 9 × 0.0004 ≈ **\$3.60** of price. So the off-the-run note is about \$3.60 *cheaper* per \$1,000 than the on-the-run — that is what you save (and the liquidity you give up) by owning the slightly older bond. A hedge fund running a "stub" trade might buy the cheap off-the-run and short the rich on-the-run, betting the gap closes — a classic *relative value* trade. (When such trades go wrong at scale, they can blow up; the 1998 collapse of Long-Term Capital Management was, in part, this exact trade gone wrong.)

*Intuition: two bonds with the same issuer and almost the same cash flows can still trade apart, and the gap is the market's live price tag on liquidity itself.*

## The 10-year as the benchmark for everything

Now we arrive at the heart of the post — and the reason this series keeps coming back to Treasuries. The **10-year Treasury yield** is not just one number among many. It is the reference rate off which an astonishing share of global finance is priced. Almost every other interest rate in the world is quoted, explicitly or implicitly, as "the 10-year, plus a spread."

![An influence graph showing the 10-year Treasury yield as the global benchmark that mortgages corporate bonds emerging-market debt and stock valuations all price off](/imgs/blogs/us-treasuries-the-risk-free-benchmark-of-the-world-3.png)

Trace the fan-out:

- **Mortgages.** The 30-year fixed US mortgage rate tracks the 10-year Treasury yield plus a spread (typically 1.5–3 percentage points) for prepayment and credit risk. When the 10-year rises 50 bps, mortgage rates usually rise nearly as much, and a generation of homebuyers feels it in their monthly payment.
- **Corporate bonds.** A company like Northwind Corp does not borrow at the Treasury rate — it borrows at the Treasury rate *plus a credit spread* that compensates lenders for its default risk. An investment-grade firm might pay the 10-year + 1.2%; a junk-rated one + 5% or more. The Treasury yield is the floor; the spread is the risk premium. (See [credit spreads, pricing the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default).)
- **Emerging-market debt.** When Brazil or Indonesia borrows in dollars, it pays the US Treasury yield plus a sovereign spread. A rising 10-year tightens financial conditions for every dollar-borrower on the planet, which is why a Fed-driven move in US yields can trigger crises thousands of miles away.
- **Swaps and derivatives.** Interest-rate swaps, the multi-hundred-trillion-dollar market that lets institutions trade fixed for floating rates, are quoted relative to Treasury yields.
- **Stocks.** Even equities are priced off the 10-year. A stock is worth the present value of its future earnings, and you discount those earnings using a rate built on the 10-year. When the 10-year rises, the discount rate rises, and the present value of far-off earnings falls — which is why high-growth tech stocks (whose value is mostly far-future) sell off hardest when Treasury yields jump. (See [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything).)

This is what "risk-free benchmark" really means in practice. The Treasury yield is the *baseline cost of money* for the safest borrower, and everyone else's cost of money is measured as a markup over that baseline. Change the baseline and you change every price layered on top of it.

The reason the 10-year in particular became *the* benchmark — rather than the 2-year or the 30-year — is partly habit and partly fit. Ten years is long enough to reflect the market's view of the economy over a full cycle (not just the next Fed meeting), yet short enough to stay deeply liquid and to roughly match the duration of a 30-year mortgage after you account for prepayments. It sits in the sweet spot between "what the Fed is doing right now" (the 2-year's job) and "what happens over a generation" (the 30-year's job). So when commentators say "rates went up today," they almost always mean the 10-year — the bond the whole market has agreed to treat as the pulse of the cost of money.

#### Worked example: how a 10-year move hits a homebuyer

A family is shopping for a \$400,000 home with a 20% down payment, so they need a \$320,000 30-year fixed mortgage. The 10-year Treasury yields **4.00%**, and the prevailing mortgage spread is about **2.5%**, so the mortgage rate is roughly 6.5%. At 6.5% on \$320,000, the monthly principal-and-interest payment is about **\$2,022**. Now the 10-year jumps to **5.00%** — a full percentage point — over a few months on stronger inflation data. The mortgage rate rises with it to about 7.5%, and the same \$320,000 loan now costs about **\$2,237** a month. That is **\$215 more every month**, or about \$2,580 a year, and roughly \$77,000 more over the life of the loan — all because the world's risk-free benchmark moved one point. The family's credit did not change; the house did not change; the benchmark did. This is the transmission belt from a number on a Bloomberg screen to a family's kitchen-table budget.

*Intuition: the 10-year Treasury yield is the gravitational field of finance — when it shifts, every borrower on earth, right down to a single household, is pulled along with it.*

### The deepest, most liquid market on earth

One more property cements the Treasury's role as the benchmark: sheer **depth**. A market is "deep" when you can trade enormous size without moving the price much. The US Treasury market is the deepest market that has ever existed. Around \$28 trillion of marketable debt is outstanding, daily trading volume runs into the hundreds of billions of dollars, and the bid-ask spread on the on-the-run 10-year — the gap between the price you can buy at and the price you can sell at — is often a fraction of a single basis point. You can sell \$1 billion of Treasuries in seconds at a price within a hair of the last trade, because there is *always* a buyer.

Depth is not a luxury; it is *why* Treasuries can serve as the benchmark at all. A reference rate is only useful if it is set by a continuous, liquid, hard-to-manipulate market. A thin market jumps around on a single large trade, and its "price" means little. The Treasury market's depth means its yield is a genuine, real-time consensus of millions of participants — exactly the property you want in the number you are going to price the rest of the world against. It is also why a *threat* to that depth (the 2020 dash-for-cash, when even Treasuries briefly stopped trading smoothly) terrifies policymakers: if the deepest market on earth seizes, there is no deeper one to flee to.

#### Worked example: pricing Northwind Corp off the benchmark

Northwind Corp wants to issue a 10-year bond. The 10-year Treasury yields **4.50%**. Northwind is investment-grade but not pristine, so investors demand a **credit spread of 1.50%** (150 bps) to lend to it instead of the government. Northwind's borrowing cost is:

$$ \text{Northwind yield} = \text{Treasury yield} + \text{credit spread} = 4.50\% + 1.50\% = 6.00\% $$

To raise \$100 million, Northwind must offer roughly a 6% coupon. Now suppose the Fed surprises the market and the 10-year jumps to **5.00%** while Northwind's credit is unchanged, so its spread stays 150 bps. Northwind's new cost is 5.00% + 1.50% = **6.50%** — half a percentage point more, or \$500,000 a year extra on \$100 million, *purely* because the benchmark moved. Northwind did nothing; the world's risk-free rate rose, and Northwind's cost rose with it.

*Intuition: a company's borrowing cost is the Treasury benchmark plus a tax for its own risk — the benchmark moves the floor, and every borrower on earth rides up and down with it.*

## Who owns Treasuries — and why the market is so hard to break

If the world prices everything off Treasuries, who actually *holds* the roughly \$28 trillion of marketable Treasury debt outstanding (as of early 2025)? The answer is "almost everyone, in pieces," and that diffusion is exactly why the market is so resilient: no single holder is big enough to break it.

Roughly speaking, the holders break into three blocks. **The Federal Reserve** holds a large slice (around \$4–5 trillion in early 2025, down from a 2022 peak as it shrinks its balance sheet via "quantitative tightening"). **Foreign holders** — central banks and investors in Japan, China, the UK, and dozens of other countries — hold roughly a quarter, parking their dollar reserves in the safest dollar asset there is. And a broad spread of **domestic holders** — banks, insurers, pension funds, mutual funds, money-market funds, and households — holds the rest. (The companion post [who buys bonds, the global demand for safe income](/blog/trading/fixed-income/who-buys-bonds-the-global-demand-for-safe-income) details each buyer's structural reason.)

This breadth is the market's superpower. Because demand comes from so many independent pools — a Japanese pension fund, a US bank's liquidity buffer, the People's Bank of China's reserves, a retiree's money-market fund — no single seller can overwhelm it, and the government rarely struggles to find lenders. It is the opposite of a fragile market dependent on one whale.

It also helps explain why fears that one country "could dump Treasuries and crash the US" are usually overblown. Suppose a large foreign holder owned roughly \$800 billion of Treasuries and decided to sell it all. That sounds catastrophic, but it is a few days of normal trading volume in a \$28 trillion market, and the seller would crater the *price* of its own holdings on the way out — punishing itself as much as anyone. More to the point, the dollars it received would have to go *somewhere*, and there is no other market large, safe, and liquid enough to absorb them; in practice the cash tends to flow right back into other dollar assets, often Treasuries again. The depth and breadth of the holder base is precisely what makes the market hard to weaponize. That said, a *gradual* decline in foreign appetite is real and matters: it shifts more of the financing burden onto price-sensitive domestic buyers, which can nudge yields up over time even if no single dramatic "dump" ever occurs.

#### Worked example: how a reserve manager uses Treasuries

Imagine the central bank of a mid-sized exporting country has accumulated **\$200 billion** of foreign-exchange reserves from years of trade surpluses. It must hold those reserves in *something* — and the requirements are brutal: the asset must be safe (it is the nation's rainy-day fund), liquid (it must be sellable instantly in a crisis to defend the currency), and able to absorb \$200 billion without moving the price. Only one market on earth checks all three boxes at that size: US Treasuries. So the reserve manager buys, say, \$150 billion of Treasuries — a mix of bills for liquidity and notes for a bit more yield. That single decision, multiplied across every surplus country, is why foreign official holders own trillions of Treasuries and why a US deficit can be financed by savings generated on the other side of the planet.

*Intuition: the world's giant pools of safe money have nowhere else big, safe, and liquid enough to go, so they keep buying Treasuries — and that structural demand is what makes the market unbreakable.*

## Treasuries as collateral: the repo plumbing

There is one more role Treasuries play that almost no beginner has heard of, yet it may be the most important of all: Treasuries are the **collateral** that lubricates the entire financial system. This happens in the **repo market** (short for *repurchase agreement*), the plumbing that moves trillions of dollars of overnight cash every single day.

![A pipeline of a repo transaction where a borrower pledges Treasuries as collateral for overnight cash and buys them back the next day](/imgs/blogs/us-treasuries-the-risk-free-benchmark-of-the-world-6.png)

A **repo** is a collateralized loan dressed up as a sale. Here is the mechanism, step by step:

1. A borrower (say a hedge fund or dealer) owns Treasuries but needs cash overnight.
2. It *sells* those Treasuries to a lender (a money-market fund, say) for cash today, and simultaneously agrees to *buy them back* tomorrow at a slightly higher price.
3. The tiny price difference is the interest on the loan — the **repo rate.**
4. If the borrower fails to repay, the lender simply keeps the Treasuries it already holds. The loan is fully *secured* by the collateral.

Why Treasuries? Because the lender must be certain the collateral will hold its value and be sellable if the borrower defaults. Treasuries are the only collateral that everyone, everywhere, accepts without analysis — they are *information-insensitive*. A money-market fund will lend \$1 billion overnight against Treasuries without blinking, because if the borrower vanishes, the fund holds the safest, most liquid asset on earth. This is why Treasuries are not just an investment but the *foundation* of the short-term funding markets. Trillions of dollars of daily borrowing rest on them.

#### Worked example: a one-day repo against Treasuries

A dealer needs **\$100 million** in cash overnight and pledges **\$100 million** of Treasuries as collateral. The overnight repo rate is **5.00%** annualized. The dealer sells the Treasuries for \$100 million today and agrees to buy them back tomorrow for:

$$ \text{repurchase price} = \$100{,}000{,}000 \times \left(1 + 0.05 \times \frac{1}{360}\right) = \$100{,}013{,}889 $$

So the dealer pays back **\$100,013,889** the next day — the original \$100 million plus **\$13,889** of one-night interest. In practice the lender also demands a small **haircut** — lending, say, \$99.5 million against \$100 million of collateral — so it is over-collateralized in case Treasury prices wobble overnight. The haircut is the lender's safety margin: even a default leaves it whole.

*Intuition: a repo turns a Treasury into cash for a night without selling it, and the whole arrangement works only because the collateral is the one asset no lender has to think twice about.*

## "Risk-free" has limits: rate risk and the debt ceiling

It would be dishonest to end without the caveats, because "risk-free" is the most misunderstood phrase in finance. Treasuries are free of *default* risk — the risk you simply do not get paid. They are *not* free of every risk.

**Rate risk (price risk).** If you buy a 10-year Treasury at a 1.5% yield and rates then rise to 4.5%, your bond is now worth far less than you paid — anyone can buy a new bond paying 4.5%, so no one will pay full price for your 1.5% one. You will get every promised payment if you hold to maturity, but if you must sell early, you take a loss. This is not a hypothetical. It is the core of the **Silicon Valley Bank** collapse in March 2023: SVB had loaded up on long Treasuries and agency bonds when yields were near zero, the Fed hiked aggressively, those "safe" bonds fell in market value, and when depositors pulled cash and the bank had to sell at a loss, it was insolvent. The bonds never defaulted. They simply fell in price. *Risk-free of default is not risk-free of loss.* (See [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**The debt ceiling.** The US Congress periodically caps how much the Treasury may borrow, and reaching that cap without raising it would force the government to choose which bills to pay — potentially missing a Treasury payment. This would be a *self-inflicted, technical* default: not because the US lacks the money, but because of a political impasse. Markets have repeatedly priced a small but nonzero chance of this during standoffs (2011, 2013, 2023), pushing the yields on bills maturing right around the deadline visibly higher. In 2011, the standoff led S&P to strip the US of its top AAA rating — the first time in history. The benchmark is sturdy, but it rests on a political assumption that the US *chooses* to pay.

**Inflation risk.** A regular (non-TIPS) Treasury pays you back in dollars, and inflation erodes what those dollars buy. Get \$10,000 back in ten years and, if prices doubled, it buys what \$5,000 buys today. You did not default — you were simply repaid in weaker money. This is the risk TIPS exist to neutralize.

#### Worked example: the rate-risk loss that sank a bond, not a default

You buy a 10-year Treasury at par — \$1,000 for \$1,000 — when it yields **1.5%**, paying a \$15 annual coupon. Two years later, the Fed has hiked hard and new 10-year Treasuries yield **4.5%**. Why would anyone pay you \$1,000 for a bond paying \$15 a year when they can buy a fresh one paying \$45 a year? They will not. Your bond's price falls until *its* yield matches the market — roughly to the low **\$800s** for a bond with about 8 years left and that yield gap (a back-of-envelope drop of roughly 8 years of duration × 3% ≈ 24% of value). You have lost about **\$200 per \$1,000**, on paper, on the "risk-free" asset. If you hold to maturity you still get every \$15 coupon and your \$1,000 back, and your loss evaporates. But if you are a bank that must sell to meet deposit withdrawals — as Silicon Valley Bank was in 2023 — that paper loss becomes a real, fatal one. The bond never missed a payment. It just fell in price.

*Intuition: a Treasury's "risk-free" promise is only that it will pay you in full if you wait — it says nothing about what it is worth if you are forced to sell into higher rates.*

## Common misconceptions

**"Risk-free means you can't lose money."** The single most common error. Treasuries are free of *default* risk only. They carry real *price* risk if rates rise and you sell before maturity (the SVB story), and real *inflation* risk because they repay in nominal dollars. "Risk-free" is a statement about getting paid, not about never losing.

**"T-bills pay a coupon like other bonds."** They do not. Bills have no coupon at all — they are sold at a discount and redeemed at face value, and the gap *is* the interest. If you see a "rate" on a bill, check whether it is the misleading discount rate (divides by face, 360-day year) or the honest investment yield (divides by price, 365-day year); the latter is always a touch higher.

**"The Fed sets the 10-year Treasury yield."** The Fed sets the *overnight* policy rate. The 10-year yield is set by the market — by millions of buyers and sellers pricing in expected future short rates, growth, inflation, and a term premium. The Fed *influences* the 10-year (and can buy it directly via QE), but it does not set it. The 10-year often moves *against* the Fed: it can fall while the Fed hikes if the market expects the hikes to slow the economy. (See [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).)

**"On-the-run and off-the-run Treasuries are the same thing, so they must trade at the same yield."** Nearly the same — but not quite. The on-the-run issue commands a small liquidity premium, so it yields a few basis points *less* than its off-the-run twin. That gap is one of the cleanest live measures of how much the market values liquidity, and it blows out in a crisis.

**"A government that prints its own money can never have a problem selling debt."** Default risk is near zero, yes — but *price* still matters. If the world demands a higher yield (weak auctions, a soft bid-to-cover, a "tail"), the government must pay more to borrow, and those higher yields ripple into every mortgage and corporate loan. The constraint is not "can it sell?" but "at what price?" — and that price is set by the same auction demand we measured with the bid-to-cover ratio.

**"Treasuries are an investment, full stop."** They are also the financial system's *plumbing*. Treasuries are the universal collateral behind the trillions-of-dollars-a-day repo market, the high-quality liquid assets banks must hold by law, and the reserves central banks stockpile. A Treasury is doing three jobs at once: an investment, a piece of collateral, and a store of safety. That triple role is why a disruption in Treasuries is so dangerous — it is not one market breaking, it is the floor under all of them.

## How it shows up in real markets

**The 2020 "dash for cash."** In March 2020, as the pandemic hit, even *Treasuries* — the supposed safe haven — briefly sold off as everyone, everywhere, scrambled for actual cash dollars and dumped whatever they could sell, including Treasuries. The world's deepest market seized up; bid-ask spreads widened and the on-the-run/off-the-run gap blew out. The Federal Reserve had to step in and buy roughly a trillion dollars of Treasuries in a matter of weeks to restore order. The lesson: even the risk-free benchmark depends on functioning *liquidity*, and in a true panic, "I need cash now" can overwhelm "this is the safest asset."

**Silicon Valley Bank, March 2023.** SVB held a large book of long-dated Treasuries and agency mortgage bonds bought when yields were near zero. As the Fed hiked the policy rate from ~0% to ~5% across 2022–23, the market value of those long bonds fell sharply (long duration means high price sensitivity). When depositors fled and SVB had to sell, it crystallized billions in losses on assets that had *zero* default risk. The bank failed in days. It is the cleanest real-world proof that "risk-free" means default-free, not loss-free.

**The 2011 debt-ceiling downgrade.** A political standoff over raising the borrowing cap pushed the US to the brink of a technical default in August 2011. Standard & Poor's responded by cutting the US credit rating from AAA to AA+ — the first downgrade in the nation's history. Paradoxically, in the chaos that followed, investors *bought* Treasuries (yields fell), because in a global scare there is still nowhere safer to hide — even from a fear about Treasuries themselves. The episode showed both the political fragility of the "risk-free" label and the world's stubborn reliance on it.

**Foreign reserve accumulation and the "savings glut."** Through the 2000s and 2010s, surplus countries — China above all, plus oil exporters and other Asian economies — accumulated trillions of dollars of reserves and parked them in US Treasuries. This vast, price-insensitive demand pushed Treasury yields lower than US economic conditions alone would have set them, which (former Fed chair Ben Bernanke argued) helped keep US mortgage rates low and inflated the mid-2000s housing bubble. It is the most vivid illustration of the benchmark's global role: savings generated in Asia, recycled into Treasuries, set the cost of a mortgage in Arizona.

**Long-Term Capital Management, 1998.** The famous hedge fund, run by Nobel laureates, ran (among many trades) the on-the-run/off-the-run convergence trade — long the cheap off-the-run bond, short the rich on-the-run — betting the tiny liquidity gap would close. When Russia defaulted in August 1998 and the world fled to the *most* liquid assets, the gap *widened* instead of closing, the leveraged trade hemorrhaged, and the Fed had to organize a bailout to prevent contagion. The lesson: even a trade on the world's safest asset can blow up when liquidity premiums, not credit, move against you.

## When this matters to you

If you have ever taken out a mortgage, your monthly payment was set, in large part, by the 10-year Treasury yield on the day you locked your rate. If you own a stock fund, its value swings when Treasury yields move, because those yields are the discount rate baked into every valuation. If you hold cash in a money-market fund, it is almost certainly earning its return by lending against Treasuries in the repo market. The risk-free benchmark is not an abstraction for traders — it is the gravitational center of the financial world, and it touches nearly every dollar you save, borrow, or invest.

To go deeper from here: see [the yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) for how Treasuries of every maturity line up into the most-watched chart in finance; [who buys bonds](/blog/trading/fixed-income/who-buys-bonds-the-global-demand-for-safe-income) for the structural demand that keeps the market deep; [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) for how the whole edifice connects to the economy; and [government bonds, the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) for how an allocator uses Treasuries to anchor a portfolio. None of this is financial advice — it is the map of how the benchmark works, so the next time you see "the 10-year," you know exactly what it is moving.
