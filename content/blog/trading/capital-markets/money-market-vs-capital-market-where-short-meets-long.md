---
title: "Money Market vs Capital Market: Where Short Meets Long"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "The maturity divide that organizes all of finance — how the under-one-year money market and the over-one-year capital market work, why borrowing short to invest long is both the engine and the fragility of the system, and what happens when the money market breaks."
tags: ["capital-markets", "money-market", "treasury-bills", "repo", "commercial-paper", "maturity-transformation", "money-market-funds", "yield-curve", "liquidity", "fixed-income"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — One number, the maturity of a claim, splits all of finance into two markets: a **money market** for everything maturing in a year or less, and a **capital market** for everything longer. Short funds long, and that single act is both the engine of the economy and its most reliable fault line.
>
> - The **money market** (Treasury bills, commercial paper, repo, CDs, fed funds) is huge, near-cash, and low-risk — a place to *park* money, not to *grow* it. The US Treasury alone issued roughly \$23 trillion of bills and short paper in 2023.
> - The **capital market** (stocks and bonds maturing past a year) is where savings turn into long-lived real investment: factories, rail, software, houses.
> - The two are joined by **maturity transformation** — banks and funds borrow short and invest long, capturing the term spread but inheriting **run risk**. **Repo** literally finances capital-market bonds with money-market cash, so the markets are physically wired together.
> - When the money market breaks — Reserve Primary "breaking the buck" in 2008, the "dash for cash" in March 2020 — the damage doesn't stay in the money market. It freezes the funding that holds up the long end too.

On the morning of September 16, 2008, a phone call went out from a fund called The Reserve Primary Fund that should have been impossible. The fund managed about \$62 billion in what every investor on earth treated as a cash equivalent — a money-market fund, the kind of thing a corporate treasurer sweeps payroll into overnight, the kind of thing your brokerage account holds as "cash." Its entire promise was a fixed share price of exactly one dollar. Put in a dollar, take out a dollar, earn a little yield in between. The fund had held that dollar through every storm since 1971.

That morning it announced its shares were worth \$0.97. The fund had been holding \$785 million of short-term IOUs issued by Lehman Brothers — commercial paper, the workhorse of corporate borrowing — and Lehman had filed for bankruptcy the day before. The paper was suddenly worth nothing. The fund had "broken the buck." Within days, investors yanked roughly \$300 billion out of prime money-market funds, the commercial-paper market that finances payrolls and inventories across the real economy seized up, and the US Treasury had to extend a temporary guarantee over the entire \$3.4 trillion money-fund industry to stop the bleeding.

Nothing about that crisis was, on paper, about stocks or long-term bonds. It was about the *short* end — instruments maturing in days or weeks. And yet it nearly took the whole financial system with it. To understand why, you have to understand the single dividing line that organizes everything: **maturity**. How long until you get your money back? Everything under about a year lives in the **money market**; everything longer lives in the **capital market**. This post is about that line — where it sits, why finance is built around it, how the two sides are wired together, and what happens when the short side cracks.

![Maturity spectrum from overnight repo to perpetual stock](/imgs/blogs/money-market-vs-capital-market-where-short-meets-long-1.png)

## Foundations: maturity is the axis that organizes all of finance

Start with the most everyday version of the idea. You have money you won't need for a while, and someone else needs money now and will pay to use it. That is the whole of finance in one sentence. But there is a crucial question buried in "for a while": *how long?* The answer to that one question — the **maturity** — changes everything about the deal.

If you lend money for one night, you barely care about anything. The borrower can't go bankrupt and the world can't change much between tonight and tomorrow morning, so you'll accept a tiny return and you'll want your money back, in full, at dawn. If you lend money for thirty years, you care about everything: whether inflation eats your return, whether the borrower is still solvent in 2056, whether you'll wish you'd done something else with the cash. You demand a much bigger return, and you accept that you can't easily get the money back early.

That difference is so fundamental that the financial system splits cleanly in two around it:

- The **money market** is the market for borrowing and lending for **one year or less**. Its instruments are near-cash: very safe, very liquid (easy to sell without moving the price), and very low-yielding. You use it to *store* value safely for a short time, not to grow wealth.
- The **capital market** is the market for **maturities greater than one year**, all the way out to *perpetual* (stock, which never matures). Its instruments — bonds, notes, and shares — fund long-lived real things and carry more risk and more return. You use it to *grow* wealth and to *fund* multi-year projects.

A quick vocabulary primer, because the rest of this post leans on it:

- A **security** is a tradable financial contract — a standardized claim you can buy and sell. A Treasury bill, a corporate bond, and a share of stock are all securities.
- **Maturity** is the date the borrower must repay the principal. A 13-week T-bill matures in 13 weeks; a 10-year note in 10 years; a stock never matures (it's *perpetual*).
- **Liquidity** is how easily you can turn the security back into cash at a fair price. Money-market instruments are extremely liquid; a thinly traded 30-year municipal bond is not.
- **Yield** is the return you earn, expressed as an annual percentage. Short, safe instruments yield little; long or risky ones yield more.
- The **primary market** is where securities are *created* and sold for the first time (a company issues new bonds; the Treasury auctions new bills). The **secondary market** is where those existing securities then *trade* between investors. This whole series turns on one fact: secondary-market liquidity — the ability to sell tomorrow — is what makes anyone willing to buy in the primary market today.

Hold that last point. It's the spine of the entire "Capital Markets" series, and it's exactly why the money market matters so much. We unpack the full machine in [what a capital market is and how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use). Here we zoom into the single seam that runs down the middle of it.

Why one year, specifically? The line is partly convention and partly mechanical. One year is roughly the horizon over which a holder treats an instrument as a cash-management tool rather than an investment — within a year you're managing liquidity, beyond it you're taking a view on the future. It also lines up with accounting (current versus non-current assets), with regulation (the US 270-day ceiling on unregistered commercial paper, the one-year cutoff in many fund rules), and with how risk scales: over a few months, the dominant question is "will I get my cash back," answerable from the issuer's near-term solvency; over many years, inflation, rate moves, and structural change dominate, and those require a genuine forecast. The boundary isn't a law of nature, but it's not arbitrary either — it's where the *character* of the risk you're taking changes from liquidity risk to investment risk.

The *"so what?"* of this section: maturity isn't a footnote on a security — it is the property that determines which market the security lives in, who buys it, how risky it is, and how it behaves in a crisis. Get the maturity, and you can predict almost everything else.

## The money market: a parking lot, not an engine

The money market is where the financial system keeps its cash overnight. It is enormous, boring by design, and absolutely essential. Picture it less as a place to invest and more as a high-security parking lot: you don't drive to a parking lot to make money, you drive there because you need somewhere safe to leave the car for a few hours.

### What's parked there

Five instruments do most of the work. Each is just a short-term IOU with a slightly different issuer and structure.

**Treasury bills (T-bills).** Short-term debt of the US government, issued in maturities of 4, 8, 13, 17, 26, and 52 weeks. They pay no coupon; instead you buy them at a **discount** to their face value and collect the full face value at maturity. The gap is your return. T-bills are the closest thing on earth to risk-free, which is exactly why they anchor the whole money market — every other short rate is quoted relative to them. The US government issues them in staggering size to smooth out the timing mismatch between when taxes arrive (lumpy, around filing deadlines) and when the government spends (steady, every day).

**Commercial paper (CP).** Short-term unsecured IOUs issued by large, creditworthy corporations and financial firms to fund day-to-day needs — payroll, inventory, receivables. Maturities run from a few days up to 270 days (the 270-day ceiling lets issuers skip full securities registration in the US). CP is cheaper than a bank loan for a blue-chip borrower, which is why a company like Toyota or Coca-Cola would rather roll CP than draw a credit line. It is *unsecured*, though — there's no collateral behind it — so when the issuer's credit is in doubt, CP is the first thing to freeze. That's precisely what happened to Lehman's paper in 2008.

**Repurchase agreements (repo).** A repo is a loan dressed up as a sale. One party sells a security (usually a Treasury) today and agrees to buy it back tomorrow at a slightly higher price; the buyer is really making a secured overnight loan, with the security as collateral. Repo is the plumbing that lets bond dealers and hedge funds *finance* the bonds they hold — and it's the single most important wire connecting the money market to the capital market, which we'll return to in detail.

**Certificates of deposit (CDs).** A time deposit at a bank: you agree to leave a fixed sum for a fixed term (say 90 days) in exchange for a fixed rate, and you can't withdraw early without a penalty. Large-denomination *negotiable* CDs can themselves be traded, which makes them a money-market instrument rather than just a savings product.

**Federal funds (fed funds).** The market where US banks lend their reserve balances at the Federal Reserve to each other, overnight, unsecured. The interest rate on these loans — the **fed funds rate** — is the rate the Fed targets when it "sets interest rates," and it cascades into every other short rate in the economy. (The mechanics of how the Fed actually steers this rate are macro-policy territory; we link out to the macro-trading treatment of policy and liquidity rather than re-derive them here.)

### Who uses it, and why it's so big

The money market exists because three kinds of players have a structural need to manage cash over very short horizons:

- **Corporate treasurers** have cash that's spoken for but not yet spent — next month's payroll, a tax payment due in 60 days, the proceeds of a bond sale waiting to be deployed. They cannot put it in stocks (too risky for money they'll need imminently) or in a checking account (earns nothing). The money market is the answer.
- **Banks and dealers** need to fund their balance sheets day to day. A bond dealer holding \$10 billion of Treasuries finances most of that position in the repo market overnight, rolling it every single morning. A bank that ends the day short of reserves borrows the gap from a bank that ends long, overnight, in the fed funds market — the inter-bank plumbing that lets the whole banking system settle each evening without anyone holding idle cash. For these players the money market isn't an investment at all; it's the funding that keeps their much larger capital-market books alive.
- **Governments** smooth the mismatch between lumpy tax receipts and steady spending by issuing bills. The Treasury is by far the largest single participant.

These motives share a shape: nobody is in the money market to get rich. They're there because they have cash that's spoken for and want it safe and available, or because they need short-term funding to carry a long-term position. That's why the money market clears at razor-thin spreads and trades in enormous size — when your goal is "don't lose it and let me have it back Tuesday," you'll accept almost no yield, and the market obliges. The flip side is that the money market is exquisitely sensitive to *trust*: the moment a lender doubts they'll get their cash back on time, the near-zero spread that made the market work becomes a near-infinite spread (no rate is high enough to lend at), and the instrument freezes. There's no middle gear. A market built on "perfectly safe" has only two states — open and shut.

How big is "big"? Look at gross US bond issuance in 2023, broken out by type:

![US bond issuance by type in 2023 with Treasury dominant](/imgs/blogs/money-market-vs-capital-market-where-short-meets-long-2.png)

The Treasury's roughly \$23 trillion of issuance — most of it short-dated bills rolled over and over — dwarfs every other category combined. That single bar is the money market's gravitational center. Corporate bonds, the headline of the capital market, are about \$1.4 trillion; the money market's churn is an order of magnitude larger because short paper *turns over* constantly. A 13-week bill issued and repaid four times a year shows up as four times its face value in annual issuance. This is the first thing that surprises people about the money market: by the *flow* of issuance it is far larger than the capital market, even though by the *stock* of outstanding wealth the capital market wins. The money market is a river; the capital market is a lake.

### The money-market fund: how the money market reaches you

Most individuals never buy a Treasury bill or a piece of commercial paper directly — the lot sizes are institutional, often \$1 million minimums. Instead they reach the money market through a **money-market fund (MMF)**: a pooled vehicle that buys a diversified basket of money-market instruments and sells you shares, each priced (by long convention) at a stable \$1.00. You park \$10,000, you get 10,000 shares; the fund earns the short rate on its T-bills, CP, and repo, passes most of it to you as yield, and you can redeem any day at \$1.00 a share. It feels exactly like a high-yield checking account, which is the whole point — and also the whole danger, because that \$1.00 is a *convention*, not a guarantee.

There are two flavors, and the distinction is load-bearing in a crisis:

- **Government MMFs** hold only Treasury bills and government repo. They are about as safe as the money market gets, because the underlying paper has no real credit risk. After 2008, these became the default place to hide.
- **Prime MMFs** hold corporate commercial paper and CDs alongside government paper, reaching for a few extra basis points of yield. That reach is precisely what blew up Reserve Primary in 2008 — its prime holdings included Lehman's CP. The yield pickup is small in good times and the loss is total in bad times.

The MMF is the retail-facing edge of the money market, and it's why a panic that starts in obscure institutional paper can drain a schoolteacher's "cash" account in 48 hours. We'll see exactly that mechanism play out twice in the case studies.

### How discount yields actually work

The one piece of money-market math worth getting right is the **discount yield**, because T-bills and most CP don't pay interest the way a bond does. They're sold cheap and redeemed at face value, and there are two different ways the market quotes the return — which trips up almost everyone the first time.

#### Worked example: a 13-week T-bill bought at a discount

Suppose you buy a 13-week (91-day) Treasury bill with a face value of \$1,000,000, and you pay \$987,800 for it. In 91 days the Treasury pays you the full \$1,000,000. Your dollar gain is:

```
gain = 1,000,000 - 987,800 = 12,200
```

There are two standard ways to express that as a yield.

The **bank discount yield** (the rate the bill is quoted at) uses the *face* value as the base and a 360-day year:

```
discount yield = (gain / face) * (360 / days)
              = (12,200 / 1,000,000) * (360 / 91)
              = 0.0122 * 3.9560
              = 0.04826  ->  4.83%
```

The **bond-equivalent yield** (what you actually earned, comparable to a coupon bond) uses the *price you paid* as the base and a 365-day year:

```
bond-equiv yield = (gain / price) * (365 / days)
                = (12,200 / 987,800) * (365 / 91)
                = 0.012351 * 4.0110
                = 0.04954  ->  4.95%
```

So the same bill is "a 4.83% bill" on the quote screen but earned you 4.95% in real, comparable terms. The discount-yield convention *understates* your true return, because it divides by the bigger number (face) and the shorter year (360 days). The lesson: always convert a discount quote to a bond-equivalent yield before comparing a bill to a bond, or you'll systematically think bills pay less than they do.

#### Worked example: a corporate treasurer parking \$50M for 90 days

Now the everyday use case. A treasurer at a mid-sized company has \$50,000,000 from a bond sale that won't be deployed for three months. Leaving it in a non-interest checking account earns \$0. Parking it in 90-day T-bills at a 4.95% bond-equivalent yield earns, for the quarter:

```
interest = 50,000,000 * 0.0495 * (90 / 365)
        = 50,000,000 * 0.0495 * 0.246575
        = 610,274
```

About \$610,000 for three months of doing nothing but choosing the right parking lot, with essentially zero credit risk. The lesson: the money market doesn't make a company rich, but on large balances the difference between "earning the short rate" and "earning nothing" is real money, and it's free.

The *"so what?"* of the money market: it's where the system holds its breath between transactions. Low risk, low yield, immense size, constant turnover. Nobody gets wealthy here — but the entire economy depends on it staying boring.

## The capital market: where savings become real things

Cross the one-year line and the character of the market flips. The **capital market** is for maturities greater than a year, and its job is fundamentally different: not to *store* value safely but to *channel* savings into long-lived productive investment and to let savers share in the returns. Two instrument families dominate.

**Bonds and notes (debt).** A loan in tradable form. The issuer promises to pay periodic coupons and return the principal at a maturity years out — a 5-year corporate note, a 10-year Treasury note, a 30-year mortgage bond. As a lender you have a *contractual* claim: you get paid before shareholders, but your upside is capped at the agreed interest. The pricing, duration, and curve mathematics of bonds belong to fixed income; we link out to [the yield-curve explainer](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) rather than re-derive them, because this series owns the *system*, not the instrument math.

**Stocks (equity).** A share of ownership with no maturity at all — it is *perpetual*. As a shareholder you have a *residual* claim: you get whatever is left after everyone else is paid, which means unlimited upside and the risk of being wiped out. Companies issue stock to raise permanent capital they never have to repay. The split between funding with debt versus equity is its own deep topic, covered in [debt vs equity: the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital).

The difference between the two claims is the most important thing to internalize about the capital market, because it determines who gets paid in what order. Stack a company's funding from safest to riskiest and you get the **capital structure**: senior secured debt at the bottom (paid first, lowest return), then unsecured bonds, then preferred stock, then common equity at the very top (paid last, highest potential return). When a company earns money, it flows *up* the stack — bondholders take their fixed coupon, and whatever is left belongs to shareholders. When a company fails, the losses flow *down* from the top — shareholders are wiped out first, then junior creditors, and senior secured lenders are made whole last if there's anything left. A bondholder is renting money to the company at a fixed price; a shareholder owns the leftover. That single distinction explains why a 10-year bond and a share in the *same* company behave so differently: the bond is a capped, contractual claim, while the share is an uncapped, residual one. The capital market exists to price and trade both kinds of claim, all along the maturity spectrum past one year.

Where do these securities come from? The capital market *creates* them in the **primary market** — the Treasury auctions a new 10-year note, a company sells \$500 million of fresh bonds through an underwriting syndicate, a startup goes public in an IPO. Then they trade, forever after, in the **secondary market**. The two are inseparable: an investor only buys a freshly issued 30-year bond because they're confident the secondary market will let them sell it next year if their plans change. That is the series spine again — secondary liquidity is the precondition for primary issuance — and it's why a deep, liquid secondary market is a country's single most valuable piece of financial infrastructure.

The capital market's whole reason to exist is the **time horizon mismatch** between savers and projects. A factory, a railway, a fiber network, a drug pipeline — these take years to pay off and can't be funded with overnight money. But almost no individual saver wants to lock cash away for thirty years with no escape. The capital market resolves this through the primary/secondary split: the project gets *permanent or long-term* funding via the primary market, while the individual saver keeps *liquidity* because the secondary market lets them sell their claim to someone else whenever they want. The project never has to repay early; the saver never has to wait thirty years. That trick — turning a stock of patient capital into something individually liquid — is the capital market's core magic.

![Global capital markets by size in trillions](/imgs/blogs/money-market-vs-capital-market-where-short-meets-long-5.png)

The scale is hard to overstate. Global equity market capitalization runs around \$115 trillion and the global bond market around \$140 trillion; the US alone accounts for roughly \$55 trillion of each. This is the accumulated *stock* of long-term claims on the world's productive assets — every public company, every government and corporate bond outstanding. And unlike the money market's constant turnover, this stock compounds. US equity market cap has more than doubled in a decade:

![US equity market cap by year-end in trillions](/imgs/blogs/money-market-vs-capital-market-where-short-meets-long-8.png)

That upward march is the visible result of savings being recycled into productive investment, year after year — the capital market doing its job. The money market's chart would be flat and boring (it's a parking lot, the level just reflects how much cash needs parking); the capital market's chart trends up and to the right, because it owns a claim on growth.

### Liquidity versus return: the trade you can't escape

The maturity divide is, underneath, a single trade-off that every saver makes whether they realize it or not: **liquidity versus return**. The more readily you can get your money back at a fair price, the less you earn; the longer you're willing to lock it up and bear risk, the more you're paid. The money market sits at the high-liquidity, low-return corner; the capital market sits at the low-liquidity, higher-return corner. The extra yield you earn for giving up liquidity and accepting time is the **term premium** (for taking maturity risk) plus the **credit and equity risk premia** (for taking default and ownership risk). These premia aren't a gift — they're compensation for real risks the money market spares you.

This is why "where should I put my money?" has no universal answer: it depends entirely on *when you'll need it back*. Money you need next month belongs in the money market, full stop — chasing yield with it is how treasurers and savers get caught when the short instrument they reached for turns out to be an unsecured loan to a wobbly borrower. Money you won't touch for a decade belongs in the capital market, because over that horizon the liquidity you sacrificed costs you nothing while the return you gave up by hiding in T-bills compounds into a fortune. The maturity of your *liability* (when you need the cash) should match the maturity of your *asset* (when it pays off). Mismatching them on purpose is maturity transformation; mismatching them by accident is how individuals blow up too.

The *"so what?"* of the capital market: it converts the public's appetite for *liquid* savings into the *patient* capital that long-term projects require. The money market preserves; the capital market grows. Different jobs, different risk, different math — and the price of crossing from one to the other is always paid in liquidity.

## Why the split matters: maturity transformation and its fragility

Here's where the two markets stop being two separate stories and become one. The most important thing the financial system does — and the most dangerous — is to stand *between* the two markets and convert one into the other. The technical name is **maturity transformation**: borrow short, lend or invest long.

A commercial bank is the textbook example. Your deposit is a money-market liability — you can demand it back any day, so its effective maturity is roughly zero. The bank turns around and uses it to fund a 30-year mortgage, a capital-market asset. The bank pockets the difference between the low short rate it pays you and the higher long rate the mortgage earns. That difference — the **term spread** — is, in large part, how banks make money. (The full mechanics of the bank balance sheet and lending are banking's domain; we link out to the banking series rather than reproduce them, because this series owns the *securities* side of the system.)

![Maturity transformation borrow short invest long with run risk](/imgs/blogs/money-market-vs-capital-market-where-short-meets-long-4.png)

Maturity transformation is genuinely productive. It lets the economy fund long-lived projects using the public's preference for liquid, short-term savings — squaring the exact circle we described above. Without it, nobody could get a 30-year mortgage, because no single saver would lend for 30 years. The intermediary stitches together thousands of short, impatient deposits into one long, patient loan.

Put numbers on how the intermediary actually earns its keep, because the economics are simpler than they sound.

#### Worked example: a bank's term spread on \$1B of deposits

A bank funds \$1,000,000,000 of deposits at a 1.5% deposit rate (a money-market liability — depositors can leave any day) and lends it out as 30-year mortgages at a 6.0% rate (a capital-market asset). Ignore credit losses and operating costs for the moment; the gross **net interest margin** is the difference between the two rates, earned on the whole balance:

```
interest earned on assets   = 1,000,000,000 * 0.060 = 60,000,000
interest paid on deposits   = 1,000,000,000 * 0.015 = 15,000,000
gross net interest income   = 60,000,000 - 15,000,000 = 45,000,000
```

That \$45 million a year — a 4.5% margin — is the term spread the bank captures purely for standing between short money and long assets. Now watch what an inverted curve does to it. If short rates rise to 5.5% (the deposit rate must follow, or depositors flee to a money fund) while the old mortgages still earn 6.0%, the margin collapses:

```
interest paid on deposits (new)  = 1,000,000,000 * 0.055 = 55,000,000
gross net interest income (new)  = 60,000,000 - 55,000,000 = 5,000,000
```

The same balance sheet that earned \$45 million now earns \$5 million — before costs, which can easily push it negative. The lesson: the bank's profit *is* the term spread, so a maturity-transforming institution is implicitly long the steepness of the yield curve, and an inversion quietly strangles it long before any depositor panics.

But notice the asymmetry baked into the structure. The liabilities can leave **fast** (depositors withdraw tomorrow; CP holders refuse to roll; repo lenders demand cash back) while the assets are **stuck** (you can't call a 30-year mortgage early). When everyone tries to get their short money back at once and the long assets can't be sold quickly enough to meet them, you get a **run**. The institution isn't necessarily insolvent — its assets may be worth more than its liabilities — it's *illiquid*: it has the value, just not in cash, today. That gap between *illiquid* and *insolvent* is where panics live.

A run has a vicious internal logic worth spelling out, because it's not irrationality — it's the opposite. Suppose an intermediary holds long assets that are *fundamentally* worth more than its short liabilities, but it can only sell those assets slowly without crashing their price. If you're a short-term lender and you believe everyone else will stay put, the rational move is to stay too; you'll get paid. But if you believe others might pull out, the rational move is to pull out *first*, because the people who redeem early get paid at full value and the stragglers get whatever's left after a fire sale. Every lender reasoning this way races for the exit, and the race itself causes the collapse it feared. This is why runs are *self-fulfilling*: the belief that a run might happen is sufficient to cause one, even at an institution that would have been perfectly solvent if everyone had simply held. Deposit insurance exists precisely to break this logic — if your \$100,000 is guaranteed, you have no reason to race anyone, so the run never starts. The money market's wholesale instruments (CP, repo, large CDs) are mostly *uninsured*, which is why the race dynamic is alive and well there.

The maturity mismatch also has a quieter, slower failure mode that doesn't require a panic at all: rising rates. If you funded long fixed-rate assets with short floating-rate liabilities, then when short rates climb you keep earning the old low yield on your 30-year bonds while paying the new high rate on your overnight funding. Your spread can flip negative and stay there. That's the *carry* version of the trap, distinct from the *run* version, and it's exactly what the inverted-curve chart below shows squeezing the system through 2022–2024.

#### Worked example: rolling \$50M of commercial paper

A finance company funds a portfolio of 5-year auto loans (capital-market assets) by issuing 30-day commercial paper (money-market liabilities). Say it has \$50,000,000 of CP outstanding at a 5.0% annualized rate. Every 30 days it must repay maturing paper and sell new paper to replace it. The monthly interest cost is small:

```
monthly interest = 50,000,000 * 0.05 * (30 / 360)
                = 50,000,000 * 0.05 * 0.08333
                = 208,333
```

About \$208,000 a month to fund \$50 million — cheap, as long as the paper *rolls*. But the entire \$50 million principal comes due every 30 days and must be refinanced. On a normal day, buyers happily roll it. On a bad day — a credit scare, a frozen market — buyers vanish, and the company must suddenly find \$50 million in cash it doesn't have, because the money is tied up in 5-year auto loans it can't sell overnight. The lesson: the interest cost of short funding is trivially small, which is exactly the seduction — the real cost is the rollover risk you can't see on the income statement until the day it detonates.

This is also where the split connects to the **yield curve** — the plot of yield against maturity. In normal times long rates exceed short rates (an upward-sloping curve), so borrowing short and lending long is *profitable*: that positive term spread is the maturity transformer's bread and butter. When the curve **inverts** (short rates above long rates), the trade reverses — borrowing short to lend long now *loses* money on a carry basis, which squeezes banks and is one reason an inverted curve so reliably precedes recessions. We don't re-derive curve mechanics here; the [yield-curve explainer](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) does that. What matters for *this* story is that the short rate and the long rate are set in two different markets, and the gap between them is the whole economics of maturity transformation. Watch how differently the two ends move:

![Short rates versus long rates from 2020 to 2026](/imgs/blogs/money-market-vs-capital-market-where-short-meets-long-3.png)

The amber line — the fed funds rate, the money market's anchor — jumps in discrete steps because the Fed *sets* it administratively, slamming from near zero in 2021 to 5.5% by mid-2023. The blue 10-year line — the capital market's benchmark — wanders continuously because it's the *market's* aggregate guess about the next decade of growth, inflation, and policy. For most of 2022–2024 the short rate sat *above* the long rate: the curve was inverted, the maturity-transformation trade was underwater, and the financial system was under quiet strain (Silicon Valley Bank's 2023 failure was, at root, a maturity-transformation accident — long bonds funded by flighty deposits, in an inverted-curve world).

The *"so what?"*: maturity transformation is simultaneously the system's greatest productive feat and its built-in fragility. The same act that funds every mortgage and every multi-year corporate investment is the act that makes runs possible. You cannot have one without the other.

## The plumbing connection: repo wires the two markets together

If maturity transformation is the *economic* link between the money market and the capital market, **repo** is the *physical* wire. It's worth slowing down on, because repo is the single mechanism that most directly proves the two markets are one system — and it sets up the dedicated Track D post on [securities lending and repo, the financing plumbing](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing).

Recall a repo is a collateralized overnight loan: I "sell" you a bond today and agree to buy it back tomorrow for slightly more. The slightly-more is the interest — the **repo rate**. From the cash lender's view it's a safe short-term investment (a money-market instrument); from the bond owner's view it's a way to *finance* a capital-market security without selling it. That's the magic: a bond dealer or hedge fund can hold \$10 billion of Treasuries while putting up only a sliver of its own cash, borrowing the rest in repo against the bonds themselves, rolling the loan every morning.

Two numbers protect the cash lender. The **repo rate** is the price (the interest). The **haircut** is the cushion: the lender advances *less* cash than the collateral is worth, so if the borrower defaults and the collateral has to be sold, there's a buffer against a price drop. A 2% haircut on a \$50 million bond means the lender hands over \$49 million and holds \$1 million of collateral value as protection. The haircut scales with how risky and how volatile the collateral is: near-zero on Treasuries, a few percent on investment-grade corporate bonds, much larger on anything illiquid. In a crisis the haircut is the dial that lenders crank — and a haircut that jumps from 2% to 10% overnight forces the borrower to find five times the cash cushion against the same bond, which is its own kind of margin call.

Repo comes in two plumbing flavors worth naming. In **bilateral repo**, two parties face each other directly and arrange their own collateral. In **tri-party repo**, a clearing bank sits in the middle, holding the collateral and valuing it daily for both sides — the dominant form for the largest dealers, and the part of the system the Fed watches most closely because so much funding runs through so few clearing banks. The Fed itself transacts in this market every day: through its overnight reverse repo facility it *takes in* cash from money funds against Treasuries, effectively setting a floor under short rates, and through its standing repo facility it can *lend* cash against Treasuries to relieve funding squeezes. The central bank is, in other words, a permanent participant in the very wire that joins the money market to the capital market — which is exactly how it can put out a money-market fire so fast.

![Overnight repo trade with cash and collateral legs and a haircut](/imgs/blogs/money-market-vs-capital-market-where-short-meets-long-6.png)

#### Worked example: an overnight repo trade

A dealer owns a \$50,000,000 Treasury note and needs cash to finance it overnight. It enters an overnight repo at a 4.90% repo rate with a 2% haircut.

**Leg 1 (today):** The dealer pledges the \$50,000,000 note and receives cash equal to face minus the haircut:

```
cash advanced = 50,000,000 * (1 - 0.02) = 49,000,000
```

**Leg 2 (tomorrow):** The dealer repays the cash plus one day of interest at the repo rate (money-market convention: 360-day year):

```
interest = 49,000,000 * 0.049 * (1 / 360)
        = 49,000,000 * 0.049 * 0.0027778
        = 6,669
repayment = 49,000,000 + 6,669 = 49,006,669
```

The dealer just financed a \$50 million capital-market bond for one night for \$6,669, using \$49 million of *money-market* cash — and tomorrow morning it does the whole thing again. The lesson: repo lets the capital market's long bonds be funded continuously by the money market's short cash, which is wonderfully efficient when markets are calm and catastrophic when the lender suddenly demands a bigger haircut or refuses to roll.

There's a second layer that makes repo even more wired into the capital market: the same bond can back several loans in a chain. The lender who receives a bond as collateral can often *re-use* it — pledge it onward to back its own borrowing, a practice called **rehypothecation**. A single Treasury note can travel through three or four sets of hands as collateral on the same day, each link extending credit against it. This **collateral velocity** is wonderfully efficient — it lets a finite pile of high-quality bonds support a much larger volume of short-term lending — but it also means the chain is only as strong as its most nervous link. If one lender in the chain pulls back and refuses to keep re-using the collateral, the contraction ripples down the whole chain at once, and credit that seemed to exist simply evaporates. The bond never moved and never defaulted; the *funding built on top of it* vanished.

That last point is the danger. Because repo is rolled daily, a lender who gets nervous can act *instantly* — either by yanking the cash (refusing to renew the loan) or by raising the haircut (demanding more collateral for the same cash). When haircuts jump across the whole market at once, every leveraged bond holder must simultaneously post more cash or sell bonds. They sell, prices fall, which makes lenders demand even bigger haircuts — a self-reinforcing spiral. Economists call this a **margin spiral** or a "run on repo," and it was a central mechanism of the 2008 crisis: the capital market's bonds couldn't hold their value because the money market's repo funding evaporated underneath them. The order-book and adverse-selection dynamics of those fire sales are formal-microstructure territory; we link out to the [quant-finance order-book treatment](/blog/trading/quantitative-finance/order-book-simulator-quant-research) rather than model them here.

There's a striking historical detail in the 2008 numbers: a sizable share of the investment banks' total funding was overnight repo, rolled every single morning. Bear Stearns and Lehman were financing long-dated, hard-to-sell securities with money they had to re-borrow daily. The day repo lenders declined to roll, those firms had no funding by sundown — the assets hadn't changed, but the wire connecting them to cash had been cut. A firm can be solvent at 9 a.m. and gone by 5 p.m. when it's funded overnight. That speed is the money market's gift in calm times and its guillotine in a crisis.

The *"so what?"*: repo is the proof that "money market" and "capital market" are not two markets but two ends of one. The short end *finances* the long end, continuously and invisibly, until the day it doesn't — and then trouble flows straight up the wire from short to long.

## Common misconceptions

**"The money market is for small, retail savers."** Backwards. The money market is overwhelmingly *wholesale* — it trades in lots of millions and the players are governments, banks, dealers, money funds, and corporate treasurers. A retail saver touches it indirectly, through a money-market *fund* or a bank deposit. The instruments themselves (a \$50 million CP issue, a \$1 billion repo) are institutional. The capital market, by contrast, is where retail actually shows up directly — buying stocks and bonds.

**"Money-market instruments are completely risk-free."** Mostly, not entirely — and the exceptions are exactly where crises start. Treasury bills are about as close to risk-free as exists. But commercial paper is *unsecured corporate credit*: if the issuer fails, you can lose everything, as Reserve Primary's holders discovered with Lehman's paper. Repo is *secured*, but secured by collateral whose value can fall faster than the haircut protects against. The whole point of 2008 and 2020 is that "near-cash" is an assumption that holds right up until it spectacularly doesn't.

**"Short-term means low risk; long-term means high risk."** Maturity and risk are *related* but not the same axis. A 30-year Treasury bond has long maturity but no credit risk (the government will pay) — its risk is *interest-rate* risk (its price swings a lot when rates move). A 30-day piece of commercial paper has short maturity but real *credit* risk if the issuer is shaky. Maturity drives *interest-rate* and *liquidity* risk; the issuer drives *credit* risk. Conflating them is how people misjudge a "safe" short instrument that happens to be an unsecured loan to a wobbly borrower.

**"A money-market fund is just a bank account."** It looks like one — stable \$1 share price, withdraw any day — but it's structurally different. A bank deposit is *insured* (up to a limit) and the bank holds capital against losses; a money fund is an *uninsured* pool of money-market securities whose \$1 price is a convention, not a guarantee. When the underlying paper loses value, the fund can break the buck. That structural gap is precisely why money funds run and insured deposits (mostly) don't.

**"The Fed sets interest rates."** It sets *one* rate — the overnight fed funds target, the money market's anchor — and only *influences* the rest. The 10-year Treasury yield, the rate that actually prices mortgages and long corporate borrowing, is set by the capital market's collective bet on the next decade. That's why the chart of short versus long rates shows one line that jumps in clean administrative steps and another that wanders continuously: the Fed controls the front end directly and the long end only by persuasion. The Fed can pin the cost of overnight money; it cannot pin the cost of thirty-year money, and the gap between what it controls and what it merely influences is the yield curve itself.

**"If the money market freezes, only Wall Street suffers."** The opposite is the scary part. The commercial-paper market funds real-economy payrolls, inventories, and receivables; repo funds the bond inventory that keeps long-term markets liquid. When the short end freezes, companies can't make payroll and bond markets can't clear — which is exactly why central banks intervene in money-market panics within *days*, not weeks. The damage doesn't stay short; it climbs the maturity ladder.

## How it shows up in real markets: when the money market breaks

The cleanest way to feel why this divide matters is to watch the short end crack — twice, twelve years apart, for the same structural reason.

### 2008: Reserve Primary breaks the buck

We opened with the moment; here's the mechanism. Money-market funds had spent years reaching for a little extra yield by holding commercial paper instead of only Treasury bills. The Reserve Primary Fund held \$785 million of Lehman Brothers CP. When Lehman filed for bankruptcy on September 15, 2008, that paper went from "near-cash" to "worth roughly nothing" overnight. The fund's net asset value fell to \$0.97 — it had "broken the buck," the cardinal sin of a vehicle whose entire promise is \$1.00.

![Money fund breaking the buck normal versus Reserve Primary 2008](/imgs/blogs/money-market-vs-capital-market-where-short-meets-long-7.png)

What happened next is the run dynamic in pure form. Because *every* prime money fund held *some* commercial paper, and because nobody knew which fund held the next Lehman, investors did the rational individual thing: pull money out of all of them, immediately. Roughly \$300 billion fled prime funds in a matter of days. To meet redemptions, funds dumped their CP holdings and refused to buy new paper — so the commercial-paper market, the thing real companies use to fund payroll, simply stopped. The US Treasury responded within a week by guaranteeing money-fund balances, and the Fed stood up emergency facilities to buy commercial paper directly. The point: a problem in one fund's *short-term* holdings nearly froze the funding of the entire real economy. Maturity transformation in reverse.

### March 2020: the dash for cash

Twelve years later, the same fault line, a different trigger. As COVID lockdowns hit, investors and institutions all wanted the same thing at the same time: *cash*, right now. This "dash for cash" hit the money market from every direction at once. Prime money funds again saw heavy redemptions and again had to dump commercial paper and CDs to raise cash — re-running the 2008 script. Even the Treasury market, normally the deepest and safest in the world, briefly seized: investors were selling Treasuries *to raise cash*, and dealers — funding their bond inventory in repo — couldn't absorb the flood. For a few days in mid-March 2020, the most liquid securities on earth couldn't find buyers at sensible prices.

The Fed's response was even larger and faster than in 2008: a Money Market Mutual Fund Liquidity Facility (to backstop money-fund redemptions), a Commercial Paper Funding Facility (to buy CP directly), and unlimited Treasury purchases to unfreeze the long end. Within roughly two weeks the panic broke. The lesson repeated: when the short end "dashes for cash," the stress travels straight into the capital market — even Treasuries — because repo and money funds are the wire connecting the two.

That 2020 episode is doubly instructive because the post-2008 reforms were *supposed* to have fixed money funds. After Reserve Primary, regulators forced institutional prime funds to abandon the fixed \$1.00 price and report a **floating net asset value** that moves with the market value of their holdings, and they allowed funds to impose redemption gates and fees in stress. The theory: if a prime fund's price visibly floats, holders won't treat it as cash and won't run. The reality in March 2020: holders ran anyway, *toward* government funds and *away* from prime funds, fast enough that the Fed still had to stand up the same emergency facility it used in 2008. The deeper point is structural and humbling — you can paper over maturity transformation with disclosure and gates, but you cannot regulate away the fact that short money is funding long assets. The mismatch is the product, not a defect to be patched out.

### 2023: a bank-shaped version of the same accident

The most recent reminder didn't involve money funds at all. Silicon Valley Bank failed in March 2023 because it had taken a flood of tech-startup deposits — money-market liabilities, withdrawable any day — and parked them in long-dated Treasuries and mortgage bonds, capital-market assets, bought when yields were near zero. When the Fed jacked the short rate to 5%+ (the amber step-function in the chart above), two things happened at once: the bank's long bonds lost market value (rates up, bond prices down), and its uninsured depositors realized they could earn far more in a government money fund than in their checking account. They moved to redeem. To meet the outflow, SVB had to sell its long bonds at a loss it could no longer hide, the loss became public, and the realization triggered a textbook run — \$42 billion attempted out in a single day. SVB held genuine value; it was *illiquid*, not initially insolvent, and the race for the exit did the rest. Strip away the labels and it is the identical accident as 1907, 2008, and 2020: long assets funded by short, flighty money, in a world where short rates had moved against the trade. The maturity divide doesn't just organize finance on the calm days — it dictates exactly how things break on the bad ones.

### The pattern under both

Both episodes share one shape. A money-market instrument that everyone *treated* as cash turned out not to be — Lehman's CP in 2008, the general scramble for liquidity in 2020. Because the money market funds the capital market (money funds buy CP that companies issue; repo finances the bonds dealers hold), the freeze propagated *up* the maturity spectrum. And in both cases the central bank had to step in fast, because a frozen money market doesn't just inconvenience traders — it stops payrolls and chokes the funding that keeps long-term markets liquid. These weren't failures of exotic derivatives; they were failures of the most "boring," "safe," short-dated corner of finance. That's the whole point: the boring corner is load-bearing.

There's a longer lineage here — every great financial crisis has a maturity-transformation accident at its core, from the 1907 panic to the savings-and-loan collapse to [LTCM in 1998](/blog/trading/finance/ltcm-1998-when-genius-failed), where a leveraged fund financing long, illiquid positions with short funding got caught when the funding vanished. Different decade, same divide.

## The takeaway: one line, two markets, one system

If you remember nothing else, remember the line. Everything in finance sorts itself by a single question — *how long until I get my money back?* — and that one question cleaves the world into a money market (a year or less: near-cash, safe, immense, a place to park) and a capital market (longer: bonds and perpetual stock, riskier, the place where savings become factories and growth). The money market preserves; the capital market grows.

But the deeper lesson is that the line is a *seam*, not a wall. The system's central act — maturity transformation, with repo as its physical wire — deliberately stitches the two sides together: short money funds long assets, capturing the term spread that pays for banks and bond dealers and funds the economy's long-horizon projects. That stitch is what makes a 30-year mortgage possible from overnight deposits, and it's the very same stitch that tears in every panic. The fragility isn't a bug bolted onto the system; it's the flip side of the most useful thing the system does.

This is also why the spine of this whole series — *secondary-market liquidity is what makes primary issuance possible* — runs straight through the money market. The reason anyone will buy a long-dated bond or stock in the primary market is the confidence they can sell it, or finance it, tomorrow. And "finance it tomorrow" *is* the money market: repo, money funds, the short end. When the short end is healthy, the long end can be funded and traded with confidence, and new capital gets raised. When the short end freezes, primary issuance stops cold, because nobody funds a thirty-year project when they can't be sure of cash next week.

There's a practical habit this gives you, too. Whenever you meet any financial instrument, ask the maturity question first — *how long until I get my money back, and how easily can I get it back early?* — and most of the instrument's behavior falls out of the answer. A money fund and a 30-year bond fund both say "fixed income" on the label, but one is a parking lot and the other is a decade-long bet on rates; the maturity tells you which. A "high-yield savings" product that pays suspiciously more than T-bills is almost certainly reaching into unsecured short credit, and the maturity-plus-issuer question exposes the risk the yield is paying you for. The divide isn't just a way to organize a textbook; it's a lens that turns an unfamiliar instrument into a known quantity in one question.

And it scales all the way up. The same maturity question that tells you whether *your* cash is safe is the question regulators ask of a bank, that a central bank asks of the whole system, and that every post-mortem of every crisis ends up answering. Short funding long, mispriced or unhedged, is the recurring murder weapon. Learn to see the maturity seam and you're not just reading finance — you're reading where it's most likely to break.

So the next time you read that "the Fed cut rates" (a money-market event) or that "a company raised \$500 million in a bond sale" (a capital-market event), you'll see they're not separate stories. They're the two ends of one machine, joined at a seam that runs right down the middle of finance — the place where short meets long.

## Further reading & cross-links

- [What a capital market is and how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use) — the full machine this post zooms into: primary and secondary markets, plumbing, intermediaries.
- [Debt vs equity: the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital) — how the capital market's two instrument families differ in claim, maturity, and risk.
- [Securities lending and repo: the financing plumbing](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing) — the dedicated deep dive on the repo wire we previewed here.
- [The yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — the short-rate-vs-long-rate relationship, derived properly (we link out rather than re-derive it).
- [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed) — a maturity-transformation accident in a hedge fund, the same divide one decade earlier.
- [Order-book simulator (quant research)](/blog/trading/quantitative-finance/order-book-simulator-quant-research) — the microstructure of the fire sales that follow a repo run.
