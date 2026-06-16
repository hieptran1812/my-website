---
title: "Credit risk: the chance you don't get paid back"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "What credit risk really is, why a US Treasury is treated as risk-free while a corporate bond is not, and how probability of default, loss given default, and exposure combine into an expected loss you can compute in dollars."
tags: ["fixed-income", "bonds", "credit-risk", "default", "credit-spread", "probability-of-default", "loss-given-default", "expected-loss", "corporate-bonds", "credit-cycle"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — credit risk is the chance that the issuer of a bond doesn't pay you back in full, and the entire discipline of credit boils down to pricing that chance in dollars.
> - A **US Treasury** is treated as credit-risk-*free* because the government borrows in a currency it can print; a **corporate bond** is not, because a company can run out of money and **default** — miss a coupon or fail to return your principal.
> - Credit loss has exactly three ingredients: the **probability of default (PD)** — how likely the issuer misses; the **loss given default (LGD)** — how much of your money you don't get back if it does; and the **exposure** — how many dollars are on the line. Multiply them and you get the **expected loss**.
> - For our running example — a \$1,000 Northwind Corp bond with a 2% annual PD and 40% LGD — the expected annual credit loss is \$8. That tiny number hides a wildly skewed reality: 98% of years you lose nothing, and the rare default does all the damage.
> - Investors won't bear that risk for free, so a corporate bond pays a **credit spread** on top of the Treasury yield — extra yield that covers the expected loss *and* a premium for the uncertainty around it.
> - Default risk is **cyclical, not constant**: defaults are rare in good times and cluster violently in recessions, which is why credit spreads blow out exactly when the economy turns.

Here is a question that sounds simple and isn't: why does a 10-year US Treasury yield 4% while a 10-year bond from a perfectly respectable company yields 6.5%? They both promise to pay you a fixed stream of coupons and hand back your \$1,000 at the end. They have the same maturity. They trade in the same market, to the same investors, on the same day. So why does one pay you half-again as much as the other for what looks like the same deal?

The answer is the single most important idea in all of corporate finance, and it has a plain-English name: the company *might not pay you back*. The US government, borrowing in dollars it can create at will, essentially cannot be forced to miss a dollar payment. A company has no printing press. If Northwind Corp's business collapses, it can run out of cash, stop paying its bondholders, and leave you holding a claim worth a fraction of what you were promised. That possibility — the chance you don't get paid back — is **credit risk**, and the extra 2.5% of yield is the market's price for bearing it.

![A US Treasury that pays every promised dollar in full beside a corporate bond that carries a real chance of a missed coupon or a default haircut](/imgs/blogs/credit-risk-the-chance-you-dont-get-paid-back-1.png)

The diagram above is the mental model to hold through the whole post. On the left is the Treasury: it promises \$1,000 plus coupons, and it delivers \$1,000 plus coupons, 100 cents on every dollar, because the issuer can always make the payment. On the right is Northwind Corp: it makes the *same* promise, but there is a real chance — say 2% in any given year — that it defaults, and if it does, you recover only part of your money. That gap between "promised" and "received" is what credit risk is about. Everything else in this post — probability of default, loss given default, credit spreads, the credit cycle — is just machinery for measuring that gap and getting paid for it. This is the first post in a series on credit, and it builds the whole subject from zero.

## Foundations: bonds, default, and the risk-free benchmark

Before we can price credit risk, we need to be precise about a handful of terms. None are hard, but credit risk lives in the relationship between them, so let's define each from scratch.

A **bond** is a tradable loan. You hand the issuer money today (you *buy* the bond), and in return the issuer promises a fixed stream of payments — the **coupons** — plus the return of the original amount, the **principal** or **par value**, at a set future date called **maturity**. A standard bond might be "a 5-year \$1,000 par note with a 4% coupon": you pay roughly \$1,000 today, you receive \$40 a year (4% of \$1,000) for five years, and you get your \$1,000 back at the end. (For the full anatomy of how those pieces fit together, see [anatomy of a bond: par, coupon, maturity, issuer](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer).)

The **issuer** is whoever is borrowing — and this is where credit risk enters. When the issuer is the **US Treasury**, it is the US federal government borrowing dollars. Because the government controls the printing of dollars, it can always create the dollars it needs to make a dollar-denominated payment. It can choose not to (a political default, like a debt-ceiling standoff), but it can never be *unable* to. For that reason, US Treasuries are treated throughout finance as the **risk-free asset** — the closest thing the world has to a loan that always pays in full. Their yield is the **risk-free rate**: the pure price of lending money for a given length of time, stripped of any worry about getting paid back. (That price of money is the master variable behind everything; see [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).)

When the issuer is a **corporation** — Apple, Ford, a regional utility, or our fictional **Northwind Corp** — the picture changes. A company's ability to pay depends on its business: its cash flow, its assets, its other debts. If the business deteriorates badly enough, the company can find itself unable to make a scheduled payment. That event is a **default**: the issuer fails to meet its contractual obligation, whether by missing a coupon, missing the principal repayment at maturity, or breaching a covenant that triggers the same consequences. Default is the thing credit risk is *about*. Everything we measure is some version of "how likely is default, and how bad is it when it happens?"

A crucial subtlety: **default does not mean you lose everything.** When a company defaults, it doesn't vanish. Its assets — factories, inventory, cash, brand, contracts — still have value, and bondholders have a legal claim on those assets ahead of the shareholders. Through bankruptcy or restructuring, bondholders typically **recover** some fraction of what they were owed. The amount you *don't* recover is your actual loss. So credit loss is really two questions stacked on top of each other: *will* it default, and *if it does*, how much do I get back?

Finally, a unit you'll see everywhere in credit: the **basis point**, written *bp* or *bps*, is one hundredth of a percentage point — 0.01%. A credit spread of "250 basis points" is 2.50%. Credit people quote spreads in basis points the way temperatures are quoted in degrees, so it's worth burning into memory now: 100 bps = 1.00%.

With those terms in hand, we can state the whole subject in one sentence. **Credit risk is the risk that the issuer defaults, and you fail to recover the full value of what you were promised.** The rest of this post turns that sentence into numbers.

### Why "risk-free" is a benchmark, not a literal claim

It's worth being honest about the phrase "risk-free." A US Treasury is not free of *all* risk. Its price still moves when interest rates move — that's **interest-rate risk** or **duration risk**, the subject of [duration: the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) — and its real value still erodes with inflation. What a Treasury is free of is **credit risk specifically**: the risk of *not getting paid the dollars you were promised*. When we call it the risk-free benchmark, we mean it is the reference point against which we measure credit risk and nothing else. A corporate bond carries *everything a Treasury carries* — the same rate risk, the same inflation risk — *plus* the chance of default. The extra yield it pays is compensation for that one extra layer. Holding the rate and inflation pieces constant by comparing same-maturity bonds is exactly how we isolate the credit piece, and it's the trick behind the entire idea of a spread.

There's one more reason the US Treasury sits in a class of its own, and it's worth stating plainly because it explains why *not all* government bonds are risk-free. The US borrows in dollars, and it controls the supply of dollars. If a payment comes due and there isn't enough tax revenue, the government can — through its central bank — create the dollars to pay it. That doesn't make the debt costless (printing dollars can stoke inflation, which is its own tax on bondholders), but it removes the specific possibility of being *forced* to miss a dollar payment. Contrast that with a country that borrows in a *foreign* currency it cannot print, or a company that has no printing press at all: for them, running out of the relevant currency is a live possibility, and that possibility is credit risk. So "risk-free" is not a statement about how trustworthy or well-run an issuer is — it's a statement about whether the issuer can be *forced* to default on its own currency's debt. A AAA-rated company is extremely trustworthy and still carries credit risk; the US Treasury carries essentially none, not because it is more virtuous, but because of this currency-printing asymmetry.

### Ratings: the market's shorthand for default probability

In practice, most investors don't estimate PD from scratch — they lean on **credit ratings**. The three big agencies (Moody's, S&P, and Fitch; see [credit rating agencies: Moody's, S&P, Fitch](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch)) assign each issuer a letter grade that is, at its core, an opinion about default probability and recovery. The scale runs from the pristine **AAA** down through **AA, A, BBB** — together the **investment-grade** band — and then into **BB, B, CCC** and below, the **high-yield** or **speculative-grade** ("junk") band. The dividing line between BBB and BB is the single most consequential boundary in credit, because a huge population of investors is contractually allowed to hold only investment-grade bonds; cross below it and you lose a chunk of your natural buyers overnight.

Roughly speaking, these ratings map onto long-run average annual default rates: AAA is a fraction of a basis point per year (near-zero), A is on the order of 0.05-0.1%, BBB perhaps 0.2-0.3%, BB around 1%, B around 3-4%, and CCC into the double digits. The numbers wobble by source and era, but the *ordering* and the *order-of-magnitude jumps between rungs* are robust. Our Northwind, at a 2% annual PD, sits squarely in the high-yield band — somewhere around BB/B — which is exactly the kind of issuer where credit risk is large enough to be worth analyzing carefully. The key intuition is that ratings are not a continuous dial; each step down roughly multiplies the default odds, so the gap between a BBB and a BB bond is far larger than the one letter suggests.

## The three building blocks of credit loss

Here is the engine of the whole subject. Any credit loss, on any bond, decomposes into exactly three numbers that multiply together.

![Expected loss equals probability of default times loss given default times the dollar exposure at risk](/imgs/blogs/credit-risk-the-chance-you-dont-get-paid-back-2.png)

The first is the **probability of default (PD)**: the chance the issuer defaults over some period, usually a year. If Northwind has a 2% annual PD, it means that in any given year there is a 2-in-100 chance it fails to pay. PD is a *probability*, so it lives between 0 and 1 (or 0% and 100%). A pristine investment-grade company might have a PD well under 0.1% per year; a struggling, deeply indebted firm might have a PD of 10% or more. PD is what credit-rating agencies are really trying to estimate when they assign a rating like AAA or BB (see [credit rating agencies: Moody's, S&P, Fitch](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch)).

The second is the **loss given default (LGD)**: *if* the issuer defaults, what fraction of your exposure do you lose? Its complement is the **recovery rate** — the fraction you get back. If you recover 60 cents on the dollar, your recovery rate is 60% and your LGD is 40% (they always sum to 100%). LGD depends heavily on where your bond sits in the company's pecking order — a **senior secured** bond backed by specific assets recovers far more than a **junior unsecured** bond that stands behind everyone else. (That pecking order is the *capital structure*, and it's a whole topic of its own in this series.) Typical senior unsecured corporate LGD runs around 60% (i.e. ~40% recovery), but it varies enormously by seniority, industry, and how bad the bankruptcy is.

The third is the **exposure** — sometimes called **exposure at default (EAD)** — the number of dollars actually at risk. For a plain bond you bought at par, that's roughly your principal, \$1,000. For a portfolio it's the total invested; for a loan that can be drawn down, it's the expected outstanding balance at the moment of default. For our purposes, exposure is just "how much money is on the line."

Multiply the three and you get the **expected loss (EL)** — the average dollar loss you should expect over the period, accounting for both the chance of default and its severity:

$$
\text{Expected Loss} = \text{PD} \times \text{LGD} \times \text{Exposure}
$$

Here PD is the probability of default (a fraction), LGD is the loss given default (a fraction), and Exposure is the dollars at risk. The formula is almost embarrassingly simple, and yet it is the foundation that banks, insurers, and bond investors build entire risk systems on. The word *expected* is doing heavy lifting: it's a probability-weighted average, not a prediction of what will actually happen in any single year. In most years you'll lose exactly zero. The expected loss is the long-run average across many bonds and many years — and that distinction, as we'll see, is the whole reason credit is dangerous.

#### Worked example: Northwind's expected annual credit loss

Let's put numbers on it. You own one **Northwind Corp** bond: \$1,000 par, you hold it at par, so your **exposure is \$1,000**. Northwind's **annual PD is 2%** (0.02). It's a senior unsecured bond, and you estimate that in default you'd recover 60 cents on the dollar, so your **LGD is 40%** (0.40). Plug in:

$$
\text{EL} = 0.02 \times 0.40 \times \$1{,}000 = \$8 \text{ per year}
$$

Your expected credit loss is **\$8 a year**. Read that carefully, because it's counterintuitive. It does *not* mean you'll lose \$8 next year. In 98 years out of 100, Northwind doesn't default and you lose \$0. In the other 2 years, it defaults and you lose \$400 (40% of \$1,000). The *average* of "\$0, ninety-eight percent of the time; \$400, two percent of the time" is:

$$
0.98 \times \$0 + 0.02 \times \$400 = \$8
$$

— exactly the expected loss. The \$8 is a blend of a common nothing and a rare disaster.

*Expected loss is the price tag the market would put on Northwind's credit risk if it only had to cover the average — but the average is a fiction no single year ever delivers.*

#### Worked example: the same bond, but senior secured

Now change one input. Suppose instead of a senior *unsecured* bond, you hold a **senior secured** Northwind bond — one backed by a specific pile of collateral (say, the company's real estate). In default, secured creditors get paid first out of that collateral, so recovery is much higher: say you'd recover 80 cents on the dollar, making **LGD = 20%**. PD and exposure are unchanged (the same company can still default at 2%; collateral doesn't make default less likely, it makes it less *painful*). Now:

$$
\text{EL} = 0.02 \times 0.20 \times \$1{,}000 = \$4 \text{ per year}
$$

Halving the LGD halved the expected loss, from \$8 to \$4, even though the probability of default never moved. This is why where you sit in the capital structure matters as much as which company you lend to.

*Two bonds from the same issuer, defaulting at the same rate, can carry very different credit losses — seniority decides how much you keep when things go wrong.*

## Default risk is cyclical: it clusters in recessions

So far we've treated PD as a fixed number — 2% a year, every year. That's a useful simplification for learning the arithmetic, but it hides the most important fact about credit risk: **default probabilities are not constant. They are wildly cyclical.** In a healthy, growing economy, companies make money, refinance their debt easily, and almost never default. In a recession, revenues fall, credit markets freeze, and defaults don't just rise — they *cluster*, hitting many companies at once.

![The high-yield corporate default rate over time, low in good years and spiking into double digits during the dot-com recession, the 2008 to 2009 financial crisis, and the 2020 COVID shock](/imgs/blogs/credit-risk-the-chance-you-dont-get-paid-back-3.png)

The figure shows the pattern (the numbers are illustrative, but the shape is real and well-documented). The **speculative-grade default rate** — the annual default rate among riskier, "high-yield" or "junk" companies — sits in the low single digits during expansions, often under 2-3%. Then a recession hits, and it explodes: it spiked toward 10% during the 2001 dot-com bust, breached double digits in 2009 during the global financial crisis, and jumped again in 2020 during the COVID shock. Between those spikes, in the calm middle of each expansion, it grinds back down toward its lows.

This clustering is the defining feature of credit risk, and it has two consequences that shape everything.

First, **default risk is correlated across issuers.** If Northwind defaults this year, it's probably because the whole economy is in trouble — which means other companies are *also* more likely to be defaulting at the same time. Defaults don't arrive independently like coin flips; they arrive in waves, driven by a common engine (the business cycle, interest rates, credit availability). A portfolio of 100 corporate bonds is not 100 independent 2% bets; it's 100 bets that all get worse together exactly when you can least afford it. This correlation is why a diversified bond portfolio is *less* diversified against credit risk than it looks, and it's the reason 2008 was a systemic event rather than a series of isolated bankruptcies.

Second, **the time you most need your bonds to be safe is the time they're most at risk.** Recessions are when you might lose your job, when stocks are falling, when you want your "safe" income to keep flowing. That's precisely when corporate defaults spike. Treasuries shine here — they pay regardless of the economy, which is why they rally in a crisis — while corporates wobble. The cyclicality of credit risk is the deep reason a Treasury and a corporate, identical on paper, are not the same asset.

#### Worked example: the same 2% PD, but it's not really 2%

Northwind's "2% annual PD" is best understood as a long-run *average* across the cycle, not the rate in any particular year. Suppose the real pattern is: in the ~8 good years of a typical decade, PD is just **0.5%**; in the ~2 bad years, PD jumps to **8%**. The average is:

$$
\frac{8 \times 0.5\% + 2 \times 8\%}{10} = \frac{4\% + 16\%}{10} = 2.0\%
$$

The blended average is the 2% we've been using — but the lived experience is nothing like a steady 2%. It's four years of near-perfect safety punctuated by a single terrifying year where the default odds are sixteen times higher. The expected-loss math still works on the average, but a risk manager who only looks at the average and forgets the clustering will be blindsided every recession.

*A constant-looking PD is usually a smooth average hiding a violently lumpy reality, and the lumps all land in the same bad years.*

### What actually drives the default rate up and down

It's worth being concrete about *why* defaults cluster, because the mechanism is what makes the cyclicality predictable rather than mysterious. Three forces do most of the work, and they tend to fire together.

The first is **revenue.** In a recession, customers spend less, so companies' sales fall. A firm with thin margins and a lot of fixed costs can swing from profit to loss on a modest drop in revenue — this is **operating leverage**. The more a company has borrowed relative to its earnings (its **financial leverage**), the less room it has to absorb that swing before it can't cover its interest payments. So the same downturn that dents a strong company can be fatal to a heavily indebted one, which is why high-yield default rates move so much more than investment-grade ones across the cycle.

The second is **the refinancing window.** Most companies don't repay their bonds out of cash — they *refinance*, issuing a new bond to pay off the maturing one, rolling the debt forward indefinitely. That works beautifully until the credit market freezes. In a crisis, lenders pull back, new issuance dries up, and a company with a bond maturing next month can suddenly find there's no one willing to lend it the replacement. A perfectly solvent business can default purely because the refinancing window slammed shut at the wrong moment. This is why a **wall of maturities** — a large amount of debt all coming due in a short window — is something credit analysts watch nervously: it concentrates refinancing risk in time.

The third is **interest rates themselves.** When the central bank raises rates to fight inflation, every company's borrowing cost rises as it refinances, eating into the cash available to service debt. Higher rates also slow the economy (hitting revenue) and tighten credit (hitting the refinancing window) — so a hiking cycle pushes all three drivers in the wrong direction at once. The lag between rate hikes and the default wave they eventually produce is one of the slow-motion mechanisms behind the credit cycle.

#### Worked example: how leverage turns a small revenue drop into a default

Picture two companies, each with \$100 of revenue and \$8 of annual interest to pay. **Solid Co** has \$20 of operating profit (before interest), so it covers its \$8 interest 2.5 times over — a comfortable **interest coverage ratio** of 2.5. **Stretch Co** has the same \$100 revenue but only \$10 of operating profit, so it covers its \$8 interest just 1.25 times. Now a recession cuts both companies' revenue by 15%, and because of operating leverage, operating profit falls by more — say a third. Solid Co's profit drops to about \$13, still covering \$8 of interest 1.6 times: tight, but it survives. Stretch Co's profit drops to about \$6.70 — *less than its \$8 interest bill.* It can no longer pay its lenders from operations. Unless it can refinance or raise cash, it defaults.

Same recession, same revenue hit, opposite outcomes — the only difference was how much cushion each company had between its earnings and its interest bill. This is the entire reason PD is so much higher and so much more cyclical for indebted, low-coverage firms: they have no margin for the bad year.

*Leverage doesn't change whether the bad year comes — it changes whether the company survives it, which is why the same recession spares the strong and kills the stretched.*

## The credit spread: getting paid to bear the risk

If a corporate bond can lose you money and a Treasury can't, no rational investor would buy the corporate unless it paid *more*. That extra yield is the **credit spread**: the difference between a corporate bond's yield and the yield of a comparable-maturity Treasury.

$$
\text{Corporate yield} = \text{Treasury yield} + \text{Credit spread}
$$

If the 5-year Treasury yields 4.0% and Northwind's 5-year bond yields 6.5%, the credit spread is **2.5%, or 250 basis points**. That spread is the compensation, paid every year in extra yield, for taking on Northwind's default risk. (For how spreads behave across the quality ladder, from investment grade to high yield, see [corporate credit: investment grade, high yield, spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads).)

Here's the part that surprises people: **the credit spread is bigger than the expected loss.** You might expect the spread to be exactly the expected loss expressed as a percentage — after all, that's the average cost of the risk. But it's reliably *larger*, and the gap is one of the most studied facts in finance (the "credit spread puzzle").

![The credit spread decomposed into the risk-free Treasury yield plus an expected-loss component plus an extra risk premium for uncertainty and illiquidity](/imgs/blogs/credit-risk-the-chance-you-dont-get-paid-back-4.png)

The figure decomposes the corporate yield. The base is the Treasury yield — the risk-free floor every bond is priced above. On top of it sits the credit spread, and that spread splits into two pieces:

1. **The expected-loss component.** This covers the average annual credit loss. For Northwind, expected loss is \$8 on \$1,000, or 0.8% a year, so roughly 0.8% of the spread is just covering the math we did above. This piece is "fair" in the actuarial sense — it's the long-run cost of defaults.

2. **The risk premium.** This is the rest — the part of the spread *beyond* expected loss. Investors demand it for several reasons. Defaults are uncertain and clustered, so even if the *average* loss is 0.8%, the *actual* loss in any year is highly variable and tends to spike exactly when investors are already hurting (the correlation problem). Risk-averse investors charge extra to bear that uncertainty. On top of that, corporate bonds are **less liquid** than Treasuries — harder to sell quickly without moving the price — so part of the spread is a **liquidity premium**. And there's a **tax and ratings** component too. Add it up and the risk premium often *exceeds* the pure expected-loss piece.

#### Worked example: splitting Northwind's 250 bp spread

Northwind's spread is 250 bps (2.50%). We computed its expected loss as 0.8% a year. So the decomposition is:

$$
\underbrace{2.50\%}_{\text{spread}} = \underbrace{0.80\%}_{\text{expected loss}} + \underbrace{1.70\%}_{\text{risk premium}}
$$

Less than a third of the spread is covering the average default cost. The other 170 bps is pure compensation for *uncertainty*: for the chance that this is one of the bad years, for the fact that Northwind's troubles would coincide with everyone else's, and for the difficulty of selling the bond in a panic. If you collected the 250 bp spread every year and defaults exactly matched the 0.8% average, you'd earn 170 bps of "excess" return for bearing risk that, on average, didn't show up. That excess is the historical reward for being a corporate-credit investor — and it's also why credit looks like free money right up until the cycle turns and the average year finally arrives all at once.

*The spread pays you for the average loss plus a premium for the terror of the tail — and the premium, not the average, is most of the check.*

#### Worked example: does the spread actually compensate you?

Let's sanity-check whether bearing Northwind's risk is worth it. You buy the \$1,000 bond yielding 6.5%; the Treasury yields 4.0%. Over one year, in the *no-default* case (98% likely), you earn the full 6.5%, or \$65, versus \$40 on the Treasury — \$25 more. In the *default* case (2% likely), you lose 40% of principal, \$400, partly offset by some coupon — call the net loss roughly \$360. Your probability-weighted outcome relative to the Treasury:

$$
0.98 \times (+\$25) + 0.02 \times (-\$385) = \$24.50 - \$7.70 = +\$16.80
$$

On average you come out about \$17 ahead of the Treasury per year for bearing the risk — that's your reward for credit. But notice the shape: a near-certain small gain (\$25) against a rare large loss (\$385). You're being paid a steady premium to occasionally absorb a punch. Whether \$17 of expected excess is *enough* depends on how much you fear the punch — which is exactly what the risk premium is haggling over.

*A credit spread is a bet that pays you a little every year and costs you a lot once in a while; the question is never just the average, it's whether you can survive the once-in-a-while.*

### Backing PD out of the spread

The spread doesn't just compensate you for default risk — it secretly *encodes the market's view* of that risk, and you can read it back out. If you're willing to assume the spread is pure expected-loss compensation (ignoring the risk premium for a moment), there's a famous rule of thumb that connects the three quantities:

$$
\text{Spread} \approx \text{PD} \times \text{LGD}
$$

In words: the annual spread roughly equals the annual probability of default times the loss given default. Rearranged, it gives you the market-**implied PD**:

$$
\text{Implied PD} \approx \frac{\text{Spread}}{\text{LGD}}
$$

This is exactly the calculation that the **credit default swap (CDS)** market does for a living. A CDS is an insurance contract on a bond: the buyer pays an annual premium (quoted as a spread, in basis points) and, if the issuer defaults, the seller pays out the loss. Because a CDS spread is a clean, traded price for default protection, it lets the whole market converge on a single number for an issuer's credit risk — and that number, divided by an assumed LGD, is the market's implied probability of default.

#### Worked example: what spread is the market implying for Northwind?

Reverse-engineer Northwind. The market quotes its 5-year spread at 250 bps (2.50%), and assume the standard 60% LGD that the CDS market often uses as a convention (i.e. 40% recovery). The naive implied PD is:

$$
\text{Implied PD} \approx \frac{2.50\%}{0.60} = 4.17\% \text{ per year}
$$

So if the spread were *purely* expected-loss compensation, the market would be implying a 4.17% annual default probability — much higher than our fundamental 2% estimate. Why the gap? Because the spread is *not* pure expected loss; it contains the risk premium we dissected above. The "implied PD" backed out of a spread is therefore an **upper bound** that mixes true default odds with the market's risk aversion. The wedge between the implied 4.17% and a fundamental 2% *is* the risk premium, expressed as a probability. Sophisticated credit investors live in exactly this gap: when the market's implied PD looks far higher than their fundamental estimate, the bond may be cheap (you're being overpaid for the risk); when it looks too low, the bond is expensive and the spread isn't compensating you enough.

*A spread is a two-way mirror: read one way it's your compensation, read the other it's the market's implied probability of default — and the difference between implied and fundamental is where credit investors make their living.*

## Default probability compounds over the life of the bond

We've been working with a one-year PD. But a bond lives for years, and the chance it defaults *at some point before maturity* is larger than the one-year chance — it accumulates. This is the **cumulative probability of default**, and getting it right matters enormously for longer bonds.

The logic is the mirror image of survival. If Northwind has a 2% chance of defaulting each year, it has a **98% chance of surviving** each year. To survive *two* years, it must survive year one *and* year two: 0.98 × 0.98 = 0.9604, a 96.04% chance. The cumulative *default* probability over two years is therefore 1 − 0.9604 = **3.96%** — a bit less than 2 × 2% = 4%, because the second year's default can only happen if it survived the first. Generalizing, the chance of surviving $t$ years is $0.98^t$, and the cumulative default probability is:

$$
\text{Cumulative PD}(t) = 1 - (1 - \text{annual PD})^t = 1 - 0.98^{\,t}
$$

where $t$ is the number of years and the annual PD is 2%.

![Cumulative default probability rising over the years as a small constant annual default chance compounds across the life of the bond](/imgs/blogs/credit-risk-the-chance-you-dont-get-paid-back-5.png)

The figure plots it. The curve starts at zero and climbs, steeply at first and then bending as the compounding eats into the survivors. At 1 year it's 2.0%; by year 5 (Northwind's maturity) it has reached about **9.6%**; if the bond somehow ran to year 10, it would be **18.3%**. A "2% a year" company has nearly a 1-in-10 chance of defaulting over a 5-year horizon and nearly 1-in-5 over a decade. The longer you lend, the more chances the company has to fail — which is one more reason long-dated corporate bonds yield more than short-dated ones from the same issuer.

#### Worked example: the 5-year cumulative loss on Northwind

You hold Northwind's 5-year bond. What's the chance you eat a default loss *somewhere* in those five years?

$$
\text{Cumulative PD}(5) = 1 - 0.98^5 = 1 - 0.9039 = 0.0961 = 9.61\%
$$

So roughly a **9.6% chance** of a default event over the bond's life. If a default happens, your loss is LGD × exposure = 40% × \$1,000 = \$400. The cumulative expected loss over the five years is approximately:

$$
0.0961 \times \$400 \approx \$38
$$

That's the lifetime expected credit loss — versus \$8 a year, which over five years would naively sum to \$40. (The two don't match exactly because compounding survival slightly reduces the cumulative default chance below 5 × 2%, and because a default in year 1 stops the clock.) Either way, the lesson holds: a "small" 2% annual default rate translates into a meaningful ~10% chance of trouble over a realistic holding period, and that's before you've considered that the 2% itself spikes in recessions.

*A default probability that looks tiny per year compounds into a real probability over a bond's life — time is the credit investor's slow enemy.*

## The shape of credit risk: a skewed distribution

We keep returning to one idea, and it deserves its own section because it's what makes credit genuinely treacherous: **the distribution of credit outcomes is extremely skewed.** It is not a gentle bell curve centered on the expected loss. It is a giant spike at "lost nothing" with a thin, ugly tail of "lost a lot."

![The credit loss distribution showing a huge probability of losing nothing and a thin tail of rare large losses on the thousand dollar bond](/imgs/blogs/credit-risk-the-chance-you-dont-get-paid-back-6.png)

The figure shows the one-year outcome distribution for the Northwind bond, broken out by what you recover if it defaults. The overwhelming bar is at **\$0 loss**: 98% of the time, no default, you lose nothing. The remaining 2% of probability is spread across a tail of large losses, depending on the recovery rate in the specific default — lose \$200 if you recover 80%, \$400 if you recover 60%, \$600 if you recover 40%, \$800 if you recover only 20%. The mean of this whole distribution is just \$8, but **no single outcome is anywhere near \$8.** You either lose nothing or you lose hundreds.

This skew has three brutal implications.

**Averages lie about your experience.** A portfolio earning a credit spread looks like it's quietly clipping a steady premium — until it isn't. Most periods are "better than average" (you lose nothing, you keep the whole spread), which lulls investors into thinking the risk is mild. Then the rare year arrives, the losses are far "worse than average," and they all land together. The arithmetic mean is honest about the long run and deeply misleading about any short run.

**The tail is where the money is made and lost.** Because the distribution is dominated by a rare large loss, the *recovery rate* — the thing that determines how big that loss is — matters enormously, and it is itself uncertain and worse in bad times (recoveries fall in recessions, exactly when defaults rise, a vicious double-whammy). Credit analysis is largely the art of estimating the tail, not the average.

**Diversification helps, but less than you'd hope.** Spreading across 100 issuers smooths the spike-and-tail of a single bond into something more bell-shaped — *if* defaults are independent. But we already know they aren't: they cluster in recessions. So the diversified portfolio still has a fat tail, just a systemic one. You can diversify away a single company's bad luck; you cannot diversify away the business cycle.

#### Worked example: variance, not just the mean

Let's quantify how dispersed Northwind's one-year outcome is. Simplify to two outcomes: lose \$0 with probability 0.98, lose \$400 with probability 0.02. The mean is \$8. The variance is the probability-weighted squared deviation from the mean:

$$
\text{Var} = 0.98 \times (0 - 8)^2 + 0.02 \times (400 - 8)^2
$$
$$
= 0.98 \times 64 + 0.02 \times 153{,}664 = 62.7 + 3{,}073 = 3{,}136
$$

The standard deviation is $\sqrt{3{,}136} = \$56$. So the *typical* swing around the \$8 mean is about \$56 — seven times the mean itself. A risk whose standard deviation dwarfs its expected value is the signature of a skewed, tail-dominated bet. This is the statistical fingerprint of credit, and it's why credit losses always feel like they come "out of nowhere": the mean is small and calm, the dispersion is enormous and lumpy.

*Credit's mean is a whisper and its tail is a scream; managing credit risk means managing the scream, not the whisper.*

## Putting it together: the expected-loss matrix

The three building blocks — PD, LGD, exposure — give you a single, flexible tool. Once you can compute expected loss, you can ask "what if?" by flexing the inputs, which is exactly what every credit desk does to stress-test a position.

![A matrix of Northwind expected annual credit losses across different probability of default and loss given default assumptions on the thousand dollar bond](/imgs/blogs/credit-risk-the-chance-you-dont-get-paid-back-7.png)

The figure is the expected-loss table for the \$1,000 Northwind bond across a grid of assumptions. Columns vary the **PD** — 1% for a strong issuer, 2% for our base case, 5% for a distressed one. Rows vary the **LGD** — 30% if well secured, 40% base case, 60% if junior and unsecured. Each cell is just PD × LGD × \$1,000:

| | PD 1% (strong) | PD 2% (base) | PD 5% (distressed) |
|---|---|---|---|
| **LGD 30% (secured)** | \$3 / yr | \$6 / yr | \$15 / yr |
| **LGD 40% (base case)** | \$4 / yr | **\$8 / yr** | \$20 / yr |
| **LGD 60% (junior)** | \$6 / yr | \$12 / yr | \$30 / yr |

Three things jump out. First, the base case sits in the middle at **\$8 a year** — our running number. Second, expected loss scales **linearly** with both inputs: double the PD and you double the EL; double the LGD and you double the EL. There's no hidden nonlinearity in the expected-loss formula itself (the nonlinearity is all in the *distribution*, which the formula averages over). Third, the range is wide — from \$3 to \$30, a 10× spread — and that's only flexing two inputs over plausible ranges. A small change in your view of either default odds or recovery moves the price of the credit risk a lot, which is why credit analysts fight so hard over exactly these two numbers.

#### Worked example: stressing Northwind into a recession

Use the matrix to run a recession scenario. In normal times, Northwind is the base case: PD 2%, LGD 40%, EL = \$8. Now a recession hits. Two things happen at once: default odds rise (PD jumps to 5%) *and* recoveries fall, because in a downturn the company's assets are worth less and more creditors are fighting over them (LGD rises from 40% to 60%). Read off the matrix — bottom-right region:

$$
\text{EL}_{\text{recession}} = 0.05 \times 0.60 \times \$1{,}000 = \$30 \text{ per year}
$$

The expected loss didn't rise modestly — it nearly **quadrupled**, from \$8 to \$30, because the two inputs got worse *together*. This is the recession double-whammy in one number: PD and LGD are positively correlated, both deteriorating in the same bad state of the world. A model that stresses only PD, holding LGD fixed, badly understates the danger. The honest stress moves both.

*Credit losses don't add up in a recession, they multiply — defaults get more likely and more severe at the same time, and the product is what hurts.*

## How credit risk connects to everything else

Credit risk isn't a niche corner of finance; it's a master variable that radiates outward, which is the throughline of this whole series — bonds are the price of money, and credit is the price of *risky* money.

When credit spreads widen — when the market suddenly demands a bigger premium to lend to companies — it raises the cost of borrowing for every firm, not just the ones in trouble. Companies postpone investment and hiring; some can't refinance maturing debt and are pushed toward the default they were being charged to insure against. Widening spreads are both a *symptom* of economic stress and a *cause* of more of it — a feedback loop that turns a credit scare into a real-economy slowdown. This is why central banks watch credit spreads as closely as they watch the [yield curve](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance): a sharp spread widening is the bond market pricing in a recession, and sometimes helping cause one.

Credit risk also reaches into assets that don't look like bonds at all. A bank loan is credit risk. A mortgage is credit risk (will the homeowner pay?). The receivables a company is owed by its customers are credit risk. Even a stock is, in a sense, the most junior slice of a company's capital structure — the first to be wiped out in a default and the last to recover — which is why stocks and a company's risky bonds tend to move together when its credit deteriorates. Understanding PD × LGD × exposure gives you a lens that works on all of them.

#### Worked example: a tiny spread move, a big price move

Spreads matter for *prices*, not just yields, and the link can be violent for longer bonds. Suppose Northwind's 5-year bond's spread widens from 250 bps to 350 bps — a 100 bp (1.0%) move — because the market's view of its PD worsened. A bond's price falls when its yield rises, and the size of that fall scales with the bond's **duration** (its sensitivity to yield changes; see [modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk)). For a 5-year bond, duration is roughly 4.5 years, so a 1.0% yield rise drops the price by approximately:

$$
\Delta \text{Price} \approx -\text{Duration} \times \Delta \text{yield} = -4.5 \times 1.0\% = -4.5\%
$$

The bond loses about 4.5% of its value — \$45 on \$1,000 — not because Northwind defaulted, but merely because the market's *fear* of default rose by 100 bps. You don't need an actual default to lose money on credit; you only need the market to reprice the risk. This is **spread risk** (or **mark-to-market credit risk**), and for a long bond it can dwarf the year's coupon. Many credit losses in practice are these repricings, not realized defaults.

*You can be paid a credit spread and still lose money when the spread widens — credit risk shows up in prices long before, and far more often than, it shows up in actual defaults.*

## Common misconceptions

**"Default means you lose all your money."** No — default means the issuer failed to pay *on schedule*, not that your claim is worthless. Bondholders have a legal claim on the company's assets ahead of shareholders, and through bankruptcy or restructuring they typically recover a meaningful fraction — historically around 40 cents on the dollar for senior unsecured corporate bonds, more for secured, less for junior. Your actual loss is LGD × exposure, not the full exposure. Confusing default with total loss leads people to massively overestimate the cost of credit risk — and to ignore the recovery rate, which is half the equation.

**"The credit spread is just the expected loss."** It isn't, and the gap is the whole reason corporate credit has historically rewarded investors. The spread reliably *exceeds* expected loss because investors demand a premium for uncertainty (defaults cluster in bad times), for illiquidity (corporates are harder to sell than Treasuries), and for taxes and other frictions. For our Northwind example, expected loss was 0.8% but the spread was 2.5% — most of the spread was risk premium, not actuarial cost. An investor who thinks the spread only covers expected loss will conclude credit is fairly priced when, on average, it has paid more than that.

**"A higher yield means a better deal."** A corporate bond yields more than a Treasury *because* it's riskier, not because it's a free lunch. The extra yield is compensation for a real expected loss plus the chance of a much larger one. Reaching for yield — buying riskier and riskier bonds because they "pay more" — is buying more credit risk, and that risk has a habit of arriving all at once in a recession. The yield you see is the yield you're promised; the yield you *realize* is reduced by defaults you haven't experienced yet.

**"Defaults are random and independent, so a diversified portfolio is safe."** Defaults are emphatically *not* independent. They are driven by a common engine — the business cycle, interest rates, credit availability — and they cluster: many companies default in the same recessionary windows. A portfolio of 100 corporate bonds is far more concentrated in *systemic* credit risk than a naive "100 independent 2% bets" model suggests. Diversification smooths out single-company bad luck; it does almost nothing against a recession that hits every issuer at once.

**"Investment-grade bonds are essentially safe."** Investment grade (roughly BBB and above) has a *low* default rate, but low is not zero, and the category includes a huge volume of BBB-rated debt sitting one downgrade away from "junk." In a severe recession, even investment-grade default rates rise, and — more commonly — investment-grade *spreads* widen sharply, inflicting mark-to-market losses without any default at all. "Safe" in credit is always relative to the Treasury benchmark, never absolute.

**"If the company is profitable, its bonds can't default."** Default is about *cash and timing*, not accounting profit. A profitable company can default if it can't refinance a maturing bond when credit markets are frozen, or if a covenant breach triggers acceleration, or if its profits are real but its cash is tied up. Many defaults are **liquidity** events — the company is solvent on paper but can't make a payment when it comes due. This is exactly what makes recessions so dangerous for credit: the refinancing window slams shut precisely when companies need it.

## How it shows up in real markets

**The 2008 global financial crisis: correlated default in action.** The crisis is the textbook case of credit risk's worst feature — correlation. Mortgage defaults, which models had treated as roughly independent, turned out to be driven by a common factor (national house prices), and when prices fell everywhere at once, defaults clustered catastrophically. Securities built on the assumption of independence — and rated AAA on that basis — suffered losses that "couldn't happen" under the models. The high-yield default rate spiked toward double digits, investment-grade spreads blew out to levels not seen in decades, and the lesson was seared into the industry: the PD × LGD framework is only as good as your assumption about *correlation*, and in a systemic crisis that correlation goes to one.

**The energy sector, 2015-2016: a sector-specific default wave.** When oil prices collapsed from over \$100 to under \$30 a barrel, US energy companies — many of which had borrowed heavily in the high-yield market to fund shale drilling — saw their cash flows evaporate. Default rates in the energy sector soared while the broader economy kept growing, a reminder that credit risk clusters not just in economy-wide recessions but within *sectors* hit by a common shock. Investors who thought they were diversified across "many high-yield bonds" discovered they were concentrated in a single bet on oil. Recoveries, too, were poor: in a glut, the collateral (oil reserves, drilling equipment) was worth far less, so LGD rose just as PD did.

**COVID-19, March 2020: spreads gap before defaults.** When the pandemic shut down the economy, corporate credit spreads exploded in a matter of weeks — high-yield spreads roughly tripled — pricing in a wave of defaults that the market expected but that had not yet happened. This is spread risk in its purest form: investors lost money on the *fear* of default long before defaults materialized. Then the Federal Reserve did something unprecedented: it announced it would buy corporate bonds directly. Spreads collapsed almost as fast as they had widened, the feared default wave was muted, and the episode demonstrated both how violently credit reprices on expectations and how central-bank intervention can short-circuit the credit-fear feedback loop. Many of the worst-case defaults never arrived — but anyone who sold at the March lows realized the loss anyway.

**Argentina and sovereign default: not even governments are always safe.** We've treated the *US* Treasury as risk-free because it borrows in its own printable currency. But governments that borrow in a *foreign* currency — or that simply choose not to pay — absolutely carry credit risk. Argentina has defaulted on its sovereign debt repeatedly over the past century, most spectacularly in 2001 with roughly \$100 billion of debt, and again in subsequent decades. Sovereign default brings the same PD × LGD machinery, with the wrinkle that "loss given default" is decided by political restructuring negotiations rather than bankruptcy courts, and recoveries can be brutal. (The broader dynamic — how bond markets discipline governments — is covered in [sovereign debt and the bond vigilantes](/blog/trading/macro-trading/sovereign-debt-and-the-bond-vigilantes).) The point: "credit-risk-free" is a property of the *US* Treasury specifically, not of government bonds in general.

**The fallen angels of 2020: the downgrade cliff.** A "fallen angel" is a bond that was investment grade and gets downgraded to high yield. Because many large investors (pension funds, insurers, index funds) are *required* to hold only investment-grade bonds, a downgrade forces selling — sometimes into a market with few buyers — which crushes the price beyond what the change in default odds alone would justify. In 2020, a record volume of debt fell from BBB to junk, and the forced-selling dynamic amplified the losses. This is credit risk interacting with market structure: the *rating*, not just the underlying default probability, determines who is allowed to own the bond, so a ratings change can be a price event all by itself. It's a vivid reminder that in real markets, credit risk is never just about whether the company pays — it's about how the whole ecosystem of lenders reacts to the *risk* that it won't.

## When this matters to you, and where to go next

Credit risk touches your life far beyond a bond portfolio. The interest rate on your mortgage, your car loan, and your credit card is, in large part, the lender pricing *your* PD and LGD — your probability of not paying and how much they'd lose if you didn't. The yield on the corporate bond fund in your retirement account is the credit spread we just dissected. And when you read that "spreads are widening" in the financial news, you now know that's the bond market raising its estimate of corporate default risk — often the earliest warning that the economy is turning.

The single idea to carry forward is the decomposition: **expected loss = PD × LGD × exposure**, sitting inside a *skewed, cyclical, correlated* distribution that the expected-loss number quietly averages away. Master the formula, then never trust it without remembering the distribution behind it.

From here, the natural next steps in this series go deeper into each piece: how the capital structure determines LGD (who gets paid first in a default), how ratings agencies estimate PD, how credit spreads are quoted and traded, and how default risk is hedged with credit derivatives. For the allocation view of where credit fits among other assets, see [corporate credit: investment grade, high yield, spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) and [government bonds: the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration). For the heavier mathematics of valuing risky cash flows, see [bond pricing](/blog/trading/quantitative-finance/bond-pricing). And to keep the macro context in view — why all of this beats to the rhythm of the business cycle — return to [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

*This is educational material about how credit risk works, not advice to buy or sell any security.*
