---
title: "The great bond crises: 1994, 2008, 2020, 2022, and 2023"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The series finale: five great bond shocks since 1994, each a real-world stress test of one idea this series built — duration, credit, liquidity, the correlation flip, and the duration gap."
tags: ["fixed-income", "bonds", "bond-crisis", "duration-risk", "credit-risk", "liquidity", "svb-2023", "orange-county-1994", "2008-financial-crisis", "ldi-crisis"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — every great bond crisis since 1994 was the same lesson taught five different ways: when the price of money moves faster than the people holding bonds expected, the ones who ignored duration, credit, liquidity, or the match between their assets and their promises get carried out.
> - **1994 — the "great bond massacre."** A surprise round of Fed rate hikes cut the value of long bonds, and **Orange County's leveraged duration bet** turned a manageable loss into a \$1.6 billion bankruptcy. *Lesson: duration plus leverage.*
> - **2008 — the global financial crisis.** Subprime mortgages were pooled into **MBS, re-sliced into CDOs, and insured with CDS**; when the loans defaulted the whole chain broke and counterparties stopped trusting each other. *Lesson: credit and counterparty risk.*
> - **2020 — the "dash for cash."** For a few days even **US Treasuries**, the safest asset on Earth, were hard to sell, until the Fed promised to buy unlimited amounts. *Lesson: liquidity and market structure.*
> - **2022 — the worst bond year in modern history.** Inflation forced rates up ~4 percentage points and **long bonds fell ~30%**, while stocks fell too — the stock-bond hedge broke. *Lesson: duration risk and the correlation flip.*
> - **2023 — SVB and the UK gilt crisis.** **Silicon Valley Bank** held long Treasuries against deposits that could leave in an afternoon — an unmanaged duration gap — and **UK pension funds** got margin-called into a doom loop. *Lesson: immunization and asset-liability matching.*

Here is a question worth sitting with before we start. Why do bond crises keep happening to people who, by any reasonable measure, were *not* taking wild risks? Orange County in 1994 was a sleepy municipal investment pool for schools and water districts. Silicon Valley Bank in 2023 was holding US Treasuries — the single safest financial instrument that exists. UK pension funds in 2022 were doing the most conservative thing a pension can do: hedging their long-term promises. None of these were degenerate gamblers. And yet each of them detonated.

The answer is the thread that has run through this entire series. **Bonds are the price of money, and the price of money sets every other price.** When that price moves — when interest rates jump, or credit fear spikes, or the buyers all want to sell at once — it does not move politely. It moves through *leverage*, through *duration*, through *liquidity*, and through the *match (or mismatch) between what you own and what you owe.* Every crisis below is the same machine running, just with a different part snapping first.

![A horizontal timeline of five great bond crises from 1994 to 2023, each labeled with its one-line lesson: duration plus leverage in 1994, credit and counterparty risk in 2008, liquidity in 2020, the correlation flip in 2022, and the duration gap in 2023](/imgs/blogs/the-great-bond-crises-1994-2008-2020-2022-and-2023-1.png)

The diagram above is the mental model for this whole post, and for the series it closes. Five crises, five lessons, one underlying machine. This is the **final post of "The Bond Market, From the Ground Up"** — the capstone. Across forty-one earlier posts we built the bond from scratch: [what a bond is](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer), [why its price moves opposite to rates](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds), [how much it moves](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income), [what credit risk is](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back), and [how institutions match assets to liabilities](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge). Now we watch all of it get stress-tested by reality. (Everything here is educational, not investment advice — the goal is to understand the mechanism, not to call the next crash.)

## Foundations: the four forces that turn a bond into a bomb

Before we walk the five crises, let's rebuild — from zero — the four ideas that explain all of them. If you have read the rest of the series these will be familiar; if this is your first stop, this section is all you need to follow everything below.

### Force 1 — duration: how far a bond falls when rates rise

A **bond** is a tradable loan. You hand over money today; the borrower pays you a fixed stream of cash — periodic **coupons** (the interest) plus your **principal** (the original amount) back at the end, the **maturity** date. Because the cash is fixed, the bond's *price* is just the present value of those payments, and present value falls when the interest rate you discount at rises. That is the seesaw at the heart of fixed income: **when rates go up, bond prices go down.**

The number that tells you *how far* the price falls is **duration** — measured in years. The rule of thumb is the single most useful sentence in all of bond math:

$$\text{price change} \approx -\text{duration} \times \text{change in yield}$$

A bond with a duration of 6 loses about 6% of its value for every 1 percentage point that rates rise. (A *percentage point* is a full point of yield, e.g. from 4% to 5%; a *basis point* is one hundredth of that, 0.01%.) Longer-maturity bonds have higher duration, so they fall harder. A 30-year Treasury can have a duration around 17; a 2-year note, around 2. Same rate move, wildly different damage. We built this number in detail in [duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income), and turned it into dollars in [modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk).

There is a second-order correction called **convexity** — duration is a straight-line approximation, and the true price curve bends — but for the size of the moves in these crises, plain duration already tells the story. (Convexity matters enormously for mortgage bonds, which is its own crisis ingredient; we'll come back to it.)

### Force 2 — leverage: borrowing to own more bonds than your money

**Leverage** means using borrowed money to control more assets than your own cash could buy. If you have \$10 and borrow \$30, you control \$40 of bonds with \$10 of your own equity — that's 4-to-1 leverage. Leverage does not change a bond's duration, but it multiplies the *effect of that duration on your equity.* A 6% price drop on \$40 of bonds is a \$2.40 loss — but that's 24% of your \$10. Leverage turns a flesh wound into a kill shot. The cheapest, most common way bond investors lever up is **repo** (a *repurchase agreement*) — you sell a bond and agree to buy it back tomorrow at a tiny markup, which is economically a one-day loan secured by the bond. Roll it every day and you have permanent cheap borrowing — until the lender wants more collateral.

### Force 3 — credit: the chance you don't get paid back

A US Treasury pays you back because the government can always print dollars to do so; it has, in practice, no **default risk** (the risk the borrower fails to pay). Every other bond carries some chance the borrower can't pay — that is **credit risk**. The market charges for it with a **credit spread**: the extra yield a risky bond pays over a Treasury of the same maturity.

$$\text{corporate yield} = \text{Treasury yield} + \text{credit spread}$$

When investors get scared about defaults, spreads *widen* — risky bonds fall in price even if Treasuries don't move at all. We built this in [credit spreads, pricing the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default). Credit risk also has a hidden cousin, **counterparty risk** — the risk that the institution on the *other side* of your trade or insurance contract fails before it pays you. In a normal market you never think about it. In a crisis it is everything.

### Force 4 — liquidity: can you actually sell?

A bond's price on a screen assumes you can find a buyer. **Liquidity** is the ability to sell a position quickly without crushing the price. The visible cost of liquidity is the **bid-ask spread** — the gap between the price a dealer will buy from you at (the *bid*) and sell to you at (the *ask*). In calm markets that gap is tiny for Treasuries. In a panic, dealers widen it or step away entirely, and "the market price" becomes a fiction — there is no buyer there. Bonds trade in a dealer-based, **over-the-counter** market (negotiated dealer-to-client, not on a public exchange), which makes liquidity especially fragile when everyone wants out the same door at once.

### Putting them together — the funded promise

The last idea ties the others to the institutions that actually blow up. Pensions, insurers, and banks hold bonds not to speculate but to **fund future promises** — a retiree's pension, a depositor's money back, an insurance payout. The art is matching the *duration* of the bonds to the *duration* of the promises, so a rate move hits both sides equally. The gap between them is the **duration gap** (asset duration minus liability duration). A gap of zero is **immunized** — safe against rate moves. A big positive gap is a bet that rates will fall, dressed up as prudence. When that bet loses, the institution can owe more than it owns. This is the subject of [immunization and duration matching](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge), and it is the lesson of 2023.

### The anatomy of a bond crisis — the same four-act play

Before the timeline, it helps to see the *shape* every one of these crises shares, because once you spot it you will recognize the next one. A bond crisis is a four-act play.

**Act one — the calm that breeds the position.** A long stretch of stability teaches investors that a particular bet is safe. In 1994 it was "rates only fall, so borrow and buy long bonds." In 2008 it was "house prices only rise, so mortgage paper is safe." In 2022–23 it was "rates will stay near zero, so reach for yield with duration." The calm is not incidental — it is the *cause*, because it is what lets the dangerous position grow large and crowded.

**Act two — the regime change.** Something the position assumed away actually happens: the Fed hikes by surprise, house prices fall, a pandemic hits, inflation returns. The price of money moves in the direction nobody was positioned for.

**Act three — the amplifier.** The raw price move would be survivable on its own. What turns it into a crisis is an amplifier: **leverage** that forces selling (1994, LDI 2022), a **chain of derivatives** that transmits the loss (2008), a **funding structure that can run** (SVB 2023), or a **market that can't absorb the selling** (2020). The amplifier is what converts a bad year into a blowup.

**Act four — the backstop, or the bankruptcy.** Either a central bank or government steps in to break the loop (the Fed in 2020, the Bank of England in 2022), or the institution dies (Orange County, Lehman, SVB). Whether you get act four as a rescue or a funeral depends on whether you are too systemic to fail — a deeply uncomfortable truth the series does not pretend away.

Hold that four-act structure in mind and re-read the lessons: 1994 is acts one-to-four with leverage as the amplifier; 2008 is the same play with a derivatives chain; 2020 with market structure; 2022 with raw duration and a broken hedge; 2023 with the duration gap. **Different amplifier, same play.**

With those four forces in hand — duration, leverage, credit, liquidity — plus the duration gap that combines them, every crisis below becomes legible. Here is the recurring pattern across all five, on one timeline.

![An illustrative schematic chart with time from 1994 to 2023 on the horizontal axis and stress on the vertical axis, showing a solid line for the ten-year Treasury yield and a dashed line for the credit spread, both spiking at the marked crises: rate spikes in 1994 and 2022, credit spread blowouts in 2008 and 2020, and a smaller funding scare in 2023](/imgs/blogs/the-great-bond-crises-1994-2008-2020-2022-and-2023-2.png)

That figure is the thesis of the series in one picture: the price of money does not move smoothly. It lurches. Sometimes the lurch is in the **yield** itself (1994, 2022 — rate shocks); sometimes it is in the **credit spread** while Treasuries are calm or even rallying (2008, 2020 — panics); sometimes, as in 2023, it is a localized funding scare that never becomes systemic because someone steps in. The shapes are schematic, not to scale — the point is the *pattern*, repeated.

## 1994: the great bond massacre — duration meets leverage

For three years, bonds had been a one-way bet. The Fed had cut its policy rate to 3% to nurse the economy out of the early-1990s recession, and anyone holding long bonds had been richly rewarded as prices rose. The lesson everyone absorbed was: borrow cheap, buy long bonds, collect the yield and the price gains. It felt like free money.

Then, on February 4, 1994, the Fed raised rates by a quarter point — and, crucially, **kept going**, hike after hike, eventually doubling the policy rate to 6% by early 1995. The market had not priced this in. The 10-year Treasury yield climbed from about 5.6% to over 8% across the year. Remember the seesaw: yields up means prices down. Long bonds had high duration, so they fell hard. Globally, the episode wiped out something on the order of a trillion dollars of bond value — a journalist dubbed it the **"great bond massacre."**

A normal investor who simply held long bonds had an ugly year — say a 10%–20% paper loss — but they still owned the bonds and could hold to maturity. The people who got destroyed were the ones who had added the second force: **leverage**.

#### Worked example: the same rate move, levered and unlevered

Picture two investors, each starting with \$7,500,000 of their own money, both buying a portfolio of intermediate Treasuries with a duration of 6.

**Investor A is unlevered.** She buys \$7,500,000 of bonds. Rates rise 3 percentage points over the year. Her price loss is approximately:

$$-\text{duration} \times \Delta y = -6 \times 3\% = -18\%$$

So she loses about \$7,500,000 × 18% = **\$1,350,000**. Painful — but she still has \$6,150,000 and her bonds, and if she holds to maturity she gets every dollar of principal back. A bad year, not a catastrophe.

**Investor B levers 3-to-1.** He puts up the same \$7,500,000 of equity but borrows another \$12,500,000 via repo, buying \$20,000,000 of the *same* bonds. The same 3-point rate rise produces the same 18% price loss — but now on \$20,000,000:

$$-18\% \times \$20{,}000{,}000 = -\$3{,}600{,}000$$

His loss is \$3,600,000, which is *48%* of his \$7,500,000 equity gone in a year — and that is before his repo lenders, watching the collateral fall, demand he post more cash or sell. Forced to sell into a falling market, he realizes losses and can wipe out the rest. Same bonds, same rate move; one investor has a bad year and the other is insolvent.

*The intuition: duration sets how far the bonds fall, but leverage sets whether that fall lands on your savings or on your survival.*

This is exactly what happened to **Orange County, California**. Its treasurer, Robert Citron, ran a \$7.5 billion investment pool for local schools, cities, and districts — and quietly levered it up to roughly \$20 billion of exposure using repo, plus structured notes whose payoffs were themselves leveraged bets that rates would *stay low*. When the Fed hiked, the pool's leveraged long-duration position cratered. Margin calls came. The pool lost about \$1.6 billion, and in December 1994 Orange County filed for **bankruptcy** — at the time the largest municipal bankruptcy in US history. The figure below shows the mechanism that turned a county treasury into a casualty.

![A before-and-after comparison showing an unlevered seven-and-a-half billion dollar bond book losing about eighteen percent for a painful but survivable loss, versus the same bonds levered three times to twenty billion of exposure, where the same eighteen percent price drop produces a loss that exceeds the original equity and renders the fund insolvent](/imgs/blogs/the-great-bond-crises-1994-2008-2020-2022-and-2023-3.png)

What makes Orange County such a clean teaching case is *how invisible* the risk was. Citron's pool reported strong returns for years and was widely admired; local governments competed to put their money in. The leverage was buried in two places a casual observer wouldn't look: the repo financing, which simply doesn't appear as "debt" the way a loan does, and **inverse floaters** — structured notes whose coupon *rises when rates fall and falls when rates rise*, which is a leveraged bet on rates staying low dressed up as an ordinary bond. To an auditor glancing at the holdings, it was a portfolio of high-grade notes. To anyone who computed its *duration*, it was a time bomb with a duration far longer than its stated maturities implied. The lesson there is as much about measurement as about risk: **if you don't compute the duration of the whole levered position, you do not know your risk — and the people who blow up are almost always the ones who didn't.** This is exactly why the series spent so long on [measuring rate risk with modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk).

The 1994 lesson is the foundation for everything that follows: **a bond is not risky because it might default; a high-quality bond is risky because its price moves with rates, and leverage decides whether that move is survivable.** Orange County's bonds were not bad credits — many were Treasuries and agency notes that paid every penny on schedule. The blowup was pure duration-times-leverage. For the policy backdrop of how the Fed sets the rate that triggered it, see [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

## 2008: the global financial crisis — when credit and counterparties break

1994 was about rates. 2008 was about the *other* great force: **credit**. And it showed how credit risk, once it is sliced, re-packaged, leveraged, and insured, can metastasize from a corner of the mortgage market into a global heart attack.

The raw material was the **subprime mortgage** — a home loan to a borrower with weak credit, often with a low "teaser" rate that would reset higher after a couple of years. On its own, one such loan is a small, knowable credit risk. The trouble began with what Wall Street did to thousands of them at once.

### The chain: from a loan to a global panic

Step one, **securitization**: pool thousands of mortgages and sell claims on the pooled cash flows as a bond — a **mortgage-backed security (MBS)**. We covered the machinery in [securitization, turning loans into bonds](/blog/trading/fixed-income/securitization-abs-and-turning-loans-into-bonds). The pool is then sliced into **tranches** by seniority: the **senior tranche** gets paid first and was rated AAA ("safe as Treasuries"); the **equity tranche** absorbs the first losses and pays the highest yield. The seniority logic is the [capital structure](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure) idea applied to a pool.

Step two, **re-securitization**: take the *risky* leftover tranches that nobody wanted, pool *those*, and slice them again into a new structure — a **collateralized debt obligation (CDO)**. Through the alchemy of pooling, the senior slice of a CDO made of junky mortgage tranches got stamped AAA too. Garbage in, gold-rated out — on paper.

Step three, **insurance**: investors who bought the AAA CDO slices, and speculators who wanted to bet against them, used **credit default swaps (CDS)** — a contract where one party pays a premium and the other promises to pay out if the bond defaults. Insurance on a bond, in other words; we explained it in [credit default swaps](/blog/trading/fixed-income/credit-default-swaps-insurance-on-bonds). The insurance giant **AIG** sold tens of billions of dollars of CDS protection on this stuff, collecting premiums and assuming — like everyone — that AAA-rated mortgage paper would never blow up at scale.

The single assumption holding the whole tower up was **low correlation** — the belief that a mortgage in Florida and a mortgage in Nevada would not default for the same reason at the same time. That assumption is what let the rating agencies stamp the senior tranches AAA: if defaults are independent, pooling thousands of them makes the *pool's* loss rate extremely predictable, and the senior slice almost never gets touched. It is the same statistical magic that makes an insurance company safe — as long as your policyholders don't all crash their cars on the same day. The mortgage tower bet, implicitly and enormously, that homeowners across America were like independent drivers.

They were not. A nationwide housing boom is a single, correlated cause. When home prices stopped rising, the teaser rates reset, and subprime borrowers began defaulting *together*, the low-correlation assumption — the keystone — collapsed, and with it every AAA rating that depended on it. The whole tower was built on the assumption that mortgages across the country would not all go bad at once. They did. Now run the chain in reverse: defaults hit the loans → losses tore through the MBS tranches → the CDOs that re-pooled the risky tranches were obliterated → the CDS contracts triggered → **AIG owed far more in payouts than it could pay**, and had to be rescued by the government. And because nobody knew which bank was holding how much of this poison, or who had written CDS to whom, **counterparty trust collapsed.** When **Lehman Brothers** failed in September 2008, banks stopped lending to each other at any price. The credit spread — that measure of default fear — exploded to levels never seen.

![A flow graph of the 2008 securitization chain, starting from subprime loans pooled into mortgage-backed securities, sliced into senior and equity tranches, with the risky tranches re-pooled into a CDO that is insured by credit default swaps sold by AIG, and a parallel path showing defaults flowing into AIG's failure and a system-wide credit freeze after Lehman fails](/imgs/blogs/the-great-bond-crises-1994-2008-2020-2022-and-2023-4.png)

#### Worked example: how a thin equity tranche wipes out

Why did "AAA" mortgage paper lose money at all? Because the senior tranche was only safe if losses stayed *small*. Consider a simplified \$100,000,000 mortgage pool sliced into three tranches:

- **Equity tranche: \$5,000,000** (first 5% of losses)
- **Mezzanine tranche: \$15,000,000** (next 15% of losses)
- **Senior tranche: \$80,000,000** (rated AAA; loses nothing until losses exceed 20%)

In normal times, pool losses might run 1%–2% — \$1,000,000 to \$2,000,000 — all absorbed by the equity tranche, and the senior holders sleep fine. Now suppose the subprime pool suffers **25% losses** — \$25,000,000. The equity tranche (\$5,000,000) is wiped out, the mezzanine (\$15,000,000) is wiped out, and the senior tranche eats the remaining \$5,000,000 — a 6.25% loss on something sold as risk-free. A 6% hit doesn't sound apocalyptic — until you remember banks held these AAA slices *levered 20-to-1 or more*, so a 6% loss on the assets is a 100%+ loss on the equity. Duration's cousin — leverage — strikes again, this time on credit.

*The intuition: pooling does not destroy risk, it concentrates and disguises it; when the losses are correlated, the "safe" senior slice is only as safe as the worst-case loss rate, and leverage turns even a small senior loss into a bank failure.*

There is a duration twist hiding inside the credit story, too, and it connects 2008 back to the duration arc. Mortgage bonds have **negative convexity** — when rates fall, homeowners refinance and pay off early, so the bond's duration *shrinks* just when you'd want it to lengthen; when rates rise, refinancing stops and the duration *extends* just when you'd want it to shorten. So MBS holders got the worst of both worlds: in the panic, the bonds that were supposed to be safe behaved unpredictably as prepayment expectations whipsawed, on top of the credit losses. We unpacked this in [mortgage-backed securities, bonds with negative convexity](/blog/trading/fixed-income/mortgage-backed-securities-bonds-with-negative-convexity). It is a reminder that the four forces are not independent — credit risk and duration risk arrived in the same instrument, compounding each other.

The 2008 lessons are the credit half of fixed income made vivid: **a credit spread is the price of default fear, and counterparty risk is the credit risk you forgot to count.** Ratings are opinions, not guarantees — see [bond ratings, how Moody's, S&P and Fitch grade debt](/blog/trading/fixed-income/bond-ratings-how-moodys-sp-and-fitch-grade-debt) and the agencies' role in [credit rating agencies](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch). And the whole episode shows why the [investment-grade versus high-yield divide](/blog/trading/fixed-income/investment-grade-vs-high-yield-the-great-divide) is not a formality: a label is only as good as the analysis behind it.

## 2020: the dash for cash — when even Treasuries can't be sold

1994 and 2008 were about *value* — bonds being worth less because rates rose or credit cracked. March 2020 was about something more primal and, to many professionals, more shocking: **liquidity.** For a few terrifying days, the safest, deepest, most liquid market on Earth — US Treasuries — stopped working.

When the pandemic shut down the world in March 2020, the first instinct of nearly every institution was the same: **raise cash.** Companies drew down credit lines. Funds faced redemptions. Foreign central banks needed dollars. And to raise cash, you sell what you can — which means you sell your most liquid asset. That asset is Treasuries. So instead of the usual crisis pattern (sell risky assets, *buy* safe Treasuries), the world tried to **sell Treasuries too**, all at once, to get dollars. Everyone ran for the same exit, and the exit was not wide enough.

Here is the mechanism, and it is pure market structure. The dealers who normally stand ready to buy Treasuries from sellers — the big banks — have limited **balance sheet** (capacity to hold inventory, constrained by post-2008 regulation). When the selling overwhelmed that capacity, dealers widened their bid-ask spreads and pulled back. The **bid-ask spread on Treasuries blew out roughly tenfold**, and the price of a Treasury could differ meaningfully depending on whether you wanted to trade the on-the-run (newest) or off-the-run (older) issue — a sign the market was fracturing. A "risk-free" asset was suddenly hard to convert to cash without taking a real hit. Levered players — including some [relative-value hedge funds](/blog/trading/fixed-income/how-bonds-actually-trade-otc-dealers-and-treasury-market-structure) running the "basis trade" — were forced to dump Treasuries into a market that couldn't absorb them, feeding the spiral.

The Federal Reserve ended it with overwhelming force. On March 23, 2020, it pledged to buy **unlimited** quantities of Treasuries and agency MBS — to become the buyer of last resort for the entire market. Within days, the bid-ask spread collapsed back toward normal. The Fed had not changed any bond's *value*; it had restored the *ability to sell*, which in a panic is the only thing that matters. This is the [central bank's plumbing role](/blog/trading/fixed-income/how-central-banks-use-bonds-qe-qt-and-the-plumbing) in extremis.

![An illustrative schematic chart of the Treasury bid-ask spread from February to April 2020, showing it near normal in calm February, blowing out roughly tenfold to a seizure peak in mid-March, then collapsing back toward normal after a dotted line marks the Fed's March twenty-third pledge to buy unlimited Treasuries and mortgage-backed securities](/imgs/blogs/the-great-bond-crises-1994-2008-2020-2022-and-2023-5.png)

#### Worked example: the liquidity haircut on a "risk-free" sale

Suppose your fund needs to raise \$50,000,000 of cash *today* in mid-March 2020, and you hold \$50,000,000 face value of an off-the-run 10-year Treasury. In a normal week, the bid-ask spread on that bond is a sliver — maybe 1/32 of a point, so selling \$50,000,000 costs you roughly \$15,000 in spread. Trivial.

In the seizure, the effective spread to move size in off-the-run issues widened dramatically — say, to a full point or more on the bonds nobody wanted. Selling \$50,000,000 face into that market could cost you **1% = \$500,000** in spread alone, before any price impact from your own selling pressure. The bond's *value* — what it pays at maturity — hadn't changed by a cent. Its *liquidity* had evaporated, and liquidity, when you're a forced seller, is the only price that counts.

*The intuition: a quoted price assumes a buyer exists; liquidity is the difference between a price on a screen and cash in your account, and in a panic that difference can be enormous even for the safest asset on Earth.*

#### Worked example: how the "basis trade" turned forced sellers loose

The deeper reason Treasuries cracked is that liquidity and leverage met. Some hedge funds run the **Treasury "basis trade"**: they buy a cash Treasury and simultaneously sell a Treasury futures contract, capturing the tiny price gap (the *basis*) between them. The gap is minuscule — a few cents — so to make it worthwhile the fund levers the position enormously through repo, often **50-to-1 or more.**

Make it concrete. A fund puts \$100,000,000 of its own capital into a basis position and, at 50-to-1 leverage, controls \$5,000,000,000 of Treasuries, financed in the repo market. The trade earns a thin, steady spread — until March 2020. As Treasury prices gyrated and repo lenders demanded more collateral, the fund faced margin calls it could only meet by **selling its \$5,000,000,000 of cash Treasuries.** Now multiply that by the whole industry running the same trade: tens of billions of forced Treasury selling hit a dealer system that was already out of balance-sheet capacity. A 1% adverse move on \$5,000,000,000 is a **\$50,000,000** loss — *half the fund's capital* — so even a small dislocation forced the unwind.

*The intuition: the safest asset on Earth was made fragile by the leverage stacked on top of it — when liquidity vanished, the levered holders became forced sellers, and forced selling is what turns an illiquid moment into a seizure.*

The 2020 lesson is the one the prior two crises understate: **even a perfectly safe bond can fail you if you cannot sell it when you need to.** This is why Treasury **market structure** — dealers, balance sheet, the on-the-run/off-the-run split — is not a technicality but a load-bearing wall of the financial system, and why the Fed's willingness to backstop it is, quietly, what makes Treasuries "risk-free" in the first place.

## 2022: the worst bond year in modern history — duration, again, and the broken hedge

2022 was 1994's bigger, meaner sibling. The setup was the same — inflation forced the Fed to hike — but the magnitude was historic, and it broke something that 60/40 investors had treated as a law of nature.

After the 2020–2021 stimulus, inflation surged to 40-year highs. The Fed, having held its policy rate near zero, raised it by about **4.25 percentage points in a single year** — the fastest tightening in four decades. Run the seesaw on long bonds. A 30-year Treasury has a duration around 17. A ~2.5-point rise in long yields (yields didn't rise the full policy-rate amount at the long end) does this:

$$-17 \times 2.5\% \approx -42\%$$

Long Treasuries fell on the order of **30%–40%** in 2022 — a stock-market-sized crash in the asset class people buy *for safety*. The broad US bond index fell about 13%, by far its worst year on record. There was nothing exotic here, no subprime, no leverage required: just duration, and a rate move nobody had positioned for. The longer your bonds, the more it hurt — the brutal core of [why bond prices move when rates move, and by how much](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much).

### The correlation flip — when the hedge stops hedging

The deeper shock of 2022 was not that bonds fell. It was that **stocks fell at the same time.** For most of the prior two decades, bonds and stocks were *negatively* correlated — when stocks dropped, investors fled to bonds, bonds rose, and the bond sleeve of a portfolio cushioned the equity losses. That relationship is the entire engine of the classic [60/40 portfolio](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine): 60% stocks for growth, 40% bonds for ballast.

In 2022 the ballast became an anchor. Because the *cause* of the pain was high inflation and rising rates, **both** assets fell together: higher discount rates hammered long-duration stocks (especially tech) *and* hammered bonds directly. The stock-bond correlation flipped positive. A 60/40 portfolio had its worst year in a century, down roughly 17%, because both legs broke at once. The case study of that year is told from the allocation side in [2022, when stocks and bonds both fell](/blog/trading/cross-asset/case-study-2022-stocks-and-bonds-both-fell).

The deep mechanism is worth making explicit, because it explains *when* the hedge works and when it breaks. A stock is worth the present value of its future profits, and a bond is worth the present value of its future coupons — both are discounted at an interest rate. So **the rate is a common input to both prices.** What usually saves the 60/40 portfolio is that stocks and bonds get hit by *different* shocks: in a growth scare, profits fall (stocks down) but the Fed cuts rates (bonds up), and the two offset. The correlation is negative because the dominant shock is to *growth*. But when the dominant shock is to *rates and inflation* — as in 2022 — the common input moves against both at once: higher discount rates crush the present value of distant profits *and* the present value of distant coupons. The hedge depends on which kind of shock is in charge, and inflation is precisely the regime in which it fails. This is the discount-rate logic we built in [bonds versus stocks, discount rates, the 60/40 and correlation](/blog/trading/fixed-income/bonds-vs-stocks-discount-rates-the-60-40-and-correlation).

#### Worked example: the long-bond holder versus the short-bond holder in 2022

Two retirees each hold \$1,000,000 of Treasuries, but in different maturities, going into 2022.

**Retiree Long** holds 30-year Treasuries, duration ≈ 17. Long yields rise ~2.5 points. Her price loss:

$$-17 \times 2.5\% \approx -42.5\% \;\Rightarrow\; \text{about } \$425{,}000 \text{ lost}$$

**Retiree Short** holds 2-year Treasuries, duration ≈ 2. Short yields rise more — say ~4 points — but the duration is tiny:

$$-2 \times 4\% \approx -8\% \;\Rightarrow\; \text{about } \$80{,}000 \text{ lost}$$

Same asset class, same "safe" Treasuries, same year — and a five-fold difference in damage, entirely because of duration. And note the cruel twist for Retiree Long: if she holds to maturity she still gets her \$1,000,000 principal back, but she has locked in low coupons for decades and watched inflation erode the real value of every payment. There was nowhere comfortable to hide in 2022 — which is exactly why it shattered the "bonds are safe" reflex.

*The intuition: in a rate shock, "bonds" is not one thing — duration determines who gets hurt, and when the shock is driven by inflation, the bond sleeve can fail at the very moment the stock sleeve needs it most.*

The 2022 lesson closes the duration arc the series built: **duration is the master risk of a bond, and the stock-bond hedge is a correlation, not a constant — it can flip exactly when you're counting on it.** For why inflation is the variable that decides the regime, see [real versus nominal and the real-yield master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) and [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything).

## 2023: SVB and the UK gilt crisis — the duration gap comes due

The 2022 rate shock didn't end in 2022. Its bill arrived in 2023, at the institutions that had quietly let their **duration gap** blow open while rates were low — and it arrived twice, on two continents, in two different disguises. Both are the [immunization](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge) lesson, learned the hard way.

### Silicon Valley Bank — an unmanaged duration gap funded by deposits that could run

A bank's business is a maturity and duration mismatch by design: it borrows short (deposits you can withdraw anytime) and lends/invests longer. The job — the *whole* job of bank treasury risk management — is to keep that mismatch within bounds. **Silicon Valley Bank** did the opposite. Flooded with deposits from the 2020–2021 tech boom, it poured tens of billions into **long-dated Treasuries and agency MBS** when yields were near record lows. That gave it a large pile of long-duration assets — call it duration ~6 across the book — funded by deposits that, crucially, **could leave in an afternoon** and were mostly above the \$250,000 insurance limit, so depositors had every reason to flee at the first sign of trouble.

Here is the accounting sleight-of-hand that hid the danger. Banks can classify bonds as **held-to-maturity (HTM)**, which lets them carry the bonds at original cost on the books *even as the market value falls.* So as 2022's rate shock cut the market value of SVB's long bonds by roughly a quarter, the reported balance sheet barely flinched — the losses were real but **unrealized**, invisible in the headline numbers. The duration gap was enormous and growing, but the accounting let everyone pretend it wasn't.

The trap sprung in March 2023. SVB needed to raise cash (deposits were slowly leaving as the tech boom cooled), which forced it to *sell* some of those underwater bonds — crystallizing a real loss and revealing the hole. Word spread through a tightly networked depositor base, and the run was instant and digital: roughly **\$42 billion of deposits tried to leave in a single day**, March 9, 2023. Forced to sell the rest of its bond book at a loss far larger than its entire equity, SVB was insolvent. The FDIC took it over within 48 hours. It was the second-largest bank failure in US history at the time — and it was, at its core, **a textbook unmanaged duration gap**, the exact failure mode the immunization post warned about. (The 2023 bank-run episode, including Credit Suisse, is covered in [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

![A before-and-after comparison of Silicon Valley Bank, showing in 2021 a roughly one hundred twenty billion dollar book of long bonds with duration around six funded by mostly uninsured on-demand deposits and a hidden duration gap, then in 2023 the bonds repriced down twenty-four percent as rates rose four percent, deposits fleeing forty-two billion in one day, and a mark-to-market loss larger than the bank's capital that causes failure](/imgs/blogs/the-great-bond-crises-1994-2008-2020-2022-and-2023-6.png)

#### Worked example: sizing an SVB-style mark-to-market loss

Let's size the damage the way a risk manager should have. Take a stylized bank with a **\$100,000,000,000 bond book, duration 6**, and roughly **\$16,000,000,000 of equity** (a typical ~16-to-1 leverage for a bank). Now walk the rate rise of 2022 up in steps using the duration rule, $\text{loss} \approx -\text{duration} \times \Delta y \times \text{book size}$:

- **Rates +1%:** −6 × 1% × \$100bn = **−\$6,000,000,000** — about 38% of equity gone.
- **Rates +2%:** −6 × 2% × \$100bn = **−\$12,000,000,000** — about 75% of equity gone.
- **Rates +3%:** −6 × 3% × \$100bn = **−\$18,000,000,000** — equity *wiped out*; the bank is insolvent on a mark-to-market basis.
- **Rates +4%:** −6 × 4% × \$100bn = **−\$24,000,000,000** — an **\$8,000,000,000 hole beyond equity.**

The 2022 hiking cycle was about +4.25%. So a bank that let its book sit at duration 6 against a thin equity cushion was, on a mark-to-market basis, deeply insolvent by late 2022 — *the only question was whether it would ever be forced to sell and reveal it.* HTM accounting let SVB delay that reckoning; a deposit run is what forced it.

![A matrix showing, for a one hundred billion dollar bond book with duration six, how each one percent rate rise produces roughly a six billion dollar mark-to-market loss, escalating from a six billion loss at plus one percent that is thirty-eight percent of equity, to a twenty-four billion loss at plus four percent that is an eight billion dollar hole beyond a sixteen billion equity cushion](/imgs/blogs/the-great-bond-crises-1994-2008-2020-2022-and-2023-7.png)

*The intuition: a duration gap is a hidden short position on rates; HTM accounting can hide the loss on paper, but it cannot make the loss go away, and a deposit that can run is the worst possible funding for a long-duration asset.*

### The UK gilt/LDI crisis — immunization with leverage that margin-called itself

The British version, in September–October 2022, is the most ironic of all the crises, because it struck institutions doing the *most conservative* thing: **hedging.** UK defined-benefit pension funds had long-dated liabilities (pensions owed decades out), which behave like very long-duration bonds. To immunize — to make the assets move with the liabilities — they ran **liability-driven investing (LDI)**: hold long gilts (UK government bonds) and, to get enough duration without tying up all their cash, use **leverage and interest-rate derivatives** so a small amount of capital controlled a large amount of duration exposure.

This is textbook immunization, and in slow-moving markets it works. The flaw was the *leverage*. The derivatives and repo that gave the funds their duration required them to **post collateral (margin)** when gilt prices moved against them. After the UK government's September 2022 "mini-budget" announced large unfunded tax cuts, gilt yields spiked violently — the 30-year gilt yield rose more than a full point in *days*. Gilt prices crashed, and the LDI funds faced enormous, sudden **margin calls.** To raise cash for margin, they had to **sell gilts** — which pushed gilt prices *down further*, which triggered *more* margin calls. A self-reinforcing **doom loop**: the hedge was forcing the very selling that was breaking it.

The **Bank of England** had to step in with emergency gilt purchases to stop the spiral — restoring liquidity exactly as the Fed had in 2020, for the same reason: not to change gilt *values*, but to break a forced-selling loop before it took down the pension system. The crisis is the cleanest illustration in history that **immunization done with leverage is only as safe as your ability to meet the margin calls in a panic.**

#### Worked example: the LDI margin spiral

Suppose an LDI fund has \$1,000,000,000 of liabilities to hedge and, rather than buy \$1,000,000,000 of long gilts outright, uses leverage to control \$1,000,000,000 of gilt duration with only \$200,000,000 of capital posted as collateral (5-to-1). The position has a duration of, say, 20.

Now 30-year gilt yields jump **1 percentage point** in days. The mark-to-market loss on \$1,000,000,000 of duration-20 exposure is:

$$-20 \times 1\% \times \$1{,}000{,}000{,}000 = -\$200{,}000{,}000$$

That single move erases the *entire* \$200,000,000 of posted collateral. The counterparties demand the fund top it back up — another \$200,000,000 of cash, *now.* The only liquid thing the fund can sell to raise it is more gilts — into a market where every other LDI fund is doing exactly the same thing. Their combined selling drives gilt yields up *further*, generating the next round of margin calls. Without an outside buyer, the loop does not stop on its own.

*The intuition: leverage turns a prudent hedge into a margin engine — immunization protects you against rate moves only if you can survive the cash demands those moves create along the way.*

The 2023 lessons close the series' asset-liability arc: **the duration gap is the master risk of an institution, immunization is the cure, and leverage is the thing that turns the cure back into the disease.** It is the [immunization](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge) post and the [duration](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) post, fused, and stress-tested to failure.

## What the five crises share — the series in one table

Step back and the five episodes line up as a single argument. Each one took a concept this series built and demonstrated it the only way the lesson truly sticks: by destroying someone who ignored it.

| Crisis | The trigger | The amplifier | The concept it stress-tested | Who stepped in |
|---|---|---|---|---|
| **1994 / Orange County** | Surprise Fed hikes (3% → 6%) | Repo leverage ~3x | Duration × leverage | Nobody — bankruptcy |
| **2008 / GFC** | Subprime defaults, correlated | Securitization + CDS + bank leverage | Credit & counterparty risk | Fed/Treasury bailouts |
| **2020 / dash for cash** | Pandemic scramble for dollars | Dealer balance-sheet limits, basis-trade leverage | Liquidity & market structure | Fed (unlimited purchases) |
| **2022 / the rout** | Inflation, +4.25% in a year | Long duration, broken hedge | Duration risk & the correlation flip | Nobody — a market loss |
| **2023 / SVB & LDI** | The 2022 rate rise's delayed bill | Runnable deposits; LDI margin loops | The duration gap & immunization | FDIC; Bank of England |

Read down the "concept" column and you have read the spine of the series: duration, then credit, then liquidity, then the correlation that links bonds to everything else, then the asset-liability match that decides who survives. Read down the "amplifier" column and you find the same villain in four of five rows — **leverage**, in its many disguises (repo, securitization, derivatives, runnable funding). And read down the last column and you confront the uncomfortable lesson of modern crises: **the price of money is so central that the state cannot let its market fully break** — which is both a reassurance and a moral hazard the system still hasn't resolved.

There is one more pattern worth naming, because it is the influence thesis of the entire series in its sharpest form. In *every* case, the crisis did not stay in the bond market. 1994 bankrupted a county government and its schools. 2008 became a global recession and millions of lost jobs. 2020 threatened to freeze corporate funding for the real economy. 2022 cut the retirement savings of ordinary 60/40 investors worldwide. 2023 took down banks and threatened a pension system. **Because bonds are the price of money, a bond crisis is never only a bond crisis** — it is the price of money breaking, and the price of money is the input to every other price there is.

## Common misconceptions

**"Government bonds are safe, so they can't cause a crisis."** Four of these five crises were driven primarily by *government* bonds — Treasuries and gilts. Default risk is only one kind of bond risk, and for sovereigns in their own currency it's near zero. But **duration risk** (price falls when rates rise) and **liquidity risk** (you can't sell when you need to) are very much alive in Treasuries, and they did the damage in 1994, 2020, 2022, and SVB in 2023. "Safe from default" is not "safe from loss."

**"If I hold to maturity, I can't lose, so mark-to-market losses don't matter."** This is the most dangerous half-truth in fixed income. It's true that a default-free bond held to maturity returns its principal — but it ignores two things. First, **inflation**: getting \$1,000 back in 30 years after rates and prices have risen can mean a brutal real loss, as 2022's long-bond holders learned. Second, and fatally, **you may not get to hold to maturity.** SVB's whole tragedy is that a deposit run *forced* it to sell underwater bonds before maturity. Unrealized losses become very real the moment someone makes you sell.

**"Leverage is for reckless gamblers; conservative investors don't use it."** Orange County was a municipal pool for schools. UK LDI funds were pensions *hedging* their risk. Both used leverage — repo and derivatives — in the name of prudence, and both detonated. Leverage is woven invisibly through the supposedly safe corners of fixed income (repo, structured notes, derivative overlays), and it is precisely *because* it hides inside conservative strategies that it keeps causing crises.

**"AAA means safe."** 2008 buried this. A AAA rating is an *opinion* about default probability under a model, and the models for structured mortgage products assumed losses would not be highly correlated across the country. When they were, AAA CDO tranches lost most of their value. A rating is an input to your own analysis, not a substitute for it — see [bond ratings](/blog/trading/fixed-income/bond-ratings-how-moodys-sp-and-fitch-grade-debt).

**"A liquid market is always liquid."** The Treasury market is the deepest in the world — and in March 2020 it seized for days. Liquidity is a *property of conditions*, not a fixed attribute of an asset. It is most abundant when you don't need it and evaporates exactly when everyone wants it at once. The only reliable backstop has turned out to be a central bank willing to buy without limit.

**"Diversification across stocks and bonds always protects you."** The 60/40 portfolio works because stocks and bonds *usually* move opposite each other. But that's a correlation, and 2022 showed it can flip positive — when inflation is the driver, both assets fall together. Diversification is real but conditional; it is weakest in exactly the inflation-driven regime where you most want it.

## How it shows up in real markets

**Orange County, 1994 (~\$1.6 billion loss, bankruptcy).** Treasurer Robert Citron levered a \$7.5 billion municipal pool to ~\$20 billion of duration exposure via repo and structured notes that bet rates would stay low. The Fed's surprise hiking cycle (3% to 6%) crushed the position; margin calls forced selling; the county filed the largest municipal bankruptcy of its era. The lesson is pure duration-times-leverage — the bonds themselves were high quality and mostly paid in full.

**Long-Term Capital Management, 1998 (a related epilogue to 1994's lesson).** A hedge fund run by Nobel laureates levered fixed-income relative-value trades ~25-to-1. When Russia defaulted and spreads moved against them, the leverage that had produced steady gains produced catastrophic losses, and the Fed organized a private bailout to prevent a chain reaction. Different bonds, same machine: small edges, enormous leverage, a tail event that turned the leverage lethal — the [credit and counterparty](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) dimension of the 1994 lesson.

**The 2008 global financial crisis (trillions in losses; a global recession).** Subprime mortgages → MBS → CDOs → CDS, insured at scale by AIG, held levered by banks. Correlated defaults broke the chain, AIG needed an ~\$182 billion rescue, Lehman failed, and interbank credit froze. This is the credit-and-counterparty lesson at full scale, and the reason post-crisis rules forced more capital and central clearing onto the system.

**The March 2020 dash for cash (Fed pledged unlimited purchases on March 23).** A global scramble for dollars made even Treasuries hard to sell; bid-ask spreads blew out ~tenfold; levered basis-trade funds were forced sellers. The Fed's unlimited-purchase pledge restored function within days. The lesson — that the world's safest market depends on dealer balance sheet and a central-bank backstop — reshaped the debate about Treasury market structure.

**The 2022 bond rout (broad bond index −13%, long Treasuries −30%+, 60/40 ≈ −17%).** The fastest Fed tightening in 40 years (+4.25%) and an inflation shock drove the worst bond year in modern history and flipped the stock-bond correlation positive, breaking the 60/40 hedge. It re-taught a generation that "bonds are safe" depends entirely on duration and on which regime you're in. See the allocation-side telling in [2022, when stocks and bonds both fell](/blog/trading/cross-asset/case-study-2022-stocks-and-bonds-both-fell).

**Silicon Valley Bank, March 2023 (~\$42 billion deposit run in a day; FDIC takeover).** A large long-duration bond book (HTM-classified, so losses stayed hidden) funded by uninsured on-demand deposits — a textbook unmanaged duration gap. The 2022 rate rise made the book deeply underwater; a forced sale revealed the loss; a digital deposit run finished it in 48 hours. The cleanest modern illustration of the [immunization](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge) lesson.

**The UK gilt/LDI crisis, September–October 2022 (Bank of England emergency gilt purchases).** Pension funds immunizing their liabilities with *leveraged* gilt exposure faced a margin doom loop when the "mini-budget" spiked gilt yields: margin calls forced gilt sales, which spiked yields further, which forced more sales. The Bank of England's intervention broke the loop. It is the definitive case study that immunization plus leverage is only as safe as your liquidity to meet margin under stress.

## When this matters to you

You will probably never run a \$20 billion municipal pool or an LDI book. But every one of these forces touches ordinary money. The duration of your bond fund decides how much it falls when rates rise — the same arithmetic that sank Retiree Long in 2022 governs the bond sleeve of your retirement account. The "held-to-maturity" comfort that lulled SVB is the same comfort that makes people ignore paper losses until they're forced to sell. And the reason your bank deposit feels safe at all is that, after 2008 and 2023, regulators and the Fed stand behind the plumbing these crises exposed.

The deeper takeaway — the thesis of this entire series, now closing — is that **bonds are the price of money, and the price of money sets every other price.** When that price moves violently, it moves through the four forces we built from zero: duration (how far prices fall), leverage (whether the fall is survivable), credit (whether you get paid back at all), and liquidity (whether you can sell when you need to) — and through the duration gap that combines them inside every bank, insurer, and pension. Master those four forces and the entire history of financial crises stops looking like a series of unrelated disasters and starts looking like one machine, running again and again, breaking at whichever part was weakest that decade.

That is where this series ends, and where your own understanding of markets can begin. If you want to go deeper on the policy lever behind the rate shocks, read [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [the central bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance); for the heavy math under duration and the curve, [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics) and [yield-curve modeling](/blog/trading/quantitative-finance/yield-curve-modeling); and for the institution whose name is synonymous with bond investing through these very crises, [PIMCO and the bond market](/blog/trading/finance/pimco-and-the-bond-market). The bond market built every other market. Now you know how it breaks — and why, every time, it is the same five lessons wearing a new disguise.
