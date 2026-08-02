---
title: "Mortgage-Backed Securities: Pass-Throughs, Prepayment, and Convexity"
date: "2026-08-01"
publishDate: "2026-08-01"
description: "A first-principles guide to mortgage-backed securities, agency guarantees, prepayment risk, CMOs, duration, convexity, and the mortgage market's current structure."
tags: ["structured-finance", "mortgage-backed-securities", "mbs", "mortgages", "prepayment-risk", "convexity", "fixed-income", "securitization"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 27
---

> [!important]
> **TL;DR** — An MBS is a claim on mortgage cash flows, and its hardest risk is that borrowers can change the timing of principal repayments.
>
> - Mortgage borrowers can prepay when rates fall and refinance, or remain locked in when rates rise.
> - A pass-through distributes principal and interest from a mortgage pool, after servicing and guarantee fees.
> - Agency MBS and private-label MBS have different credit-risk structures.
> - CMOs redistribute prepayment and extension risk across tranches; they do not make the risk disappear.
> - The key numbers are not only yield and duration, but also prepayment speed, weighted-average life, spread duration, and convexity.

The most important feature of a mortgage is that the borrower usually has an option. If rates fall, the homeowner can refinance or sell the house and repay the mortgage. If rates rise, the borrower can keep the old cheap loan while the investor is left holding a security whose cash flows last longer than expected.

That option changes the behavior of mortgage-backed securities. An MBS is not simply a corporate bond with a fixed maturity. Its principal can arrive faster when investors least want reinvestment risk, and slower when investors most want their money back.

![Mortgage borrowers generate principal and interest that flow through a mortgage pool into MBS investors](/imgs/blogs/mortgage-backed-securities-mbs-explained-1.webp)

This article builds the product from zero. We will separate mortgage credit risk from prepayment risk, explain agency guarantees, calculate weighted-average life, and show why mortgage bonds often have negative convexity.

## Foundations: mortgage loans and MBS cash flows

### A mortgage is collateralized debt

A residential mortgage is a loan secured by real property. The borrower receives money to buy or refinance a home and promises monthly principal and interest. The lender receives a lien on the property. If the borrower defaults, the lender may foreclose, subject to law and time.

The mortgage payment contains interest and principal. Early payments usually contain more interest because the outstanding balance is larger. Over time, scheduled principal reduces the balance.

#### Worked example: a simple mortgage payment

Suppose a borrower takes a \$300,000, 30-year fixed-rate mortgage at 6% annual interest. Ignoring taxes, insurance, and fees, the monthly rate is 0.5% and the contractual payment is approximately \$1,799.

1. First-month interest is \$300,000 × 0.5% = \$1,500.
2. Approximate first-month scheduled principal is \$1,799 − \$1,500 = \$299.
3. The balance after the first payment is approximately \$299,701.
4. Later payments contain more principal because interest is calculated on a smaller balance.

The exact payment depends on the amortization formula and rounding. The important intuition is that scheduled cash flows are only one part of the outcome; prepayment can change the schedule.

### From mortgages to a pass-through

A mortgage pool combines many loans. A pass-through security gives investors a claim on the pool's collected principal and interest, less servicing, guarantee, and administrative fees.

The security does not normally pass through each borrower payment one-for-one at the exact moment it is made. The servicer collects payments, reconciles delinquencies and advances, and distributes cash according to the pool documents.

### Agency and private-label MBS

Agency MBS are issued or guaranteed through government-sponsored enterprises or government agencies. The guarantee structure affects credit risk, but it does not remove interest-rate, prepayment, liquidity, or operational risk.

Private-label MBS are issued without the same agency guarantee. They may use subordination, excess spread, reserve accounts, representations, and other enhancements to allocate mortgage credit risk.

![Agency and private-label MBS use different mechanisms to allocate mortgage credit and cash-flow risk](/imgs/blogs/mortgage-backed-securities-mbs-explained-2.webp)

## 1. Mortgage servicing and the cash-flow chain

The servicer collects monthly payments, manages escrow, handles delinquencies, processes modifications, and reports performance. A master servicer or trustee may oversee the transaction, while a subservicer performs day-to-day work.

Servicing matters because the investor's cash is not produced by a formula alone. Borrowers pay, move, refinance, miss payments, request forbearance, or enter foreclosure. The servicer determines how those events become data and cash.

### Advances and delinquencies

Agency structures may use advances or guarantees to stabilize investor cash flows. Private-label transactions define their own advance obligations. An advance can protect timing but creates a claim that must eventually be reimbursed.

### Escrow and non-interest cash

Mortgage payments may include taxes and insurance escrow. These amounts are not the same as interest available to MBS investors. A cash report must separate borrower payment components.

#### Worked example: pass-through deduction

Suppose a mortgage pool receives \$1,000,000 of scheduled interest and \$500,000 of scheduled principal in a month. Servicing and guarantee fees total \$25,000.

1. Gross interest: \$1,000,000.
2. Less fees: \$25,000.
3. Net interest passed through: \$975,000.
4. Principal passed through: \$500,000, subject to any permitted adjustments.

The investor's yield is based on net cash, not the mortgage coupon alone.

## 2. Prepayment risk: the borrower's embedded option

Prepayment is principal paid before its contractual schedule. A borrower can refinance, sell the property, make extra principal payments, or use a permitted payoff option.

When market mortgage rates fall, refinancing becomes more attractive. When rates rise, borrowers with old low-rate mortgages tend to remain in place. This creates an asymmetric relationship between rates and prepayment.

![Falling rates can accelerate mortgage prepayments while rising rates can extend the life of an MBS](/imgs/blogs/mortgage-backed-securities-mbs-explained-3.webp)

### CPR and SMM

The **conditional prepayment rate**, or CPR, expresses an annualized prepayment assumption. The **single monthly mortality**, or SMM, expresses the equivalent monthly rate. A common conversion is:

\[
\text{SMM} = 1 - (1 - \text{CPR})^{1/12}
\]

This equation is a mathematical conversion, not a guarantee of actual borrower behavior.

#### Worked example: converting CPR to SMM

Suppose the assumed CPR is 12%.

1. Monthly survival rate is approximately \((1 - 0.12)^{1/12}\).
2. SMM is one minus that survival rate.
3. The resulting SMM is approximately 1.06%.

The precise number depends on rounding. The intuition is that an annualized prepayment assumption must be translated into monthly principal behavior.

### Refinance incentive

The borrower compares the current mortgage rate with the available refinance rate. The difference must be large enough to compensate for closing costs, hassle, expected time in the home, and credit or property constraints.

A pool's refinance incentive varies by loan. The average mortgage rate can hide a distribution of borrowers: some are deeply out of the money, some are close to refinancing, and some cannot refinance despite an apparent rate incentive.

### Turnover and burnout

Borrowers also prepay when they sell homes. Turnover depends on employment, household formation, local housing supply, seasonality, and mobility. **Burnout** describes the idea that borrowers who did not refinance during an earlier opportunity may be less likely to refinance during a later, similar opportunity.

## 3. Duration, weighted-average life, and extension

The maturity date of an MBS is not the same as its expected life. An MBS may legally mature in 30 years but be expected to pay principal over 5 or 10 years depending on prepayments.

### Weighted-average life

Weighted-average life measures the average time at which principal is received, weighted by principal amount:

\[
\text{WAL} = \frac{\sum_t t \times \text{principal paid at }t}{\text{total principal paid}}
\]

This is an explanatory formula for the concept. The exact convention depends on the market and security.

#### Worked example: weighted-average life

Suppose a \$1 million MBS pays \$400,000 of principal at the end of year 1, \$300,000 at the end of year 2, and \$300,000 at the end of year 4.

1. Year-1 weighted principal: 1 × \$400,000 = \$400,000.
2. Year-2 weighted principal: 2 × \$300,000 = \$600,000.
3. Year-4 weighted principal: 4 × \$300,000 = \$1.2 million.
4. Total weighted principal: \$2.2 million.
5. WAL: \$2.2 million ÷ \$1 million = 2.2 years.

### Extension risk

When rates rise, prepayments can slow. Principal arrives later, duration increases, and the investor remains exposed to a higher-rate environment and potentially wider spreads. This is extension risk.

### Contraction risk

When rates fall, prepayments can accelerate. Principal arrives sooner, and the investor must reinvest at lower yields. This is contraction or reinvestment risk.

## 4. Negative convexity

For a plain option-free bond, falling yields usually increase price more than rising yields reduce it because of positive convexity. An MBS contains a borrower prepayment option. That option makes the security's cash flows change as rates move.

When rates fall, the MBS may pay back faster and lose some of the benefit of the lower discount rate. When rates rise, it may pay back slower and become more rate-sensitive. The price-yield curve bends in an unfavorable direction: negative convexity.

![Mortgage prepayment options create negative convexity by shortening cash flows when rates fall and extending them when rates rise](/imgs/blogs/mortgage-backed-securities-mbs-explained-6.webp)

#### Worked example: duration changes with rates

Suppose an MBS has an expected duration of 4 years when rates are stable.

1. If rates fall and prepayments accelerate, expected duration may decline to 2.5 years.
2. If rates rise and prepayments slow, expected duration may extend to 6 years.
3. The same security therefore has different rate sensitivity in different scenarios.

The figures are illustrative. The exact duration depends on coupon, seasoning, loan characteristics, prepayment model, and market assumptions.

## 5. CMOs: redistributing principal timing

A collateralized mortgage obligation, or CMO, divides a mortgage pool into classes with different principal priorities. The goal may be to create tranches with more predictable cash-flow windows.

### Sequential-pay CMO

In a sequential structure, principal pays Class A first, then Class B, then Class C. Class A receives principal quickly and becomes less exposed over time. Class C receives principal later and bears more extension risk.

![A sequential-pay CMO distributes mortgage principal to earlier classes before later classes](/imgs/blogs/mortgage-backed-securities-mbs-explained-5.webp)

### PAC and support tranches

A planned amortization class, or PAC, is designed to receive principal within a target range of prepayment speeds, supported by a companion or support tranche. If prepayments move outside the assumed range, the support tranche absorbs more variability and the PAC can lose its schedule protection.

### IO and PO strips

An interest-only, or IO, strip receives interest cash flows without ordinary principal. A principal-only, or PO, strip receives principal without the same interest stream. Their values respond differently to prepayment. IO holders can be hurt by fast prepayment because the interest base disappears. PO holders may benefit from faster principal.

## 6. Agency guarantees and mortgage credit risk

An agency guarantee can shift or reduce investor exposure to borrower default, depending on the exact program and conditions. It does not guarantee market price, liquidity, prepayment behavior, or the investor's reinvestment rate.

Agency MBS also depend on the soundness of the guaranteeing framework and the legal treatment of the securities. Investors should read the guarantee language rather than assume that all “agency” products have identical protection.

Private-label MBS use capital stacks and credit enhancement. Senior-subordinate structures place mezzanine and equity below senior notes. Credit risk can be substantial in the lower classes even when the collateral is diversified.

## 7. Current mortgage securitization data

FHFA states that Fannie Mae and Freddie Mac provide more than \$8.5 trillion in funding to U.S. mortgage markets and financial institutions as of the 2025 Scorecard context. This is a broad funding figure, not a claim that all of it is outstanding MBS held by one investor. [FHFA 2025 Scorecard](https://www.fhfa.gov/news/news-release/fhfa-releases-2025-scorecard-for-fannie-mae-freddie-mac-and-common-securitization-solutions)

Fannie Mae reported \$409 billion of liquidity provided to the U.S. mortgage market in 2025, including approximately \$133 billion of single-family MBS issued through whole-loan conduit transactions and \$276 billion through single-family and multifamily lender swaps. These are Fannie Mae reported figures for 2025, not total U.S. mortgage issuance. [Fannie Mae 2025 Annual Housing Activities Report](https://www.fanniemae.com/media/56766/display)

Freddie Mac reported issuing \$48 billion of fully guaranteed multifamily securitizations and \$18.8 billion of senior-subordinate securitizations during 2025. These figures are specific to the report's multifamily activities. [Freddie Mac 2025 Annual Housing Activities Report](https://www.freddiemac.com/about/pdf/2025-annual-housing-activities-report.pdf)

FHFA's Q2 2025 Prepayment Monitoring Report explains that prepayment alignment across Fannie Mae and Freddie Mac cohorts is important for the liquidity and fungibility of UMBS. [FHFA Prepayment Monitoring Report](https://www.fhfa.gov/reports/prepayment-monitoring-report/2025/Q2)

## 8. Mortgage cash-flow modeling

An MBS model begins with loan characteristics: coupon, balance, maturity, amortization, seasoning, geography, credit quality, occupancy, loan purpose, and refinance incentive. It then projects prepayments, scheduled principal, defaults, recoveries, servicing, and guarantee fees.

The model should produce both cash flows and risk measures. Useful outputs include price, yield, duration, convexity, WAL, expected principal timing, option-adjusted spread, and scenario-specific returns.

### Model risk

Prepayment models can be wrong because borrower behavior changes. A model trained on one rate regime may fail when housing turnover, refinancing technology, borrower equity, or underwriting changes. The analyst should test assumptions rather than treat a model output as a fact.

#### Worked example: model disagreement

Suppose one model assumes a \$100 million pool has 10% CPR and another assumes 20% CPR.

1. The higher-prepayment model returns principal faster.
2. Faster principal lowers expected WAL.
3. The higher-prepayment model may increase price for a discount MBS but reduce price for an IO strip.
4. A single “fair value” is incomplete without the prepayment assumption.

## 9. TBA liquidity and uniform securities

The To-Be-Announced market allows investors to trade agency MBS before identifying the exact pools that will be delivered. Standardization improves liquidity, but it does not remove prepayment differences across cohorts.

Uniform Mortgage-Backed Securities, or UMBS, were introduced to make Fannie Mae and Freddie Mac securities more fungible. FHFA monitors prepayment alignment because materially different borrower behavior can create different cash flows even when securities share coupon and maturity labels.

![The mortgage securitization ecosystem links originators, agencies, servicers, investors, and the TBA market](/imgs/blogs/mortgage-backed-securities-mbs-explained-7.webp)

The distinction is important: standardization is a market-infrastructure benefit, not a promise that every pool has identical prepayment behavior.

## 10. MBS basis and hedging

An MBS investor may hedge duration with Treasury futures, swaps, or other instruments. The hedge can be imperfect because mortgage duration changes as rates move. This is mortgage basis risk.

![MBS hedges must adapt because mortgage duration changes when prepayments speed up or slow down](/imgs/blogs/mortgage-backed-securities-mbs-explained-4.webp)

When rates fall, the MBS may shorten while the Treasury hedge behaves differently. When rates rise, the MBS may extend. A static hedge can therefore become wrong precisely when volatility rises.

### Dollar duration and convexity management

Portfolio managers monitor the change in price for a small rate movement and the curvature of that response. They may use dynamic hedges, swaptions, or mortgage-specific instruments to manage the changing exposure.

The cost of hedging is part of the investment return. A higher nominal MBS yield does not automatically mean a higher hedged return.

## 11. Private-label MBS and the 2008 lesson

Private-label MBS expose investors to mortgage credit, representations, servicing, foreclosure, and legal risk. Before 2008, structures relied on historical relationships that did not survive a nationwide housing shock. Defaults became correlated, recoveries fell, and liquidity disappeared.

The lesson is not that mortgage securitization is inherently defective. It is that geographic diversification, model-based ratings, and historical loss data can fail together when the common factor is national house prices and credit availability.

## 12. MBS and M&A platform strategy

Mortgage finance is increasingly a platform business. A platform may own origination channels, mortgage servicing rights, securitization technology, data, capital-markets distribution, and insurance or guarantee capabilities.

Fannie Mae's 2025 report described a model that combines whole-loan purchases, MBS swaps, securitization, and secondary-market liquidity. The existence of a standardized platform shows why acquisitions of mortgage technology, servicing, and specialty finance can matter even when the buyer is not directly acquiring an MBS portfolio.

The M&A analyst should ask whether a deal adds:

- servicing scale;
- mortgage data;
- low-cost funding;
- origination distribution;
- risk-transfer capacity;
- technology for loan-level reporting;
- access to insurance and pension capital.

## 13. Mortgage loan characteristics that drive prepayment

Two pools with the same coupon can have very different prepayment behavior. The borrower, the loan, the property, and the local market all matter.

### Loan age and seasoning

Newly originated loans often have different prepayment behavior from seasoned loans. A borrower may refinance soon after origination if rates fall, but some loans have closing costs or operational friction that delays the decision. Seasoning also interacts with burnout: borrowers who already refinanced may be less likely to refinance again soon.

### Balance and loan size

Large-balance borrowers may respond differently to rate incentives because closing costs, tax treatment, and property value vary. Small loans can be less economical to refinance. The average balance is not enough; the distribution matters.

### Credit and equity

Borrowers with weaker credit may be unable to refinance even when market rates fall. Borrowers with negative or limited home equity may also be unable to obtain a new loan. The pool's weighted-average coupon can therefore overstate the refinance incentive available to the average borrower.

### Geography and turnover

Housing turnover differs by region, employment base, climate, household formation, and supply. A pool concentrated in a mobile labor market may prepay through property turnover even when refinance incentives are modest. A pool in a low-turnover region may extend longer.

#### Worked example: identical coupon, different prepayment

Suppose two \$100 million pools both have a 5.5% coupon and the market refinance rate falls to 4.5%.

1. Pool A contains newer loans with high borrower credit and high home equity.
2. Pool B contains older loans, weaker credit, and high closing-cost sensitivity.
3. Pool A may prepay faster because refinancing is feasible and valuable.
4. Pool B may prepay more slowly because borrowers cannot qualify or do not find the transaction worthwhile.

The rate incentive is the same at the market level, but borrower ability and willingness differ.

## 14. Prepayment models and scenario construction

An MBS model should not produce one prepayment number and stop. It should show how cash flows change over a range of rates and borrower responses.

### Base, fast, and slow scenarios

A base case uses an expected prepayment curve. A fast case assumes stronger refinance and turnover. A slow case assumes higher rates, lower mobility, tighter credit, or operational friction. The purpose is not to predict the exact future; it is to measure the range of principal timing.

### Rate path matters

The same final rate can produce different prepayments depending on the path. A gradual decline gives borrowers time to refinance. A brief rate decline may produce less activity. A sharp rise followed by a decline can create a cohort of borrowers whose decisions differ from a stable-rate history.

### Volatility matters

Mortgage options become more valuable when rate volatility rises. Borrower prepayment behavior becomes harder to model, and the value of the embedded option increases. A security's option-adjusted spread can therefore change even if the simple yield does not move much.

#### Worked example: path dependence

Suppose a pool begins with a 6% mortgage coupon.

1. In Scenario A, market rates fall to 5% for one month and return to 6%.
2. In Scenario B, market rates fall to 5% and remain there for twelve months.
3. Both scenarios have the same low rate, but Scenario B gives borrowers more time to refinance.
4. The pool can prepay materially faster in Scenario B, changing WAL and the value of IO and PO positions.

## 15. CMO tranche behavior under prepayment stress

CMO analysis requires looking at both the collateral and the tranche rules. A class can be protected against one form of variability and exposed to another.

### Sequential classes

The first class receives principal early. Its average life may be stable under a range of speeds, but it can contract sharply in a fast-prepayment scenario. Later classes receive principal after earlier classes are paid, so they may extend when prepayments slow.

### PAC bands

A PAC class is designed to receive principal within a band of prepayment speeds. The support class absorbs variability inside the band. Outside the band, the PAC may lose its schedule protection.

### Support class

The support class receives the principal variability that the PAC is designed to avoid. Its expected life can change significantly with prepayment speed. A support class may offer a high yield because it carries the hard part of the timing risk.

#### Worked example: PAC protection fails outside its band

Suppose a PAC class is designed for CPR speeds between 8% and 20%, supported by a companion tranche.

1. At 12% CPR, the PAC receives principal close to its planned schedule.
2. At 18% CPR, the support class absorbs more contraction variability.
3. At 30% CPR, the support may be exhausted or insufficient.
4. The PAC can then receive principal earlier or later than planned, depending on the structure.

The protection is conditional, not absolute.

## 16. IO, PO, and mortgage option exposure

Interest-only and principal-only securities make the prepayment trade-off visible.

An IO investor wants the interest stream to continue. Faster prepayment reduces the outstanding balance and therefore reduces future interest. An IO can lose value when rates fall and refinancing accelerates.

A PO investor wants principal. Faster prepayment can return principal sooner and may increase value if the PO was purchased at a discount. But the PO remains exposed to borrower credit, timing, and market liquidity.

### Option-adjusted thinking

The mortgage borrower owns an option to prepay. The investor is effectively short that option in a pass-through. A model that discounts fixed cash flows without modeling the option can overstate value and understate risk.

The option is not traded in a simple standardized form. It is inferred from borrower behavior, historical data, rate incentives, housing conditions, servicing, and model assumptions.

## 17. Credit risk in private-label RMBS

Agency MBS and private-label RMBS should not be analyzed with the same credit framework. A private-label deal may include senior, mezzanine, and residual tranches, representations and warranties, reserve accounts, excess spread, and mortgage insurance.

### Loss allocation

If a private-label pool has \$100 million of mortgages and \$10 million of junior protection, the senior class begins to suffer principal loss only after the junior protection and other available buffers are exhausted. But the exact threshold can move because fees, advances, modifications, and recoveries affect available cash.

### Housing correlation

Mortgage defaults can be correlated through house prices, unemployment, rates, underwriting, and regional concentration. A pool diversified by borrower may still be concentrated in the national housing factor.

#### Worked example: junior protection and recovery

Suppose a \$100 million private-label pool has \$8 million of equity and \$12 million of mezzanine protection. A stress causes \$15 million of net losses.

1. Equity absorbs \$8 million.
2. Mezzanine absorbs the next \$7 million.
3. Mezzanine retains \$5 million of principal before timing and other expenses.
4. Senior principal remains protected in this simplified loss-only calculation.

If recoveries arrive late, the cash-flow outcome can be worse than the loss allocation suggests.

## 18. Mortgage servicing rights and platform M&A

Mortgage servicing rights, or MSRs, are economic rights to receive servicing income and perform servicing obligations. Their value changes with mortgage rates and prepayment. When rates fall and prepayments accelerate, the servicing asset may lose value because loans leave the servicing book faster. When rates rise and prepayments slow, servicing income may last longer, but borrower delinquency and liquidity risks can change.

This makes mortgage servicing an important M&A asset. A buyer may want servicing scale, a borrower relationship, data, origination distribution, or a hedgeable stream of fees. But the buyer inherits operational, regulatory, liquidity, and prepayment exposure.

### Current platform examples

Fannie Mae reported \$409 billion of mortgage-market liquidity in 2025, including \$133 billion of single-family MBS issued through whole-loan conduit transactions and \$276 billion through lender swaps. The figures show the scale of a standardized agency platform, but they do not describe the economics of every private mortgage lender.

An acquirer of a mortgage platform should separate:

- servicing income;
- mortgage origination revenue;
- gain-on-sale revenue;
- hedge results;
- MSR valuation changes;
- repurchase exposure;
- warehouse financing;
- capital requirements.

#### Worked example: MSR sensitivity

Suppose a servicing portfolio has \$1 billion of mortgage balances and earns a 0.25% annual servicing fee before costs.

1. Gross annual servicing income is \$2.5 million before runoff.
2. If fast prepayment reduces the average balance by 20%, the fee base falls materially.
3. If slow prepayment keeps balances outstanding, income lasts longer but the portfolio may have greater duration and borrower-performance exposure.
4. The value of the servicing asset therefore depends on both fee rate and prepayment path.

## 19. Data, disclosure, and operational resilience

MBS investors depend on loan-level data, pool factors, remittance reports, prepayment histories, servicing disclosures, and agency publications. Data must be timely and comparable across issuers.

FHFA's National Mortgage Database provides aggregate statistics from a representative sample of residential mortgages, including outstanding mortgage and performance data through Q1 2026 in the latest release context. It is a public statistical resource, not a substitute for transaction-level diligence. [FHFA NMDB](https://www.fhfa.gov/data/nmdb)

Operational resilience matters because a security can remain legally outstanding while reporting or settlement systems are disrupted. The Common Securitization Platform and its successor technology infrastructure are part of the market's plumbing. Data, payment instructions, disclosure, and settlement are all essential to liquidity.

## 20. A complete MBS stress walkthrough

Consider a \$50 million pass-through pool with a 5.5% gross coupon. Assume servicing and guarantee fees total 0.5%, so the net coupon is 5.0% before other adjustments.

#### Worked example: contraction and extension

1. In a fast-prepayment scenario, 30% of principal returns over the modeled horizon.
2. The investor receives cash sooner but must reinvest at lower market yields.
3. In a slow-prepayment scenario, only 10% of principal returns over the same horizon.
4. The investor keeps receiving 5.0% net coupon on a larger balance, but the security's duration extends and its price can fall if rates remain high.
5. A hedge sized for the base case may be too large in the fast case and too small in the slow case.

The MBS investor is managing a changing cash-flow distribution, not a fixed maturity bond.

## 21. Mortgage economics and borrower choice

The borrower is not a passive source of cash. A household decides whether to refinance, move, make an extra payment, request a modification, or remain in the existing loan. Those decisions are affected by rates, income, house prices, taxes, closing costs, job mobility, family changes, and expectations about the future.

### Refinancing is an exercise of judgment

A borrower with a 6% mortgage and a 5% refinance offer may still refuse to refinance. The borrower may plan to move soon, have insufficient equity, lack documentation, face a high fee, or expect rates to fall further. Another borrower may refinance at a smaller rate difference because the new loan changes payment stress or removes a risky feature.

Models often summarize this decision with an incentive variable, but the variable is only a proxy. The model should be tested against actual borrower response by coupon, geography, credit, loan age, and channel.

### Turnover is not just a rate variable

Home sales can prepay a mortgage even when refinancing is unattractive. Employment relocation, divorce, household formation, death, and changes in local housing conditions can change turnover. A pool with low refinance incentive can still prepay through housing activity.

### Servicing and borrower contact

Servicer communications can influence whether a borrower completes a refinance, cures a delinquency, modifies a loan, or enters foreclosure. Digital servicing can speed decisions but also create fraud, verification, and operational risks. A mortgage pool is therefore partly a behavioral system and partly a servicing system.

## 22. Measuring mortgage bond risk correctly

Yield is a starting point, not a complete answer. Investors use several measures because each captures a different feature.

**Duration** approximates price sensitivity to a small parallel change in rates, holding cash-flow assumptions fixed.

**Convexity** captures curvature, but for an MBS the curvature is affected by changing prepayment.

**Weighted-average life** describes principal timing, not all rate sensitivity.

**Spread duration** measures sensitivity to changes in mortgage spreads.

**Option-adjusted spread**, or OAS, attempts to compare value after accounting for the embedded prepayment option and modeled cash flows.

**Dollar duration** converts sensitivity into a monetary amount for a position size.

#### Worked example: duration approximation

Suppose a \$10 million MBS portfolio has a duration of 5 years. A parallel rate increase of 0.10%, or 10 basis points, gives a first-order price impact of approximately:

1. Rate change: 0.10%.
2. Duration impact: 5 × 0.10% = 0.50%.
3. Approximate dollar price change: \$10 million × 0.50% = \$50,000 loss.

This is a first-order illustration. It ignores convexity, spread movement, prepayment changes, liquidity, and hedge performance. For an MBS, those omitted effects can be material.

### Scenario duration

Because duration changes with rates, investors often calculate effective duration under a rate shock rather than rely on one static number. A down-rate shock may shorten the MBS, while an up-rate shock may extend it. A portfolio can therefore have an asymmetric hedge requirement.

## 23. Agency MBS, government support, and investor interpretation

Agency terminology should be read precisely. A government agency may issue or guarantee a security. A government-sponsored enterprise may guarantee the timely payment of principal and interest under its program. The guarantee can protect investors from defined mortgage credit losses, but investors still depend on the legal terms, the guarantor, the servicing system, and the settlement infrastructure.

FHFA's oversight of Fannie Mae, Freddie Mac, and Common Securitization Solutions reflects the system-wide importance of standardized mortgage funding. FHFA's 2025 Annual Report to Congress describes enterprise portfolios, MBS issuance, and the Common Securitization Platform's role in processing and disclosure. [FHFA 2025 Annual Report](https://www.fhfa.gov/document/d/arc/fhfa-2025-annual-report-to-congress.pdf)

The investor's conclusion should be specific: the security may have strong defined credit protection, but it still has market risk, prepayment risk, basis risk, and operational dependency.

## 24. A practical MBS investor checklist

Before buying an MBS or mortgage-related tranche, ask:

### Collateral

- What is the coupon distribution?
- How seasoned are the mortgages?
- What are the loan balances, geographies, occupancy, and purposes?
- What proportion has refinance incentive?
- How concentrated is the pool in turnover-sensitive regions?

### Cash flow

- What is the base CPR and SMM?
- What are fast and slow prepayment cases?
- What is the expected WAL under each case?
- How do principal and interest pass through the servicer?
- Are advances, fees, and recoveries treated consistently?

### Structure

- Is it a pass-through, CMO, PAC, support, IO, PO, or private-label tranche?
- What happens after a delinquency, trigger, or counterparty failure?
- Which class receives principal first?
- Who absorbs credit and timing losses?

### Market

- What is the bid-ask spread?
- Can the security be financed?
- How does it trade relative to Treasuries and swaps?
- What hedge is required under contraction and extension?

### Governance and M&A

- Who owns the servicing platform?
- Is the servicer part of an acquired group?
- Are origination and servicing incentives aligned?
- Could a platform acquisition change data, systems, or asset allocation?

This checklist does not predict the next prepayment report. It makes the uncertainty visible enough to price and monitor.

## 25. Portfolio construction and hedge discipline

MBS are often held in portfolios rather than alone. A manager may combine coupons, specified pools, TBA positions, Treasury futures, interest-rate swaps, swaptions, and cash. The portfolio objective can be income, duration management, liquidity, or relative value.

The portfolio must be tested under both a rate shock and a prepayment shock. A hedge that works for a pass-through with a 5-year duration may fail when a rate decline shortens the mortgage. A position that appears diversified by issuer can remain concentrated in the same borrower option and the same rate factor.

### Specified pools

Specified pools are selected for characteristics such as low loan balance, geographic profile, investor property, or high-balance loans. They may trade at a pay-up because their prepayment behavior differs from generic TBA collateral. The pay-up is valuable only if the behavior persists and the cost is justified.

### Liquidity reserve

A manager should hold enough liquidity to meet margin, hedge, redemption, and operational needs. MBS can be liquid in normal markets and less liquid during a rate shock or a mortgage basis event. Liquidity planning is part of credit and market-risk management.

#### Worked example: hedge mismatch

Suppose a manager owns \$20 million of MBS with a base duration of 4 years and hedges \$80 million of Treasury-equivalent duration. Rates fall and the MBS duration shortens to 2 years.

1. Original MBS duration exposure: \$20 million × 4 = \$80 million-years.
2. New MBS duration exposure: \$20 million × 2 = \$40 million-years.
3. The unchanged hedge now offsets twice the intended exposure.
4. The portfolio can lose from the hedge even if the MBS price rises.

The example is stylized, but it captures why mortgage hedging must respond to changing cash flows.

The final discipline is to keep the borrower option visible. An MBS is a mortgage portfolio, a servicing system, a legal guarantee, a capital-markets instrument, and an interest-rate option at the same time. Any analysis that removes one of those layers is incomplete.

That is the core MBS lesson.

Cash-flow timing is risk.

Always.

For MBS.

In practice.

For investors.

Everywhere.

For a long time.

And carefully.

Together.

Precisely.

Always.

## Common misconceptions

### “Agency MBS have no risk”

Agency guarantees may reduce defined credit exposure, but investors still face prepayment, duration, convexity, liquidity, basis, operational, and reinvestment risks.

### “A 30-year MBS has a 30-year duration”

The legal maturity is not the expected principal timing. Prepayment can make the expected life much shorter or, when rates rise, longer.

### “Falling rates are always good for mortgage bonds”

Falling rates can accelerate prepayment and force investors to reinvest principal at lower yields.

### “CMOs remove prepayment risk”

CMOs redistribute timing risk across classes. A PAC may have a target window, while support and subordinate classes absorb more variability.

### “A prepayment model is a fact”

It is an assumption about behavior. The model must be stress-tested across rates, housing turnover, borrower incentives, and seasoning.

## How it shows up in real markets

Mortgage securitization supports housing finance by connecting mortgage originators with investors. The system depends on standardized data, predictable guarantees, servicing continuity, and a liquid secondary market.

FHFA's prepayment monitoring work shows why alignment across Fannie Mae and Freddie Mac cohorts matters for UMBS liquidity. A security can be legally standardized while its borrower cash flows still differ if prepayment behavior diverges.

The current market also demonstrates the interaction between public and private credit. The same mortgage ecosystem includes agency MBS, private-label RMBS, mortgage REITs, servicing-rights investors, insurers, banks, and specialty-finance platforms. M&A can change who owns the origination and servicing infrastructure without changing the underlying mortgage cash flows.

## When this matters to you

MBS affect mortgage rates because lenders price the funding and hedging cost of the mortgage assets they originate. Borrowers experience the system through mortgage availability, rate locks, refinancing economics, and servicing quality.

For an MBS investor, the practical checklist is:

1. Separate agency credit protection from market and prepayment risk.
2. Identify coupon, seasoning, geography, loan purpose, and borrower incentive.
3. Model CPR, SMM, default, recovery, and turnover assumptions.
4. Calculate expected WAL under contraction and extension scenarios.
5. Stress the hedge as well as the MBS.

## Sources & further reading

- [FHFA 2025 Scorecard](https://www.fhfa.gov/news/news-release/fhfa-releases-2025-scorecard-for-fannie-mae-freddie-mac-and-common-securitization-solutions), published 20 December 2024.
- [Fannie Mae 2025 Annual Housing Activities Report](https://www.fanniemae.com/media/56766/display), 2025 data.
- [Freddie Mac 2025 Annual Housing Activities Report](https://www.freddiemac.com/about/pdf/2025-annual-housing-activities-report.pdf), 2025 data.
- [FHFA Q2 2025 Prepayment Monitoring Report](https://www.fhfa.gov/reports/prepayment-monitoring-report/2025/Q2), published 26 September 2025.
- [FHFA National Mortgage Database](https://www.fhfa.gov/data/nmdb), data release 26 June 2026.
- [Structured Finance from First Principles](/blog/trading/finance/structured-finance-from-first-principles).
- [Asset-Backed Securities](/blog/trading/finance/asset-backed-securities-abs-explained).
