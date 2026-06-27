---
title: "Credit and Distressed Debt Valuation: Spreads, Recovery Rates, and Bankruptcy"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "How to value bonds, distressed debt, and companies in bankruptcy by understanding credit spreads, the Merton model, CDS pricing, recovery rates by seniority, and the fulcrum security."
tags: ["valuation", "credit", "distressed-debt", "bankruptcy", "merton-model", "credit-default-swap", "recovery-rates", "bonds", "fulcrum-security", "asset-valuation"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR** — When a company teeters on the edge of default, the entire valuation playbook flips: equity becomes a lottery ticket, debt becomes the new equity, and recovery rate analysis replaces earnings forecasting.
>
> - Credit spread = corporate yield minus the risk-free rate; it prices in the market's estimate of default probability times loss-given-default.
> - The Merton model shows that equity is literally a call option on firm assets, with the face value of debt as the strike — when assets fall below debt, equity goes to zero.
> - Distressed investors buy senior debt at 40-60 cents on the dollar and target 70-90 cents in recovery through restructuring, targeting 15-25% IRR.
> - Recovery rates depend critically on seniority: senior secured lenders recover ~70¢, senior unsecured ~38¢, subordinated ~18¢ (Moody's Ultimate Recovery Database, as of 2023).
> - The "fulcrum security" is the layer in the capital stack where enterprise value tips from full recovery to partial — whoever holds it controls the restructuring outcome.
> - A CDS (credit default swap) is priced as default probability times loss-given-default — the same math as the credit spread, but in swap form.

In the autumn of 2008, Lehman Brothers' bonds were trading at 12 cents on the dollar. Meanwhile, the equity had already gone to zero. Any investor still holding Lehman stock faced a near-total wipeout. But investors who had purchased Lehman's senior unsecured bonds at those distressed prices? They ultimately recovered roughly 21 cents on the dollar — not much, but 75% more than zero.

This story captures everything essential about credit and distressed debt valuation: the same company, the same moment in time, but radically different values depending on *where you sit in the capital structure*. The equity investor lost everything. The senior bondholder lost most of it. A sophisticated distressed investor who bought the bonds at 10 cents and recovered 21 cents earned a 110% return on that trade.

How do you analyze these situations? How do you price a bond for credit risk? How do you value a company that is actively going bankrupt? How do you decide which layer of debt is the most attractive bet? These are the questions this post answers — from first principles, with real numbers.

![Capital structure seniority stack showing recovery rates by layer](/imgs/blogs/credit-distressed-debt-valuation-recovery-rates-1.png)

## Foundations: Credit Risk and What Makes Debt Risky

Before we can value distressed debt, we need to understand what makes any debt risky in the first place.

### The two promises a bond makes

When a company borrows money by issuing a bond, it makes two promises to the lender:
1. **Pay interest** (the "coupon") at regular intervals — say, 6% per year on a \$1,000 bond = \$60/year.
2. **Repay the principal** (the face value = \$1,000) at maturity — say, in 10 years.

A US Treasury bond makes these same two promises, but backed by the full faith and credit of the United States government. The government can print money; it virtually never defaults. So the Treasury yield is treated as the **risk-free rate** — the baseline return you earn for locking up money for a given period, with essentially no default risk.

A corporate bond makes the same promises, but the company can go bankrupt and fail to pay. You are taking additional risk. You need to be compensated for that risk. The compensation comes in the form of a higher yield.

### What is a credit spread?

The **credit spread** is simply the extra yield you earn above the risk-free rate:

```
Credit Spread = Corporate Bond Yield − Risk-Free Rate (Treasury Yield)
```

If a 10-year corporate bond yields 6.5% and the 10-year US Treasury yields 4.5%, the credit spread is 200 basis points (bps). One basis point = 0.01%, so 200 bps = 2.0%.

That 200 bps is the market's price for the default risk of this specific company over 10 years. But what determines whether 200 bps is fair? We need to decompose what a credit spread actually compensates for.

### The credit spread decomposition

A credit spread compensates investors for three things:

**1. Expected loss from default:**
```
Expected Loss = Probability of Default (PD) × Loss Given Default (LGD)
```

If the market thinks there's a 3% chance this company defaults each year, and if it defaults you lose 60% of your money (LGD = 60%), then:
- Expected loss = 3% × 60% = 1.80% per year = 180 bps/year

**2. Unexpected loss (volatility risk premium):** Even if expected losses average 180 bps, the actual default either happens or it doesn't — you get 0% loss or 60% loss. That binary outcome adds risk beyond the expected value. Investors demand compensation for bearing this volatility.

**3. Liquidity risk premium:** Corporate bonds are harder to trade than Treasuries. In a crisis, you might not find a buyer at a fair price. Illiquidity deserves compensation.

So in practice:
```
Credit Spread ≈ (PD × LGD) + Volatility Risk Premium + Liquidity Premium
```

The more rigorous way to think about it: the credit spread is the price of **insurance against default**, packaged into the yield differential.

### Loss given default vs recovery rate

These two terms are mirror images:
- **Recovery Rate (RR):** What fraction of the face value you get back if default occurs. Example: 40% recovery rate = you recover 40 cents on the dollar.
- **Loss Given Default (LGD):** How much you lose. LGD = 1 − RR. A 40% recovery means LGD = 60%.

Recovery rate is *not* uniform — it depends critically on where in the capital structure your claim sits. We'll explore this in depth below.

### Investment grade vs high yield

The bond market divides issuers into two broad buckets based on credit quality:

- **Investment grade (IG):** Rated BBB-/Baa3 or above by S&P/Fitch/Moody's. These companies have relatively low default risk. IG spreads typically run 50–200 bps over Treasuries. The US IG corporate market is approximately \$10 trillion in size (as of 2024).

- **High yield (HY):** Rated BB+/Ba1 or below — also called "junk bonds." Higher default risk, spreads typically 300–700+ bps over Treasuries. The US HY market is approximately \$1.4 trillion (as of 2024).

When a company falls from IG to HY, it is called a "fallen angel." The downgrade often triggers forced selling from IG-only mandated funds, creating dislocations that distressed investors exploit.

---

## Credit Spread Mechanics: The Math Behind the Price

### The simple credit spread formula

Let's build up the intuition step by step with the risk-free rate data we have.

At end of 2024, the US 10-year Treasury (UST 10Y) yielded 4.57% (Federal Reserve H.15). This is our risk-free anchor.

A BBB-rated (lowest IG) corporate bond might yield 6.07% at the same maturity. The spread is:
```
6.07% − 4.57% = 1.50% = 150 bps
```

A BB-rated (highest HY) corporate bond might yield 7.57%:
```
7.57% − 4.57% = 3.00% = 300 bps
```

A CCC-rated (near-distress) bond might yield 12.57%:
```
12.57% − 4.57% = 8.00% = 800 bps
```

What justifies these differences? The market's implied default probability and recovery rate.

#### Worked example: Extracting implied default probability from a credit spread

You observe a 5-year corporate bond with a credit spread of 400 bps. Industry data suggests the recovery rate for this type of debt is about 40% (so LGD = 60%). What annual default probability is implied?

**Step 1: Approximate the relationship.**

For an approximate calculation, assuming defaults are small and independent each year:
```
Credit Spread ≈ PD × LGD (the "hazard rate" approximation)
```
```
400 bps = PD × 60%
PD = 400 bps / 60% = 667 bps = 6.67% per year
```

**Step 2: Interpret.** The market is implying roughly a 6.67% annual default probability. Over 5 years, the cumulative survival probability is approximately:
```
Survival = (1 − 0.0667)^5 = (0.9333)^5 ≈ 0.699 = 69.9%
Cumulative PD over 5 years ≈ 30.1%
```

**Step 3: Check against reality.** A 30% cumulative 5-year default rate is consistent with CCC-rated bonds historically, but high for a BB-rated issuer (historical 5-year cumulative default rate for BB is roughly 8–12%). If this bond is rated BB, the spread may be pricing in extra distress risk, liquidity, or market-wide risk aversion.

**Key intuition:** Credit spread = the cost of default insurance, per year. A 400 bps spread means you're paying 4% annually above the risk-free rate to compensate for the risk of losing 60% of your money with ~6.7% annual probability.

![Corporate credit spreads vs US Treasury yields 2010-2024](/imgs/blogs/credit-distressed-debt-valuation-recovery-rates-2.png)

### Spread widening and tightening

Credit spreads are not static — they move with the economic cycle and market sentiment:

- **Spread tightening** (spreads narrow): Investors are optimistic about default risk. This usually happens in boom periods when companies generate strong free cash flow and can easily service debt. HY spreads hit as low as ~275 bps in 2021 (FRED data).

- **Spread widening** (spreads widen): Investors fear default. In recessions or financial crises, spreads blow out. HY spreads hit ~750 bps in early 2020 (COVID shock) and peaked around 2,000 bps in December 2008 (financial crisis).

The chart above shows how dramatically spreads moved from 2010 to 2024. Notice how IG spreads (blue) are much tighter and less volatile than HY spreads (red) — reflecting the dramatically different default risk profiles.

---

## The Merton Model: Equity as a Call Option

One of the most profound insights in finance came from Robert Merton in 1974. He showed that if you have a firm with assets financed by debt and equity, you can use option pricing theory to value both. The insight is elegant: **equity is a call option on the firm's assets.**

### The setup

Imagine a simple company:
- **Firm assets:** Value \$V\$ today, evolving stochastically over time.
- **Debt:** One class of zero-coupon debt with face value \$D\$ maturing in \$T\$ years.
- **Equity:** Residual claim on assets after debt is repaid.

What happens at maturity?

**If \$V_T > D\$ (firm value exceeds debt):**
- Equity holders pay off the debt (\$D\$) and keep the rest: **Equity = \$V_T − D\$**
- Debt holders get their full \$D\$ back.

**If \$V_T < D\$ (firm value less than debt — default):**
- Equity holders walk away (limited liability). **Equity = \$0\$**
- Debt holders get whatever the firm is worth: **Debt = \$V_T\$** (less than face value)

This is *exactly* the payoff of a call option! The equity payoff at maturity is:
```
Equity = max(V_T − D, 0)
```

This is a call option on firm assets (\$V\$), struck at the face value of debt (\$D\$), maturing when the debt matures (\$T\$).

### And debt is a short put

If equity is a long call on firm assets, what is debt?

The firm's total assets must equal debt plus equity:
```
V = Equity + Debt
Debt = V − Equity = V − max(V − D, 0) = min(V, D)
```

This is equivalent to:
```
Debt = D × e^(-rT) − max(D − V, 0)
```

Which is: risk-free bond value minus a put option on firm assets. The debt holder has lent money (the risk-free bond component) but written a put option — they bear the downside if firm value falls below the face value of debt.

![Merton model pipeline showing equity as call option on firm assets](/imgs/blogs/credit-distressed-debt-valuation-recovery-rates-3.png)

### Applying Black-Scholes to the Merton model

Since equity is a call option, we can price it using the Black-Scholes formula:

```
Equity = V × N(d1) − D × e^(-rT) × N(d2)

d1 = [ln(V/D) + (r + σ²/2) × T] / (σ × √T)
d2 = d1 − σ × √T
```

Where:
- \$V\$ = current firm asset value
- \$D\$ = debt face value (the "strike")
- \$r\$ = risk-free rate
- \$σ\$ = volatility of firm assets
- \$T\$ = time to maturity
- \$N(\cdot)\$ = cumulative normal distribution

The credit spread implied by the Merton model is:
```
Credit Spread = −(1/T) × ln[N(d2) + (V/D × e^(rT)) × N(−d2)] − r
```

#### Worked example: Merton model credit spread calculation

Consider a company with the following parameters:
- Firm asset value \$V = \$1,000\$
- Debt face value \$D = \$500\$ (leverage ratio = 50%)
- Asset volatility \$σ = 30%\$
- Time to debt maturity \$T = 1\$ year
- Risk-free rate \$r = 4.57%\$ (UST 10Y end-2024)

**Step 1: Compute d1 and d2**
```
d1 = [ln(1000/500) + (0.0457 + 0.5 × 0.09) × 1] / (0.30 × 1)
   = [0.6931 + 0.0907] / 0.30
   = 0.7838 / 0.30
   = 2.613
```
```
d2 = 2.613 − 0.30 = 2.313
```

**Step 2: Compute equity value**
```
N(d1) = N(2.613) ≈ 0.9955
N(d2) = N(2.313) ≈ 0.9896
```
```
Equity = 1000 × 0.9955 − 500 × e^(-0.0457) × 0.9896
       = 995.5 − 500 × 0.9553 × 0.9896
       = 995.5 − 472.0
       = 523.5
```

**Step 3: Debt value**
```
Debt = 1000 − 523.5 = 476.5
```

**Step 4: Implied credit spread**

The fair value of the debt is \$476.5\$ on a face of \$500\$. If risk-free debt would be:
```
Risk-free PV of D = 500 × e^(-0.0457) = 500 × 0.9553 = 477.7
```
The credit spread adjusts the yield upward from the risk-free rate. At \$476.5\$:
```
Bond YTM = -ln(476.5/500) / 1 = 4.84%
Credit Spread = 4.84% − 4.57% = 0.27% = 27 bps
```

**Intuition:** With leverage only 50% and asset volatility 30%, the default probability is low (the firm would need to lose more than half its value), so the spread is modest (27 bps). This company is IG-quality.

#### Worked example: Merton model — leverage sensitivity table

Now what happens when leverage rises to 80% (debt = \$800\$)?

```
d1 = [ln(1000/800) + 0.0907] / 0.30 = [0.2231 + 0.0907] / 0.30 = 1.046
d2 = 1.046 − 0.30 = 0.746
```
```
N(d2) = N(0.746) ≈ 0.772
```

At 80% leverage the implied spread jumps dramatically — now in HY territory, often 300–400 bps depending on volatility assumptions.

To make this sensitivity concrete, here is how the implied credit spread changes across leverage ratios (D/V) and asset volatility levels, keeping \$V = \$1,000\$ and \$r = 4.57%\$, \$T = 1\$ year:

| Leverage (D/V) | Asset Volatility (σ) | d2 | N(d2) | Implied Spread (bps) | Rating Equivalent |
|---|---|---|---|---|---|
| 30% (\$D=\$300\$) | 20% | 4.11 | 0.9999 | ~2 bps | AAA |
| 50% (\$D=\$500\$) | 20% | 2.95 | 0.9984 | ~6 bps | AA |
| 50% (\$D=\$500\$) | 30% | 2.31 | 0.9896 | ~27 bps | A/BBB |
| 65% (\$D=\$650\$) | 30% | 1.38 | 0.916 | ~130 bps | BB |
| 75% (\$D=\$750\$) | 35% | 0.79 | 0.785 | ~325 bps | B |
| 85% (\$D=\$850\$) | 40% | 0.21 | 0.583 | ~680 bps | CCC |

**Key insight:** The Merton model quantifies what practitioners know intuitively — higher leverage + higher asset volatility = wider credit spread = more expensive borrowing. A seemingly modest increase in leverage from 50% to 85% combined with higher asset volatility takes the spread from 27 bps (IG comfort zone) to 680 bps (CCC distress territory) — a non-linear explosion driven by the optionality embedded in equity.

![Merton model: equity and debt value as a function of firm asset value](/imgs/blogs/credit-distressed-debt-valuation-recovery-rates-6.png)

The chart above makes this concrete. When firm asset value \$V\$ far exceeds the debt face value \$D = \$500\$, equity behaves like the full residual claim (the green line tracks \$V − 500\$). But as \$V\$ approaches \$D\$ from above, equity's value is *higher* than a naive linear calculation (green vs dashed red) — because of the optionality. Equity holders can't lose more than zero, but they keep all the upside above \$D\$.

---

## Distressed Debt Investing: Buying at 40 Cents

When a company gets into serious financial trouble, its bonds trade at a steep discount to face value. A \$1,000 face-value bond might trade at \$400 — "40 cents on the dollar." This is the domain of distressed debt investors: specialists who buy claims in companies in or near bankruptcy and profit from the restructuring process.

### Why bonds trade at distressed prices

A bond trades at a steep discount when:
1. The market believes a default is likely.
2. Recovery in default is uncertain.
3. Other holders are forced sellers (funds with IG-only mandates, funds facing redemptions).

The *opportunity* for distressed investors is that forced selling and uncertainty often drive prices below the rational value of the recovery.

### The distressed investing math

A distressed investor's return has two sources:
1. **The cash recovery** in the restructuring (cents on the dollar actually received).
2. **The exit price** if they sell before resolution.

The fundamental bet is: "I can buy this claim at \$0.40 on the dollar and recover \$0.65, earning a 62.5% return in 18 months." The annualized IRR depends on how fast the restructuring resolves.

#### Worked example: Distressed debt trade anatomy

An energy company files for Chapter 11 in January 2024. It has:
- Senior secured bonds (first lien): \$500M face value, trading at **72 cents** = \$360M market value
- Senior unsecured bonds: \$400M face value, trading at **38 cents** = \$152M market value
- Equity: Worthless, suspended from trading

A distressed hedge fund analyzes the company:

**Step 1: Enterprise value analysis.** The company has an EBITDA of \$80M. Comparable energy companies trade at 5–7× EBITDA. EV estimate: 6× \$80M = \$480M.

**Step 2: Waterfall analysis (who gets paid in order):**
- Administrative claims (lawyers, advisors): ~\$30M
- Senior secured (first lien): \$500M face → but enterprise value is only \$480M − \$30M = \$450M
- Available to senior secured: \$450M / \$500M = **90 cents on the dollar**
- Senior unsecured: zero (the enterprise value is exhausted by senior secured)

**Step 3: The fund's entry and exit:**

If the fund buys \$100M face of senior secured bonds at 72 cents = \$72M investment:
- Expected recovery: 90 cents = \$90M
- **Profit: \$18M on \$72M invested = 25% return**
- Time horizon: 18 months
- **Annualized IRR: ~16%**

If the fund buys the senior unsecured at 38 cents thinking they can argue for a higher EV:
- The waterfall shows zero recovery for unsecured
- Unless the fund can negotiate a higher EV or provide new money to get paid
- The unsecured position is **out of the money** in this EV scenario

**Key insight:** Distressed investing is all about the waterfall — the sequential order in which creditors get paid. The return depends entirely on where enterprise value intersects the capital structure.

---

## Credit Default Swaps: Insurance on Default

A **credit default swap (CDS)** is a derivative that transfers credit risk from one party to another. It is, in essence, insurance against default.

### How a CDS works

**Protection buyer:** Pays a fixed annual premium (the "CDS spread") on the notional amount. Gets paid if there is a "credit event" (default, restructuring, failure to pay). Think of this as the insurance premium.

**Protection seller:** Collects the premium stream. Must pay out if there is a credit event. The payout is: Notional × (1 − Recovery Rate).

![CDS mechanics pipeline](/imgs/blogs/credit-distressed-debt-valuation-recovery-rates-5.png)

### CDS pricing

The fair CDS spread is determined by the same logic as the credit spread:
```
CDS Spread ≈ PD × LGD (per year, for small PDs)
```

More precisely, the present value of premium payments equals the present value of expected protection payments:
```
CDS Spread × PV(premium leg) = PD × LGD × PV(protection leg)
```

#### Worked example: CDS pricing from first principles

Assume a 5-year CDS on a corporate issuer where:
- Annual default probability: 4% (derived from credit spread analysis)
- Recovery rate: 40% (LGD = 60%)
- Risk-free rate: 4.57% (UST 10Y 2024)
- Notional: \$10 million

**Step 1: Expected annual payout (protection leg)**
```
Annual expected payout = 4% × 60% × $10M = $240,000/year
```

**Step 2: PV of protection leg** (probability-weighted payout each year, discounted)

For simplicity, using a flat approximation:
```
PV protection ≈ $240,000 × (annuity factor at 4.57%, 5yr)
Annuity factor = (1 − (1.0457)^(-5)) / 0.0457 = 4.394
PV protection ≈ $240,000 × 4.394 = $1,054,560
```

**Step 3: Set equal to PV of premium leg**
```
CDS Spread × $10M × 4.394 = $1,054,560
CDS Spread × $43,940,000 = $1,054,560
CDS Spread = $1,054,560 / $43,940,000 = 2.40% = 240 bps/year
```

**Step 4: Annual premium payments**
```
Annual premium = 240 bps × $10M = $240,000/year
```

**If default occurs in year 3:**
- Premium payments stop
- Seller pays: \$10M × (1 − 40%) = **\$6,000,000** to the protection buyer

**Key insight:** The CDS spread is essentially the same math as the credit spread — it prices default probability times loss severity. The difference is that a CDS is a pure credit derivative (you don't need to own the bond to buy protection), making it more liquid and flexible.

### CDS spread vs bond credit spread: the "basis"

In theory, the CDS spread and the bond credit spread should be identical — both price the same default risk on the same reference entity. In practice, they often differ. The difference is called the **CDS-bond basis**:

```
Basis = CDS Spread − Bond Credit Spread
```

**Positive basis** (CDS spread > bond spread): CDS protection is expensive relative to the bond. This can happen when:
- Demand for protection outstrips supply (everyone wants insurance at once — as happened in 2008)
- There is a "cheapest-to-deliver" option in physical settlement that makes CDS more valuable than the bond

**Negative basis** (CDS spread < bond spread): The bond is cheap relative to CDS. This can indicate:
- Forced selling in the bond market (IG mandate redemptions, margin calls) that doesn't affect CDS
- Liquidity premium in the bond market not present in CDS

#### Worked example: Exploiting the negative basis trade

A telecom company has:
- 5-year bond credit spread: 350 bps (bond yield = UST + 350 bps = 4.57% + 3.50% = 8.07%)
- 5-year CDS spread: 280 bps

**Basis = 280 − 350 = −70 bps (negative basis)**

The bond yields 70 bps *more* than it should relative to CDS — it's cheaper in the cash market than the derivatives market implies.

**The negative basis trade:**
1. Buy the bond (earn 350 bps over Treasuries)
2. Buy CDS protection on the same issuer (pay 280 bps)
3. **Net carry = 350 − 280 = 70 bps per year**

If no credit event occurs, you earn 70 bps annually risk-free (ignoring funding costs and counterparty risk). If a credit event occurs, the bond loss is covered by the CDS payout. The trade is theoretically riskless — a form of capital structure arbitrage.

Basis trades require careful execution: funding the bond purchase (repo rate, haircuts) and collateral for the CDS must be accounted for. After 2008, many basis traders who had built large negative basis books suffered severe losses when funding dried up and repo haircuts spiked — even though the eventual credit outcome was fine. The lesson: **carry trades funded with leverage can be destroyed by funding liquidity crises even when the credit thesis is right.**

### Why CDS markets matter for valuation

CDS markets are often *more liquid* than cash bond markets. Investors watch CDS spreads to:
1. Get real-time market pricing of default probability.
2. Hedge bond portfolios against credit risk.
3. Speculate on credit deterioration or improvement.

When a company's CDS spreads surge (e.g., from 100 bps to 800 bps), it signals the market sees sharply higher default risk — often a leading indicator before rating agencies downgrade.

---

## Bankruptcy Valuation: What Is a Distressed Company Worth?

When a company files for bankruptcy, the central valuation question becomes: **what is this company worth, and who gets what?**

There are two competing answers: **liquidation value** and **going-concern value**.

### Liquidation value

If the company shuts down and sells everything immediately, what do the assets fetch?

Liquidation values are *discounted from book values* because:
- Forced sales happen at below-market prices.
- Specialized assets (custom machinery, intellectual property) may have limited buyers.
- The "going-concern premium" (value of assembled workforce, customer relationships, brand) is destroyed.

Typical liquidation haircuts by asset type:
- **Cash and equivalents:** 100 cents on the dollar (no haircut)
- **Accounts receivable:** 75–85 cents
- **Inventory:** 50–75 cents (raw materials higher, finished goods lower if sector-specific)
- **Real estate:** 70–85 cents (usually can be sold at close to market value over time)
- **Equipment and machinery:** 30–60 cents (highly variable; specialized equipment = low recovery)
- **Goodwill and intangibles:** Often 0–20 cents (most intangibles are worthless without the ongoing business)

#### Worked example: Liquidation waterfall

A retail company files Chapter 7 (liquidation). Balance sheet:

| Asset | Book Value | Recovery % | Liquidation Value |
|---|---|---|---|
| Cash | \$50M | 100% | \$50M |
| Accounts receivable | \$80M | 80% | \$64M |
| Inventory | \$120M | 55% | \$66M |
| Real estate | \$100M | 80% | \$80M |
| Equipment | \$60M | 40% | \$24M |
| Goodwill | \$90M | 0% | \$0M |
| **Total** | **\$500M** | | **\$284M** |

Claims (in order of seniority):
1. Administrative costs: \$15M
2. Senior secured (first lien): \$200M face
3. Senior unsecured bonds: \$150M face
4. Preferred equity: \$50M
5. Common equity: \$200M book

**Waterfall:**
- Start: \$284M
- After admin: \$284M − \$15M = \$269M
- Pay senior secured: \$269M available vs \$200M claim → **senior secured gets paid in full: 100¢, \$200M out, \$69M remaining**
- Pay senior unsecured: \$69M available vs \$150M claim → **senior unsecured gets \$69M / \$150M = 46¢ on dollar**
- Preferred equity: \$0M (nothing left)
- Common equity: \$0M

**Key insight:** The liquidation waterfall is brutally sequential. Once a layer is exhausted, all lower layers get nothing. The senior secured "fulcrum" here was the unsecured layer — they didn't get wiped out but didn't recover in full.

### Going-concern value

If instead the company can restructure its balance sheet (shed some debt, renegotiate leases, emerge from bankruptcy as a leaner entity), the business may be worth more as a going concern than in liquidation.

Going-concern value is estimated using the same methods as any operating company valuation:
1. **EV / EBITDA multiples:** What do comparable companies trade at? A distressed retailer might be valued at 5× EBITDA vs 8× for a healthy one (stressed multiple).
2. **DCF:** Project free cash flows under the restructuring business plan, discount at an appropriate (high) WACC reflecting the elevated risk.

The key question in bankruptcy: is going-concern value higher or lower than liquidation value? If higher, a restructuring (Chapter 11 reorganization) creates more value for all creditors. If lower (the business is fundamentally broken), liquidation is the better path.

![Going-concern vs liquidation valuation before-after diagram](/imgs/blogs/credit-distressed-debt-valuation-recovery-rates-7.png)

---

## Recovery Rates by Seniority: The Hierarchy of Claims

Recovery rates in bankruptcy are not uniform. They depend almost entirely on where you sit in the **capital structure hierarchy** — the legal pecking order for who gets paid first.

### The seniority ladder

The capital structure typically has five or more layers:

**1. Senior Secured Debt (First Lien)**
- Backed by specific collateral (real estate, equipment, accounts receivable).
- First in line for proceeds from collateral liquidation.
- Moody's Ultimate Recovery Database (as of 2023): **average recovery ~70¢; range 55–90¢** depending on collateral quality.

**2. Senior Secured Debt (Second Lien)**
- Also collateral-backed, but second in line on that collateral (first lien gets satisfied first).
- Moody's mean recovery: **~43¢**; range 25–65¢.

**3. Senior Unsecured Debt**
- No specific collateral backing. General creditor claim on the estate.
- Moody's mean recovery: **~38¢**; range 20–60¢. This is where most IG corporate bonds sit.

**4. Senior Subordinated / Mezzanine Debt**
- Contractually subordinated to senior debt — only paid after senior claims satisfied.
- Moody's mean recovery: **~25¢**; range 10–45¢.

**5. Subordinated Debt / Junior Notes**
- Deep subordination. Recovery is often near zero in severe distress.
- Moody's mean recovery: **~18¢**; range 5–35¢.

**6. Equity (Preferred and Common)**
- The residual — gets paid only after *all* debt is satisfied.
- In most corporate bankruptcies, equity recovers zero. Preferred: 0–15¢; Common: essentially zero.

![Recovery rates by debt seniority: Moody's data](/imgs/blogs/credit-distressed-debt-valuation-recovery-rates-4.png)

The chart makes this visual: there is a dramatic recovery rate gradient from top to bottom of the capital structure. This is why distressed investors focus intensely on seniority — it's the primary determinant of how much you get back.

### What drives recovery rate variation within a seniority class?

Even within senior secured, recovery rates vary from 55¢ to 90¢ depending on:

1. **Collateral quality:** Hard assets (real estate, equipment) at market values → higher recovery. Soft assets (customer lists, software IP) → lower.
2. **Industry:** Capital-intensive industries (oil & gas, manufacturing, real estate) tend to have higher recoveries because assets have liquid secondary markets. Service businesses or tech companies (where value is in the people and brand) often have lower recoveries.
3. **Debt quantum relative to assets:** A company with \$100M of senior debt against \$200M of hard assets will recover much better than one with \$100M of debt against \$80M of assets.
4. **Speed of bankruptcy:** Prepackaged bankruptcies (where debt holders agree on terms before filing) resolve faster, preserving more going-concern value. Contentious multi-year restructurings destroy value.

---

## The Fulcrum Security: Where Value Tips from Full to Zero

Perhaps the most important concept in distressed investing is the **fulcrum security**: the layer in the capital structure where enterprise value is exactly used up.

### The definition

The fulcrum security is the tranche of debt (or equity) where:
- Creditors *above* the fulcrum receive **full recovery** (their claims are fully covered by enterprise value).
- The fulcrum tranche itself receives **partial recovery** (enterprise value runs out partway through this layer).
- Creditors *below* the fulcrum receive **zero** (no value left for them).

The fulcrum security effectively *becomes the new equity* in the restructured company. If a reorganization plan converts debt to equity (standard in Chapter 11), the fulcrum holders are the ones who receive the new common shares — they become the new owners.

### Why the fulcrum matters

The fulcrum holders have the most negotiating leverage in a bankruptcy. They can:
- Block any restructuring plan that doesn't adequately compensate them.
- Push for a higher enterprise valuation (which enriches their recovery).
- Negotiate to receive a larger equity stake in the reorganized company.

Distressed investors often specifically target the fulcrum security because controlling it means controlling the restructuring outcome.

#### Worked example: Finding the fulcrum security

A manufacturing company files Chapter 11. Capital structure and enterprise value:

**Capital Structure:**
| Tranche | Face Value |
|---|---|
| Senior secured revolving credit | \$100M |
| Senior secured term loan (1L) | \$300M |
| Senior unsecured notes (8.5% due 2028) | \$400M |
| Subordinated notes | \$200M |
| Preferred equity | \$50M |
| Common equity | (Book: \$150M) |

**Total claims:** \$1,050M (excluding equity)

**Restructuring advisors estimate going-concern enterprise value:** \$750M

**Waterfall:**
- Revolver (\$100M): Fully paid → \$650M remaining
- Term loan 1L (\$300M): Fully paid → \$350M remaining
- Senior unsecured (\$400M claim): **Only \$350M available → recovery = 87.5¢**
- Subordinated notes: **Zero** (\$0 remaining)
- Preferred equity: **Zero**
- Common equity: **Zero** (wiped out)

**Fulcrum security: The senior unsecured notes.** They are the layer where enterprise value runs out. They receive 87.5 cents on the dollar — not full recovery, but not zero. In a debt-to-equity exchange, these noteholders would receive *new common equity* in the reorganized company.

If enterprise value turns out to be \$800M instead (which distressed investors will argue for):
- Revolver: Paid (\$700M remaining)
- Term loan: Paid (\$400M remaining)
- Senior unsecured: **Fully paid — \$400M of \$400M = 100¢**
- Subordinated notes: **\$0M → partial recovery (0¢)** — the fulcrum shifts down to subordinated!

**Key insight:** The fulcrum security is *highly sensitive* to small changes in enterprise value estimates. A \$50M change in EV can shift the fulcrum from one tranche to another, completely changing who controls the restructuring and who recovers anything.

#### Worked example: Quantifying the fulcrum security sensitivity to EV assumptions

Continuing the manufacturing company example above, a distressed fund is deciding whether to buy senior unsecured notes at **60 cents on the dollar**. The central question: what EV range makes this position worthwhile?

**Entry price:** 60¢ on \$400M face = \$240M investment

**Scenario analysis:**

| EV Assumption | Revolver (paid) | 1L TL (paid) | Unsecured recovery | Exit value per \$100 face | Return on 60¢ entry |
|---|---|---|---|---|---|
| \$600M | \$100M out → \$500M | \$300M out → \$200M | \$200M / \$400M = **50¢** | 50¢ | **−17%** |
| \$750M (base) | paid → \$650M | paid → \$350M | \$350M / \$400M = **87.5¢** | 87.5¢ | **+46%** |
| \$850M (bull) | paid → \$750M | paid → \$450M | **100¢** (fully paid) | 100¢ | **+67%** |
| \$1,000M (optimistic) | paid | paid | **100¢** + sub notes get 25¢ | 100¢ | **+67%** |

**What this tells the investor:**
- At EV ≥ \$750M, the 60-cent entry earns 46–67% — compelling for a 12–18 month restructuring (25–40% annualized IRR).
- At EV = \$600M, the position loses money (recovered 50¢ < 60¢ paid).
- The **break-even EV** is \$500M (revolver) + \$300M (TL) + 60% × \$400M (unsecured face at 60 cents) = \$500M + \$300M + \$240M = **\$1,040M total claims before unsecured** — no, the actual break-even is simpler: the unsecured needs to recover at least 60¢ × \$400M = \$240M, which requires EV above \$100M (revolver) + \$300M (TL) + \$240M = **\$640M.**

The fund's due diligence centers on whether \$640M is a reasonable floor for the going-concern EV. If they model \$700–800M as the likely range with bear case at \$600M, the position has positive expected value at 60¢ entry — but carries meaningful downside risk in the bear scenario.

---

## Common Misconceptions

### Misconception 1: "Senior secured debt is always safe — it always recovers close to par"

Reality: senior secured debt *does* recover better than junior claims, but "safe" is too strong a word. Moody's data shows first-lien secured average recovery of ~70¢, but the *range* is 55¢ to 90¢ — meaning a significant fraction of first-lien holders lose 30–45% of their investment. The wide range reflects collateral quality, industry, and total leverage.

The clearest counter-example is a **collateral collapse**. If a company borrows \$500M backed by a retail real-estate portfolio worth \$600M, the first-lien lenders feel safe. But if the retail collapse drives that real estate down to \$350M, the first-lien holders recover only \$350M / \$500M = **70 cents** — not par. In energy company bankruptcies during 2015–2016 (low oil prices), first-lien lenders on exploration assets routinely recovered 60–75¢ because the oil in the ground lost half its value.

**The lesson:** Senior secured is relatively safer, not absolutely safe. When analyzing distressed credit, always estimate the *range* of collateral values, not just the point estimate.

### Misconception 2: "CDS is just insurance — identical to buying a put on a bond"

A CDS is often called "credit insurance," but the analogy breaks down in important ways:

| Feature | CDS | Traditional Insurance |
|---|---|---|
| Insurable interest | **Not required** — you can buy protection on a bond you don't own | Required — you must own the insured asset |
| Counterparty risk | Yes — protection seller can default | Insurance company regulated solvency |
| Settlement | Cash or physical delivery of bonds | Cash payment only |
| Standardization | ISDA-standardized, liquid | Bespoke terms |
| Deliverable bonds | Can deliver *any* bond in the cheapest-to-deliver basket | N/A |

The most important difference: **you don't need to own the bond to buy CDS protection.** This means CDS can be used to speculate on credit deterioration. In 2007–2008, hedge funds bought billions of notional CDS protection on mortgage-backed securities they did *not* own — the "The Big Short" trade. The notional CDS outstanding on some reference entities far exceeded the actual bonds outstanding, which created significant concerns about systemic concentration of credit risk.

The "cheapest-to-deliver" option in physical settlement also matters: when a credit event occurs and settlement is physical, the protection buyer delivers the cheapest available bond in the reference entity's debt — often a deeply discounted, illiquid bond that is worth less than the average recovery. This option has value for the protection buyer and must be incorporated into proper CDS pricing.

### Misconception 3: "High-yield bonds are not real investments — just speculative gambling"

This conflates credit quality with investment merit. High-yield bonds are a legitimate and large (\$1.4 trillion) asset class with a long history of positive risk-adjusted returns. The ICE BofA US High Yield Index returned approximately 5–7% annually over the 2010–2024 period (factoring in defaults), with Sharpe ratios comparable to equities. Professional investors explicitly target HY credit as a core allocation. "Junk" is a colloquial term from the 1970s; "high yield" is now the standard market terminology because it better describes the return profile.

This conflates credit quality with investment merit. High-yield bonds are a legitimate and large (\$1.4 trillion) asset class with a long history of positive risk-adjusted returns. The ICE BofA US High Yield Index returned approximately 5–7% annually over the 2010–2024 period (factoring in defaults), with Sharpe ratios comparable to equities. Professional investors explicitly target HY credit as a core allocation. "Junk" is a colloquial term from the 1970s; "high yield" is now the standard market terminology because it better describes the return profile.

### Misconception 4: "In bankruptcy, equity always goes to zero"

Usually yes, but not always. In cases where enterprise value significantly exceeds total liabilities, equity can recover value — this happens in "asset-rich, cash-poor" bankruptcies where the company simply needed liquidity relief, not debt reduction. Hertz (2020) is a famous example: it emerged from bankruptcy in 2021, and the original shareholders — initially thought to be wiped out — received meaningful value because the used-car market surge dramatically increased the enterprise value of Hertz's fleet. Existing shareholders received warrants and some cash in the plan, worth hundreds of millions.

### Misconception 5: "The credit spread directly tells you the default probability"

The spread overstates default probability because it also includes the risk premium (compensation for volatility of default outcomes) and a liquidity premium. Historically, investment-grade bonds have significantly lower realized default rates than the credit spread alone would imply. Academics and practitioners call this the "credit spread puzzle" — spreads are systematically wider than pure expected-loss models predict. The excess is compensation for systematic risk (credit defaults cluster in recessions) and illiquidity.

### Misconception 6: "Recovery rate is fixed at 40% — I've heard that everywhere"

The "40% recovery" is a commonly cited average, but it masks enormous variation. Moody's data (as of 2023) shows that recovery rates range from ~10¢ for junior subordinated debt to ~90¢ for first-lien secured with quality collateral. The 40% figure is most representative for senior unsecured bonds, which happen to be the most common type analyzed. For specific distressed situations, actual recovery depends on: (a) seniority, (b) collateral, (c) industry, and (d) enterprise value at emergence — not a fixed assumption.

### Misconception 7: "Buying bonds at 40 cents guarantees profit"

The distressed investor isn't guaranteed anything. The risks are real:
- Enterprise value may be lower than estimated (EV could be \$300M when you needed \$400M for recovery).
- The restructuring may take 3–4 years, reducing IRR.
- New money investors (DIP lenders) may prime your position.
- A liquidation scenario may yield less than a going-concern analysis assumed.

Distressed investing requires deep credit analysis, restructuring expertise, and the capacity to be wrong on timing and value. Mediocre distressed investors can earn zero or negative returns. The best firms (Oaktree, Apollo, KKR Credit) generate 15–20% IRRs because they do this analysis better than the market.

---

## How It Shows Up in Real Markets

### Case study 1: General Motors (2009) — When \$27 billion of unsecured bonds recovered 10 cents

GM's June 2009 Chapter 11 bankruptcy was the fourth-largest in US history and provides one of the cleanest real-world illustrations of how the waterfall works under political pressure.

**The capital structure at filing:**
- US government and Canadian government: DIP (debtor-in-possession) financing of ~\$30 billion (priming lien — paid first)
- Secured bank debt: ~\$6.9 billion
- Unsecured bonds held by public investors: ~\$27 billion face value, trading at **15–20 cents on the dollar** before filing
- UAW retiree medical trust (VEBA): ~\$20.4 billion claim
- Equity: Worthless

**The contested enterprise value:**

The US government (as the DIP lender and de facto senior creditor) pushed for a "363 sale" — selling GM's best assets to a new "New GM" entity within 40 days, leaving the old liabilities behind. This is enormously faster than a normal reorganization but gives creditors less time to negotiate.

The going-concern value of "New GM" was estimated at \$50–70 billion. The distribution looked like:
- Government/DIP claims: satisfied through equity stake in New GM
- UAW VEBA: received 17.5% equity stake in New GM
- Unsecured bondholders (\$27 billion face): received 10% equity stake in New GM + warrants
- Old equity: effectively zero

**What did the unsecured bondholders actually recover?**

New GM went public in November 2010 at a \$33 billion market cap (partial IPO). The 10% stake received by bondholders was worth approximately \$3.3 billion at IPO — against a \$27 billion face value = **~12 cents recovered on a nominal basis**, approximately in line with the pre-bankruptcy trading price of 15–20 cents.

**The controversy:** Unsecured bondholders argued (loudly) that the UAW VEBA received preferential treatment — getting a higher effective recovery than the bonds, despite being at the same seniority level legally. This became a landmark debate about "out-of-court" pressure distorting the absolute priority rule. The courts ultimately approved the plan, and bondholders who bought at 15 cents recovered approximately 12 cents in equity — a small loss on the distressed entry price, but far below what they'd recover in a purely legal priority waterfall.

#### Worked example: GM unsecured bond recovery math

A distressed fund bought \$100M face of GM senior unsecured bonds in May 2009 at **17 cents** = \$17M purchase price.

Recovered in the plan: 10% of New GM equity. At IPO (\$33B market cap), the total bondholder stake is \$3.3B. Proportional to \$100M face / \$27B total unsecured = **0.37% of the bondholder pool**, worth:
```
0.37% × $3.3B = approximately $12.2M
```
Plus warrants worth approximately \$0.8M at IPO price.

**Total recovery: \$13M on a \$17M investment = 76% of invested capital = a loss of 24%.**

But the fund sold the New GM equity at \$38/share in 2011 (post-lockup expiry), when the total stake was worth ~\$15M. Final result: roughly break-even on the distressed purchase — not the intended 15–20% IRR.

**Lesson:** Even buying bonds at distressed levels is no guarantee when political dynamics distort the legal waterfall. Distressed investing in government-involved bankruptcies carries extra risks that don't appear in a pure legal analysis.

### Case study 2: Sri Lanka Sovereign Default (2022) — Applying credit spread logic to sovereigns

In April 2022, Sri Lanka became the first Asia-Pacific country to default on its sovereign debt since Pakistan in 1999. Sri Lanka had \$12.55 billion of international sovereign bonds outstanding when it suspended external debt payments.

**The credit spread warning signs:**

In January 2021, Sri Lanka's international bonds maturing in 2025 (the "SLB 2025") were trading at yields of approximately 8.5%, against a US Treasury rate of 1.0% — implying a credit spread of ~750 bps. That was already deep into distressed territory for a sovereign bond.

By January 2022, the spread had blown out to 1,850 bps (yield ~20.5%, UST ~1.6%). The spread implied:
```
PD × LGD = 18.5% per year
```
Using a typical sovereign LGD of ~50% (sovereigns tend to restructure rather than liquidate):
```
PD = 18.5% / 50% = 37% annual default probability
```
Over 3 years: cumulative PD ≈ 1 − (1−0.37)³ = **75%** — the market was almost certain of default.

**The restructuring math:**

Sri Lanka's bondholders ultimately agreed to restructuring terms in 2024. The headline: bonds exchanged for new bonds with:
- 30% haircut on the principal (face value cut from 100 to 70)
- Maturity extended by 10+ years
- Coupon reduced from average ~6% to 4%

The net present value loss to bondholders at a 12% discount rate (reflecting ongoing country risk) was approximately **45–55 cents on the dollar** compared to the original contractual cash flows.

Investors who bought SLB 2025 at 30 cents in August 2022 (near the trough) received new bonds worth approximately 45 cents in NPV terms in the 2024 restructuring — a **50% gain in 2 years**, or roughly 22% annualized IRR.

**The sovereign vs corporate difference:** Sovereigns cannot be liquidated (you can't foreclose on a country). The entire recovery comes from negotiation and the country's willingness to repay. IMF involvement (Sri Lanka received a \$3 billion program) is critical — the IMF's "comparability of treatment" requirement means the sovereign must negotiate with bondholders at least as favorably as with bilateral lenders (China, Japan), providing a floor for the restructuring.

### Case study 3: Evergrande (2021–2024) — When the "too big to fail" assumption fails

China Evergrande Group, once the world's most indebted property developer with \$300+ billion in total liabilities, began missing bond payments in September 2021. Its offshore dollar bonds (totaling ~\$19 billion) are one of the largest distressed situations in history.

**The CDS and spread dynamics:**

By September 2021, Evergrande's 5-year CDS spread had blown out from ~300 bps in early 2021 to over 6,000 bps — implying near-certain default (6,000 bps spread / 60% LGD = implied 100% annual PD). For context:
- 6,000 bps CDS spread = \$6 million annual premium to protect \$100 million of notional for 5 years
- Most protection sellers disappeared from the market; the CDS became illiquid

**The capital structure complexity:**

Unlike a US corporate Chapter 11, China's restructuring framework is different:
- Onshore creditors (domestic banks, suppliers, homebuyers) sit in a separately regulated Chinese legal framework
- Offshore dollar bondholders are structurally subordinated — they hold claims on offshore subsidiaries, which in turn hold claims on the onshore operating entities
- The Chinese government's priority was homebuyer protection (completing ~1.2 million presold apartments) over offshore bondholder recovery

**The recovery outcome:**

As of early 2024, Evergrande had filed for Chapter 15 protection in the US and bankruptcy in Hong Kong. Offshore dollar bondholders received a restructuring proposal offering new bonds equivalent to roughly **20–25 cents on the dollar** — consistent with the bonds' trading prices at the time (20–30 cents). The restructuring collapsed when Evergrande's chairman was placed under investigation in September 2023, and as of mid-2024 the situation remained unresolved.

**The key valuation lesson:** When investing in emerging-market corporate bonds, the legal framework for enforcing priority differs dramatically from US/European bankruptcy law. Offshore bondholders in Chinese SOE-adjacent companies lack the recourse to court-enforced priority that US creditors rely on. The credit spread must include a "jurisdiction risk" premium that is absent from straightforward US corporate credit analysis.

#### Worked example: Evergrande offshore bond — the basis trade

In February 2021, Evergrande's 8.25% bonds due 2022 traded at 78 cents, yielding approximately 25% to maturity (vs UST 1Y at 0.1%). The spread was ~2,490 bps.

The implied 1-year default probability:
```
2,490 bps / 6,000 bps (LGD for offshore bonds with jurisdiction risk) = 41.5% annual PD
```

(Using LGD = 60% gives PD = 41.5%; using LGD = 80% for jurisdiction-adjusted offshore gives PD = 31%.)

A distressed fund that bought at 78 cents in Feb 2021 and sold at 25 cents in Feb 2022 (as default became undeniable) lost:
```
Loss = (25 − 78) / 78 = −68% in 12 months
```
This catastrophic result illustrates that distressed bonds can get *much more distressed* before restructuring — the timing risk of distressed investing in opaque, politically driven situations can overwhelm even a correct directional thesis on recovery.

### Case study 4: Caesars Entertainment (2015) — The battle over the fulcrum

Caesars Entertainment filed for Chapter 11 in January 2015 with approximately \$18 billion of consolidated debt. The capital structure included:
- **First-lien bank debt and bonds:** ~\$7.7 billion
- **Second-lien bonds:** ~\$5.2 billion
- **Senior unsecured bonds:** ~\$5.5 billion
- **Subordinated bonds:** ~\$3.7 billion (at the operating subsidiary level)

The central dispute: what was Caesars' enterprise value? The company argued around \$10–11 billion. Senior creditors argued \$12–13 billion. The difference of \$1–2 billion determined whether the second-lien bonds received meaningful recovery or zero — making the second-lien the contested fulcrum security.

Distressed investors including Elliott Management and Paul Singer's firm bought second-lien bonds at prices implying zero recovery (pennies on the dollar) and then fought hard for a higher EV estimate. After two years of litigation and negotiation, Caesars emerged from bankruptcy in 2017 with a plan that gave second-lien holders equity in the new company — validating the higher EV argument. The distressed buyers made 2–5× their investment over roughly 3 years.

Caesars Entertainment filed for Chapter 11 in January 2015 with approximately \$18 billion of consolidated debt. The capital structure included:
- **First-lien bank debt and bonds:** ~\$7.7 billion
- **Second-lien bonds:** ~\$5.2 billion
- **Senior unsecured bonds:** ~\$5.5 billion
- **Subordinated bonds:** ~\$3.7 billion (at the operating subsidiary level)

The central dispute: what was Caesars' enterprise value? The company argued around \$10–11 billion. Senior creditors argued \$12–13 billion. The difference of \$1–2 billion determined whether the second-lien bonds received meaningful recovery or zero — making the second-lien the contested fulcrum security.

Distressed investors including Elliott Management and Paul Singer's firm bought second-lien bonds at prices implying zero recovery (pennies on the dollar) and then fought hard for a higher EV estimate. After two years of litigation and negotiation, Caesars emerged from bankruptcy in 2017 with a plan that gave second-lien holders equity in the new company — validating the higher EV argument. The distressed buyers made 2–5× their investment over roughly 3 years.

### Case study 2: Hertz (2020) — When equity survives bankruptcy

When Hertz filed Chapter 11 in May 2020, the equity was essentially worthless — the company had \$19 billion of debt against a fleet that was falling in value (early COVID panic). Retail investors bought Hertz stock at \$0.40–\$0.80 purely as a lottery ticket.

But Hertz's key assets were its cars, and by late 2020 the used-car market exploded due to supply chain disruptions and surging consumer demand. The fleet's value recovered dramatically. By the time Hertz emerged from bankruptcy in 2021, the enterprise value was large enough that even original equity holders received something — warrants and a small cash recovery.

This is the exception, not the rule. But it illustrates a crucial valuation lesson: **the right EV estimate is the one that reflects the business's future cash flows, not its current distressed state.** EV estimates in bankruptcy can shift dramatically with changed circumstances.

### Case study 3: Using UST yields as the spread anchor (2022)

The surge in UST 10-year yields from 1.52% (end-2021) to 3.88% (end-2022) — an increase of 236 bps — created a fascinating dynamic. Because credit spreads are measured *over* Treasuries, the absolute yields on corporate bonds rose even more than Treasuries:

- **IG corporate bond yields:** Rose from ~2.3% (Jan 2022) to ~5.7% (Dec 2022) — a 340 bps increase. The extra 104 bps came from spread widening (IG spreads went from ~90 bps to ~150 bps).
- **HY corporate bond yields:** Rose from ~4.3% to ~8.5% — a 420 bps increase.

For an investor who bought IG bonds in January 2022 at a 2.3% yield and had to mark-to-market in December 2022 at 5.7%, the price loss was enormous (higher yield = lower price). A 10-year IG bond lost roughly 15–20% of its market value in 2022 — comparable to equity losses.

This illustrates that even "safe" investment-grade bonds carry significant interest rate risk (duration) in addition to credit risk. The credit spread portion of the yield loss was smaller; the rate-driven portion was the dominant factor. For our cross-link on this dynamic, see the [bond valuation post on yield, duration, and convexity](/blog/trading/asset-valuation/bond-valuation-yield-duration-convexity) and the discussion of WACC and discount rates in [discount rates in practice](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta).

### Connecting to LBO analysis and real options

Distressed debt sits at the intersection of two other valuation frameworks covered in this series:

**LBO connection:** A leveraged buyout firm essentially creates the preconditions for distress — high leverage, interest burden. When an LBO goes wrong (revenues miss, rates rise), the debt trades at distressed prices and the original equity is wiped out. Distressed investors and LBO sponsors sometimes find themselves negotiating the same restructuring. See [LBO valuation](/blog/trading/asset-valuation/leveraged-buyout-lbo-valuation-private-equity) for how these capital structures are built.

**Real options connection:** The Merton model is, at its core, an options model applied to the balance sheet. The same framework — option value, volatility, time — drives both equity value in distress and real options embedded in strategic investments. A company's ability to delay a default (by refinancing, asset sales) is itself a real option with value. See [real options valuation](/blog/trading/asset-valuation/real-options-valuation-flexibility-strategic-investments) for the broader framework.

---

## Further Reading and Cross-Links

**Within this series:**
- [Bond Valuation: Yield, Duration, and Convexity](/blog/trading/asset-valuation/bond-valuation-yield-duration-convexity) — the mechanics of bond pricing and interest rate risk that underpin credit spread analysis.
- [Discount Rates in Practice: WACC, Cost of Equity, and Unlevered Beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta) — how the risk-free rate and equity risk premium combine to set discount rates; directly relevant to going-concern DCF in bankruptcy.
- [LBO Valuation: Private Equity's Leverage Playbook](/blog/trading/asset-valuation/leveraged-buyout-lbo-valuation-private-equity) — the capital structure engineering that creates distressed situations.
- [Real Options Valuation: Flexibility and Strategic Investments](/blog/trading/asset-valuation/real-options-valuation-flexibility-strategic-investments) — the option-theoretic framework that the Merton model draws on.

**Academic and practitioner resources:**
- Merton, R.C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates." *Journal of Finance*, 29(2), 449–470. — The original structural model paper.
- Moody's Investors Service, "Annual Default Study: Corporate Default and Recovery Rates, 1920–2023" — The primary source for recovery rate data by seniority.
- Altman, E.I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy." *Journal of Finance*, 23(4), 589–609. — The original Z-Score model for predicting distress.
- Roe, M.J. & Skeel, D.A. (2010). "Assessing the Chrysler Bankruptcy." *Michigan Law Review*, 108(5), 727–772. — A real-world case study in distressed valuation and reorganization.

**Market data:**
- FRED (Federal Reserve Bank of St. Louis): BAMLC0A0CM (IG spreads), BAMLH0A0HYM2 (HY spreads), H.15 (UST yields) — Free, updated daily.
- Moody's Ultimate Recovery Database — subscription, the gold standard for recovery statistics.
- Distressed debt screeners: Bloomberg DRSK function, Reorg Research for restructuring intelligence.

---

## Sources and Further Reading

- Federal Reserve H.15 Selected Interest Rates. US 10-Year Treasury yields 2010–2024. federalreserve.gov. As of December 31, 2024.
- ICE BofA US Corporate Index Option-Adjusted Spread (BAMLC0A0CM). FRED, Federal Reserve Bank of St. Louis. As of December 31, 2024.
- ICE BofA US High Yield Index Option-Adjusted Spread (BAMLH0A0HYM2). FRED, Federal Reserve Bank of St. Louis. As of December 31, 2024.
- Moody's Investors Service. "Annual Default Study: Corporate Default and Recovery Rates, 1920–2023." January 2024. (Recovery rate statistics by seniority class from Ultimate Recovery Database.)
- Damodaran, A. "Equity Risk Premiums (ERP): Determinants, Estimation and Implications — The 2025 Edition." Stern School of Business, NYU. January 2025.
- Merton, R.C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates." *Journal of Finance*, 29(2), 449–470.
- SIFMA. "US Corporate Bonds Statistics." sifma.org. As of Q4 2024. (US IG market size ~\$10 trillion, HY ~\$1.4 trillion.)
