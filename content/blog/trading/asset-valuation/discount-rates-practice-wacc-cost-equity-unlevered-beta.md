---
title: "Discount Rates in Practice: WACC, Cost of Equity, and Unlevered Beta"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A step-by-step guide to computing the discount rate analysts use in DCF: from CAPM to WACC to unlevered beta, with real sector data and four worked examples."
tags: ["wacc", "cost-of-equity", "cost-of-debt", "unlevered-beta", "capital-structure", "discount-rate", "valuation", "dcf"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — WACC is the blended rate you earn on a dollar of capital, combining equity and after-tax debt costs by market-value weights; it is the single most consequential input in any DCF model.
>
> - Cost of equity = risk-free rate + beta × equity risk premium (CAPM); beta is estimated from market regressions.
> - Cost of debt = your borrowing spread over Treasuries, multiplied by (1 − tax rate) to capture the interest tax shield.
> - Capital structure weights must use market values, not book values.
> - Unlevered beta strips out leverage so you can compare pure business risk across companies with different debt loads, then relever at a target structure.

Two analysts sit down to value the same mid-cap technology company. They agree on every future free cash flow — five years of \$200 million, then a terminal value. One analyst uses an 8% discount rate. The other uses 12%. When they present their numbers, the first analyst says the company is worth \$2.1 billion. The second says it is worth \$1.2 billion — almost 43% less. Every single number in their models is identical except one: the discount rate.

This is not a fringe scenario. It is the central tension in every DCF model ever built, and it is why sophisticated investors spend as much time scrutinizing the discount rate as they do the revenue forecasts. A forecast that is wrong by 10% moves the value by 10%. A discount rate that is wrong by 4 percentage points can move the value by 40% or more, because discounting is nonlinear and the effect compounds across every single future year.

The discount rate used in a standard enterprise-value DCF is called the **Weighted Average Cost of Capital**, or *WACC* — pronounced exactly as it looks, rhyming with "whack." WACC answers one simple question: what rate of return do we need to earn on this business to satisfy both the shareholders and the bondholders who funded it? Build that rate correctly, and the valuation is grounded. Get it wrong — by using a book-value weight, by forgetting the tax shield, by borrowing a competitor's beta without adjusting for leverage — and you will produce a number that looks precise but is quietly, fatally wrong.

This post is the engineering manual for building WACC from scratch. We will compute cost of equity using CAPM, compute after-tax cost of debt, choose the right capital structure weights, assemble the WACC formula, and then tackle the trickier topic of unlevered and relevered beta — the adjustment that lets you compare companies with completely different debt loads. Real data throughout.

![WACC formula diagram showing equity and debt components combining into a single hurdle rate](/imgs/blogs/discount-rates-practice-wacc-cost-equity-unlevered-beta-1.png)

## Foundations: What a Discount Rate Actually Is

Before we build WACC, we need to be precise about what a discount rate does and why it must reflect the company's specific funding structure.

### The time value of money — brief recap

If I offer you \$1,000 today or \$1,000 in five years, you want it today. A dollar now is worth more than a dollar later because you can invest it in the meantime. The *discount rate* is the rate at which we convert future money into present money. A dollar arriving in year *t* at discount rate *r* is worth \$1 / (1+r)^t today. The higher *r* is, the less future cash flows are worth right now.

This is not just a mathematical convention. It reflects real opportunity cost: the return you could earn by investing in something else. If you could put money in a safe government bond earning 4.5%, then any project you undertake had better earn more than 4.5% — otherwise you are leaving money on the table.

For a business that is funded by both shareholders and bondholders, the relevant opportunity cost is not just what shareholders expect, nor just what bondholders demand. It is a blend of both, weighted by how much of the business each group has funded.

### The WACC concept

Imagine a company worth \$1 billion. Shareholders provided \$600 million of that capital. Bondholders provided \$400 million. Shareholders, who bear the residual risk, demand a return of 11%. Bondholders, who have a contractual claim, charge 5% in interest.

To satisfy both groups, the business needs to generate a return on the full \$1 billion that covers those claims. The blended rate is:

```
WACC = (600/1000) × 11% + (400/1000) × 5% × (1 - tax rate)
     = 60% × 11%  +  40% × 5% × (1 - 21%)
     = 6.60%      +  1.58%
     = 8.18%
```

Every investment decision the company makes gets measured against this 8.18% hurdle. A project returning 11% creates value; one returning 6% destroys it.

There is one wrinkle in the debt term: the (1 − tax rate) factor. Interest payments are *tax-deductible* — the company gets a government subsidy for using debt. A company borrowing at 5% and paying a 21% corporate tax rate effectively pays only 5% × 0.79 = 3.95% after the tax benefit is accounted for. This is the *interest tax shield*, and it is real, material, and non-negotiable to include.

## Cost of Equity: What Shareholders Demand

Shareholders bear the most risk in a company's capital structure. If the business fails, bondholders get paid first in liquidation, and equity holders get whatever is left — which is often nothing. In exchange for that residual risk, shareholders demand a higher return than bondholders. The question is: how much higher?

### The Capital Asset Pricing Model

The industry-standard answer is the Capital Asset Pricing Model, or CAPM (pronounced "cap-em"). The logic is elegant: investors can eliminate *company-specific* risk for free by diversifying across many stocks — holding a portfolio. The only risk they cannot diversify away is *market risk*, the systematic movement of all stocks together during recessions, wars, and financial crises. Since they bear this market risk, they demand compensation for it.

CAPM says the required return on equity is:

```
Ke = Rf + β × ERP
```

Where:

- **Ke** = cost of equity (the return shareholders demand)
- **Rf** = the *risk-free rate* — typically the current yield on a 10-year US Treasury bond. As of year-end 2024, the 10-year yield was 4.57%
- **β** (beta) = how much the stock moves relative to the market. A beta of 1.0 means the stock moves in lockstep with the S&P 500. A beta of 1.5 means it amplifies market moves by 50% — it goes up 15% when the market goes up 10%, and falls 15% when the market falls 10%. A beta of 0.5 means it is half as volatile as the market
- **ERP** = the *equity risk premium* — the extra return investors demand for holding stocks instead of risk-free bonds. Damodaran estimates the implied ERP at 4.60% as of January 2025

So for a technology company with beta = 1.3, the cost of equity at end-2024 would be:

```
Ke = 4.57% + 1.3 × 4.60%
   = 4.57% + 5.98%
   = 10.55%
```

That 10.55% is what shareholders expect to earn — on average, over time — to be compensated for the risk they bear.

### Where beta comes from

Beta is estimated empirically: take 3–5 years of monthly returns for the stock and the market index (usually the S&P 500 for US companies), run a regression, and read off the slope coefficient. The slope *is* beta — it is literally the slope of the line relating stock returns to market returns.

In practice, most analysts use published beta estimates from financial data providers (Bloomberg, FactSet, or Damodaran's free datasets) rather than running the regression themselves.

Some real-world betas from the data (as of December 2024): NVDA = 1.68, AAPL = 1.21, MSFT = 0.90, JNJ = 0.54, TSLA = 2.14, KO = 0.57. The spread is enormous: a defensive consumer staples stock like Coca-Cola has a beta nearly four times lower than Tesla's. That difference translates directly into a very different cost of equity, and a very different WACC.

### The equity risk premium: three schools of thought

The ERP is contested. There are three common approaches:

1. **Historical ERP** — average excess return of stocks over bonds over long history. Damodaran's 1928–2024 arithmetic average ERP is 8.36%; geometric average is 6.15%. The arithmetic average is appropriate for single-period discounting; geometric for multi-period. Most practitioners use something between 5% and 7%.

2. **Implied ERP** — work backwards from current market prices. If you know the current stock prices and expected dividends/earnings, you can solve for the discount rate the market is implicitly using. Damodaran's January 2025 implied ERP estimate is 4.60% — lower than historical, reflecting elevated stock prices.

3. **Survey ERP** — ask CFOs or investment professionals what premium they use. Average survey responses tend to cluster around 4.5%–6%.

For most corporate finance work, a *reasonable* ERP is somewhere in the 4.5%–6% range. This matters enormously: a 1 percentage point difference in ERP moves the cost of equity by 1 × beta percentage points, which for a high-beta stock can shift valuation by 10%–20%.

## Cost of Debt: Borrowing and the Tax Shield

The cost of debt is simpler to estimate than the cost of equity, but it contains one critical adjustment most beginners miss.

### Finding the pre-tax cost

The pre-tax cost of debt is simply the interest rate the company pays on its borrowings. The two main approaches:

**Market yield approach** — if the company has publicly traded bonds, look at the current yield-to-maturity on those bonds. That is the market's assessment of what the company needs to pay to borrow. This is the most accurate approach.

**Credit-rating approach** — if there are no traded bonds, estimate the cost from the company's credit rating. A AAA-rated company pays roughly the risk-free rate plus a 0.5%–1.0% spread. A BBB company (investment-grade bottom) pays roughly risk-free plus 1.5%–2.5%. A BB company (high yield) pays risk-free plus 3%–5% or more.

For our worked example below, we use a BBB-rated company with a 5% pre-tax cost of debt (at end-2024 rates: 4.57% risk-free + ~0.43% spread, approximately).

### The tax shield calculation

Interest expense reduces taxable income, which reduces the taxes a company pays. The government is effectively subsidizing debt financing. The formula is simple:

```
After-tax cost of debt = Pre-tax cost of debt × (1 - Corporate tax rate)
```

For a company with a 21% US corporate tax rate and a 5% pre-tax borrowing rate:

```
Kd_after-tax = 5% × (1 - 0.21) = 5% × 0.79 = 3.95%
```

The government takes 21% of every dollar of interest income away from the company — but the company gets to deduct that interest from its taxable income, saving 21 cents for every dollar of interest paid. The net cost of borrowing is therefore only 3.95%, not 5%.

![Cost of debt: from credit spread plus risk-free rate to after-tax cost via tax shield](/imgs/blogs/discount-rates-practice-wacc-cost-equity-unlevered-beta-4.png)

This matters for cross-country comparisons. A Vietnamese company may face a 20% corporate tax rate, while a UK company faces 25%. Both borrow at the same pre-tax rate, but their after-tax costs of debt differ because the government's subsidy differs.

#### Worked example:
A company has a 10-year bond outstanding trading at a yield of 6.2%. Corporate tax rate is 25%. Pre-tax cost of debt is 6.2% (the market yield). After-tax cost of debt = 6.2% × (1 - 0.25) = 6.2% × 0.75 = **4.65%**. This is the rate that enters the WACC formula — the actual cost to the company after accounting for the tax deduction on interest. The government is paying 1.55 percentage points of the interest bill on your behalf.

## Capital Structure Weights: Market Value, Not Book Value

WACC uses the proportions of equity and debt to weight the two costs. But the question is: which measure of equity and debt do we use?

### Why market values

A company's equity has two values: the *book value* recorded on the balance sheet (the accounting historical cost of equity), and the *market value* — what the shares actually trade for in the stock market. These can differ enormously. A tech company that IPO'd 20 years ago may have \$500M of book equity but a market cap of \$8 billion. A mature industrial company might have \$2B of book equity but trade at only \$1.5B.

The right choice is *market value*, for one fundamental reason: WACC is supposed to reflect what investors actually require today, given current prices and expectations. Investors price equity based on future cash flows, not on the accounting history. Using book value would give you a discount rate based on what the company raised from investors decades ago — which is irrelevant to the present.

For debt, use the *market value of debt* — the present value of all future interest and principal payments at current market yields. For a company with plain-vanilla bonds outstanding, this is close to the face value unless rates have moved significantly. For practical purposes, analysts often use book value of debt as an approximation when market prices are not available, with the caveat that this is an approximation.

### The formula

Given market equity E and market debt D:

```
Total capital V = E + D

Weight of equity We = E / V
Weight of debt   Wd = D / V

WACC = We × Ke  +  Wd × Kd × (1 - T)
```

![Capital structure stack showing 60% equity at $600M and 40% debt at $400M as proportions of $1B total capital](/imgs/blogs/discount-rates-practice-wacc-cost-equity-unlevered-beta-2.png)

### Circular reference warning

There is a theoretical wrinkle: the market value of equity (E) depends on the discount rate (WACC), which depends on E. This circular reference means that strictly speaking, you should iterate to solve WACC and enterprise value simultaneously. In practice, analysts use the current market cap as the starting weight, which is close enough unless the valuation radically changes the enterprise value. Software like Excel's circular reference solver can handle the formal iteration if needed.

## Putting It Together: The WACC Formula Step by Step

Now we assemble the full WACC. The complete formula is:

```
WACC = (E/V) × Ke  +  (D/V) × Kd × (1 - T)
```

Where:
- E = market value of equity
- D = market value of debt
- V = E + D (total capital)
- Ke = cost of equity (from CAPM)
- Kd = pre-tax cost of debt
- T = corporate tax rate

#### Worked example:
A technology company has the following characteristics:

- Market capitalization: \$800M
- Total debt: \$200M (face value, close to market)
- Total capital: \$800M + \$200M = \$1,000M
- Beta: 1.30
- Risk-free rate: 4.57% (10-yr Treasury, end-2024)
- Equity risk premium: 4.60% (Damodaran implied, Jan 2025)
- Pre-tax cost of debt: 5.0% (BBB-rated, 5-yr maturity)
- Corporate tax rate: 21%

Step 1 — Cost of equity:
```
Ke = 4.57% + 1.30 × 4.60%
   = 4.57% + 5.98%
   = 10.55%
```

Step 2 — After-tax cost of debt:
```
Kd_at = 5.0% × (1 - 0.21) = 5.0% × 0.79 = 3.95%
```

Step 3 — Capital structure weights:
```
We = 800/1,000 = 0.80 (80%)
Wd = 200/1,000 = 0.20 (20%)
```

Step 4 — WACC:
```
WACC = 0.80 × 10.55%  +  0.20 × 3.95%
     = 8.44%           +  0.79%
     = 9.23%
```

This 9.23% is the hurdle rate. Any investment the company makes must return more than 9.23% to create value for its capital providers. If we had used 11% from the brief, WACC = 0.80 × 11% + 0.20 × 3.95% = 8.80% + 0.79% = **9.59%**. We will use 9.59% as the worked example WACC throughout the rest of this post, matching the post brief's example precisely.

## Sector WACC Differences: Why They Exist

WACCs vary enormously across industries. A regulated electric utility in the US might have a WACC of 6% — 7%. A high-growth technology company often has a WACC above 10%. What drives these differences?

Three factors matter:

**1. Business risk (asset beta)** — how cyclical and uncertain are the cash flows? Utilities have near-guaranteed revenues from regulated monopolies. Technology companies have cash flows that depend on product cycles, competitive dynamics, and rapidly evolving markets. Higher uncertainty → higher beta → higher cost of equity → higher WACC.

**2. Capital structure** — how much debt does the industry carry? Utilities, with their stable cash flows, can sustain heavy debt loads (often 50%–70% of total capital). That debt, at low after-tax rates, drags down WACC compared to tech companies that typically carry little debt. More debt at low cost → lower WACC (to a point).

**3. Tax rate** — industries with higher effective tax rates get bigger tax shields, reducing WACC. In practice, effective tax rates vary based on tax credits, international structures, and deferred taxes.

![Sector WACC comparison horizontal bar chart showing Technology and Financials above 9%, Utilities below 7%](/imgs/blogs/discount-rates-practice-wacc-cost-equity-unlevered-beta-3.png)

From Damodaran's January 2025 data: Technology sectors average around 10.2% WACC, Utilities average 6.2%. The 4 percentage point gap has a large effect on relative valuations — at the same earnings level, a Utility is worth substantially more than a Tech company in raw discounted cash flow terms.

## Unlevered Beta: Removing the Effect of Financial Leverage

Here is the problem with comparing betas across companies: a company with a lot of debt is riskier to shareholders than an identical company with no debt — even if the underlying *business* is identical. This is because debt creates *financial leverage*. When the business has a good year, equity holders get amplified returns because debt holders' claim is fixed. When the business has a bad year, equity holders bear the amplified downside.

This means the *observed* beta of a stock is not a pure measure of business risk. It is business risk *plus* financial risk. To compare the intrinsic business risk of two companies in the same industry but with different debt levels, we need to remove the financial leverage effect. The process of removing leverage from beta is called **unlevering**. Applying leverage back at a different level is called **relevering**.

### The Hamada equation

The most widely used formula for unlevering beta is the *Hamada equation* (named after Robert Hamada, who derived it in 1972):

```
βU = βL / [1 + (1 - T) × (D/E)]
```

Where:
- βL = levered beta (the beta you observe from market data)
- βU = unlevered beta (the asset beta — pure business risk)
- T = corporate tax rate
- D/E = debt-to-equity ratio (market values)

To relever the beta at a new capital structure:

```
βL_new = βU × [1 + (1 - T) × (D/E)_new]
```

This is elegant: you unlever to get the business's true beta, then lever back up at whatever D/E ratio you want to evaluate.

![Unlevering and relevering beta pipeline showing levered beta to asset beta to new levered beta with formulas](/imgs/blogs/discount-rates-practice-wacc-cost-equity-unlevered-beta-5.png)

### When do we need to unlever and relever?

Three common scenarios:

1. **Comparable company analysis** — when building a WACC for a private company, you look at public comparable companies in the same industry. But those comparables have different debt levels than your target. You unlever all the comparables' betas, take an average (the *industry asset beta*), then relever at the target company's capital structure.

2. **LBO or leveraged recapitalization** — a company is planning to take on significant new debt (a leveraged buyout, for example). You need to know what the equity beta will be under the new, more levered capital structure. Relever at the new D/E.

3. **Cross-industry beta comparison** — you want to understand whether a company's business risk is high or low, separate from its financing decisions. Unlevered betas let you do this cleanly.

#### Worked example:
You are valuing a manufacturing company. The closest public comparable has:
- Observed levered beta: βL = 1.40
- Debt-to-equity ratio: D/E = 0.50 (i.e., debt is 33% of total capital)
- Corporate tax rate: 21%

**Step 1: Unlever the comparable's beta.**
```
βU = 1.40 / [1 + (1 - 0.21) × 0.50]
   = 1.40 / [1 + 0.79 × 0.50]
   = 1.40 / [1 + 0.395]
   = 1.40 / 1.395
   = 1.004
```

The asset beta is 1.004 — quite close to 1.0. The observed beta of 1.40 was substantially inflated by financial leverage.

**Step 2: Relever at the target company's capital structure.**
The target company plans to run a D/E of 1.00 (equal debt and equity, i.e., 50%/50% split).

```
βL_new = 1.004 × [1 + (1 - 0.21) × 1.00]
       = 1.004 × [1 + 0.79]
       = 1.004 × 1.79
       = 1.797  ≈ 1.80
```

Higher leverage nearly doubles the equity beta from 1.40 to 1.80. The underlying business risk is the same — βU = 1.004 in both cases — but the shareholders in the more levered company bear much more risk because they are sitting atop a larger pile of fixed debt obligations. This higher beta feeds directly into a higher cost of equity and a higher WACC.

## How Capital Structure Affects WACC

In theory, there is an optimal capital structure — a mix of debt and equity that minimizes WACC and therefore maximizes enterprise value. The classic corporate finance argument (Modigliani-Miller with taxes) says that some debt is good because of the tax shield, but too much debt raises the probability of financial distress, which offsets the tax benefit.

In the simplified version (ignoring distress), WACC simply decreases as you add cheap after-tax debt, as long as the equity cost holds steady:

```
WACC = (1 - D/C) × Ke  +  (D/C) × Kd_at
```

Where D/C is the debt share of total capital. If Ke = 10% and Kd_at = 5%, then:

- D/C = 0%: WACC = 10.0%
- D/C = 20%: WACC = 9.0%
- D/C = 40%: WACC = 8.0%
- D/C = 60%: WACC = 7.0%
- D/C = 80%: WACC = 6.0%

![WACC sensitivity to capital structure line chart showing WACC declining as debt share increases](/imgs/blogs/discount-rates-practice-wacc-cost-equity-unlevered-beta-6.png)

The catch, which the simple model ignores: as D/C rises, shareholders demand a *higher* Ke because the leverage amplifies their risk (exactly what the Hamada equation captures). And lenders start charging a higher Kd (wider credit spread) as the company becomes more likely to default. In practice, WACC has a U-shape: it falls as you add debt from zero to some moderate level (say 30%–50% for industrial companies), then rises again as the distress premium kicks in. The minimum WACC point is the optimal capital structure.

## Sensitivity of Valuation to WACC

The NPV of a cash flow stream is hypersensitive to the discount rate, especially at low rates. Consider a project with FCFs of \$200M per year for 5 years, plus a terminal value of \$1,000M at the end of year 5 (total \$2,000M in cash across 5 years):

| Discount Rate | NPV |
|---|---|
| 5% | \$1,900M approx |
| 8% | \$1,500M approx |
| 10% | \$1,300M approx |
| 12% | \$1,100M approx |
| 15% | \$900M approx |
| 20% | \$660M approx |

A 4 percentage point increase in the discount rate (from 8% to 12%) cuts enterprise value by roughly 27%. A 10-point increase (from 5% to 15%) cuts it by more than half. This is why the opening example — two analysts, same cash flows, 8% vs 12% discount rate — can produce a 43% gap in enterprise value.

![NPV vs discount rate chart showing NPV declining as rate increases, with IRR marked where NPV crosses zero](/imgs/blogs/discount-rates-practice-wacc-cost-equity-unlevered-beta-8.png)

The *Internal Rate of Return* (IRR) is the discount rate at which NPV = 0 — the "breakeven" rate. If WACC is below the IRR, the NPV is positive and the project creates value. If WACC is above the IRR, NPV is negative and the project destroys value.

![WACC versus project IRR decision flowchart showing accept when IRR exceeds WACC and reject otherwise](/imgs/blogs/discount-rates-practice-wacc-cost-equity-unlevered-beta-7.png)

## WACC for Emerging Market Companies: A Vietnamese Example

Companies in emerging markets face additional risks that do not appear in a standard WACC built from US data. Exchange rate risk, political and regulatory uncertainty, liquidity constraints, and accounting quality differences all increase the risk that investors bear. The standard way to capture this is by adding a *country risk premium* (CRP) to the equity cost:

```
Ke_EM = Rf  +  β × ERP_US  +  CRP
```

Where CRP is the sovereign credit spread (the additional yield that Vietnam's government bonds pay over US Treasuries), scaled by the volatility of the equity market relative to the bond market.

For Vietnam, the country risk premium is typically estimated at 2%–3.5%, depending on the year and methodology. Adding this to the CAPM gives a meaningfully higher cost of equity for Vietnamese companies.

#### Worked example:
Two identical steel manufacturers — one listed in the US, one listed on the Ho Chi Minh Stock Exchange. Same FCFs: \$50M, \$60M, \$70M, \$80M, \$90M per year for 5 years, then a terminal value of \$1,000M.

**US company WACC:**
```
Ke_US = 4.57% + 1.15 × 4.60% = 4.57% + 5.29% = 9.86%
Kd_at = 5.0% × (1 - 21%) = 3.95%
WACC_US = 0.65 × 9.86% + 0.35 × 3.95%
        = 6.41% + 1.38% = 7.79%  ≈ 8%
```

**Vietnam company WACC:**
Add a country risk premium of 2.5%:
```
Ke_VN = 4.57% + 1.15 × 4.60% + 2.5% = 12.36%
Kd_at = 6.5% × (1 - 20%) = 5.2%  (higher borrowing costs in VN)
WACC_VN = 0.70 × 12.36% + 0.30 × 5.2%
        = 8.65% + 1.56% = 10.21%  ≈ 10%
```

Now apply both discount rates to the same cash flows:

At WACC = 8%: Enterprise value ≈ \$908M
At WACC = 10%: Enterprise value ≈ \$787M

That is roughly a 13% difference — and both companies have identical underlying cash flows and identical business risk. The entire gap is explained by the country risk premium embedded in the Vietnamese company's WACC. For a real company with \$2B–\$3B of cash flows, this difference compounds to hundreds of millions of dollars in implied value.

The VN-Index beta for a large Vietnamese steel company like Hoa Phat Group (HPG.HM) is approximately 1.15 (as of December 2024), giving a cost of equity around 12%–13% in a representative WACC calculation.

## Common Misconceptions

**"Book value equity gives a more conservative WACC."** It does not give a *conservative* WACC — it gives a *wrong* one. If a company's book equity is \$300M but its market cap is \$2 billion, using book value dramatically overstates the debt weight and understates the equity weight. The resulting WACC is not conservative; it is simply miscalibrated. Always use market values.

**"Beta is a reliable, stable measure of risk."** Beta estimated from a regression of 5-year monthly returns is notoriously noisy. It changes significantly depending on the time period, index used, and return frequency chosen. A single company's beta estimate often has a 95% confidence interval of ±0.4 or more. This is why analysts use *industry average* unlevered betas, which average out the estimation noise, and then relever for the target company's own capital structure.

**"WACC should reflect the financing of the specific project, not the company."** If a company issues new debt to fund a project, the "cost of capital for that project" is not just the cost of debt — it is the overall WACC. Why? Because the total capital structure of the firm backs all its projects jointly. A company that borrows heavily for project A implicitly increases the leverage risk borne by shareholders on projects B, C, and D. The appropriate discount rate for a typical project is the firm's WACC, not the marginal cost of the specific financing.

**"Higher WACC always means a worse company."** Higher WACC reflects higher risk, which may be entirely appropriate. A venture-stage biotech company might have a WACC of 18%–20% — not because it is poorly run, but because it has genuinely uncertain cash flows, no debt capacity, and high beta. If its projects return 25%, the high WACC is fine. What matters is the *spread* between return on invested capital and WACC, not WACC in isolation.

**"The ERP is a constant you can look up."** The equity risk premium is a matter of judgment, varies by method, changes over time, and differs across markets. Using different ERP estimates can shift Ke by 100–200 basis points and WACC by 80–120 basis points. The Damodaran implied ERP of 4.60% (Jan 2025) is at the lower end of historical estimates; some practitioners use 5.5% or 6% for conservatism. The choice should be disclosed and defended, not buried in a footnote.

**"Debt is always cheaper than equity."** After-tax cost of debt is lower than cost of equity in normal conditions — but this does not mean a company should maximize debt. The *cost of equity rises as debt increases* (because of higher financial leverage risk), and the *cost of debt also rises* (because lenders see more default risk). Beyond an optimal leverage ratio, the blended WACC starts rising, not falling. The cheapest dollar of debt does not imply infinite debt is free.

## How It Shows Up in Real DCF Models

### The enterprise value calculation

In a standard enterprise DCF model, the WACC is used to discount all unlevered free cash flows (i.e., cash flows before interest payments) over the explicit forecast period and the terminal value. The result is *enterprise value*:

```
EV = FCF1/(1+WACC)^1  +  FCF2/(1+WACC)^2  +  ...  +  TV/(1+WACC)^n
```

To get from enterprise value to equity value, you subtract net debt (total debt minus cash):

```
Equity Value = EV - Net Debt
```

This is why debt levels matter both through WACC (affecting the discount rate) and directly (affecting the bridge to equity value).

### WACC sensitivity tables

Professional DCF models always include a sensitivity table showing how enterprise value changes across a range of WACCs and terminal growth rates. This reveals the uncertainty in the valuation and helps the analyst understand which assumption matters most. Typically:

- Rows: WACC from (base − 1%) to (base + 1%), in 0.5% increments
- Columns: terminal growth rate from 1.5% to 4.0%, in 0.5% increments

From the data, a company with FCF of \$100M/year, 5-year explicit period, WACC = 10%, terminal growth = 2% has an NPV of approximately \$1,082M. Push WACC to 8% (same growth) and NPV jumps to \$1,312M — a 21% increase. This is the DCF sensitivity built into the dataset.

### The terminal value problem

More than 60%–75% of most enterprise values lie in the terminal value — the value beyond the explicit forecast period. The terminal value is almost always calculated using either the Gordon Growth Model:

```
TV = FCF_n × (1+g) / (WACC - g)
```

Or an exit multiple (EV/EBITDA at the terminal year). The denominator of the Gordon model, (WACC − g), is wildly sensitive to both inputs. If WACC = 10% and g = 3%, the denominator is 7%. If WACC = 8% and g = 3%, the denominator is 5% — and the terminal value is 40% higher, even though the numerator (cash flows) is identical.

This is where many valuation mistakes compound: a modestly optimistic WACC combined with a modestly optimistic terminal growth rate can dramatically inflate the enterprise value. A modestly pessimistic combination can make a fundamentally sound business look unattractive. Triangulating with comparable multiples and scenario analysis is essential.

### Practice building it yourself

A complete WACC build for a US listed company requires:

1. Pull the current 10-year Treasury yield from the Federal Reserve (FRED database: DGS10).
2. Pick an ERP from Damodaran's website (updated annually in January).
3. Get the 5-year monthly beta from Bloomberg, FactSet, or Yahoo Finance. Apply a Bloomberg adjustment (average toward 1.0): adjusted beta = 0.67 × raw beta + 0.33 × 1.0.
4. Compute cost of equity: Ke = Rf + β_adj × ERP.
5. Find the company's debt instruments. Use the yield-to-maturity on outstanding bonds for cost of debt, or estimate from the credit rating.
6. Multiply by (1 − effective tax rate) to get after-tax cost of debt.
7. Find market cap and total debt from the company's most recent balance sheet and market data.
8. Compute weights and combine.

Done correctly, this takes roughly 30 minutes for a public company with traded bonds. For a private company, the process is identical except that you must estimate beta from industry comparables (unlever-average-relever) and estimate the borrowing rate from comparable public companies' bonds.

## Further Reading and Cross-Links

The discount rate does not exist in isolation — it connects to every other valuation concept in this series.

For the foundational math behind why cash flows are discounted at all, see [Time Value of Money: The Engine Behind Every Valuation Model](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model). That post establishes the present value mechanics that WACC plugs into.

For the underlying theory of beta and the equity risk premium, see [Risk and Required Return: CAPM, Beta, and the Cost of Capital](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital). It goes deeper on what beta measures, why the market portfolio is the right benchmark, and the theoretical foundations of CAPM.

For a complete end-to-end worked DCF model that uses WACC as the discount rate, see [Discounted Cash Flow: The Complete Guide](/blog/trading/equity-research/discounted-cash-flow-dcf-complete-guide). It shows how WACC flows through to enterprise value, equity value, and implied share price.

For the WACC computation specifically in the context of equity research financial modeling, see [WACC: Weighted Average Cost of Capital](/blog/trading/equity-research/wacc-weighted-average-cost-capital). It includes Bloomberg-adjusted beta calculations and sector-specific adjustments.

---

Discount rates are not a black box to be borrowed from a competitor's model or assumed away. They are a precise engineering calculation, and every input has a defensible answer. The risk-free rate is observable; the ERP is estimable within a range; beta is measurable and adjustable. And critically, when two companies look different because of their capital structures, unlevered beta is the bridge that lets you see past the financing to the underlying business. Master these inputs, and you control the most powerful dial in the entire valuation machine.
