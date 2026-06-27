---
title: "Valuing Private Companies: Illiquidity Discounts and Adjusted Methods"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "How DCF, comps, and asset-based methods adapt for private companies — with size premia, DLOM discounts, VC revenue multiples, and a full worked example."
tags: ["valuation", "private-companies", "illiquidity-discount", "DLOM", "venture-capital", "DCF", "asset-pricing"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Valuing a private company means taking every public-market technique and adding two to three layers of discount: a higher discount rate (add 2–6% size premium plus 1–5% specific risk), a marketability discount (DLOM of 10–30%), and potentially a minority discount (15–30%) or control premium (20–40%).
>
> - Private companies have no market price, no float, limited information, and illiquid equity — every one of these gaps costs value.
> - DCF, comparable company analysis (comps), and asset-based valuation all work, but each requires explicit adjustments before you apply them to a private company.
> - The Discount for Lack of Marketability (DLOM) is not optional — empirical restricted stock studies show 25–35% historical discounts; Damodaran's pragmatic range is 10–25% for profitable companies.
> - The one number to remember: a profitable mid-market private company typically trades at a 15–25% discount to its observable public peer group, before any control or minority layer.

---

In late 2021, a private equity firm was negotiating to acquire a Midwest precision manufacturer — call it MidCo. The company had \$45 million in revenue and \$8 million in EBITDA. Its sector traded publicly at 9× EBITDA. By that logic, MidCo was worth \$72 million. Simple.

Except the deal closed at \$52 million. The gap — nearly \$20 million — was not a negotiating failure. It was the market correctly pricing three invisible factors: the company was illiquid, it had key-man risk concentrated in one founder-CEO, and buyers had no certainty the financials were as clean as the seller claimed. Each of those factors gets a name, a methodology, and a number. This post is about all three.

Private company valuation is where the craft of valuation actually lives. Public company valuation has a shortcut: the market tells you the price. Private company valuation has no such crutch. You have to derive the value from first principles, add the appropriate discounts, and then defend a number in front of a buyer, a bank, a tax authority, or an arbitration panel. That is genuinely hard, and it is worth understanding in detail.

![Public vs private company valuation bridge](/imgs/blogs/valuing-private-companies-illiquidity-discount-methods-1.png)

---

## Foundations: Why Private Company Valuation Is Different

Before any formulas, understand the structural differences between a public and a private company from a valuation standpoint. They share the same accounting language, but the inputs to that language are profoundly different.

**No observable market price.** The most fundamental difference. For a public company, the market is continuously aggregating the opinions of thousands of investors with real money. The share price is not perfect, but it is an enormous piece of information. For a private company, the only price discovery happens at a transaction event — a financing round, a partial sale, or an exit. Between those events, there is no price. You are not marking to market; you are marking to model.

**Information asymmetry and reliability.** Public companies are audited annually, subject to SEC disclosure, and have legal exposure for material misstatements. Private companies — especially smaller ones — often have reviewed or compiled financials, not audited ones. Their accounting practices may be idiosyncratic. Owners frequently run personal expenses through the business (the family vacation billed to marketing, the car lease through the company). Before you can value a private company, you have to recast the income statement to normalize earnings. This process is called "adjusted EBITDA" or "seller's discretionary earnings (SDE)" in the SMB market. It is both necessary and adversarial — the seller has an incentive to maximize it, the buyer has an incentive to challenge every add-back.

**Concentrated ownership and thin management.** A typical mid-market private company has one or two people who hold nearly all the equity and run most of the operations. The loss of either creates genuine business risk — not just an HR inconvenience. This is key-man risk, and it is priced. If the founder-CEO leaves after you buy the company, revenue might drop 30%. That probability needs to live somewhere in the discount rate or in the cash flow projections.

**Customer and supplier concentration.** Many private companies have one or two customers that account for 40–60% of revenue. Lose one, and the business is structurally impaired. The risk is real and measurable — if the top customer is 50% of revenue and has a 3-year contract with no renewal guarantee, that is a specific risk that public-company beta does not capture.

**Illiquidity of the equity itself.** If you own 1,000 shares of a publicly traded company, you can sell them tomorrow morning at the bid price with negligible friction. If you own 20% of a private company, selling your stake requires finding a buyer, negotiating, doing legal due diligence, and potentially waiting years. That friction has real economic value — and it is quantified by the Discount for Lack of Marketability (DLOM).

**Capital structure complexity.** Many private companies have complicated capital structures — multiple classes of common stock, preferred with liquidation preferences, warrants, convertible notes, revenue-based financing. These all affect what a share of equity is actually worth. The waterfall matters. In a VC-backed company, common shares might be worth essentially nothing unless the company exits above the liquidation preference stack.

**Normalization of "owner economics."** Private company owners often pay themselves below-market or above-market salaries. A founder running the business might pay themselves \$150,000 when a professional CEO would cost \$300,000. Or the opposite — an owner extracting \$500,000 in salary when the business only warrants a \$200,000 replacement cost. You need to normalize compensation to market rates before any valuation multiple is meaningful.

Understanding these six differences is the prerequisite to everything that follows.

---

## The Three Methods Adapted for Private Companies

Public-company valuations typically use three families of methods: discounted cash flow (DCF), comparable company analysis (comps), and precedent transaction analysis. For private companies, the same families apply, but each requires explicit adjustments. Asset-based valuation, rarely used for public companies, becomes a fourth relevant method for certain private company profiles.

![Private company valuation methods compared](/imgs/blogs/valuing-private-companies-illiquidity-discount-methods-4.png)

### DCF with Higher Discount Rates

The DCF framework is identical to what you would apply to a public company — project free cash flows, discount at the weighted average cost of capital (WACC), sum to a present value, add terminal value. The difference is entirely in the discount rate.

For a public company in the Industrials sector, Damodaran's data (January 2025) puts the sector WACC at approximately 8.7%. That figure incorporates the observable beta from public-market trading, the size of the company relative to its peers, and a market-implied equity risk premium of about 4.6%.

For a private company in the same sector, you start from that same 8.7% baseline and add:

1. **Size premium.** Duff & Phelps (now Kroll) publishes annual size premia derived from the Fama-French data. The smallest decile of public companies — firms with market caps below roughly \$500 million — show historical returns 3–5 percentage points above what CAPM would predict. For private companies, which are typically even smaller, the convention is to add a size premium of 2–6% depending on revenue scale. A company with \$10M in revenue might add 5–6%; one with \$100M might add 2–3%.

2. **Specific company risk premium.** This is the subjective layer, and it is where judgment lives. Key-man risk, customer concentration, thin management, lack of audited financials, geographic concentration — each can add 1–3%. The total specific risk premium typically runs 1–5%, and in extreme cases (single-customer, founder-critical) can reach 7–8%.

3. **Industry-specific adjustments.** If the private company is in a cyclical industry, operates with high fixed costs, or has a non-diversified revenue stream, you may overlay an additional 1–2%.

The result is a total required return — effectively a private company WACC — that typically runs 13–22% depending on company profile.

![Private company discount rate build-up stack](/imgs/blogs/valuing-private-companies-illiquidity-discount-methods-2.png)

#### Worked example:

**Private DCF for Midwest Precision Manufacturing (MidCo)**

Inputs:
- Sector (Industrials) WACC from Damodaran: 8.7%
- Size premium (revenue ~\$45M, mid-small range): +3.5%
- Key-man risk (founder-CEO, 60% relationship-dependent revenue): +2.0%
- Customer concentration (top 3 customers = 58% revenue): +1.5%
- Lack of audited financials (reviewed only): +0.5%
- **Total private WACC: 8.7% + 3.5% + 2.0% + 1.5% + 0.5% = 16.2%**

Projected free cash flows (normalized, after eliminating owner perks):
- Year 1: \$5.2M
- Year 2: \$5.7M
- Year 3: \$6.2M
- Year 4: \$6.8M
- Year 5: \$7.3M
- Terminal value: \$7.3M × 1.03 / (0.162 − 0.03) = \$7.519M / 0.132 = \$56.96M

Present values:
- Year 1: \$5.2M / 1.162 = \$4.48M
- Year 2: \$5.7M / 1.162² = \$4.22M
- Year 3: \$6.2M / 1.162³ = \$3.95M
- Year 4: \$6.8M / 1.162⁴ = \$3.71M
- Year 5: \$7.3M / 1.162⁵ = \$3.43M
- Terminal value PV: \$56.96M / 1.162⁵ = \$26.77M

**Total enterprise value: \$46.56M**

At a 16.2% WACC, the same cash flows that would be worth roughly \$72M at an 8.7% sector WACC (9× EBITDA × \$8M) compress to \$46.56M. That is not pessimism — it is the accurate pricing of private-company risk.

*Intuition: every 1 percentage point you add to the discount rate reduces the terminal value by roughly 7–8% in this range. Three points of size and specific risk premium shave about \$20M off the value.*

### Comparable Company Analysis — the Private Discount

Public comparable company analysis (comps) establishes a valuation range using observable trading multiples. For a private company, you run the same analysis — pull the EV/EBITDA, EV/Revenue, and P/E multiples of the closest public peers — and then apply a haircut to reflect the private company's illiquidity and information asymmetry.

The standard haircut is 10–30% on the enterprise value multiple. The wide range is because it depends on:
- **Company size:** Smaller companies get larger discounts (more illiquid, less institutional coverage).
- **Profitability and stability:** A profitable, stable private company with audited financials might get a 10–15% haircut. A volatile, founder-dependent one might get 25–30%.
- **Transaction context:** A strategic buyer who has full information access (post due diligence) may apply a smaller discount than a minority investor who has limited visibility.

This haircut is separate from the DLOM — we address that more formally in the next section. Some practitioners fold them together; many treat them as distinct layers applied sequentially.

![DLOM determination process pipeline](/imgs/blogs/valuing-private-companies-illiquidity-discount-methods-3.png)

#### Worked example:

**EV/EBITDA comps with private discount for MidCo**

Public industrial manufacturing peers (mid-size precision machining):
- Peer A: EV/EBITDA = 9.2×
- Peer B: EV/EBITDA = 8.6×
- Peer C: EV/EBITDA = 10.1×
- Peer D: EV/EBITDA = 8.9×
- Median public peer multiple: **9.0×**

MidCo EBITDA (normalized): \$8.0M
MidCo EV at public multiple: 9.0 × \$8.0M = \$72.0M

Private company discount:
- MidCo is smaller than all peers (revenue \$45M vs peer median \$180M): −5%
- Reviewed financials only (peers are audited): −3%
- Customer concentration: −4%
- Key-man risk: −3%
- Total private haircut: **−15%**

Adjusted EV: \$72.0M × (1 − 0.15) = **\$61.2M**

Less net debt (assume \$5M debt, \$1M cash): −\$4.0M
**Equity value from comps method: \$57.2M**

*Intuition: the private discount directly reduces the multiple from 9.0× to effectively 7.65× EBITDA — a significant but defensible adjustment given the risk profile.*

### Asset-Based Valuation

Asset-based valuation is rarely the right answer for an operating company with significant goodwill, but it is the right answer — and sometimes the only answer — in specific situations:

- **Asset-heavy businesses:** Real estate holding companies, equipment-intensive firms, timberland, farmland, or maritime fleets where the value is primarily in the tangible assets.
- **Distressed situations:** When a company is losing money and the going-concern assumption is questionable, the question is not "what is this worth as a business" but "what do the assets liquidate for."
- **Holding companies:** Investment holding companies whose main assets are financial positions — equity stakes, bonds, private investments.
- **Establishing a floor:** Even for operating companies, the adjusted net asset value (ANAV) establishes a lower bound — rational buyers should not pay less than what they could get by liquidating the assets.

The mechanics: start with the book value of assets, mark each to fair market value (not historical cost), subtract all liabilities including off-balance-sheet items, and arrive at adjusted net asset value. For liquidation scenarios, apply haircuts for forced-sale conditions (typically 20–40% below fair market value for physical assets).

#### Worked example:

**Net asset value for a private real estate holding company**

Assets (book value → fair market value):
- Office building: book \$12.0M → appraised FMV \$18.5M
- Retail strip: book \$8.0M → appraised FMV \$7.2M (impaired)
- Land (undeveloped): book \$3.5M → appraised FMV \$5.8M
- Marketable securities: book \$2.1M → FMV \$2.3M
- Accounts receivable (net): book \$0.9M → FMV \$0.9M
- **Total assets at FMV: \$34.7M**

Liabilities:
- Mortgage on office building: \$9.5M
- Mortgage on retail strip: \$4.2M
- Deferred taxes on unrealized gains: \$1.8M
- Other liabilities: \$0.6M
- **Total liabilities: \$16.1M**

**Adjusted Net Asset Value: \$34.7M − \$16.1M = \$18.6M**

For a minority stake purchase (say, 30% of the company), apply DLOM and DLOC:
- 30% × \$18.6M = \$5.58M
- DLOM for real estate holding co (moderately liquid assets): −15%
- DLOC for minority with no veto rights: −20%
- Minority value: \$5.58M × (1 − 0.15) × (1 − 0.20) = **\$3.79M**

*Intuition: the discount layers are multiplicative, not additive — applying 15% DLOM and 20% DLOC is a 32% combined haircut, not 35%.*

---

## The DLOM: Quantifying Illiquidity

The Discount for Lack of Marketability deserves its own section because it is the most empirically grounded and the most contested element of private company valuation. Tax courts, SEC enforcement actions, and acquisition disputes have all turned on DLOM estimation. Understanding the methodology in depth is not optional for anyone doing serious private company work.

### What DLOM Measures

DLOM represents the discount you apply to reflect the fact that a private company interest cannot be readily converted to cash. It is not about control — a controlling shareholder in a private company still faces illiquidity. It is specifically about marketability: can you sell when you want to?

Three approaches dominate DLOM estimation:

**1. Restricted stock studies**

The cleanest empirical approach. The SEC allows companies to issue restricted shares — shares that are identical to freely traded common stock in all respects except that they cannot be sold for a specified period (historically two years, now six months to one year under Rule 144). Because everything about the share is identical except marketability, the price difference between restricted and freely traded shares is a direct measure of the illiquidity discount.

The landmark Silber (1991) study found an average restricted stock discount of 33.75%. The Maher (1976) study found 35%. More recent studies (post Rule 144 amendment, which reduced the lockup from two years to one year) show lower discounts in the 20–28% range. The key insight: these discounts apply to shares in large, well-known public companies — private companies, being harder to value and with no expected liquidity event, should trade at equal or larger discounts.

**2. Pre-IPO studies**

These look at transactions in private company stock that occurred in the 18–24 months before an IPO, and compare those prices to the IPO price. The logic: the IPO price is an arm's-length observable value; the pre-IPO private price reflects the same asset with the added discount for illiquidity. Findings typically show 40–60% discounts for pre-IPO transactions vs. IPO price, but this overstates DLOM because it also embeds an IPO premium (companies are often underpriced at IPO) and a control premium for block trades.

**3. Option-theoretic approaches (Longstaff; Black-Scholes put)**

Fischer Black and Robert Litterman, and separately Francis Longstaff, developed option-based models for DLOM. The intuition: if you own an asset but cannot sell it for a period T, you are effectively short a put option on that asset — you cannot protect yourself against price declines during the lockup. The cost of that put is the DLOM.

Longstaff (1995) modeled the DLOM as the cost of a lookback put option — the maximum possible loss during the restricted period, assuming a perfect market timer had the option. This gives an upper bound: for a one-year lockup with 30% volatility (typical for mid-size private companies), the Longstaff bound is approximately 18–22%. For a two-year lockup, it rises to 28–35%.

The Black-Scholes approach is simpler: model DLOM as the cost of an at-the-money put with strike = current asset value, maturity = expected illiquidity period, and volatility = company-specific vol (usually estimated from comparable public companies). With 35% vol and a 2-year holding period, a typical ATM put costs about 15–22% of asset value.

**Damodaran's pragmatic range**

Aswath Damodaran synthesizes this literature and proposes a practical framework:
- **Highly profitable, stable private companies with audited financials:** DLOM 10–15%
- **Moderately profitable, growing companies with reviewed financials:** DLOM 15–20%
- **Profitable but volatile or founder-dependent:** DLOM 20–25%
- **Unprofitable or pre-revenue companies:** DLOM 25–35%

The intuition is straightforward: the harder it would be to sell your stake and the wider the range of reasonable prices, the larger the illiquidity penalty.

![DLOM ranges by private company size](/imgs/blogs/valuing-private-companies-illiquidity-discount-methods-5.png)

### Size Matters Enormously

Micro-cap companies (under \$10M revenue) can have DLOMs of 20–35%. The bid-ask spread on information alone is enormous — you have no audited financials, no institutional interest, and a buyer universe of perhaps ten or twenty local strategic acquirers and a handful of small PE shops. Large private companies (\$250M+ revenue) with investment-grade credit and multiple potential buyers might warrant only 8–15% DLOM.

The chart above shows the empirical ranges. Notice that even at the "large" tier, the lower bound of 8% is economically meaningful — it represents roughly \$20M of discount on a \$250M equity value.

---

## Control Premium vs Minority Discount

DLOM addresses marketability. A completely separate adjustment addresses control — whether the buyer is acquiring a controlling or minority interest in the company.

**Control premium.** When you acquire 100% (or a majority) of a private company, you get to decide dividends, management compensation, capital allocation, and ultimately the sale of the company. That decision-making power has real value. Empirical studies (primarily from precedent transaction databases like S&P Capital IQ and PitchBook) consistently show that acquisition prices for controlling stakes are 20–40% above the observable minority market price of the same company. The premium compensates for synergies the buyer expects to extract.

**Discount for lack of control (DLOC).** The mirror image: if you are buying a minority stake in a private company — say, 20% — you have no ability to force a dividend, change management, or trigger a sale. You are a passenger. Academic and practitioner literature puts the minority discount at 15–30% relative to a controlling interest value. This is not the same as DLOM — a controlling interest in a private company also faces illiquidity. DLOC and DLOM compound.

#### Worked example:

**Minority stake valuation in a private family business**

A family business (a regional food distributor) has been valued at \$30M enterprise value on a controlling-interest, normalized basis using EBITDA comps.

The founding family wants to sell a 25% stake to bring in a minority investor to help fund expansion. The family retains control and management. What should the investor pay?

Step 1: Equity value at controlling-interest, normalized basis:
- EV: \$30.0M
- Less debt (net): −\$4.5M
- Controlling equity value: \$25.5M

Step 2: 25% pro-rata share:
- 25% × \$25.5M = \$6.375M

Step 3: Apply DLOC (minority, no protective provisions):
- DLOC: 22% (no board seats, no veto on transactions)
- Post-DLOC: \$6.375M × 0.78 = \$4.97M

Step 4: Apply DLOM (private company, reviewed financials, profitable):
- DLOM: 18%
- Post-DLOM: \$4.97M × 0.82 = **\$4.08M**

Step 5: Sense-check the effective multiple:
- Implied EV/EBITDA at \$4.08M for 25% stake = effective EV ≈ \$16.3M
- Original peer multiple: 8.0×; Adjusted effective multiple: \$16.3M / EBITDA ≈ 5.4×
- DLOC + DLOM together reduced the multiple from 8.0× to 5.4× — a combined 32% haircut

*Intuition: minority investors in private companies are dramatically penalized relative to the headline "company value." They cannot sell, cannot control, and cannot force liquidity. The math is not subjective — it is a rational pricing of those constraints.*

---

## VC Valuation Methods

Venture capital valuation is a different animal from PE valuation or M&A valuation. Most early-stage companies have no earnings, minimal revenue, and valuations that are fundamentally speculative. The frameworks that PE applies — EBITDA multiples, LBO models, DCF — are largely inapplicable. Instead, VC uses a combination of:

**Revenue multiples.** For SaaS or recurring-revenue businesses, EV/Revenue multiples from comparable public companies (adjusted for growth rate) drive the valuation. A private Series A SaaS company growing at 80% year-over-year might be valued at 8–12× NTM revenue, benchmarked against public peers but discounted 20–30% for illiquidity. For comparison, the public SaaS median multiple in late 2024 was approximately 6–8× NTM revenue — down from the 2021 peak of 20–25×.

**The VC method (back-solve from target IRR).** The classic approach. The VC knows: (a) the expected exit multiple — typically 3–7× MOIC (multiple on invested capital) for a 5-year hold; (b) the exit valuation, estimated from revenue or earnings projections at exit year; (c) the required ownership stake to hit that return. Back-solving:

Required ownership = Investment / Post-money valuation
Pre-money valuation = Post-money − Investment

If a VC needs to invest \$8M, expects the company to be worth \$60M at exit in 5 years, and has a fund that requires 3× MOIC: the required investment value at exit is \$24M (3× × \$8M). At a \$60M exit, they need 40% ownership. Post-money = \$8M / 0.40 = \$20M. Pre-money = \$20M − \$8M = \$12M.

**Pre-money vs post-money mechanics.** These terms confuse many people.

- **Pre-money valuation:** What the company is worth before the new money comes in.
- **Post-money valuation:** Pre-money + the new investment.
- **Investor ownership:** Investment / Post-money valuation.

If a startup has a pre-money of \$10M and raises \$5M, the post-money is \$15M. The investor owns \$5M / \$15M = 33.3%.

**Why VC multiples differ from PE multiples.** PE buys mature, profitable businesses and uses leverage. VC buys optionality — the chance that a company becomes worth 100× the investment. PE pricing is driven by cash flows; VC pricing is driven by expected exit value and dilution. PE DLOMs are modest because there is a clear exit path; VC DLOMs are higher (25–35%) because exits are uncertain and long-dated.

![VC pre/post money valuation and founder dilution across rounds](/imgs/blogs/valuing-private-companies-illiquidity-discount-methods-6.png)

The chart above shows a typical three-round structure. Notice how at Seed (\$1.5M raised at \$4.5M post-money), the founders retain 67% of the company — but after Series A (\$8M at \$20M post) and Series B (\$20M at \$55M post), they might retain only 35–40% of the equity. Each round's dilution is locked in at the moment of closing. Pro-rata rights and anti-dilution provisions can modify this, but the fundamental math is mechanical.

**The VC lifecycle of valuation events.** From founding through exit, a company's valuation is formally established at each financing event, M&A transaction, or secondary sale.

![Private company lifecycle valuation timeline](/imgs/blogs/valuing-private-companies-illiquidity-discount-methods-7.png)

---

## Private Credit: Asset Coverage Thinking

When a bank or direct lender is underwriting debt to a private company, the valuation framework shifts from equity-centric to asset-coverage-centric. The lender does not care primarily about the equity upside — they care about whether the assets can cover the loan if the company defaults.

**Loan-to-value (LTV) ratios.** Senior secured lenders typically lend up to 50–65% of the liquidation value of hard assets (real estate, equipment, inventory). For enterprise value lending (cash-flow lending), the standard is 3.5–5× EBITDA for senior debt, with total leverage up to 5.5–6.5× in PE-sponsored deals. The EBITDA base is adjusted — normalized, LTM, sometimes blended LTM/NTM.

**Collateral haircuts.** Every asset class has a standard haircut for lenders:
- Accounts receivable: 80–85% of face value (lenders discount for collection risk)
- Inventory (raw materials): 50–60%
- Inventory (finished goods): 60–70%
- Machinery and equipment: 60–80% of orderly liquidation value (OLV)
- Real estate: 70–80% of appraised value

**Coverage ratios.** The interest coverage ratio (EBITDA / interest expense) needs to be ≥ 2.0× for most lenders; senior leverage ratio (debt / EBITDA) ≤ 3.5× for senior, ≤ 5.0× total. These covenants are negotiated in the credit agreement and tested quarterly. A breach triggers an event of default.

For private equity buyouts, the lender performs its own valuation of the company — not just the borrower's or sponsor's projection — and stress-tests the coverage ratios under a downside scenario (typically 20–30% EBITDA decline from LTM).

---

## Full Worked Example: Valuing Midwest Precision Manufacturing Co.

Let us put the whole framework together with MidCo, the company from the opening scenario.

**Company profile:**
- Industry: Precision machining / industrial manufacturing
- Revenue: \$45.0M (LTM, normalized)
- Gross profit: \$14.5M (32.2% margin)
- Adjusted EBITDA: \$8.0M (17.8% margin) — after adding back \$1.2M owner salary excess, \$0.4M personal expenses, \$0.6M one-time items
- Capex: \$1.5M/year (maintenance); growth capex: \$0.5M
- Net working capital: \$6.0M
- Net debt: \$4.0M (\$5M debt, \$1M cash)
- Ownership: 100% private, founder-owned (age 58)
- Financials: Reviewed (not audited)
- Top 3 customers: 58% of revenue, contracts expire within 18 months
- Management depth: Founder-CEO + two operations managers; no CFO

**Step 1: Normalize the financials**

The income statement showed \$7.2M EBITDA. After normalization:
- Add back excess owner compensation: +\$1.2M (market CEO rate is \$150K; owner pays himself \$300K, we normalize to market)
- Add back personal expenses run through company: +\$0.4M (insurance, travel, car)
- Add back one-time legal expense: +\$0.3M
- Subtract normalized stock-based comp equivalent: −\$0.1M
- **Normalized EBITDA: \$9.0M** (Note: we use \$8.0M in what follows as the "conservative" base, which applies a quality-of-earnings haircut; a buyer would want audited figures to confirm the add-backs.)

**Method 1: DCF**

Using the discount rate built up in the earlier worked example: 16.2% WACC.

Projected FCF (normalized EBITDA less capex, less working capital changes, less normalized taxes at 25%):

| Year | EBITDA | Capex | D&A | EBIT | Taxes | NOPAT | + D&A | − ΔWC | FCF |
|------|--------|-------|-----|------|-------|-------|-------|-------|-----|
| 1 | \$8.0M | \$2.0M | \$1.2M | \$7.2M | \$1.8M | \$5.4M | \$1.2M | \$0.3M | \$6.3M |
| 2 | \$8.5M | \$2.0M | \$1.2M | \$7.7M | \$1.9M | \$5.8M | \$1.2M | \$0.3M | \$6.7M |
| 3 | \$9.0M | \$2.0M | \$1.3M | \$8.3M | \$2.1M | \$6.2M | \$1.3M | \$0.3M | \$7.2M |
| 4 | \$9.5M | \$2.1M | \$1.3M | \$8.7M | \$2.2M | \$6.5M | \$1.3M | \$0.2M | \$7.6M |
| 5 | \$10.0M | \$2.1M | \$1.3M | \$9.2M | \$2.3M | \$6.9M | \$1.3M | \$0.2M | \$8.0M |

Terminal value: \$8.0M × 1.025 / (0.162 − 0.025) = \$8.2M / 0.137 = \$59.9M
(Terminal growth rate 2.5% — matching long-run industrial GDP growth)

PV calculations at 16.2%:
- PV of Year 1–5 FCFs: \$6.3M/1.162 + \$6.7M/1.162² + ... ≈ \$27.1M
- PV of terminal value: \$59.9M / 1.162⁵ ≈ \$28.2M

**DCF Enterprise Value: \$55.3M**

Less net debt: −\$4.0M
**DCF Equity Value: \$51.3M**

**Method 2: EV/EBITDA Comps**

Public industrial manufacturing peers (precision, mid-size):
- Median EV/EBITDA: 9.0× (consistent with \$72M at public multiple)

Apply private company discount:
- Size: −5% (MidCo is 30% of peer revenue median)
- Financials quality: −3%
- Concentration risk: −4%
- Key-man: −3%
- Total: −15%

Adjusted multiple: 9.0× × 0.85 = 7.65×
EV at adjusted multiple: 7.65 × \$8.0M = **\$61.2M**
Less net debt: −\$4.0M
**Comps Equity Value: \$57.2M**

**Method 3: Asset-Based (Floor)**

Tangible assets at fair market value:
- Machinery and equipment (OLV): \$8.5M
- Real property (owned, appraised): \$5.5M
- Inventory: \$2.8M
- Receivables: \$4.1M
- Other: \$0.5M
- **Total tangible assets: \$21.4M**

Less liabilities:
- Term loan: \$5.0M
- Accounts payable: \$2.1M
- Accrued liabilities: \$0.8M
- **Total liabilities: \$7.9M**

**Net tangible asset value: \$13.5M**

This is the floor. No rational buyer would pay less than this because they could liquidate and recover \$13.5M.

**Step 2: Apply DLOM**

MidCo is profitable, stable, 17.8% EBITDA margin, reviewed financials. Damodaran range: 15–20%.
We use 18% as the central estimate.

| Method | Pre-DLOM Equity | DLOM 18% | Post-DLOM Equity |
|--------|----------------|----------|-----------------|
| DCF | \$51.3M | −\$9.2M | \$42.1M |
| Comps | \$57.2M | −\$10.3M | \$46.9M |
| Asset-Based (floor) | \$13.5M | N/A | \$13.5M |

**Step 3: Reconcile to a point estimate**

We weight DCF and Comps equally and note the asset-based floor:
- DCF: \$42.1M (weight 50%)
- Comps: \$46.9M (weight 50%)
- **Weighted average: \$44.5M**

The asset-based floor of \$13.5M confirms we are well above liquidation — the going-concern premium is \$31M, justified by the \$8M EBITDA run-rate.

**Final equity value point estimate: \$44.5M**, with a range of \$42–\$47M.

Recall the actual deal closed at \$52M. The difference? The strategic buyer (a PE firm) was willing to pay above our "fair market value" estimate because they had a clear plan to add a CFO, reduce customer concentration, and eventually roll the company up with two peers — synergies and execution confidence that a financial buyer cannot price into a standalone DCF. The DLOM they applied was closer to 12% rather than 18%, reflecting their specific plans to professionalize the financials quickly.

*Intuition: fair market value (what a hypothetical willing buyer pays without special synergies) and strategic value (what this specific buyer pays given their plan) are legitimately different numbers. Know which one you are computing.*

---

## Common Misconceptions

**Misconception 1: "The DLOM is just a negotiating discount."**

Wrong. DLOM is an empirically grounded adjustment with four decades of academic support. The Silber restricted stock studies, the pre-IPO studies, and option-theoretic frameworks all converge on the same range. Tax courts in the US routinely review and accept DLOM arguments in estate and gift tax cases — the IRS publishes guidelines, and there is extensive case law. It is not a thumb-on-the-scale adjustment; it is a required correction for the fundamental difference between liquid and illiquid assets.

**Misconception 2: "Private companies should trade at 2–3× EBITDA."**

This persists in the SMB M&A world (companies with \$1–5M EBITDA), but conflates owner-operated lifestyle businesses with investable companies. A dentist practice where the owner IS the business (no salaries paid to the owner's family, no management team, no real replication) might trade at 2–4× SDE — but that is Seller's Discretionary Earnings, not EBITDA, and the multiple reflects the owner-dependence risk being priced in. A well-managed \$8M EBITDA private industrial company with professional management should trade at 6–8× EBITDA, not 2–3×.

**Misconception 3: "Higher revenue = higher valuation multiple."**

Not necessarily. Revenue multiples only work when you know the margin profile. A company with \$50M revenue and 5% EBITDA margin (\$2.5M EBITDA) is worth dramatically less than a \$30M revenue company with 25% EBITDA margin (\$7.5M EBITDA). Revenue multiples are useful shortcuts for early-stage or negative-EBITDA companies where you have no earnings to anchor; for profitable operating companies, EV/EBITDA is the primary multiple for good reason — it normalizes for margin.

**Misconception 4: "The DLOC and DLOM are alternative adjustments — you apply one or the other."**

They are additive (and multiplicative in application). DLOM addresses the marketability of your interest. DLOC addresses your power over the company. A minority stake in a private company suffers from both. A controlling stake suffers from DLOM but earns a control premium. The adjustments are independent and compound.

**Misconception 5: "The VC pre-money valuation is what the company is 'worth.'"**

Pre-money is a negotiated price for a specific transaction at a specific moment, not a fundamental fair market value. It reflects what a VC is willing to pay for ownership given their portfolio construction, fund economics, and the competitive dynamics of that funding round. Two identical companies can have very different pre-money valuations depending on how many VCs are competing for the deal. Pre-money is real in the sense that contracts are priced on it — it is not real in the sense that a DCF would produce it.

---

## How It Shows Up in Real Markets

**Estate and gift tax valuations.** When a founder of a private company dies or gifts shares to heirs, the IRS requires a formal appraisal. The DLOM is central. In *Estate of Mandelbaum v. Commissioner* (1995), the Tax Court allowed DLOMs ranging from 30–35% for restricted interests in private companies. The IRS regularly challenges aggressive DLOMs in estate planning. This is not hypothetical — it is one of the largest practical applications of private company valuation methodology, affecting hundreds of billions of dollars in estate planning annually.

**PE buyout underwriting.** Every PE deal involves at least two valuations: the sponsor's own model (used to determine price and target returns) and the debt underwriter's independent analysis (used to determine how much debt the capital structure can support). The DCF and EBITDA multiples drive both, but the inputs diverge — sponsors model upside cases, lenders model downside cases. The DLOM concept appears implicitly in the lender's collateral analysis: if the company fails and assets must be liquidated, the realized value is dramatically less than going-concern value.

**409A valuations for employee equity.** Under IRS Section 409A, startup employees who receive stock options must have those options granted at fair market value (the exercise price). The "409A valuation" is a formal third-party appraisal of the common stock in a VC-backed private company. These valuations apply DLOM (typically 20–35% for early-stage companies), DLOC, and often a Black-Scholes allocation model across the capital structure. Companies that get this wrong face IRS penalties. The 409A industry (Carta, Capshare, and dozens of boutique appraisal firms) processes tens of thousands of these per year.

**M&A due diligence.** When a large public company acquires a private target, the financial team builds a full valuation model — DCF, comps, and sometimes an LBO analysis to understand what a PE buyer would pay. The private company discount is embedded implicitly in the negotiated price, which is usually set relative to what a "financial buyer" would pay. Strategic buyers typically pay 10–30% above financial buyer value for synergies. DLOM is effectively zero in a 100% acquisition because the acquirer is providing immediate liquidity — they are paying for the underlying business, not the illiquid minority interest.

---

## Further Reading and Cross-Links

For the discount rate mechanics underlying the build-up method used in this post, see:
[Discount Rates in Practice: WACC, Cost of Equity, and Unlevered Beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta)

For a full DCF walkthrough applied to both US and Vietnamese public companies:
[DCF in Practice: Valuing VCB, Hoa Phat, and Apple](/blog/trading/asset-valuation/dcf-practice-valuing-vcb-hoa-phat-apple)

For how to build and use comparable company analysis:
[Comparable Company Analysis and Precedent Transactions](/blog/trading/asset-valuation/comparable-company-analysis-precedent-transactions-comps)

For understanding enterprise value and what market cap implies about growth:
[Enterprise Value vs Market Cap: Implied Growth Rates](/blog/trading/asset-valuation/enterprise-value-vs-market-cap-implied-growth-rates)

For the equity risk premium data underlying the Damodaran ERP estimates used here:
[Discount Rates in Practice: WACC, Cost of Equity, and Unlevered Beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta)

For Black-Scholes option pricing methodology (used in the put-option DLOM approach):
[Black-Scholes Model and Options Pricing](/blog/trading/options-volatility/black-scholes-model-options-pricing)

---

Private company valuation is not an exotic subspecialty — it is what valuation looks like in the real economy. The overwhelming majority of businesses in the world are private. When you own a small business, participate in a family partnership, invest in a PE fund, or hold stock options at a startup, private company valuation is directly determining your wealth. Mastering the methodology — the rate build-up, the DLOM, the control adjustments — is one of the highest-return skills in applied finance.

The key discipline: do not let the adjustments become decorative. Every point of size premium, every percentage of DLOM, should be traceable to a specific, defensible rationale. If a counterparty challenges your 18% DLOM, you should be able to cite the Silber study, the Damodaran range, and the specific facts of the company that put it in that part of the range. That rigor is what separates a valuation that holds up in court, in due diligence, or in a tax audit from one that falls apart under the first hard question.
