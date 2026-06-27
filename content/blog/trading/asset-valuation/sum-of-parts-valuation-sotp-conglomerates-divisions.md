---
title: "Sum-of-the-Parts Valuation: SOTP for Conglomerates and Divisions"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "Learn how to value a multi-segment conglomerate by pricing each business division with the right multiple, then bridging to equity value per share."
tags: ["valuation", "sotp", "conglomerates", "enterprise-value", "ev-ebitda", "comps", "equity-research", "asset-valuation", "financial-modeling", "holding-company"]
category: "trading"
subcategory: "Finance"
series: "Asset Valuation: How to Price Stocks, Options & Companies"
seriesOrder: 17
author: "Hiep Tran"
featured: false
draft: false
readTime: 45
---

> [!important]
> **TL;DR** — Sum-of-the-Parts (SOTP) values each business segment of a conglomerate with its own appropriate multiple, then adds them up and subtracts net debt to get equity value — because a single blended multiple will almost always misprice a diversified company.
>
> - Identify segments from 10-K "segment reporting" disclosures; each segment needs its own revenue, EBITDA, and growth rate.
> - Select the right multiple for each segment type: tech gets EV/Revenue or high EV/EBITDA; industrials get mid EV/EBITDA; financials get Price-to-Book.
> - SOTP equity value = Sum of segment enterprise values − corporate overhead − net debt; divide by shares outstanding.
> - The "conglomerate discount" is the gap between your SOTP price and the actual market cap — historically 10–30% — and it is the arbitrage thesis for activist investors.

When General Electric reported earnings in 2017, analysts covering the stock faced a genuinely awkward problem. GE at that moment was simultaneously a jet-engine maker, a power-plant builder, a healthcare imaging company, a financial-services giant, and an oil-and-gas equipment supplier. Applying a single P/E multiple to the whole enterprise — the way you might value a simple retailer — was almost comically wrong. The jet-engine business deserved the premium multiples of an aerospace duopoly; the financial arm needed a Price-to-Book lens; the power segment, wracked by oversupply, deserved a steep discount. Every analyst worth their Bloomberg terminal was running a Sum-of-the-Parts model instead.

SOTP is the answer to a simple question: what if the company you are analyzing is not really one company, but five companies wearing a single stock ticker? The method forces you to value each underlying business on its own terms — with the right comparables, the right discount rate, and the right cash flow metric — and then assemble the pieces into a coherent whole. It is the professional default whenever a company owns businesses across industries, and it surfaces the single most important number for activists and deal-makers: the conglomerate discount.

In this post we will build the full SOTP toolkit from first principles. We will read segment disclosures like analysts do, choose multiples with discipline, work through a complete three-segment worked example with explicit dollar math, and then examine how the same logic plays out at Berkshire Hathaway and VinGroup in Vietnam. By the end you will be able to open any conglomerate's annual report and produce a credible SOTP estimate in an afternoon.

![SOTP pipeline from segment identification to equity value per share](/imgs/blogs/sum-of-parts-valuation-sotp-conglomerates-divisions-1.png)

---

## Foundations: Why One Multiple Isn't Enough

### The blending problem

Imagine you own two rental properties. One is a luxury apartment in a prime downtown district that commands a 3% cap rate (meaning buyers pay 33× annual net income for it). The other is a strip-mall space in a secondary city, priced at a 7% cap rate (14× income). If you tried to sell both together and applied the average cap rate — say 5%, or a 20× multiple — you would be dramatically underpricing the luxury apartment and overpricing the strip-mall space. The blended approach destroys information.

Conglomerates face exactly this problem. When a single parent company owns a fast-growing software business alongside a mature commodity chemicals plant, the software division's true value evaporates into the average. Investors punish this information loss with a "conglomerate discount" — they pay less for the whole than they would for the pieces listed separately. This discount is the core motivation for SOTP analysis and, historically, for the wave of break-up activism that reshaped Western corporate structures in the 1990s and 2000s.

### The SOTP equation in plain English

The SOTP equation is:

**Equity Value = (Sum of Segment Enterprise Values) − Corporate Overhead (capitalized) − Net Debt**

And then:

**Price per Share = Equity Value ÷ Shares Outstanding**

The "segment enterprise values" are computed exactly the way you would compute enterprise value for a standalone company: multiply the segment's EBITDA (or revenue, or book value — depending on the industry) by the appropriate market multiple drawn from comparable pure-play companies. The corporate overhead subtraction accounts for the holding-company costs — CEO pay, legal, corporate staff — that serve no segment but still consume real cash. Net debt (total debt minus cash) converts enterprise values to equity value, just as in any other EV-based analysis.

Each term in this equation corresponds to a distinct analytical step, which we will now take one at a time.

---

## Step 1 — Identifying Segments from 10-K Disclosures

### What the accounting rules tell you

Under US GAAP (ASC 280) and IFRS 8, a public company must disclose financial information for any "operating segment" that is material. An operating segment is a component of the business about which separate financial information is regularly reviewed by the chief operating decision maker (CODM) — in plain English, what the CEO actually looks at when running the company. Crucially, if a segment is ≥ 10% of total revenue, operating profit, or total assets, it must be disclosed separately. For SOTP purposes this is your starting point.

Open any major conglomerate's 10-K and navigate to the "Segment Information" note in the financial statements, typically Note 18–22 depending on the company. You will find a table that breaks down, at minimum:
- Segment revenues
- Segment operating profit or EBIT
- Sometimes segment depreciation and amortization (giving you EBITDA)
- Segment assets (sometimes; varies)

Some companies go further and break out segment capex, working capital, and headcount. Others are stingy and give only revenue and a broadly defined "operating income" that may include cost allocations you need to reverse. Your job as an analyst is to reconstruct segment-level EBITDA from whatever the company provides.

### Reconstructing segment EBITDA

Companies often allocate shared costs (IT, HR, finance functions) down into segments. When you are building an SOTP model, you have two choices:

1. **Take segment EBITDA as reported** — and make sure you capture the unallocated costs separately as "corporate overhead."
2. **Gross up the segment EBITDA** to a "standalone" basis — remove the allocations and then handle all overhead in one corporate line.

Either approach works; what you cannot do is double-count. If you remove cost allocations from segments (making them look more profitable), you must capture that cost in the corporate overhead line that you will subtract later.

A practical shorthand: look for the reconciliation table in the segment note. US GAAP requires companies to reconcile segment totals to consolidated totals, and the "reconciling items" line is where unallocated corporate costs, eliminations, and interest live. That is your corporate overhead starting point.

#### Worked example:

VinGroup JSC (VIC on HoSE) is Vietnam's largest conglomerate with operating segments that span real estate development, consumer retail (Vinmart, now divested), automotive (VinFast), and technology/services (Vintech). In VinGroup's 2023 annual report, the segment note shows (approximately):

- **Real Estate (Vinhomes):** Revenue \$3.2B, EBIT \$1.1B, D&A \$0.2B → EBITDA \$1.3B
- **Automotive (VinFast):** Revenue \$1.5B, EBIT −\$1.8B (pre-investment losses), D&A \$0.4B → EBITDA −\$1.4B (cash-burn segment, use revenue multiple instead)
- **Technology & Services (Vintech + other):** Revenue \$0.8B, EBIT \$0.12B, D&A \$0.05B → EBITDA \$0.17B
- **Corporate / Eliminations:** −\$0.3B EBIT (unallocated holding company costs)

The critical observation: VinFast has deeply negative EBITDA because it is burning cash to ramp capacity. Applying an EV/EBITDA multiple here is meaningless — you would get a negative enterprise value. Instead, analysts value VinFast on an EV/Revenue basis (how the market prices early-stage EV companies like Rivian or Lucid), or use a DCF that explicitly models the path to profitability. This is the first reason SOTP requires segment-by-segment multiple selection, not a uniform formula.

The intuition: SOTP forces you to choose the right tool for each job, just as a surgeon uses a different instrument for each incision. A blended multiple is a Swiss Army knife applied to open-heart surgery.

---

## Step 2 — Selecting the Right Multiple for Each Segment

### Why the multiple must match the business

Every valuation multiple is a compressed DCF — it embeds assumptions about growth, margins, and risk. A high-growth software company trading at 15× EBITDA is priced that way because the market expects EBITDA to triple in five years and the business has high recurring revenue with low capital intensity. A regulated utility trading at 8× EBITDA is priced for stable, modest growth with predictable capex. Applying the utility multiple to the software segment destroys \$7 of value for every \$1 of EBITDA; applying the software multiple to the utility inflates value equally badly.

The discipline of SOTP is: find pure-play comparable companies in each segment's industry, observe the median EV multiple those comps trade at, and apply that multiple to the segment's financial metric. This is identical to the [comparable company analysis](/blog/trading/asset-valuation/comparable-company-analysis-precedent-transactions-comps) (comps) methodology — SOTP is just comps applied separately to each division rather than to the whole company at once.

![Segment-to-multiple matching grid](/imgs/blogs/sum-of-parts-valuation-sotp-conglomerates-divisions-2.png)

### The taxonomy of multiples by segment type

**Technology / Software:** High-growth segments are typically valued on EV/Revenue when EBITDA is negative (early stage) or EV/EBITDA at 15–25× when profitable. SaaS businesses often add ARR-based multiples. The key driver is growth rate: a segment growing revenue at 30%+ per year can command 10–15× revenue; at 15% growth, 5–8× revenue is more appropriate. See the [EV multiples deep-dive](/blog/trading/asset-valuation/ev-multiples-evebitda-evsales-enterprise-value-valuation) for the mechanics of EV/Revenue.

**Industrials / Manufacturing:** EV/EBITDA at 6–10× is the workhorse. For capex-light industrial businesses, P/E is also common. The relevant comparables are industrial conglomerates that have already split up — Honeywell's peers, for example, or the pure-play HVAC companies that emerged after United Technologies' 2020 break-up.

**Consumer Staples / Retail:** EV/EBITDA at 8–14× for established brands with pricing power; lower for commodity retailers. EV/EBIT is sometimes preferred when depreciation policies vary significantly across comparables.

**Financial Services (Insurance, Asset Management):** The EV framework breaks down for banks and insurers because debt is a raw material, not a capital structure choice. Price-to-Book (P/BV) is standard for banks; for insurance, P/BV or Price-to-Embedded-Value (P/EV). Asset managers are often valued on a percentage of AUM (assets under management), typically 1–3% of AUM.

**Energy / Commodities:** EV/EBITDA is common, but EV/DACF (debt-adjusted cash flow) is preferred for oil majors because depreciation is driven by depletion accounting that can distort EBITDA. Proven reserves-based valuation (EV/BOE — barrels of oil equivalent) is also standard.

**Real Estate:** For operating real estate businesses, Price-to-FFO (Funds from Operations) or EV/EBITDA adjusted for real estate. For real estate developers (like Vinhomes), NAV (Net Asset Value of land bank) is the primary method, sometimes alongside EV/Revenue for the current-year delivery backlog.

**Utilities:** EV/EBITDA at 7–10×; EV/RAB (Regulated Asset Base) in markets where returns on the regulatory capital base are fixed. Utilities' low WACC (see the sector data below) means their stable cash flows are worth more per dollar than industrial cash flows.

### How sector WACC reinforces the multiple selection

The sector WACC data (Damodaran, Jan 2025) provides a quantitative foundation for why multiples differ. A higher WACC means each dollar of future cash flow is worth less today — implying lower multiples. Conversely, utilities and consumer staples with WACCs of 6–7% support higher multiples because even modest, stable cash flows have high present value.

![Sector WACC bar chart — discount rates vary dramatically by industry](/imgs/blogs/sum-of-parts-valuation-sotp-conglomerates-divisions-3.png)

When you pick a 12× EV/EBITDA for your tech segment and a 7× for your industrial segment, you are implicitly embedding a higher discount rate — and thus a lower multiple — in the industrial valuation. The SOTP framework makes that implicit assumption explicit and defensible.

#### Worked example:

A conglomerate has three segments. You are building the comparable universe for each:

- **Segment A — Enterprise Software:** Peers include Salesforce, ServiceNow, Workday. Median EV/NTM EBITDA = 22×. Segment A's NTM EBITDA = \$400M. **Segment A EV = 22 × \$400M = \$8,800M.**

- **Segment B — Specialty Chemicals:** Peers include Ashland, Cabot, Innospec. Median EV/EBITDA = 8.5×. Segment B's EBITDA = \$350M. **Segment B EV = 8.5 × \$350M = \$2,975M.**

- **Segment C — Consumer Packaged Goods:** Peers include Church & Dwight, Prestige Consumer Healthcare. Median EV/EBITDA = 13×. Segment C's EBITDA = \$200M. **Segment C EV = 13 × \$200M = \$2,600M.**

**Total sum of segment EVs = \$8,800M + \$2,975M + \$2,600M = \$14,375M.**

The insight: if you had applied a single "average" EV/EBITDA to the whole company's \$950M total EBITDA, using, say, 10×, you would have gotten \$9,500M — 34% less than the disaggregated answer. The software segment's high multiple was being crushed by the blended average.

---

## Step 3 — Allocating and Capitalizing Corporate Overhead

### The hidden cost of the holding company

One of the most commonly botched steps in student SOTP models is the treatment of corporate overhead. The holding company — the legal entity that owns all the operating subsidiaries — incurs real costs that no operating segment generates revenue to cover: the parent company CEO, investor relations staff, legal counsel, board fees, accounting and audit fees for the consolidated entity, and corporate-level IT systems. These costs are real and recurring, and they must be deducted from the sum of segment values.

The standard approach is to capitalize the annual overhead at an appropriate multiple — typically the same EV/EBITDA multiple you might apply to a generic "overhead cost stream" — or simply treat it as a perpetuity:

**Capitalized overhead value = Annual overhead cost ÷ Discount rate**

If corporate overhead runs \$50M per year and you use an 8% discount rate (reflecting that these costs are as certain as the business), the capitalized value is \$50M ÷ 0.08 = \$625M. Some analysts use a simpler approach: apply the median segment EV/EBITDA multiple to the overhead cost with a negative sign, treating it like a "negative EBITDA" business. Both methods are defensible; pick one and be consistent.

Alternatively, many analysts capitalize overhead as a P/E multiple on after-tax cost. At a 20% tax rate and 15× P/E, \$50M pre-tax overhead costs (\$40M after-tax) → capitalized value of 15 × \$40M = \$600M. Close to the first method.

### What gets included in "corporate overhead"?

Include:
- Parent company compensation that is not allocated to segments
- Corporate-level legal, accounting, and audit costs not in segment expenses
- Treasury and investor relations costs
- Insurance costs at the holding company level
- Any goodwill impairment charges that are corporate-level (not segment-specific)

Exclude:
- Segment-level management compensation (already in segment EBITDA)
- Interest expense (that is handled in the net debt step)
- Acquisition costs related to a specific deal (one-time; normalize out)

---

## Step 4 — Bridging to Equity Value: The Net Debt Deduction

### From enterprise value to equity

Once you have the sum of segment enterprise values minus capitalized overhead, you have arrived at the consolidated enterprise value for the conglomerate under your SOTP model. The next step is identical to any EV-based valuation: subtract net financial debt to get equity value. See [Enterprise Value vs Market Cap](/blog/trading/asset-valuation/enterprise-value-vs-market-cap-implied-growth-rates) for the full mechanics.

**SOTP Equity Value = (Sum of Segment EVs) − Capitalized Overhead − Net Debt**

Where **Net Debt = Total Debt + Capital Leases + Preferred Stock + Minority Interest − Cash − Liquid Investments**

Two subtleties unique to conglomerates:

**1. Minority interests.** If the parent owns only 60% of a subsidiary, the subsidiary's full enterprise value flows into the SOTP sum (because you valued the whole enterprise using 100% of the subsidiary's EBITDA). But minority shareholders have a claim on 40% of that enterprise value. You must subtract the minority interest from equity value. The minority interest carrying value on the balance sheet is often a reasonable proxy, but sophisticated models compute it as 40% × the segment's SOTP value.

**2. Holding company debt vs operating-company debt.** Large conglomerates often have debt at both the parent level and within individual subsidiaries. Subsidiary debt is technically already embedded in the enterprise value (since you are computing EV, not equity value, for each segment). Be careful not to double-count: the standard practice is to deduct only the **net consolidated debt** from the consolidated balance sheet, and let the segment-level EV reflect the total enterprise. The key is that you are computing equity value for the **consolidated entity's shareholders**, so consolidated net debt is the right deduction.

![SOTP equity value bridge from segment values to per-share equity](/imgs/blogs/sum-of-parts-valuation-sotp-conglomerates-divisions-4.png)

#### Worked example:

Continuing from Step 2, our three-segment conglomerate has:

- Sum of Segment EVs: \$14,375M
- Corporate overhead: \$60M/year, capitalized at 8% → \$60M ÷ 0.08 = \$750M
- Total debt (from balance sheet): \$3,200M
- Cash and equivalents: \$800M
- **Net Debt = \$3,200M − \$800M = \$2,400M**
- Minority interests (20% of Segment B, which is worth \$2,975M at the SOTP level) = 20% × \$2,975M = \$595M

**SOTP Equity Value = \$14,375M − \$750M − \$2,400M − \$595M = \$10,630M**

With 500 million diluted shares outstanding:

**SOTP Price per Share = \$10,630M ÷ 500M = \$21.26 per share**

If the stock is trading at \$17.50, the implied conglomerate discount is:
(\$21.26 − \$17.50) ÷ \$21.26 = **17.7% discount to SOTP**

This 17.7% discount is the investment thesis for an activist: either the discount narrows as the company restructures, or the activist pushes management to spin off segments, crystallizing value.

---

## Step 5 — Understanding and Exploiting the Conglomerate Discount

### Why diversified companies trade at a discount

The conglomerate discount is one of the most-studied phenomena in corporate finance. Academic research (Berger and Ofek 1995; Lamont and Polk 2002) consistently finds that diversified companies trade at a 10–25% discount to their SOTP value. Several forces drive this:

**Opacity and complexity.** When investors cannot clearly understand the economic drivers of each business, they apply a "complexity penalty." The more opaque the conglomerate's reporting, the larger the discount.

**Cross-subsidization.** Profitable segments subsidize underperforming ones, destroying value. Analysts call this the "internal capital market inefficiency": the holding company allocates capital to divisions based on political considerations rather than return on capital. Cash generated by the high-ROIC software segment might fund a low-ROIC industrial acquisition that management finds strategically interesting.

**Management distraction.** A CEO trying to simultaneously manage a software product roadmap, a chemical plant's environmental compliance, and a consumer brand's marketing strategy will be worse at all three than three dedicated CEOs. The market prices this management-bandwidth discount.

**Liquidity and index inclusion.** Pure-play companies fit neatly into sector-specific ETFs and indices. A conglomerate falls across multiple GICS sectors, reducing its natural buyer base. Some institutional investors have sector mandates that prevent them from owning cross-sector names.

**The activist arbitrage.** Precisely because of this discount, conglomerates are prime targets for activist investors. The value-creation thesis is simple: break the company up → each piece trades at its intrinsic SOTP value → the discount evaporates → shareholders gain 15–25% without any operational improvement.

![SOTP waterfall showing segment contributions and conglomerate discount](/imgs/blogs/sum-of-parts-valuation-sotp-conglomerates-divisions-5.png)

### When the conglomerate discount doesn't exist

Not all conglomerates trade at a discount. Berkshire Hathaway is the canonical exception. Berkshire has traded near or sometimes above its SOTP value for decades because Warren Buffett's capital allocation track record is so exceptional that investors pay for the "Buffett premium" — the intangible value of having a world-class allocator sitting above the operating businesses. The holding company is itself a value-creating asset, not a value-destroying overhead structure.

Other exceptions:
- **Virgin Islands and holding companies in emerging markets** sometimes trade at premiums because they provide investors access to otherwise illiquid or inaccessible assets.
- **Companies with genuine synergies** across divisions — where having a common owner demonstrably reduces costs or creates revenue that standalone companies could not achieve — can trade above SOTP.
- **Tax-efficient structures** where operating inside a holding company shelter generates tax savings that standalone companies would lose.

---

## Step 6 — Single-Multiple Error vs SOTP: A Before-After Comparison

The graphic below makes the analytical error concrete. When you apply one blended EV/EBITDA multiple to the whole company, you are implicitly assuming every business inside the conglomerate has the same risk profile, growth trajectory, and capital intensity. That assumption is almost never true.

![Single blended multiple versus SOTP disaggregated approach](/imgs/blogs/sum-of-parts-valuation-sotp-conglomerates-divisions-6.png)

The before (blended) approach takes \$2,000M of total EBITDA and applies 8× to get \$16,000M of enterprise value. The after (SOTP) approach correctly identifies that \$600M of that EBITDA sits in a tech segment worth 12× (\$7,200M), \$830M in an industrial segment worth 7× (\$5,810M), and \$570M in a consumer segment worth 6× (\$3,420M), summing to \$16,430M — only slightly different in this case, but the composition is radically different. More importantly, in real-world examples where the segments have dramatically different multiples, the divergence is enormous.

---

## A Complete Three-Segment SOTP Model

Let us now build a full SOTP from scratch for a fictitious but realistic conglomerate, "GlobalCorp," to demonstrate every step end-to-end.

### GlobalCorp profile

GlobalCorp is a US-listed industrial conglomerate with three operating segments disclosed in its 10-K:
- **Aerospace & Defense (A&D):** Defense electronics and commercial aviation components
- **Industrial Automation:** Robotics, sensors, and factory automation systems
- **Specialty Chemicals:** Performance chemicals for consumer and industrial applications

From the 10-K segment note (fiscal year ended Dec 31, 2024):

| Segment | Revenue | EBIT | D&A | EBITDA |
|---------|---------|------|-----|--------|
| A&D | \$4,200M | \$630M | \$210M | \$840M |
| Automation | \$2,100M | \$420M | \$120M | \$540M |
| Chemicals | \$1,800M | \$270M | \$90M | \$360M |
| **Corporate** | — | −\$95M | \$5M | −\$90M |
| **Consolidated** | \$8,100M | \$1,225M | \$425M | \$1,650M |

From the balance sheet: Total Debt = \$4,500M, Cash = \$1,100M, Net Debt = \$3,400M. Minority interests: GlobalCorp owns 75% of its Chemical subsidiary; the 25% minority stake's carrying value = \$380M. Shares outstanding: 600M diluted.

### Step 1: Build the comparable universe

**A&D comparables:** L3Harris, Curtiss-Wright, Heico, TransDigm. Median NTM EV/EBITDA = 14.5×.

**Industrial Automation comparables:** Rockwell Automation, Cognex, Keyence (Japan-listed), Roper Technologies. Median NTM EV/EBITDA = 18.0×.

**Chemicals comparables:** Ashland, Cabot, H.B. Fuller, Innospec. Median NTM EV/EBITDA = 9.5×.

#### Worked example:

Apply the median multiple from each comparable universe to the corresponding segment EBITDA. Use NTM (next twelve months) EBITDA when forward estimates are available; otherwise use LTM (last twelve months) EBITDA from the segment note and apply a growth adjustment.

For GlobalCorp, using LTM EBITDA as above (conservative; assume comps are also on LTM):

- **A&D segment EV:** 14.5× × \$840M = **\$12,180M**
- **Automation segment EV:** 18.0× × \$540M = **\$9,720M**
- **Chemicals segment EV:** 9.5× × \$360M = **\$3,420M**
- **Sum of Segment EVs = \$25,320M**

### Step 2: Capitalize corporate overhead

Corporate overhead (excluding D&A, since we are computing enterprise value):
Overhead EBITDA = −\$90M (negative = cost center)
Capitalized at 8% discount rate: \$90M ÷ 0.08 = **\$1,125M** (present value of overhead as a perpetuity)

Alternatively: apply 12.5× EV/EBITDA multiple to the −\$90M overhead → −\$1,125M (same answer by coincidence here).

### Step 3: Bridge to equity value

**SOTP Equity Value = \$25,320M − \$1,125M − \$3,400M − \$380M = \$20,415M**

**SOTP Price per Share = \$20,415M ÷ 600M shares = \$34.03 per share**

If GlobalCorp's stock trades at \$27.50, the implied conglomerate discount is:
(\$34.03 − \$27.50) ÷ \$34.03 = **19.2% discount to SOTP**

The intuition: GlobalCorp is worth \$34 if it sold each division to the highest-bidding buyer who would pay the industry multiple — but the market only prices it at \$27.50 because it is lumped together under one roof, adding overhead and obscuring the quality of the A&D and Automation businesses. An activist buying in at \$27.50 with a plan to spin off Automation separately would capture most of that \$6.50/share gap.

---

## SOTP in Practice: Berkshire Hathaway

Berkshire Hathaway is the most-analyzed conglomerate in the world, and running a SOTP on Berkshire teaches more about the method than any textbook example.

### Berkshire's segment structure (2024)

Berkshire discloses seven broad operating groups in its 10-K:
1. **BNSF Railway** — railroad; valued on EV/EBITDA vs Union Pacific and CSX (peers at 13–15×)
2. **Berkshire Hathaway Energy (BHE)** — regulated utilities and pipelines; valued at EV/EBITDA 10–12× or on RAB (Regulated Asset Base)
3. **Insurance underwriting** — GEICO plus reinsurance; valued at Price-to-Book or combined-ratio-adjusted earnings
4. **Insurance float** — the investment portfolio funded by float; marked to market (equity securities at market value, bonds at amortized cost)
5. **Manufacturing, Service & Retail** — diverse industrials; median comps at 10–12× EBITDA
6. **Pilot Flying J / truck stops** — partial ownership stake; EV/EBITDA 7–9×
7. **Equity investments** — Apple (5.7% stake, \$170B market value as of late 2024), Bank of America, Coca-Cola, etc. — valued at current market price × shares owned

The equity investments segment is the key analytical insight: Berkshire holds enormous publicly traded equity stakes that must be valued at market price, not at a private-company multiple. This is a pure "look-through" valuation: you add up the fair market value of each public holding directly. Apple alone accounts for roughly \$170B of Berkshire's SOTP value.

#### Worked example:

A simplified Berkshire SOTP estimate (approximate, using 2024 annual report data):

- **BNSF Railway:** EBITDA ≈ \$7.0B; 14× multiple → **\$98B**
- **BHE:** EBITDA ≈ \$5.2B; 11× → **\$57B**
- **Insurance operations:** Earnings ≈ \$5.5B after-tax; 15× P/E → **\$82B** (simplified)
- **Equity portfolio (public securities):** Marked to market → **\$310B** (per 13-F, Q3 2024 approx)
- **Other operating segments (Mfg/Service/Retail/Other):** EBITDA ≈ \$6.5B; 10× → **\$65B**
- **Corporate cash and T-bills:** \$325B (as of Q3 2024)
- **Less: HoldCo debt and liabilities:** −\$38B

**Rough SOTP equity value ≈ \$98B + \$57B + \$82B + \$310B + \$65B + \$325B − \$38B = \$899B**

Berkshire's actual market cap hovered around \$950B–\$1T through much of 2024 — remarkably close to, and sometimes at a premium to, the SOTP estimate. This premium reflects the "Buffett Capital Allocation" intangible that no SOTP formula can capture directly. It is also why Buffett himself has endorsed buybacks whenever Berkshire trades below 1.2× book value — a simple rule of thumb for when the market is clearly undervaluing the sum of parts.

---

## SOTP for Vietnamese Conglomerates: VinGroup

VinGroup (VIC) is Vietnam's equivalent of a mini-Berkshire: real estate, automobiles, technology, retail, hospitality, and education under one roof. Analysts at Vietnam-focused funds routinely run SOTP models precisely because the VN-Index P/E (see chart below), while useful for the index, is useless for VIC's complex mix of high-growth (VinFast EV) and mature cash-generating (Vinhomes real estate delivery) businesses.

![S&P 500 P/E history illustrating why market multiples shift over time](/imgs/blogs/sum-of-parts-valuation-sotp-conglomerates-divisions-7.png)

The VN-Index P/E ranged from 11.2× (2022 bear market) to 17.9× (2021 peak). Applying either extreme to VinGroup's blended earnings would swing the implied valuation by 60%. SOTP bypasses this by anchoring each segment to global sector peers.

### VinGroup SOTP sketch (illustrative, FY2023 approximations)

- **Vinhomes (VHM):** Vietnam's largest listed real estate developer. Analysts use NAV of undeveloped land bank plus backlog EV/Revenue. Approximate SOTP contribution: \$8–10B (VHM has its own market cap of ≈ \$9B at 2024 prices; VinGroup owns ~70% → contribution ≈ 70% × \$9B = \$6.3B at market value).

- **VinFast Auto:** Loss-making EV startup listed on NASDAQ (VFS). Market cap has been highly volatile (\$5B–\$85B range since IPO). Analysts take VinGroup's ownership stake (≈ 82%) × VFS market cap → contribution varies wildly. At \$6B VFS market cap, VinGroup's stake ≈ \$4.9B.

- **Vintech + Vinpearl + Other:** Hospitality (Vinpearl resorts) valued at EV/EBITDA 10–12× for Vietnamese hotel comps; tech/education businesses at revenue multiples. Combined ≈ \$2–3B.

- **Corporate overhead + net debt:** VinGroup carries substantial parent-level debt (≈ \$4B+). Net of overhead capitalization and net debt, the equity value is sensitive to VinFast's market cap assumption.

The analytical challenge: VinFast is so volatile and speculative that it dominates the SOTP model. Analysts typically present a range of SOTP estimates at different VFS market cap assumptions — the bear case (VFS = \$3B), base case (\$6B), and bull case (\$12B) — to bound the uncertainty. This scenario analysis within SOTP is standard practice for any segment with high valuation uncertainty.

---

## Common Misconceptions

**Misconception 1: SOTP is only relevant for giant conglomerates.**

False. Any company with two or more materially different business lines — a bank with both retail banking and investment banking, a media company with streaming and legacy broadcast, a pharma company with branded drugs and generics — benefits from SOTP analysis. The relevant threshold is whether applying a single multiple meaningfully distorts the result, not whether the company calls itself a "conglomerate."

**Misconception 2: The sum of parts is always higher than the market cap.**

Not necessarily. A holding company that provides genuine, quantifiable synergies — where operating together demonstrably generates more value than separate ownership — can trade above its SOTP. Holding companies with captive finance arms that reduce the cost of capital for operating units, or companies with centralized procurement savings, sometimes demonstrate this. The SOTP is a floor for break-up value; the premium above SOTP is the price of the synergy.

**Misconception 3: You should always use EBITDA as the segment metric.**

Wrong for capital-intensive or highly leveraged segments. EBITDA ignores capex, and for businesses with heavy ongoing capital requirements (BNSF's \$3.5B annual capex, or a semiconductor fab), EBITDA overstates economic earnings. EBIT or EBITDA minus capex (a proxy for free cash flow) is more appropriate for these segments. The multiple must match the metric: if you use EV/EBIT, the comparable companies' EV/EBIT is your benchmark, not their EV/EBITDA.

**Misconception 4: You can look up "the" conglomerate discount and apply it.**

Analysts sometimes reverse the process: observe that a conglomerate trades at a 20% discount to SOTP and conclude "this is the conglomerate discount." But that is a tautology, not an insight. The discount must be explained — is it overhead? Cross-subsidization? Opacity? Management incentive misalignment? A genuine discount that persists for structural reasons is not going away; the one driven by investor misunderstanding or temporary complexity can close quickly. Understanding the mechanism is what separates a real investment thesis from a number.

**Misconception 5: SOTP gives a single precise answer.**

SOTP gives a range, not a point estimate. The key assumptions — which multiple to apply, whether to use LTM or NTM EBITDA, how to capitalize overhead, which minority interests to include — all carry uncertainty. Professional analysts present SOTP as a price range (e.g., \$28–\$38 per share) corresponding to the 25th-75th percentile of comparable company multiples, not a single \$33.40 bullseye. The precision of the arithmetic should not mislead you about the imprecision of the inputs.

---

## How It Shows Up in Real Markets

### The spin-off wave and SOTP crystallization

The corporate history of the 2000s–2020s is largely a story of SOTP analysis driving capital allocation decisions. When analysts showed GE management that the industrial segments were worth far more than the blended stock price implied, it accelerated the eventual break-up. Honeywell, United Technologies, DowDuPont, and Siemens all went through major structural changes at least partly because activist or fundamental investors demonstrated through SOTP models that the parts were worth more than the whole.

The pattern is consistent: a diversified company underperforms its sector peers for several years; activists build a position and commission a SOTP model showing a 20–30% discount; management announces a "strategic review" that culminates in a spin-off or split; each piece re-rates to the appropriate sector multiple; shareholders gain.

The irony is that the companies that get broken up often underperformed in part because of the very opacity that the SOTP analysis reveals. The conglomerate discount is self-fulfilling: because investors cannot value the parts, they apply a discount; management cannot observe the discount directly in segment performance metrics, so they do not fix it; activists eventually force the issue.

### Acquisition pricing uses reverse SOTP

When a strategic acquirer or private equity firm bids for a conglomerate, the bid price is often constructed as an SOTP: the buyer assigns a value to each segment based on what they could sell it for to a willing third party (or operate it under their own umbrella). The "break-up premium" in M&A — typically 20–40% above the pre-announcement market price — is essentially the acquirer offering to eliminate the conglomerate discount and take the associated execution risk.

This also explains why leveraged buyouts of conglomerates have a "sum of parts" exit thesis: the PE firm buys the diversified company at a discount to SOTP, sells each division to strategic buyers at the full sector multiple over 3–5 years, and profits on the spread.

### SOTP in equity research reports

Walk through any sell-side initiation on a diversified industrial, healthcare conglomerate, or media company and you will almost always find an SOTP table: segment-by-segment EV, a corporate overhead deduction, a net debt bridge, a per-share target price, and a "implied discount to SOTP" if the target price is below the SOTP estimate. Morgan Stanley's initiation of 3M in 2023, for example, valued each of 3M's four segments separately before deducting net debt and litigation reserves (the PFAS liability was treated as a negative-value "segment" in many analyst models).

For further depth on the mechanics of EV calculations that feed into SOTP, see [Enterprise Value vs Market Cap: Implied Growth Rates](/blog/trading/asset-valuation/enterprise-value-vs-market-cap-implied-growth-rates). For the comparable company methodology that underpins each segment valuation, see [Comparable Company Analysis and Precedent Transactions](/blog/trading/asset-valuation/comparable-company-analysis-precedent-transactions-comps). For EV/EBITDA, EV/Sales, and EV/EBIT mechanics in detail, see [EV Multiples: EV/EBITDA and EV/Sales Deep Dive](/blog/trading/asset-valuation/ev-multiples-evebitda-evsales-enterprise-value-valuation).

---

## Building SOTP Sensitivity Tables

No professional SOTP model presents a single point estimate. The standard deliverable is a two-dimensional sensitivity table: one axis tests different multiple assumptions (e.g., EV/EBITDA ranging from the 25th to 75th percentile of comps), and the other axis tests a key operating assumption (NTM EBITDA growth, or a key segment's revenue). The resulting grid of equity values per share tells you: "Under what range of assumptions does the stock look cheap vs the current price?"

A practical format for a three-segment company:

| A&D Multiple | Automation Multiple | SOTP/Share (Low Automation) | SOTP/Share (Mid) | SOTP/Share (High Automation) |
|---|---|---|---|---|
| 13× | 16× | \$29.20 | \$31.80 | \$34.40 |
| 14.5× | 18× | \$31.50 | \$34.03 | \$36.60 |
| 16× | 20× | \$33.80 | \$36.30 | \$38.80 |

The current stock price of \$27.50 falls below even the bear case here, which is what makes the investment thesis compelling (or raises the question of what the market is seeing that the model is not).

#### Worked example:

For GlobalCorp, test the impact of a 1× change in the Automation segment multiple (the highest-value segment):

- Base case: Automation at 18× → Automation EV = \$9,720M → SOTP/share = \$34.03
- Bear case: Automation at 16× → Automation EV = \$8,640M → SOTP/share = \$34.03 − (\$9,720M − \$8,640M)/600M = \$34.03 − \$1.80 = \$32.23
- Bull case: Automation at 20× → Automation EV = \$10,800M → SOTP/share = \$34.03 + (\$10,800M − \$9,720M)/600M = \$34.03 + \$1.80 = \$35.83

A 2-turn range in the Automation multiple produces a \$3.60 range in SOTP/share — about 10.5% of the base case value. This quantifies the "multiple risk" in the model and tells you how sensitive the thesis is to getting the Automation comps right. If you have high conviction that Automation comps trade at 18–20× and low conviction about the precise A&D multiple, you know where to spend your research time.

The intuition: sensitivity analysis in SOTP is not a hedge or a disclaimer — it is the analysis. The range of the sensitivity table is the actual answer.

---

## SOTP vs Other Valuation Methods

### When SOTP beats DCF

A DCF of a conglomerate requires a single WACC applied to the whole company's consolidated free cash flows. But as we have seen, the correct discount rate for the Automation segment (higher WACC, higher growth) is different from the correct rate for Chemicals (moderate WACC, stable cash flows). Forcing one WACC onto both segments produces a blended rate that misprices both. SOTP solves this by letting each segment's valuation use its natural discount rate — either explicitly (in a segment-level DCF) or implicitly (through the sector-appropriate multiple which embeds the right WACC for that industry).

A hybrid approach — running a separate DCF for each segment and then bridging to equity value as in SOTP — is actually the most rigorous methodology for segments where the growth trajectory is clear and the cash flow model is reliable. This is how investment banking fairness opinions for conglomerate break-ups are typically built: DCF per segment + SOTP bridge.

### When comps alone miss the point

Comparable company analysis applied to the whole conglomerate picks up, as its peer set, other diversified conglomerates — companies also suffering the conglomerate discount. The result: a multiple that is already discounted, leading to an implied value that bakes in the discount rather than exposing it. SOTP fixes this by using pure-play comparables for each segment, setting the benchmark at what each business would trade at in the market as a standalone — which is the actual value that a break-up or sale would crystallize.

---

## Building Your Own SOTP: Step-by-Step Checklist

For the next conglomerate you want to value, work through this sequence:

1. **Download the 10-K.** Go directly to Note 18–22 (segment information). Extract revenue, EBIT or operating income, and D&A for each reported segment. Compute EBITDA = EBIT + D&A for each.

2. **Check the reconciliation table.** Find where segment totals differ from consolidated totals — the gap is your corporate overhead and eliminations starting point. Confirm whether segment costs include shared-service allocations.

3. **Build a comparable set for each segment.** Minimum 4–6 pure-play peers per segment. Pull trailing and NTM EV/EBITDA from Bloomberg, FactSet, or even free sources like Macrotrends + Statista for approximate figures. Compute median and quartile range.

4. **Apply the median multiple to get segment EV.** Note your multiple choice and why (cite the comp set, the metric, the time period).

5. **Capitalize corporate overhead.** Divide annual overhead cost by an appropriate discount rate (use the same WACC as the largest segment, or use 8% as a default for recurring HoldCo costs).

6. **Pull net debt from the balance sheet.** Total debt (short + long term) + capital lease obligations + preferred stock + book value of minority interests − cash and cash equivalents.

7. **Compute SOTP equity value.** Sum of segment EVs − overhead capitalized value − net debt.

8. **Divide by diluted shares outstanding.** Pull from the income statement denominator or the 10-K cover page.

9. **Compare to current stock price.** Compute the implied discount or premium.

10. **Build a sensitivity table.** Test the 25th and 75th percentile multiples for the two highest-value segments. Show the resulting per-share range.

---

## Further Reading & Cross-Links

The SOTP method sits at the intersection of three core valuation disciplines. Build depth in all three to become truly fluent:

- **[EV Multiples: EV/EBITDA and EV/Sales](/blog/trading/asset-valuation/ev-multiples-evebitda-evsales-enterprise-value-valuation)** — The mechanics of the enterprise value multiples you apply to each segment. Understanding why EV/EBITDA and EV/Revenue differ, and when each is appropriate, is essential before building any SOTP.

- **[Enterprise Value vs Market Cap: Implied Growth Rates](/blog/trading/asset-valuation/enterprise-value-vs-market-cap-implied-growth-rates)** — The EV-to-equity bridge (subtracting net debt, adding cash) that you execute in every SOTP's final step. Also covers how the market implies growth rates from observable multiples — useful for sanity-checking your SOTP segment valuations.

- **[Comparable Company Analysis and Precedent Transactions](/blog/trading/asset-valuation/comparable-company-analysis-precedent-transactions-comps)** — The full comps methodology: how to screen the peer universe, compute multiples, handle outliers, and build the quartile range that feeds your SOTP multiple selection.

- **[SOTP Valuation in Equity Research Practice](/blog/trading/equity-research/sum-of-parts-valuation-sotp)** — Equity research perspective on how analysts format and present SOTP models in initiation reports, target price setting, and recommendation letters. Practical templates for the sell-side deliverable format.

- **[WACC: Weighted Average Cost of Capital](/blog/trading/equity-research/wacc-weighted-average-cost-capital)** — The discount rate that implicitly underpins every EV multiple. When you switch from SOTP-via-multiples to SOTP-via-segment-DCF, WACC is the analytical input you are solving for.

---

## The Analyst's Mental Model

Sum-of-the-Parts valuation is, at its core, an act of disaggregation. A conglomerate is a portfolio of businesses, and like any portfolio, the value of the whole is only the sum of the parts when there are no interactions — positive or negative — between the components. The conglomerate discount is the market's persistent observation that, in most cases, the interactions are negative: corporate overhead, capital misallocation, and opacity destroy more value than shared services or diversification benefits create.

The SOTP model is not just a spreadsheet exercise. It is a way of forcing yourself to ask: what is this specific business worth, to a buyer who understands it well, at the multiples that the market currently assigns to comparable businesses? That question strips away the noise of the consolidated P&L and gets to the economic truth of each division. When the answer across all divisions, netted down through overhead and debt, exceeds the stock price by a material amount, you have found either a mispriced security or a break-up candidate.

For a beginning analyst, the SOTP framework also teaches something more fundamental: that every company, no matter how simple it looks from the outside, is actually a collection of asset-earning bets on different markets, risks, and growth trajectories. Learning to see those bets clearly — and to price them correctly, each with its own right multiple — is the core skill that separates a good financial analyst from a great one.

---

## Advanced Topics in SOTP

### Segment-level DCF versus multiple-based segment valuation

Most SOTP models apply a comparable company multiple to each segment's EBITDA. This is fast, transparent, and market-anchored — the multiple encodes what sophisticated buyers are currently paying for comparable cash flows. But there are situations where a DCF at the segment level is more appropriate and more accurate:

**When to prefer segment-level DCF:**
- The segment is in a rapid transition (shifting from hardware to subscription revenue, for example) where trailing metrics mislead
- The segment has lumpy, project-based revenue that makes trailing multiples meaningless (think large defense contracts or one-time infrastructure projects)
- The segment is genuinely unique, with no close comparables — early-stage EV manufacturing inside a conglomerate, or a semiconductor IP licensing business
- You need to capture an explicit terminal value assumption that differs from what the comps market currently implies

**How to run a segment-level DCF:**
Treat each segment as a standalone company. Project the segment's unlevered free cash flow (EBIT × (1 − tax rate) + D&A − capex − change in working capital) for five to ten years. Compute a segment-specific WACC using the sector beta from Damodaran's data (or from the comparable pure-play company set, relevered to the appropriate capital structure). Discount the cash flows and add a terminal value. The result is the segment enterprise value — exactly what goes into the SOTP sum.

The discipline is the same as a full-company DCF; you just run it once per segment instead of once for the consolidated entity. See the WACC mechanics at [Weighted Average Cost of Capital](/blog/trading/equity-research/wacc-weighted-average-cost-capital).

**Blending the two approaches:** Many sell-side analysts use DCF for segments where they have high conviction in the forecast (e.g., a regulated utility with known rate cases) and multiples for segments with shorter forecast visibility. This hybrid approach is entirely defensible and often more intellectually honest than pretending you have equal conviction across all divisions.

### How to handle segments with negative EBITDA

Early-stage or loss-making segments — VinFast's automotive unit, or Amazon's AWS before it turned profitable — pose a valuation challenge because EV/EBITDA breaks down (a negative denominator produces a negative enterprise value, which is nonsensical for a business that has real option value).

Standard approaches:

**1. EV/Revenue multiple.** If the segment has meaningful revenue even with negative EBITDA, use a revenue multiple benchmarked to pre-profit comparable companies. For EV makers, this might be 2–5× NTM revenue depending on the growth rate and competitive positioning. The limitation: revenue multiples hide margin differences between comparables, so you need to verify that the comps have similar margin trajectories.

**2. EV/Gross Profit.** Better than EV/Revenue for segments with variable cost structures. If gross margins differ dramatically between your segment and the comps, EV/Revenue overstates or understates the relative value. EV/Gross Profit normalizes for first-level cost differences.

**3. Probability-weighted DCF or real options.** For highly uncertain, binary-outcome segments (a biotech subsidiary awaiting FDA approval, or an EV startup betting on mass market penetration by 2028), a real-options or scenario-weighted DCF is the most rigorous approach. Assign a 40% probability to the "success" case (where the segment reaches profitability and commands a high multiple in 5 years) and a 60% probability to a "failure/restructure" case. The expected value across scenarios becomes the segment's contribution to SOTP.

**4. Liquidation or book value as a floor.** For segments with substantial tangible assets (plant, equipment, inventory) but poor earnings, the net asset value — assets minus liabilities at book, or at estimated market value of assets — provides a floor. This is particularly relevant for capital-intensive businesses like mining, shipping, or real estate development where the physical assets are worth something even if current operations are unprofitable.

#### Worked example:

A pharmaceutical holding company has three segments:
- **Branded Drugs:** \$1.2B EBITDA, 14× EV/EBITDA → **\$16.8B EV**
- **Generics:** \$300M EBITDA, 7× EV/EBITDA → **\$2.1B EV**
- **Biotech Pipeline (pre-revenue):** \$0 EBITDA, but 3 Phase III candidates. Analysts model: 50% probability of approval for lead compound valued at \$4B if approved, 30% for the second at \$2.5B if approved, 20% for the third at \$1.5B if approved. Expected value: 0.50 × \$4.0B + 0.30 × \$2.5B + 0.20 × \$1.5B = \$2.0B + \$0.75B + \$0.30B = **\$3.05B expected EV**

**Sum of Segment EVs = \$16.8B + \$2.1B + \$3.05B = \$21.95B**

Less \$500M overhead capitalized at 8% = \$6.25B overhead value... wait, let us be precise: \$500M ÷ 0.08 = \$6,250M. Subtract net debt of \$4.0B and minority interests of \$0.3B:

**SOTP Equity Value = \$21.95B − \$0.625B − \$4.0B − \$0.3B = \$17.025B**

With 200M shares outstanding: **\$17.025B ÷ 200M = \$85.13 per share**

The intuition: the biotech pipeline contributes \$3.05B in probabilistic value even though it has no current earnings. A single EBITDA multiple on the whole company would have assigned it zero value — which is precisely the kind of systematic undervaluation that makes pharma conglomerates targets for activist spin-off campaigns.

### Adjusting for controlling vs minority ownership stakes

A parent company rarely owns 100% of every subsidiary. When the ownership stake is less than 100%, you must adjust both the segment EV contribution and the minority interest deduction carefully.

**Two consistent approaches:**

**Approach A (proportionate EBITDA):** Multiply the subsidiary's EBITDA by the parent's ownership percentage before applying the multiple. Then there is no minority interest deduction needed, because you only valued the parent's proportionate share.

**Approach B (full EBITDA, minority deduction at the end):** Value the subsidiary at its full 100% enterprise value using 100% of EBITDA, then deduct the minority interest at the end (using book value or a proportionate share of the SOTP segment value). This is the standard approach in investment banking because it clearly shows the total enterprise value of each subsidiary.

Example: Parent owns 60% of a subsidiary with EBITDA = \$500M and applicable EV/EBITDA = 10×.

- **Approach A:** 60% × \$500M = \$300M EBITDA; 10× → \$3,000M contribution to SOTP sum. No minority deduction.
- **Approach B:** 100% × \$500M = \$500M EBITDA; 10× → \$5,000M full EV. Then subtract 40% × \$5,000M = \$2,000M minority interest from the final equity bridge. Net contribution = \$5,000M (goes in the sum) then \$2,000M deducted in the bridge → same \$3,000M net to the parent's equity.

Both approaches give identical results if applied consistently. The error occurs when analysts apply 60% of EBITDA to the multiple but also deduct a minority interest — double-counting the minority stake.

### The role of intercompany eliminations

Conglomerates with segments that sell to each other create intercompany revenues and costs that must be eliminated in consolidated reporting. The segment note typically shows segment revenues before eliminations, with a reconciling line for intercompany sales. When you value each segment, you must decide: should you value the segment at its reported revenue (including intercompany sales) or at its external revenue only?

For most SOTP purposes, use **external revenue** (post-elimination) where that data is available, because the intercompany revenue would disappear in a break-up scenario — each division would have to find external customers for what it currently sells internally. However, if the intercompany arrangement reflects a genuine competitive advantage (captive supply chain that is lower cost than market), you can argue that the standalone company would replicate that advantage through long-term contracts, and the full segment revenue is appropriate.

In practice, analysts often value the eliminations as a separate item — the "captive supply" value — rather than double-counting it in both buyer and seller segments.

### Cross-holdings and listed subsidiary stakes

When a conglomerate holds a significant stake in a publicly listed company, that stake is trivially valued: ownership percentage × current market cap of the listed entity. This is called "mark-to-market" valuation and eliminates any need for multiple selection on that specific holding.

The Berkshire Hathaway example is again instructive: Berkshire's public equity portfolio (Apple, Bank of America, Coca-Cola, etc.) is valued in the SOTP at market value as of the valuation date. This portion of the SOTP is not an estimate — it is a fact, updated daily. The analytical uncertainty lies entirely in the private operating businesses.

For conglomerates holding substantial listed stakes, the convention is to break the SOTP into two tiers:
1. **Publicly listed holdings** — valued at market (no multiple selection required)
2. **Private or unlisted subsidiaries** — valued at appropriate multiples

This split makes clear where the uncertainty in your SOTP lies and makes it easier to update the model as market prices change.

---

## The Conglomerate Discount: Deeper Evidence and Debate

### Academic findings on the conglomerate discount

The seminal paper by Philip Berger and Eli Ofek (1995) found that US conglomerates traded at an average 13–15% discount to an imputed value based on stand-alone trading multiples of their business segments. Later work by Lamont and Polk (2002) confirmed the discount and linked it to capital misallocation — divisions within a conglomerate invested less efficiently than comparable stand-alone firms.

However, some academics challenge the discount's universality. Graham, Lemmon, and Wolf (2002) argue that the apparent discount is partly a measurement artifact: conglomerates often acquire already-discounted companies, so the diversification is not the cause of the discount — the discount was there before the acquisition and the conglomerate structure simply perpetuates it.

For practical SOTP analysis, the debate matters less than the observation: your SOTP will typically come out above the market cap for diversified companies, and the gap is real enough that it drives real market outcomes (break-ups, spin-offs, acquisitions).

### Why the discount varies across markets and time

The conglomerate discount is not constant. It varies with:

**Market maturity:** In developing markets with thin capital markets, the conglomerate structure sometimes creates value by providing internal financing that bank lending or public capital markets cannot supply. Vietnamese conglomerates like VinGroup historically commanded lower discounts (or even premiums) because VinGroup's scale gave it access to capital that smaller Vietnamese companies could not obtain at any price.

**Interest rates:** In low-rate environments, investors accept more complexity for yield and diversification. When rates are high (as in 2022–2024), investors prefer clean, simple businesses with predictable cash flows and pricing power — diversified conglomerates look like portfolios of mediocre businesses when the risk-free rate is 5%.

**Activist climate:** When activist hedge funds are well-funded and aggressive (as in the 2012–2018 period), conglomerates trade at tighter discounts because the market prices in a higher probability of forced restructuring. When activists are dormant (2008–2010 post-financial crisis), discounts widen.

**Reporting quality:** Companies that disclose richer segment detail — granular EBITDA, capex, and working capital by segment — allow analysts to build more precise SOTP models, which reduces the uncertainty premium that investors attach. Poor segment disclosure widens the discount.

---

## SOTP in Practice: Checklist for Common Errors

Even experienced analysts make systematic errors in SOTP models. Here are the most common, with corrections:

**Error 1 — Using the wrong fiscal year end for segments.** Some subsidiaries have different fiscal year ends than the parent. If Segment B closes its books in September while the parent closes in December, the "LTM" EBITDA for Segment B lags by three months and uses a different market multiple period. Always align your EBITDA and multiple to the same time period, or explicitly bridge.

**Error 2 — Not normalizing for one-time items.** Segment EBITDA as reported may include restructuring charges, legal settlements, or gains on asset sales. For SOTP, use "adjusted EBITDA" that strips these items — because the comparable company multiples you pull from Bloomberg are also typically applied to adjusted consensus EBITDA, not GAAP EBITDA.

**Error 3 — Double-counting pension obligations.** Many industrials carry large defined-benefit pension deficits that are technically debt-like claims on the company's assets. If you include all balance-sheet debt in your net debt calculation but ignore the underfunded pension obligation (which sits off the face of the balance sheet in a footnote), you are underestimating the equity value bridge correctly, but then comparing to a stock price that the market has already discounted for the pension. Include unfunded pension obligations in your net debt deduction: Net Debt = Financial Debt + Unfunded Pension + Capital Leases − Cash.

**Error 4 — Ignoring earnouts and contingent liabilities.** In recent acquisitions, the acquirer may have promised to pay additional consideration if performance targets are met (earnouts). These are contingent liabilities that reduce equity value but do not appear in reported financial debt. Check the M&A disclosure notes for the value of outstanding earnouts.

**Error 5 — Using book value of minority interest without checking against SOTP proportionate value.** Book value of minority interest (per the balance sheet) can differ significantly from the market value of the minority stake when the underlying business is growing or declining rapidly. A subsidiary whose book value is \$500M but whose SOTP segment value is \$2,000M means the minority holder's claim is 25% × \$2,000M = \$500M — by coincidence equal here, but often very different. Use the SOTP-proportionate minority value for a consistent framework.

---

## Practical Tips for Sourcing Segment EBITDA

The most tedious but critical step in SOTP is reconstructing clean segment-level EBITDA from what companies actually disclose. Some companies make this easy (GE's historical segment disclosures were famously detailed); others provide the absolute minimum required under GAAP. Here is a practical sourcing hierarchy:

1. **10-K segment note (primary source):** Revenue, EBIT or "segment operating profit," and sometimes D&A explicitly. If D&A is not in the segment note, look for the "depreciation and amortization" disclosure by segment sometimes found in the management discussion and analysis (MD&A) section.

2. **Quarterly earnings supplements:** Many companies provide supplemental Excel or PDF files with their earnings releases that break out metrics the 10-K segment note omits. Honeywell, for example, provides a detailed segment performance supplement with gross margin, operating margin, and organic growth by business unit.

3. **Investor day presentations:** Management typically provides detailed segment financial targets and historical metrics during investor days. These often include EBITDA by segment explicitly, plus capex guidance and margin targets.

4. **Bloomberg Intelligence or sell-side models:** If you have access, existing sell-side SOTP models for the same company can save hours of data reconstruction. But cross-check the segment EBITDA against the 10-K; analysts sometimes use adjusted or estimated figures that deviate from reported.

5. **Back-calculation from margins:** If you know segment revenue and the company discloses segment operating margins, you can compute segment operating profit. Add D&A (from any available source, including the company's press release income statement) to get EBITDA. For D&A by segment, the PP&E rollforward footnote sometimes helps — capex by segment gives you a rough proxy for segment D&A if you know the useful life assumptions.

The discipline is: document every assumption, use cited data, and note where you estimated versus directly sourced. An SOTP model is only as credible as the weakest assumption in the chain.

---

## Conclusion: SOTP as the Disaggregation Imperative

Sum-of-the-Parts valuation is ultimately an act of intellectual honesty about what a company actually is. Conglomerates are legal and financial structures, but the businesses inside them operate according to the economic logic of their own industries — their own growth rates, competitive dynamics, and capital requirements. Forcing a single blended multiple onto that complexity does not simplify the analysis; it falsifies it.

The discipline of SOTP forces the analyst to do the hard work: understand each segment as its own business, find the right comparables, select the right metric, and then carefully bridge from enterprise values to equity value in a way that accounts for every claim on the business. That hard work is rewarded — both in accuracy (SOTP consistently outperforms blended-multiple valuation in predicting break-up prices and M&A outcomes) and in insight (the sensitivity table tells you not just what a company is worth but why, and what could change that estimate).

Whether you are a retail investor trying to assess whether VinGroup is cheap, a buy-side analyst building a full-scale model for a US industrial, or a private equity associate structuring a conglomerate bid, the SOTP process is the same: disaggregate, price each piece correctly, bridge to equity, compare to market. The conglomerate discount is the opportunity. SOTP is how you measure it.
