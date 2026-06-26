---
title: "Comparable Company Analysis: Comps and Precedent Transactions"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "How investment bankers use trading comparables and precedent transactions to anchor a valuation to the real market, build a comps table, and construct a football field."
tags: ["valuation", "asset-pricing", "comps", "comparable-company-analysis", "precedent-transactions", "investment-banking", "mergers-acquisitions", "ev-ebitda", "football-field", "control-premium"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Relative valuation anchors a company's price to what the market is already paying for similar businesses, providing a fast, market-tested cross-check against a DCF.
>
> - A comps table selects 6–12 truly comparable public peers, computes normalized multiples (EV/EBITDA, P/E, EV/Revenue), and applies those to the target's financials.
> - Precedent transactions use the same framework but on completed M&A deals — they include a control premium of 20–40% above the unaffected trading price.
> - A football field chart overlays all valuation methods (comps, precedent transactions, DCF, 52-week trading range) to show where the consensus range lies.
> - The number to memorize: if comps say \$50 and precedent transactions say \$65, the implied control premium is 30% — right in the middle of the historical norm.

---

It is June 2024. You are advising a mid-cap industrial software company on a potential sale. Your DCF model — built with meticulous free cash flow projections and a carefully calibrated WACC — spits out an intrinsic value of \$58 per share. The CEO nods politely. Then she asks the question that actually matters: "But what would someone pay for us right now?"

That question is not answered by a DCF. It is answered by comparables. Trading comps tell you what the market is paying for similar businesses today, priced into current stock prices and observable every morning. Precedent transactions tell you what acquirers have historically paid when they needed control — the right to set strategy, fire the board, and capture synergies that the target cannot capture on its own. The spread between these two numbers is one of the most reliable data points in finance, and it has a name: the control premium.

This post is a craftsperson's manual for building both analyses from scratch. We will construct a comps table, argue through peer selection methodology, decode which multiples belong in which industries, compute a control premium anatomy step by step, and assemble the result into a football field that a board will actually believe. Every number will be real-looking and traceable. By the end, you will know not just how to build the table but what it means when the table disagrees with your DCF.

![Valuation triangle showing DCF, trading comps, and precedent transactions](comparable-company-analysis-precedent-transactions-comps-1.png)

---

## Foundations: What Relative Valuation Actually Does

To understand why comparables work, you need to understand what they are assuming — and what they are not.

A DCF asks: *given this company's future cash flows, what is the present value?* It is forward-looking, assumption-heavy, and entirely internal. Change your terminal growth rate by 50 basis points and the value swings by 10–20%. That sensitivity is a feature (it forces you to articulate your assumptions) but also a vulnerability (a motivated analyst can justify almost any number by tweaking two inputs).

Relative valuation takes a different approach. It asks: *given what the market is paying for similar businesses, what should this one be worth?* The underlying logic is the Law of One Price — two assets with identical cash flow profiles should trade at identical prices. If they do not, arbitrage (or M&A) closes the gap.

This is the same logic used in real estate. Before your realtor tells you what your house is worth, she runs "comps" — recent sales of similar properties on your street. She is not building a DCF of your future rental income; she is anchoring to observed market transactions. Investment bankers are doing precisely the same thing, with more Excel.

### The two flavors of relative valuation

**Trading comparables (public comps)** look at currently listed, publicly traded companies in the same business. You are using stock market prices, which reflect the collective judgment of millions of investors, as your benchmark. The multiples derived here represent *minority interest, liquid market prices* — the price of buying a small piece of an ongoing business.

**Precedent transactions (deal comps)** look at completed M&A deals involving similar businesses. You are using actual takeover prices, which reflect what an acquirer was willing to pay to own and control the entire company. These prices include a control premium — the extra amount an acquirer pays for the right to redirect the company's strategy, realize synergies, and integrate operations.

Both are necessary. Neither alone is sufficient.

### Why you always need the DCF too

A DCF is the only tool that forces you to articulate *why* a business has the value it does. Comps tell you the market's price; they do not tell you whether that price is right. In 1999, comps for internet companies were stratospheric — and using those comps to value more internet companies was circular insanity. A DCF would have forced the analyst to ask: "What free cash flow growth rate is embedded in this 50x EV/Revenue multiple? Is that realistic?"

The correct approach — the one that survives board scrutiny — is triangulation. Run the DCF. Run the comps. Run the precedent transactions. Build the football field. If all three point to \$45–65, you have a defensible range. If the DCF says \$35 and the comps say \$70, you have a story to tell about why — perhaps the market is pricing in an acquisition bid, or your DCF is too conservative on margins. Divergence is information. See [DCF practice: valuing VCB, Hoa Phat, and Apple](/blog/trading/asset-valuation/dcf-practice-valuing-vcb-hoa-phat-apple) for a detailed walk-through of building the intrinsic value anchor.

---

## Building the Comps Table: Step by Step

A comps table is deceptively simple in structure. The rows are your selected peer companies. The columns are the normalized financial multiples. The skill — the part that separates a good analyst from a mediocre one — is in the row selection and the normalization.

### Step 1: Define the peer universe

The starting point is every publicly traded company in roughly the same business. You get there through a systematic screening process, not by Googling "companies like [target]."

**Industry classification codes.** GICS (Global Industry Classification Standard, used by MSCI and S&P) or SIC (Standard Industrial Classification, used by SEC) codes give you a first filter. If your target is a cloud-based enterprise resource planning software company, you start with GICS sub-industry 45103010 (Application Software) or SIC 7372. This immediately reduces 5,000+ US-listed companies to perhaps 200–300.

**Revenue and market cap screen.** A company with \$20 billion in revenue does not compare to one with \$200 million. Investors price them differently: the large-cap commands a premium for liquidity, scale, and lower operational risk; the small-cap trades at a discount for the inverse reasons. As a rule of thumb, stay within 0.3x to 3x of your target's revenue. If your target has \$800 million in revenue, a reasonable screen is \$250 million to \$2.5 billion.

**Margin and growth profile.** A company growing at 25% annually with 35% EBITDA margins does not compare to one growing at 3% with 12% margins, even if both sell software. Investors price growth and profitability differently — explicitly so in frameworks like EV/Revenue, which is essentially capitalizing the growth optionality. Screen for peers within ±15 percentage points of your target's EBITDA margin and ±10 percentage points of revenue growth rate.

**Geography.** US-listed companies command different multiples than comparable businesses listed in Europe or Asia, driven by differences in investor base, liquidity, accounting standards, and country risk premium. For a US target, the core comp set should be US-listed; you can add international ADRs as a supplementary reference.

**The final set: 6–12 companies.** More than 12 and you are diluting the peer quality to include businesses that are only tangentially comparable. Fewer than 6 and your median multiple is too sensitive to outliers. The ideal set is 8–10 genuinely comparable businesses where you can defend every inclusion.

![Peer selection funnel from universe to final comp set](comparable-company-analysis-precedent-transactions-comps-2.png)

### Step 2: Gather and normalize the financials

Once you have your peer set, you need the financial data. For each peer, you want:

- **LTM (Last Twelve Months) financials**: Revenue, EBITDA, EBIT, net income, EPS. LTM = four most recent reported quarters, which is more current than the last full fiscal year.
- **NTM (Next Twelve Months) estimates**: Wall Street consensus forecasts for revenue, EBITDA, EPS. NTM multiples are forward-looking and more useful for fast-growing companies.
- **Fully diluted shares outstanding**: Basic shares + stock options + restricted stock units + warrants (using the treasury stock method for options). This feeds into equity value.
- **Net debt**: Total debt minus cash and cash equivalents. This bridges equity value to enterprise value.

#### Worked example: Computing EV and EV/EBITDA for a peer

Suppose one of your peers, call it PEER3, has the following data as of the LTM period:

- Stock price: \$48.50
- Basic shares outstanding: 85 million
- Dilutive securities (options, RSUs): 4.2 million (treasury stock method net dilutive shares)
- Fully diluted shares: 89.2 million
- Cash and equivalents: \$320 million
- Total debt (long-term + current): \$850 million
- Net debt: \$850M − \$320M = \$530 million
- LTM Revenue: \$1.12 billion
- LTM EBITDA: \$268 million
- LTM EBIT: \$198 million
- LTM Net Income: \$142 million

**Equity value** = fully diluted shares × price = 89.2M × \$48.50 = \$4,326 million = \$4.33 billion

**Enterprise value** = equity value + net debt = \$4,326M + \$530M = \$4,856 million = \$4.86 billion

**EV/EBITDA** = \$4,856M ÷ \$268M = **18.1x**

**EV/Revenue** = \$4,856M ÷ \$1,120M = **4.3x**

**P/E** = \$48.50 ÷ (\$142M ÷ 89.2M shares) = \$48.50 ÷ \$1.59 = **30.5x**

You repeat this calculation for every peer in your set. The result is the raw comps table.

### Step 3: Normalize for non-recurring items

Raw reported financials are contaminated with one-time items that have nothing to do with the ongoing earning power of the business. A comps table built on dirty numbers produces meaningless multiples.

**What to add back (these reduce reported EBITDA but are not recurring):**
- Restructuring charges (layoff costs, facility closures)
- Transaction costs (M&A advisory fees, legal fees)
- Impairment charges (goodwill write-downs, asset impairments)
- Stock-based compensation (SBC) — *this is controversial; some analysts add it back, others do not. The cleaner approach for cross-company comparisons is to add it back to get to "cash EBITDA," then note the SBC intensity separately*
- Legal settlements, regulatory fines (if genuinely one-time)
- Gains/losses on asset sales

**What to exclude (these boost reported metrics but are not recurring):**
- Insurance proceeds
- Government grants (if non-recurring)
- Gains on debt extinguishment

The normalization is an art as much as a science. The principle is always the same: you want a number that represents what this business earns in a normal operating year, stripped of accounting noise.

#### Worked example: Adjusting EBITDA for non-recurring items

Continuing with PEER3: suppose the LTM income statement includes \$42 million in restructuring charges related to a one-time headcount reduction, and \$18 million in transaction costs from a bolt-on acquisition. Both are non-recurring.

**Reported LTM EBITDA:** \$268 million
**Add: restructuring charges:** \$42 million
**Add: transaction costs:** \$18 million
**Adjusted LTM EBITDA:** \$268M + \$42M + \$18M = **\$328 million**

**Adjusted EV/EBITDA** = \$4,856M ÷ \$328M = **14.8x** (vs. reported 18.1x)

This matters enormously. A 3.3x difference in the multiple applied to your target's \$200M EBITDA swings the implied enterprise value by \$660 million. Sloppy normalization is not a minor error — it is the difference between a deal that happens and one that does not.

---

## Selecting the Right Multiple for Each Industry

Not all multiples are created equal. The appropriate multiple depends on the business model and where the earnings "show up" most cleanly in the income statement.

### EV/EBITDA — the workhorse

EV/EBITDA is the most widely used multiple in M&A for good reason. It is capital-structure-neutral (EV accounts for debt; EBITDA is pre-interest), it strips out depreciation differences (which vary with accounting policies), and it is positive for most mature businesses, unlike net income for high-growth companies.

**When to use it:** Industrials, consumer goods, retail, media, healthcare services, business services, technology hardware. Essentially any business with significant tangible assets or depreciation.

**When it fails:** High-growth software companies often trade on EV/Revenue because their EBITDA is near zero or negative as they invest in growth. Asset-light businesses with very low capex (and therefore minimal D&A) make EBITDA close to EBIT, so the multiple distinction becomes less important.

Typical EV/EBITDA ranges by sector (2024 data, Damodaran):
- Technology software: 15–25x
- Healthcare (devices, services): 12–18x
- Consumer Staples: 10–14x
- Industrials: 8–12x
- Energy (E&P, midstream): 5–9x

![EV/EBITDA multiple ranges by sector](comparable-company-analysis-precedent-transactions-comps-5.png)

### EV/Revenue — for high-growth or early-stage businesses

When a company is unprofitable or early in its investment cycle, EBITDA is negative or trivially small. Revenue multiples serve as a proxy. The multiple implicitly embeds assumptions about the eventual margin the business will achieve.

**Interpretation:** A company trading at 8x EV/Revenue with a long-run EBITDA margin trajectory of 25% is effectively trading at 32x LTM EBITDA margin-adjusted — which implies either very high growth expectations or a premium for strategic value. Decompose it explicitly: EV/Revenue = (EV/EBITDA) × (EBITDA margin).

**For the post-2020 SaaS market:** EV/Revenue multiples for high-growth SaaS companies ranged from 8x to 40x at the peak (2021), compressing to 3–10x by 2023 as rates rose. This compression — driven by higher discount rates — illustrates exactly why relative valuation is a snapshot of a market moment, not a permanent truth.

### P/E — for financial institutions and mature businesses

Price-to-earnings is appropriate when a business's earnings are the cleanest representation of value generation. For banks, EV/EBITDA is meaningless (interest expense is revenue, not a cost to be added back). Banks trade on P/E, P/Book (price-to-book value), and P/TBV (price-to-tangible book value).

For mature industrial or consumer companies with stable earnings, P/E is an intuitive supplement to EV/EBITDA. When the S&P 500 traded at 27.6x trailing P/E at end of 2024 (per our dataset), that was the market-wide anchor for earnings-based valuation.

### EV/EBIT — when capex intensity matters

EV/EBIT is useful when companies in the same peer set have meaningfully different depreciation levels — for instance, one company owns its manufacturing facilities and the other leases them. EBIT is after depreciation (a real economic cost when assets must be replaced) but before interest. It splits the difference between EBITDA (too forgiving of capex-intensive businesses) and net income (contaminated by capital structure).

---

## Precedent Transactions: The Deal Price Layer

Trading comps tell you what the market thinks the company is worth as a going-concern minority stake. Precedent transactions tell you what buyers were willing to pay to *own* the entire business — to have control.

### Why deal prices are higher than trading prices

When an acquirer buys a company, they are buying something that an ordinary stock market investor cannot buy: the right to make decisions. The acquirer can replace the management team, cut duplicative costs, cross-sell to their existing customers, eliminate a competitor, and redirect the entire capital allocation strategy. These are called **synergies**, and they have real dollar value that accrues to the acquirer, not to the target's standalone shareholders.

The acquirer, competing against other potential buyers (the market for corporate control is not monopolistic), must share some of this synergy value with the target's shareholders to win the deal. The amount they share is the control premium.

Empirically, across thousands of US public company M&A deals from 2010 to 2023, the median control premium paid over the 30-day unaffected stock price is approximately 28–32%. The 20th percentile is around 15%, the 80th percentile is around 45%. A "normal" deal carries a 20–40% premium.

### How to measure the unaffected price

The word "unaffected" is doing critical work in that definition. You do NOT measure the premium against the pre-announcement price on the day before the deal is announced. Why? Because deals leak. Analyst chatter, SEC investigations of unusual options activity, and information flow mean the stock price often already reflects partial probability of a bid in the weeks leading up to an announcement.

The standard practice: use the **30-day or 60-day unaffected price** — the average or closing price 30 to 60 days before any rumor hit the market. Bankers sometimes use the 52-week trading range low as the anchor for distressed sellers.

#### Worked example: Computing the control premium anatomy

A target company's stock traded at \$45.00 per share on a "clean" unaffected basis 45 days before rumors surfaced. On the day rumors hit the press, the stock jumped to \$50.85 (a 13% move — the "leak premium"). Three weeks later, the acquirer announced a definitive agreement at \$60.75 per share.

**Control premium over unaffected:** (\$60.75 − \$45.00) ÷ \$45.00 = **34.9%** — solidly in the normal 20–40% range.

**Premium over "day before announcement" price (wrong metric):** (\$60.75 − \$50.85) ÷ \$50.85 = 19.5% — this understates the true economic premium because the leak has already been priced.

Why does this matter? Because when you build a precedent transactions table to benchmark deal pricing, you must use the unaffected price as the denominator — otherwise, you are comparing apples to oranges, and your "average control premium" will be systematically understated.

![Control premium anatomy from stock price to deal price](comparable-company-analysis-precedent-transactions-comps-4.png)

### Building the precedent transactions table

The structure mirrors the trading comps table, with two key differences:

1. **The transaction EV is based on the deal price paid, not the current trading price.** For public targets, this is the per-share offer price times fully diluted shares plus assumed net debt. For private targets, you rely on disclosed deal terms.
2. **You include "NTM" multiples based on the consensus estimates at the time of announcement**, not current consensus. Financial conditions change — a deal struck in 2019 at a 12x EV/EBITDA multiple was priced relative to 2019 interest rates, not 2024 ones.

For the financial model in practice, you screen deal databases (Bloomberg, FactSet, Refinitiv) for transactions in your sector over the past 5–10 years, select 8–15 comparable deals based on: target industry, target size (within 0.2x–5x of your target), deal structure (all-cash vs. stock), and whether synergies were explicitly disclosed.

You then compute the multiple paid on LTM and NTM EBITDA at the time of announcement, and derive the implied range. Applied to your target's EBITDA, this gives you the precedent transaction valuation range.

---

## The EV/EBITDA vs. Margin Relationship: Why Comps Must Be Earned

One of the most important analytical skills in comps work is understanding that a multiple is not a fixed property of a company's industry — it is a function of the company's profitability and growth profile. Two software companies in the same GICS sub-industry can trade at 12x and 26x EV/EBITDA for entirely rational reasons.

The relationship is predictable and testable: **higher EBITDA margins command higher multiples.** The intuition is that a high-margin business has more cash per dollar of revenue available for reinvestment or return to shareholders, making each dollar of revenue more valuable.

![EV/EBITDA vs EBITDA margin scatter plot for tech peer group](comparable-company-analysis-precedent-transactions-comps-6.png)

This scatter plot illustrates a well-established empirical pattern: in any sector, there is a roughly linear relationship between EBITDA margin and EV/EBITDA. The slope of the regression line — about 0.6x multiple per 1% improvement in margin, in a typical tech peer set — tells you how much the market rewards incremental profitability.

The practical implication: when you apply a median EV/EBITDA from your comps set to your target, you should check whether your target's margin is at, above, or below the peer median. If your target is at the 75th percentile of margins but you are applying the median multiple, your valuation will be conservative. If your target is at the 25th percentile of margins, you should not expect to trade at the median multiple without a margin improvement story.

#### Worked example: Multiple adjustment for margin differential

Your target has an EBITDA margin of 28%. The peer median margin is 20%, and the peer median EV/EBITDA is 14.0x. The regression slope in your peer set is 0.55x per 1% of margin. Your target's margin is 8 percentage points above the peer median.

**Unadjusted implied EV:** \$200M EBITDA × 14.0x = \$2,800M

**Regression-adjusted implied multiple:** 14.0x + (8 × 0.55) = 14.0x + 4.4x = **18.4x**

**Regression-adjusted implied EV:** \$200M × 18.4x = \$3,680M

The adjustment adds \$880 million to the enterprise value — a 31% increase. This is not a minor footnote. In a real deal, this is the argument the sell-side banker makes to justify the higher valuation: "Our client trades in line with high-margin peers, not at the sector average."

---

## The Football Field: Presenting the Range

A football field chart is the standard way to present the output of a multi-method valuation exercise to a board, an investment committee, or a fairness opinion recipient. The name comes from the visual appearance of the chart: overlapping horizontal bars that look like a football field's yard lines.

Each bar represents one method's implied per-share value range. The key methods shown:

- **52-week trading range**: The stock's actual trading range over the past year — a market anchor, not a valuation method per se, but useful context.
- **DCF (bear/base/bull)**: The intrinsic value range under your three scenarios.
- **Trading comps EV/EBITDA**: The range from applying the 25th to 75th percentile peer multiple.
- **Trading comps P/E** (or EV/Revenue for growth companies): A second multiple-based method.
- **Precedent transactions**: The range from applying the 25th to 75th percentile deal multiple.

A vertical line marks the current stock price (for a fairness opinion) or the proposed deal price (for a board presentation on whether to accept an offer).

![Football field valuation ranges by method](comparable-company-analysis-precedent-transactions-comps-3.png)

### Reading the football field

**When methods converge:** If the DCF, comps, and precedent transaction ranges all cluster around \$44–62, you have a robust valuation. The proposed deal price of \$58 sits comfortably within the range, and the board has a strong foundation for accepting.

**When methods diverge:** If the DCF says \$35–45 and precedent transactions say \$60–74, you have to explain the gap. Common explanations:
- The DCF uses conservative inputs (verify against the market-implied growth rate: what WACC and terminal growth rate produce the comps-implied price?)
- The market is pricing in acquisition speculation (comps are inflated by deal rumors)
- Synergies are unusually high for this particular target (justifying the premium transaction price)
- The target is a "strategic jewel" — there are only a few companies in the world that could absorb it, and the scarcity premium is real

**Never present the football field without a narrative.** The visual shows a range; your job is to explain which parts of that range are most relevant and why the current or proposed price is fair, full, or insufficient.

---

## The Control Premium in Depth: Why 20–40%?

The 20–40% control premium is one of the most cited statistics in M&A — but most people who cite it do not understand what drives it. It is not arbitrary. It has a structure.

### The three components of the deal premium

**Component 1: The pure control value.** Even absent any synergies, controlling a business has option value. The acquirer can replace underperforming management, stop value-destructive capital allocation, optimize the balance sheet, and make decisions that minority shareholders cannot force. Academic research (e.g., Barclay and Holderness, 1989) estimates pure control value at 10–20% for US public companies.

**Component 2: Operational synergies.** These are the cost-saving and revenue-enhancing benefits of combining two businesses. Cost synergies (eliminating duplicative headquarters, shared procurement, technology rationalization) are typically more certain and faster to realize. Revenue synergies (cross-selling, new market entry) are less certain and take longer. A typical corporate M&A deal budgets 5–15% of target revenue in synergies, which translates to material enterprise value when capitalized at the combined entity's WACC.

**Component 3: The competitive auction premium.** When multiple bidders compete — in a formal or informal process — the winning bid must exceed all other offers. Competitive processes reliably drive prices 8–15% above the first offer price. This is not irrational; it is the result of bidder asymmetries in synergy realization and strategic value.

The mathematical floor of the premium is zero (a distressed seller with no alternatives). The practical ceiling is around 60–70% before the deal's economics become difficult for the acquirer to justify to its own shareholders (the acquirer is destroying value for the deal to create value for the target's shareholders).

![M&A control premium distribution histogram](comparable-company-analysis-precedent-transactions-comps-7.png)

The empirical distribution is right-skewed: most deals cluster at 20–40%, with a long tail of expensive "strategic" acquisitions. The median is around 28–30%. The 25th percentile (cheap deals, often distressed sellers or friendly negotiations) is around 15–18%. The 75th percentile (competitive auctions, premium strategic assets) is around 40–48%.

#### Worked example: Decomposing a \$1.2 billion deal premium

A technology services company is acquired at \$68 per share, against an unaffected price of \$50 per share — a 36% premium. Total deal value: \$3.4 billion (50 million fully diluted shares × \$68). Unaffected equity value: \$2.5 billion.

**Total premium paid:** \$900 million

**Estimated pure control value (15% of unaffected equity):** \$375 million
**Estimated synergies (articulated in acquirer press release):** \$420 million
- Annual cost synergies: \$60M, capitalized at 10x = \$600M gross; probability-adjusted at 70%: \$420M
**Competitive auction uplift (remainder):** \$900M − \$375M − \$420M = **\$105M**

This decomposition, while approximate, is exactly what an acquirer's fairness opinion would attempt to document to justify to their shareholders that they did not overpay.

---

## When Comps and DCF Disagree: What It Means

The most analytically valuable moment in a valuation exercise is when your methods disagree. Agreement is easy — it confirms your assumptions. Disagreement forces you to think.

### The market-implied DCF check

Here is the most powerful diagnostic: take the comps-implied enterprise value and work backward through your DCF to find the implied assumptions.

If comps say EV = \$2,800M and your DCF (with a 10% WACC and 3% terminal growth) says EV = \$1,900M, the question is: what WACC, terminal growth, or margin assumption produces \$2,800M? Solve for it. You might find that a 2% terminal growth rate and 9% WACC gives you \$2,800M — which means the market is pricing in 1% more terminal growth and 1% lower cost of capital than you used. Is that reasonable? Maybe — if rates have been falling and the company's growth is genuinely durable.

Or you might find that to reconcile the two, the company would need a 40% EBITDA margin in perpetuity versus your modeled 25%. That is a 50-year business case for a company currently at 20% margins. A serious analyst would question that.

### Common reasons for divergence

**DCF higher than comps:** Your model is too optimistic (check terminal growth rate, long-run margin assumptions), or the company is genuinely undervalued by the market (possible if it is small-cap, underfollowed, or has near-term headwinds that are masking long-run earnings power).

**DCF lower than comps:** Your model is too conservative, or the market is pricing in an M&A premium (the stock is trading up because smart money thinks it will be acquired), or the sector is in a bubble.

**Precedent transactions higher than comps:** This is the normal case — deal prices include control premium. If the gap is larger than 40%, investigate whether the precedent transactions you selected involved unusually large synergies or were struck during a different rate environment.

**Precedent transactions lower than comps:** Unusual, but can happen if the industry's public market multiple has expanded rapidly (2020–2021 tech) while deal multiples lagged. Acquirers were unwilling to pay comps-level prices.

---

## Common Misconceptions

### Misconception 1: "The median multiple is the right multiple for every company"

Wrong. The median multiple represents the median *quality* company. A company at the 70th percentile of EBITDA margin, revenue growth, and ROIC should trade at the 70th percentile of EV/EBITDA multiples — which might be 30–40% above the median. Mechanically applying the median ignores the most important thing comps teach you: the cross-sectional relationship between fundamentals and multiples.

### Misconception 2: "Precedent transactions are stale — use only trading comps"

This misunderstands what each tool is for. Precedent transactions answer a different question: what would an acquirer pay? If the purpose of your analysis is to advise a board on whether to accept a buyout offer, precedent transactions are more relevant than current trading comps. The "staleness" concern is addressed by excluding deals more than 5–7 years old and adjusting for the market cycle (the rate environment in 2015 versus 2022 changes multiples materially).

### Misconception 3: "EV/EBITDA is universal — use it for every industry"

Financial companies (banks, insurance, asset managers) are priced on P/E or P/Book because interest expense is a revenue item, not a cost. Real estate companies are priced on FFO multiples (Funds From Operations), not EBITDA. Commodity companies (oil producers, miners) are often priced on EV/2P reserves or EV/production. Always verify that the multiple you are using is standard for the industry in question. For more on [EV multiples construction](/blog/trading/asset-valuation/ev-multiples-evebitda-evsales-enterprise-value-valuation), see that dedicated post.

### Misconception 4: "A higher premium means the acquirer overpaid"

Not necessarily. The correct test is whether the acquirer's synergies exceed the premium paid. If a strategic buyer pays 35% over the unaffected price and the synergy value is 50% of the target's unaffected enterprise value, the acquirer has created value for their shareholders even at the premium price. Value destruction occurs when the premium exceeds achievable synergies — which happens, but not as often as deal critics suggest.

### Misconception 5: "You can use any 10 comparable companies you find"

The quality and defensibility of your comps set is entirely determined by how well you can justify each inclusion. For a formal fairness opinion (a legal document delivered to a board), every peer selection decision must survive cross-examination by opposing counsel. Even for informal internal analysis, weak peer selection produces a table that sophisticated readers will immediately challenge. Be restrictive, not inclusive.

---

## How It Shows Up in Real Markets

### Microsoft's acquisition of Activision Blizzard (2022)

In January 2022, Microsoft announced a deal to acquire Activision Blizzard at \$95 per share — a 45% premium over Activision's unaffected 30-day trading price of approximately \$65. Enterprise value: roughly \$68.7 billion.

How did the comps look? Activision's pre-announcement EV/EBITDA was around 18–20x on forward estimates. The deal was struck at roughly 26–28x forward EBITDA — a multiple consistent with the precedent transactions in gaming M&A, including EA's acquisitions and Take-Two's acquisition of Zynga (also announced in early 2022 at a 64% premium).

The football field for a banker advising Activision at the time would have shown:
- Trading comps EV/EBITDA: \$55–70 per share
- DCF: \$50–75 depending on assumptions about mobile game growth
- Precedent transactions (gaming + entertainment): \$72–90

At \$95, Microsoft paid at the top of the precedent transactions range and above the DCF/comps range. The justification: the gaming content library's strategic value to Xbox Game Pass was a synergy that no standalone DCF for Activision could capture.

### Facebook/Meta's acquisition of WhatsApp (2014)

Meta paid \$19 billion for WhatsApp, which had essentially zero revenue and \$10.2 million in trailing revenue — making EV/Revenue equal to approximately 1,863x. Trading comps? There were none. The acquisition was priced entirely on strategic value (eliminating a competitive threat, acquiring 450 million users), with no traditional multiple justification. This illustrates the limits of relative valuation: when there are no true comparables, the method fails, and you are back to a DCF-of-optionality framework.

### VN-Index M&A: Techcombank / HDBank sector

In Vietnam's banking sector, M&A control premiums have historically been lower (15–25%) than the US median, for structural reasons: concentrated ownership, regulatory constraints on foreign acquirers (foreign ownership cap of 30%), and thinner public float reduce competitive auction dynamics. This is why the sector-specific and geography-specific calibration of the control premium matters — the US empirical distribution does not translate directly to every market. For [P/E valuation mechanics](/blog/trading/asset-valuation/price-to-earnings-ratio-pe-valuation-stocks) in market-context, see that post in this series.

---

## Putting It Together: A Mini Comps Table

Here is a complete, worked mini comps table for a hypothetical industrial software target with \$200M LTM EBITDA and \$1.0B LTM revenue.

**The peer set (8 companies):**

| Ticker | Revenue (\$M) | EBITDA (\$M) | EV (\$M) | EV/Revenue | EV/EBITDA | P/E (NTM) |
|--------|--------------|-------------|---------|-----------|----------|----------|
| PEER1  | 780          | 148         | 2,052   | 2.6x      | 13.9x    | 21.4x    |
| PEER2  | 1,120        | 268         | 4,856   | 4.3x      | 18.1x    | 30.5x    |
| PEER3  | 890          | 196         | 3,136   | 3.5x      | 16.0x    | 24.8x    |
| PEER4  | 1,450        | 348         | 5,568   | 3.8x      | 16.0x    | 26.1x    |
| PEER5  | 620          | 118         | 1,612   | 2.6x      | 13.7x    | 20.2x    |
| PEER6  | 980          | 215         | 3,698   | 3.8x      | 17.2x    | 27.9x    |
| PEER7  | 1,320        | 290         | 4,872   | 3.7x      | 16.8x    | 27.3x    |
| PEER8  | 750          | 155         | 2,325   | 3.1x      | 15.0x    | 22.5x    |

**Summary statistics:**

| Metric     | Min   | 25th pct | Median | 75th pct | Max   |
|-----------|-------|---------|--------|---------|-------|
| EV/EBITDA | 13.7x | 14.6x   | 16.4x  | 17.4x   | 18.1x |
| EV/Revenue| 2.6x  | 3.0x    | 3.6x   | 3.8x    | 4.3x  |
| P/E (NTM) | 20.2x | 21.7x   | 25.5x  | 27.5x   | 30.5x |

#### Worked example: Applying the comps to the target

Target financials: LTM Revenue = \$1,000M, LTM Adjusted EBITDA = \$200M, NTM EPS = \$2.85, net debt = \$400M, fully diluted shares = 60M.

**EV/EBITDA range (25th to 75th pct):**
- Low: 14.6x × \$200M = \$2,920M EV − \$400M net debt = \$2,520M equity ÷ 60M shares = **\$42.00/share**
- High: 17.4x × \$200M = \$3,480M EV − \$400M net debt = \$3,080M equity ÷ 60M shares = **\$51.33/share**

**P/E range (25th to 75th pct):**
- Low: 21.7x × \$2.85 = **\$61.85/share**
- High: 27.5x × \$2.85 = **\$78.38/share**

Notice the disconnect: EV/EBITDA implies \$42–51, while P/E implies \$62–78. The gap (\$11–27) is explained by leverage: the \$400M net debt depresses the equity value in the EV/EBITDA calculation. The P/E is unaffected by debt level (NTM EPS already reflects after-interest income). For a heavily levered company, P/E is a cleaner equity multiple; for a debt-light company, EV/EBITDA and P/E should produce similar implied share prices. Always check your multiples for internal consistency.

---

## Further Reading and Cross-Links

The comps methodology sits at the heart of how [EV multiples are constructed and decomposed](/blog/trading/asset-valuation/ev-multiples-evebitda-evsales-enterprise-value-valuation) — that post goes deeper on enterprise value mechanics and the bridge between equity value and EV that underpins every multiple in this table.

For the intrinsic anchor that comps should be reconciled against, [DCF practice: valuing VCB, Hoa Phat, and Apple](/blog/trading/asset-valuation/dcf-practice-valuing-vcb-hoa-phat-apple) walks through full DCF builds in real-world settings.

[Price-to-earnings ratio: P/E valuation](/blog/trading/asset-valuation/price-to-earnings-ratio-pe-valuation-stocks) covers the P/E multiple in depth, including the decomposition of the P/E into a growth expectations formula and the Gordon-Growth anchoring that makes P/E interpretable.

For the equity research practitioner workflow in which comps fit (alongside credit analysis, management meetings, and channel checks), [comparable company analysis in equity research](/blog/trading/equity-research/comparable-company-analysis-comps) covers the workflow context.

**On the discount rate underpinning all multiples:** Every EV/EBITDA multiple embeds an implicit WACC. When the Fed raises rates, WACCs rise, and multiples compress — mechanically, not mysteriously. Understanding [WACC and cost of capital](/blog/trading/equity-research/wacc-weighted-average-cost-capital) is the prerequisite for understanding why sector multiples contracted 30–40% from 2021 to 2023 as rates rose.

---

### Summary: The Practitioner's Checklist

When you sit down to build a comps analysis, run through this sequence:

1. **Define the target clearly.** What business does this company actually operate? GICS code is a starting point, not the answer.
2. **Screen broadly, then narrow.** Start with 200+ companies, apply size/margin/geography screens, land at 6–12 peers.
3. **Normalize every peer's financials.** Add back restructuring, transaction costs, impairments. Document every adjustment.
4. **Use the right multiple for the industry.** EV/EBITDA for most; EV/Revenue for high-growth; P/E or P/Book for financials; FFO for REITs.
5. **Look at the cross-sectional relationship.** Build the EV/EBITDA vs. margin scatter. Understand where your target sits.
6. **Run precedent transactions separately.** Select 8–15 deals from the past 5–7 years. Use unaffected prices for premium calculation.
7. **Build the football field.** Overlay DCF, comps (multiple methods), and precedent transactions.
8. **Explain the divergences.** The analytical value is not in the ranges — it is in the story of why they agree or disagree.
9. **Triangulate to a point estimate.** Boards make decisions, not ranges. Weight the methods by their reliability for this specific situation.

Relative valuation is the market's verdict. Treat it with respect — and equal doses of critical judgment.
