---
title: "The Valuation Spectrum: Absolute, Relative, and Contingent Claims"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A complete map of every major valuation method — DCF, multiples, options pricing, and asset-based — with a practical decision framework for when to use which."
tags: ["valuation-methods", "absolute-valuation", "relative-valuation", "contingent-claims", "dcf", "multiples", "options-pricing", "valuation-framework", "finance", "investing"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — There is no single correct way to value an asset; each method answers a different question, and a thoughtful analyst uses several simultaneously, then reconciles the range.
>
> - **Absolute valuation** (DCF, DDM, NAV) asks: what is this asset worth on its own fundamentals?
> - **Relative valuation** (multiples, comps) asks: what is the market paying for similar assets right now?
> - **Contingent claims** (option-based models) asks: what is the value of flexibility — the right, not the obligation, to act?
> - **Asset-based** valuation asks: what would it cost to replicate or liquidate the asset today?
> - The "football field" chart, where an investment bank stacks four different value ranges on one chart, is not indecision — it is honest acknowledgment that value depends on which question you are asking.

It is the morning of an IPO. The bankers, lawyers, and company founders are crowded around a conference table in Midtown Manhattan. On the screen is a single slide: the "football field." It shows five horizontal bars, each representing a different valuation methodology, with dollar ranges stretching across the chart. The DCF says \$800 million to \$1.1 billion. The EV/EBITDA comparable companies analysis says \$850 million to \$1.0 billion. The precedent transactions analysis says \$950 million to \$1.15 billion. The 52-week trading range says \$820 million to \$1.02 billion. They all point toward a similar neighborhood, but none of them agree exactly. The bankers recommend pricing the IPO at \$900 million.

Why four methods? Why not just one correct answer?

The answer is that each method is literally answering a different question. The DCF asks: if you model this company's cash flows for the next ten years and discount them back to today, what do you get? The comparable companies analysis asks: what are investors currently paying for similar businesses in the public market? The precedent transactions analysis asks: what have acquirers historically paid for businesses like this one? The 52-week range asks: what has the market actually been willing to pay recently?

These are four distinct questions. They will naturally give four distinct answers. The skill of valuation is not finding the one true number — it is understanding what question each method is asking, when each question is most relevant, and how to reconcile the answers into a defensible range. That is what this post teaches.

![Valuation landscape — four families of methods in a grid](/imgs/blogs/valuation-spectrum-absolute-relative-contingent-claims-1.png)

## Foundations: The Four Families of Valuation

Before diving into any specific method, we need a mental map. Every valuation approach in existence belongs to one of four families. Understanding the family tells you the method's underlying logic — and its blind spots.

### The Four Families

**1. Absolute valuation (intrinsic value methods)**

These methods look only at the asset itself — its cash flows, dividends, or net asset composition — without reference to what the market is currently paying for similar assets. The central idea is that an asset is worth the present value of everything it will ever pay you. The most important absolute methods are:
- Discounted Cash Flow (DCF)
- Dividend Discount Model (DDM)
- Net Asset Value (NAV)

**2. Relative valuation (market-based methods)**

These methods anchor value to comparable assets that are already priced by the market. If similar businesses trade at 10 times EBITDA, and your business has \$100 million of EBITDA, then your business should be worth approximately \$1 billion — adjusted for differences in quality, growth, and risk. The most important relative methods are:
- Price/Earnings ratio (P/E)
- EV/EBITDA multiples
- Price/Book ratio (P/B)
- Comparable companies analysis ("comps")
- Precedent transactions analysis

**3. Contingent claims valuation (option-based methods)**

These methods treat an asset as an option — a right to receive a payoff if certain conditions are met. Equity in a leveraged company is literally a call option on firm assets. A mining company's oil reserve is a real option to produce when prices are high. These methods are the most sophisticated and the most misapplied. The most important methods are:
- Black-Scholes model (for financial options and equity)
- Real options analysis (for capital investment decisions)
- Merton model (for equity as a call on assets)

**4. Asset-based valuation**

These methods value an asset by tallying up its component parts — physical assets, intellectual property, working capital, minus liabilities. They are most relevant when the going-concern value (the value of the business as an operating enterprise) is near zero or uncertain. The most important methods are:
- Book value (accounting net worth)
- Liquidation value
- Replacement cost
- Adjusted Net Asset Value (ANAV)

### Why Methods Give Different Numbers

These four families are not interchangeable. They give different answers because they are measuring different things:

| Family | What it measures | Implicit assumption |
|--------|-----------------|---------------------|
| Absolute | Present value of future cash | The future can be modeled |
| Relative | Market consensus on similar assets | Comparable assets exist and are fairly priced |
| Contingent claims | Value of optionality | Volatility and time have quantifiable worth |
| Asset-based | Breakup or replacement value | Assets have observable standalone prices |

A manufacturing plant in a declining industry might have:
- A DCF value of \$350 million (if it keeps operating profitably)
- A comparable companies value of \$280 million (if peers are cheap)
- A liquidation value of \$120 million (if you shut it down and sell the machines)
- A replacement cost of \$500 million (to build it from scratch today)

All four numbers are correct. The question is which one is relevant to your decision.

## Absolute Valuation: Intrinsic Value from Fundamentals

The foundational idea behind absolute valuation is deceptively simple: an asset is worth the present value of all future cash flows it will generate for you. This idea goes back to John Burr Williams, who articulated it in *The Theory of Investment Value* in 1938, and it underlies virtually every sophisticated valuation done today.

### Discounted Cash Flow (DCF)

The DCF model says: forecast every cash flow the asset will produce, then discount each one back to today at a rate that reflects the riskiness of those cash flows.

For a company, the relevant cash flows are **free cash flows** — the cash generated by operations after reinvestment in the business, before financing costs. The discount rate is the **Weighted Average Cost of Capital (WACC)** — the blended required return across debt and equity.

The formula is:

```
Enterprise Value = Sum of [FCF_t / (1 + WACC)^t]  for t = 1 to infinity
```

Since we cannot model infinity directly, we break this into two pieces:
1. A detailed projection period (typically 5–10 years)
2. A terminal value that captures everything beyond the projection period

The terminal value typically uses either the Gordon Growth Model (assume FCF grows at a stable rate forever) or an exit multiple (assume the business is sold at a market multiple in year 10).

```
Terminal Value (Gordon Growth) = FCF_{n+1} / (WACC - g)
```

Where `g` is the long-run growth rate (typically close to GDP growth — 2–3% for a mature business in the US).

![DCF value driver tree — FCF, discount rate, terminal value branches](/imgs/blogs/valuation-spectrum-absolute-relative-contingent-claims-2.png)

#### Worked example: Manufacturing firm DCF

A manufacturing company generates \$100 million of EBITDA. After depreciation, interest, taxes, and capex, its free cash flow is \$60 million per year. You expect FCF to grow at 3% per year for 10 years, then stabilize. The WACC is 9%.

Step 1: Project FCF for 10 years (Year 1: \$61.8M, Year 2: \$63.7M, ... Year 10: \$78.1M)

Step 2: Discount each year's FCF back:
- Year 1: \$61.8M / 1.09 = \$56.7M
- Year 2: \$63.7M / 1.09² = \$53.6M
- ...continuing to year 10: sum of present values = approximately \$470M

Step 3: Terminal value using Gordon Growth:
- FCF in year 11 = \$78.1M × 1.03 = \$80.4M
- Terminal Value = \$80.4M / (0.09 − 0.03) = \$1,340M
- Present value of terminal value = \$1,340M / 1.09¹⁰ = \$549M

Step 4: Enterprise Value = \$470M + \$549M = **\$1,019M** ≈ **\$1.0 billion**

The intuition: about half the value comes from the explicit forecast period, half from the terminal value. This is typical for a stable business. For a high-growth company, the terminal value can represent 70–80% of total value — which is why growth assumptions are so sensitive.

### Dividend Discount Model (DDM)

The DDM is the oldest and simplest absolute valuation model. It says: the value of a share is the present value of all future dividends.

```
P = D_1 / (r - g)
```

Where `D_1` is next year's expected dividend, `r` is the required return on equity, and `g` is the expected long-run dividend growth rate.

The DDM is elegant but narrow. It works best for:
- Mature companies with long, stable dividend histories (utilities, REITs, consumer staples)
- Banks (which pay dividends from income rather than free cash flow)
- Situations where dividends are a reliable indicator of earnings quality

It fails for companies that pay no dividends (tech startups, growth companies), companies where dividends and earnings are badly misaligned, or companies with leverage that makes free cash flow the better metric.

#### Worked example: Utility company DDM

A regulated electric utility pays \$2.40 per share in annual dividends. Dividends have grown at 3% per year for 20 years and are expected to continue at that pace. The required return on utility stocks (based on beta and CAPM) is 7%.

Value per share = \$2.40 × 1.03 / (0.07 − 0.03) = \$2.47 / 0.04 = **\$61.75 per share**

If the stock trades at \$65, it is modestly overvalued relative to its dividend stream at these inputs. If rates rise and the required return moves from 7% to 8%, value drops to \$2.47 / 0.05 = \$49.40 — a 20% decline. This is why utility stocks are so sensitive to interest rates.

### Net Asset Value (NAV)

NAV is the asset-by-asset sum of a portfolio or holding company, net of liabilities. It is the natural valuation method for:
- Investment funds and ETFs (each share's value = fund's portfolio / shares outstanding)
- Real estate companies and REITs (value of properties + other assets − debt)
- Natural resource companies (present value of reserves, net of development costs and debt)

For a REIT, the calculation is:
1. Value each property using comparable cap rates (net operating income / market cap rate)
2. Add liquid assets (cash, short-term investments)
3. Subtract total debt and preferred equity obligations
4. Divide by shares outstanding

The result is NAV per share. If the REIT trades below NAV, investors are getting the properties at a discount. If it trades above NAV, they are paying a premium for the management team's capital allocation skill.

## Relative Valuation: Value From Comparables

Relative valuation is the most widely used approach in practice. It is fast, intuitive, and requires far fewer assumptions than a DCF. The central logic: markets are reasonably efficient at pricing similar assets relative to each other — even if the absolute level of the market is uncertain.

### How Multiples Work

A multiple is a ratio that normalizes price by some fundamental. The most common:

| Multiple | Formula | Denominator meaning |
|---------|---------|---------------------|
| P/E | Price / EPS | Earnings per share |
| EV/EBITDA | Enterprise Value / EBITDA | Earnings before interest, taxes, D&A |
| EV/Revenue | Enterprise Value / Revenue | Useful for unprofitable companies |
| P/B | Price / Book Value per share | Net accounting assets |
| P/FCF | Price / Free Cash Flow | Cash generated for owners |

The workflow is:
1. Identify a group of comparable companies ("comps")
2. Calculate each comp's multiple
3. Compute the median (or mean) of the peer group
4. Apply that multiple to the subject company's fundamental
5. This gives you an implied equity or enterprise value

![Multiples valuation pipeline — 5-node process from peer selection to equity value](/imgs/blogs/valuation-spectrum-absolute-relative-contingent-claims-3.png)

#### Worked example: Same company, four methods

Return to our manufacturing company with \$100M EBITDA and \$60M free cash flow. After-tax earnings are \$45M. Book value of equity is \$450M.

**Method 1 — DCF:** \$1,019M enterprise value (computed above)

**Method 2 — EV/EBITDA comps:** Comparable manufacturers trade at a median of 8.5× EBITDA.
- Implied enterprise value = \$100M × 8.5 = **\$850M**

**Method 3 — P/E comps:** Comparable companies trade at a median of 17.3× earnings.
- Implied equity value = \$45M × 17.3 = **\$779M**
- If net debt is \$50M, implied enterprise value = \$779M + \$50M = \$829M

**Method 4 — Book value:** Net equity is \$450M. This is not a market value — it is the accounting residual. It provides a floor (what you would recover if the business generated zero returns above the cost of capital).

Why the spread? The DCF's \$1,019M is the highest because it builds in 3% annual growth for 10 years. If peer companies are declining or stagnant, their 8.5× EBITDA multiple already prices in lower growth — hence the lower relative value. The P/E approach is lower still because earnings (\$45M) contain the full tax and interest burden, making the multiple more sensitive to leverage. Book value (\$450M) is lowest of all — it is historical cost, not future earning power.

The spread of \$779M to \$1,019M — a 31% range — is not a sign that valuation is impossible. It is the honest range of uncertainty. A reasonable analyst would center on \$850–950M and set a buy/sell threshold around the current price.

### Precedent Transactions

A close relative of comparable companies is precedent transactions analysis — instead of looking at public market multiples, you look at what acquirers have paid to buy similar companies outright. Historically, acquisition multiples are 20–40% higher than comparable companies multiples because:
1. Control premium: the buyer must compensate existing shareholders for giving up optionality
2. Synergies: the buyer expects to generate value through cost savings, cross-selling, or market power
3. Competition: in contested processes, bidders push prices up

For our manufacturing example: if precedent transactions in the sector averaged 9.5× EBITDA (a 1× control premium over the 8.5× trading multiple), then:
- Precedent transactions value = \$100M × 9.5 = **\$950M**

This is why M&A advisors show both comps and precedent transactions — the spread between them tells you how much of a deal premium is embedded in the current offer price.

### Why Multiples Differ Across Sectors and Time

Multiples are not constants — they are functions of growth rates, discount rates, and profit margins. The fundamental relationship between a P/E ratio and underlying economics is:

```
P/E = (1 − b) / (r − g)
```

Where `b` is the reinvestment (plowback) rate, `r` is the required return on equity, and `g` is the long-run earnings growth rate. This is the DDM restated in earnings terms (the "earnings model" derived by Modigliani and Miller). From this formula:

- **Higher growth → higher P/E.** If `g` rises from 3% to 5% while `r` stays at 10%, P/E rises from 10× to 16.7×. This explains why fast-growing tech companies trade at higher P/E than slow-growing utilities.
- **Higher interest rates → lower P/E.** If `r` rises from 10% to 12% (e.g., the Fed hikes rates), P/E falls from 10× to 7×. This is exactly what happened in 2022: as the Fed raised rates by 5 percentage points, P/E multiples contracted by 30–40% across the market.
- **Higher profit margins → higher revenue multiples.** A software company with 30% net margins justifies a much higher EV/Revenue multiple than a retailer with 2% net margins.

Understanding multiples as implied functions of growth, return, and payout makes you much harder to fool. When a banker says "this company deserves a premium multiple," the burden of proof is on them to show that growth, margins, or capital efficiency genuinely exceed the peer group — not just that management is enthusiastic.

### LBO Analysis: The Private Equity Version of Relative Valuation

In leveraged buyout (LBO) transactions, private equity buyers use a variant of relative valuation that is anchored to the return they need to earn. The "LBO analysis" asks: at what acquisition price can a financial sponsor buy this business, lever it up, improve it over 5 years, and sell it at a market multiple — while achieving a target IRR (typically 20–25%)?

The output of an LBO analysis is the maximum entry price consistent with the target return — which functions as a floor in competitive auction processes. Sellers want to understand the LBO floor because it tells them the minimum they should accept from a financial buyer (as opposed to a strategic acquirer who can pay more due to synergies).

A simplified LBO:
- **Acquisition price**: \$1,000M (10× EBITDA of \$100M)
- **Leverage**: \$650M debt, \$350M equity (65% LTV)
- **EBITDA growth**: 7% per year for 5 years → EBITDA grows to \$140M
- **Exit multiple**: 9× EBITDA → exit enterprise value = \$1,260M
- **Debt paydown**: repay \$200M of debt over 5 years → exit debt = \$450M
- **Exit equity value**: \$1,260M − \$450M = **\$810M**
- **Return on \$350M equity investment**: \$810M / \$350M = 2.3× money (MOM) in 5 years → approximately 18% IRR

At 18%, this deal is just below the 20% hurdle. A buyer targeting 20% IRR would only pay approximately \$930M (9.3× EBITDA) — creating a "bid-ask spread" with the seller who wants \$1,000M.

### Sector-Appropriate Multiples

Not every multiple works for every sector. Banks, for example, cannot be valued on EV/EBITDA because "debt" (deposits) is their raw material — you cannot separate operating and financing cash flows. For banks, P/B (price to book value) and P/E are the dominant metrics. For REITs, Price/Funds from Operations (P/FFO) is standard. For tech startups with no earnings, EV/Revenue and growth-adjusted multiples (PEG ratio) dominate.

![Sector-appropriate methods heatmap — rows are sectors, columns are methods](/imgs/blogs/valuation-spectrum-absolute-relative-contingent-claims-7.png)

## Contingent Claims Valuation: Pricing Optionality

This is the most intellectually rich — and most misunderstood — family of valuation methods. The key insight: **equity in a company with debt is literally a call option.**

### Equity as a Call Option

Imagine a company has \$1,000M of assets and \$400M of debt (face value). The debt matures in one year. At maturity:
- If assets are worth more than \$400M: equity holders pay off the debt and keep the rest. Equity value = assets − \$400M.
- If assets are worth less than \$400M: equity holders walk away (limited liability). Equity value = \$0.

This is *exactly* the payoff of a call option:
- Underlying: firm asset value
- Strike price: debt face value (\$400M)
- Equity value = max(Assets − Debt, 0)

![Equity as a call option on firm assets — before/after around the debt level](/imgs/blogs/valuation-spectrum-absolute-relative-contingent-claims-4.png)

The Black-Scholes formula for a call option is:

```
C = S × N(d1) − K × e^(-rT) × N(d2)

d1 = [ln(S/K) + (r + σ²/2) × T] / (σ × sqrt(T))
d2 = d1 − σ × sqrt(T)
```

Where `S` = current asset value, `K` = debt face value, `r` = risk-free rate, `T` = time to debt maturity, `σ` = asset volatility, and `N()` = cumulative normal distribution.

Applying Black-Scholes to value equity (the Merton model):
- If assets = \$1,000M, debt = \$400M, maturity = 5 years, asset volatility = 25%, risk-free rate = 4.5%
- d1 = [ln(1000/400) + (0.045 + 0.03125) × 5] / (0.25 × 2.236) = 2.11
- d2 = 2.11 − 0.559 = 1.55
- N(2.11) ≈ 0.983, N(1.55) ≈ 0.939
- Equity (call) = \$1,000 × 0.983 − \$400 × e^(-0.225) × 0.939 = \$983 − \$302 = **\$681M**

Why does this differ from the simple \$600M (\$1,000M − \$400M)? Because optionality has value. The equity holders hold an option that benefits from upside volatility (assets could go to \$1,500M) while being floored at zero (limited liability caps the downside). That time-and-volatility value adds \$81M above the intrinsic value.

### Real Options

Real options extend this logic to physical investment decisions. A mining company with an undeveloped gold deposit holds a real option: the right, but not the obligation, to mine if gold prices rise above the cost of extraction. The option has value even if current prices are below the break-even — because prices are volatile and time remains.

Real option value = DCF value of developing the project (if undertaken today) + value of the option to wait

The canonical real option types:
- **Option to defer**: wait for better market conditions before committing capital
- **Option to expand**: if the first phase succeeds, invest more
- **Option to abandon**: sell or liquidate if conditions deteriorate
- **Option to switch**: use inputs or outputs flexibly based on prices

Real options analysis is most valuable when:
1. There is significant uncertainty that will be resolved over time
2. Management has genuine flexibility to respond to information
3. The investment is staged (can be stopped mid-way)

For a \$50M pharmaceutical R&D project with a 20% chance of FDA approval in 5 years, and a payoff of \$500M if approved:
- Simple NPV: −\$50M + 0.20 × \$500M / (1.10)^5 = −\$50M + \$62M = +\$12M
- Real option value adds the value of being able to stop spending if Phase 2 fails — could add another \$5–10M depending on volatility of success probability

## The Mechanics of Real Options: When Flexibility Has a Price

The contingent claims framework is especially powerful for companies whose value is mostly "option-like" — where a large part of the payoff depends on whether certain conditions are met in the future. Three practical applications deserve attention.

### Pharmaceutical R&D Pipelines

A biotech company with 10 drugs in Phase 1 trials has a portfolio of binary options. Each drug either gains FDA approval (a highly valuable payoff) or fails (the investment is largely lost). A simple DCF would take expected cash flows — probability-weighted average of success and failure — and discount them at a high rate. This approach systematically undervalues the pipeline because it ignores the ability to abandon failing drugs early.

Real option value = cost of running Phase 1 (the "option premium") + value of proceeding to Phase 2 only if Phase 1 looks promising (the ability to stop spending). For a 5-year program costing \$80M where management can pull the plug at each phase:
- DCF value of full commitment: −\$80M + 0.15 × \$600M present value of launch = +\$10M
- Real option value of staged program: +\$10M + value of flexibility to stop ≈ +\$25M

The "option to abandon" is worth \$15M here — real money, and it gets larger when:
1. Uncertainty is high (volatile outcomes)
2. Commitment costs are front-loaded (most spending comes early)
3. Resolution time is long (you learn slowly whether the drug works)

### Natural Resource Reserves

A gold mining company has a reserve of 500,000 ounces of gold. Current extraction cost: \$1,600/oz. Current gold price: \$1,850/oz. The mine is profitable today — but should the company extract immediately or wait?

If gold prices could rise to \$2,500/oz or fall to \$1,200/oz over the next 3 years, and the company can choose to mine when prices are high and wait when they are low, then:
- Expected value of mining now: (1,850 − 1,600) × 500,000 = \$125M
- Value of option to delay: at \$2,500/oz → \$450M profit; at \$1,200/oz → mine is unprofitable, do not mine; weighted probability value > \$125M

The real option approach gives a higher value than the DCF "mine it now" approach because it captures the asymmetric payoff: participate in upside, walk away from downside. The Black-Scholes formula (or a binomial tree for longer horizons) can quantify this premium precisely.

### Technology Investment Platforms

Platform investments — cloud infrastructure, foundational AI models, electric vehicle charging networks — cannot be valued by DCF alone because their primary value is the *options they create*: to expand into adjacent markets, launch new products, or foreclose competitors.

Amazon's investment in AWS was not justified by near-term cloud revenue in 2006. It was a real option on becoming the infrastructure layer for the entire internet economy. Netflix's content investment is a real option on subscriber loyalty and international expansion. These "growth options" are why high-growth technology companies trade at P/E ratios of 30–50× — the multiple embeds the option value of future platform extensions, not just current earnings.

## Asset-Based Valuation: What the Pieces Are Worth

Asset-based valuation is the simplest in concept: add up the assets, subtract the liabilities, and the remainder is equity value. In practice, the question is: which value of assets — historical cost, current market price, liquidation value, or replacement cost?

### Book Value

Book value is accounting net worth: assets recorded at historical cost (adjusted for depreciation) minus all liabilities. It is almost never an accurate estimate of market value because:
1. Real estate and equipment are carried at cost, not current market price
2. Intangible assets (brands, patents, customer relationships) are often not on the balance sheet at all
3. Goodwill from past acquisitions may be carried at values that no longer reflect reality

Book value is most useful as a **floor** — the accounting minimum from which any going-concern should generate a return. A company trading below book value is saying: the market believes the business will destroy value over time (returns on equity below the cost of equity).

#### Worked example: Bank valuation — why P/B dominates

Banks cannot be valued on DCF because their "operating expenses" and "financing costs" are inseparable — the spread between deposit rates and loan rates is their entire business model. The natural metric is Price/Book.

A regional bank has:
- Total loans: \$8.0 billion
- Securities: \$1.5 billion
- Other assets: \$0.5 billion
- Total assets: \$10.0 billion
- Total liabilities (deposits + borrowings): \$9.1 billion
- **Book value of equity: \$0.9 billion**

If the bank earns a return on equity (ROE) of 12% and the cost of equity is 10%, it should trade **above** book — because it is generating value. A premium of roughly ROE/CoE = 1.2× is justified, implying a market cap of \$1.08 billion.

If a competing bank is struggling with credit losses and earns ROE of only 7% against a cost of equity of 10%, it should trade **below** book — say 0.8×, implying market cap of only \$720M against a \$900M book. Investors are pricing in future value destruction.

The formula: **P/B = (ROE − g) / (r − g)**

Where `r` is the cost of equity and `g` is long-run growth. This is the DDM restated in book-value terms, and it explains why P/B ratios vary so much across banks — they are ultimately driven by the spread between ROE and cost of equity.

### Liquidation Value

Liquidation value is what you would actually receive if you sold every asset immediately and paid off all liabilities. It is always less than going-concern value because:
- Fire-sale discounts apply to equipment and inventory
- Intangibles (customer relationships, brands) have near-zero liquidation value
- Transaction costs and professional fees eat into proceeds

Liquidation value is the relevant number when:
1. A company is in or near bankruptcy
2. A creditor is deciding whether to push for liquidation or support a restructuring
3. You are computing a "net net" value (Ben Graham's margin of safety approach)

### Replacement Cost

Replacement cost asks: how much would it cost to build this business from scratch today? It is most relevant for capital-intensive businesses where competitive advantage comes primarily from physical infrastructure — refineries, transmission networks, railroads, mines.

If a company's enterprise value is well below replacement cost, no rational competitor will build a new facility — existing capacity is more valuable than new capacity. This provides a durable competitive advantage (often called an "economic moat").

If enterprise value is well above replacement cost, new entrants will be attracted to the industry, and competition will eventually compress returns back toward the cost of capital.

## Choosing the Right Method: A Decision Framework

Given four families of valuation methods, how do you decide which to use? The answer depends on five factors: the sector, the company's stage, data availability, the purpose of the valuation, and the nature of the assets.

![Method selection decision tree — key branching questions leading to method choices](/imgs/blogs/valuation-spectrum-absolute-relative-contingent-claims-5.png)

### Factor 1: Sector

Different sectors have evolved their own standard methods because their economics differ:

**Banks and insurers**: P/B and P/E. DCF is not applicable (cannot separate operating and financing cash flows). Regulatory capital requirements make book value a meaningful constraint.

**Utilities and regulated infrastructure**: DDM or DCF with high terminal value weight. Stable, contractual cash flows make DCF very reliable. P/E is also used for regulated utilities.

**Technology companies**: EV/Revenue (for pre-profit companies), DCF (for profitable ones with predictable cash flows), and scenario-based real options for early-stage bets.

**Real estate**: NAV (for REITs and property companies), P/FFO (price to funds from operations), and direct cap rate valuation of properties.

**Natural resources (mining, oil & gas)**: NAV based on reserve valuation, real options for undeveloped assets, and EV/EBITDA for producing companies.

**Retail and consumer**: EV/EBITDA, P/E, and same-store sales multiples.

**Financial sponsors (private equity)**: LBO analysis (what IRR does a buyer earn at a given entry multiple and exit multiple?) + DCF + EV/EBITDA comps.

### Factor 2: Company Stage

**Early-stage (pre-revenue)**: Traditional multiples do not apply. Use venture capital method (revenue multiple at exit × probability of exit = expected value) or real options.

**Growth stage (revenue but unprofitable)**: EV/Revenue, sometimes EV/Gross Profit. DCF requires aggressive assumptions but is still used as a sanity check.

**Mature/stable**: DCF, EV/EBITDA, P/E all apply. The spread between methods narrows because the business is predictable.

**Distressed or declining**: Liquidation value becomes relevant. The key question shifts from "what will this company earn?" to "what will lenders recover?"

### Factor 3: Purpose

**IPO valuation**: Comparable companies analysis dominates. Banks want a number anchored to current market conditions. DCF is the "reality check."

**M&A (buy-side)**: DCF with synergies is the primary tool. The acquirer wants to know: what is it worth *to us*, including the value we can create?

**M&A (sell-side/defense)**: Comparable companies, precedent transactions, and a DCF with standalone projections. The defense wants to show shareholders what the company is worth without a deal.

**Credit analysis**: Liquidation value, debt coverage ratios, and enterprise value relative to total debt. The creditor asks: can we get our money back?

**Activist investing**: Book value and asset-based methods. Activists look for companies where liquidation value > market cap ("sum of the parts" trades).

### Factor 4: Data Availability

DCF requires a multi-year financial model, which requires management guidance, industry data, and significant analyst effort. For private companies, even the starting data may be unavailable or unreliable.

Relative valuation requires a peer group. If there are no publicly traded comparables (for a very unique business, or in a private market), this approach is limited.

Contingent claims requires volatility inputs, which are not always observable.

### Factor 5: The Purpose of the Valuation

The same company can be "worth" different amounts depending on who is asking and why. Consider a consumer food brand with \$50M of EBITDA and \$500M of enterprise value:

- **To a passive financial investor** (mutual fund): \$500M is roughly fair value based on a 10× peer multiple. The investor is willing to pay up to \$500M for a minority stake.
- **To a strategic acquirer** (a larger food conglomerate): The target is worth \$650M, because the acquirer can cut \$20M of duplicated overhead (synergy = \$20M / 0.09 WACC = \$222M additional value; minus integration costs of \$72M = \$150M net synergy). The acquirer can rationalize paying a 30% premium.
- **To a financial sponsor** (private equity): The LBO analysis suggests maximum entry price of \$580M (given leverage constraints and target IRR).
- **To a creditor** in a distress scenario: Liquidation value of the brand + physical assets + working capital = \$180M.

These four "values" — \$500M, \$650M, \$580M, \$180M — are all correct given their respective contexts. "What is it worth?" is always shorthand for "what is it worth to whom, for what purpose, under what scenario?"

#### Worked example: Sum-of-parts valuation

A diversified conglomerate has three divisions:
1. **Consumer goods division**: \$120M EBITDA × 10× (consumer sector multiple) = \$1,200M enterprise value
2. **Industrial manufacturing**: \$80M EBITDA × 7× (industrial multiple) = \$560M
3. **Financial services subsidiary**: Book value \$300M × 1.4× P/B = \$420M

Total sum-of-parts enterprise value: \$1,200M + \$560M + \$420M = **\$2,180M**

Subtract: conglomerate-level net debt of \$400M + corporate overhead (capitalize \$30M at 10×) = \$700M

Implied equity value: \$2,180M − \$700M = **\$1,480M**

If the stock's current market cap implies only \$1,100M of equity value, there is a \$380M "conglomerate discount." Activists often target such companies, arguing that management should spin off the divisions and unlock the \$380M of hidden value. This is exactly what pressure from Carl Icahn on Motorola (2008) or Nelson Peltz on Procter & Gamble (2017) looked like: a sum-of-parts analysis showing that the parts were worth more than the whole.

### The Decision Matrix

A quick heuristic: 

| Situation | Primary method | Secondary check |
|-----------|---------------|-----------------|
| Mature company, stable cash flows | DCF | EV/EBITDA comps |
| Bank or insurer | P/B | P/E, DDM |
| Utility or regulated asset | DDM or DCF | P/E comps |
| REIT or property company | NAV | P/FFO comps |
| Early-stage startup | VC method | Real options |
| Distressed company | Liquidation value | DCF (recovery case) |
| Natural resource company | Reserve NAV | EV/EBITDA of production |
| Pharma/biotech | Risk-adjusted NPV | Sum of parts |
| LBO target | LBO analysis | DCF, EV/EBITDA comps |

## The Football Field Chart: How Bankers Use All Methods Together

The "football field" is an investment banking staple: a horizontal bar chart where each bar represents a valuation method's implied range. The visual is deliberately designed to show overlap, confirm that the various methods are giving roughly consistent answers, and identify outliers.

![Football field chart — valuation ranges by method with current price marker](/imgs/blogs/valuation-spectrum-absolute-relative-contingent-claims-6.png)

### How to Read It

The football field for our hypothetical company shows:
- **DCF**: \$800M–\$1,100M (wide range reflects sensitivity to growth and discount rate assumptions)
- **EV/EBITDA comps**: \$850M–\$1,000M (range reflects 25th–75th percentile of peer multiples)
- **P/E comps**: \$780M–\$980M (range reflects earnings variability and peer multiple spread)
- **Precedent transactions**: \$950M–\$1,150M (elevated due to control premium)
- **52-week trading range**: \$820M–\$1,020M (where the market has traded this stock over the past year)
- **Current price**: \$900M (marked with a vertical line)

The fact that \$900M falls comfortably within most of the bars is a sign that the current price is within a reasonable valuation range. If the current price were at \$1,100M, it would be at or above the top of the DCF range and well above the comps — a warning signal of potential overvaluation.

### Building the Ranges

For each method, the range comes from sensitivity analysis:
- **DCF range**: Run the model at a bull/bear/base discount rate (WACC of 8%/9%/10%) and growth rate (2%/3%/4%), and take the outer bounds of the realistic scenarios.
- **Comps range**: Take the 25th percentile multiple and the 75th percentile multiple from the peer group, then apply each to your company's fundamental.
- **Precedent transactions range**: Take the low and high transaction multiples from the last 3–5 years of relevant M&A deals.

When ranges do *not* overlap, that is a signal to investigate. Either a method is being misapplied (wrong peer group, wrong cash flow definition), or the asset has unusual characteristics that make one family of methods inappropriate.

## How Methods Give Different Answers — and Why That Is OK

One of the most common anxieties in valuation is: "If the methods give different answers, which one is right?" The answer is that each method is right about what it is measuring, and the analyst's job is to understand *why* the methods diverge, not to arbitrarily average them.

### Sources of Divergence

**1. Different time horizons**

DCF captures the long-run intrinsic value of the business, including growth that has not yet materialized. Market multiples capture what investors are willing to pay *today*, which reflects current sentiment, liquidity conditions, and near-term earnings visibility. In 2020, many profitable companies traded at discounts to their DCF values because near-term cash flows were uncertain during COVID-19 lockdowns.

**2. Market mispricing**

If the entire peer group is overvalued, comparable companies analysis will give you an inflated number. This is exactly what happened in the dot-com bubble: internet companies were valued on EV/Eyeballs because no other metric worked — and the "comps" were themselves wildly overvalued. Using a DCF would have revealed the absurdity.

Conversely, if the market is in a panic (March 2020, October 2008), comps will give you values below long-run intrinsic value. DCF becomes the more reliable anchor.

**3. Capital structure differences**

EV/EBITDA compares enterprise values (which include debt) to pre-financing earnings. P/E compares equity price to post-financing earnings. A company with a lot of debt will appear cheaper on EV/EBITDA relative to P/E than a debt-free peer. Neither is wrong — they are just measuring different layers of the capital structure.

**4. Accounting differences**

Two identical businesses using different accounting methods (FIFO vs. LIFO inventory, different depreciation schedules) will have very different P/E ratios but similar EV/EBITDA ratios (because EBITDA adds back depreciation). This is why EV/EBITDA is often preferred to P/E when comparing companies across industries or geographies.

### Reconciliation: Weighting the Methods

Investment banks typically weight the methods based on their reliability for the specific situation. For a mature consumer goods company:
- DCF: 40% weight (high reliability for stable cash flows)
- EV/EBITDA comps: 40% weight (many comparable peers)
- Precedent transactions: 20% weight (sparse recent M&A data)

If DCF gives \$950M, comps give \$880M, and transactions give \$1,020M:
- Weighted value = 0.40 × \$950M + 0.40 × \$880M + 0.20 × \$1,020M = \$380M + \$352M + \$204M = **\$936M**

![Valuation reconciliation — weighting four methods into a single range](/imgs/blogs/valuation-spectrum-absolute-relative-contingent-claims-8.png)

The weighted value is \$936M. If the company is available for purchase at \$850M, the \$86M gap between price and value gives you your margin of safety.

## Common Misconceptions

### Misconception 1: "The DCF is the most accurate method."

The DCF is only as good as its inputs. Because 40–60% of DCF value often comes from the terminal value, and the terminal value is exquisitely sensitive to the assumed discount rate and growth rate, small changes in inputs produce enormous changes in output. A WACC change from 9% to 10% (one percentage point) can reduce a DCF value by 15–20%. The DCF gives the *illusion* of precision because it produces a single number — but that number has a massive confidence interval. The honest use of DCF involves sensitivity tables and scenario analysis, not a single point estimate.

### Misconception 2: "Relative valuation is just picking an arbitrary multiple."

The multiple is not arbitrary — it is derived from a peer group that reflects what investors are currently willing to pay for similar cash flows and similar risk profiles. The discipline is in selecting the peer group (are these companies truly comparable in growth, margins, and risk?) and in normalizing for differences (adjusting for leverage, non-recurring items, accounting differences). Done sloppily, it is arbitrary. Done carefully, it is a powerful reality check on DCF assumptions.

### Misconception 3: "Book value is a good floor for any stock."

Book value is a reliable floor only for businesses where assets are fairly valued on the balance sheet — mainly asset-heavy industrials, financial companies, and real estate. For a software company, the most valuable asset is intellectual property and customer relationships, neither of which typically appears on the balance sheet. Microsoft's book value per share might be \$28 while its market price is \$420. This does not mean Microsoft is overvalued by 15×; it means that book value is simply the wrong method for software.

### Misconception 4: "Options pricing is too complex for equity valuation."

The Black-Scholes model feels intimidating, but the underlying logic — equity is a call option, and call options are worth more when the underlying is volatile and time remains — is simple and powerful. In practice, the Merton model is most useful for distressed companies (where the probability of the equity call expiring worthless is meaningful) and for biotechs or mining companies (where cash flows are binary based on a future event like a trial result or a resource discovery). For normal profitable companies, DCF and multiples are more practical.

### Misconception 5: "There is always a 'right' valuation number."

Valuation is estimation, not calculation. Unlike an accounting balance sheet, which has rules and auditors, a valuation depends on assumptions about an unknowable future. Two equally skilled analysts, starting from the same financial statements, can produce valuations that differ by 30–40% — and both can be defensible. The goal is not precision but calibration: understanding the range of reasonable values, the key drivers of that range, and the price at which you are adequately compensated for the uncertainty.

## How It Shows Up in Real Markets

### IPO Valuation: The Airbnb Case

When Airbnb went public in December 2020, investment banks used a combination of methods:
- **Comparable companies**: Booking Holdings (BKNG) and Expedia traded at roughly 15–25× revenue. Airbnb's unique marketplace model and higher growth rate justified a premium.
- **DCF**: With COVID-19 suppressing near-term bookings, the DCF was heavily weighted toward a recovery scenario — effectively a probability-weighted scenario analysis.
- **Precedent transactions**: OTA acquisitions from 2015–2019 provided transaction multiples for context.

The IPO priced at \$68 per share (implying roughly 18× 2021 estimated revenue). The stock opened at \$146 on day one — nearly 2× the IPO price — suggesting the banks had intentionally left money on the table to ensure a successful "pop," or that they significantly underestimated demand. In either case, the IPO football field undervalued the company.

### M&A Valuation: The Distressed Case

#### Worked example: Distressed company — which value matters?

A retailer has three valuations:
- Going-concern DCF: \$350M (if the turnaround works and margins recover)
- Comparable companies: \$200M (peers are all distressed; the sector multiple is depressed)
- Liquidation value: \$120M (real estate \$80M + inventory \$60M − severance/wind-down costs \$20M)

The company has \$300M of debt. Which value matters?

It depends on what happens:
- If lenders extend the debt maturity and management executes the turnaround: DCF value is relevant. Equity is worth \$350M − \$300M = \$50M.
- If lenders force bankruptcy and the business is liquidated: Liquidation value is relevant. Lenders recover \$120M out of \$300M owed — an 60% loss. Equity is worth \$0.
- If a strategic buyer acquires the business for its store network: Precedent transaction multiples apply. An acquirer might pay 6× EBITDA (\$180M for \$30M EBITDA), implying equity value of \$180M − \$300M = −\$120M (deeply underwater — acquirer assumes debt or requires it to be paid off pre-close).

In distressed situations, the equity option value is real but small — equity holders are betting on the call option (assets > liabilities), but the strike is deep in-the-money of the lenders, not the equity.

### Venture Capital: When Nothing Works But One Thing

#### Worked example: Startup valuation with no earnings

A Series A startup has:
- Current annual recurring revenue (ARR): \$10M
- Growing 200% year over year
- No profits (EBITDA: −\$8M)
- No dividends, no debt, no book value to speak of

Traditional DCF requires cash flows — there are none. Comps are sparse. Asset-based valuation gives you furniture and laptops.

The venture capital method:
1. Estimate exit ARR in 5 years: \$10M × (1 + 2.0)^5 is unrealistic over the full five years; a more reasonable growth decay path gives \$100M ARR by year 5.
2. Apply exit revenue multiple: SaaS companies with 30–40% growth rates trade at 8–12× ARR. Use 10×.
3. Exit value: \$100M × 10 = \$1.0 billion (a "unicorn" exit)
4. Expected exit proceeds to this investment: VC owns 25% → \$250M
5. Discount for time and risk: VC requires 30% annual return → PV = \$250M / (1.30)^5 = \$71M

Post-money valuation at Series A: if VC invests \$20M for 25%, the implied post-money is \$80M (\$20M / 0.25).

If the DCF-equivalent probability-weighted exit value is \$71M and the investor is paying \$80M post-money, the deal is marginally expensive at these exit assumptions. The investor would either negotiate a lower valuation or require better terms (liquidation preference, anti-dilution).

This exercise shows why venture investing requires a "power law" mindset: the rare 10× or 100× outcomes must more than compensate for the majority of investments that return zero or near-zero.

## Further Reading and Cross-Links

The valuation spectrum connects to every other piece of valuation theory in this series and beyond:

**Within this series:**
- [What Is Value: Philosophy, Frameworks, and Asset Pricing](/blog/trading/asset-valuation/what-is-value-philosophy-frameworks-asset-pricing) — the philosophical grounding for why assets have value at all
- [The Time Value of Money: The Engine Behind Every Valuation Model](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model) — the mathematics that makes DCF and DDM work
- [Risk and Required Return: CAPM, Beta, and Cost of Capital](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital) — how the discount rate is derived from first principles
- [Discount Rates in Practice: WACC, Cost of Equity, and Unlevered Beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta) — the mechanics of computing the denominator in a DCF

**Sibling series:**
- [Discounted Cash Flow: The Complete Guide](/blog/trading/equity-research/discounted-cash-flow-dcf-complete-guide) — a deep operational walkthrough of building a DCF model from scratch, in the equity research series

**External depth:**
- Damodaran, Aswath. *Investment Valuation*. 3rd ed. Wiley, 2012. — The definitive textbook on valuation; Damodaran's website (damodaran.com) maintains free datasets of sector multiples, betas, and risk premiums.
- McKinsey & Company. *Valuation: Measuring and Managing the Value of Companies*. 7th ed. Wiley, 2020. — The practitioner's companion; especially strong on DCF mechanics and M&A valuation.
- Koller, Tim, Marc Goedhart, and David Wessels. *Value: The Four Cornerstones of Corporate Finance*. Wiley, 2011. — A shorter distillation focused on the core economic principles.

---

## The Interaction Between Methods: When One Discipline Informs Another

The most sophisticated analysts do not simply pick one method and run it in isolation. They use the methods as a system of checks: each method's output constrains and informs the others.

### DCF Implied Multiples

One of the most useful exercises is to run a DCF and then ask: "What multiple is my DCF implying?" If your DCF for a consumer staples company gives an enterprise value of \$12 billion, and the company has \$1.2 billion of EBITDA, then your DCF implies a 10× EV/EBITDA multiple. Now check: are mature consumer staples companies actually trading at 10× EBITDA in the public market? If the sector trades at 12–14×, your DCF may be using a discount rate that is too high or a growth rate that is too conservative. If the sector trades at 8–9×, your DCF may be too optimistic.

This "DCF-implied multiple" cross-check is standard practice among experienced analysts and a reliable way to catch modeling errors or unreasonable assumptions.

### Multiple Implied Growth Rates

The reverse exercise is equally powerful. If a company's stock trades at 25× forward P/E and your required return on the sector is 10%, you can back into the implied growth rate:

```
P/E = (1 - b) / (r - g)
25 = (1 - 0.40) / (0.10 - g)
25 × (0.10 - g) = 0.60
0.10 - g = 0.024
g = 7.6%
```

The market is implying 7.6% long-run earnings growth. Now ask: is that reasonable for this company? A 7.6% long-run growth rate is high — it requires real (above-inflation) earnings expansion for perpetuity. If this is a cyclical manufacturer, that growth assumption is likely unsustainable; the stock is probably overvalued. If this is a compounding platform business with pricing power, 7.6% may be achievable; the valuation is reasonable.

This "multiple implied growth" approach directly links the market price to a testable economic prediction — a much more analytical way to assess valuation than simply saying "25× P/E seems expensive."

### Arbitrage Between Methods: When One Method Is Wrong

Sometimes the methods diverge dramatically — not because of modeling differences, but because one method is structurally wrong for the situation.

**Example: Spinoff valuation.** When a conglomerate spins off a subsidiary, the subsidiary initially trades based on index inclusion, liquidity, and investor familiarity — not fundamentals. Comparable companies multiples reflect the parent's sector, not the subsidiary's independent economics. In the weeks after a spinoff, the DCF is often the only reliable method because the comparables market is still calibrating. Investors who anchor to the early trading multiples and ignore the DCF frequently leave money on the table.

**Example: Merger arbitrage.** After an acquisition is announced at a premium (say, \$50 per share), the target stock might trade at \$48 — a \$2 spread that exists because the deal may not close. In this situation, the DCF is nearly irrelevant — the company is no longer an independent going concern. The relevant "valuation" is actually probability × deal price + (1 − probability) × unaffected stock price. This is a form of contingent claims valuation: the event-driven investor is pricing an option on deal completion, not on fundamental cash flows.

**Example: Distressed credit analysis.** When a company is burning cash and debt maturity approaches, equity analysts' DCF models break down because the going-concern assumption fails. Credit analysts take over, using recovery analysis (how much of par value can lenders recover in bankruptcy?), which is essentially asset-based liquidation valuation plus legal priority modeling. The equity becomes a call option that is deeply out-of-the-money; the real value debate is between different classes of creditors.

## Putting It All Together: A Step-by-Step Valuation Workflow

To make the spectrum practical, here is the workflow a thoughtful analyst uses for a new company:

**Step 1: Understand what you are valuing.**
Is this an operating business, a holding company, a distressed entity, or an asset pool? The answer determines which family of methods is primary.

**Step 2: Select the primary method based on sector and stage.**
Use the decision framework above. For a mature industrial company, DCF is primary. For a bank, P/B is primary. For a startup, VC method is primary.

**Step 3: Build the primary model carefully.**
For DCF: build a proper three-statement financial model (income statement, balance sheet, cash flow statement), derive free cash flow, compute WACC from first principles (cost of equity via CAPM + cost of debt from market yields), and run a terminal value. Spend time on sensitivity analysis — the answer is a range, not a point.

**Step 4: Run 2–3 cross-checks from other families.**
If DCF is primary, run EV/EBITDA comps as a first cross-check and P/E comps as a second. Compute the DCF-implied multiple and compare to observed trading multiples.

**Step 5: Explain the divergences.**
If DCF gives \$1,000M and comps give \$850M, what is the explanation? Is your DCF growth assumption too high? Are the comps structurally lower quality? Is the sector currently depressed by macro conditions? Write down the reason — this forces you to take an explicit position rather than blindly averaging.

**Step 6: Set a price range and a recommendation.**
Your final output is not a single number but a range (say, \$880M–\$980M) with a central estimate (\$930M) and a price-based recommendation: if the company can be acquired for \$850M, it is a buy with a 9% margin of safety relative to the central estimate. If it trades at \$1,050M, it is fully valued or slightly expensive.

**Step 7: Identify the key uncertainty.**
For every valuation, one or two assumptions drive most of the value. For a DCF, it is usually the terminal growth rate and discount rate. For a bank, it is the long-run ROE. For a startup, it is the exit multiple and probability of success. Flag these assumptions explicitly — because if they change, so does your conclusion.

*Valuation is not a search for the true price — it is a structured way of asking what a rational, informed buyer would pay for a given set of cash flows at a given level of risk. The methods differ because they embody different views of what "rational and informed" means: intrinsic value from fundamentals, market price from comparables, option value from flexibility, or liquidation value from physical assets. Mastering the spectrum means knowing which question to ask — and when the market is answering a different question than the one you are asking.*
