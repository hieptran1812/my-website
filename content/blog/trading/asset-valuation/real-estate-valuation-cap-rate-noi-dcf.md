---
title: "Real Estate Valuation: Cap Rates, NOI, and the Income Approach"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "Master real estate valuation from first principles: cap rates, NOI computation, GRM screening, full DCF with terminal values, leverage effects, and Vietnam rental yield context."
tags: ["real estate", "cap rate", "NOI", "DCF", "valuation", "income approach", "GRM", "Vietnam real estate", "cash-on-cash", "property valuation"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
series: "Asset Valuation: How to Price Stocks, Options & Companies"
seriesOrder: 26
---

> [!important]
> **TL;DR** — Real estate valuation is corporate valuation with different vocabulary: Net Operating Income (NOI) plays the role of EBIT, and the cap rate is the inverse of a price-to-earnings multiple.
>
> - **Income approach** (cap rate = NOI ÷ Value) is the primary method for investment properties; it directly links cash flow to price.
> - **NOI** = Gross rents − vacancy − operating expenses (no mortgage payments; NOI is pre-debt).
> - **Full DCF** models annual NOI over a hold period plus a terminal resale price anchored by a terminal cap rate.
> - **Leverage amplifies returns**: when the mortgage constant is below the cap rate, debt boosts cash-on-cash return on equity above the unlevered yield.
> - **The one number to remember**: cap rate = the yield you earn if you paid all cash and held the property flat — everything else is an adjustment to that baseline.

---

In the summer of 2021, apartment buildings across the United States were trading at cap rates of 3.5–4.5% — historically low numbers that implied buyers were willing to earn barely \$35,000 of annual income on a \$1,000,000 property before any financing costs. Critics called it a bubble. Buyers argued that with Treasury yields near zero, even a 4% cap rate represented a handsome spread over riskless alternatives. Both camps were engaging in valuation — translating a stream of rental income into a price — and the framework they were using, whether they knew it or not, was the income approach to real estate.

Real estate sits at an interesting junction in the asset valuation universe. Unlike a stock, a building produces a tangible, contractually specified cash flow every month: rent. Unlike a bond, that rent can grow over time (lease renewals, market rent increases), and the underlying asset has replacement value (land and construction costs set a floor). Unlike a private company, the "comparables" are physically observable in the same zip code. These features mean real estate has developed its own vocabulary — cap rate, NOI, GRM, debt service coverage ratio, loan-to-value — but underneath that vocabulary is the same logic we apply everywhere else in this series: discount future cash flows at a required return, or equivalently, divide current income by a rate that reflects riskiness and growth expectations.

This post builds real estate valuation from the ground up. We will define every term, work through the arithmetic in explicit dollars, explain why cap rates differ by property type and geography, show how to build a full multi-year DCF for a hold-and-sell scenario, examine how debt reshapes equity returns, and ground everything in real market data including Vietnam's residential rental market. By the end you will be able to pick up any commercial real estate offering memorandum and immediately understand what the seller is claiming and where you should push back.

![Three approaches to real estate valuation pipeline diagram](/imgs/blogs/real-estate-valuation-cap-rate-noi-dcf-1.png)

---

## Foundations: What Real Estate Value Means

### Property as an income-producing asset

When you buy a rental property, you are buying a machine that converts square footage into rent checks. The value of that machine depends on two things: how much rent it produces, and how much an investor requires to be compensated for the risk and illiquidity of owning it. These two forces interact through a deceptively simple equation:

```
Value = Income / Required Return
```

This is the exact same logic as the Gordon Growth Model for stocks (where Price = Dividend / (Ke − g)), or the P/E ratio (where Price = Earnings × Multiple), just applied to a physical asset. In the real estate world, "income" is Net Operating Income and "required return" is the capitalization rate (cap rate). So:

```
Property Value = NOI / Cap Rate
```

Or equivalently:

```
Cap Rate = NOI / Property Value
```

The cap rate tells you the yield you earn as an investor if you paid entirely in cash and the NOI stayed flat forever. It is simultaneously a measure of return (the higher the better for buyers) and a measure of price (low cap rate = high price relative to income, like a high P/E multiple is expensive for a stock).

### Why real estate needs its own vocabulary

A publicly traded stock has a live price, audited financials, and millions of analysts combing through every disclosure. A multi-family apartment building in Da Nang has none of that. There is no Bloomberg ticker, no quarterly earnings call, no SEC filing. What the market does have is:

1. **Visible comparables** — other buildings nearby that traded recently, from which price-per-square-foot or price-per-unit can be observed.
2. **Observable rental income** — leases are written contracts; rent rolls are auditable.
3. **Replacement cost** — anyone can estimate what it would cost to buy the land and build an identical structure from scratch.

These three sources of evidence correspond to the three standard approaches to real estate valuation: the **income approach**, the **sales comparable approach**, and the **cost approach**. Professional appraisers are required to consider all three and then reconcile them into a final opinion of value. Investors, being less formal, tend to anchor on the income approach for any asset large enough to have meaningful rental income, and use comps as a sanity check.

### A brief note on types of real estate

Real estate is not a single asset class. Before applying any valuation formula, you need to know what type of property you are analyzing, because each type has different income volatility, lease structures, tenant quality, and market cap rate norms:

- **Multifamily (apartments):** Month-to-month or 12-month leases; income resets frequently to market; recession-resistant (people always need housing); cap rates 4.5–6%.
- **Industrial (warehouses, logistics):** Long-term leases (5–15 years) with creditworthy tenants (Amazon, FedEx); low management intensity; cap rates 5–7%.
- **Office:** Long leases but tenant concentration risk; remote work has disrupted the sector structurally post-2020; cap rates 6–9% depending on location.
- **Retail:** Strip malls to regional malls; highly variable depending on anchor tenants and e-commerce competition; cap rates 6–8%.
- **Hospitality (hotels):** No leases — essentially daily income with high operational leverage; most volatile; cap rates 7–10%.

Understanding these distinctions is table stakes before you start computing anything.

---

## The Three Approaches to Real Estate Valuation

### 1. The income approach (primary for investment properties)

The income approach says: the value of a property equals the present value of the income it will generate over time, discounted at an appropriate rate. In its simplest form this is the direct capitalization method:

```
Value = NOI / Cap Rate
```

In its more sophisticated form it is a full discounted cash flow model projecting year-by-year NOI over a hold period, plus a terminal resale value. We will spend the bulk of this post on both variants.

The income approach is the dominant method for commercial and investment real estate precisely because the income stream is the point of the investment. A buyer of a 100-unit apartment building does not care how many square feet the building is; they care how much rent they will collect after paying operating expenses. The income approach formalizes that intuition.

### 2. The sales comparable approach

The sales comparable approach (or "market approach") says: the value of a property is what similar properties have recently sold for, adjusted for differences in size, location, age, condition, and lease terms. The key metrics are:

- **Price per square foot (PSF):** A 10,000-square-foot office building that sells for \$2,000,000 trades at \$200/SF. If comparable buildings trade at \$175–\$225/SF, that is a reasonable sanity check.
- **Price per unit (for multifamily):** A 50-unit apartment complex selling for \$5,000,000 implies \$100,000/door. Markets with tight housing supply often see \$200,000+/door for new product.
- **Gross Rent Multiplier (GRM):** We cover this in its own section below.

Sales comps are essential for residential appraisals (where the income approach does not apply to owner-occupied homes) and serve as a cross-check for any commercial valuation. The challenge is that no two properties are truly identical — adjustments for differences in lease expiry, tenant credit quality, condition, and submarket location can be subjective and contentious.

### 3. The cost approach

The cost approach says: no rational buyer will pay more than the cost to build an identical asset from scratch. Cost is:

```
Value (cost approach) = Land Value + Replacement Cost of Improvements − Depreciation
```

This approach is most relevant for:
- **New construction** (where cost sets the ceiling before lease-up)
- **Special-purpose properties** (schools, churches, hospitals) where there are no comparables
- **Insurance purposes** (how much would it cost to rebuild after a fire?)

The cost approach has a major weakness: it ignores what the market is willing to pay for the income. A warehouse in an oversupplied submarket might cost \$80/SF to build but be worth only \$60/SF at current cap rates if rents are too low to justify construction economics. Conversely, in a supply-constrained gateway city, existing buildings often sell at a significant premium to replacement cost because land is scarce and permitting takes years.

In practice, investors use the cost approach primarily as a floor — "would I rather buy this existing building at X, or build a new one?" — rather than as the primary valuation.

---

## Cap Rate Mechanics: The Inverse P/E of Real Estate

### The core relationship

The cap rate (capitalization rate) is the single most important number in commercial real estate. Let us be precise about what it is and what it is not.

**Definition:** The cap rate is the ratio of Net Operating Income to property value (or price).

```
Cap Rate = NOI / Property Value
```

Rearranged: `Property Value = NOI / Cap Rate`

Think of it this way: if you buy a \$1,000,000 property with a 6% cap rate, you are saying "I am willing to pay a price such that my annual NOI yield is 6%." At that price, NOI = \$60,000.

The cap rate is the **exact inverse of a P/E ratio** in equity markets. A P/E of 20× means you pay \$20 for every \$1 of earnings. A cap rate of 5% means you pay \$20 for every \$1 of NOI (because 1 ÷ 0.05 = 20). A P/E of 25× corresponds to a cap rate of 4%. Low cap rates = high prices = expensive assets, just as low interest rates imply high bond prices.

### The cap rate is NOT the investor's expected return

This is the most misunderstood aspect of cap rate mechanics, and we will return to it in the misconceptions section. The cap rate tells you your return only if:
1. You paid entirely in cash (no leverage)
2. NOI stays flat in perpetuity
3. You hold forever (no terminal value from a sale)

In practice, investors use debt (which changes equity returns), NOI grows over time (lease escalations, rent growth), and they sell after 5–10 years at an exit price determined by future market cap rates. The full DCF model captures all of this; the direct cap rate is a quick approximation.

### How the cap rate embeds growth and risk

The Gordon Growth Model for stocks says `P = D/(r − g)`, which can be rewritten as `r − g = D/P`. The analogous expression for real estate is:

```
Cap Rate ≈ Required Return − Expected NOI Growth Rate
```

So a 5% cap rate in a high-quality multifamily market with 3% expected annual rent growth implies a required return of roughly 8% (5% + 3%). A 8% cap rate on a hospitality asset with 0% expected long-run growth implies the same required return — the extra cap rate compensates for the lack of growth rather than for extra required return.

This decomposition explains why:
- **Gateway cities** (New York, San Francisco, Singapore) have low cap rates: real rent growth expectations are high, so buyers accept a lower current yield.
- **Secondary cities** often have higher cap rates: lower growth expectations, or higher risk, or both.
- **Retail and hospitality** have higher cap rates than industrial: structural headwinds mean growth expectations are muted or negative, so buyers demand a higher current yield to compensate.

#### Worked example: Backing out the implied growth expectation

A suburban Chicago office building is being marketed at a 7.5% cap rate. The investor's required return on this type of asset is 9.5% (reflecting office sector risk and leverage costs). What NOI growth rate is the cap rate implying?

```
Cap Rate ≈ Required Return − Growth Rate
7.5% ≈ 9.5% − g
g ≈ 2.0%
```

The market is pricing in roughly 2% annual NOI growth. The investor now asks: is that realistic given the tenant mix, lease expiry schedule, and local office demand? If they believe it is too optimistic, the asset is overpriced at this cap rate. If the portfolio has locked-in 3% annual rent escalations in long-term leases, the cap rate looks cheap relative to the embedded growth. This kind of triangulation between market price and implied assumptions is the heart of real estate investing.

---

## Computing NOI: From Gross Rents to the Bottom Line

NOI is the numerator in the cap rate equation, so getting it right is everything. The calculation seems simple but contains several important nuances that trip up beginners.

![NOI computation waterfall from gross rents to net operating income](/imgs/blogs/real-estate-valuation-cap-rate-noi-dcf-2.png)

### Step 1: Gross Potential Rent (GPR)

Gross Potential Rent is the total rent you would collect if every unit or square foot was leased at the current market rent with no vacancies. For an apartment building you multiply (number of units) × (market monthly rent) × 12. For a commercial building you multiply (leasable square feet) × (market annual rent per SF).

GPR is a hypothetical ceiling — the maximum possible income. No property actually achieves it.

### Step 2: Subtract Vacancy and Credit Loss

Real properties do not stay 100% occupied. Tenants move out, units sit empty between leases, some tenants pay late or default. The standard adjustment is:

```
Vacancy & Credit Loss (%) = (Vacant Units + Uncollected Rents) / GPR
```

For a stabilized multifamily asset in a healthy market, vacancy allowance is typically 5–7%. For a value-add apartment with deferred maintenance, budget 10–15% during the renovation period. For office in a struggling submarket, 20–30% vacancy is realistic. The industry average vacancy assumption (as opposed to the actual current vacancy, which can be temporarily high or low) is called the **stabilized vacancy**.

**Effective Gross Income (EGI) = GPR − Vacancy & Credit Loss**

Sometimes analysts add miscellaneous income here (laundry machines, parking revenue, storage fees, pet deposits) to arrive at a more complete EGI figure.

### Step 3: Subtract Operating Expenses

Operating expenses are all the costs of running the property. Critically, they do **not** include:
- Mortgage principal or interest payments (those belong to debt service, below NOI)
- Income taxes (NOI is pre-tax)
- Depreciation (a non-cash accounting item)
- Capital expenditures for major improvements (these are above-the-line in the full DCF)

What operating expenses do include:

| Category | Typical % of EGI |
|---|---|
| Property taxes | 15–25% |
| Insurance | 3–6% |
| Property management fee | 6–10% of collected rents |
| Maintenance & repairs | 5–10% |
| Utilities (landlord-paid) | 3–8% |
| Administrative / legal | 1–3% |
| **Total OpEx** | **35–55%** |

The **expense ratio** (OpEx ÷ EGI) for a well-run multifamily asset is typically 35–45%. For a full-service hotel it can be 60–70%.

**Net Operating Income = EGI − Operating Expenses**

#### Worked example: Computing NOI for a 20-unit apartment building

You are analyzing a 20-unit apartment building. All units rent for \$1,500/month at current market rates. Vacancy in the submarket runs 6%. Annual operating expenses (taxes + insurance + management + maintenance + utilities) total \$84,000.

```
GPR:        20 units × $1,500/month × 12 months = $360,000
Vacancy:    $360,000 × 6% = −$21,600
EGI:        $360,000 − $21,600 = $338,400
OpEx:       −$84,000
NOI:        $338,400 − $84,000 = $254,400
```

Expense ratio: \$84,000 ÷ \$338,400 = 24.8% (quite low — typical for a fully stabilized, owner-managed small building where the property tax assessment is modest).

Now apply a market cap rate. If comparable 20-unit buildings in this market trade at 6.0% cap rates:

```
Implied Value = NOI / Cap Rate = $254,400 / 0.060 = $4,240,000
Price per unit = $4,240,000 / 20 = $212,000/door
```

A seller asking \$4,500,000 is pricing the asset at a 5.65% cap rate (\$254,400 ÷ \$4,500,000), which is tighter than market comps. The buyer's negotiating argument: "Your cap rate is below market; you need to either reduce the price or demonstrate why this building commands a premium."

### The replacement reserve trap

One of the most common errors in real estate underwriting is omitting a **capital reserve** from operating expenses. Roofs need replacement every 20 years. HVAC systems wear out. Parking lots crack. These are not monthly expenses, but they are real economic costs of ownership. Professional underwriters budget a reserve for replacement — often \$150–\$300 per unit per year for multifamily — and include it in the operating expense line. Sellers often "forget" to include it in their NOI presentations, making the NOI look better and justifying a higher asking price. Always add it back when assessing seller-provided financials.

---

## Gross Rent Multiplier: Quick-Screen Before the Full Model

Before you build a full NOI waterfall and cap rate analysis, experienced investors often run a quick "back of the envelope" screen using the **Gross Rent Multiplier (GRM)**:

```
GRM = Property Price / Gross Annual Rents (GPR)
```

A building selling for \$4,200,000 with GPR of \$360,000 has a GRM of 11.7×. If the market trades at 10–12× GRM, that is in the ballpark. If it is trading at 15× GRM, the price needs to be justified by below-market rents that have upside, or the seller is fishing.

**Why GRM is useful:** It takes 30 seconds and requires only the asking price and the rent roll. No need to source expense data (which sellers may manipulate). It is an especially common screen in residential investment markets.

**Why GRM is limited:** It ignores operating expenses entirely. A property with a lower GRM but higher taxes and management costs can have a worse NOI than a higher-GRM property with lower expenses. Two buildings with the same GRM in different cities can have entirely different cap rates if local expense structures differ. Always use GRM as a first filter, not a final answer.

#### Worked example: GRM as a screening tool

You are screening four small apartment buildings advertised in the same city:

| Property | Price | Annual Rents | GRM |
|---|---|---|---|
| A | \$1,800,000 | \$180,000 | 10.0× |
| B | \$2,100,000 | \$180,000 | 11.7× |
| C | \$2,400,000 | \$192,000 | 12.5× |
| D | \$1,650,000 | \$144,000 | 11.5× |

At first glance, Property A looks cheapest (lowest GRM). But if you dig into the expense data: Property A is in a high-tax jurisdiction where property taxes eat \$40,000/year vs \$18,000 for Property B. After a full NOI build-out, Property A may actually have a worse cap rate than Property B despite the lower GRM. GRM got you to the table; the full model tells you what to bid.

---

## Full DCF for Multi-Period Holds

Direct capitalization (Value = NOI / Cap Rate) is elegant but static — it assumes NOI stays flat forever. Real investments are not static. Leases expire, rents grow (or fall), capital expenditures occur, and eventually you sell. The full **Discounted Cash Flow (DCF) model** captures this dynamic reality.

The structure of a real estate DCF is:

1. **Project annual NOI** for each year of the hold period (typically 5–10 years)
2. **Subtract capital expenditures** (CapEx) that are not captured in operating expenses — roof replacement, major renovations, tenant improvement allowances for commercial space
3. **Estimate a terminal value** (resale price) at the end of the hold period
4. **Discount all cash flows** to present value at the investor's required return
5. **Compare PV of cash flows to the asking price** — if PV > asking price, the IRR exceeds your hurdle rate

![Multi-period hold DCF cash flow timeline](/imgs/blogs/real-estate-valuation-cap-rate-noi-dcf-7.png)

### Projecting annual NOI growth

NOI growth comes from two sources:
- **Rent growth:** Leases typically include annual escalations of 2–3% (for CPI or fixed step-ups). Market rents can grow faster in supply-constrained markets.
- **Expense management:** Operational improvements (better property management, energy-efficiency upgrades) can reduce expenses over time.

For a multifamily asset, a conservative underwriting assumption is 2–3% annual NOI growth matching expected CPI. For a value-add play (buying a below-market-rent building and raising rents to market), NOI might grow 10–20% in the first two years and then normalize.

### Terminal cap rate: the most sensitive assumption

At the end of the hold period, you model a sale. The sale price is typically computed as:

```
Terminal Value = NOI in Year (Hold + 1) / Terminal Cap Rate
```

If you hold for 10 years and Year 11 NOI is projected to be \$140,000, and the market terminal cap rate is 6.0%:

```
Terminal Value = $140,000 / 0.060 = $2,333,333
```

The terminal cap rate is the **most sensitive assumption in any real estate DCF**. A difference of 50 basis points in the terminal cap rate can change the implied property value by 8–10%. Conservative underwriting uses an exit cap rate **higher** (worse, more conservative) than the going-in cap rate to reflect:
- The building is 10 years older at exit
- Cap rate market conditions may deteriorate
- The new buyer will want to be compensated for uncertainty

A disciplined rule of thumb: exit cap rate = entry cap rate + 25–50 bps. Never underwrite an exit cap rate tighter than your entry cap rate unless you have a specific thesis for why the market will reprice.

![DCF sensitivity property value vs terminal cap rate and discount rate](/imgs/blogs/real-estate-valuation-cap-rate-noi-dcf-5.png)

The chart above shows how sensitive property value is to these two assumptions. For a baseline NOI of \$110,000/year with 2.5% annual growth over 10 years: moving from a 4.5% terminal cap rate to an 8.5% terminal cap rate cuts the indicated value nearly in half. This is why sellers always use the most optimistic exit cap rate in their offering memoranda, and why sophisticated buyers stress-test scenarios.

### The discount rate in real estate DCF

The discount rate in a real estate DCF is the investor's **required total return** — what they need to earn on equity to justify the investment risk. This is analogous to the cost of equity in the corporate valuation framework, and it can be estimated using:

- **CAPM:** risk-free rate + Beta × equity risk premium. Real estate sector WACC from Damodaran data is approximately 7.8%, consistent with SECTOR_WACC["Real Estate"] in our data file.
- **Market-derived hurdle rates:** Institutional real estate investors typically target 8–12% IRR for core/core-plus assets, 12–18% for value-add, and 18%+ for opportunistic deals.
- **Spread over cap rate:** The discount rate should exceed the cap rate to reflect growth and risk. Historically, the spread between IRR and going-in cap rate is 200–400 bps.

#### Worked example: Full 10-year DCF for a multifamily property

**Inputs:**
- Purchase price: \$2,000,000
- Year 1 NOI: \$110,000
- Annual NOI growth: 2.5%
- Hold period: 10 years
- Terminal cap rate: 6.0% applied to Year 11 NOI
- Discount rate: 9.0% (investor's hurdle rate for this asset type and market)
- No major CapEx (stabilized property)

**Step 1: Project NOI for years 1–10**

| Year | NOI |
|---|---|
| 1 | \$110,000 |
| 2 | \$112,750 |
| 3 | \$115,569 |
| 4 | \$118,458 |
| 5 | \$121,419 |
| 6 | \$124,455 |
| 7 | \$127,566 |
| 8 | \$130,755 |
| 9 | \$134,024 |
| 10 | \$137,375 |

**Step 2: Terminal value**

Year 11 NOI = \$137,375 × 1.025 = \$140,809

Terminal Value = \$140,809 / 0.060 = \$2,346,817

**Step 3: Discount each year's cash flow at 9%**

PV of NOI streams (using the formula PV = CF / (1+r)^t):

- Year 1: \$110,000 / 1.09^1 = \$100,917
- Year 2: \$112,750 / 1.09^2 = \$94,917
- Year 3: \$115,569 / 1.09^3 = \$89,271
- Year 4: \$118,458 / 1.09^4 = \$83,958
- Year 5: \$121,419 / 1.09^5 = \$78,954
- Year 6: \$124,455 / 1.09^6 = \$74,241
- Year 7: \$127,566 / 1.09^7 = \$69,801
- Year 8: \$130,755 / 1.09^8 = \$65,617
- Year 9: \$134,024 / 1.09^9 = \$61,675
- Year 10 NOI: \$137,375 / 1.09^10 = \$57,959
- PV of Terminal Value: \$2,346,817 / 1.09^10 = \$990,363

**Total PV of cash flows = Sum of PV NOI + PV Terminal Value**

PV of NOIs = \$777,310
PV of Terminal Value = \$990,363
**Total Present Value = \$1,767,673**

**Step 4: Compare to asking price**

The property is listed at \$2,000,000. At a 9% discount rate, the DCF produces a value of approximately \$1,768,000. This means the property — at the asking price — yields an IRR just below 9% if all assumptions hold.

If the investor's hurdle is 9%, the asking price is slightly above fair value at their hurdle rate. They might negotiate to \$1,800,000–\$1,850,000 to secure their required return, or accept that this is a core asset where slightly sub-hurdle pricing is offset by lower risk.

*The intuition: the DCF is simply asking, "what is the most I should pay today such that all future NOI and the terminal sale, discounted at my required return, equals that price?"*

---

## Why Cap Rates Differ: Property Type, Location, and Risk

![Cap rates by property type US 2024 bar chart](/imgs/blogs/real-estate-valuation-cap-rate-noi-dcf-3.png)

The chart above shows the approximate spread of cap rates across property types in the US market as of 2024. Understanding why these differences exist reveals the underlying risk and growth pricing embedded in each sector.

### Property type differences

**Multifamily (4.5–5.5%):** The lowest cap rates reflect the combination of (a) near-certain demand (housing is a necessity), (b) strong and persistent rent growth especially in supply-constrained markets, and (c) deep institutional investor appetite. The monthly lease reset is actually an advantage in inflationary environments — rents adjust faster than long-term commercial leases.

**Industrial/Logistics (5.0–6.5%):** Cap rates compressed dramatically during 2020–2023 as e-commerce growth created explosive demand for last-mile distribution space. Industrial now competes with multifamily as the most aggressively priced sector. Long-term net leases with CPI escalations and credit tenants justify tight pricing.

**Retail (6.0–7.5% for strip centers, 7.0–9%+ for malls):** The sector is bifurcated. Grocery-anchored neighborhood centers with essential-service tenants have held value; enclosed regional malls face structural demand destruction from e-commerce. Cap rates reflect whether investors believe the specific asset's rental stream is durable.

**Office (6.5–9%+):** The post-pandemic work-from-home structural shift has repriced office broadly. Central business district trophy buildings in gateway cities still command 6–7% cap rates because of scarcity; suburban office parks can trade at 9–12% or refuse to trade at any price.

**Hospitality (7.5–10%+):** Hotels operate more like businesses than traditional real estate. Revenue per available room (RevPAR) swings wildly with macro cycles. The high cap rates reflect operating risk that goes well beyond a standard rental stream.

### Location differences within a property type

Even within a sector, cap rates vary enormously by geography. The drivers:

1. **Supply constraints:** San Francisco or Manhattan can barely add new apartments due to zoning; high barriers to new supply compress cap rates because future competition is limited.
2. **Population and income growth:** Markets with strong job growth (Austin, Dallas, Nashville in the US; Ho Chi Minh City in Vietnam) attract capital, compressing cap rates.
3. **Liquidity premium:** More liquid markets (more buyers and sellers, more transaction volume) command tighter cap rates because investors accept lower yields when they know they can exit.
4. **Interest rate environment:** Cap rates correlate loosely with long-term interest rates. When Treasury yields fell to near zero in 2020–2021, real estate cap rates followed. As rates rose in 2022–2023, cap rates expanded — repricing the present value of future income streams.

#### Worked example: Geographic cap rate spread

A 100-unit apartment complex in Manhattan, New York is marketed at a 3.8% cap rate. An identical building (same unit mix, same age) in Cleveland, Ohio would sell at a 7.5% cap rate. Both have Year 1 NOI of \$500,000.

| | Manhattan | Cleveland |
|---|---|---|
| NOI | \$500,000 | \$500,000 |
| Cap Rate | 3.8% | 7.5% |
| Implied Value | \$13,157,895 | \$6,666,667 |
| Price per unit | \$131,579 | \$66,667 |

The 3.7 percentage point cap rate gap represents the market's collective judgment that Manhattan rent growth expectations + supply scarcity + liquidity premium justify a price that is roughly 2× that of the Cleveland building. The buyer in Manhattan is not getting less yield out of carelessness — they are betting on rent growth catching up to their price over a hold period, and on capital appreciation (exit price) that Cleveland cannot match.

This is exactly the stock market analogue: a growth stock trades at a high P/E not because investors are irrational but because they expect earnings to grow faster than average.

---

## Leverage, Debt Service, and Cash-on-Cash Return

Real estate is almost always purchased with borrowed money. Mortgage debt amplifies both returns and risks in a precise, calculable way. Understanding this arithmetic is essential for any investor and for any valuation model.

![Leverage effect on cash-on-cash return vs unlevered cap rate](/imgs/blogs/real-estate-valuation-cap-rate-noi-dcf-4.png)

### The mechanics of positive leverage

**Positive leverage** exists when the cap rate exceeds the mortgage constant (annual debt service ÷ loan amount). When this condition holds, adding debt increases the equity investor's cash-on-cash return.

**Mortgage constant:** The total annual debt service (principal + interest) per dollar of loan. For a 30-year amortizing loan at 5.5% interest, the mortgage constant is approximately 6.82% (computed from standard amortization tables).

**Cap rate:** Suppose the property has a 6.5% cap rate. Since 6.5% > 5.5% interest rate (and also > 6.82% mortgage constant only barely — this is tight), debt still generates positive leverage if the interest rate is the right reference.

Actually, the correct comparison for positive leverage is:

```
Positive leverage condition: Cap Rate > Mortgage Constant
```

If the cap rate is 7.5% and the mortgage constant is 6.5%, every dollar of debt earns 7.5% at the property level but costs only 6.5% in debt service — the surplus accrues to equity.

### The key metrics

**Loan-to-Value (LTV):** The percentage of the purchase price financed by debt. Typical commercial real estate loans: 60–75% LTV. Higher LTV = more leverage = higher equity returns when things go well, more severe losses when they do not.

**Debt Service Coverage Ratio (DSCR):** NOI ÷ Annual Debt Service. Lenders typically require DSCR ≥ 1.20–1.25×. If NOI is \$110,000 and annual debt service is \$85,000, DSCR = 1.29× — acceptable. If DSCR falls below 1.0×, the property cannot cover its own mortgage from operations — a distress signal.

**Cash-on-Cash Return (CoC):** The annual pre-tax cash flow to equity investors ÷ total equity invested. This is what an equity investor actually receives in cash each year, expressed as a percentage of their invested capital.

```
Cash-on-Cash = (NOI − Annual Debt Service) / Equity Invested
```

#### Worked example: The lever of debt

Purchase price: \$2,000,000. NOI: \$120,000. Cap rate: 6.0%.

**Scenario A — All cash, no leverage:**
- Equity invested: \$2,000,000
- Annual cash flow: NOI = \$120,000
- Cash-on-Cash = \$120,000 / \$2,000,000 = **6.0%** (equals the cap rate, as expected)

**Scenario B — 65% LTV mortgage at 5.8% interest, 25-year amortization:**
- Loan amount: \$2,000,000 × 65% = \$1,300,000
- Equity invested: \$2,000,000 × 35% = \$700,000
- Annual debt service (principal + interest): approximately \$99,400 (mortgage constant ~7.65% — wait, let me recalculate for 25-year amortization at 5.8%)

For a \$1,300,000 loan at 5.8% over 25 years, monthly payment ≈ \$8,189, annual ≈ \$98,268.

- Pre-tax annual cash flow to equity: \$120,000 NOI − \$98,268 debt service = \$21,732
- Cash-on-Cash = \$21,732 / \$700,000 = **3.1%**

In this case, leverage *reduces* cash-on-cash return below the cap rate because the mortgage constant (~7.6%) exceeds the cap rate (6.0%) — this is **negative leverage**. The interest rate environment of 2022–2024, where mortgage rates jumped well above prevailing cap rates, created widespread negative leverage conditions across commercial real estate.

**Scenario C — More favorable debt: 6.5% cap rate property, 5.0% interest, 30-year amortization:**
- Cap rate: 6.5%; NOI: \$130,000 on a \$2,000,000 property
- 65% LTV: \$1,300,000 loan at 5.0%/30yr → annual debt service ≈ \$83,700 (mortgage constant ~6.44%)
- Pre-tax equity cash flow: \$130,000 − \$83,700 = \$46,300
- Equity invested: \$700,000
- Cash-on-Cash = \$46,300 / \$700,000 = **6.6%**

Now leverage works — CoC (6.6%) exceeds the cap rate (6.5%) because the mortgage constant (6.44%) is below the cap rate (6.5%). Positive leverage, as the theory predicts.

*The intuition: debt is profitable to the equity investor only when the property earns more than the cost of the debt. When cap rates compress and interest rates rise simultaneously, positive leverage disappears.*

---

## Vietnam Real Estate: Rental Yields and Market Context

Vietnam's real estate market operates under different institutional constraints than US or European markets, but the same valuation logic applies. Understanding the local specifics reveals why yields are structured the way they are.

![Residential rental yields Asia-Pacific city comparison Q4 2024](/imgs/blogs/real-estate-valuation-cap-rate-noi-dcf-6.png)

### Gross vs net yields in Vietnam

Vietnam residential properties — particularly condominiums in Ho Chi Minh City (HCMC) and Hanoi — generate **gross rental yields of 3.5–5%** in prime areas, and **net yields of 2.5–3.5%** after accounting for:

- **Management fees:** If using a professional property manager, expect 8–12% of gross rent.
- **Maintenance and repairs:** Older buildings require more; newer high-rises have HOA fees (phí quản lý) of 1,000,000–3,000,000 VND per month.
- **Vacancy:** Urban Vietnamese rental markets have moderate liquidity, but high-end units can sit vacant for 2–4 months between tenants.
- **Withholding tax:** Rental income in Vietnam is subject to personal income tax at a flat rate of 5% (revenue tax) for landlords earning above 100 million VND/year.

At a net yield of 3.0%, the implied cap rate multiple is approximately 33×. With Vietnam's 10-year government bond yielding around 2.5–3.0% in 2024, the spread over the risk-free rate is thin — justifying purchase only if you have a thesis for capital appreciation.

### HCMC vs Hanoi market dynamics

The two primary markets behave differently:

**Ho Chi Minh City (HCMC / Saigon):**
- More speculative and investor-driven; prices have appreciated faster.
- District 2 (Thu Duc City / Thao Dien area) is favored by expatriates and high-income locals; premium rents (\$1,500–\$3,000/month for a 2BR near international schools).
- District 9 / Thu Duc City is the tech/industrial corridor with growing middle-class demand.
- Foreign buyers face restrictions: eligible to own condominiums (not land) under a 50-year renewable lease through the Housing Law.

**Hanoi:**
- More government-employee and institutional demand; less speculative.
- West Lake (Tay Ho) area commands premium rents from diplomats and expats.
- Cau Giay / My Dinh corridor near major universities and the National Convention Center.
- Yields slightly lower than HCMC in prime areas due to lower absolute rent levels relative to property prices.

### The cap rate lens on Vietnam real estate

Applying cap rate logic directly:

| Scenario | HCMC District 2 | Hanoi Tay Ho |
|---|---|---|
| Condo price | 5,000,000,000 VND (≈\$200,000) | 4,500,000,000 VND (≈\$180,000) |
| Monthly rent | 25,000,000 VND (≈\$1,000) | 22,000,000 VND (≈\$880) |
| Gross annual rent | 300,000,000 VND | 264,000,000 VND |
| Gross yield | 6.0% | 5.9% |
| Net yield (after 30% opex) | ~4.2% | ~4.1% |

At 4.2% net yield, an all-cash investor earns roughly the same as Vietnam's 10-year bond — but with the optionality of capital appreciation if property prices continue rising. The bet on Vietnamese real estate is fundamentally a bet on urbanization, income growth, and continued supply constraints in central urban areas, not on the current income yield.

### HCMC vs Hanoi: Rental Yield Divergence and SBV Rate Effects

The two markets have converged in price but diverged in yield, and the State Bank of Vietnam (SBV) plays a direct role in how that gap shifts over time.

**Rental yield comparison (mid-2025 data):**

| Metric | HCMC (District 2) | Hanoi (Tay Ho) |
|---|---|---|
| Average condo price (2BR, 80m²) | 8,000,000,000 VND | 6,200,000,000 VND |
| Monthly market rent | 28,000,000 VND | 22,000,000 VND |
| Gross annual rent | 336,000,000 VND | 264,000,000 VND |
| Gross yield | **4.2%** | **4.3%** |
| Net yield (after 30% opex + tax) | **~2.9%** | **~3.0%** |

Despite HCMC's higher absolute price and prestige, Hanoi's net yield is fractionally higher because the price premium in HCMC has compressed faster than rents have risen. This is the same dynamic as comparing Manhattan to Brooklyn in the US: the higher-prestige market prices in more capital appreciation, leaving a thinner current yield.

**How SBV rate cycles move cap rates:**

Vietnam's mortgage rates for retail borrowers track the SBV base rate with a spread of ~3–4%. When the SBV cut its base rate from 6.0% to 4.5% in 2023 (a 150-bps easing cycle to support growth post-Covid), mortgage rates dropped from roughly 10–11% to 8–9%. This had two effects:

1. **Demand impulse:** Lower mortgage costs brought marginal buyers back into the market, increasing transaction volume and supporting prices — which compressed gross yields slightly further (prices rose, rents did not).
2. **Investor cap rate benchmarking:** With the 5-year Vietnam government bond at ~2.8% in early 2024, a net rental yield of 3.0% offered only ~20 bps of spread over the risk-free rate. This made yield-seeking investors reluctant to buy at current prices, capping further price appreciation.

The SBV policy rate is therefore a direct input to the cap rate equation in Vietnam, just as the Federal Reserve funds rate anchors cap rates in the US market — the channel runs through both mortgage availability (credit channel) and the risk-free rate benchmark (discount rate channel).

#### Worked example: Cash-on-cash return vs cap rate with leverage in a Vietnamese context

A Hanoi investor buys a 2BR condo in Tay Ho for 6,200,000,000 VND (\$248,000 at 25,000 VND/USD) with a 50% LTV mortgage at the prevailing rate of 9% per annum over 20 years.

**All-cash scenario (unlevered):**
- Net annual NOI: 264,000,000 VND × (1 − 0.30 opex) = 184,800,000 VND
- Cash-on-cash = 184,800,000 ÷ 6,200,000,000 = **3.0%** (= the net cap rate)

**50% LTV mortgage scenario:**
- Loan amount: 3,100,000,000 VND
- Equity invested: 3,100,000,000 VND
- Annual debt service (9%/20yr mortgage constant ≈ 10.8%): 3,100,000,000 × 0.108 = 334,800,000 VND
- Pre-tax equity cash flow: 184,800,000 − 334,800,000 = **−150,000,000 VND** (negative)

This illustrates classic **negative leverage**: the mortgage constant (10.8%) exceeds the net cap rate (3.0%), so debt service consumes more than the property earns, resulting in an annual cash deficit. The Vietnamese investor is effectively making a capital appreciation bet, not an income bet. The investment only makes sense if the property value grows at a rate that compensates for annual cash shortfall — roughly 5–7%/year in a bull market, which HCMC and Hanoi delivered from 2015–2022 but not from 2022–2024.

This is why Vietnamese real estate investors routinely speak of "accepting negative carry" — a term borrowed from fixed-income markets that precisely captures the cash-flow-negative, capital-gain-dependent nature of current Vietnamese urban property pricing.

### Key risks in Vietnam real estate valuation

1. **Legal title complexity:** Many properties trade on "pink book" (residential land use rights certificate). Verifying clean title and understanding 50-year lease renewal risks for foreigners is essential.
2. **Liquidity risk:** The Vietnamese secondary market is less liquid than markets in Singapore or Bangkok. Transaction costs (transfer tax ~2%, agent fees ~2–3%) are higher. Assume a 3–6 month exit timeline.
3. **Currency risk:** Rents denominated in VND face devaluation risk for USD-based investors (VND has depreciated ~2–3%/year against USD historically).
4. **Regulatory changes:** The 2023–2024 real estate law revisions and the central bank's credit tightening significantly affected developer cash flows and secondary market prices. Regulatory risk is real.

---

## REITs as an Investable Proxy for Real Estate

Not every investor wants to own physical property directly. Real Estate Investment Trusts (REITs) offer the economic exposure of real estate with the liquidity of a stock. From a valuation perspective, REITs are interesting because they bridge two worlds:

- They are valued in public markets using stock-market multiples (Price/FFO, where FFO = Funds From Operations, the REIT equivalent of earnings)
- Their underlying assets are valued using cap rates and NOI

This dual-world existence means REIT prices can deviate significantly from the Net Asset Value (NAV) of their underlying properties. When the stock market sells off but private property transactions remain firm, REITs trade at a discount to NAV. This arbitrage is not always easily exploitable (private properties cannot be bought/sold in milliseconds), but it is real and tracked by real estate analysts.

![Risk-return scatter REITs vs asset classes 2000 to 2024](/imgs/blogs/real-estate-valuation-cap-rate-noi-dcf-8.png)

From the data in our asset valuation dataset (JP Morgan Guide to the Markets Q1 2025), REITs delivered an average annual return of 9.1% with a standard deviation of 19.3% over 2000–2024. This places REITs close to equities in terms of return, but with volatility slightly above US stocks (15.2% for S&P 500 vs 19.3% for REITs). The volatility premium reflects REITs' leverage (most REITs operate at 35–50% LTV) and their sensitivity to interest rates (rising rates increase cap rates, which depresses REIT NAV).

The real estate sector WACC from Damodaran (approximately 7.8% as of January 2025) is consistent with this observed return history — the market has priced real estate to deliver returns consistent with its risk characteristics.

---

## Common Misconceptions

### Misconception 1: "High cap rate = bad investment; low cap rate = good investment"

This is exactly backwards. A high cap rate means a high current yield — which is good for cash flow but may reflect higher risk, lower growth, or a less desirable market. A low cap rate means a low current yield but typically signals higher quality, better growth prospects, or a stronger market. Neither high nor low is inherently better. The question is whether the cap rate correctly prices the risk and growth embedded in the asset. A 7.5% cap rate on a sound, well-located industrial asset in a growing market might be a screaming buy. A 4.5% cap rate on a struggling retail strip center is probably a disaster regardless of how "low" the cap rate looks.

*The number: in the 2021 US apartment frenzy, assets traded at 3.5% cap rates with 4.5% mortgage rates — instant negative leverage. Not all low-cap-rate deals work out.*

### Misconception 2: "The cap rate is my return"

Only if you paid all cash and NOI stays flat forever. In practice, most investors use leverage, hold for a finite period, sell at a terminal value, and experience NOI growth or decline. Your actual total return is the IRR of all cash flows including the exit. In value-add deals, the going-in cap rate may be 5% but the IRR could be 14% if you successfully increase NOI from \$80,000 to \$120,000 over 3 years and sell at the same 5% cap rate — collecting \$2,400,000 on an asset you bought for \$1,600,000.

*The number: in the above scenario, IRR ≈ 14% on a 3-year hold — nearly 3× the going-in cap rate.*

### Misconception 3: "Higher LTV always means higher returns"

More leverage amplifies both upside and downside. If the property value falls 20% and you are at 80% LTV, your equity is wiped out entirely (\$1,000,000 loan on a property now worth \$800,000 = negative equity). Institutions cap LTV at 65–75% for core assets specifically to maintain a safety buffer against market downturns. The highest returns in real estate come from value creation (operational improvement, lease-up, repositioning), not from maximum leverage.

*The number: the 2008 financial crisis was partly a story of 90–100% LTV residential mortgages — a 10% price decline eliminated all equity.*

### Misconception 4: "NOI from the seller's brochure can be trusted"

Sellers present NOI in the most favorable light. Common manipulations: (a) using current (perhaps temporarily low) vacancy rather than stabilized vacancy, (b) omitting capital reserves, (c) including non-recurring income (a one-time lease termination fee), (d) presenting in-place rents well below market to suggest "upside" that is actually already priced into the asking cap rate. Always rebuild NOI from first principles: get the rent roll, verify occupancy, source expense data from property tax records and insurance quotes, and add a capital reserve independently.

*The number: in institutional acquisitions, buyers routinely find that seller-stated NOI is 5–15% higher than the underwritten NOI after due diligence.*

### Misconception 5: "A building in a good location is always a good investment"

Location quality is necessary but not sufficient — the price paid relative to the income it generates is what determines return. A trophy apartment in a prime district trading at a 2.5% gross yield and 1.5% net yield requires capital appreciation of 6–7%/year just to match an investor's 8% total return hurdle. If appreciation moderates (a realistic scenario once urbanization matures), the investor locked in permanent underperformance. The "good location" thesis must be translated into an explicit NOI growth and cap rate exit assumption; if those numbers do not support the purchase price, location alone does not save the investment.

*The number: Tokyo's prime residential market delivered near-zero capital appreciation from 1991 to 2012 — two decades during which investors relying on "great location" in one of the world's premier cities still lost purchasing power.*

### Misconception 6: "Real estate always appreciates"

Real estate prices can fall and stay down for extended periods. Japan saw commercial real estate values fall 80% from 1991 peak to mid-2000s and never fully recover in many markets. US commercial real estate prices fell 35–40% in the 2008–2009 financial crisis. Office real estate in San Francisco lost 50%+ of its value from 2019 to 2023. The "real estate only goes up" narrative confuses long-run trends (population growth, inflation, urbanization) with the path of prices, which can deviate severely from trend for decades.

---

## How It Shows Up in Real Markets

### The 2022–2024 cap rate expansion cycle

The Federal Reserve raised the federal funds rate from 0.25% to 5.50% between March 2022 and August 2023 — the steepest and fastest rate hiking cycle in 40 years. US Treasury 10-year yields moved from 1.5% to nearly 5.0%. What happened to real estate?

Cap rates expanded significantly — rising 100–200 bps across property types — but with a lag. Private real estate transactions are slower than stock market repricing; the full cap rate reset took 18–24 months. The clearest evidence: the National Council of Real Estate Investment Fiduciaries (NCREIF) Property Index — which tracks unleveraged returns on institutional real estate — showed negative total returns in 2022–2023 for the first time since the financial crisis.

The repricing math is precise: if you paid \$2,000,000 for a property at a 4.0% cap rate (NOI = \$80,000), and the market reprices to 5.5% cap rates while NOI stays flat, your property is now worth \$80,000 / 0.055 = \$1,454,545 — a \$545,455 (27%) decline in value from interest rate movement alone, even if nothing changed about the physical property or the rental income.

This dynamic explains why REITs (which reprice in real time) fell 25–30% in 2022 while private real estate transaction volumes simply froze. Owners refused to sell at lower prices; buyers demanded the new market cap rates. The standoff was only resolved gradually as sellers ran out of options (refinancing needs, partner redemptions, fund life cycle endings) and accepted market pricing.

### Value-add underwriting in practice

A common institutional real estate strategy: buy a building with below-market rents, spend capital to renovate units, re-lease at market rents, and sell 3 years later at the same cap rate (but now on higher NOI). Example from a 2024 secondary market deal:

- **Acquisition:** 80-unit apartment building, HCMC outskirts, purchase price \$8,000,000 (all cash equivalent). Current NOI: \$360,000 at 60% occupancy. Going-in cap rate: 4.5%.
- **Business plan:** Spend \$1,200,000 renovating 60 units, raising rents from below-market 4,000,000 VND/month to 7,000,000 VND/month. Expected stabilized occupancy: 90% after 24 months.
- **Stabilized NOI:** 80 units × 7,000,000 VND × 0.90 occupancy × 12 months = 6,048,000,000 VND/year ≈ \$240,000 (at 25,200 VND/USD exchange rate used for illustration)

Wait — this illustrates a key point: in Vietnamese VND terms, the NOI growth is clear; in USD terms, currency moves affect the dollar-denominated return. Let us run the example in VND:

- Stabilized NOI: 6,048,000,000 VND (from 3,456,000,000 VND current)
- Total cost basis: 8,000,000,000 + 1,200,000,000 = 9,200,000,000 VND
- Exit cap rate: 5.0% (slight expansion from going-in due to market conditions and older building age)
- Exit value: 6,048,000,000 / 0.050 = 120,960,000,000 VND
- Profit: 120,960,000,000 − 9,200,000,000 = 111,760,000,000 VND

The simple profit math on a \$8M equivalent investment turns into an almost 13× return on the renovation capital — but only if the rent increase is achievable, the timeline holds, and the exit cap rate does not expand further. This is the leverage of operational improvement in real estate: the NOI multiplier works on the entire building value.

---

## Further Reading & Cross-Links

The real estate income approach is fundamentally the same DCF framework applied to corporate assets. If you want to go deeper on the mechanics shared between property and corporate valuation:

- **[Free Cash Flow Valuation: FCFE, FCFF, and the DCF Framework](/blog/trading/asset-valuation/free-cash-flow-valuation-fcfe-fcff-dcf-framework)** — the corporate analog of real estate's NOI-to-value translation; understanding the DCF mechanics here makes real estate DCF intuitive.

- **[EV Multiples: EV/EBITDA, EV/Sales, and Enterprise Value Valuation](/blog/trading/asset-valuation/ev-multiples-evebitda-evsales-enterprise-value-valuation)** — the cap rate is the real estate equivalent of EV/EBITDA in reverse; this post builds the multiple intuition.

- **[Discount Rates in Practice: WACC, Cost of Equity, and Unlevered Beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta)** — the discount rate used in real estate DCF is derived from the same CAPM machinery; the Real Estate WACC (~7.8%, Damodaran 2025) and how it feeds into required returns.

- **[How to Value Real Estate Investment](/blog/trading/real-estate/how-to-value-real-estate-investment)** — a practical companion post in the real estate series focusing on investment property analysis from the investor's workflow perspective.

### Suggested reading progression

If you are brand new to real estate investing:

1. Start with cap rate mechanics (you just read this)
2. Move to the full DCF framework post above to deepen multi-period intuition
3. Study WACC and discount rate derivation to understand where the 8–10% hurdle rates come from
4. Apply to a real offering memorandum using the NOI rebuild discipline from this post

### Tools and data sources

- **CBRE US Cap Rate Survey** (published semi-annually): the most comprehensive cap rate data by market and property type
- **CoStar / LoopNet:** Commercial real estate listing and transaction data; essential for comp analysis
- **Damodaran Online:** Annual WACC and beta data by sector, including Real Estate (~7.8% WACC)
- **Global Property Guide:** Cross-country rental yield data for residential markets, including Vietnam
- **Knight Frank / JLL Research:** Asia-Pacific market reports with Vietnam-specific data
- **NCREIF Property Index:** Quarterly returns on institutional unleveraged real estate in the US

---

Real estate valuation is not exotic. It is the income approach you already know from corporate finance, stripped of the accounting complexity and made tangible by the physical nature of the underlying asset. The cap rate is just a yield — the ratio of the property's earning power to its price. NOI is just operating income before debt. The DCF is just a projection of those earnings over a hold period, plus a terminal value. The sophistication lies not in the formulas but in the quality of the assumptions: what vacancy to budget, what rent growth to underwrite, what exit cap rate to apply, and whether the market cap rate for this specific asset type and location in this specific macro environment justifies the price being asked. Master those judgment calls and you have mastered real estate valuation.
