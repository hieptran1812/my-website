---
title: "Price-to-Earnings Ratio: How P/E Multiples Value Stocks"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "A complete, from-first-principles guide to P/E multiples: what they measure, how to derive a fair value from the Gordon Growth Model, when they break, and how to use them correctly in practice."
tags: ["pe-ratio", "price-to-earnings", "valuation", "relative-valuation", "multiples", "stock-analysis", "peg-ratio", "cape", "shiller-pe", "equity-research", "fundamental-analysis"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 48
---

> [!important]
> **TL;DR** — The P/E ratio is not just a shorthand for "cheap" or "expensive"; it is the market's answer to a precise question: how many years of today's earnings are investors willing to pay upfront for this business?
>
> - **What P/E measures**: price divided by earnings per share, equal to the implied payback period in years at the current earnings run-rate.
> - **Three versions matter**: trailing (actual), forward (estimated), and normalised/CAPE (cycle-averaged) — they can diverge 30% or more in volatile years.
> - **The theoretical anchor**: Gordon Growth gives `P/E = (1 − b) / (ke − g)`, so a "fair" multiple is entirely determined by the payout ratio, required return, and long-run growth rate.
> - **When P/E breaks**: negative earners, cyclical peaks, banks/insurers, and asset-heavy businesses all require a different multiple.
> - **The number to remember**: the S&P 500 averaged a trailing P/E of roughly 21x from 2010 to 2024 — a multiple that peaked at 38x in the COVID distortion of 2020 and troughed at 13x in the post-financial-crisis trough of 2011.

It is January 2021. The S&P 500 is trading at a trailing P/E of roughly 28x. A television anchor holds up a chart and says "stocks have never been this expensive." Three years later in January 2024, the index is trading at a trailing P/E of 25x. A different anchor holds up a similar chart and says "stocks look reasonably priced." The underlying market price between those two moments went up about 30%.

How can the same number — or a number very close to it — tell two completely opposite stories at two different times? The answer reveals why the P/E ratio is simultaneously the most used and the most abused tool in equity analysis. It is not that the P/E ratio lied. It is that both anchors forgot to ask the most important follow-up question: "compared to what?"

The P/E ratio is a ratio, which means it is only meaningful in context — relative to interest rates, relative to growth expectations, relative to history, relative to sector peers. A P/E of 25x when the risk-free rate is 0.5% is a very different proposition from a P/E of 25x when the risk-free rate is 5%. This post teaches you exactly what the P/E ratio measures, why it is anchored in a rigorous theoretical framework, when it is the right tool, and when to reach for something else instead.

![P/E ratio mental model flow diagram](/imgs/blogs/price-to-earnings-ratio-pe-valuation-stocks-1.png)

---

## Foundations: What the P/E Ratio Actually Measures

Before diving into the mechanics, let's establish why the P/E ratio exists at all and what problem it was designed to solve.

When you buy a share of stock, you are buying a claim on the future earnings of a business. But future earnings are uncertain and arrive over time, so their value is not the same as today's value. Fundamentally, any valuation problem reduces to: *how much is this uncertain stream of future cash worth today?*

A full discounted cash flow analysis is the rigorous answer — discount each year's expected cash flow at an appropriate rate and sum them all up. But a DCF requires forecasting revenues, margins, reinvestment needs, and terminal value for 10 years into the future. For many applications, analysts want a quick check: is this business approximately cheap or expensive relative to its peers, relative to history, relative to the broad market? That is where multiples come in.

The P/E ratio is the **most natural shorthand** because earnings is the closest accounting approximation to the cash flow that shareholders care about. Revenue is too far up the income statement (ignores costs). EBITDA ignores capex. Book value ignores the quality of the assets. Earnings per share — net income attributable to common shareholders divided by diluted share count — is an imperfect but widely available and economically meaningful anchor.

The ratio captures, in a single number, the market's collective judgment about the interplay of risk, growth, and return for a business. When you see a P/E of 35x, you are reading a signal: the market collectively believes this business's earnings will grow rapidly, or it requires a lower return to hold this stock (because the business is safe and predictable), or both. Your job as an analyst is to evaluate whether those embedded beliefs are warranted.

### The simplest definition: implied payback years

Strip away all the jargon and the P/E ratio answers one question: **if the company earns exactly what it earned last year, every year, forever, how many years would it take to earn back your purchase price?**

Mathematically:

```
P/E = Price per share / Earnings per share
```

If you pay \$100 for a stock with \$5 of earnings per share, the P/E is 20. At \$5 per year with no growth, you have bought 20 years of earnings. That is the raw meaning.

The number feels abstract until you compare it to something. A 20-year payback period sounds long. A 10-year Treasury note in 2024 yielded about 4.57%, meaning \$100 invested there returns \$4.57 per year — that is a P/E of roughly 21.9 (\$100 / \$4.57). So at that yield environment, the stock's 20x P/E is not dramatically different from the bond's implied "multiple." At a 2% yield, the bond's P/E equivalent would be 50x, making stocks look cheap. Context is everything.

### The earnings yield: the reciprocal that tells you more

The reciprocal of P/E — E/P — is called the **earnings yield** and it is often more intuitive:

```
Earnings yield = 1 / P/E = EPS / Price
```

A 20x P/E translates to a 5% earnings yield. A 25x P/E translates to a 4% earnings yield. A 40x P/E translates to a 2.5% earnings yield.

Now the comparison to the risk-free rate becomes natural. In a world where 10-year Treasuries yield 4.5%, an earnings yield of 5% suggests only a thin risk premium for owning equities — and that is before accounting for the fact that earnings are uncertain while the Treasury coupon is guaranteed. In a world where Treasuries yield 1%, a 5% earnings yield represents a large and attractive gap.

The **equity risk premium (ERP)** is defined informally as: `earnings yield − risk-free rate`. When that gap is historically wide, equities are cheap relative to bonds; when it narrows or inverts, equities are expensive. This single comparison explains why the same P/E number told opposite stories in 2010 (Treasuries at 3.3%, P/E at 16.3x → earnings yield 6.1%, fat premium) versus 2020 (Treasuries at 0.9%, P/E at 38.3x → earnings yield 2.6%, still a premium but far smaller).

### The Gordon Growth derivation: the theoretical anchor for a "fair" P/E

The P/E ratio did not fall from the sky. There is a rigorous valuation model that pins down exactly what the multiple *should* be, given certain assumptions about the business.

Recall the **Dividend Discount Model** (covered in [Dividend Discount Model and Gordon Growth](/blog/trading/asset-valuation/dividend-discount-model-gordon-growth-multi-stage)):

```
P = D1 / (ke - g)
```

Where `D1` is next year's dividend, `ke` is the required return on equity, and `g` is the sustainable long-run growth rate. If the payout ratio is `(1 − b)` — that is, the firm pays out a fraction `(1 − b)` of earnings as dividends and retains the rest at rate `b` — then:

```
D1 = EPS1 × (1 - b)
P = [EPS1 × (1 - b)] / (ke - g)
P / EPS1 = (1 - b) / (ke - g)
```

This is the **Gordon Growth formula for the P/E ratio**. It tells you exactly what drives a fair multiple:

1. **Payout ratio `(1 − b)`**: higher payout → higher P/E, all else equal. A firm that returns more cash to shareholders looks more attractive at the same earnings level.
2. **Required return `ke`**: higher required return → lower P/E. If investors demand 12% to own a stock (because it is risky), they will pay less per dollar of earnings than for a stock where they accept 7%.
3. **Growth rate `g`**: higher growth → higher P/E. A business growing at 10% per year can be worth a much higher multiple than one growing at 2%, because future earnings will be much larger.

The denominator `(ke − g)` is the critical spread. When growth expectations rise or required returns fall, that spread narrows and the multiple expands dramatically. When the spread is large, the multiple is low. This is why rising interest rates mechanically compress P/E multiples — `ke` rises (the discount rate increases), the spread widens, and the formula produces a lower multiple.

For the [cost-of-equity mechanics, see the CAPM and beta explainer](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital).

The Gordon Growth P/E formula also reveals a counterintuitive result about payout ratios. Suppose a company increases its payout ratio from 40% to 60% — should the P/E rise (more cash coming back) or fall (less reinvestment, lower future growth)?

The answer depends on the ROE. If the company has an ROE greater than `ke`, retained earnings create value, and cutting the payout (reducing `b`) to reinvest more actually *raises* the P/E because the growth impact outweighs the lower immediate payout. If ROE equals `ke`, payout ratio doesn't matter — both the numerator and denominator shift in proportion. If ROE is below `ke`, the company is destroying value by retaining and reinvesting; in that case, a higher payout (higher `1 − b`) *increases* the P/E because less value is being destroyed.

This principle — **that value is only created by reinvestment when ROE exceeds the cost of capital** — is one of the most important insights in finance and it is embedded directly in the Gordon Growth P/E formula. A mature utility with ROE of 9% and cost of equity of 8% adds modest value through reinvestment. An exceptional tech platform with ROE of 35% and cost of equity of 10% creates enormous value by retaining as much capital as possible and reinvesting aggressively. The first deserves a modest multiple; the second deserves a very high multiple.

### The P/E ratio as a discount-rate barometer

Another way to read P/E is as a real-time measure of the market's required equity return:

```
ke = Earnings yield + Expected earnings growth rate
ke = (1 / P/E) + g
```

Rearranging the Gordon Growth formula gives this relationship. If the S&P 500 trades at 25x trailing P/E and you assume long-run earnings growth of 5-6%, the implied required return is about 10% (4% earnings yield + 6% growth). Compare that to the risk-free rate of 4.5% in late 2024, and the implied equity risk premium is about 5.5% — which is historically moderate.

If the market's P/E rises to 35x without any change in the risk-free rate or growth expectations, the implied equity risk premium compresses to about 3.4% (2.9% + 6% − 4.5%). Historically, a compressed risk premium is associated with eventual underperformance — investors are accepting less compensation for equity risk than they historically demand.

This "implied ERP" calculation is exactly what practitioners like Aswath Damodaran run every month to gauge whether equities are collectively cheap or expensive. The January 2025 implied ERP for the S&P 500 (from data.py) was 4.60% — below the long-run historical arithmetic average of 8.36% but above the geometric average of 6.15%, suggesting equities were priced for moderate rather than spectacular future returns.

#### Worked example:

A company earns \$4.00 per share this year. The retention ratio is 60% (so payout is 40%). Investors require a 9% return. The company is expected to grow at 4% per year in perpetuity.

Using the Gordon Growth P/E formula:

```
P/E = (1 - 0.60) / (0.09 - 0.04)
P/E = 0.40 / 0.05
P/E = 8x
```

The fair multiple is 8x. At \$4 EPS, the fair price is \$32.

Now suppose the growth rate rises to 6% — perhaps the company enters a new market:

```
P/E = 0.40 / (0.09 - 0.06)
P/E = 0.40 / 0.03
P/E = 13.3x
```

Fair price jumps to \$53.33. The stock went up 67% without earnings changing at all — purely from a 2-percentage-point rise in the expected growth rate. This is the "growth premium" you pay when you buy a high-P/E stock. It is not irrational; it is mathematically grounded. The risk is that the growth never materialises.

---

## Three Versions of P/E: Trailing, Forward, and Normalised

Every time someone quotes a P/E ratio, you need to ask: which earnings? The answer matters enormously because trailing, forward, and normalised P/E can diverge by 30% or more in any given year.

![Three P/E flavours pipeline diagram](/imgs/blogs/price-to-earnings-ratio-pe-valuation-stocks-3.png)

### Trailing P/E (LTM — last twelve months)

The **trailing P/E** divides the current share price by actual reported earnings from the last twelve months (LTM). This is the version most data providers display by default — it uses numbers that are already known, audited, and real.

Advantages:
- Based on audited facts, not guesses
- Comparable across time on a consistent basis
- The standard for index-level comparisons (the S&P 500 P/E series you see on financial news uses trailing)

Disadvantages:
- Backward-looking: what a company earned in the last year may bear little resemblance to what it will earn in the next year, especially for cyclicals
- Can be distorted by one-time items (write-downs, tax windfalls, restructuring charges) that inflate or depress the EPS denominator
- In recessions, earnings collapse while prices hold somewhat steadier, making the trailing P/E spike dramatically even if the "true" value has not changed

### Forward P/E (NTM — next twelve months)

The **forward P/E** divides the current share price by the consensus analyst forecast for earnings over the next twelve months (NTM). It is forward-looking, which means it incorporates market expectations about where earnings are going.

Advantages:
- Relevant for pricing: you are buying tomorrow's earnings, not yesterday's
- Less distorted by one-time items that analysts typically strip out of their "adjusted" EPS forecasts
- Preferred by active investors and analysts doing relative-value comparisons

Disadvantages:
- Analyst estimates are notoriously optimistic, especially at cycle peaks. Consensus forecasts at the start of a recession are almost always too high
- "Adjusted EPS" (the denominator) strips out real costs (stock comp, restructuring) that reduce actual shareholder value
- A company can manage earnings expectations downward, making the forward P/E look cheap just before disappointment

#### Worked example:

The S&P 500 closed 2024 with a price of roughly 4,769 (adjusted for dividends and based on end-of-year data). Reported trailing EPS for 2024 was approximately \$172. Trailing P/E = 4,769 / 172 ≈ **27.7x** (consistent with the 27.6x in the data series).

Consensus analyst estimates for 2025 EPS at that time were roughly \$242 (an optimistic 14% growth forecast). Forward P/E = 4,769 / 242 ≈ **19.7x**.

The same stock market, the same date, shows as 27.7x on one screen and 19.7x on another. Neither is lying. They are answering different questions. The 19.7x forward P/E assumes analysts' 14% EPS growth forecast materialises — if the economy slows, those estimates fall, and the "cheap" forward multiple transforms into an expensive one.

### Normalised P/E and Shiller CAPE

Both trailing and forward P/E share a problem: they use a single year's earnings, which can be wildly above or below trend at different points in the economic cycle. The solution is to **normalise earnings** — average them across a full cycle to get a smoother denominator.

The most famous normalised multiple is the **Shiller CAPE** (Cyclically Adjusted Price-to-Earnings), named after Nobel laureate Robert Shiller. The calculation:

```
CAPE = Current market price / Average of last 10 years of real EPS
```

Where "real" means inflation-adjusted. By averaging ten years of earnings, CAPE smooths through recessions (where EPS collapses) and booms (where EPS spikes). It measures whether the market is cheap or expensive relative to its long-run earnings power — not just this moment's earnings.

The S&P 500's long-run average CAPE is approximately 17x. The CAPE hit 44 in January 2000 just before the dot-com bust, peaked near 38 in December 2021, and stood at approximately 34 in late 2024 — significantly above long-run average but below the dot-com extreme.

CAPE's limitation is that it is very slow-moving and persistently high in modern eras due to structural changes: higher profit margins (technology and asset-light businesses are more profitable), lower interest rates over the 2010s keeping multiples elevated, and accounting changes that reduce reported depreciation. CAPE fans argue this makes the market permanently "look expensive" and that investors who avoided equities because of elevated CAPE missed massive gains.

The practical lesson: CAPE is most useful for long-horizon asset allocation decisions, not for timing individual stock purchases. A trader looking at a 12-month horizon cares far more about forward earnings estimates and near-term earnings revisions than 10-year cycle averages.

---

## The S&P 500 P/E in Practice: What the Data Shows

Before building further theory, it is worth grounding ourselves in the actual historical record.

![S&P 500 trailing P/E 2010-2024 time series chart](/imgs/blogs/price-to-earnings-ratio-pe-valuation-stocks-2.png)

The chart above shows annual trailing P/E from 2010 to 2024. Several patterns are immediately apparent:

**The 2020 spike to 38.3x** is the most dramatic distortion. In 2020, COVID lockdowns devastated corporate earnings — especially in services, travel, and retail. But the Fed slashed rates to near-zero, the government injected trillions in fiscal stimulus, and stock prices recovered rapidly on expectations of a V-shaped rebound. The result: prices stayed up, earnings collapsed, and the trailing P/E doubled. Was the market "expensive"? By trailing P/E, yes. By forward P/E using 2021 earnings forecasts, the ratio looked far more normal because the market was looking through the one-time earnings depression.

**The 2018 compression to 17.5x** happened as the Fed raised rates from near-zero toward 2.5%. Higher rates → higher `ke` in the Gordon Growth formula → compression of the multiple. The market fell roughly 20% in Q4 2018 even as corporate earnings grew solidly, simply because the discount rate rose.

**The 2022 reset to 18.3x** repeated this story more aggressively. The Fed hiked rates from 0.25% to 4.5% in a single year — the fastest tightening cycle in four decades. The S&P 500 fell 18% even as earnings held up reasonably well. Again, it was multiple compression, not an earnings collapse, that drove the decline. The P/E moved from 28x to 18x not because companies earned less, but because investors demanded a higher return.

**The post-2022 re-expansion to 27.6x** by end-2024 reflected multiple expansion driven by AI optimism, a resilient economy, and the expectation that the Fed would cut rates. Earnings growth was solid (~11%), but multiple expansion from 18x to 28x was the dominant driver of the market's 52% gain from the October 2022 trough.

---

## The PEG Ratio: Growth-Adjusting the Multiple

The most obvious criticism of P/E as a standalone metric is that it ignores growth. A pharmaceutical company growing earnings at 25% per year and trading at 40x P/E looks expensive compared to a utility growing at 3% per year at 20x. But is it really more expensive? You are getting very different future streams.

The **PEG ratio** (Price/Earnings to Growth) addresses this by dividing the P/E by the EPS growth rate:

```
PEG = P/E / EPS growth rate (%)
```

Peter Lynch, the legendary Fidelity fund manager who compounded Magellan at 29% per year in the 1980s, popularised a rule of thumb: **a PEG of 1.0 or below is considered fair value; PEG below 0.75 is bargain territory; PEG above 1.5 is expensive**.

The intuition: if a company is growing earnings at 20% per year, paying 20x earnings is "free" — you are not paying anything extra for the growth. Paying 30x for 20% growth (PEG of 1.5) means you are paying a premium and your margin of safety shrinks.

![PEG ratio fair P/E at different growth rates chart](/imgs/blogs/price-to-earnings-ratio-pe-valuation-stocks-7.png)

The chart shows how the PEG framework implies different "fair" P/E multiples at different growth rates. A stock growing at 10% per year is fairly valued at 10x on a PEG=1 basis, and expensive above 15x (PEG=1.5). A stock growing at 25% per year is fairly valued at 25x and only expensive above 37.5x.

#### Worked example:

Company A: Stock price \$120. Trailing EPS \$6.00. Forward EPS growth forecast: 8% per year.

```
Trailing P/E = 120 / 6 = 20x
PEG = 20 / 8 = 2.5 → Expensive
```

Company B: Stock price \$180. Trailing EPS \$6.00. Forward EPS growth forecast: 22% per year.

```
Trailing P/E = 180 / 6 = 30x
PEG = 30 / 22 = 1.4 → Borderline, but more acceptable
```

On a simple P/E basis, B looks 50% more expensive (30x vs 20x). On a PEG basis, A is the expensive one. If you are comparing two technology companies with similar risk profiles, PEG gives a more honest relative ranking.

PEG limitations:
- Growth rate estimates are inherently uncertain — garbage in, garbage out
- Assumes constant growth, which is unrealistic; a cyclical company may not grow at all in year 3 even if it guided 20%
- Does not adjust for the **quality** of growth: a company growing via aggressive accounting or unsustainable promotional spending deserves a lower PEG tolerance than one growing organically with pricing power
- Does not adjust for capital structure: two companies with the same P/E and the same EPS growth could have very different balance sheets (one is net cash, one is levered 4x) — PEG treats them identically

### Extending PEG: the forward PEG and sector-relative PEG

In practice, analysts use a **forward PEG** — forward P/E divided by the 3-5 year consensus EPS CAGR rather than the trailing growth rate:

```
Forward PEG = Forward P/E / Expected 3-yr EPS CAGR (%)
```

The longer time horizon smooths out near-term noise and better captures the investor's actual investment thesis: will this company compound earnings over a multi-year period? A business with lumpy near-term earnings but a powerful compounding engine (think Amazon Web Services or Visa's network effect flywheel) will appear more attractive on a forward PEG using a 3-year horizon than on a 1-year growth rate.

A **sector-relative PEG** compares a stock's PEG to the median PEG for its sector. This is useful because different sectors have different average growth rates — consumer staples at 4-5% growth will never show a PEG of 1 on an absolute basis; the relevant benchmark is whether a specific staples company is cheap or expensive *relative to its sector peers*.

#### Worked example:

Three technology companies, all growing earnings at different rates:

| Company | Stock Price | EPS | P/E | 3-yr EPS CAGR | Forward PEG |
|---------|------------|-----|-----|----------------|-------------|
| TechA | \$200 | \$8 | 25x | 20% | 1.25 |
| TechB | \$150 | \$7.50 | 20x | 10% | 2.00 |
| TechC | \$300 | \$7.50 | 40x | 35% | 1.14 |

On P/E alone: TechB (20x) looks cheapest, TechC (40x) looks most expensive. On forward PEG: TechC (1.14) and TechA (1.25) are in the acceptable zone; TechB (2.00) is actually the most expensive relative to its growth. TechC's high P/E is justified if you believe the 35% earnings growth forecast — a belief you need to stress-test by asking whether the addressable market, competitive position, and margin trajectory support it.

This is exactly the sort of analysis that separates superficial stock commentary ("40x P/E is crazy!") from rigorous valuation work.

---

## Sector and Country P/E Differences: Why 15x Can Be Cheap in One Market and Expensive in Another

One of the most common P/E mistakes is comparing multiples across sectors or countries as if a single "fair" number applies everywhere. It does not. The Gordon Growth formula explains exactly why sector P/E ranges differ:

`P/E = (1 − b) / (ke − g)`

Different sectors have different required returns (their WACC/cost of equity) and different growth profiles. Plug in those numbers and you get structurally different fair multiples.

![Sector implied fair P/E from Gordon Growth bar chart](/imgs/blogs/price-to-earnings-ratio-pe-valuation-stocks-6.png)

The chart above derives implied fair P/E for each S&P 500 sector using Damodaran's WACC estimates from January 2025, sector-specific growth assumptions, and a 40% payout ratio. The key observations:

**Utilities trade at low P/E for a good reason**: high payout, low growth, and a low WACC (6.2%) — but the growth rate is also very low (2%), so the spread `ke − g` is wide relative to other sectors, producing a low implied multiple. A 14-15x utility P/E is perfectly rational.

**Technology commands a high P/E structurally**: the sector's growth assumption (5-6%) is high, and while the WACC is also high (10.2%), the net spread is smaller than it first appears. A 20-25x technology P/E is justified if the growth materialises.

**Banks are excluded from P/E-based analysis entirely**: the concept of "enterprise value" does not apply cleanly to banks (debt is their raw material, not just their financing), and bank P/E must be read alongside price-to-book (P/B), price-to-tangible-book (P/TBV), and return-on-equity. A bank at 12x earnings with ROE of 15% is very different from a bank at 12x earnings with ROE of 8%.

### Cross-country P/E comparisons

![S&P 500 vs VN-Index trailing P/E comparison chart](/imgs/blogs/price-to-earnings-ratio-pe-valuation-stocks-4.png)

The S&P 500 and VN-Index data show a consistent gap: the S&P 500 typically trades 5-12 points higher in P/E terms. This gap is not irrational — it reflects:

1. **Risk premium differences**: Vietnam is an emerging market with currency risk, liquidity risk, and governance risk that commands a higher required return (`ke`) — compressing the multiple
2. **Growth expectations**: while Vietnam's nominal GDP grows faster, this does not automatically translate to higher per-share EPS growth due to frequent share dilution (frequent equity raises) in Vietnamese listed companies
3. **Earnings quality**: Vietnamese GAAP allows more flexibility; international investors discount earnings with a quality haircut
4. **Liquidity**: foreign ownership limits, thin order books, and periodic market circuit breakers reduce the price investors will pay

The practical lesson: do not buy Vietnamese stocks "because the P/E is low compared to the US." The discount exists for structural reasons. You need to understand whether the specific discount is justified or whether you see a catalyst that narrows it.

### How global factors move P/E multiples

Cross-country P/E differences are not static — they shift with macroeconomic conditions, capital flows, and sentiment cycles. Several mechanisms drive P/E expansion and contraction at the market level:

**1. Interest rate cycles**: as demonstrated in 2022, rising rates mechanically compress multiples. The US 10-year yield moved from 1.52% (end-2021) to 3.88% (end-2022). For stocks, `ke` rises proportionally through the CAPM: if the risk-free rate rises 2.4 percentage points, `ke` rises roughly `2.4% × beta`. For a market-beta stock (beta = 1), that is a direct 2.4pp increase in the required return — almost exactly what the P/E compression from 28x to 18x implies.

**2. Earnings growth expectations**: in 2020, despite the COVID crash, the market recovered quickly because investors believed earnings would rebound sharply in 2021-2022 (they did). The P/E spiked to 38.3x not because investors were irrational but because the denominator (TTM earnings) collapsed while the numerator (forward-looking price) held in anticipation of recovery. Forward P/E in mid-2020 was a far more reasonable 20-22x.

**3. Risk premium shifts**: geopolitical shocks, financial crises, or liquidity events compress multiples by raising the risk premium investors require. The 2011 trough P/E of 13x coincided with the European debt crisis, US credit downgrade, and fears of a double-dip recession — all driving the risk premium up, compressing multiples even as the underlying US economy was recovering.

**4. Structural regime changes**: the 2010s saw persistently higher P/E multiples than the 1990s or 2000s in part because technology platforms (Apple, Google, Microsoft, Amazon) with near-monopoly competitive positions and 30-40% operating margins dominated the index. These businesses structurally deserve higher multiples than the industrial conglomerates and financial firms they replaced in index weight. When the index composition changes, the "average" P/E changes even without a change in monetary conditions.

#### Worked example:

Vietcombank (VCB.HM) traded at approximately 12-13x trailing P/E in December 2024 (FiinGroup data). An analyst looking at the S&P 500 bank sector (averaging around 14-15x in 2024) might say Vietcombank looks cheap. But:

- VCB's required equity return: roughly 13-14% (Vietnam risk-free ~4.5% + ERP ~8-9%)
- VCB's sustainable ROE: roughly 18-20% (high quality for Vietnam)
- Gordon Growth: with payout ~30%, retention ~70%, g ≈ ROE × b ≈ 0.19 × 0.70 ≈ 13.3% — but this exceeds `ke`, which violates the model (g must be below `ke` for the perpetuity to converge)

This illustrates an important practical point: for high-growth businesses in emerging markets, you cannot use a simple Gordon Growth P/E. You must use a multi-stage model where the first few years see above-ke growth that then normalises. The steady-state P/E is still anchored by the Gordon formula at the terminal growth rate.

---

## When P/E Breaks: Four Failure Modes to Know Cold

The P/E ratio is a powerful tool with a known set of blind spots. Every serious analyst memorises these failure modes.

![When P/E breaks failure mode diagram](/imgs/blogs/price-to-earnings-ratio-pe-valuation-stocks-5.png)

### The mechanics of EPS: what goes into the denominator

Before examining failure modes, it is worth understanding what "earnings per share" actually contains — because the quality of the denominator is just as important as the multiple itself.

**Diluted EPS** (the standard denominator) is computed as:

```
Diluted EPS = Net income attributable to common shareholders
              / (Basic shares outstanding + dilutive securities)
```

Where "dilutive securities" includes stock options in the money, convertible notes, and warrants that would increase the share count if exercised. Diluted EPS is always ≤ basic EPS.

Several items can make reported EPS misleading:

1. **Non-recurring charges**: a company that takes a one-time restructuring charge of \$500 million will show depressed EPS for that year. Analysts typically "normalise" by adding back these charges to get "adjusted EPS" or "operating EPS." The risk: companies abuse this to consistently strip out costs that are not really one-time.

2. **Share buybacks**: a company can grow EPS without growing net income by buying back shares — the same earnings divided by fewer shares produces higher EPS. A company growing EPS at 10% through buybacks alone is not the same as one growing earnings organically at 10%, but the P/E calculation treats them identically.

3. **Leverage effects**: a company that finances acquisitions with debt can grow EPS through financial engineering (adding acquired earnings, paying debt interest that is below the acquisition earnings yield) without creating genuine value. This is another source of "EPS growth" that does not merit a P/E expansion.

4. **Pension and accounting assumptions**: changes in pension discount rates, depreciation methods, or goodwill treatment can materially shift reported EPS without any operational change.

The practical implication: when evaluating a P/E multiple, always ask whether the earnings quality is high (clean, recurring, cash-backed) or whether you are looking at a managed number. For high-quality businesses with straightforward financials, reported EPS is a reliable anchor. For complex conglomerates, highly acquisitive businesses, or companies with large non-cash charges, you need to rebuild normalised earnings from the cash flow statement.

#### Worked example:

Company X reports net income of \$500 million. Of this:
- \$120 million is a one-time gain from selling a division
- \$80 million is a litigation settlement received (non-recurring)
- \$50 million is from a tax credit that will not recur

Adjusted (normalised) net income: \$500m − \$120m − \$80m − \$50m = **\$250 million**

If diluted share count is 100 million shares:
- Reported EPS: \$500m / 100m = \$5.00
- Normalised EPS: \$250m / 100m = \$2.50

If the stock trades at \$75:
- P/E on reported EPS: 75 / 5.00 = **15x** (looks cheap)
- P/E on normalised EPS: 75 / 2.50 = **30x** (looks expensive)

The same stock. The same price. Two very different stories — and which one is right depends entirely on your view of whether those "one-time" items were truly non-recurring. This is why every serious analyst builds their own adjusted EPS model rather than accepting the headline number.

### Failure Mode 1: Negative or Near-Zero Earnings

If a company earns negative EPS, the P/E ratio is negative — and a negative P/E has no intuitive interpretation. A startup with \$50 stock and -\$2 EPS has a P/E of -25x. A more mature company with 1 cent of EPS has a P/E of 5,000x. Neither number is useful.

**What to use instead**:
- **EV/Revenue (EV/Sales)**: useful for high-growth companies where profitability is a future promise
- **EV/Gross Profit**: if revenue has high variability in gross margin
- **Price-to-Book (P/B)**: useful if the balance sheet is informative (e.g., pre-revenue biotech sitting on a cash hoard)
- **Reverse DCF**: instead of computing a P/E, ask "what growth rate does the current market price imply, and do I believe it?" — see [Absolute Valuation methods in this series](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims)

### Failure Mode 2: Cyclical Peak Earnings

Cyclical industries (steel, mining, energy, chemicals, autos) have earnings that track the commodity or demand cycle. At cycle peaks, earnings are inflated well above what is sustainable over a full cycle.

The danger: at the top of a commodity cycle, earnings are high, so P/E looks low — the stock appears cheap just as it is most dangerous. At the bottom of a cycle, earnings are depressed or negative, P/E looks high or meaningless, and the stock is often actually cheap.

The classic example is US steel companies in 2021-2022. EPS more than tripled as steel prices surged post-COVID. P/E ratios compressed to 3-5x. Value investors piled in. When steel prices normalised in 2023, earnings fell 60%+, and the "cheap" P/E evaporated.

**What to use instead**: normalised or mid-cycle earnings. For commodities, use mid-cycle commodity price assumptions. For autos, use normal-volume earnings. For energy, use a long-run oil price (Damodaran often uses \$65-75/bbl for normalisation).

#### Worked example:

A steel company has trailing EPS of \$18 at cycle peak (steel price \$1,100/tonne). The stock trades at \$90. Trailing P/E = 5x — looks incredibly cheap.

Analyst estimates mid-cycle EPS at \$6 (at a normalised steel price of \$700/tonne). Normalised P/E = 90 / 6 = **15x** — suddenly looks fairly valued or slightly expensive given the steel sector's capital intensity and cyclicality typically warrant a below-market multiple.

The company is only "cheap" if you believe the current cycle peak is the new normal. More often, the cycle mean-reverts, and the "5x P/E" transforms into a 15-20x on normalised earnings.

### Failure Mode 3: Banks and Financial Institutions

Banks present a structural problem: their debt (deposits, borrowed funds) is their raw material, not just their capital structure. Earnings are computed after interest expense on deposits, which is not comparable to manufacturing earnings computed before financing costs. The enterprise value concept breaks down.

For banks, the practical metrics are:
- **P/TBV (Price-to-Tangible Book Value)**: how much are you paying over the book value of the loan portfolio less intangibles?
- **ROE vs. Cost of Equity**: if a bank earns ROE of 15% and cost of equity is 10%, it creates value and deserves a P/B above 1.0x. If ROE equals cost of equity, fair value is P/B = 1.0x. Below cost of equity, P/B should be below 1.0x.

P/E can still be used for rough comparisons within the banking sector (bank A at 10x vs bank B at 14x), but only as a screening tool, not as a framework for computing intrinsic value.

### Failure Mode 4: Asset-Heavy Businesses with High Depreciation

For real estate investment trusts (REITs), pipelines, tower companies, and capital-heavy infrastructure businesses, GAAP earnings are depressed by depreciation on long-lived physical assets. A REIT owning apartment buildings depreciates them under GAAP, even though the properties are often appreciating in real terms.

**What to use instead**:
- **P/FFO (Funds From Operations)**: add back depreciation and amortisation to earnings — this better reflects the cash-generating power of the assets
- **P/AFFO (Adjusted FFO)**: subtract maintenance capex — what the REIT actually has available to pay dividends
- **EV/EBITDA**: for capital-intensive industrials and energy companies, strips out D&A and financing costs for comparability

---

## Common Misconceptions

### Myth 1: A low P/E always means cheap

A stock at 8x P/E is not automatically cheap. It might be in a business model that is in structural decline (think legacy media in 2015-2020), facing rising competitive pressure, or have earnings that are set to fall sharply. The P/E is simply a snapshot. The question is what earnings trajectory justifies that multiple.

The practical test: compute a "reverse PEG" — if the P/E is 8x and the market requires PEG of 1.0 to be fair value, the stock is only worth the multiple if earnings are growing at 8% per year sustainably. If they are growing at 0% or shrinking, 8x may still be expensive.

### Myth 2: High P/E always means speculative bubble

Amazon traded above 100x P/E for most of the 2010s. At the time, financial commentators called it "absurdly overvalued." But the forward earnings growth was explosive — AWS was barely in its infancy. By 2023, Amazon's P/E had compressed to 60x, not because the stock fell, but because earnings quadrupled. The investors who held through 100x and 80x and 60x were right to do so because the growth justified the premium.

The key question for a high-P/E stock is never "is 40x too high?" but "does the embedded growth rate in the Gordon Growth formula actually match what this business can deliver over the next decade?"

### Myth 3: You should compare a stock's P/E to the market average

Comparing Apple's P/E to the S&P 500 average is somewhat useful but significantly incomplete. Apple competes for capital against other technology companies, not against the utility sector or the banking sector. The relevant comparison is within-sector (Apple vs. Microsoft vs. Alphabet), adjusted for growth differentials (use PEG within the group), and checked against the fundamental anchor (Gordon Growth).

### Myth 4: Forward P/E is more accurate than trailing P/E because it uses better information

Forward P/E uses analyst estimates, which are systematically optimistic — especially at the top of the cycle. Multiple academic studies show that at 12-month forward horizons, consensus EPS estimates are typically 8-12% too high, on average, across the market cycle. The forward P/E should be viewed with an appropriate confidence interval — usually ±3-5x depending on the volatility of the business.

### Myth 5: The long-run S&P 500 P/E average of 15-17x is a reliable anchor for today

The long-run average P/E — going back to 1900 — is approximately 15-17x. But that average includes decades of very high interest rates (1970s: rates of 15%+, P/E of 7-8x) and very low rates (2010s: rates near zero, P/E of 20-25x). The average is not a stable physical constant — it is the average outcome of a wide distribution driven primarily by the rate environment. In a structurally lower-rate world (as the 2010s appeared to be), a higher average P/E is rationally justified.

The proper reference for any given moment is the **rate-conditional P/E**: given today's 10-year Treasury yield, what is the historically "normal" P/E? Running a regression of S&P 500 P/E against the 10-year yield over the past 40 years shows a clear inverse relationship — each 1 percentage point rise in yields is associated with roughly a 2-3 point decline in the market P/E. Using this relationship as your anchor is more rigorous than citing the raw unconditional average.

### Myth 6: A P/E that has "always been high" is special

Some investors argue that certain companies — often high-quality compounders — "deserve" to always trade at a premium and that paying a premium P/E is fine because the premium persists. There is truth in this: genuinely high-ROIC businesses with durable competitive moats (strong brands, network effects, switching costs) do command structurally higher multiples. But the statement is often used to justify paying any price.

The test: at a given P/E, is the embedded growth rate in the Gordon formula achievable, and is the required return at that price competitive with alternatives? A company that "always trades at 40x" but whose earnings growth has decelerated to 8% is now offering an earnings yield of 2.5% plus 8% growth = 10.5% implied return. If you can get 4.5% on a Treasury, the equity risk premium is only 6%, which may or may not be adequate compensation for the business risk. "Always expensive" is not a substitute for running the numbers.

---

## How It Shows Up in Real Markets: Three Case Studies

### Case Study 1: The 2022 Rate Shock and Multiple Compression

At the end of 2021, the S&P 500 trailing P/E was 28.1x. One year later, at end-2022, it was 18.3x. The index fell approximately 18%.

Let's decompose the return:

- S&P 500 price end-2021: ~4,766
- S&P 500 price end-2022: ~3,839
- Trailing EPS end-2021: ~\$170
- Trailing EPS end-2022: ~\$210 (earnings actually grew)
- Change in earnings: +24%
- Change in price: -19%
- **Implied change in P/E**: from 28.1x to 18.3x = **-35%**

The entire market decline was driven by P/E compression, not an earnings collapse. Earnings grew 24%. The multiple fell 35%. The net result: -19% for the index. This decomposition — separating earnings growth from P/E change — is the analyst's most powerful tool for understanding return drivers.

Why did the multiple compress? The Fed raised rates from 0.25% to 4.5% in a single year. In the Gordon Growth formula:
```
P/E = (1 - b) / (ke - g)
```
If `ke` rises by 3 percentage points (as it roughly did, tracking the increase in the risk-free rate through beta), and `g` stays constant, the denominator widens and the multiple compresses. For a typical market portfolio, a 3pp rise in `ke` from 8% to 11%, with `g` fixed at 4%, would produce:

```
Old P/E = 0.40 / (0.08 - 0.04) = 0.40 / 0.04 = 10x
New P/E = 0.40 / (0.11 - 0.04) = 0.40 / 0.07 = 5.7x
```

The multiple falls 43%. The real-world compression of 35% is directionally consistent with this mechanical effect.

### Case Study 2: Microsoft's P/E Journey 2015-2024

Microsoft is one of the most analysed stocks in the world and its P/E history illustrates almost every concept in this post.

In 2015-2016, Microsoft traded at roughly 20-22x trailing P/E. The business was transitioning from boxed software (declining) to cloud services (growing rapidly). Many analysts considered it "expensive for a slow-growth company," comparing it to traditional software peers on a trailing basis.

By 2023, Microsoft was trading at 35-38x trailing P/E. By the trailing-multiple logic, it was now dramatically more expensive. But:

1. Azure had grown from near-zero to \$100 billion annualised run rate
2. Operating margins expanded from ~35% to ~45%
3. Free cash flow grew from ~\$20 billion to ~\$60 billion per year

If you had refused to buy Microsoft in 2016 because "20x is expensive," you missed a 5x return by 2024. The P/E was not the right lens — the growth trajectory was.

The lesson is not that P/E doesn't matter for Microsoft. It is that the correct question in 2016 was: "Is 20x fair given where Azure is going?" A discounted cash flow built on realistic Azure growth assumptions would have told you the answer was yes.

### Case Study 3: Using P/E to Filter, Not to Decide

A practical institutional workflow: a portfolio manager screens the S&P 500 for stocks with forward P/E below 15x and EPS growth forecast above 10% (implicit PEG below 1.5). The screen returns 40-60 names. These are candidates for deeper work — they are not automatically buys.

The next filter: are there obvious P/E "traps" (cyclicals at peak earnings, negative surprises in forecasts, accounting concerns)? This removes perhaps half the screen. The remaining 20-25 names go to full fundamental analysis (DCF, competitive moat analysis, management quality).

The P/E ratio is used as a **coarse filter** to focus analytical attention — it is not the buy signal. This is its proper role in professional practice.

### Case Study 4: Decomposing S&P 500 Returns 2010-2024

One of the most powerful exercises an analyst can do is decompose past index returns into their three components: (1) earnings growth, (2) multiple expansion/contraction, and (3) dividend yield. The P/E data makes this exercise straightforward.

Using the data series:
- End-2010: P/E = 16.3x; assume EPS ≈ \$83 (implicit price ~\$1,353)
- End-2024: P/E = 27.6x; EPS ≈ \$172 (implicit price ~\$4,747)

**Earnings growth contribution**: EPS grew from \$83 to \$172 = +107%, or approximately +5.2% per year annualised over 14 years.

**Multiple expansion contribution**: P/E rose from 16.3x to 27.6x = +69%. This contributed approximately +3.8% per year annualised.

**Dividend yield contribution**: the S&P 500 dividend yield averaged approximately 1.9% over this period.

**Total return decomposition** (approximate): 5.2% earnings growth + 3.8% multiple expansion + 1.9% dividends ≈ **10.9% per year** — closely matching the actual annualised total return of the S&P 500 during this period.

The decomposition reveals a sobering point for forward-looking investors: roughly 35% of the S&P 500's annualised gain since 2010 came from P/E expansion, not from actual earnings growth. A P/E that expanded from 16x to 28x cannot do the same again — it would require moving from 28x to 48x, which is historically unprecedented outside of specific bubble episodes. Investors who expect 10%+ annualised S&P 500 returns going forward need to identify where those returns will come from: earnings growth will need to do most of the heavy lifting if multiples are to remain stable or compress.

#### Worked example:

Suppose you are building a 10-year forward return model for the S&P 500 as of end-2024:

- Starting P/E: 27.6x
- Assumed ending P/E (in 10 years): two scenarios
  - Bull case: P/E stays at 27.6x (no multiple change)
  - Bear case: P/E reverts to 20-year average of ~22x (compression)
- EPS growth assumption: 7% per year (consensus long-run estimate, roughly in line with nominal GDP + margin improvement)
- Dividend yield: 1.3% (the late-2024 S&P 500 yield)

**Bull case return** (flat multiple):
```
10-yr EPS growth: (1.07)^10 = 1.967x (EPS from ~172 to ~338)
Multiple unchanged: 27.6x
Price target: 338 × 27.6 = ~9,329
Annualised capital return: (9,329/4,769)^(1/10) - 1 = ~6.9%
Total return (with dividends): ~8.2%
```

**Bear case return** (multiple compression to 22x):
```
Same EPS growth: price target = 338 × 22 = ~7,436
Annualised capital return: (7,436/4,769)^(1/10) - 1 = ~4.5%
Total return: ~5.8%
```

The range — 5.8% to 8.2% — is well below the 10.9% historical average, and entirely because the starting multiple is elevated. An investor who fully understands P/E mechanics will not be "surprised" by a decade of moderate returns from an elevated-multiple starting point; they will have priced it in from the start.

This kind of forward-looking return decomposition is exactly the toolkit that long-horizon investors — endowments, pension funds, family offices — use to set asset allocation. P/E is not just about picking individual stocks. It is a fundamental input to any return expectation model.

---

## Further Reading & Cross-Links

The P/E ratio is most powerful when combined with the broader toolkit:

- **[The Valuation Spectrum: Absolute, Relative, and Contingent Claims](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims)** — situates P/E within the full map of valuation methods and explains when to use relative vs. absolute approaches.
- **[Risk, Required Return, CAPM and Beta](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital)** — the `ke` in the Gordon Growth P/E formula comes directly from CAPM; understanding how to compute required return is essential for deriving a defensible "fair" multiple.
- **[Relative Valuation Multiples and Comps](/blog/trading/equity-research/relative-valuation-multiples-comps)** — P/E sits within a broader family of multiples including EV/EBITDA, P/B, EV/Sales, and P/FFO; this post covers how to run a full comparable companies analysis.
- **[Discount Rates in Practice: WACC, Cost of Equity, and Unlevered Beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta)** — for companies where debt complicates the equity cost calculation, understanding WACC is a prerequisite for applying the Gordon Growth P/E anchor correctly.

### Integrating P/E with the DCF: the bridge between relative and absolute methods

One of the most powerful applications of P/E is as a **terminal value multiple in DCF models**. Instead of using a Gordon Growth terminal value (which requires you to specify a perpetual growth rate), analysts sometimes exit the explicit forecast period by applying a P/E multiple to the terminal year's EPS.

For example, if a company is projected to earn \$12 EPS in year 10, and the analyst believes the business will trade at 18x earnings at that point (based on sector comps and the company's mature growth profile), the terminal value is simply \$12 × 18 = \$216 per share. Discounted back at `ke` for 10 years, this terminal value anchors the DCF.

This approach makes the terminal value assumption more transparent: instead of debating whether a 3.5% perpetual growth rate is defensible, you debate whether 18x is a reasonable exit multiple. Both approaches are mathematically equivalent (a 18x P/E implies a specific `g` given `ke`), but the multiple form is easier to benchmark against observable market prices. The [full DCF framework is covered in the absolute valuation posts in this series](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims) — using P/E as the terminal anchor is one of the most common bridge techniques in professional practice.

### Key takeaways for working analysts

1. **Always specify which P/E** — trailing, forward, or normalised — before making a comparison. They can differ by 30% in a single year.
2. **Anchor every P/E to its Gordon Growth drivers**: what required return, payout, and growth rate justify the current multiple? If those assumptions are unrealistic, the multiple is not justified.
3. **Rate environment dominates**: the single most important input to the "fair" market P/E is the level of interest rates. Never evaluate a market P/E without noting the concurrent 10-year Treasury yield.
4. **Use P/E as a filter, PEG for growth adjustment, and DCF for the final call** — the three-tool sequence is more powerful than any single metric.
5. **Know the failure modes cold**: negative earnings, cyclical peaks, banks, and asset-heavy businesses all require alternative metrics. Reaching for the wrong multiple is one of the most common valuation errors in practice.

### The P/E ratio in portfolio construction: market timing vs. position sizing

Professional investors use P/E at two levels beyond individual stock analysis:

**Market-level P/E for tactical allocation**: when the market-level forward P/E is in the top decile of its historical distribution (above ~24x on a rate-adjusted basis), history suggests below-average returns over the next 5-10 years. This does not predict the timing of drawdowns, but it informs the expected return assumption used in portfolio optimisation models. A pension fund using a 10% equity return assumption at a market P/E of 28x is making an optimistic bet; at 15x, the same assumption is conservative.

**Relative P/E for sector rotation**: when the technology sector's P/E premium over the market reaches historical extremes, a mean-reversion overlay might suggest tilting toward value sectors. This is not market timing in the short-term sense; it is a systematic recognition that extreme relative valuations tend to revert.

**P/E within factor models**: in quantitative investing, low P/E (or high earnings yield) is one of the original "value" factors. Research going back to Fama and French (1992) shows that stocks with low P/E ratios have historically outperformed over long horizons, though the magnitude of the value premium has varied significantly by era. Understanding *why* low-P/E stocks outperform — primarily through re-rating as sentiment normalises and through the earnings themselves being more persistent than implied by the low multiple — deepens your intuition for how to use P/E in practice.

The P/E ratio has been misread, misquoted, and misused more than almost any other number in finance. That is not a reason to avoid it — it is a reason to understand it more precisely than the commentators who throw it around carelessly. The analyst who can state — at any moment — the Gordon Growth drivers behind the current market P/E, the implied equity risk premium, and which specific conditions would cause the multiple to expand or contract has a significant edge over one who simply looks up the trailing P/E on a data screen and forms a gut view. A practitioner who can decompose a P/E into its Gordon Growth drivers, switch fluently between trailing/forward/normalised versions, identify the failure modes, and cross-check with PEG and DCF has a genuine analytical edge over the crowd quoting a single number in isolation.
