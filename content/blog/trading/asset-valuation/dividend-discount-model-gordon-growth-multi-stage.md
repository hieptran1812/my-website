---
title: "Dividend Discount Model: From Gordon Growth to Multi-Stage DDM"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "A step-by-step guide to pricing dividend-paying stocks using the zero-growth DDM, Gordon Growth Model, and multi-stage DDM, with worked examples from Coca-Cola, VCB, and US utilities."
tags: ["dividend-discount-model", "ddm", "gordon-growth-model", "dividend-valuation", "intrinsic-value", "stock-valuation", "cost-of-equity", "multi-stage-ddm"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR** — The Dividend Discount Model (DDM) prices a stock as the present value of all future dividends, and the Gordon Growth Model collapses that infinite sum to a single formula: P = D1 / (Ke − g).
>
> - The zero-growth DDM (P = D/Ke) treats a stock as a perpetuity — best for preferred stock or utilities with flat payouts.
> - The GGM adds one assumption — constant dividend growth — and is exquisitely sensitive to the spread (Ke − g).
> - Two-stage and three-stage DDMs handle firms transitioning from high growth to maturity, capturing real dividend trajectories.
> - DDM works best for banks, utilities, REITs, and consumer staples; it breaks down for non-dividend-paying or buyback-only firms.

---

In 1938, an economist named John Burr Williams wrote a doctoral thesis that changed how professionals think about stocks. The title was *The Theory of Investment Value*, and his central argument was deceptively simple: **a stock is worth the present value of all the dividends it will ever pay.** Nothing more and nothing less. No price targets, no chart patterns, no gut feeling — just a stream of future cash flows, discounted back to today.

Wall Street largely ignored him for a decade. But by the 1950s, his framework had become the bedrock of academic finance, and today every major bank's equity research team keeps a spreadsheet descended, in spirit, from Williams's 1938 equations.

The Dividend Discount Model (DDM) is the direct descendant of Williams's insight. It is both the oldest formal stock-valuation model and one of the most misunderstood. Critics call it naive ("most companies don't pay dividends!") and defenders call it the only *theoretically pure* method ("everything else is a shortcut"). Both groups are partly right. Learning the DDM properly means understanding exactly where it is illuminating, where it is blind, and how to extend it when a single-stage formula is not enough.

This post builds the DDM from scratch — from the intuition behind discounting future cash, through the algebra of the Gordon Growth Model, all the way to multi-stage frameworks used in professional equity research. By the end, you will know not just *what* the DDM says but *why* each assumption matters and how a 1-percentage-point error in your growth estimate can swing a valuation by 50%.

![DDM pipeline: future dividends discounted to intrinsic value then compared to price](/imgs/blogs/dividend-discount-model-gordon-growth-multi-stage-1.png)

The diagram above captures the whole logic: forecast future dividends, discount each one back to today at your required return, sum them all, and compare the result to what the market is charging. If intrinsic value exceeds market price, the stock is cheap. If not, it is expensive. Everything in the rest of this post is about how to do those four steps rigorously.

---

## Foundations: What a Dividend Is and Why It Drives Value

### The equity claim

When you buy a share of stock, you are buying a *residual claim*. Residual means you get paid last — after the firm's employees, suppliers, debt holders, and tax authorities all collect what they are owed. What remains belongs to equity holders. The firm can distribute that remainder as **dividends** (cash paid directly to shareholders), retain it as **reinvested earnings** (plowed back into the business to fund future growth), or use it to **repurchase shares** (buy back stock, which also returns cash to shareholders who sell).

DDM focuses on the dividend leg of this picture. It asks: if a company paid out all of its residual value as dividends over time, what would those dividends be worth today?

### Why cash flows, not earnings, determine value

A common intuition is that a stock's value tracks its earnings per share (EPS). Earnings matter — but they are an accounting measure, not a cash measure. A company can report positive EPS while its bank account is empty (if it extended credit to customers, for example), and it can report zero EPS while generating abundant cash (if it is depreciating heavily). What you actually receive as a shareholder is cash, not an EPS number.

The DDM sidesteps this entirely by anchoring to **dividends** — the actual cash that arrives in your brokerage account. This is why it is sometimes called the purest valuation model: it prices exactly what you physically receive as an equity holder.

### The time value of money

Before we can sum future dividends, we need to handle the fact that \$1 received today is worth more than \$1 received next year. Why? Because \$1 today can be invested at a return, so in one year it becomes \$1 × (1 + r). Equivalently, \$1 received in one year is worth only \$1 / (1 + r) today — a process called **discounting**.

The *discount rate* we use is the **cost of equity (Ke)** — the return investors require to hold this particular stock, given its risk. If a stock is riskier than average, investors demand a higher return, which means a higher Ke, which means future dividends are discounted more aggressively, which means the intrinsic value is lower. Risk and price move in opposite directions, which is exactly what intuition suggests.

For a deeper treatment of the time-value mechanics, see [Time Value of Money: The Engine Behind Every Valuation Model](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model).

### The general DDM formula

In its most general form, the DDM says:

$$P_0 = \sum_{t=1}^{\infty} \frac{D_t}{(1 + K_e)^t}$$

Where:
- \$P_0\$ = intrinsic value of the stock today
- \$D_t\$ = dividend expected at time \$t\$
- \$K_e\$ = required return on equity (cost of equity)

This is an infinite sum — a daunting thing to compute. The special cases (zero-growth, Gordon Growth, multi-stage) are all clever tricks for making this infinite sum tractable without throwing it on a computer.

---

## The Zero-Growth DDM: A Perpetuity

### What it assumes

The simplest special case assumes dividends never change. The company pays \$D\$ every single year, forever — same amount, no adjustments for inflation, no growth, no cuts. This is called a **perpetuity** (from the Latin *perpetuus*, meaning continuous).

The infinite sum collapses to a single formula via the perpetuity formula from basic finance:

$$P = \frac{D}{K_e}$$

Where \$D\$ is the constant annual dividend and \$K_e\$ is the required return.

### Intuition: buying a fixed annuity

Think of it like this. You are offered an investment that pays you \$100 every year, forever. You require a 10% return on your money. How much is this worth to you?

Answer: \$100 / 0.10 = \$1,000. At \$1,000, a \$100 annual payment is exactly a 10% yield on your investment. If you paid \$800, you'd earn 12.5% (more than your required return — a good deal). If you paid \$1,200, you'd earn only 8.3% (less than required — a bad deal).

That is the zero-growth DDM in full.

### When does it apply?

This model is appropriate when dividends really are flat and stable. The canonical examples:
- **Preferred stock**: companies issue preferred shares with a fixed dividend (e.g., \$5 per share per year forever). The zero-growth DDM prices preferred stock almost exactly.
- **Utilities with rate-controlled cash flows**: some regulated electric utilities have dividends that are effectively contractual and barely move year to year.
- **REITs with fixed lease income**: a real estate investment trust with long-duration fixed-rent leases can approximate the zero-growth case.

#### Worked example:

A preferred share pays a fixed dividend of \$2.00 per year. An investor requires a 10% return on equity.

**Intrinsic value** = \$2.00 / 0.10 = **\$20.00**

Now suppose interest rates fall and the investor's required return drops to 8%.

**New intrinsic value** = \$2.00 / 0.08 = **\$25.00**

A 2-percentage-point drop in the required return raised the intrinsic value by **25%** — from \$20 to \$25. This explains why utility and preferred-stock prices are so sensitive to interest rate movements. When the Fed cuts rates, preferred shares re-rate sharply upward, not because earnings improved but because the denominator (Ke) shrank.

---

## The Gordon Growth Model: Adding Growth

The zero-growth model is too simple for most common stocks, which typically raise their dividends over time as their earnings grow. In 1956, economist Myron Gordon (building on Williams's work) showed how to extend the perpetuity formula to handle constant dividend growth — and the result is the most widely used DDM formula in professional equity research.

### Derivation

Suppose dividends grow at a constant annual rate \$g\$ forever. Then:
- \$D_1\$ = the next annual dividend (one year from now)
- \$D_2 = D_1 \times (1+g)\$
- \$D_3 = D_1 \times (1+g)^2\$
- ... and so on.

The present value of this growing stream, discounted at Ke, is a **growing perpetuity**:

$$P_0 = \frac{D_1}{K_e - g}$$

This formula is the **Gordon Growth Model (GGM)**. The key insight: instead of discounting at \$K_e\$, you discount at the *spread* \$(K_e - g)\$. Growth effectively narrows the discount rate because each future dividend is larger than it would have been under zero growth.

### What the symbols mean

- **\$D_1\$** — the *next* year's dividend (not the current year). This is an easy source of error: if a stock just paid \$D_0 = \$2.00\$ and dividends grow at 5%, then \$D_1 = \$2.00 \times 1.05 = \$2.10\$.
- **\$K_e\$** — the cost of equity. For most public companies, analysts estimate this using the Capital Asset Pricing Model (CAPM): \$K_e = r_f + \beta \times (r_m - r_f)\$, where \$r_f\$ is the risk-free rate, \$\beta\$ is the stock's sensitivity to market moves, and \$(r_m - r_f)\$ is the equity risk premium.
- **\$g\$** — the constant long-run dividend growth rate. This is the most sensitive input. In theory, \$g\$ should be the sustainable long-run growth rate, often approximated as the retention rate × return on equity (ROE): \$g = (1 - \text{payout ratio}) \times \text{ROE}\$.

![Zero-growth DDM versus Gordon Growth Model structural comparison](/imgs/blogs/dividend-discount-model-gordon-growth-multi-stage-2.png)

The left panel shows the zero-growth case — flat dividends, discounted at the full Ke. The right panel shows the GGM — dividends growing at g, which means the effective discount rate shrinks to (Ke − g), and the resulting price is higher. This is the core mechanism: for the same next dividend, a company with growth is worth more.

### The dangerous zone: when Ke approaches g

The GGM has a mathematical singularity when the growth rate approaches the discount rate. If \$g = K_e\$, the denominator is zero and the formula produces infinity. If \$g > K_e\$, the formula gives a negative price, which is economically meaningless.

This is not a bug — it is the model telling you something important. A company cannot grow forever at a rate equal to or above investors' required return. In practice, the long-run growth rate of any company is bounded by the growth rate of the overall economy (roughly 2–5% nominal in developed markets, 5–8% in Vietnam).

When you see implied growth rates approaching Ke, ask: am I in the Gordon Growth zone at all, or should I use a multi-stage model? The GGM is only valid when \$g < K_e\$ and when you believe that growth rate is truly sustainable forever.

#### Worked example:

Coca-Cola's 2024 annual dividend per share was approximately \$1.94. Suppose analysts forecast \$D_1 = \$2.02\$ (4% growth, in line with the historical trajectory). Estimate cost of equity at 8% (risk-free rate 4.5%, beta ~0.6, equity risk premium 5.8%).

$$P = \frac{\$2.02}{0.08 - 0.04} = \frac{\$2.02}{0.04} = \$50.50$$

Coca-Cola's market price in mid-2025 was approximately \$60. What implied growth rate does the market embed?

Solving for \$g\$:

$$\$60 = \frac{\$2.02}{0.08 - g}$$

$$0.08 - g = \frac{2.02}{60} = 0.0337$$

$$g = 0.08 - 0.0337 = \textbf{4.63\%}$$

The market is pricing in about 4.6% perpetual dividend growth, slightly above the analyst estimate of 4.0%. That is a reasonable tension — it suggests the stock is modestly rich relative to the base case, but the premium is not absurd. This is the kind of reverse-engineering that makes the GGM genuinely useful: rather than just computing a price, you use the observed price to back out what growth rate the market has priced in, then judge whether that is realistic.

---

## GGM Sensitivity: Small Inputs, Huge Swings

One of the most important things to internalize about the Gordon Growth Model is how explosively sensitive it is to the inputs. A 1-percentage-point change in either Ke or g can shift the estimated price by 20–50%. This is not a limitation of the model per se — it is an accurate reflection of the mathematics of growing perpetuities.

![GGM sensitivity heatmap showing price vs required return Ke and growth g, D1 = 4](/imgs/blogs/dividend-discount-model-gordon-growth-multi-stage-3.png)

The heatmap above holds \$D_1 = \$4\$ constant and shows how the estimated price changes as Ke varies from 7% to 11% and g from 1% to 5%. Read it like a multiplication table for valuation uncertainty.

Key observations:
- **Moving left to right** (increasing g while holding Ke): prices rise sharply, especially where Ke is low.
- **Moving top to bottom** (increasing Ke while holding g): prices fall, sometimes by half.
- **The best combination** (Ke = 7%, g = 5%) produces prices near \$200, while the worst (Ke = 11%, g = 1%) gives roughly \$40. Same dividend, five-fold price difference.
- **Near the diagonal** (Ke ≈ g), prices blow up. This is the dangerous zone.

#### Worked example:

A utility stock pays a dividend of \$3.60 next year (D1). You estimate Ke = 9% and g = 3%.

$$P = \frac{\$3.60}{0.09 - 0.03} = \frac{\$3.60}{0.06} = \$60$$

Now suppose the central bank cuts rates by 200 basis points and investors revise their required return down to 7%.

$$P = \frac{\$3.60}{0.07 - 0.03} = \frac{\$3.60}{0.04} = \$90$$

The stock re-rated from \$60 to \$90 — a **50% gain** — without any change in dividends or earnings. This is not irrational; it is the Gordon Growth Model in action. Utility stocks and REITs are sometimes called "bond proxies" precisely because their valuations are driven primarily by the discount rate (Ke), not by the growth story. When the Fed cuts rates, long-duration assets like utilities get a mechanical re-rating — even if business operations are unchanged.

This also explains why the 2022 rate-hike cycle was so damaging to utility valuations: a 4-percentage-point rise in Ke, without a commensurate rise in g, can cut a utility's GGM value nearly in half.

---

## The Two-Stage DDM: Growth Then Maturity

Most interesting companies do not grow at the same rate forever. They go through a life cycle: a period of above-average growth (when they are taking market share, investing heavily, or operating in a fast-expanding sector) followed by eventual maturity (when growth slows to match the overall economy).

The **two-stage DDM** handles this directly. Stage 1 is a finite high-growth period (often 5–10 years) where dividends grow at \$g_1\$. At the end of stage 1, the company transitions to a stable growth rate \$g_n\$ that is assumed to last forever. The terminal value at the end of stage 1 is then priced via a regular GGM formula.

$$P_0 = \sum_{t=1}^{n} \frac{D_t}{(1+K_e)^t} + \frac{P_n}{(1+K_e)^n}$$

Where:
- Each \$D_t = D_0 \times (1 + g_1)^t\$ during stage 1
- \$P_n = D_{n+1} / (K_e - g_n)\$ is the terminal value at the end of year \$n\$
- \$D_{n+1} = D_n \times (1 + g_n)\$ is the first dividend under the stable-growth rate

![Two-stage DDM timeline showing stage 1 dividends and terminal value in year 5](/imgs/blogs/dividend-discount-model-gordon-growth-multi-stage-4.png)

The timeline above shows the structure: years 1–5 yield individually discounted dividends (stage 1, high growth), and at year 5 the company switches to a terminal-value GGM. The stage-1 dividends are relatively small in present value; the terminal value — being an infinite stream — typically dominates.

#### Worked example:

A company just paid a dividend of \$D_0 = \$1.00\$. It is expected to grow at \$g_1 = 15\%\$ for 5 years as it expands, then slow to a sustainable \$g_n = 4\%\$ thereafter. The cost of equity is \$K_e = 10\%\$.

**Stage 1: compute each dividend and its PV**

| Year | Growth factor | Dividend \$D_t\$ | PV factor \$1/(1.10)^t\$ | PV of \$D_t\$ |
|------|-------------|---------------|----------------------|--------------|
| 1 | 1.15 | \$1.15 | 0.9091 | \$1.045 |
| 2 | 1.3225 | \$1.32 | 0.8264 | \$1.093 |
| 3 | 1.5209 | \$1.52 | 0.7513 | \$1.143 |
| 4 | 1.7490 | \$1.75 | 0.6830 | \$1.195 |
| 5 | 2.0114 | \$2.01 | 0.6209 | \$1.249 |

**Sum of stage-1 PVs** = \$1.045 + \$1.093 + \$1.143 + \$1.195 + \$1.249 = **\$5.73**

**Stage 2: terminal value at year 5**

\$D_6 = D_5 \times (1 + g_n) = \$2.0114 \times 1.04 = \$2.092\$

\$P_5 = \frac{\$2.092}{K_e - g_n} = \frac{\$2.092}{0.10 - 0.04} = \frac{\$2.092}{0.06} = \$34.87\$

**PV of terminal value** = \$34.87 / (1.10)^5 = \$34.87 / 1.6105 = **\$21.65**

**Total intrinsic value** = \$5.73 + \$21.65 = **\$27.38**

The key observation: the stage-1 dividends contribute \$5.73, but the terminal value alone contributes \$21.65 — nearly **four times** as much. This is typical: in any DDM with a non-trivial growth period, the terminal value dominates. It means the most important number in the model is not how fast the company grows next year, but what stable growth rate you assume it converges to.

---

## The Three-Stage DDM

Some companies do not switch abruptly from high growth to stable growth — they transition gradually. A pharmaceutical company coming off a patent cliff, or a consumer brand expanding internationally, might realistically show 15% dividend growth for 3 years, then gradually decelerate to 8% over the next 4 years, and finally stabilize at 4%. The three-stage DDM formalizes this.

The structure is identical to the two-stage model but with an added transition stage:

1. **Stage 1** (high growth, years 1–n1): dividends grow at \$g_1\$ per year
2. **Stage 2** (transition, years n1+1 to n2): growth rate declines linearly from \$g_1\$ to \$g_n\$
3. **Stage 3** (stable growth, year n2+1 onward): dividends grow at constant \$g_n\$ forever

The terminal value at the end of stage 2 is again computed via GGM, and all the intermediate dividends (including the transition period) are discounted individually.

In practice, the three-stage model is most commonly used by sell-side equity analysts valuing large-cap growth companies — consumer discretionary, technology companies beginning to pay dividends, or pharmaceutical companies. The complexity is warranted because the transition between stages is often the most economically interesting part of the story.

**How to set the transition rate**: a common approach is a linear decline from \$g_1\$ to \$g_n\$ across the transition years. If \$g_1 = 15\%\$ and \$g_n = 4\%\$, and the transition spans 4 years, then the annual growth rates in the transition period would be: 12%, 9%, 7%, 5% (each step roughly 2.75 percentage points down).

### When does a three-stage model add value over two-stage?

The honest answer is: rarely, and only when you have good reasons to believe the transition will be gradual rather than step-function. In most professional equity research, analysts use two-stage models for simplicity — the additional precision from modeling the transition year by year is often swamped by the uncertainty in the terminal growth rate estimate. The three-stage model genuinely earns its complexity for:

1. **Companies transitioning due to regulatory change**: a privatizing utility moving from regulated to partially deregulated pricing may take 5–8 years to fully adjust margins and payout policies. A linear transition in stage 2 captures this more accurately than a step function.

2. **Companies with explicit management guidance**: if management has stated "we expect to increase the payout ratio from 30% to 60% over the next 5 years as the business matures," you can model exactly that in a three-stage framework.

3. **Cross-cycle analysis**: for highly cyclical companies (mining, chemicals, shipping), a three-stage model lets you explicitly model the current cyclical peak/trough in stage 1, the cycle mean-reversion in stage 2, and the long-run steady state in stage 3.

For most ordinary dividend-paying stocks, the two-stage model is sufficient — and the analyst's time is better spent stress-testing the terminal growth assumption than adding a transition stage.

#### Worked example — three-stage: Vietnamese insurance holding company

A Vietnamese insurance holding company just paid a dividend of VND 2,000 per share. Management has guided:
- Stage 1 (years 1–3): 18% annual dividend growth as the company benefits from Vietnam's rapidly expanding middle class.
- Stage 2 (years 4–7): growth decelerates linearly from 18% to 7%.
- Stage 3 (year 8 onward): stable growth at 7% (matching Vietnam's nominal GDP growth).
- Cost of equity: 13% (risk-free 5%, beta 1.2, ERP 6.7%).

**Stage 1 dividends (growth at 18%)**:

| Year | Dividend (VND) | PV factor (1.13)^-t | PV |
|------|---------------|--------------------|----|
| 1 | 2,360 | 0.885 | 2,088 |
| 2 | 2,785 | 0.783 | 2,181 |
| 3 | 3,285 | 0.693 | 2,276 |

**Sum stage 1 PV** = VND 6,545

**Stage 2 dividends (linear decline 18% → 7%)**:

The step-down rate is (18% − 7%) / 4 = 2.75% per year.

| Year | Growth rate | Dividend (VND) | PV factor | PV |
|------|------------|---------------|----------|-----|
| 4 | 15.25% | 3,786 | 0.613 | 2,321 |
| 5 | 12.50% | 4,260 | 0.543 | 2,313 |
| 6 | 9.75% | 4,675 | 0.480 | 2,244 |
| 7 | 7.00% | 5,002 | 0.425 | 2,126 |

**Sum stage 2 PV** = VND 9,004

**Terminal value at year 7**:

\$D_8 = 5,002 \times 1.07 = \text{VND } 5,352\$

\$P_7 = \frac{5,352}{0.13 - 0.07} = \frac{5,352}{0.06} = \text{VND } 89,200\$

**PV of terminal value** = VND 89,200 × (1.13)^{-7} = VND 89,200 × 0.425 = **VND 37,910**

**Total intrinsic value** = 6,545 + 9,004 + 37,910 = **VND 53,459**

The terminal value contributes VND 37,910 of VND 53,459 — about **71%** of the total. Even with a three-stage model designed to capture a nuanced growth trajectory, the terminal value still dominates. This is a universal feature of growing perpetuity models: the terminal assumption matters more than every year you forecast explicitly.

---

## Estimating the Inputs

The DDM formula is simple. The hard work is estimating the inputs — and that work cannot be automated. Here is the practitioner's approach.

### Estimating Ke: the cost of equity

Most analysts use the **Capital Asset Pricing Model (CAPM)**:

$$K_e = r_f + \beta \times ERP$$

Where:
- \$r_f\$ = risk-free rate (typically the 10-year government bond yield). For USD-denominated analysis, use the 10-year US Treasury; for VND-denominated analysis, use the 10-year Vietnamese government bond (approximately 4.5–5.5% in 2024).
- \$\beta\$ = the stock's sensitivity to the market index. \$\beta > 1\$ = more volatile than market; \$\beta < 1\$ = less volatile.
- \$ERP\$ = equity risk premium — the extra return investors demand for holding stocks instead of bonds. Damodaran estimates the US ERP at roughly 4.5–5.5% as of 2024; emerging markets like Vietnam carry a higher premium (often 7–9%).

For a utility stock with \$\beta = 0.5\$, US risk-free rate of 4.5%, and ERP of 5.0%:

$$K_e = 4.5\% + 0.5 \times 5.0\% = 7.0\%$$

For a Vietnamese bank with \$\beta = 1.2\$, local risk-free rate of 5.0%, and ERP of 8.0%:

$$K_e = 5.0\% + 1.2 \times 8.0\% = 14.6\%$$

Notice how much higher the implied discount rate is for emerging-market equities. That higher Ke means that for the same dividend stream, intrinsic values look lower — which is why emerging-market stocks often trade at lower P/E multiples even when their growth prospects look similar to US peers.

### Estimating g: the dividend growth rate

Sustainable dividend growth is bounded by several factors:

1. **Earnings growth**: dividends cannot grow faster than earnings per share over the long run. If payout ratios rise indefinitely, the firm eventually pays out more than it earns.
2. **Retention × ROE**: \$g = \text{retention rate} \times \text{ROE}\$. A firm that pays out 60% of earnings (retention rate 40%) and earns a 15% return on equity can sustain \$g = 0.40 \times 0.15 = 6\%\$.
3. **Macroeconomic ceiling**: in the long run, no firm can grow faster than the economy it operates in. Terminal growth rates above nominal GDP growth (roughly 5–7% in Vietnam, 3–4% in the US) are unsustainable.

In practice, analysts triangulate between the historical dividend growth rate (look at 5- and 10-year CAGRs), the consensus earnings growth forecast (often from Bloomberg or FactSet), and the sustainable growth rate from the retention × ROE calculation.

### Anchoring terminal g to macroeconomic reality

A convenient heuristic: terminal g should be no higher than the long-run nominal GDP growth rate of the company's primary market. Here is why. If a company grows its dividends faster than the economy indefinitely, its dividends eventually outgrow the entire economy — an absurdity. "Indefinitely" is what the GGM's perpetuity assumes, so the terminal g truly is a claim about the infinite long run.

Long-run nominal GDP growth by region (as of 2025 consensus estimates):
- **United States**: 3.0–3.5% (roughly 2% real + 1.5% inflation target)
- **Eurozone**: 2.5–3.0%
- **Vietnam**: 8–10% (roughly 6% real + 3–4% inflation)
- **India**: 9–11% (high real growth + moderate inflation)
- **Japan**: 1.5–2.0% (slow real growth + near-zero inflation)

For Vietnam specifically, the standard assumption among local analysts is a terminal nominal growth rate of 7–9%, anchored to the country's GDP trajectory. This is why Vietnamese bank stocks can justify higher P/E multiples than at first glance — the higher terminal g compresses the (Ke − g) spread, supporting a higher valuation.

The flip side: you must use a Vietnamese Ke (reflecting the higher risk-free rate and country risk premium), which partially offsets the benefit. The net effect on valuation is that Vietnamese stocks are not simply "cheap" or "expensive" relative to US counterparts — they operate on a different (Ke, g) axis entirely.

---

## When DDM Works and When It Fails

![DDM suitability: works for utilities banks REITs; fails for unprofitable tech early-stage buyback-only](/imgs/blogs/dividend-discount-model-gordon-growth-multi-stage-6.png)

### DDM works well for:

**Utilities and regulated industries**

Electric, gas, and water utilities are the DDM's natural home. Their dividends are contractual obligations tied to regulated rate-of-return frameworks. Growth rates are slow and predictable (tied to rate base expansion). Cost of equity is well-defined. The GGM is not just applicable here — it is the dominant method used by utility-focused fund managers.

**Banks and financial institutions**

Large commercial banks pay substantial, growing dividends and have relatively predictable payout ratios once you model the regulatory capital requirements. Banks cannot easily use free-cash-flow-to-equity models because "capital expenditures" and "working capital" do not have the same meaning as for industrial companies. The DDM sidesteps this problem.

**Real Estate Investment Trusts (REITs)**

REITs are legally required to distribute at least 90% of taxable income as dividends in the US, making the dividend the central financial metric. A well-run REIT with stable occupancy and rent escalation clauses is almost perfectly suited to the GGM.

**Consumer staples with long dividend histories**

Coca-Cola has raised its dividend for 62 consecutive years (as of 2024), qualifying as a "Dividend King." Johnson & Johnson, Procter & Gamble, Colgate-Palmolive, and Nestlé are similar. For these companies, the DDM produces estimates that align closely with market prices because the market itself uses DDM logic to price them.

### DDM struggles or fails for:

**Non-dividend-paying companies**

If a company pays no dividends — Tesla, Alphabet pre-2024, Amazon — the DDM literally produces a price of zero for any finite forecast period. You can try to forecast when dividends *will* start, but this introduces enormous uncertainty.

**Companies returning cash primarily via buybacks**

Apple, Google, and many mature US tech companies return more cash to shareholders via share repurchases than via dividends. The DDM ignores buybacks. The fix is to use a **Free Cash Flow to Equity (FCFE) model** instead, which is the DDM's broader cousin, or to add back implied "equivalent dividends" from buybacks — but that requires assumptions the DDM itself does not make.

**Early-stage, high-growth, or loss-making companies**

A startup or a pre-profitability tech company may have enormous intrinsic value based on future earnings potential, but zero or negative near-term cash flow. DDM cannot price this. Venture capital and growth equity use different frameworks: comparable transaction multiples, discounted revenue or user metrics, or scenario-weighted EBITDA trees.

**Companies with volatile or inconsistent payout policies**

If a company's dividend history looks like a random walk — cut one year, doubled the next, suspended during a recession — the GGM's constant-g assumption is simply false. You can use a multi-stage model, but only if you have genuine visibility into the future payout trajectory.

---

## Sensitivity Analysis: The Practitioner's Reality Check

Before using any DDM output to make a buy or sell decision, every serious analyst runs a sensitivity table. The standard format is a two-dimensional matrix: Ke on one axis, g on the other, with the implied price at each intersection. The goal is to understand the *range of plausible valuations*, not a single point estimate.

A typical analyst's workflow:
1. Estimate base-case Ke (e.g., 9%), g (e.g., 4%), and D1 (e.g., \$3.00).
2. Base case: P = \$3.00 / (0.09 − 0.04) = \$60.
3. Run sensitivity: Ke from 7% to 11%, g from 2% to 6%.
4. Map the range of implied prices.
5. Ask: at what part of that range does the stock look clearly cheap, fairly valued, or clearly expensive?

The sensitivity table does two things. First, it shows you where the model is robust (prices cluster tightly — sign of a mature, slow-growth company) and where it is fragile (wide range — sign of growth-sensitive assumptions). Second, it forces you to commit to a view on the range of realistic inputs, which is itself a discipline.

---

## Common Misconceptions

### Misconception 1: "The DDM only works if the stock pays dividends right now"

False. The two-stage DDM often models a first stage where dividends are zero (the company reinvests everything) and a second stage where dividends begin. What matters is that dividends *eventually* materialize — a perpetual non-dividend-paying company has zero DDM value, but a company expected to start paying in year 5 can absolutely be valued this way.

### Misconception 2: "A higher dividend always means higher intrinsic value"

Tricky. Paying a higher dividend today reduces retained earnings, which reduces reinvestment, which reduces future growth. All else equal, a company that pays \$5 today and grows at 2% may be worth less than a company that pays \$2 today and grows at 8%. The DDM captures this through the g term.

### Misconception 3: "The DDM requires constant growth — real companies are messier"

This was true of the original GGM, but two-stage and three-stage DDMs handle non-constant growth directly. The only binding constraint is that *at some point in the future*, the company must be in a stable, perpetual-growth mode — which is a reasonable assumption for most going-concern businesses.

### Misconception 4: "A DDM price above market price means 'buy'"

Not necessarily. The DDM is only as good as its inputs. If your Ke estimate is too low (you underestimated risk) or your g estimate is too high (you were overly optimistic about growth), the model says "buy" when the market's assessment is more accurate. DDM outputs are hypotheses, not certainties. Always ask: which assumption is the market disagreeing with, and why?

### Misconception 5: "DDM is an outdated method — analysts prefer DCF"

The Discounted Cash Flow (DCF) model and the DDM are close cousins. When you discount free cash flow to equity (FCFE) instead of dividends, and assume the firm's financing choices do not destroy value, the two models converge to the same estimate. The DDM is *especially* appropriate when FCFE is hard to estimate (as in banking). See [Discounted Cash Flow: The Complete DCF Guide](/blog/trading/equity-research/discounted-cash-flow-dcf-complete-guide) for the full DCF treatment.

### Misconception 6: "A low dividend yield means the stock is expensive"

The dividend yield (annual dividend / stock price) is often used as a quick valuation proxy, but it is deeply ambiguous without context. A 1% dividend yield can mean the stock is expensive (overpriced relative to its dividends) *or* that the company retains most of its earnings to fund high-return reinvestment (low payout ratio, high g). Amazon's dividend yield was essentially 0% for two decades — yet Amazon was not expensive by most value measures because it was reinvesting at extraordinary returns on capital.

The correct way to use dividend yield in a DDM context is to back out the implied growth: if yield = D/P and GGM says P = D1/(Ke−g), then yield ≈ (Ke − g). A 2% yield on a utility with Ke = 7% implies the market expects g ≈ 5% perpetually — which for a regulated utility would be quite aggressive. A 2% yield on a bank in a high-growth emerging market with Ke = 12% implies only g ≈ 10% — potentially quite conservative given the bank's growth trajectory.

### Misconception 7: "Terminal value is just a plug number"

This is probably the most dangerous misconception in all of valuation. When a professional says "terminal value is just a plug," they usually mean they do not know what growth rate to use so they chose a number that made their target price come out where they wanted. This is the wrong way to work.

Terminal value in the DDM is a real economic claim: you are asserting that from year N onward, this company will grow its dividends at rate \$g_n\$ forever. That rate must be justified. In practice, \$g_n\$ should be close to the nominal GDP growth rate of the company's primary market — for Vietnam, roughly 5–8%; for the US, 3–4%. A terminal growth rate far above GDP is a mathematical claim that the company will eventually become larger than the entire economy. It is not a "plug" — it is your most important assumption.

The standard exercise: find the terminal growth rate that makes your DDM match the current market price (the implied g), then ask whether that implied rate is more or less realistic than your own estimate. If the market implies 7% perpetual growth and you think 4% is the right number, you have identified a genuine valuation disagreement — not just an assumption difference.

---

## How DDM Shows Up in Real Markets

### Coca-Cola: the dividend compounder

Coca-Cola is the quintessential DDM stock. The company raised its dividend for 62 consecutive years through 2024. The payout ratio has been stable (roughly 70–80% of earnings). The business — selling concentrate to bottlers who sell to billions of consumers — has extremely predictable economics.

![Coca-Cola dividend per share 2000 to 2024 bar chart with trend line](/imgs/blogs/dividend-discount-model-gordon-growth-multi-stage-5.png)

The bar chart above shows Coca-Cola's dividend per share from 2000 to 2024. The compound annual growth rate over this 24-year period is approximately 4.4% — right in the ballpark of what a GGM would assume as a terminal growth rate. The brief dip in 2012 reflects a special accounting adjustment, not a true dividend cut; the streak of annual increases remained intact.

For Coca-Cola, a GGM analysis is entirely appropriate: \$D_1 \approx \$2.02\$, \$g \approx 4.0–4.5\%\$, \$K_e \approx 8\%\$, yielding an intrinsic value of roughly \$50–58. The stock has historically traded in this range plus a small premium for its brand and balance-sheet quality.

### US utilities: the interest rate relay

Duke Energy, Dominion, and Southern Company are all priced primarily via the GGM (or its two-stage cousin). During the 2010–2021 era of near-zero interest rates, the Fed Funds rate collapsed from 5.25% to ~0%, dragging Ke for utilities from roughly 8–9% down to 5–6%. A 3-percentage-point drop in Ke, with g held constant at ~3%, roughly doubles the GGM price. Utility stocks did, in fact, double or more in that era, not because their businesses improved dramatically, but because the discount rate fell.

The reverse happened in 2022–2023: the Fed raised rates aggressively, pushing the 10-year Treasury yield from 1.5% to 5.0%. Ke for utilities rose by ~2.5–3.5 percentage points. The GGM implied prices fell 30–45%. Utility stocks dropped roughly that much.

This is the DDM made real. The model does not just explain prices in hindsight — it *predicts* how rate-sensitive utility valuations are, giving investors a tool to position before rate moves materialize.

### Singapore REITs: DDM as primary valuation tool

Singapore's REIT market (S-REITs) is one of the most mature and analytically sophisticated in Asia. By law, Singapore REITs must distribute at least 90% of distributable income as dividends. This makes them almost ideal DDM candidates — high, stable, predictable distributions, with growth tied to rental escalations and property acquisitions.

A typical S-REIT analysis:
- **Cost of equity (Ke)**: 7–9%, reflecting Singapore's low-risk environment (risk-free rate ~3.5%, beta ~0.5–0.8, ERP ~4.5%).
- **Growth rate (g)**: 2–4%, tied to Singapore's modest but stable inflation and rent escalation clauses (typically CPI + 1–2% in long-term leases).
- **Dividend (D1)**: distributed quarterly; annualize to get the GGM input.

For a REIT paying S\$0.12 in quarterly distributions (S\$0.48 annualized), with Ke = 8% and g = 3%:

$$P = \frac{S\$0.48}{0.08 - 0.03} = \frac{S\$0.48}{0.05} = S\$9.60$$

S-REITs went through a textbook GGM re-rating episode in 2022–2023: as US rates rose and Singapore dollar rates followed, Ke for S-REITs rose from roughly 7% to 9–10%. A REIT previously valued at S\$9.60 under 8% Ke would be worth S\$6.00 under 10% Ke — a 37.5% decline — with no change in rents or distributions. S-REIT indices fell approximately 30–40% during this period, almost precisely matching what the GGM would predict.

### VCB (Vietcombank): emerging-market DDM

Vietcombank (VCB) is Vietnam's largest commercial bank by assets. As of 2024, VCB paid an annual cash dividend of approximately VND 1,200 per share. The stock price was approximately VND 80,000.

**Implied dividend yield** = VND 1,200 / VND 80,000 = **1.5%**

#### Worked example:

Using a conservative Ke estimate for VCB: risk-free rate 5.0% (10-year Vietnam government bond, 2024), beta ~1.1, ERP = 5.5% for a state-owned bank with lower perceived risk.

$$K_e = 5.0\% + 1.1 \times 5.5\% = 5.0\% + 6.05\% = 11\%$$

Apply the GGM to back out the implied growth rate:

$$\text{VND } 80,000 = \frac{D_1}{K_e - g}$$

With \$D_1 \approx \text{VND} 1,272\$ (= 1,200 × 1.06, assuming 6% near-term growth):

$$0.11 - g = \frac{1,272}{80,000} = 0.0159$$

$$g = 0.11 - 0.0159 = \textbf{9.4\%}$$

Alternatively, using the simpler back-of-the-envelope directly on the current dividend:

**Implied g** = Ke − (D/P) = 11% − (1,200/80,000 × 100%) = 11% − 1.5% = **9.5%**

So the market is pricing in roughly 9.5% perpetual dividend growth for VCB. VCB has grown earnings at roughly 15–20% annually for the past decade, driven by Vietnam's rapidly expanding credit market. But as the bank matures and Vietnam's credit-to-GDP ratio rises, that pace will slow. Whether 9.5% is achievable long-term is a judgment call — but having the number makes the conversation precise.

![VN-Index PE ratio 2015 to 2024 and implied dividend yield dual axis chart](/imgs/blogs/dividend-discount-model-gordon-growth-multi-stage-7.png)

The dual-axis chart above shows the VN-Index's trailing P/E ratio (left axis, blue) alongside an implied dividend yield line (right axis, green) calculated by assuming a 30% aggregate payout ratio: yield ≈ 30% / P/E. In periods where P/E expanded (2016–2018, 2020–2021), the implied yield compressed — meaning the market was pricing in higher growth expectations. The sharp P/E compression in 2022 pushed implied yields back up, reflecting higher required returns as global rates rose.

This is the DDM embedded in the aggregate market: as rates rose worldwide in 2022, Ke rose, the denominator (Ke − g) widened, and prices fell even if g was unchanged. The VN-Index P/E compression from ~18x to ~12x in that year maps almost exactly to what a DDM with a rising Ke would predict.

![DDM model comparison grid: zero-growth formula best-for key-risk vs Gordon Growth vs Multi-Stage](/imgs/blogs/dividend-discount-model-gordon-growth-multi-stage-8.png)

The grid above summarizes the three DDM variants side by side. The zero-growth model's perpetuity formula works best for preferred stock and regulated utilities. The GGM is the workhorse for mature dividend compounders. And the multi-stage DDM is needed whenever a company's growth is expected to slow meaningfully over the forecast horizon.

---

## The Payout Ratio: Bridging Earnings and Dividends

One concept that sits at the center of every DDM estimate is the **payout ratio** — the fraction of earnings per share that a company distributes as dividends. If a company earns \$5.00 per share and pays a \$2.00 dividend, the payout ratio is 40%.

The payout ratio matters because it connects the earnings story (what you forecast from revenue, margins, and capital efficiency) to the dividend story (what the DDM actually discounts). A company can grow earnings rapidly but keep dividends flat by retaining everything. Conversely, a company can grow dividends faster than earnings — temporarily — by raising the payout ratio, but this is unsustainable above 100%.

### The sustainable growth formula

There is a beautiful identity connecting the payout ratio, return on equity (ROE), and the sustainable dividend growth rate:

$$g = (1 - b) \times ROE$$

Where:
- \$g\$ = sustainable long-run dividend growth rate
- \$b\$ = payout ratio (the fraction paid as dividends)
- \$(1 - b)\$ = retention ratio (the fraction reinvested)
- \$ROE\$ = return on equity (after-tax earnings divided by book equity)

This formula says: the more you reinvest, and the more profitable that reinvestment is, the faster dividends can sustainably grow. Let us run a few examples.

**Example A — High-payout utility**: Payout ratio = 75%, ROE = 12%.
\$g = (1 - 0.75) \times 12\% = 0.25 \times 12\% = 3\%\$. Slow but stable dividend growth, which is exactly what you see at American electric utilities.

**Example B — High-growth bank**: Payout ratio = 30%, ROE = 18%.
\$g = (1 - 0.30) \times 18\% = 0.70 \times 18\% = 12.6\%\$. This level of growth is what drove VCB's extraordinary dividend trajectory over 2015–2023 — the bank reinvested heavily, compounding at high returns.

**Example C — Mature consumer staple**: Payout ratio = 60%, ROE = 28%.
\$g = (1 - 0.60) \times 28\% = 0.40 \times 28\% = 11.2\%\$ — higher than you might expect. This is the mathematics behind why Procter & Gamble and Colgate-Palmolive can pay generous dividends *and* grow them at 5–7% annually: their ROEs are extraordinarily high because of strong brand economics, and even a small retention fraction compounds at high rates.

The sustainable growth formula is not a forecast — a company can grow faster or slower than this rate for years. But over long horizons, companies tend to revert toward it, which is why it is a useful sanity check on terminal growth rate assumptions.

### When payout ratios are misleading

Two situations make the payout ratio a poor guide:

1. **Temporarily depressed earnings**: if earnings collapse in a recession, the payout ratio can spike above 100% (the company is paying more than it earns). This is not necessarily a dividend-cut signal if you believe earnings will recover — the payout ratio should be evaluated on normalized earnings.

2. **GAAP earnings vs. cash earnings**: accounting depreciation, amortization, or one-time charges can depress reported EPS below cash earnings per share. A company with heavy goodwill amortization might report a 90% payout ratio but actually be paying dividends well within its cash generation capacity. When in doubt, cross-check the payout ratio against free cash flow to equity, not just EPS.

---

## DDM and the Valuation Spectrum

The DDM does not exist in isolation. It is one of three broad approaches to valuation, and understanding where it fits in the full spectrum helps you deploy it correctly. For a full treatment of this taxonomy, see [The Valuation Spectrum: Absolute, Relative, and Contingent Claims](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims). Here is the summary relevant to DDM:

**Absolute valuation methods** (of which DDM is one) compute intrinsic value by forecasting the firm's own cash flows and discounting them. They do not require a reference company. The DDM, DCF, and Residual Income Model (RIM) all fall here. Strength: theoretically grounded, not subject to relative mispricing. Weakness: sensitive to assumptions.

**Relative valuation methods** (P/E, EV/EBITDA, Price/Book) price a company relative to a peer group. They are fast to apply and widely used by equity analysts, but they inherit the mispricing of the reference group. A whole sector can be simultaneously overvalued on absolute terms but look "cheap" on a relative basis.

**Contingent claims methods** (real options, Black-Scholes applied to corporate equity) price optionality — the value of strategic flexibility or survival. Rarely used for dividend-paying mature companies.

The DDM is the *natural complement* to relative valuation: when P/E multiples look rich, the DDM helps you understand whether the premium is justified by a defensible growth assumption (Ke − g spread), or whether the market is simply in a momentum-driven expansion.

### DDM and P/E: the implied connection

There is a direct algebraic link between the GGM and the P/E ratio. Start from the GGM:

$$P = \frac{D_1}{K_e - g}$$

Express \$D_1\$ as (EPS₁ × payout ratio \$b\$):

$$P = \frac{EPS_1 \times b}{K_e - g}$$

Divide both sides by \$EPS_1\$:

$$\frac{P}{EPS_1} = \frac{b}{K_e - g}$$

This is the **Gordon Growth justified P/E** — the P/E ratio implied by the GGM. It tells you that a company deserves a higher P/E when:
- Its payout ratio is higher (more earnings flow to shareholders now)
- Its cost of equity is lower (it is safer or lower-risk)
- Its growth rate is higher (larger g narrows the denominator)

This formula is enormously useful for cross-sector P/E comparisons. Why does a regulated utility trade at 18× earnings while a cyclical steel company trades at 8×? The GGM gives the precise answer: the utility has a lower Ke (less risk), a higher payout ratio, and a more stable growth rate — all of which expand the justified P/E.

---

## Advanced Topics: Adjusting DDM for Real-World Complications

### What to do when the first dividend has not been declared

Companies typically pay dividends quarterly (in the US) or annually (in most Asian markets including Vietnam). When you do a DDM as of today and the next dividend is paid in 3 months, the DDM formula as written already handles it: \$D_1\$ is whatever dividend arrives next, and the formula discounts it at the appropriate fraction of a year. For simplicity, most analysts use annual dividends and assume they arrive at the end of each year. The difference from using exact quarterly timing is small for stable companies.

### Handling special dividends

Some companies pay occasional *special dividends* — one-time distributions on top of regular dividends. Should you include these in a DDM? The answer depends on your view of the company's capital allocation policy. If special dividends are truly ad hoc (a one-time distribution of excess cash after selling a division), they are not part of the permanent dividend stream and should be excluded from the DDM — they are one-time cash flows that you can value separately.

If special dividends occur regularly (some companies in Vietnam and Southeast Asia use them as a regular top-up), they should be incorporated into your normalized dividend estimate.

### Currency considerations for cross-border DDM

When valuing a Vietnamese company in VND, use a VND risk-free rate and VND equity risk premium — do not mix currencies. If you use a USD discount rate on VND dividends, you will systematically mis-value the stock because Vietnamese interest rates (reflecting VND inflation and credit risk) are significantly higher than US dollar rates. The rule is: **the discount rate currency must match the dividend currency.**

If you want to compute a USD intrinsic value of a Vietnamese stock, convert the VND dividends to USD using forward exchange rates (or use purchasing power parity to forecast the exchange rate), then discount at a USD Ke. In practice, most local investors analyze Vietnamese stocks in VND and most international investors run the analysis in USD — and the two groups can disagree significantly on valuation because of currency assumptions.

---

## Putting It All Together: A DDM Workflow

For a practitioner applying DDM to a real stock, the workflow looks like this:

1. **Screen for suitability.** Is this a dividend-paying company? Is the payout policy stable and predictable? If yes, DDM is a primary tool. If not, use FCF or relative valuation instead.

2. **Estimate Ke.** Use CAPM with a market-appropriate risk-free rate (match the currency) and a beta estimate from Bloomberg or a regression against the relevant index. Add a country-risk premium for emerging markets.

3. **Choose the model variant.** Flat, stable dividend → zero-growth. Growing but mature company → GGM. High-growth transitioning to maturity → two-stage. Gradual deceleration → three-stage.

4. **Estimate D1 and g.** For D1, use the next announced or forecasted dividend. For g (stage 1), use consensus earnings growth × assumed payout ratio, or the historical dividend CAGR. For terminal g (stable stage), anchor to nominal GDP growth of the company's primary market.

5. **Run the formula and build a sensitivity table.** Never use a single-point estimate. Map combinations of (Ke, g) and identify where the stock looks clearly cheap or clearly expensive.

6. **Sanity-check against market price.** If the market price implies a g you think is unrealistically high, the stock may be expensive. If it implies a g below what you think is achievable, the stock may be cheap.

7. **Cross-reference with other methods.** P/E, EV/EBITDA, and DDM should broadly agree. Material divergences demand explanation — usually one input assumption is wrong.

---

## Further Reading and Cross-Links

The DDM sits within a family of absolute valuation methods. To understand it deeply, you need to master the adjacent frameworks:

- **[What Is Value: Philosophy, Frameworks, and Asset Pricing](/blog/trading/asset-valuation/what-is-value-philosophy-frameworks-asset-pricing)** — why intrinsic value exists and how different valuation frameworks connect.

- **[Time Value of Money: The Engine Behind Every Valuation Model](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model)** — the mechanics of discounting that underlie every DDM calculation.

- **[The Valuation Spectrum: Absolute, Relative, and Contingent Claims](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims)** — how to know when DDM is appropriate versus when relative valuation or contingent-claims models are more suitable.

- **[Discounted Cash Flow: The Complete DCF Guide](/blog/trading/equity-research/discounted-cash-flow-dcf-complete-guide)** — the broader cousin of DDM, applicable when companies do not pay dividends or when you want to value the enterprise independently of financing choices.

---

## What to Take Away

The Dividend Discount Model does one thing: it makes explicit the claim that every stock price is a compressed bet on future cash flows. That is simultaneously its greatest strength and its most honest limitation.

For the right kind of company — a Coca-Cola, a Duke Energy, a Vietcombank, a Singapore REIT — the DDM produces valuations that match professional estimates and historical market prices remarkably well. For these companies, understanding the GGM is not just an academic exercise; it is how portfolio managers and sell-side analysts actually think.

For the wrong kind of company — a pre-revenue biotech, a buyback-only technology firm, a startup — the DDM produces nonsense, and professionals know not to use it. The skill is knowing which camp your company falls into.

The deeper lesson of the DDM is not the formula itself but the *discipline*: to value a stock, you must forecast cash flows, estimate risk, and discount. Everything else — growth stories, management quality, competitive moats — matters only insofar as it flows through those three elements. The company that impresses you at a roadshow is worth its discounted dividends, no more and no less.

Williams wrote that formula in 1938. Markets have changed beyond recognition since then. The formula has not.

The DDM rewards discipline: if you can honestly forecast dividends, honestly estimate risk, and honestly acknowledge what you do not know, the model will give you an honest answer. The difficulty is not the math — it is the honesty. Every analyst who has ever built a DDM and then looked at the sensitivity table knows the unsettling feeling of watching a confident price target dissolve into a range spanning \$30 to \$120. That feeling is not a failure of the model; it is the model working exactly as intended, forcing you to confront the genuine uncertainty in owning a piece of a business and its future.

---

*This article is for educational purposes only and does not constitute investment advice. All valuations are illustrative and use estimates as of 2024–2025. Past dividend growth does not guarantee future growth.*
