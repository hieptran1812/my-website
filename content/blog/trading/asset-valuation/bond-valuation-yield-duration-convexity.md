---
title: "Bond Valuation: Yield, Duration, and Convexity Explained"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "A bond's price is the present value of its coupon and principal cash flows — and understanding yield, duration, and convexity tells you exactly how that price moves when interest rates change."
tags: ["bonds", "bond-valuation", "yield-to-maturity", "duration", "convexity", "fixed-income", "credit-spread", "asset-valuation", "interest-rates"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR** — A bond is priced by discounting its scheduled cash flows (coupons + par) at the market's required yield; when yields rise, prices fall — and that relationship is not a straight line but a curve.
>
> - **Price = PV of all coupons + PV of par.** Sum every future cash flow, discounted at the yield to maturity (YTM).
> - **YTM is the one rate that makes the price equation balance.** It is not the same as the coupon rate; it is the *market's* required return given today's price.
> - **Duration** measures how sensitive the price is to a 1% yield change — roughly the weighted-average time until you get your money back.
> - **Convexity** captures the curvature: because of convexity, a bond rallies *more* when yields fall than it loses when yields rise by the same amount.
> - **Credit spread** = a bond's yield minus the risk-free Treasury yield — the market's price for default risk.
> - The US 10-year Treasury yield rose from 0.93% at end-2020 to 4.57% at end-2024 (Federal Reserve H.15) — a 364-basis-point move that crushed long-duration bond prices, a real-world lesson in duration risk.

---

September 2022. The US Federal Reserve had been raising interest rates at the fastest pace in four decades to fight surging inflation. The US 10-year Treasury yield, which had sat at a generational low of 0.93% at the end of 2020, was rocketing past 3%, then 4%. And somewhere in a spreadsheet at Silicon Valley Bank's treasury desk, a question that should have been asked months earlier finally became urgent: *how much had the value of their long-duration bond portfolio actually fallen?*

The answer — more than \$15 billion in unrealized losses on a \$120 billion portfolio — was a textbook illustration of duration risk. It was also a reminder that bonds are not "safe" in any simple sense. They are mathematical objects: their price is precisely calculable, their sensitivity to rate changes is precisely measurable, and ignoring that math has very real consequences.

This post is about exactly that math. We'll start from the absolute beginning — what a bond actually is and how it pays — and build all the way up to convexity, credit spreads, and how professionals actually price different types of bonds. Every step comes with a worked example using real numbers, because that is the only way the abstract formulas become intuitive.

![Bond cash-flow schedule: coupon payments then par at maturity](/imgs/blogs/bond-valuation-yield-duration-convexity-1.png)

---

## Foundations: What a Bond Is and How It Pays

### The loan that trades

Forget the word "bond" for a moment. Think about what you do when you lend your friend \$1,000. You give them the money now; they promise to pay you \$50 per year for five years, then return your \$1,000 at the end. That's a bond — except instead of your friend, the borrower is a government, a corporation, or a municipality, and instead of a handshake the promise is written into a legal contract called an **indenture**.

The three core components of any bond:

1. **Face value (par value)**: the amount you lend, typically \$1,000 per bond. This is the amount returned at maturity.
2. **Coupon rate**: the annual interest rate the issuer pays on the face value. A 5% coupon on a \$1,000 bond = \$50 per year. Most bonds pay semi-annually, so \$25 every six months.
3. **Maturity date**: the date the issuer returns your face value and the bond "expires."

The word "coupon" dates to the 1800s when bonds literally had physical coupons attached — you clipped a coupon and handed it to a bank to collect your interest payment.

### The cash-flow schedule is the bond

Treating a bond as a fixed contract, its value becomes obvious: it is worth the present value of everything it is going to pay you. A 3-year, \$1,000 bond with a 5% annual coupon pays:

- Year 1: \$50 (coupon)
- Year 2: \$50 (coupon)
- Year 3: \$50 + \$1,000 = \$1,050 (final coupon + par)

Nothing more, nothing less. The issuer cannot pay you early, pay you late, or change the amounts. That fixed, predictable schedule is why bonds are called "fixed income."

### Key vocabulary — defined once, used everywhere

**Yield to maturity (YTM)**: The single interest rate that, when used to discount all future cash flows, gives you the current market price. Think of it as the *market's answer* to the question: "given what this bond costs today, what annual return are you actually getting?" We will compute it below.

**Current yield**: Coupon payment / current price. Simpler than YTM but ignores capital gain or loss from buying at a premium or discount to par.

**Par, premium, and discount**: A bond trades *at par* when price = \$1,000 (face value). *At a premium* means price > \$1,000 (yield < coupon rate). *At a discount* means price < \$1,000 (yield > coupon rate). These three states flow directly from the pricing formula.

**Basis point (bps)**: one-hundredth of one percent. 1% = 100 bps. If yields rise by 50 bps, they rise from, say, 4.00% to 4.50%.

**Duration**: a measure of price sensitivity to yield changes. We cover this thoroughly below.

**Convexity**: the curvature of the price-yield relationship. Duration is the slope; convexity is how that slope changes.

**Credit spread**: the extra yield a bond pays above a comparable-maturity Treasury, compensating investors for default risk.

With these terms established, we can now build the pricing machine.

---

## The Pricing Formula: Present Value of Every Cash Flow

### Why present value? The time value of money

A dollar today is worth more than a dollar tomorrow. That's not an opinion — it's a consequence of the fact that you could invest a dollar today and have more than a dollar later. The mechanics of present value are covered fully in [the time value of money foundation post](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model); here's the essential version.

If you can earn rate *r* per year, then \$1 received in *t* years is worth \$1 / (1+r)^t today. This is called the **discount factor** for period *t* at rate *r*. To find the present value of a series of cash flows, you discount each one and sum them.

### The bond pricing formula

A bond with face value *F*, coupon rate *c*, *n* periods, and yield *y* per period is priced as:

```
P = C/(1+y) + C/(1+y)^2 + ... + C/(1+y)^n + F/(1+y)^n
```

Where *C* = coupon payment per period = *c* × *F*.

This simplifies to:

```
P = C × [1 - (1+y)^(-n)] / y  +  F / (1+y)^n
```

The first term is the **present value of the coupon annuity** — the stream of fixed payments. The second term is the **present value of the principal** — the lump sum you get at maturity.

#### Worked example:

You are considering buying a 10-year US Treasury bond with:
- Face value: \$1,000
- Annual coupon rate: 3% (so coupon = \$30 per year, paid semi-annually as \$15 every 6 months)
- Market yield: 5% per year (2.5% per semi-annual period)
- Number of periods: 20 (10 years × 2)

Let's compute the price:

**Step 1 — PV of coupons:**
```
PV_coupons = 15 × [1 - (1.025)^(-20)] / 0.025
           = 15 × [1 - 0.6103] / 0.025
           = 15 × [0.3897 / 0.025]
           = 15 × 15.589
           = 233.84
```

**Step 2 — PV of par:**
```
PV_par = 1000 / (1.025)^20
       = 1000 / 1.6386
       = 610.27
```

**Step 3 — Bond price:**
```
P = 233.84 + 610.27 = 844.11
```

The bond trades at **\$844.11**, a discount to its \$1,000 face value. Why? Because the coupon rate (3%) is less than the market yield (5%). An investor buying at \$844.11 and receiving 3% coupons on a \$1,000 face earns an *effective* return of 5% — the extra return comes from the capital gain of getting \$1,000 at maturity despite paying only \$844.11.

**The insight:** Price and yield are inversely related. When the market demands a higher yield (because rates rose), it pays a lower price for the fixed cash flows. When rates fall, the price rises. The cash flows don't change — only the discount rate does.

---

## Yield to Maturity vs. Other Yield Measures

### Three different "yields" — three different questions

Bond traders and analysts talk about yield constantly, but they can mean three different things. Getting these confused leads to real mistakes.

**1. Coupon rate**: The interest rate printed on the bond when it was issued. A bond with a \$50 annual coupon and \$1,000 face has a 5% coupon rate. This rate never changes — it is fixed at issuance. The coupon rate tells you what the *issuer pays* on face value, not what *you* earn.

**2. Current yield**: Annual coupon ÷ current price. If that 5% bond now costs \$900, the current yield is \$50/\$900 = 5.56%. Better than the coupon rate, because you capture the coupon on a cheaper price. But it ignores the \$100 capital gain you'll earn when the bond matures at \$1,000. So current yield overstates your return if the bond is at a discount, and understates it if at a premium.

**3. Yield to maturity (YTM)**: The one rate that equates the present value of all future cash flows to today's price. It is the *total* annualized return you earn if you: (a) buy at today's price, (b) receive all coupons, (c) reinvest coupons at that same YTM, and (d) hold to maturity. YTM is the gold standard measure of bond return.

#### Worked example:

The same 5% coupon bond (\$1,000 face, 10 years remaining) now trades at \$900. Let's compare all three yields:

- **Coupon rate**: 5% (fixed, unchanged)
- **Current yield**: \$50 / \$900 = **5.56%**
- **YTM**: the rate *y* that solves:

```
900 = 50/(1+y) + 50/(1+y)^2 + ... + 50/(1+y)^10 + 1000/(1+y)^10
```

Solving numerically (a financial calculator or spreadsheet), *y* ≈ **6.15%**

The YTM is the highest of the three, because it accounts for both the annual \$50 coupons *and* the \$100 capital gain from buying at \$900 and receiving \$1,000 at maturity.

**The insight:** Current yield is a quick-and-dirty approximation. For any serious analysis, use YTM — it is the rate that makes the present-value equation balance, accounting for *all* cash flows including the final par payment.

![US 10-Year Treasury yield 2010-2024 showing rate cycle](/imgs/blogs/bond-valuation-yield-duration-convexity-2.png)

The chart above shows how much yields moved in the real world. The US 10-year Treasury yield fell from 3.29% at end-2010 to a COVID-era low of 0.93% at end-2020, then surged to 4.57% at end-2024 (Federal Reserve H.15). That 364-basis-point rise between 2020 and 2024 translated into massive bond price *declines* for anyone holding long-duration bonds — a direct application of the pricing formula.

---

## Price-Yield Relationship: Inverse and Convex

### Why they move in opposite directions

This is perhaps the most important mechanical insight in all of fixed income: **bond prices and yields move in opposite directions.** Here is why, intuitively:

Imagine you own a bond paying 3% per year. The market rate then rises to 5%. Your bond is now *less attractive* than new bonds — why would anyone pay full price for 3% when they can buy a new bond at 5%? They wouldn't. Your bond's price drops until it is cheap enough that the combination of its fixed \$30 coupons and the par-value capital gain makes the total return competitive with the 5% market rate. The price must fall to compensate for the below-market coupon.

Run the logic in reverse: when market rates fall, your existing 3% coupon looks generous compared to new bonds at lower rates. Demand rises, and so does your bond's price.

### The price-yield curve

If you plot bond price on the y-axis and yield on the x-axis, you don't get a straight line — you get a **curve that bows toward the top-left**. This shape is called **convexity**, and it has a crucial practical implication:

![Bond price vs yield curve showing inverse and convex relationship](/imgs/blogs/bond-valuation-yield-duration-convexity-3.png)

Look at what the chart shows. When yield equals the coupon rate (3%), the price equals par (\$1,000). As yield rises above the coupon rate, price falls — but the curve bows upward (convex). As yield falls below the coupon rate, price rises — and again, the curve bows upward. The curve is always above any straight-line (tangent) approximation.

Why does this matter? Because it means a bond *gains more in price* when yields fall by 1% than it *loses* when yields rise by 1%. The math of convexity works in bondholders' favor.

At a 1% yield, our 3% bond is worth about \$1,193. At a 3% yield (par), it is \$1,000. At a 7% yield, it is about \$789. From 3% to 1% (yield drops 200 bps), the bond gains \$193. From 3% to 5% (yield rises 200 bps), the bond loses \$156. Asymmetric, and the asymmetry benefits the bondholder. That is the convexity gift.

---

## Duration: Measuring Price Sensitivity

### What duration actually is

Duration answers one practical question: **if yields change by 1%, how much does my bond's price change?**

There are two related but distinct duration concepts you need to know:

**Macaulay duration** is the weighted-average time until you receive all the bond's cash flows. Coupons received early reduce duration (you don't wait as long); larger future cash flows increase it. For a zero-coupon bond (no coupons, just par at maturity), Macaulay duration exactly equals the maturity in years. For a coupon bond, it is always less than maturity.

**Modified duration** is Macaulay duration adjusted for the yield level, and it directly gives you the price sensitivity:

```
Modified Duration = Macaulay Duration / (1 + y/m)
```

Where *y* is the annual yield and *m* is the number of coupon periods per year (typically 2 for semi-annual).

The price-change approximation using modified duration:

```
ΔP/P ≈ -Modified Duration × Δy
```

Or in dollar terms:

```
ΔP ≈ -Modified Duration × P × Δy
```

#### Worked example:

A 10-year, \$1,000 bond with a 3% semi-annual coupon, currently priced at \$844.11 (from our earlier example), has a YTM of 5% and a Macaulay duration of approximately 8.62 years.

**Modified duration:**
```
Modified Duration = 8.62 / (1 + 0.05/2) = 8.62 / 1.025 = 8.41 years
```

**If yields rise by 50 basis points (0.50%):**
```
ΔP ≈ -8.41 × 844.11 × 0.005 = -35.49
New price ≈ 844.11 - 35.49 = 808.62
```

**If yields fall by 50 basis points (0.50%):**
```
ΔP ≈ -8.41 × 844.11 × (-0.005) = +35.49
New price ≈ 844.11 + 35.49 = 879.60
```

(Actual computed prices will differ slightly due to convexity — the linear approximation understates the gain and overstates the loss. More on this shortly.)

**The insight:** Modified duration tells you your "price per 1% yield change" ratio. A bond with duration 8 loses roughly 8% of its price for every 1% rise in yield. A bond with duration 2 loses only 2%. Duration is the bond's "interest rate risk lever."

#### Worked example: computing Macaulay duration from first principles

Take our benchmark bond: 3-year, \$1,000 face, 5% annual coupon (one payment per year), YTM = 5% (so the bond is at par, \$1,000).

The Macaulay duration is the weighted average of the time of each cash flow, with weights equal to that cash flow's present value as a fraction of total bond price.

| Year | Cash Flow | PV at 5% | PV Weight | Year × Weight |
|---|---|---|---|---|
| 1 | \$50 | \$50 / 1.05 = \$47.62 | 47.62 / 1,000 = 0.04762 | 1 × 0.04762 = 0.04762 |
| 2 | \$50 | \$50 / (1.05)² = \$45.35 | 45.35 / 1,000 = 0.04535 | 2 × 0.04535 = 0.09070 |
| 3 | \$1,050 | \$1,050 / (1.05)³ = \$907.03 | 907.03 / 1,000 = 0.90703 | 3 × 0.90703 = 2.72109 |
| **Total** | | **\$1,000.00** | **1.00000** | **2.85941** |

**Macaulay Duration = 2.86 years**

**Modified Duration:**
```
Modified Duration = 2.86 / (1 + 0.05) = 2.86 / 1.05 = 2.724 years
```

**What does 2.86 years mean?** On a 3-year bond, the average dollar arrives in 2.86 years — not 3 — because most of the par value comes back at year 3, but the year 1 and year 2 coupons pull the weighted average slightly forward. For a zero-coupon bond with no coupons at all, duration would equal exactly 3.0 (all the money arrives at the single maturity date).

**Modified duration interpretation:** A 1% rise in yield reduces the bond's price by approximately 2.72%. At a price of \$1,000:
```
ΔP ≈ -2.724 × 1,000 × 0.01 = -$27.24
New price ≈ $972.76 (actual: $972.77 — nearly perfect)
```

### What drives duration?

Three factors determine a bond's duration:

1. **Maturity**: Longer-maturity bonds have higher duration — you wait longer for cash flows, so each cash flow's present value is more sensitive to rate changes.

2. **Coupon rate**: Higher-coupon bonds have *lower* duration — you receive more cash earlier (as coupons), reducing the weighted-average time. A zero-coupon bond has the highest possible duration (equal to its maturity). This is why the SVB portfolio problem was so acute: they had loaded up on long-maturity, low-coupon bonds bought during the 2020–2021 rate nadir — maximizing duration.

3. **Yield level**: Higher yields reduce duration (slightly), because higher discount rates shrink the weight of distant cash flows relative to near cash flows.

For the discount-rate framework connecting duration to all valuation, see [discount rates in practice](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta).

### Duration in portfolio management

Duration is the primary tool professional bond portfolio managers use to control interest rate risk. Here is how it is applied in practice.

**Measuring portfolio duration:** A portfolio's duration is the market-value-weighted average of the individual bond durations:

```
Portfolio Duration = Σ (Market Value of Bond_i / Total Portfolio Value) × Duration_i
```

If you hold \$5 million in a 10-year Treasury (duration 8.5) and \$5 million in a 2-year Treasury (duration 1.9):

```
Portfolio Duration = 0.5 × 8.5 + 0.5 × 1.9 = 4.25 + 0.95 = 5.20 years
```

A 1% yield move shifts the portfolio by approximately 5.20%.

**Duration targeting:** Liability-driven investing (LDI) — used by pension funds and insurance companies — involves matching asset duration to the duration of liabilities. A pension fund with obligations stretching 20 years into the future has liabilities with a duration around 15–20 years. It should hold assets with a similar duration. If it holds short-duration assets (say, 5 years) and rates fall, liabilities rise more than assets — a funding shortfall emerges. This mismatch is exactly what SVB suffered, except its "liabilities" were demand deposits (effectively zero duration) while its assets were long-duration bonds.

**Duration hedging with futures:** Bond portfolio managers often use Treasury futures to adjust duration without buying or selling bonds (which is costly in commissions and bid-ask spread). The number of futures contracts needed to increase duration by ΔD:

```
N = (ΔD × Portfolio Value) / (Duration_futures × Futures Price × Contract Size)
```

#### Worked example: duration adjustment via futures

A fund manager has a \$100 million portfolio with duration 6.0 years. She wants to reduce duration to 4.0 years ahead of a Fed meeting where she expects a rate hike. She uses 10-year Treasury note futures, which have a duration of approximately 8.5 years per contract, priced at \$110,000 per contract.

**Contracts to short:**
```
N = (6.0 - 4.0) × 100,000,000 / (8.5 × 110,000)
  = 200,000,000 / 935,000
  ≈ 214 contracts to short
```

By shorting 214 contracts, the manager effectively removes 2 years of duration from the portfolio. If yields rise 50 bps, the portfolio now loses approximately:

```
Without hedge: -6.0% × 0.5% = -3.0% → -$3,000,000
With hedge (net duration 4.0): -4.0% × 0.5% = -2.0% → -$2,000,000
Hedge saved: $1,000,000 in this scenario
```

**Duration and the yield curve shape:** Duration is not just about parallel yield shifts — a single yield move applied to all maturities equally. In the real world, the yield curve *twists*: short rates might rise while long rates fall (a flattening), or the whole curve might pivot. Portfolio managers decompose their duration exposure into:

- **Key rate durations (KRDs)**: the sensitivity of the portfolio to yield changes at specific maturities (2-year, 5-year, 10-year, 30-year). This is important because a portfolio barbell (50% in 2-year + 50% in 30-year) has the same *average* duration as a bullet (100% in 10-year) but very different responses to a yield curve twist.
- **DV01 ladders**: the dollar change in portfolio value for a 1 basis point yield move at each maturity point. Traders use this to spot where their curve exposure is concentrated.

### Yield curve shape and bond pricing

The shape of the yield curve profoundly affects bond pricing beyond just the level of rates.

**Normal (upward-sloping) curve:** Long-dated bonds yield more than short-dated bonds. This is the typical environment — investors demand extra return for locking up money longer. In this environment, a buy-and-hold investor in a 10-year bond earns extra yield versus rolling 2-year bonds. The extra yield is called the **term premium** — compensation for duration risk.

**Inverted curve:** Short rates exceed long rates, as happened dramatically in 2022–2023 (the US 2-year hit 5.1% while the 10-year sat at 3.8%). An inverted curve signals the market expects rates to fall (because the economy slows and the Fed cuts). For bond investors, the implication is counterintuitive: long-dated bonds yield *less* than short-dated bonds, yet they still have higher price sensitivity (higher duration). Whether to hold long or short depends on whether you believe rates will fall quickly enough to compensate.

**Humped curve:** Yields peak at intermediate maturities (say, 5–7 years) and fall at both the short and long ends. This sometimes occurs when the market expects a series of near-term rate hikes followed by cuts — a classic late-cycle pattern.

#### Worked example: how curve shape affects two-bond comparison

Suppose an investor compares two strategies in early 2024, when the yield curve was mildly inverted:

- **Strategy A:** Buy a 2-year Treasury at 4.80%, modified duration 1.90 years. Hold to maturity.
- **Strategy B:** Buy a 10-year Treasury at 4.20%, modified duration 8.50 years. Hold for one year, then sell.

**Strategy A — return (hold to maturity):**
```
Annual return: 4.80% (all coupon, no price risk if held to maturity)
Annualized return over 2 years: 4.80%
```

**Strategy B — return (one-year horizon, rates unchanged):**
```
Coupon income: 4.20%
Price change (yield curve roll-down): the bond, now a 9-year bond, is repriced at the
  9-year yield. If the curve is flat at 4.20%, no price change.
Total return: ≈ 4.20%
```

**Strategy B — return (one-year horizon, rates fall 50 bps):**
```
Coupon income: 4.20%
Price change: -8.50 × (-0.005) = +4.25%
Total return: 4.20% + 4.25% = 8.45%
```

**Strategy B — return (one-year horizon, rates rise 50 bps):**
```
Coupon income: 4.20%
Price change: -8.50 × 0.005 = -4.25%
Total return: 4.20% - 4.25% = -0.05% (essentially flat)
```

The analysis shows: in late 2024, picking between the 2-year and 10-year was fundamentally a bet on whether rates would fall (favoring the 10-year) or stay flat/rise (favoring the 2-year, which paid more and carried no price risk). Understanding duration converts this vague judgment into precise numbers.

---

## Convexity: The Curvature Advantage

### Duration's shortcoming

Modified duration gives a linear approximation of price sensitivity. But the actual price-yield relationship is a curve (the convex one we plotted earlier). For small yield changes (say, 10 bps), the linear approximation is excellent. For large yield changes (100 bps or more), it starts to diverge meaningfully.

The error always works the same way: the linear approximation *underestimates* the actual price. Why? Because the curve bows upward. The true price is always *above* the tangent line.

**Convexity** quantifies that curvature. The full price-change formula adding convexity is:

```
ΔP/P ≈ -D_mod × Δy  +  (1/2) × Convexity × (Δy)²
```

The convexity term is always positive (for a standard bond), so it always adds to the price change — it softens losses when yields rise and amplifies gains when yields fall.

![Duration linear estimate vs convexity-adjusted actual price change](/imgs/blogs/bond-valuation-yield-duration-convexity-4.png)

### Computing convexity: the formula

Convexity is the second derivative of the price-yield function, divided by price. For a bond with *n* semi-annual periods, face value *F*, semi-annual coupon *C*, and semi-annual yield *y*:

```
Convexity = [1 / (P × (1+y)²)] × Σ [t(t+1) × CF_t / (1+y)^t]
```

Where the sum runs from *t* = 1 to *n*, *CF_t* is the cash flow in period *t* (coupon or coupon + par), and the sum is then divided by *m²* = 4 (to convert from semi-annual periods to annual convexity).

In practice, you almost never compute convexity by hand — Bloomberg, Excel, or any bond calculator returns it directly. But understanding the formula reveals why higher-maturity and lower-coupon bonds have higher convexity: the *t(t+1)* weights grow quadratically with time, so distant cash flows — which dominate in low-coupon, long-maturity bonds — contribute disproportionately to convexity.

#### Worked example: full convexity calculation for a 5-year bond

Take a 5-year bond, \$1,000 face, 4% annual coupon paid semi-annually, currently priced at par (YTM = 4%, semi-annual yield = 2%, 10 semi-annual periods).

Cash flows: \$20 per semi-annual period for periods 1–9, then \$1,020 (coupon + par) in period 10.

**Step 1 — Compute the weighted sum Σ [t(t+1) × CF_t / (1.02)^t]:**

| Period t | CF_t | (1.02)^t | CF_t / (1.02)^t | t(t+1) | t(t+1) × CF_t / (1.02)^t |
|---|---|---|---|---|---|
| 1 | 20 | 1.0200 | 19.61 | 2 | 39.22 |
| 2 | 20 | 1.0404 | 19.22 | 6 | 115.34 |
| 3 | 20 | 1.0612 | 18.85 | 12 | 226.15 |
| 4 | 20 | 1.0824 | 18.48 | 20 | 369.54 |
| 5 | 20 | 1.1041 | 18.12 | 30 | 543.47 |
| 6 | 20 | 1.1262 | 17.76 | 42 | 745.94 |
| 7 | 20 | 1.1487 | 17.41 | 56 | 974.88 |
| 8 | 20 | 1.1717 | 17.07 | 72 | 1,229.14 |
| 9 | 20 | 1.1951 | 16.74 | 90 | 1,506.12 |
| 10 | 1,020 | 1.2190 | 836.75 | 110 | 92,042.61 |

Sum ≈ 97,792.41

**Step 2 — Divide by [P × (1+y)²] = [1,000 × (1.02)²] = 1,040.4:**

```
Convexity (semi-annual) = 97,792.41 / 1,040.4 = 93.99
```

**Step 3 — Convert to annual convexity (divide by m² = 4):**

```
Annual Convexity ≈ 93.99 / 4 = 23.5
```

**Step 4 — Apply the full price-change formula for a 200 bps yield shock:**

Modified duration for this bond ≈ 4.45 years. If yield rises from 4% to 6% (Δy = 0.02):

```
Duration contribution: -4.45 × 0.02 = -8.90%
Convexity contribution: (1/2) × 23.5 × (0.02)² = +0.47%
Total estimated price change: -8.90% + 0.47% = -8.43%

Estimated new price: 1,000 × (1 - 0.0843) = $915.70
Actual computed price at 6% YTM: $914.70 (via full discounting formula)
```

The convexity-adjusted estimate of \$915.70 is within \$1 of the actual price — versus a duration-only estimate of \$911.00, which is off by nearly \$4. For a \$10 million portfolio, that \$4 difference scales to a \$40,000 gap in your risk estimate. Convexity is not a footnote; it is material.

#### Worked example: pricing a 10-year 5% coupon bond when yield = 6%

This is the clean, self-contained example many textbooks assign — let's do it with full transparency:

**Bond parameters:**
- Face value: \$1,000
- Annual coupon rate: 5%, paid semi-annually → \$25 per period
- Maturity: 10 years → 20 semi-annual periods
- Market YTM: 6% per year → 3% per semi-annual period (Δy = 0.03)

**Step 1 — PV of coupon annuity:**
```
PV_coupons = 25 × [1 - (1.03)^(-20)] / 0.03
           = 25 × [1 - 0.5537] / 0.03
           = 25 × [0.4463 / 0.03]
           = 25 × 14.877
           = $371.93
```

**Step 2 — PV of par:**
```
PV_par = 1,000 / (1.03)^20
       = 1,000 / 1.8061
       = $553.68
```

**Step 3 — Bond price:**
```
P = 371.93 + 553.68 = $925.61
```

The bond trades at a **\$74.39 discount** to par. Why? The market requires 6% but the bond only pays 5% in coupons. Investors compensate themselves by paying less than face value — the \$74.39 capital gain they earn at maturity (buying at \$925.61, receiving \$1,000) makes up the yield shortfall.

**Sanity check with the dollar shortfall:** The bond pays \$50/year in coupons on a \$1,000 face; at 6% YTM on \$1,000 face an investor would expect \$60/year. The shortfall is \$10/year for 10 years. The present value of \$10/year for 10 years at 6% is approximately \$73.60 — close to the \$74.39 discount, confirming the intuition.

#### Worked example:

The convexity term is always positive (for a standard bond), so it always adds to the price change — it softens losses when yields rise and amplifies gains when yields fall.

Using our 10-year bond with modified duration 8.41 and convexity approximately 87 (a typical value for a 10-year bond at 5% yield):

**Yield rises 1% (100 bps):**
```
Duration contribution: -8.41 × 0.01 = -8.41%
Convexity contribution: (1/2) × 87 × (0.01)² = +0.44%
Total price change: -8.41% + 0.44% = -7.97%
```

**Yield falls 1% (100 bps):**
```
Duration contribution: -8.41 × (-0.01) = +8.41%
Convexity contribution: (1/2) × 87 × (0.01)² = +0.44%
Total price change: +8.41% + 0.44% = +8.85%
```

Compare: when yields *rise* 1%, the bond loses 7.97%. When yields *fall* 1%, the bond gains 8.85%. The convexity term (+0.44%) is the same in both cases — always positive, always adding to the price change.

**The insight:** Convexity is a built-in asymmetry that benefits bondholders. More convex bonds are worth more (other things equal) because they outperform less convex bonds in both rising and falling rate environments. Portfolio managers actively seek bonds with high convexity, especially in volatile rate environments.

### What drives convexity?

Convexity increases with:
- **Longer maturity**: distant cash flows have more curvature in their discount functions
- **Lower coupon**: zero-coupon bonds have the highest convexity for a given maturity
- **Lower yield**: when yields are low, the curvature is more pronounced

A concrete comparison across bond types:

| Bond | Maturity | Coupon | Convexity (approx.) |
|---|---|---|---|
| 2-year Treasury | 2 yr | 5% | ~4 |
| 10-year Treasury | 10 yr | 5% | ~87 |
| 30-year Treasury | 30 yr | 5% | ~350 |
| Zero-coupon 10-year | 10 yr | 0% | ~120 |
| Callable corporate | 10 yr (call @ 5) | 5% | ~30–50 (negative convexity risk near call) |
| Agency MBS | ~7 yr effective | 5% | Negative convexity possible |

Mortgage-backed securities (MBS) have *negative convexity* — we explain this in the bond types section. For callable corporate bonds, convexity can also turn negative at low yields because the issuer will call (redeem early) the bond.

---

## Credit Spread: Pricing Default Risk

### The risk-free rate as a foundation

US Treasury bonds are considered **risk-free** — the US government has never defaulted on a dollar-denominated bond, and it controls its own currency. This makes Treasury yields the universal reference point in fixed income. Every other bond yield is quoted as Treasury yield plus a **credit spread**.

```
Bond Yield = Risk-Free Rate + Credit Spread
```

The credit spread compensates investors for three risks:

1. **Default risk**: the probability the issuer won't pay. Higher probability → wider spread.
2. **Recovery risk**: if there is a default, how much do investors recover? Lower recovery → wider spread.
3. **Liquidity risk**: can you sell the bond quickly without moving the price? Less liquid → wider spread.

### Credit ratings and spreads

Credit rating agencies (Moody's, S&P, Fitch) assign letter grades that compress the credit risk into a single signal:

| Rating | Category | Typical spread above UST (end-2024) |
|---|---|---|
| AAA/Aaa | Highest investment grade | 30–60 bps |
| AA/Aa | High investment grade | 50–100 bps |
| A | Upper medium grade | 80–150 bps |
| BBB/Baa | Lower investment grade | 120–250 bps |
| BB/Ba | Speculative ("junk") | 250–450 bps |
| B | Highly speculative | 400–700 bps |
| CCC/Caa and below | Very high risk | 700–1500+ bps |

The line between BBB (investment grade) and BB (junk) matters enormously — institutional investors like pension funds and insurance companies are often prohibited from holding below-investment-grade bonds. When a bond is downgraded from BBB to BB (a "fallen angel"), forced selling can push the spread dramatically wider.

![Credit spread stack: risk-free Treasury plus IG and HY spreads 2010-2024](/imgs/blogs/bond-valuation-yield-duration-convexity-5.png)

#### Worked example:

In January 2025, the US 10-year Treasury yield stood at approximately 4.6%. A BBB-rated 10-year corporate bond from a large US company traded at a spread of roughly 150 bps (1.50%) above Treasuries, putting its yield at approximately 6.1%.

A bond investor buying \$100,000 face value of this corporate bond:

**Annual coupon income:**
```
Coupon rate ≈ 5.5% (set at issuance, when rates were lower)
Annual coupon = 0.055 × 100,000 = $5,500
```

**YTM at current price (yield = 6.1%):**
```
If coupon rate is 5.5% and YTM is 6.1%, the bond trades at a discount.
Simplified: price ≈ par × (coupon rate / YTM) ≈ 100,000 × (5.5/6.1) ≈ $90,164
(actual calculation uses the full discounting formula)
```

**Additional yield over Treasury:**
```
Credit spread income = 1.50% × 100,000 = $1,500/year extra vs. a Treasury
```

**The insight:** The credit spread is not "free money." It is compensation for the risk that the company might default and you receive less than the face value. For BBB companies in stressed conditions, default rates can rise sharply — investors who bought the spread also "sold" default protection, whether or not they thought of it that way.

---

## Pricing Different Bond Types

### Government bonds: the risk-free baseline

**US Treasury bonds, notes, and bills** are priced exactly by the formula above — no credit spread, pure interest rate risk only. The US Treasury market is the world's most liquid bond market, with about \$27 trillion outstanding as of 2024 (US Treasury Department). Yields are directly set by Federal Reserve policy (for the short end) and by inflation expectations and growth expectations (for the long end).

For a refresher on how the Fed influences rates, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

Treasury Inflation-Protected Securities (**TIPS**) add one wrinkle: the principal adjusts with the Consumer Price Index. You still discount the cash flows, but the cash flows themselves grow with inflation. TIPS are priced at a **real yield** — the yield after subtracting expected inflation.

**Non-US government bonds** follow the same math but carry an additional **sovereign credit spread** if the country's creditworthiness is questioned. German Bunds trade as the European risk-free equivalent; Italian BTPs trade at a spread above Bunds because Italy carries more fiscal risk.

### Corporate bonds: adding credit risk

Corporate bonds price identically to Treasuries — PV of coupons plus PV of par, discounted at the YTM. The difference is that the YTM includes a credit spread. The broader valuation framework (relating to different asset types) is covered in the [valuation spectrum post](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims).

Two additional complications for corporate bonds:

**Call provisions**: Many corporate bonds are callable — the issuer can redeem the bond early at a specified price (the call price, typically at or above par). This is valuable to the issuer (they can refinance if rates fall) and harmful to the investor (the bond gets called away just when it was rising in value). For callable bonds, analysts use **yield to call (YTC)** — the YTM assuming the bond is called at the earliest call date. When rates are low and the bond trades at a premium, YTC < YTM, and YTC is the more relevant measure.

**Covenants**: Legal restrictions in the bond indenture that protect holders — for example, limits on additional debt, restrictions on dividends, or maintenance of financial ratios. Bonds with stronger covenants trade at tighter spreads.

![Bond market taxonomy: government, corporate, MBS, and municipal bonds](/imgs/blogs/bond-valuation-yield-duration-convexity-6.png)

### Mortgage-backed securities: prepayment risk

A mortgage-backed security (MBS) is a pool of residential mortgages packaged into a bond. Homeowners' monthly mortgage payments become the coupon stream; the gradual payoff of principal becomes an ongoing stream of principal returns (not just at maturity).

The central complication: **prepayment risk**. When rates fall, homeowners refinance — they pay off their mortgages early and take out new ones at lower rates. This is exactly the wrong outcome for MBS investors: the bond gets "called away" just when it was rising in price (exactly like a callable corporate bond). This gives MBS **negative convexity** at low yields.

When rates rise, homeowners stop refinancing (they're locked into their low-rate mortgages), so the MBS extends in maturity — giving you a longer-duration bond in a rising-rate environment. That's also bad for investors. MBS investors face a double whammy: extension when rates rise and prepayment when rates fall. This is priced in via an **option-adjusted spread (OAS)** — the spread over Treasuries after adjusting for the value of the embedded prepayment option.

### Municipal bonds: the tax adjustment

Municipal bonds ("munis") are issued by US state and local governments. Their defining characteristic: coupon interest is typically exempt from federal income tax, and sometimes state income tax too.

This tax exemption makes munis *yield less* than comparably rated corporate bonds — and that lower yield is still competitive on an after-tax basis. The key calculation:

```
Tax-equivalent yield = Muni yield / (1 - marginal tax rate)
```

#### Worked example:

A AAA-rated municipal bond yields 3.2%. A corporate bond of similar risk yields 4.8%.

For an investor in the 37% federal tax bracket:

```
Tax-equivalent muni yield = 3.2% / (1 - 0.37) = 3.2% / 0.63 = 5.08%
```

The muni at 3.2% is equivalent (after tax) to a taxable bond yielding 5.08%. Since the corporate bond only yields 4.8%, the muni is the better choice for this high-bracket investor.

For an investor in the 12% bracket:

```
Tax-equivalent muni yield = 3.2% / (1 - 0.12) = 3.2% / 0.88 = 3.64%
```

The muni's tax-equivalent yield of 3.64% is less than the corporate's 4.8%. The corporate bond is the better choice.

**The insight:** Muni yields are a function of credit quality *and* the investor's tax bracket. The higher your tax rate, the more valuable the tax exemption — which is why munis are disproportionately held by high-net-worth individuals and some insurance companies.

---

## Common Misconceptions

### Misconception 1: "A bond is always safer than a stock"

This is contextually true but mechanically false. A bond's price can fall dramatically — and more predictably — than many stocks in the right environment. Consider: a 30-year US Treasury bond has a modified duration of roughly 18 years. A 1% rise in yield drops its price by approximately 18%. That is larger than many stock corrections.

What bonds *do* provide over stocks: seniority in bankruptcy (bondholders are paid before equity holders), contractually fixed cash flows, and a defined maturity date when par is returned. These are valuable properties — but they do not immunize a bond from interest rate risk.

### Misconception 2: "The coupon rate is your return"

The coupon rate tells you what the issuer pays on face value. Your *actual* return (YTM) depends on what you paid for the bond. If you bought at \$900 and the coupon rate is 5% on \$1,000 face, you receive \$50/year on a \$900 investment — that's a 5.56% current yield — *plus* a \$100 capital gain at maturity. Your total YTM is around 6.15%.

If you paid \$1,100 for the same bond, you receive \$50/year on \$1,100 (4.55% current yield) *minus* the \$100 capital loss at maturity. Your YTM is around 3.9%, well below the 5% coupon rate.

Always think in YTM when evaluating bond returns.

### Misconception 3: "Duration is measured in dollars"

Duration is measured in *years*. Macaulay duration is literally the weighted-average time (in years) until you receive the bond's cash flows. Modified duration is slightly different (it equals Macaulay duration divided by 1 plus the per-period yield), but it is still dimensionless in terms of the price percentage: it tells you the *percentage price change* per *unit yield change*.

The dollar sensitivity — how many dollars the price changes for a 1 basis point yield move — is called the **dollar value of a basis point (DV01)** or **price value of a basis point (PVBP)**:

```
DV01 = Modified Duration × Price × 0.0001
```

For our \$844.11 bond with modified duration 8.41:
```
DV01 = 8.41 × 844.11 × 0.0001 = $0.71 per $1,000 face per 1 bp yield move
```

Portfolio managers use DV01 to measure and hedge interest rate risk.

### Misconception 4: "A higher-yield bond is always a better investment"

Higher yield = higher risk. A bond yielding 12% does so because the market prices in a meaningful probability of default. If the company defaults and recovery is 40 cents on the dollar, the holder loses 60% of principal — wiping out years of coupon income.

The expected return on a junk bond is not its stated yield — it's the yield minus the expected loss from default. A 12% yield with a 3% annual expected default loss rate and 40% recovery nets you closer to 12% - (3% × 60%) = 10.2% in expected value, and with more volatility and downside risk than a Treasury.

### Misconception 5: "Reinvesting coupons at the YTM is realistic"

YTM assumes you reinvest every coupon payment at the same YTM. In reality, when you receive a coupon, market rates have changed — you reinvest at *whatever rate prevails then*. If rates have fallen, your reinvestment return is lower than the YTM implied. If rates have risen, it's higher.

This is called **reinvestment risk**, and it's the mirror image of price risk: rising rates hurt bond prices but help reinvestment; falling rates help bond prices but hurt reinvestment. The only bond immune to reinvestment risk is a zero-coupon bond — it makes no interim payments, so there's nothing to reinvest.

---

## How It Shows Up in Real Markets

### The 2022 bond market crash: duration risk at 400 basis points

The US 10-year Treasury yield at end-2021 was 1.52%. By end-2022 it had reached 3.88% — a rise of 236 basis points in a single year. The full move from the 2020 low of 0.93% to the 2023 peak of approximately 5.0% was 407 basis points (Federal Reserve H.15). Let's price that through the duration-convexity framework precisely.

**A 10-year Treasury at end-2020 (duration baseline):**
- Face value: \$1,000; Coupon: 0.93% (just-issued, at-par bond); YTM: 0.93%
- Price: \$1,000 (par, newly issued)
- Modified duration: approximately 9.6 years (low coupon + 10-year maturity)
- Convexity: approximately 110

**When yield rises to 5.0% in late 2023 (Δy = +4.07%):**

```
Duration contribution: -9.6 × 0.0407 = -39.1%
Convexity contribution: (1/2) × 110 × (0.0407)² = +9.1%
Total price change: -39.1% + 9.1% = -30.0%

Estimated new price: $1,000 × (1 - 0.300) = $700
```

The actual computed price of a 0.93% coupon 10-year Treasury discounted at 5.0% yield is approximately \$694 — a \$306 loss on every \$1,000 bond, a 30.6% loss, with no default, no fraud, and no change in the bond's contractual terms. The loss was 100% interest rate risk, precisely predicted by the duration-convexity formula.

For a 30-year Treasury (duration ~20 years), the same rate move implied:

```
Duration contribution: -20 × 0.0407 = -81.4%
Convexity contribution: (1/2) × 350 × (0.0407)² = +29.0%
Total price change: -81.4% + 29.0% = -52.4%
```

A 30-year Treasury bought at the 2020 low of 1.65% coupon lost roughly half its market value by 2023. Not because of any credit event — purely duration. The Bloomberg US Aggregate Bond Index, which is dominated by investment-grade bonds, lost 13.0% in 2022 alone — its worst year since at least 1976. Long-duration Treasury ETFs lost 25–30% in that single year. "Safe" bonds delivered some of the worst returns in a generation, entirely because of interest rate risk — not credit risk.

![Risk vs return: bonds vs other asset classes 2000-2024](/imgs/blogs/bond-valuation-yield-duration-convexity-7.png)

Over the full 2000–2024 period (JP Morgan Guide to the Markets Q1 2025), US bonds delivered 3.8% annual return with 5.9% standard deviation — the lowest risk profile among major asset classes. But within that period, year-to-year volatility could be dramatic, driven entirely by yield movements.

### The 2020 COVID rate shock: the benefit of duration

The inverse: when the Fed cut rates to near zero in 2020, the US 10-year yield fell from 1.92% (end-2019) to 0.93% (end-2020) — a 99 basis point decline. Long-duration bonds surged. A 30-year Treasury with duration ~20 would have gained roughly:

```
ΔP/P ≈ -20 × (-0.0099) + (1/2) × convexity × (0.0099)²  ≈ +19.8% + convexity bump ≈ +20–22%
```

The duration and convexity math, precisely as described in this post, drove the gains.

### SVB: the danger of mismatched duration

Silicon Valley Bank held a portfolio of long-duration Agency MBS and Treasuries that it had purchased at 2020–2021 rates. When rates surged in 2022, the portfolio fell dramatically in value — creating unrealized losses of more than \$15 billion against a capital base of around \$16 billion. The bank had failed to hedge its duration exposure. When depositors started withdrawing en masse and the bank was forced to *realize* those losses by selling bonds, the bank failed in March 2023.

The lesson is not unique to banks: any fixed-income investor who ignores duration — who fails to measure how much their portfolio moves per 1% yield change — is flying blind.

### The yield curve and relative value

In practice, bonds of different maturities are priced against each other via the **yield curve** — the graph of yield against maturity. A normal yield curve slopes upward (longer maturities yield more). When the curve inverts (short-term rates exceed long-term rates), it often signals a recession — investors are willing to accept lower long-term yields because they expect rates to fall as the economy slows.

Relative-value traders compare individual bond yields against the yield curve to identify mispriced securities. If a 7-year bond yields more than the curve implies (given interpolation between the 5-year and 10-year), it might be "cheap" relative to its maturity peers and represent a buying opportunity. This is the domain of yield curve analysis covered in depth at [the yield curve explained](/blog/trading/quantitative-finance/yield-curve-explained).

#### Worked example: riding the yield curve

An investor in early 2023 observes:
- 2-year Treasury yield: 4.8%
- 10-year Treasury yield: 3.97% (inverted curve — short rates higher than long rates)
- 2-year modified duration: ~1.9 years
- 10-year modified duration: ~8.5 years

**Bull case (rates fall):** If the 10-year yield falls from 3.97% to 3.5% (47 bps decline):
```
10-year price gain ≈ -8.5 × (-0.0047) = +4.0%
Plus 3.97% coupon income
Total return ≈ ~8.0% for the year
```

**Bear case (rates rise):** If the 10-year yield rises from 3.97% to 4.5% (53 bps rise):
```
10-year price loss ≈ -8.5 × 0.0053 = -4.5%
Plus 3.97% coupon income
Total return ≈ -0.5% for the year (a small loss despite the coupon)
```

The 2-year, meanwhile, would have barely moved in price due to its low duration, but would have yielded 4.8% either way (assuming it is held to maturity). The investor faces a classic duration-vs-yield-pickup tradeoff.

---

### Credit spread widening as a case study: the 2020 COVID shock

Credit spreads are not static — they compress in good times and blow out in stress. In March 2020, as COVID lockdowns hit, US high-yield (junk bond) spreads spiked from roughly 350 bps (early 2020) to more than 1,000 bps in a matter of weeks. Investment-grade spreads moved from 100 bps to over 350 bps.

**What that meant for a BBB corporate bondholder:**

Assume you held a 10-year BBB bond on January 1, 2020:
- Treasury yield: 1.88%
- Credit spread: 120 bps
- Bond yield: 3.08%; Price: approximately par (\$1,000)
- Modified duration: approximately 8.6 years

By March 20, 2020, two things happened simultaneously:
1. **Treasury yields fell** (flight to safety) — 10-year Treasury dropped to 0.76%, a 112 bps rally.
2. **Credit spreads blew out** — BBB spreads widened from 120 bps to 380 bps, a +260 bps move.

Net yield change = -112 bps (rate rally) + 260 bps (spread widening) = **+148 bps net yield rise**.

```
Price change ≈ -8.6 × 0.0148 = -12.7%
New price ≈ $873
```

Despite a massive rally in risk-free Treasuries, the corporate bond *lost* 12.7% because the credit spread widening more than offset the interest rate benefit. This is the key difference between Treasury risk and credit risk: they can move independently, and sometimes in opposing directions.

**The rebound:** The Fed announced its corporate bond purchase program on March 23, 2020. Spreads tightened sharply. By end-2020, BBB spreads were back at approximately 110 bps — tighter than at the start of the year. A holder who stayed through the volatility recovered fully, plus earned the coupon. A holder who panic-sold at the March lows locked in the loss permanently.

---

## Further Reading & Cross-Links

The bond pricing formula is an application of present value discounting — the foundational engine of all valuation. Every term, every cash flow, every risk premium connects back to the framework in [the time value of money engine every valuation model](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model).

The discount rate used for bonds (the yield) is a specific instance of the required return concept. For the full framework of how required returns are built up — including equity cost, WACC, and beta — see [discount rates in practice: WACC, cost of equity, and unlevered beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta).

To see where bonds fit in the landscape of valuation methods across asset classes — absolute vs. relative vs. contingent claims — read [the valuation spectrum: absolute, relative, and contingent claims](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims).

For a deeper technical treatment of bond pricing math, yield curve construction (bootstrapping, Nelson-Siegel), short-rate models (Vasicek, CIR, Hull-White), and OAS analysis for MBS, see the companion series at [bond pricing and yield to maturity](/blog/trading/fixed-income/bond-pricing-yield-to-maturity).

For the macro context — how the Federal Reserve's actions translate into bond yields and ultimately into asset prices across all markets — see [interest rates, bonds, and stocks: the relationship](/blog/trading/macro-trading/interest-rates-bonds-stocks-relationship).

For readers wanting to understand how bond duration interacts with equity valuation — specifically why rising rates punish long-duration growth stocks — the connection is direct: a growth company's earnings are heavily back-weighted (most value arrives many years out), giving it a "long duration" analogous to a zero-coupon bond. When the discount rate rises, both long-dated bonds and long-duration growth stocks fall. This parallel is explored in [equity valuation and the discount rate](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta).

The yield curve's slope — normal, flat, or inverted — is one of the most reliable economic indicators historically available. For the full framework connecting curve shape to recession signals, Fed policy, and cross-asset implications, see [the yield curve explained](/blog/trading/quantitative-finance/yield-curve-explained) and [reading the bond market for macro signals](/blog/trading/macro-trading/interest-rates-bonds-stocks-relationship).

---

## Sources & Further Reading

- **Federal Reserve H.15**: US Treasury yield time series, 2010–2024. [federalreserve.gov/releases/h15](https://www.federalreserve.gov/releases/h15/)
- **JP Morgan Guide to the Markets Q1 2025**: Asset class risk-return data 2000–2024.
- **US Treasury Department**: US Treasury market outstanding amounts, 2024. [treasurydirect.gov](https://www.treasurydirect.gov)
- **Bloomberg US Aggregate Bond Index**: 2022 annual return data.
- **ICE BofA Bond Index series**: Investment-grade and high-yield credit spread approximations.
- Fabozzi, F.J. (2012). *Fixed Income Mathematics, Analysis, and Valuation* (4th ed.). McGraw-Hill — the standard reference for duration, convexity, and OAS computation.
- Tuckman, B. & Serrat, A. (2011). *Fixed Income Securities: Tools for Today's Markets* (3rd ed.). Wiley — rigorous yield-curve and term-structure models.
