---
title: "Bond Pricing: Discounting Cashflows, Conventions, and the Curves That Matter"
date: "2026-05-03"
publishDate: "2026-05-03"
description: "A senior-quant deep dive into bond pricing: discounted-cashflow framework, yield to maturity, clean vs dirty price, duration, convexity, DV01, key-rate DV01s, spreads, embedded options, repo, futures, production architecture, and named failure modes."
tags:
  [
    "bond-pricing",
    "fixed-income",
    "yield-to-maturity",
    "duration",
    "convexity",
    "dv01",
    "key-rate-dv01",
    "z-spread",
    "oas",
    "repo",
    "callable-bonds",
    "mbs",
    "treasuries",
    "quantitative-finance",
    "python",
  ]
category: "trading"
subcategory: "Quantitative Finance / Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

A bond is the simplest derivative instrument in the world. It pays a known stream of cashflows on known dates. There is no payoff function, no expectation, no model uncertainty about *what* will happen at expiry. The only uncertainty is about the discount rate that converts those future cashflows into a present-value price — and that discount rate is the yield curve. *Bonds are model-light, curve-heavy*: the math of pricing is trivial, the engineering of the curve is everything. Senior fixed-income quants spend their careers thinking about curves, not about bonds.

![Bond pricing: discounting cashflows is the whole game](/imgs/blogs/bond-pricing-1.png)

The diagram above is the mental model. A bond is a deterministic schedule of cashflows. Pricing is summation under discount. The price the desk shows is the discounted sum of those cashflows under the appropriate curve, plus or minus accrued interest, plus or minus an embedded-option adjustment. Everything else in fixed-income — yield to maturity, duration, convexity, spread, key-rate DV01s, repo, futures, OAS — is a tweak on or a derivative of that one calculation.

This article is the deep dive on bond pricing for a senior quant or staff-level engineer. It covers the discounted-cashflow framework, the yield-to-maturity quote convention, clean-vs-dirty price and day-count conventions, the duration / convexity / DV01 risk hierarchy, spreads (Z-spread, OAS) for credit and embedded options, the major bond classes (Treasuries, corporates, munis, MBS, sovereigns, high-yield), embedded options (callable, puttable, convertible), repo as the funding mechanism, bond futures and the cheapest-to-deliver, production architecture, and a long catalog of named bond-pricing failures.

The companion articles are [Yield Curve Modeling](/blog/trading/quantitative-finance/fixed-income/yield-curve-modeling) (which goes deeper on curve construction itself) and [Fixed Income Analytics](/blog/trading/quantitative-finance/fixed-income/fixed-income-analytics) (which covers portfolio-level analytics). [Short-Rate Models](/blog/trading/quantitative-finance/rates-models/short-rate-models-vasicek-hull-white) covers the dynamics that price callable bonds and rate options.

### 0.0 Bonds vs other fixed-income instruments

Before diving in, a quick taxonomy of "fixed income" — bonds are one of several closely-related instrument families:

- **Bonds.** Debt securities with fixed cashflow schedules. The focus of this article.
- **Loans.** Bilateral or syndicated debt; less standardised; pricing similar but often without an active secondary market.
- **Money market.** Short-dated (< 1 year) debt: T-bills, commercial paper, repo, certificates of deposit. Pricing simpler (one discount factor) but volume enormous.
- **Interest-rate swaps.** Exchange of fixed for floating cashflows. Pricing similar to bonds but with par-swap-rate convention.
- **Asset-backed securities (ABS).** Securitised pools of underlying assets (auto loans, credit cards, student loans). Cashflow models vary by collateral type.
- **Mortgage-backed securities (MBS).** Securitised mortgage pools. Discussed in §10.
- **Collateralised loan obligations (CLO).** Tranched ABS. Pricing requires a tranche cashflow waterfall model.
- **Inflation-linked bonds (TIPS).** Indexed to CPI; require a real curve plus inflation expectations.
- **Structured notes.** Hybrid bond + embedded derivative. Pricing in [the exotic derivatives post](/blog/trading/quantitative-finance/exotics/exotic-derivatives) and [autocallables post](/blog/trading/quantitative-finance/exotics/autocallables).

The pricing framework is similar across all of these — discount cashflows under the appropriate curve — but the cashflow modelling and curve selection differ. Senior fixed-income quants are typically deep specialists in one or two of these families.

### 0.1 The size and shape of the global bond market

To set context: the global bond market is roughly $130T in outstanding notional across all currencies and issuers. By comparison, the global equity market is about $90T and the global derivatives market is about $640T (notional, but dominated by interest-rate swaps which net out). Daily trading volume in US Treasuries alone is ~$700B; the next-largest market is JGBs at ~$40B.

A few size-shape facts that frame fixed-income engineering:

- **US Treasuries**: $25T outstanding; centralised at the New York Fed; OTC dealer-to-client and inter-dealer plus electronic platforms (BrokerTec, Tradeweb, FIT). Most liquid bond market in the world.
- **Global investment-grade corporate**: $30T outstanding; predominantly OTC; secondary liquidity varies wildly by issue.
- **Mortgage-backed securities (US)**: $11T outstanding; passes through Fannie Mae / Freddie Mac / Ginnie Mae. The TBA (to-be-announced) market is the most liquid sub-market.
- **Sovereign bonds (ex-US)**: $25T+; varies by country (Germany, Japan, UK, France major issuers).
- **Munis (US)**: $4T; mostly retail-held; less liquid than corporates.
- **High-yield corporate**: $1.5T; active-trader market; higher volatility.
- **Emerging-market sovereign**: $2T+; FX risk, restructuring history.

The bond infrastructure spans more notional than any other asset class. Senior fixed-income quants know they are the silent majority of finance — most of the dollars run through their systems.

## 1. Why bonds are different from options

The first move in any fixed-income engineering conversation is to internalise that bonds are *not* like options. Option pricing is dominated by uncertainty in the underlying; bond pricing is dominated by uncertainty in the discount rate. The pricing engineering is therefore inverted:

- For options, you take the underlying as given and search for a model of its dynamics that gives the right price. Black-Scholes, Heston, SABR, local-vol — every effort goes into modelling the underlying.
- For bonds, you take the cashflow as given and search for the right discount curve. Bootstrapping, Nelson-Siegel-Svensson, splines, multi-curve OIS-LIBOR — every effort goes into modelling the curve.

The implication for a quant team: the curve infrastructure is *the* infrastructure of fixed income. A bond pricing system without a robust curve service is useless; a curve service that publishes coherent, arbitrage-free curves can price most fixed-income products with a thin pricing layer on top.

A second important fact: bonds are mostly *deterministic*. A non-callable Treasury has known cashflows, known dates, no embedded options, no early termination. The only randomness is in the discount rate (which moves daily) and in the credit (which can default). Once you fix today's curve and credit, the price is a closed-form summation. This makes bond pricing fast, auditable, and tractable in ways that option pricing is not.

A third fact: the *risk* of a bond is more nuanced than its price. Duration measures the first-order sensitivity to yield; convexity the second-order; DV01 the dollar P&L per basis point; key-rate DV01s the dollar P&L per basis point at each maturity bucket. A senior fixed-income trader operates on these risk metrics, not on the bond price directly. The price is a derived quantity; the risks are the trade.

A fourth fact: bonds have *funding costs* that options don't. A long bond position is financed via repo; the repo rate is below the unsecured rate by an amount that depends on the haircut and the collateral quality. The funding cost can be a meaningful fraction of the total return, especially in low-yield environments. Senior fixed-income engineering treats funding as a first-class component of the price, not an afterthought.

The final fact: bond markets are enormous and mostly OTC. The US Treasury market alone is $25T+ in outstanding notional; the global investment-grade corporate bond market is another $30T+. Most of this is OTC dealer-to-client business, with thin order books and dealer-quoted spreads. The pricing infrastructure of a fixed-income desk is the daily mark-to-market of an inventory orders of magnitude larger than the firm's option inventory. Even a 1bp pricing error on a multi-billion-dollar book is real money.

## 2. The cashflow schedule

A vanilla coupon bond is fully specified by: the face value (par), the coupon rate, the coupon frequency, the maturity date, the first coupon date, and the day-count convention. From these, the cashflow schedule is deterministic.

![Cashflow schedule: what the bond pays and when](/imgs/blogs/bond-pricing-2.png)

For a $1000 face, 4% annual coupon, semi-annual frequency, 5-year maturity bond:

- Coupons of $20 every 6 months for 5 years (10 coupons total).
- Final cashflow at $T = 5$ is $20 + $1000 = $1020 (last coupon plus face).

The pricing formula:

$$
P = \sum_{i=1}^{N} C_i \cdot D(t_i) + F \cdot D(T)
$$

where $C_i$ is the coupon at time $t_i$, $D(t)$ is the discount factor at time $t$, and $F$ is the face value. For our example with a flat 4.5% YTM and semi-annual compounding:

```python
def vanilla_bond_price(face, coupon_rate, periods_per_year, maturity, ytm):
    """Price a vanilla coupon bond at flat YTM."""
    n = int(maturity * periods_per_year)
    coupon = face * coupon_rate / periods_per_year
    discount = lambda t: 1.0 / (1.0 + ytm / periods_per_year) ** (periods_per_year * t)
    pv = sum(coupon * discount((i + 1) / periods_per_year) for i in range(n))
    pv += face * discount(maturity)
    return pv


print(vanilla_bond_price(face=1000, coupon_rate=0.04, periods_per_year=2,
                          maturity=5, ytm=0.045))
## ≈ 978.0  (bond trades at a discount because YTM > coupon)
```

A few important variations:

- **Zero-coupon bonds** have no intermediate coupons; only one cashflow at maturity. Pricing reduces to $P = F \cdot D(T)$. Treasury bills, STRIPS, and many corporate zero coupons fall here.
- **Floating-rate notes (FRNs)** have coupons that reset based on a reference rate (LIBOR, SOFR). Pricing requires the *forward curve* to estimate future coupons.
- **Inflation-linked bonds (TIPS)** have coupons and principal indexed to CPI. Pricing requires a real curve and inflation expectations.
- **Step-up bonds** have coupons that increase on a schedule. Cashflow schedule has variable $C_i$.
- **Callable bonds** have an embedded short-call to the issuer. Cashflow schedule has uncertain termination.

The vanilla formula generalises to any bond by varying the cashflow schedule and the discount factors.

## 3. Yield to maturity

The yield to maturity (YTM) is the unique constant rate $y$ such that

$$
P = \sum_i \frac{C_i}{(1 + y/m)^{m t_i}}
$$

where $m$ is the compounding frequency. YTM is a *quote convention*: it is the single rate that, used as a flat discount curve, reproduces the bond's market price. Real curves are not flat; YTM is an aggregation of the curve into one number that traders find convenient.

![Yield to maturity: the single rate that reproduces the price](/imgs/blogs/bond-pricing-3.png)

Properties of YTM:

1. **Existence and uniqueness.** For a vanilla coupon bond, YTM is the unique solution in $y > -1/m$ (assuming positive cashflows).
2. **Decreasing in price.** Higher price ⟺ lower yield. Trivial but worth stating: the inverse relationship is the foundation of bond trading intuition.
3. **YTM = coupon rate iff price = par.** When the bond trades at $P = F$, the YTM equals the coupon rate. Above par, YTM < coupon. Below par, YTM > coupon.
4. **Assumes flat curve.** YTM uses one rate for all cashflow discounts, which is wrong if the curve is non-flat. The error is small for short-dated bonds and grows with maturity.
5. **Assumes coupon reinvestment at YTM.** The realised return matches YTM only if every coupon is reinvested at YTM until the bond matures. In practice, coupons are reinvested at prevailing rates which can differ.

When YTM lies:

- **Non-parallel curve shifts.** If the curve steepens, two bonds with similar YTMs but different durations are no longer comparable; the longer-duration bond loses more.
- **Reinvestment risk.** A 30-year bond's realised return depends heavily on what happens to short-term rates over the next three decades; YTM doesn't capture this.
- **Embedded options.** A callable bond's YTM is computed assuming no call; if called, the realised yield is different (yield-to-call vs yield-to-worst).
- **Default risk.** YTM ignores credit; the realised return on a defaulted bond is recovery × face, not YTM.

A senior trader's habit: *quote in YTM, model in curve*. YTM is the universal language for traders ("the 10-year is at 4.5"); the curve is the engineering object for pricing and risk. The two-language fluency is essential.

```python
def yield_to_maturity(price, face, coupon_rate, periods_per_year, maturity,
                     tol=1e-10, max_iter=100):
    """Compute YTM by Newton-Raphson."""
    coupon = face * coupon_rate / periods_per_year
    n = int(maturity * periods_per_year)
    y = 0.05  # initial guess
    for _ in range(max_iter):
        f = sum(coupon / (1 + y/periods_per_year)**(i+1) for i in range(n))
        f += face / (1 + y/periods_per_year)**n
        f -= price
        df = sum(-coupon * (i+1) / periods_per_year /
                 (1 + y/periods_per_year)**(i+2) for i in range(n))
        df -= face * n / periods_per_year / (1 + y/periods_per_year)**(n+1)
        y -= f / df
        if abs(f) < tol:
            return y
    return y
```

The Newton-Raphson converges in 4-6 iterations for typical inputs. Production libraries use a hybrid Newton+bisection scheme for robustness against deep-discount bonds where Newton can diverge.

### 3.1 Yield-to-call and yield-to-worst

For callable bonds, the analyst computes yield-to-call (YTC) — the YTM under the assumption the bond is called at each call date — and reports the *yield-to-worst* (YTW), which is the minimum across YTM and all YTCs. YTW is the conservative yield estimate: it assumes the issuer exercises the option in the way most disadvantageous to the bondholder.

A 5-year corporate bond callable at $102 in years 2, 3, 4 might have:
- YTM (held to maturity): 5.20%
- YTC (called at year 2 at $102): 4.85%
- YTC (called at year 3 at $102): 5.00%
- YTW: 4.85%

The bondholder is conservative; the bond is quoted with YTW. If rates fall such that the issuer's refinancing cost drops below 5.20%, the issuer calls; the realised yield is YTC, not YTM. Senior credit traders quote YTW reflexively for callables.

### 3.2 Compounding conventions

YTM depends on the compounding convention. Three conventions are common:

- **Annual compounding** ($m = 1$): $P = \sum C_i / (1 + y)^{t_i}$.
- **Semi-annual compounding** ($m = 2$): $P = \sum C_i / (1 + y/2)^{2 t_i}$. Standard for US Treasuries.
- **Continuous compounding** ($m \to \infty$): $P = \sum C_i e^{-y t_i}$. Used in pricing models; "academic" convention.

The same bond has different numerical YTMs under different conventions; the price is the same. Senior quants must always check the compounding convention before comparing yields. A 5% annual is roughly 4.94% semi-annual is roughly 4.88% continuous.

```python
def convert_yield_compounding(y_in, m_in, m_out):
    """Convert a yield from m_in compounding to m_out compounding."""
    import math
    if m_in == float("inf"):  # continuous in
        return m_out * (math.exp(y_in / m_out) - 1)
    if m_out == float("inf"):  # continuous out
        return m_in * math.log(1 + y_in / m_in)
    return m_out * ((1 + y_in / m_in) ** (m_in / m_out) - 1)
```

## 4. Clean vs dirty price

Bond prices are quoted *clean* — without accrued interest. Settlement uses the *dirty* price — including accrued interest. The convention is operationally important and frequently confused.

![Clean vs dirty price: the accrued-interest convention](/imgs/blogs/bond-pricing-4.png)

Accrued interest is the portion of the next coupon that has accrued from the last coupon date to the settlement date:

$$
\text{AI} = \frac{\text{days since last coupon}}{\text{days in coupon period}} \cdot \text{coupon}.
$$

The clean price strips this out:

$$
P_{\text{clean}} = P_{\text{dirty}} - \text{AI}.
$$

Why the convention? Because the dirty price has a *saw-tooth* pattern over time: it grows linearly from one coupon date to the next, then drops by the coupon amount on the coupon date. The clean price is smooth, which is what traders prefer to quote. Settlement uses dirty because the seller has earned the interest accrued so far; the buyer pays it.

Day-count conventions matter:

| Convention | Used for |
| --- | --- |
| ACT/ACT (Act/Act) | US Treasuries |
| ACT/360 | US corporates, money market, US repo |
| 30/360 | US municipals, some US corporates |
| ACT/365 | UK gilts, some Eurobonds |
| 30E/360 | Some European corporates |
| ACT/365L | Australian government bonds |

The choice of day-count is *not* arbitrary; it is part of the bond's contract spec. A pricing system that uses the wrong day-count convention computes the wrong AI, which results in wrong settlement amounts. Errors of $0.01 to $0.10 per $100 face are typical for a one-day error; multiplied by a $1B portfolio, this is $10K to $100K in misvaluation.

A war story: I once spent half a day debugging a $50K daily settlement error that turned out to be a single bond using ACT/360 in the spec but ACT/365 in the discount factor. The fix was a one-line change in the day-count library; the lesson was that day-count must be a first-class field in the bond spec, not a hardcoded global.

## 5. Duration

Duration measures the first-order sensitivity of bond price to yield. There are three flavours, each useful in a different context.

![Duration: first-order sensitivity to yield](/imgs/blogs/bond-pricing-5.png)

**Macaulay duration** is the weighted-average time of cashflows, weighted by present-value:

$$
D_{\text{Mac}} = \sum_i t_i \cdot \frac{PV(C_i)}{P}.
$$

It has units of years. For a zero-coupon bond, Macaulay duration equals maturity. For a coupon bond, Macaulay duration is less than maturity because earlier coupons pull the centre of mass forward.

**Modified duration** is the price sensitivity:

$$
D_{\text{Mod}} = \frac{D_{\text{Mac}}}{1 + y/m}.
$$

It has units of (1 / yield). The interpretation: a 1% rise in yield causes a $D_{\text{Mod}}$% drop in price. Modified duration is the most commonly cited "duration" in trader conversation.

**Effective duration** is computed via scenarios:

$$
D_{\text{Eff}} = -\frac{P_+ - P_-}{2 P_0 \cdot \Delta y}
$$

where $P_+$ and $P_-$ are prices at $y \pm \Delta y$. Effective duration handles non-flat curves, embedded options, and other complications that break the analytical Macaulay/Modified formulas.

For our 10-year 4% coupon Treasury at 4.5% YTM:

- Macaulay duration ≈ 8.2 years
- Modified duration ≈ 8.0
- Expected price drop on +100 bp yield rise ≈ -8.0%

Duration is the *workhorse* of fixed-income trading. Every duration-based hedge sizes the offsetting position by matching DV01s (modified duration times price). A portfolio's duration is the value-weighted average of component durations. The "duration of the bond market" is a daily-quoted statistic that drives fixed-income index strategies.

```python
def macaulay_duration(face, coupon_rate, periods_per_year, maturity, ytm):
    n = int(maturity * periods_per_year)
    coupon = face * coupon_rate / periods_per_year
    pv_total = 0
    weighted_t = 0
    for i in range(n):
        t = (i + 1) / periods_per_year
        pv = coupon / (1 + ytm/periods_per_year)**(i+1)
        pv_total += pv
        weighted_t += t * pv
    pv_face = face / (1 + ytm/periods_per_year)**n
    pv_total += pv_face
    weighted_t += maturity * pv_face
    return weighted_t / pv_total
```

### 5.1 Duration of common bond types

Approximate duration shortcuts a senior trader knows by heart:

| Bond | YTM | Maturity | Approximate D_Mod |
| --- | --- | --- | --- |
| 2y Treasury | 4.5% | 2 | ~1.9 |
| 5y Treasury | 4.5% | 5 | ~4.5 |
| 10y Treasury | 4.5% | 10 | ~8.0 |
| 30y Treasury | 4.5% | 30 | ~17.5 |
| 30y zero-coupon | 4.5% | 30 | 30 |
| 30y MBS | 4.5% | 30 | ~6 (effective, prepay-adjusted) |
| 5y FRN | 4.5% | 5 | ~0.25 (next reset) |

The pattern: durations track maturity for vanilla coupon bonds, scale to maturity for zeros, are heavily compressed for prepay-bearing MBS, and are very small for FRNs. A senior fixed-income manager builds intuition for the approximate duration of any instrument from these reference points.

### 5.2 Spread duration vs rate duration

For credit-bearing bonds, two distinct sensitivities matter:

- **Rate duration** (or "Treasury duration"): sensitivity to a parallel shift in the risk-free curve. Approximately equal to the bond's modified duration if the spread is constant.
- **Spread duration** (or "credit duration"): sensitivity to a parallel shift in the credit spread. Often called *DV01-spread*.

Rate duration moves with the underlying Treasury curve; spread duration moves with the credit spread. A 5y BBB bond might have rate duration 4.5 and spread duration 4.5; a 30y BBB bond might have rate duration 17.5 and spread duration 17.5. They are usually similar but can diverge for callable bonds (where rate moves and spread moves affect call probability differently).

Senior credit-portfolio managers compute and hedge both separately. Rate duration is hedged with Treasuries or futures; spread duration is hedged with CDS, credit indexes, or sector-rotation trades.

## 6. Convexity

Duration is linear in yield; the actual price-yield relationship is convex. *Convexity* captures the second-order effect.

![Convexity: the second-order term that saves you in big moves](/imgs/blogs/bond-pricing-6.png)

The second-order Taylor expansion:

$$
\frac{\Delta P}{P} \approx -D_{\text{Mod}} \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2
$$

where $C$ is the *convexity*:

$$
C = \frac{1}{P} \sum_i t_i (t_i + 1/m) \cdot \frac{PV(C_i)}{(1 + y/m)^2}.
$$

Convexity has units of years². Always positive for plain-vanilla bonds. The asymmetry effect: for a bond with $D_{\text{Mod}} = 8$ and $C = 70$:

- Yield up 1%: $-D_{\text{Mod}} \cdot 0.01 + 0.5 \cdot C \cdot 0.0001 = -0.08 + 0.0035 = -7.65\%$
- Yield down 1%: $+D_{\text{Mod}} \cdot 0.01 + 0.5 \cdot C \cdot 0.0001 = +0.08 + 0.0035 = +8.35\%$

Long bonds gain *more* than they lose for symmetric yield moves. This is the convexity premium: long-bond positions implicitly own positive convexity, which is valuable in volatile rate environments.

Negative convexity arises in:

- **Callable bonds.** When rates fall, the issuer calls the bond; the gain is capped. When rates rise, the bond stays outstanding; the loss is full. Asymmetric in the bad direction.
- **Mortgage-backed securities (MBS).** Mortgages have prepayment options; falling rates cause refinancing waves and prepayment, while rising rates extend duration. MBS exhibit dramatic negative convexity.
- **Some auto-callable structured notes.** The autocall feature caps gains in favourable scenarios.

Negative convexity is the canonical risk in MBS trading; senior MBS quants spend their careers modelling prepayment behaviour and managing the resulting convexity exposure. We'll return to this in the [Exotic Derivatives post](/blog/trading/quantitative-finance/exotics/exotic-derivatives) and [Autocallables post](/blog/trading/quantitative-finance/exotics/autocallables).

## 7. DV01 and key-rate DV01s

DV01 (dollar value of a basis point) is the dollar P&L per basis point of yield change. It is the desk-level risk metric of choice.

![DV01 and key-rate DV01s: the desk-level risk metrics](/imgs/blogs/bond-pricing-7.png)

For a single bond:

$$
\text{DV01} = D_{\text{Mod}} \cdot P / 10000.
$$

The factor 10000 converts from "% per 1 yield point" to "$ per 1bp yield point" times $P$. For our 10-year 4% Treasury at $96.06$ with $D_{\text{Mod}} = 8$:

$$
\text{DV01} = 8 \cdot 96.06 \cdot 10^6 / 10000 = \$769 \text{ per bp on \$1M face.}
$$

A 1bp parallel rise in the 10-year yield costs the holder \$769 per million face. To hedge, the desk would short DV01-equivalent quantities of another instrument (futures, swaps, another bond).

**Key-rate DV01s** decompose the DV01 by maturity bucket. The 10-year bond has its DV01 mostly concentrated in the 10-year bucket, but with smaller contributions from 5-year and 7-year buckets (because of intermediate coupons and the curve interpolation). The decomposition is computed by perturbing one knot of the yield curve at a time:

$$
\text{KR-DV01}_i = -\frac{\partial P}{\partial y_i} \cdot 10^{-4}
$$

where $y_i$ is the yield at the $i$-th key-rate maturity. The sum of key-rate DV01s equals the total DV01 (under the right scaling).

Key-rate DV01s reveal *curve exposure* that total DV01 hides. A barbell portfolio (long 2y and 30y, short 10y) might have zero total DV01 but large positive 2y and 30y key-rate DV01s and large negative 10y key-rate DV01s. The portfolio is exposed to *curve steepening or flattening* even though it is duration-neutral.

Senior fixed-income traders manage their books along the key-rate DV01 dimensions:

- **Parallel shift hedge.** Match total DV01.
- **Steepener / flattener.** Manage the spread between long and short key-rate DV01s.
- **Bucketed hedge.** Match DV01 in each maturity bucket independently.
- **Butterfly hedge.** Match the curvature exposure (long the wings, short the body).

## 8. Spreads

Bonds trade at a *spread* over a benchmark curve. The spread compensates for credit risk, embedded options, liquidity, and other premiums above the risk-free rate.

![Spreads: spread to Treasuries, Z-spread, OAS](/imgs/blogs/bond-pricing-8.png)

**Spread to Treasuries.** Subtract the YTM of a similar-maturity Treasury from the bond's YTM. Simple but biased: it implicitly assumes a parallel curve, which fails for non-flat shapes.

**Z-spread.** The constant basis points added to the Treasury *zero curve* such that the discounted cashflows under the spread-shifted curve equal the bond's price. Curve-aware. Used for non-callable corporates, high-grade municipals, and similar non-optional bonds.

$$
P = \sum_i \frac{C_i}{(1 + (r_i + s)/m)^{m t_i}}, \quad \text{solve for } s.
$$

**Option-adjusted spread (OAS).** The Z-spread minus the value of embedded options, expressed in basis points. Computed by running an option-pricing model (lattice, Monte Carlo) under the curve and stripping out the embedded-option value. OAS is the *true* spread for callable, puttable, and prepayable bonds.

A worked example for a 5-year corporate bond:

- Treasury YTM (matched maturity): 4.0%
- Bond YTM: 5.0%
- Spread to Treasuries: 100 bp
- Z-spread: 98 bp (a small curve adjustment, because the Treasury curve is slightly sloped)
- OAS: 85 bp (the bond is callable; the embedded short-call to the issuer is worth ~13 bp)

Senior fixed-income traders compare OAS, not Z-spread, when evaluating callable bonds. Z-spread overstates the credit premium for callables; OAS gives the apples-to-apples credit comparison.

### 8.1 Spread duration and bps mapping

A useful concept for credit traders: 1 bp of spread × 1 unit of spread duration = 1 bp of price change. A bond with spread duration 5 and a spread shift of 20 bp moves price by 100 bp = 1% of par. This linear mapping is the operational rule of thumb for credit P&L attribution.

For comparison across bonds:

| Bond | Spread Duration | 20 bp shift price impact |
| --- | --- | --- |
| 2y BBB corporate | 1.9 | -0.38% |
| 10y BBB corporate | 7.5 | -1.50% |
| 30y BBB corporate | 14 | -2.80% |
| 5y high-yield | 4 | -0.80% |

Senior portfolio managers run their books in *spread-duration-weighted* terms when comparing relative value across credit issues. A 30-year corporate that looks "rich" by 5 bp is meaningful; a 2-year that looks rich by 5 bp is mostly noise.

### 8.2 The carry trade in fixed income

Unlike options, bonds usually have *positive carry*: holding the bond earns coupon yield minus funding cost, both of which are fairly predictable. Carry per day is approximately

$$
\text{Carry} = (\text{coupon yield} - \text{repo rate}) / 365 \cdot \text{notional}.
$$

For a 4% coupon bond financed at 5% repo, daily carry is negative — the bond loses money on funding alone. For a 5% coupon bond financed at 4% repo, daily carry is positive 1bp per day, or ~3.65% annual. Levered carry trades amplify this by 5-10x.

Senior fixed-income traders build their books with a target carry plus convexity profile, then manage hedges around that. Pure carry is exposed to spread blowouts; pure convexity costs theta-equivalent in opportunity cost. The combination is the trade.

## 9. Bond classes

Different bond classes have different risk profiles, conventions, and pricing models, but they all share the discount-cashflow core.

![Bond classes: Treasuries, corporates, MBS, munis, sovereigns](/imgs/blogs/bond-pricing-9.png)

**US Treasuries.** Considered the risk-free benchmark; calibration anchor for everything. Day-count ACT/ACT, semi-annual coupons. Yield curve construction starts here. Liquid through 30 years. Treasury Inflation-Protected Securities (TIPS) are inflation-linked variants.

**US Corporates.** Investment-grade and high-yield. Credit-spread driven; rating-sensitive (transitions matter). Often have embedded options (callable, puttable, sinking-fund). Day-count ACT/360 typical; semi-annual coupons.

**Municipals (US).** Tax-exempt at the federal level (and often state level). Convention 30/360. Tax-equivalent yield is the relevant comparison number for taxable investors. Credit-driven; insurance from monolines was historically common, less so post-2008.

**Mortgage-Backed Securities (MBS).** Pass-through securities collateralised by mortgages. Prepayment options. Negative convexity. Pricing requires a prepayment model and OAS. The largest single bond class in the US (~$11T outstanding). Agency vs non-agency split.

**Sovereigns.** Government bonds of countries other than the issuer's domestic. Emerging-market sovereigns introduce FX risk and sometimes restructuring risk. Yields denominated in the local currency; quoted in basis points over Treasuries (for hard-currency issues) or as standalone yields (for local-currency issues).

**High-Yield (junk).** Below-investment-grade corporates. Recovery as in CDS. Default models. Active-trader market. Spreads of 300-1000+ bp over Treasuries typical.

**Floating-Rate Notes (FRNs).** Coupons reset off a reference rate (LIBOR, SOFR). Pricing requires the forward curve. Duration is small (the coupon resets remove most of the rate risk); spread duration is large.

**Inflation-Linked.** TIPS, UK gilt linkers, Eurozone HICP-linkers. Coupons and principal indexed to inflation. Pricing requires a real curve and inflation expectations.

Each class has its own analytics suite, but the discount-cashflow core is identical. The pricing system is layered: a thin payoff/cashflow generator per class, a shared discount engine, a class-specific risk module.

## 10. Embedded options

Many bonds have embedded options that the simple cashflow-discount framework cannot price.

![Embedded options: callable, puttable, convertible](/imgs/blogs/bond-pricing-10.png)

**Callable bonds** give the issuer the right to redeem the bond at a call price (often par) on or after a call date. The issuer calls when refinancing is cheap (rates fall). The bond price is bounded above by the call price; gains in rallies are capped. Negative convexity.

**Puttable bonds** give the investor the right to sell the bond back at a put price on or after a put date. The investor puts when rates rise (the bond becomes less valuable). The bond price is bounded below; losses in selloffs are capped. Asymmetric upside.

**Convertible bonds** are bonds plus an equity option struck at a conversion ratio. The investor can convert the bond into shares. Pricing requires a 2-factor model (rate + equity) and is its own specialised topic.

**Sinking-fund bonds** retire principal on a schedule (the issuer redeems some fraction each year). Effectively shortens duration and can include partial calls.

**Make-whole calls** are calls at a price computed from the present value of remaining cashflows discounted at a small spread. Less aggressive than a fixed-call schedule.

Pricing engines for embedded options:

- **Callable / puttable.** Hull-White or Black-Karasinski lattice on a short-rate model. Backward induction with the call/put constraint at exercise dates.
- **Convertible.** Two-factor model: short-rate × equity SDE. PDE or Monte Carlo.
- **MBS prepayment.** Reduced-form intensity model with a prepayment rate that depends on rate level, seasoning, refinancing burnout, and macro factors. OAS Monte Carlo.

Embedded-option pricing is its own engineering layer. We'll cover the rates models in the [Short-Rate Models post](/blog/trading/quantitative-finance/rates-models/short-rate-models-vasicek-hull-white).

### 10.1 The MBS prepayment model in depth

MBS pricing requires modelling prepayments: borrowers can refinance their mortgages when rates fall, accelerating principal returns to MBS investors. The standard models (PSA model, internal proprietary models at Wall Street firms) decompose prepayment rate by:

- **Refinancing incentive.** $\text{incentive} = (\text{coupon} - \text{current mortgage rate}) / \text{coupon}$. When current rates are far below the coupon, refinancing is profitable for the borrower.
- **Seasoning.** Newly issued mortgages prepay slowly; mortgages 1-3 years old prepay faster as borrowers settle and refinance.
- **Burnout.** A mortgage pool that has already been through multiple refinancing waves has fewer remaining "rate-sensitive" borrowers; subsequent refinancing rates decline.
- **Macro factors.** Home prices, household income, geographic distribution.

A simple form: monthly prepayment rate (CPR) is a function of incentive, seasoning, burnout. Annualised CPRs of 5-50% are typical depending on market regime; CPRs below 5% indicate a dormant pool, above 30% indicate active refinancing.

OAS pricing of MBS proceeds by:
1. Simulate $N$ paths of the short-rate (or full curve) under a calibrated model.
2. For each path and each month, compute prepayment rate from the prepay model.
3. Generate the cashflow path: scheduled principal + scheduled interest + prepayment principal.
4. Discount at the simulated short-rate plus the OAS.
5. Average over paths to get the model price; iterate OAS until model price = market price.

The OAS is the spread that compensates for credit (small for agency MBS) plus the prepay-uncertainty risk premium. Production OAS Monte Carlo runs on $10^4$-$10^5$ paths and takes seconds per CUSIP; daily revaluation of an MBS book of 10,000 CUSIPs takes minutes on dedicated hardware.

### 10.2 Convertible bond pricing

Convertibles combine a bond floor with an equity option. The standard pricing approach:

- **Bond floor.** Discount the cashflows assuming no conversion. This is the lower bound.
- **Conversion value.** $S \times \text{conversion ratio}$ at any time.
- **Convertible value.** $\max(\text{bond floor}, \text{conversion value})$ plus a premium for early-conversion optionality.

The pricing model is typically a 2-factor PDE or Monte Carlo with:
- Short-rate dynamics (Hull-White)
- Equity dynamics (Black-Scholes or local-vol)
- Correlation between rates and equity

Conversion can be voluntary (optimal at high stock prices) or forced (the issuer can force conversion if the stock trades above a threshold for some period). Valuation is dominated by the equity option for in-the-money convertibles and by the bond floor for out-of-the-money convertibles.

Convertibles are a niche product (~$300B outstanding globally) but a high-margin one for the dealers who specialise in them.

## 11. Repo: how bonds are funded

A bond inventory has to be financed. The standard mechanism is *repo* — a sell-and-repurchase agreement.

![Repo: how bonds are funded](/imgs/blogs/bond-pricing-11.png)

How it works:

- **Day 0.** The desk delivers the bond to a counterparty (the cash lender). The counterparty pays cash equal to the bond price minus a haircut (typically 2-5% for Treasuries, 5-15% for corporates).
- **Day N (repo term).** The desk repays the cash plus repo interest. The counterparty returns the bond. The desk has effectively borrowed at the repo rate.

The repo rate is determined by:
- **General collateral (GC) rate.** The rate for collateral that is not in special demand. Typically close to the central bank's policy rate (Fed funds for USD).
- **Special collateral.** When a specific bond is in high demand for short-covering, its repo rate can spike well below GC. The "specialness" of an issue is a daily-traded quantity.
- **Term structure.** Overnight repo, term repo (1 week, 1 month, 3 months). Term repo rates reflect expectations of future overnight rates.

The repo rate matters for bond pricing because:

1. **Bonds are discounted at the repo curve in financed-position contexts.** A bond held in a repo-financed inventory has carry = coupon yield - repo rate. The carry is part of the trade's economics.
2. **Special repo squeezes signal scarcity.** A bond trading 50+ bp below GC is in special demand; the issuance might be over-shorted. Trading the special is a separate sub-market.
3. **Cash-and-carry trades hinge on repo.** Long bond, short futures, financed at repo. The trade locks in a small spread that accumulates with funding.
4. **Repo squeezes propagate to bond prices.** When repo rates spike (2008 mortgage repo, September 2019 SOFR), bond inventories become too expensive to hold, and dealers liquidate, depressing prices.

The 2008 repo squeeze on mortgages was a key contributor to Bear Stearns' collapse: the firm's mortgage inventory could not be funded at acceptable repo rates, forcing fire-sale liquidations. The 2019 SOFR spike (overnight rates touching 10% briefly) was a Fed-managed balance-sheet event that tested the repo plumbing under stress.

## 12. Bond futures

Bond futures contracts are exchange-traded; they trade the cheapest-to-deliver bond from a basket.

![Bond futures: the listed exchange-traded sister](/imgs/blogs/bond-pricing-12.png)

The contract specifies a basket of deliverable bonds (e.g., "any 6-10 year US Treasury"). At expiry, the seller chooses which bond to deliver. The buyer pays a price that is the futures price times a *conversion factor* (a fixed coefficient for each deliverable).

The cheapest-to-deliver (CTD) is the bond with the highest *implied repo rate* — the rate at which buying the bond and delivering it via the future breaks even:

$$
\text{Implied repo} = \frac{F \cdot CF + \text{coupons received} - P_0}{(P_0 - \text{accrued}) \cdot t} \cdot 360.
$$

The CTD changes as the curve moves; this is *CTD switching* and is an important feature of futures trading.

Trades around bond futures:

- **Basis trade.** Long bond, short future. Captures the cash-future spread. Risk: CTD switches mid-trade and the basis changes.
- **Calendar roll.** Roll the future from front to back. Trade the calendar spread (the difference between front and back prices).
- **Duration overlay.** A portfolio adds or hedges duration cheaply via futures, without trading individual bonds.
- **Cheapest-to-deliver arbitrage.** When the CTD trade is rich or cheap relative to futures, the basis can be traded directly.

Futures are the *liquid* end of the cash market; cash bonds are OTC and slower. Senior fixed-income traders move between the two depending on size, urgency, and basis.

### 12.1 Implied repo math worked

For a US Treasury futures contract, the implied repo formula:

$$
\text{Implied repo} = \frac{F_0 \cdot CF + \text{coupons received during life} - P_0}{(P_0 - \text{accrued at trade date}) \cdot t} \cdot 360
$$

where $F_0$ is futures price, $CF$ is the conversion factor, $P_0$ is the cash bond price including accrued, and $t$ is the time to delivery.

A worked example. Suppose:
- 10y futures price $F = 110$
- Cash bond (CTD candidate) at $P_0 = 95.50$ clean, accrued $= 0.50$
- CTD conversion factor $CF = 0.8650$
- Days to delivery: 70 days
- Coupon during the period: 0.625

Implied repo = $((110 \times 0.8650) + 0.625 - 96.00) / (96.00 \times 70 / 360) = (95.15 + 0.625 - 96.00) / 18.667 = -0.225 / 18.667 = -1.20\%$.

A negative implied repo means it's cheaper to buy the bond and deliver it via futures than to fund the inventory at GC repo. The trade is a buy-cash, sell-futures basis trade. Senior basis traders compute implied repo for every CTD candidate daily and trade against the highest-yielding one.

### 12.2 Bond futures vs interest-rate swaps for hedging

For duration hedging, a fixed-income desk has two main listed options:

- **Bond futures.** Liquid, exchange-traded, low frictional cost. Drawbacks: CTD switching, calendar roll mechanics, basis risk.
- **Interest-rate swaps.** OTC, more bespoke, can match any duration. Drawbacks: bilateral credit, ISDA, margin, less liquid in stress.

For most desk-level hedging, futures win on cost and immediacy. For long-dated structured products that need 30-year hedges, swaps are necessary because no listed futures contract trades that maturity. Modern desks typically use a mix: futures for short-end and intermediate, swaps for long-end and bespoke.

## 13. Production architecture

A serious bond-pricing system separates curves, instruments, and engines as orthogonal axes.

![Production architecture: bond pricing as a curve-driven service](/imgs/blogs/bond-pricing-13.png)

**Curve service.** Bootstraps yield, OIS, repo, and swap curves from market quotes. Snapshots, versions, and publishes. The single source of truth for curves.

**Instrument library.** Bond specs as data: face, coupon, frequency, maturity, schedule, day-count, embedded options. Cashflow generator that takes a spec and produces the cashflow schedule. Day-count library that handles all conventions. Schedule library that handles holidays, business-day conventions, and end-of-month rules.

**Pricing engine.** Discounts cashflows. Computes YTM, duration, convexity, DV01, key-rate DV01s. For embedded options, calls into the option-pricing module (lattice, PDE, MC).

**Risk service.** Aggregates DV01 and key-rate DV01s across portfolios. Runs stress scenarios (parallel shifts, steepeners, flatteners, butterflies). Computes VaR. Submits to capital and regulatory reports.

The architecture should be:

- **Versioned.** Every curve and price is associated with a snapshot ID; history is reproducible.
- **Audited.** Every price computation is logged with inputs, outputs, and version IDs.
- **Testable.** Each layer has unit tests; cross-layer integration tests verify end-to-end correctness.
- **Performant.** Pricing a $100K-instrument portfolio should take seconds, not minutes.
- **Multi-currency.** Discount curves per currency; FX layer for cross-currency conversion.

Common antipatterns:

- **Hardcoded day-count.** Bond specs that don't carry day-count are a recipe for errors.
- **Single-curve assumption.** Pre-2008, many libraries used LIBOR for both forward projection and discounting. Post-2008, multi-curve is mandatory.
- **Mixing pricing and risk in one module.** Pricing should produce the price; risk should compute Greeks via bumping or AAD over the pricing function. Separating them allows independent testing and evolution.

## 14. Failure modes

Bond pricing fails in named regimes; senior quants recognise the symptom early.

![Failure modes: where bond pricing goes wrong](/imgs/blogs/bond-pricing-14.png)

**Default surprise.** A corporate bond's credit spread blows out; mark-to-market drops to recovery × face. Recovery is uncertain (35-40% is the historical average, but bond-by-bond varies). Pricing systems must handle distressed-bond marks (cessation of accrual, recovery-based pricing, liquidation overrides).

**Call surprise.** A callable bond gets called when the model expected it to remain outstanding (or vice versa). The realised yield differs from the model's expected yield. The OAS calibration must be reviewed when call patterns deviate from expectations.

**Prepay surprise (MBS).** Prepayment rates differ from the model's prediction. Negative convexity is realised when rates fall and prepayment accelerates beyond expectations. The 2003-2004 refinancing wave caught many MBS desks off-guard; the 2020 COVID-era refinancing wave was even larger but better modelled.

**Repo spike.** Funding rates jump. Bond inventories become too expensive to hold; dealers liquidate. The 2019 SOFR spike (touched 10% overnight) and the 2008 mortgage repo squeeze are textbook examples.

**Curve regime shift.** Negative rates (post-2014 EUR), zero-bound (post-2008), policy-rate jumps (2022-2024 hike cycle). Each shift changes the structural shape of the curve and breaks assumptions in pricing systems. Shifted models, alternative curve fitters, and architectural flexibility are required.

## 15. Case studies

### 15.1 The 1994 bond-market crash

The Federal Reserve raised rates sharply in 1994 (Fed funds from 3% to 6% over the year). The 30-year Treasury yield rose ~200 bp; the mark-to-market on bond inventories dropped 15-20%. Several major firms posted large losses; Procter & Gamble's notorious leveraged interest-rate swap with Bankers Trust was settled in 1996 for $200M. The lesson: duration risk is real; long-bond positions can lose double-digit percentages in single quarters during rate-hike cycles.

### 15.2 LTCM 1998 (revisited)

LTCM held large convergence positions in Treasury, swap, and mortgage spreads. When credit spreads widened in Q3 1998, the marks went heavily against them; their leverage forced liquidation at the worst time. The lesson, repeated throughout this series: spreads are more volatile than markets believe, and leveraged spread trades require sizing for extreme moves.

### 15.3 The 2008 MBS crisis

Mortgage-backed securities, which had been priced under prepayment models calibrated on benign data, suddenly faced a regime where home prices fell, refinancing dried up, and defaults rose. Prepayment models were systematically off; OAS-based pricing produced wildly wrong marks. Several MBS desks took losses of $5-30B each. The lesson: structural-model risk is real; calibrating on benign history doesn't protect against regime change.

### 15.4 Bear Stearns repo squeeze, March 2008

Bear Stearns' mortgage inventory was financed via repo. As confidence eroded, repo counterparties refused to roll the trades; Bear had to liquidate at fire-sale prices over a single weekend. JPM acquired the firm for $2/share ($10/share after revisions). The lesson: a bond's price and its *funded value* are different in stress. Risk systems must track funding stability as a first-class metric.

### 15.5 The 2010 European sovereign crisis

Greek, Irish, Portuguese, Italian, and Spanish bond yields blew out to 10-30% as default fears mounted. The eurozone's "no-bailout" assumption was violated; sovereign restructurings (Greek 2012) produced 50%+ haircuts. Pricing systems that treated eurozone sovereigns as risk-free were forced to introduce credit spreads and recovery models. The lesson: even sovereign bonds carry credit risk, and the modelling must reflect it.

### 15.6 LIBOR-OIS divergence (2008-2010)

Pre-2008, swap pricing used LIBOR for both forward projection and discounting. Post-Lehman, OIS-discounting became standard for collateralised trades, with LIBOR retained for forward projection. Every fixed-income pricing system was retrofitted. The lesson: discount curves are a first-class engineering concept that must support multiple curves per cashflow.

### 15.7 The 2019 SOFR spike

On 17 September 2019, the SOFR rate spiked from ~2.2% to 5.25% intraday before the Fed intervened. Repo trades that priced in stable funding suddenly faced 3% higher cost overnight. Several hedge funds and dealer books took marks against. The lesson: even short-term funding rates can move dramatically; bond-funding models should include stress scenarios for repo spikes.

### 15.8 SVB and the held-to-maturity Treasury portfolio (March 2023)

We covered this in [the derivatives pricing case study](/blog/trading/quantitative-finance/derivatives/derivatives-pricing#11-8-svb-march-2023-held-to-maturity-bonds-mark-to-model-vs-mark-to-market). For bond pricing specifically, the lesson: the *accounting* convention can hide mark-to-market losses, but the loss is real. Risk systems must report mark-to-market regardless of accounting choice; the gap between book and market value is itself a first-class risk metric.

### 15.9 The 2022-2024 rate hike cycle

The Fed raised rates from 0.25% to 5.5% over 2022-2024. The Bloomberg Aggregate Bond Index lost 13% in 2022 alone — the worst calendar year for fixed income on record. Long-duration bond positions took 30-40% mark-to-market losses. The lesson: duration risk is asymmetric and large; long-end exposures must be sized for hike-cycle scenarios that can persist for years.

### 15.10 The 2024 GBP gilt crisis (Truss mini-budget aftermath)

In late September 2022, the UK government announced unfunded tax cuts. UK gilt yields rose 100+ bp in days; the LDI (liability-driven investment) pension funds faced margin calls on their interest-rate swaps; forced liquidations of gilts amplified the move. The Bank of England intervened. The pension industry was restructured. The lesson: leveraged fixed-income positions in pension portfolios can produce systemic-level forced selling; the risk of correlated forced flows is real and difficult to model.

### 15.11 The 1979-1982 Volcker disinflation

When Paul Volcker raised the Fed funds rate to 19% in 1981 to break inflation, long-bond yields followed. The 30-year Treasury hit 15%; long bonds priced at 50% of par. Many institutional bond portfolios took 30-50% mark-to-market losses. Pension funds, insurance companies, and savings & loans suffered. The lesson: extreme rate moves reshape the entire bond-pricing landscape; risk frameworks must accommodate scenarios beyond recent history.

### 15.12 GE Capital and corporate-debt overhang (2008-2010)

GE Capital, the financing arm of General Electric, was the largest issuer of corporate debt in the world. In 2008, the firm faced a near-collapse as commercial paper markets froze. The Federal Reserve created the Commercial Paper Funding Facility (CPFF) to backstop GE and similar issuers. Pricing systems that priced GE Capital debt at "AAA risk-free spread" suddenly had to introduce stressed-corporate spreads of 500-1500 bp. The lesson: even highest-rated corporates have credit risk; pricing must be robust to spread regime changes.

### 15.13 The 2020 COVID liquidity crisis

In March 2020, even US Treasury liquidity briefly broke. Bid-ask spreads on 30-year Treasuries widened from 1bp to 5+ bp; on-the-run vs off-the-run differentials blew out. Several large fixed-income funds had to mark inventories down. The Fed announced unlimited Treasury purchases; the market normalised in weeks. The lesson: even risk-free assets have liquidity premiums in stress; robust pricing must be able to mark to *transactable* prices, not theoretical ones.

### 15.14 The 2024 yen carry unwind and JGB market

When the BOJ raised rates in mid-2024, the yen carry trade unwound rapidly. Japanese government bond (JGB) yields jumped 30 bp in days; long-end JGBs sold off dramatically. Carry trades that had been profitable for years took 6-month worth of losses overnight. The pricing lesson: cross-asset correlations spike in stress; FX-funded fixed-income positions need joint stress scenarios.

## 16. When to use which framework

| Bond type | Primary tool | Why |
| --- | --- | --- |
| Vanilla Treasury / corporate | DCF + curve discount | Closed-form, fast |
| Floating-rate note | DCF + forward curve | Forward projection of resets |
| TIPS / inflation-linked | Real curve + inflation expectations | Real vs nominal |
| Callable / puttable | Hull-White lattice | Embedded option value |
| MBS | OAS Monte Carlo + prepay model | Path-dependent prepayments |
| Convertible | 2-factor (rate × equity) | Hybrid asset class |
| Sovereign with FX | Multi-curve + FX | Cross-currency |
| Distressed | Recovery model | Default scenarios |

## 17. Three closing principles

**Curves first.** The curve is the central infrastructure of fixed income; bonds are a thin layer on top. Invest in the curve service.

**Day-count and conventions matter.** Bugs in day-count, business-day, and roll conventions are surprisingly common and surprisingly costly. Treat conventions as first-class data.

**Stress beyond historical experience.** The 2022 hike cycle was beyond most pre-2022 stress scenarios. The 2008 mortgage crisis was beyond pre-2008 prepayment models. Build flexibility for regime shifts that haven't happened yet.

## 18. Production checklist

A condensed checklist for a bond-pricing library:

1. **Curve service** versioned and snapshotted.
2. **Multi-curve discounting** support (OIS, LIBOR, repo).
3. **Day-count library** supporting all major conventions.
4. **Schedule library** with holidays, business-day, end-of-month rules.
5. **Cashflow generator** taking instrument spec → cashflow schedule.
6. **Pricing engine** computing price, YTM, duration, convexity, DV01, key-rate DV01s.
7. **Embedded-option engine** (lattice / PDE / MC) for callable, puttable, MBS.
8. **Spread engine** computing spread to Treasuries, Z-spread, OAS.
9. **Risk service** aggregating DV01 across portfolios; running stress scenarios.
10. **Audit logging** for every price computation.
11. **Cross-validation** of pricing against alternative implementations (QuantLib, internal reference).
12. **Multi-currency support.**

A library that ticks all 12 is production-grade.

### 18.1 A worked end-to-end example

To bring everything together, a complete worked example for a 10y BBB corporate bond.

**Inputs:**
- Face: $1,000,000
- Coupon: 4.5% annual, paid semi-annually
- Maturity: 10 years
- Day-count: ACT/360
- Settlement: T+2
- Today: 2026-05-03
- Treasury 10y YTM: 4.0%
- Bond mid-market price: 95.50 (clean)
- Last coupon date: 2026-04-15
- Days since last coupon: 18
- Days in coupon period: ~182.5

**Step 1: compute accrued interest.**
$\text{AI} = (18 / 182.5) \times 22.50 = 2.22$

**Step 2: dirty price.**
$P_{\text{dirty}} = 95.50 + 2.22 = 97.72$ → settlement = $977,200

**Step 3: yield to maturity.**
Using Newton-Raphson, YTM ≈ 5.05% semi-annual.

**Step 4: spread to Treasuries.**
$\text{Spread} = 5.05\% - 4.00\% = 105 \text{ bp}$

**Step 5: Z-spread.**
Discounting under the Treasury zero curve plus a constant $s$ such that NPV = $97.72:
Z-spread ≈ 100 bp (slightly less than nominal spread due to upward-sloping curve).

**Step 6: duration and convexity.**
Macaulay D ≈ 8.0 years, Modified D ≈ 7.81 (semi-annual to annual conversion), Convexity ≈ 73.

**Step 7: DV01.**
$\text{DV01} = 7.81 \times 977{,}200 / 10{,}000 = \$763 \text{ per bp}.$

**Step 8: hedge.**
Short ~99 contracts of 10y Treasury futures (each ~$77 DV01) to flatten parallel rate exposure. Spread duration ≈ 7.81; spread DV01 = $763. To hedge spread, buy CDS protection on the issuer with 10y maturity at $763 DV01-equivalent.

**Step 9: scenario analysis.**
- Treasury yield +50 bp: bond price drops by ~3.9%; spread unchanged; total -3.9%.
- Spread +50 bp: bond price drops by ~3.9% from spread; total -3.9%.
- Issuer downgrade (BBB → BB): spread widens to 250+ bp; price drops to ~89.

The 9-step exercise is roughly 30 minutes of manual work; in production, every step is automated and runs in milliseconds. But every senior fixed-income quant should be able to do each step by hand to verify their pricing system's outputs.

## 19. The cultural side of fixed income

Fixed-income desks have a different culture than options desks. The trades are larger (notional, not premium), more curve-driven, more macro-sensitive. The hedges are more mechanical (DV01-matched). The wins and losses are smaller in percentage but larger in dollars.

Senior fixed-income quants are usually deep specialists in their corner: rates, mortgages, credit, sovereigns, munis. Cross-desk collaboration is high (rates and FX, rates and credit, MBS and rates). The math is less elaborate than options math; the engineering is more demanding (bigger portfolios, more conventions, more curve plumbing).

Cultural practices that distinguish strong fixed-income teams:

- **Daily curve review.** The curve team meets daily to review yesterday's curve, validate against alternative bootstrappers, flag any discontinuities.
- **Convention discipline.** Every bond's day-count, schedule, and embedded options are explicitly documented in the trade ticket.
- **Stress-test suite.** Regular runs of historical (2008, 1994, 2022) and hypothetical (parallel shifts, steepeners, repo spikes) scenarios.
- **Collaboration with risk and capital.** Fixed income is more capital-intensive than options; the capital team is a key stakeholder.

### 18.1a Curve choice: a buyer's vs seller's discount rate

A subtle and important issue in modern fixed-income pricing: the *seller* and the *buyer* may discount the same bond at *different* rates depending on their funding situation.

A traditional dealer, financing inventory at the dealer's repo rate, discounts the bond's cashflows under that repo curve. A pension fund, holding the bond unfunded against assets, discounts under a different curve (maybe LIBOR or the pension's actuarial discount rate). A central bank, holding the bond as policy infrastructure, discounts under no curve at all.

The bond's price the *market* shows is consistent with one of these — typically the dealer's repo-discounted price. Other holders may value the same bond differently. This is why we see persistent demand from "natural buyers" (pension funds, insurance companies) at prices the dealer wouldn't touch: they value the bond at a different rate.

For pricing-system engineering: every consumer of pricing data should be aware of which discount curve was used. A pension-fund-owned bond marked at dealer-curve prices is a different number from the pension's economic value. Senior architects sometimes publish *both* — a market price (dealer-curve-consistent) and a holder-economic-value (under the holder's funding) — to make this distinction explicit.

### 18.2 Common bond-pricing bugs and how to spot them

A non-exhaustive list of bugs encountered in production:

**Wrong day-count.** Settlement amount off by a few cents per $100 face. Diagnostic: cross-check accrued-interest computation against the issuer's prospectus.

**Wrong settlement convention.** T+1 vs T+2 vs T+3 mishandled. Diagnostic: compare against bond market data feeds; settlement is published.

**Curve interpolation off-by-one.** Discount factor for a cashflow exactly on a curve knot uses adjacent-knot value. Diagnostic: stress with cashflows at knot dates.

**End-of-month conventions.** A bond with 31 January payment date may roll to 28/29 February. Diagnostic: schedule generation tests with edge dates.

**Compounding-frequency mismatch.** YTM computed at semi-annual compounding compared with annual-compounded reference. Diagnostic: explicit compounding tags on every yield value.

**Forward-rate errors in FRN pricing.** Future floating coupons computed at wrong forward rate. Diagnostic: analytic test on a flat curve (FRN should price at par).

**Stale curve.** Pricer using yesterday's curve. Diagnostic: every price logged with curve timestamp; refuse stale curves.

**Sign convention.** DV01 with wrong sign produces hedges in the wrong direction. Diagnostic: DV01 of a long-bond position should be positive (negative price impact for positive yield change, but DV01 is the *dollar* sensitivity).

**Float-vs-double precision.** Long-duration bonds with deep-discount discount factors hit precision limits in 32-bit float. Diagnostic: 64-bit double everywhere in production; never 32-bit.

**Wrong currency.** EUR-denominated bond priced with USD curve. Diagnostic: every input tagged with currency; cross-currency layer explicit.

A senior quant maintains a personal log of every bug encountered with diagnostic and fix. The log compounds in value.

## 19a. Daily routine of a fixed-income desk

A typical day on a fixed-income trading desk follows a predictable rhythm:

**07:00.** Pre-open. Curves overnight from London desk. New economic data releases (CPI, NFP, Fed minutes) digested. Risk dashboard reviewed: total DV01, key-rate DV01s, spread DV01, by sector.

**08:30.** US economic releases (jobs, CPI, GDP). Curves move; pre-positioned trades execute or roll. Pricing systems recalibrate.

**09:00.** US market open. Cash and futures liquidity peaks. Active trading; client RFQs come in from sales. Pricer responds with bids/offers.

**09:30 - 11:30.** Most active period. Auctions, primary issuance, mark-to-market.

**11:30 - 13:00.** Lunch. Liquidity thinner; less interactive trading.

**13:00 - 14:00.** Treasury auctions (when scheduled). Pricing systems mark off the auction print.

**14:00 - 15:30.** Afternoon trading. Position adjustments. Risk re-review.

**15:30 - 16:00.** Closing rush. End-of-day mark-to-market preparation.

**16:00 - 17:00.** Settlement and EOD process. Final marks, P&L, risk reports.

**17:00 - 19:00.** Quant team daily review: curve quality, calibration drift, validation gates. Trader debrief: what worked, what didn't, what to do tomorrow.

This rhythm has been the same for 30 years; only the tooling has evolved. A senior fixed-income quant fits into this rhythm by ensuring the pricing infrastructure runs invisibly through it.

### 19.0 The economics of a fixed-income desk

A typical sell-side fixed-income desk runs P&L driven by:

- **Bid-ask spreads.** ~1-5 bp on Treasuries, 10-50 bp on corporates, larger on illiquid issues. With $100B daily volume, even 1 bp average spread generates $10M/day. Annualised: $2.5B/year.
- **Inventory funding.** Dealer holds bonds at repo. Carry on inventory: coupon yield - repo rate. Net carry can be positive or negative; depends on curve shape and funding cost.
- **Mark-to-market.** Daily mark of the inventory at the published curve. Drives daily P&L volatility.
- **Customer financing.** Repo book — lending out cash against collateral. Earns the haircut spread.
- **Structuring.** Customised structured notes earn 50-200+ bp spreads. Higher margin but lower volume.

Total: a major sell-side fixed-income desk generates $500M-$2B annual gross revenue, with operating costs of $200M-$500M (mostly compensation), generating $300M-$1.5B in net contribution. The pricing infrastructure investment of $50M-$100M annually is small relative to the business.

A senior fixed-income engineer should know roughly what the desk's economics look like; it informs prioritisation. Investing engineering effort in marginal-improvement of bid-ask is high-value; investing in marginal-improvement of structured-products pricing is even higher value (higher margin per trade); investing in pretty dashboards is low value relative to the alternatives.

### 19.1 Engineering for fixed-income at scale

A few engineering observations from building bond systems at scale:

**Caching matters more than algorithms.** A naive bond pricer evaluates discount factors for every cashflow on every call. With careful caching of (curve, cashflow date) → discount factor, a portfolio revaluation can be 10-100x faster. Modern fixed-income systems memoise aggressively.

**Curve evaluation should be vectorised.** Curves are evaluated thousands of times per pricing call (for every cashflow of every bond). Vectorise the curve evaluator across cashflow dates; let NumPy or Eigen broadcast over the date array. Single-bond evaluation drops from milliseconds to microseconds.

**Day-count is hot code.** Day-count conversions are called for every cashflow in every bond every time. A naive Python day-count library is the bottleneck of the system; rewriting in C or hand-rolling SIMD accelerators pays back enormously.

**Schedule generation is also hot.** Generating cashflow schedules for thousands of bonds, with holiday handling and end-of-month adjustments, takes meaningful CPU. Cache schedules by spec hash; regenerate only when specs change.

**Audit logs should be append-only.** Every pricing call produces a row in an audit log. Use append-only storage (Kafka, immutable filesystem) so the log can never be retroactively edited. Regulators love this.

**Cross-validation in CI.** Every CI build runs a regression suite of textbook bond examples plus a sample of last week's production trades. Any deviation > 0.01% from the reference fails the build.

**Multi-language interop.** Fixed-income systems often span C++ (pricing core), Python (research and analytics), Java (middle-office), maybe Rust (newer pieces). The cross-language interface must be carefully defined; protobuf or FlatBuffers for inter-process serialisation, with a shared schema repo.

**Curve-as-a-service architecture.** The curve service is a separate microservice with its own SLA, scaling, and failover. Pricing services subscribe to curve updates and cache locally. This decoupling lets curve and pricing teams iterate independently.

**Stress-test infrastructure parallel to production.** Stress runs are too expensive for production paths but too important to skip. Run stress in a parallel infrastructure on the same code paths, scheduled nightly.

A senior fixed-income engineering manager makes these architectural choices early; they compound over the system's life.

### 19.2 The role of central banks in bond markets

A senior fixed-income quant must understand central-bank policy because it dominates the curve.

**Conventional monetary policy.** Setting the short-term policy rate (Fed funds, ECB deposit rate, BoJ overnight rate). Affects the front of the curve directly.

**Forward guidance.** Central-bank communication about future rate paths. Affects the medium part of the curve via expectations.

**Quantitative easing (QE).** Central-bank purchases of long-end government bonds. Compresses term premiums; flattens the curve.

**Quantitative tightening (QT).** Reverse of QE. Steepens the curve as term premiums normalise.

**Yield curve control (YCC).** As practiced by the BoJ from 2016-2024, the central bank explicitly targets a long-end yield. Pricing in YCC regimes requires modelling the central bank as a counterparty.

**Currency interventions.** A central bank intervening in FX affects domestic bond yields via the FX-rates parity arbitrage.

A bond quant must read FOMC minutes, ECB press conferences, BoJ statements, etc. The curve is not autonomous; it is shaped by policy. Senior traders watch policy as carefully as they watch flows.

The 2022-2024 hike cycle was a master class in this dynamic. The Fed's communication about rate paths shifted the curve weeks before actual rate changes. Forward guidance moved the 2-year yield more than the actual hike did. A pricing system that ignored policy commentary would have missed the actual market dynamics.

## 20. Conclusion

Bond pricing is the simplest non-trivial pricing problem in finance: discount the cashflows. The complexity comes from the curve (which is a research and engineering problem of its own), the conventions (day-count, schedule), the embedded options (which require their own pricing models), the funding (repo), and the risk decomposition (duration, convexity, DV01, key-rate DV01s).

A subtle observation about pricing maturity: the field is so well-developed that vanilla bond pricing is essentially a solved problem. The remaining frontier is in the *interfaces* — how does pricing connect to risk, capital, regulation, accounting, audit? Modern engineering investment goes into these interfaces rather than into the math. A new pricing library is unlikely to differ from a 2010 library on price; it will differ on auditability, scaling, multi-currency support, regulatory compliance, and integration with the rest of the firm. Junior quants attracted to elegant math sometimes find this disappointing; senior engineers find it liberating, because it means the value-creation is now in engineering rather than in clever formulas.

A senior fixed-income quant operates fluently in two languages: YTM (the trader's language) and curve (the engineer's language). They can price a bond by hand, decompose its risk into key-rate DV01s, identify its funding cost, and recognise when a regime shift will break their pricing assumptions.

The remaining articles in this series — [Yield Curve Modeling](/blog/trading/quantitative-finance/fixed-income/yield-curve-modeling), [Fixed Income Analytics](/blog/trading/quantitative-finance/fixed-income/fixed-income-analytics), [Short-Rate Models](/blog/trading/quantitative-finance/rates-models/short-rate-models-vasicek-hull-white), [Exotic Derivatives](/blog/trading/quantitative-finance/exotics/exotic-derivatives), [Autocallables](/blog/trading/quantitative-finance/exotics/autocallables), and [Cliquets](/blog/trading/quantitative-finance/exotics/cliquets) — go deeper on each layer.

The fixed-income world is large, conservative, and detail-oriented. Mastering it requires patience for conventions and respect for the curve. The reward is a clear-headed view of one of the largest financial markets in the world.

### 19.2a The role of regulatory capital in fixed-income decisions

Post-2008, regulatory capital became a first-class input to every fixed-income trade. The basic framework: every trade consumes regulatory capital; the trade must earn enough revenue to cover the capital cost plus a spread.

Regulatory capital has several components:

- **Standardised market risk capital.** Computed via a regulator-provided formula on bond positions. Higher for longer-duration, lower-rated, less liquid bonds.
- **Internal-models market risk capital.** Banks with regulator-approved internal models can compute capital based on historical VaR. Generally lower than standardised but with model-validation overhead.
- **Credit valuation adjustment (CVA) capital.** Capital against the counterparty credit risk of trades. Significant for uncollateralised counterparties.
- **Leverage ratio capital.** Capital against the gross notional, regardless of risk weight. The Basel III leverage ratio caps the firm's gross-notional / equity ratio.

Each of these is computed daily by the risk and capital teams. The trade-level economics include the capital cost (typically 5-15% of capital, depending on the firm's hurdle rate). A trade that produces 50 bp gross revenue but consumes capital costing 30 bp may not pass the desk's economic threshold.

A senior fixed-income engineering team integrates capital computation into the pricing-quote workflow. Junior teams price the trade and let capital be computed downstream; senior teams compute capital pre-trade and reject economically marginal trades at the quoting stage. The senior approach is much more efficient and is the modern standard.

### 19.3 Bond-portfolio-level analytics

Beyond single-bond pricing, fixed-income desks aggregate analytics at the portfolio level. The standard reports:

**Total market value.** Sum of dirty prices times notional across positions.

**Duration-weighted maturity.** Average maturity weighted by DV01 contribution. The "duration" of a portfolio.

**Spread duration breakdown.** DV01 by sector (Treasuries, agencies, corporates, MBS, munis), by rating bucket, by issuer.

**Key-rate DV01 surface.** 2D heatmap of DV01 by maturity bucket × sector. Reveals concentrated exposures.

**Yield-to-worst weighted average.** Portfolio's expected yield assuming worst-case option exercise.

**Credit profile.** Distribution of credit ratings; weighted-average credit rating.

**Liquidity profile.** Notional in liquid vs illiquid issues; concentration in single issues.

**VaR and stress P&L.** Daily VaR (typical 99% 1-day for trading books); stress P&L under historical scenarios (1994, 2008, 2020) and hypothetical (parallel +200 bp, steepener +100 bp 30y -100 bp 2y, credit blowout +100 bp).

These reports are produced daily by the risk service and consumed by traders, portfolio managers, risk committee, and regulators. The aggregation engineering is non-trivial because the inputs are heterogeneous (different bond types, different conventions, different curves) and the outputs must be consistent. Senior fixed-income engineering invests heavily in this aggregation layer.

## 20.0 Closing observations on craft

After 50+ pages of bond pricing, a final summary of the craft:

The pricing math is simple — sum the discounted cashflows. The engineering complexity is in the curve, the conventions, the embedded options, the funding, the auditing, and the integration with the rest of the firm. Each of these layers has its own depth.

A senior fixed-income quant develops three layered skills:

1. **Mathematical fluency.** Can derive duration, convexity, OAS from first principles; can write a bond pricer from scratch in any language; can debug a pricing discrepancy by walking through the math.
2. **Engineering taste.** Knows which architectural decisions matter (curve service, day-count library, instrument-as-data); knows which premature optimisations to avoid.
3. **Market intuition.** Reads the curve and knows what is normal vs unusual for the regime; knows when a calibration is going wrong before the validation gate fires.

These skills compound over a career. A 5-year quant can do (1); a 10-year quant adds (2); a 15-year quant adds (3). Senior fixed-income quants are valuable because of the integrated combination, not any single skill in isolation.

## 20.1 Final practical advice

If you are starting your career as a fixed-income quant or engineer:

1. **Learn day-count conventions cold.** Spend a week understanding ACT/ACT, ACT/360, 30/360, ACT/365 in detail. Build a small library that handles them all and verify against authoritative sources (the SIFMA standard).
2. **Build a curve from scratch once.** Bootstrap a real Treasury curve from market quotes. Understand every step. The understanding pays back for years.
3. **Price a callable bond by hand once.** Use a small lattice and walk through the backward induction. Watch how the call constraint binds and unbinds with rate scenarios.
4. **Spend time on a trading desk.** A week of shadowing traders teaches more about how curves move than a year of textbooks. The intraday rhythm, the macro narratives, the customer flows — all of it informs the engineering.
5. **Read the SIFMA conventions document.** The industry's standard reference for US bond conventions. Boring but indispensable.
6. **Subscribe to one good fixed-income newsletter.** Reuters, Bloomberg, or specialised research. A daily window into how traders think about the markets.

The journey from junior to senior fixed-income engineer is a decade of learning conventions, regimes, and engineering patterns. The reward is being one of the silent majority of finance — the people who price the largest market in the world correctly, day after day, conventionally and reliably. There is dignity and craft in that role.
