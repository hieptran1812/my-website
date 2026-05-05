---
title: "Fixed Income Analytics: From DV01 to Capital, the Numbers That Run the Desk"
date: "2026-05-03"
publishDate: "2026-05-03"
description: "A senior-quant deep dive into fixed-income analytics: DV01 aggregation, key-rate DV01s, spread DV01, OAS analytics, scenario stress, VaR, P&L attribution, carry and roll-down, factor models, liquidity, capital, production architecture, and named failure modes."
tags:
  [
    "fixed-income-analytics",
    "dv01",
    "key-rate-dv01",
    "spread-dv01",
    "oas",
    "var",
    "stress-test",
    "p&l-attribution",
    "carry",
    "roll-down",
    "factor-risk",
    "frtb",
    "quantitative-finance",
    "python",
  ]
category: "trading"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

A pricing system tells you what each bond is worth. An *analytics* system tells you what risks the book is carrying, where the P&L came from yesterday, what scenarios would hurt today, what capital is consumed by each position, and what trades would best rebalance the book. The pricing layer answers the bond-by-bond question; the analytics layer answers the *book-level* question. They are operationally distinct, often run by separate teams, and built with different engineering priorities. Pricing prizes accuracy and reproducibility; analytics prizes throughput, multi-dimensional aggregation, and slice-and-dice query speed.

![Fixed-income analytics: aggregating risk across thousands of positions](/imgs/blogs/fixed-income-analytics-1.png)

The diagram above is the mental model. Inputs are positions, prices, curves, spreads, ratings, sectors, funding rates, and accruals. The aggregator combines position-level inputs into portfolio-level metrics broken down across many dimensions: by sector, by rating, by tenor bucket, by issuer, by trader, by strategy. Outputs are the daily reports that drive the desk: total DV01, key-rate DV01s, spread DV01, VaR, scenario P&L, factor risk, performance attribution, capital consumption.

This article is the deep dive on fixed-income analytics for a senior quant or staff-level engineer. It covers DV01 aggregation, key-rate DV01s, spread DV01s, OAS analytics for embedded-option bonds, scenario stress testing, VaR (historical, parametric, Monte Carlo), P&L attribution, carry and roll-down, factor models (PCA and macroeconomic), liquidity analytics, capital and regulatory analytics, production architecture, and a long catalog of named analytics failures.

The companion articles are [Bond Pricing](/blog/trading/quantitative-finance/fixed-income/bond-pricing) (which covers the pricing layer) and [Yield Curve Modeling](/blog/trading/quantitative-finance/fixed-income/yield-curve-modeling) (which covers the curve infrastructure). Both are necessary inputs to the analytics described here.

## 1. Why analytics is its own discipline

The single biggest mental shift moving from pricing into analytics is to internalise that the questions are *fundamentally different*. Pricing asks: given today's market, what is this bond worth? Analytics asks: given today's positions and risk metrics, what should the trader do, what should the risk committee approve, and what does the regulator need to see?

This shift drives different engineering priorities:

- **Pricing**: per-bond accuracy to a basis point. Reproducibility from versioned inputs. Sub-second per bond.
- **Analytics**: throughput across 100K+ positions. Multi-dimensional aggregation (sector × rating × tenor × issuer). Sub-100ms slice-and-dice queries on aggregated cubes.

Pricing is *vertical* (deep on each instrument); analytics is *horizontal* (broad across the book). A pricing engine that handles all instrument types but cannot aggregate across them is incomplete; an analytics engine that aggregates well but is wrong on a few instruments produces nonsense. The two layers must be tightly coupled but separately optimised.

A second mental shift: *analytics is an interactive workflow*. Traders and risk managers explore the book by clicking through dimensions: total DV01 by sector, then by rating within sector, then by issuer within rating. Each click should return in <100ms. The engineering pattern is a *pre-aggregated cube* with rolled-up totals at every dimension intersection, plus efficient drill-down primitives.

A third shift: *historical analytics matters*. Daily reports compare today's metrics to yesterday's, last week's, last month's. The analytics service must store and query historical snapshots efficiently. A 3-year history of daily DV01 by 1000 sub-categories is a billion-cell time-series; the storage and query infrastructure is non-trivial.

A fourth shift: *analytics drives compensation*. Trader bonuses key off P&L attribution; risk-committee escalations key off VaR breaches; capital allocations key off sector concentrations. The analytics outputs are not just informational — they directly affect dollars in the firm. Errors are not abstract; they show up in salaries and bonuses.

A senior fixed-income engineer treats the analytics layer with at least as much engineering investment as the pricing layer. In some firms the analytics team is larger than the pricing team because the surface area is larger.

## 2. DV01 aggregation

The simplest and most-used analytics: aggregate position-level DV01 into portfolio-level totals.

![DV01 aggregation: rolling individual sensitivities into book risk](/imgs/blogs/fixed-income-analytics-2.png)

Position-level DV01 is computed once per day per position by the pricing engine:

$$
\text{DV01}_i = D_{\text{mod},i} \cdot P_i / 10000
$$

where $D_{\text{mod},i}$ is the modified duration and $P_i$ is the dirty price. For embedded-option bonds, use effective duration:

$$
\text{DV01}_i = -\frac{P_i^+ - P_i^-}{2 \cdot \Delta y \cdot 10000}
$$

with $\Delta y = 1$ bp.

Portfolio aggregation is *linear*:

$$
\text{DV01}_{\text{portfolio}} = \sum_i \text{DV01}_i.
$$

The aggregation introduces no model risk; it's exact. The engineering challenge is keeping all 100K positions consistent and current. Every position must have the latest:

- Pricing snapshot.
- Curve snapshot.
- Spread / OAS snapshot.
- Position quantity (any trades since last snapshot).
- Sector, rating, tenor, issuer attributes.

A serious analytics service tracks the *staleness* of each input and refuses to aggregate when staleness exceeds threshold. A "total DV01" computed against a stale price is misleading.

Standard breakdowns:

- **By sector**: corporates / agencies / MBS / munis / sovereigns / supranationals.
- **By rating**: AAA / AA / A / BBB / IG / high yield.
- **By tenor**: 0-2y / 2-5y / 5-10y / 10-30y / 30y+.
- **By issuer**: top 10 names with DV01.
- **By currency**: USD / EUR / GBP / JPY / other.
- **By trader / strategy**: each trader's book; each strategy's book.
- **By legal entity**: separate aggregations for different regulated entities.

A serious analytics dashboard shows total DV01 in a header and any of these breakdowns in click-to-expand panels.

```python
def aggregate_dv01(positions, group_by):
    """Aggregate DV01 across positions, grouped by attribute."""
    from collections import defaultdict
    totals = defaultdict(float)
    for p in positions:
        key = tuple(p[attr] for attr in group_by)
        totals[key] += p['dv01']
    return totals


## Example: aggregate by sector + rating
breakdown = aggregate_dv01(book.positions, group_by=['sector', 'rating'])
```

For 100K positions, the aggregation is microseconds; the engineering is in the cube architecture that pre-computes all rollups. Modern in-memory analytics use Apache Arrow / DuckDB or proprietary cube engines.

### 2.1 The aggregation cube in detail

The architectural heart of an analytics service is the *cube* — a multi-dimensional aggregation of position-level metrics. A cube has:

- **Facts** (what is being aggregated): DV01, market value, P&L, accrued interest, capital, etc.
- **Dimensions** (how it is aggregated): position, sector, rating, tenor bucket, issuer, currency, trader, strategy, legal entity, settlement date, etc.
- **Hierarchies** (drill-down paths): sector → sub-sector → issuer; tenor bucket → tenor → settlement date; rating → rating bucket → issuer.

A typical bank's cube has:
- 100K-500K positions (rows).
- 50-100 dimensions.
- 10-20 fact measures.

Pre-aggregated rollups across all hierarchy intersections explode the storage to ~$10$-$100$ GB per snapshot. Modern cubes use columnar formats (Apache Arrow, Parquet) and support sub-100ms slice-and-dice queries on billion-cell aggregates.

The query language is typically a SQL dialect with cube-specific extensions (CUBE, ROLLUP, GROUPING SETS). DuckDB and ClickHouse are state-of-the-art open-source engines; commercial alternatives include kdb+ (high-end fixed-income standard) and proprietary in-house systems.

A senior architect designs the cube schema carefully:
- Which dimensions are *first-class* (always rolled up)?
- Which are *second-class* (materialised on demand)?
- Which intersections are pre-computed vs computed at query time?

Decisions made at schema design compound for the system's life. A cube that did not anticipate liquidity dimensions cannot easily add them later.

### 2.2 Versioning and time-travel queries

A serious analytics cube supports time-travel queries: "What was the firm's total DV01 at 14:32 last Wednesday?" This requires:

- **Immutable snapshots** of every input (positions, prices, curves) at sub-minute granularity.
- **Versioned cube updates** rolling forward from each snapshot.
- **Index on (snapshot_id, dimension_keys)** for efficient time-bounded queries.

Time-travel is essential for:

- Regulatory audits ("show your VaR computation at each end-of-day for the past quarter").
- Trade attribution disputes ("at what mark was this trade?").
- Post-mortem analysis ("what did the analytics show 2 hours before the loss?").

Production systems run multi-terabyte time-series databases for these queries. The engineering investment is substantial; the regulatory and audit value is immense.

## 3. Key-rate DV01s

Total DV01 measures sensitivity to a parallel curve shift. Real curves move non-parallel: the 2y might rise while the 30y falls (steepening). Key-rate DV01s decompose total DV01 across maturity buckets to reveal *curve* exposure.

![Key-rate DV01s: revealing curve exposure](/imgs/blogs/fixed-income-analytics-3.png)

The procedure: for each key tenor (2y, 5y, 10y, 30y), shift only that knot of the yield curve by 1 bp; recompute every position's price; take the difference. The KR DV01 of position $i$ at key tenor $j$ is

$$
\text{KR-DV01}_{i,j} = -\frac{\partial P_i}{\partial y_j} \cdot 10^{-4}.
$$

Aggregating across positions:

$$
\text{KR-DV01}_j = \sum_i \text{KR-DV01}_{i,j}.
$$

Sum invariant: $\sum_j \text{KR-DV01}_j = \text{DV01}$ (under the right scaling and bucketing).

A worked example. Consider three positions:
- Long $1B notional 10y Treasury, DV01 +$769K.
- Short $0.5B notional 5y Treasury, DV01 -$225K.
- Short $0.5B notional 30y Treasury, DV01 -$880K.

Total DV01 = +$769K - $225K - $880K = -$336K.

Key-rate DV01:
- 5y bucket: -$225K (from the 5y short).
- 10y bucket: +$769K (from the 10y long).
- 30y bucket: -$880K (from the 30y short).

The position is short total duration ($-336K$) but the key-rate breakdown reveals a strong butterfly trade: long the 10y body, short the 5y and 30y wings. A parallel curve move is mostly hedged; a steepener (5y/30y move differently) creates large mark-to-market swings.

Standard bucket choices:

- **Coarse**: 2y, 5y, 10y, 30y (4 buckets).
- **Standard**: 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y (8 buckets).
- **Fine**: 1y, 2y, 3y, 5y, 7y, 10y, 12y, 15y, 20y, 25y, 30y (11 buckets).

Trade-off: more buckets = better resolution but more computation cost. Standard banks use 8-bucket; large dealers use 11-bucket; some hedge funds use 16+ buckets for very curve-sensitive trades.

Production tip: KR DV01 computation is essentially free if the pricing engine uses AAD (algorithmic differentiation). One forward pass + one backward pass returns all bucketed DV01s simultaneously. Naive bumping is $K$ revaluations where $K$ is the number of buckets; AAD is one.

## 4. Spread DV01 and credit risk

For credit-bearing bonds (corporates, MBS, sovereigns, munis), the price is sensitive to *both* the risk-free rate and the credit spread. Two distinct sensitivities matter:

- **Rate DV01**: sensitivity to a parallel shift in the risk-free curve.
- **Spread DV01**: sensitivity to a parallel shift in the credit spread.

![Spread DV01 and credit risk metrics](/imgs/blogs/fixed-income-analytics-4.png)

For a vanilla corporate bond, rate DV01 ≈ spread DV01 numerically (both equal modified duration times price / 10000). But they hedge with different instruments:

- Rate DV01 hedges with Treasuries or interest-rate swaps.
- Spread DV01 hedges with CDS, CDX index, or sector ETFs.

The two sensitivities can diverge for callable bonds, MBS, and other embedded-option products. A callable bond's rate DV01 < spread DV01 because rate moves affect call probability; spread moves don't.

Beyond simple spread DV01, more granular credit metrics:

- **Sector spread DV01**. Bump the spread of one sector (banks, industrials, energy, utilities) by 1 bp; recompute. Reveals sector concentration.
- **Rating bucket DV01**. Bump spreads at each rating bucket separately. A book with $50M total spread DV01 might be 70% in BBB and 20% in BB; rating-migration risk is concentrated.
- **Single-name spread DV01**. Bump the spread of one issuer by 1 bp. Useful for managing concentration risk to specific issuers (e.g., financials in 2008).

Hedge instruments:

- **CDS** on individual issuer: hedges single-name spread risk.
- **CDX / iTraxx** index: hedges aggregate sector or rating-bucket spread risk.
- **Sector ETFs** (LQD, HYG): hedge sector-level credit beta.
- **Rating-bucket ETFs**: harder to find; some firms use synthetic baskets.
- **Cross-currency overlay**: for emerging-market sovereigns where credit spread is correlated with FX.

A typical sell-side fixed-income desk runs an *aggregate* spread DV01 limit (e.g., $5M total spread DV01), plus bucketed limits per sector and per single name. Limit breaches escalate to risk committee.

### 4.1 The interplay between rate DV01 and spread DV01

For credit-bearing bonds, the total mark-to-market change decomposes into:

$$
\Delta P / P \approx -D_{\text{rate}} \cdot \Delta y_{\text{Treas}} - D_{\text{spread}} \cdot \Delta s + \text{cross terms}.
$$

In normal markets these two sensitivities are similar in magnitude for vanilla corporate bonds (both ≈ modified duration). But the *moves* of rate vs spread are often anti-correlated: in flight-to-quality, Treasuries rally and credit spreads widen. So a long-corporate book is exposed to both directions of this anti-correlation.

A senior credit-portfolio manager tracks the *sum* of rate DV01 and spread DV01 separately:

- Total rate DV01: aggregated rate sensitivity. Hedge with Treasuries.
- Total spread DV01: aggregated spread sensitivity. Hedge with CDS or sector ETFs.

In stress (2008, 2020, 2022 Q4), the two sensitivities can diverge. Rate DV01 might fall (Treasuries rally) but spread DV01 grows (credit blows out). Hedge ratios must be re-balanced as market regimes shift.

### 4.2 Rating-migration risk

Beyond simple spread movement, credit-bearing bonds carry *rating-migration risk*. A BBB bond downgraded to BB sees its credit spread widen by 50-200+ bp over a few weeks. The risk is discrete (event-driven) but quantifiable.

Rating-migration analytics typically include:

- **Migration matrices** (probability of moving from rating $i$ to rating $j$ over horizon $T$). S&P and Moody's publish these annually based on historical default and migration data.
- **Position-level migration risk**. Probability-weighted spread widening from each potential migration.
- **Portfolio-level migration risk**. Aggregate impact of expected and stressed migration scenarios.

A book heavily concentrated in BBB names faces higher migration risk than one spread across IG. The senior risk leader watches the migration distribution and limits BBB concentration explicitly.

## 5. OAS analytics

For embedded-option bonds (callable, puttable, MBS, ABS), the standard pricing produces three option-adjusted metrics: OAS, OAD, OAC.

![OAS analytics: option-adjusted measures for embedded-option bonds](/imgs/blogs/fixed-income-analytics-5.png)

**OAS (Option-Adjusted Spread)** is the constant basis points added to the risk-free curve such that the option-pricing model's price equals the market price. It strips out embedded-option value and gives a *credit-only* spread comparable across bonds with and without options.

**OAD (Option-Adjusted Duration)** is the effective duration computed under the OAS-bumped scenario. For a callable bond, OAD < non-callable duration because rate moves affect call probability; the bond doesn't move as much.

**OAC (Option-Adjusted Convexity)** is the effective convexity. Often *negative* for callable bonds and MBS because the gain on rallies is capped by the call.

Production computation:

1. Price the bond under the chosen rate model with embedded option (Hull-White lattice for callables, MC for MBS).
2. Solve for OAS: spread that makes model price = market price. Newton-Raphson on the unknown spread.
3. Bump rates ±1 bp; re-solve for new model prices; compute OAD by central finite difference.
4. Bump in both directions; compute OAC.

A typical OAS Monte Carlo run for a single MBS CUSIP uses 10,000 paths and takes ~100 ms. Running OAS analytics across an entire MBS book (10,000 CUSIPs) takes 15-20 minutes nightly.

```python
def compute_oas(bond_spec, market_price, rate_model, tolerance=1e-6):
    """Solve for OAS via Newton-Raphson."""
    from scipy.optimize import brentq
    def price_residual(spread):
        return rate_model.price(bond_spec, spread) - market_price
    return brentq(price_residual, -0.05, 0.10, xtol=tolerance)


def compute_oad(bond_spec, oas, rate_model, bump=1e-4):
    """Effective duration under OAS scenario."""
    p_up = rate_model.price(bond_spec, oas, rate_shift=+bump)
    p_dn = rate_model.price(bond_spec, oas, rate_shift=-bump)
    p0 = rate_model.price(bond_spec, oas)
    return -(p_up - p_dn) / (2 * bump * p0)
```

Senior credit-portfolio managers compare bonds by OAS, not Z-spread or YTM. A 5y BBB callable trading at 100 bp Z-spread but 85 bp OAS is *less attractive* than a 5y BBB non-callable trading at 90 bp Z-spread (= 90 bp OAS): the callable's spread is partly compensation for the embedded short-call.

## 6. Scenario stress

Beyond first-order risk metrics (DV01, KR-DV01), books are stress-tested against complete market scenarios.

![Scenario stress: testing the book across regimes](/imgs/blogs/fixed-income-analytics-6.png)

**Historical scenarios.** Replay actual market moves from past events:

- **1994 Fed surprise hike**. 30y Treasury yield up 150 bp over 9 months.
- **1998 LTCM crisis**. Spreads blow out; flight to quality compresses Treasuries.
- **2008 Lehman crash**. Credit spreads +500 bp; LIBOR-OIS +350 bp; Treasury rally +200 bp.
- **2010 European sovereign crisis**. Greek yields +1000 bp; peripheral sovereigns wide.
- **2020 COVID dislocation**. Spreads +400 bp; vol spike; liquidity break.
- **2022 hike cycle**. Cumulative +500 bp over 18 months.

**Hypothetical scenarios.** Designed market moves:

- **Parallel ±200 bp**. All curves shift uniformly.
- **Steepener +100/-100**. 30y up, 2y down (or reverse).
- **Inversion 2y +200 bp**. Front-end spike.
- **Credit blowout +200 bp**. All credit spreads widen.
- **Cross-currency basis +50 bp**. XCY basis widens.
- **Repo spike +200 bp**. Funding rates jump.
- **Combined**: +200 bp parallel + $100 bp$ credit blowout + 50% liquidity reduction.

**Reverse scenarios.** Find the market move that would produce a target loss:

- "What scenario produces a 5% portfolio loss?"
- "What scenario would breach our $50M loss limit?"

Reverse scenarios identify hidden tail risks the standard scenarios miss.

The pipeline:

1. Apply scenario shifts to inputs (curves, spreads, vols).
2. Reprice every position under shifted inputs.
3. Aggregate P&L across positions.
4. Compute breakdowns (which sector lost most, which rating, which trader).

Full book stress takes seconds-to-minutes depending on book size and pricing complexity. Standard cadence: daily for short scenarios, weekly for full historical replay. Intraday triggered runs on threshold market moves.

A senior risk committee reviews stress P&Ls weekly; persistent worst-case losses exceeding tolerance force position reduction. The 2022-2024 hike cycle worked through standard scenarios within 18 months — books that survived had been stressed against worse scenarios in advance.

### 6.1 Stress-test design principles

Designing effective stress scenarios is more art than science. A senior risk leader's principles:

**Pick scenarios that hurt structurally.** A book that is long-rate-short-spread is hurt by a rate rally + spread widening combination. The stress scenarios should target this combination, not generic shocks.

**Include cross-asset propagation.** A "rates up 200 bp" scenario should also include FX moves, equity selloff, credit blowout. Real markets don't move in isolation.

**Test reverse scenarios.** Find the move that produces a specific loss target. This often reveals exposures the linear stress missed.

**Calibrate scenarios to historical extremes.** Don't pick "1 standard deviation" if you've been through 5-sigma events recently. Calibrate to actual realized stress.

**Update scenarios after each crisis.** 2008 added LIBOR-OIS scenarios; 2014 added negative-rate scenarios; 2020 added COVID-style scenarios. The scenario library grows.

**Run reverse scenarios as regular discipline.** Quarterly: find the worst-case scenario from the firm's perspective. Review what would have to happen to realise it.

A serious bank maintains a *scenario library* of 50-200 named scenarios, each with detailed market shifts. Scenarios are tagged by category (rates, credit, FX, cross-asset, geopolitical) and reviewed semi-annually.

### 6.2 The hidden cost of stress testing

Stress testing infrastructure is more expensive than people realise:

- Computing stress P&L on 100K positions across 100 scenarios takes 10K position-revaluations per scenario × 100 = 1M revaluations. At 10 ms per revaluation, that's 3 hours of compute.
- Doing this nightly requires dedicated infrastructure.
- Storing historical stress results for trend analysis adds 100GB per quarter.

Smaller firms run smaller scenario sets; tier-1 banks run the full battery. The investment scales with book size.

## 7. Value-at-Risk

VaR is the threshold below which losses occur with a specified probability over a specified horizon. The 99% 1-day VaR of a portfolio is the loss that has a 1% probability of being exceeded in any single day.

![VaR: the daily loss budget](/imgs/blogs/fixed-income-analytics-7.png)

Three computation methods:

**Historical VaR.** Take the position-level Greeks; apply yesterday's, day-before's, ... 250+ historical days of curve / spread / vol shifts to compute hypothetical P&L; rank the resulting P&Ls; the 99th percentile loss is the VaR.

```python
def historical_var(positions, historical_shifts, percentile=0.99):
    """99% historical VaR."""
    pnls = []
    for shift in historical_shifts:
        pnl = sum(p['dv01'] * shift['rate']
                  + p['spread_dv01'] * shift['spread']
                  + p['vega'] * shift['vol']
                  for p in positions)
        pnls.append(-pnl)  # loss is negative P&L
    return sorted(pnls)[int(len(pnls) * percentile)]
```

Pros: model-free, no parametric assumptions.
Cons: assumes past distribution = future; sample size limited (250 trading days = 2.5 events at 1% level); slow to react to regime changes.

**Parametric VaR.** Assume position-level P&L is jointly normal; compute portfolio-level standard deviation from a covariance matrix of risk factors; VaR = $z \times \sigma_{\text{portfolio}}$ where $z = 2.33$ for 99%.

Pros: fast, tractable, easy to attribute.
Cons: Gaussian assumption ignores fat tails; underestimates extreme risk.

**Monte Carlo VaR.** Simulate $N$ joint market scenarios under a calibrated multivariate model; reprice positions on each scenario; rank P&L; take percentile.

Pros: handles non-linear positions, fat tails, non-Gaussian.
Cons: slow; model risk in the scenario generator.

Operational standards:

- Regulatory standard is 99% 1-day VaR (some regulators require 99.9%).
- Internal limits are typically much tighter than regulatory minimums.
- VaR breaches are tracked via *back-testing*: if 99% VaR is breached in more than ~3% of days over a year, the model is under-estimating.
- Persistent breaches require model recalibration or methodology change.

Senior risk teams understand VaR has known weaknesses (especially at the 1% / 0.1% tails) and complement it with stress testing.

## 8. Performance attribution

P&L attribution decomposes daily returns into contributing sources.

![Performance attribution: where did the P&L come from](/imgs/blogs/fixed-income-analytics-8.png)

The standard decomposition:

$$
\text{daily P\&L} \approx \underbrace{\sum_j \text{KR-DV01}_j \cdot \Delta y_j}_{\text{rate move}} + \underbrace{\sum_s \text{Spread DV01}_s \cdot \Delta s_s}_{\text{spread move}} + \underbrace{\text{carry}}_{\text{coupon - funding}} + \underbrace{\text{roll-down}}_{\text{curve aging}} + \underbrace{\text{trade P\&L}}_{\text{position changes}} + \text{cross terms} + \text{residual}.
$$

Each component is computed from observed market moves and position-level Greeks:

- **Rate move**: aggregate KR-DV01 × observed daily yield shift at each tenor.
- **Spread move**: aggregate spread DV01 × observed spread move.
- **Carry**: (coupon - funding) / 365 × notional, summed over positions held overnight.
- **Roll-down**: (curve at $T-1$) - (curve at $T$) effect on bond value, holding rate constant.
- **Trade P&L**: realised P&L from any positions opened or closed.
- **Cross terms**: gamma × (rate move)², vol-of-vol effects, etc.
- **Residual**: actual P&L minus sum of attributed components. Should be small.

A persistent non-zero residual is a warning. It means either:
- The model is missing a risk dimension (e.g., a hedge fund with significant cross-asset correlation that the linear model misses).
- A position is mispriced (stale, bug, manual override).
- The Greeks are stale relative to today's positions (intra-day trades not captured).

Senior risk reviewers monitor the residual time series; persistent positive or negative residuals trigger investigation.

Granularity: attribution at the *position* level, sector level, book level, trader level. Daily for traders' compensation; weekly aggregated for risk committee; monthly for senior management.

Reconciliation: attribution sum vs actual mark-to-market P&L should match within tolerance. Mismatch is a system bug.

### 8.1 Cross-validation of attribution against actual P&L

A daily ritual at every fixed-income desk: reconcile the attribution sum against the actual mark-to-market P&L. The two should match within tolerance:

$$
\text{Sum of attribution components} - \text{actual P\&L} = \text{residual}.
$$

Tolerance varies by firm; typical: $\pm 5$ bp of book NAV. A residual exceeding tolerance triggers investigation.

Common sources of residual:

- **Stale Greeks.** Position-level Greeks computed at start-of-day; intraday changes not captured.
- **Cross terms missed.** Gamma × $(\Delta y)^2$, vol-of-vol effects, etc.
- **Late prices.** A position priced at a stale mark vs. actual settlement.
- **Trade timing.** A trade booked at one price but Greeks computed at a different price.
- **Currency conversion.** FX rate snapshot mismatch.
- **Manual overrides.** Trader-overridden marks not reflected in Greeks.

A senior P&L attribution engineer maintains a residual time series. Persistent positive or negative residuals are bugs to investigate; random fluctuations within tolerance are acceptable.

### 8.2 Attribution at different granularity levels

Attribution is performed at multiple levels:

**Position level.** Each position's daily P&L decomposed into rate move, spread move, carry, roll-down, etc. Useful for trader debugging.

**Strategy level.** Aggregate across positions in a strategy. Useful for strategy-level performance review.

**Trader level.** Aggregate across trader's book. Drives trader compensation.

**Sector level.** Aggregate by sector. Useful for sector-rotation analysis.

**Book level.** Total firm book. Senior-management reporting.

Each level has its own reconciliation; consistency across levels is verified daily.

## 9. Carry and roll-down

Carry and roll-down are the structural daily P&L when the market doesn't move. Together they are the desk's *baseline* — what they earn just by holding positions overnight.

![Carry, roll-down, and the structural P&L](/imgs/blogs/fixed-income-analytics-9.png)

**Carry per day** for a financed long position:

$$
\text{Carry} = (\text{coupon yield} - \text{repo rate}) / 365 \cdot \text{notional}.
$$

For a 5% coupon bond financed at 4.5% repo, daily carry on $1B notional is $(5\% - 4.5\%) / 365 \times 10^9 \approx \$1{,}370$. That's $\sim \$500K$ per year per $1B in financed-position carry.

**Roll-down per day**: as a bond ages by one day, it moves down the yield curve. If the curve is upward-sloping, the bond's yield falls slightly, and the price rises. The roll-down per day on a 5y bond on an upward-sloping curve might be $\sim 0.1$ bp/day in yield terms, contributing perhaps $\$8K$ per day on $1B notional.

Together: carry + roll-down = structural daily P&L when the market doesn't move. For a typical fixed-income book, this is $\sim 1$-$3$ bp per day or $\sim 5$-$15\%$ per year.

The implication: a book that just *holds positions* earns this baseline. The trader's job is to add *alpha* on top — relative-value trades that earn more than carry + roll-down. Active trading should beat passive holding net of trading costs.

A subtle but important point: *carry compensates for risk*. A junk bond yields 8% and finances at 4%; the 400 bp carry compensates the holder for default risk and liquidity risk. The carry is not free; it's a *risk premium*. Senior traders know to size positions by carry-adjusted-risk, not raw carry.

## 10. Factor models

Beyond simple aggregations, factor models reduce many position-level exposures to a handful of orthogonal risk factors.

![Factor models: PCA and macroeconomic factors](/imgs/blogs/fixed-income-analytics-10.png)

**PCA factors** (we covered the math in [the yield curve modeling post](/blog/trading/quantitative-finance/fixed-income/yield-curve-modeling#13-curve-risk-pca-decomposition)):

- PC1 (level): 70-80% of curve variance.
- PC2 (slope): 10-15%.
- PC3 (curvature): 3-5%.
- Higher PCs: rapidly diminishing.

For a fixed-income book, project key-rate DV01s onto PCA factors:

$$
\text{Exposure}_{PC_k} = \mathbf{v}_k^\top \mathbf{d}
$$

where $\mathbf{v}_k$ is the $k$-th PCA loading vector and $\mathbf{d}$ is the key-rate DV01 vector.

PCA factors are *orthogonal by construction*; they decompose total risk into uncorrelated components. They are also stable in time (PCA loadings change slowly), making them suitable for long-term risk management.

**Macroeconomic factors** are interpretable but non-orthogonal:

- Real growth (GDP, employment).
- Inflation expectations (5y5y, breakeven).
- Monetary policy stance (Fed funds, ECB deposit, BoJ).
- Credit / risk appetite (CDS index, VIX).

The hybrid approach:

1. Project portfolio exposures onto PCA factors.
2. Regress PCA factors against macro factors via OLS over historical data.
3. Multiply the two projections to get macro-factor exposures.

Output: portfolio risk decomposed into "exposure to growth," "exposure to inflation," "exposure to credit appetite." Senior risk committees report both PCA-based and macro-based factor exposures.

A typical factor risk dashboard shows:

| Factor | Exposure | Daily volatility | Daily VaR (95%) |
| --- | --- | --- | --- |
| PC1 (level) | +$5M DV01 | 5 bp | $40M |
| PC2 (slope) | -$2M | 8 bp | $25M |
| PC3 (curvature) | +$0.5M | 5 bp | $4M |
| Inflation exp | +$2M | 4 bp | $13M |
| Credit appetite | -$10M | 6 bp | $98M |

The credit-appetite exposure dominates; the desk is heavily short credit appetite. A risk-on rally would hurt; a risk-off selloff would help. Senior traders read this dashboard and adjust positions to balance exposures or to express specific views.

### 10.1 Factor model construction in detail

Building a fixed-income factor model is a multi-step process:

**Step 1: Choose factors.** PCA factors are extracted from historical curve changes; macro factors are predefined (real growth, inflation, credit appetite, FX, etc.).

**Step 2: Estimate loadings.** Each position's exposure to each factor is computed via projection of its key-rate DV01s onto the factor loadings. For PCA, this is a matrix multiplication; for macro, this is a regression.

**Step 3: Aggregate.** Sum factor exposures across positions to get portfolio-level factor exposures.

**Step 4: Compute factor risk.** Multiply factor exposures by factor volatilities (computed from historical data) to get factor-level VaR contributions.

**Step 5: Account for correlations.** PCA factors are orthogonal; macro factors require a correlation matrix. The portfolio VaR is computed via the standard quadratic form.

**Step 6: Decompose attribution.** When factor risk is realised, attribute the actual P&L back to factor moves. This closes the loop.

A serious factor model is rebuilt monthly with the latest historical data. Loadings are stable but not static; gradual drift is expected.

### 10.2 Factor stability and the danger of overfitting

A common pitfall: building a factor model with too many factors. With 50+ factors, the model fits historical P&L well but generalises poorly to future regimes. The factors capture noise rather than structural risk.

A senior risk leader's preference: 5-10 factors with clear interpretation. PC1, PC2, PC3 (curve shape); inflation; credit appetite; FX; perhaps 2-3 sector factors. Beyond that, marginal explanatory value diminishes rapidly.

Overfit factor models often produce paradoxical results: portfolio risk decreasing as positions grow, factor exposures changing sign without economic justification. Senior reviewers spot this; junior modellers often don't.

## 11. Liquidity analytics

Not all DV01 is equal. A long position in a 30y muni issue with $50K daily volume is vastly less liquid than the same DV01 in 30y Treasuries with $5B daily volume. Liquidity analytics quantify this difference.

![Liquidity analytics: not all DV01 is equal](/imgs/blogs/fixed-income-analytics-11.png)

Standard metrics:

**Days-to-liquidate**. Position size / average daily volume, capped at some fraction of ADV (typically 5-25%). For a $1B muni position with $50K ADV at 5% participation rate, that's $1B / ($50K × 0.05) = 400 trading days = 1.5 years.

**Bid-ask cost**. Typical spread in basis points. Treasuries 1-2 bp, IG corporates 5-15 bp, HY 50-200 bp, illiquid munis 100-500 bp. Estimated transaction cost on liquidation.

**Market depth**. The size at which the next price tick widens by a defined amount. A $100M block at 1 bp wider; a $1B block at 5 bp wider.

**Liquidity score**. Composite metric, 0 (deepest) to 100 (near-impossible). Firm-specific weighting of the above metrics.

**Stressed liquidity**. Days-to-liquidate × stress factor (typically 2-5x). Captures the reality that liquidity dries up in stress when liquidation is needed most.

A *risk-adjusted* DV01 multiplies raw DV01 by liquidity score. A book with $50M total DV01 might have $20M in liquidity-adjusted DV01 (the liquid Treasuries) and $30M in raw illiquid DV01 that would face severe slippage.

Senior risk frameworks include explicit *liquidity caps* on inventory:

- Total inventory cap by issuer (e.g., max 20% of issue size).
- Days-to-liquidate cap (e.g., max 30 days at 5% ADV).
- Stress-liquidity P&L cap (e.g., position must survive 50% liquidation discount).

A position that violates a cap forces position reduction or escalates to risk committee.

## 12. Capital and regulatory analytics

Post-2008, regulatory capital analytics became first-class. Each position consumes regulatory capital; trade-level economics include the capital cost.

![Capital and regulatory analytics](/imgs/blogs/fixed-income-analytics-12.png)

**FRTB market-risk capital**. The Basel III standard for trading-book market risk. Two methods:

- **Standardised**: regulator-specified risk weights per instrument, tenor, sector. Simpler, generally higher capital.
- **Internal models**: bank's own VaR / ES models, regulator-approved. Generally lower capital but with model-validation overhead.

Most major banks use internal models for the bulk of their fixed-income trading book.

**Leverage ratio capital**. Basel III leverage ratio caps the bank's gross-notional / Tier 1 capital ratio. For fixed-income with large gross notionals, this can bind even when market risk is small.

**CVA capital**. Capital against counterparty credit risk on derivatives. Significant for un-collateralised trades.

**Counterparty credit risk capital**. Separate from CVA, this is capital against potential future exposure on bilateral derivatives.

Pre-trade economics:

- Gross trade revenue: bid-ask spread × notional.
- Funding cost: difference between trade-funded rate and risk-free.
- Capital cost: trade-attributed regulatory capital × hurdle rate (typically 8-15%).
- Net economic profit: gross - funding - capital.

A trade earning 50 bp gross may consume capital costing 30 bp at the desk's hurdle rate. Net is 20 bp. If the desk's threshold is 15 bp, this trade clears.

Production quoting systems integrate capital computation into the pre-trade workflow. A trader requesting a quote sees not just the gross spread but also the capital cost and net economic profit. Senior risk leaders insist on this integration; trades that look attractive on gross but consume disproportionate capital are flagged.

### 12.1 The capital-aware quoting flow

Modern fixed-income desks integrate capital cost into the quote workflow:

1. **Quote request received.** Salesperson asks for a price on a $100M notional 5y BBB bond.
2. **Pre-trade capital computation.** The system computes capital impact: market risk capital, counterparty credit capital, leverage usage.
3. **Spread calculation.** Required spread = funding cost + capital cost × hurdle rate + risk reserve + profit margin.
4. **Quote generated.** System produces a bid-ask quote that covers all costs.
5. **Trader override.** Trader can adjust based on relationship, market view, or competitive pressure.
6. **Quote sent.** Within ~100 ms of request.

The pre-trade capital computation is the senior architecture point. Without it, traders quote based on intuition and the firm absorbs capital costs as overhead. With it, every quote reflects true economics.

### 12.2 Regulatory reporting workflows

Beyond risk management, the analytics service feeds regulatory reports:

- **FRTB market-risk capital.** Daily computation per regulator-defined methodology; submitted weekly.
- **Liquidity coverage ratio (LCR).** Bank-wide liquidity computation; daily.
- **Net stable funding ratio (NSFR).** Bank-wide funding stability; daily.
- **Large exposures reporting.** Single counterparty concentration; weekly.
- **Stress test submissions.** Annual / semi-annual to Federal Reserve, ECB, BoE, etc.
- **Credit-loss provisioning.** Forward-looking expected credit loss; quarterly.

Each report has its own methodology, format, and submission timeline. The analytics service must support all of them with full audit trails.

A senior architect plans the analytics service to support multiple reporting purposes from a single underlying data model. Rebuilding the data pipeline for each new regulatory requirement is impractical; designing for reuse is essential.

## 13. Production architecture

The analytics service has its own architecture distinct from the pricing service.

![Production architecture: the analytics service](/imgs/blogs/fixed-income-analytics-13.png)

**Inputs.** Prices and Greeks from the pricing service; positions and attributes from the position management system (PMS); market data snapshots; scenario definitions.

**Engine.** An in-memory cube. Every position has a row; every aggregation dimension has a column. Standard rollups (sum DV01 by sector × rating) are pre-computed at materialisation time. Slice-and-dice queries hit the cube and return in <100ms.

Modern technology stacks:

- **Apache Arrow** for columnar in-memory representation.
- **DuckDB** or **Polars** for SQL-style aggregations.
- **Parquet** for persistence.
- **Time-series databases** (InfluxDB, TimescaleDB) for historical analytics.
- **GraphQL** or REST APIs for consumer access.
- **React / D3.js** dashboards.

A high-end analytics service handles 100K+ positions, 50+ aggregation dimensions, 10K+ historical snapshots, with sub-100ms query latency. The engineering investment is comparable to a tech startup's analytics platform.

**Outputs.**

- **Trader dashboards.** Real-time book metrics; click-through to drill down.
- **Risk committee reports.** Weekly aggregated; PDF or interactive web.
- **Regulatory reports.** Standardised formats per regulator (FRTB, FFIEC, etc.).
- **Capital attributions.** Trade-level capital consumption.
- **Attribution histories.** Daily P&L attribution time series.

**Tech stack considerations.** The analytics tech stack is *separate* from the pricing tech stack at most banks. Pricing is C++ / Java for accuracy and reproducibility; analytics is Python / SQL / web tech for productivity and querying. The integration point is well-defined APIs.

## 14. Failure modes

Analytics failures usually look like *small numerical differences* that compound into wrong actions.

![Failure modes: where analytics goes wrong](/imgs/blogs/fixed-income-analytics-14.png)

**Stale price**. A position is marked at yesterday's price; the analytics aggregation reflects yesterday's risk. The trader sees a metric and acts on it; the action is wrong by the price drift overnight.

**Aggregation bug**. A position is double-counted (e.g., included via two different paths in the cube), or missing entirely. Total DV01 is wrong; sector breakdowns mismatch.

**Definition mismatch**. DV01 in zero-rate space vs par-rate space; different by a few percent. Two consumers of the analytics see different numbers; the divergence is a bug, not a feature.

**Sector / rating classification**. A bond is miscategorised (a financial reclassified as an industrial); sector totals are wrong; rating bucket totals are wrong.

**Stale curve**. Pricing engine uses yesterday's curve; risk engine uses today's. Greeks are off by the difference.

**Position quantity mismatch**. PMS shows 1.5x what the analytics has (a trade processed in PMS but not yet flowed to analytics). Aggregations are off by the missing trade.

**Currency conversion**. Cross-currency book aggregated under wrong FX. A 5% FX move shifts cross-currency totals by 5%.

A senior analytics engineer maintains a daily reconciliation suite:

- Sum of trader-attributed DV01 = total firm DV01.
- Sum of attribution components = actual P&L.
- Cross-check yesterday's prices against today's after rolling forward.
- Reconciliation against external risk providers (Bloomberg, Risk Metrics).

Daily reconciliation catches most failures within hours.

## 15. Case studies

### 15.1 Fannie Mae 2002 mark-to-model

Fannie Mae used internal models to mark its derivative positions; the resulting marks were systematically different from external benchmarks. Investigation revealed multiple analytics issues: stale curves, definition mismatches, model parameter drift. The firm restated $11B of earnings. The lesson: analytics outputs must be validated against external benchmarks, not just internally consistent.

### 15.2 Bear Stearns 2007 super-senior CDOs

Bear Stearns marked its super-senior CDO positions at theoretical model prices that diverged from where similar instruments traded in the market. Analytics inputs (correlation, recovery, default intensity) had not been recalibrated to the deteriorating mortgage market. Mark-to-model produced numbers 30-50% higher than mark-to-market on the same risks. The 2007 collapse revealed the gap. The lesson: analytics that doesn't reconcile to market is a fiction; reconciliation is essential.

### 15.3 LTCM 1998 VaR underestimation

LTCM used parametric VaR with Gaussian assumptions. The 1998 Russian default and Asian crisis produced a series of 6+ standard deviation moves that the Gaussian VaR model said should never happen. The firm's VaR breaches were treated as 1-in-1000-day events; in practice they happened multiple times per year. The lesson: VaR with thin-tailed distributions systematically underestimates extreme risk.

### 15.4 J.P. Morgan London Whale 2012

JPMorgan's London CIO desk took $6B in losses on synthetic credit positions. Risk metrics (VaR, CVA) computed from the positions had been miscalibrated, understating the true risk. The position concentrations exceeded internal limits but were under-flagged because the analytics rolled them into broader, less-concentrated buckets. The lesson: limits and metrics must be defined at the right granularity; aggregation can mask concentration.

### 15.5 GE Capital 2008 commercial paper

GE Capital's $300B commercial paper portfolio froze in October 2008. Pricing systems showed valuations close to par; analytics reports showed manageable risk. The reality was a complete liquidity collapse — the risk wasn't in the prices, it was in the inability to refinance. The lesson: liquidity risk is often invisible in standard analytics; explicit liquidity-stressed metrics are essential.

### 15.6 The 2010 flash crash and pricing dislocations

For 30 minutes on 6 May 2010, mark-to-market analytics on Treasury and corporate bond positions produced wild numbers as bid-asks widened 10-20x. Risk dashboards showed alarming P&L moves; some traders panicked. The lesson: real-time analytics need explicit handling of dislocation regimes, not just point-in-time mark-to-market.

### 15.7 Long-Term Capital pension scandal

A US pension fund's analytics service incorrectly classified a portfolio of A-rated mortgage securities as AAA. The true risk was higher; the actual capital consumption was understated; downgrade losses in 2008 were larger than expected. The lesson: classification errors compound; explicit reconciliation of attributes against external sources is essential.

### 15.8 SVB 2023 HTM portfolio

We covered this in [Bond Pricing](/blog/trading/quantitative-finance/fixed-income/bond-pricing#15-8-svb-and-the-held-to-maturity-treasury-portfolio-march-2023). For analytics: SVB's risk system surely knew the HTM portfolio was deeply underwater on mark-to-market; the analytics layer should have surfaced this gap. The accounting convention hid the loss; the analytics layer's job is to make the inconvenient truth visible regardless of accounting. Apparently the gap was not surfaced clearly enough to senior management.

### 15.9 The 2022 LDI pension crisis

UK pension funds running LDI strategies had risk metrics that significantly underestimated their leverage. When yields spiked 100+ bp in days post-mini-budget, margin calls forced fire-sale liquidations. The risk analytics had not adequately stressed the leverage exposure. The lesson: leverage compounds tail risk in non-linear ways; standard linear analytics under-estimate.

### 15.10 The 2024 yen carry analytics

In August 2024, several quant funds with yen-carry positions saw analytics degrade rapidly: cross-asset correlations spiked, factor models became unstable, Greek estimates lost accuracy. Some funds had to pause trading and rebuild analytics in real time. The lesson: regime shifts can break factor models; resilient analytics include scenario-based stress alongside model-based metrics.

### 15.11 The 1994 Orange County collapse

Orange County, California's investment fund lost $1.7B in 1994 from leveraged interest-rate positions. The county's risk metrics had not adequately captured the convexity and leverage exposures; standard parametric VaR understated the risk by an order of magnitude. The lesson: leveraged positions in repo-financed inventory have non-linear risk that linear analytics under-estimate.

### 15.12 The 1998 Russian default

Russian sovereign default was a cliff-event for emerging-market portfolios. Credit spread analytics that had averaged historical defaults at "1-2% per year" suddenly faced 50%+ losses. Risk metrics that assumed continuous spread distributions failed catastrophically. The lesson: discrete default risk requires explicit jump-modelling; continuous spread analytics miss the bimodal nature of default.

### 15.13 The 2007 quant fund deleveraging

In August 2007, several quantitative equity funds simultaneously deleveraged due to overlapping factor exposures across the industry. Many had analytics that captured *their* book's risk in isolation; few captured the cross-firm correlation of factor exposures. The lesson: industry-wide concentration in similar strategies creates systemic risk invisible to single-firm analytics.

### 15.14 The 2020 March COVID dislocation

In March 2020, even Treasury market liquidity broke. Analytics dashboards showed alarming P&L moves driven not by fundamental risk but by liquidity dislocation. Several firms paused trading because their risk metrics were temporarily meaningless. The Fed's intervention restored normalcy. The lesson: in extreme dislocation, standard analytics output requires careful interpretation; "panic-pause" protocols are valuable.

## 16. When to use which analytic

| Question | First-line tool | Notes |
| --- | --- | --- |
| "What's my book's rate exposure?" | Total DV01 | Quick sanity check |
| "What about non-parallel curve moves?" | Key-rate DV01s | Standard for desk-level |
| "What's my credit exposure?" | Spread DV01 + sector breakdown | Per-sector limits |
| "How much can I lose tomorrow?" | 99% 1-day VaR | Plus stress scenarios |
| "What scenarios hurt most?" | Historical + hypothetical stress | Reverse scenarios for hidden risks |
| "What drove yesterday's P&L?" | P&L attribution | Daily |
| "Where is my structural P&L?" | Carry + roll-down | Holds when market is flat |
| "What's my factor exposure?" | PCA + macro factor models | Weekly |
| "Can I unwind without slippage?" | Liquidity analytics | Position size vs ADV |
| "How much capital does this trade consume?" | Pre-trade capital calc | Integrated into quote workflow |

## 17. Three closing principles

**Reconcile, reconcile, reconcile.** Analytics outputs must reconcile against actual P&L, against external risk providers, against the books. Persistent discrepancies are bugs.

**Granularity matters.** Aggregating at the wrong level (sector instead of issuer) hides concentrations. Define the right granularity for the question being asked.

**Liquidity is the silent dimension.** Standard analytics often ignore liquidity; in stress, liquidity is the dominant risk. Explicit liquidity metrics are not optional.

## 18. Production checklist

1. **Cube architecture** with pre-computed rollups across many dimensions.
2. **DV01 + key-rate DV01 + spread DV01** with reconciliation.
3. **OAS analytics** for embedded-option bonds.
4. **Scenario stress** (historical + hypothetical + reverse) running daily/weekly.
5. **VaR** (historical + parametric + Monte Carlo) with back-testing.
6. **P&L attribution** with sub-percent reconciliation.
7. **Factor models** (PCA + macro) with weekly refresh.
8. **Liquidity metrics** alongside risk metrics.
9. **Capital analytics** integrated pre-trade.
10. **Audit logs** of every analytic computation.
11. **Cross-validation** against external risk providers.
12. **Real-time dashboards** with sub-100ms slice-and-dice.

A service that ticks all 12 is production-grade.

### 18.1 Common analytics implementation choices

A few engineering decisions every analytics service makes:

**Sparse vs dense storage.** Most positions have non-zero exposures only on a few dimensions. Storing 100% of cells uses more memory; sparse storage saves memory but adds query complexity. Modern columnar formats handle this naturally.

**Pre-computed vs on-demand aggregations.** Pre-computing every aggregation enables sub-100ms queries but explodes storage. On-demand computation saves storage but increases query latency. Production systems pre-compute the most common queries and compute the rest on demand.

**Materialised views vs derivative metrics.** Some metrics (e.g., 99th percentile P&L over 250 days) are expensive; materialising them reduces query cost. Others (e.g., today's DV01) are computed fresh per query.

**Push vs pull architecture.** Push: positions pushed to analytics on every change. Pull: analytics polls positions periodically. Push is real-time; pull is simpler but introduces latency.

**Single-machine vs distributed.** A 1M-position cube fits in 100GB RAM on a modern server; multi-server distribution is needed for 10M+ positions or for high concurrency.

A senior architect picks each choice based on the firm's specific needs and revisits annually.

### 18.2 Testing analytics services

Testing an analytics service is harder than testing a pricing service because the inputs are large and the outputs are aggregates. Standard testing patterns:

**Unit tests on aggregation logic.** Synthetic positions; known totals; verify rollups.

**Cross-validation against pricing.** Sum of position-level metrics from pricing should equal aggregated metrics from analytics.

**Reconciliation against external sources.** Bloomberg, RiskMetrics, peer banks (where comparable).

**Time-travel reproducibility.** Re-running analytics on a historical snapshot should produce the original results.

**Stress test reproducibility.** A scenario should produce the same P&L when re-run.

**Round-trip serialisation.** Cube exported to file and re-imported should produce identical query results.

A serious analytics team runs hundreds of automated tests on every code deploy. Manual testing for edge cases.

## 19. The cultural side of fixed-income analytics

Fixed-income analytics teams are typically larger than pricing teams at major banks (5-15 quants for analytics vs 5-10 for pricing). The work is more cross-functional: analytics interfaces with traders, risk, finance, compliance, regulatory affairs.

Cultural practices that distinguish strong analytics teams:

- **Daily morning calls.** Quants, traders, risk meet to review yesterday's analytics, attribute P&L, flag anomalies.
- **Reconciliation discipline.** Every analytic output reconciled against ground truth; no silent discrepancies.
- **Scenario-thinking culture.** Beyond VaR, the team thinks in terms of "what scenario would hurt us"; weekly tabletop exercises.
- **Cross-team visibility.** Analytics dashboards available to risk, finance, regulators in real time.
- **Continuous improvement.** Each P&L attribution residual >5 bp investigated; methodology improvements deployed monthly.

A senior analytics quant's career path: 5 years building specific analytic categories; 5 more years owning a portfolio's full analytics; eventually leading the firm-wide analytics platform.

### 18.3 Practical SQL patterns for analytics queries

A few SQL patterns commonly needed in analytics service:

**Total DV01 by sector and rating:**
```sql
SELECT sector, rating, SUM(dv01) AS total_dv01
FROM positions
WHERE snapshot_date = '2026-05-03'
GROUP BY GROUPING SETS ((sector, rating), (sector), (rating), ())
ORDER BY sector, rating;
```

**Key-rate DV01 attribution:**
```sql
SELECT bucket, SUM(kr_dv01_2y) AS dv01_2y,
       SUM(kr_dv01_5y) AS dv01_5y,
       SUM(kr_dv01_10y) AS dv01_10y,
       SUM(kr_dv01_30y) AS dv01_30y
FROM positions
WHERE snapshot_date = '2026-05-03'
GROUP BY bucket;
```

**Time-travel: DV01 history of a single trader's book:**
```sql
SELECT snapshot_date, SUM(dv01) AS total_dv01
FROM positions
WHERE trader = 'JSMITH'
  AND snapshot_date BETWEEN '2026-04-01' AND '2026-05-03'
GROUP BY snapshot_date
ORDER BY snapshot_date;
```

**Top 10 issuers by DV01:**
```sql
SELECT issuer, SUM(dv01) AS total_dv01
FROM positions
WHERE snapshot_date = '2026-05-03'
GROUP BY issuer
ORDER BY total_dv01 DESC
LIMIT 10;
```

These queries should return in <100ms on a properly indexed cube. The patterns are simple; the challenge is the underlying infrastructure.

### 19.1 The day in the life of an analytics quant

A typical day for a senior fixed-income analytics quant:

**07:00.** Pre-open. Review overnight curves and any stress P&L from London/Asia desks. Check that overnight reconciliation completed without errors.

**08:00.** Morning meeting with traders. Walk through yesterday's P&L attribution. Discuss any anomalies. Plan today's analytics priorities.

**08:30 - 12:00.** Active development. Build new factor models, add new regulatory reports, optimise cube performance. Address bug reports from traders.

**12:00 - 13:00.** Lunch. Catch up on industry news.

**13:00 - 16:00.** More development; risk-committee preparation if it's that day; cross-team meetings (with traders, with risk, with finance).

**16:00 - 17:00.** End-of-day. Verify EOD analytics ran clean. Reconcile against actuals. Sign off on tomorrow's risk reports.

**17:00 - 18:00.** Personal research. Read papers, explore new methodologies, prototype features.

The role mixes deep technical work with cross-team collaboration. Senior analytics quants are valued because they understand both the math and the business.

### 19.2 Common interview questions for analytics quants

A senior analytics manager might ask candidates:

1. "Walk me through your firm's P&L attribution methodology."
2. "How would you compute key-rate DV01 for a callable bond?"
3. "What's the difference between historical and parametric VaR?"
4. "How do you reconcile the cube against actual positions?"
5. "Design a scenario stress test for a credit-portfolio book."
6. "How would you scale analytics for a 10x larger book?"
7. "What's the right granularity for sector aggregation?"
8. "How do you detect a stale price in the analytics?"

Strong candidates can articulate clean answers to all 8 plus extensions. Weak candidates get stuck on operational details.

### 19.3 Cross-team coordination patterns

Analytics quants spend 30-50% of their time on cross-team coordination. The major touchpoints:

**With traders.** Daily morning reviews; ad-hoc ticket requests for new metrics; trader-initiated bug reports.

**With pricing.** Joint debugging when analytics outputs reveal pricing bugs; agreement on metric definitions; reconciliation against pricing-engine outputs.

**With risk management.** Daily risk-committee preparation; weekly stress-test reviews; ad-hoc deep dives on specific exposures.

**With finance / accounting.** Reconciliation of P&L attribution against accounting books; explanation of model-vs-actual differences; quarterly audit support.

**With regulators.** Regulatory submissions (FRTB, capital, stress tests); occasional examination support; model approval cycles.

**With IT.** Infrastructure scaling; database tuning; security patches; deployment processes.

A senior analytics quant becomes a kind of internal-firm diplomat, fluent in the language of each constituency. The skill is rare and valuable.

## 20. The future of fixed-income analytics

Several trends shape the next decade:

**ML-augmented anomaly detection.** Machine learning models flag anomalous P&L attributions, position changes, or risk metrics. Anomalies that would take analysts hours to spot are flagged in minutes.

**Real-time analytics.** Sub-second risk metric updates as positions change, market data ticks. The engineering challenge is non-trivial; modern stacks (Apache Flink, ClickHouse) make it feasible.

**Cross-asset unified frameworks.** Equity, FX, rates, credit, commodities under one analytics framework with consistent factor models. Several major banks are building this.

**Climate / ESG analytics.** Carbon exposures, transition risks, physical risks integrated into fixed-income analytics. Required by regulators; emerging in 2024-2026.

**Automated trade-decision support.** Analytics-driven trade recommendations based on risk decomposition, liquidity, capital. Currently in research at major firms; production deployment likely 2025-2027.

**Scenario expansion via reinforcement learning.** Adversarial RL agents searching for stress scenarios that exploit the book's specific exposures. Beyond historical replay; produces tail scenarios that humans wouldn't construct.

A senior analytics quant entering the field in 2026 will likely work on at least two of these frontiers in their career.

### 20.1 The analytics-pricing handoff

A subtle but important architectural question: where does pricing end and analytics begin?

The clean separation:

- **Pricing**: produces price + Greeks + diagnostics for each position.
- **Analytics**: aggregates and analyses across positions.

The unclean middle ground:

- **Risk-Greeks computed by pricing** vs **risk-Greeks computed by analytics**. Sometimes both teams compute Greeks for different purposes; reconciliation is essential.
- **Position-level scenario P&L**. Computed by pricing (per scenario, per position) and aggregated by analytics. The pricing service must support batch scenario computation.
- **Cross-currency aggregation**. Pricing produces per-currency Greeks; analytics aggregates across currencies via FX. The FX layer is shared infrastructure.

A senior architect insists on clear interface contracts:

- Pricing publishes position-level metrics in a versioned format.
- Analytics consumes the metrics and produces aggregations.
- Both teams agree on the metric definitions explicitly.

Vague contracts produce reconciliation bugs that take months to track down. Senior teams invest in the interface as much as in the components.

### 20.2 Scaling analytics for the largest books

The largest fixed-income books at major banks have:

- 500K-1M positions.
- 100+ aggregation dimensions.
- 1000+ named scenarios.
- 10K+ historical snapshots.
- 100+ concurrent users querying the cube.

Scaling analytics to this size requires:

- **Distributed cube computation.** Spark, Dask, or proprietary cluster frameworks.
- **Tiered storage.** Hot data in memory, warm in SSD, cold in object storage.
- **Pre-computed materialised views** for common query patterns.
- **Query routing.** Different queries route to different optimised paths.

A mid-tier bank's analytics infrastructure costs $5-20M/year to operate; tier-1 banks spend $50-100M/year. The investment is non-trivial but pays back across the firm's risk-aware trading business.

### 20.3 Specific ML applications in analytics

Machine learning is increasingly applied to fixed-income analytics. Useful applications:

**Anomaly detection on P&L attribution.** Train a model on years of P&L attribution data; flag days where the residual or component sizes are anomalous. Catches bugs and regime shifts faster than manual review.

**Position-classification ML.** Bonds are tagged with sector, rating, etc. by manual rules. ML models trained on historical labels can spot misclassifications and propose corrections.

**Stress-scenario generation.** Generative models (GANs, autoencoders) trained on historical market data produce stress scenarios that explore the tail of the distribution. Adversarial RL can find specific scenarios that hurt the firm's book.

**Curve-quote validation.** Rule-based filters miss subtle quote issues; ML classifiers trained on labelled "good vs bad" historical quotes catch them.

**Capital optimisation.** Given a set of trading opportunities, ML can suggest the trade combination that maximises return per unit of capital consumed.

**Liquidity scoring.** ML models predict bid-ask spreads, days-to-liquidate from bond features. More accurate than rule-based heuristics.

A senior analytics quant who can deploy ML responsibly is valuable; the quant who deploys ML without rigorous validation is dangerous.

### 20.45 Open-source vs commercial analytics platforms

A perennial question for fixed-income teams: build vs buy?

**Open-source options:**
- **QuantLib + Python ecosystem.** Free, comprehensive for vanilla products; limited for advanced exotics or large-scale aggregation.
- **Apache Arrow + DuckDB / Polars.** Modern columnar analytics; great for cube-style queries; not finance-specific.
- **MQL5 / R / pandas.** Specialised research environments; not production-grade.

**Commercial options:**
- **Bloomberg PORT / TOMS.** Tier-1 for fixed income; expensive; less customisable.
- **MSCI BarraOne / RiskMetrics.** Established risk-analytics platform; integration cost.
- **Numerix.** Pricing + analytics; mid-tier option.
- **Risk-by-Default proprietary platforms.** Custom-built at major banks.

**In-house build.** Common at tier-1 banks; rare at smaller firms.

The decision depends on:
- Trading volume and book complexity.
- Engineering capacity.
- Regulatory requirements (model approval cycles).
- Strategic differentiation (analytics as competitive moat or commodity).

A typical mid-size firm uses commercial platforms with a thin in-house customisation layer. Tier-1 banks build extensive in-house systems. Startups often start with open-source and graduate to commercial or in-house as they scale.

### 20.5 Building analytics infrastructure from zero

For a fintech building analytics from scratch, a 12-month roadmap:

**Months 1-3.** MVP: a single-currency single-cube system. Position-level DV01 and P&L. Daily snapshots. Basic dashboard.

**Months 4-6.** Multi-currency. Sector / rating / tenor breakdowns. Historical attribution.

**Months 7-9.** Stress testing. VaR. Factor models. Capital integration.

**Months 10-12.** Real-time updates. ML-augmented anomaly detection. Cross-asset framework.

A 5-person team can deliver this scope in 12 months. The investment scales linearly with book size; small books need lighter infrastructure.

## 21. Conclusion

Fixed-income analytics is the lens through which a large, complex book is understood. The math is largely linear (DV01 aggregation), the engineering is the daily craft, and the operational discipline is what distinguishes strong teams from weak ones. Reconciliation, granularity, liquidity, and capital integration are the structural quality dimensions.

### 21.15 The hidden complexity of multi-legal-entity reporting

A practical complexity often missed: large banks have many legal entities (parent + subsidiaries), each with its own regulator, balance sheet, and reporting requirements. Analytics must aggregate consistently across legal entities while supporting per-entity regulatory reports.

The challenges:

- **Inter-entity transfers.** A trade between two entities in the same firm is invisible at firm level but visible at each entity level.
- **Currency / FX consistency.** Each entity has its own functional currency; cross-entity aggregation needs FX conversion.
- **Per-entity capital.** Each entity has separate regulatory capital; the firm-level total is not just the sum.
- **Regulator-specific reports.** Each regulator wants a different report; the analytics must produce all of them.

A senior architect at a multi-entity firm spends meaningful time on this. The complexity is operational, not mathematical, but the engineering investment is significant.

### 21.2 Three closing principles for the analytics quant

**Reconcile against ground truth.** Every metric must reconcile against actual P&L, against external benchmarks, against trader's intuition. Discrepancies are bugs, not curiosities.

**Granularity is everything.** Choose the right level of aggregation for each question. Too coarse hides risks; too fine produces noise.

**The dashboard is the product.** Traders, risk, finance, regulators all consume the dashboard. UX, performance, and clarity matter as much as accuracy.

A senior fixed-income analytics quant operates fluently in three roles: as an engineer (building the system), as a data architect (designing the cube), and as a quant (interpreting the numbers). The combination is rare and valuable.

The remaining articles in this series — [Short-Rate Models](/blog/trading/quantitative-finance/rates-models/short-rate-models-vasicek-hull-white), [Exotic Derivatives](/blog/trading/quantitative-finance/exotics/exotic-derivatives), [Autocallables](/blog/trading/quantitative-finance/exotics/autocallables), and [Cliquets](/blog/trading/quantitative-finance/exotics/cliquets) — go deeper on specific product categories.

### 20.3a Real-time analytics: the next frontier

The classical analytics service publishes once or twice per day. Modern frontier is *real-time analytics*: sub-second updates as positions and market data change.

The architecture differs significantly:

- **Streaming ingest** (Kafka, Apache Flink) for incremental updates.
- **Materialised views** that update incrementally on each input change.
- **Cache invalidation** that propagates updates to dashboards.
- **Push-based dashboards** (WebSockets, server-sent events) for traders.

The engineering challenge: maintaining sub-second latency on a billion-cell cube with 10K updates/second.

Tier-1 banks have started deploying real-time analytics in 2024-2026. The early use cases:

- Intraday hedge-rebalancing: positions changed → real-time DV01 update → automatic hedging trigger.
- Pre-trade economic check: quote requested → real-time capital impact → quote price.
- Risk-committee escalation: limit threshold approached → real-time alert with full context.

Real-time analytics is non-trivial but feasible with modern infrastructure. The investment is significant; the operational benefits are real.

### 20.4 Common analytics bugs and how to spot them

A non-exhaustive list of bugs encountered in production:

**Wrong scaling on DV01.** A factor of 100 or 10000 wrong; total DV01 reads $769K instead of $76.9M. Diagnostic: cross-check against position-level estimates.

**Mixed currency aggregation.** EUR positions added to USD without FX conversion. Diagnostic: every metric tagged with currency.

**Stale snapshot.** Positions from yesterday, prices from today. Diagnostic: every aggregation tagged with input snapshot IDs; refuse aggregation when snapshots disagree.

**Missing positions.** A trade not yet propagated from PMS. Diagnostic: position-count reconciliation against PMS daily.

**Double-counted positions.** Same trade flowed via two different paths. Diagnostic: unique trade ID validation.

**Wrong day-count in carry.** Day-count library inconsistency between pricing and analytics. Diagnostic: side-by-side comparison.

**Off-by-one in time-bucketing.** A 1-year-and-1-day bond classified as "2-year bucket" instead of "1-year bucket." Diagnostic: explicit bucket boundary tests.

**Sign error in spread DV01.** Spread DV01 reported with wrong sign; trader hedges in wrong direction. Diagnostic: long-corporate position should have positive spread DV01 (in the convention of "DV01 of an asset is positive when its yield rising hurts").

**Wrong rating bucket.** Bond's current rating not the rating bucket used. Diagnostic: rating reconciliation against issuer database daily.

**Currency-FX cross-contamination.** Cross-currency basis effect not properly attributed. Diagnostic: FX-attribution component reconciled separately.

A senior analytics engineer maintains a personal log of bugs encountered with diagnostic and fix. The log compounds.

### 21.0 The economics of analytics investment

A back-of-envelope on the value of strong analytics:

- **Bid-ask compression**: better analytics → tighter quotes → 1 bp lower bid-ask. On $1T annual gross volume, that's $100M revenue.
- **Stress-test discoveries**: analytics that catches a bad scenario before it materialises → avoidance of $10-100M loss.
- **Capital optimisation**: better attribution of capital → 5% capital reduction. On $5B regulatory capital, $250M freed.
- **P&L disputes resolution**: clear attribution → fewer disputes → less management distraction. Hard to quantify but real.
- **Regulatory cost reduction**: clean audit trails → fewer regulatory enforcement actions → $10-100M penalty avoidance.

Total: a strong analytics service generates $200-500M annual value at a tier-1 bank, against $30-100M annual cost. The ROI is high but indirect.

Smaller firms benefit proportionally; even a 10-person quant fund can justify $500K-$1M annual investment in analytics infrastructure for the risk-aware-trading benefits.

### 20.5a The trader-analytics user experience

A subtle but important consideration: the analytics dashboard is not just a reporting tool — it is the trader's primary interface to their book. UX matters as much as accuracy.

Strong dashboards have:

- **At-a-glance summary** at the top: total DV01, vega, P&L, key alerts.
- **Drill-down hierarchy** that takes one click to expand any aggregate.
- **Time-series view** showing recent history alongside current state.
- **Scenario shortcuts**: pre-defined stress scenarios with one-click application.
- **Trade-level detail** accessible from any aggregate.
- **Filtering and grouping** by user-customisable dimensions.
- **Export to Excel / PDF** for offline analysis.

Weak dashboards have:
- Cluttered screens with too many metrics.
- Hidden navigation requiring multi-click drilling.
- No history alongside current state.
- Static screens that don't update.
- No way to filter / group dynamically.

Senior analytics teams take UX seriously and iterate based on trader feedback. A dashboard that traders actually use produces better decisions; one they ignore wastes the analytics investment.

### 20.55 The relationship between analytics and front-office trading

A subtle but important question: should analytics be in the front office (alongside traders) or middle office (alongside risk)?

Arguments for front office:
- Faster iteration on trader-requested features.
- Tighter feedback loop with desk needs.
- Analytics quants understand trading intuitively.

Arguments for middle office:
- Independence from trading P&L pressure.
- Cleaner audit trail for regulatory purposes.
- Standardisation across desks.

The modern compromise: analytics *infrastructure* in middle office (cube, storage, audit), with *desk-specific analytics* in front office (custom views, trader-specific reports). The infrastructure is firm-wide; the views are desk-specific.

Senior architects design for this hybrid. The infrastructure team owns the cube; desk teams own the specific reports their traders consume. Coordination is key.

### 20.6 The maturity ladder for analytics teams

Analytics teams at financial institutions evolve through a maturity ladder:

**Level 1 (basic).** Spreadsheet-based aggregation; daily reports manually compiled. Adequate for small books in stable markets. Fails under audit pressure.

**Level 2 (functional).** Database-backed aggregation; automated daily reports. Standard at small/medium banks. Can produce DV01 and basic risk metrics.

**Level 3 (mature).** Cube architecture; multi-dimensional slicing; real-time dashboards; full P&L attribution. Standard at major banks.

**Level 4 (frontier).** Sub-second cube updates; ML-augmented anomaly detection; capital-aware quoting; cross-asset unified framework. Tier-1 banks.

A senior architect assesses the team's level and plans investment to advance. Going from Level 2 to Level 3 is typically 2-3 years; Level 3 to Level 4 is comparable.

### 21.1 The analytics-pricing-curve trinity

A final integrated view: pricing, curves, and analytics are three pillars of fixed-income engineering.

- **Pricing** answers: what is each bond worth, given today's market?
- **Curves** answer: what is today's market state in standardised form?
- **Analytics** answer: what does our book look like, under that pricing and those curves?

The three pillars depend on each other:

- Pricing needs curves; analytics needs pricing.
- Analytics drives capital decisions, which affect what trades to do; trades change positions; analytics aggregates positions.
- Curves are calibrated to liquid trades; trades produced by traders informed by analytics; loop closed.

A senior fixed-income engineering leader treats all three as a coherent system. Investment in any one without the others is wasted; balanced investment compounds.

Analytics is the daily window into a multi-trillion-dollar fixed-income business. Doing it well — accurate, real-time, multi-dimensional, reconciled, integrated — is the silent competence that powers risk-aware trading at scale. The reward is a clear-headed view of one of the most operationally complex businesses in finance.

A final reflection on craft: the senior analytics quant produces nothing tangible. No bond is priced; no trade is executed; no curve is published. What they produce is *clarity* — the ability for traders, risk managers, finance, and regulators to *see* the firm's risk in a coherent way. That clarity, day after day, is what separates a firm that operates on insight from one that operates on hope. In a multi-trillion-dollar fixed-income business, the difference between insight and hope is hundreds of millions of dollars annually.

I have come to think of analytics work as the *epistemic infrastructure* of finance. It is the system by which a firm knows what it owns, what it owes, and what could go wrong. Without strong analytics, a firm is flying blind even when it appears to know what it is doing. With strong analytics, every decision can be examined, attributed, and improved.

A useful mental model for analytics engineering: it is the *interpreter* between the messy reality of trading and the structured world of risk management. The interpreter must be precise (no information loss), fast (real-time decisions), and trusted (every consumer relies on it). Building good interpreters is hard; the analytics quant who can build them well is rare.

For engineers entering the field: the work is unglamorous on the surface and deeply consequential underneath. The math is mostly aggregation and reconciliation; the engineering is mostly database design and dashboard latency. The reward is the satisfaction of building infrastructure that everyone trusts, that everyone uses, and that nobody thinks about — exactly as good infrastructure should be.
