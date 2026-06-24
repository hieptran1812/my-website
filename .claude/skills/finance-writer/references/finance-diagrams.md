# Finance diagrams: figure kinds & semantic palette

Read this in Phase B (planning) and keep it open in Phase C (authoring). It is a **thin overlay** on blog-writer's `../blog-writer/references/diagram-authoring.md`, which owns all the *mechanics* — canvas size (2400×1600), the two font families, the exact 6 palette hexes, arrow binding, the 20-px snap grid, the anti-dead-space rule, the sharpness floor. **Read that file too in Phase C.** This file only does two things: (1) re-assign what the palette colors *mean* in finance, and (2) catalog the finance-native figure kinds and how to build them with the shared tooling.

The validator (`author-scene.mjs`) is unchanged — it still enforces the same 6 hexes, fonts, geometry, and density rules. You are not adding colors; you are using the existing ones with finance meaning.

## Semantic palette (finance meaning of the 6 allowed hexes)

Pick by meaning, never aesthetically. Max 2–3 colors per figure; everything else transparent.

| Color | Hex | Finance meaning |
|---|---|---|
| Success green | `#b2f2bb` | **Money in / gains / profit** — inflows, coupons received, dividends, the "you receive" leg, the profitable region of a payoff |
| Danger red | `#ffc9c9` | **Money out / losses** — outflows, the "you pay" leg, downside, default, the loss region of a payoff, a drawdown |
| Caution amber | `#ffec99` | **Risk / cost / friction** — fees, spreads, taxes, leverage, volatility, uncertainty, the bottleneck |
| Primary blue | `#a5d8ff` | **The instrument / entity being explained** — the bond, the option, the firm, the principal balance; the thing the figure is *about* |
| Neutral lavender | `#d0bfff` | **Counterparty / intermediary** — the bank, exchange, clearinghouse, broker, issuer; a third party to the main actor |
| Soft gray | `#e9ecef` | **Market context / backdrop** — the surrounding market, "everything else", grouping boxes, axis gridlines |

**Sign convention is load-bearing.** Inflows are *always* green and outflows are *always* red, across every figure in the post. A reader must never have to guess whether an arrow means money coming in or going out. This is checked in Phase C2.

Before/after framing: left (the naive / costly / "before" state) leans red/amber; right (the better / hedged / "after" state) leans green/blue.

## Finance-native figure kinds

Map each abstraction to the kind that matches its *shape*. Vary kinds across the post (blog-writer's diversity rule applies: ≥ 8 figures → ≥ 4 distinct kinds; no two adjacent figures share a skeleton).

### Cash-flow timeline — `timeline` engine

The bread-and-butter finance figure: time on the horizontal axis, money flows as arrows above (in, green) and below (out, red) the axis. Use for: a bond's coupon-then-principal schedule, a loan amortization, an annuity, the legs of a swap, an investment's cash flows.

- Axis = time (label periods: `t=0, 1y, 2y, …`).
- Up-arrow green = inflow; down-arrow red = outflow. Label each with the amount (`+$50`, `−$1,000`).
- Put the net / NPV as an annotation if the figure's claim is about present value.

### Payoff diagram — hand-authored element figure (XY chart)

The defining picture of options/derivatives: underlying price on the x-axis, profit/loss on the y-axis, the payoff as a line, breakeven where it crosses zero. The DSL engines don't do XY plots — **author elements directly** (lines + axis + region shading + annotations).

To clear the anti-dead-space rule (sparse line charts otherwise fail rule 8), make the chart *dense*:
- Draw both axes as lines with tick labels and units (`Stock price ($)`, `Profit / Loss ($)`).
- **Shade the regions**: profit zone tinted green (`#b2f2bb`), loss zone tinted red (`#ffc9c9`) — these rectangles fill the frame and carry meaning.
- Draw the payoff as 2–3 line segments (kinked at the strike); mark the strike and breakeven with labeled reference lines.
- Annotate `max loss = premium`, `breakeven = strike + premium` with Cascadia annotations near the relevant point.
- The numbers must equal the worked example in the prose.

### Yield curve / term-structure / growth curve — hand-authored XY chart

Same construction as a payoff diagram: x-axis (maturity in years, or time), y-axis (yield %, or value \$), a curve, labeled axes, shaded regions or reference lines (par, inversion zone). Use for the yield curve, a discount-factor curve, compound-vs-simple-interest growth, a drawdown chart.

### Balance-sheet / capital stack — `before-after` or hand-authored two-column stack

A balance sheet is two stacked columns of equal total height: **assets** on the left (blue), **liabilities + equity** on the right (amber for debt, green for equity). The visual claim is "the two sides balance". Use `before-after` (left vs right) or author two stacked-bar columns directly. Also use a single stacked column for a *capital stack* (senior debt → mezzanine → equity, riskiest on top).

### Process / mechanism flow — `pipeline` or `graph`

How something *happens* step by step: how a trade clears (you → broker → exchange → clearinghouse → settlement), how a loan is securitized, how a stablecoin mint/redeem works, how a payment rail moves money. Linear → `pipeline`; branching/merging → `graph`. Color the entity being explained blue, intermediaries lavender, the fee/risk leg amber.

### Comparison matrix — `matrix` or `grid`

Axes × choices, or instrument-vs-attribute. Use for "stocks vs bonds vs cash" (rows) × "return / risk / liquidity" (columns), or a 2×2 of risk vs return quadrants.

### Taxonomy / hierarchy — `tree`

Types of an instrument or class of players: kinds of bonds, the layers of market participants, the structure of a fund-of-funds. Top = the general category, leaves = the specific instances.

### Decision flow — `graph`

"When should you use X?" as a branching decision: conditions in nodes, outcomes at the leaves. (Put conditions *in* the nodes, not on the edges — layout engines drop unfittable edge labels.)

## Authoring notes specific to finance figures

- **Hand-authored XY charts (payoff, yield curve, distribution)** are where finance differs most from blog-writer's box-and-arrow defaults. Build them dense (shaded regions + axes + ticks + reference lines + annotations) so they pass the occupancy check, and label both axes with units. A bare two-line payoff on a blank frame *will* fail the dead-space rule — that failure means "add the shaded regions and labels", not "stretch the lines".
- **Every number on a figure must appear in the surrounding ±200 lines of prose.** Don't invent a coupon, a strike, or a rate for visual balance.
- **Caption is a thesis, not a label.** Bad: "Bond cash-flow diagram." Good: "A bond is just a stream of fixed inflows whose present value moves opposite to rates."
- **Quantify**: every figure should carry real units (\$, %, bps, years). ≥ 60% of nodes/marks carry a number or qualifier, per the shared density rule.

## Per-depth figure floors (minimums, never caps)

- Concept ≥ 3 figures
- Explainer ≥ 4
- Deep-dive ≥ 7

Most posts exceed these. The ceiling is set by content: if a paragraph introduces an abstraction and the next ~30 lines have no figure, add one.

## Abstraction inventory format (Phase B output)

For each abstract concept the post introduces, emit one bullet with:

- **Claim** (≥ 8 words): the single sentence the figure proves
- **Caption** (one sentence thesis, not a label restatement)
- **Section anchor**: the heading it sits under
- **Kind**: `cash-flow-timeline` | `payoff` | `yield-curve` | `balance-sheet` | `pipeline` | `graph` | `matrix` | `grid` | `tree` | hand-authored
- **Sketch**: which boxes / lines / arrows / labels / numbers appear

Figure count = abstraction count. Vary the kinds.
