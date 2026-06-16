---
title: "Free Cash Flow: Defining FCFF and FCFE Correctly"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero guide to free cash flow: what makes cash flow free, how to build FCFF and FCFE step by step, why each must be paired with its own discount rate, and how flattering free cash flow is one of the easiest numbers to manipulate."
tags: ["equity-research", "corporate-finance", "free-cash-flow", "fcff", "fcfe", "dcf", "valuation", "wacc", "owner-earnings", "fundamental-analysis"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A discounted cash flow valuation is only as good as the cash-flow definition it discounts, and the most common modeling error in the world is mixing up the two free cash flows. Get the definition exactly right and the rest is arithmetic.
>
> - **Free cash flow** is the cash a business has left over *after* the reinvestment it needs to sustain itself — capital expenditure and the working capital that growth ties up. It is the money that can actually be handed to the people who funded the company.
> - **Free Cash Flow to the Firm (FCFF)** is the cash available to *all* capital providers — both lenders and shareholders — before any financing flows. It is computed before interest, so it is **capital-structure-neutral**.
> - **Free Cash Flow to Equity (FCFE)** is what is left for *shareholders alone* after the lenders have been served: after-tax interest paid and any net debt repayment removed, net new borrowing added back.
> - The single rule that ties it together: **FCFF is discounted at WACC and gives you enterprise value; FCFE is discounted at the cost of equity and gives you equity value directly.** Pair the wrong cash flow with the wrong rate and your valuation is nonsense.
> - Because only a slice of capex is truly mandatory, and because stock-based compensation is a real cost dressed up as non-cash, "free" cash flow is one of the easiest numbers to flatter — by underinvesting, by capitalizing, by stretching payables, or by inventing an adjusted metric.

Picture two analysts handed the same company and the same spreadsheet. The first builds a forecast of the cash the whole business throws off, discounts it at a blended rate that reflects both the company's lenders and its shareholders, and arrives at a value for the entire enterprise. The second builds a forecast of the cash left over for shareholders after the lenders are paid, discounts it at the rate shareholders demand, and arrives at a value for the equity. Done correctly, both should land on the *same equity value*. That is not a coincidence; it is a law of the way these numbers are constructed.

But it only works if each analyst keeps the cash flow and the discount rate matched. The most expensive mistake in valuation is not a wrong growth assumption or a sloppy terminal value. It is taking a cash flow that belongs to everyone — debt and equity together — and discounting it at the rate that belongs only to shareholders, or the reverse. The valuation that results is internally incoherent, and it can be off by tens of percent in either direction. The discount rate is doing its job; the cash flow is doing its job; they are just measuring two different things and have been glued together by accident.

This piece is about getting the cash-flow definition exactly right *before* you ever pick a discount rate. We will build both free cash flows from the ground up — what makes cash flow "free" at all, how FCFF is assembled from operating profit, how FCFE is carved out of FCFF, and why the matching rule between cash flow and discount rate is not a convention you can ignore but the only thing that makes a discounted cash flow model self-consistent. The figure below is the mental model we are building toward: the same pool of cash, viewed two ways, depending on whether you are standing in the shoes of the whole firm or only the shareholders.

![A side by side comparison showing free cash flow to the firm as the whole pool available to lenders and shareholders versus free cash flow to equity as the smaller slice left for shareholders after debt is served](/imgs/blogs/free-cash-flow-fcff-vs-fcfe-1.png)

We will use a running fictional company, **Northwind Industries** — a maker of industrial machinery — so every number compounds across the piece. If you have read the companion on [the cash flow statement](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from), you already have the raw material; here we turn that statement into the precise inputs a valuation needs. By the end you will be able to build FCFF from operating profit, carve FCFE out of FCFF, reconcile the two valuation paths to the same equity value, normalize a lumpy capex year, and spot the half-dozen ways a clean-looking free cash flow number hides a business that is quietly starving itself. Let us start with the foundations.

## Foundations: what "free" actually means

Before we can argue about FCFF versus FCFE, we have to be precise about the word that does all the work: *free*. It is not a synonym for "leftover" or "profit." It has a specific meaning, and every later subtlety follows from it.

### Free cash flow is cash after the reinvestment needed to survive

A business generates cash by operating — selling products, collecting from customers, paying suppliers and workers. But not all of that operating cash is available to hand out. Some of it must go straight back into the business just to keep it running: machines wear out and must be replaced, software must be maintained, and growth ties up cash in inventory and receivables before it ever produces a dollar of profit. **Free cash flow is what remains after that mandatory reinvestment.** It is the cash you could, in principle, take out of the business without impairing its ability to keep doing what it does.

That word *could* is the whole point. Free cash flow does not mean cash that *was* distributed; a company can plow its free cash flow into share buybacks, dividends, debt repayment, or simply let it pile up on the balance sheet. Free cash flow measures *capacity to distribute* — the size of the pie that is genuinely available to the people who funded the company. That is exactly why valuation discounts free cash flow rather than reported earnings or even operating cash flow: the value of a business is the present value of the cash it can ultimately return to its owners, and free cash flow is the cleanest measure of that returnable cash.

Contrast it with two numbers it is often confused with. **Net income** (profit) is an accrual figure — it includes revenue not yet collected and excludes capital spending entirely, so it is a poor proxy for distributable cash. **Operating cash flow (CFO)** is closer — it is real cash — but it ignores the capital expenditure the business needs to sustain itself, so it overstates what is free. Free cash flow sits one step past CFO: it is operating cash flow *minus the reinvestment*.

### The two reinvestment drains: capex and working capital

Two outflows stand between operating cash and free cash flow, and you must subtract both.

The first is **capital expenditure (capex)** — cash spent on long-lived physical and intangible assets: factories, machines, vehicles, capitalized software. A business that stops spending on capex entirely will, sooner or later, watch its productive capacity decay. Some minimum level of capex is therefore not optional; it is the cost of standing still. (We will see shortly that the part *above* that minimum — growth capex — is genuinely discretionary, and that distinction matters enormously.)

The second is the **increase in net working capital (NWC)**. Working capital is the cash tied up in the day-to-day machinery of the business: inventory sitting in the warehouse and receivables owed by customers, minus the payables the company owes its own suppliers. When a company grows, it must carry more inventory and extend more credit to customers, which *consumes* cash before the corresponding profit is collected. An *increase* in net working capital is therefore a cash *outflow* — money submerged in the operating cycle. (A decrease releases cash. The full mechanics live in [working capital and the cash conversion cycle](/blog/trading/equity-research/working-capital-and-the-cash-conversion-cycle).)

So the skeleton of free cash flow, in words, is: *take the cash the operations produce, subtract the capex needed, subtract the extra working capital growth ties up.* Everything that follows is just a question of **which providers of capital** we are computing that free cash flow *for*. That single question splits free cash flow into its two species.

It is worth pausing on why the working-capital drain is so easy to underestimate. The amount that matters for free cash flow is not the *level* of net working capital but its *change* — the year-over-year increase. A company growing revenue at 20% will typically grow its receivables and inventory at roughly the same pace, and that growth is a recurring cash outflow that scales with the business. A company growing at 20% with thin margins can find that nearly all of its operating cash is being swallowed by the working capital that growth demands, leaving little or no free cash flow despite a healthy income statement. This is one of the most underappreciated reasons a profitable, fast-growing company can run negative free cash flow: not capex, but the cash quietly submerging into the operating cycle to fund the next quarter's higher sales. The faster the growth and the longer the cash conversion cycle, the larger this drain — which is why working capital intensity is a first-order input to any free cash flow forecast, not an afterthought.

### NOPAT, EBIT, and the financing-blind view of profit

To build the firm-wide version cleanly, we need one more term: **NOPAT — net operating profit after tax**. NOPAT is the profit a company would earn *if it had no debt at all*. You take **EBIT** (earnings before interest and tax — operating profit) and tax it at the company's tax rate: NOPAT = EBIT × (1 − tax rate).

Why strip out interest? Because FCFF is supposed to be the cash available to *everyone who funded the firm* — and interest is a payment to one group of those funders, the lenders. If we started from net income (which is *after* interest), we would already have paid the lenders before measuring the firm-wide pool, which is exactly backwards. NOPAT deliberately ignores how the company is financed. It answers: how much after-tax operating profit does the business itself generate, before we decide who gets it? That financing-blind quality is what makes FCFF **capital-structure-neutral**, and it is the reason most professionals reach for FCFF first.

With *free*, *capex*, *working capital*, and *NOPAT* defined, we can build the firm-wide free cash flow.

## FCFF: the cash available to everyone who funded the firm

Free Cash Flow to the Firm is the cash a business generates that is available to **all** of its capital providers — both the lenders who hold its debt and the shareholders who hold its equity — *before* any of that cash is split among them. It is the size of the whole pie. Because it is measured before interest, it does not care whether the company is financed with 10% debt or 60% debt; the operating business produces the same FCFF either way. That neutrality is its defining feature.

The cleanest way to build FCFF is from NOPAT. Start with after-tax operating profit, add back the non-cash charge that depressed it, then subtract the two reinvestment drains:

$$\text{FCFF} = \text{NOPAT} + \text{D\&A} - \text{Capex} - \Delta\text{NWC}$$

Read it left to right. **NOPAT** is the after-tax operating profit. We **add back depreciation and amortization (D&A)** because it reduced NOPAT but consumed no cash this period — it is a non-cash accounting charge spreading out the cost of assets bought in earlier years. We **subtract capex** because that is real cash leaving the building to buy assets. And we **subtract the increase in net working capital** because that is real cash trapped in the operating cycle. What is left is the genuine, financing-blind, free cash the firm threw off.

The figure below walks the build vertically, with the running logic beside each step.

![A vertical waterfall building free cash flow to the firm starting from net operating profit after tax, adding back depreciation, then subtracting capital expenditure and the increase in net working capital to reach FCFF discounted at the weighted average cost of capital](/imgs/blogs/free-cash-flow-fcff-vs-fcfe-2.png)

#### Worked example: building Northwind's FCFF from NOPAT

Northwind Industries reports operating profit (EBIT) of \$92 million and pays tax at 22%. Here is the build:

- **EBIT:** \$92M. This is profit from operations, before interest and tax.
- **NOPAT = EBIT × (1 − tax) = \$92M × 0.78 = \$72M.** This is the after-tax operating profit *as if the company carried no debt*. Crucially, we tax the full EBIT — we do *not* give the company credit for the tax it actually saves on its interest, because that "interest tax shield" belongs in the discount rate (WACC), not in the cash flow. Double-counting it here is a classic error.
- **Add back D&A: +\$30M.** Northwind's machines and capitalized software depreciate by \$30M this year, a charge that reduced EBIT but moved no cash. Add it straight back.
- **Subtract capex: −\$40M.** Northwind spent \$40M on new and replacement equipment. This is real cash out.
- **Subtract the increase in net working capital: −\$8M.** Northwind grew, so its inventory and receivables rose by \$8M more than its payables — \$8M of cash submerged in the operating cycle.
- **FCFF = \$72M + \$30M − \$40M − \$8M = \$54M.**

So Northwind's operating business generated **\$54 million of free cash flow available to all of its capital providers** this year. Notice that this number never mentions interest, dividends, or debt repayment — those are decisions about *who gets the \$54M*, not about how much the business produced.

*FCFF measures the cash the operating business throws off before anyone argues over how to split it — which is exactly why it does not move when you change the debt mix.*

### The interest tax shield: why we tax full EBIT, not EBIT minus interest

The single subtlest point in the entire FCFF build is the one we flagged in passing: when computing NOPAT, you tax the *full* EBIT, ignoring the fact that the company's real tax bill is lower because interest is deductible. This feels wrong the first time you see it — surely the company *does* save tax on its interest, and that saving is real cash? It is. The resolution is that **the value of that tax saving is captured in the discount rate, not in the cash flow.**

Here is the logic. Interest is tax-deductible, so a levered company pays less tax than an otherwise-identical all-equity company. That reduction — the **interest tax shield** — is a genuine benefit of using debt. The question is only *where in the model* to count it. WACC handles it by using the *after-tax* cost of debt: it multiplies the pre-tax cost of debt by (1 − tax rate), which lowers WACC by exactly the value of the tax shield. So the tax benefit of debt lives inside the discount rate. If you *also* gave the company credit for the interest tax shield inside FCFF — by taxing EBIT minus interest rather than full EBIT — you would count the same benefit twice, once in the cash flow and once in the rate, and overstate the value. The discipline is therefore: **compute NOPAT on unlevered (full-EBIT) taxes, and let WACC carry the tax shield.** It is the cleanest division of labor in valuation, and getting it backwards is one of the most common errors in junior models.

#### Worked example: the tax shield Northwind would double-count

Northwind's EBIT is \$92M, its interest is \$24M, and its tax rate is 22%. The *correct* NOPAT taxes the full EBIT: \$92M × 0.78 = \$72M. A careless analyst instead taxes EBIT *after* interest, reasoning that the company "really" earns \$92M − \$24M = \$68M of pre-tax profit: \$68M × 0.78 = \$53M, then adds interest back differently and tangles the model. The mistake hands Northwind the interest tax shield of \$24M × 22% = \$5.28M *inside the cash flow* — and if that same shield is then captured again by the after-tax cost of debt inside WACC, the \$5.28M of annual benefit gets valued twice. At a perpetual 9% rate, double-counting \$5.28M a year inflates the valuation by roughly \$5.28M ÷ 0.09 ≈ \$59M of phantom value — a 9% overstatement of Northwind's \$675M enterprise value, conjured from a single misplaced subtraction.

*Tax the full operating profit when you build FCFF, and trust WACC to give the company credit for its debt; doing both is the most common way to accidentally inflate a valuation.*

### The second route: FCFF from cash from operations

There is a second, equally valid path to the same \$54M, and you should know it because it starts from a number that appears directly on the cash flow statement: **cash from operations (CFO)**. The CFO route is:

$$\text{FCFF} = \text{CFO} + \text{after-tax interest} - \text{Capex}$$

Why add back after-tax interest? Because CFO, as reported, is already *after* interest has been paid — interest expense flowed through net income, which is the starting point of the indirect-method CFO. But FCFF is supposed to be *before* the lenders are paid. So to get from the after-interest CFO back to the before-interest firm-wide number, you add the interest back — on an after-tax basis, because the interest also generated a tax saving that is baked into CFO. (Working capital changes do not appear explicitly in this version because they are already *inside* CFO — the indirect method already subtracted the increase in net working capital when it built CFO.)

#### Worked example: reconciling Northwind's FCFF the CFO way

Let us confirm the \$54M from the CFO direction. Northwind's cash from operations is \$82 million. It paid \$24 million of interest, and at a 22% tax rate the after-tax interest is \$24M × (1 − 0.22) = \$18.72M ≈ **\$18.7M**. Capex is \$40M.

- **CFO:** \$82M.
- **Add after-tax interest: +\$18.7M.** Strips out the financing cost so we are back to the firm-wide, pre-lender number.
- **Subtract capex: −\$40M.**
- **FCFF = \$82M + \$18.7M − \$40M = \$60.7M.**

That does not match the \$54M from the NOPAT route — and the reason is instructive. Reconciling the two routes is a standard check that your two-statement model ties out. CFO of \$82M already contains the add-back of D&A (+\$30M) and the working-capital drain (−\$8M), but it also reflects items the NOPAT build did not, such as the tax actually paid differing from the simple 22% × EBIT, and other non-cash adjustments. When a real model is internally consistent, the two routes agree to the penny. For a clean teaching model, force them to agree by holding the moving parts fixed: with Northwind's CFO reconstructed as NOPAT \$72M + D&A \$30M − ΔNWC \$8M − after-tax interest \$18.7M − (cash taxes − book taxes) = \$75.3M, the CFO route then gives \$75.3M + \$18.7M − \$40M = \$54M. The lesson is not the algebra; it is that *the two routes are the same identity rearranged*, and if they disagree in your model, you have a linkage error to hunt down, not a choice to make.

*If your NOPAT-based FCFF and your CFO-based FCFF do not match, your model has a broken link between the income statement and the cash flow statement — find it before you trust the valuation.*

## FCFE: what the shareholders actually get

Free Cash Flow to Equity is the cash left for **shareholders alone**, after the company has met every obligation to its lenders. If FCFF is the whole pie, FCFE is the slice that remains once the lenders have eaten. You get there by taking FCFF and routing the debt-related flows through it:

$$\text{FCFE} = \text{FCFF} - \text{after-tax interest} - \text{net debt repayment}$$

or, written with net *borrowing* (new debt drawn minus debt repaid) as a positive:

$$\text{FCFE} = \text{FCFF} - \text{after-tax interest} + \text{net borrowing}$$

Two adjustments separate FCFE from FCFF, and both concern the lenders. First, **subtract after-tax interest** — this is the cash the company actually paid its lenders, and it belongs to them, not to shareholders, so it leaves the equity pool. We use the after-tax figure because interest is tax-deductible; the tax saving it generates *does* stay with the company. Second, **adjust for net borrowing** — if the company drew down \$10M more in new debt than it repaid, that \$10M of fresh cash is, for now, available to shareholders, so you add it back; if it repaid more than it borrowed, that net repayment is cash leaving the equity pool, so you subtract it.

The figure below carves FCFE out of FCFF.

![A vertical waterfall building free cash flow to equity from free cash flow to the firm by subtracting after-tax interest paid to lenders and adding back net new borrowing, arriving at the cash available to shareholders discounted at the cost of equity](/imgs/blogs/free-cash-flow-fcff-vs-fcfe-3.png)

#### Worked example: building Northwind's FCFE from FCFF

We have Northwind's FCFF of \$54M. Now carve out the equity slice:

- **FCFF:** \$54M — the firm-wide pool.
- **Subtract after-tax interest: −\$18.7M.** Northwind paid \$24M of interest; after the 22% tax shield, the real after-tax cost is \$18.7M. This cash went to lenders, so it leaves the equity pool.
- **Add net borrowing: +\$10M.** During the year Northwind drew \$10M more in new loans than it repaid, so \$10M of fresh cash landed in the pot available to shareholders.
- **FCFE = \$54M − \$18.7M + \$10M = \$45.3M.**

So of the \$54M the firm produced, **\$45.3M is genuinely available to Northwind's shareholders** this year, after the lenders have been served and after accounting for the net new debt the company took on.

*FCFE is FCFF with the lenders' claim removed: pay them their after-tax interest, settle the change in their principal, and what is left belongs to the owners.*

### The direct route: FCFE from cash from operations

Just as FCFF has a CFO route, so does FCFE — and it is the most intuitive of all, because every term is something you can read straight off the cash flow statement:

$$\text{FCFE} = \text{CFO} - \text{Capex} + \text{net borrowing}$$

Read it plainly: cash from operations (already after interest, because the company really did pay it) minus the capex the business needs, plus any net new debt the company raised. There is no interest add-back here, because — unlike FCFF — FCFE is *supposed* to be after interest; the lenders' interest has already correctly left the pool inside CFO. This is the version many practitioners use day to day because it requires the fewest manual adjustments. With Northwind's CFO of \$82M held consistent (the reconstructed \$75.3M figure used in the FCFF reconciliation), CFO − capex + net borrowing = \$75.3M − \$40M + \$10M = \$45.3M. The same \$45.3M, by a shorter road.

## The matching rule: cash flow and discount rate must agree

Here is the single most important idea in this entire piece, and the one that separates a coherent valuation from an expensive mistake. **Each free cash flow must be discounted at its own matching rate, and each gives you a different thing.**

- **FCFF is discounted at the weighted average cost of capital (WACC), and it gives you ENTERPRISE VALUE** — the value of the whole business, debt and equity together. This makes sense: FCFF is the cash available to *all* capital providers, so you discount it at the *blended* rate that reflects what *all* of them demand. WACC weights the cost of equity and the after-tax cost of debt by how much of each the company uses. The result is the value of the entire enterprise.
- **FCFE is discounted at the cost of equity (Kₑ), and it gives you EQUITY VALUE directly** — the value of the shareholders' stake, with no further bridge needed. This also makes sense: FCFE is the cash available to *shareholders only*, so you discount it at the rate *shareholders* demand. The lenders never enter the calculation, because they were already paid inside the cash flow. The result is the equity value, full stop.

Get this pairing backwards and the numbers are not slightly off — they are meaningless. Discount FCFF (cash for everyone) at the cost of equity (the rate for shareholders only), and you have valued the whole firm as if shareholders demanded their return on the lenders' money too — a number that is too low, often badly. Discount FCFE (cash for shareholders) at WACC (a blended rate that is lower than the cost of equity), and you have given the shareholders' cash a discount rate diluted by cheap debt that does not apply to them — a number that is too high. The figure makes the pairing explicit.

![A side by side comparison of two valuation paths, the firm path discounting free cash flow to the firm at WACC to reach enterprise value then subtracting net debt to reach equity value, and the equity path discounting free cash flow to equity at the cost of equity to reach equity value directly](/imgs/blogs/free-cash-flow-fcff-vs-fcfe-4.png)

Notice the asymmetry in the figure. The firm path gives you enterprise value, so you must then **bridge down to equity** by subtracting net debt (and adjusting for other claims). The equity path skips the bridge entirely — FCFE already accounts for debt, so discounting it lands you on equity value directly. Neither is more correct; they are two routes to the same destination, which is the subject of the reconciliation later in this piece. The full mechanics of building WACC live in the forward companion on [cost of capital, WACC and CAPM](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm); the bridge from enterprise to equity to per share lives in [enterprise value to per share](/blog/trading/equity-research/enterprise-value-to-per-share-the-bridge). Here, the only thing you must internalize is the pairing itself.

#### Worked example: the same cash flows, the wrong rate, the wrong answer

Let us make the cost of the mismatch concrete. Suppose Northwind's FCFF is \$54M growing at 1% forever, its WACC is 9%, and its cost of equity is 12.3%. Net debt is \$275M.

**Done correctly (FCFF at WACC):** Enterprise value = \$54M ÷ (0.09 − 0.01) = \$54M ÷ 0.08 = **\$675M**. Bridge to equity: \$675M − \$275M net debt = **\$400M equity value**. That is the right answer.

**Done wrong (FCFF at cost of equity):** A careless analyst discounts the firm-wide \$54M at the 12.3% cost of equity: \$54M ÷ (0.123 − 0.01) = \$54M ÷ 0.113 = **\$478M**. They then subtract net debt: \$478M − \$275M = **\$203M equity value**. They have just undervalued Northwind's equity by nearly half — \$203M versus the correct \$400M — purely by pairing the firm-wide cash flow with the shareholders-only rate. The cash flow was right. The growth was right. The mismatch alone destroyed half the value.

*A discount rate is a price tag for a specific claim; staple it to the wrong cash flow and you are pricing a claim that does not exist.*

## Why FCFF is usually the professional's default

If both approaches reconcile to the same equity value, why do most analysts reach for FCFF first? Three reasons, all flowing from its capital-structure neutrality.

**First, FCFF is stable when leverage changes; FCFE is not.** Imagine Northwind takes on a large new loan next year. Its FCFE will *spike* (net borrowing adds cash) and then *sag* in later years as it pays the interest and repays principal — even though the underlying business has not changed at all. FCFF, computed before any financing flows, sails straight through the leverage change untouched. For a company whose capital structure is shifting — a leveraged buyout, a deleveraging turnaround, a firm issuing or retiring debt — FCFF gives you a clean, comparable cash-flow stream while FCFE lurches around for reasons that have nothing to do with operations.

**Second, FCFF lets you value the business independently of a financing decision you may want to change.** When a private-equity buyer evaluates a target, the *operating* business is what they are buying; the debt they will layer on is a separate decision they control. Valuing FCFF at WACC (or, in an LBO, valuing the unlevered free cash flow and modeling the debt explicitly) separates the value of the operations from the value created by financial engineering. FCFE blends the two together, which is exactly what you do *not* want when the financing is a variable.

**Third, FCFF avoids a circularity trap.** The cost of equity depends on the company's leverage (more debt makes equity riskier, raising Kₑ via a higher levered beta). If you forecast FCFE under a *changing* capital structure, your discount rate should change every year too — a genuine headache. FCFF at WACC sidesteps much of this when leverage is roughly stable, and modern unlevered-FCF-plus-explicit-debt models handle the changing case cleanly.

When *is* FCFE the right tool? When the entity's value is most naturally expressed as equity and leverage is stable — classically, **banks and insurers**, where debt (deposits, policy reserves) is part of the operating model itself and the FCFF/WACC framework breaks down. For financial firms you almost always value equity directly with FCFE or dividends. For an ordinary industrial or software company, FCFF is the safer default.

### Levered vs unlevered free cash flow — the same distinction, different words

You will frequently hear "unlevered free cash flow" and "levered free cash flow," especially from bankers, and it is worth nailing down that these are *the same distinction we have been drawing*, under different names. **Unlevered free cash flow is FCFF** — "unlevered" because it is computed before the effect of leverage (interest), available to all capital. **Levered free cash flow is FCFE** — "levered" because it is *after* the effect of leverage, available only to equity. The terminology is a banking convention; the substance is identical. When someone hands you a "levered FCF" forecast and asks you to discount it, your first move is to make sure they discount it at the cost of equity, not WACC — because levered FCF *is* FCFE, and the matching rule does not care what you call it. The terminology trips people up precisely because it sounds like a property of the cash flow rather than a statement about *who the cash belongs to*; once you map "unlevered → firm → WACC → enterprise value" and "levered → equity → cost of equity → equity value," the labels stop being a source of confusion and become a quick mnemonic for the matching rule itself.

## Maintenance vs growth capex: the judgment that makes FCF an art

Here is where free cash flow stops being a formula and starts requiring judgment. The capex line we have been subtracting is a single reported number, but it bundles together two economically different things.

**Maintenance capex** is the spending required just to keep the existing business running at its current scale — replacing worn machines, refreshing aging trucks, maintaining the software that already exists. This spending is *not optional*: skip it long enough and the business shrinks. Maintenance capex is a genuine, recurring cost of staying in business, and it absolutely must be subtracted to compute the cash the business can *sustainably* distribute.

**Growth capex** is the spending *above* maintenance — building a new factory, entering a new market, adding capacity the business does not yet need. This spending is *discretionary*. A company can stop it tomorrow and continue running at its current size indefinitely. Growth capex is a *bet on future cash flows*, not a cost of producing today's.

![A split of total reported capital expenditure into a mandatory maintenance portion that sustains current operations and a discretionary growth portion that funds expansion, with the implication that owner earnings deducts only the maintenance slice](/imgs/blogs/free-cash-flow-fcff-vs-fcfe-5.png)

Why does the split matter so much? Because a fast-growing company can show *low or even negative* free cash flow purely because it is spending heavily on growth capex — and that low FCF is not a sign of a weak business; it is a sign of a business choosing to invest. If you value that company on its current depressed FCF, you will badly undervalue it. Conversely, a mature company can show *high* free cash flow that is partly an illusion, because it is quietly *underspending* even on maintenance — harvesting the business, letting the assets decay, and booking the deferred spending as "free" cash that the business will have to pay back later in a wave of catch-up capex. The reported FCF looks great right up until the machines start failing.

The problem is that **companies do not disclose the split.** The 10-K shows total capex, not how much was maintenance versus growth. Estimating the split is a genuine analytical skill — you can anchor on depreciation as a rough floor for maintenance (in steady state, replacement spending roughly tracks depreciation), study capex in years when the company was *not* expanding, or read management's own commentary in the [MD&A](/blog/trading/equity-research/reading-the-10k-footnotes-and-mda). There is no formula; there is informed judgment, and two careful analysts can reasonably disagree.

#### Worked example: normalizing Northwind's lumpy capex year

Suppose this year was unusual for Northwind. Its \$40M of capex included a one-time \$18M new production line for a product it just launched — pure growth capex. Its *normal* maintenance capex, judging from depreciation of \$30M and from prior non-expansion years, runs about \$22M.

A naive analyst takes reported FCFF of \$54M (which subtracted the full \$40M of capex) and projects it forward. But this conflates a one-time growth bet with the run-rate of the business. To find the *sustainable* free cash flow — what the business throws off when it is *not* building a new line — normalize the capex to maintenance:

- Reported FCFF (full \$40M capex): \$54M.
- Add back the \$18M of one-time growth capex: \$54M + \$18M = **\$72M of maintenance-level FCFF.**

This \$72M is closer to what Northwind can sustainably distribute *if it stops expanding* — and it is what you would use to estimate "owner earnings" (next section). But beware the symmetric trap: that \$72M is *not* the right number to capitalize into perpetuity either, because a healthy growing company *will* keep spending growth capex and *will* keep growing the cash flows that justify it. The correct treatment in a full DCF is to model the growth capex explicitly alongside the growth in cash flows it produces — not to strip it out and pretend the company will both stop investing and keep growing. Normalizing tells you the *quality* of this year's FCF; it does not license you to assume free growth.

*Reported free cash flow in any single year is a blend of run-rate cash and one-time bets; the analyst's job is to separate them before drawing a trend line.*

## Owner earnings: Buffett's refinement of free cash flow

The idea that you should deduct *maintenance* capex rather than total capex has a famous name: **owner earnings**, the concept Warren Buffett laid out in his 1986 Berkshire Hathaway letter. Buffett defined owner earnings as reported earnings, plus depreciation and amortization and other non-cash charges, *minus* the average annual capitalized expenditure the business requires to fully maintain its long-term competitive position and unit volume — in other words, minus **maintenance** capex, not total capex.

Owner earnings is FCFF's more conservative, more judgment-laden cousin. The crucial difference from textbook FCFF is precisely the capex term: textbook FCFF subtracts *all* capex; owner earnings subtracts only the *maintenance* slice, on the theory that growth capex is a discretionary investment whose returns will show up in *future* earnings and should not be charged against *current* owner earnings. Buffett's point was philosophical as much as technical: a business's true earning power is the cash it could distribute while standing still, and that requires the analyst to make the maintenance-vs-growth judgment that GAAP refuses to make for you. The deeper Buffett worldview — intrinsic value, moats, and the owner's mindset — is the subject of [Warren Buffett, Berkshire, and value investing](/blog/trading/finance/warren-buffett-berkshire-value-investing); here the relevant lesson is narrow and sharp: **the "free" in free cash flow hides a judgment call about capex, and owner earnings makes that judgment explicit.**

#### Worked example: Northwind's owner earnings

Using the normalized maintenance capex of \$22M from the previous example, Northwind's owner earnings looks like this. Start from net income (after interest and tax). Suppose Northwind's net income is \$53M.

- **Net income:** \$53M.
- **Add back D&A: +\$30M.** Non-cash.
- **Subtract maintenance capex: −\$22M.** Only the spending needed to stand still — not the \$18M growth line.
- **Owner earnings = \$53M + \$30M − \$22M = \$61M.**

Buffett would say this \$61M is a fairer picture of what Northwind's owners earned this year than either the \$53M net income (which ignores the gap between D&A and real replacement spending) or the \$54M reported FCFF (which charges the full growth capex against a single year). It is not a number you will find in any filing; it is a number you *construct*, and the construction is where the analysis lives.

*Owner earnings asks a sharper question than free cash flow: not "how much cash was left this year," but "how much could the owners have taken while keeping the business exactly as strong as they found it."*

## Stock-based compensation: the cost that hides in plain sight

Now to one of the most important — and most abused — adjustments in modern free cash flow. **Stock-based compensation (SBC)** is what a company pays employees in shares or options instead of cash. On the cash flow statement, SBC is a *non-cash expense*: it reduced reported net income, but no cash left the company, so the indirect method *adds it back* when computing CFO. And because it is added back to CFO, it flows straight into free cash flow as if it were free.

It is not free. SBC is a real cost — it is just paid in *ownership* rather than *cash*. When Northwind hands an engineer \$2M of restricted stock instead of \$2M of salary, it has saved \$2M of cash, yes — but it has handed over a \$2M slice of the company, diluting every existing shareholder. The cost did not disappear; it moved from the income statement (where it correctly appeared as an expense) to the share count (where it shows up as dilution). Adding SBC back to free cash flow and then ignoring the dilution is *double counting the benefit and never counting the cost* — you get the cash savings *and* you pretend the shares were free.

This matters enormously for technology companies, where SBC can run to 10%, 20%, even 30% of revenue. A software company that reports glowing free cash flow "before" the drag of SBC may be generating far less value per share than the headline suggests, because the share count is quietly ballooning. The clean way to handle it: **treat SBC as the cash cost it economically represents.** Either subtract SBC from free cash flow (treating the stock as if the company had paid cash and immediately bought the same value of shares back), or — equivalently — keep SBC as the expense it is and value the company on a *fully diluted* share count that captures the dilution. What you may *not* do is add SBC back to FCF *and* value the company on today's share count; that is the trick that flatters per-share value.

#### Worked example: SBC reducing Northwind's true free cash flow

Suppose Northwind, expanding into software-enabled machinery, paid \$12M of stock-based compensation this year, all of which was added back inside its CFO and therefore sits inside the reported FCFF of \$54M.

- **Reported FCFF (SBC added back, treated as free):** \$54M.
- **Subtract SBC as a real economic cost: −\$12M.**
- **SBC-adjusted FCFF = \$54M − \$12M = \$42M.**

That is a \$12M, or 22%, haircut to Northwind's "free" cash flow — and it is the more honest number, because the \$12M of stock Northwind handed out diluted its owners by exactly that value. If you value Northwind on the \$54M and then count its shares as if no new ones were issued, you will overstate the per-share value by precisely the amount of the dilution you ignored. Quality-of-earnings analysts watch this line closely; the companion on [quality of earnings](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) treats SBC as a recurring red flag, not a one-off.

*Stock-based compensation is a cost paid in ownership instead of cash; pretending it is free because no cash moved is the most expensive accounting illusion in modern tech valuation.*

## Two paths to the same equity value

We have built FCFF and FCFE, and we have stated the matching rule. Now we close the loop and prove the claim from the very first paragraph: **done with internally consistent assumptions, the FCFF road and the FCFE road arrive at the same equity value.** This is not a happy accident; it is a mathematical identity, and seeing it tie out is the best way to confirm your model is coherent.

![Two valuation roads converging on the same equity value, one discounting free cash flow to the firm at WACC to enterprise value then subtracting net debt, the other discounting free cash flow to equity at the cost of equity directly, both reaching four hundred million dollars](/imgs/blogs/free-cash-flow-fcff-vs-fcfe-6.png)

#### Worked example: reconciling Northwind's two valuation paths

We will value Northwind both ways, assuming a perpetual growth rate of 1% for simplicity (a real model uses an explicit forecast plus a terminal value, but the reconciliation logic is identical).

**The firm path (FCFF at WACC):**

- FCFF = \$54M, growing at g = 1%.
- WACC = 9%. (Built from a cost of equity of 12.3% weighted ~59% and an after-tax cost of debt of ~4.7% weighted ~41% — the standard market-value-weighted blend.)
- Enterprise value = FCFF ÷ (WACC − g) = \$54M ÷ (0.09 − 0.01) = \$54M ÷ 0.08 = **\$675M.**
- Bridge to equity: subtract net debt of \$275M (gross debt \$300M minus \$25M of cash). Equity value = \$675M − \$275M = **\$400M.**

**The equity path (FCFE at cost of equity):**

- FCFE = \$45.3M, growing at g = 1%.
- Cost of equity Kₑ = 12.3%. (From CAPM: risk-free 4.3% + levered beta 1.6 × equity risk premium 5.0% = 12.3%.)
- Equity value = FCFE ÷ (Kₑ − g) = \$45.3M ÷ (0.123 − 0.01) = \$45.3M ÷ 0.113 = **\$401M ≈ \$400M.**

Both roads land on roughly **\$400M of equity value.** The tiny \$1M gap is pure rounding (after-tax interest of \$18.72M, Kₑ of exactly 12.3%); with unrounded inputs the two equal each other exactly. That is the reconciliation: the firm-wide cash discounted at the blended rate, minus debt, equals the equity cash discounted at the equity rate.

The deep point is *why* they must agree. FCFE differs from FCFF by the debt flows (after-tax interest and net borrowing). The cost of equity differs from WACC by exactly the leverage that those debt flows imply. The two differences are not independent — they are linked through the company's capital structure, and when you keep that structure consistent across both the cash flows and the discount rates, the differences cancel and the equity values coincide. **A gap between your two paths is therefore never a real disagreement about value; it is a flashing light that one of your assumptions — the growth rate, the leverage, a discount-rate input — is inconsistent between the two models.** Sophisticated analysts run both precisely to use the gap as an error-check.

*When the firm path and the equity path disagree, the market has not given you two opinions — your spreadsheet has given you one bug.*

## Common misconceptions

**"Free cash flow is just operating cash flow."** No — operating cash flow ignores capital expenditure entirely, and capex is a real, often massive, cash drain. A capital-intensive business can have strong CFO and *negative* free cash flow because it must constantly reinvest just to stand still. Free cash flow is CFO *minus* the reinvestment. Anyone quoting CFO as "free cash flow" is omitting the single largest claim on the cash.

**"FCFF and FCFE are interchangeable — just pick one."** They are not interchangeable; they measure cash available to *different groups* and must be paired with *different discount rates* to give *different outputs* (enterprise value versus equity value). They reconcile to the same *equity* value only when you respect the matching rule. Treating them as interchangeable — discounting FCFF at the cost of equity, say — produces a number that corresponds to no real claim on the business.

**"Stock-based compensation is non-cash, so it is free."** This is the most expensive misconception in modern valuation. SBC is non-*cash*, but it is not non-*cost*: it is paid in ownership, diluting existing shareholders by exactly its value. Adding it back to free cash flow while valuing the company on a non-diluted share count counts the cash benefit twice and the dilution cost never. Treat SBC as the real cost it is.

**"Lower capex means the business is more efficient and throwing off more free cash."** Sometimes — and sometimes it means the business is *underinvesting*, quietly deferring maintenance to flatter this year's FCF. The reported free cash flow looks better precisely because the company is harvesting its asset base, a benefit it will pay back later in a wave of catch-up capex or in lost competitiveness. You cannot tell which is happening from the FCF number alone; you have to judge whether capex is below the maintenance level the business actually needs.

**"A negative free cash flow company is a bad business."** Not necessarily. A young, fast-growing company can run negative free cash flow for years because it is spending aggressively on *growth* capex and tying up cash in working capital to fund expansion — investments that should produce far larger future cash flows. The question is never just the *sign* of FCF; it is *why* it is negative. Growth capex burning cash on high-return projects is value-creating; maintenance capex the company cannot cover is value-destroying. The composition matters more than the total — the same lesson the [cash flow statement](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from) teaches about the three sections.

**"The DCF discount rate is where all the modeling skill lives."** The discount rate gets the attention, but the cash-flow *definition* is where most real errors hide. A mismatched cash flow and rate can swing a valuation by half (as the worked example showed) — a far bigger error than a reasonable disagreement about WACC of half a percentage point. Define the cash flow correctly first; only then does the discount rate get to do its job.

## How it shows up in real markets

**Amazon and the negative-FCF growth story.** For much of its history, Amazon reported thin or negative free cash flow while growing revenue at extraordinary rates — and bulls and bears fought over whether this was a furnace or a flywheel. The bull case was a maintenance-vs-growth argument in disguise: Amazon's enormous capex (fulfillment centers, then data centers for AWS) was overwhelmingly *growth* capex funding future cash flows, not maintenance spending the business could not afford. Investors who looked only at the headline FCF saw a company that "didn't make money"; investors who decomposed the capex saw a business deliberately suppressing current free cash flow to build a far larger future one. The decade that followed largely vindicated the second reading — but the analytical move that mattered was separating the growth bet from the run-rate, exactly the normalization we did with Northwind.

**The stock-based compensation debate in software.** Across high-growth software, the gap between "free cash flow" and "free cash flow after honestly charging SBC" has become one of the most contested numbers in equity research. Many software companies generate strong reported FCF in large part because SBC — frequently 15–25% of revenue — is added back as non-cash. Skeptical investors recompute free cash flow with SBC treated as a cash-equivalent cost (or value the companies on fully diluted shares including all the stock being issued), and the picture often dims considerably. This is not an abstract dispute: two analysts looking at the same filing can produce per-share values that differ by a wide margin depending solely on how they handle this one line. Whenever you see a company emphasizing "free cash flow" while burying its share-count growth, the matching adjustment is the first thing to make.

**Underinvesting to flatter free cash flow.** A recurring pattern in mature, slowing businesses — and a classic short-seller flag — is the company that boosts free cash flow by cutting capex below the level needed to sustain the asset base. For a period, FCF rises and the stock is rewarded; then the deferred maintenance comes due, capex spikes, and the "free" cash flow that was celebrated turns out to have been borrowed from the future. The tell is capex running persistently *below* depreciation in a business whose assets are genuinely consumed in operations — a sign that replacement spending is being deferred, not that the business has become magically capital-light.

**Where fake cash flow met its limit.** The ultimate version of free-cash-flow manipulation is not flattering it but *fabricating* it — and that is where the cash flow statement's discipline reasserts itself. In frauds like [Wirecard](/blog/trading/finance/wirecard-the-german-fintech-fraud), reported cash and cash flow were simply invented, supposedly sitting in escrow accounts that did not exist. The lesson cuts both ways: free cash flow is harder to fake than earnings because cash must ultimately reconcile to a real bank balance — but when the cash *itself* is fictional, even free cash flow can be a fiction, which is why forensic analysts confirm that reported cash is real before they trust any cash-flow-based valuation built on top of it. And in [Enron](/blog/trading/finance/enron-2001-accounting-fraud), the manipulation of how cash flows were *classified* — pushing operating cash into and out of categories to disguise the true picture — is a reminder that even the structure of the cash flow statement can be gamed by a determined fraud.

The figure below catalogs the legitimate-looking ways free cash flow gets flattered, short of outright fraud — the moves a careful reader checks for in any FCF-based pitch.

![A grid of six ways free cash flow gets flattered, including underinvesting in capex, capitalizing costs instead of expensing them, stretching payables, ignoring stock based compensation, factoring receivables, and cherry picking an adjusted free cash flow metric](/imgs/blogs/free-cash-flow-fcff-vs-fcfe-7.png)

The common thread across all six is the same: each one either pushes a real cost *below* the line you are looking at, or pulls future cash *forward* into the current period. None of them creates value; they only relocate where the costs appear. A reader who knows the honest definition of free cash flow — the cash left after the reinvestment the business genuinely needs — can spot each move by asking the one question that defines the whole concept: *is this number really free, or is the company quietly borrowing it from its own future?*

## When this matters, and further reading

Free cash flow is the cash a discounted cash flow model actually discounts, so getting its definition exactly right is the foundation everything else rests on. The two free cash flows are not interchangeable: **FCFF is the firm-wide pool discounted at WACC to give enterprise value; FCFE is the shareholders' slice discounted at the cost of equity to give equity value directly.** Respect that matching rule and the two paths reconcile; break it and your valuation is incoherent. Past the definition, the real judgment lives in the gray areas — separating maintenance capex from growth capex, treating stock-based compensation as the cost it is, and refusing to take a flattered free cash flow number at face value.

From here, the natural next steps are forward in the valuation chain. To turn these cash flows into a forecast, see [building a DCF, part 1: forecasting](/blog/trading/equity-research/building-a-dcf-part-1-forecasting). To build the discount rates the matching rule demands, see [cost of capital, WACC and CAPM](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm). To convert the enterprise value from the FCFF path into a per-share number, see [enterprise value to per share: the bridge](/blog/trading/equity-research/enterprise-value-to-per-share-the-bridge). And to ground the whole exercise in the owner's mindset that gave us owner earnings, see [Warren Buffett, Berkshire, and value investing](/blog/trading/finance/warren-buffett-berkshire-value-investing). The cash-flow definition you have now is the input every one of those steps depends on — get it right, and the rest is arithmetic.
