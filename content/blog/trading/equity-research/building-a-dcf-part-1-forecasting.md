---
title: "Building a DCF, Part 1: Forecasting Revenue, Margins, and Reinvestment"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A discounted cash flow model is only as good as its forecast — and 90% of the work, and 90% of the error, lives in projecting the cash flows, not the discounting. This post builds an explicit five-year forecast for a real-feeling company from the ground up."
tags: ["equity-research", "corporate-finance", "dcf", "valuation", "forecasting", "free-cash-flow", "reinvestment", "roic", "operating-margin", "financial-modeling", "intrinsic-value"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — A discounted cash flow model is a forecast that gets discounted, and the forecast is the hard part. The discounting is a few lines of arithmetic; the cash flows you feed into it are where 90% of the work — and 90% of the error — actually live. This post builds the explicit forecast for a fictional company, Northwind Industries, from the ground up: revenue, margins, reinvestment, and the free cash flow they produce.
>
> - **A DCF has six steps, and only the first two are hard.** Forecast the cash flows for an explicit horizon, cap them with a terminal value, discount everything to today, sum to enterprise value, subtract net debt to get equity value, divide by shares for a per-share number. The arithmetic of discounting is trivial; the *forecast* is the entire game.
> - **Forecast revenue by fading growth toward a sustainable rate.** Real companies don't grow at one rate forever — growth starts high and *fades* toward something a mature business can hold (roughly nominal GDP). A forecast where revenue compounds at 20% to infinity is a fantasy, not a model.
> - **Forecast margins toward a mature ceiling, not to the moon.** Operating margin can expand as scale spreads fixed costs (operating leverage), but it levels off at a defensible peer-supported level. Margins that only ever rise are the second-most-common forecasting sin.
> - **Reinvestment is the consistency link, and it's the one everyone skips.** Growth has to be *funded*: to grow at rate `g` with a return on invested capital of `ROIC`, you must plow back a reinvestment rate of **`g / ROIC`** of your after-tax operating profit. Free cash flow is what's left *after* that reinvestment. A forecast that grows fast with no reinvestment is internally impossible — it implies infinite returns on capital.
> - This is the engine that produces the [free cash flow](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe) a DCF discounts, built on top of the [three-statement model](/blog/trading/equity-research/how-the-three-financial-statements-connect). Get the forecast right and the rest is mechanics; get it wrong and a precise discount rate just gives you a precisely wrong answer.

There is a comforting myth among people learning valuation that a discounted cash flow model is *about* the discount rate. You hear it in the way people argue: "what cost of capital did you use?", "is the discount rate 8% or 9%?", as if the whole exercise turned on getting that one number right. It does not. The discount rate matters, and we will spend a whole post on it. But the discounting is the *easy* part of a DCF — it is a few lines of arithmetic that any spreadsheet does without complaint. The hard part, the part that separates a model you can defend from a model you should be embarrassed by, is the **forecast**: the explicit projection of how much cash this business will actually throw off, year by year, for the next five or ten years.

This is where 90% of the work lives, and — not coincidentally — where 90% of the error lives too. Two analysts can agree on the discount rate to the basis point and still produce valuations that differ by a factor of three, because one of them forecast revenue growing at 15% and margins expanding to 30%, and the other forecast growth fading to 5% and margins holding at 15%. The discount rate is a knob you turn at the end. The forecast is the thing you are actually arguing about.

So this post is about building that forecast properly, from the ground up, for a company concrete enough that the numbers compound and stay with you. We will not touch the discount rate at all — that is [Part 2](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm). Here we build the *numerator*: revenue, margins, the after-tax operating profit they produce, the reinvestment that growth requires, and the free cash flow that survives once growth is funded. By the end you will be able to look at any analyst's forecast and immediately see whether it is a disciplined projection or a fantasy dressed up in a spreadsheet. The figure below is the map of where the forecast sits in the whole machine — keep it in mind as the orientation for everything that follows.

![A vertical pipeline of six stacked stages showing a DCF flowing from an explicit cash-flow forecast to a terminal value, then discounting to today, summing to enterprise value, subtracting net debt to reach equity value, and dividing by shares to get value per share, with side notes that this post covers the forecast and later posts cover the discounting](/imgs/blogs/building-a-dcf-part-1-forecasting-1.png)

The thesis of this post is one sentence: **a DCF is only as good as its forecast, and a credible forecast is one where every assumption is internally consistent with every other.** Growth must be funded by reinvestment. Margins and capital turnover must be jointly plausible. Returns on capital must be achievable. When those consistency links hold, you have a model. When they don't, you have a wish.

## Foundations: the building blocks of a cash-flow forecast

Before we forecast anything, let's pin down the vocabulary. If you have read the earlier posts in this series — especially [how the three statements connect](/blog/trading/equity-research/how-the-three-financial-statements-connect) and [free cash flow to the firm versus to equity](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe) — most of these terms are familiar. But forecasting uses them in a specific, forward-looking way, so let's be precise. I will keep every definition to the minimum you need to follow the build.

**Discounted cash flow (DCF).** A method for valuing a business as the present value of all the cash it will generate for its investors over its entire life. "Present value" means we shrink future dollars to account for the fact that a dollar next year is worth less than a dollar today (because you could have invested today's dollar). The full machine is: project the cash flows, discount them, sum them. This post builds the projection; later posts handle the discounting.

**Explicit forecast horizon.** The number of years you forecast *explicitly*, line by line, before you switch to a single shorthand for "everything after that." Usually 5 to 10 years. The horizon should run until the business reaches **steady state** — a condition where growth, margins, and returns have settled into rates the company can hold indefinitely. You forecast explicitly until things stop changing, then summarize the rest in a terminal value.

**Steady state.** The mature condition a business converges to: growth has faded to a sustainable rate (no faster than the overall economy, long-run), margins have reached a defensible level, and the return on new investment has settled. A young company is far from steady state; a 100-year-old utility is essentially *in* it. The art of choosing a horizon is choosing how many years it takes *this* company to get there.

**Terminal value (TV).** A single number that captures the value of *all* cash flows after the explicit horizon, collapsed into one lump at the horizon year. It exists because you cannot forecast year-by-year forever, and you don't need to once the business is in steady state. Terminal value typically dominates a DCF — it is often 60–80% of the total value — which is why it gets [its own post](/blog/trading/equity-research/terminal-value-the-part-that-dominates). Here we just need to know it sits at the end of the explicit forecast.

**Revenue (sales).** The top line: the total value of goods or services the company sold in a period. Everything in the forecast cascades down from revenue, which is why forecasting *starts* here. Get revenue wrong by 20% and every downstream number is wrong by roughly 20% too.

**Operating margin.** Operating profit (also called EBIT — earnings before interest and taxes) divided by revenue, expressed as a percentage. It measures how many cents of operating profit the company keeps from each dollar of sales, *before* the effects of debt and taxes. If revenue is \$1,000 and EBIT is \$120, the operating margin is 12%. We forecast margin as a percentage and apply it to forecast revenue to get forecast EBIT. (For a full tour of the margin ladder, see [profitability margins](/blog/trading/equity-research/profitability-margins-gross-operating-net).)

**NOPAT (net operating profit after tax).** This is the single most important profit number in a DCF, and it confuses people because it is *not* net income. NOPAT is operating profit (EBIT) taxed *as if the company had no debt*:

$$\text{NOPAT} = \text{EBIT} \times (1 - \text{tax rate})$$

Why "as if no debt"? Because a DCF that values the *whole firm* (the [FCFF](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe) approach) wants the cash flow available to *all* capital providers — both lenders and shareholders — *before* any is paid to lenders as interest. The tax shield from interest is handled separately, inside the discount rate. So we strip out the financing entirely and tax operating profit at the full rate. NOPAT is the after-tax profit the business produces from its operations, debt-blind. It is the seed from which free cash flow grows.

**Reinvestment.** The cash a company plows back into the business to grow it: capital expenditures (capex) for new plant and equipment, *plus* the increase in working capital (the receivables and inventory that a bigger business ties up), *minus* depreciation and amortization (which is a non-cash charge that funds replacement, not growth). Net reinvestment is:

$$\text{Reinvestment} = (\text{Capex} - \text{D\&A}) + \Delta\text{Working capital}$$

Reinvestment is the price of growth. You cannot sell more widgets next year without building (or buying) the capacity to make them and financing the larger pile of receivables and inventory a bigger operation carries. **Free cash flow is what's left after reinvestment** — and forgetting to subtract reinvestment is the cardinal forecasting sin.

**Capex and D&A.** Capital expenditures are cash spent buying long-lived assets — factories, machines, servers. Depreciation and amortization is the non-cash accounting charge that spreads an asset's cost over its useful life. In a growing company, capex exceeds D&A (you are adding capacity faster than you wear it out); in a steady-state company, capex roughly equals D&A (you are just replacing what wears out). The *gap* between them, `Capex − D&A`, is the part of capex that funds growth rather than maintenance.

**Working capital.** The short-term operating accounts: accounts receivable (money customers owe you), inventory (goods made but unsold), minus accounts payable (money you owe suppliers). When a business grows, these grow with it — more sales means more receivables, more inventory on the shelves. That *increase* is cash you have to tie up, so a rise in working capital is a cash outflow, a form of reinvestment. We cover the mechanics in depth in [working capital and the cash conversion cycle](/blog/trading/equity-research/working-capital-and-the-cash-conversion-cycle); here, the key fact is that growth *consumes* working capital, and a forecast that grows revenue without growing working capital is quietly cheating.

**ROIC (return on invested capital).** The after-tax operating profit a company earns on each dollar of capital invested in the business:

$$\text{ROIC} = \frac{\text{NOPAT}}{\text{Invested capital}}$$

ROIC is the measure of how *good* a business is at turning capital into profit. A company with a 20% ROIC turns each invested dollar into 20 cents of after-tax operating profit a year. ROIC is the hinge of the whole forecast, because — as we will see — it is what links growth to reinvestment. (The deep treatment is in [returns on capital](/blog/trading/equity-research/returns-on-capital-roic-roe-roa) and the [ROIC–WACC spread post](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value).)

**Reinvestment rate.** The fraction of NOPAT a company plows back into the business:

$$\text{Reinvestment rate} = \frac{\text{Reinvestment}}{\text{NOPAT}}$$

This is the link that makes a forecast consistent or inconsistent. We will derive — and lean on heavily — the identity that the reinvestment rate a given growth rate *requires* is `g / ROIC`. Hold that thought; it is the spine of this post.

**FCFF (free cash flow to the firm).** The cash a business generates that is free to be distributed to *all* its investors (lenders and shareholders), after paying for the reinvestment that growth requires:

$$\text{FCFF} = \text{NOPAT} - \text{Reinvestment} = \text{NOPAT} \times (1 - \text{reinvestment rate})$$

This is the number a firm-level DCF discounts. Everything in this post is in service of building FCFF for each forecast year. The two forms of the equation are worth burning in: FCFF is NOPAT minus the dollars you reinvest, *or equivalently* NOPAT times the fraction you *don't* reinvest. We will use the second form constantly.

### Meet Northwind Industries

Throughout this post we follow one fictional company, **Northwind Industries**, the same industrial-widget manufacturer from earlier posts. Using one firm lets the numbers compound, so by the end you will have built a complete five-year forecast for a single business and watched every assumption interlock. Here is Northwind at the starting line:

| Northwind — base year (Year 0) | Value |
|---|---:|
| Revenue | \$1,000M |
| Operating (EBIT) margin | 12.0% |
| Operating profit (EBIT) | \$120M |
| Tax rate | 25% |
| NOPAT | \$90M |
| ROIC | 16% |
| Invested capital | ~\$563M |

Northwind does \$1,000M of revenue at a 12% operating margin, so it earns \$120M of EBIT. Taxed at 25%, that is \$90M of NOPAT. It earns a 16% ROIC, which means its invested capital is about \$90M / 0.16 = \$563M. It is a good, profitable business — its ROIC of 16% comfortably exceeds the ~9% cost of capital we'll use later, which (as the [ROIC–WACC spread post](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value) explains) is the entire reason growth is worth anything. Now let's forecast its next five years.

## The shape of a DCF: six steps, and only two are hard

Let's frame the whole exercise before we dive into the forecast, so you always know where you are. A DCF runs in six steps, shown in the figure at the top of this post:

1. **Explicit forecast.** Project free cash flow (FCFF) for each year of the horizon — say years 1 through 5. This is revenue, margins, NOPAT, and reinvestment, year by year. *This post.*
2. **Terminal value.** Capture all the cash flows after year 5 in a single lump sum sitting at the horizon. *A [later post](/blog/trading/equity-research/terminal-value-the-part-that-dominates).*
3. **Discount to today.** Divide each year's cash flow (and the terminal value) by `(1 + WACC)` raised to the number of years away it is, converting future dollars into present value. *[Part 2](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm).*
4. **Enterprise value.** Sum all the discounted cash flows. This is the value of the entire business — its operating engine, independent of how it's financed.
5. **Equity value.** Subtract net debt (debt minus cash) from enterprise value to get the value belonging to shareholders.
6. **Value per share.** Divide equity value by the share count. This is the number you compare against the market price.

Notice the structure. Steps 3 through 6 are *pure arithmetic* — discount, sum, subtract, divide. There is no judgment in them; a spreadsheet does them mechanically and two analysts who agree on the inputs will agree on the outputs to the penny. Steps 1 and 2 — the forecast and the terminal value — are where *all* the judgment, all the assumptions, and all the disagreement live. This is why the cliché "garbage in, garbage out" is so brutally true for DCFs: the model faithfully amplifies whatever forecast you feed it. A beautiful discounting framework wrapped around a fantasy forecast produces a fantasy valuation, expressed to two decimal places.

So we are going to spend this entire post on step 1, building the explicit forecast for Northwind one layer at a time: revenue, then margins and NOPAT, then reinvestment, then the FCFF that falls out. Let's start at the top.

## Forecasting revenue: fade growth toward a sustainable rate

Revenue is the foundation. Everything downstream — margins, profit, reinvestment, cash flow — is computed *from* revenue, so an error here propagates through the entire model. There are two ways to build a revenue forecast, and a disciplined analyst uses both as cross-checks.

**Top-down.** Start with the total addressable market (TAM) — the total annual spending on this kind of product — and forecast the company's *share* of it. Revenue = TAM × market share. If the global market for industrial widgets is \$50 billion and Northwind has a 2% share, that's \$1 billion of revenue. To forecast, you forecast how the market grows and how Northwind's share moves. Top-down is good for sanity: it stops you from forecasting a company into a revenue larger than its entire market, a surprisingly common embarrassment.

**Bottom-up.** Build revenue from the units up: units sold × price per unit. If Northwind sells 100 million widgets at \$10 each, that's \$1 billion. To forecast, you forecast unit growth and price growth separately. Bottom-up is good for *mechanism*: it forces you to say *why* revenue grows — more units, higher prices, or both — rather than just asserting a growth rate.

But however you build the *level*, the single most important discipline in forecasting revenue is the shape of the *growth*: **growth fades.** No company grows at a constant high rate forever. The reasons are structural, not incidental:

- **The law of large numbers.** A \$100M company can double to \$200M by winning a few big customers. A \$100B company cannot double to \$200B without effectively becoming a meaningful fraction of the world economy. The bigger you are, the harder each additional percentage point of growth becomes.
- **Competition.** High growth and high returns attract competitors, who erode both. A market growing 30% a year draws in capital and rivals until the growth normalizes.
- **Market saturation.** Eventually most of the people who want the product have it. Growth then slows to roughly the rate at which the customer base and prices grow — which, in the long run, is close to the growth of the overall economy (nominal GDP, maybe 4–5%).

So a credible revenue forecast starts with the company's current growth and *fades* it, year by year, toward a sustainable terminal rate — a rate the business could hold forever, which by the logic above cannot durably exceed the growth of the economy. The figure below shows exactly this shape for Northwind.

![A line chart of Northwind revenue rising from one thousand million dollars over five forecast years, with growth fading from twelve percent down through ten, eight, and six to four percent, so the line keeps climbing but its slope visibly shrinks each year toward a sustainable rate](/imgs/blogs/building-a-dcf-part-1-forecasting-2.png)

The line still rises every year — Northwind keeps growing — but the *slope* shrinks. That bend, that flattening, is the signature of a disciplined forecast. A straight line that keeps its slope to the horizon is the signature of a fantasy.

#### Worked example: forecasting five years of Northwind revenue with fading growth

Northwind starts at \$1,000M of revenue (Year 0). We judge that it can sustain double-digit growth for a year or two but will fade toward a mature ~4% as the widget market saturates and competitors crowd in. So we set a *fading* growth path: 12%, 10%, 8%, 6%, 4%. Apply each rate to the prior year's revenue:

| Year | Growth | Revenue ($M) |
|---|---:|---:|
| 0 (base) | — | 1,000.0 |
| 1 | 12% | 1,120.0 |
| 2 | 10% | 1,232.0 |
| 3 | 8% | 1,330.6 |
| 4 | 6% | 1,410.4 |
| 5 | 4% | 1,466.8 |

Year 1 is \$1,000M × 1.12 = \$1,120M. Year 2 is \$1,120M × 1.10 = \$1,232M. And so on — each year compounds on the last, but at a *lower* rate. Over five years, revenue grows from \$1,000M to about \$1,467M, a 47% cumulative increase, or roughly 8% a year on average. By Year 5, growth has faded to 4% — a rate Northwind could plausibly hold forever, which is exactly what we want at the edge of the explicit forecast, because it sets us up cleanly for a terminal value.

Contrast this with the lazy alternative: "Northwind grows 12% a year for five years." That would put Year 5 revenue at \$1,000M × 1.12⁵ = \$1,762M — \$295M, or 20%, higher than our faded forecast. And it would leave Northwind *still* growing at 12% at the horizon, with no clean way to compute a terminal value (because 12% is far above any sustainable long-run rate). The fade is not a cosmetic choice; it is what makes the forecast both believable and terminable.

*Faded growth is the difference between a forecast that respects how businesses age and one that pretends they never do.*

### Choosing the forecast horizon

How many years should you forecast explicitly? The answer is conceptual, not arbitrary: **forecast until the business reaches steady state** — until growth has faded to its sustainable rate and margins and returns have stabilized. For most companies that's 5 to 10 years.

A mature, slow-growing business (a utility, a consumer staple) might need only 3–5 years, because it's already close to steady state. A fast-growing company that's nowhere near maturity might need 10 years to fade growth down to a sustainable rate. The mistake is using a *too-short* horizon for a high-growth company: if you forecast a company growing 25% a year for 5 years and then slam it to a 4% terminal rate, you've created a discontinuity that the terminal value can't handle gracefully — the firm is still "in the middle" of fading when you cut it off. For Northwind, five years works because growth fades cleanly to 4% by Year 5, which *is* steady state.

## Forecasting margins: toward a mature ceiling, not to the moon

With a revenue path in hand, the next layer is the **operating margin** — how much of each revenue dollar becomes operating profit. Margin forecasting has its own discipline, and its own characteristic failure mode.

The forces that move margins over a forecast:

- **Operating leverage.** Many costs are fixed (the factory, the head office, the R&D team). As revenue grows, these fixed costs spread over more sales, so margin *expands* — each extra dollar of revenue carries mostly variable cost, so a larger share of it drops to operating profit. This is the legitimate engine of margin expansion in a growing company.
- **Scale economies.** Bigger companies buy inputs cheaper, run plants closer to capacity, and amortize technology over more units. Real, but bounded.
- **Competition and mix.** Working the other way: competitors force prices down, and growth sometimes comes from lower-margin products or customers, dragging the blended margin.

The net effect for a healthy growing company is usually a margin that *rises* toward a mature level and then *flattens*. The flattening is crucial. Margins do not expand forever, because competition, wage pressure, and the physical limits of efficiency cap how profitable any business can durably be. A forecast where the operating margin marches from 12% to 18% to 25% to 35% with no ceiling is the second-most-common forecasting sin (right after ignoring reinvestment) — it implicitly assumes the company becomes more profitable than essentially any real business in its industry, forever. The figure below shows the disciplined shape: margin rising toward a mature ceiling and leveling off.

![A line chart of Northwind operating margin rising from twelve percent through twelve-point-eight, thirteen-point-five, and fourteen-point-three to fifteen percent across five years, with a dashed green band marking a fifteen percent mature ceiling that the margin approaches and then levels off against](/imgs/blogs/building-a-dcf-part-1-forecasting-3.png)

The right way to forecast margins is to anchor on a **mature target margin** — the level you believe this business can durably sustain, informed by its best historical margin, the margins of mature peers, and the structural economics of the industry — and then forecast a *path* from today's margin to that target. The path should be a smooth climb that flattens as it approaches the ceiling, mirroring the way operating leverage delivers diminishing margin gains as the business matures.

#### Worked example: projecting Northwind's operating margin and computing NOPAT

Northwind earns a 12.0% operating margin today. We judge that operating leverage and scale can lift it toward a mature **15%** — a level its best-run peers achieve and that the industry's economics can support — but no higher, because the widget business is competitive and 15% is about as good as it durably gets. So we forecast a fading-gains path from 12.0% to 15.0%: 12.0%, 12.8%, 13.5%, 14.3%, 15.0%, with the biggest gains early (when operating leverage bites hardest) and the climb flattening as it nears the ceiling.

Now combine the margin path with the revenue path to get EBIT, then tax at 25% to get NOPAT:

| Year | Revenue ($M) | Op margin | EBIT ($M) | NOPAT ($M) |
|---|---:|---:|---:|---:|
| 1 | 1,120.0 | 12.0% | 134.4 | 100.8 |
| 2 | 1,232.0 | 12.8% | 157.7 | 118.3 |
| 3 | 1,330.6 | 13.5% | 179.6 | 134.7 |
| 4 | 1,410.4 | 14.3% | 201.7 | 151.3 |
| 5 | 1,466.8 | 15.0% | 220.0 | 165.0 |

Take Year 3: revenue of \$1,330.6M × 13.5% margin = \$179.6M of EBIT. Taxed at 25%, that's \$179.6M × 0.75 = \$134.7M of NOPAT. NOPAT climbs from \$100.8M in Year 1 to \$165.0M in Year 5 — a 64% increase — driven by *both* revenue growth (47%) *and* margin expansion (12% → 15%). That dual engine, growing the top line while expanding the margin, is what makes the early forecast years so valuable, and exactly why the margin assumption deserves as much scrutiny as the growth assumption.

*NOPAT is revenue times margin, after tax — the operating profit the business produces before a cent goes to lenders or growth, and the seed of every cash flow downstream.*

## Forecasting reinvestment: the consistency link everyone skips

Here is where most beginner forecasts fall apart, and where this post earns its keep. We have NOPAT — the after-tax operating profit. The naïve move is to call that the free cash flow and discount it. **That is wrong, and the error is enormous.** NOPAT is not free cash flow, because a growing company has to *spend* to grow. Free cash flow is what's left *after* you pay for the growth.

Think about what growing from \$1,000M to \$1,467M of revenue actually requires in the real world. To make and sell 47% more widgets, Northwind needs more factory capacity (capex), and it ties up more cash in receivables and inventory (working capital). That spending is **reinvestment**, and it comes straight out of NOPAT before any cash is free for investors. Skip it, and you are valuing a company as if it could grow for free — as if it could conjure 47% more output from the same plant and the same working capital. No business can.

So how much reinvestment does a given amount of growth require? This is the question the entire forecast hinges on, and there is a beautiful, exact answer. Start from the definition of ROIC. A company's NOPAT grows when it adds invested capital. Specifically, next year's *new* NOPAT equals this year's *new* invested capital times the return earned on it:

$$\Delta\text{NOPAT} = \Delta\text{Invested capital} \times \text{ROIC}$$

The growth rate of NOPAT is `g = ΔNOPAT / NOPAT`. And the reinvestment — the new capital you put in — is exactly `ΔInvested capital`. So the *reinvestment rate*, the fraction of NOPAT you plow back, is:

$$\text{Reinvestment rate} = \frac{\Delta\text{Invested capital}}{\text{NOPAT}} = \frac{\Delta\text{NOPAT} / \text{ROIC}}{\text{NOPAT}} = \frac{g}{\text{ROIC}}$$

There it is, the spine of the whole post:

$$\boxed{\text{Reinvestment rate} = \frac{g}{\text{ROIC}}}$$

Read it slowly. **To grow at rate `g`, a company must reinvest a fraction `g / ROIC` of its operating profit.** Faster growth demands more reinvestment (numerator up). A higher-quality business — one with a higher ROIC — needs *less* reinvestment to fund the same growth (denominator up), because each reinvested dollar produces more growth. This single identity is what ties growth, returns, and cash flow together into a consistent system. The figure below lays out the link.

![A tree diagram showing growth g and ROIC feeding into the identity that reinvestment rate equals g divided by ROIC, worked as eight percent over sixteen percent equals fifty percent, which then splits into the half of NOPAT that funds the growth and the half that is free for owners as free cash flow](/imgs/blogs/building-a-dcf-part-1-forecasting-4.png)

And once you have the reinvestment rate, FCFF falls right out:

$$\text{FCFF} = \text{NOPAT} \times (1 - \text{reinvestment rate}) = \text{NOPAT} \times \left(1 - \frac{g}{\text{ROIC}}\right)$$

This is the cleanest, most consistency-proof way to build a forecast: pick a growth path and an ROIC, derive the reinvestment rate as `g / ROIC` each year, and FCFF is whatever's left. It is *impossible* to forecast unfunded growth this way, because the reinvestment is mechanically tied to the growth. That's the point.

#### Worked example: deriving Northwind's reinvestment and FCFF from g and ROIC

Northwind earns a 16% ROIC, and we'll assume it sustains that on new investment (more on that assumption shortly). Apply the identity year by year. In Year 1, growth is 12%, so the reinvestment rate is 12% / 16% = **75%** — Northwind must plow back three-quarters of its NOPAT to fund 12% growth. In Year 5, growth has faded to 4%, so the reinvestment rate is 4% / 16% = **25%** — only a quarter goes back in. As growth fades, reinvestment falls and free cash flow *rises*, even though NOPAT keeps climbing. Here is the full derivation:

| Year | g | NOPAT ($M) | Reinvest rate (g/ROIC) | Reinvestment ($M) | FCFF ($M) |
|---|---:|---:|---:|---:|---:|
| 1 | 12% | 100.8 | 75% | 75.6 | 25.2 |
| 2 | 10% | 118.3 | 62% | 73.9 | 44.4 |
| 3 | 8% | 134.7 | 50% | 67.4 | 67.4 |
| 4 | 6% | 151.3 | 38% | 56.7 | 94.5 |
| 5 | 4% | 165.0 | 25% | 41.3 | 123.8 |

Take Year 1. NOPAT is \$100.8M. The reinvestment rate is 12%/16% = 75%, so reinvestment is \$100.8M × 0.75 = \$75.6M, and FCFF is \$100.8M − \$75.6M = **\$25.2M**. Notice how *little* free cash flow Northwind throws off in Year 1 despite \$100.8M of operating profit — because it's pouring three-quarters of it back into capacity and working capital to fund 12% growth. Now look at Year 5: NOPAT of \$165.0M, reinvestment rate of just 25%, so reinvestment of \$41.3M and FCFF of **\$123.8M**. Free cash flow nearly *quintuples* from Year 1 to Year 5 — not because the business is dramatically more profitable, but because it has stopped having to spend so heavily to grow.

*This is the single most important pattern in DCF forecasting: high-growth years produce little free cash flow because growth eats the cash, and free cash flow blooms only as growth fades and the reinvestment burden lifts.*

### What reinvestment is, mechanically

The `g / ROIC` identity gives us the *total* reinvestment cleanly, which is the right way to build a top-down forecast. But it's worth seeing what that reinvestment *is*, because when you build a model from the actual statements you'll forecast its pieces separately. Reinvestment has two parts:

**Net capital expenditure** = capex − depreciation. Capex is the cash spent on new long-lived assets; depreciation is the non-cash recovery of money spent on *old* assets. The difference is the *growth* portion of capex — the part building new capacity rather than replacing worn-out capacity. A common shortcut for this is the **sales-to-capital ratio**: how many dollars of revenue each dollar of invested capital supports. If Northwind generates \$1.78 of revenue per \$1 of capital (a sales-to-capital ratio of 1.78), then \$467M of new revenue needs \$467M / 1.78 ≈ \$262M of new capital over the five years. That's another route to the same reinvestment, and a good cross-check against the `g / ROIC` route.

**Change in working capital** = the increase in receivables and inventory (net of payables) that a bigger business ties up. If Northwind runs working capital at, say, 10% of revenue, then growing revenue by \$467M over five years ties up an extra \$46.7M of cash in working capital. That is real reinvestment — cash that's spoken for, not free for investors — even though it never shows up as "capex."

When you forecast these pieces individually, they should sum to roughly the `g / ROIC` reinvestment. If they don't, one of your assumptions is off, and that disagreement is *information* — it's telling you the implied ROIC of your line-item forecast differs from the ROIC you assumed. Reconciling the two is exactly the consistency discipline this post is about.

## Internal consistency: the principle that separates models from fantasies

We've now built every layer for Northwind — revenue, margin, NOPAT, reinvestment, FCFF — and the whole point of building it the way we did is that it is **internally consistent.** Let me make that principle explicit, because it is the single idea that distinguishes a credible model from a spreadsheet of wishes.

A forecast is internally consistent when every assumption is compatible with every other assumption. The three big consistency links:

1. **Growth must be funded by reinvestment.** You cannot grow without plowing capital back in. The `g / ROIC` identity *enforces* this — reinvestment is mechanically tied to growth. A forecast that grows revenue while reinvestment stays flat or falls to zero is violating the most basic law of the model: it's claiming free growth, which implies an infinite ROIC.
2. **Margins and capital turnover must be jointly plausible.** A business can earn high returns two ways: fat margins on slow-turning capital (a luxury brand), or thin margins on fast-turning capital (a discount retailer). What it *cannot* do is have *both* fat margins *and* fast turnover *and* low reinvestment — that combination implies a return on capital no real business sustains. When you forecast margin expanding *and* capital turnover improving *and* reinvestment falling, stop: you're probably forecasting an impossible business.
3. **ROIC must be achievable and bounded by competition.** Sustained high ROIC attracts competition that erodes it. A forecast that holds a 40% ROIC for a decade while the company grows fast is assuming the company has a moat no rival can breach — possible for a handful of exceptional businesses, fantasy for most. (This is the [ROIC–WACC spread](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value) story: the spread is what creates value, and competition is always trying to close it.)

The cleanest test of consistency is to **back out the implied numbers** from your forecast and ask if they're believable. Given your revenue, margin, and reinvestment assumptions, what ROIC do they imply? What sales-to-capital ratio? If the implied ROIC is 60% and the company's history and peers say 16%, your forecast is inconsistent — you've assumed away the reinvestment that 16%-ROIC growth requires. The figure below contrasts the two forecasts side by side: the inconsistent fantasy and the disciplined model.

![A before-and-after comparison with the inconsistent fantasy on the left growing twenty percent forever with ever-expanding margins and near-zero reinvestment implying infinite ROIC, and the consistent forecast on the right fading growth from twelve to four percent, capping margin at fifteen percent, reinvesting g over ROIC, and holding ROIC at sixteen percent above the nine percent cost of capital](/imgs/blogs/building-a-dcf-part-1-forecasting-6.png)

#### Worked example: an inconsistent forecast, diagnosed and fixed

Suppose a junior analyst hands you this forecast for Northwind: "Revenue grows 20% a year for five years, operating margin expands from 12% to 25%, and reinvestment is negligible — capex roughly equals depreciation, working capital is flat." It looks fantastic on the page: by Year 5, revenue is \$1,000M × 1.20⁵ = \$2,488M, margin is 25%, so EBIT is \$622M and NOPAT (at 25% tax) is \$466M — and with no reinvestment, *all* of it is "free cash flow." The model spits out an enormous valuation.

Now let's diagnose it with the consistency tools. Growth of 20% with near-zero reinvestment implies a reinvestment rate near 0%. But the identity says reinvestment rate = `g / ROIC`, so a 0% reinvestment rate at 20% growth implies an **infinite ROIC** — the company is growing 20% a year while adding *no* capital. That is impossible: you cannot make and sell 20% more widgets every year with the same factory and the same working capital. The forecast is asserting free growth, the cardinal sin.

Here's the fix. Keep growth at 20% in the early years if you can justify it, but *fund it*. At a realistic 20% ROIC (generous for Northwind, but let's grant it), 20% growth requires a reinvestment rate of 20% / 20% = **100%** — the company must plow back *every* dollar of NOPAT to grow that fast, leaving *zero* free cash flow in the high-growth years. Suddenly the forecast looks completely different: the early years generate no free cash at all (it's all going into capacity), and value comes only later, as growth fades and reinvestment drops below 100%. Then cap the margin at a defensible 15% rather than a fantastical 25%. The "fantastic" forecast and the corrected one can differ in value by more than half — and the entire difference is the reinvestment the original forecast forgot.

*The inconsistent forecast wasn't wrong because its growth was too high; it was wrong because it grew without paying for the growth — and the `g / ROIC` identity is the lie detector that catches it every time.*

### The six forecasting sins to recognize on sight

After you've built and reviewed a few hundred forecasts, you stop seeing infinite variety and start seeing the same handful of mistakes over and over. Almost every broken DCF forecast commits one of six recognizable sins, and each one has a tell you can spot in seconds and a discipline that fixes it. The figure below catalogs them; learn to recognize them and you'll catch most fantasy valuations before you've finished reading the assumptions tab.

![A grid of six forecasting sins with their fixes, arranged in two rows of three: hockey-stick growth fixed by justifying every inflection, margins that only expand fixed by capping at a mature ceiling, growth without reinvestment fixed by setting reinvestment to g over ROIC, an impossible implied ROIC fixed by holding ROIC sane, ignored working capital fixed by scaling it with sales, and a horizon too short fixed by extending to steady state](/imgs/blogs/building-a-dcf-part-1-forecasting-7.png)

Walk through them, because each maps to a discipline we've already built. The **hockey stick** — flat for a year, then a sudden acceleration — is a forecast that asserts an inflection it never justifies; the fix is to demand a *reason* for any acceleration (a new product, a capacity expansion coming online) or fade growth smoothly instead. **Margins that only expand** ignore the ceiling that competition imposes; the fix is the mature-target-margin discipline from the margin section. **Growth without reinvestment** is the cardinal sin we've hammered; the fix is `g / ROIC`. An **impossible implied ROIC** is what you find when you back the return out of an inconsistent forecast and get 80% or infinity; the fix is to hold ROIC to a level history and peers support. **Ignored working capital** understates the cash growth consumes; the fix is to scale working capital with sales rather than holding it flat. And a **horizon that's too short** cuts off a company while it's still growing fast, leaving no clean steady state for the terminal value; the fix is to extend the explicit forecast until growth has genuinely faded.

Notice that five of the six sins are *consistency* failures — the forecast asserts something that some other part of the model contradicts. That's not a coincidence. The reason consistency is the master discipline of forecasting is that the human mind is very good at writing down an attractive number in one cell and forgetting what it implies three cells over. The `g / ROIC` identity, the margin ceiling, the working-capital scaling — these are all just devices for forcing the cells to talk to each other, so that an attractive growth rate can't quietly coexist with an impossible reinvestment line.

## The complete forecast: Northwind's five-year FCFF

Let's assemble everything into the single artifact a DCF needs: the table of free cash flow for each forecast year. This is the output of all the work above, and the input to the discounting we'll do in [Part 2](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm). The figure below is the full build, year by year, left to right.

![A five-row matrix of Northwind's forecast with one row per year, showing revenue rising from eleven hundred twenty to fourteen hundred sixty-seven million, operating margin from twelve to fifteen percent, NOPAT from one hundred to one hundred sixty-five million, reinvestment falling from seventy-six to forty-one million as the reinvestment rate drops from seventy-five to twenty-five percent, and free cash flow to the firm rising from twenty-five to one hundred twenty-four million](/imgs/blogs/building-a-dcf-part-1-forecasting-5.png)

#### Worked example: the full five-year FCFF table for Northwind

Here is the complete forecast, every line, for the five-year horizon:

| Line ($M) | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---:|---:|---:|---:|---:|
| Revenue | 1,120.0 | 1,232.0 | 1,330.6 | 1,410.4 | 1,466.8 |
| Revenue growth | 12% | 10% | 8% | 6% | 4% |
| Operating margin | 12.0% | 12.8% | 13.5% | 14.3% | 15.0% |
| EBIT | 134.4 | 157.7 | 179.6 | 201.7 | 220.0 |
| NOPAT (after 25% tax) | 100.8 | 118.3 | 134.7 | 151.3 | 165.0 |
| Reinvestment rate (g/ROIC) | 75% | 62% | 50% | 38% | 25% |
| Reinvestment | 75.6 | 73.9 | 67.4 | 56.7 | 41.3 |
| **FCFF** | **25.2** | **44.4** | **67.4** | **94.5** | **123.8** |

Read down any column and you can reconstruct it from the assumptions: revenue times margin is EBIT, times (1 − tax) is NOPAT, times (1 − reinvestment rate) is FCFF. Read across the FCFF row and you see the story of the whole forecast: free cash flow starts tiny (\$25.2M) because Northwind is plowing capital into 12% growth, and swells to \$123.8M by Year 5 as growth fades and the reinvestment burden lifts. The sum of the five years of FCFF is about \$355M of undiscounted free cash flow — and that, plus a terminal value capturing everything after Year 5, is what the next post will discount into a present value.

This table is the entire deliverable of forecasting. Everything before it was the reasoning that makes it credible; everything after it (in Parts 2 and beyond) is the arithmetic that turns it into a price.

*A DCF forecast, done right, is just this table — and the table is trustworthy precisely because every number in it descends from a small set of internally consistent assumptions, not from a row of independent guesses.*

## Common misconceptions

**"NOPAT (or net income, or EBITDA) is the free cash flow."** No — and this is the single most expensive mistake in DCF forecasting. None of those is free cash flow, because none of them subtracts the reinvestment that growth requires. NOPAT is operating profit after tax, *before* reinvestment. EBITDA is even further off — it ignores taxes, capex, *and* working capital, which is exactly why it flatters capital-intensive businesses. Free cash flow is NOPAT *minus* reinvestment, full stop. Discounting NOPAT instead of FCFF systematically overvalues every growing company, because it counts cash that's actually being spent to grow as if it were free for investors.

**"Higher growth always means higher value."** Only if the growth earns a return above the cost of capital. Growth funded by reinvestment that earns *exactly* the cost of capital adds *zero* value — you're putting in a dollar and getting back a dollar's worth of present value. Growth that earns *below* the cost of capital actively *destroys* value: the faster the company grows, the more value it burns. This is why the `g / ROIC` link matters so much, and why the [ROIC–WACC spread](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value) — not growth alone — is the real engine of intrinsic value. A high-growth company with a 6% ROIC against a 9% cost of capital is worth *less* the faster it grows.

**"A longer, more detailed forecast is more accurate."** A ten-year, fifty-line model is not more *accurate* than a five-year, five-line one — it's more *precise*, which is different and dangerous. Precision without accuracy is false confidence: a forecast carried to two decimal places for ten years is still just a chain of assumptions, and the later years are pure guesswork. The detail can actively mislead, because it makes a guess look like a measurement. Forecast only as far as you can reason about (until steady state), keep the drivers few and explicit, and put your effort into getting the *assumptions* right, not the spreadsheet bigger.

**"If I get the discount rate exactly right, the valuation will be right."** The discount rate is one input among many, and not the one with the most leverage. A reasonable range for the discount rate (say 8–10%) moves the valuation far less than a reasonable range for the growth-and-margin forecast. Obsessing over whether WACC is 8.7% or 8.9% while waving through a forecast of 20% growth and ever-expanding margins is straining at a gnat while swallowing a camel. The forecast is where the value — and the error — concentrates.

**"Margins can keep expanding as the company scales."** Operating leverage is real, but it's bounded. Margins expand toward a *ceiling* set by competition, input costs, and the physics of the business, then flatten. A forecast where margin rises every single year with no asymptote is assuming the company outruns competition forever — which essentially no business does. The discipline is to name a mature target margin (from peers and history) and fade the gains toward it, not to let the margin line drift upward unchecked.

**"Reinvestment is just capex."** Reinvestment is capex *minus depreciation* (only the growth portion counts) *plus the change in working capital* (the cash a bigger business ties up in receivables and inventory). Forecasts that count gross capex but forget working capital understate the cash growth consumes — sometimes badly, for businesses with long cash conversion cycles. And forecasts that count capex but forget to net out depreciation double-count the maintenance spending. Get the definition right: net reinvestment = (capex − D&A) + ΔWC.

## How it shows up in real markets

**The Amazon "no profits" puzzle.** For years Amazon reported razor-thin or negative net income, and commentators called it overvalued or a bubble. A naïve earnings-based valuation made it look absurd. But Amazon was the textbook case of a company plowing essentially *all* its operating cash flow back into reinvestment — fulfillment centers, AWS data centers, working capital — to fund enormous growth. Its NOPAT was real; its *free* cash flow was small precisely *because* it was reinvesting heavily at a high incremental return. Anyone who understood the `g / ROIC` link could see that the low free cash flow was a *choice* funding high-return growth, not a sign of a broken business. The value was in the reinvestment, and it showed up later, exactly as a disciplined forecast would predict. (These figures are illustrative of the pattern, not precise historical numbers.)

**The hockey-stick startup pitch.** Venture-stage and pre-profit growth companies routinely present forecasts with a "hockey stick": flat or slow for a year, then a sudden steep acceleration, with margins expanding to best-in-class and reinvestment curiously absent. This is the inconsistent forecast in its native habitat. The discipline of `g / ROIC` is the antidote: ask what reinvestment the projected growth requires, and watch the "free cash flow" in the hockey-stick years evaporate into the capacity and working capital the growth actually needs. The most expensive mistakes in growth investing are forecasts that priced in the growth but not the cost of the growth.

**Cyclical companies and the margin trap.** Forecasting margins is most dangerous for cyclical businesses — commodity producers, automakers, semiconductors — whose margins swing wildly with the cycle. The classic error is to forecast off a *peak*-margin year as if it were normal, extrapolating boom margins into a permanent forecast. When the cycle turns, the forecast is exposed as a fantasy. A disciplined analyst forecasts a *mid-cycle* margin, not the current one, and is especially suspicious of any forecast built at the top of a boom. The same logic applies to quality-of-earnings work: a margin propped up by one-off gains, as covered in [quality of earnings](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags), should never be the base for a forecast.

**Accounting fraud and the cash-flow check.** The companies at the center of the great accounting frauds — [Enron](/blog/trading/finance/enron-2001-accounting-fraud), [Wirecard](/blog/trading/finance/wirecard-the-german-fintech-fraud) — reported beautiful, growing earnings while their *cash* told a different (or nonexistent) story. A forecast anchored on free cash flow rather than reported earnings is partly inoculated against this, because it forces you to ask where the cash is and whether the growth is being funded by real reinvestment. When a company's earnings grow but its free cash flow never appears — when reinvestment swallows everything and never produces the cash a high-ROIC business should — the consistency tools in this post are the ones flashing red.

**Value investors and the discipline of conservatism.** The reason great long-term investors like [Warren Buffett](/blog/trading/finance/warren-buffett-berkshire-value-investing) emphasize "businesses I can understand" and durable competitive advantages is, in DCF terms, a demand for *forecastable* cash flows. A business with a stable, predictable revenue and margin path lets you build a forecast you can actually trust; a business whose next five years are a coin flip does not, no matter how sophisticated your model. The forecast is the foundation, and the foundation has to be something you can reason about — which is why the most disciplined investors would rather own a boring, forecastable business at a fair price than an exciting, unforecastable one at any price.

## When this matters and further reading

Forecasting is where intrinsic valuation is won or lost. Once you can build a revenue path that fades, a margin path that approaches a ceiling, and a reinvestment line tied to growth by `g / ROIC`, you have the numerator of every DCF you will ever build — and the judgment to spot when someone else's forecast is a fantasy.

The forecast we built here is the *input* to the discounting machinery. The next posts in the series turn this table into a price:

- **[Building a DCF, Part 2: Cost of Capital (WACC and CAPM)](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm)** — the discount rate that turns these future cash flows into present value, and why it's the *easy* part.
- **[Terminal Value: The Part That Dominates](/blog/trading/equity-research/terminal-value-the-part-that-dominates)** — how to capture every cash flow after Year 5 in a single number, and why it's usually most of the valuation.
- **[The ROIC–WACC Spread: The Engine of Intrinsic Value](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value)** — why growth only creates value when ROIC exceeds the cost of capital, the principle that made our reinvestment assumption matter.

And for the foundations this post stands on:

- **[How the Three Financial Statements Connect](/blog/trading/equity-research/how-the-three-financial-statements-connect)** — the wired machine that produces the revenue, margins, and reinvestment we forecast here.
- **[Free Cash Flow: FCFF vs FCFE](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe)** — the precise definition of the cash flow we built toward, and the firm-versus-equity distinction that decides which one a DCF discounts.

Build the forecast with discipline, keep every assumption consistent with every other, and the rest of the DCF is arithmetic. Skip the discipline, and the most elegant discounting in the world will just give you a precisely wrong number.
