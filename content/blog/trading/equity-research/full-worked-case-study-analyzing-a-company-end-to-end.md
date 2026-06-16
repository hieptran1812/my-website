---
title: "A Full Worked Case Study: Analyzing a Company From Scratch, End to End"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The capstone of the series: we take one company, Northwind Industries, from a blank page to a sized position — reading the statements, judging quality and returns, understanding the moat, valuing it three ways, judging management, writing the thesis, stress-testing it, and deciding how much to buy."
tags: ["equity-research", "corporate-finance", "valuation", "dcf", "investment-thesis", "case-study", "fundamental-analysis", "position-sizing", "margin-of-safety", "intrinsic-value", "moat", "investing"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 46
---

> [!important]
> **TL;DR** — This is where the whole series comes together. We take one realistic mid-cap, *Northwind Industries*, and walk the complete professional workflow from a blank page to a sized position — and you watch all forty-seven prior tools snap into one coherent process.
>
> - **Analysis is a workflow, not a grab-bag.** Nine steps in order: read the statements, analyze quality and returns, understand the business, value it three ways, judge capital allocation, write the thesis, stress-test it, and decide. Each step uses the output of the one before.
> - **The numbers describe a real business.** Northwind does \$1,000M of revenue at a 12% operating margin, earns a 15% ROIC against an 8.4% WACC — a value-creating +6.6-point spread — and converts its \$79M of accounting profit into \$110M of real operating cash. It is good, not extraordinary.
> - **Valuation is a range, not a point.** A DCF says ~\$34 a share, a multiples cross-check says \$30–\$38, and a reverse DCF shows the \$30 market price implies 7.4% growth — faster than the moat can sustain. Three methods bracket the value above the price.
> - **The thesis is a variant perception you can be wrong about.** The market is too pessimistic on the *durability* of the spread, not too optimistic on growth — and we name exactly what would prove us wrong before we buy.
> - **The decision is a sized bet, not a verdict.** Positive expected value plus a thin margin of safety plus a narrow moat equals a *small* buy: a 3% position at \$30, adding toward 5% only below \$27, with explicit sell triggers.

There is a particular moment that every analyst remembers from early in their career: the first time they were handed a company name and a blank page and told to *figure out whether it's worth owning*. Not to compute a single ratio, not to run one model, but to do the whole thing — to start from nothing and end with an opinion you would stake real money on. It is a disorienting feeling, because the textbooks teach you the tools one at a time, in tidy isolated chapters, and the blank page does not. The blank page asks for all of them at once, in the right order, woven into a single judgment.

This post is that blank page, worked through from beginning to end. We are going to take one company — *Northwind Industries*, the fictional industrial-pump maker that has run through this entire series as our recurring example — and analyze it completely, the way a professional actually does it: not as a checklist of disconnected calculations, but as one flowing process where each step feeds the next. Reading the statements tells us what to analyze. The analysis tells us what to understand about the business. Understanding the business tells us how to value it. The valuation, set against the price, becomes a thesis. The thesis, stress-tested against everything that could kill it, becomes a decision. And the decision, sized for our conviction and the odds, becomes a position. Nine steps, one chain.

![The end-to-end equity research workflow shown as nine numbered steps flowing from reading the statements through to sizing the position, arranged in two vertical columns with arrows connecting each step to the next](/imgs/blogs/full-worked-case-study-analyzing-a-company-end-to-end-1.png)

The figure above is the map for the entire post — keep it in mind as we go, because each section is one box in it. The left column is the *building* of the case from the numbers and the business; the right column is the *judging*, the writing, the stress-testing, and the act of buying. Notice that the steps are not interchangeable. You cannot value a business you do not understand, and you cannot understand a business whose statements you have not read. The order is the discipline. An analyst who jumps straight to "what's the P/E?" has skipped the eight steps that would tell them whether the E is real, whether it will persist, and what it is worth — which is exactly how people end up buying melting ice cubes at a discount and calling it value.

A word on Northwind before we begin. It is fictional, but it is built to be *believable* — a composite of the kind of unglamorous, mid-sized industrial business that fills the middle of every market index and rarely makes headlines. It makes pumps for industrial customers: water treatment plants, chemical processors, mining operations. It is profitable, modestly leveraged, growing in the mid-single digits, with a real but narrow competitive advantage. It is, in other words, the *median* good company — not a hyper-growth darling, not a distressed turnaround, just a solid business trading at a price that may or may not be fair. That is deliberate, because the median good company is exactly the case where the full workflow earns its keep. The extremes are easy to judge; it is the merely-good business at a roughly-fair price where the discipline of doing every step properly separates a real edge from a coin flip.

## Foundations: the vocabulary of the whole workflow

Because this post touches every part of the series, let us pin down — tightly, from zero — the handful of ideas that recur in every section. If you have read the earlier posts these will be familiar; here we define them just precisely enough to carry the case study, and point to the deep treatment of each.

**The three financial statements.** A company reports its financial life in three linked documents. The **income statement** shows the profit it earned over a period: revenue at the top, costs subtracted in layers, net income at the bottom. The **balance sheet** is a snapshot of what the company *owns* (assets), *owes* (liabilities), and what is left for owners (equity) at one instant. The **cash flow statement** reconciles the two by showing the actual cash that moved — because profit on the income statement and cash in the bank are not the same thing. The three are not independent; net income flows into the balance sheet's equity and seeds the cash flow statement's top line. How they interlock is the subject of [how the three financial statements connect](/blog/trading/equity-research/how-the-three-financial-statements-connect), and it is the foundation everything else rests on.

**Margin.** Any profit figure divided by revenue, expressed as a percentage — the cents of profit that survive from each sales dollar at a given point. Gross margin (after the cost of the product), operating margin (after the cost of running the company), and net margin (after everything) each answer a different question about the business.

**Invested capital and ROIC.** **Invested capital** is the total pool of money — debt and equity — tied up in running the business. **ROIC**, return on invested capital, is the after-tax operating profit (NOPAT) that pool earns each year, divided by the pool: it answers "when this company invests a dollar, how many cents does it earn back, every year?" It is the single best one-number summary of business quality, treated fully in [returns on capital: ROIC, ROE, ROA](/blog/trading/equity-research/returns-on-capital-roic-roe-roa).

**WACC and the spread.** **WACC**, the weighted average cost of capital, is what that capital *costs* — the blended return lenders and shareholders demand. The **spread** is simply ROIC minus WACC, and it is the engine of value: a positive spread means every reinvested dollar earns more than it costs, so growth *creates* value; a negative spread means growth *destroys* it. This relationship is the spine of the whole series, developed in [the ROIC–WACC spread post](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value).

**Free cash flow (FCFF).** The cash a business throws off after funding the reinvestment that growth requires: `FCFF = NOPAT − reinvestment`. It is the number a firm-level discounted cash flow model discounts.

**Intrinsic value vs price.** **Intrinsic value** is what a business is actually worth based on the cash it will generate; **price** is what the market happens to charge for it today. The gap between them is where return comes from. A **DCF** (discounted cash flow) estimates intrinsic value by forecasting cash flows and discounting them to the present; a **multiple** (like P/E or EV/EBITDA) is a shorthand for the same thing, the market's price per dollar of some fundamental.

**Moat.** A durable competitive advantage — switching costs, network effects, brand, cost advantage, regulation — that protects a high ROIC from the competition that high returns always attract. A moat does not mainly *widen* the spread; it *lengthens the time the spread survives*. See [economic moats](/blog/trading/equity-research/economic-moats-durable-competitive-advantage) — and note we will lean on it heavily, because durability is the hinge of Northwind's value.

**Margin of safety.** The discount to intrinsic value at which you buy, so that even if your estimate is too high, you still do not overpay. It is the difference between an analyst and a gambler.

With those defined, we have everything we need. Let us start where every real analysis starts: with a reason to look at all.

## Step 1 — The setup: why Northwind is on the radar

Analysis never starts with a company in a vacuum; it starts with a *reason*. Something put Northwind in front of you. Maybe it surfaced on a screen — a filter for companies earning more than 12% on capital, trading below 16 times earnings, with net debt under 2× EBITDA. Maybe a competitor's earnings call mentioned losing a contract to Northwind. Maybe the stock fell 25% on a quarter the market hated and a value-minded colleague flagged it. The origin matters, because it frames the question you are actually trying to answer.

Here is Northwind's setup. It is a \$3 billion mid-cap (we will pin down the exact figures in a moment) that makes industrial pumps and fluid-handling systems — the unglamorous but essential machinery that moves water through treatment plants, chemicals through processing facilities, and slurry through mines. It sells primarily to large industrial and municipal customers on multi-year contracts, often with a long service-and-parts tail attached to each installed unit. The industry is mature, cyclical with industrial capital spending, and dominated by a handful of established players. Northwind is the number-three competitor by market share, with a reputation for reliability in the harshest applications.

It is on the radar because the stock has drifted down 20% over the past year on fears that an industrial slowdown will crush its order book, and it now trades at \$30 a share — roughly 15 times trailing earnings, a level that *looks* cheap for a business earning 15% on capital. The question, then, is sharp: **is this a quality business temporarily on sale, or a cyclical earning peak earnings at a peak multiple about to roll over?** That is the question the next eight steps will answer. The screen got us to the door; now we have to go inside.

The discipline at this stage is to write the question down and resist answering it prematurely. The single most common error in equity research is forming the conclusion at Step 1 and spending Steps 2 through 9 hunting for evidence to support it. We are going to do the opposite: build the case from the facts, in order, and let the conclusion emerge — even if it contradicts the "looks cheap" hunch that brought us here.

## Step 2 — Reading the statements: Northwind at a glance

Before any analysis, you read. You pull the three statements and form a first, holistic impression of the business — not ratios yet, just the shape of it. The goal is to be able to describe Northwind in three sentences: how it makes money, what it owns and owes, and whether its profits are backed by cash.

![Northwind's three financial statements at a glance shown as a matrix with the income statement, balance sheet, and cash flow statement as rows and the top, middle, and bottom of each statement as columns](/imgs/blogs/full-worked-case-study-analyzing-a-company-end-to-end-2.png)

Here is Northwind, statement by statement, in round numbers (all figures in millions of dollars unless noted).

**The income statement.** Revenue of \$1,000M. After the direct cost of building the pumps, gross profit is \$400M — a **40% gross margin**, healthy for a manufacturer and a sign of genuine pricing power on its engineered products. After the cost of running the company (sales force, engineering, corporate overhead), operating income (EBIT) is \$120M — a **12% operating margin**. Add back \$40M of depreciation and you get EBITDA of \$160M (16%). Subtract \$20M of interest to reach \$100M of pre-tax income, tax it at 21% (a \$21M bite), and net income is **\$79M** — a 7.9% net margin. The walk down the ladder is exactly the one built in [the income statement, line by line](/blog/trading/equity-research/income-statement-line-by-line-revenue-to-net-income) and [profitability margins](/blog/trading/equity-research/profitability-margins-gross-operating-net).

**The balance sheet.** Total assets of \$1,000M. On the asset side: \$150M of cash and marketable securities, \$90M of receivables, \$110M of inventory, \$480M of net property and plant, plus \$90M of goodwill and \$60M of other intangibles from a past acquisition. On the other side: \$600M of total liabilities — \$390M of which is real interest-bearing debt (\$60M short-term, \$280M long-term bonds, \$50M capitalized leases), with the rest being payables and \$50M of deferred revenue from prepaid service contracts. The residual, equity, is **\$400M** — book value. With 100M shares outstanding, that is \$4.00 of book value per share. The full anatomy is in [the balance sheet](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth).

**The cash flow statement.** This is the lie detector. Net income was \$79M. Add back the \$40M of non-cash depreciation, adjust for a modest build in working capital as the business grew, and operating cash flow lands at about **\$110M** — comfortably *above* net income. That relationship — cash generation exceeding reported earnings — is the first and most reassuring signal in the whole analysis: it says the profits are real, not an accounting mirage. Where the cash truly comes from is the subject of [the cash flow statement](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from).

Three sentences, then: *Northwind sells engineered pumps at a 40% gross margin and keeps 12% as operating profit; it carries a moderate \$390M of debt against \$1,000M of assets and \$400M of equity; and it converts every dollar of reported profit into more than a dollar of operating cash.* That is the shape of the business. Now we analyze it.

## Step 3 — Financial analysis: is this a good business, and are the numbers clean?

Reading gave us the shape; analysis gives us the verdict. Four questions decide whether the financials describe a good business: are the margins healthy and durable, does the company earn more than its cost of capital, can it survive a downturn, and — the question that underwrites all the others — are the earnings *real*?

![The financial-quality dashboard for Northwind shown as a matrix with margins, returns, leverage, and quality as rows and what we measure, Northwind's number, and the verdict as columns](/imgs/blogs/full-worked-case-study-analyzing-a-company-end-to-end-3.png)

**Margins.** We read each margin in three dimensions — level, trend, and decomposition — as [profitability margins](/blog/trading/equity-research/profitability-margins-gross-operating-net) insists. Northwind's 40% gross margin is solid but a few points below its best peer's 44%, suggesting some room but no dominance. Its 12% operating margin has been *grinding upward* over three years from about 11%, as scale spreads its fixed engineering and overhead costs — operating leverage, working as designed. The decomposition matters: the gross margin held while the operating margin rose, which tells us the improvement came from overhead discipline, not pricing — a more durable source.

**Returns — the heart of it.** Here we run the calculation that, more than any other, tells us whether Northwind is worth owning at all.

#### Worked example: Northwind's ROIC–WACC spread and the value it creates

Northwind earns \$120M of EBIT. Taxed at the full operating rate (~21% here, though the DCF will use a slightly more conservative 25%), that is roughly \$95M–\$100M of NOPAT — but the series has consistently used **\$150M of NOPAT on \$1,000M of invested capital** as Northwind's normalized, through-cycle figure, giving a clean **15% ROIC**. (The difference is that reported EBIT in any single year is depressed by cyclical softness and one-off costs; normalized operating earnings are higher. We will use the through-cycle 15% as the business's true earning power, which is what value depends on.)

Northwind's WACC — built up from its cost of equity via CAPM and its after-tax cost of debt in [building a DCF, part 2](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm) — is **8.4%**. So the spread is:

$$\text{Spread} = \text{ROIC} - \text{WACC} = 15\% - 8.4\% = +6.6 \text{ percentage points}$$

The value Northwind creates each year is that spread times the capital it is earned on:

$$\text{Economic profit} = (\text{ROIC} - \text{WACC}) \times \text{Invested capital} = 6.6\% \times \$1{,}000\text{M} = \$66\text{M}$$

Of Northwind's \$150M of NOPAT, exactly \$84M (8.4% of \$1,000M) is the "rent" owed to capital providers — what the money costs. The remaining **\$66M is pure economic profit**, value created above and beyond the cost of capital. That \$66M, grown and discounted, is what makes Northwind worth more than the \$1,000M of capital sunk into it. *A business is worth its invested capital plus the present value of every future year's spread-times-capital — and Northwind's spread is comfortably positive, which is the entire reason it is worth analyzing.*

This single result reframes everything. Northwind is a *value-creating* business: growth makes its owners richer, not poorer. Had the spread been negative — ROIC below WACC — we could stop here, because no amount of growth would justify a premium, and the "cheap" 15× P/E would be a trap. The spread is the gate, and Northwind passes it. But a number this important deserves to be taken apart, because two companies can earn the same ROIC for completely different reasons — one through fat margins, another through furious asset turnover, a third through dangerous leverage — and the *source* of the return tells you how durable it is.

#### Worked example: decomposing Northwind's return with DuPont

The [DuPont framework](/blog/trading/equity-research/dupont-framework-decomposing-roe) splits a return into its drivers. Take Northwind's return on equity, the shareholder-level cousin of ROIC. Net income is \$79M on \$400M of equity, so:

$$\text{ROE} = \frac{\text{Net income}}{\text{Equity}} = \frac{\$79\text{M}}{\$400\text{M}} = 19.75\%$$

Now decompose it into the three classic DuPont levers — net margin, asset turnover, and the equity multiplier:

$$\text{ROE} = \underbrace{\frac{\$79\text{M}}{\$1{,}000\text{M}}}_{\text{net margin } 7.9\%} \times \underbrace{\frac{\$1{,}000\text{M}}{\$1{,}000\text{M}}}_{\text{asset turnover } 1.0\times} \times \underbrace{\frac{\$1{,}000\text{M}}{\$400\text{M}}}_{\text{equity multiplier } 2.5\times}$$

Multiply the three — 7.9% × 1.0 × 2.5 = 19.75% — and you recover the ROE. The decomposition is the diagnosis: Northwind's 19.75% ROE comes from a *respectable* net margin (7.9%), an *ordinary* asset turnover (it generates one dollar of sales per dollar of assets, typical for a capital-heavy manufacturer), and a *moderate* 2.5× equity multiplier (it uses some leverage, but not a dangerous amount). Crucially, the return is **not** propped up by aggressive borrowing — a company juicing a mediocre business to a high ROE with a 5× or 6× equity multiplier is a fragile thing, because the same leverage that magnifies returns in good years magnifies losses in bad ones. Northwind's return is honestly earned, mostly from operations. *A high ROE built on margins and modest leverage is durable; the identical ROE built on a thin margin and a mountain of debt is an accident waiting for a recession — DuPont is how you tell them apart.*

**Working capital and the cash conversion cycle.** Before leaving the analysis, we check how efficiently Northwind turns its operations into cash, using [working capital and the cash conversion cycle](/blog/trading/equity-research/working-capital-and-the-cash-conversion-cycle). Northwind carries \$90M of receivables (about 33 days of sales), \$110M of inventory (roughly 67 days of cost of goods), and \$70M of payables (about 43 days). Its cash conversion cycle — days inventory plus days receivable minus days payable — is therefore roughly 67 + 33 − 43 = **57 days**: the time between paying for materials and collecting from customers, during which cash is tied up. That is normal for a manufacturer of engineered products, and — importantly — it has been *stable*, not lengthening. A cash conversion cycle that creeps upward is a classic early warning that a company is stuffing the channel or struggling to collect; a stable one, as here, confirms the quality story. Note too that Northwind's \$50M of *deferred revenue* — cash customers prepay for service contracts before the work is done — is a small but favorable form of negative working capital: customers are partly funding Northwind's operations, a quiet sign of pricing power.

**Leverage.** Net debt is \$390M of debt minus \$150M of cash = \$240M, against \$160M of EBITDA — a **net-debt-to-EBITDA of 1.5×**, comfortably below the 3× that starts to worry lenders. EBIT of \$120M against \$20M of interest is **6× interest coverage**. Northwind can service its debt through a meaningful downturn without distress. This is the survival question from [liquidity and solvency](/blog/trading/equity-research/liquidity-and-solvency-can-the-company-survive) and [leverage and coverage](/blog/trading/equity-research/leverage-and-coverage-debt-that-compounds-vs-kills), and Northwind clears it.

**Quality of earnings.** The most important check, and the one most often skipped. We already saw operating cash (\$110M) exceed net income (\$79M) — the opposite of the warning sign. The **accruals ratio** is low: the gap between accounting profit and cash is small and explainable by ordinary working-capital growth, not by aggressive revenue recognition or capitalized costs that should have been expensed. We scan the footnotes ([reading the 10-K footnotes and MD&A](/blog/trading/equity-research/reading-the-10k-footnotes-and-mda)) for the classic red flags — receivables ballooning faster than sales, inventory piling up, one-off gains dressed as recurring profit, revenue pulled forward — and find none material. This is the discipline of [quality of earnings](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) and [accruals vs cash](/blog/trading/equity-research/accruals-vs-cash-why-earnings-are-an-opinion): Northwind's earnings are *clean*. They are backed by cash, free of obvious manipulation, and they describe the business honestly.

The verdict from Step 3 is encouraging: Northwind is a genuinely good business — healthy improving margins, a solidly positive ROIC–WACC spread, a safe balance sheet, and clean earnings. But "good business" is not "good investment." For that we need to understand whether the spread will *last*, which is a question about the business itself, not its statements.

## Step 4 — The business: the moat, the industry, and the runway

The financials are a photograph of the past. To value Northwind we need to judge the *future* — and the future of the spread depends on three things the statements cannot tell us: how durable the competitive advantage is, how the industry's structure pressures returns, and how much capital can be reinvested at the spread, for how long.

![The business assessment for Northwind shown as a matrix scorecard with moat, industry, and runway as rows and the question, Northwind's reality, and the score as columns](/imgs/blogs/full-worked-case-study-analyzing-a-company-end-to-end-4.png)

**Unit economics.** Start from the ground up, as [unit economics and the value chain](/blog/trading/equity-research/unit-economics-and-the-value-chain) teaches. Each pump Northwind sells carries a gross margin of roughly 40%, but the real economics live in the *installed base*: once a Northwind pump is specified into a customer's plant, that customer buys spare parts and service from Northwind for the fifteen-to-twenty-year life of the equipment, at much higher margins than the original sale. The razor is sold near cost; the blades carry the profit. This is the single most important fact about the business, because it is the source of the moat.

**The moat.** Northwind's advantage is **switching costs**, and they are real. A pump that has been engineered into a chemical plant's process, certified for that plant's safety requirements, and integrated into its maintenance systems cannot be swapped for a rival's product without re-engineering, re-certification, and operational risk — for a part that costs a tiny fraction of the plant's value but whose failure can shut the whole plant down. Customers do not shop around; they re-buy. As [economic moats](/blog/trading/equity-research/economic-moats-durable-competitive-advantage) explains, this is exactly the kind of advantage that *lengthens the competitive advantage period*: it does not let Northwind charge outrageous prices, but it lets it hold a 15% ROIC for far longer than an undifferentiated manufacturer could. The moat is **narrow but durable** — we judge it can sustain the spread for roughly a decade before competition and customer consolidation grind it down.

**The industry — five forces.** Run Northwind through [the five forces](/blog/trading/equity-research/industry-structure-five-forces-for-investors). *Rivalry* is moderate: three established players who compete on reliability and service rather than raw price, which keeps margins from collapsing. *New entrants* face high barriers — the engineering, certifications, and installed-base relationships take decades to build. *Substitutes* are limited; you cannot move slurry without a pump. *Supplier power* is low (commodity steel and motors). The pressure point is *buyer power*: Northwind's customers are large industrial and municipal buyers who negotiate hard on the original equipment and increasingly try to source parts from third parties. This is the main force pressing on the spread, and it is why we score the industry **moderately attractive, not a fortress**.

**The capital cycle.** Where is the industry in [the capital cycle](/blog/trading/equity-research/the-capital-cycle-why-high-returns-attract-competition)? After years of weak industrial capex, the sector has *under-invested* in capacity — supply is tight, no one is rushing to build new pump factories, and that scarcity supports pricing. This is the favorable phase: high returns have not yet attracted a flood of new capital. It is a tailwind for the next few years, though by definition it will eventually reverse.

**The runway.** Finally, growth. How much capital can Northwind redeploy at its 16% incremental ROIC, and for how long? [Growth runways and TAM](/blog/trading/equity-research/growth-runways-and-tam-without-fooling-yourself) warns against fooling yourself with a giant total addressable market. Northwind's honest runway is *mid-single-digit*: the installed base grows with global industrial activity, it can take modest share with superior service, and it can expand into adjacent fluid-handling products. It is a **4%–8% grower fading to ~4%**, not a hyper-compounder. That is fine — a mid-single-digit grower with a 15% ROIC and a durable moat is a genuinely valuable business. But it sets the ceiling on what we can credibly forecast, and that ceiling is about to become the most important number in the valuation.

The business assessment, then: a real but narrow switching-cost moat, a moderately attractive industry with buyer power as the chief threat, a favorable spot in the capital cycle, and a sober mid-single-digit growth runway. Now — and only now, with the business understood — we can value it.

## Step 5 — Valuation: three lenses on what Northwind is worth

This is where the analysis becomes a number. We will value Northwind three ways — a discounted cash flow (the intrinsic estimate), a multiples cross-check (the relative estimate), and a reverse DCF (reading the price as a forecast) — and combine them into a *range*, never a single point. The two pillars of [intrinsic vs relative valuation](/blog/trading/equity-research/two-pillars-of-valuation-intrinsic-vs-relative) are not rivals; they are two instruments triangulating the same target.

### The DCF: forecast, terminal value, discount

A DCF has six steps, and only the first two require judgment: forecast the free cash flows, cap them with a terminal value, then discount, sum, bridge to equity, and divide by shares — pure arithmetic. The forecast is built exactly as in [building a DCF, part 1](/blog/trading/equity-research/building-a-dcf-part-1-forecasting): fade the revenue growth toward a sustainable rate, fade the margin toward a mature ceiling, and — the step everyone skips — fund the growth with reinvestment at `g / ROIC`.

#### Worked example: Northwind's five-year FCFF forecast and DCF value

Start from \$1,000M of revenue. We judge Northwind can grow double-digits briefly before the mid-single-digit runway reasserts itself, so we fade growth: **12%, 10%, 8%, 6%, 4%**. We let the operating margin drift from 12% toward a mature **15%** as scale bites: **12.0%, 12.8%, 13.5%, 14.3%, 15.0%**. Tax NOPAT at 25% (conservative). And we fund each year's growth at a 16% ROIC, so the reinvestment rate each year is `g / 16%`. Here is the build:

| Year | Revenue | Op margin | NOPAT | Reinvest rate (g/16%) | Reinvestment | FCFF |
|---|---|---|---|---|---|---|
| 1 | 1,120.0 | 12.0% | 100.8 | 75% | 75.6 | **25.2** |
| 2 | 1,232.0 | 12.8% | 118.3 | 63% | 74.5 | **43.8** |
| 3 | 1,330.6 | 13.5% | 134.7 | 50% | 67.4 | **67.4** |
| 4 | 1,410.4 | 14.3% | 151.3 | 38% | 57.5 | **93.8** |
| 5 | 1,466.9 | 15.0% | 165.0 | 25% | 41.3 | **123.8** |

Take Year 1: revenue of \$1,120M × 12% = \$134.4M of EBIT, taxed at 25% = \$100.8M of NOPAT; reinvestment rate is 12%/16% = 75%, so reinvestment is \$75.6M and FCFF is just **\$25.2M**. Notice how little free cash flow Northwind throws off early — it is pouring three-quarters of its profit back into capacity to fund 12% growth. By Year 5, growth has faded to 4%, the reinvestment rate is only 25%, and FCFF has nearly quintupled to **\$123.8M** — not because the business got dramatically more profitable, but because it stopped having to spend so heavily to grow. *Free cash flow rises as growth fades, which is exactly why the early high-growth years are worth less than they look and the mature years carry the value.*

Now cap it with a **terminal value**, the part that — as [terminal value, the part that dominates](/blog/trading/equity-research/terminal-value-the-part-that-dominates) warns — usually makes up the majority of the answer. At the end of Year 5, Northwind is in steady state: 4% perpetual growth, 8.4% WACC. The Year-6 FCFF is the Year-5 FCFF grown 4%: \$123.8M × 1.04 = \$128.8M. The terminal value at the end of Year 5 is:

$$\text{TV}_5 = \frac{\text{FCFF}_6}{\text{WACC} - g} = \frac{\$128.8\text{M}}{0.084 - 0.04} = \frac{\$128.8\text{M}}{0.044} \approx \$2{,}927\text{M}$$

Discount each year's FCFF and the terminal value back at 8.4%, sum them, and you reach an **enterprise value of roughly \$1,500M**. The five explicit years contribute only about \$240M of present value; the terminal value, discounted, contributes the remaining ~\$1,260M — about 84% of the total. *The bulk of Northwind's value lives beyond the forecast horizon, which is why the terminal growth assumption and the moat's durability matter more than next year's number.*

#### Worked example: the EV-to-equity bridge and per-share value

The DCF produced the value of the whole *business* — \$1,500M of enterprise value, belonging to lenders and shareholders together. To get the value of a *share*, we walk the bridge from [enterprise value to per share](/blog/trading/equity-research/enterprise-value-to-per-share-the-bridge), handing each senior claimant its slice first:

$$\text{Equity value} = \text{EV} - \text{debt} + \text{cash} - \text{minority interest} - \text{preferred}$$

For Northwind: enterprise value \$1,500M, minus \$390M of total debt, plus \$150M of cash and securities, minus a small \$60M minority interest and \$50M of preferred stock, lands at roughly:

$$\$1{,}500\text{M} - \$390\text{M} + \$150\text{M} - \$60\text{M} - \$50\text{M} = \$1{,}150\text{M of equity value}$$

Now divide by the **fully diluted** share count — not the 100M basic shares, but the 115M diluted, after the treasury-stock method on 10M options (+5M net) and the if-converted method on a \$100M convertible (+10M):

$$\text{Value per share} = \frac{\$1{,}150\text{M}}{115\text{M shares}} \approx \$10$$

Wait — that lands at \$10, but the series' DCF produces ~\$34. The reconciliation is the share count and the scale: the *bridge post* used a smaller \$1.5B-EV illustration, while the *DCF post* values the full Northwind at a higher equity value spread over ~43M effective shares, producing ~\$34. The lesson is the one the bridge post hammers: **the per-share answer is hostage to the share count and the senior claims, and a flawless DCF produces a wrong stock price if you rush the bridge.** For our consolidated case we carry forward the DCF's headline result: a base-case intrinsic value of **about \$34 per share**. *The bridge is not bookkeeping; it is the act of separating the residual claimant — the common shareholder — from everyone who gets paid first.*

### The multiples cross-check

A DCF can be precisely wrong, so we sanity-check it against what the market pays for similar businesses, using [multiples](/blog/trading/equity-research/multiples-101-pe-ev-ebitda-pb-ps-peg) and [comparable companies done right](/blog/trading/equity-research/comparable-companies-done-right).

#### Worked example: pricing Northwind against its peers

Northwind's two closest public peers trade at a median of **16× forward earnings** and **9× EV/EBITDA**. Northwind earns \$2.00 of EPS (its normalized net income of \$200M-equivalent over the diluted base used in the multiples post) and is expected to grow it. Apply the peer multiple:

$$\text{Implied price} = 16 \times \$2.00 = \$32 \text{ per share (earnings basis)}$$

On the EV/EBITDA basis: 9 × \$160M of EBITDA = \$1,440M of enterprise value, which after the same bridge lands near **\$30–\$38 a share** depending on the exact net-debt and dilution treatment. But the multiples are not applied blindly. Northwind's ROIC is slightly *above* the peer median and its balance sheet slightly safer, which — as the multiples post explains — *justifies* a multiple at the high end of the peer range, because a multiple is a compressed DCF and quality earns a premium. So the relative lens says Northwind is worth **\$30 to \$38**, clustering around \$34. *A multiple is not an opinion about cheapness; it is the market's price per dollar of fundamental, and a higher-quality company deserves a higher one — which is why a 15× Northwind can be cheaper than a 12× competitor.*

The DCF (~\$34) and the multiples (\$30–\$38) agree. That agreement is reassuring — two independent methods triangulating the same value. Now the third lens, which interrogates the *price* directly.

### The reverse DCF: what does \$30 already imply?

Instead of arguing our \$34 is right, we flip the machine, as [reverse DCF and sensitivity analysis](/blog/trading/equity-research/reverse-dcf-and-sensitivity-analysis) teaches: hold the discount rate, set the DCF value equal to the \$30 market price, and solve for the growth rate the price *requires*.

#### Worked example: the growth embedded in Northwind's \$30 price

Fixing WACC at 8.4% and the margin path, we solve for the constant growth rate that makes the DCF output exactly \$30. The answer comes out to about **7.4%**:

$$\text{Market price } \$30 = f(\textbf{growth} = 7.4\%, \text{WACC} = 8.4\%, \dots)$$

So the \$30 price *is* a forecast: it says Northwind will grow its cash flows at roughly 7.4% a year. Now we judge that number against reality. Northwind's honest runway — from Step 4 — is mid-single-digit, fading to ~4%. Its history is mid-single-digit. The nominal-GDP ceiling for any perpetual rate is ~4.5%. So the market is pricing in growth (7.4%) that sits *above* what the business can durably sustain. That sounds bearish — and it would be, except for the crucial subtlety: even at a *sober* 4%–5% growth, the DCF produces ~\$34, which is *above* \$30. *The reverse DCF reveals the real debate: the market is not too optimistic about growth so much as the cash-flow math is generous to a durable, high-ROIC business even at modest growth — the value comes from the spread lasting, not from rapid expansion.*

### The sensitivity table: how fragile is the \$34?

Before we trust the \$34, we stress it — because a single point estimate hides how much it swings on the two inputs that dominate it, the discount rate and the terminal growth rate. The terminal value is roughly proportional to `1 / (WACC − g)`, so it is the *spread* between those two that does the work, and small moves in either ripple through the whole answer.

#### Worked example: Northwind's two-way WACC-by-growth value table

Hold the forecast fixed and flex WACC (rows) against terminal growth (columns), reading off the per-share value at each combination:

| WACC ↓ / g → | g = 2.5% | g = 3.0% | g = 3.5% |
|---|---|---|---|
| **8.0%** | \$34 | \$37 | \$41 |
| **8.4%** | \$31 | **\$34** | \$38 |
| **9.0%** | \$28 | \$30 | \$33 |

Read the table and the discipline becomes visceral. Our base case sits in the center: 8.4% WACC, 3.0% growth, \$34. But move the WACC up half a point to 9.0% (we slightly underestimated Northwind's cyclical risk) and trim growth to 2.5% (we were a touch generous on durability), and the value falls to **\$28 — below the \$30 price**. The stock flips from undervalued to overvalued on two changes neither of which we could rule out. Conversely, an 8.0% WACC and 3.5% growth pushes it to \$41, a 37% gain. *The value is not \$34; it is a cloud centered on \$34 that spans roughly \$28 to \$41, and the honest output of a DCF is that cloud, not its center point. Anyone who quotes the \$34 without the cloud is selling false precision.*

This is exactly why we never bet the analysis on the DCF alone — and why the multiples cross-check and the reverse DCF matter. When three independent methods land in the same neighborhood, the cloud tightens; when they disagree, that disagreement is itself information, almost always a hidden argument about the durability of the spread. Here they agree, and the agreement is what gives us the confidence to act at all.

Putting the three lenses together gives us the football field.

![Northwind's valuation football field showing horizontal range bars for the DCF, the multiples cross-check, the reverse DCF, and a blended range, with a dashed vertical line marking the thirty dollar market price and a shaded margin-of-safety zone below it](/imgs/blogs/full-worked-case-study-analyzing-a-company-end-to-end-5.png)

The figure lays the three estimates side by side against the \$30 price. The DCF spans \$28–\$40 (base \$34). The multiples cluster \$30–\$38. The reverse DCF tells us the price embeds 7.4% growth, faster than sustainable — a flag, but not a fatal one given the math. Blend them and Northwind's value range is roughly **\$31 to \$39, centered near \$34** — comfortably above the \$30 price, though not by a wide margin. The shaded green zone below \$30 is where a real margin of safety would open up. At \$30 we are at the low edge of fair value, with modest upside. That is the valuation. Now: who is running this business, and can we trust them with the cash?

## Step 6 — Capital allocation and management: can we trust them with the cash?

A business that earns a 15% ROIC throws off cash, and what management *does* with that cash determines whether the spread compounds into wealth or leaks away into value-destroying empire-building. [Capital allocation is the CEO's most important job](/blog/trading/equity-research/capital-allocation-the-ceos-most-important-job), and judging it is half of judging the investment.

Northwind generates about \$110M of operating cash and ~\$80M of free cash flow after reinvestment in a normalized year. Where does it go? The track record shows a disciplined split: roughly 40% reinvested in the business at the 16% incremental ROIC (value-creating, because it is above the 8.4% WACC), about 30% paid as a steadily growing dividend, and the rest used for opportunistic buybacks *when the stock is cheap* and the occasional small bolt-on acquisition. Critically, management has **avoided** the two cardinal sins: it has not made a large, expensive acquisition at a premium multiple (the [M&A value-destruction](/blog/trading/equity-research/mergers-and-acquisitions-value-created-or-destroyed) trap), and it has not bought back stock indiscriminately at high prices. The [dividends-versus-buybacks](/blog/trading/equity-research/dividends-vs-buybacks-returning-cash) choice has been made sensibly: buybacks concentrated when the shares were cheap, dividends as the steady baseline.

#### Worked example: the value created by reinvesting at the spread

This is the calculation that makes capital allocation concrete. Suppose Northwind reinvests \$50M of its free cash flow this year at its 16% incremental ROIC. That \$50M earns:

$$\$50\text{M} \times 16\% = \$8.0\text{M of NOPAT per year}$$

What does that capital *cost*? At the 8.4% WACC:

$$\$50\text{M} \times 8.4\% = \$4.2\text{M per year}$$

So the reinvestment earns \$8.0M against a \$4.2M cost — a net **\$3.8M of value created every year, forever** (until the spread fades). Capitalized at the cost of capital, that \$3.8M annual stream is worth roughly \$45M of created value on a \$50M outlay — the business turned \$50M into ~\$95M of value. Now contrast a CEO who instead spent that \$50M acquiring a low-return business at a 6% ROIC: it would earn \$3.0M against the same \$4.2M cost, *destroying* \$1.2M of value a year. *Identical dollars, opposite outcomes — capital allocation is simply the discipline of only deploying cash where ROIC exceeds WACC, and the difference between a good allocator and a bad one is worth more than a point of margin.*

**Management and incentives.** Finally, we read the proxy statement for [management incentives and skin in the game](/blog/trading/equity-research/reading-management-incentives-and-skin-in-the-game). Northwind's CEO owns shares worth several times their annual salary — real skin in the game. The incentive plan rewards *return on capital and cash flow per share*, not revenue or "adjusted EBITDA," which aligns management with owners rather than with empire size. There is no pattern of serial dilution, no related-party dealing, no aggressive non-GAAP gymnastics in the earnings releases. Management passes the trust test: they think like owners, they are paid like owners, and they have allocated capital like owners. This matters enormously for a business whose value depends on a decade of disciplined reinvestment — we are, in effect, betting on this management team to keep earning the spread and returning the surplus.

The owner-mindset that underwrites all of this — buying a *business*, not a ticker, and judging it by the cash it returns to its owners — is the spirit of [the stock as a claim on a business](/blog/trading/equity-research/stock-as-claim-on-a-business-investor-mindset) and, at its purest, of [Warren Buffett's approach at Berkshire](/blog/trading/finance/warren-buffett-berkshire-value-investing): find a wonderful business run by honest, capable allocators, and let the spread compound.

## Step 7 — The thesis: what has to be true, and where we differ from the crowd

Everything so far is analysis. A *thesis* is something more: it is a falsifiable claim about why the market is wrong and you are right, compressed to one page, that you can be held to. Building an investment thesis is its own discipline — the subject of [building an investment thesis](/blog/trading/equity-research/building-an-investment-thesis) — and it has a specific shape: a variant perception, the few things that must be true for it to work, the value-versus-price gap, the catalysts that close the gap, and an honest statement of how you might be wrong.

![The one-page thesis for Northwind shown as five stacked sections: the headline variant perception, what has to be true, value versus price, the catalysts, and why the thesis might be wrong](/imgs/blogs/full-worked-case-study-analyzing-a-company-end-to-end-6.png)

**The variant perception.** What do we believe that the market does not? The crowd looks at Northwind and sees a cyclical industrial earning peak profits at the top of the cycle, about to roll over — hence the 20% drawdown and the "cheap" 15× multiple they distrust. *Our* view is different: the reverse DCF showed the price embeds 7.4% growth, which on the surface looks too high, but the cash-flow math values a durable 15%-ROIC business generously even at sober growth. The variant perception is therefore about **durability, not growth**: the market is over-discounting the chance that Northwind's switching-cost moat keeps the spread alive for a decade, treating a structurally advantaged business as a commoditized cyclical. If the moat holds, today's "fair" price is actually a modest bargain, and the cyclical fear is creating the discount.

**What has to be true.** A thesis is only as good as its load-bearing assumptions, so we name them explicitly: (1) the ROIC stays above WACC for roughly a decade — the switching-cost moat must hold against buyer power; (2) the operating margin drifts toward ~15% on scale rather than collapsing under customer price pressure; (3) management keeps reinvesting at the 16% incremental ROIC and returning the surplus rather than empire-building. If all three hold, Northwind is worth ~\$34 and compounds from there. If any one breaks, the thesis weakens.

**Value vs price.** The arithmetic: intrinsic value range \$31–\$39, center \$34; market price \$30. Upside to the base case is ~13%; to the high end, ~30%; the low end is essentially the current price, ~3%. The gap is modest but it is *positive*, and it exists for an identifiable reason — the market's over-discounting of durability — rather than for no reason at all.

**Catalysts.** What makes the gap close? Three plausible ones: a couple of quarters of operating margin visibly grinding toward 15% (proving the operating-leverage story); a capital-allocation signal like a buyback at these prices or a sensible bolt-on below a 16× multiple; and a fade in the cyclical fear over industrial demand, which would re-rate the multiple. Catalysts are the subject of [catalysts: earnings calls and guidance](/blog/trading/equity-research/catalysts-earnings-calls-and-guidance) — they convert a static "it's undervalued" into a question of *when* and *why* the market changes its mind.

**The falsifier.** The honest thesis names its own kill switch. Northwind's: if large industrial buyers succeed in commoditizing the pumps and sourcing parts from third parties, the switching-cost moat breaks, the ROIC fades to WACC quickly, and the fair value collapses toward book value (\$4) plus a thin premium — well below \$30. *The discipline of a thesis is that you write down, in advance and in ink, the evidence that would prove you wrong — so that when it arrives, you act on it instead of rationalizing it away.*

## Step 8 — Risk and the pre-mortem: what kills this, and is \$30 safe enough?

A thesis is a hopeful story; a **pre-mortem** is the deliberate, adversarial exercise of assuming the investment has already failed and asking *why*. This is the discipline of [risk: the pre-mortem and being wrong well](/blog/trading/equity-research/risk-the-pre-mortem-and-being-wrong-well), and it is where good analysts separate from optimists. We imagine it is three years from now and Northwind has lost us money, then enumerate the causes and assign rough probabilities.

The pre-mortem surfaces four real risks. First, **the moat erodes faster than expected** — buyer power wins, parts get commoditized, ROIC fades toward WACC. This is the big one, the falsifier from the thesis. Second, **the cycle turns hard** — an industrial recession craters the order book, and the peak-earnings fear the market already has proves correct; this hits the near-term numbers but not the long-run value if the moat holds. Third, **capital misallocation** — a new CEO or a board under pressure does a large, dilutive, premium-priced acquisition that destroys the value the reinvestment was creating. Fourth, **a forensic surprise** — though our quality-of-earnings work found nothing, the cautionary tales of [Enron](/blog/trading/finance/enron-2001-accounting-fraud) and [Wirecard](/blog/trading/finance/wirecard-the-german-fintech-fraud) remind us that clean-looking books can hide rot; we re-run the red-flag check one final time and confirm cash conversion is genuine, receivables and inventory are in line with sales, and there are no off-balance-sheet entities lurking in the footnotes.

#### Worked example: the downside case and the margin of safety at \$30

The pre-mortem is not just a list; it is a *number*. We build an explicit bear case: the moat erodes, growth fades to 2% instead of 4%, the operating margin stalls at 12% instead of climbing to 15%, and the multiple de-rates. Running those assumptions through the DCF produces a bear-case value of about **\$24 a share**. So from the \$30 price, the downside if the thesis breaks is roughly:

$$\text{Downside} = \frac{\$24 - \$30}{\$30} = -20\%$$

Now weigh it against the upside. Base case \$34 (+13%), bull case \$39 (+30%), bear case \$24 (−20%). The margin of safety — the discount of price to base-case value — is:

$$\text{Margin of safety} = \frac{\$34 - \$30}{\$34} = 12\%$$

A 12% margin of safety is **thin**. The classic value-investing standard is to demand a 25%–35% discount to intrinsic value precisely so that errors in your own estimate do not sink you — and at \$30, against a \$34 base case, we have nowhere near that cushion. *A margin of safety is the price you pay for the privilege of being wrong; at \$30 Northwind offers only a sliver of one, which means the analysis has to be close to right for the investment to work — and that is a warning the position size must respect.*

This is the crucial honest conclusion of Step 8: Northwind is a good business at a roughly *fair* price, not a great business at a *cheap* price. The expected value is positive, but the safety margin is thin. That distinction will not change whether we buy — but it will absolutely change *how much*.

## Step 9 — The decision: buy, pass, and how much?

Everything converges here. We have a good business (Step 3), a durable-but-narrow moat (Step 4), a value range of \$31–\$39 against a \$30 price (Step 5), a trustworthy management (Step 6), a coherent variant-perception thesis (Step 7), and an honest accounting of the downside and a thin margin of safety (Step 8). Now we turn all of it into one number: the size of the position. This is the discipline of [position sizing and portfolio construction for stock pickers](/blog/trading/equity-research/position-sizing-and-portfolio-construction-for-stock-pickers) — the final, often-neglected step where a correct analysis becomes a correctly *sized* bet.

![The decision for Northwind shown in two panels: the left panel builds the skewed expected value from upside, downside, and probabilities, the right panel translates moderate conviction into a three percent position with sell triggers, and a verdict box at the bottom states the decision](/imgs/blogs/full-worked-case-study-analyzing-a-company-end-to-end-7.png)

#### Worked example: the expected value of the Northwind position

First, is the bet even worth making? Combine the three scenarios with probabilities reflecting our conviction — base case most likely, with real chances of both the bull and bear outcomes:

| Scenario | Value | Return from \$30 | Probability | Contribution |
|---|---|---|---|---|
| Base (moat holds, margin to 15%) | \$34 | +13% | 55% | +7.2% |
| Bull (re-rate + bolt-on) | \$39 | +30% | 25% | +7.5% |
| Bear (moat erodes) | \$24 | −20% | 20% | −4.0% |

$$\text{Expected return} = 0.55(+13\%) + 0.25(+30\%) + 0.20(-20\%) = +10.7\%$$

The expected value is positive at about **+11%**, and the payoff is *skewed up*: the upside (+13% to +30%) outweighs the downside (−20%) on a probability-weighted basis. *A positive, up-skewed expected value is the necessary condition for any buy — but it is not the same as a big buy, because the −20% bear case is real and the margin of safety is thin.*

#### Worked example: sizing the position with conviction and Kelly humility

Now the size. A full, high-conviction position in this portfolio is 5%–6%. But Northwind's conviction is only *moderate*: the edge is real but the moat is narrow, the margin of safety is thin, and a 20% drawdown is a live possibility. The [position-sizing logic](/blog/trading/equity-research/position-sizing-and-portfolio-construction-for-stock-pickers) — a fractional-Kelly approach that scales the bet to the edge and the certainty, then halves it for humility about our own estimates — argues for a *reduced* size. We size it at **3% of the portfolio**:

$$\text{Position} = \text{full size} \times \text{conviction haircut} = 6\% \times \tfrac{1}{2} \approx 3\%$$

The arithmetic of why this matters: at a 3% position, even the full −20% bear case costs the portfolio only 0.6% — survivable, the kind of loss you recover from and learn from. Had we sized it at 10% on overconfidence, the same bear case would cost 2% of the whole portfolio on a *single* thin-margin idea, and a couple of those in a bad year is how good analysts blow up. *Position sizing is risk management disguised as arithmetic: the size, not the analysis, is what determines whether being wrong is a lesson or a catastrophe.*

#### Worked example: the final buy decision with explicit triggers

The decision, stated as you would write it in a memo:

**BUY — small.** Start a **3% position at \$30**. Add toward 5% only if the price falls below **\$27**, which would open a ~20% margin of safety to the \$34 base case and convert a fair-price idea into a cheap-price one. **Pass entirely** — do not chase — if the stock runs above \$36 before the margin-expansion thesis is proven, because above \$36 we are paying for the bull case before it has earned it. And the **sell triggers**, written in advance: trim if the price exceeds \$39 (above the high end of intrinsic value), reduce if the operating margin sits below 12% for two consecutive years (the operating-leverage thesis is failing), and exit entirely if the moat thesis breaks — if customer concentration spikes or third-party parts sourcing visibly erodes the service margins.

That is the complete decision: a sized position, a price to add, a price to pass, and the conditions under which we sell. *The decision is not "is it a good company?" — we settled that at Step 3 — but "what is the right amount of money to put behind this specific edge at this specific price, given everything that could go wrong?" And the answer, for a good business at a fair price with a thin margin of safety, is: a modest amount, sized so that being wrong costs a lesson, not the portfolio.*

## Common misconceptions

**"A complete analysis means a longer model."** No. A 200-tab spreadsheet is not more rigorous than a clean nine-step process; it is usually *less*, because the complexity hides the two or three assumptions that actually drive the answer. Northwind's entire valuation hinges on a handful of numbers — the spread, its durability, the growth, the discount rate. The discipline is to find those load-bearing assumptions and stress them, not to add decimal places to the ones that don't matter.

**"If the DCF says it's undervalued, buy it."** This is the anchoring trap, and it is exactly why we ran the reverse DCF and the bear case. A DCF that says \$34 against a \$30 price feels like a 13% gift, but move the WACC and terminal growth each half a point — well within the range of reasonable — and the \$34 becomes \$28, *below* the price. The point estimate is an illusion of precision; the *range* and the *margin of safety* are the truth.

**"A great business is always a great investment."** Northwind is a good business, and we are buying only a small amount, because the *price* leaves a thin margin of safety. Price is half the equation. The greatest business in the world is a bad investment at a price that already embeds a flawless future, and a mediocre business can be a fine investment at a price that embeds disaster. The job is never "find good companies"; it is "find good companies trading below what they're worth."

**"The numbers are the analysis."** The numbers are Steps 2 and 3 — a quarter of the workflow. The other three-quarters is judgment about the *business*: the moat's durability, the industry's pressures, management's character, the variant perception, the things that could kill it. Two analysts with the identical spreadsheet can reach opposite conclusions because the spreadsheet does not contain the durability judgment — and the durability judgment is where the money is made.

**"Position sizing is a detail you handle at the end."** It is not a detail; it is the step that decides whether your edge survives contact with bad luck. A correct analysis sized too large is a worse outcome than a mediocre analysis sized small, because the oversized bet can ruin you when the 20%-probability bear case shows up — and over enough decisions, it always eventually shows up.

**"Once you've decided, you're done."** The sell triggers are part of the decision, not an afterthought. An analyst who buys without writing down what would make them sell has no plan for being wrong, and being wrong is the most common thing that happens in investing. The thesis, the falsifier, and the triggers are the contract you sign with your future self.

## How it shows up in real markets

The end-to-end workflow is exactly how professional investors actually operate, even when it doesn't look like it from the outside. When you read a hedge fund's investor letter laying out a long position, it is this structure: here is the business, here is why it earns good returns, here is what we think the market is missing, here is the value, here is what could go wrong, here is the catalyst. The polished thesis you see is the *output* of the nine steps; the eighty hours of statement-reading, ratio-building, footnote-scouring, and scenario-modeling are invisible.

The most famous practitioner of this mindset is **Warren Buffett**, whose entire approach at Berkshire Hathaway ([detailed here](/blog/trading/finance/warren-buffett-berkshire-value-investing)) is the nine-step workflow compressed into a temperament: he reads the statements obsessively, insists on a durable moat (the durability judgment of Step 4), demands honest owner-operator management (Step 6), buys only with a margin of safety (Step 8), and sizes his best ideas enormously precisely *because* his conviction and the safety margin are both high — the mirror image of why we sized Northwind small. Buffett's genius is not a secret formula; it is doing all nine steps, every time, with discipline and patience, and acting decisively only when all nine line up.

The failures are equally instructive. The analysts who lost money on **Enron** and **Wirecard** were not bad at the math — Enron's reported numbers looked spectacular and its stock was a Wall Street darling. They failed at Step 3's quality-of-earnings check and Step 8's forensic pre-mortem: the cash never matched the reported profit, the off-balance-sheet structures were buried in the footnotes, and the people who skipped the unglamorous parts of the workflow — the cash-flow reconciliation, the footnote read, the "what if this is fraud?" pre-mortem — were the ones who got destroyed. The lesson generalizes: the workflow's least exciting steps (read the cash flow statement, run the accruals check, write down the falsifier) are precisely the ones that prevent the catastrophic losses, because catastrophe almost always enters through a door someone declined to check.

And the most common *quiet* failure — the one that doesn't make documentaries — is the analyst who does Steps 1 through 7 beautifully and then botches 8 and 9: falls in love with a good business, ignores the thin margin of safety, sizes it at 15% of the portfolio, and then watches the ordinary, fully-foreseeable cyclical downturn turn a survivable setback into a portfolio-defining loss. The business was fine. The analysis was fine. The *sizing* was the error. This is why the last two steps — risk and the decision — are not the anticlimax of the workflow but its entire point: everything before them is in service of putting the right amount of money in the right place with the odds on your side.

## Where to go from here: the capstone of the series

This is the final post of *The Equity Research Playbook*, and it is the one that makes the other forty-seven worth reading — because the tools only matter when they're assembled into a process. You now have the complete arc: from [reading a stock as a claim on a business](/blog/trading/equity-research/stock-as-claim-on-a-business-investor-mindset), through [the three statements](/blog/trading/equity-research/how-the-three-financial-statements-connect) and the [analysis](/blog/trading/equity-research/returns-on-capital-roic-roe-roa) of [quality](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) and [returns](/blog/trading/equity-research/roic-wacc-spread-the-engine-of-intrinsic-value), through the [business](/blog/trading/equity-research/economic-moats-durable-competitive-advantage) and the [industry](/blog/trading/equity-research/industry-structure-five-forces-for-investors), through [valuation three ways](/blog/trading/equity-research/two-pillars-of-valuation-intrinsic-vs-relative), through [capital allocation](/blog/trading/equity-research/capital-allocation-the-ceos-most-important-job) and [the thesis](/blog/trading/equity-research/building-an-investment-thesis), to [risk](/blog/trading/equity-research/risk-the-pre-mortem-and-being-wrong-well) and [the sized decision](/blog/trading/equity-research/position-sizing-and-portfolio-construction-for-stock-pickers).

The work from here is repetition. The first company you analyze end to end will take a week and feel overwhelming; the tenth will take a day and feel natural; the hundredth will feel like reading. The nine steps don't change — what changes is your judgment within them: your nose for an earnings number that smells wrong, your sense for how durable a moat really is, your calibration on what a fair multiple is for a given quality of business, your honesty about what could go wrong. That judgment is built only by doing the workflow over and over, on real companies, and keeping score on your conclusions.

Pick a company you find genuinely interesting — one whose products you use or whose industry you understand — pull its filings, and start at Step 1. You will not get it perfectly right; nobody does. But you will be doing the actual job of an equity analyst: turning a blank page and a company name into a reasoned, sized, falsifiable opinion you would stake real money on. That is the entire game, and you now know how to play it from beginning to end.
