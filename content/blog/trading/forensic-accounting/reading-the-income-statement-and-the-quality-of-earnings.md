---
title: "Reading the Income Statement and the Quality of Earnings"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "A beginner-friendly deep dive into every line of the income statement — revenue to net income — and the forensic question that sits underneath it: are these profits real, repeatable, and backed by cash?"
tags: ["income-statement", "earnings-quality", "profit-and-loss", "gross-margin", "operating-margin", "non-gaap", "adjusted-earnings", "restructuring-charges", "cookie-jar-reserves", "revenue-recognition", "forensic-accounting", "financial-statement-analysis", "earnings-management"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 53
---

> [!important]
> **TL;DR** — The income statement is a funnel with six or seven subtractions in it, and *every single subtraction is a judgment somebody makes*. Reading it well means knowing which judgments are honest and which ones are dials.
>
> - Top to bottom, the funnel is: **revenue − COGS = gross profit**, **− operating expenses = operating income**, **− interest − tax = net income**. Each stage answers a different question about the business, which is why the three margins are not interchangeable.
> - **Earnings quality** is one idea with two tests: *is the profit backed by cash* (compare net income to operating cash flow, and then to free cash flow), and *will it happen again* (strip out the one-offs, and check whether the "one-offs" recur).
> - The "adjusted" number is not a lie by construction — but it is a **dumping ground**. In fiscal 2025, 361 of the S&P 500 (72%) reported an adjusted net income or EPS figure, and in aggregate those adjusted figures were **\$271 billion higher** than GAAP net income (Calcbench / Suffolk University, 2025).
> - Gross margin is the hardest line to fake and the most informative when it moves. A manipulator usually has clean access to only *one* margin at a time — the mismatch between the three is the tell.
> - The number to remember: WorldCom moved about **\$3.8 billion** of network line costs out of operating expense and onto the balance sheet as capital assets. Not one dollar of cash changed hands differently. The restatement eventually reached roughly **\$11 billion**, and it was enough to keep the company reporting profits through periods when it was really losing money.

Here is the question every earnings report is really answering, and almost never answers honestly: **how much money did this company actually make?**

You would think this is the easiest question in finance. The company publishes a statement. The statement has a number at the bottom. The number is the profit. Done.

Except the number at the bottom is the *output* of about forty separate judgment calls, and a competent finance department can move most of them a little, all in the same direction, without breaking a single rule. Revenue can be booked a week early. A cost can be reclassified from "expense" to "asset". A truck's useful life can be eight years or eleven. A reserve set aside three years ago for warranty claims that never came can be quietly released back into this quarter's profit. Each move is defensible in isolation. Stacked, they are the difference between missing analyst estimates and beating them by a penny — which, for the executives involved, is often the difference between a bonus and no bonus.

So this article does two things. First, it builds the income statement from zero, one line at a time, assuming you have never read one. Second — and this is the part that matters — it teaches you **earnings quality**: the forensic discipline of asking not *how much profit did they report* but *how much of that profit is real, repeatable, and backed by cash*.

The diagram below is the mental model for everything that follows. The income statement is a funnel. Money enters at the top as revenue and gets narrowed, stage by stage, by subtractions. What comes out the bottom is net income. And every narrowing — every place the funnel gets tighter — is a place where a human being decided how tight it should be.

![The income statement as a funnel: revenue of \$500.0M narrows through COGS, operating expenses, interest and tax down to net income of \$11.85M, with a management judgment dial marked at each subtraction.](/imgs/blogs/reading-the-income-statement-and-the-quality-of-earnings-1.webp)

Those little tags on the right — *when a cost becomes COGS*, *expense it or capitalize it*, *what counts as operating*, *the effective rate* — are the dials. We are going to learn what each of them does, why a manager would want to turn it, and exactly how you would notice.

Throughout, we will use one running example: **Harbor Lantern Coffee Co.**, a completely hypothetical mid-sized coffee roaster and retailer. Every Harbor Lantern number in this article is invented for teaching. Every number attached to a *named real company* is sourced, and the sources are listed at the end.

## Foundations: how the income statement actually works

Skip nothing here if you are new. Everything in the second half of the article stands on these definitions, and the manipulations are only visible if you know precisely what each line is supposed to contain.

### A period, not a snapshot

The first thing to understand is what *kind* of statement this is.

A **balance sheet** is a photograph. It says: at 11:59pm on 31 December, here is everything the company owned and everything it owed. It is a statement of *position*, taken at an instant. (That is the subject of [reading the balance sheet](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here).)

An **income statement** is a film. It says: *over the twelve months of the year*, here is what came in and what went out. It is a statement of *performance over an interval*. It always carries a period label — "for the year ended 31 December 2025", "three months ended 30 June" — and if you cannot see that label you are not reading it properly.

You will hear it called several things and they all mean the same document: the **income statement**, the **profit and loss statement** (or **P&L**), or the **statement of operations**. In the UK and much of Europe it is often the "statement of comprehensive income".

The third statement, the [cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income), is also a film covering the same period — but it films the bank account instead of the business. The gap between those two films is where this entire article lives. The three statements are welded together in ways that make certain lies impossible to tell in only one place; that interlocking is covered in [how the three statements connect](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock).

### The one rule that makes manipulation possible

Before the lines, the rule that governs them.

Public companies do not report on a cash basis. They report on an **accrual basis**, which means revenue is recorded when it is *earned* — when you deliver the thing — not when the customer's money lands. Expenses are recorded when they are *incurred*, in the same period as the revenue they helped produce, not when the invoice is paid.

Accrual accounting is genuinely better. It stops a business from looking bankrupt in the month it does its best work. But it means the income statement is not a record of cash movements — it is a record of *management's assessment* of what was earned and consumed. That assessment layer has a name (accruals), it is where essentially every earnings manipulation lives, and it is worked through from first principles in [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits). Read that one if any of the following feels slippery. For now, one sentence is enough: **the income statement is an opinion; the cash flow statement is closer to a fact.**

### Line 1 — Revenue

**Revenue** (also *sales*, or *turnover* in British usage, or the *top line*) is the value of goods and services the company delivered to customers during the period.

Three refinements matter immediately.

**Gross versus net revenue.** If you sell a \$100 jacket and expect \$5 of it to come back as returns and \$3 in customer discounts, you report **net revenue** of \$92, not \$100. The \$8 is netted off through a "returns and allowances" reserve — an estimate. Estimate number one.

**Gross versus net *presentation*.** This is a different question and it matters enormously for platforms and marketplaces. If a travel site sells a \$1,000 flight and keeps \$60 of commission, does it report revenue of \$1,000 or \$60? The answer turns on whether the company is the **principal** (it controls the good before transferring it, bears the inventory and pricing risk) or the **agent** (it just arranges the transaction). Principal reports gross, agent reports net. Same economics, same \$60 of profit, revenue line differs by 16x. Companies chasing a growth story have a strong pull toward "principal".

**When is it earned?** Under the current standards — ASC 606 in US GAAP, IFRS 15 internationally — revenue is recognized when the customer obtains *control* of the promised good or service. The framework runs in five steps: identify the contract, identify the distinct performance obligations in it, determine the transaction price, allocate that price across the obligations, and recognize revenue as each obligation is satisfied. If you sell a three-year software licence with support bundled in, you must split the price across the licence and the support and recognize each on its own schedule. Every one of those five steps requires a judgment.

### Line 2 — Cost of goods sold

**Cost of goods sold** (COGS, sometimes *cost of revenue* or *cost of sales*) is the direct cost of producing the specific things you sold this period.

For a coffee roaster: green coffee beans, packaging, the wages of the people running the roasters, the electricity for the roasting line, freight in. For a software company: hosting, the support engineers assigned to customers, third-party licence fees embedded in the product.

The key word is **direct**. The CEO's salary is not COGS. The marketing budget is not COGS. The line is supposed to hold only what varies with, and is consumed by, the units you actually sold.

Two things make COGS a dial. First, the boundary between "direct" and "indirect" is genuinely fuzzy for overhead — is the factory manager's salary COGS or an operating expense? Companies choose, disclose the choice, and can revisit it. Second, COGS is fed by **inventory**, and inventory carries its own thicket of estimates: how much overhead gets absorbed into the cost of each unit, whether slow-moving stock is written down, and which cost-flow assumption (FIFO, weighted average) is used. Costs parked in inventory sit on the balance sheet as an asset and do not touch profit until the goods are sold. Push more cost into inventory and this period's COGS falls.

### Line 3 — Gross profit

$$\text{Gross profit} = \text{Revenue} - \text{COGS}$$

**Gross profit** answers the single most fundamental question about a business: *can we make the thing for less than we sell it for, and by how much?*

Expressed as a percentage of revenue it becomes **gross margin**:

$$\text{Gross margin} = \frac{\text{Revenue} - \text{COGS}}{\text{Revenue}}$$

Gross margin is a structural fact about a business model. Software is 70–90%. Branded consumer goods might be 40–60%. Grocery retail is 20–30%. Contract manufacturing can be under 10%. It moves slowly, for real reasons — input costs, pricing power, product mix, scale — and when it moves fast without one of those reasons, something is wrong.

This makes gross margin the most useful line on the statement for forensic purposes. It is the hardest to fake, because faking it requires either inventing revenue at a plausible margin or hiding real production costs somewhere else, and both leave marks elsewhere in the accounts.

### Line 4 — Operating expenses

Below gross profit sit the costs of *running the company* rather than making the product. Presentation varies, but you will typically see:

- **SG&A** — selling, general and administrative. Salaries of everyone not in production, sales commissions, marketing, rent, legal, insurance, the head office. Often the largest single line.
- **R&D** — research and development. Under US GAAP, most research is expensed as incurred; software development costs can be capitalized once technological feasibility is established, which is a boundary companies get to draw themselves.
- **Depreciation and amortization (D&A)** — the periodic write-down of long-lived assets. If you buy a \$1.2 million roasting line and expect it to last eight years, you expense \$150,000 a year rather than \$1.2 million at once. **Depreciation** applies to physical assets; **amortization** to intangibles like software or acquired customer relationships. Note that D&A is often not shown as its own line — it is frequently buried inside COGS and SG&A, and you have to find it on the cash flow statement.
- **Restructuring, impairment, and "other"** — charges the company presents as outside normal operations: severance from a layoff, writing an acquired brand down to zero, closing a factory.

That last bucket is the one to watch. Keep it in mind; it gets a whole section later.

### Line 5 — Operating income

$$\text{Operating income} = \text{Gross profit} - \text{Operating expenses}$$

**Operating income** — also called **EBIT**, earnings before interest and taxes — is what the *business itself* earns, before considering how it is financed or taxed. Two companies with identical operations but different debt loads should have similar operating income and very different net income. That is exactly the point of the line: it isolates operating performance from financing choices.

You will also constantly meet **EBITDA** — earnings before interest, taxes, depreciation and amortization — which is operating income with D&A added back. EBITDA is not a GAAP measure. It exists because it approximates the cash the operations throw off before capital spending, which is useful for comparing capital-intensive businesses and is what most loan covenants are written against. It is also, for exactly that reason, heavily gamed. Charlie Munger's much-repeated advice — that whenever you see the word EBITDA you should mentally substitute "bullshit earnings" — lands because depreciation is a real cost of a real machine wearing out, and adding it back does not make the machine last longer.

### Below the line — interest, other income, and tax

Beneath operating income, we account for the capital structure and the government:

- **Interest expense** — the cost of debt. **Interest income** — what the company earns on its cash.
- **Other income / (expense)** — foreign exchange gains, gains on selling assets, investment income, the company's share of profits from joint ventures. A grab-bag.
- **Pre-tax income** — operating income adjusted for all of the above.
- **Income tax expense** — and note this is an *expense*, not a cash payment. The tax booked on the income statement and the cash actually wired to the tax authority differ, often substantially, because of timing differences that create deferred tax assets and liabilities.
- **Net income** — the bottom line. What is left for shareholders.

The **effective tax rate** is income tax expense divided by pre-tax income. It is *not* the statutory rate. It moves with geographic mix, tax credits, one-off settlements, and the release of reserves the company held against uncertain tax positions. It is a legitimate line with several legitimate dials on it.

### Net income and earnings per share

**Earnings per share (EPS)** is net income divided by shares outstanding, and it is the number the market actually trades on.

- **Basic EPS** uses the weighted-average shares actually outstanding during the period.
- **Diluted EPS** also counts shares that *would* exist if all options, restricted stock and convertible securities converted. Diluted is the honest one and the one analysts quote.

EPS has a denominator, which means there are two ways to raise it. Raise net income, or shrink the share count through buybacks. A company that misses on revenue, misses on operating income, and still "beats on EPS" has usually done the second thing.

#### Worked example: building Harbor Lantern's P&L from scratch

Let us construct one statement from raw activity, so the lines stop being abstract. Harbor Lantern Coffee Co. is hypothetical; the figures are invented for teaching. Everything is in millions of dollars, for the year ended 31 December 2025.

During the year, Harbor Lantern:

1. Shipped coffee to grocers and ran its own cafés, invoicing customers a total of \$515.0M. Of that, it expects \$15.0M of returns, spoilage credits and volume rebates. **Net revenue: \$500.0M.**
2. Consumed \$315.0M of green coffee, packaging, roasting labour and inbound freight on the goods it actually sold. **COGS: \$315.0M.**
3. Therefore **gross profit: \$500.0M − \$315.0M = \$185.0M**, a gross margin of 185.0 / 500.0 = **37.0%**.
4. Paid \$120.0M for everything else it takes to run the company — store staff, head office, marketing, rent, an \$8.0M restructuring charge for closing eleven underperforming cafés, \$4.0M of stock-based compensation, and \$2.0M of costs from a small acquisition. **SG&A: \$120.0M.**
5. Spent \$15.0M developing new blends, packaging and its ordering app. **R&D: \$15.0M.**
6. Wrote down its roasting equipment, store fit-outs and acquired brands by \$25.0M for the year's wear and tear. **D&A: \$25.0M.**
7. Total operating expenses: \$120.0M + \$15.0M + \$25.0M = **\$160.0M**.
8. **Operating income: \$185.0M − \$160.0M = \$25.0M**, an operating margin of 25.0 / 500.0 = **5.0%**.
9. Paid \$10.0M of interest on its bank debt. **Pre-tax income: \$25.0M − \$10.0M = \$15.0M.**
10. Booked tax at a 21% effective rate: \$15.0M × 0.21 = **\$3.15M**.
11. **Net income: \$15.0M − \$3.15M = \$11.85M**, a net margin of 11.85 / 500.0 = **2.37%** (rounded to 2.4% in the tables and figures below).
12. With 30.0 million diluted shares, **EPS = \$11.85M / 30.0M = \$0.395**, which rounds to **\$0.40**.

Analysts covering Harbor Lantern were expecting **\$0.78**. The company's annual bonus plan pays out only if operating income reaches **\$40.0M**. Its bank covenant requires EBITDA of at least **\$50.0M**, and EBITDA here is \$25.0M + \$25.0M = **\$50.0M** — exactly on the line, one bad quarter from a breach.

**The intuition: the income statement is a single arithmetic chain, and by the time you reach the bottom, twelve separate decisions have each nudged the answer.** Hold onto Harbor Lantern's honest numbers. We are going to run the year again.

## The three margins and what legitimately moves them

Before we turn any dials, learn to read the three margins as three different questions. Beginners treat them as one number of varying strictness. They are not.

![The same \$500.0M of revenue cut three ways: gross profit \$185.0M (37.0%), operating income \$25.0M (5.0%), net income \$11.85M (2.4%), each labelled with the question it answers.](/imgs/blogs/reading-the-income-statement-and-the-quality-of-earnings-2.webp)

**Gross margin — 37.0%.** *Can we make it for less than we sell it for?* This is about the product and the market: input costs, pricing power, mix, manufacturing efficiency. It has essentially nothing to do with how well the company is run at the corporate level.

**Operating margin — 5.0%.** *Can we run the whole company on what's left?* This adds the question of overhead discipline. A business can have a beautiful 60% gross margin and a negative operating margin if it spends recklessly on sales and marketing.

**Net margin — 2.4%.** *What survives lenders and the tax authority?* This adds the capital structure and the tax position. A heavily indebted company and a debt-free one can have identical operating margins and wildly different net margins.

Now the forensic point. Each margin has a distinct set of things that move it *honestly*, and a distinct set of things a manipulator reaches for. They rarely overlap.

| Margin | What legitimately moves it | What a manipulator moves | What that looks like |
| --- | --- | --- | --- |
| **Gross** | Input prices, selling price changes, product mix, scale, freight, factory utilisation | Absorbing more overhead into inventory; delaying inventory write-downs; recognising supplier rebates early; shifting costs into a later period | Gross margin rises while inventory days *also* rise — real efficiency reduces inventory, accounting tricks bloat it |
| **Operating** | Headcount, marketing spend, rent, R&D intensity, operating leverage on fixed costs | Capitalising costs that used to be expensed; stretching depreciable lives; releasing prior-year reserves; reclassifying recurring costs as "restructuring" | Operating margin rises while capex, intangible assets or "one-time" charges rise alongside it |
| **Net** | Debt levels, interest rates, tax jurisdiction mix, one-off asset sales | Booking gains through "other income"; dropping the effective tax rate via reserve releases; buying back shares to lift EPS | Net income grows faster than operating income for several periods, with no change in the business |

The single most useful reading habit is to check the three margins *against each other*. Real improvement in a business usually shows up at the gross line and flows down. Manufactured improvement usually appears at exactly one level and nowhere else.

#### Worked example: same revenue, three different margin stories

Three hypothetical companies each report revenue of \$500.0M and each grew net income 20% this year. All figures invented.

- **Company A** — gross margin up from 35.0% to 37.0%, operating margin up from 4.0% to 5.0%, net margin up from 2.0% to 2.4%. The gain starts at the top and propagates. Almost certainly real: they either raised prices or cut input costs, and the benefit carried down.
- **Company B** — gross margin flat at 37.0%, operating margin up from 4.0% to 5.0%, net margin up from 2.0% to 2.4%. The gain appears only below gross profit. Could be genuine overhead discipline. Could also be \$5.0M of costs moved off the income statement. Go look at capex, capitalized software, and the reserve roll-forward.
- **Company C** — gross margin flat at 37.0%, operating margin flat at 4.0%, net margin up from 2.0% to 2.4%. Nothing about the *business* improved. The entire gain is below operating income: interest, "other income", or tax. \$500.0M × 0.4% = \$2.0M of net income that came from the finance department, not the coffee.

**The intuition: identical bottom-line growth can have three completely different causes, and only the margin ladder tells you which one you are looking at.**

## The dials: how a \$0.40 miss becomes a \$0.79 beat

Harbor Lantern's honest year produced EPS of \$0.40 against a \$0.78 consensus, an operating income of \$25.0M against a \$40.0M bonus threshold, and EBITDA sitting exactly on its covenant. That is a bad year, badly timed.

Now suppose the CFO decides, in the last three weeks of December, to make the year work. Not by inventing customers. Not by forging documents. By turning four dials that every finance department has, each of which is individually defensible and each of which will survive an audit if argued well.

![The four dials: shipping next quarter's orders early (+\$4.5M operating income, receivables +\$12.0M), capitalizing internal software (+\$5.5M, capex +\$6.0M), stretching equipment lives from 8 to 11 years (+\$3.0M, no cash effect), and releasing a prior-year returns reserve (+\$2.0M, no cash effect) — \$15.0M in total.](/imgs/blogs/reading-the-income-statement-and-the-quality-of-earnings-4.webp)

**Dial 1 — pull the orders forward.** Sales calls the twelve largest grocery customers and offers 120-day payment terms if they take their January order in the last week of December. \$12.0M of revenue moves from Q1 2026 into Q4 2025. The goods leave the warehouse, so \$7.5M of cost moves with it. Gross profit rises \$4.5M. No cash arrives — the customers do not pay for four months — so receivables rise \$12.0M.

**Dial 2 — capitalize the software.** Harbor Lantern's engineering team spent \$6.0M this year rebuilding the ordering app. Historically the whole thing was expensed through SG&A. This year, the company determines that \$6.0M of it was incurred *after* technological feasibility was established and therefore qualifies for capitalization. It goes onto the balance sheet as an intangible asset with a three-year life, placed in service in October, so only one quarter of amortization — \$0.5M — hits this year. SG&A falls \$6.0M, D&A rises \$0.5M, operating income rises \$5.5M.

**Dial 3 — stretch the useful lives.** An engineering review concludes that the roasting equipment, with the maintenance programme now in place, will last eleven years rather than the eight originally assumed. Depreciation on that asset base falls \$3.0M for the year. This is a change in accounting *estimate*, applied prospectively; it does not require restating prior years, and the disclosure is a sentence in the notes.

**Dial 4 — release the reserve.** Two years ago Harbor Lantern set aside a reserve for product returns on a packaging change that customers turned out not to mind. \$2.0M of that reserve is no longer needed. Releasing it credits SG&A by \$2.0M — profit appears from a decision made in a prior period.

Total effect on operating income: \$4.5M + \$5.5M + \$3.0M + \$2.0M = **\$15.0M**. That takes operating income from \$25.0M to exactly \$40.0M — the bonus threshold, hit to the dollar.

#### Worked example: the same year, as reported

Here is the full P&L, run again with the dials turned. Same twelve months, same coffee, same customers. All figures hypothetical, in millions.

| Line | As it happened | As reported | Change |
| --- | --- | --- | --- |
| Revenue | \$500.0 | \$512.0 | +\$12.0 |
| COGS | \$315.0 | \$322.5 | +\$7.5 |
| **Gross profit** | **\$185.0** | **\$189.5** | +\$4.5 |
| Gross margin | 37.0% | 37.0% | **unchanged** |
| SG&A | \$120.0 | \$112.0 | −\$8.0 |
| R&D | \$15.0 | \$15.0 | — |
| D&A | \$25.0 | \$22.5 | −\$2.5 |
| Total operating expenses | \$160.0 | \$149.5 | −\$10.5 |
| **Operating income** | **\$25.0** | **\$40.0** | +\$15.0 |
| Operating margin | 5.0% | 7.8% | +2.8 pts |
| Interest | \$10.0 | \$10.0 | — |
| Pre-tax income | \$15.0 | \$30.0 | +\$15.0 |
| Tax at 21% | \$3.15 | \$6.3 | +\$3.15 |
| **Net income** | **\$11.85** | **\$23.7** | **+100%** |
| **EPS (30.0M shares)** | **\$0.40** | **\$0.79** | +\$0.39 |

![The same fiscal year presented twice: as it happened, with a \$0.40 EPS miss and a missed \$40.0M bonus threshold; and as reported, with \$0.79 EPS beating the \$0.78 consensus — while gross margin sits unchanged at 37.0%.](/imgs/blogs/reading-the-income-statement-and-the-quality-of-earnings-3.webp)

Net income exactly doubled. EPS came in at \$0.79 against a \$0.78 consensus — the classic penny beat. EBITDA is now \$40.0M + \$22.5M = \$62.5M against a \$50.0M covenant, comfortable. The bonus pays.

And look at the row that did not move. **Gross margin is 37.0% in both columns.** The channel stuffing added revenue *at the normal margin*, so it left no trace on the most-watched ratio in the statement. Anyone screening for margin anomalies would find nothing.

**The intuition: the four moves that mattered most were invisible on the line most people check.** That is not an accident — it is why those particular dials are the popular ones.

## Earnings quality: cash is the referee

We now have the central question of forensic accounting. Harbor Lantern reported \$23.7M of net income. Is it *good* \$23.7M?

**Earnings quality** is the degree to which reported profit reflects the actual, sustainable economics of the business. High-quality earnings are backed by cash and likely to repeat. Low-quality earnings are backed by estimates and unlikely to repeat. Same number at the bottom of the page; completely different information content.

There are exactly two tests, and everything else is a refinement of one of them.

### Test one: is it backed by cash?

$$\text{Cash conversion} = \frac{\text{Operating cash flow}}{\text{Net income}}$$

**Operating cash flow (CFO)** is the cash the core business generated, reported on the cash flow statement. Because net income is built on accruals and CFO is built on cash, the ratio between them measures how much of the reported profit actually arrived as money.

For a mature company the ratio should typically exceed 1.0, often comfortably — depreciation is a large non-cash expense that reduces net income but not cash, so CFO usually runs above net income. A ratio persistently *below* 1.0 means profit is being recognised faster than cash is being collected, which is either a growing business funding its receivables, or an accounting problem, and you have to work out which.

But there is a trap, and Harbor Lantern walks straight into it.

#### Worked example: Harbor Lantern's cash, both ways

Start with the honest year. Operating cash flow is built by taking net income, adding back non-cash charges, and adjusting for changes in working capital. All figures hypothetical, in millions.

**As it happened:**

- Net income: \$11.85
- Add back D&A (non-cash): +\$25.0
- Receivables rose \$8.0 (cash tied up): −\$8.0
- Inventory rose \$5.0: −\$5.0
- Payables rose \$4.0 (supplier financing): +\$4.0
- **Operating cash flow: \$27.85**
- Capital expenditure: \$20.0
- **Free cash flow: \$27.85 − \$20.0 = \$7.85**

Cash conversion: 27.85 / 11.85 = **2.35x**. Healthy.

**As reported:**

- Net income: \$23.7
- Add back D&A: +\$22.5 (lower, because of the stretched lives)
- Subtract the reserve release (it added to profit but no cash came in): −\$2.0
- Receivables rose \$20.0 — the normal \$8.0 plus \$12.0 of channel-stuffed shipments nobody has paid for: −\$20.0
- Inventory *fell* \$2.5 — the normal \$5.0 increase, less the \$7.5 of goods shipped out early: +\$2.5
- Payables rose \$4.0: +\$4.0
- **Operating cash flow: \$30.7**
- Capital expenditure: \$26.0 — the normal \$20.0 plus the \$6.0 of capitalized software: −\$26.0
- **Free cash flow: \$30.7 − \$26.0 = \$4.7**

Cash conversion: 30.7 / 23.7 = **1.30x**. Still above 1.0. Still, on its face, fine.

![Net income doubled from \$11.85M to \$23.7M, operating cash flow rose only 10% from \$27.85M to \$30.7M, and free cash flow fell 40% from \$7.85M to \$4.7M.](/imgs/blogs/reading-the-income-statement-and-the-quality-of-earnings-5.webp)

Now look at the three numbers side by side:

| Measure | As it happened | As reported | Change |
| --- | --- | --- | --- |
| Net income | \$11.85M | \$23.7M | **+100%** |
| Operating cash flow | \$27.85M | \$30.7M | +10% |
| Free cash flow | \$7.85M | \$4.7M | **−40%** |
| Cash conversion (CFO ÷ NI) | 2.35x | 1.30x | −45% |
| Free cash conversion (FCF ÷ NI) | 0.66x | 0.20x | −70% |

Profit doubled. Cash from operations moved 10%. Free cash flow went *down by 40%*.

The reason CFO barely fell is the trap: **capitalizing a cost inflates operating cash flow too.** The \$6.0M spent on software left the bank account either way, but capitalizing it moves the outflow from the operating section of the cash flow statement to the investing section. CFO looks better, capex looks worse, and the two exactly offset. Which means CFO alone does not catch dial 2 — only **free cash flow**, which subtracts capex, survives that manipulation.

**The intuition: cash conversion is the right question, but free cash flow is the version of it that cannot be gamed by moving a cost from one statement section to another.**

#### Worked example: two companies, one net income, two realities

The simplest version of the same test, on two hypothetical fixtures manufacturers. Both report net income of \$10.0M for the year.

- **Northwind Fixtures**: CFO \$12.0M. Receivables rose \$1.0M on 8% revenue growth. Cash conversion 12.0 / 10.0 = **1.20x**.
- **Southgate Fixtures**: CFO \$2.0M. Receivables rose \$9.0M on 8% revenue growth. Cash conversion 2.0 / 10.0 = **0.20x**.

Both companies will report "\$10 million of net income" in the press release, and both headlines will look identical. But Southgate's profit is sitting in the receivables ledger, not the bank. Either its customers are slow, or its customers do not exist, or the goods will come back. In every one of those cases, next year has a hole in it: \$9.0M of revenue has been recognised whose cash may never arrive, and the *comparison base* for next year now includes it.

**The intuition: two identical bottom lines can mean opposite things, and one division tells you which.**

### Test two: will it happen again?

The second test is recurrence, and it is where the modern earnings game is mostly played.

A profit that came from selling coffee will probably recur. A profit that came from selling a warehouse will not. A profit that came from releasing a reserve absolutely will not — reserves are finite, and once released the same trick is unavailable next year. This is why analysts try to compute *normalized* or *core* earnings: strip out anything that will not repeat, and value the company on what is left.

The problem is that companies figured this out, and now they do the stripping for you.

## The "adjusted" number and the dumping ground beneath it

Open any earnings press release and you will find two sets of numbers. GAAP net income, reported because the law requires it, sitting somewhere below the fold. And **adjusted** net income, adjusted EPS, adjusted EBITDA — displayed prominently at the top, discussed by management on the call, and used by most of the sell side to compute the P/E ratio.

The scale of this is not marginal. According to a 2025 study by Calcbench with Suffolk University, **361 S&P 500 companies — 72% — reported an adjusted net income or EPS figure for fiscal 2025**, and in aggregate those adjusted figures exceeded GAAP net income by **\$271 billion**. Of the companies that adjusted, **87% adjusted upward**. Earlier research by Audit Analytics tracked the same trend: the share of S&P 500 companies using non-GAAP measures rose from 59% in 1996 to 96% in 2016.

Adjustment is not automatically illegitimate. If a company sold a division for a \$400M gain, telling you what the ongoing business earned without it is genuinely helpful. The SEC permits non-GAAP measures under Regulation G, requires the most comparable GAAP measure to be presented with equal or greater prominence, and requires a reconciliation between them. The rules are real.

The problem is *what gets excluded*, and the fact that the exclusions are chosen by the same people whose bonus depends on the answer.

### The three exclusions to interrogate

**Stock-based compensation.** Excluded by a large fraction of technology companies on the grounds that it is "non-cash". It is non-cash in the sense that no money leaves the bank — but shares are issued, existing owners are diluted, and the company would have to pay cash for that labour otherwise. Excluding SBC treats an ownership transfer as free. Warren Buffett has made the same argument in Berkshire Hathaway's shareholder letters for decades: if compensation is not an expense, what is it — and if expenses do not belong in the calculation of earnings, where do they belong?

**Acquisition-related costs and amortization of acquired intangibles.** For a company that acquires something once a decade, excluding these is reasonable. For a serial acquirer that buys six companies a year, "acquisition-related costs" *are* the operating model, and excluding them permanently overstates what the business earns.

**Restructuring.** The most abused of the three, and the one with a name that does the work for it.

### The restructuring charge that never stops

A restructuring charge is supposed to be a discrete event: you close a factory, you take the hit, you move on. The accounting is meant to reflect a one-time economic act.

In practice, a company can announce a restructuring programme every year — a new efficiency initiative, a transformation programme, a network optimisation — and exclude the cost of each one from adjusted earnings. Individually each is presented as non-recurring. Collectively they are a permanent line item.

![Five years of "one-time" restructuring charges of \$9M, \$7M, \$11M, \$6M and \$8M, with adjusted operating income staying above GAAP operating income by exactly the charge each year — a gap that never closes.](/imgs/blogs/reading-the-income-statement-and-the-quality-of-earnings-6.webp)

#### Worked example: five years of one-time charges

Harbor Lantern's restructuring history, hypothetical, in millions:

| Year | "One-time" restructuring charge |
| --- | --- |
| 2021 | \$9 |
| 2022 | \$7 |
| 2023 | \$11 |
| 2024 | \$6 |
| 2025 | \$8 |
| **Five-year total** | **\$41** |

Every single year, adjusted operating income was higher than GAAP operating income by exactly that year's charge. The gap never closed. On average the company excluded \$8.2M a year, and the 2025 charge of \$8.0M is **20% of GAAP operating income of \$40.0M** — a fifth of the reported operating profit, disappeared by calling a cost that arrives every year like a utility bill "one-time".

Now build the full 2025 adjusted bridge from the reported GAAP figures:

- GAAP operating income: **\$40.0M**
- Add back restructuring: +\$8.0M
- Add back stock-based compensation: +\$4.0M
- Add back acquisition-related costs: +\$2.0M
- **Adjusted operating income: \$54.0M**
- Add back D&A of \$22.5M → **Adjusted EBITDA: \$76.5M**

And on the per-share line:

- GAAP net income \$23.7M, GAAP EPS **\$0.79**
- Add back \$14.0M of pre-tax exclusions (\$8.0M + \$4.0M + \$2.0M), tax-effected at 21% — so multiply by (1 − 0.21): \$14.0M × 0.79 = +\$11.06M
- Adjusted net income: \$23.7M + \$11.06M = \$34.76M
- **Adjusted EPS: \$34.76M / 30.0M shares = \$1.16**

GAAP EPS \$0.79. Adjusted EPS \$1.16. The gap is \$0.37, or **47% of the GAAP number**. At a 20x multiple, that gap is worth \$7.40 a share.

**The intuition: a "one-time" charge that appears five years running is an operating expense wearing a costume, and the adjusted number that excludes it is not a cleaner picture of the business — it is a more flattering one.**

### Big baths and cookie jars

Two closely related manoeuvres complete the picture, and both were named publicly by an SEC chairman.

On 28 September 1998, at the NYU Center for Law and Business, SEC Chairman Arthur Levitt gave a speech titled **"The Numbers Game"** that remains the clearest official statement of the problem. He described a "game of nods and winks" among managers, auditors and analysts, named the driver as the pressure to make the numbers and meet street expectations, and enumerated five forms of what he called **accounting hocus-pocus**: **"big bath" restructuring charges**, **creative acquisition accounting**, **"cookie jar" reserves**, abuse of the **materiality** threshold, and premature **revenue recognition**. Every technique in this article is one of those five, twenty-eight years on.

**The big bath.** If you are going to report a loss anyway — a new CEO's first year is the classic moment — report a *huge* one. Take every write-down you might conceivably need, set up generous reserves, kitchen-sink it. The stock has already fallen; the incremental damage of a bigger loss is small. And you have just created a pool of reserves.

**The cookie jar.** In later periods, when you need earnings, you release those reserves. The release credits an expense line, profit appears, and no cash was involved. The loss you took in year one becomes profit in years two and three. A "turnaround" is manufactured out of a chart of accounts.

This is not theoretical. It is precisely what the SEC alleged at Sunbeam, and what it alleged at Nortel — both covered in the case studies below.

## The dials, line by line

We have seen four dials in action. Here is the fuller inventory, organised by where on the statement they live. This is the mechanic's manual: what can be turned, which direction it moves profit, and what mark it leaves.

### On the revenue line

- **Cut-off manipulation** — hold the books open past period end, or ship early. The mark: receivables grow faster than revenue; days sales outstanding jumps; the following quarter is unusually weak.
- **Channel stuffing** — push inventory onto distributors with incentives (discounts, extended terms, generous return rights). The mark: DSO up, distributor inventory up, next-quarter revenue down, returns up two quarters later.
- **Bill-and-hold** — recognise revenue on goods the customer has bought but you are still storing. Legitimate only under a narrow set of conditions (the customer must have requested it, the goods must be identified as theirs, ready to ship, and unavailable to fill other orders). The mark: revenue with no corresponding inventory decline, and a disclosure buried in the revenue-recognition note.
- **Gross-versus-net presentation** — report the whole transaction as revenue when you were really the agent. The mark: revenue growing far faster than gross profit, and a gross margin that falls year after year for no operational reason.
- **Percentage-of-completion estimates** — on long-term contracts, revenue is recognised as the project progresses, and "progress" is management's estimate of costs incurred versus total expected costs. Underestimate total costs, and you recognise revenue faster. The mark: contract assets ("unbilled receivables") growing faster than billings.
- **Related-party or round-trip sales** — sell to an entity you control, or to a customer you are simultaneously buying from. The mark: a customer concentration that appears from nowhere; receivables from related parties.

### On the cost of goods sold line

- **Overhead absorption into inventory** — capitalise more indirect cost into each unit produced. Costs sit in inventory instead of hitting COGS. The mark: gross margin up *and* inventory days up simultaneously — real efficiency does the opposite.
- **Delaying inventory write-downs** — obsolete stock that should be written to net realisable value stays at cost. The mark: inventory growing faster than sales; a big write-down eventually, framed as one-time.
- **Supplier rebate timing** — recognising volume rebates and discounts from suppliers before they are earned, reducing COGS today. The mark: gross margin improvement that management attributes to "procurement savings" with no volume change to justify it.
- **Cost shifting between periods** — treating a payment for this year's raw material as an advance on next year's crop. The mark: prepaid assets or advances to suppliers rising sharply.

### On the operating expense lines

- **Capitalizing operating costs** — the WorldCom manoeuvre in its purest form, and the software-development version in its everyday form. An expense becomes an asset; profit rises now and the cost dribbles back as amortization over years. The mark: capitalized software or "internal-use software" on the balance sheet growing faster than revenue; capex rising while the expense line falls.
- **Stretching depreciable lives** — a change in estimate, applied prospectively, disclosed in a sentence. The mark: D&A falling as a share of gross PP&E; a note explaining a change in useful lives.
- **Pension assumptions** — for companies with defined-benefit plans, the assumed rate of return on plan assets flows directly into operating income. Raise the assumed return, lower the pension expense. The mark: an assumed return far above what the plan's asset mix could plausibly deliver.
- **Reserve releases** — warranty, returns, bad debt, legal, restructuring. Any of them can be released into income. The mark: the allowance for doubtful accounts falling as a percentage of receivables while receivables rise.
- **Reclassifying recurring costs as restructuring** — same cost, new label, excluded from adjusted earnings. The mark: restructuring charges in three or more consecutive years.

### Below the operating line

- **"Other income"** — asset sale gains, FX gains, investment gains parked here to pad pre-tax income. The mark: other income exceeding 10% of pre-tax income, or swinging wildly year to year.
- **Effective tax rate** — releasing reserves held against uncertain tax positions, or shifting income to lower-tax jurisdictions, drops the rate and lifts net income directly.
- **Share count** — buybacks shrink the denominator of EPS. A company can report falling net income and rising EPS indefinitely, for a while.

#### Worked example: the tax-rate dial on its own

Take Harbor Lantern's reported pre-tax income of \$30.0M. Suppose the tax department resolves an old dispute and releases a reserve, dropping the effective rate from 21% to 17%.

- Tax at 21%: \$30.0M × 0.21 = \$6.3M → net income \$23.7M → EPS \$0.79
- Tax at 17%: \$30.0M × 0.17 = \$5.1M → net income \$24.9M → EPS \$0.83

The business did not change by one cup of coffee. EPS rose \$0.04, or 5%. This is why the effective tax rate belongs on your checklist: a four-point move in the rate is worth about the same as a five-percent move in operating income, and it appears in a footnote rather than a headline.

**The intuition: by the time you reach the tax line, you have passed six independent opportunities to manufacture a few cents of EPS, and a manager only needs to find enough of them to clear the bar.**

## Why they do it: the map of incentives

Manipulation is not usually a heist. It is far more often a series of small, rationalised accommodations made under pressure. Understanding *what* the pressure is tells you *when* to look hardest.

**Bonus targets.** The most direct incentive there is. If the annual bonus pays out at operating income of \$40.0M and the company is running at \$25.0M in November, the finance department knows the number it needs and knows to the dollar what each dial is worth. In the Nortel case, the SEC alleged the company released roughly \$500 million of excess reserves in the first two quarters of 2003 to fabricate a return to profitability — which allowed it to pay tens of millions of dollars in "return to profitability" bonuses, largely to a select group of senior managers.

**Debt covenants.** Loan agreements typically require the borrower to keep ratios inside limits: net debt to EBITDA below some multiple, EBITDA to interest above some multiple. Breaching a covenant can trigger a repricing, a forced repayment, or default. Since covenants are usually written against EBITDA, and EBITDA excludes D&A, the dials that matter most to a covenant are the ones that lift operating income *above* the D&A line. This is why Harbor Lantern's covenant tension mattered: it made the pull-forward more valuable than the depreciation change.

**Analyst consensus.** Public companies are graded quarterly against a consensus EPS number. Missing it, even by a cent, produces a share-price reaction wildly out of proportion to the economics — because the market reads the miss as information about *control* and *honesty*, not just about the quarter. Which produces a very specific statistical fingerprint.

![A histogram of EPS surprise versus consensus showing an abnormal dip at minus one cent and a spike at zero and plus one cent — too few small misses, too many small beats.](/imgs/blogs/reading-the-income-statement-and-the-quality-of-earnings-7.webp)

If earnings were unmanaged, the distribution of EPS surprises around consensus would be roughly smooth. It is not. Across large samples of company-quarters there are far *fewer* small misses and far *more* small beats than chance allows — a visible dip just below zero and a spike just above it. This kink was documented by Burgstahler and Dichev (1997) for earnings decreases and losses, and by Degeorge, Patel and Zeckhauser (1999) for analyst thresholds specifically. The shape is not a fact about how businesses perform. It is the shape of an incentive.

**Equity compensation and the vesting calendar.** When a large tranche of options or restricted stock vests on a share price or an EPS condition, the incentive to hit that condition is measured in personal millions.

**Transactions.** An IPO prospectus, an acquisition where the buyer pays in stock, an earn-out that pays the sellers if the acquired business hits a profit target — every one of these puts a specific number in front of a specific person on a specific date.

**Simple career survival.** A division head who misses the plan three quarters running is replaced. The costs of a small accommodation land on the company; the benefits land on the individual. That asymmetry, more than greed, is what makes the first step easy.

Two questions turn this from a list into a tool. **Who benefits from this exact number?** And **what would have happened if it had come in 5% lower?** If the answer to the second is "a covenant breach", "no bonus", or "a broken acquisition", you are looking at a statement that had a reason to be flattering.

## How to detect it: a checklist you can run in twenty minutes

None of these tests proves anything. Each is a screen: a cheap question whose answer either costs nothing or tells you where to dig. Three of them lighting up together on the same company is the signal.

![A numbered eight-step earnings-quality checklist with the trigger threshold for each test, from cash conversion below 1.0x to EPS landing on consensus every quarter.](/imgs/blogs/reading-the-income-statement-and-the-quality-of-earnings-9.webp)

**1. Cash conversion.** Operating cash flow ÷ net income. Red flag below 1.0 for two consecutive years. Then repeat with free cash flow (CFO − capex) ÷ net income, because that version survives capitalization games.

**2. Days sales outstanding.** DSO = (accounts receivable ÷ revenue) × 365. Red flag if it rises more than 10% year on year without an explained change in customer mix or terms.

**3. Inventory days against gross margin.** Inventory days = (inventory ÷ COGS) × 365. Red flag if inventory days *and* gross margin rise together — genuine efficiency should reduce inventory, not build it.

**4. The adjusted-to-GAAP gap.** (Adjusted EPS − GAAP EPS) ÷ GAAP EPS. Red flag above 20%. Then read the reconciliation and ask which exclusions would still be there in five years.

**5. Restructuring frequency.** Count the years with a restructuring or "transformation" charge. Red flag at three or more consecutive years. Sum them and compare the total to cumulative GAAP operating income.

**6. Effective tax rate.** Income tax expense ÷ pre-tax income. Red flag on a decline of more than two or three points with no statutory rate change and no explanation in the tax footnote.

**7. The "other income" share.** Other income ÷ pre-tax income. Red flag above 10%, or on a swing from small to large.

**8. The consensus pattern.** Plot reported EPS against consensus for twelve quarters. Red flag if it lands exactly on, or one cent above, consensus almost every time. Real businesses are noisier than that.

#### Worked example: running the checklist on Harbor Lantern

An outside analyst who has never seen the "as it happened" column applies the eight tests to Harbor Lantern's reported 2025 statements. Hypothetical figures, in millions where marked.

Assume the balance sheet shows opening receivables of \$60.0 against prior-year revenue of \$470.0, and closing receivables of \$80.0.

**Test 1 — cash conversion.** CFO \$30.7 ÷ net income \$23.7 = **1.30x**. Above 1.0. *Pass, on the face of it.* But free cash flow \$4.7 ÷ \$23.7 = **0.20x**, against 0.66x last year. **Flag.**

**Test 2 — DSO.**
- Prior year: (60.0 ÷ 470.0) × 365 = **46.6 days**
- Reported year: (80.0 ÷ 512.0) × 365 = **57.0 days**
- An increase of 10.4 days, or 22%. At \$512.0M of revenue, one day of sales is \$1.4M, so 10.4 extra days is roughly \$15M of receivables that would not exist at last year's collection pace. **Flag, loudly.**

**Test 3 — inventory days against gross margin.** Inventory fell and gross margin was flat. *Pass* — and this is the honest limit of the screen. Channel stuffing empties the warehouse, so it makes the inventory test look *good*. No single test catches everything.

**Test 4 — the adjusted gap.** (\$1.16 − \$0.79) ÷ \$0.79 = **47%**. **Flag.**

**Test 5 — restructuring frequency.** Charges in 2021, 2022, 2023, 2024 and 2025 — five consecutive years, \$41M in total against \$40.0M of GAAP operating income this year. **Flag.**

**Test 6 — effective tax rate.** \$6.3 ÷ \$30.0 = 21.0%, unchanged. *Pass.*

**Test 7 — other income.** Zero. *Pass.*

**Test 8 — the consensus pattern.** \$0.79 reported against \$0.78 expected. One data point proves nothing; pull the last twelve quarters. If ten of them landed on or one cent above, **flag**.

Score: four clear flags out of eight, and the two loudest — free cash conversion collapsing and DSO jumping 10.4 days — both point at receivables. That is enough to go read the revenue recognition note, the segment disclosures, and the subsequent-events section, and to ask management a specific question on the call: *what were your fourth-quarter payment terms relative to the prior year?*

**The intuition: the checklist does not tell you a company is lying. It tells you which twenty pages of the 10-K to read carefully, which is the entire value proposition.**

## Common misconceptions

**"Net income is the number that matters."** Net income is the *most-quoted* number and the *least informative* one, because it has absorbed every judgment made above it, plus the capital structure, plus the tax position. For understanding the business, operating income is better; for understanding whether the profit is real, free cash flow is better; for understanding the business model, gross margin is better.

**"Non-GAAP earnings are fake earnings."** Too simple. Excluding a genuine one-time gain on selling a factory gives you a cleaner picture of the ongoing business, and analysts would compute something like it themselves if the company did not. What makes an adjustment illegitimate is not that it exists but that it *recurs*, or that it excludes a real economic cost like stock compensation. Read the reconciliation and ask one question per line: will this still be here in five years? If yes, it is an operating expense.

**"If the auditors signed off, the numbers are right."** An audit opinion says the statements are *fairly presented in all material respects in accordance with GAAP*. It is not a certification that the estimates were the best ones available, that the business is healthy, or that no fraud occurred. Most of the manipulations in this article live comfortably inside GAAP. In the Sunbeam case the SEC also charged the Arthur Andersen engagement partner on the audits — the sign-off had happened.

**"Rising revenue means a healthy company."** Revenue is the easiest line to inflate and the least connected to value. Revenue that arrives with a 120-day payment term, a generous return right, and a discount that destroys the margin is worse than no revenue: it consumes working capital, borrows from next quarter, and can come back as a return. Always read revenue growth next to gross profit growth and receivables growth.

**"A cash flow statement can't be manipulated."** It is much harder to manipulate than the income statement, which is why we use it as the referee — but it is not immune. Capitalizing operating costs shifts an outflow from operating to investing, which flatters CFO. Stretching payables, factoring receivables, and classifying items between the operating and financing sections all move the CFO line. Free cash flow, and a look at the *quality* of working capital changes, is the more robust check.

**"Small, technical accounting choices can't matter much."** WorldCom's entire fraud was one reclassification decision, repeated. No fictitious customers, no forged contracts — a category change on costs the company genuinely incurred. It produced roughly \$3.8 billion of fake profit in the first disclosure and grew to around \$11 billion, and it was enough to sustain the appearance of profitability through the collapse of the telecom market.

## How it shows up in real markets

Eight cases, each isolating one mechanism from this article. All figures are from regulatory filings and enforcement documents, with sources listed at the end.

### WorldCom, 2002: capitalizing an operating expense

WorldCom leased capacity on other carriers' networks and paid "line costs" for it. These are as operating as an expense gets — the recurring cost of the thing the company sells. Beginning in 2001, WorldCom instead recorded roughly **\$3.8 billion** of those costs as capital expenditure, transferring them to asset accounts on the balance sheet.

The economics did not change by a cent. The same cash left the company for the same leases. But an operating expense reduces this quarter's profit in full, while a capitalized asset reduces profit only through depreciation, spread across years. The reclassification let WorldCom go on reporting profits through periods in which, accounted for correctly, it was losing money.

![WorldCom's line costs of \$3.8 billion shown twice: recorded inside operating expenses producing an operating loss, versus moved onto the balance sheet as property, plant and equipment with only a sliver of depreciation returning each year, producing an operating profit.](/imgs/blogs/reading-the-income-statement-and-the-quality-of-earnings-8.webp)

WorldCom announced the restatement on 25 June 2002 — the largest ever at the time. The SEC filed its civil action the following day, charging a fraud of more than \$3.8 billion. As the investigation continued, the total reached roughly **\$11 billion**. The company filed for bankruptcy in July 2002, then the largest in US history.

The detection lesson is precise: capitalized costs show up as capital expenditure. Free cash flow — operating cash flow minus capex — is unchanged by the reclassification. A reader watching FCF instead of net income would have seen no improvement at all.

### Sunbeam, 1996–1998: the big bath and the cookie jar

Sunbeam is the textbook case for the two-step manoeuvre, and the SEC's May 2001 action against former CEO Albert Dunlap and five others lays it out. The scheme began at year-end 1996 with the creation of inappropriate accounting reserves that *increased* Sunbeam's reported 1996 loss — including **\$18.7 million** of 1996 restructuring costs that the SEC said management knew or was reckless in not knowing did not conform to GAAP.

Those reserves were then released into 1997, manufacturing the appearance of a rapid turnaround. Sunbeam also used contingent sales and improper **bill-and-hold** arrangements — most famously selling barbecue grills to retailers in the autumn, months before the grilling season, with the grills remaining in Sunbeam's warehouses. According to the SEC, **at least \$60 million of Sunbeam's reported \$189 million in 1997 earnings from continuing operations before income taxes was the result of accounting fraud.**

The detection lesson: a large loss in a new CEO's first year is a moment to read the reserve disclosures with particular care, and a "turnaround" whose profit growth outpaces its cash flow growth deserves the cash conversion test before the celebration.

### Nortel Networks, 2000–2003: reserves released for bonuses

The SEC's 2007 case against Nortel is the clearest documented link between reserve manipulation and executive pay. The Commission alleged that Nortel engaged in accounting fraud from 2000 through 2003 to close the gap between its true performance, its internal targets, and Wall Street expectations — and that by the time it announced fiscal 2002 results, it was improperly maintaining **over \$400 million** in excess reserves.

In the first and second quarters of 2003, Nortel improperly released approximately **\$500 million** of those excess reserves to boost earnings and fabricate a return to profitability. The inflated earnings allowed the company to pay tens of millions of dollars in "return to profitability" bonuses, largely to a select group of senior managers. Nortel settled by paying a **\$35 million** civil penalty.

The detection lesson: read the reserve roll-forward tables, which most annual reports disclose. A reserve balance falling sharply in a quarter when earnings surprised to the upside is not a coincidence, and it is disclosed.

### Under Armour, 2015–2016: pulling orders forward

This is dial 1, in a real company, disclosed by the SEC in detail — which makes it the best available check on whether the Harbor Lantern example is realistic.

An unusually warm winter in 2015 hurt sales of Under Armour's higher-priced cold-weather apparel. Rather than miss its revenue guidance, the company asked customers who had requested future shipment dates to take their orders early. The SEC found that Under Armour accelerated, or "pulled forward", a total of approximately **\$408 million** in existing orders across **six consecutive quarters**, from the third quarter of 2015 through the fourth quarter of 2016, while making positive public statements about its revenue growth rate and the factors driving it — without disclosing the impact of the pull-forward practice.

The charge was a *disclosure* failure rather than a recognition failure: the revenue was arguably earned when the goods shipped. That distinction is the whole lesson. You do not need to break a recognition rule to mislead. You need only to borrow from the next quarter repeatedly and let readers assume the growth was organic. Under Armour agreed to pay **\$9 million** on 3 May 2021, neither admitting nor denying the findings.

The detection lesson: pull-forwards leave two marks. Receivables and DSO climb because the accelerated orders are not paid faster, and the *following* period is weak because its demand has already been consumed. A revenue line growing faster than receivables can support, followed by a soft quarter management attributes to "timing", is the pattern.

### Kraft Heinz, 2015–2018: manufacturing a lower COGS

Between the last quarter of 2015 and the end of 2018, Kraft Heinz engaged in what the SEC described as a years-long expense management scheme, recognising unearned discounts from suppliers and maintaining false and misleading supplier contracts — the effect of which was to improperly reduce cost of goods sold and report inflated cost savings.

Kraft restated its financials in June 2019, correcting **\$208 million** of improperly recognised cost savings across nearly 300 transactions. On 3 September 2021 the company agreed to pay a **\$62 million** civil penalty; the SEC also charged its former Chief Operating Officer and former Chief Procurement Officer.

The detection lesson: this one *does* show up in gross margin, which is the point. Procurement-driven gross margin expansion with no volume or price change behind it is the specific pattern to challenge — and it is one of the very few manipulations that leaves a mark on the most-watched line.

### Diamond Foods, 2010–2011: moving a cost into next year

Diamond Foods bought walnuts from growers. The SEC alleged that former CFO Steven Neil directed an effort to underreport the money paid to those growers by delaying the recording of payments into later fiscal periods — treating a "momentum payment" for the 2010 crop as an advance on a crop to be delivered in autumn 2011, rather than as a cost of the walnuts already sold.

Lower recorded walnut costs meant lower COGS, higher gross profit, and higher EPS — enough to exceed analyst estimates in fiscal 2010 and 2011. In internal emails, according to the SEC, Neil referred to commodity costs as a "lever" to manage earnings. Diamond settled for **\$5 million** in January 2014; former CEO Michael Mendes paid a \$125,000 penalty and forfeited more than \$4 million in bonuses. The stock fell from a 2011 high of about \$90 to roughly \$17.

The detection lesson: watch prepaid assets and advances to suppliers. A cost that has been pushed into next year has to be sitting somewhere on this year's balance sheet, and it usually sits there under an innocuous name.

### Groupon, 2011–2012: the metric that had to go

Before its IPO, Groupon's filings led with a non-GAAP metric it called **ACSOI** — adjusted consolidated segment operating income — which excluded online marketing expense on the theory that customer acquisition spending was an investment rather than a cost of doing business. For a company whose entire model was buying customers with marketing, this excluded the main cost of the business. Following SEC scrutiny, Groupon removed ACSOI from its amended S-1 in August 2011.

The story did not end there. In March 2012, Groupon revised its reported fourth-quarter and full-year 2011 results, increasing its refund reserve to better reflect a shift in deal mix toward higher-priced offers with higher refund rates. The revision cut fourth-quarter revenue by **\$14.3 million**, reduced operating income by **\$30.0 million**, reduced net income by **\$22.6 million**, and cut EPS by **\$0.04**.

The detection lesson: when a company invents its own headline metric, the first question is always *what does this exclude, and would the business exist without it?* If the excluded item is the company's largest operating cost, the metric is marketing.

### Luckin Coffee, 2019–2020: fabricating both sides

The most extreme case, and instructive precisely because of how it had to be constructed. The SEC alleged that from at least April 2019 through January 2020, Luckin Coffee intentionally fabricated more than **\$300 million** in retail sales using related parties to create false transactions through three separate purchasing schemes.

But fake revenue creates a hole: cash that should have arrived did not. So employees also **inflated the company's expenses by more than \$190 million**, along with creating a fake operations database and altering accounting and bank records. That second act is the tell for the whole category — you cannot fabricate revenue on the income statement without doing something about the cash flow statement, and the something is always visible if you look. Luckin agreed to pay a **\$180 million** penalty on 16 December 2020.

### Toshiba, 2008–2014: estimates on long contracts

A non-US example, because the mechanism is universal. Following an internal report to Japan's Securities and Exchange Surveillance Commission in February 2015 and an investigation by a third-party committee, Toshiba announced a reduction of **¥224.8 billion** in its pre-tax income for the period from April 2008 through December 2014.

A significant portion came from the social infrastructure division's use of the **percentage-of-completion** method, where revenue on long-term contracts is recognised in proportion to costs incurred against total expected costs. Understating expected total costs accelerates revenue recognition — an estimate, not an invention, and one that unwinds only when the project finishes.

The detection lesson: for any business with long-term contracts, watch unbilled receivables (contract assets). Revenue recognised but not yet billed is revenue recognised on the strength of an estimate.

## When this matters to you

You do not need to be a forensic accountant for this to be useful. Three situations bring it into contact with an ordinary life.

**If you own individual stocks**, the cash conversion test takes ninety seconds and is the highest-value ninety seconds in the whole exercise. Pull net income from the income statement, operating cash flow and capex from the cash flow statement, and compute both ratios for the last three years. A company whose profit grows while free cash flow does not is telling you something the headline does not.

**If you own funds or index products**, the aggregate matters. When 72% of the S&P 500 reports an adjusted figure and those figures exceed GAAP by \$271 billion in a single year, the "market P/E" quoted in the press is often computed on adjusted earnings. The index is cheaper or dearer than it looks depending on which E you use.

**If you work inside a company**, this is the part nobody tells you: the first step down this road almost never looks like fraud from the inside. It looks like a reasonable accounting judgment made under pressure, and the person making it usually intends to reverse it next quarter when things recover. The Sunbeam, Nortel and Diamond Foods cases all began with a defensible-sounding estimate. The escalation is structural — pulling profit forward creates a hole that has to be filled next period, which is why these schemes grow rather than quietly end. That mechanism is worked through in detail in [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits).

Next in this series, the same treatment applied to the other two statements: what companies hide on [the balance sheet](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here), and why [cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) when the two disagree.

This is educational material about how financial statements work, not investment advice.

## Sources & further reading

**Enforcement actions and primary filings**

- SEC Litigation Release No. 17588, *SEC v. WorldCom, Inc.* — the \$3.8 billion capitalization of line costs. [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-17588)
- SEC Litigation Release No. 17001 and Press Release 2001-49 (15 May 2001), *SEC v. Albert J. Dunlap, Russell A. Kersh, et al.* — the Sunbeam reserves, bill-and-hold sales, and the \$60 million of \$189 million figure. [sec.gov](https://www.sec.gov/news/press/2001-49.txt)
- SEC Litigation Release No. 20333 and Press Release 2007-217 (15 October 2007), *SEC v. Nortel Networks Corporation* — the \$400 million of excess reserves, the \$500 million release, and the \$35 million penalty. [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-20333)
- SEC Press Release 2021-174 (3 September 2021), *SEC Charges The Kraft Heinz Company and Two Former Executives* — the \$208 million restatement and \$62 million penalty. [sec.gov](https://www.sec.gov/newsroom/press-releases/2021-174)
- SEC Press Release 2021-78 (3 May 2021), *SEC Charges Under Armour Inc. With Disclosure Failures* — the \$408 million of pulled-forward orders across six quarters and the \$9 million penalty. [sec.gov](https://www.sec.gov/newsroom/press-releases/2021-78)
- SEC Press Release 2014-4 (9 January 2014), *SEC Charges Diamond Foods and Two Former Executives* — the walnut "momentum payments" and the \$5 million settlement. [sec.gov](https://www.sec.gov/newsroom/press-releases/2014-4)
- SEC Litigation Release No. 24987 (16 December 2020), *SEC v. Luckin Coffee Inc.* — the \$300 million of fabricated sales, \$190 million of inflated expenses, and \$180 million penalty. [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-24987)
- Groupon, Inc., "Groupon Announces Revised Fourth Quarter and Full Year 2011 Results" (30 March 2012) — the \$14.3 million revenue revision and \$22.6 million net income reduction. [investor.groupon.com](https://investor.groupon.com/press-releases/press-release-details/2012/Groupon-Announces-Revised-Fourth-Quarter-and-Full-Year-2011-Results-Confirms-First-Quarter-Guidance/default.aspx)
- Toshiba Corporation third-party committee investigation, July 2015 — the ¥224.8 billion pre-tax income reduction covering April 2008 to December 2014.

**Regulatory and conceptual**

- Arthur Levitt, "The Numbers Game," remarks at the NYU Center for Law and Business, 28 September 1998 — the SEC chairman's enumeration of big-bath charges, creative acquisition accounting, cookie jar reserves, materiality abuse and premature revenue recognition.
- FASB ASC 606 / IFRS 15, *Revenue from Contracts with Customers* — the five-step recognition model.
- SEC Regulation G and Item 10(e) of Regulation S-K — the rules governing non-GAAP financial measures and the required GAAP reconciliation.

**Data and academic**

- Calcbench and Suffolk University, non-GAAP reporting study, 2025 — 361 S&P 500 companies (72%) reported adjusted net income or EPS for fiscal 2025, exceeding GAAP net income in aggregate by \$271 billion, with 87% adjusting upward.
- Audit Analytics, "Trends in Non-GAAP Disclosures" — the rise in S&P 500 non-GAAP usage from 59% in 1996 to 96% in 2016.
- Burgstahler, D. and Dichev, I. (1997), "Earnings management to avoid earnings decreases and losses," *Journal of Accounting and Economics* 24(1), 99–126.
- Degeorge, F., Patel, J. and Zeckhauser, R. (1999), "Earnings Management to Exceed Thresholds," *Journal of Business* 72(1), 1–33.
</content>
