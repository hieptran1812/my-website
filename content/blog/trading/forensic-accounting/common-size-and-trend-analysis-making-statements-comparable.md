---
title: "Common-size and trend analysis: making statements comparable"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "Learn how to turn financial statements into comparable percentages and indexed trends, then use the drift to find receivables, margins, expenses, and cash-flow questions worth investigating."
tags: ["forensic-accounting", "financial-statements", "common-size-analysis", "trend-analysis", "financial-ratios", "earnings-quality", "accounting"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 28
---

> [!important]
> **TL;DR** — Common-size statements answer “what share of the base is this line?” while trend indexes answer “how fast did this line move from a fixed starting point?”
>
> - Divide every income-statement line by revenue; divide every balance-sheet line by total assets or revenue, and label the denominator.
> - A falling gross margin, receivables rising faster than sales, or an expense line that disappears from the presentation is a signal to investigate, not proof of fraud.
> - Trend analysis starts a selected base year at 100; the useful comparison is the gap between lines, not a round index by itself.
> - In Apple’s fiscal 2024 Form 10-K, net sales were $391,035 million and accounts receivable were $33,410 million; the same filing lets us compute a 2024 receivables-to-sales ratio of about 8.5%.

Financial statements are designed to report a business in its own units: dollars, euros, shares, tonnes, or whatever the company uses. That is useful for judging scale. It is awkward for comparison. A $10 million increase means something very different for a $50 million company than for a $50 billion company. Even within one company, inflation, acquisitions, divestitures, and growth can make a raw-dollar chart look more dramatic than the underlying economics.

Common-size analysis and trend analysis are two small transformations that make the first pass more honest. Common-size analysis turns a statement into percentages of a base. Trend analysis turns a starting period into an index of 100 and expresses later periods relative to it. Neither technique proves that management manipulated a number. Both help you decide which footnote, estimate, customer, contract, or journal entry deserves your attention.

![The first-pass statement screen](/imgs/blogs/common-size-and-trend-analysis-making-statements-comparable-1.webp)

The workflow is deliberately simple: start with reported statements, normalize scale, compare movement, and then investigate the most economically meaningful drift. The investigation is where forensic accounting begins. The percentage is only the map.

## Foundations: the building blocks

### What a financial statement is measuring

An income statement covers a period. It answers what the company earned and spent during, for example, a fiscal year. Revenue is the amount recognized from providing goods or services under the company’s accounting rules. Cost of revenue is the cost assigned to those goods or services. Gross profit is revenue less cost of revenue. Operating expenses are costs of running the business after gross profit, such as research, sales, marketing, and general administration. Net income is the bottom-line result after the relevant operating, financing, and tax items.

A balance sheet is a snapshot at a date. Assets are resources controlled by the company, such as cash, receivables, inventory, and property. Liabilities are obligations to outsiders, such as payables and debt. Equity is the residual claim after liabilities. The accounting identity is:

\[
\text{Assets} = \text{Liabilities} + \text{Equity}.
\]

A cash-flow statement explains how cash changed through operating, investing, and financing activities. It is not simply a second income statement. Profit includes accruals: revenue can be recognized before cash arrives, and an expense can be recognized before the supplier is paid. The cash-flow statement is therefore a useful cross-check on whether reported earnings are converting into cash.

### The denominator is the argument

“Common-size” does not mean “divide everything by the same number.” The denominator should match the question.

For an income statement, revenue is usually the cleanest denominator:

\[
\text{Common-size percentage} = \frac{\text{line item}}{\text{revenue}} \times 100\%.
\]

Revenue is set to 100%. Cost of revenue as a percentage of revenue is the inverse of the gross-margin contribution: if cost is 60% of revenue, gross margin is 40%.

For a balance sheet, total assets are a natural denominator when asking about asset composition:

\[
\text{Asset mix percentage} = \frac{\text{asset line}}{\text{total assets}} \times 100\%.
\]

Revenue can also be a useful denominator for working-capital intensity. Receivables divided by revenue asks how much balance-sheet funding is tied up per dollar of annual sales. It is not a textbook “margin,” and it does not have to stay constant; seasonality and payment terms matter.

For a cash-flow cross-check, operating cash flow divided by net income is often called the cash-conversion ratio. It can be above or below 100% in an ordinary year. A sustained gap needs a bridge: receivables, inventory, payables, tax timing, stock compensation, and other non-cash items can all move it.

### Journal entries are the hidden machinery

Statements are summaries of ledger activity. A basic credit sale produces an illustrative entry:

```journal
Dr Accounts receivable       $100
    Cr Revenue                         $100
```

When the customer pays:

```journal
Dr Cash                       $100
    Cr Accounts receivable             $100
```

The first entry increases both revenue and receivables. The second converts the receivable into cash without creating new revenue. That is why comparing revenue growth with receivables growth is informative: if sales rise but collections do not keep pace, the company may be granting longer credit, selling to weaker customers, recognizing sales near period-end, or simply experiencing a timing difference. The ratio cannot tell you which explanation is true.

#### Worked example: a one-year income statement

Suppose a fictional retailer reports revenue of $100, cost of revenue of $60, operating expenses of $25, and net income of $11. These are illustrative numbers chosen to show the arithmetic, not a claim about a real company.

1. Revenue is the base: $100 / $100 = 100%.
2. Cost of revenue is $60 / $100 = 60%.
3. Gross profit is $100 − $60 = $40, so gross margin is $40 / $100 = 40%.
4. Operating expenses are $25 / $100 = 25%.
5. Net margin is $11 / $100 = 11%.

The compressed common-size statement is therefore: revenue 100%, cost 60%, gross profit 40%, operating expenses 25%, and net income 11%. The intuition is that the statement now describes the economics of one sales dollar rather than the size of the retailer.

## 1. Common-size income statements: see the margin drift

Raw dollars answer “how many dollars?” Common-size percentages answer “how many cents of each sales dollar?” That second question is often more useful when a company is growing.

Imagine two periods with revenue of $100 and $120. Cost of revenue rises from $60 to $78. In dollars, sales increased by $20 and cost increased by $18. The company still grew gross profit from $40 to $42. A superficial read might say: sales and gross profit are both higher.

The common-size view says cost moved from 60% of sales to 65% of sales. Gross margin moved from 40% to 35%. Gross profit increased by $2, but the business kept only 35 cents of each new sales dollar as gross profit, compared with 40 cents before. If operating expenses stay at $25 in both periods, net income could fall even while revenue grows.

![Same growth, different economics](/imgs/blogs/common-size-and-trend-analysis-making-statements-comparable-2.webp)

The causes can be mundane: a greater share of low-margin products, a temporary freight shock, discounting, warranty costs, foreign exchange, or an acquisition whose margins differ. They can also be accounting questions: revenue may have been reclassified, costs may have moved between cost of revenue and operating expenses, or a one-time charge may have been excluded from a non-GAAP presentation. The chart tells you where to read; it does not choose the explanation.

#### Worked example: decomposing margin movement

Use the same fictional numbers: period one revenue $100, cost $60; period two revenue $120, cost $78.

1. Period-one gross margin: ($100 − $60) / $100 = 40%.
2. Period-two gross margin: ($120 − $78) / $120 = 35%.
3. Gross profit change: ($120 − $78) − ($100 − $60) = $42 − $40 = $2.
4. Revenue growth: ($120 − $100) / $100 = 20%.
5. Gross-profit growth: ($42 − $40) / $40 = 5%.

Revenue grew 20% while gross profit grew only 5%. That spread is the economic content hidden inside the raw totals. The intuition is that growth in sales is not the same thing as growth in profit capacity.

### Expense lines that vanish

An expense “disappearing” from the face of a statement is not automatically a missing expense. Companies can change presentation, combine immaterial lines, move a cost into cost of revenue, or change segment disclosure. The correct first response is a reconciliation.

Build a three-column bridge: prior label, current label, and note or policy explanation. If “restructuring” is absent this year, search the current-year and prior-year comparative notes. If “stock-based compensation” moves from a separate line into research and development, total operating expenses may be unchanged even though the visible mix changed. If the total does not reconcile, the gap is a higher-priority question.

An analyst should distinguish presentation drift from economic drift. Presentation drift changes where a cost appears. Economic drift changes how many dollars the company spends. A common-size statement can surface both, but only the notes can tell you whether the comparatives were recast.

### The denominator can mislead

Revenue is not always a stable economic base. A bank’s interest income and interest expense need a different analysis from a manufacturer’s sales and cost of goods. A marketplace may report gross bookings, net revenue, or both. A subscription business can grow billings ahead of recognized revenue. A company with a major acquisition may have a step-change in revenue and expense lines that makes a year-over-year percentage look like a trend when it is really a perimeter change.

Record the denominator beside every ratio. “R&D is 8%” is incomplete. “R&D is 8% of reported revenue in fiscal 2024” is auditable. Also record whether the line is reported, adjusted, or your own recomputation.

## 2. Trend analysis: make the base year 100

Trend analysis answers a different question. Instead of asking what share of a period’s base a line represents, ask how much it has changed relative to a selected starting period.

The standard formula is:

\[
\text{Trend index}_t = \frac{\text{value}_t}{\text{value}_{\text{base}}} \times 100.
\]

The base period is 100. An index of 140 means the line is 1.4 times its base-year value, or 40% above it. An index of 85 means it is 15% below the base. The index is not a percentage margin and it is not a forecast.

![Trend analysis turns a base year into 100](/imgs/blogs/common-size-and-trend-analysis-making-statements-comparable-5.webp)

The valuable pattern is relative movement. Suppose a company’s sales index moves 100, 112, 125, 140 over four years. Receivables move 100, 130, 165, 215. An expense moves 100, 108, 118, 129. Sales grew steadily; the expense grew more slowly; receivables accelerated. The screen should direct attention to collection, credit terms, customer concentration, allowances, and period-end revenue—not declare that the company fabricated sales.

#### Worked example: computing a trend index

Suppose illustrative revenue is $80 in the base year and $100 three years later. Receivables are $8 in the base year and $18 later.

1. Revenue trend index: $100 / $80 × 100 = 125.
2. Receivables trend index: $18 / $8 × 100 = 225.
3. Revenue growth from the base: ($100 − $80) / $80 = 25%.
4. Receivables growth from the base: ($18 − $8) / $8 = 125%.

Receivables did not merely grow “more.” They grew five times as fast in percentage terms: 125% versus 25%. The intuition is that a trend index makes a small but economically important base line visible.

### Choosing the base year

Choose a base that is economically interpretable, not merely convenient. A normal pre-acquisition year is often better than a pandemic-disrupted year. If the base line is zero or negative, the index can be meaningless or unstable; use dollar changes, margins, or a different base instead. If the business has changed through a divestiture, use recast comparatives when available and state the perimeter.

Never compare indexes without checking units. If one year is in thousands and another is in millions, the arithmetic can produce a plausible-looking but useless series. Keep the source unit in the working paper and convert once.

### Common-size and trend analysis work together

The two views are strongest as a pair. A line can grow quickly but become less important as a share of revenue. Or it can grow slowly in dollars but consume a larger share of a shrinking revenue base.

For example, illustrative revenue may rise from $100 to $120 while advertising expense rises from $10 to $15. The expense trend index is 150. Its common-size percentage is 10% in the first period and 12.5% in the second. Both views say the expense is becoming more prominent. If revenue instead falls from $100 to $80 while advertising stays at $10, the expense trend index is 100 but its common-size percentage rises from 10% to 12.5%. A flat dollar line can still become a margin problem.

## 3. Balance-sheet common-size analysis: follow the asset mix

Income statements show flows over time. Balance sheets show stocks at a date. A common-size balance sheet turns the asset side into a composition question: how much of the asset pool is cash, receivables, inventory, property, or “other”?

![A balance-sheet common-size view](/imgs/blogs/common-size-and-trend-analysis-making-statements-comparable-4.webp)

The same approach applies to liabilities and equity. Debt as a percentage of total assets is a capital-structure mix measure. Payables as a percentage of revenue or cost can give a working-capital signal. Retained earnings as a percentage of equity may help explain changes in the capital base, but accumulated losses, buybacks, and other comprehensive income can make simple interpretations unsafe.

#### Worked example: receivables intensity

Suppose an illustrative company reports revenue of $200 and $240 in two years. Receivables are $20 and $36.

1. Year-one receivables-to-sales ratio: $20 / $200 = 10%.
2. Year-two ratio: $36 / $240 = 15%.
3. Revenue growth: ($240 − $200) / $200 = 20%.
4. Receivables growth: ($36 − $20) / $20 = 80%.

Receivables rose four times as fast as revenue in percentage terms, and the receivables-to-sales ratio increased five percentage points. The intuition is that the company may be financing more of its sales through customers, so reported growth is demanding more cash support.

This ratio is not days sales outstanding. A rough DSO calculation uses average receivables divided by revenue multiplied by the number of days in the period. If average receivables were $28 on revenue of $240, a 365-day illustrative DSO would be $28 / $240 × 365 ≈ 42.6 days. Use average, not just ending, receivables when seasonality matters. State whether you used 365 or 360 days, and do not compare a quarter’s ending balance with a full year’s revenue without adjusting the basis.

### The working-capital bridge

Receivables are one part of the operating cash bridge. In the indirect cash-flow method, an increase in receivables generally reduces operating cash flow because revenue has been recognized without the cash collection. An increase in inventory also generally consumes cash; an increase in payables generally supplies cash temporarily.

The word “generally” matters. Acquisitions, foreign exchange, reclassifications, factoring, and non-cash transfers can make the statement presentation more complicated. Read the cash-flow note and the accounting policy before building a conclusion from one line.

![Receivables growing faster than sales](/imgs/blogs/common-size-and-trend-analysis-making-statements-comparable-6.webp)

If receivables-to-sales rises, branch the investigation. Ask whether customers are paying later, the company changed credit terms, a distributor bought inventory before period-end, returns increased, an allowance is too low, or revenue cut-off is wrong. Then test the branches against aging reports, subsequent cash receipts, credit memos, customer contracts, and the allowance roll-forward.

#### Worked example: a cash-conversion warning

Suppose illustrative net income is $30 and operating cash flow is $12. The cash-conversion ratio is $12 / $30 = 40%. That is not automatically bad: a company may have made a large inventory build for a launch, paid a prior-year payable, or recognized a non-cash gain. But it is a prompt to reconcile.

Assume the bridge shows a $14 increase in receivables, a $6 increase in inventory, and a $5 increase in payables. The combined working-capital effect is approximately −$14 − $6 + $5 = −$15 before considering other adjustments. A $30 accounting profit and $12 operating cash flow are now less mysterious, but the analyst still has to ask whether the receivables are collectible and whether the inventory will sell.

The intuition is that earnings quality is a reconciliation problem, not a single ratio threshold.

## 4. A forensic first-pass screen

The best screen is repeatable. It should produce the same set of questions when two analysts use the same filings. Avoid turning the process into a hunt for a dramatic percentage.

![Read the drift, then ask why](/imgs/blogs/common-size-and-trend-analysis-making-statements-comparable-3.webp)

### Pass one: normalize the statements

Create a worksheet with periods in columns and these calculated rows:

| Area | Calculated view | What it asks |
| --- | --- | --- |
| Income statement | Every line / revenue | What does one sales dollar cost? |
| Balance sheet | Every asset or liability / total assets | What changed in the composition? |
| Working capital | Receivables, inventory, payables / revenue or cost | Who is financing the cycle? |
| Cash flow | Operating cash flow / net income | Is profit converting to cash? |
| Multi-year trend | Current line / base-year line × 100 | Which line outran the business? |

Use formulas rather than hand-typing percentages. Round only in the display layer; keep full precision in the worksheet so a two-point movement is not created by premature rounding.

### Pass two: rank the drift

Rank changes by economic significance. A one-percentage-point change in a large cost line can matter more than a 20% change in a small line. Consider absolute dollars, percentage-point change, percentage growth, and recurrence. A single restructuring charge may be large but non-recurring. A small annual increase in an allowance shortfall can be more important if it repeats.

Pay attention to opposing signals. Sales rise, but operating cash falls. Gross margin falls, but adjusted margin rises. Receivables rise, but the allowance percentage falls. Debt falls, but interest expense rises because the company refinanced at a higher rate. Opposing movements are not proof of manipulation; they are good prompts for note-reading.

### Pass three: trace to documents

For every red or amber signal, write a one-sentence hypothesis and one piece of evidence that could falsify it. “Receivables rose because customers paid later” can be tested with subsequent receipts and aging. “Gross margin fell because product mix changed” can be tested with segment or product disclosures. “An expense vanished because it was capitalized” can be tested with the accounting policy, additions to property, and amortization.

The evidence ladder is: filed statement, note disclosure, management discussion, accounting policy, ledger or journal entry, and external corroboration. Public investors usually cannot access the ledger, but the order still helps. Do not jump from a ratio directly to an allegation.

![The analyst's evidence ladder](/imgs/blogs/common-size-and-trend-analysis-making-statements-comparable-7.webp)

### Pass four: examine journal-entry direction

Common-size analysis becomes forensic when you ask which entry could create the observed movement. A credit sale increases revenue and receivables. A write-off decreases receivables and allowance. A capitalization entry can move a cost from the income statement to an asset, postponing expense recognition. A reserve release can reduce expense and increase profit without a cash inflow.

For an illustrative reserve release:

```journal
Dr Allowance for doubtful accounts     $5
    Cr Bad-debt expense                           $5
```

That entry reduces the allowance and expense. It does not collect a customer’s debt. If receivables rise while the allowance falls as a percentage of receivables, inspect the aging and the reserve methodology. There may be a sound reason, but the entry direction explains why a margin can improve without better cash conversion.

### Pass five: test cut-off

Cut-off means recording a transaction in the correct reporting period. A sale recorded on December 31 rather than January 2 increases the current period’s revenue and receivables, assuming the recognition criteria are met only later. The next period may show a return, credit memo, or collection that changes the story.

The first-pass test is not “find a suspicious invoice.” It is “compare the last days of the period with the first days of the next period.” Review shipping terms, delivery evidence, invoices, credit notes, unusual discounts, and subsequent cash. A common-size spike near year-end may be seasonal and legitimate, so compare with prior years and peers before escalating.

## 5. Mechanics and edge cases behind the screen

### Seasonality changes the meaning of an ending balance

An annual income statement adds twelve months of activity. A balance sheet gives you one date. If a retailer sells most of its goods in the final quarter, the year-end receivables and inventory balances may be unusually high even when the business is healthy. If a construction company bills by milestone, the year-end contract-asset balance may move with project timing rather than a sudden deterioration in collection.

The practical fix is to increase the frequency of the comparison. Use quarterly revenue with quarterly ending receivables, or use average beginning-and-ending receivables with a full-period revenue denominator. If the business is seasonal, compare the same quarter with the same quarter in prior years. A December balance compared with a March balance can manufacture a trend that disappears when the calendar is aligned.

#### Worked example: ending balance versus average balance

Suppose an illustrative company has beginning receivables of $20, ending receivables of $40, and annual revenue of $240. The ending-balance ratio is $40 / $240 = 16.7%. Average receivables are ($20 + $40) / $2 = $30, so the average-balance ratio is $30 / $240 = 12.5%. Both are valid descriptions of different questions. The first asks how much was outstanding at the reporting date; the second approximates the amount tied up during the year. The intuition is that a point-in-time ratio and a period-average ratio should not be treated as interchangeable.

### Acquisitions and divestitures break naive trends

When a company buys a business, revenue and expenses may jump because the reporting perimeter is larger. A trend index can correctly show a 160 reading while still failing to tell you whether the original business grew. Ask for organic growth, pro forma comparatives, acquisition dates, purchase accounting, and discontinued-operations treatment. Do not silently splice a pre-acquisition standalone figure into a post-acquisition consolidated series.

The same issue appears in common-size analysis. An acquired business may have a different gross margin, sales commission model, depreciation policy, or working-capital cycle. The consolidated gross margin can move even when neither legacy business changed its pricing. Segment information and the acquisition note help separate mix from execution.

### Currency translation can look like operating growth

Multinational companies often report in one presentation currency while earning revenue in several local currencies. A stronger presentation currency can reduce translated revenue and receivables even when local-currency activity grows. A weaker presentation currency can make reported lines grow without the same change in local operations.

Record whether the filing provides constant-currency information, and do not mix a translated income-statement flow with an unadjusted balance-sheet interpretation. Foreign-exchange movements can also change the relationship between beginning and ending balances. The answer is not to discard the ratio; it is to tag the source of movement before calling it an operating signal.

### Reclassifications create false breaks in the series

Companies sometimes change line-item presentation and recast prior periods. If the filing says prior-year amounts were reclassified to conform to the current presentation, use the recast numbers. If no recast is available, preserve both the original label and the current label in your worksheet. A line that “disappears” may have moved into another line, and the right test is whether the total expense bridge reconciles.

Forensic work benefits from a mapping table:

| Prior label | Current label | Amount reconciled? | Evidence to read |
| --- | --- | --- | --- |
| Selling expense | Sales and marketing | Yes or no | Expense note |
| Other operating cost | Cost of revenue | Yes or no | Accounting policy |
| Restructuring | Included in operating expense | Yes or no | Reconciliation |
| Contract cost asset | Deferred contract costs | Yes or no | Revenue note |

If the labels change but the total and policy reconcile, it is usually a presentation question. If the labels change and the total cannot be tied, elevate the issue.

### Materiality is a filter, not permission to ignore patterns

Materiality depends on size, nature, context, and the needs of users. A small amount can matter if it changes a covenant, turns a loss into profit, affects executive compensation, or masks a recurring control failure. A large amount can be less informative if it is a clearly disclosed one-time disposal.

Use two filters. First, quantify the effect on revenue, gross profit, operating income, net income, assets, and cash. Second, consider qualitative sensitivity: unusual timing, related parties, management estimates, and entries posted manually near close. Common-size analysis helps with the first filter. Footnotes and controls address the second.

### Peer comparisons need an accounting dictionary

Comparing a company with peers is useful only after defining each line. One company may include shipping in cost of revenue; another may include it in operating expense. One may report a contract liability separately; another may include it in deferred revenue. One may capitalize development costs while another expenses them.

Build a dictionary for revenue, cost of revenue, operating income, adjusted EBITDA, receivables, contract assets, inventory, and operating cash flow. Quote the filing’s terminology in the worksheet. If a peer comparison requires an adjustment, show the bridge rather than pretending the published percentages are directly comparable.

### The screen should produce a ranked queue

Do not create fifty red flags and call the process complete. Rank the queue by cash exposure, recurrence, judgment, and reversibility. A receivable that is large, overdue, and concentrated in one customer should rank above a small research expense reclassification. A margin change that repeats for four periods should rank above a one-quarter weather disruption, unless the weather item conceals a control issue.

For each item, record the signal, possible explanations, evidence requested, owner, and status. This turns analysis into a review trail. It also makes it easier to close a false positive: “seasonal; same quarter prior year; subsequent receipts normal” is a better outcome than leaving an unexplained red cell on a dashboard.

### What a clean result looks like

A clean result does not mean every ratio is flat. Healthy businesses change. It means the important changes can be explained, reconciled, and supported by documents. A rising receivables ratio may be justified by a disclosed shift to enterprise customers with contractual payment terms. A falling gross margin may be explained by an investment cycle or product launch. A lower operating-cash ratio may reflect a disclosed inventory build that later sells.

The analyst’s job is to make the explanation testable. Ask for the next document, not the most dramatic accusation. If the explanation survives the evidence, close the item with a date and source. If it does not, quantify the possible effect and escalate according to the review mandate.

## 6. Named case study: Apple’s fiscal 2024 filing

Apple Inc. is a useful named case because its 2024 Form 10-K presents three years of consolidated statements and clear units. The figures below are reported in millions of U.S. dollars and come from Apple’s SEC filing for the fiscal year ended September 28, 2024. Apple reported net sales of $391,035 million in 2024, $383,285 million in 2023, and $394,328 million in 2022. The filing reported gross margin of $180,683 million, $169,148 million, and $170,782 million, respectively; research and development expense of $31,370 million, $29,915 million, and $26,251 million; and net income of $93,736 million, $96,995 million, and $99,803 million. These are company-reported figures, not estimates. [Apple’s 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm)

#### Worked example: Apple’s common-size income statement

Compute the following from the reported lines.

1. 2024 gross margin: $180,683 / $391,035 ≈ 46.2%.
2. 2023 gross margin: $169,148 / $383,285 ≈ 44.1%.
3. 2022 gross margin: $170,782 / $394,328 ≈ 43.3%.
4. 2024 net margin: $93,736 / $391,035 ≈ 24.0%.
5. 2023 net margin: $96,995 / $383,285 ≈ 25.3%.
6. 2022 net margin: $99,803 / $394,328 ≈ 25.3%.

The common-size read is specific: reported gross margin dollars rose from 2023 to 2024 and the gross-margin percentage also improved, while net income and net margin fell from 2023. That combination tells an analyst to look below gross profit: operating expenses, other income or expense, tax, share count, and the company’s explanations. It does not support the claim that Apple’s overall profitability improved merely because gross margin improved.

Research and development was $31,370 million in 2024, which is about 8.0% of net sales; it was about 7.8% in 2023 and about 6.7% in 2022. The line grew in dollars and consumed a larger share of sales. That is not inherently negative: a company may be investing in products or services. It is exactly the kind of drift the common-size view makes legible.

The case also shows why a first pass is not a verdict. Apple’s business mix, product cycle, services mix, foreign exchange, and capital-allocation choices affect these ratios. A ratio screen should lead to the filing’s product and services disclosures, expense descriptions, and cash-flow statement.

#### Worked example: Apple’s receivables-to-sales ratio

Apple reported accounts receivable, net of $33,410 million at September 28, 2024, $29,508 million at September 30, 2023, and $28,184 million at September 24, 2022. Dividing by the corresponding annual net sales gives:

1. 2024: $33,410 / $391,035 ≈ 8.5%.
2. 2023: $29,508 / $383,285 ≈ 7.7%.
3. 2022: $28,184 / $394,328 ≈ 7.1%.

The ending receivables-to-sales ratio increased across these three fiscal year-ends. That is a signal to examine payment timing, customer and distributor terms, seasonality, allowance disclosures, and operating cash flow. It is not evidence that Apple’s revenue was fabricated. A ratio based on an ending balance can be distorted by the fiscal-year date, so an analyst would improve the test with average receivables and quarterly data.

The filing also reports total assets of $364,980 million at the 2024 year-end, which provides the denominator for an asset-mix view. Accounts receivable were about 9.2% of total assets ($33,410 / $364,980). The choice of denominator changes the question: 8.5% describes receivables relative to annual sales; 9.2% describes receivables inside the year-end asset pool.

## 6. What the ratios cannot tell you

### A ratio is not an allegation

The same movement can have multiple causes. Receivables can rise because a large customer paid late, because the company grew rapidly, because the reporting date fell before a normal collection cycle, because terms changed, or because cut-off is wrong. The ratio only narrows the search.

### Percentages can hide scale

A tiny line can double from $1 to $2 and still be immaterial. A large line can move from 10% to 11% and add a substantial cost. Always pair the percentage-point movement with dollar movement and the materiality of the line.

### Trend indexes can be unstable

If the base year is unusually small, the index can look explosive. If the base is negative, a higher number can produce an apparently lower index even when the economics improved. Use a normal base, disclose the base, and retain the raw-dollar series alongside the index.

### Common-size statements are not automatically comparable across companies

Different accounting policies, business models, fiscal calendars, segment mix, capitalization choices, and revenue presentation can make two percentages look comparable when they are not. A marketplace’s net revenue is not the same economic base as a manufacturer’s gross sales. Read definitions before ranking companies.

### Non-GAAP presentations can blur the line

Adjusted expenses and adjusted margins may be useful, but they are not the same as the reported statement. Keep a reported common-size analysis and an adjusted bridge. If a cost is excluded from adjusted profit, show the reconciliation and ask whether it recurs.

## How it shows up in real markets

### Apple: gross-margin improvement alongside lower net income

Apple’s fiscal 2024 filing illustrates why analysts need both common-size and raw-dollar views. The SEC filing reported net sales of $391,035 million, gross margin of $180,683 million, and net income of $93,736 million for the year ended September 28, 2024. The prior-year figures were $383,285 million, $169,148 million, and $96,995 million. Gross margin therefore improved as a share of sales from roughly 44.1% to 46.2%, while net margin declined from roughly 25.3% to 24.0%. [SEC filing, fiscal 2024](https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm)

The useful lesson is not a simplistic “Apple got better” or “Apple got worse.” Different layers moved in different directions. An analyst would reconcile the change through operating expenses, other income, taxes, and share count, then read the company’s segment and product disclosures. A gross-margin trend can be real and still fail to translate into higher net income. In a market, that distinction matters because investors price future earnings and cash flows, not one attractive ratio.

### WorldCom: ratio analysis must meet the ledger

WorldCom is a historical named case of alleged accounting fraud, and it is important to phrase the claim carefully. The U.S. Securities and Exchange Commission’s complaint alleged that WorldCom improperly capitalized line costs, moving expenses from the income statement into property and equipment. Capitalization can make current operating expenses look lower and assets look higher than they would under the alleged proper treatment. [SEC complaint against WorldCom](https://www.sec.gov/litigation/complaints/complr17588.htm)

The common-size lesson is a disappearing or shrinking expense line. The forensic lesson is stronger: trace the line to journal entries and the capitalization policy. An analyst who sees an unusually favorable margin should ask whether the expense was paid, capitalized, reclassified, or excluded from an adjusted metric. The ratio creates the question; the ledger and evidence establish what happened.

### A fictional distributor: sales grow, cash does not

Consider an illustrative distributor with sales of $200 and $240, receivables of $20 and $36, and operating cash flow that falls from $28 to $12. The sales trend index is 120; receivables are at 180; the receivables-to-sales ratio rises from 10% to 15%. No one number proves a problem, but all three views point to the same working-capital question.

The analyst would request an aging report, inspect cash received after year-end, compare credit terms, review returns and credit memos, and test large invoices around the reporting date. If subsequent cash is strong and the growth is concentrated in customers with longer contractual terms, the explanation may be ordinary. If the aging deteriorates and credit memos follow year-end, the risk is higher. The evidence changes the conclusion.

### A fictional software company: disappearing implementation expense

Suppose a software company reports revenue of $100 and implementation expense of $12 in year one. In year two, revenue is $130 and the expense line is absent, while deferred contract costs and capitalized assets rise. A common-size screen shows the expense moving from 12% to zero, but that is not the end of the analysis.

The analyst should locate the accounting policy, ask whether implementation work creates a controlled resource or merely helps fulfill a contract, and reconcile amortization. A legitimate capitalization policy can shift expense over time. An aggressive policy can defer a current cost without sufficient future benefit. The important point is to connect presentation drift to the entry that created it.

## When this matters to you

For an investor, this screen is a way to spend limited attention. Start with the lines that can change cash, margins, or the durability of growth. For a lender, receivables, inventory, payables, and debt mix can affect liquidity before a covenant is breached. For an operator, the same analysis can show whether pricing, product mix, hiring, and collection discipline are moving in the intended direction.

Use it before reading a long earnings deck, not instead of reading the filing. Keep reported and adjusted numbers separate. Date every real number, retain the source link, and state the denominator. If a ratio looks alarming, seek an explanation that could be wrong and then test it.

> A ratio is a smoke alarm, not a fire report.

This is educational analysis, not investment, accounting, audit, or legal advice. Financial statements can be complex, and a conclusion about fraud requires much more evidence than a percentage table.

## Sources & further reading

- [Apple Inc. fiscal 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm), SEC EDGAR, filed November 1, 2024; statements for the fiscal year ended September 28, 2024.
- [Apple investor-relations SEC filing record](https://investor.apple.com/investor-relations/sec-filings/sec-filings-details/default.aspx?FilingId=17933082), Apple Inc., 2024 Form 10-K filing record.
- [SEC complaint: WorldCom, Inc.](https://www.sec.gov/litigation/complaints/complr17588.htm), U.S. Securities and Exchange Commission, June 2002; allegation and accounting-treatment source for the historical case discussion.
- [SEC Investor Bulletin: How to read a company’s financial statements](https://www.investor.gov/introduction-investing/investing-basics/glossary/financial-statements), U.S. Securities and Exchange Commission, accessed August 4, 2026.
- [Financial statement analysis](https://www.investor.gov/introduction-investing/investing-basics/glossary/financial-statements), SEC Investor.gov glossary and educational material.
