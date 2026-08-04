---
title: "Non-GAAP and adjusted EBITDA: the metrics companies invent"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "A forensic guide to GAAP net income, EBITDA, legitimate adjustments, recurring costs disguised as one-offs, and WeWork's infamous Community Adjusted EBITDA."
tags: ["non-gaap", "adjusted-ebitda", "gaap", "earnings-quality", "wework", "forensic-accounting", "financial-statements", "investor-analysis", "one-off-costs", "cash-flow"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Adjusted EBITDA can make a useful operating question visible, but it can also hide the cost of running the business. Treat it as a reconciliation to investigate, never as a replacement for GAAP net income or cash flow.
>
> - GAAP net income is the standardized bottom line after operating costs, interest, taxes, depreciation and amortization. EBITDA starts from that bottom line and removes four items; adjusted EBITDA then adds a company-specific layer of judgment.
> - A legitimate adjustment is narrow, clearly defined, material, and genuinely unusual. A recurring payroll bill, ordinary rent, routine stock compensation, or the cost of opening the next location is not made “free” by being labelled one-time.
> - The fastest forensic test is repetition: if the same category appears every year, it is part of the business model even when each individual invoice is different.
> - WeWork’s 2018 filing reported a **\$1.927 billion net loss** and **\$467.125 million Community Adjusted EBITDA** for the same year. Both figures were in the filing; the gap was the story.
> - The number to remember: adjusted EBITDA is a bridge. Read every plank between GAAP and the adjusted destination, then ask which planks require cash, recur, or belong to the business’s normal capacity.

Why can a company report a loss of almost two billion dollars and a profit of almost half a billion dollars in the same year? The answer is not necessarily fraud. It is that “profit” has several definitions, and the further a metric travels from a standardized accounting statement, the more the author chooses what to remove.

That choice can be useful. A retailer opening a new country may want to show the earnings power of stores that have reached a steady state. A software company may separate a genuine acquisition bill from the cost of serving its existing customers. A lender may want a covenant measure that approximates the cash available before financing decisions. Those are reasonable questions.

The danger begins when the answer to a hard question is made to look like the answer to an easier one. “What would this mature location earn?” becomes “what did the company earn?” “This restructuring charge is unusual” becomes “all restructuring charges are irrelevant.” “The employee was paid in shares” becomes “the compensation cost was free.” The vocabulary gets softer while the cash leaving the company stays hard.

The diagram below is the mental model for this article. Start with the standardized statement, move through EBITDA, and then follow each company-specific adjustment separately. A reconciliation is not a single number; it is a set of claims about what the business is and is not.

![A forensic bridge from GAAP net income to EBITDA and then adjusted EBITDA, with each company-specific add-back separated for testing.](/imgs/blogs/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent-1.webp)

The rest of the article builds that bridge from first principles, tests the arithmetic with hypothetical statements, and then applies the method to a real filing: WeWork’s 2018 Community Adjusted EBITDA.

## Foundations: the building blocks

### GAAP is the common starting language

**GAAP**, or generally accepted accounting principles, is the accounting framework used for a company’s standardized financial statements in the United States. It determines how the company recognizes revenue, measures assets and liabilities, and presents expenses. GAAP does not make every estimate objective. Inventory reserves, useful lives, lease assumptions, impairment tests and expected credit losses still require judgment. Its value is that the rules are defined, disclosed and applied across the statement in a way that makes companies more comparable than a private vocabulary would.

The income statement is a film of one period. It begins with revenue and subtracts the resources consumed to produce that revenue. It ends with **net income**, the residual attributable to the owners after operating costs, financing costs and income taxes. When the residual is negative, it is a net loss.

The balance sheet is a snapshot of what the company owns and owes at a date. The cash flow statement tracks cash moving through operations, investing and financing. Net income and cash are connected, but they are not synonyms. A company can recognize revenue before collecting it, expense a non-cash impairment, pay rent for a future period, or buy equipment with cash while recording no immediate full expense. That is why an adjusted earnings metric cannot be evaluated without the other two statements.

### EBITDA removes four categories, not every difficult cost

**EBITDA** means earnings before interest, taxes, depreciation and amortization. In its common construction, one starts with net income and adds back:

$$\text{EBITDA} = \text{Net income} + \text{Interest} + \text{Income tax} + \text{Depreciation} + \text{Amortization}$$

Interest is removed to compare businesses with different debt choices. Tax is removed to compare businesses with different jurisdictions or tax attributes. Depreciation and amortization are removed because they allocate the cost of a long-lived asset across periods rather than recording a new cash payment in every period.

That last explanation needs care. Depreciation is non-cash *in the current period*, but it represents the consumption of an asset. If a delivery van cost \$100,000 and lasts five years, the company cannot avoid eventually replacing it merely because the annual depreciation entry has no cash line. EBITDA therefore says “before this allocation,” not “cash available to spend.” The SEC describes EBITDA as a non-GAAP measure and requires a reconciliation to the most directly comparable GAAP measure when it is publicly disclosed. See the [SEC’s Regulation G guidance](https://www.sec.gov/rules-regulations/staff-guidance/corporation-finance-interpretations/non-gaap-financial-measures).

### Adjusted EBITDA is a company-defined second layer

**Adjusted EBITDA** takes EBITDA and removes additional items chosen by management. There is no single universal list. Common adjustments include stock-based compensation, acquisition expenses, restructuring charges, impairment, litigation, foreign-exchange movements, pre-opening expenses, integration costs and sometimes “other” items.

The word *adjusted* is therefore incomplete until the footnote supplies the noun. Adjusted EBITDA at a subscription software company may exclude stock compensation and hosting migration costs. Adjusted EBITDA at a hotel company may exclude hurricane damage. Adjusted EBITDA at a flexible-office company may exclude costs at newly opened locations. These measures are not automatically comparable even when the labels match.

| Measure | What it asks | Standardization | Forensic question |
| --- | --- | --- | --- |
| GAAP net income | What residual profit did the accounting framework report? | Highest of these three | Are revenue, expenses, assets and liabilities recognized appropriately? |
| EBITDA | What did operations show before financing, tax and D&A? | A familiar convention, but non-GAAP | Is D&A a meaningful cost of the business? |
| Adjusted EBITDA | What would performance look like after management’s selected exclusions? | Company-specific | Do the adjustments recur, require cash, or remove core capacity? |

> A label can describe a calculation. It cannot repeal an expense.

### The reconciliation is the primary evidence

When a company presents adjusted EBITDA, look for a table that starts at net income or loss and walks to the adjusted figure. The table should name every adjustment, explain it, and show the arithmetic. The SEC’s rules require a quantitative reconciliation for historical non-GAAP measures and prohibit presentations that are misleading or give undue prominence to the non-GAAP number. The [SEC’s 2003 release on Regulation G](https://www.sec.gov/rules-regulations/2003/03/conditions-use-non-gaap-financial-measures) is the original rulemaking source.

The reconciliation is more useful than the headline because it exposes the distance between the standardized number and the management number. A \$50 million adjustment may be harmless in a \$10 billion business and decisive in a \$100 million business. A non-cash charge may still measure a resource consumed. A charge called “one-time” may be the tenth version of the same annual event.

#### Worked example: the shortest honest bridge

Suppose Harbor Lantern Coffee Co. reports the following hypothetical annual statement, in millions of dollars:

| Statement line | Amount |
| --- | ---: |
| Revenue | \$100.0 |
| Operating expenses including D&A | \$(92.0) |
| Operating income | \$8.0 |
| Interest expense | \$(2.0) |
| Income tax | \$(1.2) |
| GAAP net income | \$4.8 |

Within the \$92.0 million of operating expenses is \$3.0 million of depreciation and amortization. EBITDA is therefore \$4.8M + \$2.0M + \$1.2M + \$3.0M = **\$11.0M**.

Management also reports \$0.5M of clearly identified acquisition-adviser fees for a transaction that closed this year. Adjusted EBITDA is \$11.0M + \$0.5M = **\$11.5M**.

The adjustment does not say Harbor Lantern generated \$11.5M of cash. It says one narrow cost is excluded from a performance lens. The cash flow statement still shows the adviser being paid, and the acquisition may create future integration costs.

**Intuition:** the farther right the bridge goes, the more important it is to inspect each labelled plank rather than admire the final total.

## 1. What legitimate add-backs are trying to isolate

The best adjusted metrics answer a question that GAAP net income was not designed to answer. They are not “better earnings” in the abstract; they are a measurement lens. Before accepting an add-back, state the lens in one sentence.

If the question is “how much recurring cash can service debt?”, adding back routine cash payroll is usually absurd. If the question is “what did the mature stores produce before corporate overhead?”, excluding corporate overhead may be useful for unit economics, but it cannot be presented as company-wide profit. If the question is “what did the continuing business earn after a factory closure?”, a genuinely nonrecurring closure cost might be separated, while the continuing factory payroll cannot.

### Four properties of a defensible adjustment

First, it has a **clear boundary**. “Restructuring” should identify the program, affected employees or facilities, and the period in which it is expected to finish. “Other operating costs” is not a boundary; it is an invitation to search.

Second, it is **unusual relative to the company’s own history**, not merely unusual relative to an idealized future. A cost can be large and still ordinary. A retailer’s holiday marketing may be seasonal. A technology company’s annual employee grant may be non-cash but routine.

Third, it is **separately useful for the stated purpose**. Acquisition fees can be removed to compare pre-acquisition operations, but not if acquisitions are the company’s growth engine and happen every quarter. Pre-opening costs can be useful in a mature-store analysis, but the company cannot open locations without paying them.

Fourth, it is **transparent enough to reverse**. The reader should be able to start at GAAP net income, reproduce the bridge, and decide whether to accept or reject an item. If the company changes the definition every period, the metric is not a stable measuring instrument.

![A matrix separating legitimate, debatable and abusive add-backs by recurrence and connection to normal capacity.](/imgs/blogs/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent-2.webp)

### Non-cash does not mean costless

The most common conceptual error is to equate “non-cash” with “not economic.” Stock-based compensation illustrates the problem. The journal entry may be non-cash on the grant date, but employees receive an economic claim on the company. Existing shareholders may be diluted, or the company may need to repurchase shares to offset dilution. Calling the expense non-cash can be useful when reconciling accounting profit to a narrow operating measure; it is misleading when the reader hears “the company did not pay for labor.”

Impairment is another example. An impairment charge can be a non-cash revision to the carrying value of an asset. It may not reduce this month’s bank balance, but it records that the company’s past spending did not produce the expected value. Repeated impairment is evidence about capital allocation, not merely noise.

### Cash, capacity and comparability are different tests

Every proposed adjustment should be tested along three axes:

| Test | Question | A “yes” means |
| --- | --- | --- |
| Cash test | Did cash leave, or will it leave soon? | Do not call the adjusted number cash flow. |
| Capacity test | Does the cost maintain or create the capacity that produces revenue? | Treat it as part of the business economics. |
| Comparability test | Does removing it make periods or peers more comparable? | It may have analytical value, if disclosed narrowly. |

The same item can pass one test and fail another. Depreciation may fail the cash-this-period test but pass the comparability test. Stock compensation may be useful to exclude when comparing cash operating margins, but still fail the shareholder-economics test. A restructuring payment may be both cash and nonrecurring; it can be excluded from a normalized period while remaining relevant to liquidity.

#### Worked example: the same \$6.0M adjustment under three lenses

Imagine a hypothetical delivery company reports \$20.0M of EBITDA. It also has:

1. \$2.0M of stock-based compensation;
2. \$1.0M of a one-time legal settlement from a discontinued product; and
3. \$3.0M of annual driver hiring and training costs.

An “all adjustments” presentation would produce \$26.0M: \$20.0M + \$2.0M + \$1.0M + \$3.0M. But the capacity test rejects the hiring and training cost: drivers are required to deliver the product. A cautious normalized operating view might be \$23.0M, excluding stock compensation and the discontinued-product settlement while retaining driver costs. A cash-preservation view might retain the legal payment too, because \$1.0M actually left the bank this year, yielding \$22.0M.

None of these is the one true answer. The error would be showing \$26.0M without explaining the different economic questions.

**Intuition:** normalization is not a hunt for the largest number of add-backs; it is a choice of measurement purpose followed by consistent exclusions.

## 2. The recurring-cost trap

“One-time” is a description of management’s hope unless the history supports it. The forensic test is not whether this invoice happened once. Every invoice happens once. The test is whether the underlying economic activity repeats.

Companies often re-label recurring costs in one of five ways:

- **Restructuring:** annual layoffs, site closures or reorganization programs that recur because the business model keeps changing.
- **Integration:** acquisition-related consulting that appears after every acquisition in a serial-acquirer strategy.
- **Pre-opening:** opening costs that recur whenever the company pursues growth.
- **Transformation:** technology, brand or “efficiency” programs that become permanent operating work.
- **Other:** a residual bucket that grows when named categories become embarrassing.

The forensic reader builds a five-year schedule by category, not by the company’s chosen adjective. If “one-time” costs are \$4M, \$9M, \$6M, \$11M and \$8M, the line is not one-time in any economically meaningful sense. The exact annual values in that illustration are hypothetical; the method is not.

![A recurrence test that rolls annual “one-time” adjustments into a five-year pattern and routes repeated costs back into normal earnings.](/imgs/blogs/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent-3.webp)

### Journals reveal what the label hides

Forensic accounting is not only reading the earnings release. It is asking what entry the adjustment represents. Consider a hypothetical \$10.0M facility closure:

```journal
Dr Restructuring expense                 $10.0M
    Cr Cash / severance payable                       $6.0M
    Cr Lease termination liability                    $4.0M
```

The entry is a real cost. If the facility is genuinely closed and no similar program is expected, excluding it from a *post-closure* run-rate analysis may be reasonable. But the \$6.0M cash payment belongs in liquidity analysis, and the \$4.0M lease liability belongs on the balance sheet until settled. The adjusted EBITDA table does not erase either.

Now consider a hypothetical annual transformation program:

```journal
Dr Transformation expense                $10.0M
    Cr Cash / accrued vendors                         $10.0M
```

If the same type of program appears in four consecutive years, the label is not evidence of nonrecurrence. It is evidence that management treats transformation as a recurring way of operating.

### The “two-year” idea is a warning, not a permission slip

SEC staff guidance discusses the prohibition on adjusting a performance measure to eliminate an item identified as nonrecurring, infrequent or unusual when the nature of the charge is reasonably likely to recur, including when a similar charge occurred in the prior two years. This is not a universal accounting definition of “one-time,” and it does not mean a charge becomes acceptable on the third year. It is a regulatory guardrail against misleading presentation, not a formula for laundering recurring costs.

#### Worked example: recurring restructuring changes the valuation question

Suppose a hypothetical company has \$30.0M of reported EBITDA and excludes \$5.0M of “restructuring” each year. Its revenue is \$150.0M.

Reported EBITDA margin is \$30.0M / \$150.0M = **20.0%**. Adjusted EBITDA margin is \$(30.0M + \$5.0M) / \$150.0M = **23.33%**.

If the charge happens once, a reader may reasonably model a 23.33% steady-state margin after the program ends. If it happens every year, a repeatable margin is 20.0% unless there is concrete evidence that the next program will stop. On a hypothetical 10× EBITDA multiple, the difference is \$50.0M of headline enterprise value: \$5.0M × 10. That is not a prediction or a market statistic; it is the arithmetic consequence of trusting or rejecting the add-back.

**Intuition:** recurring “one-offs” do not improve normalized earnings; they reveal the recurring cost of keeping the business in its chosen shape.

## 3. WeWork and Community Adjusted EBITDA

WeWork is a useful case because the company’s own filing lets us see both the standardized loss and the invented metric. This section reports what WeWork disclosed; calling the metric aggressive is an analytical judgment, not a claim that the filing secretly changed GAAP.

### The 2018 starting point

In its 2019 registration statement, WeWork Companies Inc. reported revenue of **\$1.821751 billion** for the year ended 31 December 2018, a **net loss of \$1.927419 billion**, and a **net loss attributable to WeWork Companies Inc. of \$1.610792 billion**. The same filing’s key performance table reported **Adjusted EBITDA of negative \$665.653 million** and **Community Adjusted EBITDA of \$467.125 million**. The figures are from the [SEC-hosted filing](https://www.sec.gov/Archives/edgar/data/1533523/000162827919000125/filename1.htm); the filing’s tables present dollar amounts in thousands.

The contrast is not a rounding difference. The filing reported Community Adjusted EBITDA margin of **27.5%** for 2018, compared with an Adjusted EBITDA margin of **negative 36.5%**. Community Adjusted EBITDA was not GAAP net income and was not ordinary EBITDA. It was a further operating lens focused on the economics WeWork associated with its locations and community-level activity.

![WeWork’s 2018 filing figures: revenue \$1.821751B, net loss \$(1.927419)B, adjusted EBITDA \$(665.653)M, and Community Adjusted EBITDA \$467.125M.](/imgs/blogs/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent-4.webp)

### Reconstructing the bridge

The filing’s reconciliation gives the reader the right order of operations: begin with net loss, add back the conventional EBITDA categories, then add the company-specific adjustments that lead to Adjusted EBITDA and Community Adjusted EBITDA. The most important conceptual point is that Community Adjusted EBITDA was not a claim that WeWork had \$467.125M of free cash. It was a claim about a selected layer of the business after excluding costs management associated with corporate, growth and community-level operations.

The filing also reported that in 2018 WeWork had **401,000 memberships**, **466,000 desks**, and **425 facilities**. Those operational numbers help explain the appeal of a location-level lens: the company was expanding a physical network and wanted readers to see what established sites might produce. But expansion is not free. Lease commitments, build-outs, staff, utilities, marketing and corporate support are not optional simply because a site’s local contribution is positive.

The question is therefore not “was the metric invented?” It plainly was a company-defined metric. The question is “what decision does it improve?” It may help a reader compare the contribution of locations at different maturity stages. It does not answer whether the consolidated company covered its leases, central staff, financing costs and growth investments.

### Why the name matters

The word *community* shifts attention from the consolidated entity to a selected operating layer. That can be analytically valid, just as a restaurant chain studies four-wall store profit before headquarters costs. But a four-wall number is not a company profit number. The analyst must put the excluded central and growth costs back when valuing the whole firm or assessing solvency.

The historical context makes the distinction material. WeWork’s 2018 filing was not a mature, asset-light software filing; it described a rapidly expanding office-leasing and subleasing platform. The business signed long-term obligations and invested in locations before all the member revenue arrived. A location contribution metric could be positive while the consolidated company’s fixed commitments and expansion cash burn remained negative.

#### Worked example: reading WeWork without changing its figures

Use only the filing’s headline amounts, in millions, rounded here for readability: revenue **\$1,821.751M**, net loss **\$(1,927.419)M**, Adjusted EBITDA **\$(665.653)M**, and Community Adjusted EBITDA **\$467.125M**.

1. Net loss margin = \$(1,927.419)M / \$1,821.751M = **negative 105.8%** approximately.
2. Adjusted EBITDA margin = \$(665.653)M / \$1,821.751M = **negative 36.5%**, matching the filing’s reported margin.
3. Community Adjusted EBITDA margin = \$467.125M / \$1,821.751M = **25.64%** using the rounded headline revenue, while the filing reports **27.5%** based on the relevant revenue denominator used in its KPI presentation. This is a useful warning: do not recompute a company’s custom margin from a nearby but different revenue line and silently call it identical.

The analysis survives the denominator issue: the filing’s own Community Adjusted EBITDA is positive while its consolidated GAAP net loss and Adjusted EBITDA are negative. The metric answers a narrower question.

**Intuition:** WeWork’s positive community number may describe local contribution, but it cannot pay a corporate bill that the metric excludes.

## 4. From adjusted earnings to cash: the bridge companies cannot rename

EBITDA is often treated as a rough proxy for operating cash flow. That shortcut can be useful for a first pass, but it breaks when working capital, rent, capital expenditure, taxes, interest or restructuring payments matter.

Operating cash flow starts with net income and adjusts for non-cash items and changes in operating assets and liabilities. Investing cash flow includes purchases of property, equipment and software. Financing cash flow includes debt and equity. A company can report positive adjusted EBITDA while cash falls because it is building inventory, funding receivables, paying lease deposits, buying equipment, or settling liabilities accumulated in prior periods.

### A useful cash bridge

![The bridge from adjusted EBITDA to operating cash and then free cash, showing working capital, cash restructuring, interest, taxes and capital expenditure as distinct uses.](/imgs/blogs/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent-5.webp)

$$\text{Cash available after operations and reinvestment} \approx \text{EBITDA} - \Delta\text{working capital} - \text{cash taxes} - \text{cash interest} - \text{capital expenditure} - \text{cash restructuring}$$

There is no universal formula for free cash flow; the equation above is an explicitly explanatory bridge, not a company-reported definition:

This is an analytical bridge, not a formula claimed by every company. The symbols mean: working capital is operating receivables, inventory and payables; capital expenditure is cash spent on long-lived operating assets; cash restructuring is the payment component of a restructuring provision.

The bridge is intentionally conservative. If adjusted EBITDA adds back stock compensation, the cash bridge must separately ask what dilution or repurchase is required. If it adds back rent, the bridge must include the lease payment. If it adds back pre-opening costs, the bridge must include the cost of future growth.

#### Worked example: \$25.0M of adjusted EBITDA is not \$25.0M of cash

Suppose a hypothetical company reports \$25.0M of adjusted EBITDA. During the year:

- receivables and inventory absorb \$4.0M of cash;
- cash taxes are \$2.0M;
- cash interest is \$3.0M;
- capital expenditure is \$8.0M; and
- a previously excluded restructuring program pays \$2.0M.

The illustrative cash bridge is \$25.0M − \$4.0M − \$2.0M − \$3.0M − \$8.0M − \$2.0M = **\$6.0M**. The company can truthfully report \$25.0M of adjusted EBITDA and still have only \$6.0M left before dividends, debt repayment or acquisitions.

**Intuition:** EBITDA is a starting altitude, not a bank balance.

### Lease economics are especially important for flexible space

For a company that leases office space and sells access to that space, rent is not a peripheral corporate cost. It is the raw material of the product. A contribution metric that excludes building-level lease economics may be useful for a narrow cohort analysis, but it is dangerous as a proxy for consolidated cash generation.

This is why WeWork’s case remains instructive even if a reader never analyzes a coworking company again. The more fixed the obligations beneath the revenue, the more dangerous it is to treat a site-level margin as enterprise-level earnings. The same reasoning applies to airlines, hotels, data centers, warehouses and retailers.

## 5. Comparability: the hidden tax in peer tables

Two companies can report “Adjusted EBITDA” and mean materially different things. One may add back stock compensation but retain recurring restructuring. Another may add back restructuring but retain stock compensation. A third may use an “Adjusted EBITDA before growth” measure that removes all new-site costs. A peer multiple table that ignores definitions can create false precision.

Build a normalized peer table yourself. Put GAAP net income, operating income, EBITDA, each major adjustment, adjusted EBITDA and operating cash flow in separate columns. Then compute the adjustment burden:

$$\text{Adjustment burden} = \frac{\text{Adjusted EBITDA} - \text{EBITDA}}{\text{Revenue}}$$

This ratio is an analytical abstraction, not a GAAP metric. It makes the size of the company-defined layer visible relative to the business. Track it over time. A rising burden is not automatically bad: a genuine acquisition year can be expensive. But a rising burden with no completed program, no cash explanation and no improvement in operating cash flow deserves skepticism.

![A peer-comparison matrix showing why identical “adjusted EBITDA” labels can hide different add-back definitions and different cash outcomes.](/imgs/blogs/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent-6.webp)

#### Worked example: same label, different economics

Imagine two hypothetical subscription companies, both with \$200.0M of revenue and \$20.0M of reported EBITDA.

Company A adds back \$2.0M of one-time acquisition fees and \$1.0M of a settled legal case. Its adjusted EBITDA is **\$23.0M**, and its adjustment burden is \$3.0M / \$200.0M = **1.5%** of revenue.

Company B adds back \$8.0M of stock compensation, \$5.0M of recurring annual transformation work and \$3.0M of “temporary” customer migration costs. Its adjusted EBITDA is **\$36.0M**, and its adjustment burden is \$16.0M / \$200.0M = **8.0%** of revenue.

A screen that ranks by adjusted EBITDA margin sees 18.0% for Company B versus 11.5% for Company A. A forensic table shows the difference is mostly the definition, not necessarily the underlying business.

**Intuition:** the comparable part of adjusted EBITDA is not the label; it is the documented bridge.

### A disciplined peer protocol

For each company, answer these questions:

1. Does the reconciliation start at net income or another GAAP subtotal?
2. Is the definition unchanged from the prior year?
3. Which adjustments are cash, and when will the cash leave?
4. Which adjustments occurred in at least two of the last three years?
5. Are stock compensation and dilution discussed separately?
6. Are the adjustments included in management bonuses or debt covenants?
7. Does operating cash flow improve with adjusted EBITDA?
8. Does the company present the GAAP metric with at least equal prominence?

The answers form an evidence record. They also help avoid an opposite mistake: rejecting every non-GAAP measure because some are abused. A transparent, stable, small reconciliation can be more informative than a GAAP number distorted by a genuinely unusual acquisition or disaster.

## 6. Detecting metric drift before it becomes a crisis

Metric drift is the slow expansion of what a company excludes. It often begins innocently. A company adds back a transaction fee. The next year it adds “integration.” The next year it adds “strategic initiatives.” By year four, the metric is a portrait of the business after removing the costs of doing business.

Look for five signals.

### The denominator stays stable while exclusions grow

If revenue grows from \$100M to \$130M but adjustments grow from \$3M to \$12M, the adjusted margin may look stable only because exclusions are absorbing the operating costs of growth. Ask whether the business is becoming more efficient or merely more adjusted.

### “Other” becomes material

An “other” line of \$0.2M is a nuisance. An “other” line of \$8.0M is a missing explanation. Request the schedule. Companies often use “other” for individually small items that collectively reveal a recurring pattern.

### The reconciliation changes without a restatement

Management can change a non-GAAP definition without changing GAAP statements. The new number may be arithmetically correct under its new definition and still not be comparable with last year. Recalculate the prior period under the new definition if the data permits.

### Adjusted EBITDA beats targets while cash misses

If compensation or debt covenants use adjusted EBITDA, incentives are visible. A business can hit a non-GAAP target by excluding a cost that still consumes cash. Compare the measure used in the bonus plan with the measure used in the investor presentation; they may differ.

### The company says “temporary” but cannot give an end date

A real project has a start, a scope, a budget and an expected completion date. “Temporary” without a finish line is a narrative, not evidence.

![A decision flow for testing an adjustment: define it, test recurrence, test cash and capacity, then accept, separately model or reject it.](/imgs/blogs/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent-7.webp)

#### Worked example: a five-year adjustment ledger

Suppose a hypothetical company reports the following “one-time” adjustments:

| Year | Restructuring | Integration | Other |
| --- | ---: | ---: | ---: |
| 2022 | \$4.0M | \$0.0M | \$1.0M |
| 2023 | \$3.0M | \$2.0M | \$1.0M |
| 2024 | \$5.0M | \$0.0M | \$2.0M |
| 2025 | \$4.0M | \$3.0M | \$2.0M |
| 2026 | \$6.0M | \$0.0M | \$3.0M |

The five-year total is \$22.0M restructuring, \$5.0M integration and \$9.0M other, or **\$36.0M** of exclusions. The pattern does not prove abuse. It does prove that “one-time” is not an adequate summary. A normalized model should at least retain an expected recurring amount and separately model the cash timing.

**Intuition:** the ledger turns adjectives into a time series, and time series are much harder to charm.

## 7. Common misconceptions

### “GAAP is always true and adjusted EBITDA is always fake”

GAAP statements can contain estimates and judgment, and a one-off event can make a single GAAP period a poor guide to future operating performance. Adjusted EBITDA can improve the view when it is narrow, stable and reconciled. The right response is not tribal loyalty to one label; it is to understand the measurement purpose and inspect the bridge.

### “Depreciation is non-cash, so it is irrelevant”

Depreciation is not a current-period cheque, but it measures the use of an asset. A business that never replaces its equipment may temporarily show strong EBITDA and weak long-term economics. Compare EBITDA with capital expenditure and the age of the asset base.

### “One-time means it happened only once”

The relevant unit is the economic activity, not the invoice. Annual reorganizations, recurring acquisitions and routine pre-opening costs can each be made of distinct invoices while representing a repeatable business process.

### “Positive adjusted EBITDA means the company is profitable”

It means positive under that company’s selected exclusions. The company may still have a GAAP operating loss, net loss, negative operating cash flow, large lease obligations or high capital expenditure. Say which definition you mean.

### “Stock compensation is free because nobody was paid cash”

Employees received value. The cost can appear as dilution, a future repurchase, or a smaller claim for existing owners. Excluding it may be useful for a cash operating view, but the shareholder-return analysis should put it back or measure dilution separately.

### “The largest adjusted EBITDA is the best company”

The largest number may simply have the broadest exclusions. Compare GAAP margins, adjustment burden, cash conversion, revenue quality and the stability of the definition. A smaller but cash-generative business can be economically stronger.

## How it shows up in real markets

### WeWork’s 2018 filing: a location lens versus a consolidated company

WeWork’s 2019 SEC registration statement is the named case study for this post because the filing places the numbers beside each other. For 2018 it reported \$1.821751 billion of revenue, a \$1.927419 billion net loss, negative \$665.653 million of Adjusted EBITDA and positive \$467.125 million of Community Adjusted EBITDA. It also reported 401,000 memberships and 425 facilities at year end. These are dated figures from the filing, not current operating statistics.

The analytical lesson is not that every location was unprofitable. A location-level contribution can be a useful management statistic. The lesson is that the metric’s scope must remain visible. Costs excluded because they occur at the building, community or corporate level still have to be funded by the consolidated company. When the company carries long-term property commitments, a local contribution margin cannot stand in for solvency.

### A standard SEC warning: reconciliation is necessary but not sufficient

The SEC’s Regulation G framework requires a company to present the most directly comparable GAAP measure and a quantitative reconciliation when it publicly discloses material non-GAAP information. It also says the presentation cannot be misleading. That is why a clean table does not end the analysis. The investor still has to ask whether the exclusions are recurring, cash, or central to the operating model.

### Serial acquirers: integration as a permanent department

Consider a real-world pattern without assigning an unsourced company figure: a serial acquirer buys a business every year and excludes transaction and integration expenses each time. Each acquisition is individually unusual, but acquisitions are not unusual to the strategy. The adjusted metric may be helpful for comparing the acquired business’s pre-deal operations, while a shareholder model should include the recurring cost of acquiring growth.

The distinction is analogous to a retailer’s “new store” metric. A new store can be loss-making while it fills. That does not make the loss irrelevant if expansion is the way the retailer grows. The proper model can show mature-store economics and the cash cost of expansion as two separate layers.

### Technology companies: stock compensation and the cash-versus-ownership split

Many technology companies present adjusted earnings that exclude stock-based compensation. The adjustment can improve a view of cash expenses in the period, especially when comparing companies with different grant policies. It does not answer what employees’ claims do to existing shareholders. Analysts should pair adjusted EBITDA with dilution, repurchase spending and the share count used for diluted EPS.

### Airlines, hotels and data centers: depreciation is a maintenance question

For asset-heavy businesses, EBITDA can look attractive because depreciation is large. That is not automatically a flaw: depreciation can make comparisons across financing structures easier. But the reader must compare capital expenditure with depreciation over a cycle. If capex persistently exceeds depreciation, the company may be growing; if it falls far below depreciation, the company may be harvesting an aging asset base. Neither conclusion follows from EBITDA alone.

### Litigation and disaster charges: unusual does not mean unimportant

A flood, lawsuit or regulatory settlement can be genuinely unusual. Excluding the cost may help compare normalized operations, but the event can still change insurance, reputation, liquidity and future legal exposure. The correct presentation is often two views: normalized earnings without the unusual charge, and the reported cash and balance-sheet consequences with it.

### Private-credit covenants: the contract can define a different EBITDA

Debt agreements frequently define “Consolidated EBITDA” or “Adjusted EBITDA” for covenant purposes. That definition is a contract, not a claim that the number is GAAP profit. It may permit negotiated add-backs for synergies, pro forma cost savings or restructuring. A lender cares about the exact covenant and baskets; a public investor cares about whether that covenant measure maps to cash available for debt service. Read the definition in the credit agreement rather than borrowing the investor-presentation definition.

## When this matters to you

You do not need to reject every adjusted metric. You need to read it in the right order.

Start with revenue and GAAP net income. Then read operating income and operating cash flow. Then calculate ordinary EBITDA if the company has not shown it. Only after that should you read adjusted EBITDA. Copy every add-back into a small ledger with four columns: amount, cash or non-cash, recurring or unusual, and operating capacity or peripheral.

For a public company, compare at least three periods. For a private company or a deal, request monthly data and the exact definition in the purchase agreement. For a lender, map each add-back to the debt-service calculation. For a shareholder, include dilution, reinvestment and lease obligations. The appropriate “adjusted” number depends on the decision, but the underlying cash and commitments do not.

This is educational analysis, not individualized investment advice. The practical discipline is simple: never let a company-defined subtotal replace the statements from which it was built.

## 8. A practical forensic workflow

The most reliable analysis is deliberately boring. It does not begin with a view about management. It begins by transcribing the table and asking the same questions every period. That makes the work reproducible and prevents a memorable case such as WeWork from becoming a shortcut for judging every company.

### Step one: preserve the reported statement

Save the exact filing, earnings release or investor presentation and record its publication date. Copy the GAAP income statement, balance sheet and cash flow statement before copying any adjusted table. Keep the units visible: a statement in thousands can make a number look three orders of magnitude smaller if it is transcribed as dollars. Record whether the period is a quarter, year-to-date period or full year. A quarter’s restructuring payment may be large and still not represent the annual run rate.

Then write down the consolidation perimeter. Does the adjusted metric include all subsidiaries, joint ventures, discontinued operations and noncontrolling interests? WeWork’s filing, for example, showed both net loss and net loss attributable to WeWork Companies Inc. Those are related but not identical denominators. A reconciliation that starts at one and ends at a metric calculated on another perimeter needs an explanation.

### Step two: turn labels into transactions

Do not copy “strategic initiatives” into your model as if it were a natural kind. Translate it into the underlying transaction: severance, consultant invoices, lease termination, software implementation, marketing campaign, litigation settlement or acquisition fee. Each has a different recurrence and cash profile.

The journal entry is a useful mental model even when the filing does not disclose the entry. For a \$3.0M consultant bill that has been paid:

```journal
Dr Operating expense                      $3.0M
    Cr Cash                                           $3.0M
```

An adjusted EBITDA table may put the \$3.0M back. The bank account does not. For a \$3.0M impairment:

```journal
Dr Impairment expense                     $3.0M
    Cr Accumulated impairment                         $3.0M
```

No cash moved on the entry date, but the asset is now worth less according to management’s own test. The two adjustments require different follow-up questions.

### Step three: build the history before accepting the adjective

Use a table with one row per adjustment category and one column per period. Keep categories stable even when management renames them. “Business optimization,” “cost transformation” and “restructuring” may be three labels for one family of activity. If a category disappears, search the footnotes for where its costs went. A metric can improve merely because a recurring expense was moved from one named bucket into “corporate costs.”

Flag any category present in two of the last three periods. That is not an automatic rejection. It is a requirement to explain why the next period will be different. Ask for a program end date and compare it with headcount, site count, product launches and acquisition activity. A company that calls growth costs one-time while increasing its growth target has not shown that the costs will end.

### Step four: reconcile to cash and obligations

A useful schedule separates the adjustment into four time buckets:

| Bucket | Example | What to do |
| --- | --- | --- |
| Paid this period | Cash severance, advisers, settlement | Include in cash analysis even if normalized out of earnings. |
| Payable later | Accrued lease exit, unpaid vendor bill | Follow the liability on the balance sheet. |
| Non-cash but dilutive | Stock compensation | Track share count and repurchases. |
| Non-cash valuation loss | Impairment | Test whether capital allocation was poor or recurring. |

Then reconcile adjusted EBITDA to operating cash flow. The signs do not have to move together in every quarter; working capital can create timing differences. But a persistent gap should have a business explanation. A growing company may consume cash in receivables and inventory. A declining company may release working capital and temporarily look cash generative. Cash conversion is a pattern, not a single-period verdict.

### Step five: compare the metric with incentives

Read the compensation discussion and the credit agreement when available. If management receives a bonus on adjusted EBITDA, list the permitted exclusions. If debt covenants permit “reasonable costs and expenses” or pro forma savings, the lender may accept a different number from a public-market analyst. That is not necessarily contradictory: the lender is negotiating a contract. It does mean a screen should not mix covenant EBITDA with investor adjusted EBITDA.

Incentives do not prove manipulation. They identify where precision matters. The same person can honestly believe a restructuring is temporary and still benefit if the adjustment makes a target easier. A good analysis separates intent from measurement quality.

### Step six: publish two numbers when two questions matter

If you are writing an investment memo, show reported earnings and a normalized view side by side. Do not hide the rejected add-backs inside a single “conservative” number. A table might show GAAP net income, reported EBITDA, management adjusted EBITDA, analyst normalized EBITDA and operating cash flow. The reader can then see where judgment entered.

#### Worked example: a complete analyst schedule

Suppose a hypothetical manufacturer reports \$12.0M of net income. Interest is \$4.0M, tax is \$3.0M, D&A is \$5.0M, stock compensation is \$2.0M, a genuine plant-fire loss is \$1.0M, annual maintenance shutdown is \$2.0M and acquisition fees are \$1.0M.

1. EBITDA = \$12.0M + \$4.0M + \$3.0M + \$5.0M = **\$24.0M**.
2. Management adjusted EBITDA, if it adds every listed item, is \$24.0M + \$2.0M + \$1.0M + \$2.0M + \$1.0M = **\$30.0M**.
3. A cautious analyst may exclude the fire loss and acquisition fee but retain stock compensation and maintenance shutdown: \$24.0M + \$1.0M + \$1.0M = **\$26.0M**.
4. A cash operating view might retain the fire payment and acquisition fee as well, depending on when paid, while separately disclosing that stock compensation is non-cash but dilutive.

The schedule does not accuse management of lying. It makes the decision visible. If the manufacturer has a plant fire every year, the \$1.0M moves back into normal earnings. If the maintenance shutdown is the only annual period in which equipment can be serviced, it is a core cost even though the factory is temporarily idle.

**Intuition:** the analyst’s job is not to discover a magic adjusted number; it is to show which economic question each number answers.

## 9. What a high-quality reconciliation looks like

A high-quality reconciliation is easy to audit with a calculator. It starts with a GAAP subtotal that is clearly named, lists the adjustments in a stable order, uses consistent signs, gives comparative periods, and explains material changes. It does not introduce a new subtotal without defining it. It does not place the adjusted number in a larger font while burying net income in a footnote. It does not call a cash expense non-cash merely because management paid it in a different period.

The best disclosures also show why management uses the measure. “We use adjusted EBITDA to evaluate operating performance and trends” is more useful when paired with a definition that has not changed. The purpose should constrain the adjustment. If the measure is for operating comparability, remove financing differences but do not erase recurring sales payroll. If the measure is for covenant capacity, show cash interest, capital expenditure and lease obligations separately.

### Reconciliation quality checklist

Use this checklist on the next earnings release:

- Is the closest GAAP metric presented first and with equal prominence?
- Are historical values reconciled quantitatively, not described only in prose?
- Does each adjustment have a plain-English explanation and a dollar amount?
- Is the same adjustment visible in the cash flow statement, balance sheet or footnotes?
- Did the company rename or regroup categories this period?
- Do “one-time” items recur in the prior two years?
- Are recurring stock grants, rent, maintenance, marketing or hiring excluded?
- Is the metric used in bonuses or debt covenants?
- Does adjusted EBITDA convert into operating cash flow over several periods?
- Are future commitments left outside the bridge?

If the answer to several questions is unknown, lower confidence in the adjusted metric rather than filling the gaps with an assumption.

### A note on mathematical precision

Financial tables often use rounded millions while the underlying filing uses thousands. Recompute ratios from the same source-level numbers when possible. If a company reports a margin, quote the company’s reported margin; if you recompute it from rounded revenue and EBITDA, label the result as approximate. The WeWork example shows why: a custom KPI may use a denominator or perimeter that is not obvious from the nearby income statement.

This is a small discipline with a large payoff. It prevents false “gotcha” claims based on rounding and prevents a company’s own percentage from being silently replaced by an analyst’s incompatible one.

## 10. The second-order effects of aggressive adjustments

Aggressive add-backs do more than make one period look better. They can alter valuation, compensation, borrowing and the company’s strategic choices.

### Valuation

A valuation multiple is a ratio. If enterprise value is held constant and adjusted EBITDA rises because exclusions widen, the reported multiple falls even if the underlying cash economics do not change. In the opposite direction, a lender may allow more debt because a covenant EBITDA is higher. The apparent capacity can then fund acquisitions or expansion, increasing the obligations that the adjusted metric excluded.

### Incentives

When bonuses use adjusted EBITDA, employees may rationally prioritize projects that qualify for add-backs. A cost can be moved into a program code, a recurring activity can be bundled into a “transformation” initiative, or a maintenance project can be delayed so the current period is cleaner. None of these possibilities proves a breach. They explain why the definition and audit trail matter.

### Strategy

If investors reward adjusted EBITDA, management may choose growth with high excluded start-up costs over profitable but less adjustable growth. That can be perfectly rational if mature economics are strong. It is dangerous when mature economics are never reached, or when the company keeps expanding the denominator while excluding the cost of expansion.

### Trust

The market can forgive a bad quarter. It is less forgiving of a metric that changes after the fact. A stable definition lets an analyst rebuild history. A moving definition forces the reader to choose between incomparable numbers and often causes management’s preferred metric to lose credibility even when some adjustments are fair.

The forensic conclusion is deliberately modest: adjusted EBITDA is neither an accounting crime nor a free pass. It is a claim about what should count for one purpose. The reader’s responsibility is to name that purpose, trace the cash and keep recurring capacity in the model.

### What to ask management

The most useful questions are concrete enough to answer with a schedule. Ask which excluded costs were paid in cash during the period, how much remains payable, and when the remaining payments are expected. Ask what percentage of the workforce, locations or revenue the adjustment touches. Ask whether the same program existed in each of the previous three years and whether a budget includes another one next year. Ask whether the company would continue spending the money if it stopped growing.

For an acquisition adjustment, ask how many acquisitions closed, how many are planned, and whether the integration team is now a permanent department. For a pre-opening adjustment, ask the number of openings, the average cost per opening and the proportion of the current revenue base represented by locations still below steady state. For stock compensation, ask for diluted shares and repurchase spending, not just the expense excluded from EBITDA. These questions turn a rhetorical “one-time” into an operating forecast.

Finally, ask what the metric would be if no adjustments were allowed. That counterfactual is not a replacement for the reported table. It is a way to see how much of the company’s story depends on permission to subtract.

### The difference between a bridge and a forecast

An adjusted EBITDA reconciliation describes the past period. A forecast applies assumptions to the future. Confusing the two creates another form of metric abuse. A company may remove \$5.0M of current restructuring expense and then forecast that the saving will appear next year. That forecast is a hypothesis. It needs evidence about headcount, rent, vendors and revenue, not just the historical add-back.

The same distinction applies to synergies. If an acquisition is expected to save \$4.0M next year, the historical reconciliation cannot simply add the \$4.0M to this year’s EBITDA unless the saving was actually achieved in the period or the measure is explicitly pro forma. Otherwise the table combines a historical result with a future promise. A reader can model the promise separately, with a probability and a timing assumption, while retaining the reported result.

This is particularly important in a downturn. When revenue falls, management may call layoffs, site closures and renegotiation costs “one-time” while the company is repeatedly shrinking. The costs may be unusual in the sense that they are not part of a healthy steady state, but they are still the cash cost of the current strategy. A normalized model can show a hypothetical post-restructuring business, but a liquidity model must show whether the company survives long enough to reach it.

Do not mix a forecast saving into historical EBITDA. Keep the forecast assumption visibly separate from the reported result.

That separation also makes disagreement productive. One analyst may reject stock compensation while another may retain it, but both can show the exact line where their models diverge. The reader can then test the assumption against dilution, cash repurchases and future hiring rather than arguing over an unexplained headline.

The same habit helps when reading management commentary. “We expect margins to improve” is not a reconciliation. It becomes testable only when the release identifies the cost, the action that removes it, the expected timing and the evidence that revenue will remain intact. A reader can then revisit the next filing and mark the prediction as realized, delayed or absent. That feedback loop is more valuable than a single optimistic quarter because it measures whether the company’s explanations earn trust over time.

## Sources & further reading

- [WeWork Companies Inc. registration statement and 2018 financial tables](https://www.sec.gov/Archives/edgar/data/1533523/000162827919000125/filename1.htm), SEC EDGAR, filed 2019. Source for 2017–2018 revenue, net loss, Adjusted EBITDA, Community Adjusted EBITDA, memberships, desks and facilities.
- [SEC Compliance & Disclosure Interpretations: Non-GAAP Financial Measures](https://www.sec.gov/rules-regulations/staff-guidance/corporation-finance-interpretations/non-gaap-financial-measures), SEC, last reviewed December 13, 2022. Source for reconciliation, prominence and recurring-charge guidance.
- [Conditions for Use of Non-GAAP Financial Measures](https://www.sec.gov/rules-regulations/2003/03/conditions-use-non-gaap-financial-measures), SEC Release No. 33-8176 / 34-47226, 2003. Primary rulemaking source for Regulation G.
- [Reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings), companion post in this forensic-accounting series.
- [Reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income), companion post on cash conversion.
- [Accrual accounting versus cash: the gap fraud exploits](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits), companion post on timing and estimates.
