---
title: "The accruals ratio and the accruals anomaly: measuring how much of a profit is just a promise"
date: "2026-08-09"
publishDate: "2026-08-09"
description: "A first-principles guide to splitting earnings into cash and accruals, computing the accruals ratio two different ways, understanding why the two answers disagree, and using the result to rank earnings quality without mistaking fast growth for fraud."
tags: ["forensic-accounting", "accruals-anomaly", "accruals-ratio", "earnings-quality", "cash-flow-analysis", "financial-statement-analysis", "earnings-management", "sunbeam", "quantitative-investing", "richard-sloan", "red-flags"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 57
---

> [!important]
> **TL;DR** — Reported profit is cash plus a layer of estimates called accruals, and the accrual layer does not last. Measure how thick it is and you have a fast, mechanical read on the quality of a company's earnings.
>
> - **Earnings = cash flow + accruals.** Richard Sloan's 1996 paper showed the cash component of profit predicts next year's profit much better than the accrual component does. Accruals mean-revert; cash tends to repeat.
> - The **accruals ratio** scales accruals by average total assets so that a \$50 billion company and a \$50 million company can be compared on the same axis.
> - There are **two standard ways to compute it** — off the balance sheet, and off the cash flow statement — and for the same firm-year they routinely give different answers. Hribar and Collins (2002) showed why: acquisitions, divestitures, currency translation and discontinued operations move working capital onto and off the balance sheet without any accrual ever passing through the income statement. That is why the cash-flow version is the modern default.
> - Use the ratio as a **ranking**, not a threshold. Sort a peer group into deciles and spend your attention on the top one.
> - **A high accruals ratio is a question, not a verdict.** Fast growth generates large accruals perfectly honestly. What separates growth from manipulation is whether receivables and inventory are growing roughly in line with sales — or far ahead of them.
> - The number to remember: in fiscal 1997, **Sunbeam Corporation** reported net earnings of **\$109.4 million** while its operations *consumed* **\$8.2 million** of cash. Accruals of \$117.7 million — larger than the entire reported profit. Sunbeam restated in November 1998 and filed for Chapter 11 in February 2001.

Two companies file their annual reports on the same morning. Both report net income of \$120 million. Both have roughly the same share count, so both report roughly the same earnings per share. A screen that ranks companies by price-to-earnings puts them side by side, indistinguishable.

Turn one page further, to the cash flow statement. The first company collected \$110 million of actual cash from operations during the year. The second collected \$10 million.

Same profit. Same P/E. One of these companies has been paid. The other has been promised. And over the following year, on average and across thousands of firms, the second one disappoints — not because anybody necessarily lied, but because the difference between profit and cash is made of estimates, and estimates come due.

![Net income of \$120 million split into a \$40 million cash component and an \$80 million accrual component, with the cash component persisting into next year and the accrual component mean-reverting.](/imgs/blogs/the-accruals-ratio-and-the-accruals-anomaly-1.webp)

That split is the mental model for everything below. A dollar of reported earnings is not one thing; it is a blend of two things with very different shelf lives. This article is about how to measure the blend with nothing but a company's own filed statements, how to turn the measurement into a comparable number, and — just as importantly — how to avoid the two mistakes that make the number useless: computing it the wrong way, and reading a high value as proof of fraud when it is very often proof of growth.

If you have not yet met accrual accounting itself, the companion piece on [accrual accounting versus cash and the gap fraud exploits](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) builds the concept from zero. Here we assume nothing either, but we move faster through the basics in order to spend most of our time on the measurement.

## Foundations: the building blocks of an accrual

Let us define every term we will use, from scratch. Skim this section if you already read financial statements for a living; do not skip it otherwise, because the rest of the article is arithmetic on these definitions and the arithmetic only makes sense if the definitions are solid.

**Revenue** is the value of what a company sold during a period. **Expenses** are the costs it incurred to make those sales. **Net income** — also called net earnings, net profit, or "the bottom line" — is revenue minus every expense, including interest and tax. It sits at the bottom of the **income statement**, the first of the three financial statements every public company must file.

Crucially, net income is computed on an **accrual basis**. That means revenue is recorded when it is *earned* — when the goods ship or the service is delivered — regardless of whether the customer has paid. And an expense is recorded when it is *incurred*, regardless of whether the company has written the cheque. This is not optional; both US GAAP and IFRS require it, for a good reason we will come to.

**Cash flow from operations** — universally abbreviated **CFO**, and labelled "net cash provided by operating activities" on the actual filing — is the money the core business actually took in, minus the money it actually paid out, during the same period. It lives on the **cash flow statement**, the third statement. It is a count of currency, not a judgment. The deep dive on [reading the cash flow statement and why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) covers this statement line by line.

The **balance sheet**, the second statement, is a snapshot of what the company owns (**assets**) and owes (**liabilities**) on one specific day. **Total assets** is the sum of everything it owns.

Now the term this whole article is built on.

**An accrual is the difference between accrual-basis profit and cash.** Formally:

$$\text{Accruals} = \text{Net income} - \text{Cash flow from operations}$$

That is not a definition somebody invented for convenience. It is what is left over once you strip the currency out of the profit. If a company reports \$120 million of profit and collected \$40 million of cash, then \$80 million of that profit consists of things that have been *recognised* but not *received*: sales invoiced and not yet collected, costs incurred but not yet paid, estimates of future returns, provisions, and depreciation charges on assets bought in earlier years.

### Why accrual accounting exists at all

It would be simpler to run the world on cash. It would also be badly misleading.

Consider a construction firm that signs a \$60 million contract in December, does \$20 million of work before the year ends, and gets paid the following March. On a pure cash basis it earned nothing in the year it did the work and everything in the year it did none. Accrual accounting fixes this with two rules:

- The **revenue-recognition principle**: book revenue when you have delivered the goods or performed the service, not when the money arrives.
- The **matching principle**: book a cost in the same period as the revenue it helped generate. A factory that will produce for ten years is not a ten-year-lump expense; it is expensed a slice at a time as **depreciation**.

Both rules make reported profit a *better* description of a period's economics than raw cash would be. They also both require somebody to exercise judgment — about when delivery really happened, about how long the factory will last, about how many customers will actually pay. That judgment layer is precisely the accrual.

So accruals are not a defect. They are the point. But they are the part of the profit that a company decides rather than counts, which makes them the part worth measuring.

### Working capital, the account where accruals live

Most accruals show up in a handful of short-term balance-sheet accounts, collectively called **working capital**:

| Account | What it is | What a rise means for cash |
| --- | --- | --- |
| **Accounts receivable** (AR) | Money customers owe you for goods already delivered | Cash *not yet* collected — a use of cash |
| **Inventory** | Goods you have bought or built but not yet sold | Cash already spent — a use of cash |
| **Prepaid expenses** | Costs paid in advance (insurance, rent) | Cash already spent — a use of cash |
| **Accounts payable** (AP) | Money you owe suppliers for goods already received | Cash *not yet* paid — a source of cash |
| **Accrued expenses** | Costs incurred but not yet billed (wages, utilities) | Cash not yet paid — a source of cash |
| **Deferred revenue** | Cash collected for goods not yet delivered | Cash already received — a source of cash |

The rule underneath the table: **an increase in an operating asset consumes cash; an increase in an operating liability provides it.** If your receivables rise by \$50 million, you booked \$50 million of revenue that has not turned into money. If your payables rise by \$20 million, you recognised \$20 million of costs you have not paid.

**Non-cash working capital** is working capital with the cash balance itself removed, because cash is the thing we are measuring *against*. It is the accountant's version of "everything short-term except the money".

#### Worked example: splitting one year of earnings into cash and accruals

Meet **Northfield Tools Inc.**, an illustrative industrial manufacturer we will use throughout this article. Every Northfield number in this post is invented for teaching purposes — it is not a real company and these are not real financials.

For fiscal year 2024, Northfield reports:

- Net income: **\$120 million**
- Cash flow from operations: **\$40 million**

Apply the definition:

- Accruals = Net income − CFO = \$120m − \$40m = **\$80 million**

So two-thirds of Northfield's reported profit — 80 divided by 120, or 66.7% — never became money during the year. That is not automatically alarming. It might be a company growing so fast that it is stuffing every spare dollar into receivables and inventory to serve new customers. It might equally be a company recognising revenue that will never collect. The raw split does not tell us which. What it tells us is *how much is at stake* if the estimates turn out to be wrong: \$80 million of Northfield's profit is riding on judgment.

**The takeaway: subtract operating cash flow from net income and you have isolated, in one arithmetic step, the entire portion of a company's profit that rests on estimates rather than currency.**

### Persistence, and what "mean reversion" actually means

One more definition, because it is the engine of everything that follows.

**Persistence** is how much of this year's profit shows up again next year. If a company earns \$100 million this year and, on average, \$85 million of that carries into next year's earnings, the persistence of its earnings is high. If only \$40 million carries over, persistence is low.

**Mean reversion** is the flip side: the tendency of an unusually high number to fall back towards its normal level. When we say accruals mean-revert, we are making a specific and mechanical claim — not a vague statement about markets being fair. We mean that an accrual is a claim on the future which must eventually be settled one way or the other, and the settlement lands in a *later* period's earnings.

## Why the accrual component is the discretionary one

Cash flow from operations is not impossible to manipulate — we will get to how — but it is *hard*, because at some point a bank statement has to agree with it. Accruals are different. Every accrual embeds at least one estimate, and an estimate is a choice made inside the company by people who know what number they would like the estimate to produce.

There are four main dials:

1. **Revenue timing.** When exactly is a sale "earned"? Ship the goods on 30 December instead of 2 January and the revenue lands in this year. Push distributors to accept product they do not need yet — **channel stuffing** — and this quarter borrows from the next. Book revenue on goods that sit in your own warehouse under a **bill-and-hold** arrangement and the sale exists on paper before it exists in the world. The full catalogue is in [revenue recognition games: channel stuffing and bill-and-hold](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold).

2. **Reserves and provisions.** How much of your receivables will go bad? How large should the warranty reserve be? Over-reserve in a bad year and you can release the excess into profit in a good one — the **cookie jar** technique described in [cookie-jar reserves and big-bath accounting](/blog/trading/forensic-accounting/cookie-jar-reserves-and-big-bath-accounting).

3. **Capitalise versus expense.** A cost booked as an expense reduces this year's profit. The same cost booked as an asset does not; it is depreciated slowly over future years. Move the line and profit moves with it.

4. **Estimates of useful life and salvage value.** Depreciation is an accrual — a negative one. Lengthen the assumed life of a factory from ten years to fifteen and this year's depreciation charge falls, and profit rises, with no change whatsoever to the business.

Each of these dials turns a real economic outcome into a reported number, and each dial has a range of defensible settings. Within that range, the number is a choice.

### The reversal is mechanical, not moral

Here is the property that makes the accruals ratio predictive rather than merely descriptive.

Pull revenue forward and you have not created a sale; you have *moved* one. The quarter you moved it into is richer and the quarter you moved it out of is poorer. Book a receivable that never collects and accounting rules will eventually force you to write it off, which reverses the profit — in a later period. Release a reserve into income and the reserve is gone; you cannot release it twice. Stretch a factory's useful life and you have lowered depreciation now at the cost of depreciating a stub later.

![The lifecycle of an accrual: a credit sale creates a receivable, which either collects as cash or unwinds through a write-off that reverses earnings in a later period.](/imgs/blogs/the-accruals-ratio-and-the-accruals-anomaly-2.webp)

Every one of these is a loan from the future to the present, and the future always collects. That is why accrual-heavy earnings are less persistent than cash-heavy earnings — not because accrual-heavy companies are dishonest, but because *arithmetic*. An accrual is by construction a temporary item; a cash collection is by construction a completed one.

#### Worked example: the accrual that unwinds

Suppose an illustrative distributor, Calder Distribution, has a genuinely steady business earning \$50 million a year on \$500 million of sales. In year one, under pressure to hit a target, it persuades distributors to take an extra \$100 million of product they do not need, at a 30% margin, and books the revenue.

**Year one, as reported:**

- Underlying profit: \$50m
- Extra profit from the pulled-forward shipments: \$100m × 30% = \$30m
- Reported net income: **\$80 million**
- Cash collected: nothing extra, because the distributors will pay in year two if they pay at all
- Accruals = \$80m − \$50m = **\$30 million**

Year one looks like a 60% jump in profit. Now year two arrives, and the distributors are sitting on a year's worth of extra inventory. They order nothing.

**Year two, as reported:**

- Underlying profit: \$50m
- Sales lost because the channel is full: \$100m × 30% = −\$30m
- Reported net income: **\$20 million**

Profit has fallen 75%, from \$80 million to \$20 million, in a business whose underlying economics never changed. Nobody stole anything. One year borrowed \$30 million of profit from the next and paid it back exactly.

And notice the crucial detail: the accrual in year one — \$30 million of profit with no cash behind it — was *visible on the face of the filed statements* twelve months before the collapse, to anyone who subtracted cash flow from net income.

**The takeaway: an accrual is a loan from next year's earnings, and the accruals ratio is a measure of how large that loan is before it comes due.**

## Computation one: the balance-sheet approach

Now we build the measurement. There are two standard recipes, and this is the older one — the one Richard Sloan used in the 1996 paper that started the literature.

The logic is simple: since accruals live in working capital, measure how much working capital *changed* over the year, then subtract depreciation because depreciation is a large negative accrual that does not touch working capital at all.

$$\text{Accruals}^{\text{BS}} = \Delta\text{NCCA} - \Delta\text{CL}^{*} - \text{D\&A}$$

Where:

- $\Delta\text{NCCA}$ is the change in **non-cash current assets** — total current assets minus cash and cash equivalents — over the year.
- $\Delta\text{CL}^{*}$ is the change in **current liabilities, excluding short-term debt and taxes payable**. Sloan excludes short-term debt, in his words, "because it relates to financing transactions as opposed to operating transactions", and excludes taxes payable for consistency with the earnings measure he uses on the other side of the comparison.
- $\text{D\&A}$ is **depreciation and amortisation** for the year, taken from the cash flow statement.

Every one of these numbers is on the face of the filings. Nothing here requires a database subscription.

Then we scale, because a \$125 million accrual means something very different at a \$1 billion company than at a \$100 billion one:

$$\text{Accruals ratio} = \frac{\text{Accruals}}{\text{Average total assets}}, \qquad \text{Average total assets} = \frac{\text{TA}_{t-1} + \text{TA}_{t}}{2}$$

Why *average* total assets rather than ending total assets? Because the accrual itself inflates ending assets. A company that books \$100 million of fictitious receivables has, by that very act, made its ending balance sheet \$100 million bigger — which shrinks the ratio precisely when it should be growing. Averaging the opening and closing balance sheets dilutes that feedback. It is a small effect in most years and a meaningful one in extreme ones, which is exactly when you care.

#### Worked example: the balance-sheet accruals ratio, line by line

Northfield Tools, fiscal 2024. All figures in millions of dollars, all illustrative.

| Balance-sheet line | FY2023 | FY2024 |
| --- | ---: | ---: |
| Cash and cash equivalents | 60 | 40 |
| Accounts receivable | 180 | 300 |
| Inventory | 140 | 230 |
| Other current assets | 20 | 30 |
| **Total current assets** | **400** | **600** |
| Accounts payable | 110 | 130 |
| Accrued expenses | 40 | 45 |
| Taxes payable | 15 | 20 |
| Short-term debt | 35 | 55 |
| **Total current liabilities** | **200** | **250** |
| **Total assets** | **1,000** | **1,500** |

And from the cash flow statement, depreciation and amortisation for FY2024 was **\$70 million**.

Step one — non-cash current assets:

- FY2023: 400 − 60 = **340**
- FY2024: 600 − 40 = **560**
- Change: 560 − 340 = **+220**

Step two — current liabilities excluding short-term debt and taxes payable:

- FY2023: 200 − 35 − 15 = **150**
- FY2024: 250 − 55 − 20 = **175**
- Change: 175 − 150 = **+25**

Step three — the change in non-cash working capital:

- 220 − 25 = **+195**

Step four — subtract depreciation and amortisation:

- 195 − 70 = **\$125 million of accruals**

Step five — scale:

- Average total assets = (1,000 + 1,500) ÷ 2 = **1,250**
- Accruals ratio = 125 ÷ 1,250 = **10.0%**

Ten percent of Northfield's entire asset base was created, during one year, by accrual rather than by cash. In most industrial peer groups that would place the company near the top of the distribution. Note where it came from: receivables rose 67%, from 180 to 300, and inventory rose 64%, from 140 to 230, while payables rose only 18%. The working capital ballooned.

**The takeaway: the balance-sheet method reads accruals off two consecutive balance sheets and one depreciation line, which is why it worked on decades of data that pre-date the modern cash flow statement.**

## Computation two: the cash-flow-statement approach

The second recipe is shorter, and it is the one most practitioners now use. Instead of reconstructing accruals from the balance sheet, take them straight from the definition:

$$\text{Operating accruals} = \text{Net income} - \text{CFO}$$

That is it. One line off the income statement, one line off the cash flow statement.

There is also a broader version, which many stock screeners label the **Sloan ratio**, that subtracts investing cash flow as well:

$$\text{Total accruals} = \text{Net income} - \text{CFO} - \text{CFI}$$

Where **CFI** is **cash flow from investing** — the cash spent on capital expenditure and acquisitions, net of cash received from selling assets. It is usually a negative number, so subtracting it *adds* to accruals.

Why would anyone include investing? Because the narrow version only captures accruals that pass through working capital, and a great many do not. Capitalised costs, acquired intangibles, and growth in long-lived operating assets are all ways of moving spending off the income statement and onto the balance sheet, and none of them touch receivables or inventory. Richardson, Sloan, Soliman and Tuna (2005) made this case formally, extending the accrual concept across the whole balance sheet. In their framework total accruals are the sum of three pieces — the change in non-cash working capital, the change in net non-current operating assets, and the change in net financial assets — and each piece carries a different **reliability** rating. Financial assets are the most reliable, because a marketable security has an observable price; working capital is middling; and the least reliable categories are current and non-current *operating* assets, which is where receivables, inventory and capitalised costs live. Their argument, and it is the one that matters for this article, is that the less reliable an accrual category is, the less persistent the earnings it produces — and that investors do not fully anticipate the shortfall, which is what produces the mispricing.

The cost of the broader version is that it no longer distinguishes an aggressive accounting choice from an honest, cash-funded factory. A company building a genuine plant shows a large negative CFI and therefore a large "total accrual", and there is nothing wrong with it at all. Use the broad version to ask *how much of this business's growth is being capitalised rather than expensed*; use the narrow version to ask *how much of this year's profit is uncollected*.

One caution about the screener version, because it is widely repeated as though it were gospel. Commercial screens usually pair the "Sloan ratio" with a rule of thumb — readings beyond roughly plus or minus 10% deserve scrutiny, readings beyond plus or minus 25% are alarming. **Those bands are practitioner convention, not something Sloan proposed.** His paper used a working-capital accrual scaled by average total assets, sorted firms into deciles, and set no absolute threshold at all. Treat the 10% line the way you would treat any round number somebody chose because it was round: useful as a prompt, worthless as a verdict.

#### Worked example: the same firm-year, off the cash flow statement

Northfield Tools again, fiscal 2024, illustrative, in millions of dollars. From the cash flow statement:

- Net income: **120**
- Net cash provided by operating activities (CFO): **40**
- Net cash used in investing activities (CFI): **−105**, comprising capital expenditure of −65 and the acquisition of Ridgeline Fastener for −40 in cash, net of cash acquired

Narrow version:

- Operating accruals = 120 − 40 = **80**
- Ratio = 80 ÷ 1,250 = **6.4%**

Broad version:

- Total accruals = 120 − 40 − (−105) = 120 − 40 + 105 = **185**
- Ratio = 185 ÷ 1,250 = **14.8%**

![Side-by-side computation of Northfield's accruals ratio: the balance-sheet method gives 10.0%, the narrow cash-flow method 6.4%, and the broad cash-flow method 14.8%.](/imgs/blogs/the-accruals-ratio-and-the-accruals-anomaly-3.webp)

Look carefully at that figure, because it contains the single most important practical warning in this article. The same company, the same fiscal year, the same audited statements — and three different accruals ratios: **10.0%**, **6.4%**, and **14.8%**. An analyst who says "Northfield's accruals ratio is 10%" has told you almost nothing unless they also tell you which recipe they used.

**The takeaway: always state the method alongside the number, because the methods do not agree and the disagreement is not small.**

## Why the two disagree — and which one to trust

If accruals are the gap between profit and cash, and both methods are measuring the same gap, why do they differ at all?

In a perfectly self-contained company they would not. The balance sheet, the income statement and the cash flow statement **articulate** — they are three views of the same set of double-entry books, and they must reconcile. If working capital rises by \$195 million and depreciation is \$70 million, then net income really should exceed operating cash flow by \$125 million.

The problem is that a real company does things other than operate. It buys businesses. It sells divisions. It earns money in currencies that move. It classifies a segment as discontinued. Every one of these events changes the balance sheet *without* a corresponding entry passing through the operating section of the income statement or the cash flow statement. Accountants call this **non-articulation**, and Hribar and Collins documented its consequences in a 2002 paper in the *Journal of Accounting Research* that quietly changed how the whole field computes accruals.

The four main sources:

1. **Acquisitions.** When you buy a company, its receivables, inventory and payables land on your consolidated balance sheet on day one. Your working capital jumps. But you did not *accrue* anything — you *bought* it, and the cash went out through the investing section. The balance-sheet method reads that jump as an accrual. It is not one.

2. **Divestitures.** The mirror image. Sell a division and its working capital leaves your balance sheet in one step, making the balance-sheet method read a large *negative* accrual where no reversal occurred.

3. **Foreign currency translation.** A subsidiary that keeps its books in euros has its balance sheet translated into dollars at the closing rate each year. If the euro strengthens, the dollar value of that subsidiary's receivables rises with no change in the underlying business and no accrual at all. The offset goes to a reserve in equity, not through profit.

4. **Discontinued operations.** Reclassifying a business as held-for-sale moves its assets and liabilities to separate single lines, distorting every working-capital comparison against the prior year.

Hribar and Collins put the weight mainly on the first, second and fourth of these — mergers, acquisitions and discontinued operations — with currency translation described as a smaller contributor. Their conclusion was blunt: the balance-sheet approach introduces significant measurement error into accrual estimates, and where the thing a researcher is sorting on happens to correlate with acquisition activity, that error is enough to make earnings management appear where none exists.

The direction of the bias matters. The firms most affected are the *acquisitive* ones — and acquisitive firms are already over-represented among companies that later disappoint. So the balance-sheet method does not just add random noise; it adds noise that is correlated with the very outcome you are trying to predict.

#### Worked example: the acquisition that manufactures \$45 million of phantom accruals

Return to Northfield Tools and reconcile the two answers we computed. Balance-sheet accruals were **\$125 million**. Net income minus CFO was **\$80 million**. The gap is **\$45 million**. Where did it come from?

Mid-year, Northfield acquired Ridgeline Fastener. The transaction was part cash, part newly issued Northfield stock. Ridgeline arrived carrying:

- Accounts receivable and inventory: **\$50 million**
- Accounts payable: **\$5 million**

So the acquisition alone added 50 − 5 = **\$45 million** to Northfield's consolidated non-cash working capital, on the day it closed.

![How an acquisition drops working capital onto the balance sheet without any accrual passing through the income statement, creating a \$45 million wedge between the two methods.](/imgs/blogs/the-accruals-ratio-and-the-accruals-anomaly-4.webp)

Now trace what each statement saw:

- The **balance sheet** saw non-cash working capital rise by \$45 million. It cannot tell the difference between working capital that was accrued and working capital that was purchased.
- The **income statement** saw nothing. No revenue was recognised, no expense incurred, no estimate made. Ridgeline's receivables were recorded at fair value on the acquisition date as part of the purchase price allocation.
- The **cash flow statement** saw cash leave — through the *investing* section, where it belongs, not the operating section.

Reconciliation: \$125m of balance-sheet accruals − \$45m of acquired working capital = **\$80m**, exactly the net income minus CFO figure. The two methods now agree, and the entire original discrepancy was one purchase.

**The takeaway: the balance-sheet method cannot distinguish working capital a company accrued from working capital it bought, so for any acquisitive company it systematically overstates accruals.**

### So which one should you use?

For any analysis of a company filing after the late 1980s — when the modern cash flow statement became mandatory in the United States — **use the cash-flow-statement version.** It is shorter, it requires fewer judgment calls about which liabilities to exclude, and it is immune to all four non-articulation problems above, because the cash flow statement's own operating section already routes acquisition effects to investing.

Two honest caveats:

- The balance-sheet method is not wrong, it is *older*. Sloan needed it because his sample stretched back to 1962, and the standard that created the modern US cash flow statement — SFAS No. 95 — was in force for only the last four years of his thirty-year sample. Any study using long historical data faces the same constraint. This is a genuine tension in the literature rather than an oversight: the paper that established the anomaly used the measure that a later paper showed to be noisy.
- The cash-flow method inherits whatever games have been played with the classification of cash flows themselves. Moving an operating outflow into the investing section, or selling receivables to a bank to convert a future operating collection into cash today, both flatter CFO and therefore shrink measured accruals. That family of tricks has its own article: [cash flow statement manipulation and classification shifting](/blog/trading/forensic-accounting/cash-flow-statement-manipulation-classification-shifting). We will see a textbook example of the receivables-sale version when we get to Sunbeam.

A practical habit: compute both. When they agree, you have one clean signal. When they diverge sharply, you have learned something specific — go find the acquisition, the divestiture or the currency move that explains the gap, and if you cannot find one, that itself is worth a question.

## Turning the ratio into a ranking: the decile view

A single accruals ratio, in isolation, is nearly meaningless. Is 10% high? It depends entirely on the industry, the growth rate, and the year. A software company collecting subscriptions a year in advance runs *structurally negative* accruals, because deferred revenue is an operating liability that keeps growing. A capital goods manufacturer with 90-day payment terms runs structurally positive ones. Comparing the two on an absolute scale tells you about their business models, not their earnings quality.

The fix is to stop reading the ratio as a level and start reading it as a **rank**.

### Building the ranking

The recipe used by essentially every implementation, academic and practitioner:

1. **Choose a universe and clean it.** Exclude **financials** — for a bank, "working capital" is not a meaningful concept, since receivables *are* the business — and usually utilities, whose regulated asset bases behave differently. Exclude firms below a minimum size, because a \$5 million-asset shell produces absurd ratios from trivial dollar amounts.

2. **Compute the ratio consistently.** One method, one scaling choice, applied identically to every firm. Mixing methods across a universe destroys the ranking.

3. **Winsorise the extremes.** Set every observation above the 99th percentile equal to the 99th percentile, and likewise at the 1st. A handful of firms with near-zero average assets will otherwise dominate the entire distribution.

4. **Rank within industry, or at least within sector.** Otherwise your "worst accruals" decile is just a list of the industries with the longest cash conversion cycles.

5. **Sort into ten equal buckets — deciles.** Decile 1 holds the lowest (most negative) accruals; decile 10 holds the highest.

6. **Respect the calendar.** The annual report is not public on the last day of the fiscal year. Any ranking that uses fiscal-2024 data must be formed *after* the fiscal-2024 report was actually filed — commonly three or four months after year-end — or the exercise is using information nobody had.

![A decile ladder ranking a universe by accruals ratio, running from roughly −14% in decile 1 to +19% in decile 10.](/imgs/blogs/the-accruals-ratio-and-the-accruals-anomaly-5.webp)

The ladder in the figure shows illustrative cut-points; the exact values shift with the universe, the year and the method. What is stable is the *shape*: a smooth monotonic run from firms whose earnings are more than fully cash-backed, through a middle where accruals are close to zero, to a top decile where accruals represent a large fraction of the entire asset base.

#### Worked example: ranking ten industrial peers

Here is a ten-company illustrative peer group in the same industry, using the narrow cash-flow method. All figures in millions of dollars, all invented.

| Company | Net income | CFO | Avg. total assets | Accruals (NI − CFO) | Accruals ratio | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cornice Bearings | 40 | 95 | 500 | −55 | **−11.0%** | 1 |
| Haldane Pumps | 62 | 110 | 640 | −48 | **−7.5%** | 2 |
| Trent Valve | 35 | 58 | 460 | −23 | **−5.0%** | 3 |
| Marlow Castings | 88 | 105 | 700 | −17 | **−2.4%** | 4 |
| Ashfield Gears | 51 | 56 | 500 | −5 | **−1.0%** | 5 |
| Deverel Seals | 44 | 40 | 400 | +4 | **+1.0%** | 6 |
| Northfield Tools | 120 | 40 | 1,250 | +80 | **+6.4%** | 7 |
| Oakridge Fasteners | 30 | 12 | 250 | +18 | **+7.2%** | 8 |
| Pellham Hydraulics | 76 | 22 | 540 | +54 | **+10.0%** | 9 |
| Quarry Lane Industrial | 55 | −5 | 300 | +60 | **+20.0%** | 10 |

Three observations worth more than the table itself.

**First, look at Quarry Lane.** It reported \$55 million of profit in a year its operations *consumed* \$5 million of cash. Every dollar of reported profit — and then \$5 million more — was accrual. This is the single most informative pattern in the whole exercise: positive net income alongside negative operating cash flow. It is not always fraud, but it always needs an explanation.

**Second, look at Northfield.** On the narrow method it ranks seventh of ten at 6.4% — elevated, not alarming. But we computed its *broad* total accruals ratio at 14.8%, which would put it at the very top of this table. A company can look moderate on one measure and extreme on another, and the difference is exactly the capitalised and acquired growth that the narrow method ignores.

**Third, look at Cornice Bearings at −11.0%.** Negative accruals mean cash flow exceeded profit — usually a good sign, sometimes a great one. But not always. A company that has just taken an enormous restructuring charge reports a deeply negative accrual in the charge year, because the expense hit profit without any cash leaving. That is the **big bath**, and it sets up artificially clean earnings for years afterwards. Decile 1 deserves a glance, not blind trust.

**The takeaway: the accruals ratio earns its keep as a sorting device that tells you which twenty companies out of two hundred deserve an afternoon of your attention.**

## What Sloan found, and what happened to the finding afterwards

We can now state the result that made this measure famous.

In 1996, the accounting professor **Richard G. Sloan** published *"Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About Future Earnings?"* in *The Accounting Review* (volume 71, number 3, pages 289–315). He split earnings into their cash and accrual components using the balance-sheet method described above, and asked two questions in sequence.

**Question one: which component predicts next year's earnings better?** The answer was the cash component, decisively. A dollar of cash-backed profit persisted into the following year far better than a dollar of accrual-backed profit. This part of the paper is straightforward accounting measurement, and it has held up across subsequent decades and many international samples. It is the finding this entire article rests on.

**Question two: does the stock market know?** Sloan's answer was no. Share prices behaved as though investors *fixated* on the headline earnings number, treating a dollar of accrual profit as though it were as durable as a dollar of cash profit. Because high-accrual firms' earnings were about to fall and low-accrual firms' earnings were about to hold, high-accrual firms were systematically over-priced and low-accrual firms under-priced. Sloan reported that a hedge portfolio — long the lowest-accrual decile, short the highest-accrual decile — earned a size-adjusted return of **10.4%** in the first year after portfolio formation. That result is what the literature calls **the accruals anomaly**.

Four things about that 10.4% figure deserve to be said plainly, because they are routinely dropped when the number is quoted.

- It is **what Sloan's paper reports for his own sample**: 40,679 firm-year observations on NYSE and AMEX between 1962 and 1991, with financial firms excluded. It is not a forecast, not a live return, and not something anybody is currently earning.
- It is a **size-adjusted paper return on a long-short portfolio**, computed before transaction costs, before borrowing costs on the short leg, and before the practical difficulty of actually shorting small illiquid companies. Real implementations do not capture paper returns.
- It is a **first-year** number. The paper reports the effect decaying quickly: roughly 4.8% in the second year after formation and roughly 2.9% in the third, the third year not being statistically significant. Quoting 10.4% as though it were a durable annual edge misstates what the paper found.
- The two legs are **not symmetric**: the long low-accrual leg contributed about 4.9% and the short high-accrual leg about −5.5%. And the paper reports that the effect was positive in 28 of the 30 sample years, the exceptions being 1966 and 1981.

One further detail from the paper is the most persuasive evidence that the effect is really about earnings rather than about risk. Sloan found that a large share of the predictable return clusters around the *subsequent quarterly earnings announcements* — the paper reports roughly 4.5 percentage points of the total arriving in announcement windows that make up less than 5% of trading days. If the return were compensation for bearing some unmeasured risk, it would accrue smoothly through the year. Instead it arrives on the days when the market learns that the accruals did in fact reverse.

### The anomaly decayed

This is the part that gets left out of the confident version of the story, and leaving it out would be dishonest.

![A timeline of the accruals anomaly from Sloan's 1996 publication through the arrival of arbitrage capital to the studies documenting its decay.](/imgs/blogs/the-accruals-ratio-and-the-accruals-anomaly-6.webp)

The finding was published, widely read, and then traded. Sloan himself later moved into asset management. Quantitative funds added accruals to their factor models. And the return attributable to the signal fell.

- **Green, Hand and Soliman**, writing in *Management Science* in 2011 under the title *"Going, Going, Gone? The Apparent Demise of the Accruals Anomaly"*, examined the strategy's performance after publication and concluded that hedge returns to the accruals strategy had decayed in US stock markets to the point that they were, on average, no longer reliably positive. Their proposed explanation is measurable rather than rhetorical: they attribute the decay principally to the growth in capital deliberately trading the signal, proxied by hedge fund assets under management and by trading volume in extreme-accrual firms, with a smaller secondary role for a decline in the size of the mispricing signal itself.

- **McLean and Pontiff**, in *"Does Academic Research Destroy Stock Return Predictability?"* (*Journal of Finance*, 2016), studied this pattern across 97 published return predictors rather than accruals alone. Their headline finding: portfolio returns are on average **26% lower out of sample** and **58% lower after publication**. The difference between those two figures — about 32 percentage points of decay — is the part attributable to publication itself, that is, to investors reading the paper and trading it. The remainder is the original result having been partly statistical luck. The accruals anomaly is one instance of a general phenomenon, not a special case.

- **Fama and French**, in *"Dissecting Anomalies"* (*Journal of Finance*, 2008), tested it across company sizes. It is worth correcting a claim that circulates widely here: they did **not** find accruals to be a small-company-only effect. They report that the accruals return is pervasive and shows up in micro, small and big stocks alike. It is the separate *asset growth* anomaly that they find in small stocks and not in large ones. What their work does underline is a capacity problem — microcaps, defined as firms below the 20th percentile of NYSE market equity, are roughly 60% of listed names but only about 3% of total market value. The segment where anomalies are sharpest is the segment where large amounts of money cannot fit.

- **Not everyone agrees it died.** Lev and Nissim, in *"The Persistence of the Accruals Anomaly"* (*Contemporary Accounting Research*, 2006), reached the opposite conclusion five years earlier: that the anomaly still persisted and its magnitude had not declined over time. Their explanation for its survival is the more interesting contribution. Institutions do trade on accruals, they found, but only slightly — the accruals-related change in institutional ownership amounts to substantially less than 10% of the average quarterly change in institutional ownership — because extreme-accrual firms are small, unprofitable and risky, a profile that conflicts with most institutional mandates. Individual investors are blocked by information and transaction costs. A mispricing that almost nobody is permitted to arbitrage can survive being famous.

- **Rising liquidity eroded anomalies generally.** Chordia, Subrahmanyam and Tong, in the *Journal of Accounting and Economics* in 2014, found that increases in market liquidity and trading activity significantly decreased exploitable anomaly returns in US equities across the board — a mechanism that operates on the accruals strategy without being about accruals at all.

- **Limits to arbitrage** were part of the story from the beginning. High-accrual firms tend to be small, illiquid, expensive to borrow for shorting, and volatile — the exact profile that makes a mispricing persist because nobody can profitably correct it.

### What survives

Quite a lot, and it is the more useful half.

The **mispricing** result has weakened. The **measurement** result has not. Accrual-heavy earnings are still less persistent than cash-heavy earnings, for the mechanical reason set out earlier in this article: accruals reverse. That is an accounting identity playing out over time, and no amount of arbitrage capital can trade it away, because it is not a market phenomenon at all.

So the honest modern use of the accruals ratio is not "short the top decile". It is:

- A **screening filter** that tells you where to look first among hundreds of candidates.
- A **quality overlay** on a valuation you have already done — the same earnings deserve a lower multiple when they are less likely to repeat.
- A **forensic starting point**, which is how this series uses it: not to price a stock, but to decide which company's footnotes are worth three hours.

**The takeaway: the trade decayed; the accounting did not. Treat the accruals ratio as a measure of earnings durability, not as a strategy.**

## High accruals are not proof of anything

The most common error made with this measure is treating a high reading as an accusation. It is not. It is a description of a company's balance sheet, and there are at least three innocent explanations that are more common than fraud.

### One: the company is growing

This is the big one, and it is not a minor caveat — it is a rival explanation for the entire anomaly.

Growth consumes working capital. To sell more, you must hold more inventory. When you sell it on 60-day terms, receivables rise in proportion. A company doubling its revenue will roughly double its receivables and inventory, which produces a large positive accrual with no accounting judgment involved whatsoever.

**Fairfield, Whisenant and Yohn** made this argument formally in *The Accounting Review* in 2003, in *"Accrued Earnings and Growth: Implications for Future Profitability and Market Mispricing"*. Their claim is that the lower persistence of accrual earnings is not special to accruals at all, but is one instance of a general property of **growth in net operating assets** — diminishing marginal returns on new investment. Expand the asset base and the return on that base tends to fall, whether the expansion happened through receivables or through a new factory. **Zhang**, writing in the same journal in 2007 in *"Accruals, Investment, and the Accrual Anomaly"*, pushed in the same direction with a sharper test. He reported that the magnitude of the accrual anomaly increases monotonically with how much *investment* information the accruals carry, measured by how closely a firm's accruals move with its **employee growth**. Where accruals track hiring — that is, where they are a by-product of the company genuinely getting bigger — the anomaly is strong. Where they do not, it is much weaker. That is close to the opposite of what a pure earnings-manipulation story would predict.

If they are right, a large part of what the accruals ratio flags is companies investing heavily — some of which will earn poor returns on that investment, none of which are committing fraud.

### Two: the business model produces it

Long payment terms, seasonal inventory builds, project accounting on multi-year contracts, and industries where customers habitually pay late all generate structurally elevated accruals. A defence contractor recognising revenue on percentage-of-completion will look accrual-heavy forever, and correctly so.

### Three: something structural happened

An acquisition, as we have seen at length. A currency move. A change in accounting standard. A shift from selling receivables to holding them — which *raises* measured accruals while making the accounts more conservative, not less.

### So how do you tell the difference?

You stop looking at the ratio in isolation and start looking at what is underneath it. The diagnostic is the relationship between working capital and sales.

![A diagnostic fork: the same high accruals ratio branches into benign growth, a genuine red flag, or a structural cause, depending on how receivables move relative to revenue.](/imgs/blogs/the-accruals-ratio-and-the-accruals-anomaly-7.webp)

The core test uses **days sales outstanding (DSO)** — the average number of days a company waits to get paid, computed as receivables divided by revenue, multiplied by 365. If receivables are growing because the business is growing, DSO stays roughly flat. If receivables are growing because sales are being recognised that customers are not paying for, DSO climbs. The [inventory and receivables inflation](/blog/trading/forensic-accounting/inventory-and-receivables-inflation-the-classic-red-flag) article works through this test in detail, and [the cash conversion cycle](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) extends it to inventory and payables.

#### Worked example: two companies, identical ratio, opposite stories

Both of the following are illustrative. Both have an accruals ratio of roughly +11%. All figures in millions of dollars.

**Ashgrove Medical — the growth story**

| | Prior year | Current year | Change |
| --- | ---: | ---: | ---: |
| Revenue | 500 | 700 | **+40%** |
| Accounts receivable | 85 | 121 | **+42%** |
| Days sales outstanding | 62 days | 63 days | **+1 day** |

DSO check: 85 ÷ 500 × 365 = 62 days, and 121 ÷ 700 × 365 = 63 days. Receivables grew 42% against revenue growth of 40%, so the collection period barely moved. Ashgrove is selling to more customers on the same terms and funding the growth out of working capital. Its accruals are large because it is expanding, and the expansion is real.

**Belmont Instruments — the red flag**

| | Prior year | Current year | Change |
| --- | ---: | ---: | ---: |
| Revenue | 480 | 499 | **+4%** |
| Accounts receivable | 82 | 127 | **+55%** |
| Days sales outstanding | 62 days | 93 days | **+31 days** |

DSO check: 82 ÷ 480 × 365 = 62 days, and 127 ÷ 499 × 365 = 93 days. Receivables grew 55% on revenue growth of 4%. The collection period rose by half. Belmont is not selling to more customers; it is either recognising sales that will not collect, or shipping to distributors who cannot move the product, or extending desperate terms to buy one more quarter of reported growth.

Same accruals ratio. Completely different companies.

**The takeaway: the accruals ratio tells you where to look, and the trend in days sales outstanding tells you what you are looking at.**

## Case study: Sunbeam Corporation, fiscal 1997

Everything above can be seen in one real set of filed statements.

Sunbeam Corporation made toasters, blenders, grills and outdoor furniture. In July 1996 it hired Albert J. Dunlap, a restructuring executive nicknamed "Chainsaw Al", to turn it around. Fiscal 1996, the year of his arrival, produced a net loss of **\$228.3 million**. Fiscal 1997 produced net earnings of **\$109.4 million** on net sales of **\$1,168.2 million**, up 18.7% year over year. It was hailed as one of the great corporate turnarounds of the decade, and the shares rose to a high of **\$52 in March 1998**.

Every figure in this section comes from Sunbeam's own SEC filings or from the Securities and Exchange Commission's enforcement releases; the ratios are computed from those filed lines, and I mark them as computed where that is the case.

### The number on the next page

In the same Form 10-K405 for the fiscal year ended 28 December 1997, filed on 6 March 1998, the consolidated statement of cash flows reported that operating activities had **used \$8.2 million of cash**. The prior year, the disastrous one, operations had *provided* \$14.2 million.

So in the celebrated turnaround year, profit went up by \$337.7 million and operating cash flow went *down*.

Compute the accruals:

- Accruals = Net income − CFO = \$109,415 thousand + \$8,249 thousand = **\$117.7 million** *(computed from filed lines)*

That number is larger than the entire reported profit. Every dollar of Sunbeam's 1997 net earnings, and \$8.2 million besides, was accrual.

Scaling it: Sunbeam's total assets were \$1,072.7 million at the end of fiscal 1996 and \$1,120.3 million at the end of fiscal 1997, giving average total assets of \$1,096.5 million.

- Accruals ratio = \$117.7m ÷ \$1,096.5m = **10.7%** *(computed from filed lines)*

That is a top-decile reading in almost any universe, on data that was public, free, and printed in the same document as the headline everybody was celebrating.

### Where the accrual lived

The cash flow statement itself names the culprits.

![A waterfall bridging Sunbeam's fiscal 1997 net earnings of \$109.4 million down to operating cash flow of negative \$8.2 million, with receivables and inventories accounting for the entire gap.](/imgs/blogs/the-accruals-ratio-and-the-accruals-anomaly-8.webp)

Working down from profit to cash, in millions of dollars:

| Line | Amount |
| --- | ---: |
| Net earnings | **+109.4** |
| Increase in receivables, net | **−84.6** |
| Increase in inventories | **−100.8** |
| All other adjustments, net | **+67.7** |
| **Net cash used in operating activities** | **−8.2** |

*Figures are rounded to the nearest \$0.1 million from a statement filed in thousands, so the rounded column does not sum exactly. On the filed figures the bridge reconciles to the dollar.*

Two working-capital lines, together \$185.4 million of cash consumed, wiped out a \$109.4 million profit.

Cross-check against the balance sheet, which tells the same story from the other side:

| Balance | FY1996 | FY1997 | Growth |
| --- | ---: | ---: | ---: |
| Net sales | 984.2 | 1,168.2 | **+18.7%** |
| Receivables, net | 213.4 | 295.6 | **+38.5%** |
| Inventories | 162.3 | 256.2 | **+57.9%** |

Receivables grew at roughly twice the rate of sales, and inventories at roughly three times. This is precisely the Belmont pattern from the worked example above, on a real company, a year before the collapse.

#### Worked example: the two methods disagree on Sunbeam too — by a little

Sunbeam also demonstrates the non-articulation problem in miniature, and it is worth doing the arithmetic because it shows the effect is real but usually modest.

The **balance sheet** says receivables rose from \$213.4 million to \$295.6 million — a change of **\$82.1 million**. The **cash flow statement's** own line for the change in receivables is **\$84.6 million**. Those are not the same number.

Same for inventory. The balance sheet moves from \$162.3 million to \$256.2 million, a change of **\$93.9 million**. The cash flow statement says **\$100.8 million**.

| Line | Balance-sheet change | Cash-flow-statement change | Wedge |
| --- | ---: | ---: | ---: |
| Receivables | 82.1 | 84.6 | **2.5** |
| Inventories | 93.9 | 100.8 | **6.9** |
| | | | **9.3 total** |

*Again, rounded from thousands: the wedge is \$2,464 thousand on receivables and \$6,882 thousand on inventories, \$9,346 thousand in total.*

A \$9.3 million wedge between two methods that are supposedly measuring the same thing — and Sunbeam was both acquiring and divesting during this period, with the fiscal 1997 investing section including \$91.0 million of proceeds from selling divested operations. Against \$117.7 million of accruals, a \$9.3 million discrepancy is about 8%: too small to change the conclusion here, easily large enough to change it at a company nearer the decile boundary.

**The takeaway: non-articulation is usually a rounding issue and occasionally decisive, which is a good argument for computing both methods and investigating the gap rather than picking one and hoping.**

### The cash flow number was itself flattered

There is one more layer, and it is the classification-shifting trick mentioned earlier.

Sunbeam's own management discussion in the fiscal 1997 10-K discloses that the operating cash flow figure "reflects \$59 million of proceeds from the sale of trade accounts receivable" under a revolving securitisation programme entered into in **December 1997** — that is, in the final weeks of the fiscal year.

Selling your receivables to a bank converts a future operating collection into cash today, and the proceeds land in operating cash flow. Strip that programme out and Sunbeam's fiscal 1997 operating cash flow was closer to **negative \$67 million**, and the accrual correspondingly larger. The already-alarming published number was the *flattering* version.

### What the SEC later found

On 15 May 2001, the SEC brought its case: *SEC v. Albert J. Dunlap, Russell A. Kersh, Robert J. Gluck, Donald R. Uzzi, Lee B. Griffith and Phillip E. Harlow* (Litigation Release No. 17001, AAER No. 1395), together with a settled cease-and-desist proceeding against the company itself (*In the Matter of Sunbeam Corporation*, Securities Act Release No. 7976, AAER No. 1393).

The Commission's allegations map directly onto the accrual dials set out earlier in this article:

- **Cookie-jar reserves.** Management allegedly created approximately **\$35 million** of improper restructuring and other reserves as part of the year-end 1996 restructuring, which were then reversed into income the following year. A loss year made artificially worse to make the turnaround year artificially better.
- **Bill-and-hold.** Revenue recognised on goods that never left Sunbeam's control. The SEC identified, among other items, roughly **\$29 million** of fourth-quarter 1997 bill-and-hold sales contributing about **\$4.5 million** of income, and a second-quarter 1997 arrangement worth about **\$14 million** of revenue and over **\$6 million** of income.
- **Guaranteed sales and channel stuffing.** Product pushed into distribution ahead of demand, with reports showing customers holding as much as **80 weeks** of inventory of specific products.

The Commission's summary of the year: *"At year-end 1997, at least \$62 million of Sunbeam's reported income of \$189 million came from accounting fraud"* — that \$189 million being pre-tax income from continuing operations, against the \$109.4 million after-tax net earnings figure we used above.

### The unwinding

Sunbeam restated in a Form 10-K/A filed on **12 November 1998**. The restatement note reports fiscal 1997 net sales cut from \$1,168.2 million to **\$1,073.1 million**, pre-tax earnings from continuing operations cut from \$189.3 million to **\$92.7 million**, and net earnings cut from \$109.4 million to **\$38.3 million** — diluted earnings per share falling from \$1.25 to **\$0.44**. Restated fiscal 1997 operating cash flow was still negative, at **−\$6.0 million**.

The rest followed the usual sequence. An earnings warning on 3 April 1998 dropped the stock over 24%. Dunlap was terminated in June 1998 following a board investigation. Sunbeam filed for Chapter 11 bankruptcy protection on **6 February 2001**. In September 2002 (Litigation Release No. 17710), Dunlap and Kersh consented to permanent injunctions and permanent officer-and-director bars, with civil penalties of **\$500,000** and **\$200,000** respectively; the same release records that Dunlap paid **\$15 million** and Kersh \$250,000 out of their own funds to settle a related class action.

**The takeaway: the highest-quality signal about Sunbeam's 1997 earnings was printed in Sunbeam's own 1997 annual report, roughly eight months before the restatement, and it took one subtraction to find.**

### A second case: Lucent Technologies, fiscal 1999

One case can be a coincidence, so here is a second, from a company nobody thought of as a small-cap accounting risk.

For the fiscal year ended 30 September 1999, Lucent Technologies reported revenues of **\$38,303 million** and net income of **\$4,766 million** — a headline profit that made it one of the most widely held stocks in America. The same Form 10-K405 reported net cash **used in** operating activities of **\$276 million**.

- Accruals = \$4,766m − (−\$276m) = **\$5,042 million** *(computed from filed lines)*
- Average total assets = (\$29,363m + \$38,775m) ÷ 2 = \$34,069m
- Accruals ratio = \$5,042m ÷ \$34,069m = **14.8%** *(computed from filed lines)*

The cash flow statement attributes the swing to two familiar lines: an increase in receivables of **\$3,183 million** and an increase in inventories and contracts in process of **\$1,612 million**. Together, \$4,795 million — about 95% of the entire accrual — sat in receivables and inventory, while revenue grew 20.4%, from \$31,806 million the previous year.

On **21 December 2000**, Lucent announced a restatement cutting fourth-quarter fiscal 2000 revenue by **\$679 million**. In May 2004, the SEC settled charges alleging that Lucent had "fraudulently and improperly recognized approximately \$1.148 billion of revenue and \$470 million in pre-tax income" during fiscal 2000, through undisclosed side agreements and credits; the company paid a **\$25 million** penalty, which the Commission stated was imposed for its lack of cooperation in the investigation rather than for additional violations.

To be careful about what is being claimed: neither the SEC nor any academic paper says "Lucent's accruals ratio predicted this". The arithmetic above is mine, applied to Lucent's own filed lines, and the inference — that a 14.8% accruals ratio with 95% of it in receivables and inventory was a reason to ask harder questions in 1999 — is an inference, not a documented finding. But the numbers are Lucent's, and they were public a full year before the restatement.

## Common misconceptions

**"Accruals are bad accounting."** Accruals are *required* accounting, and they usually make reported profit more informative than raw cash would be. A construction firm on percentage-of-completion, a software firm with deferred revenue, a manufacturer depreciating a plant — all are producing better information because of accruals, not despite them. The signal is never the existence of accruals; it is their size relative to the company's own history and its peers.

**"A high accruals ratio means the company is committing fraud."** It usually means the company is growing, or acquiring, or operating in an industry with long payment terms. Fairfield, Whisenant and Yohn's 2003 work suggests a substantial part of the effect is a growth phenomenon rather than a manipulation one. The ratio narrows a search; it does not conclude an investigation.

**"Negative accruals are always a good sign."** Not always. A company taking a large restructuring charge records an expense that hits profit without cash leaving, producing a deeply negative accrual in the charge year — and a conveniently low cost base afterwards. That is the big-bath technique, and decile 1 of an accruals screen will reliably contain some of it alongside the genuinely cash-rich companies.

**"The two computation methods should give the same answer, so one of them must be wrong."** Neither is wrong. They diverge for structural reasons — acquisitions, divestitures, currency translation, discontinued operations — that Hribar and Collins identified in 2002. The divergence is information: it points at a corporate event, and finding that event is part of the analysis.

**"Cash flow cannot be manipulated, so the cash-flow method is safe."** Cash flow is *harder* to manipulate, not impossible. Selling receivables at year end, stretching payables past the reporting date, and reclassifying operating outflows into the investing section all flatter CFO. Sunbeam's \$59 million December 1997 receivables securitisation is a documented example that made its published operating cash flow look better than the underlying business.

**"The accruals anomaly is free money."** It was a paper return in a paper published in 1996, computed before trading costs on a long-short portfolio concentrated in small illiquid stocks. Green, Hand and Soliman's 2011 work concluded the hedge return was no longer reliably positive in US markets, and McLean and Pontiff's 2016 study of 97 published predictors found returns 58% lower after publication on average. In fairness, the evidence is not unanimous — Lev and Nissim argued in 2006 that the anomaly had persisted undiminished, precisely because the firms at the extremes are too small and too risky for institutions to trade. Either way, nobody credible is describing this as free money. The measurement remains useful; the trade does not.

**"You can run this screen on any company."** Not banks, insurers or other financials. For a bank, loans receivable *are* the operating business, so "non-cash working capital" measures something entirely different. Exclude them from the universe rather than reading their ratios as if they were comparable.

## How it shows up in real markets

**As a quantitative quality factor.** Accruals-based measures are standard inputs to systematic "quality" and "earnings quality" factor definitions used by quantitative equity managers. They rarely stand alone now; they sit alongside profitability, leverage and investment measures in composite scores, which is a reasonable response to the evidence that the standalone return decayed.

**Inside the Beneish M-Score.** Messod Beneish's *"The Detection of Earnings Manipulation"* (*Financial Analysts Journal*, volume 55, number 5, 1999) built an eight-variable model for flagging likely earnings manipulators. One of the eight variables is **TATA — Total Accruals To Total Assets** — defined in the paper as the change in working-capital accounts other than cash, less depreciation, which is the same balance-sheet family as Sloan's measure. Beneish is explicit about why it is there: total accruals "proxy for the extent to which cash underlies reported earnings", and higher positive accruals mean less cash and a higher likelihood of manipulation. TATA also carries the largest coefficient of any variable in his model. The single most famous quantitative fraud screen has the accruals ratio sitting at its centre.

A note on the threshold, since it is almost always quoted wrongly. The cutoff in Beneish's own paper is **−1.78** (with −1.49 given for a different assumed ratio of error costs); a score above the cutoff marks a firm as a likely manipulator. The **−2.22** figure that circulates in blog posts and screener documentation does not appear in the 1999 paper. It is a later practitioner convention of uncertain origin, and it should not be attributed to Beneish.

**Inside the Piotroski F-Score.** Joseph Piotroski's 2000 paper in the *Journal of Accounting Research* scored value stocks on nine binary financial-health signals. One of the nine is an accruals test in its simplest possible form: the company scores a point if cash flow from operations exceeds return on assets, both scaled by beginning-of-year total assets — which, since the denominator is shared, is the same as asking whether operating cash flow exceeds net income before extraordinary items. In other words, one point for negative accruals. Piotroski reported that applying the full nine-signal score to high book-to-market firms raised the mean return to that strategy by at least seven and a half percentage points annually over his 1976–1996 sample.

**As the short seller's first pass.** Published short theses on accounting-driven situations characteristically open with the profit-versus-cash divergence, because it is the fastest way to establish that something needs explaining before any expensive fieldwork begins. The gap does not prove the thesis; it justifies the research budget.

**In credit analysis.** A lender underwriting against EBITDA cares intensely whether that EBITDA converts to cash, because covenants are tested on accounting measures while interest is paid in currency. A borrower with persistently high accruals is a borrower whose covenant headroom is more fragile than the ratio suggests.

**In audit risk assessment.** Auditors are required to assess the risk of material misstatement, and unusual growth in accruals relative to peers and to the company's own history is a standard analytical procedure for directing audit effort. The article on [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) covers the limits of that process.

**As a cross-check on a narrative.** The most valuable use for a non-professional is the least technical. When a company's story is unusually compelling and its reported growth unusually smooth, the accruals ratio is a two-minute test of whether the cash agrees with the story. This is the discipline described in [narrative addiction: when a good story beats the data](/blog/trading/analyst-edge/narrative-addiction-when-a-good-story-beats-the-data), applied to one number.

## When this matters to you

You do not need a data terminal or a quantitative background to use any of this. If you can find a company's income statement and its cash flow statement — both free on the SEC's EDGAR system for US filers, and in the annual report everywhere else — you can compute the whole thing in under five minutes:

1. Find **net income** at the bottom of the income statement.
2. Find **net cash provided by operating activities** at the bottom of the operating section of the cash flow statement.
3. Subtract the second from the first. That is accruals.
4. Find **total assets** on this year's and last year's balance sheets, average them, and divide.
5. Do the same for four or five competitors, and rank them.

What you learn from those five minutes is not whether to buy anything. It is something more durable: **how much of a company's reported profit is a fact and how much is a forecast.** When the answer is "mostly fact", you can lean on the earnings. When the answer is "mostly forecast" — and especially when receivables and inventory are sprinting ahead of sales — the burden of proof shifts, and the right response is not to sell, but to go and read the footnotes with a specific question in mind.

That habit compounds. It is the same discipline behind [reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings), and it is what separates reading a financial statement from being fooled by one. Sunbeam's 1997 annual report contained both the celebrated headline and the number that contradicted it, in the same document, eight months before anybody restated anything. The information was never hidden. It was just on a different page from the one everyone was reading.

*This article is educational and is not investment advice.*

## Sources & further reading

**The academic literature**

- Richard G. Sloan, "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About Future Earnings?" *The Accounting Review*, vol. 71, no. 3 (July 1996), pp. 289–315 — the original paper: the persistence result, the balance-sheet accrual definition scaled by average total assets, and the 10.4% size-adjusted first-year hedge return across 40,679 NYSE and AMEX firm-years from 1962 to 1991. [publications.aaahq.org](https://publications.aaahq.org/accounting-review/article/71/3/289/18989)
- Paul Hribar and Daniel W. Collins, "Errors in Estimating Accruals: Implications for Empirical Research," *Journal of Accounting Research*, vol. 40, no. 1 (2002), pp. 105–134 — the non-articulation problem; why mergers and acquisitions, discontinued operations and (to a lesser extent) currency translation make the balance-sheet method diverge from the cash-flow method, and the case for using net income before extraordinary items minus operating cash flow.
- Scott A. Richardson, Richard G. Sloan, Mark T. Soliman and İrem Tuna, "Accrual Reliability, Earnings Persistence and Stock Prices," *Journal of Accounting and Economics*, vol. 39, no. 3 (2005), pp. 437–485 — accruals decomposed across the whole balance sheet into working capital, net non-current operating assets and net financial assets, each with a reliability rating.
- Patricia M. Fairfield, J. Scott Whisenant and Teri Lombardi Yohn, "Accrued Earnings and Growth: Implications for Future Profitability and Market Mispricing," *The Accounting Review*, vol. 78, no. 1 (January 2003), pp. 353–371 — accruals and growth in long-term net operating assets have equivalent negative associations with next-year profitability, framing the accruals anomaly as a special case of a more general growth anomaly.
- X. Frank Zhang, "Accruals, Investment, and the Accrual Anomaly," *The Accounting Review*, vol. 82, no. 5 (October 2007), pp. 1333–1363 — the anomaly's magnitude increases with the investment information in accruals, measured by their covariation with employee growth.
- Patricia M. Dechow and Ilia D. Dichev, "The Quality of Accruals and Earnings: The Role of Accrual Estimation Errors," *The Accounting Review*, vol. 77, supplement (2002), pp. 35–59 — accrual quality measured as the residual from regressing changes in working capital on past, present and future operating cash flows.
- Jeremiah Green, John R. M. Hand and Mark T. Soliman, "Going, Going, Gone? The Apparent Demise of the Accruals Anomaly," *Management Science*, vol. 57, no. 5 (2011), pp. 797–816 — hedge returns "no longer reliably positive", attributed principally to hedge fund capital entering extreme-accrual names.
- Baruch Lev and Doron Nissim, "The Persistence of the Accruals Anomaly," *Contemporary Accounting Research*, vol. 23, no. 1 (2006), pp. 193–226 — the dissenting view that the anomaly persisted undiminished, and the evidence that accruals-related institutional trading is very small.
- R. David McLean and Jeffrey Pontiff, "Does Academic Research Destroy Stock Return Predictability?" *The Journal of Finance*, vol. 71, no. 1 (2016), pp. 5–32 — across 97 predictors, returns 26% lower out of sample and 58% lower post-publication.
- Tarun Chordia, Avanidhar Subrahmanyam and Qing Tong, "Have Capital Market Anomalies Attenuated in the Recent Era of High Liquidity and Trading Activity?" *Journal of Accounting and Economics*, vol. 58, no. 1 (2014), pp. 41–58 — rising liquidity and trading activity reducing exploitable anomaly returns.
- Eugene F. Fama and Kenneth R. French, "Dissecting Anomalies," *The Journal of Finance*, vol. 63, no. 4 (2008), pp. 1653–1678 — the accruals effect found across micro, small and big stocks; it is asset growth, not accruals, that is absent in large firms.

**The detection screens**

- Messod D. Beneish, "The Detection of Earnings Manipulation," *Financial Analysts Journal*, vol. 55, no. 5 (September/October 1999), pp. 24–36 — the eight-variable M-Score, the TATA accruals term (which carries the model's largest coefficient), and the paper's own −1.78 cutoff. The widely circulated −2.22 threshold does not appear in this paper.
- Joseph D. Piotroski, "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers," *Journal of Accounting Research*, vol. 38, supplement (2000), pp. 1–41 — the nine-signal F-Score, one signal of which scores a point when operating cash flow exceeds return on assets, both scaled by beginning-of-year total assets.

**Primary filings and enforcement actions**

- Sunbeam Corporation, Form 10-K405 for the fiscal year ended 28 December 1997, filed 6 March 1998 — the source of the \$109.4 million net earnings, −\$8.2 million operating cash flow, receivables, inventory and total-asset figures, and the \$59 million receivables securitisation disclosure. [SEC EDGAR](https://www.sec.gov/Archives/edgar/data/0000003662/000095017098000413/0000950170-98-000413.txt)
- Sunbeam Corporation, Form 10-K/A filed 12 November 1998, Note 13 "Restatement" — restated fiscal 1996 and 1997 figures. [SEC EDGAR](https://www.sec.gov/Archives/edgar/data/3662/0000950170-98-002145.txt)
- US Securities and Exchange Commission, *In the Matter of Sunbeam Corporation*, Securities Act Release No. 7976 / AAER No. 1393 (15 May 2001) — the cookie-jar reserve, bill-and-hold, guaranteed-sale and channel-stuffing allegations and the "at least \$62 million of \$189 million" finding. [sec.gov](https://www.sec.gov/enforcement-litigation/administrative-proceedings/33-7976)
- US Securities and Exchange Commission, Litigation Release No. 17001 / AAER No. 1395 (15 May 2001), *SEC v. Albert J. Dunlap et al.* [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-17001)
- US Securities and Exchange Commission, Litigation Release No. 17710 (4 September 2002) — the Dunlap and Kersh settlements, penalties and officer-and-director bars. [sec.gov](https://www.sec.gov/litigation/litreleases/lr17710.htm)
- Lucent Technologies Inc., Form 10-K405 for the fiscal year ended 30 September 1999, filed 21 December 1999 — revenue, net income, operating cash flow, receivables, inventory and total assets. [SEC EDGAR](https://www.sec.gov/Archives/edgar/data/1006240/0000950123-99-011082.txt)
- US Securities and Exchange Commission, Press Release 2004-67 / Litigation Release No. 18715 (17 May 2004) — the Lucent settlement, the approximately \$1.148 billion revenue and \$470 million pre-tax income allegations, and the \$25 million penalty for lack of cooperation. [sec.gov](https://www.sec.gov/news/press/2004-67.htm)

**Related articles in this series**

- [Accrual accounting versus cash: the gap fraud exploits](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) — the concept this article measures.
- [Reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) — the statement supplying the CFO half of every calculation here.
- [Cookie-jar reserves and big-bath accounting](/blog/trading/forensic-accounting/cookie-jar-reserves-and-big-bath-accounting) — the technique behind Sunbeam's 1996 reserves.
- [Cash flow statement manipulation and classification shifting](/blog/trading/forensic-accounting/cash-flow-statement-manipulation-classification-shifting) — how the denominator of the safest method can itself be gamed.
- [Inventory and receivables inflation: the classic red flag](/blog/trading/forensic-accounting/inventory-and-receivables-inflation-the-classic-red-flag) — the days-sales-outstanding test in depth.
- [Reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings) — where the accrual earnings are built in the first place.
- [Quality of earnings: accruals, one-offs and red flags](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) — the same toolkit from an equity-research angle.
