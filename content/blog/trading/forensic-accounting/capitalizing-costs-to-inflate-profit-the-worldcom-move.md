---
title: "Capitalizing costs to inflate profit: the WorldCom move"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "How moving ordinary expenses onto the balance sheet mechanically lifts current profit, where software and R&D capitalization is legitimate, and how WorldCom's line-cost fraud exposed the difference."
tags: ["forensic-accounting", "worldcom", "capital-expenditures", "software-accounting", "earnings-quality", "financial-statements", "fraud-detection"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Capitalizing a cost does not make it disappear; it changes the timing and the statement on which it first appears.
>
> - Expensing a cost lowers this period's operating profit. Capitalizing it records an asset first and moves the cost into depreciation or amortization later.
> - The mechanical boost is dollar-for-dollar in the current period: move $100 from expense to an asset and current pre-tax income rises by $100, before later amortization.
> - Legitimate capitalization requires an identifiable future benefit and a defensible rule for measuring and amortizing it. Ordinary network capacity, advertising, training, research, and maintenance do not become assets because management wants a better quarter.
> - The software and R&D boundary is a real gray zone, but it is not a blank cheque. Under US GAAP, the project stage and intended use matter.
> - In the SEC's WorldCom complaint, improper entries reduced reported line costs by approximately $3.8 billion across the first quarter of 2001 through the first quarter of 2002; the complaint says the entries increased capital-asset accounts instead.

The most dangerous accounting trick is often not a fake customer or a forged bank statement. It is a real invoice put in the wrong place.

The company really paid a supplier. The engineers really worked. The data center really exists. What changes is the label in the ledger: **expense today**, or **asset to be consumed later**?

That label changes the story investors read. Expenses sit on the income statement and reduce operating profit now. Assets sit on the balance sheet and are charged to expense gradually through depreciation or amortization. Cash usually leaves at the same moment either way. Profit does not.

This is the core forensic question: *does the balance sheet contain a resource that will produce probable future benefit, or is it carrying an old expense so today's income looks healthier?*

The first figure is the mental model for the entire post. The red path is the cost that reaches the income statement immediately. The blue path is a legitimate asset only when a future benefit exists; the amber charge is the later amortization that eventually reaches profit.

![Statement-line bridge showing an expense moving into an asset and later amortization](/imgs/blogs/capitalizing-costs-to-inflate-profit-the-worldcom-move-1.webp)

The diagram is deliberately mechanical. Capitalization can improve this year's profit without creating a dollar of cash. That is why the cash-flow statement, the asset roll-forward, the accounting policy note, and the business physics must be read together.

The exercise belongs beside [reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings), because capitalization is one way reported profit can become lower quality than it first appears.

## Foundations: the building blocks of capitalization

Before looking for fraud, define the vocabulary. Accounting is a translation system: it turns economic events into statement lines. A forensic reader has to know what each translation is allowed to say.

### Expense, asset, liability, equity, and profit

An **expense** is a cost recognized as reducing profit for a reporting period. Rent for a month is an expense of that month. A **revenue** is income recognized when the company has delivered what it promised under its accounting rules. **Profit**, also called net income, is revenue minus expenses and other income or costs for the period.

An **asset** is a present economic resource controlled by the company because of a past event. “Economic resource” means a right that can produce benefits: cash, a receivable from a customer, equipment, a patent, or qualifying software. An asset is not simply “something expensive.” A large invoice can still be an expense.

A **liability** is a present obligation to transfer an economic resource. **Equity** is the residual interest after liabilities are subtracted from assets. The balance-sheet identity is:

$$\text{Assets} = \text{Liabilities} + \text{Equity}.$$

If a company moves a cost from expense into an asset, the debit and credit still have to balance. The entry may increase assets and equity through higher reported profit, but it does not create cash. The accounting equation balances; the economics may still be misleading.

### Operating expense versus capital expenditure

An **operating expense**, or OpEx, is a cost of running the business in the current period: salaries for routine support, utilities, rent, advertising, and purchased network capacity are common examples. A **capital expenditure**, or CapEx, is spending that creates or improves a long-lived resource. Equipment purchased for several years of production is the familiar case.

Capitalization is the bookkeeping act of recording an eligible cost as an asset. **Depreciation** allocates the cost of a tangible asset, such as equipment, over its useful life. **Amortization** does the analogous job for an intangible asset, such as software or a patent. The allocation is not a second cash payment. It is the later recognition of a cost that was deferred.

| Question | Expense now | Capitalize first |
| --- | --- | --- |
| First statement hit | Income statement | Balance sheet |
| Current operating profit | Lower | Higher, all else equal |
| Cash paid | Usually unchanged | Usually unchanged |
| Later statement hit | None for that cost | Depreciation or amortization |
| Main forensic risk | Understatement of expenses | Inflated assets and deferred losses |

### The matching intuition, without worshipping the slogan

The **matching principle** is the idea that a cost should be recognized in the periods helped by the revenue it supports. If a machine helps produce goods for five years, charging all of its cost in the purchase month would make that month look artificially bad and later months artificially good. Depreciation spreads the cost across the useful period.

Matching is not permission to postpone every unpleasant cost. It works only when there is a real resource with a measurable consumption pattern. A cost that merely keeps the business operating is normally consumed immediately, even if management hopes it will support sales next year.

### Journal entries: the grammar of the trick

A **journal entry** is the debit-and-credit record of a transaction. A debit is an entry on the left side of a ledger account; a credit is an entry on the right. The labels are not synonyms for good and bad. Their effect depends on the account.

Suppose a company pays $100 for routine maintenance.

```journal
Correct expense entry
Dr Maintenance expense       $100
    Cr Cash                         $100
```

If the company instead calls it a long-lived asset:

```journal
Improper or unsupported capitalization
Dr Capitalized asset          $100
    Cr Cash                         $100
```

The second entry leaves current expense $100 lower. Pre-tax income is therefore $100 higher than it would have been. The balance sheet carries an asset that may not meet the definition of an asset. That is the whole current-period profit boost in two lines.

![Before-and-after journal entries for expensing versus unsupported capitalization](/imgs/blogs/capitalizing-costs-to-inflate-profit-the-worldcom-move-2.webp)

The difference is not that one entry is more sophisticated. It is that one makes a claim about future benefits. Forensic accounting tests that claim.

#### Worked example: one invoice, two income statements

Imagine a small carrier with $1,000 of revenue and $700 of other legitimate expenses. It also pays $100 for a month of purchased network capacity. The numbers are illustrative.

Under the correct treatment:

```journal
Revenue                                      $1,000
Other operating expenses                       (700)
Network capacity expense                       (100)
Operating profit                               $200
```

If the $100 is moved to a capital-asset account, the current-period statement becomes:

```journal
Revenue                                      $1,000
Other operating expenses                       (700)
Network capacity expense                         —
Operating profit                               $300
Capitalized asset on balance sheet             $100
```

The reported profit increase is $300 − $200 = $100. Cash is still $1,000 − $700 − $100 = $200 before any other cash items. No customer paid an extra dollar. The asset is a promise that a future benefit exists.

**Intuition:** capitalization can move a cost out of today's profit without moving the cash payment or improving the business.

#### Worked example: the reversal arrives later

Continue the same illustrative example. Suppose management amortizes the $100 asset evenly over 4 periods, so each period carries $25 of amortization.

In period 1, the improper treatment shows $300 of operating profit before amortization; after the $25 amortization, it shows $275. The correct treatment showed $200. The temporary profit lift is therefore $75, not $100, if the asset is amortized immediately.

Across periods 1 through 4, the capitalized cost produces $25 + $25 + $25 + $25 = $100 of amortization. The total four-period expense is eventually the same as the correctly expensed $100, assuming the asset is never impaired and the useful life is not changed.

But the timing matters. A company facing a covenant test, bonus threshold, or earnings forecast in period 1 may value the $75 lift today more than the $25 charges in each later period. That incentive is why the roll-forward matters more than a single income statement.

**Intuition:** capitalization is a timing lever; the longer the claimed useful life, the larger the current boost and the slower the later drag.

## The mechanical profit boost, statement by statement

The cleanest way to investigate capitalization is to trace one dollar through all three statements. Avoid starting with the word “fraud.” Start with the flows.

### Income statement: expense disappears, profit rises

The **income statement** reports revenue, expenses, and profit over a period. If a $100 cost is capitalized instead of expensed, current operating expenses fall by $100. If no immediate amortization offsets it, operating profit and pre-tax income rise by $100.

That is a mechanical result, not a judgment. It does not tell us whether the accounting is correct. It tells us what the entry was designed to do.

### Balance sheet: assets rise, equity follows profit

The **balance sheet** reports assets, liabilities, and equity at a point in time. The $100 capitalization entry increases assets by $100 and leaves cash $100 lower. If the asset is recorded instead of an expense, retained earnings are also $100 higher through the income statement, so the balance still balances.

The red flag is not “assets rose.” Healthy companies invest. The red flag is an asset balance that rises faster than the underlying productive capacity, or one that is not supported by project records, useful-life evidence, or future cash-generating ability.

### Cash-flow statement: classification can hide the pressure

The **cash-flow statement** explains changes in cash through operating, investing, and financing activities. Under common presentation, cash paid for an operating cost appears in operating cash flow; cash paid for property or certain long-lived assets appears in investing cash flow.

That means unsupported capitalization can do two things at once: increase accounting profit and make operating cash flow look better by pushing the cash payment into investing cash flow. Total cash does not change, but the quality of operating cash flow does.

The reconciliation from net income to operating cash flow can therefore look deceptively favorable. A forensic reader compares capital expenditures with physical additions, depreciation, asset disposals, and free cash flow. **Free cash flow** is commonly approximated as operating cash flow minus capital expenditures; it is not a universal GAAP line, but it is a useful cash discipline.

![Three-statement flow: the same cash payment changes profit, assets, and cash-flow classification](/imgs/blogs/capitalizing-costs-to-inflate-profit-the-worldcom-move-3.webp)

#### Worked example: profit up, operating cash flow up, total cash unchanged

Use the same illustrative $100 payment. Before the payment, assume cash is $500, operating cash flow is $250, and investing cash flow is negative $50.

Correct expense treatment:

```journal
Cash after payment                         $400
Operating cash flow                         $150   ($250 − $100)
Investing cash flow                         $(50)
Net change in cash                          $100
```

Capitalized treatment, assuming the payment is presented as investing cash flow:

```journal
Cash after payment                         $400
Operating cash flow                         $250
Investing cash flow                        $(150)  ($(50) − $100)
Net change in cash                          $100
```

Total cash and the $100 net change are identical. The operating cash-flow headline is $100 higher, while investing cash flow is $100 lower. A reader who looks only at operating cash flow can mistake classification for stronger cash generation.

**Intuition:** capitalization can polish both profit and operating cash flow while leaving the bank account exactly where it was.

### Ratios: the optical changes investors notice

The **operating margin** is operating profit divided by revenue. In the first illustrative statement, correct operating margin is $200 ÷ $1,000 = 20%. After unsupported capitalization, it is $300 ÷ $1,000 = 30%.

**EBITDA** means earnings before interest, taxes, depreciation, and amortization. It is a profit measure before financing costs, tax, and non-cash depreciation/amortization. Capitalizing a current operating cost can lift EBITDA if the cost never enters current operating expense. Later amortization may not reduce EBITDA at all, because amortization is excluded from the metric. This is one reason EBITDA can be especially vulnerable to cost classification.

The ratio improvement can create a feedback loop: higher margins support a higher valuation multiple, the share price supports executive confidence, and the higher price can make future financing easier. None of those effects validates the original asset.

## Where capitalization is legitimate

A forensic reader should not treat every capitalized cost as suspicious. The discipline is to distinguish an asset from an expense using evidence.

### The four questions behind a defensible asset

Ask four questions in order:

1. **What specific resource was created or acquired?** Name the machine, software module, patent, contract right, or other controlled resource.
2. **What future benefit is expected?** Show how the resource will produce revenue, reduce cost, or provide service capacity.
3. **Can the cost be measured reliably?** Payroll records, vendor invoices, time sheets, and project codes should support the amount.
4. **What is the consumption pattern?** A useful life, amortization method, impairment trigger, and retirement plan should follow from how the benefit is consumed.

If the answer is “the team was busy” or “the cost supports growth,” the evidence is incomplete. Work is not automatically an asset. A resource has to be identifiable and controlled, and its future benefit must be more than management optimism.

### Tangible assets are easier, not automatically safe

A factory machine is easier to observe than software. It has a serial number, a location, an invoice, and production output. Yet even tangible CapEx can be abused. Repairs can be folded into equipment. Routine maintenance can be called an upgrade. Idle equipment can remain on the balance sheet after its economics deteriorate.

The audit trail should connect the invoice to the asset register, the asset register to the physical site, and the asset to production or service capacity. The same asset should also appear in depreciation, insurance, maintenance, and impairment records. Broken links are more informative than a large total alone.

#### Worked example: repair or improvement?

Suppose a factory pays an illustrative $120 for a repair that returns a machine to its previous operating condition. The repair does not extend useful life or increase capacity. The defensible entry is:

```journal
Dr Repairs expense              $120
    Cr Cash                           $120
```

Now suppose a separate $120 project adds a new production module that increases output and is expected to be used for 3 years. A simplified capitalization entry is:

```journal
Dr Equipment asset               $120
    Cr Cash                           $120
```

With straight-line depreciation and no residual value, the later periodic depreciation is $120 ÷ 3 = $40 per year. The first project reduces current profit by $120; the second initially defers the cost and later recognizes $40 per year. The difference is the documented future service capacity, not the invoice's size.

**Intuition:** the accounting follows the new resource or service potential, not management's preferred label for the invoice.

### Intangible assets are harder because the resource is invisible

An **intangible asset** is a non-physical resource such as software, a patent, a license, or a customer-related contractual right. Its lack of physical form makes the evidence more judgmental. A project can be real, expensive, and strategically important while still containing large amounts of expense that must be recognized immediately.

That is why software rules separate project stages and intended use. The exact standard depends on the reporting framework and the software's purpose. This article describes the US GAAP boundary at a high level; companies using IFRS or another local framework must apply that framework's rules.

## The software and R&D gray zone

Software is where a legitimate rule can look like a loophole. A program can provide benefits for years, so some development costs are eligible for capitalization. But software work also includes research, maintenance, training, data conversion, bug fixing, and routine operations. Those activities do not all create a controlled asset.

### Internal-use software under US GAAP

For internal-use software, ASC 350-40 generally distinguishes the **preliminary project stage**, the **application development stage**, and the **post-implementation-operation stage**. FASB's September 18, 2025 explanation says current GAAP requires capitalization of qualifying development costs depending on the nature of the cost and the project stage. The IRS's summary of ASC 350-40 likewise describes application-development-stage costs as generally capitalized while other development costs are generally expensed.

In plain language:

| Stage or activity | Typical treatment under the simplified US GAAP map |
| --- | --- |
| Preliminary investigation and deciding whether to proceed | Expense |
| Application development that creates usable functionality | Qualifying costs may be capitalized |
| Training and data conversion | Usually expense |
| Routine maintenance and operation | Expense |
| Amortization after the software is ready for use | Expense over useful life |

The word **qualifying** carries the weight. Payroll has to be tied to eligible development activity, not merely to a department called “engineering.” A developer fixing production incidents is not automatically creating a new asset. A project can also move between stages, and a cloud arrangement may involve separate implementation guidance.

![Software cost decision map from preliminary work to development to operation](/imgs/blogs/capitalizing-costs-to-inflate-profit-the-worldcom-move-4.webp)

### Software sold to customers is a different question

Software developed for sale, lease, or external marketing has its own guidance, including the technological-feasibility boundary described in ASC 985-20. The IRS describes the broad pattern: costs before technological feasibility are generally R&D expense; eligible costs after feasibility and before general release may be capitalized and then amortized.

Do not collapse internal-use and external-use software into one rule. “We are a software company” is not an accounting conclusion. The purpose of the software, the stage of the project, the nature of the work, and the reporting framework all matter.

### Research and development is not synonymous with an asset

**Research and development**, or R&D, is spending to discover knowledge, design products, solve technical uncertainty, or develop new processes. R&D often creates future value in an economic sense. That does not automatically mean the accounting standards permit an asset today.

This distinction is uncomfortable but important: economic investment and accounting capitalization are not identical. A company may spend $10 million on brilliant research and still record $10 million of R&D expense. Conversely, qualifying software-development costs may be capitalized even though the project later fails. The accounting rule is a recognition rule, not a score for scientific importance.

#### Worked example: the same $300 engineering payroll across three buckets

Imagine an illustrative internal software program with $300 of engineering payroll in one month:

* $100 is spent evaluating vendors and deciding whether to build.
* $150 is spent coding a defined application feature during the application-development stage.
* $50 is spent training users and migrating old data.

Using the simplified US GAAP map, the entry pattern is:

```journal
Dr Preliminary-project expense       $100
Dr Capitalized software asset         $150
Dr Training/data-conversion expense    $50
    Cr Cash or payroll payable               $300
```

Current expense is $150, not $300. The asset is $150. If the completed feature is amortized over 3 years, the later annual amortization is $50, assuming straight-line amortization and no impairment. The treatment is not “capitalize engineering”; it is “capitalize the eligible work performed in the eligible stage.”

**Intuition:** a department name cannot turn research, training, or operations into a software asset; the project record has to do that work.

### The forensic tests for capitalized software

A reviewer should request more than a capitalization policy. Ask for:

* the project charter and approval date;
* stage-gate evidence showing when application development began;
* time sheets or automated project coding tied to specific modules;
* payroll and contractor reconciliations to the capitalized ledger;
* release or “ready for intended use” evidence;
* the useful-life analysis and amortization start date;
* defect, maintenance, and support tickets excluded from the asset;
* impairment testing when adoption, revenue, or technical feasibility weakens.

The most revealing test is often a sample from the asset ledger back to the engineer's calendar. If a developer spent the week responding to outages but the ledger says all hours created a new platform, the policy is not the problem; the operational evidence is.

## The WorldCom case: line costs put on the balance sheet

WorldCom is the named case because it makes the mechanism visible in a real company and a real statement line.

WorldCom was a telecommunications company whose **line costs** were amounts paid to other carriers for access to network capacity. Those costs were a major operating expense. They were not a machine, a fiber route controlled by WorldCom, or a software product created by the company. The SEC alleged that, from at least the third quarter of 2000 through the first quarter of 2002, senior management directed entries that concealed the true extent of line costs.

The SEC's June 26, 2002 complaint says officers and employees made entries that effectively erased approximately $941 million from line-cost expense for the fourth quarter of 2001 and correspondingly increased capital-asset accounts. The complaint also says improper line-cost entries were made from the third quarter of 2000 through the first quarter of 2002.

The SEC's later exhibit describes the total improper reduction in reported line costs as approximately $3.8 billion from the first quarter of 2001 through the first quarter of 2002, principally by capitalizing about $3.5 billion of line costs. Those are enforcement allegations and reported findings in the cited SEC materials; this post does not treat every allegation as an independent court finding.

![WorldCom timeline from line-cost pressure to capitalization entries and restatement](/imgs/blogs/capitalizing-costs-to-inflate-profit-the-worldcom-move-5.webp)

### The statement lines the SEC compared

The SEC's earlier complaint provides a compact before-and-after comparison for 2001. WorldCom's 2001 Form 10-K reported line costs of $14.739 billion and income before income taxes and minority interests of $2.393 billion. The complaint says the true line costs were approximately $17.794 billion and the company suffered a loss of approximately $662 million.

The difference in line costs is $17.794 billion − $14.739 billion = approximately $3.055 billion. The difference between reported pre-tax income of $2.393 billion and an approximately $662 million loss is approximately $3.055 billion. The statement-line arithmetic lines up.

For the first quarter of 2002, the same SEC complaint says WorldCom reported line costs of $3.479 billion and pre-tax income of $240 million. It says the true line costs were approximately $4.276 billion and the company suffered a loss of approximately $557 million. The line-cost difference is approximately $797 million, and the pre-tax swing is $240 million − (−$557 million) = approximately $797 million.

Those are not hypothetical examples. They are dated, attributed figures from the SEC complaint. They show exactly how a cost reclassification can turn a reported profit into a reported loss without requiring fictitious revenue.

#### Worked example: reconstructing WorldCom's 2001 statement line

Start with the SEC-reported numbers:

```journal
Reported 2001 line costs                         $14.739bn
SEC complaint's approximate true line costs       $17.794bn
Difference                                        $3.055bn
```

Now follow the pre-tax line:

```journal
Reported income before tax and minority interests  $2.393bn
Less approximate line-cost correction              (3.055bn)
Reconstructed result                                $(0.662bn)
```

The subtraction produces an approximately $662 million loss, matching the SEC complaint's description. The figure is a statement-line bridge, not a claim that every dollar of the company's entire accounting was corrected by this one line; it isolates the line-cost comparison described by the SEC.

**Intuition:** when the expense reduction and the profit overstatement are the same dollars, the fraud can be understood as a classification bridge before it is understood as a story about personalities.

#### Worked example: the first-quarter 2002 bridge

Again use the dated figures in the SEC complaint:

```journal
Reported Q1 2002 line costs                     $3.479bn
Approximate true Q1 2002 line costs              $4.276bn
Approximate correction                            $0.797bn
```

The reported pre-tax income was $240 million. Subtracting the approximate $797 million correction gives $240 million − $797 million = approximately a $557 million loss. The line-cost correction therefore explains the entire reported-to-approximate pre-tax swing in the complaint's comparison.

The lesson is not that every company with rising CapEx is WorldCom. The lesson is that a forensic analyst can ask whether the asset additions and the profit change are connected by the same dollars.

**Intuition:** a single expense line can carry enough weight to reverse the sign of quarterly profit when margins are thin.

#### Worked example: why the quarter's percentage matters

The SEC exhibit says that $818 million of line costs were improperly capitalized in the first quarter of 2002 and that this allowed WorldCom to report pre-tax income of $240 million instead of a $578 million pre-tax loss in that particular exhibit's comparison. The numbers are close in scale to the complaint's approximately $797 million bridge, but the documents use different comparisons and rounding; do not silently merge them.

Using the exhibit's figures, the pre-tax swing is $240 million − (−$578 million) = $818 million. The exhibit therefore ties the specific $818 million capitalization amount to the reported pre-tax result. The nearby complaint uses approximately $797 million for the line-cost difference and approximately a $557 million loss. A careful reader preserves the source distinction rather than manufacturing a single false-precision number.

**Intuition:** source documents can describe related periods with different rounding or scopes; reconcile the method before reconciling the headline.

### Why the entries were not ordinary CapEx

The SEC complaint says the line costs were among WorldCom's major operating expenses and that the treatment was not in conformity with GAAP. The economic reason is straightforward: payments to other carriers for current network access were consumed in providing telecommunications service. Calling those payments “capital assets” did not give WorldCom control of the carriers' networks or a new resource it could use independently for years.

This is the precise boundary between a real network investment and a line-cost expense. Building a fiber route might create a controlled asset. Buying capacity from another carrier for a period is a service consumed in that period. Both can be described casually as “network spending”; only one is automatically a capital asset.

![A red-flag dashboard contrasting physical network investment with purchased line-cost service](/imgs/blogs/capitalizing-costs-to-inflate-profit-the-worldcom-move-6.webp)

### The investigation lesson: ratios can reveal the behavior

The SEC exhibit says WorldCom's line-cost expense-to-revenue ratio would typically have exceeded 50% had it not capitalized line costs. This is a dated, attributed statement about the exhibit's analysis, not a universal telecom benchmark.

That ratio is useful because management's entries did not change the physical network service consumed. If revenue slowed but purchased capacity remained high, a normal cost ratio would show pressure. Capitalization suppressed the ratio and made the operating margin look more stable than the underlying economics.

The forensic workflow is therefore:

1. identify a cost that should move with business activity;
2. compare its ratio with revenue and physical drivers;
3. inspect whether unusual entries move it into capital accounts;
4. reconcile the capital account to useful assets, depreciation, and cash;
5. read the accounting policy and subsequent restatement together.

The ratio is a signal, not proof. A company can have real fixed costs, long-term contracts, or a temporary utilization shock. The evidence becomes strong when several independent signals point in the same direction.

## How to spot capitalization that deserves investigation

The right question is not “did CapEx rise?” It is “did the balance-sheet asset grow in a way the business could plausibly absorb?”

### The capitalized-cost roll-forward

An asset roll-forward reconciles beginning balance, additions, amortization or depreciation, disposals, impairment, and ending balance. A simple roll-forward is:

$$\text{Ending asset} = \text{Beginning asset} + \text{Additions} - \text{Depreciation or amortization} - \text{Disposals} - \text{Impairment}.$$

Every term should be defined. **Additions** are new costs recorded in the asset account. **Disposals** remove assets sold or retired. **Impairment** is a write-down when the asset's carrying amount is no longer recoverable under the relevant rules.

If additions are large but depreciation is unusually small, ask whether assets are being put into service late, whether useful lives are stretched, or whether costs are being parked in construction-in-progress. A long construction phase may be legitimate; an indefinite parking lot for completed projects is not.

### Red flag 1: CapEx rises while physical capacity does not

Compare capital additions with the company's operational unit: route miles, data-center capacity, stores, subscribers, production units, or software releases. There is no universal “correct” spend per unit, but a sudden divergence deserves a question.

For a carrier, purchased access capacity should show up in operating expense, while owned network build should show up in property, plant, and equipment. For a SaaS business, capitalization of qualifying internal-use software should connect to releases, modules, and amortization. “Technology investment” is too broad a category for a conclusion.

### Red flag 2: Depreciation and amortization lag the asset boom

When assets grow, future depreciation or amortization should eventually grow. The lag can be reasonable if construction is not yet complete. It becomes concerning when management repeatedly extends useful lives, delays the ready-for-use date, or uses residual values that are difficult to support.

Useful life is a judgment, but it has a physical and commercial anchor. A customer-facing app that is replaced every two years is difficult to justify as a ten-year asset without strong evidence. A utility pole may have a different pattern. The question is always “what is being consumed, and how fast?”

### Red flag 3: operating cash flow improves only because CapEx classification changes

Watch for a gap between net income, operating cash flow, and free cash flow. If operating cash flow rises while total cash does not improve and investing cash outflows expand, the quality of the rise matters.

The forensic question is not whether free cash flow is a perfect metric. It is whether cash paid for recurring operations is being moved into an investment bucket. Compare the cash-flow classification with the nature of the invoice.

That comparison is the same cash-versus-accrual discipline developed in [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits); here, the distortion is a classification bridge rather than a timing difference in receivables or payables.

### Red flag 4: “adjusted” metrics exclude the later charge

If a company capitalizes software and then reports adjusted EBITDA before amortization, the later charge may be excluded twice: first because the cost avoided current operating expense, then because amortization is excluded from EBITDA. That does not prove manipulation, but it makes the reconciliation essential.

Build a bridge from GAAP operating income to the company's non-GAAP measure. Label every adjustment. Ask whether the adjustment removes a one-time item or removes a recurring cost of delivering the service.

### Red flag 5: accounting policy changes follow performance pressure

Changes in capitalization thresholds, project stage definitions, useful lives, or impairment assumptions can be legitimate responses to a new business model. They can also arrive when earnings are under pressure. Timing is not proof, but it is a reason to read the policy note, auditor communication, and management discussion with more attention.

#### Worked example: a simple forensic ratio screen

Suppose an illustrative company reports:

| Period | Revenue | Capital additions | Depreciation/amortization | Operating cash flow |
| --- | ---: | ---: | ---: | ---: |
| Year 1 | $1,000 | $100 | $80 | $180 |
| Year 2 | $1,100 | $220 | $85 | $260 |

Revenue grows 10%, while additions grow 120% from $100 to $220. Depreciation grows only $5, from $80 to $85. That is not evidence of wrongdoing; Year 2 could contain a genuine new data center or platform. It is a prompt to ask what $120 of incremental additions bought and when it will enter service.

If the physical business has no new capacity, the claim becomes harder to defend. If the project records show a platform released late in Year 2, the depreciation lag may be reasonable. A screen narrows the question; it does not answer it.

**Intuition:** a ratio is a smoke alarm, not a conviction; operational evidence tells you whether there is a fire.

#### Worked example: useful life changes the earnings profile

Assume an illustrative $600 software asset with no residual value. Under a 3-year life, straight-line amortization is $600 ÷ 3 = $200 per year. Under a 6-year life, it is $600 ÷ 6 = $100 per year.

In the first year, the longer life increases reported pre-tax income by $100 relative to the shorter-life policy. Over the full life, both policies recognize $600 if the asset remains useful for the full period. But the 6-year policy also carries more asset value after Year 3, when the 3-year policy would be fully amortized.

The evidence must come from expected use, obsolescence, contract terms, release cadence, and historical replacement cycles. “The asset still exists” is not enough; an obsolete codebase can exist physically while no longer producing benefits.

**Intuition:** useful life is an earnings-timing assumption, so a small change can be material even when the total cost never changes.

## What auditors, boards, and analysts should ask

### Questions for management

Ask management to explain the asset in plain language. What can the company do after spending the money that it could not do before? Who controls the resource? When was it ready for use? Which employees' hours were included? Which activities were excluded?

Ask for a bridge from the general ledger to the footnote. The total in the accounting policy should reconcile to additions in the asset roll-forward. The roll-forward should reconcile to the cash-flow statement and, where relevant, to the fixed-asset register or software project system.

The best answers are specific: project names, release dates, capacity metrics, approved budgets, and sample invoices. The weak answers are adjectives: transformational, strategic, platform, innovation, and recurring investment.

### Questions for the audit committee

The audit committee should ask whether the capitalization policy creates incentives around quarterly targets. It should also ask how internal audit samples time sheets, how impairments are triggered, and whether the external auditor tested the largest additions back to source evidence.

The committee should separate the policy question from the execution question. A policy can be reasonable while employees miscode expenses. A policy can also be written so broadly that every engineering hour becomes “development.” Both deserve remediation, but the controls differ.

### Questions for the investor

An investor can make a compact checklist:

| Check | What to compare |
| --- | --- |
| Asset growth | Additions versus revenue and physical capacity |
| Expense mix | Operating expense ratios before and after policy changes |
| Amortization | Later charges versus prior additions |
| Cash classification | Operating cash flow versus investing cash flow |
| Software evidence | Stage gates, release dates, eligible payroll |
| Failure response | Impairments, restatements, and policy changes |

Do not punish investment merely because it is large. A high-growth company may rationally invest ahead of revenue. The test is whether the accounting records a resource and whether later economics confirm it.

## Common misconceptions

### “Capitalizing a cost is always fraud.”

No. Capitalization is necessary for many tangible assets and permitted for some software-development costs. The problem is unsupported capitalization: assigning an asset label when the cost was consumed immediately or when future benefit cannot be demonstrated.

### “It is non-cash, so it does not matter.”

Depreciation and amortization are non-cash in the period recognized, but the original cash payment mattered. More importantly, capitalization changes the timing of profit, operating cash flow presentation, asset balances, covenants, and valuation ratios. Non-cash does not mean irrelevant.

### “Cash flow cannot be manipulated this way.”

Total cash is harder to change with a classification entry, but operating cash flow can be made to look better if a payment is moved from operating to investing cash flow. Read the total change in cash and the classification together.

### “Every engineer's salary can be capitalized.”

No. The project stage, intended use, eligible activity, and documentation matter. Research, training, data migration, maintenance, and support are not automatically qualifying development costs.

### “A big asset balance proves growth.”

An asset balance proves that the ledger carries a claim. It does not prove that customers value it, that it is usable, or that it will earn a return. Impairment is the accounting system's admission that the claim may have been too optimistic.

### “WorldCom proves every telecom company was doing this.”

No. WorldCom is a named case with SEC allegations and dated statement-line comparisons. It teaches a mechanism. It is not evidence that unrelated companies used the same entries.

## How it shows up in real markets

### WorldCom: a service cost presented as an asset

The WorldCom case is the clearest example because the cost had a plain economic identity: line costs paid for network access from other carriers. The SEC complaint attributes the entries to senior management and describes the reduction of line-cost expense and increase in capital-asset accounts. The company could improve reported income without inventing a subscriber or a dollar of revenue.

The lesson for markets is that investors should not stop at revenue growth. A business can report a convincing top line while its unit economics deteriorate. The line-cost-to-revenue ratio was the physical clue: the company needed the cost to remain low on the page even as the network service was consumed.

### Software companies: legitimate complexity, real judgment

Software accounting is not a WorldCom allegation by default. It is a recurring area of judgment because a modern product mixes research, coding, maintenance, cloud implementation, training, and customer support. The right response is not to expense every software dollar blindly. It is to read the capitalization policy and test it against project behavior.

FASB's 2025 update announcement notes that current GAAP requires capitalization of internal-use software development costs depending on the cost's nature and project stage. The announcement itself also says the standard was being updated to address changes in software development methods. That is a reminder that standards evolve because real workflows evolve; it is not permission to ignore the current rule.

### A capital-heavy manufacturer: the physical corroboration test

For a manufacturer, capitalized equipment should leave physical traces: installed machinery, commissioning records, production output, maintenance contracts, and depreciation. A company that reports $600 of illustrative equipment additions but no new capacity, no commissioning, and no increase in depreciation has an evidence gap.

The market implication is subtle. Early capitalization can make margins look weak if the asset is immediately depreciated, or strong if it is parked in construction-in-progress. The analyst should follow the asset into service, not infer quality from one year's margin.

### A cloud migration: implementation cost versus service cost

Cloud arrangements can contain implementation activities and ongoing service fees. The service contract itself does not turn every invoice into an owned software asset. The reader must identify what the company controls, what functionality was created, and what costs are recurring access or support.

This is why “technology spend” is a poor single line for comparison. Two companies can spend the same $300 illustrative amount while one buys a controlled internal tool and the other buys a year of hosted access. Their economics and accounting can differ without either company being dishonest.

### R&D-heavy biotech: economic investment without a balance-sheet asset

Biotech research can be extremely valuable but still be expensed under the applicable accounting rules. Investors therefore have to distinguish accounting conservatism from poor economics. A large R&D expense may be the cost of building an option on a future product, not proof that management destroyed value.

The forensic risk goes the other way when management tries to make uncertain research look like a completed asset. The more uncertain the technical and commercial outcome, the stronger the need for evidence before capitalization.

## A repeatable forensic workflow

Start with the reported statements. Mark every asset category that grew materially and every expense ratio that improved unusually. Then write down the economic driver for each cost.

Next, retrieve the accounting policy note. Identify the capitalization threshold, eligible project stages, useful lives, amortization method, and impairment triggers. Compare the policy with the ledger and operational data.

Then trace a sample. Select the largest additions and a random sample of smaller additions. Follow each from the balance sheet to the subledger, invoice, payroll record, project approval, completion evidence, and later amortization.

Finally, build the counterfactual. What would operating profit, operating margin, EBITDA, operating cash flow, and free cash flow look like if the questionable costs had been expensed? Do not label the result as a restatement unless the evidence supports that conclusion. Call it an analytical adjustment or sensitivity.

![Forensic workflow from statement scan to source documents, operational corroboration, and counterfactual](/imgs/blogs/capitalizing-costs-to-inflate-profit-the-worldcom-move-7.webp)

#### Worked example: the counterfactual earnings bridge

Suppose an illustrative company reports revenue of $2,000, operating profit of $300, and $120 of costs capitalized during the year. Assume the analyst's evidence suggests all $120 should have been expensed and that no current amortization was recorded.

Reported operating margin is $300 ÷ $2,000 = 15%. The analytical correction is:

```journal
Reported operating profit                    $300
Less questionable capitalization              (120)
Adjusted operating profit                    $180
Adjusted operating margin                    9%
```

The $120 is not automatically a fraud loss. It is the amount that would move into current expense if the evidence says the costs were consumed now. The analyst should separately consider taxes, any legitimate future asset, and whether the entries were already corrected.

**Intuition:** the counterfactual turns a vague concern about “aggressive accounting” into a measurable question about margin and asset quality.

## When this matters to you

If you read a company's earnings release, do not ask only whether revenue beat the forecast. Ask what happened to operating expenses, capital additions, amortization, and operating cash flow at the same time.

If you work in finance, the practical safeguard is a project-level trail: approved stage gates, time coding, exclusion rules, useful-life evidence, and a review of completed projects. If you work in engineering, accurate project labels protect you as much as they protect the statements. A ticket marked “production support” should not become an asset because a quarterly target is tight.

If you are learning forensic accounting, practice with the three statements. Take one cost, write the correct journal entry, write the aggressive alternative, and trace both through profit, assets, cash-flow classification, and later amortization. The exercise is small, but it reveals the entire mechanism.

The main discipline is simple: never confuse money spent with an asset created. Capitalization is justified by a controlled future resource, not by the manager's need for a better number today.

## The deeper mechanics: timing, impairment, and incentives

The simple examples assume that an asset is amortized smoothly and never fails. Real accounting becomes more revealing when those assumptions are stressed. The asset can be placed in service late, written down suddenly, sold, abandoned, or kept alive through optimistic estimates. Each outcome creates a different pattern in the statements.

### Construction in progress can be a legitimate waiting room

**Construction in progress** is an asset account for qualifying costs of a project that is not yet ready for its intended use. It prevents depreciation from beginning before the resource can provide service. A new data center, factory, or internally developed platform may spend months in this account.

The forensic issue is not the existence of construction in progress. It is the absence of a credible completion event. Ask what remains to be done, who accepts the project, and what operational evidence will start depreciation or amortization. If a project is repeatedly “almost ready” while the company continues to capitalize payroll and vendor costs, the account may be acting as a storage drawer for expenses.

### Impairment is the moment optimism meets evidence

**Impairment** is a write-down when an asset's carrying amount is no longer supported by expected benefits under the relevant accounting rules. It is often non-cash at the moment of recognition, but it is economically important: it admits that earlier capitalization or valuation assumptions were too high.

An impairment test is not a license to wait until failure is undeniable. Product cancellation, falling user adoption, a lost customer, a technology replacement, or a project budget that has doubled can be indicators that the asset deserves scrutiny. Understated impairment can keep old capitalization in the balance sheet long after the benefit has gone.

#### Worked example: the abandoned project

Assume an illustrative $240 software asset was capitalized over two years. The company expected a three-year useful life, so it records $80 of annual amortization. At the start of Year 3, a replacement platform makes the old software unusable, and no residual value is expected.

Before considering impairment, the carrying amount after two years is:

```journal
Original asset                              $240
Less Year 1 amortization                     (80)
Less Year 2 amortization                     (80)
Carrying amount at start of Year 3           $80
```

If the old platform has no future benefit, the remaining $80 should not stay on the balance sheet. A simplified entry is:

```journal
Dr Impairment loss                            $80
    Cr Software asset                             $80
```

The failure is not that the company ever had software. The question is whether the evidence justified the remaining $80 after the replacement decision. Delaying the write-down would preserve current profit but make the balance sheet less truthful.

**Intuition:** impairment is the accounting test that asks whether yesterday's asset still exists economically today.

### Incentives concentrate at thresholds

Capitalization pressure is strongest when a small change moves a company across a threshold: a debt covenant, a bonus target, a forecast range, or a public earnings expectation. The size of the entry can be less important than the size of the resulting decision.

For a company with $100 million of pre-tax profit, a questionable $10 million capitalization is 10% of reported pre-tax profit. For a company with $1 billion of profit, the same $10 million is 1%. Materiality is therefore both quantitative and qualitative. A smaller entry can matter if it changes a loss into a profit or prevents a covenant breach.

The analyst should read the incentive documents and timing: was the entry posted in the last days of a quarter, did it reverse early in the next period, and did the same project receive a different treatment after the target was met? Reversals are not automatically improper, but they are highly testable.

### Capitalization changes the denominator as well as the numerator

Profit ratios are not the only optics. Capitalization increases assets and equity, so return on assets and return on equity may move in the opposite direction from operating margin. A company can report a higher margin and a lower return on invested capital because the denominator has been enlarged by the new asset.

**Return on invested capital**, or ROIC, compares operating profit after tax with the capital invested in the business. Definitions vary, but the forensic intuition is stable: if management moves costs into invested capital, current operating profit may rise while the capital base also rises. The result can hide a weaker productivity trend rather than reveal a better one.

The analyst should therefore look at the numerator and denominator together. Do not accept “margin up” as a complete quality statement. Ask whether sales, cash generation, capacity, and returns on the added asset moved with it.

For a broader balance-sheet reading sequence, pair this test with [reading the balance sheet: what companies hide here](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here). The asset roll-forward is where a capitalization concern becomes testable.

### The tax effect is real but not a defense

Accounting income and taxable income are not always identical. A company can capitalize an item for books while tax rules treat it differently, creating a **deferred tax** balance. Deferred tax accounting recognizes that book and tax timing differences may reverse later.

That timing can complicate the profit bridge, but it does not make unsupported capitalization correct. A tax benefit, a book asset, or a deferred tax asset cannot substitute for evidence that the underlying cost created a qualifying resource. For an analytical adjustment, state whether the correction is before or after tax and avoid implying a tax conclusion that the filing does not support.

#### Worked example: before-tax and after-tax presentation

Suppose an illustrative $100 cost is improperly capitalized and the relevant tax rate is assumed to be 25% solely for this arithmetic example. The current pre-tax profit lift is $100. The current tax expense may rise by $25 if the book-tax rules treat the lift as taxable income, leaving a net-income lift of $75. If tax treatment differs, a deferred tax item may arise instead.

The forensic correction should begin with the pre-tax statement line because that is the mechanical operating issue. Only after the book-tax difference is verified should an analyst convert the bridge to net income. A neat after-tax number built on an unverified tax assumption is false precision.

**Intuition:** tax accounting can change the size and timing of the profit effect, but it cannot turn a nonexistent asset into a real one.

### Why later growth can hide the first-period trick

If revenue grows after capitalization, management can say the asset was justified because the business eventually expanded. That is hindsight. The recognition question is what evidence existed when the cost was recorded, not whether the company later succeeded.

The reverse is also true. A failed project does not prove the original capitalization was improper if the project met the rule and was supported when built. Accounting requires estimates under uncertainty. The forensic task is to test whether the estimates were reasonable and consistently applied at the date of recognition.

This is why dated workpapers matter. Preserve the approval date, forecast, technical feasibility evidence, stage classification, and expected useful life. Do not evaluate a 2022 capitalization only with a 2026 outcome and call the outcome proof.

### A compact evidence hierarchy

When evidence conflicts, rank it by proximity to the resource:

| Evidence | What it can establish | Limitation |
| --- | --- | --- |
| Invoice and payment | A real cash transaction occurred | Not what the money purchased economically |
| Contract and purchase order | Rights and obligations | May not prove the work was an asset |
| Project records and time coding | Nature and stage of work | Can be miscoded or changed later |
| Release, commissioning, or production data | Resource became usable | Does not prove every prior cost qualified |
| Customer, utilization, or cost savings data | Benefits were realized | Later success cannot retroactively justify all entries |
| Impairment and disposal records | Whether benefits persisted | Often arrives after the original judgment |

No one document settles the question. A defensible balance-sheet asset is a chain whose links agree.

## Sources & further reading

- [SEC v. WorldCom, Inc. complaint](https://www.sec.gov/litigation/complaints/comp17829.htm), filed June 26, 2002. Primary enforcement complaint describing the line-cost entries, including the approximately $941 million fourth-quarter 2001 entry.
- [SEC v. WorldCom, Inc. complaint](https://www.sec.gov/litigation/complaints/complr17588.htm), dated 2002. Primary complaint containing the reported and approximate true line-cost and pre-tax statement lines for 2001 and Q1 2002.
- [SEC exhibit describing WorldCom's accounting](https://www.sec.gov/Archives/edgar/data/723527/000093176303001862/dex991.htm), filed 2003. Exhibit describing approximately $3.8 billion of reported line-cost reductions and the $818 million Q1 2002 capitalization comparison.
- [FASB: targeted improvements to internal-use software guidance](https://fasb.org/news-and-meetings/in-the-news/fasb-issues-standard-that-makes-targeted-improvements-to-internal-use-software-guidance-423046), September 18, 2025. Official overview of current US GAAP project-stage treatment and the ASU 2025-06 update.
- [IRS: software and ASC 350-40 FAQ](https://www.irs.gov/businesses/corporations/faqs-irc-41-qres-and-asc-730-lbi-directive-2017), accessed August 4, 2026. Official summary of internal-use software characteristics and the distinction between ASC 350-40 and R&D guidance.
- [FASB ASU 2025-06 PDF](https://asc.fasb.org/layoutComponents/getPdf?fileName=ASU+2025-06.pdf&isSitesBucket=true), 2025. Official standard update and background discussion for internal-use software accounting.

This is an educational framework, not individualized investment or accounting advice. Apply the reporting framework that governs the company and consult the primary filings and standards.
