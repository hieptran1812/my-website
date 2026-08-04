---
title: "Hidden liabilities: leases, guarantees, and contingencies"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "How to reconstruct obligations that sit in lease notes, guarantee disclosures, litigation reserves, and pension footnotes—and how ASC 842 and IFRS 16 changed the lease problem."
tags: ["forensic-accounting", "financial-statements", "leases", "guarantees", "litigation", "pensions", "balance-sheet", "risk-analysis"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 30
---

> [!important]
> **TL;DR** — A clean balance sheet can still sit on top of a large stack of contractual and uncertain obligations.
>
> - Before ASC 842 and IFRS 16, many operating leases were expensed through rent while the remaining payment stream lived in a footnote; today, most leases longer than 12 months create a right-of-use asset and lease liability, subject to exemptions.
> - A guarantee is not automatically debt, and a lawsuit is not automatically a reserve. The key questions are who must pay, when payment becomes probable, and whether the amount can be estimated.
> - Reconstruct obligations in layers: committed cash, present value, probability-weighted exposure, and the cash resources that can actually meet it.
> - As a dated reality check, Amazon’s 2018 SEC filing disclosed $26.666 billion of future operating-lease payments and $3.4 billion of 2018 operating-lease rent expense, before its 2019 accounting change.

You can read a company’s headline debt, subtract its cash, and still miss the bill that matters most. The obligation may be a warehouse lease, a parent guarantee for a subsidiary, a product-liability case, or a pension promise made decades ago. None of these is mysterious once you ask the right question: *what future cash claim does this sentence in the notes represent?*

The accounting presentation matters because ratios are built from presentation. A rent expense can look like an ordinary operating cost. A lease liability looks like financing. A litigation reserve can look precise even though the loss distribution is wide. A pension deficit can be a small balance-sheet number today but a large contribution schedule tomorrow. The forensic task is not to call every footnote “hidden debt.” It is to translate each footnote into an obligation, a timing profile, and a plausible stress range.

![A company’s reported balance sheet is only the visible layer; the footnotes reveal committed, conditional, and estimated claims on future cash.](/imgs/blogs/hidden-liabilities-leases-guarantees-and-contingencies-1.webp)

The mental model is a funnel. Start with what is legally or contractually possible, narrow it to what is expected, then discount or probability-weight it, and finally compare it with cash generation and liquidity. That discipline keeps us from treating a $100 million lease payment, a $100 million maximum guarantee, and a $100 million litigation reserve as three equivalent liabilities.

## Foundations: the building blocks

### What “hidden” means in a financial statement

“Hidden” does not mean illegal or omitted. It means economically important information that is not obvious from the three primary statements: the income statement, balance sheet, and cash-flow statement. The information may be in a note, in a contractual-obligations table, in a pension reconciliation, or in a legal-proceedings section. Sometimes it is recognized on the balance sheet but buried in an aggregated line such as “other liabilities.”

We need four distinctions:

| Layer | Plain-English question | Typical evidence | Forensic output |
| --- | --- | --- | --- |
| Committed cash | What must be paid if the contract runs as written? | Lease maturity table, debt schedule, purchase contract | Undiscounted payment stream |
| Recognized liability | What amount has accounting recorded today? | Balance sheet and note reconciliation | Carrying amount |
| Conditional exposure | What may be paid if an event occurs? | Guarantee, lawsuit, tax dispute, indemnity | Scenario range and probability |
| Funding requirement | What cash must be contributed to keep a promise? | Pension funded status, covenant, liquidity note | Timing of cash drain |

The same obligation can appear in more than one layer. A lease has committed payments, a present-value liability, and future interest. A pension has an estimated benefit obligation, plan assets, and future contributions. A guarantee can be a zero-dollar recognized liability and still be a serious credit risk.

### Present value, probability, and notional

The *present value* of a payment is its value today after allowing for time and a discount rate. A dollar due in three years is not normally valued like a dollar due tomorrow because cash today can be invested, and because risk affects the rate. A simple explanatory abstraction is:

$$PV = \sum_{t=1}^{T} \frac{C_t}{(1+r)^t}$$

Here, $C_t$ is the cash payment in period $t$, $r$ is the discount rate per period, and $T$ is the number of periods. This is a teaching abstraction, not a claim that every accounting measurement uses one identical rate or one identical convention.

*Notional* means the stated amount used to calculate a guarantee, derivative, or other obligation; it is not necessarily the amount that will change hands. *Exposure* is the amount that could actually be lost after recoveries, collateral, caps, and probability. A $100 million guarantee may have a $100 million notional but a lower expected loss—or a much larger strategic consequence if it causes the parent to support the subsidiary.

![Undiscounted payment totals, present value, and expected loss are different measurements of the same obligation.](/imgs/blogs/hidden-liabilities-leases-guarantees-and-contingencies-2.webp)

#### Worked example: turning rent into a present-value liability

Suppose a company signs a three-year lease with payments of $100 at the end of each year. Assume, for illustration, a 10% annual discount rate.

1. Year 1 present value: $100 ÷ 1.10 = $90.91.
2. Year 2 present value: $100 ÷ 1.10² = $82.64.
3. Year 3 present value: $100 ÷ 1.10³ = $75.13.
4. Total undiscounted payments: $100 + $100 + $100 = $300.
5. Present value at commencement: $90.91 + $82.64 + $75.13 = $248.68, rounded to $249.

The journal-entry shape at commencement is:

```
Dr Right-of-use asset             $249
    Cr Lease liability                         $249
```

The future cash bill is still $300. The $51 difference is not free money; it is the time-value component released through interest and the reduction of the liability. The intuition is simple: a payment schedule is a cash fact, while a lease liability is a discounted measurement of that fact.

### Recognition is a threshold, not a measure of importance

Accounting recognition asks whether an amount meets a rule for entering the statements. It does not ask whether the amount is irrelevant when it fails the rule. This is why a reasonable analyst reads disclosure even when the balance sheet shows zero.

For a U.S. loss contingency under ASC 450, the classic recognition test is whether the loss is *probable* and the amount is *reasonably estimable*. The SEC’s Staff Accounting Bulletin guidance also says that a material reasonably possible loss beyond an accrual should be disclosed, including an estimated range when one can be made, or a statement that the estimate cannot be made. A company may therefore disclose a large exposure without recording a liability.

IFRS uses different terminology and detailed requirements, but the analytical habit is the same: identify the present obligation, assess likelihood, estimate the amount, and read what remains outside recognition. “No accrual” is not the same as “no risk.”

## 1. The operating-lease era: why the footnote mattered

Before the new lease models, a lessee generally classified a lease as either operating or finance. The operating version produced a straight-line rent expense and did not put the full future payment obligation on the balance sheet. A finance lease looked more like borrowing: an asset and obligation were recognized.

That split created an obvious analytical problem. Two retailers could operate identical stores. Retailer A could own the buildings with mortgage debt; Retailer B could lease them. Retailer B might report less debt and fewer assets, even though both had a long, hard-to-avoid claim on store cash flow. The income statement did not make the difference disappear; the balance-sheet presentation did.

Amazon’s 2018 Form 10-K is a useful dated example. At December 31, 2018, Amazon reported $26.666 billion of future operating-lease payments in its contractual-commitments table: $3.127 billion due in 2019, $3.070 billion in 2020, $2.775 billion in 2021, $2.473 billion in 2022, $2.195 billion in 2023, and $13.026 billion thereafter. The same filing reported operating-lease rent expense of $3.4 billion for 2018, compared with $2.2 billion in 2017 and $1.4 billion in 2016. These are from Amazon’s SEC filing, not estimates.

The table is not a present value and it is not a debt number. It is a map of contractual cash. It also excludes or separates other commitments and may not capture every economic dependence on property. But it tells the analyst that a rent expense of $3.4 billion sat on top of a much longer payment runway.

#### Worked example: the rent-to-debt adjustment

Imagine two otherwise identical companies with $500 of operating profit before rent, $100 of annual rent, and $50 of depreciation on owned assets. Company A owns its premises and has $400 of debt. Company B leases the premises and reports $400 of debt, $100 of rent, and no lease liability under the old operating-lease presentation.

1. Company A’s reported operating profit after depreciation is $500 − $50 = $450.
2. Company B’s reported operating profit after rent is $500 − $100 = $400.
3. A rough lease-adjusted EBITDA-style comparison adds rent back for both companies, giving Company A $500 and Company B $500.
4. If the analyst capitalizes three years of $100 rent at 10%, the illustrative lease debt is $249, using the prior example’s present-value calculation.
5. Lease-adjusted debt becomes Company B’s $400 reported debt + $249 lease liability = $649, versus Company A’s $400 mortgage debt.

This is not a universal rating-agency formula. It is a transparent reconstruction. The intuition is that operating profit and debt ratios can change when the same economics are reclassified, so comparison requires putting owned and leased capacity on a common footing.

![The old operating-lease presentation showed rent expense while the analyst reconstructed the payment stream; ASC 842 and IFRS 16 move most of that reconstruction onto the balance sheet.](/imgs/blogs/hidden-liabilities-leases-guarantees-and-contingencies-3.webp)

## 2. ASC 842 and IFRS 16: the fix, and what it did not fix

The standards changed the starting point. IFRS 16 is effective for annual periods beginning on or after January 1, 2019. It uses a single lessee model and requires recognition of a right-of-use asset and lease liability for leases longer than 12 months, unless the low-value exemption applies. The IFRS Foundation’s 2019 explanation estimated that listed companies worldwide had around $3 trillion of future lease payments that had not been recognized on balance sheets under the previous requirements.

U.S. GAAP’s ASC 842 likewise brought operating-lease right-of-use assets and liabilities onto the balance sheet for public companies adopting in 2019. The SEC filing of Target’s first-quarter 2019 report describes its January 1, 2019 adoption and the recognition of operating- and finance-lease right-of-use assets and liabilities.

The central entry is familiar:

```
Dr Right-of-use asset             $PV of lease payments
    Cr Operating lease liability               $PV of lease payments
```

After commencement, the liability is reduced by payments and increased by imputed interest. The right-of-use asset is reduced by amortization or a single lease cost, depending on the framework and classification. A balance-sheet reader should therefore reconcile three things: opening liability, new leases and remeasurements, cash payments, and closing liability.

The fix is not complete visibility. Short-term leases, low-value leases, variable payments, extension options, termination options, leases not yet commenced, and judgments about whether renewal is reasonably certain can remain outside the headline recognized amount or require separate disclosure. IFRS 16 explicitly asks for information about potential future cash outflows not reflected in the lease liability, including variable payments and options. A forensic reader therefore treats “lease liabilities” as the recognized core, not the total economic lease dependence.

#### Worked example: a lease modification changes the picture

Suppose the original lease liability is $249 after the three-year, $100-per-year example. At the end of year 1, the company pays $100. At a 10% rate, the interest for year 1 is $249 × 10% = $24.90, so the liability before the payment is $273.90 and after the payment is $173.90.

Now suppose the company signs a modification that adds one more $100 payment at the end of year 4. Assume the revised discount rate is 12% and the modification is measured at the end of year 1. The added payment’s illustrative present value is $100 ÷ 1.12³ = $71.18 because three years remain from the modification date. The company increases the lease liability and right-of-use asset by $71.18, subject to the applicable accounting rules.

The important forensic point is not the exact journal-entry label. It is that the future commitment changed before cash moved. A static ratio based only on last year’s lease liability can miss a new warehouse, renewal option, or expansion embedded in a signed contract.

### How to read a lease note

Read the note in this order:

1. Identify the reporting framework and transition date.
2. Find current and non-current lease liabilities, then reconcile the opening and closing balances.
3. Compare undiscounted maturity payments with the recognized liability. The difference is mostly discounting, but may also reflect payments excluded from measurement.
4. Look for weighted-average remaining lease term and discount rate.
5. Search for variable payments, renewal and termination options, sale-and-leaseback transactions, residual-value guarantees, and leases signed but not yet commenced.
6. Compare cash-flow classification. Under U.S. GAAP, operating lease payments generally remain operating cash flows, while finance-lease principal is financing cash flow. Under IFRS, principal and interest presentation can differ by policy; do not compare free cash flow mechanically without reading the policy.

## 3. Guarantees: a promise that may never become a payable

A guarantee is a promise by one party to make good on another party’s obligation if a specified failure occurs. It can support a bank loan, a supplier payment, a lease, a performance contract, or a tax and legal obligation. The guaranteed party may be a subsidiary, joint venture, customer, or unrelated counterparty.

The first forensic mistake is to treat the guaranteed amount as current debt. The second is to ignore it because no liability is recorded. A guarantee has at least five dimensions:

| Dimension | Question | Why it changes risk |
| --- | --- | --- |
| Trigger | What failure activates payment? | Default, non-performance, insolvency, or a regulatory event may differ |
| Notional | What is the maximum stated amount? | A cap may include principal, interest, fees, or damages |
| Duration | When does the promise expire? | A short guarantee and an evergreen guarantee have different tail risk |
| Recovery | What collateral or reimbursement exists? | A parent may recover from a solvent subsidiary, or not |
| Concentration | Are many guarantees tied to one project or borrower? | Correlated failures defeat the simple sum of small exposures |

Guarantee disclosures often use terms such as “maximum potential future payments.” That phrase is not an expected-loss estimate. It is closer to a stress ceiling under the contract. The analyst should separately estimate a base case, a stressed case, and a recovery case.

#### Worked example: a parent guarantee with a recovery assumption

Suppose Parent guarantees Subsidiary’s $100 debt. Management estimates a 20% probability that Subsidiary defaults, and if default occurs, the lender recovers $60 from collateral. Ignore discounting for this illustrative calculation.

1. Maximum potential payment: $100.
2. Loss given default after $60 recovery: $100 − $60 = $40.
3. Expected cash loss: 20% × $40 = $8.
4. Stress loss if default occurs and collateral fails: $100.

The $8 expected loss, $40 loss-given-default, and $100 maximum payment answer different questions. If the parent also depends on the subsidiary for a supply chain or dividend, the strategic exposure can exceed the accounting estimate. A guarantee is therefore a contingent cash-flow option written by the parent: usually unexercised, but potentially very expensive at exactly the moment liquidity is scarce.

![A guarantee has a maximum promise, a probability of activation, and a recovery assumption; expected loss is not the same as the contractual ceiling.](/imgs/blogs/hidden-liabilities-leases-guarantees-and-contingencies-4.webp)

### What to hunt in guarantee language

Search the annual report for “guarantee,” “guaranty,” “indemnification,” “surety,” “letter of credit,” “standby,” “keepwell,” “joint and several,” and “maximum potential.” Then read the surrounding paragraphs, not only the table. An indemnity can cover environmental remediation, tax, intellectual property, purchase price adjustments, or a former business. A letter of credit may be collateralized today but consume cash or borrowing capacity when drawn.

Also ask whether the guarantee is consolidated away. A parent guarantee of a wholly owned subsidiary may not create a new consolidated liability when the subsidiary’s debt is already on the group balance sheet, but it can matter to the parent-only entity, lenders, and minority investors. Conversely, a guarantee of an unconsolidated affiliate can be invisible in group debt while still being a claim on group cash.

## 4. Litigation and loss contingencies: the reserve is a judgment

Litigation accounting is an exercise in thresholds and uncertainty, not a verdict on who is morally right. The company may face a claim, deny liability, and still need to recognize or disclose a loss. A case can have a probable outcome but an unknowable amount, or a measurable amount but a likelihood below the recognition threshold.

Under the U.S. ASC 450 framework summarized by FASB’s Statement No. 5, a loss is accrued when information indicates that an asset was impaired or a liability incurred by the reporting date, the loss is probable, and the amount is reasonably estimable. If a loss is reasonably possible and material, disclosure may be required even without an accrual. The SEC cautions that saying “not expected to be material” is not enough when a material additional loss is reasonably possible; the registrant should provide an estimate or range, or say that it cannot make one.

This creates three different numbers in a note:

1. The recorded reserve: management’s best estimate of a recognized loss.
2. The disclosed range: additional exposure that is reasonably possible.
3. The legal claim or demand: what the plaintiff asks for, which may be larger than the economic loss.

Do not substitute one for another. A $50 million demand is not automatically a $50 million liability. But a $5 million reserve can still understate the liquidity risk if the disclosed range extends much higher and the case is correlated with a product recall or covenant breach.

#### Worked example: from a probability range to an accrual

Suppose counsel believes a company will probably lose a case. The estimated loss range is $20 to $50. Management’s best estimate is $30. Under the illustrative U.S. rule, the entry is:

```
Dr Litigation loss expense        $30
    Cr Litigation liability                    $30
```

If no amount in the $20–$50 range is a better estimate, a simple teaching example might accrue the low end, $20, while disclosing the possible additional loss up to $30 above the accrual. The exact accounting conclusion depends on the applicable guidance and facts; the example is designed to show why “reserve” and “maximum exposure” diverge.

Now add defense costs of $4 that are probable and separately incurred. Cash exposure in the base case becomes $30 + $4 = $34, while the stress case could be $50 + $4 = $54 before insurance recoveries. The intuition is that a reserve is a point estimate inside a distribution, not the distribution itself.

![A litigation reserve is the recognized point inside a wider outcome distribution; disclosure can extend beyond the booked amount.](/imgs/blogs/hidden-liabilities-leases-guarantees-and-contingencies-5.webp)

### Red flags in litigation notes

Look for a reserve that declines while the case count or legal language grows; a large “reasonably possible” range without a bridge to the recorded reserve; repeated statements that an estimate cannot be made; insurance recoveries treated as if they were certain; and legal matters described as immaterial despite a balance sheet with little liquidity. Compare the litigation footnote with cash paid for settlements, operating cash flow, restructuring charges, and subsequent events.

The most useful question is not “is management optimistic?” It is “what information would have to change for the reserve to move by $10 million, $100 million, or more?” That question converts lawyerly uncertainty into a sensitivity analysis.

## 5. Pension underfunding: a promise whose cash timing can surprise you

A defined-benefit pension promises a formula-based benefit, such as a payment tied to salary and years of service. The company must estimate the present value of benefits earned by employees and compare it with the fair value of plan assets. If the obligation exceeds assets, the plan is underfunded.

The estimate is sensitive to discount rates, salary growth, mortality, retirement timing, inflation, and asset returns. A higher discount rate usually lowers the present value of a long-dated obligation; a lower rate usually raises it. That means a reported improvement in funded status can come from assumptions rather than from a permanent improvement in operating cash generation.

Ford’s 2024 Annual Report, filed with the SEC, reported worldwide defined-benefit plans were underfunded by $0.5 billion at December 31, 2024. It reported funded plans $3.4 billion overfunded and unfunded plans $3.9 billion underfunded. The same report explained that the unfunded plans were “pay as you go,” with benefits paid from company cash. This is exactly why a net $0.5 billion figure needs decomposition: the gross components have different liquidity behavior.

![Net pension funded status can hide opposite gross positions: funded plans may be overfunded while pay-as-you-go plans remain a direct call on corporate cash.](/imgs/blogs/hidden-liabilities-leases-guarantees-and-contingencies-6.webp)

#### Worked example: why netting can hide the cash call

Suppose a group has a funded pension plan with assets of $130 and obligations of $100, so it is overfunded by $30. It also has a pay-as-you-go plan with obligations of $70 and no assets, so it is underfunded by $70.

1. Net funded status: $30 overfunded − $70 underfunded = $40 underfunded.
2. The net number says $40, but the pay-as-you-go plan has $70 of gross obligations with no asset pool.
3. If the funded plan cannot transfer its assets to the pay-as-you-go plan, the $30 surplus may not offset near-term cash payments.
4. If the company pays $15 of benefits from cash during the year, the liquidity question is about $15 of cash plus future contributions, not only the $40 net deficit.

The intuition is that a net balance-sheet number can be mathematically correct and economically incomplete when assets are ring-fenced by plan, jurisdiction, or regulation.

### Reconstructing pension stress

Read the funded-status table by plan geography and type. Note discount rates and the sensitivity table. Separate service cost, interest cost, expected return, actual return, contributions, and benefits paid. Then ask whether the company has contribution holidays, minimum funding requirements, or unfunded plans that pay directly from corporate cash.

A pension deficit is not automatically debt. It is a long-duration claim with uncertain timing. For leverage analysis, an analyst might add a portion of the deficit to debt, forecast required contributions, or stress the discount rate. The method should be disclosed, because adding the entire deficit to debt and adding the future contributions again would double count.

## 6. Reconstructing the obligation stack

Now combine the layers without adding unlike numbers together. The proper workflow is a reconciliation, not a single “hidden liability” line.

![A forensic reconstruction moves from footnotes to comparable obligation measures, then tests those measures against cash flow and liquidity.](/imgs/blogs/hidden-liabilities-leases-guarantees-and-contingencies-7.webp)

### Step 1: inventory the claims

Create a table with one row per obligation: lease, debt, purchase commitment, guarantee, lawsuit, tax dispute, pension plan, environmental remediation, asset retirement obligation, and indemnity. Record the legal entity that owes it, the beneficiary, the trigger, the currency, the maturity, and whether payment is fixed or variable.

### Step 2: separate cash schedules from accounting measurements

For fixed leases and debt, the undiscounted schedule describes cash. The recognized liability is a present value. For litigation, the recognized reserve is a probability-and-estimate threshold. For guarantees, the maximum potential payment is a ceiling. For pensions, the funded status is a net position built from assumptions. Keep separate columns for each.

### Step 3: model overlap and recoveries

A parent guarantee may cover debt already included in consolidated debt. An insurance policy may offset a lawsuit, but the insurer may dispute coverage. A pension asset may be inaccessible to another plan. A lease may be embedded in a supply contract and already captured in a right-of-use asset. Mark each overlap instead of blindly summing.

### Step 4: convert to scenarios

Use at least three scenarios:

| Scenario | Leases | Guarantees | Litigation | Pension |
| --- | --- | --- | --- | --- |
| Base | Contractual payments, recognized liabilities | Expected loss after recovery | Booked reserve plus defense costs | Scheduled contributions |
| Stress | Renewals exercised, rates higher, revenue lower | Default with partial recovery | High end of disclosed range | Lower discount rate and weak asset returns |
| Liquidity shock | Payments bunch during refinancing | Several correlated draws | Settlement before insurance recovery | Pay-as-you-go benefits and cash contributions |

The aim is not false precision. The aim is to discover whether a business can meet claims in the same bad year. Lease payments, guarantee draws, legal settlements, and pension contributions can all be individually manageable and collectively destabilizing.

#### Worked example: a four-bucket obligation bridge

Suppose an illustrative company has:

- Recognized debt of $400 and cash of $150.
- A three-year lease with a $249 present value and $300 undiscounted payments.
- A $100 guarantee with a 20% default probability and $60 recovery, giving $8 expected loss.
- A litigation reserve of $30, with a disclosed possible range up to $50.
- A pension deficit of $40, including a $70 pay-as-you-go plan partly offset by a $30 surplus elsewhere.

Build three views:

1. Reported net debt: $400 − $150 = $250.
2. Recognized-obligation view: $400 + $249 + $30 + $40 − $150 = $569.
3. Stress cash-claim view: $400 + $300 + $100 + $50 + $70 − $150 = $770, before timing, tax, insurance, or recoveries.

The second and third lines are not accounting statements. They are analytical views with explicit assumptions. The $8 expected guarantee loss is not substituted for the $100 stress claim; the $30 litigation reserve is not substituted for the $50 disclosed high case; and the $40 pension net deficit is not substituted for the $70 cash-funded plan. The intuition is that a useful reconstruction is a range with a bridge, not a dramatic single number.

## 7. Ratios that improve—and ratios that become misleading

Bringing leases onto the balance sheet usually increases assets and liabilities. It can make leverage look higher, asset turnover lower, and return on assets different even when the underlying business has not changed. The income statement may also shift from a single rent expense toward depreciation and interest under a finance-style model, which changes EBITDA and operating-profit presentation.

That is why the analyst should prefer ratios that explain the economics:

- Lease-adjusted fixed-charge coverage: cash available for fixed claims divided by interest, scheduled principal, rent or lease cash, and other fixed charges.
- Contractual-cash coverage: operating cash flow divided by the next one to three years of fixed payments.
- Pension-adjusted leverage: debt plus a clearly defined pension adjustment, without double counting contribution forecasts.
- Guarantee concentration: maximum exposure and expected loss by counterparty, not just the aggregate total.
- Reserve development: cash paid and reserve releases or additions over time compared with opening reserves.

#### Worked example: fixed-charge coverage after reconstruction

Suppose operating cash flow is $180. Annual interest is $30, debt principal due is $20, lease cash payments are $40, and expected pension contributions are $10.

1. Cash available for fixed claims: $180.
2. Fixed claims: $30 + $20 + $40 + $10 = $100.
3. Fixed-charge coverage: $180 ÷ $100 = 1.80×.
4. If a stress case adds $30 of guarantee draws and $20 of litigation settlement, fixed claims become $150.
5. Stress coverage: $180 ÷ $150 = 1.20×.

This is a deliberately simple ratio. It does not forecast taxes, working capital, capex, or refinancing. But it teaches the right habit: test the cash bill, not only the reported interest expense.

## 8. The forensic reading procedure

The most reliable review is repeatable. Start with the annual report’s table of contents and mark every note containing “commitments,” “contingencies,” “leases,” “retirement,” “guarantees,” “indemnification,” “legal proceedings,” “pension,” “other liabilities,” or “subsequent events.” Do not begin by searching for a single magic debt number. Begin by building a map of where a future cash claim could hide.

Next, tie each note back to a primary statement line. If a lease note says the liability is $249, locate the current and long-term portions on the balance sheet. If a litigation note says $30 is accrued, locate the accrued-liability line or the separate reserve. If pension expense changes, trace it to operating income, other comprehensive income, and cash contributions. A number that cannot be tied out is not automatically wrong, but it is a reason to slow down and understand the entity structure.

Then read the language around the number for verbs. “Will pay” suggests a fixed obligation. “May be required” signals a contingent claim. “Has entered into” points to a signed commitment. “Expects to” is management’s forecast, not a contract. “Cannot estimate” is information about uncertainty, not a blank space. “Recoverable” may reduce the net economic loss, but only if the counterparty, insurer, or indemnitor can actually pay and the recovery is legally enforceable.

Finally, compare the note with the next report. A liability can be hidden by time rather than by location: a renewal option becomes reasonably certain, a guarantee expires, a lawsuit settles, or a pension contribution is made after year-end. Subsequent-event disclosures can reveal a settlement or refinancing that changes the interpretation of the prior balance sheet. The comparison should be dated, because a 2024 contract table is not evidence of the same obligation in 2026.

### Build a maturity ladder, not a pile

Put fixed claims on a year-by-year ladder. Add debt principal, interest, lease cash, purchase commitments, expected pension contributions, and the base litigation estimate in separate rows. Put guarantee draws and high-case settlements in scenario rows rather than in the base schedule. This exposes bunching. A company may have ample total assets but a refinancing wall or lease-heavy year that operating cash cannot cover.

For each row, record whether it is cancellable, secured, subordinated, tax-deductible, insured, or dependent on a counterparty. A commitment that can be cancelled at no cost is not equivalent to a non-cancellable payment. A guarantee with a solvent, collateralized subsidiary is not equivalent to a guarantee of a weak affiliate. A pension surplus in a regulated plan is not equivalent to unrestricted corporate cash.

### Ask what the accounting number excludes

Lease liabilities may exclude variable payments tied to sales or usage. Litigation reserves may exclude defense costs, fines, or claims that are only reasonably possible. Guarantees may exclude obligations already recognized in a consolidated subsidiary or may state a maximum that includes interest and fees. Pension funded status may exclude benefits covered by a different plan or jurisdiction. The exclusion is often more informative than the headline.

This does not justify adding every exclusion to an adjusted balance sheet. It means the analyst should write an explicit sentence for each adjustment: “I add this because it is a fixed cash claim,” “I do not add this because it is already consolidated,” or “I show it only in the stress case because activation is uncertain.” That audit trail makes the analysis falsifiable and prevents a dramatic but incoherent total.

### The final cross-check: can the business fund the promise?

A liability becomes a crisis when it meets weak cash generation, restricted cash, covenant pressure, or a closed financing market. Compare the obligation ladder with operating cash flow after maintenance capital expenditure, cash on hand that is genuinely available, committed credit lines, and near-term maturities. Consider whether a guarantee draw would consume the same borrowing capacity needed to refinance debt. Consider whether a pension contribution would arrive in the same downturn that increases litigation claims or reduces lease utilization.

The output should be a short conclusion with three layers: what is already recognized, what is contractually committed, and what could happen in a stress case. That wording is less sensational than “the company has $X of hidden debt,” but it is more useful. It tells the reader which claim is certain, which estimate is judgmental, and which scenario would actually threaten solvency.

## Common misconceptions

### “Off-balance-sheet means fraudulent.”

No. The old operating-lease model was an accepted accounting classification, and the footnote disclosed future commitments. The forensic issue was comparability and visibility, not automatically misconduct.

### “ASC 842 and IFRS 16 eliminate lease risk.”

They improve recognition of many leases. They do not eliminate variable payments, renewal judgments, short-term exemptions, supplier contracts containing leases, or the risk that fixed payments arrive when revenue collapses.

### “A guarantee is debt.”

A guarantee is a contingent promise. Its maximum amount, expected loss, collateral, duration, and consolidation context all matter. Treating the maximum as current debt overstates the base case; ignoring it understates tail risk.

### “No litigation reserve means no litigation loss.”

It may mean the loss is not probable, not estimable, or both. A material reasonably possible exposure can still require disclosure. Read the range, insurance language, and subsequent settlements.

### “A small net pension deficit is harmless.”

Netting may combine an asset-backed plan with an unfunded pay-as-you-go plan. The gross plan structure and required cash contributions tell you whether the deficit is a balance-sheet statistic or an immediate liquidity claim.

### “Add every footnote number to debt.”

That double counts and mixes measures. Reconcile recognized liabilities, undiscounted cash, expected loss, maximum exposure, and funded status separately before building an adjusted metric.

## How it shows up in real markets

### Amazon’s pre-ASC 842 lease footprint

Amazon’s 2018 Form 10-K, filed before the new operating-lease balance-sheet model, disclosed $26.666 billion of future operating-lease payments as of December 31, 2018. The maturity schedule showed $3.127 billion due in 2019 and $13.026 billion thereafter. Amazon also reported $3.4 billion of operating-lease rent expense in 2018. The filing is a clean case study because it gives both the annual income-statement cost and the multi-year contractual tail.

An analyst who used only debt would miss the scale of fixed property and equipment commitments. An analyst who added the full $26.666 billion to debt would overstate present value and ignore timing. The correct reconstruction starts with the schedule, discounts it using a defensible rate, then checks what ASC 842 subsequently recognized and what payments remained variable or outside the liability. The lesson is not that Amazon did anything improper; it is that fast-growing asset-light labels can carry very large leased capacity.

### Ford’s 2024 pension decomposition

Ford’s 2024 Annual Report reported a $0.5 billion worldwide defined-benefit underfunded status at December 31, 2024. The same disclosure decomposed that into $3.4 billion of overfunding in funded plans and $3.9 billion of underfunding in unfunded plans. It described the unfunded plans as pay-as-you-go, with benefits paid from company cash.

The net number is the right accounting aggregation under the relevant rules, but it is not the full liquidity story. A cash-funded plan does not become funded merely because another plan has a surplus. The analyst should forecast benefit payments and contributions, review discount-rate sensitivity, and ask whether the operating business can fund pensions during a downturn. The lesson is to inspect gross plan structure whenever netting crosses legal entities or funding regimes.

### Target and the 2019 U.S. lease transition

Target’s first-quarter 2019 SEC filing described its adoption of ASC 842 on January 1, 2019 and the recognition of operating- and finance-lease right-of-use assets and liabilities. A transition can create a visible balance-sheet jump without a corresponding new cash payment on the adoption date. That jump is an accounting recognition event for contracts already in force.

This is why time-series analysis around adoption needs a bridge. A debt-to-assets ratio may move because leases entered the balance sheet, not because management borrowed cash that quarter. The analyst should recast prior periods where possible, or at least label the break in the series. The lesson is that a new standard can improve economic visibility while reducing mechanical comparability unless the analyst adjusts the history.

### SEC contingency guidance and the unbooked range

The SEC’s Staff Accounting Bulletin Topic 5 gives a practical warning: when a material additional loss is reasonably possible beyond the amount accrued, a registrant should disclose an estimated additional loss or range, or state that it cannot make the estimate. It also says a generic statement that a contingency is not expected to be material does not satisfy the requirement when a material additional loss is reasonably possible.

This is not a claim about one company’s guilt or about the outcome of a particular case. It is a reading rule. When a filing says “reasonably possible,” find the range; when it says “cannot estimate,” find the cash resources, insurance, indemnities, and timing that would absorb an adverse outcome. The lesson is that uncertainty often moves from the balance sheet into prose, and prose still carries information.

## When this matters to you

For a lender, these obligations compete for the same repayment capacity. For an equity analyst, they change enterprise value, free-cash-flow quality, and the durability of margins. For a board or operator, they identify commitments that can become fixed exactly when revenue is variable. For an employee or pension beneficiary, funding status and plan design affect the security of a long-dated promise.

The practical habit is to read the notes before making a clean-company judgment. Build the obligation inventory, mark which numbers are cash schedules and which are present values, model recoveries and overlaps, and stress the year in which several claims arrive together. This is educational analysis, not individualized investment advice.

## Sources & further reading

- [IFRS Foundation: IFRS 16 is now effective](https://www.ifrs.org/news-and-events/news/2019/01/ifrs-16-is-now-effective/) — issued January 2016; effective for annual periods beginning on or after January 1, 2019; includes the IFRS Foundation’s estimated $3 trillion of previously unrecognized future lease payments.
- [IFRS Foundation: IFRS 16 Leases](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-16-leases/) — recognition model, 12-month and low-value exemptions, and disclosure objectives.
- [Amazon 2018 Form 10-K, SEC](https://www.sec.gov/Archives/edgar/data/1018724/000101872419000004/amzn-20181231x10k.htm) — December 31, 2018 operating-lease maturity schedule and 2016–2018 rent expense.
- [Target first-quarter 2019 Form 10-Q, SEC](https://www.sec.gov/Archives/edgar/data/29989/000002998919000011/a2019q110-q.htm) — January 1, 2019 ASC 842 adoption disclosure.
- [Ford 2024 Annual Report, SEC](https://www.sec.gov/Archives/edgar/data/37996/000110465925029103/tm259451d1_ars.pdf) — December 31, 2024 worldwide pension funded status and gross funded/unfunded plan figures.
- [FASB: Summary of Statement No. 5](https://fasb.org/page/PageContent?bcpath=tff&pageId=%2Freference-library%2Fsuperseded-standards%2Fsummary-of-statement-no-5.html) — historical U.S. loss-contingency recognition principle summarized by FASB.
- [SEC Staff Accounting Bulletin Topic 5](https://www.sec.gov/oca/sab-code-t5) — disclosure expectations for material reasonably possible losses beyond recorded accruals.
- [Reading the balance sheet](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here) — the visible balance-sheet layer.
- [The footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) — where commitment and uncertainty disclosures live.
- [How an audit works—and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) — why an audit opinion is not a guarantee about future cash obligations.
