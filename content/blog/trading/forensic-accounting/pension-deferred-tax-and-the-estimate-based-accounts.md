---
title: "Pension, Deferred Tax and the Estimate-Based Accounts"
date: "2026-08-05"
publishDate: "2026-08-05"
description: "A beginner-friendly forensic guide to the accounts where management's assumptions decide reported profit: pension discount rates and expected returns, deferred-tax valuation allowances, and how to read the footnotes that give the game away."
tags: ["forensic-accounting", "pension-accounting", "deferred-tax", "valuation-allowance", "discount-rate", "estimates-and-judgements", "effective-tax-rate", "quality-of-earnings", "financial-statements", "fraud-detection", "footnotes"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 50
---

> [!important]
> **TL;DR** — Some accounts are measured. Others are *assumed*. In defined-benefit pensions and deferred tax, a management assumption is the direct input to reported profit, and changing it moves earnings without moving a dollar of cash.
>
> - A pension obligation is a present value. Its size is decided by a **discount rate** management picks. At a typical duration of 14 years, moving that rate by 50 basis points moves the obligation by about 7%.
> - Under the old rules, the income statement recorded an **expected** return on plan assets, not the actual one. A sponsor that assumed 8% while markets delivered 2% still booked 8% — and could report pension *income* inside operating profit.
> - A **deferred tax asset** is a stored future tax saving. The **valuation allowance** against it is a yes/no judgement about whether future profits will ever arrive — and reversing it drops the whole asset into earnings as a one-off gain.
> - The tell is arithmetic, not intuition: an effective tax rate far from the statutory rate, a big allowance release in a conveniently weak year, and profit that appears in non-cash lines while cash taxes paid stay near zero.
> - The forensic question for every estimate account is the same one: **what evidence supported this assumption when it was made, and what happened to the cash?**

Two companies run the same factories, sell the same products, and collect the same cash. One reports a profit of \$1.08 billion. The other reports \$300 million. Nothing about the business differs. The gap is a single accounting judgement about whether future profits are likely enough to justify recognising a tax asset today.

That is not a hypothetical loophole. It is ordinary, legal, standard-compliant accounting — and it happens every reporting season somewhere. Most of the financial statements you read are dominated by things that can be counted: cash in the bank, shares issued, invoices sent. But a handful of accounts cannot be counted at all. They can only be *estimated*, because the event they describe has not happened yet. A pension paid in 2061. A tax deduction used in 2034. A warranty claim that may never be filed.

For those accounts, management does not report a fact. Management reports a forecast. And a forecast has a dial on it.

![Three assumption inputs feed non-cash income statement lines and move reported net income, while a parallel track shows cash paid to pensioners and cash taxes paid leaving the cash flow statement unchanged.](/imgs/blogs/pension-deferred-tax-and-the-estimate-based-accounts-1.webp)

The diagram above is the mental model for the whole article. On the top track, an assumption feeds a non-cash income statement line, and reported net income moves. On the bottom track, actual cash moves at its own pace, driven by real payments to real pensioners and real cheques to real tax authorities. The two tracks are connected eventually — every estimate is settled in cash someday — but in any single reporting period they can point in opposite directions.

This is the article where forensic accounting stops being about hidden transactions and starts being about disclosed judgements. Nobody is forging invoices here. Everything is in the footnotes. The skill is knowing which footnote to open, and what a reasonable number would have looked like.

## Foundations: the building blocks of an estimate-based account

Before any of the mechanics, we need four ideas defined from scratch. If you already know what a present value is, skim; if you do not, nothing later will make sense without them.

### Accrual accounting, in one paragraph

Financial statements are prepared on an **accrual** basis, which means a cost is recorded in the period when the company *becomes obligated* for it, not the period when it pays. If you promise an employee a pension in 2061 in exchange for work they do in 2026, the cost belongs to 2026. That is the entire reason estimate accounts exist: accrual accounting forces companies to put a number on the future today. The companion article on [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) covers the gap this creates in detail.

### Present value, and why the discount rate is everything

A **present value** is what a future payment is worth today. Money today is worth more than the same money later, because today's money can earn a return in the meantime. To convert a future amount into today's terms, you divide by a growth factor. The rate you use to do that division is the **discount rate**.

$$\text{Present value} = \frac{\text{Future payment}}{(1 + r)^{n}}$$

where $r$ is the discount rate and $n$ is the number of years until the payment. A \$1,000 payment due in 20 years, discounted at 5%, is worth \$1,000 ÷ 1.05²⁰ = \$377 today. Discount the same payment at 4.5% instead and it is worth \$415 today — 10% more, from a half-point change in one assumption.

That is the mechanic underneath the entire pension section. The company does not choose how much it owes retirees. It chooses the rate at which it converts that promise into a number on the balance sheet, and that choice moves the number a great deal.

### Duration: how sensitive a present value is

**Duration** is the standard measure of how much a present value moves when the discount rate moves. It is expressed in years, and the rule of thumb is:

$$\frac{\Delta \text{Value}}{\text{Value}} \approx -\,D \times \Delta r$$

where $D$ is duration and $\Delta r$ is the change in the discount rate. A duration of 14 years means a 1 percentage point rise in rates cuts the value by roughly 14%, and a 50 basis point cut (a **basis point** is one hundredth of a percentage point, so 50 bp = 0.50%) raises it by roughly 7%. Mature defined-benefit pension obligations commonly sit in the low-to-mid teens for duration, because the payments stretch decades into the future.

Keep that number in your head: **at duration 14, one basis point of assumption is worth 0.14% of the obligation.**

### The difference between an estimate and a lie

This is the part that separates forensic analysis from cynicism. An estimate is not wrong because it later turns out to be different from reality. Estimates are *supposed* to be revised — that is what makes them estimates. The forensic questions are narrower and answerable:

1. **Was the assumption supportable by evidence available when it was made?** A discount rate is meant to reference observable high-quality corporate bond yields. If bond yields say 5.0% and the company says 5.9%, that gap needs an explanation.
2. **Is it consistent with peers running similar plans?** Assumptions are disclosed. Peer comparison is free.
3. **Did the assumption move in a direction that conveniently helped the reported number, in a period when management needed help?**
4. **Did cash ever validate it?** An estimate that flatters profit and never produces cash is the pattern worth chasing.

None of these produce a verdict on its own. Together they produce a *confidence statement*, which is the real output of this work.

## Part 1: defined-benefit pensions, the original assumption machine

### What a defined-benefit promise actually is

There are two kinds of workplace pension. In a **defined-contribution** plan, the employer pays a fixed amount into an account each year (say, 5% of salary), the employee owns the account, and whatever it grows to is what the employee gets. The employer's obligation ends when the payment is made. There is almost nothing to estimate, and almost nothing to manipulate.

In a **defined-benefit** plan, the employer promises a *benefit* — for example, 1.5% of final salary for each year worked, paid every year from retirement until death. The employer, not the employee, carries the risk that markets disappoint or that retirees live longer than expected. To report this promise, the company must estimate:

- how long each employee will keep working,
- what their salary will be at retirement (salary growth),
- how long they will live after retiring (mortality),
- and what all of those future payments are worth today (the discount rate).

Four forecasts, each of which moves the answer. This is why defined-benefit accounting is the canonical estimate-based account, and why so many sponsors closed these plans.

### The obligation and the assets: reading the funded status

The measured promise has a name. Under US accounting it is the **projected benefit obligation (PBO)** — the present value of benefits earned to date, including the effect of expected future salary increases. Under international standards the equivalent is the **defined benefit obligation (DBO)**. Against it sits a pool of investments, the **plan assets**, held in a separate legal trust for the beneficiaries.

The difference between them is the **funded status**:

$$\text{Funded status} = \text{Plan assets} - \text{Projected benefit obligation}$$

A negative funded status is a **deficit**, and it appears on the balance sheet as a liability. A positive one is a **surplus**.

![A two-column stack showing plan assets of 8.5 billion dollars against a taller projected benefit obligation of 10.0 billion dollars, with a 1.5 billion dollar deficit band, and a side panel showing service cost of 150 million reaching operating profit while remeasurements go to other comprehensive income.](/imgs/blogs/pension-deferred-tax-and-the-estimate-based-accounts-2.webp)

The figure shows the crucial asymmetry. The balance sheet carries the entire \$1.5 billion hole. The income statement, in the same year, may carry a tiny fraction of it. A reader who looks only at the profit line will not see the size of the problem, which is precisely why pension underfunding lived off the face of the statements for decades — a pattern the article on [hidden liabilities](/blog/trading/forensic-accounting/hidden-liabilities-leases-guarantees-and-contingencies) treats more broadly.

Throughout Part 1 we will use a single illustrative sponsor. Call it Northfield Industrial. Every Northfield number below is invented arithmetic chosen to be easy to follow — it is not a real company and not a benchmark. The real-company figures come later, and they are sourced.

**Northfield Industrial, as disclosed in its pension footnote:**

| Line | Amount |
| --- | --- |
| Projected benefit obligation | \$10.0 billion |
| Fair value of plan assets | \$8.5 billion |
| Funded status (deficit) | \$1.5 billion |
| Discount rate assumption | 5.0% |
| Expected long-term return on plan assets | 8.0% |
| Rate of compensation increase | 3.0% |
| Duration of the obligation | ~14 years |
| Service cost for the year | \$150 million |

### The discount rate: the largest dial in the accounts

The discount rate converts decades of promised payments into one number. Both US and international standards anchor it to observable market yields on high-quality corporate bonds of matching duration — the logic being that a bond portfolio of that quality could, in principle, defease the promise.

"High-quality corporate bonds of matching duration" sounds precise. In practice it leaves room: which bonds, which quality cut-off, which curve-fitting method, which measurement date. Reasonable actuaries land in a range. The forensic point is not that a company picked a number inside the range — it is *where* inside the range, and whether that position drifts in a helpful direction over time.

![An XY chart with discount rate on the horizontal axis from 4.0 to 6.0 percent and projected benefit obligation on the vertical axis in billions, showing a slightly convex curve through five labelled points from 11.43 billion at 4.0 percent down to 8.76 billion at 6.0 percent, with the 4.5 to 5.0 percent band shaded and annotated.](/imgs/blogs/pension-deferred-tax-and-the-estimate-based-accounts-3.webp)

#### Worked example: what 50 basis points is worth

Northfield discloses a 5.0% discount rate and a \$10.0 billion obligation with a duration of about 14 years. Suppose the actuary had used 4.5% instead — still a defensible number, half a point lower.

**Step 1 — the rule of thumb.** Using the duration approximation:

$$\frac{\Delta \text{PBO}}{\text{PBO}} \approx -D \times \Delta r = -14 \times (-0.005) = +0.07$$

So the obligation rises by about 7%, or \$700 million.

**Step 2 — the exact discounting.** Re-discounting the actual payment stream at 4.5% gives \$10.69 billion, an increase of \$690 million, or 6.9%. The rule of thumb was accurate to within \$10 million on a \$10 billion number. Across the full range:

| Discount rate | Obligation | Change vs 5.0% |
| --- | --- | --- |
| 4.0% | \$11.43 billion | +\$1.43 billion |
| 4.5% | \$10.69 billion | +\$0.69 billion |
| **5.0% (disclosed)** | **\$10.00 billion** | — |
| 5.5% | \$9.36 billion | −\$0.64 billion |
| 6.0% | \$8.76 billion | −\$1.24 billion |

**Step 3 — what it does to the deficit.** At 5.0%, the deficit is \$10.0bn − \$8.5bn = \$1.5 billion. At 4.5%, it is \$10.69bn − \$8.5bn = \$2.19 billion. The deficit grew by 46% because one assumption moved by half a point.

**Step 4 — the counterintuitive part.** You would expect a bigger obligation to mean a bigger annual expense. It does not work that way. The **interest cost** component of pension expense is the discount rate multiplied by the obligation:

- At 5.0%: 5.0% × \$10.00 billion = \$500 million
- At 4.5%: 4.5% × \$10.69 billion = \$481 million

Interest cost *falls* by \$19 million even though the obligation rose by \$690 million. Service cost rises — a benefit earned this year is discounted over a longer horizon, so at roughly duration 20 for the newly-earned slice, \$150 million becomes about \$165 million. Add the two and the annual charge goes from \$650 million to \$646 million: the income statement is about \$4 million **better**, on a change that made the balance sheet \$690 million worse.

**The intuition:** the discount rate moves the balance sheet enormously and the income statement barely at all. A reader watching only the profit line will never see a company's pension position deteriorate.

That asymmetry runs the other way too, and it is the one that matters for manipulation. A company that wants a *smaller* reported obligation nudges the discount rate up. The deficit shrinks, equity improves, leverage ratios improve, and the income statement barely flinches — so there is nothing in the profit line to alert a casual reader that the change happened at all.

### Salary growth and mortality: the quieter assumptions

The discount rate gets attention because it moves the most. Two others deserve a look.

**Rate of compensation increase** matters because a final-salary promise is indexed to a number the company is forecasting. Assume 3.0% salary growth and the promise is one size; assume 2.0% and it is smaller. A sponsor forecasting salary growth persistently below its own actual wage bill growth is telling you two inconsistent stories in the same annual report — one in the pension note, one in the cost line.

**Mortality** is the assumption nobody argues about publicly and everybody feels. Longevity assumptions come from published mortality tables and projection scales. When those tables are updated, obligations move — and the direction of the update is not a management choice, but the *speed of adoption* sometimes is.

#### Worked example: one year of longevity

Northfield's actuary updates the mortality table. Retirees are now expected to live, on average, one year longer than the previous table assumed.

**Step 1 — what changes.** Every retiree receives roughly one extra year of payments, but that extra year arrives at the far end of the payment stream, so it is discounted heavily.

**Step 2 — the size.** As a working approximation used by pension actuaries, one additional year of life expectancy adds roughly 3–4% to a mature obligation. Take 3.5%: \$10.0 billion × 3.5% = **\$350 million** added to the obligation.

**Step 3 — where it lands.** Under current rules this is a *remeasurement*: it goes straight to other comprehensive income, not to profit. Net income does not move. The balance sheet liability rises by \$350 million and equity falls by the same amount.

**Step 4 — the forensic read.** Check which published mortality table the footnote names and when it was issued. A sponsor still using a table two generations old while peers have adopted the current one is carrying an obligation that is understated by a knowable amount.

**The intuition:** the assumption that quietly costs the most is the one with no market price to check it against.

### Plan assets and the expected return: the assumption that became income

Here is where pension accounting produced its most notorious effect.

A pension trust holds real investments — equities, bonds, property, private funds — whose value bounces around every year. Standard-setters in the 1980s faced a genuine problem: if a company's reported profit swung by hundreds of millions because the stock market had a bad year, the income statement would tell you almost nothing about the business.

The solution was **smoothing**, and it arrived in the US with **SFAS 87**, "Employers' Accounting for Pensions", issued in December 1985 and effective for fiscal years beginning after 15 December 1986. Rather than record the actual return on plan assets, the income statement recorded an **expected long-term return on plan assets** — a percentage management selected — applied to the asset base. The difference between expected and actual was deferred, and only slowly fed into profit through an amortisation mechanism: the famous **10% corridor**, which ignored cumulative gains and losses entirely until they exceeded 10% of the larger of the obligation or the assets, and then released only the excess over the workforce's average remaining service life.

The consequence follows directly: **the pension line in the income statement recorded management's assumption about markets, not what markets did.**

![An XY chart across five years showing a flat amber line at 8.0 percent labelled expected return assumption against a jagged blue actual return line at plus 18, minus 22, plus 11, minus 5 and plus 14 percent, with the vertical gaps shaded green or red each year.](/imgs/blogs/pension-deferred-tax-and-the-estimate-based-accounts-4.webp)

#### Worked example: expected return versus what actually happened

Northfield assumes an 8.0% long-term return and holds \$8.5 billion of plan assets. Over five years, actual returns are +18%, −22%, +11%, −5%, +14%.

**Step 1 — what the income statement recorded.** 8.0% every single year. Applied to the asset base, that is roughly \$680 million of expected return credited against pension cost in year one, and a similar figure in each subsequent year.

**Step 2 — what actually happened.** The arithmetic average of the five actual returns is:

$$\frac{18 - 22 + 11 - 5 + 14}{5} = \frac{16}{5} = 3.2\%$$

The compound (geometric) return is lower still. Multiplying the five years through: 1.18 × 0.78 × 1.11 × 0.95 × 1.14 = 1.106, so the assets grew 10.6% in total over five years — a compound rate of about **2.0% per year**.

**Step 3 — the gap.** Compounding 8.0% for five years gives 1.08⁵ = 1.469, a 46.9% total gain. Applied to \$8.5 billion of starting assets:

- Expected asset growth: \$8.5bn × 46.9% = **\$3.99 billion**
- Actual asset growth: \$8.5bn × 10.6% = **\$0.90 billion**
- Shortfall: **\$3.09 billion**

**Step 4 — where the shortfall went.** Not into reported profit. It accumulated as an unrecognised actuarial loss, drip-fed into future expense over many years. Meanwhile the income statement had spent five years reporting a return the plan never earned.

**The intuition:** expected-return accounting let a sponsor book a market forecast as though it were a market result, and postpone the correction almost indefinitely.

Now push the dial one notch further. Suppose Northfield had assumed **9.0%** instead of 8.0% — a single percentage point, and a number many large sponsors genuinely used in the late 1990s. Expected return becomes 9.0% × \$8.5 billion = \$765 million instead of \$680 million. That extra \$85 million is a straight reduction in reported pension cost, which is to say a straight increase in pretax profit. No cash changed hands. No pensioner's cheque changed. One line in a footnote changed.

### Funded status versus what hits the P&L

The annual charge is called **net periodic benefit cost**, and under the older US framework it had four main components:

| Component | What it is | Direction |
| --- | --- | --- |
| Service cost | Present value of benefits earned this year | Increases cost |
| Interest cost | Discount rate × obligation (the promise gets one year closer) | Increases cost |
| Expected return on plan assets | Assumed return × asset base | **Decreases** cost |
| Amortisation of gains/losses and prior service cost | Slow release of deferred differences | Either |

![A before-and-after comparison showing the old four-component net periodic pension cost of 50 million dollars sitting entirely in operating profit versus the new presentation where only 150 million of service cost reaches operating profit and net interest of 75 million sits below it.](/imgs/blogs/pension-deferred-tax-and-the-estimate-based-accounts-5.webp)

#### Worked example: how an 8% assumption produces pension income

Northfield's four components for the year:

| Component | Calculation | Amount |
| --- | --- | --- |
| Service cost | given | +\$150 million |
| Interest cost | 5.0% × \$10.0 billion | +\$500 million |
| Expected return on plan assets | 8.0% × \$8.5 billion | −\$680 million |
| Amortisation | given | +\$80 million |
| **Net periodic pension cost** | | **+\$50 million** |

A plan \$1.5 billion in deficit produced a reported annual cost of \$50 million — about 3% of the hole it is standing in. And under the old presentation, all four components sat together inside operating profit.

**Now change one assumption.** Raise the expected return to 9.0%:

| Component | Calculation | Amount |
| --- | --- | --- |
| Service cost | | +\$150 million |
| Interest cost | | +\$500 million |
| Expected return on plan assets | 9.0% × \$8.5 billion | −\$765 million |
| Amortisation | | +\$80 million |
| **Net periodic pension result** | | **−\$35 million** |

The cost has become **pension income of \$35 million**, sitting inside operating profit. An \$85 million swing in reported operating results, produced entirely by a one-point change in a forecast about long-run capital markets, in a plan that remains \$1.5 billion underfunded and pays out real cash every month.

**The intuition:** a sufficiently optimistic return assumption does not merely reduce pension expense — it converts a pension deficit into a contributor to operating profit.

This is not a theoretical embarrassment. In the late 1990s and early 2000s, pension credits were a meaningful part of reported operating income at several large industrial sponsors, and analysts began stripping them out precisely because they were not operating results at all. The habit of separating reported operating profit from the parts of it that are not operating is the same discipline covered in [non-GAAP and adjusted EBITDA](/blog/trading/forensic-accounting/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent) — except here the adjustment runs in the opposite direction from the usual one, because the company is the party inflating the number.

### The reform: what the standard-setters did about it

Standard-setters closed the two worst gaps in stages, and knowing which regime a set of accounts was prepared under is essential to reading historical filings.

**Getting the deficit onto the balance sheet.** For years, the funded status was disclosed in a footnote while the balance sheet showed a smoothed, netted figure that could bear little resemblance to the real hole. **SFAS 158**, issued in September 2006 and effective for fiscal years ending after 15 December 2006, ended that by requiring the funded status itself — the plain difference between plan assets and the obligation — to be recognised on the balance sheet.

**Killing the expected return.** International standards took the sharper approach. The June 2011 amendment to **IAS 19**, effective for annual periods beginning on or after **1 January 2013**, abolished the expected-return-on-assets assumption outright. In its place, the income statement records **net interest** on the net defined benefit liability, computed at the *discount rate* — the same rate used to measure the obligation. The difference between that and the actual return on assets becomes a remeasurement, recognised in other comprehensive income and never recycled into profit. The corridor was removed at the same time, so actuarial gains and losses are recognised immediately rather than deferred.

The effect on Northfield is stark. Under the new approach:

- Service cost: \$150 million
- Net interest: 5.0% × \$1.5 billion deficit = **\$75 million**
- Total charge to profit: **\$225 million**
- Remeasurements: to other comprehensive income

There is no \$680 million expected-return credit, because there is no expected-return assumption. The pension cost rises four and a half times relative to the \$50 million reported under the old framework — from the same plan, in the same year, with the same assets and the same retirees.

**Moving the components out of operating profit.** US standards kept the expected-return assumption but attacked the presentation problem instead. **ASU 2017-07**, issued in March 2017 and effective for public entities for fiscal years beginning after 15 December 2017, requires that only **service cost** — the part that genuinely reflects employees earning benefits this year — be reported in the same line items as other employee compensation, which is to say inside operating profit. Interest cost, expected return and amortisation must be presented separately, outside operating income. Only service cost remains eligible for capitalisation into an asset.

For Northfield, that means operating profit now bears \$150 million of service cost, not the \$50 million net figure. The other components still exist, and the expected-return assumption still flatters them — but it can no longer flatter the operating line that analysts build their multiples on.

**What this means when you read old filings.** A pre-reform income statement and a post-reform one are not comparable. If you are looking at a sponsor's operating margin across that boundary, part of the change is presentation, not performance. And if a company's own multi-year "adjusted" history straddles the change without saying so, that is worth a note.

## Part 2: deferred tax, and the allowance that works as a profit dial

Pensions are about the future *cost* of a promise. Deferred tax is about the future *benefit* of a loss. The manipulation surface is different, and in some ways cruder, because the whole asset can be switched on or off by a single judgement.

### What a deferred tax asset is

Companies keep two sets of books, entirely legally. One follows accounting standards and produces the profit reported to investors. The other follows tax law and produces the profit reported to the tax authority. The rules differ — different depreciation schedules, different timing for provisions, different treatment of losses — so the two numbers rarely match.

Where those differences will reverse in future periods, accounting requires the future tax effect to be recorded today:

- A **deferred tax liability (DTL)** is tax you will owe later on income already reported to investors — typically because tax depreciation ran ahead of book depreciation.
- A **deferred tax asset (DTA)** is tax you will *save* later, because you have already taken a hit for accounting purposes that the tax authority has not yet let you deduct, or because you have losses you can carry forward.

The most important source of DTAs, and the one that matters forensically, is the **net operating loss carryforward**. When a company loses money, most tax systems let it carry that loss forward to offset future taxable profits. A loss is therefore a stored future tax saving — a real economic asset, provided the company ever earns enough profit to use it.

#### Worked example: the loss that becomes an asset

Northfield's sister company, Larkspur Manufacturing (also illustrative), accumulates \$4.0 billion of tax losses over a brutal five-year stretch. The statutory corporate tax rate is 21%.

**Step 1 — measure the asset.** The stored benefit is the loss multiplied by the rate at which it will be deducted:

$$\text{Gross deferred tax asset} = \$4.0\text{bn} \times 21\% = \$840\text{ million}$$

**Step 2 — ask whether it is real.** \$840 million of value exists only if Larkspur eventually earns \$4.0 billion of taxable profit. A company that never returns to profitability never uses the losses, and the asset is worth nothing.

**Step 3 — that question is the whole game.** Accounting cannot verify a forecast. So the standard hands the answer to management, subject to a threshold.

**The intuition:** a deferred tax asset is a claim on the company's own future success, valued today by the people whose performance is being measured.

### The valuation allowance: a yes/no switch worth \$840 million

US accounting recognises the gross deferred tax asset and then reduces it by a **valuation allowance** — a contra-asset that writes the DTA down to the amount considered realisable. Under ASC 740-10-30-5(e), an allowance is required if it is **more likely than not** — a probability greater than 50% — that some portion or all of the asset will *not* be realised. International standards reach a similar place by a different route: IAS 12 has no separate allowance account at all, and instead recognises a deferred tax asset only to the extent that it is *probable* that future taxable profit will be available. The IFRS balance sheet shows one adjusted number where the US balance sheet shows a gross asset and an allowance against it — which means the US presentation, helpfully for us, makes the dial visible.

Either way, the accounting collapses a continuous forecast into a binary. Above the threshold, the asset is on the balance sheet in full. Below it, it is written off in full. There is no partial credit for a company that is 45% likely to recover.

![A pipeline showing four billion dollars of tax losses multiplied by a 21 percent rate to give an 840 million dollar gross deferred tax asset, feeding a more-likely-than-not decision node that branches to either a full valuation allowance leaving zero on the balance sheet or a release crediting 840 million straight to tax expense.](/imgs/blogs/pension-deferred-tax-and-the-estimate-based-accounts-6.webp)

Two features make this the single most concentrated earnings dial in the accounts:

**It is binary.** Nothing else in the financial statements flips \$840 million on and off with one judgement.

**The evidence standard is asymmetric.** Accounting guidance treats **cumulative losses in recent years** as significant negative evidence — objectively verifiable, and very hard to overcome with management forecasts. In practice "recent years" is generally read as the current year plus the two prior years. That is a genuinely sensible rule: a company that has lost money three years running should not be allowed to book an asset premised on the profits it keeps failing to earn. But it creates a mechanical consequence. Once a company climbs out of cumulative losses, the negative evidence disappears, and the release becomes not merely permitted but *required*. The timing of that release is where judgement lives.

### Releasing the allowance: manufacturing a one-off earnings beat

When the allowance comes off, the credit does not go to equity or to a reserve. It goes to **income tax expense** — which means it goes straight through the income statement into net income.

![A waterfall chart in millions of dollars showing pretax income of 300 million, tax at 21 percent of minus 63 million, a valuation allowance release of plus 840 million, and reported net income of 1,077 million, with a note that cash taxes paid were approximately zero.](/imgs/blogs/pension-deferred-tax-and-the-estimate-based-accounts-7.webp)

#### Worked example: the release year

Larkspur returns to profit. Pretax income for the year is \$300 million, and management concludes the valuation allowance should be released in full.

**Step 1 — the ordinary tax charge.** At 21%: \$300 million × 21% = **\$63 million** of tax expense.

**Step 2 — the release.** Removing the \$840 million allowance credits income tax expense by \$840 million.

**Step 3 — net tax line.** \$63 million expense minus \$840 million credit = a net tax **benefit** of \$777 million.

**Step 4 — reported net income.**

$$\$300\text{m} + \$777\text{m} = \$1{,}077\text{ million}$$

**Step 5 — the cash.** Zero, or near it. Larkspur still has \$4.0 billion of losses to offset against taxable income, so it pays essentially no cash tax. Operating cash flow is unaffected by the entire exercise.

**The intuition:** \$300 million of pretax profit became \$1.08 billion of net income, and the \$777 million difference was a change of mind about the future, not an event in the business.

Now consider what the earnings release looks like. Net income up several hundred percent. Record annual profit. Earnings per share far above consensus, because analysts model tax as a rate, not as an allowance decision. And every word of it is true and standard-compliant.

The forensic issue is never that the release happened — it is usually correct and often mandatory. The issues are three:

1. **Was it recurring?** It is a once-per-lifetime item. Any valuation metric built on that year's earnings is meaningless.
2. **Was the timing convenient?** A release lands in the year management chooses to conclude the evidence has tipped. If that conclusion arrives exactly when a weak underlying year needed rescuing, the timing is a fact worth recording.
3. **Was the reverse also convenient?** *Establishing* an allowance is the mirror trick: it dumps a large non-cash charge into an already-bad year, clearing the deck for later. That is [big-bath accounting](/blog/trading/forensic-accounting/cookie-jar-reserves-and-big-bath-accounting) using the tax line as the bucket.

### The effective tax rate reconciliation: where the accounts confess

Companies must disclose a reconciliation between the tax expense implied by the statutory rate and the tax expense actually reported. In US filings this is required for public entities; internationally, IAS 12 requires an equivalent explanation. It is a short table, usually buried deep in the tax footnote, and it is the most reliably informative table in the annual report.

The reason it works is that it is *forced arithmetic*. The company must start at the statutory rate, list every reconciling item, and arrive at the number it actually reported. Anything unusual has to appear as a named line.

![A three-column grid comparing Year 1, Year 2 and Year 3, showing pretax income, tax at the 21 percent statutory rate, the valuation allowance change, reported tax expense and effective tax rate, with the Year 2 release year highlighted and its negative 259 percent effective rate marked in red.](/imgs/blogs/pension-deferred-tax-and-the-estimate-based-accounts-8.webp)

#### Worked example: reading three years of Larkspur's rate reconciliation

| Line | Year 1 | Year 2 (release) | Year 3 |
| --- | --- | --- | --- |
| Pretax income | \$120m | \$300m | \$340m |
| Tax at statutory 21% | −\$25m | −\$63m | −\$71m |
| Valuation allowance change | −\$5m | +\$840m | \$0m |
| **Reported tax expense / (benefit)** | **−\$30m** | **+\$777m** | **−\$71m** |
| **Effective tax rate** | **25%** | **−259%** | **21%** |
| Cash taxes paid | ~\$0 | ~\$0 | ~\$0 |

**Step 1 — Year 1 reads normally.** A 25% effective rate against a 21% statutory rate is unremarkable; the \$5 million allowance increase explains the gap.

**Step 2 — Year 2 is the tell.** The effective rate is:

$$\frac{-\$777\text{m}}{\$300\text{m}} = -259\%$$

A **negative effective tax rate on positive pretax income** is one of the loudest signals in financial reporting. It means the tax line contributed profit rather than consuming it, and the reconciliation is obliged to tell you exactly which item did it.

**Step 3 — Year 3 is the quiet lesson.** The effective rate is back to a boring 21%, and Larkspur reports \$71 million of tax expense. Cash taxes paid are still approximately zero, because the carried-forward losses are doing the sheltering. Using the losses now draws down the recognised DTA rather than reducing tax expense. So Year 3's tax charge is *also* largely non-cash — in the opposite direction.

**Step 4 — the normalisation.** To value Larkspur, strip the release entirely and tax the pretax income at the statutory rate:

- Year 1: \$120m × 79% = \$95m
- Year 2: \$300m × 79% = \$237m
- Year 3: \$340m × 79% = \$269m

That is the earnings trajectory — steady growth from \$95m to \$269m. The reported sequence of \$90m, \$1,077m, \$269m describes an accounting event, not a business.

**The intuition:** the rate reconciliation converts a headline about a record year into a line item with a name.

### The rate-change trap

There is one more deferred-tax mechanism worth knowing, because it produces enormous numbers that mean nothing operationally.

Deferred tax balances are measured at the rate expected to apply when the temporary difference reverses. So when a government *changes* the corporate tax rate, every deferred tax balance on every affected company's books must be remeasured immediately, and the entire adjustment runs through income tax expense in the period the law is **enacted** — not the period it takes effect, and not spread over the years it actually applies to.

The direction depends on which side of the balance sheet the company sits:

- A company with large deferred tax **assets** takes a **charge** when rates fall — its stored future deductions are now worth less.
- A company with large deferred tax **liabilities** takes a **gain** when rates fall — its stored future obligations are now smaller.

Neither reflects anything the business did. Banks, with their large loss-related DTAs, sit in the first group. Capital-intensive companies with decades of accelerated depreciation sit in the second. The same tax cut produces multi-billion-dollar charges at one and multi-billion-dollar gains at the other, in the same quarter, for the same reason.

## Part 3: the rest of the estimate family

Pensions and deferred tax are the two biggest estimate accounts in most industrial filings, but the logic generalises. Whenever you see an account whose value depends on a forecast, apply the same four questions.

**Loan-loss provisions.** Banks must estimate credit losses on loans that have not defaulted. Both US and international frameworks now use *expected* credit loss models, which require a forward-looking forecast of the economy — a genuinely predictive judgement embedded in the largest expense line a bank has. Provisions can be built up in good years and released in bad ones, and a bank's profit in any single quarter is substantially a statement about its own forecast. Watch the provision as a percentage of loans against peers, and watch releases arriving in quarters that needed them.

**Warranty reserves.** A manufacturer estimates the cost of future repairs on products already sold, based on historical claim rates. If claim rates are estimated too high, the excess sits as a liability that can be released into profit later. If they are estimated too low, current profit is overstated and the true cost arrives later. The roll-forward — opening balance, additions, utilisation, releases, closing balance — is the disclosure that makes this visible.

**Asset retirement obligations.** A mining, oil or utility company must recognise the present value of the cost of decommissioning its assets decades from now. Every input is a forecast: the cost of the work, the timing, and the discount rate applied over 30 or 40 years. At those horizons, discounting is doing enormous work — a 40-year obligation discounted at 6% instead of 4% is worth roughly half as much today.

**Insurance reserves.** An insurer's largest liability is its estimate of claims it will eventually pay, including claims not yet reported. For long-duration products the estimation horizon runs decades, and reserve adequacy is a judgement that can be revisited — sometimes in very large amounts, and sometimes long after the business was written.

**Goodwill and intangibles.** Impairment testing depends on discounted cash flow forecasts of the acquired business. The mechanics get their own treatment in [goodwill, intangibles and the impairment timing game](/blog/trading/forensic-accounting/goodwill-intangibles-and-the-impairment-timing-game).

The common structure is worth stating plainly:

| Account | The forecast | Which way optimism helps | Where the truth eventually shows up |
| --- | --- | --- | --- |
| Pension obligation | Discount rate, mortality, salary | Higher discount rate shrinks the liability | Cash contributions to the plan |
| Pension expense (old rules) | Expected return on assets | Higher assumed return raises profit | Actual asset values in the funded status |
| Deferred tax asset | Future taxable profit | Recognition raises profit | Cash taxes paid |
| Loan-loss provision | Future default rates | Lower provisioning raises profit | Actual charge-offs |
| Warranty reserve | Future claim rates | Lower accrual raises profit | Actual repair spending |
| Asset retirement obligation | Cost, timing, discount rate | Higher discount rate shrinks the liability | Actual decommissioning spend |

The last column is the forensic accountant's friend. Estimates are opinions; cash is evidence. Every one of these accounts is eventually reconciled against a cash outcome, and the reconciliation is usually disclosed somewhere. The article on [reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) is the companion piece for that discipline.

## Common misconceptions

**"If it is in the footnotes, the market has already priced it."** The footnotes are public, but they are long, dense and often ignored. Pension assumption tables and tax rate reconciliations sit dozens of pages into a filing, and the numbers in them are not in any standard data feed as headline items. Information being disclosed is not the same as information being processed — which is the premise of the article on [the footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried).

**"An aggressive assumption is fraud."** It usually is not. Assumptions live in ranges, and sitting at the aggressive end of a defensible range is legal, disclosed and common. What you are building is a *quality of earnings* assessment, not an accusation. The right output is "this year's profit contains \$X that came from assumption changes rather than operations", not "these people are crooks."

**"A pension deficit means the company is about to fail."** A deficit is a long-dated obligation measured at a point-in-time discount rate, and it swings violently with rates without the underlying promise changing at all. What matters is the required cash contribution schedule — which is set by funding rules that differ from accounting rules — and whether the company can generate that cash. Accounting deficits and funding requirements are two different numbers.

**"A valuation allowance release means the company is healthy again."** It means management concluded that future profits are more likely than not to materialise. That conclusion may well be right — it is usually made when a business genuinely has turned — but the release itself contributes nothing to the turnaround. It is an accounting acknowledgement of a recovery, not evidence of one, and it inflates the very year it lands in.

**"Non-cash items do not matter because cash flow is what counts."** Non-cash items matter enormously, for three reasons. They drive reported EPS, which drives index membership, covenant tests, and executive compensation. They change balance-sheet equity, which drives leverage ratios and borrowing capacity. And a non-cash estimate today is usually a cash event later: Northfield's \$1.5 billion funded-status deficit sits nowhere on the income statement, yet it is a claim on the sponsor's future cash — a schedule of contributions still to be paid, not a number that stays on paper.

**"Rising rates are bad for companies with pensions."** For the accounting deficit, rising rates are usually *good*: a higher discount rate shrinks the obligation, often faster than it hurts the bond holdings in plan assets, so funded status improves. This is one of the few places in finance where a company's balance sheet improves as rates rise, and it means an improving funded status may say more about the bond market than about the sponsor.

## How it shows up in real markets

Everything above is mechanism. Here is the mechanism in filings, with dated figures and their sources. Note the pattern across these cases: almost none of them are fraud. The two enforcement actions in this list are the exceptions, and even they turned on *disclosure*, not on the estimates being illegal.

### The expected-return assumption fell by three percentage points, and it took twenty years

The single best-documented piece of evidence that the expected-return assumption was too high is that sponsors kept lowering it. Milliman's annual study of the 100 largest US corporate defined-benefit plans tracks the average assumed rate of return, and its 2020 Corporate Pension Funding Study reports the following series:

| Fiscal year | Average assumed return on plan assets |
| --- | --- |
| 2000 | 9.4% |
| 2010 | 8.0% |
| 2014 | 7.3% |
| 2016 | 7.0% |
| 2018 | 6.6% |
| 2019 | 6.5% |

Source: Milliman, *2020 Corporate Pension Funding Study*.

Read that as a forensic document rather than a statistic. In 2000, the largest pension sponsors in America were, on average, crediting 9.4% against their pension costs every year — and under the accounting of the day, that credit flowed into operating profit. The subsequent two decades of reductions were not a change in the pension promise. They were a two-decade admission that the earlier assumption had been too generous, arriving one basis point at a time, with each reduction quietly raising reported pension cost in the year it happened.

Milliman noted in its 2024 study that the assumption rose for the first time in the study's history — a reminder that the direction is set by markets, not by sentiment.

### What a real sensitivity disclosure looks like

The Northfield worked example used a duration of about 14 years and produced a 6.9% obligation move per 50 basis points. Real filings disclose their own versions of this, usually in 25 basis point increments. Two examples:

- **General Motors**, in its FY2004 Form 10-K, disclosed that a 25 basis point decrease in the discount rate would increase its pension obligation by approximately **\$2.3 billion**.
- **Boeing**, in its FY2022 Form 10-K, disclosed that a 25 basis point *increase* would reduce the obligation by about **\$1,270 million**, while a 25 basis point *decrease* would raise it by about **\$1,415 million**.

The Boeing pair is worth pausing on, because the two numbers are not equal. A quarter-point down moves the obligation \$145 million more than a quarter-point up. That asymmetry is **convexity** — the same effect that made the sensitivity curve in the figure above bow rather than run straight. Rate cuts hurt a pension sponsor more than equivalent rate rises help.

These disclosures are the reason the assumption table is worth reading. The company has already done the arithmetic for you and printed the answer. What it has not done is tell you where its own assumption sits relative to peers.

### Ford, 2011: \$12.4 billion of profit from a change of mind

Ford Motor Company spent the late 2000s accumulating enormous tax losses. Those losses created a large deferred tax asset, and against it Ford carried a valuation allowance — because a company posting losses of that magnitude could not assert that future taxable profit was more likely than not.

By the end of 2011 the evidence had turned. Ford released the bulk of the allowance in the fourth quarter, and as reported at the time, the release produced a **one-time non-cash tax benefit of roughly \$12.4 billion**. Ford's full-year 2011 net income came in at **\$20.2 billion**, against \$6.6 billion in 2010 — its best result since 1998 (CNNMoney, 27 January 2012).

Run the subtraction. Roughly \$12.4 billion of a \$20.2 billion result was a tax accounting entry. The underlying operating business had genuinely recovered — that was the whole reason the release was permitted — but an investor comparing "2011 net income of \$20.2 billion" against "2012 net income" without knowing about the release would conclude the company had collapsed the following year. It had not. The comparison base contained a once-per-corporate-lifetime item.

### Delta Air Lines, 2013: the same mechanism, disclosed in the 10-K

Delta ran the same play two years later, and its own filing states it plainly. In the fourth quarter of 2013 Delta reversed the valuation allowance against its deferred tax assets, recording a **non-cash income tax benefit of \$8.0 billion**. Delta's net income for fiscal 2013 was **\$10.5 billion**, of which the fourth quarter alone accounted for \$8.5 billion, or \$9.89 per diluted share (Delta Air Lines FY2013 Form 10-K).

Delta's fourth-quarter earnings that year were, on the face of it, larger than the annual profits of most companies in the index. Nearly all of it was the reversal of an accounting judgement made years earlier, and none of it was cash. The scale of the allowance being released is worth stating: Delta had entered the year carrying roughly **\$11.0 billion** of valuation allowance against its deferred tax assets, and the \$8.0 billion release came against pre-tax income of about **\$2.5 billion**.

The cash side tells the other half of the story. Delta's *current* federal income tax line for 2013 was a **\$24 million benefit**, and with approximately **\$15.3 billion** of federal net operating loss carryforwards still on hand, the company told investors it did not expect to pay cash federal income taxes for several years. The profit was real as accounting; the tax it implied was not a payment.

Neither Ford nor Delta did anything improper. Both were required to release once the evidence supported it. The lesson is not about their conduct — it is about what a reader must do with a number like that before putting a multiple on it.

### December 2017: one law, and multi-billion swings in both directions

When the Tax Cuts and Jobs Act was signed on 22 December 2017, cutting the US federal corporate rate from 35% to 21%, every affected company had to remeasure its deferred tax balances immediately, in the quarter of enactment. The results in the fourth quarter of 2017 are the cleanest natural experiment in estimate accounting ever run, because the trigger was identical for everyone and only the sign differed:

| Company | Q4 2017 deferred tax effect | Direction | Why |
| --- | --- | --- | --- |
| Citigroup | ~\$22.6 billion charge (\$12.4bn DTA remeasurement, \$7.9bn additional valuation allowance, \$2.3bn foreign tax credit reduction) | Loss | Large deferred tax **assets** worth less at 21% |
| Goldman Sachs | \$4.40 billion tax expense (roughly two-thirds repatriation) | Loss | Same direction, smaller balance |
| AT&T | ~\$20.3 billion non-cash benefit (provisional) | Gain | Large deferred tax **liabilities** now smaller |
| Comcast | \$12.7 billion net income tax benefit | Gain | Deferred tax liability revaluation |

Sources: Citigroup FY2017 Form 10-K; Goldman Sachs Q4 2017 earnings release (Form 8-K); AT&T FY2017 results; Comcast Q4 2017 earnings release (Form 8-K).

One nuance worth preserving, because it is easy to overstate the tidiness of the experiment: Citigroup does not attribute its \$12.4 billion deferred-tax-asset remeasurement to the rate cut alone. Its filing ascribes the remeasurement to the rate reduction **and** the move to a quasi-territorial tax system together. AT&T's own tax-expense decrease of **\$20,271 million** drove its reported effective tax rate to **(97.2)%** — a negative effective rate on positive pretax income, which is the loudest signal described earlier in this article, here produced by an act of Congress rather than by anything management chose.

Citigroup reported a net loss of about \$6.8 billion for 2017 as a result. AT&T reported net income of \$29.45 billion, up from \$12.98 billion in 2016. Goldman Sachs posted a fourth-quarter net loss of \$1.93 billion. Not one of these numbers tells you anything about how the business performed. They tell you which side of the deferred tax ledger each company happened to sit on when a law changed.

The SEC recognised the estimation problem immediately and issued **Staff Accounting Bulletin 118** on the same day the law was signed, allowing companies a measurement period of up to twelve months to finalise the accounting using provisional amounts. AT&T's provisional \$20.3 billion was later trued up to about \$22.2 billion in 2018 under that relief. Even the correction to the estimate was measured in billions.

### General Electric: the SEC case was about estimates, and not about the pension

GE is the company most associated in the public mind with pension trouble, and it genuinely has one of the largest corporate plans in the world. In October 2019 it announced it would freeze US pension benefits for approximately **20,000 salaried employees** effective 1 January 2021, and offer lump-sum payouts to about **100,000 former employees** with vested benefits. (Published estimates of the resulting deficit reduction varied between outlets, so no figure is quoted here.)

But the pension is not what the SEC pursued, and getting this right matters. On **9 December 2020**, GE agreed to pay a **\$200 million civil penalty** to settle SEC charges relating to disclosure failures in two other businesses:

- **GE Power** — the SEC found GE had not disclosed that a substantial share of reported profit came from **reductions in prior cost estimates** on long-term service agreements rather than from operations. Per the Commission's findings, this accounted for approximately 25% of GE's 2016 profit and roughly half of its profit for the first nine months of 2017.
- **GE Capital** — the SEC found GE had lowered projected claim costs on its **long-term-care insurance** run-off book between 2015 and 2017 without adequately disclosing the uncertainty in those projections.

Source: SEC press release 2020-312, 9 December 2020.

That is this article's thesis stated by a regulator. Neither charge was that GE's estimates were illegal. Both were that estimate changes were doing the work the market believed operations were doing, and investors were not told. Long-term service agreement cost estimates and insurance claim reserves sit in exactly the same family as pension assumptions and valuation allowances: forecasts that feed profit directly.

### Nortel, 2007: where estimate releases became enforcement

The line between aggressive estimation and fraud is crossed when the estimate is managed to hit a target. Nortel Networks is the canonical case.

The SEC filed suit on **15 October 2007** (Litigation Release LR-20333), alleging a revenue recognition fraud together with an earnings-management scheme in which Nortel improperly released approximately **\$500 million** of excess reserves in the first two quarters of 2003 — converting losses into reported profits, and in doing so triggering "return to profitability" bonuses for senior management. Nortel paid a **\$35 million** civil penalty under a final judgment entered on 25 October 2007, with the money placed in a Fair Fund for shareholders.

The forensic detail worth keeping is the bonus. The reserves were not released because the underlying obligations had resolved; they were released in the specific quarters that flipped a loss into a profit, and the profit threshold was the trigger for management compensation. That is the pattern the earlier article on [cookie-jar reserves](/blog/trading/forensic-accounting/cookie-jar-reserves-and-big-bath-accounting) describes in detail — and it is why "which quarter did the release land in, and what did that quarter need?" is a question worth asking every time.

### What the academic evidence says

The suspicion that sponsors manage the expected-return assumption is not only a journalistic one. Bergstresser, Desai and Rauh's study in the *Quarterly Journal of Economics* (2006) examined how firms set assumed rates of return on pension assets and found the choice associated with capital-market incentives rather than purely with the plans' investment prospects. The paper is the standard academic reference for treating the assumption as a discretionary reporting choice rather than a neutral forecast.

## The detection routine

Everything above collapses into a short, repeatable footnote-reading procedure. It takes about twenty minutes per company once you know where to look.

![A decision flow starting from the pension and tax footnotes, branching into five check nodes covering the discount rate versus market yields, expected return versus asset mix, mortality table vintage, valuation allowance roll-forward and effective versus statutory tax rate, converging on two verdict nodes.](/imgs/blogs/pension-deferred-tax-and-the-estimate-based-accounts-9.webp)

### Step 1: the pension assumption table, against three benchmarks

The footnote discloses the discount rate, the compensation increase assumption, and — under US rules — the expected long-term return on plan assets, usually for the current and prior year. Compare each against:

- **The market.** A discount rate should sit close to yields on high-quality corporate bonds of similar duration at the measurement date. A gap of more than 25–50 basis points in the company's favour deserves an explanation.
- **The peers.** Companies in the same industry with similar workforces should have similar assumptions. Assumption tables are disclosed, so this comparison costs nothing but time.
- **Its own history.** Assumptions should move with markets. An assumption that stays flat while bond yields move 150 basis points is telling you the company is managing to a number.

### Step 2: sanity-check the expected return against the asset mix

The same footnote discloses the plan's asset allocation. If a plan is 70% fixed income yielding 4% and 30% equities, an 8% expected total return requires the equity sleeve to deliver roughly 17% annually, forever. That is not a forecast; it is a wish. Do the weighted arithmetic — it takes one line and it is the single most effective sanity check in the whole footnote.

### Step 3: read the funded status, not the expense

Find plan assets and the benefit obligation. The difference is the real economic position. Then compare it to the annual pension cost in the income statement. A company reporting a \$50 million cost against a \$1.5 billion deficit is not reporting a small problem — it is reporting a small *slice* of a large problem.

### Step 4: track the valuation allowance roll-forward

The tax footnote shows the gross deferred tax asset, the allowance, and the net. Track the allowance across three years. Look for a large *establishment* in a bad year (a bath) and a large *release* in a year that needed help. Then ask what changed in the evidence between those two dates.

### Step 5: compare effective and statutory tax rates

Read the rate reconciliation. Any effective rate far from the statutory rate has a named cause sitting in the table. A negative effective rate on positive pretax income is the loudest version, but persistent single-digit rates and sudden reversals matter too.

### Step 6: split profit into cash-backed and estimate-based

The closing move. Take reported net income and subtract everything that came from an estimate change: allowance releases, reserve releases, pension credits, remeasurements. Compare what is left with operating cash flow. If reported profit is large and the residual is small, the year's earnings are a statement about assumptions.

| Finding | What it means | The next test |
| --- | --- | --- |
| Assumption inside peer range, cash-backed profit | Earnings stand as reported | Normal monitoring |
| Assumption at the aggressive edge of the range | Reported profit is flattered by a knowable amount | Quantify the effect; restate at the peer median |
| Large allowance release in a weak year | A one-off, non-recurring, non-cash gain | Strip it and revalue on normalised earnings |
| Assumption drifts helpfully year after year | A pattern, not a judgement | Compare the drift to the earnings targets it met |
| Estimate profit with no cash ever following | The estimate was never validated | Trace the obligation to actual cash settlement |

The output is deliberately not a verdict. It is a labelled confidence statement — what is known, what is inferred, and which document would resolve the difference. That is the same discipline the [balance sheet](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here) and [income statement](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings) articles apply to their own accounts.

## When this matters to you

If you own shares, estimate-based profit is the part of earnings least likely to repeat. A price-to-earnings multiple applied to a year containing a valuation allowance release is arithmetic applied to the wrong number, and the correction usually arrives the following year when the comparison base proves impossible to match.

If you lend, the funded status and the recognised deferred tax asset both sit inside the equity figure your covenants are measured against. An assumption change can move covenant headroom without any operating event, in either direction.

If you work somewhere with a defined-benefit plan, the assumption table is a statement about your own retirement security. The discount rate is an accounting convention, but the funded status it produces feeds into decisions about funding, freezing and settling the plan.

And if you are learning to read financial statements, this is the chapter that changes how you read all the others. Once you have seen an \$85 million swing in operating profit come out of a single footnote percentage, you stop reading the income statement as a record of what happened and start reading it as a document with authors.

Start with one company you already follow. Open the pension footnote and the tax footnote. Write down the discount rate, the expected return, the funded status, the valuation allowance balance, and the effective tax rate — five numbers, ten minutes. Then do it for a competitor. The comparison is where the analysis actually lives.

This article is educational, not individualised investment, accounting, tax or legal advice.

## Sources & further reading

Every real-company figure in this article is dated and attributed below. The Northfield Industrial and Larkspur Manufacturing walkthroughs are illustrative arithmetic with invented round numbers, chosen so the mechanics are easy to follow — they are not benchmarks and not drawn from any real filing.

**Enforcement and regulatory sources**

- U.S. Securities and Exchange Commission, [SEC Charges General Electric with Disclosure Failures in Connection with Prior Reporting of Its Power and Insurance Businesses, Press Release 2020-312](https://www.sec.gov/newsroom/press-releases/2020-312), 9 December 2020. Primary source for the \$200 million civil penalty, the GE Power long-term service agreement cost-estimate findings, and the GE Capital long-term-care insurance reserve findings.
- U.S. Securities and Exchange Commission, [SEC v. Nortel Networks Corporation, Litigation Release No. 20333](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-20333), 15 October 2007. Primary source for the approximately \$500 million of improper reserve releases in the first half of 2003, the "return to profitability" bonus trigger, and the \$35 million civil penalty.
- U.S. Securities and Exchange Commission, Staff Accounting Bulletin No. 118, 22 December 2017. The measurement-period relief for accounting for the Tax Cuts and Jobs Act under ASC 740.

**Company filings**

- Delta Air Lines, [Annual Report on Form 10-K for fiscal year 2013](https://www.sec.gov/Archives/edgar/data/0000027904/000002790414000003/dal1231201310k.htm). Primary source for the \$8.0 billion non-cash income tax benefit from the valuation allowance reversal and fiscal 2013 net income of \$10.5 billion.
- Citigroup, Annual Report on Form 10-K for fiscal year 2017. Source for the fourth-quarter 2017 deferred tax asset revaluation and deemed repatriation charge.
- The Goldman Sachs Group, [Fourth Quarter 2017 Earnings Results, Form 8-K Exhibit 99.1](https://www.sec.gov/Archives/edgar/data/0000886982/000119312518011730/d480179dex991.htm). Source for the \$4.40 billion income tax expense related to the Tax Act.
- Comcast Corporation, [Fourth Quarter 2017 Earnings Release, Form 8-K Exhibit 99.1](https://www.sec.gov/Archives/edgar/data/1166691/000110465918003683/a18-3600_1ex99d1.htm). Source for the \$12.7 billion net income tax benefit.
- AT&T Inc., fiscal year 2017 results and Form 10-K. Source for the provisional \$20.3 billion non-cash tax benefit and its 2018 true-up under SAB 118.
- General Motors, Form 10-K for fiscal year 2004, and Boeing, Form 10-K for fiscal year 2022. Sources for the disclosed discount-rate sensitivities quoted in the real-markets section.

**Standards**

- Financial Accounting Standards Board, Statement of Financial Accounting Standards No. 87, *Employers' Accounting for Pensions*, December 1985 (effective for fiscal years beginning after 15 December 1986), and No. 158, September 2006 (effective for fiscal years ending after 15 December 2006). Now codified within ASC 715.
- Financial Accounting Standards Board, Accounting Standards Update 2017-07, March 2017 (effective for public entities for fiscal years beginning after 15 December 2017).
- IFRS Foundation, [IAS 19 *Employee Benefits*](https://www.iasplus.com/content/27d4939c-7305-4ced-bbae-ed56fc4a3535), as amended June 2011, effective for annual periods beginning on or after 1 January 2013. Source for the removal of the corridor and the expected-return assumption, and the net-interest approach.
- Financial Accounting Standards Board, ASC 740 *Income Taxes*, including the more-likely-than-not threshold at ASC 740-10-30-5(e) and the effective-rate reconciliation disclosure at ASC 740-10-50-12; IFRS Foundation, IAS 12 *Income Taxes*, including the recognition test and the paragraph 81(c) rate reconciliation.

**Studies**

- Milliman, [2020 Corporate Pension Funding Study](https://www.milliman.com/en/insight/2020-Corporate-Pension-Funding-Study). Source for the series of average assumed rates of return on plan assets across the 100 largest US corporate defined-benefit plans.
- Society of Actuaries, "Duration and Convexity for Pension Liabilities", *Pension Section News*, September 2013. Background for the duration range of mature defined-benefit obligations.
- Daniel Bergstresser, Mihir Desai and Joshua Rauh, ["Earnings Manipulation, Pension Assumptions, and Managerial Investment Decisions"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=551681), *Quarterly Journal of Economics*, 2006.

**Read next in this series**

- [The footnotes and MD&A: where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) — how to work through the disclosure sections this article sends you into.
- [Cookie-jar reserves and big-bath accounting](/blog/trading/forensic-accounting/cookie-jar-reserves-and-big-bath-accounting) — the same release mechanics applied to operating reserves.
- [Reading the balance sheet: what companies hide here](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here) and [reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) — where the funded status and the cash-versus-estimate test actually live.
- [Non-GAAP and adjusted EBITDA: the metrics companies invent](/blog/trading/forensic-accounting/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent) — the adjustment discipline, running in the other direction.
