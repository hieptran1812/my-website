---
title: "Accrual vs Cash Accounting: Why Earnings Are an Opinion and Cash Is a Fact"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero guide to accrual accounting, the matching principle, and the accrual — the gap between reported earnings and cash — including the four accrual buckets, the accrual ratio, Sloan's accrual anomaly, and the estimates that quietly bend EPS."
tags: ["equity-research", "corporate-finance", "accrual-accounting", "earnings-quality", "accruals", "sloan-anomaly", "financial-statements", "accounting", "fundamental-analysis"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Reported earnings are an opinion built on accrual accounting; cash is a fact. The gap between the two is the *accrual*, and a large, growing accrual is one of the most reliable warning signs in all of equity analysis.
>
> - **Accrual accounting** records revenue when it is *earned* and expenses when they are *incurred*, regardless of when cash moves. That matching gives a truer picture of a period's economics than raw cash — but it hands management dozens of judgment calls.
> - The **accrual** is the part of profit that is not cash. Mechanically, **total accruals = net income − cash flow from operations**, and they pile up in four buckets: receivables, inventory, accrued/deferred expenses, and depreciation estimates.
> - The **accrual ratio** = total accruals ÷ average total assets turns the accrual into a comparable, scaled number. A negative ratio (cash exceeds profit) is high quality; a large positive ratio is a red flag.
> - **Sloan's accrual anomaly**: firms with high accruals systematically *underperform* firms with low accruals in the following year — by roughly 10 percentage points in the original study. High-accrual earnings are lower-quality and tend to reverse.
> - Estimates bend earnings without touching cash: stretching a useful life, softening a bad-debt allowance, capitalizing a cost instead of expensing it, building a **cookie-jar reserve** in good years to release in bad ones, or taking a **big bath** to reset the baseline. Learn to read the accrual, and you read past the opinion to the fact.

There is a sentence every serious investor eventually internalizes, and it is worth stating bluntly at the outset: **earnings are an opinion; cash is a fact.** A company's reported profit is not a measurement you could confirm with a stopwatch or a scale. It is the *output of a model* — a model with rules, yes, but also with hundreds of estimates, timing choices, and allocations, every one of which is a place where management exercises judgment. Change the judgment and you change the profit, often without a single dollar moving in or out of the bank. Cash is different. The cash balance is a number a bank can confirm. It either moved or it did not.

The bridge between these two worlds — between the opinion of earnings and the fact of cash — is **accrual accounting**, and the gap it opens up has a name: the **accrual**. Most of the time the accrual is small and benign, the ordinary friction of a business that bills customers before they pay and pays for inventory before it sells. But when the accrual gets large and keeps growing, it is telling you that reported profit is running ahead of real cash — and decades of academic research and forensic experience say that is one of the most reliable warning signs you will ever find in a set of financial statements.

The figure below is the mental model we will build toward. A single sale — \$1 million of machinery shipped in December on 90-day terms — books its entire revenue the day the goods leave the dock, but the cash does not arrive until March. For three months the income statement says the company earned \$1 million while the bank account says it earned nothing. That gap, multiplied across thousands of transactions and stretched by management's estimates, is the accrual.

![A timeline of one sale showing revenue booked in December under accrual accounting while the cash does not arrive until March ninety days later](/imgs/blogs/accruals-vs-cash-why-earnings-are-an-opinion-1.png)

We will build this from nothing. If you have never thought about *why* a company can be profitable and cash-poor at the same time, that is exactly the gap this piece fills. By the end you will understand the difference between cash-basis and accrual-basis accounting; why accrual is genuinely *more useful* and yet *more manipulable*; the four buckets where accruals hide; how to compute total accruals and the accrual ratio in two lines; what Sloan's famous accrual anomaly says and why it works; and the specific estimates — bad-debt allowances, warranty reserves, useful lives, capitalize-versus-expense — that let an honest-looking management team bend earnings legally. We will use a recurring high-quality company, **Northwind Industries**, and introduce a low-quality twin, **Riverstone Equipment**, so the contrast compounds as we go. Let us start with the foundations.

## Foundations: cash basis, accrual basis, and the matching principle

Before we can talk about the accrual, we need to be crystal clear about the two ways a business can keep its books. Take your time here — every later section depends on getting this distinction exactly right.

### Cash-basis accounting: a lemonade stand

Imagine a child running a lemonade stand for a summer. The simplest possible way to keep the books is **cash-basis accounting**: you record revenue when cash comes in and expenses when cash goes out. Someone buys a cup, hands over \$2, you write down \$2 of revenue. You buy a bag of lemons for \$5, you write down \$5 of expense. At the end of the day, profit is just the change in the cash box. Nothing is recorded until money physically moves.

Cash-basis accounting has one enormous virtue: it is impossible to lie about, because it *is* cash. There is no estimate, no judgment, no timing choice. The downside is that it gives a distorted picture of economic performance the moment a business gets even slightly complicated. Suppose on the last day of summer the child buys \$20 of lemons and sugar to use next week, and also sells \$15 of lemonade on credit to a neighbor who will pay tomorrow. Cash basis records a \$20 expense and \$0 of that \$15 sale — so the last day looks like a \$20 loss, even though the child actually had a great day (sold \$15 of product, used almost none of the \$20 of supplies). The cash moved at the wrong times relative to the economic activity.

### Accrual-basis accounting: matching effort to reward

**Accrual-basis accounting** fixes this by recording revenue when it is *earned* and expenses when they are *incurred*, regardless of when cash changes hands. The \$15 credit sale gets recorded as revenue today, because the child *delivered the lemonade* today — the sale was earned even though the cash will arrive tomorrow. The \$20 of supplies bought for next week is *not* an expense today, because none of it was used today; it sits as an asset (inventory) and becomes an expense only as it is consumed. Under accrual accounting, the last day shows the \$15 of revenue it truly earned and almost none of the supply cost — a much truer picture of how the day actually went.

This is the **matching principle**, and it is the heart of accrual accounting: *expenses should be recognized in the same period as the revenues they helped generate.* If you build a product in March, ship it in April, and the customer pays in May, accrual accounting books the revenue and the matching cost of goods in April — the period the economic transaction actually happened — not March (when you spent the cash) or May (when you collected it). The whole point is to align the recorded numbers with the *economics* of the period rather than the *cash timing*, which is often arbitrary.

For any business with credit sales, inventory, long-lived equipment, or obligations that span periods — which is to say, essentially every real company — accrual accounting gives a far more informative income statement than cash basis. That is why generally accepted accounting principles (GAAP) and international standards (IFRS) both *require* accrual accounting for public companies. Cash-basis books are for the lemonade stand; accrual books are for the corporation. The full anatomy of how accrual revenue becomes net income is the subject of the companion piece on the [income statement line by line](/blog/trading/equity-research/income-statement-line-by-line-revenue-to-net-income); here we only need the punchline.

### The accrual: the part of profit that is not cash

Here is the crucial consequence. Because accrual accounting records revenues and expenses on a different *clock* than cash, reported profit and operating cash flow diverge. The difference between them is the **accrual** — the slug of reported earnings that has not (yet) shown up as cash. In the lemonade example, the \$15 credit sale created \$15 of "accrual revenue" with no matching cash; the receivable is the accrual sitting on the balance sheet.

Stated as cleanly as possible:

$$
\text{Total accruals} = \text{Net income} - \text{Cash flow from operations}
$$

When net income exceeds operating cash flow, accruals are *positive*: the company is reporting more profit than it is collecting in cash. When operating cash flow exceeds net income, accruals are *negative*: the company is collecting more cash than it reports as profit (usually because non-cash expenses like depreciation depress reported earnings below true cash generation). We will return to this formula constantly — it is the single most important equation in earnings-quality analysis.

A quick terminology note, because it trips people up. "Accruals" in this analytical sense (net income minus operating cash flow) is a broader idea than the bookkeeping entry called "accrued liabilities." Accrued liabilities are one *specific* accrual — an expense incurred but not yet paid. The *total accruals* we care about as analysts capture *every* difference between accrual earnings and cash, including receivables, inventory, depreciation, and reserves. When this piece says "the accrual," it means the total. With cash basis, accrual basis, the matching principle, and the definition of the accrual in hand, we can build the rest.

## Why accrual is more useful — and more manipulable

This is the central tension of the entire piece, so it deserves its own section. Accrual accounting is simultaneously the *best* way to measure a period's economics and the *most dangerous* way, and both facts flow from the same source: judgment.

**Why accrual is more useful.** Cash flow in a single period is noisy. A company that prepays a year of insurance in January looks like it had a terrible January and eleven great months, even though the insurance was used evenly. A company that collects a big receivable in March looks like it had a spectacular March, even though the sale happened months earlier. Cash timing is full of these accidents. Accrual accounting smooths them out by matching: the insurance expense is spread across the year it covers, the revenue is booked when the sale was earned. The result is an income statement that, in principle, tells you how the business actually performed *this period* — not how the cash happened to slosh around. For comparing one period to the next, or one company to another, accrual earnings are usually far more informative than raw cash.

**Why accrual is more manipulable.** Every one of those matching decisions is an *estimate*, and estimates are judgments. How long will this factory last — 5 years or 10? What fraction of our receivables will go bad — 2% or 5%? How much will we owe in warranty claims on the products we sold this quarter? When does a software sale count as "earned" — at delivery, or spread over the support period? None of these has a single objectively correct answer. Reasonable accountants disagree, and the rules deliberately leave room for judgment because businesses genuinely differ. But that room for judgment is also room to *bend* the numbers. A management team that wants higher earnings can lengthen useful lives, soften its bad-debt assumptions, recognize revenue more aggressively, and release reserves — each move perfectly legal, each lifting reported profit, and *none of them touching cash.*

That is the whole game in one sentence: **the same judgment that makes accrual earnings informative also makes them an opinion.** Cash flow cannot be bent this way (not without outright fraud, anyway — and even then it is far harder, as the [cash flow statement](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from) companion piece explains). The accrual is precisely the part of earnings that lives in the space of opinion, which is why measuring it tells you how much of reported profit you should trust.

## Net income, operating cash flow, and the accrual as the gap

Let us make the accrual concrete with our two companies. The most powerful way to see it is to put two firms side by side that report the *identical* net income but have completely different cash underneath — because that is exactly the situation where the income statement lies to you and the accrual tells the truth.

The figure below shows it. Both Northwind and Riverstone report \$50 million of net income. But Northwind generated \$85 million of operating cash flow (cash *exceeds* profit, so its accrual is a healthy −\$35 million), while Riverstone generated only \$10 million (cash *far below* profit, so its accrual is a dangerous +\$40 million). Same headline earnings; opposite cash reality. The accrual is the gap, and the sign of the gap is the whole story.

![A before and after comparison showing two firms with identical fifty million dollar net income but opposite operating cash flow making one accrual negative and healthy and the other positive and dangerous](/imgs/blogs/accruals-vs-cash-why-earnings-are-an-opinion-2.png)

#### Worked example: total accruals for two firms with identical net income

Both companies report **net income of \$50 million** this year. Compute total accruals for each using the formula total accruals = net income − cash flow from operations:

**Northwind Industries.**
- Net income: \$50m
- Cash from operations: \$85m
- **Total accruals = \$50m − \$85m = −\$35m**

Northwind's accruals are *negative \$35 million*. Its operating cash flow exceeds its reported profit by \$35m, mostly because depreciation is a large non-cash expense that depresses earnings but not cash. This is the signature of high-quality earnings: the company is collecting more cash than it books as profit.

**Riverstone Equipment.**
- Net income: \$50m
- Cash from operations: \$10m
- **Total accruals = \$50m − \$10m = +\$40m**

Riverstone's accruals are *positive \$40 million*. It reports \$50m of profit but collects only \$10m of operating cash — \$40m of its "earnings" is sitting in receivables and inventory, not in the bank. This is the signature of low-quality earnings: profit is running far ahead of cash.

An investor who looked only at the income statement would see two equally profitable companies and might not be able to tell them apart. An investor who computes the accrual sees a cash machine and a paper-profit factory. *Identical net income can hide opposite cash realities, and the accrual — net income minus operating cash flow — is the single number that exposes the difference.*

This is the moment to absorb the sign convention, because it confuses everyone at first. **Large positive accruals are the warning sign**, because they mean reported profit is exceeding cash. **Negative accruals are reassuring**, because they mean cash exceeds reported profit. It feels backwards — surely "more accruals" sounds like "more profit," which sounds good? — but the logic is exactly the opposite. Positive accruals are profit you have booked but not collected; the higher they are, the more of your earnings is a promise rather than a fact.

## The four buckets where accruals hide

Total accruals is a single number, but it is built from movements in specific accounts, and you cannot interpret the total without knowing which bucket it came from. There are four big ones, and the figure below organizes them: two on the revenue side (where the lever is *recognizing income faster*) and two on the expense side (where the lever is *spreading or deferring cost*).

![A grid laying out the four accrual buckets receivables and inventory on the revenue side and accrued or deferred items and depreciation on the expense side each a legal lever over reported earnings](/imgs/blogs/accruals-vs-cash-why-earnings-are-an-opinion-3.png)

**Bucket 1 — Receivables (revenue timing).** When a company books revenue but the customer has not paid, the unpaid amount sits as **accounts receivable**, an asset. Receivables are the most common and most dangerous accrual: every dollar of receivables is a dollar of reported profit that has not become cash. A company can inflate earnings simply by recognizing revenue earlier and more aggressively — shipping product to distributors who do not need it ("channel stuffing"), recognizing multi-year contracts up front, or booking sales before the customer has truly committed. Receivables balloon, profit looks great, and cash quietly lags. *Rising receivables, especially faster than revenue, is the classic accrual red flag.*

**Bucket 2 — Inventory.** When a company buys or builds goods it has not yet sold, the cost is **capitalized** into inventory (an asset) rather than expensed. The expense (cost of goods sold) is recognized only when the item sells. This is correct under the matching principle, but it is also a lever: a company can lift gross margin by *over-producing* (spreading fixed factory costs across more units, lowering the cost per unit it expenses) or by failing to write down inventory that is obsolete or unsellable. Inventory that grows faster than sales is a yellow flag — either demand is weakening or stale goods are being carried at inflated values.

**Bucket 3 — Accrued and deferred items (expense timing & reserves).** This is the richest bucket for manipulation. **Accrued expenses** are costs incurred but not yet paid (wages, interest, warranties). **Reserves** are estimates set aside for future costs — the allowance for doubtful accounts (receivables expected to go bad), warranty reserves, restructuring reserves. Because these are *estimates*, management can make them bigger (depressing current profit, banking earnings for later) or smaller (boosting current profit). **Deferred revenue** — cash collected before the service is delivered — is the honest mirror image: a liability that becomes revenue later. This bucket is where cookie-jar reserves and big baths live.

**Bucket 4 — Depreciation and amortization (estimates).** When a company buys a long-lived asset, it spreads the cost over the asset's **useful life** as depreciation (for physical assets) or amortization (for intangibles). But "useful life" is an estimate. A factory might be depreciated over 5 years or 20; goodwill and intangibles carry their own assumptions. Lengthen the assumed life and the annual depreciation charge drops, lifting reported profit — with no change in cash, because the cash for the asset was spent long ago. This is the subtlest accrual lever and the hardest for an outsider to detect, because the assumption is buried in the footnotes.

Notice the common thread: in *every* bucket, the lever moves reported earnings without moving cash. That is what makes accruals the place to look. Each bucket maps to a section below, where we will work the numbers.

## The balance-sheet view of accruals

There are two ways to compute total accruals, and a careful analyst knows both because they cross-check each other.

The **cash-flow view** is the one we have used: total accruals = net income − cash flow from operations. It is the easiest to compute because both numbers come straight off the statements, and it is the one most academic work uses (the cash flow statement became mandatory in the US in 1987, which is why older studies used the next method instead).

The **balance-sheet view** builds accruals from the ground up, by measuring the change in the accrual accounts directly:

$$
\text{Accruals} = \Delta(\text{non-cash working capital}) - \text{Depreciation \& amortization}
$$

In words: take the change in **net operating working capital** (receivables + inventory + other current operating assets, minus payables + accrued liabilities + other current operating liabilities — *excluding* cash and short-term debt), then subtract depreciation and amortization. The working-capital change captures buckets 1, 2, and 3 (receivables, inventory, accrued/deferred items); subtracting D&A captures bucket 4. The result should approximately equal the cash-flow-view number.

Why two methods? Because they can disagree, and the disagreement is informative. The cash-flow view captures *everything* that drove the gap between profit and cash, including items that do not flow through working capital. The balance-sheet view isolates the *operating* accruals specifically. In Sloan's original 1996 study — which we will meet shortly — accruals were measured the balance-sheet way, because reliable cash flow statements did not exist for the older years in his sample. The two methods give similar rankings in most cases, but when they diverge sharply, something unusual is happening (an acquisition, a large non-operating item, a discontinued operation), and that itself is worth investigating.

#### Worked example: Northwind's accruals two ways

Let us confirm the two methods agree for Northwind. Recall Northwind's net income is \$50m and operating cash flow is \$85m, so the cash-flow view gives **total accruals = \$50m − \$85m = −\$35m**.

Now the balance-sheet view. Suppose over the year Northwind's operating working capital changed as follows:
- Accounts receivable: +\$12m (use of cash, *increases* accruals)
- Inventory: +\$5m (use of cash, *increases* accruals)
- Accounts payable: +\$9m (source of cash, *decreases* accruals)
- Accrued liabilities: +\$7m (source of cash, *decreases* accruals)
- Change in net operating working capital = \$12m + \$5m − \$9m − \$7m = **+\$1m**

Depreciation & amortization for the year was **\$28m**. Plug in:

$$
\text{Accruals} = \Delta(\text{NWC}) - \text{D\&A} = \$1\text{m} - \$28\text{m} = -\$27\text{m}
$$

The balance-sheet view gives −\$27m versus the cash-flow view's −\$35m. The \$8m difference is other non-cash items that ran through operating cash flow but not working capital or D&A — in Northwind's case, \$6m of stock-based compensation and a \$2m non-cash loss that were also added back to get to the \$85m of CFO. Both methods agree on the *sign and the story*: Northwind's accruals are solidly negative, dominated by depreciation, which is exactly what a healthy capital-intensive manufacturer looks like. *The two accrual methods rarely match to the dollar, but when they tell the same story you can trust the conclusion, and when they diverge you have found something worth a footnote read.*

## The accrual ratio: making accruals comparable

A raw accrual number — −\$35m for Northwind, +\$40m for Riverstone — is meaningful within one company, but you cannot compare it across companies of different sizes. A \$40m accrual is alarming for a \$200m company and trivial for a \$200 billion one. To compare, we **scale** the accrual by the size of the business, and the standard scalar is average total assets:

$$
\text{Accrual ratio} = \frac{\text{Total accruals}}{\text{Average total assets}}
$$

(Average total assets is the average of the beginning-of-year and end-of-year total assets — it smooths out asset growth during the year.) The accrual ratio expresses accruals as a *percentage of the asset base*, which makes it comparable across companies and across time. It is the workhorse metric of academic accrual research.

The figure below puts the ratio on a scale. Below roughly −5% (cash far exceeds profit) is the highest-quality region. The −5% to +5% band is normal and healthy — cash and profit track each other. The +5% to +15% band means profit is running ahead of cash, worth watching. Above +15% is the danger zone: accrual-heavy earnings that tend to reverse. Northwind sits at −8% (high quality); Riverstone sits at +22% (red flag).

![A horizontal scale of the accrual ratio from negative high quality on the left through a normal band to a positive red flag zone on the right with two firms marked at minus eight percent and plus twenty-two percent](/imgs/blogs/accruals-vs-cash-why-earnings-are-an-opinion-4.png)

#### Worked example: the accrual ratio for Northwind and Riverstone

Compute the accrual ratio for each company. Suppose both have **average total assets of about \$440 million** (they are similar-sized industrial firms).

**Northwind.**
$$
\text{Accrual ratio} = \frac{-\$35\text{m}}{\$440\text{m}} \approx -8.0\%
$$

A negative ratio: Northwind generates \$8 of cash for every \$100 of assets *beyond* what it reports as profit. This is firmly in the high-quality region.

**Riverstone.**
$$
\text{Accrual ratio} = \frac{+\$40\text{m}}{\$180\text{m}} \approx +22\%
$$

(Riverstone is smaller — about \$180m of average assets — which makes its \$40m accrual even more extreme as a fraction of the asset base.) A +22% accrual ratio means accruals equal nearly a quarter of the entire asset base in a single year. That is deep into the danger zone — the kind of reading that, in academic samples, predicts sharp underperformance and frequently precedes a restatement.

*Scaling accruals by total assets turns an unintelligible dollar figure into a comparable percentage, and the percentage is where the −8%-versus-+22% gulf between an investment and a trap becomes obvious.*

A practical note on benchmarks: the academic literature often sorts the entire market into ten deciles by accrual ratio. The lowest decile (most negative accruals) might average around −10% or lower; the highest decile (most positive) might average around +15% to +20% or higher. Any single company's ratio is best read *relative to its industry and its own history* — a stable −2% to +2% is unremarkable, a jump from +3% to +18% in one year is a flashing light, regardless of the absolute level.

The accrual ratio also has a close cousin you should compute alongside it: the **cash conversion ratio**, operating cash flow divided by net income. The two are just different framings of the same gap — where the accrual ratio expresses the gap as a fraction of *assets*, cash conversion expresses it as a fraction of *profit*. They move together by construction: a firm with large positive accruals (profit far above cash) will have cash conversion well *below* 1.0, while a firm with negative accruals (cash above profit) will have conversion *above* 1.0. For Northwind, cash conversion is \$85m ÷ \$50m = **1.70** — every dollar of profit becomes \$1.70 of cash, the mark of high quality. For Riverstone it is \$10m ÷ \$50m = **0.20** — only twenty cents of cash per dollar of reported profit, the mark of an accrual-laden, low-quality earnings stream. When the two ratios disagree — a benign accrual ratio but collapsing cash conversion, or vice versa — it usually means an unusual item (an acquisition, a large non-operating gain) is distorting one of the denominators, and that is your cue to dig into the notes. The companion piece on the [cash flow statement](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from) treats cash conversion in full; here it is enough to know that it and the accrual ratio are two windows onto the same fact.

## Sloan's accrual anomaly: why high accruals predict low returns

Everything so far has been mechanics. Now comes the payoff — the empirical result that made accruals one of the most studied phenomena in all of finance. In 1996, the accounting researcher **Richard Sloan** published a paper with a deceptively simple finding: *firms with high accruals subsequently earn lower stock returns than firms with low accruals.* This is the **accrual anomaly**, and it has held up across decades, countries, and methodologies.

The figure below shows the shape of the result. Sort every stock into portfolios by accrual ratio, then measure each portfolio's return over the following year. The portfolio of *lowest-accrual* firms earns the highest subsequent return; the portfolio of *highest-accrual* firms earns the lowest. The returns fall steadily as accruals rise, and the spread between the extremes — going long the low-accrual firms and short the high-accrual firms — was roughly 10 percentage points a year in Sloan's original work.

![A bar chart showing five portfolios sorted from lowest to highest accruals with next-year returns falling steadily from fourteen percent down to four percent producing a roughly ten point long short spread](/imgs/blogs/accruals-vs-cash-why-earnings-are-an-opinion-5.png)

Why does this work? The mechanism is **mean reversion in earnings**, and it is worth understanding because it is not magic — it is the natural consequence of what accruals *are*.

Recall that accruals are the non-cash part of earnings, and they are full of estimates that have to *reverse*. A company that booked aggressive revenue into receivables will eventually have to collect that cash or write it off — either way, the accrual unwinds. A company that under-reserved for bad debts will eventually have to take the loss. A company that capitalized costs will eventually have to expense them. Accruals, by their nature, are temporary; they are timing differences that reverse over time. So a year of high accruals tends to be followed by years where those accruals reverse and *drag down* future earnings.

The anomaly arises because **investors do not fully appreciate this.** They see high reported earnings and price the stock as if those earnings will persist, not noticing that a large slice of them is low-quality accrual that is about to reverse. When the reversal comes and earnings disappoint, the high-accrual stock underperforms. The low-accrual firm — whose earnings were *backed by cash* and therefore more persistent — keeps delivering, and outperforms. Sloan's insight was that the market systematically *overweights* the accrual component of earnings and *underweights* the cash component, even though the cash component is far more persistent.

The "persistence" word is doing real work here, and it is worth making precise, because it is the statistical engine of the whole anomaly. *Persistence* means how much of this year's earnings shows up again next year. If you split earnings into its cash component and its accrual component and ask how predictive each is of next year's earnings, the data say the cash component is *substantially more persistent* than the accrual component. Concretely, imagine a company reports \$100m of earnings made of \$90m cash and \$10m accrual versus a twin that reports the same \$100m but split \$60m cash and \$40m accrual. Next year, on average, the first company's earnings hold up far better, because cash earnings recur while accrual earnings partly evaporate. A rational market would price the high-accrual twin lower to reflect its flimsier earnings — but Sloan showed the market prices them about the *same*, treating a dollar of accrual earnings as if it were as durable as a dollar of cash earnings. The accrual anomaly is the correction of that mistake playing out over the following year, as the high-accrual earnings fail to repeat and the stock re-rates down. This is why the screen is not really a "trick" but a bet on a well-documented behavioral error: investors fixate on the bottom-line number and ignore its *composition*.

#### Worked example: building a simple accrual screen

You do not need a quant team to use this. Suppose you have a universe of 500 stocks and you compute the accrual ratio for each. You sort them into five buckets of 100 and look at the average accrual ratio and the historical next-year return pattern that the anomaly predicts:

| Bucket | Avg accrual ratio | Earnings quality | Predicted next-year return |
|---|---|---|---|
| 1 (lowest accruals) | −12% | Highest — cash exceeds profit | ~14% |
| 2 | −4% | High | ~11% |
| 3 (middle) | +3% | Normal | ~8% |
| 4 | +11% | Low | ~6% |
| 5 (highest accruals) | +20% | Lowest — profit far exceeds cash | ~4% |

The pattern is monotone: every step up in accruals corresponds to a step down in expected return. A long-short portfolio — buy bucket 1, short bucket 5 — would have earned roughly the 14% − 4% = **10-point spread** the anomaly describes. Even just *avoiding* bucket 5 (the highest-accrual quintile) for your long book removes the names most likely to disappoint. *The accrual anomaly turns a piece of accounting hygiene into an investing edge: the firms whose earnings are most backed by cash tend to keep delivering, and the firms whose earnings are mostly paper tend to give it back.*

A few honest caveats, because the anomaly is real but not a money-printing machine. First, like most anomalies, it has **weakened since publication** — once a pattern is widely known, capital flows to exploit it and the spread compresses (though it has not vanished). Second, it works *on average across many names*, not on every stock; a single high-accrual company can have a perfectly good reason (rapid, healthy growth that consumes working capital) and do fine. Third, the short side is hard and risky to implement in practice. The durable lesson is not "short high-accrual stocks" but the gentler, more robust one: **treat the accrual component of earnings with suspicion, and prefer earnings backed by cash.**

## Estimates that bend earnings: the accrual levers in detail

We have established *that* accruals matter. Now let us get specific about the *levers* — the actual accounting estimates a management team can pull to bend earnings without touching cash. This is where you learn to read past the opinion. Each of these is legal; the line into fraud is crossed only when the estimate becomes unsupportable, but long before that line, aggressive-but-legal estimates can materially flatter the numbers.

### Lever 1: useful lives and depreciation

The depreciation estimate is the cleanest illustration of how an assumption becomes profit. A company spreads an asset's cost over its assumed useful life. Lengthen the life, and the annual charge drops — instantly lifting reported profit, with zero effect on cash (the cash was spent buying the asset, long ago and unchanged).

The figure below shows the mechanics on a single \$200 million asset. Depreciate it over 5 years and the annual charge is \$40m; stretch the assumed life to 10 years and the charge halves to \$20m. That \$20m of "saved" expense flows straight through to pretax profit, lifting EPS — purely from changing a number in a footnote.

![A before and after comparison showing a two hundred million dollar asset depreciated over five years versus ten years where doubling the assumed life halves annual depreciation and lifts earnings per share with no change in cash](/imgs/blogs/accruals-vs-cash-why-earnings-are-an-opinion-6.png)

#### Worked example: stretching the useful life from 5 to 10 years lifts EPS

Northwind buys a **\$200 million** production line. It has 50 million shares outstanding, a 25% tax rate, and \$140m of pretax profit *before* depreciating this asset.

**Under a 5-year useful life:**
- Depreciation = \$200m ÷ 5 = \$40m per year
- Pretax profit = \$140m − \$40m = \$100m
- Net income = \$100m × (1 − 0.25) = \$75m
- EPS = \$75m ÷ 50m shares = **\$1.50**

**Under a 10-year useful life:**
- Depreciation = \$200m ÷ 10 = \$20m per year
- Pretax profit = \$140m − \$20m = \$120m
- Net income = \$120m × (1 − 0.25) = \$90m
- EPS = \$90m ÷ 50m shares = **\$1.80**

By changing one assumption — the useful life — EPS rose from \$1.50 to \$1.80, a **20% increase**, with not one extra dollar of cash and not one extra unit sold. The cash cost of the asset (\$200m) is identical under both assumptions; only the *pace* at which it is expensed changed. This is why a sudden, unexplained lengthening of useful lives is a classic earnings-management flag. *Depreciation is an estimate, and stretching the estimate converts a footnote assumption directly into earnings per share — which is exactly why analysts read the useful-life disclosures, not just the EPS headline.*

(Note this lever is *temporary*: stretching the life lifts profit now but leaves more to depreciate later, so it reverses over time — the hallmark of an accrual. And it shows up in the accrual: lower depreciation means higher net income relative to cash flow, pushing accruals *up*.)

### Lever 2: the allowance for doubtful accounts (bad-debt reserve)

When a company sells on credit, it knows some customers will not pay. GAAP requires it to estimate that and set up an **allowance for doubtful accounts** — a reserve that reduces receivables and records a bad-debt expense *now*, matching the loss to the period of the sale. The estimate is a judgment: maybe 2% of receivables, maybe 5%.

This estimate is a lever in both directions. *Lower* the assumed bad-debt rate and you record less expense, lifting current profit (and inflating reported receivables). *Raise* it and you depress current profit — which is sometimes the goal, as we will see with cookie jars. A company growing receivables fast while *shrinking* its allowance as a percentage of receivables is waving a flag: it is reporting that its customers are getting *more* creditworthy at the same time it is extending *more* credit, which is exactly backwards from what usually happens.

### Lever 3: warranty and restructuring reserves

The same logic applies to any estimated future cost. A **warranty reserve** estimates the cost of honoring warranties on products sold this period. A **restructuring reserve** estimates the cost of a layoff or plant closure. Because they are estimates, they are levers: over-reserve in a good year (depress profit now, bank it for later) or under-reserve (boost profit now). Restructuring reserves are especially abused because they are large, one-time, and easy to over-book — the seed of the "big bath," which we will meet shortly.

### Lever 4: capitalize versus expense

This is the most consequential lever of all, because it can move very large numbers. When a company spends money, it must decide whether the cost is an **expense** (hits the income statement immediately, reducing this period's profit) or an **asset** to be **capitalized** (recorded on the balance sheet and expensed gradually over future periods via depreciation/amortization). The cash outflow is *identical* either way; only the *timing of the expense* differs.

Capitalizing a cost that should have been expensed is one of the most powerful earnings-management techniques in existence — it moves the entire cost off the current income statement and parks it on the balance sheet, to be dribbled out over years. It is also the mechanism behind some of the largest frauds in history.

#### Worked example: capitalizing versus expensing a \$30M cost

Riverstone spends **\$30 million** on something genuinely ambiguous — say, a major software development effort or a line-rental cost that *could* arguably be treated as building a long-lived asset. It has 40 million shares, a 25% tax rate, and \$80m of pretax profit before accounting for this \$30m.

**If it expenses the \$30m (conservative, correct if the cost is really an operating expense):**
- Pretax profit = \$80m − \$30m = \$50m
- Net income = \$50m × 0.75 = \$37.5m
- EPS = \$37.5m ÷ 40m = **\$0.94**

**If it capitalizes the \$30m (aggressive; spreads the cost over, say, 10 years at \$3m/year):**
- This year's expense = only \$3m of amortization, not the full \$30m
- Pretax profit = \$80m − \$3m = \$77m
- Net income = \$77m × 0.75 = \$57.75m
- EPS = \$57.75m ÷ 40m = **\$1.44**

By capitalizing instead of expensing, EPS leaps from \$0.94 to \$1.44 — a **53% increase** — and *the cash flow is identical*: \$30m left the building either way. The difference is entirely in *where* the cost is recorded. On the cash flow statement, the tell is stark: an expensed cost reduces operating cash flow, but a capitalized cost is reclassified as *investing* cash flow (capex), so operating cash flow looks *better* even as the balance sheet swells with a questionable asset. This exact maneuver — capitalizing ordinary operating costs as assets — was the heart of the WorldCom fraud, where billions of line-cost expenses were capitalized to manufacture profits. *Capitalize-versus-expense changes nothing about the cash and everything about the reported profit, which is why a sudden rise in capitalized costs alongside flat cash flow is one of the most important accruals to hunt for.*

## Cookie-jar reserves and big baths

The levers above can be pulled in either direction, and management's most cynical use of them is not to inflate every period but to *manage the trajectory* of earnings — to make a volatile business look like a smooth, predictable one, because markets reward smoothness with higher valuations. Two named techniques dominate.

### Cookie-jar reserves: smoothing earnings across years

A **cookie-jar reserve** is an over-provision built up in a good year and released in a bad year. In a fat year, management over-reserves — booking a larger-than-necessary bad-debt allowance, warranty reserve, or restructuring charge — which *depresses* the fat year's profit and stashes the difference in a balance-sheet reserve (the "cookie jar"). In a lean year, management *releases* the reserve — reversing the over-provision — which *boosts* the lean year's profit. The result is a smooth earnings line that hides the underlying volatility.

The figure below shows it across four years. True earnings (gray bars) swing wildly: \$120m, then \$60m, then \$130m, then \$52m. But by banking surplus in the fat years and refunding it in the lean years, reported earnings (blue bars) march up in a smooth, deceptive line: \$92m, \$96m, \$100m, \$104m. The cash never changed; only the *reported* profit was redistributed across time.

![A bar chart showing true earnings swinging from one hundred twenty million down to sixty up to one hundred thirty and down to fifty-two while reported earnings march smoothly upward as a reserve banks surplus in fat years and releases it in lean years](/imgs/blogs/accruals-vs-cash-why-earnings-are-an-opinion-7.png)

#### Worked example: a cookie-jar reserve smooths two years

Take just Years 1 and 2 from the figure. True (economic) earnings are \$120m in Year 1 and \$60m in Year 2 — a business that had a great year and then a mediocre one, a 50% drop that would alarm any investor.

- **Year 1:** management over-reserves by \$28m (an unnecessarily large warranty and restructuring provision). Reported profit = \$120m − \$28m = **\$92m**. The \$28m sits in the cookie jar.
- **Year 2:** the true result is only \$60m, but management *releases* \$36m of the previously banked reserve (reversing prior over-provisions). Reported profit = \$60m + \$36m = **\$96m**.

To an investor reading the reported numbers, the company grew earnings from \$92m to \$96m — steady, reliable, up 4%. In reality, the business *halved* its profit from \$120m to \$60m. The reserve absorbed the entire swing. No cash was created or destroyed; \$28m of Year 1's real profit was simply relabeled as Year 2's. *Cookie-jar reserves do not change the total profit over time — they only change which year it is reported in, manufacturing an illusion of stability that the cash flow statement, which cannot be smoothed this way, would have exposed.*

### Big baths: resetting the baseline

The mirror image of smoothing is the **big bath**. When a company is going to have a bad year anyway — a new CEO arrives, a recession hits, a division is failing — management has an incentive to make it *as bad as possible*: pile on every write-down, over-reserve aggressively, kitchen-sink every conceivable future cost into this one already-ruined period. The logic is brutal and rational: the market has already written off this year, so an extra billion of charges barely moves the stock further. But those over-reserves become *future* cookie jars: in subsequent years, the excessive charges reverse and boost profit, making the recovery look spectacular and the new management look like heroes.

The big bath is most common around CEO transitions (the incoming CEO blames the outgoing one and resets the baseline low) and at the bottom of recessions. The tell is a single enormous "non-recurring" charge — restructuring, impairment, write-downs — that conveniently sets up easy comparisons for the following years. A skeptical analyst treats a giant one-time charge not as water under the bridge but as a *future* earnings reservoir, and watches the subsequent reversals.

Both cookie jars and big baths are accrual phenomena — they live entirely in the estimate buckets, move reported earnings without moving cash, and (by definition) reverse over time. That reversal is exactly why the accrual anomaly works: aggressively managed earnings, in either direction, do not persist.

## Common misconceptions

A handful of confusions trip up almost everyone learning this material. Clearing them is most of what separates someone who can define "accrual" from someone who can actually use it.

**"More accruals means more profit, which is good."** This is the most natural and most wrong intuition. High *positive* accruals mean reported profit is exceeding cash — that profit is a promise, not a fact, and it tends to reverse. The accrual anomaly says high-accrual firms *underperform*. Low or negative accruals — cash exceeding profit — are the high-quality signal. The sign convention feels backwards, but internalize it: *positive accruals are a warning, not a virtue.*

**"Accrual accounting is just a worse, fuzzier version of cash accounting."** No — accrual accounting is genuinely *better* at measuring a period's economics, because it matches effort to reward instead of letting arbitrary cash timing dominate. The problem is not that accrual is fuzzy; it is that accrual's superior matching *requires* estimates, and estimates can be bent. You want accrual earnings *and* you want to check them against cash. Neither statement alone is enough.

**"A high accrual means the company is committing fraud."** Usually not. The vast majority of high accruals are perfectly legal and often perfectly innocent — a fast-growing company that sells on credit will *naturally* build receivables and inventory faster than cash, producing high accruals during healthy expansion. High accruals are a *prompt to investigate*, not a verdict. The question to ask is *why* accruals are high: benign growth that will convert to cash, or aggressive recognition that will reverse? The accrual flags the question; the footnotes and the trend answer it.

**"Depreciation is the bad kind of accrual."** Depreciation is an accrual, but it pushes accruals in the *reassuring* direction. Depreciation is a non-cash expense that *reduces* reported profit below cash flow — making accruals more negative, the high-quality direction. The dangerous accruals are the ones that push profit *above* cash: aggressive revenue, under-reserving, capitalizing costs. Don't lump all accruals together; the *sign* and the *bucket* matter.

**"You can fully fix earnings by just using cash flow instead."** Cash flow is harder to manipulate, but not impossible (Wirecard fabricated cash itself), and operating cash flow has its own levers — stretching payables, factoring receivables, and especially reclassifying costs from operating to investing (the capitalize trick). The robust approach is not "ignore earnings, trust cash" but "read both, and treat the *gap* between them — the accrual — as the diagnostic." A company can manage CFO upward by capitalizing costs even as accruals look fine; only by reading the whole picture do you catch it.

**"Earnings management is the same as accounting fraud."** They sit on a spectrum, not on two sides of a wall. Choosing the aggressive end of a legitimate estimate range (a longer useful life, a softer bad-debt rate) is *legal* earnings management. Fabricating transactions, booking fictitious revenue, or hiding liabilities is *fraud*. The accrual signal catches both, because both inflate reported profit relative to cash — but the analyst's job is to figure out which side of the line a given company is on, and most of the time the answer is "aggressive but legal," which is still a reason to discount the earnings.

## How it shows up in real markets

The abstractions above are not academic. The accrual — the gap between reported earnings and cash — sits at the center of nearly every major accounting scandal and a great many ordinary investment disappointments. Here is how it shows up in the wild.

**WorldCom and the capitalize lever.** WorldCom inflated profits by roughly \$11 billion, largely through one mechanism from this piece: it capitalized ordinary operating costs (line-lease expenses) as long-lived assets instead of expensing them. The cash outflow was real and unchanged; the *accounting* moved billions of expense off the income statement and onto the balance sheet, manufacturing profit out of thin air. The accrual signal would have screamed — reported earnings vastly exceeding the cash the business produced, with capex inexplicably swollen by costs that should have been operating expenses. WorldCom is the textbook case for *why* the capitalize-versus-expense lever is the most dangerous one.

**Enron and earnings that never became cash.** Enron reported soaring profits in the late 1990s through aggressive mark-to-market accounting and off-balance-sheet vehicles, but its operating cash flow never came close to supporting its reported earnings. The gap between booked profit and real cash — a massive, persistent positive accrual — was visible to anyone tracking the accrual rather than the headline EPS. The full anatomy is in the dedicated [Enron 2001 case study](/blog/trading/finance/enron-2001-accounting-fraud), but the accrual lesson is the durable one: when a company is wildly profitable on paper yet its cash never shows up, the accrual is telling you the profit is an opinion the cash does not share.

**The bank reserve cookie jar.** Banks are the natural home of cookie-jar reserves because their single largest expense estimate — the **loan-loss provision** — is pure judgment. In good years a bank can over-provision (banking earnings), and in bad years it can under-provision or release reserves (boosting earnings). Regulators and the SEC have repeatedly pursued banks for using loan-loss reserves to smooth earnings precisely because the provision is so discretionary. When you analyze a bank, the trajectory of its reserve relative to its actual losses is one of the most important accrual signals you can read.

**The serial acquirer and amortization games.** Companies that grow by acquisition accumulate large intangible assets and goodwill, and the assumptions about their useful lives and impairment become a recurring accrual lever. A serial acquirer can flatter "adjusted" earnings by adding back amortization of acquired intangibles while quietly stretching the lives of others, and can take a big-bath impairment in a down year to reset the baseline. The accrual ratio and the gap between GAAP and "adjusted" earnings are where these games surface.

**Healthy high-accrual growth that is not a fraud at all.** It cuts both ways, and the honest analyst remembers it. A genuinely thriving company in a high-growth phase — adding capacity, building inventory for surging demand, extending credit to win share — will *naturally* run high positive accruals as working capital balloons ahead of collections. This looks identical, on the accrual screen, to the early stage of a channel-stuffing fraud. The difference only becomes clear over time: the healthy company's accruals *convert to cash* as the growth matures, while the fraud's accruals reverse into write-downs. This is why the accrual is a *prompt*, not a verdict, and why you pair it with the cash conversion trend, the receivables-versus-revenue growth, and the footnotes before you draw a conclusion.

## When this matters and further reading

The accrual matters most precisely when the earnings look best — when profit is soaring, the story is compelling, and the headline EPS is beating every estimate. That is exactly when you should compute total accruals (net income minus operating cash flow), scale them into the accrual ratio, and ask the unglamorous question: *how much of this profit is cash, and how much is opinion?* If accruals are negative or small and stable, the earnings are backed by cash and you can take them largely at face value. If accruals are large, positive, and growing — especially faster than the business — you have a reason to be careful, whatever the income statement says, and the burden shifts to *explaining why* the gap is benign.

The discipline is simple to state and hard to apply consistently: earnings are an opinion shaped by estimates, timing, and allocations; cash is a fact you can confirm; and the accrual is the part of the opinion the fact does not yet support. Master the four buckets, the two ways to compute accruals, the accrual ratio, and the levers — useful lives, reserves, capitalize-versus-expense, cookie jars, big baths — and you will read a company's earnings the way a forensic analyst does: not as a number to accept, but as a claim to test against the cash.

To go deeper, the natural next steps within this series are the [cash flow statement: where the cash really comes from](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from) (the statement that supplies the operating-cash-flow half of every accrual calculation), the [income statement line by line](/blog/trading/equity-research/income-statement-line-by-line-revenue-to-net-income) (where the accrual earnings we have been deconstructing are built), [quality of earnings: accruals, one-offs, and red flags](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) (the full forensic toolkit that takes the accrual and adds the rest of the red-flag checklist), and [revenue recognition and expense timing: where discretion hides](/blog/trading/equity-research/revenue-recognition-and-expense-timing-where-discretion-hides) (a deep dive on the single most important accrual bucket). And for the cautionary tale that ties it all together, the [Enron 2001 accounting fraud case study](/blog/trading/finance/enron-2001-accounting-fraud) shows what happens when the accrual is left to grow unchecked until the opinion and the fact can no longer be reconciled.
