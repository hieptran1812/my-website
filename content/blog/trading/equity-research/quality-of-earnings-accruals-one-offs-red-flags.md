---
title: "Quality of Earnings: Accruals, One-Offs, and the Red Flags That Matter"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero guide to judging how real, repeatable, and cash-backed a company's reported profit is — the cash-flow test, recurring vs non-recurring items, revenue and expense quality, the major red-flag families, and the Beneish M-score, all worked in dollars."
tags: ["equity-research", "corporate-finance", "earnings-quality", "accruals", "non-gaap", "beneish-m-score", "red-flags", "forensic-accounting", "financial-statements", "fundamental-analysis"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Two companies can report the exact same EPS while one's earnings are durable cash and the other's are accounting smoke. *Quality of earnings* is the discipline of judging how real, repeatable, and cash-backed a reported profit actually is — and it is the single most valuable analytical skill in equity research.
>
> - **High-quality earnings** are **cash-backed** (operating cash flow keeps up with net income), **repeatable** (they come from the core business, not one-off gains), **conservatively measured** (reserves are full, useful lives honest), and **transparently disclosed** (GAAP and non-GAAP reconcile cleanly).
> - The **cash-flow test** is the master test: compare cumulative operating cash flow to cumulative net income over several years. When earnings keep rising while cash flow flattens, the widening gap is accruals you have not collected — the most reliable warning sign in all of analysis.
> - **Recurring vs non-recurring** is where managed numbers hide: restructuring charges that recur every year, gains on asset sales, one-time tax benefits, and "adjusted" figures that add back genuinely recurring costs like stock-based compensation.
> - The **red flags cluster into families**: earnings beating cash, rising DSO and inventory days, serial one-offs, aggressive non-GAAP, restatements and auditor changes, related-party deals, and suspiciously smooth numbers that just beat estimates. One flag is noise; several from different families pointing the same way is signal.
> - The **Beneish M-score** combines eight ratios into one number that statistically separates manipulators from honest firms. It is a screen, not a verdict — but it puts numbers on the intuition. Learn to score earnings on all four dimensions, and you read past the headline EPS to the business underneath.

Two companies hand you their annual reports. Both earned exactly **\$1.00 per share** last year. Both grew that figure at a steady clip for five years. Both trade at twenty times earnings. On the surface — the only surface most investors ever look at — they are twins. Buy either one and you are paying the same price for the same earnings.

They are not twins. One of them collected \$1.15 of cash for every dollar of reported profit, earned it all from selling its actual product to actual customers, set aside full reserves for everything that might go wrong, and disclosed its numbers so plainly you could rebuild them from scratch. The other collected only thirty-five cents of cash per dollar of profit, booked a fifth of its "earnings" from selling a building and a one-time tax windfall, quietly stretched the assumed life of its factories to shave the depreciation charge, and reports an "adjusted" EPS on its slides that is more than double what the accounting rules actually allow. The first company's dollar of earnings is worth something close to a dollar. The second company's dollar is worth maybe fifty cents — and falling.

The skill that tells these two apart is called **quality of earnings**, and it is, without much competition, the most valuable analytical skill in this entire series. Valuation models, growth forecasts, multiple comparisons — all of them take the reported earnings number as an input. If that input is half smoke, every model built on top of it is wrong by half, and no amount of spreadsheet sophistication will save you. Quality of earnings is the discipline of interrogating the number itself, *before* you trust it: asking how real it is, how repeatable, how conservatively measured, how honestly disclosed. The figure below is the mental model — two firms, one EPS, opposite quality.

![A comparison of two firms each reporting one dollar of EPS where the low quality firm is built on weak cash backing one time gains and released reserves while the high quality firm is cash backed recurring and conservative](/imgs/blogs/quality-of-earnings-accruals-one-offs-red-flags-1.png)

We will build this from nothing. You do not need an accounting background. By the end you will understand what "earnings quality" actually means and how to score it on four dimensions; the cash-flow test that anchors the whole analysis; how to strip recurring "one-time" charges to find true earnings power; how to read revenue and expense quality from the footnotes; the full taxonomy of red flags and how to weigh them; how to compute the Beneish M-score by hand; and the crucial line between aggressive-but-legal and outright fraudulent. We will keep two recurring companies throughout — **Northwind Industries**, the high-quality firm you would happily own, and **Riverstone Equipment**, the low-quality twin that looks identical until you look hard. They are the same two companies from the companion piece on [why earnings are an opinion](/blog/trading/equity-research/accruals-vs-cash-why-earnings-are-an-opinion); here we turn that opinion into a checklist. Let us start with the foundations.

## Foundations: what "earnings quality" actually means

Before we can judge earnings quality we need to define it precisely, because the phrase gets used loosely. "High-quality earnings" does not mean "high earnings" — a company can earn a fortune from low-quality sources, and a company can earn modestly from impeccably high-quality ones. Quality is about the *character* of the profit, not its size. Four dimensions, each independently testable, together make up earnings quality. The figure below lays them out; we will define each from zero.

![A four dimension framework showing earnings quality as cash backed repeatable conservative and transparent each with a test that passes and a failure mode that signals managed numbers plus a scorecard that combines them](/imgs/blogs/quality-of-earnings-accruals-one-offs-red-flags-2.png)

### Dimension 1 — Cash-backed

The first and most important dimension is whether the reported profit shows up as **cash**. Reported earnings are built on accrual accounting, which records revenue when it is *earned* (a sale is made) and expenses when they are *incurred* — regardless of when cash actually moves. That is the right way to measure a period's economics, but it means reported profit is full of items that have not yet become cash: sales booked but not collected (receivables), costs estimated but not yet paid (accruals), and non-cash charges like depreciation. The gap between reported earnings and the cash the business actually generated is the **accrual**, and it is the central concept in earnings quality.

High-quality earnings are *cash-backed*: over time, operating cash flow keeps up with net income, so the profit you see on the income statement is matched by real money in the bank. Low-quality earnings are not: profit runs ahead of cash year after year, which means a growing chunk of "earnings" is sitting in receivables and inventory rather than in the account. The cash flow statement, covered in depth in its [own companion piece](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from), is where this dimension is measured. We will make it concrete shortly.

### Dimension 2 — Repeatable

The second dimension is whether the profit will *happen again*. The whole point of valuing a business on its earnings is the assumption that those earnings recur — that next year looks something like this year. So a dollar of profit from selling your product to a repeat customer is worth far more than a dollar from selling a building you will only ever sell once, or from a tax benefit that will not return, or from releasing a reserve you built up in a prior year. The first dollar is a *run-rate*; the second is a *windfall*. Repeatable earnings come from the core, recurring operations of the business. Non-recurring earnings are real cash, often, but they are one-time — and paying twenty times earnings for a one-time gain means paying twenty dollars for one dollar that will never come back.

### Dimension 3 — Conservatively measured

The third dimension is whether the *estimates* baked into the earnings lean conservative or aggressive. Accrual accounting requires dozens of judgment calls: How long will this factory last (which sets the depreciation charge)? What fraction of receivables will go bad (which sets the bad-debt expense)? How much will we owe in warranty claims? Each of these is an estimate, and each can be made optimistically (lifting current profit) or prudently (depressing it). A company that uses honest, even slightly conservative, assumptions produces earnings you can trust. A company that stretches every assumption to its limit — lengthening useful lives, softening bad-debt allowances, under-reserving for known problems — produces earnings that are technically legal but inflated, and that will reverse when the optimistic assumptions meet reality.

### Dimension 4 — Transparently disclosed

The fourth dimension is whether you can actually *see* what is going on. High-quality companies disclose plainly: their footnotes are clear, their non-GAAP adjustments reconcile cleanly to GAAP, their segments are broken out, their related-party transactions are minimal and explained. Low-quality companies obscure: vague footnotes, large unexplained "other" line items, aggressive adjusted metrics with sketchy reconciliations, frequent restatements, and revenue or cash routed through entities they control. Transparency is partly a quality in its own right and partly a *meta*-signal: a management team that hides things usually has something to hide. The art of reading what they disclose — and noticing what they don't — is the subject of the [10-K footnotes companion](/blog/trading/equity-research/reading-the-10k-footnotes-and-mda).

### Putting the four together

These four dimensions are not independent in practice — a firm that is bending its numbers usually fails several at once — but they *are* separable as tests, and that is what makes them useful. A high-quality firm passes all four: cash-backed, repeatable, conservative, transparent. A genuinely troubled firm fails most of them. The analyst's job is to score each dimension and combine them into a judgment about how much to trust the headline number. With the four dimensions defined, let us go deep on each, starting with the one that matters most.

## The cash-flow test: the master test of earnings quality

If you only ever run one earnings-quality test, run this one: **compare net income to operating cash flow, over several years.** Net income is the opinion; operating cash flow is much closer to a fact (it can still be manipulated, but far less easily). When the two track each other, earnings are cash-backed and probably high-quality. When they diverge — net income climbing while operating cash flow flattens or falls — the gap is accruals, and a widening accrual gap is the most reliable warning sign in all of equity analysis.

The figure below shows the master red flag. Net income marches up and to the right, year after year, looking like a wonderful growth story. Operating cash flow stays stubbornly flat. The space between the two lines is the accrual — profit reported but not collected — and it gets wider every year. By year five the company "earned" \$145 million but generated only \$48 million of operating cash. Where did the other \$97 million go? Into receivables that may never be collected, inventory that may never sell, and reserves that were quietly released. The income statement says boom; the cash flow statement says nothing happened.

![A line chart over five fiscal years where net income rises steeply from fifty million to one hundred forty five million while operating cash flow stays flat near forty eight million and the widening red gap between them marks uncollected accrual profit](/imgs/blogs/quality-of-earnings-accruals-one-offs-red-flags-3.png)

There are two ways to make the cash-flow test quantitative, and you should use both.

**The accrual ratio.** The cleanest single number is total accruals scaled by assets:

$$
\text{Accrual ratio} = \frac{\text{Net income} - \text{Cash flow from operations}}{\text{Average total assets}}
$$

A negative ratio (cash exceeds profit) is high quality. A small positive ratio is normal. A large positive ratio — say above 10% — is a red flag. The mechanics, the four buckets accruals hide in, and Sloan's famous finding that high-accrual firms underperform are all worked out in detail in the [accruals companion piece](/blog/trading/equity-research/accruals-vs-cash-why-earnings-are-an-opinion); here we just need the ratio as one input to the quality scorecard.

**The cumulative coverage ratio.** Even simpler and harder to game across years: add up several years of operating cash flow and divide by several years of net income.

$$
\text{Cash coverage} = \frac{\sum \text{operating cash flow}}{\sum \text{net income}}
$$

A healthy company converts most of its earnings to cash over time, so this ratio sits near or above 1.0. A ratio that drifts well below 1.0 — say 0.6 or lower — over a multi-year window means a persistent, structural gap between reported profit and cash. That is the signature of low-quality earnings, and the multi-year window defeats the "it was just timing" excuse, because timing differences wash out over several years while genuine quality problems do not.

#### Worked example: the cash-flow test on two identical-EPS firms

Both Northwind and Riverstone report **net income of \$50 million** this year, on **50 million shares**, so both show **\$1.00 of EPS**. An investor looking only at the income statement cannot tell them apart. Now run the cash-flow test.

**Northwind Industries.**
- Net income: \$50m
- Operating cash flow: \$57.5m
- Cash coverage = \$57.5m ÷ \$50m = **1.15**
- Accrual ratio = (\$50m − \$57.5m) ÷ \$500m average assets = **−1.5%**

Northwind collects \$1.15 of cash for every \$1.00 of reported profit. Its accrual ratio is *negative* — cash exceeds earnings, the hallmark of high quality, driven mostly by depreciation (a non-cash expense that depresses profit but not cash).

**Riverstone Equipment.**
- Net income: \$50m
- Operating cash flow: \$17.5m
- Cash coverage = \$17.5m ÷ \$50m = **0.35**
- Accrual ratio = (\$50m − \$17.5m) ÷ \$500m average assets = **+6.5%**

Riverstone collects only thirty-five cents of cash per dollar of "profit." Thirty-two and a half million dollars of its reported earnings never became cash — it is sitting in receivables and inventory. Same \$1.00 EPS, completely different reality underneath.

*Identical reported EPS can sit on top of cash coverage of 1.15 or 0.35; the income statement hides the difference and the cash-flow test exposes it.*

This is why the cash-flow test comes first. It is fast, it is hard to fool over multiple years, and it catches the single most common quality problem — profit running ahead of cash. Everything else in this post refines and extends it.

## Recurring vs non-recurring: stripping the one-offs

The second dimension is repeatability, and the central skill here is separating **recurring** earnings (the core, run-rate business) from **non-recurring** items (one-offs that will not return). This matters enormously for valuation, because you should pay a high multiple for repeatable earnings and a multiple of *one* — just the cash itself — for one-time gains. A company that earns \$100 million of repeatable operating profit is worth far more than one that earns \$60 million of repeatable profit plus \$40 million from selling a division, even though both "earned \$100 million."

The trouble is that managers know this, and they have learned to dress up non-recurring items as recurring (to inflate the run-rate) and recurring items as non-recurring (to inflate "adjusted" earnings). The whole game lives in this classification. Here are the items to watch.

**Gains and losses on asset sales.** When a company sells a building, a division, or a portfolio of investments for more than its book value, the gain flows through earnings. It is real cash — but it will not happen again next year. A company that pads its earnings with a steady stream of "gains on sale" is borrowing from its balance sheet to feed its income statement; eventually it runs out of things to sell.

**One-time tax benefits.** A favorable tax ruling, the release of a tax reserve, or a one-time benefit from a change in tax law can drop straight to net income and boost EPS. It is non-recurring almost by definition. A quarter where EPS beat estimates "thanks to a lower tax rate" deserves a hard look — the operating business may have missed.

**Litigation and insurance settlements.** A large one-time recovery (winning a lawsuit, collecting an insurance claim) inflates earnings in the period it is recognized. The reverse — a large one-time *charge* to settle a lawsuit — depresses it. Both are non-recurring and both should be stripped to see the underlying trend.

**Restructuring charges — the recurring "one-time" item.** This is the most important and most abused category, so it gets its own treatment. A restructuring charge — severance, plant closures, write-offs from a reorganization — is *presented* as a one-time, non-recurring cost. And a genuine, once-a-decade reorganization is. But many companies take a "restructuring" charge **every single year**, year after year, as a normal feature of how they run the business. At that point the charge is not non-recurring at all — it is a recurring operating cost the company is excluding from its "adjusted" numbers to make the core look more profitable than it is.

The figure below makes this vivid. Look down the columns: restructuring, impairment, litigation, and stock comp charges appear in nearly every year. The company calls each one "special" and adds it back to reach "adjusted" earnings. But a cost that arrives every year is, by definition, a cost of doing business.

![A matrix over five fiscal years showing restructuring impairment litigation and stock compensation charges recurring in almost every year proving that supposedly one time charges are really an ongoing operating cost being excluded from adjusted earnings](/imgs/blogs/quality-of-earnings-accruals-one-offs-red-flags-4.png)

#### Worked example: stripping serial one-offs to find true earnings power

Riverstone reports **\$50 million of net income** this year and presents it as clean operating profit. But read the footnotes and the past five years of filings, and a pattern emerges. Over the last five years Riverstone has reported, every single year, items it labels "non-recurring":

| Item | This year | Recurs? |
|---|---|---|
| Reported net income | \$50m | — |
| Less: gain on sale of a distribution center | (\$12m) | No — one-time, strip it out |
| Less: one-time tax benefit | (\$8m) | No — one-time, strip it out |
| Add back: "restructuring" charge taken **every year** | \$0m | Yes — it is recurring, leave it in earnings |
| Add back: litigation charge taken **4 of last 5 years** | \$0m | Mostly recurring, leave it in earnings |

Now reconstruct true, repeatable earnings power. The gain on sale (\$12m) and the tax benefit (\$8m) are genuine one-offs and should be *removed* — they will not return. That alone drops repeatable earnings to **\$50m − \$12m − \$8m = \$30 million.**

Then notice what Riverstone *wants* you to do: add back the recurring "restructuring" and litigation charges to lift the number. But those charges arrive every year, so they belong *in* the earnings, not added back. Riverstone's "adjusted" EPS adds them back to claim \$1.30 of EPS; the honest figure, with one-time gains removed and recurring charges left in, is closer to **\$0.60.**

*A company padding its results with one-time gains while excluding recurring charges as "special" can report \$1.00 of EPS on top of only \$0.60 of true, repeatable earnings power — and the gap only shows up when you widen the lookback to several years.*

The technique generalizes: pull five years of filings, list every item the company called "one-time" or "non-recurring," and ask which ones actually recurred. The serial offenders — the restructuring charge that arrives like clockwork, the "unusual" item that is suspiciously usual — are recurring costs in disguise, and you should fold them back into your estimate of normalized earnings.

## Revenue quality: where the top line bends

Earnings start with revenue, and revenue is the most common place for quality to erode — because recognizing revenue earlier and more aggressively is the easiest way to lift reported profit without doing more business. A few specific signals tell you whether revenue is high or low quality.

**Days sales outstanding (DSO) and its trend.** DSO measures how long, on average, it takes to collect a sale: `DSO = (accounts receivable ÷ revenue) × 365`. A stable DSO means the company collects on a consistent schedule. A *rising* DSO means receivables are growing faster than sales — customers are taking longer to pay, or, more worryingly, the company is recognizing revenue it has not really earned (booking sales to customers who have not committed, or shipping product to distributors who do not need it, a practice called *channel stuffing*). Rising DSO is one of the cleanest revenue-quality red flags, because it shows up as receivables ballooning ahead of revenue.

**Deferred revenue trends.** Deferred revenue is the opposite of a receivable — it is cash collected *before* the product or service is delivered, sitting as a liability until it is earned. For a subscription or software business, *growing* deferred revenue is a sign of strength (customers are pre-paying for future service). *Shrinking* deferred revenue while reported revenue rises is a warning: the company may be recognizing previously-deferred revenue faster than it is signing new business, borrowing from the future to make the present look good.

**Gross vs net revenue.** Some companies report the full dollar value of a transaction as revenue (gross) when they should report only the commission or margin they actually keep (net). A marketplace that processes \$1 billion of transactions but keeps only \$50 million in fees should report \$50 million of revenue, not \$1 billion. Reporting gross when net is appropriate inflates the top line dramatically without adding a dollar of profit, and it makes growth look faster than it is.

**Related-party sales.** When a company sells to entities it controls or is otherwise connected to, the "revenue" may not be a real arm's-length transaction. Round-trip deals — selling product to a related entity that books it as a purchase, with cash quietly cycling back — can manufacture revenue out of thin air. Heavy related-party revenue is a serious quality concern and a frequent feature of outright frauds.

#### Worked example: DSO and deferred revenue reveal pulled-forward revenue

Riverstone reports revenue growth that looks impressive — up 25% this year — and management is taking a victory lap. But two numbers from the balance sheet tell a different story.

**DSO.** Last year, Riverstone had \$200m of revenue and \$33m of receivables:
- DSO = (\$33m ÷ \$200m) × 365 = **60 days**

This year, revenue grew to \$250m but receivables jumped to \$75m:
- DSO = (\$75m ÷ \$250m) × 365 = **110 days**

DSO leapt from 60 days to 110 days. Receivables grew **127%** while revenue grew only **25%**. The company is booking sales far faster than it is collecting them — a classic sign that revenue is being recognized aggressively or pulled forward from future periods.

**Deferred revenue.** Meanwhile, Riverstone's deferred revenue *fell* from \$40m to \$22m even as reported revenue rose. The company recognized \$18m of previously-deferred revenue this year — revenue it had collected in prior periods — to help hit its growth number, while signing less new pre-paid business.

Put the two together. Of Riverstone's \$50m of revenue *growth*, roughly \$42m of receivables build-up and \$18m of deferred-revenue release suggest a large fraction of that "growth" is timing, not durable demand. Strip the pulled-forward revenue and the underlying business grew far less than 25%.

*A 25% revenue jump that arrives alongside DSO nearly doubling and deferred revenue falling is mostly pulled-forward timing, not real growth — and the balance sheet, not the income statement, is what exposes it.*

The lesson: never look at revenue growth in isolation. Always check whether receivables and deferred revenue are moving in a way that supports or undercuts the top line. High-quality revenue growth is matched by stable DSO and (for subscription models) growing deferred revenue. Low-quality growth shows up as receivables and DSO ballooning while deferred revenue drains.

## Expense quality: the costs that quietly disappear

If revenue quality is about recognizing income too fast, expense quality is about recognizing costs too slowly — or not at all. Every dollar of cost a company can defer, capitalize, or exclude is a dollar that drops to reported profit. Four expense-side levers matter most.

**Capitalizing vs expensing.** When a company spends money, it either *expenses* it (the full cost hits this period's income statement) or *capitalizes* it (the cost goes onto the balance sheet as an asset and is expensed gradually over future years as depreciation or amortization). Capitalizing a cost that should be expensed is one of the most powerful manipulation levers, because it removes the cost from *this* period's earnings entirely. The textbook case is treating ordinary operating expenses as if they were long-lived investments. WorldCom famously capitalized billions of dollars of routine line costs that should have been expensed, manufacturing profit out of an accounting reclassification. Watch for capitalized software development, capitalized interest, and capitalized "customer acquisition costs" growing faster than the business.

**Understated reserves.** A reserve (or allowance) is an estimate set aside for a future cost: the allowance for doubtful accounts (receivables expected to go bad), warranty reserves, restructuring reserves. Because reserves are estimates, under-reserving lifts current profit. A company that knows 5% of its receivables will go bad but reserves for only 2% reports higher earnings now — and a nasty write-off later when reality catches up. The reverse, *over*-reserving in a good year to release the reserve in a bad year, is the **cookie-jar reserve**, a smoothing trick that makes earnings look more stable than the business really is.

**Pension and other long-term assumptions.** Companies with defined-benefit pension plans must assume a long-term rate of return on the plan's assets. A higher assumed return lowers the pension cost that hits earnings — with no change in the actual plan. Assuming an aggressive 9% return when peers assume 6% manufactures profit out of an assumption buried deep in the footnotes. The same logic applies to discount-rate and useful-life assumptions throughout the financials.

**Stock-based compensation ignored in "adjusted" earnings.** This is the single most common and most consequential expense-quality issue in modern equity research, so it deserves emphasis. Stock-based compensation (SBC) — paying employees in stock and options rather than cash — is a *real cost*. It transfers ownership of the company from existing shareholders to employees; that dilution is as real as a cash salary. GAAP rightly requires SBC to be expensed. But a huge number of companies, especially in technology, add SBC *back* to reach an "adjusted" or "non-GAAP" earnings figure, on the theory that it is non-cash. The "non-cash" defense is misleading: SBC does not consume cash precisely because the company is paying with ownership instead — the cost is dilution, not a cash outflow, but it is a cost all the same. A company whose "adjusted" earnings are wildly higher than GAAP, almost entirely because of added-back SBC, is reporting a number that ignores one of its largest real expenses.

#### Worked example: an "adjusted EBITDA" that adds back \$80M of recurring costs

Riverstone's investor presentation leads with **Adjusted EBITDA of \$260 million**, up nicely from last year, and the stock trades on a multiple of that figure. Let us reconcile it to GAAP and see what is inside the adjustment.

Start from GAAP operating income and the standard EBITDA build:

| Line | Amount |
|---|---|
| GAAP operating income | \$80m |
| Add back: depreciation and amortization | \$100m |
| **EBITDA (standard)** | **\$180m** |
| Add back: stock-based compensation | \$50m |
| Add back: "restructuring" charges (taken every year) | \$22m |
| Add back: "one-time" integration costs (third year running) | \$8m |
| **Adjusted EBITDA (as presented)** | **\$260m** |

The \$80m of add-backs beyond standard EBITDA — \$50m of SBC, \$22m of recurring restructuring, \$8m of recurring "integration" — turns a \$180m figure into a \$260m headline, a **44% inflation**. And every one of those add-backs is a *recurring* cost: SBC is paid every year and dilutes shareholders every year; restructuring has hit every year for five years; integration costs are in their third straight year. None of them is genuinely one-time.

Strip the recurring add-backs and Riverstone's real cash-generative EBITDA is closer to **\$180m**, not \$260m. An investor paying ten times "Adjusted EBITDA" is paying \$2.6 billion; ten times the honest figure is \$1.8 billion. The \$800 million difference is the cost of believing the adjusted number.

*An "adjusted EBITDA" that adds back \$80m of stock comp and serial restructuring is not a measure of cash earnings — it is a measure of what management wishes its earnings were, and the gap to GAAP is the size of the optimism.*

The defensive habit: never accept an "adjusted" or "non-GAAP" figure without reading the reconciliation line by line and asking, of each add-back, *does this cost recur?* If it does, it belongs in earnings. SBC almost always recurs. Restructuring that happens every year recurs. Genuine one-offs — a single acquisition's deal fees, a one-time legal settlement — can fairly be excluded; serial "one-offs" cannot.

## The red-flag families: how earnings quality really fails

We have now covered the four dimensions and the specific revenue and expense levers. The practical question is: when you sit down with a company's filings, what are you actually looking for? The answer is that earnings-quality problems cluster into a handful of recognizable **families**, and the discipline is to scan for all of them. The figure below organizes the seven that matter most.

![A grid of seven red flag families covering earnings beating cash working capital bloat serial one offs aggressive non gaap disclosure decay related party deals and too smooth numbers with a rule that confluence across families is the real signal](/imgs/blogs/quality-of-earnings-accruals-one-offs-red-flags-5.png)

**Family 1 — Earnings beating cash.** The master family, already covered: net income rising while operating cash flow stays flat, a climbing accrual ratio, cash coverage drifting below 1.0 over multiple years. This is where most quality problems eventually show up, because almost every form of earnings management widens the gap between profit and cash.

**Family 2 — Working-capital bloat.** Receivables (DSO) and inventory (days inventory outstanding, or DIO) growing faster than revenue. Rising DSO suggests aggressive revenue recognition or collection problems; rising DIO suggests weakening demand or obsolete stock carried at inflated values. Both tie up cash and both inflate accrual earnings. When DSO and DIO both rise together while revenue growth slows, the working-capital build is funding the income statement.

**Family 3 — Serial one-offs.** "Non-recurring" charges that recur every year — restructuring, impairment, "unusual" items — used to flatter adjusted earnings, as worked above. The tell is frequency: open five years of filings and count how many years carry a "special" item. If the answer is "most of them," the items are not special.

**Family 4 — Aggressive non-GAAP.** Adjusted EPS or adjusted EBITDA far above the GAAP figure, with the gap driven by adding back recurring costs like SBC and serial restructuring. The bigger the gap between the headline metric and GAAP, and the more it depends on recurring add-backs, the lower the quality.

**Family 5 — Disclosure decay.** The meta-signals that something is being hidden: financial restatements (the company had to correct prior numbers), auditor changes (especially a respected auditor resigning), late filings, a sudden increase in vague "other" line items, and footnotes that get longer and murkier. A single auditor change can be innocent; an auditor resignation followed by a restatement is a five-alarm fire.

**Family 6 — Related-party deals.** Revenue, purchases, or financing routed through entities the company or its insiders control. These transactions may not be arm's-length, and round-trip structures can manufacture revenue or hide losses. Heavy related-party activity is a hallmark of the most serious frauds — it was central to both Enron and Wirecard.

**Family 7 — Too-smooth numbers.** Earnings that are suspiciously stable, that *just* beat consensus estimates quarter after quarter by a penny, that hit suspiciously round numbers, or that show far less volatility than the underlying business should produce. Real businesses are lumpy; earnings that are too smooth have usually been smoothed, via cookie-jar reserves and timing adjustments. A company that beats by exactly one cent for twenty straight quarters is managing the number, not reporting it.

**The rule of confluence.** Here is the single most important judgment principle in this entire post: *no single red flag convicts.* Any one of these families has innocent explanations. Rising DSO might be a deliberate, healthy move into a market with longer payment terms. A restructuring charge might be a genuine once-a-decade reorganization. An auditor change might be a routine fee dispute. What turns red flags into a real warning is **confluence** — several flags, from *different* families, all pointing the same way. Rising DSO *and* falling deferred revenue *and* a climbing accrual ratio *and* serial restructuring *and* an auditor who just resigned is not five coincidences. It is a pattern, and the pattern is what you are looking for. One flag is noise; a cluster from independent families is signal.

## The Beneish M-score: putting numbers on the intuition

Everything so far has been qualitative judgment dressed in a few ratios. The Beneish M-score, developed by accounting professor Messod Beneish, turns the intuition into a single statistical score. It was built by studying companies that had been caught manipulating earnings and finding the combination of ratios that best separated them from honest firms. The model is not a verdict — plenty of high-scoring firms are honest, and some manipulators score low — but as a *screen* it is remarkably effective, and it forces you to compute things you might otherwise skip. (It is most famous for having flagged Enron as a likely manipulator from its public financials, before the collapse.)

The M-score combines **eight ratios**, each comparing this year to last year, each capturing one way earnings get bent. The figure below catalogs them.

![A matrix of the eight Beneish M score components listing days sales in receivables gross margin asset quality sales growth depreciation selling and admin expense leverage and total accruals each paired with the manipulation signal it captures](/imgs/blogs/quality-of-earnings-accruals-one-offs-red-flags-6.png)

The eight components, in plain language:

1. **DSRI** — Days Sales in Receivables Index. Are receivables ballooning relative to sales? A jump suggests pulled-forward or fictitious revenue. (This is the DSO red flag, formalized.)
2. **GMI** — Gross Margin Index. Are gross margins deteriorating? A company whose margins are falling has a stronger *motive* to manage the rest of its numbers.
3. **AQI** — Asset Quality Index. Are "soft" assets (everything other than current assets and plant) growing as a share of the balance sheet? Rising soft assets can mean costs are being capitalized instead of expensed.
4. **SGI** — Sales Growth Index. Is the company growing fast? Fast growth itself is not manipulation, but it creates pressure to *keep* growing, which is when management cracks.
5. **DEPI** — Depreciation Index. Is the depreciation rate slowing? A falling rate suggests useful lives are being stretched to reduce the depreciation charge and lift profit.
6. **SGAI** — Selling, General & Administrative Expenses Index. Is overhead rising faster than sales? Disproportionate overhead growth signals a business under strain — and a motive to manage.
7. **LVGI** — Leverage Index. Is leverage increasing? Rising debt can mean covenant pressure, which is a classic motive to inflate earnings.
8. **TATA** — Total Accruals to Total Assets. The accrual ratio in disguise — the core signal that profit is running ahead of cash.

The eight are combined in a weighted formula (the weights come from Beneish's original statistical fit):

$$
M = -4.84 + 0.92\,\text{DSRI} + 0.528\,\text{GMI} + 0.404\,\text{AQI} + 0.892\,\text{SGI}
$$
$$
+\; 0.115\,\text{DEPI} - 0.172\,\text{SGAI} + 4.679\,\text{TATA} - 0.327\,\text{LVGI}
$$

The threshold to remember is **−2.22**. A score *above* −2.22 (less negative, e.g. −1.5) flags the firm as a likely manipulator; a score *below* −2.22 (more negative, e.g. −3.0) suggests the firm is probably not manipulating. The large positive weight on TATA (4.679) tells you what the model "believes": accruals are the heart of the matter, exactly as the cash-flow test said.

#### Worked example: computing a Beneish M-score that flags manipulation

Let us score Riverstone. From its filings we compute the eight indices (each is this-year-over-last-year, so 1.0 means "no change"):

| Component | Value | Reading |
|---|---|---|
| DSRI | 1.85 | receivables way up vs sales — big flag |
| GMI | 1.20 | margins deteriorating |
| AQI | 1.15 | soft assets rising — capitalization |
| SGI | 1.25 | growing 25% — pressure to continue |
| DEPI | 1.10 | depreciation slowing — lives stretched |
| SGAI | 1.05 | overhead up slightly |
| LVGI | 1.10 | leverage rising |
| TATA | 0.08 | accruals 8% of assets — the core signal |

Plug into the formula:

- Constant: −4.84
- 0.92 × 1.85 = +1.702
- 0.528 × 1.20 = +0.634
- 0.404 × 1.15 = +0.465
- 0.892 × 1.25 = +1.115
- 0.115 × 1.10 = +0.127
- −0.172 × 1.05 = −0.181
- 4.679 × 0.08 = +0.374
- −0.327 × 1.10 = −0.360

Sum: −4.84 + 1.702 + 0.634 + 0.465 + 1.115 + 0.127 − 0.181 + 0.374 − 0.360 = **−0.96**

Riverstone's M-score is **−0.96**, well above the −2.22 threshold. The model flags it as a likely manipulator. Notice *why*: the two biggest positive contributions come from DSRI (receivables ballooning, +1.70) and SGI (fast growth, +1.12) — exactly the revenue-quality problems we found by hand. The M-score did not tell us anything the qualitative analysis missed; it *confirmed* it, with a number, and it would have caught the pattern even if we had only had the financials and no intuition.

*The Beneish M-score is the red-flag families compressed into one weighted number; a score above −2.22 does not prove fraud, but it tells you to do the deep work — and it puts a quantitative floor under a judgment that is otherwise all art.*

Two cautions on the M-score. First, it is a *screen*, not a verdict: a high score means "investigate," not "guilty." Honest fast-growing companies can score high (rapid growth and rising receivables look the same as manipulation to the model), and patient frauds that keep accruals low can score below the threshold. Second, it works best on manufacturing and product companies, where the ratios behave as the model assumes; it is less reliable for banks, insurers, and asset-light businesses whose balance sheets do not fit the template. Use it as one input to the scorecard, never as the whole answer.

## Aggressive vs fraudulent: the line that matters

We have used the word "manipulation" loosely, and now we need to draw the most important distinction in this entire field: the line between **aggressive accounting** and **fraud**. They look similar on the surface — both inflate reported earnings — but they are categorically different, and confusing them will make you either too paranoid or too trusting.

**Aggressive accounting is legal.** It uses the *discretion the accounting rules genuinely allow* to present results in the most favorable light. Choosing the longest defensible useful life for an asset, reserving at the low end of a reasonable range for bad debts, recognizing revenue at the earliest defensible moment, adding back every arguably-non-recurring cost to reach "adjusted" earnings — all of these are within the rules. A reasonable auditor will sign off on them. They are not crimes. But they are *choices*, and a company that makes the aggressive choice on every single judgment call is telling you something about its character and producing earnings that are systematically inflated and prone to reverse. Aggressive accounting is the territory of most of this post — it is legal, common, and detectable, and detecting it is the everyday work of equity research.

**Fraud is illegal.** It involves *reporting things that are not true* — fictitious revenue, fake customers, forged bank statements, related-party round-trips designed to deceive, hidden liabilities moved off the balance sheet. Fraud crosses from "optimistic interpretation of the rules" to "lying about the facts." Enron's off-balance-sheet special-purpose entities that hid debt and manufactured earnings, Wirecard's roughly €1.9 billion of cash that simply did not exist, WorldCom's capitalization of operating costs that no honest reading of the rules permitted — these are frauds, not aggressive choices.

The line matters for two reasons. First, **most companies you analyze are aggressive, not fraudulent.** Outright fraud is rare; aggressive accounting is everywhere. If you treat every aggressive choice as evidence of fraud, you will never invest in anything. The job is usually to judge *how* aggressive a company is and discount its earnings accordingly — not to assume it is lying. Second, **the techniques that detect aggressive accounting are the same ones that detect fraud, just earlier.** A company sliding from aggressive toward fraudulent shows the same signals — earnings beating cash, ballooning receivables, serial one-offs, related-party deals — getting worse over time. The cash-flow test, the red-flag families, and the M-score catch both; they just catch fraud as the extreme tail of the same distribution. The deep, dedicated techniques for the fraud end of the spectrum — confirming cash exists, tracing related-party webs, spotting forged documents — are the subject of the [forensic accounting companion piece](/blog/trading/equity-research/forensic-accounting-spotting-manipulation-and-fraud); this post is the everyday quality screen that flags the cases worth investigating that deeply.

## Building an earnings-quality scorecard

Let us pull everything together into a repeatable process. An earnings-quality scorecard is just a structured way to score each of the four dimensions, gather the red flags, and arrive at a single judgment: *how much do I trust this earnings number, and how should I adjust it?*

**Step 1 — Cash backing (the master test).** Compute cash coverage (sum of operating cash flow ÷ sum of net income) over five years and the accrual ratio for the latest year. Coverage near or above 1.0 and a negative or low accrual ratio: high quality. Coverage well below 1.0 and a rising accrual ratio: low quality. This is the most heavily weighted line in the scorecard.

**Step 2 — Repeatability.** Pull five years of filings and list every item labeled "non-recurring," "special," "one-time," or "unusual." Count how many years each type appears. Strip genuine one-time *gains* from earnings; fold genuine recurring *charges* back in. Compute normalized, repeatable earnings power and compare it to the headline.

**Step 3 — Conservatism.** Check the estimates: is DSO stable or rising? Is the bad-debt reserve adequate relative to receivable aging? Are useful lives in line with peers (a falling depreciation rate is a flag)? Are pension assumptions reasonable? Each aggressive assumption is a point against quality.

**Step 4 — Transparency.** Read the non-GAAP reconciliation and quantify the gap between adjusted and GAAP earnings, and what drives it. Check for restatements, auditor changes, late filings, related-party transactions, and rising "other" line items. Murky disclosure is its own demerit and a meta-signal.

**Step 5 — Red-flag scan and confluence.** Run the seven red-flag families. Count how many fire and, crucially, whether they come from *different* families pointing the same way. A single flag: note it. A cluster from independent families: serious concern.

**Step 6 — The M-score (quantitative cross-check).** Compute the Beneish M-score as an independent confirmation. Above −2.22: the model agrees there is something to investigate. Below: one reassuring data point, but not a clean bill of health.

**Step 7 — The verdict.** Combine the six into a single judgment and, most importantly, *act on it*. High quality: trust the reported number and value the business normally. Low quality: discount the earnings (use normalized, cash-backed figures, not the headline), demand a larger margin of safety, or simply pass. The goal is not to label companies "good" or "fraud" but to calibrate how much of the reported number to believe.

#### Worked example: the full scorecard on Northwind vs Riverstone

Run both companies through the scorecard. Both report **\$1.00 of EPS.**

| Dimension | Northwind | Riverstone |
|---|---|---|
| **Cash backing** | Coverage 1.15, accrual ratio −1.5% | Coverage 0.35, accrual ratio +6.5% |
| **Repeatability** | All from core operations | \$0.20 from one-time gains and tax |
| **Conservatism** | DSO stable at 58 days; full reserves | DSO 60 → 110 days; reserves released |
| **Transparency** | Adjusted ≈ GAAP; clean footnotes | Adjusted \$1.40 vs GAAP \$0.60; SBC added back |
| **Red flags** | None fire | 5 of 7 families fire |
| **Beneish M-score** | −3.10 (below threshold) | −0.96 (flagged) |
| **Verdict** | Trust the \$1.00; value normally | Discount to ~\$0.60 of real EPS |

Northwind passes every dimension. Its \$1.00 of EPS is cash-backed, repeatable, conservatively measured, and transparently disclosed; its M-score sits safely below the threshold. You can trust the number and value the business on it.

Riverstone fails almost every dimension. Its \$1.00 collects only thirty-five cents of cash, includes twenty cents of one-time items, sits on receivables that nearly doubled in collection time, is dressed up to \$1.40 with recurring add-backs, trips five of seven red-flag families, and scores −0.96 on Beneish. Its true, durable EPS is closer to **\$0.60** — and an investor paying twenty times the reported \$1.00 is really paying *thirty-three* times the honest figure.

*Two companies with identical headline EPS can carry completely different earnings quality, and the scorecard turns that difference into an action: trust Northwind's dollar, discount Riverstone's to sixty cents, and price each accordingly.*

The whole point of the GAAP-to-adjusted bridge is visible in that last comparison. The figure below shows how a company travels from a modest GAAP figure to an inflated adjusted headline, one defensible add-back at a time.

![A before and after bridge showing a company starting at sixty cents of GAAP EPS and reaching a one dollar forty adjusted headline by adding back stock compensation restructuring and amortization each individually defensible but together more than doubling the number](/imgs/blogs/quality-of-earnings-accruals-one-offs-red-flags-7.png)

## Common misconceptions

**"High earnings quality just means high earnings."** No — quality and quantity are independent. A company can earn a lot from low-quality sources (one-time gains, aggressive recognition, under-reserving) and a company can earn modestly from impeccably high-quality ones. Quality is about the *character* of the profit — how cash-backed, repeatable, conservative, and transparent it is — not its size. A small, cash-rich, conservatively-reported profit is higher quality than a large, accrual-laden, aggressively-adjusted one.

**"Non-GAAP earnings are fake; only GAAP matters."** Too strong in the other direction. Non-GAAP figures *can* be legitimate and useful — excluding a genuine one-time acquisition cost, or the non-cash amortization of an acquired intangible, can give a clearer picture of run-rate economics. The problem is not non-GAAP as a concept; it is *which adjustments* a company makes. Adding back genuine one-offs is fair; adding back recurring costs like stock-based compensation and serial restructuring is not. Read the reconciliation and judge each add-back on whether the cost recurs — do not reflexively accept or reject the adjusted number.

**"If the auditors signed off, the earnings are clean."** An unqualified audit opinion means the financials are free of *material misstatement* under the rules — it does not mean the company used conservative judgment or that the earnings are high quality. Auditors sign off on aggressive-but-legal accounting all the time; that is exactly what "within the rules" means. Aggressive useful lives, thin reserves, and early revenue recognition all pass audit. And in the rare cases of outright fraud, the auditors are sometimes deceived too. An audit is a floor, not a quality stamp.

**"A single red flag means avoid the stock."** No single flag convicts — every one of the families has innocent explanations. Rising DSO might be a deliberate move into a market with longer terms; a restructuring charge might be a genuine once-a-decade event; an auditor change might be a routine fee dispute. What matters is *confluence*: several flags from different families pointing the same way. Treating every isolated flag as disqualifying will keep you out of perfectly good businesses; ignoring clusters will get you into bad ones.

**"The Beneish M-score detects fraud."** It detects the *statistical fingerprint* of earnings manipulation — which correlates with fraud but is not the same thing. A high M-score means "this firm's ratios resemble those of known manipulators; investigate further," not "this firm is committing fraud." Fast-growing honest companies score high; patient frauds can score low. It is a screen that points you where to dig, valuable precisely because it is mechanical and unbiased, but it is one input, not a verdict.

**"Cash flow can't be manipulated, so just use it instead of earnings."** Cash flow is *harder* to manipulate than earnings, not impossible. Companies can flatter operating cash flow by stretching payables (delaying payments to suppliers), selling receivables (factoring), or misclassifying outflows between the operating, investing, and financing sections of the cash flow statement. The cash-flow test is powerful, but you still have to read the cash flow statement carefully and watch for these games — the topic of its [own companion piece](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from).

## How it shows up in real markets

**Enron (2001).** The defining case. Enron used off-balance-sheet special-purpose entities to hide debt and manufacture earnings, reporting profits that bore little relationship to cash, with heavy related-party dealings and disclosure so opaque that even sophisticated analysts could not penetrate it. Multiple red-flag families fired — earnings far ahead of cash, related-party deals, impenetrable disclosure — and the Beneish M-score famously flagged Enron as a likely manipulator from its public financials. It was, eventually, fraud rather than mere aggressiveness, but the quality signals were visible years before the collapse to anyone running the cash-flow test and reading the related-party footnotes. The full anatomy is in the [Enron case study](/blog/trading/finance/enron-2001-accounting-fraud).

**Wirecard (2020).** A more recent and more brazen case from the fraud end of the spectrum. The German payments company reported roughly €1.9 billion of cash that simply did not exist — fabricated through a web of related-party trustees and third-party acquirers in Asia. The earnings-quality signals were there for those willing to see them: cash that could not be independently confirmed, an outsized reliance on opaque third-party partners, growth that did not reconcile to the cash flows, and years of short-sellers and journalists pointing at exactly these problems while the company and its defenders dismissed them. The lesson is that *confirming cash actually exists* — not just that it is reported — is the deepest version of the cash-flow test, and the subject of forensic work. The story is told in the [Wirecard case study](/blog/trading/finance/wirecard-the-german-fintech-fraud).

**The serial-restructurer pattern (everywhere).** Far more common than outright fraud is the perfectly legal, perfectly auditable company that takes a "non-recurring" restructuring charge every single year and reports an "adjusted" EPS well above GAAP. Many large industrials and consumer companies have done versions of this for years. No fraud, no restatement, nothing illegal — just a steady, aggressive exclusion of recurring costs that makes the core look more profitable than it is. This is the everyday work of earnings-quality analysis: not catching crooks, but correctly normalizing the earnings of honest companies that present themselves in the best possible light.

**Stock-based compensation and the technology sector.** The most consequential earnings-quality issue in modern markets is the routine exclusion of stock-based compensation from "adjusted" earnings, especially among high-growth technology companies. Firms that are barely profitable or loss-making on a GAAP basis can show robust "adjusted" profits almost entirely by adding back SBC. The dilution is real — existing shareholders own less of the company each year — but it is invisible in the adjusted metric the company leads with. Investors who took the adjusted numbers at face value through the 2010s and into the 2020s systematically overpaid; those who insisted on GAAP earnings (or at least counted SBC as the real cost it is) had a much truer picture. The discipline of refusing to add back recurring SBC has been, quietly, one of the most valuable earnings-quality habits of the last decade.

## When this matters and further reading

Quality of earnings is not an exotic forensic specialty reserved for catching frauds. It is the foundational check you run on *every* company before you trust its reported number — because every valuation, every multiple, every growth forecast is built on top of an earnings figure, and if that figure is half smoke, everything downstream is wrong. The four dimensions (cash-backed, repeatable, conservative, transparent), the cash-flow test, the red-flag families, and the Beneish M-score together give you a repeatable scorecard that turns a vague unease about "managed numbers" into a concrete adjustment: trust this dollar, discount that one to sixty cents.

It matters most exactly when the temptation to skip it is highest — a fast-growing company with a beautiful adjusted-earnings story, a stock the market loves, a management team that radiates confidence. Those are precisely the situations where aggressive accounting flourishes and where the cash-flow test, run patiently over five years, earns its keep. The companies that pass every dimension are the ones you can own with conviction; the ones that fail are the ones the discipline is built to protect you from.

To go deeper:

- **[Accrual vs cash accounting: why earnings are an opinion](/blog/trading/equity-research/accruals-vs-cash-why-earnings-are-an-opinion)** — the mechanics behind the cash-flow test: the four accrual buckets, the accrual ratio, and Sloan's accrual anomaly.
- **[The cash flow statement: where the cash really comes from](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from)** — how to read the statement at the heart of the master test, and the games played with operating cash flow.
- **[Forensic accounting: spotting manipulation and fraud](/blog/trading/equity-research/forensic-accounting-spotting-manipulation-and-fraud)** — the deep end: confirming cash exists, tracing related-party webs, and the techniques for the fraud tail of the distribution.
- **[Enron, 2001: the accounting fraud](/blog/trading/finance/enron-2001-accounting-fraud)** and **[Wirecard: the German fintech fraud](/blog/trading/finance/wirecard-the-german-fintech-fraud)** — the two canonical cases where earnings quality, ignored, became catastrophe.

Master this one skill and you will have the single most durable edge in fundamental investing: the ability to look at two companies reporting the same EPS and know, before anyone else does, which dollar is real.
