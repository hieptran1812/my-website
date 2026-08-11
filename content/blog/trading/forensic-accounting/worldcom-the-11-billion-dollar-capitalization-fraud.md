---
title: "WorldCom: the \\$11 billion capitalization fraud, quarter by quarter"
date: "2026-08-11"
publishDate: "2026-08-11"
description: "How WorldCom held one ratio at 42% for a year by moving network operating costs onto its balance sheet, why the fraud had to grow every quarter, why 'just look at cash flow' would have missed it, and how an internal audit team found it against an account it could not reconcile."
tags: ["forensic-accounting", "worldcom", "capitalizing-costs", "line-costs", "cash-flow-analysis", "free-cash-flow", "internal-audit", "financial-statement-fraud", "sarbanes-oxley", "accrual-releases", "earnings-quality", "case-study"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 48
---

> [!important]
> **TL;DR** — WorldCom's fraud was one journal entry, repeated for five quarters, on costs the company genuinely paid.
>
> - WorldCom bought network capacity from other carriers. Those payments — **line costs** — were its largest expense, roughly half of all expenses from 1999 to 2001. They are consumed in the period, so they belong on the income statement.
> - Management held the ratio of line costs to revenue at **about 42%** through 2001 by moving the excess into capital accounts labelled "prepaid capacity". Without those entries the ratio would have been **typically above 50%** — the Special Investigative Committee's own words.
> - The mechanics: **debit an asset, credit the expense.** In the first quarter of 2001 that was \$771 million of line costs removed from expense, \$629 million parked in Other Long Term Assets and \$142 million in Construction in Progress, booked on Friday 20 April 2001 off a one-page schedule with no supporting documents.
> - It had to grow. Every fake asset starts depreciating, so each quarter needed a bigger entry: **\$544m, \$560m, \$743m, \$841m, \$818m** across the five quarters from Q1 2001 to Q1 2002. In three of those five quarters, the reported profit was actually a pre-tax loss.
> - **The cash flow statement would have caught it, but not the way most people are taught.** Capitalizing an operating cost lifts operating cash flow *and* net income together, so "operating cash flow comfortably exceeds profit" — the standard quality-of-earnings test — passed with flying colours. Free cash flow did not move by a dollar. WorldCom's reported 2001 free cash flow was **+\$108 million on \$35.2 billion of revenue.**
> - The number to keep straight: **approximately \$3.8 billion** was what WorldCom announced on 25 June 2002. The larger totals came later, as the investigation widened. They are different figures and get conflated constantly.

The most expensive accounting fraud of its era did not involve a single fake customer.

Nobody at WorldCom forged a bank statement, invented a subsidiary in the Cayman Islands, or booked revenue from a company that did not exist. The carriers were real. The invoices were real. The cash genuinely left the building and genuinely arrived at AT&T, at Sprint, at the regional Bell companies, in exchange for the right to carry a customer's phone call across somebody else's wire.

What was false was a single word in the ledger. Instead of **expense**, the entry said **asset**.

That one substitution, repeated for five quarters, turned three quarterly losses into three quarterly profits, kept a ratio the entire market was watching pinned to a number that had stopped being true, and postponed the collapse of a company that filed for the largest bankruptcy in United States history at the time.

![One dollar of line cost and its two possible destinations: expensed immediately against operating income, or capitalized onto the balance sheet as transmission equipment and depreciated over years](/imgs/blogs/worldcom-the-11-billion-dollar-capitalization-fraud-1.webp)

The diagram above is the mental model for everything that follows. A dollar paid to another carrier can travel one of two roads. Down the green road it lands on the income statement this quarter and reduces profit by a dollar. Down the red road it lands on the balance sheet as equipment, and reaches the income statement only in slivers, as depreciation, over years. The cash leaves the building either way. Only the label changes.

The general technique — when capitalizing a cost is legitimate, where the software and research boundary genuinely is grey, and how the profit boost works arithmetically — is the subject of a companion article, [capitalizing costs to inflate profit: the WorldCom move](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move). This one goes the other way: deep into the single case, quarter by quarter, entry by entry, with the pressure that produced it and the audit that ended it.

A note on the sourcing before we start, because this case is unusually badly reported.

## Foundations: the building blocks of a line-cost fraud

If you have never read a telecom income statement, nothing below requires that you have. Here is everything the case needs, from zero.

### What a long-distance carrier actually sold

WorldCom began life in 1983 as Long Distance Discount Services, a reseller. When you dialled a long-distance number, your call left your local phone company's wires, travelled across a long-distance carrier's network, and arrived at the recipient's local phone company. WorldCom's business was that middle leg.

Crucially, for much of its history WorldCom did not own all the middle leg. It **bought capacity wholesale** from the companies that did — including the very competitors it was undercutting — and resold it. In the Special Investigative Committee's description, the early company "made these connections by reselling long distance capacity that it purchased from the major long distance carriers on a wholesale basis."

### What line costs are

The fees WorldCom paid other carriers for that capacity are **line costs**. In its own later filings the company renamed them *access costs*, which is a clearer name: they are the cost of access to somebody else's network.

Two components dominate: **access charges**, paid to local phone companies for originating or terminating a call on their wires, and **transport charges**, paid for carrying traffic between points. WorldCom's own 2001 Form 10-K says so directly: "The principal components of line costs are access charges and transport charges."

The economically important feature is that a line cost buys **this period's traffic**. You pay for the capacity you use in the quarter you use it. Next quarter you pay again. There is no lasting resource left over — you did not acquire a wire, you rented time on one.

That is what makes line costs an unambiguous **operating expense**: a cost consumed in producing this period's revenue, which therefore belongs on this period's income statement. Contrast it with digging a trench and laying your own fibre. That produces something you control for twenty years, and it is a genuine **capital expenditure** — an asset, charged to profit gradually as *depreciation* (the accounting spreading of an asset's cost across the years it is useful).

Line costs were not a rounding error. They were WorldCom's single largest expense: from 1999 to 2001, the investigators found, "line costs accounted for approximately half of the Company's total expenses."

### The E/R ratio, and why one number mattered so much

Because line costs were half of all expenses, both management and outside analysts watched them through a single ratio:

```
line cost E/R ratio  =  line cost expense  ÷  revenue
```

"E/R" is simply expense-to-revenue. Read it as: *for every dollar of revenue we bring in, how many cents go straight back out to other carriers?*

The ratio is a genuinely good measure of a reseller's health, which is why the market used it. If it rises, one of three things is happening: you are paying more per unit of traffic, you are selling at lower prices, or your traffic mix has shifted towards routes you do not own. All three squeeze the margin. A carrier whose E/R ratio is climbing is a carrier whose economics are deteriorating, no matter what its revenue line says.

Here is WorldCom's reported ratio, taken from the common-size table in its own FY2001 Form 10-K, filed 13 March 2002:

| Year | Revenue | Line costs | Line cost E/R ratio, as reported |
| --- | --- | --- | --- |
| 1999 | \$35,908m | — | 41.0% |
| 2000 | \$39,090m | \$15,462m | 39.6% |
| 2001 | \$35,179m | \$14,739m | 41.9% |

Hold on to two things in that table. The ratio looks stable — a touch under 42% for three straight years, with a dip in the middle. And revenue **fell 10%** in 2001, from \$39.1 billion to \$35.2 billion.

### Debits, credits, and the only accounting rule this article needs

Every accounting entry has two halves that must be equal. Increasing an expense or an asset is a **debit**; increasing a liability, equity, or revenue is a **credit**. The books balance because the two halves always match.

The consequence that matters here: because the halves always match, **an entry can be completely fraudulent and the books will still balance perfectly**. Debiting an asset and crediting an expense by the same amount is arithmetically flawless. Nothing about the balance sheet's equality is disturbed. This is why "the balance sheet balances" is not evidence of anything, and why a control that only checks for balance finds nothing.

### The four numbers a reclassification moves

Move a dollar from expense to asset and four things happen at once:

1. **Operating income rises** by a dollar, because an expense vanished.
2. **Total assets rise** by a dollar, and equity rises with the extra profit.
3. **Future depreciation rises**, because the new asset must be written off over its life.
4. **Cash does not change at all.** The payment already happened.

Point four is the one people skip, and it is the hinge of the whole case. We will come back to it twice.

## The business pressure: a roll-up that ran out of things to buy

Frauds have mechanics and they have motives. WorldCom's motive was structural, and it arrived on a specific date.

### The acquisition machine

WorldCom grew by buying companies and paying in its own stock. The Special Investigative Committee's background section traces the ladder: LDDS acquired or merged with at least seven companies between 1991 and 1993, then merged with Metromedia Communications and Resurgens Communications Group in September 1993 in a transaction worth \$1.3 billion. By the end of 1993 it was the fourth-largest long-distance carrier in the country, on annual revenues of roughly \$1.5 billion.

Then the deals got serious: IDB Communications in late 1994 for \$936 million, WilTel in early 1995 for \$2.5 billion, and MFS Communications in late 1996 for \$12.4 billion. LDDS formally became WorldCom after a shareholder vote on 25 May 1995. The MCI acquisition followed, and the effect on the balance sheet is visible in the company's own selected financial data: **total assets went from \$24.4 billion at the end of 1997 to \$87.1 billion at the end of 1998.**

This is a **roll-up** — a company whose growth comes from acquiring other companies rather than from selling more to existing customers. Roll-ups have a specific vulnerability. A high stock price is the currency for the next deal, and the next deal is what justifies the high stock price. The machine runs on its own output.

### The date the machine stopped

On 5 October 1999, WorldCom and Sprint announced they had agreed to merge in a deal valued at **\$115 billion**. It would have handed WorldCom the wireless business analysts kept noting it lacked.

The Antitrust Division of the Department of Justice refused to approve it on terms the two companies would accept, and they terminated discussions on **13 July 2000**.

The investigators are unusually blunt about what that meant inside the company:

> The termination of this merger was a significant event in WorldCom's history. Within WorldCom, it was perceived to mean that large-scale mergers were no longer a viable means of expanding the business. A number of witnesses told us that, after this point, Ebbers appeared to lack a strategic sense of direction, and the Company began drifting.

Simultaneously, the telecom bubble was deflating. The regional Bell companies were pushing into long distance, long-distance carriers were pushing into local calls, and everyone was chasing the same internet data revenue. WorldCom's reported growth rates "declined by a percentage point each quarter" through 2000 even as its earnings releases kept highlighting double-digit growth.

So: an acquisition engine with nothing left to acquire, a core product under severe price pressure, and a management team that had promised Wall Street double-digit growth. The SEC's own characterisation of the motive, in the litigation release announcing the case, is that the entries "were intended to mislead investors and manipulate WorldCom's earnings to keep them in line with estimates by Wall Street analysts."

### There was also a personal reason

Two details from the investigators' section on "Compromising Financial Arrangements" explain why senior people had more than career incentives.

Ebbers had borrowed from WorldCom itself — the report notes he "owed WorldCom tens of millions of dollars, which had been lent to him based on his assertion that he did not have sufficient cash to meet his financial obligations without selling WorldCom stock." A chief executive whose personal solvency depends on the share price not falling is not a neutral party to an accounting question.

And in late 2000, CFO Scott Sullivan gave personal gifts totalling at least \$140,000 to seven managers — \$20,000 each, written as two personal cheques of \$10,000, one to the officer and one to the officer's spouse. The recipients included the Controller, the Director of General Accounting, the VP of Financial Reporting, and the VP of Investor Relations. Sullivan told them he had received a \$10 million bonus, felt it was partly due to their work, and wanted them to view the money as coming from the company. The investigators found no company rule prohibiting this. They also noted what it does to the willingness of a subordinate to object.

### The ratio that had to hold

Put the pressure and the metric together and you get the fraud's design specification. The investigators state the objective in one sentence: to "hold reported line costs to approximately 42% of revenues (when in fact they typically reached levels in excess of 50%), and to continue reporting double-digit revenue growth when actual growth rates were generally substantially lower."

![WorldCom's reported line cost E/R ratio held flat at 42% through the four quarters of 2001 against a shaded band showing the actual ratio typically exceeding 50%, with the quarterly amounts of line costs moved into capital accounts shown beneath](/imgs/blogs/worldcom-the-11-billion-dollar-capitalization-fraud-2.webp)

The green line is what the market was shown. The red band is what was underneath it. The gap between them is the fraud, and the columns along the bottom are its quarterly cost.

#### Worked example: what the E/R ratio was really doing in 2001

Take only figures from primary documents and do the division yourself.

WorldCom's FY2001 Form 10-K reports revenue of **\$35,179 million** and line costs of **\$14,739 million**.

```
Reported line cost E/R ratio  =  14,739 ÷ 35,179  =  41.9%
```

That matches the 41.9% printed in the 10-K's own common-size table, so we have read the filing correctly.

Now substitute the actual line costs. The SEC's complaint puts WorldCom's true 2001 line costs at approximately **\$17,754 million**:

```
Actual line cost E/R ratio    =  17,754 ÷ 35,179  =  50.5%
```

And check the income statement consequence:

```
Line cost difference                       17,754 − 14,739  =  \$3,015m
Reported income before tax and minority interests   =   \$2,393m
Less the line cost correction                       =  (\$3,015m)
Reconstructed result                                =   (\$622m)  loss
```

A reported pre-tax profit of \$2.4 billion becomes a pre-tax loss of \$622 million. The company did not report a bad year — it reported the wrong sign.

Notice how well the two independent sources agree. An enforcement complaint says the true ratio would have exceeded 50%; an internal investigation says "typically exceeding 50%"; and dividing two numbers out of the company's own annual report gives 50.5%. That triangulation is what a sourced claim looks like.

**Intuition:** the ratio was not a summary of the business, it was the target the entries were sized to hit — which means the ratio's stability was the fraud's signature, not evidence of a well-run company.

## The companion trick that came first: releasing the reserves

The capitalization scheme is the famous part. It was not the first part, and understanding what preceded it explains why the famous part started when it did.

### Accruals, in plain English

An **accrual** is money set aside on the financial statements for a bill you expect but have not yet received or paid. If you know your carriers will invoice you roughly \$100 million for December traffic and the invoices arrive in February, you record a \$100 million expense in December and a \$100 million liability, so the cost sits in the period that caused it.

**Releasing** an accrual means reversing part of it because it turned out you needed less than you set aside. That is entirely proper when it happens honestly — and here is the mechanically important bit: releasing an accrual **reduces reported expense in the period of the release**. The money set aside comes back as an offset. Expenses fall and pre-tax income rises.

Which makes an over-stuffed accrual a store of future profit. This is the "cookie jar" mechanism, covered in general terms in [cookie-jar reserves and big bath accounting](/blog/trading/forensic-accounting/cookie-jar-reserves-and-big-bath-accounting).

### What WorldCom did with them

The investigators' finding is precise:

> In 1999 and 2000, WorldCom reduced its reported line costs by approximately **\$3.3 billion**. This was accomplished by improperly releasing "accruals" ...

And they identify three distinct manipulations, which are worth naming separately because each is a different red flag:

1. **Releases with no analysis at all.** "In some cases accruals were released without any apparent analysis of whether the Company actually had an excess accrual in the account." Reported line costs fell with no basis whatsoever.
2. **Rainy-day timing.** Even where a genuine excess existed, it was often not released in the period it was identified. Instead, "certain line cost accruals were kept as rainy day funds and released to improve reported results when managers felt this was needed."
3. **Re-aiming accruals set up for something else.** Accruals established for other purposes were released against line costs, which flatters the specific ratio the market was watching rather than the expense line the accrual actually related to.

The common features are the forensic tells. The releases were directed by senior finance — the CFO, the Controller, and the Director of General Accounting. They "did not occur in the normal course of day-to-day operations, but instead in the weeks following the end of the quarter in question." And "the timing and amounts of the releases were not supported by contemporaneous analysis or documentation."

An entry made after the quarter closes, in an amount senior management specifies, with no analysis behind it, is not accounting. It is arithmetic applied to a target.

#### Worked example: how a released accrual becomes profit

Illustrative figures, to see the mechanism cleanly.

You are a carrier with a \$500 million liability on your balance sheet called "accrued line costs". Reported line costs this quarter are \$4,200 million, revenue is \$10,000 million, so your E/R ratio is 42.0%. Next quarter, traffic costs rise and honest line costs come in at \$4,600 million on the same \$10,000 million of revenue:

```
Honest quarter
  Revenue                     10,000
  Line costs                   4,600
  E/R ratio                     46.0%
```

Your CFO does not want 46%. So you release \$400 million of the accrual against line costs:

```journal
Dr  Accrued line costs (liability)        400
    Cr  Line cost expense                       400
```

Now:

```
After the release
  Revenue                     10,000
  Line costs                   4,200
  E/R ratio                     42.0%
  Pre-tax income          +      400  versus honest
  Accrual remaining              100
```

The ratio is back to 42%, pre-tax income is \$400 million higher, and the balance sheet still balances — the liability shrank and equity grew by the same amount.

But look at the last line. You started with \$500 million and used \$400 million. **You can do this exactly once more, and only for a quarter of the amount.**

**Intuition:** an accrual release is a fixed-size battery, not a generator. It works beautifully until it is flat, and the date it goes flat is knowable in advance.

### The moment the reserves ran out

That is precisely what happened. The investigators:

> By the end of 2000, WorldCom had essentially exhausted available accruals, at least on the scale needed to continue this manipulation of reported line costs.

And then, in the same paragraph:

> Thereafter, from the first quarter of 2001 through the first quarter of 2002, WorldCom improperly reduced its reported line costs by **\$3.8 billion, principally by capitalizing \$3.5 billion of line costs** — at Sullivan's direction — in violation of WorldCom's capitalization policy and well-established accounting standards.

The word "thereafter" is doing a great deal of work. The capitalization scheme did not begin because someone had a new idea about network economics. It began in the first quarter after the previous method stopped working. One fraud was the successor of another, and the handover is dated.

![A two-phase timeline showing accrual releases of approximately \$3.3 billion across 1999 and 2000 giving way to line-cost capitalization of approximately \$3.8 billion from Q1 2001 through Q1 2002, with the quarterly capitalized amounts and the parallel revenue scheme](/imgs/blogs/worldcom-the-11-billion-dollar-capitalization-fraud-6.webp)

The thin lane at the bottom of that figure is worth a moment, because the line-cost story is only half the picture. On the revenue side, the investigators identified **\$958 million** of revenue improperly recorded between Q1 1999 and Q1 2002, with their accounting advisors flagging a further **\$1.107 billion** as questionable. Most of it went into management-controlled accounts called "Corporate Unallocated", booked after the quarter had closed, in large round-dollar amounts, on a schedule whose distribution was tightly restricted. There was a named internal process for it — "Close the Gap".

Without those revenue entries, the investigators concluded, WorldCom "would have failed, in six out of the twelve quarters between the beginning of 1999 and the end of 2001, to achieve the double-digit growth it reported."

One primary document from that side of the house deserves quoting, because it is the clearest statement of intent in the entire record. On **19 June 2001** — a year before anything became public — Sullivan left Ebbers a voicemail:

> Hey Bernie, it's Scott. This MonRev just keeps getting worse and worse. The copy, um the latest copy that you and I have already has accounting fluff in it . . . all one time stuff or junk that's already in the numbers. ... We are going to dig ourselves into a huge hole because year to date it's disguising what is going on on the recurring, uh, service side of the business . . . .

The chief financial officer describing his own reported revenue as "accounting fluff" and "junk", and predicting the hole, twelve months early.

## The trick itself: an operating expense becomes transmission equipment

Now the entry. Because the investigators reconstructed it from the ledger, we can look at the real one rather than a stylised version.

### The first one, on a Friday in April 2001

On **17 April 2001**, operating managers concluded they could not cut line costs enough to bring the first-quarter E/R ratio down to 2000 levels. Three days later, on Friday **20 April 2001**, Troy Normand, a Director in General Accounting, told a manager in his department to book an entry reducing line costs by **\$771 million** and increasing two asset accounts by the same total: **\$629 million into Other Long Term Assets and \$142 million into Construction in Progress.**

The instruction came as a one-page document titled "March 2001 Adjustments", listing amounts and account numbers. The investigators record what else came with it: "he did not provide any other support or explanation for the entry."

The manager who booked it named the transferred amounts using the term from that one-page document: **"Prepaid Capacity Costs"**. She had no concerns at the time — and the report explains why, in a sentence that says more about WorldCom's controls than any control-deficiency finding could: "Walter often booked entries in amounts between \$500 million and \$1 billion without any detailed support."

The \$629 million in Other Long Term Assets was temporary. Asked about it, Normand said they were just "parking" the line costs there until they decided what to do with them. Over 23 and 24 April, \$402 million of it was moved into Construction in Progress, bringing the total in that account to \$544 million — which is the figure that appears as the first quarter's capitalization.

Note the anatomy. The amount was determined by the ratio it needed to produce, not by any asset. It was booked after the quarter closed but before the earnings release on 26 April. It moved between accounts as people worked out where it would attract least attention. And the supporting documentation was a page of numbers.

![The honest journal entry debiting line cost expense against the fraudulent entry debiting transmission equipment and crediting line cost expense, with the net effect on the trial balance showing expenses down, assets up and equity up](/imgs/blogs/worldcom-the-11-billion-dollar-capitalization-fraud-3.webp)

The right-hand panel of that figure is the point of the whole diagram: **the books still balance.** Expenses fall by the amount, assets rise by the amount, equity rises by the amount. Nothing is out of place. There is no arithmetic error to find, which is exactly why finding it required someone to ask what the asset actually was.

#### Worked example: one quarter's line costs, two sets of books

Illustrative figures, round for clarity, \$ in millions.

A carrier has revenue of \$8,000 and genuinely incurs \$3,600 of line costs. Other operating expenses are \$3,900. Its target is a 42.0% E/R ratio.

**Honest books:**

```
Revenue                          8,000
Line costs                      (3,600)      E/R = 45.0%
Other operating expenses        (3,900)
Operating income                   500
```

**With \$240 of line costs capitalized** (enough to bring 3,600 down to 3,360, which is 42.0% of 8,000):

```journal
Dr  Transmission equipment              240
    Cr  Line cost expense                     240

Revenue                          8,000
Line costs                      (3,360)      E/R = 42.0%
Other operating expenses        (3,900)
Operating income                   740
```

Operating income is 48% higher. The E/R ratio hits the target exactly. And the amount — 240 — was not derived from any equipment. It was derived by solving for the ratio:

```
required line costs  =  0.42 × 8,000  =  3,360
entry needed         =  3,600 − 3,360  =  240
```

That backwards derivation is the fraud's fingerprint, and it is why the investigators found, of one quarter's amounts, that the sums "had been backed into in order to achieve a particular level of reported line costs."

**Intuition:** when the entry is computed from the answer you want rather than from the thing you bought, no amount of documentation can make it an asset.

### What it did to the reported numbers

![A table of ten reported numbers comparing an expensed and a capitalized treatment of the same \$1,000m, with eight metrics improving, next year's depreciation rising, and free cash flow unchanged in a boxed green row](/imgs/blogs/worldcom-the-11-billion-dollar-capitalization-fraud-4.webp)

#### Worked example: what \$1,000m of reclassification does to ten reported numbers

Illustrative figures throughout, \$ in millions except per-share amounts. A tax rate of 35% and a four-year asset life are assumed for the arithmetic.

Move \$1,000 from line cost expense to transmission equipment and trace every headline number:

```
                                 Expensed    Capitalized    Change
Operating income                      500          1,500    +1,000
EBITDA                              1,200          2,200    +1,000
Pre-tax income                        300          1,300    +1,000
Net income (35% tax)                  195            845      +650
EPS (2,900m shares)                 $0.07          $0.29    +$0.22
Total assets                      100,000        101,000    +1,000
Depreciation next year              5,000          5,250      +250
Cash from operations (CFO)            800          1,800    +1,000
Cash used in investing (CFI)       (2,000)        (3,000)   −1,000
------------------------------------------------------------------
Free cash flow (CFO − capex)       (1,200)        (1,200)     ZERO
```

Work through the individually interesting rows.

**EBITDA rises by the full amount, permanently.** EBITDA is earnings before interest, taxes, depreciation and amortization. Because it adds depreciation back, the later unwinding of a capitalized cost never reaches it. An expensed dollar reduces EBITDA forever; a capitalized dollar never touches it at all. Any company judged primarily on EBITDA has a standing incentive here, which is one reason the metric deserves the scepticism given it in [non-GAAP and adjusted EBITDA: the metrics companies invent](/blog/trading/forensic-accounting/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent).

**EPS more than quadruples,** from \$0.07 to \$0.29, because the base is small. Thin margins are what make a reclassification decisive: the same \$1,000 against a \$5,000 profit would be a 20% improvement, not a 300% one. WorldCom's 2001 reported operating margin was 10.0% of revenue, down from 20.9% in 2000. Margins had thinned to exactly the point where the entry could flip the sign.

**Next year's depreciation rises by \$250** — a quarter of the asset over a four-year life. This is the bill arriving, and it is the reason the scheme cannot stand still.

**CFO rises by the full \$1,000 and CFI falls by the full \$1,000.** The cash never moved. Only the section it was reported in moved.

**And free cash flow does not change at all.** Not approximately. Exactly zero, arithmetically, always. Hold that thought for two sections.

**Intuition:** the reclassification improves every number that is computed above the capital-expenditure line and no number that is computed below it.

## Why the fraud had to grow every quarter

A one-off reclassification is a one-off lie. A recurring one is a treadmill, and the speed increases on its own.

![A stacked column chart across five quarters showing the entry required each quarter rising from 800 to 1,000 as an amber depreciation-drag segment accumulates on top of a flat red gap-to-fill segment](/imgs/blogs/worldcom-the-11-billion-dollar-capitalization-fraud-5.webp)

Three forces push the required entry up each quarter.

**First, the depreciation drag.** Every fake asset you create begins depreciating immediately. That depreciation is a real charge against reported operating income, so next quarter you must cover both the new gap *and* the depreciation on all the fake assets you have already made. The drag compounds.

**Second, the underlying business keeps deteriorating.** The gap you are filling is not static. WorldCom's revenue fell through 2001 while purchased capacity did not fall in step, so the honest ratio kept rising and the entry needed to suppress it kept growing.

**Third, the comparison is against your own previous lie.** Having reported 42% last year, this year's 42% is the floor, not the ceiling. You cannot let the ratio drift up without inviting exactly the questions the entries exist to prevent.

The actual quarterly amounts capitalized show the escalation:

| Quarter | Amount capitalized | Reported pre-tax income | Pre-tax result without the entry |
| --- | --- | --- | --- |
| Q1 2001 | \$544m | — | — |
| Q2 2001 | \$560m | \$159m | (\$401m) loss |
| Q3 2001 | \$743m | — | — |
| Q4 2001 | \$841m | \$401m | (\$440m) loss |
| Q1 2002 | \$818m | \$240m | (\$578m) loss |

Those five amounts sum to \$3,506 million — the "principally by capitalizing \$3.5 billion" in the investigators' summary.

And the third column against the fourth is the whole case in one comparison. The investigators state it plainly: "Had WorldCom not capitalized these expenses, it would have reported a pre-tax loss in three of the five quarters in which the improper capitalization entries occurred."

#### Worked example: the depreciation drag, quarter by quarter

Illustrative figures, \$ in millions, to see why the entry must grow even if the business does not get worse.

Assume the honest gap between actual line costs and the target is a constant \$800 every quarter, and each capitalized amount depreciates over five years — that is 5% per quarter, near enough for arithmetic.

Quarter 1: the gap is 800. There are no prior fake assets, so the entry is **800**. Fake asset balance: 800.

Quarter 2: the gap is still 800, but the 800 of fake assets now generates depreciation of 5% × 800 = 40, which is a charge against operating income. So you must cover 800 + 40 = **840**. Fake asset balance: 1,640.

Quarter 3: depreciation on 1,640 is 82. Entry needed: 800 + 82 = **882**. Balance: 2,522.

Quarter 4: depreciation on 2,522 is 126. Entry needed: 800 + 126 = **926**. Balance: 3,448.

Quarter 8, extrapolating: the accumulated fake assets are past \$7 billion and their quarterly depreciation alone exceeds \$350 — nearly half the original gap — before you have addressed the current quarter at all.

```
Quarter   Gap    Depreciation drag    Entry required
   1      800            0                  800
   2      800           40                  840
   3      800           82                  882
   4      800          126                  926
```

**Intuition:** capitalizing an expense does not remove the cost, it schedules it — and the schedule of everything you have already hidden is a bill that arrives every quarter, forever, growing.

### There was an exit plan, and it tells you they knew

Sullivan had a plan for the accumulated asset balances. The investigators: "Sullivan made comments indicating that he intended ultimately to reduce these inflated asset accounts by including them in a large restructuring charge later in 2002."

That is a **big bath** — dumping accumulated garbage into one enormous charge that the market forgives as a clean-up. It would have made the fake assets disappear into a line item nobody models.

It also settles the question of intent. You do not plan to write off assets you believe are real. When the plan came up at the Audit Committee, KPMG's engagement partner said writing off the prepaid capacity amounts in a restructuring would be inappropriate.

### And they hid it from the auditors, specifically

One detail from the third quarter of 2001 is worth isolating, because it removes any reading of this as an aggressive-but-arguable accounting position.

The capitalized line costs from the first two quarters of 2001 had been booked in Construction in Progress. Before the end of the third quarter, the investigators found, "Property Accounting also transferred the previously-capitalized amounts out of Construction in Progress just before the Company's auditors planned to do test work in that area."

Assets were moved out of an account because the auditors were about to look in it. That is not a judgement call about the matching principle.

## Why the cash flow statement would have exposed it — and why "just look at cash flow" would not

This section is the one that changes how you read a financial statement, so it is worth going slowly.

The standard advice given to investors is: profit is an opinion, cash is a fact, so check that operating cash flow tracks net income. It is good advice against most frauds. Against this one it fails completely — and understanding *why* it fails is more valuable than any single red flag.

### What capitalizing does to the cash flow statement

The cash flow statement has three sections: **operating** (CFO — cash from running the business), **investing** (CFI — cash spent on or received from long-lived assets), and **financing** (borrowing, repaying, issuing shares, dividends).

When a cost is expensed, it reduces net income, and net income is where the operating section starts. The cash payment is an operating outflow.

When the same cost is capitalized, two things happen. Net income is higher, which raises CFO. And the payment is reported as capital expenditure, which is an *investing* outflow. So the outflow does not disappear — **it relocates from the operating section to the investing section.**

The consequence is precise, and it is the opposite of what most people expect: **capitalizing an operating cost raises reported operating cash flow.** Net income and CFO rise together, by the same amount. The comfortable test — "CFO comfortably exceeds net income, so the earnings are cash-backed" — comes out looking better than before the fraud.

![Two illustrative cash flow statements side by side, showing net income and deferred tax lifting operating cash flow from 800 to 1,800 while capital expenditure rises from 2,000 to 3,000, with an amber arrow marking \$1,000m of outflow relocated rather than removed, and free cash flow unchanged at negative 1,200](/imgs/blogs/worldcom-the-11-billion-dollar-capitalization-fraud-7.webp)

#### Worked example: the illustrative cash-flow bridge

Illustrative figures, \$ in millions. The deferred tax amount is invented to make both columns tie arithmetically; the structure is the real point.

**Expensed (honest):**

```
Net income                          195
+ Depreciation                    5,000
+ Deferred tax                        0
− Working capital change         (4,395)
= Cash from operations              800
Capital expenditure              (2,000)
= Free cash flow                 (1,200)
```

**Capitalized:**

```
Net income                          845
+ Depreciation                    5,000
+ Deferred tax                      350
− Working capital change         (4,395)
= Cash from operations            1,800
Capital expenditure              (3,000)
= Free cash flow                 (1,200)
```

Check both columns: 195 + 5,000 + 0 − 4,395 = 800, and 845 + 5,000 + 350 − 4,395 = 1,800.

Why does CFO rise by the full \$1,000 when net income only rose by \$650? Because the \$350 of tax that the extra profit would attract is *deferred*, not paid — the deduction was taken for tax purposes even though the expense was capitalized for reporting. Add the \$650 of extra net income to the \$350 of deferred tax and you get the full \$1,000. The whole pre-tax reclassification lands in operating cash flow.

There is no line item anywhere on the capitalized statement saying "expense moved out of operations". The relocation is invisible in the statement's own structure. That is what the amber arrow in the figure marks: \$1,000m of outflow relocated, not removed.

Now the row that matters:

```
Free cash flow  =  CFO − capital expenditure
Honest:            800 − 2,000  =  (1,200)
Capitalized:     1,800 − 3,000  =  (1,200)
```

Identical. And it is identical *by construction*: the fraud adds the same number to CFO and subtracts it from capex, so their difference cannot change. Free cash flow is arithmetically immune.

**Intuition:** operating cash flow can be improved by moving an outflow one section down the page; free cash flow cannot, because it spans both sections.

#### Worked example: WorldCom's real 2001 cash flow statement, rebuilt

Now with primary-source figures rather than illustrative ones. All from WorldCom's FY2001 Form 10-K.

Reported:

```
Net cash provided by operating activities        \$7,994m
Capital expenditures                            \$7,886m
Free cash flow                                  \$  108m
Net income applicable to common shareholders    \$1,384m
```

Start with the standard quality-of-earnings test. Operating cash flow of \$7,994 million against net income of \$1,384 million is a ratio of **5.8 times**. On the conventional reading, that is a company converting profit into cash exceptionally well. The test does not merely fail to fire — it actively reassures.

Now the free cash flow line. **\$108 million, on \$35.2 billion of revenue.** After paying for its capital programme, one of the largest telecoms in the world generated about three-hundredths of one percent of its revenue in surplus cash. That is not a company with \$1.4 billion of genuine earnings.

Set it in a three-year trend, all from the same filing:

| Year | CFO | Capital expenditure | Free cash flow |
| --- | --- | --- | --- |
| 1999 | \$11,005m | \$8,716m | **+\$2,289m** |
| 2000 | \$7,666m | \$11,484m | **(\$3,818m)** |
| 2001 | \$7,994m | \$7,886m | **+\$108m** |

Free cash flow went from comfortably positive, to \$3.8 billion negative, to barely breakeven — while reported net income applicable to common shareholders was \$3,941m, \$4,088m and \$1,384m. The two series tell completely different stories about the same three years, and only one of them was true.

Finally, strip the fraud out. The four 2001 capitalization entries total \$544 + \$560 + \$743 + \$841 = **\$2,688 million**. Move that amount back from investing to operating:

```
                                Reported     Corrected
Cash from operations              7,994         5,306
Capital expenditure             (7,886)       (5,198)
Free cash flow                     108           108
```

Operating cash flow falls by a third. Capital expenditure falls by a third. **Free cash flow does not move by one dollar.**

**Intuition:** the fraud was hiding in the difference between two numbers that both looked fine, and the one number it could not touch was already telling the truth in the published accounts.

### Which detection tests actually fire

![A seven-row matrix of detection tests showing which fire on a capitalization fraud and which do not, with net margin and CFO-exceeds-net-income failing to fire and free cash flow, the accruals ratio, PP&E per unit of traffic, implied asset life and the peer ratio comparison all firing](/imgs/blogs/worldcom-the-11-billion-dollar-capitalization-fraud-8.webp)

Walk the ones that work.

**Free cash flow.** Covered above. Arithmetically immune, and the reason [reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) puts free cash flow rather than CFO at the centre of the analysis.

**The accruals ratio, measured on net operating assets.** The accruals ratio compares reported earnings to the cash actually generated, and the version that works here is the balance-sheet one: the change in net operating assets over average net operating assets. Capitalizing an operating cost inflates net operating assets by the full amount, so the balance-sheet accruals ratio rises even though the CFO-versus-net-income comparison stays flat. The distinction is the practical payoff of [the accruals ratio and the accruals anomaly](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly) — the two formulations are not interchangeable, and this case is exactly where they diverge.

For WorldCom, the balance-sheet version had plenty to work with. Total assets rose from \$98,903 million at the end of 2000 to **\$103,914 million** at the end of 2001, while revenue fell 10%. Assets growing while revenue shrinks is not impossible for a carrier mid-build, but it demands an explanation and the explanation must be physical.

**Assets per unit of physical capacity.** This is the test that cannot be defeated by any journal entry, because it compares the ledger to the world. Gross property and equipment at the end of 2001 was \$48,661 million, of which **\$23,814 million was transmission equipment** and \$5,706 million was construction in progress. The question a forensic analyst asks is simple: how many more route-miles, switches, or terabits does the company control this year than last, and does the increase in the asset account correspond to it?

Roughly \$2.7 billion of that transmission equipment and construction in progress was capacity WorldCom had *rented and consumed*, not capacity it owned. No physical count would ever reconcile to it.

**Depreciation divided by gross property and equipment.** If new assets are being added faster than depreciation is charged on them, the implied average asset life stretches. On the 2001 figures, depreciation and amortization of \$5,880 million against gross property and equipment of \$48,661 million implies an average life of about 8.3 years. Watching that number drift upward over several years is one of the quieter ways a capitalization scheme shows itself, because the "assets" being added have no real life at all.

**The expense ratio against peers.** A cost ratio that holds perfectly steady while every competitor's deteriorates is not evidence of superior management. Sector-wide price pressure reaches everybody. The investigators noted that WorldCom's presentation "made it appear that softening markets were not reducing the Company's profitability, when the opposite was the case."

### The contrast: an honest capital-heavy carrier

It is worth being clear that heavy capital spending is not itself suspicious. A carrier genuinely building a network has high capex, negative free cash flow for years, rising assets, and rising depreciation. All of WorldCom's individual symptoms appear in honest capital-intensive businesses.

What distinguishes them is **corroboration**. Illustratively, an honest carrier building 5,000 route-miles of fibre shows: a capital budget announced in advance and tied to named projects; asset additions that reconcile to those projects; a physical asset register an engineer could walk; depreciation rising in step with the asset base a year or two later; and capacity metrics — route-miles, lit fibre, switch ports — growing in proportion to the spending.

The fraudulent version has the financial symptoms without any of the physical corroboration. Assets grow, and nothing you can stand next to grows with them. That is the test, and it is why the general treatment in the companion post insists on the physical-corroboration question rather than any ratio.

One honest note about detection difficulty. This series' detection models are not equally armed against this trick. The Beneish M-Score, discussed in [the Beneish M-Score: detecting earnings manipulation](/blog/trading/forensic-accounting/the-beneish-m-score-detecting-earnings-manipulation), includes an asset quality index and a depreciation index precisely to catch capitalization games — but those two variables carry the two smallest coefficients in the equation, 0.404 and 0.115. The model is structurally under-weighted against exactly this fraud. No citable source computes an actual M-Score for WorldCom, so no number is offered here.

## How it was actually caught

Not by the SEC. Not by a short-seller. Not by the outside auditor. By an internal audit team, working nights, on a capital-expenditure account it could not reconcile.

### The 27 days

The sequence below comes from the Special Investigative Committee's own reconstruction.

**March 2002.** The company receives an inquiry letter from the SEC. It concerned other matters, but it raised the temperature.

**Late May 2002.** Cynthia Cooper, Vice President of Internal Audit, decides to accelerate a follow-up capital-expenditure audit that had been planned for late 2002 or 2003, and to broaden its scope to include the *accounting treatment* of capital expenditures rather than just the spending. The investigators heard conflicting accounts of the trigger. Candidates include an article a colleague had circulated, lingering concerns from the previous year's capex audit, the SEC inquiry letter, and Sullivan's conduct in an unrelated audit of wireless bad debt — during which, Cooper said, Sullivan went into "a rage" and told her that "she may know how to run an audit, but she did not know how to run a business."

**29 May 2002.** Internal Audit meets a finance executive to discuss why several internal capital-expenditure reports disagree. He explains that some of the variance is "prepaid capacity" — apparently the first time anyone in Internal Audit had heard the term. He says the prepaid capacity adjustments "had been running approximately \$800 million per quarter", that he does not understand them, and refers Internal Audit to the Controller. Later the same day, another manager confirms he has been booking prepaid capacity into property, plant and equipment accounts, and also refers them to the Controller.

**4 June 2002.** The Controller, David Myers, begins discouraging the audit. He emails Cooper: "what is there to do in Capex since we are spending nothing, in relative terms?" He follows up the next day with more questions that Cooper read as a suggestion to look elsewhere.

**Around 11 June 2002.** Cooper mentions to Sullivan that the audit has been moved up and that they are trying to reconcile a **\$3 billion difference between the cash and accrual numbers**. Sullivan says he is familiar with the term prepaid capacity and explains that it is line costs that had been capitalized. He then asks Internal Audit to delay the audit until the third quarter and to examine only the second-quarter numbers, because senior management needs to "clean up" some things regarding capital expenditures in a restructuring planned for the second quarter.

Internal Audit disregarded the request.

**Mid-June 2002.** Gene Morse, a manager in Internal Audit, finds the entries in WorldCom's computerised journal-entry system, starting with the first-quarter 2002 entry. His description of why is the single most useful sentence in the whole file for a working analyst: the prepaid capacity entries "were easy to find because they were very large, round-dollar entries."

**17 June 2002.** Cooper and a colleague walk the entry back through the people who touched it. The accountant who booked one says she made it but does not know what it was for, and that the Director of General Accounting or the Controller would have support. The Director of General Accounting says he has no idea what prepaid capacity is. Then they reach Myers, who first asks whether they had spoken to Sullivan, since he understood the audit was being postponed. Then he gives the answer that ends the fraud: **he does not have support for the prepaid capacity journal entries, and he is not going to create support.** He adds that he was uncomfortable with the entries, but that once they started it was difficult to stop making them.

**18–19 June 2002.** KPMG's engagement partner, Farrell Malone — newly appointed, still learning the books — meets Myers, who repeats that he knew the capitalization was wrong and that there was no support. Malone then meets Sullivan, who argues the matching principle: line costs were capitalized to match costs with related revenue in the future. Asked for documentation, Sullivan says the quarterly amounts were determined "at a high level" from line-cost trend reports. When Myers indicates he would have to *create* documentation, Malone says he is not interested in seeing fabricated support.

**20 June 2002.** The Audit Committee meets at KPMG's Washington offices. Malone says the costs cannot properly be capitalized. Sullivan presents his theory without documentation and is given until Monday to produce a position paper. The investigators record something important about that room: "To some non-accountants, Sullivan's justifications seemed reasonable, and some thought KPMG did not sufficiently understand the Company or the industry." A plausible technical story, told confidently by a respected CFO, is genuinely hard for a board to reject.

**24 June 2002.** The Audit Committee meets again. Sullivan defends the accounting but does not produce the promised paper. KPMG tells him his theory does not hold water. Arthur Andersen, participating by telephone, says the accounting was not in accordance with GAAP and that it is **withdrawing its audit opinions for 2001 and its review of the first quarter of 2002.** After the meeting, Sullivan is asked to resign and refuses; he is fired. Myers is asked separately and resigns.

**25 June 2002.** WorldCom announces the restatement.

**26 June 2002.** The SEC files suit.

### What that story actually says about fraud detection

Four things, and none of them is the moral usually drawn.

**The detection was a reconciliation, not an insight.** Nobody deduced the fraud from the financial statements. Internal Audit had two internal reports about capital expenditure that disagreed by about \$3 billion, and refused to stop asking why. Most fraud is found this way — by someone who will not let an unexplained difference go.

**The entries were easy to find once someone looked.** Very large, round-dollar entries, posted after the quarter closed, without support. This is the least sophisticated concealment imaginable. What protected the fraud was not its cleverness but the fact that the people with authority to look were being told not to.

**The obstruction is the signal.** Read the timeline again as a series of attempts to slow the audit down: the Controller suggesting there is nothing to do in capex; the CFO requesting a delay to the third quarter and a narrowed scope; the earlier instruction to withhold general-ledger access to the area showing corporate adjustments; Sullivan's 1999 note about Cooper's document request — "Do not give her the total picture." When a request for documentation produces a negotiation about scope and timing rather than documents, the negotiation is the finding.

**And the auditor's failure was structural, not merely careless.** The investigators found no evidence Andersen knew, and they were candid about their limits — they had access to only part of Andersen's files and "Andersen personnel refused to speak with us." What they could assess was the method. Andersen used an approach it characterised as different from the "traditional audit approach", focused on identifying risks and assessing whether controls mitigated them rather than on substantive testing of the records. The report's verdict on the consequence is the sentence every analyst reading an audit report should keep:

> a consequence of this approach was that if Andersen failed to identify a significant risk, or relied on Company controls without adequately determining that they were worthy of reliance, there would be insufficient testing to make detection of fraud likely.

And they found the controls were not worthy of reliance: "hundreds of huge, round-dollar journal entries made by the staff of the General Accounting group without proper support", including unsupported entries of \$334,000,000 and \$560,000,000 on 21 July 2000 and 17 July 2001. WorldCom personnel, for their part, "maintained inappropriately tight control over information that Andersen needed, altered documents with the apparent purpose of concealing from Andersen items that might have raised questions", and Andersen, knowing it was getting less than full cooperation, "failed to bring this to the attention of WorldCom's Audit Committee."

That is the practical content of [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch), stated by an investigation rather than a textbook.

## Common misconceptions

**"WorldCom was an \$11 billion fraud announced in June 2002."** No. Two different numbers, years apart. What WorldCom announced on 25 June 2002 was that transfers from line cost expenses to asset accounts of approximately **\$3.8 billion** during 2001 and the first quarter of 2002 were not in accordance with GAAP — \$3.852 billion, in the investigators' precise figure. The SEC's complaint the next day was headlined "SEC Charges WorldCom with \$3.8 Billion Fraud". The larger totals accumulated afterwards, as the review widened: an additional amount announced on 8 August 2002, a statement on 5 November 2002 that the restatements "could total in excess of \$9 billion", and the SEC's amended complaint of the same date stating that the company had acknowledged overstating reported income by **approximately \$9 billion** from at least as early as 1999 through the first quarter of 2002. By March 2003 the Special Investigative Committee had put the line-cost manipulation alone at **over \$7 billion** across Q2 1999 to Q1 2002, plus \$958 million of improperly recorded revenue and a further \$1.107 billion its advisors considered questionable. The round \$11 billion by which the case is now usually known is an aggregate that grew as the investigation continued. It is not what was announced in June 2002, and anyone who presents it as such has not read the sequence.

**"The restatement was \$11 billion, so that is the total damage."** Also no, and in the other direction. The restated accounts filed in March 2004 reduced previously reported net income by **\$17.1 billion for 2001 and \$53.1 billion for 2000**, and cut shareholders' equity by a cumulative **\$70.8 billion** as of 31 December 2001. But the overwhelming bulk of that is not the capitalization fraud — it is the impairment of assets and goodwill from the acquisition spree, recorded as charges of **\$47.2 billion in 2000, \$12.8 billion in 2001 and \$5.0 billion in 2002**. All existing goodwill was written off, and property, plant, equipment and other intangibles previously carried at \$44.4 billion were restated to approximately \$23.8 billion. Those write-downs are the roll-up collapsing, which is a real but different story — the impairment-timing dynamics are covered in [goodwill, intangibles and the impairment timing game](/blog/trading/forensic-accounting/goodwill-intangibles-and-the-impairment-timing-game). Conflating the two makes the accounting fraud look six times larger than it was.

**"Cash flow analysis would have caught it."** Only the right cash flow analysis. The most widely taught version — check that operating cash flow keeps pace with net income — would have handed WorldCom a clean bill of health, because CFO in 2001 was 5.8 times reported net income. Capitalizing an operating cost raises CFO. What would have caught it is free cash flow, which the fraud cannot move, and which was \$108 million.

**"It was a grey-area judgement that got out of hand."** The record does not support this. There was no supporting documentation, and the Controller told Internal Audit he would not create any. Amounts were derived backwards from a target ratio. Entries were booked after quarters closed, in round numbers, off one-page schedules. Assets were moved out of Construction in Progress before the auditors tested that account. And the CFO planned to bury the balances in a restructuring charge. Every one of those facts is inconsistent with a good-faith accounting position.

**"Better outside auditors would have prevented it."** Perhaps, but note who did find it, and with what. Internal Audit found it with access to the general ledger, a difference it could not reconcile, and a refusal to be managed away from the question. The people who obstructed it were the Controller and the CFO — that is, precisely the people an outside auditor relies on for representations. This is why Sarbanes-Oxley's response was aimed at the audit committee's independence and the whistleblower's protection, not only at audit technique.

**"The company was profitable, it just had accounting problems."** In three of the five quarters of the capitalization scheme it was reporting profits that were losses. For full-year 2001 a reported pre-tax profit of \$2,393 million was, on the SEC's figures, a pre-tax loss of \$622 million. The accounting was not decorating a healthy business. It was substituting for one.

## How it shows up in real markets

### The bankruptcy, and the number that came from the fraudulent balance sheet

WorldCom and substantially all of its US subsidiaries filed voluntary Chapter 11 petitions on **21 July 2002** in the United States Bankruptcy Court for the Southern District of New York, case number 02-13533. It was the largest bankruptcy filing in US history at the time.

The size figure usually quoted for it, about **\$103.9 billion of assets**, is worth pausing on. It comes from the total assets line of the FY2001 balance sheet — \$103,914 million — which is to say from the accounts that were about to be restated. Even the headline measure of the collapse was a number the fraud had inflated.

Its stock had already gone. Nasdaq delisted the WorldCom group and MCI group shares effective at the opening of trading on **30 July 2002**; as of 30 June 2002 the WorldCom group stock was quoted at an average bid and asked price of **\$0.83 per share.** The plan of reorganization was confirmed on **31 October 2003**, and the company emerged as MCI in 2004.

### The enforcement outcome

The SEC filed on 26 June 2002 charging violations of Exchange Act Sections 10(b) and 13(a) and Rules 10b-5, 12b-20, 13a-1 and 13a-13. It amended the complaint on 5 November 2002 to add Securities Act Section 17(a) and the books-and-records and internal-controls provisions, Sections 13(b)(2)(A) and 13(b)(2)(B) — the internal-controls charge being the formal recognition that the failure was systemic rather than a single bad entry.

On the money, the sequence is instructive about what enforcement can actually extract from a bankrupt company. The SEC first proposed a penalty of \$1.51 billion. Judge Jed S. Rakoff approved a revised settlement on **7 July 2003** carrying a civil penalty of **\$2,250,000,000** — a figure the court observed would be "75 times greater than any prior such penalty" — while providing that, on confirmation of a plan of reorganization, the obligation would be satisfied by **\$500 million in cash and \$250 million of stock in the reorganized company**, distributed to victims under **Section 308 (Fair Funds for Investors) of the Sarbanes-Oxley Act of 2002.** The bankruptcy court approved it on 6 August 2003. A \$2.25 billion judgment, \$750 million actually paid.

Individually, the SEC brought civil actions against Controller **David F. Myers** (26 September 2002), Director of General Accounting **Buford "Buddy" Yates, Jr.** (7 October 2002), and accountants **Betty L. Vinson** and **Troy M. Normand** (10 October 2002). In March 2004 the SEC charged CFO **Scott D. Sullivan**, who consented to a permanent antifraud injunction, a permanent officer-and-director bar, and a permanent suspension from practising as an accountant before the Commission; the same day he pleaded guilty to criminal charges brought by the US Attorney's Office for the Southern District of New York, and that office announced the related indictment of former chief executive **Bernard J. Ebbers**. Criminal proceedings against Ebbers followed in that court. The specific custodial sentences later imposed are not restated here, because the primary court records could not be verified with the sources available for this article — a gap worth naming rather than papering over with a number from memory.

Judge Rakoff also appointed former SEC Chairman **Richard C. Breeden** as corporate monitor on 3 July 2002. Breeden's report, *Restoring Trust*, issued 26 August 2003, became one of the era's governance templates.

### The legislative response

The **Sarbanes-Oxley Act of 2002**, Public Law 107-204, was approved on **30 July 2002** — nine days after the bankruptcy filing and five weeks after WorldCom's announcement. Four of its provisions map directly onto what went wrong here.

**Section 302, Corporate Responsibility for Financial Reports,** requires the chief executive and chief financial officer to certify each periodic report personally. WorldCom's fraud ran through the CFO. Section 302 makes the signature an individual assertion rather than a corporate one, and **Section 906** attaches criminal liability to a knowingly false certification.

**Section 404, Management Assessment of Internal Controls,** requires management to assess and report on internal control over financial reporting, and the auditor to attest. The whole WorldCom mechanism depended on hundreds of large journal entries without support — a control failure that no financial-statement audit was designed to surface but that a controls assessment is designed to catch.

**Section 301, Public Company Audit Committees,** places the audit committee in charge of appointing and overseeing the outside auditor, requires its members to be independent, and requires it to establish procedures for handling complaints about accounting matters, including confidential and anonymous employee submissions. Recall that Andersen knew it was receiving less than full cooperation and did not tell the Audit Committee, and that Internal Audit's access to the ledger area showing corporate adjustments had been restricted at the Controller's instruction.

**Section 806** protects employees of public companies who provide evidence of fraud from retaliation. The people who broke this case were employees whose superiors were actively trying to redirect them.

Whether that machinery works is a fair question and not this article's. What is not in doubt is that its shape was drawn around this case.

### The pattern in later cases

The same reclassification logic recurs, which is why it is worth learning as a shape rather than a story.

**Software and cloud capitalization.** The routine modern version is capitalized software development cost, where the boundary between research (expense) and development of an asset (capitalize) is genuinely a judgement. The forensic questions are the same ones: does capitalized development rise as margins come under pressure, does amortization lag the capitalized balance, and is the policy stable across periods?

**Classification shifting more broadly.** Moving items between statement sections to flatter a metric is a family of techniques, not one trick — including moving financing inflows into operating cash flow. The family is catalogued in [cash flow statement manipulation: classification shifting](/blog/trading/forensic-accounting/cash-flow-statement-manipulation-classification-shifting).

**Reserve releases before the main event.** WorldCom used up its accruals first, then escalated. That order is common: the softer technique is exhausted before the harder one begins, which means a company that has been quietly releasing reserves is a company that may be about to do something worse. The escalation itself is a signal.

**Round numbers after the quarter closes.** \$544 million. \$560 million. \$743 million. \$841 million. \$818 million. Round-dollar entries booked into the post-close window, in amounts a business does not naturally produce, are among the most reliable markers there are — the reason [Benford's law and digit analysis for fraud](/blog/trading/forensic-accounting/benfords-law-and-digit-analysis-for-fraud) works at all.

## When this matters to you

If you read financial statements — as an investor, a lender, a credit analyst, or someone deciding whether to join a company — WorldCom leaves four things worth carrying.

**Free cash flow, not operating cash flow.** Make CFO minus capital expenditure the number you look at first, and look at it over five years rather than one. It is the only headline measure that a reclassification between operating and investing cannot touch. WorldCom's was \$108 million in the year it reported \$1.4 billion of net income.

**Assets have to correspond to something you could stand next to.** When an asset account grows, ask what physically arrived. Route-miles, square feet, servers, machines. If the answer is a category rather than an object — "network capacity", "platform development", "customer relationships" — that is where to spend your attention.

**A ratio that never moves is not a well-managed ratio.** Real operating metrics are noisy because the world is noisy. Uncanny stability in a number management is judged on is a reason to look harder, not to relax. WorldCom's E/R ratio was 42% quarter after quarter, and that was the tell.

**Detection comes from unresolved differences, and from people who will not drop them.** Not from models. Two internal reports disagreed by \$3 billion, and someone kept asking. If you are ever inside a company and cannot reconcile something, and the response you get is about scope and timing rather than about the difference, you have already learned the important thing.

This is educational material about how to read financial statements, not investment advice, and nothing here is a recommendation about any security.

For the general technique that this case made famous — including where capitalization is entirely legitimate — read [capitalizing costs to inflate profit: the WorldCom move](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move). For the statement that would have exposed it, [reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

## Sources & further reading

Primary documents, all consulted directly for this article.

- **Report of Investigation by the Special Investigative Committee of the Board of Directors of WorldCom, Inc.** (Dennis R. Beresford, Nicholas deB. Katzenbach, C.B. Rogers, Jr.; counsel Wilmer, Cutler & Pickering; accounting advisors PricewaterhouseCoopers LLP), dated 31 March 2003, filed as Exhibit 99.1 to WorldCom's Form 8-K of 9 June 2003. The source for the E/R ratio target of about 42% and the actual level "typically exceeding 50%"; the approximately \$3.3 billion of accrual releases in 1999–2000 and the three manipulations; the "over \$7 billion" of total line-cost reduction from Q2 1999 to Q1 2002; the quarterly capitalization amounts of \$544m, \$560m, \$743m, \$841m and \$818m; the April 2001 entry and the "Prepaid Capacity Costs" label; the \$958 million of improper revenue and \$1.107 billion of questionable items; the internal-audit discovery timeline; and the assessment of Arthur Andersen's audit approach. [sec.gov](https://www.sec.gov/Archives/edgar/data/723527/000093176303001862/dex991.htm)
- **WorldCom, Inc. Annual Report on Form 10-K for the fiscal year ended 31 December 2001**, filed 13 March 2002. The source for reported revenue of \$35,179m, reported line costs of \$14,739m, the reported line cost E/R ratios of 41.0%/39.6%/41.9% for 1999–2001, operating cash flow of \$7,994m, capital expenditures of \$7,886m, total assets of \$103,914m, gross property and equipment of \$48,661m including \$23,814m of transmission equipment, and depreciation and amortization of \$5,880m. [sec.gov](https://www.sec.gov/Archives/edgar/data/723527/000100547702001226/d02-36461.txt)
- **WorldCom/MCI Annual Report on Form 10-K for the fiscal year ended 31 December 2002**, filed 12 March 2004 — the restated accounts. The source for the chronology of the 25 June 2002, 8 August 2002, 5 November 2002 and 13 March 2003 announcements; the restatement's \$17.1 billion and \$53.1 billion reductions to previously reported net income for 2001 and 2000; the \$70.8 billion cumulative reduction in shareholders' equity; the impairment charges of \$47.2 billion, \$12.8 billion and \$5.0 billion; the write-off of all goodwill and the restatement of property, plant, equipment and other intangibles from \$44.4 billion to approximately \$23.8 billion; and the bankruptcy filing details. [sec.gov](https://www.sec.gov/Archives/edgar/data/723527/000119312504039709/d10k.htm)
- **SEC Litigation Release No. 17588 / Accounting and Auditing Release No. 1585**, 27 June 2002, *SEC v. WorldCom, Inc.*, Civil Action 02 CV 4963 (S.D.N.Y.). "SEC Charges WorldCom with \$3.8 Billion Fraud" — the source for the approximately \$3.055 billion and \$797 million overstatements of income before income taxes and minority interests, the charges brought, and the SEC's statement that the entries were intended to keep earnings "in line with estimates by Wall Street analysts". [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-17588)
- **SEC Litigation Release No. 17829**, 5 November 2002 — the amended complaint, the added Securities Act Section 17(a) and internal-controls charges, and the company's acknowledgement of overstating reported income by approximately \$9 billion from at least as early as 1999 through Q1 2002. [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-17829)
- **SEC Litigation Release No. 18219**, July 2003 — Judge Rakoff's opinion approving the settlement, the \$2,250,000,000 civil penalty, its satisfaction by \$500 million cash and \$250 million of stock, the Section 308 Fair Funds distribution, and the dates of the civil actions against Myers, Yates, Vinson and Normand. [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-18219)
- **SEC Litigation Release No. 18277 / AAER No. 1834**, 7 August 2003 — bankruptcy court approval of the settlement on 6 August 2003, and the district court's prior approval on 7 July 2003; *In re WorldCom, Inc.*, Ch. 11 Case No. 02-13533 (Bankr. S.D.N.Y.). [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-18277)
- **SEC Litigation Release No. 18605**, March 2004, *SEC v. Scott D. Sullivan*, Civil Action No. 04 CV 1706 — Sullivan's consent to injunction, officer-and-director bar and Rule 102(e) suspension, his guilty plea to criminal charges filed by the US Attorney for the Southern District of New York, and that office's announcement of the related indictment of Bernard J. Ebbers. [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-18605)
- **Sarbanes-Oxley Act of 2002**, Public Law 107-204 (H.R. 3763), approved 30 July 2002 — Sections 301, 302, 308, 404, 806 and 906 as cited. [govinfo.gov](https://www.govinfo.gov/content/pkg/PLAW-107publ204/html/PLAW-107publ204.htm)

A note on what is illustrative and what is sourced. Every dated figure attributed to a document above is quoted from that document. The worked examples using round numbers — the \$1,000m reclassification, the accrual-release walkthrough, the depreciation-drag table, the \$8,000 revenue carrier, and the two-column cash flow statement including its \$350 deferred tax line — are invented teaching arithmetic, labelled as illustrative where they appear. Where sources differ slightly, both figures are given with their provenance rather than merged: the SEC's litigation release describes a 2001 income overstatement of approximately \$3.055 billion, while the sum of the four quarterly capitalization amounts in the Special Investigative Committee's report is \$2,688 million and the complaint's implied 2001 line-cost difference is \$3,015 million. These describe related but distinct things, and reconciling the method matters more than producing a single tidy number.
