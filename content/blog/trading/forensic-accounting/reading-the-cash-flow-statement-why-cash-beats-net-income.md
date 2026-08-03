---
title: "Reading the Cash Flow Statement: Why Cash Beats Net Income"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "A beginner-friendly deep dive into the three sections of the cash flow statement, the indirect method line by line, free cash flow, and the four manipulations that actually work on the statement everyone calls unfakeable."
tags: ["cash-flow-statement", "operating-cash-flow", "free-cash-flow", "indirect-method", "working-capital", "forensic-accounting", "earnings-quality", "classification-shifting", "receivable-factoring", "reverse-factoring", "capitalization", "financial-statement-analysis"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 53
---

> [!important]
> **TL;DR** — Net income is an opinion; cash is closer to a fact. The cash flow statement is where you check the opinion, and learning to rebuild it line by line is the single highest-leverage skill in reading financial statements.
>
> - The statement has exactly **three sections** — operating (CFO), investing (CFI), financing (CFF) — and they must sum to the change in the bank balance. That built-in check is what makes the statement useful.
> - The **indirect method** starts at net income and walks to cash by adding back non-cash charges and adjusting for working capital. One rule generates every working-capital line: **assets and cash move in opposite directions; liabilities and cash move together.**
> - **Free cash flow = CFO − capital expenditures.** It is the number that actually funds dividends and buybacks, and it is the only headline metric immune to the WorldCom trick.
> - CFO is the hardest number to fake because cash eventually has to sit in a bank — but it is not unfakeable. Four things genuinely work: **classification shifting** (Dynegy moved \$300 million of borrowing into CFO), **receivable factoring** (GE pulled forward more than \$1.4 billion in 2016), **stretching payables** (Carillion owed roughly £498 million through an off-borrowings early-payment facility), and **capitalizing what should be expensed** (WorldCom moved \$3.852 billion of line costs to asset accounts).
> - The number to remember: **WorldCom's fraud raised reported operating cash flow, not just reported profit.** The most famous accounting fraud of the era is precisely the one that "just look at cash flow" would have missed — which is why you look at free cash flow too.

There is a piece of investing folklore that goes: *earnings can be manipulated, but cash is cash.* It is repeated so often that it has hardened into a rule, and like most hardened rules it is about eighty percent true and dangerous in the remaining twenty.

The eighty percent is real. Net income is assembled from estimates — how much of this receivable will we collect, over how many years does this machine wear out, what is this warranty going to cost us — and estimates can be shaded. Cash flow from operations is assembled from bank movements, and a bank movement either happened or it did not. That asymmetry is why the cash flow statement exists, why regulators made it mandatory, and why it is the first place a forensic reader looks after the headline number.

The twenty percent is what this article is really about. There are four manipulations that work *on the cash flow statement itself*, they have all been used at scale by companies you have heard of, and one of them inflated operating cash flow by nearly four billion dollars at a company that everyone later held up as the reason you should watch cash flow. Learning the statement properly means learning both halves: the mechanics that make it trustworthy, and the seams where that trust breaks.

We are going to build the whole thing from zero. No accounting background assumed. By the end you will be able to take a company's income statement and two balance sheets and reconstruct its cash flow statement yourself — which, it turns out, is exactly the skill that lets you notice when the published one has been massaged.

The diagram below is the mental model for everything that follows: three buckets, one bank account.

![The three sections of the cash flow statement — operating +13,800, investing −8,800, financing −1,300 — summing to a net change in cash of +3,700 that carries beginning cash of 9,000 to ending cash of 12,700.](/imgs/blogs/reading-the-cash-flow-statement-why-cash-beats-net-income-1.webp)

Three sections. Every dollar of cash that moved in or out of the company during the period lands in exactly one of them. They add up to the change in the bank balance, and the bank balance is a number the company cannot argue with. Hold that structure in your head; the rest of this article is filling it in.

## Foundations: how the cash flow statement actually works

Before the tricks, the machinery. This section assumes you know nothing and defines everything on first use. If you already read financial statements for a living, skim to the signature table; if you do not, read every line, because the forensic half of the article stands entirely on this.

### Why a third statement exists at all

A company publishes three primary financial statements, and they answer three different questions. I have written about how they lock together in [the three financial statements and how they interlock](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock); here is the short version.

The **income statement** answers *did we do profitable business this period?* It is built on **accrual accounting** — revenue is recorded when it is *earned* (you delivered the thing), and expenses when they are *incurred* (you consumed the resource), regardless of when cash moved. That is a genuinely better description of a period's economics than a bank statement, and it is why every serious accounting regime requires it.

The **balance sheet** answers *what do we own and owe right now?* It is a snapshot at one instant: assets on one side, liabilities and equity on the other.

The **cash flow statement** answers *where did the money actually go?* It exists because the first two statements, taken alone, can describe a company that is simultaneously profitable and running out of money. That is not a hypothetical failure mode — it is the single most common way a growing business dies. You sell more, you build more inventory to sell it, your customers pay you in ninety days, your suppliers want paying in thirty, and you are profitable and insolvent at the same time.

So a third statement was required. In the United States, the modern version arrived with **SFAS 95** in 1987, which replaced a vaguer "statement of changes in financial position" and mandated the three-section format we still use; internationally, **IAS 7** does the same job. Both require the same three buckets.

The reason the statement resists manipulation is structural rather than moral. The income statement has no external anchor — nothing outside the company forces "revenue" to equal any particular number. The cash flow statement has one: it must reconcile to the cash line on the balance sheet, and the cash line on the balance sheet is confirmed by an outside party, the bank. That anchor is the whole ballgame, and when we get to Wirecard we will see what happens when the anchor itself is forged.

### The one identity the statement must satisfy

Everything on the statement serves a single equation:

$$\text{CFO} + \text{CFI} + \text{CFF} + \text{FX effect} = \Delta\text{Cash}$$

where **CFO** is net cash from operating activities, **CFI** is net cash from investing activities, **CFF** is net cash from financing activities, the **FX effect** is the translation impact of holding cash in foreign currencies (usually small, often zero for a domestic company), and **ΔCash** is the change in cash and cash equivalents between the start and end of the period.

*Cash equivalents*, since we are defining everything: highly liquid investments with an original maturity of three months or less — treasury bills, money-market funds, overnight deposits. Close enough to cash that they are counted as cash.

This identity is not a convention. It is arithmetic. Every dollar that entered or left the bank account is assigned to one of three categories, so the categories must add up to the movement in the account. A cash flow statement that does not foot is not a controversial cash flow statement; it is a broken one.

And here is the consequence that matters forensically, which we will return to repeatedly: **a manipulation that moves a dollar between two sections does not change the bottom of the statement at all.** The bank balance is identical. The identity still holds. Nothing fails to foot. All that changes is which bucket the dollar sits in — and since the market prices CFO very differently from CFF, that is often the entire point of the exercise.

### Operating activities: cash the business itself made

**Cash flow from operations (CFO)** is the cash generated by the company doing the thing it exists to do — selling its products or services and paying the costs of doing so. In principle it is the cash version of the income statement: collections from customers minus payments to suppliers, employees, landlords, and the tax authority.

What belongs here:

- Cash collected from customers
- Cash paid to suppliers and employees
- Cash paid for rent, utilities, marketing, insurance
- Income taxes paid
- Interest paid and interest received (under US GAAP; IFRS gives you a choice, which we will come back to, because that choice is a manipulation surface)

CFO is the number a forensic reader looks at first, and the reason is worth stating precisely. Revenue can be recognized on a delivery you made to a distributor who has not sold it on and may return it. An expense can be deferred by lengthening an asset's assumed useful life. But a customer either wired you money or did not. CFO is the closest thing in the financial statements to a claim that can be checked against the outside world.

The standard way to say this is that CFO is where accounting judgment goes to die. Not entirely, as we will see — but mostly, and that is enough to make it the most informative single line in the filing.

### Investing activities: cash spent on, or released by, long-lived assets

**Cash flow from investing (CFI)** covers the purchase and sale of things the company expects to hold and use for more than one period, plus the purchase and sale of financial investments.

What belongs here:

- **Capital expenditures (capex)** — cash spent buying property, plant, and equipment: factories, servers, machines, vehicles. This is almost always the largest line.
- Purchases and sales of marketable securities (the corporate treasury parking spare cash in bonds)
- Cash paid for acquisitions, net of any cash that came with the acquired company
- Cash received from selling a division, a building, or a piece of equipment

CFI is usually negative for a healthy company, and that is a good sign, not a bad one: a business that is not spending on assets is a business that is not investing in its future. Persistently *positive* CFI, on the other hand, means the company is a net seller of its own assets — which is either a deliberate divestiture programme or a slow-motion liquidation, and it is worth knowing which.

### Financing activities: cash from and to the people who funded you

**Cash flow from financing (CFF)** covers transactions with the two groups who supplied the company's capital — lenders and shareholders.

What belongs here:

- Proceeds from issuing debt; repayments of debt principal
- Proceeds from issuing shares
- Share repurchases (buybacks)
- Dividends paid
- Principal payments on lease liabilities

Notice a subtlety that trips up nearly every beginner: **interest paid is an operating item under US GAAP, but repaying the loan principal is a financing item.** The same loan generates cash flows in two different sections. That is not sloppiness — it reflects the idea that interest is a cost of doing business in the period while principal is a return of capital — but it is a seam, and IFRS handles it differently, which creates a comparability problem we will get to.

#### Worked example: Northwind Tools' complete cash flow statement

Let us build a full statement for a hypothetical company so that every later example has something concrete to attach to. **Northwind Tools is invented** — the numbers are illustrative arithmetic, not a real company's results — but the structure and the relationships between the lines are exactly what you will find in a real 10-K.

All figures are in thousands of dollars, which is how most mid-cap filings present them.

Northwind's income statement for FY2025:

| Line | FY2025 |
| --- | ---: |
| Revenue | 90,000 |
| Cost of revenue | (46,000) |
| **Gross profit** | **44,000** |
| Research and development | (6,000) |
| Selling, general and administrative | (18,000) |
| Depreciation and amortization | (4,500) |
| **Operating income** | **15,500** |
| Interest expense | (600) |
| **Pretax income** | **14,900** |
| Income tax expense | (2,900) |
| **Net income** | **12,000** |

And its cash flow statement for the same year:

| Operating activities | FY2025 |
| --- | ---: |
| Net income | 12,000 |
| Depreciation and amortization | 4,500 |
| Stock-based compensation | 2,000 |
| Deferred income taxes | 300 |
| Accounts receivable | (6,000) |
| Inventory | (3,000) |
| Prepaid expenses | (200) |
| Accounts payable | 2,500 |
| Accrued liabilities | 700 |
| Deferred revenue | 1,000 |
| **Net cash provided by operating activities** | **13,800** |

| Investing activities | FY2025 |
| --- | ---: |
| Purchases of property and equipment | (5,200) |
| Purchases of marketable securities | (3,000) |
| Maturities of marketable securities | 1,800 |
| Acquisition, net of cash acquired | (2,400) |
| **Net cash used in investing activities** | **(8,800)** |

| Financing activities | FY2025 |
| --- | ---: |
| Proceeds from revolving credit facility | 5,000 |
| Repayments of long-term debt | (2,000) |
| Repurchases of common stock | (3,500) |
| Dividends paid | (1,200) |
| Proceeds from stock option exercises | 400 |
| **Net cash used in financing activities** | **(1,300)** |

| Reconciliation | FY2025 |
| --- | ---: |
| Net increase in cash | 3,700 |
| Cash at beginning of year | 9,000 |
| **Cash at end of year** | **12,700** |

Check the identity: 13,800 − 8,800 − 1,300 = 3,700, and 9,000 + 3,700 = 12,700. It foots. That reconciliation at the bottom is the statement telling you it has not lost track of any money.

Now read the story it tells. Northwind earned 12,000 of accounting profit and collected 13,800 of actual cash from operations — cash *exceeded* profit, which is the healthy direction. It spent 5,200 on equipment and 2,400 buying a small company, drew 5,000 on its revolver, paid down 2,000 of term debt, bought back 3,500 of stock, and paid 1,200 of dividends. It ended the year with 3,700 more cash than it started.

One thing should already look mildly odd, and we will come back to it: the company drew 5,000 on a revolving credit facility in a year when it generated 13,800 of operating cash. Why borrow when you are cash-generative? Sometimes there is a perfectly good answer — funding the acquisition, a seasonal working-capital swing, a rate arbitrage. Sometimes there is not. **The intuition: the cash flow statement is not a scoreboard, it is a narrative — the three sections together tell you what kind of company this is and where its money comes from.**

### Reading the signature: what the three signs tell you

Before any arithmetic, look at the *signs* of the three sections. The pattern alone classifies the business.

![A four-row grid of cash-flow signatures: mature healthy (+,−,−), growth or startup (−,−,+), harvesting or shrinking (+,+,−), and distressed (−,+,+), each with what the pattern usually means.](/imgs/blogs/reading-the-cash-flow-statement-why-cash-beats-net-income-4.webp)

| Signature | CFO | CFI | CFF | What it usually means |
| --- | :---: | :---: | :---: | --- |
| Mature, healthy | + | − | − | Funds itself, invests in the business, returns the surplus to shareholders |
| Growth / startup | − | − | + | Burning cash to grow, financed by investors or lenders |
| Harvesting / shrinking | + | + | − | Generating cash and selling assets, using both to pay down debt |
| Distressed | − | + | + | Losing cash, selling assets *and* borrowing to stay alive |

Northwind's signature is (+, −, −): the mature, healthy pattern. A pre-revenue biotech will show (−, −, +) and that is entirely appropriate — it is supposed to be burning cash. The one to look at hard is the last row. A company with negative operating cash flow that is simultaneously selling assets and raising new money is funding its existence from sources that will run out. That signature does not prove distress, but the number of genuinely healthy companies that display it is small.

This is a thirty-second read that you can do on any filing before you look at a single number, and it will orient everything that follows.

## The indirect method, line by line

Now the mechanics. Almost every company you will ever read presents CFO using the **indirect method**, which starts at net income and adjusts its way to cash. Learning to do that walk yourself is the core skill of this article, because a forensic reader does not just read the CFO section — they *predict* it, and then look at where reality diverges.

![A vertical waterfall from net income of 12,000 through non-cash add-backs of D&A 4,500, stock-based compensation 2,000, and deferred taxes 300, then working-capital changes for receivables, inventory, prepaids, payables, accrued liabilities and deferred revenue, landing on cash from operations of 13,800.](/imgs/blogs/reading-the-cash-flow-statement-why-cash-beats-net-income-2.webp)

### Why they start at net income at all

There are two permitted formats. The **direct method** lists actual cash flows: cash collected from customers, cash paid to suppliers, cash paid to employees, taxes paid. It is far more intuitive — it reads like a bank statement — and standard-setters have preferred it for decades.

Almost nobody uses it. Under IFRS the direct method is encouraged; in practice a small minority of filers choose it, and under US GAAP a company that presents the direct method must *also* present the indirect reconciliation, which means doing the work twice. The practical objection is that most accounting systems are not built to tag every cash movement by its operating purpose, so producing a direct-method statement is genuine extra work with no extra revenue attached.

So we get the **indirect method**: start at net income, undo everything about it that was not cash, and see what is left. It is less readable and more informative, because the adjustments themselves are the interesting part. The gap between net income and CFO *is* the accrual layer — the estimate-driven, judgment-laden portion of earnings — and the indirect method itemizes it for you. I go into what that layer means for earnings quality in [accrual accounting versus cash: the gap fraud exploits](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits).

The walk has two stages: remove non-cash charges, then adjust for working capital.

### Stage 1: add back the non-cash charges

Some expenses reduced net income without any money leaving the building. They must be added back.

**Depreciation and amortization (D&A)** is the big one. When Northwind bought a machine for 10,000 with a ten-year life, the cash left the building on the day of purchase and was recorded in *investing*. The income statement then charges 1,000 a year for ten years to spread the cost across the periods the machine helps produce revenue. That annual 1,000 is a real economic cost but not a cash payment — the payment already happened, years ago, in a different section. So we add it back. Northwind adds back 4,500.

*Amortization* is the same idea applied to intangible assets — a purchased patent, customer relationships acquired in a deal, capitalized software. Same mechanic, different asset class.

**Stock-based compensation (SBC)** is the second big one. When a company pays an engineer partly in shares, it records compensation expense on the income statement — the shares have value, and the accounting correctly says so — but no cash moved. So it is added back. Northwind adds back 2,000.

SBC deserves a warning label. It is a genuine cost: the company gave away a slice of itself, and existing shareholders own proportionally less than they did. Adding it back to reach *cash* flow is correct arithmetic. Treating the added-back version as a measure of what shareholders earned is not, and we will return to this when we discuss free cash flow, because it is the single most common way modern technology companies flatter themselves.

**Deferred income taxes** capture the gap between tax expense on the income statement and tax actually paid to the government in the period. Northwind's income statement shows 2,900 of tax expense; 300 of that was deferred rather than paid, so 300 is added back.

Other add-backs you will see: impairments and write-downs (an asset was marked down; no cash moved), losses or gains on asset sales (removed from CFO because the whole proceeds belong in investing), provisions for bad debts, and non-cash lease expense.

### Stage 2: adjust for working capital

This is the part that confuses beginners, and it should not, because it is generated by one rule.

**Working capital** is the short-term operating assets and liabilities that turn over as the business runs: receivables, inventory, prepaid expenses, payables, accrued liabilities, deferred revenue. Net income was calculated on the accrual basis, which means it counted sales you have not been paid for and ignored costs you have already paid. The working-capital adjustments correct for exactly that.

Here is the rule, and it is the only thing you need to memorize in this entire article:

![A two-by-two matrix of working-capital sign rules: an operating asset increasing means cash out, an operating asset decreasing means cash in, an operating liability increasing means cash in, an operating liability decreasing means cash out.](/imgs/blogs/reading-the-cash-flow-statement-why-cash-beats-net-income-3.webp)

**Assets and cash move in opposite directions. Liabilities and cash move together.**

Work through why, once, and you will never need the mnemonic again:

- **Accounts receivable increases** → cash *decreases*. Receivables going up means you booked sales that customers have not paid for. Net income counted them; the bank did not. Subtract the increase. Northwind's receivables rose 6,000, so CFO takes −6,000.
- **Inventory increases** → cash *decreases*. You spent money buying or building goods that have not been sold yet. Net income does not reflect that spending at all — inventory sits on the balance sheet until it is sold and becomes cost of goods sold. Subtract it. Northwind: −3,000.
- **Prepaid expenses increase** → cash *decreases*. You paid the annual insurance premium up front; the income statement will recognize it over twelve months. Cash left now. Northwind: −200.
- **Accounts payable increases** → cash *increases*. Payables going up means you took delivery of goods and services and have not paid for them. The income statement expensed them; your bank account did not. Add the increase. Northwind: +2,500.
- **Accrued liabilities increase** → cash *increases*. Same logic for wages earned but not yet paid, taxes owed but not yet remitted. Northwind: +700.
- **Deferred revenue increases** → cash *increases*. This one is the most cheerful line on the statement. Deferred revenue means a customer paid you *in advance* for something you have not yet delivered. The cash is in your account; the income statement will not recognize the revenue until you deliver. Add it. Northwind: +1,000.

That last one is worth dwelling on. A software company with rapidly growing deferred revenue is collecting cash a year ahead of recognizing it — its CFO will run structurally above its net income, and that is a sign of pricing power, not manipulation. Conversely, a company whose deferred revenue is *shrinking* is recognizing revenue it collected in earlier years, and its CFO will run below net income for entirely honest reasons. Signs on this line tell you about the business model before they tell you anything about integrity.

#### Worked example: building CFO from net income, one rung at a time

Let us do the full walk on Northwind, showing every step.

Start: **net income 12,000**.

Non-cash add-backs:

1. `+ 4,500` depreciation and amortization → running total **16,500**
2. `+ 2,000` stock-based compensation → **18,500**
3. `+ 300` deferred income taxes → **18,800**

At this point 18,800 is what we might call "net income if nothing non-cash had happened." Now the working-capital corrections:

4. `− 6,000` receivables rose → **12,800**
5. `− 3,000` inventory rose → **9,800**
6. `− 200` prepaid expenses rose → **9,600**
7. `+ 2,500` payables rose → **12,100**
8. `+ 700` accrued liabilities rose → **12,800**
9. `+ 1,000` deferred revenue rose → **13,800**

**Cash from operations: 13,800.**

Two summary numbers fall out immediately. The **cash conversion ratio** — CFO divided by net income — is 13,800 / 12,000 = **1.15**. And **accruals**, defined as net income minus CFO, are 12,000 − 13,800 = **−1,800**. Negative accruals mean cash outran profit, which is the conservative direction; that relationship is the entire basis of the [accruals ratio and the accruals anomaly](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly).

Notice what the walk revealed that the headline never would. Northwind's non-cash add-backs contributed 6,800 — more than half of net income. Its working capital *consumed* 5,000. Both facts matter, and neither is visible from the income statement. **The intuition: the indirect method is a confession, laid out in order — the add-backs tell you how much of profit was bookkeeping, and the working-capital lines tell you how much of the business's growth had to be financed out of pocket.**

#### Worked example: deriving the adjustments yourself, from two balance sheets

Here is the skill that turns you from a reader into an analyst. You do not have to take the company's working-capital adjustments on faith; you can recompute them from the balance sheet. If they do not match, that discrepancy is information.

Northwind's balance sheet, the relevant lines only:

| Line | FY2024 | FY2025 | Change |
| --- | ---: | ---: | ---: |
| Cash and equivalents | 9,000 | 12,700 | +3,700 |
| Accounts receivable | 20,000 | 26,000 | +6,000 |
| Inventory | 9,000 | 12,000 | +3,000 |
| Prepaid expenses | 1,300 | 1,500 | +200 |
| Accounts payable | 5,600 | 8,100 | +2,500 |
| Accrued liabilities | 2,400 | 3,100 | +700 |
| Deferred revenue | 3,000 | 4,000 | +1,000 |

Every change column matches a line in the CFO section, with the sign flipped for assets. That is the reconciliation, and you just did it in your head.

Now use it to answer a question the income statement cannot. Northwind reported revenue of 90,000. **How much cash did it actually collect from customers?**

Cash collected = revenue − increase in receivables = 90,000 − 6,000 = **84,000**.

So Northwind recognized 90,000 of sales and banked 84,000 of them. The other 6,000 is sitting in receivables, a promise rather than a deposit. You can put a time value on that too. **Days sales outstanding (DSO)** measures how long, on average, a sale sits as a receivable before it turns into cash:

$$\text{DSO} = \frac{\text{Accounts receivable}}{\text{Revenue}} \times 365$$

For FY2025: 26,000 / 90,000 × 365 = **105.4 days**. For FY2024, with revenue of 72,000: 20,000 / 72,000 × 365 = **101.4 days**. Receivables aged by four days.

Four days is not alarming. Twenty-five days would be. **The intuition: you can rebuild the entire operating section from the income statement and two balance sheets, which means you can also compute what CFO *should* have been — and any company whose reported CFO beats your reconstruction is telling you something you need to understand.**

### The direct method, and one thing it would have shown you

Under the direct method, Northwind's operating section would open with a line reading roughly *cash received from customers: 84,000* — the number we just derived. Then *cash paid to suppliers*, *cash paid to employees*, *income taxes paid*, and so on down to the same 13,800.

It is strictly more informative. A reader could see collections falling while revenue rose, without doing any arithmetic. That is precisely the disclosure the indirect method makes you work for, and it is a reasonable suspicion that this is not entirely an accident of accounting-system architecture. Standard-setters keep proposing the direct method; preparers keep resisting on cost grounds; the reconciliation requirement makes adopting it strictly more work than not adopting it. The net result is that the most useful presentation of the most trustworthy statement is the one almost nobody publishes.

## Free cash flow: the number that actually funds things

CFO tells you how much cash the operations produced. It does not tell you how much was *left over*, because a business that needs to spend heavily on equipment just to stand still has not really produced spendable cash.

**Free cash flow (FCF)** fixes that:

$$\text{FCF} = \text{CFO} - \text{Capital expenditures}$$

Capital expenditures are the purchases-of-property-and-equipment line in the investing section. FCF is the cash a company generated after paying for the physical assets it needs to keep operating — and it is the pool from which dividends, buybacks, debt repayment, and acquisitions are actually funded.

![A ladder from cash from operations of 13,800 minus capital expenditures of 5,200 to free cash flow of 8,600, then minus dividends of 1,200 and buybacks of 3,500 to discretionary cash of 3,900.](/imgs/blogs/reading-the-cash-flow-statement-why-cash-beats-net-income-5.webp)

#### Worked example: Northwind's free cash flow and what it pays for

Take the statement we built:

- Cash from operations: **13,800**
- Less capital expenditures: **(5,200)**
- **Free cash flow: 8,600**

Now check whether the shareholder returns were self-funded:

- Dividends paid: **(1,200)**
- Share repurchases: **(3,500)**
- Total returned to shareholders: **(4,700)**
- **Discretionary cash remaining: 3,900**

Northwind returned 4,700 out of 8,600 of free cash flow — a **payout ratio on FCF of about 55%** — and had 3,900 left for debt reduction, acquisitions, or the balance sheet. That is a comfortably self-funded capital return.

Compare the three profitability-ish numbers for the same year: net income 12,000, CFO 13,800, FCF 8,600. They differ by wide margins and each answers a different question. Net income asks whether the business was economically profitable. CFO asks whether the operations produced cash. FCF asks how much of that cash the shareholders could actually have.

Run the same test on a company where dividends and buybacks exceed free cash flow year after year and the arithmetic gets uncomfortable fast: the shortfall has to come from somewhere, and the only somewheres are the cash pile, asset sales, or new borrowing. A capital return funded by rising debt is not a return of profits; it is a leveraged recapitalization presented as generosity. **The intuition: free cash flow, not net income and not CFO, is the number that funds dividends and buybacks — and if a company's returns persistently exceed it, look immediately at the financing section to find out who is really paying.**

### Where FCF definitions differ, and why you should care

FCF is not a defined term under GAAP or IFRS. There is no rule for it, which means companies compute it however they like and label the result "free cash flow" in their earnings releases. The common variants:

| Definition | Formula | What it is good for |
| --- | --- | --- |
| Simple / equity FCF | CFO − capex | The standard. What most people mean. |
| FCF to the firm (FCFF) | CFO + after-tax interest − capex | Valuing the whole enterprise, debt and equity together |
| "Adjusted" FCF | CFO − *maintenance* capex only | Arguably better; entirely unverifiable from outside |
| Company-defined FCF | Varies | Read the reconciliation before you use it |

That third row deserves a hard look. Splitting capex into "maintenance" (what you must spend to stand still) and "growth" (what you choose to spend to expand) is economically sound — a company building three new factories is not really consuming 100% of its capex to sustain current earnings. But no accounting standard requires the split, no auditor certifies it, and the classification is entirely management's. A company under pressure to show FCF growth has an obvious incentive to reclassify maintenance spending as growth spending, and no external party can check.

The fourth row is where the real mischief lives. Companies have defined "free cash flow" to exclude restructuring payments, to exclude litigation settlements, to exclude cash spent on acquisitions of intangibles that are functionally capex, and — most commonly — to leave stock-based compensation added back without comment.

### The stock-based compensation argument

Here is the tension in one paragraph. SBC is a non-cash expense, so removing it from net income to compute *cash* flow is arithmetically correct. But the company did pay for that labor; it paid in ownership. If the company then buys back shares to offset the resulting dilution — which most large technology companies do — the buyback appears in the *financing* section, so the cash cost of employee compensation never touches CFO or FCF at all. The employee was paid, the cash went out, and the metric everybody quotes does not see it.

The honest way to handle this as a reader is not to argue about whether the add-back is correct. It is: **look at SBC as a percentage of revenue and as a percentage of FCF, and look at whether buybacks merely offset dilution or actually shrink the share count.** A company whose FCF is 8,600, whose SBC is 2,000, and whose buybacks of 3,500 leave the diluted share count flat has not returned 3,500 to shareholders; it has spent most of it retiring the shares it issued to staff. That is a legitimate business model. It is just not what the buyback headline says.

## Why CFO is the hardest number to fake

Now we can state the central claim properly, with its limits attached.

CFO resists manipulation because of a constraint that no other statement line has: **the cash must eventually exist in a bank account, and an outside party can be asked whether it does.** When auditors perform a *bank confirmation* — writing directly to the financial institution to verify balances — they are testing the anchor that holds the entire cash flow statement in place.

That constraint has a second-order consequence which is more interesting than the first. Consider a company fabricating revenue. It books a fake sale, and net income rises. But a fake sale creates a fake receivable, and a receivable that is never collected sits on the balance sheet aging, quarter after quarter, until it becomes conspicuous. Days sales outstanding climbs. Eventually the receivable must be written off, which reverses the fake profit.

So the fraud has to escalate. To make fake revenue look real, you need fake *cash*, and to produce fake cash you need a fake counterparty who appears to pay. And once money is apparently coming in from somewhere, it has to be seen going out somewhere, or the balance sheet accumulates a cash pile that does not match the business.

This is exactly the shape of the **Luckin Coffee** fraud. In its April 2020 investigation announcement and the SEC's subsequent action, Luckin was found to have fabricated retail sales — and to have manufactured the appearance of the corresponding cash by routing money through related parties and inflating expenses to move the fabricated cash back out. In December 2020 the SEC announced a settled action in which Luckin agreed to pay a **\$180 million** penalty over the fabricated transactions. The forensic lesson: faking revenue is a one-statement problem, but faking *cash* is a three-statement problem, and the extra work leaves fingerprints — implausible expense growth, related-party volume, cash conversion that is suspiciously perfect.

### The limit case: when the bank statement itself is forged

The anchor holds only if someone actually pulls on it. Three of the largest frauds of the last quarter-century worked by attacking the anchor directly.

**Satyam Computer Services (2009).** In a resignation letter dated 7 January 2009 and filed with the SEC, chairman B. Ramalinga Raju admitted that the balance sheet as of 30 September 2008 carried **inflated, non-existent cash and bank balances of ₹5,040 crore** against ₹5,361 crore reflected in the books — roughly a billion US dollars of cash that simply was not there. The company had been fabricating bank statements and fixed-deposit receipts.

**Wirecard (2020).** On 22 June 2020 Wirecard's management board conceded in an ad-hoc disclosure that **€1.9 billion** it had reported as held in trustee accounts in the Philippines most likely did not exist. Philippine banks stated that documents bearing their letterhead were fabrications and that the money had never entered the Philippine financial system. Wirecard filed for insolvency three days later, on 25 June 2020.

**Parmalat (2003)** collapsed after a document purporting to confirm a €3.95 billion Bank of America account turned out to be forged.

Note the common structure. In all three cases the cash flow statement footed perfectly. The three sections summed to the change in cash. The change in cash matched the balance sheet. Everything reconciled — to a cash balance that did not exist. **Internal consistency is not evidence of truth; it is evidence of competence.** A forensic reader treats the reconciliation as a necessary condition, never a sufficient one, and remembers that in every one of these cases the failure was ultimately an audit failure: nobody independently confirmed the balance with the bank.

## The four tricks that actually work

With the honest mechanics in place, we can now look at where the seams are. These are not theoretical. Each has an enforcement action or a public collapse attached.

### Trick 1: classification shifting

This is the purest cash flow statement manipulation because it does not require inventing a single dollar. You take cash that genuinely arrived and put it in the wrong section.

![Two panels comparing an honest presentation with CFO 13,800, CFI −8,800, CFF −1,300 against a shifted presentation with CFO 18,800, CFI −8,800, CFF −6,300; the net change in cash is +3,700 in both.](/imgs/blogs/reading-the-cash-flow-statement-why-cash-beats-net-income-6.webp)

Recall that Northwind drew 5,000 on its revolving credit facility, correctly recorded in financing. Now suppose it structured that borrowing to look like an operating transaction instead — a customer prepayment, say, or a commodity trade with an embedded loan. CFO becomes 18,800; CFF becomes −6,300; investing is unchanged; and the net change in cash is **still +3,700**.

Nothing fails to reconcile. The bank balance is identical. The company's cash conversion ratio, however, has jumped from 1.15 to 1.57, and any screen that ranks companies by operating cash generation has just been fooled completely.

**The real case: Dynegy's Project Alpha.** In April 2001 the energy company Dynegy entered a structured natural-gas transaction internally called Project Alpha. According to the SEC's administrative proceeding of 24 September 2002, Dynegy implemented Alpha to enhance cash flow from operations by **\$300 million** in 2001 and to achieve a \$79 million tax benefit. The economic substance was a loan: Dynegy received cash up front and repaid it through above-market gas purchases later. The SEC found that Dynegy had violated the antifraud, reporting, books-and-records and internal-controls provisions in reflecting Alpha as \$300 million of operating cash flow. Dynegy paid a \$3 million civil penalty, and subsequently restated, reclassifying approximately \$290 million of previously reported 2001 operating cash flow to financing.

**The bigger case: Enron's prepays.** Enron ran the same structure at far greater scale. In a *prepay* transaction, a bank pays cash up front for future delivery of a commodity; the commodity leg is then hedged or circled back through a third party, so what actually happened was a loan with a trade wrapped around it. The Senate Permanent Subcommittee on Investigations found in July 2002 that Enron used prepays to obtain **more than \$8 billion** of financing over roughly six years — including \$3.7 billion across 12 transactions with Chase and \$4.8 billion across 14 transactions with Citigroup — and that the transactions were understood by the participating institutions to be structured so Enron could characterize the funds as cash flow from operations rather than financing, while keeping the corresponding debt off the balance sheet.

**How you detect it.** Look for CFO growth that outpaces revenue growth without a working-capital explanation. Read the operating section for lines that do not sound like operations — "proceeds from," "prepayments received," "structured transactions." Cross-check against the debt footnote: if total debt rose but the financing section shows little borrowing, the borrowing went somewhere else. And treat any single-item CFO swing that neatly closes a guidance gap as a question, not a coincidence.

### Trick 2: selling the receivables

**Factoring** means selling your accounts receivable to a bank or a specialist finance house at a discount, in exchange for cash today. It is ancient, entirely legal, and often sensible — a company with slow-paying customers and an urgent need for working capital may rationally accept a 2% haircut to get paid ninety days early.

The accounting is where it gets interesting. If the sale qualifies as a *true sale* — the risks and rewards genuinely transfer to the buyer — the receivable leaves the balance sheet and the cash arrives inside **operating** activities, as a decrease in receivables. The company has effectively borrowed against its receivables, but the borrowing appears as operating cash generation.

#### Worked example: what factoring does to a quarter

**Ridgeline Industrial is a hypothetical company.** Annual revenue 120,000 (in thousands). At the start of the last week of the fiscal year, accounts receivable stand at 36,000.

$$\text{DSO} = \frac{36{,}000}{120{,}000} \times 365 = 109.5\ \text{days}$$

Management is going to miss its cash flow guidance. So on the second-to-last day of the year it sells 8,000 of receivables to a bank at a 2% discount.

- Cash received: 8,000 × 0.98 = **7,840**
- Receivables after the sale: 36,000 − 8,000 = **28,000**
- The 160 discount is recorded as a financing-related expense in "other income (expense)" on the income statement

The effects:

- **CFO is roughly 7,840 higher** than it would have been, because the receivables balance fell by 8,000
- **New DSO:** 28,000 / 120,000 × 365 = **85.2 days** — a 24-day improvement in a single week, with revenue unchanged
- **Net income** is 160 lower, buried in a line nobody reads
- **Next quarter** starts with 8,000 less of receivables to collect, so unless the company factors again, CFO takes the hit back

That last point is the defining property of every trick in this section: **it borrows from the future.** Factoring does not create cash; it accelerates it, and the acceleration must be repeated at increasing scale to keep producing the same benefit. **The intuition: a one-week 24-day drop in DSO with flat revenue is not an operational improvement — it is a financing transaction wearing an operating costume.**

**The real case: General Electric.** On 9 December 2020 the SEC announced a settled action in which GE agreed to pay a **\$200 million** civil penalty for disclosure violations. Among the findings: GE had increased current-period industrial cash collections in a way it did not adequately disclose was coming at the expense of future years, primarily through internal receivable sales between GE Power and GE Capital. The SEC's order describes GE boosting a publicly reported cash-flow measure by **more than \$1.4 billion in 2016** and **more than \$500 million in the first three quarters of 2017** through this deferred-monetization practice. When GE later wound its factoring programmes down, the reversal was a multi-billion-dollar drag on reported free cash flow — the future paying back what the past had borrowed, exactly as the mechanic predicts.

**How you detect it.** Track DSO quarter by quarter and be suspicious of sharp improvements concentrated at period ends. Search the filing for *"transfers of financial assets"*, *"accounts receivable purchase agreement"*, *"receivables facility"*, *"sales of receivables"*, and *"securitization"*. US GAAP requires disclosure of transfers of financial assets, so the information is generally there — it is just in a footnote thirty pages after the cash flow statement. And check whether the company discloses the *amount* factored: if it does, add it back to receivables and recompute DSO on a like-for-like basis.

### Trick 3: stretching the payables

The mirror image of factoring. Instead of collecting faster, pay slower. Every day you delay paying a supplier is a day their money sits in your bank account, and the working-capital rule says an increase in accounts payable increases CFO.

Stretching payables is the easiest of all these manipulations because it requires no counterparty agreement, no structure, and no accounting judgment. You simply do not press "send" on the payment run until January.

#### Worked example: separating growth from stretch in the payables line

Northwind's payables rose 2,500, and we added all of it to CFO. But how much of that increase was the business growing and how much was the business paying later? Split it.

**Days payable outstanding (DPO)** measures how long the company takes to pay:

$$\text{DPO} = \frac{\text{Accounts payable}}{\text{Cost of revenue}} \times 365$$

- FY2024: cost of revenue 37,000, so daily cost = 37,000 / 365 = **101.37**. DPO = 5,600 / 101.37 = **55.24 days**
- FY2025: cost of revenue 46,000, so daily cost = 46,000 / 365 = **126.03**. DPO = 8,100 / 126.03 = **64.27 days**

Northwind is paying its suppliers **9.03 days later** than it did last year. Now decompose the 2,500 increase:

- **Volume effect** — payables that would have risen anyway because the company buys more: (126.03 − 101.37) × 55.24 days = 24.66 × 55.24 = **1,362**
- **Stretch effect** — payables that rose because payment terms lengthened: 126.03 × 9.03 days = **1,138**
- Total: 1,362 + 1,138 = **2,500** ✓

So of the 2,500 that flattered CFO, about 1,362 is organic and about **1,138 is a one-time benefit from paying suppliers later**. Strip it out and Northwind's CFO is really about 12,662, and its cash conversion ratio falls from 1.15 to 1.06.

And it does not repeat. To get the same 1,138 benefit next year, DPO would have to rise another nine days, to 73.3. The year after, to 82.3. **The intuition: a working-capital benefit from stretching terms is a one-off that must be re-earned at ever-larger scale, which is why companies that start doing it usually cannot stop.**

**The modern version: reverse factoring.** Also called supply-chain finance or an early-payment facility. The company arranges for a bank to pay its suppliers promptly, at a small discount; the company then repays the bank on extended terms, often 120 days or more. Suppliers get paid faster, the company pays later, and the bank earns the spread. Everyone is happy — and the company's obligation, which is now to a bank rather than a supplier, has often been classified as a trade payable rather than as debt.

**The real case: Carillion.** The UK construction and services group collapsed into liquidation in January 2018. Its 2016 balance sheet showed £148 million of bank loans and overdrafts — but the company owed roughly **£498 million** through an "Early Payment Facility," presented as amounts owed to other creditors rather than as borrowings. Moody's and Standard & Poor's both argued the EPF structure meant Carillion had a financial liability to the banks that belonged in borrowings, and UK parliamentary committees concluded the company had used its suppliers to prop up a failing business model. The effect on the cash flow statement was direct: cash that was economically borrowed from banks appeared as operating cash flow retained by stretching trade terms.

**How you detect it.** Compute DPO across several years — a steady multi-year rise is the signature. Compare it with sector peers, since payment terms are industry-conventional. Read the payables footnote for *"supply chain finance"*, *"supplier finance programme"*, *"early payment"*, *"confirming"*, or *"reverse factoring"*.

This last one has become dramatically easier. In September 2022 the FASB issued **ASU 2022-04**, requiring companies to disclose the key terms of supplier finance programmes and the outstanding obligation, effective for fiscal years beginning after 15 December 2022, with an annual rollforward of the obligation effective for fiscal years beginning after 15 December 2023. The IASB followed with amendments to IAS 7 and IFRS 7 effective 1 January 2024. Carillion is the reason those disclosures exist. Use them.

### Trick 4: capitalizing what should be expensed

This is the most powerful of the four, and the reason is worth stating carefully: it is the only one that flatters net income and CFO *simultaneously*.

The choice is this. A cost can be **expensed** — charged in full against this period's income — or **capitalized** — recorded as an asset and depreciated across future periods. Where the cost creates a long-lived asset, capitalizing is correct. Where it does not, it is not.

![Two panels comparing a 3,000 cost expensed — net income impact −3,000, CFO impact −3,000, CFI impact 0 — against the same cost capitalized over ten years — net income impact −300, CFO impact 0, CFI impact −3,000.](/imgs/blogs/reading-the-cash-flow-statement-why-cash-beats-net-income-7.webp)

#### Worked example: the capitalization double win

A company spends 3,000 (in thousands) on something. The cash leaves the building either way. Consider the two treatments, with a ten-year straight-line life if capitalized.

**Expensed:**
- Income statement: −3,000 this year
- Cash flow: −3,000 in **operating**
- Investing: 0

**Capitalized:**
- Income statement: −300 this year (one-tenth of 3,000 in depreciation)
- Cash flow: 0 in operating
- Investing: −3,000 in **capex**

Same 3,000 out of the bank. But capitalizing raises reported net income by **2,700** in year one, *and* raises reported CFO by **3,000**. Both headline numbers improve, and the identity still holds because investing absorbs the full amount.

The cost is deferred, not avoided — years two through ten each carry 300 of depreciation. But depreciation is added back in the CFO walk, so those future years' CFO is unaffected too. From the cash flow statement's point of view, the 3,000 has simply vanished from operations forever. **The intuition: capitalization is the one manipulation that improves earnings and operating cash flow at the same time, which is exactly why it is the one that "just look at cash flow" fails to catch.**

**The real case: WorldCom.** On 25 June 2002 WorldCom disclosed to the SEC and the public that transfers totaling **\$3.852 billion** — \$3.055 billion during 2001 and \$797 million in the first quarter of 2002 — had been made from "line cost" expenses to capital asset accounts, and were not in accordance with GAAP. Line costs were the fees WorldCom paid other carriers to route traffic across their networks: a recurring operating expense with no long-lived asset attached. The SEC's litigation release of June 2002 describes senior management concealing the true extent of line costs by improperly reducing reserves held against them and by transferring line costs to capital asset accounts. WorldCom's own announcement stated that without these transfers it would have reported a net loss for 2001 and for the first quarter of 2002. The restatement ultimately grew far beyond the initial figure, and the company filed the largest bankruptcy in US history at the time.

Sit with the cash flow implication, because it is the most important single fact in this article. WorldCom's fraud moved \$3.852 billion out of operating expenses and into capital expenditures. **That raised reported operating cash flow by the same \$3.852 billion.** An analyst screening for companies whose CFO lagged net income would not have flagged WorldCom. The screen that everyone recommends — and that I have spent this article teaching you — is blind to this specific fraud.

What was not blind to it: free cash flow. Because FCF subtracts capex, the 3,000 that vanished from operations reappears immediately. In the worked example, FCF is −3,000 under both treatments. **Free cash flow is invariant to the operating-versus-investing classification choice**, which is precisely why it is worth computing even though no accounting standard defines it.

**How you detect it.** Track capex as a percentage of revenue over five years and against sector peers — WorldCom's ratio diverged from the industry. Watch for capex growing faster than revenue with no announced expansion programme. Read the accounting-policy note for what the company capitalizes: internal software development, customer acquisition costs, contract costs, cloud implementation costs, and — in software and biotech — development spending. And compute the gap between CFO and FCF: if it is widening while the business is not visibly building anything, ask what is in the capex line.

### A fifth, for completeness: round-tripping and the boundary items

Two smaller seams deserve a mention.

**Round-trip transactions.** Sell something at period end with a simultaneous agreement to buy it back afterwards. The sale generates operating cash; the repurchase happens in the next period. In its complaint filed on 30 October 2006, the SEC alleged that in the fourth quarter of 2000 **Delphi** entered two improper inventory schemes, agreeing to sell approximately **\$270 million** of metals, automotive batteries and generator cores to two third parties at year end while simultaneously agreeing to repurchase the inventory in the following quarter at the original price plus interest and structuring fees. The purpose and result, per the SEC, was to inflate cash flow from operations by **\$200 million**, engineer \$270 million of inventory reductions, and improperly report \$80 million of net income. Economically these were financings; they were accounted for as sales.

**The IFRS classification choices.** Under IAS 7, interest paid may be classified as either operating or financing, and interest and dividends received as either operating or investing. US GAAP is more prescriptive — interest paid and received are operating. This means a European and an American company with identical economics can report materially different CFO, entirely legitimately. It also means an IFRS filer can improve CFO by moving interest paid to financing. The policy must be applied consistently and disclosed, so the fix for a reader is mechanical: find the policy note, and if you are comparing across regimes, normalize by moving interest paid back into operating for everyone.

## How to detect it: the screens that work

Four manipulations, four detection routes. Here is how to run them in order.

### The primary screen: CFO versus net income over five years

Plot the two series side by side. In a healthy business they track each other with CFO usually a little higher, because depreciation is added back and working capital is roughly stable.

![A five-year line chart with net income rising from 100 to 180 while cash from operations falls from 95 to 82, the gap widening in the later years and labelled as the accrual gap.](/imgs/blogs/reading-the-cash-flow-statement-why-cash-beats-net-income-8.webp)

The pattern to look for is divergence, and specifically *widening* divergence. Net income climbing while CFO stagnates or falls means an increasing share of reported profit is not converting into cash. There are innocent explanations — rapid growth genuinely consumes working capital; a shift from prepaid to post-paid billing genuinely reduces deferred revenue — and the innocent explanations are all *visible in the working-capital lines*. That is the test. If the divergence is explained by identifiable, temporary, business-model-consistent working-capital movements, it is probably fine. If it is unexplained or persistent, it is not.

The single-number version is the **cash conversion ratio**, CFO divided by net income. Above 1.0 is comfortable. Persistently below 1.0 while the company is profitable and not growing explosively is the flag. And the closely related **accruals ratio** — net income minus CFO, scaled by average total assets — is the formalization of the same idea, with decades of academic evidence that high-accrual firms underperform.

Two warnings about this screen. First, it does not catch capitalization; WorldCom would have passed. Second, it is noisy quarter to quarter — use annual data over at least three years, and preferably five.

### The four decompositions

When CFO does lag, the next question is *which line*. The answer determines which manipulation, if any, is in play.

![A decision tree rooted at CFO persistently below net income, branching to receivables ballooning, payables ballooning, capex ballooning, and one-off items inside CFO, each with the metric to check and the disclosure to read.](/imgs/blogs/reading-the-cash-flow-statement-why-cash-beats-net-income-9.webp)

| If the drag is in… | Compute | Then read |
| --- | --- | --- |
| Receivables | DSO by quarter, and the sequential change | The transfers-of-financial-assets / receivables-facility footnote |
| Inventory | Days inventory outstanding; inventory growth vs revenue growth | The inventory note and any obsolescence reserve |
| Payables | DPO by year, decomposed into volume and stretch | The supplier-finance-programme disclosure (ASU 2022-04 / IAS 7) |
| Nothing obvious, but CFO looks too good | CFO minus capex; capex as a share of revenue | The capitalization policy note and the PP&E rollforward |

Run the payables decomposition from the worked example above on any company whose DPO is drifting upward: it separates "we buy more" from "we pay later," and only the second is borrowed cash.

### The composite checks

Three further tests catch things the line-by-line review misses.

**CFO minus capex versus dividends plus buybacks.** If the capital return exceeds free cash flow for more than a year or two, find out what is funding it. The answer is in the financing section.

**Cash conversion versus peers.** Working-capital norms are industry-specific — a supermarket collects cash instantly and pays suppliers in 45 days, so it runs negative working capital by design; a defence contractor waits years. Compare within a sector, never across.

**Period-end concentration.** Many of these manipulations only work if executed in the final days of a reporting period. If you can see quarterly data, look for cash flow that is heavily backloaded into the last few weeks, or for receivables and payables that swing sharply at period ends and revert immediately after.

## Common misconceptions

**"Cash flow cannot be manipulated."** The claim that motivates the whole article, and the four cases above are the counter-argument. What is true is narrower and more useful: cash flow cannot be manipulated *without leaving a trace somewhere else* — a rising capex ratio, a footnote about receivable sales, a supplier-finance disclosure, a debt balance that does not match the financing section. Manipulation of the cash flow statement is a displacement problem: the dollar has to come from somewhere and go somewhere.

**"Positive operating cash flow means the company is healthy."** It means operations generated cash this period. A company can have positive CFO and still be consuming cash overall if capex exceeds it — which is exactly the situation of most capital-intensive businesses in a build phase, and it is not automatically bad. It is also possible to have positive CFO produced entirely by shrinking: collecting old receivables, running down inventory, and not replacing it. That is a liquidating business, and its CFO looks great right up until it stops.

**"Free cash flow is a standardized metric."** It is not defined by GAAP or IFRS. Every company computes it its own way, and "adjusted free cash flow" in an earnings release can exclude almost anything. Always find the reconciliation to the nearest GAAP measure, which SEC-registered issuers are required to provide for non-GAAP metrics, and recompute CFO minus capex yourself.

**"Depreciation is added back because it is not a real expense."** Depreciation is added back because it is not a *cash* expense in the current period. It is entirely real — the cash left when the asset was bought, and it will leave again when the asset is replaced. A company whose net income is small but whose EBITDA is large is not thereby a good business; it is a business with heavy fixed assets whose replacement will consume cash. The add-back is a timing correction, not an exoneration.

**"A company with lots of cash on the balance sheet is safe."** The cash balance is a snapshot of one instant and can be borrowed the day before the period ends and repaid the day after — a practice known as *window dressing*. Read the cash flow statement, not the cash line: it tells you where the cash came from, and cash that arrived via the financing section on 30 December is a different animal from cash that operations generated over twelve months. And in the Satyam, Wirecard, and Parmalat cases, the cash line was not merely flattering; it was fictional.

**"If the statement reconciles, the numbers are right."** Reconciliation proves that the preparer is competent at arithmetic. Every major fraud discussed here produced statements that footed perfectly. The reconciliation is a necessary condition and nothing more.

## How it shows up in real markets

**WorldCom, 2001–2002 — the capitalization case.** WorldCom transferred \$3.852 billion of line costs to capital asset accounts across 2001 and the first quarter of 2002, disclosed on 25 June 2002. The mechanism from this article: costs that belonged in operating expense moved to investing, raising both net income and operating cash flow. The lesson is uncomfortable and important — the standard "watch cash flow, not earnings" advice would not have caught the defining fraud of its era. Free cash flow, and a five-year chart of capex as a share of revenue, would have.

**Dynegy, 2001 — the classification case.** Project Alpha delivered \$300 million of what Dynegy reported as operating cash flow in 2001, and \$79 million of tax benefit, from what the SEC characterized as a structured financing. Dynegy paid a \$3 million penalty in September 2002 and reclassified approximately \$290 million to financing. The mechanism: a dollar that genuinely arrived, placed in the wrong bucket. The lesson: when a company's cash generation improves without any corresponding improvement in the working-capital lines, the improvement came from somewhere other than operations.

**Enron, 1997–2001 — the classification case at scale.** More than \$8 billion of prepay financing over roughly six years, per the Senate Permanent Subcommittee on Investigations in July 2002, including \$3.7 billion from Chase and \$4.8 billion from Citigroup, structured so the proceeds could be characterized as operating cash flow rather than debt. The mechanism is identical to Dynegy's; only the scale and the number of participating banks differ. The lesson: this trick requires willing counterparties, which means the disclosure trail runs through the banks as well as the company.

**Delphi, Q4 2000 — the round-trip case.** Approximately \$270 million of metals, batteries and generator cores sold at year end with a simultaneous agreement to repurchase in the next quarter, inflating operating cash flow by \$200 million and net income by \$80 million, per the SEC's complaint filed 30 October 2006. The mechanism: a financing dressed as a sale. The lesson: sharp period-end movements in inventory that reverse immediately afterward are worth investigating, and the repurchase leg is usually visible in the next quarter's numbers.

**General Electric, 2016–2017 — the factoring case.** The SEC's December 2020 order found GE had boosted a publicly reported cash-flow measure by more than \$1.4 billion in 2016 and more than \$500 million in the first three quarters of 2017 through deferred monetization — largely internal receivable sales from GE Power to GE Capital — without adequately disclosing that current collections came at the expense of future years. GE paid a \$200 million penalty. The mechanism: pulling cash forward. The lesson: acceleration is not generation, and the wind-down is as large as the build-up. This one is entirely detectable from DSO plus the receivables footnote.

**Carillion, 2016–2018 — the reverse-factoring case.** Roughly £498 million owed through an Early Payment Facility, presented outside borrowings, against £148 million of disclosed bank loans and overdrafts on the 2016 balance sheet. The company entered liquidation in January 2018. The mechanism: supplier payment terms extended using bank money, with the bank obligation classified as a trade payable. The lesson is the most actionable in this list, because it produced regulation: ASU 2022-04 and the IAS 7 amendments now require the disclosure whose absence made Carillion opaque. When a rule exists because of a specific collapse, the disclosure it mandates is the one worth reading first.

**Satyam (2009), Parmalat (2003), Wirecard (2020) — the anchor cases.** ₹5,040 crore, €3.95 billion, and €1.9 billion of cash that did not exist. The mechanism: forged confirmations from banks that had no such account. The lesson: the cash flow statement's credibility is borrowed from the audit of the cash balance, and where that audit is weak — third-party trustees standing between the auditor and the bank, in Wirecard's case — the entire statement is worth exactly as much as the confirmation behind it.

## When this matters to you

If you own a share of a company, or are thinking about it, three habits follow directly from everything above.

**Read the cash flow statement before the income statement.** It takes ninety seconds to check the three signs, compute CFO minus capex, and compare CFO with net income. That is the highest information-per-second in the entire filing.

**Rebuild one company's operating section by hand, once.** Take a filing you care about, pull the income statement and two balance sheets, and derive the working-capital adjustments yourself as we did with Northwind. It is an hour of arithmetic and it permanently changes how you read every statement afterward, because you will have felt where the numbers come from and therefore where they could be bent.

**When something looks too good, ask which section paid for it.** Cash does not appear. If operating cash flow jumped, either the business improved, or a dollar was moved from investing or financing, or the future was borrowed from. Those three possibilities are distinguishable with the screens in this article, and the distinction is usually the whole investment case.

Two habits for the road. Compare cash conversion within a sector, never across one — working-capital norms are industry facts, not company virtues. And treat every improvement concentrated in the last two weeks of a reporting period as a question rather than an achievement.

The next statements to learn are the two this one checks: [reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings), where the estimates are made, and [reading the balance sheet: what companies hide here](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here), where the consequences accumulate. None of the three makes complete sense alone, which is the point of [how they interlock](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock).

*This article is educational. It explains accounting mechanics and historical enforcement cases; it is not investment advice and does not recommend any security.*

## Sources & further reading

**Primary regulatory and enforcement sources**

- U.S. Securities and Exchange Commission, *Litigation Release No. 17588* — SEC v. WorldCom, Inc. (June 2002), on the \$3.852 billion of line-cost transfers (\$3.055 billion in 2001, \$797 million in Q1 2002): [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-17588)
- U.S. Securities and Exchange Commission, *Administrative Proceeding 33-8134* — In the Matter of Dynegy Inc. (24 September 2002), on Project Alpha's \$300 million of operating cash flow and the \$3 million penalty: [sec.gov](https://www.sec.gov/enforcement-litigation/administrative-proceedings/33-8134)
- U.S. Securities and Exchange Commission, *Litigation Release No. 19891* — SEC v. Delphi Corporation et al. (30 October 2006), on the \$270 million round-trip inventory transactions and \$200 million of inflated operating cash flow: [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-19891)
- U.S. Securities and Exchange Commission, *Press Release 2020-312* — General Electric Agrees to Pay \$200 Million Penalty for Disclosure Violations (9 December 2020), and the accompanying order (Release 33-10899) on deferred monetization of receivables: [sec.gov](https://www.sec.gov/newsroom/press-releases/2020-312)
- U.S. Senate Permanent Subcommittee on Investigations, hearings and staff materials on Enron prepay transactions (July 2002), on the \$8 billion-plus of prepay financing with Chase and Citigroup: [hsgac.senate.gov](https://www.hsgac.senate.gov/wp-content/uploads/imo/media/doc/072302roach.pdf)
- B. Ramalinga Raju, resignation letter to the Satyam board, filed with the SEC as Exhibit 99.2 (7 January 2009), on the ₹5,040 crore of non-existent cash and bank balances: [sec.gov](https://www.sec.gov/Archives/edgar/data/1106056/000114554909000025/u00107exv99w2.htm)

**Accounting standards and disclosure rules**

- FASB *Accounting Standards Update 2022-04*, "Liabilities — Supplier Finance Programs (Subtopic 405-50): Disclosure of Supplier Finance Program Obligations" (September 2022) — effective for fiscal years beginning after 15 December 2022, with the obligation rollforward effective for fiscal years beginning after 15 December 2023
- IASB, *Supplier Finance Arrangements — Amendments to IAS 7 and IFRS 7* (May 2023), effective 1 January 2024
- IAS 7, *Statement of Cash Flows* — the three-section requirement and the operating/financing classification options for interest and dividends
- FASB SFAS 95 (1987), the standard that established the modern three-section US cash flow statement

**Case background and press**

- UK Parliament, Work and Pensions and BEIS Committees, joint inquiry into the collapse of Carillion (2018), on the Early Payment Facility and its presentation outside borrowings: [committees.parliament.uk](https://committees.parliament.uk/committee/164/work-and-pensions-committee/news/97957/carillion-used-its-suppliers-to-prop-up-failing-business-model/)
- Moody's and S&P commentary on Carillion's supply-chain finance classification, summarised in *CFO*, "Carillion Collapse Exposes Flaws in Trade Finance Disclosure" (March 2018): [cfo.com](https://www.cfo.com/news/carillion-collapse-exposes-flaws-in-trade-finance-disclosure/659262/)
- Wirecard AG ad-hoc disclosure of 22 June 2020 conceding that €1.9 billion of trustee balances most likely did not exist; insolvency filing 25 June 2020
- Neal Batson, *Final Report of the Court-Appointed Examiner*, In re Enron Corp. (2003), on the accounting treatment of prepay transactions: [concernedshareholders.com](https://www.concernedshareholders.com/CCS_ENRON_Report.pdf)

**Academic**

- Richard G. Sloan, "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About Future Earnings?", *The Accounting Review* 71(3), 1996 — the foundational evidence that the market over-trusts accrual-heavy earnings relative to cash flow

*Northwind Tools and Ridgeline Industrial are hypothetical companies. Every figure attributed to them is illustrative arithmetic constructed for this article, not the result of any real business. All figures attributed to named companies are sourced above.*
</content>
