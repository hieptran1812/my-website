---
title: "The Three Financial Statements and How They Interlock"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "A beginner-friendly deep dive into what the income statement, balance sheet, and cash flow statement each measure, how a single transaction moves through all three, and why a fake sale leaves a receivables-versus-cash gap that is the fingerprint of revenue fraud."
tags: ["financial-statements", "income-statement", "balance-sheet", "cash-flow-statement", "double-entry", "articulation", "forensic-accounting", "revenue-recognition", "accounts-receivable", "earnings-quality", "financial-statement-analysis", "fraud-detection"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 49
---

> [!important]
> **TL;DR** — The three financial statements are not three reports. They are three views of one ledger, welded together by arithmetic that must tie exactly — and a fake sale cannot satisfy all three at once.
>
> - The **income statement** measures performance over a period, the **balance sheet** is a snapshot of position at one instant, and the **cash flow statement** explains how the cash line moved between two snapshots. Two are movies; one is a photograph.
> - Two links close the loop, and they are exact, not approximate. **Link 1:** net income flows into retained earnings, so `Ending RE = Beginning RE + Net income − Dividends`. **Link 2:** the cash flow statement's ending cash *is* the balance sheet's cash line.
> - Because of double entry, revenue cannot be conjured out of nothing. Every dollar booked as a sale must land somewhere on the balance sheet. If it did not land in cash, it landed in **accounts receivable** — and that is the whole game.
> - A fictitious \$10,000 sale raises net income by \$10,000 and operating cash flow by exactly \$0. Net income climbing while operating cash flow stalls is the **fingerprint** of fake revenue, and it is visible from the outside with no inside information.
> - The number to remember: in fiscal 1997 **Sunbeam Corporation** reported net earnings of **\$109.4 million** and operating cash flow of **negative \$8.2 million** — a gap of \$117.7 million, printed in the same audited annual report. Sunbeam restated in 1998 and filed for bankruptcy in 2001.

In its annual report for fiscal 1997, Sunbeam Corporation reported net earnings of **\$109.4 million**. A few pages later, in the same audited document, it reported that operating activities had *consumed* **\$8.2 million** of cash.

Neither figure was a typo. Both were signed by the auditor. They describe the same twelve months of the same company, and they disagree about whether that company made money by roughly \$118 million.

Only one of them was later restated. The other never had to be — because cash is not a matter of opinion.

That pairing, sitting quietly inside a document anyone could download for free, was the single loudest warning available about one of the most famous accounting frauds of the 1990s. And you did not need a forensic accounting qualification to see it. You needed to know one thing: **the three financial statements are not three independent reports. They are three views of the same ledger, and they are bolted together by arithmetic that has to tie exactly.** Once you understand how they interlock, a lie told on one of them becomes visible on the others — not because a whistleblower talks, but because the arithmetic will not close.

This article builds that frame from nothing. No accounting background is assumed. By the end you will be able to take any transaction — honest or fraudulent — and trace it through all three statements, and you will understand exactly why the most common revenue fraud on Earth leaves a mark it cannot hide.

The diagram below is the mental model for everything that follows. The three boxes are the statements. The two thick arrows are the links. Everything else in this article is detail hanging off those two arrows.

![The articulation map: the income statement produces net income, which flows into retained earnings on the balance sheet and simultaneously starts the cash flow statement, whose ending cash must equal the balance sheet's cash line.](/imgs/blogs/the-three-financial-statements-and-how-they-interlock-1.webp)

Read it as a circuit. The income statement runs for a period and spits out one number, **net income**. That number does two jobs at once: it flows into **retained earnings** on the balance sheet (Link 1), and it is simultaneously the *first line* of the cash flow statement (Link 2's starting point). The cash flow statement then grinds net income down into the cash that actually moved, adds up the three cash categories, and produces an **ending cash** figure that must equal — to the dollar — the cash line on the balance sheet. Close the loop, and the balance sheet balances. Fail to close it, and something is wrong.

Accountants call this property **articulation**. It is the reason financial statements are hard to fake convincingly, and it is the foundation the entire discipline of forensic accounting stands on.

## Foundations: the building blocks of the three statements

Before we can trace anything through the system, we need to know what each statement is and what question it answers. This section assumes zero prior knowledge and defines every term on first use. A practitioner can skim it. A beginner should not.

### The income statement: did we perform?

The **income statement** (also called the *profit and loss statement*, the *P&L*, or the *statement of operations*) answers one question: **over a stretch of time, did the business create more value than it consumed?**

It runs top to bottom, starting with the money the business earned from customers and subtracting the costs of earning it:

| Line | What it means |
| --- | --- |
| **Revenue** (or *net sales*) | The value of goods and services delivered to customers this period |
| − **Cost of goods sold (COGS)** | The direct cost of the specific things that were sold |
| = **Gross profit** | What is left to cover everything else |
| − **Operating expenses** (SG&A, R&D) | Salaries, rent, marketing, admin — the cost of running the business |
| = **Operating income** | Profit from the core business, before financing and tax |
| − **Interest expense** | The cost of borrowed money |
| − **Income tax** | The government's share |
| = **Net income** | The bottom line. The number in the headline |

The critical word in the revenue line is **delivered**, not *paid*. Under **accrual accounting** — the system every public company on Earth is required to use — revenue is recorded when it is *earned*, meaning when you have delivered what you promised, regardless of whether the customer has paid you yet. Costs are recorded in the same period as the revenue they helped produce, regardless of when the cheque cleared. (If that distinction is new to you, the companion piece on [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) builds it from scratch; this article takes it as given.)

That single design decision is why the income statement is useful *and* why it is manipulable. It is useful because it tells you when the business actually did business, rather than when envelopes were opened. It is manipulable because "earned" is a judgment, and judgments have room in them.

### The balance sheet: what do we own and what do we owe?

The **balance sheet** (or *statement of financial position*) answers a completely different question: **at one specific instant, what does the business own, what does it owe, and what is left over for the owners?**

It has three sections, and they are related by the most important equation in accounting:

$$\text{Assets} = \text{Liabilities} + \text{Equity}$$

- **Assets** are resources the business controls that are expected to produce future benefit: cash, inventory, accounts receivable (money customers owe you), buildings, machines, patents.
- **Liabilities** are obligations to others: accounts payable (money you owe suppliers), loans, bonds, accrued wages, taxes due.
- **Equity** is the residual — what the owners would have left if every asset were converted to cash and every liability settled. It is not a pot of money; it is a subtraction. Its two main components are **paid-in capital** (money investors put in) and **retained earnings** (profits the business earned and did not pay out).

The equation is not a rule someone imposed. It is a tautology. Everything the business has (the left side) came from somewhere: either a lender gave it, or an owner gave it, or the business earned it. There is no third source. That is why the balance sheet *balances*, and why it is called a balance sheet.

Note the tense. The income statement covers a *period*: "for the year ended December 31." The balance sheet is dated a single *day*: "as of December 31." One is a movie of what happened. The other is a photograph of where things stand when the music stops.

### The cash flow statement: where did the cash actually go?

The **cash flow statement** answers the checkbook question: **over the same period, how much cash actually moved, and what moved it?**

It splits every cash movement into three buckets:

| Bucket | Abbreviation | What goes in it |
| --- | --- | --- |
| **Operating activities** | **CFO** | Cash from running the core business: collecting from customers, paying suppliers, wages, tax, interest |
| **Investing activities** | **CFI** | Buying and selling long-lived things: equipment, buildings, acquisitions, securities |
| **Financing activities** | **CFF** | Dealing with the people who funded you: borrowing, repaying debt, issuing shares, buybacks, dividends |

Add the three together and you get the **net change in cash** for the period. Add that to the cash you started with and you get the cash you ended with.

$$\text{Cash}_{\text{end}} = \text{Cash}_{\text{begin}} + \text{CFO} + \text{CFI} + \text{CFF}$$

This statement exists because of a hard lesson learned repeatedly: **profit is an opinion, cash is a fact.** A company can be profitable and still fail, because obligations are settled in cash, not in net income. The cash flow statement was made mandatory in the United States by the Financial Accounting Standards Board in 1987 (FASB Statement No. 95), replacing a vaguer "statement of changes in financial position." It is the youngest of the three statements, and it exists precisely because the other two, on their own, let too many failing companies look healthy.

### Flow versus stock: two movies and a photograph

The single most common source of beginner confusion is mixing up **flows** and **stocks**.

A **flow** is a quantity measured *over an interval*: revenue for the quarter, cash used in operations for the year, calories eaten today. A **stock** (in this sense, nothing to do with equities) is a quantity measured *at an instant*: cash on hand right now, inventory in the warehouse this morning, your body weight when you step on the scale.

The income statement and the cash flow statement are flow statements. The balance sheet is a stock statement. And the relationship between them is the relationship between a bathtub's water level and its taps:

- The **balance sheet** is the water level in the tub at a given moment.
- The **income statement** and **cash flow statement** are the taps and the drain — how fast water went in and out over the interval between two readings.

This is why every flow statement, if it is honest, must exactly explain the change between two consecutive balance sheets. The change in the water level *is* the net flow. There is no third possibility. Hold onto that, because it is the mechanism that traps the fraudster.

![A three-column comparison of what each statement measures: the question it answers, its time shape, whether it is accrual or cash based, and how hard it is to manipulate.](/imgs/blogs/the-three-financial-statements-and-how-they-interlock-2.webp)

The last row of that figure is the one worth memorizing. The income statement is the **easiest** statement to manipulate, because almost every line on it involves an estimate: when was this revenue earned? How much of this receivable will we collect? Over how many years does this machine wear out? The cash flow statement is the **hardest**, because the operating section is anchored to a bank balance that a third party — the bank — independently confirms. The balance sheet sits in between: some lines (cash, debt) are externally confirmable, others (goodwill, inventory valuation, receivable allowances) are pure judgment.

Fraud flows downhill, toward the easy statement. That is why the great majority of enforcement actions involve revenue recognition, and why the gap between the easy statement and the hard one is where you look.

### Double entry: the reason the balance sheet cannot break by accident

Now the mechanism. Every transaction a company records touches **at least two accounts**, in equal and opposite amounts. This is **double-entry bookkeeping**, and it has been the standard since a Franciscan friar named Luca Pacioli wrote it down in 1494.

The rule, stated plainly: for every entry, the accounting equation must still hold afterward. If an asset goes up, then either another asset goes down by the same amount, or a liability goes up, or equity goes up. There is no way to increase one side alone.

That is not a bureaucratic formality. It is a **conservation law**, and it is the reason financial statements can be interrogated at all. In physics, you cannot create energy; the books must balance, so if energy appears here it must have left there. In accounting, you cannot create value; if an asset appears here, it must be matched somewhere.

Which leads directly to the sentence this entire article is built around:

> **You cannot book revenue without creating an asset or destroying a liability. If the revenue did not arrive as cash, it must be sitting on the balance sheet as something else — and that something else is nearly always accounts receivable.**

The fraudster does not get to choose whether the other side exists. Double entry chooses for them. All they get to choose is *where it goes*, and every choice leaves a different, detectable mark.

![Before and after a \$10,000 credit sale on \$6,000 of goods: assets rise from \$200,000 to \$204,000, equity rises from \$120,000 to \$124,000, and both sides of the accounting equation still match.](/imgs/blogs/the-three-financial-statements-and-how-they-interlock-3.webp)

#### Worked example: one credit sale, four ledger lines

Let us make double entry concrete with a company we will use throughout. **Northwind Tools** is a hypothetical hardware distributor — every Northwind figure in this article is an illustrative example, invented to make the arithmetic clean, not a real company's results.

Northwind's balance sheet on December 31 of Year 0 looks like this (all figures in dollars):

| Assets | | Liabilities & Equity | |
| --- | ---: | --- | ---: |
| Cash | 50,000 | Accounts payable | 25,000 |
| Accounts receivable | 30,000 | Long-term debt | 55,000 |
| Inventory | 40,000 | **Total liabilities** | **80,000** |
| Property & equipment, net | 80,000 | Common stock | 60,000 |
| | | Retained earnings | 60,000 |
| | | **Total equity** | **120,000** |
| **Total assets** | **200,000** | **Total liabilities & equity** | **200,000** |

Both sides come to \$200,000. Good.

Now Northwind sells a pallet of tools for **\$10,000** on 60-day credit. Those tools cost Northwind **\$6,000** and are sitting in the warehouse. We will ignore tax for this single transaction to keep the arithmetic visible.

Four ledger entries fire, in two pairs:

1. **Accounts receivable +\$10,000** (an asset appears: the customer's promise to pay)
2. **Revenue +\$10,000** (which flows to net income, and therefore to equity)
3. **Inventory −\$6,000** (an asset disappears: the tools left the warehouse)
4. **Cost of goods sold +\$6,000** (an expense, which reduces net income and therefore equity)

Net effect on assets: +10,000 − 6,000 = **+\$4,000**.
Net effect on equity: +10,000 − 6,000 = **+\$4,000** of retained earnings.
Liabilities: unchanged at \$80,000.

New totals: assets \$204,000; liabilities plus equity \$80,000 + \$124,000 = \$204,000. Still balanced, exactly as the figure above shows.

**The intuition: a sale is not one event but a swap plus a gain — \$6,000 of goods was exchanged for a \$10,000 claim, and the \$4,000 difference is the profit that lands in equity.**

Notice what did *not* happen: **cash did not move.** Not one dollar. Northwind is \$4,000 richer on the income statement and has exactly as much money in the bank as it did this morning. That is not fraud, and nothing is wrong. It is simply what accrual accounting does, and it is the seed of everything that follows.

## 1. The two links that lock the statements together

We now nail down the two connections that make the three statements a single system. They are exact identities, not approximations, and you can check both of them on any real annual report in about ninety seconds.

### Link 1: net income becomes retained earnings

The income statement's bottom line does not evaporate at the end of the period. It flows into equity, into the specific bucket called **retained earnings** — the cumulative pile of every profit the company has ever earned and not distributed to shareholders.

The identity, called the **retained earnings roll-forward**:

$$RE_{\text{end}} = RE_{\text{begin}} + \text{Net income} - \text{Dividends}$$

That is the whole of Link 1. Start with last period's retained earnings, add this period's profit, subtract anything paid out to shareholders, and you must land exactly on this period's retained earnings.

It is worth pausing on why this must be true. Retained earnings is *defined* as accumulated undistributed profit. If the company earned \$100 and paid out \$30, then \$70 more profit has been retained than before. There is nowhere else for it to go. Any company whose retained earnings do not roll forward is either reporting something unusual (a restatement of prior periods, certain other-comprehensive-income items, a share buyback charged to retained earnings) and telling you about it in the notes, or has an error.

#### Worked example: Sunbeam's 1997 retained earnings tie exactly

Here is the identity on a real company, using figures straight from Sunbeam Corporation's Form 10-K for the fiscal year ended December 28, 1997, as filed with the SEC (figures in thousands of dollars):

- Retained earnings at December 29, 1996: **\$35,118**
- Net earnings for fiscal 1997: **\$109,415**
- Dividends paid on common stock in 1997: **\$3,399**
- Retained earnings at December 28, 1997: **\$141,134**

Run the roll-forward:

\$35,118 + \$109,415 − \$3,399 = **\$141,134**

That is not a rounding-close match. It is exact to the last thousand dollars, and it comes from three different statements — the equity section of the balance sheet, the bottom of the income statement, and the financing section of the cash flow statement. Sunbeam was, as we will see, running a substantial accounting fraud that year. Link 1 still tied perfectly.

**The intuition: articulation being intact tells you the bookkeeping is internally consistent. It tells you nothing whatsoever about whether the underlying transactions were real.** Fraud does not break double entry. Fraud uses double entry, correctly, on transactions that did not happen.

### Link 2: ending cash ties back to the balance sheet

The second link closes the loop. The cash flow statement's final line — cash at the end of the period — must equal the cash line at the top of the balance sheet on the same date.

$$\text{Cash}_{\text{end}} = \text{Cash}_{\text{begin}} + \text{CFO} + \text{CFI} + \text{CFF}$$

And `Cash_begin` must equal the cash line on the *previous* balance sheet. The cash flow statement is, quite literally, an explanation of the change in one balance sheet line, categorized by cause.

#### Worked example: Sunbeam's 1997 cash tie-out

Same document, same fiscal year, figures in thousands:

| Line | Amount |
| --- | ---: |
| Cash and cash equivalents at beginning of year | 11,526 |
| Net cash used in operating activities (CFO) | (8,249) |
| Net cash provided by investing activities (CFI) | 32,724 |
| Net cash provided by financing activities (CFF) | 16,377 |
| Net increase in cash | 40,852 |
| **Cash and cash equivalents at end of year** | **52,378** |

Check it: \$11,526 − \$8,249 + \$32,724 + \$16,377 = \$52,378. And \$52,378 is precisely the "Cash and cash equivalents" line at the top of Sunbeam's December 28, 1997 balance sheet.

Two things are worth extracting from this table before we move on, because they preview the rest of the article.

First, **Sunbeam's cash went up \$40.9 million in a year when its operations consumed cash.** Every dollar of that increase came from investing (\$32.7 million, mostly \$91.0 million of proceeds from selling divested operations, offset by \$58.3 million of capital spending) and financing (\$16.4 million of borrowing and stock option exercises). The core business was a net drain. A reader who looked only at the balance sheet's cash line, or only at the headline profit, would have seen a company whose cash pile more than quadrupled. A reader who looked at the *composition* saw a company selling assets and borrowing to fund a business that was not generating cash.

Second, the negative CFO figure was itself flattered. Sunbeam's own MD&A discloses that cash used in operating activities "reflects \$59 million of proceeds from the sale of trade accounts receivable under the Company's revolving trade accounts receivable securitization program entered into in December 1997." Selling your receivables converts a future operating collection into cash *today* and books it in CFO. Strip that December-1997 program out and Sunbeam's 1997 operating cash flow was closer to **negative \$67 million**.

**The intuition: the cash flow statement is the only statement whose bottom line is confirmed by an outside party. That is what makes it the anchor — and it is why sophisticated manipulation aims at moving items *between* CFO, CFI, and CFF rather than at the cash total itself.**

### Why this is called a model with no plug

If you have ever built a "three-statement model" in a spreadsheet, you have met articulation from the other side. You forecast revenue and costs, which give you net income. Net income drives retained earnings. Working-capital assumptions drive receivables, inventory, and payables. Capital spending drives property and equipment. Debt schedules drive interest. All of it feeds the cash flow statement, which produces ending cash, which goes back onto the balance sheet.

And then you check whether assets equal liabilities plus equity. If they do not, you have made an error — you have double-counted something or dropped something. A well-built model **has no plug**: no fudge line inserted to force the balance. The balance emerging on its own is the proof that every flow was routed to exactly one destination.

The same logic runs in reverse for the analyst. If you are handed three statements and they articulate, you know the preparer routed every flow somewhere. Your job then is not to check the arithmetic — the arithmetic will be fine — but to ask whether the *transactions underneath* were real, and whether the estimates were honest.

## 2. Following one honest sale through all three statements

Now we do the thing this article exists to teach: take a single transaction and watch it appear on all three statements at once. Start with an honest one, so the fraudulent version has something to be compared against.

Back to Northwind Tools and the \$10,000 credit sale of goods that cost \$6,000. Tax ignored.

![One honest \$10,000 credit sale traced across all three statements: net income rises \$4,000, receivables rise \$10,000, inventory falls \$6,000, and operating cash flow is unchanged at zero.](/imgs/blogs/the-three-financial-statements-and-how-they-interlock-4.webp)

#### Worked example: the \$10,000 credit sale, statement by statement

**On the income statement:**

| Line | Change |
| --- | ---: |
| Revenue | +10,000 |
| Cost of goods sold | (6,000) |
| **Net income** | **+4,000** |

**On the balance sheet:**

| Line | Change |
| --- | ---: |
| Cash | 0 |
| Accounts receivable | +10,000 |
| Inventory | (6,000) |
| **Total assets** | **+4,000** |
| Retained earnings (equity) | +4,000 |
| **Total liabilities & equity** | **+4,000** |

**On the cash flow statement** (using the indirect method, explained in a moment):

| Line | Change |
| --- | ---: |
| Net income | +4,000 |
| Less: increase in accounts receivable | (10,000) |
| Add: decrease in inventory | +6,000 |
| **Cash flow from operations** | **0** |

Every statement is telling the truth, and the three truths look completely different. The income statement says Northwind made \$4,000. The balance sheet says Northwind is \$4,000 richer, but not in cash. The cash flow statement says Northwind's operations produced nothing at all this period.

Look closely at how the cash flow statement got to zero. It started with net income of \$4,000 — the accrual answer. Then it *undid* the two accrual judgments embedded in that number. Revenue of \$10,000 was recognized but not collected, so the \$10,000 increase in receivables is subtracted. Inventory of \$6,000 was expensed but paid for in a previous period, so the \$6,000 decrease in inventory is added back. What remains is the cash answer: zero.

**The intuition: the cash flow statement is a machine for reversing accrual judgments. Feed it net income and every balance-sheet change, and it hands back the number that a bank statement would show.**

#### Worked example: the customer pays 60 days later

Sixty days pass. The customer sends \$10,000.

**Income statement:** nothing at all. Zero revenue, zero expense. The revenue was already recognized when the goods shipped; recognizing it again would be double counting.

**Balance sheet:** cash +\$10,000, accounts receivable −\$10,000. Total assets unchanged. Equity unchanged. Nothing but a swap of one asset for another.

**Cash flow statement:** net income \$0, plus a \$10,000 *decrease* in receivables, giving CFO of **+\$10,000**.

So across the two periods combined: net income \$4,000, cumulative operating cash flow \$10,000 − \$6,000 (the cash Northwind paid for the inventory in an earlier period) = \$4,000. **The two measures converge.** They always do, eventually — over the full life of a business, total net income equals total operating cash flow minus the cash invested and returned. Accrual accounting changes the *timing*, never the *destination*.

**The intuition: net income and cash flow are the same journey measured on different clocks. They separate in the short run and reconcile in the long run — which is precisely why a persistent, growing gap is informative.**

### Why net income and cash diverge — the honest reasons

Before treating every gap as suspicious, it is worth cataloguing the many completely legitimate reasons net income and CFO differ. A forensic reader who cannot distinguish an innocent gap from a guilty one is just an alarmist.

**Non-cash expenses inflate the gap in the healthy direction.** **Depreciation** (spreading the cost of a machine over its useful life) and **amortization** (the same for intangibles like software or patents) are recorded as expenses but involve no cash leaving the building this period — the cash left when the asset was bought. Stock-based compensation is the same: a real cost to shareholders, no cash out the door. All of these are added back on the cash flow statement, so a capital-intensive company routinely reports CFO well *above* net income. That is healthy.

**Growth in working capital consumes cash.** A company growing fast has to fund more receivables (customers owe it more) and more inventory (it stocks more goods) before the sales convert. Fast growth genuinely eats cash. This is why a good business can go bankrupt while growing — a condition sometimes called *overtrading*.

**Timing of payables works the other way.** Stretching supplier payments — paying in 90 days instead of 45 — is a cash inflow in the period you stretch. It flatters CFO once and then stops helping, and it is a favorite of companies under pressure.

**Seasonality.** A retailer builds inventory before the holidays and collects after. Any single quarter's CFO can be wildly unrepresentative.

The discipline, then, is not "CFO below net income equals fraud." It is: **over a full business cycle, for a company that is not investing heavily in growth, cumulative CFO should be at least as large as cumulative net income.** When it is persistently and increasingly smaller, and the shortfall is concentrated in receivables, you are looking at something that needs an explanation.

### The indirect method: the bridge from net income to CFO

Almost every company you will read uses the **indirect method** for the operating section: start at net income, then adjust. (A **direct method** exists — listing actual cash received from customers and paid to suppliers — and is far more readable, which may be why almost nobody uses it.)

The indirect bridge has three kinds of adjustment:

1. **Add back non-cash expenses**: depreciation, amortization, stock-based compensation, non-cash impairments, deferred taxes.
2. **Reverse gains and losses that belong elsewhere**: a gain on selling a building is real profit, but the cash from that sale belongs in investing, so the gain is subtracted from CFO to avoid counting it twice.
3. **Adjust for changes in working capital**: an *increase* in an operating asset (receivables, inventory, prepaid expenses) is a *use* of cash and is subtracted; an *increase* in an operating liability (payables, accrued expenses, deferred revenue) is a *source* of cash and is added.

That third rule is the one beginners find backwards, so here is the way to hold it. A receivable going up means you delivered goods and did not get paid — you funded your customer. Money you have lent out is money you do not have. Subtract it. A payable going up means you took goods and did not pay — your supplier funded you. Subtract nothing; add it.

![A left-to-right bridge from Northwind's reported net income of \$99,000 down to operating cash flow of \$43,000, with the \$62,400 increase in receivables as the dominant negative bar.](/imgs/blogs/the-three-financial-statements-and-how-they-interlock-5.webp)

That bridge is Northwind's Year 2 — a year we have not described yet, and which we will spend the next two sections dissecting. Note the shape of it now: one bar dwarfs all the others, and it is the receivables bar. When a bridge looks like this, the story of the year is written in that one bar.

## 3. The fake sale: one fictitious entry, three statements

Here is the most common accounting fraud in the world, in one sentence: **record revenue for a sale that did not happen, or has not happened yet.**

The Securities and Exchange Commission has consistently found improper revenue recognition to be the largest single category of financial reporting enforcement. It is the default fraud because revenue is the top line — it drives the stock price, the analyst estimate, the executive bonus, and the debt covenant — and because "when was this earned?" is genuinely a judgment call, which gives the fraudster a story to tell.

The forms it takes are a spectrum, from aggressive to criminal:

| Technique | What it is | Where it sits |
| --- | --- | --- |
| **Channel stuffing** | Shipping more product to distributors than they can sell, with discounts or extended terms, to pull next period's sales forward | Aggressive to fraudulent |
| **Bill and hold** | Invoicing a customer for goods that stay in your own warehouse, claiming the sale is complete | Fraudulent unless very narrow conditions are met |
| **Premature recognition** | Booking a multi-year contract's full value on signature rather than over the delivery period | Usually improper |
| **Round-tripping** | Two companies buy from each other in matched transactions, inflating both top lines with no economic substance | Fraudulent |
| **Wholly fictitious sales** | Invoices to customers who do not exist, or for goods never shipped | Criminal |

They differ enormously in culpability. They are near-identical in their statement fingerprint, because they all do the same thing: **create revenue without creating cash.**

![The same three-panel trace for a fictitious \$10,000 sale: net income rises \$10,000, receivables rise \$10,000, and operating cash flow does not move at all.](/imgs/blogs/the-three-financial-statements-and-how-they-interlock-6.webp)

#### Worked example: booking \$10,000 of revenue that never happened

Northwind's quarter is coming up \$10,000 short of the number management promised. Someone raises an invoice for \$10,000 to a customer who did not order anything and to whom nothing was shipped. Tax ignored, as before.

**On the income statement:**

| Line | Change |
| --- | ---: |
| Revenue | +10,000 |
| Cost of goods sold | 0 |
| **Net income** | **+10,000** |

**On the balance sheet:**

| Line | Change |
| --- | ---: |
| Cash | 0 |
| Accounts receivable | +10,000 |
| Inventory | 0 |
| **Total assets** | **+10,000** |
| Retained earnings (equity) | +10,000 |

**On the cash flow statement:**

| Line | Change |
| --- | ---: |
| Net income | +10,000 |
| Less: increase in accounts receivable | (10,000) |
| **Cash flow from operations** | **0** |

Read those three tables side by side and the entire discipline of forensic accounting comes into focus.

**The income statement moved by \$10,000. The cash flow statement did not move at all.** The very adjustment that makes the cash flow statement honest — subtracting the increase in receivables — mechanically cancels the fake revenue out. The fraud is invisible on the income statement and self-erasing on the cash flow statement.

That cancellation is not luck, and it is not a loophole the fraudster failed to close. **It is a consequence of double entry, and it is unavoidable.** The fake revenue had to have a matching entry. Cash was impossible, because no cash arrived and the bank confirms the balance. So the other side went to receivables. And an increase in receivables is exactly what the cash flow statement subtracts.

**The intuition: fake revenue can raise your profit or your operating cash flow, but not both — because the entry that raises profit is the same entry the cash flow statement reverses.**

### Where else could the other side go?

A determined fraudster might ask: must it be receivables? Let us take the alternatives seriously, because each one is a real technique and each one leaves a different mark.

**Put it in cash.** This requires actual money to arrive. It is not impossible — you can send your own money out through a related party and have it come back as a customer payment — but it demands a *source* of cash, which itself has to be recorded, and it costs real money in tax and transaction friction. Luckin Coffee, discussed later, did roughly this. The tell moves to the expense side and the cash source.

**Put it in inventory.** Instead of leaving inventory alone (which produces the suspicious 100% gross margin above), relieve \$6,000 of inventory as COGS to make the margin look normal. Now net income rises only \$4,000, and:

| Line | Change |
| --- | ---: |
| Accounts receivable | +10,000 |
| Inventory (per the books) | (6,000) |
| Inventory (actually in the warehouse) | 0 |

The goods are still physically on the shelf. The books say they left. That is a **\$6,000 inventory shortfall** that a physical count will find, and physical inventory counts are a standard audit procedure. This is why fully fictitious sales are usually booked at 100% margin, and why an unexplained rise in gross margin alongside a rise in receivables is such a strong pair of signals.

**Reduce a liability.** Recognizing revenue by drawing down **deferred revenue** — money customers paid in advance for goods not yet delivered — is real revenue recognition when delivery occurs, and premature recognition when it does not. This one does *not* create a receivable and does *not* open a CFO gap, because the cash came in earlier. It is genuinely harder to spot from the outside. The tell is deferred revenue falling while bookings are supposedly strong.

So the receivables route is not the only one. It is simply the cheapest, the fastest, and the one that requires no accomplice — which is why it is overwhelmingly the most common, and why the receivables-versus-cash gap is the highest-yield single test an outside reader has.

### Why the fraud has to escalate

Here is the property that turns a detectable pattern into an inevitable collapse.

The fake receivable never collects. There is no customer, so no cash ever arrives. It sits on the balance sheet ageing. Eventually, accounting rules require the company to reserve against uncollectible receivables — a **bad debt expense** that reverses the fake profit. And that reversal lands in a future period, on top of whatever that future period's real shortfall is.

So the fraudster faces a compounding problem. To hide period 1's \$10,000 hole, they must book \$10,000 of fake revenue. In period 2, they need enough fake revenue to cover period 2's real shortfall *and* to avoid writing off period 1's receivable. The required lie grows every period.

This is why accounting frauds do not fade away quietly. They end in a restatement, a bankruptcy, or a confession — and they leave, in the years before the end, a signature of net income and operating cash flow diverging on a widening curve.

![A two-line chart over eight quarters: net income climbs steadily from \$12,000 to \$32,000 while operating cash flow peaks at \$16,000 and falls away to \$6,000, opening a widening accrual gap.](/imgs/blogs/the-three-financial-statements-and-how-they-interlock-7.webp)

That widening wedge is the shape you are looking for. Not one bad quarter — every business has those — but a trend in which the two lines, which should travel together, begin to travel apart and never reconverge.

## 4. Detecting it: four tests you can run from the outside

We now turn the frame into a procedure. All four tests use only figures from published statements. None requires access to a company's systems. Together they take about ten minutes per company.

To run them on something concrete, here is Northwind Tools' Year 1 and Year 2, side by side. Year 2 is presented two ways: as it *would* have looked honestly, and as it was *reported* after management inserted \$60,000 of fictitious sales at 100% margin. Northwind pays tax at 25%, and — this matters — it pays that tax in cash, on the fake profit as well as the real profit.

| | Year 1 | Year 2 (honest) | Year 2 (as reported) |
| --- | ---: | ---: | ---: |
| Revenue | 500,000 | 540,000 | 600,000 |
| Pre-tax income | 66,667 | 72,000 | 132,000 |
| Income tax at 25% | 16,667 | 18,000 | 33,000 |
| **Net income** | **50,000** | **54,000** | **99,000** |
| Accounts receivable, year-end | 30,000 | 32,400 | 92,400 |
| **Cash flow from operations** | **55,000** | **58,000** | **43,000** |

The fake sales add \$60,000 to revenue and \$60,000 to receivables. They add \$45,000 to net income (\$60,000 less \$15,000 of extra tax). And they *reduce* operating cash flow by exactly \$15,000 — the cash tax paid on profit that does not exist.

That last point is worth its own sentence, because it is counterintuitive and it is the fraudster's cruellest trap. **Fake revenue does not merely fail to generate cash. It actively burns cash, because the tax authority collects real money on imaginary profit.** A company running a revenue fraud is paying to keep the lie alive.

#### Worked example: running all four tests on Northwind's Year 2

![Four diagnostic tests with a healthy reading and an alarm reading each: CFO divided by net income, receivables growth versus revenue growth, days sales outstanding, and accruals.](/imgs/blogs/the-three-financial-statements-and-how-they-interlock-8.webp)

**Test 1 — Operating cash flow divided by net income.** This is the blunt instrument, and it is usually enough.

- Year 1: \$55,000 ÷ \$50,000 = **1.10**
- Year 2 as reported: \$43,000 ÷ \$99,000 = **0.43**

A healthy, mature business converts profit to cash at a ratio at or above 1.0 averaged over a cycle. Northwind's conversion collapsed by more than half while its reported profit doubled. Sustained readings well below 1.0, especially while earnings accelerate, are the primary screen.

*What can innocently cause a low reading:* rapid genuine growth funding working capital; a large one-off legal settlement paid in cash; a young company scaling. Check the trend and the composition before concluding anything.

**Test 2 — Receivables growth versus revenue growth.** Test 1 tells you there is a gap. Test 2 tells you where the gap lives.

- Honest Year 2: revenue +8%, receivables +8%. They move together.
- Reported Year 2: revenue **+20%**, receivables **+208%**.

Receivables should grow roughly in line with sales, because they *are* sales that have not been collected yet. Receivables growing at ten times the rate of sales means one of three things: the company's customers have suddenly stopped paying, the company has drastically loosened its credit terms to buy revenue, or some of those sales are not real. All three are bad news; only the third is a crime.

**Test 3 — Days sales outstanding.** The same information as Test 2, expressed in a unit that is easier to reason about and easier to compare across companies.

$$\text{DSO} = \frac{\text{Accounts receivable}}{\text{Revenue}} \times 365$$

- Year 1: (30,000 ÷ 500,000) × 365 = **21.9 days**
- Year 2 as reported: (92,400 ÷ 600,000) × 365 = **56.2 days**

DSO answers: on average, how long between making a sale and collecting the cash? Northwind's collection cycle went from three weeks to eight weeks in a single year, with no change in its customer base or its stated terms. DSO is powerful because it is scale-free — you can compare it to the company's own history, to its competitors, and to its own stated payment terms. A company that tells investors it sells on 30-day terms and reports 56-day DSO is telling you something, whether it means to or not.

**Test 4 — Accruals.** The summary statistic, and the bridge to the rest of the forensic toolkit.

$$\text{Accruals} = \text{Net income} - \text{Cash flow from operations}$$

- Year 1: \$50,000 − \$55,000 = **−\$5,000**
- Year 2 as reported: \$99,000 − \$43,000 = **+\$56,000**

The accrual is the part of reported profit that is not cash — the judgment layer. Small or negative accruals mean earnings are backed by money. Large positive accruals mean earnings are backed by estimates. Scaled by Northwind's roughly \$200,000 of total assets, Year 2's accrual is about **28% of assets**, which is extreme by any standard.

This measure generalizes: dividing accruals by average total assets gives the **accruals ratio**, one of the most studied variables in accounting research and the input to composite screens like the Beneish M-Score. A fuller treatment lives in [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits).

**The intuition: all four tests are the same test. They each ask whether reported profit is backed by money, and they differ only in how they scale the question.**

### What these tests do not catch

Honesty about a tool's blind spots is what separates a method from a superstition. The receivables-versus-cash gap is blind to at least five important things.

**Capitalized expenses.** If a company moves an operating cost onto the balance sheet as an asset, net income rises *and* operating cash flow rises, because the cash payment relocates from operating to investing. Both statements improve together, and the gap never opens. This is the WorldCom pattern, examined below, and it is the single most important exception to everything in this article.

**Fabricated cash.** If the fraud is on the balance sheet — a cash balance that does not exist — then CFO looks magnificent, because the fake cash was reported as having come from operations. This is the Wirecard pattern.

**Funded fake revenue.** If the fraudster routes real money out through fake expenses and back in as fake customer payments, the receivable never opens and the CFO gap never appears. This is the Luckin pattern.

**Receivables securitization and factoring.** Selling receivables to a bank converts them to cash and shrinks the receivables balance, compressing both the gap and DSO. Sunbeam's \$59 million December 1997 program did precisely this. The disclosure is usually in the notes, not the face of the statements — which is a good reason to read the notes.

**Deferred-revenue manipulation.** Recognizing customer prepayments too early produces no receivable and no gap.

The lesson is not that the tests are weak. It is that **a fraud has to hide somewhere, and each hiding place has its own tell.** The receivables gap is the highest-yield first screen because it catches the most common fraud. It is a first screen, not a verdict.

## 5. Common misconceptions

**"The balance sheet balancing means the statements are correct."** It means the bookkeeping is internally consistent. Sunbeam's statements articulated perfectly in the year the SEC later said at least \$60 million of its reported pre-tax earnings from continuing operations came from accounting fraud. Double entry is a consistency check, not a truth check. Every entry in a well-executed fraud is a valid double entry describing a transaction that did not happen.

**"Cash flow cannot be manipulated."** The *total* change in cash is essentially unmanipulable, because a bank confirms the balance. The *classification* is very manipulable indeed. Moving an outflow from operating to investing, or an inflow from financing to operating, changes the number every analyst quotes without changing the cash. Always read CFO next to CFI and CFF, and be suspicious of a company whose CFO is strong while its free cash flow — CFO minus capital expenditure — is not.

**"A profitable company cannot go bankrupt."** Bankruptcy is a cash event. You default because you cannot pay an obligation on its due date, and obligations are settled in currency, not in net income. Fast-growing, genuinely profitable companies fail this way regularly, because growth consumes working capital faster than profit replenishes it. Sunbeam reported \$109.4 million of net earnings in 1997 and filed for Chapter 11 in February 2001.

**"Rising receivables always signal fraud."** No. A company that just landed a large enterprise customer with 90-day payment terms will show rising receivables and rising DSO for entirely honest reasons. The signal is the *combination* — receivables outrunning revenue, DSO rising against the company's own stated terms, no disclosed explanation, and the pattern persisting across several periods. One quarter is noise.

**"Auditors would have caught it."** Auditors examine a sample and rely substantially on management representations and third-party confirmations. Every case in the next section had a major auditing firm signing off. Wirecard's auditor accepted balance confirmations for €1.9 billion of cash that, it later emerged, did not exist. Audit is a control, not a guarantee, and the statements are published so that outsiders can check the work too.

**"You need inside information to spot this."** Every number used in this article's Sunbeam analysis came from a document filed with the SEC and available free on EDGAR. The gap was visible in the annual report months before the fraud was public. The information was not hidden. It was merely on a different page from the one everyone was reading.

## 6. How it shows up in real markets

Five cases. Two follow the pattern exactly; three break it in instructive ways.

### Sunbeam, 1997: the textbook receivables gap

Albert Dunlap arrived at Sunbeam in July 1996 with a reputation for aggressive restructuring and a mandate to turn the appliance maker around. Fiscal 1997 looked like a spectacular success: net sales of **\$1,168.2 million**, up 18.7% on the prior year, and net earnings of **\$109.4 million** against a 1996 loss of \$228.3 million.

The cash flow statement in the same 10-K told a different story. Operating activities **used \$8.2 million** of cash, against \$14.2 million *provided* in 1996 and \$81.5 million in 1995. Net income minus CFO — the accrual — was **\$117.7 million**, roughly 10.5% of the company's \$1,120.3 million of total assets.

And the balance sheet said where it went. Receivables, net rose from **\$213.4 million to \$295.6 million**, up 38.5% while sales grew 18.7%. DSO went from **79.2 days to 92.3 days**. Inventories rose from \$162.3 million to \$256.2 million, up 57.9%. Two of the three tests screamed, and the third would have if anyone had run it.

The mechanism, as the SEC later described it, was a combination of "cookie-jar" reserves created in the 1996 loss year and released into 1997 income, discounts and inducements offered to customers to take product early — channel stuffing — and improper bill-and-hold sales in which Sunbeam invoiced customers for barbecue grills that remained in Sunbeam's own warehouses. The SEC's May 2001 complaint against Dunlap and others stated that at least **\$60 million** of Sunbeam's reported **\$189.3 million** of 1997 earnings from continuing operations before income taxes came from accounting fraud. That \$189.3 million figure matches the 10-K's income statement line exactly.

Sunbeam restated in 1998, cutting reported 1997 net income by roughly two thirds. The SEC noted the share price fell from about \$52 in early March 1998 to about \$7 after the restated financials were issued. Dunlap was ousted in June 1998; Sunbeam filed for Chapter 11 in February 2001. Dunlap later settled with the SEC without admitting or denying the allegations, agreeing to a \$500,000 penalty and a bar from serving as an officer or director of a public company.

**The lesson:** the loudest signal was free, public, and printed inside the fraud's own annual report. The company disclosed the receivables securitization that flattered its cash flow. It disclosed the receivables balance. It disclosed the negative CFO. Detection required no access and no genius — only the habit of turning to the cash flow statement before believing the income statement.

### WorldCom, 2002: the trick that flatters cash flow too

WorldCom is the essential counterexample, and it is why nobody should treat the receivables gap as a complete method.

WorldCom's problem was **line costs** — the fees it paid other carriers to complete calls on their networks. These were ordinary operating expenses, and as revenue growth slowed in 2000 and 2001 they were crushing reported margins. So beginning in 2001, WorldCom stopped expensing a portion of them and started recording them as **capital assets** instead.

In a Form 8-K dated June 25, 2002, WorldCom disclosed that transfers of approximately **\$3.852 billion** from line cost expenses to asset accounts during 2001 and the first quarter of 2002 had not been made in accordance with generally accepted accounting principles. Subsequent investigation expanded the total to roughly **\$11 billion**. WorldCom filed for Chapter 11 bankruptcy protection on July 21, 2002 — at the time, the largest bankruptcy in United States history.

Now trace it. Suppose \$100 million of cash is paid for line costs in a year.

**Expensed correctly:** the income statement takes a \$100 million operating expense, so pre-tax income falls \$100 million. On the cash flow statement, the \$100 million cash payment is an operating outflow, so CFO falls \$100 million.

**Capitalized improperly:** the income statement takes no expense this year — only depreciation on the new "asset," say \$10 million on a ten-year life. Pre-tax income falls only \$10 million. And on the cash flow statement, that \$10 million of depreciation is a non-cash charge and is added straight back, so operating cash flow is unaffected. The \$100 million cash payment reappears as **capital expenditure in investing activities**.

Net result: reported income improves by \$90 million *and* reported operating cash flow improves by \$100 million. The gap between them does not widen. It **narrows**.

The tell for capitalization has to come from elsewhere: capital expenditure rising far faster than revenue or than any announced building programme; the gross property and equipment balance ballooning; depreciation rising as a share of revenue in later years; and above all **free cash flow** — CFO minus capital expenditure — failing to improve even as CFO does.

**The lesson:** track free cash flow, not just operating cash flow. Capitalization is a transfer between two lines of the cash flow statement, and it is invisible to anyone who reads only one of them.

### Enron, 2001: the fraud in the structures

Enron's collapse was not primarily a receivables story. Its central mechanism was the use of **special purpose entities** — nominally independent partnerships, several controlled by Enron's own executives — to hold debt and underperforming assets off Enron's balance sheet, and to book gains on transactions with entities Enron effectively controlled.

On November 8, 2001, Enron filed a Form 8-K restating its results for 1997 through 2000. The restatement **reduced reported net income by \$586 million** and **increased reported debt by \$2.6 billion**, principally because several SPEs should have been consolidated into Enron's financial statements years earlier. The year-by-year reductions were \$28 million for 1997, \$133 million for 1998, \$248 million for 1999, and \$99 million for 2000. Enron filed for bankruptcy on December 2, 2001.

The articulation lesson here is about the balance sheet rather than the income statement. Enron's reported statements tied out perfectly, because the obligations that would have unbalanced them were *not on the balance sheet at all*. Off-balance-sheet structures are a way of taking things out of the system, and articulation cannot detect what was never entered. The signals had to come from the notes on related-party transactions, from the sheer opacity of the disclosures, and from the mismatch between reported earnings quality and the business's cash generation.

**The lesson:** articulation checks the statements against each other. When the fraud is about what has been kept off the statements, you must read the notes — and treat unreadable disclosure as information in its own right.

### Wirecard, 2020: the fraud that lived in the cash line

The German payments company Wirecard makes the point that even the anchor can be attacked, if the attacker forges the anchor's confirmation.

Wirecard's business included third-party acquiring, where partner firms processed payments on Wirecard's behalf and the proceeds sat in trustee escrow accounts. On **June 18, 2020**, Wirecard announced that its auditor, EY, had been unable to obtain sufficient evidence for **€1.9 billion** of cash balances said to be held in those trustee accounts, and could not sign off the 2019 financial statements. Two banks in the Philippines said they had no such accounts and that documents purporting to show them were spurious. On **June 22, 2020**, Wirecard said the €1.9 billion likely did not exist. Chief executive Markus Braun was arrested on June 23. On **June 25, 2020**, Wirecard filed for insolvency at the Munich district court.

Consider what this does to our framework. If a company reports cash it does not have, and reports the corresponding fake profits as having been collected, then operating cash flow looks *excellent*. Net income and CFO grow together. Receivables stay low. DSO looks fine. Every test in Section 4 returns a clean result.

The tells were different in kind: profitability far above every comparable peer in the same business; a very large share of profit arising in opaque third-party arrangements in jurisdictions where verification was hard; cash accumulating on the balance sheet while the company simultaneously raised debt and equity; and years of specific, documented allegations from journalists and short sellers that the company answered with litigation and lobbying rather than with evidence. The Financial Times published detailed allegations from 2015 onward; German regulators responded at one point by banning short selling in the stock and investigating the journalists.

**The lesson:** the cash flow statement is only as trustworthy as the confirmation behind the cash. When a company's cash sits in unusual places, with unusual counterparties, in jurisdictions where confirmation is hard, the anchor is not anchored. And a company that answers evidence with lawsuits is telling you something.

### Luckin Coffee, 2020: paying for your own fake cash

The Chinese coffee chain Luckin shows the most sophisticated response to the problem this article describes — what a fraudster does once they understand that fake revenue with no cash leaves a fingerprint.

On April 2, 2020, Luckin disclosed that an internal investigation had found its chief operating officer and several subordinates had fabricated transactions. The independent investigation concluded that 2019 sales had been inflated by roughly **RMB 2.12 billion** and costs and expenses by roughly **RMB 1.34 billion**. In December 2020, the SEC announced that Luckin had agreed to pay a **\$180 million** penalty to settle accounting fraud charges; the SEC's order stated that from at least April 2019 through January 2020 Luckin intentionally fabricated more than **\$300 million** in retail sales, using related parties to create false sales transactions through three separate purchasing schemes, and that funds were funnelled back to Luckin to support the falsified transactions.

That last clause is the whole point. The fake revenue was **funded**. Money was pushed out of the company through inflated expenses, routed through related parties, and returned as apparent customer payments. Cash genuinely arrived. Receivables did not balloon. The classic gap did not open.

But the conservation law does not stop applying — it just moves the evidence. If cash comes in to pay for fake sales, that cash had to leave somewhere first, and it left through the inflated cost and expense lines. Hence RMB 1.34 billion of inflated expenses alongside RMB 2.12 billion of inflated sales. The pattern to look for is a company whose revenue is soaring while its unit economics quietly refuse to improve, whose costs rise in suspiciously close proportion to sales, and whose related-party disclosures are extensive. In Luckin's case, the most cited outside work was a lengthy anonymous report circulated in early 2020 that relied on physical observation — thousands of hours of store video and tens of thousands of collected receipts — to argue that per-store transaction counts were far below the reported figures. When the books cannot be trusted, count the customers.

**The lesson:** a fraud that closes the cash gap has to pay for it, and paying for it shows up somewhere else. Follow the conservation law to wherever the money had to come from.

## 7. Where this leaves you

The frame is now built, and everything else in forensic accounting is detail hanging off it.

The three statements are one system. The income statement measures performance across a period, the balance sheet snapshots position at an instant, and the cash flow statement explains how the cash line got from the first snapshot to the second. Net income flows into retained earnings; ending cash ties to the balance sheet. Both links are exact, and you can verify them on any annual report in a couple of minutes.

The practical habit that falls out of this is small and worth adopting permanently. **When you read a company's results, read the cash flow statement first.** Not the press release, not the earnings-per-share line, not the adjusted-EBITDA slide. Find operating cash flow, put it next to net income, and ask whether the profit is backed by money. Then look at receivables and ask whether they are growing in line with sales. If those two checks are clean, the aggressive stuff is probably confined to estimates rather than to invention. If they are not clean, you have a question that deserves an answer, and the answer is usually somewhere in the notes.

That is a screen, not a verdict, and it should be held with appropriate humility. Honest companies fail it during periods of fast growth. Dishonest companies pass it when the fraud lives on the balance sheet, in capitalized costs, or in funded round-trip transactions. The tests narrow the field; they do not close a case. Nothing here is investment advice — it is a way of reading a document.

From here the natural next steps go one statement at a time. [Reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings) takes apart the revenue and expense lines and the estimates buried in them. [Reading the balance sheet: what companies hide here](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here) works through the asset and liability lines where value is parked, inflated, or omitted. [Reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) goes deep on the classification games between CFO, CFI, and CFF that this article has only pointed at. And [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) formalizes the judgment layer that every one of these techniques operates inside.

One closing exercise, and it is the best hour you can spend on this material. Pull the most recent annual report of a company you actually care about — it is free on EDGAR for any United States issuer. Find retained earnings on this year's and last year's balance sheets, find net income and dividends, and check that Link 1 ties. Then find beginning cash, CFO, CFI, and CFF, and check that Link 2 ties. Then put net income next to operating cash flow and divide.

Ninety-five times out of a hundred the answer will be reassuring, and you will have learned the shape of normal. The other five times you will have a specific, well-formed question about a specific company — which is exactly what this whole frame was built to produce.

## Sources & further reading

**Primary filings and regulatory documents**

- Sunbeam Corporation, Form 10-K405 for the fiscal year ended December 28, 1997, filed with the US Securities and Exchange Commission. Source of all Sunbeam statement figures used above: net sales \$1,168.2m, net earnings \$109.4m, pre-tax earnings from continuing operations \$189.3m, CFO −\$8.2m, CFI +\$32.7m, CFF +\$16.4m, cash \$11.5m → \$52.4m, receivables \$213.4m → \$295.6m, inventories \$162.3m → \$256.2m, retained earnings \$35.1m → \$141.1m, dividends \$3.4m, total assets \$1,120.3m, and the \$59m receivables securitization disclosure. Available on SEC EDGAR.
- US Securities and Exchange Commission, Litigation Release No. 17001, *SEC v. Albert J. Dunlap, Russell A. Kersh, Robert J. Gluck, Donald R. Uzzi, Lee B. Griffith, and Phillip E. Harlow* (May 15, 2001) — the \$60 million of fraudulent earnings within the reported \$189.3 million, the cookie-jar reserves, channel stuffing and bill-and-hold findings.
- US Securities and Exchange Commission, Administrative Proceeding File No. 33-7976, *In the Matter of Sunbeam Corporation* (May 15, 2001) — the restatement and the share-price decline from approximately \$52 to approximately \$7.
- US Securities and Exchange Commission, Litigation Release No. 17710 (September 4, 2002) — the Dunlap settlement: a \$500,000 civil penalty and a permanent bar from serving as an officer or director of a public company, consented to without admitting or denying the allegations.
- WorldCom Inc., Form 8-K filed June 25, 2002 — the approximately \$3.852 billion of improper transfers from line cost expense to asset accounts.
- Enron Corp., Form 8-K filed November 8, 2001 — the restatement of 1997–2000 results reducing net income by \$586 million and increasing debt by \$2.6 billion.
- US Securities and Exchange Commission, "Luckin Coffee Agrees to Pay \$180 Million Penalty to Settle Accounting Fraud Charges," Press Release 2020-319 (December 2020), and the associated litigation release — more than \$300 million of fabricated retail sales between April 2019 and January 2020.
- Luckin Coffee Inc., Form 6-K disclosures (2020) — the special committee investigation findings of approximately RMB 2.12 billion of inflated 2019 sales and RMB 1.34 billion of inflated costs and expenses.

**Standards**

- Financial Accounting Standards Board, Statement of Financial Accounting Standards No. 95, *Statement of Cash Flows* (1987) — the standard that made the cash flow statement mandatory in US GAAP, and the source of the operating/investing/financing classification.
- International Accounting Standards Board, IAS 1 *Presentation of Financial Statements* and IAS 7 *Statement of Cash Flows* — the IFRS equivalents.

**Background reporting**

- Congressional Research Service, *The Enron Collapse: An Overview of Financial Issues* (RS21135) and *WorldCom: The Accounting Scandal* (RS21253) — concise, sourced summaries of both restatements.
- Financial Times, multi-year investigative coverage of Wirecard beginning in 2015, and contemporaneous reporting of the June 2020 collapse.

**Note on figures.** All Sunbeam, WorldCom, Enron, Wirecard and Luckin figures above are taken from the filings and regulatory documents listed. **Northwind Tools is a hypothetical company** invented for this article; every Northwind figure, and every single-transaction walkthrough using \$10,000 and \$6,000, is an illustrative example chosen to make the arithmetic legible, not a report of any real company's results.
