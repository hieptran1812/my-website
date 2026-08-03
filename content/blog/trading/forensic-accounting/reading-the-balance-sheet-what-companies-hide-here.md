---
title: "Reading the Balance Sheet: What Companies Hide Here"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "A beginner-friendly deep dive into how a balance sheet is built, why most asset lines are estimates rather than facts, and how equity silently absorbs every overstated asset and every hidden liability."
tags: ["balance-sheet", "forensic-accounting", "financial-statement-analysis", "goodwill", "accounts-receivable", "off-balance-sheet", "working-capital", "solvency", "liquidity", "financial-statement-fraud", "contingent-liabilities"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 61
---

> [!important]
> **TL;DR** — A balance sheet always balances, which means balancing proves nothing. Most of what sits on the asset side is not a fact but a forecast, and every overstated forecast lands, dollar for dollar, in equity.
>
> - The accounting identity — **assets = liabilities + equity** — is true by construction, because equity is *defined* as assets minus liabilities. It is a definition, not a test. A balance sheet full of lies balances perfectly.
> - The balance sheet is a **photograph at one instant**; the income statement and cash flow statement are **films of the period between two photographs**. Management chooses the day it is photographed, and that choice alone is worth billions.
> - Reading down the asset side is reading **from fact toward forecast**: cash, then receivables (a promise), then inventory (a bet on demand), then net property (a guess at useful life), then capitalized costs, then goodwill (a bet that an acquisition works, tested only against management's own projections).
> - Two families of manipulation. Accounts that **flatter** — receivables, inventory, capitalized costs, goodwill, "other assets" — inflate equity by inflating an asset. Accounts that **hide** — contingent liabilities, off-balance-sheet vehicles, the pre-2019 operating lease, related-party balances — inflate equity by keeping a liability off the page.
> - **Solvency and liquidity are different questions.** Solvency asks whether assets exceed liabilities; liquidity asks whether cash arrives before the bills do. Companies die of illiquidity while still solvent, and they do it fast.
> - The number to remember: per its chairman's confession letter of 7 January 2009, Satyam's balance sheet at 30 September 2008 carried **Rs 5,040 crore** — roughly one billion US dollars at the exchange rate of the time — of cash and bank balances that did not exist. Cash is the line everyone assumes is a fact. It was the line that was fake.

Here is a question that people who have never read a financial statement find strange, and people who read them for a living find obvious: **what does it actually mean that a balance sheet balances?**

The natural assumption is that it means something. It sounds like a check. Two columns, both adding to the same number — surely if the accountants got something wrong, the columns would disagree and someone would notice. That is what the word *balance* implies. It implies a scale, and a scale that tips when you cheat.

It does not work like that. The two sides of a balance sheet are equal because one of them is *defined* as the other. Equity — the shareholders' stake, the "book value" of the company — is not measured independently. It is calculated as whatever is left over after you subtract liabilities from assets. So if you inflate an asset by three hundred million dollars, the balance sheet does not tip. It absorbs the lie into equity and balances just as perfectly as before, only now the company appears three hundred million dollars richer. WorldCom's balance sheet balanced. Enron's balance sheet balanced. Satyam's balance sheet balanced beautifully, right up to the morning its chairman wrote a letter admitting that about a billion dollars of the cash line was imaginary.

This is why the balance sheet is where forensic accounting actually lives. The income statement is where a fraud is *committed* — that is where the fake profit gets announced. The balance sheet is where the fraud is *stored*. Every fake dollar of profit has to sit somewhere afterwards, and it sits on the asset side, under a heading, waiting to be found. A manipulator can make one quarter's income statement say almost anything. What they cannot do is make the residue disappear. It accumulates. That accumulation is the trail.

The diagram below is the mental model for the whole article. Watch the difference between the wide band and the two thin columns.

![The income statement and cash flow statement measure flow across the whole of FY2025, while the balance sheet is a single vertical instant at each year-end: a photograph whose date management chooses.](/imgs/blogs/reading-the-balance-sheet-what-companies-hide-here-1.webp)

The wide band is a *film*: the income statement and the [cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) both measure what happened over twelve months. The two thin columns are *photographs*: the balance sheet describes one instant, 31 December, and says nothing whatsoever about the 364 days in between. That distinction sounds academic. It is worth tens of billions of dollars, and we will see exactly how in the section on window dressing.

We are going to build the balance sheet from nothing — no accounting background assumed, every term defined the first time it appears — and then read it the way a forensic accountant does: as a stack of claims about the future, most of which are unverifiable, some of which are wrong, and a few of which are lies.

## Foundations: how the balance sheet actually works

Skip nothing here if you are new. Everything in the second half of this article is a consequence of the mechanics in this section, and the mechanics are simpler than the jargon suggests.

### The one equation that can never be wrong

A **balance sheet** (in international accounting language, a *statement of financial position*) is a list of everything a company owns and everything it owes, at one moment in time. It has exactly three kinds of line.

**Assets** are resources the company controls that are expected to produce future economic benefit. Cash in the bank. Money customers owe. Goods sitting in a warehouse. Machines. Buildings. The right to use a brand.

**Liabilities** are obligations — amounts the company must hand over to someone else. Bills from suppliers. Wages earned but not yet paid. Bank loans. Bonds.

**Equity** (also called *shareholders' equity*, *net assets*, or *book value*) is what belongs to the owners. And here is the sentence that most explanations bury: equity is not counted. It is *derived*.

$$\text{Assets} = \text{Liabilities} + \text{Equity}$$

Rearrange it and the point becomes obvious:

$$\text{Equity} = \text{Assets} - \text{Liabilities}$$

Nobody walks around a company counting up its equity. Equity is the plug — the number that makes the equation close. Which means the equation *always* closes, regardless of whether the assets and liabilities feeding it are honest.

There is a second reason a balance sheet always balances, and it is mechanical rather than definitional. Companies keep books using **double-entry bookkeeping**, a technique formalized in fifteenth-century Venice, in which every transaction is recorded twice: once as a debit and once as a credit, of equal size. Buy a machine for cash, and cash goes down while equipment goes up by the same amount. Borrow money, and cash goes up while debt goes up. There is no transaction you can enter that breaks the equality, because the software will not let you enter one. A fraudster using double-entry bookkeeping is not fighting the system; they are using it exactly as designed, and the system dutifully keeps the books balanced while they do it.

So: **balancing is not evidence.** It is the accounting equivalent of a sentence being grammatical. "The warehouse contains four hundred million dollars of tools" is grammatical whether or not the warehouse is empty.

### A photograph, not a film

Public companies publish three primary statements, and beginners routinely mix up what each one measures. The distinction is about *time*.

| Statement | What it measures | Time shape |
| --- | --- | --- |
| Balance sheet | What is owned and owed | An **instant** — 31 December, one moment |
| Income statement | Revenue earned minus expenses incurred | A **period** — the whole of the year |
| Cash flow statement | Cash actually in and out | A **period** — the whole of the year |

The three are welded together, and how they interlock is its own subject — see [the three financial statements and how they interlock](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock). The weld that matters most here is this one: **net income flows into the balance sheet through retained earnings**. Whatever profit the income statement declares gets added to a line inside equity called *retained earnings* — the cumulative profit a company has earned since its founding and not paid out as dividends. That is the pipe through which every income-statement lie eventually reaches the balance sheet. And because equity is the plug, an inflated profit must be balanced by an inflated asset or a suppressed liability. There is nowhere else for it to go.

The instant-versus-period distinction has a consequence that is easy to state and easy to underestimate: **the balance sheet describes a day that management chose in advance and knew was coming.** Nobody is surprised by 31 December. A company that wants its year-end picture to look a particular way has a full quarter to arrange it — collect receivables early, delay paying suppliers until 2 January, sell securities temporarily and buy them back in the new year. None of this is necessarily fraud. All of it makes the photograph unrepresentative of the other 364 days.

#### Worked example: building a balance sheet in ten moves

The fastest way to internalize the identity is to build a balance sheet from zero. You are opening a small workshop that makes steel benches. We will do ten things and watch both sides after each one. Every number is in plain dollars.

1. **You invest \$100,000 of your own savings.** Cash \$100,000 on the left; paid-in capital \$100,000 in equity on the right. Total assets \$100,000 = liabilities \$0 + equity \$100,000.
2. **The bank lends you \$60,000 for five years.** Cash rises to \$160,000; long-term debt \$60,000 appears. Assets \$160,000 = liabilities \$60,000 + equity \$100,000. Notice: borrowing made you no richer. Both sides grew equally, and your equity did not move.
3. **You buy machines for \$90,000 cash.** Cash falls to \$70,000; equipment \$90,000 appears. Total assets are still \$160,000, and the right-hand side did not move at all. You swapped one asset for another.
4. **You buy \$40,000 of steel on 60-day supplier credit.** Inventory \$40,000 appears; accounts payable \$40,000 appears. Assets \$200,000 = liabilities \$100,000 + equity \$100,000.
5. **You pay \$15,000 of that supplier bill.** Cash falls to \$55,000; payables fall to \$25,000. Assets \$185,000 = liabilities \$85,000 + equity \$100,000. Paying a bill shrinks both sides simultaneously and leaves you no poorer.
6. **You sell benches for \$50,000 on 30-day credit; the steel in them cost \$30,000.** Accounts receivable \$50,000 appears, inventory falls to \$10,000, and \$20,000 of profit lands in retained earnings. Assets \$205,000 (cash 55,000 + receivables 50,000 + inventory 10,000 + equipment 90,000) = liabilities \$85,000 + equity \$120,000. **This is the first move that made you richer** — and no cash has changed hands.
7. **The customer pays the \$50,000.** Cash rises to \$105,000, receivables fall to zero. Total assets: unchanged at \$205,000. Collecting cash does not create profit; the profit was already recorded in step 6. This is the [accrual](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) at work.
8. **You pay \$8,000 of wages in cash.** Cash \$97,000; retained earnings fall to \$12,000. Assets \$197,000 = liabilities \$85,000 + equity \$112,000.
9. **A year of use wears the machines: \$9,000 of depreciation.** Equipment falls to \$81,000; retained earnings fall to \$3,000. Assets \$188,000 = liabilities \$85,000 + equity \$103,000. No cash moved. An accountant's estimate of wear reduced an asset and reduced your wealth.
10. **A customer sues you for \$500,000 over a bench that collapsed.** Nothing changes. Not one line. The balance sheet is exactly as it was: assets \$188,000, liabilities \$85,000, equity \$103,000. The lawsuit goes in a footnote.

Read step 10 again. The single largest financial fact about this business — a claim five times its equity — appears nowhere on the balance sheet. It will appear on the balance sheet only when the accountants judge that losing is *probable* and the amount is *estimable*, and the word "probable" is a judgment call made by the people whose bonus depends on the answer.

**The intuition: the balance sheet is complete only in the sense that it lists what the rules require it to list — and the rules leave enormous, deliberate holes.**

Here is what a grown-up version of that same company looks like. Meet **Northwind Tools Inc.**, a hypothetical industrial manufacturer we will use for every worked example in this article. All figures are in millions of dollars. It is invented; none of its numbers are real-company figures.

![Northwind Tools' balance sheet drawn as two stacked columns of equal height: assets of \$7,000 on the left, liabilities of \$4,000 and equity of \$3,000 on the right.](/imgs/blogs/reading-the-balance-sheet-what-companies-hide-here-2.webp)

| Assets | \$m | Liabilities and equity | \$m |
| --- | ---: | --- | ---: |
| Cash and equivalents | 400 | Accounts payable | 700 |
| Accounts receivable | 900 | Accrued expenses | 300 |
| Inventory | 1,100 | Short-term debt | 500 |
| Prepaid expenses | 100 | **Total current liabilities** | **1,500** |
| **Total current assets** | **2,500** | Long-term debt | 2,500 |
| Property, plant and equipment, net | 3,000 | **Total liabilities** | **4,000** |
| Goodwill | 1,200 | Paid-in capital | 1,800 |
| Other assets | 300 | Retained earnings | 1,200 |
| | | **Total equity** | **3,000** |
| **Total assets** | **7,000** | **Total liabilities and equity** | **7,000** |

Northwind's income statement for the same year: revenue \$6,000m, cost of goods sold \$4,200m, selling and administrative expense \$1,000m, depreciation and amortization \$280m, interest \$100m, pre-tax profit \$420m. A 7.0% pre-tax margin on \$6,000m of sales. Unremarkable, which is the point — most manipulated balance sheets look unremarkable.

### Current and non-current: the one-year line

Both sides of the balance sheet are split by a single question: **does this turn into cash, or come due, within twelve months?**

- **Current assets** are expected to be converted to cash or consumed within a year: cash itself, marketable securities, receivables, inventory, prepaid expenses.
- **Non-current assets** (also called long-term or fixed assets) are everything else: property, plant and equipment, goodwill, long-lived intangibles, long-term investments.
- **Current liabilities** come due within a year: trade payables, accrued wages, the portion of long-term debt maturing within twelve months, short-term borrowings.
- **Non-current liabilities** come due later: term loans, bonds, pension obligations, deferred tax liabilities.

Within current assets, lines are conventionally ordered by **liquidity** — how quickly and reliably they turn into spendable cash without losing value. Cash first, then receivables, then inventory. That ordering is a useful hint from the accounting profession, and most readers ignore it. It is telling you, line by line, how far each asset sits from being money.

![A two-by-two grid splitting assets and liabilities into current and non-current, with Northwind's working capital of \$1,000m shown as the gap between \$2,500m of current assets and \$1,500m of current liabilities.](/imgs/blogs/reading-the-balance-sheet-what-companies-hide-here-3.webp)

### Working capital: the buffer between the two

**Working capital** is the simplest and most useful derived number on the whole statement:

$$\text{Working capital} = \text{Current assets} - \text{Current liabilities}$$

For Northwind: \$2,500m − \$1,500m = **\$1,000m**. In plain English: after everything due within a year is paid, \$1,000m of short-term resources remain. It is the cushion between what arrives soon and what is owed soon.

Two ratios express the same idea:

- **Current ratio** = current assets ÷ current liabilities = 2,500 ÷ 1,500 = **1.67**. Rules of thumb put "healthy" somewhere above 1.5 for a manufacturer, though it varies enormously by industry — a supermarket runs comfortably below 1.0 because it sells inventory for cash long before it pays suppliers.
- **Quick ratio** (or *acid-test*) = (cash + receivables) ÷ current liabilities = (400 + 900) ÷ 1,500 = **0.87**. This strips out inventory, on the theory that in a crisis you cannot sell a warehouse of tools quickly at full price.

Hold on to the gap between those two numbers — 1.67 versus 0.87. It exists entirely because of \$1,100m of inventory. Whether that inventory is worth \$1,100m is a question the balance sheet cannot answer and will not raise.

### Book value is not what the company is worth

One more foundation before the forensics, because it prevents a whole class of confusion.

**Book value** is equity as stated on the balance sheet: \$3,000m for Northwind. **Market capitalization** is what the stock market thinks the equity is worth: shares outstanding times share price. These are almost never equal, and the gap is not a sign that anyone is lying.

They differ because accounting is **backward-looking and cost-based**. Most assets are carried at historical cost less accumulated depreciation, not at what they would fetch today. A warehouse bought in 1994 for \$8m and depreciated to \$1m might be worth \$40m; the balance sheet says \$1m and will keep saying \$1m. And the things that make many modern companies valuable — a brand built internally, a software team, a customer base, a research pipeline — cost money to create that was expensed as incurred, so they appear on the balance sheet at exactly zero.

This asymmetry produces one of accounting's strangest rules, which we will return to: a brand you *build* is worth nothing on your balance sheet, but a brand you *buy* shows up as goodwill and can be worth billions. Same brand, different accounting, purely because of how it was acquired.

## The balance sheet is a set of claims about the future, not a set of facts

Now the shift in perspective that the whole article is built on.

You probably read a balance sheet the way you read a bank statement: as a record of things that are true. \$400m of cash, \$900m owed to us, \$1,100m of goods, \$3,000m of factories. Facts.

Almost none of that is a fact. **Nearly every line on the asset side is a statement about the future**, dressed in the grammar of the present tense.

- "Accounts receivable \$900m" does not mean \$900m exists. It means: *we believe customers will pay us \$900m.*
- "Inventory \$1,100m" does not mean the warehouse contains \$1,100m of value. It means: *we believe we can sell these goods for at least what we paid.*
- "Property, plant and equipment, net \$3,000m" means: *we paid more than this once, and we estimate this much of the useful life remains.*
- "Goodwill \$1,200m" means: *we paid \$1,200m more than the identifiable net assets of a company we bought were worth, and we continue to believe that premium was justified.*

Every one of those is a forecast. Forecasts made by people whose compensation depends on the forecast being optimistic, reviewed by auditors who see the company for a few weeks a year and largely rely on management's own models to test them.

### The certainty ladder

Not all forecasts are equally speculative. Reading down the asset side is reading from fact toward fiction, and it is worth having the ladder explicitly in your head.

![A seven-rung ladder ordering asset lines from most verifiable to most speculative: cash, receivables, inventory, net PP&E, capitalized costs, intangibles, and goodwill at the bottom.](/imgs/blogs/reading-the-balance-sheet-what-companies-hide-here-4.webp)

| Rung | Line | What it actually asserts | How hard to verify |
| --- | --- | --- | --- |
| 1 | Cash and equivalents | Money exists in a named account | A bank confirmation — the easiest to audit, and the one Satyam and Wirecard both faked |
| 2 | Accounts receivable | Named third parties will pay us | Confirmable in principle; laborious and sample-based in practice |
| 3 | Inventory | These goods can be sold at or above cost | Counting is easy; **valuing** is a judgment about demand |
| 4 | PP&E, net | Cost, less an estimate of consumed useful life | The existence is checkable; the depreciation schedule is an assumption |
| 5 | Capitalized costs | A cost already spent has future value | Tests management's classification choices against a vague standard |
| 6 | Identifiable intangibles | A patent, licence or customer list is worth this | An appraiser's opinion, commissioned by the buyer |
| 7 | Goodwill | An acquisition premium remains justified | Tested only against management's own forecasts of its own business |

Rung 7 deserves a sentence on its own. **Goodwill is tested for impairment by discounting management's projections of the acquired business's future cash flows.** If those projections stay optimistic, goodwill stays on the books at full value. The asset that requires the most faith is the one whose valuation is most fully controlled by the party with the strongest incentive.

And rung 1 deserves the opposite warning. Cash is the line everyone treats as beyond question — which is exactly what made it attractive to fake at both Satyam and Wirecard. A forged bank confirmation is a piece of paper. Auditors are supposed to obtain confirmations directly from the bank; when they instead accept scans, copies, or documents routed through the client or an intermediary, the most "certain" line on the balance sheet becomes the least.

### Equity is the plug

Now put the identity together with the ladder and you get the central mechanic of balance-sheet fraud.

Because equity is defined as assets minus liabilities, and liabilities are relatively hard to inflate downward (someone else is on the other side of a liability, and they tend to notice if you stop acknowledging their claim), the manipulator's lever is the asset side. **Every dollar you add to an asset that is not really there adds exactly one dollar to equity** — and, because assets are increased by *not* recording an expense, exactly one dollar to pre-tax profit as well.

One entry. Three numbers move: the asset, equity, and profit. A fourth follows for free: leverage looks lower, because leverage is measured against equity.

#### Worked example: one inflated line, four numbers moved

Northwind's inventory is stated at \$1,100m. Suppose \$300m of it is tooling for a product line the company quietly discontinued eighteen months ago. Nobody will buy it. Under both US GAAP and IFRS, inventory must be written down when its net realizable value falls below cost, so the honest treatment is a \$300m write-down.

Management does not take it. Here is the before and after, with nothing else changed.

| | Honest | With the \$300m left in | Change |
| --- | ---: | ---: | --- |
| Inventory | 1,100 | 1,400 | +300 |
| Total assets | 7,000 | 7,300 | +300 |
| **Total liabilities** | **4,000** | **4,000** | **unchanged** |
| Total equity | 3,000 | 3,300 | +300 |
| Debt-to-equity | 1.33 | 1.21 | *improves* |
| Pre-tax profit | 420 | 720 | +71% |
| Pre-tax margin | 7.0% | 12.0% | +5.0 pts |
| Book value per share (100m shares) | \$30.00 | \$33.00 | +10% |

![Two balance-sheet stacks side by side: with \$300m of worthless inventory left on the books, assets rise to \$7,300m and equity to \$3,300m while liabilities stay at exactly \$4,000m.](/imgs/blogs/reading-the-balance-sheet-what-companies-hide-here-5.webp)

Look at the row in bold. **Liabilities did not move.** They cannot, because inventory obsolescence has no counterparty. That is the fingerprint of asset inflation: the right-hand side stays still while the left-hand side and equity rise together.

And look at what one decision bought. Pre-tax profit rose 71%, from a pedestrian \$420m to a strong \$720m. The pre-tax margin went from 7.0% to 12.0% — the difference between a mediocre industrial and a well-run one. Leverage *improved*, from 1.33 to 1.21, potentially creating headroom under a debt covenant. Book value per share rose 10%, which flatters every price-to-book comparison an analyst will run.

No cash moved. No supplier was defrauded. No invoice was forged. Somebody decided not to write something down, and every headline number improved at once.

**The intuition: the income statement is where the fraud is announced, but the balance sheet is where it is stored — and because equity is the plug, the storage is always exactly the size of the lie.**

Now we can go line by line.

## The accounts that flatter

These are the lines a company inflates to make itself look richer and more profitable than it is. They share a structure: an expense that should have been recognized was instead parked on the balance sheet as an asset.

### Accounts receivable: revenue that has not become money yet

**Accounts receivable** (often *trade receivables*, or just *debtors*) is the money customers owe for goods already delivered. It is created the moment revenue is recognized on credit, which makes it the direct balance-sheet consequence of the revenue-recognition decisions covered in [reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings).

The key metric is **days sales outstanding (DSO)**, which converts the receivables balance into a number of days of sales:

$$\text{DSO} = \frac{\text{Accounts receivable}}{\text{Revenue}} \times 365$$

For Northwind in Year 1: (900 ÷ 6,000) × 365 = **55 days**. On average, the company waits 55 days between delivering a bench and being paid for it.

DSO is powerful because it is a *ratio*, and ratios are much harder to manage than levels. A company can grow receivables honestly — if you sell more, customers owe you more. What a company cannot do honestly, for long, is grow receivables *faster than the sales that created them*. That means each dollar of revenue is taking longer to collect, and there are only three explanations: customers have grown poorer, the company loosened credit terms to buy sales, or some of those sales are not real.

Three levers sit behind this line:

- **The allowance for doubtful accounts** — a contra-asset (a negative asset that reduces the gross figure) representing the portion of receivables the company expects never to collect. Net receivables = gross receivables − allowance. Because the allowance is an estimate, keeping it flat in dollars while gross receivables grow quietly reduces the expense charged this year.
- **Channel stuffing** — shipping more product to distributors than they can sell, often with generous return rights, booking the revenue, and letting the receivable sit. The sale is technically real and economically fictional.
- **Bill-and-hold** — invoicing a customer for goods that remain in your own warehouse. Legitimate under narrow conditions (the buyer must have requested it and taken on the risks of ownership); routinely abused. This was central to the Sunbeam case of the late 1990s.

#### Worked example: receivables outrunning revenue

Northwind over three years. Revenue and receivables both grow; the question is whether they grow together.

| | Year 1 | Year 2 | Year 3 |
| --- | ---: | ---: | ---: |
| Revenue (\$m) | 6,000 | 6,600 | 7,100 |
| Accounts receivable (\$m) | 900 | 1,180 | 1,560 |
| Revenue index (Year 1 = 100) | 100 | 110 | 118 |
| Receivables index (Year 1 = 100) | 100 | 131 | 173 |
| **DSO (days)** | **55** | **65** | **80** |

![An indexed line chart showing revenue rising from 100 to 118 while accounts receivable rise from 100 to 173 over three years, with DSO drifting from 55 to 80 days and \$490m of excess receivables at Year 3.](/imgs/blogs/reading-the-balance-sheet-what-companies-hide-here-6.webp)

Over three years revenue grew 18%; receivables grew 73%. Receivables grew **four times as fast as the sales that supposedly created them**.

Now quantify the gap. If DSO had held at its Year 1 level of 55 days, Year 3 receivables should have been:

7,100 × 55 ÷ 365 = **\$1,070m**

The actual balance is \$1,560m. The excess is **\$490m** — money the company says it is owed, above and beyond what its sales volume can explain.

That \$490m has to be one of three things. It is money owed by customers who cannot pay, in which case the allowance for doubtful accounts is far too small and \$490m of expense is missing. It is product pushed onto distributors who will eventually send it back, in which case revenue was pulled forward and will reverse. Or it corresponds to sales that were never made at all.

The scale matters: \$490m against equity of \$3,000m is **16% of book value**. If that receivable is written off, one-sixth of the shareholders' stake vanishes in a single line — and on the Year 3 balance sheet we build later, debt-to-equity jumps from 1.78 to 2.13, which for a covenant-constrained borrower can be the difference between a going concern and a restructuring.

**The intuition: revenue is a claim, and receivables are the unpaid part of that claim piling up. When the pile grows faster than the claims, some of the claims are not going to be honored.**

A companion check applies on the liability side. Compare receivables growth against **deferred revenue** (cash collected from customers before delivery, which is a *liability* because you owe them a product). Healthy demand tends to lift both. Receivables rising while deferred revenue falls is a company recognizing revenue earlier and collecting cash later at the same time — the two things a stretched business does.

### Inventory: a bet on demand, carried at cost

**Inventory** is goods held for sale plus the raw materials and work-in-progress that will become them. It is measured at the lower of cost and net realizable value, which sounds conservative and hides two dials.

The first dial is **obsolescence**. Inventory is written down when it will not sell for what it cost. Deciding *when* that moment arrives is a judgment, and the judgment has a direct, dollar-for-dollar effect on profit — exactly the \$300m move in our earlier worked example. Slow-moving stock carried at full cost is the single most common form of quiet asset inflation, because unlike a fake receivable it involves no counterparty and no forged document. It involves *not doing something*.

The second dial is subtler and entirely legal, which makes it more interesting. Under **absorption costing** — required by both US GAAP and IFRS — a manufacturer's fixed factory overheads (rent, supervisory salaries, machine depreciation) are attached to the units produced, not expensed as incurred. Those overheads then sit inside inventory on the balance sheet until the unit is sold, at which point they flow into cost of goods sold.

The consequence: **producing more units than you sell moves fixed overhead off the income statement and onto the balance sheet.** Suppose a plant has \$100m of annual fixed overhead. Produce 1,000,000 units and sell 800,000, and roughly \$20m of overhead attaches to the 200,000 unsold units and sits in inventory rather than hitting this year's cost of goods sold. Gross profit is \$20m higher — because the factory ran hot, not because anything was sold.

IAS 2 constrains this by requiring overhead allocation based on *normal* capacity, with unallocated overhead expensed immediately, so extreme over-production is not supposed to work. But "normal capacity" is itself an estimate. The detection signal is a simple pair: **inventory days rising while revenue is flat**, and production volume exceeding sales volume for consecutive periods.

The inventory analogue of DSO is **days inventory outstanding (DIO)** = inventory ÷ cost of goods sold × 365. For Northwind in Year 3: 1,600 ÷ 4,900 × 365 = **119 days**. Nearly four months of goods sitting in warehouses, against a Year 1 figure of 1,100 ÷ 4,200 × 365 = **96 days**. Twenty-three extra days of stock, on a business whose sales grew 18%, is a company building product it is not selling.

### Property, plant and equipment: cost minus a guess

**Property, plant and equipment (PP&E)** is the physical long-lived stuff: land, buildings, machinery, vehicles, fixtures. It appears at cost less **accumulated depreciation** — the running total of the asset's cost that has been charged to the income statement as it wears out.

Depreciation requires two assumptions: **useful life** (how many years the asset will serve) and **residual value** (what it will be worth at the end). Both are management estimates, both are disclosed only in general terms, and both move profit directly.

Extending an asset's assumed useful life reduces the annual depreciation charge and raises profit for every remaining year, with no cash effect and no counterparty. A fleet of trucks depreciated over 8 years instead of 5 costs 37.5% less per year in depreciation. Across a \$3,000m asset base, small changes in assumed life are worth tens of millions annually. The change must be disclosed as a change in accounting estimate; it is usually a sentence in a footnote, and the sentence does not say "this added \$60m to profit."

Two signals worth checking. First, **accumulated depreciation as a share of gross PP&E** tells you how old the asset base is; a number drifting toward 70–80% on a company reporting rising profits means the plant is aging and a capital-spending cliff is coming. Second, **depreciation expense divided by gross PP&E** gives an implied depreciation rate; if that rate falls year over year without a change in the asset mix, lives are being extended.

### Capitalized costs: turning an expense into an asset

This is the mechanism behind the largest accounting fraud in American corporate history, and it is simple enough to explain in one sentence: **when you spend money, you either expense it (it hits profit now) or capitalize it (it becomes an asset and hits profit gradually over years).**

The rule is supposed to be about future benefit. A cost that buys something with lasting value — a machine, a building — is capitalized. A cost that is consumed now — rent, wages, maintenance — is expensed. The boundary is genuinely blurry for a large class of spending: software development, exploration costs, customer acquisition, refurbishment that might be "maintenance" or might be "improvement."

#### Worked example: expense it or capitalize it

Northwind spends \$120m on network maintenance in Year 1. Treatment A expenses it — which is what produces the \$420m of pre-tax profit we have been using all along. Treatment B capitalizes it and depreciates it straight-line over four years, at \$30m a year.

| Year 1 | Expense it | Capitalize it | Difference |
| --- | ---: | ---: | ---: |
| Charge to Year 1 pre-tax profit | −120 | −30 | +90 |
| **Reported pre-tax profit** | **420** | **510** | **+90** |
| Asset on the balance sheet at year end | 0 | 90 | +90 |
| Reported equity | 3,000 | 3,090 | +90 |
| Cash actually paid out | −120 | −120 | 0 |
| Shown in operating cash flow | −120 | 0 | **+120** |
| Shown in investing cash flow | 0 | −120 | −120 |

![Two vertical paths for the same \$120m of cost: expensing it cuts pre-tax profit by \$120m and creates no asset, while capitalizing it cuts profit by only \$30m, creates a \$90m asset, and moves the \$120m outflow from operating to investing cash flow.](/imgs/blogs/reading-the-balance-sheet-what-companies-hide-here-7.webp)

Three things improved at once from a single classification choice. Year 1 pre-tax profit is **\$90m higher**. Total assets and equity are **\$90m higher**. And operating cash flow — the number analysts reach for precisely *because* it is supposed to be harder to manipulate than profit — is **\$120m higher**, because the outflow was reclassified from operating to investing.

That third line is the reason this technique is so dangerous. The standard advice for detecting earnings manipulation is "compare net income to operating cash flow." Capitalization defeats that check, because it lifts both numbers together. Relative to Northwind's base pre-tax profit of \$420m, capitalizing \$120m of ordinary maintenance raises reported profit to \$510m — a 21.4% increase — while making the cash flow statement look *better*, not worse.

Over four years the totals are identical: \$120m of cost is \$120m of cost either way. But Year 1 is the year the bonus is paid and the guidance is met, and the reversal lands in years when the current management may be gone.

**The intuition: capitalization is the only manipulation that improves profit, assets, and operating cash flow simultaneously — which is why it is the most-used balance-sheet trick, and why the check has to be on capital expenditure, not just cash flow.**

The detection signal: capital expenditure rising much faster than revenue, especially in a business whose asset intensity should be stable; capex persistently and dramatically above depreciation; and a rising balance of "capitalized software" or "deferred costs" with no corresponding growth in the business those costs supposedly support.

WorldCom ran precisely this play at scale. Between 2001 and the first quarter of 2002 it transferred approximately **\$3.852 billion** of "line costs" — the fees it paid other carriers for network access, an ordinary operating expense — into asset accounts. The SEC's complaint alleged that this overstated income before taxes by about **\$3.055 billion in 2001 and \$797 million in the first quarter of 2002**. There was no forged invoice and no fake customer. Real costs, really incurred, were simply filed in the wrong drawer.

### Goodwill: the price of optimism

**Goodwill** arises only from acquisitions. When Company A buys Company B, it allocates the purchase price across B's identifiable assets and liabilities at fair value. Whatever it paid above that total becomes goodwill — an asset representing, in principle, the value of things that cannot be separately identified and sold: reputation, workforce, expected synergies.

In practice, goodwill is the accounting record of an opinion: **this is how much more than the parts we thought the whole was worth.**

Before 2001 in the US, goodwill was amortized — written down on a schedule over up to 40 years, guaranteeing it eventually disappeared. SFAS 142 (and its IFRS counterpart, IFRS 3 with IAS 36) replaced amortization with **annual impairment testing**. Goodwill now sits on the balance sheet indefinitely at full value, and is reduced only when management concludes the acquired business is worth less than its carrying amount.

Read that sentence again with the incentives in mind. The test that determines whether an asset stays on the books is a discounted cash flow model of the acquired business, built by the executives who approved the acquisition, whose reputations and often whose compensation are tied to the acquisition having been a good idea. Recognizing an impairment is a public admission of having overpaid.

The predictable result is that goodwill impairments arrive **late, in bulk, and usually after a change of management or a share-price collapse that makes denial impossible**. They are lagging indicators, not warnings.

- AOL Time Warner wrote off roughly **\$99 billion** of goodwill across 2002 — a **\$54 billion** charge in the first quarter and a further **\$45.5 billion** in the fourth — contributing to a full-year net loss of **\$98.7 billion**, the largest annual loss in US corporate history at the time. The merger that created it had closed only two years earlier.
- Hewlett-Packard wrote down **\$8.8 billion** in November 2012 on its acquisition of Autonomy, bought for more than \$11 billion barely a year before. HP attributed roughly \$5 billion of that write-down to alleged accounting improprieties at Autonomy, including — fittingly — aggressive revenue classification and aggressive capitalization of costs.

#### Worked example: what an unimpaired goodwill balance really is

Northwind carries \$1,200m of goodwill against equity of \$3,000m. Two numbers make the exposure legible.

**Goodwill as a share of equity:** 1,200 ÷ 3,000 = **40%**. Two-fifths of the shareholders' stated stake consists of an unamortized acquisition premium.

**Tangible book value** = equity − goodwill − other intangibles = 3,000 − 1,200 = **\$1,800m**. The company's book value net of the things that cannot be sold separately is 60% of its stated book value.

Now impair half of it:

| | Before | After a \$600m impairment |
| --- | ---: | ---: |
| Goodwill | 1,200 | 600 |
| Total assets | 7,000 | 6,400 |
| Total liabilities | 4,000 | 4,000 |
| Total equity | 3,000 | **2,400** |
| Debt-to-equity | 1.33 | **1.67** |

Equity falls 20% and leverage rises from 1.33 to 1.67 — all in a single non-cash line. If Northwind's bank covenant caps net debt to equity at 1.50, the impairment alone puts the company in technical default, without a dollar of cash having left the building. This is why goodwill impairments so often coincide with refinancing crises: the write-down does not cause the distress, but it converts distress into a covenant breach.

**The intuition: goodwill that has never been impaired is not evidence that acquisitions worked. It is evidence that nobody has yet been forced to say they didn't.**

Carillion, the UK construction and services group, is the canonical illustration. Its 2016 accounts carried goodwill of about **£1.57 billion** — an amount **more than double its £730 million of net assets**, accumulated from acquisitions including Mowlem (£431 million of goodwill), Alfred McAlpine (£615 million) and Eaga (£329 million). Management concluded no impairment was required. Strip the goodwill out and Carillion's tangible net assets were deeply negative. Four months after its auditors signed a clean opinion on those accounts in March 2017, the company announced an **£845 million** provision against its construction contracts, rising to **£1,045 million** by September. It went into compulsory liquidation in January 2018.

### The quiet ones: deferred charges, capitalized software, deferred tax assets

Three smaller lines that follow the same logic and attract far less scrutiny.

**Deferred charges** (or *deferred costs*) are costs paid now and spread over future periods: debt issuance costs, contract acquisition costs, pre-opening expenses. Legitimate in themselves, they are attractive to a stretched company precisely because the standards leave room and the balances are individually small.

**Capitalized software development** is capitalizable once a project reaches "technological feasibility" (US GAAP) or meets the recognition criteria of IAS 38 (IFRS). Both tests hinge on management's assertion about a project's viability. A software company that suddenly capitalizes a much larger share of its development spend has not become more efficient; it has reclassified engineers' salaries from expense to asset. The signal is capitalized development as a percentage of total R&D spend, tracked over time.

**Deferred tax assets** represent future tax savings from past losses or timing differences. They are an asset only if the company will earn enough future profit to use them, which requires a forecast of profitability. A **valuation allowance** must reduce the asset if realization is not more likely than not. A loss-making company carrying a large, un-allowanced deferred tax asset is asserting confidence in a profitable future that its own income statement contradicts — and the reversal, when it comes, is a large non-cash hit to equity.

## The accounts that hide

The flattering accounts inflate equity by overstating an asset. The hiding accounts achieve the same result from the other side: they keep a liability off the page entirely. The effect on equity is identical, and the detection problem is much harder, because you are looking for something that is not there.

### Contingent liabilities: the footnote that owns the company

A **contingent liability** is a potential obligation whose existence depends on a future event: a lawsuit, a guarantee of someone else's debt, a tax dispute, an environmental remediation order, a product warranty claim.

The accounting is a three-way sort, and the boundaries are words:

| Assessment | US GAAP (ASC 450) | IFRS (IAS 37) | Treatment |
| --- | --- | --- | --- |
| Probable and estimable | "Probable" — read in practice as a high threshold | "More likely than not", explicitly above 50% | **Accrue** as a liability on the balance sheet |
| Reasonably possible | Disclose in the footnotes | Disclose in the footnotes | **Footnote only** — no balance-sheet line |
| Remote | No disclosure required | No disclosure required | **Nothing** |

Two things are worth noticing. First, the difference between "a liability" and "a footnote" is a single adjective, applied by management. Second, the two standards do not use the same threshold: IFRS spells out more-likely-than-not, while US GAAP's "probable" is conventionally read as a considerably higher bar. The same lawsuit can be a balance-sheet liability in Frankfurt and a footnote in New York.

The practical instruction is simple and almost nobody follows it: **read the commitments-and-contingencies footnote before you read the balance sheet.** It contains the obligations the balance sheet has decided not to show you, and in a company heading for trouble it is where the trouble is first visible in writing.

Related items that live in the same footnote territory: **guarantees** of subsidiary or joint-venture debt; **letters of credit and performance bonds**; **purchase commitments** (a contracted obligation to buy a minimum quantity for years); and **pension obligations**, where the *funded status* — plan assets minus the projected benefit obligation — is what matters and the balance-sheet presentation may show only a net figure whose components depend heavily on the assumed discount rate.

### Off-balance-sheet vehicles: three percent outside the line

The most systematic way to hide a liability is to move it into a company you control but do not have to consolidate.

**Consolidation** is the rule that a parent company's financial statements must include the assets and liabilities of the entities it controls, added line by line as if they were one company. The boundary of what gets included is *the consolidation line*, and everything the accounting rules place outside it is invisible on the balance sheet.

Before FIN 46 in 2003, US practice held that a **special purpose entity (SPE)** — a company created for a single narrow transaction — could stay off a sponsor's balance sheet if an independent third party held at least **3%** of its capital, genuinely at risk. Three percent. A vehicle could hold a billion dollars of debt and remain wholly invisible to the sponsor's shareholders provided thirty million of genuinely outside money sat in it.

Enron built its accounting around that threshold. The most instructive example is not the famous LJM partnerships but the duller one: **Chewco Investments**, an entity used to hold an outside interest in a joint venture called **JEDI**. Chewco was supposed to have independent outside equity. It did not — the supposedly at-risk outside capital was backed by cash collateral that Enron itself provided, which meant the 3% was never genuinely at risk and the entity should have been consolidated from the start.

![A dashed consolidation boundary with Enron inside and the Chewco/JEDI special purpose entity outside, holding \$711m of 1997 debt; the November 2001 restatement pulls the entity inside the line.](/imgs/blogs/reading-the-balance-sheet-what-companies-hide-here-8.webp)

On 8 November 2001, Enron filed a Form 8-K announcing it would restate its accounts back to 1997. Consolidating JEDI and Chewco alone **increased reported debt by \$711 million for 1997, \$561 million for 1998, \$685 million for 1999 and \$628 million for 2000**, and **reduced reported net income by \$28 million, \$133 million, \$153 million and \$91 million** in those same years.

Notice the asymmetry in those numbers. The debt effect is five to twenty-five times the income effect. The vehicles were not primarily an earnings machine — they were a **leverage-hiding machine**. Enron's business needed enormous amounts of debt, and its credit rating could not survive showing it. That is the general pattern: off-balance-sheet structures are usually about the liability side, and an analyst watching only the income statement will not see them at all.

The rules tightened substantially — FIN 46/ASC 810 in the US moved the test from a numerical equity threshold to a control-and-risk analysis, and IFRS 10 did the same internationally. But the underlying pressure has not gone anywhere. Modern equivalents include securitization vehicles, joint ventures accounted for by the **equity method** (where the investment appears as a single asset line and the venture's debt appears nowhere), **supply chain finance** and reverse factoring programs (which can convert bank debt into trade payables), and receivables sold to a conduit.

The check is always the same question: **what obligations does this company have that its balance sheet is not showing?** Read the footnote on unconsolidated entities. Compare the parent's share of joint-venture debt against its own. Look for a payables balance that has grown far faster than cost of goods sold — the classic signature of a reverse-factoring program financing the company through its suppliers.

### The operating-lease legacy: 3.3 trillion dollars in the footnotes

Until very recently, one of the largest liabilities in the corporate world was, by design, not on any balance sheet.

Under the old standards, leases were sorted into two types. A **finance lease** (or capital lease) was economically a purchase funded by debt, and both the asset and the liability went on the balance sheet. An **operating lease** was treated as a rental: no asset, no liability, just an annual rent expense, with future commitments disclosed in a footnote.

The distinction rested on bright-line tests — was the lease term at least 75% of the asset's useful life, was the present value of payments at least 90% of its fair value — and structuring around bright lines is what corporate finance departments are for. A 74%-of-life lease was off balance sheet. A 76% lease was on it. Airlines, retailers, restaurant chains and hotel groups built their entire capital structures on the correct side of that line.

The scale of what this concealed is genuinely hard to hold in your head. In January 2016, when it published IFRS 16, the IASB estimated that listed companies using IFRS or US GAAP had around **US\$3.3 trillion of lease commitments, over 85% of which did not appear on their balance sheets.**

IFRS 16 and ASC 842, effective for most companies from 2019, largely ended it: lessees now recognize a **right-of-use asset** and a corresponding **lease liability** for essentially all leases. Reported debt at lease-heavy companies jumped overnight, with no change to the underlying business.

Three reasons this history still matters. First, **any comparison across the 2019 boundary is broken** — a retailer's leverage in 2017 and 2021 are not the same measurement, and a five-year trend chart that ignores this is meaningless. Second, it is the clearest demonstration available that *the absence of a liability from the balance sheet says nothing about whether the obligation exists*; \$3.3 trillion of real, contractual, enforceable commitments were invisible and entirely legal. Third, the structuring instinct did not retire with the standard — it moved to service contracts, take-or-pay arrangements, and other commitments that fall outside the lease definition.

### Window dressing: the balance sheet is dated, and the date is chosen

Return to the first figure. The balance sheet describes one instant. The instant is known months in advance. Everyone involved has both the motive and the means to make that instant unrepresentative.

The polite version is routine and mostly legal: chase collections hard in the last two weeks of December, defer supplier payments to early January, delay discretionary purchases, draw down a revolver on 2 January rather than 28 December. Cash looks higher, payables look lower, leverage looks better, and by mid-January everything is back where it was.

The impolite version is **Repo 105**, and it is worth understanding in mechanical detail because it shows how far the technique can be pushed.

A **repurchase agreement (repo)** is a short-term secured loan: you sell securities for cash and agree to buy them back days later at a slightly higher price. Economically it is borrowing. Accounting-wise it is normally treated as a *financing* transaction — the securities stay on your balance sheet and a liability appears alongside the cash.

But under the accounting standard then in force, if the securities transferred were worth *materially more* than the cash received — say 105% of it — the transfer could be accounted for as a **true sale** rather than a financing. The securities left the balance sheet. The cash came in. And because the transaction was a "sale," no liability was recorded. Lehman Brothers used the incoming cash to pay down other liabilities, shrinking the balance sheet on both sides, then reversed the whole thing seven to ten days later — after the reporting date had passed.

According to the examiner's report by Anton Valukas, published in March 2010, Lehman removed approximately **\$39 billion** from its balance sheet at the end of the fourth quarter of 2007, **\$49 billion** at the end of the first quarter of 2008, and about **\$50 billion** at the end of the second quarter of 2008. The Q2 2008 transactions improved Lehman's reported net leverage ratio from **13.9 to 12.1**.

Nothing about Lehman's economic position changed. Fifty billion dollars of assets and the borrowings against them were temporarily elsewhere on the one day of the quarter when a picture was taken.

The detection tools are limited but real. Some jurisdictions and some disclosures require **average** balances alongside period-end balances; a large and persistent gap between the two is the signature. Quarterly balance sheets that oscillate — leverage low at each reporting date, higher in between where you can infer it — tell the same story. And a company whose year-end cash balance is spectacular while its average cash balance is thin is showing you a photograph, not a life.

### Related-party balances: the counterparty is you

A **related party** is anyone on both sides of a transaction with the company: a subsidiary, an affiliate, an entity controlled by an executive or a major shareholder, a family member.

Related-party balances are dangerous because the entire evidentiary basis of a balance sheet is that assets and liabilities represent claims against *independent* third parties who will confirm them. Once the counterparty is controlled by the same people as the company, a receivable is no longer a claim on the outside world — it is a claim on yourself, and it can be created, revalued, or extinguished at will.

The patterns to look for in the related-party footnote:

- **Receivables from related parties that grow and never settle.** A genuine trade receivable turns into cash within 30–90 days. One that grows every year and never converts is a loan, a disguised distribution, or a placeholder for something that does not exist.
- **Revenue concentrated in related entities.** Sales to yourself are not sales.
- **Assets acquired from related parties at prices set without an independent valuation.** The purchase price becomes the carrying value, and the carrying value becomes an "objective" number on the balance sheet.
- **Guarantees given to related entities.** These sit in the contingency footnote and can exceed the company's equity.

Wirecard is the modern archetype. A large share of its reported profit came from "third-party acquiring" partners in Asia, and the cash those partners supposedly generated was reported as sitting in **trust accounts** — controlled not by Wirecard directly but by a third-party trustee, at banks with which Wirecard had no direct relationship. Every layer of that structure put distance between the asset and anyone who could independently verify it. When EY finally insisted on a direct confirmation from the banks, in June 2020, the structure collapsed within days.

### "Other": the drawer nobody opens

Every balance sheet has lines called "other assets", "other current assets", "other liabilities", "prepaid expenses and other". They exist for a legitimate reason: aggregating genuinely immaterial items keeps the statement readable.

They are also where things go that nobody wants named, and they attract no analytical attention at all because the label promises there is nothing to see.

The test is a ratio, tracked over time: **"other" as a percentage of total assets.** Northwind's other assets were \$300m against \$7,000m of total assets in Year 1 — **4.3%**. By Year 3 they are \$850m against \$8,350m — **10.2%**. Other assets grew 183% while total assets grew 19%.

There is no innocent explanation for a residual category becoming one of your largest balances. A category defined as "everything too small to name individually" cannot legitimately grow to a tenth of the company. Either the disclosure is inadequate — items are being aggregated that materiality rules require to be broken out — or something is being parked there.

The same logic applies to any line whose *name does not tell you what it is*: "deferred charges", "sundry debtors", "amounts recoverable on contracts", "accrued income". Each is legitimate somewhere. Each is also a good place to keep something you do not want a reader to ask about. The general rule: **the analytical attention a line receives is inversely proportional to how vague its name is, and manipulators know this.**

## Solvency versus liquidity

Everything so far has been about whether the numbers are true. This section is about a distinction that matters even when every number is honest — and it is the distinction that kills companies.

### Two different questions

**Solvency** asks: *are the assets worth more than the liabilities?* It is a question about the balance sheet as a whole, and its answer is positive or negative equity.

**Liquidity** asks: *will cash be available when the obligations fall due?* It is a question about the *calendar*, and the balance sheet answers it only indirectly, because the balance sheet has just two time buckets — under one year and over one year — and a bill due next Tuesday and a bill due in eleven months sit in the same bucket.

![A two-by-two matrix of solvency against liquidity, highlighting the solvent-but-illiquid quadrant where a company with \$3,000m of equity and a quick ratio of 1.14 still faces a \$352m cash shortfall within 45 days.](/imgs/blogs/reading-the-balance-sheet-what-companies-hide-here-9.webp)

The quadrant that surprises people is the top-left. **Solvent and illiquid companies die faster than insolvent and liquid ones.** A company with negative equity but reliable incoming cash can trade for years — plenty of leveraged buyouts and turnarounds run with negative book equity indefinitely. A company with abundant equity and no cash on the day a loan matures is in default, and default triggers cross-default clauses in every other credit agreement, and the whole structure unwinds in weeks.

Insolvency is a condition. Illiquidity is an event, and events have dates.

#### Worked example: solvent on paper, out of cash in 45 days

Northwind at the end of Year 3. Here is the balance sheet in full:

| Assets | \$m | Liabilities and equity | \$m |
| --- | ---: | --- | ---: |
| Cash | 150 | Accounts payable | 700 |
| Accounts receivable | 1,560 | Accrued expenses | 300 |
| Inventory | 1,600 | Short-term debt | 500 |
| Prepaid expenses | 90 | **Total current liabilities** | **1,500** |
| **Total current assets** | **3,400** | Long-term debt | 3,850 |
| PP&E, net | 2,900 | **Total liabilities** | **5,350** |
| Goodwill | 1,200 | Total equity | 3,000 |
| Other assets | 850 | | |
| **Total assets** | **8,350** | **Total liabilities and equity** | **8,350** |

Run the standard checks and the company passes:

- **Equity \$3,000m, positive.** Assets exceed liabilities by three billion dollars. Solvent.
- **Working capital** = 3,400 − 1,500 = **\$1,900m**, *up* from \$1,000m in Year 1. Improving, apparently.
- **Current ratio** = 3,400 ÷ 1,500 = **2.27**. Comfortably above any textbook threshold.
- **Quick ratio** = (150 + 1,560) ÷ 1,500 = **1.14**. Above 1.0, which the textbooks call safe.

Now build a calendar instead of a ratio. What must Northwind pay in the next 45 days?

| Obligation | \$m |
| --- | ---: |
| Trade payables falling due within 45 days | 620 |
| Short-term debt maturing on day 40 | 500 |
| Payroll and rent (1.5 months at \$80m) | 120 |
| Semi-annual interest payment on long-term debt | 140 |
| **Total due within 45 days** | **1,380** |

And what will arrive?

Cash on hand is \$150m. Collections come from the \$1,560m receivable balance — but at a DSO of 80 days, that balance converts at roughly 1,560 ÷ 80 = **\$19.5m per day**. Over 45 days: 45 × 19.5 = **\$878m**. Nothing sold today will be collected inside the window, because the collection cycle is 80 days long.

Total available: 150 + 878 = **\$1,028m**. Total required: **\$1,380m**.

**Shortfall: \$352m.**

The \$1,600m of inventory is no help. At a DIO of 119 days, it is nearly four months from becoming cash, and dumping it at a discount would both crater the margin and confirm to the market that the company is in trouble. The \$1,200m of goodwill is no help — it cannot be sold at all. The \$2,900m of plant is no help on a 45-day horizon.

A company with \$3,000m of equity, a current ratio of 2.27 and a quick ratio above 1.0 cannot pay its bills next month.

**The intuition: solvency is measured in dollars, liquidity is measured in days, and the balance sheet reports dollars. You have to supply the calendar yourself.**

### The cash conversion cycle

The tool that converts the balance sheet into days is the **cash conversion cycle (CCC)** — how long a dollar is tied up between paying a supplier and collecting from a customer:

$$\text{CCC} = \text{DIO} + \text{DSO} - \text{DPO}$$

where **DPO** (days payable outstanding) = accounts payable ÷ cost of goods sold × 365, the average time the company takes to pay its own suppliers.

For Northwind in Year 3, with cost of goods sold of \$4,900m:

- DIO = 1,600 ÷ 4,900 × 365 = **119 days**
- DSO = 1,560 ÷ 7,100 × 365 = **80 days**
- DPO = 700 ÷ 4,900 × 365 = **52 days**
- **CCC = 119 + 80 − 52 = 147 days**

Every dollar of Northwind's growth requires funding 147 days of working capital. That is why a company can grow revenue, report profits, and run out of money simultaneously — a phenomenon with the deceptively cheerful name **overtrading**. Growth consumes cash before it produces it, and the faster the growth, the larger the hole.

Compare with a business at the other extreme. A supermarket sells inventory in about 20 days, collects instantly at the till (DSO near zero) and pays suppliers in 45 days: a CCC of roughly 20 + 0 − 45 = **−25 days**. It is financed by its own suppliers, generates cash as it grows, and can operate with a current ratio well below 1.0 in perfect safety.

**The same current ratio means opposite things at those two companies.** This is why balance-sheet ratios must always be read against the industry and against the company's own history, never against a universal threshold.

## The forensic reading: what to actually compute

Here is the practical routine. None of it requires anything beyond the published statements and a spreadsheet, and all of it works on a five-year series — a single year tells you almost nothing, because the entire method is about *divergence over time*.

### The ratio pairs that matter

Each of these pairs a balance-sheet line against the flow that should drive it. A stable ratio is unremarkable; a diverging one is the signal.

| Ratio | Formula | What divergence means |
| --- | --- | --- |
| **DSO** | AR ÷ revenue × 365 | Rising: sales booked that cash is not following — bad debt, channel stuffing, or fiction |
| **DIO** | Inventory ÷ COGS × 365 | Rising: obsolete stock carried at cost, or overhead being absorbed into unsold units |
| **DPO** | AP ÷ COGS × 365 | Sharply rising: cash stress, or supply-chain financing being used as hidden debt |
| **Capex ÷ depreciation** | — | Persistently above ~1.5× without expansion: costs being capitalized |
| **Capitalized software ÷ total R&D** | — | Rising: development salaries reclassified from expense to asset |
| **Goodwill ÷ equity** | — | Above ~30–40%: a large share of book value is untested acquisition premium |
| **"Other assets" ÷ total assets** | — | Rising above ~5%: inadequate disclosure or a parking space |
| **Accumulated depreciation ÷ gross PP&E** | — | Rising toward 70%+: aging asset base, capex cliff coming |
| **Allowance ÷ gross receivables** | — | Falling while DSO rises: the reserve is being under-provisioned |
| **Cash conversion cycle** | DIO + DSO − DPO | Lengthening: growth is consuming more cash per dollar of sales |

### A short checklist

Six questions, in the order I would ask them.

1. **Are the estimate-heavy assets growing faster than the business?** Compute the growth rate of receivables, inventory, capitalized costs and "other" against revenue growth. Anything growing at more than about 1.5× revenue needs an explanation, and the explanation should be in the footnotes.
2. **Has goodwill ever been impaired?** If a company has made a decade of acquisitions and never once written any of it down, it is asserting a perfect acquisition record. Almost no acquirer has one. Compute tangible book value — equity minus goodwill minus intangibles — and see whether it is positive.
3. **What is in the contingencies footnote?** Read it before the balance sheet. Lawsuits, guarantees, purchase commitments, tax disputes. Compare the total against equity.
4. **How much is off balance sheet?** Unconsolidated entities, joint ventures accounted for by the equity method, securitizations, supply-chain finance, long-term purchase obligations. The commitments table in the annual report is the starting point.
5. **What does the calendar look like, not the ratio?** Build a 90-day cash schedule: what falls due, what will actually be collected given the DSO, what facilities are committed and undrawn. Ratios average away timing; timing is what defaults on.
6. **Do the three statements agree?** Reconcile the balance-sheet movement in each working-capital line against the corresponding line in the cash flow statement. A receivables balance that grows \$400m while the cash flow statement shows a \$150m working-capital drag on receivables means the difference went somewhere — an acquisition, a reclassification, or a sale of receivables that should have been disclosed.

That last check is the most powerful and the least used. It is worked through in detail in [reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

## Common misconceptions

**"If it balances, the numbers are right."** The two sides are equal by definition, because equity is calculated as assets minus liabilities, and mechanically guaranteed by double-entry bookkeeping. A balance sheet containing a billion dollars of imaginary cash balances perfectly. Satyam's did.

**"Assets are things the company owns and therefore facts."** Most asset lines are estimates of future benefit. Receivables assume collection, inventory assumes saleability, net PP&E assumes a useful life, goodwill assumes an acquisition thesis still holds. Only cash — and only cash that has been directly confirmed with the bank — approaches being a fact.

**"A clean audit opinion means the balance sheet is right."** An audit provides reasonable, not absolute, assurance that the statements are free of material misstatement. It is sample-based, and for the most judgmental lines it largely tests whether management's estimates fall within a defensible range using management's own models. Carillion received a clean opinion in March 2017 on accounts carrying £1.57 billion of unimpaired goodwill against £730 million of net assets; the company was in liquidation ten months later.

**"A high current ratio means the company is safe."** The current ratio treats a receivable collectible in 80 days and cash in the bank as equivalent, and treats a bill due tomorrow and one due in eleven months as equivalent. Northwind's Year 3 current ratio is 2.27 and it is \$352m short inside 45 days. Liquidity is about dates; ratios erase dates.

**"Book value tells you what the company is worth."** Book value is historical cost less depreciation, plus whatever acquisition premiums have not yet been written off. It excludes internally created brands, technology and customer relationships entirely, and includes goodwill that may be worthless. It is an accounting residual, not a valuation.

**"Off-balance-sheet means small or exotic."** Around US\$3.3 trillion of lease commitments sat off balance sheets globally as recently as 2016, per the IASB's own estimate — entirely legally, at ordinary retailers and airlines. Off-balance-sheet is the normal condition of a great many real obligations.

**"Non-cash charges don't matter."** A goodwill impairment moves no cash, but it reduces equity, which raises leverage ratios, which can breach covenants, which can accelerate debt. Non-cash accounting entries cause very cash-like consequences.

## How it shows up in real markets

Six cases, each isolating one mechanism from this article.

### Satyam, 2009: the cash line was fake

On 7 January 2009, B. Ramalinga Raju, chairman of the Indian IT services company Satyam Computer Services, wrote to the board and to India's securities regulator confessing that the balance sheet as of 30 September 2008 carried **inflated, non-existent cash and bank balances of Rs 5,040 crore** against Rs 5,361 crore stated in the books — roughly one billion US dollars at the exchange rate of the time. The letter also disclosed non-existent accrued interest of Rs 376 crore, an understated liability of Rs 1,230 crore, and an overstated debtor position of Rs 490 crore (Rs 2,651 crore recorded against Rs 2,161 crore actual).

The income statement had been inflated to match: for the September 2008 quarter the company reported revenue of Rs 2,700 crore and an operating margin of Rs 649 crore (24% of revenue), against actual revenue of Rs 2,112 crore and an actual operating margin of Rs 61 crore — **3%**. Raju's own description of the mechanism has become the most quoted line in the case: it was "like riding a tiger, not knowing how to get off without being eaten."

The lesson is the one from the certainty ladder. The fake was placed on rung 1, the line every reader treats as verified. Raju later disavowed the letter through his lawyers, but the underlying fraud was never in dispute; Satyam was sold to Tech Mahindra in 2009 and rebranded.

### WorldCom, 2002: an expense in the wrong drawer

WorldCom transferred approximately **\$3.852 billion** of line costs — payments to other carriers for network access, an unambiguous operating expense — into capital asset accounts across 2001 and the first quarter of 2002. The SEC alleged this overstated income before income taxes and minority interests by about **\$3.055 billion in 2001** and **\$797 million in the first quarter of 2002**.

What makes WorldCom the definitive teaching case is how mundane the mechanism was. No shell companies, no offshore vehicles, no forged confirmations. Real costs, really incurred, filed under the wrong heading. The balance sheet grew, profit appeared, and — because capitalized costs move cash outflow from operating to investing — operating cash flow looked *better* while the fraud ran. The internal audit team found it by examining capital expenditure accounts and asking what the assets actually were.

### Enron, 2001: the leverage that was three percent outside

Enron's restatement announced on 8 November 2001 consolidated the Chewco and JEDI entities retroactively to 1997. The effect on reported debt was **+\$711 million (1997), +\$561 million (1998), +\$685 million (1999) and +\$628 million (2000)**; the effect on reported net income was **−\$28 million, −\$133 million, −\$153 million and −\$91 million** respectively.

The debt effect dwarfed the income effect by an order of magnitude, and that ratio is the lesson. The structures existed to keep leverage invisible so the credit rating would survive; the earnings benefit was incidental. An analyst tracking only earnings quality would have seen very little. An analyst asking "what does this company owe that I cannot see?" would have been looking in the right place.

### Lehman Brothers, 2008: a photograph, retouched

Lehman's Repo 105 transactions removed roughly **\$50 billion** from its balance sheet at the end of the second quarter of 2008 — following about \$39 billion at the end of Q4 2007 and \$49 billion at the end of Q1 2008 — by structuring short-term secured borrowings so they qualified as sales, then reversing them seven to ten days later. The Q2 2008 transactions improved the reported net leverage ratio from **13.9 to 12.1**. The mechanism was documented in the bankruptcy examiner's report published in March 2010.

No counterparty was deceived about the economics; the repo lenders knew exactly what these were. The audience for the deception was the reader of the quarterly balance sheet, and the deception worked because the balance sheet reports one day. It is the strongest available argument for demanding average balances rather than period-end balances from any leveraged financial institution.

### Carillion, 2018: goodwill that was never wrong

Carillion's 2016 accounts carried about **£1.57 billion** of goodwill — more than double its **£730 million** of net assets — built up through acquisitions including Mowlem (£431 million), Alfred McAlpine (£615 million) and Eaga (£329 million). None of it had been impaired. Tangible net assets, once goodwill was removed, were substantially negative.

Auditors signed a clean opinion in March 2017. In July 2017 the company announced an **£845 million** contract provision, raised to **£1,045 million** in September. It entered compulsory liquidation in January 2018, taking down suppliers and subcontractors across the UK and triggering a parliamentary inquiry into audit quality and the goodwill regime itself.

Two mechanisms compounded here. Goodwill that had never been tested against reality made the balance sheet look far stronger than it was, and aggressive recognition on long-term construction contracts — an income-statement problem — piled up as receivables and "amounts recoverable on contracts" that eventually had to be written off. The forensic signal was available years earlier in a single ratio: goodwill at more than 200% of net assets, never impaired.

### Wirecard, 2020: an asset nobody could confirm

Wirecard reported **€1.9 billion** of cash held in escrow accounts at two Philippine banks, held on its behalf by a third-party trustee, arising from "third-party acquiring" partnerships in Asia. EY spent months in 2020 attempting to obtain direct confirmation from the banks and received copies, scans and documents routed through intermediaries instead. On 18 June 2020 EY declined to sign off the 2019 accounts. Both banks — BDO Unibank and the Bank of the Philippine Islands — stated that the confirmation documents were forgeries and that they had never held an account for Wirecard. Wirecard filed for insolvency on 25 June 2020.

Everything about the structure was a warning that was visible in advance: the profit came from related third parties, the cash sat with a trustee rather than the company, and it sat at banks with which the company had no direct relationship. Each layer put distance between the asset and independent verification. The single most valuable question a reader of that balance sheet could have asked is the same question that broke it: **who, exactly, can independently confirm that this asset exists?**

## When this matters and where to go next

You will use this most often in three situations, and only one of them involves fraud.

**Reading a company you might invest in.** The routine is the five-year ratio table above. Most of what it finds is not fraud but strain — a business whose growth is consuming more cash each year, whose receivables are lengthening because customers are struggling, whose goodwill records an acquisition that plainly did not work. Strain is far more common than fraud and considerably more actionable, because it is visible early.

**Reading a company you work for, or supply, or lend to.** The liquidity calendar matters more than any ratio here. A supplier's solvency is not what determines whether your invoice gets paid in March; their cash conversion cycle and their debt maturity schedule are. Carillion's subcontractors learned this in a way no ratio would have taught them.

**Reading the news.** When a company announces a large non-cash charge, you now know what happened: an estimate that had been optimistic for years was finally reset, equity fell by the full amount, and leverage rose. When a company restates, you know where to look for the residue.

The one habit worth building above all others is this: **whenever you see an asset, ask what future event has to happen for it to be worth what it says.** Cash requires nothing. A receivable requires a customer to pay. Inventory requires someone to want it. Goodwill requires an entire acquired business to perform as forecast. Written that way, the asset side of a balance sheet stops looking like a list of possessions and starts looking like what it is — a portfolio of predictions, sorted from safest to most speculative, with the shakiest ones usually the largest.

To go deeper into the mechanics behind each of these lines: [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) explains why the estimate layer exists at all, [reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings) covers the revenue and expense decisions that create these balances, and [the three financial statements and how they interlock](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock) shows how to reconcile all three so that a manipulation in one becomes visible in another.

This article is educational. It explains how balance sheets are constructed and how they have been misused; it is not investment advice and does not recommend any security.

## Sources & further reading

**Primary filings and official reports**

- Enron Corp., Form 8-K filed 8 November 2001 (restatement of 1997–2000 accounts; consolidation of Chewco and JEDI) — [SEC EDGAR](https://www.sec.gov/Archives/edgar/data/1024401/000095012901503835/h91831e8-k.txt)
- SEC Litigation Release No. 17588, *SEC v. WorldCom, Inc.* (line-cost capitalization; \$3.055bn in 2001 and \$797m in Q1 2002) — [SEC.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-17588)
- Congressional Research Service, *WorldCom: The Accounting Scandal*, RS21253 (August 2002) — [EveryCRSReport](https://www.everycrsreport.com/reports/RS21253.html)
- B. Ramalinga Raju, resignation and confession letter to the Satyam board, 7 January 2009, filed with the SEC as Exhibit 99.2 — [SEC EDGAR](https://www.sec.gov/Archives/edgar/data/1106056/000114554909000025/u00107exv99w2.htm)
- House of Commons Business, Energy and Industrial Strategy and Work and Pensions Committees, *Carillion*, HC 769 (May 2018) — [publications.parliament.uk](https://publications.parliament.uk/pa/cm201719/cmselect/cmworpen/769/769.pdf)
- IFRS Foundation, *IFRS 16 Leases: Effects Analysis* (January 2016), and the accompanying announcement estimating US\$3.3 trillion of lease commitments with over 85% off balance sheet — [ifrs.org effects analysis](https://www.ifrs.org/content/dam/ifrs/project/leases/ifrs/published-documents/ifrs16-effects-analysis.pdf) and [IASB announcement](https://www.ifrs.org/news-and-events/news/2016/01/iasb-shines-light-on-leases-by-bringing-them-onto-the-balance-sheet/)
- European Parliament, *Banking Union: Wirecard* briefing (2020) — [europarl.europa.eu](https://www.europarl.europa.eu/RegData/etudes/BRIE/2020/651357/IPOL_BRI(2020)651357_EN.pdf)

**Reporting and analysis**

- Anton R. Valukas, *Report of the Examiner in the Chapter 11 proceedings of Lehman Brothers Holdings Inc.* (March 2010) — Repo 105 volumes and the 13.9-to-12.1 net leverage effect; summarized at [Knowledge at Wharton](https://knowledge.wharton.upenn.edu/article/lehmans-demise-and-repo-105-no-accounting-for-deception/) and [NPR Planet Money](https://www.npr.org/sections/money/2010/03/repo_105_lehmans_accounting_gi.html)
- CNN Money, "HP takes \$8.8 billion writedown on Autonomy" (20 November 2012) — [money.cnn.com](https://money.cnn.com/2012/11/20/technology/enterprise/hp-earnings/index.html)
- CFO.com, "AOL Time Warner Reports \$100 Billion Loss" (January 2003) — the \$54bn Q1 and \$45.5bn Q4 2002 goodwill charges and the \$98.7bn full-year loss — [cfo.com](https://www.cfo.com/news/aol-time-warner-reports-100-billion-loss/681215/)
- Accountancy Age, "EY's Wirecard audit exposes potential fraud" (22 June 2020) — [accountancyage.com](https://accountancyage.com/2020/06/22/eys-wirecard-audit-exposes-potential-fraud/)
- Accountancy Age, "Carillion inquiry: missed red flags, aggressive accounting and the pension deficit" (26 February 2018) — [accountancyage.com](https://accountancyage.com/2018/02/26/carillion-inquiry-missed-red-lights-aggressive-accounting-pension-deficit/)

**Standards referenced**

- ASC 450 / IAS 37 — contingencies and provisions (the probable / reasonably possible / remote sort). On the differing thresholds — IAS 37's explicit "more likely than not" against US GAAP's higher "probable" bar — see RSM, *U.S. GAAP vs. IFRS: Contingencies and provisions at a glance* — [rsmus.com](https://rsmus.com/pdf/contingencies-provisions-at-a-glance.pdf)
- ASC 350 / IAS 36 — goodwill and impairment testing
- ASC 330 / IAS 2 — inventory measurement and the normal-capacity constraint on overhead absorption
- ASC 842 / IFRS 16 — leases, effective for most reporting entities from 2019
- ASC 810 (formerly FIN 46R) / IFRS 10 — consolidation and variable interest entities

All Northwind Tools figures in this article are illustrative and belong to a hypothetical company constructed to make the arithmetic legible. Every figure attributed to a named company traces to the filing, report or contemporaneous news source listed above.
