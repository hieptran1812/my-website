---
title: "How the Three Financial Statements Connect: The Integrated Model"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The income statement, balance sheet, and cash flow statement are not three separate documents — they are three views of one wired machine, and once you see the wiring you can read any company and even build a model."
tags: ["equity-research", "corporate-finance", "financial-statements", "three-statement-model", "income-statement", "balance-sheet", "cash-flow-statement", "financial-modeling", "double-entry", "valuation", "dcf", "accounting"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — The income statement, balance sheet, and cash flow statement are not three reports that happen to come stapled together; they are three windows onto one system, and every transaction a company does shows up in all of them at once. Once you can see the wires that connect them, financial statements stop being a wall of numbers and become a machine you can read — and even build.
>
> - **Net income is the master wire.** The bottom line of the income statement is copied into two places at once: it grows *retained earnings* (equity) on the balance sheet, and it is the *first line* of the cash flow statement. One number, two destinations, no double counting.
> - **Depreciation appears in all three statements.** A single non-cash charge lowers profit on the income statement, gets *added back* on the cash flow statement (because no cash actually left), and shrinks the asset on the balance sheet. One entry, three footprints.
> - **Capex builds the asset; depreciation wears it down.** The cash you spend on equipment leaves through the cash flow statement and lands on the balance sheet as PP&E, then bleeds back into the income statement slowly over years as depreciation. That is why a profitable company can be cash-poor and a cash-rich company can show thin profits.
> - **The balance sheet always balances** because every transaction has two equal-and-opposite sides — that is double-entry, and it is the reason the whole machine is self-checking. **Ending cash on the cash flow statement *is* the cash line on the balance sheet**: if those two numbers don't match to the penny, a wire is broken.
> - This wired system is the **three-statement model** that every analyst and banker builds, and it is the engine that produces the free cash flow a [DCF](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) discounts into a price. Forecasting one statement *forces* the other two — that's the whole trick.

If you have read the three earlier posts in this series — the [income statement line by line](/blog/trading/equity-research/income-statement-line-by-line-revenue-to-net-income), the [balance sheet](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth), and the [cash flow statement](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from) — you now know each of the three financial statements on its own. You can read an income statement and find net income. You can read a balance sheet and check that assets equal liabilities plus equity. You can read a cash flow statement and find out where the cash really came from.

But here is the thing nobody tells you when you're learning them one at a time: **the three statements are not three things.** They are three views of one thing. A company is a single system — money comes in, money goes out, assets get bought and worn down, debt gets raised and repaid — and the three statements are just three camera angles on that single system. The income statement watches *profitability*. The balance sheet watches *position* — what you own and owe at a frozen instant. The cash flow statement watches the *cash* moving in and out. Same company, same period, three lenses.

And because it's all one system, the three statements are *wired together*. You cannot change a number in one without it rippling into the other two. Record a sale, and revenue rises (income statement), receivables or cash rise (balance sheet), and — if you got paid — operating cash rises (cash flow statement). Buy a machine, and cash goes out (cash flow), PP&E goes up (balance sheet), and nothing happens to profit *this year* — but depreciation will nibble at profit for years to come. Every transaction touches all three. The figure below is the map of those wires, and it is the single most important picture in this entire series — internalize it and the rest of equity research becomes legible.

![The three financial statements drawn as three columns with arrows showing net income flowing to equity and to operating cash, the three cash flow sections summing to ending cash, and ending cash plus retained earnings landing on the balance sheet](/imgs/blogs/how-the-three-financial-statements-connect-1.png)

This post is about those wires. We'll name each one, trace a single transaction through all three statements until the linkages are reflexive, and then assemble the whole thing into the **three-statement model** — the integrated spreadsheet that analysts and investment bankers live inside, the one that turns a handful of assumptions into a forecast and a forecast into a valuation. By the end, you won't see three documents. You'll see one machine.

## Foundations: the vocabulary of the wiring

Before we trace anything, let's pin down the terms. You've met most of these in the earlier posts, but the wiring uses them in specific ways, so let's be precise. I'll keep each definition to the *minimum* you need to follow the linkages.

**Income statement (IS).** A *flow* statement covering a period (a quarter, a year). It starts with **revenue** (sales), subtracts **costs and expenses** — cost of goods sold (COGS), operating expenses, **depreciation and amortization (D&A)**, interest, and taxes — and ends at **net income**, the bottom line. Net income is the profit that belongs to shareholders for the period. Crucially, the income statement is built on *accrual accounting*: it records revenue when it's *earned* and expenses when they're *incurred*, not when cash changes hands. That gap between "earned" and "paid" is where half the wiring lives.

**Balance sheet (BS).** A *stock* statement — a snapshot at one instant (the last day of the period). It has three sections that obey one identity:

$$\text{Assets} = \text{Liabilities} + \text{Equity}$$

**Assets** are what the company owns or controls (cash, accounts receivable, inventory, property/plant/equipment). **Liabilities** are what it owes (accounts payable, debt). **Equity** is the residual — what's left for owners if you sold every asset and paid off every debt. The identity is not a coincidence or a rule someone made up; it's an accounting *tautology* that falls out of double-entry, which we'll get to. Because the balance sheet is a snapshot, every line on it is a *balance* — a level — not a flow.

**Cash flow statement (CFS).** A flow statement, like the income statement, but it tracks only one thing: **cash**. It has three sections. **Cash from operations (CFO)** — cash generated by running the business. **Cash from investing (CFI)** — cash spent on (or raised from) long-term assets, dominated by **capex** (capital expenditures: buying property, plant, equipment). **Cash from financing (CFF)** — cash from raising or repaying debt and from issuing or buying back stock and paying dividends. Add the three sections to the cash you started with, and you get the cash you ended with.

**Retained earnings.** A line *inside the equity section* of the balance sheet. It is the running total of every dollar of profit the company has ever earned and *not* paid out as dividends. Each period it changes by exactly one rule:

$$\text{Ending RE} = \text{Beginning RE} + \text{Net income} - \text{Dividends}$$

This single equation is the first big wire: net income (an income-statement number) flows directly into equity (a balance-sheet number) through retained earnings.

**Depreciation and amortization (D&A).** When a company buys a long-lived asset — a \$500,000 machine — it does *not* record a \$500,000 expense that year. Instead it spreads the cost over the asset's useful life as depreciation (for physical assets) or amortization (for intangibles). If the machine lasts ten years, the company records \$50,000 of depreciation a year for ten years. The key fact for the wiring: **depreciation is a non-cash expense.** It lowers profit, but no cash leaves the building when you record it — the cash left years earlier when you *bought* the asset. This non-cashness is why D&A shows up, with opposite signs, on the income statement and the cash flow statement.

**Working capital.** The short-term, day-to-day operating accounts on the balance sheet: **accounts receivable** (money customers owe you), **inventory** (goods you've made but not sold), and **accounts payable** (money you owe suppliers). When these accounts *change* between two balance sheets, the change is a cash adjustment on the cash flow statement. Receivables rising means you booked sales but haven't collected — profit without cash, so you *subtract* the increase from cash. The mechanics of this single point trip up more beginners than anything else in finance, so we'll work it slowly.

**PP&E (property, plant, and equipment).** The big physical assets — factories, machines, buildings, servers. It's reported on the balance sheet **net of accumulated depreciation**: gross cost minus all the depreciation charged so far. Net PP&E rolls forward each period by adding capex and subtracting depreciation. This is the second big wire, and it touches all three statements.

**Double-entry bookkeeping.** The 500-year-old rule that makes the whole machine self-checking: *every transaction is recorded in (at least) two places, with equal and offsetting effects.* Sell goods for cash, and cash goes up *and* equity goes up (via profit). Borrow money, and cash goes up *and* a liability goes up. Because every transaction has two equal sides, the balance sheet *cannot* go out of balance — assets and (liabilities + equity) move in lockstep. When your model doesn't balance, you didn't break double-entry; you *forgot one of the two sides* of some transaction. The balance check is your alarm bell.

With that vocabulary in place, let's meet our company and start tracing.

### Meet Northwind Industries

Throughout this post we'll follow one fictional firm, **Northwind Industries**, a small manufacturer of industrial widgets. Using one company across every example lets the numbers compound, so by the end you'll have watched a single business move through a full year. Here's Northwind at the start of the year (the **beginning balance sheet**), in thousands of dollars:

| Beginning Balance Sheet (Jan 1) | \$000s |
|---|---:|
| **Assets** | |
| Cash | 300 |
| Accounts receivable | 200 |
| Inventory | 150 |
| Net PP&E | 2,000 |
| **Total assets** | **2,650** |
| **Liabilities** | |
| Accounts payable | 120 |
| Debt | 800 |
| **Total liabilities** | **920** |
| **Equity** | |
| Common stock | 1,000 |
| Retained earnings | 730 |
| **Total equity** | **1,730** |
| **Total liabilities + equity** | **2,650** |

Check the identity: assets of \$2,650 equal liabilities of \$920 plus equity of \$1,730. It balances. Good. Now let's run Northwind through a year of transactions, tracing each one through all three statements, and at the end we'll prove the balance sheet still balances and that ending cash ties out. That proof *is* the three-statement model.

## Wire 1: net income → retained earnings *and* the top of the cash flow statement

Start with the most important wire, because it's the one most beginners get half-right. Net income — the bottom line of the income statement — goes to **two places at once.** Not one. Two.

**Destination one: retained earnings.** Profit that the company keeps (doesn't pay out as dividends) accumulates in the retained-earnings line of equity. If Northwind earns \$200 of net income and pays \$40 of dividends, retained earnings grows by \$160. This is how a profitable company's equity compounds over time — it's the mathematical heart of why owning profitable businesses builds wealth, the same engine [Warren Buffett](/blog/trading/finance/warren-buffett-berkshire-value-investing) has compounded inside Berkshire for sixty years.

**Destination two: the top of the cash flow statement.** The most common way to build a cash flow statement — the **indirect method** — *starts* at net income and then makes a series of adjustments to convert that accrual profit into actual cash. So net income is literally the first line of CFO.

Here's the part that confuses people: *isn't that double counting?* The same \$200 going into equity and into cash? No — and seeing why is the key to the whole system. Equity is a **stock** (a level); the cash flow statement is a **flow** (a change). Net income *increases* the equity level by \$200 *and* it *is* the starting point for computing the cash *change*. They're measuring different things with the same input. The figure below shows the one number splitting toward its two homes.

![Net income of two hundred shown leaving the income statement and arriving in two places, the retained earnings line of equity and the first line of cash from operations, with a note that there is no double counting because equity is a level and cash flow is a change](/imgs/blogs/how-the-three-financial-statements-connect-2.png)

#### Worked example: Northwind sells \$100 of widgets on credit

Northwind sells \$100 of widgets to a customer **on credit** — the customer will pay later. The widgets cost Northwind \$60 to make. Let's trace this single sale through all three statements.

**Income statement.** Revenue rises by \$100. COGS rises by \$60. So gross profit, and ultimately net income (ignore taxes for this micro-example), rises by **\$40**. The income statement doesn't care that no cash arrived — accrual accounting books the revenue when it's *earned*, which is now.

**Balance sheet.** Three things move. **Accounts receivable** rises by \$100 (the customer owes us). **Inventory** falls by \$60 (we shipped \$60 of goods). And **retained earnings** rises by \$40 (the profit we just earned). Check the balance: assets changed by +\$100 − \$60 = **+\$40**; equity changed by **+\$40**. The two sides moved by the same amount — the balance sheet stays balanced. That's double-entry doing its job automatically.

**Cash flow statement.** Here's the punchline. CFO starts at net income of +\$40. But we collected *no cash* — the customer owes us. So we must *remove* the non-cash profit. Receivables rose by \$100, which is a *use* of cash (we earned it but didn't collect it), so we subtract \$100. Inventory fell by \$60, which is a *source* of cash (we converted inventory we already paid for into a receivable), so we add \$60. Net cash effect: +\$40 − \$100 + \$60 = **−\$60**. Wait — *negative*? We "made" \$40 of profit and our cash went *down* \$60?

That negative number is the single most important lesson in financial-statement analysis. *Profit is not cash.* Northwind booked \$40 of profit and consumed \$60 of cash, all from one sale, because it shipped real goods (that cost \$60) in exchange for a promise to pay. Until that customer pays, the sale is a cash *drain*, not a cash *source*.

The figure below traces this one sale through all three statements as a pipeline, so you can watch the single event fan out into the three views and still leave the balance sheet in balance.

![One credit sale traced as five steps through the three statements, the event of selling one hundred of goods that cost sixty, the income statement showing revenue and profit, the balance sheet showing receivables and equity rising while inventory falls, the cash flow statement reversing the non cash profit, and a final check that the balance sheet still balances with no cash collected yet](/imgs/blogs/how-the-three-financial-statements-connect-5.png)

*The same sale makes the income statement happier and the cash flow statement sadder — and the gap between them is exactly the change in working capital.*

### Double-entry: why there are always exactly two sides

The reason every transaction touches multiple statements — and the reason the balance sheet can never drift out of balance — is **double-entry bookkeeping**, the convention Luca Pacioli codified in 1494 and that has run accounting ever since. The rule is deceptively simple: *every transaction is recorded with two equal and offsetting entries.* For each thing that goes up, something else goes up or down by the same amount. There is no such thing as a one-sided transaction.

Walk back through the credit sale with double-entry in mind. When Northwind ships \$100 of widgets that cost \$60, *four* entries fire at once, in two offsetting pairs. Pair one: revenue +\$100 (which lifts equity through profit) is matched by receivables +\$100 (an asset). Pair two: COGS +\$60 (which lowers equity through profit) is matched by inventory −\$60 (an asset). Net the equity effects and you get +\$40; net the asset effects and you get +\$40. The two sides of the balance sheet moved by the *same* \$40, because every single entry had a partner. You did not have to *check* that it balanced — it balanced by construction.

This is the deepest idea in the whole system, so let me state it as plainly as I can. **The balance sheet identity (assets = liabilities + equity) is not a law you obey; it is a tautology that double-entry guarantees.** If your three-statement model is built honestly — every flow recorded with both of its sides — the balance sheet *must* balance. When it doesn't, you haven't violated some accounting principle; you've simply *forgotten one side* of some transaction, or recorded one side twice. The imbalance is the accountant's smoke detector: it goes off precisely when a wire is missing, and the amount it's off by points you straight at the missing entry.

That's also why the three statements stay consistent with each other for free. Net income is computed once (on the income statement) and *reused* — as the change in retained earnings (balance sheet) and as the start of CFO (cash flow statement). Depreciation is computed once and *reused* with opposite signs across the three statements. Capex is recorded once and *reused* in investing and in PP&E. Nothing is entered twice independently; every number has one source of truth and flows everywhere it's needed. Break that discipline — type a number into two statements by hand instead of linking them — and your model will silently disagree with itself. The whole art of building a clean model is making every shared number flow from one place.

## Wire 2: depreciation appears in all three statements

The second wire is the one that makes people's eyes glaze, so let's make it concrete and visual. **Depreciation is a single accounting entry that shows up — with different signs and different meanings — on all three statements at once.** Watch one \$300 charge land in three places.

![One depreciation charge of fifty dollars branching to three footprints, an expense that lowers net income on the income statement, an add back that raises operating cash on the cash flow statement, and a reduction of net property plant and equipment on the balance sheet](/imgs/blogs/how-the-three-financial-statements-connect-3.png)

**On the income statement:** depreciation is an **expense**. It lowers operating profit and therefore net income. If Northwind records \$300 of depreciation this year, net income is \$300 lower than it would be otherwise.

**On the cash flow statement:** depreciation is an **add-back**. Because CFO starts at net income (which already had the \$300 subtracted) but *no cash actually left the building* when we recorded depreciation, we add the \$300 right back. The cash left years ago when we bought the asset; recording depreciation today is purely an accounting entry. So depreciation *lowers* profit but *doesn't* lower cash — and the add-back is how the cash flow statement undoes the non-cash hit.

**On the balance sheet:** depreciation *reduces net PP&E*. Each year of depreciation shrinks the book value of the asset (it accumulates in a contra-account called accumulated depreciation). The \$300 charge makes net PP&E \$300 smaller.

So the same \$300: −\$300 to profit (IS), +\$300 add-back to cash (CFS), −\$300 to net PP&E (BS). Three statements, one entry, three different faces.

This is also why **CFO usually exceeds net income** for an established, asset-heavy company: you add back a big depreciation number that depressed profit but didn't touch cash. A company with \$200 of net income and \$300 of depreciation has at least \$500 of cash from operations before any working-capital effects. That gap — CFO minus net income — is one of the first things an analyst eyeballs. A healthy, capital-intensive business has CFO comfortably above net income; a business where CFO is *chronically below* net income is one where profits aren't turning into cash, which is a classic warning sign we'll see in the misconceptions section.

#### Worked example: Northwind records \$300 of depreciation

Northwind's factory and machines depreciate by \$300 this year. Trace it:

**Income statement:** depreciation expense of \$300 lowers pre-tax profit by \$300. (At a 21% tax rate, it also saves \$63 of tax — the famous "depreciation tax shield" — but hold that thought; we'll keep the micro-example clean and revisit taxes in the full roll-forward.)

**Cash flow statement:** CFO gets a +\$300 add-back. Net income was depressed by \$300 that never cost cash, so we restore it.

**Balance sheet:** net PP&E falls by \$300, from (say) \$2,200 to \$1,900. The asset is one year more worn out on the books.

*Depreciation is the accountant's way of admitting that an asset you bought once is being used up gradually — and the three statements split that admission into a profit hit, a cash non-event, and a shrinking asset.*

## Wire 3: capex builds PP&E, and PP&E rolls forward

Capex — capital expenditure — is the mirror image of depreciation, and together they form a closed loop. **Capex is cash you spend now to buy a long-lived asset; depreciation is how that cost slowly hits profit over future years.** The wire runs: cash flow statement (capex out) → balance sheet (PP&E up) → income statement (depreciation, for years) → cash flow statement (add-back).

Here is the governing equation, the **PP&E roll-forward**:

$$\text{Ending net PP\&E} = \text{Beginning net PP\&E} + \text{Capex} - \text{Depreciation}$$

Capex *increases* PP&E (you bought stuff); depreciation *decreases* it (the stuff wore down). Net PP&E is just last period's balance pushed forward by those two flows. The figure below shows the roll.

![Property plant and equipment rolling forward across a year, beginning balance of two thousand plus capex of five hundred from the investing section minus depreciation of three hundred from the income statement equals an ending balance of two thousand two hundred that becomes next period's opening property plant and equipment](/imgs/blogs/how-the-three-financial-statements-connect-4.png)

This roll-forward is your first taste of how a *model* works. In a three-statement model you don't type the ending PP&E number in by hand — you *compute* it from the opening balance, the capex assumption (from the investing section), and the depreciation assumption (from the income statement). The balance sheet line is the *output* of a formula that pulls from the other two statements. That is the essence of an integrated model: balance-sheet lines are roll-forwards driven by flows from the income statement and cash flow statement.

#### Worked example: Northwind buys a \$500 machine

On July 1, Northwind buys a new machine for \$500, paid in cash. Trace it:

**Cash flow statement:** capex of \$500 appears in the **investing** section as a cash *outflow*. CFI is −\$500. Cash drops by \$500.

**Balance sheet:** net PP&E *increases* by \$500 (gross PP&E goes up by the purchase). Cash *decreases* by \$500. Notice that total assets are *unchanged* — we swapped \$500 of cash for \$500 of machine. Assets shifted form, not size, so the balance sheet stays balanced with *no* change to equity. Buying an asset with cash is purely an asset-side reshuffle.

**Income statement:** *nothing happens this period* (beyond a partial year of depreciation on the new machine, which we'll fold into the annual depreciation figure). This is the crux: a \$500 cash outlay produces a \$0 expense in the year you spend it. That \$500 will hit the income statement \$50 at a time over the machine's ten-year life — *as depreciation, in future years.*

This is exactly why capex-heavy companies can show healthy profits while bleeding cash, and why cash-generative companies can show modest profits. The income statement smooths the cost of long-lived assets across years; the cash flow statement records the full hit the moment the cash leaves. The two only reconcile through the PP&E roll-forward.

*Capex is the bill you pay today for productive capacity you'll expense slowly tomorrow — which is precisely why you can't judge a capital-intensive business by its income statement alone.*

## Wire 4: changes in working capital are cash flow adjustments

We touched this in Wire 1, but it deserves its own treatment because it's where models most often spring a leak. **When a working-capital account on the balance sheet *changes*, that change is a cash adjustment on the cash flow statement.** The rule, stated cleanly:

- An **asset** account (receivables, inventory) *rising* is a **use** of cash → **subtract** it from CFO.
- An **asset** account *falling* is a **source** of cash → **add** it to CFO.
- A **liability** account (payables) *rising* is a **source** of cash → **add** it to CFO.
- A **liability** account *falling* is a **use** of cash → **subtract** it from CFO.

The intuition: if your receivables go up, you sold things but haven't collected the cash — your profit overstates your cash, so you take the increase back out. If your payables go up, you bought things but haven't paid for them yet — you're holding onto cash you "owe," so you add it in. Working capital is, in essence, the *timing* difference between when accrual accounting books a transaction and when the cash actually moves. Master this single mechanism and you've mastered the hardest hinge between the income statement and the cash flow statement.

There's a strategic angle here that great investors obsess over. **Whether working capital is a cash *drain* or a cash *spring* depends on the business model, and it's one of the most underrated signals of business quality.** A company that grows by extending more credit and stuffing warehouses with inventory has *positive* working capital that *grows with sales* — every dollar of growth ties up more cash in receivables and inventory, so fast growth actually *consumes* cash even when it's profitable. Compare that to a business with *negative* working capital: it collects from customers *before* it pays its suppliers. A subscription company billing annually upfront, a retailer that sells inventory in days but pays suppliers in 60, a restaurant paid in cash that settles invoices monthly — these businesses get *handed cash* as they grow, because growth expands the float of money they're holding before they have to pay it out. Negative working capital is a near-magical property: growth funds itself. When you trace the working-capital wire and find it running *toward* the company rather than away from it, you've found a structurally advantaged business — and you'd never see it on the income statement, only by reading the balance sheet against the cash flow statement.

#### Worked example: Northwind collects the \$100 it was owed

Remember Wire 1, where Northwind sold \$100 of widgets on credit and its cash went *down* \$60? Now the customer **pays the \$100**. Trace this second transaction:

**Income statement:** *nothing*. The sale was already booked as revenue back when it was earned. Collecting the cash is not a new sale — it's just the customer settling up. Zero income-statement effect. This trips people up constantly: cash arriving does *not* mean profit.

**Balance sheet:** **accounts receivable falls by \$100** (the customer no longer owes us), and **cash rises by \$100**. Total assets unchanged — we swapped a receivable for cash. No equity change.

**Cash flow statement:** receivables *fell* by \$100. A falling asset is a *source* of cash, so we **add \$100** to CFO. Cash from operations rises by \$100.

Now stitch Wire 1 and this one together. The sale (Wire 1) hit cash for −\$60; the collection (this one) hits cash for... let's see the full picture. Over the two transactions combined, Northwind sold \$100 of goods that cost \$60 and got fully paid. Net income across both: +\$40 (booked at the sale). Net cash across both: it received \$100 and the goods cost \$60 (the \$60 was spent earlier on inventory), so the *operating cash* from the complete cycle is the \$40 of profit — once the cash actually comes in, profit and cash reconcile. Working capital was just the *temporary* gap between them. Receivables ballooned, then drained back to zero; over a full cycle, the working-capital swings net out and cash equals profit.

*Working capital is the lag between earning a dollar and touching it — it can swing cash wildly within a period, but over a complete cycle it washes out and cash converges to profit.*

## Wire 5: financing flows move debt and equity, and the balance sheet still balances

The financing section of the cash flow statement (CFF) is wired straight into the bottom of the balance sheet — the debt and equity that fund the company. **Debt issued raises cash and raises the debt liability; debt repaid lowers both. Dividends paid lower cash and lower retained earnings. Stock issued raises cash and raises equity; buybacks lower both.** Each financing transaction has its two double-entry sides, one in cash and one in the capital structure.

#### Worked example: Northwind borrows \$1,000 and pays a \$40 dividend

Two financing transactions. First, Northwind takes out a \$1,000 loan. Second, it pays its shareholders a \$40 dividend. Trace both, and we'll *prove* the balance sheet still balances — the whole point of double-entry.

**The \$1,000 loan:**

- **Cash flow statement:** +\$1,000 in the **financing** section (CFF). Cash inflow.
- **Balance sheet:** cash rises by \$1,000 (asset) *and* debt rises by \$1,000 (liability). Both sides up by \$1,000 — balanced. Assets +\$1,000 = liabilities +\$1,000. Equity untouched. Borrowing doesn't make you richer or poorer; it just inflates both sides of the balance sheet.
- **Income statement:** *nothing* this instant. (Interest *expense* will hit the income statement over time as the loan accrues interest — another wire that runs IS → CFS via the interest add-back/payment — but the act of borrowing the principal is not income or expense.)

**The \$40 dividend:**

- **Cash flow statement:** −\$40 in the **financing** section (CFF). Cash outflow.
- **Balance sheet:** cash falls by \$40 (asset) *and* retained earnings falls by \$40 (equity). Both sides down by \$40 — balanced. Assets −\$40 = equity −\$40. Note: a dividend is *not* an expense and never touches the income statement. It's a distribution of profit already earned, so it bypasses net income entirely and hits retained earnings directly. (Recall the retained-earnings formula: beginning RE + net income − *dividends*.)

**Prove the balance.** Across both transactions: assets changed by +\$1,000 − \$40 = **+\$960** (cash). The right side changed by +\$1,000 (debt) − \$40 (retained earnings) = **+\$960**. Left side +\$960, right side +\$960. The balance sheet is still in balance, *automatically*, because every one of the four entries had an equal-and-opposite partner. We never had to *force* it to balance — we just recorded both sides of each transaction and balance fell out.

*Financing transactions inflate or deflate both sides of the balance sheet in lockstep; a dividend uniquely skips the income statement, distributing profit straight out of equity.*

## Wire 6: ending cash *is* the cash line — the tie-out

This is the wire that closes the loop and makes the whole system self-checking. The cash flow statement builds up to a single number — **ending cash** — by this formula:

$$\text{Ending cash} = \text{Beginning cash} + \text{CFO} + \text{CFI} + \text{CFF}$$

And that ending-cash number is not *related to* the cash on the balance sheet. It **is** the cash on the balance sheet. They are the same number, by definition. The cash flow statement is, at its core, just an *explanation* of how the cash line on the balance sheet got from its beginning value to its ending value. The figure below shows the tie-out.

![The cash flow statement building from beginning cash of three hundred plus operations of two hundred fifty plus investing of minus five hundred plus financing of four hundred sixty to ending cash of five hundred ten, which must equal exactly the cash and equivalents line of five hundred ten at the top of this year's balance sheet or the model is broken](/imgs/blogs/how-the-three-financial-statements-connect-6.png)

This tie-out is the analyst's and modeler's single most important check. When you build a three-statement model, you compute ending cash on the cash flow statement, and you *link* the balance-sheet cash line to that number. Then you check that **total assets equal total liabilities plus equity.** If the balance sheet balances, your wiring is almost certainly correct. If it's off by, say, \$37, you have a bug — you forgot one of the two sides of some transaction, or double-counted a flow. The amount of the imbalance is often a clue: a \$37 imbalance might mean you forgot a \$37 line somewhere. The balance check is the model's built-in unit test.

#### Worked example: Northwind's full-year cash roll-forward

Let's pull Northwind's whole year together. Beginning cash was \$300. Here are the three sections for the year (in \$000s), built from everything we've traced plus the normal operating activity:

| Cash flow statement (full year) | \$000s |
|---|---:|
| **Operations** | |
| Net income | 200 |
| + Depreciation (add-back) | 300 |
| − Increase in receivables | (150) |
| − Increase in inventory | (50) |
| + Increase in payables | 30 |
| **Cash from operations (CFO)** | **330** |
| **Investing** | |
| Capex (machine + maintenance) | (500) |
| **Cash from investing (CFI)** | **(500)** |
| **Financing** | |
| Debt issued | 1,000 |
| Dividends paid | (40) |
| **Cash from financing (CFF)** | **960** |
| **Net change in cash** | **790** |
| Beginning cash | 300 |
| **Ending cash** | **1,090** |

Ending cash = 300 + 330 − 500 + 960 = **\$1,090**. Now build the **ending balance sheet** and confirm the cash line reads \$1,090 *and* that the whole thing balances:

| Ending Balance Sheet (Dec 31) | \$000s | How it rolled |
|---|---:|---|
| Cash | 1,090 | = ending cash from CFS |
| Accounts receivable | 350 | 200 + 150 increase |
| Inventory | 200 | 150 + 50 increase |
| Net PP&E | 2,200 | 2,000 + 500 capex − 300 deprec. |
| **Total assets** | **3,840** | |
| Accounts payable | 150 | 120 + 30 increase |
| Debt | 1,800 | 800 + 1,000 issued |
| **Total liabilities** | **1,950** | |
| Common stock | 1,000 | unchanged |
| Retained earnings | 890 | 730 + 200 NI − 40 dividends |
| **Total equity** | **1,890** | |
| **Total liabilities + equity** | **3,840** | |

Look at what just happened. **Cash on the balance sheet is \$1,090 — exactly the ending cash from the cash flow statement.** Wire 6 holds. And **total assets of \$3,840 equal total liabilities plus equity of \$3,840.** The balance sheet balances. Every single line on this ending balance sheet was *computed* — rolled forward from the beginning balance sheet using flows from the income statement and cash flow statement. Receivables rolled by their increase. PP&E rolled by capex minus depreciation. Retained earnings rolled by net income minus dividends. Cash rolled by the cash flow statement. We didn't type in a single ending balance by hand, and yet it balanced. *That is the three-statement model.*

*If you can roll every balance-sheet line forward from flows and the sheet still balances, you haven't just read three statements — you've reconstructed the company's entire year from its wiring, and that is exactly the skill that lets you forecast it.*

## The order you build them in — and the one loop that bites everyone

The six wires tell you *what* connects to what. But when you sit down to actually build a model, you also need to know *in what order* to compute the statements, because the wiring imposes a sequence. Get the order wrong and you'll be reaching for a number that hasn't been computed yet.

The canonical build order is: **income statement first, then most of the cash flow statement, then the balance sheet, then close the loop.** Here's the logic, and it falls right out of the wires:

1. **Build the income statement down to net income.** You can't start anywhere else — net income is the input to both the cash flow statement (Wire 1) and retained earnings (Wire 1 again). Everything depends on it. Project revenue, COGS, operating expenses, D&A, interest, and taxes, and land on net income.

2. **Build cash from operations.** Start at net income, add back D&A (Wire 2), and adjust for working-capital changes (Wire 4) — which you get by comparing this period's projected working-capital balances to last period's. CFO falls out.

3. **Build investing and financing.** Capex (Wire 3) comes from your capex assumption; debt issuance/repayment and dividends (Wire 5) come from your financing schedule. Now you have all three cash-flow sections.

4. **Compute ending cash, and roll the balance sheet.** Ending cash = beginning cash + CFO + CFI + CFF (Wire 6). Then roll every other balance-sheet line: PP&E by capex minus depreciation, receivables/inventory/payables by their projected changes, debt by issuance minus repayment, retained earnings by net income minus dividends, common stock by issuance minus buybacks.

5. **Check that it balances.** Total assets versus total liabilities plus equity. If they match, you're done. If not, hunt the broken wire.

#### Worked example: the interest circularity, and how the model closes the loop

Here's the loop that traps every beginner — and it's worth understanding because it shows just how *tightly* the three statements are wired. Notice in step 1 that the income statement needs an **interest expense** figure. Interest is charged on debt — but how much debt? Debt depends on whether the company had to *borrow* this year to cover a cash shortfall (or could *repay* with surplus cash), which depends on its cash flow, which depends on... net income, which depends on interest expense. The snake is eating its tail.

Concretely: suppose Northwind's model has a "revolver" — a credit line it draws on automatically when cash runs low. If operations leave it \$200 short, it draws \$200 on the revolver. But that \$200 of new debt carries, say, 5% interest = \$10 of extra interest expense. That \$10 lowers net income, which lowers CFO, which makes the cash shortfall *bigger*, which means it must draw *more* on the revolver, which adds *more* interest... You have a genuine circular reference: interest → net income → cash → debt → interest.

Modelers resolve this in one of two ways. The clean way is **iterative calculation** — the spreadsheet loops the circular formulas until the numbers stop changing (they converge quickly, because each round's adjustment shrinks). The robust way is a **circularity switch** — a toggle that breaks the loop by temporarily hard-coding interest, so a single bad input doesn't send the whole model spinning into `#REF!` errors. Either way, the *existence* of the circularity is itself proof of how integrated the statements are: you literally cannot compute the income statement without first knowing the balance sheet's debt, and you can't know the debt without the cash flow statement, and you can't build the cash flow statement without the income statement. The three statements aren't just linked — for a company with debt, they're *simultaneous*. They have to be solved together.

*The interest circularity is the clearest possible proof that the three statements are one system: you cannot finish any one of them in isolation, because each needs an answer that only the others can supply.*

## The integrated three-statement model: a forecasting engine

Everything we've done — tracing transactions, rolling balances forward, tying out cash — is what analysts and investment bankers do mechanically inside a spreadsheet called the **three-statement model**. The historical version reconstructs the past; the powerful version *forecasts the future*. And the reason it's powerful is the wiring: **once the three statements are linked, you only need to forecast a handful of drivers, and the linkages force everything else.**

Here's how a forecast model flows, and why it ends in a valuation:

![The integrated model as a five step engine, driver assumptions feed a forecast income statement, which flows through the linked cash flow statement and balance sheet, producing free cash flow each year, which a discounted cash flow model turns into an estimated share price](/imgs/blogs/how-the-three-financial-statements-connect-7.png)

**Step 1 — drivers.** You make assumptions about a small number of *drivers*: revenue growth, gross margin, operating margin, capex as a percent of sales, depreciation schedule, working-capital days, tax rate, dividend policy, debt schedule. Maybe a dozen numbers.

**Step 2 — forecast the income statement.** From the drivers, project revenue, costs, D&A, interest, taxes, and net income for each future year.

**Step 3 — the cash flow statement and balance sheet follow automatically.** Net income starts CFO (Wire 1). D&A is added back (Wire 2). Working-capital accounts on the balance sheet roll forward from your days-assumptions, and their changes hit CFO (Wire 4). Capex flows out of investing and into PP&E (Wire 3). Debt and dividends flow through financing into the balance sheet (Wire 5). Ending cash ties out and the balance sheet must balance (Wire 6). You forecast *one* statement explicitly; the wiring builds the other two.

**Step 4 — free cash flow.** Out of the linked statements falls **free cash flow** — roughly CFO minus capex — the cash the business throws off that owners could actually take out each year. This is the number valuation cares about, because it's the cash, not the accounting profit, that ultimately belongs to investors.

**Step 5 — valuation.** Discount those future free cash flows back to today at a discount rate (the weighted-average cost of capital, WACC), add a terminal value for the years beyond your forecast, and you get an estimate of what the whole business is worth — and, divided by shares, what one share is worth. That's a **discounted cash flow (DCF)** valuation, and it's the subject of the [next post in this series](/blog/trading/equity-research/building-a-dcf-part-1-forecasting).

The deep point: **the three-statement model is the bridge between accounting and valuation.** Accounting (the three statements) tells you what happened and, when wired together and projected, what *will* happen to the cash. Valuation takes that projected cash and turns it into a price. You cannot do a credible DCF without an integrated model underneath it, because the DCF needs free cash flow, and free cash flow only emerges when all three statements are linked. This is why "build the three-statement model" is the first task on every analyst's and banker's desk — it's the engine; valuation is just the dial on the front.

#### Worked example: forecasting Northwind one year forward

Suppose for next year you assume: revenue grows 10% (to \$2,200 of sales, say), net income comes in at \$240, depreciation is \$320, capex is \$400, receivables and inventory each rise \$30 with the business, payables rise \$10, the company repays \$100 of debt and pays a \$50 dividend.

You don't build the balance sheet by guessing. You roll it: CFO = 240 + 320 − 30 − 30 + 10 = **\$510**. CFI = −400. CFF = −100 − 50 = −150. Net change in cash = 510 − 400 − 150 = **−\$40**. Ending cash = 1,090 − 40 = **\$1,050**. Net PP&E = 2,200 + 400 − 320 = **\$2,280**. Retained earnings = 890 + 240 − 50 = **\$1,080**. Debt = 1,800 − 100 = **\$1,700**. And free cash flow = CFO − capex = 510 − 400 = **\$110** — the number a DCF would discount.

Every one of those balance-sheet lines was *computed from drivers through the wiring*, not typed in. And if you footed the full ending balance sheet, it would balance — because we respected every wire. *Forecasting isn't filling in three separate statements; it's setting a few dials and letting the linkages do the rest.*

## Common misconceptions

**"Profit and cash are the same thing."** The single most expensive mistake a beginner makes. A company can report record profits and run out of cash (it's selling on credit faster than it collects, or pouring cash into inventory and capex), and a company can report thin profits and gush cash (depreciation is huge, working capital is shrinking). The two diverge through exactly the wires in this post: working-capital changes, D&A, and capex. Profit is an *opinion* shaped by accrual rules; cash is a *fact*. Always read all three statements; never judge a business on the income statement alone.

**"The cash flow statement is just a summary of the other two."** It's the opposite of a summary — it's a *reconciliation*. It takes the accrual profit from the income statement and the balance-changes from the balance sheet and explains, line by line, why the cash actually moved the way it did. It's the statement that *catches the lie* when reported profit isn't backed by cash. That's why forensic analysts and short-sellers live in the cash flow statement: a company whose net income keeps rising while CFO stagnates is flashing a warning that its profits aren't real cash — the exact pattern that preceded blowups like [Enron](/blog/trading/finance/enron-2001-accounting-fraud) and [Wirecard](/blog/trading/finance/wirecard-the-german-fintech-fraud).

**"If a company is profitable, retained earnings equals the cash it has piled up."** No. Retained earnings is *cumulative profit kept* — but that profit may be sitting in receivables, inventory, factories, or paid-down debt, not in the bank. A company can have \$5 billion of retained earnings and \$50 million of cash, because the earnings were *reinvested* into assets. Retained earnings is an equity line, not a cash line; the only cash line is "cash," and it ties to the cash flow statement, not to retained earnings.

**"Buying an asset is an expense."** Buying a \$500 machine is *not* an expense in the year you buy it — it's an *investment*. It moves cash into PP&E (an asset swap), and only the *depreciation* portion becomes an expense, spread over years. Confusing capex with expense makes you think a growing, investing company is unprofitable when it's just spending cash on its future. The income statement deliberately separates the two through depreciation.

**"Dividends are an expense that reduces profit."** Dividends never touch the income statement. They are a *distribution* of already-earned profit, paid out of retained earnings, recorded only on the balance sheet (equity down) and cash flow statement (financing outflow). Net income is computed *before* any dividend decision — the dividend just decides how much of that profit stays in the company versus goes to owners.

**"If my model doesn't balance, I should plug the difference."** Never. An imbalance is a *signal*, not a nuisance to paper over. It means you forgot one side of a transaction or mis-linked a flow. Plugging the difference hides the bug and makes every downstream number wrong. The discipline of a real modeler is to *find* the broken wire — and the size and sign of the imbalance is usually the clue that points right at it.

## How it shows up in real markets

**Amazon's two-decade lesson in profit-versus-cash.** For years, Amazon reported tiny or negative net income while its stock soared, baffling people who only read the income statement. The cash flow statement told the real story: Amazon was generating enormous *operating* cash, then plowing it straight back into capex (warehouses, data centers) and working-capital advantages (it collects from customers *before* it pays suppliers — negative working capital, a cash *source*). Read only the bottom line and Amazon looked unprofitable; trace the wires — modest net income, huge D&A add-back, negative working capital feeding CFO, gigantic capex — and you see a cash machine deliberately suppressing accounting profit to reinvest. The three-statement view was the *only* way to understand the business, and the investors who read it correctly were enormously rewarded.

**The capex-heavy utility versus the asset-light software firm.** A regulated electric utility and a SaaS company can report identical net income, yet their statements look nothing alike. The utility has massive PP&E, enormous depreciation (a huge non-cash add-back lifting CFO above net income), and relentless capex (draining that CFO back out) — its free cash flow is modest and its balance sheet is heavy. The software firm has tiny PP&E, negligible capex, and CFO that's mostly real, spendable cash — its free cash flow nearly equals its net income. Same profit, wildly different *cash*. An analyst who values both off net income alone will badly misprice them; only the integrated three-statement view reveals that the software firm's earnings are worth far more per dollar because they convert to free cash.

**How fraud hides — and how the wiring exposes it.** Accounting frauds almost always attack the income statement first, because that's the number management is judged on: they book fake revenue or capitalize expenses to inflate profit. But the three-statement wiring makes that hard to sustain. Fake revenue inflates receivables (you "sold" but can't collect) — and receivables ballooning faster than sales, with CFO falling behind net income, is the classic tell. Capitalizing operating costs as assets inflates PP&E and props up profit, but it bloats the balance sheet and depresses free cash flow. WorldCom did exactly this — capitalizing \$3.8 billion of ordinary line costs as assets — and the giveaway was a balance sheet swelling with "assets" while cash generation didn't follow. Frauds that fool one statement get *caught by the other two*, because the wires force consistency that lies cannot maintain. This is why the deepest part of [quality-of-earnings analysis](/blog/trading/finance/enron-2001-accounting-fraud) is reading the three statements *against each other*, not one at a time.

**Why every banker's first model is the three-statement model.** Walk onto any sell-side or buy-side desk and the first thing a junior analyst builds for a new company is the integrated three-statement model. M&A bankers build it to test what a deal does to the combined company's earnings and cash. Equity analysts build it to forecast and value. Lenders build it to test whether the borrower will generate enough cash to service debt. The model is the shared language of corporate finance precisely *because* it's wired — change one assumption and you instantly see the effect on profit, cash, *and* financial position, all consistent with each other. There is no more fundamental skill in equity research.

## When this matters and further reading

You now have the one thing that separates someone who can *read* financial statements from someone who can *use* them: you see the wires. Net income flowing to equity and to cash. Depreciation appearing in triplicate. Capex building assets that bleed back as expense. Working capital as the timing gap between profit and cash. Financing inflating both sides of the balance sheet. And ending cash tying out to the balance sheet, with the whole thing balancing because every transaction has two sides. That is not six facts to memorize — it's *one system* you can now picture.

This matters everywhere downstream. You can't analyze a company's *quality of earnings* without comparing net income to CFO. You can't forecast without rolling balances forward through the wires. And you absolutely cannot build a [DCF valuation](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) without an integrated model producing free cash flow — which is exactly where this series goes next.

If you want to go deeper on the individual statements first, revisit the three siblings to this post: the [income statement line by line](/blog/trading/equity-research/income-statement-line-by-line-revenue-to-net-income), the [balance sheet](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth), and the [cash flow statement](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from). For a sense of how reading the statements *against each other* exposes manipulation, the [Enron](/blog/trading/finance/enron-2001-accounting-fraud) and [Wirecard](/blog/trading/finance/wirecard-the-german-fintech-fraud) post-mortems are the best teachers there are — both companies looked fine on one statement and fell apart the moment you traced the wires.
