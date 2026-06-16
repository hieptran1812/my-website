---
title: "The Cash Flow Statement: Where the Cash Really Comes From"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero guide to the three sections of the cash flow statement, the indirect method that bridges profit to cash, free cash flow, and how to spot a profitable company that is quietly burning money."
tags: ["equity-research", "corporate-finance", "cash-flow-statement", "free-cash-flow", "indirect-method", "financial-statements", "fundamental-analysis", "accounting", "valuation"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Profit is an opinion; cash is a fact. The cash flow statement strips away accounting accruals and shows the actual money that moved into and out of a business, which is why it is the hardest statement to fake and the one professionals trust most.
>
> - The statement has three sections: **operations** (cash from running the business), **investing** (cash spent on or raised from long-term assets), and **financing** (cash from lenders and shareholders). They sum to the change in the cash line on the balance sheet.
> - The **indirect method** rebuilds cash from operations by starting at net income, adding back non-cash charges like depreciation and stock-based compensation, and adjusting for changes in working capital — receivables up means cash out, payables up means cash in.
> - **Free cash flow** = cash from operations minus capital expenditure. It is the money a business actually has left over after keeping the lights on, and it is what a buyer of the whole company ultimately gets.
> - A company can report rising profit while its operating cash flow falls or turns negative. Sometimes that is healthy growth; sometimes it is a warning that earnings are being manufactured faster than cash is collected.
> - The single most diagnostic ratio for earnings quality is **cash conversion** — operating cash flow divided by net income. When it persistently sits well below one, the profit is not turning into money, and you should ask why.

A company can be profitable and still go bankrupt. It happens more often than you would think, and it happens for one reason: a business does not pay its suppliers, its workers, or its lenders with profit. It pays them with cash. Profit is a number an accountant computes by matching revenues to expenses under a set of rules. Cash is the actual balance in the bank account. Most of the time these two move together, but they can diverge for months or years — and in that gap between *reported earnings* and *real money* lives most of the financial trouble, and most of the financial fraud, in the public markets.

The income statement tells you whether a company earned a profit. The balance sheet tells you what it owns and owes at a moment in time. But neither directly answers the question a serious investor cares about most: *did the business actually generate cash, and where did it come from?* That is the job of the third statement — the **cash flow statement**. It is the youngest of the three (it only became a required statement in the United States in 1987), it is the least glamorous, and it is the one that experienced analysts read first. Because while a clever or dishonest management team can flatter earnings in dozens of legal and illegal ways, cash is stubborn. You either have it or you do not. The reported cash balance has to reconcile to a number a bank can confirm.

The figure below is the mental model we will build toward. A business starts a period with some cash in the bank, money flows in and out through three distinct channels — running the business, investing in it, and funding it — and you end the period with a new cash balance. The cash flow statement is simply a disciplined accounting of every one of those flows, organized so you can see at a glance whether the company is a cash machine or a cash furnace.

![The three sections of the cash flow statement flowing as a waterfall from opening cash to closing cash](/imgs/blogs/cash-flow-statement-where-the-cash-really-comes-from-1.png)

We are going to build this up from nothing. If you have never read a financial statement, that is fine. By the end of this piece you will understand what each of the three sections contains and why; you will be able to take a company's net income and walk it, line by line, all the way to its operating cash flow using the indirect method; you will know what free cash flow is and why it is the number that ultimately drives what a company is worth; and — most importantly — you will be able to look at a "profitable" company and tell whether it is actually making money or quietly bleeding cash behind a wall of accounting. We will use a fictional company, **Northwind Industries**, as our running example so the numbers compound as we go. Let us start with the foundations.

## Foundations: profit, cash, and why they are not the same

Before we can read the cash flow statement, we need to be clear about a handful of terms. Take your time here; everything later depends on getting these straight.

### Revenue, expense, profit — and the accrual that separates them from cash

**Revenue** is the total value of goods or services a company billed its customers for during a period. **Expenses** are the costs it incurred to produce that revenue. **Profit** — also called **net income** or **earnings** — is what is left when you subtract every expense from revenue. This is the bottom line of the income statement, and it is the number that makes headlines. If you want the full anatomy of how revenue becomes net income, the companion piece walks the [income statement line by line](/blog/trading/equity-research/income-statement-line-by-line-revenue-to-net-income); here we only need the punchline.

The crucial thing to understand is that profit is computed on an **accrual basis**, not a cash basis. Accrual accounting has one core principle: *record revenue when it is earned and expenses when they are incurred, regardless of when cash actually changes hands.* This is a genuinely good idea — it matches effort to reward in the same period and gives a truer picture of economic performance than raw cash would. But it means the income statement is full of revenues for which no cash has yet arrived and expenses for which no cash has yet been paid.

Two examples make the gap concrete. Suppose Northwind ships \$1 million of machinery to a customer in December but gives that customer 90 days to pay. Under accrual accounting, Northwind records \$1 million of revenue in December — even though not a single dollar has hit the bank. The customer's promise to pay shows up on the balance sheet as **accounts receivable**, an asset. Profit went up by the margin on that sale; cash did not move at all. Now flip it: Northwind buys \$200,000 of steel in December but does not have to pay the supplier until February. The cost gets recorded when the steel is used, and in the meantime Northwind owes the supplier — that promise shows up as **accounts payable**, a liability. Cash has not left the building, but an expense (when the steel is consumed) eventually will be recorded.

Multiply these timing differences across thousands of transactions and you see why net income and cash from operations can be very different numbers in any given period. The cash flow statement exists to undo the accruals and show you the cash underneath.

### Non-cash expenses: depreciation, amortization, stock-based comp

There is a second, even larger source of the gap: expenses that reduce reported profit but involve no cash outflow at all in the current period.

The biggest is **depreciation**. When Northwind buys a \$10 million factory, it does not record a \$10 million expense in year one. Accounting says the factory will be useful for, say, 20 years, so it spreads the cost out — recording roughly \$500,000 of **depreciation** expense each year for 20 years. That \$500,000 reduces profit every year, but the cash for the factory all left in year one. In years 2 through 20, depreciation is a pure paper expense: it lowers earnings without touching the bank account. **Amortization** is the same idea applied to intangible assets like patents or acquired software. We bundle the two as **D&A**.

A second large non-cash expense in modern companies, especially in technology, is **stock-based compensation (SBC)**. When a firm pays an engineer partly in stock options or restricted shares, that compensation is a real expense — it reduces reported profit — but no cash leaves the company. The company hands over a slice of ownership instead of money. So SBC, like depreciation, gets added back when we compute operating cash flow. (Whether you *should* treat SBC as "free" is a genuinely contested question we will return to — it is not free to existing shareholders, who get diluted, even if it does not cost cash.)

The pattern to internalize: **any expense that reduced profit without using cash gets added back; any cash outflow that did not reduce profit gets subtracted.** The whole indirect method is just the disciplined application of that one idea.

### Working capital: the cash trapped in running the business

The last foundational concept is **working capital**, and it is where most of the subtle action happens. Working capital is the cash a business has tied up in the day-to-day machinery of operating: the inventory sitting in the warehouse, the receivables owed by customers who have not paid yet, minus the payables it owes suppliers who have not been paid yet.

Think of it as a pool of cash the business must keep submerged in its operations just to function. Three accounts dominate it:

- **Accounts receivable (AR)** — money customers owe you. When AR *rises*, it means you booked sales as revenue but have not collected the cash. Rising AR is a **use** of cash.
- **Inventory** — goods you have bought or built but not yet sold. When inventory *rises*, you spent cash to acquire it but have not recovered that cash through a sale. Rising inventory is a **use** of cash.
- **Accounts payable (AP)** — money you owe suppliers. When AP *rises*, you received goods or services but have not paid for them yet — you are holding onto cash you eventually owe. Rising AP is a **source** of cash.

The direction is the part beginners always get backwards, so here is the rule, stated once, cleanly: **an increase in an asset (receivables, inventory) is a use of cash; an increase in a liability (payables) is a source of cash.** Decreases flip the sign. We will use this rule constantly, and the figure later in this piece (the working-capital effects grid) exists precisely to make it muscle memory.

With revenue, accrual, non-cash expenses, and working capital defined, we can read the statement itself.

## The three sections: operations, investing, financing

Every cash flow statement is divided into three sections, and the division is not arbitrary. It answers three different questions about the business.

1. **Cash from operating activities (CFO)** — Does the core business, the thing the company actually does, generate cash? This is the cash thrown off by selling products and services, after paying the costs of running the operation. For a healthy mature company this should be reliably positive and the largest of the three.
2. **Cash from investing activities (CFI)** — Is the company spending cash to grow or maintain its asset base, or raising cash by selling assets? The dominant line here is **capital expenditure** (capex) — money spent on property, plant, equipment, and other long-lived assets. CFI is usually negative for a growing company, because it is investing.
3. **Cash from financing activities (CFF)** — How is the company funding itself, and how is it returning cash to the people who provided that funding? This captures borrowing and repaying debt, issuing or buying back shares, and paying dividends.

Add the three together and you get the **net change in cash** for the period. Add that to the cash you started with, and you arrive at the cash you ended with — which must exactly match the cash line on the balance sheet. That tie-out is the statement's built-in lie detector: the three sections are not free-floating; they must reconcile to a real, confirmable bank balance. The waterfall figure above shows exactly this: opening cash, plus or minus each of the three sections, equals closing cash.

Let me make the structure concrete with our running company before we dig into each section.

#### Worked example: Northwind's cash flow statement at a glance

Northwind Industries makes industrial machinery. For the year just ended, here is the top-level shape of its cash flow statement (we will derive every number that follows):

- It started the year with **\$40 million** in cash.
- **Cash from operations: +\$85 million.** The core business threw off real cash.
- **Cash from investing: −\$70 million.** It spent \$60 million on factories and equipment (capex) and \$10 million acquiring a small competitor.
- **Cash from financing: −\$25 million.** It paid \$15 million in dividends, bought back \$20 million of its own stock, and borrowed a net \$10 million.
- **Net change in cash:** \$85m − \$70m − \$25m = **−\$10 million.**
- **Ending cash:** \$40m − \$10m = **\$30 million** — which is exactly the cash line on its year-end balance sheet.

Notice the company's cash balance actually *fell* by \$10 million during a year in which it generated \$85 million of operating cash and was solidly profitable. That is not a problem — it deliberately invested \$70 million and returned \$35 million to shareholders. The statement tells you not just *whether* cash changed but *why*, and the "why" here is a confident, growing company spending and returning more than it earned in a single year.

*The change in cash is almost never the interesting number; the composition of the three sections is.*

## Reading cash from operations: the indirect method

This is the heart of the statement and the section you must learn to read fluently. There are two ways to present operating cash flow. The **direct method** simply lists cash received from customers and cash paid to suppliers and employees — clean, but almost no company uses it because it is more work and reveals more. The **indirect method**, used by the overwhelming majority of public companies, starts at net income and adjusts it back to cash. We focus on the indirect method because it is what you will actually see, and because the adjustments themselves are diagnostic.

The logic is a bridge. You begin at the bottom of the income statement — net income — and you make exactly the corrections we set up in the foundations: add back the non-cash expenses that reduced profit, and adjust for the working-capital changes that consumed or released cash. What you arrive at is the cash the operations actually produced.

The figure below is that bridge, drawn as a before-and-after: net income on the left, and each adjustment carrying you across to operating cash flow on the right.

![The indirect method bridging net income to operating cash flow through non-cash add-backs and working capital adjustments](/imgs/blogs/cash-flow-statement-where-the-cash-really-comes-from-2.png)

Let us walk the bridge step by step, then do it with Northwind's real numbers.

**Step 1 — Start at net income.** This is profit after everything: after costs, after interest, after tax. It already reflects all the accrual revenue and all the non-cash expenses, so we have to reverse the non-cash parts.

**Step 2 — Add back depreciation and amortization.** D&A reduced net income but consumed no cash this period. So we add it straight back. This is almost always the single largest add-back, and for capital-heavy businesses it can dwarf net income itself.

**Step 3 — Add back stock-based compensation.** SBC reduced net income but cost no cash. Add it back. (Hold the question of whether this is economically honest; we will hit it in the misconceptions section.)

**Step 4 — Add back other non-cash items.** Deferred taxes, impairment write-downs, losses on asset sales, and similar paper charges all get reversed because they hit profit without moving cash. (A *gain* on an asset sale gets subtracted here, because the cash from that sale belongs in investing, not operations — leaving the gain in CFO would double-count it.)

**Step 5 — Adjust for changes in working capital.** This is the step that separates novices from analysts. For each working-capital account, apply the rule: an increase in an asset (AR, inventory) is a *subtraction* (cash out); an increase in a liability (AP, accrued expenses) is an *addition* (cash in). This step is where a growing company often consumes cash, and where a deteriorating company often hides.

The result is **cash from operating activities**. Now let us make it real.

#### Worked example: reconciling Northwind's net income to CFO

Northwind reported **net income of \$50 million** for the year. Here is the full reconciliation to its \$85 million of operating cash flow, using the indirect method:

| Line | Amount |
|---|---|
| Net income | **\$50.0m** |
| + Depreciation & amortization | +\$28.0m |
| + Stock-based compensation | +\$6.0m |
| + Loss on equipment retirement (non-cash) | +\$2.0m |
| − Increase in accounts receivable | −\$12.0m |
| − Increase in inventory | −\$5.0m |
| + Increase in accounts payable | +\$9.0m |
| + Increase in accrued liabilities | +\$7.0m |
| **= Cash from operations (CFO)** | **\$85.0m** |

Walk it slowly. Net income is \$50m. We add back \$28m of D&A and \$6m of SBC — \$34m of expenses that never touched cash — plus a \$2m non-cash loss. That already lifts us to \$86m. Then working capital: receivables grew \$12m (Northwind sold more on credit and is owed more, so \$12m of "profit" is stuck in customers' hands — subtract it), and inventory grew \$5m (cash spent on goods not yet sold — subtract it). But payables grew \$9m and accrued liabilities grew \$7m (Northwind is holding \$16m it owes suppliers and employees but has not paid — that is cash it still has, add it). Net working-capital effect: −\$12m − \$5m + \$9m + \$7m = **−\$1m**. So \$86m − \$1m = **\$85m of operating cash flow.**

*Northwind turned \$50m of accounting profit into \$85m of operating cash — mostly because depreciation is a huge paper expense that does not consume cash, which is typical of a capital-intensive manufacturer.*

That CFO-above-net-income relationship is the signature of a healthy, capital-heavy business: real cash generation comfortably exceeds the accrual profit because the biggest expense (depreciation) is non-cash. Hold that benchmark in mind, because soon we will look at a company where the relationship is inverted — and that inversion is a red flag.

One subtlety worth flagging now, because it confuses almost everyone: under US accounting rules (GAAP), **interest paid and interest received both sit inside operating cash flow**, not financing or investing, even though interest is fundamentally a financing cost. Income taxes paid also live in CFO. So when you compare operating cash flow across two companies, remember that a heavily indebted company's CFO already has its interest burden baked in, while its financing section shows only the principal movements. (International rules under IFRS give companies more latitude to classify interest in financing, which is one reason cross-border comparisons of CFO require care.) The practical takeaway: CFO is "operating" cash in name, but it absorbs the cash cost of debt and taxes, so it is closer to "cash the business produced after paying its lenders and the government" than to a pure measure of operational output.

There is also a discipline in *how* you read the adjustments, not just their total. The non-cash add-backs (D&A, SBC) tell you how capital- or equity-compensation-intensive the business is. The working-capital line tells you whether the business is consuming or releasing cash as it operates. A company whose CFO is propped up by a one-time release of working capital — collecting a backlog of old receivables, or stretching its suppliers unusually far in a single quarter — is producing cash that will not repeat. Conversely, a growing company that consumes working capital every period is showing you a real, recurring cash cost of its own growth. Reading the *composition* of the bridge, not just its endpoint, is what separates a mechanical reading from an analytical one.

## Cash from investing: capex, acquisitions, and asset sales

The investing section is shorter and more intuitive, but one line in it — capital expenditure — is so important that it deserves real attention, because it stands between operating cash flow and the number that actually matters to an owner.

**Capital expenditure (capex)** is cash spent to acquire or upgrade long-lived physical assets: factories, machines, servers, vehicles, buildings. It appears in investing because the asset will benefit the business for years, not just the current period. Capex is almost always a cash *outflow* (negative), and for most industrial, telecom, energy, and retail companies it is the dominant line in the whole statement.

The single most useful distinction in this section is between two kinds of capex:

- **Maintenance capex** — the spending required just to keep the existing business running at its current level: replacing worn-out machines, refreshing aging servers, repaving the parking lot. This is non-optional. A business that stops spending maintenance capex is liquidating itself slowly.
- **Growth capex** — the spending to *expand*: building a new plant, opening new stores, adding capacity the company did not have before. This is discretionary. A company can cut growth capex in a downturn and survive; it is investing for a bigger future.

Companies do not report this split — it is one of the most important judgments an analyst makes, and getting it roughly right is essential to estimating the true free cash flow a business produces. A useful first approximation: **maintenance capex is often close to the depreciation charge**, because depreciation is, in theory, the accounting estimate of how fast existing assets wear out. Spending above depreciation is then a rough proxy for growth.

The other investing lines are simpler. **Acquisitions** — buying other companies — are cash outflows (the cash you paid, net of any cash that came with the target). **Asset sales** and **divestitures** are cash inflows (selling a division, a building, a fleet). **Purchases and sales of securities** (a company parking spare cash in marketable securities) net out here too, though they are usually financial noise rather than operating substance.

#### Worked example: splitting Northwind's capex into maintenance and growth

Northwind spent **\$60 million of total capex** and also paid **\$10 million to acquire a competitor**, for total investing outflows of \$70 million. How much of that \$60m of capex was just keeping the lights on, and how much was real expansion?

Recall Northwind's depreciation was **\$28 million**. Using the depreciation-as-maintenance-proxy approach:

- **Maintenance capex ≈ \$28 million** (roughly the cost of replacing assets as they wear out).
- **Growth capex ≈ \$60m − \$28m = \$32 million** (the spending above replacement, building genuinely new capacity).

So a little over half of Northwind's capital spending is discretionary growth investment. This matters enormously for valuation: if you want to know the cash the *current* business produces without growth, you subtract only the \$28m of maintenance capex, not the full \$60m. We will see in a moment why that choice changes the free-cash-flow number — and the company's apparent value — dramatically.

*The maintenance-versus-growth split converts a single capex number into a judgment about how much of a company's spending is mandatory upkeep versus an optional bet on a bigger future.*

The figure below makes the split visual: total capex as a stack, with the maintenance floor that the business cannot avoid sitting beneath the growth layer it chooses to add.

![Total capital expenditure split into a mandatory maintenance floor and a discretionary growth layer on top](/imgs/blogs/cash-flow-statement-where-the-cash-really-comes-from-6.png)

## Cash from financing: debt, dividends, buybacks, equity

The financing section records every transaction with the two groups who fund the company: **lenders** (who provide debt) and **shareholders** (who provide equity). It is the cleanest of the three sections because the items are explicit cash movements, but it tells you a great deal about management's character and the company's life stage.

The lines fall into two buckets:

**Debt financing.**
- **Debt issued** (borrowing) is a cash inflow — the company takes in cash and creates a liability to repay later.
- **Debt repaid** (paying down principal) is a cash outflow. (Interest paid, confusingly, usually sits up in operating cash flow under US GAAP, not here — a quirk worth remembering.)

**Equity financing.**
- **Equity issued** — selling new shares — is a cash inflow. Common for young companies raising capital and for any company doing a secondary offering.
- **Share buybacks (repurchases)** — buying back the company's own stock — are a cash outflow. The company returns cash to shareholders by shrinking the share count.
- **Dividends paid** — direct cash payments to shareholders — are a cash outflow.

The composition of CFF tells a story. A young, growing company tends to show *positive* financing cash flow: it is raising money (issuing equity, taking on debt) to fund a business that does not yet generate enough cash itself. A mature, cash-rich company tends to show *negative* financing cash flow: it is returning money (dividends, buybacks, debt paydown) because it generates more cash than it can profitably reinvest. Neither is inherently good or bad — but a mature company with persistently *positive* financing flows (constantly raising cash to survive) is one to scrutinize, and a young company hemorrhaging cash into buybacks instead of growth is another.

#### Worked example: Northwind's financing section and what it signals

Northwind's financing section nets to **−\$25 million**:

- Debt issued: **+\$40 million** (it took out a new term loan).
- Debt repaid: **−\$30 million** (it retired older, higher-rate debt). Net new debt: **+\$10 million.**
- Dividends paid: **−\$15 million.**
- Share buybacks: **−\$20 million.**
- Net financing: \$10m − \$15m − \$20m = **−\$25 million.**

What does this say? Northwind generated \$85m of operating cash, spent \$70m investing in itself, and *still* returned \$35m to shareholders (\$15m dividends + \$20m buybacks) while modestly increasing its debt. This is the profile of a confident, mature, self-funding business: the core throws off enough cash to invest heavily *and* reward owners. The modest net borrowing suggests management is comfortable using a little leverage to optimize its capital structure rather than out of necessity. That mindset — a company funding its own growth and returns from internal cash — is exactly what long-term owners like [Warren Buffett look for](/blog/trading/finance/warren-buffett-berkshire-value-investing).

*Financing cash flow is the company's relationship with its funders written in cash: who is putting money in, and who is taking it out.*

## Free cash flow: the number that actually matters

We now have everything we need for the single most important derived number in fundamental analysis. Operating cash flow tells you the business generates cash. But a business cannot keep generating that cash without continually spending on the assets that produce it. The cash that is *truly* left over — after the business has paid to sustain itself — is **free cash flow**.

The simplest and most common definition is:

$$
\text{Free Cash Flow} = \text{Cash from Operations} - \text{Capital Expenditure}
$$

That is it. Take the cash the operations produced, subtract the cash the company had to spend on long-lived assets, and what remains is the cash genuinely available to do whatever management wants: pay dividends, buy back stock, pay down debt, make acquisitions, or simply pile up in the bank. Free cash flow is what an owner of the entire business could pull out without starving it.

The figure below shows the relationship as a simple before-and-after: operating cash flow is the gross cash the business made; capex is the bite the business has to take out of it to keep producing; free cash flow is what is left.

![Free cash flow shown as operating cash flow minus capital expenditure leaving the cash available to owners](/imgs/blogs/cash-flow-statement-where-the-cash-really-comes-from-3.png)

#### Worked example: Northwind's free cash flow, two ways

Northwind generated **\$85 million of CFO** and spent **\$60 million of total capex**. The headline free cash flow is:

$$
\text{FCF} = \$85\text{m} - \$60\text{m} = \$25\text{m}
$$

But remember the maintenance-versus-growth split. If we believe only \$28m of that capex is mandatory maintenance and \$32m is discretionary growth, then the *owner's* free cash flow — the cash the existing business produces if it chose to stop growing — is much higher:

$$
\text{Owner FCF} = \$85\text{m} - \$28\text{m maintenance capex} = \$57\text{m}
$$

The same company can credibly be said to generate \$25m of free cash flow (if it keeps investing for growth) or \$57m (if it harvested the existing business). Neither is wrong; they answer different questions. The \$25m is the cash actually left after this year's full investment program; the \$57m is the cash-generating power of the assets already in place. A thoughtful analyst reports both and is explicit about which capex assumption drives the number.

*Free cash flow is not one number but a judgment, and the biggest lever in that judgment is how much of capex you treat as mandatory versus optional.*

### FCF yield versus earnings yield

Once you have a free cash flow number, you can ask what it is worth relative to the price you pay for the stock. Two related yields make the comparison concrete. **Earnings yield** is net income (or earnings per share) divided by the market value of the equity — it is simply the inverse of the price-to-earnings ratio, and it tells you the accounting return the business throws off per dollar of price. **Free cash flow yield** is free cash flow divided by the same market value — it tells you the *cash* return per dollar of price. Because earnings and free cash flow can diverge so sharply, these two yields can tell very different stories about the same stock, and the gap between them is itself diagnostic.

#### Worked example: FCF yield versus earnings yield on Northwind

Suppose the market values Northwind's equity at **\$1,000 million** (its market capitalization). Using the numbers we have built:

$$
\text{Earnings yield} = \frac{\$50\text{m net income}}{\$1{,}000\text{m}} = 5.0\%
$$

$$
\text{FCF yield} = \frac{\$25\text{m free cash flow}}{\$1{,}000\text{m}} = 2.5\%
$$

The earnings yield (5.0%) is double the headline free-cash-flow yield (2.5%), and the reason is entirely Northwind's heavy growth capex: the business earns \$50m but, after reinvesting \$60m to grow, only \$25m of cash is genuinely free this year. An investor focused on earnings sees a 5% return; an investor focused on free cash sees half that — but also knows the \$32m of growth capex should produce *more* earnings and cash in future years. Now compare a hypothetical no-growth version: if Northwind harvested the business and spent only the \$28m of maintenance capex, owner free cash flow would be \$57m, an owner FCF yield of 5.7% — *above* the earnings yield, because depreciation depresses reported earnings below true cash generation. *The gap between earnings yield and free cash flow yield is a measure of how much of a company's profit is being plowed back into growth rather than handed to owners — neither high nor low is inherently better, but you must know which you are buying.*

### FCFF versus FCFE: a teaser

There are two refinements you will meet the moment you start valuing companies, and it is worth planting the seeds now. **Free cash flow to the firm (FCFF)** is the cash available to *all* providers of capital — both lenders and shareholders — before any payments to debt holders. **Free cash flow to equity (FCFE)** is the cash available to *shareholders only*, after debt holders have been paid their interest and principal. The bridge between them is the cash flows to and from lenders: start with FCFF, subtract after-tax interest and net debt repayment, and you arrive at FCFE. Which one you use depends on whether you are valuing the whole enterprise or just the equity slice of it. The dedicated companion piece develops [FCFF versus FCFE](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe) in full; for now, just hold that "free cash flow" comes in a firm flavor and an equity flavor, and that mixing them up is one of the most common valuation errors beginners make.

## When earnings rise but cash falls: the red flag

Here is the scenario that justifies this entire piece. A company reports a beautiful income statement — revenue climbing, profit climbing, every headline metric pointing up. And yet its operating cash flow is flat, falling, or negative. How is that possible, and what does it mean?

Recall the indirect method. Net income flows into CFO, but it gets adjusted by changes in working capital. If a company's receivables are ballooning — if it is booking more and more revenue that it has not collected — then a growing slice of its reported profit is trapped in AR and never becomes cash. Profit goes up; cash from operations goes down. The income statement and the cash flow statement diverge, and that divergence is one of the most reliable early-warning signs in all of fundamental analysis.

There are benign explanations and malignant ones, and learning to tell them apart is the skill. The benign case: a fast-growing company that sells on credit will *naturally* see receivables and inventory grow as sales grow, consuming cash even as it thrives. Growing businesses are often cash-hungry, and a temporary gap between profit and cash during rapid expansion is normal. The malignant case: a company is **channel stuffing** — shipping product to distributors who do not need it, booking the shipments as revenue, recognizing profit, but never collecting the cash because the goods come back or the "sales" were never real demand. Receivables explode, profit looks great, and cash quietly evaporates. Eventually the receivables have to be written off, the prior "profits" reverse, and the stock collapses.

The figure below shows the shape of the warning: two lines over time, reported earnings marching steadily up while operating cash flow stalls and then turns down. When you see those two lines diverge and stay diverged, you have found something worth investigating.

![Reported earnings rising steadily over time while operating cash flow stalls and turns negative as receivables balloon](/imgs/blogs/cash-flow-statement-where-the-cash-really-comes-from-4.png)

#### Worked example: Northwind's troubled rival "Riverstone" stuffs the channel

Consider a competitor, **Riverstone Equipment**, over three years. Its income statement looks fantastic; its cash flow statement tells a darker story.

| | Year 1 | Year 2 | Year 3 |
|---|---|---|---|
| Revenue | \$200m | \$260m | \$340m |
| Net income | \$20m | \$28m | \$40m |
| Accounts receivable (balance) | \$40m | \$78m | \$150m |
| Increase in AR (cash impact) | — | −\$38m | −\$72m |
| Cash from operations | \$24m | −\$6m | −\$25m |

Look at the divergence. Net income *doubled* from \$20m to \$40m across three years — a growth story any investor would cheer. But operating cash flow went the *opposite* direction, from a healthy +\$24m to **negative \$25m**. The reason is right there in the receivables: AR exploded from \$40m to \$150m, nearly quadrupling while revenue merely doubled. Receivables are growing *far faster than sales*, which is the classic fingerprint of channel stuffing — Riverstone is booking "sales" to distributors who are not paying, so the profit is real on paper and fictional in cash. By year 3, the company is profitable and bleeding \$25m of cash a year. This is exactly the gap that destroyed companies in real accounting scandals, the dynamic forensic analysts hunt for when they study cases like [Enron's 2001 collapse](/blog/trading/finance/enron-2001-accounting-fraud).

*When receivables grow much faster than revenue and operating cash flow diverges from net income, the "profit" is being manufactured faster than it can be collected — treat it as a warning until proven otherwise.*

The mechanics of *why* working capital moves cash the way it does is worth one dedicated figure, because the signs trip up everyone at first. Receivables and inventory going up drains cash; payables going up provides cash. The grid below lays out all the directions at once.

![A grid showing how rising and falling receivables inventory and payables each move operating cash up or down](/imgs/blogs/cash-flow-statement-where-the-cash-really-comes-from-5.png)

## Cash conversion: the earnings-quality ratio

If you remember one number from this entire piece, make it **cash conversion** — also called the cash conversion ratio or the cash flow to net income ratio:

$$
\text{Cash Conversion} = \frac{\text{Cash from Operations}}{\text{Net Income}}
$$

It answers the most important question in earnings quality in a single fraction: *for every dollar of profit the company reports, how many dollars of actual operating cash does it produce?* A ratio comfortably **above 1.0** is the signature of high-quality earnings — the company is converting its accounting profit into real cash, often more than the profit itself because of non-cash add-backs like depreciation. A ratio persistently **below 1.0**, and especially one that is *declining*, is a yellow-to-red flag: profit is being reported that is not turning into cash.

The benchmarks are rough but useful. Over a full business cycle, a healthy company's cumulative CFO should at least match and usually exceed its cumulative net income, so multi-year cash conversion should sit around or above 1.0. A single weak year proves nothing — a heavy investment in working capital can drag one year's ratio down legitimately. But a *pattern* of CFO running well below net income, year after year, means the gap is structural, and the most common structural reason is that reported earnings are softer than they look.

#### Worked example: cash conversion separates Northwind from Riverstone

Put our two companies side by side using the most recent year:

- **Northwind:** CFO \$85m ÷ net income \$50m = **1.70.** For every dollar of reported profit, Northwind produces \$1.70 of operating cash. This is excellent — the earnings are not just real, they understate the cash generation because depreciation is a large non-cash drag on reported profit.
- **Riverstone (year 3):** CFO −\$25m ÷ net income \$40m = **−0.63.** For every dollar of reported profit, Riverstone *loses* sixty-three cents of operating cash. The earnings are, in cash terms, worse than fictional — the business is consuming cash while claiming record profits.

A reader who looked only at the income statements would see two profitable, growing companies and might even prefer Riverstone for its faster earnings growth. A reader who computes cash conversion sees one cash machine and one cash furnace. *Cash conversion is the fastest single test of whether reported profit is real, and the gap between Northwind's 1.70 and Riverstone's −0.63 is the gap between an investment and a trap.*

The figure below contrasts cash conversion across a quality company and a low-quality one, making the divergence in earnings quality immediate.

![Cash conversion ratios contrasted across a high quality firm well above one and a low quality firm below one](/imgs/blogs/cash-flow-statement-where-the-cash-really-comes-from-7.png)

### The cash conversion cycle: how long cash stays trapped

Cash conversion (the ratio) tells you *whether* profit becomes cash. A closely related operational metric — the **cash conversion cycle (CCC)** — tells you *how long* cash stays trapped in the working-capital pool before it comes back. It is built from three timing measures, all expressed in days:

- **Days sales outstanding (DSO)** — how many days, on average, it takes to collect cash after making a sale. Computed as accounts receivable divided by revenue, times 365. A rising DSO means customers are paying more slowly, which drains cash and is exactly the symptom that shows up when a company starts channel stuffing.
- **Days inventory outstanding (DIO)** — how many days inventory sits in the warehouse before it sells. Computed as inventory divided by cost of goods sold, times 365. Rising DIO ties up more cash in unsold goods.
- **Days payable outstanding (DPO)** — how many days the company takes to pay its own suppliers. Computed as accounts payable divided by cost of goods sold, times 365. A *higher* DPO is good for cash: the longer you can defer paying suppliers, the longer you hold onto cash.

Put them together and the cash conversion cycle is:

$$
\text{CCC} = \text{DSO} + \text{DIO} - \text{DPO}
$$

In plain English: the number of days between when you pay cash out for inventory and when you finally collect cash from the customer. A *short* cycle means cash whips through the business quickly and the company needs little working capital to grow. A *long* cycle means cash is stuck for months, and growth devours cash. The best businesses in the world — think of a retailer that sells goods for cash before it has to pay the supplier — can run a *negative* cash conversion cycle, meaning customers fund the business's growth for free. Understanding the CCC is what turns the working-capital line on the cash flow statement from a number into a story about how the business actually operates.

#### Worked example: Northwind's and Riverstone's cash conversion cycles

Take the receivables side, where the contrast is starkest. Northwind has \$60m of receivables on \$400m of revenue, so its DSO is (\$60m ÷ \$400m) × 365 ≈ **55 days** — customers pay in under two months, which is normal for industrial sales on credit terms. Riverstone, in year 3, has \$150m of receivables on \$340m of revenue, so its DSO is (\$150m ÷ \$340m) × 365 ≈ **161 days** — customers are taking more than five months to pay, and the trend across the three years (from roughly 73 days to 161) is the smoking gun. Receivables are not just large; the *collection period is exploding*, which is the operational fingerprint of revenue being booked far ahead of any cash. *A receivables balance is just a number, but DSO turns it into a clock — and a clock that keeps slowing tells you the cash is moving the wrong way.*

## How the statement ties to the balance sheet

A loose end worth closing: the cash flow statement does not float free of the other two statements. It is wedged precisely between them, and the linkages are what make the three-statement model an internally consistent machine. The companion piece on [how the three financial statements connect](/blog/trading/equity-research/how-the-three-financial-statements-connect) develops this in full, but the cash-specific links are worth stating here because they are what make the statement so hard to fake.

The most important tie is the one we have already met: **the net change in cash on the cash flow statement equals the change in the cash line on the balance sheet.** Start-of-year cash plus the period's net change equals end-of-year cash, and that end-of-year number is what sits in the balance sheet's current assets. If those do not match, the statements are wrong — full stop.

The second tie runs through net income and depreciation. **Net income, the top line of the indirect method, flows into the balance sheet's retained earnings** (profit the company kept rather than paid out). And **depreciation, the largest add-back in CFO, simultaneously reduces the carrying value of property, plant, and equipment** on the balance sheet. Every line on the cash flow statement is, in this sense, the *change* in some balance sheet account between two periods. The cash flow statement is literally the bridge that explains how the balance sheet moved from last year to this year. That is why it is so robust: to fake the cash flow statement convincingly, you would have to fake both the income statement and the balance sheet in perfectly consistent ways, and still produce a cash balance a bank could confirm. It can be done — Wirecard did it — but it is far harder than simply flattering an earnings number.

## Common misconceptions

A handful of confusions trip up almost everyone learning to read this statement. Clearing them is most of what separates someone who can recite the three sections from someone who can actually use them.

**"Depreciation is a source of cash."** No. You hear people say "add back depreciation to get cash," and they start to imagine depreciation somehow *generates* cash. It does not. Depreciation is a non-cash *expense*; adding it back in the indirect method merely *reverses* a subtraction that never involved cash. A company with huge depreciation does not have more cash because of it — it has more cash *relative to its reported profit* because the profit was artificially depressed by a paper charge. The cash itself came from selling products, not from the depreciation entry.

**"Positive net income means the company is generating cash."** As the entire Riverstone example demonstrates, no. A company can report record profits and burn cash every quarter. Net income and operating cash flow are different measurements, computed differently, and they routinely diverge — sometimes for legitimate reasons (growth, working-capital investment), sometimes for sinister ones (channel stuffing, aggressive revenue recognition). Never assume profit equals cash; check.

**"Free cash flow is an objective, single number."** It is not, and anyone who quotes one precise FCF figure without saying how they treated capex is hiding a judgment. The headline FCF (CFO minus total capex) is well-defined, but the *economically meaningful* FCF depends on the maintenance-versus-growth capex split, which is an estimate. Two honest analysts can compute materially different free cash flow for the same company and both be defensible. Always ask: which capex did they subtract, and why?

**"Stock-based compensation is free because it does not cost cash."** This is the most contested one, and the honest answer is: it costs no *cash*, but it is not free. When a company pays employees in stock, it does not spend cash — so SBC is correctly added back in computing operating cash flow. But it dilutes existing shareholders: the share count rises, and each existing owner's slice of the company shrinks. Treating CFO (which adds SBC back) as if it accrues entirely to current shareholders overstates their economics. A careful analyst either subtracts SBC again when valuing the equity, or accounts for the dilution directly. Companies that lean heavily on SBC and then point to their fat operating cash flow are, intentionally or not, flattering the picture.

**"Negative cash flow is always bad."** Not at all — and confusing the three sections causes this error. Negative *operating* cash flow in a mature company is a serious concern. But negative *investing* cash flow is normal and usually good: it means the company is investing in its future. Negative *financing* cash flow is often excellent: it means the company is returning cash to owners. Northwind's total cash fell \$10m in a year it was thriving. You must read which section is negative and why before you judge it.

**"The direct method and indirect method give different operating cash flow."** They do not. Both arrive at the identical CFO figure — they are two presentations of the same total. The direct method lists actual cash receipts and payments; the indirect method reconciles from net income. The number at the bottom is the same. The indirect method simply dominates in practice because it ties cleanly to the income statement and reveals the working-capital adjustments analysts want to see.

## How it shows up in real markets

The abstractions above are not academic. The single most consequential pattern in this piece — profit that does not convert to cash — sits at the center of nearly every major accounting scandal and many ordinary investment disappointments. Here is how it shows up.

**Enron and the cash-versus-earnings gap.** Enron reported soaring profits in the late 1990s through aggressive mark-to-market accounting and off-balance-sheet vehicles, but its operating cash flow never came close to supporting its reported earnings, and the company relied on continuous external financing to survive. The gap between booked profit and real cash was visible in the statements years before the collapse to anyone who tracked cash conversion rather than headline EPS. The full anatomy is in the dedicated [Enron 2001 case study](/blog/trading/finance/enron-2001-accounting-fraud), but the cash-flow lesson is the durable one: when a company is wildly profitable on paper yet perpetually needs to raise outside money, ask where the cash is.

**Wirecard and fake cash itself.** Most frauds fake *earnings*; Wirecard went further and faked the *cash* — claiming roughly €1.9 billion sat in trustee-controlled escrow accounts in Asia that, it turned out, did not exist. This is the nightmare scenario for the "cash is a fact" thesis: the one number we trust most can itself be fabricated if no independent party actually confirms the bank balance. The case is a reminder that the cash flow statement is *harder* to fake, not *impossible* to fake — and that the final safeguard is someone physically confirming the money is where the statement says it is. The forensic detail is in the [Wirecard case study](/blog/trading/finance/wirecard-the-german-fintech-fraud). The lesson cuts both ways: trust the cash flow statement more than the income statement, but verify that the ending cash balance is real.

**Capital-intensive businesses and the capex question.** Look at any telecom, airline, or semiconductor manufacturer and you see the maintenance-versus-growth capex problem in the wild. These companies generate large operating cash flow and spend nearly all of it on capex, so their free cash flow is thin and entirely sensitive to how much of that capex you call "maintenance." A semiconductor firm building a new fab is spending enormous growth capex that depresses FCF today for capacity that pays off for a decade. An investor who naively subtracts all capex sees a business that generates no free cash; one who separates maintenance from growth sees a cash machine investing for the future. The judgment is everything.

**High-growth software and the SBC debate.** Many fast-growing software companies report strong operating cash flow precisely because they add back enormous stock-based compensation. Their *cash* generation is genuine, but the dilution is real and ongoing, and the gap between "cash flow if SBC were free" and "cash flow accounting for dilution" can be the difference between an expensive stock and a cheap one. This is the modern frontier of the cash-versus-earnings debate, and it is why sophisticated investors look at free cash flow *per share* and track dilution, not just aggregate FCF.

**Distressed companies and the financing tell.** A company sliding toward trouble often shows a telltale financing pattern: operating cash flow weakening, investing cash flow shrinking as it cuts capex to conserve cash, and financing cash flow turning sharply positive as it scrambles to raise debt or equity to plug the gap. When a mature business that should be returning cash to owners is suddenly raising cash to survive, the cash flow statement has told you the story before the income statement admits it.

## When this matters and further reading

The cash flow statement matters most precisely when the income statement looks best — when profits are soaring, the narrative is compelling, and everyone is excited. That is exactly when you should turn to the cash flow statement and ask the unglamorous question: is this profit actually becoming money? If operating cash flow tracks net income and cash conversion sits at or above one, the story checks out. If earnings are climbing while operating cash flow stalls and receivables balloon, you have found a reason to be careful, whatever the income statement says.

The discipline is simple to state and hard to consistently apply: read all three statements, but trust the cash flow statement most, and within it, watch operating cash flow versus net income above everything else. Profit is an opinion shaped by accounting choices; cash is a fact you can confirm.

To go deeper, the natural next steps within this series are the [income statement line by line](/blog/trading/equity-research/income-statement-line-by-line-revenue-to-net-income) (where the net income that starts the indirect method comes from), [how the three financial statements connect](/blog/trading/equity-research/how-the-three-financial-statements-connect) (the linkages that make the model internally consistent), [quality of earnings: accruals, one-offs, and red flags](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) (a deeper treatment of the cash-versus-profit gap and how to quantify it), and [free cash flow: FCFF versus FCFE](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe) (which turns the free cash flow we computed here into the inputs of a valuation). Master those, and you will read a company's cash flow statement the way a professional does: not as a footnote to the income statement, but as the place where the truth finally has to show up.
