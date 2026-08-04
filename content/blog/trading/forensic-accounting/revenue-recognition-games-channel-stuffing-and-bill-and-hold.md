---
title: "Revenue recognition games: how tomorrow's sales become today's profit"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "A forensic-accounting deep dive into channel stuffing, bill-and-hold sales, premature recognition, and percentage-of-completion abuse, with a sourced Sunbeam case study and practical statement-line tests."
tags: ["forensic-accounting", "revenue-recognition", "channel-stuffing", "bill-and-hold", "financial-statements", "fraud-detection", "sunbeam", "accounting-red-flags"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Revenue recognition games make a weak quarter look strong by recording a sale before the customer has really accepted the product, the service, or the risk.
>
> - Channel stuffing pushes extra inventory into distributors with discounts, extended terms, or return rights. The invoice rises now; the customer's need has not.
> - Bill-and-hold can be legitimate, but only when the customer—not the seller—has a substantive reason to delay delivery and the seller has genuinely completed its performance.
> - Premature recognition and percentage-of-completion abuse turn incomplete work into completed revenue. The accounting entry can look ordinary while the contract economics are not.
> - The best forensic clue is a bridge: revenue up without cash, receivables or contract assets up faster than sales, returns and credits appearing later, or next-quarter sales going missing.
> - In the SEC's Sunbeam proceeding, the agency reported that improper 1997 practices included $14 million of second-quarter bill-and-hold sales and $29 million of fourth-quarter bill-and-hold sales; Sunbeam later restated 1997 income from $189 million to about $93 million.

Imagine a sales manager who is two days short of a quarterly target. A retailer will take 1,000 blenders eventually, but only 400 fit in its warehouse this month. The manager offers a discount, invoices all 1,000, promises to store the remaining 600, and tells the finance team that the quarter is saved.

The cash may arrive later. The product may come back. The retailer may never have accepted the risk. Yet the income statement can already show the entire sale if the accounting conclusion is wrong. That is the central forensic problem: the journal entry is not necessarily exotic. The timing and the hidden side agreement are.

![A statement-line bridge showing how a pressured seller can pull future revenue into the current quarter, while cash and customer demand lag behind.](/imgs/blogs/revenue-recognition-games-channel-stuffing-and-bill-and-hold-1.webp)

The figure is the mental model for this post. A genuine sale moves through three gates: a real customer commitment, a transfer of control, and an amount the seller can reasonably expect to keep. A revenue game jumps one of those gates. We will build the idea from zero, then inspect four common routes—channel stuffing, bill-and-hold, premature recognition, and percentage-of-completion abuse—before replaying the Sunbeam case using the SEC's dated account.

## Foundations: what a sale is supposed to mean

### Revenue is not the same as an invoice

*Revenue* is the income a company earns from delivering goods or services to customers. It is not simply every invoice printed by the billing system. An *invoice* is a request for payment; revenue is an accounting conclusion about what the company has performed and what consideration it is entitled to keep.

That distinction matters because the income statement is organized by period. A quarter ending on September 30 includes September performance, not October performance merely because someone created an invoice on September 30. The accrual system tries to match economic activity with the period in which it occurred. This is why a company can have revenue without cash—credit sales create accounts receivable—and cash without current revenue—an upfront deposit can create a contract liability until the company performs.

### The five questions a forensic accountant asks

The exact accounting literature depends on the reporting framework, contract, and industry. The practical questions below are a useful plain-English map, not a substitute for applying the relevant standard.

| Question | What it tests | Typical evidence |
| --- | --- | --- |
| Is there a contract? | A customer has enforceable rights and obligations. | Signed order, purchase order, master agreement |
| What was promised? | The *performance obligation*, meaning the distinct good or service to deliver. | Product list, milestones, acceptance clause |
| What price is really expected? | *Transaction price*, including rebates, returns, credits, and variable consideration. | Side letters, discount approvals, return history |
| Has control transferred? | The customer can direct use and obtain benefits. | Shipping terms, acceptance, title, risk, installation |
| Is the amount collectible and final? | Revenue is not booked for a sale that is effectively a refundable deposit or consignment. | Credit file, subsequent cash, returns, cancellations |

The phrase *control transferred* is more useful than the old shorthand “we shipped it.” Shipping is evidence, not a magic spell. A customer who can return everything, has not accepted the goods, and has no obligation to pay may not have assumed the meaningful risks and benefits of ownership.

### The four statement lines that carry the trail

The *income statement* reports revenue, expenses, and profit for a period. The *balance sheet* reports what the company owns and owes at a date. The *cash-flow statement* reconciles profit to cash movement. The forensic trail usually crosses all three.

| Line | What a genuine growth quarter often needs | What a timing game can leave behind |
| --- | --- | --- |
| Revenue | Customer demand and repeatable orders | A one-quarter spike followed by a hole |
| Accounts receivable | Cash collection on normal terms | Receivables growing much faster than sales |
| Inventory | Shipment and acceptance reduce seller inventory | Goods remain in seller-controlled warehouses |
| Contract assets / liabilities | Work and billing move with performance | Unbilled claims or deposits grow out of proportion |
| Returns, rebates, credits | Estimates reflect expected concessions | A later wave of credits reverses the earlier profit |

The red flag is not “receivables are high.” A fast-growing company can legitimately have high receivables. The question is whether the balance makes sense relative to the stated sales terms, customer concentration, collection history, and next-period reversals.

![The three-statement footprint of a questionable sale: revenue and receivables rise first, cash lags, and later returns or inventory reversals reveal the timing problem.](/imgs/blogs/revenue-recognition-games-channel-stuffing-and-bill-and-hold-2.webp)

### Debit and credit, without the intimidation

A *journal entry* is the accounting record of a transaction. A *debit* and a *credit* are the two sides of that record; they do not inherently mean “good” or “bad.” For a credit sale, a simplified entry is:

```ledger
Dr Accounts receivable       $100
    Cr Revenue                         $100
```

If the goods cost $60, the seller also records:

```ledger
Dr Cost of goods sold          $60
    Cr Inventory                         $60
```

The $40 difference is gross profit. A fraudulent or premature sale can use perfectly normal entries. The forensic question is whether the entry describes a completed economic exchange at the reporting date.

#### Worked example: the ordinary credit sale

Suppose a company ships and the customer accepts 10 machines at an illustrative $100 each. The invoice is $1,000, and the machines cost $600 in total.

1. Revenue is $1,000: `10 × $100`.
2. Accounts receivable rises by $1,000 because the customer has not paid.
3. Inventory falls by $600.
4. Cost of goods sold rises by $600.
5. Gross profit is `$1,000 − $600 = $400`, or a 40% gross margin.

The journal entries are:

```ledger
Dr Accounts receivable     $1,000
    Cr Revenue                        $1,000

Dr Cost of goods sold         $600
    Cr Inventory                         $600
```

The numbers are illustrative, but the structure is real: revenue, receivable, inventory, and cost all tell the same story. The intuition is that a sale is a four-part event, not just a top-line number.

## The first game: channel stuffing

*Channel stuffing* is the practice of shipping or invoicing more product to distributors than the downstream market currently needs, often near a reporting deadline. A distributor is an intermediary that buys from the manufacturer and sells onward. The manufacturer can show a strong quarter even though the end customer has not bought more.

Channel stuffing is not automatically illegal or fraudulent. A company may legitimately sell more because a distributor is building inventory for a known launch. The forensic issue is whether the seller used unusual inducements or concealed terms that make the “sale” reversible, uncollectible, or economically a consignment.

### Why management is tempted

Public-company compensation, debt covenants, analyst expectations, and acquisition plans can all create pressure around a quarter-end number. The temptation is strongest when a manager believes the product will sell eventually, so the intervention feels like borrowing from next quarter rather than inventing demand.

But the next quarter is not free. The distributor now has excess stock. It may stop ordering, demand a rebate, return the goods, or sell them at a discount that damages the brand. The seller has moved the date of the invoice while moving the commercial risk nowhere.

![A clean channel versus stuffed channel: real end-customer pull flows through the distributor, while a stuffed channel stops at the intermediary and sends risk back to the seller.](/imgs/blogs/revenue-recognition-games-channel-stuffing-and-bill-and-hold-3.webp)

#### Worked example: a distributor order that borrows from next quarter

Assume an illustrative manufacturer normally sells 400 units in September and 400 in October. Each unit invoices at $100 and costs $60. On September 29, management offers a distributor a 10% rebate and a 90-day payment term if it accepts 1,000 units.

If the arrangement is a genuine sale with no substantive return right and the customer has accepted control, September revenue could be $100,000. Gross profit before the rebate would be `$100,000 − $60,000 = $40,000`. The 10% rebate is $10,000, so expected net revenue is $90,000 and expected gross profit is `$90,000 − $60,000 = $30,000`.

But suppose the distributor only has demand for the normal 400 units and returns 600 in October. The seller has pulled 600 units of future volume into September: $60,000 of gross invoice value, $36,000 of cost, and—before the return estimate—$24,000 of apparent gross profit. The 90-day term also means September cash is zero.

The right forensic schedule is not “September sales = $100,000.” It is:

| September line | Illustrative amount |
| --- | ---: |
| Invoice value | $100,000 |
| Expected rebate | $(10,000) |
| Expected returns on 600 units | $(60,000) |
| Revenue that reflects expected retained sales | $30,000 |

The exact accounting depends on the contract and reporting framework, but the commercial test is clear: a $100,000 invoice is not $100,000 of durable demand. The intuition is that channel stuffing inflates the current period by making the distributor warehouse carry tomorrow’s unsold inventory.

### Tells in the data room

An investigator should request customer-level shipment and return data, not only the general ledger. Look for quarter-end shipments that are unusually large, unusual discounts, payment terms that lengthen at the same time as sales accelerate, freight paid by the seller, and products sitting in third-party warehouses that the seller can still redirect.

Compare shipments with sell-through. *Sell-in* is what the manufacturer ships to the distributor. *Sell-through* is what the distributor sells to the final customer. When sell-in jumps but sell-through does not, the manufacturer may be stocking the channel rather than meeting demand.

Other useful comparisons include sales by day in the final week of a quarter, credit memos issued in the following 30–90 days, receivables aging, and gross margin by customer. A distributor accepting a large shipment at a deep discount and then returning it shortly after period-end is not proof by itself, but it is exactly the pattern that deserves contract-level testing.

## The second game: bill-and-hold

A *bill-and-hold* arrangement is a sale in which the seller bills the customer but keeps physical possession of the goods for a period. There can be a legitimate business reason: the customer’s warehouse is being renovated, a seasonal product must be manufactured early, or the customer requested delivery on a specified future date.

The physical location is not the only issue. The key question is whether the customer has obtained control while the seller is acting as a custodian, or whether the seller is still carrying the meaningful risks. A customer-requested delay, a substantive business purpose, a fixed commitment to buy, a separately identified product, readiness for immediate transfer, and no ability for the seller to use the product for another customer are the kinds of facts an accountant tests. The applicable literature and facts control; this list is a forensic checklist, not a universal safe harbor.

![A bill-and-hold decision path: customer request, substantive reason, segregated finished goods, and no seller substitution risk must all support recognition.](/imgs/blogs/revenue-recognition-games-channel-stuffing-and-bill-and-hold-4.webp)

### The difference between custody and control

Think about buying a tailored suit. If you have paid for your specific finished suit, the tailor has set it aside, and you ask the tailor to store it until your move, the tailor may be holding your property. If the tailor has merely cut generic cloth, can sell it to someone else, and hopes you will eventually buy, the economic sale is not complete.

That is why “the customer was billed” and “the title passed” are weak answers without the rest of the facts. A side agreement can return the risk to the seller even when the invoice and legal title appear to move forward.

#### Worked example: legitimate storage versus hidden consignment

Use illustrative numbers. A customer orders 100 seasonal heaters at $200 each, pays 30% upfront, and asks the seller to store the finished, individually tagged units until November 1 because its warehouse is being rebuilt. The customer cannot cancel, the seller cannot substitute the heaters, and the customer bears loss risk after the agreed transfer date.

The invoice is $20,000. The upfront cash is `$20,000 × 30% = $6,000`; the remaining receivable is $14,000. If the arrangement meets the applicable recognition criteria, the seller may recognize the appropriate revenue when control transfers, while the storage service may be a separate obligation if one exists.

Now change one fact: the customer may return any unsold heaters, the seller pays storage and insurance, and the seller can redirect the units to another buyer. The same $20,000 invoice now resembles a consignment or a contingent sale. The seller has not convincingly transferred the risk of ownership.

Possible simplified entries for the first, completed-sale fact pattern are:

```ledger
Dr Cash                         $6,000
Dr Accounts receivable         $14,000
    Cr Revenue                           $20,000

Dr Cost of goods sold          $12,000
    Cr Inventory                         $12,000
```

For the second pattern, the safer commercial conclusion is to keep the goods in inventory and record the $6,000 as a contract liability or deposit until the recognition conditions are met. The exact account names depend on the framework and contract. The intuition is that “bill” describes paperwork; “hold” describes custody. Neither proves transfer of control.

### What to inspect

Ask who requested the delay and when. A customer request recorded after management proposed the program is weaker than a contemporaneous warehouse constraint supported by emails, construction records, or a delivery schedule. Inspect whether the goods are complete, separately labeled, and physically protected from substitution. Trace insurance, storage, freight, and damage claims. Confirm whether the customer can cancel, return, or exchange the goods.

Read the credit notes after year-end. A bill-and-hold population with a high return rate is not merely a sales success delivered late; it may be a quarter-end entry that never became a sale.

## The third game: premature recognition and side letters

*Premature recognition* means recording revenue before the company has completed the performance required by the contract. It can happen at shipment instead of acceptance, when a customer still has a substantive installation obligation, or before a milestone is achieved. A *side letter* is a separate written or unwritten agreement that changes the apparent terms—such as a return right, price protection, cancellation option, or promise to repurchase.

The danger is that the main contract in the contract-management system can look clean while the real deal lives in emails, sales-chat messages, credit memos, or an executive promise to “make the customer whole.” Forensic work therefore follows behavior and incentives, not just signed documents.

![A before-and-after journal-entry bridge: the premature entry credits revenue and debits a receivable, while the corrected entry leaves an advance or contract liability until acceptance.](/imgs/blogs/revenue-recognition-games-channel-stuffing-and-bill-and-hold-5.webp)

#### Worked example: acceptance is the missing event

Suppose a software vendor sells an illustrative $120,000 implementation package. The customer pays $30,000 upfront. The contract says the customer must accept the configured system after a user-acceptance test scheduled for October 15. On September 30, the system is installed but the test has not occurred.

If the vendor prematurely records all revenue, it might post:

```ledger
Dr Cash                         $30,000
Dr Accounts receivable         $90,000
    Cr Revenue                          $120,000
```

The commercial facts say the vendor has not yet cleared the acceptance gate. A more defensible interim shape is:

```ledger
Dr Cash                         $30,000
    Cr Contract liability                 $30,000
```

The remaining $90,000 is not a receivable merely because the vendor expects to bill it someday. After successful acceptance, the revenue entry can reflect the performance completed under the contract. The exact allocation may include separate implementation and support obligations.

The forensic bridge is the difference between “installed” and “accepted.” Search the customer’s test results, unresolved defects, acceptance certificate, and post-period invoices. The intuition is that an invoice can be legally issued before the customer has received the promised economic benefit.

### The common side-letter patterns

| Hidden term | Why it changes the accounting question |
| --- | --- |
| “Return it if your customers do not buy it” | Customer may not bear inventory risk |
| “We will protect your margin” | Consideration is not fixed and may become a rebate |
| “You can cancel before payment” | Enforceable obligation may not exist |
| “We will buy it back next month” | Sale may be financing, not a completed sale |
| “We will finish installation after year-end” | Performance obligation remains incomplete |

None of these phrases is automatically proof of fraud. They are prompts to reconcile the legal contract, sales incentive, customer understanding, and journal entry.

## The fourth game: percentage-of-completion abuse

Long-term construction, engineering, and software contracts create a different timing risk. *Percentage-of-completion* is a method that recognizes revenue as performance occurs rather than waiting for the entire contract to finish, when the applicable accounting criteria are met. It can match revenue with work delivered, but it requires estimates: total contract cost, remaining cost, approved change orders, variable consideration, and expected losses.

The abuse is not “using estimates.” Every long project uses estimates. The abuse is making the project appear further complete than it is, understating expected costs, or treating an unapproved change order as if it were firm. A small change in the estimated total cost can move the completion percentage and therefore current revenue.

![A percentage-of-completion cost bridge: current revenue depends on measured cost-to-date divided by estimated total cost, so understated remaining cost pulls future profit into today.](/imgs/blogs/revenue-recognition-games-channel-stuffing-and-bill-and-hold-6.webp)

#### Worked example: the estimate that moves profit into today

Consider an illustrative fixed-price contract worth $1,000,000. At September 30, the contractor has incurred $300,000 of costs. The original estimate of total cost is $600,000.

Under a cost-to-cost approach, estimated completion is `$300,000 / $600,000 = 50%`. Cumulative revenue is `50% × $1,000,000 = $500,000`. Cumulative gross profit is `$500,000 − $300,000 = $200,000`, assuming no prior recognized amounts.

Now suppose an overlooked design problem means total cost will really be $800,000. Completion is `$300,000 / $800,000 = 37.5%`. Cumulative revenue should be `$1,000,000 × 37.5% = $375,000`, and cumulative gross profit should be `$375,000 − $300,000 = $75,000`.

The difference is $125,000 of revenue and $125,000 of apparent profit pulled forward by the understated cost estimate. A realistic investigation would also test whether the $200,000 of remaining cost is supported by supplier quotes, labor schedules, approved change orders, and defect logs.

The simplified journal entry for the 50% estimate might be:

```ledger
Dr Contract asset / accounts receivable   $500,000
    Cr Revenue                                      $500,000
Dr Cost of goods sold                       $300,000
    Cr Work in progress / inventory                  $300,000
```

The entry is illustrative and account names vary. The intuition is that percentage-of-completion turns a forecast into a reported number; the forensic job is to audit the forecast’s denominator.

### The contract-asset trap

A *contract asset* is a right to consideration for work already performed when the right is still conditional on something other than the passage of time. It is not the same as an unconditional receivable. A contractor can therefore report revenue and a contract asset before it can issue a bill.

That accounting can be entirely legitimate. The red flag is a contract-asset balance that grows faster than revenue, remains unbilled for long periods, or depends on disputed milestones. Trace it to signed progress certificates, customer correspondence, subsequent invoices, and cash receipts. A spreadsheet that says “90% complete” is not evidence of customer acceptance.

## Why the games work on the income statement

Revenue timing games exploit the difference between a period and a business. A business may have multi-quarter demand, but the reporting system forces management to label each quarter. If management moves the label, the current period looks stronger even if lifetime economics are unchanged—or worse, if the lifetime economics deteriorate because the customer received a concession.

### The period-shift equation

An explanatory abstraction—not a formula stated by a particular accounting standard—is:

$$\text{reported revenue}_t = \text{economic sales earned}_t + \text{sales pulled from future periods}_t - \text{sales deferred to future periods}_t$$

Here $t$ is the reporting period. The abstraction helps us see why a company can report growth without creating growth. If $10 million of revenue is pulled from next quarter into this quarter, the current quarter gains $10 million while next quarter starts with a hole of roughly $10 million before normal growth.

#### Worked example: the next-quarter hole

Assume illustrative underlying demand is $80 million per quarter. In Q3, management accelerates $10 million of Q4 shipments. Reported Q3 revenue becomes $90 million; reported Q4 begins with only $70 million of the underlying demand left.

| Period | Underlying demand | Timing shift | Reported revenue |
| --- | ---: | ---: | ---: |
| Q3 | $80m | +$10m | $90m |
| Q4 | $80m | −$10m | $70m |

Q3 growth looks like 12.5% versus a hypothetical $80 million baseline: `$90m / $80m − 1 = 12.5%`. Q4 looks like a 12.5% decline: `$70m / $80m − 1 = -12.5%`. The company has not created a dollar of lifetime demand; it has changed the calendar.

If the acceleration required a 10% concession, Q3 net revenue is only `$10m × (1 − 10%) = $9m` for the shifted portion. That is why a growth headline can coexist with weaker unit economics. The intuition is that pulling sales forward is a loan from the next quarter, and the interest is usually a discount, a return, or a lost customer relationship.

### Why cash can mislead too

Investigators often say “follow the cash,” correctly but incompletely. A company can collect cash on a questionable sale, especially if it offers a discount or if the customer needs the product eventually. Cash confirms that money moved; it does not by itself prove that the seller recognized the right amount in the right period.

Conversely, a legitimate credit sale can create no immediate cash. That is why the strongest analysis triangulates: contract terms, shipment and acceptance, receivable aging, collections, subsequent returns, margin, inventory, and customer sell-through.

![A red-flag dashboard linking revenue growth to receivables, cash conversion, returns, and customer sell-through; no single ratio is treated as proof.](/imgs/blogs/revenue-recognition-games-channel-stuffing-and-bill-and-hold-7.webp)

## A forensic workflow for finding pulled-forward revenue

### Step 1: freeze the period boundary

Obtain the quarter-end calendar, shipping cut-off policy, order-entry logs, and the last 10 business days of shipments. Normalize time zones and delivery dates. Quarter-end activity is not suspicious merely because it is high; it is suspicious when it is unusually concentrated, paired with unusual terms, or reversed soon afterward.

### Step 2: build a contract population

Start with the largest and most unusual transactions. Stratify by customer, product, sales representative, shipping location, discount, payment term, and return history. Include manual journal entries posted after the normal close process. Compare what the general ledger says with what the customer contract says.

### Step 3: test the four transfer gates

For each selected sale, document: the enforceable commitment, the promised good or service, the price after variable consideration, and the evidence of transfer. Ask who can cancel, who pays storage and freight, who bears damage, whether the seller can substitute the goods, and whether acceptance remains outstanding.

### Step 4: trace forward

Subsequent events are powerful. Search the next 30, 60, and 90 days for cash, credit memos, returns, price protection, replacement shipments, write-offs, and customer complaints. A return is not automatically evidence that the original sale was improper, but a concentrated wave of returns from quarter-end deals is a fact pattern that needs a revised estimate or a corrected recognition date.

#### Worked example: an investigator’s $1 million sample

Suppose an illustrative sample contains 100 quarter-end invoices totaling $1,000,000. The investigator finds:

- $600,000 shipped in the last three days of the quarter;
- $200,000 with payment terms extended from 30 to 90 days;
- $150,000 returned in the next 45 days;
- $100,000 with a side email promising a rebate not in the contract.

These amounts overlap; do not add them mechanically. The right next step is to trace each invoice to the contract and customer confirmation, then quantify the population-level exposure. If the $150,000 returns are a subset of the $600,000 late shipments, the return rate for that late-shipment subset is `$150,000 / $600,000 = 25%`. That is a sample signal, not a conclusion about the whole company.

The journal-entry review should identify whether the original $1,000,000 entry included estimated returns and rebates. The intuition is that forensic accounting is an evidence chain: unusual shipment, unusual term, subsequent reversal, and missing disclosure become persuasive together.

### Step 5: reconcile operational data to the statements

The sales subledger should reconcile to the general ledger. The warehouse should reconcile to inventory. The customer-confirmation population should reconcile to accounts receivable. Differences are not automatically fraud—they can be timing, currency, or system issues—but unexplained differences at the quarter boundary deserve escalation.

## What the ratios can and cannot tell you

Ratios are screening tools. They narrow the population; they do not establish intent.

| Screen | Calculation | What a rise may mean | Necessary follow-up |
| --- | --- | --- | --- |
| Receivables growth | Change in A/R ÷ change in revenue | Slower collection or credit-heavy growth | Aging, cash receipts, terms |
| Days sales outstanding | A/R ÷ revenue × days | More days to collect | Customer confirmations and disputes |
| Return rate | Returns ÷ gross shipments | Weak acceptance or over-shipment | Credit notes and side agreements |
| Contract assets | Contract assets ÷ revenue | More unbilled performance | Milestone evidence and approval |
| Gross margin | Gross profit ÷ revenue | Pricing power or concessions | Discounts, rebates, product mix |

#### Worked example: days sales outstanding without false precision

Assume illustrative quarter revenue is $12 million, ending accounts receivable is $6 million, and the quarter has 90 days. A simple ending-balance screen gives `$6m / $12m × 90 = 45 days`. If the prior quarter was $10 million revenue and $4 million receivables, the comparable screen is `$4m / $10m × 90 = 36 days`.

The screen rose by 9 days. That could reflect real growth, a new customer mix, or a deliberate extension of terms. It does not prove manipulation. The investigator should compare invoice terms, cash collected after quarter-end, customer concentration, and the aging buckets. A ratio is a smoke alarm, not a fire report.

## Sunbeam: a named, dated case study

Sunbeam Corporation was a U.S. maker of household appliances and outdoor products. The [SEC’s administrative proceeding](https://www.sec.gov/enforcement-litigation/administrative-proceedings/33-7976), published in 2001, described conduct from late 1996 through June 1998 and said an internal investigation began in June 1998 after financial-press reports about sales practices. The agency’s account is the source for the figures in this section; allegations and enforcement findings should not be confused with a general rule that every bill-and-hold sale is improper.

### The setup and the 1997 pressure

According to the SEC, Sunbeam management used several techniques to create the appearance of a successful restructuring. The proceeding reported that the company created $35 million in improper restructuring and other reserves at the end of 1996 and reversed them into income in 1997. That reserve issue is separate from revenue recognition, but it explains the pressure context: management was trying to present improving earnings while the underlying business was deteriorating.

The SEC reported that Sunbeam booked $1.5 million of revenue and $400,000 of income from a purported barbecue-grill sale at the end of March 1997. The wholesaler held the merchandise without accepting the risks of ownership, could return all of it, and Sunbeam paid shipping and storage. The grills were returned in the third quarter, according to the proceeding.

#### Worked example: reconstructing the reported Q1 grill entry

The following is a reconstruction from the SEC’s reported $1.5 million revenue and $400,000 income figures, not Sunbeam’s full journal. If the purported sale was booked, a simplified entry would have looked like:

```ledger
Dr Accounts receivable / cash     $1,500,000
    Cr Revenue                              $1,500,000

Dr Cost of goods sold             $1,100,000
    Cr Inventory                              $1,100,000
```

The implied gross profit is `$1,500,000 − $1,100,000 = $400,000`, matching the SEC’s reported income amount for that transaction. The precise cost accounts and cash/receivable split are not supplied by the enforcement page, so this is clearly labeled as a simplified reconstruction.

The forensic correction is not “reverse $1.5 million because the goods were physically held.” It is “test whether the wholesaler accepted the risks, had a real obligation, and requested the arrangement.” The intuition is that an entry can be arithmetically balanced and economically empty.

### The bill-and-hold program

The SEC proceeding said Sunbeam began using improper bill-and-hold sales in the second quarter of 1997. Customers were offered financial incentives to place purchase orders before they needed the goods; Sunbeam held the product, paid storage, shipment, and insurance, and customers often retained return rights. The SEC reported $14 million of second-quarter sales revenue and more than $6 million of income from bill-and-hold sales.

The same proceeding reported that fourth-quarter bill-and-hold sales contributed $29 million in sales and $4.5 million in income. It also said Sunbeam’s 1997 filing disclosed bill-and-hold sales of approximately 3% of consolidated revenues, while the SEC alleged that these sales had contributed approximately 10% of fourth-quarter sales revenue and had pulled sales from 1998 into 1997.

Those percentages answer different questions. Three percent was presented as a share of full-year consolidated revenue; approximately 10% referred to the fourth quarter. A forensic reader must keep the denominator and period attached to every percentage.

#### Worked example: the $29 million fourth-quarter shift

Take the SEC-reported fourth-quarter bill-and-hold sales of $29 million and income of $4.5 million. The implied income margin is `$4.5m / $29m ≈ 15.5%`. That is a derived ratio from the SEC’s two reported amounts, not an independent company-wide margin.

If those sales would otherwise have occurred in 1998, the 1997 statement received $29 million of revenue and $4.5 million of income early. The 1998 statement then faced the reverse effect, before considering normal growth, returns, or other changes. A simplified entry showing the timing would be:

```ledger
Dr Accounts receivable / cash     $29,000,000
    Cr Revenue                             $29,000,000

Dr Cost of goods sold             $24,500,000
    Cr Inventory                             $24,500,000
```

The $24.5 million cost is a reconstruction calculated as `$29m − $4.5m`; it is not a reported Sunbeam cost line. The intuition is that a large late-period entry can be profitable and still be a timing loan that makes the next year weaker.

### The broader channel-stuffing allegations

The SEC said Sunbeam’s December 1997 distributor program offered discounts, favorable payment terms, guaranteed mark-ups, and rights to return or exchange unsold product. The proceeding said at least $62 million of Sunbeam’s reported $189 million of 1997 income did not comply with GAAP requirements, and that the company’s 1997 reported income was materially misstated. Those are enforcement allegations and findings described by the SEC proceeding; they are not a license to infer fraud from one discount or one return.

In November 1998, the SEC account said Sunbeam issued substantially restated financial statements for the six quarters from the fourth quarter of 1996 through the first quarter of 1998. For 1997, the proceeding reported that Sunbeam’s restated income was $93 million, approximately one half of the previously reported amount of $189 million. The stock price, according to the same SEC page, declined from approximately $52 in early March 1998 to approximately $7 after the restated financial statements.

#### Worked example: the restatement bridge

The SEC-reported bridge is:

| 1997 income measure | Amount |
| --- | ---: |
| Previously reported income | $189m |
| Restated income | $93m |
| Difference | $96m |
| Restated as a share of previously reported | `$93m / $189m ≈ 49.2%` |

The $96 million difference is the arithmetic gap between the two reported figures; it should not be described as a single revenue-recognition error because the SEC described multiple practices, including reserves, guaranteed sales, bill-and-hold sales, and other accelerated sales. The intuition is that restatement size is an outcome measure, not a shortcut to assigning every dollar to one scheme.

### What the Sunbeam case teaches

First, the scheme was not one clever entry. It was a system of incentives, customer terms, inventory storage, returns, disclosure omissions, and management forecasts. Second, the ordinary ratios were not enough; the decisive evidence lived in customer agreements and the timing of later returns. Third, the future-period hole mattered. The SEC described customers holding as much as 80 weeks of inventory of specific products as Sunbeam entered 1998, an operational fact that made the earlier “sales” difficult to repeat.

Finally, the case shows why a forensic accountant must separate three questions: Was the accounting wrong? Was the disclosure misleading? Was there intent or recklessness? The first is an accounting analysis, the second a reporting analysis, and the third a legal and evidentiary conclusion. A careful report says “the SEC alleged” or “the SEC proceeding reported” when that is what the source supports.

## How to write the accounting conclusion

The final memo should separate facts, accounting analysis, and intent. This sounds procedural, but it prevents two common failures: calling a strange transaction fraud before the evidence is complete, or burying a material timing issue inside neutral language because nobody wants to challenge a successful sales executive.

### Facts first

Write the transaction as a timeline. On September 28, the customer issued a purchase order. On September 29, the sales manager approved a 15% rebate by email. On September 30, the warehouse marked the units shipped. On October 12, the customer requested a return. This sequence is more useful than “revenue was aggressive.” It allows another investigator to reproduce the conclusion.

For each fact, identify the source and its limitations. A signed contract may not capture a sales representative’s promise. A customer confirmation may describe its understanding but not the seller’s internal approval. A later credit memo proves that consideration changed, but it does not by itself prove what management knew on the reporting date.

### Accounting conclusion second

State which recognition gate failed or remains uncertain. Examples:

| Evidence | Narrow conclusion |
| --- | --- |
| Customer could return all goods and seller paid storage | Consideration and transfer of control require reassessment |
| Product was installed but acceptance test remained open | Performance may be incomplete at period-end |
| Change order was unapproved and cost estimate excluded it | Completion percentage may be overstated |
| Cash arrived but contract called it a refundable deposit | Cash does not settle the performance question |

Avoid saying “the sale was fake” when the evidence supports only “the sale was not unconditional at September 30.” Precision protects both the report and the people who rely on it.

### Intent last

Intent is often inferred from patterns: repeated quarter-end entries, management warnings, concealed side letters, unusual overrides, and instructions to destroy or omit records. But an accounting error, a control failure, and intentional misstatement are different findings. A forensic report should use “alleged,” “reported,” or “the regulator found” with a dated source when discussing contested conduct.

### The evidence hierarchy

Not all evidence answers the same question. A customer confirmation may be strongest for whether goods were accepted. The warehouse system may be strongest for where goods sat. The sales compensation plan may be strongest for incentive. The general ledger is strongest for what was booked, but weak for whether the entry was economically correct.

| Question | High-value evidence | Common trap |
| --- | --- | --- |
| Did the customer commit? | Contract, purchase order, legal enforceability | Treating an internal forecast as an order |
| Did control transfer? | Acceptance, delivery, segregation, risk of loss | Equating a shipping label with acceptance |
| Was price final? | Side-letter search, credit approvals, rebate ledger | Reading only the master contract |
| Was the work complete? | Site records, labor, milestone certificate | Relying on management’s percentage field |
| Did the amount persist? | Subsequent cash, returns, credits, usage | Stopping at the invoice date |

The aim is triangulation. Three independent systems that tell the same story are more persuasive than three copies of the same spreadsheet.

#### Worked example: turning one suspicious invoice into a test plan

Suppose invoice 8472 is an illustrative $250,000 sale dated September 30. The general ledger says debit accounts receivable and credit revenue. The warehouse log says 500 units remained in a seller-controlled location. A customer email dated October 3 says, “We will take the units only if our retail campaign works.” A credit memo dated October 20 reverses $75,000.

The test plan is:

1. Confirm whether the October 3 email reflects a pre-existing term or a new negotiation.
2. Identify the 500 units and test whether they were complete, segregated, and unavailable for substitution on September 30.
3. Inspect the contract’s return and cancellation clauses.
4. Reconcile the $75,000 credit to the returned quantity, price, and any rebate.
5. Search other invoices for the same customer, salesperson, product, and period-end pattern.

The $75,000 credit is 30% of the invoice: `$75,000 / $250,000 = 30%`. That arithmetic helps prioritize the customer, but it does not tell us whether the correct September revenue was $175,000, zero, or $250,000. The contract and transfer evidence answer that question. The intuition is that forensic accounting converts a red flag into a reproducible population test.

### Controls that reduce the opportunity

Good controls do not assume every salesperson is dishonest. They make unusual arrangements visible before the close. Useful controls include independent approval for quarter-end discounts, a legal review of return and repurchase terms, a shipping cut-off report reviewed by operations, customer confirmation of bill-and-hold requests, and a post-period credit-note report sent to the audit committee.

For long-term contracts, require a documented estimate-to-complete review. The reviewer should challenge the cost-to-cost denominator, compare forecast labor with time sheets, test supplier commitments, and require approval for change orders. The control is valuable because it moves the challenge upstream, before a weak estimate becomes reported profit.

No control is perfect. A manager can override a workflow, a customer can give a misleading confirmation, and a system can classify a return incorrectly. That is why the strongest design combines preventative controls with independent subsequent-event analytics.

### The investor’s five-minute version

When time is short, begin with five questions:

1. Did revenue accelerate at the exact time management needed it to?
2. Did receivables, contract assets, or inventory grow faster than revenue?
3. Did the company offer unusual terms, discounts, or return rights?
4. Did customers actually use, resell, accept, and pay for the product?
5. Did the next quarter reveal returns, credits, weak sell-through, or a margin reversal?

These questions do not replace a filing review or an audit. They do prevent a reader from treating top-line growth as self-authenticating. A credible growth story has an operational counterpart: factories produce, customers accept, cash collects, and the next quarter does not collapse merely because the previous quarter was made to look good.

### A note on materiality

*Materiality* means that an omission or misstatement could reasonably influence the decisions of a financial-statement user. It is not a universal percentage, and it is not only about the effect on net income. A small revenue error can matter if it turns a loss into a profit, lets management meet a covenant, supports a debt offering, or changes a trend that investors were watching.

For example, an illustrative $2 million timing error may be immaterial to a company with $2 billion of stable revenue, but material if it is the difference between $1 million of reported profit and a $1 million loss. Qualitative context matters too: a transaction involving the chief executive, a related party, or a concealed side letter may deserve attention even when the amount is not large in isolation.

Forensic teams should report both the amount and the decision context. Quantify the revenue, gross-profit, receivable, inventory, and cash effects separately. Then explain whether the entry affected a target, a covenant, a forecast, or a recurring trend. Do not hide behind a net-income percentage that makes a strategically important misstatement look small.

### A note on estimates and hindsight

The existence of a later reversal does not automatically prove that the original estimate was unreasonable. Businesses face uncertainty, and a customer can unexpectedly fail after a valid sale. The proper question is what was knowable at the reporting date. Did prior return history, customer correspondence, or a known defect make the concession foreseeable? Did management update its estimate when new evidence arrived, or did it suppress that evidence until after the close?

This is why contemporaneous documents matter. A forecast prepared before the quarter closed can show what management expected; a forecast revised after the audit challenge may show only the final position. Preserve both versions. Compare the assumptions, the author, the approval trail, and the operational data available at each date.

The same discipline applies to percentage-of-completion. A project that eventually loses money was not necessarily misreported at every earlier date. The investigation must reconstruct the information set at each reporting date and determine whether the estimate was supportable then. Hindsight is useful for testing bias, not for replacing the original accounting analysis.

### The board and audit committee lens

An audit committee does not need to become a warehouse operator, but it should understand the company’s revenue engine. Ask management to explain the largest contract types in plain language, the percentage of revenue subject to return or acceptance, the amount of bill-and-hold inventory, and the size and aging of contract assets. Ask for a bridge from reported revenue to cash collected and to post-period credits.

The committee should also ask what would have happened without the unusual quarter-end program. If the answer is “the company would have missed guidance,” that is an incentive fact worth documenting. It does not prove wrongdoing; it explains why controls and challenge are needed precisely when the number feels most important.

The best governance question is often simple: “What will next quarter look like if we exclude the sales that were accelerated?” A credible answer has customer-level support. A vague answer is a reason to widen the sample.

One final practical point: document the population boundary. Say whether the review covers all invoices, the largest transactions, manual entries, or a statistical sample. Record excluded credit notes and explain why a transaction was included or excluded. A conclusion about a sample should remain a conclusion about a sample until the team projects it using an appropriate method. This prevents a persuasive anecdote from being mistaken for a quantified company-wide exposure.

When communicating to non-accountants, translate the finding twice. First describe the operational event—goods stayed in the seller’s warehouse, the customer could return them, or the project cost estimate omitted an approved change. Then describe the statement effect—revenue, receivable, inventory, contract asset, or profit was recorded in the wrong period. That translation is what allows a board member, lender, or investor to understand why the issue matters without memorizing accounting jargon.

## Common misconceptions

### “Bill-and-hold is always fraud”

No. A customer may genuinely request delayed delivery for a substantive reason, and the goods may be complete, identified, unavailable for substitution, and subject to a firm customer commitment. The accounting depends on the full facts and applicable standard.

### “If the customer paid, revenue must be real”

No. A customer can pay a deposit before the seller performs. Cash reduces collection risk, but it does not automatically establish that the promised good or service was transferred in the current period.

### “A revenue spike proves channel stuffing”

No. A product launch, a seasonal event, or a large legitimate contract can create a spike. The signal becomes stronger when the spike coincides with unusual terms, concentrated quarter-end shipments, weak sell-through, returns, or later credits.

### “Receivables rising faster than revenue proves fraud”

No. New customers, longer industry terms, or a change in product mix can explain the movement. Use the ratio to select transactions, then test contracts, aging, subsequent cash, and disputes.

### “Percentage-of-completion is just management’s guess”

It is an estimate constrained by project evidence, cost records, contract terms, and the applicable framework. It becomes vulnerable when the denominator is unsupported, change orders are not approved, losses are delayed, or progress claims do not match customer acceptance.

### “The auditor’s clean opinion means timing risk is gone”

An audit opinion is not a guarantee that every concealed side agreement was discovered. The auditor evaluates evidence under an audit standard and materiality threshold. Investors and boards should still ask how revenue is earned and where the judgment sits.

## How it shows up in real markets

Revenue recognition risk matters to lenders, equity investors, acquisition teams, auditors, internal investigators, and anyone who relies on a company’s growth narrative. It is especially relevant when a company is near a covenant threshold, reporting an improbable acceleration, or using a distribution model with generous returns.

### A distributor-led consumer business

The first practical scenario is a consumer-goods company that reports 25% sales growth while its distributors’ sell-through is flat. The investigation should not begin by accusing the sales team. It should map the terms: rebates, return windows, price protection, freight, payment dates, and who carries inventory insurance. A quarter-end shipment chart by day and customer often reveals whether the growth was broad-based or concentrated in a handful of accommodating distributors.

### A software company with acceptance clauses

The second scenario is enterprise software. A contract may contain a license, implementation, training, and support. A vendor that books the full contract on delivery of a license while implementation is incomplete may have a performance-obligation problem. The tell is a gap between invoice and acceptance: unresolved defects, unsigned acceptance certificates, or customers paying only the deposit. Review the contract’s deliverables and allocate consideration before looking at management’s revenue forecast.

### A construction company with optimistic change orders

The third scenario is a builder whose margin rises as projects become more complex. The key evidence is not a single percentage-complete figure. It is the project cost ledger, subcontractor commitments, approved change orders, claims correspondence, unpriced work, and expected-loss review. Recalculate the completion percentage using independent evidence. If a $1 million project’s cost-to-date is $300,000, changing expected total cost from $600,000 to $800,000 changes the illustrative completion percentage from 50% to 37.5%; that is an accounting sensitivity worth investigating.

### A distressed issuer approaching a refinancing

The fourth scenario is a company approaching a debt maturity. The temptation to show strong EBITDA, a common profit measure before interest, taxes, depreciation, and amortization, can make revenue timing especially consequential. A lender should read the covenant definition, test add-backs, inspect post-period credit notes, and compare cash conversion. The goal is not to punish judgment; it is to understand whether reported earnings will survive the next reporting period.

## A compact red-flag dashboard

Use this as a starting checklist, not a scoring model. No point total can replace transaction evidence.

| Area | Questions |
| --- | --- |
| Cut-off | Did shipments cluster in the final days? Were delivery dates changed after close? |
| Terms | Did payment, discount, return, or acceptance terms change near the target? |
| Control | Who requested storage? Can the seller substitute or redirect the goods? |
| Demand | Did distributor sell-through, end-customer orders, or usage rise too? |
| Statements | Did receivables, contract assets, or inventory move unusually relative to revenue? |
| Subsequent events | Were there returns, credits, cancellations, or write-offs soon after close? |
| Incentives | Was management paid on revenue, EBITDA, EPS, or a covenant ratio? |
| Disclosure | Does the filing explain unusual programs and their likely next-period effect? |

The final discipline is to write the conclusion at the same level as the evidence. “The selected transactions had return rights inconsistent with unconditional sale” is stronger than “management committed fraud” when intent has not been established. Conversely, if an enforcement agency has made an allegation or finding, name the agency, date, and source rather than laundering it into anonymous prose.

## When this matters to you / further reading

If you read financial statements, the practical habit is simple: when revenue accelerates, ask what had to happen operationally for the customer to accept the product and pay for it. Then inspect the other statements. Did cash follow? Did receivables age? Did inventory remain under the seller’s control? Did next-quarter sales and margins behave like the company’s story predicted?

This is educational analysis, not individualized investment or accounting advice. For a broader toolkit, pair this post with the site’s [valuation by sector](/blog/trading/vietnam-stocks/valuation-by-sector-pe-pb-nav-ev-ebitda) discussion and its [financial-chain framework](/blog/trading/vietnam-stocks/the-financial-chain-banks-property-brokers-steel-construction): a revenue problem can become a credit problem when lenders, suppliers, and shareholders all rely on the same overstated number.

## Sources & further reading

- [SEC, In the Matter of Sunbeam Corporation, Administrative Proceeding No. 3-10287](https://www.sec.gov/enforcement-litigation/administrative-proceedings/33-7976), published 2001. Primary source for the dated Sunbeam allegations, reported bill-and-hold amounts, restatement figures, and described sequence of events.
- [SEC, Phillip E. Harlow, CPA](https://www.sec.gov/enforcement-litigation/administrative-proceedings/34-47261), published 2003. SEC proceeding describing the audit-partner matter and bill-and-hold recognition criteria discussed in the case.
- [SEC, Revenue Recognition speech discussing SAB 101](https://www.sec.gov/news/speech/spch495.htm), dated 2000. Background on revenue-recognition risk, including arrangements that can resemble financing or consignment.
- [SEC, Report pursuant to Section 704 of the Sarbanes-Oxley Act of 2002](https://www.sec.gov/news/studies/sox704report.pdf). Enforcement-study context on premature revenue recognition and bill-and-hold arrangements.
- The relevant revenue-recognition standard and the reporting framework used by the company under review. Always read the current authoritative text for the jurisdiction and contract at issue; this article explains forensic reasoning, not a substitute for professional accounting advice.
