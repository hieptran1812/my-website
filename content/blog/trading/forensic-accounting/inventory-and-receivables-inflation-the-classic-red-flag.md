---
title: "Inventory and receivables inflation: The classic red flag"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "A beginner-friendly forensic guide to overstated inventory, uncollectible receivables, write-down avoidance, gross-margin tells, DSO, and the American Tissue case."
tags: ["forensic-accounting", "inventory", "receivables", "dso", "gross-margin", "working-capital", "financial-statement-fraud", "red-flags"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 30
---

> [!important]
> **TL;DR** — Inventory and receivables are not suspicious because they are large; they become suspicious when the balance-sheet story stops matching shelves, customers, contracts, margins, and cash.
>
> - Inventory can be overstated by counting goods that are missing, retaining obsolete goods at old cost, or postponing a needed write-down.
> - Receivables can be overstated by recording sales that are not genuine, failing to remove returns and credits, or keeping uncollectible invoices on the books.
> - Days sales outstanding (DSO) is a useful collection-time estimate, but a falling DSO is not proof of healthy customers and a rising DSO is not proof of fraud.
> - Gross margin and DSO read together are more revealing: smooth margins plus weakening aging, returns, or post-period cash can indicate that losses are being delayed.
> - In the SEC’s March 10, 2003 case against American Tissue, the complaint alleged approximately \$21.8 million of bogus bill-and-hold sales and related receivables in the quarter ended June 30, 2001.

Imagine a grocer who says the warehouse contains 1,000 cases of milk, while customers supposedly owe the store for 1,000 cases already delivered. You visit the warehouse and find 600 cases. Then you call the customers and discover that several invoices are disputed, two orders were never placed, and one customer returned the product after the reporting date.

That is the forensic-accounting problem in miniature. The balance sheet says “asset.” The investigation asks whether the asset exists, is measured at a defensible amount, and will produce the cash or benefit the company claims. Inventory and accounts receivable are especially useful because both are close to physical or contractual reality. Goods can be counted. Customers can confirm orders. Cash receipts can be traced.

The first figure shows the clean trail. Cash buys inventory; inventory becomes a cost of goods sold when a product is sold; the sale creates a receivable when the customer has not yet paid; collection turns the receivable back into cash. Inflation breaks one link while leaving the headline statements temporarily attractive.

![The clean inventory-to-cash trail shows where an overstated asset stops becoming cash.](/imgs/blogs/inventory-and-receivables-inflation-the-classic-red-flag-1.webp)

This article is an analytical guide, not a conclusion about any company you own. A red flag is a reason to obtain better evidence. It is not, by itself, evidence that management committed fraud.

## Foundations: the building blocks

### What an asset means here

An **asset** is a resource controlled by a company because of a past event and expected to provide future economic benefit. That definition sounds abstract, so use two ordinary objects. A box of finished goods is an asset because the company can sell it. A customer invoice is an asset because the company expects the customer to pay it.

**Inventory** is the cost assigned to goods a company has bought or made but has not yet sold. It may include raw materials, **work in process** (partly completed goods), and finished goods. Inventory is not recorded at the company’s hoped-for selling price. It begins with an accounting cost, then the company considers whether that cost is recoverable.

**Accounts receivable** are amounts customers owe for goods or services already provided on credit. “On credit” means the company delivered now and agreed to collect later. A receivable is not cash. It is a promise to pay, and the promise can be late, disputed, returned, or worthless.

An **allowance for credit losses** is a contra-asset: an account that reduces gross receivables to the amount management expects to collect. A **write-off** removes a specific receivable when collection is no longer expected. The allowance is an estimate across a portfolio; a write-off is a decision about a particular balance. Confusing the two can hide whether estimates are being updated promptly.

**Revenue** is income recognized from providing goods or services to a customer. **Cost of goods sold (COGS)** is the cost assigned to the goods that produced that revenue. **Gross profit** is revenue minus COGS. **Gross margin** is gross profit divided by revenue:

\[
\text{Gross margin} = \frac{\text{Revenue} - \text{COGS}}{\text{Revenue}}.
\]

The margin is a percentage, not a cash balance. A company can report a high gross margin while its customers pay slowly and its warehouse contains unsellable goods.

### The accounting engine

Double-entry accounting records each transaction with at least one debit and one credit. A **debit** and a **credit** are directions in the ledger, not synonyms for good and bad. For an asset, a debit normally increases the balance and a credit normally decreases it. For revenue, a credit normally increases the balance.

Suppose a company buys goods for an illustrative \$600,000 in cash. The simplified entry is:

| Debit | Credit |
| --- | --- |
| Inventory \$600,000 | Cash \$600,000 |

When it sells those goods for an illustrative \$1,000,000 on credit, and the cost of the goods is \$600,000, two entries are needed:

| Debit | Credit |
| --- | --- |
| Accounts receivable \$1,000,000 | Revenue \$1,000,000 |
| COGS \$600,000 | Inventory \$600,000 |

The first entry records the customer’s promise and the sale. The second moves the goods’ cost out of inventory and into the income statement. The difference, \$400,000, is gross profit before other expenses and taxes.

![A valid credit sale removes inventory at cost, creates a receivable, and reports revenue; every arrow needs real shipment and customer evidence.](/imgs/blogs/inventory-and-receivables-inflation-the-classic-red-flag-2.webp)

The same journal-entry shape can be used to record a fake sale. That is why the shape is not evidence. The evidence is the purchase order, shipping record, customer acceptance, return terms, invoice, subsequent collection, and inventory movement.

#### Worked example: a clean statement-line bridge

This is an illustrative walkthrough, not a company’s reported transaction. A manufacturer buys 100 units at \$6,000 each, so inventory is \$600,000. It sells all 100 units for \$10,000 each on 30-day credit terms.

1. Revenue is (100 \times {$}10{,}000 = {$}1{,}000{,}000).
2. COGS is (100 \times {$}6{,}000 = {$}600{,}000).
3. Gross profit is ({$}1{,}000{,}000 - {$}600{,}000 = {$}400{,}000).
4. Gross margin is ({$}400{,}000 / {$}1{,}000{,}000 = 40\%).
5. Before collection, receivables rise by \$1,000,000 and cash does not rise.
6. When the customer pays, receivables fall by \$1,000,000 and cash rises by \$1,000,000.

The intuition: a real sale creates a receivable that should have a visible path to cash; a fake or uncollectible sale creates the first half of the path and struggles with the second.

### DSO: turning a receivable into time

**Days sales outstanding (DSO)** estimates how many days of credit sales are sitting in receivables. In plain English, it asks: if the current receivable balance behaved like an average day of sales, how long would collection take?

\[
\mathrm{DSO} = \frac{\text{average trade receivables}}{\text{net credit sales}} \times \text{days in period}.
\]

The strongest numerator is average trade receivables, usually the beginning and ending trade-receivable balances averaged together. A period-end balance is a rougher snapshot. The denominator should be credit sales, not automatically total revenue, because cash sales do not create receivables. Public filings may not provide the clean numerator and denominator, so analysts must state their approximation.

DSO can rise because customers are slower, terms became longer, sales were pulled into the last days of a period, or a large invoice is disputed. It can fall because collections improved, because a company sold receivables to a finance provider, or because management happened to collect a batch just before the reporting date.

### Inventory days and the gross-margin connection

**Days inventory outstanding (DIO)** estimates how many days of COGS are represented by inventory:

\[
\mathrm{DIO} = \frac{\text{average inventory}}{\text{COGS}} \times \text{days in period}.
\]

Inventory is measured at cost, so COGS is the natural flow denominator. If a company reports inventory of \$3,000,000, annual COGS of \$18,000,000, and a 365-day year, the simple period-end DIO is (3{,}000{,}000 / 18{,}000{,}000 \times 365 = 60.83) days, or about 61 days. That is a turnover estimate, not a physical count.

When inventory is carried above the amount it can generate through sale or use, the company should recognize a **write-down**. A write-down is an expense that reduces the asset to a more supportable amount. Depending on the applicable accounting framework and inventory category, the test may involve net realizable value, market, or another recoverability concept. The basic forensic question is stable across frameworks: can the recorded cost be recovered?

## 1. The core red flag: the balance sheet outruns the business

The most useful first question is not “is inventory high?” It is “does inventory grow faster than the business has a plausible reason to require?” Receivables deserve the same test. A growing company can legitimately need more working capital, the short-term money tied up in ordinary operations. But the growth should connect to sales, production, customer terms, and cash.

There are four different ways a current asset can be wrong:

1. **Existence:** the quantity or receivable is not real.
2. **Rights:** the company does not control the goods, or the customer has a return or cancellation right that changes the economics.
3. **Valuation:** the asset exists, but its recorded amount is too high.
4. **Completeness and cutoff:** returns, credits, obsolete stock, or payments were not recorded in the correct period.

These routes matter because each requires different evidence. A warehouse count tests quantity. A customer confirmation tests existence and terms. A subsequent cash receipt tests collection. A markdown report tests recoverability. A single ratio cannot replace those procedures.

![Inventory and receivables inflation can arise from missing quantity, excessive value, a nonexistent sale, or weak collectibility; each branch needs a different test.](/imgs/blogs/inventory-and-receivables-inflation-the-classic-red-flag-4.webp)

### Inventory quantity inflation

The simplest manipulation is to report goods that are not there. The risk is highest when counts are manual, warehouses are dispersed, goods are held by third parties, or the accounting record is updated without a reliable movement log. A count difference can also be an honest control failure, not fraud. The forensic distinction comes from who knew, when they knew, and how the difference was handled.

Useful tests include independent test counts, serial-number or lot-number tracing, receiving reports, shipping records, location reports, and inspection of damaged or quarantined goods. **Cutoff** means recording a transaction in the period when it economically belongs. Test the last receipts before period-end and first receipts after it. A company that includes goods received after year-end has overstated both inventory and, potentially, profit.

### Inventory valuation inflation

Goods can be physically present and still be worth less than recorded cost. Fashion, electronics, food, seasonal products, spare parts, and discontinued components can lose value before management records the loss. If a \$100 item can now be sold for only \$70 after \$5 of selling costs, its recoverable amount is not the original \$100. The exact accounting test depends on the applicable standard, but retaining the old cost without support is the warning.

Look for aging by SKU, inventory turns, markdown history, return rates, gross-margin by product, purchase commitments, post-period sales, and goods that are being moved between warehouses rather than sold. A “reserve” is an estimate set aside against an expected loss. A low reserve can make current profit look better, but it creates pressure for a later catch-up charge.

### Receivable existence inflation

An invoice is not proof that a customer ordered, received, accepted, and owes the product. The red flags are unusual quarter-end shipments, invoices with no shipping document, duplicate invoice numbers, round-dollar entries, customers with no independent contact information, and sales that are reversed soon after the reporting date.

**Bill and hold** is a transaction in which the seller invoices goods while holding them rather than shipping them immediately. It can be legitimate only when the relevant revenue-recognition conditions are satisfied, including a substantive customer request and a reason for the delayed delivery under the applicable rules. If the customer did not order the goods, or the seller can freely use them for another customer, the invoice may not represent a completed sale.

### Receivable collectibility inflation

The more common version is less dramatic: the sale occurred, but the customer probably will not pay in full. **Aging** sorts receivables by how long they have been outstanding, such as current, 31–60 days past due, or older buckets. Older balances usually need more scrutiny, though an aging schedule is not a law of nature.

Test the allowance against actual collections after the reporting date. Review credit memos, returns, legal disputes, customer concentration, covenant waivers, and changes in payment terms. If a company reports a stable allowance while the old bucket grows, the estimate may be lagging the economic loss.

#### Worked example: the allowance that is too small

This is illustrative. A company reports gross receivables of \$10,000,000. It expects \$250,000 of current invoices to go unpaid and \$750,000 of older invoices to go unpaid. Its supportable allowance is therefore \$1,000,000.

Management has recorded an allowance of only \$400,000. The reported net receivable is:

\[
{$}10{,}000{,}000 - {$}400{,}000 = {$}9{,}600{,}000.
\]

The evidence-supported net receivable is:

\[
{$}10{,}000{,}000 - {$}1{,}000{,}000 = {$}9{,}000{,}000.
\]

The missing allowance is \$600,000. Before tax effects, correcting it reduces receivables and pretax income by \$600,000. The adjustment is not “new cash leaving the bank”; it is recognition that some previously reported profit will not turn into cash.

The intuition: an allowance is not a secret cash reserve; it is the accounting admission that a receivable is worth less than its invoice value.

## 2. DSO: useful signal, dangerous verdict

DSO is popular because it compresses a messy receivables ledger into a familiar unit: days. If DSO rises from 45 days to 70 days while credit sales are flat, more cash is likely tied up. But the ratio is a summary of a balance, not a movie of every invoice.

### The denominator can hide the timing

Suppose a company has quarterly credit sales of \$30,000,000 and period-end receivables of \$6,000,000. Using 90 days, the simple DSO is 18 days. If \$4,000,000 of the sales happened in the final five days, the denominator is heavily weighted toward sales that have had little time to be collected. The low ratio can look healthy even though the new invoices are untested.

The solution is to examine weekly or monthly sales and collections, not just the period-end ratio. Ask what DSO would be using average receivables. Reconcile the reported number to the filing’s definition. Separate trade receivables from tax receivables, employee advances, contract assets, and other balances.

### A falling DSO can be engineered

Receivables can fall without customers paying through factoring, securitization, offsets, reclassification, or a short-term collection campaign. **Factoring** means selling or financing receivables with a finance provider. The company may receive cash, but it may retain recourse if customers default. That is financing risk, not necessarily improved customer quality.

#### Worked example: DSO falls because receivables are sold

This is explicitly illustrative. Before quarter-end, credit sales are \$30,000,000 and trade receivables are \$6,000,000. The simple DSO is:

\[
\frac{{$}6{,}000{,}000}{{$}30{,}000{,}000} \times 90 = 18\text{ days}.
\]

The company sells \$1,500,000 of receivables to a finance company for \$1,470,000. The reported trade-receivable balance falls to \$4,500,000, and the same rough DSO becomes:

\[
\frac{{$}4{,}500{,}000}{{$}30{,}000{,}000} \times 90 = 13.5\text{ days}.
\]

DSO improves by 4.5 days. But the company paid an illustrative \$30,000 discount and may retain default exposure through recourse. Customers did not necessarily pay faster; the company converted an invoice into financing proceeds.

The intuition: trace the cash source before celebrating a lower DSO. Customer cash, not any cash, is the relevant evidence of collection quality.

### DSO can rise for innocent reasons

Longer terms may be a competitive choice. A large customer may negotiate 60-day terms rather than 30-day terms. A seasonal business may invoice heavily just before its strongest collection month. An acquisition may bring a different customer mix. Foreign exchange can change the reported balance in a company’s presentation currency.

The forensic response is not to accuse. It is to decompose the movement: volume, price, terms, customer mix, currency, collections, credit notes, and financing. Then test whether management’s explanation matches the ledger.

## 3. Gross margin: the write-down-avoidance tell

Gross margin looks simple: sales minus COGS, divided by sales. But inventory valuation controls the timing of COGS. If obsolete inventory is not written down, the income statement may avoid an expense today. If that inventory is later sold at a deep discount, the economic loss appears later through a lower margin.

This creates a useful but subtle signal. A company with rising inventory days and unusually stable gross margin may be deferring markdowns or write-downs. It may also have a valid reason: a new product launch, deliberate capacity build, or a temporary supply disruption. The signal strengthens when inventory growth, lower sell-through, aggressive reserves, and later discounts appear together.

![Gross margin and DSO form a dashboard: each ratio needs aging, return, margin, and post-period cash evidence before escalation.](/imgs/blogs/inventory-and-receivables-inflation-the-classic-red-flag-3.webp)

### Margin arithmetic

#### Worked example: an avoided inventory write-down

This is an illustrative statement-line bridge. A company carries inventory at \$1,000,000. Based on expected selling prices and costs to complete and sell, the amount it can recover is \$800,000. The needed write-down is \$200,000.

Before correction:

| Line | Reported amount |
| --- | ---: |
| Inventory | \$1,000,000 |
| COGS or inventory-loss expense | \$0 additional |
| Pretax income | \$200,000 too high relative to the corrected case |

The correcting entry is:

| Debit | Credit |
| --- | --- |
| Inventory write-down expense \$200,000 | Inventory \$200,000 |

After correction, inventory is \$800,000 and pretax income is \$200,000 lower, before tax effects. If the company sells the goods later for \$800,000, the cash proceeds do not create the missing profit. The earlier write-down simply puts the loss in the period when the evidence showed it.

The intuition: refusing to write down bad inventory does not preserve value; it preserves an overstated asset and postpones the loss.

![A ${\$}200,000 inventory write-down reduces the recorded asset and pretax income before tax effects.](/imgs/blogs/inventory-and-receivables-inflation-the-classic-red-flag-6.webp)

### Why margin and DSO should be read together

The table is a compact starting dashboard, not a scoring system.

| DSO | Gross margin | First question |
| --- | --- | --- |
| Down | Stable | Did customers actually pay, or were receivables sold? |
| Up | Stable | Are old invoices and allowances worsening? |
| Up | Down | Are weak customers, returns, and discounts converging? |
| Flat | Down | Is COGS catching up with inventory valuation or mix? |

The most concerning combination is not a particular direction. It is a divergence between accounting smoothness and operating evidence: margin stays smooth, DSO looks manageable, but aging worsens, customer credits increase, inventory sits longer, and cash from operations trails reported earnings.

## 4. When journal entries tell the story

A journal-entry review is not a hunt for unusual debits in isolation. It is a search for entries that change the timing or location of economic loss. **Manual journal entries** are entries posted outside the normal subledger process. A manual entry can be entirely legitimate, but entries made late in a reporting period, posted by senior users, rounded to large amounts, or reversing immediately afterward deserve context.

### The inventory route

An ordinary sale moves inventory to COGS. To inflate profit without inventing revenue, someone may debit a balance-sheet asset and credit an expense, thereby postponing the cost. For example, an illustrative \$300,000 production cost that should be expensed might be left in inventory:

| Debit | Credit |
| --- | --- |
| Inventory \$300,000 | COGS \$300,000 |

This entry raises inventory and pretax income by \$300,000 relative to the correct treatment. The question is whether the goods still exist and are expected to produce benefit. If they do not, the asset is a parking place for an expense.

### The receivables route

To create a receivable without a real sale, an illustrative entry might be:

| Debit | Credit |
| --- | --- |
| Accounts receivable \$500,000 | Revenue \$500,000 |

The balance sheet and income statement both improve on paper. No cash arrives. A subsequent reversal might be hidden in a later period, or the balance might be kept alive by moving it between customers. Test the customer master file, invoice sequence, shipping record, acceptance, and post-period cash.

### The reserve route

If a company should increase its allowance for doubtful accounts by \$150,000 but does not, the missing entry is:

| Debit | Credit |
| --- | --- |
| Bad-debt expense \$150,000 | Allowance for credit losses \$150,000 |

The absence of the entry leaves assets and pretax income too high. This is not necessarily a fabricated invoice. It can be an optimistic estimate, a delayed update, or deliberate avoidance of a known loss.

#### Worked example: three entries, one inflated quarter

This combined illustration uses three separate issues:

1. Inventory cost of \$300,000 is kept in inventory rather than COGS.
2. A \$500,000 sale is recorded without a genuine customer order.
3. The allowance is understated by \$150,000.

The pretax overstatement is:

\[
{$}300{,}000 + {$}500{,}000 + {$}150{,}000 = {$}950{,}000.
\]

The balance-sheet effect is not one big “fraud account.” It is \$300,000 too much inventory, \$500,000 too much gross receivables, and \$150,000 too little contra-asset allowance. A reviewer who tests only revenue can miss the inventory route; a reviewer who tests only the warehouse can miss the uncollectible invoice.

The intuition: inflation can be distributed across several ordinary-looking accounts, but the statement equation still has to reconcile to physical goods and customer cash.

## 5. Subsequent events: let the next period testify

The days after the reporting date are often the most useful evidence available. A **subsequent event** is an event occurring after the reporting period that may provide information about conditions existing at the period-end date. A customer payment a week later can support the existence and collectibility of a year-end receivable. A credit note issued a week later can reveal that the original balance was overstated or that the sales terms were incomplete.

Do not treat every later credit memo as proof of a prior error. Returns are normal in many industries. Ask whether the return rate matches history, whether the original invoice reflected the customer’s return right, and whether the company reserved for that obligation.

![A period-end receivable should leave a trace in confirmation, collection, or a documented dispute; post-close credits and aging escalation are risk paths.](/imgs/blogs/inventory-and-receivables-inflation-the-classic-red-flag-5.webp)

### A practical post-close procedure

Start with the period-end receivable listing and select large, old, unusual, and round-dollar balances. For each one, trace:

- the customer order and agreed terms;
- the invoice and shipping or service-completion evidence;
- customer acceptance, if required;
- cash received after period-end;
- credit memos, returns, rebates, and disputes;
- whether the customer was related to management or another counterparty.

For inventory, trace the reverse direction: count or inspect the goods, match them to the ledger, review later sales, inspect markdowns and write-offs, and test whether goods are held on consignment or owned by someone else.

#### Worked example: the quarter-end invoice that fails the cash test

This is illustrative. A company records \$2,000,000 of sales on the final day of a quarter. The terms say payment is due in 60 days. The customer confirms only \$1,200,000, and the remaining \$800,000 is credited 20 days later because no purchase order existed.

The reported entry for the full amount would be:

| Debit | Credit |
| --- | --- |
| Accounts receivable \$2,000,000 | Revenue \$2,000,000 |

The later credit note reverses \$800,000 of the receivable and revenue. If the quarter-end statements did not reflect the known absence of a customer order, revenue and receivables were overstated by \$800,000 in that period. The \$1,200,000 confirmed balance still requires a collection test; confirmation alone does not guarantee payment.

The intuition: subsequent evidence does not rewrite history automatically, but it tells you whether the period-end balance had a credible foundation.

## 6. The American Tissue case: an allegation with a working-capital trail

American Tissue, Inc. was a paper manufacturer. On March 10, 2003, the SEC announced a civil action against the company and three former officers. The [SEC litigation release](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-18022) says that during 2000 and 2001, the defendants allegedly inflated revenues and earnings through, among other things, overvalued inventory, improperly capitalized expenses, and bogus bill-and-hold sales. The release says the company offered and sold \$165 million of securities during that period.

Those are allegations in an enforcement action, not a finding that every reported transaction was false. The complaint is useful because it shows the mechanism in concrete statement lines.

The [SEC complaint](https://www.sec.gov/litigation/complaints/comp18022.htm) alleges that, during the quarter ended June 30, 2001, company personnel recorded approximately \$21.8 million of bogus bill-and-hold sales and resulting accounts receivable. It alleges that the customers had not ordered the quantity, the invoiced product was not shipped, and the product remained in American Tissue’s reported third-quarter inventory. The complaint further alleges that reported net income for the first nine months of fiscal 2001, stated as \$15.5 million, was overstated by approximately \$21.8 million.

That last comparison is arresting because it is internally intelligible: the alleged unsupported sales were larger than the reported nine-month income. It does not mean the company had no real sales. It means that, if the allegations were proven as stated, the unsupported entries were large enough to reverse the reported profit picture.

The alleged financing incentive also matters. The complaint says American Tissue’s revolving credit agreement allowed borrowing up to 85% of the face amount of accounts receivable assigned to the lender. The alleged bogus receivables therefore affected not only revenue and profit but also borrowing capacity. This is why a forensic accountant follows a receivable into the lender’s collateral file, not just into the income statement.

The complaint also alleges that in fiscal 2000, supplies of approximately \$15.6 million that had previously been expensed were reclassified as a company asset called “supplies inventory.” In plain language, an expense allegedly moved to the balance sheet. The effect is the same pattern as the illustrative inventory entry above: current-period expense is delayed, assets are increased, and profit rises relative to the correct accounting.

The case’s red-flag cluster is therefore:

| Reported area | Alleged mechanism | Forensic test |
| --- | --- | --- |
| Revenue and receivables | \$21.8m bogus bill-and-hold sales | Customer orders, shipping, acceptance, collection |
| Inventory | Unshipped product remained in reported inventory | Physical count, ownership, cutoff, subsequent sale |
| Expenses and assets | \$15.6m of supplies allegedly reclassified | Vendor invoices, consumption records, capitalization policy |
| Liquidity | Receivables supported lender advances | Borrowing-base certificate and lender confirmations |

The lesson is not “American Tissue proves every high DSO is fraud.” The lesson is that a receivable, inventory balance, reported margin, and borrowing base can be different views of the same underlying claim. When those views rely on the same unsupported invoice, the risk compounds.

## A field worksheet for the curious reader

The following worksheet turns the concepts into repeatable questions. It is deliberately more operational than a ratio checklist. Each question asks for a document, a person, or a cash movement that can answer it.

### 1. Reconcile the opening and closing balances

For receivables, begin with opening gross receivables. Add credit sales. Subtract cash collections, credit notes, returns, and write-offs. Add or subtract acquisitions, disposals, and foreign-exchange effects when relevant. The result should reconcile to closing gross receivables. Do not let “other” become a permanent plug.

For inventory, begin with opening inventory. Add purchases and production costs. Subtract COGS, write-downs, disposals, and shrinkage. Adjust for acquisitions, foreign exchange, and transfers when relevant. Then compare the resulting ledger balance with the physical count. A reconciliation is not a control merely because it exists; inspect the supporting schedule and the person who approved each adjustment.

#### Worked example: a receivable roll-forward

This is illustrative. Opening gross receivables are \$4,000,000. Credit sales add \$12,000,000. Customers pay \$10,500,000. Credit notes reduce the balance by \$300,000. Write-offs reduce it by \$100,000.

The expected closing balance is:

\[
{\$}4{,}000{,}000 + {\$}12{,}000{,}000 - {\$}10{,}500{,}000 - {\$}300{,}000 - {\$}100{,}000 = {\$}5{,}100{,}000.
\]

If the ledger says \$5,600,000, the unexplained difference is \$500,000. That does not prove a false sale. It tells the investigator to find the missing collection, credit, write-off, acquisition adjustment, or journal entry. If management calls it “timing,” ask which invoice and which bank deposit demonstrate the timing.

The intuition: a roll-forward changes “receivables grew” into a finite list of movements that can be tested one by one.

### 2. Separate volume, price, and terms

Sales can rise because a company sold more units, charged a higher price, or granted longer credit terms. Those causes have different effects on receivables and risk. A 20% sales increase with unchanged units and unchanged terms is not the same as a 20% increase driven by a large final-day shipment on 90-day terms.

Ask for units sold, average invoice value, customer terms, and the share of sales in the final week of the period. If those disclosures are not public, use management’s explanation as a hypothesis and test it against the ledger. A company may not be required to publish every operational number, so the absence of disclosure is a prompt for caution, not a substitute for evidence.

#### Worked example: the terms change hidden inside stable sales

This is illustrative. A company reports annual credit sales of \$36,500,000 and average receivables of \$3,000,000. Using 365 days, DSO is (\frac{{\$}3{,}000{,}000}{{\$}36{,}500{,}000} \times 365 = 30) days.

Next year, sales remain \$36,500,000, but average receivables rise to \$4,500,000 because standard terms move from 30 days to 45 days. DSO becomes (\frac{{\$}4{,}500{,}000}{{\$}36{,}500{,}000} \times 365 = 45) days.

The ratio has deteriorated by 15 days, but the cause could be a documented commercial decision rather than a fabricated sale. The next tests are customer contracts, collection experience under the new terms, and whether the allowance reflects the larger exposure. If management says terms did not change, the same numbers become more difficult to explain.

The intuition: the ratio tells you that financing grew; the contract tells you whether the growth was a business choice or a hidden problem.

### 3. Test inventory with both count and economics

A count answers “is it there?” It does not answer “can it be sold for more than its recorded cost?” For each material category, compare units on hand with recent sales, current price lists, customer orders, returns, markdowns, and expected costs to complete. Include damaged, expired, consigned, and customer-owned goods in the review.

**Net realizable value (NRV)** means the estimated selling price in the ordinary course of business less costs of completion and costs needed to make the sale. It is an estimate, so document the assumptions. A sharp difference between the assumptions used for a reserve and the prices actually achieved after period-end is a useful challenge.

#### Worked example: units are real, value is not

This is illustrative. The ledger contains 8,000 units at \$50 each, or \$400,000. A physical count confirms all 8,000 units. However, market evidence suggests a selling price of \$43 per unit and selling costs of \$3 per unit. The estimated recoverable amount is (8{,}000 \times ({\$}43 - {\$}3) = {\$}320{,}000).

The needed write-down is \$80,000. The goods are real, so an existence test passes. The valuation test does not. A forensic conclusion should say exactly that distinction instead of calling the inventory “fake.”

The intuition: physical existence and financial value are separate assertions; passing the count does not pass the valuation test.

### 4. Follow incentives without assuming intent

Incentives explain where to look, not what happened. Review bonuses tied to revenue, gross margin, earnings per share, borrowing-base availability, or debt covenants. Review whether a senior executive approved late-period entries or whether the same customer repeatedly receives unusual terms. Then seek corroborating evidence.

The word **alleged** matters in public cases. Regulators may describe allegations in a complaint; a later judgment, settlement, or restatement can add a different procedural outcome. Use the strongest source available and say what the document actually establishes. “The complaint alleged” is more accurate than “the company committed” when the cited document is a complaint.

### 5. Rank evidence by independence

Evidence generated by the company is useful but not equally independent. A customer’s bank payment is generally stronger for collection than an internal spreadsheet saying “paid.” A third-party warehouse confirmation is stronger for custody than a management-prepared location report. A lender’s borrowing-base certificate can reveal what receivables were pledged, although it also depends on the lender’s controls.

The hierarchy is not absolute. A bank payment can be a transfer from another company under common control. A customer confirmation can be signed by someone without authority. The practical rule is to combine evidence with different failure modes. Physical count, customer confirmation, and cash receipt do not all fail in the same way.

#### Worked example: a simple evidence scorecard

This is an illustrative review, not a statistical model. A \$900,000 receivable has a signed invoice, a shipping record, a customer confirmation, and a \$600,000 payment after period-end. The remaining \$300,000 is 75 days past due and later receives a full credit memo.

The evidence supports existence for the delivered transaction and collection of \$600,000. It does not support recognizing the remaining \$300,000 at full value at period-end if the credit condition already existed then. The analyst should investigate the reason and date of the credit, not average the two amounts into a made-up “probably collectible” figure.

The intuition: evidence is granular. Conclude about the supported portion, isolate the unsupported portion, and do not let one good document validate an entire balance.

### 6. Turn a red flag into a falsifiable question

Weak question: “Is management manipulating inventory?” Strong question: “If the inventory build is a deliberate pre-launch investment, do purchase orders, launch dates, unit forecasts, and subsequent sell-through support it?”

Weak question: “Are receivables fake?” Strong question: “For the largest period-end invoices, can an independent customer confirm the order and terms, can shipping prove transfer, and did cash arrive under the stated contract?”

This language protects both the analyst and the company. Many unusual balances are legitimate. The job is to specify evidence that would make the explanation more credible and evidence that would make it less credible.

## Common misconceptions

### “High inventory means fraud.”

No. Inventory rises before a launch, during supply disruption, or when a company deliberately builds safety stock. The question is whether the quantity and valuation are supported by demand, sell-through, and recoverable economics.

### “A low DSO means customers are healthy.”

No. DSO can fall because receivables were sold, because a reporting-date collection batch was unusually strong, or because sales were concentrated at a different point in the period. Check customer cash and terms.

### “A write-down means management failed.”

Not automatically. A write-down can be good reporting: it recognizes a loss promptly. The red flag is not the write-down itself but a pattern of avoiding it while evidence accumulates, followed by a sudden catch-up charge.

### “A customer confirmation proves the receivable.”

No. A confirmation can be incomplete, sent to the wrong person, or answered without checking the underlying terms. It is stronger when combined with an order, delivery evidence, acceptance, and cash collection.

### “Gross margin is an operating fact.”

Gross margin is an accounting result built from revenue and COGS. Inventory valuation and cutoff affect when costs enter COGS. Use margin as a signal to investigate, not as an independent measurement of product economics.

### “Negative operating cash flow proves fraud.”

No. A growing company can consume cash because receivables and inventory legitimately expand. Negative cash flow becomes more concerning when working-capital growth is unexplained, reported profit is smooth, and post-period evidence contradicts the balances.

## How it shows up in real markets

An investor rarely gets a perfect fraud label in real time. More often, the available evidence is a sequence of small inconsistencies:

1. Inventory grows faster than sales.
2. Gross margin stays unusually smooth.
3. DSO rises or becomes dependent on quarter-end sales.
4. The allowance does not move with aging.
5. Returns, discounts, or markdowns appear after the reporting date.
6. Operating cash flow trails profit for a reason management cannot explain clearly.

Any one item can be ordinary. Several items that point to the same economic loss deserve deeper work.

### A compact investigation sequence

Start with the filings. Read the revenue-recognition policy, inventory valuation policy, allowance roll-forward, aging disclosure, related-party note, factoring or securitization disclosure, and commitments. Then recompute DSO and DIO using the company’s actual period and definitions.

Next, build a bridge from opening balance to closing balance. For receivables, bridge sales, collections, credits, write-offs, acquisitions, currency movements, and closing receivables. For inventory, bridge purchases or production, COGS, write-downs, disposals, acquisitions, and closing inventory. A bridge makes a large unexplained plug visible.

Finally, connect the bridge to independent evidence. Use bank statements for cash, customer confirmations for contracts, warehouse records for inventory, lender collateral reports for borrowing, and post-period transactions for reality. The goal is triangulation: three imperfect views that agree are more persuasive than one polished ratio.

### When this matters to an investor

These tests matter because current assets can make a company look both more profitable and more liquid than it really is. An overstated receivable increases net working capital until it is written off. An overstated inventory reduces COGS until it is written down or sold at a loss. If debt covenants use earnings or eligible receivables, the accounting issue can also become a financing issue.

Do not trade on a ratio alone. A disciplined reader asks what changed, why it changed, what evidence would confirm the explanation, and what evidence would falsify it. That is the difference between forensic analysis and suspicion.

![The red-flag dashboard connects receivables, inventory, income, and cash evidence; a cluster is stronger than one isolated ratio.](/imgs/blogs/inventory-and-receivables-inflation-the-classic-red-flag-7.webp)

## Sources & further reading

- [SEC Litigation Release No. 18022: American Tissue](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-18022), March 10, 2003. Source for the case date, \$165 million securities figure, and SEC allegations concerning inventory, capitalized expenses, and bogus bill-and-hold sales.
- [SEC complaint: American Tissue, Inc.](https://www.sec.gov/litigation/complaints/comp18022.htm), filed March 10, 2003. Source for the alleged \$21.8 million bill-and-hold sales and receivables, the \$15.6 million supplies-inventory reclassification, the 85% borrowing-base detail, and the alleged \$15.5 million reported nine-month income.
- [The cash conversion cycle: What working capital reveals before earnings do](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals), a companion guide to DSO, DIO, DPO, and cash tied up in operations.
- [The income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings), for reading revenue, COGS, gross profit, and operating cash together.
- [Reading the cash flow statement: Why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income), for the cash-flow reconciliation that should follow a working-capital red flag.
- [The footnotes and MD&A: Where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried), for policies, estimates, related parties, and management explanations.
