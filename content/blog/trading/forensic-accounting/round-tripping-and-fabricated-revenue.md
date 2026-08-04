---
title: "Round-tripping and fabricated revenue: when the top line has no customer behind it"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "A beginner-friendly forensic guide to fake sales, wash trades, related-party revenue, and phantom customers—and the statement clues that expose growth without economic substance."
tags: ["forensic-accounting", "revenue-recognition", "round-tripping", "related-party-transactions", "earnings-quality", "fraud-detection", "cash-flow", "enron", "luckin-coffee", "financial-statements"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 37
---

> [!important]
> **TL;DR** — Revenue is not cash and a recorded sale is not automatically an economic sale. Round-tripping, related-party deals, and invented customers can make the top line grow while the bank account, margins, and receivables tell a different story.
>
> - A real sale transfers control of something valuable to an independent customer who has both the ability and intention to pay. A paper sale can create the journal entry without creating that economic event.
> - The simplest forensic test is a three-way tie-out: revenue on the income statement, receivables and contract assets on the balance sheet, and cash collected on the cash-flow statement.
> - In a round trip, two parties pre-arrange offsetting purchases and sales. Gross revenue and expenses can inflate while profit and cash barely move.
> - The SEC said Reliant Resources reported $36.5 billion of 2001 gross revenue, including more than $3.8 billion from round-trip trades; the trades overstated both revenue and expenses.
> - Enron and Luckin show two different versions of the same warning: a transaction can look busy in a ledger while the underlying customer demand is absent or misrepresented.

Imagine a café tells you it sold 100 coffees. The cash register says 100 coffees, the inventory system says 100 cups left the storeroom, and the income statement says $100 of sales. That is a coherent story.

Now imagine the café's owner buys $100 of coffee from a friend, and the friend immediately buys $100 of coffee from the owner's café. Each business can show a sale. Neither has found a new end customer. If the café records only its outward leg, revenue appears. If both parties record their legs gross, the market looks much larger than it is.

This is the core forensic question: **what changed outside the accounting system?** Did a new customer receive a useful product? Did cash arrive from that customer? Did inventory leave the company for a reason that makes commercial sense? Or did two ledgers simply agree to tell the same flattering story?

The diagram below is the mental model. A fabricated sale starts as a document, travels through a journal entry, inflates the income statement, and then runs into the cash-flow statement. The missing economic substance is not always visible in one line; it appears in the gap between the statements and in the identity of the supposed customer.

![A fake sale moves from a document to revenue and receivables, but the cash-flow statement and independent customer evidence fail to follow.](/imgs/blogs/round-tripping-and-fabricated-revenue-1.webp)

The purpose of this article is not to label every unusual transaction fraudulent. Businesses use distributors, consignment arrangements, channel financing, barter, and related parties for legitimate reasons. The job is to separate a complicated but real sale from a sale that exists mainly to manufacture a headline.

## Foundations: the building blocks

### Revenue, cash, receivables, and profit are different things

**Revenue** is the amount a company reports for goods or services transferred to customers during a reporting period. It is the top line because it sits near the top of the income statement. It is not a synonym for cash collected.

**Cash** is money already received or paid through the bank account or cash drawer. Cash is a balance-sheet asset and a cash-flow statement movement. It is harder to invent than revenue, although a company can still disguise restricted cash, borrow money temporarily, or circulate cash between related accounts.

**Accounts receivable**, usually shortened to A/R, is an amount a customer owes after the company has recorded a sale but before the customer has paid. It is an asset because the company claims it will convert into cash. A/R is normal in many businesses; an unusual surge in A/R relative to revenue is a question, not a verdict.

**Profit** is what remains after expenses are matched against revenue. A company can report profit while collecting little cash because accrual accounting records earned revenue before collection. That timing difference is useful when it reflects normal credit terms. It is dangerous when the “customer” is not independent, cannot pay, or never received the promised product.

The three statements answer different questions:

| Statement | Main question | What a fake sale usually touches |
| --- | --- | --- |
| Income statement | What performance did management report for the period? | Revenue and sometimes cost of sales or profit |
| Balance sheet | What does the company claim to own or be owed at period end? | A/R, contract assets, inventory, related-party receivables |
| Cash-flow statement | What cash actually moved, and from whom? | Operating cash, customer collections, loans, or unexplained transfers |

The statements are linked. Revenue may create A/R. Collection reduces A/R and increases cash. A sale of inventory reduces inventory and creates cost of goods sold. If a reported sale produces revenue and A/R but never produces collection, the story is incomplete.

### The five questions behind a real sale

Accounting standards describe revenue through a contract and performance obligations. You do not need to memorize the standard to use the forensic intuition. Ask five plain-English questions:

1. **Who is the customer?** A named legal entity is not enough. Identify its owners, address, business, funding, and relationship to the seller.
2. **What was promised?** The product, service, quantity, price, delivery terms, returns, rebates, and side agreements should fit together.
3. **When did control transfer?** A signed invoice is not proof that the buyer obtained the goods or service.
4. **Can the customer pay?** A receivable from a shell or a buyer funded by the seller is not equivalent to cash demand.
5. **What happened after the period ended?** Collections, returns, cancellations, credit notes, and write-offs often reveal whether a period-end sale was genuine.

**Economic substance** means the real business effect of a transaction, not merely its legal form or paperwork. A contract can be genuine in form but hollow in substance if the parties have agreed to reverse it, if the seller retains the risks of ownership, or if the buyer is not meaningfully independent.

### Round-tripping and wash trades

A **round trip** is a pair of pre-arranged transactions that offset one another. In energy trading, the same volume may be bought and sold with the same counterparty at the same price, with no delivery intended and no profit expected. In a retail setting, a supplier and distributor may buy from one another to create gross sales. In securities markets, a **wash trade** is a matched buy and sell that creates the appearance of market activity without a genuine change in beneficial ownership.

The exact legal treatment depends on the facts and the reporting rules. The forensic signature is economic: lots of activity, little risk transfer, little independent demand, and no durable cash generation.

### Related parties and phantom customers

A **related party** is a person or entity connected to the company through control, significant influence, common ownership, key management, or close family relationships. Related-party transactions are not automatically improper. A company may legitimately sell to a subsidiary or buy services from an affiliate. They deserve extra disclosure and testing because the normal market discipline of an independent buyer is weaker.

A **phantom customer** is a customer recorded in the sales system that is fictitious, inactive, misidentified, or used to disguise the true party to a transaction. The customer can be a paper entity, a friendly intermediary, or a real business whose name was used without a real order.

### Worked example: the $100 sale with no cash

This is illustrative arithmetic, not a claim about a real company. Suppose a company ships a product and invoices a customer for $100. It records:

```journal
Dr Accounts receivable       $100
    Cr Revenue                         $100
```

Revenue rises by $100. A/R rises by $100. Cash is still $0. If the product cost $60, the company also records:

```journal
Dr Cost of goods sold         $60
    Cr Inventory                         $60
```

Reported gross profit is $100 − $60 = $40. But no cash has arrived. If the customer pays later, the collection entry is:

```journal
Dr Cash                       $100
    Cr Accounts receivable              $100
```

The forensic question is not “can revenue exist before cash?” It can. The question is whether the missing cash is ordinary credit timing or evidence that the customer, order, delivery, or collectability was never real.

**Intuition:** a sale can be an accounting event before it is a cash event, but a durable business must eventually make the two meet.

## 1. How fabricated revenue gets onto the page

The easiest way to understand manipulation is to follow the journal entry. A **journal entry** is the debit-and-credit record that posts an event into the ledger. Debits and credits are not “good” and “bad”; they are the two-sided bookkeeping mechanism that keeps the accounting equation balanced.

The common entry for a credit sale is debit A/R and credit revenue. That entry is powerful because it increases the top line without requiring cash today. A manipulator can exploit the gap by creating an invoice, recording a delivery, or booking a contract asset even when the customer relationship is conditional or circular.

Four patterns recur:

| Pattern | What the ledger says | What the investigator asks |
| --- | --- | --- |
| Early shipment | Product was delivered this period | Was control transferred, or was it sitting in a warehouse or with a distributor? |
| Bill-and-hold | Customer was billed before physical delivery | Who requested the hold, and was the product complete and separately identified? |
| Channel stuffing | Distributor bought unusually large quantities | Did sell-through to end customers happen, or did returns and discounts follow? |
| Phantom or related-party sale | Named buyer owes the company | Is the buyer independent, solvent, and economically motivated? |

### Worked example: the journal entry that inflates the top line

Assume a fictional company wants to show $100 of extra revenue at year end. It creates an invoice to “North Star Trading,” a related entity, and records:

```journal
Dr Accounts receivable       $100
    Cr Revenue                         $100
```

No cash moves. No inventory moves. The income statement shows $100 more revenue and, if no cost is recorded, $100 more profit. The balance sheet shows $100 more A/R. The cash-flow statement starts with higher net income, then subtracts the $100 increase in A/R in operating cash flow. Cash from operations is therefore unchanged by the entry.

That is why a revenue-only test is weak. The income statement looks better, while the cash-flow statement quietly says, “this profit is still owed to us.” If the receivable later gets written off, the profit reverses through bad-debt expense.

**Intuition:** the debit and credit can balance perfectly while the business event they purport to represent never happened.

![Before-and-after journal entries show how an unsupported $100 invoice raises revenue and receivables while operating cash remains unchanged.](/imgs/blogs/round-tripping-and-fabricated-revenue-3.webp)

### The evidence trail a genuine customer leaves

A real sale usually leaves multiple independent traces: a purchase order, credit approval, shipping record, carrier scan, customer receipt, product usage, bank collection, tax invoice, and a customer who confirms the terms. No single document proves substance. The strength comes from independent evidence agreeing.

Forensic accountants therefore confirm more than the invoice. They may send positive confirmations, inspect subsequent cash receipts, trace shipping documents, match serial numbers, review email and pricing approvals, and compare the transaction with the customer's own records. A response from an address controlled by the seller is weaker than a payment from an independently verified bank account.

## 2. Why round-tripping inflates the top line without creating a business

Round-tripping is especially confusing because it can involve real contracts, real invoices, and sometimes real cash. The problem is not that a transaction is literally imaginary. The problem is that the two legs are economically pre-arranged and cancel each other.

Consider two energy traders, A and B. A agrees to sell 1,000 units to B for $1,000,000. At the same time, B agrees to sell the same 1,000 units back to A for $1,000,000. Neither intends delivery. If A records both a sale and a purchase, gross revenue and cost of goods sold each rise by $1,000,000. Gross profit is unchanged. If B does the same, both businesses look busier even though no new end demand appeared.

The SEC described a similar economic pattern in its CMS Energy order: pre-arranged purchases and sales of the same volume at the same price, with no delivery contemplated and neither party making a profit. The order said CMS overstated revenue and expenses by $1.0 billion, or 10% of revenue, in 2000 and by $4.2 billion, or 36% of revenue, in the first three quarters of 2001. The SEC said the trades did not affect net earnings, but they distorted reported revenue and trading volume.

### Worked example: a $1,000,000 round trip

These figures are illustrative. Company A buys and sells the same notional amount under a pre-arranged offsetting trade:

```journal
Sale leg:       Dr Accounts receivable      $1,000,000
                    Cr Revenue                         $1,000,000

Purchase leg:   Dr Cost of goods sold       $1,000,000
                    Cr Accounts payable                $1,000,000
```

Reported revenue increases by $1,000,000. Reported COGS increases by $1,000,000. Gross profit changes by $0. If both receivable and payable settle for the same amount, the company may see cash inflow and cash outflow that cancel. The headline “revenue grew by $1,000,000” is technically compatible with the ledger, but it says almost nothing about new customers or profitable demand.

Now add a 1% fee paid to an intermediary. The company pays $10,000 to make the loop happen. It has created $1,000,000 of gross activity at a real economic cost of $10,000. That cost is the clue: the company paid to rent the appearance of scale.

**Intuition:** gross revenue can be huge when the business is only passing the same economic value back and forth.

![A round trip pairs a $1,000,000 sale with an equal offsetting purchase; revenue and COGS rise together while net economic demand remains zero.](/imgs/blogs/round-tripping-and-fabricated-revenue-4.webp)

### Gross versus net presentation

Accounting sometimes requires a company to report a transaction gross, and sometimes net. A **principal** controls a good before transferring it and may report the customer price as revenue. An **agent** arranges for another party to provide the good and may report only its commission. The judgment turns on control and risk, not on which presentation makes growth look better.

The forensic issue is not “gross revenue is always suspicious.” It is whether the company is reporting a pass-through as if it were an independent business sale, whether it bears inventory risk, and whether the gross amounts are meaningful to readers. A marketplace can be legitimate and still have a large difference between gross transaction volume and net revenue.

## 3. The cash-conversion test: follow the sale beyond the income statement

**Cash conversion** asks how reported earnings turn into cash. A common starting point is operating cash flow compared with net income. The ratio is not a universal health score; industries differ, and a fast-growing company can consume cash while building inventory. But a widening gap is a prompt to investigate.

The most direct revenue test is a bridge:

1. Start with reported revenue.
2. Ask how much became cash from customers.
3. Reconcile the rest to changes in A/R, contract assets, returns, refunds, and financing.
4. Examine whether later collections were normal, delayed, or reversed.

**Contract assets** are rights to consideration that depend on something other than the passage of time, while A/R is an unconditional right to payment. Both can sit between revenue and cash. Their labels matter less than the question: what precisely must happen before the company can collect?

### Worked example: revenue grows while customer cash does not

Imagine a hypothetical distributor with two years of activity:

| Illustrative amount | Year 1 | Year 2 |
| --- | ---: | ---: |
| Revenue | $100 | $160 |
| Cash collected from customers | $95 | $98 |
| Ending A/R | $20 | $82 |
| Returns and credits after year end | $2 | $25 |

Revenue grew $60, or 60%. Customer cash grew only $3. A/R grew $62, and later credits jumped from $2 to $25. None of those facts proves fraud. A new enterprise customer could have negotiated 90-day terms, or the company could have acquired a receivables portfolio. But the growth story now needs an explanation, not applause.

If Year 2 net income was $20 and operating cash flow was negative $5, the company might still be investing in growth. If it claims the growth is driven by thousands of cash-paying consumers, the figures conflict with that story. The next steps are customer-level aging, subsequent receipts, shipment evidence, credit notes, and related-party disclosures.

**Intuition:** when reported sales sprint ahead of customer cash, the unpaid balance is not a footnote; it is the central object of the investigation.

### What the cash-flow statement can and cannot prove

The operating section of the cash-flow statement adjusts net income for working-capital movements. An increase in A/R usually reduces operating cash because revenue has been recognized without collection. That is useful, but not magic. A company can borrow money and deposit it into a customer-looking account, sell receivables, or classify cash flows in ways that obscure the source.

Look at customer cash receipts where the filing provides them, not merely total cash. Compare operating cash flow with gross profit, not only net income. Examine restricted cash, factoring, non-recourse receivable sales, and advances from customers. Then ask whether the cash came from ordinary customers or from a related party that received funding from the company.

## 4. Related parties: the hidden hand behind an apparently independent sale

The independence of a customer is a control, not a detail. A third-party buyer has its own budget, inventory risk, and reason to reject an uneconomic deal. A related party may agree because the seller wants a target, the buyer wants financing, or both sides share an owner.

The warning is strongest when three features cluster:

- the buyer appears in a related-party note, affiliate list, or ownership record;
- the transaction has unusual pricing, payment terms, guarantees, or return rights; and
- the seller's reported revenue depends materially on the buyer.

Related-party disclosure rules require companies to describe material transactions and relationships, but disclosure is only as good as management's identification process. Search for common directors, addresses, phone numbers, bank accounts, beneficial owners, and intermediaries. Compare the buyer's purchase volume with its own size and business model.

### Worked example: the distributor that is really the seller's wallet

Suppose a fictional company sells $500 of inventory to an affiliate for $500 on 31 December. The affiliate has no employees, no warehouse, and no external customers. The seller records revenue and A/R. The affiliate records inventory and A/P. Two weeks later, the seller lends the affiliate $500 so it can pay the invoice:

```journal
Seller:     Dr Note receivable from affiliate    $500
                Cr Cash                                      $500

Affiliate:  Dr Cash                               $500
                Cr Note payable to seller                    $500

Affiliate:  Dr Accounts payable                   $500
                Cr Cash                                      $500

Seller:     Dr Cash                               $500
                Cr Accounts receivable                        $500
```

Cash briefly moved through the affiliate, but the seller financed its own collection. Consolidated financial statements may eliminate the intercompany balances if the affiliate is consolidated, but a non-consolidated related party can make the cash trail look stronger than the end demand.

The questions are: was the loan disclosed, did the affiliate have the ability to pay without it, and did any independent customer buy the product? A round trip of money is not the same as customer cash.

**Intuition:** the identity and funding of the buyer matter as much as the invoice amount.

![A related-party network shows the company funding an affiliate that buys its reported revenue, creating a circular cash trail rather than independent demand.](/imgs/blogs/round-tripping-and-fabricated-revenue-6.webp)

## 5. Red flags that deserve a second look

No red flag is a verdict. A fast-growing company can have high A/R, a distributor can be legitimate, and a related party can provide useful services. Forensic work looks for combinations that are hard to explain together.

### Red flag 1: revenue growth without matching cash or capacity

Compare revenue growth with customer cash, receivable aging, inventory movement, warehouse capacity, employee counts, delivery miles, usage data, or other operating measures. The correct comparison depends on the business. A software company may have deferred revenue and usage logs; a retailer may have point-of-sale data and store traffic; an energy trader may have volumes, delivery points, and counterparty confirmations.

### Red flag 2: end-of-period spikes

Revenue concentrated in the final days of a quarter can be ordinary seasonality. It can also reflect pressure to hit a target. Review invoices, shipping terms, acceptance clauses, side letters, return rates, and post-period credit notes. A customer who accepts goods only after inspection may mean control did not transfer when management said it did.

### Red flag 3: unusual margins or gross-versus-net growth

Round trips can increase both revenue and COGS without changing gross profit. That can make gross margin fall while reported scale rises. A company may also report enormous gross transaction volume while keeping only a small commission. Ask which line management calls revenue and whether the presentation changed when growth slowed.

### Red flag 4: receivables that age, roll, or require financing

An **aging schedule** groups receivables by how long they have been outstanding. A growing share past due, repeated extensions, or large balances from a few counterparties increases collection risk. Receivable factoring can turn A/R into cash, but it may also move credit risk and fees into footnotes rather than making the underlying sale stronger.

### Red flag 5: customers with shared infrastructure

Look for customer addresses matching employees, directors, affiliates, or other “customers.” Compare domain registrations, phone numbers, bank details, shipment destinations, and contact people. A cluster does not prove a scheme; it tells you where to request independent confirmation.

### Red flag 6: management measures that outrun audited revenue

Operational metrics such as orders, users, stores, gross merchandise volume, or “sales generated” may not equal recognized revenue. Define the unit. Ask whether an order was paid, delivered, returned, netted against discounts, or generated by a related party. A metric can be useful while still being a poor substitute for revenue.

![A red-flag dashboard links rising A/R, period-end spikes, related-party concentration, weak cash conversion, and unusual margins into a single investigation queue.](/imgs/blogs/round-tripping-and-fabricated-revenue-5.webp)

### A compact investigation dashboard

| Question | Benign explanation | Escalation test |
| --- | --- | --- |
| Did A/R rise faster than revenue? | Longer standard terms or a new enterprise contract | Subsequent cash by customer and aging by invoice |
| Did revenue spike at period end? | Seasonal demand | Shipping, acceptance, returns, and side agreements |
| Did gross revenue and COGS rise together? | A principal genuinely took inventory risk | Delivery, price risk, and matched counterparties |
| Is a buyer related? | Normal affiliate supply chain | Funding, ownership, independent end demand |
| Did margins improve while cash worsened? | Temporary mix or investment cycle | Customer confirmations and post-period reversals |

## 6. The forensic workflow: from suspicion to evidence

Start with the general ledger, not the press release. Extract revenue postings by customer, date, product, location, salesperson, journal preparer, and approval. Search for manual top-side entries, round dollar amounts, entries posted after close, and reversals in the next period.

Then reconcile the population to the subledger and the financial statements. The **subledger** is the detailed record behind a control account such as A/R. A difference between the subledger and the general ledger can be timing, but it can also reveal manual adjustments or incomplete interfaces.

Next, stratify transactions. Sample ordinary small customers and large unusual customers separately. A random sample alone can miss the handful of deals that drive the quarter. For each selected sale, inspect the contract, order, invoice, delivery, customer acceptance, collection, returns, and accounting entry.

### Worked example: a risk-based sample

Assume a fictional quarter has 10,000 invoices totaling $10 million. Nine thousand nine hundred invoices are ordinary and total $8 million. One hundred invoices are period-end transactions totaling $2 million, or 20% of revenue.

A purely random sample of 20 invoices might miss the 100 unusual items. A risk-based design tests the period-end population, the largest invoices, all related-party invoices, and a smaller sample of ordinary transactions. If 10 of the 100 period-end invoices are returned or unpaid while ordinary invoices behave normally, the issue is not “the whole ledger is fake.” It is a concentrated control or recognition problem.

**Intuition:** forensic sampling follows economic risk, not just statistical convenience.

### Confirmations are evidence, not a magic spell

A confirmation asks a customer to verify a balance or transaction. Positive confirmation requires a response whether the balance is right or wrong; non-response does not prove the balance is correct. A weak confirmation can be routed through the seller, answered by a friendly employee, or limited to a balance without terms.

Use alternative procedures: inspect bank receipts, independent shipping records, customer purchase orders, tax records, and the customer's own inventory movement. If the customer confirms the balance but cannot explain what it bought, who paid, or where the product went, the response is not strong evidence of substance.

## 7. Real company case: Enron's sham sales and the appearance of earnings

Enron is a named historical case, not a template for accusing every complex company. The facts below are framed as allegations and findings in SEC enforcement materials, not as a claim that every transaction in the company operated the same way.

In a 2002 SEC release concerning former CFO Andrew Fastow, the Commission alleged that two transactions involving Nigerian energy barges and a Cuiabá power plant were sham sales or “asset-parking” arrangements. For the Nigerian barges, the SEC alleged that Enron recorded approximately $12 million of earnings in 1999 even though risk did not truly pass and a later take-out was arranged. In a separate SEC case concerning Merrill Lynch, the Commission alleged that an energy transaction helped Enron report $50 million of income to reach a year-end earnings target in 1999. The SEC's description said Merrill paid or received fees in arrangements designed to be effectively risk-free.

The lesson is not simply “Enron used related parties.” It is more precise: a sale can be a financing arrangement in disguise when the seller promises to repurchase, guarantees the buyer's return, or retains the meaningful risks and rewards. The balance sheet may show an asset sale, while the economics still look like a loan.

The red flags were structural: side agreements, a motivated intermediary, a reporting deadline, and a result that helped management reach a target. A forensic review would compare the stated sale with the buyer's downside risk, funding source, repurchase rights, and subsequent cash flows.

![A case-study matrix compares Enron, Reliant, CMS, and Luckin by mechanism, reported effect, evidence source, and forensic lesson.](/imgs/blogs/round-tripping-and-fabricated-revenue-7.webp)

## 8. Cautious preview: Luckin Coffee and fabricated retail sales

Luckin belongs in this article because the SEC's 2020 litigation release described a different but related revenue problem: alleged fabricated retail sales using related parties and false records. The SEC said that from at least April 2019 through January 2020, Luckin intentionally fabricated more than $300 million in retail sales through three purchasing schemes. It also said employees allegedly inflated expenses by more than $190 million, created a fake operations database, and altered accounting and bank records.

The SEC release said reported revenue was allegedly overstated by approximately 28% for the period ended 30 June 2019 and 45% for the period ended 30 September 2019. It also said Luckin raised more than $864 million from debt and equity investors during the period of the alleged misconduct and agreed to pay a $180 million penalty in December 2020, without admitting or denying the allegations.

Luckin's special committee announced on 1 July 2020 that its internal investigation had substantially completed. The company's release said the investigation reviewed more than 550,000 documents from more than 60 custodians, interviewed more than 60 witnesses, and found 2019 net revenue inflated by approximately RMB 2.12 billion. Those are reported findings from the company's disclosure and SEC enforcement record; they should not be casually generalized to other companies or countries.

The forensic lesson is the interaction among operational data, related parties, expenses, and bank records. A fake retail sale is not fixed by adding a customer name. It must survive the trail: an actual order, payment, delivery or service, independent customer behavior, and a consistent operational database. When management creates both false sales and false expenses, the goal may be to make the income statement look internally plausible rather than simply to maximize profit.

## 8. Statement-line bridges that expose the missing substance

The most useful forensic analysis is often not a ratio. It is a bridge that starts with one reported line and asks what other lines must move if the claim is true. Bridges make the accounting relationship visible and prevent a reader from treating isolated metrics as proof.

### Revenue to receivables

For a simple business without acquisitions, foreign-exchange effects, or unusual reclassifications, the basic bridge is:

$$
\text{Closing A/R} \approx \text{Opening A/R} + \text{Credit revenue} - \text{Customer cash} - \text{Credits and write-offs}
$$

This is an explanatory abstraction, not a claim that every filing uses exactly these lines. The symbols mean: **A/R** is accounts receivable; **credit revenue** is revenue not paid at the point of sale; **customer cash** is cash collected from customers; **credits and write-offs** reduce what the company expects to collect.

![The three-statement bridge: a $100 illustrative sale creates revenue and receivables first, then cash only if the customer actually pays.](/imgs/blogs/round-tripping-and-fabricated-revenue-2.webp)

![A cash-conversion timeline shows revenue first, receivables next, and customer cash later; returns and write-offs reveal whether the bridge ever completes.](/imgs/blogs/round-tripping-and-fabricated-revenue-8.webp)

If revenue rises and A/R rises, the bridge may be ordinary growth. If revenue rises, A/R rises, customer cash does not, and credits appear just after year end, the bridge is telling you where to look. The right response is not to conclude “fraud.” It is to request a commercial explanation and test it against the ledger.

### Revenue to inventory and cost of sales

When a company sells physical goods, a genuine sale normally has two sides: the revenue side and the inventory side. Inventory leaves the company and becomes cost of goods sold. If revenue is rising while inventory does not move, check whether the business is a service company, whether inventory is held by a distributor, or whether the revenue is only a commission. If revenue and inventory both rise, inspect whether the goods were actually shipped and accepted.

**Inventory days** estimate how long stock stays before sale. The metric is sensitive to seasonality and accounting policy, so use trends and customer-level evidence rather than a universal cutoff. A warehouse full of returned or unsold goods is not the same as product consumed by end customers.

### Revenue to contract liabilities

**Deferred revenue**, also called a contract liability, is cash collected before the company has delivered the promised good or service. It is not a failure. In a subscription business, customer cash can arrive before revenue, so deferred revenue can be a healthy sign of demand.

The reverse pattern deserves care: revenue is recognized before cash, while deferred revenue falls or fails to grow despite supposedly accelerating bookings. The explanation may be a change in contract mix or recognition policy. It may also be that management is converting future obligations into current revenue too aggressively. Read the revenue note and the contract-liability roll-forward together.

### Worked example: three bridges, one story

Use a hypothetical software company with the following illustrative facts:

| Line | Year 1 | Year 2 |
| --- | ---: | ---: |
| Revenue | $100 | $150 |
| Customer cash | $96 | $100 |
| A/R | $18 | $64 |
| Deferred revenue | $30 | $20 |
| Credits after year end | $1 | $12 |

Revenue grew $50. Customer cash grew only $4. A/R grew $46, while deferred revenue fell $10 and post-period credits rose $11. One explanation is a major shift from annual prepayment to monthly invoicing; another is aggressive recognition. The numbers do not choose the explanation by themselves. They tell the investigator to inspect contract terms, customer concentration, delivery logs, and the timing of credit notes.

**Intuition:** a bridge turns “sales grew” into a set of testable claims about invoices, cash, obligations, inventory, and reversals.

## 9. What a careful review does with uncertainty

Forensic accounting is not the art of finding one dramatic clue. It is the discipline of weighing evidence that has different reliability. A signed contract can be strong evidence of agreed terms but weak evidence of delivery. A bank receipt is strong evidence that money moved but may be weak evidence that the money came from an independent customer. A customer confirmation is useful but can be compromised by collusion.

### Rank evidence by independence

Evidence generated inside the company is not useless. It is simply less independent than a record created by an outside party for its own purpose. A shipping scan generated by a third-party carrier, a bank statement obtained directly from the bank, and an end customer's purchase record each answer different parts of the question.

The reviewer should map each claim to evidence:

| Claim management makes | Evidence that supports it | Weakness to consider |
| --- | --- | --- |
| Product was delivered | Carrier record, signed receipt, customer inventory | Seller may control the delivery address or receipt |
| Customer owes money | Direct confirmation, contract, subsequent payment | Payment may be funded by the seller or affiliate |
| Buyer is independent | Ownership, directors, address, bank, tax records | Nominee owners can hide control |
| Customer used the product | Usage logs, resale, consumption, inventory depletion | A distributor can hold stock without selling it |
| Sale was final | Return history, side letters, credit notes | Terms can be outside the main contract |

### The difference between error, aggressive accounting, and fraud

An **error** is an unintentional misstatement. **Aggressive accounting** pushes a judgment toward the favorable end of a permitted range or uses a strained interpretation. **Fraud** involves intentional deception or concealment for an improper benefit. The boundary depends on evidence of intent and the applicable law.

Do not use the word “fraud” merely because cash conversion is weak. Use more precise language: “unpaid revenue,” “unusual related-party concentration,” “unsupported delivery evidence,” or “reported allegation.” In a public case, quote the regulator's characterization and date it. That protects the reader from turning a risk signal into an unsupported accusation.

### Worked example: the same number, three possible explanations

Suppose a company records $1,000 of revenue on 30 December and receives $0 by 31 December. That fact has at least three plausible explanations:

1. **Ordinary credit:** an independent customer received the product under documented 60-day terms and pays in February.
2. **Aggressive timing:** the customer had not accepted the product by 31 December, but management recorded revenue early.
3. **Fabrication:** the invoice names a shell or related party, no product moved, and the entry reverses when the reporting pressure passes.

The number alone cannot distinguish the cases. The evidence can: contract acceptance, shipping, customer capacity, subsequent cash, return patterns, related-party funding, and reversal entries.

**Intuition:** forensic judgment is not about making an unusual number sound sinister; it is about finding the independent evidence that separates competing explanations.

## 10. Incentives, controls, and the people who can create a loop

Round-tripping rarely happens because a spreadsheet accidentally typed the same number twice. It requires an incentive and a control path. The incentive might be a revenue target, a debt covenant, a bonus threshold, a financing round, an IPO narrative, or a desire to appear larger than a competitor. The control weakness may be that sales creates customers, finance approves its own invoices, and no one independently checks collections.

### Map the incentive before mapping the transaction

Ask what changes when revenue crosses a threshold. Does a bonus vest? Does a lender waive a test? Does a valuation model use revenue multiples? Does management's public guidance become achievable? The presence of an incentive does not prove misconduct, but it tells you which periods and thresholds deserve concentrated testing.

### Separate duties that should not share one hand

**Segregation of duties** means different people authorize, execute, record, and reconcile a transaction. A strong revenue process separates customer creation from credit approval, shipping from invoicing, and cash application from sales compensation. A small company may not have enough staff for perfect separation, so it needs compensating reviews by an owner, audit committee, or outside accountant.

Controls do not need to be sophisticated to catch a round trip. A monthly list of new customers, customers with the same address, invoices posted after close, and receivables past due can expose an unusual pattern. A rule that requires independent approval for related-party sales can stop the easiest loops.

### Top-side entries and post-close reversals

A **top-side entry** is an adjustment posted at the consolidated or reporting level rather than through the ordinary transaction system. Such entries can be legitimate: consolidation eliminations, tax adjustments, and audit corrections often live there. They are also a place where unsupported revenue can be added with fewer operational traces.

Look for entries with round amounts, vague descriptions, late timestamps, manual preparers, unusual accounts, and reversals in the next period. A reversal is not proof of fraud; accruals reverse routinely. The question is whether the reversal corresponds to a documented estimate or quietly removes a sale that never collected.

### Worked example: a target-driven quarter

Imagine an illustrative bonus plan that pays $10,000 if quarterly revenue reaches $1,000,000. Before the close, the ledger shows $980,000. Management records a $25,000 invoice to a related distributor and a $10,000 invoice to a customer whose goods remain in the seller's warehouse. The reported total becomes $1,015,000 and the target is reached.

The investigation does not start with the bonus payment. It traces the two invoices: ownership, delivery, acceptance, financing, returns, and the next-quarter reversal. If the distributor was funded by the seller and the goods were not controlled by the customer, the accounting treatment may not represent two genuine sales. The incentive explains why these particular entries deserve attention; it does not replace evidence.

**Intuition:** a pressure point tells you where to look, while the transaction trail tells you what happened.

## 11. A practical reading order for a new company

When you first encounter a company, do not begin by building a complicated model. Use a repeatable sequence:

1. Read the revenue-recognition policy and identify the major revenue streams.
2. Write down the exact unit behind each stream: product sold, subscription month, commission, energy volume, user order, or other measure.
3. Compare revenue growth with A/R, customer cash, contract liabilities, inventory, and returns.
4. Read the related-party note and identify every material buyer, seller, lender, and guarantor.
5. Search the filing for “side agreement,” “right of return,” “bill and hold,” “consignment,” “factoring,” “customer concentration,” “variable consideration,” and “subsequent event.”
6. Read the cash-flow statement and the MD&A together. The cash-flow statement shows the movement; the MD&A supplies management's explanation.
7. Compare the current policy, metric definition, and segment presentation with prior years.

The point of this sequence is to keep the analysis grounded. Revenue recognition is not a contest to find a suspicious phrase. It is a model of how a customer transaction is supposed to travel from demand to delivery to cash.

### Worked example: turning a filing into questions

Assume a hypothetical filing says revenue grew from $200 to $300, A/R from $40 to $130, customer cash from $190 to $205, and returns from $3 to $20. Rather than writing “the company may be fraudulent,” write a test plan:

- obtain the $90 increase in A/R by customer and invoice age;
- identify the customers behind the $100 of period-end revenue;
- test whether the $20 of returns relates to those invoices;
- trace $205 of customer cash to payer names and bank accounts;
- inspect related-party and financing disclosures;
- compare shipped, accepted, and returned units.

That plan can produce a benign answer, an accounting correction, or evidence of intentional fabrication. It is useful under all three outcomes.

**Intuition:** good forensic analysis converts suspicion into a finite list of evidence requests.

## Common misconceptions

### “Revenue before cash is automatically fraud.”

No. Credit sales are normal. A company can deliver today under 30-day terms and collect next month. The test is whether the customer has a genuine obligation and whether subsequent collections behave like the company's stated credit policy.

### “If profit did not rise, a round trip did no harm.”

Round trips can inflate revenue, expenses, trading volume, growth rates, market share claims, and the perceived scale of the business even when net income is unchanged. Those measures can influence valuation, lending, bonuses, and investor decisions.

### “A related-party sale is invalid.”

A related-party sale can be real. The issue is independence, pricing, funding, disclosure, and end demand. A subsidiary may buy inventory for a genuine retail network; a shell with no operations that is funded by the seller is a different risk.

### “A clean audit opinion proves every sale.”

An audit provides reasonable assurance, not a guarantee that every fraud is found. Collusion, false documents, management override, and transactions outside normal systems can defeat ordinary procedures. That is why an audit opinion and a forensic investigation answer different questions.

### “The biggest red flag is one strange invoice.”

One invoice can be an error. The stronger signal is a pattern across timing, customers, related parties, cash collection, returns, journal entries, and operational data. The more independent traces disagree, the more important the explanation becomes.

## How it shows up in real markets

### Reliant Resources: gross scale as a misleading signal

In an SEC administrative proceeding concerning Reliant Resources, the Commission said the company's 2001 annual report reported $36.5 billion of gross revenue, with more than $3.8 billion resulting from round-trip trades. The SEC also said the trades inflated revenue and expenses by 17.7% in 1999, 5.3% in 2000, and 10.6% in 2001, and that Reliant published inflated trading volumes. These figures come from the SEC's account of the amended annual report and enforcement findings.

The practical lesson is to ask what “revenue” measures in a trading business. A power marketer can have legitimate gross flows, but a pre-arranged same-volume, same-price exchange with no delivery is not equivalent to finding a new customer or taking market risk. Revenue growth should be read alongside net trading revenue, realized margin, counterparty concentration, and delivery evidence.

### CMS Energy: a volume story that did not change earnings

The SEC's 2003 order concerning CMS Energy described round-trip energy transactions in 2000 and 2001. It said the transactions overstated revenue and expenses by $5.2 billion over the relevant period: $1.0 billion, or 10% of revenue, in 2000 and $4.2 billion, or 36% of revenue, for the first three quarters of 2001. The SEC said the activity also overstated reported energy-trading volume by 78% in 2000 and 72% in 2001, while not affecting net earnings.

This is a useful caution for beginners. “No change in net income” does not mean “no accounting problem.” Investors use revenue and volume to infer scale, market position, and future earnings power. A business can mislead through the numerator and denominator of a growth story even when the final profit line is unchanged.

### Enron: when a sale behaves like a loan

The SEC's 2002 materials about Enron and its intermediaries alleged that the Nigerian barge transaction generated approximately $12 million of 1999 earnings even though the risk of ownership did not genuinely leave Enron. The SEC's 2003 complaint concerning Merrill Lynch alleged a separate sham energy transaction that enabled Enron to report $50 million of income in 1999 and described a $17 million fee paid for participation in an essentially risk-free arrangement.

These allegations show why side agreements matter. A sale with a guaranteed repurchase, a fixed return to the buyer, or a promise to remove the buyer's exposure may be financing in economic substance. The accounting label “sale” cannot erase the retained risk.

### Luckin Coffee: operational data and related-party schemes

The SEC's December 2020 litigation release alleged more than $300 million of fabricated retail sales from April 2019 through January 2020, more than $190 million of inflated expenses, and altered accounting and bank records. It also described alleged revenue overstatements of approximately 28% and 45% for two 2019 reporting periods and a $180 million settlement penalty.

The lesson is to triangulate reported sales against the system that should have produced them. Retail revenue should have a relationship with orders, stores, payments, coupons, delivery records, customer accounts, and inventory. When a company allegedly fabricates both operational records and financial records, a single confirmation or a single database extract is not enough.

## When this matters to you

This topic matters whenever you compare businesses using revenue growth, gross merchandise volume, market share, customer counts, or “record sales.” Those numbers are useful only when you know what they include and how they become cash.

For a public-company reader, start with the revenue-recognition note, receivables table, related-party note, contract liabilities, cash-flow statement, and subsequent-event disclosures. Then read the management discussion for explanations of changes in DSO, returns, allowances, customer concentration, and gross-versus-net presentation. **Days sales outstanding**, or DSO, is an estimate of how many days of sales are tied up in receivables; it is a diagnostic, not a verdict.

For an analyst, build a customer-level bridge where the data permits: opening A/R + revenue − cash collected − credits = closing A/R. Separate independent customers from affiliates. Track the same cohort after the reporting date. Review manual entries and reversals. Compare the company's stated business model with the physical and digital traces it should leave.

For an employee or supplier, the warning signs can be more immediate: pressure to backdate delivery, requests to issue invoices before shipment, unexplained side letters, customers who never contact the business directly, or a request to send cash through a related entity. Preserve records and use appropriate professional or legal channels; this article is educational, not individualized legal or financial advice.

The calm forensic habit is simple: **follow the value, not the label**. A revenue number is persuasive when an independent customer wanted something, received it, paid for it, and would do the same transaction without management's pressure. When those links break, the top line may be measuring paperwork rather than a business.

## Sources & further reading

- [SEC: Reliant Resources and Reliant Energy administrative proceeding](https://www.sec.gov/enforcement-litigation/administrative-proceedings/33-8232), including the $36.5 billion 2001 gross-revenue figure and over $3.8 billion of round-trip trades.
- [SEC: CMS Energy Corp. and Terry Woolley order](https://www.sec.gov/enforcement-litigation/administrative-proceedings/33-8403), describing the $1.0 billion 2000 and $4.2 billion first-three-quarters-2001 round-trip overstatements.
- [SEC: Andrew Fastow enforcement release](https://www.sec.gov/news/press/2002-143.htm), dated 2 October 2002, describing the alleged $12 million Nigerian barges sham sale and related Enron transactions.
- [SEC: Merrill Lynch and Enron complaint](https://www.sec.gov/litigation/complaints/comp18038.htm), dated 12 February 2003, describing the alleged $50 million 1999 energy-trade income and $12 million barge transaction.
- [SEC: Luckin Coffee litigation release](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-24987), dated 16 December 2020, describing the alleged fabricated sales, inflated expenses, revenue overstatements, and $180 million penalty.
- [Luckin Coffee special committee investigation release](https://www.sec.gov/Archives/edgar/data/1767582/000110465920079446/a20-23914_1ex99d1.htm), dated 1 July 2020, reporting the internal investigation's document review, interviews, and approximately RMB 2.12 billion 2019 net-revenue inflation finding.
- [Reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings) for the lines that fabricated revenue tries to inflate.
- [Reading the cash-flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) for the operating-cash bridge.
- [The footnotes and MD&A: where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) for related-party disclosures, estimates, and management explanations.
