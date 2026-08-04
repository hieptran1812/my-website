---
title: "Related-party transactions and self-dealing: when insiders move value in circles"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "A forensic guide to finding insider, affiliate, and controlled-entity transactions that transfer value—or manufacture it—inside the notes and statements."
tags: ["forensic-accounting", "related-party-transactions", "self-dealing", "earnings-quality", "financial-statements", "fraud-detection", "corporate-governance", "adelphia"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Related-party transactions are not fraud by definition; they are a map of where ordinary market discipline may be weakest. The forensic question is whether an insider-controlled counterparty moved independent cash and value, on market terms, with independent approval—or whether the transaction mainly moved a number between ledgers.
>
> - Search for the relationship before you judge the transaction: directors, executives, families, affiliates, joint ventures, trusts, and entities that share control can all matter.
> - Test every important deal across four dimensions: who benefited, what was exchanged, whether cash really moved from outside the control circle, and whether the reported amount is recoverable at a market value.
> - The highest-risk pattern is a cluster: insider financing, long-dated receivables, guarantees, non-cash “sales,” unusual equity placements, and vague note disclosure.
> - The SEC’s Adelphia complaint said that by the end of 2001, $2,283,416,421 of co-borrowing debt had been excluded from Adelphia’s books and placed on Rigas-entity books; the allegation was not that the debt vanished, but that its presentation concealed who remained exposed.
> - A related-party note is a starting point. The real answer comes from reconciling the note to the balance sheet, cash-flow statement, debt agreements, equity roll-forward, and subsequent collections.

If the chief executive sells a building to the company, the invoice may look ordinary. The buyer may record an asset, the executive may receive cash, and the contract may have been approved by the board. Yet the transaction is not ordinary in the way a purchase from an unrelated seller is ordinary. The person who negotiated the deal may be the person who benefits from it. The company may have paid more than a market buyer would pay. The board may have had less information than the seller. A receivable may be collectible only because the company keeps financing the buyer.

That is why related-party activity is such a reliable fraud marker. It is not reliable because every affiliate deal is dishonest. It is reliable because the transaction points to a place where the normal external checks—competitive pricing, an independent customer, an outside lender, and an arm’s-length negotiation—may be missing. A weak control environment leaves fingerprints in the places where value crosses the boundary between the public company and the people who control it.

The picture below is the mental model. A deal begins as a real or alleged economic exchange, passes through a control relationship, and becomes a journal entry that investors must interpret. The red path is not a conclusion of fraud; it is the route where an internal transfer can be mistaken for independent value.

![A related-party transaction can move cash or credit from a public company to an insider-controlled entity, then return as a flattering accounting signal.](/imgs/blogs/related-party-transactions-and-self-dealing-1.webp)

The rest of this article builds a practical method: define the relationship, understand the journal entry, read the note, reconstruct the cash and legal exposure, and only then decide whether the reported value is credible. The examples use round, clearly illustrative numbers unless a named source and date are given.

## Foundations: the building blocks

### What “related party” means in plain English

A related party is a person or entity whose connection to the reporting company can influence a transaction. The connection may be direct—such as a director owning the counterparty—or indirect, such as a family trust owning a chain of limited partnerships that controls the counterparty. It may arise through **control**, where one party can direct another’s relevant decisions; **significant influence**, where a party can participate in decisions without controlling them; or **key management personnel**, whose position gives them power over the company’s resources.

The exact accounting definition depends on the reporting framework and jurisdiction. The practical search universe is broader than a single line in a related-party note. It includes:

- directors, executives, founders, and their close family members;
- companies controlled by those people or by the reporting company;
- associates and joint ventures where influence is significant but not controlling;
- entities sharing a parent, owner, trustee, or management team;
- lenders, vendors, customers, landlords, and consultants whose ownership is linked to insiders; and
- special-purpose vehicles, partnerships, and trusts that hold assets or debt outside the obvious corporate chart.

![A reporting company’s related-party universe branches through insiders, families, controlled companies, joint ventures, and special-purpose vehicles.](/imgs/blogs/related-party-transactions-and-self-dealing-2.webp)

The word “related” describes a connection, not an accusation. A public company may buy from an affiliate because the affiliate owns a specialized factory. It may lend to an employee relocation program. A parent may guarantee a subsidiary’s bank debt. These can be commercially sensible. The question is whether the connection changes the price, terms, risk allocation, approval process, or quality of disclosure.

### The four economic questions

Every related-party transaction can be reduced to four questions:

1. **Who benefited?** Identify the cash recipient, the party receiving an asset, the party whose debt was guaranteed, and the party whose loss was avoided.
2. **What changed outside the ledger?** Look for a new customer, independent cash, a delivered asset, a discharged liability, or a service that the company actually consumed.
3. **Would an independent counterparty accept the terms?** Compare price, maturity, collateral, interest, return rights, warranties, and cancellation provisions with external deals.
4. **Can an outsider understand it from the filing?** A reader should be able to identify the relationship, transaction type, amount, balance due, terms, and the accounting effect.

The first two questions address substance. The third addresses market discipline. The fourth addresses disclosure. A transaction can fail any one of them without proving fraud, but failing all four is a serious signal.

### Statement vocabulary: value, not labels

**Revenue** is the income statement amount recognized for goods or services transferred during a reporting period. A related-party sale can produce revenue even if the customer is not economically independent. Revenue therefore needs a customer test and a collection test.

**Accounts receivable** is an asset representing amounts owed by customers or other counterparties. A receivable from an affiliate is not equivalent to cash. Its value depends on the affiliate’s ability and willingness to pay without fresh support from the reporting company.

**Payables and loans receivable** show financing relationships. A loan to an insider-controlled entity may be recorded as an asset, but the asset can be impaired if the borrower lacks cash or if repayment depends on another related-party transfer.

**Guarantees and co-borrowing obligations** are legal exposures that may not look like ordinary debt on the face of the balance sheet. If two entities are jointly liable, moving the balance to one entity’s ledger does not necessarily release the other entity from the lender’s claim.

**Equity** represents the residual interest after liabilities. An equity placement to a related party can bring real cash and strengthen the company. It can also be non-cash, overvalued, or structured to make debt look like equity. Read the cash-flow statement and the equity roll-forward together.

**Consolidation** combines a parent and controlled subsidiaries as if they were one economic group, eliminating internal sales, receivables, payables, and profits. A controlled entity can still be a disclosure issue because a reader needs to understand governance and exposure; an entity outside consolidation can be where obligations and assets are parked.

### A transaction is a bundle of journal entries

The contract is only the beginning. Follow the accounting entry through the statements. Suppose a company sells inventory for an illustrative $100 to an affiliate for cash, and the inventory cost the company $60:

```
Dr Cash                         $100
    Cr Revenue                              $100

Dr Cost of goods sold            $60
    Cr Inventory                             $60
```

On its own, this looks like a normal sale. The forensic questions are whether the affiliate paid from outside funds, whether it received and can use the inventory, whether the price is comparable, and whether the sale is later reversed. If the $100 is a loan from the company, the cash is not independent customer demand. If the affiliate returns the goods after period end, the revenue may not represent a completed sale. If the company retains the risks and rewards, the legal invoice may not tell the whole economic story.

#### Worked example: a sale that is really a receivable

Suppose Company P records a $100 sale to an insider-controlled distributor on December 31. The distributor promises to pay in 180 days. P’s cost is $70.

The initial entries are:

```
Dr Accounts receivable           $100
    Cr Revenue                              $100

Dr Cost of goods sold             $70
    Cr Inventory                             $70
```

The income statement reports $100 of revenue, $70 of cost of goods sold, and $30 of gross profit. The balance sheet reports $100 of receivables and $30 of increased equity before tax. The cash-flow statement, however, reports no customer cash: the $100 sale is a non-cash working-capital increase.

If the distributor pays $100 in the next period using a bank loan guaranteed by P, cash collection is evidence that money moved, but not strong evidence of independent demand. If the distributor cannot pay and P records a $40 allowance, the original $30 gross profit may be partly or entirely illusory after credit loss. A $100 invoice can therefore create a $30 profit signal before the company knows whether it has transferred useful value to a solvent customer.

**Intuition:** a related-party sale is only as strong as the independent cash, delivery, and collectibility behind the receivable.

## 1. Why insiders can move value more easily than outsiders

In an arm’s-length market, the seller wants a high price and the buyer wants a low price. Each side has a reason to challenge the other’s assumptions. A board approving a purchase from an outside vendor can compare bids. A bank lending to an unrelated borrower can refuse the loan. A customer can walk away.

Inside a control circle, those frictions can weaken. The executive may control the company’s procurement staff, the affiliate’s pricing, the board agenda, and the disclosure language. The parties can agree to a price that transfers wealth without looking like a dividend or compensation. They can set a maturity that will never be tested in an independent market. They can call a distribution “consulting fees,” a personal loan “accounts receivable,” or a debt assumption “equity consideration.” The labels differ; the forensic question is the same: who bore the risk and who received the benefit?

![An ordinary trade has independent price and approval checks; self-dealing risk rises when the counterparty, price, and beneficiary sit inside the same control circle.](/imgs/blogs/related-party-transactions-and-self-dealing-3.webp)

### The agency problem

Public shareholders are the owners, but managers control day-to-day decisions. This separation creates an **agency problem**: the agent can take actions that benefit the agent while appearing to benefit the principal. A related-party deal is a visible place where that problem can become measurable.

There are two common forms:

- **Tunneling:** value leaves the public company for an insider or a private affiliate through a low-priced asset sale, an inflated purchase, an unsecured loan, or a guarantee.
- **Propping:** an insider or affiliate temporarily supports the public company so that weak performance looks stronger—for example by buying inventory it does not need, prepaying for a service that will not be delivered, or extending a loan just before a reporting date.

Tunneling damages assets and future cash flow. Propping can be just as dangerous because it creates a performance signal that may reverse when the support stops. A transaction can be both: the public company may report a short-term sale while the insider receives favorable financing or a valuable security.

### The circularity test

Ask whether the company is both the source and the destination of the counterparty’s funding. Trace bank statements where available, but start with public clues:

- a customer receivable rises while operating cash flow remains weak;
- the same affiliate appears as both customer and vendor;
- a related party receives a loan, then buys the company’s stock or inventory;
- “cash received” is followed by a similarly sized loan, guarantee, or advance;
- the counterparty’s accounts show a payable to the company but little external revenue; or
- the deal is concentrated near a reporting date and reverses soon afterward.

Circularity does not require the exact same dollars to travel in a perfect loop. A lender may fund the affiliate, the affiliate may pay the company, and the company may guarantee the lender. The economic risk can still remain with the public company even though cash briefly touched the affiliate’s account.

#### Worked example: the circular customer

Assume P sells $1,000 of equipment to Affiliate A and records revenue. A pays with a $1,000 loan from P made one day earlier. This is illustrative, not a real-company amount.

P first records the loan:

```
Dr Loan receivable—Affiliate A   $1,000
    Cr Cash                                  $1,000
```

P then records the sale:

```
Dr Cash                           $1,000
    Cr Revenue                              $1,000
```

At the end, cash is back where it started. P has a $1,000 loan receivable and $1,000 of revenue. If equipment cost $600, it also reports $400 of gross profit, even though the funding source was P itself.

The cash-flow statement may show a $1,000 investing outflow for the loan and a $1,000 operating inflow from the customer. A casual reader sees operating cash. A forensic reader nets the economic story: P financed the supposed customer and still holds the credit risk. The next test is whether A has independent operating cash to repay the loan.

**Intuition:** cash can move twice and still represent one economic source—especially when the seller finances its own customer.

## 2. How to find the transactions in the notes

The related-party note is often the most efficient starting point because it gives the company’s own vocabulary. Read it as a data table, not a paragraph. Extract five fields for every named relationship:

| Field | What to capture | Why it matters |
| --- | --- | --- |
| Relationship | Owner, family, director, parent, subsidiary, affiliate, JV | Determines the possible conflict and consolidation boundary |
| Transaction | Sale, purchase, loan, guarantee, lease, fee, asset transfer, equity issue | Identifies the value pathway |
| Period amount | Revenue, expense, interest, purchase price, repayment | Shows the period’s income or cash signal |
| Closing balance | Receivable, payable, loan, guarantee, commitment | Shows what remains exposed after the deal |
| Terms and approval | Maturity, collateral, interest, price, board process | Tests market terms and governance |

Companies do not always use the same label each year. A “loan” may become an “advance,” an “other receivable,” or a “trade balance.” A family partnership may appear under an abbreviated legal name. A controlled entity may be described as “an entity in which certain officers have an interest,” which is technically informative but operationally vague.

### Search terms that reveal euphemisms

Search the filing for `related party`, `affiliate`, `common control`, `director`, `executive officer`, `family`, `loan`, `advance`, `receivable`, `guarantee`, `co-borrower`, `jointly and severally`, `non-cash`, `equity placement`, `commitment`, `indemnification`, `unconsolidated`, and `other`. Then search for names, initials, addresses, and recurring counterparties.

The word “other” deserves attention because it can be a hiding place for aggregation. A company may be allowed to aggregate immaterial items, but a large balance or unusual term should not disappear into a broad category. Compare the related-party note with the “other receivables” line and the debt footnote. If the note says a related party owes $20 million but the balance sheet has $200 million of other receivables, identify the remaining $180 million rather than assuming it is unrelated.

### Read the note against the ownership note

The ownership or governance section can reveal relationships the related-party note names only partially. Look for:

- voting agreements and dual-class shares;
- family members in management and on the board;
- significant shareholders with “entities controlled by” them;
- entities with the same registered office, directors, or legal advisers;
- joint ventures with unusual rights, side letters, or put options; and
- changes in control during the year.

The key is the chain. “Company A sold to Company B” is not enough. Build `A → B → owner → family trust → lender` until you know who can benefit and who can force a decision.

### The note is a map, not a valuation

Disclosure does not make a transaction fair. A company can disclose a $50 million insider loan and still overstate the loan’s value. The note tells you where to investigate; the balance sheet tells you the carrying amount; the cash-flow statement tells you whether cash moved; external sources tell you whether the terms resemble a market.

![The note is the map, while the balance sheet, debt, equity, and cash-flow statements test whether related-party value has economic weight.](/imgs/blogs/related-party-transactions-and-self-dealing-4.webp)

#### Worked example: reading a note and testing the closing balance

Imagine an illustrative note says: “During the year, the company sold services of $500 to an affiliate. At year-end, the affiliate owed $400, unsecured, interest-free, and payable on demand.”

The income statement shows $500 of related-party revenue. The balance sheet shows a $400 receivable. The cash-flow statement’s operating reconciliation shows a $400 increase in receivables, so only $100 was collected during the year.

Now compare the economics with an outside customer. A market customer paying for services would normally have a stated credit term, a collection history, and a late-payment consequence. An unsecured, interest-free, on-demand balance from a controlled affiliate has no obvious commercial reason to exist. The company may have chosen the invoice label to make a funding transfer look like revenue.

The correct response is not to erase the $500 automatically. Request the contract, invoices, work product, bank receipts, board approval, and subsequent collections. Recalculate the expected credit loss. If $300 is later written off, the forensic adjustment is not simply “remove related-party revenue”; it may be a $300 credit-loss expense, a lower receivable, and a question about whether the original service transfer met the revenue criteria.

**Intuition:** the note’s amount is a lead; terms, collections, and recoverability determine whether the lead is economically credible.

## 3. The four tests: independence, cash, market terms, governance

No single red flag proves self-dealing. A strong first-pass screen combines four tests. Think of them as four independent witnesses. If all four tell the same story, the risk is much higher than if only one is unusual.

### Test one: counterparty independence

Find the beneficial owner, not only the invoice name. A genuine outside customer should have its own business purpose, employees or operating capacity appropriate to the transaction, independent financing, and a history that makes the purchase plausible. A special-purpose entity may be legitimate, but its thin capitalization and shared directors mean you need stronger evidence.

Useful comparisons include the counterparty’s public filings, property records, litigation history, credit information, import or export records, and the company’s own prior disclosures. A small affiliate buying a large volume is not automatically fake; it is a mismatch to explain.

### Test two: cash settlement

Trace the cash from the counterparty’s bank, if access is available. In public data, reconcile collections to receivables and inspect whether the company also reports loans, advances, guarantees, or deposits with that same party. Cash from an unrelated bank is stronger evidence than a journal entry. Cash funded by the seller, a parent, or a circular affiliate is weaker.

Distinguish **gross cash flow** from **net economic cash flow**. A company can report $10 million collected from an affiliate and $10 million lent to that affiliate. Gross operating cash may look healthy while net exposure is unchanged. The statement classification matters, but the balance-sheet risk matters more.

### Test three: market terms

Compare the price and terms with outside transactions. For a loan, compare interest rate, maturity, collateral, covenants, seniority, and repayment history. For a sale, compare volume discounts, return rights, warranties, delivery terms, and payment timing. For an asset transfer, compare independent valuations and recent comparable sales.

“Below market” and “above market” are not always suspicious. A strategic relationship can justify a difference. The problem is an unexplained difference that benefits an insider, especially when the company reports no compensation or distribution elsewhere.

### Test four: governance and disclosure

Was the interested director excluded from the decision? Did an independent committee negotiate? Was an independent valuation obtained? Were shareholders asked to approve the deal? Does the note disclose the relationship, amount, balance, terms, and accounting effect clearly enough for a reader to understand the risk?

Governance is not a substitute for economics. A conflicted transaction approved by independent directors can still be overpriced. But a transaction in which the beneficiary votes, negotiates, and signs the disclosure has a much weaker control story.

#### Worked example: a below-market insider loan

Suppose a public company lends $2,000 to an executive-controlled entity for two years at 0% interest, unsecured. Assume an illustrative market rate for a comparable independent borrower would be 10% annually, with annual compounding.

![A practical first-pass matrix asks whether the counterparty, settlement, pricing, and governance are independently supported or show warning signs.](/imgs/blogs/related-party-transactions-and-self-dealing-5.webp)

The simple two-year interest comparison is:

\[
\text{Foregone interest} = \$2{,}000 \times 10\% \times 2 = \$400.
\]

Using annual compounding as an explanatory abstraction, the market-value difference is approximately:

\[
\text{Market repayment value} = \frac{\$2{,}000}{(1+0.10)^2} \approx \$1{,}653.
\]

![A below-market insider loan transfers value through favorable terms even when the principal balance is fully recorded.](/imgs/blogs/related-party-transactions-and-self-dealing-9.webp)

The exact accounting treatment depends on the applicable framework and facts. The forensic point is simpler: the company has transferred financing value. If the borrower is also buying the company’s assets, the $400 is part of the transaction economics even if no line is called “compensation.” If the loan is never repaid, the credit loss is another transfer.

The best follow-up is to inspect subsequent cash, collateral, and the borrower’s external income. A 0% loan to an employee with a documented relocation purpose is different from a 0% loan to a private entity that has no independent business and is owned by the employee’s family.

**Intuition:** price the hidden benefit, not just the stated principal; favorable terms are value even when the cash balance does not move twice.

## 4. The accounting mechanics of value transfer

Related-party self-dealing often works because several entries each look plausible in isolation. The forensic task is to see the sequence.

### Asset sales and purchases

An insider can benefit when the company buys an asset above market value. Suppose the company pays an illustrative $1,200 for an asset worth $800. The entry may be:

```
Dr Property or investment          $1,200
    Cr Cash                                  $1,200
```

The asset is overstated by $400 at acquisition. If management later depreciates it over four years, the annual depreciation is $300 instead of $200, so reported profit is understated by $100 per year relative to a fair $800 purchase—but the company has already transferred $400 of value to the seller. If the asset is not depreciated, the overstatement remains hidden until impairment or sale.

The opposite pattern is an insider buying an asset below market value. The company may record a gain or a loss that does not reflect the value transferred. Compare cash received with independent appraisals, tax records, and the asset’s later resale.

### Fees and services

Consulting, management, licensing, and “brand” fees are particularly difficult because the service can be intangible. Ask what was delivered, who performed it, how many hours or units were provided, and whether the company would buy the same service from an outside vendor. Repeated rounded invoices, identical descriptions, and fees that grow with revenue without a measurable service are weak evidence.

If an affiliate charges $300 for services and the company pays cash, the entry is ordinary:

```
Dr Operating expense                $300
    Cr Cash                                    $300
```

But if the affiliate immediately returns $300 as an “investment” in the company, the economic story is not a normal expense. The company may have shifted cash out and back to create a fee, a capital contribution, or a tax result. Follow both sides.

### Loans, advances, and guarantees

An insider loan is an asset until it is not. The company must assess collectibility, but a related party may have an incentive to delay default or refinance indefinitely. A guarantee can be more important than a recorded receivable: if the affiliate borrows from a bank and the public company guarantees the loan, the public company may bear the loss even when no cash has yet left.

Read debt covenants for cross-defaults, subsidiary guarantees, joint liability, and restrictions on related-party payments. A debt note may list the borrowing entity but not make the group exposure intuitive. Ask the lender’s legal claim: if the affiliate fails, can the lender pursue the public company?

### Equity placements and non-cash consideration

An equity issue to a related party can strengthen the balance sheet only if the company receives real consideration. If the company issues shares for cash, the basic entry is:

```
Dr Cash                         $1,000
    Cr Share capital / APIC               $1,000
```

If it issues shares in exchange for the affiliate “assuming” a liability that the company remains legally responsible for, the entry can overstate equity and understate debt. The test is whether the liability was extinguished in law and substance, not whether the ledger was reclassified.

#### Worked example: debt “assumed” by a controlled entity

Suppose Public Co. and Family Co. are jointly liable for an illustrative $5,000 bank facility. Public Co. records $3,000 of the balance and Family Co. records $2,000. Public Co. then issues $1,000 of shares to Family Co. and records that Family Co. assumed $1,000 of Public Co.’s debt.

The public-company entry might be presented as:

```
Dr Debt                         $1,000
    Cr Equity                              $1,000
```

But if the bank agreement still makes Public Co. jointly and severally liable, Public Co.’s legal exposure remains $5,000. The entry has reduced reported debt to $2,000 and increased equity, but it has not changed the lender’s claim. If Family Co. receives the shares without paying cash, the equity is not independent new capital. The company has swapped a presentation for an obligation.

An analyst would reconstruct both numbers:

| Measure | Reported after entry | Reconstructed exposure |
| --- | ---: | ---: |
| Debt on Public Co. ledger | $2,000 | $5,000 legal co-borrowing |
| New cash from share issue | $0 | $0 |
| Equity increase | $1,000 | Requires substance test |
| Remaining lender exposure | Not obvious | $5,000, subject to agreement |

The $5,000, $1,000, and $2,000 figures are illustrative. In a real case, obtain the credit agreement, lender confirmation, guarantees, board minutes, and the stock issuance documents.

**Intuition:** a debt transfer is real only when the lender’s rights change, not when two related ledgers use different labels.

![A co-borrowing obligation can be shifted between an affiliate ledger and the public company’s report even though the lender’s legal claim remains.](/imgs/blogs/related-party-transactions-and-self-dealing-6.webp)

## 5. Why related-party transactions are a high-reliability fraud marker

Fraud markers are not proof. They are places where the expected relationship between economic activity and reported numbers is more likely to break. Related-party transactions are unusually informative for five reasons.

### The counterparty is not an independent witness

If the customer, supplier, lender, and owner are connected, confirmations can become circular. Management may be able to persuade the other entity to confirm a balance that the other entity cannot pay. A confirmation proves that two ledgers agree; it does not by itself prove the balance is collectible or arm’s length.

### The transaction can hide a distribution

Public companies distribute value openly through dividends, compensation, share repurchases, or acquisitions. A self-dealing transaction can disguise the same transfer as a business expense, investment, loan, or asset purchase. The disguise matters because each label changes how investors evaluate earnings, capital, and governance.

### Disclosure is often fragmented

The relationship may appear in a governance note, the transaction in an expense note, the balance in receivables, the guarantee in debt, and the cash in an investing line. A reader who checks only the related-party note can miss the total exposure. Fragmentation is normal in complex groups, but it creates an opportunity to understate the whole.

### The numbers can be made to satisfy a target

If management needs a leverage ratio below a covenant threshold, shifting debt to an affiliate may make the reported ratio pass. If it needs revenue growth, selling to an affiliate can create the top line. If it needs an equity cushion, a non-cash placement can look like capital. Related-party mechanics are therefore useful when the company has a specific reporting target.

### Reversal risk is asymmetric

The benefit appears now; the cost appears later. A receivable is recognized at the sale date, but the write-off arrives after the reporting period. A guarantee is disclosed today but paid only after the affiliate fails. A debt reclassification can help the current covenant but force a restatement when the legal exposure is discovered. The timing makes the transaction valuable as a short-term reporting tool.

### A reliability rule

Treat the marker as strongest when at least three elements co-occur:

1. a relationship that gives one side control or influence;
2. terms that differ from outside transactions;
3. cash or settlement that is circular, delayed, or non-cash;
4. a balance that grows faster than the underlying business; and
5. disclosure that is vague, late, aggregated, or inconsistent with the statements.

> The question is not “Is this related?” The question is “Which independent check would exist if this party were a stranger—and where is the evidence that the check still operated?”

## 6. A repeatable forensic workflow

The following workflow is designed for an annual report, a filing set, or a data room. It is not a substitute for an audit, legal advice, or a full investigation. It is a way to spend attention where the accounting is most likely to conceal economic risk.

### Step one: build the relationship graph

Start with the related-party note, ownership, governance, subsidiaries, and joint ventures. Put every named person and entity into a simple table. Add control links, shared directors, family links, and common addresses. Mark whether each entity is consolidated.

Do not stop at the first layer. If an affiliate is owned by a family partnership, find the partnership’s partners. If a vendor is owned by a director’s spouse, record that. If a joint venture has a put option or guarantee, include the option holder and guarantor.

### Step two: classify the value pathway

For each transaction, label the pathway:

- cash from company to related party;
- cash from related party to company;
- asset from company to related party;
- asset from related party to company;
- debt or guarantee from company to related party;
- revenue or expense between the parties; or
- equity issued, repurchased, or pledged.

Then ask whether the pathway can move wealth without passing through profit. A guarantee can create a liability without an expense until default. A below-market asset sale can create a small accounting gain while transferring a large economic benefit. A share pledge can alter control without appearing as a new expense.

### Step three: tie the note to statements

Create a reconciliation with columns for note amount, balance-sheet line, cash-flow line, income-statement effect, and unresolved difference. Check opening balance plus additions less repayments, write-offs, and foreign-exchange movements against closing balance. If the math does not tie, the difference is not automatically fraud; it is an investigation item.

#### Worked example: reconciling an affiliate receivable

Assume an illustrative affiliate receivable has:

- opening balance: $800;
- new sales during the year: $1,200;
- cash collected: $900;
- credit note or return: $100; and
- closing balance reported: $1,000.

The roll-forward is:

\[
\text{Closing receivable} = \$800 + \$1{,}200 - \$900 - \$100 = \$1{,}000.
\]

That arithmetic ties. The next question is quality. If $700 of the $900 collection came from a loan made by the company to the affiliate, the apparent collection is not independent. If the $100 credit note was issued on January 2 and relates to goods returned on December 30, the year-end sale may be overstated. If the affiliate’s entire cash generation is a guarantee from the company, the $1,000 closing asset may be less valuable than its face amount.

**Intuition:** a clean roll-forward proves arithmetic, not economic substance; every large collection needs a funding-source test.

### Step four: compare margins and terms with outside business

Separate related-party revenue and gross margin if the filing provides enough data. Compare payment days, return rates, bad debts, discounts, and growth. A related-party customer that pays more slowly and earns a different margin may be a financing or transfer-pricing vehicle rather than a normal channel.

For expenses, compare the affiliate’s fee per unit or per employee with external vendors. For asset purchases, compare independent valuations and later impairment. For loans, compare yield and collateral with third-party exposures.

### Step five: inspect subsequent events

The period after year-end is a natural lie detector. Look for:

- collections and write-offs;
- returns, cancellations, and credit notes;
- refinancing or debt forgiveness;
- asset sales and impairment charges;
- changes in control or resignations;
- covenant breaches and waivers; and
- restatements or delayed filings.

The timing matters. A balance that is collected in ordinary course supports its existence, but it does not prove the original price was fair. A balance written off shortly after year-end challenges both collectibility and the judgment used at the reporting date.

### Step six: write the reconstruction in one sentence

A useful conclusion should be specific: “The company reported $X of revenue from Affiliate A, but $Y remained unpaid, $Z of the collection was funded by the company, and the independent price evidence is absent.” That is more useful than “related-party activity is suspicious.” Keep allegations attributed and separate from verified accounting mechanics.

### What evidence should change your mind?

Forensic work is not a hunt for confirmation. A good process states in advance what would lower the risk. Evidence that should make a related-party transaction more credible includes an independently negotiated contract, a second bidder or comparable price, board minutes showing the interested director did not participate, an appraisal that uses observable market inputs, delivery records, and cash collected from the counterparty’s own operating receipts. A profitable subsidiary with a long history of external customers is different from a newly formed entity that exists only to transact with the listed company.

Evidence that should raise the risk includes a side letter, a return right that is absent from the main contract, a payment made immediately after a loan from the public company, a guarantee that leaves the public company with the downside, or an unexplained change in counterparty name near year-end. So does a note that gives a period amount but not a closing balance, or a balance that appears in “other” assets but is omitted from the related-party table. These are not proof of wrongdoing; they are gaps between the economic question and the evidence.

Keep a distinction between three conclusions. “The transaction occurred” means documents and cash support an event. “The transaction was recorded correctly” means the accounting treatment follows the applicable framework. “The transaction was fair and independent” means the price, terms, and governance do not transfer an unexplained benefit. One conclusion does not imply the other two. A real payment can still be an unfair payment; a fair price can still be incorrectly classified; a correctly disclosed loan can still be unrecoverable.

That separation is especially important when communicating an allegation. Say “the SEC complaint alleged that the debt was omitted from Adelphia’s reported liabilities” rather than “Adelphia hid debt” when describing the complaint itself. Then separately state the mechanical point that a joint-liability agreement can leave a borrower exposed even when the ledger assigns the balance elsewhere. Precision protects both the reader and the analysis.

![The final reconciliation moves from the reported story to external cash, recoverability, legal exposure, and the corrected statement effect.](/imgs/blogs/related-party-transactions-and-self-dealing-8.webp)

## 7. The Adelphia/Rigas case: co-borrowing, family entities, and sham equity

Adelphia Communications Corporation is a useful case because the alleged mechanism was not a single secret invoice. It connected a public company, family-controlled entities, credit facilities, intercompany accounts, equity placements, and disclosure. The facts below are framed as allegations or findings of the cited U.S. authorities, not as a general claim that every family-controlled company behaves this way.

### The relationship structure

The Rigas family founded and controlled Adelphia. John Rigas was founder and chief executive; his sons held senior executive and board roles. The SEC complaint also described Rigas entities that owned or operated cable systems and were co-borrowers with Adelphia subsidiaries. The result was a group of public and private entities connected by family control and shared financing.

The control relationship was the first clue. A transaction between Adelphia and an ordinary bank would have been tested by the bank’s credit process. A transaction involving a family entity that shared borrowing facilities and personnel required a reader to understand which company borrowed, which company benefited, and which company remained liable.

### The co-borrowing mechanism

The SEC complaint alleged that from at least mid-1999 through the last quarter of 2001, Adelphia understated consolidated liabilities by up to $2.3 billion by failing to record a portion of credit-facility liabilities for which Adelphia was a co-borrower and jointly and severally liable. The complaint’s quarter-by-quarter schedule states that by the end of 2001, $2,283,416,421 had been excluded from Adelphia’s books and placed on the books of Rigas entities. These are the SEC’s allegations in its July 2002 complaint, not an independently recomputed figure in this article.

The economic idea is easy to miss if you read only the ledger. A lender’s contract can make two co-borrowers each responsible for the whole outstanding amount. An internal accounting decision can attribute a draw to the family entity, create an intercompany payable, and leave the public company’s reported debt lower. The lender’s legal claim does not become smaller merely because the company’s books use a different attribution.

The SEC complaint also alleged that Adelphia used a centralized cash-management system through which funds were disbursed according to the needs of Adelphia or Rigas entities and were accounted for through related-party payables and receivables. That structure is not automatically improper; centralized treasury systems are common. But it makes the source and destination of cash a central forensic question.

### The stock placement example

The complaint described a January 24, 2000 direct placement in which a $368 million drawdown was used to repay other debt and was attributed to a Rigas entity; Adelphia then issued Class B shares to a Rigas entity. The SEC alleged that the transaction helped conceal $368 million of Adelphia liabilities and that the shares were not paid for with independent cash as represented. The complaint also alleged that direct placements to Rigas entities represented approximately $1 billion as of December 31, 2001.

The accounting signal was equity. An investor could see shares issued and infer new capital. The forensic reconstruction asked a different question: what did Adelphia actually receive, and did the company’s legal debt disappear? If the answer was “an internal assumption of debt for which Adelphia remained liable,” the equity label overstated the strength of the balance sheet.

### The disclosure problem

The SEC complaint said Adelphia’s footnote disclosed that certain subsidiaries were co-borrowers with Rigas entities under facilities for borrowings up to $3,751,250,000, but did not explain that additional drawdowns recorded on Rigas-entity books were not included in the reported “total subsidiary debt.” It also said the relevant GAAP rule for related-party disclosure, FAS 57, required disclosure of the relationship, transaction nature, dollar amounts, and amounts due to or from the related party.

The lesson is not that a footnote must contain every operational detail in the first sentence. The lesson is that a disclosure can be technically shaped like a warning while still failing to communicate the full exposure. “Certain subsidiaries are co-borrowers” is materially different from “the public company is jointly and severally liable for amounts borrowed by family-controlled entities and those amounts are recorded outside the public-company debt total.”

### The aftermath and attribution

The U.S. Department of Justice’s 2002 performance report said John Rigas, two sons, and other executives were indicted on charges including wire fraud, securities fraud, bank fraud, and conspiracy; it also said the indictment charged the Rigas family with embezzling hundreds of millions of dollars of Adelphia funds and assets. A later DOJ tax release described allegations that $1.85 billion was diverted and entered as intercompany receivables and loans. Those figures and characterizations are reported allegations from the cited government releases, not amounts to generalize to all related-party activity.

The SEC later said Deloitte had issued an unqualified opinion on Adelphia’s 2000 financial statements despite knowledge or reason to know that Adelphia had failed to record or disclose $1.6 billion of debt, failed to disclose significant related-party transactions, and overstated stockholders’ equity by $375 million. That enforcement statement is a regulatory finding about the auditor’s conduct. It shows why a clean audit opinion does not end a forensic review: the question is what evidence was obtained, what judgments were challenged, and whether the audit team understood the control relationships.

![The Adelphia timeline shows the reported progression from omitted co-borrowing debt in 1999, through a $368 million 2000 placement and growing omitted exposure, to public acknowledgment on March 27, 2002.](/imgs/blogs/related-party-transactions-and-self-dealing-7.webp)

### What the case teaches

First, the relationship came before the number. Once public and private family entities shared executives, cash management, and credit facilities, a normal subsidiary-debt line needed a more careful reading.

Second, the most important risk was legal exposure, not only accounting presentation. The SEC alleged that Adelphia remained liable under co-borrowing facilities even when the debt was put on Rigas-entity books.

Third, equity can be an illusion if consideration is non-cash or does not eliminate the liability. A share issue is not automatically capital simply because the equity account increased.

Fourth, disclosure quality is about decision-useful meaning. A note can mention the correct words and still omit the fact an investor needs to understand the risk.

Finally, the cluster mattered. Family control, related-party balances, co-borrowing, internal cash management, stock placements, debt covenants, and disclosure ambiguity reinforced one another. A single family-owned vendor would not carry the same signal.

## 8. Common misconceptions

### “A related-party transaction is automatically fraudulent.”

No. Groups transact with subsidiaries, joint ventures, founders, and employees for legitimate operating reasons. The relationship changes the evidence threshold; it does not determine the conclusion. An affiliate sale supported by independent pricing, outside cash, delivery records, and independent approval can be economically sound.

### “If it was disclosed, shareholders accepted the risk.”

Disclosure is necessary, not magical. A vague description may not reveal the amount, terms, legal exposure, or beneficiary. A disclosed insider loan can still be impaired; a disclosed guarantee can still be large enough to threaten solvency.

### “Cash collection proves the sale was real.”

Cash collection proves that cash moved from one account to another. It does not prove the source was independent, the price was fair, or the transaction was not financed by the seller. Trace funding and look for offsetting loans, deposits, guarantees, or immediate reversals.

### “Only private companies use related parties to move value.”

Public companies can have more disclosure and more controls, but they also have more reporting targets: earnings, covenants, leverage, acquisitions, and share-price expectations. A public-company related-party note should be read as a control and incentive disclosure, not as a private-company curiosity.

### “A guarantee is not debt because no cash has left.”

A guarantee is a contingent exposure, not necessarily a current debt liability under every framework. Economically, however, it can transfer downside risk to the guarantor. If the affiliate fails, the guarantee can become a cash claim. Read the trigger conditions, collateral, maturity, and cross-default language.

### “The auditor would have caught it.”

An audit provides reasonable assurance under an audit framework; it is not a guarantee that every fraud will be found. Complex control relationships, management override, fabricated documents, and collusion can defeat ordinary procedures. An unqualified opinion is one piece of evidence, not a replacement for reading the notes and legal agreements.

### “A small affiliate balance cannot matter.”

Materiality is relative. A $5 million receivable can be immaterial to a large multinational but central to a thinly capitalized company. Also look at the direction and persistence: a small balance that grows every year, rolls forward without collection, or appears alongside guarantees can reveal a larger pattern.

## How it shows up in real markets

### Adelphia: the family balance sheet inside the public balance sheet

The Adelphia/Rigas episode is the central case in this article because the alleged scheme connected a public company to family entities through co-borrowing facilities. The SEC’s July 2002 complaint alleged that $2,283,416,421 of co-borrowing debt had been excluded from Adelphia’s books by the end of 2001. It also described a $368 million January 2000 transaction involving a drawdown, repayment of other debt, attribution to a Rigas entity, and a direct placement of Adelphia stock. The alleged effect was to make leverage look lower and equity look stronger while the family entities and Adelphia remained connected to the same financing system.

For a reader today, the transferable technique is to reconcile the borrower named in a note with the borrower named in the credit agreement. Then ask whether “assumed,” “attributed,” or “reclassified” changed the lender’s legal rights. If not, the accounting presentation may be hiding rather than reducing risk.

### A controlled subsidiary that sells to its parent

Consider a legitimate industrial group with a manufacturing subsidiary and a distribution subsidiary. Consolidated statements eliminate internal sales, so the group’s external revenue should reflect sales to customers outside the group. Separate-company statements can still show a large internal balance, and a minority shareholder in one subsidiary may care about the price and terms.

The red flags arise when the parent’s reported growth depends on sales to an unconsolidated affiliate, when the affiliate cannot pay without a parent loan, or when the transaction creates a receivable that is later written off. The correct comparison is not “affiliate versus no affiliate.” It is “external customer demand versus internally generated demand.”

### An insider real-estate purchase

Suppose a founder sells an office building to the public company at an appraisal value. The disclosure may be clean and the purchase may be approved. A forensic reader still checks whether the appraisal used comparable properties, whether the founder retained a lease or repurchase option, whether the company paid cash or issued shares, and whether the building later suffers impairment.

If the company paid with shares, calculate dilution and compare the fair value of the shares with the asset. If the founder receives a guarantee or a long-dated note, the seller may have transferred limited risk while extracting value. If the building is essential, the price may include strategic value; that explanation should be visible in the governance record.

### A vendor owned by a director’s family

A director’s family company may provide software, logistics, or construction services. The invoices can be real. The question is whether the price, staffing, and deliverables match the claimed service. Compare headcount, work orders, unit prices, and outside bids. Search for the same address, bank account, or personnel in other vendors.

The accounting effect may be an ordinary expense, but the governance effect may be compensation by another name. When a fee rises sharply while the service description stays generic, ask whether the transaction is a dividend, a bonus, or a transfer of assets from shareholders to the insider.

### A loan to a joint venture

A loan to a joint venture may support a strategic project. Analyze priority, collateral, interest, maturity, and the company’s share of the venture’s cash flows. A “loan” that is perpetually rolled, subordinated to every other lender, and repayable only after a project is sold may behave like equity.

The note should distinguish ordinary trade balances from financing. On the cash-flow statement, a loan may be investing cash flow while interest is operating cash flow. The labels can split the story across sections; the forensic reconstruction puts principal, interest, guarantees, and expected loss back together.

### A customer concentration hidden behind an affiliate

A company can reduce the apparent concentration of customer risk by selling through an affiliate or distributor. If the distributor is controlled by the same owner as the end customer, the group may be one economic buyer, not several independent customers. Search ownership, common addresses, shared guarantees, and payment sources.

This matters for revenue quality and working capital. A single buyer can negotiate terms that no diversified customer base could obtain. If that buyer fails, receivables, inventory returns, and revenue reversals can arrive together.

## When this matters to you

Related-party analysis matters whenever you are evaluating a company whose owners, executives, family, or affiliates have substantial influence. It is especially important for founder-led groups, conglomerates, companies with complex subsidiaries, businesses with large guarantees, and issuers that report rapid growth in receivables or non-cash transactions.

For a shareholder, the practical question is not whether the company has any related parties. It is whether related parties control the terms of transactions that determine earnings, cash flow, debt, or dilution. For a lender, the key question is which entity ultimately bears the loss. For an employee or supplier, it is whether a private affiliate can drain the public company’s resources before ordinary stakeholders are paid.

Use the method in order:

1. map the control relationships;
2. list every transaction and closing balance;
3. tie the list to all three statements and the debt note;
4. trace outside cash and subsequent settlement;
5. compare terms with independent transactions; and
6. state the reconstructed exposure in one plain sentence.

This is educational analysis, not individualized investment or legal advice. A red flag is a reason to investigate, not a verdict. Contested allegations should remain attributed to the source that made them, and a legitimate transaction should be allowed to survive scrutiny when the evidence supports it.

## Sources & further reading

- [SEC complaint in SEC v. Adelphia Communications Corporation and others](https://www.sec.gov/litigation/complaints/complr17627.htm), U.S. Securities and Exchange Commission, filed July 2002. Source for the alleged $2,283,416,421 omitted co-borrowing debt by the end of 2001, the $368 million January 2000 transaction, the approximate $1 billion of related stock placements as of December 31, 2001, and the described disclosure failures.
- [SEC order on Deloitte’s Adelphia audit](https://www.sec.gov/news/press/2005-65.htm), U.S. Securities and Exchange Commission, April 26, 2005. Source for the regulator’s findings concerning the 2000 audit, including the stated $1.6 billion debt omission, $375 million equity overstatement, and undisclosed related-party transactions.
- [Adelphia complaint press statement](https://www.justice.gov/archive/dag/speeches/2002/072402adelphiapressstmt.htm), U.S. Department of Justice, July 24, 2002. Source for the government’s attributed description of the alleged more-than-$2.28 billion concealed borrowing and alleged stock transactions.
- [U.S. Department of Justice 2002 performance report](https://www.justice.gov/archive/ag/annualreports/pr2002/Section02.htm), published 2002. Source for the dated indictment description and its characterization of alleged embezzlement.
- [DOJ tax release on the Adelphia indictment](https://www.justice.gov/archive/tax/usaopress/2005/txdv05100705.htm), October 7, 2005. Source for the attributed allegation concerning $1.85 billion of diverted funds and intercompany receivables and loans.
- [SEC enforcement release on related-party disclosure standards](https://www.sec.gov/litigation/admin/34-52537-o.pdf), U.S. Securities and Exchange Commission. Useful primary-source context for the disclosure dimensions described in the Adelphia enforcement record.
- For the broader statement-reading method, see [the footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried), [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue), and [the cash conversion cycle](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals).
