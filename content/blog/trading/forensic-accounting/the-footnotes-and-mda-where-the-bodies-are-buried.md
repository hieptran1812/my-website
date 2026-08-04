---
title: "The footnotes and MD&A: where the bodies are buried"
date: "2026-08-04"
publishDate: "2026-08-04"
description: "A forensic, beginner-friendly method for reading footnotes and MD&A to find accounting-policy changes, segment shifts, related parties, contingencies, and post-year-end surprises before they reach the face statements."
tags: ["forensic-accounting", "footnotes", "mda", "annual-reports", "accounting-policies", "segment-reporting", "related-parties", "contingent-liabilities", "financial-statement-analysis", "red-flags"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — The face of an annual report tells you what the company reported; the footnotes and MD&A tell you how much judgment, timing, concentration, and uncertainty sit behind it.
>
> - Start with accounting policies, then follow changes in estimates and the words management uses to explain them. A new method can move reported profit without moving cash.
> - Segment tables reveal where the consolidated number comes from, while related-party notes reveal whether the business is transacting with economically independent customers and suppliers.
> - Contingencies and subsequent events are not footnotes to the story. They are the part of the story that may not yet qualify for a line on the balance sheet.
> - Read MD&A as management's causal narrative, then test each cause against the cash-flow statement, segment data, debt maturities, and the notes.
> - The fast workflow is: **map the report, mark changes, reconcile words to numbers, search concentrations, read uncertainty, then build questions.**

The most dangerous sentence in an annual report is often not a lie. It is a sentence that is technically true while leaving out the context that would change your interpretation.

“Revenue increased” can be true because a company bought another company. “Margins improved” can be true because a loss-making segment was moved into discontinued operations. “Cash was sufficient” can be true on the reporting date while a large debt payment is due soon after it. “Transactions with related parties were conducted on terms similar to those with unrelated parties” can be true as a policy statement while the identity of the counterparty tells you that the apparent customer is also the chief executive's private investment vehicle.

The face statements are the compressed output. The notes are the decompressor. Management's Discussion and Analysis, usually called **MD&A**, is the management-written explanation of financial condition, changes in financial condition, and results of operations. The U.S. Securities and Exchange Commission describes MD&A as a way for investors to see the company “through the eyes of management,” while also asking for discussion of known trends, demands, commitments, events, and uncertainties ([SEC MD&A guidance](https://www.sec.gov/rules-regulations/2003/12/commission-guidance-regarding-managements-discussion-analysis-financial-condition-results-operations)). That makes MD&A valuable and dangerous at the same time: it contains the most useful context in the filing, written by the people with the strongest incentive to frame the context favorably.

The first figure is the mental model for this entire article. The face statements are the headline; the footnotes supply the definitions and exceptions; MD&A supplies the causal story; later events test whether that story survived contact with reality.

![An annual report is a layered evidence system: face statements show outcomes, footnotes define measurement and exposure, MD&A supplies management's causal story, and subsequent events test it.](/imgs/blogs/the-footnotes-and-mda-where-the-bodies-are-buried-1.webp)

The order matters. If you begin with MD&A, its narrative can anchor your thinking before you have inspected the measurements. If you begin with the notes alone, you can collect caveats without understanding which ones matter economically. A forensic read alternates between the two: claim, definition, number, cash consequence, and later evidence.

## Foundations: the building blocks of a filing

### What the face statements can and cannot tell you

An annual report normally presents a balance sheet, an income statement, a cash-flow statement, a statement of changes in equity, notes to those statements, and narrative sections such as business description, risk factors, and MD&A. The **face statements** are the prominent tables. They are not “the real statements” with the notes as optional commentary. Together, the tables and notes are the financial statements.

The income statement measures recognized revenue and expenses over a period. The balance sheet measures assets, liabilities, and equity at a date. The cash-flow statement explains changes in cash over the period. The notes explain the recognition rules, estimates, breakdowns, commitments, related parties, and events that make those numbers intelligible.

Think of a restaurant menu. The face statements are the prices and totals. The footnotes tell you whether “steak” includes a sauce charge, whether a “combo” includes a drink, and whether the restaurant is owned by the person supplying the meat. MD&A is the manager explaining why this month was different. You need all four to decide whether the business is improving.

#### Worked example: the same profit, different evidence

Suppose two companies each report \$100 of revenue, \$60 of cost of sales, and \$40 of operating profit. Company A collected \$95 from customers and carries \$5 of ordinary receivables. Company B collected \$20, carries \$80 of receivables, and says in its note that one customer represents most of the balance.

The face income statements are identical:

```
Revenue                         $100
Cost of sales                   (60)
Operating profit                 $40
```

The economic evidence is not identical. Company B's reported profit depends on converting a promise to pay into cash. That may be perfectly legitimate, but it creates a different question: who owes the \$80, and what happened after year-end? The footnote changes the next test from “is profit positive?” to “is the customer independent, solvent, and paying?”

The intuition: a face-statement total is an address, not a complete description of what is stored there.

### Policies, estimates, and judgments are different things

An **accounting policy** is the rule for recognizing and measuring a class of transactions: when revenue is recognized, how inventory is costed, whether development spending is expensed or capitalized where permitted, or how a lease is measured. An **accounting estimate** is an amount produced by applying a policy to uncertain facts: useful life, expected credit losses, warranty claims, percentage of completion, or fair value.

A **judgment** is the choice made when the rule requires interpretation: whether a contract contains one performance obligation or several, whether an acquisition gives control, or whether an entity is a principal or an agent. Policies are the recipe, estimates are uncertain ingredients, and judgments are the chef deciding which recipe applies.

The distinction matters because the earnings pattern differs. A policy change can re-time recognition. A change in estimate usually revises the amount or timing of future expense using the same underlying policy. A judgment change can move an arrangement between accounting models. The annual report should explain material changes and critical estimates; the SEC has specifically emphasized that critical policy disclosure should discuss the methodology, assumptions, effect on presentation, and reasonably likely changes ([SEC critical-accounting-policy release](https://www.sec.gov/rules-regulations/2001/12/accounting-policies-cautionary-advice-regarding-disclosure)).

![Accounting policy, estimate, and judgment changes travel through different channels: a policy changes the rule, an estimate changes an uncertain input, and a judgment changes which rule is applied.](/imgs/blogs/the-footnotes-and-mda-where-the-bodies-are-buried-2.webp)

#### Worked example: policy change versus cash

Imagine a software company that sells a \$120 annual contract paid upfront. Under a hypothetical old policy, it recognizes the whole \$120 at signing. Under a hypothetical revised policy, it recognizes \$10 per month as service is delivered. The customer still pays \$120 on day one.

```
Cash received at signing                         $120
Old-policy revenue at signing                   $120
Revised-policy revenue at signing                 $10
Revised-policy contract liability after signing  $110
```

Cash did not change. Reported revenue and profit did. The note should tell you whether the change is a correction, a new standard, a change in estimate, or a change in the business arrangement. If management describes a large revenue decline as “weak demand” but the policy note shows a recognition change, the narrative is incomplete even if every individual sentence is defensible.

The intuition: when profit moves but the related cash and contract balances do not, look for a rule, estimate, or classification change before you infer a change in demand.

### Materiality is not a magic percentage

**Materiality** asks whether omitting or misstating information could influence the decisions of a reasonable user. It is not a universal “five percent” switch. A small amount can be material if it changes a loss into a profit, breaks a debt covenant, hides a related party, or changes the apparent trend. A large amount can be less important if it is plainly described and economically unsurprising.

For a reader, this means the note heading is not a priority ranking. “Other” may contain the most important risk. “Immaterial” may describe an amount, not the relationship or pattern around it. Read for decision impact: would this fact change your view of recurring earnings, liquidity, control, or the people receiving money?

## 1. Read the report as a chain of claims

An annual report is easier to investigate when you turn prose into testable claims. MD&A says what changed. The income statement shows that it changed. A note explains the measurement. The cash-flow statement tells you whether cash followed. The balance sheet records the accumulated consequence. Subsequent events provide an out-of-sample test.

Use a five-column scratchpad:

| Management claim | Face-statement evidence | Note definition | Cash or balance-sheet test | Question left open |
| --- | --- | --- | --- | --- |
| Demand strengthened | Revenue and volume | Revenue policy and customer concentration | Receivables and operating cash | Did independent customers pay? |
| Margin expanded | Gross or operating margin | Cost allocation and segment basis | Inventory, payables, cash conversion | Was a cost reclassified? |
| Liquidity is strong | Cash and current assets | Restricted cash, covenants, maturities | Near-term debt and free cash flow | What is unavailable or due? |
| Diversification improved | Consolidated growth | Segment revenue and profit | Segment assets and capex | Which segment produced the change? |

Do not treat the table as a scoring model. It is a way to prevent the prose from floating free of the accounting. Each row should end in a document location or a question.

#### Worked example: turning “margin expansion” into a test

Suppose revenue rises from \$200 to \$240 and operating profit rises from \$20 to \$36. The reported operating margin changes from:

```
Old margin = $20 / $200 = 10%
New margin = $36 / $240 = 15%
```

That is a five-percentage-point improvement. Now suppose the note says \$8 of the new profit came from a one-time gain on selling equipment. Continuing operating profit is \$28, not \$36:

```
Continuing operating margin = $28 / $240 = 11.67% (rounded)
```

The arithmetic does not prove manipulation. It tells you that “margin expanded to 15%” describes a reported result that is not the same as the recurring operating engine. The next test is whether MD&A identifies the gain and whether the cash-flow statement places proceeds from the equipment sale in investing cash flow.

The intuition: calculate the metric twice—once as reported and once after the note explains unusual items.

![A forensic reading loop connects MD&A claims to statement totals, note definitions, cash-flow tests, and subsequent-event evidence before a conclusion is formed.](/imgs/blogs/the-footnotes-and-mda-where-the-bodies-are-buried-3.webp)

## 2. Accounting-policy changes: follow the bridge, not the headline

The first note to read carefully is the summary of significant accounting policies, followed by the note describing new standards, changes in policy, restatements, and errors. Look for verbs: “adopted,” “changed,” “reclassified,” “revised,” “restated,” “prospective,” and “retrospective.” They describe different paths through the numbers.

An adoption may affect the current year and comparatives. A retrospective presentation may rewrite prior periods for comparability. A prospective estimate change affects the current and future periods without rewriting history. A reclassification may leave total profit unchanged while moving an amount between revenue and another line, or between operating and non-operating categories.

The red flag is not “the company changed a policy.” Businesses and standards change. The red flag is a material change that appears late in the report, is explained only in boilerplate, coincides with a target-sensitive period, or makes the current period look better without a clear economic reason.

Ask four questions:

1. What economic event caused the change?
2. Did cash, contract balances, receivables, inventory, or debt change in the same direction?
3. Was the effect quantified by year and by line item?
4. Does MD&A explain the operational cause separately from the accounting effect?

#### Worked example: a reclassification that changes the story

Suppose a company reports \$50 of operating expense and \$10 of “other expense.” It later reclassifies \$8 from operating expense to other expense. Total expense remains \$60, but operating profit improves by \$8.

```
Before: operating expense $50; other expense $10; total $60
After:  operating expense $42; other expense $18; total $60
```

If management says “operating efficiency improved,” the statement-level improvement is real but the economic cost did not disappear. A reader must decide whether the reclassified item is genuinely outside operations or merely less visible in the chosen subtotal. In a covenant or valuation model that uses operating profit, the distinction can matter even when net income does not move.

The intuition: a subtotal is a lens. When the lens changes, preserve the old lens long enough to compare the business honestly.

### Critical estimates: the sensitivity paragraph is the treasure

Critical-estimate disclosures often mention impairment, revenue recognition, expected credit losses, inventory obsolescence, pensions, tax positions, warranties, and fair values. The useful part is not the list of topics; it is the sensitivity. Look for an assumption that is both uncertain and capable of moving a material line.

An impairment test compares the carrying amount of an asset or cash-generating unit with an estimate of recoverable value. A small change in forecast growth, margin, discount rate, or terminal value can change the answer. A provision for litigation depends on probability and estimated loss. A deferred tax asset depends on future taxable profit. These are not automatically wrong. They are locations where the reported number depends on management's model.

Read the estimate note alongside MD&A. If MD&A calls demand “stable” while the impairment note lowers forecast growth, the tension is information. If MD&A highlights adjusted earnings while the critical-estimate note explains a large capitalization judgment, the reconciliation matters more than the adjusted label.

![Critical estimates are pressure points: uncertain assumptions feed reported assets, profit, and liabilities, while sensitivity and later cash outcomes provide independent tests.](/imgs/blogs/the-footnotes-and-mda-where-the-bodies-are-buried-4.webp)

#### Worked example: impairment sensitivity without pretending to value a company

Suppose an illustrative asset has a carrying amount of \$100. Management's base case estimates future cash flows whose present value is \$108, so no impairment is recorded. A downside case reduces the present value to \$92.

```
Base headroom   = $108 - $100 = $8
Downside shortfall = $100 - $92 = $8 impairment
```

The \$8 is not a forecast of what will happen. It demonstrates why the note's assumptions matter: the same asset can appear fully supported under one scenario and require a write-down under another. The investigative question is whether the downside case is inconsistent with other disclosures—capacity closures, customer losses, falling segment revenue, or debt refinancing pressure.

The intuition: a “no impairment” conclusion is not the same as a low-risk asset; it may mean only that the base case has a little headroom.

## 3. Segment data: consolidated growth can hide a shrinking engine

A **segment** is a separately disclosed part of the business used by management to make operating decisions. Segment reporting can show revenue, profit or loss, assets, depreciation, capital spending, or other measures. The precise table varies, and the company may use a management view rather than a clean legal-entity view.

Start with three comparisons: segment growth versus consolidated growth, segment profit versus segment revenue, and segment cash demands versus segment profit. Then look at eliminations and “corporate” columns. A business can show healthy consolidated growth because a newly acquired segment was added, while its older core is flat. A segment can report profit while consuming cash because receivables and inventory absorb working capital.

Watch for changes in segment composition. A segment may be renamed, combined, divided, or moved into another category. Such changes can be reasonable, but they break the time series unless prior periods are recast. If comparatives are not comparable, management's growth percentages need a bridge.

#### Worked example: acquisition-led growth

Imagine a company with two segments. In year one, Core revenue is \$180 and NewCo revenue is \$20, for total revenue of \$200. In year two, Core revenue is still \$180 while NewCo revenue is \$80, for total revenue of \$260.

```
Consolidated growth = ($260 - $200) / $200 = 30%
Core growth         = ($180 - $180) / $180 = 0%
NewCo growth        = ($80 - $20) / $20 = 300%
```

The 30% group headline is correct. It is also incomplete. The business that existed before the acquisition did not grow. You would next read the acquisition note, purchase-price allocation, goodwill, customer concentration, and cash-flow statement. If NewCo's revenue is accompanied by large receivables and little cash collection, the growth deserves a different risk label.

The intuition: always decompose group growth into old business, acquired business, currency, and eliminations before calling it organic.

#### Worked example: profit without cash in a segment

Suppose a segment reports \$30 of profit. Its receivables increase by \$22, inventory increases by \$10, and payables increase by \$5. Ignoring other working-capital items, the cash effect is:

```
Profit                                   +$30
Receivables increase                     (22)
Inventory increase                       (10)
Payables increase                         +5
Approximate operating cash from these   +$3
```

The segment did not necessarily “fake” \$30 of profit. It earned accounting revenue while cash was tied up in customers and stock. But a reader should not value the segment as though \$30 immediately arrived as cash. The note and MD&A should explain whether the working-capital build is a seasonal investment, a deliberate growth choice, or evidence of weak collection.

![Segment analysis decomposes consolidated performance into core, acquired, and corporate components, then reconnects segment profit to working-capital cash demands.](/imgs/blogs/the-footnotes-and-mda-where-the-bodies-are-buried-5.webp)

## 4. Related parties: follow the relationship before the amount

A **related party** is a person or entity connected to the reporting company through control, significant influence, management, family relationships, or another defined relationship. The exact accounting definition depends on the reporting framework, but the forensic question is universal: would this transaction have happened on the same terms if the parties were economically independent?

Related-party transactions are not automatically improper. A controlled subsidiary may buy services from its parent. A founder may lease property to the company. A director may lend money during a crisis. The risk is that the counterparty can influence price, timing, collectability, guarantees, or disclosure.

Read the related-party note in four passes:

1. Identify the people and entities, not merely the transaction categories.
2. Separate sales, purchases, loans, guarantees, leases, and balances due.
3. Compare the terms—interest, maturity, collateral, pricing, and settlement—with ordinary transactions.
4. Trace the relationship into customers, suppliers, debtors, and subsequent events.

Pay attention to “amounts due from” and “amounts due to.” A company can report revenue to a related party and then carry the receivable for a long time. A supplier can fund the company indirectly by accepting delayed payment. A guarantee can create economic exposure without an immediate expense.

#### Worked example: revenue concentration plus a related-party receivable

Suppose reported revenue is \$100. A related-party customer accounts for \$30, and \$24 of that amount remains receivable at year-end. An unrelated customer produces \$70 of revenue and leaves \$7 receivable.

```
Related-party revenue concentration = $30 / $100 = 30%
Related-party collection ratio       = ($30 - $24) / $30 = 20%
Unrelated collection ratio           = ($70 - $7) / $70 = 90%
```

These ratios are illustrative, not an accusation. The pattern says the revenue-quality question is concentrated: most related-party revenue has not yet become cash, while the ordinary customer base paid more quickly. You would inspect the next reporting period, credit terms, returns, side agreements, and whether the counterparty has independent financing.

The intuition: the amount is only half the related-party risk; the identity and settlement behavior are the other half.

### Named case: Enron and the disclosure problem

Enron is a useful case because it shows why a footnote can be economically central even when the face statements look polished. SEC-hosted material discussing Enron describes special-purpose entities and related-party disclosure issues, including concerns about partnerships that kept liabilities off Enron's balance sheet and generated income for Enron ([SEC-hosted Enron discussion](https://www.sec.gov/comments/other/other-initiatives/otherinitiatives-72.pdf)). That document is not a generic lesson that every special-purpose entity is fraudulent; it is evidence of the specific disclosure and related-party issues that became part of the Enron story.

The reading lesson is procedural. When a company uses a complex structure, do not stop at the label “special purpose entity,” “joint venture,” or “nonconsolidated affiliate.” Ask who supplied capital, who bore losses, who received fees, who could force a transaction, and whether the company recognized gains with a counterparty that management could influence. Then reconcile the footnote's risk description to the debt, guarantees, cash flows, and related-party balances.

The point is not that a dramatic historical case lets us declare a current company guilty by analogy. The point is that the footnotes often contain the map of an exposure before the exposure is obvious in the face statements.

## 5. Contingencies: liabilities that wait for probability or measurement

A **contingency** is a possible obligation or gain whose outcome depends on a future event. Litigation, tax disputes, environmental remediation, product claims, guarantees, regulatory investigations, and purchase commitments can appear here. Accounting rules generally distinguish among recognized provisions, disclosed possible obligations, and matters that are not disclosed because they are remote or not material under the applicable framework.

The absence of a balance-sheet liability does not mean the absence of economic risk. It may mean that the amount cannot yet be measured reliably, the loss is not judged probable, or the disclosure threshold has not been met. Forensic reading therefore asks what the company says about probability, range, insurance, indemnification, timing, and precedent.

Look for language that changes over time: “not reasonably possible,” “reasonably possible,” “probable,” “unable to estimate,” “substantially all,” and “we believe.” Compare the current language with prior years. A new lawsuit is not necessarily a red flag. A repeatedly renewed statement that a material exposure cannot be estimated deserves a question about what information management actually has.

#### Worked example: a range is not a number you can ignore

Suppose a company discloses a lawsuit with a possible loss between \$10 and \$40, but says no provision is recorded because loss is not probable. The face balance sheet does not show the \$10 or \$40. A conservative reader can still show the exposure as a scenario:

```
Reported equity before scenario             $200
Equity if the low-end loss occurs            $190
Equity if the high-end loss occurs           $160
```

The scenario is not an accounting adjustment and should not be presented as one. It is a stress test. Then ask whether the company has cash, insurance, covenant headroom, and access to refinancing if a loss occurs. The note's range is valuable precisely because it prevents false precision.

The intuition: disclosed uncertainty belongs in your risk range even when it does not belong in the reported liabilities line.

![Contingencies sit outside the face statements until probability and measurement cross a recognition threshold; a forensic reader still carries the disclosed range into a liquidity stress test.](/imgs/blogs/the-footnotes-and-mda-where-the-bodies-are-buried-6.webp)

### Commitments are future cash before they are current expense

Read purchase commitments, leases, guarantees, debt maturities, letters of credit, and take-or-pay arrangements as a future cash calendar. A commitment may not be an expense yet, but it can compete with payroll, capex, dividends, and debt repayment. The SEC has noted the usefulness of contractual-obligation information for understanding future payments and supply-chain risk, while later disclosure reforms changed how some of those obligations are presented ([SEC Regulation S-K discussion](https://www.sec.gov/newsroom/speeches-statements/lee-crenshaw-statement-amendments-regulation-s-k)).

The workflow is simple: copy every material future payment into a timeline, then place expected operating cash beside it. If the report says liquidity is strong but most cash is restricted or future obligations cluster in a short period, the claim needs qualification.

## 6. Subsequent events: the page after the photograph

A balance sheet is measured at a date. A subsequent event occurs after that date but before the financial statements are issued or available for issue, depending on the framework. Some events provide evidence about conditions that already existed at year-end; others reflect new conditions and are disclosed rather than adjusted. A later refinancing, bankruptcy of a customer, factory fire, acquisition, lawsuit settlement, covenant breach, or dividend decision can change how you interpret the year-end picture.

Read the subsequent-events note last among the core notes, but do not treat it as an appendix. It is a reality check on assumptions made at year-end. A customer failing shortly after year-end may support a year-end credit-loss concern. A new fire may not change the old balance sheet, but it changes forward liquidity and business continuity.

#### Worked example: the customer who fails after year-end

Suppose a company reports \$50 of receivables at year-end. In the following month, a customer owing \$8 enters bankruptcy. Assume, purely illustratively, that the bankruptcy reflects financial difficulty already present at year-end and that no recovery is expected.

```
Reported receivables                     $50
Post-year-end customer balance             8
Illustrative collectible receivables     $42
Illustrative exposure ratio                8 / 50 = 16%
```

The 16% is a teaching calculation, not a claim about any actual company. The forensic question is whether the subsequent event provides evidence about the earlier estimate, and whether MD&A's statement that credit quality was stable is compatible with the event. The answer depends on the facts and accounting framework; the reader's job is to connect the timing.

The intuition: subsequent events do not automatically rewrite the old period, but they can reveal whether the old assumptions were reasonable.

#### Worked example: refinancing after the reporting date

Imagine a company has \$30 of cash, \$20 of current payables, and \$25 of debt due soon after year-end. It announces a \$40 refinancing after the reporting date.

```
Cash before refinancing                    $30
Payables plus near-term debt               $45
Immediate shortfall before new funding     $15
New refinancing announced                  $40
```

The announcement may reduce going-concern risk, but it does not mean the company had \$40 of cash at the year-end date. Check the terms, conditions, collateral, interest cost, maturity, and whether the refinancing is committed or merely expected. Management's MD&A should distinguish liquidity on the reporting date from funding obtained later.

## 7. MD&A: read the verbs, then rebuild the bridge

MD&A is strongest when it explains causation rather than repeating tables. The SEC says MD&A should help users understand financial condition, changes in financial condition, results of operations, liquidity, capital resources, and critical accounting estimates. It also asks companies to focus on material information and known trends and uncertainties ([SEC Commission guidance](https://www.sec.gov/rules-regulations/2003/12/commission-guidance-regarding-managements-discussion-analysis-financial-condition-results-operations)).

Start by underlining causal verbs: “driven by,” “primarily due to,” “offset by,” “reflects,” “resulted from,” “benefited from,” and “impacted by.” For every verb, find a number. If revenue was “driven by volume,” find units or customer counts. If margin was “benefited by mix,” find the segment table. If cash was “impacted by working capital,” compute receivables, inventory, and payables. If debt was “managed prudently,” read maturities and covenant headroom.

Then mark evasive verbs: “may,” “could,” “we believe,” “we expect,” “not expected to,” and “cannot assure.” They are not bad words. They are uncertainty markers. The question is whether they are attached to a quantified range, a date, or a scenario.

#### Worked example: rebuilding an MD&A bridge

Suppose MD&A says operating profit rose from \$25 to \$34. It attributes \$6 to volume, \$5 to price, and \$2 to cost savings, while saying currency reduced profit by \$4.

```
Starting operating profit                $25
Volume effect                              +6
Price effect                               +5
Cost savings                               +2
Currency effect                            (4)
Implied ending operating profit           $34
```

The bridge adds up. Now test the quality of each component. Does volume appear in segment data? Does price show up in revenue per unit? Do cost savings appear as lower expenses or merely as a reclassification? Does the currency effect agree with the geographic mix and disclosed exchange exposure? A bridge that reconciles numerically can still be economically weak if its labels are not independently observable.

The intuition: a bridge is a set of hypotheses. Reconciliation proves the pieces sum; it does not prove the labels are right.

![The MD&A bridge turns management's causal verbs into testable components: start with the reported result, add claimed drivers, subtract offsets, and reconcile every component to a note or cash-flow line.](/imgs/blogs/the-footnotes-and-mda-where-the-bodies-are-buried-7.webp)

### Liquidity language deserves its own read

Liquidity is the ability to meet obligations as they fall due. It is not the same as having positive equity or positive net income. Read the liquidity section with the debt note, lease note, commitments, restricted-cash note, and cash-flow statement open.

Look for: negative operating cash flow, recurring reliance on asset sales, supplier-finance programs, factoring of receivables, covenant waivers, short-term debt used to fund long-lived assets, customer deposits, and debt that technically matures after the reporting date but has a demand feature if a covenant fails.

#### Worked example: positive earnings, negative operating cash

Suppose net income is \$18. Depreciation is \$6, receivables rise by \$20, inventory rises by \$8, and payables rise by \$4.

```
Net income                               $18
Depreciation                              +6
Receivables increase                      (20)
Inventory increase                         (8)
Payables increase                          +4
Illustrative operating cash                $0
```

The company can be profitable and cash-neutral in this simplified example. If it also spends \$12 on equipment and pays \$5 of debt, it needs \$17 of external or beginning cash to fund those uses. A statement that emphasizes earnings but barely discusses this funding need is not necessarily false, but it is not a complete liquidity discussion.

The intuition: earnings answer “what did accounting recognize?” Liquidity answers “what must be funded before the next cash arrives?”

## 8. The fast 10-K or annual-report red-flag workflow

You do not need to read every page at equal intensity. A fast first pass can identify the locations that deserve a slower second pass.

![A fast annual-report workflow moves from report map to changes, reconciliations, concentrations, uncertainty, later events, and finally a short list of questions.](/imgs/blogs/the-footnotes-and-mda-where-the-bodies-are-buried-8.webp)

### Pass 1: map the document

Record the reporting date, comparative periods, accounting framework, auditor, basis of consolidation, and the table of contents. Note whether the company changed its year-end, acquired or disposed of a business, restated prior periods, or changed its segment structure.

### Pass 2: search high-yield words

Search the PDF or HTML for:

```
change in accounting policy
change in estimate
restated
reclassified
critical accounting
related party
guarantee
contingent
commitment
subsequent event
going concern
covenant
restricted cash
factoring
customer concentration
supplier finance
```

Search is triage, not analysis. Read the surrounding paragraphs and the table referenced by each hit.

### Pass 3: mark the five bridges

Build five small bridges: profit to operating cash, revenue to receivables, segment profit to segment cash demand, reported debt to maturities, and stated liquidity to post-year-end funding. Use the company's units and rounding. Do not create false precision by adding numbers that are not comparable.

### Pass 4: scan for concentration

Concentration can be by customer, supplier, geography, lender, product, segment, counterparty, or related party. Concentration is not bad by itself. It is a fragility multiplier: if one relationship changes, multiple lines may move together.

### Pass 5: read the uncertainty language

Circle ranges, probabilities, assumptions, and “unable to estimate” statements. For each, ask what would make the estimate change and whether the report contains an observable early-warning indicator.

### Pass 6: inspect the next event

Read subsequent events and the next available filing or earnings release. Compare what management predicted with what happened. This is not hindsight accounting; it is a way to evaluate whether the assumptions and narrative were calibrated.

### Pass 7: write questions, not accusations

The output of a forensic screen is a question list. Examples: “Why did receivables grow faster than customer cash?” “What changed in the policy and why now?” “Which related party bears credit risk?” “What does ‘available liquidity’ exclude?” “Which segment carries the new debt?” A red flag is a prompt for evidence, not proof of wrongdoing.

## 9. Journal-entry logic: where a story must land

The fastest way to sharpen a footnote read is to ask what journal entry would be required if management's explanation were true. You do not need to become the company's bookkeeper. You need only preserve the two-sided nature of accounting: every recognized event affects at least two accounts, and a claimed benefit must have a corresponding asset, liability, equity, revenue, expense, or cash movement somewhere.

If management says it sold more on credit, revenue rises and receivables usually rise. If it says customers paid faster, cash should rise or receivables should fall. If it says costs were saved, an expense should fall, a liability should fall, or an asset should be acquired with a defensible future benefit. If it says a liability was extinguished, cash, debt, or another obligation should change. The point is not to demand a one-to-one movement every period; timing and classification create legitimate differences. The point is to identify the other side of the claim.

#### Worked example: find the missing other side

Suppose management says a \$15 cost was “avoided” because a supplier gave the company a concession. There are several possible accounting paths:

```
Cash paid less than expected       Cash outflow falls by $15
Payable forgiven                   Liability falls by $15; gain may appear
Price reduction on inventory       Inventory cost falls by $15
Payment merely delayed             Payable remains; cash timing changes only
```

Those paths produce different forensic conclusions. A real forgiveness may be non-recurring. A price reduction may improve future gross margin only because inventory was purchased more cheaply. A delayed payment may make current cash look better while increasing supplier-finance risk. “Supplier support” is not a sufficient explanation; identify the accounting side and the cash side.

### The balance-sheet roll-forward is a lie detector with limits

Roll-forwards are tables that reconcile opening balance, additions, disposals, foreign-exchange movements, impairment, amortization, and closing balance. Read them because they force management to expose movement in an account that a single closing balance hides. Good candidates are goodwill, property and equipment, intangible assets, contract assets, provisions, debt, leases, and deferred tax assets.

But a roll-forward is not independent verification. It is internally consistent evidence produced by the same reporting system. Its forensic value comes from comparison: additions versus capex cash flow, disposals versus investing proceeds, impairment versus segment weakness, debt repayments versus financing cash flow, and provisions versus later settlements.

#### Worked example: asset additions and cash

Imagine property and equipment opens at \$90, receives \$25 of additions, records \$10 of depreciation, and closes at \$105.

```
Opening property and equipment       $90
Additions                             +25
Depreciation                          (10)
Closing property and equipment       $105
```

The roll-forward works. Now suppose the cash-flow statement shows only \$5 of capital expenditure. The \$20 difference may reflect non-cash acquisitions, finance leases, foreign-exchange movements, construction accruals, or a classification difference. It is not proof of an error. It is a bridge that must be explained. Read the lease note, additions table, debt note, and investing cash flow before drawing a conclusion.

The intuition: the most useful forensic question is often “what is the other side of this movement?”

## 10. Classification and non-GAAP measures: the same business in different clothing

Companies often present adjusted, underlying, organic, constant-currency, or normalized measures in MD&A. These can be useful if they are defined consistently and reconciled to the closest reported measure. They become hazardous when exclusions are asymmetric, recurring, or chosen so that the answer looks better every year.

Create a two-column bridge: reported result on the left, each adjustment in the middle, and the adjusted result on the right. Then classify each adjustment by whether it is genuinely unusual, non-cash, non-operating, acquisition-related, or simply a cost management wants the reader to ignore. A cash expense can be “non-recurring” and still recur. A non-cash expense can still signal a real economic loss. Stock-based compensation, restructuring, acquisition costs, impairment, and foreign-exchange effects all require context rather than a universal rule.

#### Worked example: adjusted profit that is mostly adjustments

Suppose reported operating profit is \$12. Management adds back a \$5 restructuring charge, a \$4 acquisition expense, and a \$3 stock-based compensation expense.

```
Reported operating profit              $12
Restructuring add-back                  +5
Acquisition expense add-back            +4
Stock compensation add-back             +3
Adjusted operating profit              $24
Adjustments as share of adjusted       12 / 24 = 50%
```

The 50% is arithmetic for this hypothetical example. It does not mean adjusted profit is invalid. It means the reader should ask whether the business can produce \$24 without repeatedly paying costs management excludes. Compare the adjustment list over several years, check cash paid for restructuring, and inspect the acquisition note for integration costs. The reported and adjusted measures answer different questions; neither should silently replace the other.

### “Other” lines deserve a decomposition request

“Other assets,” “other liabilities,” “other operating income,” and “corporate costs” are aggregation buckets. Aggregation saves space but can conceal heterogeneity. A small table can contain tax receivables, derivative balances, prepaid expenses, disputed claims, or related-party balances with completely different risk profiles.

Track the bucket's size and composition over time. The first year a line becomes material is usually more informative than the tenth year it remains material. If the report says the balance is “primarily” one item, look for the remainder. If the bucket grows after an acquisition, compare it with purchase-price allocation and goodwill. If a liability bucket rises while operating cash improves, ask whether obligations were reclassified or deferred.

#### Worked example: an “other” bucket with two economic meanings

Suppose “other assets” is \$30. The note says \$18 is prepaid insurance and \$12 is a tax receivable.

```
Prepaid insurance                 $18 / $30 = 60%
Tax receivable                    $12 / $30 = 40%
```

Insurance prepayments unwind as coverage is consumed. A tax receivable depends on filing position, jurisdiction, audit, and collection timing. Treating the entire \$30 as one liquid asset would be a category error. The note turns a seemingly simple number into two separate questions.

The intuition: aggregation is a compression algorithm; the note tells you which information was compressed away.

## 11. How to compare years without being fooled by presentation

Comparability is a prerequisite for trend analysis. Check whether the company changed its fiscal year, acquired or sold a subsidiary, changed functional currency, moved costs between segments, adopted a new standard, or restated prior periods. A chart with five annual columns can still represent five different business perimeters.

When prior periods are recast, retain both the originally reported and restated figures if the filing provides them. The restated series is usually the better basis for current comparison, while the original series can show how much the accounting change altered the old narrative. When periods are not recast, build a pro forma bridge cautiously and label it as your analysis rather than company-reported data.

#### Worked example: a twelve-month year is not always comparable

Imagine a company reports a ten-month transition period followed by a twelve-month year. Revenue in the transition period is \$100 and revenue in the following period is \$120.

```
Naive comparison = ($120 - $100) / $100 = 20%
```

The 20% is not a valid annual growth rate if the first period covers fewer months. A simple monthly run-rate illustration would be \$100 / 10 = \$10 per month versus \$120 / 12 = \$10 per month, suggesting no change under that crude assumption. Seasonality, acquisitions, and closures make the actual comparison more complex, but the issue is visible immediately: verify the period before calculating growth.

The intuition: before asking whether the line grew, ask whether the two lines measure the same amount of time and the same perimeter.

### Use repeated language as a time series

The prose itself can be tracked. Copy the same risk paragraph from one annual report to the next and highlight changed nouns, verbs, and qualifiers. “We may experience” becoming “we have experienced” is a change in fact. “We expect” becoming “we expect could” is a change in confidence. A new sentence about supplier dependence or customer concentration can matter even when the related amount is not yet large.

This is especially useful for contingencies, liquidity, litigation, and critical estimates. A company may not change the number because the loss is not yet measurable, but it may change the probability language. That linguistic movement is not a substitute for quantification; it is a signal to locate the underlying event.

## 12. A practical reading sheet you can reuse

Create one page for each annual report with these boxes:

```
Reporting date and comparative periods:
Accounting framework and auditor:
Acquisitions, disposals, restatements, policy changes:

MD&A claim 1 -> note -> statement -> cash test -> open question:
MD&A claim 2 -> note -> statement -> cash test -> open question:
MD&A claim 3 -> note -> statement -> cash test -> open question:

Largest customer / supplier / lender concentrations:
Related parties and balances due:
Contingencies and disclosed ranges:
Debt, leases, commitments, and covenant dates:
Subsequent events and next-filing test:
```

Limit yourself to three important MD&A claims on the first pass. If you try to investigate every sentence, the document will win by exhaustion. Rank claims by their effect on recurring earnings, liquidity, control, and downside. A small policy change that touches revenue recognition may rank above a larger but transparent acquisition. A moderate contingent liability that could trip a covenant may rank above a large ordinary trade payable.

The final line on the sheet should be a confidence statement: “reported result well supported,” “reported result supported but fragile,” or “important evidence unresolved.” Those are analytical states, not investment recommendations. They preserve uncertainty instead of forcing a binary verdict.

One more habit makes the sheet useful across a series of reports: freeze your definitions. Decide whether “cash” means cash and cash equivalents or includes restricted cash. Decide whether “debt” includes leases, supplier-finance obligations, and convertible instruments. Decide whether “revenue growth” is reported, constant-currency, organic, or pro forma. Write the choice at the top of the page and use it consistently. Many apparent contradictions are really unit or perimeter changes.

For example, a company can say cash increased while the cash-flow statement shows cash and cash equivalents falling if management is using a broader liquidity definition that includes an undrawn facility. A company can say net debt fell while total debt rose if it acquired cash or changed the treatment of leases. Neither statement should be accepted or rejected from the headline alone. Put the definition beside the number, then ask whether the definition is the one a lender, supplier, or shareholder would actually face.

Also preserve the report's rounding. If the filing reports amounts in millions, do not manufacture exact dollars by multiplying the displayed number. A displayed “\$1,245 million” may represent a rounded amount; your analysis should say “about \$1.245 billion” or retain the filing's units. Precision is not the same as accuracy. False precision makes a small reconciliation error look like a discovery when it may be rounding.

Finally, save the page or filing date for every conclusion. Annual reports can be amended, later filings can restate prior periods, and a subsequent event can change the evidence. A dated note that says “the report disclosed a possible loss range, as issued on the filing date” is more defensible than an undated statement that the company “has” a liability. Forensic accounting is document analysis; document analysis needs version control.

### What would change your mind?

For every red flag, write the evidence that would reduce the concern. For a receivables build, it may be subsequent cash collections, aging, and independent customer confirmations. For a related-party sale, it may be third-party pricing, cash settlement, and board approval. For an impairment estimate, it may be a sensitivity table and observable market evidence. For a liquidity concern, it may be a committed refinancing with clear terms and no covenant waiver dependence.

This habit protects against confirmation bias. The purpose of forensic reading is not to find a sinister interpretation for every unusual disclosure. It is to find the most decision-relevant uncertainty and test it fairly.

## Common misconceptions

### “The footnotes are boilerplate.”

Some paragraphs are boilerplate. The existence of boilerplate does not make every note unimportant. The high-value signal often appears as a changed sentence, a new table, a new counterparty, a changed estimate, or a new qualification inside familiar language.

### “If the auditor signed, there is no accounting risk.”

An audit opinion is not a guarantee that the company will perform, that every forecast is accurate, or that every risk is obvious. It is an opinion on whether the financial statements are presented fairly under the applicable framework, subject to materiality and audit evidence. The footnotes remain part of the audited statements, and MD&A remains a separate narrative disclosure with its own purpose.

### “A related-party transaction is automatically fraud.”

No. Related parties can transact for ordinary business reasons. The risk is dependence, pricing, collectability, governance, and disclosure. Independent corroboration and settlement behavior matter more than the label alone.

### “A contingency not recorded is not a liability.”

It may not meet the recognition threshold, but the disclosed exposure can still affect valuation and liquidity. Carry the range as a scenario and read the probability language.

### “Subsequent events are irrelevant because they happened next year.”

They may be irrelevant to measurement of the old period, or they may provide evidence about conditions that already existed. Either way, they are relevant to understanding the risk a reader faced at issuance.

### “The consolidated number is the business.”

Consolidation is an accounting presentation. The economics may be distributed across segments, subsidiaries, affiliates, joint ventures, and related parties. Segment and related-party notes tell you where the group number is actually generated and who stands opposite it.

## How it shows up in real markets

### 1. Enron: the footnote map mattered

Enron is the named case to remember when “off balance sheet” sounds like a technical phrase. SEC-hosted material about Enron discusses special-purpose entities, liabilities kept off the balance sheet, income generated through partnerships, and related-party disclosure concerns ([SEC-hosted Enron material](https://www.sec.gov/comments/other/other-initiatives/otherinitiatives-72.pdf)). The historical lesson is not that every affiliate is suspicious. It is that a structure can move risk away from the face statements while leaving clues in ownership, guarantees, financing, and related-party notes.

For a current filing, begin with the legal structure and then ask the economic questions: who funded the vehicle, who controlled decisions, who absorbed losses, and who received the reported gain? If the answer is “the company’s own executives or an entity they influenced,” the transaction deserves independent confirmation and a close read of consolidation judgments.

### 2. WorldCom: capitalization changes where the expense appears

WorldCom is a case study in why policy and classification analysis matters. The SEC's enforcement archive and historical accounting-enforcement materials document the Commission's work around accounting and auditing enforcement ([SEC accounting and auditing enforcement archive](https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases)). The lesson for a reader is mechanical: when a cost is capitalized rather than expensed, current-period profit and assets rise while future periods inherit depreciation or impairment.

In a live filing, do not rely on the label “capital expenditure.” Trace the accounting policy, the journal-entry logic, the cash-flow classification, and the asset roll-forward. Ask whether the spending created a controlled resource with future benefit or merely moved an operating cost into a balance-sheet account.

### 3. The recurring modern pattern: growth plus receivables

This is not a claim about one named issuer. It is a repeatable pattern across industries: revenue accelerates, receivables grow faster, management describes demand as strong, and the subsequent-events note later describes customer distress or returns. The correct response is not to declare revenue fake. It is to examine credit terms, allowance methodology, concentration, contract assets, bill-and-hold arrangements, channel inventory, and cash collection after year-end.

The pattern matters because accrual accounting recognizes economic activity before cash settlement. That is useful when the estimates are sound and dangerous when management controls the assumptions. A reader should compare revenue growth with operating cash and receivables over several periods, then read the policy note that defines when revenue becomes revenue.

### 4. The recurring modern pattern: “available liquidity” with conditions

A company may describe a cash balance, revolving facility, or refinancing as evidence of liquidity. The footnotes may reveal restricted cash, borrowing-base conditions, collateral, covenants, near-term maturities, or a facility that was undrawn but not unconditionally available. The question is not whether the facility exists. It is how much can be drawn, by when, at what cost, and under what event of default.

This is why MD&A and the debt note must be read together. MD&A gives management's funding narrative; the note gives maturity and contractual detail. Subsequent events tell you whether the funding was actually completed.

## When this matters to you

This method is useful whenever you are comparing businesses, reading an investment memo, evaluating a borrower, or simply trying to understand why a company with impressive earnings still feels financially fragile. It is also useful for avoiding the opposite mistake: treating every estimate and related party as evidence of misconduct.

The discipline is to separate three layers:

1. **Reported fact:** what the filing says happened under its accounting rules.
2. **Economic interpretation:** what that fact may imply about cash, control, concentration, and future obligations.
3. **Open question:** what evidence would confirm or weaken the interpretation.

Keep those layers separate in your notes. “Receivables increased” is fact. “Revenue quality deteriorated” is an interpretation. “Show post-year-end collections by major customer” is a question. That separation is the difference between forensic accounting and storytelling.

The fastest durable habit is to read one note before accepting one narrative. When MD&A says why, ask the footnotes how. When the footnotes disclose an exposure, ask the cash-flow statement when. When subsequent events arrive, ask whether the earlier assumptions were reasonable. The bodies are not buried in one mysterious sentence. They are distributed across the joins.

That is where disciplined reading earns its keep.

## Sources & further reading

- [Commission guidance regarding MD&A](https://www.sec.gov/rules-regulations/2003/12/commission-guidance-regarding-managements-discussion-analysis-financial-condition-results-operations), U.S. Securities and Exchange Commission, issued December 29, 2003; accessed August 4, 2026.
- [Accounting policies: cautionary advice regarding disclosure](https://www.sec.gov/rules-regulations/2001/12/accounting-policies-cautionary-advice-regarding-disclosure), U.S. Securities and Exchange Commission, issued December 2001; accessed August 4, 2026.
- [Disclosure about critical accounting policies](https://www.sec.gov/rules-regulations/2002/05/disclosure-managements-discussion-analysis-about-application-critical-accounting-policies), U.S. Securities and Exchange Commission, issued May 10, 2002; accessed August 4, 2026.
- [Enron's professional intersection](https://www.sec.gov/comments/other/other-initiatives/otherinitiatives-72.pdf), SEC-hosted historical material discussing special-purpose entities and related-party disclosure issues; accessed August 4, 2026.
- [Accounting and auditing enforcement releases](https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases), U.S. Securities and Exchange Commission archive; accessed August 4, 2026.
- [The three financial statements and how they interlock](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock), related post in this series.
- [Reading the cash-flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income), related post in this series.
- [Reading the balance sheet: what companies hide here](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here), related post in this series.
