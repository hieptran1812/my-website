---
title: "Satyam and Toshiba: Two Faces of Asian Accounting Fraud"
date: "2026-08-12"
publishDate: "2026-08-12"
description: "One company invented cash that never existed. The other forged nothing at all and still overstated profit for years. Two mechanisms, two forensic toolkits, and one lesson about where to look."
tags:
  [
    "forensic-accounting",
    "satyam",
    "toshiba",
    "financial-statement-fraud",
    "percentage-of-completion",
    "revenue-recognition",
    "auditing",
    "india",
    "japan",
    "fraud-detection",
    "earnings-quality",
  ]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 64
depth: "deep-dive"
---

> [!important]
> **TL;DR:** There are only two ways to make a company look richer than it is. You can invent a fact, or you can bend an estimate. Satyam Computer Services did the first, Toshiba did the second, and almost nothing that catches one of them catches the other.
>
> - **Fabrication** means the asset or the sale never existed. It needs forged documents, it compounds every quarter because the fake balance has to keep growing, and one letter from an auditor directly to a bank ends it.
> - **Stretching an estimate** means the asset and the sale are real and only the measurement is bent. Nothing is forged, so confirmation procedures find nothing wrong. It surfaces through patterns instead: margins that are too smooth, cash that lags earnings for years, and cost estimates that keep getting revised in a convenient direction.
> - Double-entry bookkeeping is the constraint that makes fabrication expensive. A fake asset needs a fake credit, and every credit you could choose leaves a different forensic trail.
> - Percentage-of-completion accounting is where the second kind of fraud lives, because it lets an estimate of costs that have not been incurred yet decide how much profit you report this quarter.
> - Most investors are trained to hunt forgeries. The second pattern is far more common, and the tests that catch it are a different set entirely.
> - **The two numbers to remember:** 94% of Satyam's reported cash at 30 September 2008 did not exist, per its chairman's own letter. And Toshiba's reported net income for the year to March 2012 was nearly 22 times the correct figure, per Japan's securities regulator, with nothing forged at all.

Ask most people what accounting fraud looks like and they will describe a forgery. Someone in a back office is typing numbers that are not true, printing a bank statement that no bank ever issued, inventing a customer, moving a decimal point. It is a satisfying picture because it has a villain and a moment.

That picture is right about roughly half of the cases and badly wrong about the other half. The other half is quieter. Nobody forges anything. Every invoice is real, every customer is real, every bank balance is exactly what the bank says it is. And the reported profit is still too high, by billions, for years, because a set of numbers that the accounting rules *require* management to estimate were estimated optimistically, over and over, in the same direction, under pressure from the top of the company.

Two Asian cases sit almost perfectly at the two ends of that spectrum, close enough in time to compare directly.

**Satyam Computer Services** was an Indian IT services company listed in Mumbai and, through American Depositary Receipts, on the New York Stock Exchange. In January 2009 its founder and chairman handed the board a letter admitting that a large part of the cash on its balance sheet did not exist. Not overvalued. Not illiquid. Not there.

**Toshiba** was, and to a degree still is, one of Japan's flagship industrial companies: nuclear reactors, elevators, semiconductors, laptops, medical scanners. In 2015 an independent investigation committee found that Toshiba had overstated pretax profit for years. It found no fabricated bank balances, no invented customers, no forged confirmations. It found a corporate culture in which senior management set profit targets that divisions could not meet honestly, and an accounting model, percentage of completion, that gave those divisions a legal-looking place to put the difference.

![Two ways to overstate profit: fabricate a fact, or stretch an estimate. Each is caught by a different test.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-1.webp)

The diagram above is the mental model for this entire post. Down the left is fabrication: something that never existed is asserted to exist, the evidence supporting it must be forged, and the detection tool is independent confirmation from a third party who has no reason to lie for the company. Down the right is estimate-stretching: something real is measured too favourably, the evidence is genuine, and the detection tool is pattern recognition across time.

They are not degrees of the same thing. They are different crimes with different mechanics, different footprints on the financial statements, different detection methods, and, as we will see at the end, very different legal outcomes.

This post builds both mechanisms from zero, with worked numbers you can follow on paper, and then walks each real case across the actual statement lines that moved. If you have read the earlier posts in this series on [fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) and [revenue recognition games](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold), this is where those two threads get put side by side and compared.

## Foundations: how a lie gets onto a financial statement

Before we can talk about either case, we need three ideas. None of them requires prior finance knowledge, and all three do real work later.

### Idea one: every entry has two sides

A financial statement is not a list of numbers that management types in. It is the output of a system called **double-entry bookkeeping**, which has one iron rule: every transaction is recorded twice, once as a *debit* and once as a *credit*, and the two must be equal.

You do not need the vocabulary to feel the constraint. Think about your own finances. If your bank balance goes up by \$1,000, something else must have happened. Either you earned \$1,000, or you sold something worth \$1,000, or somebody lent you \$1,000. The money did not appear from nowhere, and if you kept a household ledger, the ledger would insist that you say which of those three it was.

Companies work the same way, which produces the **balance sheet identity**:

$$
\text{Assets} = \text{Liabilities} + \text{Equity}
$$

Read in plain English: everything the company owns (*assets*) was paid for either with money it owes to someone else (*liabilities*) or with money that belongs to its owners (*equity*), where equity includes every dollar of profit the company has ever earned and not paid out. If you want to add an asset to the left-hand side, you must add something to the right-hand side too, or the statement does not balance and the accounting software refuses to close the books.

This is the single most important fact about financial fraud, and it is the reason fabrication is so much more expensive than people assume. A fraudster cannot simply add \$100 million of cash. They must also answer the question *where did it come from*, in a way that survives being written down.

![You cannot add \$100 million of cash to a balance sheet without saying where it came from, and every answer leaves a trail.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-3.webp)

#### Worked example: the entry that creates a fictitious bank balance

Suppose you run a company and you want \$100,000,000 of cash to appear on your balance sheet that is not actually in any bank account. The debit side is easy and it is the same in every version:

```
Dr  Cash and bank balances        $100,000,000
Cr  ???                                        $100,000,000
```

Now you must fill in the credit. You have exactly three families of answer, and each one has a different cost.

**Option A: credit revenue.** You claim the cash came from sales.

```
Dr  Cash and bank balances        $100,000,000
Cr  Revenue                                    $100,000,000
```

This is the most attractive option, because it does two jobs at once: it explains the cash *and* it inflates the income statement, which is usually what you actually wanted. But it is also the most expensive to maintain. Revenue implies customers, and customers imply invoices, contracts, purchase orders, delivery records and, eventually, payments. If you book \$100 million of revenue to customers who do not exist, you have to keep generating documents for them forever, and the sales never turn into a phone call from a real person.

In practice a fraudster usually cannot make the fake cash arrive instantly, so the entry runs through receivables first:

```
Dr  Trade receivables             $100,000,000
Cr  Revenue                                    $100,000,000

...later, when the invoice is "collected":

Dr  Cash and bank balances        $100,000,000
Cr  Trade receivables                          $100,000,000
```

That intermediate step is why fabricated revenue almost always shows up first as a ballooning receivables balance, which is the classic red flag covered in the earlier post on [inventory and receivables inflation](/blog/trading/forensic-accounting/inventory-and-receivables-inflation-the-classic-red-flag).

**Option B: credit equity.** You claim the cash came from issuing shares.

```
Dr  Cash and bank balances        $100,000,000
Cr  Share capital                              $100,000,000
```

This is the cheapest to write and the easiest to catch. Share issuance is recorded by a share registrar and a stock exchange, both of which are outside the company. Anyone can check. Nobody serious uses this route.

**Option C: credit a liability you do not disclose.** You claim, silently, that the cash was borrowed.

```
Dr  Cash and bank balances        $100,000,000
Cr  Borrowings (not disclosed)                 $100,000,000
```

This one does not inflate profit at all, so on its own it is useless for making the company look profitable. Its actual use is different: it is how you *fund* a fraud once the fake cash starts having to do real things. If the company has to pay a real bill out of a balance that does not exist, someone has to put real money in, and the person who does that is usually an insider who quietly lends it.

**The intuition:** a fake asset can never stand alone, because a debit without a credit is not an accounting entry, it is an error message. Whichever credit the fraudster picks, they have committed to a story that leaves a specific, findable trail.

### Idea two: some numbers are facts and some are opinions

The second foundation is that a balance sheet is not uniformly solid. Some lines on it are facts that a stranger can verify for you. Others are the output of a judgement that only management can make.

The word for the verification is **confirmation**: an auditor writes to a third party who holds or owes something, and asks them to state the amount directly back to the auditor. Send a letter to the bank, and the bank replies with the balance. Send a letter to a customer, and the customer replies with what they owe. The point is that the reply does not pass through the company's hands.

![The confirmation ladder: at the top a stranger can prove the number for you, at the bottom only management's judgement produces it.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-2.webp)

Read the ladder from the top down and the character of the numbers changes completely:

- **Cash at a bank** is the hardest fact on the balance sheet. There is a specific institution that either holds your money or does not, and it will say so in writing.
- **Listed securities** are nearly as hard. A custodian confirms you own them and a public market prices them.
- **Trade receivables** are a hybrid. Whether the invoice exists is checkable by writing to the customer. Whether the customer will *pay* is a judgement, expressed as an allowance for doubtful debts.
- **Inventory** flips the balance. You can count the boxes in the warehouse, so existence is a fact. What the boxes are worth is a judgement, and a large one, because inventory has to be written down to net realisable value if it will not sell for cost.
- **Revenue on a long-term contract** has no external confirmation at all in the middle years, for reasons we will build up in a moment. The number depends on an estimate of costs that have not yet been incurred.
- **Goodwill, warranty reserves and pension liabilities** are pure model output. They are produced by assumptions management selects: discount rates, failure rates, mortality tables, growth forecasts.

Satyam's fraud lived on the top rung. Toshiba's lived on the fifth. That single fact explains almost everything else about how the two cases behaved.

### Idea three: an estimate can be wrong without anyone lying

Here is the part that most people find genuinely counterintuitive, and it is the hinge of this whole post.

Accounting rules do not merely *permit* management to estimate. They **require** it. There is no version of a financial statement that avoids estimates, because many economically real transactions are not finished when the reporting period ends.

Consider a construction company that signs a four-year contract in year one. If accounting only recognised finished work, the company would report zero revenue and heavy losses for three years and then an enormous profit in year four. That would be a worse description of reality than the alternative, so the rules take the alternative: recognise revenue and profit gradually, in proportion to how much of the job has been done. But "how much of the job has been done" is not a fact you can look up. It has to be estimated. And the standard way to estimate it uses the one thing you can measure, cost:

$$
\text{Percent complete} = \frac{\text{Costs incurred to date}}{\text{Total costs expected for the whole contract}}
$$

The numerator is a fact: you know what you have spent. The denominator is a forecast of the future, made by the same people whose bonus depends on the answer.

This means two things at once, and you have to hold both:

1. A company can report a percent-complete figure that turns out to be wrong, revise it later, and have committed no fraud whatsoever. Estimates are supposed to change as information arrives. That is what an estimate is.
2. A company can also report a percent-complete figure it knows is wrong, on purpose, to hit a target. This is fraud. But the document trail looks identical, because in both cases what exists is a real contract, real costs, and a number in a spreadsheet.

The forensic consequence is severe. When someone forges a bank statement, there is a physical artefact that is false and a real bank that will contradict it. When someone shaves a cost estimate, there is no artefact that is false. The contract is real, the costs are real, the invoices are real, the customer is real. What is wrong is a judgement, and judgements do not confess.

You cannot catch that with a confirmation letter. You catch it, if you catch it, by noticing that the judgements all point the same way, quarter after quarter, in the divisions under the most pressure.

## Mechanism one: fabricating an asset

Let us build the first mechanism properly, because its economics are much stranger than they look.

Fabrication has a structural problem that estimate-stretching does not: **it compounds**. Once you have claimed \$100 million of cash that does not exist, that claim is now permanent. It sits on the balance sheet in every future period. And because the company is supposed to be growing, and because growing companies generate more cash, the fake balance has to grow too, or the growth story stops making sense.

So the fraudster is not committing one act. They are running a machine that must be fed every quarter, forever.

![The manufactured-cash loop: each turn of the wheel converts a fake sale into a fake bank balance, and the fake balance then earns fake interest.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-4.webp)

#### Worked example: one fake invoice, all the way to a fake fixed deposit

Follow \$10,000,000 of invented revenue through the machine. Every step is an entry that a real accounting system would accept.

**Step 1. Create the customer and the invoice.** You need a name, an address and an invoice number. In a services business this is easier than in a manufacturing business, because there is no physical product that has to move. You are billing for hours of work.

**Step 2. Book the sale.**

```
Dr  Trade receivables             $10,000,000
Cr  Revenue                                    $10,000,000
```

The income statement now shows \$10 million more revenue. If your real operating margin is 20%, this fake sale also carries essentially 100% margin, because there are no real costs against it. That is an important side effect: fabricated revenue *raises* your reported margin, which is a tell, because it means the company looks more profitable the more it lies.

**Step 3. Pretend the invoice was paid.**

```
Dr  Cash and bank balances        $10,000,000
Cr  Trade receivables                          $10,000,000
```

The receivable disappears and the cash line grows. Notice what this step is really for. It is not for the money. It is to stop the receivables balance from growing so fast that it becomes the obvious anomaly. Fraudsters convert fake receivables into fake cash precisely because analysts watch receivables.

**Step 4. Produce evidence for the cash.** This is the step that turns an accounting entry into a crime with a physical exhibit. The company's own records say the money is in the bank. The auditor will want the bank to say so too. So the fraud must produce something that looks like a bank statement or a fixed-deposit receipt, and it must arrive on the auditor's desk in a way that appears to have come from the bank.

There is a reason frauds of this type so often put the fake money into **fixed deposits** (in other markets: term deposits, certificates of deposit) rather than a current account. A current account has traffic. Money moves in and out every day, and the statement is thick with transactions that all have to be internally consistent. A fixed deposit is a single number that sits still for a year. It is a much easier document to fake, and it looks prudent and boring, which is exactly what you want.

**Step 5. Accrue the interest.** And here the machine turns on its operator.

```
Dr  Interest receivable              $800,000
Cr  Interest income                              $800,000
```

If you claim to be holding \$10 million in a deposit yielding 8%, then next year you must report \$800,000 of interest income, because a deposit that earns nothing is not a deposit. That interest is also fake, so it becomes another asset that does not exist, and the total hole grows by \$800,000 without anyone doing anything at all.

**The intuition:** fabricated cash is not a static lie, it is a compounding liability. It generates fake income, which generates more fake assets, which require more fake evidence, which is why every fraud of this kind ends the same way: it gets too big to service and someone has to confess.

### What the fake balance cannot do

The deepest weakness of fabricated cash is that it cannot be spent. Real cash pays salaries, buys equipment, funds acquisitions and pays dividends. Fake cash does none of those things, and yet it sits on the balance sheet looking exactly like the real kind.

This creates a permanent operational squeeze. The company appears to be swimming in money, so it *should not* need to borrow, and analysts will ask why it does. But its real cash is much smaller than reported, so it often *does* need to borrow, and it does so quietly, or it takes money from insiders, or it delays payments.

That gap between apparent liquidity and actual behaviour is the single most reliable tell for this kind of fraud, and it is behavioural rather than numerical. A company with a mountain of cash that keeps raising short-term debt, never pays a meaningful dividend, and earns almost no interest on its balance is telling you something.

#### Worked example: the interest-income cross-check

This is the test I would run first on any company reporting a suspiciously large cash pile. It takes about two minutes and it needs only the published accounts.

Suppose a company reports:

- Cash and bank balances at the start of the year: \$1,000,000,000
- Cash and bank balances at the end of the year: \$1,200,000,000
- Interest income for the year: \$12,000,000

Average cash over the year is:

$$
\frac{\$1{,}000{,}000{,}000 + \$1{,}200{,}000{,}000}{2} = \$1{,}100{,}000{,}000
$$

So the implied yield on that balance is:

$$
\frac{\$12{,}000{,}000}{\$1{,}100{,}000{,}000} = 1.09\%
$$

Now ask the only question that matters: **what could a large corporate depositor actually have earned on cash in that currency, in that year?** If short-term rates in that market were around 7% or 8%, as they were in India in 2008, then a 1.09% realised yield is not a conservative treasury policy. It means one of two things. Either most of that balance is not earning anything, which for a large corporate deposit is close to impossible, or most of that balance is not there.

Run the arithmetic backwards to see how sharp the test is. At a plausible 7% yield, \$1.1 billion of average cash should have produced roughly:

$$
\$1{,}100{,}000{,}000 \times 0.07 = \$77{,}000{,}000
$$

The company reported \$12 million. The implied *real* balance is therefore about:

$$
\frac{\$12{,}000{,}000}{0.07} = \$171{,}000{,}000
$$

That is roughly 16% of the reported balance. You have just estimated the size of the hole from two published lines, without access to a single bank statement.

**The intuition:** fake cash does not earn interest, and interest income is disclosed. Any balance that is real must be visibly productive, and if it is not, the burden of explanation sits with the company.

There is a real subtlety worth naming: some companies genuinely hold large non-interest-bearing balances, particularly if they hold customer float in current accounts, or hold cash in a currency with near-zero rates, or hold it in jurisdictions with capital controls. So a low implied yield is a question, not a verdict. But it is a question that a legitimate company can answer in one sentence and a fraudulent one cannot answer at all.

## Mechanism two: stretching an estimate

Now the other end of the spectrum, which is subtler and, in aggregate, much more common.

### Percentage of completion, from zero

Imagine you are building a power plant. You sign a contract in January 2021 to deliver it by December 2024 for a fixed price of \$1,000 million. You expect it to cost you \$800 million to build, so you expect to make \$200 million of profit over four years.

How much profit should you report in 2022?

The strict answer, "none, because the plant is not finished", produces financial statements that describe reality badly. For four years the company would look like it was losing money, and then in one quarter it would look like a spectacular success. Investors would learn nothing about how the business was actually performing.

So accounting standards let you recognise revenue and profit **over time**, in proportion to progress. The near-universal way of measuring progress is the cost-to-cost method already introduced:

$$
\text{Percent complete} = \frac{\text{Costs incurred to date}}{\text{Total estimated cost at completion}}
$$

$$
\text{Revenue recognised to date} = \text{Percent complete} \times \text{Contract price}
$$

$$
\text{Profit to date} = \text{Revenue recognised to date} - \text{Costs incurred to date}
$$

Every symbol here matters. *Costs incurred to date* is a historical fact from the general ledger. *Contract price* is a fact from a signed document. *Total estimated cost at completion*, usually called the **estimate at completion** or the **cost to complete**, is the only forward-looking number, and it is entirely a management judgement.

And it sits in the denominator. That is the whole game.

![Shaving \$100 million off an estimate of future costs releases \$71 million of profit today, with no change to the contract, the customer or the cash.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-7.webp)

#### Worked example: a cost-to-complete revision and the profit it releases

Take the power plant. It is the end of 2022 and you have spent \$400 million so far.

**Version one: the honest estimate.** You still believe the job will cost \$800 million in total.

$$
\text{Percent complete} = \frac{\$400\text{m}}{\$800\text{m}} = 50\%
$$

$$
\text{Revenue to date} = 50\% \times \$1{,}000\text{m} = \$500\text{m}
$$

$$
\text{Profit to date} = \$500\text{m} - \$400\text{m} = \$100\text{m}
$$

Sensible: you are half done, so you have booked half the revenue and half the expected profit.

**Version two: shave the estimate.** Now suppose you are \$70 million short of your annual profit target, and someone senior makes that everybody's problem. You do not need to forge anything. You need only to become more optimistic about the second half of the job. You revise the estimate at completion from \$800 million down to \$700 million, on the grounds that the team has learned the process, steel prices have eased, and the commissioning phase will go faster than budgeted.

$$
\text{Percent complete} = \frac{\$400\text{m}}{\$700\text{m}} = 57.1\%
$$

$$
\text{Revenue to date} = 57.1\% \times \$1{,}000\text{m} = \$571\text{m}
$$

$$
\text{Profit to date} = \$571\text{m} - \$400\text{m} = \$171\text{m}
$$

You have just released **\$71 million of additional profit into the current period**. Consider what did and did not happen:

| | Before the revision | After the revision |
| --- | --- | --- |
| Contract price | \$1,000m | \$1,000m |
| Costs actually incurred | \$400m | \$400m |
| Cash received from the customer | unchanged | unchanged |
| Estimate at completion | \$800m | \$700m |
| Revenue recognised to date | \$500m | \$571m |
| **Profit to date** | **\$100m** | **\$171m** |
| Documents forged | none | none |

Not one cent of cash moved. No invoice was created. No customer was invented. The only thing that changed is a number in a spreadsheet describing costs that have not happened yet, and that number is one management is required by the accounting rules to produce.

**Version three: the reckoning.** Long-run contracts do not let you keep this. Suppose the plant actually costs \$900 million, because commissioning goes badly. The contract is now a loss:

$$
\text{Final profit} = \$1{,}000\text{m} - \$900\text{m} = -\$100\text{m}
$$

Every dollar of the \$171 million already booked has to come back, plus the \$100 million loss on top. Accounting standards make this worse by requiring that once a contract is expected to be loss-making, **the entire expected loss is recognised immediately**, not spread over the remaining years. So the reversal does not arrive gently. It arrives as one enormous charge in one quarter, usually described in the press release as an "unforeseen cost overrun".

**The intuition:** percentage-of-completion accounting does not create profit, it only relocates profit in time, and an optimistic estimate is a loan from the future that the future always collects.

### Why this mechanism is so hard to attack from outside

Three properties make estimate-stretching structurally resistant to the standard forensic toolkit.

**It has no counterparty.** There is nobody an auditor can write to. The customer can confirm the contract price and the milestones, but the customer has no view on what the remaining work will cost the contractor. That information exists only inside the company.

**It is individually defensible.** Any single revision has a story, and the story is usually true in part. Steel prices really did move. The team really did get faster. A reviewer challenging one revision is arguing about engineering judgement with the engineers, from the outside, with less information.

**It is only visible in aggregate.** One optimistic estimate is noise. Forty optimistic estimates across eleven divisions over seven years, all in the same direction, all landing in the quarters where the target was missed, is a pattern. But you can only see the pattern if you can see all forty, and an outside investor sees none of them individually. They see one consolidated number.

This is why frauds of this type are so often surfaced by a **whistleblower or an internal investigation** rather than by an analyst or a short seller. The pattern is only visible from inside.

### The second family: moving costs between periods

Percentage of completion is the most elegant version of estimate-stretching, but it is not the only one. The same logic (real transactions, bent measurement) applies to any account where the timing of a cost is a judgement call.

**Deferring costs.** Costs are supposed to be recognised in the period they relate to. If a cost can plausibly be attributed to a future product, a future period, or an asset rather than an expense, then recognising it later moves profit into today. The judgement is genuine, which is why this works. Development costs, ramp-up costs, tooling, rework: each has a legitimate accounting argument for capitalising or deferring, and each also has a version that is simply not paying for what you used. This is a cousin of the capitalisation abuse covered in the [WorldCom post](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move), but softer, because the accounts involved are estimates rather than a straightforward misclassification.

**Not writing down inventory.** Inventory is carried at the lower of cost and net realisable value. If a warehouse of last year's television panels will now only fetch 60% of what you paid, you are required to write it down and take the loss. Deciding that it will *probably* sell at cost next quarter defers that loss. Again, no forgery: the inventory is real and the count is accurate. Only the valuation judgement is bent.

**The masked-markup transaction.** This one is worth its own treatment, because it looks like fabrication at first glance and is not.

![The Buy-Sell loop: profit appears when parts leave, and only disappears when the finished laptop comes back and is sold.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-8.webp)

Many electronics companies do not assemble their own products. They design the product, buy the components, and hand assembly to a contract manufacturer, often called an **ODM** (original design manufacturer). A common and entirely legitimate structure is that the brand owner buys the parts centrally, for scale, and sells them on to the assembler, who builds the machine and sells it back as a finished unit.

The structure creates a temptation. The price at which you sell parts to your assembler is a price you control, and the assembler does not care what it is, because they will charge it straight back to you inside the finished-goods price. So you can sell the parts at a large markup, book the markup as profit today, and repurchase it inside the cost of the finished product later.

#### Worked example: the masked markup and its reversal

Per laptop:

1. You buy the components from suppliers for **\$100**.
2. You sell those same components to the assembler for **\$400**. Your books record a gain of **\$300** now. (In practice this is often recorded not as revenue but as a reduction of cost of sales, which makes it much less visible.)
3. The assembler builds the machine and sells it back to you for **\$430**: the \$400 of parts plus \$30 of assembly. That \$430 goes into your inventory.
4. You sell the finished laptop to a retailer for **\$450**. Your gross profit on the sale is **\$450 minus \$430 = \$20**.

Add it up across the whole cycle:

$$
\$300 + \$20 = \$320
$$

$$
\$450 - \$100 - \$30 = \$320
$$

The two agree, which is the point: **over a complete cycle the mechanism creates no profit at all.** It only decides *when* the \$320 gets reported. If parts go out and finished goods come back within the same quarter, the reported profit is the honest \$320 and nothing is distorted.

Now break the symmetry. In the last two weeks of a quarter, ship a much larger volume of parts to the assembler than the assembler can possibly build and return. Say you push out an extra 100,000 units' worth of components:

$$
100{,}000 \times \$300 = \$30{,}000{,}000
$$

Thirty million dollars of profit lands in this quarter. It is not fake in the sense that a forged bank statement is fake: the parts physically exist, they physically moved, and there is a genuine invoice for them. But it is profit on a transaction with your own assembler for goods that will come straight back to you, and next quarter it reverses, so next quarter you have to push even more parts out to stand still.

**The intuition:** a transaction can be completely real and still be an accounting device, because what makes it a device is not the transaction, it is the timing and the intent behind the volume.

Notice the family resemblance to the compounding problem in mechanism one. Both machines have to be fed with a larger dose each period. The difference is that the fabrication machine is fed with forged documents and the estimate machine is fed with real ones.

## Satyam: the letter

Everything in this section comes from two primary documents. The first is B. Ramalinga Raju's letter to the board of Satyam Computer Services, dated 7 January 2009, which Satyam filed with the US Securities and Exchange Commission the same day as Exhibit 99.2 to a Form 6-K. The second is the SEC's own account of the case, published on 5 April 2011 when it settled charges against the company and against its auditors. Where I quote, I am quoting those filings.

### A note on Indian units

Indian financial reporting uses two counting words that are unfamiliar elsewhere and that you need in order to read the letter at all.

- One **lakh** is 100,000.
- One **crore** is 100 lakh, which is **10 million**.

So Rs 5,040 crore means 5,040 × 10,000,000 rupees, or 50.4 billion rupees. Indian statements also group digits differently (Rs 50,40,00,00,000 rather than Rs 50,400,000,000), which is worth knowing if you ever read an Indian annual report directly.

For US dollar equivalents I use **48.56 rupees per dollar**, the Federal Reserve H.10 rate for 7 January 2009, the date of the letter. Every rupee figure below is converted at that single rate so the comparisons stay internally consistent.

### What the company looked like from outside

Satyam was one of India's large IT services exporters, headquartered in Hyderabad, listed on the Indian exchanges, with American Depositary Shares trading on the New York Stock Exchange. In the letter, Raju describes the business he built: it had grown "from few people to 53,000 people, with 185 Fortune 500 companies as customers and operations in 66 countries."

The letter also gives the scale of the reported financials at the September 2008 quarter: an "annualized revenue run rate of Rs. 11,276 crore" (about \$2.32 billion) and "official reserves of Rs. 8,392 crore" (about \$1.73 billion). This was not a small company or an obscure one. It was audited by an affiliate of one of the largest audit networks in the world, and it had passed every check that the Indian and American disclosure regimes put in front of it.

### The four lines

The letter opens without preamble. Here is what it says the balance sheet carried as of 30 September 2008, in the letter's own order and its own words:

| Item as stated in the letter | Rupees | US dollars at 48.56 |
| --- | --- | --- |
| "Inflated (non-existent) cash and bank balances" | Rs 5,040 crore | \$1,038 million |
| ...against the amount "reflected in the books" | Rs 5,361 crore | \$1,104 million |
| "An accrued interest of Rs. 376 crore which is non-existent" | Rs 376 crore | \$77 million |
| "An understated liability... on account of funds arranged by me" | Rs 1,230 crore | \$253 million |
| "An over stated debtors position" | Rs 490 crore | \$101 million |
| ...against debtors "reflected in the books" | Rs 2,651 crore | \$546 million |

![The balance sheet at 30 September 2008: Rs 5,361 crore of reported cash was Rs 321 crore of real money wearing a Rs 5,040 crore costume.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-6.webp)

#### Worked example: reading the confession as a balance sheet

Take the letter's numbers and rebuild the two lines it touches.

**Cash and bank balances.** The books said Rs 5,361 crore. Of that, Rs 5,040 crore did not exist. So the real balance was:

$$
\text{Rs } 5{,}361\text{ crore} - \text{Rs } 5{,}040\text{ crore} = \text{Rs } 321\text{ crore}
$$

That is about \$66 million of real money, against \$1,104 million reported. As a proportion:

$$
\frac{5{,}040}{5{,}361} = 94.0\%
$$

**Ninety-four percent of the reported cash was not there.** This is worth sitting with, because it is the number that makes the case famous. This was not a company that shaded a balance. It reported roughly sixteen times more cash than it had.

**Trade debtors.** The books said Rs 2,651 crore, overstated by Rs 490 crore, so the real figure was:

$$
\text{Rs } 2{,}651\text{ crore} - \text{Rs } 490\text{ crore} = \text{Rs } 2{,}161\text{ crore}
$$

About \$445 million against \$546 million reported. Notice how *small* this distortion is relative to the cash one: 18% overstated versus 94%. That asymmetry is the fingerprint of the machine described earlier. Fake receivables were being converted into fake cash as fast as possible, precisely because receivables are the line analysts watch and cash is the line they trust.

**The total.** The four confessed items are:

$$
5{,}040 + 376 + 1{,}230 + 490 = \text{Rs } 7{,}136\text{ crore}
$$

About **\$1.47 billion**. The letter does not itself state this total: it is simply the sum of the four items it lists, and I give it as arithmetic rather than as a quotation.

Cross-check that against the other primary document. The SEC's press release of 5 April 2011 says the scheme "resulted in more than \$1 billion in fictitious cash and cash-related balances, representing half the company's total assets." The cash-related items in the letter are the non-existent cash of Rs 5,040 crore plus the non-existent accrued interest of Rs 376 crore, which is Rs 5,416 crore, or about \$1.12 billion at the January 2009 rate. The two documents agree.

**The intuition:** the confession is not a narrative, it is a list of balance-sheet lines, and every fraud of this type can eventually be written down the same way.

### The quarter where you can watch the machine run

The letter's second paragraph is the most forensically interesting sentence in the whole document, and almost nobody quotes it. Here it is in full:

> "For the September quarter (Q2) we reported a revenue of Rs.2,700 crore and an operating margin of Rs. 649 crore (24% 0f revenues) as against the actual revenues of Rs. 2,112 crore and an actual operating margin of Rs. 61 Crore ( 3% of revenues). This has resulted in artificial cash and bank balances going up by Rs. 588 crore in Q2 alone."

Work the arithmetic and something remarkable falls out.

![One number, three places: the Rs 588 crore of revenue Satyam did not earn in the September 2008 quarter is exactly the profit it did not make, and exactly the fake cash it added.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-5.webp)

#### Worked example: the Rs 588 crore identity

**The revenue gap.** Reported revenue was Rs 2,700 crore (\$556 million). Actual revenue was Rs 2,112 crore (\$435 million).

$$
\text{Rs } 2{,}700\text{ crore} - \text{Rs } 2{,}112\text{ crore} = \text{Rs } 588\text{ crore}
$$

That is about \$121 million of revenue in a single quarter that did not happen.

**The profit gap.** Reported operating margin was Rs 649 crore (\$134 million), which the letter notes was 24% of reported revenue. Actual operating margin was Rs 61 crore (\$13 million), which was 3% of actual revenue.

$$
\text{Rs } 649\text{ crore} - \text{Rs } 61\text{ crore} = \text{Rs } 588\text{ crore}
$$

**The same number.** The revenue gap and the profit gap are identical, to the crore.

This is not a coincidence and it is not rounding. It is the arithmetic signature of fabricated revenue. Invented sales have no cost of delivery, because there is no delivery. So every rupee of fake revenue drops straight through to operating profit, and the revenue gap and the profit gap must be equal.

Look at what that does to the reported margin. The real business was earning 3% operating margins in that quarter, which for an IT services company in 2008 was poor. The reported business was earning 24%, which was excellent. **The entire difference between a struggling company and a star was fabrication**, and the fabrication made the company look not just bigger but structurally more profitable than it was.

**And then the cash.** The letter's own next sentence closes the loop: "This has resulted in artificial cash and bank balances going up by Rs. 588 crore in Q2 alone." The same Rs 588 crore. Fake revenue became fake profit became fake cash, one quarter at a time, and the balance sheet hole grew by exactly the size of the lie told in the income statement.

**The intuition:** in a fabrication fraud the three statements do not disagree with each other, they agree perfectly, and the fact that they agree is what makes the fraud invisible from inside the numbers.

### Why it could not stop

The letter is unusually direct about the mechanics of being trapped, and the passages matter because they explain the compounding problem from mechanism one in the words of someone who lived inside it.

On how it began: "What started as a marginal gap between actual operating profit and the one reflected in the books of accounts continued to grow over the years. It has attained unmanageable proportions as the size of company operations grew significantly."

On why it could not be unwound: "As the promoters held a small percentage of equity, the concern was that poor performance would result in a take-over, thereby exposing the gap. It was like riding a tiger, not knowing how to get off without being eaten."

That sentence is the one the SEC later quoted in its own press release, and it deserves its reputation. But the operationally revealing part is elsewhere.

Remember the structural weakness of fabricated cash: **it cannot be spent**. The letter describes exactly that squeeze and exactly the workaround, which is the Rs 1,230 crore of undisclosed liability from the four-line list:

> "That in the last two years a net amount of Rs. 1,230 crore was arranged to Satyam (not reflected in the books of Satyam) to keep the operations going by resorting to pledging all the promoter shares and raising funds from known sources... The last straw was the selling of most of the pledged share by the lenders on account of margin triggers."

Read that against the balance sheet. A company reporting Rs 5,361 crore of cash (\$1.1 billion) was quietly borrowing Rs 1,230 crore (\$253 million) against pledged founder shares in order **to keep paying its bills**. That is the behavioural tell from Checklist A in its purest form: apparent liquidity and actual behaviour pointing in opposite directions.

And it is what ended the fraud. When markets fell in 2008, the lenders holding those pledged shares hit their margin triggers and sold. The quiet source of real cash was cut off, and a company with \$1.1 billion of reported cash could no longer fund itself.

The letter also explains the last manoeuvre, an attempt to buy two companies associated with the founder's family: "The aborted Maytas acquisition deal was the last attempt to fill the fictitious assets with real ones." That sentence is worth reading twice. It describes an acquisition whose purpose was not strategic. It was to convert a fictitious asset into a real one by spending fake cash on a real business, so that the balance sheet would afterwards contain something that existed. Shareholders rejected it, and seventeen days later Raju wrote the letter.

### The audit failure, in the regulator's own words

This is the part of the case that generalises furthest, and it is why Satyam appears in a post about *where to look* rather than only in a post about what happened.

Recall the confirmation ladder. Cash is the top rung precisely because a third party will confirm it. So how does \$1 billion of non-existent cash survive multiple annual audits?

The SEC answered that question directly on 5 April 2011, when it sanctioned five India-based PricewaterhouseCoopers affiliates (Lovelock & Lewes, Price Waterhouse Bangalore, Price Waterhouse & Co. Bangalore, Price Waterhouse Calcutta, and Price Waterhouse & Co. Calcutta) over the Satyam audits. From the SEC's account of its order:

> PW India's "failure to properly execute third-party confirmation procedures resulted in the fraud at Satyam going undetected" for years.

And on why this was not a one-client problem:

> PW India staff "routinely relinquished control of the delivery and receipt of cash confirmations entirely to their audit clients and rarely, if ever, questioned the integrity of the confirmation responses they received from the client by following up with the banks."

That is the whole case in two sentences. The procedure that makes cash the hardest number on a balance sheet is *independence of the reply*. The letter must go from the auditor to the bank, and the answer must come from the bank to the auditor, without the company handling either. Route it through the client and you have not performed a confirmation. You have asked the company whether the company is telling the truth.

The SEC's description of the mechanism matches the machine from mechanism one exactly: former senior officials "used false invoices and forged bank statements to inflate the company's cash balances", creating "more than 6,000 phony invoices to be used in Satyam's general ledger and financial statements", with employees creating "bogus bank statements to reflect payment of the sham invoices."

Six thousand invoices. That is the cost of running a fabrication machine for years, and it is why this kind of fraud always requires many participants and always leaves a mountain of physical evidence. The post on [what an audit does and does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) goes further into why a clean opinion is not a guarantee.

### What followed

From the SEC's 5 April 2011 press releases, which are the primary record I am relying on here:

- **Satyam** settled with the SEC, agreeing to a **\$10 million penalty**, to specific securities-law and accounting training for officers and employees, to improvements in its internal audit function, and to hiring an independent consultant to evaluate its internal controls. It neither admitted nor denied the allegations.
- **The five PW India affiliates** settled for a **\$6 million penalty**, which the SEC described as "the largest ever by a foreign-based accounting firm in an SEC enforcement action", plus a censure and a cease-and-desist order. In a related proceeding, Lovelock & Lewes and Price Waterhouse Bangalore agreed to pay the **Public Company Accounting Oversight Board a \$1.5 million penalty**.
- **The Indian government** dissolved Satyam's board, appointed government-nominated directors, removed the former top managers, and oversaw a bidding process to select a new controlling shareholder.

That bidding process is documented in the SEC's own filing archive. On 8 June 2009, **Venturbay Consultants Private Limited and Tech Mahindra Limited** filed a Schedule TO tender offer for Satyam's common shares at **Rs 58 per share in cash**, valuing the roughly 199 million shares sought at about **\$251 million** using the filing's own stated conversion of one dollar to 46 rupees.

On the criminal side, I am going to be careful, because this is where secondary sources go wrong most often. What I can source directly is the SEC's statement of 5 April 2011: that Indian authorities "filed criminal charges against several former officials", that the case "resulted in criminal charges against seven former executives", and that as of that date Raju, other former Satyam executives, and two lead engagement partners from PW India "are defendants in a criminal trial now underway in India."

**I am deliberately not stating the outcome of those Indian proceedings in this post.** Criminal verdicts in India are frequently appealed and sentences frequently suspended pending appeal, and I could not reach a primary court record to establish the current status. A number you cannot source is worth less than an acknowledged gap, and this series has been burned before by a confident secondary summary.

## Toshiba: the estimate

Now the other end of the spectrum. Again, the primary documents do the work: Toshiba's own press release of 21 July 2015, and the Securities and Exchange Surveillance Commission of Japan's recommendation of 7 December 2015. Yen figures are converted at **123.93 yen per dollar** for July 2015 and **123.31 yen per dollar** for December 2015, both Federal Reserve H.10 rates for the dates in question.

### What the committee was asked to look at

On 15 May 2015, Toshiba established an Independent Investigation Committee chaired by **Koichi Ueda, attorney-at-law**. Toshiba's own announcement states the four areas of "inappropriate accounting" it was set up to investigate, and the list is a remarkably clean map of mechanism two:

1. "Accounting in relation to the **percentage-of-completion method**"
2. "Accounting in relation to **recording of operating expenses in the Visual Products Business**"
3. "Accounting in relation to **valuation of inventory in the Semiconductor Business**, mainly discrete and system LSIs"
4. "Accounting in relation to **component transactions**, etc., **in the PC Business**"

Compare that to the mechanisms built earlier in this post. Item 1 is the cost-to-complete estimate. Item 2 is the timing of cost recognition. Item 3 is the inventory valuation judgement. Item 4 is the masked-markup transaction with a contract assembler. Four different accounts, one shared logic: **real transactions, bent measurement**.

There is no fifth item about forged bank statements, because there were none.

### The headline number

Toshiba received the committee's report "just after 5:30 p.m. on July 20" 2015, and published the full version, with redactions for trade secrets and privacy, on 21 July. In the same 21 July announcement, the company stated the finding itself:

> "a substantial amount of inappropriate accounting over a long period of time, from fiscal 2008 to fiscal 2014. The outcome is that the cumulative amount of income before income tax to be corrected, discovered within the scope of the investigation carried out by the Independent Investigation Committee, is minus 151.8 billion yen."

Three things in that sentence deserve attention.

**"Income before income tax."** This is a pretax profit figure, not revenue and not net income. Precision matters here because the same case gets quoted with several different numbers attached, and they measure different things.

**"From fiscal 2008 to fiscal 2014."** Seven fiscal years. At 123.93 yen per dollar, 151.8 billion yen is about **\$1.22 billion** of pretax profit that had to be taken back out.

**"Within the scope of the investigation."** The committee looked at four specified areas. This is a floor on the problem rather than a complete accounting of it.

### The regulator's numbers, which are sharper

The committee's aggregate is the famous figure, but for a reader trying to learn *what estimate-stretching does to a set of accounts*, the Japanese securities regulator's numbers are far more instructive, because they are per-year and they are net income.

On 7 December 2015 the Securities and Exchange Surveillance Commission recommended that an administrative monetary penalty order be issued against Toshiba for violation of disclosure requirements. Its recommendation covers Toshiba's annual securities reports for the years ended 31 March 2012 and 31 March 2013, plus five shelf-registration supplements used to issue bonds between December 2010 and December 2013. The specific findings:

![What a bent estimate is worth: Japan's securities regulator found Toshiba's stated net income for the year to March 2012 was 70.1 billion yen against a correct 3.2 billion.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-9.webp)

| Year ended | Consolidated net income as stated | Correct amount | Overstatement |
| --- | --- | --- | --- |
| 31 March 2012 | 70,054 million yen (\$568m) | 3,194 million yen (\$26m) | ~66,860 million yen (\$542m) |
| 31 March 2013 | 77,366 million yen (\$627m) | 13,425 million yen (\$109m) | ~63,941 million yen (\$519m) |

Do the division:

$$
\frac{70{,}054}{3{,}194} = 21.9 \qquad \frac{77{,}366}{13{,}425} = 5.8
$$

For the year to March 2012, Toshiba reported **nearly twenty two times** its correct net income. Not 22% too much. Twenty two times.

Hold that against the mental model this post opened with. No bank statement was forged. No customer was invented. Every contract was real, every shipment happened, every yen of cash was where the bank said it was. And the bottom line was out by a factor of twenty two.

The SESC also names the cause, in language that maps straight onto the mechanics built earlier: the errors included **"understating provisions for contract losses and overstating sales."**

That first phrase is the technical heart of the case, and it deserves a worked example of its own, because it is the single most powerful lever in percentage-of-completion accounting and almost nobody outside the industry knows it exists.

#### Worked example: understating a provision for contract losses

Go back to the power plant. Contract price \$1,000 million. You have spent \$400 million so far. But this time the news is bad: your honest current estimate is that the job will cost **\$1,100 million** to finish.

**Step 1: recognise that the contract is a loser.**

$$
\$1{,}000\text{m} - \$1{,}100\text{m} = -\$100\text{m}
$$

The contract will lose \$100 million over its life.

**Step 2: work out what percentage-of-completion has recognised so far.**

$$
\text{Percent complete} = \frac{\$400\text{m}}{\$1{,}100\text{m}} = 36.4\%
$$

$$
\text{Revenue to date} = 36.4\% \times \$1{,}000\text{m} = \$364\text{m}
$$

$$
\text{Result to date} = \$364\text{m} - \$400\text{m} = -\$36\text{m}
$$

So the ordinary running of the method has already booked a \$36 million loss.

**Step 3: apply the rule that catches people out.** Accounting standards do not let you spread an expected loss over the remaining life of a contract. Once a contract is expected to be loss-making, **the entire remaining loss must be provided for immediately**. You have recognised \$36 million of the \$100 million, so you must book a provision for the rest:

$$
\text{Provision required} = \$100\text{m} - \$36\text{m} = \$64\text{m}
$$

```
Dr  Contract loss (income statement)     $64,000,000
Cr  Provision for contract losses                     $64,000,000
```

**Step 4: now do not book it.** Decide, instead, that the cost estimate of \$1,100 million is too pessimistic. Argue that commissioning will go better than the engineers fear, that a supplier claim will be settled favourably, that the schedule can be recovered. Revise the estimate at completion back to \$1,000 million, and the contract is no longer expected to make a loss at all. Nothing needs providing.

**This period's reported profit is \$64 million higher.** No invoice was created, no customer invented, no document forged. A provision that should have been recorded was not recorded, because the estimate that would have required it was revised.

**The intuition:** the most valuable number in a long-term contract business is the one that says the future will be worse than planned, and it is also the only number in the accounts that management can simply decline to write down.

### The culture that generated the estimates

A worked example shows what one revision is worth. It does not explain why the revisions all pointed the same way for seven years across several unrelated divisions. For that, the committee's own conclusion is the evidence, and Toshiba reported it in the 21 July announcement:

> "The Independent Investigation Committee has pointed to the involvement of top management in respect of the causes of the inappropriate accounting."

The committee's recommendations, as Toshiba summarised them, were that the company needed "a change of thinking on the part of top management", a "strong internal control function", a strengthened "auditing function", and more outside directors with redefined roles.

Read those as a diagnosis rather than as boilerplate. Every item describes a failure of *counter-pressure*, not a failure of detection. The accounts were not wrong because nobody checked the arithmetic. They were wrong because the people making the estimates were under sustained pressure from above to produce a particular answer, and nothing in the governance structure was strong enough to push back. That is the structural precondition for mechanism two, and it is why Checklist B ends with a question about incentives rather than a ratio.

It also explains why this fraud was found by an internal investigation rather than by the market. There was nothing for an outsider to confirm. There was only a pattern, and the pattern was legible only to people who could see all the divisions at once.

### The consequences

**The people.** Effective 21 July 2015, Toshiba announced the resignation of eight directors. The list, from the company's own announcement: **Hisao Tanaka**, Representative Executive Officer, President and CEO; **Norio Sasaki**, Vice Chairman of the Board; Hidejiro Shimomitsu, Masahiko Fukakushi, Kiyoshi Kobayashi and Toshio Masaki, all Representative Executive Officers and Corporate Senior Executive Vice Presidents; **Makoto Kubo**, Chairman of the Audit Committee; and Keizo Maeda, who resigned as Representative Executive Officer and Director. Separately and on the same day, **Atsutoshi Nishida**, Adviser to the Board and a former president, resigned his position. Masashi Muromachi, the Chairman, took over as interim President and CEO from 22 July.

Note who is on that list. Two former chief executives and the chairman of the audit committee. This was not a rogue division.

**The company.** The SESC's 7 December 2015 recommendation was for an administrative monetary penalty of **7,373,500,000 yen**, about **\$59.8 million** at the December 2015 rate. Under Japan's system this is a civil monetary penalty for defective disclosure, assessed against the issuer.

**The auditor.** On 22 December 2015 Japan's Financial Services Agency took disciplinary action against **Ernst & Young ShinNihon LLC** over its audits of Toshiba for the years ended 31 March 2010, 2012 and 2013. The firm was suspended from accepting new engagements for **three months, from 1 January to 31 March 2016**, and ordered to improve its operations. Individual partners were suspended from providing services for one, three or six months. The FSA separately commenced a hearing procedure toward an administrative monetary penalty order against the firm of **2,111 million yen**, about **\$17.1 million**, on the finding that its partners had, "in negligence of due care, attested that the financial statements of TOSHIBA CORPORATION for FY2011 and FY2012... containing material misstatements as if they contained no material misstatements."

Compare that finding with the one against PW India. The SEC's charge against Satyam's auditors was that they did not perform a procedure: they never independently confirmed cash. The FSA's charge against Toshiba's auditors was **negligence of due care in an attestation**: they looked at judgements that were wrong and signed anyway. Different failures, matching the different frauds. You can fail to check a fact. You cannot "check" an estimate at all, you can only challenge it, and challenging it means telling a client that its own engineers are too optimistic.

## What the two enforcement records actually show

I want to draw one comparison carefully, and mark clearly where the evidence stops.

Set the two verified records side by side.

| | Satyam | Toshiba |
| --- | --- | --- |
| Mechanism | Fabricated cash, revenue and receivables | Stretched estimates across four accounts |
| Evidence created | ~6,000 phony invoices, forged bank statements | Real contracts, real shipments, revised spreadsheets |
| Detected by | The chairman's own confession | An internal independent investigation |
| Regulatory action against the company | SEC: \$10 million penalty, April 2011 | SESC recommendation: 7.3735 billion yen (\$59.8m), December 2015 |
| Regulatory action against the auditor | SEC: \$6 million; PCAOB: \$1.5 million, April 2011 | FSA: 3-month new-business suspension; 2,111 million yen (\$17.1m) penalty procedure, December 2015 |
| Individuals | SEC states Indian authorities charged seven former executives criminally; trial underway as of April 2011 | Eight directors resigned, 21 July 2015 |

The asymmetry in the last row is the point of this whole post, and I want to state it precisely rather than dramatically.

In the Satyam case there is a document that is false on its face. A forged bank statement says a thing that a real bank will contradict in writing. That is what a prosecutor needs: a specific artefact, a specific person who made it, and an independent witness who will testify that it is fake. Fabrication produces criminal evidence as a by-product of the fraud itself.

In the Toshiba case there is no such artefact. There is a spreadsheet containing an estimate of what a power plant will cost to finish, and the estimate turned out to be wrong. To make that criminal you must prove what somebody *believed* at the moment they wrote the number, and that they wrote a different number on purpose. The engineering judgement is genuinely contestable, the pressure from above is usually verbal, and the defence writes itself: we were optimistic and we were wrong.

**What I am not claiming.** I have not verified whether Japanese prosecutors brought criminal charges against any Toshiba executive, and I am not asserting either that they did or that they did not. What the table records is the enforcement action I could source to a named regulator with a date. Likewise I am not stating the outcome of the Indian criminal proceedings. The point stands on the difference in *evidentiary shape* between a forged document and a bent judgement, which is visible in the regulators' own language: the SEC's finding against PW India is that a **procedure was not performed**, while the FSA's finding against Ernst & Young ShinNihon is **negligence of due care in an attestation**. One is a missing step. The other is a professional opinion that should have been harder to give.

This is also why estimate-stretching is the more durable problem. A fraud that requires 6,000 forged invoices ends, because the forging becomes unmanageable and someone breaks. A fraud that requires one optimistic spreadsheet per quarter can run until the contracts themselves come due.

## The same statements, two different footprints

Here is the part that most surprises people who have absorbed the standard advice about reading financial statements, and it is the most useful idea in this post.

The standard advice is: **trust cash, not earnings**. Net income is an opinion, cash is a fact, so if you want to know whether a company is really making money, look at cash flow from operations and compare it to net income. That advice is genuinely good, and this series makes the case for it at length in [reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

It works beautifully against one of our two mechanisms and it fails completely against the other. Worse, it fails against the one people are more afraid of.

### Why the cash flow statement does not catch fabricated cash

The cash flow statement is not an independent measurement of your bank account. It is **derived from the same general ledger** that produced the balance sheet and the income statement. Its job is to explain the change in the cash balance by classifying the entries that moved it.

So if you fabricate the bank balance, you do not break the cash flow statement. You complete it. The fake sale created a receivable, the fake collection converted that receivable into cash, and the cash flow statement dutifully reports an operating inflow. Every subtotal reconciles. The statement is internally perfect, because the ledger it came from is internally perfect. What is wrong sits entirely outside the accounting system, in the fact that the bank does not agree.

This is why a company running a fabrication fraud can post beautiful cash conversion for years. It is not that they beat the test. The test was never pointed at them.

### Why the cash flow statement does catch stretched estimates

Now run the same logic against mechanism two, and the result inverts.

When you shave a cost-to-complete estimate, you recognise revenue you have not yet billed and profit you have not yet collected. The corresponding debit is not cash. It is a *contract asset*, historically called unbilled receivables or costs and estimated earnings in excess of billings: an asset that says "the customer owes us for work we have done but not yet invoiced".

That asset is an accrual, and accruals are exactly what the cash-versus-earnings comparison is built to find. The masked-markup transaction behaves the same way: profit is booked when parts leave, and the cash to match it does not arrive until a finished machine is sold to an outside customer, which may be next quarter or never.

So a company stretching estimates reports profit that persistently outruns cash. Not for one quarter, which is normal and can mean nothing, but structurally, year after year, with the gap parked in contract assets and inventory.

#### Worked example: the accruals gap under each pattern

Take two companies, both reporting \$200,000,000 of net income on \$2,000,000,000 of total assets. Use the simplest accruals proxy there is:

$$
\text{Accruals ratio} = \frac{\text{Net income} - \text{Cash flow from operations}}{\text{Total assets}}
$$

The idea is plain: it measures how much of your reported profit did not show up as cash, scaled by the size of the business.

**Company F, the fabricator.** Reported operating cash flow: \$210,000,000.

$$
\frac{\$200{,}000{,}000 - \$210{,}000{,}000}{\$2{,}000{,}000{,}000} = -0.5\%
$$

A negative accruals ratio means cash came in *faster* than profit was reported. On a screen, this is a gold star. It is also completely meaningless, because \$120 million of that operating inflow is the recorded collection of invoices sent to customers who do not exist.

**Company E, the estimate-stretcher.** Reported operating cash flow: \$60,000,000.

$$
\frac{\$200{,}000{,}000 - \$60{,}000{,}000}{\$2{,}000{,}000{,}000} = 7.0\%
$$

Seven percent is high. It says \$140 million of the year's reported profit exists as balance-sheet assets rather than money. Look at where it went and you will find contract assets and inventory rising faster than revenue, which is the signature.

**The intuition:** the accruals test measures the gap between profit and cash *inside the ledger*, so it catches a fraud that lives inside the ledger and misses a fraud that lives outside it.

![Most investors are trained to hunt forgeries. The tests that find a bent estimate are a different set, and they are the ones that matter more often.](/imgs/blogs/satyam-and-toshiba-two-faces-of-asian-accounting-fraud-10.webp)

### The test that does catch fabrication

If cash flow cannot catch fake cash, what can? Three things, in order of power.

**One: independent confirmation.** An auditor sends a request directly to the bank, receives the reply directly from the bank, and never lets the document touch the company. This is not a clever technique. It is the most basic procedure in auditing, and in every large fabrication case the failure is not that the procedure did not exist but that it was subverted: the request was routed through the client, the reply came back to a company-controlled address, or the "bank" that replied was not the bank. The earlier post on [what an audit does and does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) walks through why this happens.

**Two: does the asset behave like the asset?** Fake cash earns no interest, so check the yield. Fake inventory occupies no warehouse space, so check the square metres. Fake receivables are never disputed, so check the ageing. Every real asset produces a second-order economic consequence somewhere else in the statements, and a fabricated one does not.

**Three: does the company act rich?** A company with genuine surplus cash behaves in specific ways. It pays a meaningful dividend, or buys back stock, or funds acquisitions internally, or at least stops borrowing short-term at rates above what its deposits earn. A company that holds an enormous reported balance and still behaves as though it is short of money is describing its real position through its actions while contradicting it in its accounts.

## Common misconceptions

**"Cash on the balance sheet is the one number you cannot fake."** It is the number a third party can most easily *verify*, which is not the same thing. Nothing about a cash line is self-proving. It is as reliable as the confirmation procedure behind it, and if that procedure runs through the company's own hands, it proves nothing at all.

**"If the cash flow statement is clean, the earnings are clean."** The cash flow statement is a reclassification of the same ledger, not an independent audit of your bank account. It catches profit that has not become cash. It cannot catch cash that never existed, because the fabricated collection enters the statement as a perfectly ordinary operating inflow.

**"Changing an accounting estimate is not really fraud."** Changing an estimate because your information changed is not fraud, and is required. Changing an estimate because you are short of your target, while keeping the information the same, is fraud, and it is prosecutable in most jurisdictions. What makes the difference is not the size of the change but the reason, which is why cases of this type turn on emails and meeting notes rather than on documents that are false on their face.

**"An unqualified audit opinion means the numbers were checked."** An audit opinion states that the statements are free of material misstatement *in the auditor's opinion, based on samples and on evidence they were able to obtain*. It is not a guarantee, it is not a fraud investigation, and it is not a certification of every balance. Both of our cases carried clean opinions for years.

**"Frauds like this are obvious in hindsight, so I would have seen it."** In both cases the published accounts contained the information needed to ask a hard question, and in both cases thousands of professional investors read those accounts and did not ask it. The information being present is not the same as the anomaly being salient. That gap is the subject of the [narrative addiction](/blog/trading/analyst-edge/narrative-addiction-when-a-good-story-beats-the-data) post: a growth story tells you what to look at, and what to look at is never the boring line.

**"Estimate-stretching is a lesser offence, so it matters less."** It is treated more leniently by prosecutors, which is a statement about evidentiary difficulty rather than about economic harm. Measured in destroyed value, misallocated capital and mispriced securities, the second mechanism is the larger problem, precisely because it is survivable. A fabrication fraud ends the company. An estimate fraud can run for a decade inside a company that continues to operate.

## How it shows up in real markets

Once you have the two shapes in your head, most of the famous cases sort themselves, and so do a lot of ordinary companies that never became cases at all.

**Wirecard is Satyam's shape, eleven years later and 6,000 kilometres away.** A payments company reported a large cash balance held through third-party partners in jurisdictions its auditor found hard to reach, and the balance turned out not to exist. The mechanism differed in the details, and the geography of the concealment was more sophisticated, but the forensic lesson is identical: a cash number is only as good as the confirmation procedure behind it, and the procedure failed for years because the evidence was routed through parties the company controlled. The [Wirecard post](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros) walks the published accounts line by line, and reading it next to Satyam is the fastest way to internalise that this is a repeating pattern rather than a national one.

**WorldCom is a third shape, and it is worth naming so you do not confuse it with either of ours.** WorldCom did not invent revenue and did not stretch an estimate. It took a real, correctly measured operating cost, line costs paid to other carriers, and recorded it on the wrong line: as a capital asset rather than an expense. Nothing was fabricated and no judgement was bent. A real number was simply put in the wrong place, which meant it was spread across future years instead of hitting this year's profit. That is **misclassification**, and it is the easiest of the three to detect from outside, because capital expenditure and depreciation both move in ways that do not match the physical business. The [WorldCom post](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move) covers the mechanics.

**Enron sits mostly on the estimate side, which surprises people.** The popular memory of Enron is of hidden entities and shredded documents, and the concealment machinery was real. But the engine that produced the reported profit was an accounting judgement: mark-to-market recognition on long-dated energy contracts, where the "market" value of a twenty-year contract in an illiquid market was a model output that Enron itself produced. That is the purest possible form of mechanism two. There is no forged document in a discounted cash flow, only assumptions. The [forensic re-read of Enron](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market) in this series separates the estimate engine from the concealment structure, and the [longer case study](/blog/trading/finance/enron-2001-accounting-fraud) covers the collapse itself.

**Long-cycle industrials are where mechanism two lives permanently.** Aerospace, defence, shipbuilding, engineering and construction, rail signalling, nuclear services: any business where a contract runs for years and revenue is recognised over time. In these industries percentage of completion is not a loophole, it is the normal and correct accounting, and the profit reported in any given quarter is genuinely a function of an estimate. That does not make these companies fraudulent. It makes their reported profit structurally softer than the profit of a company that sells a physical product for cash today, and it means the cost-to-complete disclosures in their footnotes carry more information than the income statement does.

**Modern software and subscription businesses have their own estimates.** Revenue standards require companies bundling several deliverables into one contract to allocate the price across them using a standalone selling price that often has no observable market. Capitalised software development requires a judgement about which phase of a project is research (expense it) and which is development (capitalise it), and about the useful life over which it amortises. These are the same mechanism in a different costume: real transactions, real customers, real cash eventually, and a measurement in the middle that only management can make.

**And the ordinary case, which is most of them.** The great majority of estimate-stretching never becomes a scandal. It is a company that is slightly slow to write down inventory in a weak quarter, slightly quick to recognise a contract modification in a strong one, and slightly optimistic about warranty rates every year. None of it is prosecutable. All of it means the earnings you are capitalising at a multiple are a little better than the cash, permanently. That is not a fraud problem, it is a valuation problem, and it is the reason this checklist is worth running on companies you have no suspicion about at all.

## What each one teaches you about where to look

The practical output of comparing these two cases is two short checklists that you run against different companies for different reasons. Neither is a scoring system and neither produces a verdict. They produce questions, and the value is in whether the company can answer them in one sentence.

### Checklist A: is a reported asset actually there?

Run this when a balance sheet line is unusually large relative to the business, when the asset sits in a jurisdiction the auditor cannot easily reach, or when the company's behaviour and its reported liquidity disagree.

1. **Yield check.** Divide interest income by average cash and compare it to the short-term rate available in that currency and that year. A large unexplained gap is the strongest single signal available from public filings.
2. **Does the company act rich?** Look for the combination of a large cash balance with continued short-term borrowing, a token dividend, and equity raises. A company that has money does not usually pay to rent more of it.
3. **Where is the asset held, and who confirms it?** Read the footnotes for the geography and the counterparty. Assets held through partners, escrow agents, or entities in jurisdictions with weak audit access are structurally harder to confirm, and difficulty of confirmation is where this fraud lives.
4. **Are the receivables ageing in a way that looks human?** Real customers dispute invoices, pay late, and pay partially. A receivables ledger that is unnaturally clean is describing customers who are not real people.
5. **Margin plausibility.** Fabricated revenue carries no cost, so it lifts the reported margin. A company whose margin quietly exceeds every peer, in a business with no obvious structural advantage, is either exceptional or is adding revenue that costs nothing to produce.
6. **Read the auditor's report itself, and the auditor's history.** The earlier post on [red flags in the audit report](/blog/trading/forensic-accounting/red-flags-in-the-audit-report-and-auditor-changes) covers what a key audit matter, a resignation, or an unusual fee ratio actually tells you.

### Checklist B: is a reported profit actually earned?

Run this on any company with long-term contracts, heavy internal manufacturing arrangements, large inventories, or a business model where revenue depends on estimating something.

1. **Accruals gap over a full cycle.** Compare cumulative net income to cumulative operating cash flow over five to seven years, not one. Any single year can diverge for honest reasons. A persistent one-directional gap cannot.
2. **Where did the gap go?** If profit is not becoming cash, it is sitting on the balance sheet with a name. Find it. Contract assets, unbilled receivables, capitalised development costs and inventory are the four places to look.
3. **Read the cost-to-complete disclosures.** Companies with long-term contracts disclose changes in contract estimates, and the number is often material. If revisions are large, frequent and always favourable, you are reading the mechanism directly.
4. **Margin smoothness.** A real project business has lumpy margins, because projects go well and badly at random. Suspiciously stable segment margins in a lumpy business mean something is absorbing the variance, and the thing that absorbs variance is an estimate.
5. **Quarter-end volume.** Look for revenue or shipments concentrated in the final days of a period, especially to distributors, assemblers or related parties rather than end customers. A distribution of sales that spikes on the last day of every quarter is describing an internal deadline, not customer demand.
6. **Read the incentive structure before the accounts.** If the annual report describes aggressive top-down profit targets, if segment heads are publicly held to numbers they do not control, and if the culture rewards commitment over accuracy, the estimates in that company's accounts are under pressure. The [quality of earnings](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) framing is useful here: earnings quality is largely a question about incentives, not about arithmetic.

### Which one to run first

If you only have time for one, run Checklist B. Fabrication frauds are rarer, more spectacular and better covered by the press, and they tend to be discovered by someone else before you need to act. Estimate-stretching is ordinary. It is present, in mild form, in a large fraction of listed companies, and the difference between mild and fraudulent is a matter of degree that only shows up over years.

The other reason is asymmetry of exposure. A fabrication fraud usually destroys the equity in a single week, which is terrible but rare. An estimate fraud slowly overstates the earnings you are paying a multiple of, which means you are systematically overpaying for a large number of ordinary companies. The second one costs most investors more money over a lifetime.

## When this matters to you

You are unlikely to be the person who uncovers the next Satyam. Fabrication frauds are rare, and by the time they are visible from outside they are usually already collapsing.

What you will meet, repeatedly, is the ordinary end of the second pattern. Any company whose revenue depends on estimating something is making judgements every quarter that could reasonably have gone the other way, and the direction they actually go is a fact about that company's culture rather than about its accounting policy. Reading a construction firm's cost-to-complete disclosures, or a hardware company's inventory provisioning, or a software company's capitalised development, is not fraud investigation. It is just knowing what the reported profit is made of.

Three habits carry most of the value:

1. **Compare profit to cash over a full cycle, never one year.** Five to seven years of cumulative net income against cumulative operating cash flow tells you almost everything the accruals literature has to say, and it takes ten minutes.
2. **Ask what a large asset earns.** Cash should earn interest. Inventory should turn. Receivables should be collected. Every asset that produces no second-order economic consequence anywhere else in the statements is a question worth asking out loud.
3. **Read the footnotes on the estimates before the headline numbers.** The [footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) are where a company is required to tell you which of its numbers are judgements and how those judgements changed. A company that revises its estimates in one direction, year after year, is describing itself accurately in a place almost nobody reads.

None of this is investment advice, and none of it produces verdicts. It produces questions, and the useful signal is whether a company can answer them in a sentence.

## Sources & further reading

Every figure in this post traces to one of the following. Where I could not reach a primary document, I said so in the text rather than filling the gap.

**Satyam, primary documents**

- B. Ramalinga Raju, letter to the Board of Directors of Satyam Computer Services Ltd., 7 January 2009. Filed with the US Securities and Exchange Commission as Exhibit 99.2 to Satyam's Form 6-K of the same date. This is the source for every rupee figure attributed to the letter: [sec.gov/Archives/edgar/data/1106056/000114554909000025/u00107exv99w2.htm](https://www.sec.gov/Archives/edgar/data/1106056/000114554909000025/u00107exv99w2.htm)
- US SEC, press release 2011-81, "SEC Charges Satyam Computer Services With Financial Fraud", 5 April 2011. Source for the \$10 million penalty, the "more than \$1 billion in fictitious cash and cash-related balances", the 6,000 phony invoices, and the status of the Indian criminal proceedings as of that date: [sec.gov/news/press/2011/2011-81.htm](https://www.sec.gov/news/press/2011/2011-81.htm)
- US SEC, press release 2011-82, "SEC Charges India-Based Affiliates of PWC for Role in Satyam Accounting Fraud", 5 April 2011. Source for the \$6 million penalty, the \$1.5 million PCAOB penalty, and the findings on third-party confirmation procedures: [sec.gov/news/press/2011/2011-82.htm](https://www.sec.gov/news/press/2011/2011-82.htm)
- Venturbay Consultants Private Limited and Tech Mahindra Limited, Schedule TO (tender offer statement) for Satyam Computer Services Limited, filed 8 June 2009. Source for the Rs 58 per share offer price: [sec.gov/Archives/edgar/data/1106056/000119312509126718/dsctot.htm](https://www.sec.gov/Archives/edgar/data/1106056/000119312509126718/dsctot.htm)

**Toshiba, primary documents**

- Toshiba Corporation, "Notice on Publication of the Full Version of the Investigation Report by the Independent Investigation Committee, Action to be Taken by Toshiba, and Clarification of Managerial Responsibility", 21 July 2015. Source for the minus 151.8 billion yen figure, the fiscal 2008 to fiscal 2014 period, the four investigation scopes, the committee chair, and the named resignations: [archived copy of Toshiba's investor-relations release](https://web.archive.org/web/20150724024354/http://www.toshiba.co.jp/about/ir/en/news/20150721_1.pdf)
- Securities and Exchange Surveillance Commission of Japan, "Recommendation for Administrative Monetary Penalty Payment Order for Violation of Disclosure Requirements by TOSHIBA CORPORATION", 7 December 2015. Source for the 7,373,500,000 yen recommended penalty and the per-year stated versus correct net income figures: [fsa.go.jp/sesc/english/news/reco/20151207-1.htm](https://www.fsa.go.jp/sesc/english/news/reco/20151207-1.htm)
- Japan Financial Services Agency, "Commencement of a Hearing Procedure regarding Administrative Monetary Penalty Payment Order against Ernst & Young ShinNihon LLC", 22 December 2015. Source for the 2,111 million yen penalty procedure and the "negligence of due care" finding: [fsa.go.jp/en/news/2015/20151222-1.html](https://www.fsa.go.jp/en/news/2015/20151222-1.html)
- Japan Financial Services Agency, "Disciplinary action against an Audit firm and Certified Public Accountants", 22 December 2015. Source for the three-month suspension from new engagements and the partner suspensions: [fsa.go.jp/en/news/2015/20151222-2.html](https://www.fsa.go.jp/en/news/2015/20151222-2.html)

**Exchange rates**

All rupee and yen conversions use US Federal Reserve H.10 daily rates, retrieved via the St. Louis Fed's FRED series `DEXINUS` and `DEXJPUS`: 48.56 rupees per dollar on 7 January 2009, 123.93 yen per dollar on 21 July 2015, and 123.31 yen per dollar on 7 December 2015. Conversions are shown at the rate for the date of the underlying document, so a figure from 2009 and a figure from 2015 are each converted at their own contemporaneous rate rather than at a common one.

**Elsewhere in this series**

- [The three financial statements and how they interlock](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock), for the double-entry constraint in full
- [Round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) and [revenue recognition games](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold), for the mechanics of mechanism one
- [How an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch), for why a clean opinion is not a guarantee
- [Wirecard: the missing EUR 1.9 billion](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros), for the same mechanism as Satyam in a European setting
- [Capitalizing costs to inflate profit: the WorldCom move](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move), for the third shape, misclassification
- [Enron: a forensic re-read of SPEs and mark-to-market](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market), for estimate-stretching at its most extreme

