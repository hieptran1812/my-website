---
title: "Steinhoff and the anatomy of a global accounting collapse"
date: "2026-08-12"
publishDate: "2026-08-12"
description: "How a retailer spanning dozens of countries used its own complexity as concealment, what the published statements showed before anyone knew, and the six tests you can run on a tangled group with nothing but a calculator."
tags: ["forensic-accounting", "steinhoff", "financial-statement-fraud", "consolidation", "related-party-transactions", "goodwill", "cash-flow", "segment-reporting", "fraud-detection", "south-africa"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 67
depth: "deep-dive"
---

> [!important]
> **TL;DR:** Steinhoff is the case study for a specific idea: that a group's complexity is not a side effect of the fraud, it is the equipment the fraud runs on.
>
> - A group's accounts add up controlled entities and then delete every transaction between them, and both halves rest on one assertion by management: who is inside the group. The circle exploits it. The group funds a buyer through an intermediary, sells that buyer an asset whose value is a matter of opinion, and books the gain. Net group cash moves by zero; reported profit moves by the whole gain.
> - On 6 December 2017 the Mail & Guardian reported that Steinhoff had announced accounting irregularities and that chief executive Markus Jooste had resigned. The shares fell 58% in Frankfurt that morning, to EUR 1.26, leaving a market value of about EUR 5.7 billion.
> - Steinhoff published an 11-page overview of PwC's forensic investigation on 15 March 2019. Its own Table 1 totals EUR 6,506,596,428 of income from transactions it calls "fictitious and/or irregular" across the financial years 2009 to 2017, with counterparties "said to be, and made to appear to be, third party entities independent of the Steinhoff Group". The roughly 3,000-page full report has not been published.
> - **The number to keep: EUR 6.5 billion of transactions, across eight years, inside a group whose audited statements were public the whole time.** The forensic report named the mechanism. The annual reports had already shown that something did not reconcile.
> - Six tests run on a published annual report: sum the segments, size the goodwill, divide cash flow by profit, locate the cash, read the related-party note for what is missing, and watch for changes in the frame. Steinhoff's scheme happened to defeat the segment test, because the overview says invented income was created at holding-company level and pushed down into weak operating units as "contributions". It could not defeat the cash test, and that is the lesson.
> - Shareholders ended with contingent value rights that pay only from residual value, if any, after all external debt is repaid. The operating businesses largely survived under a new unlisted owner; the listed equity did not.

In December 2017, a company that owned Poundland in Britain, Conforama in France, Mattress Firm in the United States and Pepkor across southern Africa told the market that there were accounting irregularities and that its chief executive had resigned. By the next morning most of its value in Frankfurt was gone.

The interesting question is not how the shares fell. Shares fall. The interesting question is this: for years beforehand, everything about that company was public. Audited statements, filed in two jurisdictions, running to hundreds of pages, covered by analysts at large banks, held by index funds all over the world. What, exactly, was a careful reader supposed to have seen?

That question has a real answer, and it is not "nothing".

![You read one column; the group is hundreds of entities, and profit can be invented anywhere a transaction crosses an entity boundary without being eliminated.](/imgs/blogs/steinhoff-and-the-anatomy-of-a-global-accounting-collapse-1.webp)

The diagram above is the mental model for the whole post. On the left is what an investor reads: one consolidated column, a single set of totals for the whole group. On the right is the thing that produced it: a large number of separate legal companies, in a large number of countries, most of which the investor will never see named. Between them runs a line, and the line is drawn by the company.

Almost every serious accounting failure of the last thirty years is, at bottom, an argument about where that line goes. Enron argued that certain entities were outside it. Steinhoff, according to the overview of the forensic investigation its own board published, transacted with parties presented as third parties. In both cases the accounting was correct given the assertion, and the assertion was the fraud.

So this post does two things. First it builds the machinery from zero: what consolidation is, what control means, why goodwill is an opinion and how a transaction with a friendly counterparty turns into reported profit. We will run the whole mechanism through a worked example with visible numbers, and then run it backwards to see what proper consolidation would have deleted.

Then it turns to the record: what Steinhoff was, what its own published documents said, what the investigation found, and, most usefully, which specific tests on the published statements would have fired before anyone outside the company knew anything.

A word on what this post is not. It is not a retelling of a scandal, and it is careful about the parts that involve real people and unfinished legal proceedings. Where something is alleged rather than established, it says so and gives the date. Where a number comes from a particular document, it names the document. The forensic value here is in the method, not in the drama.


## Foundations: how a hundred companies become one set of accounts

Before any of the forensic work makes sense, you need to know exactly what a set of group accounts *is*, and, more importantly, what it is not. If you already know this, skim. If you do not, everything later in this post depends on it.

### A company, a subsidiary and a group

A **company** is a legal person. It can own things, owe things, sign contracts and be sued. When you buy a share, you own a slice of one specific legal person.

A **subsidiary** is a company controlled by another company. The controller is the **parent**. Together, a parent and all its subsidiaries are a **group**.

Here is the thing that surprises people the first time: a group is not a legal person. You cannot sue "the group". You cannot lend money to "the group". Every contract in the world is signed by one specific legal entity. The group exists only as an *accounting* construct: a way of adding up entities that a single parent controls, so that an investor buying a share in the parent can see the whole economic machine rather than one legal fragment of it.

That gap, between a group as an accounting construct and the legal entities that actually transact, is where this entire post lives.

### Consolidation: add everything, then delete the inside

**Consolidated financial statements** are the group's accounts. The rule for producing them, under IFRS 10 (the international accounting standard on consolidation, in force for annual periods beginning on or after 1 January 2013), is two steps:

1. **Add up** every line of every controlled entity, as if the group were one company.
2. **Eliminate** every transaction that happened *between* those entities.

Step 2 is the one nobody explains properly, and it is the one that matters here.

![Consolidation adds subsidiaries up and then deletes what they sold each other, so only outsider money survives to the reported line.](/imgs/blogs/steinhoff-and-the-anatomy-of-a-global-accounting-collapse-2.webp)

#### Worked example: why elimination exists

Suppose a group has three subsidiaries.

| Entity | Revenue |
| --- | --- |
| Subsidiary A | \$3,000m |
| Subsidiary B | \$1,600m |
| Subsidiary C | \$900m |
| **Simple sum** | **\$5,500m** |

Now suppose \$500m of Subsidiary A's revenue came from selling goods to Subsidiary B. Did the group earn \$5,500m from the outside world? No. It earned \$5,000m. The other \$500m was the group's left hand invoicing its right hand. No outsider paid a cent for it.

So consolidation deletes it:

```
Simple sum of subsidiary revenue          $5,500m
Less: intra-group sales A to B             ($500m)
Consolidated revenue                       $5,000m
```

The same deletion runs down the whole balance sheet. If A holds a \$500m receivable from B, and B holds a \$500m payable to A, both disappear: the group does not owe itself money. If A sold B an asset at a \$60m profit, that \$60m profit disappears too, and the asset goes back to what it cost the group in the first place.

**The intuition: consolidated profit is only the profit a group earned from people outside the group. Elimination is the mechanism that enforces it.**

### The word that does all the work: control

Step 1 said "every controlled entity". So who is inside the line?

Most people assume the answer is a percentage: you own more than 50% of the shares, you consolidate. That is a rough approximation, and it is exactly the approximation a determined group exploits.

The actual standard is **control**, and IFRS 10 defines it with three elements that must all be present. An investor controls an entity when it has:

1. **power** over the entity (the ability to direct the activities that most affect its returns),
2. **exposure to variable returns** from its involvement with the entity, and
3. **the ability to use its power to affect those returns**.

Notice what is absent from that list: share ownership. Ownership is usually *evidence* of power, but it is not the test. You can control an entity you own no shares in at all, through contractual rights, through the ability to appoint and remove the people who run it, through funding arrangements that mean you take all its economic risk, or simply through the fact that it does whatever you tell it to.

The mirror image is a **related party**, defined in IAS 24. A related party is a person or entity related to the reporting entity: a parent, a subsidiary, a fellow subsidiary, an associate, a joint venture, a member of key management personnel, a close family member of any of those, or an entity controlled or jointly controlled by any of those. If a counterparty is a related party, the group must *disclose* the transaction, its amount, its terms and any outstanding balances, whether or not it consolidates the entity.

So there are two separate protections, and they fail in two separate ways:

| Protection | What it does | How it fails |
| --- | --- | --- |
| Consolidation (IFRS 10) | Deletes the transaction entirely | The group asserts it does not control the counterparty |
| Related-party disclosure (IAS 24) | Leaves the transaction in but forces you to see it | The group asserts the counterparty is not related |

Both protections rest on the same foundation: an assertion, by the group, about who the counterparty is. Neither auditor nor investor sees the counterparty's ownership register. And an assertion is not a fact.

### Goodwill: the asset that is really an opinion

One more foundation, because you cannot read an acquisitive group without it.

When a company buys another company, it usually pays more than the fair value of the identifiable net assets it receives. If you pay \$1,000m for a business whose buildings, inventory, receivables and brands are worth \$600m net of its debts, you have paid \$400m for something you cannot point at: the customer relationships, the assembled workforce, the market position, the expected synergies.

Accounting parks that \$400m on the balance sheet as **goodwill**. It is an asset. It sits there. It is not depreciated.

Instead, at least annually, the company must run an **impairment test**: is this thing still worth what we are carrying it at? The test compares the **carrying amount** of the cash-generating unit that holds the goodwill against its **recoverable amount**, which is the higher of fair value less costs of disposal and **value in use**. Value in use is a discounted cash flow: the company's own forecast of the unit's future cash flows, discounted at a rate the company chooses.

Read that again. The asset's value is tested against a forecast the company writes, discounted at a rate the company picks. Goodwill is the only large asset class on a balance sheet whose carrying value is defended almost entirely by management's own model.

This matters enormously for what follows. A group that grows by acquisition accumulates goodwill. A group that accumulates goodwill accumulates balance sheet that only its own forecasts support. And a group whose reported profit is partly invented has a strong incentive never to let those forecasts come down, because an impairment would force it to explain why.

### Segments: the map of the group you are actually given

The last foundation. IFRS 8 requires a listed group to disclose **operating segments**: the components of the business whose results the chief operating decision maker actually reviews. For each reportable segment you get, at minimum, a measure of profit or loss, and often revenue, assets and capital expenditure.

Then, crucially, IFRS 8 requires a **reconciliation**: the total of the reportable segments' profit must be reconciled to the group's consolidated profit before tax. The difference goes in a line usually called "corporate and other", "unallocated", "central costs" or "reconciling items".

That reconciliation line is the single most useful number in a suspicious annual report, and we will spend a whole section on why.

---

## The machine: how a sale to a friend becomes profit

Now the mechanism. I am going to build it with an illustrative company so that every number is visible, and then we will hold the real record up against it.

Meet **Harlow Group**, a listed multinational retailer. Harlow is not real; the numbers below are illustrative arithmetic, chosen to be round enough to follow in your head. The *structure* is what matters, and the structure is not invented.

Harlow's operating businesses are fine. Not spectacular: fine. Furniture sells at low margins, discount retail is a grind, and the European household-goods business has been flat for three years. Management, however, has promised the market growth. There is a gap between the profit the businesses produce and the profit that has been promised.

There are three honest ways to close that gap: sell more, cost less, or admit the promise was wrong. And there is a fourth way, which is to find a transaction that produces accounting profit without producing any economic activity at all.

### The shape of the fourth way

The recipe has four ingredients:

1. **An asset whose value is a matter of opinion.** Brands, intellectual property, development rights, a stake in a private company. Never inventory: inventory has a market price and everyone knows it.
2. **A buyer who will pay any price you name.** Which means a buyer you control, or one so closely connected that the distinction is academic.
3. **Money for the buyer to pay with.** Which the group provides, through a route that does not look like the group providing it.
4. **A story about why the buyer is independent.** This is the part that has to hold up, and it is the part that requires the group structure to be complicated.

Put them together and you get a circle.

![The money goes out one door and comes back through another; only the profit stays.](/imgs/blogs/steinhoff-and-the-anatomy-of-a-global-accounting-collapse-3.webp)

#### Worked example: the round trip that creates \$60m of profit

Harlow has a subsidiary, **Harlow Brands Ltd**, which holds a portfolio of brand names on its books at a **carrying value** (what the accounts say the asset is worth) of \$40m.

There are two outside-looking entities. **Meridian Trading BV** is presented in Harlow's accounts as an independent third-party buyer. **Ashfield Capital** is presented as an independent financier. Neither appears in Harlow's related-party note.

Here is the sequence, over about six weeks near the financial year end.

**Step 1: the group sends money out.** Harlow's treasury subsidiary, Harlow Finance Ltd, advances \$100m to Ashfield Capital. In the accounts this is recorded as a loan receivable, or a deposit, or a prepayment against future supply. It is an asset, so it does not touch the income statement at all.

```
At Harlow Finance Ltd
Dr  Advance to Ashfield Capital        $100,000,000
    Cr  Cash                                        $100,000,000
```

Group cash: down \$100m. Group profit: unchanged. Nobody looking at the income statement sees anything.

**Step 2: the money reaches the buyer.** Ashfield lends \$100m to Meridian Trading BV. This transaction is invisible to Harlow's accounts entirely, because neither Ashfield nor Meridian is consolidated. It happens in a jurisdiction with no public filing requirement for a private company's loan book.

**Step 3: the sale.** Meridian buys the brand portfolio from Harlow Brands Ltd for \$100m.

```
At Harlow Brands Ltd
Dr  Receivable from Meridian Trading   $100,000,000
    Cr  Brand portfolio (intangible)                 $40,000,000
    Cr  Other operating income                       $60,000,000
```

There it is. **\$60m of profit**, being the \$100m sale price less the \$40m the brands were carried at. It lands in "other operating income", a line most readers skip.

**Step 4: the money comes home.** Meridian pays the \$100m. It arrives at Harlow Brands, and is swept up to group treasury, where it replaces the \$100m that left in Step 1.

Now tally the group.

| | Amount |
| --- | --- |
| Cash out (Step 1) | (\$100m) |
| Cash in (Step 4) | \$100m |
| **Net group cash movement** | **\$0** |
| **Reported operating profit** | **+\$60m** |

Not one customer bought a sofa. Not one store opened. The group's bank balance is exactly where it started. And the income statement is \$60m better.

**The intuition: when a group both funds the buyer and books the gain, the profit is a bookkeeping artefact of the circle, and the giveaway is that no net cash ever entered the group.**

### Why the structure has to be complicated

Look again at what Step 1 through Step 4 required.

The \$100m had to leave the group and reach Meridian without any reader being able to follow it. If Harlow Finance had simply wired \$100m directly to Meridian and Meridian had wired \$100m straight back for the brands, a reasonably alert auditor tracing the cash would have closed the loop in an afternoon.

So the circle needs **length**. Ashfield sits in the middle for exactly this reason: it turns one traceable hop into two untraceable ones. Add a third intermediary, in a third jurisdiction, with a different financial year end and no public filings, and the loop is no longer traceable at all from the outside.

The circle also needs **plausible reasons for each hop**. "We advanced \$100m to a financing partner" is a sentence that survives a board meeting. "We paid the buyer's money" is not. Every leg must have a commercial story attached: supply prepayment, joint development funding, a property co-investment, a working-capital facility for a distribution partner.

And it needs **the assertion**. Somebody, inside the group, has to state that Meridian is not a related party and not controlled. That assertion goes into the accounts, gets relied upon by the auditor, and becomes the load-bearing beam of the whole structure.

This is what people mean when they say complexity is a red flag, and it is worth being precise about it, because the phrase is usually used lazily. Complexity is not a red flag because complicated things are suspicious. Plenty of legitimate multinationals are genuinely complicated: tax law, local ownership requirements, regulatory ring-fencing and acquisition history all generate real entities for real reasons.

Complexity is a red flag because **it is the raw material the circle is built from**. Every extra entity is another place a transaction can hide. Every extra jurisdiction is another place an auditor's letter goes unanswered. Every extra layer of holding company is another step between the reader and the cash. A group cannot run this scheme in a simple structure, so a group running this scheme must produce a complicated one. The complexity is not a coincidence and it is not innocent overhead. It is the machine's housing.

---

## What consolidation would have deleted

Here is the part most explanations skip, and it is the part that makes the fraud legible.

Everything in the worked example above is *legitimate accounting* if, and only if, Meridian is genuinely an unrelated third party. A company really can sell an asset to an outsider for more than its carrying value and book the gain. That is not fraud; that is a disposal.

The fraud is entirely in the classification of the counterparty. Change one fact, that Meridian is in truth controlled by or closely connected to people inside the group, and every number moves.

![One entity on the wrong side of the consolidation line was worth \$60m of profit and \$160m of assets.](/imgs/blogs/steinhoff-and-the-anatomy-of-a-global-accounting-collapse-4.webp)

#### Worked example: redrawing the consolidation line

Assume Harlow's underlying group, before any of this, reported:

```
Revenue                              $5,000m
Cost of sales                       ($3,900m)
Operating expenses                    ($850m)
Underlying operating profit            $250m
```

Now add the circle. As reported, with Meridian outside the line:

| Line | As reported |
| --- | --- |
| Revenue | \$5,000m |
| Cost of sales | (\$3,900m) |
| Operating expenses | (\$850m) |
| Other operating income | \$60m |
| **Operating profit** | **\$310m** |
| Receivable from Meridian | \$100m |
| Advance to Ashfield | \$100m |
| Brand portfolio | \$0m |

Now consolidate Meridian and Ashfield properly, because control, not the share register, is the test. Four eliminations follow mechanically:

1. **The gain goes.** Harlow sold the brands to itself. Intra-group profit on an asset transfer is eliminated in full: **-\$60m of other operating income**.
2. **The asset comes back at cost.** The brands return to the consolidated balance sheet at their original \$40m carrying value, not the \$100m Meridian "paid": **+\$40m of intangibles**.
3. **The receivable cancels.** Harlow Brands' \$100m receivable from Meridian meets Meridian's \$100m payable to Harlow Brands. The group does not owe itself money: **-\$100m of assets**.
4. **The advance cancels.** Harlow Finance's \$100m advance to Ashfield meets Ashfield's \$100m payable: **-\$100m of assets**.

| Line | As reported | Correctly consolidated | Difference |
| --- | --- | --- | --- |
| Revenue | \$5,000m | \$5,000m | 0 |
| Other operating income | \$60m | \$0m | (\$60m) |
| **Operating profit** | **\$310m** | **\$250m** | **(\$60m)** |
| Receivable from Meridian | \$100m | \$0m | (\$100m) |
| Advance to Ashfield | \$100m | \$0m | (\$100m) |
| Brand portfolio | \$0m | \$40m | +\$40m |
| **Effect on total assets** | | | **(\$160m)** |

Operating profit falls from \$310m to \$250m. That is a **19.4% overstatement** (\$60m / \$310m = 19.4% of the reported figure; equivalently the true number was overstated by 24%). Total assets fall by \$160m.

And note the direction of the asset effect, because it is counter-intuitive and it is the tell. The scheme *inflates* assets by more than it inflates profit. The \$60m of fake profit came with \$200m of fake assets attached, offset by \$40m of real asset that had been improperly removed. A group running this repeatedly does not just report profit it did not earn; it accumulates a balance sheet stuffed with receivables from, and advances to, entities that will never pay.

**The intuition: one entity on the wrong side of the consolidation line was worth \$60m of profit and \$160m of assets, and the assertion that put it there was a sentence, not a fact.**

### Why nobody catches it from the outside

You may be wondering why an investor cannot simply do what we just did.

Because the investor does not know Meridian exists. It is not named. It is not in the related-party note, by construction, because the whole scheme depends on it not being there. It does not appear in the subsidiary list, because it is not a subsidiary. The \$100m receivable from it sits inside a subtotal called "trade and other receivables" alongside \$2,400m of genuine amounts owed by real customers.

What the investor *can* see is the shape of the consequences. The gain shows up somewhere. The inflated assets show up somewhere. The absent cash shows up somewhere. The next three sections are the three places it shows up, and each one is a test you can run on a published annual report with nothing but a calculator.

---

## Test one: the balance sheet that results

Run the circle once and you get \$60m. Run it for eight years, in several currencies, against several assets, and the balance sheet starts to have a distinctive shape.

Now add the second engine, which almost always accompanies the first: **acquisitions**.

A group closing a gap with invented income has a problem: the invented income has to keep growing, and there is a limit to how much brand-sale profit you can plausibly book. Acquisitions solve several problems at once. They grow reported revenue and profit without organic improvement. They make year-on-year comparison impossible, because this year's numbers include six months of a business that was not there last year. They justify large one-off items that nobody scrutinises. And they generate goodwill, which is balance sheet capacity: somewhere to park value that nobody can independently price.

The result is a balance sheet like this.

![Goodwill and intangibles were 57.5% of the balance sheet and exceeded equity by \$1,300m.](/imgs/blogs/steinhoff-and-the-anatomy-of-a-global-accounting-collapse-5.webp)

#### Worked example: the goodwill test, and how much the answer moves

Harlow's consolidated balance sheet, after five years of acquisitions:

| Assets | | Liabilities and equity | |
| --- | --- | --- | --- |
| Goodwill | \$9,600m | Equity | \$12,500m |
| Other intangibles (brands) | \$4,200m | Borrowings | \$8,200m |
| Property, plant and equipment | \$3,200m | Payables | \$3,300m |
| Inventory | \$3,100m | | |
| Receivables | \$2,400m | | |
| Cash | \$1,500m | | |
| **Total assets** | **\$24,000m** | **Total** | **\$24,000m** |

Two numbers to compute, both of which take ten seconds.

**Ratio 1: goodwill and intangibles as a share of total assets.**

\$9,600m + \$4,200m = \$13,800m, against \$24,000m of total assets.

$$\frac{13{,}800}{24{,}000} = 0.575 = 57.5\%$$

Fifty-seven and a half percent of everything this group owns cannot be touched, counted, or sold to a stranger at a known price.

**Ratio 2: tangible equity.**

Equity of \$12,500m less \$13,800m of goodwill and intangibles:

$$\$12{,}500\text{m} - \$13{,}800\text{m} = -\$1{,}300\text{m}$$

Tangible equity is **negative \$1,300m**. Strip out the assets that exist only because management says so, and the group's liabilities exceed its remaining assets. For a retailer, which is a business made of buildings, stock and tills, this is a strange thing to be true.

Now the sensitivity, which is the part that shows you how little it takes to move.

The US furniture business is one cash-generating unit. Its carrying amount is \$4,000m, of which \$2,400m is goodwill. Management's value-in-use model forecasts next year's cash flow at \$260m, growing at 2% a year in perpetuity, discounted at 9%.

Value in use, as a growing perpetuity:

$$VIU = \frac{CF_1}{r - g} = \frac{\$260\text{m}}{0.09 - 0.02} = \frac{\$260\text{m}}{0.07} = \$3{,}714\text{m}$$

where ${CF_1}$ is next year's cash flow, ${r}$ is the discount rate and ${g}$ is the perpetual growth rate.

Carrying amount \$4,000m exceeds value in use \$3,714m, so the impairment is \$286m, charged against goodwill.

Now move the discount rate by 200 basis points, from 9% to 11%. A **basis point** is one hundredth of a percentage point, so 200 basis points is 2%.

$$VIU = \frac{\$260\text{m}}{0.11 - 0.02} = \frac{\$260\text{m}}{0.09} = \$2{,}889\text{m}$$

The impairment is now \$4,000m less \$2,889m, or **\$1,111m**.

| Discount rate | Value in use | Impairment | As % of group equity |
| --- | --- | --- | --- |
| 9% | \$3,714m | \$286m | 2.3% |
| 10% | \$3,250m | \$750m | 6.0% |
| 11% | \$2,889m | \$1,111m | 8.9% |
| 12% | \$2,600m | \$1,400m | 11.2% |

A single percentage point of discount rate, an assumption nobody outside the company can audit, moves the charge by roughly \$400m, which is around 3% of the group's entire equity. Across four percentage points the swing is \$1,114m.

**The intuition: when goodwill and intangibles are most of the balance sheet, the group's equity is not a fact about assets, it is the output of a discounted cash flow model whose inputs management chooses.**

This is also why an impairment, when it finally comes, tends to arrive as a catastrophe rather than a drip. Every year the model is defended is a year the gap between carrying value and reality widens. The charge is not the moment the value was lost. It is the moment the group stopped being able to argue.

---

## Test two: the cash test that never lies

Here is the good news, and it is genuinely good news, because it is what makes a reader's position less hopeless than it sounds.

Invented profit does not come with cash.

It cannot. Cash is the one number in a set of accounts that is confirmed by somebody outside the company: a bank. Profit is a judgement about when to recognise things. Cash is a balance somebody else holds. You can book a \$60m gain on a sale to a friend, but the friend's money came from you, so the group's net cash position did not move.

This is why the cash flow statement is the forensic accountant's first stop, and specifically the relationship between operating profit and **cash generated from operations**.

![Profit rose 54% over three years while operating cash flow went nowhere; the widening amber gap is exactly the non-cash income being booked.](/imgs/blogs/steinhoff-and-the-anatomy-of-a-global-accounting-collapse-6.webp)

One note on reading that chart: its vertical axis starts at \$200m rather than at zero, which is why the two lines look so far apart. Trust the plotted values and the gap figures, not the visual distance.

#### Worked example: cash conversion over three years

Harlow's reported numbers:

| | Year 1 | Year 2 | Year 3 | Cumulative |
| --- | --- | --- | --- | --- |
| Reported operating profit | \$250m | \$310m | \$385m | \$945m |
| Cash generated from operations | \$235m | \$240m | \$232m | \$707m |
| **Gap** | **\$15m** | **\$70m** | **\$153m** | **\$238m** |
| Cash conversion (OCF / operating profit) | 94% | 77% | 60% | 75% |

Reported operating profit grew 54% over three years, from \$250m to \$385m. Operating cash flow went from \$235m to \$232m, which is to say nowhere.

Now look at the "other operating income" line in the same three years: \$15m, then \$70m, then \$153m. Total \$238m.

The gap between profit and cash is \$238m. The other income is \$238m. They are the same number, because they are the same thing seen from two sides.

This is the whole test, and you can run it on any company in about ninety seconds:

$$\text{Cash conversion} = \frac{\text{Cash generated from operations}}{\text{Operating profit}}$$

For a healthy retailer this sits somewhere around 90% to 110% and is stable. Retail is a cash business: customers pay at the till, and depreciation (a non-cash charge) adds back, so cash conversion above 100% is normal in a chain that is not building many new stores. A number that starts near 90% and walks down to 60% over three consecutive years, while reported profit rises, is not a working-capital wobble. It is a message.

Two honest reasons cash conversion falls, which you must rule out before you conclude anything:

- **Genuine growth.** A fast-growing retailer buys inventory before it sells it, so working capital absorbs cash. Check whether inventory and receivables are growing roughly in line with revenue. If revenue is up 6% and receivables are up 40%, that is not growth absorbing cash.
- **A change in payment terms.** If the group started paying suppliers faster, payables fall and cash goes out. This shows up plainly in the working-capital movements in the cash flow statement, and it should reverse.

If neither explains it, ask the third question: **which line of profit is not turning into cash?** Find it in the income statement, and you will usually find you are looking at the scheme.

**The intuition: profit is an opinion and cash is a fact, so when the two separate and keep separating, believe the cash.**

---

## Test three: the segment page, and why it is the most useful page in the report

If I could keep only one test, it would be this one. It is the least known and the most powerful, because it uses the group's own disclosure against the group's own summary.

Recall from the foundations that IFRS 8 makes a listed group publish a profit measure for each reportable segment, and then reconcile the sum of those segments to the consolidated total. The reconciling line, "corporate and other", "unallocated", "central items", is where anything that does not belong to an operating business goes.

Which is exactly where invented profit has to go. A brand-disposal gain booked in a treasury or holding entity does not belong to the furniture segment or the discount-retail segment. It belongs to nothing, so it lands in the reconciliation.

#### Worked example: the segment reconciliation, three years running

Harlow's segment note, reproduced:

| Segment operating profit | Year 1 | Year 2 | Year 3 |
| --- | --- | --- | --- |
| Africa retail | \$120m | \$128m | \$132m |
| Europe household goods | \$95m | \$101m | \$96m |
| US furniture | \$40m | \$34m | \$21m |
| UK discount | \$18m | \$19m | \$15m |
| **Sum of reportable segments** | **\$273m** | **\$282m** | **\$264m** |
| Corporate and other (unallocated) | (\$23m) | \$28m | \$121m |
| **Consolidated operating profit** | **\$250m** | **\$310m** | **\$385m** |

Read the two bold lines against each other, because that comparison is the entire test.

The **operating businesses**, the actual shops and warehouses and lorries, went from \$273m to \$264m. They got 3% *worse* over three years. US furniture nearly halved.

The **group** went from \$250m to \$385m. It got 54% better.

Every cent of the improvement, and more, came from a line that represents no shops at all. "Corporate and other" swung from a \$23m cost, which is what a head office normally is, to a \$121m profit. A head office is a cost centre. When your head office becomes your best-performing division, you have found the thing worth asking about.

Three refinements that make this test sharper:

1. **Sum the segments yourself.** Groups do not usually print the subtotal. Add the column up with a calculator; the absence of a printed subtotal is itself mildly informative.
2. **Track the reconciling line as a percentage of consolidated profit.** Here: -9%, then 9%, then 31%. A line that goes from meaningfully negative to a third of group profit in three years is the story.
3. **Compare the number of reportable segments to the number of countries.** Harlow reports four segments across, say, thirty countries. That is not necessarily wrong, IFRS 8 follows how management actually runs the business, but it tells you how much aggregation stands between you and the underlying units. Four segments across thirty countries means a single segment number can hide a lot of divergence, and it means you cannot check any individual country against local filings.

**The intuition: the segment reconciliation is the group telling you, in its own numbers, how much of its profit came from no business in particular.**

---

## Test four: cash you cannot trace, and the confirmation problem

The cash test above assumes the cash number is real. Usually it is, because cash is confirmed by a third party. But "confirmed" hides a procedure, and the procedure has a weak point.

An auditor verifies a bank balance by sending a **bank confirmation**: a request, to the bank, asking it to state directly what the client holds. The control works because the answer comes from outside the client. It stops working when the request or the reply is routed through the client, or through a partner the client introduces, or through a jurisdiction where the auditor has no relationship with the bank and relies on a local correspondent firm.

That is not a hypothetical failure mode. It is what happened at Wirecard, and this series covers it in detail in [Wirecard: the missing EUR 1.9 billion](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros).

For a sprawling group, the analyst's version of this test does not require access to confirmations. It requires asking a geographic question:

**Where does the group say its cash is, and does that match where it says its sales are?**

A retailer's cash accumulates at the tills and moves to a national account. If a group reports 60% of revenue in Europe and 65% of cash in a treasury entity in a jurisdiction with no retail operations, that is not fraud on its own, cash pooling is a real and legitimate treasury practice, but it means the cash is one step further from anything you can check. Add the second question:

**Is the cash usable?** Groups disclose "restricted cash", "cash held in jurisdictions with exchange controls", and cash held at subsidiaries where a minority shareholder or a local regulator can block distribution. A group can report \$1,500m of cash of which several hundred million cannot be moved to pay group debt. The balance sheet says cash; the reality says somebody else's cash.

Practical version of this test, all from published documents:

- Compare cash by geography (in the segment note) with revenue by geography.
- Read the borrowings note for cash held at entities outside the guarantee group.
- Check whether net debt is calculated using cash that the group has just disclosed as restricted.
- Look at the interest received line in the cash flow statement, and divide it by average cash. If a group reports \$1,500m of cash and \$4m of interest income in a year when short rates were 4%, the cash was not there for the year, or it was not there at all.

That last one is a genuinely useful arithmetic check and it costs one division.

---

## Test five: the related-party note, read for what is missing

The related-party note under IAS 24 is the disclosure designed to catch precisely the transaction we built in the worked example. It fails in a specific and readable way.

It fails by **omission**, not by misstatement. A group running the circle does not write a false related-party note. It writes a *short* one, because it has asserted that the counterparties are not related, so there is nothing to disclose. The note will be honest about the things that were classified as related-party transactions: directors' remuneration, a joint venture, a pension fund, a property leased from a director's family trust. It will be silent about the \$100m advance, because that advance was classified as a transaction with an independent financing partner.

So read the note for scale, not content:

| What you check | What is normal | What is odd |
| --- | --- | --- |
| Length of the note vs group size | Grows with subsidiaries, geographies and joint ventures | A \$24,000m group with a one-page note listing only director pay |
| Named counterparties | Named entities you can look up | "A company in which a director has an interest", unnamed |
| Balances outstanding at year end | Small relative to the transaction value | Large balances outstanding, repeatedly rolled |
| Terms | Stated: interest rate, security, repayment | "On commercial terms", no terms given |
| Movement year to year | Roughly stable | Appears once, large, then vanishes |

And then read the **subsidiary list**, usually buried in the last pages of the annual report or filed separately. Count the entities. Count the jurisdictions. Note how many are described as holding companies or finance companies rather than operating companies, and note the ones incorporated in places with no operations at all. You are not looking for a smoking gun. You are measuring the size of the surface a scheme could hide on.

---

## Test six: changes in the frame

The last test is not about a number at all. It is about the *frame* the numbers are presented in, and it is the one that most often fires early.

When a group changes any of the following, the year-on-year comparison breaks, and a broken comparison is where a bad year goes to hide:

- **The auditor.** Especially a change to a firm with less international reach, or a change of the lead partner or the group audit firm's home jurisdiction. This series covers what to look for in [red flags in the audit report and auditor changes](/blog/trading/forensic-accounting/red-flags-in-the-audit-report-and-auditor-changes).
- **The financial year end.** A change of year end produces a stub period or an 18-month period, which is genuinely incomparable to anything.
- **The country of incorporation or the primary listing.** A move changes the accounting framework, the regulator, the disclosure regime and the set of local filings an outsider can retrieve.
- **The reporting currency.** A change from one currency to another makes every historical series require restatement, and restated history is history nobody checks.
- **The segment definitions.** Redrawing segment boundaries means last year's segment numbers are restated by the company, and the new boundaries can put a struggling unit inside a healthy one.

None of these is wrong on its own, and every one of them has boring legitimate reasons. Companies redomicile for tax and market access. Auditors get rotated by law. Groups genuinely reorganise.

The signal is in the **conjunction and the timing**. Two or three of these at once, in the same year, at a group whose cash conversion is deteriorating and whose unallocated segment profit is rising, is not a coincidence of housekeeping. It is a comparison being deliberately broken.


---

## What the record actually says

Everything so far has been mechanism, built on an illustrative company so the numbers stay visible. Now the real case. I am going to be careful here, because this part involves named people and legal proceedings, so each claim below carries its source and its date, and anything alleged rather than established is labelled as such.

### A group assembled by acquisition

Steinhoff began as a German furniture business and grew, through South Africa, into a retail group of genuinely unusual reach. Reviewing the group's 2016 position, CNBC Africa reported that it operated in more than 32 countries across four continents, with around 12,000 retail outlets, 26 manufacturing facilities, roughly 40 brands and about 130,000 employees (CNBC Africa, 28 June 2018).

The growth was overwhelmingly bought rather than built, and the pace in a single stretch of 2015 and 2016 is worth listing, because the pace is itself part of the story:

| Deal | Announced or agreed | Price as reported |
| --- | --- | --- |
| Pepkor Group (southern African discount retail) | 2015 | About R60 billion, roughly R15 billion in cash plus 839 million shares (CNBC Africa, 28 June 2018) |
| Kika-Leiner (Austrian furniture) | 2015 | Not stated in the sources used here |
| Poundland (UK single-price discount) | Agreed July 2016 | GBP 597 million (BBC, 13 July 2016), later raised (BBC, 11 August 2016), completed September 2016 (Daily Telegraph, 7 September 2016) |
| Mattress Firm (US specialty retail) | August 2016 | About \$3.8 billion (Fortune, reporting Reuters, 8 August 2016) |

Four large acquisitions in two years, on three continents, in four different retail formats: discount variety, furniture, single-price and specialty bedding. Set aside everything you know about what came later, and ask the question a forensic reader asks about any acquisitive group: **after that, what does a year-on-year comparison of this company actually mean?**

The answer is: not much. Each year's consolidated statements include a different collection of businesses from the year before. Revenue growth is not growth; it is arithmetic. Margin movement is not operating performance; it is mix. This is the non-comparability point from the balance-sheet section, and Steinhoff is close to a pure specimen of it. A reader in 2017 comparing 2016 to 2015 was not comparing like with like, and could not.

There was also an alteration to the frame in exactly the period the tests in this post would flag. In December 2015 the group moved its main listing to Frankfurt, where it traded on the German market while retaining its Johannesburg listing (IOL, reporting Bloomberg, 8 December 2015), with the group held through a Dutch-registered holding company, Steinhoff International Holdings N.V. A change of primary listing venue, a change of holding-company domicile and a run of transformational acquisitions all landed within roughly eighteen months of each other. Every one of those has an ordinary commercial justification. Together they are test six.

![The published structure showed operating pillars; the transacting structure ran to parties presented as independent third parties, outside the consolidation line but not outside the group's economics.](/imgs/blogs/steinhoff-and-the-anatomy-of-a-global-accounting-collapse-7.webp)

The right-hand panel of that figure runs ahead of the story on purpose. Those counterparty names and amounts come from the company's own published overview of the forensic investigation, which did not exist until March 2019 and which we come to shortly. In 2016 a reader could see the left-hand side of the picture only. That asymmetry is the point: the structure that was disclosed and the structure that was transacting were not the same shape, and only one of them was visible.

### December 2017

On 6 December 2017 the Mail & Guardian reported that the Steinhoff board had announced accounting irregularities requiring further investigation, and that chief executive Markus Jooste had resigned with immediate effect. It quoted the company saying that "new information has come to light today which relates to accounting irregularities requiring further investigation". Chairman Christo Wiese took over as interim executive chairman, and the board engaged PwC to investigate (Mail & Guardian, 6 December 2017).

A small note on the date, because precision matters more than tidiness. The announcement is commonly placed on the evening of 5 December 2017, with the share price reacting when markets opened on the 6th, and the reporting above is consistent with that. I have not been able to settle 5 versus 6 December against a primary document, so this post dates the event by the reporting rather than asserting a specific announcement date. Steinhoff's own forensic overview treats the 6th as the dividing line, referring to board members serving or employed "prior to 6th December 2017".

The market reaction was immediate and it was severe.

| | Value | Source |
| --- | --- | --- |
| Frankfurt share price, morning of 6 December 2017 | EUR 1.26, down 58%, about \$1.48 | Mail & Guardian, 6 December 2017 |
| Johannesburg share price, 6 December 2017 | Down more than 56% | Mail & Guardian, 6 December 2017 |
| Market value after the fall | About EUR 5.7 billion, roughly \$6.7 billion converting at about \$1.18 per euro | Mail & Guardian, 6 December 2017 |
| Share price on 23 May 2017, before any of this | R50.25, a market capitalisation of R240.5 billion, roughly \$18 billion converting at about R13.1 per dollar | CNBC Africa, 28 June 2018 |
| Decline within days of the announcement | About 85% | CNBC Africa, 28 June 2018 |
| Share price on 11 May 2018 | R1.60 | CNBC Africa, 28 June 2018 |

Those last two rows are the ones to sit with. A share at R50.25 in May 2017 was at R1.60 in May 2018: a fall of about 97% in under a year. The market capitalisation that went from R240.5 billion to a small fraction of it was, in dollar terms at the exchange rates of the day, roughly \$18 billion of value, most of it belonging to pension funds, index funds and ordinary savers who had never read a page of the annual report and had no reason to think they needed to.

Note also what the announcement itself did *not* contain. It did not say what the irregularities were, how large they were, or how far back they went. It said an investigation was required. The scale would take fifteen months to be published, and the details would take longer than that.

### What the board's own published overview said

On 15 March 2019, Steinhoff published an "Overview of Forensic Investigation", the public summary of PwC's work. It is the most important document in this case, and because it is the company's own publication rather than a press account, it is worth quoting from directly.

**It is an overview, not the report.** The overview states that the underlying report runs to "in excess of three thousand pages with over four thousand documents as annexures", that the work took 14 months, and that the findings are stated as at 28 February 2019. It records that Steinhoff "does not currently intend to publish the Report", citing confidentiality and legal privilege. That decision became the subject of its own litigation, in which it was argued that the company could not selectively disclose portions while withholding the rest (Mail & Guardian, 31 January 2022).

**The headline figure, to the euro.** The overview's Table 1 totals **EUR 6,506,596,428** of income from fictitious and/or irregular transactions recorded across the financial years 2009 to 2017. Call it **EUR 6.5 billion**, about **R100 billion** (Mail & Guardian, 31 January 2022), and approximately **\$7.3 billion** converting at about \$1.13 per euro, the rate prevailing around publication in March 2019. Reuters later described it as a "\$7 billion accounting fraud" (as reported via Moneycontrol, 13 August 2019), which is consistent.

The year-by-year shape is worth a moment, because it says the scheme was accelerating rather than winding down. The largest single years in the overview's table are **FY2016 at EUR 1,350,569,951**, **FY2015 at EUR 1,023,746,271** and **FY2017 at EUR 997,030,579**. Those three years alone are about EUR 3.37 billion, slightly more than half the eight-year total.

**Who it describes.** The overview's first finding reads:

> "A small group of Steinhoff Group former executives and other non Steinhoff executives, led by a senior management executive, structured and implemented various transactions over a number of years which had the result of substantially inflating the profit and asset values of the Steinhoff Group over an extended period."

Note what that sentence does and does not do. It says "a small group", it says "led by a senior management executive", and it does not name that executive. The overview names no individual as the leader. It also records that PwC interviewed or questioned 22 current and former directors and officers, and that Markus Jooste "and certain other individuals have not yet made themselves available for an interview".

**How the counterparties are characterised.** This is the sentence that makes the whole mechanism section of this post concrete:

> "Fictitious and/or irregular transactions were entered into with parties said to be, and made to appear to be, third party entities independent of the Steinhoff Group and its executives but which now appear to be closely related to and/or have strong indications of control by the same small group of people".

"Said to be, and made to appear to be, third party entities independent of the Steinhoff Group." That is the assertion from the foundations section, described by the company itself, and it is the load-bearing beam. Everything downstream of it was accounted for correctly given the assertion.

**And the documents existed.** The overview also records that supporting documents, "including legal documents and other professional opinions", were "in many instances, created after the fact and backdated". Keep that in mind for the misconceptions section below: the paper trail was not missing. It was manufactured.

**The counterparties are named, by the company.** Because Steinhoff published this itself, the principal counterparty groups are on the public record. The overview identifies three: **the Campion / Fulcrum Group**, **the Talgarth Group** and **the TG Group**. Table 1 attributes the income by counterparty, and the concentration is stark:

| Counterparty grouping (per the overview's Table 1) | Income from fictitious and/or irregular transactions |
| --- | --- |
| Talgarth Group (excluding Triton) | About EUR 4.16bn |
| TG Group | About EUR 1.02bn |
| GT Global Trademarks | About EUR 660m |
| Triton | About EUR 416m |
| Tulett Holdings | About EUR 169m |
| Group adjustments | About EUR 69m |
| SVF SA | About EUR 7.6m |
| Koenig | About EUR 3.5m |
| **Total** | **EUR 6,506,596,428** |

One grouping accounts for roughly 64% of the total. This is the concentration point from test five: a group of that size, transacting at that scale with a handful of counterparties, and the related-party note did not put them in front of the reader.

### The mechanism the overview describes, and the test it defeats

Now the part I have to be straight with you about, because it corrects the emphasis of one of my own tests.

I told you in test three that invented profit tends to land in the "corporate and other" reconciling line, because a gain booked at a treasury or holding entity does not belong to any operating segment. That is the usual signature, and it is why the segment reconciliation is such a good test in general.

Steinhoff, on the overview's own account, did the opposite. The finding reads that the fictitious or irregular income:

> "was, in many cases, created at an intermediary Steinhoff Group holding company level and then allocated to underperforming Steinhoff operating entities as so called 'contributions'".

Read that carefully. The income was manufactured at holding-company level, and then pushed **down** into operating entities that were not performing, dressed as "contributions". The effect is that the weak operating businesses did not look weak. The profit did not sit in an unallocated line waiting to be noticed; it was distributed into exactly the segments whose deterioration would otherwise have been the signal.

That is a scheme designed, whether deliberately or not, to defeat the test I gave you. And it is worth saying plainly: had you run the segment reconciliation on Steinhoff, it would not have lit up the way it lights up in the Harlow example.

So what would have fired? Two things.

**First, the operating units that improved without a reason.** If a business is receiving contributions from the centre to flatter its result, its margin improves without any operational explanation: no gross-margin gain, no cost programme, no volume growth, no store-count effect. The forensic question is not "did profit rise" but "which line made it rise, and is that line something a shop can do". The overview is specific that this was targeted: it records that neither Pepkor Europe, including Pepco and Poundland, nor Pepkor Holdings, nor the other South African operating entities were identified as having received such contributions. Some units were flattered and some were not, which means a comparison *between* the group's own units was informative.

**Second, and this is the point of the whole post: the cash test still fires.** Contributions allocated from a holding company do not create customer cash. Whichever segment the accounting entry lands in, the group's operating cash flow does not move, and the divergence between consolidated profit and consolidated cash generation is unaffected by how the profit was distributed internally. A scheme can choose which segment to hide in. It cannot choose to have the cash.

That is why, of the six tests, the cash conversion test is the one I would not trade for the other five.

Hold those three facts together, because their combination is the thesis of this post. Roughly EUR 6.5 billion of transactions, over eight years, in a group that published audited annual accounts every one of those years, in two listing jurisdictions, and none of it was visible as *itself* to anyone outside.

The scale of the correction that followed gives the same message from the other direction. Reuters reported in June 2018 that Steinhoff took a writedown of about \$12 billion following the accounting scandal (Reuters, 29 June 2018). In April 2018 the group announced that its Hemisphere international property portfolio was worth about EUR 1.1 billion, roughly half its previously stated value (CNBC Africa, 28 June 2018, dating the announcement to 4 April 2018). And in February 2018 a Dutch court ruled that Steinhoff had to amend its 2016 accounts (Reuters, 19 February 2018).

There was operational damage too, and it is worth naming because it shows the difference between the accounting and the businesses. Mattress Firm, bought for about \$3.8 billion in 2016, filed for Chapter 11 bankruptcy protection in the United States in October 2018 and emerged from it the following month (Bloomberg, 5 October 2018; Phoenix Business Journal, November 2018).

### The shape the reader could see

Here is the part that matters for your own reading, and it is worth stating plainly rather than dramatically.

Nobody outside the company could see the transactions. They were not disclosed as related-party dealings, because on the group's own assertion the counterparties were not related parties. That is precisely the failure mode described in the mechanism section: the related-party note fails by omission, and omission is invisible.

What was visible was the *shape*. Specifically:

**The frame kept changing.** Primary listing venue, holding-company domicile, and the constituent businesses of the group all changed inside about eighteen months. Any comparison a reader tried to make across those years was being reconstructed by the company rather than observed by the reader.

**The group was assembled faster than it could be understood.** More than 32 countries, roughly 40 brands, four retail formats, and four large acquisitions in two years. There is no reader, and realistically no audit team, that understands that group in the depth required within a single reporting cycle. Complexity of that magnitude does not merely hide things; it exhausts the people whose job is to look.

**The reach of the audit was necessarily partial.** A group spanning that many jurisdictions is audited through component auditors in each of them, with scoping decisions made in advance about which entities are material enough to examine. That is a structural limit, not an accusation, and it is the reason the misconception section above insists that "the auditors would have caught it" misunderstands what an audit is.

**And there were public warning signs before December 2017.** German prosecutors' interest in the group's accounting was reported publicly well before the collapse: Manager Magazin reported in August 2017 that the chief executive of the furniture group was under investigation by the authorities in connection with accounting matters (Manager Magazin, 24 August 2017). A reported criminal investigation into a listed group's accounting, more than three months before the group itself announced irregularities, is about as loud as a pre-collapse signal gets.

None of that is proof of fraud, and I want to be precise about the claim I am making. A reader running the six tests in 2017 would not have concluded "this company is committing a EUR 6.5 billion fraud". They would have concluded something weaker and much more useful: **that the reported numbers could not be checked, that the comparisons had been broken, that the group's structure exceeded what its operations required, and that a public authority was already asking questions.** That is a conclusion available from published documents, and it is a sufficient basis for a decision.


### The people and the proceedings

This is the part of the post that needs the most care, so the rule I am following is explicit: I state only what a named authority or court has said, I date it, I attribute it, and anything charged but not adjudicated is described as charged or alleged. Anything I could not verify against such a source has been left out rather than softened.

**The regulator's action against the company.** On 12 September 2019 the Financial Sector Conduct Authority, South Africa's market-conduct regulator, imposed an administrative penalty of **R1.5 billion** on Steinhoff International, roughly **\$100 million** converting at about R14.8 to the dollar in September 2019. The FSCA then remitted the great majority of it under section 173 of the Financial Sector Regulation Act, and the amount actually paid was **R53 million**, around 3.5% of the sum originally imposed (FSCA press release, 12 September 2019; the remittance is restated in the FSCA's press release of 20 March 2024, which notes it was paid before Steinhoff International became a new private Dutch holding company at the end of 2023).

**The insider-trading matter, and how long enforcement actually takes.** This one is instructive precisely because it did not go smoothly. On 30 October 2020 the FSCA imposed a penalty of **R161,568,068** on Markus Jooste, about **\$10 million** at roughly R16.3 to the dollar at the time, in relation to a warning message sent on 30 November 2017 encouraging four people to sell Steinhoff shares, under sections 78(4)(a) and 78(5) of the Financial Markets Act. The Financial Services Tribunal set that decision aside on 13 December 2021, finding no contravention of section 78(4)(a) and describing the message as vague and imprecise. The FSCA then re-imposed a penalty of **R20,000,000**, about **\$1.2 million** at about R17.4 to the dollar, on 7 December 2022, and the Tribunal subsequently dismissed Jooste's application for reconsideration, which the FSCA noted on 29 September 2023 (FSCA press releases, 30 October 2020, 7 December 2022 and 29 September 2023).

Three years, one reversal, one re-imposition and one appeal, over a single text message. Hold that against the idea that a market has fast, reliable consequences for misconduct.

**The financial-statement penalty.** On 20 March 2024 the FSCA imposed an administrative penalty of **R475 million** on Markus Jooste, roughly **\$25 million** converting at about R19 to the dollar in March 2024. The sum included a R10 million contribution to costs plus interest at 11.75%, and was payable on or before 19 April 2024. The FSCA found that Jooste, and also Dirk Schreiber, had made or published false, misleading or deceptive statements about the Steinhoff companies, contravening sections 81(1)(a) and (b) of the Financial Markets Act 19 of 2012, in respect of the annual financial statements and annual reports for the 2014 to 2016 financial years and for the 2017 half year (FSCA press release, 20 March 2024).

That last clause is worth pausing on. The regulator's finding attaches to **the published annual reports**, the very documents an investor was reading. In the same release the FSCA recorded that it had found Schreiber to have contravened the same sections, but that he had entered a leniency agreement under section 156(1) of the Financial Sector Regulation Act and no penalty was imposed on him.

**The death of Markus Jooste.** On 22 March 2024 the FSCA published a statement which, in its own words, "notes the various media reports on the death of former Steinhoff CEO, Mr Markus Jooste on Thursday 21 March 2024". The penalty described above had been issued on Wednesday 20 March 2024, one day earlier. I state both dates because both are on the regulator's own record, and I draw no inference from their proximity. I am not reporting a cause of death: I could not verify one against a police, prosecuting-authority or court statement, so it is not in this post.

In the same statement the FSCA said that his passing does not affect its ongoing Steinhoff investigation, since other investigated parties are involved, that it would continue to assist the Hawks and the National Prosecuting Authority, and that it is legally entitled to recover the penalty from his estate (FSCA press release, 22 March 2024).

**Germany: the Oldenburg proceedings.** Criminal proceedings against former Steinhoff managers ran in Germany, at the Landgericht (regional court) in Oldenburg. The first Steinhoff proceedings opened there in April 2023, and Jooste did not appear (JUVE, 19 April 2023).

On 22 August 2023 the court convicted two former Steinhoff managers. One, aged 52, was convicted on two counts of accounting fraud together with aiding credit fraud, and sentenced to three years and six months' imprisonment, with one year credited on account of the excessive length of the proceedings. The other, aged 64, his predecessor, was convicted on two counts of aiding the false presentation of financial statements and received a two-year suspended sentence. The court's findings concerned manipulations of about **EUR 1.2 billion**, roughly **\$1.3 billion** at about \$1.09 per euro in August 2023, and credit fraud of EUR 200 million and EUR 680 million (JUVE, 22 August 2023). German reporting convention does not name the convicted men in this kind of case and the source does not, so this post does not either.

As of 12 August 2026 I have no verified further outcome from those proceedings beyond the convictions of 22 August 2023. Anything else that may have been charged should be treated as charged or alleged and not adjudicated.

### The end state: what the shareholders received

The company did not recover, and the way it ended is the most concrete possible answer to the question "what happens to an equity holder in this situation".

The restructuring ran through the Dutch **WHOA** procedure, a court-supervised restructuring process under Dutch law. On the record of the successor entity's own annual report: Steinhoff International Holdings N.V. filed its restructuring plan request with the Amsterdam District Court on **31 May 2023**, the hearing was held on **15 June 2023**, the court confirmed the plan on **21 June 2023**, and it took effect on **30 June 2023** (Ibex Topco B.V. annual report, FY2023).

What the plan did to shareholders is the part to understand:

- Steinhoff's assets were transferred to **an unlisted new holding company**, and the entirety of the new structure's ownership was transferred to **five independent foundations**. The listed shareholders did not own the successor.
- Former shareholders received only **contingent value rights (CVRs)**: up to 20% of the total CVRs went to shareholders of record on **31 August 2023**, and up to 80% to affected creditors. The total number of CVR units was 21,348,045,255 (CVR Deed Poll executed 30 June 2023).
- A CVR pays out only from **residual value, if any, after repayment of external debt** on a liquidation.

That final bullet is the whole story of the equity. A CVR is not a share. It is a claim on whatever is left after every external lender has been paid in full, at some future winding-up, in a group whose debt was the reason the restructuring happened at all. Calling it "effectively wiped out" is not rhetoric; it is a description of where CVR holders sit in the queue.

An extraordinary general meeting of Steinhoff International Holdings N.V. in Amsterdam on **26 July 2023** voted in favour of the dissolution of the company. The Citizen reported on **9 October 2023** that the shares had been removed from the markets that week. The successor group operates as **Ibex** (Ibex Topco B.V.), which holds roughly 72.15% of Pepco Group N.V. along with Pepkor Holdings (Ibex Topco B.V. annual report, FY2023). The operating businesses, in other words, largely survived. The listed vehicle that had owned them did not, and neither did its shareholders' economic interest.

There was also a **global litigation settlement**, with a settlement effective date of **15 February 2022**. A Steinhoff Recovery Foundation was incorporated on 24 August 2021 to distribute the funds. Its recorded cost contributions include EUR 16.5 million from a Steinhoff entity, EUR 1.1 million from Deloitte and EUR 1.1 million from the directors' and officers' insurers, and its first distribution included additional contributions by the Deloitte firms and the D&O insurers (Ibex Topco B.V. annual report, FY2023). So Steinhoff's auditor did contribute to the settlement of claims. I have deliberately not put a headline figure on the total settlement pot or on Deloitte's total contribution, because I could not verify either against a primary document, and an unverified number in a post about fabricated numbers would be a poor joke.

### What this case is evidence of

It is worth being precise about the claim, because "Steinhoff was complicated and Steinhoff was a fraud" is not an argument.

The claim is narrower and more useful. A group's accounts rest on an assertion about which entities are inside it and who its counterparties are. That assertion cannot be verified from outside, and in this case the company's own published overview says the counterparties were "said to be, and made to appear to be, third party entities independent of the Steinhoff Group". The more entities and jurisdictions a group has, the more places such an assertion can be made and the less able anyone outside is to test any of them. That is what makes complexity a red flag: not that it is suspicious in itself, but that it is the material the concealment is built from, and it degrades every external check simultaneously.

And the corollary, which is the practical takeaway: since you cannot test the assertion, test the **consequences** of the assertion. They show up in the cash.

---

## The six tests as a working checklist

Everything above collapses into six questions you can answer from a published annual report, with no insider, no leak and no short-seller's dossier. None of them requires more than a calculator.

![Every one of these six tests runs on published statements alone, with no insider and no leak.](/imgs/blogs/steinhoff-and-the-anatomy-of-a-global-accounting-collapse-8.webp)

**1. Do the segments add up to the group?**
Sum the reportable segment profits yourself. Compare the trend in that sum to the trend in consolidated profit. If the operating businesses are flat or falling while the group is growing, the growth lives in the reconciling line, and the reconciling line represents no business at all.

**2. How much of the balance sheet is an opinion?**
Compute (goodwill + intangibles) / total assets, and compute equity minus (goodwill + intangibles). Above roughly 30% of assets is worth a second look for a retailer or an industrial; above 50%, or a negative tangible equity, means the group's book value is a discounted cash flow model with management's inputs.

**3. Does profit arrive as cash?**
Compute cash generated from operations divided by operating profit, for at least three consecutive years. Look at the trend, not the level. A walk down from 90% to 60% while profit rises is the single most reliable quantitative signal in this whole post, because it is the one thing the scheme cannot fake.

**4. Where is the cash, and can the group spend it?**
Compare cash by geography to revenue by geography. Read the restricted-cash disclosure. Divide interest received by average cash and see whether the implied rate is plausible for the year in question.

**5. Who is on the other side of the large transactions?**
Read the related-party note for scale rather than content. Then read the subsidiary list and count entities, jurisdictions, and the proportion described as finance or holding companies. You are measuring the surface area, not looking for a confession.

**6. Has the frame changed?**
Auditor, year end, domicile, listing venue, reporting currency, segment definitions. Any one is routine. Two or three in the same year, at a group already failing tests 1 to 3, is a comparison being deliberately broken.

A useful way to hold all six: **five of them are about the numbers and one is about the container**. A group can massage any single number. Making all six agree with each other, year after year, while the underlying business is not producing the profit, is very hard. That is why the failure shows up as a pattern rather than as one bad line.

---

## Common misconceptions

**"An audit would have caught this."**

An audit is designed to obtain reasonable assurance that the financial statements are free from material misstatement. It is not designed to detect a collusive fraud by senior management using entities the auditor does not know exist. The auditor tests the transactions the company shows it, against the documents the company provides, using representations the company signs. If the group asserts that a counterparty is independent, and produces a contract, an invoice and a bank statement showing the money arrived, the transaction looks like a disposal, because on the documents it is a disposal. The auditor's own standards acknowledge this: management override of controls is treated as a risk present in every audit precisely because it is the one the ordinary procedures do not reach. This series covers what an audit does and does not do in [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch).

**"The auditors must have been in on it."**

Sometimes audit work is genuinely poor, and regulators have said so in specific cases. But the more common and more useful explanation is structural. A group audit is performed by a network of separate national firms. The group auditor signs the opinion but relies on **component auditors** in other countries for the local entities. The group auditor's ability to reach into a component depends on materiality thresholds, on scoping decisions made before the work starts, and on the cooperation of a firm it does not employ. A transaction placed in a component below the scoping threshold, in a jurisdiction where the local firm is small, is not audited by the person who signs the opinion. That is not complicity. It is a design limitation, and it is one more reason complexity is not neutral.

**"If the profit was fake, the cash would obviously be missing."**

Not obviously, no. In the circle we built, the group's cash position is unchanged: \$100m left and \$100m came back. Nothing is missing from the bank. What is missing is the *increase* in cash you would expect from an extra \$60m of profit. That is a much subtler thing to notice, and it only becomes visible when you compare cash generation to profit over several years rather than looking at the cash balance in isolation. A group can run this scheme for a long time with a bank balance that reconciles perfectly every single night.

**"Complexity means the business is sophisticated."**

Multinational groups do accumulate real entities for real reasons: local ownership rules, regulatory ring-fencing, acquisition history, genuine tax structuring. But there is a difference between complexity that follows the operations and complexity that exceeds them. A retailer with 40 countries and 400 entities has roughly ten legal persons per country, and most of them are dormant acquisition leftovers. A retailer with 40 countries and several hundred entities of which a substantial share are finance and holding vehicles in jurisdictions with no shops has built something the operations did not require. The question is never "is this complicated". It is "does the complexity map onto anything that sells things".

**"You would need the forensic report to know."**

The forensic report tells you what happened and who did it, and it usually arrives more than a year after the share price has already moved. What the published statements tell you, in advance, is that something does not reconcile. Those are different questions, and only the second one is available to an investor in time to matter. Every test in this post is a test of internal consistency, and internal consistency is checkable from outside.

**"A big four auditor, a major-market listing and a large market capitalisation are themselves reassurance."**

They are reassurance about liquidity and about disclosure *volume*, not about disclosure *quality*. A large listing means more pages, more analysts and more index funds obliged to own the shares regardless of what the pages say. Index inclusion in particular creates a large body of shareholders whose mandate is to hold the stock because it is in the index, not because anybody read the segment note. Size is not evidence.

---

## How it shows up in real markets

The mechanism in this post is not one company's invention. It is a recurring shape, and it is worth seeing it in several forms, because the family resemblance is the point.

### Enron: the entity you do not consolidate

Enron's special purpose entities are the canonical version of the consolidation-boundary problem. The structures were used to hold assets and debt off the consolidated balance sheet and to book gains on transactions with counterparties that were, in substance, funded by Enron itself. The company filed for Chapter 11 bankruptcy protection on 2 December 2001. The mechanism, an entity kept outside the line so that transactions with it become real to the income statement, is the same mechanism as the circle above, dressed in derivative rather than retail clothing. This series covers it in [Enron: a forensic re-read of SPEs and mark to market](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market) and in [Enron and the 2001 accounting fraud](/blog/trading/finance/enron-2001-accounting-fraud).

### WorldCom: the simplest possible version

WorldCom did not need entities at all. It reclassified ordinary operating costs, principally line costs paid to other carriers, as capital expenditure, which moved them off the income statement and onto the balance sheet to be depreciated over years. The company filed for Chapter 11 on 21 July 2002. It is the useful counter-example to the "complexity is the tell" thesis: a fraud can be a single misclassification repeated. What WorldCom shares with the circle is the cash signature. Capitalised costs still leave the bank, so operating cash flow does not improve even as profit does. The cash conversion test fires on both. See [WorldCom: the \$11 billion capitalisation fraud](/blog/trading/forensic-accounting/worldcom-the-11-billion-dollar-capitalization-fraud).

### Wirecard: the cash that was confirmed by the wrong people

Wirecard's failure was a confirmation failure rather than a consolidation failure. Substantial balances said to be held in trust accounts with partners in Asia turned out not to exist, and the company's administrators and auditors were unable to locate roughly EUR 1.9 billion. Wirecard AG filed for insolvency in June 2020. What it shares with Steinhoff is the geography of the problem: the thing that could not be checked sat in a jurisdiction the group's own auditor could not easily reach, behind a partner the group introduced. See [Wirecard: the missing EUR 1.9 billion](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros).

### The pattern across all of them

Strip out the industry detail and three things are common to every case in this list.

**The profit was real on the documents.** In each case there were contracts, invoices, confirmations and board minutes. The fraud was not a false entry in a ledger. It was a true entry about a false relationship.

**The cash never followed.** Enron's operating cash flow was persistently weak relative to its reported earnings. WorldCom's capitalised line costs left the bank on the day they were incurred. Wirecard's reported cash was not spendable because it was not there. In every case, the cash flow statement was closer to the truth than the income statement, for the whole period of the fraud.

**The disclosure that would have shown it was published.** Not the fraud itself, which was hidden, but the *shape*: the reconciling items, the growth in intangibles, the divergence of profit and cash, the notes describing counterparties nobody could name. It was in the annual report, in the sections readers skip.

---

## When this matters to you

You are unlikely to be auditing a multinational. But the tests above are not really about fraud detection; they are about a general skill, which is checking whether a story is consistent with the numbers that are supposed to support it.

If you own shares, directly or through a fund, in any company that grows by acquisition, three of these tests take under ten minutes and are worth doing once a year: sum the segments, compute cash conversion for three years, and compute goodwill plus intangibles as a share of assets. You will run them on twenty companies and find nothing on nineteen. That is the correct hit rate. The point is not to find fraud; it is that on the twentieth you will have a specific, answerable question rather than a vague unease, and a specific question is something you can put to management, to a broker, or to the annual report itself.

If you work in or near finance, the transferable lesson is about assertions. Almost every serious accounting failure in this series rests on a statement about a *relationship* rather than a statement about an *amount*: this counterparty is independent, this entity is not controlled, this balance is unrestricted, this partner holds our cash. Amounts get tested. Relationships get asserted. When you find a number whose correctness depends on a relationship you cannot verify, you have found the part of the accounts that is load bearing.

And if you take one thing away: **the cash flow statement is the part of the accounts that somebody outside the company has to agree with**. Everything else is the company describing itself.

Where to go next in this series: [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing) is the disclosure regime this whole post turns on; [off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities) covers the consolidation boundary in depth; [goodwill, intangibles and the impairment timing game](/blog/trading/forensic-accounting/goodwill-intangibles-and-the-impairment-timing-game) goes further into the balance-sheet test; and [shell companies, reverse mergers and how fraud gets listed](/blog/trading/forensic-accounting/shell-companies-reverse-mergers-and-how-fraud-gets-listed) covers the entities on the other side of the transaction. For the ratio toolkit, see [forensic ratios: DSO, DIO, DPO and margin anomalies](/blog/trading/forensic-accounting/forensic-ratios-dso-dio-dpo-and-margin-anomalies) and [reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

This is educational material about how to read financial statements. It is not investment advice, and nothing here is a recommendation to buy or sell any security.


---

## Sources & further reading

Every figure in the Steinhoff sections above traces to one of the documents below. The Harlow Group numbers throughout the mechanism sections are illustrative arithmetic, chosen to be followable, and are labelled as such where they appear.

**Primary documents**

- Steinhoff International Holdings N.V., **"Overview of Forensic Investigation"**, 15 March 2019. The source for the EUR 6,506,596,428 total, the FY2009 to FY2017 period, the per-year and per-counterparty breakdown in Table 1, the named counterparty groupings, the "said to be, and made to appear to be, third party entities" characterisation, the "contributions" mechanism, the backdated-documents finding, and the statement that the full report was not to be published. Mirrored at [corruptionwatch.org.za](https://www.corruptionwatch.org.za/wp-content/uploads/2020/08/overview-of-forensic-investigation.pdf).
- **Ibex Topco B.V., annual financial statements FY2023**, at [ibexholdings.co.za](https://ibexholdings.co.za/downloads/2023/Ibex-Topco-BV-AFS-2023.pdf). The source for the WHOA timeline (filing 31 May 2023, hearing 15 June 2023, confirmation 21 June 2023, effective 30 June 2023), the transfer of ownership to five independent foundations, the CVR allocation, the 26 July 2023 dissolution vote, the Pepco and Pepkor holdings, and the Steinhoff Recovery Foundation contributions.
- **CVR Deed Poll**, executed 30 June 2023, at [ibexholdings.co.za](https://ibexholdings.co.za/downloads/2023/CVR-DEED.pdf). The source for the CVR terms and the total of 21,348,045,255 units.
- **Financial Sector Conduct Authority (South Africa) press releases**, at [fsca.co.za](https://www.fsca.co.za): 12 September 2019 (R1.5 billion penalty on Steinhoff International, later remitted to R53 million paid); 30 October 2020 (R161,568,068 insider-trading penalty); 7 December 2022 (re-imposed R20,000,000); 29 September 2023 (Tribunal dismisses the reconsideration application); 20 March 2024 (R475 million penalty, sections 81(1)(a) and (b) of the Financial Markets Act, covering the 2014 to 2016 annual reports and the 2017 half year, and the Schreiber leniency agreement); 22 March 2024 (statement on the implications of the passing of Mr Markus Jooste).

**Court and legal reporting**

- **JUVE**, 22 August 2023, on the Landgericht Oldenburg convictions of two former Steinhoff managers, the sentences, and the EUR 1.2 billion of manipulations: [juve.de](https://www.juve.de/verfahren/flick-gocke-und-grezesch-bachmann-mandanten-kassieren-haftstrafen/).
- **JUVE**, 19 April 2023, on the opening of the first Steinhoff proceedings at Oldenburg: [juve.de](https://www.juve.de/verfahren/erste-steinhoff-verfahren-starten-mit-hindernissen/).

**Dated press reporting**

- **Mail & Guardian**, 6 December 2017, for the announcement, the Jooste resignation, the 58% Frankfurt fall to EUR 1.26 and the EUR 5.7 billion market value: [mg.co.za](https://mg.co.za/article/2017-12-06-steinhoff-shares-plunge-60-as-jooste-quits-over-alleged-accounting-irregularities/).
- **Mail & Guardian**, 31 January 2022, for the 11-page overview, the roughly 3,000-page full report, the EUR 6.5 billion and R100 billion figures and the 2009 to 2017 period: [mg.co.za](https://mg.co.za/business/2022-01-31-steinhoff-cannot-cherry-pick-the-pwc-report-for-the-public/).
- **CNBC Africa**, 28 June 2018, for the group's 2016 scale (more than 32 countries, about 12,000 outlets, roughly 40 brands, about 130,000 employees), the Pepkor consideration, the R50.25 share price and R240.5 billion market capitalisation of 23 May 2017, the roughly 85% fall, the R1.60 price of 11 May 2018 and the Hemisphere property revaluation of 4 April 2018: [cnbcafrica.com](https://www.cnbcafrica.com/2018/steinhoff-rise-fall/).
- **Financial Times**, 15 March 2019, "Ex-Steinhoff executives used EUR 6.5bn in fake transactions, report says".
- **Reuters**, 19 February 2018 (Dutch court ruling on the 2016 accounts) and 29 June 2018 (the approximately \$12 billion writedown); and Reuters via Moneycontrol, 13 August 2019, for the "\$7 billion accounting fraud" characterisation.
- **IOL**, reporting Bloomberg, 8 December 2015, on the move of the primary listing to Frankfurt.
- **Manager Magazin**, 24 August 2017, reporting that the chief executive was under investigation by the authorities in connection with accounting matters, more than three months before the company's own announcement.
- **BBC**, 13 July 2016 and 11 August 2016, and **Daily Telegraph**, 7 September 2016, on the Poundland acquisition; **Fortune**, reporting Reuters, 8 August 2016, on Mattress Firm at about \$3.8 billion; **Bloomberg**, 5 October 2018, on the Mattress Firm Chapter 11 filing.
- **The Citizen**, 9 October 2023, reporting that the shares had been removed from the markets that week.

**Accounting standards referred to**

IFRS 10 (consolidated financial statements and the control definition), IAS 24 (related-party disclosures), IFRS 8 (operating segments and the reconciliation to consolidated totals), and IAS 36 (impairment of assets, including the value-in-use test).

**Claims deliberately left out.** An unverified number in a post about fabricated numbers would be indefensible, so several things a reader might expect to find here are absent because I could not get them from a primary source:

- **A cause of death for Markus Jooste, and the existence or date of any warrant for his arrest.** Neither is stated in this post.
- **Headline totals for the global litigation settlement pot, and Deloitte's overall contribution to it.** Note carefully that the EUR 1.1 million from Deloitte cited above is a contribution to the Steinhoff Recovery Foundation's *operating costs*, not a settlement contribution. The foundation's records confirm that additional contributions were made by the Deloitte firms and the D&O insurers to the first distribution but do not quantify them, so no aggregate appears here.
- **Steinhoff's FY2016 revenue and net profit, and goodwill and intangibles as a share of total assets.** I wanted these badly, since a pre-collapse balance-sheet ratio would have been the sharpest possible illustration of test two. The relevant annual reports exist in the Internet Archive but could not be retrieved, and the company's own domain no longer resolves. Rather than run a secondary source's figures as though they were filed accounts, the post makes the balance-sheet and margin arguments with the illustrative Harlow numbers and attaches no Steinhoff ratio to them.
- **Exact JSE and Frankfurt delisting dates.** The post says only what The Citizen reported on 9 October 2023.
- **The precise date of the December 2017 announcement.** The event is dated here by the reporting, not asserted as 5 or 6 December, for the reason given in the text.

**Related posts in this series**

The mechanics behind each test: [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing), [off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities), [goodwill, intangibles and the impairment timing game](/blog/trading/forensic-accounting/goodwill-intangibles-and-the-impairment-timing-game), [shell companies, reverse mergers and how fraud gets listed](/blog/trading/forensic-accounting/shell-companies-reverse-mergers-and-how-fraud-gets-listed), [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue), [the footnotes and MD&A, where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) and [transfer pricing and offshore profit shifting](/blog/trading/forensic-accounting/transfer-pricing-and-offshore-profit-shifting).

Beyond this series: [quality of earnings: accruals, one-offs and red flags](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) applies the same lens inside an equity-research workflow, and [narrative addiction: when a good story beats the data](/blog/trading/analyst-edge/narrative-addiction-when-a-good-story-beats-the-data) is about why a growth story this loud kept being believed for eight years.
