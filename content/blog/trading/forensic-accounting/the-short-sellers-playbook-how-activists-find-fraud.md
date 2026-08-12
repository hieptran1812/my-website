---
title: "The short seller's playbook: how activists find fraud"
date: "2026-08-12"
publishDate: "2026-08-12"
description: "The research process behind a published fraud thesis, stage by stage: screening, ground-truthing revenue in the physical world, mapping ownership through corporate registries, cross-reading filings, tracing cash, and sizing a position that can lose more than it can win."
tags: ["forensic-accounting", "short-selling", "activist-short-sellers", "financial-statement-fraud", "due-diligence", "channel-checks", "related-party-transactions", "securities-lending", "position-sizing", "fraud-detection"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 62
depth: "deep-dive"
---

> [!important]
> **TL;DR:** An activist short thesis is not a hunch about valuation. It is a research funnel that tries to answer one physical question, "is this business real?", using evidence the company did not write.
>
> - The funnel has six stages, and each one costs more than the last: screen, ground-truth, map the ownership, cross-read the filings, trace the cash, then size and publish. Most theses are supposed to die in the first two stages.
> - Screens do not find fraud. Fraud is rare, so even a good screen leaves you with a candidate list that is mostly false positives: a screen that catches 90% of frauds and clears 90% of honest companies still leaves you only about a 4% chance of being right when it flags a company.
> - The strongest evidence in any short report is a document the company did not write, or two documents it did write that cannot both be true.
> - The trade is structurally hostile: your maximum gain is the size of the position, your loss has no ceiling, and you pay rent on the borrow every day you wait. A 25% annual borrow fee held for nine months moves the breakeven on a \$50.00 short down to \$40.62, so the stock has to fall 19% before you have made anything.
> - Short sellers are also wrong, and sometimes abusively so. The honest version of this playbook includes the failure modes, the incentives, and the cases where the company was telling the truth.

Most investment research asks what a company is worth. A fraud short asks something smaller, harder, and much more answerable: *is this company real?*

That difference is the whole discipline. "Worth" is an opinion about the future, and reasonable people can hold opposite ones forever. "Real" is a claim about the present physical world. If a restaurant chain says it served four hundred million cups of coffee last year, somebody poured them, somewhere, into cups, in stores that exist, and the electricity bill, the milk deliveries, the payroll taxes and the queue at 8:15 on a Tuesday all leave traces. Reported revenue makes physical promises. Physical promises can be checked.

![The six stages of a fraud short thesis, arranged as a funnel where each stage costs more than the last](/imgs/blogs/the-short-sellers-playbook-how-activists-find-fraud-1.webp)

The diagram above is the mental model for everything that follows. A fraud short is a funnel with six stages, ordered so that the cheap stages run first and kill most candidates before you have spent anything. Screen the numbers for something that does not fit. Ground-truth the business against the physical world. Map who actually owns the counterparties. Cross-read the company's own documents against each other. Trace whether the cash behaves like cash that exists. Only then size the position, arrange the borrow, and decide whether to publish. The great majority of ideas die at stage one or two, and that is the process working, not failing.

This post reconstructs that funnel as something you can run. Not at the scale of a firm with a field team in three countries, but at the scale of a person with an internet connection, a spreadsheet, and a company in their portfolio they have started to worry about. The last section is explicitly about that scaled-down version. Along the way we will walk a published short report end to end as a specimen, and we will spend real time on the part most write-ups skip: the cases where the short seller was wrong, the incentives that make being wrong profitable for a while, and why that matters for how you read any short report you are handed.

Nothing here is investment advice. It is a description of how a particular kind of research is done, including how it fails.

## Foundations: what a short seller is and how the trade actually works

If you have never sold anything short, start here. Everything after this section assumes these terms, and each one is defined from zero.

### Selling something you do not own

Ordinary investing is *long*: you buy a share at one price and hope to sell it higher. Short selling reverses the order. You sell first at one price and buy later at a lower one, and you pocket the difference.

The obvious problem is that you cannot sell what you do not have. The market solves this with a loan. A *securities lending* desk, usually inside your broker, finds an institution that owns the shares and is willing to lend them: a pension fund, an index fund, an insurer. These holders lend because they earn a fee for doing so and they were not planning to sell anyway. You borrow the shares, sell them into the market, and now you owe someone shares rather than money. To close the position you buy the shares back on the open market and hand them to the lender. This is called *covering*.

#### Worked example: the smallest possible short

You believe a share trading at \$50.00 is worth less. You borrow 100 shares from your broker and sell them.

1. You sell 100 shares at \$50.00 and receive \$5,000 in cash. You now owe 100 shares.
2. Three months later the share trades at \$30.00. You buy 100 shares for \$3,000 and return them to the lender.
3. Your gross profit is \$5,000 minus \$3,000, which is \$2,000.

Now run it the other way. The share goes to \$80.00 instead. You buy 100 shares back for \$8,000 against the \$5,000 you received, and you lose \$3,000. Notice the asymmetry, because it governs everything else in this post: the best case is that the share goes to zero and you keep the full \$5,000, while the worst case has no limit at all, because there is no ceiling on a share price. A long position can lose 100% of what you put in. A short position can lose 300% or 900%.

The one-sentence intuition: a short caps your upside at the size of the position and leaves your downside open.

### The borrow, the locate, and the fee

You cannot simply decide to sell shares you do not have. In the United States, Rule 203(b)(1) of Regulation SHO (17 CFR 242.203) says a broker may not accept or effect a short sale unless it has "borrowed the security, or entered into a bona-fide arrangement to borrow the security", or has "reasonable grounds to believe that the security can be borrowed so that it can be delivered on the date delivery is due", and has documented that compliance. This is the *locate* requirement. Market makers doing bona fide market making are excepted under 203(b)(2)(iii).

Borrowing is not free. The lender charges a fee, quoted as an annualised percentage of the position's market value. For a large, widely held stock that everybody owns, the fee is tiny, a few basis points a year, where a *basis point* is one hundredth of a percent. These are called *general collateral*. For a stock where lots of people want to be short and few shares are available to lend, the fee can run to double-digit percentages a year. These are *hard to borrow* or *special*. The fee is not fixed for the life of your trade, either. It floats, and it tends to rise exactly when your thesis is becoming popular, which is exactly when you least want to pay more.

There is a second, sharper risk hiding in the loan. The lender can *recall* the shares. If the fund that lent them decides to sell, or moves the position to a different custodian, your broker has to find replacement shares or buy you in, which means closing your position at the market price on someone else's schedule. A forced buy-in during a rally is one of the ordinary ways a correct short thesis loses money.

### Margin: the money you must keep in the account

Because the loss is open-ended, a short position is a credit exposure and the rules treat it that way. Two separate requirements apply in a US margin account.

The *initial* requirement comes from the Federal Reserve's Regulation T. Under 12 CFR 220.12(c)(1), a short sale of a nonexempted equity security requires 150 percent of the current market value of the security. The proceeds of the sale, which stay in the account as collateral, count for 100 of that 150, so in practice you post your own equity equal to 50% of the position.

The *maintenance* requirement, which applies continuously afterwards, comes from FINRA Rule 4210(c). For a stock short in the account selling at \$5.00 per share or above, subparagraph (c)(3) requires "\$5.00 per share or 30 percent of the current market value, whichever amount is greater". For a stock selling at less than \$5.00 per share, subparagraph (c)(2) requires "\$2.50 per share or 100 percent of the current market value, whichever amount is greater".

Read that second one again, because it produces one of the strangest facts in short selling: **when your thesis works spectacularly and the stock falls below \$5.00, your maintenance requirement per share goes up, not down.** A stock at \$4.00 requires \$4.00 per share of maintenance rather than \$1.20. Being right can tie up more of your capital, not less.

### The vocabulary of crowding

Four numbers describe how crowded a short is, and every one of them will matter when we size a position.

- **Shares outstanding**: every share the company has issued.
- **Free float**: the shares actually available to trade. Founder stakes, strategic holders, employee shares still in lock-up and government holdings are all outstanding but not floating. This is the number that matters, and it is often far smaller than shares outstanding.
- **Short interest**: the number of shares currently sold short. In the United States, FINRA collects short interest positions from member firms on a periodic schedule and publishes them, so the figure you see is a snapshot with a lag rather than a live number. Check the current reporting and publication schedule at the source before treating any short-interest figure as current or complete.
- **Days to cover**, also called the short interest ratio: short interest divided by average daily trading volume. If 24 million shares are short and the stock trades 4 million shares a day, days to cover is 6. That is a rough estimate of how many days of *all* the trading in the stock it would take for every short to get out. It is the single best warning sign for a squeeze.

### Four kinds of short, and only one of them is this post

The word "short seller" covers people doing genuinely different jobs.

| Kind | The claim | How it resolves | Time horizon |
| --- | --- | --- | --- |
| Valuation short | The price is too high for the business | Multiple compression, often never | Years, or never |
| Cyclical short | Earnings are about to fall | The next few earnings reports | Quarters |
| Structural short | The business model is being destroyed | Slow erosion of revenue | Years |
| Fraud short | The reported numbers are not real | A restatement, a resignation, a halt, an enforcement action | Unknowable, then sudden |

Only the fourth is what this playbook is about. It is the one with an objective answer, and it is the one where research effort translates most directly into edge, because the evidence exists and almost nobody has gone to look for it. It is also the one where being early is indistinguishable from being wrong for an uncomfortably long time.

### Activist versus quiet, and why anyone publishes

A short seller can simply put on the position and wait. An *activist* short seller publishes the research: a report, usually free, usually on the firm's own website, usually explicitly disclosing that the author is short and stands to profit if the price falls.

Publishing is a strange choice, and it is worth being honest about why anyone does it. Three reasons, in descending order of how flattering they are.

1. **It shortens the horizon.** A quiet short waits for the market to discover what you already know, and pays borrow the whole time. A published short tries to force the discovery. Given that borrow can cost tens of percent a year, this is not a small consideration. It is the main one.
2. **It invites contradiction.** A good report is a public falsification test. If the company can produce the bank confirmation, the land title, the customs record, you were wrong and you find out fast. This is a real epistemic benefit and serious practitioners use it that way.
3. **It creates its own catalyst.** A report can trigger an auditor's questions, a special committee, a regulator's inquiry, a lender's covenant review. Sometimes that is the mechanism by which a genuine fraud is exposed. Sometimes it is the mechanism by which a solvent company is pushed into a liquidity crisis it would otherwise have survived. Both happen, and the short seller does not get to choose which.

That third reason is the ethical core of the whole activity, and we come back to it at length. Publishing a report is an exercise of power over a company's cost of capital, and the person exercising it is paid according to how much the price falls.

### Why short sellers find frauds that other people miss

It is worth asking why this work falls to a group of people with an obvious financial interest in the answer, rather than to auditors, regulators or analysts.

The structural answer is incentives. An auditor is paid by the company and its engagement is scoped to whether the statements are fairly presented in accordance with the framework, not to whether management is lying in a way the evidence provided does not reveal. The series post on [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) covers that gap in detail. A sell-side analyst needs continued access to management and works at a firm that would like the company's banking business. A regulator has a caseload, a budget, and a duty of process. None of them are rewarded for going to stand in a car park counting cars.

A short seller is rewarded for exactly that, and only for that. The reward is contingent on being right, which is the good part, and on the price falling, which is the part that creates every problem in the second half of this post.

## Stage 1: the screen, and why it is only a place to start

The first stage is cheap, fast, and almost entirely about generating candidates rather than conclusions. You are looking for a company where the reported numbers have a shape that honest businesses rarely have.

The tools here are the subject of most of this series, so this section is a map rather than a repeat.

- **Accruals**: the gap between reported profit and cash. Persistent, growing accruals mean the earnings are increasingly made of estimates rather than money. See [the accruals ratio and the accruals anomaly](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly) and [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits).
- **Working capital ratios**: receivables growing faster than revenue (days sales outstanding rising), inventory growing faster than cost of sales (days inventory outstanding rising), payables stretched to fund it. [Forensic ratios: DSO, DIO, DPO and margin anomalies](/blog/trading/forensic-accounting/forensic-ratios-dso-dio-dpo-and-margin-anomalies) walks the arithmetic.
- **Composite scores**: the [Beneish M-score](/blog/trading/forensic-accounting/the-beneish-m-score-detecting-earnings-manipulation) for manipulation likelihood, the [Altman Z-score](/blog/trading/forensic-accounting/the-altman-z-score-predicting-financial-distress) for distress, [Benford's law digit analysis](/blog/trading/forensic-accounting/benfords-law-and-digit-analysis-for-fraud) on reported figures.
- **Governance events**: an auditor resignation or a change to a much smaller firm, CFO turnover, a delayed filing, a late 10-K, a material weakness disclosure, a director resigning with a pointed letter. [Red flags in the audit report and auditor changes](/blog/trading/forensic-accounting/red-flags-in-the-audit-report-and-auditor-changes) covers what each one signals.
- **Peer-relative anomalies**: the highest margin in an industry where everyone else earns half as much, with no identifiable reason.

![The evidence ladder, from screens that cost an hour and prove nothing to a physical count that costs three months and settles the question](/imgs/blogs/the-short-sellers-playbook-how-activists-find-fraud-2.webp)

The ladder in the figure is the thing to internalise. A screen sits on the bottom rung. It is worth an hour and it proves nothing. Every rung above it costs more and carries more weight, and the two rungs that actually settle arguments are the ones the company did not write: third-party records, and a physical count.

Here is the arithmetic reason a screen cannot be a thesis.

#### Worked example: what a red flag is actually worth

Suppose serious financial statement fraud affects roughly 1 company in 200 in a given year. That is a prior probability of 0.5%. Suppose you have an unusually good screen: it flags 90% of the companies that really are committing fraud (sensitivity 0.90) and correctly clears 90% of the honest ones (specificity 0.90, so a 10% false positive rate).

Run 10,000 companies through it.

1. 50 of the 10,000 are frauds. The screen flags 90% of them: 45 true positives.
2. 9,950 are honest. The screen wrongly flags 10% of them: 995 false positives.
3. Total flags: 45 + 995 = 1,040.
4. Probability that a flagged company is a fraud: 45 / 1,040 = **4.3%**.

A screen that is right nine times out of ten in both directions still hands you a list where 19 out of every 20 names are innocent. If you shorted every flag you would lose money on the overwhelming majority of them, and you would pay borrow on all of them the whole time.

The intuition: because fraud is rare, a flag is not evidence of fraud. It is a reason to spend one more day. Its entire value is in what it lets you *stop* looking at.

That is why the funnel exists. Stage one is a filter for where to spend stage two's money, and nothing more. If your research process ends at a screen, you do not have a thesis, you have a spreadsheet.

## Stage 2: ground-truthing the business against the physical world

This is the stage that distinguishes a fraud short from ordinary skeptical analysis, and it is the one an individual investor most underrates.

The technique is simple to state. Take the reported number. Convert it into a physical quantity that must be true if the number is true. Then go and measure the physical quantity.

Reported revenue implies cups poured, boxes shipped, hectares planted, trucks delivered, subscriptions billed, kilowatt-hours consumed, employees paid. Every one of those has a footprint outside the company's control:

| Reported claim | The physical quantity it implies | Where you can measure it |
| --- | --- | --- |
| Retail revenue | Transactions per store per day | Store visits, receipt counts, queue timing, till numbers on receipts |
| Store count | Buildings that exist and are open | Map services, delivery-app store lists, local business registries, site visits |
| Manufacturing revenue | Units shipped, inputs consumed | Customs and bill-of-lading records, power consumption, supplier disclosures |
| Agricultural or resource assets | Land, standing crop, ore in the ground | Land registries, satellite imagery, independent survey, local title records |
| App or platform revenue | Active users, sessions, transactions | App-store rank panels, third-party download and usage panels, web traffic |
| Headcount and payroll | People on a payroll somewhere | Social insurance filings, job postings, professional-network headcounts |
| Logistics volume | Vehicles moving | Parking-lot and yard satellite counts, port call data, toll and weigh-station data |

Some of these are cheap and available to anyone. Delivery-app store listings, satellite imagery in a free map service, job postings and customs databases are all a browser away. Some are expensive: a firm that sends people to sit in 1,600 store-hours of surveillance is spending real money, and that is the point at which this becomes a professional activity.

Here is the arithmetic that turns a store visit into a claim.

#### Worked example: implied revenue per store versus what you can count

A retail chain reports annual revenue of \$1.44 billion and says it operated an average of 4,000 stores during the year. Its disclosed average ticket, the amount a typical customer spends per visit, is \$4.00.

Start with what the filings force to be true:

1. Revenue per store per year: \$1,440,000,000 / 4,000 = **\$360,000**.
2. Revenue per store per day: \$360,000 / 365 = **\$986**.
3. Items sold per store per day: \$986 / \$4.00 = **247**.

So the accounts assert that the average store sells roughly 247 items a day, every day, including Mondays in February. That is now a testable physical claim.

Now go and count. Suppose you cover 12 stores for 10 hours each, recording every transaction, and the average across those 12 stores works out to **160 items per day** once you extrapolate the observed hours to the store's full opening hours.

The gap is large, but a sample of 12 out of 4,000 stores is 0.3% of the estate, so you have to ask whether 160 could just be sampling noise around a true 247.

4. Suppose the standard deviation of daily items across your 12 stores is 60. The standard error of the mean is 60 / √12 = **17.3**.
5. A 95% confidence interval around your observed 160 is 160 ± (1.96 × 17.3), which is roughly **126 to 194**.
6. The implied 247 sits (247 − 160) / 17.3 = **5.0 standard errors** above your sample mean.

![Implied items per store per day versus what a field count actually observed, with the confidence interval on the count](/imgs/blogs/the-short-sellers-playbook-how-activists-find-fraud-3.webp)

7. If the true average is 160 rather than 247, revenue is overstated by 1 − (160 / 247) = **35%**, or about **\$0.5 billion** of the reported \$1.44 billion.

The intuition: you cannot count 4,000 stores, but you do not have to. You only have to show that the reported number is many standard errors away from anything you can observe, and then hand the company the burden of explaining the gap.

### The four ways this goes wrong

Ground-truthing is the most persuasive evidence in a short report and also the easiest to do badly. Four failure modes, all of which have appeared in real reports.

**Selection bias in the sample.** If you pick the 12 stores that look quiet, you will find they are quiet. The discipline is to select the sample before you observe it, using a rule you write down first: every store in one postcode, or a random draw from the company's own published store list. Serious reports state their selection rule. If a report does not tell you how it chose its sample, you cannot evaluate its central number.

**Seasonality and time of day.** Ten hours on a Tuesday in a business district is not a random ten hours. A single-week count in a category with strong seasonality can be off by a third in either direction with no fraud involved.

**Measuring the wrong quantity.** Counting customers is not counting revenue if the mix has changed, delivery orders do not walk through the door, and a company that shifted to corporate bulk sales will look dead at the counter while its revenue is real. The rebuttal "you counted retail footfall in a business that is now 40% wholesale" has ended more than one short thesis, correctly.

**Panel bias in alternative data.** Third-party app-usage and card-spending panels are samples of a particular population, skewed by geography, device, and demographic. They are excellent for *changes* and unreliable for *levels*. Treating a panel's absolute number as the company's absolute number is a category error.

The honest way to hold all of this: field evidence is strong against a *large* claimed number and weak against a small one. It works when the gap is 35%, not when it is 5%.

## Stage 3: mapping the ownership behind the counterparties

Stage two asks whether the business is real. Stage three asks whether the *counterparties* are real, and whether they are actually the company itself wearing a different name.

This matters because the most common way to fabricate revenue is to buy it. Money leaves the company as a payment for something, travels through one or more entities, and comes back as a customer receipt. Reported revenue and profit rise. Cash does not, or does so only briefly. The series post on [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) covers the accounting mechanics; this section covers how you find out.

The instrument is the corporate registry, and the useful fact is that registries are public in most places, cheap, and almost never consulted.

| Jurisdiction | Registry | What it gives you |
| --- | --- | --- |
| United States | State secretary of state filings, SEC EDGAR | Incorporation date, registered agent, officers, filed exhibits |
| United Kingdom | Companies House | Free full filing history, directors, shareholders, charges, accounts |
| China | National Enterprise Credit Information Publicity System, formerly the SAIC and local AIC filings | Registered capital, shareholders, legal representative, annual reports filed locally |
| Hong Kong | Companies Registry (ICRIS) | Directors, share capital, annual returns |
| Singapore | ACRA | Officers, shareholdings, filed accounts |
| Cross-border | OpenCorporates, national beneficial-ownership registers where they exist | Entity matching across jurisdictions |
| Everywhere | Court dockets, land registries, trademark and patent filings, UCC or charge filings, tender and procurement records | Sworn statements, security interests, addresses, real assets |

What you are looking for is not a smoking gun. It is coincidence that is too expensive to be coincidence.

- The largest customer and the largest supplier share a registered address.
- A "customer" was incorporated two months before it placed its first order.
- The registered capital of an entity is a rounding error against the revenue attributed to it.
- The same natural person is the legal representative of four counterparties in three provinces.
- A director of a counterparty sits on the issuer's audit committee, or is the spouse or sibling of an officer.
- The phone number on a supplier's registry filing is the issuer's switchboard.

![A registry search turning a disclosed customer and a disclosed supplier into the same undisclosed party](/imgs/blogs/the-short-sellers-playbook-how-activists-find-fraud-5.webp)

#### Worked example: the customer that is too small to be the customer

An issuer's filings disclose that its largest customer accounted for 31% of revenue, or **\$450 million**, and its largest supplier accounted for 28% of cost of goods sold, or **\$280 million**. Both are named. Neither is disclosed as a related party.

You pull both from the local registry. Three facts come back.

1. The customer has registered capital of RMB 1 million. At an illustrative 7.2 yuan per dollar that is about **\$139,000**. It is being credited with \$450 million of purchases, or roughly **3,200 times** its registered capital.
2. The customer was incorporated four months before the issuer's first disclosed sale to it, and its filed annual report lists **3 employees**.
3. A single holding company appears as a 70% shareholder of the customer and a 55% shareholder of the supplier, and its registered address matches the issuer's operating subsidiary.

None of these three facts is illegal on its own. Small trading companies do intermediate large flows, new entities do win large contracts, and shared addresses happen in office parks. Together, they change the question from "is this revenue growing?" to "why is the issuer's biggest customer and biggest supplier the same party, and why was that not in the related-party footnote?"

The intuition: a related-party disclosure that is missing is worth more than one that is present, and registries are how you find the missing ones. The series post on [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing) covers what should have been disclosed.

### Channel checks and human sources

Registries are documents. The other half of stage three is people: former employees, current suppliers, distributors, competitors, local officials, franchisees, and the landlord.

This is the part of the process with the most legal and ethical exposure, and the rules are not optional.

- **Material non-public information from an insider is not research, it is a crime.** A former employee describing how the warehouse was run three years ago is a legitimate source. A current employee handing you next quarter's unpublished revenue is insider trading, for both of you.
- **Do not misrepresent who you are.** Pretexting your way into a facility or a phone call is both illegal in many places and fatal to the report's credibility once discovered.
- **Corroborate every human claim with a document.** A former employee's memory is a lead, not evidence. If they say the second plant never opened, the evidence is the satellite image, the utility connection record, or the absence of an environmental permit.
- **Understand the source's incentive.** A fired employee, a losing competitor and a supplier in a payment dispute all have reasons to say damaging things, and all of them are sometimes telling the truth. Weight accordingly and disclose the relationship in the report if it is material.

Expert-network calls, distributor surveys and store-manager conversations are the industrial version of this. The individual version is reading the company's own reviews from employees and customers, its job postings (a company with 4,000 stores that is hiring 11 people is telling you something), and the complaints in its local press.

## Stage 4: reading the company's own filings against each other

The single most durable form of evidence in a short report is a contradiction between two documents the company itself produced. It cannot be dismissed as an outsider's misunderstanding, because the company wrote both halves.

![What a parent tells investors set against what the same group files elsewhere](/imgs/blogs/the-short-sellers-playbook-how-activists-find-fraud-4.webp)

There are seven cross-reads worth running on any company, in rough order of how often they find something.

**1. Consolidated statements against subsidiary statutory filings.** In many jurisdictions the operating subsidiaries must file their own accounts locally, with a different regulator, in a different language, for a different purpose (usually tax). Those local filings are frequently much smaller than the consolidated numbers presented to international investors. There are legitimate reasons for a gap: different accounting standards, different consolidation scope, intercompany eliminations, entities that file on a different year end. There is no legitimate reason for a gap of 40% that management cannot explain line by line.

**2. The same fact in two places in the same annual report.** Segment note against management discussion. Share count on the cover page against the diluted share count in the earnings-per-share note. Employee numbers in the business description against payroll expense. Capital commitments in the footnotes against the capex line in the cash flow statement. The [footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) post is entirely about this reading.

**3. This year's comparatives against last year's originals.** Companies restate prior-period figures for legitimate reasons and are supposed to say so. Quietly changed comparatives, with no restatement note, are one of the highest-yield checks in the whole playbook, and one almost nobody runs. Download the prior filing and diff the numbers.

**4. The filing against the transcript and the deck.** Investor presentations and earnings calls are written by investor relations under lighter liability than the audited statements. When the deck says "80% gross margin business" and the statements say 52%, the definition being used off-statement is usually the tell.

**5. The equity filing against the debt documents.** A bond prospectus or a bank credit agreement is drafted for a different audience by different lawyers, and often contains covenant definitions, subsidiary guarantor lists, and asset schedules that do not appear anywhere in the equity disclosure. Guarantor lists in particular reveal which subsidiaries actually hold the operating assets.

**6. The financial filing against the non-financial regulator.** Companies file with regulators who do not care about their share price: drug approvals, spectrum licences, environmental permits, mine safety reports, insurance statutory returns, bank call reports, food-safety inspections. A pharmaceutical company describing a trial as ongoing while the trial registry lists it as terminated has a problem that no amount of accounting can fix.

**7. The filing against the litigation record.** Court exhibits are sworn under penalty of perjury. Employment claims, supplier disputes and shareholder actions routinely put contracts, invoices and internal emails into the public record. Dockets are searchable and mostly free.

#### Worked example: the profit that pays no tax

An issuer reports pre-tax profit of **\$300 million** and operates in a jurisdiction with a 25% statutory corporate rate. Its income statement shows a tax expense of **\$75 million**, exactly 25%, which looks fine.

Then you turn to the cash flow statement and the tax footnote.

1. Cash taxes actually paid during the year: **\$8 million**. Effective cash tax rate: 8 / 300 = **2.7%**.
2. The difference of \$67 million has to be somewhere. In an honest company it appears as a deferred tax liability on the balance sheet, and the footnote explains why (accelerated depreciation, loss carryforwards, profits in a lower-tax jurisdiction that have not been repatriated).
3. Suppose the footnote's reconciliation attributes the gap to "profits earned in overseas jurisdictions", but the segment note attributes 92% of revenue to the domestic market.

Those two disclosures cannot both be true. Either the profit is domestic, in which case somebody should have paid \$75 million of tax on it, or it is offshore, in which case the segment note is wrong.

The intuition: tax authorities are the one counterparty a company cannot flatter, because overstating profit to them costs real money. Profit that generates no tax anywhere is either a disclosed structure or an invention, and the footnote has to pick one. [Transfer pricing and offshore profit shifting](/blog/trading/forensic-accounting/transfer-pricing-and-offshore-profit-shifting) covers what a legitimate structure looks like.

## Stage 5: tracing the cash

Revenue can be fabricated with journal entries. Cash is harder, because cash makes promises about the outside world too: it earns interest, it sits at a bank that will confirm it, and a company that has it behaves differently from a company that does not.

The central test is the simplest one in this post, and it has caught several of the largest frauds of the last twenty years.

#### Worked example: the interest income that is missing

A company reports cash and cash equivalents averaging **\$2.0 billion** across the year. Short-term deposit rates in its market are around **3.0%**.

1. Interest income you would expect on \$2.0 billion at 3.0%: **\$60 million**.
2. Interest income actually reported in the income statement: **\$6 million**.
3. Implied yield on the reported cash: 6 / 2,000 = **0.3%**.

Now the second leg, which is the one that turns an anomaly into a contradiction.

4. During the same year the company raised **\$300 million** of new equity, diluting existing shareholders.
5. It also borrowed **\$200 million** at a 9% coupon, costing **\$18 million** a year in interest.
6. So it is simultaneously earning 0.3% on \$2.0 billion and paying 9% on \$200 million, a negative carry of 8.7 percentage points on the borrowed money, while sitting on ten times that amount in cash.

![Interest income the reported cash should have earned, against the interest income actually reported, alongside the money the company raised anyway](/imgs/blogs/the-short-sellers-playbook-how-activists-find-fraud-6.webp)

There are innocent explanations, and you must chase every one before you write a word. The cash may be restricted, pledged as collateral, or trapped in a jurisdiction with capital controls, in which case it should be disclosed as such and the "unrestricted cash" headline is misleading rather than fictional. It may have arrived at the very end of the year, in which case the average balance was much lower than the closing balance. It may sit in a currency with near-zero rates. Each of those is checkable and each leaves a disclosure trail.

What is not innocent is a company that has none of those explanations, cannot say which banks hold the money, and keeps raising expensive capital it does not need.

The intuition: cash you cannot see still leaves two footprints, the interest it should have earned and the money the company raised instead of using it.

### The rest of the cash tests

The interest test is the headline. Five more belong in the same stage.

**Cash that never converts into anything.** Real cash eventually pays a dividend, funds a buyback, retires debt, or buys an asset that shows up. Cash that only ever grows on the balance sheet, year after year, while the company borrows for everything it actually does, is a number rather than a balance.

**Cash that leaves through capex and prepayments.** A common structure routes the money out as prepayments for future assets: deposits on land, advances to suppliers, prepayments for equipment, construction in progress that never completes. These sit on the balance sheet as assets, are hard to audit, and are frequently where the cash actually went. Watch "other non-current assets" and "prepayments" grow faster than anything else.

**Operating cash flow that all arrives in the fourth quarter.** Fraudulent cash collection is often arranged around the audit date. Quarterly cash flow that is negative for nine months and enormous in the last six weeks deserves an explanation.

**Where the money is banked.** A large multinational whose cash sits at a small institution in a jurisdiction where confirmation procedures are weak is describing an audit risk. The [Wirecard case in this series](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros) is the reference example of what happens when the bank confirmation is routed through the company rather than obtained directly by the auditor.

**Classification shifting inside the cash flow statement.** Moving an operating outflow into investing, or an investing inflow into operating, changes the headline cash generation without changing the total. [Cash flow statement manipulation and classification shifting](/blog/trading/forensic-accounting/cash-flow-statement-manipulation-classification-shifting) covers the mechanics, and [why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) covers the reading order.

**Financing the receivables.** A company that reports strong collections while also factoring or securitising its receivables, or running a supplier-finance programme, is generating cash from the balance sheet rather than from customers, and the disclosure for both is often thin. See [factoring, supplier financing and hiding debt in plain sight](/blog/trading/forensic-accounting/factoring-supplier-financing-and-hiding-debt-in-plain-sight).

## Stage 6: sizing the position, paying for time, and deciding whether to publish

Suppose all five earlier stages worked. You have a physical count the company cannot match, two of its own filings that contradict each other, a registry search that turns its largest customer into a related party, and cash that earns nothing while the company borrows at 9%.

You are now at the most dangerous point in the process, because you are certain, and certainty is exactly what the structure of a short position punishes.

### The payoff, and what the borrow does to it

![The payoff of a short position with the borrow-cost drag, showing a capped gain, an open-ended loss, and a breakeven that moves against you over time](/imgs/blogs/the-short-sellers-playbook-how-activists-find-fraud-7.webp)

#### Worked example: what nine months of borrow costs you

You short **100,000 shares at \$50.00**, a position of **\$5.0 million**. The stock is hard to borrow at an annualised fee of **25%**.

1. Under Reg T you must have 150% of the position's market value in the account. The \$5.0 million of sale proceeds covers 100%, so you post **\$2.5 million** of your own equity.
2. Borrow fee for nine months, assuming the price stays near \$50.00: 25% × 0.75 years × \$5,000,000 = **\$937,500**.
3. Per share, that is \$937,500 / 100,000 = **\$9.375**.
4. Your breakeven price is therefore \$50.00 − \$9.38 = **\$40.62**.
5. As a percentage: the stock has to fall **18.75%** in nine months before you have made a single dollar.

Now the shape of the whole payoff:

6. Maximum gain, if the stock goes to zero: **\$5.0 million**, the full position, and no more, because a share price cannot go below zero.
7. Loss if the stock triples to \$150.00: 100,000 × (\$150 − \$50) = **−\$10.0 million**, which is twice your maximum possible gain.
8. At \$150.00, FINRA maintenance under Rule 4210(c)(3) is 30% of \$15.0 million, or **\$4.5 million**, on top of the \$15.0 million the position is now worth. Against \$2.5 million of posted equity and a \$10.0 million mark-to-market loss, you receive a margin call you cannot meet, and you cover at the worst possible price.

The intuition: the borrow fee is rent on being early, the margin call arrives before the thesis resolves, and the maximum you can win is smaller than what one bad quarter can cost you.

### Sizing on the outcome you fear

![Four outcomes for the same short position, with the P&L and the thing that actually forces your hand in each](/imgs/blogs/the-short-sellers-playbook-how-activists-find-fraud-8.webp)

Take the same \$5.0 million position inside a **\$200 million** fund, which makes it 2.5% of net asset value, and run it through four outcomes.

#### Worked example: the same thesis, four ways

1. **Right and fast**: \$50.00 to \$5.00 in nine months. Gross gain 100,000 × \$45 = **\$4.5 million**. Borrow accrues on a falling market value, roughly \$0.52 million if the decline is steady, so net is about **\$3.98 million**, a 2.0% gain on the fund.
2. **Right and slow**: the same eventual collapse, three years later. Borrow at 25% on a position that sits near \$50.00 for most of that time costs roughly 25% × 3 × \$5.0 million = **\$3.75 million**. Gross gain \$4.5 million, net about **\$0.75 million**. You were completely correct and you earned 0.4% on the fund over three years.
3. **Wrong**: \$50.00 to \$150.00. Loss **\$10.0 million**, which is **5% of the fund**, from a position you sized at 2.5%. This is the number that matters: a short position's weight in your portfolio grows as it goes against you, automatically, with no decision from you.
4. **Right, but squeezed first**: \$50.00 to \$120.00, then eventually to \$5.00. If you were stopped out or bought in at \$120.00, you lost **\$7.0 million** on a thesis that was correct.

The intuition: you must size the position on the third and fourth outcomes. The first outcome never needed careful sizing, and the second is a reminder that borrow cost, not being right, is what determines whether a slow fraud short makes money.

### Reading the crowding before you enter

#### Worked example: float, days to cover, and the shape of a squeeze

A company has **200 million** shares outstanding. Founders, strategic holders and locked-up employees hold **140 million**, leaving a free float of **60 million** shares. Short interest is **24 million** shares and average daily volume is **4 million** shares.

1. Short interest as a percentage of shares outstanding: 24 / 200 = **12%**, which sounds moderate.
2. Short interest as a percentage of *float*: 24 / 60 = **40%**, which is not moderate at all.
3. Days to cover: 24,000,000 / 4,000,000 = **6 days** of the entire market's trading volume.
4. Your own position of 100,000 shares is only 2.5% of one day's volume, so you can get out. The other 23.9 million shares cannot all get out at once, and their exit is what would move the price against you.

The intuition: percentage of float and days to cover measure how many other people have to buy the same shares you need to buy, on the same day, if the story turns. A 40%-of-float short with 6 days to cover is a position where a 20% rally for any reason, or none, mechanically forces covering that causes a further rally.

This is also the point at which to ask whether shorting the stock is even the right instrument. Buying put options caps the loss at the premium and puts a clock on the thesis. A pair trade against a healthy peer strips out the sector move and leaves the company-specific claim. Sometimes the right expression of "this company's numbers are not real" is simply not owning it, which costs nothing and cannot be squeezed. The analyst's-edge post on [choosing the instrument to express your thesis](/blog/trading/analyst-edge/choosing-the-instrument-to-express-your-thesis) works through that choice, and [from conviction to size](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge) works through the sizing bridge in general.

### Deciding whether to publish, and what the first 48 hours look like

Publishing turns a private position into a public accusation. Three things change the moment you press send.

**The evidence standard rises.** A private thesis can rest on probability. A published one has to survive a defamation claim, which in practice means every factual assertion needs a document behind it, opinion has to be clearly framed as opinion, and the report should say what it does *not* know. Serious reports disclose their position, their methodology and their sample selection for exactly this reason.

**The company gets to respond, and its first response is almost never the informative one.** The predictable sequence is a same-day press release calling the report "false and misleading" without addressing a specific number, an announcement of a buyback or a strategic review, a threatened lawsuit, and then, days or weeks later, the responses that actually matter: an independent committee, a change of auditor, a delayed filing, a resignation, or a trading halt.

**The market reprices before it verifies.** The price moves within minutes, on the report's credibility rather than its contents. This is the mechanism that makes activist short selling profitable and also the mechanism that makes a false report dangerous, and no honest account of this business can claim otherwise.

The question to ask of any short report you read, including your own, is the one that separates the two: **does this report contain a fact the company can settle with a single document?** A bank confirmation, a land title, a customs record, a store list. If yes, the report has put a falsifiable claim on the table and the company's failure to produce the document is itself evidence. If the report is a chain of inferences with no single checkable link, it is an opinion in the costume of an investigation, and it should move a price a lot less than it usually does.

## The anatomy of a short report: Nikola, walked end to end

The best way to see the six stages working together is to take a published report apart. The specimen here is Hindenburg Research's report on Nikola Corporation, titled "Nikola: How to Parlay An Ocean of Lies Into a Partnership With the Largest Auto OEM in America", published on 10 September 2020.

Two reasons to use this one. First, its central claim was physical, which makes it the clearest possible illustration of stage two. Second, the outcome is documented by a regulator in its own words, so we are not relying on the report's own account of whether it was right.

### The setup

Nikola was, in the SEC's later description, "a publicly traded company created through a special purpose acquisition company transaction". A *SPAC* is a listed shell that raises cash and then merges with a private company, which is a route to a public listing that historically involved lighter forward-looking-statement scrutiny than a conventional initial public offering. The company's business was hydrogen and electric trucks.

The screening observation, in the language the SEC used when it settled with the company on 21 December 2021, was that "before Nikola had produced a single commercial product, Milton embarked on a public relations campaign aimed at inflating and maintaining Nikola's stock price". A company being worth a great deal before it has sold anything is not fraud, and by itself it is a stage-one observation and nothing more. Most companies in that position are simply early.

### Stage two: the report went and looked

Nikola had promoted a video of its Nikola One truck apparently in motion. The report's central factual assertion was that the truck was not driving under its own power. According to the report, "Nikola had the truck towed to the top of a hill on a remote stretch of road and simply filmed it rolling down".

What makes this the textbook example of ground-truthing is what the researchers did next. They did not stop at an allegation from a source. They identified the location, described in the report as a stretch of the Mormon Trail near Grantsville, Utah, with "a 2-mile-long perfectly straight stretch with a consistent 3 percent grade". Then they reproduced the experiment: an investigator's vehicle was put in neutral at the top and rolled roughly 2.1 miles, reaching 56 mph.

Look at what that does to the argument. The claim is no longer "a former employee says the truck was not powered". It is now "here is a road, here is its gradient, here is what happens to any vehicle released at the top of it, and here is why the footage is consistent with that and not with a truck under power". Anyone with a car and an afternoon can check it. On the evidence ladder in the second figure, that is the top rung: a physical observation the company cannot talk its way out of, only rebut with its own telemetry.

### Stages three and four: the company's own materials against each other

The report also used the company's own promotional material against the company's own claims. Nikola had presented certain inverters as proprietary technology developed in house. The report pointed to screenshots from the company's own video in which masking tape appears to cover a third-party manufacturer's label on the hardware.

This is the stage-four cross-read in its purest form, and note the structure: nobody has to trust the short seller. The two contradicting artifacts were both produced by the company. That is exactly the property that makes this class of evidence durable, as figure four argues.

A second cross-read concerned hydrogen. The report contrasted a public claim that Nikola produced hydrogen below \$3 per kilogram with an acknowledgment, in July 2020, of producing no hydrogen at all. Again, both halves are the company's own statements at different dates. The technique is the same one described in stage four: put the filing next to the transcript next to the deck, and read them as a set.

### The human sources, and how they were used

The report describes drawing on "recorded phone calls, text messages, private emails and behind-the-scenes photographs", along with former employees, including a conversation relayed from a former employee about the video.

This is the correct relationship between human sources and documents, and it is worth being explicit about it. The former employees pointed at the hill. The hill was then measured. The text messages indicated development had stopped after the show. The masking tape was in the company's own footage. Every human claim that mattered was converted into something physical or documentary. A report that had stopped at the interviews would have been a much weaker document, and much easier to dismiss.

### Stage six: the disclosure

The report disclosed its own interest in one sentence: "After extensive research, we have taken a short position in shares of Nikola Corp."

That sentence is doing necessary work. It tells you the author profits if the price falls, which is a real conflict, and it puts the reader in a position to weigh the evidence knowing it. It does not make the evidence weaker or stronger. It makes the reader responsible for checking it.

### How it resolved

On 21 December 2021 the SEC announced that Nikola Corporation "has agreed to pay \$125 million to settle charges that it defrauded investors by misleading them about its products, technical advancements, and commercial prospects" (SEC press release 2021-267). The order, in the SEC's summary, "finds that Milton misled investors about Nikola's technological advancements, in-house production capabilities, hydrogen production, truck reservations and orders, and financial outlook", and that the company "further misled investors by misrepresenting or omitting material facts about the refueling time of its prototype vehicles, the status of its headquarters' hydrogen station, the anticipated cost and sources of electricity for its planned hydrogen production, and the economic risks and benefits associated with its contemplated partnership with a leading auto manufacturer". The same release notes that the settlement followed a litigated action the SEC had filed earlier in 2021 against Trevor Milton, the founder and former chief executive.

Read the SEC's list against the report's list. Technological advancements, in-house production capabilities, hydrogen production. Those are the three things the report went and physically checked, in that order.

The lesson of the specimen is not that short sellers are right. It is *what kind of work* produced a claim that survived a regulator's own investigation: a claim about the physical world, tested in the physical world, cross-checked against the company's own published material, with the author's conflict declared on the first page.

## When short sellers are wrong: the failure modes and the incentives

Everything above is the case for taking these reports seriously. Here is the other half, and a description of this business that leaves it out is propaganda.

![A two-by-two of whether the thesis is true against whether the market believes it, and who pays in each of the four cells](/imgs/blogs/the-short-sellers-playbook-how-activists-find-fraud-9.webp)

Start with the incentive, stated plainly: **an activist short seller is paid when the price falls, not when the truth comes out.** Those two things coincide often enough to make the activity socially useful, and they come apart in four specific ways.

**Overstating a real problem.** The most common failure is not fabrication. It is a genuine finding, inflated. A company with an aggressive revenue-recognition policy becomes "a fraud". A related-party transaction that should have been disclosed becomes "the entire business is fake". The finding is real, the headline is not, and the price reacts to the headline.

**Publishing before the work is finished.** Borrow costs money and a competing firm may be circling the same name. Both pressures argue for publishing at 70% confidence, and the report will not tell you which parts are the 70%.

**Confusing an unanswered question with an answer.** "The company has not explained X" is a legitimate observation and is not evidence that X is sinister. A report built entirely of unanswered questions has a rhetorical shape that feels like proof and is not.

**The exit is not the thesis.** The report's author can cover into the volatility the report itself created. That is legal and disclosed, and it means the realized profit on the trade can be almost unrelated to whether the analysis was correct. A reader evaluating a track record on "did the stock fall after publication?" is measuring the wrong thing.

Then there is the category where the short seller is simply wrong, and the harm lands on real shareholders and employees.

### A documented case: Farmland Partners

Farmland Partners Inc. (NYSE: FPI) is a listed farmland real estate investment trust that became the subject of an anonymous critical article published under the pseudonym "Rota Fortunae". The company sued the author.

In its own quarterly results announcement filed with the SEC as an exhibit on 4 August 2021, FPI listed among its highlights that it had "reached settlement with Quinton Mathews regarding the falsity of claims that were used to launch the 'short and distort' scheme targeting FPI, its management, and its stockholders". That characterisation is the company's own, in the company's own filing, and it is quoted here as such.

The numbers in the same filing are the part worth sitting with. FPI reported that for the six months ended 30 June 2021, "legal and accounting expense included \$5.2 million related to litigation and revenue included \$0.6 million of litigation settlement proceeds related to Rota Fortunae, resulting in a net impact of \$4.6 million". The comparable figure for the first six months of 2020 was \$0.8 million of litigation-related legal and accounting expense.

#### Worked example: the arithmetic of being wrongly accused

Take those disclosed figures at face value and run them.

1. Litigation-related legal and accounting expense, six months to 30 June 2021: **\$5.2 million**.
2. Litigation settlement proceeds recognised in the same period: **\$0.6 million**.
3. Net cost to the company in that half-year alone: **\$4.6 million**.
4. Ratio of money spent to money recovered in the period: 5.2 / 0.6, or roughly **8.7 to 1**.
5. For scale, the same company's litigation-related expense in the first half of 2020 was \$0.8 million, so the half-year litigation cost rose by about **\$4.4 million**, or **6.5 times**.

The intuition: the remedy is wildly asymmetric. Even where a company pursues an author and reaches a settlement it describes in these terms, the disclosed recovery in that period was a fraction of the disclosed legal cost, and none of it compensates the shareholders who sold at the bottom or the management time consumed. That asymmetry is the single strongest argument the critics of activist short selling have, and it is an argument about remedies rather than about whether the research is valuable.

Two structural problems make it worse. **Anonymity** means a reader cannot assess the author's conflicts, track record, or position size, and the company's only route to accountability is expensive litigation. And **speed**: the price moves in minutes, the rebuttal takes weeks, and the correction, if it comes, arrives after the selling is done.

Disclosure rules for net short positions differ by jurisdiction and have changed repeatedly over the last decade, including proposed and contested US rules on short-position reporting. Rather than quote a threshold that may already be out of date, the honest instruction is to check the current rule in the relevant jurisdiction before relying on any published short-interest figure as complete.

### How to read a short report

The practical payoff of all of this is a checklist. Run it on any short report you are handed, and on any bull case too.

1. **Does it disclose the author's position?** A report that does not is not research.
2. **Does it state how it selected its sample?** If it counted stores, which stores, chosen how, and before or after seeing the results.
3. **Is every factual claim attached to a document, and can you reproduce it?** The gold standard is a claim you could check yourself: a registry entry, a court exhibit, a road with a measurable gradient.
4. **Does it separate fact from inference from opinion?** Good reports label their own speculation. Bad ones let the reader do the merging.
5. **Does it say what it does not know?** An admission of a gap is a marker of honesty and of a real research process.
6. **Does it make a claim the company can settle with one document?** If yes, watch whether the document appears. That silence is the most informative thing that will happen in the next month.
7. **Is the author accountable?** Named, with a track record you can check, including the misses.

If a report fails items 1, 3 and 6, treat it as an opinion with a position behind it, whatever it looks like. If it passes all seven, the company's response becomes the interesting document, and the absence of a specific answer to a specific number is itself a finding.

## Common misconceptions

**"Short sellers make money by spreading rumours."** Some have tried, and it is a prosecutable offence rather than a business model. The structure of the trade is what makes it a bad one: a rumour that does not resolve into something checkable leaves you paying borrow on a position that the market forgets about within a week, and the shares you sold have to be bought back at whatever price the rumour left behind. The durable version of this business is the opposite of a rumour. It is a document. That does not mean manipulative short selling never happens, and the section above on the failure modes takes it seriously. It means the default assumption that a critical report is a rumour is a way of avoiding reading it.

**"The author is short, so the research is worthless."** Disclosed interest is a reason to check the evidence carefully, not a reason to discard it. Every buy recommendation you have ever read was also written by someone with an interest: a bank that would like the company's next financing mandate, a fund that already owns the stock, a management team paid in equity. The difference is that the short seller usually tells you at the top of the first page. The correct response to a disclosed conflict is to evaluate the documents, and the documents in a good short report are mostly things the company or a government produced.

**"The stock went up after the report, so the report was wrong."** Price is not a verdict, and the gap between "wrong" and "early" is invisible in real time. A company under attack can raise capital, announce a buyback, report another quarter of fabricated growth, and trade higher for a long time. Some of the most-cited short reports in the last two decades were followed by months or years of rising prices before the thesis resolved. What settles the question is whether the specific factual claim was answered: did the bank confirmation appear, did the land title exist, did the store list match. If the report made a checkable claim and the company never checked it, a rising price is not an answer.

**"Short selling drives the price down."** A short sale adds one seller today and one guaranteed buyer later, which is why short interest is sometimes described as latent demand. The natural experiment for this is the wave of emergency short-selling bans imposed across dozens of countries during the 2007–09 crisis, imposed and lifted on different dates and applied to different stock lists. Studying that variation, Alessandro Beber and Marco Pagano found that the bans "(i) were detrimental for liquidity, especially for stocks with small capitalization and no listed options; (ii) slowed down price discovery, especially in bear markets, and (iii) failed to support prices, except possibly for U.S. financial stocks" (Journal of Finance, vol. 68 no. 1, 2013, pp. 343–381). Stopping the shorts made markets worse at trading and no better at holding prices up. The larger point is that shorting cannot make a solvent company insolvent by itself: it can only change the price at which the company raises its next round of capital, which for a healthy company is an inconvenience and for a company that needs constant financing is closer to fatal. That distinction is worth holding on to, because it explains why the companies most damaged by short reports are disproportionately the ones with the weakest underlying cash generation.

**"Fraud is found by clever accounting analysis."** Occasionally. Far more often it is found by clerical persistence: downloading a five-year-old filing and diffing the comparatives, reading a registry entry in a language you had to translate, calling a phone number listed on a supplier's tax filing, walking into a branch. The analytical part of this work is the easy part and it happens in the first hour. The expensive part is the errand.

**"A company suing its short seller must have something to hide."** No. Companies sue for both reasons: because the report was defamatory and because the report was correct and litigation is a way to buy time and impose cost. The lawsuit itself carries almost no information. What carries information is discovery, because it forces both sides to produce documents, and that is why so many of these cases settle before it.

**"The auditors would have caught it."** An audit is designed to give reasonable assurance that the statements are free of material misstatement, using evidence that is largely provided by the entity being audited, under a scope and fee negotiated with that entity's management. It is not an investigation, it is not adversarial, and it is not resourced to send anyone to count stores. [How an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) sets out the gap precisely. Every major fraud in this series had an unqualified audit opinion on the statements that were later restated.

## How it shows up in real markets

Three cases where the mechanisms in this post did the work, each documented by a regulator or a court rather than by a short seller.

### Luckin Coffee: the related-party round trip, at scale

Luckin Coffee Inc. was a Chinese coffee chain whose American Depositary Shares traded on Nasdaq until 13 July 2020. On 16 December 2020 the SEC charged it "with defrauding investors by materially misstating the company's revenue, expenses, and net operating loss", and Luckin agreed to pay a **\$180 million** penalty (SEC press release 2020-319).

The mechanism is exactly the one in stage three and figure five. According to the SEC's complaint, from at least April 2019 through January 2020 Luckin "intentionally fabricated more than \$300 million in retail sales by using related parties to create false sales transactions through three separate purchasing schemes". The concealment ran through the accounts the same way: the complaint alleges employees inflated expenses "by more than \$190 million, creating a fake operations database, and altering accounting and bank records to reflect the false sales".

Two details are worth carrying forward into your own reading. The first is the size of the overstatement the SEC alleged: "approximately 28% for the period ending June 30, 2019, and by 45% for the period ending Sept. 30, 2019". That range is worth remembering next time you compute an implied per-store number, because it tells you what magnitude of gap real cases actually involve. The 35% used in the worked example earlier in this post sits deliberately inside it.

The second is what the company did with the money it appeared to be earning: the SEC's complaint alleges that during the fraud period Luckin "raised more than \$864 million from debt and equity investors". A business generating the cash it claimed would not have needed to. That is the stage-five test, and it is visible from outside.

### Wirecard: the cash trace

Wirecard is the reference case for stage five, and this series covers it in full in [Wirecard: the missing EUR 1.9 billion](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros). The short version for our purposes: the disputed asset was cash, the thing that ought to be the easiest item on a balance sheet to confirm, and the failure was in the confirmation procedure rather than in a subtle accounting estimate.

For a reader running the playbook, Wirecard is the case that justifies asking apparently rude questions about a cash balance: which banks, in which jurisdictions, confirmed by whom, and directly to whom. A company that cannot answer those four questions about its largest asset has told you something, and the interest-income test in stage five is how you get to the question without needing access to management.

### Enron: the filings against each other

Enron is the case that makes stage four's argument, because almost everything necessary was in the public filings and was simply hard to read. The structures were disclosed in the footnotes in language that defeated most readers, and the gap between reported earnings and the cash the business generated was visible to anyone who looked at both. This series re-reads it forensically in [Enron: a forensic re-read of SPEs and mark-to-market](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market), and there is a narrative account in [the 2001 Enron accounting fraud](/blog/trading/finance/enron-2001-accounting-fraud).

The relevant lesson for this playbook is uncomfortable: the information was not hidden, it was expensive to read. The stage-four cross-reads exist because the highest-yield work is usually not obtaining a secret, it is reading two public documents carefully enough to notice they disagree.

### What the three have in common

In every one of them the decisive evidence was something outside management's control: a bank's own confirmation, a related party's own registry and transaction records, the arithmetic linking reported profit to reported cash. None of the three required access, a source, or a leak. All three required somebody to do arithmetic on public documents and then refuse to accept the first explanation.

## Running a scaled-down version on a company you own

Almost nobody reading this is going to short a stock, and the point of the playbook is not that you should. The point is that the same six stages answer a much more common question: **should I keep owning this?**

That version of the question is strictly easier, in four ways. You do not pay borrow, so time is free. You cannot be squeezed, because you already own the shares. You do not have to be certain, because the action available to you is to reduce rather than to accuse. And you do not have to publish, which removes the entire legal and ethical apparatus around the last stage. The whole hard part of activist short selling is the trade, not the research. You get the research without the trade.

Here is the funnel, scaled to about nine hours spread over four weekends and one afternoon.

### Weekend one: the numbers, two hours

Download the last three annual reports and put four series into a spreadsheet: revenue, operating cash flow, receivables, and inventory.

- Compute the **accruals ratio**: (net income minus operating cash flow) divided by average total assets. Rising, and positive, for three straight years is the single most informative number on this list.
- Compute **days sales outstanding** (receivables ÷ revenue × 365) and **days inventory outstanding** (inventory ÷ cost of sales × 365) for each year. Both drifting up while revenue grows is the classic shape.
- Compare all four series to the two closest listed competitors. You are looking for the company that is the outlier in its own industry, not the outlier in the market.

The stopping rule for this stage: if profit and cash have tracked each other for three years and the working-capital ratios look like the peer group, you are done. Go and do something else. That is most companies, and it should be.

### Weekend two: the company's own words, three hours

Read four things in the most recent annual report, in this order, and read them properly rather than skimming.

1. **The audit report.** Who signed it, is it the same firm as three years ago, is the opinion unqualified, and what are the key audit matters. A change to a materially smaller firm is worth an hour on its own.
2. **The related-party footnote.** Every name in it, and what the transaction was. Then ask the harder question: which of the company's large customers or suppliers are *not* in this note, and how would you know if they should be.
3. **The tax footnote.** Compare tax expense to cash taxes paid, and the effective rate to the statutory rate. If they diverge, the reconciliation has to explain it in a way you can follow.
4. **The segment note against the management discussion.** Same numbers, two presentations, written by different people. Any gap is worth chasing.

Then run the cheapest high-yield check in this whole post: open last year's annual report alongside this year's and compare the prior-year comparatives. They should be identical. If they are not, and there is no restatement note, you have found something in ten minutes.

### Weekend three: the outside record, two hours

Now leave the company's own documents.

- Search the corporate registry in the company's home jurisdiction for the parent, the main operating subsidiaries, and any named customer or supplier. Registered capital, incorporation date, directors, addresses.
- Pull the company's own published list of stores, branches, plants or offices, and check a random ten of them against a map service and a delivery or review platform. Do they exist, are they open, when did the reviews start.
- Search court records for the company's name. Employment cases and supplier disputes put contracts and internal documents into the public record.
- Read job postings. A company adding 500 stores a year and hiring nine people is describing two different businesses.

### Weekend four: the cash, two hours

Run the interest test from stage five on the real numbers.

- Average cash and equivalents across the year, times a plausible short-term deposit rate for the currency it is held in. Compare to reported interest income.
- List everything the company did that a company with that much cash would not need to do: equity raises, expensive borrowing, factored receivables, deferred supplier payments, delayed capex.
- Read the cash flow statement by quarter if quarterly data exists. Where in the year does the operating cash arrive.

### The afternoon: count something

Go to one store, one branch, one site. Spend an hour. Count transactions, or cars, or people in the office, or trucks at the gate. One observation is not a sample and cannot prove anything about the company. What it can do is convert a number on a page into something you have seen with your own eyes, and that changes how you read the next annual report more than any ratio will.

Then run the two divisions from the ground-truthing stage on what you counted, which takes five minutes and is the whole method in miniature. Say you count 20 transactions in an hour at a branch that is open 12 hours. That extrapolates to roughly 240 a day, before you adjust for the fact that you almost certainly picked a quiet hour. Now go back to the filing and compute what the accounts require, exactly as the earlier figure lays it out: revenue divided by the store count, divided by 365, divided by the average ticket.

Three outcomes, and only one of them is interesting. If the filing implies 150 items a day and your hour implies 240, the company is being conservative and you can put the question down. If the two land near each other, the reported number survived contact with the physical world, which is the most reassuring thing you will learn all month. If the filing implies 247 and your hour implies 160, you have not found fraud, and you should not say that you have. What you have found is the one question worth putting to the company in writing, with a number attached to it.

One hour, one branch, and two divisions. That is the entire ground-truthing stage of a professional short thesis, run at the scale of a Saturday afternoon, and it is available to anyone willing to stand somewhere and count.

### What to do with the answer

This is arithmetic, not advice, and the assumptions below are yours to set.

#### Worked example: what a doubt is worth to a holder

You own 500 shares at \$50.00, a **\$25,000** position inside a **\$250,000** portfolio, so 10% of your money. Your unrealised gain is **\$10,000** and your capital gains tax rate is 20%.

Before the research, you would have said the chance of a serious restatement at this company was about **1%**. After four weekends, you would put it at **12%**. Assume, as an explicit and arguable assumption, that a confirmed accounting fraud costs a holder **80%** of the position.

1. Expected loss from restatement risk before: 0.01 × 0.80 × \$25,000 = **\$200**.
2. Expected loss after: 0.12 × 0.80 × \$25,000 = **\$2,400**.
3. The research changed your expected loss by **\$2,200**, or 8.8% of the position.
4. The cost of acting is the tax on the gain: 20% × \$10,000 = **\$2,000**, plus whatever upside you give up if you are wrong.

So the full-sale decision is genuinely close, which is the honest answer and not a satisfying one. But look at the third option.

5. Selling **half** realises \$1,000 of tax, removes \$1,200 of the expected loss, and leaves you holding the position that lets you keep watching. On these assumptions it is the best of the three, and it is available precisely because you are not short and do not have to be right on a schedule.

The intuition: a holder's version of a short thesis does not need to reach a verdict. It only needs to move a probability enough to change a position size, and partial answers are enough for that.

Write down, in advance, the two things that would end the question in each direction: the disclosure that would settle your worry, and the event that would make you sell the rest. Doing that before you are emotionally committed is the entire content of [stress-testing your thesis with a pre-mortem](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem), and the failure to do it is what [confirmation bias and the thesis you fall in love with](/blog/trading/analyst-edge/confirmation-bias-and-the-thesis-you-fall-in-love-with) is about. Both apply with equal force to the person who has decided a company is a fraud and to the person who has decided it cannot be.

## Sources & further reading

The rules and figures behind this post, with the primary document for each.

**The mechanics of the trade**

- The locate requirement: Regulation SHO, Rule 203(b)(1), [17 CFR 242.203](https://www.law.cornell.edu/cfr/text/17/242.203). A broker may not accept or effect a short sale unless it has "borrowed the security, or entered into a bona-fide arrangement to borrow the security", or has "reasonable grounds to believe that the security can be borrowed so that it can be delivered on the date delivery is due", and has documented that compliance. The bona fide market-making exception is 203(b)(2)(iii).
- Initial margin on a short sale: Regulation T, [12 CFR 220.12(c)(1)](https://www.law.cornell.edu/cfr/text/12/220.12), which requires "150 percent of the current market value of the security".
- Maintenance margin: [FINRA Rule 4210(c)](https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210). Subparagraph (c)(3) requires "\$5.00 per share or 30 percent of the current market value, whichever amount is greater" for stock short at \$5.00 or above; subparagraph (c)(2) requires "\$2.50 per share or 100 percent of the current market value, whichever amount is greater" for stock short below \$5.00. Subparagraph (c)(1) sets the 25 percent maintenance requirement on long positions.

**The specimen**

- Hindenburg Research, "Nikola: How to Parlay An Ocean of Lies Into a Partnership With the Largest Auto OEM in America", published 10 September 2020. All quotations from the report in this post, including the description of the Mormon Trail location, the "consistent 3 percent grade", the neutral-gear test reaching 56 mph over roughly 2.1 miles, the masking-tape inverter screenshots, and the short-position disclosure, are from that report.
- U.S. Securities and Exchange Commission, press release 2021-267, 21 December 2021, "SEC Charges Nikola Corporation with Fraud": [sec.gov/news/press-release/2021-267](https://www.sec.gov/news/press-release/2021-267). Source for the \$125 million settlement, the SPAC description, the "before Nikola had produced a single commercial product" language, and the list of findings in the order.

**The other cases**

- U.S. Securities and Exchange Commission, press release 2020-319, 16 December 2020, on Luckin Coffee Inc.: [sec.gov/news/press-release/2020-319](https://www.sec.gov/news/press-release/2020-319). Source for the \$180 million penalty, the "more than \$300 million" of fabricated retail sales "by using related parties", the "more than \$190 million" of inflated expenses, the 28% and 45% revenue overstatements, the "more than \$864 million" raised during the fraud period, and the 13 July 2020 Nasdaq delisting date.
- Farmland Partners Inc., second-quarter 2021 results, filed with the SEC as Exhibit 99.1 on 4 August 2021: [sec.gov Archives, accession 0001104659-21-100280](https://www.sec.gov/Archives/edgar/data/1591670/000110465921100280/tm2124185d1_ex99-1.htm). Source for the company's own description of the settlement with Quinton Mathews and for the \$5.2 million of litigation-related legal and accounting expense, the \$0.6 million of settlement proceeds, the \$4.6 million net impact, and the \$0.8 million comparable figure for the first half of 2020. The characterisation of the article as a "short and distort" scheme and of the claims as false is the company's, quoted as such.

**On the effects of restricting short selling**

- Alessandro Beber and Marco Pagano, "Short-Selling Bans around the World: Evidence from the 2007-09 Crisis", *The Journal of Finance*, vol. 68 no. 1 (2013), pp. 343–381. The working-paper version is [CSEF Working Paper 241](https://www.csef.it/WP/wp241.pdf). Finding quoted in the misconceptions section: bans "(i) were detrimental for liquidity, especially for stocks with small capitalization and no listed options; (ii) slowed down price discovery, especially in bear markets, and (iii) failed to support prices, except possibly for U.S. financial stocks."

**Further reading on who actually detects fraud**

- Alexander Dyck, Adair Morse and Luigi Zingales, "Who Blows the Whistle on Corporate Fraud?", *The Journal of Finance*, vol. 65 no. 6 (2010), pp. 2213–2253. The standard empirical treatment of which actors, including analysts and short sellers, bring corporate fraud to light.

**Earlier posts in this series that this playbook depends on**

- [The three financial statements and how they interlock](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock)
- [Reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income)
- [The accruals ratio and the accruals anomaly](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly)
- [Related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing)
- [Round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue)
- [The footnotes and MD&A: where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried)
- [How an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch)

**A note on what is not in this post.** Two things were left out for lack of a primary source I could reach and verify: the widely discussed Muddy Waters research on Sino-Forest Corporation, which would have been a second specimen, and the criminal proceedings against Nikola's founder, which are referenced here only to the extent the SEC's own press release describes its parallel civil action. A separate pending US matter involving a well-known short seller's own disclosures was also left out: it is unadjudicated, and repeating charges without a verified current status would be worse than saying nothing.
