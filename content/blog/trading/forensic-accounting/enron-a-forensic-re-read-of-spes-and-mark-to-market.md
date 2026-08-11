---
title: "Enron: A Forensic Re-Read of SPEs and Mark-to-Market"
date: "2026-08-11"
publishDate: "2026-08-11"
description: "Reading Enron's own 10-K filings with this series' detection toolkit — and finding that most of the ratio screens passed it while the footnotes did not."
tags: ["forensic-accounting", "enron", "special-purpose-entities", "mark-to-market", "financial-statement-fraud", "related-party-transactions", "off-balance-sheet", "accruals", "financial-statements"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 49
---

> [!important]
> **TL;DR** — Enron's fraud was not hidden in a vault. Most of it was printed in the filings, and the reason nobody acted is that the ratio screens this series has spent thirty posts teaching you mostly gave Enron a clean reading.
>
> - Two engines produced the reported profit: **mark-to-market** recognition of the full modelled value of twenty-year contracts, and **special-purpose entities** capitalised with Enron's own stock that absorbed losses the income statement never saw.
> - Run the screens on the FY2000 10-K and the cash-flow accruals ratio comes out at **−7.7%** — which looks *conservative* — and Altman's Z at **2.45**, the grey zone. Neither fires.
> - What does fire: return on assets falling from **3.62%** in 1996 to **1.49%** in 2000, total assets growing **+96%** in a year while net income grew **+9.6%**, and Note 16.
> - **Note 16 of the FY2000 10-K is the confession.** Enron disclosed that it contributed assets "valued at approximately \$1.2 billion" to related-party entities, took back "\$1.2 billion in notes receivable", parked the entities' cash of "\$172.6 million" in *Enron demand notes*, and recognised "approximately \$500 million" of revenue on derivatives with them.
> - The one number to remember: the November 2001 restatement cut 1997 net income from **\$105 million to \$9 million** — and 1997 diluted EPS from **\$0.16 to negative one cent**.

Every account of Enron eventually reaches for the word *hidden*. The debt was hidden, the losses were hidden, the conflicts were hidden. It is a comforting story, because it implies that an ordinary careful reader had no chance, and that the only failure was one of access.

The filings do not support that story. Enron's last annual report before the collapse is a public document. It says, in a numbered footnote, that the company did roughly a billion dollars of business with partnerships run by one of its own officers, and that it booked half a billion dollars of revenue on derivative contracts written by those partnerships. It says that the partnerships' cash was invested in Enron's own promissory notes. Nobody had to leak this. It was typed, printed, filed with the Securities and Exchange Commission on 2 April 2001, and mailed to shareholders.

So this post asks a narrower and more useful question than "what happened at Enron". It asks: **if you had run this series' toolkit over Enron's filings in 1999 and 2000, what would each tool have told you, and in what order?** The answer is genuinely uncomfortable. The ratio screens — the ones that are easy to automate, the ones that scale across a thousand companies — mostly passed Enron. The things that caught it were slower, and required reading prose.

![Enron's reported income came from marking its own models and from derivatives written by entities it had capitalised with its own stock](/imgs/blogs/enron-a-forensic-re-read-of-spes-and-mark-to-market-1.webp)

The diagram above is the mental model for the whole post. There are two engines on the left, and they meet in the same reported operating income on the right. The top engine takes long-dated contracts with no observable market price and recognises their entire modelled value as revenue immediately. The bottom engine takes investments that are losing money and sells or "hedges" them into entities that Enron itself has funded with its own shares. Neither engine produces cash. Both produce reported profit. And the balance sheet at the bottom is where the difference accumulates — which is exactly why the balance sheet, not the income statement, is where this fraud was visible first.

If you want the narrative history — the culture, the traders, the California power market, the shredding, the human cost — that is a different post, and it already exists: [Enron's 2001 accounting fraud](/blog/trading/finance/enron-2001-accounting-fraud) covers the story end to end. This one stays inside the filings.

## Foundations: the building blocks of an Enron-shaped fraud

Enron's accounting is often described as impossibly complex. The individual mechanisms are not complex. There are five of them, and each one is a single idea. What was complex was the *number* of them stacked together, and the deliberate use of that complexity as camouflage.

Read this section from zero even if you know some accounting, because the precise definitions matter later.

### Revenue, profit, and cash are three different things

A company's **revenue** is what it has earned by delivering something. **Profit** (or **net income**) is revenue minus every cost of earning it. **Cash flow from operations** is the money that actually moved into the bank account from running the business.

Under **accrual accounting**, which is what every listed company uses, revenue is recorded when it is *earned*, not when it is *paid*. If you deliver gas in December and get paid in February, December's income statement shows the revenue and December's cash flow statement does not. The gap sits on the balance sheet as a **receivable** — a promise of money.

That gap is normal and unavoidable. It is also the single most exploited feature of financial reporting, because the company decides when something has been "earned". This series covers the general case in [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits); Enron is the extreme case.

### Mark-to-market: profit as an opinion about the future

Suppose you sign a contract today to supply natural gas every month for the next twenty years at a fixed price. You have delivered nothing. No money has changed hands.

**Mark-to-market accounting** (often "MTM", also called **fair-value accounting**) says: value that contract today at what it is worth, and put the change in value through the income statement now. If your model says the contract will earn you \$10 million a year for twenty years, and you discount those future amounts back to today's money, you get a single number — the **net present value**, or **NPV** — and you can book it as income this quarter.

The **discount rate** is the annual percentage you use to shrink future money into today's money, because a dollar in 2020 is worth less than a dollar today. A higher discount rate makes future money worth less, and therefore makes the contract worth less.

Mark-to-market is not fraud. For a share of Apple, it is obviously right: the price is on a screen, everyone can see it, and pretending you paid \$50 for something now worth \$200 would be the distortion. The question is always the same: **is there an observable price, or is there a model?** When there is a model, the profit is an opinion, and the opinion belongs to the person whose bonus depends on it.

Enron adopted the industry standard for this — the Emerging Issues Task Force's Issue No. 98-10, *Accounting for Contracts Involved in Energy Trading and Risk Management Activities* — as of 1 January 1999, taking a \$131 million after-tax charge on adoption alongside a second accounting change, according to Note 18 of its FY2000 Form 10-K.

### Consolidation: whose balance sheet is it on?

If a parent company controls another entity, it must **consolidate** it — add all of that entity's assets, liabilities, revenues and expenses into its own statements, line by line, as though they were one company. Consolidation is what makes a group's debt visible.

If the parent does *not* control the entity, it does not consolidate. It shows only its own investment in it, as a single line. The entity's debt stays off the parent's balance sheet entirely.

That distinction is worth an enormous amount of money to a company with a lot of debt, and it is the whole game here.

### Special-purpose entities and the three percent rule

A **special-purpose entity** (SPE) is a company or partnership created to do one narrow thing — own a power plant, hold a portfolio of loans, finance a single building. SPEs are ordinary and mostly legitimate; securitisation and project finance depend on them.

Under the accounting rules in force before 2003, an SPE could be kept off the sponsor's balance sheet if an **independent third party** made a real equity investment of at least **three percent** of the SPE's total assets, that equity was **genuinely at risk**, and the third party had control. Three percent is a strikingly thin sliver — it means 97% of the vehicle can be funded by the sponsor and its bankers, and the whole thing still sits outside the sponsor's statements.

The rule's entire protective force lives in the phrase *genuinely at risk*. If the outside investor is guaranteed against loss — by a side agreement, a put option, a guarantee, or collateral supplied by the sponsor — then there is no outside equity, and the vehicle must be consolidated.

The rules changed afterwards. FASB Interpretation No. 46, issued January 2003 and revised as FIN 46(R) in December 2003, replaced the mechanical percentage test with a **variable-interest** model that asks who absorbs the expected losses and receives the expected returns. This series covers the mechanics and the modern rules in [off-balance-sheet financing and special-purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities); here we only need the pre-2003 version, because that is the regime Enron operated in.

### Related parties

A **related party** is a person or entity on both sides of a transaction — an officer, a director, a family member, or an entity they control. Related-party transactions are not automatically improper, but they remove the one protection an arm's-length deal provides: a counterparty whose interest is to negotiate against you. Companies must disclose them in the footnotes. Reading those footnotes is the highest-yield habit in forensic accounting, and this series makes the case in [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing).

#### Worked example: the simplest possible mark-to-market profit

Start with the smallest version of the trick, with round illustrative numbers.

You run an energy company. On 1 December you sign a contract to deliver gas for one year. Your model says you will earn a **\$12 million** margin over the year, spread evenly at \$1 million a month.

- **Accrual accounting** says: you have delivered nothing in December, so December's income statement shows roughly one month of margin, \$1 million, and eleven months of margin arrive as you deliver.
- **Mark-to-market accounting** says: the contract has a value today. Discounting \$1 million a month for twelve months at 8% a year gives an NPV of about **\$11.5 million**. Book \$11.5 million of income in December.

Your December profit is 11.5 times larger under mark-to-market, from the identical contract, with the identical cash flows, and zero dollars received.

Now stretch the contract from one year to twenty. The accrual answer barely changes — you still recognise about a month of margin in December. The mark-to-market answer grows enormously, because you are now pulling twenty years of modelled margin into a single quarter.

**The intuition this teaches: mark-to-market does not change how much money a contract makes. It changes which quarter's income statement gets to claim it — and the longer the contract, the more quarters get robbed to pay the current one.**

## 1. What Enron said it was, and what the balance sheet said

By 2000, Enron described itself as an asset-light network business. It had reinvented itself away from owning pipelines toward *intermediating* — making markets in gas, power, bandwidth, weather, credit. The pitch was that Enron was a logistics and market-making platform whose earnings would scale without a matching build-out of physical plant.

An asset-light business has a recognisable financial signature: modest and slow-growing assets, high returns on the capital employed, and cash generation that tracks reported earnings. Enron's filings show almost the opposite signature.

![Enron's balance sheet at 31 December 2000: \$21.0bn of the \$65.5bn of assets were price-risk-management positions, sitting on \$11.5bn of equity](/imgs/blogs/enron-a-forensic-re-read-of-spes-and-mark-to-market-2.webp)

Here is the series as each year's own annual report filed it, in millions of dollars. Every figure below comes from Enron Corp's own Form 10-K for the year in question.

| Fiscal year | 1996 | 1997 | 1998 | 1999 | 2000 |
| --- | --- | --- | --- | --- | --- |
| Total revenues | 13,289 | 20,273 | 31,260 | 40,112 | **100,789** |
| Operating income | n/a | **15** | 1,378 | 802 | 1,953 |
| Net income | 584 | 105 | 703 | 893 | 979 |
| Diluted EPS (\$) | n/a | 0.16 | 1.01 | 1.10 | 1.12 |
| Cash from operations | 1,040 | 501 | 1,640 | 1,228 | 4,779 |
| Total assets | 16,137 | 23,422 | 29,350 | 33,381 | **65,503** |

Two things in that table should stop you.

**Revenue reached \$100.8 billion in 2000 and produced \$979 million of net income.** That is a net margin of 0.97%. A company selling a hundred billion dollars of anything and keeping less than one cent on the dollar is not a platform business with network effects. It is a trading book, and gross revenue is close to a meaningless number for a trading book — a fact this series returns to in [the metrics companies invent](/blog/trading/forensic-accounting/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent).

**Total assets grew from \$33.4 billion to \$65.5 billion in a single year — up 96% — while net income grew 9.6%.** An asset-light company had just doubled its balance sheet.

What was in the new assets? The FY2000 balance sheet answers directly. "Assets from price risk management activities" appear twice, in current assets at **\$12,018 million** and in long-term assets at **\$8,988 million**, for a total of **\$21,006 million**. A year earlier the same two lines totalled \$5,134 million. So this one category grew **4.1 times** in a year and reached **32% of total assets**.

"Assets from price risk management activities" is the balance-sheet residue of mark-to-market accounting. It is what the company believes its open derivative and contract positions are worth. It is not cash, not receivables, not plant. Roughly a third of Enron's balance sheet was its own valuation opinion, supported by \$11,470 million of shareholders' equity.

And Enron told you how much of the profit that opinion produced. In Note 3 of the FY2000 10-K: *"The income before interest, taxes and certain unallocated expenses arising from price risk management activities for 2000 was \$1,899 million."* Consolidated operating income that year was \$1,953 million. The two figures are not on the same accounting basis — the first excludes unallocated expenses, so it is a segment-style measure rather than a subtotal of the second — but the magnitudes are impossible to ignore. Essentially all of Enron's operating income came from the activity whose valuation nobody outside the company could check.

#### Worked example: what happened to the return on capital

Return on assets is net income divided by total assets. It answers a plain question: for every dollar of stuff the company controls, how many cents of profit does it produce? Compute it from the filed lines:

| Year | Net income (\$m) | Total assets (\$m) | Return on assets |
| --- | --- | --- | --- |
| 1996 | 584 | 16,137 | **3.62%** |
| 1997 | 105 | 23,422 | 0.45% |
| 1998 | 703 | 29,350 | 2.40% |
| 1999 | 893 | 33,381 | 2.68% |
| 2000 | 979 | 65,503 | **1.49%** |

Now put that next to the earnings-per-share line from the table above: diluted EPS rose from \$1.01 in 1998 to \$1.10 in 1999 to \$1.12 in 2000.

**EPS went up every year while the return on capital went down.** Over the four years from 1996 to 2000, assets multiplied by **4.06 times** and net income by **1.68 times**.

That divergence has exactly one arithmetic explanation: the company was adding capital faster than it was adding profit, and reporting the growth per share anyway. A business that genuinely earns high returns does not need to quadruple its balance sheet to raise per-share earnings by eleven cents.

**The intuition this teaches: earnings per share can be manufactured by adding capital. Return on capital cannot. When the two move in opposite directions for several years, believe the second one.**

This is also the signal that the short seller Jim Chanos has said drew Kynikos Associates to Enron in late 2000 — not a clever forensic reconstruction, but the observation that a company celebrated as a high-return network business was earning a return on capital below its cost of capital.

## 2. Mark-to-market: how twenty years of profit lands in one quarter

Enron began using mark-to-market accounting for its energy trading business in the early 1990s, and by 1999 was applying the formal industry standard, EITF 98-10, to energy trading and risk management contracts. The mechanism is legitimate. The problem is what happens when you apply it to a contract that has no observable price and a twenty-year life.

![Booking a long-dated contract at modelled NPV puts the whole profit in this year's income statement and none of it in this year's bank account](/imgs/blogs/enron-a-forensic-re-read-of-spes-and-mark-to-market-3.webp)

The timeline above is the entire mechanism. Reported profit appears as one block at time zero. Cash appears as twenty small arrows across twenty years. Both describe the same contract.

#### Worked example: booking a twenty-year contract on day one

Round illustrative numbers, so the arithmetic is checkable.

You sign a twenty-year contract to supply electricity. Your model says it earns a **\$10 million** gross margin every year for twenty years. Your discount rate is **8%**.

Step one — the annuity factor. The present value of \$1 a year for twenty years at 8% is:

$$\frac{1 - (1.08)^{-20}}{0.08} = \frac{1 - 0.2145}{0.08} = 9.8181$$

Step two — the NPV:

- \$10 million × 9.8181 = **\$98.2 million**

Step three — book it. Under mark-to-market, that \$98.2 million is income *now*. This quarter's income statement shows \$98 million of profit from a contract on which nothing has been delivered and nothing has been paid.

Step four — the cash. In year one, \$10 million arrives. In year two, another \$10 million. The cash flow statement will show \$10 million a year for twenty years, and the income statement will show nothing further from this contract ever again, because the profit was all recognised in year zero.

Step five — what this does to next year. Having booked \$98 million, you start next year with a hole. To grow reported earnings, you need a *new* contract at least as large. Then a larger one. The accounting creates a treadmill in which the company must sign progressively bigger long-dated deals to show growth, regardless of whether those deals are good.

**The intuition this teaches: mark-to-market on long-dated contracts converts a company's growth rate into a function of its deal-signing rate, not its cash generation. The treadmill is a feature of the accounting, not of the business.**

### The assumptions were unobservable, and small changes moved everything

There is a second problem, and it is worse than the timing one. To compute that \$98.2 million you needed two numbers nobody can verify: the annual margin twenty years out, and the discount rate.

There was no liquid market for twenty-year electricity in 2000. There is barely one now. So the forward price curve past a few years was a modelling choice. And the discount rate was a modelling choice. Both choices sit inside the company.

![Nobody could check the marks: the same twenty-year contract books anywhere from \$45m to \$138m of day-one profit depending on assumptions no outsider could observe](/imgs/blogs/enron-a-forensic-re-read-of-spes-and-mark-to-market-4.webp)

#### Worked example: how far one honest disagreement moves the profit

Same illustrative contract. Change only the discount rate, from 8% to 12% — a difference two competent analysts could argue about all afternoon without either being unreasonable.

Annuity factor at 12%:

$$\frac{1 - (1.12)^{-20}}{0.12} = \frac{1 - 0.1037}{0.12} = 7.4694$$

- At 8%: \$10m × 9.8181 = **\$98.2 million** of day-one profit
- At 12%: \$10m × 7.4694 = **\$74.7 million** of day-one profit

A \$23.5 million swing in reported profit, on one contract, from one assumption, with no change to the business whatsoever.

Now let the margin assumption move too, between \$6 million and \$14 million a year:

| Assumed annual margin | Day-one profit at 8% | Day-one profit at 12% |
| --- | --- | --- |
| \$6m | \$58.9m | \$44.8m |
| \$10m | \$98.2m | \$74.7m |
| \$14m | \$137.5m | \$104.6m |

The same contract supports anything from **\$44.8 million to \$137.5 million** of immediate profit — a range of more than three times — entirely within the space of defensible assumptions.

**The intuition this teaches: when profit is a model output, the audit question is not "did they add it up correctly" but "who chose the inputs, and what were they paid if the answer came out high". No amount of arithmetic checking substitutes for an observable price.**

This is where the auditor's role becomes almost impossible, and why [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) matters so much here. An auditor can verify that a model was applied consistently. Verifying that a twenty-year power price assumption is *correct* is not an audit procedure. It is a forecast.

### The tell in the filings: earnings without cash, at scale

Mark-to-market profit shows up in the statements as a growing pile of non-cash assets. That is precisely the \$21,006 million of "assets from price risk management activities". The forensic reading is mechanical:

- Reported profit rises.
- The matching asset is a valuation, not a receivable from a customer who owes money.
- Cash from operations does not rise with profit, unless something else is supplying it.

That last clause turns out to matter enormously, and it is the reason the accruals screens failed on Enron. We come back to it in section 5.

## 3. The SPE web: Chewco, LJM, and the three percent that wasn't

The second engine solved a different problem. Mark-to-market manufactured profit. It did not manufacture *buyers* for investments that were losing money, and it did not remove debt.

Enron's merchant investment portfolio — stakes in power projects, pipelines, water companies, and a large number of late-1990s technology companies — contained assets that were falling in value and assets whose debt Enron did not want on its balance sheet. Selling them to a real buyer would have crystallised real losses. So Enron built entities to sell them to.

![The pre-2003 rule asked for 3% of outside equity at risk; collateralise that equity and the whole vehicle consolidates](/imgs/blogs/enron-a-forensic-re-read-of-spes-and-mark-to-market-5.webp)

The figure above states the rule and the way it breaks. Everything that follows is an instance of the right-hand panel.

### Chewco: the cleanest violation in the file

Chewco is the best place to start, because the numbers are precise, and because it is the vehicle whose retroactive consolidation caused most of the eventual restatement.

The account below follows the **allegations in the SEC's civil complaint against Andrew Fastow**, filed in 2002. These are allegations in an enforcement action, framed as such — but they are allegations about arithmetic, and Enron's own 8-K subsequently confirmed the accounting consequence.

In 1993, Enron and the California Public Employees' Retirement System (CalPERS) formed a joint venture partnership called **JEDI** — Joint Energy Development Investments. Enron was the general partner and contributed \$250 million in Enron stock; CalPERS was the limited partner and contributed \$250 million in cash. Because CalPERS was a large, genuinely independent equity investor, Enron did not consolidate JEDI, and JEDI's debt did not appear in Enron's statements. That was legitimate.

In 1997 Enron wanted CalPERS out of JEDI so CalPERS would commit to a larger successor fund. CalPERS set a deadline of **6 November 1997** and a price of **\$383 million**. Enron needed a buyer for that interest who was independent enough to preserve non-consolidation. It did not find one. It created one: **Chewco Investments, L.P.**

Chewco's funding, per the SEC's complaint:

- Two bridge loans of **\$191.5 million each**, from Barclays Bank and Chase Manhattan Bank — \$383 million in total — **with repayment guaranteed by Enron**.
- At year-end 1997, that bridge was replaced with a structure whose purported outside equity was **approximately \$11.49 million**.
- Of that \$11.49 million, **approximately \$11.36 million was itself borrowed from Barclays** by entities set up and controlled by an Enron employee, Michael Kopper.
- Those Barclays loans were **secured by approximately \$6.58 million in cash** — cash generated by JEDI's own November 1997 sale of an asset, held in accounts fully pledged to Barclays.
- The remainder of the "outside equity" — **approximately \$125,000** — came from Kopper and his domestic partner.

#### Worked example: the three percent test, applied to Chewco

Run the rule.

Step one — what the rule required. Outside equity of at least 3% of the vehicle's assets, genuinely at risk. On a \$383 million transaction:

- 3% × \$383 million = **\$11.49 million**

Step two — what was presented. Exactly \$11.49 million. The number matches the requirement to the cent, which is itself the tell: this is not what an investor decided to risk, it is what a rule demanded.

Step three — what was actually at risk. Strip out the \$11.36 million that was borrowed, because borrowed money that the lender is protected on is not the borrower's equity at risk:

- \$11.49 million − \$11.36 million = **\$125,000**

Step four — express that as a percentage of the vehicle:

- \$125,000 ÷ \$383,000,000 = **0.033%**

Step five — the gap. The rule wanted 3%. The structure delivered 0.033%:

- \$11.49 million ÷ \$125,000 = **92 times** short

And the \$6.58 million of pledged cash makes it worse still, because that collateral came from inside the structure itself, protecting Barclays against the loss the equity was supposed to absorb.

**The intuition this teaches: the three percent test is not a calculation, it is a question about who loses money first. Any structure where you can trace the outside investor's money back to the sponsor or to the vehicle's own assets has no outside investor.**

The consequence, as Enron itself stated in its Form 8-K of 8 November 2001: Chewco "did not meet the accounting criteria to qualify as an unconsolidated SPE", and because Chewco was JEDI's limited partner, JEDI failed too. Both should have been consolidated **beginning in November 1997**.

That is a four-year error, and we will price it exactly in section 7.

### LJM1, LJM2, and a waiver of the code of conduct

In June 1999, Enron's chief financial officer proposed an entity called **LJM Cayman, L.P.** — LJM1 — and Enron's board granted him a limited waiver of the company's conflict-of-interest rules so that he could serve as its general partner. LJM1's two limited partners were entities owned by Credit Suisse First Boston and National Westminster Bank, each investing **\$7.5 million**. In October 1999 a second and much larger vehicle followed, **LJM2 Co-Investment, L.P.**, with the same arrangement.

The SEC's complaint alleges an undisclosed agreement — referred to internally as the "Global Galactic" agreement — under which any Enron transaction that resulted in a loss to the LJM entities would be made good in later deals, so that the LJM entities would not lose money in their dealings with Enron. If that is what happened, then every LJM vehicle was consolidatable from the start, because none of the outside equity was ever at risk.

### The Raptors, and outside equity that was returned before it was risked

In April 2000, Enron created **Talon LLC**, the vehicle behind the structure known as **Raptor I**. Per the SEC's complaint, Talon was funded mainly by Enron itself, through a promissory note and Enron's own stock; the remaining **\$30 million** came from LJM2, "representing the purported three percent outside equity required for Talon to be off Enron's balance sheet".

Note what \$30 million at three percent implies about the size of the vehicle: roughly **\$1 billion** of assets, supported by \$30 million of nominally independent money.

The complaint then alleges the detail that destroys it. Under an undisclosed side deal, Enron agreed that *before* conducting any hedging activity with Talon, it would return LJM2's full investment plus a guaranteed return. After investing \$30 million, LJM2 received **\$41 million** from Talon on or about **7 September 2000** — its capital back plus **\$11 million** of profit. To disguise the payment, Enron purchased a put option on its own stock from Talon for a \$41 million premium, then settled it early so Talon could distribute the cash.

A put option on your own stock is a bet that your own share price will fall. Enron, according to the complaint, bought one for \$41 million with no business purpose other than moving money to LJM2.

Once LJM2's capital had been returned, there was no outside equity at risk at all — and Talon should have been consolidated. It was not.

The same complaint alleges a related manipulation of the mark-to-market machinery: a hedge relating to Enron's holding in AVICI Systems was **backdated to 3 August 2000**, the date AVICI had traded at an all-time high of **\$163.50**, booking **\$75 million** of additional mark-to-market gains that would not otherwise have been recognised. The two engines from Figure 1 were not separate. The SPE web existed partly to feed the mark-to-market engine better inputs.

### What Note 16 actually said

Here is the part that should change how you read filings.

![Note 16 of the FY2000 10-K, drawn as a diagram: \$1.2bn of assets out, \$1.2bn of notes receivable back, and \$500m of revenue on derivatives with the same counterparty](/imgs/blogs/enron-a-forensic-re-read-of-spes-and-mark-to-market-6.webp)

Note 16 of Enron's FY2000 Form 10-K, "Related Party Transactions", opens:

> In 2000 and 1999, Enron entered into transactions with limited partnerships (the Related Party) whose general partner's managing member is a senior officer of Enron. The limited partners of the Related Party are unrelated to Enron. Management believes that the terms of the transactions with the Related Party were reasonable compared to those which could have been negotiated with unrelated third parties.

Then it describes the transactions. Enron disclosed, in its own words, that it:

- contributed to newly-formed entities "assets valued at approximately \$1.2 billion, including \$150 million in Enron notes payable, 3.7 million restricted shares of outstanding Enron common stock and the right to receive up to 18.0 million shares of outstanding Enron common stock in March 2003";
- received in return "a special distribution from the Entities in the form of \$1.2 billion in notes receivable";
- received a further "\$309 million in notes receivable, of which \$259 million is recorded at Enron's carryover basis of zero";
- disclosed that "Cash in these Entities of \$172.6 million is invested in Enron demand notes";
- "paid \$123 million to purchase share-settled options from the Entities on 21.7 million shares of Enron common stock";
- entered into derivative transactions with the entities "with a combined notional amount of approximately \$2.1 billion to hedge certain merchant investments and other assets"; and
- **"recognized revenues of approximately \$500 million related to the subsequent change in the market value of these derivatives"**.

Note 16 also discloses that Enron sold part of its dark fibre inventory to the same related party for \$30 million cash and a \$70 million note receivable, and "recognized gross margin of \$67 million on the sale".

Read those bullets as a closed loop. Enron sent out its own notes payable and its own shares. It took back notes receivable. The counterparty's cash was lent straight back to Enron as demand notes. And Enron recognised half a billion dollars of revenue on derivatives with that counterparty.

No part of this required investigation. It required reading Note 16.

### The disclosure got vaguer as the numbers got bigger

There is one more tell, and it is available only if you compare two consecutive filings — which is exactly why comparing consecutive filings is a habit worth building, as [the footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) argues.

Note 16 of the **FY1999** 10-K names the entities outright:

> In June 1999, Enron entered into a series of transactions involving a third party and LJM Cayman, L.P. (LJM). LJM is a private investment company which engages in acquiring or investing in primarily energy-related investments. **A senior officer of Enron is the managing member of LJM's general partner.**

It goes on to name "LJM2 Co-Investment, L.P. (LJM2)", to disclose that LJM2 acquired approximately \$360 million of merchant assets and investments from Enron in the fourth quarter of 1999 on which Enron recognised pre-tax gains of approximately \$16 million, and to note — obliquely but unmistakably — that "an officer of Enron has invested in the limited partner of JEDI and from time to time acts as agent on behalf of the limited partner's management". That limited partner was Chewco.

In the **FY2000** 10-K, the names are gone. LJM1 and LJM2 become "the Related Party" and "the Entities". The transactions have grown by an order of magnitude, and the disclosure has become less specific.

**A footnote that becomes more generic while the amounts in it become larger is a deliberate act.** Nobody accidentally removes a counterparty's legal name from a footnote.

## 4. Why the "hedge" was circular

The word doing the most work in Note 16 is *hedge*. Enron said the derivatives with the Entities hedged its merchant investments. A hedge is a position that gains when the thing you own loses, so that the two offset.

For a hedge to work, the counterparty must be able to pay when you need it. That is the entire point. If the counterparty fails exactly when you need the money, you did not own a hedge — you owned a piece of paper describing one.

![The Entities could absorb Enron's losses only while Enron's share price was rising, which is exactly when there were no losses to absorb](/imgs/blogs/enron-a-forensic-re-read-of-spes-and-mark-to-market-7.webp)

The Entities' ability to pay rested on Enron's own stock, because that is what Enron had capitalised them with — 3.7 million restricted shares, the right to 18.0 million more, share-settled options on 21.7 million shares, share-settled collars on 15.4 million shares. All of it Enron equity.

So consider the two states of the world:

- **Enron stock rising.** The Entities' collateral appreciates. Their capacity to pay grows. But a rising Enron share price came in the same conditions as rising markets generally, when Enron's merchant investments were not falling — so there was little to hedge.
- **Enron stock falling.** Enron's technology and energy investments fall. And the Entities' only substantial asset, Enron stock, falls at the same time. Their capacity to pay collapses precisely when the hedge is called.

The hedge and the thing being hedged had the same driver. This is not a subtle flaw discovered later; it is visible from the structure alone. The SEC's complaint describes the mechanism plainly when discussing the earlier Rhythms NetConnections hedge, noting that when Enron and Rhythms shares both rose in early 2000, this made the vehicle's "main asset (its Enron shares) more valuable while decreasing its potential liability on the Rhythms put option".

An entity whose asset and liability both track your share price is not a counterparty. It is a mirror.

#### Worked example: Cuiabá, where both engines ran at once

This one is real, and it shows the two engines working together. The figures are allegations from the SEC's complaint against Fastow.

Enron held about a 65% interest in a troubled power plant and pipeline project in Cuiabá, Brazil. It had two problems: the project's debt was heading for Enron's balance sheet, and Enron wanted to mark a related power supply contract to market, which required not controlling the project. No independent buyer would take an interest in it.

Step one — the sale. On **30 September 1999**, Enron sold LJM1 a **13% interest** for **\$11.3 million**. With that interest went a board seat, which was the basis for concluding Enron no longer controlled the project.

Step two — what the sale bought. Deconsolidation kept the project's debt off Enron's balance sheet, and permitted mark-to-market treatment of the power supply contract. Per the complaint, this "enabled Enron to recognize a total of approximately **\$65 million** of income in the third and fourth quarters of 1999".

Step three — the ratio. Compare what was sold to what was recognised:

- \$65 million of income from an \$11.3 million sale = **5.75 times the sale price**

Step four — why LJM1 agreed. Per the complaint, because Enron agreed in an undisclosed side deal to repurchase the interest if necessary and to guarantee LJM1 a profit — an agreement kept out of the final documents partly out of concern that Arthur Andersen would not approve the sale with such a provision.

Step five — the settlement. On **15 August 2001**, after further cost overruns had reduced Cuiabá's value, Enron bought the interest back for **\$13,752,000** — a price calculated to give LJM1 a profit regardless of the project's performance.

So Enron sold 13% of a failing project for \$11.3 million, booked \$65 million of income on the strength of that sale, and bought the stake back two years later for \$13.75 million.

**The intuition this teaches: when a sale's purpose is an accounting outcome rather than a transfer of risk, the price paid is small and the income recognised is large. A wildly high ratio of recognised income to sale proceeds is one of the most reliable quantitative markers of a manufactured transaction.**

The complaint alleges the same pattern in a smaller December 1999 transaction involving three Nigerian power barges, which permitted Enron to record approximately \$12 million of earnings in the fourth quarter of 1999. Both are instances of the general mechanism this series describes in [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue).

## 5. What the statements told a careful reader in 1999 and 2000

Now the central question. Forget everything discovered later. You have the FY1999 and FY2000 10-Ks on your desk, and the toolkit from this series. What fires?

![Total assets grew from \$16.1bn to \$65.5bn between 1996 and 2000 while reported net income went from \$584m to \$979m — and the November 2001 restatement cut 1997's to \$9m](/imgs/blogs/enron-a-forensic-re-read-of-spes-and-mark-to-market-8.webp)

### The accruals ratio does not fire, and that is the important finding

The [accruals ratio](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly) is one of the strongest general-purpose earnings-quality screens there is. The cash-flow version subtracts cash from operations from net income and scales by average total assets. High positive accruals mean profit is arriving as promises rather than cash — the classic manipulation signature.

#### Worked example: computing Enron's accruals ratio from the filed lines

$$\text{Accruals ratio} = \frac{\text{Net income} - \text{Cash from operations}}{\text{Average total assets}}$$

| Year | Net income | Cash from ops | Average total assets | Accruals ratio |
| --- | --- | --- | --- | --- |
| 1997 | 105 | 501 | 19,780 | **−2.00%** |
| 1998 | 703 | 1,640 | 26,386 | **−3.55%** |
| 1999 | 893 | 1,228 | 31,366 | **−1.07%** |
| 2000 | 979 | 4,779 | 49,442 | **−7.69%** |

Take 2000 step by step:

- Net income minus cash from operations: 979 − 4,779 = **−3,800**
- Average total assets: (33,381 + 65,503) ÷ 2 = **49,442**
- Ratio: −3,800 ÷ 49,442 = **−7.69%**

Every year is **negative**. Cash from operations exceeded net income every single year, and by a very wide margin in 2000. On this screen Enron does not look like a manipulator. It looks like a conservative company that collects more cash than it reports as profit.

This is not a defect in the ratio. It is the ratio being fed a manipulated denominator — because the cash flow statement was itself being managed.

Enron's own MD&A explains the 2000 increase: cash from operations rose \$3,551 million "primarily reflecting decreases in working capital, positive operating results and **a receipt of cash associated with the assumption of a contractual obligation**". That final clause describes borrowing. A prepay transaction — cash received today for commodity to be delivered later — is economically a loan, and classifying its proceeds as operating cash flow moves borrowed money into the line investors treat as the quality check on earnings. This series covers the mechanism in [cash-flow statement manipulation](/blog/trading/forensic-accounting/cash-flow-statement-manipulation-classification-shifting), and the Enron bankruptcy examiner's reports later analysed these prepay structures in detail.

**The intuition this teaches: the accruals ratio assumes cash from operations is the honest number. When a company can classify borrowings as operating cash flow, the screen inverts — a suspiciously *clean* accruals ratio, alongside a balance sheet that is exploding, is itself the anomaly.**

The defensible reading available at the time was not "the accruals ratio is fine, move on". It was: cash from operations swung from \$501 million to \$1,640 million to \$1,228 million to \$4,779 million in four years. **A real operating business does not have a cash flow that moves like that.** Lumpiness of that magnitude in the supposedly most stable line in the statements is a question, not a reassurance.

### Altman's Z does not fire either

The [Altman Z-score](/blog/trading/forensic-accounting/the-altman-z-score-predicting-financial-distress) combines five ratios into a single distress score, with a distress zone below 1.81, a grey zone from 1.81 to 2.99, and a safe zone above 2.99.

#### Worked example: Enron's Z-score at the end of 2000

Every input below is read off the FY2000 10-K. Total liabilities are total current liabilities (28,406) plus long-term debt (8,550) plus deferred credits and other liabilities (13,759) = **50,715**. Working capital is current assets minus current liabilities: 30,381 − 28,406 = **1,975**. For the market value of equity we use the figure Enron itself printed on the 10-K cover page: the aggregate market value of voting stock held by non-affiliates on 15 February 2001, **\$60,207 million**.

$$Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5$$

| Ratio | Definition | Value | Weighted |
| --- | --- | --- | --- |
| X1 | Working capital / total assets = 1,975 / 65,503 | 0.0302 | 0.036 |
| X2 | Retained earnings / total assets = 3,226 / 65,503 | 0.0492 | 0.069 |
| X3 | EBIT / total assets = 1,953 / 65,503 | 0.0298 | 0.098 |
| X4 | Market equity / total liabilities = 60,207 / 50,715 | 1.1872 | 0.712 |
| X5 | Sales / total assets = 100,789 / 65,503 | 1.5387 | 1.539 |

- **Z = 2.45** → the **grey zone**. Not distress. *(Computed from filed lines.)*

Look at where the score came from. Of the 2.45, fully **1.54 came from X5** — sales over assets — and that ratio is driven by \$100.8 billion of gross trading revenue which, as we established in section 1, is nearly meaningless for a trading book. A further **0.71 came from X4**, the market's own valuation of Enron's equity. Together those two terms supply 92% of the score, and both are contaminated: one by an inflated revenue convention, one by the very market opinion the score is supposed to help you challenge.

The three ratios built from the balance sheet and the operating line — X1, X2, X3 — contribute a combined **0.20**.

Now use the variant designed for companies whose asset turnover is not comparable, which drops the sales ratio and uses book equity:

$$Z'' = 6.56X_1 + 3.26X_2 + 6.72X_3 + 1.05X_4$$

with X4 now book equity over total liabilities = 11,470 / 50,715 = 0.2262:

- 6.56 × 0.0302 = 0.198
- 3.26 × 0.0492 = 0.161
- 6.72 × 0.0298 = 0.200
- 1.05 × 0.2262 = 0.237
- **Z'' = 0.80** → below the 1.1 distress line. *(Computed from filed lines.)*

**Two models, one balance sheet, and answers on opposite sides of their own distress thresholds.** The difference is entirely whether you credit Enron for gross trading revenue and for its market capitalisation. Strip both out, and the company that Z placed comfortably in the grey zone lands in distress.

**The intuition this teaches: a composite score is only as honest as its most contaminated input. When one term supplies most of a score, you have not measured a company — you have measured that term.**

### What did fire

Four things, none of them a packaged ratio.

**One: return on assets, falling while EPS rose.** 3.62% in 1996 to 1.49% in 2000, as computed in section 1.

**Two: assets growing far faster than earnings.** In 2000 alone, total assets +96%, net income +9.6%. Over four years, assets ×4.06, net income ×1.68.

**Three: the composition of the balance sheet.** "Assets from price risk management activities" reaching \$21,006 million, 32% of total assets, up 4.1 times in one year. A third of the company's assets were its own valuation opinion. Reading the balance sheet for *what kind* of asset is growing, rather than just how fast, is the habit this series builds in [reading the balance sheet](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here).

**Four: Note 16.** \$1.2 billion out, \$1.2 billion of notes back, \$172.6 million of the counterparty's cash lent to Enron, \$500 million of revenue recognised on derivatives with that counterparty, \$67 million of gross margin on a fibre sale to it. And the counterparty's general partner's managing member was an Enron officer.

There is a fifth, available only to someone comparing filings across years, and it is the most damning of all: **the 1997 comparative column kept changing.** The FY1997 10-K reported total assets of \$23,422 million and cash from operations of \$501 million. The FY1998 10-K, showing 1997 as its comparative, reported \$22,552 million and \$211 million. Those were reclassifications rather than fraud restatements — but a company whose prior-year figures move when nobody is looking is a company whose prior-year figures are worth checking.

### What about the Beneish M-Score?

The [Beneish M-Score](/blog/trading/forensic-accounting/the-beneish-m-score-detecting-earnings-manipulation) is the other major packaged screen in this series, and Enron is bound up with its folklore — a Cornell graduate class is documented as having analysed Enron and concluded it should be sold in 1998. That sibling post covers the episode and the model's genuine limits.

We do not compute an M-Score for Enron here, deliberately. The eight inputs require two consecutive years of specific line items on a consistent basis, and Enron's revenue convention, its shifting comparatives, and its 1999 accounting-change charge make several of the indices unstable enough that any single number would carry more precision than the underlying data supports. A screen that requires clean inputs cannot be trusted on a company whose inputs are the thing being manipulated — which is, itself, the lesson.

## 6. Which red flags fired, and in what order

Putting section 5 into a table changes the conclusion of this post from a victory lap into something more useful.

![Run the series' own toolkit over Enron's FY2000 filing and the ratio screens mostly pass it — the tells are in Note 16 and in the return on assets](/imgs/blogs/enron-a-forensic-re-read-of-spes-and-mark-to-market-9.webp)

| Order | Signal | When it was available | Effort to see it |
| --- | --- | --- | --- |
| 1 | Return on capital below cost of capital while EPS rose | From the 1997 statements onward | Two lines, one division |
| 2 | Balance sheet growing far faster than earnings | Visibly from 1997, screaming in 2000 | Two lines, one division |
| 3 | Related-party partnerships run by a senior officer | FY1999 10-K, Note 16, filed March 2000 | Read one footnote |
| 4 | Revenue recognised on derivatives with that related party | FY2000 10-K, Note 16, filed April 2001 | Read one footnote |
| 5 | Counterparty's cash invested in Enron's own demand notes | FY2000 10-K, Note 16 | Read one footnote carefully |
| 6 | A third of assets being model-valued positions | FY2000 balance sheet | Add two lines |
| 7 | Cash from operations swinging 4× year to year | Across the 1997–2000 filings | Compare four filings |
| 8 | The prior-year comparative column changing | FY1998 10-K versus FY1997 10-K | Compare two filings |

The ordering has a clear structure, and it is not the structure most people assume.

**The earliest and cheapest signals were ratios of profit to capital.** They required no footnotes and no forensic skill. They fired years before the collapse, and they fired continuously.

**The decisive signals were all prose.** Note 16 in two consecutive years told a reader nearly everything that mattered about the SPE web: the conflict, the scale, the circularity, and the revenue. It cost one afternoon.

**The packaged screens fired last, or not at all.** The accruals ratio never fired. Z never fired. Z'' fired, but only because it discards the two inputs Enron had inflated.

That is the honest lesson of Enron for a toolkit like this one: **automated screens are for triage, not for verdicts.** They tell you which of a thousand companies deserves an afternoon. They cannot tell you what you will find in that afternoon, and a clean screen is not evidence of anything — particularly when the company is large, celebrated, complicated, and audited by a firm earning substantial consulting fees alongside its audit fee.

## 7. The unravelling, priced from the filings

The end came fast, and the filings record it precisely.

**Third quarter 2001.** Enron reported a net loss of **\$618 million** for the quarter, and disclosed a **\$1.2 billion** reduction in shareholders' equity. The equity reduction was an accounting correction, not a trading loss: Enron had recorded notes receivable from the SPEs as an *increase* in shareholders' equity, when amounts receivable for stock issued should have reduced it.

The 8 November 2001 restatement table prices that error exactly. The "Raptor equity adjustment" line shows **\$172 million** against 2000 and **\$1,000 million** against the first and second quarters of 2001 — \$1,172 million in total, the "\$1.2 billion" that had been announced.

**8 November 2001.** Enron filed a Form 8-K announcing that it would restate its financial statements for 1997 through 2000 and the first two quarters of 2001. The filing states three conclusions: Chewco's financial activities "should have been consolidated beginning in November 1997"; JEDI's likewise; and a wholly-owned subsidiary of LJM1 — the vehicle that had "hedged" the Rhythms NetConnections investment — "should have been consolidated into Enron's financial statements beginning in 1999".

Here is Table 1 of that 8-K, which is the single most useful page in the entire Enron record, in millions of dollars:

| | 1997 | 1998 | 1999 | 2000 |
| --- | --- | --- | --- | --- |
| Net income as reported | 105 | 703 | 893 | 979 |
| Consolidation of JEDI and Chewco | (45) | (107) | (153) | (91) |
| Consolidation of LJM1 subsidiary | — | — | (95) | (8) |
| Prior year proposed audit adjustments and reclassifications | (51) | (6) | (2) | (33) |
| **Net income restated** | **9** | **590** | **643** | **847** |
| Diluted EPS as reported (\$) | 0.16 | 1.01 | 1.10 | 1.12 |
| **Diluted EPS restated (\$)** | **(0.01)** | **0.86** | **0.79** | **0.97** |
| Debt as reported | 6,254 | 7,357 | 8,152 | 10,229 |
| **Debt restated** | **6,965** | **7,918** | **8,837** | **10,857** |
| Equity as reported | 5,618 | 7,048 | 9,570 | 11,470 |
| **Equity restated** | **5,305** | **6,600** | **8,736** | **10,306** |

#### Worked example: what four years of non-consolidation was worth

Step one — the total earnings overstatement. Sum the reported net income for 1997 to 2000, then the restated:

- Reported: 105 + 703 + 893 + 979 = **2,680**
- Restated: 9 + 590 + 643 + 847 = **2,089**
- Overstatement: **591**, or **22%** of four years of reported profit

Step two — attribute it. The three restatement lines sum exactly:

- Chewco and JEDI: 45 + 107 + 153 + 91 = **396** (67% of the total)
- LJM1 subsidiary: 95 + 8 = **103** (17%)
- Prior year proposed audit adjustments: 51 + 6 + 2 + 33 = **92** (16%)
- Total: 396 + 103 + 92 = **591** ✓

Step three — note what the third line is. "Prior year proposed audit adjustments" are corrections the auditor had **proposed and the company had declined to record**, on the basis that they were individually immaterial. In 1997 that line is \$51 million against reported net income of \$105 million — the passed-over adjustments alone were **49%** of that year's reported profit. This is why "immaterial" is a word to read with suspicion in an audit context.

Step four — the debt that had been missing. The Chewco and JEDI consolidation added **\$711 million** of debt at year-end 1997, against \$6,254 million reported:

- 711 ÷ 6,254 = **11.4%** understatement of reported debt

and it added \$561 million, \$685 million and \$628 million at the following three year-ends.

Step five — the worst single year. 1997 net income fell from \$105 million to \$9 million, a **91%** cut, and diluted EPS from \$0.16 to **negative \$0.01**. A year Enron had reported as marginally profitable was, restated, a loss per share.

**The intuition this teaches: non-consolidation compounds. Each year the vehicle stays off the balance sheet, the gap between reported and real grows, and the eventual correction is not one year's error but every year's error arriving at once.**

**2 December 2001.** Enron Corp filed for protection under Chapter 11 of the US Bankruptcy Code in the Southern District of New York, weeks after a rescue merger with Dynegy collapsed. Enron's common stock had traded as high as **\$90.75** during the third quarter of 2000, per the price table in its own FY2000 10-K.

**1 February 2002.** The *Report of Investigation by the Special Investigative Committee of the Board of Directors of Enron Corp.* — universally called the **Powers Report**, after committee chair William Powers — was released. It remains the most important single account of how the SPE transactions were approved internally, and it is the source most subsequent narratives draw on for the internal decision-making.

**2002 onward: the enforcement record.** Arthur Andersen was convicted of obstruction of justice in June 2002 and ceased auditing public companies; the Supreme Court unanimously reversed that conviction in *Arthur Andersen LLP v. United States*, 544 U.S. 696 (2005), on the ground that the jury instructions had been flawed — by which time the firm no longer meaningfully existed. Andrew Fastow pleaded guilty to two counts of conspiracy in January 2004 and cooperated with prosecutors. Jeffrey Skilling was convicted in May 2006; the Supreme Court narrowed the honest-services fraud theory used against him in *Skilling v. United States*, 561 U.S. 358 (2010), and he was resentenced in 2013. Kenneth Lay was convicted in May 2006 and died before his appeal could be heard, which under the doctrine of abatement vacated his conviction.

**30 July 2002: Sarbanes-Oxley.** The Act responds to specific failures visible in the story above:

| Section | What it requires | Which Enron failure it addresses |
| --- | --- | --- |
| Title I | Creates the PCAOB to inspect audit firms | Self-regulation of auditors |
| §201 | Bars auditors from most non-audit services for audit clients | Consulting fees compromising audit independence |
| §302, §906 | CEO and CFO must personally certify the financial statements | No individual accountable for the filings |
| §401(a) | Requires disclosure of material off-balance-sheet arrangements | Chewco, JEDI, the Raptors |
| §404 | Management must assess, and the auditor attest to, internal control over financial reporting | Controls that permitted undisclosed side agreements |

§401(a) is the most direct descendant of this post. It exists because a company could carry billions of dollars of obligations in vehicles it controlled and disclose them in prose vague enough to be unusable.

## Common misconceptions

**"Enron's fraud was hidden, so nobody could have known."** The most important single disclosure — Note 16 — was published in the annual report. It named the conflict in 1999 and quantified roughly a billion dollars of related-party dealing in 2000, including \$500 million of revenue recognised on derivatives with that party. What was hidden were the side agreements. What was disclosed was enough.

**"Mark-to-market accounting is inherently fraudulent."** It is the correct treatment for a position with an observable market price, and pretending otherwise creates its own distortions. The failure at Enron was applying it to twenty-year contracts with no observable price, where the inputs were chosen by the people paid on the output. The forensic question is never "is this marked to market" but "marked to *what*, chosen by *whom*".

**"Special-purpose entities are a fraud device."** SPEs are how securitisation, project finance and most large infrastructure funding work. The Enron vehicles failed a specific and narrow test — whether the outside equity was genuinely at risk. That test is the whole thing, and it was failed by ordinary means: guarantees, side letters, collateral supplied from inside the structure.

**"The accruals ratio would have caught it."** It would not have. Computed from the filed lines, Enron's cash-flow accruals ratio was negative in every year from 1997 to 2000 and most negative — that is, most apparently conservative — in 2000. The screen was defeated because the cash flow statement was managed alongside the income statement.

**"A high Altman Z-score means a company is safe."** Enron's Z was 2.45 at the end of 2000, in the grey zone, with 92% of the score coming from gross revenue over assets and from the market's own valuation of the equity. A composite score inherits the reliability of its inputs, and can be propped up by exactly the market opinion you are trying to test.

**"The auditors simply missed it."** The restatement's own third line shows that \$92 million of adjustments across four years had been *proposed* and declined as immaterial, including \$51 million in a year that reported \$105 million of net income. That is not a failure of detection. It is a failure of insistence.

## How it shows up in real markets

### 1. WorldCom, 2002

The next great American accounting failure used almost the opposite mechanism — capitalising ordinary operating costs so they became assets rather than expenses — but produced the same statement signature: reported profit rising while the capital base swelled and returns on capital fell. The general mechanics are in [capitalizing costs to inflate profit](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move). The forensic lesson repeats: when profit growth and capital growth diverge for years, the balance sheet is telling you where the earnings came from.

### 2. Lehman Brothers and Repo 105, 2008

Lehman used repurchase transactions structured to qualify as sales rather than financings, removing assets from the balance sheet at quarter-end and returning them days later. It is Enron's question in a different instrument: does the accounting treatment match the transfer of risk? As at Enron, the disclosure was thin, the mechanism was documented internally, and the eventual correction arrived all at once.

### 3. Wirecard, 2020

Wirecard's failure centred on cash that did not exist in trustee accounts in Asia, and on a business whose reported profitability was concentrated in third-party acquiring operations that outsiders could not verify. The structural echo of Enron is the *unverifiable segment*: a large share of reported profit arising in an activity whose economics no external party could independently confirm — the same role that price risk management activities played in Enron's 2000 accounts.

### 4. Valeant Pharmaceuticals, 2015–2016

Valeant's undisclosed relationship with the specialty pharmacy Philidor is the related-party lesson without the derivatives. A company transacting with an entity it effectively controlled, disclosed inadequately, produced revenue that could not survive the relationship becoming public. Note 16 exists to prevent exactly this, and it only works if readers read it.

### 5. The recurring shape

In each case, a company reported results that were too good relative to the cash it produced; the difference accumulated on the balance sheet as an asset whose value depended on the company's own assertions; and disclosure of the mechanism existed but was written to be skimmed past. The instruments change. The shape does not.

## When this matters to you

You are unlikely to be auditing an energy trading book. But the habits this post exercises are the ones that transfer.

**Read the related-party footnote first, before the income statement.** It is short, it is legally required, and it is where the incentive conflicts are named. If a company transacts with entities run by its own officers, you have learned something no ratio will tell you.

**Compare consecutive filings, not just consecutive years within one filing.** The prior-year column in this year's report is not always the number that was published last year. When it moves, find out why.

**Ask what kind of asset is growing, not just how fast.** An asset base that grows through receivables, model-valued positions, goodwill, or "other assets" is a different animal from one that grows through plant and inventory.

**Divide profit by capital, every time.** Return on assets and return on capital are two lines and one division, they are extremely hard to manipulate in both directions at once, and at Enron they were the earliest and most persistent signal available.

**Treat a clean screen as permission to look, not a reason not to.** Enron passed the accruals test and sat in Altman's grey zone in its final full year. A screen that comes back clean on a company whose balance sheet has doubled has not told you the company is fine. It has told you that the screen's inputs are the wrong place to look.

This is educational material about reading financial statements, not investment advice, and nothing here is a recommendation about any security.

## Sources & further reading

**Primary sources behind the headline figures:**

- **Enron Corp, Form 10-K for fiscal year 2000**, filed 2 April 2001, SEC accession `0001024401-01-500010`. Source of: the FY2000 and FY1999 balance sheets; total revenues of \$100,789m; operating income of \$1,953m; net income of \$979m; cash from operations of \$4,779m; total assets of \$65,503m; assets from price risk management activities of \$12,018m current and \$8,988m long-term; shareholders' equity of \$11,470m; retained earnings of \$3,226m; the \$1,899m of income from price risk management activities; Note 16 "Related Party Transactions" in full; the \$90.75 third-quarter 2000 share price high; and the \$60,207,479,342 aggregate market value of voting stock held by non-affiliates as of 15 February 2001.
- **Enron Corp, Form 10-K for fiscal year 1999**, filed 30 March 2000, SEC accession `0001024401-00-000002`. Source of: the 1997 income statement showing operating income of \$15m; the 1997–1999 revenue and net income series; and Note 16 naming LJM Cayman, L.P. and LJM2 Co-Investment, L.P.
- **Enron Corp, Form 8-K**, filed 8 November 2001, SEC accession `0000950129-01-503835`. Source of: Table 1 in full — the restated net income, diluted EPS, debt and equity for 1997–2000; the \$591m cumulative reduction in net income; the Raptor equity adjustment of \$172m and \$1,000m; the \$618m third-quarter 2001 reported net loss; and Enron's statements that Chewco and JEDI should have been consolidated beginning in November 1997 and the LJM1 subsidiary beginning in 1999.
- **Enron Corp, Form 10-K filings for fiscal years 1996, 1997 and 1998** (SEC accessions `0000072859-97-000009`, `0001024401-98-000009`, `0001024401-99-000007`). Source of the 1996–1998 revenue, net income, cash flow and total asset figures as originally reported.
- **Securities and Exchange Commission v. Andrew S. Fastow**, civil complaint filed 2002, [SEC Litigation](https://www.sec.gov/litigation/complaints/comp17762.htm). Source, **as allegations**, of: the JEDI and CalPERS structure; the \$383 million CalPERS buyout price and 6 November 1997 deadline; the two \$191.5 million Barclays and Chase bridge loans guaranteed by Enron; Chewco's \$11.49 million purported outside equity, of which \$11.36 million was borrowed from Barclays and secured by approximately \$6.58 million of pledged cash, leaving approximately \$125,000; the LJM1 and LJM2 formations and the \$7.5 million each from CSFB and NatWest entities; the "Global Galactic" agreement; the Cuiabá sale of a 13% interest for \$11.3 million, the approximately \$65 million of 1999 income, and the \$13,752,000 buyback of 15 August 2001; the Nigerian barge transaction and approximately \$12 million of fourth-quarter 1999 earnings; Talon's \$30 million of purported three percent outside equity and the \$41 million returned to LJM2 on or about 7 September 2000; the AVICI hedge backdated to 3 August 2000 at \$163.50 and \$75 million of additional mark-to-market gains; and the Rhythms NetConnections and Swap Sub structure.
- **SEC litigation release concerning Richard Causey and Jeffrey Skilling** (2004), [LR-18582](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-18582), for the allegations concerning the four Raptor vehicles.

**Documents cited as the source of claims attributed to them:**

- *Report of Investigation by the Special Investigative Committee of the Board of Directors of Enron Corp.* (the **Powers Report**), 1 February 2002 — the primary account of internal approval of the SPE transactions.
- The reports of the Enron bankruptcy examiner, Neal Batson (2002–2003), which analysed the prepay transactions and their treatment in the cash flow statement.
- *Arthur Andersen LLP v. United States*, 544 U.S. 696 (2005); *Skilling v. United States*, 561 U.S. 358 (2010).
- The Sarbanes-Oxley Act of 2002, Pub. L. 107-204, signed 30 July 2002.

**Further reading on this blog:**

- [Enron's 2001 accounting fraud](/blog/trading/finance/enron-2001-accounting-fraud) — the narrative history, the culture, the auditor conflict and the human cost.
- [Off-balance-sheet financing and special-purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities) — the SPE and VIE rules in detail, including the Raptor I and Talon structures.
- [Related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing) — how to find and test these transactions in any filing.
- [The accruals ratio and the accruals anomaly](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly) — the screen that did not fire here, and why it usually does.
- [The Altman Z-score](/blog/trading/forensic-accounting/the-altman-z-score-predicting-financial-distress) — the model, its three variants, and its zones of ignorance.
- [The Beneish M-Score](/blog/trading/forensic-accounting/the-beneish-m-score-detecting-earnings-manipulation) — the eight-variable manipulation screen and the Cornell episode.
- [Cash-flow statement manipulation and classification shifting](/blog/trading/forensic-accounting/cash-flow-statement-manipulation-classification-shifting) — prepays, round-trip cash, and why operating cash flow is not automatically trustworthy.
- [Reading the balance sheet](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here) — asset composition as a forensic signal.
- [The footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) — reading the prose where disclosure actually lives.
</content>
