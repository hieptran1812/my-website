---
title: "Lehman's Repo 105 and the Art of Window Dressing a Balance Sheet"
date: "2026-08-12"
publishDate: "2026-08-12"
description: "How Lehman Brothers moved tens of billions of assets off its balance sheet in the last days of a quarter, reported a flattered leverage ratio, and brought the assets straight back, and how to spot the same trick anywhere by comparing period-end figures to averages."
tags: ["forensic-accounting", "lehman-brothers", "repo-105", "window-dressing", "financial-statement-fraud", "leverage", "off-balance-sheet", "repo-market", "sfas-140", "auditing"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 57
depth: "deep-dive"
---

> [!important]
> **TL;DR:** Lehman Brothers took an ordinary trade, booked it in an extraordinary way, and made its balance sheet look about \$50 billion smaller on the four days a year anyone measured it. Almost none of it was ever found to be illegal.
>
> - An ordinary repo is a secured loan, so it makes a balance sheet *bigger*. Lehman deliberately posted 105% collateral, which pushed the trade outside an accounting safe harbour and let it be recorded as a *sale* instead, after which the cash was used to repay debt so both sides of the balance sheet shrank.
> - The rule was counterintuitive: the more collateral you posted, the less the standard believed you controlled the asset. Below 98% cash against the securities delivered, you were deemed to have surrendered control.
> - No United States law firm would give the true-sale opinion the rule required, so the trades were routed through Lehman's London entity under an opinion on English law.
> - At the quarter ended 31 May 2008 the firm moved \$50.38 billion off its balance sheet and published a net leverage ratio of 12.1x rather than the 13.9x it was actually running.
> - The court-appointed Examiner found *colorable claims* of breach of fiduciary duty against four named executives and professional malpractice against Ernst & Young. No court ever ruled on them, no executive was criminally charged, the SEC closed its investigation without suing, and Ernst & Young settled with New York for \$10 million while admitting nothing.
> - The number to remember: the amount removed was roughly **1.8 times the firm's entire tangible equity capital**. The skill to take away: compare period-end figures with period averages.

Imagine you had to stand on a scale in public once every three months. You know the date years in advance. The reading is printed in a newspaper and analysts build models from it. Nobody weighs you on any other day of the year.

You would not need to lie about your weight. You would only need to be careful about Tuesday.

That is roughly the position of every listed company on earth, and Lehman Brothers produced the most instructive response to it on record. In the days before each quarter closed, the firm moved tens of billions of dollars of securities off its balance sheet, reported a flattering leverage ratio, and took the securities back a few days later. The transactions were real. The accounting entries were arguably correct. The disclosure was the problem, and to this day almost nobody has been held liable for it.

![Two-panel chart showing Repo 105 and 108 usage rising across four Lehman quarter-ends from 36.4bn to 50.38bn, and below it the reported net leverage line running well beneath the leverage excluding Repo 105, with gaps of 1.7, 1.9 and 1.8 turns](/imgs/blogs/lehman-repo-105-and-window-dressing-the-balance-sheet-7.webp)

The chart above is the whole article in one picture, and it is worth pausing on. The bars are how much Lehman took off its balance sheet at each quarter-end, rising quarter after quarter as the firm's position worsened. The two lines below are the leverage ratio it published and the leverage ratio it was actually running. Every number in it comes from the report of the court-appointed bankruptcy examiner, and we will build up to all of them from zero.

What follows is in three parts. First, the mechanics: what a repo is, why the ordinary version makes a balance sheet larger, and the specific accounting rule that let the same trade be recorded as a sale. Second, the record: what Lehman actually did, what its own people called it, and what the law did about it, which is remarkably little. Third, and most useful to you, the general skill, because period-end window dressing is a whole family of techniques that is mostly legal, entirely deceptive, and detectable from public filings if you know which two numbers to put next to each other.

This post is about the accounting. For the story of the collapse itself, the funding run, the failed rescue and the weekend that ended it, see [Lehman Brothers and the 2008 global financial crisis](/blog/trading/finance/lehman-brothers-2008-financial-crisis).

## Foundations: the building blocks

Nothing in this story requires prior finance knowledge. It requires four ideas, and this section builds all four from zero. If you already know what a repo is and how leverage is computed, skim to the last sub-section, because the definition of *sale versus financing* is where the whole article turns.

### A balance sheet is two lists that are forced to agree

A **balance sheet** is a photograph of what a company owns and owes on one specific day. It has exactly three parts.

- **Assets**: everything the company owns or is owed. Cash, buildings, inventory, loans it has made, securities it holds.
- **Liabilities**: everything the company owes to somebody else. Bank loans, bonds it has issued, money borrowed overnight, wages not yet paid.
- **Equity**: what would be left for the owners if you sold every asset at its stated value and paid off every liability. It is not a separate pot of money. It is a subtraction.

The three are locked together by one identity that can never be violated:

$$\text{Assets} = \text{Liabilities} + \text{Equity}$$

That is why it is called a *balance* sheet. The two sides balance by construction, because equity is *defined* as the difference. If a company's assets fall in value by \$1 and its liabilities do not change, its equity falls by exactly \$1.

The critical word above is **day**. A balance sheet is not a movie of the year. It is a still frame of one date, usually the last day of a quarter or a financial year. Everything in this article follows from that single fact. A photograph can be posed.

### Leverage is the number that decides whether you survive

**Leverage** measures how much a company owns relative to how much of it the owners actually paid for. The simplest version:

$$\text{Gross leverage} = \frac{\text{Total assets}}{\text{Equity}}$$

Think about a house. You buy a \$500,000 house with \$50,000 of your own savings and a \$450,000 mortgage. Your assets are \$500,000, your liabilities are \$450,000, and your equity is \$50,000. Your leverage is \$500,000 divided by \$50,000, which is 10 times, usually written **10x**.

Now watch what leverage does to risk. If the house rises 10% to \$550,000, your equity goes from \$50,000 to \$100,000. You made 100% on your money from a 10% move. That is the appeal. But if the house falls 10% to \$450,000, your equity goes to zero. You still owe \$450,000 and you own something worth \$450,000. One more percent down and you are insolvent: your assets are worth less than your debts.

That is the whole reason anyone cares about a financial firm's leverage. At 10x, a 10% fall in asset values wipes out the owners. At 25x, a 4% fall does it. At 30x, 3.3% does it. Investment banks in 2007 ran in the high twenties and low thirties, which means a small, ordinary move in the value of what they held could erase everything the shareholders owned.

Two refinements you need, because Lehman used both.

- **Gross leverage** is the simple version above: total assets divided by total equity.
- **Net leverage** is a friendlier number that firms computed themselves. It strips out of the numerator certain assets the firm argued carried little risk (most importantly securities bought under resale agreements, which are effectively secured loans the firm has *made*, plus identifiable intangible assets and goodwill), and it divides by *tangible equity capital* rather than plain equity. Because the numerator falls a lot and the denominator changes less, net leverage always prints far lower than gross leverage.

Net leverage was not a regulatory ratio. It was a management-defined measure that the firm chose to publish and that analysts had learned to quote. That matters: a number a company defines itself, and that the market treats as a headline, is a number the company has both the ability and the motive to manage.

> A leverage ratio is a promise about how much room you have to be wrong. Managing the ratio without changing the risk is a way of lying about the room.

### A repo is a pawn shop for bonds

A **repurchase agreement**, universally shortened to **repo**, is the plumbing of the bond market. Stripped of jargon it works like this.

You own \$10.2 million of government bonds. You need cash today. Instead of selling the bonds, you hand them to a counterparty (a *counterparty* is simply the other side of a trade: another bank, a money market fund, a central bank) and receive \$10.0 million of cash. You simultaneously agree to buy those same bonds back a few days later for the \$10.0 million plus a small fee. The fee, expressed as an annual rate, is the **repo rate**.

Three terms fall out of that description, and you need all three.

- **Collateral**: the securities you handed over. They protect the lender. If you fail to buy them back, the lender keeps them and sells them.
- **Haircut**: the gap between the value of the collateral and the cash you received. Here you gave \$10.2 million of bonds and got \$10.0 million of cash, so the haircut is roughly 2%. The lender wants that cushion in case the bonds fall in value before it can sell them.
- **Overcollateralisation**: the same gap seen from the other direction. Your collateral is 102% of the cash. This number is about to become the single most important quantity in this article.

Economically, a repo is a **secured loan**. You still bear the gains and losses on the bonds, because you have committed to buy them back at a fixed price. If the bonds rally, you profit. If they fall, you lose. You have borrowed cash and pledged property. Nothing about your ownership has really changed.

The market runs on standard contracts. In the United States the usual document is the Master Repurchase Agreement; in cross-border and European trading it is the **Global Master Repurchase Agreement**, or GMRA. Which document you sign, and which country's law governs it, will turn out to matter enormously.

#### Worked example: an ordinary repo, booked as a financing

Take a simplified dealer. Round numbers, because the point is the mechanism rather than the arithmetic.

**Starting position on 31 March.**

| | Assets | | Liabilities and equity |
|---|---|---|---|
| Government bonds | \$10.2bn | Liabilities | \$96.0bn |
| Other assets | \$89.8bn | Equity | \$4.0bn |
| **Total assets** | **\$100.0bn** | **Total** | **\$100.0bn** |

Gross leverage is \$100.0bn divided by \$4.0bn, which is **25.0x**.

**The trade.** The dealer repos the \$10.2bn of government bonds and receives \$10.0bn of cash. Collateral is 102% of the cash.

**The bookkeeping, as a financing.** Because this is a secured loan, the accounting mirrors the economics. The bonds do not leave. The dealer records:

- Debit Cash \$10.0bn (a new asset arrives)
- Credit Securities sold under agreements to repurchase \$10.0bn (a new liability arrives)

The government bonds stay exactly where they were, at \$10.2bn, because the dealer never gave up the risk of owning them.

**Ending position.**

| | Assets | | Liabilities and equity |
|---|---|---|---|
| Government bonds | \$10.2bn | Liabilities | \$96.0bn |
| Cash | \$10.0bn | Repo payable | \$10.0bn |
| Other assets | \$89.8bn | Equity | \$4.0bn |
| **Total assets** | **\$110.0bn** | **Total** | **\$110.0bn** |

Gross leverage is now \$110.0bn divided by \$4.0bn, which is **27.5x**.

The intuition to carry forward: **an ordinary repo makes a balance sheet bigger, not smaller, and makes reported leverage worse, not better.** Borrowing money adds an asset and a liability at the same time. This is the exact opposite of what a firm trying to look safe would want.

![Two balance sheet stacks side by side showing that an ordinary repo booked as a financing raises total assets from 100.0bn to 110.0bn and leverage from 25.0x to 27.5x](/imgs/blogs/lehman-repo-105-and-window-dressing-the-balance-sheet-2.webp)

### Sale or financing: the only question that matters

Every transfer of a financial asset gets recorded one of two ways, and there is no third option.

- **A financing.** You did not really sell anything, you borrowed against it. The asset stays on your balance sheet, and a new borrowing appears next to it. Both sides get bigger.
- **A sale.** You really sold it. The asset comes off your balance sheet, cash comes on, and any gain or loss goes through your income statement. No new borrowing appears.

Now hold those two side by side and notice the asymmetry. In a financing, your balance sheet grows by the amount you borrowed. In a sale, your balance sheet does not grow at all, and if you use the cash to repay something you already owed, it actually *shrinks*.

Same trade. Same cash. Same risk. Two completely different pictures for anyone reading your accounts.

This is the seam that Repo 105 was built to exploit. Lehman did not invent a new transaction. It found a way to have an ordinary repo, a trade the firm did thousands of times a day, classified as a sale instead of a financing for a few days at a time.

## 1. The 102% cliff: how posting more collateral meant losing control

Here is the counterintuitive heart of the whole affair, and it is worth slowing down for, because almost every casual summary of Repo 105 gets it backwards.

The governing rule in United States accounting at the time was **SFAS 140**, the Financial Accounting Standards Board's standard on transfers of financial assets. Its logic is that a transfer counts as a sale only if the transferor has genuinely **surrendered control** of the asset. If you still control it, you have not really sold it, and you must keep it on your books.

SFAS 140 tested that with three conditions, all of which had to be satisfied for sale treatment.

1. The transferred assets have been **isolated** from the transferor, put beyond the reach of the transferor and its creditors even in bankruptcy. In practice this condition is satisfied by obtaining a **true-sale legal opinion**: a written opinion from a law firm concluding that a court would treat the transfer as a real sale rather than a disguised loan.
2. The transferee has the **right to pledge or exchange** the assets it received. In a normal repo this is satisfied, because the lender can re-use the collateral.
3. The transferor does **not maintain effective control** over the transferred assets, for example through an agreement that both entitles and obliges it to repurchase them before maturity.

Conditions one and two are hard to manipulate. Condition three is where the door was.

### Why more collateral meant less control

An ordinary repo obviously *does* leave the transferor with an agreement to repurchase. So on its face, condition three should always fail and every repo should be a financing. That is indeed the normal outcome, and it is why the enormous majority of repos in the world sit on balance sheets as borrowings.

But the standard did not stop at "is there a repurchase agreement". It asked whether the transferor had retained the *practical ability* to reacquire the assets. The reasoning was sensible enough: if you have promised to buy something back but you have no realistic way to fund the purchase, your promise does not amount to control in any meaningful sense.

To make that testable, the standard set a numeric safe harbour. The transferor was treated as retaining effective control when the cash it received was enough to fund substantially all of the cost of buying replacement securities, and the standard put that at roughly **98% to 102% of the fair value of the securities transferred**.

Read that carefully, because the direction trips people up. The band applies to **the cash you receive, measured against the securities you hand over**. A normal repo sits comfortably inside it. Hand over \$10.2bn of bonds, receive \$10.0bn of cash, and the cash is 98.0% of the collateral. Inside the band. Effective control retained. Financing.

Now hand over more collateral for the same cash.

- Deliver securities worth **105%** of the cash, and the cash you receive is 100 divided by 105, which is **95.2%** of the securities' value. Below 98%. Outside the band.
- Deliver securities worth **108%** of the cash, and the cash is 100 divided by 108, which is **92.6%** of the securities' value. Further outside.

Once you fall outside the band, the rule concluded that you did not receive enough cash to buy replacement securities, therefore you did not retain the practical ability to reacquire them, therefore you did **not** maintain effective control, therefore condition three was satisfied, therefore, if conditions one and two were also met, **the transfer was a sale**.

This is genuinely perverse when you say it in plain English. *The more collateral you post, the less the rule believes you own the asset.* Posting extra collateral makes you economically worse off, not better off. You have handed over more property for the same money. And that self-harm was precisely the qualification.

![Step chart showing accounting treatment against collateral posted, with a safe harbour band from 98 to 102 percent where the trade is a financing, and a cliff at 102 percent above which it is deemed a sale, with Repo 105 and Repo 108 marked](/imgs/blogs/lehman-repo-105-and-window-dressing-the-balance-sheet-4.webp)

Lehman's internal names for the two variants came straight from the thresholds. **Repo 105** meant delivering securities worth 105% of the cash, and was used for fixed income: government bonds and other debt securities. **Repo 108** meant delivering 108%, and was used for equities, which are more volatile and therefore needed a bigger margin to be safely outside the band.

![Decision tree of the three SFAS 140 control tests, showing that the isolation test and the pledge test could not be engineered but the effective control test could be failed on purpose by posting collateral above 102 percent](/imgs/blogs/lehman-repo-105-and-window-dressing-the-balance-sheet-5.webp)

### The cost of the trick, and why that cost was the point

Nothing here is free. Posting 105% collateral instead of 102% means tying up roughly 3% more securities for the same cash. On \$50bn of cash raised, that is over \$1.5bn of extra securities pledged, earning nothing extra. Counterparties also charged more for these trades than for ordinary repos, because they knew what they were being asked to facilitate and they were taking more collateral than the market standard.

So the firm was paying real money for an accounting outcome. That fact is doing a lot of work in this story. A transaction that costs you money and changes nothing about your economics has only one plausible purpose, which is the effect it has on what other people see.

#### Worked example: the same trade, booked as a sale

Take the identical dealer from the previous worked example, and change only the collateral ratio.

**Starting position on 31 March.**

| | Assets | | Liabilities and equity |
|---|---|---|---|
| Government bonds | \$10.5bn | Liabilities | \$96.0bn |
| Other assets | \$89.5bn | Equity | \$4.0bn |
| **Total assets** | **\$100.0bn** | **Total** | **\$100.0bn** |

Gross leverage is **25.0x**, exactly as before.

**Step 1: the trade, booked as a sale.** The dealer delivers \$10.5bn of government bonds and receives \$10.0bn of cash. Collateral is 105% of the cash, so the cash is 95.2% of the collateral, outside the 98% to 102% band. With a true-sale opinion in hand, the transfer qualifies as a sale.

The entries are now completely different:

- Credit Government bonds \$10.5bn (the asset is derecognised, meaning removed from the balance sheet entirely)
- Debit Cash \$10.0bn
- Debit Forward repurchase commitment \$0.5bn (a derivative asset recording the right and obligation to buy the securities back, carried at the value of the over-collateral)

| | Assets | | Liabilities and equity |
|---|---|---|---|
| Cash | \$10.0bn | Liabilities | \$96.0bn |
| Repurchase commitment | \$0.5bn | Equity | \$4.0bn |
| Other assets | \$89.5bn | | |
| **Total assets** | **\$100.0bn** | **Total** | **\$100.0bn** |

Notice that total assets have not moved yet. The bonds left, but cash and a receivable arrived in their place. Leverage is still 25.0x. **No borrowing was recorded**, because in the eyes of the accounts nothing was borrowed.

**Step 2: use the cash to pay down debt.** This is the step that makes the trick work, and it is the step most retellings leave out. The dealer takes the \$10.0bn of cash and repays \$10.0bn of existing short-term borrowings.

- Debit Liabilities \$10.0bn
- Credit Cash \$10.0bn

| | Assets | | Liabilities and equity |
|---|---|---|---|
| Repurchase commitment | \$0.5bn | Liabilities | \$86.0bn |
| Other assets | \$89.5bn | Equity | \$4.0bn |
| **Total assets** | **\$90.0bn** | **Total** | **\$90.0bn** |

Gross leverage is now \$90.0bn divided by \$4.0bn, which is **22.5x**.

**Compare the two worked examples.** The same dealer, the same securities, the same counterparty, the same cash, and economically the same risk in both cases, because in both cases the firm has committed to take the bonds back.

| | Ordinary repo (financing) | Repo 105 (sale) |
|---|---|---|
| Securities delivered | \$10.2bn | \$10.5bn |
| Cash received | \$10.0bn | \$10.0bn |
| Bonds on balance sheet after | \$10.2bn | \$0 |
| New borrowing recorded | \$10.0bn | \$0 |
| Reported total assets | \$110.0bn | \$90.0bn |
| Reported gross leverage | 27.5x | 22.5x |

The reported balance sheet differs by \$20.0bn and reported leverage differs by **5.0 turns**, purely from the classification of a trade whose economics are identical. The firm that used Repo 105 also gave up more collateral and paid a higher fee, so on any economic measure it was slightly *worse* off. It just looked considerably better.

![Three balance sheet stacks showing the same trade booked as a sale, with the bonds derecognised, then the cash used to repay debt, taking total assets from 100.0bn to 90.0bn and leverage from 25.0x to 22.5x](/imgs/blogs/lehman-repo-105-and-window-dressing-the-balance-sheet-3.webp)

The single sentence to keep: **the trick did not reduce risk by a dollar, it reduced the reported measurement of risk by twenty billion.**

## 2. The London detour: why the trade had to leave the country

Clearing the 102% threshold satisfied condition three. It did nothing for condition one, and condition one is the reason this story has a passport.

Condition one required that the transferred assets be isolated, put beyond the reach of the transferor and its creditors even in bankruptcy, and the practical way to establish that was a true-sale legal opinion. The problem is that under United States law a repurchase agreement with a fixed buy-back price looks like precisely what it economically is: a secured loan. According to the Examiner's Report in the Lehman bankruptcy, Lehman was unable to obtain an opinion from a United States law firm that its Repo 105 transactions would be treated as true sales under US law.

So the transaction emigrated.

The trades were executed by **Lehman Brothers International (Europe)**, universally abbreviated **LBIE**, the group's London-based broker-dealer, under an English-law repurchase agreement. The law firm **Linklaters** provided an opinion that, under English law, the transfer constituted a true sale.

![Six-step pipeline showing US inventory moving to Lehman's London broker-dealer, an English-law repo with a Linklaters true-sale opinion, sale treatment in London, and the result consolidating back into the group's US GAAP balance sheet](/imgs/blogs/lehman-repo-105-and-window-dressing-the-balance-sheet-6.webp)

Two features of that arrangement deserve to be stated plainly, because they are what make it work.

**The opinion answered a different question than the one the accounts turned on.** A true-sale opinion under English law says something about how an English court would characterise a transfer. It is not an opinion on United States generally accepted accounting principles, and it did not purport to be. Yet the accounting condition it was used to satisfy sat inside a US accounting standard, applied to a US-listed parent company's consolidated financial statements.

**The result travelled home.** LBIE was a subsidiary. Its numbers consolidated into the accounts of Lehman Brothers Holdings Inc, the New York-listed parent, which reported under US GAAP. Sale treatment obtained in London therefore reduced the total assets that appeared in the filings American investors and rating agencies read. The trade left the country; the benefit did not.

Operationally this required moving inventory. Securities held by Lehman's US broker-dealer had to be transferred to LBIE so that the London entity could be the party to the repo. Consider what that means: a firm shifting securities across an ocean and an entity boundary, at some cost and operational effort, days before a reporting date, in order that a rule about control would read differently.

### Was this a loophole?

It is worth being precise, because the honest answer is more interesting than a simple yes.

Condition one asks a *legal* question: would a court treat this as a sale? Legal questions have answers that depend on which country's law you ask. Nothing in the accounting standard said the opinion had to be an opinion on US law, and multinational financial groups genuinely do transact across jurisdictions for ordinary commercial reasons. In that narrow sense, obtaining an English-law opinion for a trade executed by an English entity under an English-law contract is not obviously improper.

What the standard did not contemplate was a firm treating the choice of jurisdiction as a *variable to be solved for*: identifying the answer it wanted, finding the legal system that would give it, and routing the assets there for a few days at a time around reporting dates. The individual steps were each defensible. The purpose that assembled them was the problem.

This is the pattern that recurs across this whole series. Serious statement manipulation is rarely a single forged number. It is usually a sequence of individually permissible choices arranged to produce an impression that none of them would produce alone. The same architecture appears in [off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities), and in [Enron's re-read](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market), where the legal form of an entity was engineered to control what consolidated. The difference here is *time*: Enron's structures were built to keep obligations off the balance sheet permanently, whereas Repo 105 kept assets off it for a matter of days.

## 3. What Lehman actually did, quarter by quarter

Everything to this point has been mechanism. Here is the record, and the record is unusually good, because after Lehman Brothers Holdings Inc filed for bankruptcy in September 2008 the court appointed an examiner, Anton R. Valukas of Jenner & Block, whose report ran to nine volumes and was filed on 11 March 2010 in Case No. 08-13555 (JMP) before the United States Bankruptcy Court for the Southern District of New York. Volume 3 covers Repo 105. All page references below are to that volume.

### The programme

Lehman's Repo 105 programme **began in 2001**. The equities version, Repo 108, did not start until **May 2006** (Vol. 3 at 762). The firm's own internal description of what it was doing is admirably clear:

> "Repo 105 and Repo 108 transactions refer to repos with a counterparty in which we sell securities valued at a minimum of 105% (for fixed income securities) or 108% (for equity securities) of the cash received." (Vol. 3 at 776)

Note the phrasing: *we sell*. Internally there was no ambiguity about the intended accounting.

There was an internal ceiling on the programme, and its provenance matters. The Repo 105 limit was **\$20 billion**, and in January 2008 a request went round to expand it to **\$23 billion**. The Examiner records that this limit "was a management decision, was not based on accounting rules" (Vol. 3 at 762). A cap set by management rather than by the standard tells you the constraint being managed was appetite, not eligibility.

### The numbers

Lehman's fiscal quarters ended in late November, February, May and August. Here is what the firm removed from its balance sheet at each quarter-end, and what its published leverage looked like as a result (Vol. 3 at 742, 748).

| Quarter end | Repo 105/108 used | Reported net leverage | Net leverage without Repo 105 | Difference |
|---|---|---|---|---|
| 30 August 2007 (Q3 2007) | \$36.4bn | not tabulated by the Examiner | not tabulated | |
| 30 November 2007 (Q4 2007) | \$38.6bn | 16.1x | 17.8x | 1.7 |
| 29 February 2008 (Q1 2008) | \$49.1bn | 15.4x | 17.3x | 1.9 |
| 31 May 2008 (Q2 2008) | \$50.38bn | 12.1x | 13.9x | 1.8 |

The reported ratios come from Lehman's own filings: the 2007 Form 10-K at p. 64, the Form 10-Q filed 9 April 2008 at p. 72, and the Form 10-Q filed 10 July 2008 at p. 89 (Vol. 3 at 748).

Two details worth having. The precise firm-wide usage figures were **\$38.634 billion** on 30 November 2007, **\$49.102 billion** on 29 February 2008, and **\$50.383 billion** on 30 May 2008; 31 May 2008 fell on a Saturday, which is why the last measurement is dated the Friday (Vol. 3 at 742). And the trend was upward: \$36.4bn, \$38.6bn, \$49.1bn, \$50.38bn across four consecutive quarter-ends. Whatever the programme was for, the firm needed more of it as its position deteriorated.

### What "net leverage" meant, precisely

Because the headline number was Lehman's own construction, you need its definition to read the table. In its Forms 10-K and 10-Q, Lehman defined the **net leverage ratio** as *net assets divided by tangible equity capital*, where (Vol. 3 at 734):

- **Net assets** = total assets *excluding* (1) cash and securities segregated and on deposit for regulatory and other purposes, (2) securities received as collateral, (3) securities purchased under agreements to resell, (4) securities borrowed, and (5) identifiable intangible assets and goodwill.
- **Tangible equity capital** = stockholders' equity *plus* junior subordinated notes, *minus* identifiable intangible assets and goodwill.

Lehman's plainer "leverage ratio" was simply total assets divided by stockholders' equity, and it printed far higher.

The exclusions matter for our purposes. Securities purchased under resale agreements come out of the numerator, which is why an ordinary repo done in the other direction does not help this ratio. What *does* help is removing ordinary trading inventory, which is exactly what Repo 105 removed.

#### Worked example: the Q2 2008 leverage ratio, reported and adjusted

Work directly with the Examiner's two figures for the quarter ended 31 May 2008: a reported net leverage ratio of **12.1x**, and **13.9x** without the Repo 105 benefit, on **\$50.38bn** of transactions.

**Step 1. Size one turn of leverage.** The two ratios differ by 13.9 minus 12.1, which is **1.8 turns**. That entire difference was produced by removing \$50.38bn from the numerator. So one turn of net leverage corresponded to:

\$50.38bn divided by 1.8 = **\$27.99bn**, call it **\$28.0bn**

That \$28.0bn is the denominator: Lehman's tangible equity capital. (Because the published ratios are rounded to one decimal place, this is an implied figure with real error bars, not a reported one. Treat it as approximately \$28bn.)

**Step 2. Recover the reported numerator.** If reported net leverage was 12.1x on an equity base of \$28.0bn, then reported net assets were:

12.1 × \$28.0bn = **\$338.8bn**

**Step 3. Add back what was removed.**

\$338.8bn + \$50.38bn = **\$389.2bn**

**Step 4. Recompute.**

\$389.2bn divided by \$28.0bn = **13.9x**, which reconciles to the Examiner's figure.

So the picture is this. On 31 May 2008 Lehman told the market it was running about \$339bn of net assets against roughly \$28bn of tangible equity capital. On an ordinary day that same week it was running about \$389bn against the same \$28bn. **The firm reported holding \$50.38bn less than it held. That is roughly 15% of the net balance sheet it published, roughly 13% of the one it was actually running, and roughly 1.8 times its entire tangible equity capital.**

The intuition: **when the quantity you remove is nearly twice your equity, the ratio you publish is no longer a measurement of your firm.**

#### Worked example: the round trip, and how quickly it reversed

The most useful single sentence in the whole Examiner's Report for our purposes is not from Lehman at all. It is from the handwritten notes of an Ernst & Young auditor, Hillary Hansen, taken on 12 June 2008 during an interview with a Lehman whistleblower. Her note records that Lehman's rates and liquid markets businesses used Repo 105/108 to "reduce[] assets by 50B [by] moving off B/S [balance sheet] in Europe & back in 5 days later" (Vol. 3 at 957).

*Back in five days later.* Trace the calendar for Q2 2008, using the reported ratios and the implied \$28.0bn equity base from the previous example.

| Date | Net assets | Net leverage | Who saw it |
|---|---|---|---|
| Late May, before the trades | ~\$389bn | ~13.9x | Nobody outside the firm |
| 30 May 2008 (Friday) | ~\$339bn | **12.1x** | **The market, in the Form 10-Q** |
| Roughly 4 June 2008 | ~\$389bn | ~13.9x | Nobody outside the firm |

The economic position on 4 June was the same as on 28 May. The only day that was different was the day that was published.

Work the cost of that difference. To take \$50.38bn of assets off the balance sheet, Lehman had to deliver securities worth at least 105% of the cash for the fixed-income portion, so it pledged upward of \$52.9bn of inventory to raise \$50.38bn. The extra collateral, on the order of \$2.5bn, earned the firm nothing. It was the fee for the appearance.

![Timeline around the 30 May 2008 quarter end showing net assets of about 389bn and leverage of about 13.9x before the trades, 339bn and 12.1x on the reported date, and a return to about 389bn and 13.9x roughly five days later](/imgs/blogs/lehman-repo-105-and-window-dressing-the-balance-sheet-8.webp)

The intuition: **an exposure that reverses within a week was never reduced, only rescheduled around the camera.**

### What the people involved called it

The Examiner's Report reproduces internal communications that make the purpose difficult to dispute. In April 2008, asked whether he was familiar with the use of Repo 105 to reduce net balance sheet, Bart McDade, then Lehman's head of equities and shortly to become its president and chief operating officer, replied:

> "I am very aware . . . it is another drug we r on." (Vol. 3 at 742 and 815)

A July 2008 exchange between two Lehman employees is more explicit still:

> Vallecillo: "So what's up with repo 105? Why are we doing less next quarter end?"
>
> McGarvey: "It's basically window-dressing. We are calling repos true sales based on legal technicalities."
>
> Vallecillo: "I see . . . so it's legally do-able but doesn't look good when we actually do it?" (Vol. 3 at 860)

That last line is the entire article in a sentence, written by someone who worked there.

There is also a striking piece of internal evidence that Lehman knew it was an outlier. In a May 2008 email, a senior external-reporting executive noted that Citigroup and JPMorgan "likely do not do Repo 105 and Repo 108 which are UK-based specific transactions on opinions received by LEH from Linklaters", adding that this "would be another reason why LEH's daily balance sheet is larger intra-month then at month-end" (Vol. 3 at 740). The firm had noticed, in writing, both that competitors did not do this and that it produced precisely the intra-month-versus-month-end signature this article has been teaching you to look for.

### The materiality point, which is devastating

Lehman's own auditor had a working definition of what counted as material to the balance sheet. Ernst & Young's walkthrough papers for the balance-sheet close process stated:

> "Materiality is usually defined as any item individually, or in the aggregate, that moves net leverage by 0.1 or more (typically \$1.8 billion)." (Vol. 3 at 747)

Set that against the table above. Repo 105 moved net leverage by 1.7, 1.9 and 1.8 turns. As the Examiner put it in a single sentence: "Repo 105 moved net leverage not by tenths, but by whole points" (Vol. 3 at 747).

By the auditor's own working threshold, the programme was material by a factor of roughly seventeen to nineteen. Lehman never disclosed it. Not the programme, not the volumes, not the effect on the ratio.

### What was said out loud instead

On the Q1 2008 earnings call, Lehman's then chief financial officer Erin Callan told analysts that the firm "did, very deliberately, take leverage down for the quarter. We ended with a net leverage ratio of 15.4 times down from 16.1 at year end" (Vol. 3 at 846).

Read that against the table, carefully, because the honest reading is more interesting than the cartoon version.

The 15.4x was arrived at with \$49.1bn of Repo 105 in place. Without it the figure was 17.3x. So three things are simultaneously true:

1. **Underlying leverage genuinely did fall.** Excluding Repo 105, the ratio went from 17.8x at year end to 17.3x, a real reduction of 0.5 turns. The firm was deleveraging.
2. **The reduction advertised was larger than the reduction achieved.** The market was shown a fall of 0.7 turns, from 16.1x to 15.4x. The underlying fall was 0.5 turns.
3. **The level was wrong by far more than the change.** This is the part that matters. The true ratio, 17.3x, was *higher* than the 16.1x figure the firm had reported for the previous quarter and was now inviting analysts to compare against. A reader tracking the published series saw a firm moving from 16.1 to 15.4. The firm was actually sitting at 17.3.

So the criticism is not that Callan announced a decline that never happened. It is that both endpoints of the comparison were understated, by different amounts, in a series the market was reading as a trend. The word "deliberately" was accurate about the intent to reduce leverage. It was the arithmetic underneath it that had been rearranged.

## 4. The whistleblower, the auditor, and what the law actually did

### Matthew Lee

On **16 May 2008**, Matthew Lee, then a senior vice president in Lehman's finance division responsible for the firm's global balance sheet and legal entity accounting, sent a letter to certain members of Lehman's senior management identifying possible violations of the firm's ethics code relating to accounting and balance-sheet issues (Vol. 3 at 956).

Be precise about what that letter was and was not. It raised balance-sheet and accounting concerns; it was not a memo headed "Repo 105". The Repo 105 detail came out in the follow-up. On **12 June 2008**, two Ernst & Young auditors, William Schlich and Hillary Hansen, interviewed Lee privately (Vol. 3 at 957). Hansen's handwritten notes from that meeting are the ones quoted earlier: the rates and liquid markets businesses were using Repo 105/108 to reduce assets by \$50bn by moving them off the balance sheet in Europe and taking them back five days later.

Now put that against the calendar. Lehman filed the Form 10-Q containing the 12.1x net leverage figure on **10 July 2008** (Vol. 3 at 748). The auditor had the mechanism, described by a firm insider and written down in its own working papers, roughly four weeks before that filing went out.

The Examiner records that Ernst & Young had no further conversations with Lee about Repo 105 (Vol. 3 at 958).

### The auditor's position

Ernst & Young's answer, then and since, has been essentially jurisdictional: the transactions complied with the accounting standard. When the Examiner invited the firm to explain why Repo 105 transactions were proper and did not result in materially misleading financial statements, Schlich replied that the transactions were proper if they complied with SFAS 140 (Vol. 3 at 991).

The Examiner's response to that framing is the intellectual core of the whole report, and it is worth stating carefully because it is the part most often garbled. He did **not** conclude that Repo 105 violated SFAS 140. He concluded that the question was beside the point. One of his section headings puts it flatly: whether Lehman's Repo 105 transactions technically complied with SFAS 140 does not affect whether a colorable claim exists (Vol. 3 at 964). In the body:

> "the answer to that question does not impact whether there is sufficient evidence to support a colorable claim regarding Lehman's failure to disclose its Repo 105 practice and whether that failure rendered the firm's periodic reports materially misleading." (Vol. 3 at 734)

And on the specific defect:

> "Lehman's description of its net leverage was misleading because it omitted disclosing that the ratio was reduced by means of temporary, accounting-motivated transactions." (Vol. 3 at 750)

The Examiner also noted that Ernst & Young did not evaluate the possibility that Repo 105 transactions were accounting-motivated transactions lacking a business purpose (Vol. 3 at 962).

**Recognition was never the charge. Disclosure was.** A firm can apply a standard correctly and still publish a set of accounts that leaves a reader with a false impression, and it is the second thing that securities law reaches.

### What "colorable claim" means, and what it does not

This is where precision matters most, and where popular retellings of Repo 105 routinely overreach.

An examiner in a bankruptcy is not a judge and does not decide liability. Valukas was appointed to investigate and to identify claims the estate might have. A **colorable claim** is a claim with enough evidentiary support to be worth pursuing: it could plausibly survive a motion to dismiss. It is a screening threshold, not a verdict. Finding a colorable claim says "a trier of fact could find for the plaintiff on this record". It does not say "the defendant did it".

With that established, here is exactly what the Examiner concluded (Vol. 3 at 750):

> "The Examiner concludes that colorable claims of breach of fiduciary duty exist against Richard Fuld, Chris O'Meara, Erin Callan, and Ian Lowitt, and that a colorable claim of professional malpractice exists against Ernst & Young."

He also concluded that, with the exception of Fuld, there was not sufficient evidence to support colorable claims of breach of fiduciary duty against Lehman's directors arising from Repo 105 (Vol. 3 at 991).

So: **no finding of fraud, by the Examiner or by anyone else.** The causes of action identified were breach of fiduciary duty and professional negligence, which are civil and which turn on carelessness and duty rather than on intent to deceive. And no court ever adjudicated any of it.

### What the law actually did

Here is the full public record of consequences arising from Repo 105.

**The New York Attorney General sued the auditor, not the bank.** On **21 December 2010**, Attorney General Andrew Cuomo filed suit against Ernst & Young in New York Supreme Court under the **Martin Act**, New York's unusually broad securities-fraud statute. The complaint alleged that Repo 105 transactions "served no legitimate business purpose" and that Ernst & Young had approved the accounting and issued unqualified opinions while knowing Lehman was not disclosing the practice. The state sought the return of the entire fee stream Ernst & Young had collected from Lehman between 2001 and 2008, a sum exceeding **\$150 million**, plus investor damages and equitable relief.

**It settled for a fraction, with no admission.** On **15 April 2015**, Attorney General Eric Schneiderman announced a settlement of **\$10 million** under the Martin Act and Executive Law section 63(12). Most of the money went to Lehman investors, with the remainder reimbursing New York State for the costs of its investigation and litigation. Ernst & Young admitted no wrongdoing. The firm's own statement was: "After many years of costly litigation, we are pleased to put this matter behind us, with no findings of wrongdoing by EY or any of its professionals" (Forbes, 15 April 2015).

Set the two numbers beside each other. The state asked for more than \$150 million and received \$10 million, roughly 6.7 cents on the dollar of the fees alone, nearly seven years after the collapse and more than four years after filing.

**The SEC brought nothing.** According to contemporaneous reporting of the settlement, the Securities and Exchange Commission closed its investigation in 2012 without bringing suit (Forbes, 15 April 2015).

**Nobody was charged with a crime.** No Lehman executive was criminally prosecuted over Repo 105. Federal prosecutors had opened investigations into the collapse in 2008, and no charges relating to the Repo 105 programme resulted from them.

### The uncomfortable conclusion

Assemble that.

A firm removed up to \$50.38 billion from its published balance sheet at quarter-end, on four consecutive occasions, using a technique its own employees called window dressing based on legal technicalities, one it described internally as a drug, which required routing assets to another country to obtain a legal opinion nobody in its own would give, which cost it real money in extra collateral and fees, which moved a headline ratio by nearly twenty times its auditor's own materiality threshold, and which it never disclosed. Its regulator brought no action. No executive was charged. The auditor paid \$10 million without admitting anything.

That is not a story about a rule being broken. It is a story about what the rules did not require, and it is the reason this article spends more space on how to detect window dressing yourself than on what happened to the people who did it. The enforcement system did not protect the readers of Lehman's accounts. The only protection available was the ability to read them sceptically.

> The lesson of Repo 105 is not that fraud gets punished. It is that the most damaging things in a set of accounts are often the things that were technically allowed.

## 5. Window dressing is a family, not an incident

If you take only one idea from this article, take this one: Repo 105 was an unusually elaborate instance of an entirely ordinary practice. **Period-end window dressing** is the general name for any action taken to make a point-in-time measurement look better than the period it summarises, and it turns up wherever four ingredients are present.

1. **A metric measured at a single instant** rather than over a period. A balance sheet date. A holdings report date. A regulatory reporting date.
2. **A firm that knows the date in advance.** Reporting dates are published years ahead. Nobody is surprised by 31 December.
3. **An action that is cheap and reversible.** If reducing the number required permanently selling a business, nobody would do it for cosmetic reasons. If it requires not rolling over some overnight funding for three days, many people will.
4. **A disclosure regime that reports the point and not the path.** The reader receives the snapshot. The firm alone sees the film.

Where all four hold, expect the behaviour. Where any one is missing, it tends to disappear. That is a genuinely useful predictive rule, and it is why the regulatory response to this whole family has consistently been to attack ingredient four by requiring averages.

### The four main branches

**Banks and dealers: the balance sheet itself.** The measured quantities are total assets, leverage, and liquidity ratios. The levers are letting short-term funding roll off without replacing it, shrinking the repo book, reducing market-making inventory, and netting positions that are separately reported on other days. Repo 105 sits in this branch, as its most engineered example.

**Funds: the marks and the holdings.** Two distinct tricks live here. **Portfolio pumping**, sometimes called *marking the close*, means buying more of a position you already hold late in the last session of a reporting period, so that the closing price used to value your entire holding is lifted. Because the fund's whole position is revalued at that closing price, a modest amount of buying can move a much larger reported figure. Separately, **holdings window dressing** means selling embarrassing positions and buying respectable ones shortly before the date on which holdings must be disclosed, so the published portfolio flatters the manager's judgment. The first inflates reported performance. The second manages reputation.

**Operating companies: working capital.** The measured quantities are cash, receivables, payables, inventory, and the ratios computed from them. The levers are delaying payments to suppliers so cash sits with you on the last day, pushing hard on collections in the final week, selling receivables for cash just before the date, and timing inventory purchases. None of this is fictional: the cash really is there on the day. It simply was not there on the other days, and will not be there next week. Our series covers the diagnostic in [the cash conversion cycle](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) and the ratio mechanics in [forensic ratios](/blog/trading/forensic-accounting/forensic-ratios-dso-dio-dpo-and-margin-anomalies).

**Anyone: the cash balance.** The crudest member of the family. Draw down a credit line, or borrow from a related party, shortly before the reporting date so a large cash figure appears, then repay it immediately afterwards. The balance sheet shows the cash. It may also show the borrowing, which is why this version is easier to catch than the others: check whether cash and short-term debt rose together.

### The family compared

| Branch | What is measured | The lever | What it improves | The tell |
|---|---|---|---|---|
| Banks and dealers | Total assets, leverage, liquidity ratios | Let funding roll off, shrink the repo book, reclassify transfers as sales | Reported leverage and apparent capacity to absorb loss | Period-end borrowings far below the period average |
| Funds, marks | Net asset value, reported return | Buy into the close on the last session | Reported performance and therefore fees and rankings | Abnormal returns on the last day and a reversal on the first day of the next period |
| Funds, holdings | The published portfolio | Sell the embarrassing, buy the respectable | Perceived skill | Turnover concentrated immediately before disclosure dates |
| Operating companies | Cash, payables, receivables, working-capital ratios | Delay supplier payments, accelerate collections, sell receivables | Cash balance, days payable, apparent liquidity | Cash spikes at period end and falls immediately after; payables balloon |
| Anyone | Cash balance | Borrow just before the date, repay just after | Apparent liquidity and solvency | Cash and short-term debt rise together, both reverse |

### Why most of it is legal, and why that is the point

Here is the uncomfortable part. Almost nothing in the table above is inherently unlawful.

A firm is entitled to decide how much overnight funding to take on any given night. A fund manager is entitled to buy a security on the last day of the quarter. A company is entitled to pay a supplier on the 32nd day rather than the 28th. None of these acts is a misstatement, because none of them is a statement at all. They are transactions, and the accounts report them accurately.

The legal exposure attaches somewhere else: to **disclosure**. When a firm publishes a ratio as a description of its financial condition, and that ratio was produced by a temporary transaction that reversed days later, and the firm does not tell you that, the accounts have become misleading even though every individual entry is correct. Securities law in most jurisdictions addresses this through general anti-fraud provisions and through requirements that management's discussion of results not omit information necessary to make what is stated not misleading.

That is a much harder case to bring than "they wrote down a false number", and the difficulty is not an accident of drafting. It reflects a real tension: regulators cannot forbid firms from managing their businesses, and any rule that says "you may not reduce your borrowing before a reporting date" is absurd on its face. So the law reaches the deception rather than the transaction, which means the outcome depends on what was said, to whom, and with what knowledge.

> Window dressing is rarely a lie about what happened. It is a truthful account of an unrepresentative day, presented as a description of the year.

### The structural fix: measure the path, not the point

Because the practice depends on ingredient four, the durable remedy is to stop reporting only the point.

That is exactly the direction post-crisis regulation moved. The general approach across jurisdictions has been to require that certain prudential measures be computed from averages of daily or monthly figures over the reporting period rather than from the balance on the final day, and to require disclosure of maximum and average amounts of short-term borrowings alongside the period-end amount. Where a firm must report the average, shrinking on one day accomplishes almost nothing, because it moves the average by roughly one ninetieth of the reduction.

The lesson generalises well beyond finance. Any incentive attached to a measurement taken on a known date will produce behaviour concentrated on that date. If you want to know how a system behaves, measure it at times it cannot predict, or measure it continuously and report the distribution. This is a management principle as much as an accounting one.

## 6. How to spot period-end window dressing from the outside

You are not going to get an examiner's subpoena power. You have public filings. That is still enough, because window dressing has an unavoidable signature: it makes the reporting date look different from every other date. Any disclosure that reveals a non-reporting-date number is therefore a test.

Here is the procedure, in the order you should run it.

### Step 1: find any average, maximum, or intra-period figure the company already discloses

This is the highest-value habit in the whole article, and it costs about two minutes per filing.

Financial companies routinely publish an **average balance sheet** in the management discussion section: average interest-earning assets, average borrowings, average deposits, computed over the quarter rather than on the last day. Many disclose the **maximum** amount of short-term borrowings outstanding during the period. Banks disclose average daily balances for regulatory ratios. Funds disclose average net assets for fee calculations.

Every one of those is a window into the other 89 days of the quarter. Put the period-end number and the average number next to each other and compute the ratio.

$$\text{Window-dressing ratio} = \frac{\text{Average balance over the period}}{\text{Balance on the reporting date}}$$

A ratio near 1.0 means the reporting date looked like an ordinary day. A ratio meaningfully above 1.0 means the firm carried more of the item on ordinary days than it showed you on the day it was measured.

#### Worked example: reading the ratio across four quarters

An illustrative firm discloses both its period-end short-term borrowings and its average short-term borrowings for the quarter.

| Quarter | Average during the quarter | Reported at period end | Ratio | Read |
|---|---|---|---|---|
| Q1 | \$60bn | \$58bn | 1.03 | Ordinary |
| Q2 | \$60bn | \$42bn | 1.43 | Something happens at quarter end |
| Q3 | \$62bn | \$43bn | 1.44 | It happened again |
| Q4 | \$61bn | \$40bn | 1.53 | And it is getting bigger |

Work the second row. Average borrowings of \$60bn against a period-end figure of \$42bn is a gap of \$18bn, or 30% of the average. For that to be innocent, the firm would need a reason why its funding need genuinely collapsed by 30% in the last days of the quarter and rebuilt immediately after. Such reasons exist: a large asset sale that settled, a seasonal business, a deliberate one-off deleveraging. So do not stop at the ratio. Ask the firm's own disclosures whether anything real happened.

What makes the table above damning is not any single row. It is that the pattern repeats every quarter and widens. Real events do not queue up politely on the last day of March, June, September and December.

The intuition: **one gap is a fact, a repeating gap on the same date each quarter is a policy.**

![Grouped column chart for an illustrative firm comparing quarterly average short-term borrowings against the period-end reported figure across four quarters, with the ratio widening from 1.03 to 1.53](/imgs/blogs/lehman-repo-105-and-window-dressing-the-balance-sheet-10.webp)

### Step 2: compare the balance sheet to the cash flow statement

The cash flow statement covers the whole period, not one day. That makes it much harder to pose. If a firm's period-end borrowings barely moved year on year but its cash flow statement shows enormous gross proceeds from, and repayments of, short-term borrowings, the firm has been cycling far more debt than the balance sheet reveals.

Look specifically at the financing section, where borrowings are often shown gross: proceeds from issuance of short-term debt on one line and repayments on the next. Large gross flows against a small net change is the fingerprint of activity that starts and ends inside the period.

Our series covers this reading in more depth in [reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

### Step 3: read the accounting-policy footnote for transfers of financial assets

Companies must describe the accounting policies they apply. A firm that treats some repos as sales has to say so somewhere, even if it says so in language designed not to be noticed. You are looking for a policy note that describes when transfers of financial assets are accounted for as sales rather than as secured financings, and specifically for any reference to legal opinions, to foreign subsidiaries, or to over-collateralisation thresholds.

The general lesson about where such things live is in [the footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried).

### Step 4: check whether the ratio moves when the incentive moves

The strongest evidence is behavioural. If a firm's period-end-to-average gap widens exactly when it has the most reason to look strong, that is not coincidence. The reasons to look strong are usually visible: a credit rating under review, a capital raise being marketed, a covenant threshold approaching, a bonus period ending, a regulatory ratio being reported for the first time.

Pair the ratio series with a timeline of the firm's pressures. When they line up, you have something worth writing down.

### Step 5: ask what the trick would cost, and whether the firm is paying it

Every window-dressing technique has a price. Over-collateralised repos cost extra collateral and a wider fee. Buying securities at quarter end to pump a fund's marks costs commissions and market impact. Delaying supplier payments costs goodwill and sometimes discounts. Parking cash to show a balance costs interest.

So a useful final question is: **is this firm paying money for an appearance?** Expenses that cannot be explained by revenue or by risk, but can be explained by optics, are among the most honest signals a set of accounts will ever give you. The technique in [the accruals ratio](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly) rests on a related idea: the gap between reported performance and cash performance is where intent shows up.

## Common misconceptions

**"Repo 105 hid losses."** It did not. Repo 105 did not touch the income statement in any meaningful way and did not conceal a single dollar of loss. It changed the *size* of the balance sheet on four days a year, and therefore the leverage ratio computed from it. Losses at Lehman were reported, painfully and publicly. What was managed was the appearance of the firm's capacity to absorb them. This is a useful distinction generally: statement manipulation splits into schemes that change reported profit and schemes that change reported financial position, and they leave different fingerprints.

**"The assets were sold, so the firm was less risky for those few days."** No. The firm was contractually committed to buy the securities back, at a fixed price, within days. It retained every dollar of the gain or loss on them in the interim. The economic exposure never went anywhere. Only the recording of it did.

**"An ordinary repo already hides debt."** The opposite. An ordinary repo *adds* a visible liability, which is why it makes leverage look worse. If you see large repo balances on a dealer's balance sheet, the firm is showing you its borrowing, not hiding it. The concern with a normal repo book is refinancing risk (it must be rolled over constantly) rather than concealment.

**"This was just an aggressive interpretation of a vague rule."** The threshold was specific and numeric. A firm choosing to deliver 105% of collateral instead of the market-standard 102% was not resolving an ambiguity. It was clearing a bright line on purpose, at a cost, and then deciding not to describe having done so. The ambiguity, such as it was, lived in the disclosure obligation rather than in the recognition rule.

**"If the auditors signed it, it must have been fine."** Auditors opine on whether financial statements are fairly presented in accordance with the accounting rules. That is a narrower question than "does this give a reader an accurate impression". A technique can satisfy the recognition rule and still leave the reader with a false picture, which is exactly why disclosure requirements exist alongside recognition requirements. For what an audit does and does not cover, see [how an audit works](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch).

**"Window dressing is illegal."** Mostly it is not, and that is the uncomfortable centre of this article. Choosing to do less business on the last day of the quarter is not a crime. The legal exposure attaches to *disclosure*: telling investors your leverage is a certain number, while knowing that number was produced by a temporary and reversing transaction you did not describe, is where securities law starts to bite. The gap between "permitted accounting" and "complete disclosure" is where this entire family of tricks lives.

## How it shows up in real markets

### 1. Lehman, second quarter 2008: the largest and the last

The quarter ending 31 May 2008 was the biggest use of the programme and the last one Lehman ever reported. The firm moved **\$50.38 billion** off the balance sheet, published a net leverage ratio of **12.1x** in a Form 10-Q filed on 10 July 2008, and was running roughly **13.9x** on any ordinary day of that week (Vol. 3 at 742, 748).

The mechanism from this article, in one line: an ordinary repo would have *raised* the reported balance sheet, so the firm paid extra collateral to have the same trade classified as a sale, then used the cash to retire liabilities so both sides shrank.

The lesson is about magnitude relative to the buffer. The amount removed, \$50.38bn, was roughly 1.8 times the firm's entire tangible equity capital of about \$28bn. When the quantity you can move at will exceeds the quantity standing between you and insolvency, the published ratio has stopped measuring anything about your resilience. Lehman filed for bankruptcy on 15 September 2008, roughly nine weeks after that 10-Q.

### 2. Lehman, first quarter 2008: the quarter a careful reader could have questioned

On the Q1 2008 earnings call, chief financial officer Erin Callan told analysts the firm "did, very deliberately, take leverage down for the quarter. We ended with a net leverage ratio of 15.4 times down from 16.1 at year end" (Vol. 3 at 846).

An outsider could not have known about Repo 105. But the general shape of the problem was available. The firm was under visible pressure to deleverage, it was reporting an improvement in a self-defined ratio, and the ratio was a point-in-time measure. That combination is exactly the pattern in this article's step 4: the gap widens when the incentive is strongest. The right response was not "they are lying", which nobody could have known, but "this improvement is in a management-defined snapshot metric during the period of maximum pressure to improve it, so how much of it is real?" That is a question, correctly asked, that the accounts could not answer.

### 3. Lehman noticed its own tell, in writing

The single most instructive document in the whole record, for a reader trying to learn detection, is an internal Lehman email from May 2008. A senior external-reporting executive observed that Citigroup and JPMorgan "likely do not do Repo 105 and Repo 108 which are UK-based specific transactions on opinions received by LEH from Linklaters", and added that this "would be another reason why LEH's daily balance sheet is larger intra-month then at month-end" (Vol. 3 at 740).

Read that twice. Inside the firm, in writing, someone recorded both that the practice was unusual among peers *and* that it produced a visible difference between the daily balance sheet and the month-end balance sheet. That difference is precisely the window-dressing ratio from section 6. The signature this article teaches you to hunt for was identified by the people creating it, in an email, four months before the bankruptcy.

### 4. The auditor's own materiality threshold

Ernst & Young's walkthrough papers defined materiality for the balance-sheet close as "any item individually, or in the aggregate, that moves net leverage by 0.1 or more (typically \$1.8 billion)" (Vol. 3 at 747).

This is a gift to anyone learning to think about materiality, because it converts a vague legal concept into arithmetic you can do. If 0.1 turns is the threshold, then a programme moving 1.7, 1.9 and 1.8 turns is material by a factor of roughly seventeen to nineteen. You do not need a legal opinion to reach that conclusion; you need the auditor's own number and a division. When you can find a company's or an auditor's stated materiality threshold, use it as the yardstick against which to measure everything else you find.

### 5. Enron's Raptors, for contrast: permanent structures versus temporary timing

It is worth putting Repo 105 next to the other great accounting failure of the era, because the contrast sharpens both.

Enron's special purpose entities were built to move obligations off the balance sheet and keep them off. The engineering was in the *structure*: who owned the equity, who bore the variability, what consolidated. Repo 105 moved nothing permanently. Every dollar came back within days. The engineering was in the *timing*, and in the choice of the one day that would be photographed.

That difference changes how you detect them. Structural off-balance-sheet financing is found by reading the footnotes and reconstructing the obligation stack, the method in [off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities) and [Enron's forensic re-read](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market). Timing manipulation is found by comparing dates: period-end against average, this quarter-end against last, the balance sheet against the cash flow statement. A reader who only knows the first method will not see the second.

### 6. Banks at quarter-end, generally

Lehman's version was extreme, but the underlying incentive is structural and did not disappear in 2008. Wherever a bank's regulatory or published metric is computed from a balance on a single reporting date, there is a reason to be smaller on that date, and the mechanics are easy: allow short-term funding to mature without replacing it, reduce market-making inventory, and trim the repo book for a few days.

This is why the disclosure reforms that followed the crisis pushed toward averages, and why, when you read a financial institution's filings, the average balance sheet in the management discussion is often a more honest document than the balance sheet itself. If a bank publishes both and the two disagree persistently in the same direction, you have found something worth a question, whatever the explanation turns out to be.

### 7. Funds and operating companies at period end

The same four ingredients appear away from banking, with different levers.

For funds, the measured thing is the closing mark or the disclosed holdings, and the levers are buying into the close or reshuffling the portfolio shortly before a disclosure date. For operating companies, the measured thing is cash and the working-capital ratios, and the levers are the timing of supplier payments and collections. In both cases the transactions are real and the accounts are accurate; it is the representativeness of the date that fails.

The detection method does not change. Find any disclosure that reveals a non-reporting-date figure, compute the ratio of the ordinary day to the reported day, and track it across periods. For operating companies the richest source is usually the relationship between reported cash and the cash flow statement, plus days-payable trends across quarters, which is the ground covered in [the cash conversion cycle](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals).

## When this matters to you

Most readers of this article will never analyse a broker-dealer's repo book. The habit generalises anyway, and it generalises cheaply.

Whenever you are handed a number measured on one specific date, ask what the number looked like on the other days. That question applies to a bank's leverage ratio, a fund's holdings, a company's cash balance, a startup's monthly burn, a portfolio manager's year-end positions, and a government's debt figure. The date on which a measurement is taken is a choice, and anyone who knows the date in advance can prepare for it.

Three practical habits follow.

1. **Prefer averages to snapshots whenever both are available.** If a filing gives you an average balance and a period-end balance, the average is almost always the more honest description of how the business actually ran. Use the period-end figure to compute the ratio, not to form the view.
2. **Treat a repeating calendar pattern as a finding.** Anything that happens in the last week of every reporting period and reverses in the first week of the next is, by construction, about the reporting period. That is true even when every individual transaction is legitimate.
3. **Follow the cost.** If a firm is spending money to produce an appearance, it has told you what it cares about. Costs incurred for optics are among the clearest statements of management intent you will find in a set of accounts.

A closing caution about proportion. The presence of period-end management does not by itself mean a firm is failing, and its absence does not mean a firm is safe. Plenty of solid institutions tidy up a little at quarter end, and plenty of firms that never tidied anything went under for entirely different reasons. What the ratio gives you is a question to ask, not a verdict to reach. Treat it as an input to the kind of structured, falsifiable view described in [structuring a thesis](/blog/trading/analyst-edge/structuring-a-thesis-claim-evidence-and-catalyst), rather than as a conclusion.

This is educational material about how financial statements are constructed and read. It is not investment advice, and nothing here is a recommendation to buy or sell any security.

## Sources & further reading

Every Lehman figure in this article comes from the sources below. Page references in the text are to Volume 3 of the Examiner's Report, using the report's own pagination.

**The primary record**

- [Report of Anton R. Valukas, Examiner](https://web.stanford.edu/~jbulow/Lehmandocs/), *In re Lehman Brothers Holdings Inc.*, Case No. 08-13555 (JMP), United States Bankruptcy Court for the Southern District of New York, filed 11 March 2010. Nine volumes; **Volume 3** covers Repo 105. The specific pages relied on here: the net leverage definition and the disclosure question at 734; the Linklaters and LBIE arrangement, and the intra-month versus month-end email, at 740; quarter-end volumes and the "drug we r on" email at 742; the Ernst & Young materiality threshold at 747; the leverage table at 748; the Examiner's conclusions at 750; the 2001 origin and the \$20 billion internal limit at 762; Lehman's own definition of Repo 105 and 108 at 776; SFAS 140 paragraph 218 at 778; McDade at 815; the Q1 2008 earnings call at 846; the "window-dressing" email exchange at 860; Matthew Lee and the 12 June 2008 auditor interview at 956 to 958; accounting-motivated transactions at 962; the SFAS 140 compliance question at 964; and the fiduciary duty findings at 991.
- **Lehman Brothers Holdings Inc. filings**, via SEC EDGAR, CIK 0000806085: the Form 10-K for fiscal 2007 (net leverage definition at p. 63, reported ratio at p. 64), the Form 10-Q filed 9 April 2008 (p. 72), and the Form 10-Q filed 10 July 2008 (pp. 88 to 89). These are the filings the reported ratios of 16.1x, 15.4x and 12.1x come from.
- **SFAS 140**, *Accounting for Transfers and Servicing of Financial Assets and Extinguishments of Liabilities*, Financial Accounting Standards Board. The 98% to 102% collateralisation guidance is in paragraph 218, quoted in the Examiner's Report at Volume 3, page 778.

**The legal aftermath**

- New York State Attorney General, ["Attorney General Cuomo Sues Ernst & Young For Assisting Lehman Brothers In Financial Fraud"](https://ag.ny.gov/press-release/2010/attorney-general-cuomo-sues-ernst-young-assisting-lehman-brothers-financial-fraud), 21 December 2010. Filed in New York Supreme Court under the Martin Act, seeking the return of fees exceeding \$150 million.
- New York State Attorney General, ["A.G. Schneiderman Announces Settlement With Ernst & Young Over Auditor's Involvement In Alleged Fraud At Lehman Brothers"](https://ag.ny.gov/press-release/2015/ag-schneiderman-announces-settlement-ernst-young-over-auditors-involvement), 15 April 2015. The \$10 million settlement, under the Martin Act and Executive Law section 63(12), with most of the proceeds going to investors.
- Antoine Gara, ["Ernst & Young Settles With New York Over Lehman Brothers Repo 105 Deals"](https://www.forbes.com/sites/antoinegara/2015/04/15/ernst-young-settles-new-york-lehman-brothers-repo-105-deals/), *Forbes*, 15 April 2015. The source for the SEC having closed its investigation in 2012 without bringing suit, and for Ernst & Young's statement that there were no findings of wrongdoing.

**A note on what is not claimed here**

The Examiner identified *colorable claims*, a screening threshold meaning a claim is supported well enough to be worth pursuing. No court has ever adjudicated them, and there has been no judicial finding of fraud against any Lehman executive or against Ernst & Young in relation to Repo 105. This article states the Examiner's conclusions as the Examiner's conclusions, and nothing more.

Two further things a reader might expect to find here and will not. Published research exists on quarter-end window dressing in bank repo markets and on period-end portfolio pumping by fund managers, and both patterns are described in section 5 on their mechanics. Specific figures from that literature are deliberately omitted because they could not be verified against a primary source while writing, and an unsourced number is worse than no number.

**Related reading on this blog**

- [Lehman Brothers and the 2008 global financial crisis](/blog/trading/finance/lehman-brothers-2008-financial-crisis), for the collapse itself rather than the accounting.
- [Off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities), the structural cousin of this timing trick.
- [Enron: a forensic re-read of SPEs and mark to market](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market).
- [Hidden liabilities: leases, guarantees and contingencies](/blog/trading/forensic-accounting/hidden-liabilities-leases-guarantees-and-contingencies).
- [How an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch).
- [Reading the balance sheet: what companies hide here](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here).
