---
title: "Reading financial statements in Vietnam: red flags and the FLC case"
date: "2026-08-12"
publishDate: "2026-08-12"
description: "How Vietnamese listed-company accounts actually work, the structural risks that dominate an ecosystem-group market, and why market manipulation and accounting fraud are different crimes that leave their evidence in different documents."
tags: ["forensic-accounting", "vietnam", "vas", "ifrs", "flc", "related-party-transactions", "emerging-markets", "financial-statement-fraud", "market-manipulation", "auditing", "real-estate"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 64
depth: "deep-dive"
---

> [!important]
> **TL;DR:** A Vietnamese annual report is not a weaker version of a Western one. It is a different document, built on a prescribed chart of accounts, a historical-cost measurement basis, and a filing calendar in which three of the four reports a company publishes each year carry no audit opinion at all. Read it as though it were a 10-K and you will trust the wrong lines.
>
> - Most of the numbers people quote about a Vietnamese company come from the **unaudited quarterly report**. The audited consolidated statements arrive months later and often disagree, which is why the market has its own phrase for the gap.
> - Vietnamese Accounting Standards have **no general impairment standard and almost no fair value**. A carrying value is a record of what something cost, not an opinion about what it is worth.
> - The structural risks that dominate this market are **group-shaped, not line-item-shaped**: related-party lending inside an ecosystem, charter capital that is registered rather than paid in, founder stakes pledged for margin loans, and real-estate revenue that arrives in one lump years after the cash.
> - The FLC case is the sharpest teaching example precisely because it contains **two different crimes**. Market manipulation left its evidence in the tape and the trading accounts. The Faros charter-capital case left its evidence in the accounts. A reader who conflates them looks in the wrong document and finds nothing.
> - The single most useful ratio for this market is **three-year cumulative operating cash flow divided by three-year cumulative net profit**. If a group cannot convert reported profit into cash across a full cycle, nothing else on the page needs checking first.

This is the last post in the *Cooking the Books* series. The previous thirty-nine built a toolkit on companies most readers can look up in English: [Enron](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market), [WorldCom](/blog/trading/forensic-accounting/worldcom-the-11-billion-dollar-capitalization-fraud), [Wirecard](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros). Those cases share a convenience: the filings are in English, the standards are IFRS or US GAAP, the regulator publishes in a language you can read, and the court documents are online.

Take that toolkit somewhere else and the first thing you discover is how much of it was leaning on the infrastructure rather than the technique. The [accruals ratio](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly) still works. The [cash conversion cycle](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) still works. But the questions you ask before you compute anything are different, because the accounting standard is different, the assurance calendar is different, the ownership structure is different, and the enforcement record you would use to calibrate your suspicion is thinner and mostly not in English.

So this post does two jobs. First, it teaches you to read a Vietnamese listed-company annual report as a native document, with its own conventions and its own load-bearing lines. Second, it works through the FLC Group case to make a point that generalises far beyond Vietnam and that a surprising number of otherwise careful readers get wrong.

That point is the mental model for the whole post.

![A two-column comparison showing that market manipulation leaves evidence in the order book, trade timestamps and price series, while accounting fraud leaves evidence in the statements, the capital account and the auditor file](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-1.webp)

The diagram above is the mental model, and it is the sentence this post exists to defend: **market manipulation and accounting fraud are different crimes with different mechanics, different evidence, and different detection tools.** A company can ramp its own share price for a year with a completely clean set of accounts, because the ramping happens in trading accounts that never touch the general ledger. A company can falsify its balance sheet without a single suspicious trade ever printing. When both happen at the same firm, as they did at FLC, the temptation is to describe them as one big fraud. Resist it, because the two require you to open different documents, and studying the income statement to detect a pump is like reading a restaurant's menu to find out whether the kitchen is clean.

A note on what this post is and is not. It is educational, not investment advice. Where I describe a structural risk, I am describing something you should **test for**, with the test written out, not making an allegation about any company that is not named. Where I describe the FLC case, I state what a named court, regulator, or dated news report actually said, and I distinguish carefully between what was alleged, what a first-instance court found, and what survived appeal. Where the English-language record is thin, I say so rather than filling the gap.

## First principles: what a Vietnamese listed company actually publishes

Start with the documents, because every later mistake traces back to reading the wrong one.

A company listed on the Ho Chi Minh City Stock Exchange (HOSE) or the Hanoi Stock Exchange (HNX) publishes four different financial reports in a normal year, and they are not four versions of the same thing. They carry four different levels of assurance. *Assurance* is the word auditors use for how much independent checking stands behind a number, and it is the single most under-appreciated variable in emerging-market financial analysis.

![A four-rung ladder of assurance, from unaudited quarterly reports at the bottom through reviewed semi-annual reports and audited separate statements to audited consolidated statements at the top](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-2.webp)

Working up the ladder:

**Rung one: the quarterly report.** Management prepares it, nobody independent checks it, and it is filed within a few weeks of the quarter end. This is the document that drives the news cycle. When you read "Company X reported a 40% jump in third-quarter profit", you are reading rung one. It carries no audit opinion, no reviewer's conclusion, and no independent verification of a single number in it. That does not make it dishonest. It makes it a management assertion, which is a different kind of object from a verified fact, and the distinction matters most in exactly the situations where you care most.

**Rung two: the semi-annual report.** This one is *reviewed*, not audited. A review is a genuinely weaker procedure than an audit: the auditor performs analytical procedures and asks questions of management, and concludes with a negative statement, something close to "nothing came to our attention that causes us to believe the statements are materially misstated." Compare that with an audit opinion, which is a positive assertion: "in our opinion, the statements present fairly." The difference between "we found nothing" and "we checked and it is right" is the difference between a smoke alarm and a building inspection.

**Rung three: the audited separate statements.** These are the parent company alone. Subsidiaries do not appear as businesses; they appear as one line, "investments in subsidiaries", carried at what the parent paid for them. For a holding company at the top of a group, the separate statements can be almost content-free: a few investments, some receivables, a bank balance.

**Rung four: the audited consolidated statements.** The whole group, with intra-group transactions eliminated. This is the document that describes the actual economic entity, and it is the one that arrives last.

Here is the practical consequence. The market forms its view of a company from rung one, four times a year. The verified picture arrives once a year, months later. In between, there is a well-known Vietnamese market phenomenon that has its own name in local financial slang, the *chênh lệch sau kiểm toán*, the post-audit difference: the gap between the profit a company reported in its unaudited fourth-quarter statement and the profit that survives the annual audit. Sometimes it is small. Sometimes a company that reported a profit reports a loss after audit. The gap itself is the signal.

> A post-audit swing is not automatically fraud. Auditors force provisions the company did not want to take, reclassify revenue that failed a recognition test, and consolidate an entity management had left out. All of that is the system working. What makes it a red flag is **repetition**: a company whose audited profit is materially below its reported profit two or three years running is telling you something about the quality of its internal reporting that no single year's number can.

### The chart of accounts is prescribed, and that is a gift

Here is a structural feature of Vietnamese accounting that most foreign readers underuse. Vietnam does not let companies invent their own line items. The chart of accounts is prescribed by the Ministry of Finance, most recently in Circular 200/2014/TT-BTC, which specifies numbered accounts and the layout of the statements. Account 131 is trade receivables. Account 138 is other receivables. Account 411 is owners' invested capital, the account that holds charter capital. Account 331 is trade payables. Every company uses the same codes.

For a forensic reader, uniformity is enormously valuable. In a US filing, a company that wants to bury something can invent a line called "other operating assets, net" and roll six unrelated things into it, and you have to read the footnotes to unroll it. In Vietnam, the buckets are fixed. Which means that when something unusual shows up, it shows up **in a specific numbered account that has the same meaning at every company in the market**, and you can compare it directly against peers without normalising anything.

The line I want you to memorise is **"other receivables" (account 138, and its long-term sibling)**. In an ordinary manufacturing or retail company, other receivables should be small: deposits, advances to employees, a tax refund pending. When other receivables becomes one of the largest assets on a balance sheet, you are looking at money that left the company and went somewhere that is not a customer. That single line is the entry point to three of the four structural risks in this post.

## VAS versus IFRS: five differences that change what you can trust

Vietnamese Accounting Standards (VAS) were issued between 2001 and 2005, drawing on the International Accounting Standards of that era, and then substantially stopped moving while IFRS kept developing. The result is not "IFRS with mistakes". It is a coherent, conservative, historical-cost system that answers a different question than IFRS does.

IFRS is built to answer: *what is this business worth today, on the best available estimates?* VAS is built to answer: *what did this business actually transact, at cost, verifiably?* Both are defensible. But if you read a VAS balance sheet expecting IFRS answers, you will systematically misread five specific things.

![A five-row comparison grid showing where VAS and IFRS diverge on impairment, fair value, credit losses, revenue recognition, and the chart of accounts, with the practical consequence for a reader in each row](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-3.webp)

**One: there is no general impairment standard.** IFRS has IAS 36, which requires a company to test assets for impairment whenever there is an indicator that the carrying amount exceeds the recoverable amount, and to write the asset down if it does. VAS has provisioning rules for specific categories, notably financial investments, inventory, and doubtful receivables, but it has no general, economy-wide requirement to test a cash-generating unit for impairment and write it down.

The practical consequence is large. A factory built in 2012 that has not produced anything since 2019 can still sit on the balance sheet at cost less accumulated depreciation, with no write-down, because depreciation is a schedule and not a judgement about recoverable value. A stake in an associate that has become worthless can persist at cost. **Carrying value in a VAS balance sheet is a historical record, not a valuation.** When you compute [book value or price-to-book](/blog/trading/equity-research/multiples-101-pe-ev-ebitda-pb-ps-peg) on a Vietnamese company, you are computing a ratio to a number that has a very different meaning than the same ratio in a US filing.

**Two: fair value is barely used.** Under IFRS, investment property can be carried at fair value, financial instruments are measured at fair value through profit or loss or through other comprehensive income depending on their classification, and biological assets are fair-valued. Under VAS, historical cost dominates almost everywhere.

This cuts both directions and that is the interesting part. Land a developer bought in 2010 sits at 2010 cost, which usually **understates** its value dramatically in a market where land prices have risen for a decade. That is a hidden asset, and it is why sum-of-the-parts and net asset value approaches are more common in Vietnamese equity research than in developed markets. But an investment in an affiliate whose business has collapsed also sits at cost, which **overstates** it. The same convention hides value in one place and hides losses in another, and you cannot tell which without opening the notes.

**Three: there is no expected credit loss model.** IFRS 9 requires forward-looking expected credit losses: a lender books a provision for losses it expects, from day one, before any borrower misses a payment. VAS provisioning for receivables is largely aging-based and rule-driven: a receivable overdue by a given number of months attracts a given provision percentage.

The consequence is that **provisions lag reality instead of anticipating it.** In a deteriorating credit environment, a VAS balance sheet keeps looking healthy for several quarters longer than an IFRS balance sheet would. This matters most for banks and for any company with large trade or related-party receivables, which is a large share of the market.

**Four: revenue recognition is transfer-of-risks-and-rewards, not the five-step model.** IFRS 15 asks whether a performance obligation is satisfied over time or at a point in time, and gives specific criteria for over-time recognition. VAS 14 uses the older test: revenue is recognised when the significant risks and rewards of ownership have transferred to the buyer.

For most businesses this makes no difference. For real estate it makes an enormous one, and I devote a full section and a worked example to it later, because it produces the single most misread pattern in Vietnamese financial statements: a developer with a sold-out project reporting zero revenue.

**Five: the chart of accounts is prescribed.** Covered above. Worth repeating in this list because it is the difference that works *in your favour*, and the one most readers never exploit.

There is a sixth difference that is not about measurement at all but changes how much you can see: **the notes are typically shorter.** A Vietnamese annual report's notes to the financial statements are usually a fraction of the length of an equivalent US or European filing's. Segment disclosure is thinner. The related-party note, which I will treat as the most important note in the entire document, is often a single table of balances and transactions with counterparty names, and sometimes not even the names. This is not a defect the reader can fix. It is a constraint the reader has to work around, and working around it is most of what the rest of this post teaches.

For the general treatment of where disclosure hides, see the series post on [the footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried). The technique transfers. The raw material is thinner.

## Reading Vietnamese dong without getting lost

Vietnamese financial statements are denominated in Vietnamese dong (VND), and the unit scale trips up nearly every first-time reader. The dong is a low-denomination currency, so the numbers are large, and Vietnamese financial writing uses its own magnitude words.

| Vietnamese term | Meaning | In dong | Roughly, in US dollars |
| --- | --- | --- | --- |
| *triệu* | million | 1,000,000 | about US\$38 |
| *tỷ* | billion | 1,000,000,000 | about US\$38,500 |
| *nghìn tỷ* | thousand billion (a trillion) | 1,000,000,000,000 | about US\$38.5 million |

Statements themselves are usually presented in plain dong, so a mid-cap company's revenue line looks like `5,432,109,876,543`. Analysts and the press speak in *tỷ đồng*, billion dong, and the largest figures in *nghìn tỷ*, thousand-billion. Throughout this post I write "VND 500 bn" to mean five hundred billion dong.

For conversions I use a **round rate of VND 26,000 to one US dollar** to keep the arithmetic readable. That is a rounding convention for this article, not a market quote; the actual rate moves, and where a specific historical figure matters I give the period-appropriate rate alongside it. At this convention, VND 1 bn is about US\$38,500 and VND 1,000 bn (one *nghìn tỷ*) is about US\$38.5 million.

One more habit worth building: Vietnamese statements use the accounting convention of showing negative numbers in parentheses, and the cash flow statement presents outflows as negatives inside the operating, investing, and financing sections in the standard [three-statement structure](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock) you already know. The mechanics are familiar. It is the measurement basis underneath that is different, and that is the next section.

## Separate versus consolidated: the same group, two balance sheets

Every Vietnamese listed company with subsidiaries files both a separate (parent-only) set of statements and a consolidated set. Most readers glance at whichever one their data provider defaults to. That is a mistake, because the two documents disagree in a specific, informative way, and the disagreement is where a group-shaped problem becomes visible.

![Two side-by-side balance sheet stacks for the same group, one parent-only and one consolidated, showing that the related-party receivable and investments-at-cost lines exist only on the parent statement while the group's real borrowings appear only on the consolidated one](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-4.webp)

Consider a stylised holding company. On its **separate** balance sheet, total assets are VND 1,350 bn (about US\$51.9 million at our convention): investments in subsidiaries at cost of VND 800 bn (US\$30.8 million), other receivables from related parties of VND 500 bn (US\$19.2 million), and cash of VND 50 bn (US\$1.9 million). Against that, borrowings of VND 400 bn (US\$15.4 million) and equity of VND 950 bn (US\$36.5 million). It looks like a lightly levered holding company: debt is 30% of assets, equity is comfortable.

On its **consolidated** balance sheet, total assets are VND 1,980 bn (US\$76.2 million): property and inventory of VND 1,500 bn (US\$57.7 million), trade receivables of VND 400 bn (US\$15.4 million), cash of VND 80 bn (US\$3.1 million). Against that, borrowings of VND 1,100 bn (US\$42.3 million), non-controlling interests of VND 180 bn (US\$6.9 million), and equity attributable to the parent of VND 700 bn (US\$26.9 million).

Two things changed and both are informative.

**The group's real leverage only appears on the consolidated statement.** Borrowings went from VND 400 bn to VND 1,100 bn because the subsidiaries carry debt that the parent's own balance sheet never showed. Read only the parent and you would conclude this group has VND 400 bn of debt. It has VND 1,100 bn, and the equity supporting it is smaller than the parent statement suggested once non-controlling interests are stripped out.

**The related-party receivable only appears on the separate statement.** The VND 500 bn the parent lent to group companies is eliminated on consolidation, because a group cannot owe money to itself. That elimination is correct accounting. It is also the reason a reader who only ever opens consolidated statements will never see that the parent has half a billion dong of its assets tied up in loans to entities it controls.

This is the general rule and it is worth stating plainly:

> **Consolidated statements show you the group's obligations to the outside world. Separate statements show you the flows inside the group.** A forensic reader needs both, and the interesting number is frequently the one that disappears between them.

The same asymmetry applies to the income statement. A parent company's separate income statement often consists almost entirely of *financial income*: dividends received from subsidiaries and interest on loans to them. When you see a holding company whose separate profit is large and whose consolidated profit is small, the group is moving money upward through dividends and interest faster than the underlying businesses are earning it. That is not necessarily improper. It is always worth understanding.

For the general mechanics of how the statements interlock and where consolidation adjustments hide, the series post on [off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities) covers the version of this problem that Enron made famous. The Vietnamese version is usually less exotic and more direct: not an off-balance-sheet vehicle with a 3% outside equity sliver, but an ordinary affiliate that simply is not a subsidiary and therefore is not consolidated.

## Structural risk one: related-party lending inside an ecosystem group

Vietnamese corporate ownership is concentrated. A large share of listed companies have a founding family or a single dominant shareholder holding a controlling or near-controlling stake, and many of those founders sit at the top of a group of related companies that spans several industries. The market word for this is an *ecosystem*: a property developer, a construction arm, a securities company, a resort operator, an airline, and a handful of unlisted holding vehicles, all connected by common ownership rather than by a formal group structure.

Ecosystems are not inherently improper. They are a rational response to a market where external capital is expensive and trust is personal. But they create a specific accounting risk, and it is the risk that dominates this market: **money moves between related companies for reasons that have nothing to do with the transaction the accounts record.**

The cleanest version is a loan.

![A directed loop showing a listed parent lending to a related company, accruing interest income that flows back into its income statement while other receivables rise on its balance sheet and cash interest received stays at zero](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-5.webp)

#### Worked example: the related-party loan that leaves group cash unchanged and reported profit higher

Suppose a listed company, call it P, has an operating business generating revenue of VND 3,000 bn (about US\$115.4 million) and operating profit of VND 120 bn (US\$4.6 million). A 4% operating margin, which is unremarkable for a construction or trading business.

Now P lends VND 500 bn (US\$19.2 million) to a related company, A. A is controlled by the same family but is not a subsidiary of P, so it is not consolidated. The loan carries interest at 12% per year, which is a plausible commercial rate.

Follow it through the three statements.

**The balance sheet.** Cash falls by VND 500 bn. Other receivables (or short-term loan receivables) rise by VND 500 bn. Total assets unchanged. Nothing about this looks wrong.

**The income statement.** P accrues interest of VND 500 bn × 12% = VND 60 bn (US\$2.3 million) per year. This lands in *financial income*, below the operating line. Pre-tax profit becomes:

- Operating profit: VND 120 bn
- Plus financial income: VND 60 bn
- Pre-tax profit: **VND 180 bn** (about US\$6.9 million)

Reported pre-tax profit just rose by 50%, from VND 120 bn to VND 180 bn. One third of it is now interest from a related party.

**The cash flow statement.** The VND 500 bn shows up as an investing outflow, "loans granted to other entities". The VND 60 bn of interest income is accrued, not received, so in the operating section it is deducted as a non-cash item and in the investing section the "interest received" line shows approximately **zero**.

Now the forensic point. Nothing here breaks a rule. Lending to a related party at a commercial rate and accruing interest on it is ordinary accounting. But look at what the group as a whole has done: **its consolidated cash position is unchanged if A is inside the group, and its reported profit is a third higher.** The interest is real as an accounting entry and imaginary as cash, and it will stay imaginary for as long as A does not pay.

The tests, in order of how quickly you can run them:

1. **Interest received versus interest income.** Take financial income from the income statement and interest actually received from the investing section of the cash flow statement. In a healthy company they track. When financial income is VND 60 bn and interest received is VND 2 bn, the income is an accrual against a party that is not paying.
2. **Other receivables as a share of total assets.** Compute it, and compute its trend over three years. A line that grows from 3% of assets to 25% of assets is the group financing itself through the listed entity's balance sheet.
3. **The related-party note.** VAS 26 requires disclosure of related-party relationships and transactions. Read the table. Note the counterparty names, then check whether those names appear in the ownership disclosures elsewhere in the annual report.
4. **The provision line.** If a related-party receivable has been outstanding for years with no provision against it, the company is asserting it is fully collectible. Ask what supports that assertion.

**The intuition:** interest accrued from a related party is profit a company can print at will. The income statement cannot tell you whether it is real. The cash interest line can.

### Why this shape is so hard to see from outside

Three features of the market make related-party flows harder to trace here than in a developed market.

**Ownership disclosure runs to the legal owner, not the beneficial one.** A company's major-shareholder table shows the entities on the register. If the entity on the register is an unlisted limited-liability company with a generic name, the trail stops there unless you pull corporate registry records, which are available in Vietnam but not conveniently and not in English.

**"Related party" is a definitional test, not a common-sense one.** VAS 26, like IAS 24, defines related parties by control, joint control, significant influence, and key management personnel and their close family. A company that is genuinely part of a founder's ecosystem, but where the ownership link is held through someone outside the definition of close family, may not appear in the related-party note at all. The note is a floor on the relationships, not a complete map.

**Directors and shareholders approve their own transactions.** Vietnam's Enterprise Law and Securities Law require certain related-party transactions to be approved by the board or the general meeting of shareholders, with the interested party excluded from voting. That is a real protection. It works less well when the dominant shareholder's allies hold enough of the remaining votes.

The general treatment of this problem is in the series post on [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing). What is specific to Vietnam is the frequency and the thinness of the disclosure, not the mechanism.

### Where it becomes a systemic issue: bank lending inside a group

The most damaging version of related-party lending is not a company lending to an affiliate. It is a **bank** lending to its own shareholder's ecosystem.

Vietnam's Law on Credit Institutions sets limits on lending to related parties of a bank and on single-borrower concentration precisely because this risk is understood. The way those limits get circumvented, everywhere in the world and not only in Vietnam, is by routing the borrowing through entities that do not appear related on paper: dozens or hundreds of separate companies, each borrowing an amount below the concentration limit, each apparently unconnected, all ultimately serving one borrower.

I discuss a named Vietnamese case of exactly this shape later in the post. The forensic tests at the bank end are: the growth rate of the loan book relative to peers, the concentration of lending by sector, the ratio of loans secured on real estate or on shares, and the gap between reported non-performing loan ratios and restructured-loan disclosures. At the borrower end, the test is simpler and it is the one this whole section has been building toward: **follow other receivables**.

## Structural risk two: capital that is raised and recycled rather than deployed

This is the risk that most directly connects to the FLC case, and it is the one that a reader trained on developed markets is least prepared for, because in a developed market the mechanism is largely blocked by the plumbing.

Start with a definition, because the term is jurisdiction-specific. **Charter capital** (*vốn điều lệ*) is the amount of capital a company's owners have registered as having contributed. It appears on the balance sheet in account 411, owners' invested capital, and it is stated in the company's business registration certificate. When a Vietnamese company "increases charter capital", it registers a larger number with the authorities after the shareholders contribute.

The critical property: **charter capital is a registration fact.** The registered number says money was contributed. Whether the money arrived, stayed, and became productive assets is a separate question that only the balance sheet and the cash flow statement can answer.

![A four-step round-trip showing cash entering as a capital contribution, leaving the same week as an advance to a related entity, returning to the next contributor, and the resulting balance sheet where charter capital has grown two hundredfold while cash has not moved](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-6.webp)

#### Worked example: charter capital that grows two hundredfold while the bank balance does not move

Suppose a company, NewCo, is registered with charter capital of VND 1.5 bn (about US\$58,000). Over eighteen months it registers three increases and ends with charter capital of VND 300 bn (about US\$11.5 million). Here is one round, in journal entries.

**Day 1, money in.** Shareholder S1 transfers VND 100 bn (US\$3.85 million) into NewCo's bank account as a capital contribution.

```
Dr  Cash (account 112)                      100,000,000,000
    Cr  Owners' invested capital (411)          100,000,000,000
```

The bank statement is real. The money is genuinely in the account. An auditor confirming the balance on Day 1 confirms a true fact.

**Day 2, money out.** NewCo advances VND 100 bn to a related entity, X, described as a business advance, a deposit for a future project, or a short-term loan.

```
Dr  Other receivables (account 138)         100,000,000,000
    Cr  Cash (account 112)                      100,000,000,000
```

Also a real transaction, also properly recorded.

**Day 3, money home.** X transfers VND 100 bn to S2, who uses it to make the next capital contribution. Or X transfers it back to S1, who has now recovered the money he contributed on Day 1.

**Repeat three times.** After three rounds, NewCo's balance sheet looks like this:

| Assets | VND bn | Liabilities and equity | VND bn |
| --- | --- | --- | --- |
| Cash | 1.5 | Charter capital | 300.0 |
| Other receivables (related parties) | 300.0 | Retained earnings | 1.5 |
| **Total** | **301.5** | **Total** | **301.5** |

Registered capital has grown from VND 1.5 bn to VND 300 bn, a factor of two hundred. Net cash that entered and stayed in the business: approximately **zero**. Every individual entry is correct. Every individual transaction happened. The bank statements are genuine. And the company has no operating assets, no plant, no inventory, no meaningful cash, and one line on the asset side representing money owed by parties connected to its own shareholders.

**The intuition:** charter capital tells you what was registered, not what arrived. When registered capital rises sharply and the asset side of the balance sheet fills with receivables rather than with productive assets, the money did a lap and left.

### Why this is hard to catch and easy to detect

It is hard to catch because each step is individually legitimate. An auditor performing a bank confirmation on the day the money arrives gets a truthful confirmation. An auditor testing the capital contribution to supporting documents finds a board resolution, a share subscription agreement, and a bank credit advice. The fraud, if there is one, lives in the *sequence*, and sequences are exactly what a substantive test of an individual balance is not designed to see. The series post on [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) makes the general version of this argument.

It is easy to detect from outside, though, if you know to look, because the resulting balance sheet has an unmistakable shape. Run these four checks on any company whose registered capital has grown fast:

1. **Capital raised versus fixed assets added.** Take cash flow from financing over three years, specifically the "proceeds from share issuance" line. Compare it with the increase in fixed assets plus the increase in inventory over the same period. If a company raised VND 300 bn and its productive asset base grew by VND 5 bn, ask where the money went.
2. **The composition of the asset side.** Compute other receivables plus short-term loan receivables plus advances to suppliers as a share of total assets. Above roughly 30% for a non-financial company, that is not a working-capital pattern.
3. **Cash relative to registered capital.** A company whose charter capital is VND 300 bn and whose cash balance is VND 1.5 bn either deployed the money into assets you can see, or it did not stay.
4. **The age of the receivables.** If the same balances persist across year ends without moving, they are not trade advances. Trade advances turn over.

This is the pattern the series post on [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) describes in the revenue context, applied to the equity account instead. And the reason it matters for a *listed* company is the next step in the chain: a company with large registered capital can list, and once listed, its shares can be sold to the public. See [shell companies, reverse mergers, and how fraud gets listed](/blog/trading/forensic-accounting/shell-companies-reverse-mergers-and-how-fraud-gets-listed) for the general mechanics of using a listing as the exit.

## Structural risk three: pledged shares and the reflexive loop

A *pledge* is when a shareholder posts shares as collateral for a loan. It is completely ordinary. Founders all over the world borrow against their stakes rather than sell them, because selling triggers tax and signals a lack of confidence, while borrowing does neither.

In a market with concentrated founder ownership, high retail participation, and a securities industry that competes on margin lending, pledging becomes something more than an individual financing choice. It becomes a transmission mechanism that connects a share price to the solvency of the people who control the company, and then connects their solvency back to the share price.

Two pieces of Vietnamese market plumbing make the loop tighter than it would be elsewhere.

**Margin lending is regulated and concentrated.** Securities companies in Vietnam may lend to clients against securities, subject to regulatory limits on how much a broker may lend relative to its own equity, an initial margin requirement, a maintenance margin, and a published list of securities eligible for margin. The State Securities Commission and the exchanges maintain and update the eligibility list, and a stock can be removed from it, which forces every margin position in that stock to be unwound at once. That last mechanism has no clean analogue in a US retail account and it is a genuine cliff.

**Daily price limits cut both ways.** Under the exchanges' trading rules, HOSE applies a daily price band of plus or minus 7% around the reference price, with wider bands on HNX and UPCoM. These bands have been adjusted before, so confirm the current one on the exchange's own site rather than taking it from an article. The band is intended to damp volatility. What it actually does in a panic is prevent the market from clearing: a stock pinned at its lower limit with millions of shares of sell orders and no bid cannot be sold at all. A forced seller in that market does not get a bad price. They get **no price**, for days.

Put those together with a large pledged stake and you get a spiral.

![A declining share price chart with a horizontal margin-call trigger line, a shaded forced-sale zone below it, and annotated markers showing the pledge terms, the first forced sale, and the coverage ratio falling back through the trigger after the sale itself pushes the price down](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-7.webp)

#### Worked example: the pledged-share spiral, in numbers

Suppose a founder owns 200 million shares of a listed company L, trading at VND 20,000 per share (about US\$0.77). The stake is worth VND 4,000 bn, about US\$154 million.

The founder pledges **100 million shares** to a lender and borrows **VND 1,000 bn** (US\$38.5 million).

- Collateral value at the outset: 100 million × VND 20,000 = **VND 2,000 bn** (US\$76.9 million)
- Loan: VND 1,000 bn
- Coverage ratio: 2,000 / 1,000 = **2.0x**

The lender requires coverage of at least **1.5x**. The trigger price is therefore:

$$
P_{\text{trigger}} \;=\; \frac{\text{Loan} \times 1.5}{\text{Shares pledged}} \;=\; \frac{1{,}000\text{ bn} \times 1.5}{100\text{ mn}} \;=\; \text{VND }15{,}000
$$

Here $P_{\text{trigger}}$ is the share price at which the pledged collateral is worth exactly 1.5 times the loan, the point at which the lender is contractually entitled to act.

**Step 1. The price falls 25%, to VND 15,000.** Collateral is now 100 million × VND 15,000 = VND 1,500 bn, exactly 1.5x the VND 1,000 bn loan. Margin call. The founder must post cash or additional collateral.

Here is the trap. The founder's wealth is almost entirely the stake itself, and whatever cash exists sits inside group companies that are themselves illiquid. There is no cash to post. So the lender sells.

**Step 2. The lender sells 20 million shares at VND 15,000**, recovering VND 300 bn and reducing the loan to VND 700 bn. Remaining collateral: 80 million × VND 15,000 = VND 1,200 bn. Coverage: 1,200 / 700 = **1.71x**. Momentarily safe.

But 20 million shares is a real quantity. If the stock's average daily volume is 8 million shares, the lender has just pushed **2.5 days of normal trading volume** into a market that is already falling, and every other holder can see it happening.

**Step 3. The price falls to VND 12,000.** That is roughly three consecutive limit-down sessions at minus 7%: ${15{,}000 \times 0.93^3 \approx 12{,}070}$. Remaining collateral: 80 million × VND 12,000 = VND 960 bn against a loan of VND 700 bn. Coverage: 960 / 700 = **1.37x**, back below the 1.5x threshold. Sell again.

**Step 4. The lender sells another 15 million shares at VND 12,000**, recovering VND 180 bn, reducing the loan to VND 520 bn. Remaining collateral: 65 million × VND 12,000 = VND 780 bn. Coverage: 780 / 520 = **1.5x**. Back at the threshold, with nothing left over, and another 15 million shares now sitting in the market's memory as supply that appeared from nowhere.

The founder started with a VND 4,000 bn stake and a VND 1,000 bn loan, a 25% loan-to-value on the whole holding, which sounds conservative. Two moves later, 35 million shares are gone, the price is down 40%, and the position is still at the threshold.

**The intuition:** a pledged stake is a short option on the founder's own stock, written to a counterparty with no reason to be patient. Once the price falls through the trigger, the lender's selling and the price decline feed each other, and the daily price limit that was supposed to protect the market prevents the position from clearing.

### The reflexive part, which is the part that matters

The spiral above is a financing problem. What makes it a *forensic* problem is the feedback into the company's own accounts and behaviour.

A founder facing a margin call on a pledged stake has an acute, personal, dated need for the share price to stay above a specific number. That is a motive, and motive is what the [fraud triangle](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) calls pressure. The pressure does not create fraud on its own. But it changes the expected value of every discretionary accounting choice the company faces that quarter: whether to take the provision, whether to recognise the revenue now or next period, whether to consolidate the entity, whether to publish the quarterly report on the early or the late side of the deadline. It also changes the expected value of things that are not accounting choices at all, including buying the company's own shares, arranging for related parties to buy them, and announcing news.

This is why "how much of the founder's stake is pledged" belongs on a forensic checklist alongside the accruals ratio. It is not an accounting number. It is the variable that tells you how much the person who signs the accounts needs them to say a particular thing.

**How visible is it?** Less visible than a reader would like. Vietnamese listed companies and their insiders are subject to disclosure obligations covering insider transactions and significant shareholdings, and pledges are recorded through the depository system when shares are used as collateral. But pledge disclosure is generally less prominent and less consistently reported in English than, for example, US Schedule 13D footnotes on hypothecation, and where a stake is held through intermediate entities, the pledge may be at the level of the entity rather than the listed shares. Treat what you can see as a lower bound.

The related market-level mechanism, how margin balances across the whole broker sector amplify the index cycle, is covered in [liquidity and the margin cycle in Vietnam](/blog/trading/vietnam-stocks/liquidity-and-the-margin-cycle-vietnam) and in [the securities-broker sector, the market's highest beta](/blog/trading/vietnam-stocks/securities-brokers-sector-vietnam-highest-beta).

## Structural risk four: real-estate revenue recognition

Real estate and construction are a large share of the Vietnamese listed market by both count and market capitalisation, and the sector produces the single most misread pattern in Vietnamese financial statements. Understanding it takes five minutes and saves a reader from two opposite errors.

The mechanism: Vietnamese developers sell apartments off-plan, collecting payments in instalments over the construction period. Under VAS 14, revenue is recognised when the significant risks and rewards of ownership transfer to the buyer, which for a residential unit means **handover**. Everything collected before handover is a liability, disclosed on the balance sheet as *người mua trả tiền trước*, advances from customers.

![A three-year comparison of cash collected against revenue recognised, showing cash arriving steadily across all three years while VAS revenue is zero until the handover year, with the IFRS over-time alternative shown for contrast](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-8.webp)

#### Worked example: three years of cash, one year of revenue

Suppose a developer sells 500 apartments off-plan at VND 3.0 bn each (about US\$115,000 per unit). Total contract value: **VND 1,500 bn**, about US\$57.7 million.

The payment schedule is typical for the market: 30% on signing, 40% in instalments during construction, 25% on handover, 5% on issue of the ownership certificate.

**Year 1.** The developer collects the 30% signing instalment: VND 450 bn (US\$17.3 million).

- Revenue recognised under VAS 14: **VND 0**
- Balance sheet: advances from customers **+VND 450 bn** (a liability), inventory (construction in progress) rising
- Cash flow: operating inflow of VND 450 bn

Net profit for the year from this project: zero. Operating cash flow: strongly positive.

**Year 2.** The developer collects the 40% construction instalments: VND 600 bn (US\$23.1 million). Cumulative collected: VND 1,050 bn.

- Revenue recognised: **VND 0**
- Advances from customers now VND 1,050 bn
- Cash flow: operating inflow of VND 600 bn

Two full years, VND 1,050 bn of cash collected, zero revenue reported.

**Year 3, handover.** The units are delivered. Now everything recognises at once.

- Revenue: **VND 1,500 bn** (US\$57.7 million)
- Cost of goods sold: VND 1,050 bn
- Gross profit: **VND 450 bn** (US\$17.3 million), a 30% gross margin
- Cash collected in year 3: 25% + 5% = VND 450 bn
- Advances from customers: released to revenue, falling from VND 1,050 bn to near zero

The income statement across the three years reads **0, 0, 1,500**. The cash flow statement reads **450, 600, 450**. Under IFRS 15, if the contract met the over-time criteria (no alternative use for the asset and an enforceable right to payment for work completed to date), the same project might recognise revenue as **300, 750, 450** on a percentage-of-completion basis.

Neither presentation is wrong. They answer different questions. But three consequences follow, and all three are forensically useful.

**Consequence one: zero revenue is the healthy state, mid-project.** A developer reporting no revenue for two years is not necessarily in trouble. The line that tells you whether the project is selling is **advances from customers**, and its trend. A developer with rising advances and zero revenue is doing well. A developer with falling advances and zero revenue has stopped selling, and you will not learn that from the income statement for another year.

**Consequence two: the handover year is a timing lever.** The entire VND 1,500 bn lands in whichever period the handover falls in. Moving a handover across a year end moves the whole amount. This is not fraud, it is a legitimate operational decision with a large accounting consequence, and it is exactly the kind of discretion a management team under pressure will use. The test: compare the handover schedule disclosed in the annual report with the revenue actually recognised, and watch for handovers that cluster in December or slip from December to January. The general treatment is in [revenue recognition games](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold).

**Consequence three: the real red flag is who the buyer is.** Because revenue is lumpy and large, a developer that needs a number in a given year has an obvious move available: sell a project, a subsidiary that holds a project, or a block of land, to a related party, and recognise the gain. This produces revenue with no operating substance, and it is far more common than fabricated apartment sales because it requires no fake customers.

The tests for that version:

1. **Is the year's revenue concentrated in one transaction?** If a developer's revenue jumped and the increase is one project disposal rather than handovers, say so explicitly in your own notes.
2. **Did the cash arrive?** A project sold to a related party on deferred terms produces revenue, a receivable, and no cash. Check operating and investing cash flow against the reported gain.
3. **What happened to the land bank?** A genuine disposal reduces inventory or investment property. A circular one may not.

The [Vietnamese real-estate sector post](/blog/trading/vietnam-stocks/real-estate-sector-vietnam-land-banks-presales-bonds) covers the sector economics, including the corporate-bond channel that developers used heavily and that seized up in 2022, which the [2022 bond-crisis case study](/blog/trading/vietnam-stocks/case-study-2022-bond-crisis-property-bank-contagion) traces through to the banks.

## Who audits the accounts, and what the opinion is worth

Vietnam has a licensed audit profession, a Ministry of Finance that regulates it, and a requirement that companies with public-interest status in the securities sector be audited by firms accepted for that purpose. The market has the familiar tiers: the Big Four, a set of mid-tier international network members, and a long tail of domestic firms.

I want to make two points about this, and the second is more useful than the first.

**The first point is the one everyone already knows and overweights.** A Big Four signature is a positive signal. Those firms have more to lose, better methodology, and international quality review. All else equal, an audit by a large international firm is worth more than an audit by a firm you have never heard of.

**The second point is that "all else equal" is doing enormous work in that sentence, and the tier of the auditor is a much weaker signal than three other things you can read for free.**

**Read the opinion, not the letterhead.** The audit report is a short document with a specific structure, and its most informative parts are the ones a reader skips. A **qualified opinion** ("except for") means the auditor found something material they could not accept. A **disclaimer of opinion** means the auditor could not gather enough evidence to form a view at all, which is a far more serious statement than most readers realise: it means the audit did not conclude. An **emphasis of matter** or **material uncertainty related to going concern** paragraph is the auditor telling you, without qualifying the opinion, that something in the notes deserves your attention. Any of these on a Vietnamese filing is worth more of your time than the entire rest of the annual report.

**Read the change of auditor.** An auditor who resigns, or is replaced, in the period between the year end and the signing of the accounts is the single most informative event in this entire section. Companies change auditors for boring reasons all the time: fee negotiations, rotation requirements, a network reorganisation. But an auditor change that coincides with a delayed filing, or with a disagreement disclosed anywhere, or that happens repeatedly, is a signal that costs nothing to observe. The series treats the general version of this in [red flags in the audit report and auditor changes](/blog/trading/forensic-accounting/red-flags-in-the-audit-report-and-auditor-changes).

**Read whether the accounts arrived at all.** This is the Vietnamese-specific one, and it is the most powerful. A company that misses its filing deadline, files without an audit opinion, or simply stops publishing is telling you something no ratio can. Vietnamese exchanges have real consequences for this: shares can be moved to a warning or control list, suspended from trading, and ultimately compulsorily delisted for serious and repeated disclosure violations. **The delisting mechanism is a forensic signal, not just an administrative one**, because the reason a company stops filing is usually that it cannot produce statements an auditor will sign.

I have deliberately not given you a count of approved audit firms or a market-share figure for the Big Four in Vietnam. Those numbers exist in Ministry of Finance and professional-body publications, and I could not verify a current one from a primary source while writing this, so I am not going to state one. That is the honest position, and it is worth naming because it illustrates the broader constraint of this market: **the English-language record on Vietnamese enforcement and audit statistics is genuinely thin**, and a reader who wants those numbers needs to work in Vietnamese and go to the Ministry of Finance directly.

## Where Vietnam is heading, and why I am not giving you a date

Vietnam has an official policy of moving toward International Financial Reporting Standards. The Ministry of Finance approved a roadmap in 2020 setting out a phased transition with a preparation stage, a voluntary-application stage for a defined set of entities, and an eventual mandatory stage.

I am not going to give you the phase dates, and I want to explain why rather than quietly omit them.

Adoption timetables in this area have moved, in Vietnam and in most jurisdictions that have attempted the same transition, and I could not reach a primary text or an authoritative current statement of the timetable while writing this. Under the rule this series has followed for forty posts, a number I cannot source is one I do not get to write. So: the policy direction is real and official, the destination is IFRS, and **you should check the Ministry of Finance's own current publication before relying on any specific year**, because a stale roadmap date repeated confidently is exactly the kind of false precision that makes the rest of an analysis untrustworthy.

What matters for you as a reader today is simpler and is not in doubt: **the statements you are reading now are VAS statements**, prepared on the historical-cost, prescribed-format basis described above, and you should read them as such. A minority of large Vietnamese groups, particularly those with foreign parents, foreign listings, or international lenders, already prepare an additional IFRS-basis set of accounts alongside the statutory VAS ones. Where such a set exists, it is usually the more informative document, and the difference between the two is itself worth reading: the reconciliation shows you exactly which of the five differences above bites hardest for that particular company.

## The FLC case: what the record actually says

Now the case. I am going to be pedantic about the difference between what was alleged, what a first-instance court found, and what survived appeal, because this is a living person and a criminal matter, and because the pedantry is also where the teaching is.

![A two-track timeline showing the market track of undisclosed share sales, a cancelled trade, a regulatory fine and an arrest running in parallel with the accounting track of Faros charter-capital increases and the delistings, both converging on the 2024 trial and the 2025 appeal](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-9.webp)

FLC Group is a Vietnamese conglomerate that grew through the 2010s across real estate, resorts, construction, and, from 2019, an airline. Trinh Van Quyet was its chairman. The group is the textbook example of the ecosystem structure this post described: a listed parent, a listed construction affiliate, a securities company, resorts, and an airline, connected by common control.

Note that all dong figures below are converted at the round VND 26,000 per US dollar convention used throughout this post, purely to give you a sense of scale. These are 2016 to 2025 dong amounts converted at a 2026 rate, not contemporaneous dollar values.

### The market track

On **10 January 2022**, Quyet sold 74.8 million FLC shares without making the prior disclosure that Vietnamese rules require of an insider. On **11 January 2022**, the Ho Chi Minh City Stock Exchange cancelled the transaction and investors on the other side were refunded. VnExpress described the cancellation as the first time HOSE had done such a thing. On **17 January 2022**, the State Securities Commission fined him VND 1.5 bn (about US\$58,000), described in reporting as the maximum then available under the securities rules, and suspended him from trading for five months. That administrative penalty was subsequently **annulled in April 2022**, after the criminal prosecution had begun.

Two things are worth pausing on there. First, the largest financial penalty the securities regulator could impose for the episode was on the order of US\$58,000, against a sale of 74.8 million shares: **an administrative penalty regime calibrated to ordinary misconduct is not a deterrent at that scale**, and that is part of why the matter escalated. Second, the annulment is not the regulator backing down. It is the ordinary consequence of the same conduct moving from the administrative track to the criminal one, and it is a useful reminder for anyone reading enforcement records as a data series: **a withdrawn fine can mean the case got more serious, not less.** If you screen companies on regulatory sanctions, a disappearing penalty is a lead to follow, not a clean slate.

On **29 March 2022**, the Criminal Police Agency (C01) of the Ministry of Public Security announced his prosecution and arrest for manipulating the securities market under **Article 211 of the Penal Code**, together with concealing information in securities activities.

Everything on this track happened in trading accounts, on the exchange, and in disclosure filings. **None of it is visible in FLC's financial statements.** A reader who spent January 2022 studying FLC's balance sheet would have learned nothing about any of it.

### The accounting track

The second track runs through a different company. Faros Construction JSC, whose shares traded as **ROS**, was acquired in 2011 with charter capital of **VND 1.5 bn** (about US\$58,000). Through a series of capital increases it reached charter capital of **VND 4,300 bn** (about US\$165.4 million), and ROS shares were listed on HOSE.

At the VND 10,000 par value standard in the Vietnamese market, VND 4,300 bn of charter capital corresponds to 430 million shares. The Hanoi People's Court found that of that amount, **VND 3,102 bn was inflated**, that is, not genuinely contributed. The court put that at 72.15% of the registered capital and translated it into a per-share figure: of each VND 10,000 share, **VND 7,215 was inflated value**. Counting later increases, the court identified total charter capital of VND 5,675 bn (about US\$218.3 million), of which VND 2,573 bn (about US\$99.0 million) was genuine and VND 3,102 bn (about US\$119.3 million) was not.

Those numbers are the worked example from earlier in this post, with the court's arithmetic attached. This is exactly the round-trip pattern: registered capital rising while contributed cash does not stay.

The court found that shares representing that inflated capital were then sold to the public, and that **VND 3,621 bn** (about US\$139.3 million) was appropriated from **25,853 investors** in the initial sale.

**This track is an accounting fraud, and it is visible in accounts.** Not in FLC's accounts, in Faros's. That distinction is the whole point and I will return to it.

### The delistings

**5 September 2022**: ROS was compulsorily delisted from HOSE. At that point, according to later reporting on the compensation process, roughly **63,075 investors held over 567 million ROS shares**. Note the gap between that number and the 25,853 investors in the initial sale: most of the eventual holders bought in the secondary market, years after the capital was inflated, from sellers who were not the defendants.

**2023**: FLC was compulsorily delisted from HOSE. Both stocks were subsequently suspended on UPCoM, the market for unlisted public companies, from the time they left HOSE, for information-disclosure violations. As of December 2025 reporting, **FLC had not disclosed operating results since mid-2022**.

**19 December 2025**: the State Securities Commission revoked the public-company status of both FLC and FLC Faros, which removed their eligibility to trade on UPCoM at all.

Follow that chain, because it is the practical consequence of everything this post has argued. The company stopped producing statements an auditor would sign. Because it stopped filing, it was delisted. Because it was delisted and remained in violation, it was suspended. Because it eventually lost public-company status, its shareholders were left holding an instrument with no market. **The failure to file was not a technicality that preceded the real damage. For an ordinary shareholder, it was the mechanism of the damage.**

### The trial and the appeal

On **5 August 2024**, the Hanoi People's Court delivered its first-instance verdict in a trial with **50 defendants**. Quyet was sentenced to:

- **18 years** for fraudulent appropriation of property
- **3 years** for manipulating the securities market
- **21 years in total**

The court put the illicit profit from the manipulation at **over VND 700 bn** (about US\$26.9 million), a separate figure from the VND 3,621 bn appropriated in the Faros matter. At the time of the first-instance verdict, restitution stood at VND 237 bn (about US\$9.1 million).

On **26 June 2025**, the High People's Court in Hanoi ruled on the appeal. It **upheld the convictions** and partially accepted the appeal on sentence. The result:

- The fraud sentence was reduced from 18 years to **7 years**
- The 3-year manipulation sentence was replaced by a **fine of VND 4 bn** (about US\$154,000)
- A total reduction of 14 years from the original 21

The appellate court cited mitigating factors including restitution, which by then had reached **VND 1,886 bn** (about US\$72.5 million), his health, over 100 injured parties and around 5,000 petitions requesting leniency, and his admissions.

On **24 July 2025**, reporting described compensation beginning to reach **28,014 people** (133 classified as direct victims and 27,881 as related rights-holders), totalling approximately **VND 1,786 bn** (about US\$68.7 million), with the 133 direct victims compensated at **VND 7,215 per ROS share**, the court's per-share measure of inflated value.

### Where the case stands, as of August 2026

Two dated facts, both from VnExpress. On **26 January 2026**, Quyet made his first public reappearance in nearly four years, attending a meeting between FLC Group, Bamboo Airways, and the South Korean ambassador. On **27 March 2026**, he was presented in the role of chairman of FLC Group at an event in Gia Lai, roughly four years after leaving the position.

**I could not find, in the reporting available to me, a statement of the legal basis on which he left custody**, whether that was early release, amnesty, or another mechanism. I am not going to speculate about it. What I can state is what those two dated reports say: as of the most recent reporting I could verify, he is out of custody and has resumed the chairmanship of FLC Group.

## Two crimes, two documents: the point of the case

Go back to the first figure in this post. The FLC matter contains three legally distinct things, and they are separated by which document holds the evidence.

**One: a disclosure violation.** The 10 January 2022 sale of 74.8 million shares without prior disclosure. Handled administratively by the State Securities Commission with a fine and a trading suspension. The evidence is the disclosure filing that was not made and the trade record that shows the sale. It is not in any financial statement.

**Two: market manipulation.** Prosecuted under Article 211, with illicit profit the court put at over VND 700 bn. The evidence for this kind of offence is trade data: which accounts traded, when, in what sequence, and whether they were connected. Exchange surveillance and the regulator find it. **It leaves no trace in a company's accounts at all**, because the trading happens in brokerage accounts belonging to individuals, not in the company's general ledger.

**Three: fraudulent appropriation of property, via inflated charter capital.** Prosecuted under a different article entirely, with a finding that VND 3,102 bn of registered capital was not genuinely contributed and that VND 3,621 bn was appropriated from 25,853 investors. **This one is an accounting fraud in the strict sense.** Its evidence is the capital account, the contribution records, the bank flows around each contribution date, and the audit file for those years.

Now the practical lesson, which is why this case earns the last slot in a forensic accounting series.

**A reader who conflates the two criminal offences looks in the wrong place and finds nothing.** If you had been told in 2021 that something was wrong at the FLC ecosystem and you had responded by pulling FLC's financial statements and running an M-Score, you would have been analysing the wrong company for the capital case and the wrong document for the manipulation case. The capital fraud was at Faros, and it had already happened years earlier, in the equity section rather than the income statement, in a form that no earnings-manipulation model is built to detect. The manipulation was in trading accounts that never touched a financial statement.

This generalises, and the generalisation is the closing argument of the series:

> **Forensic accounting is a tool with a domain.** It detects misstatement in financial statements. It does not detect price manipulation, insider dealing, disclosure violations, or theft that never passes through the accounts. When you suspect something is wrong at a company, the first question is not "what does the M-Score say", it is **"if this were true, which document would it be in?"** Answer that first, and then reach for the tool that reads that document.

The corollary is more encouraging. Once you have located the right document, the technique from the previous thirty-nine posts works, and it works in Vietnamese as well as it works in English, because the arithmetic of a round-trip does not care what language the statement is in. The capital fraud in this case has the exact shape of the worked example in the round-tripping section: money in, money out, a receivable left behind, registered capital that grew while cash did not. A reader who had Faros's balance sheet, who knew to compare registered capital against the productive assets it was supposed to have bought, and who knew that "other receivables" is where the money goes, had everything they needed.

## How it shows up in real markets

**The listing as the exit.** The Faros pattern is a specific and important one: the fraud is not in the operating accounts of a listed company, it is in the capital account of a company **on its way to becoming** listed. The listing is not incidental, it is the point, because a listing converts an inflated private capital account into shares that can be sold to the public at a market price. The series post on [shell companies, reverse mergers, and how fraud gets listed](/blog/trading/forensic-accounting/shell-companies-reverse-mergers-and-how-fraud-gets-listed) covers the general mechanics. The forensic implication is that **the highest-risk window is the two or three years before a listing, and that is exactly the period for which the least information is published.** For a newly listed company, read the prospectus's capital history the way you would read a cash flow statement: where did the registered capital come from, and what did it buy.

**Related-party lending through a bank.** The largest Vietnamese fraud case by amount is structurally about the risk this post's first structural section described, scaled up to a bank. In the Van Thinh Phat matter, the High People's Court in Ho Chi Minh City ruled on appeal on 3 December 2024. It found that Truong My Lan controlled over 91% of Saigon Commercial Bank (SCB) through others while holding no official position at the bank, and that bank officers were directed to disburse against fabricated loan files, withdrawing customer deposits to fund real-estate projects. Reporting on that ruling put outstanding SCB loans in the case at more than VND 673,000 bn (on the order of US\$25.9 billion at our convention) and the embezzlement finding against her at VND 415,000 bn (on the order of US\$16.0 billion). The sentencing position in that case has continued to move through subsequent proceedings, so treat the 3 December 2024 ruling as a dated snapshot rather than the current state.

The mechanism is worth stating plainly because it is the general one: **concentration limits and related-party lending limits are defeated by multiplying the number of apparent borrowers.** No single loan breaches a limit. The aggregate is one borrower. The detection tests are at the aggregate level (loan growth versus peers, sector concentration, collateral composition), not at the level of any individual credit file, which is precisely why file-by-file testing does not find it.

**The post-audit swing.** Every year, Vietnamese financial media publish comparisons of companies whose audited results differ materially from their reported fourth-quarter results. This is a free, recurring, market-wide screen that most foreign readers never run. It requires two documents that are both public, and it identifies companies whose internal reporting either cannot produce an auditable number or does not try to.

**Removal from margin eligibility.** When a stock is removed from the list of securities eligible for margin trading, every leveraged position in it must be unwound. This is a scheduled, published, mechanical forced-selling event, and it interacts directly with the pledged-share spiral described earlier: the same news that removes a stock from margin eligibility often also marks the collateral down. Vietnamese financial media report these removals; ROS itself was reported as being made ineligible for margin trading well before the criminal case broke.

**Delisting is a wealth event, not a formality.** The FLC and ROS chain, from disclosure violation to suspension to compulsory delisting to loss of public-company status, took roughly three and a half years and ended with shareholders holding paper with no venue. When a company's filings go late, the risk to a minority shareholder is not primarily that the next number will be bad. It is that there will be no next number, and eventually no market. On the pattern of what happens to Vietnamese shareholders when the property and bond cycle turns, see the [2022 bond-crisis case study](/blog/trading/vietnam-stocks/case-study-2022-bond-crisis-property-bank-contagion) and the [theme-wave post on how speculative pumps work](/blog/trading/vietnam-stocks/theme-waves-song-nganh-how-speculative-pumps-work).

**The compensation arithmetic.** One last observation, because it is unusual and instructive. The court in the Faros matter did not compensate holders for their losses. It compensated them for **the inflated portion of the par value**, VND 7,215 per share, derived from the finding that 72.15% of the registered capital was not genuinely contributed. A holder who bought ROS in the secondary market at many times par is not made whole by that measure, and most of the 63,075 holders at delisting were exactly such buyers. The gap between "what the fraud was measured as" and "what an investor lost" is very wide, and it is a reminder that **a conviction is not a recovery**.

## Common misconceptions

**"Vietnamese accounts are less reliable than Western ones."** This is the wrong frame and it leads to the wrong behaviour, which is either dismissing the whole market or trusting the audited statements of a Big Four client uncritically. VAS is a conservative, historical-cost, prescribed-format system. It tells you less about value and more about transactions. The disclosure is thinner, which is a real constraint. But the reliability question is not standard-versus-standard, it is **which specific numbers each standard leaves to management discretion**, and VAS leaves fewer of them to discretion than IFRS does, not more. What VAS does not do is force a company to tell you when an asset has become worthless.

**"A Big Four audit means the numbers are safe."** Auditors from every tier have signed statements that later turned out to be wrong, in every market. The series covered [Enron and Arthur Andersen](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market) and [Wirecard and EY](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros). The auditor's identity is a weak prior. What is informative is the **opinion itself**, any emphasis-of-matter or going-concern paragraph, and, most of all, **a change of auditor that is not explained**. Read the opinion paragraph, not the letterhead. See [red flags in the audit report and auditor changes](/blog/trading/forensic-accounting/red-flags-in-the-audit-report-and-auditor-changes).

**"If the share price collapsed, the accounts must have been fraudulent."** This is the specific error this post exists to correct, and it runs in both directions. A share price can collapse because a manipulated price was never supported by anything, with the accounts perfectly clean. And accounts can be fraudulent for years while the share price does nothing interesting at all. Price is evidence about the market. Accounts are evidence about the company. They are separate observations.

**"Zero revenue means the company is not selling anything."** For a Vietnamese developer mid-project, zero revenue is the normal state, and the line that tells you about demand is advances from customers. Applying a price-to-sales screen naively to this sector produces nonsense.

**"Related-party transactions are inherently improper."** They are not. A group that shares a treasury function, buys materials centrally, or leases property from an affiliate at market rates is doing something ordinary and often efficient. The forensic question is never "are there related-party transactions", because in a concentrated-ownership market the answer is always yes. It is **"are they priced at arm's length, do they settle in cash, and are they growing faster than the business"**.

**"The consolidated statements are the real ones, so I can skip the separate statements."** The consolidated statements are the ones that describe the group's position against the outside world, and they are the right basis for valuation. They are also the ones in which every intra-group flow has been eliminated, which is to say every flow this post has taught you to look for. Read both.

**"A price limit protects investors in a crash."** A daily band limits how far a price can move in one session. It does not create buyers. In a forced-selling episode, the band converts a price decline into a queue: a stock pinned at its lower limit with no bid is not a stock that fell 7%, it is a stock you cannot sell at all, for as long as the queue lasts. Holders discover that the protection was a delay.

## The checklist: turning an annual report into a testable object

The point of a checklist is that it can fail. A list of things to "consider" is not a checklist, it is a mood. Each row below names a document to pull, a number to compute, and a level at which you stop reading and start asking questions.

![A checklist table with columns for the document to pull, the number to compute, and the threshold at which to stop, with a row each for related parties, cash quality, capital, audit and pledged shares](/imgs/blogs/reading-financial-statements-in-vietnam-red-flags-and-the-flc-case-10.webp)

**1. Related parties.** Pull the related-party note (VAS 26) and the parent-only balance sheet. Compute related-party receivables plus other receivables as a share of total assets, for three years. Stop if it is above roughly 20% and rising, or if the counterparties are not named.

**2. Cash quality.** Pull three years of consolidated cash flow statements. Compute cumulative operating cash flow divided by cumulative net profit. Stop if it is below 0.5. This is the single highest-yield test in the entire post, it works in every market, and it is the reason [cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) is the series' most-repeated sentence. A company can manage one year's cash conversion. Three years of a cumulative gap means the profit is an accrual.

**3. Capital.** Pull the charter capital history and the financing section of the cash flow statement. Compare capital raised against the increase in fixed assets plus inventory. Stop if capital was raised and the productive asset base did not move.

**4. Audit.** Pull the auditor's name, the opinion paragraph, and the prior year's. Note how long the firm has served and whether the opinion changed. Stop on a qualified opinion, a disclaimer, an adverse opinion, a going-concern emphasis, or an auditor change with no stated reason.

**5. Pledges and insider position.** Pull the insider ownership disclosures and any pledge disclosures. Estimate what share of the founder's stake is pledged. Stop if the majority of the controlling stake is collateral, and treat what you can see as a lower bound.

**6. The post-audit gap.** Pull the fourth-quarter report and the audited annual report for the same year, for three years. Compute the change in profit between them. Stop if it exceeds roughly 20%, and stop harder if it happens repeatedly.

Two notes on how to use this. First, a single failed row is a question, not a verdict. Companies fail individual rows for innocent reasons all the time, and the honest output of this checklist is usually "I need to understand row 3 before I can hold a view", not "this is a fraud". Second, the rows are ordered by cost, not by importance. Rows 2 and 6 take fifteen minutes with a spreadsheet. Rows 1 and 5 can take a day. Run the cheap ones first and let them tell you whether the expensive ones are worth it.

For the quantitative screens that sit behind rows 1 and 2, the series has dedicated posts: [the Beneish M-Score](/blog/trading/forensic-accounting/the-beneish-m-score-detecting-earnings-manipulation), [the Altman Z-Score](/blog/trading/forensic-accounting/the-altman-z-score-predicting-financial-distress), [the accruals ratio](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly), and [forensic ratios: DSO, DIO, DPO and margin anomalies](/blog/trading/forensic-accounting/forensic-ratios-dso-dio-dpo-and-margin-anomalies). A caution on transplanting them: all of these models were estimated on US data, with US disclosure and US accounting. Their **thresholds do not transfer**, and a score computed on VAS statements is not comparable to the published cutoffs. What transfers is the *direction* of each component. Use them to rank companies within the same market against each other, not to compare a Vietnamese company against an American cutoff.

## When this matters to you

You may reasonably be wondering why a reader outside Vietnam should learn a foreign accounting standard's quirks. Two answers, one practical and one general.

**The practical answer is that the market is being reclassified, and the money follows the label.** FTSE Russell has announced and confirmed Vietnam's reclassification from Frontier to Secondary Emerging market status, to be implemented in stages: a first tranche on **21 September 2026 at 10% of full weight**, with full inclusion scheduled for **September 2027**. Note the tense carefully, because it is widely misreported: as of this writing the reclassification is **announced and scheduled, not yet in effect**. And the picture is not unanimous. **MSCI still classifies Vietnam as a Frontier market and declined to add it to its review watch list in June 2026.** Two of the largest index providers therefore disagree about what kind of market this is, which is itself the most honest single sentence anyone can write about Vietnam's current status.

What follows from a reclassification is mechanical: passive funds tracking the emerging-market benchmarks must buy, and active emerging-market managers who were previously out of scope come into scope. A large number of people are about to read Vietnamese financial statements for the first time, many of them applying developed-market habits to a VAS document. Which is to say: the specific misreadings this post catalogues, the impairment that never happens, the developer with no revenue, the parent-only balance sheet that hides the group's debt, are about to get a great deal more common.

The composition of the market matters too. Vietnam's trading is heavily retail. Putting a precise number on it is harder than it looks: the widely repeated "80 to 90% retail" figure does not trace to a regulator or exchange publication that I could find. The most recent attributable estimate I can offer is from the chief executive of VCBF, who put the retail share of trading at **75 to 80% in March 2026**. Treat it as a practitioner's estimate rather than an official statistic. Either way the implication holds: price discovery in this market leans on a shallower base of professional analysis than in a developed market, which is exactly the condition under which careful statement reading is worth the most.

**The general answer is that this post was never really about Vietnam.** Every technique here transfers to any market where ownership is concentrated, the accounting standard is conservative and historical-cost, the disclosure is thinner than you would like, and the enforcement record is hard to research in your own language. That describes a large majority of the world's listed companies. Substitute the local terms for charter capital, customer advances, and the prescribed chart of accounts, and the checklist runs unchanged.

And that is the honest close to forty posts. The series has spent a lot of pages on frauds that were eventually discovered, which creates a survivorship illusion: the cases we can study are the ones that ended. What the toolkit actually buys you is not certainty. It is the ability to convert a vague unease into a specific, cheap, falsifiable question, and then to notice when the answer does not come. Companies that will not answer a specific question are the most reliable signal in this entire field, and noticing that a question went unanswered costs nothing but attention.

Where to go next, in the series: [reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) if you only ever read one more of these; [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing) for the mechanism that dominates concentrated-ownership markets; [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) for why the round-trip in this post survives a competent audit. For the Vietnamese market itself, [the full Vietnam sector playbook](/blog/trading/vietnam-stocks/capstone-a-full-vietnam-sector-investing-playbook) and [the real-estate sector post](/blog/trading/vietnam-stocks/real-estate-sector-vietnam-land-banks-presales-bonds). For turning a red flag into an actual decision rather than a permanent suspicion, [what would change my mind](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) and [stress-testing your thesis with a pre-mortem](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem). And for the valuation side of the same statements, [quality of earnings: accruals, one-offs, red flags](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags).

This is educational material, not investment advice, and nothing here is a recommendation about any security.

## Sources & further reading

**The FLC and Faros case.** All of the following are VnExpress reports; the Vietnamese-language record is substantially fuller than the English-language one, and every figure in the case section above is taken from these dated reports rather than from a secondary summary.

- [Arrest and initial prosecution](https://vnexpress.net/chu-tich-flc-trinh-van-quyet-bi-bat-4444881.html), VnExpress, 29 March 2022. Source for the 29 March 2022 prosecution under Penal Code Article 211, the 10 January 2022 sale of 74.8 million FLC shares without prior disclosure, HOSE's cancellation of the trade on 11 January, and the State Securities Commission's VND 1.5 bn fine and five-month trading suspension of 17 January 2022.
- [First-instance verdict](https://vnexpress.net/toa-tuyen-an-voi-ong-trinh-van-quyet-va-49-bi-cao-4777616.html), VnExpress, 5 August 2024. Source for the Hanoi People's Court verdict, the 50 defendants, the 18 years for fraudulent appropriation plus 3 years for market manipulation totalling 21 years, the VND 3,621 bn appropriated from 25,853 investors, the charter capital raised from VND 1.5 bn to VND 4,300 bn, and the manipulation profit of over VND 700 bn.
- [How the per-share compensation figure was derived](https://vnexpress.net/bi-hai-nhan-lai-7-215-dong-mot-co-phieu-ros-trong-vu-trinh-van-quyet-4778037.html), VnExpress, 5 August 2024. Source for the court's finding that VND 3,102 bn of registered capital was inflated, that this was 72.15% of the total and therefore VND 7,215 of each VND 10,000 share, and for the VND 5,675 bn total charter capital of which VND 2,573 bn was found genuine.
- [Appellate judgment](https://vnexpress.net/cuu-chu-tich-flc-trinh-van-quyet-duoc-giam-14-nam-tu-4906432.html), VnExpress, 26 June 2025. Source for the High People's Court in Hanoi upholding the convictions, reducing the fraud sentence from 18 years to 7, replacing the 3-year manipulation sentence with a VND 4 bn fine, the total 14-year reduction, and restitution of VND 1,886 bn.
- [Compensation payments beginning](https://vnexpress.net/hon-28-000-nha-dau-tu-mua-co-phieu-ros-cua-flc-bat-dau-duoc-tra-tien-4918148.html), VnExpress, 24 July 2025. Source for the 28,014 recipients (133 direct victims and 27,881 related rights-holders), the approximately VND 1,786 bn total, the 5 September 2022 delisting of ROS from HOSE, and the 63,075 investors holding over 567 million ROS shares at that point.
- [Loss of public-company status](https://vnexpress.net/tap-doan-flc-sap-roi-san-chung-khoan-4995986.html), VnExpress, 19 December 2025. Source for the State Securities Commission revoking the public-company status of FLC and FLC Faros, the compulsory delisting of ROS from HOSE in 2022 and FLC in 2023, the subsequent UPCoM suspensions for disclosure violations, and FLC having published no operating results since mid-2022.
- [First public reappearance](https://vnexpress.net/cuu-chu-tich-flc-trinh-van-quyet-tai-xuat-5010229.html), VnExpress, 26 January 2026, and [return to the chairmanship](https://vnexpress.net/doanh-nhan-trinh-van-quyet-tro-lai-vai-tro-chu-tich-tap-doan-flc-5055452.html), VnExpress, 27 March 2026. Neither report states the legal basis on which he left custody, which is why this post does not either.

**The Van Thinh Phat comparison.**

- [Appellate ruling](https://vnexpress.net/ba-truong-my-lan-bi-tuyen-y-an-tu-hinh-4822890.html), VnExpress, 3 December 2024. Source for the High People's Court in Ho Chi Minh City ruling, the finding of control over more than 91% of SCB shares while holding no position at the bank, the disbursement against fabricated loan files, and the figures of more than VND 673,000 bn of outstanding loans and VND 415,000 bn attributed to the embezzlement finding. The sentencing position in this case has continued to move in subsequent proceedings, so read that ruling as a dated snapshot.

**Accounting and regulatory framework.** Circular 200/2014/TT-BTC (Ministry of Finance) is the source of the prescribed chart of accounts and statement formats, including the account codes named in this post. VAS 14 governs revenue recognition and VAS 26 related-party disclosures. Circular 96/2020/TT-BTC governs disclosure of information on the securities market, including the filing calendar for quarterly, semi-annual and annual reports. The Ministry of Finance approved Vietnam's IFRS application roadmap by decision in 2020; as explained above, I have deliberately not quoted its phase dates because I could not confirm the current timetable from a primary text, and you should check the Ministry's own current publication before relying on any year.

**Market classification and structure.** FTSE Russell's country classification announcements are the source for Vietnam's reclassification to Secondary Emerging status, with a first implementation tranche on 21 September 2026 at 10% weight and full inclusion in September 2027. MSCI's June 2026 market classification review is the source for Vietnam remaining a Frontier market and not being added to the watch list. The 75 to 80% retail share of trading is attributed to the chief executive of VCBF, speaking in March 2026, and is a practitioner's estimate rather than an official statistic.

**Currency.** Dong figures in this post are converted at a round VND 26,000 per US dollar for readability. For reference, the market rate was approximately VND 26,033 per US dollar on 11 August 2026. Case figures dated between 2016 and 2025 are converted at that same rate purely to convey scale; they are not contemporaneous dollar values, and because the dong was stronger against the dollar earlier in that period, a period-appropriate conversion would give somewhat larger dollar figures.

**Where the record is thin.** I could not verify from a primary source, and have therefore not stated: a current count of audit firms approved for securities-sector public-interest entities, a Big Four market share of Vietnamese listed-company audits, the current phase dates of the IFRS roadmap, or a regulator-published figure for the retail share of trading. Each of those numbers exists somewhere in Vietnamese-language Ministry of Finance and professional-body material. If you need them, go there directly rather than to an English-language secondary source.

