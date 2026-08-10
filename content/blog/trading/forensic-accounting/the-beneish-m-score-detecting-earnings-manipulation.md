---
title: "The Beneish M-Score: Detecting Earnings Manipulation"
date: "2026-08-10"
publishDate: "2026-08-10"
description: "A first-principles walkthrough of the eight-variable statistical model that scores a company's financial statements for signs of earnings manipulation, including the full formula, a complete worked calculation, and an honest account of what the score cannot see."
tags: ["beneish-m-score", "earnings-manipulation", "forensic-accounting", "financial-statement-fraud", "accruals", "red-flags", "enron", "fraud-detection", "earnings-quality", "screening", "base-rates"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 50
---

> [!important]
> **TL;DR** — The Beneish M-Score compresses two years of a company's financial statements into a single number that says how much that company's accounts *look like* the accounts of firms later caught manipulating earnings.
>
> - It is built from **eight ratios**, each comparing year `t` to year `t-1`: DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI and TATA. Each one is a detector aimed at a specific accounting trick.
> - Beneish (1999) reports the weighted formula **M = -4.840 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI + 0.115·DEPI - 0.172·SGAI + 4.679·TATA - 0.327·LVGI**, with firms classified as manipulators when the score exceeds **-1.78**.
> - **TATA — total accruals to total assets — carries a coefficient of 4.679, roughly five times any other input.** Strip away the presentation and the M-Score is an accruals detector with seven supporting witnesses.
> - The number everyone quotes, "76% of manipulators caught, 17.5% of non-manipulators misclassified", is the *most permissive end* of the range Beneish reports for his estimation sample. At the -1.78 cutoff itself the paper reports 74% and 13.8% in-sample, and 50% and 7.2% in the holdout sample.
> - Because manipulation is rare, **most of the companies a good screen flags are innocent**. That is arithmetic, not a flaw in the model, and it is the single most important thing to understand before using it.
> - A team of Cornell students applied it to Enron in **May 1998** and wrote that the model suggested Enron "may be manipulating its earnings" — more than three years before the bankruptcy. The score also could not have seen the off-balance-sheet vehicles that actually broke the company.

Suppose you are handed the annual reports of two thousand companies and told that somewhere in the pile are the twenty or thirty that are quietly lying. You have a week. You cannot call management, you cannot visit a warehouse, you cannot subpoena a bank. All you have are the numbers those companies chose to publish about themselves.

Where do you even start?

The instinctive answer is to read carefully and trust your nose. The problem is that a company cooking its books is *specifically optimising* to survive a careful read. Its statements balance. Its auditor signed. Its footnotes are dull. What it cannot easily do is make its numbers move the way an honest company's numbers move, because the lie has to be paid for somewhere — a receivable that never converts to cash, a cost parked on the balance sheet instead of the income statement, a depreciation schedule that quietly lengthens.

In 1999, an accounting professor named Messod D. Beneish published a paper asking a blunt question: if you take every company the SEC later caught manipulating earnings, and compare their financial statements in the year *before* anyone knew, against thousands of ordinary companies — do the manipulators look different? And can you write down the difference as an equation?

They do, and he could. The result is the **Beneish M-Score**, and the diagram below is the mental model for the whole of this article: two years of statements go in on the left, eight ratios measure how the second year differs from the first, a weighted sum collapses them into one number, and that number gets compared to a threshold.

![Two years of financial statements feed eight ratio detectors, which feed a weighted sum, which produces a single M-score compared against the -1.78 threshold](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-1.webp)

Everything else in this post is a tour of that picture: what each of the eight detectors is looking for and why, where the weights come from, what the threshold means, how to compute the whole thing by hand from real statement lines, and — the part most write-ups skip — a clear-eyed account of how often it is wrong and what kinds of fraud it is structurally incapable of seeing.

This is educational material about a screening tool, not investment advice, and a high M-Score is not an accusation. Hold that thought; we will come back to it repeatedly, because it is the single easiest way to misuse this model.

## Foundations: how the M-Score actually works, from first principles

Before any formula, we need four ideas. If you already know them, skim; if you do not, nothing later will make sense without them.

### What "earnings manipulation" actually means

A company's **earnings** — its profit, its net income, the bottom line — is not a measured quantity like a bank balance. It is a *constructed* quantity. Under accrual accounting, which every listed company uses, revenue is recorded when it is **earned** (you delivered the goods) rather than when the cash arrives, and costs are recorded in the period they helped generate revenue rather than when you paid them.

That construction is not a scam; it is the only way to make a twelve-month slice of a continuing business meaningful. If you build bridges and get paid eighteen months after you start, cash accounting would show you making enormous losses and then enormous profits at random intervals. Accrual accounting smooths that into something that describes the actual economics.

But it means profit is an **opinion built on estimates**. How much of this year's shipments will actually be paid for? How long will this machine last? Is that software development cost an expense today or an asset to be used up over five years? Every one of those answers is a judgement, every judgement has a range, and management chooses where in the range to sit.

*Earnings manipulation* is the deliberate exploitation of that discretion to make reported profit differ from underlying economics — usually to hit a forecast, protect a share price, satisfy a debt covenant, or trigger a bonus. It runs on a spectrum from aggressive-but-legal, through misleading, into outright fraud. This post's sibling on [the gap between accrual accounting and cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) walks that spectrum in detail.

The key structural fact, and the reason a statistical model can work at all:

> Manipulation cannot create cash. It can only move profit between periods and between statement lines. Every dollar of profit invented today leaves a footprint somewhere else on the statements — and footprints are ratios.

### What an "index" is

Seven of the eight variables in the M-Score are *indices*. An index here is simply **this year's version of a ratio, divided by last year's version of the same ratio**.

If a company collected its receivables in 60 days last year and 60 days this year, that index is 60/60 = 1.00. If it now takes 90 days, the index is 90/60 = 1.50. An index of exactly 1.00 means "nothing changed". Above 1.00 means the thing grew; below means it shrank.

This design choice matters enormously, and it is the most-missed point about the model. **The M-Score does not measure whether a company's accounting is aggressive. It measures whether a company's accounting *changed*.** A business that has been booking revenue too early, consistently, for a decade, at a stable rate, produces indices near 1.00 and scores as ordinary. The model detects *deterioration*, not *level*.

### Where the weights come from

Beneish did not choose the eight coefficients by judgement. He fitted them.

The technique is a **probit model**. In plain terms: you have a set of companies, each labelled either "manipulator" or "not", and a set of measurements for each. Probit finds the weighted combination of measurements that best separates the two labelled groups — it asks "what mixture of these eight ratios, if I add them up, gives me the highest scores for the known manipulators and the lowest for everyone else?" The coefficients are the answer to that optimisation.

Two consequences follow immediately and both matter:

1. **The coefficients are descriptions of one historical sample, not laws of nature.** They describe how manipulators caught in a particular window differed from their peers. They are not a claim about causation, and they need not hold in a different decade, market or accounting regime.
2. **The output is a score on a probit scale, not a probability.** An M-Score of -1.78 corresponds to a fitted probability of about 0.0376 in Beneish's estimation sample. The score is monotonic in that probability — higher score, higher fitted likelihood — but the number itself is not a percentage and should never be read as one.

### The sample the model was built on

Beneish's paper — "The Detection of Earnings Manipulation", published in the *Financial Analysts Journal* in 1999 — describes a final sample of **74 firms that manipulated earnings and 2,332 Compustat non-manipulators matched by two-digit SIC industry**.

The manipulators came from two sources: 49 firms drawn from the SEC's Accounting and Auditing Enforcement Releases (AAERs) numbered 132 to 502 over 1987–1993, plus 25 more found through a LEXIS/NEXIS search of news media over the same window. Every one required *ex post* evidence of manipulation in the form of an earnings restatement. The estimation subsample covered 1982–1988 (50 manipulators against 1,708 controls), with 1989–1992 held back as an out-of-sample test.

Hold those numbers in mind. Seventy-four manipulators is a small sample by modern standards, the accounting is pre-Sarbanes-Oxley and pre-fair-value, and the identification method — "companies the SEC caught" — necessarily excludes the manipulators nobody ever caught. None of that makes the model useless. All of it constrains what you can honestly claim for it.

### The eight detectors at a glance

Here is the whole cast. Each row is a ratio, what it compares, and the specific accounting trick it exists to catch.

![A matrix listing all eight Beneish indices with what each compares between year t and year t-1 and the manipulation each is designed to catch](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-2.webp)

The rest of this article takes them in order of how much they actually contribute, starting with the ones that catch the act and moving to the ones that measure the motive.

## 1. DSRI — the receivables detector

**Days Sales in Receivables Index.** This is the first variable in the formula and, apart from accruals, the most direct evidence of the act itself.

### The intuition

Imagine a shop that sells \$1,000 of goods a day. At any moment, customers owe it money for goods already delivered — that is **accounts receivable**. If customers typically pay in 30 days, receivables sit at roughly \$30,000. That relationship is stable because it is physical: it reflects how the shop's customers actually behave.

Now suppose the shop's owner needs a better quarter. She ships goods to a distributor who did not order them, books the revenue, and privately agrees the distributor can send them back next quarter. Revenue rises. Profit rises. But no cash arrives, because there was never a real sale. The invoice sits in receivables.

Do that at scale and the physical relationship breaks: sales rise a bit, receivables rise a lot. The company is carrying more and more "sales" that no customer intends to pay for. That divergence is the signature of channel stuffing, bill-and-hold arrangements, and outright fictitious customers — the family of tricks covered in [revenue recognition games](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold).

![A two-line chart showing sales rising modestly while accounts receivable rise far faster, with the widening gap shaded and days-sales-in-receivables climbing from 62 to 118 days](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-3.webp)

### The formula

Beneish defines DSRI as the ratio of days sales in receivables in year `t` to the corresponding measure in year `t-1`:

$$
\text{DSRI} = \frac{\text{Receivables}_t / \text{Sales}_t}{\text{Receivables}_{t-1} / \text{Sales}_{t-1}}
$$

Note the elegance: because the same conversion factor (365) would appear in both numerator and denominator, you can work directly with the raw receivables-to-sales fraction and get exactly the same index. You do not need to convert to days at all — though it helps intuition to do so.

A DSRI above 1.00 means each dollar of sales is now supported by more unpaid invoice than it was last year.

#### Worked example: DSRI at a company stuffing the channel

Take the four-year illustration in the chart above. All figures here are **illustrative** — invented to show the arithmetic cleanly, not drawn from any real company.

A distributor's sales and receivables, indexed to 100 in Year 1:

| | Year 1 | Year 2 | Year 3 | Year 4 |
| --- | --- | --- | --- | --- |
| Sales (index) | 100 | 118 | 139 | 160 |
| Accounts receivable (index) | 100 | 130 | 178 | 250 |
| Days sales in receivables | 62 | 74 | 92 | 118 |

Sales grew a respectable 60% over three years. Receivables grew 150%. Translated into days, the company went from waiting 62 days to get paid to waiting 118 days.

The Year 4 index is:

$$
\text{DSRI}_{Y4} = \frac{118}{92} = 1.28
$$

Multiply by the published coefficient of 0.920 and this single variable contributes 0.920 × 1.28 = 1.18 to the score.

Is this proof of fraud? Absolutely not. A company that just signed a huge government customer with 120-day payment terms produces exactly this pattern honestly. A company expanding into a market with slower payment norms produces it honestly. What the index says is narrower and more useful: *the relationship between what this company books and what it collects has changed materially, and someone should find out why.*

**The intuition to keep:** revenue you invent has to live somewhere on the balance sheet, and receivables is the most convenient room in the house.

## 2. GMI and SGI — the motive pair

The next two variables are different in kind, and understanding why is the key to understanding the model's philosophy.

DSRI looks for evidence of the act. **GMI and SGI look for the conditions under which people commit the act.** They are pressure gauges.

### GMI: gross margin index

**Gross margin** is the share of each sales dollar left after the direct cost of producing the thing sold. Sell a widget for \$100 that cost \$60 in materials and labour, and your gross margin is \$40/\$100 = 40%. It is the cleanest single measure of whether a business has pricing power.

Beneish defines GMI as **last year's gross margin divided by this year's** — deliberately inverted, so that a *deteriorating* margin produces an index *above* 1.00:

$$
\text{GMI} = \frac{(\text{Sales}_{t-1} - \text{COGS}_{t-1}) / \text{Sales}_{t-1}}{(\text{Sales}_t - \text{COGS}_t) / \text{Sales}_t}
$$

Margins falling means the economics of the business are getting worse. A company whose real profitability is eroding, while the market still expects growth, is a company under pressure — and pressure is the first leg of the classic fraud triangle.

### SGI: sales growth index

$$
\text{SGI} = \frac{\text{Sales}_t}{\text{Sales}_{t-1}}
$$

That is all it is: this year's revenue over last year's. A company growing 30% has an SGI of 1.30.

Why would *growth* be suspicious? It is not, on its own, and this is where the model is most often misread. The reasoning is about consequences. High-growth companies trade on the expectation of continued growth. Their valuations, their financing, their employees' equity and their managements' compensation all assume the line keeps going up. The cost of a single missed quarter is therefore far higher for them than for a stable business — and controls, systems and accounting staff at fast-growing firms are frequently stretched. Beneish found manipulators disproportionately among high-growth firms; the coefficient encodes that empirical pattern.

![A two-by-two matrix with sales growth on the horizontal axis and gross margin trend on the vertical, marking fast growth on deteriorating margins as the highest-pressure quadrant](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-4.webp)

The interesting quadrant is the top right: **fast growth on worsening economics.** A company still posting big revenue increases while each dollar of revenue earns less than it used to. Growth is buying it a valuation it can only keep by continuing to grow, and the underlying business is telling it that growth is getting more expensive. That is where the incentive lives.

#### Worked example: GMI and SGI for a hardware maker

**Illustrative figures.** A device manufacturer reports:

| (\$ millions) | Year t-1 | Year t |
| --- | --- | --- |
| Sales | 800.0 | 1,040.0 |
| Cost of goods sold | 480.0 | 655.2 |
| Gross profit | 320.0 | 384.8 |
| Gross margin | 40.0% | 37.0% |

**SGI:** 1,040.0 / 800.0 = **1.300**. Revenue up 30%.

**Gross margin, year t-1:** (800.0 - 480.0) / 800.0 = 320.0 / 800.0 = 0.400, or 40.0%.
**Gross margin, year t:** (1,040.0 - 655.2) / 1,040.0 = 384.8 / 1,040.0 = 0.370, or 37.0%.

**GMI:** 0.400 / 0.370 = **1.081**.

Both above 1.00. The company grew 30% while its margin fell three percentage points. Contributions to the score: 0.892 × 1.300 = 1.160 from SGI, and 0.528 × 1.081 = 0.571 from GMI. Together, 1.73 — more than either accruals or receivables will contribute in most cases.

Notice how much of the M-Score can be driven by a company simply *growing fast with thinning margins*, which describes a large fraction of honest, competitive, scaling businesses. This is not a bug being pointed out; it is the model working exactly as fitted. It is also the mechanical origin of most false positives, which we will quantify later.

**The intuition to keep:** GMI and SGI do not accuse. They measure how much a company would lose by telling the truth.

## 3. AQI and DEPI — the capitalisation escape hatch

These two variables share a target: **costs that should have hit the income statement but were parked on the balance sheet instead.**

### The intuition

There are two ways to account for money you spend. **Expense** it, and it reduces this year's profit by the full amount. **Capitalise** it — record it as an asset — and it reduces profit only gradually, over years, as depreciation or amortisation.

For genuine long-lived assets, capitalising is correct: a factory should not destroy one year's earnings and then produce free output for twenty. But the boundary is a judgement, and moving a cost across it is the purest form of profit creation available to a manipulator, because it needs no fake customer and no accomplice. You simply decide that a cost was an investment. It is exactly [the WorldCom move](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move).

The consequence shows up in two places at once, which is why Beneish uses two variables.

![A before-and-after stacked asset chart showing the soft-asset share of total assets rising from 22 percent to 31 percent, alongside a depreciation rate falling from 12.0 percent to 9.6 percent](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-5.webp)

### AQI: asset quality index

"Asset quality" here has a precise, slightly counter-intuitive definition. Beneish measures it as **the share of total assets made up of non-current assets other than property, plant and equipment** — everything that is neither current (cash, receivables, inventory) nor hard physical plant. Goodwill, intangibles, capitalised development costs, deferred charges, other long-term assets. Call them **soft assets**: their value rests on an estimate rather than a market price or a legal claim on cash.

$$
\text{AQI} = \frac{1 - (\text{Current assets}_t + \text{PP\&E}_t) / \text{Total assets}_t}{1 - (\text{Current assets}_{t-1} + \text{PP\&E}_{t-1}) / \text{Total assets}_{t-1}}
$$

An AQI above 1.00 means a bigger fraction of the balance sheet is now made of judgement.

### DEPI: depreciation index

If you stretch the assumed useful life of your assets — decide the machines last fifteen years, not ten — your annual depreciation charge falls, and profit rises, with no operational change whatsoever.

Beneish measures the depreciation rate as depreciation divided by (depreciation + net PP&E), and defines DEPI as **last year's rate over this year's**, again inverted so that a *slowing* depreciation rate produces an index *above* 1.00:

$$
\text{DEPI} = \frac{\text{Depreciation}_{t-1} / (\text{Depreciation}_{t-1} + \text{Net PP\&E}_{t-1})}{\text{Depreciation}_t / (\text{Depreciation}_t + \text{Net PP\&E}_t)}
$$

#### Worked example: AQI and DEPI at a software company

**Illustrative figures**, matching the chart above.

A software firm capitalises a growing share of its development spending. Its soft assets — capitalised development costs, intangibles, deferred charges, other non-current assets — move from 22% of total assets to 31%:

$$
\text{AQI} = \frac{0.31}{0.22} = 1.41
$$

At the same time it revises the useful lives of its equipment upward. Its depreciation rate falls from 12.0% of gross plant to 9.6%:

$$
\text{DEPI} = \frac{12.0}{9.6} = 1.25
$$

Contributions: 0.404 × 1.41 = 0.570 from AQI, and 0.115 × 1.25 = 0.144 from DEPI.

Now look at the coefficients: 0.404 and 0.115. DEPI is the *weakest* variable in the entire model, and AQI is only middling. This is worth sitting with, because the capitalisation trick is one of the largest frauds in history and the model barely weights it. The reason is statistical, not moral: in Beneish's sample, the depreciation index simply did not separate manipulators from controls very well. Plenty of honest companies revise useful lives; plenty of manipulators did not bother. A variable earns weight by discriminating, not by being important in principle.

**The intuition to keep:** an expense that becomes an asset shows up twice — as soft assets rising and as depreciation slowing — but the model gives that pattern less credit than you would expect.

## 4. SGAI and LVGI — the two negative weights

Two variables carry negative coefficients, which confuses almost everyone on first reading. Both are small, and both are best understood as corrections rather than detectors.

### SGAI: selling, general and administrative expense index

$$
\text{SGAI} = \frac{\text{SG\&A}_t / \text{Sales}_t}{\text{SG\&A}_{t-1} / \text{Sales}_{t-1}}
$$

Overhead per dollar of sales. Beneish's prior was that rising overhead intensity signals a deteriorating business and therefore motive — the same logic as GMI. The fitted coefficient came out **negative** (-0.172): in the sample, manipulators had *lower* SGAI than controls, meaning their overhead grew more slowly relative to sales.

There is a reasonable story for that. A company inflating revenue books sales that carry no selling cost, no commission, no delivery expense — a fictitious sale is cheap to make. So the ratio of SG&A to sales *falls* precisely because the denominator is inflated. Suspiciously *efficient* overhead can be an artefact of fake revenue. But this is interpretation after the fact; what the model actually contains is a small negative weight that the data produced.

### LVGI: leverage index

$$
\text{LVGI} = \frac{\text{Total debt}_t / \text{Total assets}_t}{\text{Total debt}_{t-1} / \text{Total assets}_{t-1}}
$$

Rising leverage tightens debt covenants — the contractual promises to lenders about ratios a borrower must maintain — and covenant pressure is a documented motive for manipulation. Yet the coefficient is again negative (-0.327): in the sample, manipulators' leverage rose *less* than controls'.

Once more there is a plausible story. Manipulation inflates assets and equity, which flatters the leverage ratio by expanding the denominator. And a company with a rising share price can raise equity instead of debt.

Note also that in Beneish's reported probit results, the depreciation index, the SG&A index and the leverage index were **not statistically significant**, while days sales in receivables (coefficient 0.920, t = 6.02), gross margin, asset quality (coefficient 0.404, t = 3.20), sales growth and total accruals were. Three of the eight variables in the published model are, on the paper's own evidence, weak contributors retained for completeness. That is a legitimate modelling choice, and it is also a reason not to over-interpret any single index.

**The intuition to keep:** a negative coefficient does not mean "this is good". It means that in one historical sample, this ratio moved the other way for manipulators — and two of the three negative or weak variables did not clear statistical significance at all.

## 5. TATA — the variable that does most of the work

**Total Accruals to Total Assets.** Coefficient **4.679**. Everything else in the model is a supporting witness.

### The intuition

Return to the foundational identity. Profit is cash plus accruals. Rearranged:

$$
\text{Accruals} = \text{Net income} - \text{Operating cash flow}
$$

Accruals are the part of reported profit that has not turned into money. Some accruals are entirely normal — a business that grows must fund more inventory and more receivables, so a healthy growing company routinely reports profit ahead of cash. But over any meaningful stretch, cash must catch up. A sale is only real when the money arrives.

Manipulation of the earnings statement almost always increases accruals, because manipulation *cannot create cash*. Whatever the specific trick — early revenue, deferred cost, understated reserve, capitalised expense — the effect is profit that is not accompanied by money. TATA measures precisely the size of that wedge, scaled by the company's assets so a \$100 million gap means something different for a \$500 million company than a \$50 billion one.

![A two-line chart showing net income climbing from 100 to 205 while cash flow from operations falls from 100 to 88, with the widening accrual gap shaded](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-7.webp)

### Two ways to compute it, and why they differ

Beneish's paper defines total accruals from the **balance sheet**: the change in working capital accounts other than cash, less depreciation. In practice that expands to:

$$
\text{TA} = \Delta\text{CA} - \Delta\text{Cash} - (\Delta\text{CL} - \Delta\text{CMLTD} - \Delta\text{TaxPayable}) - \text{Depreciation}
$$

where CA is current assets, CL is current liabilities, and CMLTD is the current maturities of long-term debt. Divide by total assets in year `t` for TATA.

Most modern implementations use the simpler **cash-flow statement** version:

$$
\text{TATA} = \frac{\text{Income from continuing operations}_t - \text{Cash flow from operations}_t}{\text{Total assets}_t}
$$

The two do not give identical answers — the cash-flow version captures accruals the balance-sheet version misses, such as those arising from acquisitions. They are usually close. What matters is that you **state which one you used**, because a published M-Score without that disclosure is not reproducible. In the worked calculation later in this post I compute both for the same company so you can see the size of the difference.

#### Worked example: TATA when profit and cash separate

**Illustrative figures**, matching the chart above.

A company reports four years of results, in \$ millions:

| | Year 1 | Year 2 | Year 3 | Year 4 |
| --- | --- | --- | --- | --- |
| Net income | 100 | 130 | 165 | 205 |
| Cash flow from operations | 100 | 104 | 96 | 88 |
| Accruals (NI - CFO) | 0 | 26 | 69 | 117 |

By Year 4 the company reports it doubled its profit. Its operating cash flow has *fallen* by 12%. Every extra dollar of reported profit, and then some, is an accrual.

With total assets of \$1,300 million:

$$
\text{TATA}_{Y4} = \frac{205 - 88}{1{,}300} = \frac{117}{1{,}300} = 0.090
$$

Contribution to the score: 4.679 × 0.090 = **0.421** from this one variable.

To feel the weight, compare like for like. Getting a 0.421 contribution out of DSRI would require an index of 0.421 / 0.920 = 0.46 — impossible, since it would mean receivables days more than halving in the wrong direction. Getting it out of DEPI would require an index of 3.66, meaning the depreciation rate fell by nearly three quarters in a single year. TATA reaches the same contribution from a company whose accruals are 9% of assets, which is unusual but entirely achievable.

**The intuition to keep:** if you only ever look at one thing on a set of financial statements, look at the gap between net income and operating cash flow. Beneish's model, by the arithmetic of its own coefficients, mostly agrees.

## 6. The formula, the weights and the threshold

Now we can assemble the whole thing.

### The equation

Beneish (1999) reports the eight-variable model as:

$$
\begin{aligned}
M = -4.840 &+ 0.920 \times \text{DSRI} + 0.528 \times \text{GMI} + 0.404 \times \text{AQI} \\
&+ 0.892 \times \text{SGI} + 0.115 \times \text{DEPI} - 0.172 \times \text{SGAI} \\
&+ 4.679 \times \text{TATA} - 0.327 \times \text{LVGI}
\end{aligned}
$$

Eight inputs, eight coefficients, one intercept, one output.

![A horizontal bar chart of the eight Beneish coefficients, with TATA at 4.679 dwarfing DSRI at 0.920 and SGI at 0.892, and LVGI and SGAI extending to the left of zero](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-6.webp)

Seeing the coefficients drawn to scale changes how you read the model. **TATA is not one of eight variables. It is the variable, with seven modifiers.** Its coefficient is 5.1 times DSRI's and more than 40 times DEPI's.

There is a subtlety that makes this less lopsided than it first appears. The seven indices are all ratios centred near 1.00, so they contribute roughly their coefficient's worth to every score, moving up or down by tenths. TATA is a fraction of assets centred near zero, typically running from about -0.10 to +0.10. So the *typical range of contribution* is more balanced than the coefficients alone suggest: TATA's 4.679 applied to a swing of 0.20 gives about 0.94 of range, comparable to DSRI's 0.920 applied to a swing of 1.0. What remains true is that TATA is the variable most able to move a score decisively, and the only one where a single unusual reading can push a company over the line on its own.

#### Worked example: the score of a company where nothing changed

Here is a calculation that teaches more about the M-Score than any real company can, and it uses nothing but the published coefficients.

Imagine a company whose second year is a perfect copy of its first. Same margin, same growth, same receivable days, same asset mix, same depreciation rate, same overhead ratio, same leverage. Every index is exactly 1.00. And suppose its profit converts perfectly to cash, so accruals are zero and TATA = 0.

$$
\begin{aligned}
M &= -4.840 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 + 0 - 0.327 \\
&= -4.840 + 2.360 \\
&= \mathbf{-2.480}
\end{aligned}
$$

**A company at complete standstill scores -2.48.** That is the model's zero point, and it is enormously useful:

- The distance from the standstill score to the -1.78 flag line is **0.70**. That is the total "suspicion budget" a company must consume across all eight variables to get flagged.
- The distance to the wider -2.22 screen is only **0.26**.

Now ask what it takes to spend 0.70 using a single variable:

| Variable acting alone | Index needed to reach -1.78 | What that would mean for the business |
| --- | --- | --- |
| TATA | 0.70 / 4.679 = **0.150** | Accruals equal to 15% of total assets |
| DSRI | 1 + 0.70 / 0.920 = **1.761** | Receivable days up 76% in one year |
| SGI | 1 + 0.70 / 0.892 = **1.785** | Revenue up 78.5% in one year |
| GMI | 1 + 0.70 / 0.528 = **2.326** | Gross margin cut by more than half |
| AQI | 1 + 0.70 / 0.404 = **2.733** | Soft-asset share nearly tripling |
| DEPI | 1 + 0.70 / 0.115 = **7.087** | Depreciation rate falling ~86% |

Read the third row again. **A company that grows revenue 78.5% in a year, changing nothing else, is flagged by the Beneish M-Score.** Not because it did anything wrong — because Beneish's manipulators grew fast, and the model faithfully encodes that.

**The intuition to keep:** -2.48 is the baseline, 0.70 is the budget, and a hyper-growth company can spend the whole budget honestly.

### The threshold, and a widespread mix-up

Beneish derives cutoffs from the **relative cost of the two kinds of error**: how much worse is it to miss a manipulator than to wrongly flag an innocent company? Different answers give different thresholds. In his Table 5 results for the unweighted probit model:

- At a relative error cost of **10:1**, firms are classified as manipulators when the fitted probability exceeds 0.0685, which corresponds to a score greater than **-1.49**.
- At relative error costs of **20:1 and 30:1**, the cutoff is a fitted probability above 0.0376, corresponding to a score greater than **-1.78**.

That is where the famous -1.78 comes from. It is not a natural constant; it is the answer to "assume missing a fraud is twenty to thirty times worse than a false alarm".

The other number you will see everywhere is **-2.22**, and it deserves a direct answer because it is repeated constantly, including by sources that are otherwise careful.

**The value -2.22 does not appear anywhere in Beneish (1999).** The cutoffs that paper reports for its eight-variable probit model are -1.49 and -1.78, corresponding to the relative error costs above. It is a natural guess that -2.22 is simply a third cutoff from the same table under some other assumed cost of errors — a legitimate sibling of -1.49 and -1.78. It is not. There is no such row.

Where -2.22 comes from is Beneish's *earlier and different* model: the five-variable specification in "Detecting GAAP violation: implications for assessing earnings management among firms with extreme financial performance", published in the *Journal of Accounting and Public Policy* in 1997. A different model, fitted on a different sample, with a different number of inputs, has a different intercept and therefore a different natural scale — so its threshold is not comparable to the eight-variable model's.

That is the conflation. Two numbers circulate because there are genuinely **two models**, and the thresholds got separated from the equations they belong to. The practical rule:

> If you are using the eight-variable equation with the -4.840 intercept, the threshold that belongs to it is **-1.78**. Applying -2.22 to an eight-variable score is mixing one model's output with another model's cutoff — and because -2.22 is the lower number, doing so silently widens the net and inflates your false-positive count.

And here is the mix-up, which appears in an alarming number of otherwise careful write-ups. Because -2.22 is a *smaller* number than -1.78, and the rule is "flag when M is greater than the cutoff", screening at -2.22 flags **more** companies, not fewer.

> A cutoff of -2.22 is a wider net, not a finer filter. It catches more manipulators and drags in far more innocent companies. If a source tells you -2.22 is "stricter" in the sense of producing fewer false alarms, that source has the inequality backwards.

![A number line from -5 to 0 showing a tall green distribution of ordinary companies overlapping a small red distribution of manipulators, with cutoffs marked at -2.22 and -1.78 and the false-positive and miss regions shaded](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-8.webp)

The picture makes the tradeoff unavoidable. The two populations overlap. Wherever you put the line, you are choosing a mix of two errors, never eliminating them:

- **Type I error — a miss.** A manipulator scores below the cutoff and passes. Left of the line, under the red curve.
- **Type II error — a false positive.** An honest company scores above the cutoff and gets flagged. Right of the line, under the green curve.

Move the line left and you catch more frauds and accuse more innocents. Move it right and you accuse fewer innocents and miss more frauds. There is no setting that does both, because the distributions genuinely overlap — some honest companies really do have manipulator-shaped financials.

## 7. A complete worked calculation, end to end

Everything so far has been one variable at a time. Now we do the whole thing from statement lines, the way you would in a spreadsheet.

**The company below is illustrative.** Meridian Systems is invented, and its figures were chosen to make the arithmetic clean and the outcome instructive. The *coefficients and threshold* are Beneish's published ones; the *company* is not real.

### Step 1: pull the statement lines

Twelve lines from two consecutive annual reports, in \$ millions:

| Line item | Year t-1 | Year t |
| --- | --- | --- |
| Total revenues | 800.0 | 1,040.0 |
| Cost of goods sold | 480.0 | 655.2 |
| Selling, general & administrative expense | 152.0 | 208.0 |
| Depreciation & amortisation | 36.0 | 33.0 |
| Net income | 60.0 | 92.0 |
| Cash and equivalents | 60.0 | 41.0 |
| Receivables, net | 137.0 | 228.8 |
| Total current assets | 450.0 | 585.0 |
| Property, plant & equipment, net | 300.0 | 330.0 |
| Total assets | 1,000.0 | 1,300.0 |
| Total current liabilities | 220.0 | 262.0 |
| Total debt (short-term + long-term) | 380.0 | 546.0 |
| Cash flow from operations | 71.0 | 15.0 |

At a glance this looks like a strong year: revenue up 30%, profit up 53%. That is exactly the sort of year a screen exists to interrogate.

### Step 2: compute the eight indices

**DSRI.** Receivables over sales, this year against last:

- Year t-1: 137.0 / 800.0 = 0.171250 (equivalently 62.5 days)
- Year t: 228.8 / 1,040.0 = 0.220000 (equivalently 80.3 days)
- DSRI = 0.220000 / 0.171250 = **1.285**

**GMI.** Last year's margin over this year's:

- Year t-1: (800.0 - 480.0) / 800.0 = 0.400
- Year t: (1,040.0 - 655.2) / 1,040.0 = 0.370
- GMI = 0.400 / 0.370 = **1.081**

**AQI.** The soft-asset share, this year against last:

- Year t-1: 1 - (450.0 + 300.0) / 1,000.0 = 1 - 0.750 = 0.250
- Year t: 1 - (585.0 + 330.0) / 1,300.0 = 1 - 0.703846 = 0.296154
- AQI = 0.296154 / 0.250 = **1.185**

**SGI.** Sales growth:

- SGI = 1,040.0 / 800.0 = **1.300**

**DEPI.** Depreciation rate last year over this year:

- Year t-1: 36.0 / (36.0 + 300.0) = 36.0 / 336.0 = 0.107143
- Year t: 33.0 / (33.0 + 330.0) = 33.0 / 363.0 = 0.090909
- DEPI = 0.107143 / 0.090909 = **1.179**

**SGAI.** Overhead intensity, this year against last:

- Year t-1: 152.0 / 800.0 = 0.190
- Year t: 208.0 / 1,040.0 = 0.200
- SGAI = 0.200 / 0.190 = **1.053**

**LVGI.** Leverage, this year against last:

- Year t-1: 380.0 / 1,000.0 = 0.380
- Year t: 546.0 / 1,300.0 = 0.420
- LVGI = 0.420 / 0.380 = **1.105**

**TATA**, cash-flow version:

- Accruals = 92.0 - 15.0 = 77.0
- TATA = 77.0 / 1,300.0 = **0.0592**

**TATA**, balance-sheet version, for comparison. Suppose the current maturities of long-term debt were 20.0 and 24.0, and income taxes payable 12.0 and 15.0:

- Change in current assets: 585.0 - 450.0 = 135.0
- Change in cash: 41.0 - 60.0 = -19.0
- Change in current liabilities: 262.0 - 220.0 = 42.0
- Change in current maturities of LTD: 24.0 - 20.0 = 4.0
- Change in taxes payable: 15.0 - 12.0 = 3.0
- Total accruals = (135.0 - (-19.0)) - (42.0 - 4.0 - 3.0) - 33.0 = 154.0 - 35.0 - 33.0 = 86.0
- TATA = 86.0 / 1,300.0 = **0.0662**

The two methods give 0.0592 and 0.0662 — a gap of 0.007, which flows through to a difference of 4.679 × 0.007 = 0.033 in the final score. Small, but real, and a reminder that two analysts can publish different M-Scores for the same company and both be right about their own method. We use the cash-flow figure below.

#### Worked example: assembling Meridian Systems' M-Score

Multiply each index by its coefficient:

| Variable | Index | Coefficient | Contribution |
| --- | --- | --- | --- |
| DSRI | 1.285 | 0.920 | +1.182 |
| GMI | 1.081 | 0.528 | +0.571 |
| AQI | 1.185 | 0.404 | +0.479 |
| SGI | 1.300 | 0.892 | +1.160 |
| DEPI | 1.179 | 0.115 | +0.136 |
| SGAI | 1.053 | -0.172 | -0.181 |
| TATA | 0.0592 | 4.679 | +0.277 |
| LVGI | 1.105 | -0.327 | -0.361 |
| | | **Sum** | **+3.261** |
| | | Intercept | -4.840 |
| | | **M-Score** | **-1.579** |

$$
M = -4.840 + 3.261 = -1.579
$$

**-1.58 is greater than -1.78, so Meridian is flagged.**

### Step 3: read the result properly

The verdict line is the least interesting output. The *decomposition* is where the value is.

Meridian's score is 0.90 above the standstill baseline of -2.48. Where did that 0.90 come from? Ranking the contributions against what a standstill company would contribute (its coefficient, or zero for TATA):

| Variable | Excess over standstill |
| --- | --- |
| SGI | +0.268 |
| TATA | +0.277 |
| DSRI | +0.262 |
| AQI | +0.075 |
| GMI | +0.043 |
| DEPI | +0.021 |
| SGAI | -0.009 |
| LVGI | -0.034 |

Three variables — sales growth, accruals and receivables — supply almost 90% of the movement. That is a specific, checkable story: *this company grew fast, its receivables grew faster than its sales, and its profit did not turn into cash.*

That story is exactly what you would take to the footnotes. Did revenue recognition policy change? Who are the new customers and what are their terms? Why did operating cash flow fall from \$71.0 million to \$15.0 million while profit rose by \$32.0 million? The [cash conversion cycle](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) and the [footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) are where those answers live.

Equally, a plain and innocent version of this story exists: Meridian won a large enterprise contract with 90-day terms, built inventory to service it, and will collect in the first quarter of the next year. Both stories produce the same M-Score. **The score cannot distinguish between them, and it does not claim to.** It says: here is a company whose numbers moved the way manipulators' numbers moved. Go and look.

**The intuition to keep:** the M-Score's job is to tell you where to spend your reading time, not to tell you what you will find.

## 8. What the model gets right

It would be easy to spend the rest of this post on limitations. That would be unfair, because the model does something genuinely difficult.

**The in-sample performance.** Beneish reports that across the cutoffs examined in the estimation sample, the percentage of correctly classified manipulators ranges from **58% to 76%**, while the percentage of incorrectly classified non-manipulators ranges from **7.6% to 17.5%**. At the -1.49 cutoff (10:1 relative error cost) the paper reports 58% of manipulators identified with 7.6% of non-manipulators misclassified. At the -1.78 cutoff (20:1 and 30:1) it reports **74% of manipulators identified with 13.8% of non-manipulators misclassified**.

That "76% and 17.5%" pairing you see quoted everywhere is worth pausing on. Both numbers are from Beneish, but they are the **endpoints of a range**, and specifically the end with the *highest* false-positive rate. Quoting the best detection rate alongside its worst-case false-positive partner as if they were a single reported operating point is a small but persistent distortion in the secondary literature.

**The out-of-sample performance.** More impressive, and more honestly reported. In the 1989–1992 holdout sample, the paper reports correctly classified manipulators ranging from **37.5% to 56%** and misclassified non-manipulators from **3.5% to 9.1%**. At the -1.78 cutoff: **50% of manipulators identified, 7.2% of non-manipulators misclassified**.

Detection roughly halves out of sample. That is what honest out-of-sample testing usually looks like, and it is to Beneish's credit that the paper reports it plainly.

**The long-run evidence.** Fourteen years later, Beneish, Lee and Nichols published "Earnings Manipulation and Expected Returns" in the *Financial Analysts Journal* (2013), testing the model far outside its estimation window. Their headline findings: companies with a higher probability of manipulation earn **lower** subsequent returns, and the effect survives sorting on size, book-to-market, momentum, accruals and short interest. The predictive power comes from the model's ability to forecast changes in accruals, and it is most pronounced among low-accrual stocks — that is, among companies that *look* like they have high earnings quality on the standard measure. They also report that in an out-of-sample test the model identified, in advance of public disclosure, **71% of the prominent accounting fraud cases** that surfaced after the original estimation period.

That last finding is the strongest case for the model. Not that it is accurate in the everyday sense, but that it is **early**. It works on information already public, in the year before anyone knew.

## 9. False positives and the base-rate problem

This is the section to read twice. Almost every misuse of the M-Score traces to skipping it.

### The arithmetic of rare events

Earnings manipulation serious enough to draw an enforcement action is uncommon. Beneish's own construction — 74 manipulators against 2,332 controls — implies roughly 3% in a deliberately matched sample, and matched samples overstate base rates by design. In a broad market screen, the true rate of serious manipulation in any given year is plausibly around 1%, perhaps lower. Nobody knows it exactly, because the denominator includes the frauds nobody ever caught.

Whatever the exact figure, it is *small*. And when the thing you are hunting is rare, a test's false-positive rate matters far more than its detection rate.

![A funnel showing 1,000 screened companies splitting into 10 manipulators and 990 clean firms, producing 8 true flags and 173 false flags for a total of 181 flags](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-9.webp)

#### Worked example: what a flag list actually contains

**This arithmetic is illustrative.** The detection and false-positive rates are Beneish's reported figures; the 1% base rate is an assumption, chosen because nobody can observe the true rate.

Screen 1,000 companies. Assume 1% are genuinely manipulating, so 10 manipulators and 990 clean companies. Apply the model at Beneish's most permissive reported operating point — 76% of manipulators caught, 17.5% of non-manipulators misclassified:

- Manipulators caught: 10 × 0.76 = 7.6, call it **8**
- Manipulators missed: **2**
- Clean companies falsely flagged: 990 × 0.175 = 173.25, call it **173**
- Clean companies correctly cleared: **817**

Total flagged: 8 + 173 = **181 companies**. Of those, 8 are real.

$$
\text{Precision} = \frac{8}{181} = 4.4\%
$$

**About one flag in twenty-three is a genuine manipulator.** Ninety-six percent of your list is innocent.

Now redo it at the -1.78 operating point the paper actually reports for the estimation sample (74% and 13.8%):

- Caught: 10 × 0.74 = 7.4, call it **7**
- Falsely flagged: 990 × 0.138 = 136.6, call it **137**
- Total flagged: **144**, precision 7/144 = **4.9%** — about 1 in 21.

And at the holdout operating point (50% and 7.2%):

- Caught: **5**
- Falsely flagged: 990 × 0.072 = 71.3, call it **71**
- Total flagged: **76**, precision 5/76 = **6.6%** — about 1 in 15.

Look carefully at that last one. The holdout model catches *fewer* manipulators — half instead of three quarters — and yet produces a **better** flag list, because its false-positive rate fell faster than its detection rate. When the base rate is low, reducing false positives is worth more than increasing detection. This is the opposite of the intuition most people bring.

**The intuition to keep:** with a rare event, even a good model produces a list that is overwhelmingly innocent. That is not the model failing. It is what a screen *is*.

### What follows from this

Three practical consequences:

1. **A high M-Score is a research trigger, never a conclusion.** Its correct use is to reorder a reading queue. Treating it as a verdict is not aggressive analysis; it is a misreading of conditional probability.
2. **Never publish a company's M-Score as an accusation.** The base-rate arithmetic above says you would be wrong more than nine times in ten. There are also obvious legal and ethical reasons.
3. **Combine it with an independent signal.** The false positives of a ratio-based model — fast growth, working-capital build, an acquisition — are largely *different* from the false positives of, say, an auditor-change screen, a short-interest screen, or [related-party transaction analysis](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing). Two weakly correlated screens agreeing raises precision far more than either one tightened.

## 10. What the M-Score cannot see

Every model has a domain. The M-Score's is: *effects that show up as year-over-year changes in eight ratios computed from consolidated financial statements.* Anything outside that domain is invisible to it, no matter how large.

![A diagram showing the eight ratios the model reads above a dashed limit line, with six categories of fraud below it in a red zone marked invisible to the score](/imgs/blogs/the-beneish-m-score-detecting-earnings-manipulation-10.webp)

**Off-balance-sheet vehicles.** If debt and losses are parked in entities that never consolidate, they never reach the ratios. Total assets, leverage and accruals all look fine because the bad things are legally somewhere else. This is not a hypothetical limitation — it is precisely the mechanism that destroyed Enron, and the model that flagged Enron could not see it. See [off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities).

**Mark-to-market revenue from unobservable prices.** When a company books today the entire estimated value of a twenty-year contract, that revenue is real to every ratio in the model. Sales growth rises, margins may improve, receivables may not move at all. The M-Score has no way to ask whether the estimate was reasonable.

**Related-party transactions.** If a company sells to an entity it controls, the sale looks like a sale. Revenue, receivables and margin all behave normally. The fact that the counterparty is not independent lives in the footnotes, not the ratios — which is why [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) can pass a purely quantitative screen.

**Fabricated cash.** This is the model's sharpest blind spot, and it is close to an inversion. A company that invents a bank balance reports *high* cash, *strong* operating cash flow and therefore *low* accruals. TATA — the model's heaviest variable — moves in the reassuring direction. Fake cash makes a company look cleaner to the M-Score.

**First-year fraud, and companies without a comparable prior year.** Every index needs a year `t-1`. A newly listed company, a company that just completed a transformative acquisition, one that changed fiscal year end, or one that restated its comparatives, produces indices that are undefined or meaningless. Recent IPOs — a population with real fraud risk — are structurally hard for the model to assess.

**Disclosure and footnote fraud.** Undisclosed contingent liabilities, hidden guarantees, misrepresented customer concentration, an undisclosed regulatory investigation. None of these is a ratio. See [hidden liabilities](/blog/trading/forensic-accounting/hidden-liabilities-leases-guarantees-and-contingencies).

**And the biggest one of all: the model is old.** The coefficients were fitted on 1982–1992 US data, before Sarbanes-Oxley, before the expensing of stock compensation, before fair-value accounting expanded, before software and intangible-heavy businesses dominated market capitalisation. A modern software company legitimately has a soft-asset-heavy balance sheet, high sales growth and low physical depreciation. The model was not fitted on such companies and reads their normal shape as unusual.

## Common misconceptions

**"A high M-Score means the company is committing fraud."** No. It means the company's ratio changes resemble those of firms later found to have manipulated. Given a low base rate, most flagged companies are innocent — by the arithmetic above, roughly nineteen or twenty in every twenty-one.

**"-2.22 is the strict threshold and -1.78 the loose one."** Wrong twice over. First, the two numbers belong to two *different models*: -1.78 is a cutoff Beneish (1999) reports for the eight-variable equation, while -2.22 belongs to the earlier five-variable specification of Beneish (1997). The value -2.22 does not appear in the 1999 paper at all, which reports -1.49 and -1.78 for relative error costs of 10:1 and 20:1/30:1. Second, even taken at face value the "strict" label is backwards: the rule flags companies scoring *above* the cutoff, so the lower number would flag *more* companies, not fewer.

**"The M-Score is a probability of manipulation."** It is a probit score. It maps monotonically to a fitted probability — -1.78 corresponds to about 0.0376 in the estimation sample — but the score itself is not a percentage, and averaging M-Scores across companies has no probabilistic meaning.

**"A low M-Score means the accounts are clean."** It means nothing changed much year over year in eight specific ratios. A company that has been consistently aggressive for a decade scores well. A company with fabricated cash scores *better* than it should. And in the holdout sample, the model missed about half of the manipulators at the -1.78 cutoff.

**"Eight variables means eight independent pieces of evidence."** In the paper's reported probit results, the depreciation index, the SG&A index and the leverage index were not statistically significant. The model's discriminating power sits mainly in accruals, receivables, sales growth, gross margin and asset quality.

**"You can compare M-Scores across sources."** Only if they used the same TATA definition, the same treatment of missing data, the same handling of negative denominators, and the same fiscal alignment. The Meridian example above produced 0.0592 or 0.0662 for TATA depending on method — a 0.03 swing in the final score from a definitional choice alone. Two published M-Scores for the same company that differ by a tenth are usually a methodology difference, not a disagreement about facts.

## How it shows up in real markets

### 1. Enron, and the students who called it in 1998

This is the case that made the model famous, and the details matter more than the legend.

In **May 1998**, a team of students in Professor Charles M.C. Lee's financial statement analysis course at Cornell University's Johnson Graduate School of Management chose Enron as the subject of their term project. Enron was then one of the most admired companies in America, trading at around **\$48 a share**.

Working only from public financial statements, the team returned a **"Sell"** recommendation. Their report stated that "the 8-variable Beneish model shows that Enron may be manipulating its earnings". Notably — and this is usually omitted — the sell call rested primarily on valuation: they estimated intrinsic value at roughly **\$35 a share** against a market price near \$48 (figures as recounted in published accounts of the student report, 1998). The M-Score was corroborating evidence, not the thesis.

The recommendation was, as these things go, ignored. Enron's share price continued to rise for more than two years. The company filed for Chapter 11 bankruptcy protection on **2 December 2001**, weeks after restating its financial statements for 1997 through 2000 in November 2001.

Two lessons sit in this story, pulling in opposite directions, and honest use of the model requires holding both.

The first is genuinely remarkable: a purely mechanical screen, run by students on published data, produced a warning **more than three years before** the collapse, at a time when the professional analyst community was overwhelmingly positive. Ratios saw something reputation did not.

The second is sobering. **The M-Score did not detect what actually killed Enron.** Enron's central mechanisms were off-balance-sheet special purpose entities that hid debt and losses outside the consolidated statements, and mark-to-market accounting on long-dated energy contracts that booked estimated future profits as current revenue. Neither is visible to any of the eight variables. What the model saw were the *shadows* — growing receivables, expanding soft assets, profit running ahead of cash — cast by a structure it could not perceive directly. It was right, and it was right for reasons it could not articulate. Our [full account of the Enron collapse](/blog/trading/finance/enron-2001-accounting-fraud) covers the mechanisms in detail.

That is worth generalising: **a screen can be correct without being diagnostic.** The score told the students where to look; the looking is what would have found the SPEs, and only the footnotes could have supported that.

### 2. WorldCom, and the trick the model under-weights

WorldCom is the textbook case of capitalising costs to inflate profit. The company treated ordinary "line costs" — fees paid to other carriers to use their networks, an operating expense by any reading — as capital expenditure, moving them from the income statement to the balance sheet where they would be depreciated over years instead of hitting profit at once.

WorldCom announced in **June 2002** that it had improperly capitalised expenses, initially reported at approximately **\$3.8 billion**, a figure that grew substantially as the investigation continued. The SEC filed fraud charges within days, and the company filed for bankruptcy in July 2002.

Mechanically, this is exactly what AQI and DEPI are built to catch: costs moving into non-current assets, and a depreciation charge that no longer matches the underlying economics. It should also have inflated accruals, since capitalised costs are cash out with no expense recognised, widening the gap between profit and operating cash flow.

But note the weights the model assigns to those detectors. Look back at the coefficient chart in section 6: AQI sits at 0.404 and DEPI at 0.115, the second-weakest and weakest bars in the whole equation. The trick WorldCom used is one the M-Score is *designed* to notice and *statistically* under-armed against, because in Beneish's 1982–1992 sample those particular indices did not separate the groups well. It is a clean illustration of the difference between what a model is designed to notice and what it was empirically able to weight.

### 3. Wirecard, and the blind spot that cannot be patched

Wirecard, the German payments company, collapsed after its auditor refused to sign off on its accounts because roughly **€1.9 billion** of cash purportedly held in trustee accounts in Asia could not be verified to exist (announced and widely reported in June 2020). The company filed for insolvency later that month, and its former chief executive was subsequently prosecuted.

Now apply the M-Score logic. If a company reports cash that does not exist, and reports the operating cash flow that supposedly produced it, then:

- Operating cash flow is overstated, so **accruals are understated**, so TATA — the model's heaviest variable — reads *low*.
- Total assets are inflated by the phantom cash, so leverage ratios look *better*.
- The soft-asset share of the balance sheet falls, because cash is the hardest asset there is, so AQI reads *low*.

The fraud does not merely evade the model. It pushes three of the eight variables in the reassuring direction. No recalibration fixes this, because the model's core assumption — that reported cash flow is a harder number than reported profit — is exactly the assumption the fraud violates. Our [account of the Wirecard fraud](/blog/trading/finance/wirecard-the-german-fintech-fraud) traces how the phantom balances survived scrutiny for years.

The general lesson: **when a model's most heavily weighted input is the thing being falsified, the model does not degrade gracefully. It inverts.**

### 4. The everyday case: the flagged company that was fine

For every named fraud there are dozens of companies that screen badly and are simply doing something ordinary.

A retailer opening 200 stores in a year builds inventory and receivables ahead of the revenue they will generate: accruals rise, DSRI may rise, SGI is high by construction. A manufacturer that completes a large acquisition sees goodwill and intangibles jump — AQI spikes — while total assets grow faster than depreciation, lifting DEPI. A software company shifting from perpetual licences to subscriptions sees receivables and deferred revenue behave in ways no 1990s-fitted model anticipated.

None of these companies is doing anything wrong. All of them can score above -1.78. This is the population that fills the 173 false positives in the worked example above, and it is why the correct response to a flag is a question, not a position.

### 5. How practitioners actually use it

In institutional practice the M-Score is rarely used as a standalone signal, and almost never at its nominal threshold. The common patterns:

- **As a ranking, not a classifier.** Instead of "is this company above -1.78", the question becomes "where does this company sit in the distribution of M-Scores across its sector this year". Relative position handles industry effects that the raw score does not.
- **As one factor among several.** The Beneish, Lee and Nichols (2013) result — that high-probability-of-manipulation firms earn lower subsequent returns — is a factor-investing finding, expressed across a portfolio, not a stock-picking rule.
- **As a triage tool for a reading queue.** For an analyst covering forty companies, the score's honest job is to decide which five annual reports get read line by line this quarter.
- **Watching the trend in the score,** not its level. A company whose M-Score moves from -3.0 to -1.9 over two years has changed, and change is what the model is actually built to detect.

## When this matters to you

If you invest in individual companies, the M-Score is worth knowing for one reason above all others: it forces you to look at the gap between reported profit and operating cash flow, and it tells you, through a coefficient of 4.679 fitted on real fraud cases, how much that gap matters. You do not need to compute the full score to get most of that benefit. Pull two numbers from any annual report — net income and cash flow from operations — and look at how they have moved together over five years. If profit is climbing while cash is not, you have found the question worth asking, and everything else in this article is elaboration.

If you want to compute the whole thing, do it. It is thirteen line items and about twenty minutes in a spreadsheet, and the worked example above is a complete template. But compute it for a company you already follow, so you have the context to interpret it, and compute the *decomposition* rather than only the total — the contribution table is where the analysis lives.

What you should not do is treat the output as a verdict. The base-rate arithmetic is not a technicality. If you screen a broad universe and act on the flags, the great majority of what you act on will be ordinary companies having an unusual year. The model's honest offer is narrower and still valuable: *here is where to spend your attention.*

This is educational material about a published screening model, not investment advice, and a high M-Score is not evidence of wrongdoing by any company.

If you want to go deeper into the mechanisms the score is reaching for, the natural next reads on this blog are [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) for the identity underneath TATA, [inventory and receivables inflation](/blog/trading/forensic-accounting/inventory-and-receivables-inflation-the-classic-red-flag) for what DSRI is chasing, and [quality of earnings: accruals, one-offs and red flags](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) for how this fits into a full research process.

## Sources & further reading

**The primary sources behind the numbers in this post:**

- Messod D. Beneish, ["The Detection of Earnings Manipulation"](https://www.calctopia.com/papers/beneish1999.pdf), *Financial Analysts Journal*, 1999. Source for: the eight-variable equation and all coefficients; the definitions of DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI and TATA; the sample of 74 manipulators and 2,332 Compustat non-manipulators matched by two-digit SIC; the AAER #132–502 and LEXIS/NEXIS identification method; the 1982–1988 estimation and 1989–1992 holdout split; the -1.49 and -1.78 cutoffs and their 10:1 and 20:1/30:1 relative error costs; and all classification-accuracy figures quoted here (58%–76% and 7.6%–17.5% in the estimation sample, 37.5%–56% and 3.5%–9.1% in the holdout, and the 74%/13.8% and 50%/7.2% figures at the -1.78 cutoff).
- Messod D. Beneish, ["Detecting GAAP violation: implications for assessing earnings management among firms with extreme financial performance"](https://www.sciencedirect.com/science/article/abs/pii/S0278425497000239), *Journal of Accounting and Public Policy*, vol. 16, no. 3, 1997, pp. 271–309. The earlier, separate five-variable model, and the source to which the -2.22 threshold belongs. Verified directly against the 1999 paper: the value -2.22 does not appear in it, and the only cutoffs it reports for the eight-variable model are -1.49 and -1.78.
- Messod D. Beneish, Charles M.C. Lee and D. Craig Nichols, ["Earnings Manipulation and Expected Returns"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2241717), *Financial Analysts Journal*, vol. 69, no. 2, 2013, pp. 57–82. Source for the out-of-sample return predictability results and the reported identification of 71% of prominent post-estimation fraud cases in advance of public disclosure.
- [SEC Accounting and Auditing Enforcement Releases](https://www.sec.gov/divisions/enforce/friactions.htm) — the enforcement record from which Beneish's manipulator sample was drawn, and the primary source for the WorldCom and other US enforcement actions referenced here.

**A note on what is illustrative.** Every company-level walkthrough in this post — the four-year receivables and accruals charts, the software company's asset mix, and Meridian Systems' full calculation — uses invented figures, chosen for arithmetic clarity and labelled illustrative where they appear. The coefficients, thresholds, sample sizes and accuracy statistics are Beneish's published figures. The Cornell/Enron account, and the WorldCom and Wirecard cases, describe reported real events.

**Further reading on this blog:**

- [Enron 2001: the accounting fraud](/blog/trading/finance/enron-2001-accounting-fraud) — the mechanisms the M-Score could not see.
- [Wirecard: the German fintech fraud](/blog/trading/finance/wirecard-the-german-fintech-fraud) — the fabricated-cash case that inverts the model.
- [Accrual accounting versus cash: the gap fraud exploits](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits) — the identity underneath TATA.
- [Revenue recognition games: channel stuffing and bill-and-hold](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold) — what DSRI is built to catch.
- [Capitalizing costs to inflate profit: the WorldCom move](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move) — what AQI and DEPI are built to catch.
- [Off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities) — the structural blind spot.
- [Quality of earnings: accruals, one-offs and red flags](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) — fitting the score into a full research process.
