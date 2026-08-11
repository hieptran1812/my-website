---
title: "The Altman Z-score: reading a company's odds of going broke off two statements"
date: "2026-08-09"
publishDate: "2026-08-09"
description: "How five ordinary financial ratios, weighted by a formula fitted in 1968, collapse into a single number that sorts a company into safe, grey or distress — worked line by line on Sears Holdings and Eastman Kodak, including the case where the score got it wrong."
tags: ["forensic-accounting", "altman-z-score", "bankruptcy-prediction", "financial-distress", "credit-analysis", "financial-statements", "discriminant-analysis", "sears-holdings", "eastman-kodak", "solvency", "working-capital", "retained-earnings"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 50
---

> [!important]
> **TL;DR** — The Altman Z-score takes five ratios you can read off a balance sheet and an income statement, multiplies each by a fixed weight, and adds them up into one number. Above 2.99 the company looks like the survivors in Altman's sample; below 1.81 it looks like the bankrupts; in between is what Altman himself called the zone of ignorance.
>
> - The five inputs are working capital, retained earnings, EBIT and sales — each divided by total assets — plus the market value of equity divided by total liabilities. Four come from the accounts; one comes from the stock market.
> - The weights (1.2, 1.4, 3.3, 0.6 and 1.0) were not derived from theory. They were fitted by discriminant analysis on 33 bankrupt US manufacturers and 33 survivors, in Altman's 1968 *Journal of Finance* paper.
> - Two variants exist because two of the inputs travel badly: Z-prime swaps the market value of equity for book value (private firms), and Z-double-prime also drops the sales ratio (non-manufacturers and emerging markets). Every weight and every cutoff changes.
> - Sears Holdings' Z-double-prime was already below the distress line at the fiscal year-end of 28 January 2012 and fell to **-4.40** by 3 February 2018, eight months before it filed for Chapter 11 on 15 October 2018.
> - Eastman Kodak is the counter-case: its Z-score *rose* from roughly 1.9 to roughly 2.2 in the four years before its Chapter 11 filing on 19 January 2012, because total assets — the denominator of four of the five ratios — collapsed faster than the numerators did.
> - The one thing to remember: **the Z-score measures resemblance, not causation.** It tells you a company's ratios look like those of firms that failed. It does not know why, and it cannot see anything that is not on the two statements.

There is a particular kind of corporate death that everybody claims to have seen coming afterwards and almost nobody acts on beforehand. The company keeps filing. The auditor keeps signing. The press releases keep using the word "transformation." And then one Thursday morning there is a petition in a bankruptcy court and a press release that begins "to facilitate an orderly."

In 1968 a young finance professor at New York University asked a deliberately narrow question about this. Not *why* do companies fail — that question is bottomless. Instead: if you line up the companies that went bankrupt next to the companies that did not, and you look only at the ratios you can compute from their published accounts, **can you tell them apart?**

The answer was mostly yes. And the tool that came out of it — five ratios, five weights, one number — is still, half a century later, one of the first things a credit analyst computes when a name lands on their desk. It is a smoke detector. It is not a fire inspector. Knowing the difference is the whole point of this article.

![Five financial ratios feeding a weighted sum that maps to three distress zones](/imgs/blogs/the-altman-z-score-predicting-financial-distress-1.webp)

The figure above is the whole machine. On the left are five ratios, each of which asks a different question about survival. In the middle they are multiplied by fixed weights and added. On the right the single resulting number falls into one of three bands. Everything else in this article is a tour of that picture: where each ratio comes from, where the weights came from, what the bands mean, when the machine works, and — the part that matters most for a forensic reader — the specific circumstances under which it will look you in the eye and lie.

## Foundations: the building blocks of a bankruptcy score

If you have never read a financial statement, you can still follow all of this. We are going to define every piece from zero.

### What "going bankrupt" actually means

A company does not go bankrupt because it loses money. Plenty of companies lose money for a decade and survive comfortably, because somebody keeps funding them. A company goes bankrupt when it **runs out of ways to pay what it owes when it is owed** — and someone with a legal claim forces the issue, or management files pre-emptively to get the protection of a court.

In the United States the two relevant chapters of the Bankruptcy Code are **Chapter 11** (reorganisation — the business keeps operating under court supervision while it restructures its debts) and **Chapter 7** (liquidation — the business stops, the assets are sold, the proceeds are distributed). Both of our case studies filed Chapter 11.

The distinction that matters for the Z-score is between two different ways of being in trouble:

- **Illiquidity** — you have valuable assets but you cannot turn enough of them into cash this month to pay the bill that is due this month. A profitable business with all its money tied up in inventory can be illiquid.
- **Insolvency** — the value of what you owe exceeds the value of what you own, so there is no arrangement of the assets that makes the creditors whole.

Most real failures are a slow drift from the second into the first: the balance sheet erodes for years, and then one refinancing fails and the illiquidity arrives all at once. The Z-score tries to pick up the drift.

If you want the full anatomy of these two tests before continuing, the companion piece on [liquidity and solvency](/blog/trading/equity-research/liquidity-and-solvency-can-the-company-survive) walks the individual ratios one at a time.

### The two statements the score reads

Every input to the Z-score comes from one of two documents.

The **balance sheet** is a photograph taken at one instant — the last day of the fiscal year. It has two sides that must be equal:

- **Assets** — everything the company owns or is owed. Split into *current* assets (cash, money customers owe you, inventory — things expected to become cash within twelve months) and *non-current* assets (factories, machines, patents, goodwill).
- **Liabilities and equity** — everything the company owes, plus whatever is left over for the owners. Liabilities are also split into *current* (bills, wages, debt repayable within twelve months) and *long-term*. **Equity** is the residual: assets minus liabilities.

The **income statement** is a video of the whole year: revenue at the top, costs subtracted step by step, profit at the bottom. Two lines from it matter here — **sales** (also called revenue or net sales: the value of what the company sold) and **EBIT**, earnings before interest and taxes, which is profit measured *before* the cost of borrowing and *before* tax.

Why EBIT rather than net profit? Because EBIT measures what the business earns from operating its assets, independent of how those assets were financed. Two identical factories, one funded with debt and one with equity, have the same EBIT and very different net profit. Since the Z-score is trying to judge the *business*, and separately judge the *financing* through other ratios, it wants the two kept apart.

![Balance sheet and income statement annotated with where each Z-score input is read off](/imgs/blogs/the-altman-z-score-predicting-financial-distress-2.webp)

The figure above shows exactly which line each input is lifted from. Note the one item that is not on either statement: the numerator of the fourth ratio is the company's **market value of equity** — shares outstanding multiplied by the share price. That is a number the stock market produces, not the accountant. It is what makes the Z-score a hybrid: four-fifths accounting, one-fifth market opinion. We will come back to how much that matters.

### Why everything is divided by total assets

Four of the five ratios have **total assets** on the bottom. This is not decoration. A raw number like "working capital of \$553 million" is meaningless on its own — it is enormous for a bakery and rounding error for an airline. Dividing by total assets turns every input into a *proportion*, which lets you compare a \$500 million company with a \$50 billion one on the same axis.

This is the single most useful habit in financial analysis, and the same instinct drives [common-size and trend analysis](/blog/trading/forensic-accounting/common-size-and-trend-analysis-making-statements-comparable), where every line of a statement is expressed as a percentage of a common base.

It also, as we will see with Kodak, contains a trap. If a company is shrinking, total assets falls. A falling denominator makes a ratio rise even when nothing good is happening. Keep that in your pocket.

### What "discriminant analysis" means, without the algebra

Here is the statistical idea in one paragraph, with no equations.

You have two groups of companies: the ones that went bankrupt and the ones that did not. For each company you have five measurements. If you plot the two groups on any *single* one of those five measurements, the clouds overlap — there are bankrupt firms with decent profitability and healthy firms with thin working capital. No single ratio separates them cleanly.

**Multiple discriminant analysis** asks: is there a *weighted combination* of the five measurements that separates the two clouds better than any measurement alone? It searches for the set of weights that pushes the two groups as far apart as possible relative to how spread out each group is internally. The output is a formula — one number per company — and a cutoff value. Companies scoring on one side of the cutoff get classified as likely-bankrupt; the other side, likely-healthy.

That is all the Z-score is. It is a line drawn through a cloud of dots so as to separate them as well as a line can. It carries no theory of *why* firms fail. It only knows what failed firms' accounts looked like in the sample it was fitted on.

#### Worked example: the whole machine on one illustrative firm

Let us run the formula once, end to end, on made-up round numbers so the mechanics are clear before any real company arrives. **Every number in this example is illustrative** — invented for teaching, not taken from a real filing.

Suppose a manufacturer with the following (illustrative) figures, all in millions:

| Line | Value |
| --- | --- |
| Total assets | \$1,000 |
| Current assets | \$400 |
| Current liabilities | \$250 |
| Retained earnings | \$300 |
| EBIT | \$120 |
| Total liabilities | \$500 |
| Market value of equity | \$600 |
| Sales | \$1,400 |

Step one, compute the five ratios:

1. **Working capital** is current assets minus current liabilities: \$400 − \$250 = \$150. So X1 = 150 / 1,000 = **0.15**.
2. **Retained earnings** over total assets: X2 = 300 / 1,000 = **0.30**.
3. **EBIT** over total assets: X3 = 120 / 1,000 = **0.12**.
4. **Market value of equity** over total liabilities: X4 = 600 / 500 = **1.20**.
5. **Sales** over total assets: X5 = 1,400 / 1,000 = **1.40**.

Step two, multiply each by its weight and add:

- 1.2 × 0.15 = 0.18
- 1.4 × 0.30 = 0.42
- 3.3 × 0.12 = 0.396
- 0.6 × 1.20 = 0.72
- 1.0 × 1.40 = 1.40

Total: 0.18 + 0.42 + 0.396 + 0.72 + 1.40 = **3.12**. That is above 2.99, so this illustrative firm sits in the safe zone.

![Horizontal bars showing the five weighted contributions to an illustrative firm's Z-score of 3.12](/imgs/blogs/the-altman-z-score-predicting-financial-distress-3.webp)

The figure above draws those five contributions to scale, and it makes a point that the arithmetic hides. The contributions are wildly unequal. **X5 alone supplies 1.40 of the 3.12 — about 45% of the entire score.** That is not because asset turnover is the most important thing about a company. It is because X5 is the only ratio that routinely exceeds 1.0: a manufacturer that sells \$1.40 of goods for every \$1.00 of assets it owns is completely ordinary, whereas a firm earning \$1.40 of EBIT per \$1.00 of assets would be a miracle.

**The intuition this example teaches:** the weights are not importance rankings. A weight of 3.3 on EBIT does not mean profitability matters 3.3 times as much as anything; it means the *typical spread* of EBIT-to-assets across firms is small, so it takes a big multiplier for that ratio to move the total at all.

## 1. The five ratios, one at a time

Each ratio is a different question about survival. A company can answer one of them badly and be fine. The score collapses when several go wrong together.

![Matrix of the five ratios, the question each asks, and the failure each front-runs](/imgs/blogs/the-altman-z-score-predicting-financial-distress-5.webp)

### X1 — working capital over total assets: can it get through the year?

**Working capital** is current assets minus current liabilities: the cash and near-cash the company has, minus the bills coming due inside twelve months. A positive figure means the next year's obligations are covered by the next year's resources. A negative figure means they are not, and the gap has to be filled by borrowing, by selling something, or by persuading suppliers to wait.

This ratio is the closest thing in the score to a pure liquidity test. It is also the one most easily dressed up at a period end — stretching payables for two weeks over the year-end, or drawing down a revolver into cash, both move it. Anyone reading this ratio should also read [what the cash conversion cycle reveals about working capital](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals), which is where the seasonal and year-end games show up.

One structural caveat, which matters enormously for modern companies: **negative working capital is not always distress.** A supermarket collects cash from customers immediately and pays its suppliers in 45 days; it runs permanently negative working capital and it is a fine business. A software company that bills annually in advance carries a huge *deferred revenue* balance in current liabilities — money already received for service not yet delivered — which drives X1 deeply negative on a company with no debt and a growing cash pile.

### X2 — retained earnings over total assets: has it ever kept a profit?

**Retained earnings** is a cumulative account. Every year, the profit the company made and did *not* pay out as dividends gets added to it; every year it loses money, the loss is subtracted. Over decades it becomes a running tally of lifetime profitability that was retained inside the business.

Altman's reasoning for including it was that it captures *age and cumulative profitability at once*. A young company has had no time to accumulate anything, and young companies fail more often. An old company that has bled for a decade will have ground its retained earnings down, or into an **accumulated deficit** — the same account with a negative balance.

This is the most powerful single term in the score for a mature firm, and it is also the most treacherous, for a reason we will spend real time on in the Kodak section: **retained earnings is a historical record, not a present-day cushion.** Profits earned in 1985 and spent on share buybacks in 2005 still sit in retained earnings in 2011. The account says the money was earned. It does not say the money is still there.

### X3 — EBIT over total assets: do the assets earn anything?

This is the productivity of the asset base, measured before financing costs and tax. It is the closest thing in the score to a test of **economic insolvency**: if the assets do not generate a return before you even consider the interest bill, then no capital structure saves the business, only a change to the business itself.

It carries the largest weight — 3.3 — because the spread of this ratio across firms is narrow. Moving from 5% to 0% return on assets is a large economic event that shows up as a change of only 0.05 in the raw ratio, so it needs a big multiplier to register.

### X4 — market value of equity over total liabilities: how far can the equity fall?

This is the only forward-looking input, and the only one the accountant does not produce.

The numerator is the **market value of equity** — every share outstanding multiplied by the market price. The denominator is the **book value of total liabilities**, straight from the balance sheet. The ratio answers: by what proportion can the market's valuation of this company drop before the debts exceed the value of the whole enterprise?

Two features are worth pausing on. First, it makes the score partly a *price* signal — it will fall the moment the market gets nervous, which is often well before the accounts show anything. Second, it makes the score partly *circular*: a stock falling because investors fear bankruptcy pushes the Z-score down, which is then cited as evidence of bankruptcy risk. The market's opinion is an input, not an independent check on it.

Note also what the denominator does and does not include. Obligations kept off the balance sheet do not appear. Everything in the piece on [hidden liabilities — leases, guarantees and contingencies](/blog/trading/forensic-accounting/hidden-liabilities-leases-guarantees-and-contingencies) is, by construction, invisible to X4 in the years before accounting rules dragged it on-balance-sheet.

### X5 — sales over total assets: do the assets turn over?

**Asset turnover** measures how much revenue the company generates per dollar of assets. It is a capital-intensity measure: a distributor might turn its assets over three times a year, a utility a third of a time.

Altman noted this ratio was the weakest discriminator of the five when taken alone, but it earned its place because of how it interacted with the others in the combination. It is also the input that travels worst across industries — which is precisely why the four-variable variant drops it.

#### Worked example: two firms, identical assets, different survival odds

Consider two illustrative manufacturers, each with exactly \$1,000 million of total assets and \$500 million of total liabilities, and each earning \$50 million of EBIT. **Both firms are illustrative.**

Firm A has current assets of \$450 million against current liabilities of \$200 million, retained earnings of \$400 million, sales of \$1,200 million, and a market value of equity of \$700 million.

Firm B has current assets of \$260 million against current liabilities of \$300 million, retained earnings of \$40 million, sales of \$700 million, and a market value of equity of \$250 million.

Firm A:

- X1 = (450 − 200) / 1,000 = 0.25 → 1.2 × 0.25 = 0.30
- X2 = 400 / 1,000 = 0.40 → 1.4 × 0.40 = 0.56
- X3 = 50 / 1,000 = 0.05 → 3.3 × 0.05 = 0.165
- X4 = 700 / 500 = 1.40 → 0.6 × 1.40 = 0.84
- X5 = 1,200 / 1,000 = 1.20 → 1.0 × 1.20 = 1.20
- **Z = 3.07** — safe zone.

Firm B:

- X1 = (260 − 300) / 1,000 = −0.04 → 1.2 × −0.04 = −0.048
- X2 = 40 / 1,000 = 0.04 → 1.4 × 0.04 = 0.056
- X3 = 50 / 1,000 = 0.05 → 3.3 × 0.05 = 0.165
- X4 = 250 / 500 = 0.50 → 0.6 × 0.50 = 0.30
- X5 = 700 / 1,000 = 0.70 → 1.0 × 0.70 = 0.70
- **Z = 1.17** — distress zone.

The two firms have identical size, identical leverage on a book basis, and identical operating profit. The gap of 1.90 points comes entirely from three things: Firm B has negative working capital, almost no accumulated profit, and assets that turn over far more slowly.

**The intuition this example teaches:** the Z-score is not a leverage ratio in disguise. Two firms with the same debt-to-assets can sit in opposite zones, because the score is asking about liquidity, history and productivity as well.

## 2. Where the weights came from

### The 1968 study, precisely

The model comes from Edward I. Altman, "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy," published in the *Journal of Finance*, Volume 23, Number 4, September 1968, pages 589–609.

The sample was deliberately small and deliberately matched. Altman took **33 manufacturing corporations that filed bankruptcy petitions under the US National Bankruptcy Act between 1946 and 1965**, and paired them with **33 manufacturing corporations that did not**, matched by industry and by approximate asset size — 66 firms in total. The bankrupt firms sat in a range of roughly \$1 million to \$25 million of assets — small companies, in 1960s dollars, all of them makers of physical things.

He started from a list of 22 candidate ratios drawn from the existing literature and from what practitioners actually used, and reduced them to the five that performed best *in combination*. The five that survived are the ones above.

The resulting function, in the form almost everyone quotes today:

$$Z = 1.2\,X_1 + 1.4\,X_2 + 3.3\,X_3 + 0.6\,X_4 + 1.0\,X_5$$

### The percentage-versus-decimal trap

There is a genuine and very common source of confusion here, and it is worth thirty seconds because it produces answers that are wrong by a factor of a hundred.

In the original paper the first four ratios were expressed as **percentages**. A working-capital-to-assets ratio of 0.15 was entered as the number **15**. To make that work, the coefficients were correspondingly smaller: 0.012, 0.014, 0.033, 0.006 and 0.999.

Modern usage expresses the ratios as **decimals** — 0.15 rather than 15 — and scales the first four coefficients up by a hundred to compensate: 1.2, 1.4, 3.3 and 0.6. The fifth coefficient, which multiplies a ratio that was always entered as a decimal, stays at 0.999, usually rounded to 1.0.

The two forms give the same answer. The failure mode is mixing them: entering decimals into the small coefficients, or percentages into the large ones.

#### Worked example: the same firm in both conventions

Take the illustrative firm from earlier, with X1 = 0.15, X2 = 0.30, X3 = 0.12, X4 = 1.20 and X5 = 1.40. **Illustrative numbers again.**

In the **decimal** convention, which we already computed: 1.2(0.15) + 1.4(0.30) + 3.3(0.12) + 0.6(1.20) + 1.0(1.40) = 0.18 + 0.42 + 0.396 + 0.72 + 1.40 = **3.12**.

In the **percentage** convention the first four ratios become 15, 30, 12 and 120, the fifth stays 1.40, and the coefficients shrink:

- 0.012 × 15 = 0.18
- 0.014 × 30 = 0.42
- 0.033 × 12 = 0.396
- 0.006 × 120 = 0.72
- 0.999 × 1.40 = 1.399

Total: **3.12**, to the rounding. Identical, as it must be.

Now the failure mode. If you enter decimals into the percentage-form coefficients, you get 0.012(0.15) + 0.014(0.30) + 0.033(0.12) + 0.006(1.20) + 0.999(1.40) = 1.41 — a firm that looks like it is about to die. If you enter percentages into the decimal-form coefficients you get roughly 208, which is obviously nonsense but has been known to survive into a spreadsheet unnoticed because nobody looked.

**The intuition this example teaches:** always sanity-check a Z-score against the range 0 to 6. Anything outside that band for a normal operating company means you have mixed the conventions.

### The weights are not laws of nature

This deserves saying plainly. The coefficients 1.2, 1.4, 3.3, 0.6 and 1.0 are the output of a statistical fit to 66 American manufacturing companies observed between 1946 and 1965. They are not derived from accounting identities, they are not implied by any theory of the firm, and there is no reason to expect them to be optimal for a Vietnamese property developer in 2026 or a US software company in 2015.

What has kept the model alive is not that the weights are right. It is that the five *ratios* turn out to be durably informative, and that the combination is robust enough to remain useful even when the exact weights are stale.

## 3. The three zones, and the two ways to be wrong

### The cutoffs

Altman's fitted model produced a single optimal cutoff score of **2.675** — the point that minimised total misclassification in his sample. A company scoring above it was classified as a non-bankrupt, below it as a bankrupt.

But he observed something important about the region around that cutoff: the two groups overlapped there. Rather than pretend a single line was clean, he reported a band running from **1.81 to 2.99** in which classification was unreliable — the **grey zone**, which he described as a zone of ignorance. Below 1.81, every firm in his sample that scored there was bankrupt. Above 2.99, every firm that scored there was not. In between, both kinds lived.

That is why the practical convention has three zones rather than two:

| Zone | Original Z | Reading |
| --- | --- | --- |
| Safe | above 2.99 | Ratios resemble the survivors |
| Grey | 1.81 to 2.99 | The model cannot separate the two groups here |
| Distress | below 1.81 | Ratios resemble the bankrupts |

![Number line showing the distress, grey and safe zones with the 1.81, 2.675 and 2.99 cutoffs and the two error types](/imgs/blogs/the-altman-z-score-predicting-financial-distress-4.webp)

### What the reported accuracy actually says

In its initial test the model was reported as **72% accurate at predicting bankruptcy two years before the event, with a Type II error of 6%** (Altman, 1968); the one-year-ahead classification on that original matched sample is commonly reported at around **95%**. In a series of subsequent tests covering three periods up to 1999, the model was found to be approximately **80–90% accurate one year before the event, with false positives of roughly 15–20%**. Accuracy declines substantially as the horizon lengthens to three, four and five years.

Three qualifications belong next to those numbers every single time they are quoted.

First, they describe **classification of the sample the model was fitted on**, plus small secondary tests. A model fitted to 66 companies and then scored on those 66 companies will always flatter itself. Out-of-sample performance on later decades has generally been lower.

Second, the sample was **matched fifty-fifty** — half the firms in it went bankrupt. In the real world, the annual bankruptcy rate among listed companies is a small fraction of one percent. When a test with a fixed false-positive rate is applied to a population where the base rate is tiny, the great majority of the alarms it raises will be false. This is the base-rate problem and it is not a flaw in the arithmetic; it is a fact about what any screen does when the thing it screens for is rare.

Third, "accuracy" bundles two very different errors together, and the figure above separates them:

- A **Type I error** is a company that fails despite scoring in the safe zone. This is the expensive one for a lender or a bondholder: you were told it was fine, and you lost the principal.
- A **Type II error** is a healthy company that scores in the distress zone. This is a false alarm. For an analyst it costs a wasted afternoon. For a bank it costs the margin on a loan it declined, and for a supplier it may cost a customer.

Moving the cutoff trades one against the other and cannot reduce both. Raise it and you catch more failures and generate more false alarms; lower it and the reverse. Any claim that a distress model is "more accurate" without saying which error it reduced is not saying anything.

### How to actually use a score

The productive use of the Z-score is not the level. It is the **direction, the slope and the decomposition.**

- **Direction**: which zone, and has it changed since last year?
- **Slope**: how fast is it moving? A company that fell 0.9 points in two years is telling you more than a company that has sat at 2.4 for a decade.
- **Decomposition**: *which of the five terms* moved? A score that fell because X4 collapsed is a story about the market's opinion. A score that fell because X3 went negative is a story about the business. They call for completely different follow-up work.

This is the same discipline as [monitoring a live thesis with a watch dashboard](/blog/trading/analyst-edge/monitoring-a-live-thesis-building-your-watch-dashboard): a single indicator is an alarm, not an answer, and the value is in what it makes you go and check.

## 4. Z-prime and Z-double-prime: the two variants

The original model has two hard requirements that most of the world's companies fail. It needs a **market price** for the equity, which private companies do not have. And it uses **asset turnover**, which varies so much between a steel mill and a consultancy that a coefficient fitted on manufacturers is close to meaningless elsewhere.

Altman addressed both with re-estimated models. The important thing to understand is that these are not adjustments to the original formula. Each is a **complete refit**: every coefficient changes, and so does every cutoff.

![Comparison grid of the Z, Z-prime and Z-double-prime models, their inputs, coefficients and cutoffs](/imgs/blogs/the-altman-z-score-predicting-financial-distress-6.webp)

### Z-prime — for private firms

X4's numerator becomes the **book value of equity** instead of the market value. Everything else keeps its definition, and the model is refitted:

$$Z' = 0.717\,X_1 + 0.847\,X_2 + 3.107\,X_3 + 0.420\,X_4 + 0.998\,X_5$$

The zones move down: distress below **1.23**, grey from 1.23 to **2.90**, safe above 2.90.

Note how much smaller the coefficient on X4 becomes — 0.420 against 0.600. Book equity is a duller instrument than market value; it does not react to news, and the refit gives it correspondingly less say.

### Z-double-prime — for non-manufacturers and emerging markets

This variant keeps the book-value definition of X4 and **drops X5 entirely**, leaving four variables:

$$Z'' = 6.56\,X_1 + 3.26\,X_2 + 6.72\,X_3 + 1.05\,X_4$$

The zones move again: distress below **1.1**, grey from 1.1 to **2.6**, safe above 2.6.

The coefficients look wildly bigger, which alarms people the first time they see it. They are bigger because the model lost the one term that was reliably contributing more than 1.0 on its own; the remaining four have to carry the entire scale.

Altman and co-authors also used this four-variable form as the base of an **emerging-market score**, adding a constant of 3.25 so that the resulting number lines up with the scale of US bond ratings, where a score of zero corresponds to a defaulted credit. If you meet a Z-double-prime that looks implausibly high by 3.25, that constant is why.

### Which one to use

| Situation | Model | Why |
| --- | --- | --- |
| Listed US-style manufacturer | Z | The fitted population |
| Private manufacturer, no share price | Z-prime | Book equity substitute |
| Retailer, services, distribution, most listed non-manufacturers | Z-double-prime | Asset turnover not comparable |
| Emerging-market corporate credit | Z-double-prime, often with the +3.25 constant | Designed for it |
| Bank, insurer, broker, any financial institution | **None of them** | Excluded from the sample by construction |

#### Worked example: one real company, three models, three answers

This is where the variants stop being an academic footnote. Take **Eastman Kodak's balance sheet at 31 December 2011**, as reported in its Form 10-K for that year (SEC EDGAR, CIK 0000031235, filed 29 February 2012), in millions of US dollars:

| Line | Value |
| --- | --- |
| Total assets | \$4,678 |
| Total current assets | \$2,703 |
| Total current liabilities | \$2,150 |
| Total liabilities | \$7,028 |
| Retained earnings | \$4,071 |
| Total shareholders' equity | **−\$2,352** |
| Loss from continuing operations before interest expense, other income (charges), net and income taxes | −\$600 |
| Net sales | \$6,022 |

The shared ratios:

- X1 = (2,703 − 2,150) / 4,678 = 553 / 4,678 = **0.1182**
- X2 = 4,071 / 4,678 = **0.8702**
- X3 = −600 / 4,678 = **−0.1283**
- X5 = 6,022 / 4,678 = **1.2873**

For the original Z we need the market value of equity. Kodak's own 10-K reports fourth-quarter 2011 trading in a range of \$0.62 to \$1.63 per share, and 271,415,654 shares outstanding as of 17 February 2012. At the low end that is a market value of roughly \$168 million; at the high end roughly \$442 million. So X4 falls between 168 / 7,028 = 0.024 and 442 / 7,028 = 0.063.

- **Z** = 1.2(0.1182) + 1.4(0.8702) + 3.3(−0.1283) + 0.6(0.024 to 0.063) + 1.0(1.2873) = **2.24 to 2.26** → **grey zone**.

For Z-prime and Z-double-prime, X4 uses book equity: −2,352 / 7,028 = **−0.3347**.

- **Z-prime** = 0.717(0.1182) + 0.847(0.8702) + 3.107(−0.1283) + 0.420(−0.3347) + 0.998(1.2873) = **1.57** → grey, but much closer to its 1.23 distress line.
- **Z-double-prime** = 6.56(0.1182) + 3.26(0.8702) + 6.72(−0.1283) + 1.05(−0.3347) = **2.40** → grey, below its 2.6 safe line.

Three models, one company, one balance sheet, and answers spread from 1.57 to 2.40 on scales whose distress thresholds are 1.23, 1.81 and 1.1 respectively. Kodak filed for Chapter 11 protection **seven weeks after that balance sheet date**, on 19 January 2012.

**The intuition this example teaches:** a Z-score without its model name and its cutoffs attached is an uninterpretable number. And on a company whose book equity is deeply negative while its market value is nearly zero, the choice between market and book value of equity moves the answer by two-thirds of a point.

## 5. Case study: Sears Holdings, a score that fell for seven straight years

Sears Holdings is the case that shows the model doing exactly what it was built to do.

Sears is a retailer, so the four-variable **Z-double-prime** is the appropriate model: no manufacturing asset turnover to compare against, and a capital structure in which the book value of equity is the honest input. All figures below come from Sears Holdings Corporation's Form 10-K filings as tagged in SEC XBRL company facts (CIK 0001310067). Sears' fiscal years end in late January or early February, so the fiscal year-end dates look odd; they are the real ones.

One methodological note, because it matters for reproducibility. Where a later filing restated a prior year's figure — usually because a business was reclassified as discontinued — the restated figure is used here. The differences are small: total assets at 1 February 2014, for instance, appear as \$18,261 million in one filing and \$18,234 million in a later one, a gap of 0.1% that moves the score by less than 0.01. Anyone reproducing this table from a different vintage of filings should expect variation at the second decimal place and none at all in the conclusion.

| Fiscal year end | Total assets | Working capital | Total liabilities | Retained earnings | Book equity | EBIT (operating income) | Z'' |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2012-01-28 | \$21,381 | \$1,032 | \$17,040 | \$1,865 | \$4,281 | −\$1,501 | **0.39** |
| 2013-02-02 | \$19,340 | \$851 | \$16,168 | \$885 | \$2,755 | −\$838 | **0.33** |
| 2014-02-01 | \$18,261 | \$774 | \$16,078 | −\$480 | \$1,739 | −\$927 | **−0.04** |
| 2015-01-31 | \$13,185 | \$268 | \$14,130 | −\$2,162 | −\$951 | −\$1,508 | **−1.24** |
| 2016-01-30 | \$11,337 | \$607 | \$13,293 | −\$3,291 | −\$1,963 | −\$1,000 | **−1.34** |
| 2017-01-28 | \$9,362 | \$315 | \$13,186 | −\$5,512 | −\$3,824 | −\$1,978 | **−3.42** |
| 2018-02-03 | \$7,262 | −\$1,103 | \$10,985 | −\$5,895 | −\$3,723 | −\$430 | **−4.40** |

All values in millions of US dollars. Sears Holdings filed for Chapter 11 protection on 15 October 2018.

![Line chart of Sears Holdings' Z-double-prime score falling from 0.39 to minus 4.40 across seven fiscal year-ends](/imgs/blogs/the-altman-z-score-predicting-financial-distress-7.webp)

Read the chart and the table together and four things stand out.

**The score was already below the distress threshold at the first observation.** At the fiscal year-end of 28 January 2012, Z-double-prime was 0.39 against a distress line of 1.1. That is nearly seven years before the filing. Anyone treating a distress reading as a prediction that bankruptcy is imminent would have been wrong for six consecutive years.

**The direction never reversed.** Seven observations, six of them lower than the one before. There is no year in this table where a shareholder could point to the score and say the trend had turned.

**The crossings are informative even when the level is not.** Retained earnings went negative between the 2013 and 2014 fiscal year-ends. Book equity went negative between 2014 and 2015. Working capital went negative in the final year. Each of those is a sign change in a term, and each one shows up as a step down in the score.

**The last year's EBIT looks better and the score still fell.** Operating loss narrowed from \$1,978 million to \$430 million in the final year, which improved the X3 contribution. The score fell anyway, from −3.42 to −4.40, because working capital turned negative and the accumulated deficit kept growing. A company can improve on the income statement while the balance sheet continues to deteriorate — a pattern that reappears constantly in the material on [why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

#### Worked example: computing Sears' Z-double-prime for the 2017-01-28 fiscal year

Let us do one of these rows by hand, in full, so nothing is taken on trust. Figures in millions of US dollars, from the Form 10-K for the fiscal year ended 28 January 2017:

- Total assets: \$9,362
- Current assets: \$4,996; current liabilities: \$4,681 → working capital = \$315
- Total liabilities: \$13,186
- Retained earnings (accumulated deficit): −\$5,512
- Total shareholders' equity: −\$3,824
- Operating loss: −\$1,978

The four ratios:

1. X1 = 315 / 9,362 = **0.0336**
2. X2 = −5,512 / 9,362 = **−0.5888**
3. X3 = −1,978 / 9,362 = **−0.2113**
4. X4 = −3,824 / 13,186 = **−0.2900**

The four contributions:

- 6.56 × 0.0336 = **+0.22**
- 3.26 × (−0.5888) = **−1.92**
- 6.72 × (−0.2113) = **−1.42**
- 1.05 × (−0.2900) = **−0.30**

Sum: 0.22 − 1.92 − 1.42 − 0.30 = **−3.42**.

Against a distress threshold of 1.1, that is not a marginal reading. The decomposition tells you where it came from: the accumulated deficit contributes −1.92 and the operating loss −1.42, together accounting for essentially the whole score. Working capital was still mildly positive and contributed the only positive term in the sum.

**The intuition this example teaches:** once retained earnings turns into an accumulated deficit and EBIT turns negative, two of the four terms are pulling in the same direction with large weights, and the score does not recover without a change in the business itself.

### What the score would and would not have told you

Honesty requires saying what this case does not prove.

It does not prove the Z-score predicted the date. It flagged distress in early 2012 and the filing came in late 2018. In between, Sears separated Lands' End into a standalone listed company, reduced its stake in Sears Canada, sold a large tranche of its store real estate into a listed property trust, and borrowed repeatedly from entities affiliated with its chairman. Each of those transactions bought time that the model had no way to anticipate, and each is visible in the table above as a step down in total assets.

What the score did do is refuse, for seven straight years, to say the company was fine. That is a lower bar than prediction, and it is still more than most narratives managed over the same period.

## 6. Case study: Eastman Kodak, where the score quietly misled

Now the uncomfortable case, and the reason this article exists in a forensic-accounting series rather than a credit-analysis one.

Eastman Kodak was a listed manufacturer — precisely the population the original Z-score was fitted on. It filed for Chapter 11 protection on 19 January 2012. If the model works anywhere, it should work here.

Here is what an analyst computing the original Z-score at each fiscal year-end would have seen. All figures come from Kodak's own Form 10-K filings (SEC EDGAR, CIK 0000031235): the 2008 filing for the 2007 and 2008 columns, the 2010 filing for 2009 and 2010, and the 2011 filing for 2011. Amounts in millions of US dollars. Because a fiscal-year-end closing price is not disclosed in the filings, the market value of equity is computed from the fourth-quarter high and low prices in each 10-K's own Item 5 market price table, which gives a range for Z rather than a point.

| Fiscal year | Total assets | Working capital | Retained earnings | EBIT | Net sales | Q4 price range | Z (range) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2007 | \$13,659 | \$1,607 | \$6,474 | −\$230 | \$10,301 | \$21.42–\$29.60 | **1.85–1.98** |
| 2008 | \$9,179 | \$1,542 | \$5,879 | −\$821 | \$9,416 | \$5.83–\$15.68 | **1.94–2.14** |
| 2009 | \$7,691 | \$1,407 | \$5,676 | −\$28 | \$7,606 | \$3.26–\$4.74 | **2.30–2.33** |
| 2010 | \$6,239 | \$966 | \$4,969 | −\$336 | \$7,187 | \$3.84–\$5.95 | **2.36–2.41** |
| 2011 | \$4,678 | \$553 | \$4,071 | −\$600 | \$6,022 | \$0.62–\$1.63 | **2.24–2.26** |

Look at the last column. Over the four years leading into a Chapter 11 filing, **Kodak's Z-score went up.**

It started at roughly 1.9 — barely above the 1.81 distress line — and finished at roughly 2.25, comfortably inside the grey zone. At no point in this window did the original Z-score put Eastman Kodak in the distress zone. Seven weeks before the petition, the model's answer was "I cannot tell."

### Why the score rose while the company died

There are two mechanisms and both are general.

**Mechanism one: the denominator collapsed.** Total assets fell from \$13,659 million to \$4,678 million — a decline of about 66% in four years — as Kodak sold businesses, wrote down goodwill, and shrank. Four of the five ratios have total assets on the bottom. When the bottom of a fraction falls faster than the top, the fraction rises.

![Kodak total assets collapsing while the retained-earnings and sales ratios rise](/imgs/blogs/the-altman-z-score-predicting-financial-distress-8.webp)

The figure above shows this directly. Retained earnings over total assets climbed from 0.47 to 0.87. Sales over total assets climbed from 0.75 to 1.29. Both of those look, on the face of the ratio, like improvements. Both were produced by the company getting smaller, not better. Weighted, those two terms contributed 1.22 and 1.29 respectively — **2.51 points of a final score of 2.24**, with the other three terms netting out to −0.27. Two ratios that rose because the company was liquidating were carrying more than the whole score.

**Mechanism two: retained earnings is a fossil.** This is the more subtle failure and the more important one.

At 31 December 2011 Kodak reported retained earnings of **positive \$4,071 million** and total shareholders' equity of **negative \$2,352 million**. Both numbers are correct and both are on the same balance sheet. They differ because retained earnings is only one component of equity; the others — accumulated other comprehensive losses, largely from pension obligations, and treasury stock, the cumulative cost of shares the company had bought back over decades — were together far larger and negative.

X2 was reading \$4.07 billion of "profits kept in the business" from a business whose owners' stake was worth negative \$2.35 billion. The profits were real. They were earned in the 1980s and 1990s. They had long since left, spent on share repurchases and absorbed by pension deficits. The account remembers the earning; it does not record the spending in the same place.

This is the same structural blindness that runs through the material on [what the balance sheet hides](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here) and on [stock-based compensation and buyback optics](/blog/trading/forensic-accounting/stock-based-compensation-buybacks-and-eps-optics): a cumulative account and a current-value account can tell opposite stories about the same firm, and a formula that reads only one of them will believe whichever it was handed.

#### Worked example: Kodak's Z-score, term by term, at 31 December 2011

Here is the arithmetic in full, using the figures from the table above and the fourth-quarter low price of \$0.62 with 271,415,654 shares outstanding.

Market value of equity = 271,415,654 × \$0.62 ≈ **\$168 million**.

The five ratios:

1. X1 = 553 / 4,678 = **0.1182**
2. X2 = 4,071 / 4,678 = **0.8702**
3. X3 = −600 / 4,678 = **−0.1283**
4. X4 = 168 / 7,028 = **0.0239**
5. X5 = 6,022 / 4,678 = **1.2873**

The five contributions:

- 1.2 × 0.1182 = **+0.142**
- 1.4 × 0.8702 = **+1.218**
- 3.3 × (−0.1283) = **−0.423**
- 0.6 × 0.0239 = **+0.014**
- 1.0 × 1.2873 = **+1.287**

Sum: 0.142 + 1.218 − 0.423 + 0.014 + 1.287 = **2.24**.

Now look at what is holding it up. The retained-earnings term contributes +1.218 and the asset-turnover term +1.287. Between them, **2.51 points of a 2.24 score** — the other three terms net out to −0.27. If you replaced the fossil retained-earnings figure with the actual book equity of −\$2,352 million, X2 would be −0.503 and its contribution −0.704, dragging the score to roughly **0.32**, deep in the distress zone.

The market, incidentally, was not fooled. X4 had fallen to 0.024, meaning the entire equity of a company with \$7 billion of liabilities was being priced at \$168 million. But X4 carries the smallest weight in the model, so the one input that had correctly concluded the company was finished contributed **+0.014** to the score.

**The intuition this example teaches:** the Z-score is a weighted average, and a weighted average can be dominated by the terms that happen to be large. When the informative input carries a small weight and the misleading input carries a large one, the total is worse than the best single indicator you already had.

### Was the model wrong, exactly?

It is worth being precise rather than dramatic. The Z-score never said Kodak was safe. From 2007 onward it sat in the grey zone, which Altman explicitly defined as the region where the model cannot classify. An analyst using the model correctly — as a trigger for investigation rather than a verdict — would have been investigating Kodak every year from 2007.

What the model failed to do was *deteriorate*. The most valuable property of a distress score is its slope, and here the slope was the wrong way. Anyone watching the trend rather than the level would have been actively reassured.

And the audit report was no help either. Kodak's Form 10-K for 2011 does carry going-concern language — the financial statements state that the bankruptcy filing "raises substantial doubt about the Company's ability to continue as a going concern." But that filing was made on 29 February 2012, six weeks **after** the Chapter 11 petition. The going-concern flag was a description of an event that had already happened, not a warning about one that had not. That timing is the general case rather than the exception, and it is the subject of the piece on [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch).

## 7. Where the model breaks

Kodak is one failure mode. Here are the others, and the structural reason for each.

![Three balance-sheet shapes — a 1960s manufacturer, a bank and an asset-light software firm — showing why the ratios misfire](/imgs/blogs/the-altman-z-score-predicting-financial-distress-9.webp)

### Banks and financial institutions: do not use it at all

This is not a caveat, it is an exclusion. Altman's sample was manufacturers, and financial firms were left out by construction.

The reasons are structural. A bank's balance sheet is almost entirely financial claims on both sides: loans and securities as assets, deposits and borrowings as liabilities. Leverage of ten to twenty times equity is normal and healthy for a regulated bank, and would signal terminal distress in a manufacturer. The current-versus-non-current distinction that X1 depends on is barely meaningful when a deposit is repayable on demand but is in practice the stickiest funding the bank has. And X5, sales over assets, has no natural interpretation when "sales" is net interest income.

The relevant analysis for a bank is a completely different discipline — regulatory capital ratios, liquidity coverage, funding concentration, asset quality. The [Lehman Brothers case](/blog/trading/finance/lehman-brothers-2008-financial-crisis) is the canonical illustration of a failure whose mechanics are invisible to every one of the Z-score's five inputs.

### Asset-light firms: the ratios misfire in the same direction

Consider a growing enterprise-software business. Its most valuable asset is the software, but under US and international accounting rules most research and development is **expensed as incurred**, so the thing that generates all the revenue is not on the balance sheet at all. Total assets is therefore small and consists mostly of cash.

Three consequences follow, and all three push the score down on a healthy company:

- **X1 goes negative by design.** Annual contracts billed in advance create **deferred revenue** — cash received for service not yet delivered — which is a current liability. A fast-growing subscription business has a large and rising deferred revenue balance and therefore negative working capital precisely when it is doing well.
- **X2 is negative or trivial.** Years of expensed R&D produce an accumulated deficit. A company that has never reported a GAAP profit has a large negative X2, and X2 carries the second-largest weight.
- **X3 is negative for the same reason.** Expensing the investment that creates the future asset makes current EBIT negative.

Meanwhile X5 can be misleadingly high, because the tiny asset base flatters turnover. The net result is a distress reading on a company with no debt, growing revenue and a decade of runway. This is the single most common false positive the model produces today, and it is entirely a consequence of applying a formula fitted on firms whose value was in their factories to firms whose value is in their code.

### The lease problem, and why the model's inputs are moving targets

Until accounting rules changed — IFRS 16 from 2019, and ASC 842 for US filers from 2019 for public companies — most operating leases sat off the balance sheet entirely. A retailer with hundreds of leased stores carried none of that obligation in total liabilities.

When those rules took effect, the same retailer suddenly reported a large right-of-use **asset** and a matching lease **liability**. Total assets went up, total liabilities went up, and every one of the five ratios moved — with no change whatsoever in the underlying business. X4 fell because its denominator grew. X2, X3 and X5 all fell because their denominator grew.

This is worth stating clearly because it is easy to miss: **a Z-score time series that spans an accounting-standard change is not a like-for-like series.** The same is true of any period in which a company changed how much it capitalised, consolidated a previously off-balance-sheet vehicle, or restated. The mechanics of exactly what moves on and off the sheet are covered in the pieces on [off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities) and [hidden liabilities](/blog/trading/forensic-accounting/hidden-liabilities-leases-guarantees-and-contingencies).

### The inputs are exactly the numbers a manipulator controls

This is the deepest problem for a forensic reader, and it is not a statistical objection at all.

Every input to the Z-score comes from the accounts. If the accounts are wrong, the score is wrong, and it will be wrong in the flattering direction, because that is the direction manipulation runs:

- **Capitalising costs that should be expensed** — the WorldCom mechanism — raises total assets and raises EBIT simultaneously. X3's numerator goes up, and so does everyone's denominator. The effect on the score is generally positive.
- **Recognising revenue early** raises sales (X5) and receivables (current assets, so X1).
- **Inventory overstatement** raises current assets and therefore working capital, lifting X1, while also lifting EBIT by understating cost of goods sold.
- **Moving debt off the balance sheet** shrinks total liabilities, lifting X4.

A company that is actively falsifying its statements will produce a *better* Z-score than the same company reporting honestly. The model has no defence against this because it has no independent source of truth. It reads what it is given.

This is why in practice the Z-score is run alongside a manipulation screen rather than instead of one, and why the series covers [accrual accounting versus cash](/blog/trading/forensic-accounting/accrual-accounting-versus-cash-the-gap-fraud-exploits), [inventory and receivables inflation](/blog/trading/forensic-accounting/inventory-and-receivables-inflation-the-classic-red-flag), [the accruals ratio](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly) and the [forensic ratio dashboard](/blog/trading/forensic-accounting/forensic-ratios-dso-dio-dpo-and-margin-anomalies) as separate tools. A firm that screens as both a probable manipulator and a probable bankruptcy is the classic profile of a fraud shortly before it is exposed — because the fraud is usually a failing business hiding its failure.

### The sample is nearly sixty years old

The final limitation is the simplest. The coefficients were fitted on American manufacturers with \$1 million to \$25 million of assets, observed between 1946 and 1965. Since then: the composition of listed markets has shifted decisively away from manufacturing toward services and intangibles; leverage norms have changed several times; the accounting rules governing pensions, leases, goodwill, financial instruments and revenue have all been rewritten; and the bankruptcy code the sample was drawn under has been replaced.

Subsequent research has generally found that the model's accuracy on later periods is lower than the original paper reported, and that re-estimating the coefficients on modern data improves it. That is unsurprising and it is not a scandal. It does mean that the specific numbers 1.81 and 2.99 should be treated as conventions inherited from a particular study, not as thresholds with independent meaning.

## Common misconceptions

**"A Z-score below 1.81 means the company will go bankrupt."** It means the company's five ratios resemble those of the bankrupt firms in a sample of 66 companies observed before 1966. Sears sat below the line for nearly seven years before filing. The score is a statement about resemblance, not a forecast with a date attached.

**"A Z-score above 2.99 means the company is safe."** It means nothing was found in five ratios computed from the company's own reported statements. It cannot see fraud, a covenant that trips next quarter, a customer concentration, a lawsuit, a refinancing wall, or a bank that changes its mind. Kodak never printed a distress reading and filed anyway.

**"The weights tell you which ratio matters most."** They do not. The weights compensate for the different scales the ratios naturally occupy. EBIT-to-assets has the largest coefficient because its spread across firms is narrow, not because profitability is the most important thing about a company.

**"Z, Z-prime and Z-double-prime are the same model with small adjustments."** They are three separate fits. Every coefficient differs and every cutoff differs. On Kodak's 2011 balance sheet they produced 2.24, 1.57 and 2.40 — against distress thresholds of 1.81, 1.23 and 1.1 respectively. Quoting a Z-score without saying which model and which cutoffs is like quoting a temperature without saying which scale.

**"You can use it on any company."** Financial institutions are excluded by construction. Asset-light and pre-profit companies produce systematic false positives. Firms mid-way through an accounting-standard change produce time series that are not comparable with themselves.

**"Retained earnings measures the cushion the company has."** It measures the cumulative profit that was, at some point in history, not paid out as a dividend. It says nothing about whether that money is still inside the business. Kodak reported \$4.07 billion of retained earnings and negative \$2.35 billion of shareholders' equity on the same balance sheet.

## How it shows up in real markets

### 1. The credit analyst's first ten minutes

A name lands on a desk. Before reading a word of the annual report, the analyst pulls the five inputs, computes the score under the appropriate variant, and — more importantly — computes it for the previous four years. The purpose is not to reach a conclusion. It is to decide **where to spend the afternoon**. A score that fell 0.8 points on a collapsing X3 sends you to the segment disclosures. A score that fell on X4 alone sends you to the news flow and the short interest. This triage function, rather than prediction, is what keeps the model in daily use.

### 2. Loan covenants and the pricing grid

Banks rarely put a Z-score directly into a credit agreement, but they routinely put its components in: minimum working capital, minimum tangible net worth, maximum leverage, minimum interest coverage. A borrower whose Z-score is sliding is usually a borrower approaching a covenant test, and a covenant breach is one of the standard mechanisms by which a slow deterioration becomes a sudden liquidity event. The score is a rough proxy for how much room is left before somebody else gets a say in the company's decisions.

### 3. Supplier and customer credit decisions

The largest population of Z-score users is not investors at all. It is trade-credit and factoring departments deciding whether to ship goods on 60-day terms to a customer they cannot investigate in depth. Here the Type II error — refusing a healthy customer — has an immediate, measurable cost in lost margin, while the Type I error costs the whole receivable. The asymmetry drives where they set the threshold, and it is why commercial credit scoring generally uses cutoffs tuned to a portfolio's own loss experience rather than 1.81.

### 4. Sears Holdings, 2012 to 2018

Covered in full above, and the cleanest available illustration of the model working: seven consecutive fiscal year-ends, six of them lower than the one before, the score below the distress threshold from the very first observation, and a Chapter 11 filing on 15 October 2018. It also illustrates the model's chief practical weakness — it says "this looks like a company that fails," and it has nothing whatsoever to say about when. Six years of asset sales, spin-offs and related-party financing intervened.

### 5. Eastman Kodak, 2007 to 2012

Also covered in full above, and the reason a forensic reader should never treat the score as a verdict. A listed manufacturer, exactly the fitted population, whose Z-score *improved* into a Chapter 11 filing on 19 January 2012 because total assets fell by two-thirds and because a positive retained-earnings balance recorded profits earned and spent decades earlier. The lesson generalises: whenever a ratio improves during a period in which the company is liquidating, check whether the numerator or the denominator moved.

### 6. The 2019 lease-accounting reset

When IFRS 16 and ASC 842 brought operating leases onto balance sheets, thousands of retailers, restaurant chains and airlines reported large new assets and matching liabilities on unchanged businesses. Screens that compared a company's post-2019 Z-score against its own pre-2019 history generated a wave of apparent deteriorations that were pure accounting. Anyone running a distress screen across that boundary has to either restate the earlier years or accept a discontinuity, and this is a live problem for any long time series today.

## When this matters to you

If you own shares, lend money, extend trade credit, or work at a company whose survival matters to you, the Z-score is worth ten minutes because of what it forces you to do rather than what it tells you. Pulling the five inputs means reading the balance sheet and the income statement properly, and computing four years of them means noticing what changed. Most of the value is in that process.

Three habits are worth carrying away:

**Compute the trend, not the level.** One score is nearly meaningless. Four consecutive scores, with the five contributions broken out, is a genuine piece of analysis.

**Always decompose.** When the number moves, find out which term moved it. A score that fell because of X4 is the market's opinion; a score that fell because of X3 is the business. They lead to different questions.

**Ask what the denominator did.** If total assets is falling fast, four of the five ratios are being flattered. This is the trap Kodak illustrates, and it is invisible unless you look at the raw lines alongside the ratios.

And the honest limitation to hold alongside all of it: this is a screen fitted on 66 companies from the middle of the last century. It reads only what the company chose to report. It has no opinion about anything off the statements — which, in a series about how companies cook their books, is exactly where the interesting things live.

*This article is educational and is not investment advice. Nothing here is a recommendation to buy or sell any security.*

## Sources & further reading

**The primary academic work**

- Edward I. Altman, "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy," *The Journal of Finance*, Vol. 23, No. 4 (September 1968), pp. 589–609. The original model: the five ratios, the discriminant coefficients (1.2, 1.4, 3.3, 0.6 and 1.0 in the decimal convention), the 33-plus-33 matched sample of manufacturers filing under the US National Bankruptcy Act between 1946 and 1965, the asset range of roughly \$1m–\$25m, the reported initial-test accuracy of 72% two years before the event with a Type II error of 6%, the optimal cutoff of 2.675 and the 1.81-to-2.99 zone of ignorance. The approximately 80–90% one-year-ahead accuracy with 15–20% false positives quoted above comes from Altman's later out-of-sample tests over three periods running to 1999, summarised in the 2000 NYU Stern paper below.
- Edward I. Altman, "Predicting Financial Distress of Companies: Revisiting the Z-Score and ZETA Models," NYU Stern working paper (2000). Altman's own restatement of the original model, the Z-prime revision for private firms (coefficients 0.717, 0.847, 3.107, 0.420, 0.998; cutoffs 1.23 and 2.90), and the four-variable Z-double-prime for non-manufacturers (coefficients 6.56, 3.26, 6.72, 1.05; cutoffs 1.1 and 2.6).
- Edward I. Altman, John Hartzell and Matthew Peck, "Emerging Market Corporate Bonds: A Scoring System" (1995), in which the four-variable model is used with an added constant of 3.25 to align the resulting score with US bond-rating equivalents.
- Edward I. Altman, *Corporate Financial Distress and Bankruptcy* (Wiley; editions from 1983 onward) — the book-length treatment where the Z-prime and Z-double-prime models and their cutoffs are set out in full.
- Edward I. Altman, Robert Haldeman and Paul Narayanan, "ZETA Analysis: A New Model to Identify Bankruptcy Risk of Corporations," *Journal of Banking and Finance* (1977) — the seven-variable commercial successor, whose exact coefficients were not published.

**Neighbouring models, for context**

- James A. Ohlson, "Financial Ratios and the Probabilistic Prediction of Bankruptcy," *Journal of Accounting Research* (1980) — the O-score, a logit model that outputs a probability rather than a discriminant value.
- Mark E. Zmijewski, "Methodological Issues Related to the Estimation of Financial Distress Prediction Models," *Journal of Accounting Research* (1984) — a probit model, and the standard critique of the choice-based sampling used by matched-pair studies including Altman's.
- Robert C. Merton, "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates," *The Journal of Finance* (1974) — the structural, market-price-based alternative that underlies distance-to-default measures.
- Messod D. Beneish, "The Detection of Earnings Manipulation," *Financial Analysts Journal* (1999) — the M-score, which asks the different question of whether the statements themselves are being manipulated.

**Company filings behind every figure in this article**

- Eastman Kodak Company, Form 10-K for the year ended 31 December 2008 (SEC EDGAR, CIK 0000031235) — the 2007 and 2008 balance sheets (total assets \$13,659m and \$9,179m; total current assets \$6,053m and \$5,004m; total current liabilities \$4,446m and \$3,462m; total liabilities \$10,630m and \$8,218m; retained earnings \$6,474m and \$5,879m), net sales of \$10,301m and \$9,416m, the loss from continuing operations before interest expense, other income (charges), net and income taxes of \$230m and \$821m, the Item 5 market price table (2008 fourth quarter \$5.83–\$15.68; 2007 fourth quarter \$21.42–\$29.60), and the cover-page non-affiliate market value of approximately \$4.2 billion as of 30 June 2008.
- Eastman Kodak Company, Form 10-K for the year ended 31 December 2010, filed 25 February 2011 (accession 0000031235-11-000025) — the 2009 and 2010 figures used above, the Item 5 market price table (2010 fourth quarter \$3.84–\$5.95; 2009 fourth quarter \$3.26–\$4.74), 268,882,900 shares outstanding as of 11 February 2011, non-affiliate market value of approximately \$1.2 billion as of 30 June 2010, and the disclosure that the Board suspended future cash dividends effective 30 April 2009.
- Eastman Kodak Company, Form 10-K for the year ended 31 December 2011, filed 29 February 2012 (accession 0000031235-12-000036) — total assets \$4,678m, total current assets \$2,703m, total current liabilities \$2,150m, total liabilities \$7,028m, retained earnings \$4,071m, total shareholders' equity −\$2,352m, net sales \$6,022m, the \$600m loss before interest, other income and taxes; the Item 5 market price table (2011 fourth quarter \$0.62–\$1.63) and the note that the stock was delisted from the NYSE in January 2012 with a highest reported bid of \$0.36 on 28 February 2012; 271,415,654 shares outstanding as of 17 February 2012; non-affiliate market value of approximately \$963 million as of 30 June 2011; the chapter 11 filing date of 19 January 2012; and the statement that the bankruptcy filing "raises substantial doubt about the Company's ability to continue as a going concern."
- Sears Holdings Corporation, Forms 10-K for the fiscal years ended 28 January 2012 through 3 February 2018 (SEC EDGAR / XBRL company facts, CIK 0001310067) — the total assets, current assets, current liabilities, total liabilities, retained earnings (accumulated deficit), total shareholders' equity and operating income figures tabulated above.

**Related reading on this blog**

- [Reading the balance sheet: what companies hide here](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here)
- [Hidden liabilities: leases, guarantees and contingencies](/blog/trading/forensic-accounting/hidden-liabilities-leases-guarantees-and-contingencies)
- [The cash conversion cycle and what working capital reveals](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals)
- [Liquidity and solvency: can the company survive?](/blog/trading/equity-research/liquidity-and-solvency-can-the-company-survive)
- [Forensic accounting: spotting manipulation and fraud](/blog/trading/equity-research/forensic-accounting-spotting-manipulation-and-fraud)
- [The accruals ratio and the accruals anomaly](/blog/trading/forensic-accounting/the-accruals-ratio-and-the-accruals-anomaly)
- [Forensic ratios: DSO, DIO, DPO and margin anomalies](/blog/trading/forensic-accounting/forensic-ratios-dso-dio-dpo-and-margin-anomalies)
- [Credit and distressed debt valuation: recovery rates](/blog/trading/asset-valuation/credit-distressed-debt-valuation-recovery-rates)
