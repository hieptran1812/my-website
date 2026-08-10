---
title: "Benford's law and digit analysis for fraud: why fabricated numbers have the wrong fingerprint"
date: "2026-08-09"
publishDate: "2026-08-09"
description: "A beginner-friendly forensic guide to leading-digit analysis—why real financial figures start with 1 about 30% of the time, how the first-digit, second-digit, first-two-digits and last-two-digits tests actually work, and why non-conformity is a flag for investigation and never proof of fraud."
tags: ["forensic-accounting", "benfords-law", "digit-analysis", "fraud-detection", "statistics", "auditing", "data-analytics", "financial-statements", "chi-square", "earnings-quality"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 53
---

> [!important]
> **TL;DR** — Numbers that come from the real world start with the digit 1 about 30% of the time and with 9 only about 4.6% of the time. Numbers that come out of a human imagination do not. That gap is a fraud-detection tool.
>
> - The pattern is called Benford's law. For a leading digit $d$, the expected share is $P(d) = \log_{10}(1 + 1/d)$ — which gives about 30.1% for 1, and about 4.6% for 9. Digit 1 leads roughly 6.6 times as often as digit 9.
> - It holds when data spans several orders of magnitude and grows multiplicatively. It fails for assigned numbers (prices, invoice IDs, ZIP codes), tightly bounded ranges, and small samples.
> - Forensic accountants run four tests: first-digit, second-digit, first-two-digits (the workhorse), and last-two-digits (which tests for *uniformity*, not Benford, and catches invented and duplicated amounts).
> - The single most important practical lesson in this post: **aggregate statistics routinely pass while one bin screams.** In the worked example below, a file containing 90 fraudulent purchase orders passes both the first-digit test and the overall first-two-digits chi-square — and the per-bin z-statistic for a single bin comes out at 8.12.
> - Non-conformity is a **flag telling you where to look**. It is never, on its own, proof of fraud. Every number in the worked examples here is synthetic and clearly labelled as such.

Ask someone to guess how often a big pile of real-world numbers — every invoice a company paid last year, the population of every town in a country, the market capitalisation of every listed firm — begins with the digit 1.

Almost everyone says the same thing: about one in nine, so roughly 11%. There are nine possible leading digits, 1 through 9, and no obvious reason for any of them to be special.

That answer is wrong, and it is wrong by a lot. The real figure is about 30%. And the error is not a small statistical curiosity — it is large enough, stable enough, and predictable enough that auditors, tax authorities and forensic accountants use it to decide which transactions to pull for investigation.

![Bar chart comparing Benford's expected leading-digit frequencies against a flat 11.1% uniform guess](/imgs/blogs/benfords-law-and-digit-analysis-for-fraud-1.webp)

The chart above is the mental model for this entire post. The bars are what real-world data actually does. The dashed line at 11.1% is what almost everyone assumes it does — and, crucially, it is much closer to what a person *invents* when they sit down to fabricate a number. The gap between the bars and the line is the whole detection technique.

Let us build it from nothing.

## First principles: what a leading digit is, and why it is not uniform

Before any statistics, we need to be precise about the object we are counting.

The **leading digit** — also called the *first significant digit* — is the first non-zero digit of a number, reading left to right. It ignores the decimal point entirely and it ignores leading zeros. So:

| Number | Leading digit | Second digit | First two digits |
| --- | --- | --- | --- |
| 4,827.19 | 4 | 8 | 48 |
| 0.00391 | 3 | 9 | 39 |
| 91 | 9 | 1 | 91 |
| 1,000,000 | 1 | 0 | 10 |
| 0.7 | 7 | — | — |

Two things to notice. First, the leading digit is never 0 — by definition, since we skip leading zeros. That is why there are nine possible leading digits, not ten. Second, scale is irrelevant: $4{,}827.19$ dollars and ${4.82719}$ million dollars have the same leading digit. That second observation turns out to be the deep reason the law exists, and we will come back to it.

A **distribution**, in this context, just means "the share of the data falling into each category". If we have 1,000 invoices and 301 of them start with a 1, the observed share for digit 1 is 0.301, or 30.1%.

### The naive guess, and why it feels right

The uniform guess — 1/9 for each digit, about 11.1% — comes from a reasonable-sounding argument: there is nothing special about the digit 1, so why would it appear more often?

The argument fails because it quietly assumes the numbers were *drawn* from a process that treats digits symmetrically, like a lottery machine. Real financial quantities are not drawn that way. They are *grown*. A company's revenue does not get sampled from a hat each year; it compounds from last year's revenue. A price does not get assigned at random; it drifts. And things that grow multiplicatively spend unequal amounts of time in each digit band, for reasons we can make completely concrete in a moment.

#### Worked example: counting leading digits by hand

Take a deliberately small, deliberately mundane set — the first fifteen numbers you would find on a utility bill, a bank statement and a product catalogue mixed together. Here they are:

`142.50`, `1,203.00`, `18.72`, `2,940.11`, `376.40`, `1,055.25`, `89.99`, `3,214.00`, `27.60`, `1,678.30`, `445.12`, `12.05`, `9,320.75`, `230.88`, `1,940.00`

Count the leading digits:

| Leading digit | Tally | Count | Observed share |
| --- | --- | --- | --- |
| 1 | 142.50, 1,203.00, 18.72, 1,055.25, 1,678.30, 12.05, 1,940.00 | 7 | 46.7% |
| 2 | 2,940.11, 27.60, 230.88 | 3 | 20.0% |
| 3 | 376.40, 3,214.00 | 2 | 13.3% |
| 4 | 445.12 | 1 | 6.7% |
| 8 | 89.99 | 1 | 6.7% |
| 9 | 9,320.75 | 1 | 6.7% |
| 5, 6, 7 | — | 0 | 0.0% |

Fifteen numbers is far too few to conclude anything — with a sample this small, random noise dominates completely, and we will make that precise later. But even here the shape is visible: seven of fifteen start with 1, and the low digits crowd out the high ones. Nothing about this set was rigged; it is simply what mixed real-world magnitudes look like.

**The intuition:** leading digits are not drawn from a hat, so there is no reason to expect them evenly spread — and in practice they are heavily tilted toward the low end.

## Why the law is true: three intuitions that stack

There is a formula coming, but a formula memorised without intuition is useless in an investigation — you will not know when it applies. So here are three explanations, each deeper than the last.

### Intuition 1: the units argument (scale invariance)

Suppose there really were a universal law describing the leading digits of "naturally occurring" data. Here is a constraint it must satisfy.

Imagine we measure the length of every river in the world in kilometres and record the leading digits. Now imagine a colleague does the same thing in miles. Neither unit is more natural than the other — rivers do not know what a kilometre is. So whatever leading-digit pattern exists, it must look the *same* in both datasets. Multiplying every number in a dataset by 1.609 must leave the digit distribution unchanged.

That requirement is called **scale invariance**, and it is enormously restrictive. It turns out that the *only* digit distribution that survives being multiplied by an arbitrary constant is the logarithmic one. This is not a hand-wave; it is a theorem, established rigorously in the modern treatment by Theodore Hill.

Notice what this argument buys us: it explains why the law shows up in physical constants, river areas and populations, which have nothing to do with each other. They share only the property that their units are arbitrary.

### Intuition 2: the log ruler (multiplicative growth)

This is the version to carry in your head, because it is the one that tells you *when the law applies*.

![Logarithmic ruler from 1 to 10 showing the band from 1 to 2 occupying 30.1% of the length and the band from 9 to 10 occupying 4.6%](/imgs/blogs/benfords-law-and-digit-analysis-for-fraud-2.webp)

Lay out the numbers from 1 to 10 on a **logarithmic** scale — a ruler where equal distances mean equal *ratios*, not equal differences. On such a ruler, the distance from 1 to 2 is $\log_{10}(2) - \log_{10}(1) = 0.301$, while the distance from 9 to 10 is $\log_{10}(10) - \log_{10}(9) = 0.046$.

The band of numbers with leading digit 1 — everything from 1 up to (but not including) 2 — occupies 30.1% of the ruler. The band with leading digit 9 occupies 4.6%. The bands are not equal width. **That unequal width is the law.**

Now add the dynamics. A quantity growing at a constant percentage rate moves across a logarithmic ruler at *constant speed*. So if you photograph such a quantity at random moments, you will catch it in the "leading digit 1" band 30.1% of the time and in the "leading digit 9" band 4.6% of the time — purely because those bands are wider and narrower.

And this is exactly how financial quantities behave. Revenue compounds. Prices compound. Asset values compound. A portfolio does not add a fixed number of dollars per year; it grows by a percentage.

#### Worked example: watching your money cross the digit bands

You invest \$1,000 and it grows at 10% per year. Let us track the leading digit of your balance, year by year:

| Year | Balance | Leading digit |
| --- | --- | --- |
| 0 | \$1,000 | 1 |
| 1 | \$1,100 | 1 |
| 2 | \$1,210 | 1 |
| 3 | \$1,331 | 1 |
| 4 | \$1,464 | 1 |
| 5 | \$1,611 | 1 |
| 6 | \$1,772 | 1 |
| 7 | \$1,949 | 1 |
| 8 | \$2,144 | 2 |
| 9 | \$2,358 | 2 |
| 10 | \$2,594 | 2 |
| 11 | \$2,853 | 2 |
| 12 | \$3,138 | 3 |

Count the years. Your balance spent **eight** years with a leading digit of 1, then **four** years with a leading digit of 2, then moved to 3. Let us keep going in summary: at 10% a year it takes about 7.3 years to double, so the balance would sit in the 1-band (1,000 to 2,000) for roughly 7.3 years, in the 2-band (2,000 to 3,000) for about 4.3 years, in the 3-band for about 3.0 years, and by the time it reaches the 9-band (9,000 to 10,000) it passes through in about 1.1 years.

Divide those durations by the total time to go from 1,000 to 10,000 — about 24.2 years at 10% — and you get 7.3/24.2 = 30.1% for digit 1, and 1.1/24.2 = 4.6% for digit 9.

**The intuition:** your money is not "more likely" to start with 1 in any mystical sense. It simply has further to travel to get out of the 1-band than out of the 9-band, so it lingers there longer.

### Intuition 3: mixing many distributions

The third explanation covers the messiest and most realistic case: a general ledger is not one process, it is hundreds. Payroll, rent, freight, raw materials, utilities, professional fees — each with its own typical size and its own spread.

Hill's 1995 result is that if you repeatedly pick a probability distribution *at random* and then draw samples from it, the leading digits of the pooled sample converge to Benford's law. Benford is, in a precise sense, the distribution of "random samples from random distributions".

This is why the law applies so well to a company's accounts payable file — a grab-bag of unrelated spending processes — and applies poorly to a single tightly-controlled process like a fixed daily allowance.

### The formula

With that groundwork laid, the formula is just the log-ruler argument written down. For a leading digit $d$ from 1 to 9:

$$P(d) = \log_{10}\left(1 + \frac{1}{d}\right) = \log_{10}(d+1) - \log_{10}(d)$$

where $P(d)$ is the expected proportion of numbers whose first significant digit is $d$. Reading it directly: the probability of leading digit $d$ is the *width of the d-band* on a logarithmic ruler.

Evaluating it for each digit:

| Leading digit $d$ | $\log_{10}(1 + 1/d)$ | Expected share |
| --- | --- | --- |
| 1 | $\log_{10}(2/1)$ | 30.10% |
| 2 | $\log_{10}(3/2)$ | 17.61% |
| 3 | $\log_{10}(4/3)$ | 12.49% |
| 4 | $\log_{10}(5/4)$ | 9.69% |
| 5 | $\log_{10}(6/5)$ | 7.92% |
| 6 | $\log_{10}(7/6)$ | 6.69% |
| 7 | $\log_{10}(8/7)$ | 5.80% |
| 8 | $\log_{10}(9/8)$ | 5.12% |
| 9 | $\log_{10}(10/9)$ | 4.58% |

Those nine numbers sum to exactly 1, which is a useful sanity check and also a small piece of elegance: the widths of the nine bands tile the ruler perfectly, because $\log_{10}(2/1) + \log_{10}(3/2) + \cdots + \log_{10}(10/9)$ telescopes to $\log_{10}(10) = 1$.

The ratio worth memorising: 30.10 divided by 4.58 is about **6.6**. In conforming data, leading 1s outnumber leading 9s by more than six to one.

### Where the law came from

The law carries Frank Benford's name, but he was not the first to find it.

Simon Newcomb, an astronomer, published the observation in the *American Journal of Mathematics* in 1881. His route to it is the best origin story in statistics: he noticed that the printed logarithm tables in his library were visibly more worn on the early pages than the later ones. Colleagues were looking up numbers beginning with 1 far more often than numbers beginning with 9. He worked out the logarithmic law from that, and the paper was almost entirely ignored.

Frank Benford, a physicist at General Electric, rediscovered it and published "The Law of Anomalous Numbers" in 1938. What made his paper stick was the sheer breadth of his evidence: he assembled some 20,229 numbers from twenty unrelated tables — the surface areas of rivers, population figures, physical constants, molecular weights, street address numbers, and numbers that simply appeared in the pages of a newspaper — and showed that they all shared approximately the same leading-digit distribution.

That range is the point. Rivers, molecules and newspapers have nothing in common except one thing: the numbers describing them are measured on scales that are arbitrary and span orders of magnitude. Benford's tables are the empirical demonstration of the scale-invariance argument above.

The forensic application came much later. Mark Nigrini's doctoral work and subsequent publications through the 1990s turned the observation into an audit procedure, and it is his framework — the battery of tests and the conformity thresholds — that the rest of this post describes.

### The same idea for later digits

The law is not limited to the first digit. For the second digit $d_2$ (which *can* be 0), you sum over every possible first digit:

$$P(d_2) = \sum_{d_1=1}^{9} \log_{10}\left(1 + \frac{1}{10d_1 + d_2}\right)$$

![Bar chart of Benford second-digit expected frequencies from 11.97% for digit 0 down to 8.50% for digit 9, against a uniform 10% reference line](/imgs/blogs/benfords-law-and-digit-analysis-for-fraud-4.webp)

Working that out gives a much flatter profile than the first digit:

| Second digit | Expected share |
| --- | --- |
| 0 | 11.97% |
| 1 | 11.39% |
| 2 | 10.88% |
| 3 | 10.43% |
| 4 | 10.03% |
| 5 | 9.67% |
| 6 | 9.34% |
| 7 | 9.04% |
| 8 | 8.76% |
| 9 | 8.50% |

The second digit runs from 11.97% down to 8.50% — a gentle slope, not a cliff. That gentleness is precisely what makes it *useful*: because the expected profile is nearly flat, an excess of 0s or 5s stands out immediately, and an excess of 0s and 5s is the signature of **rounding**.

And for the first *two* digits taken together as a number $k$ from 10 to 99, the formula is the same shape:

$$P(k) = \log_{10}\left(1 + \frac{1}{k}\right), \quad k = 10, 11, \ldots, 99$$

This gives 4.14% for the pair 10, and 0.44% for the pair 99. Ninety bins instead of nine, which is what makes it the workhorse test — but also what makes it need far more data.

## What the law does not say: the preconditions

This is the section that separates people who use digit analysis competently from people who embarrass themselves with it. Benford's law is not a law of nature that all numbers obey. It is a property of data generated in particular ways.

![Two-column matrix showing which data types Benford's law applies to and which it does not](/imgs/blogs/benfords-law-and-digit-analysis-for-fraud-8.webp)

Benford conformity should be **expected** when:

- **The data spans several orders of magnitude.** You need numbers in the hundreds, thousands and hundreds of thousands. If everything is between \$40 and \$90, there is no room for the pattern to form.
- **The data results from multiplication or growth**, or from combining several such quantities — prices times quantities, rates times balances.
- **There is no built-in minimum or maximum.** Truncation destroys the pattern.
- **The numbers are not assigned by a human or a system.** They must be *measured* or *computed*, not chosen.
- **The sample is large enough** — as a practical floor, several hundred records for the first-digit test and a few thousand for the first-two-digits test.

Benford conformity should **not** be expected when:

- **Numbers are assigned.** Invoice numbers, purchase order numbers, account numbers, employee IDs, phone numbers, ZIP codes. These are labels that happen to be written with digits. Testing them is meaningless.
- **Numbers are set by human pricing psychology.** A retailer's catalogue full of \$9.99, \$19.99 and \$49.99 will fail a Benford test spectacularly, and the reason is marketing, not fraud.
- **The range is tight.** ATM withdrawals in a market where the machine dispenses \$20 notes with a \$500 daily cap will cluster and will not conform. Hourly wages in a single job grade will not conform.
- **A threshold truncates the data.** Expenses above an approval limit, claims below a deductible, transactions above a reporting threshold — each creates a hard edge that the law does not anticipate.
- **The sample is small.** With 50 records, an apparent "violation" is almost certainly noise.

A useful discipline: before running the test, write down in one sentence *why you expect this particular population to conform*. If you cannot write that sentence, do not run the test — because a non-conforming result will tell you nothing you can act on.

## The four tests, and what each one is for

Forensic practice does not run one test. It runs a sequence, from coarse to fine.

![Table of the four digit tests showing bins, what each detects, and rough sample size needed](/imgs/blogs/benfords-law-and-digit-analysis-for-fraud-3.webp)

**The first-digit test** has 9 bins. It is a high-level look, appropriate as an opening move on a few hundred records or more. Its weakness is bluntness: it aggregates so heavily that a localised problem can vanish into the average. We will demonstrate exactly that failure below.

**The second-digit test** has 10 bins and a nearly flat expected profile. Its specific job is detecting **rounding** — an excess of 0s and 5s in the second position.

**The first-two-digits test** has 90 bins (10 through 99). This is the primary test in forensic work, because 90 bins give enough resolution to localise a problem to a narrow value band — "an unusual number of amounts starting with 48" — which is directly actionable. It needs roughly a thousand records at minimum, and is more comfortable with several thousand.

**The last-two-digits test** has 100 bins (00 through 99) and — this is the part people get wrong — **it is not a Benford test at all.** The final digits of a number that spans orders of magnitude are essentially uniform, so each of the 100 bins is expected at 1.0%. Its purpose is finding invented and duplicated amounts, and detecting rounding at the far end of the number.

A related non-Benford check that belongs in the same toolkit is the **number-duplication test**: simply count how many times each exact amount appears, and sort descending. And the **round-number test**: count amounts that are exact multiples of 100, 1,000 or 500. Neither uses a logarithm; both are extremely effective.

## The statistics: how to tell "a bit off" from "genuinely wrong"

Every observed distribution will differ from the expected one, because of random sampling noise. The statistical question is whether the difference is bigger than noise can explain. Forensic practice uses three measures, and they disagree with each other often enough that you must understand all three.

### Mean absolute deviation (MAD)

The simplest measure. For each bin, take the absolute difference between observed proportion and expected proportion; average those differences across all bins:

$$\text{MAD} = \frac{1}{K} \sum_{i=1}^{K} \left| p_{\text{obs},i} - p_{\text{exp},i} \right|$$

where $K$ is the number of bins (9 for the first digit, 10 for the second, 90 for the first two), $p_{\text{obs},i}$ is the observed proportion in bin $i$, and $p_{\text{exp},i}$ is the Benford expectation for that bin.

MAD has one property that makes it the practitioner's favourite: **it does not depend on sample size.** It measures the *shape* of the discrepancy, not the statistical confidence in it. A file of 1,000 records and a file of 1,000,000 records with the same proportions get the same MAD.

That is also its weakness — it has no built-in notion of significance. So Nigrini supplies empirical cutoffs instead, calibrated by experience rather than derived from a sampling distribution:

![Three horizontal banded scales showing Nigrini's MAD conformity cutoffs for the first-digit, second-digit and first-two-digits tests](/imgs/blogs/benfords-law-and-digit-analysis-for-fraud-6.webp)

| Test | Close conformity | Acceptable | Marginally acceptable | Non-conformity |
| --- | --- | --- | --- | --- |
| First digit | 0.000 – 0.006 | 0.006 – 0.012 | 0.012 – 0.015 | above 0.015 |
| Second digit | 0.000 – 0.008 | 0.008 – 0.010 | 0.010 – 0.012 | above 0.012 |
| First two digits | 0.0000 – 0.0012 | 0.0012 – 0.0018 | 0.0018 – 0.0022 | above 0.0022 |

These bands are conventions, not theorems. They come from Nigrini's published work and are widely used in practice precisely *because* they are stable across sample sizes. Note how much tighter the first-two-digits bands are — with 90 bins, the average deviation per bin is naturally much smaller.

One caveat that trips people up: **the cutoffs are not stable across editions.** The table above is from Nigrini's 2012 book. His earlier 2000 book, *Digital Analysis Using Benford's Law*, circulates a different and stricter set — close conformity below 0.004, acceptable to 0.008, marginally acceptable to 0.012, non-conformity above that — and both sets are still quoted in the literature and baked into software. A first-digit MAD of 0.013 is therefore "marginally acceptable" under one published standard and "non-conforming" under the other.

The practical response is simple and worth making a habit: **report the MAD value itself, name the cutoff table you are judging it against, and let the reader apply their own.** A report that says only "the population does not conform" is not reproducible.

### The chi-square test

The classical statistical approach. Compare observed *counts* against expected *counts*:

$$\chi^2 = \sum_{i=1}^{K} \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ is the observed count in bin $i$ and $E_i = n \cdot p_{\text{exp},i}$ is the expected count given a sample of $n$ records. Compare the result against a chi-square critical value with $K - 1$ degrees of freedom. For the first-digit test, $K - 1 = 8$, and the 5% critical value is 15.507. A computed statistic above that means "reject conformity at the 5% level".

Chi-square gives you a genuine p-value, which MAD does not. But it carries a serious defect in this application, severe enough that Nigrini named it: the **excess power problem**.

#### Worked example: how sample size breaks the chi-square test

Take a synthetic accounts-payable file — call the company Northwind Components. It contains 1,200 invoices, and the leading digits fall as follows. *(This dataset is synthetic and illustrative, constructed to demonstrate the method.)*

| Digit | Observed | Observed % | Expected % | Expected count | Absolute difference | Chi-square term |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 353 | 29.417% | 30.103% | 361.24 | 0.686% | 0.188 |
| 2 | 218 | 18.167% | 17.609% | 211.31 | 0.558% | 0.212 |
| 3 | 155 | 12.917% | 12.494% | 149.93 | 0.423% | 0.172 |
| 4 | 112 | 9.333% | 9.691% | 116.29 | 0.358% | 0.158 |
| 5 | 97 | 8.083% | 7.918% | 95.02 | 0.165% | 0.041 |
| 6 | 78 | 6.500% | 6.695% | 80.34 | 0.195% | 0.068 |
| 7 | 72 | 6.000% | 5.799% | 69.59 | 0.201% | 0.083 |
| 8 | 58 | 4.833% | 5.115% | 61.38 | 0.282% | 0.187 |
| 9 | 57 | 4.750% | 4.576% | 54.91 | 0.174% | 0.080 |
| **Total** | **1,200** | **100%** | **100%** | **1,200** | — | **1.189** |

Let us do one row by hand so nothing is mysterious. For digit 4: the expected proportion is 0.09691, so the expected count is ${0.09691 \times 1200 = 116.29}$. We observed 112. The absolute proportion difference is $|112/1200 - 0.09691| = |0.09333 - 0.09691| = 0.00358$, which is the 0.358% in the table. The chi-square contribution is $(112 - 116.29)^2 / 116.29 = 18.4 / 116.29 = 0.158$.

Summing the absolute differences and dividing by 9 gives **MAD = 0.003379** — comfortably inside "close conformity". Summing the chi-square terms gives **1.189** against a critical value of 15.507, for a p-value of 0.997. Both measures agree: this file conforms.

Now here is the problem. Hold those *proportions* exactly fixed and scale the file up:

| Sample size | MAD | Chi-square | Critical value (5%) | Verdict |
| --- | --- | --- | --- | --- |
| 1,200 | 0.003379 | 1.19 | 15.507 | conforms |
| 12,000 | 0.003379 | 11.89 | 15.507 | conforms |
| 120,000 | 0.003379 | 118.85 | 15.507 | **rejected** |
| 1,200,000 | 0.003379 | 1,188.52 | 15.507 | **rejected** |

Chi-square scales linearly with $n$. The *identical* digit profile — a profile any forensic accountant would call clean — gets rejected at 120,000 records and annihilated at 1.2 million. Real corporate datasets are routinely that large.

**The intuition:** chi-square answers "is this deviation bigger than chance?", and with enough data the answer is always yes, because no real dataset is *exactly* Benford. MAD answers "is this deviation big enough to care about?", which is the question an investigator actually has.

### The z-statistic for a single digit

MAD and chi-square are both *omnibus* tests: they summarise the whole distribution into one number. But an investigator does not want a summary — they want to know *which* bin is anomalous, because that is where the transactions are.

For an individual bin, use a z-statistic on the proportion:

$$z = \frac{\left| p_{\text{obs}} - p_{\text{exp}} \right| - \frac{1}{2n}}{\sqrt{\dfrac{p_{\text{exp}}(1 - p_{\text{exp}})}{n}}}$$

where $p_{\text{obs}}$ is the observed proportion in that one bin, $p_{\text{exp}}$ is its Benford expectation, and $n$ is the total sample size. The $\frac{1}{2n}$ in the numerator is a **continuity correction** — a small adjustment because we are approximating a discrete count with a continuous normal curve. It is applied only when it is smaller than the absolute difference itself.

Interpreting it: roughly, $|z|$ above 1.96 is significant at the 5% level and above 2.576 at the 1% level, for a single pre-specified bin.

But there is a trap. If you scan all 90 bins of the first-two-digits test and flag anything above 1.96, you will flag several bins by pure chance — that is what a 5% error rate *means* across 90 opportunities. The standard fix is a **Bonferroni correction**: divide the error rate by the number of bins tested. For 90 bins at an overall 5% level, the per-bin threshold becomes a z of about **3.45** rather than 1.96. For the 100 bins of the last-two-digits test, about **3.48**.

## A full worked test: one clean file, one fabricated file, one that hides

Time to run the whole thing end to end. Every dataset in this section is **synthetic and illustrative** — constructed by the author to demonstrate the method, not measured from any real company's records.

![Grouped bar chart comparing Benford expected frequencies against a conforming synthetic file and a fabricated synthetic file](/imgs/blogs/benfords-law-and-digit-analysis-for-fraud-5.webp)

#### Worked example: a wholly fabricated population

Suppose an investigator isolates the manual journal entries at a company — the adjustments typed in by hand at period end, as opposed to entries generated automatically by the billing system. There are 640 of them. Manual journal entries are a classic focus for fraud work because they are where a person, rather than a system, chooses the number.

Here is the first-digit test. *(Synthetic, illustrative data.)*

| Digit | Observed | Observed % | Expected % | Expected count | Absolute difference | Chi-square term |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 96 | 15.000% | 30.103% | 192.66 | 15.103% | 48.495 |
| 2 | 82 | 12.812% | 17.609% | 112.70 | 4.797% | 8.362 |
| 3 | 78 | 12.188% | 12.494% | 79.96 | 0.306% | 0.048 |
| 4 | 74 | 11.562% | 9.691% | 62.02 | 1.871% | 2.313 |
| 5 | 80 | 12.500% | 7.918% | 50.68 | 4.582% | 16.969 |
| 6 | 72 | 11.250% | 6.695% | 42.85 | 4.555% | 19.838 |
| 7 | 66 | 10.312% | 5.799% | 37.11 | 4.513% | 22.480 |
| 8 | 52 | 8.125% | 5.115% | 32.74 | 3.010% | 11.334 |
| 9 | 40 | 6.250% | 4.576% | 29.28 | 1.674% | 3.921 |
| **Total** | **640** | **100%** | **100%** | **640** | — | **133.759** |

**MAD = 0.044902.** The non-conformity threshold for the first-digit test is 0.015, so this is three times past it.

**Chi-square = 133.759** on 8 degrees of freedom, against a 5% critical value of 15.507 and a 1% critical value of 20.090. The p-value is about ${5 \times 10^{-25}}$.

Look at the *shape* of the failure, because the shape is the tell. Digit 1 is at 15.0% against an expected 30.1% — barely half. Digits 5, 6 and 7 are each running at roughly 11–12% against expectations of 7.9%, 6.7% and 5.8%. The whole distribution has been flattened and pushed toward the middle. The observed ratio of 1s to 9s is 96/40 = 2.4, against the 6.6 that conforming data produces.

That flattening is the fingerprint of human invention, and the next section explains why it happens.

**The intuition:** when a person makes numbers up, they spread them out far too evenly and avoid the low digits that reality favours.

#### Worked example: the file where every aggregate test passes and one bin screams

This is the most important example in the post, and it is the one that changes how you use the tool.

A synthetic purchase-order file — call it Meridian Logistics — contains 6,400 purchase orders. Hidden inside it are 90 fraudulent orders, written by a manager who knows that orders of \$50,000 and above require a second signature, and who has therefore been writing everything just underneath: amounts like \$48,200, \$48,750, \$49,100. *(Synthetic, illustrative data.)*

**Test 1 — the first-digit test.** Collapsing all 6,400 records to their leading digit:

| Digit | Observed | Observed % | Expected % | Expected count | Chi-square term |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,914 | 29.906% | 30.103% | 1,926.59 | 0.082 |
| 2 | 1,131 | 17.672% | 17.609% | 1,126.98 | 0.014 |
| 3 | 787 | 12.297% | 12.494% | 799.61 | 0.199 |
| 4 | 698 | 10.906% | 9.691% | 620.22 | 9.753 |
| 5 | 498 | 7.781% | 7.918% | 506.76 | 0.151 |
| 6 | 417 | 6.516% | 6.695% | 428.46 | 0.306 |
| 7 | 355 | 5.547% | 5.799% | 371.15 | 0.703 |
| 8 | 317 | 4.953% | 5.115% | 327.38 | 0.329 |
| 9 | 283 | 4.422% | 4.576% | 292.85 | 0.331 |
| **Total** | **6,400** | **100%** | **100%** | **6,400** | **11.869** |

**MAD = 0.002840** — inside "close conformity" (below 0.006). **Chi-square = 11.869** against a critical value of 15.507, p-value 0.157. **The first-digit test passes.** The file looks clean.

Digit 4 is mildly elevated at 10.906% against 9.691%, contributing almost all of the chi-square statistic — but on its own that is nowhere near enough to raise an eyebrow.

**Test 2 — the overall first-two-digits test.** Ninety bins now. Computing MAD across all 90:

**MAD = 0.000414**, against a non-conformity threshold of 0.0022 — this is *deep* inside close conformity. **Chi-square = 86.891** on 89 degrees of freedom, against a 5% critical value of 112.022. The p-value is 0.54. **The overall first-two-digits test also passes.**

At this point a careless analyst writes "the population conforms to Benford's law; no further work performed" and closes the file. There are 90 fraudulent purchase orders in it.

**Test 3 — the per-bin z-statistics.** Now look at individual bins instead of the aggregate:

| First two digits | Observed | Expected | z-statistic |
| --- | --- | --- | --- |
| 10 | 265 | 264.91 | 0.01 |
| 47 | 57 | 58.52 | 0.13 |
| **48** | **119** | **57.31** | **8.12** |
| **49** | **83** | **56.15** | **3.53** |
| 50 | 56 | 55.04 | 0.06 |
| 51 | 54 | 53.97 | 0.00 |

Bin 48 has a z of **8.12**. Bin 49 has a z of **3.53**. Both clear the Bonferroni-corrected threshold of 3.45 for 90 simultaneous bins. Every other bin in the file is unremarkable — the largest remaining z is about 1.2.

Let us verify bin 48 by hand. Its Benford expectation is $P(48) = \log_{10}(1 + 1/48) = \log_{10}(49/48) = 0.008955$, so the expected count is ${0.008955 \times 6400 = 57.31}$. We observed 119, an observed proportion of ${119/6400 = 0.018594}$. The absolute difference is ${0.018594 - 0.008955 = 0.009639}$. The continuity correction is ${1/(2 \times 6400) = 0.000078}$. The standard error is $\sqrt{0.008955 \times 0.991045 / 6400} = 0.001178$. So $z = (0.009639 - 0.000078)/0.001178 = 8.12$.

A z of 8.12 is not a marginal call. Under the null hypothesis of conformity, a deviation that large in a pre-specified bin has a probability of roughly one in a hundred billion.

And note what the result hands the investigator: not "this company might be committing fraud", but "**pull every purchase order between \$48,000 and \$49,999**". That is a list of about 200 documents, of which 90 are the fraudulent ones. It is a morning's work to review them.

**The intuition:** omnibus statistics average anomalies away. A fraud concentrated in one narrow value band can leave the whole-file MAD and chi-square looking pristine while a single bin sits eight standard errors from expectation. Always look at the bins.

Note also what the elevated digit-4 count in Test 1 was: it was this same fraud, smeared across the entire 40–49 range and diluted to statistical invisibility. The signal was always there; the first-digit test simply lacked the resolution to see it.

#### Worked example: the second-digit test finds rounding

Different population, same company family: 2,800 employee expense claims. The first-digit test on this file is unremarkable, so we go to the second digit, whose expected profile is nearly flat and therefore sensitive to rounding. *(Synthetic, illustrative data.)*

| Second digit | Observed | Observed % | Expected % | Expected count | Absolute difference | Chi-square term |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 412 | 14.714% | 11.968% | 335.10 | 2.746% | 17.646 |
| 1 | 305 | 10.893% | 11.389% | 318.89 | 0.496% | 0.605 |
| 2 | 288 | 10.286% | 10.882% | 304.70 | 0.596% | 0.915 |
| 3 | 279 | 9.964% | 10.433% | 292.12 | 0.469% | 0.590 |
| 4 | 266 | 9.500% | 10.031% | 280.86 | 0.531% | 0.787 |
| 5 | 340 | 12.143% | 9.668% | 270.70 | 2.475% | 17.743 |
| 6 | 248 | 8.857% | 9.337% | 261.45 | 0.480% | 0.692 |
| 7 | 236 | 8.429% | 9.035% | 252.99 | 0.607% | 1.140 |
| 8 | 229 | 8.179% | 8.757% | 245.20 | 0.578% | 1.070 |
| 9 | 197 | 7.036% | 8.500% | 237.99 | 1.464% | 7.061 |
| **Total** | **2,800** | **100%** | **100%** | **2,800** | — | **48.249** |

**MAD = 0.010443.** On the second-digit scale that lands in "marginally acceptable" (0.010 to 0.012) — a shrug, not an alarm.

**Chi-square = 48.249** on 9 degrees of freedom against a 5% critical value of 16.919 and a 1% critical value of 21.666. The p-value is about ${2 \times 10^{-7}}$. That is a strong rejection.

**The per-digit z-statistics**, which resolve the disagreement:

- Second digit 0: observed 14.714% against expected 11.968%, giving $z = 4.45$.
- Second digit 5: observed 12.143% against expected 9.667%, giving $z = 4.40$.
- Second digit 9: observed 7.036% against expected 8.500%, giving $z = 2.74$.

Two bins — 0 and 5 — are each more than four standard errors high, and they are exactly the two digits that rounding produces. Together they are running about 5.2 percentage points above expectation, which on 2,800 claims is roughly 146 excess claims whose second digit is 0 or 5.

Concretely: an excess of second-digit 0 means an excess of amounts like \$40, \$105, \$2,03x; an excess of second-digit 5 means amounts like \$45, \$150, \$2,53x. People are claiming \$50 and \$250 and \$1,500 rather than \$47.30 and \$243.75. Real expenses — a taxi fare, a hotel bill, a meal — rarely land on round numbers.

Notice also that this is the mirror image of the previous example's disagreement. Here MAD says "marginal" and chi-square says "strongly reject". In the Meridian file, both aggregates said "clean" and only the z-statistics fired. **The three statistics answer different questions, and a competent test computes all three.**

**The intuition:** rounding is the easiest fabrication artifact to detect, because it lands on exactly two of the ten second-digit bins.

#### Worked example: last-two-digits, duplication and round numbers

Back to the Meridian purchase-order file, all 6,400 records. These three tests do not use Benford's law at all — the expectation is uniformity. *(Synthetic, illustrative data.)*

**The last-two-digits test.** Each of the 100 bins from 00 to 99 is expected at 1.0%, or ${6400 \times 0.01 = 64}$ records:

| Last two digits | Observed | Expected | z-statistic |
| --- | --- | --- | --- |
| 00 | 191 | 64 | 15.89 |
| 50 | 128 | 64 | 7.98 |
| 95 | 96 | 64 | 3.96 |
| 37 | 71 | 64 | 0.82 |
| 62 | 58 | 64 | 0.69 |
| 13 | 49 | 64 | 1.82 |

Across all 100 bins, **chi-square = 340.34** on 99 degrees of freedom against a 5% critical value of 123.225 — a p-value of about ${3 \times 10^{-28}}$. Bins 00, 50 and 95 all clear the Bonferroni threshold of 3.48 for 100 simultaneous bins.

Bin 00 at three times its expectation means an enormous excess of amounts ending in whole hundreds. Bin 95 is the pricing-psychology artifact: amounts like \$1,495 and \$24,995.

**The round-number test** makes the same point in the units a manager understands:

| Pattern | Expected | Observed | Ratio | z-statistic |
| --- | --- | --- | --- | --- |
| Ends in 00 (whole hundreds) | 64.0 | 191 | 3.0x | 15.89 |
| Ends in 000 (whole thousands) | 6.4 | 74 | 11.6x | 26.54 |
| Ends in 500 or 000 | 12.8 | 119 | 9.3x | 29.57 |

Seventy-four purchase orders for an exact whole number of thousands of dollars, where chance predicts six. Real invoices from real suppliers, with quantities and unit prices and freight and tax, almost never come to \$37,000.00 exactly.

**The number-duplication test** needs no statistics whatsoever — just a count of each distinct amount, sorted descending:

| Amount | Times it appears |
| --- | --- |
| \$4,800.00 | 34 |
| \$48,000.00 | 29 |
| \$4,975.00 | 26 |
| \$1,250.00 | 11 |
| \$873.42 | 2 |

An amount appearing 34 times in a purchase-order file is either a legitimate recurring standing charge — a monthly retainer, a fixed rent — or something worth a question. The investigator's job is to determine which, and that determination is made by looking at documents, not at statistics.

**The intuition:** the cheapest tests in the toolkit are not Benford tests at all. Count duplicates and round numbers first; they need no theory and they catch a great deal.

## Running the tests yourself

None of this requires specialist software. The whole toolkit is about thirty lines of Python, and writing it yourself is the fastest way to stop treating the method as a black box.

```python
import numpy as np
import pandas as pd

def leading_digits(x, n=1):
    """First n significant digits of each value, as an integer."""
    x = np.abs(np.asarray(x, dtype=float))
    x = x[x > 0]                       # zeros and blanks have no leading digit
    mag = np.floor(np.log10(x))        # order of magnitude
    return np.floor(x / 10.0**(mag - (n - 1))).astype(int)

def benford_expected(n=1):
    """Expected proportions for the first n significant digits."""
    lo, hi = 10**(n - 1), 10**n        # n=1 -> 1..9 ; n=2 -> 10..99
    k = np.arange(lo, hi)
    return pd.Series(np.log10(1 + 1 / k), index=k)

def digit_test(x, n=1):
    exp = benford_expected(n)
    obs = pd.Series(leading_digits(x, n)).value_counts().reindex(exp.index, fill_value=0)
    N = obs.sum()
    p_obs, p_exp = obs / N, exp

    mad  = (p_obs - p_exp).abs().mean()
    chi2 = (((obs - N * p_exp) ** 2) / (N * p_exp)).sum()

    se   = np.sqrt(p_exp * (1 - p_exp) / N)
    corr = np.minimum(1 / (2 * N), (p_obs - p_exp).abs())   # continuity correction
    z    = ((p_obs - p_exp).abs() - corr) / se

    table = pd.DataFrame({
        "observed": obs,
        "expected": (N * p_exp).round(2),
        "obs_pct":  (100 * p_obs).round(3),
        "exp_pct":  (100 * p_exp).round(3),
        "z":        z.round(2),
    })
    return table, mad, chi2, N
```

Call `digit_test(amounts, n=1)` for the first-digit test and `digit_test(amounts, n=2)` for the first-two-digits test. Run on the Northwind figures from the worked example above, it returns MAD = 0.003379 and chi-square = 1.189 — the same numbers computed by hand earlier.

The last-two-digits test does not use `benford_expected` at all, because its expectation is uniform:

```python
def last_two_digits_test(x):
    cents = np.round(np.asarray(x, dtype=float) * 100).astype(np.int64)
    bins  = pd.Series(cents % 100)                     # 00..99, on the cents
    obs   = bins.value_counts().reindex(range(100), fill_value=0)
    N, p  = obs.sum(), 0.01                            # uniform: 1% per bin
    se    = np.sqrt(p * (1 - p) / N)
    z     = ((obs / N - p).abs() - np.minimum(1 / (2 * N), (obs / N - p).abs())) / se
    chi2  = (((obs - N * p) ** 2) / (N * p)).sum()
    return pd.DataFrame({"observed": obs, "expected": N * p, "z": z.round(2)}), chi2

def duplication_test(x, top=20):
    return pd.Series(x).value_counts().head(top)

def round_number_test(x):
    v = np.asarray(x, dtype=float)
    return {
        "ends_in_00":  {"observed": int((v % 100 == 0).sum()),   "expected": len(v) * 0.01},
        "ends_in_000": {"observed": int((v % 1000 == 0).sum()),  "expected": len(v) * 0.001},
        "ends_500_000":{"observed": int((v % 500 == 0).sum()),   "expected": len(v) * 0.002},
    }
```

Three practical notes on applying this to real data, each of which has bitten people:

- **Filter before you test, and filter honestly.** Remove zeros, blanks and negatives — or, better, test debits and credits as *separate populations*, since pooling them mixes two different processes. Strip out any assigned-number column that has crept into the extract.
- **Test homogeneous populations separately.** Payroll and freight and capital expenditure have different digit behaviour; pooling them can create an anomaly out of nothing, or hide a real one.
- **Reconcile the extract to the ledger before you believe any result.** If your record count or total does not tie to the trial balance, you are testing a data-extraction bug, not a company. This is the single most common cause of a dramatic-looking failed test.

## Why humans are bad random number generators

Every fabrication artifact above traces to the same source: people are extremely poor at imitating randomness, and they are poor in *consistent, predictable* ways.

Three distinct biases show up in fabricated financial data.

**We over-produce middling digits.** Asked to invent a "random-looking" number, people avoid the extremes. Starting a made-up figure with 1 feels too small and too conspicuous; starting with 9 feels suspiciously close to the next round threshold. So invented numbers drift toward 3, 4, 5, 6 and 7. This is exactly the flattened, middle-heavy profile in the fabricated journal-entries example — digits 5, 6 and 7 running at 11–12% against expectations of 7.9%, 6.7% and 5.8%, while digit 1 collapsed from 30.1% to 15.0%.

**We round.** A person inventing an expense claim writes \$250, not \$247.30. A person inventing a sale writes \$40,000, not \$39,712.55. Fabricated numbers are drawn from the sparse mental lattice of "numbers that sound like numbers", which is dominated by multiples of 5, 10, 25, 50, 100 and 1,000. This produces the second-digit excess of 0s and 5s and the last-two-digit spike at 00 and 50.

**We anchor on thresholds.** Where a control exists — an approval limit, a reporting threshold, a materiality level — people who wish to avoid it cluster immediately beneath it. This is the mechanism behind the bin-48 spike in the Meridian example, and it is why the first-two-digits test is so valuable: a threshold at \$50,000 produces a signature at precisely one or two of the 90 bins.

There is a fourth artifact worth naming because it is the opposite of the first three: sophisticated fabricators sometimes **over-correct**. Knowing that round numbers look suspicious, they use amounts like \$48,237.61 — and then produce *too few* round numbers relative to a real population, because genuine data does contain some legitimately round amounts. A last-two-digits distribution that is *too* uniform is itself anomalous.

This is also why the same digit-analysis toolkit is applied outside accounting — to reported scientific measurements, survey responses, and self-reported administrative statistics. The underlying claim is not about money; it is about the difference between a number that was measured and a number that was chosen.

## The forensic workflow: a flag is not a finding

![Seven-stage left-to-right workflow from extracting the population through investigating specific records, with a callout that non-conformity is a flag and not proof](/imgs/blogs/benfords-law-and-digit-analysis-for-fraud-7.webp)

The statistics are the easy part. The discipline around them is what makes the technique useful rather than embarrassing.

**Stage 1 — Extract the whole population, not a sample.** This is the reverse of ordinary audit sampling, and it matters. Digit analysis is cheap to run on every record, and sampling destroys precisely the localised signal the test exists to find. Ninety anomalous purchase orders in 6,400 will not survive a 200-record sample.

**Stage 2 — Check the preconditions.** Write the one-sentence justification for why this population should conform. Strip out anything assigned, anything bounded, anything below the sample floor. Test each homogeneous population separately rather than pooling incompatible ones.

**Stage 3 — Run the tests coarse to fine.** First digit, then first-two-digits, then second digit, then last-two-digits, duplication and round numbers.

**Stage 4 — Compute all three statistics.** MAD against the published bands, chi-square with its degrees of freedom, and the per-bin z-statistics with a multiple-comparison correction. Never report only the one that agrees with your prior.

**Stage 5 — Identify the specific bins.** The output that matters is not a verdict on the file; it is a value range. "Amounts beginning 48" is actionable. "The file does not conform" is not.

**Stage 6 — Pull those records.** Convert the flagged bins into a document request.

**Stage 7 — Investigate.** Read the documents. Check the approvals, the counterparties, the delivery evidence, the bank records. This is where a finding is actually made.

Everything before Stage 7 is triage. Digit analysis is a way of deciding which of a million transactions deserve a human being's attention. It answers "where should I look?" — and it is genuinely excellent at that question, because the alternative is looking at random.

It does not answer "did someone commit fraud?" and it cannot be made to. There are entirely innocent explanations for almost every anomaly it produces, and the innocent explanations are more common than the guilty ones. A spike just under an approval threshold might be fraud, or it might be a legitimate procurement policy of splitting large orders, or a supplier whose standard contract sits at that price point. A flood of round numbers might be fabrication, or it might be a business that genuinely quotes in round thousands.

The correct sentence in a report is: *"This population shows a statistically significant excess of amounts beginning with 48, which warrants investigation."* The incorrect sentence — and the one that has embarrassed people in court — is: *"Benford analysis proves these transactions are fraudulent."*

This is the same division of labour that runs through all forensic work: analytics narrow the field, documents settle the question. It is worth reading alongside [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch), because digit analysis fills one specific gap in the standard audit — the gap created by sampling.

## Common misconceptions

**"Benford's law applies to all numbers."** It applies to numbers that span orders of magnitude and arise from multiplicative processes. It does not apply to assigned numbers, bounded numbers, or numbers set by human pricing convention. A dataset of shoe sizes, adult heights, ZIP codes or \$9.99 price points will fail the test while being entirely honest. Running the test on such a population is not a weak result — it is a meaningless one, and reporting it as a red flag is malpractice.

**"Failing a Benford test proves fraud."** It proves that the digit distribution differs from a theoretical expectation more than sampling noise easily explains. The list of innocent explanations is long: a change in pricing policy mid-year, a large one-off contract, a systematic rounding convention in a subsidiary's ledger, an ERP migration that reclassified transactions, a business that quotes in round thousands, a threshold in the *legitimate* approval workflow. Non-conformity ranks the population for attention. Nothing more.

**"Passing a Benford test proves there is no fraud."** This one is more dangerous than the previous, because it produces false comfort. The Meridian worked example above contains 90 fraudulent purchase orders and passes both the first-digit test *and* the overall first-two-digits chi-square. Digit analysis has essentially no power against fraud that is small relative to the population, spread evenly across magnitudes, or committed by someone who understands the test. It also has no power at all against fraud that involves no fabricated numbers — a genuine transaction with a related party at a genuine price is invisible to every test in this post. For that you need the techniques in [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing) and [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue).

**"A bigger sample makes the test better."** For MAD and for the per-bin z-statistics, yes. For chi-square, emphatically no — the excess power problem means a large enough sample rejects every real dataset, because no real dataset is exactly logarithmic. If you are working with hundreds of thousands of records, chi-square will tell you "non-conforming" every single time and you will learn nothing. Use MAD for the verdict and z-statistics for the targeting. The general statistical issue here — that significance and importance are different things, and that p-values shrink with sample size regardless of effect size — is covered more fully in [hypothesis testing and p-values](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews).

**"You should run it on the financial statements."** A published income statement contains perhaps 60 numbers. That is nowhere near enough for any of these tests; even the first-digit test wants several hundred. Digit analysis is a tool for transaction-level data — the general ledger, the payables file, the claims database, the trade blotter — not for the summary figures in an annual report. Studies that apply digit methods across *many companies'* filings work because they pool thousands of firm-years into one population, not because 60 numbers can be tested.

**"An anomalous bin tells you the amount of the fraud."** It tells you a value range and an excess count. In the Meridian example the bin-48 excess is roughly 62 records above expectation, and the true number of fraudulent orders is 90 — the test under-counted, because some fraudulent orders landed in bin 49 and some genuine orders landed in bin 48. The excess count is an order-of-magnitude estimate for scoping the review, never a quantification of loss.

**"If the fraudster knows about Benford's law, the technique is useless."** Partly true and worth taking seriously — a fabricator who generates amounts from a logarithmic distribution will defeat the first-digit test. But defeating *all* the tests simultaneously is much harder than it sounds. The amounts must be Benford in the first digit, Benford in the first two digits, uniform in the last two digits, free of duplicates, free of excess round numbers, and consistent with the approval thresholds and supplier patterns in the surrounding data. In practice, people who beat one test fail another, and the over-correction artifact — too *few* round numbers — is itself detectable.

## The limits, stated plainly

Everything above is a technique with a real and bounded domain. Here is the honest accounting of what it cannot do.

**It has no power on small populations.** Below a few hundred records, the sampling noise in each bin swamps any realistic effect. In the fifteen-number example at the top of this post, the observed share for digit 1 was 46.7% against an expected 30.1% — a deviation that looks enormous and means nothing whatsoever. Any result on a small file should be reported as inconclusive, not as a finding.

**It cannot distinguish fraud from any other cause of non-conformity.** Process changes, policy changes, system migrations, acquisitions, seasonality, currency redenomination and simple data-extraction errors all move digit distributions. In practice, data quality problems are a far more common explanation for a failed test than fraud is, and the first thing to check after an anomalous result is whether the extract itself is sound — duplicated rows, a truncated field, a mixed-currency population, or credits and debits pooled together.

**It says nothing about materiality.** A statistically overwhelming pattern can sit in a population of trivial amounts. A z of 8 on a bin of \$40 expense claims is a strong statistical result about a small amount of money.

**Its assumptions are conventions, not proofs.** The MAD cutoff bands are empirical rules of thumb published by a practitioner and adopted by convention. They are not derived from a sampling distribution and they carry no confidence level. Two competent analysts using different thresholds can reach different verdicts on the same file, and neither is doing anything wrong.

**It is easy to abuse in the direction of a conclusion you already hold.** With four tests, three statistics, 90 bins and a choice of population, an analyst who wants to find non-conformity will find it. This is the multiple-comparisons problem wearing a forensic hat, and the Bonferroni correction only addresses the version of it that happens inside a single test. Fix the population and the tests *before* looking at the results, and report every test you ran, not just the one that fired. The discipline is the same one that keeps a market view honest: decide what would change your mind before you look. That habit is the subject of [thinking in probabilities, not predictions](/blog/trading/analyst-edge/thinking-in-probabilities-not-predictions).

**It is a screening tool competing against better-targeted screens.** For most fraud types, a well-chosen ratio or reconciliation beats digit analysis. If you suspect revenue fabrication, the three-way tie-out of revenue against receivables against collected cash is far more powerful — see [revenue recognition games](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold). Digit analysis earns its place when you have a very large population and *no* specific hypothesis about where to look. That is a real and common situation, but it is not every situation.

## How it shows up in real investigations

Everything above this point used synthetic data, because inventing a plausible-looking Benford result for a real company and presenting it as measured is exactly the sin this technique exists to catch. What follows is different: these are documented applications, each traceable to a published source.

### State of Arizona v. Wayne James Nelson (1993)

This is the case that made digit analysis famous in the accounting profession, and Nigrini published the analysis in the *Journal of Accountancy* in May 1999.

Wayne James Nelson was a manager in the office of the Arizona State Treasurer. In *State of Arizona v. Wayne James Nelson* (CV92-18841), decided in 1993, he was found guilty of attempting to defraud the state of nearly \$2 million. The mechanism was ordinary: he diverted funds to a bogus vendor across 23 cheques. His defence — that he had been demonstrating the absence of safeguards in a new computer system — did not persuade the court.

What makes the case a teaching document is the digit profile of those 23 cheques, which Nigrini laid out. **Over 90% of them began with 7, 8 or 9.** Under Benford's law, those three digits together should account for about 15.5% of leading digits: 5.80% plus 5.12% plus 4.58%. Observing over 90% is roughly six times the expectation, in the three digits that conforming data uses *least*.

The mechanism behind that number is not statistical, it is behavioural. The amounts escalated over time — the classic embezzlement pattern of testing the controls with something small and growing bolder — and they clustered just below \$100,000, which is what forces a leading 7, 8 or 9.

Now notice what Nelson got *right*, because it is the most instructive part of the case. His amounts contained no round numbers at all; every one included cents. And no amount was duplicated. Those are precisely the two artifacts an amateur produces, and he avoided both — he was clearly thinking about how fabricated numbers look. He simply had no idea that the *leading* digit was carrying a signal, and his anchoring just below a round \$100,000 threshold made that signal enormous.

**The honest caveat, which matters:** digit analysis did not catch Nelson. He was convicted on other evidence, and Nigrini applied the analysis to the cheque amounts afterward, to demonstrate what a CPA familiar with the method could have spotted. This is a documented illustration on real adjudicated data, not a detection success story. Anyone who tells you Benford's law caught an Arizona embezzler has embellished the record.

### The bank that wrote off credit-card balances just under \$5,000

In the same article, Nigrini describes an audit application with no defendant and no headline, which is arguably more representative of how the technique actually earns its keep.

Bank officers had authority to write off delinquent credit-card balances up to an internal limit of \$5,000. A first-two-digits test on the write-off population produced a spike at the bin **49** — an excess of write-offs in the \$4,900 to \$4,999 range.

That is the same mechanism as the Meridian worked example above, at a different threshold, found in real data: where a control has a numeric limit, the people working around it cluster immediately beneath it, and the first-two-digits test resolves that cluster to a single bin. It is worth sitting with how mundane this is. No one needed a theory of fraud. The test said "look at bin 49", and bin 49 was a policy limit.

### Greece and the EU deficit statistics

Digit analysis is not confined to company ledgers. Rauch, Göttsche, Brähler and Engel applied a Benford test to the macroeconomic data that EU member states report to Eurostat — public deficit, public debt and gross national product, the figures underpinning the deficit criteria — and published the results in the *German Economic Review* in 2011.

Their finding: **the data reported by Greece showed the greatest deviation from Benford's law among the euro-area states.** The paper appeared after Greece's deficit figures had already been revised dramatically upward and the sovereign debt crisis was under way, so this is not a case of the test issuing an early warning.

And the authors were careful in a way that anyone using this technique should copy: a Benford deviation in reported statistics is evidence that the numbers do not behave like unmanipulated data. It is not, by itself, evidence of deliberate falsification, because accounting conventions, estimation methods, revisions and genuine structural features of an economy all move digit distributions too. The result identifies a data series that deserves scrutiny. It does not convict a finance ministry.

### The second digit of reported earnings

The longest-running academic application is not Benford's law in its pure form but its close relative — and it is the one most relevant to reading company accounts.

Carslaw, studying New Zealand firms, published "Anomalies in Income Numbers: Evidence of Goal Oriented Behavior" in *The Accounting Review* in 1988. He found **more 0s than expected in the second digit of reported earnings**. His explanation was cognitive: managers round profit *up* to cross a psychologically salient threshold. A firm that has genuinely earned \$19.7 million would rather report \$20.1 million, because the reader's mind latches onto the leading digit. Pushing 19.7 to 20.1 converts a second digit of 9 into a second digit of 0.

Thomas extended the work to US firms in "Unusual Patterns in Reported Earnings" in *The Accounting Review* in 1989, finding the same pattern in positive earnings — and, revealingly, the opposite pattern in loss-making firms, which is consistent with losses being rounded in the other direction.

This is where digit analysis connects most directly to reading a set of accounts. The effect is a population-level statistical regularity across thousands of firm-years, not a test you can run on one company's income statement. But it is a documented, replicated demonstration that reported earnings carry the fingerprints of the humans who chose them, which is the thesis of this whole series — see [reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings).

### A note on claims you will see repeated

Search for Benford's law and a famous corporate collapse and you will find confident assertions that the company's published figures failed a digit test. Many of these trace to blog posts and lecture slides rather than to a published, reproducible analysis, and they frequently do not state the population tested, the sample size, or the statistic computed.

Treat them the way you would treat any unsourced number. The published literature above is specific about its data and its method; a claim that is not specific about either cannot be checked, and an unverifiable Benford result is worth exactly as much as an unverifiable accounting figure.

## When this matters to you

If you never audit anything, the useful residue of this post is a habit of mind rather than a technique.

**If you work with data.** The first-digit distribution is a free, thirty-second data-quality check. Run it on any large numeric field you have just extracted. It will not usually find fraud; it will regularly find that your extract is truncated, that a field has been padded, that two currencies got pooled, or that a system has been silently rounding. That alone repays learning it.

**If you read financial statements as an investor.** You cannot run these tests on a published annual report — there are not enough numbers. What transfers is the underlying insight: *numbers chosen by people look different from numbers produced by processes*, and the difference shows up in roundness, in clustering just under thresholds, and in repeated amounts. When a company's disclosures are full of suspiciously round figures, or its metrics land just above a covenant limit or just above consensus quarter after quarter, that is the same phenomenon Carslaw and Thomas measured, visible without any statistics at all.

**If you are ever the person being tested.** Note that every artifact in this post is produced by trying to look normal. Nelson avoided round numbers and duplicates and was still exposed by his leading digits. That generalises: fabrication has many independent signatures, and defeating the one you know about does not help with the ones you do not.

**The single sentence worth keeping.** Digit analysis is a way of deciding where to look in a haystack too big to search. It converts a million transactions into a short list of documents to read — and then a human being has to read them. It is a ranking tool, not a verdict, and the moment anyone treats a failed test as a finding, the technique has stopped being forensic and started being an accusation.

If you want to go further into the toolkit this belongs to, the natural next steps are the analytical procedures that use ratios rather than digits, and the study of how audits are structured and where their blind spots are — which is where digit analysis was invented to help in the first place.

## Sources & further reading

**The original papers**

- Simon Newcomb, "Note on the Frequency of Use of the Different Digits in Natural Numbers", *American Journal of Mathematics*, vol. 4, 1881, pp. 39–40. The first statement of the logarithmic law, prompted by the observation that the early pages of logarithm tables were more worn than the later ones.
- Frank Benford, "The Law of Anomalous Numbers", *Proceedings of the American Philosophical Society*, vol. 78, no. 4, 1938, pp. 551–572. The paper the law is named after. Benford assembled some 20,229 numbers from twenty unrelated tables — river areas, populations, physical constants, molecular weights, street addresses, numbers appearing in newspapers — and showed they shared one leading-digit distribution. The expected frequencies used throughout this post are Benford's law as stated here.
- Theodore P. Hill, "A Statistical Derivation of the Significant-Digit Law", *Statistical Science*, vol. 10, no. 4, 1995, pp. 354–363. The rigorous modern derivation, including the result that random samples from randomly chosen distributions converge to Benford. This is the source for the scale-invariance and distribution-mixing arguments in this post.

**Forensic method and the conformity cutoffs**

- Mark J. Nigrini, *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*, Wiley, 2012. The standard practitioner reference and the source of the MAD conformity bands used here, including the widely-cited first-digit cutoffs of 0.006 / 0.012 / 0.015 and the first-two-digits cutoffs of 0.0012 / 0.0018 / 0.0022.
- Mark J. Nigrini, *Digital Analysis Using Benford's Law*, 2000. The earlier reference, which circulates a **different** set of MAD cutoffs — 0.004 / 0.008 / 0.012. This is worth knowing about: two analysts citing "Nigrini's thresholds" may be using different numbers, which is one more reason to report the MAD value itself rather than only the verdict.
- Mark J. Nigrini and Linda J. Mittermaier, "The Use of Benford's Law as an Aid in Analytical Procedures", *Auditing: A Journal of Practice & Theory*, vol. 16, no. 2, 1997, pp. 52–67. The paper that introduced the tests into the auditing literature.

**The cases**

- Mark J. Nigrini, "I've Got Your Number", *Journal of Accountancy*, vol. 187, no. 5, May 1999, pp. 79–83. Source for *State of Arizona v. Wayne James Nelson* (CV92-18841, 1993) — the 23 cheques, the "over 90% have 7, 8 or 9 as a first digit" finding, the absence of round numbers and duplicates — and for the bank credit-card write-off case with its spike at first-two-digits bin 49. Note that this article presents the Nelson analysis retrospectively, after conviction on other evidence.
- Bernhard Rauch, Max Göttsche, Gernot Brähler and Stefan Engel, "Fact and Fiction in EU-Governmental Economic Data", *German Economic Review*, vol. 12, no. 3, 2011, pp. 243–255. The Eurostat deficit-criteria study in which Greece showed the greatest deviation from Benford's law among the euro states.
- Charles A. P. N. Carslaw, "Anomalies in Income Numbers: Evidence of Goal Oriented Behavior", *The Accounting Review*, vol. 63, no. 2, 1988, pp. 321–327. Excess second-digit zeros in New Zealand firms' reported earnings.
- Jacob K. Thomas, "Unusual Patterns in Reported Earnings", *The Accounting Review*, vol. 64, no. 4, 1989. The US extension, including the reversed pattern for loss-making firms.

**Elsewhere in this series**

- [How an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) — why sampling leaves the gap that digit analysis fills.
- [Round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) — fraud that leaves no fabricated digits at all.
- [Revenue recognition games: channel stuffing and bill-and-hold](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold) — the better-targeted screens to reach for first.
- [Reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings) — where the Carslaw and Thomas findings land in practice.
- [Hypothesis testing and p-values](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews) — the excess power problem in its general form.

*This post is educational and is not financial, accounting or legal advice. All datasets in the worked examples are synthetic and were constructed by the author to demonstrate the method; none represents any real company's records. Non-conformity to Benford's law is a basis for further investigation and is not evidence of wrongdoing.*
