---
title: "Luckin Coffee: Fabricated Sales and the Report That Counted Cups"
date: "2026-08-12"
publishDate: "2026-08-12"
description: "How a Chinese coffee chain fabricated RMB 2.12 billion of sales, why every cash-based fraud test failed to catch it, and how one physically countable number in its own filings gave it away."
tags: ["forensic-accounting", "luckin-coffee", "financial-statement-fraud", "fabricated-revenue", "round-tripping", "short-selling", "china-adr", "revenue-recognition", "unit-economics", "sec-enforcement"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 54
depth: "deep-dive"
---

> [!important]
> **TL;DR:** Luckin Coffee fabricated about RMB 2.12 billion of sales in 2019, roughly US\$311 million, and it did it with real money. Because the fake customers actually paid, none of the classic cash-based fraud tests could fire. The number that gave it away was one anybody could count: cups per store per day.
>
> - Luckin's own restatement, filed 30 June 2021, shows the fraud in a single derived metric. Reported items sold per store per day ran 244 in Q1 2019, 345 in Q2, 444 in Q3. Restated, the same quarters were 244, 262 and 263. Real throughput per store was flat. The climb was invented.
> - The fraud was self-funded. In the largest of three schemes the SEC describes, seven funding companies tied to Luckin people wired real cash in to buy coupons, Luckin booked fake redemptions as revenue, then sent the cash back out through fabricated expenses to suppliers that supplied nothing. Cash flow never looked wrong because the cash was real.
> - Accounts receivable at 31 December 2019 were RMB 22.8 million, under 1% of restated revenue. In a prepaid, app-only retail business there is nothing to inflate on the receivables line, so the standard days-sales-outstanding test had no purchase at all.
> - The SEC said revenue was overstated by 45% in the third quarter of 2019. The restatement implies 83%. Both are true. The gap is entirely the choice of denominator, and knowing which one you are looking at is most of forensic numeracy.
> - Luckin settled with the SEC on 16 December 2020 for a US\$180 million penalty without admitting or denying the allegations, was delisted from Nasdaq on 13 July 2020, and is now a far larger and genuinely profitable company trading over the counter.

Every accounting fraud has a moment where the numbers stop being about accounting and start being about physics.

For Luckin Coffee that moment is this. In the third quarter of 2019 the company told investors it had 3,680 stores and had sold an average of 44.2 million items a month. Divide the second number by thirty days and by the average store count for the quarter and you get 444 items per store per day. A Luckin pick-up store is a small counter with a couple of machines. Four hundred and forty-four cups a day is thirty-seven cups an hour, every hour, across a twelve-hour day, sustained every day, at three and a half thousand locations at once.

You do not need a forensic accounting qualification to be suspicious of that. You need a stopwatch and somewhere to stand.

![How money moved in a circle through Luckin's fabricated coupon sales, entering as coupon purchases from funding companies and leaving as payments to suppliers that supplied nothing](/imgs/blogs/luckin-coffee-fabricated-sales-and-the-muddy-waters-report-1.webp)

The diagram above is the mental model for everything that follows, and it is the reason Luckin is worth a whole post rather than a paragraph in a list of frauds. Most fabricated-revenue schemes create revenue out of nothing, which means they leave a hole where the cash should be. That hole is what forensic accountants are trained to find. Luckin did not leave a hole. It moved real money in a circle: cash came in from companies its own people controlled, got booked as retail sales, and went back out as payments to vendors who delivered nothing. The cash balance was real. The bank statements reconciled, at least the ones that had not been altered. Revenue growth was matched by cash.

Which is to say: the single most-taught fraud test in the world, *does the cash follow the revenue*, returned a clean result on one of the largest fabricated-revenue frauds of the decade.

This post walks the whole thing through the actual reported lines, with the numbers Luckin itself published and the numbers it later republished. We will build the accounting from zero, so no prior finance background is needed. Then we will do the arithmetic that a reader sitting at a laptop in 2019 could have done, using nothing but the company's own press releases, and see exactly where it broke.

---

## First principles: how a cup of coffee becomes a line on an income statement

Before we can say what Luckin faked, we need to be precise about what a coffee company's financial statements even claim. If you already know what deferred revenue is, skim; nothing here is assumed later.

### The three statements, in one paragraph each

A company publishes three linked financial statements.

The **income statement** (also called the profit and loss statement, or the statement of comprehensive loss when the company is losing money) covers a period of time, usually three months or a year. It starts with **revenue**, the value of what the company sold in that period, subtracts the costs of selling it, and ends with **net income** or **net loss**. It is an *accrual* statement, which is the crucial word: it records economic events when they happen, not when money changes hands.

The **balance sheet** is a snapshot at a single instant, usually the last day of the period. It lists **assets** (things the company owns or is owed), **liabilities** (what it owes), and **equity** (the residual belonging to shareholders). Assets always equal liabilities plus equity, by construction.

The **cash flow statement** covers the same period as the income statement but tracks only actual money moving in and out, split into operating, investing and financing activities. It exists precisely because the income statement is an accrual statement and can therefore drift a long way from cash.

If you want a fuller treatment of how the three interlock and why the third one exists, this series covers it in [the three financial statements and how they interlock](/blog/trading/forensic-accounting/the-three-financial-statements-and-how-they-interlock) and [reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

### Revenue recognition, and why a coupon is not a sale

**Revenue recognition** is the set of rules that decide *when* a sale counts. The intuition is simple: you recognise revenue when you have actually delivered the thing you were paid for, not when the money arrives.

This matters enormously for Luckin, because Luckin sold coffee through coupons. A customer bought coupons inside the Luckin app using Alipay or WeChat, then redeemed a coupon later for an actual drink. The SEC's complaint records the company's stated policy precisely: Luckin "recognized revenue from the sale of a coupon when the coupon was redeemed by a customer for coffee or another product, rather than at the time of the sale of the coupon."

That policy is correct accounting. It also creates a two-step structure that turns out to matter a great deal.

**Step one, the coupon purchase.** Money arrives. Luckin owes the customer a coffee. In double-entry bookkeeping, every transaction has two equal and opposite halves, a **debit** and a **credit**. Here:

```
Dr  Cash                     100      (an asset goes up)
    Cr  Deferred revenue         100  (a liability goes up)
```

**Deferred revenue** is a liability: it is the obligation to hand over a coffee later. No revenue has been recognised. Nothing has touched the income statement.

**Step two, the redemption.** The customer collects a drink. Now the obligation is discharged and the sale is real:

```
Dr  Deferred revenue         100      (the liability goes away)
    Cr  Revenue                  100  (the income statement finally sees it)
```

And separately, the cost of the ingredients that went into the cup:

```
Dr  Cost of materials         35
    Cr  Inventory                 35
```

Hold onto that last entry. It is where the whole thing eventually shows.

### The vocabulary you need for the rest of this post

- **ADS (American Depositary Share).** A Chinese company cannot list its ordinary shares directly on Nasdaq. Instead a US bank holds the shares and issues receipts against them that trade in New York. Luckin's ADS represented eight ordinary shares each. When you read "the stock fell to US\$6.40", that is the price of one ADS.
- **Form 20-F and Form 6-K.** A foreign company listed in the US files an annual report on **Form 20-F** and furnishes interim results, press releases and material announcements on **Form 6-K**. Luckin's quarterly earnings were 6-K exhibits, not audited filings. That distinction matters: the fraud ran through unaudited quarterly numbers and was caught by the *annual* audit.
- **Restatement.** When previously published financial statements turn out to be wrong, the company reissues them. A restatement is the closest thing accounting has to a confession under oath, because it is signed, filed, and audited.
- **Net revenue.** Revenue after discounts and promotions. Luckin's promotions and coupons, other than free-product giveaways, were netted against revenue rather than shown as marketing costs. So Luckin's revenue line was already stated after the effect of its famous discounting.
- **Short seller.** An investor who profits when a share price falls, by borrowing shares, selling them, and buying them back later. Short sellers have an obvious financial motive to publish bad news, and an equally obvious motive to be right, since a wrong short loses money without limit. The process by which they find fraud has its own post in this series: [the short seller's playbook](/blog/trading/forensic-accounting/the-short-sellers-playbook-how-activists-find-fraud).
- **DSO (days sales outstanding).** The average number of days between making a sale and collecting the cash. Rising DSO is the classic tell for fabricated revenue, because fake customers do not pay. Remember this one; its failure here is the point of the post.

### Unit economics: the thing a coffee chain actually is

A retail chain's revenue is not a mystical quantity. It decomposes:

$$
\text{Revenue} = \text{stores} \times \text{days} \times \frac{\text{items}}{\text{store} \cdot \text{day}} \times \text{net price per item}
$$

Four terms. Store count is countable from the outside, and companies advertise it. Days are days. Net price per item is roughly knowable from the app. And items per store per day is a physical quantity: it is limited by the number of machines, the number of staff, the length of the counter, and the number of people who walk past.

That decomposition is the whole forensic method of this post. If revenue grows, it grows through one of those four terms. Growth in store count is expansion. Growth in items per store per day is either genuine demand or something else. There is a good general treatment of this lens in [unit economics and the value chain](/blog/trading/equity-research/unit-economics-and-the-value-chain).

---

## The company Luckin said it was

Luckin opened its first store in October 2017. By the time it listed on Nasdaq on 17 May 2019, less than two years later, it had 2,370 stores across 28 Chinese cities and, on its own account, more than 16.8 million transacting customers. The IPO priced at US\$17 per ADS and raised roughly US\$600 million, valuing the company at about US\$3.9 billion. The SEC's complaint records that Luckin described itself as China's "second largest and fastest-growing coffee network" and stated an intention to become the largest, measured by store count, by the end of 2019.

For context on the base it was growing from: Luckin's total revenue for the whole of 2018 was US\$125 million.

Then came the quarters that were later restated.

**Q2 2019, furnished 14 August 2019.** Total net revenues of RMB 909.1 million, about US\$132 million. Product revenue up 698% year on year. Analysts had expected roughly US\$130 to US\$133 million, so the result landed as "in line". The ADS closed at US\$20.68 the next day.

**Q3 2019, announced 13 November 2019.** This is the release worth reading closely, because it is the one where the story became irresistible. The headline was "Net Revenues from Products of RMB1.5 Billion, Exceeding High End of Guidance Range" with a subhead of "Store Level Operating Profit Margin of 12.5% for the Quarter".

The reported numbers, from Luckin's own 6-K exhibit:

| Q3 2019, as reported 13 Nov 2019 | RMB million | US\$ million |
| --- | --- | --- |
| Net revenues, freshly brewed drinks | 1,145.4 | 160.2 |
| Net revenues, other products | 347.8 | 48.7 |
| Net revenues, others (delivery) | 48.4 | 6.8 |
| **Total net revenues** | **1,541.6** | **215.7** |
| Cost of materials | (721.1) | (100.9) |
| Store rental and other operating costs | (477.3) | (66.8) |
| Depreciation | (108.5) | (15.2) |
| Sales and marketing | (557.7) | (78.0) |
| General and administrative | (246.1) | (34.4) |
| Store preopening and other | (21.8) | (3.0) |
| **Total operating expenses** | **(2,132.5)** | **(298.3)** |
| **Net loss** | **(531.9)** | **(74.4)** |

All US dollar translations in that release use RMB 7.1477 to US\$1.00, the 30 September 2019 rate from the Federal Reserve's H.10 statistical release. That is the rate Luckin itself chose, and it is the one this post uses whenever it converts a Luckin RMB figure from 2019 unless stated otherwise.

The operating metrics in the same release:

- Average monthly total items sold: 44.2 million, up 470.1% year on year
- Average monthly transacting customers: 9.3 million, up 397.5%
- Total stores at quarter end: 3,680, up 209.5%
- Average total net revenues from products per store: RMB 449.6 thousand, up 79.5%
- Store level operating profit: RMB 186.3 million, 12.5% of product revenue

Then-CEO Jenny Zhiya Qian said on the earnings call, in the release's own quotation, "product revenue grew at 557.6%, which was 1.2x, 1.4x and 2.7x the growth rate of average monthly items sold, average monthly transacting customers, and number of stores, respectively."

Read that sentence again, because it is the fraud confessing in the language of a highlight reel. She is saying, correctly, that revenue was growing faster than stores, faster than customers, and faster than items. Presented as operating leverage. What it actually describes is a company whose revenue per store, per customer and per item were all rising at once, in a business built on aggressive discounting, at the exact moment it was adding a thousand stores a quarter.

The stock went from US\$18.98 on 12 November to US\$28.16 on 18 November. It closed 2019 at US\$39.36.

#### Worked example: the growth arithmetic in the CEO's sentence

Take the Q3 2019 release at face value and check the claim.

- Product revenue growth: RMB 1,493.2m in Q3 2019 versus RMB 227.1m in Q3 2018 gives 1,493.2 / 227.1 = 6.58, that is +558%. Matches the reported 557.6%.
- Items growth: 44.2m per month versus 7.8m gives 5.70, that is +470%. Matches.
- Ratio of the two growth rates: 557.6 / 470.1 = 1.19, which is the "1.2x" claimed.

So the claim is arithmetically true. Now ask what it implies. If revenue grows 5.58 times and items grow 5.70 times, revenue per item is roughly flat, so the leverage is not coming from price. If revenue grows 5.58 times and stores grow 3.10 times, revenue per store grew 1.80 times. And since price per item is flat, essentially all of that per-store revenue growth has to be **more items sold per store**.

The intuition: management presented the ratio of two growth rates as evidence of efficiency, but the same ratio pins the entire story on a single physical quantity, throughput per store, and says it nearly doubled in a year while the store base tripled.

---

## The number you could count from the pavement

Luckin published, every quarter, a table titled KEY OPERATING DATA. It gave store count at period end and average monthly items sold. That is enough.

**Deriving items per store per day.** Luckin defines "average number of stores during the period" implicitly, and we can verify which convention it used. The Q3 2019 release states average product revenue per store of RMB 449.6 thousand on product revenue of RMB 1,493.2 million. Dividing gives 3,321.5 stores, which is exactly the average of the 2,963 stores at the end of Q2 and the 3,680 at the end of Q3. So Luckin used the simple average of opening and closing store count. Good: we can reproduce its own convention.

Then:

$$
\frac{\text{items}}{\text{store} \cdot \text{day}} = \frac{\text{average monthly items sold}}{30 \times \text{average stores}}
$$

#### Worked example: computing Luckin's reported throughput, quarter by quarter

**Q1 2019.** Average monthly items 16,275.8 thousand. Stores 2,073 at the start, 2,370 at the end, average 2,221.5.

16,275,800 / 30 = 542,527 items per day. 542,527 / 2,221.5 = **244 items per store per day.**

**Q2 2019.** Average monthly items 27,593.0 thousand. Stores 2,370 to 2,963, average 2,666.5.

27,593,000 / 30 = 919,767. 919,767 / 2,666.5 = **345 items per store per day.**

**Q3 2019.** Average monthly items 44,244.6 thousand. Stores 2,963 to 3,680, average 3,321.5.

44,244,600 / 30 = 1,474,820. 1,474,820 / 3,321.5 = **444 items per store per day.**

So in six months, on the reported numbers, the average Luckin store went from serving 244 items a day to serving 444, an increase of 82%, while the company simultaneously opened 1,310 net new stores. New stores are normally a drag on the average, because they ramp. Here the average nearly doubled anyway.

The intuition this teaches: revenue can be inflated invisibly, but revenue divided by a physical asset count is a claim about the physical world, and the physical world can be checked.

![Items sold per store per day, as originally reported versus as restated, for each quarter of 2019](/imgs/blogs/luckin-coffee-fabricated-sales-and-the-muddy-waters-report-2.webp)

Now the punchline, and it is worth pausing on.

When Luckin restated its 2019 numbers on 30 June 2021, it republished that same operating table. Run the identical arithmetic on the restated figures:

| Quarter | Avg monthly items, reported (000s) | Avg monthly items, restated (000s) | Items per store per day, reported | Items per store per day, restated |
| --- | --- | --- | --- | --- |
| Q1 2019 | 16,275.8 | 16,275.8 | 244 | 244 |
| Q2 2019 | 27,593.0 | 20,971.6 | 345 | 262 |
| Q3 2019 | 44,244.6 | 26,238.7 | 444 | 263 |
| Q4 2019 | never reported | 33,273.4 | n/a | 271 |

Two things jump out.

**First, Q1 2019 was not restated at all.** Identical in both tables. The Special Committee found that fabrication began in April 2019, which is the first month of Q2. The restatement's own footprint confirms the start date.

**Second, the real number is flat.** Restated throughput per store went 244, 262, 263, 271 across 2019. That is a company whose stores were doing roughly the same volume all year while it opened stores at a furious pace. Which is a perfectly reasonable, even respectable, thing for a rapidly expanding chain to be. It is simply not the story that was sold.

The gap between the two lines is the fraud, and it is arithmetically exact: 444 versus 263 in Q3 2019 is an overstatement of 68.6% on the true number.

---

## What Muddy Waters distributed, and why the field work mattered

On 31 January 2020, Muddy Waters Research published a report alleging that Luckin had fabricated its financial performance metrics from the third quarter of 2019 onwards. Muddy Waters did not write it. The report was anonymous, and the field work behind it was carried out by, in the report's own description, thousands of on-the-ground staff associated with its author.

A note on sourcing before we go further. The report itself is no longer readily available in a form this post can cite directly. What follows is quoted from the first securities class action complaint filed against Luckin, *Cohen v. Luckin Coffee Inc.*, No. 1:20-cv-01293 in the Southern District of New York, filed on 13 February 2020, two weeks after publication. A complaint is a pleading, not a finding, but it is a dated court filing that reproduces the report's language verbatim, and it is the most reliable public record of what the report actually said. Every quotation below is the report's own wording as reproduced there.

### The field work

The report's central claim was about the number we derived in the last section, and it was blunt: items per store per day "was inflated by at least 69% in 2019 3Q and 88% in 2019 4Q, supported by 11,260 hours of store traffic video."

Sit with those two numbers for a moment. Eleven thousand two hundred and sixty hours of video is about 469 days of continuous footage. It was not one investigator with a camera. It was an operation.

The counting produced this: the report's "offline tracking results of tracking 981 store-days from 2019 4Q showed 263 items per store per day only." Against that it set Luckin's reported 444 items per store per day for the third quarter of 2019, the same figure we computed from the company's own release.

On price, the report said its staff "gathered 25,843 customer receipts and found that Luckin inflated its net selling price per item by at least RMB 1.23 or 12.3% to artificially sustain the business model", and that "[i]n the real case, the store level loss is high at 24.7%-28%." The receipts also produced an incidental finding that is more interesting than it looks: "25,843 receipts indicate 1.08 and 1.75 items per order for pick-ups and delivery orders respectively or blended 1.14", marking "a continuously downward trend of items per order from 1.74 in 2018 1Q to 1.14 in 2019 4Q." A chain whose basket size is shrinking is not a chain whose per-store throughput is doubling.

On the mix, the report said revenue contribution from "other products" was "only about 6% in 2019 3Q, representing nearly 400% inflation". The supporting detail: "for the 981 store-days we tracked, only 2% of the pick-up orders were found containing non-freshly brewed products", and "[t]he 25,843 receipts further indicate that 4.9% and 17.5% of items for pick-up and delivery orders were 'other products', blended 6.2%". Luckin's reported Q3 2019 figure was RMB 347.8 million of other-products revenue against RMB 1,493.2 million of product revenue, which is 23.3%.

And on the expense side, the report asserted that "[t]hird party media tracking showed that Luckin overstated its 2019 3Q advertising expenses by over 150" (the complaint quotes the figure without a unit; in context it is a percentage), followed by the sentence that turned out to be the most prescient line in the document: "It's possible that Luckin recycled its overstated advertising expense back to inflate revenue and store-level profit."

That is the round trip, guessed from the outside, eleven months before the SEC described the mechanism in a complaint.

#### Worked example: reconstructing the report's Q4 estimate from Luckin's own guidance

The report's Q4 2019 claim is fully reproducible from public numbers, which is what makes it a good demonstration of the method. It estimated that Luckin's own fourth-quarter guidance implied 483 to 506 items per store per day.

Here are its inputs, quoted from the complaint: "4Q Guidance Product Revenue Guidance of RMB 2.1 billion to RMB 2.2 billion, divided by Net selling price per item of RMB 11.8 (Assuming Luckin to report 5% sequential growth from 2019 3Q of RMB 11.2) and average store number of 4,094."

Work the low end.

1. Guided product revenue: RMB 2,100 million, about US\$294 million at the RMB 7.1477 rate.
2. Assumed net price per item: RMB 11.8, which is the Q3 2019 figure of RMB 11.2 plus 5%.
3. Implied items for the quarter: 2,100,000,000 / 11.8 = 177.97 million items.
4. Per day, over a 90-day quarter: 177,970,000 / 90 = 1,977,400 items.
5. Per store, at 4,094 average stores: 1,977,400 / 4,094 = **483 items per store per day.**

Repeat with RMB 2,200 million and you get 506. The range is exactly as published.

Two of those inputs check out independently. The RMB 11.2 net price per item is what you get by dividing Luckin's reported Q3 2019 product revenue of RMB 1,493.2 million by its reported items sold, 44.2446 million a month times three, which gives RMB 11.25. And the 4,094 average store count is almost exactly the average of the 3,680 stores at the end of Q3 and the 4,507 at the end of Q4, which is 4,093.5. The report's arithmetic was not the weak link.

The intuition: the report did not need inside information to make its central claim. It needed the company's guidance, the company's own price and store numbers, and someone willing to stand in 981 store-days' worth of shops and count.

### The pushback

Luckin denied all of the claims.

The more instructive reaction came from another short seller. Andrew Left of Citron Research, one of the best-known activist short sellers in the market, disclosed that he had taken a **long** position in Luckin, citing data from Business Connect China, a Shanghai expert-network firm, and said the Muddy Waters report would "fall short on accuracy."

On 12 February 2020, J Capital Research published a report supporting the anonymous report's findings and rebutting Citron directly. Its rebuttal, as quoted in the complaint, is worth reading because it is a lesson in reading someone else's evidence:

> "Citron cited data from 'Biz Con China,' referring to Business Connect China (BCC) [...] We managed to see a copy of BCC's report, and right there in the first paragraph is written: 'Based on BCC's tracking, we are skeptical about some of Luckin's reported figures.' The next paragraph begins: 'Luckin's reported figure of 444 items sold per day in 3Q19 is likely to be higher than their actual sales.' Citron seems to have missed this."

The same dataset, cited by one firm as exoneration, contained a sentence saying the key number was probably overstated.

The market's immediate verdict was mild. On 31 January 2020 the ADS fell US\$3.91, or 10.74%, to close at US\$32.49, according to the complaint. A double-digit fall is not nothing, but it left the stock nearly double its IPO price. The disclosure that actually broke it was still nine weeks away, and it came from the company.

### How accurate was the count?

We can now answer that, because Luckin later published the answer itself.

| Items per store per day, Q4 2019 | Value |
| --- | --- |
| Implied by Luckin's own Q4 2019 guidance | 483 to 506 |
| Counted by the anonymous report, 981 store-days | 263 |
| Derived from Luckin's restated Q4 2019 operating data | 271 |

The field count was 263. The company's own restated figures imply 271. The report's number was about 3% below what the company eventually admitted, from counting people in shops.

And the guidance? Luckin guided to RMB 2.1 to 2.2 billion of Q4 2019 product revenue. When the quarter was finally reported in the restatement, revenue from product sales was RMB 1,034.5 million, about US\$148.6 million, under half the low end of the range. On the narrower definition the guidance actually used, freshly brewed drinks plus other products and excluding delivery, it was RMB 979.4 million, which is 47% of the low end.

For the general method behind this kind of work, rather than this particular company's statements, see [the short seller's playbook](/blog/trading/forensic-accounting/the-short-sellers-playbook-how-activists-find-fraud).

---

## The restatement, line by line

On 30 June 2021 Luckin filed an amended Form 6-K restating Q2 and Q3 2019 and, for the first time, reporting Q4 2019. Its own explanatory note gives the totals: net revenue in 2019 was inflated by approximately RMB 2.12 billion and costs and expenses were inflated by approximately RMB 1.34 billion. Quarter by quarter, the revenue inflation was RMB 0.25 billion in Q2, RMB 0.70 billion in Q3 and RMB 1.17 billion in Q4. The expense inflation was RMB 0.15 billion, RMB 0.52 billion and RMB 0.67 billion.

At the RMB 6.9618 to US\$1.00 rate that same filing uses (the 31 December 2019 H.10 rate), RMB 2.12 billion is about US\$305 million and RMB 1.34 billion about US\$192 million. The SEC, using its own conversion, put the fabricated sales at approximately US\$311 million and the fabricated expenses at more than US\$190 million. Those are the same underlying RMB amounts at slightly different exchange rates, not different findings.

Here is the third quarter, side by side.

![Luckin's third quarter of 2019 as originally reported and as restated, showing revenue, operating expenses and net loss](/imgs/blogs/luckin-coffee-fabricated-sales-and-the-muddy-waters-report-3.webp)

#### Worked example: Q3 2019 as reported versus as restated

| Q3 2019, RMB million | As reported 13 Nov 2019 | As restated 30 Jun 2021 | Difference |
| --- | --- | --- | --- |
| Total net revenues | 1,541.6 | 843.2 | (698.4) |
| Total operating expenses | (2,132.5) | (1,625.2) | 507.3 |
| Net loss | (531.9) | (723.0) | (191.1) |

Walk the three lines.

**Revenue.** RMB 1,541.6m becomes RMB 843.2m. The difference of RMB 698.4 million is the RMB 0.70 billion of fabricated coupon sales the Special Committee identified for the quarter, to the rounding. About US\$98 million of revenue for a single quarter did not exist.

**Expenses.** Operating expenses fall by RMB 507.3 million, again matching the RMB 0.52 billion of fabricated costs for the quarter. This is the half of the scheme people forget. The fraud did not just add revenue; it added costs, deliberately, because the cash used to buy the fake coupons had to be sent back to the people who had provided it, and a payment to a supplier is the most ordinary way for cash to leave a company.

**Net loss.** Reported RMB 531.9m, actual RMB 723.0m. The loss was understated by RMB 191.1 million, roughly US\$27 million. The company reported a loss 26% smaller than the real one.

The intuition: a fabricated sale that has to be funded is not one lie, it is two, because the money must come from somewhere and go somewhere. Look for the second lie on the expense line.

### The denominator trap

There is a numeracy point here that is worth its own figure, because it trips up almost everyone reading fraud coverage.

The SEC's complaint says Luckin "materially overstated its reported revenue by more than 27% for the period ending June 30, 2019, and 45% for the period ending September 30, 2019." The restatement says Q3 revenue fell from 1,541.6 to 843.2, which is a reduction of 45% but also means the reported figure was 83% *above* the true one.

![The same overstatement expressed two ways, as a share of the reported figure and as a markup on the true figure](/imgs/blogs/luckin-coffee-fabricated-sales-and-the-muddy-waters-report-4.webp)

#### Worked example: 45% or 83%?

Take the fabricated amount, RMB 700 million, and the two possible denominators.

**As a share of what was reported:** 700 / 1,541.6 = 45.4%. This is what the SEC quotes. It answers the question "how much of what they told us was invented?"

**As a markup on what was true:** 1,541.6 / 843.2 = 1.83. This is an 83% overstatement. It answers the question "how much bigger did they make the company look?"

Both are correct. They differ by a factor of 1 / (1 minus the fraud share): 45.4% / (1 minus 0.454) = 83.2%.

Apply the same to Q2 2019: the SEC's 27% is RMB 250m over reported revenue of RMB 909.1m. Against the restated RMB 653.4m, the reported figure was 39% too high.

The intuition: a percentage without its denominator is not a fact, it is a rhetorical choice. Regulators tend to quote fraud as a share of the reported number, because that is the number investors relied on. Restatements reveal the markup on truth, which is always larger. If you compare a headline from one source with a computation from another, you will produce a contradiction that does not exist.

---

## The entry behind a fabricated sale

We can now write the fraud as bookkeeping. This is what actually happened, translated into the two-line entries from the foundations section.

The SEC's complaint describes three separate purchasing schemes, escalating in size and in the distance they put between Luckin and its own money.

![The three purchasing schemes described in the SEC complaint, by size and structure](/imgs/blogs/luckin-coffee-fabricated-sales-and-the-muddy-waters-report-9.webp)

**Scheme one, from April 2019: individual customers.** Luckin employees and their family members, plus employees of two entities associated with certain Luckin officers and directors (the complaint calls them the "Two Related Entities"), moved money from personal bank accounts into WeChat and Alipay accounts tied to phone numbers they controlled. Those accounts bought coupons in the app. Then fake orders "redeemed" the coupons. No coffee was made. The SEC puts this scheme at several millions of dollars.

**Scheme two, from May 2019: four corporate customers.** Four companies, all controlled by or associated with Luckin personnel or employees of the Two Related Entities, wired money straight from corporate Alipay accounts to buy coupons. Same fake redemptions. The SEC puts this at tens of millions of dollars, "nearly triple the amount of sales fabricated in the first scheme."

**Scheme three, from May 2019: fictitious agents.** The big one. Luckin signed sham coupon-purchase agreements with shell companies presented as intermediary agents who would resell coupons to real customers. Seven funding companies, again controlled by or associated with Luckin employees or employees of the Two Related Entities, wired money into Luckin's bank accounts. Luckin employees then **altered the company's bank statements** so the money appeared to have come from the fictitious agents rather than the funding companies. The SEC alleges this scheme accounted for nearly 90% of the roughly US\$311 million of total fabricated revenue.

A May 2019 email quoted in the complaint, from an employee of one of the Two Related Entities to Luckin officers, describes the design intent: "We will try to replace the contact persons [of the Fictitious Agents] with third parties, in order to reduce the number of our internal colleagues that are aware of such issue."

#### Worked example: the full round trip of a fabricated RMB 100 sale

Follow one hundred renminbi, about US\$14, all the way around the circle. The entries are Luckin's real accounting policy applied to a transaction that never had a customer.

**Step 1. A funding company wires RMB 100 to Luckin to buy coupons.**

```
Dr  Cash                       100
    Cr  Deferred revenue           100
```

At this point nothing is even wrong. This is what any coupon pre-sale looks like. Cash is up, and Luckin owes someone a coffee.

**Step 2. A fake order redeems the coupon. No cup is made.**

```
Dr  Deferred revenue           100
    Cr  Revenue                    100
```

Revenue now exists on the income statement. Note what is *missing*: there is no matching `Dr Cost of materials / Cr Inventory` entry, because no coffee was made and no beans were consumed. A fabricated sale carries a 100% gross margin.

**Step 3. The cash has to go home. Luckin pays a "supplier" that supplied nothing.**

```
Dr  Cost of materials           95
    Cr  Cash                        95
```

Now the money is back with the network that provided it, and the books show a normal-looking purchase.

**What the statements say afterwards.** Revenue +100. Costs +95. Net loss improved by 5. Cash net change roughly zero. Operating cash flow: +100 in, 95 out, net +5. Receivables: unchanged. Deferred revenue: unchanged.

Every single classic tell is silent. And notice step 3 is not optional: without it the fabricated cash simply piles up on the balance sheet and the funding companies never get paid back. The SEC's complaint records the mechanism plainly. Luckin made payments to 13 purported suppliers of raw materials that provided no materials, overpaid two providers of human-resources outsourcing, and paid delivery fees to three companies that provided no services.

There is a real accounting artefact in step 2 worth flagging: since fabricated redemptions consume no ingredients, they carry no material cost, so they flow almost entirely into store-level profit. Store-level operating profit margin is precisely the metric Luckin put in the subheading of its Q3 2019 release, at 12.5%. When Luckin finally reported Q4 2019 in the restatement, store-level operating margin was negative 24.9%, on a basis the company says it changed. The metric management was showcasing was the metric the fabrication most directly inflated.

The general mechanics of this pattern, and how it appears elsewhere, are covered in [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) and [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing).

---

## Why the standard forensic tests never fired

This is the part of the Luckin story that has real teaching value, and it is usually skipped.

![Why each standard forensic test failed to detect Luckin's fabricated revenue](/imgs/blogs/luckin-coffee-fabricated-sales-and-the-muddy-waters-report-5.webp)

**The receivables test could not fire.** The single most reliable signal of fabricated revenue is receivables growing faster than sales. Fake customers do not pay, so the sale sits in accounts receivable and days sales outstanding climbs. Luckin's restated balance sheet at 31 December 2019 shows accounts receivable of RMB 22.8 million against full-year net revenues of RMB 3,024.9 million. That is 0.75% of revenue, a DSO of under three days. Even on the widest reading, adding the separate RMB 16.2 million of "receivables from online payment platforms" that sits on the same balance sheet, the total is RMB 39.0 million, or 1.3% of revenue. And that is *correct*: Luckin was a prepaid, app-only business where every customer paid before collecting. There was no receivable to inflate. The test had nothing to bite on, not because the fraud defeated it but because the business model made it inapplicable. The general framework is in [forensic ratios: DSO, DIO, DPO and margin anomalies](/blog/trading/forensic-accounting/forensic-ratios-dso-dio-dpo-and-margin-anomalies).

**The cash flow test could not fire.** The second most reliable signal is revenue growth without matching operating cash flow. But Luckin's fake revenue arrived as real money from real bank accounts. The Q3 2019 release reported net cash used in operating activities of only RMB 122.8 million against a reported net loss of RMB 531.9 million, and cash and short-term investments of RMB 5,543.9 million. Management attributed the small operating outflow to "a reduction of operating loss and a favorable working capital profile." Nothing about that picture looks like fabricated revenue. It looks like a discounting business finally getting operating leverage.

**The margin test was muted.** In principle a fabricated sale with no cost should blow out gross margin, which is a bright red flag. Luckin's scheme neutralised it by fabricating costs in step, RMB 1.34 billion of them, timed and sized to keep the ratios plausible. The complaint says so directly: Luckin employees "increased costs to make those costs consistent with its increased, inflated revenue."

**What did fire was the physical test.** Items per store per day is not an accounting number. It is a claim about how many cups a counter produces. No amount of internal database manipulation changes what a person standing in the shop can count.

### The balance sheet lines that did move

If the fake money entered as cash and left as expenses, some balance somewhere had to absorb the timing. Look at what grew fastest between 2018 and restated 2019.

![Luckin's restated balance sheet growth by line, 2018 versus 2019](/imgs/blogs/luckin-coffee-fabricated-sales-and-the-muddy-waters-report-6.webp)

| RMB million, 31 December | 2018 | 2019 (restated) | Multiple |
| --- | --- | --- | --- |
| Total net revenues (full year) | 840.7 | 3,024.9 | 3.6x |
| Accounts receivable | nil | 22.8 | n/a |
| Receivables from online payment platforms | 4.6 | 16.2 | 3.5x |
| Prepaid expenses and other current assets, net | 365.5 | 1,660.4 | 4.5x |
| Accrued expenses and other liabilities | 371.0 | 3,193.8 | 8.6x |
| Cash and cash equivalents | 1,631.0 | 4,865.8 | 3.0x |

Revenue grew 3.6 times. Prepaid expenses and other current assets grew 4.5 times. Accrued expenses and other liabilities grew 8.6 times.

An honest caveat, and it matters: this is the *restated* balance sheet, prepared after the investigation, and the filing does not attribute these specific balances to the fabricated transactions. A company that opened 2,400 stores in a year will legitimately have large prepayments (deposits, rent, equipment advances) and large accruals. So this is not proof of anything on its own.

What it *is*, is a lesson about where to look. Fabricated-revenue playbooks tell you to watch receivables. In a prepaid business there are none, and the interesting movement is on the other side of the working capital cycle: the prepayments and accruals that record money going *out*. The general point, that the fraud hides wherever the analyst's checklist is not looking, is the whole reason [the footnotes and MD&A](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) exist as a section of a filing.

---

## The timeline, and the raise that landed at the top

![Timeline of Luckin Coffee from its first store to the SEC settlement](/imgs/blogs/luckin-coffee-fabricated-sales-and-the-muddy-waters-report-7.webp)

The sequencing is what makes this more than an accounting story.

According to the SEC's complaint, fabrication began in April 2019, the month *before* the IPO priced. The company then furnished two quarters of materially misstated results, in August and November 2019. The stock closed 2019 at US\$39.36, roughly double the IPO price.

Then, on 14 January 2020, Luckin raised money against those numbers: approximately US\$418 million from a follow-on equity offering and approximately US\$446.7 million from a convertible bond issue, US\$864.7 million in total. The SEC's complaint states that the offering materials "included the company's previously disclosed, materially misstated financials for the second and third quarters of 2019", and that the January 2020 management presentation repeated the claim that product revenue grew 557.6% year on year in the third quarter and had "beat our Q3 guidance as a result of strong business fundamentals."

Three days later, on 17 January 2020, the ADS closed at an all-time high of US\$50.02, up 194% from the IPO price eight months earlier.

![Luckin's ADS price path from the May 2019 IPO to April 2020, with the January capital raise and the short report marked](/imgs/blogs/luckin-coffee-fabricated-sales-and-the-muddy-waters-report-8.webp)

#### Worked example: what the January raise was worth in hindsight

Price the raise against the two prices that bracket it.

At the 14 January 2020 offering, roughly US\$418 million of equity was sold into a market that put the all-time-high price at US\$50.02 three days later. On 2 April 2020, after the disclosure, the ADS closed at US\$6.40. On 6 April it closed at US\$3.39.

An investor who bought that follow-on and held to 6 April lost 93% from the January peak reference and about 87% from the 1 April close of US\$26.20. In cash terms, the roughly US\$864.7 million raised across equity and converts was supported by financial statements that, eleven weeks later, the company told investors not to rely on.

The intuition: the value of a fraud is not the accounting entry, it is the financing the accounting entry unlocks. Ask what a company did with its inflated numbers, not just how it made them.

**Then the audit found it.** Luckin's fraud, in the SEC's words, "came to light in early 2020 in the course of the annual external audit of the company's financial statements." The Board formed a Special Committee on 19 March 2020, made up of three independent directors, Sean Shao (chair), Tianruo Pu and Wai Yuen Chong, and retained Kirkland & Ellis as independent counsel with FTI Consulting as forensic accountants.

On **2 April 2020** the company published the disclosure that ended the story. Its exact words are worth quoting because they are unusually direct for a company in this position:

> "The information identified at this preliminary stage of the Internal Investigation indicates that the aggregate sales amount associated with the fabricated transactions from the second quarter of 2019 to the fourth quarter of 2019 amount to around RMB2.2 billion. Certain costs and expenses were also substantially inflated by fabricated transactions during this period."

RMB 2.2 billion is about US\$308 million at the RMB 7.1477 rate Luckin used in its Q3 2019 release. The same release named the chief operating officer and director, Jian Liu, and said that he and several employees reporting to him had engaged in misconduct "including fabricating certain transactions", and that investors should no longer rely on the financial statements for the nine months ended 30 September 2019 or the fourth-quarter guidance.

The ADS closed at US\$26.20 on 1 April 2020 and US\$6.40 on 2 April, a fall of 75.6% in one session. By 6 April it was US\$3.39.

---

## Admitted, alleged, and still only claimed

Fraud stories collapse three different epistemic categories into one. It is worth keeping them apart, especially where named living people are involved.

| Status | What it covers |
| --- | --- |
| **Found by the company's own Special Committee and published** (1 July 2020) | Fabrication began April 2019. 2019 net revenue inflated by approximately RMB 2.12 billion; costs and expenses inflated by RMB 1.34 billion. Evidence "demonstrates" that former CEO Jenny Zhiya Qian, former COO Jian Liu and certain employees reporting to them participated in the fabricated transactions, and that funds supporting them were funnelled through third parties associated with company employees and/or related parties. The Board terminated both officers, terminated 12 other employees and subjected 15 more to other disciplinary action. |
| **Alleged by the SEC, settled without admission or denial** (16 December 2020) | Three purchasing schemes; more than US\$300 million of fabricated retail sales; expenses inflated by more than US\$190 million; a fake operations database; altered bank records; more than US\$864 million raised from debt and equity investors during the fraud; violations of the antifraud, reporting, books-and-records and internal-control provisions. Luckin consented to permanent injunctions and a US\$180 million penalty **without admitting or denying the allegations.** |
| **A board judgment, not a finding of participation** | The Board resolved on 26 June 2020 to require Charles Zhengyao Lu to resign as director and chairman, and considered his removal at a 2 July 2020 meeting. The company said the Special Committee based that recommendation on "documentary and other evidence identified in the Internal Investigation and its assessment of Mr. Charles Zhengyao Lu's degree of cooperation". Note what that sentence does *not* say: unlike the wording used for Qian and Liu, it does not state that he participated in the fabricated transactions. |
| **Short-seller allegation** | The claims in the anonymous report distributed by Muddy Waters on 31 January 2020 were, at the time of publication, unverified allegations by an unnamed author. Some were subsequently corroborated by the company's own restatement; the report was not a finding by any authority. |

The distinction in row two matters more than people think. A settlement "without admitting or denying" is not a confession, and it is not an acquittal either. It is a negotiated end that lets a regulator obtain a penalty and an injunction without a trial. What makes Luckin unusual is that the company's *own* Special Committee had already published findings that go further than the SEC's settlement required, which is why the two rows above overlap so heavily.

The SEC's own quantification, for the record, and quoting its 16 December 2020 press release: Luckin "intentionally fabricated more than \$300 million in retail sales by using related parties to create false sales transactions through three separate purchasing schemes", and certain employees "attempted to conceal the fraud by inflating the company's expenses by more than \$190 million, creating a fake operations database, and altering accounting and bank records to reflect the false sales."

---

## What happened next

The interesting thing about Luckin's aftermath is that almost none of it went the way the headlines implied.

### The delisting

Nasdaq moved quickly. According to the SEC's complaint, Nasdaq filed a Form 25 to remove Luckin's ADS from listing on 1 July 2020, and the ADS were delisted on 13 July 2020. They have traded over the counter under the symbol LKNCY ever since, including in the company's most recent results release of 3 August 2026. There has been no return to a US exchange.

### The SEC penalty that was never paid to the SEC

This is the part almost nobody knows, and it is documented in Luckin's own annual report.

The 16 December 2020 settlement carried a US\$180 million civil penalty. But the SEC's own press release contains a sentence that is easy to skim past: "This payment may be offset by certain payments Luckin makes to its security holders in connection with its provisional liquidation proceeding in the Cayman Islands. The transfer of funds to the security holders will be subject to approval by Chinese authorities."

#### Worked example: following the US\$180 million

Track the provision through three annual reports.

**2020.** Luckin records a provision for the SEC settlement of US\$180 million, which its FY2022 annual report states was RMB 1,146.5 million at the rate used.

**2021.** Luckin *reverses* the entire provision. Its own explanation: "Based on all available information as of December 31, 2021, it was very probable that the SEC staff would be satisfied that the cash payment to the bond holders to be made in January 2022, estimated at the time to exceed US\$180 million, would fully offset the SEC civil penalty pursuant to the terms of the SEC settlement."

**3 February 2022.** The SEC files a notice with the SDNY court "acknowledging that our obligation to pay the civil money penalty had been satisfied."

So the accounting outcome is this: cash from the US Treasury's point of view, nil. Cash from the point of view of Luckin's bondholders, at least US\$180 million, paid in January 2022 as part of the restructuring.

The intuition: a headline penalty is a claim on a company's cash, not a transfer to the government by default. Read the offset clause. In cases where the fraud has already bankrupted the issuer, the regulator's preferred outcome is often to route the money to the people who lost it, and the press release will say so in a sentence nobody quotes.

### China's own regulator

On 23 September 2020 Luckin announced it had received penalty decisions from the Chinese State Administration for Market Regulation and certain of its sub-bureaus. The SAMR imposed an aggregate fine of RMB 61.0 million, roughly US\$8.8 million at the 31 December 2019 rate, on two Luckin entities and certain implicated third-party companies, on the basis that the conduct violated the PRC Anti-Unfair Competition Law.

Note the legal theory. Not securities fraud, which is the US framing, but unfair competition: in China the wrong was inflating your own scale to gain market advantage. Same conduct, different offence, and a fine roughly one twentieth the size of the US penalty.

### The investors

Two separate settlements resolved the securities claims.

The federal class action, *In re Luckin Coffee Inc. Securities Litigation*, No. 1:20-cv-01293-JPC-JLC in the Southern District of New York, covered purchasers of Luckin ADS between 17 May 2019 and 15 July 2020. A binding term sheet was announced on 21 September 2021 at a global settlement amount of US\$187.5 million, reduceable pro rata for investors who opted out. Final approval came from the court on **22 July 2022**, at a settlement amount of **US\$175 million**. Luckin's annual report notes that it expected to spend further amounts resolving the opt-out claims, and that it had already been named in a number of opt-out suits.

A second, separate class action in the Commercial Division of the New York State Supreme Court, Index No. 651939/2020, covered purchasers of the convertible notes who had not released their claims through the Cayman scheme. Luckin reached an agreement in principle on 9 January 2022 and the state court granted preliminary approval on 7 October 2022.

Across the three years, Luckin recorded provisions for equity litigation of US\$187.5 million in 2020, a further US\$24.4 million in 2021 and a further US\$41.9 million in 2022.

### The restructuring

The corporate machinery is worth a paragraph because it explains how a company can commit a US\$311 million fraud and still exist.

On 15 July 2020, the same month as the delisting, the Grand Court of the Cayman Islands appointed Alexander Lawson of Alvarez & Marsal Cayman Islands Limited and Wing Sze Tiffany Wong of Alvarez & Marsal Asia Limited as "light-touch" Joint Provisional Liquidators, on Luckin's own application, after a creditor presented a winding-up petition. "Light-touch" is the operative word: the board kept day-to-day control of the business under the liquidators' supervision, under a protocol executed on 16 October 2020. The stores never closed.

On 5 February 2021 the liquidators filed a Chapter 15 petition in the US Bankruptcy Court for the Southern District of New York, seeking US recognition of the Cayman proceeding. Chapter 15 is not a bankruptcy in the ordinary sense; it is the mechanism by which a foreign insolvency proceeding is recognised and enforced in the United States.

The substance of the restructuring was the US\$460 million of 0.75% convertible senior notes due 2025, the ones sold in January 2020 against the misstated financials. They were restructured through a Cayman scheme of arrangement, recognised in the US under Chapter 15. Luckin issued US\$109.9 million of 9.00% Series B senior secured notes due 2027 as part of the offshore restructuring, and announced their redemption in full on 26 August 2022.

The Cayman court dismissed the winding-up petition by an order dated 25 February 2022 and the liquidators were discharged with effect from 4 March 2022. The US bankruptcy court closed the Chapter 15 case on 8 April 2022.

### The auditors and the filings

Luckin's FY2018 financial statements were audited by Ernst & Young Hua Ming LLP. The FY2019 annual report, when it was finally filed on 21 September 2021, carries an opinion from Centurion ZD CPA & Co., which states in the opinion itself that it has served as the company's auditor since 2021. The company's FY2019 annual report lists Ernst & Young Hua Ming LLP, Marcum Bernstein & Pinchuk LLP and Centurion ZD CPA & Co. as the independent registered public accounting firms whose fees it discloses for the relevant periods. Luckin also disclosed that in connection with the FY2020 audit, it and its auditor identified a material weakness in internal control over financial reporting as of 31 December 2020.

### The company

And then the part that makes Luckin genuinely unusual among accounting frauds: it worked.

In its results for the second quarter of 2026, published on 3 August 2026, Luckin reported total net revenues of RMB 15,885.6 million, about US\$2,336.8 million, up 28.5% year on year. It reported GAAP operating income of RMB 2,122.9 million, about US\$312.3 million, a 13.4% operating margin. It ended the quarter with 36,310 stores, 23,734 of them self-operated and 12,576 run by partners, including stores in Singapore, Malaysia, Hong Kong and the United States. Average monthly transacting customers were 112.7 million.

Set that against the quarter at the centre of this post. Restated Q3 2019 total net revenues were RMB 843.2 million. The company now does more than eighteen times that in a quarter, profitably, with roughly ten times the stores.

There is one more detail in that 2026 release worth noticing. Luckin now discloses same-store sales growth for self-operated stores, and in the second quarter of 2026 it was negative 5.3%. That is the metric whose absence made the fraud possible: a per-store comparable that separates growth from expansion. The company reports it now, and it reports it when the number is bad.

---

## Common misconceptions

**"Cash flow does not lie, so a company with real cash cannot be faking revenue."**

This is the belief Luckin exists to kill. The maxim is a good default and a bad law. Cash flow is hard to fake *when the fraud has to conjure money out of nothing*. It is easy to satisfy when the fraud has a funding source. Luckin's fake customers paid with real yuan from real bank accounts, so operating cash flow behaved. What cash flow cannot tell you is *who* the payer was, and that is the question the statements never answer on their own. The right version of the maxim: cash flow does not lie about amounts, but it says nothing about identity.

**"Receivables always give away fabricated revenue."**

Only when the business has receivables. Luckin was prepaid and app-only, with accounts receivable of RMB 22.8 million against RMB 3.0 billion of restated annual revenue. Prepaid and subscription businesses invert the working capital cycle: the tell moves from receivables to deferred revenue, prepayments, and the expense side. Applying a checklist built for a business-to-business manufacturer to a consumer app business will return a clean bill of health on a fraud.

**"The auditor missed it."**

The audit found it. That is a matter of record: the SEC's complaint says the fraud "came to light in early 2020 in the course of the annual external audit", and Luckin's own disclosure says the issues were "raised to the Board's attention during the audit of the consolidated financial statements for the fiscal year ended December 31, 2019." What is fair to say is that the fraud ran for nine months through *unaudited* quarterly releases furnished on Form 6-K, and that quarterly numbers from a foreign private issuer carry far less assurance than most investors assume. The gap was not audit failure, it was the interval between audits. [How an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) is the companion piece.

**"The short seller uncovered the fraud."**

The report distributed by Muddy Waters on 31 January 2020 put the thesis in public and made it costly to ignore, and the company's own restatement later corroborated a striking part of it. But the internal investigation was triggered by the annual audit, and the Special Committee was formed on 19 March 2020. Both things are true at once, and treating either as the sole cause is a story, not a fact. What the report unambiguously did was establish that the key claim was checkable from outside the company.

**"A 45% revenue overstatement means revenue was 45% too high."**

No. It means 45% of what was reported was invented, which makes the reported figure 83% higher than the truth. See the worked example above. This confusion turns up constantly in fraud reporting and it always makes the fraud sound smaller than it was.

**"Fabricating revenue improves the numbers, so the fraud is on the revenue line."**

Half of it is. A *funded* fabrication has to return the money, which means an equal and opposite lie on the expense line: RMB 1.34 billion of it here, against RMB 2.12 billion of fake revenue. If you only audit the top line you find half the scheme, and the half you find will look inexplicable, because the money will appear to have vanished.

---

## How it shows up in real markets

**Enron, 2001: complexity as concealment.** Enron's frauds were mostly about *where* things were recorded rather than whether they happened. Special purpose entities moved debt off the balance sheet and mark-to-market accounting pulled decades of speculative profit into the current period. The common thread with Luckin is a related party: in both cases a structure nominally outside the company was controlled by people inside it. The difference is that Enron's numbers were defensible under the letter of the rules until they were not, while Luckin's were simply invented. See [Enron: a forensic re-read](/blog/trading/finance/enron-2001-accounting-fraud) and [this series' treatment of the SPEs](/blog/trading/forensic-accounting/enron-a-forensic-re-read-of-spes-and-mark-to-market).

**WorldCom, 2002: the entry nobody looked at.** WorldCom capitalised ordinary operating costs, turning expenses into assets and losses into profits, to the tune of about US\$11 billion. Structurally it is Luckin's mirror image: WorldCom lied about *costs* to protect the bottom line, Luckin lied about *revenue* to protect the growth rate, and each then had to bend the other side to keep the ratios plausible. See [WorldCom, the US\$11 billion capitalization fraud](/blog/trading/forensic-accounting/worldcom-the-11-billion-dollar-capitalization-fraud) for the case and [capitalizing costs to inflate profit](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move) for the technique.

**Wirecard, 2020: cash that was not there.** Wirecard reported roughly EUR 1.9 billion of cash that did not exist, confirmed to its auditor through documents routed via partners rather than by the banks themselves. Luckin is the inverse case and that is precisely why the pair is instructive: Wirecard's cash was fake and its receivables tell was suppressed by an offshore structure; Luckin's cash was real and its receivables tell did not exist by design. Any test that assumes cash is the incorruptible anchor fails on Wirecard; any test that assumes real cash proves real revenue fails on Luckin. See [Wirecard](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros).

**The China ADR cohort more broadly.** Luckin sits inside a longer pattern of US-listed China-based issuers where the auditor's ability to inspect work papers held in mainland China was constrained, and where short sellers doing on-the-ground field work supplied evidence that public filings could not. The SEC's own press release acknowledges "challenges in our ability to effectively hold foreign issuers and their officers and directors accountable to the same extent as U.S. issuers", and separately thanks the China Securities Regulatory Commission and the Swiss Financial Market Supervisory Authority for their assistance. That sentence is a policy statement dressed as a courtesy. This series covers the listing route itself in [shell companies, reverse mergers and how fraud gets listed](/blog/trading/forensic-accounting/shell-companies-reverse-mergers-and-how-fraud-gets-listed).

**The quarterly-versus-annual gap, everywhere.** Luckin's fabrication survived two quarterly releases and died at the first annual audit. That interval is structural, not Chinese: quarterly numbers from foreign private issuers are furnished, not filed, and are typically unaudited and unreviewed. Any company whose story depends on quarterly momentum is telling that story in the least-assured document it produces.

---

## A checklist for a growth retailer's reported sales

None of this is investment advice, and none of it identifies fraud on its own. These are the questions Luckin's own filings would have answered.

**1. Decompose revenue into physical terms.** Stores times days times items per store per day times price per item. Any revenue growth has to land in one of those. If it lands in throughput per store, ask what physically changed.

**2. Compute throughput per unit yourself, from the operating table.** Do not accept a per-store *revenue* figure, which mixes volume and price. Volume per unit per day is the number that has a ceiling.

**3. Check whether throughput rises while the unit count rises.** New units usually ramp, dragging the average down. A rising average during rapid expansion is a strong claim and needs a strong reason.

**4. Ask what the working capital cycle makes checkable.** In a business-to-business company, watch receivables. In a prepaid consumer business, receivables are structurally near zero and tell you nothing; look at deferred revenue, prepayments, accruals and the expense side.

**5. Look for expense growth that tracks revenue too neatly.** Real costs are lumpy. Costs that scale in near-perfect proportion to a suspiciously smooth revenue line are worth an hour.

**6. Concentrate on vendor and customer identity, not just amounts.** Related parties are the load-bearing element of most funded frauds. The counterparty's identity is the question the statements answer worst.

**7. Compare guidance to delivery, on the same definition.** Luckin guided to RMB 2.1 to 2.2 billion of Q4 2019 product revenue. Actual revenue from product sales, when finally reported, was RMB 1,034.5 million, and RMB 979.4 million on the narrower definition the guidance used. Under half the low end either way. The phrase "on the same definition" is the load-bearing part: companies restate their segment definitions more often than they restate their numbers.

**8. Watch what the company does with the numbers.** A capital raise priced off unaudited results that are themselves the whole investment thesis is the highest-stakes moment in the cycle, for both sides.

**9. Distinguish the audited from the unaudited.** Know which document you are reading and what assurance it carries. A 6-K exhibit is not a 20-F.

**10. Separate what is found, alleged, and claimed.** Keep the categories from the table above. It will not make you money directly, but it will stop you from being confidently wrong in public.

---

## When this matters to you

Most readers will never audit a company. The transferable skill here is narrower and more useful than fraud detection: it is the habit of converting a financial claim into a physical one and then asking whether the physical one is possible.

Luckin's filings were, in the relevant sense, honest about the impossible thing. The company published its store count and it published its item count. It did not hide the inputs to the calculation that undid it; it simply presented the *ratio* that flattered it (revenue per store) rather than the ratio that did not (items per store per day). Nobody needed access, or a source, or a leak. They needed to divide.

That generalises. A subscription business that reports revenue and subscriber count is claiming an average revenue per user. A lender reporting loan growth and headcount is claiming loans originated per underwriter per week. A logistics company reporting revenue and vehicle count is claiming trips per vehicle per day. In each case the company gives you the numerator and the denominator and reports neither ratio. Compute it, plot it over eight quarters, and see whether it is doing something a physical operation can do.

And when the ratio does something implausible, the last question is the Luckin question: *if this revenue is not real, where did the money come from, and where did it go?* A fraud that cannot answer that question leaves a hole in the cash. A fraud that can answer it leaves a trail through the expense line instead. Either way, the money went somewhere, and somewhere is a place you can look.

---

## Sources & further reading

**Primary documents**

- U.S. Securities and Exchange Commission, *SEC v. Luckin Coffee Inc.*, Complaint, No. 1:20-cv-10631 (S.D.N.Y., filed 16 December 2020). The source for the three purchasing schemes, the quarter-by-quarter fabricated amounts, the internal emails, and the ADS closing prices cited above. [sec.gov/files/litigation/complaints/2020/comp-pr2020-319.pdf](https://www.sec.gov/files/litigation/complaints/2020/comp-pr2020-319.pdf)
- SEC Press Release 2020-319, "Luckin Coffee Agrees to Pay \$180 Million Penalty to Settle Accounting Fraud Charges", 16 December 2020. [sec.gov/newsroom/press-releases/2020-319](https://www.sec.gov/newsroom/press-releases/2020-319)
- Luckin Coffee Inc., "Announces Unaudited Third Quarter 2019 Financial Results", Form 6-K exhibit 99.1, dated 13 November 2019 (furnished 20 November 2019). Source for all as-reported Q3 2019 figures, the key operating data table, and the RMB 7.1477 exchange rate. [EDGAR CIK 0001767582](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001767582&type=6-K)
- Luckin Coffee Inc., "Announces Formation of Independent Special Committee and Provides Certain Information Related to Ongoing Internal Investigation", Form 6-K exhibit 99.1, 2 April 2020. Source for the RMB 2.2 billion preliminary figure and the naming of the chief operating officer.
- Luckin Coffee Inc., "Announces the Substantial Completion of the Internal Investigation", Form 6-K exhibit 99.1, 1 July 2020. Source for the RMB 2.12 billion and RMB 1.34 billion findings, the quarterly split, the 550,000 documents and 60 custodians, and the personnel outcomes.
- Luckin Coffee Inc., "Announces Restatements of Unaudited Second and Third Quarter 2019 Financial Results and Release of Unaudited Fourth Quarter 2019 Financial Results", Form 6-K/A exhibit 99.1, 30 June 2021. Source for every restated figure, the restated operating table, the restated balance sheet, and the RMB 6.9618 exchange rate.
- Luckin Coffee Inc., "Announces Second Quarter 2026 Financial Results", Form 6-K exhibit 99.1, 3 August 2026. Source for the current store count, revenue and listing status.
- Luckin Coffee Inc., "Proposal of Resignation and Removal of the Chairman of the Board", Form 6-K exhibit 99.1, 26 June 2020.
- Luckin Coffee Inc., "Announces the Receipt of Penalty Decisions from the Chinese State Administration for Market Regulation", Form 6-K exhibit 99.1, 23 September 2020. Source for the RMB 61.0 million SAMR fine and the Anti-Unfair Competition Law basis.
- Luckin Coffee Inc., "Restructuring Efforts Move Forward with Commencement of its Chapter 15 Case in the United States", Form 6-K exhibit 99.1, 5 February 2021.
- Luckin Coffee Inc., "Enters into Binding Term Sheet to Settle U.S. Securities Class Action", Form 6-K exhibit 99.1, 21 September 2021.
- Luckin Coffee Inc., "Announces Successful Conclusion of Provisional Liquidation", Form 6-K exhibit 99.1, 7 March 2022. Source for the US\$460 million convertible note restructuring, the JPL appointment date and the discharge order.
- Luckin Coffee Inc., "Successfully Emerges from All Bankruptcy Proceedings", Form 6-K exhibit 99.1, 11 April 2022.
- Luckin Coffee Inc., Annual Report on Form 20-F for the fiscal year ended 31 December 2019, filed 21 September 2021. Source for the auditor history and the material weakness disclosure.
- Luckin Coffee Inc., Annual Report on Form 20-F for the fiscal year ended 31 December 2022, filed 6 April 2023. Source for the reversal of the SEC penalty provision, the 3 February 2022 SEC notice of satisfaction, the US\$175 million final class settlement approved 22 July 2022, the litigation provisions and the JPL protocol.

**Court records**

- *Cohen v. Luckin Coffee Inc.*, Class Action Complaint, No. 1:20-cv-01293 (S.D.N.Y., filed 13 February 2020). This is the source for every quotation from the anonymous report published by Muddy Waters on 31 January 2020, for the 31 January 2020 price move, for Citron Research's long position, and for the quoted passage from the J Capital Research report of 12 February 2020. It is a pleading rather than a finding, and it is cited here as a dated public record of what those reports said, not as evidence that their claims were correct. Available through the free RECAP archive as `gov.uscourts.nysd.532041.1.0.pdf`.
- The case was later consolidated as *In re Luckin Coffee Inc. Securities Litigation*, No. 1:20-cv-01293-JPC-JLC (S.D.N.Y.), the action whose US\$175 million settlement received final approval on 22 July 2022.

**Elsewhere in this series**

- [Round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) for the general mechanics
- [The short seller's playbook: how activists find fraud](/blog/trading/forensic-accounting/the-short-sellers-playbook-how-activists-find-fraud) for the research process rather than this company's statements
- [Related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing)
- [Forensic ratios: DSO, DIO, DPO and margin anomalies](/blog/trading/forensic-accounting/forensic-ratios-dso-dio-dpo-and-margin-anomalies)
- [How an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch)
- [Wirecard: the missing EUR 1.9 billion](/blog/trading/forensic-accounting/wirecard-the-missing-1-9-billion-euros) for the mirror-image case
- [WorldCom: the US\$11 billion capitalization fraud](/blog/trading/forensic-accounting/worldcom-the-11-billion-dollar-capitalization-fraud) for the mirror-image case, and [capitalizing costs to inflate profit](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move) for the mechanic

**A note on numbers in this post.** Every RMB figure is taken from a Luckin filing and converted at the rate that filing itself states: RMB 7.1477 to US\$1.00 for the Q3 2019 release (the 30 September 2019 Federal Reserve H.10 rate) and RMB 6.9618 to US\$1.00 for the restatement (the 31 December 2019 rate). Where the SEC's own US dollar figures differ slightly from those conversions, it is because the SEC used a different rate, not because the underlying RMB amounts differ. Items-per-store-per-day figures are derived by the author from Luckin's published operating tables using the company's own averaging convention, shown in full above.
