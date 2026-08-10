---
title: "Forensic ratios: DSO, DIO, DPO, and the margin anomalies that give manipulation away"
date: "2026-08-09"
publishDate: "2026-08-09"
description: "A practical forensic dashboard for reading a filing: how days-sales-outstanding, days-inventory-outstanding, days-payables-outstanding, peer margin spreads, and the cash-realization ratio expose manipulation, and how to assemble them into a weighted red-flag scorecard."
tags: ["forensic-accounting", "financial-ratios", "dso", "dio", "dpo", "cash-conversion-cycle", "gross-margin", "earnings-quality", "red-flags", "fraud-detection", "working-capital", "peer-analysis"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 54
---

> [!important]
> **TL;DR** — Accounting manipulation almost always inflates a flow on the income statement, and a flow you inflate without receiving cash has to pile up somewhere on the balance sheet. Ratios expressed in *days* are how you find the pile.
>
> - **DSO** (receivables ÷ revenue × 365) counts days to collect. **DIO** (inventory ÷ COGS × 365) counts days goods sit. **DPO** (payables ÷ COGS × 365) counts days before you pay. Each one pairs a balance-sheet stock with the income-statement flow that created it.
> - A ratio in isolation says nothing. A ratio moving **against its own history and against the peer median at the same time** is the signal.
> - The single most useful number on this whole dashboard is **cash realization**: cash from operations divided by net income. Sustained below 1.0 means reported profit is not arriving as money.
> - Different schemes leave different fingerprints. Channel stuffing pushes DSO up and gross margin down. Fictitious revenue pushes DSO up and gross margin *up*. Capitalizing costs pushes DIO up and margin up. The pattern identifies the trick.
> - Before you call anything fraud, kill it with four boring explanations: seasonality, mix shift, a policy change, and an industry-wide move. Most flags die there — and the ones that survive are worth real work.

Every accounting fraud has to solve the same awkward problem: how do you report a sale you did not really make?

Reporting it is the easy part. You write a number on the income statement. The hard part is the *other side of the entry*. Double-entry bookkeeping will not let you create revenue out of nothing — every credit to revenue needs a matching debit somewhere. If a real customer paid you real money, the debit goes to cash and the story is over. If nobody paid you, the debit has to go somewhere else, and in practice it goes to **accounts receivable** (money customers supposedly owe you) or, if you are hiding costs instead of inventing sales, to **inventory** or some capitalized asset.

That constraint is the forensic accountant's best friend. A manipulator can lie about a flow. What they cannot easily do is stop the lie from accumulating as a *stock* on the balance sheet — a growing pile of receivables nobody is paying, or a growing pile of inventory nobody is buying. Cash either shows up or it does not, and when it does not, the balance sheet swells.

Forensic ratios are simply the instruments that measure that swelling. They are not magic and they do not prove anything on their own. What they do is convert a fat, unreadable filing into four or five numbers you can compare against last year and against the company's competitors, so that anomalies become visible rather than buried.

The diagram below is the mental model for the entire post: three ratios, each dividing one balance-sheet stock by one income-statement flow, and one cycle that combines them.

![Diagram showing how DSO pairs accounts receivable with revenue, DIO pairs inventory with COGS, and DPO pairs accounts payable with COGS, combining into the cash conversion cycle](/imgs/blogs/forensic-ratios-dso-dio-dpo-and-margin-anomalies-1.webp)

If you are coming to this cold, the companion post on [the cash conversion cycle and what working capital reveals](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) covers the plumbing in more depth. This post is about the *detective work*: what each divergence implies, how to rule out the innocent explanations, and how to assemble the whole thing into a scorecard you can actually run on a filing in an afternoon.

## Foundations: the building blocks of a forensic ratio

Nothing in this section assumes you have opened a financial statement before. If you have, skim it — the deep material starts in the next section.

### Flows and stocks: the two kinds of number on a financial statement

A company publishes three main statements. Two of them measure fundamentally different kinds of thing, and confusing them is the single most common beginner mistake.

The **income statement** measures *flows*: things that happened over a period of time. "We sold \$690 million of equipment during the year." "The goods we sold cost us \$428 million to make." A flow has no meaning without a period attached — revenue of \$690 million *for the year* is a completely different statement from revenue of \$690 million *for the quarter*.

The **balance sheet** measures *stocks*: things that exist at a single instant, like a photograph. "On 31 December we were owed \$189 million by customers." "On 31 December we held \$120 million of unsold goods in warehouses." A stock has no period; it has a date.

Three balance-sheet stocks matter for our purposes:

- **Accounts receivable (AR)** — money customers owe you for goods or services you have already delivered and already booked as revenue. You made the sale; you have not been paid yet.
- **Inventory** — goods you have paid for or manufactured but have not yet sold. Raw materials, work in progress, and finished goods sitting in a warehouse.
- **Accounts payable (AP)** — money *you* owe suppliers for things you have already received. The mirror image of receivables, pointing the other way.

And two income-statement flows:

- **Revenue** (also called sales, or the "top line") — the value of what you sold during the period.
- **Cost of goods sold (COGS)** — the direct cost of producing the things you sold. Revenue minus COGS is **gross profit**, and gross profit divided by revenue is the **gross margin**, usually quoted as a percentage.

### Why a faked flow leaves a footprint in a stock

Here is the mechanism in one paragraph, because everything else follows from it.

When a company books revenue, accounting requires a matching entry on the asset side. If the customer paid immediately, cash rises. If the customer has not paid yet, accounts receivable rises, and the company is betting that the cash will arrive later. Both are legitimate — selling on credit is how almost all business-to-business commerce works. But notice what happens when the sale is *not real*, or is real but was pulled forward from a future period by pressuring a distributor to take goods it does not need. Revenue rises. Receivables rise. Cash does not. Do it once and nobody notices. Do it every quarter and receivables climb relentlessly relative to sales, because the pile never drains.

The same logic runs through inventory. If a company is quietly moving costs off the income statement — capitalizing manufacturing overhead into inventory rather than expensing it, or refusing to write down goods it can no longer sell — then gross margin looks healthy while the inventory balance bloats. The cost did not disappear. It is sitting on the balance sheet waiting to be recognized, and one day it will be, usually all at once, in a quarter management calls "a reset".

> A manipulated income statement is a story. The balance sheet is where the story has to be stored, and stories take up space.

### Days, not dollars: normalizing so you can compare

Raw dollar balances are almost useless for comparison. Receivables of \$189 million tells you nothing until you know how big the company is. A firm with \$50 million of revenue and \$189 million of receivables is in catastrophic trouble; a firm with \$10 billion of revenue and \$189 million of receivables is collecting cash beautifully.

The fix is to convert every balance into **days**. The question we are really asking is: *at the current rate of business, how many days of activity does this pile represent?*

The arithmetic is always the same shape. Take the stock, divide by the flow that produced it to get a fraction of a year, then multiply by 365 to express the fraction in days:

$$\text{Days} = \frac{\text{Balance sheet stock}}{\text{Annual income statement flow}} \times 365$$

That gives the three core ratios:

$$\text{DSO} = \frac{\text{Accounts receivable}}{\text{Revenue}} \times 365$$

$$\text{DIO} = \frac{\text{Inventory}}{\text{Cost of goods sold}} \times 365$$

$$\text{DPO} = \frac{\text{Accounts payable}}{\text{Cost of goods sold}} \times 365$$

**DSO** is *days sales outstanding* — the average number of days between making a sale and collecting the money. **DIO** is *days inventory outstanding* — the average number of days a unit of goods sits in your warehouse before it is sold. **DPO** is *days payables outstanding* — the average number of days you take to pay your own suppliers.

Notice the denominators. DSO uses revenue, because receivables are recorded at the *selling* price. DIO and DPO use COGS, because inventory and trade payables are recorded at *cost*. Matching the numerator to the right denominator is not pedantry: using revenue as the denominator for inventory would systematically understate DIO by whatever the gross margin happens to be, which makes cross-company comparison meaningless. This is the one rule that makes every ratio in this post comparable, and it is the rule people break most often when they build their first screen.

Combine the three and you get the **cash conversion cycle (CCC)**, the number of days between paying for your inputs and being paid for your outputs:

$$\text{CCC} = \text{DSO} + \text{DIO} - \text{DPO}$$

DSO and DIO are days your cash is *tied up*. DPO is days your suppliers are financing you, so it comes with a minus sign. A short cycle means the business funds itself. A long and lengthening cycle means the business is consuming cash to grow, and every extra day has to be paid for out of profits, debt, or equity.

#### Worked example: turning three balance-sheet lines into three day-counts

*The numbers in this and every other Northgate example are **illustrative** — a fictional industrial-equipment maker built to demonstrate the arithmetic cleanly. They are not real company data.*

Here is Northgate Instruments in its first year, all figures in millions of dollars:

| Line | FY1 |
| --- | --- |
| Revenue | \$400 |
| Cost of goods sold | \$240 |
| Accounts receivable (year end) | \$66 |
| Inventory (year end) | \$46 |
| Accounts payable (year end) | \$33 |

Step by step:

1. **DSO** = \$66 ÷ \$400 × 365 = 0.165 × 365 = **60.2 days**. On average, Northgate waits about two months to be paid.
2. **DIO** = \$46 ÷ \$240 × 365 = 0.1917 × 365 = **70.0 days**. A typical unit of inventory sits in the warehouse for ten weeks.
3. **DPO** = \$33 ÷ \$240 × 365 = 0.1375 × 365 = **50.2 days**. Northgate pays its own suppliers in about seven weeks.
4. **CCC** = 60.2 + 70.0 − 50.2 = **80.0 days**.

Read that last number out loud in plain English: *Northgate spends cash on materials roughly 80 days before the corresponding cash comes back from customers.* If Northgate wants to grow revenue, it must fund 80 days of working capital for every extra dollar of sales it adds.

**The intuition:** days convert incomparable dollar balances into a single question — how long is this company's cash trapped?

### What a "peer median" means, and why you cannot skip it

Every threshold in this post is meaningless without a comparison group. A DSO of 100 days is alarming for a supermarket that collects at the till and unremarkable for a heavy-equipment maker selling to governments on 120-day terms.

So the working method is always two comparisons at once:

- **Time series** — the company against its own history, ideally eight to twelve quarters or four to five years, so you can see a trend rather than a point.
- **Cross-section** — the company against the median of four to eight genuine competitors in the same period, so you can tell a company-specific problem from an industry-wide one.

The **median** is deliberately chosen over the mean. In small peer groups one distressed competitor can drag an average badly, and the median ignores it. Where I refer to a "peer median" below, I mean: pick your competitor set, compute the ratio for each of them for the same fiscal period, and take the middle value.

A move that shows up in the time series *and* is absent from the cross-section is the highest-quality signal on this entire dashboard. It says the industry did not change; this company did. The post on [common-size and trend analysis](/blog/trading/forensic-accounting/common-size-and-trend-analysis-making-statements-comparable) is the general version of this discipline.

## 1. DSO: the clock on getting paid

DSO is the first ratio to look at, for a simple reason. Most financial statement fraud is revenue fraud — inventing sales, recognizing them too early, or bullying customers into taking product early — and every one of those variants leaves its residue in receivables.

![Illustrative line chart showing Northgate DSO rising from 60.2 to 100.0 days while the peer median stays flat near 58 days, with the widening gap shaded](/imgs/blogs/forensic-ratios-dso-dio-dpo-and-margin-anomalies-3.webp)

The chart above is the shape you are hunting for. One line is the company. The other is the peer median. If both rise, the industry got slower — customers everywhere are stretching payment, which happens in downturns and tells you about the economy, not about this management team. If only the company's line rises, you have isolated something company-specific, and there are only a handful of things it can be.

### The three ways to compute DSO, and why the answer moves

Practitioners compute DSO at least three different ways, and they give materially different answers. This matters enormously, because a company that changes its own disclosed methodology can manufacture an improvement out of thin air.

**Year-end method.** Take the closing receivables balance and divide by the full year's revenue. Simple, uses only published numbers, and the standard for a first screen. Its weakness is that a single balance-sheet date can be gamed — a company that pushes hard on collections in the last week of December reports a flattered number.

**Average-balance method.** Use the average of opening and closing receivables. This smooths a one-day distortion and is generally the more honest annual figure. It lags a genuine deterioration by roughly half a year, which is exactly why a manipulator prefers it.

**Quarterly annualized method.** Use the quarter's ending receivables and the quarter's revenue, scaled by 91.25 days instead of 365. Far more responsive, and the right tool once you suspect something, because it can show you which specific quarter went wrong.

#### Worked example: three ways to compute DSO, and why the answer moves

Northgate's fourth year, illustrative figures in millions:

| Line | FY3 | FY4 | FY4 Q4 only |
| --- | --- | --- | --- |
| Revenue | \$600 | \$690 | \$230 |
| Accounts receivable (period end) | \$131 | \$189 | \$189 |

1. **Year-end method:** \$189 ÷ \$690 × 365 = **100.0 days**.
2. **Average-balance method:** average AR = (\$131 + \$189) ÷ 2 = \$160. Then \$160 ÷ \$690 × 365 = **84.6 days**.
3. **Quarterly annualized:** \$189 ÷ \$230 × 91.25 = **75.0 days**.

Three defensible methods, three answers spanning 25 days. Which is right?

All of them, for different questions. But look at what a company could do with that spread. Suppose management has historically reported the year-end figure, DSO has deteriorated to 100 days, and this year the annual report quietly switches to "average receivables". Reported DSO improves from 100.0 to 84.6 days — an apparent 15.4-day improvement — with no change whatsoever in the underlying business. Nothing illegal has happened. A reader who did not check the footnote has been misled anyway.

**The intuition:** compare like with like, always recompute the ratio yourself from raw statement lines, and treat any change in a company's own ratio methodology as a flag in its own right.

### What a rising DSO can mean

A DSO that rises 15% or more year over year, with the peer median flat, has a short list of possible causes. Ranked roughly from most innocent to most alarming:

1. **Customer mix shifted.** The company won a large government or enterprise contract that pays on 90-day terms instead of 30. Real, benign, and usually discussed in the management commentary.
2. **Geographic mix shifted.** Expansion into markets where slow payment is normal. Also benign, also usually disclosed.
3. **Terms were deliberately loosened** to win business. Not fraud, but a genuine deterioration in earnings quality — the company is buying revenue with its balance sheet, and the price is real.
4. **A factoring or securitization programme ended.** The company had been selling receivables to a bank for immediate cash; when the programme stops, receivables reappear on the balance sheet. Zero economic change, large DSO change. We will work through this one in detail later, because it catches people out constantly.
5. **Collections are failing** because customers cannot pay. A credit-quality problem, and often the first visible symptom of a customer base in distress.
6. **Revenue was recognized too early** — goods shipped before the customer wanted them, contracts booked before performance obligations were satisfied.
7. **Revenue was recognized that will never be collected**, because the buyer was never going to pay, or because the buyer does not exist.

Only the last three are earnings-quality or fraud findings. The first four are why you do not shout after seeing one number.

### The corroborating test: the allowance for doubtful accounts

There is one cheap follow-up that separates causes 5 to 7 from the innocent ones, and it lives in the footnotes.

Companies must estimate how much of their receivables will never be collected and record an **allowance for doubtful accounts** (sometimes "allowance for credit losses") as an offset. If receivables are genuinely getting older and riskier, that allowance should grow *at least* as fast as gross receivables — the percentage should hold or rise.

So compute the allowance as a percentage of gross receivables for each of the last four or five years. If DSO is climbing while the allowance percentage is *falling*, management is simultaneously telling you two contradictory things: that customers are taking much longer to pay, and that they are more likely to pay than before. Those cannot both be true. That contradiction is worth more than any single ratio on this page, and it is the kind of thing that lives in the footnotes rather than the headline statements — see [the footnotes and MD&A: where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) for where else to look.

## 2. DIO and DPO: the other two clocks

### DIO: the clock on selling what you made

DIO counts the days a unit of inventory sits before it is sold. A rising DIO means goods are accumulating faster than they are moving.

The benign readings are real and common. A company might build stock deliberately ahead of a product launch, or in anticipation of a supply disruption, or because it is entering a new region and needs local warehouses filled before it can sell anything. Manufacturers with long production cycles legitimately carry more inventory than distributors. Seasonality alone can double a retailer's DIO between the third and fourth quarter.

The suspicious readings come in two flavours, and they are distinguishable.

**Obsolescence not recognized.** Goods are unsellable but have not been written down. Accounting requires inventory to be carried at the lower of cost or net realizable value; if the market value has collapsed and the carrying value has not, the company is holding an overstated asset and an understated expense. The tell is DIO rising *while gross margin holds perfectly steady*. A company genuinely struggling to sell product normally discounts it, and discounting compresses margin. Rising inventory with flat margin often means the company simply has not taken the hit yet.

**Costs parked in inventory.** Manufacturing overhead, freight, or even some labour can be capitalized into inventory rather than expensed immediately. Within limits this is normal and required accounting. Pushed aggressively, it moves cost off this year's income statement and onto the balance sheet, inflating both gross margin and inventory at the same time. The tell here is DIO rising *while gross margin rises*, with no plausible pricing story to explain the margin. The [inventory and receivables inflation](/blog/trading/forensic-accounting/inventory-and-receivables-inflation-the-classic-red-flag) post walks through the mechanics of both variants.

A useful cross-check is the ratio of inventory growth to revenue growth. Over any multi-year window, inventory should grow roughly in line with sales. If inventory grows twice as fast as sales for three consecutive years, either demand forecasts have been badly wrong — itself a finding — or something is being stored there that is not really goods.

### DPO: the clock on paying your suppliers

DPO is the least-watched of the three and, in some situations, the most informative, because suppliers know things investors do not. A supplier decides how much credit to extend based on private information: whether your cheques clear, whether your orders are being cancelled, whether their credit insurer will still cover you.

DPO is unusual in that **both directions are informative**.

**A sharp rise** in DPO can mean the company negotiated better terms from a position of strength — a large buyer squeezing its supply chain, which is a real and reportable competitive advantage. It can also mean the company is simply not paying its bills because it does not have the money, which is what stretching payables looks like from the outside during a liquidity squeeze. And increasingly it can mean the company has entered a **reverse factoring** or **supply-chain finance** arrangement, where a bank pays the supplier early and the company repays the bank later. The company gets a longer payment period; the obligation, depending on how it is structured and disclosed, may sit in accounts payable rather than in debt — which flatters both the leverage ratios and operating cash flow. This structure is why DPO deserves a footnote search: look for "supply chain finance", "supplier finance programme", "reverse factoring", or "confirming" in the notes and in the MD&A. The mechanics are covered in [factoring, supplier financing, and hiding debt in plain sight](/blog/trading/forensic-accounting/factoring-supplier-financing-and-hiding-debt-in-plain-sight).

**A sharp fall** in DPO is the underrated signal. Suppliers who have been offering 60-day terms and suddenly demand 30 days, or cash on delivery, have made a credit judgement about the company that the market has not made yet. A DPO that drops meaningfully — my rule of thumb is worse than −15% year over year, absent an explanation — while the company simultaneously draws on its revolving credit facility is a classic pre-distress pattern. The company is being forced to fund with expensive bank debt the working capital its suppliers used to fund for free.

### Reading a divergence: what each move implies

Put the three ratios and the margin together and you get a decoder. Each row is a move you might observe; each row has an innocent reading, a guilty reading, and one cheap test that discriminates between them.

![Grid table decoding what a sharp move in DSO, DIO, DPO or gross margin could mean, showing the benign explanation, the possible manipulation, and the confirming test for each](/imgs/blogs/forensic-ratios-dso-dio-dpo-and-margin-anomalies-2.webp)

The discipline the table encodes is the important part, not the specific cells. **Never move straight from an observation to an accusation.** Every observation gets a benign hypothesis and a guilty hypothesis, and then you go looking for the piece of evidence that only one of them predicts. That is ordinary scientific method applied to a filing, and it is what separates forensic analysis from pattern-matching anxiety.

#### Worked example: the cash conversion cycle in days and in dollars

Illustrative Northgate, four years. Everything here is computed from the five statement lines shown earlier plus their FY2–FY4 equivalents, in millions:

| Line | FY1 | FY2 | FY3 | FY4 |
| --- | --- | --- | --- | --- |
| Revenue | \$400 | \$480 | \$600 | \$690 |
| Cost of goods sold | \$240 | \$288 | \$366 | \$428 |
| Accounts receivable | \$66 | \$86 | \$131 | \$189 |
| Inventory | \$46 | \$55 | \$82 | \$120 |
| Accounts payable | \$33 | \$38 | \$45 | \$43 |
| **DSO (days)** | 60.2 | 65.4 | 79.7 | 100.0 |
| **DIO (days)** | 70.0 | 69.7 | 81.8 | 102.3 |
| **DPO (days)** | 50.2 | 48.2 | 44.9 | 36.7 |
| **CCC (days)** | 80.0 | 86.9 | 116.6 | 165.6 |

The cycle has more than doubled, from 80 days to 165.6 days, while revenue grew 72.5%. All three clocks moved the wrong way at once: customers pay slower, goods sit longer, and suppliers demand payment faster.

Now translate days back into dollars, which is where it stops being an abstraction. Ask: *how much extra cash is trapped in FY4 compared to what would have been trapped if the FY1 ratios had held?*

1. **Receivables.** At FY1's 60.2 days, receivables on FY4 revenue of \$690 million would be 60.2 ÷ 365 × \$690 = **\$113.8 million**. Actual: \$189 million. Extra cash tied up: **\$75.2 million**.
2. **Inventory.** At FY1's 70.0 days, inventory on FY4 COGS of \$428 million would be 70.0 ÷ 365 × \$428 = **\$82.1 million**. Actual: \$120 million. Extra cash tied up: **\$37.9 million**.
3. **Payables.** At FY1's 50.2 days, payables on FY4 COGS would be 50.2 ÷ 365 × \$428 = **\$58.9 million**. Actual: \$43 million. Supplier financing lost: **\$15.9 million**.
4. **Total extra cash absorbed:** \$75.2 + \$37.9 + \$15.9 = **\$129.0 million**.

Set that against Northgate's FY4 reported net income of \$46 million. The working-capital deterioration in a single year consumed nearly **three times** the profit the company reported for that year. Whatever the income statement says, this business did not make money in FY4; it consumed it.

**The intuition:** always convert a ratio deterioration back into dollars and compare it to net income. Days are how you spot the problem; dollars are how you size it.

## 3. Margins: against peers, and against your own history

Margins are the second half of the dashboard, and they answer a different question. The days ratios ask *where is the cash*. Margins ask *is this profitability plausible*.

Three margins, computed from the income statement:

$$\text{Gross margin} = \frac{\text{Revenue} - \text{COGS}}{\text{Revenue}}$$

$$\text{Operating margin} = \frac{\text{Operating income}}{\text{Revenue}}$$

$$\text{Net margin} = \frac{\text{Net income}}{\text{Revenue}}$$

Gross margin is the most forensically useful of the three, because it is the least polluted by financing decisions, tax, and one-off items. It measures one thing: the spread between what you sell for and what it costs you to make. That spread is set by competitive reality — by how differentiated your product is and how efficient your factory is — and competitive reality is stubborn.

Which is why a gross margin that persistently exceeds the peer median is a claim that demands evidence. There are perfectly good explanations. A genuinely differentiated product commands a price premium. A structurally lower cost base — better scale, cheaper energy, a proprietary process — produces the same result. Companies with real moats do earn durable excess margins, and the whole discipline of [economic moats](/blog/trading/equity-research/economic-moats-durable-competitive-advantage) exists to identify them.

But there is a third explanation, which is that the margin is an accounting artefact: costs that should be running through COGS are being capitalized into inventory or into some long-lived asset instead. That produces exactly the same reported margin as a moat does, with none of the underlying economics. The way you tell them apart is cash. A real moat throws off cash. An accounting margin does not.

![Illustrative grouped bar chart comparing Northgate gross margin of 40.0, 40.0, 39.0 and 38.0 percent against a peer median near 34 to 35 percent, with the spread narrowing from six points to three](/imgs/blogs/forensic-ratios-dso-dio-dpo-and-margin-anomalies-6.webp)

### The margin patterns worth investigating

**Margin above peers and widening, with no product story.** The strongest single margin flag. Ask what changed. If the company cannot point to a launch, a price increase, a mix shift towards a premium line, or a specific cost programme, be suspicious.

**Margin suspiciously stable.** Real margins wobble. Input costs move, currencies move, the mix of what you sold moves. A gross margin that prints within 20 basis points of the same number for twelve consecutive quarters, in an industry where competitors swing by two or three points, is not a sign of excellent management. It is a sign that something is being smoothed — usually through reserves, which is its own family of tricks covered in [cookie-jar reserves and big-bath accounting](/blog/trading/forensic-accounting/cookie-jar-reserves-and-big-bath-accounting).

**Margin rising while inventory swells.** The capitalization signature described earlier. Costs left the income statement and went somewhere.

**Margin rising while DSO rises.** The most alarming combination on the whole dashboard, and worth dwelling on. Think about what selling harder actually requires. To move more product you generally discount, or you extend credit to weaker buyers, or you pay distributors incentives. Every one of those *compresses* margin. So a company reporting simultaneously that it is finding it harder to collect and that each sale is more profitable is describing something economically unusual. It is not impossible — a mix shift towards a high-margin product line that happens to be sold to slow-paying customers would do it — but it needs an explanation, and the explanation should be in the filing.

**Margin falling while revenue accelerates.** The channel-stuffing signature. Growth bought with discounts. Less damning than the above, but it tells you the growth is being purchased rather than earned, and purchased growth stops the moment the purchasing stops.

#### Worked example: a three-point margin premium that cost \$129 million

Illustrative Northgate again, FY4. Gross margin was 38.0% against a peer median of 35.0% — a premium of 3.0 percentage points.

1. **What the premium is worth.** On revenue of \$690 million, 3.0 percentage points of extra gross margin = 0.030 × \$690 = **\$20.7 million** of extra gross profit per year.
2. **What it cost.** From the previous worked example, the working-capital deterioration in FY4 absorbed **\$129.0 million** of cash relative to FY1 ratios.
3. **The ratio:** the company gave up roughly \$6.20 of cash for every \$1.00 of margin premium it reported.

Notice also the direction of travel: the premium was 6.0 points in FY1 and FY2, then 4.0, then 3.0. It is *narrowing* while the cash cost is *widening*. That combination argues against the capitalization story and towards a different one — Northgate is discounting to move product, and the discounts are eating the premium while the receivables pile up. That is the fingerprint of channel stuffing rather than cost capitalization, which is exactly the sort of discrimination a full dashboard lets you make and a single ratio does not.

**The intuition:** a margin premium is only real if the business keeps the cash. Price the premium in dollars, price the working capital it consumed in dollars, and compare.

## 4. Revenue versus cash from operations: the divergence that matters most

If you only had time to compute one thing about a company, compute this.

**Cash from operations (CFO)** is the top section of the cash flow statement: the actual cash the business generated from running itself, before capital spending and before financing. It starts from net income and then adds back non-cash charges and adjusts for changes in working capital, which means it explicitly undoes the accrual accounting that revenue recognition depends on. If receivables grew \$58 million during the year, that \$58 million is subtracted, because it is revenue that did not arrive as money. The mechanics are in [reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income).

Over a long enough window, revenue and CFO have to travel together. A company can outrun its cash for a year or two while it grows — that is what growth working capital *is*. It cannot do so indefinitely, because the cash to fund the gap has to come from somewhere, and the sources are finite.

![Illustrative indexed line chart with revenue rising from 100 to 172 while cash from operations falls from 100 to negative 20, the widening gap shaded](/imgs/blogs/forensic-ratios-dso-dio-dpo-and-margin-anomalies-4.webp)

The picture practitioners call "the jaws opening" is the two lines separating and staying separated. Index both series to 100 in a base year, plot four or five years, and look at whether the gap closes. A gap that opens and then closes is a growth investment that paid off. A gap that opens and keeps opening is either a business model that consumes cash structurally — which should be stated plainly by management and funded visibly — or an income statement that is describing transactions the bank account has never seen.

### How to compute the divergence

Two practical formulations, both easy:

**Growth spread.** Over three years, compute cumulative revenue growth and cumulative CFO growth, then subtract. A spread of more than 25 percentage points is, as my rule of thumb, worth investigating. This is a heuristic I use for triage, not an established empirical constant.

**Level ratio.** Compute CFO ÷ revenue for each year and watch the trend. A business converting 12% of revenue into operating cash that drifts to 4% over three years has changed in some fundamental way, and the change is rarely announced.

One caution. CFO is itself manipulable, just less so than net income. The main levers are classification shifting — moving an operating outflow into the investing section, or an investing inflow into operating — and the timing of payables at the balance-sheet date. The post on [cash flow statement manipulation](/blog/trading/forensic-accounting/cash-flow-statement-manipulation-classification-shifting) covers those. Treat CFO as harder to fake, not impossible to fake.

## 5. Cash realization: the ratio that anchors everything

The **cash realization ratio** is cash from operations divided by net income:

$$\text{Cash realization} = \frac{\text{Cash from operations}}{\text{Net income}}$$

It answers the most direct question you can ask a set of accounts: *for every dollar of profit you reported, how many dollars actually arrived?*

A ratio at or above 1.0 means reported profit is fully backed by cash. In fact, most healthy companies run comfortably *above* 1.0, because depreciation and amortization are real expenses on the income statement that consume no cash in the current period, so they are added back. A capital-intensive manufacturer with heavy depreciation might normally run at 1.4 or 1.6. A software company with little depreciation and customers who prepay might run at 1.2. What matters is the level relative to that company's own normal, and the direction.

![Illustrative bar chart of cash realization falling from 1.07 to 0.91 to 0.52 to negative 0.13 across four years, with a dashed reference line at 1.0](/imgs/blogs/forensic-ratios-dso-dio-dpo-and-margin-anomalies-5.webp)

### How to read it without over-reading it

Single years are noisy, and there are entirely legitimate reasons for one bad year:

- A genuine growth spurt where working capital had to be funded in advance of the sales it supports.
- A one-time legal settlement or restructuring payment that hits cash in one year and income in another.
- A large tax payment timing difference.
- The first year after an acquisition, when the cash flow statement and the income statement cover different scopes.

So the practical version is a **three-year average**, which averages out the timing noise and keeps the trend. My rule of thumb — again, mine, not an established constant — is that a three-year average below 0.8 in a mature, profitable business warrants real investigation, and a three-year average below 0.5 is a serious finding. For a genuinely high-growth business burning working capital deliberately, the same numbers mean much less, which is why the peer comparison stays mandatory.

There is a deeper version of this idea. The gap between net income and cash from operations is, definitionally, **accruals** — the non-cash portion of reported earnings. A large and persistent accrual component is associated with weaker subsequent earnings, a relationship first documented in the academic literature in the 1990s and explored further in [quality of earnings: accruals, one-offs, and red flags](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags). Cash realization is the crude, back-of-envelope version of that measure, and for triage purposes the crude version is usually enough.

#### Worked example: the four-year accrual gap

Illustrative Northgate, in millions:

| Line | FY1 | FY2 | FY3 | FY4 | **Total** |
| --- | --- | --- | --- | --- | --- |
| Net income | \$28 | \$34 | \$42 | \$46 | **\$150** |
| Cash from operations | \$30 | \$31 | \$22 | −\$6 | **\$77** |
| **Cash realization** | 1.07 | 0.91 | 0.52 | −0.13 | **0.51** |

Step by step:

1. **FY1:** \$30 ÷ \$28 = **1.07**. Healthy. Profit arrived, plus a little more from depreciation add-backs.
2. **FY2:** \$31 ÷ \$34 = **0.91**. Slightly below 1.0. On its own, unremarkable — one year of working-capital build during 20% revenue growth.
3. **FY3:** \$22 ÷ \$42 = **0.52**. Now only half the reported profit is arriving as cash, in a year when reported profit grew 24%. Two consecutive years of deterioration is a trend.
4. **FY4:** −\$6 ÷ \$46 = **−0.13**. The company reported its highest-ever profit and *consumed* cash from operations. This is the point at which the accounts are describing two different companies.
5. **Four-year totals:** \$150 million of cumulative reported profit produced \$77 million of cumulative operating cash. Cash realization of **0.51** across the whole period. The **accrual gap** is \$150 − \$77 = **\$73 million** — profit that exists on the income statement and has never existed in the bank.

The three-year average across FY2 to FY4 is (0.91 + 0.52 − 0.13) ÷ 3 = **0.43**, well under my 0.8 threshold and under the 0.5 serious line.

**The intuition:** cumulative profit and cumulative cash should converge over a multi-year window. A gap that grows every single year is the accounts telling you the earnings are not real, in the plainest language they have.

## 6. Rule out the boring explanation first

This is the section that separates a useful analyst from an alarmist one, and it is where most retail forensic work goes wrong. Every ratio flag in this post has at least one entirely innocent cause, and innocent causes are far more common than fraud. A screen that flags 200 companies and finds two frauds has also defamed 198 honest ones in your head.

So run every flag through four gates before it becomes a finding.

![Diagram of a four-gate gauntlet - seasonality, mix shift, policy change, and industry-wide check - each with an exit for flags that are explained, leading to a terminal box for flags that survive all four](/imgs/blogs/forensic-ratios-dso-dio-dpo-and-margin-anomalies-8.webp)

### Gate 1: Seasonality

Most businesses are not uniform across the year. A retailer's inventory peaks before the holidays and collapses after. A toy manufacturer ships almost everything in the second half. An agricultural processor's inventory tracks the harvest. Comparing the fourth quarter to the third quarter in any of these businesses produces enormous swings that mean nothing at all.

The rule is absolute: **compare the same quarter one year earlier, never the immediately preceding quarter.** If you must look sequentially, look at the four-quarter rolling average.

#### Worked example: a seasonal DSO spike that means nothing

Bramley Tools is an illustrative fictional company that ships most of its product in the fourth quarter, ahead of the construction season. In millions:

| Period | Revenue | Period-end AR | Quarterly DSO |
| --- | --- | --- | --- |
| Q3 this year | \$150 | \$90 | 54.8 days |
| Q4 this year | \$230 | \$181 | 71.8 days |
| Q4 last year | \$190 | \$149 | 71.6 days |

The arithmetic, using 91.25 days per quarter:

1. **Q3 this year:** \$90 ÷ \$150 × 91.25 = **54.8 days**.
2. **Q4 this year:** \$181 ÷ \$230 × 91.25 = **71.8 days**.
3. **Q4 last year:** \$149 ÷ \$190 × 91.25 = **71.6 days**.

The sequential comparison — Q3 to Q4 — shows DSO exploding by 17 days, a 31% jump, which looks like a five-alarm fire. The year-over-year comparison of like quarters shows DSO essentially unchanged: 71.8 against 71.6, a difference of 0.2 days. There is nothing here. The Q4 receivables balance is high because Q4 shipments are high and most of them were invoiced in the final weeks. Every year looks like this.

**The intuition:** a seasonal business will generate a false positive every single year at the same point in the calendar. Anchor every comparison to the same quarter a year earlier and the false positive disappears.

### Gate 2: Mix shift

Did the composition of the business change? Four common versions:

- **A new geography.** Payment norms differ enormously by country. Expanding into a market where 90-day terms are standard will raise consolidated DSO with no deterioration in collections anywhere.
- **A new channel.** Selling direct to consumers (paid at the point of sale) versus through distributors (paid on terms) produces completely different DSO. Shifting mix between them moves the ratio mechanically.
- **A new product line** with different margins and different inventory characteristics.
- **An acquisition consolidated mid-year.** This one is genuinely treacherous. If a company acquires a business in month nine, the balance sheet at year end includes 100% of the acquired company's receivables, while the income statement includes only three months of its revenue. DSO computed from those two numbers is meaningless and will look terrible. The fix is to use pro-forma revenue for the full year, or to compute the ratio on the acquirer's pre-existing business only, if the segment disclosure allows it.

### Gate 3: Policy or structure change

Did something change about how the company finances or reports its working capital, rather than about the working capital itself? The big three:

- **Receivables factoring or securitization starting or stopping.** Covered in the worked example below.
- **A supply-chain finance programme starting or stopping**, which moves DPO sharply in either direction.
- **A change in revenue recognition policy or an accounting standard transition**, which can shift the timing of when receivables are recognized without any change in commercial behaviour.

All three are disclosable and all three are usually disclosed — in the footnotes, not in the press release.

#### Worked example: a DSO jump with no fraud in it

Calder Components is a second illustrative fictional company. Its numbers, in millions:

For the last several years, Calder has sold \$50 million of its receivables to a bank just before each year end. The bank pays Calder cash immediately and collects from the customers later. Because the receivables have legally been sold, they leave Calder's balance sheet.

| Scenario | Revenue | Year-end AR | DSO |
| --- | --- | --- | --- |
| With the factoring programme | \$600 | \$99 | 60.2 days |
| Programme discontinued | \$600 | \$149 | 90.6 days |

1. **With factoring:** \$99 ÷ \$600 × 365 = **60.2 days**.
2. **Without factoring:** the \$50 million stays on the balance sheet, so AR is \$99 + \$50 = \$149 million. Then \$149 ÷ \$600 × 365 = **90.6 days**.

DSO jumps by 30.4 days — a 51% increase — and it would light up every screen in the market. Yet not one customer is paying more slowly, not one dollar of revenue is fake, and not one thing about the business has deteriorated. The only change is that Calder stopped renting the bank's balance sheet.

The forensic point cuts both ways, and the second direction is the important one. If a company *starts* a factoring programme, the same arithmetic runs in reverse: DSO falls by 30 days and operating cash flow gets a one-time boost as the receivables convert to cash. A company whose collections are genuinely deteriorating can hide it for a year or two by factoring an increasing share of its book. So the correct habit is to search the footnotes for the words "factoring", "securitization", "sold receivables", or "transfers of financial assets" *every time* DSO moves sharply in either direction, and to recompute the ratio on a like-for-like basis before drawing any conclusion.

**The intuition:** always ask whether the ratio moved because the business changed or because the reporting perimeter changed. The footnotes answer it in about two minutes.

### Gate 4: Industry-wide

Compute the same ratio for four to eight genuine competitors over the same period. If the peer median moved the same way at the same time, you have found a macro or industry story, not a company story. Customers stretching payment across an entire sector during a credit tightening is a real and interesting observation — but it is an observation about the sector.

A flag that survives all four gates is a finding. In my experience most do not, and that is the system working correctly.

## 7. Three schemes, three ratio fingerprints

Once a flag survives the gauntlet, the next question is *which trick*. This is where the dashboard earns its keep, because the ratios do not just say "something is wrong" — read together, they point at specific mechanisms. Each scheme has to distort a different combination of lines, and the combination is its fingerprint.

![Grid table showing how DSO, DIO, DPO, gross margin and cash realization each move under channel stuffing, fictitious revenue, and cost capitalization](/imgs/blogs/forensic-ratios-dso-dio-dpo-and-margin-anomalies-7.webp)

### Channel stuffing and bill-and-hold

The company pushes product into its distribution channel that end customers have not asked for, typically at quarter end, often sweetened with discounts, extended payment terms, or a right of return. Or it books revenue on goods it has manufactured but not shipped, holding them in its own warehouse for a customer who has not really committed — "bill and hold".

The fingerprint: **DSO up** (distributors have not paid and often have no obligation to until they resell), **DIO down first then rebounding** (goods left the warehouse, then came back as returns), **gross margin down** (the discounts that bought the order), and **cash realization falling below 1.0**.

The distinctive tell is temporal: a spike in the final weeks of a quarter followed by an air pocket in the next one, because the channel is now full and cannot absorb the normal run rate. Watch for a fourth quarter that beats guidance and a first quarter that misses badly with an explanation about "channel inventory normalization". The full anatomy is in [revenue recognition games: how tomorrow's sales become today's profit](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold).

### Fictitious revenue and round-tripping

No goods move at all. The company records sales to entities that do not exist, to entities it secretly controls, or to counterparties in a circular arrangement where it funds the "customer's" purchase — round-tripping.

The fingerprint is different in an instructive way: **DSO up, often to absurd levels** (nobody will ever pay), **DIO roughly unchanged** (no real goods left the warehouse, so inventory is undisturbed), **gross margin up** (a fake sale carries no real cost, so it drops straight into gross profit), and **cash realization falling hard, potentially negative**.

The DIO behaviour is the discriminator. Channel stuffing at least involves real product moving; fictitious revenue does not, so inventory sits there while revenue and receivables balloon around it. And gross margin moving *up* rather than down is the signature that separates the two schemes cleanly. See [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) for the structures involved.

### Costs capitalized instead of expensed

The company takes an expense that belongs on the income statement and records it as an asset instead — into inventory, into property and equipment, into capitalized software, into some intangible. The cost still exists; it has just been moved to a place where it will be recognized slowly over future years instead of immediately.

The fingerprint: **DSO roughly unchanged** (revenue is real, customers are paying), **DIO up** or capital expenditure rising faster than revenue, **gross margin up** (real costs left the income statement), and **cash realization falling below 1.0** — although here the drop is often milder, because in the cash flow statement a capitalized cost usually appears as an investing outflow rather than an operating one, which is precisely why it flatters operating cash flow.

That last point is worth underlining. Cost capitalization is the scheme that damages cash realization least, because it shifts cash outflows from the operating section to the investing section. To catch it, compare operating cash flow to **free cash flow** (operating cash flow minus capital expenditure). A company whose CFO holds up while its free cash flow collapses is spending the money; it has just chosen a section of the statement where fewer people look. [Capitalizing costs to inflate profit](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move) is the deep treatment.

## 8. The red-flag scorecard

Now assemble it. The point of a scorecard is not that the total number is meaningful in some scientific sense — it is not. The point is that it forces you to check every dimension rather than fixating on whichever ratio you happened to notice first, and it makes your reasoning reproducible, so that when you disagree with yourself six months later you can see exactly which input changed.

![Grid table of the eight-flag red-flag scorecard with the measure, rule-of-thumb trigger and point weight for each flag, and four coloured score bands underneath](/imgs/blogs/forensic-ratios-dso-dio-dpo-and-margin-anomalies-9.webp)

Here is the scorecard in text form, so you can copy it into a spreadsheet:

| # | Flag | What you measure | Rule-of-thumb trigger | Points |
| --- | --- | --- | --- | --- |
| 1 | Receivables running | Change in DSO, year over year | More than +15% | 2 |
| 2 | Slower than the industry | DSO minus peer median DSO | More than +20 days | 2 |
| 3 | Inventory piling up | Change in DIO, year over year | More than +20% | 2 |
| 4 | Suppliers pulling back | Change in DPO, year over year | Worse than −15% | 1 |
| 5 | Margin nobody else earns | Gross margin minus peer median | More than +5 pts and widening | 2 |
| 6 | Profit without cash | CFO ÷ net income, 3-year average | Below 0.8 | 3 |
| 7 | Growth without cash | Revenue growth minus CFO growth, 3 years | More than 25 pts | 2 |
| 8 | Receivables outrunning sales | Receivables growth ÷ revenue growth | More than 1.5× | 2 |

Maximum 16 points. Suggested bands: **0–3 clean**, **4–7 watchlist**, **8–11 serious**, **12–16 assume manipulation until disproven**.

### An honest word about these thresholds

**Every number in that table is my rule of thumb, not an established empirical constant.** They come from what has been useful to me as a triage filter; they are not calibrated against a labelled dataset of frauds and non-frauds, and I am not aware of any published study that validates this particular combination and weighting. Treat them as a starting point to be tuned to your sector, not as scientific cut-offs. In a business with 20-day payment terms, +20 days versus the peer median is enormous; in aerospace it may be within normal contract variation.

Two of the flags have some external anchoring, and it is worth being precise about what that anchoring does and does not cover. In Messod Beneish's 1999 study of earnings manipulation, the mean *days-sales-in-receivables index* — essentially this year's DSO divided by last year's — was **1.465** for the 50 manipulating firms in his sample against **1.031** for the 1,708 non-manipulators, and the mean *gross-margin index* was **1.193** against **1.014** (Beneish, 1999, Table 2). Separately, Richard Sloan's 1996 paper established that the accrual component of earnings is markedly less persistent than the cash-flow component — the foundation of what is now called the accruals anomaly.

Those two findings support the *direction* of my flags 1, 5 and 6: receivables growing faster than sales, and profit not backed by cash, really do carry information about manipulation and about future earnings. They do not validate my specific trigger levels or my point weights. Those remain my judgement.

### Why the weights are what they are

Flag 6 — cash realization — carries 3 points because it is the hardest to fake and the most direct. Everything else on this list is an inference about *why* cash might not be arriving. Cash realization measures whether it arrived.

Flags 1, 2, 3, 5, 7 and 8 carry 2 points each because each independently captures a real dimension of the problem, and because they overlap: flags 1 and 8 will often fire together, as will 6 and 7. That overlap is deliberate. A scheme that is large enough to matter tends to trip several correlated flags at once, and the correlated firing is itself informative.

Flag 4 — DPO contraction — carries only 1 point because it is the noisiest. Payment terms move for many reasons, including a company simply deciding to take early-payment discounts because it has spare cash, which is a *good* sign wearing a bad-sign costume.

### Read the pattern, not the number

This is the most important paragraph in the section. A score of 10 assembled from five flags at 2 points each is a different disease from a score of 10 assembled from cash realization plus DSO plus receivables growth. The first pattern is diffuse — a business under general working-capital pressure, which might be a cyclical downturn. The second is concentrated on the revenue-to-cash pathway specifically, which is the pathway revenue fraud has to travel.

So read the scorecard in three passes:

1. **The total** tells you whether to spend more time.
2. **Which flags fired** tells you which fingerprint from the previous section you are looking at.
3. **Which flags did *not* fire** is often the most diagnostic of all. In the worked example below, the margin flag stays silent, and that silence is what argues for channel stuffing rather than cost capitalization.

#### Worked example: scoring illustrative Northgate

Running the scorecard on the FY4 filing, using every figure computed earlier in this post.

| # | Flag | Northgate's value | Trigger | Fires? | Points |
| --- | --- | --- | --- | --- | --- |
| 1 | DSO change YoY | 79.7 → 100.0 days = +25.5% | More than +15% | Yes | 2 |
| 2 | DSO vs peer median | 100.0 − 58.0 = +42.0 days | More than +20 days | Yes | 2 |
| 3 | DIO change YoY | 81.8 → 102.3 days = +25.1% | More than +20% | Yes | 2 |
| 4 | DPO change YoY | 44.9 → 36.7 days = −18.3% | Worse than −15% | Yes | 1 |
| 5 | Gross margin vs peers | 38.0% − 35.0% = +3.0 pts, narrowing | More than +5 pts and widening | No | 0 |
| 6 | Cash realization, 3-yr avg | (0.91 + 0.52 − 0.13) ÷ 3 = 0.43 | Below 0.8 | Yes | 3 |
| 7 | Revenue growth − CFO growth | +72.5% − (−120.0%) = 192.5 pts | More than 25 pts | Yes | 2 |
| 8 | AR growth ÷ revenue growth | +186.4% ÷ +72.5% = 2.57× | More than 1.5× | Yes | 2 |

The arithmetic behind the two least obvious rows:

- **Flag 7.** Revenue grew from \$400 million to \$690 million, which is (690 ÷ 400) − 1 = +72.5%. CFO went from \$30 million to −\$6 million, which is (−6 ÷ 30) − 1 = −120.0%. The spread is 72.5 − (−120.0) = **192.5 percentage points**.
- **Flag 8.** Receivables grew from \$66 million to \$189 million, which is +186.4%. Revenue grew +72.5%. The ratio is 186.4 ÷ 72.5 = **2.57×**. Receivables grew more than two and a half times as fast as the sales that supposedly created them.

**Total: 14 of 16 points**, which lands in the top band.

Now do the third pass — read the pattern. Seven of eight flags fired, and the one that stayed silent is the margin flag, because Northgate's gross margin premium is *narrowing*, not widening. Look back at the fingerprint table. A narrowing margin alongside exploding receivables and collapsing cash is not the cost-capitalization signature, which would show margin *expanding*. It is the channel-stuffing signature: growth bought with discounts, shipped to distributors who are not paying, funded by a balance sheet that has absorbed \$129 million of extra working capital in a single year.

That is a specific, testable hypothesis, and it tells you exactly what to look for next: quarter-end revenue concentration, distributor inventory disclosures, return-rights language in the revenue-recognition footnote, and whether the following first quarter comes in soft.

**The intuition:** the score decides whether you keep reading. The pattern of which flags fired decides what you read next.

## Common misconceptions

**"A high DSO is bad."** No — a high DSO is *normal* in some industries and abnormal in others. Construction, capital equipment, pharmaceutical wholesale, and government contracting all run structurally high DSO for entirely legitimate reasons. What matters is the level relative to the company's own history and relative to a genuine peer group, and the *direction of change*. Absolute levels are almost meaningless across sectors.

**"Cash flow cannot be manipulated."** It is harder to manipulate than net income, not impossible. Classification shifting moves outflows from the operating section to the investing or financing sections. Stretching payables across the balance-sheet date flatters operating cash flow at the cost of the following period. Selling receivables converts a working-capital drag into operating cash inflow. Securitizing future revenue streams can bring cash forward. "Harder to fake" is the right mental model; "unfakeable" is not.

**"If the auditor signed off, the ratios must be fine."** An audit provides reasonable assurance that the statements are free of material misstatement under the applicable framework. It is not a fraud investigation, it is sample-based, and it largely relies on management representations for exactly the estimates that ratio analysis probes — the allowance for doubtful accounts, inventory obsolescence reserves, and the completeness of revenue cut-off. Several of the largest frauds on record carried clean audit opinions right up until they did not. [How an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) covers the boundary honestly.

**"One bad ratio is enough to short a stock."** It is not, and this misconception costs people real money. Ratio flags identify companies deserving *more work*, and the base rate of actual fraud among flagged companies is low. Many flagged companies are simply managing working capital badly, or growing fast, or going through a mix shift. Confusing a screening signal with a conclusion is the single most expensive error in this discipline, and [defining your invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) is the antidote.

**"A rising gross margin is always good news."** Margin expansion is good news when it comes with cash. Margin expansion accompanied by rising inventory, rising receivables, and falling cash realization is one of the more reliable fraud signatures in the literature, precisely because moving costs to the balance sheet raises margin and consumes cash simultaneously.

**"The scorecard total is the answer."** The total is a triage device. It decides how much of your time a company deserves. The composition of the score — which flags fired, which did not, and in what combination — is what carries the diagnostic information. Two companies scoring 10 can have entirely different problems.

## How it shows up in real markets

The illustrative companies above were built to make the arithmetic clean. Real cases are messier, and the four below are chosen because each one isolates a different part of the dashboard.

### 1. Under Armour: six quarters of pulling tomorrow's orders into today

On 3 May 2021 the SEC announced a settled administrative proceeding against Under Armour, charging the company with [disclosure failures](https://www.sec.gov/newsroom/press-releases/2021-78) relating to its use of "pull forward" sales. According to the SEC's order, for six consecutive quarters beginning in the third quarter of 2015, Under Armour accelerated — "pulled forward" — a total of **\$408 million** in existing orders that customers had asked to be shipped in future periods. The company agreed to pay a **\$9.0 million** civil penalty and neither admitted nor denied the findings.

One nuance matters enormously here, and glossing over it would be dishonest. The SEC's case was about **disclosure**, not about the accounting itself: the settlement did not include an allegation that the sales during these periods failed to comply with generally accepted accounting principles. The orders were real orders from real customers. What the SEC said was missing was a clear explanation to investors that revenue growth was being supported by shipping future demand early.

That is precisely why this case belongs in a post about ratios rather than about accounting rules. A pull-forward that is entirely GAAP-compliant still consumes future demand and still parks the proceeds in receivables rather than cash. No rule was necessarily broken, and the working-capital signature still appears in the filings. If you want to see it for yourself, pull Under Armour's 10-K filings covering fiscal 2015 through 2017 from EDGAR and compute receivables growth against revenue growth year by year — flag 8 on the scorecard — rather than taking anybody's word for the trend, including mine.

The lesson generalizes: your dashboard is not a legality detector. It is a *sustainability* detector. It flags growth that has been borrowed from the future regardless of whether borrowing it was permitted.

### 2. Sunbeam: bill-and-hold and a third of a year's profit

On 15 May 2001 the SEC filed a civil action in federal court in Miami against five former Sunbeam officers and the Arthur Andersen engagement partner on its audits ([Litigation Release No. 17001](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-17001)). The SEC alleged that to boost 1997 income, Sunbeam's management caused the company to recognize revenue on sales — including **bill-and-hold sales**, where revenue is booked on goods invoiced but not shipped — that did not meet the applicable accounting rules.

The magnitude is the part worth remembering. Per the SEC, **at least \$60 million of Sunbeam's reported \$189 million in 1997 earnings from continuing operations before income taxes came from accounting fraud** — very nearly a third of the reported number. Sunbeam restated on Form 10-K/A filed 12 November 1998, and the restatement spanned the fourth quarter of 1996 through the first quarter of 1998.

Bill-and-hold is the purest form of the DSO signature described earlier in this post. Revenue is recognized, a receivable is created, and no goods have moved and no cash has arrived. A reader running the dashboard would have seen receivables outrunning sales and cash realization deteriorating well before the restatement made it official. The detailed anatomy of the scheme is in [revenue recognition games](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold).

### 3. Diamond Foods: a margin built from a cost not yet recorded

On 9 January 2014 the SEC [charged Diamond Foods](https://www.sec.gov/newsroom/press-releases/2014-4) and two former executives over an accounting scheme to boost reported earnings growth. The SEC alleged that former CFO Steven Neil directed an effort to underreport money paid to walnut growers by delaying the recording of payments into later fiscal periods, affecting quarters in fiscal 2010 and 2011. In internal emails, according to the SEC, Neil referred to these commodity costs as a "lever" to manage earnings. Diamond Foods agreed to pay **\$5 million**; collectively Diamond, former CEO Michael Mendes and Neil paid a total of **\$5,250,000** in penalties, with Mendes paying \$125,000. Diamond restated its results in November 2012, and its shares fell from a 2011 high of about \$90 to roughly \$17.

This case is the reason margin sits on the dashboard next to the days ratios. Nothing here involved fake sales. Revenue was real, customers paid, and DSO would have told you nothing at all. The distortion was entirely on the *cost* side: understate what you paid for walnuts and gross margin rises, because gross margin is revenue minus COGS. A dashboard built only from DSO and cash realization would have been comparatively quiet. A dashboard that also asked *why is this company's gross margin expanding faster than its peers, and what changed in its input costs* had a question to ask.

### 4. What the research actually supports, and what it does not

It is worth closing the case list with the honest state of the evidence, because a lot of writing in this area overstates it.

Two findings are solid and replicated. First, from Beneish's 1999 work: firms subsequently identified as earnings manipulators had a mean days-sales-in-receivables index of **1.465**, against **1.031** for non-manipulators, and a mean gross-margin index of **1.193** against **1.014** (Table 2; sample of 50 manipulators and 1,708 non-manipulators, drawn from firms subject to SEC accounting enforcement actions or identified as manipulators in the press between 1987 and 1993). Receivables running ahead of sales, and margins moving in ways peers do not, genuinely carry information. Second, from Sloan's 1996 paper: the accrual component of earnings is less persistent than the cash-flow component, which is the formal statement of why cash realization deserves its 3 points.

What the research does *not* provide is a validated cut-off. Beneish's own model identifies roughly half the manipulators in his sample before public discovery — useful, and nowhere near certainty. There is no published constant that says "DSO up 15% means fraud". My thresholds are triage heuristics standing on top of findings about direction, and the honest way to use them is as a queue for further work, never as a verdict.

## When this matters to you

You will almost never be the person who proves a fraud. Proving one requires subpoena power, bank records, and access to people who will talk — none of which an outside analyst has. What ratio analysis gives you is something more modest and far more useful: the ability to decide, in about an hour with a filing and a spreadsheet, whether a company's reported earnings are being converted into money, and if not, why not.

That decision is worth making in several ordinary situations. If you are considering owning a stock, the dashboard in this post is a cheap way to find out whether the earnings you are paying a multiple of have ever existed as cash. If you are extending credit to a customer or evaluating a supplier you depend on, DPO contraction and cash realization deterioration are early warnings that arrive before a credit rating changes. If you work inside a company, the same ratios tell you whether your own employer's growth is being funded by customers or by a balance sheet that is quietly filling up.

The honest limits are worth stating plainly. These ratios generate far more false positives than true findings. They are backward-looking, arriving with the reporting lag of a filing. They can be defeated for a year or two by a company willing to factor receivables or restructure its supplier financing. And they say nothing about valuation — a company with immaculate cash realization can still be a terrible investment at the wrong price.

What they do reliably is stop you from being surprised in one specific way: by a company whose profits were an accounting opinion the entire time. That is a narrow protection, but it covers a disproportionate share of the situations where investors lose everything rather than merely losing money.

This is educational material about how to read financial statements, not investment advice, and nothing here is a recommendation to buy or sell any security.

## Sources & further reading

**A note on the numbers in this post.** Every figure attached to Northgate Instruments, Bramley Tools and Calder Components is **illustrative** — those are fictional companies built to make the arithmetic legible, and none of their numbers are real data. Every figure attached to a named real company is sourced below. The scorecard thresholds are my own rules of thumb and are labelled as such where they appear.

**Primary sources behind the headline figures**

- [SEC Charges Under Armour Inc. With Disclosure Failures](https://www.sec.gov/newsroom/press-releases/2021-78) — U.S. Securities and Exchange Commission, press release 2021-78, 3 May 2021. Source for the \$408 million of pulled-forward orders across six consecutive quarters beginning Q3 2015, and the \$9.0 million civil penalty. The underlying order is [Securities Act Release No. 33-10940](https://www.sec.gov/files/litigation/admin/2021/33-10940.pdf); Under Armour's own description of the settlement is in its [8-K press release](https://www.sec.gov/Archives/edgar/data/1336917/000133691721000021/a53uaasecsettlementpressre.htm), which is also the source for the point that the settlement concerned disclosure and did not include an allegation that the sales failed to comply with GAAP.
- [SEC v. Albert J. Dunlap, Russell A. Kersh, Robert J. Gluck, Donald R. Uzzi, Lee B. Griffith, and Phillip E. Harlow](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-17001) — SEC Litigation Release No. 17001, 15 May 2001. Source for the bill-and-hold allegations and for at least \$60 million of Sunbeam's reported \$189 million of 1997 pre-tax earnings from continuing operations arising from accounting fraud, and for the Form 10-K/A restatement filed 12 November 1998.
- [SEC Charges Diamond Foods and Two Former Executives Following Accounting Scheme to Boost Earnings Growth](https://www.sec.gov/newsroom/press-releases/2014-4) — SEC press release 2014-4, 9 January 2014. Source for the delayed recording of walnut grower payments across fiscal 2010 and 2011, the "lever" characterization, the \$5 million Diamond Foods penalty and the \$5,250,000 collective total.
- Messod D. Beneish, "The Detection of Earnings Manipulation", *Financial Analysts Journal* 55(5), 1999, pp. 24–36 ([publisher listing](https://www.tandfonline.com/doi/abs/10.2469/faj.v55.n5.2296); [working-paper version, June 1999](https://www.calctopia.com/papers/beneish1999.pdf)). Table 2 of the working-paper version is the source for the mean DSRI of 1.465 versus 1.031 and mean GMI of 1.193 versus 1.014, across 50 manipulators and 1,708 non-manipulators.
- Richard G. Sloan, "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About Future Earnings?", *The Accounting Review* 71(3), 1996, pp. 289–315 ([publisher listing](https://publications.aaahq.org/accounting-review/article/71/3/289/18989)). Source for the finding that the accrual component of earnings is less persistent than the cash-flow component.
- [EDGAR full-text search](https://www.sec.gov/edgar/search/) — U.S. SEC. Where to pull the 10-K and 10-Q filings you need to compute every ratio in this post yourself, and where to grep the footnotes for "factoring", "securitization", "supply chain finance" and "sold receivables".

**Further reading on this blog**

- [The cash conversion cycle: what working capital reveals before earnings do](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) — the plumbing behind DSO, DIO, DPO and the cycle itself.
- [Inventory and receivables inflation: the classic red flag](/blog/trading/forensic-accounting/inventory-and-receivables-inflation-the-classic-red-flag) — the two balance-sheet lines this whole dashboard watches, and how each is overstated.
- [Common-size and trend analysis: making statements comparable](/blog/trading/forensic-accounting/common-size-and-trend-analysis-making-statements-comparable) — the general discipline of time-series and cross-sectional comparison that every threshold here depends on.
- [Revenue recognition games: how tomorrow's sales become today's profit](/blog/trading/forensic-accounting/revenue-recognition-games-channel-stuffing-and-bill-and-hold) — channel stuffing and bill-and-hold in full, including the Sunbeam case.
- [Reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) — where cash from operations comes from, and the four ways it can still be flattered.
- [Quality of earnings: accruals, one-offs, and red flags](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) — the equity-research view of the same accrual gap that cash realization measures.
