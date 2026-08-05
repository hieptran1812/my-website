---
title: "Cash Flow Statement Manipulation: Classification Shifting and the Tricks That Flatter Operating Cash Flow"
date: "2026-08-05"
publishDate: "2026-08-05"
description: "Operating cash flow is supposed to be the number you cannot fake. This is a line-by-line guide to how it gets moved anyway — capitalization, factoring, payables stretch, prepays, acquired working capital, quarter-end timing — and how to rebuild it on a consistent basis."
tags: ["cash-flow-statement", "classification-shifting", "operating-cash-flow", "free-cash-flow", "forensic-accounting", "earnings-quality", "receivable-factoring", "reverse-factoring", "capitalization", "ias-7", "asc-230", "financial-statement-analysis"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 61
---

> [!important]
> **TL;DR** — Cash flow from operations is not a fact. The *total* change in cash is a fact; the split of that total into operating, investing and financing is a set of judgements, and every one of those judgements can be made to point the same way.
>
> - The cash flow statement obeys one identity: CFO + CFI + CFF = the change in cash. Nothing can change the right-hand side. Everything on the left is negotiable at the margin.
> - Seven mechanisms do almost all of the work: capitalizing an operating cost, selling receivables, stretching payables, borrowing through a prepay, buying working capital instead of building it, timing the quarter-end, and reclassifying discontinued or restructuring cash.
> - Free cash flow catches some of these and is completely blind to others. Capitalization cannot fool CFO minus capex; factoring, payables stretch and acquired working capital fool it completely.
> - US GAAP fixes where interest, dividends and tax go. IFRS makes four of those five lines a *policy choice*, so two compliant companies with identical cash can report operating cash flow \$220M apart.
> - The one habit that beats all of this: rebuild CFO yourself on a consistent basis, then re-rank the peer group. In the worked example that runs through this post, a reported CFO of \$1,700M rebuilds to \$779M — a 42% reported improvement that was really a 35% deterioration.
> - **The fact to remember:** none of this needs fake cash. When the SEC settled with Dynegy over Project Alpha in September 2002, the remedy was moving \$300 million — 37% of Dynegy's reported 2001 operating cash flow — from operating to financing. Every dollar was real, was in the bank, and was in the wrong section.

Ask an investor which financial statement they trust and most will say the cash flow statement. The reasoning is intuitive and almost right: revenue is an estimate, profit is an opinion, but cash is cash — either it arrived in the bank account or it did not.

The bank account part is true. What people forget is that the cash flow statement does not report one number. It reports *three*, and the sum of the three is the only part that the bank confirms. How the total gets divided between the three buckets is decided by the same people, using the same latitude, with the same incentives that produced the income statement you were suspicious of in the first place.

![Total change in cash is fixed; the split into operating, investing and financing is a judgement call](/imgs/blogs/cash-flow-statement-manipulation-classification-shifting-1.webp)

The diagram above is the mental model for this entire post. The wide bar at the top — total change in cash — cannot be moved. It is arithmetic, reconciled to a bank statement, confirmed by a third party. The three boxes underneath add up to that bar, and *which* box a given dollar lands in is where the discretion lives. Move \$500M of outflow from the operating box to the investing box and the top bar does not budge, the auditor's cash confirmation still ties, and operating cash flow rises 42%.

That movement is called **classification shifting**: reporting a cash flow in a different section of the statement than its economic substance warrants. It is the quietest form of financial statement manipulation, because unlike fake revenue there is often no fake transaction anywhere. The cash really moved. It really came from where the company says it came from, in some defensible reading. Only the label is wrong.

This post is the companion to [reading the cash flow statement line by line](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income). That post teaches you to build the statement. This one is about its seams: the seven places where the boundaries between the three sections are soft, what each trick is worth in dollars, and the rebuild procedure that undoes all of them at once.

A note on the numbers before we start. Every dollar figure attached to a company called **Meridian Telecom**, **Aurora Industrial**, **Calder Energy** or **Cascade Networks** is illustrative — those companies do not exist, and the arithmetic is constructed to be checkable in your head. Every dollar figure attached to a real, named company is sourced in the final section, dated, and framed as what a regulator or examiner actually alleged or found.

## Foundations: how the three buckets are built, and where the judgement lives

Start from zero. If you already know what the indirect method is, skim to "The five lines the standards argue about" — that is where the new material begins.

### Why there are three buckets at all

A company's bank balance goes up and down for reasons that mean very different things. Selling a million widgets and getting paid is not the same event as selling the factory, and neither is the same as borrowing from a bank — but all three raise cash by the same amount. If you only reported the change in the bank balance, those three would be indistinguishable.

So the statement splits cash movements by *why* the cash moved:

- **Operating activities (CFO)** — cash generated or consumed by running the business: collecting from customers, paying suppliers and staff, paying tax and (under US rules) interest. This is the recurring engine. It is the number people mean when they say "the business generates cash."
- **Investing activities (CFI)** — cash spent on or released by long-lived assets: buying equipment, building a network, acquiring another company, selling a division.
- **Financing activities (CFF)** — cash from and to the people who funded the company: issuing shares, borrowing, repaying debt, paying dividends, buying back stock.

The intuition to hold onto: **CFO is supposed to be sustainable, CFI is supposed to be discretionary, and CFF is supposed to be someone else's money.** Every trick in this post exploits that intuition by getting a non-sustainable, discretionary, or borrowed dollar counted as though it came from the engine.

### The one identity that cannot be broken

$$
\text{CFO} + \text{CFI} + \text{CFF} = \Delta\text{Cash}
$$

CFO, CFI and CFF are the three sections; ΔCash is the change in cash and cash equivalents over the period, which appears on the balance sheet and is confirmed with the bank.

This identity is the reason classification shifting is *conservative* in a very specific sense: it never creates cash. Every dollar added to CFO must be subtracted from CFI or CFF. That is a real constraint, and it is also the analyst's single best weapon, because it means every trick leaves a footprint somewhere else on the statement. Nothing disappears. It only moves.

### The indirect method hides the choice

Almost every listed company presents CFO using the **indirect method**: start at net income, add back non-cash charges like depreciation, then adjust for changes in working capital (receivables, inventory, payables). The alternative, the **direct method**, simply lists cash collected from customers and cash paid to suppliers, and almost nobody uses it.

This matters more than it sounds. Under the indirect method, a classification decision shows up as an unexplained line in a reconciliation, not as a missing cash receipt. If a company stops paying its suppliers for 25 extra days, the direct method would show a smaller "cash paid to suppliers" number that you could compare to cost of goods sold in one glance. The indirect method shows "increase in accounts payable: 164" buried in a working-capital block, which reads like housekeeping.

The indirect method does not cause the manipulation. It just means the evidence arrives pre-camouflaged.

### The five lines the standards argue about

Here is the part most people do not know: the accounting standards themselves do not agree on where several very large, very recurring cash flows belong.

![US GAAP mandates a single classification for interest, dividends and tax; IFRS offers a policy choice on four of the five](/imgs/blogs/cash-flow-statement-manipulation-classification-shifting-2.webp)

Under **US GAAP** (ASC 230, the standard governing the statement of cash flows), the answers are fixed. Interest paid is operating. Interest received is operating. Dividends received are operating. Dividends paid are financing. Income taxes paid are operating. You do not get to choose, which is why US filers cannot play this particular game — they have to play the others.

Under **IFRS** (IAS 7, the international equivalent), four of those five lines are an accounting *policy election*. Interest paid may sit in operating or financing. Interest received may sit in operating or investing. Dividends received may sit in operating or investing. Dividends paid may sit in operating or financing. Only tax is anchored to operating, and even that is qualified: taxes go in operating unless they can be specifically identified with a financing or investing activity.

The policy must be applied consistently and disclosed. It is not fraud. It is a choice — and a heavily levered company that elects to put interest paid in financing reports a structurally higher operating cash flow than an identical company that elects operating, year after year, legally.

That choice has an expiry date, but not a near one. **IFRS 18**, issued by the IASB in April 2024 and effective for annual reporting periods beginning on or after **1 January 2027**, amends IAS 7 to introduce new requirements for the classification of interest and dividend cash flows, narrowing the elections described above. Until then — and for every historical comparison you run afterwards — the policy choice is live and the comparability problem is real.

#### Worked example: the same company, two operating cash flows, both compliant

Aurora Industrial (illustrative) is an IFRS reporter. In one year:

- Cash generated from operations before interest and dividends: **\$760M**
- Interest paid on its debt: **\$180M** (an outflow)
- Dividends received from a minority stake in an associate: **\$40M** (an inflow)

Three presentations, all defensible:

| Presentation | Interest paid | Dividends received | Reported CFO |
| --- | --- | --- | --- |
| US GAAP (ASC 230) — no choice | Operating | Operating | 760 − 180 + 40 = **\$620M** |
| IFRS, Policy A | Operating | Investing | 760 − 180 = **\$580M** |
| IFRS, Policy B | Financing | Operating | 760 + 40 = **\$800M** |

Policy B reports operating cash flow **\$220M higher than Policy A** — 38% higher — on identical cash, in the same year, under the same standard. Neither company has done anything wrong. If you build a screen that ranks companies by CFO, or by CFO divided by revenue, or by CFO divided by debt, and you do not normalize for this, you are ranking accounting policies.

*The intuition: before you compare two companies' operating cash flow, check whether they even agree on what belongs in it.*

The practical fix is mechanical. Pull interest paid, interest received, dividends received and dividends paid out of wherever each company put them — IAS 7 requires them to be disclosed separately, so you can always find them — and put them all in the same place for every company in your comparison. Most analysts standardize by moving interest paid into operating, because that is the US GAAP treatment and because interest is a genuinely recurring cost of running a levered business.

With the foundations in place, we can walk the seven mechanisms. They are ordered roughly by how much money they move.

## Trick 1: Capitalize an operating cost, and the outflow moves to investing

A company pays \$1,000M in cash for something. If it is an **expense**, it reduces profit this year and the cash payment is an operating outflow. If it is a **capital asset**, it does not touch profit this year at all — it goes on the balance sheet and is depreciated over its useful life — and the cash payment is an *investing* outflow, part of capital expenditure.

Same cash. Same day. Two completely different statements.

The rule that decides which is which is genuinely a judgement: does the spending create a resource that will produce benefits in future periods? Buying a truck, yes. Paying this month's electricity bill, no. In between sits an enormous grey zone — software development, network installation costs, customer acquisition, maintenance that arguably extends an asset's life — and the grey zone is where the money is.

#### Worked example: Meridian Telecom capitalizes \$500M of line costs

Meridian Telecom (illustrative) leases capacity from other carriers to carry its customers' traffic. It pays **\$1,000M** in cash for this during the year. Historically it has expensed all of it, because it is a recurring cost of carrying traffic — you pay it every year whether or not you build anything.

This year, management decides that \$500M of those payments "establish network capability with enduring benefit" and capitalizes them as network assets, depreciated over 10 years.

Watch all three statements.

**Income statement.** Operating expense falls by \$500M. First-year depreciation on the new asset, using a half-year convention, is \$500M ÷ 10 × 0.5 = **\$25M**. So pre-tax profit rises by 500 − 25 = **\$475M**.

**Cash flow statement.** The \$500M was paid in cash either way — the bank does not care about the label. But now it is capital expenditure:

| Line | As incurred | With \$500M capitalized |
| --- | --- | --- |
| Cash flow from operations | \$1,200M | **\$1,700M** |
| Capital expenditure (in CFI) | (\$800M) | (\$1,300M) |
| Cash flow from financing | (\$280M) | (\$280M) |
| **Change in cash** | **\$120M** | **\$120M** |

**Balance sheet.** Net PP&E rises by 500 − 25 = \$475M. Retained earnings rise by the after-tax profit boost.

![Capitalizing \$500M of operating cost lifts CFO 42% while free cash flow is unchanged](/imgs/blogs/cash-flow-statement-manipulation-classification-shifting-3.webp)

Reported operating cash flow rose from \$1,200M to \$1,700M — **up 42%** — and the change in cash is identical to the dollar. If your model keys off CFO growth, Meridian just had a spectacular year. If it keys off the bank balance, nothing happened.

*The intuition: capitalization does not create cash, it relabels an outflow from the section people trust to the section people forgive.*

### Why free cash flow is immune to this one — and what that tells you

Look at the bottom of that table again and compute free cash flow, defined the standard way as CFO minus capital expenditure:

- As incurred: 1,200 − 800 = **\$400M**
- With capitalization: 1,700 − 1,300 = **\$400M**

Identical. This is not luck — it is structural. Capitalization moves a dollar from the CFO term to the capex term, and free cash flow subtracts capex from CFO, so the dollar cancels. **Free cash flow is mathematically immune to capitalization.**

That gives you a screen with almost no false positives. When operating cash flow accelerates and free cash flow does not, capitalization is the first hypothesis, and it is usually right. The mirror-image warning matters just as much: this is exactly why a company running this trick will steer you toward "operating cash flow" and "EBITDA" in its investor deck and away from free cash flow.

Two further tells, both visible in the filings:

- **Capex versus depreciation.** Capitalizing an operating cost inflates capex immediately and depreciation only gradually. A capex-to-depreciation ratio that jumps without a stated build programme is a flag. Meridian's capex went to \$1,300M against depreciation that rose only \$25M in year one.
- **The deferred tax liability.** Tax authorities apply their own rules on what may be capitalized, and they are usually stricter and independent of the book decision. If a company capitalizes for book purposes but continues to deduct the cost for tax purposes, the difference piles up as a growing **deferred tax liability** — a balance sheet line saying "we have reported more profit to shareholders than to the tax authority." At a 25% tax rate, Meridian's \$475M book gain generates roughly \$119M of new deferred tax liability while cash taxes paid do not move at all. A deferred tax liability growing much faster than the business is one of the most under-used signals in forensic accounting.

The full mechanics of what may and may not be capitalized, including the software and R&D grey zones, are in the dedicated post on [capitalizing costs to inflate profit](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move). What matters here is the cash flow consequence, which that post's most famous case demonstrates at a scale nobody has matched since.

### The WorldCom version, and what it did to operating cash flow

WorldCom's **line costs** were payments to other carriers for access to network capacity — the single largest operating expense of a long-distance telecommunications company, and the exact item the Meridian example was modelled on.

The enforcement record is specific, and it pays to keep two different numbers apart. The SEC's *Litigation Release No. 17588* of **27 June 2002** announced a case it headlined as a "\$3.8 Billion Fraud", alleging that WorldCom "fraudulently overstated its income before income taxes and minority interests by approximately \$3.055 billion in 2001 and \$797 million during the first quarter of 2002" by capitalizing rather than expensing "approximately \$3.8 billion of its costs." Those \$3.055 billion and \$797 million figures are the *income* overstatement, which also includes improper releases of line-cost reserves — they are not the capitalization total, and they are frequently quoted as though they were.

The capitalization itself is set out quarter by quarter in the SEC's complaint, filed **26 June 2002**. Line costs moved into capital asset accounts of approximately **\$771 million** in the first quarter of 2001, **\$560 million** in the second, **\$743 million** in the third and **\$941 million** in the fourth — approximately **\$3.015 billion** for 2001 — followed by approximately **\$818 million** in the first quarter of 2002. That is roughly **\$3.833 billion** of operating cost relabelled as capital expenditure in five quarters.

The income statement effect is what the case is famous for: the same complaint puts WorldCom's reported 2001 line costs at **\$14.739 billion** against **\$17.754 billion** of actual line costs, turning what it states as a **\$622 million** actual loss before taxes and minority interests into reported income of **\$2.393 billion**.

The cash flow effect is the part that matters here, and it follows mechanically from the mechanism rather than from a separate finding. Those payments left WorldCom's bank account either way — the carriers were paid. Once the payments were capitalized, they were no longer operating outflows; they were capital expenditure. So for 2001, an amount on the order of \$3.015 billion of cash outflow sat in the investing section instead of the operating section, and reported operating cash flow was correspondingly higher, with reported capital expenditure higher by the same amount.

This is the invariance from the worked example, at roughly six times the scale. **The single most consequential accounting fraud of its era inflated operating cash flow and left free cash flow untouched.** An investor who had been told that cash flow cannot be faked, and who watched operating cash flow, saw nothing. An investor who watched CFO minus capital expenditure, or simply plotted capital expenditure as a share of revenue over five years, saw a company whose capital intensity had changed without an announced build programme.

Those are enforcement allegations and reported findings from the SEC's filings, not independent verification of every underlying entry. The narrative in full, including who made the entries and how the internal audit function found them, is in the [dedicated WorldCom post](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move).

## Trick 2: Sell the receivable and pull next quarter's cash into this one

When a company makes a sale on credit, it books revenue immediately and records an **account receivable** — a claim on the customer, typically settled in 30 to 90 days. Until the customer pays, the sale has produced profit but no cash. The gap is the single largest driver of the difference between net income and operating cash flow.

**Factoring** collapses that gap. The company sells the receivable to a bank or a specialist finance company today, at a discount, and the bank collects from the customer later. The company gets cash now; the bank earns the discount.

There is nothing improper about factoring. It is a legitimate, widely used financing tool, and for a company with an expensive cost of capital it can be cheap money. The problem is entirely one of presentation: **when a receivable is sold, the cash lands in operating activities, because receivables are a working capital line.** Borrowing against the same receivable would land in financing. The economics are nearly identical; the statement is not.

#### Worked example: factoring \$200M of receivables

Meridian has annual revenue of \$3,650M — exactly \$10M per day — and accounts receivable of \$600M.

**Days sales outstanding (DSO)** measures how long customers take to pay: receivables divided by revenue per day. Meridian's DSO is 600 ÷ 10 = **60 days**.

In the third quarter, Meridian sells **\$200M** of receivables to a bank at a 1.5% discount. It receives 200 × 0.985 = **\$197M** in cash today, and records a \$3M loss on sale as an operating expense.

What happens to the statement:

| Line | Effect |
| --- | --- |
| Accounts receivable | falls \$200M |
| Cash | rises \$197M |
| Operating cash flow | rises **\$197M** (the working capital release, net of the \$3M cost) |
| Investing cash flow | unchanged |
| Financing cash flow | unchanged |
| Reported DSO | 400 ÷ 10 = **40 days**, down from 60 |

Two things just happened, and the second is the important one. First, CFO rose \$197M. Second, DSO — the metric an analyst uses to *detect* receivables problems — improved by 20 days. The trick does not just inflate the number; it disables the alarm that was supposed to catch it.

![A factoring programme is a one-time pull-forward, not a run-rate improvement, and it reverses when it stops](/imgs/blogs/cash-flow-statement-manipulation-classification-shifting-4.webp)

The chart shows what really matters about factoring: **it is a level shift, not a growth rate.** In the quarter the programme starts, cash collected jumps \$197M. In every quarter after that, if the programme is held at the same size, the benefit is exactly zero — you are simply selling this quarter's receivables instead of last quarter's. And in the quarter the programme is wound down, the \$197M comes straight back out.

So a company running a factoring programme has bought one good quarter and taken on an obligation to keep the programme running forever. That is a perfectly reasonable financing decision and a terrible thing to model as recurring cash generation.

*The intuition: factoring converts a future collection into a present one exactly once, and the reversal is as large as the boost.*

**How to detect it.** Three checks, in order of reliability:

1. **Read the receivables footnote.** Companies must disclose the existence and size of receivables sale programmes. Look for "transfers of financial assets", "sales of receivables", "securitization", "trade receivables purchase agreement". Note the balance sold and outstanding at each year end, and the *change* in that balance — the change is the CFO effect.
2. **Recompute DSO on a gross basis.** Add the receivables sold and still outstanding back into the receivables balance, then recompute. In the worked example that restores DSO to 60 days and the "improvement" vanishes.
3. **Compare CFO to the change in the programme size.** If the year-on-year growth in receivables sold is close to the year-on-year growth in CFO, you have found your explanation.

### The securitization refinement, and the rule that closed part of it

A more elaborate version routes the receivables through a **securitization** structure: the company sells receivables to a special-purpose vehicle, which issues notes to investors. The company often keeps a residual interest — a **deferred purchase price**, meaning it gets paid the remainder once the underlying receivables are collected.

For several years, companies argued that collections on that residual interest were also operating inflows, because they originated in trade receivables. That produced the best of both worlds: cash up front in operating, and the tail in operating too.

The FASB closed this. Under **ASU 2016-15**, cash received on a *beneficial interest* obtained in a securitization of the seller's own trade receivables is classified as an **investing** inflow, not operating. Companies that had been reporting the whole thing in operating had to move a slice of their reported CFO into investing, and several reported materially lower operating cash flow as a result — with no change whatsoever in their business.

That episode is worth remembering for its own sake. **A standard-setter changed a label, and reported operating cash flow at multiple large companies fell.** If the number can move that far without anything real happening, it is not the hard fact people treat it as.

## Trick 3: Stretch the payable, and let the supplier fund you

The mirror image of collecting early is paying late. **Accounts payable** is what a company owes its suppliers; **days payable outstanding (DPO)** is how long it takes to pay them, computed as payables divided by cost of goods sold per day.

Extending DPO releases cash. It is not subtle and it is not illegal — it is a negotiation, and large buyers win it routinely. But the cash it releases behaves exactly like the factoring cash: a one-time level shift dressed as performance.

#### Worked example: 25 days of DPO on \$2,400M of cost of goods sold

Meridian's cost of goods sold is **\$2,400M** per year, or 2,400 ÷ 365 = **\$6.58M per day**.

At a DPO of 45 days, accounts payable is 45 × 6.58 = **\$296M**. Management pushes payment terms to 70 days. At a DPO of 70 days, payables becomes 70 × 6.58 = **\$460M**.

The increase — 460 − 296 = **\$164M** — appears in operating cash flow as "increase in accounts payable." It is a cash inflow to Meridian and a cash outflow from its suppliers, who are now financing 25 extra days of Meridian's cost base at no charge.

![A 25-day payables stretch is worth \$164M once, then nothing, then minus \$164M when it reverses](/imgs/blogs/cash-flow-statement-manipulation-classification-shifting-5.webp)

The shape is the same as factoring, for the same reason. In the quarter the terms change, CFO gets \$164M. In the following quarters, holding DPO at 70 days, the benefit is zero — you are paying 70-day-old bills instead of 45-day-old bills, at the same rate. If suppliers push back and terms revert, CFO takes a \$164M hit.

*The intuition: a working capital ratio can only improve once; after that it has to keep improving to keep contributing, and no ratio improves forever.*

**How to detect it.** Compute DPO every year and plot it. A 20-day move in a single year is not an operational improvement — it is a decision, and it should be reflected in a discussion somewhere in the management commentary. Also compare DPO to the peer group: a company paying its suppliers 25 days slower than direct competitors is either the dominant buyer in its industry or is short of cash, and the footnotes will usually tell you which.

The second-order check is more informative. **Compare the payables balance at the reporting date to the average payables balance through the period.** A genuine terms renegotiation shows up in both. A reporting-date manoeuvre shows up only in the first. That distinction is Trick 6, and we will come back to it.

### Reverse factoring: a trade payable that is really a bank loan

The refined version is **reverse factoring**, also called supply chain finance. The buyer arranges for a bank to pay its suppliers early — say on day 15 — and the buyer then repays the bank on day 70 or later. Suppliers get their money faster, the buyer gets much longer terms, and the bank earns the spread.

The presentation question is the whole ballgame: is the buyer's obligation still a **trade payable** (operating) or has it become a **borrowing** (financing)?

If it stays classified as a trade payable, the entire \$164M of the previous example flows through operating cash flow, leverage looks unchanged, and the bank facility is invisible in the debt note. If it is reclassified as debt, CFO falls \$164M, financing cash flow rises \$164M, and reported borrowings rise \$164M.

Standard-setters on both sides of the Atlantic responded with **disclosure** rather than a bright-line classification rule — the FASB through ASU 2022-04 on supplier finance programme disclosures, the IASB through amendments to IAS 7 and IFRS 7 on supplier finance arrangements. Companies now have to tell you the programme exists, how big it is, and where the obligations sit on the balance sheet. They still, largely, get to decide where.

That disclosure is one of the highest-yield paragraphs in a modern annual report, and the leverage-hiding side of the story — including how a supply chain finance programme can conceal a genuine liquidity crisis until the week it ends — deserves its own treatment, which the next post in this series gives it. Here, note only the classification consequence: **a facility that a rating agency would call debt can sit inside your operating cash flow.**

## Trick 4: Borrow, but book it as operating — prepays and round-trip cash

Tricks 1 to 3 all bend a real transaction's label. Trick 4 is a different animal: it constructs a transaction whose only purpose is to make borrowed money arrive in the operating section.

### The prepay structure

The building block is a **prepaid forward** — a contract where a buyer pays today for a commodity to be delivered over some future period. Prepayment is a normal commercial arrangement; suppliers routinely take money up front for future delivery, and the cash they receive is a customer prepayment, which is working capital, which is operating.

Now build the loop. A bank funds a special-purpose entity. The entity signs a prepaid contract with the company and hands over cash today. The company delivers commodity over the following years — but the deliveries are priced above market by roughly the amount needed to repay the cash plus a spread. Follow the money end to end and the company received cash today and will repay it with interest over several years. That is a loan.

![A prepaid commodity sale routes borrowed cash into the operating section and the repayments back out of it](/imgs/blogs/cash-flow-statement-manipulation-classification-shifting-6.webp)

The classification consequence is the point of the whole structure:

| | As presented | Economic substance |
| --- | --- | --- |
| Cash received today | Operating inflow (customer prepayment) | Financing inflow (loan proceeds) |
| Repayments over the term | Operating outflow (cost of deliveries) | Financing outflow (principal) + interest expense |
| Reported debt | Unchanged | Higher by the principal |
| Interest coverage | Unchanged | Worse |

#### Worked example: a \$300M prepaid commodity sale

Calder Energy (illustrative) needs cash and does not want to report more debt. On 1 January a bank funds a special-purpose entity with **\$300M**. The entity signs a five-year gas supply contract with Calder and pays the **\$300M** up front. Over the following 51 months Calder delivers gas priced above the market rate, and the cumulative overpricing returns the \$300M plus a spread of roughly **\$45M** to the entity, which passes it to the bank.

What Calder reports in year one:

- Operating cash flow: **+\$300M**
- Debt on the balance sheet: **unchanged**
- Interest expense: **\$0** — the \$45M spread is buried inside cost of goods sold across 51 months
- Interest coverage ratio: **unchanged**, because the interest is not called interest

What actually happened: Calder borrowed \$300M at roughly 7% and will repay it over four and a quarter years. (A \$45M spread on \$300M amortizing over 51 months works out to about 7% a year on the average balance outstanding.)

And now the second-order effect, which is what makes the structure dangerous rather than merely misleading. In year two, if Calder needs another \$300M, it must do another prepay — and the deliveries from the first one are now running against it. Each new deal has to be larger than the last to produce the same net operating cash flow, because the repayments on the earlier deals are flowing out through the same line. The structure has a treadmill built into it, and a company that stops running falls off in the direction of a very sudden liquidity crisis.

*The intuition: if a cash inflow has to be repaid on a fixed schedule with a return to the provider, it is financing, whatever the contract calls it.*

**How to detect it.** This one is genuinely hard from the outside, which is why it has been used by very large companies for very long periods. The signals available to an outsider:

- **CFO rising without a corresponding rise in profit or a working capital explanation.** Rebuild the working capital bridge yourself. If CFO grew and none of receivables, inventory, payables or deferred revenue explain it, something structural is in there.
- **A large, growing "deferred revenue" or "customer prepayment" or "price risk management liability" balance** in a business that does not normally take money up front.
- **Contract terms with unusual tenors** — a "commodity sale" with a five-year delivery schedule and a fixed price is a financing instrument wearing a commercial hat.
- **Footnote language about "structured transactions", "prepaid forward sales", or "monetizations."** The word *monetization* in a footnote is almost always describing the conversion of a future cash flow into a present one, which is what a loan is.

### Round-trip cash: paying yourself and calling it operations

The cousin of the prepay is the **round trip**: two companies simultaneously buy roughly equal amounts from each other, at roughly equal prices, with no net economic effect. Each books revenue on the sale, and — the part that matters here — each books an operating cash inflow when it collects and an operating cash outflow when it pays.

If the two legs settle at the same moment, the cash flow statement nets to zero and only revenue is inflated. If the legs settle at *different* times — you collect in the first quarter and pay in the third — then the first quarter shows a genuine operating cash inflow that the third quarter gives back. Round-tripping across a reporting boundary is classification shifting in the time dimension.

The revenue side of this is covered in detail in [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue). The cash flow tell is a specific one: **look for a large customer who is also a large supplier.** Related-party and major-customer disclosures will sometimes give you both sides of the same name.

### Inflows travelling the other way

Everything so far has moved *outflows* out of operating. The mirror image is moving *inflows* in — taking cash that arrived from selling an asset or from an insurer and classifying it as though the business had generated it.

The three places this happens most:

- **Proceeds from selling assets.** Selling equipment is an investing inflow, and the gain is stripped out of CFO in the indirect-method reconciliation so it is not double-counted. But a business whose *model* involves buying assets, renting them out, and selling them — vehicle fleets, equipment rental, aircraft leasing — has a genuine argument that some of that flow is operating. Where the boundary sits changes CFO materially, and the argument is available to every company in the sector, which means comparing them requires reading each one's stated policy.
- **Insurance recoveries.** A factory burns down and the insurer pays \$150M. Under ASU 2016-15 the classification follows the nature of the loss: business-interruption cover replaces lost operating profit and is operating; cover for the destroyed building replaces an asset and is investing. If the settlement is a single lump sum, someone has to split it — and the split is an estimate made by a company that would prefer the operating half to be larger.
- **Sale and leaseback.** This one usually runs *against* reported operating cash flow, which is why it is worth knowing.

#### Worked example: what a sale-leaseback does to CFO

A company sells its headquarters for **\$200M** and immediately leases it back for 20 years at **\$18M** a year.

- **Year 0:** investing cash flow rises **\$200M**. Operating cash flow is unchanged.
- **Every year after:** the rent is an operating outflow of **\$18M**, so operating cash flow is **\$18M lower** than it was before the deal, forever.
- **Total cash over 20 years:** 200 − (18 × 20) = **−\$160M**, before any tax effect.

Economically the company borrowed \$200M against the building and is repaying it through rent. The statement records the borrowing as an investing inflow and the repayments as operating outflows — the exact opposite of the prepay structure, and it damages the metric everyone watches.

*The intuition: classification error is not always self-serving. A company that raises cash the honest, visible way can end up with worse-looking operating cash flow than one that raises it through a structure — which is precisely the incentive gradient that produces structures.*

### The two cases this structure is named after

The Calder Energy example above is not invented from nothing — it is modelled on a transaction the SEC brought an action over, and the illustrative \$300M is the case's own figure.

**Dynegy, Project Alpha, 2001.** According to the SEC's *Administrative Proceeding 33-8134* of **24 September 2002**, Dynegy implemented a structured natural-gas transaction, known internally as Project Alpha, that produced **\$300 million** of what the company reported as cash flow from operations in 2001, together with a **\$79 million** tax benefit. The SEC found that "the \$300 million transaction was a loan to Dynegy, not the result of Dynegy's operations," and that the restatement would reclassify the Alpha cash flow "as deriving from financing activities, rather than operations — reducing, as a consequence, Dynegy's cash flow from operations in 2001 by 37%." Dynegy paid a **\$3 million** civil penalty.

The SEC's parallel order against Citigroup, which helped structure the deal, puts the same numbers in context: Alpha accounted for \$300 million of Dynegy's reported 2001 operating cash flow **out of a gross reported figure of \$811 million**. More than a third of the year's operating cash flow was a loan.

Read that reclassification carefully, because it is the cleanest demonstration in the enforcement record of what this post is about. **Not one dollar of cash was fictitious.** The money genuinely arrived, from a real counterparty, under a real contract, and it was still in the bank when the auditors confirmed it. The correction moved \$300 million from one section of the statement to another, and it was serious enough to end in an SEC order.

**Enron, 1997–2001.** The same structure at a scale that changes what it means. The SEC's two 2003 actions against Enron's banks document it precisely. Per *Litigation Release No. 18252* of **28 July 2003**, "between December 1997 and September 2001, J.P. Morgan Chase effectively loaned Enron a total of approximately **\$2.6 billion** in the form of seven such transactions." The SEC's order against Citigroup the same day finds that "Citigroup and Enron executed **ten prepay transactions between December 1998 and June 2001**," through which Citigroup "made available to Enron a total of **\$3.8 billion** over a two and one-half year period." In both cases the proceeds were reported as cash flow from operating activities and the repayment obligation as a "price risk management liability" rather than debt. Chase paid **\$135 million** and Citigroup **\$120 million** to settle.

The Citigroup order also contains the single most arresting number in this entire post. For Enron's fiscal 1999, prepays plus a related structure called Project Nahanni accounted for approximately \$2 billion of reported net operating cash flow — and the SEC states that without them, "Enron would have reported that it used \$800 million in net cash in operating activities instead of reporting that it generated \$1.2 billion." **The structures did not flatter the number. They reversed its sign.**

Two lessons compound here. First, the treadmill is real: more than \$6 billion of prepays across two banks in under four years needed a continuous supply of new transactions to keep the operating section fed while the earlier ones repaid through the same line. Second, and more usable, **this trick requires willing counterparties.** A company cannot construct a prepay alone; there has to be a bank and usually a conduit entity. That means the disclosure trail runs through more than one filer, and it means the structures tend to be concentrated in whichever banks are prepared to write them.

Enron's wider mechanics — mark-to-market accounting, the special-purpose entities, the role of the auditor — are covered in the [Enron 2001 accounting fraud post](/blog/trading/finance/enron-2001-accounting-fraud). The prepays are the piece that belongs to this article, because they are the largest documented example of borrowed money reported as operations.

## Trick 5: Buy the working capital instead of building it

This one is elegant enough that many of the people doing it may not regard it as a trick at all.

When a company acquires another business for cash, the entire purchase price is an **investing** outflow — the line usually reads "acquisitions, net of cash acquired." But the acquired company comes with a balance sheet, and that balance sheet contains working capital: receivables the acquirer will collect, inventory it will sell, payables it will pay.

Those receivables were paid for as part of the purchase price, in the investing section. When they are collected, the cash arrives in the **operating** section.

#### Worked example: harvesting \$60M of acquired working capital

On 1 July of year two, Meridian acquires Cascade Networks (illustrative) for **\$500M** in cash. The business combination footnote discloses the acquired balance sheet:

| Acquired item | Amount |
| --- | --- |
| Accounts receivable | \$120M |
| Inventory | \$40M |
| Accounts payable | (\$60M) |
| **Net working capital acquired** | **\$100M** |

Over the following two quarters Meridian collects the acquired receivables, works down the acquired inventory, and does not replace either at the same level — it integrates Cascade onto its own systems and runs the combined working capital leaner. Net working capital attributable to Cascade settles at **\$40M**.

The **\$60M** difference appears in Meridian's operating cash flow as a favourable working capital movement. The \$500M that bought it appears in investing.

And now look at free cash flow. The standard definition is CFO minus capital expenditure. Acquisitions are not capital expenditure — they sit on a different line of the investing section. So reported free cash flow picks up the **+\$60M** and none of the **−\$500M**.

*The intuition: an acquisition can manufacture operating cash flow, and the standard free cash flow definition is built so that it will not see the price you paid.*

This is why serial acquirers deserve a specific kind of scrutiny. A company making four deals a year can keep harvesting acquired working capital indefinitely, reporting rising operating cash flow and rising free cash flow, while the cash that funded all of it went out through a section nobody puts in the headline metric.

**How to detect it.** The business combination footnote gives you the acquired balance sheet — it is required disclosure. Compare the acquired net working capital to the working capital movement in the cash flow statement for the periods after the deal. Then compute a version of free cash flow that subtracts cash paid for acquisitions:

$$
\text{FCF}_{\text{adjusted}} = \text{CFO} - \text{Capex} - \text{Cash paid for acquisitions}
$$

For a company that grows organically this changes nothing. For a serial acquirer it frequently changes the sign.

## Trick 6: The quarter-end clock

Every trick so far moves a dollar between sections. This one moves it between *dates*, and it is the cheapest of all of them to execute because it requires no counterparty, no structure, and no accounting policy change. It requires a calendar.

### Cheques written but not mailed

The balance sheet is a photograph taken on one specific day. Operating cash flow for a quarter is derived from the movement between two photographs. So anything that suppresses cash outflows in the last few days of a quarter, or accelerates inflows into them, raises that quarter's operating cash flow — and gives it back in the first days of the next one.

The oldest version is the simplest: print the supplier cheques on schedule, then hold them in the mailroom until the new quarter starts. Payables stay high, cash stays high, CFO is flattered, and no accounting entry anywhere is false — the cheques genuinely had not been sent.

![The payables balance spikes for four days around the reporting date, then returns to its normal level](/imgs/blogs/cash-flow-statement-manipulation-classification-shifting-7.webp)

#### Worked example: the four-day \$162M

Meridian's accounts payable runs at a steady **\$298M** through the third quarter — 45 days of its \$6.58M-per-day cost of goods sold. In the last week, treasury prepares **\$162M** of supplier payments and holds them until day 92, two days after the quarter closes.

- Payables at the reporting date (day 90): **\$460M**
- Cash at the reporting date: **\$162M higher** than it would otherwise have been
- Q3 operating cash flow: **+\$162M**
- Q4 operating cash flow: **−\$162M**
- Full-year operating cash flow: **unchanged**

Now compute DPO two ways:

- **On the reporting-date balance:** 460 ÷ 6.58 = **70 days**
- **On the average daily balance through the quarter:** 298 ÷ 6.58 = **45 days**

*The intuition: a working capital metric computed from a single day's balance measures what the company wanted you to see on that day.*

That gap between the point-in-time DPO and the average DPO is the entire signature, and it is why quarter-end timing is best caught in the **quarterly series** rather than the annual statement. The annual number is clean by construction — the boost and the reversal are both inside the year. Plot four quarters of CFO and the sawtooth is obvious: an unusually strong quarter followed by an unusually weak one, repeating, with the full-year total unremarkable.

### Pulling collections forward, and what it costs

The other half of the clock is acceleration on the receivables side: offering customers a discount to pay before the quarter closes, or pushing the sales force to collect early.

This one has a measurable price, and the price is the tell. A 2% discount for paying 20 days early is an annualized cost of roughly 2% × (365 ÷ 20) = **36%**. No competent treasurer pays 36% annualized for 20 days of cash unless the 20 days matter for a reason unrelated to economics — like a covenant test, a bonus threshold, or a guidance number.

So when you see gross margin dip in the same quarter that DSO improves sharply, consider that you may be looking at one transaction reported in two places: the discount in the margin, the cash in the working capital line.

**How to detect quarter-end timing generally.** Compute operating cash flow by quarter, not by year, for five years. Look for:

- A Q4 that is consistently enormous relative to Q1 — the classic pattern when annual bonuses key off full-year cash flow.
- Negative serial correlation: strong quarters systematically followed by weak ones, which is the arithmetic signature of borrowing from the next period.
- Reporting-date working capital ratios that diverge from ratios computed on average balances, where the company discloses enough to compute both.

## Trick 7: The classifications nobody argues about — until they matter

The last group is a collection of smaller reclassifications that share one property: each is individually defensible, and a company under pressure tends to make all of them in the same direction at once. That directional consistency is itself the evidence.

### Discontinued operations

When a company decides to sell a business, that business may be presented as a **discontinued operation** — separated out from continuing operations so investors can see the ongoing business clearly. The intent is genuinely investor-friendly.

The cash flow consequence is that most companies then headline **"cash flow from continuing operations."** If the unit being disposed of was consuming cash, removing it lifts the headline number without a single dollar changing hands.

Consider a group with total operating cash flow of \$500M, including a division that burned **\$90M** of operating cash during the year. Classify that division as held for sale, and "operating cash flow from continuing operations" is **\$590M** — an 18% improvement announced in a period when the group's bank balance did exactly what it was always going to do. Then the division is sold, and the proceeds arrive in *investing*.

The check is mechanical: use total operating cash flow including discontinued operations for any comparison across time, and read the disclosure of discontinued operations' cash flows, which is required precisely so you can do this.

### Restructuring, legal settlements, and the cash that is "not really operating"

Cash paid for restructuring — severance, site closures, contract terminations — is an operating outflow under both GAAP and IFRS. So are legal settlements and regulatory fines arising from the conduct of the business. There is no serious argument that they belong anywhere else.

What happens instead is that they get excluded from the *company's own* definition of cash flow, which brings us to the escape hatch.

### The non-GAAP escape hatch

Free cash flow is not a defined term under either GAAP or IFRS. Companies define it themselves, and the definitions vary enormously. Common adjustments, in roughly descending order of aggressiveness:

- Add back cash restructuring payments ("non-recurring")
- Add back cash legal settlements and fines
- Add back transaction costs on acquisitions
- Subtract only **"maintenance capex"** — a self-assessed subset of actual capital expenditure — instead of total capex
- Exclude the working capital movement entirely, presenting an "underlying" figure

#### Worked example: Meridian's three free cash flows

Take Meridian's manipulated year two, and compute free cash flow three ways.

**1. As the company presents it — "adjusted free cash flow":**

| Component | Amount |
| --- | --- |
| Reported operating cash flow | \$1,700M |
| Add back: cash restructuring payments | \$85M |
| Add back: acquisition transaction costs | \$40M |
| Less: "maintenance capex" | (\$900M) |
| **Adjusted free cash flow** | **\$925M** |

**2. GAAP free cash flow — reported CFO minus total capex:**

1,700 − 1,300 = **\$400M**

**3. Rebuilt free cash flow — CFO on a consistent basis minus true capex:**

779 − 800 = **−\$21M**

The gap between the company's headline number and the rebuilt one is **\$946M**, and the company has not broken a rule to produce it. Every adjustment was disclosed. Every reconciliation to the GAAP measure was provided. The number at the top of the press release still differs from the number describing the business by nearly a billion dollars.

*The intuition: when a company invents a metric, read the reconciliation, not the metric.*

Where the \$779M comes from is the subject of the next section.

## The detection toolkit: rebuild CFO on a consistent basis

Every trick in this post is defeated by the same procedure. You do not need to identify which one is being used. You need to put the company's cash flows onto a basis that is consistent across time and across peers, and then look at what is left.

Before the procedure, the map. Seven mechanisms, what each one moves, and — the column most people get wrong — whether free cash flow notices:

| Trick | Moves cash from | Into | Recurring? | Does CFO − capex catch it? |
| --- | --- | --- | --- | --- |
| 1. Capitalize operating costs | Operating | Investing (capex) | Yes, while the spending continues | **Yes** — exactly cancels |
| 2. Sell receivables | Next period's operating | This period's operating | No — one-time level shift | No |
| 3. Stretch payables | Next period's operating | This period's operating | No — one-time level shift | No |
| 4. Prepay / round-trip | Financing | Operating | Only if each deal is bigger | No |
| 5. Acquire working capital | Investing (acquisitions) | Operating | Only while deals continue | No |
| 6. Quarter-end timing | Next quarter's operating | This quarter's operating | No — reverses in days | No, within a year |
| 7. Discontinued ops / non-GAAP | Nowhere — redefines the metric | The headline | Yes | Depends on the definition used |

The pattern in the last column is the reason free cash flow has an undeserved reputation as manipulation-proof. It is airtight against exactly one of the seven, and that one happens to be the most famous — which is how a metric acquires a reputation it cannot support.

![Five screens, and the rebuild that undoes all of them at once](/imgs/blogs/cash-flow-statement-manipulation-classification-shifting-8.webp)

### The five screens

**Screen 1 — CFO divided by net income, over five years.** A business that converts profit into cash should show this ratio above 1.0 and reasonably stable. Sustained readings below 1.0 mean profit is not turning into cash, which is the classic accrual-manipulation signature.

Two warnings about this screen, because it is the one most often used badly. First, **absolute thresholds are useless across industries** — a capital-intensive telecom with heavy depreciation runs structurally above 2.0 with nothing wrong, and a fast-growing distributor funding inventory runs below 1.0 with nothing wrong. The signal is a break in the company's *own* trend, and its position against direct peers. Second, this screen is weak against capitalization specifically, because capitalizing an operating cost raises the numerator *and* the denominator.

Meridian makes the point. In year one it earned pre-tax profit of \$800M, which at a 25% tax rate is net income of \$600M, on operating cash flow of \$1,200M — a ratio of **2.00**. In year two the capitalization added \$475M of pre-tax profit and the factoring discount cost \$3M, so pre-tax profit was 800 + 475 − 3 = \$1,272M and reported net income was **\$954M** against reported CFO of \$1,700M — a ratio of **1.78**. The screen registered a decline of 0.22, to a number that still reads as comfortably healthy, in a year when cash generation fell by more than a third. A ratio that responds to a trick by staying well above 1.0 is not a detector.

**Screen 2 — the company's free cash flow versus CFO minus total capex.** Any gap is a definition you have to go and read. This is the screen that catches capitalization, because CFO minus total capex is invariant to it.

**Screen 3 — the change in DSO and DPO, computed on a gross basis.** Add back receivables sold and outstanding before computing DSO. A 20-day move in either metric in a single year is a decision, not an operational trend, and it should be explained somewhere in the narrative reporting. If it is not, that is the finding.

**Screen 4 — the receivables-sold and supplier-finance footnotes.** You are looking for the balance and, more importantly, the year-on-year change in the balance. The change *is* the operating cash flow contribution. A programme that grew \$197M contributed \$197M.

**Screen 5 — cash paid for acquisitions versus working capital acquired.** From the business combination footnote. Then recompute free cash flow with acquisitions subtracted.

### One more, because it compounds: cash conversion computed two ways

The five screens above are the ones in the checklist figure. There is a sixth worth running because it catches two tricks at once. "Cash conversion" usually means operating cash flow divided by EBITDA, and it is the screen management teams quote because it flatters a capital-intensive business. Compute it, then compute the version that actually constrains the company: the **cash conversion cycle**, which is days sales outstanding plus days inventory outstanding minus days payable outstanding — the number of days between paying for something and being paid for it.

Meridian's cash conversion cycle looks like this once you strip the presentation out:

| Component | Year 1 | Year 2 reported | Year 2 gross basis |
| --- | --- | --- | --- |
| Days sales outstanding | 60 | 42 | 61 |
| Days inventory outstanding | 30 | 30 | 30 |
| Days payable outstanding | (45) | (70) | (45) |
| **Cash conversion cycle** | **45 days** | **2 days** | **46 days** |

Reported, the cycle collapsed from 45 days to 2 — a transformation that would headline any investor day. On a gross basis it went from 45 days to 46. Nothing happened, twice: once to the receivables through factoring, once to the payables through the terms stretch, and the two effects compound in the same metric.

The full construction of this cycle, including what each component tells you about the business behind it, is in [the cash conversion cycle post](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals). The forensic point here is narrow: **the cash conversion cycle is built from three ratios, and two of them can be bought.**

#### Worked example: rebuilding Meridian's CFO from \$1,700M to \$779M

Here is the whole post in one table. Meridian reported operating cash flow of **\$1,700M** in year two, up 42% from **\$1,200M**. Apply the five screens.

| Adjustment | Amount | Which screen found it |
| --- | --- | --- |
| **Reported operating cash flow** | **\$1,700M** | — |
| Less: operating costs capitalized into capex | (\$500M) | Screen 2 — free cash flow did not move |
| Less: one-time cash from receivables sold | (\$197M) | Screen 4 — the receivables footnote |
| Less: one-time cash from the payables stretch | (\$164M) | Screen 3 — DPO moved 45 to 70 days |
| Less: acquired working capital harvested | (\$60M) | Screen 5 — the business combination footnote |
| **Operating cash flow, consistent basis** | **\$779M** | — |

Meridian's operating cash flow did not rise 42% from \$1,200M to \$1,700M. It **fell 35%**, from \$1,200M to \$779M. The entire reported improvement, and then some, came from four decisions about where to put things.

Now carry it through to the derived metrics, because this is where the rebuild pays for itself:

| Metric | Year 1 | Year 2 reported | Year 2 rebuilt |
| --- | --- | --- | --- |
| Operating cash flow | \$1,200M | \$1,700M | \$779M |
| Capital expenditure | \$800M | \$1,300M | \$800M |
| Free cash flow (CFO − capex) | \$400M | \$400M | (\$21M) |
| Free cash flow after acquisitions | \$400M | (\$100M) | (\$521M) |
| Days sales outstanding | 60 days | 42 days | 61 days |
| Days payable outstanding | 45 days | 70 days | 45 days |

Read the DSO row twice. Reported DSO *improved* from 60 days to 42 — the single most reassuring number on the page — while the gross figure, adding back the \$200M of receivables sold, was 61 days. Customers were paying slightly *slower*, and the metric designed to reveal that showed an 18-day improvement.

*The intuition: every one of these tricks improves the ratio that was supposed to catch it. That is not a coincidence — it is a selection effect, because the tricks that do not disable their own alarm get caught early and never become famous.*

### Making the rebuild reproducible

The rebuild is worth automating, because its value comes from applying it identically to every company in a comparison set. Five inputs, all of them disclosed:

```python
import pandas as pd

def rebuild_cfo(row):
    """Restate reported CFO onto a consistent basis.

    All inputs in the same currency unit (here, $M).
    Every field is disclosed: the first three in the cash flow statement
    and its footnotes, the last two in the receivables, supplier-finance
    and business-combination notes.
    """
    adj = (
        - row["capitalized_operating_costs"]   # Trick 1: opex routed into capex
        - row["delta_receivables_sold"]        # Trick 2: YoY change in the programme
        - row["delta_payables_from_dpo_shift"] # Trick 3: DPO move x COGS per day
        - row["acquired_wc_harvested"]         # Trick 5: from the BC footnote
    )
    return row["reported_cfo"] + adj

def gross_dso(row):
    """DSO with sold-but-outstanding receivables added back."""
    revenue_per_day = row["revenue"] / 365.0
    return (row["receivables"] + row["receivables_sold_outstanding"]) / revenue_per_day

def dpo_shift_cash(dpo_now, dpo_prior, cogs):
    """Cash released (+) or consumed (-) by a change in payment terms."""
    return (dpo_now - dpo_prior) * (cogs / 365.0)

meridian = pd.Series({
    "reported_cfo": 1700.0,
    "capitalized_operating_costs": 500.0,
    "delta_receivables_sold": 197.0,
    "delta_payables_from_dpo_shift": dpo_shift_cash(70, 45, 2400.0),  # 164.4
    "acquired_wc_harvested": 60.0,
    "revenue": 3800.0,
    "receivables": 440.0,
    "receivables_sold_outstanding": 200.0,
})

print(round(rebuild_cfo(meridian)))   # 779
print(round(gross_dso(meridian), 1))  # 61.5
```

Two notes on using it in anger. The `delta_receivables_sold` field is the **year-on-year change** in the outstanding sold balance, not the balance itself — a programme held flat contributes nothing, and only its growth or wind-down moves CFO. And `capitalized_operating_costs` is the one input you have to estimate rather than read; the usual approach is to take the year-on-year jump in capex that is not explained by a disclosed build programme, and to sanity-check it against the change in the deferred tax liability.

### What the rebuild does to the peer ranking

The final step is the one people skip. Rebuilding one company's cash flow tells you that company's number was wrong. Rebuilding the whole peer group tells you what to do about it.

Run the same five screens across every comparable company and rebuild each one. Then re-rank. Two things typically happen. The company you were suspicious of drops several places, which you expected. And a company you were not looking at rises — usually one that has been *penalized* by the market for not running a factoring programme, not stretching its suppliers, and not capitalizing aggressively. Conservative accounting looks like weak cash generation right up until you standardize, and standardizing is how you find it.

## Common misconceptions

**"Cash flow cannot be manipulated because the bank confirms it."** The bank confirms the *total*. It has no opinion on which of the three sections a payment belongs to, and neither does the cash reconciliation the auditor performs. Classification sits outside what a cash confirmation tests.

**"If free cash flow is fine, the cash flow statement is fine."** Free cash flow is immune to capitalization and blind to everything else. Factoring, payables stretch, acquired working capital and prepays all flow straight through it. In the worked example, reported free cash flow was identical in both years — \$400M — while the rebuilt figure went from \$400M to negative \$21M.

**"Classification shifting is just aggressive presentation, not real harm."** Two concrete harms. First, debt covenants, credit ratings and bonus schemes are frequently written on operating cash flow, so shifting a dollar into CFO can prevent a covenant breach, hold a rating, or trigger a payout. Second, the structures used to do it — prepays especially — carry a treadmill: each period needs a bigger transaction than the last, and the day the market stops providing them is the day the liquidity crisis becomes visible, usually with no warning.

**"IFRS versus GAAP classification differences wash out in comparisons."** They do not wash out; they are persistent and one-directional. A levered IFRS reporter that elects to classify interest paid as financing will show higher operating cash flow than an identical US filer every single year. Any cross-border screen that ranks on CFO without standardizing is partly ranking accounting policy.

**"A company with rising CFO and rising net income must be healthy."** Capitalization raises both at once, which is precisely why it survives the CFO-to-net-income screen. Meridian's ratio moved from 2.00 to 1.78 — a small decline, to a level that still looks healthy — in a year when its cash generation fell by more than a third.

**"The auditor would have caught it."** Classification is one of the areas where auditors have the least leverage, because the underlying transactions are real, the documentation supports them, and the judgement genuinely belongs to management. An auditor can challenge whether \$500M of network costs create an enduring benefit; they cannot prove they do not. See [how an audit works and what it does not catch](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) for why the assurance model is built this way.

## How it shows up in real markets

Seven episodes, each one a different mechanism from this post, all drawn from enforcement records, parliamentary inquiries and standard-setting documents. Every figure below is attributed to the document it comes from; where a body was making an allegation rather than a finding, the text says so.

**WorldCom, 2001 to Q1 2002 — capitalization.** Approximately **\$3.833 billion** of line costs transferred to capital asset accounts across five quarters — about **\$3.015 billion** in 2001 and **\$818 million** in Q1 2002 — per the SEC's complaint filed **26 June 2002**, announced in *Litigation Release No. 17588* the following day. The mechanism is Trick 1: real cash paid to real carriers, reclassified from operating expense to capital expenditure, lifting reported operating cash flow and reported capex by the same amount. **What would have caught it:** free cash flow, which is arithmetically immune, and a five-year plot of capex as a share of revenue.

**Dynegy, 2001 — the reclassification.** **\$300 million** reported as operating cash flow from Project Alpha — 37% of the year's reported operating cash flow, out of a gross reported figure of \$811 million — plus a **\$79 million** tax benefit, per SEC *Administrative Proceeding 33-8134* of **24 September 2002**, which found the transaction was in substance a loan; a **\$3 million** penalty followed. The mechanism is Trick 4. **What would have caught it:** a working capital bridge that does not reconcile — cash flow improved with no corresponding movement in receivables, inventory or payables.

**Enron, 1997 to 2001 — prepays at scale.** Approximately **\$2.6 billion** through seven J.P. Morgan Chase prepays between December 1997 and September 2001 (SEC *Litigation Release No. 18252*, **28 July 2003**) and **\$3.8 billion** through ten Citigroup prepays between December 1998 and June 2001 (SEC order, Exchange Act Release 34-48230, same date), all structured so the proceeds could be characterized as operating cash flow rather than debt. In fiscal 1999 the structures were the difference between Enron reporting \$1.2 billion generated and \$800 million consumed by operations. **What would have caught it:** the counterparty side. A structure this large needs banks, conduits and repeated renewals, and the trail is visible in more than one filer's disclosures.

**Delphi, Q4 2000 — the round trip.** Approximately **\$270 million** of metals, batteries and generator cores sold at year end with a simultaneous agreement to repurchase them in the following quarter, inflating operating cash flow by approximately **\$200 million** and net income by approximately **\$80 million**, per the SEC's complaint filed **30 October 2006** (*Litigation Release No. 19891*). This is a financing dressed as a sale, and it is the purest possible illustration of borrowing from the next period: the repurchase leg was sitting in the following quarter's numbers the whole time. **What would have caught it:** period-end inventory movements that reverse immediately afterwards.

**General Electric, 2016 to 2017 — pulling cash forward.** The SEC's order of **9 December 2020** (*Press Release 2020-312*, order Release 33-10899) found that GE had boosted a publicly reported cash-flow measure by more than **\$1.4 billion** in 2016 and more than **\$500 million** in the first three quarters of 2017 through a practice it called "deferred monetization" — largely internal sales of long-term receivables from GE Power to GE Capital, with due dates up to five years out — without adequately disclosing that present cash flow was increasing at the expense of future years. By the end of 2017 GE Power Services had monetized more than **\$2.7 billion** of long-term receivables this way. GE paid a **\$200 million** penalty. The order also covered separate disclosure failures relating to insurance reserves, which are a different matter; the receivables monetization is the part relevant here.

This one deserves emphasis because of how ordinary it was. Selling receivables is legal, common, and disclosed by thousands of companies. The finding was about **disclosure of the consequence** — that the cash arriving now would not arrive later. That is precisely the shape of the factoring chart earlier in this post: the boost and the reversal are the same size, and the only question is whether the reader was told. **What would have caught it:** the receivables footnote, gross DSO, and the arithmetic that a programme held flat contributes nothing.

**Carillion, 2016 to 2018 — reverse factoring.** Per the UK Parliament Work and Pensions and BEIS Committees' joint inquiry into the collapse (2018), Moody's and Standard & Poor's both argued that Carillion's Early Payment Facility gave it a financial liability to the banks that should have been presented as "borrowing"; Carillion instead presented it within "other creditors". **Moody's claimed as much as £498 million was misclassified as a result**, though the committees note that Carillion's own audit committee papers show the figure actually drawn was somewhat lower, at **£472 million**. The committees' report spells out the cash flow consequence directly: presenting money borrowed under the facility as "other creditors" meant "its classification within the cashflow can be seen as part of the company's operating activity rather than a financing activity." The company entered liquidation in **January 2018**. In the committees' own words, Carillion "used its early payment facility for suppliers as a credit card, but did not account for it as borrowing." **What would have caught it:** DPO against the peer group, and the question of who was funding the gap.

**The FASB, 2016 — when a label moved and CFO moved with it.** The last case has no villain. **ASU 2016-15** settled several long-running classification arguments, among them that cash received on a beneficial interest in a securitization of the company's own trade receivables is an *investing* inflow rather than operating. Companies that had reported those collections in operating had to move them, and reported operating cash flow fell — at businesses that had done nothing wrong, sold nothing differently, and collected exactly the same cash. **What this proves:** the number is a construct. If a standard-setter's clarification can move it, so can a management team's judgement, and the difference between the two is intent rather than mechanism.

### The pattern across all seven

Line them up and the same three properties appear every time.

The cash was real. In six of the seven, no transaction was fictitious and no bank confirmation was false — Delphi's round trip is the only one with a transaction constructed purely for its accounting effect, and even there the metals genuinely moved.

The trick disabled its own alarm. Capitalization raises CFO and net income together, so the CFO-to-net-income screen stays calm. Factoring raises CFO and improves DSO. The payables stretch raises CFO and improves the cash conversion cycle. Every mechanism improves the ratio designed to detect it, which is why the surviving detection methods are all *reconstructions* rather than ratios.

And the correction was a reclassification, not a writedown. Dynegy moved \$300 million between sections. That is the whole remedy, and it was enough to warrant an SEC order — which is the most direct evidence available that where a dollar sits on this statement is treated, by regulators, as a matter of fact rather than presentation.

## When this matters to you

If you never read a financial statement professionally, the transferable idea is still worth having: **any measurement that is derived from a category rather than from a total can be moved by moving things between categories.** It is the same reason a company can report a lower headcount by reclassifying staff as contractors, or a government a lower deficit by moving spending off-budget. The total is the fact. The subtotal is a decision.

If you do read statements, the practical takeaways are narrow and concrete:

- **Never use reported CFO for a cross-company comparison without standardizing** the interest, dividend and tax lines. IAS 7 requires separate disclosure precisely so you can.
- **Compute free cash flow yourself**, as CFO minus total capital expenditure, and compute a second version that also subtracts cash paid for acquisitions.
- **Read the receivables-sold and supplier-finance footnotes every year** and record the balance. The year-on-year change is the operating cash flow contribution, and it is often the largest single reconciling item you will find.
- **Work in quarterly series, not annual**, when you are looking for timing games. The annual statement is clean by construction.
- **Add back receivables sold before computing DSO.** This one adjustment defeats the most common way a deteriorating collections position is made to look like an improving one.

The next step in this series goes deeper on the financing side of these same structures — how factoring and supply chain finance are used to keep leverage off the balance sheet entirely, and what happens when the facility is withdrawn. If you want the working capital ratios in full, [the cash conversion cycle](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) builds the DSO, DPO and inventory-days toolkit from first principles.

Nothing here is investment advice. It is a description of how a statement is constructed and where its construction is soft.

## Sources & further reading

**Primary enforcement and inquiry sources**

- U.S. Securities and Exchange Commission, *Litigation Release No. 17588* — SEC v. WorldCom, Inc. (27 June 2002), headlining a "\$3.8 Billion Fraud" and alleging income before taxes and minority interests overstated by approximately \$3.055 billion in 2001 and \$797 million in Q1 2002: [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-17588)
- U.S. Securities and Exchange Commission, *SEC v. WorldCom, Inc.* complaint filed 26 June 2002 — the source for the quarterly capitalization amounts (\$771M, \$560M, \$743M and \$941M across 2001; \$818M in Q1 2002) and for the restatement table showing reported 2001 line costs of \$14.739 billion against actual line costs of \$17.754 billion, and reported income of \$2.393 billion against an actual \$622 million loss: [sec.gov](https://www.sec.gov/litigation/complaints/comp17829.htm)
- U.S. Securities and Exchange Commission, *Administrative Proceeding 33-8134* — In the Matter of Dynegy Inc. (24 September 2002), on Project Alpha's \$300 million of reported operating cash flow, the \$79 million tax benefit, the \$3 million penalty, and the restatement reclassifying the Alpha cash flow to financing and reducing 2001 operating cash flow by 37%: [sec.gov](https://www.sec.gov/enforcement-litigation/administrative-proceedings/33-8134)
- U.S. Securities and Exchange Commission, *Litigation Release No. 18252* — SEC v. J.P. Morgan Chase & Co. (28 July 2003), on the approximately \$2.6 billion effectively loaned to Enron through seven prepay transactions between December 1997 and September 2001, and the \$135 million settlement: [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-18252)
- U.S. Securities and Exchange Commission, *In the Matter of Citigroup, Inc.*, Exchange Act Release No. 48230 / AAER No. 1821 (28 July 2003), on the ten Enron prepay transactions totalling \$3.8 billion between December 1998 and June 2001, Projects Nahanni and Bacchus, the fiscal 1999 sign reversal, and the Dynegy Project Alpha findings including the \$811 million gross reported figure: [sec.gov](https://www.sec.gov/litigation/admin/34-48230.htm)
- U.S. Securities and Exchange Commission, *Press Release 2003-87* — SEC Settles Enforcement Proceedings against J.P. Morgan Chase and Citigroup (28 July 2003): [sec.gov](https://www.sec.gov/news/press/2003-87.htm)
- U.S. Senate Permanent Subcommittee on Investigations, hearings and staff materials on Enron prepay transactions (July 2002) — the source of the larger, gross-volume prepay aggregates that circulate for Enron; the figures used in this post are the narrower amounts documented in the SEC actions above: [hsgac.senate.gov](https://www.hsgac.senate.gov/wp-content/uploads/imo/media/doc/072302roach.pdf)
- Neal Batson, *Final Report of the Court-Appointed Examiner*, In re Enron Corp. (2003), on the accounting treatment of the prepay transactions: [concernedshareholders.com](https://www.concernedshareholders.com/CCS_ENRON_Report.pdf)
- U.S. Securities and Exchange Commission, *Litigation Release No. 19891* — SEC v. Delphi Corporation et al. (30 October 2006), on the approximately \$270 million round-trip inventory transactions, \$200 million of inflated operating cash flow and \$80 million of inflated net income: [sec.gov](https://www.sec.gov/enforcement-litigation/litigation-releases/lr-19891)
- U.S. Securities and Exchange Commission, *Press Release 2020-312* — General Electric Agrees to Pay \$200 Million Penalty for Disclosure Violations (9 December 2020), and the accompanying order (Release 33-10899) on deferred monetization of receivables: [sec.gov](https://www.sec.gov/newsroom/press-releases/2020-312)
- UK Parliament, Work and Pensions and BEIS Committees, *Carillion*, HC 769 (2018), on the Early Payment Facility, Moody's claim that as much as £498 million was misclassified, the £472 million actually drawn per Carillion's audit committee papers, and the report's explanation that presenting the facility as "other creditors" placed it in operating rather than financing cash flow: [publications.parliament.uk](https://publications.parliament.uk/pa/cm201719/cmselect/cmworpen/769/769.pdf)
- Moody's and S&P commentary on Carillion's supply-chain finance classification, summarised in *CFO*, "Carillion Collapse Exposes Flaws in Trade Finance Disclosure" (March 2018): [cfo.com](https://www.cfo.com/news/carillion-collapse-exposes-flaws-in-trade-finance-disclosure/659262/)

**Accounting standards — the classification rules themselves**

- FASB *ASC 230, Statement of Cash Flows* (and its predecessor SFAS 95, 1987) — the three-section US statement, and the mandated classification of interest paid, interest received, dividends received (operating), dividends paid (financing) and income taxes paid (operating)
- IASB, *IAS 7, Statement of Cash Flows* — the three-section requirement, the separate-disclosure requirement for interest and dividends, and the operating/investing/financing classification options for interest paid and received, dividends paid and received
- FASB *Accounting Standards Update 2016-15*, "Statement of Cash Flows (Topic 230): Classification of Certain Cash Receipts and Cash Payments" (August 2016) — including the treatment of beneficial interests in securitizations of the entity's own trade receivables as investing inflows, and of insurance proceeds by the nature of the loss
- FASB *Accounting Standards Update 2016-18*, "Statement of Cash Flows (Topic 230): Restricted Cash" (November 2016)
- FASB *Accounting Standards Update 2022-04*, "Liabilities — Supplier Finance Programs (Subtopic 405-50): Disclosure of Supplier Finance Program Obligations" (September 2022) — effective for fiscal years beginning after 15 December 2022, with the obligation rollforward effective for fiscal years beginning after 15 December 2023
- IASB, *Supplier Finance Arrangements — Amendments to IAS 7 and IFRS 7* (May 2023), effective 1 January 2024
- IASB, *IFRS 18, Presentation and Disclosure in Financial Statements* (issued April 2024, effective for annual reporting periods beginning on or after 1 January 2027) — which amends IAS 7 to require the operating profit or loss subtotal as the starting point for the indirect method and to introduce new requirements for the classification of interest and dividend cash flows: [ifrs.org](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-18-presentation-and-disclosure-in-financial-statements/)

**Academic**

- Richard G. Sloan, "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About Future Earnings?", *The Accounting Review* 71(3), 1996 — the foundational evidence that the market over-trusts accrual-heavy earnings relative to cash flow, and the reason cash flow acquired its reputation in the first place

**Elsewhere in this series**

- [Reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) — building the statement line by line, and the indirect-method reconciliation this post assumes
- [Capitalizing costs to inflate profit: the WorldCom move](/blog/trading/forensic-accounting/capitalizing-costs-to-inflate-profit-the-worldcom-move) — where the capitalization boundary actually sits, and the software and R&D grey zones
- [The cash conversion cycle and what working capital reveals](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) — DSO, DPO and inventory days from first principles
- [Round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue) — the revenue side of the round trip
- [Enron 2001: the accounting fraud](/blog/trading/finance/enron-2001-accounting-fraud) — mark-to-market, the special-purpose entities, and the auditor

*Meridian Telecom, Aurora Industrial, Calder Energy and Cascade Networks are hypothetical companies. Every figure attributed to them is illustrative arithmetic constructed for this article, chosen so the reader can check it, and does not describe any real business. Every figure attributed to a named real company is sourced above and is stated as what the cited document alleges, finds, or reports.*
