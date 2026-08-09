---
title: "Factoring, supplier financing, and hiding debt in plain sight"
date: "2026-08-05"
publishDate: "2026-08-05"
description: "How receivables factoring, reverse factoring, and vendor financing move borrowing into working-capital lines that no one calls debt—and the footnote, ratio, and disclosure tests that find it anyway."
tags: ["forensic-accounting", "supply-chain-finance", "reverse-factoring", "factoring", "working-capital", "hidden-debt", "financial-statements", "cash-flow", "carillion", "credit-analysis"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 49
---

> [!important]
> **TL;DR** — Supply-chain finance lets a company borrow from a bank and file the borrowing under "trade payables," so leverage, working capital, and operating cash flow all improve at once without a single new line on the debt schedule.
>
> - Three arrangements get confused constantly: **receivables factoring** (the supplier sells its invoices), **reverse factoring / supply-chain finance** (the *buyer* arranges a bank to pay its suppliers early), and **vendor financing** (the seller lends the buyer the money to buy from it).
> - Reverse factoring is the dangerous one, because the buyer's obligation switches creditor — from the supplier to a bank — while the label on the balance sheet usually stays "trade payables." Neither FASB nor the IASB has changed that classification; both have only mandated disclosure.
> - The tell is that four metrics improve simultaneously and for the same reason: DPO stretches, the cash conversion cycle shortens, reported net debt falls, and operating cash flow jumps. The jump is one-time. The unwind is not.
> - The disclosure vacuum closed late. **FASB ASU 2022-04** applies to fiscal years beginning after 15 December 2022, with the obligation rollforward only from fiscal years beginning after 15 December 2023; the **IASB's May 2023 amendments to IAS 7 and IFRS 7** apply to annual periods beginning on or after 1 January 2024.
> - The number to remember: the parliamentary inquiry into Carillion's collapse (**HC 769**, published 16 May 2018) records **Moody's** — with Standard & Poor's arguing the same thing — putting at as much as **£498 million** the borrowing from financial institutions that reverse factoring let Carillion present as "other creditors." Reported net borrowing at 31 December 2016 was **£218.9 million**. Average net borrowing across that same year was **£586.5 million**.

Suppose I offer you a deal. I will lend you 180 million dollars. You will pay me back with interest. And in exchange for a fee, I will let you record the whole thing on your balance sheet under a line item that no analyst, no rating model, and no loan covenant treats as borrowing.

You would take that deal. Almost every large company that has been offered it has taken it.

This is not a loophole somebody found in a dusty corner of the rulebook. It is a mainstream, multi-trillion-dollar product sold by every major bank, and most of the time it is entirely legitimate — a genuinely useful piece of plumbing that gets small suppliers paid faster and cheaper than they could manage alone. But the same plumbing, run hard enough and disclosed thinly enough, is one of the most effective ways ever devised to make a leveraged company look unleveraged. It helped bring down one of the UK's largest construction and public-services contractors. It destroyed a specialist financier once valued at around \$7 billion. And for most of the last decade, you could not find it in the accounts unless you already knew the words to search for.

![The three-party money flow of reverse factoring: supplier, buyer, and funding bank](/imgs/blogs/factoring-supplier-financing-and-hiding-debt-in-plain-sight-1.webp)

The diagram above is the mental model for the whole post. Look at what moves. Goods and an invoice go from the supplier to the buyer. The buyer approves the invoice and hands that approval to a bank. The bank pays the supplier early, at a small discount. Then, months later, the buyer pays the bank in full.

Now ask the question a forensic analyst asks: at the end of that sequence, who does the buyer owe money to? Not the supplier. The supplier has been paid and is out of the picture. The buyer owes a bank, on terms the buyer negotiated with that bank, at a maturity the buyer chose. That is the textbook description of a loan.

And yet, in the overwhelming majority of financial statements, that obligation sits inside **trade payables** — the same line that holds the electricity bill and the stationery invoice. This post is about why that is, how much it distorts, what the accounting standard-setters finally did about it in 2022 and 2023, and — the part that actually matters if you are reading a set of accounts — how to find it when nobody wants you to.

## The building blocks: payables, receivables, and what "working capital" is really made of

Before any of this makes sense, you need four ideas. If you already know them, skim; if you don't, nothing later will land without them.

### Trade credit is a loan that nobody calls a loan

When a supplier ships you goods and lets you pay in 30 days, it has lent you money. Not in a metaphorical sense — in a literal one. It gave you something of value and accepted a promise of cash later. That is credit.

The company that owes the money records a **trade payable** (also called *accounts payable*): a liability, because it is an obligation to hand over cash. The company that is owed records a **trade receivable**: an asset, because it is a right to receive cash.

The whole edifice of this post is built on one fact: **trade payables are a liability, but almost nobody treats them as debt.** Rating agencies compute *net debt* — borrowings minus cash — and exclude payables. Loan covenants define leverage in terms of *financial indebtedness*, and exclude payables. Analysts compute enterprise value using debt, and exclude payables. The economic logic is sound: trade credit is short, self-liquidating, non-interest-bearing, and arises from operations rather than from a decision to lever up. If you owe your paper supplier £4,000 for 30 days, that is not leverage.

But the exclusion is a *convention*, not a law of nature. And conventions can be arbitraged.

### DSO, DPO, and the cash conversion cycle

To measure how long money sits in each stage, we use three "days" ratios. Each is a balance divided by a daily flow.

**Days sales outstanding (DSO)** — how long customers take to pay you:

\[
\mathrm{DSO} = \frac{\text{Trade receivables}}{\text{Annual revenue}} \times 365
\]

**Days inventory outstanding (DIO)** — how long goods sit in the warehouse:

\[
\mathrm{DIO} = \frac{\text{Inventory}}{\text{Annual cost of goods sold}} \times 365
\]

**Days payable outstanding (DPO)** — how long you take to pay your suppliers:

\[
\mathrm{DPO} = \frac{\text{Trade payables}}{\text{Annual cost of goods sold}} \times 365
\]

Put them together and you get the **cash conversion cycle (CCC)**: the number of days between paying for your inputs and getting paid for your outputs.

\[
\mathrm{CCC} = \mathrm{DSO} + \mathrm{DIO} - \mathrm{DPO}
\]

A positive CCC means you fund the gap yourself. A negative CCC means your suppliers fund you — you collect from customers before you have to pay for what you sold them. Negative CCC is the holy grail of working-capital management, and companies that achieve it honestly (through genuine scale, genuine negotiating power, genuine speed) deserve the credit. [The cash conversion cycle post](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals) works through the ratio mechanics in more depth; here, we only need one insight from it. **DPO is the only term in the CCC that a bank can lengthen for you overnight.** You cannot phone a bank and make your customers pay faster. You cannot phone a bank and make your inventory turn quicker. But you can phone a bank and have it stand between you and your suppliers, and the DPO term will move a hundred days in a single quarter.

### Net debt, and why it is the number under attack

**Net debt** is the headline leverage figure in most of the world:

\[
\text{Net debt} = \text{Borrowings} + \text{Lease liabilities} - \text{Cash and equivalents}
\]

Divide it by EBITDA (earnings before interest, tax, depreciation and amortisation — a rough proxy for operating cash generation) and you get the leverage multiple that drives credit ratings, covenant tests, and a great deal of equity valuation.

Notice what is absent from that formula: trade payables. So if a company can convert a bank borrowing into a trade payable, its net debt falls by the full amount of the borrowing while its cash stays exactly where it is. Nothing about the company's actual obligations has changed. The ratio has moved by the entire amount of the transaction.

### Operating cash flow, and the mechanical link to payables

The cash flow statement's operating section (**CFO**) usually starts from net income and adds back non-cash items, then adjusts for changes in working capital. The working-capital line follows one rule:

- Payables **increase** → cash **increases** (you kept cash you would otherwise have paid out).
- Receivables **increase** → cash **decreases** (you made a sale but haven't collected).

That is why stretching payables shows up as an operating cash inflow. It is not a trick of presentation; it is genuinely true that you have more cash. What is misleading is the *implication*: CFO is read as a measure of the business's ability to generate cash from operations, and a one-time stretch of payment terms is not that. [Reading the cash flow statement](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income) covers the general problem of non-recurring items inside CFO; supply-chain finance is the single largest and most systematic instance of it.

Those four ideas — trade credit is uncounted debt, DPO is the movable term, net debt excludes payables, and payables changes run through CFO — are the entire mechanism. Everything below is elaboration.

## Three arrangements that look alike and are not

The vocabulary in this area is a mess. "Factoring," "reverse factoring," "supply-chain finance," "payables finance," "confirming," "vendor financing," "invoice discounting," "receivables purchase," and "early payment programme" are used loosely and often interchangeably by people who should know better. They are not the same thing, and the differences determine which company gets the benefit and which line of which statement moves.

![A comparison matrix of receivables factoring, reverse factoring, and vendor financing](/imgs/blogs/factoring-supplier-financing-and-hiding-debt-in-plain-sight-2.webp)

Here is the distinction in one sentence each.

- **Receivables factoring**: *the supplier* sells its own invoices to a financier for cash today. The supplier initiates it. The financier prices the credit of the supplier's *customers*.
- **Reverse factoring / supply-chain finance (SCF)**: *the buyer* sets up a facility under which a bank pays the buyer's suppliers early. The buyer initiates it. The financier prices the credit of the *buyer*, which is why the pricing is usually much better than the supplier could get alone.
- **Vendor financing**: *the seller* lends the buyer the money — or guarantees the buyer's borrowing — so the buyer can afford to buy. The seller initiates it, and the seller books the revenue.

The reason reverse factoring is called "reverse" is that the flow of initiative runs backwards compared with ordinary factoring: the strong party at the end of the chain organises financing for the weak parties at the start of it.

### Receivables factoring: the supplier's tool

A supplier with slow-paying customers has cash tied up in receivables. It sells those receivables — usually at a discount of one to three percent of face value, sometimes with an initial advance of 80–90% and a reserve released on collection — and gets cash now.

The pivotal accounting question is **recourse**.

- **Non-recourse factoring**: the factor buys the credit risk. If the customer never pays, that is the factor's loss. Under both IFRS 9 and US GAAP (ASC 860), if substantially all the risks and rewards have genuinely transferred, the supplier *derecognises* the receivable. The asset leaves the balance sheet. No liability appears. The cash arrives inside operating cash flow, because collecting a trade receivable is an operating activity.
- **Recourse factoring**: the supplier remains on the hook if the customer defaults. The risks have not transferred. The receivable *stays* on the balance sheet, and the cash received is recorded as a **secured borrowing**. It shows up in debt, and the cash arrives inside *financing* cash flow.

Same cash. Same day. Two completely different sets of financial statements, decided by a clause about who eats the loss. That asymmetry is the whole game, and it is why the recourse language in a factoring agreement is worth reading even when the amounts look small.

#### Worked example: Northwind factors \$2 million of invoices

The numbers here are illustrative — chosen to be easy to follow, not drawn from any real company.

Northwind Components has annual revenue of \$12.0 million and trade receivables of \$3.0 million. Its starting DSO:

\[
\mathrm{DSO} = \frac{\$3.0\text{m}}{\$12.0\text{m}} \times 365 = 91.25 \text{ days}
\]

It has \$1.0 million of bank borrowings and \$0.2 million of cash, so net debt is \$0.8 million.

Northwind factors \$2.0 million of those receivables at a 2% discount, receiving \$1.96 million in cash and recording a \$40,000 charge.

**Case A — non-recourse.** The factor takes the credit risk, so the receivables are derecognised.

- Receivables: \$3.0m → \$1.0m
- Cash: \$0.2m → \$2.16m
- Borrowings: unchanged at \$1.0m
- New DSO: \(\frac{\$1.0\text{m}}{\$12.0\text{m}} \times 365 = 30.42\) days
- New net debt: \(\$1.0\text{m} - \$2.16\text{m} = -\$1.16\text{m}\) — a net *cash* position
- CFO: up \$1.96 million, because a receivable turned into cash

Northwind's DSO fell by 61 days and its net debt went from +\$0.8 million to −\$1.16 million, all in one afternoon. It did not collect a single invoice faster. It did not improve its credit control. It sold the invoices.

**Case B — with recourse.** Identical cash, identical timing, but Northwind still bears the default risk.

- Receivables: unchanged at \$3.0m
- Cash: \$0.2m → \$2.16m
- Borrowings: \$1.0m → \$3.0m (a \$2.0m secured borrowing appears)
- DSO: unchanged at 91.25 days
- Net debt: \(\$3.0\text{m} - \$2.16\text{m} = \$0.84\text{m}\) — essentially unchanged, worse by the \$40,000 fee
- CFO: unchanged; the \$1.96 million is a *financing* inflow

The intuition: **whether factoring flatters your statements or not depends entirely on a risk-transfer clause, and the two versions can be economically almost identical.**

### Reverse factoring: the buyer's tool, and the one that hides

Now flip the initiative. A large buyer — investment grade, well known, the sort of counterparty a bank is happy to lend to for 120 days at a thin spread — goes to a bank and sets up a programme. The mechanics run like this:

1. The supplier delivers goods and issues an invoice, on whatever terms the buyer has imposed.
2. The buyer approves the invoice on the bank's platform. This approval is typically **irrevocable**: the buyer confirms the invoice is valid, the amount is agreed, and it waives any right to dispute or set off later. That irrevocability is what makes the asset financeable, because the bank is no longer taking any commercial risk on whether the goods were faulty.
3. The supplier can, at its option, click a button and be paid immediately by the bank, less a discount for the remaining days.
4. On the original due date, the buyer pays the **bank** the full face amount.

![Cash-flow timeline of one invoice through a supply-chain finance programme](/imgs/blogs/factoring-supplier-financing-and-hiding-debt-in-plain-sight-3.webp)

Look carefully at what the buyer gained. Nothing, if the terms stayed the same — it pays the same amount on the same day, just to a different party. The buyer's gain comes from the *quid pro quo* that almost always accompanies the programme: **because suppliers can now get paid on day 10 instead of day 30, the buyer extends its standard terms from 30 days to 90 or 120.**

That is the trade. The supplier gets its money earlier than before. The buyer gets to hold onto its cash three or four times longer than before. The bank earns a spread on financing that gap. Everyone can describe this as a win, and in many programmes it genuinely is one — a small supplier who was borrowing at 12% on an overdraft can now be funded at the buyer's investment-grade rate.

But notice the buyer's balance sheet. Its payables balance is now three or four times larger than it used to be, and the extra amount is, in economic substance, money borrowed from a bank. In accounting presentation, it is trade payables.

#### Worked example: one \$1,000,000 invoice, priced

Again, illustrative numbers.

A supplier issues a \$1,000,000 invoice on day 0. Under the old terms it would have been paid on day 30. Under the new SCF-enabled terms the buyer pays on day 120.

The supplier elects early payment on day 10 and receives \$985,000. It has given up \$15,000 to be paid 110 days early.

What rate is that? On the proceeds actually received:

\[
\text{Annualised cost} = \frac{\$15{,}000}{\$985{,}000} \times \frac{365}{110} = 5.05\%
\]

Just over 5% a year. That is a genuinely good rate for a small supplier — it is the *buyer's* borrowing rate plus a modest programme margin, not the supplier's.

Now run the supplier's decision properly. Suppose (illustratively) the supplier's own overdraft costs 10% a year. Financing 110 days itself would cost:

\[
\$985{,}000 \times 10\% \times \frac{110}{365} = \$29{,}685
\]

versus \$15,000 through the programme. The supplier saves about \$14,685 and joins. Rational.

But compare against the world that existed *before* the programme, when the supplier was paid on day 30 and financed nothing:

- Old world: paid \$1,000,000 on day 30. Financing cost for days 0–30 at 10%: \(\$1{,}000{,}000 \times 10\% \times \frac{30}{365} = \$8{,}219\).
- New world: paid \$985,000 on day 10. Financing cost for days 0–10 at 10%: \(\$985{,}000 \times 10\% \times \frac{10}{365} = \$2{,}699\), plus the \$15,000 discount = \$17,699.

The supplier is about \$9,480 *worse off* than under the old 30-day terms. It is better off than under 120-day terms with no programme, which is the only comparison the buyer ever puts in the slide deck.

The intuition: **supply-chain finance is usually sold as cheaper financing for the supplier, but the supplier's true benchmark is not "120 days unfunded" — it is the 30-day terms it used to have.**

### Vendor financing: the seller lends the buyer the money to buy

The third arrangement inverts the direction of the loan entirely. Here, the company selling something provides — or guarantees — the financing the customer needs to buy it. Equipment manufacturers, telecom infrastructure suppliers, and capital-goods companies do this routinely.

The accounting effect at the seller is the interesting one. The seller books revenue today, in full, and records a long-dated receivable or loan. Cash flow does not arrive for years. If the customer was only able to buy because the seller lent it the money, then the "revenue" is really the seller's own capital going out and coming back labelled as sales.

#### Worked example: vendor financing turns capital into revenue

Illustrative. A manufacturer sells \$50 million of equipment to a customer and simultaneously extends a three-year, \$50 million loan so the customer can pay for it.

- Revenue: **+\$50 million**, recognised at delivery
- Gross profit at a 40% margin: **+\$20 million**
- Cash from operations: **\$0** — no cash arrived
- A \$50 million loan receivable appears on the balance sheet
- Financing/investing cash outflow: **−\$50 million** if the loan is funded in cash

The income statement shows a \$50 million sale and \$20 million of profit. The cash flow statement shows \$50 million going *out*. The company's earnings went up and its cash went down, by the same order of magnitude, because of a single transaction.

Now add the second-order case. Suppose the customer's ability to repay depends on it raising more capital, and capital markets close. The manufacturer takes a bad-debt provision. Under the illustrative numbers, a full write-off reverses \$50 million of assets against \$20 million of previously reported profit — a net destruction of \$30 million of shareholder capital on a sale that once looked like a triumph.

The intuition: **vendor financing converts the seller's balance sheet into the buyer's purchasing power, and the revenue it produces is only as real as the customer's ability to repay.**

This is the same family of problem as [round-tripping and fabricated revenue](/blog/trading/forensic-accounting/round-tripping-and-fabricated-revenue): money leaves the company and comes back looking like a sale. Vendor financing is the legal, disclosed, GAAP-compliant cousin. It becomes fraud only when the circularity is concealed.

## Why it lands in "trade payables" instead of "debt"

Here is the question that took the accounting profession the better part of a decade to not answer: when a buyer's payable is settled by a bank, and the buyer now owes the bank instead, has the original trade payable been *extinguished* and replaced with a *financial liability*?

Under IFRS, a financial liability is derecognised only when it is extinguished — discharged, cancelled, or expired — or when it is substantially modified. Under US GAAP the analogous concept lives in ASC 405-20. If the SCF arrangement substantially modifies the original obligation, the trade payable goes and a bank borrowing appears. If it does not, the payable stays a payable.

The practical answer that emerged is that it *usually* does not, because from the buyer's perspective the amount, the due date, and the underlying commercial obligation are frequently unchanged — only the identity of the payee has moved. That is a defensible technical reading. It is also why programmes are carefully structured to preserve exactly those features: keep the amount identical, keep the buyer's payment date identical to the invoice's stated date, take no security, charge the buyer no explicit interest, and the payable survives as a payable.

The **IFRS Interpretations Committee** published an agenda decision on reverse factoring in **December 2020**, which set out what preparers should consider when deciding presentation and what they must disclose. It did not create a bright-line rule, and it did not force reclassification. It set out the *indicators* to weigh.

Those indicators are the analyst's real checklist. Ask, of any programme:

| Question | Points toward "trade payable" | Points toward "borrowing" |
|---|---|---|
| Who is the creditor now? | Still commercially the supplier's claim | A bank, with its own rights |
| Are the buyer's payment terms unchanged? | Yes, same date as the invoice | Extended beyond the original due date |
| Are the terms in line with industry norms? | Yes | Materially longer than peers |
| Did the buyer give security or a guarantee? | No | Yes |
| Does the buyer pay interest or a facility fee? | No; the supplier bears the discount | Yes, explicitly |
| Who arranged and pays for the facility? | Not applicable | The buyer |
| Is the obligation legally novated to the bank? | No | Yes |

The more of those that fall in the right column, the harder it is to argue the liability is still trade credit — and the more likely a rating agency or a careful analyst will move it regardless of where the company put it.

![The same \$180 million shown two ways: as borrowings and as trade payables](/imgs/blogs/factoring-supplier-financing-and-hiding-debt-in-plain-sight-4.webp)

#### Worked example: Meridian's \$180 million, two ways

Illustrative throughout. Meridian Industrial has annual cost of goods sold of \$730 million — a round \$2 million of cost every day of the year. Its trade payables are \$60 million, so:

\[
\mathrm{DPO} = \frac{\$60\text{m}}{\$730\text{m}} \times 365 = 30 \text{ days}
\]

It has \$300 million of borrowings, \$40 million of cash, and EBITDA of \$100 million. Reported net debt is \$260 million, and leverage is 2.6×. Its main covenant caps net debt at 3.0× EBITDA, so it has \$40 million of headroom — thin.

Meridian needs \$180 million of liquidity. It has two ways to get it.

**Option A — draw \$180 million on the revolving credit facility.**

- Cash: \$40m → \$220m
- Borrowings: \$300m → \$480m
- Trade payables: unchanged at \$60m
- Net debt: \(\$480\text{m} - \$220\text{m} = \$260\text{m}\)
- Leverage: **2.6×**
- CFO: unchanged. The \$180 million is a financing inflow.

**Option B — launch a supply-chain finance programme and extend supplier terms from 30 days to 120 days.**

Payables rise to \(\$2\text{m/day} \times 120 = \$240\text{m}\), releasing \$180 million of cash.

- Cash: \$40m → \$220m
- Borrowings: unchanged at \$300m
- Trade payables: \$60m → \$240m
- Net debt: \(\$300\text{m} - \$220\text{m} = \$80\text{m}\)
- Leverage: **0.8×**
- CFO: **up \$180 million**, because payables rose

Total liabilities in both cases: \$540 million. Same amount owed. In Option A a large part of it is owed to a bank; in Option B a large part of it is owed to *the same bank*, sitting behind the suppliers. The economics are close to identical.

Reported leverage: 2.6× versus 0.8×. Covenant headroom: \$40 million versus \$220 million. Operating cash flow: \$80 million versus \$260 million.

The intuition: **the choice between a revolver draw and a supply-chain finance programme is, from the balance sheet's point of view, a choice about which line to write the number on — and one of those lines is not counted as debt by anybody.**

This is the same structural move as the arrangements in [off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities), executed without needing an entity at all. And it belongs on the same list as the items in [hidden liabilities: leases, guarantees, and contingencies](/blog/trading/forensic-accounting/hidden-liabilities-leases-guarantees-and-contingencies) — obligations that are real, contractual, and outside the debt schedule.

## Four ratios that improve at once, for the same reason

The distinguishing signature of supply-chain finance is not that one metric moves. It is that **several metrics that are usually in tension all improve simultaneously**, and every one of them improves because of the same underlying transaction.

Ordinarily, working-capital improvements involve trade-offs. Cutting inventory risks stockouts. Tightening credit terms costs you sales. Paying suppliers later strains relationships and eventually costs you price. Supply-chain finance suspends the trade-offs, because a bank is absorbing the strain that would otherwise show up somewhere.

### DPO detaches from its peers

The cleanest fingerprint is a payables line that walks away from the industry.

![Days payable outstanding drifting away from the peer median over six years](/imgs/blogs/factoring-supplier-financing-and-hiding-debt-in-plain-sight-5.webp)

Payment terms in a given industry are sticky. They reflect the physical rhythm of the trade, the relative bargaining power of the parties, and decades of habit. A grocer's terms and an aerospace primes' terms are different from each other but each is fairly stable within its sector. So when one company's DPO climbs from 30 days to 122 days over four years while the peer median sits between 31 and 34, that is not a negotiating triumph. Something structural has changed.

In the illustrative chart, the gap reaches 89 days. At \$2 million of cost per day, that is:

\[
89 \text{ days} \times \$2\text{m/day} = \$178 \text{ million}
\]

of financing sitting inside a line called "trade payables." That is the number to add back before computing leverage.

#### Worked example: the cash conversion cycle goes negative

Continuing with Meridian, illustratively. Before the programme it had DSO of 45 days and DIO of 60 days:

\[
\mathrm{CCC} = 45 + 60 - 30 = 75 \text{ days}
\]

Meridian funds 75 days of its own operating cycle. At \$2 million of daily cost, that is roughly \$150 million of working capital permanently employed.

After the programme, with DPO at 120:

\[
\mathrm{CCC} = 45 + 60 - 120 = -15 \text{ days}
\]

A negative cash conversion cycle. In a screening tool, Meridian now sits alongside the handful of businesses whose customers pay before their suppliers do — a category that historically signals enormous commercial power.

Meridian has no such power. It has a facility. Its customers still take 45 days; its inventory still turns in 60. The only thing that changed is that a bank agreed to stand in the middle, and the fee for that is buried in supplier pricing rather than in interest expense.

The intuition: **a negative cash conversion cycle earned through scale is a moat; a negative cash conversion cycle rented from a bank is a liability with good manners.**

### Operating cash flow outruns earnings, once

The second fingerprint is in the cash flow statement, and it has a distinctive time shape.

![Cash flow from operations decomposed across four years, including the unwind](/imgs/blogs/factoring-supplier-financing-and-hiding-debt-in-plain-sight-6.webp)

#### Worked example: CFO that outruns earnings, and then does not

Illustrative. Meridian's net income is \$30 million and its depreciation and amortisation is \$50 million, steady across all four years.

**FY2020, before the programme.** CFO = \$30m + \$50m + \$0 working capital change = **\$80 million**. CFO ÷ net income = 2.7×. Unremarkable and healthy.

**FY2021, the launch year.** Terms move from 30 to 120 days, payables rise \$180 million.

\[
\mathrm{CFO} = \$30\text{m} + \$50\text{m} + \$180\text{m} = \$260 \text{ million}
\]

CFO ÷ net income = **8.7×**. Free cash flow triples. Management describes an outstanding year for cash conversion. If the programme is not disclosed, the analyst sees an extraordinary working-capital performance with no obvious cause.

**FY2022, steady state.** Terms are already at 120 days; they cannot stretch again without another renegotiation. Payables stay at \$240 million, so the working-capital contribution is zero.

\[
\mathrm{CFO} = \$30\text{m} + \$50\text{m} + \$0 = \$80 \text{ million}
\]

CFO falls 69% year on year with no deterioration whatsoever in the underlying business. This is the moment when an unaware analyst writes "disappointing cash conversion" in a note about a company whose operations did not change.

**FY2023, the facility is withdrawn.** The bank re-prices its risk appetite — a downgrade, a sector concern, a decision to exit the product — and pulls the programme. Suppliers revert to 30-day terms. Payables must fall from \$240 million back to \$60 million.

\[
\mathrm{CFO} = \$30\text{m} + \$50\text{m} - \$180\text{m} = -\$100 \text{ million}
\]

Meridian must find \$180 million of cash in a single working-capital cycle, at exactly the moment a bank has decided it does not want the exposure.

The intuition: **the payables stretch is a one-time inflow that is reported inside a recurring line, and its reversal is not optional, not gradual, and not scheduled by the borrower.**

That last point deserves emphasis, because it is the actual credit risk. Ordinary debt has a maturity date printed on it. You can see it in the maturity table, plan the refinancing, and negotiate ahead of time. A supply-chain finance facility is typically **uncommitted and cancellable at short notice**. Its economic maturity is "whenever the bank feels like it." A company that has \$180 million of financing inside its payables has \$180 million of debt with the worst maturity profile available — and no maturity table will show it.

#### Worked example: the covenant that was never really there

Illustrative. Meridian's covenant caps net debt at 3.0× EBITDA. Under Option B, reported leverage was 0.8× — apparently \$220 million of headroom.

Now apply a rating-agency style adjustment: add back the \$180 million of financing sitting inside payables.

\[
\text{Adjusted net debt} = \$300\text{m} + \$180\text{m} - \$220\text{m} = \$260\text{m}
\]

\[
\text{Adjusted leverage} = \frac{\$260\text{m}}{\$100\text{m}} = 2.6\times
\]

Real headroom: \$40 million, not \$220 million.

Now stress it, and be realistic about what happened to the cash. Meridian raised the \$180 million in order to *spend* it — that is why treasurers run these programmes — so assume it funds capital expenditure and a bolt-on acquisition, taking cash back down to \$40 million. Reported net debt is now \(\$300\text{m} - \$40\text{m} = \$260\text{m}\), or 2.6×. Still inside the covenant, on the reported basis, with \$40 million to spare.

Then the facility is withdrawn. Payables must fall from \$240 million back to \$60 million, and Meridian no longer has the cash, so it draws \$180 million on the revolver to pay the suppliers:

\[
\text{Net debt} = (\$300\text{m} + \$180\text{m}) - \$40\text{m} = \$440\text{m} \quad\Rightarrow\quad \frac{\$440\text{m}}{\$100\text{m}} = 4.4\times
\]

Covenant breached, without a single day of operational underperformance. And note the path: reported leverage went 0.8× at launch, 2.6× once the cash was deployed, 4.4× on withdrawal. Only the middle number ever looked like a warning, and only to someone who had already added the programme back.

The intuition: **covenant headroom computed on reported net debt is fictional whenever an undisclosed programme is doing the work; the adjusted number is the one that will actually be tested.**

## Recourse, disclosure, and who is really holding the risk

It is worth being precise about where risk actually sits in each arrangement, because the marketing language obscures it.

| Arrangement | Who holds the credit risk after the transaction | What breaks it |
|---|---|---|
| Non-recourse factoring | The factor holds the customer's credit risk | Customers default; factor tightens or exits |
| Recourse factoring | The supplier still holds it | Same, but the loss lands on the supplier |
| Reverse factoring | The bank holds the *buyer's* credit risk | The buyer is downgraded; bank cancels the facility |
| Vendor financing | The seller holds the buyer's credit risk | The buyer cannot pay; seller writes off both the loan and the profit |

In reverse factoring, the bank's exposure is concentrated on one name — the buyer — across potentially thousands of invoices. That concentration is exactly why programmes get cancelled fast. A bank managing a diversified loan book can tolerate one borrower deteriorating; a bank sitting on several hundred million of uncommitted exposure to a single deteriorating credit will reduce it at the first opportunity, and it has the contractual right to.

The supplier, meanwhile, has usually not understood that it has swapped one risk for another. It no longer worries about whether the buyer pays — the bank has paid it. But it has become structurally dependent on the buyer's *credit rating*, because if the rating falls the facility disappears and the supplier is back to waiting 120 days, not 30. Its terms were permanently worsened in exchange for a benefit that can be revoked by a third party it has no relationship with.

## The disclosure vacuum, and what finally closed it

For roughly a decade, the position was this: reverse factoring balances sat inside trade payables, and there was no requirement anywhere to say how much of that line was financed by banks, which banks, on what terms, or what would happen if they stopped. You could read a full set of audited accounts, cover to cover, and have no way to know that a third of the payables balance was bank money.

The regulators noticed the outline of the problem before the standard-setters acted on it. The US **Securities and Exchange Commission** issued comment letters asking companies to explain their arrangements — reportedly to Keurig Dr Pepper and Masco in 2019, and to at least four more companies including Procter & Gamble, Graphic Packaging Holding, Boeing, and Coca-Cola in 2020. The staff reportedly spotted the programmes the same way an analyst would: mentions of third-party providers in the MD&A, unusually large increases in accounts payable, and sharp jumps in DPO.

Two standards eventually closed the disclosure gap.

**FASB ASU 2022-04, *Liabilities — Supplier Finance Programs* (Subtopic 405-50).** It requires a buyer in a supplier finance programme to disclose the key terms of the programme, the amount of obligations outstanding that the buyer has confirmed as valid, where those obligations sit on the balance sheet, and — annually — a **rollforward** of the obligation balance. It is effective for fiscal years, including interim periods, beginning after **15 December 2022**, except for the rollforward, which is effective for fiscal years beginning after **15 December 2023**. Early adoption is permitted.

**IASB amendments to IAS 7 and IFRS 7, *Supplier Finance Arrangements*, issued 25 May 2023.** These require disclosure of the terms and conditions of the arrangements, the carrying amount of the liabilities that are part of them and where they sit, the portion **for which the suppliers have already been paid by the finance providers**, the range of payment due dates for both the financed and comparable non-financed payables, and liquidity-risk information. They are effective for annual reporting periods beginning on or after **1 January 2024**, with early application permitted. The UK Endorsement Board adopted them in December 2023.

Now the critical point, and it is the one most commentary gets wrong: **neither standard changes the classification.** Both are disclosure standards. A US or IFRS filer can still present the entire programme inside trade payables. What has changed is that it must now tell you how much is in there.

That is a real improvement, and it puts a number in your hands you previously had to estimate. It also means the analytical work has moved rather than disappeared: the reclassification is now *your* job, and the standards have merely handed you the input.

### What a good disclosure now looks like

Boeing's FY2024 Form 10-K illustrates what the rollforward gives you. Under the heading *Supply Chain Financing Programs*, it tabulates the changes in accounts payable to participating suppliers: a beginning balance of **\$2,871 million** at 1 January 2024, **\$12,476 million** of additions, **\$12,644 million** of reductions for payments made, and an ending balance of **\$2,703 million** at 31 December 2024.

Three things fall out of that immediately. First, the size: roughly \$2.7 billion sitting inside accounts payable, which you can now add to debt if your methodology says you should. Second, the *velocity*: \$12,476 million added against a \$2,703 million closing balance means the balance turns over roughly 4.6 times a year on the closing balance — an implied average life of about 79 days, and about 82 days if you use the average of the opening and closing balances instead. Boeing's own description corroborates that arithmetic: it states the majority of amounts payable under the programmes are due within 30 to 90 days, though some may extend up to 12 months. A balance whose implied life lands near the top of its stated majority range is a genuine working-capital rhythm rather than a term borrowing dressed up — and the 12-month tail is the part to keep an eye on. Third, the direction: the balance shrank slightly year on year, so the programme contributed a small *negative* to operating cash flow in 2024 rather than flattering it.

None of that was computable before the rollforward existed. That is the whole case for the standard.

## How the rating agencies adjust — and why the industry fought it

Rating agencies got there before the standard-setters, and they got there because of specific companies.

In late 2015, **Moody's** said the large-scale reverse factoring programme run by the Spanish engineering group **Abengoa** had "debt-like" features and announced a review of its methodology. The trade finance industry pushed back hard; the International Trade and Forfaiting Association argued that payables finance is a legitimate and long-established form of finance and should not be recharacterised wholesale.

**Fitch** took the more explicit position, in a report published in **August 2018**, seven months after Carillion's liquidation. It said reverse factoring could have a potentially large impact on vulnerability to default, and that it would adjust credit metrics to classify any extension of payment terms attributable to reverse factoring as debt. Its stated example was Carillion: reverse factoring, Fitch said, "allowed the outsourcer to show an estimated £400-£500mn of debt to financial institutions as 'other payables' compared to reported net debt of £219mn" — that last figure being the rounding of Carillion's reported **£218.9 million** of net borrowing at 31 December 2016. Fitch called it an accounting loophole. The industry rejected the general reclassification, and the objection has commercial force — the entire appeal of the product to corporate treasurers is that it is *not* counted as leverage, so counting it as leverage removes the reason to use it.

That argument is worth taking seriously rather than dismissing. There is a real distinction between a programme that finances 30-day payables at industry-standard terms (plumbing) and one that finances 120-day payables in a 30-day industry (borrowing). The honest analytical position is not "all SCF is debt" or "no SCF is debt." It is:

**The portion of payables attributable to terms longer than the company's own historical norm, or its peers' norm, is debt. The rest is trade credit.**

That gives you an operational rule:

\[
\text{Debt-equivalent} = (\mathrm{DPO}_{\text{reported}} - \mathrm{DPO}_{\text{normal}}) \times \frac{\text{Annual COGS}}{365}
\]

where \(\mathrm{DPO}_{\text{normal}}\) is the company's own pre-programme DPO, or the peer median, whichever you can defend. For Meridian: \((122 - 33) \times \$2\text{m} = \$178\) million. It is an estimate, and you should say so, but it is a far better estimate than zero.

## The detection playbook

Suppose you are reading a set of accounts and nobody has told you anything. Here is the order of operations.

![A five-step decision flow for detecting undisclosed supplier finance](/imgs/blogs/factoring-supplier-financing-and-hiding-debt-in-plain-sight-8.webp)

**Step 1 — Look for payables growing faster than cost of goods sold.** Compute DPO for the last five or six years. If payables are outgrowing COGS, days are being added rather than volume. Also watch *other payables* and *other creditors*: when a company decides its SCF obligation does not belong in trade payables but is not willing to call it debt, this is where it lands. A fast-growing, unexplained "other payables" line is a specific red flag, and it is precisely where Carillion's sat.

**Step 2 — Grep the footnotes.** These programmes are almost always named somewhere, if only in a passing sentence. Search the full filing for:

`supply chain finance` · `supplier finance` · `payables finance` · `reverse factoring` · `confirming` · `confirmación` · `early payment` · `supplier early payment` · `receivables purchase` · `invoice discounting` · `structured payables` · `vendor financing` · `dynamic discounting`

Also search for the *providers* — the platform and bank names — because a company that will not describe the arrangement will often still name the counterparty in a commitments note. And check the MD&A liquidity section, which is where a programme's existence tends to surface first, usually framed as an efficiency initiative. [The footnotes and MD&A post](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried) makes the general case for reading these sections first; supply-chain finance is the strongest single example of why.

**Step 3 — Benchmark DPO against peers.** Terms are an industry characteristic. A 20-day gap above the peer median needs an explanation; a 60-day gap almost never has an innocent one. Be careful to compute peers' DPO on the same basis (COGS, not revenue; year-end balances or averages, but consistently).

**Step 4 — Test the cash flow statement.** Compute CFO ÷ net income over five years and decompose the working-capital contribution. If CFO exceeds three times net income and the payables movement is the reason, you have found a financing inflow presented as operating cash. Treat the excess as financing when you compute free cash flow. Look specifically for the **shape**: one enormous year followed by flat years is a stretch that has run its course.

**Step 5 — Ask the withdrawal question.** This is the one that converts an accounting observation into a credit view. *If this facility were cancelled in 30 days, what happens?* Take your estimated debt-equivalent, assume terms revert to the pre-programme norm, and model the cash requirement over one working-capital cycle. Compare it to undrawn committed facilities and cash on hand. If the answer is that the company cannot fund the snapback, then the reported leverage number is not merely optimistic — it is describing a different company.

Two additional checks worth running when the data is available:

- **Reconcile disclosed programme balances against total payables.** Under ASU 2022-04 or the IAS 7 amendments, you can now often compute the financed fraction directly. A programme balance exceeding a third of total trade payables is a structural dependency, not a treasury convenience.
- **Read the supplier side too.** If you cover a company that *sells* to a large SCF user, check whether its DSO has fallen while its receivables-factoring costs have risen. The buyer's flattering DPO and the supplier's financing expense are the same transaction seen from opposite ends. [Reading the balance sheet](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here) is the right companion for that kind of two-sided reconstruction.

## Common misconceptions

**"Supply-chain finance is a fraud."** No. It is a mainstream financial product with genuine economic value, and the great majority of programmes are run by companies with no intention of misleading anyone. The FCI reported global factoring turnover of **€3,894 billion in 2024**, up 2.7% on €3,791 billion in 2023, and **€4,039 billion in 2025**. BCR Publishing's World Supply Chain Finance Report put global SCF volumes at **US\$2,184 billion** in its 2023 edition, up 21% year on year, with funds in use of US\$858 billion. An industry of that size is not a conspiracy. The problem is not the product; it is the combination of the product with a disclosure regime that, until very recently, let you use it without saying so.

**"If it were really debt, the auditors would have made them call it debt."** The auditors were applying the standards as written, and the standards genuinely did not require reclassification in most fact patterns. The IFRS Interpretations Committee's December 2020 agenda decision and the 2022–2023 standards both addressed *disclosure*, not classification. Reclassification remains a matter of judgement, which is why the answer varies between the company, its auditor, and its rating agency. [How an audit works](/blog/trading/forensic-accounting/how-an-audit-works-and-what-it-does-not-catch) is the relevant background: an audit tests compliance with a framework, not whether the framework produces an economically sensible answer.

**"Post-2024, the disclosures fix it."** They improve it enormously, but they have three gaps. The rules apply to *programmes the buyer arranges*; a supplier that independently factors its receivables from a buyer creates a similar economic effect with no buyer-side disclosure at all. The disclosures are also periodic, so a balance managed down before the reporting date tells you less than you would like — the ASU rollforward helps here, which is exactly why it matters. And they say nothing about *committed* versus *uncommitted*, which is the single most important credit fact about the facility.

**"A high DPO is always a bad sign."** It is not. Genuine scale buyers really do command long terms, and paying suppliers slowly at negotiated prices is a legitimate strategy. What matters is the *derivative* — how fast DPO is changing, and whether the change coincides with a programme, a covenant test, or an earnings target. A stable 90-day DPO in an industry that runs on 90 days is information about the industry. A 30-to-120-day move in four years is information about the treasury department.

**"Non-recourse factoring is always clean because the receivable is genuinely sold."** The sale can be genuine and the presentation can still mislead. Non-recourse factoring produces an operating cash inflow that will not repeat unless the company factors again, so a company that factors more each year is showing growing CFO from a financing decision. And "non-recourse" agreements frequently carry carve-outs — dilution, commercial disputes, concentration limits — that return risk to the seller. Read the definition of recourse rather than the label on the front page.

**"Vendor financing is the same problem as supply-chain finance."** They are opposite in direction. Supply-chain finance flatters the *buyer's* balance sheet and cash flow. Vendor financing flatters the *seller's* income statement while damaging its cash flow. Both convert a financing decision into an operating-looking number; they just do it at different ends of the transaction.

## How it shows up in real markets

### Carillion: half a billion pounds filed under "other creditors"

Carillion was one of the UK's largest construction and public-services contractors. It went into compulsory liquidation on **15 January 2018**, with just under **£7 billion** of liabilities against **£29 million** of cash — including roughly **£2 billion** owed to some **30,000 suppliers**.

![Timeline of Carillion's early payment facility, from its 2013 launch through liquidation to the post-mortem that finally put a number on it](/imgs/blogs/factoring-supplier-financing-and-hiding-debt-in-plain-sight-7.webp)

Carillion launched an **Early Payment Facility (EPF)** with Santander in **2013**. The structure was standard reverse factoring: suppliers could be paid early by the bank at a discount, and in return Carillion pushed its standard payment terms out toward **120 days** for suppliers who did not join. The parliamentary inquiries into the collapse found that Carillion had used its suppliers to prop up a failing business model, and that it enforced those long terms despite being a signatory to the UK's Prompt Payment Code.

The accounting is the part that matters here. **Moody's** and **Standard & Poor's** both argued that Carillion's treatment of the EPF concealed its true level of borrowing from financial creditors: the structure created a financial liability to the banks that should have appeared in *borrowings*, and instead was presented within *other creditors*. The joint report of the BEIS and Work and Pensions committees (**HC 769**, published 16 May 2018) records Moody's putting the misclassification at as much as **£498 million**. Carillion's 2016 balance sheet showed **£148 million** of bank loans and overdrafts, with that additional sum owed to banks excluded from the caption. Reported net borrowing at 31 December 2016 was **£218.9 million**.

Set those numbers next to each other. Reported net borrowing of £218.9 million at the balance sheet date; up to £498 million of bank money sitting in a working-capital line.

And the year-end figure was itself the flattering one. Carillion's own 2016 annual report puts **average net borrowing for the year at £586.5 million** — roughly **2.7 times** the balance-sheet number. That pairing is the real story, and you can read it without forming any view on the reverse factoring at all: a company whose year-end debt is just over a third of its average debt is telling you that the reporting date was managed. The two facts compound. The snapshot understated the average, and the caption understated the snapshot.

Then the sequence that always ends these stories. In **July 2017** Carillion announced an expected contract provision of **£845 million** as at 30 June 2017 — £375 million relating to the UK, largely three PPP projects, and £470 million to overseas markets. The chief executive, Richard Howson, and the finance director, Zafar Khan, were replaced. In **December 2017**, Santander withdrew the early payment facility. Four to six weeks later, Carillion was in liquidation.

The withdrawal is the mechanism. A company financing several hundred million pounds of payables through a facility it did not control lost that facility at the exact moment it could least afford the working-capital snapback. The reported leverage never described the company that actually existed.

Every number in this section that makes the point became public *after* the company died. The £498 million estimate arrived in the parliamentary report of May 2018; Fitch announced it would treat reverse-factoring-driven extensions of payment terms as debt in August 2018, seven months after the liquidation. The adjustment was correct and it was available to anyone who computed DPO from the primary statements in 2015. It simply was not made in time by the people whose job it was.

### Greensill: what happens when the financier is the fragile one

Carillion showed the risk sitting with the borrower. Greensill Capital showed that it can sit with the funder too.

Greensill was a specialist supply-chain finance firm that packaged the receivables it financed into notes sold to investors — most prominently through a group of funds run by **Credit Suisse Asset Management** and a fund at **GAM**. It raised roughly **\$1.5 billion** from SoftBank's Vision Fund during **2019** (reported as an \$800 million investment in May and a \$655 million follow-on in October), at a reported valuation of around \$7 billion.

The structure had two brittle points. First, concentration: a very large share of the exposure related to a single client group, **GFG Alliance**, the metals empire associated with Sanjeev Gupta. Second, and more fundamental, a portion of the assets were reportedly not conventional approved payables at all but *prospective* receivables — expected future invoices from counterparties the client had not yet transacted with. A supply-chain finance asset is supposed to be a short, self-liquidating claim on an approved invoice; a claim on an invoice that does not exist yet is a very different instrument wearing the same name.

The unwind was fast. On **1 March 2021**, Credit Suisse Asset Management gated a group of supply-chain finance funds holding roughly **\$10 billion** and moved to wind them up; GAM closed its **\$842 million** Greensill-linked fund days later. On **3 March 2021**, Germany's financial regulator **BaFin** imposed a moratorium on Greensill Bank AG and filed a criminal complaint alleging balance-sheet manipulation. On **8 March 2021**, Greensill Capital filed for administration, with Grant Thornton appointed. GFG defaulted. Credit Suisse subsequently reported recovering around **\$7.0 billion**, roughly 70%, for fund investors, and after UBS acquired Credit Suisse it offered supply-chain fund investors 90% of net asset value.

The forensic lesson is about *whose* fragility you are exposed to. A company relying on an SCF programme is not only exposed to its own credit; it is exposed to the funding model of the entity providing the facility. If that funder is itself financed by redeemable investor money in open-ended funds, the facility can vanish for reasons that have nothing to do with the borrower at all.

### Abengoa: the word to search was "confirming"

Abengoa was a Spanish engineering and renewable-energy group that ran a large-scale reverse factoring programme — in Spanish market usage, *confirming*. In late 2015, **Moody's** said the programme had "debt-like" features and announced a review of its rating methodology, prompting a public argument with the trade finance industry.

Abengoa is the case that teaches the vocabulary problem. Search an English filing for "reverse factoring" and you may find nothing; search for the local-market term and the programme is right there. The same applies across jurisdictions and across euphemism: "early payment," "structured payables," "supplier enablement," "working capital optimisation programme." If your search terms are wrong, the absence of a hit means nothing.

Abengoa's arrangements also combined with heavy project-level debt structuring that kept obligations off the consolidated group figure — the same instinct applied at two different layers of the balance sheet. It entered restructuring proceedings and became one of Spain's largest corporate insolvencies.

### NMC Health: the extreme case of unrecorded obligations

NMC Health was a FTSE 100 hospital operator based in Abu Dhabi. On **17 December 2019**, the short-seller Muddy Waters published a report alleging that NMC had overstated its cash, overpaid for assets, and understated its debt.

What followed exceeded the allegation. NMC ultimately disclosed debt of about **\$6.6 billion**, against the **\$2.1 billion** reported in its 2018 financial statements — a gap of roughly \$4.5 billion between the two disclosed figures. It was placed into administration on **9 April 2020**. The UK's **Financial Conduct Authority** later censured NMC Health Plc (in administration) for market abuse, finding that its published financial statements had misled investors by understating its debts by as much as **\$4 billion**. (The two numbers measure slightly different things: the \$4.5 billion is the difference between two disclosed debt totals, while the FCA's \$4 billion is its own finding on the understatement in the published statements. Both are worth quoting with their source attached.) The FCA's findings describe records that did not reflect real obligations, with intra-group transactions and undisclosed borrowings falling outside the reported figures. Its administrators, Alvarez & Marsal, brought a reported £2 billion negligence claim against the auditor, EY.

NMC is not a pure supply-chain-finance story, and it should not be filed as one — the misstatement was far broader. It belongs here as the boundary case: the same analytical instinct that finds a reverse-factoring programme also finds this. Both ask *what obligations exist that are not in the debt schedule?* The difference is that supply-chain finance answers that question with a real obligation put in the wrong place, and NMC answers it with obligations that were, per the regulator, not recorded at all. The related-party dimension of that story is developed further in [related-party transactions and self-dealing](/blog/trading/forensic-accounting/related-party-transactions-and-self-dealing).

### Lucent and the telecom vendor-financing cycle

For the third arrangement, the canonical case is the telecom equipment bubble of 1998–2001, and Lucent Technologies is the best documented.

At the end of fiscal 2000, Lucent had entered into agreements to provide customers with up to **\$8.1 billion** in credit or loan guarantees, of which almost **\$2.1 billion** was outstanding. Those commitments then fell — to roughly \$7.5 billion at 31 December 2000 and about \$6.9 billion by 31 March 2001 — as the market turned. Lucent recorded bad-debt provisions of **\$2.2 billion** in 2001 and **\$1.3 billion** in 2002. Its revenue fell from roughly \$30 billion to about \$12 billion over the following two years. Nortel and Cisco were reported to have run comparable programmes on the order of \$3.1 billion committed and \$2.4 billion of customer loans respectively.

The dynamic is worth naming precisely, because it recurs in every capital-equipment boom. Equipment makers lent money to customers who could not otherwise buy; those customers bought; the equipment makers booked revenue and profit; analysts extrapolated the growth; the customers' own funding dried up; the loans defaulted; and the revenue that had been booked came back as provisions. The revenue was not fictitious in an accounting sense — real equipment was really delivered. It was simply financed by the seller, which means it measured the seller's willingness to lend rather than the market's willingness to buy.

### Boeing: the disclosed, benign case

It is important to end the case studies with one where the answer is "this is fine," because otherwise the pattern-matching becomes indiscriminate.

Boeing runs supply chain financing programmes and, under ASU 2022-04, discloses them properly. Its FY2024 Form 10-K reports accounts payable to participating suppliers moving from **\$2,871 million** at the start of 2024 to **\$2,703 million** at 31 December 2024, with **\$12,476 million** of additions and **\$12,644 million** of reductions for payments made. The note describes the mechanics plainly — participating suppliers may elect to obtain payment from an intermediary, and Boeing confirms the validity of the invoices and agrees to pay the intermediary — states that the majority of amounts payable are due within **30 to 90 days** though some may extend up to 12 months, and adds that Boeing does not believe future changes in the availability of supply chain financing would significantly affect its liquidity.

That is a well-behaved programme, and the disclosure lets you verify it rather than take it on trust. The balance turned over roughly 4.6 times, implying an average life around 79 days — consistent with the stated 30-to-90-day terms rather than with a disguised term borrowing. The balance *fell* year on year, so the programme was a modest drag on 2024 operating cash flow rather than a boost. Two things still deserve your attention: the 12-month tail on some payables, and the liquidity assertion, which is exactly the claim Carillion's accounts implied and could not honour. You may still choose to add \$2.7 billion to debt in your own model — many credit analysts do — but you are making that adjustment with the number in front of you, which is the entire point.

## When this matters to you

If you invest in, lend to, or work at a company with a large payables balance, three habits pay for themselves.

**Compute DPO yourself, every year, from the primary statements.** Do not take a data provider's figure, because providers vary in whether they use COGS or revenue, and year-end or average balances. Plot six years. The shape tells you more than the level.

**Read the payables note and the liquidity section of the MD&A before the income statement.** The order matters. If you read earnings first you will spend your attention explaining the profit; the obligations that will actually determine whether the company survives are two hundred pages further back.

**Ask the withdrawal question about every facility, not just this one.** Undrawn revolvers, receivables purchase programmes, factoring lines, and supply-chain finance programmes all share the property that they are most likely to be cancelled exactly when they are most needed. A liquidity position that depends on facilities the company does not control is not a liquidity position; it is a hope with a spreadsheet attached.

And if you supply a large customer that offers you an early-payment programme: run the arithmetic in the worked example above against your *current* terms, not against the longer terms you are being offered alongside it. The programme may still be worth joining. But you should know which of the two numbers you are comparing.

This is educational material about how financial statements work, not advice about any particular security or company.

## Sources & further reading

**Standards and interpretations**

- FASB, *Accounting Standards Update No. 2022-04, Liabilities — Supplier Finance Programs (Subtopic 405-50)*, issued 29 September 2022. Effective for fiscal years beginning after 15 December 2022; rollforward requirement effective for fiscal years beginning after 15 December 2023. Summarised at [Journal of Accountancy](https://www.journalofaccountancy.com/news/2022/sep/fasb-updates-reporting-standard-supplier-finance-programs/) and [RSM US](https://rsmus.com/insights/financial-reporting/enhanced-disclosures-for-supplier-finance-program-obligations.html).
- IASB, *Supplier Finance Arrangements — Amendments to IAS 7 and IFRS 7*, May 2023. Effective for annual reporting periods beginning on or after 1 January 2024. Project page: [IFRS Foundation](https://www.ifrs.org/projects/completed-projects/2023/supplier-finance-arrangements/). Endorsement in the UK: [FRC / UK Endorsement Board, December 2023](https://www.frc.org.uk/news-and-events/news/2023/12/ukeb-adopts-supplier-finance-arrangements-amendments-to-ias-7-statement-of-cash-flows-and-ifrs-7-financial-instruments-disclosures/).
- IFRS Interpretations Committee, agenda decision on *Supply Chain Financing Arrangements — Reverse Factoring*, December 2020.
- EY, [IASB amendments to IAS 7 and IFRS 7 for supplier finance arrangements](https://www.ey.com/en_gl/technical/ifrs-technical-resources/iasb-amendments-to-ias-7-and-ifrs-7-for-supplier-finance-arrangements).

**Carillion**

- UK Parliament, Work and Pensions and BEIS Committees, *Carillion* (HC 769, published 16 May 2018) — the Moody's and S&P arguments on the Early Payment Facility, and the £498 million estimate: [full report](https://publications.parliament.uk/pa/cm201719/cmselect/cmworpen/769/769.pdf) and [committee summary](https://committees.parliament.uk/committee/164/work-and-pensions-committee/news/97957/carillion-used-its-suppliers-to-prop-up-failing-business-model/).
- The Construction Index, [Carillion hid debt implications of reverse factoring](https://www.theconstructionindex.co.uk/news/view/carillion-hid-debt-burden-of-its-reverse-factoring) — the Santander evidence published by the committees: the Moody's and S&P argument that the EPF concealed borrowing from financial creditors, Moody's £498 million estimate, the 120-day terms imposed on suppliers who did not join, and Santander's withdrawal of the facility in December 2017.
- Building, [Carillion used early payment facility to hide £500m debt](https://www.building.co.uk/news/carillion-used-early-payment-facility-to-hide-500m-debt/5093565.article).
- Construction Enquirer, [Carillion profit warning unearths £845m contract black hole](https://www.constructionenquirer.com/2017/07/10/carillion-profit-warning-unearths-845m-contract-black-hole/), 10 July 2017 — the provision of £845m at 30 June 2017, split £375m UK (majority three PPP projects) and £470m overseas; also reports first-half 2017 net borrowing of £695m against £586.5m throughout 2016.
- House of Commons Library, [Carillion collapse: what went wrong?](https://commonslibrary.parliament.uk/carillion-collapse-what-went-wrong/).

**Rating agencies**

- Global Trade Review, [Industry rejects Fitch's call to reclassify supply chain finance as debt](https://www.gtreview.com/news/global/industry-rejects-fitchs-call-to-reclassify-supply-chain-finance-as-debt/), 14 August 2018 — Fitch's £400–500 million estimate for Carillion against £219 million of reported net debt, its finding that median payables days peaked in 2017, and the industry response.
- Global Trade Review, [SCF market tackles impact of the Carillion effect](https://www.gtreview.com/magazine/volume-17-issue-2/scf-market-tackles-impact-carillion-effect/).

**Greensill**

- CNBC, [SoftBank-backed Greensill Capital files for insolvency](https://www.cnbc.com/2021/03/08/greensill-capital-has-reportedly-filed-for-administration.html), 8 March 2021.
- UK Parliament, Treasury Committee, *Lessons from Greensill Capital* (HC 151, 20 July 2021): [report](https://publications.parliament.uk/pa/cm5802/cmselect/cmtreasy/151/15102.htm).
- Crunchbase News, [Greensill Capital raises \$655M more from SoftBank Vision Fund](https://news.crunchbase.com/business/greensill-capital-raises-655m-more-from-softbank-vision-fund/), October 2019.
- SWI swissinfo, [Credit Suisse says Greensill recovery will cost clients \$291m](https://www.swissinfo.ch/eng/business/credit-suisse-says-greensill-recovery-will-cost-clients-291m/47764852) and [Credit Suisse angers investors with five-year "hard grind" on Greensill losses](https://www.swissinfo.ch/eng/business/credit-suisse-angers-investors-with-five-year-hard-grind-on-greensill-losses/47508042).

**NMC Health**

- Financial Conduct Authority, [FCA censures NMC Health Plc (in Administration) for market abuse](https://www.fca.org.uk/news/press-releases/fca-censures-nmc-health-plc-administration-market-abuse).
- CFA Institute Enterprising Investor, [The NMC Health debacle: four red flags?](https://rpc.cfainstitute.org/blogs/enterprising-investor/2020/the-nmc-health-debacle-four-red-flags), May 2020.

**Vendor financing**

- Lucent Technologies, Form 10-Q for the quarter ended 31 March 2001, [SEC EDGAR](https://www.sec.gov/Archives/edgar/data/0001006240/000095011701500276/a29665.txt) — the \$8.1 billion of commitments at 30 September 2000 and the subsequent reduction.
- Lazonick and March, [The rise and demise of Lucent Technologies](https://thebhc.org/sites/default/files/lazonickandmarch.pdf).

**Market size and current practice**

- FCI, [2025 world industry statistics: global factoring market surpasses €4 trillion](https://fci.nl/en/news/fci-releases-2025-world-industry-statistics-global-factoring-market-surpasses-eu4-trillion?language_content_entity=en) and [2024 world industry statistics](https://fci.nl/en/news/fci-release-2024-world-industry-statistics-showing-factoring-market-remains-stable?language_content_entity=en).
- BCR Publishing, [World Supply Chain Finance Report 2024](https://bcrpub.com/product/world-supply-chain-finance-report-2024/).
- Boeing, [Form 10-K for the year ended 31 December 2024](https://www.sec.gov/Archives/edgar/data/12927/000001292725000015/ba-20241231.htm), note *Supply Chain Financing Programs* — the rollforward, the 30-to-90-day terms, and the liquidity statement.
- The CPA Journal, [Supplier finance programs](https://www.cpajournal.com/2022/08/16/supplier-finance-programs/), August 2022 — background on the SEC comment letters to Keurig Dr Pepper, Masco, Procter & Gamble, Graphic Packaging, Boeing and Coca-Cola.

**Within this series**

- [The cash conversion cycle and what working capital reveals](/blog/trading/forensic-accounting/the-cash-conversion-cycle-and-what-working-capital-reveals)
- [Hidden liabilities: leases, guarantees, and contingencies](/blog/trading/forensic-accounting/hidden-liabilities-leases-guarantees-and-contingencies)
- [Off-balance-sheet financing and special purpose entities](/blog/trading/forensic-accounting/off-balance-sheet-financing-and-special-purpose-entities)
- [Reading the balance sheet: what companies hide here](/blog/trading/forensic-accounting/reading-the-balance-sheet-what-companies-hide-here)
- [Reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income)
- [The footnotes and MD&A: where the bodies are buried](/blog/trading/forensic-accounting/the-footnotes-and-mda-where-the-bodies-are-buried)
