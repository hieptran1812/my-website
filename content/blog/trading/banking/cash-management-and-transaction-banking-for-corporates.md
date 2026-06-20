---
title: "Cash Management and Transaction Banking for Corporates: The Quiet Business That Funds the Bank"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How banks run a company's day-to-day money — pooling, sweeping, payroll, collections and liquidity management — and why this fee-rich, capital-light business produces the cheap, sticky operating deposits a bank prizes above almost everything else."
tags: ["banking", "transaction-banking", "cash-management", "cash-pooling", "sweeping", "liquidity-management", "operating-deposits", "payments", "collections", "treasury", "working-capital"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Transaction banking is the unglamorous business of running a company's day-to-day money — paying its staff, collecting its invoices, pooling and sweeping its cash — and it is one of the best businesses a bank owns because it earns steady fees, ties up almost no capital, is hard for a rival to dislodge, and quietly produces the cheapest, stickiest deposits the bank funds itself with.
>
> - **Pooling and sweeping** net a company's scattered account balances into one position, so it stops paying overdraft interest on cash it already owns somewhere else — a saving the bank shares in.
> - **The real prize is the deposits.** A corporate's operating cash sits in low-rate accounts the bank uses as funding. Shift funding toward these and a bank's blended cost of funds can fall by **100 basis points** (a *basis point* is one hundredth of a percent — 0.01%).
> - **It is capital-light and sticky.** The business earns fees off a thin balance sheet, so it ties up a fraction of the capital that lending does, and once a company plugs its payroll and collections into your systems, it almost never leaves.
> - **The one number to remember:** deposits fund roughly **71%** of a typical bank's balance sheet — and transaction banking is the machine that wins the cheapest slice of that 71%.

Every two weeks, a few hundred million dollars moves through a bank without anyone celebrating. It is payroll. A company tells its bank, "pay these 40,000 people on Friday," and on Friday they are paid. No deal is announced, no trader high-fives, no headline runs. It is the single most important thing a bank does for that company — and one of the most profitable things the bank does, period.

This is transaction banking, also called cash management or corporate cash management: the business of running the everyday plumbing of a company's money. It is the opposite of the parts of banking that make the news. There is no leverage drama, no rogue trader, no record net interest margin print. There is a treasurer in a fluorescent-lit office trying to make sure the company can pay its suppliers, collect from its customers, and not leave a single dollar sitting idle overnight. The bank's job is to make that easy — and to get paid, in fees and in deposits, for doing it.

It turns out this quiet corner is one of the crown jewels of a modern bank. It throws off recurring fee income, it consumes almost no capital, customers are sticky to the point of stubbornness, and — the part that ties straight back to this whole series' spine — it generates the cheapest deposits a bank can find. Remember the thesis: a bank is a leveraged, confidence-funded maturity-transformation machine that borrows short and lends long and lives on the spread. Transaction banking is how a bank wins the *short* side of that trade on the best possible terms. The diagram above is the mental model: a company's many scattered account balances rolling up into one pooled position the bank can fund itself with.

![Cash pooling diagram many account balances roll up into one header account](/imgs/blogs/cash-management-and-transaction-banking-for-corporates-1.png)

## Foundations: what cash management actually is

Let's build this from zero, because the jargon hides a very simple set of ideas. Forget banks for a second and think about your own household money.

You have a checking account for bills, a savings account for the emergency fund, and maybe a credit card. Your "cash management" is the boring discipline of keeping enough in checking to cover the bills, parking the rest where it earns a little, and not paying overdraft fees because you forgot to move money across in time. You do this with your fingers in a banking app once a week.

Now make yourself a company. Instead of three accounts you have three hundred — one per country, one per subsidiary, one per currency, one for collecting customer payments, one for paying suppliers. Instead of a handful of bills you have payroll for tens of thousands of people, thousands of supplier invoices, tax payments, and a stream of incoming customer money. And instead of a few hundred dollars you have hundreds of millions. The household problem hasn't changed shape — keep enough where you need it, don't leave cash idle, don't pay to borrow money you already own somewhere else — but the scale makes it impossible to do by hand. **Cash management is the set of bank services that automate that problem at corporate scale.**

Here are the terms you need, each defined on first use. I'll use every one of these again, so it's worth pinning them down now.

- **Operating account (or operating deposit):** the everyday account a company runs its business through — receiving sales, paying staff and suppliers. The balance bounces around but rarely hits zero, because the business is always moving money. From the bank's side, an *operating deposit* is the cash sitting in these accounts, and it is the prize of this whole post.
- **Cash pooling:** combining the balances of many accounts into one position so the company sees, funds, and invests its cash as a single pot rather than account by account.
- **Notional pooling vs physical pooling:** two ways to do that. In *physical pooling*, the bank actually moves the money between accounts. In *notional pooling*, the money stays put but the bank treats the balances *as if* they were combined for the purpose of calculating interest. We'll go deep on the difference — it's the single most important distinction in this post.
- **Sweeping:** automatically moving cash from one account to another when a rule fires — for example, "every evening, move everything above zero in the operating accounts into one concentration account." A *sweep* is the mechanical action; *physical pooling* is sweeping done as a standing arrangement.
- **Value date:** the date on which money actually counts as yours for the purpose of earning (or paying) interest, as opposed to the date it shows up on the screen. If a payment lands today but the *value date* is tomorrow, you don't earn interest on it today. Banks used to make real money on the gap; modern systems have squeezed it, but value-dating still matters.
- **The cash conversion cycle:** the number of days between a company spending cash (to buy inventory, to run operations) and getting cash back (when customers pay). The longer the cycle, the more cash is tied up in the business and unavailable. Cash management's whole purpose is to shorten it. Formally, it's days-inventory-outstanding plus days-sales-outstanding minus days-payable-outstanding — how long you wait to sell, plus how long customers take to pay you, minus how long you take to pay your suppliers.

It's worth dwelling on the cash conversion cycle for a moment, because it is the master variable cash management exists to bend. Think of a simple business: it spends \$100 to buy inventory on day 0, sits on that inventory for 30 days before selling it, then waits another 45 days for the customer to pay. Its suppliers, meanwhile, gave it 20 days to pay them. The cash is *out the door* from day 20 (when it pays the supplier) until day 75 (when the customer pays) — 55 days during which the business has spent cash it hasn't gotten back. That 55-day gap has to be funded by something: the company's own cash, or a loan, both of which cost money. Multiply it across millions of dollars of daily activity and the cash conversion cycle becomes one of the largest, most controllable costs a company has. Every lever in this post — pool the idle balances, sweep the surplus to earn overnight, collect from customers faster, pay suppliers no sooner than you must — is an attack on some piece of that gap. A company that runs a 55-day cycle and gets it down to 45 days has, in effect, freed up ten days' worth of its spending as cash it can now use elsewhere.

One more foundational idea, because it is the punchline the rest of the post keeps returning to. To a depositor, a deposit is money you've parked at the bank. To the bank, that same deposit is **funding** — it's borrowed money the bank uses to make loans and buy securities. (If that inverse-perspective trick is new to you, the [bank balance sheet from the ground up](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity) post walks through why a deposit is the bank's liability and your asset at the same time.) Not all funding costs the same. Cash a company leaves in a low-rate operating account is dirt-cheap funding. Money the bank has to borrow on the wholesale market is expensive funding. Transaction banking is the machine that maximizes the cheap kind. Hold that thought; we'll quantify it with real percentages later.

## Why pooling exists: stop paying to borrow what you already own

Start with the most intuitive corner of the business, because it makes the whole logic click.

Picture a company — call it a manufacturer with operations across Europe. Its French unit is sitting on a credit balance of €4 million it doesn't need this week. Its German unit has €9 million parked. Its Italian unit has €1 million. And its Spanish unit is overdrawn by €6 million — it's short of cash this week and is borrowing on its overdraft line to make payroll.

Look at that as four separate accounts and something absurd is happening. The company as a whole has plenty of cash — €4 + €9 + €1 − €6 = **€8 million net positive**. And yet one of its own units is *borrowing money at a steep overdraft rate* while the others let cash sit idle earning almost nothing. The company is paying to borrow money it already owns. That's the waste pooling exists to kill.

![Before and after pooling separate accounts versus one pooled position](/imgs/blogs/cash-management-and-transaction-banking-for-corporates-2.png)

When the bank pools these accounts, the balances are treated as one. The Spanish overdraft is now funded from inside the group — by the French, German, and Italian surpluses — instead of from the bank's expensive overdraft line. The company stops paying overdraft interest on the part of its borrowing that its own cash could cover. The only thing that should ever cost real interest is the company's *true net* position, not the noise of which subsidiary happens to be flush this week.

#### Worked example: cash pooling saving interest

Let's put real numbers on it. Suppose the company's overdraft costs **9% a year**, and idle credit balances earn just **0.5% a year**. The Spanish unit is overdrawn by \$6 million; the rest of the group has more than \$6 million in idle credit. That \$6 million is the *overlap* — money that's being borrowed in one place and sitting idle in another at the same time.

Without pooling, the company pays the overdraft rate on the full \$6 million:

\$6,000,000 × 9% = **\$540,000 a year** in overdraft interest.

Meanwhile the matching \$6 million of idle credit elsewhere earns:

\$6,000,000 × 0.5% = \$30,000 a year.

So the *net* drag from running these as separate accounts is \$540,000 − \$30,000 = **\$510,000 a year**, paid for nothing — pure friction from not netting. With pooling, the \$6 million overlap nets out: the overdraft is funded internally, so the company stops paying 9% on it and stops earning the measly 0.5% on the matching idle cash. The saving is the *full spread* — 9% − 0.5% = **8.5%** — on the \$6 million overlap:

\$6,000,000 × 8.5% = **\$510,000 a year saved**.

The intuition: pooling doesn't create cash, it stops the company from paying the borrow-rate-minus-lend-rate spread on money it was lending to itself the hard way.

![Bar chart interest saved per year by pooling overlapping balances](/imgs/blogs/cash-management-and-transaction-banking-for-corporates-3.png)

The chart shows how that saving scales with how much overlapping balance the company centralizes — \$170k on a \$2M overlap, \$510k on \$6M, more than \$1M once you net \$12M. The more accounts a multinational pools, the bigger the overlap and the bigger the prize. This is why the world's largest companies run global pooling structures across dozens of countries: at their scale, the netting saves tens of millions a year. And the bank, having built and run that structure, charges a fee for it and keeps the company's pooled cash as deposits. Both sides win, which is exactly why the business is so durable.

## Notional vs physical pooling: same goal, very different mechanics

This is the distinction that separates someone who *says* "cash pooling" from someone who understands it. Both methods get you to "the group is funded as one position," but they do it in opposite ways, and the differences in tax, legal exposure, and regulation are enormous.

**Physical pooling** moves real money. At the end of each day (or on a trigger), the bank sweeps cash out of each subsidiary's account into a single *concentration account* (often called a *header account* or *master account*) — and refills the accounts the next morning for the day's payments. When the French unit's €4 million physically moves into the header account, an accounting reality is created: the French unit has effectively *lent* €4 million to the entity that owns the header account. That's a real intercompany loan, with all the tax, transfer-pricing, and interest-allocation paperwork a loan implies. The upside is that physical pooling is simple to understand, works across legal entities cleanly when documented, and is accepted everywhere.

**Notional pooling** moves nothing. The cash stays in each account exactly where it is. The bank simply *computes interest as if* the balances were combined — it offsets the credit balances against the debit balances for interest purposes and charges or pays interest on the net. The French unit keeps its €4 million on its own books; no intercompany loan is created; the legal autonomy of each subsidiary is preserved. That's a big deal for companies that don't want subsidiaries lending to each other, or that operate where intercompany loans trigger taxes or withholding.

The catch with notional pooling is that the bank is the one taking on risk. Because the cash never moves, the bank is effectively offsetting one customer's overdraft against another customer's credit balance *on its own balance sheet*. If a regulator says "you can't net those — you must hold capital against the gross overdraft and separately recognize the gross deposit," the trick gets expensive for the bank. Post-2008 rules (the leverage ratio in particular, which we'll touch on) made notional pooling more capital-costly for banks, which is why some pulled back from offering it, especially across borders.

| | Physical pooling | Notional pooling |
|---|---|---|
| Does cash move? | Yes — swept to a header account | No — balances stay put |
| Creates intercompany loans? | Yes (a real lending relationship) | No (legal autonomy preserved) |
| Who carries the offset risk? | The company (it owns the loans) | The bank (it nets on its own book) |
| Tax / transfer-pricing burden | Higher (interest must be allocated) | Lower (no internal loans) |
| Capital cost to the bank | Lower | Higher (post-Basel netting rules) |
| Cross-border friction | More documentation | Often legally harder; some jurisdictions ban it |

#### Worked example: a sweep reducing idle balances

Let's see physical pooling in action over one night. The company's four accounts close the day at: France +€4M, Germany +€9M, Italy +€1M, Spain −€6M. Run a *zero-balance sweep* — a standing rule that drains every operating account to exactly zero each evening and concentrates the cash in the header account.

After the sweep, every operating account reads **€0**, and the header account holds the net: €4M + €9M + €1M − €6M = **€8M**. Instead of €14M of credit balances sitting idle across three accounts while €6M is borrowed in a fourth, there is a single €8M surplus the treasurer can put to work overnight.

Suppose she invests that €8M overnight at **3%**. One night's earnings:

€8,000,000 × 3% × (1 / 360) = €8,000,000 × 0.00833% ≈ **€667 for one night**.

That sounds small, but it happens 250 business days a year, and the same sweep eliminated the €6M overdraft that was bleeding 9%. Over a year, the sweep is the difference between cash that works and cash that sleeps. The intuition: a sweep is how you make sure no euro ever spends a night idle when it could be earning.

![Pipeline physical zero balance sweep at end of day](/imgs/blogs/cash-management-and-transaction-banking-for-corporates-7.png)

The figure traces the nightly mechanics: cut-off time fires the sweep rule, real cash drains to zero, concentrates in one account, gets invested overnight, and the accounts refill at dawn. This is the heartbeat of corporate treasury. The treasurer's screen shows one net number to manage instead of three hundred, and the bank runs the whole choreography automatically — for a fee, and with the company's cash flowing through its systems the entire time.

One nuance worth pinning down here is the difference between a **target-balance sweep** and a **zero-balance sweep**. A zero-balance sweep drains the account to exactly zero each night, as we just walked through — clean, but it requires the account to be refilled before the next morning's payments leave. A target-balance sweep instead leaves a chosen buffer behind — say, "keep €500,000 in each operating account, sweep only the excess" — so each account always has enough to cover the day's small payments without waiting for a refill. Treasurers tune this constantly: too low a target and a large payment can fail or trip an overdraft; too high and cash sits idle, defeating the point. The bank's systems let the treasurer set these rules per account and per day-of-week (payroll Fridays need a fatter buffer than a quiet Tuesday), and that fine-grained control is itself part of what the company is paying for.

### Value date and the intraday liquidity problem

There is a subtle timing layer beneath all of this that beginners miss, and it's where the *value date* concept earns its keep. Money has two clocks: when it *appears* on the screen (the booking date) and when it actually starts *earning or costing* interest (the value date). A payment received today with a value date of tomorrow shows in the balance today but doesn't earn until tomorrow. Historically banks profited from the gap — crediting your account a day or two after they really had the money, and pocketing the interest in between. That's the old *float* business, and it's largely been competed and regulated away on domestic rails. But the value-date concept survives because it governs *intraday liquidity* — the bank's minute-by-minute cash position.

Here's why that matters even to a company. A bank must have cash *in the right account at the right moment* to settle the payments it's making on a corporate's behalf. If a company instructs a \$200 million payment to go out at 10 a.m. but the incoming \$200 million it's relying on doesn't arrive until 3 p.m., the bank has to bridge that intraday gap — fronting the cash for five hours. Banks manage this with intraday credit lines and careful scheduling, and large corporates with predictable, well-timed flows are cheaper and easier to serve than ones whose money sloshes unpredictably. Good cash management, then, isn't only about the end-of-day position; it's about smoothing the *intraday* shape of a company's flows so neither the company nor its bank gets caught short mid-morning. The treasurer who can tell the bank "my big inflows land by 9 a.m., my big outflows go after 2 p.m." gets better terms than the one whose timing is a mystery.

## Payroll and payments: the steady annuity at the heart of the relationship

Return to the moment this post opened with: payroll. It is the most emotionally load-bearing payment a company makes — get it wrong and tens of thousands of people don't get paid, and the reputational damage is instant — and it is the perfect lens on how the payments half of transaction banking works.

A payroll run is a *bulk* or *batch* payment: instead of issuing 40,000 separate transfers by hand, the company's payroll system produces one file listing 40,000 beneficiaries and amounts, hands it to the bank, and the bank disburses every payment on the value date. The rails it uses depend on size and urgency. Most payroll goes over a *net batch* system (in the US, the ACH network; in Europe, SEPA credit transfers) — cheap, processed in bulk, settled net at the end of a cycle rather than one-by-one. A handful of urgent or very large payments — a supplier that must be paid same-day, a tax deadline — go over a *real-time gross settlement* (RTGS) rail, where each payment settles individually and irrevocably in central-bank money. The trade-off is exactly what you'd expect: batch rails are pennies per payment but slower and revocable until they clear; RTGS is final in seconds but costs far more per payment. The deep mechanics of these rails — what RTGS, ACH, and instant payments each guarantee — are their own subject, but the cash-management point is that the bank routes each payment over the *cheapest rail that meets the deadline*, and that routing intelligence is part of the service.

#### Worked example: the fee-plus-deposit value of a payroll mandate

Let's price what a single corporate mandate is actually worth to the bank, because this is where the "fees are the appetizer, deposits are the meal" claim becomes arithmetic.

Suppose a company runs payroll for 40,000 staff twice a month — that's 40,000 × 2 × 12 = **960,000 payroll payments a year** — plus roughly **240,000 supplier and tax payments**, for about **1.2 million payments a year**. The bank charges an average of **\$0.20 per payment** (batch payments are cheaper, the occasional RTGS payment pricier; \$0.20 is a blended figure). Annual *fee* income:

1,200,000 × \$0.20 = **\$240,000 a year** in payment fees.

Respectable, but not exciting on its own. Now the deposits. To run all this, the company keeps an average operating balance with the bank of, say, **\$80 million** — payroll floats, collections land, suppliers get paid out of it, and the balance never empties. The bank funds itself with that \$80 million at the operating-deposit rate of **0.5%**, when its alternative — borrowing the same \$80 million on the wholesale market — would cost **5.2%**. The funding *value* of those deposits is the spread the bank avoids paying:

\$80,000,000 × (5.2% − 0.5%) = \$80,000,000 × 4.7% = **\$3.76 million a year**.

So the deposit value of the mandate is more than **fifteen times** the fee income. The intuition: when a bank fights for a corporate's payroll, it is not really fighting for \$240,000 of fees — it is fighting for \$80 million of the cheapest funding on earth, and that's why it will sharpen its pencil on the fees to win the cash.

This is also why payments and deposits are sold as one inseparable bundle. The fee line keeps the relationship operationally active and gives a clean reason to be plumbed into the company's systems; the deposits are the financial prize that justifies the whole effort. A bank that tried to win the deposits without running the payments would have no operational hook, and the cash would be flighty. A bank that ran the payments but waved the deposits off to a rival would be leaving the meal on the table. The two only make sense together.

## The order-to-cash cycle: getting paid faster is free money

Pooling is about cash you already have. The other half of cash management is about cash you're *waiting* for — and speeding that up is some of the cheapest money a company can make.

Go back to the cash conversion cycle. The most painful piece for most companies is *days sales outstanding* (DSO) — the time between sending an invoice and the customer's payment actually clearing. If you sell on "net 30" terms but customers really pay in 50 days, you are financing your customers' purchases for 50 days out of your own pocket. Every day you cut from that cycle is a day of working capital handed back to you, free.

![Pipeline the order to cash cycle the bank helps speed up](/imgs/blogs/cash-management-and-transaction-banking-for-corporates-4.png)

This is where bank collections products earn their keep. The classic problem: a customer pays, the money lands in the company's account, but nobody knows *which* invoice it was for. So an accounts-receivable clerk spends hours matching payments to invoices by hand, and until that match is done, the cash can't be recognized as collected. The bank's fix is **virtual accounts** — the company assigns each customer a unique virtual account number to pay into. The money still concentrates into one real account, but because each payer used a distinct virtual number, the bank can tell the company *exactly* who paid, automatically. Reconciliation that took a clerk a day now happens in seconds. Other collections tools — direct debits (pulling payment from the customer on the due date rather than waiting for them to push it), lockboxes for paper checks, request-to-pay messages — all attack the same enemy: the days between "you owe me" and "I have your money."

#### Worked example: the working-capital value of cutting DSO

Suppose the company does **\$3 billion** in annual sales and currently runs **45 days** of DSO. The cash tied up in unpaid invoices at any moment is:

(\$3,000,000,000 / 365) × 45 days = \$8,219,178 × 45 = **\$369.9 million** locked up in receivables.

Now the bank's virtual-accounts and direct-debit setup shaves DSO from 45 days to **40 days**. The cash now tied up:

\$8,219,178 × 40 = **\$328.8 million**.

The difference — \$369.9M − \$328.8M = **\$41.1 million** — is cash that used to be stuck in receivables and is now back in the company's hands. If that cash lets the company pay down debt costing **6%**, the annual saving is:

\$41,100,000 × 6% = **\$2.47 million a year**, from cutting five days off how long it waits to get paid.

The intuition: faster collection isn't a rounding error — at corporate scale, a handful of days is millions of dollars of working capital, and that's why companies pay banks well to compress the cash conversion cycle.

## The full product suite: it's a bundle, not a product

Cash management rarely sells alone. A bank wins a corporate's "transaction banking mandate" — the whole bundle of day-to-day money services — and that bundle is the real franchise. Once a company runs its payments, collections, liquidity, FX, and trade through one bank, switching means re-plumbing the entire finance department. That's the stickiness the bank is buying.

![Matrix the transaction banking product suite](/imgs/blogs/cash-management-and-transaction-banking-for-corporates-5.png)

Walk the five families:

- **Payments.** Paying staff (payroll), suppliers, taxes, dividends — across rails from same-day high-value transfers to bulk batch files of thousands of payments. The bank charges a small fee per payment and earns a sliver of *float* (interest on money in transit). For the deep mechanics of how those payments actually move between banks — clearing, settlement, nostro and vostro accounts — see [the payments business](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks); here the point is that the corporate's payment volume is a steady fee annuity.
- **Collections.** Virtual accounts, direct debits, lockboxes — everything that gets the company paid faster and reconciled automatically. The fee is modest, but the collected cash lands as deposits.
- **Liquidity management.** Pooling and sweeping — the netting and concentration we just walked through. Fee plus, crucially, the pooled cash sits with the bank as deposits.
- **FX and cards.** When payments cross currencies, the bank earns a spread on every conversion. Corporate cards and virtual cards add purchasing controls and another fee stream.
- **Trade finance.** Letters of credit, guarantees, supply-chain finance — the de-risking of buying and selling, especially across borders. It's a close cousin of cash management and often sold in the same conversation; the mechanics get their own deep dive in [trade finance: letters of credit, guarantees and supply-chain finance](/blog/trading/banking/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance).

Notice the pattern down the right-hand column of the matrix: nearly every product earns a fee *and* leaves cash with the bank as deposits. That double benefit — fee income now, cheap funding alongside — is the engine of the whole business, and it's where we turn next.

## The real prize: cheap, sticky operating deposits

Here is the part that ties transaction banking straight back to the spine of this series. The fees are nice. The deposits are the point.

Recall the thesis: a bank borrows short and lends long and earns the spread. The cost of the "borrow short" leg — its **cost of funds** — is one of the two levers that set its profitability (the other being what it earns on assets). Anything that lowers the cost of funds widens the [net interest margin](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) — the spread between what a bank earns on assets and pays on funding — and drops almost straight to the bottom line.

A bank funds itself from many sources, and they don't cost the same:

- **Operating deposits** — a company's everyday cash, parked in low-rate or zero-rate accounts because the company needs it liquid, not because it's chasing yield. These cost the bank almost nothing — often well under 1%.
- **Rate-sensitive retail and term deposits** — savers shopping for yield, who move their money when a competitor pays more. These cost more and are flightier.
- **Wholesale funding** — money the bank borrows from other banks and markets (interbank loans, repo, bonds). This tracks market rates closely and is the most expensive marginal funding. The bigger picture of how a bank stacks these sources is the subject of [retail deposits and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise).

Operating deposits are the best of the three on every axis: they're the cheapest, and they're the *stickiest*, because the company keeps that cash with you precisely because its payroll and collections run through your systems. It's operationally locked in. A saver chasing yield will leave for 25 extra basis points; a corporate won't re-plumb its entire treasury to chase a slightly better rate on its operating cash.

![Stacked bar more operating deposits lower the blended cost of funds](/imgs/blogs/cash-management-and-transaction-banking-for-corporates-6.png)

#### Worked example: how operating deposits lower the bank's cost of funds

Take two banks with identical assets but different funding mixes. Assume operating deposits cost **0.5%**, rate-sensitive retail deposits cost **2.5%**, and wholesale funding costs **5.2%** (a 2024-ish rate environment).

**Bank A** funds itself (on the interest-bearing side) with 20% operating deposits, 45% retail deposits, and 35% wholesale. Its blended cost of funds is:

(0.20 × 0.5%) + (0.45 × 2.5%) + (0.35 × 5.2%) = 0.10% + 1.125% + 1.82% = **3.04%**.

**Bank B** has won a big transaction-banking book, so its mix is 45% operating deposits, 40% retail, and only 15% wholesale:

(0.45 × 0.5%) + (0.40 × 2.5%) + (0.15 × 5.2%) = 0.225% + 1.00% + 0.78% = **2.01%**.

Bank B's cost of funds is **103 basis points lower** than Bank A's — for doing nothing other than funding more of itself with corporate operating cash. On a \$100 billion balance sheet, that 1.03% saving is:

\$100,000,000,000 × 1.03% = **\$1.03 billion a year** of extra net interest income, straight to the bottom line.

The intuition: transaction banking doesn't just earn fees — it re-engineers the cheapest input on a bank's income statement, and at scale that's worth more than the fees ever are.

The chart makes the mechanism visible: the green operating-deposit slice is almost free, the red wholesale slice is expensive, and Bank B's taller green slice plus shorter red slice is the entire story. This is why every large bank fights for the corporate transaction-banking mandate even when the headline fees look unexciting. The fees are the appetizer; the cheap deposits are the meal. And it's why a bank that loses a major corporate's operating accounts feels it not just in fee income but in a more expensive, flightier funding base — exactly the kind of base that turns a rate shock into a crisis.

## Why this business is capital-light and sticky

Two more properties make transaction banking exceptional, and both flow from the same source: it's mostly a *service* business riding on a thin balance sheet, not a lending business riding on a fat one.

**Capital-light.** Recall from [bank capital and why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) that a bank must hold equity against its *risk-weighted assets* — broadly, the riskier and bigger the assets it holds, the more shareholder capital regulators force it to set aside. A loan is a giant risk-weighted asset: lend \$100 and you tie up real capital against the chance it isn't repaid. A trading position ties up capital against market swings. But a payment? Collecting an invoice? Running a pool? These earn fees while putting almost nothing risky on the balance sheet. The capital you must hold per dollar of revenue is a fraction of what lending demands.

![Bar chart transaction banking ties up the least capital per dollar earned](/imgs/blogs/cash-management-and-transaction-banking-for-corporates-8.png)

The chart is illustrative, but the ranking is real: corporate lending and the trading book tie up the most regulatory capital per dollar of revenue, mortgages somewhat less, and transaction banking the least by a wide margin. That matters enormously for the metric investors judge a bank on.

#### Worked example: the return on equity of a capital-light business

Recall the bank-return identity: return on equity = return on assets × leverage (the *leverage* being assets divided by equity). For a whole bank, the rule of thumb is roughly **1% ROA × ~12× leverage ≈ ~12% ROE**, as laid out in [ROE, ROA and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged).

Now isolate transaction banking. Say a transaction-banking unit earns **\$500 million** in annual revenue, and because it's capital-light it ties up only **\$1.0 billion** of equity capital (versus the multiple billions a lending book of the same revenue would demand). After costs, suppose it nets **\$200 million** of profit. Its return on the capital it consumes is:

\$200,000,000 / \$1,000,000,000 = **20% return on allocated equity**.

A lending business earning the same \$200 million profit might need \$2.5 billion of capital behind it — a return of \$200M / \$2.5B = **8%**. Same profit, but the capital-light business earns more than double the return on the scarce thing (equity). The intuition: in a world where regulators ration a bank's capital, the businesses that earn money *without* eating capital are the most valuable — and transaction banking sits at the top of that list.

**Sticky.** The second property is operational lock-in. When a company integrates the bank into its payroll, its supplier payments, its collections reconciliation, its treasury management system — it has woven the bank into its daily operations. Switching banks doesn't mean signing a new form; it means re-mapping every payment file, re-testing every direct-debit mandate, re-issuing virtual account numbers to thousands of customers, re-training the finance team, and praying nothing breaks during the cutover. The risk of a botched migration — a payroll run that fails, a payment that bounces a supplier — is so frightening that companies stay put for years even when a rival's pricing is keener. Stickiness means the revenue is *recurring* and the deposits are *durable*, which is exactly what makes both worth more.

## Host-to-host and API connectivity: the plumbing that locks it in

How does a company actually *talk* to the bank to move all this money? The answer is increasingly the reason the relationship is so hard to leave.

In the old world, a treasurer logged into the bank's web portal and uploaded payment files by hand. That works for small volumes but breaks at scale. So large corporates connect their own systems directly to the bank's, in two main flavors:

- **Host-to-host (H2H):** a direct, automated file pipe between the company's enterprise software (its ERP or treasury system) and the bank. Payment files flow out, statement files flow back, on a schedule, with no human clicking buttons. It's batch-oriented and rock-solid — the workhorse of corporate connectivity. Many large companies route this over **SWIFT** (the global bank-messaging network) so one connection reaches many banks; for the geopolitics and mechanics of SWIFT, see [SWIFT and the weaponization of payments](/blog/trading/finance/swift-and-the-weaponization-of-payments).
- **APIs (application programming interfaces):** the modern, real-time version. Instead of dropping a file and waiting for the next batch, the company's software *asks the bank directly, instantly*: "what's my balance right now?", "make this payment now and tell me when it's done." APIs enable real-time treasury — sweeping the moment a balance crosses a threshold, paying a supplier the instant goods are confirmed, embedding a payment inside another piece of software.

The standards matter here. Payment messages increasingly use a common format called **ISO 20022**, a structured, data-rich way of describing a payment that lets the company and the bank exchange not just "pay €1,000" but all the context — what it's for, which invoice, who the ultimate beneficiary is. Richer data means better automation: more payments reconcile themselves, fewer need a human.

The strategic point is that every layer of connectivity deepens the lock-in. A company that has built H2H pipes and API integrations into your bank has sunk real engineering cost into the relationship. That sunk cost is the moat. It's also why fintechs and "banking-as-a-service" providers are trying to slide into this layer — own the connectivity, own the relationship — and why incumbent banks defend it so fiercely. The connectivity is no longer plumbing; it's the franchise.

Consider what it actually takes to switch banks once you're deeply connected. The company has to rebuild every payment-file format to the new bank's specification, re-test the entire payroll cycle end to end (run it in parallel for a month or two so nothing breaks live), re-issue thousands of virtual account numbers to its customers so collections still reconcile, re-establish every direct-debit mandate, rewrite its API integrations, and re-certify the whole thing for security and compliance. A migration like that can take a year and tie up a chunk of the finance and IT teams, with a non-trivial chance that something fails in the cutover and a real payment goes wrong. Faced with that, a treasurer will tolerate a rival's slightly better pricing rather than risk the migration — the very definition of a switching cost. This is why transaction-banking revenue is described as "recurring": it's not that the company can't leave, it's that leaving is so painful that it usually doesn't, year after year. The flip side is that *winning* a new mandate from a competitor is equally hard, so banks invest heavily in making onboarding smooth — the bank that turns a year-long migration into a three-month one has a genuine competitive weapon.

This is also the terrain on which the next generation of competition is being fought. *Banking-as-a-service* lets a non-bank — a software company, a marketplace, a fintech — embed bank-grade payments and accounts inside its own product, with a licensed bank in the background. *Embedded finance* takes it further: the payment disappears into the software entirely, so a company pays a supplier from inside its accounting app without ever consciously "going to the bank." Whoever owns that embedded layer owns the relationship and, potentially, the deposits. Incumbent banks are racing to be the bank *behind* these experiences rather than be disintermediated by them — which is why the once-sleepy connectivity layer is now one of the most strategically charged parts of the whole institution.

## Common misconceptions

**"Transaction banking is a low-margin commodity — it's just moving money."** The per-payment fee is tiny, true. But the business isn't valued on fee margin; it's valued on the *deposits* it generates and the *capital* it doesn't consume. As the worked example showed, winning enough operating deposits to shift a \$100 billion bank's cost of funds by 1% is worth on the order of \$1 billion a year — dwarfing the fee line. Judging transaction banking by its fee margin is like judging a supermarket by the markup on milk while ignoring that the milk brings customers who fill a whole cart.

**"Pooling creates money or returns — it's a yield product."** No. Pooling creates *nothing*; it only stops a company from paying the borrow-versus-lend spread on cash it was effectively lending to itself. In the worked example the company saved \$510,000 — but that was waste it was paying *because* its accounts were fragmented, not new income. The benefit is real, but it's the removal of a friction, not the manufacture of a return. Mistaking the two leads treasurers to over-engineer pooling structures whose tax and legal costs exceed the netting they save.

**"Notional and physical pooling are basically the same thing."** They reach the same goal but through opposite mechanics, and the difference is load-bearing. Physical pooling moves cash and creates intercompany loans (with tax and transfer-pricing consequences for the company). Notional pooling moves nothing and keeps the legal autonomy of each subsidiary, but pushes the offset risk and capital cost onto the *bank* — which is why some banks scaled it back after post-2008 rules made netting more capital-expensive. Treating them as interchangeable gets the tax, legal, and pricing all wrong.

**"Operating deposits are just deposits — funding is funding."** Not all funding is equal, and the gap is the whole reason this business exists. Operating deposits are both the *cheapest* (often under 1% versus 5%+ for wholesale) and the *stickiest* (operationally locked in, versus rate-shoppers who leave for 25 bps). The 2023 failures drove this home: banks whose funding leaned on flighty, concentrated, uninsured deposits ran far harder than banks anchored by sticky operating relationships — see [the SVB and Credit Suisse runs of 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs). The *quality* of a deposit, not just the quantity, decides how a bank fares under stress.

**"Float — earning interest on money in transit — is where banks make their cash management profit."** Float was a real profit center decades ago, when value-dating gaps and slow clearing let banks hold customer money for days before it counted. Faster rails, instant payments, and tighter regulation have squeezed float to a sliver. Today the money is in fees and, far more, in the cheap deposits the relationship parks — not in skimming interest on payments mid-flight.

## Where it goes wrong: the risks behind the quiet business

It would be dishonest to sell transaction banking as pure upside. The reason it pays so well is that it carries real risks — they're just *operational* rather than the credit and market risks that dominate the rest of a bank. The danger of a quiet, automated business is precisely that it runs untouched until the day it doesn't, and then a single fault touches enormous volumes at once.

**Operational risk is the headline risk.** When a bank moves a corporate's payroll, supplier payments, and tax — millions of payments a year, automated end to end — a software bug, a mis-mapped file, or a botched system migration can fail or duplicate payments at scale. Real-world examples are sobering: in 2018, the UK's TSB bank bungled a core-systems migration and locked nearly two million customers out of their accounts for weeks, with payments failing and balances showing wrong; the episode cost the bank hundreds of millions of pounds and its CEO. For a corporate, a failed payroll run on a Friday is a five-alarm fire. So the *resilience* of the bank's systems — uptime, tested failover, clean migrations — is not a back-office nicety; it is the product. A bank that can't promise payroll will run every single time will not win the mandate, no matter how cheap its fees.

**Fraud and payment security.** A payment system that moves billions on instruction is a target. *Business email compromise* — a fraudster impersonating an executive or supplier to redirect a payment — has stolen billions globally; the 2016 theft of \$81 million from Bangladesh Bank via fraudulent payment messages showed how a single forged instruction can drain an account before anyone notices. Cash-management platforms therefore wrap payments in controls: dual authorization (two people must approve a large payment), payee verification, anomaly detection, and hard limits. These controls are a selling point and a cost; the bank that gets them wrong loses both money and trust, and trust is the entire franchise.

**Concentration risk cuts both ways.** Stickiness is a strength until it becomes a single point of failure. A corporate that routes *everything* through one bank is exposed if that bank has an outage or, worse, gets into trouble itself — which is why sophisticated treasurers deliberately keep a second bank in reserve, accepting some lost pooling efficiency for resilience. From the bank's side, a transaction-banking book concentrated in one industry (say, tech startups, as SVB's was) means the deposits are correlated: when the sector turns, they all pull cash at once, and the "sticky" base proves far less sticky than it looked. Good funding is *diversified* sticky deposits, not a big pile of similar ones.

#### Worked example: the funding cost of losing sticky deposits

Quantify the downside so the stakes are concrete. Suppose a bank's transaction-banking franchise wobbles — a major outage, a reputational hit — and \$10 billion of operating deposits walks out the door. The bank still needs the funding, so it replaces that \$10 billion on the wholesale market. Operating deposits cost it **0.5%**; wholesale funding costs **5.2%**. The extra annual funding cost of the swap:

\$10,000,000,000 × (5.2% − 0.5%) = \$10,000,000,000 × 4.7% = **\$470 million a year**.

That is the price of losing sticky deposits — nearly half a billion dollars of extra interest expense, every year, for as long as the cheap deposits stay gone. The intuition: the same property that makes transaction banking so valuable when you have it — cheap, sticky funding — makes losing it brutally expensive, which is exactly why banks defend these relationships as if their margin depended on them, because it does.

## How it shows up in real banks

**The transaction-banking divisions that anchor the giants.** At the largest global banks — JPMorgan, Citi, HSBC, BNP Paribas, Deutsche Bank, Standard Chartered — transaction banking (often branded "Treasury and Trade Solutions," "Global Transaction Banking," or "Securities Services and Cash Management") is a core, stable earnings engine, not a sideshow. These units serve thousands of multinational corporates and other banks, processing trillions of dollars of payments a year. The reason they're prized is precisely the one this post has built up: recurring fees, minimal capital, and an enormous, sticky, low-cost deposit base. When a bank's leadership talks about a "fortress balance sheet" funded by stable deposits, a big slice of that stability is the corporate operating cash these divisions hold.

**The 2023 deposit runs and the quality of funding.** When Silicon Valley Bank failed in March 2023, the headline was a duration mismatch — long bonds that lost value as rates rose. But the *speed* of the collapse came from the funding side: roughly **94% of SVB's deposits were uninsured** and concentrated among venture-backed tech firms who talked to each other and could move money with a tap. Around **\$42 billion was withdrawn on a single day**. That is the anti-thesis of sticky operating deposits — a funding base that was concentrated, rate-aware, and operationally easy to flee. A bank whose deposits are diversified corporate operating relationships, woven into payroll and collections, does not bleed \$42 billion in a day. The episode turned "deposit quality" from a footnote into a front-page risk metric, and it's why funding stability — the natural output of good transaction banking — is now scrutinized as hard as solvency. The deeper ALM mechanics live in [bank treasury and asset-liability management](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit).

**The rise of real-time treasury and APIs.** Through the late 2010s and 2020s, the live front of competition shifted to connectivity. Banks built API suites so a corporate's software could query balances and trigger payments in real time, and treasury systems began sweeping and investing cash intraday rather than once at day-end. The strategic motive is lock-in: the bank that becomes the real-time financial backbone of a company's operations is nearly impossible to displace. This is also where fintechs and "banking-as-a-service" players push hardest — trying to own the connectivity layer between software and money — and why incumbents treat their API platforms as a competitive battleground rather than a back-office utility.

**Post-2008 capital rules reshaping notional pooling.** When the Basel III leverage ratio came into force, it forced banks to hold capital against gross exposures in a way that made cross-border notional pooling — where a bank nets one customer's overdraft against another's credit balance on its own book — more capital-expensive. Several banks responded by restructuring or curtailing notional pooling offerings, especially across jurisdictions, and steering clients toward physical pooling instead. It's a clean example of how a capital rule written for safety reshaped a specific corporate product, and why a treasurer has to understand the *bank's* regulatory constraints to understand which pooling structure is even on offer. The capital machinery behind this is the subject of the Basel and risk-weighted-asset posts later in this series.

**Working-capital pressure in a high-rate world.** When interest rates jumped in 2022–2023, the cost of carrying idle cash and slow receivables soared — every day of trapped working capital now cost 5%+ instead of near-zero. Corporate treasurers responded by leaning hard on cash management: tightening pooling, accelerating collections, optimizing payment timing to the day. Banks that could deliver real visibility and faster cash conversion won mandates; the value of shaving days off the cash conversion cycle, small in a zero-rate world, became material when money was expensive again. High rates turned cash management from hygiene into a boardroom priority.

## The takeaway: how to use this

The lesson of transaction banking is that the best business a bank owns is often the one nobody outside the bank notices. Strip away the drama and a bank is a machine that borrows short and lends long; its survival depends on the *quality* of the short side. Transaction banking is the purest way to win that short side well — cheap funding, sticky funding, fee income on top, and almost no capital consumed to get it.

So when you read a bank, look past the loan book and the trading revenue and ask three questions about its day-to-day money business. First, **how cheap is its funding?** A low, stable cost of funds — well below peers — usually means a deep base of operating deposits, and that means a strong transaction-banking franchise underneath. Second, **how sticky is its funding?** Deposits tied to payroll, collections, and treasury systems don't run; rate-chasing deposits do. A bank whose funding is operationally anchored is a bank that survives a scare. Third, **how much fee income comes off how little capital?** A business earning a 20% return on the capital it uses is worth far more than one earning 8%, and transaction banking is structurally the former.

For a treasurer, the takeaway flips around: the bank that runs your daily money is your most important banking relationship, and the structure you choose — physical or notional pooling, how aggressively you sweep, how tightly you integrate via H2H and APIs — directly determines how much cash you free and how much interest you stop wasting. The arithmetic is unforgiving in your favor: netting a few million of overlapping balances, or shaving five days off how long you wait to get paid, is millions of dollars a year for work the bank will largely automate.

And it all comes back to the spine. A bank lives and dies on its ability to fund itself cheaply and keep that funding from running. Transaction banking is the quiet engine that does both at once — which is why, for all its lack of glamour, it is one of the most valuable things a bank will ever build.

## Further reading & cross-links

- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — the broader story of why a cheap, sticky deposit base is the whole game; transaction banking is how a bank wins the corporate slice of it.
- [The payments business: how money actually moves between banks](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) — the clearing-and-settlement plumbing underneath every payment a corporate makes.
- [Trade finance: letters of credit, guarantees and supply-chain finance](/blog/trading/banking/trade-finance-letters-of-credit-guarantees-and-supply-chain-finance) — the close cousin of cash management, sold in the same mandate, that de-risks corporate trade.
- [Bank treasury and asset-liability management: the balance-sheet cockpit](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit) — how the bank uses the cheap deposits this business generates to manage the whole balance sheet's liquidity and rate risk.
- [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the case study in why deposit *quality*, not just quantity, decides whether a bank survives a scare.
