---
title: "Non-Performing Loans and the Workout Process: How a Bank Manages Bad Debt"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "What happens after a loan stops paying: the 90-days-past-due line, the classification buckets, the NPL ratio and coverage ratio, restructuring, write-offs, recoveries, NPL sales, and the workout desk that fights for every cent back."
tags: ["banking", "non-performing-loans", "npl", "credit-risk", "loan-workout", "coverage-ratio", "restructuring", "write-offs", "recoveries", "special-assets", "provisioning"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A loan that stops paying does not vanish; it moves down a graded conveyor belt — past-due, then non-performing, then into a workout desk that fights to claw back whatever the borrower and the collateral can still cover. How well a bank manages that journey decides how much of its lending stays profit and how much turns into loss.
>
> - A loan is conventionally called *non-performing* once it is **90 days past due** (or once full repayment is judged unlikely). That single line splits a bank's loan book into "still earning" and "in trouble".
> - Two ratios tell you almost everything: the **NPL ratio** (bad loans ÷ total loans — how big is the problem) and the **coverage ratio** (reserves ÷ bad loans — how much of the problem the bank has already swallowed). A 4% NPL ratio with 90% coverage is a very different bank from a 4% NPL ratio with 40% coverage.
> - When a loan goes bad the bank has four real moves: nurse it back (cure), rewrite the terms (restructure), sell it for cents on the dollar (NPL sale), or seize the collateral and write off the rest (foreclose and charge off). Each trades recovery against time and certainty.
> - The one number to remember: a *recovery* is never the full balance. On a typical defaulted corporate loan a bank gets back maybe **40-75 cents on the dollar** depending on collateral — and that is *before* the legal fees and the years of waiting eat into it.

In the third quarter of 2009, in the depths of the financial crisis, U.S. banks were charging off bad loans at an annualized rate of nearly 3% of all the loans they held — the highest rate in the seventy-year history of the data. Behind that one number sat millions of individual stories: a mortgage where the borrower lost their job and stopped paying, a construction loan on a half-built condo tower nobody would buy, a small-business line of credit drawn down by a company that was quietly going under. None of those loans simply disappeared from the bank's books the day the borrower missed a payment. Each one entered a machine — a slow, grinding, deeply unglamorous machine — whose entire job is to figure out, loan by loan, how much of the money is actually coming back.

That machine is the subject of this post. It is the part of banking nobody puts on a recruiting brochure. There is no trading floor, no glossy app, no billion-dollar deal. There is a desk full of people on the phone with borrowers who can't pay, lawyers filing foreclosure papers, and analysts staring at spreadsheets trying to value a half-finished building. And yet this machine is where a bank's profits are made or unmade. A bank can underwrite brilliantly for a decade and still die if, when the cycle turns, it can't manage the loans that go bad. Lending money is easy; getting it back when things go wrong is the hard part — and it is the part that separates a bank that survives a recession from one that doesn't.

The figure above is the mental model to carry the whole way through: a loan starts out *current*, paying on time. If it stops, it slides down a graded path — 30 days late, 60, 90 — and at the 90-day line it crosses into *non-performing* territory. From there it is handed to a workout desk, which fights for an outcome: cure, restructure, or recover what it can and write off the rest. Every term in this post is a station on that conveyor belt.

![A loan moving from current through past-due stages to NPL then workout and an outcome](/imgs/blogs/non-performing-loans-and-the-workout-process-1.png)

This is the operations-level reality behind a thesis that runs through this whole series: a bank is a leveraged, confidence-funded maturity-transformation machine, and its thin equity cushion has to absorb losses faster than they arrive. Bad loans are exactly how those losses arrive. The workout process is how a bank slows them down.

## Foundations: past-due, non-performing, classification, and the two ratios

Before we can talk about how a bank *manages* bad debt, we have to define — from absolute zero — what "bad" even means, and how a bank measures it. Everyday language is no help here: a loan that is "in default" is not the same as a loan that is "written off", and a "provision" is not a pile of cash. Let us build the vocabulary one term at a time.

### What a loan is, on the bank's books

Recall the one fact that everything else rests on. When a bank makes a loan, that loan is an **asset** on the bank's balance sheet — a thing the bank owns, specifically the right to receive a stream of future payments. (For the full view of how loans sit alongside deposits and equity, see [reading a bank balance sheet](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity).) The loan is worth something only as long as those payments are likely to actually arrive. The whole drama of this post is what happens to the *value* of that asset as the payments become less and less likely.

### Past-due: the clock starts ticking

A loan is **past due** (or *delinquent*) the moment a scheduled payment is missed and not made by its due date. Banks bucket delinquency by how late the payment is, in 30-day bands:

- **30 days past due** — the borrower missed one monthly payment. Often this is just a forgotten bill, a payroll timing glitch, a check in the mail. Most 30-day delinquencies cure themselves.
- **60 days past due** — two missed payments. Now it's a pattern. The bank's collections team is calling.
- **90 days past due** — three missed payments. This is the danger line. Statistically, a loan that reaches 90 days late rarely catches up on its own.

The shorthand is **dpd**, for *days past due*. "The loan is 60 dpd" means the oldest unpaid scheduled payment is two months overdue.

### Non-performing: crossing the line

A **non-performing loan** — almost always abbreviated **NPL** — is a loan that has gone bad in a way the bank must formally recognize. There is no single global definition, but the dominant convention is:

> A loan is non-performing if it is **90 or more days past due**, *or* if the bank judges that the borrower is **unlikely to pay in full** without the bank taking action like seizing collateral — even if no payment has technically been missed yet.

That "or" matters. The 90-day count is the objective, mechanical trigger. The "unlikely to pay" condition — banks call it **unlikely-to-pay**, or UTP — is the judgment trigger: if a borrower's business is clearly failing, a bank doesn't have to wait three months to call the loan non-performing. The European Banking Authority and the Basel framework both anchor the NPL definition on this "90 days past due OR unlikely to pay" pair.

The crucial consequence: once a loan is non-performing, the bank generally stops recognizing the interest on it as income. This is called putting the loan on **non-accrual**. Up to that point, even if a borrower is a bit late, the bank books the interest it is *owed* as revenue. Once the loan goes non-performing, the bank admits that interest probably isn't coming, so it stops counting it. A loan going non-performing therefore hits the bank twice: it stops earning, *and* it forces a charge against profit (we'll get to provisions in a moment).

### Classification: the five grades of a loan

Banks don't just split loans into "fine" and "bad". They run them through a **classification** system — a credit grade for the *health* of each loan, distinct from the original credit score of the borrower. The standard five-bucket ladder, used in some form by regulators worldwide, runs:

1. **Standard (or "Pass")** — performing normally, full repayment expected. Low provision.
2. **Special mention (or "Watch")** — early signs of weakness (a little late, deteriorating financials) but not yet non-performing. A yellow flag.
3. **Substandard** — well-defined weaknesses; the loan is non-performing and some loss is likely. This is the first genuinely "bad" bucket.
4. **Doubtful** — collection in full is highly questionable; significant loss is probable.
5. **Loss** — judged uncollectible; the bank writes it off.

The first two are *performing* loans; the last three are *non-performing* (or "classified") loans. The point of the ladder is that it forces a graded response: each step down the ladder triggers a heavier provision against the loan. A bank can't quietly keep calling a dying loan "standard" — examiners check the classifications.

![Five loan classification buckets standard to loss with rising provision percentages](/imgs/blogs/non-performing-loans-and-the-workout-process-2.png)

### Provision, reserve, write-off, recovery — the four words people mix up

These four terms are the spine of the whole post, and they are constantly confused. Define them precisely:

- A **provision** (or *loan-loss provision*) is an *expense* the bank books on its income statement when it expects a loan to lose value. It is a forward-looking charge against profit. You set aside an estimate of the loss *before* it is final.
- The **loan-loss reserve** (also the *allowance for loan losses*, or under the modern accounting rules the *allowance for credit losses*) is the *accumulated* stock of all those provisions sitting on the balance sheet, net of what's been used up. The provision is the *flow* into the reserve each period; the reserve is the *stock*. (The full mechanics of how the reserve is built — the IFRS 9 and CECL expected-credit-loss models — are covered in the sibling post on [collateral, security, and loan-loss provisioning](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl); here we pick up where that leaves off, once the loan has actually gone bad.)
- A **write-off** (or *charge-off*) is the moment the bank *removes* an uncollectible loan (or the uncollectible part of it) from its books, using up the reserve it set aside for it. A write-off is the *funeral*, not the *death* — the loss was already recognized via provisions; the write-off just cleans the corpse off the balance sheet.
- A **recovery** is money the bank gets back *after* it has written a loan off — from the borrower finally paying, from selling the collateral, from a legal settlement. Recoveries flow back *into* the reserve.

Here is the relationship that ties them together, and it's worth memorizing because it's how a bank's reserve actually moves:

$$\text{Ending reserve} = \text{Beginning reserve} + \text{Provisions} - \text{Write-offs} + \text{Recoveries}$$

Each symbol is a dollar amount over a period: provisions add to the reserve, write-offs (charge-offs) draw it down as bad loans are removed, and recoveries top it back up when written-off money comes back. The combination of write-offs minus recoveries is called **net charge-offs** — the bank's actual, realized loan losses for the period. Net charge-offs are the number that ultimately tells you what lending really cost.

### The two ratios that define the problem

Finally, the two ratios you'll see in every bank's report and every analyst's note:

- The **NPL ratio** is non-performing loans divided by total gross loans. It answers: *how big is the bad-debt problem relative to the whole book?* A 1-2% NPL ratio is healthy; 5%+ is a sick bank; double digits is a crisis. In the European banking system after 2008, average NPL ratios climbed above 7% and some southern-European banks carried 30-40%.
- The **coverage ratio** is the loan-loss reserve divided by non-performing loans. It answers: *how much of the bad-debt problem has the bank already absorbed into its reserves?* A coverage ratio of 70% means the bank has already set aside reserves equal to 70 cents for every dollar of bad loans. The higher the coverage, the smaller the *future* hit.

These two ratios work as a pair. The NPL ratio measures the size of the wound; the coverage ratio measures how much bandage is already on it. A bank with a scary-looking 6% NPL ratio but 90% coverage is in far better shape than one with a modest 3% NPL ratio and only 30% coverage — because the first bank has already taken most of its pain, while the second is hiding losses it has yet to admit.

With the vocabulary built, let's go deep on each stage of the conveyor belt.

## How loans slide from current to non-performing

A loan doesn't fail in a single instant — it deteriorates. Understanding the *slide* is the whole game, because the earlier a bank catches a loan on its way down, the more it can recover.

### The delinquency funnel

Take all the loans that miss a payment in a given month as entering a funnel. Most leak out the top: a 30-day delinquency cures (the borrower catches up) far more often than not. Of those that reach 60 days, fewer cure. By 90 days, the cure rate collapses. This is why the 90-day line is the conventional NPL trigger — it is roughly the point at which "late" reliably converts into "not coming back without a fight".

The numbers vary wildly by loan type. A prime mortgage that goes 30 days late might cure 80%+ of the time; a subprime auto loan or an unsecured credit card that goes 60 days late might cure far less. This is why banks watch *early-stage* delinquency (the 30-day bucket) as a leading indicator: a rise in 30-day delinquencies this quarter is a forecast of rising charge-offs two or three quarters out. The early bucket is the smoke; the charge-off is the fire.

### A roll-rate, in plain numbers

Banks model the slide with **roll rates** — the probability that a loan in one delinquency bucket "rolls" to the next, worse one. Suppose a bank observes these monthly roll rates on a pool of loans: 40% of 30-day delinquent loans roll to 60 days; 70% of 60-day loans roll to 90; and 85% of 90-day loans roll on to charge-off. Then a loan that just went 30 days late has a roughly $0.40 \times 0.70 \times 0.85 \approx 0.238$, or about a 24%, chance of ending up charged off. The other 76% cure somewhere along the way. Multiply that 24% by the loan balance and you have the expected loss from that one delinquency — which is, essentially, what a provision is trying to capture.

The reason this matters operationally: a bank's collections and workout effort is aimed at *lowering the roll rates*. Every phone call, payment plan, and restructuring offer is an attempt to push loans back up the funnel before they reach the point of no return. A collections department that cuts the 60-to-90 roll rate from 70% to 60% has just measurably reduced the bank's future losses.

### Early-warning systems: catching the slide before it starts

The very best banks don't wait for a missed payment at all. They run **early-warning systems** — dashboards that watch for signs of stress *before* a loan goes past due, while there's still time and goodwill to fix it. The signals depend on the loan type:

- For a *corporate* borrower: a covenant breach (a loan covenant is a promise the borrower made, like keeping debt below a multiple of earnings — a breach is an early tripwire); a credit-rating downgrade; deteriorating quarterly financials; a sudden, unusual full draw-down of a revolving credit line (a desperate company grabs all available cash before the bank can pull the line); or a key customer or supplier going under.
- For a *retail* borrower: a falling credit score, a missed payment on a *different* lender's loan (cross-default signals show up in credit-bureau data), or rising balances and minimum-only payments on credit cards.

When an early-warning flag fires, a good bank moves the loan to a "watch list" — the special mention bucket — and assigns a relationship manager to intervene *before* the 90-day clock ever starts. This is where the real recovery is won: a loan caught at the first sign of weakness, with the borrower still cooperative and the collateral still intact, recovers far more than one that has already been through three months of missed payments, deteriorating goodwill, and decaying collateral. The graveyard wisdom of every workout banker is the same: *the cheapest workout is the one that happens before the default.*

There is a behavioral wrinkle here that matters. The loan officer who made the loan is often the *last* person to admit it's going bad — admitting it means admitting their own underwriting was wrong, and it kills a relationship they spent years building. This is precisely why banks separate the early-warning and workout functions from origination, and why examiners scrutinize the watch list so hard: left to the originators, troubled loans would be quietly nursed and re-aged until they were beyond saving. The independence of the people who classify and work out loans is, itself, a risk control.

#### Worked example: the NPL ratio

Let's compute the headline ratio on a small, friendly balance sheet. Suppose a community bank has a total gross loan book of \$2,000 million (\$2 billion). Of that, the following loans are non-performing — 90+ dpd or judged unlikely to pay:

- \$30 million of mortgages
- \$25 million of commercial real-estate loans
- \$15 million of small-business loans
- \$10 million of consumer loans

Total non-performing loans = \$30m + \$25m + \$15m + \$10m = **\$80 million**.

$$\text{NPL ratio} = \frac{\text{Non-performing loans}}{\text{Total gross loans}} = \frac{\$80\text{m}}{\$2{,}000\text{m}} = 0.04 = 4.0\%$$

So 4% of this bank's loan book is non-performing. Is that bad? On its own, it's elevated — comfortably above the 1-2% you'd see in good times, the kind of number that shows up when a credit cycle has turned. But the NPL ratio alone is only half the story; it tells you the *size* of the problem, not how much of it the bank has already absorbed. For that we need the coverage ratio — next. The intuition: the NPL ratio is the bank's "how sick am I" thermometer, and 4% is a fever.

## The provisioning waterfall once a loan is non-performing

Once a loan crosses into non-performing territory, the bank must decide how much of it to provision for — how much of the balance to write down as an expected loss. This is where classification becomes money.

### Provisions follow classification

Each classification bucket carries a customary provisioning percentage. The exact numbers depend on the accounting regime and the collateral, but a representative ladder (used in spirit by many regulators, and shown in the classification figure above) looks like this:

| Classification | Status | Typical provision |
|---|---|---|
| Standard | Performing | ~1% |
| Special mention | Performing, watch | ~3-5% |
| Substandard | Non-performing | ~20% |
| Doubtful | Non-performing | ~50% |
| Loss | Non-performing | 100% |

Read down that column and you see the waterfall: as a loan deteriorates one grade, the provision against it roughly *doubles or more*. A loan that drops from substandard to doubtful takes its provision from around 20% to around 50% of the balance — a charge to profit of roughly 30% of the loan, all at once. This is why a wave of downgrades, even before any loan is actually written off, can crush a bank's earnings: the provisions are recognized the moment the classification worsens, not when the cash is finally lost.

### Specific versus collective provisions

There are two flavors of provision, and the distinction matters:

- A **specific provision** is set against an individually-identified bad loan. The bank looks at *this* defaulted \$5 million commercial loan, estimates it will recover \$3 million from the collateral, and books a specific provision of \$2 million against it.
- A **collective** (or general) provision is set against a *pool* of loans that look fine individually but, statistically, will produce some losses — like a portfolio of 50,000 credit cards. You can't say which cards will default, but you know roughly what fraction will, so you reserve against the pool.

Modern accounting (IFRS 9 internationally, CECL in the U.S.) pushed banks toward recognizing *expected* losses earlier and more on a forward-looking, lifetime basis — but the underlying logic of specific-versus-collective survives. For the deep mechanics of those models, see the [provisioning post](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl). For our purposes here, the key point is that the provision is the bank's *estimate* of the loss, taken in advance; the workout process is the bank's *attempt to beat that estimate* by recovering more than feared.

#### Worked example: a coverage ratio

Continue with our community bank. It has \$80 million of non-performing loans. Suppose its loan-loss reserve — the accumulated stock of all the provisions it has set aside — stands at \$56 million. The coverage ratio is:

$$\text{Coverage ratio} = \frac{\text{Loan-loss reserve}}{\text{Non-performing loans}} = \frac{\$56\text{m}}{\$80\text{m}} = 0.70 = 70\%$$

So the bank has already reserved 70 cents for every dollar of bad loans. Suppose, based on the collateral behind those loans, the bank expects to ultimately recover about 50% of the \$80 million — meaning it expects to lose about \$40 million. Its \$56 million reserve *more* than covers that expected \$40 million loss; this bank is conservatively reserved, and future surprises are more likely to be pleasant (reserve releases) than nasty.

Now take a different bank with the same \$80 million of NPLs but only a \$28 million reserve — a 35% coverage ratio. If *its* loans also recover only 50%, it faces a \$40 million loss against just \$28 million of reserves: a \$12 million shortfall it has not yet recognized, a future charge waiting to land on its equity. Same NPL ratio, wildly different health. The intuition: coverage tells you whether the bad news is already in the price.

![Coverage ratio bars showing reserving stances from thinly reserved to heavily reserved](/imgs/blogs/non-performing-loans-and-the-workout-process-6.png)

(The chart above shows the same idea as a spread of stances; the dashed line marks 100% coverage, where every dollar of NPLs is fully reserved. Note that a single bank's actual coverage depends on its collateral — a heavily-secured book can be perfectly safe at 50% coverage, while an unsecured one might need more. We'll return to why collateral changes the math.)

## The NPL ratio across a credit cycle: why it always lags

Here is one of the most important — and most counter-intuitive — facts about non-performing loans: the NPL ratio is a *lagging* indicator. It peaks not during a recession, but *after* it, often a year or more after the economy has already started to recover.

### Why bad loans arrive late

The logic is mechanical. A recession hits; companies lose revenue and people lose jobs. But it takes time for a stressed borrower to actually stop paying — they run down savings, draw on credit lines, sell assets, and try everything before they default. Then it takes 90 more days for a missed payment to become a non-performing loan. Then it takes months or years for the workout to resolve into an actual charge-off. So the wave of bad loans crests well after the economic low point. In the U.S. after 2008, the recession technically ended in mid-2009, but bank failures — the ultimate expression of bad loans overwhelming capital — *peaked in 2010*, with 157 FDIC-insured banks failing that year.

![US bank failures per year 2005 to 2025 peaking at 157 in 2010](/imgs/blogs/non-performing-loans-and-the-workout-process-3.png)

The chart above shows the systemic backdrop: bad loans arrive in waves, and when they overwhelm a bank's equity, the bank fails. The 2008-2012 wave is the classic credit cycle — loans made in the boom going bad in the bust, peaking after the recession ended. (Notice the small 2023 spike is different in character: SVB, Signature, and First Republic failed from an interest-rate and deposit-run mechanism, not a credit-loss wave — a reminder, explored in the [SVB / Credit Suisse post](/blog/trading/finance/svb-credit-suisse-2023-bank-runs), that not every bank failure is an NPL failure.)

### The pro-cyclicality trap

This lag creates a vicious trap for banks. In the good times, NPLs are low, provisions are tiny, profits look fantastic — so banks lend *more*, often loosening standards to keep growing. Then the cycle turns, those loosely-underwritten loans go bad all at once, provisions spike, and profits crater exactly when the bank can least afford it. Worse, the bank may be forced to *raise* lending standards and shrink its book just as the economy needs credit — deepening the downturn. This is why bank earnings are deeply *pro-cyclical*: they over-state health in the boom and over-state distress in the bust.

The stylized cycle chart makes the shape concrete: an NPL ratio sitting calmly below 2% through the expansion, then surging as the recession bites, peaking *after* the downturn, and only slowly grinding back down as workouts resolve over the following years.

![NPL ratio rising through a recession and peaking afterward then declining](/imgs/blogs/non-performing-loans-and-the-workout-process-9.png)

#### Worked example: a restructuring that cures versus one that re-defaults

This is the example at the heart of the workout business, because *restructuring* is the most common — and most abused — tool a bank has. A **restructuring** (when done for a struggling borrower, it's a *troubled debt restructuring*, or TDR, and in modern terminology a *forbearance* measure) means changing the loan's terms to give the borrower a realistic chance to pay: a lower interest rate, a longer term, a payment holiday, or even forgiving part of the principal.

Take a small business with a \$1,000,000 term loan at 8% interest over 5 years. Its required payment is roughly \$20,300 per month. The business hits a rough patch — revenue down 30% — and goes 90 days past due. The loan is now non-performing. The bank has a choice. It restructures: it cuts the rate to 5% and extends the term to 10 years, dropping the monthly payment to about \$10,600 — nearly half.

**Case A — it cures.** The business's revenue recovers over the next year. At the new \$10,600 payment it can comfortably pay. After it makes, say, 6-12 consecutive on-time payments — a *seasoning* period the bank requires before it trusts the cure — the loan is reclassified back to performing. The bank gave up some interest income (5% instead of 8%) and stretched its money out longer, but it avoided a default. It will collect its principal in full. The restructuring *worked*.

**Case B — it re-defaults.** The business's problem was not temporary — its main customer left and isn't coming back. The lower payment buys a few months, but by month 8 the company misses again. The loan re-defaults. Now the bank has wasted a year, the collateral has aged and possibly lost value, and the loan is right back to non-performing — except deeper in the hole. This is a **re-default**, and a restructuring that merely delays an inevitable default is sometimes called "extend and pretend": the bank dresses up a bad loan to avoid recognizing the loss today, only to take a bigger loss later.

The lesson — and it's the single most important lesson in workout — is that a restructuring only works if the borrower's *underlying cash flow* genuinely recovered. Rewriting the paperwork doesn't fix a broken business. A bank's skill in restructuring is, fundamentally, its skill at telling Case A apart from Case B *before* it commits another year. The intuition: forbearance buys time, and time only helps if the borrower can use it.

![Two restructured loans one re-defaulting and one curing back to performing](/imgs/blogs/non-performing-loans-and-the-workout-process-5.png)

## The workout desk: where bad loans go to be fought over

When a loan is small and homogeneous — a credit card, a personal loan — the bank handles default with an industrial collections process: automated calls, payment plans, and eventually selling the debt to a collection agency. But when a loan is large, complex, or secured by real assets — a \$50 million commercial real-estate loan, a corporate term loan — it goes to a specialist team. This team has many names: the **workout group**, the **special assets** department, or the **special situations** desk. Its job is to maximize recovery on loans that have gone wrong.

### What the desk actually does

The workout desk is staffed by a different kind of banker than the one who made the loan. The relationship manager who originated the loan was selling, building rapport, growing the relationship. The workout officer is a turnaround specialist and a hardball negotiator. The moment a loan transfers to workout, the relationship changes: the bank is no longer a partner in the borrower's growth; it is a creditor trying to get its money back. Banks deliberately separate the two — partly because the skills differ, and partly because the loan officer who made the loan has an emotional and reputational incentive to keep pretending it's fine.

The desk's first job is **valuation and triage**: for each bad loan, figure out what it's actually worth in recovery. That means re-appraising the collateral, analyzing the borrower's remaining cash flow, understanding where the bank stands in the line of creditors (its *seniority*), and estimating what each possible action would net. Then it picks a strategy.

### The four options and their trade-offs

For any given non-performing loan, the workout desk is choosing among four broad routes, each trading recovery against time and certainty:

1. **Cure / collect** — pressure the borrower to bring the loan current by paying the arrears, perhaps with a short payment plan. Best outcome (loan returns to performing, full recovery) but only works if the borrower can actually pay.
2. **Restructure** — rewrite the terms to a level the borrower can sustain (the example above). Preserves the relationship and often beats liquidation, but risks re-default.
3. **Sell the NPL** — sell the loan, as-is, to a third party (a distressed-debt fund, a specialist NPL buyer) for a discounted cash price. The bank gets certainty and an immediate clean-up, but accepts a haircut.
4. **Foreclose and write off** — seize and sell the collateral, pursue the borrower legally for any shortfall, and charge off whatever can't be recovered. This is the nuclear option: slow, costly, adversarial, but sometimes the only route that recovers anything.

![Workout desk choosing among cure restructure sell and foreclose leading to outcomes](/imgs/blogs/non-performing-loans-and-the-workout-process-7.png)

The art of the workout desk is matching the route to the loan. A fundamentally sound borrower hit by a temporary shock gets a restructuring. A dead business with valuable collateral gets foreclosure. A messy pile of small unsecured loans the bank doesn't want to chase gets sold. The wrong choice destroys recovery: foreclosing on a viable business throws away a relationship and crystallizes a loss that patience would have avoided; restructuring a hopeless one wastes a year and lets the collateral rot.

### Why the workout desk grows and shrinks with the cycle

In good times, the special-assets desk is a sleepy backwater with a handful of files. When the cycle turns, it explodes — banks staff up workout teams aggressively in a downturn, and the people who can value distressed collateral and negotiate with failing borrowers suddenly become the most valuable employees in the building. The Resolution Trust Corporation, set up to clean up the U.S. savings-and-loan crisis, was essentially a giant government workout desk that disposed of the assets of more than a thousand failed thrifts (the full story is in the [savings-and-loan crisis post](/blog/trading/banking/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures)).

This boom-and-bust staffing problem is exactly why a whole industry of **third-party loan servicers** exists. Rather than build and tear down a huge workout team every cycle, many banks outsource the grinding work — the collateral re-appraisals, the borrower negotiations, the legal process management — to specialist servicing firms that do nothing else and keep their expertise sharp across cycles. When a distressed-debt fund *buys* a portfolio of NPLs, it almost never works them out itself either; it hires one of these servicers to do the operational labor while the fund supplies the capital. The result is a clean division of labor: the bank (or fund) owns the credit risk and the upside, and a servicer it pays a fee handles the unglamorous fight to extract every recoverable cent. Understanding this split matters when you read a bank, because a bank that has *sold the servicing* of its bad loans has also given up some control over how hard they're worked — and the fee it pays the servicer is a direct drag on its net recovery.

## Recoveries: how much actually comes back

Here is the brutal truth the workout desk lives with: a recovery is *never* the full balance. When a loan defaults, the bank recovers some fraction of what it was owed — and that fraction depends overwhelmingly on **collateral** and **seniority**.

### Loss given default and the recovery rate

The credit-risk world measures this with a single parameter: **loss given default**, or **LGD** — the fraction of the exposure the bank *loses* if the borrower defaults. The flip side is the **recovery rate**, which is simply $1 - \text{LGD}$. If a loan has an LGD of 40%, the bank recovers 60 cents on the dollar. (The full PD-LGD-EAD machinery — probability of default times loss given default times exposure — is the subject of the [credit-risk management post](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss); here we care specifically about LGD, the recovery side, because that's what the workout desk fights over.)

What drives LGD? Above all, what stands behind the loan:

- **Senior secured loans** — backed by specific collateral (property, equipment, receivables) and first in line. These recover the most: often 70-80% (LGD ~20-30%). If the borrower defaults, the bank seizes and sells the collateral.
- **Senior unsecured loans** — no specific collateral, but ahead of junior creditors. Recover roughly half (LGD ~45%).
- **Subordinated debt** — behind the senior creditors in line. Recovers little after the seniors are paid (LGD ~65% or worse).
- **Unsecured retail** — credit cards, personal loans. Often recover only around 45% (LGD ~55%) — and even that requires chasing.

![Recovery rate by claim type from subordinated to senior secured](/imgs/blogs/non-performing-loans-and-the-workout-process-4.png)

The chart makes the hierarchy concrete: collateral and seniority are the two levers that determine how much a workout can recover. This is *why* banks insist on collateral and *why* they price unsecured lending so much higher — they know going in that recovery on a default will be poor, so they need a fatter spread to compensate. The recovery side of the loan is baked into its price on day one.

### Recovery costs time and money

The recovery rates above are *gross* — the cents you collect from the collateral or the settlement. But getting there isn't free, and this is where novices underestimate the damage:

- **Legal and process costs.** Foreclosure means lawyers, court filings, property management, brokers to sell the seized asset. These can eat 5-15% of the collateral's value.
- **Time.** A foreclosure or bankruptcy can take one to three years (or longer in slow jurisdictions). Money recovered in three years is worth less than money recovered today — you have to discount it. And the longer it drags, the more the collateral can deteriorate (an empty building decays; seized equipment depreciates).
- **Market risk on the collateral.** If you're foreclosing on real estate in a recession — which is exactly when defaults spike — you're selling into a falling market, often at a fire-sale discount because everyone is selling at once.

So the *net* recovery — what actually lands back in the bank after costs and discounting — is materially less than the gross. A 60% gross recovery can easily become a 43% net recovery once the lawyers and the years are paid for.

![Workout waterfall reducing a 60 cent gross recovery to 43 cents net of cost and time](/imgs/blogs/non-performing-loans-and-the-workout-process-8.png)

#### Worked example: a recovery net of workout cost

Take a single defaulted commercial loan with an outstanding balance of \$100 (we'll use \$100 so every number reads as cents on the dollar). It is senior secured by a commercial property. Walk the recovery through:

- **Gross recovery from the collateral:** the workout desk re-appraises the property and expects to sell it for an amount that, against the \$100 balance, returns about \$60. So gross recovery = **\$60** (a 60% gross recovery, LGD 40% gross).
- **Less legal and process costs:** lawyers, foreclosure filing, broker fees to sell the property — call it **\$9**.
- **Less the time discount:** the sale won't close for about two years, and the bank discounts that future \$51 (the \$60 net of legal that arrives later) back to today. The discounting and the carrying cost of waiting shave off roughly another **\$8**.

$$\text{Net recovery} = \$60 - \$9 - \$8 = \$43$$

So the bank, which thought it was 60% secured, actually nets about **\$43 on the \$100** — a *net* recovery of 43%, or a true LGD of 57%. The gap between the 40% LGD on paper and the 57% LGD in reality is the cost of the workout itself. The intuition: collateral promises a recovery, but lawyers and time collect a toll before the bank does.

This is why the best workout outcome is almost always the one that *avoids* the courtroom — a cure or a sustainable restructuring, where the borrower keeps paying — and why an NPL sale, even at a steep discount, can beat a foreclosure: it converts a slow, uncertain, expensive \$43 into a certain \$45 in cash today.

## NPL sales: getting bad loans off the books

When a bank has too many bad loans — or simply doesn't want to spend years working them out — it can sell them. The **NPL sale** market is large and specialized: distressed-debt funds, private-equity firms, and dedicated NPL servicers buy portfolios of bad loans at a discount, then run their own workout to extract more than they paid.

### Why a bank sells at a loss on purpose

It sounds crazy to sell a \$100 loan for \$30. But look at the bank's position. The loan is non-performing, earning nothing. It is consuming capital (regulators make banks hold capital against risky assets — see [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion)). Working it out will take years of staff time and uncertain recovery. And a big pile of NPLs scares depositors, investors, and regulators. Selling it does several things at once: it frees up capital, removes the management distraction, eliminates the uncertainty, and visibly shrinks the NPL ratio — improving the number every analyst is watching.

The buyer, meanwhile, specializes in exactly this. A distressed-debt fund with a low cost of capital, a lean workout operation, and no franchise to protect can squeeze more out of the loans than the bank can — and is happy to take the risk that the bank wants off its books. The discount is the price of certainty and speed for the seller.

### The European NPL cleanup

The clearest real-world example is post-crisis Europe. After 2008, European banks were buried in NPLs — Italian banks alone carried over €300 billion of them at the peak around 2015, and system-wide euro-area NPLs exceeded €1 trillion. They couldn't work them all out internally, so a vast NPL-sale market developed. Banks sold portfolios to funds at deep discounts; governments set up "bad banks" (asset-management companies like Spain's SAREB or Ireland's NAMA) to warehouse and dispose of the toxic assets. Over the following decade, the euro-area NPL ratio fell from above 7% to under 2% — much of it through sales rather than through cures. The cleanup worked, but it took a decade and crystallized enormous losses.

#### Worked example: an NPL sale versus a workout

A bank holds a \$200 million portfolio of non-performing commercial real-estate loans. It has two choices.

**Option 1 — work them out internally.** The workout desk estimates a 55% gross recovery (\$110 million) over an average of three years, with legal and servicing costs of about \$15 million and a time-discount drag worth roughly another \$10 million. Net recovery in present-value terms: $\$110\text{m} - \$15\text{m} - \$10\text{m} = \$85\text{m}$ — about 42.5 cents on the dollar, three years out, and uncertain.

**Option 2 — sell the portfolio now.** A distressed-debt fund offers \$78 million in cash today — 39 cents on the dollar.

On paper the workout nets \$85m vs the sale's \$78m. But the workout's \$85m is a *risky, three-year estimate*; if the property market dips, recovery could fall to \$70m or worse, and the bank carries the capital and the uncertainty the whole time. The sale's \$78m is *certain cash today*, frees capital immediately, and drops the bank's NPL ratio at once. Many banks take the \$78m — they pay a few million for certainty and a clean balance sheet. The intuition: an NPL sale is the bank buying certainty and capital relief, and the discount is the premium it pays for them.

## The examiner, the accountant, and the incentive to hide

There is a constant tension running underneath everything in this post, and it's worth naming directly: the bank that *holds* the bad loans is also the bank that gets to *classify* them and *decide* how much to provision. That is a conflict of interest the size of a building. A bank under earnings pressure has every incentive to keep a deteriorating loan in a higher bucket than it deserves, to under-provision, and to restructure aggressively so loans never officially cross the non-performing line. The whole apparatus of bank supervision exists, in large part, to stop exactly that.

### What the examiner does

Bank examiners — from the FDIC, the OCC, and the Federal Reserve in the U.S., the ECB and national supervisors in Europe — periodically descend on a bank and *re-grade its loans themselves*. They pull a sample of large credits, read the files, re-appraise the collateral, and assign their own classifications. If the examiner thinks a loan the bank called "standard" is really "substandard", the bank is forced to downgrade it and book the heavier provision. In severe cases, examiners can force a bank to recognize losses it was trying to defer — which, by eating into the equity cushion, can be what finally tips a wobbly bank over.

This is captured in the asset-quality piece of the **CAMELS** rating supervisors assign each bank (the "A" stands for asset quality). A bank that lets its NPLs pile up and its coverage thin out gets a poor asset-quality score, which triggers tighter supervision, restrictions on dividends and growth, and ultimately enforcement action. The classification system isn't an accounting nicety — it is the channel through which a bank's bad-loan management becomes the regulator's business.

### Why the accounting kept getting tightened

The history of NPL accounting is a history of regulators closing loopholes that let banks hide losses. The old "incurred loss" model only let banks provision once a loss was nearly certain — which meant reserves were always *too small, too late*, building up only after the cycle had already turned. The 2008 crisis exposed this brutally: banks had thin reserves going into the storm and had to take enormous provisions all at once, exactly when their capital was most stretched. The response was the modern **expected-credit-loss** models — IFRS 9 globally and CECL in the U.S. — which force banks to provision for losses *expected over the life of the loan* from day one, building reserves earlier and more counter-cyclically. The deeper mechanics are in the [provisioning post](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl), but the motivation is pure NPL management: regulators learned that if you let banks decide when a loss is "real", they will always decide it's real too late.

The European "calendar provisioning" rules go even further: they put bad loans on a *clock*. A non-performing loan must be provisioned up toward 100% over a fixed number of years — faster for unsecured loans, slower for secured — regardless of whether the workout has resolved. The point is to make it pointless to carry a zombie NPL forever: if you can't recover it within the calendar, you must fully reserve against it anyway, so you may as well sell it or write it off and move on. It is a direct regulatory assault on "extend and pretend".

## Common misconceptions

**"A non-performing loan is a total loss."** No — and this is the single biggest misconception. A non-performing loan is one that has *stopped paying as agreed*, not one whose money is gone. On a senior secured loan, the bank often recovers 70-80 cents on the dollar through the collateral. The *expected loss* on the NPL is the balance times the LGD — and LGD is rarely 100%. Calling an NPL a total loss confuses "in trouble" with "uncollectible", which are the substandard and loss buckets at opposite ends of the classification ladder.

**"Provisions are cash the bank has set aside."** No. A provision is an accounting *expense* and the reserve is a *contra-asset* — a number that reduces the reported value of the loan book. It is not a vault of cash. When a bank "uses its reserve" to absorb a write-off, no cash moves; the loan and an equal slice of reserve simply both come off the books. The cash was already lent out — that's the whole point. Treating reserves as a rainy-day cash fund leads people to wrongly think a bank with big reserves has lots of liquidity.

**"A restructured loan is a cured loan."** No. A restructuring *changes the terms*; it does not, by itself, fix the borrower. A meaningful share of troubled debt restructurings re-default — sometimes 30-40% within a couple of years. That's why accounting rules require a restructured loan to keep being treated cautiously and to "season" (make consecutive on-time payments for typically 6-12 months) before it can be reclassified as performing. A restructuring is a *bet* that the borrower will recover, not proof that they have.

**"A low NPL ratio means a safe bank."** Not necessarily. A low NPL ratio can mean a healthy book — or it can mean the bank is hiding problems by aggressively restructuring loans to keep them off the non-performing list ("extend and pretend"), or that the cycle simply hasn't turned yet. A low NPL ratio in the *late* stage of a long boom is one of the most dangerous numbers in banking, because it's the calm before the wave. Always read the NPL ratio next to the coverage ratio, the trend in early-stage delinquencies, and the point in the credit cycle.

**"Writing off a loan means the bank gives up on the money."** No. A write-off is an accounting event — removing an uncollectible balance from the books — not a legal forgiveness of the debt. The borrower usually still owes the money, and the bank (or a collection agency it sells the debt to) can keep pursuing it. Money collected after a write-off is a *recovery*, and recoveries flow right back into the reserve. The funeral isn't always the end of the story.

## How it shows up in real banks

**The 2009-2010 U.S. charge-off peak.** As the financial crisis ground on, U.S. banks' net charge-offs as a share of total loans peaked at nearly 3% on an annualized basis in 2009-2010 — the worst in the history of the data. Critically, this peak came *after* the recession officially ended in mid-2009. The bad loans were a lagging consequence of the boom-era lending, and bank failures crested in 2010 at 157, as the wave of NPLs finally overwhelmed the thin equity cushions of the weakest banks. The lesson: the NPL problem is biggest after the storm appears to have passed.

**Italy's NPL mountain and the bad-bank cleanup.** By around 2015, Italian banks carried roughly €340 billion of non-performing loans — a NPL ratio in the high teens, with some banks far worse. The problem was so large that internal workouts couldn't clear it. The solution was a decade-long campaign of NPL sales to distressed-debt funds, government guarantee schemes (the "GACS" securitization guarantee), and asset-management vehicles. The Italian NPL ratio fell from the high teens toward the low single digits by the early 2020s — a textbook demonstration that, past a certain scale, banks dispose of bad debt rather than nurse it, and crystallize the loss to move on.

**The "extend and pretend" of Japan's lost decade.** After Japan's asset bubble burst in 1990, its banks were left with enormous bad loans against collapsed real-estate collateral. Rather than recognize the losses, foreclose, and recapitalize, many banks kept rolling over loans to insolvent "zombie" borrowers — restructuring without recovery, the re-default trap at national scale. The hidden NPLs sat on bank balance sheets for years, starving healthy borrowers of credit and prolonging stagnation. It took until the early 2000s, and forced examinations, for the bad loans to be properly recognized and cleaned up. The lesson: refusing to recognize NPLs doesn't make them go away; it just postpones and enlarges the reckoning.

**Coverage ratios under scrutiny in Europe.** When the European Central Bank took over supervision of the largest euro-area banks in 2014, one of its first moves was an Asset Quality Review that forced banks to properly classify their loans and raise coverage. The ECB later pushed "calendar provisioning" — rules requiring banks to provision a non-performing loan up to nearly 100% over a set number of years whether or not it's resolved, specifically to stop banks from carrying under-provisioned NPLs indefinitely. The coverage ratio went from a number banks could massage to a regulated minimum, precisely because supervisors learned that a low coverage ratio is where hidden losses hide.

**The pandemic forbearance wave that didn't become a charge-off wave.** In 2020, banks worldwide granted massive forbearance — payment holidays on mortgages, business loans, and cards — as the pandemic hit. Under normal rules, many of those would have rolled to non-performing. Regulators temporarily relaxed the classification treatment so that COVID-related forbearance didn't automatically count as a troubled restructuring, and unprecedented fiscal support kept borrowers solvent. The result: the expected wave of NPLs largely never arrived — charge-offs stayed low and banks actually *released* reserves in 2021. It was a rare case where forbearance worked at scale because the underlying cash-flow shock really was temporary — the Case A outcome, system-wide.

## The takeaway: read the workout, not just the loan

If you take one idea from this post, make it this: **a bank's quality is revealed not by how it lends, but by how it manages the loans that go wrong.** Anyone can make a loan in a boom. The test is the bust — and the workout process is where that test is graded.

So when you read a bank, don't stop at the headline NPL ratio. Read the *pair*: the NPL ratio (how big is the problem) alongside the coverage ratio (how much is already absorbed). A rising NPL ratio with rising coverage is a bank facing its problems; a flat NPL ratio with falling coverage is a bank hiding them. Then look at the *trend in early-stage delinquencies* — the 30-day bucket is the leading indicator that tells you where charge-offs are headed two or three quarters out. Then ask where you are in the credit cycle, because the NPL ratio lags: a beautifully low number late in a long expansion is a warning, not a reassurance.

And remember the asymmetry baked into the whole machine. A loan that performs earns the bank a few percent a year in interest. A loan that defaults can lose 40-60% of its entire principal in one shot — and that loss lands on the thin sliver of equity that, as the spine of this series keeps insisting, is the only thing standing between the bank and insolvency. The arithmetic is merciless: it takes a lot of good loans to pay for one bad one. That is why the unglamorous workout desk — the people on the phone with borrowers who can't pay, the analysts valuing half-built towers — quietly does as much to keep a bank alive as anyone on the trading floor. They are not collecting debts. They are defending the cushion.

The next time a bank reports a "manageable" rise in non-performing loans, you'll know what to look for: not the size of the wound, but the bandage already on it, the speed it's spreading, and whether the bank is fighting the bad debt or just pretending it isn't there.

## Further reading & cross-links

- [Collateral, security, and loan-loss provisioning: IFRS 9 and CECL](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl) — the expected-credit-loss models that build the reserve this post draws down; read it for the provisioning mechanics that precede the workout.
- [Credit-risk management: PD, LGD, EAD, and expected loss](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss) — the full credit-risk engine; this post zoomed in on LGD, the recovery side that the workout desk fights over.
- [Reading a bank balance sheet: assets, liabilities, and equity](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity) — where loans, reserves, and equity actually sit, and why a write-off shrinks both sides at once.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — why a wave of loan losses, absorbed by a thin equity layer, is how most banks actually die.
- [The savings-and-loan crisis: interest-rate mismatch and a thousand failures](/blog/trading/banking/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures) — the Resolution Trust Corporation as the largest workout desk ever built.
- [Credit risk: the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) — the same default-and-recovery logic seen from the bond market's side, where a defaulted bond's recovery rate is priced into its spread.
- [SVB and Credit Suisse 2023: the bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — a reminder that not every bank failure is a credit failure; sometimes the wound is liquidity and rates, not bad loans.
