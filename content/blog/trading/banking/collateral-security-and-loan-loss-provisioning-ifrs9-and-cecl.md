---
title: "Collateral, Security and Loan-Loss Provisioning: IFRS 9 and CECL Explained"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank protects itself against loans going bad: secured versus unsecured lending, collateral haircuts, and the expected-credit-loss accounting (IFRS 9 staging and US CECL) that forces it to reserve for losses before they ever happen."
tags: ["banking", "credit-risk", "collateral", "loan-loss-provisioning", "ifrs9", "cecl", "expected-credit-loss", "lgd", "allowance", "charge-off"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 43
---

> [!important]
> **TL;DR** — A bank does not wait for a loan to default before it counts the loss. It collateralizes what it can, then *reserves* for the losses it statistically expects, building an allowance that quietly eats into profit long before any borrower misses a payment.
>
> - **Collateral** lowers the *size* of a loss, not the *chance* of one. A senior-secured loan loses about 25 cents on the dollar in default; the same loan unsecured loses about 45. Banks apply a **haircut** to collateral so they never count on the full market value.
> - The loss the bank pencils in is **expected credit loss = PD × LGD × EAD** — probability of default, loss given default, exposure at default. On a \$1m BBB-rated loan that is roughly \$1,080 a year; on a CCC loan it is over \$117,000.
> - Two accounting regimes force banks to reserve *ahead* of losses: **IFRS 9** stages a loan (Stage 1 = 12-month loss, Stage 2/3 = lifetime loss when risk jumps), and **US CECL** books the *lifetime* expected loss on **day one**, for every loan.
> - The one number to remember: a provision is an *expense*. A big enough provision build turns a profitable quarter into a reported loss — without a single dollar of cash leaving the bank.

In the first quarter of 2020, as the pandemic shut the global economy, JPMorgan Chase reported a net income of \$2.9 billion — down 69% from a year earlier. The bank had not lost money on trading. Its fee businesses were fine. What gutted the quarter was a single line: it set aside \$8.3 billion in provisions for credit losses, a number it built almost overnight against loans that were, on that day, still being paid on time. Across the four biggest US banks, the Q1 2020 provision build topped \$24 billion. None of those dollars had defaulted. None had even gone late. The banks were reserving for a recession they expected but had not yet seen.

That is the strange, almost philosophical heart of how a modern bank handles bad debt. A bank's whole business is lending money it does not have, against deposits it must give back on demand — borrowing short and lending long, earning the spread, surviving only as long as its thin equity cushion absorbs losses faster than they arrive. Loans going bad is not an accident that happens *to* that machine; it is a cost built *into* it, as predictable in aggregate as breakage is to a shipping company. The art is to recognize that cost early, smoothly, and honestly — to bleed a little every quarter rather than to hemorrhage all at once when the cycle turns.

This post is about the two ways a bank defends itself against a loan going bad. The first is *before* the loan is even at risk: take **collateral** — something it can seize and sell — and value it conservatively. The second is *as the loan ages*: **provision** for the loss you statistically expect, building a reserve called the **allowance**, governed by the expected-credit-loss accounting standards (IFRS 9 globally, CECL in the United States). The diagram above is the mental model for the whole journey — a loan starts performing, gets a provision booked against it, deteriorates, is charged off, and finally yields a partial recovery from its collateral.

![Pipeline showing a loan moving from performing to provision to charge-off to recovery](/imgs/blogs/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl-1.png)

## Foundations: the vocabulary of a loss before it happens

Before we go deep, let us build every term from zero. A bank's loan book is its biggest asset, and the machinery for protecting it has its own dense vocabulary. Here is the minimum set, defined plainly.

**A loan is an asset to the bank.** When you borrow \$300,000 for a house, you experience it as a liability — money you owe. The bank experiences the exact opposite: a \$300,000 asset, a promise of future cash flows (your repayments plus interest). A bank's balance sheet is the mirror image of yours. Everything in this post is about what the bank does when that asset becomes doubtful.

**Secured versus unsecured lending.** A *secured* loan is backed by **collateral** — a specific asset the borrower pledges, which the bank can seize and sell if the borrower stops paying. A mortgage is secured by the house; a car loan by the car; a margin loan by the securities in the account. An *unsecured* loan has no such pledge: a credit card balance, most personal loans, a corporate bond. If an unsecured borrower defaults, the bank joins the queue of general creditors and recovers whatever is left after the secured lenders have been paid.

**Collateral** is the pledged asset itself. Its job is not to stop a default — a borrower can still go bankrupt with a mortgaged house — but to *limit the loss* once default happens. Good collateral is easy to value, easy to seize, easy to sell, and holds its value when you most need to sell it (which, painfully, is usually a downturn, when everyone else is selling too).

**A haircut** is the discount a bank applies to collateral's market value before it will count on it. If a property is appraised at \$500,000 and the bank applies a 20% haircut, it treats the collateral as worth \$400,000 for lending purposes. The haircut is the bank admitting it cannot trust the appraisal, the market, or its own ability to sell in a hurry. A *basis point*, which we will use later, is one hundredth of a percent — 0.01%.

**Default** is the event of a borrower failing to meet the loan's terms — typically defined as being 90 days past due, or being judged "unlikely to pay" in full. Default is not the same as loss: a defaulted loan can still recover most of its value through collateral or a workout.

**Provision (or provision for credit losses)** is the *expense* a bank books on its income statement to recognize expected loan losses. It is an estimate of money the bank expects to lose, recorded as a cost *now*, before any actual loss is realized. The provision is the flow; it happens each period.

**Allowance for loan losses (or loan-loss reserve)** is the *stock* — the running balance built up by all the provisions booked over time, minus the losses already written off against it. It sits on the balance sheet as a *contra-asset*: a negative number that reduces the reported value of the loan book. If gross loans are \$1,000 and the allowance is \$30, *net* loans are \$970. The provision feeds the allowance the way water from a tap feeds a bathtub; the bathtub's level is the allowance.

**Charge-off (or write-off)** is the moment a bank gives up on a loan and removes it from its books, recognizing the loss against the allowance. Crucially, a charge-off does *not* hit the income statement — the cost was already taken when the provision was booked. The charge-off just drains the allowance (lowers the bathtub level) and removes the dead loan from gross loans.

**Recovery** is cash the bank gets back *after* a charge-off — usually from selling the collateral or from a collection effort. A recovery flows back into the allowance, refilling the bathtub a little.

**Expected credit loss (ECL)** is the headline concept of modern provisioning: a probability-weighted estimate of the loss a bank expects over a defined horizon. Its core formula, which we will use throughout, is:

$$\text{Expected loss} = PD \times LGD \times EAD$$

where **PD** is the *probability of default* (the chance the borrower defaults over the horizon), **LGD** is *loss given default* (the fraction of the exposure you lose *if* default happens, after recoveries), and **EAD** is *exposure at default* (how much the borrower will owe at the moment they default). Collateral works on the *LGD* term — it shrinks the loss, not the probability.

**IFRS 9** is the international accounting standard (effective 2018) that governs how non-US banks measure ECL. It uses a **three-stage model**: Stage 1 (performing, reserve 12 months of expected loss), Stage 2 (significant increase in credit risk, reserve *lifetime* expected loss), Stage 3 (credit-impaired/defaulted, reserve lifetime loss and stop accruing interest on the gross balance).

**CECL** — Current Expected Credit Loss — is the US standard (effective for large banks in 2020), which dispenses with staging entirely and requires banks to book the *lifetime* expected loss on **every loan from the day it is originated**. It is the more conservative of the two, front-loading losses harder.

With that vocabulary in hand, we can build the rest from first principles.

## Secured versus unsecured: collateral changes the loss, not the odds

Start with the simplest possible question: why does a bank ask for collateral at all? The naive answer is "so it gets paid back." But that is wrong, and the error matters. Collateral does *not* change the probability that a borrower defaults — a person who loses their job will miss the mortgage payment whether or not the bank holds a lien on the house. What collateral changes is *how much the bank loses when default happens*. It attacks the LGD term, not the PD term.

Think of it like a shipping company that knows a fixed percentage of its parcels will be damaged in transit. Insurance on the parcels does not stop them from being dropped; it changes how much the company is out of pocket per drop. Collateral is the bank's insurance on the loan, except the bank wrote the policy itself by demanding a pledge.

The data make this concrete. Below is the loss given default by how well the loan is secured — the share of the exposure a bank actually loses when a borrower defaults.

![Horizontal bar chart of loss given default by loan security type](/imgs/blogs/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl-2.png)

A senior-secured loan — first in line, backed by collateral — loses about 25% in default; the lender recovers three-quarters. The same borrower's *unsecured* senior debt loses about 45%. Subordinated (junior) debt, which gets paid only after senior creditors are made whole, loses around 65%. Unsecured retail credit (think credit cards) sits near 55%. The pattern is the whole point: **seniority and security are the levers that move LGD**, and a bank's entire collateral discipline exists to push every loan as far down that loss scale as it can.

This is also why the *type* of collateral matters enormously. Cash and government bonds are the gold standard — liquid, easy to value, stable. Real estate is decent but slow to sell and cyclical. Inventory, receivables, and equipment are weaker still: a warehouse of unsold goods is worth a lot less in the recession that caused the default than it was when the loan was made. The cruel feature of collateral is *correlation*: the value of the pledge tends to fall exactly when the borrower is most likely to default, because both are driven by the same downturn. A bank that lends against commercial real estate is doubly exposed — the borrower defaults *and* the collateral is worth less, at the same time, for the same reason.

#### Worked example: how a haircut turns an appraisal into a number you can trust

Suppose a small business asks for a \$350,000 loan and offers its commercial property as collateral. An appraiser values the property at \$500,000. Should the bank lend \$350,000 and feel safe, because the collateral exceeds the loan?

A disciplined bank does not. It applies a **haircut**. First, it assumes the appraisal is optimistic and that a *forced* sale — selling fast, in a weak market, to a buyer who knows you are desperate — fetches less. Say a 20% haircut: \$500,000 × (1 − 0.20) = **\$400,000**. Then it subtracts the cost of seizing and selling — legal fees, agent commissions, carrying costs, maybe 10%: \$400,000 × (1 − 0.10) = **\$360,000**. So the bank's *reliable* recovery from this \$500,000 property is \$360,000, not \$500,000.

![Waterfall bar chart of collateral value reduced by haircut and selling costs](/imgs/blogs/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl-6.png)

Now the \$350,000 loan looks different. The bank's **loan-to-value (LTV)** on the *appraised* figure is 350/500 = 70%, which sounds comfortable. But against the *reliable* recovery of \$360,000, the loan is 350/360 = 97% covered — much tighter. If the property's market value falls another 10% before a default, the bank is suddenly under-collateralized and facing a real loss. The intuition: **a haircut is the bank refusing to believe the best-case number, because by the time it needs the collateral, the best case will be long gone.**

The size of the haircut scales with how risky and illiquid the collateral is. In the repo market, where banks borrow overnight against bonds, US Treasuries take a haircut of just 1–2%, while lower-quality corporate bonds can take 15% or more — the lender demanding a bigger cushion the harder the collateral is to value and sell. The principle is identical across every secured product a bank runs.

### A hierarchy of collateral quality

Not all pledges are equal, and a bank ranks them by four properties: how easy they are to *value*, how easy to *seize*, how easy to *sell*, and how *stable* their value is in the downturn when seizure happens. The ranking, from best to worst, is roughly:

- **Cash and cash-equivalents.** A deposit pledged against a loan, or a cash margin in a trading account. The bank values it at face, can seize it instantly, never has to sell it, and its value does not move. Haircut: essentially zero. This is the platonic ideal of collateral.
- **Government bonds.** Highly liquid, marked-to-market every day, settle in a known market. Haircut on Treasuries: 1–4%, rising with maturity (a 30-year bond's price swings more than a 3-month bill's).
- **Listed equities and corporate bonds.** Liquid but volatile. A margin loan against a stock portfolio carries a substantial haircut — a broker might lend only 50% against volatile single stocks — precisely because the collateral can gap down 20% overnight.
- **Real estate.** Holds value over the long run but is illiquid, slow to seize (foreclosure takes months and is legally fraught), expensive to sell, and *cyclical*. Mortgage LTVs of 80% embed a 20% implicit haircut; commercial real estate often demands more.
- **Inventory, receivables, equipment.** Operating assets of a business. These are the weakest collateral because they are hardest to value, often perishable or obsolescing, and — critically — worth the least in the recession that triggered the default. A warehouse of unsold widgets fetches pennies in a fire sale.
- **Intangibles and going-concern value.** Brand, patents, future cash flows. Barely collateral at all in a liquidation, because they evaporate the moment the business stops operating.

The single most important property in that list is the last column — *stability in the downturn* — because that is when collateral is actually called upon. This is why cash and Treasuries dominate the secured-lending and repo markets: they are the only collateral whose value does *not* fall in the crisis that makes you seize it.

### Collateral is not "set and forget"

A subtle point that trips up newcomers: collateral has to be *monitored* for the life of the loan, not just valued at origination. A margin loan is re-marked daily, and if the collateral's value drops below a threshold the borrower gets a **margin call** — a demand to post more collateral or repay. A commercial real-estate loan may have a covenant requiring periodic re-appraisal, and a fall in the property's value can trip a *loan-to-value covenant* that forces the borrower to pay down principal. The bank that takes collateral at origination and never looks again is the bank that discovers, the morning the borrower defaults, that the collateral quietly halved in value over the preceding year. Collateral management — re-valuation, margin calls, perfection of the legal claim — is a continuous operational discipline, not a one-time check at underwriting.

## Expected credit loss: the formula a bank lives by

Collateral handles the *size* of a loss. But a bank has to put a number on the *whole* expected cost of lending, default chance and all — and that number is expected credit loss. This is the single most important formula in credit risk, and it is worth slowing down on, because every provision a bank books is ultimately an application of it.

$$\text{EL} = PD \times LGD \times EAD$$

Read it as three independent questions about a loan. *How likely is the borrower to default?* (PD.) *If they default, how much of what they owe do I lose?* (LGD — and this is where collateral lives.) *How much will they owe at that moment?* (EAD.) Multiply the three and you get the loss you should *expect*, on average, across many loans like this one.

The word "expect" is doing heavy lifting. Expected loss is not a prediction about any *single* loan — a given loan either defaults or it does not. It is the average outcome across a portfolio. If you make 1,000 identical loans each with a 1% PD and you expect to lose 45% of the exposure on the ones that default, you do not lose 1% of each loan; you lose 100% (of the LGD portion) on roughly 10 of them and nothing on the other 990. Expected loss smooths that into a per-loan number you can price and reserve against. This is exactly how an insurer thinks: it cannot tell you which house will burn down, but it can tell you, across 100,000 houses, how much it will pay out — and charge premiums accordingly.

The PD term varies enormously with credit quality. Here is the one-year expected loss on a \$1,000,000 senior-unsecured loan (LGD fixed at 45%) across borrower rating grades, computed directly from typical default rates.

![Bar chart of one-year expected loss in dollars by borrower rating grade](/imgs/blogs/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl-3.png)

The chart is on an ordinary scale, and notice how flat the left side is and how violently it spikes on the right. An AAA loan's expected loss is \$45 a year — a rounding error. A BBB loan (the lowest *investment-grade* rung) is about \$1,080. A BB loan ("junk", just below investment grade) jumps to \$5,400. A B loan is \$24,750. A CCC loan — a borrower already in serious trouble — has an expected loss of **\$117,000 a year on a million-dollar loan**, because its one-year default probability is around 26%. The exponential climb is why credit pricing is so sensitive to rating, and why a downgrade from BBB to BB roughly *quintuples* the loss a bank must reserve against the same loan.

#### Worked example: the same loan, secured and unsecured

Let us see all three terms move at once. A bank lends \$1,000,000 to a BB-rated company (one-year PD ≈ 1.2%). The exposure at default is the full \$1,000,000 (a simple term loan, fully drawn).

*Unsecured.* LGD is 45%. Expected loss = 0.012 × 0.45 × \$1,000,000 = **\$5,400 per year**.

*Secured by good collateral.* The borrower pledges equipment and receivables worth, after haircuts, enough to push LGD down to 25% (senior-secured). Same PD, same EAD: expected loss = 0.012 × 0.25 × \$1,000,000 = **\$3,000 per year**.

The collateral did not touch the 1.2% default probability at all — the company is exactly as likely to fail. But it cut the *expected loss by 44%*, from \$5,400 to \$3,000, purely by improving recovery. That \$2,400 difference is what the bank can either pocket as lower risk or pass on to the borrower as a lower interest rate. The intuition: **collateral is a discount on your losses, not a reduction in your chance of having them — and over a whole portfolio that discount is worth real money.**

There is a subtlety in EAD worth flagging. For a fully-drawn term loan, EAD is just the balance. But for a *revolving* facility — a credit card, a corporate line of credit — the borrower can draw *more* as they head toward default (a distressed company maxes out every line it has). So EAD must include a *credit conversion factor* on the undrawn portion: if a company has drawn \$600,000 of a \$1,000,000 line and the bank assumes 75% of the remaining \$400,000 gets drawn before default, EAD = \$600,000 + 0.75 × \$400,000 = \$900,000, not \$600,000. Underestimating EAD on revolvers is a classic way banks understate their risk.

## The provisioning waterfall: how a loss flows through a bank

Now we can assemble the machinery. A loan does not jump from "fine" to "written off" in one step. It flows through a waterfall, and each stage touches the income statement and balance sheet differently. This is where most people's intuition about bank losses is wrong, so let us trace it carefully with numbers.

**Step 1 — the loan performs.** The bank holds a \$1,000,000 loan as a gross asset. From day one (under CECL) or once risk rises (under IFRS 9), it has booked some allowance against it. Say the allowance is \$30,000. The *net* carrying value is \$970,000.

**Step 2 — the bank provisions.** Each period, the bank estimates expected losses on its whole book and tops up the allowance to the right level. If the estimate rises, it books a **provision expense** on the income statement. Suppose deterioration pushes the required allowance to \$80,000. The bank books a \$50,000 provision: profit falls by \$50,000 *this quarter*, and the allowance grows from \$30,000 to \$80,000. No cash has moved. The loan is still being paid. But reported earnings just took a \$50,000 hit.

**Step 3 — the loan defaults and is charged off.** The borrower stops paying. After workout efforts fail, the bank charges off \$80,000 of the loan as uncollectible. This is the counterintuitive part: **the charge-off does not hit the income statement.** The cost was already recognized when the provision was booked. The charge-off simply *uses up* the allowance — the allowance drops from \$80,000 back toward \$0 for this loan — and removes the dead exposure from gross loans. The income statement is untouched at the charge-off moment.

**Step 4 — recovery.** Later, the bank seizes and sells the collateral, recovering \$60,000. That \$60,000 flows back into the allowance (or is booked as a recovery), partially refilling the reserve and reducing the *net* loss on the loan to \$20,000.

This sequence — provision (P&L expense) → build allowance (balance sheet) → charge-off (drains allowance, no P&L) → recovery (refills allowance) — is the loan-loss waterfall, and getting the *timing* right is everything. The whole design pushes the pain *forward*, to the provision, so that by the time the actual loss crystallizes the bank has already absorbed it. The figure below shows how a single provision expense splits and flows through both financial statements at once.

![Graph of a provision flowing to the income statement and the balance sheet allowance](/imgs/blogs/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl-7.png)

The key insight here connects straight back to the spine of how a bank lives or dies. Because the provision hits *equity* (through retained earnings) the moment it is booked, provisioning is a direct draw on the thin capital cushion that keeps the bank solvent. A bank that under-provisions in good times is borrowing capital from its future self; when the cycle turns and reality forces a catch-up build, the provision avalanche can wipe out years of profit — and a bank's equity is only ever a few years of profit thick. For the mechanics of how that allowance interacts with the rest of the income statement, the companion post on [the income statement of a bank](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions) walks the full P&L; for what happens *after* charge-off — restructuring, the workout desk, recoveries — see [non-performing loans and the workout process](/blog/trading/banking/non-performing-loans-and-the-workout-process).

#### Worked example: the provision build that turns a profit into a loss

Here is the JPMorgan-in-miniature scenario — how provisioning alone, with no cash lost, can flip a bank's quarter from green to red.

A mid-size bank has a \$10 billion loan book. In a normal quarter it earns **pre-provision operating profit** (revenue minus costs, before any credit charge) of \$300 million. In good times it provisions \$50 million a quarter — its expected-loss run-rate — leaving a net profit of \$250 million.

Now a recession arrives. The bank's models say lifetime expected losses across the book have jumped from 2.0% to 3.5% of loans. The required allowance rises from \$200 million to \$350 million — a \$150 million gap that must be filled *this quarter*. So the provision this quarter is the normal \$50 million *plus* the \$150 million catch-up build = **\$200 million**.

Net profit = \$300 million pre-provision − \$200 million provision = **\$100 million**. Still positive, but down 60%. And if the downturn is severe and the models push expected losses to 5%, the required allowance goes to \$500 million, a \$300 million build, total provision \$350 million — and now: \$300 million − \$350 million = **−\$50 million**. The bank reports a *loss*, despite its core business earning \$300 million, despite not a single dollar of cash leaving the building, despite most of those reserved-for loans still being paid on time. The intuition: **a provision is the bank pre-paying for a recession out of this quarter's earnings — and a deep enough recession can cost more than a quarter earns.**

This is why bank earnings are so violently *pro-cyclical*: profits look fat late in the boom (low provisions, sometimes even reserve *releases* that add to profit) and crater early in the bust (huge provision builds), often well before the real defaults show up. The accounting front-runs the credit cycle — which is exactly what it is designed to do.

## IFRS 9: the three-stage model

For most of banking history, the rule was simple and dangerous: book a loss only when it is *incurred* — when there is concrete evidence a specific loan has gone bad. This "incurred loss" model meant banks kept thin reserves through the boom and were forced into enormous, sudden provisions only after the crisis hit. In 2008, this was a disaster: reserves were too small, too late, and the provision avalanche in 2008–2009 amplified the very downturn it was reacting to. Regulators called it "too little, too late." Both the international and US standard-setters responded by moving to *expected* loss — reserve for losses you foresee, not just losses you have already taken. IFRS 9 was the international answer.

IFRS 9 puts every loan into one of three **stages**, and the stage determines the size of the reserve.

**Stage 1 — performing.** The loan's credit risk has not increased significantly since it was made. The bank reserves the **12-month expected loss** — the portion of lifetime ECL attributable to defaults *possible in the next 12 months*. This is a small reserve. Interest income is recognized on the gross carrying amount. The vast majority of a healthy bank's loans live here.

**Stage 2 — significant increase in credit risk (SICR).** Something has gone wrong: the borrower has been downgraded, gone 30+ days past due, breached a covenant, or otherwise shown materially higher default risk — but has not yet actually defaulted. The reserve jumps to the **lifetime expected loss** — the ECL over the *entire remaining life* of the loan. This is the cliff. A loan can move from Stage 1 to Stage 2 in a single reporting period, and when it does, the reserve can multiply several times over, even though the loan is still being paid. Interest is still recognized on the gross balance.

**Stage 3 — credit-impaired.** The loan has defaulted or there is objective evidence of impairment. The reserve remains at lifetime ECL, but now interest is recognized only on the *net* carrying amount (gross minus the allowance) — the bank stops booking interest income it does not expect to collect. This is the accounting catching up to economic reality.

The Stage 1 → Stage 2 transition is the most consequential mechanic in the standard, because it is where the reserve cliff lives. The figure below shows the jump.

![Before and after comparison of IFRS 9 Stage 1 and Stage 2 reserves](/imgs/blogs/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl-4.png)

#### Worked example: a Stage 1 reserve versus a Stage 2 reserve on the same loan

Take a \$1,000,000 corporate loan with five years left to run. While it is performing (Stage 1), the bank reserves only the 12-month expected loss. Suppose the one-year PD is 0.5% and LGD is 45%: 12-month ECL = 0.005 × 0.45 × \$1,000,000 = **\$2,250**. Round trip: a tiny reserve against a million-dollar loan.

Now the borrower is downgraded and goes 45 days past due. This is a *significant increase in credit risk*, so the loan moves to Stage 2 and the reserve must cover the **lifetime** expected loss. Lifetime PD over the five remaining years — accounting for the elevated risk — is, say, 8%. Lifetime ECL = 0.08 × 0.45 × \$1,000,000 = **\$36,000**.

The reserve has jumped from \$2,250 to \$36,000 — a **16× increase** — on a loan that is still being paid, simply because its *risk* rose enough to trip the SICR threshold. (My cover figure uses round 0.5%/4% numbers for an 8× jump; the exact multiple depends on the lifetime PD you assume — the point is the *cliff*, not the precise factor.) The intuition: **under IFRS 9 the painful provision is taken when risk rises, not when default happens — and because the jump from 12-month to lifetime loss is a cliff, a wave of loans migrating to Stage 2 in a downturn drives a sudden, lumpy provision surge.**

That lumpiness — the cliff effect — is IFRS 9's main criticism. Because reserves jump discontinuously at the Stage 1/2 boundary, a small worsening of the economy that pushes many loans across the SICR threshold at once can trigger an outsized, simultaneous provision build. Banks and regulators spent the COVID period worrying that mechanical SICR triggers would force pro-cyclical over-provisioning, and many applied judgment-based overlays to smooth it.

### What actually trips a loan into Stage 2

The "significant increase in credit risk" trigger is not a single number — it is a basket of indicators a bank must monitor and apply judgment to. The most common triggers are:

- **A rating downgrade.** The standard compares the loan's *current* lifetime PD to its lifetime PD *at origination*. If the relative increase crosses a bank-set threshold (often a doubling, or a move of two or more rating notches), the loan moves to Stage 2.
- **Days past due.** IFRS 9 contains a *rebuttable presumption* that 30 days past due signals a significant increase in risk. A bank can rebut it with evidence, but 30 DPD is the default tripwire — well before the 90-DPD definition of default.
- **Forbearance.** If the bank has had to modify the terms (extend the maturity, cut the rate, grant a payment holiday) because the borrower was struggling, that concession is itself evidence of elevated risk.
- **Watchlist placement.** A loan flagged by the bank's internal credit monitoring — a covenant breach, a profit warning, a sector in distress — typically moves to Stage 2.

The judgment in setting these thresholds is enormous, and it is where two banks diverge. A conservative bank trips loans into Stage 2 early and carries a heavier Stage 2 book; an aggressive one keeps loans in Stage 1 as long as it can defend, flattering its reserves. Analysts watch the *Stage 2 ratio* — the share of the book in Stage 2 — and a sudden migration of loans from Stage 1 to Stage 2 is one of the earliest warning signs that a bank's credit quality is turning, often visible quarters before the Stage 3 (defaulted) numbers move.

### Expected loss is probability-weighted, not a single scenario

One more layer that both IFRS 9 and CECL demand: the expected loss must be a *probability-weighted* estimate across multiple economic scenarios, not a single best-guess forecast. A bank cannot just run its base-case economy through the models; it must run an optimistic, a base, and a pessimistic scenario, then weight them.

#### Worked example: weighting three economic scenarios

A bank models the lifetime expected loss on a \$1,000,000 loan under three futures. In a *benign* economy, lifetime ECL is \$8,000. In a *base* case, it is \$20,000. In a *severe* downturn, it is \$70,000. The bank assigns probabilities of 25% benign, 55% base, and 20% severe.

The probability-weighted ECL is: 0.25 × \$8,000 + 0.55 × \$20,000 + 0.20 × \$70,000 = \$2,000 + \$11,000 + \$14,000 = **\$27,000**.

Notice that \$27,000 is *higher* than the base-case \$20,000. That is not a mistake — it is the whole point. Because loss outcomes are skewed (the severe downturn loses far more than the benign one gains), the probability-weighted average sits above the central case. The intuition: **expected credit loss is deliberately pulled toward the bad tail, so a bank with even a modest probability on a severe scenario reserves more than its base-case forecast alone would suggest — which is exactly why provision builds spike the moment a recession enters the scenario set, even before it arrives.**

This scenario-weighting is also why the *same* loan can carry wildly different reserves at two banks: it depends entirely on which scenarios they model and what probabilities they assign. The allowance is, in this sense, a window into a bank's view of the future — and a place where an over-optimistic management team can quietly under-reserve.

## CECL: lifetime loss from day one

The United States went further. Under **CECL** — Current Expected Credit Loss, effective for large US banks from January 2020 (timing that turned out to be brutal) — there is no staging at all. A bank must estimate the **lifetime** expected credit loss on a loan **the moment it originates it**, and reserve that full amount immediately. A brand-new, perfectly performing 30-year mortgage carries a day-one reserve for the losses the bank expects over all 30 years.

This is the most conservative provisioning regime in the world, and the contrast with IFRS 9 is stark. Under IFRS 9, a healthy new loan sits in Stage 1 with only a 12-month reserve; it has to *deteriorate* before it gets a lifetime reserve. Under CECL, every loan gets a lifetime reserve on day one, deterioration or not. The figure below lays the two regimes side by side across the dimensions that matter.

![Matrix comparing IFRS 9 stages and US CECL on trigger horizon and measurement](/imgs/blogs/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl-5.png)

The practical consequence is that CECL front-loads losses onto the *origination* of a loan rather than its deterioration. This has a perverse incentive baked in: under CECL, *growing* the loan book is expensive, because every new loan demands an immediate lifetime reserve that hits earnings the quarter the loan is made — even though the interest income from that loan arrives over years. A bank that doubles its lending in a quarter takes a double provision hit up front. Critics argue this discourages lending exactly when the economy needs it (early in a recovery), making CECL *pro-cyclical in origination* even as it is *counter-cyclical in reserving*.

#### Worked example: the same new loan under IFRS 9 and CECL

A bank originates a \$500,000 small-business loan, five-year term, with a lifetime PD of 6% and an LGD of 40% (partially secured). The one-year PD is 1%.

*Under IFRS 9 (Stage 1, performing).* Day-one reserve = 12-month ECL = 0.01 × 0.40 × \$500,000 = **\$2,000**. The bank books a \$2,000 provision at origination.

*Under CECL.* Day-one reserve = lifetime ECL = 0.06 × 0.40 × \$500,000 = **\$12,000**. The bank books a \$12,000 provision at origination — six times the IFRS 9 figure — on the *exact same loan*, on the day it is made, before anything has gone wrong.

Multiply that gap across a \$50 billion loan book and the difference in day-one reserves runs into the hundreds of millions. When CECL took effect on January 1, 2020, US banks booked large one-time increases to their allowances on transition — and then, ten weeks later, COVID forced them to apply CECL's lifetime-loss logic to a sudden recession, which is precisely why the Q1 2020 provision builds were so enormous. The intuition: **CECL makes a bank pay for the whole life of a loan's risk on the day it writes it, so its reserves are bigger and steadier but its growth is more expensive — a different trade-off than IFRS 9's wait-and-jump.**

It is worth being honest about the modeling here. Estimating a *lifetime* expected loss requires forecasting the economy years out — unemployment, GDP, house prices — and feeding those forecasts through models that translate macro scenarios into PDs and LGDs. Two banks with identical loan books can report materially different allowances simply because their economic forecasts and model assumptions differ. The allowance is therefore one of the most *judgmental* numbers on a bank's balance sheet, and one of the first places a skeptical analyst looks for either hidden weakness (an under-reserved bank flattering its earnings) or sandbagging (an over-reserved bank stashing profit for a rainy day via reserve releases later).

## Provisions through the credit cycle

Step back and watch the whole thing breathe over a cycle, because this is where provisioning's character really shows. Credit losses are not random noise sprinkled evenly through time — they cluster in waves, driven by the macroeconomy. The visible tail of those waves is bank failures, and the chart below shows them clustering exactly as the credit cycle does.

![Bar chart of FDIC insured bank failures per year from 2005 to 2025](/imgs/blogs/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl-8.png)

The 2008–2012 wave is unmistakable: from zero failures in 2005–2006 to 25 in 2008, 140 in 2009, and a peak of 157 in 2010, then a long tail down. A smaller, sharper episode hit in 2023 (Silicon Valley Bank, Signature, First Republic). The lesson of provisioning is that the *reserve build should lead this curve, not lag it.* A well-run bank under an expected-loss regime is raising provisions in 2007 and early 2008 — while the failures chart still reads near zero — so that by the time the wave crests it has already absorbed most of the pain. A bank still reserving thinly into 2009 is the one that gets caught.

This is also where the *reserve release* mechanic appears, and it is easy to misread. As a recovery takes hold and the economic outlook brightens, a bank's models say expected losses are *falling*. The required allowance drops, so the bank *releases* reserves — and a release is a *negative provision*, which *adds* to profit. In 2021, as the post-COVID outlook improved, the big US banks released tens of billions of reserves they had built in 2020, and those releases flattered 2021 earnings substantially. A naive reader saw record bank profits; a careful one saw that a large slice of those profits was the *reversal* of 2020's provisions, not new operating earnings. Reserve releases are real, legitimate accounting — but they are a one-time tailwind, not a sustainable earnings stream, and conflating the two is a classic analytical error.

The deep point connects to the spine. A bank earns its spread continuously, but its losses arrive in lumps, concentrated in downturns. Provisioning's entire job is to *convert lumpy losses into a smooth annual cost* — to make each good year pay a fair share of the inevitable bad one. When it works, a bank glides through the cycle. When banks game it — under-reserving in the boom to flatter earnings and pay bonuses — they are quietly thinning the very capital cushion that is supposed to absorb the bust, and the maturity-transformation machine becomes that much more fragile. The accounting standard is, in the end, a discipline imposed on the temptation to pretend the good times will last.

There is one more cyclical wrinkle worth understanding, because it is where expected-loss accounting fights its own design. Expected-loss models are typically *point-in-time*: they reflect the bank's current view of the economy. That makes the allowance responsive — it rises as soon as the outlook darkens — but it also makes it *procyclical* in a specific way. In the boom, models see low PDs and a benign outlook, so reserves shrink and even release, boosting reported profit exactly when the bank should be cautious. In the bust, models see high PDs and a grim outlook, so reserves balloon, depressing earnings exactly when the bank can least afford it and credit is most needed. The accounting genuinely *leads* the realized-loss cycle — that is the improvement over the old incurred-loss model — but it can also *amplify* the swing in reported earnings and capital. Regulators have spent the years since 2018 debating "through-the-cycle" overlays, transitional capital relief on day-one CECL reserves, and the right way to dampen the cliff, precisely because a provisioning system that is too responsive to the latest forecast can shove a bank's capital around as violently as the losses it is meant to smooth.

The honest summary is that there is no free lunch in the timing of loss recognition. Recognize losses late (incurred-loss) and you get a flat, false calm followed by a crash. Recognize them early and responsively (expected-loss) and you get reserves that lead the cycle but lurch with every change in the forecast. The standard-setters chose the second trade-off after 2008, judging that a bank caught with adequate-but-volatile reserves is far safer than one caught with thin-but-stable ones. Provisioning, in the end, is the institutional memory of that lesson — the rule that forces a bank to admit, every single quarter, that some of the loans on its books will not be repaid, and to start paying for that fact today.

## The allowance, capital, and the two layers of defense

There is a deep structural point that ties this whole topic to a bank's solvency, and it confuses even experienced people, so let us be precise. A bank has *two* distinct layers of defense against loan losses, and they are not the same thing.

The **first layer is the allowance** — the reserve we have spent this whole post building. It is there to absorb the losses the bank *expects*. When a loan goes bad and is charged off, it is the allowance that takes the hit, not the bank's equity directly. The allowance is calibrated to *expected* loss: the average, the central tendency, the losses you can see coming.

The **second layer is capital (equity)** — the thin cushion of shareholder money discussed across this series. Capital is there to absorb the losses the bank did *not* expect — the *unexpected* loss, the tail, the recession worse than the worst scenario you modeled. When losses blow through the allowance (because reality turned out worse than the expected-loss estimate), the excess eats into equity, and when equity runs out, the bank is insolvent.

This is the cleanest way to hold the whole risk architecture in your head: **the allowance covers expected loss; capital covers unexpected loss.** Provisioning fills the first bucket out of earnings, period by period; capital is held against the possibility that the first bucket is not big enough. Regulators size capital requirements (under Basel) specifically against *unexpected* loss, on the explicit assumption that expected loss is already covered by provisions. If a bank's allowance is too thin, that assumption breaks, and the capital ratio overstates the bank's true safety — which is why a skeptical analyst always cross-checks the adequacy of the allowance before trusting the capital ratio.

#### Worked example: when losses blow through the allowance into capital

A bank has a \$10 billion loan book, a \$300 million allowance (3% coverage), and \$900 million of equity. Its models said expected lifetime losses were 3% — hence the \$300 million reserve.

A severe recession hits and *actual* losses come in at 6% of the book — \$600 million, double the expected figure. The first \$300 million is absorbed by the allowance, exactly as designed: those charge-offs drain the reserve and never touch this period's earnings or equity. But the *next* \$300 million has no allowance left to absorb it. The bank must book additional provisions of \$300 million to cover the excess, and *that* expense flows straight through the income statement into retained earnings — cutting equity from \$900 million to \$600 million.

The bank has lost a third of its capital in one cycle, not because it failed to reserve, but because losses exceeded the *expected* level the allowance was sized for. The intuition: **the allowance is the moat and capital is the keep — and a bank dies when the losses are big enough to cross both. The expected-loss accounting in this post fills the moat; everything in the capital and Basel rules exists to keep the keep standing when the moat is overrun.**

This two-layer structure is also the answer to a question that puzzles many people: if a bank already reserves for expected losses, why does it *also* need capital against credit risk — isn't that double-counting? It is not, because the two cover different parts of the loss distribution. The allowance covers the *mean*; capital covers the *variance* — the gap between an average bad year and a catastrophically bad one. The history of bank failures is overwhelmingly a history of institutions that got *both* wrong at once: thin reserves *and* thin capital, going into a downturn worse than they planned for.

## Common misconceptions

**"A charge-off is when the bank takes the loss."** No — the loss is taken when the *provision* is booked, which can be quarters or years earlier. The charge-off merely confirms a loss the income statement already absorbed, by draining the allowance. This is the single most common error about bank accounting. When you read that a bank "charged off \$2 billion," that \$2 billion did *not* hit that quarter's earnings — it had already been provisioned. What hits earnings is the *provision*, not the charge-off.

**"Collateral means the bank can't lose money."** Collateral reduces LGD, not PD, and it reduces LGD imperfectly. Haircuts exist precisely because collateral is worth less than its appraisal when you actually need to sell it, and worth least in the downturns that cause defaults. A mortgage lender in 2008 held plenty of collateral; it still lost a fortune because house prices fell 30% while defaults soared — the collateral and the borrower failed together. Over-collateralized on paper is not the same as protected in a crisis.

**"Bigger provisions mean the bank is in trouble."** Sometimes the opposite. A bank that builds reserves *early* and *generously* — taking the pain in its earnings before the cycle turns — is being prudent, and its reported losses may look worse than a peer that is under-reserving to flatter its numbers. The dangerous bank is often the one with *suspiciously low* provisions late in a boom, not the one taking a big build. Read the *coverage ratio* (allowance ÷ non-performing loans), not just the provision headline.

**"IFRS 9 and CECL are basically the same thing."** They share the expected-loss philosophy but differ in a way that produces materially different numbers. IFRS 9 reserves 12-month loss until risk rises, then jumps to lifetime; CECL reserves lifetime loss from day one for every loan. CECL is structurally more conservative — bigger, steadier reserves — but more expensive for loan growth. The same loan can carry a reserve several times larger under CECL than under IFRS 9 in its early, healthy years. They are not interchangeable.

**"Expected loss tells you what this loan will lose."** Expected loss is a *portfolio* average, not a forecast for any single loan. A given loan either defaults or it does not; "expected loss of \$5,400" does not mean this loan loses \$5,400. It means that across many loans like it, the *average* loss is \$5,400. Using EL as a point prediction for one borrower is a category error — it is the same mistake as thinking a life insurer expects *you* specifically to die at the actuarial average age.

## How it shows up in real banks

**JPMorgan, Q1 2020.** As discussed, the bank's net income fell 69% to \$2.9 billion on an \$8.3 billion provision build, almost none of which represented realized losses. It was a textbook CECL-plus-recession event: lifetime-loss accounting applied to a sudden, severe downturn forecast forced an enormous up-front reserve. The episode is the cleanest real-world illustration that provisions, not charge-offs, drive bank earnings volatility.

**The 2021 reserve releases.** A year later, the same dynamic ran in reverse. As the recovery proved faster than the 2020 forecasts, JPMorgan and peers released large chunks of those reserves — negative provisions that *added* billions to 2021 profit. The headline "record bank profits" of 2021 was substantially a reserve-release artifact. Analysts who adjusted for it saw underlying earnings that were strong but not the records the headlines implied. This is the pro-cyclicality of provisioning made visible within 18 months.

**The 2008 incurred-loss failure.** The crisis that motivated both IFRS 9 and CECL. Under the old incurred-loss model, banks could not reserve for losses they foresaw but had not yet "incurred." Reserves were thin going into 2008; when losses hit, the forced provisioning was massive and procyclical, deepening the credit crunch. The standard-setters' explicit goal in moving to expected loss was to make banks reserve *earlier* — to front-load the discipline so the next crisis would not catch reserves so flat-footed.

**Commercial real estate, 2023–2024.** A live example of the collateral-correlation problem. As office values fell sharply post-pandemic and interest rates rose, banks with heavy commercial-real-estate exposure faced the double hit: borrowers struggling to refinance (rising PD) *and* the buildings backing those loans worth far less (rising LGD). New York Community Bancorp's surprise \$552 million provision in early 2024 — and the share-price collapse that followed — was the market discovering that its CRE collateral and its CRE borrowers were deteriorating together. Collateral that looked ample at origination evaporated exactly when it was needed.

**The savings-and-loan crisis.** A historical bookend. Through the 1980s, thrifts held long-term fixed mortgages funded by short-term deposits, and when rates spiked the spread inverted. The losses were not primarily a provisioning-discipline failure but a duration mismatch — yet the episode shows how, without adequate reserves and capital, a wave of credit and rate losses can take down over a thousand institutions (the S&L crisis saw roughly 1,000 thrift failures and a \$124 billion taxpayer cleanup). The expected-loss regimes exist so that the *credit* portion of such waves is reserved against in advance.

**The 2008 mortgage collapse and collateral correlation.** Washington Mutual, the largest US bank failure in history at \$307 billion of assets, was destroyed by exactly the collateral-correlation trap described earlier. WaMu had originated vast volumes of option-ARM and subprime mortgages, all secured by houses. The "secured" label gave a false sense of safety: when US house prices fell roughly 30% from their 2006 peak, the collateral backing every one of those loans fell at the same time, while the borrowers — many of whom could only afford the teaser rates — defaulted en masse. PD and LGD spiked *together*, for the same reason, across the entire book. No amount of collateral helps when the collateral and the borrower fail in the same downturn. WaMu was seized by regulators in September 2008 and sold to JPMorgan for \$1.9 billion — a fraction of the loan losses that followed.

**European banks and the IFRS 9 COVID overlays.** When COVID struck in 2020, European banks operating under IFRS 9 faced the cliff effect head-on: mechanical SICR triggers threatened to push huge swathes of loans from Stage 1 to Stage 2 simultaneously, forcing a procyclical provision surge just as the economy needed credit. Regulators (the European Central Bank and the European Banking Authority) explicitly told banks *not* to apply the triggers mechanically, to look through the temporary shock, and to weight their economic scenarios sensibly. Banks layered on judgmental "post-model overlays" — manual adjustments to the modeled allowance — that in some cases ran into billions. The episode is a candid admission that expected-loss accounting, for all its discipline, depends on judgment at exactly the moments when judgment is hardest, and that the cliff effect is a real design flaw the system manages with discretion rather than formula.

## The takeaway: how to use this

If you remember one thing, remember that a bank's reported loan losses tell you almost nothing on their own — what tells you something is the *timing* and *adequacy* of its provisioning relative to where it is in the cycle. The provision is the live signal; the charge-off is old news.

So when you read a bank's results, do three things. First, separate the **provision** (the forward-looking expense that hits earnings) from the **charge-off** (the backward-looking confirmation that drains the already-built allowance) — they are not the same event and they hit different statements. Second, watch the **coverage ratio** (allowance as a percentage of loans, and allowance relative to non-performing loans): a bank whose coverage is *falling* while the economy weakens is borrowing earnings from its future, and a bank whose coverage is *rising* ahead of the cycle is paying its dues early. Third, treat **reserve releases** as a one-time tailwind, never as sustainable earnings — a bank "earning" its way back to profit by releasing reserves is not the same as one growing its spread.

And keep the spine in view. A bank survives only as long as its thin equity cushion absorbs losses faster than they arrive. Provisioning is the mechanism that does exactly that — it pulls the future's losses into the present, smooths the lumpy credit cycle into a steady annual cost, and protects the capital that keeps the maturity-transformation machine alive. Collateral, for its part, is the bank refusing to count on more than it can actually seize and sell. Together they are the difference between a bank that bleeds a little every quarter and survives, and one that pretends the loans are fine right up until the morning it discovers they never were. The banks that fail almost always fail this exact test — not because they could not see the losses coming, but because they chose not to reserve for them until it was too late.

## Further reading & cross-links

- [The income statement of a bank: net interest income, fees and provisions](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions) — where the provision line sits in the P&L and how it interacts with pre-provision operating profit.
- [Non-performing loans and the workout process](/blog/trading/banking/non-performing-loans-and-the-workout-process) — what happens *after* a loan goes bad: classification, restructuring, the workout desk, write-offs and recoveries.
- [Credit risk management: PD, LGD, EAD and expected loss](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss) — the credit-risk engine in full, through-the-cycle versus point-in-time estimation, and the credit cycle.
- [Credit risk: the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) — the same expected-loss logic from the bondholder's side of the market.
- [The asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — why getting ahead of losses, as provisioning tries to, matters so much more than chasing gains.
