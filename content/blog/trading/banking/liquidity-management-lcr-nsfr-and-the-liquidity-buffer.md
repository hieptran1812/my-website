---
title: "Liquidity Management: LCR, NSFR, and the Liquidity Buffer"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a bank that is rich on paper can still die in 36 hours, and how the Basel III liquidity rules build a buffer designed to outlast the panic."
tags: ["banking", "liquidity", "lcr", "nsfr", "hqla", "basel-iii", "bank-run", "treasury", "funding", "liquidity-buffer", "run-off-rate", "svb"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Liquidity is not solvency: a bank can have more assets than debts and still die because it cannot turn those assets into cash fast enough to pay depositors who all want out at once. The Basel III liquidity rules force every bank to carry a buffer of liquid assets sized to outlast a 30-day panic and to fund itself stably over a year.
>
> - **Solvency** asks "are the assets worth more than the debts?"; **liquidity** asks "can you find the cash *today*?" The two are different questions, and the second one kills faster.
> - The **Liquidity Coverage Ratio (LCR)** = high-quality liquid assets ÷ the net cash you would lose in a 30-day stress, and it must be **at least 100%**.
> - The **Net Stable Funding Ratio (NSFR)** = available stable funding ÷ required stable funding over a one-year horizon, and it must also be **at least 100%**.
> - The one number to remember: **both ratios floor at 100%** (`db.LIQUIDITY_MIN`). Below 100%, the regulator treats you as living on borrowed time.

On the morning of 9 March 2023, Silicon Valley Bank was, by the accounting that bankers and auditors use, alive. Its assets — loans, US Treasuries, and government-backed mortgage bonds — were on the books at a value comfortably above its deposits and other debts. It had a positive equity cushion. No regulator had declared it insolvent. And yet, over the next 36 hours, depositors tried to pull about \$42 billion out of it in a single day, with another roughly \$100 billion queued to leave the next morning. By Friday, 10 March, the bank was gone — seized by the FDIC, the fastest large-bank failure in modern American history.

SVB did not die because its assets were worthless. It died because it could not turn those assets into cash fast enough to hand back to the people standing at the (digital) door. Its bonds were good. They were just *long* — locked up for years, and worth less than their purchase price once interest rates had risen — and a bank cannot pay a fleeing depositor with a ten-year bond that matures in 2032. The cash ran out before the assets could be sold without crushing losses. That gap — between *owning value* and *having cash* — is the single most important idea in bank liquidity, and it is the reason this post exists.

The diagram above is the mental model for the whole topic: on the left, a bank that is genuinely solvent — its assets are worth more than its debts; on the right, the same bank a day later, unable to find the cash to meet a run, dying while still solvent. Everything that follows — the buffer, the ratios, the run-off rates, the 30-day stress — is the machinery banks and regulators built to stop the right-hand side of that figure from happening.

![Two columns showing a bank solvent on its balance sheet but unable to pay fleeing depositors](/imgs/blogs/liquidity-management-lcr-nsfr-and-the-liquidity-buffer-1.png)

## Foundations: liquidity, solvency, HQLA, and the two ratios

Before we can manage liquidity, we have to be precise about what it is — because in everyday speech "the bank is in trouble" blurs two completely different failures together. Let us build the vocabulary from zero, defining every term the first time it appears.

Here is the simplest possible analogy, and we will keep coming back to it. Imagine you are wealthy: you own a \$2 million house outright and have \$3,000 in your checking account. On paper you are rich — your *net worth* (everything you own minus everything you owe) is over \$2 million. Now suppose a \$50,000 bill lands on Monday and is due Friday. You cannot pay it. You are *solvent* (your assets dwarf your debts) but *illiquid* (you cannot lay your hands on cash quickly). You could sell the house — but not by Friday, not without dumping it at a fire-sale price. That gap between "rich on paper" and "broke this week" is exactly what kills banks. A bank is just this story at enormous scale and enormous speed.

**Solvency** is a statement about *value over time*: are your total assets worth more than your total liabilities (everything you owe)? If yes, your *equity* — assets minus liabilities, the owners' stake and the loss-absorbing cushion — is positive, and you are solvent. Solvency erodes slowly, as losses pile up over quarters.

**Liquidity** is a statement about *cash right now*: can you meet the payments coming due today and this week, in actual money, without selling good assets at a loss? Liquidity can evaporate in hours.

The cruel asymmetry is this: **a solvent bank can die of illiquidity, but an insolvent bank cannot be saved by liquidity alone.** You can lend a solvent-but-illiquid bank cash and it will recover (this is exactly what a central bank's *lender-of-last-resort* function is for). But pouring cash into a bank whose assets are genuinely worth less than its debts just delays the funeral. The whole liquidity-management apparatus assumes the bank is fundamentally sound and is trying to make sure a temporary cash crunch never *becomes* a death.

Now the term that does all the work in the modern rules: **HQLA**, or *high-quality liquid assets*. These are assets a bank can convert to cash quickly, in size, even in a crisis, without taking a big loss. Cash itself and reserves held at the central bank are the gold standard. Government bonds (US Treasuries) come next. Then a step down to high-grade corporate bonds and government-agency mortgage securities. What makes an asset "liquid" is not whether it is *valuable* but whether there is a deep, reliable market that will pay close to fair value *on a bad day*. A ten-year Treasury is liquid even in a panic; a portfolio of small-business loans is valuable but utterly illiquid — there is no one to sell it to on Tuesday afternoon.

With HQLA defined, the two Basel III liquidity rules become readable. Both were introduced after the 2008 crisis, when banks discovered that you can be perfectly capitalised and still vaporise in a funding run. Both must be **at least 100%** — that is `db.LIQUIDITY_MIN`, the floor for both the LCR and the NSFR.

The **Liquidity Coverage Ratio (LCR)** is the short-horizon, acute-stress test. It asks: *if a severe 30-day panic hit you tomorrow, do you hold enough HQLA to cover the cash that would flee?* Formally:

$$\text{LCR} = \frac{\text{HQLA}}{\text{Net cash outflows over 30 days under stress}} \geq 100\%$$

The **Net Stable Funding Ratio (NSFR)** is the long-horizon, structural test. It asks: *is your funding mix stable enough to support your assets over a full year, so you are not relying on flighty short-term money to fund long-dated loans?* Formally:

$$\text{NSFR} = \frac{\text{Available stable funding (ASF)}}{\text{Required stable funding (RSF)}} \geq 100\%$$

Two more terms we will need throughout. The **net cash outflow** in the LCR is not the gross deposit base — it is the *outflows minus the inflows* you can reliably count on over the 30 days. And the **run-off rate** is the assumed percentage of a given funding source that flees during the stress: 5% of stable insured retail deposits, but up to 100% of funding from other financial institutions. The genius — and the controversy — of the LCR lives in those run-off rates, and we will spend a whole section on them.

It is worth pausing on *why these rules exist at all*, because the history explains their shape. Before the 2008 crisis, bank regulation was almost entirely about *capital* — making sure a bank held enough equity to absorb losses (the subject of [Basel bank regulation](/blog/trading/finance/bis-and-basel-bank-regulation)). Liquidity was treated as an afterthought, something a sound bank could always source from the market. Then 2008 happened, and regulators watched well-capitalised banks die anyway — not because their equity was exhausted but because the short-term funding markets they relied on simply closed. Northern Rock had capital; it could not roll its wholesale funding. The lesson was searing: capital and liquidity are *different* defences against *different* deaths, and a bank can have plenty of one and none of the other. Basel III, finalised in stages after 2010, added the LCR (phased in from 2015) and the NSFR (from 2018) as the first global, binding *liquidity* standards — the regulatory acknowledgement that solvency rules alone had failed to prevent the crisis. Every run-off rate, every haircut, every cap in the modern framework is a fossil of a 2008-era failure that the rule is trying to make impossible to repeat.

One more foundational distinction: *deposits are not like other debts.* When a company issues a five-year bond, the lender is locked in for five years — the company cannot be asked to repay early. But a demand deposit is debt the lender (the depositor) can call back at any instant, with no notice. This is what makes banks uniquely fragile among businesses: most of their funding is debt that can be demanded *all at once*, while most of their assets are loans that cannot be called in early. No other industry runs this structure at this scale. A factory's customers cannot all demand their money back on the same Tuesday; a bank's depositors can. Liquidity management is, at bottom, the discipline of running a business whose lenders can all show up at the door simultaneously.

The figure below puts the two ratios side by side, because the most common confusion among learners is treating them as the same rule. They are not: the LCR is a *30-day* clock; the NSFR is a *one-year* clock. They measure the same underlying fragility — funding short and lending long — but at two different time horizons.

![Comparison matrix of LCR and NSFR by horizon question numerator denominator and floor](/imgs/blogs/liquidity-management-lcr-nsfr-and-the-liquidity-buffer-2.png)

Why do banks have this problem at all, when your local shop or your employer does not? Because of the trade at the heart of banking, the spine of this entire series: a bank **borrows short and lends long**. It takes deposits that can be withdrawn today and uses them to fund mortgages that will not be repaid for thirty years. This is called *maturity transformation*, and it is genuinely useful — it lets households borrow long while savers stay liquid. But it leaves the bank structurally exposed to exactly one thing: everybody wanting their short-term money back at the same time, faster than the long-term assets can be turned into cash. Liquidity management is the discipline of surviving that exposure. (For the system-level view of how this maturity transformation also *creates money*, see [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).)

## How the LCR actually works: HQLA over the 30-day stress

Let us make the LCR concrete, because the ratio is only intimidating until you compute one.

The numerator is the **HQLA buffer**: the stock of liquid assets the bank can sell or pledge for cash within days. The denominator is the **30-day net cash outflow under stress**: the supervisor hands the bank a standardised set of assumptions — how much of each funding type runs, how much of each inflow you may count — and the bank applies them to its own balance sheet. The result is a single number: *can the buffer cover the modelled bleed?*

The "stress" is not the bank's own optimistic guess. It is a prescribed combined scenario: a partial run on retail deposits, a complete loss of unsecured wholesale funding, a credit-rating downgrade that triggers collateral calls, a drawdown of committed credit lines by the bank's own clients, and a haircut on the assets in the buffer. It is meant to approximate a 2008-style or SVB-style squeeze, all hitting at once, for 30 straight days.

#### Worked example: computing an LCR

Let us take a mid-sized bank. After applying the supervisory run-off and inflow assumptions to its balance sheet, the bank projects that over a stressed 30 days:

- Gross outflows (deposits fleeing, wholesale funding not rolling, credit lines drawn): **\$120 billion**
- Inflows it may reliably count (loan repayments due, maturing reverse repos): **\$24 billion**

The **net cash outflow** is therefore \$120bn − \$24bn = **\$96 billion**. (Basel caps the inflows you may net off at 75% of outflows, so the net outflow can never fall below 25% of gross — here \$24bn is well under that cap, so the full amount counts.)

Now the buffer. The bank holds **\$120 billion of HQLA** — cash, reserves, and Treasuries, after the appropriate haircuts. The LCR is:

$$\text{LCR} = \frac{\$120\text{bn}}{\$96\text{bn}} = 1.25 = 125\%$$

That clears the **100% floor** comfortably. The interpretation: if the prescribed 30-day storm hit tomorrow, the bank could pay every modelled outflow from its liquid buffer and still have \$24 billion left over (125% − 100% = 25% of the net outflow, or \$24bn). **The LCR turns "are we liquid?" from a gut feeling into a number you can audit.**

The chart below shows exactly this calculation: the HQLA buffer on the left, the 30-day net outflow it must cover, and the resulting 125% ratio sitting above the 100% floor.

![LCR calculation showing HQLA buffer of 120 billion against 96 billion net outflow giving 125 percent](/imgs/blogs/liquidity-management-lcr-nsfr-and-the-liquidity-buffer-3.png)

A few subtleties that separate a textbook understanding from a practitioner's. First, the LCR is a *minimum at all times*, not a quarter-end target — banks must be able to demonstrate it on any business day, which is why treasury desks watch their liquidity position daily, not quarterly. Second, the LCR is reported in the bank's reporting currency but supervisors also expect adequate liquidity in each *material currency*: a bank with euro liabilities and a dollar buffer has a hidden mismatch the headline ratio can mask. Third — and this is the one that bites — the 30-day window is an assumption, not a promise. SVB's run did not take 30 days. It took about 36 hours. The LCR is a floor, not a guarantee, and a bank that runs at exactly 100% is one fast run away from the edge.

Notice how the LCR connects to the spine. The whole reason the denominator is large is that the bank funds itself short (deposits and wholesale money that can leave) while its assets are long. The LCR does not fix the maturity transformation — that is the business — it just forces the bank to keep enough genuinely liquid assets on hand to survive the first month of the mismatch turning against it.

## Run-off rates: why not all deposits are created equal

The most important — and most counter-intuitive — idea in the whole LCR framework is that a dollar of deposits is *not* a dollar of run risk. A dollar sitting in a retired schoolteacher's insured savings account behaves nothing like a dollar of overnight funding from a hedge fund. The first will almost certainly still be there next month; the second can vanish by lunchtime. The LCR captures this with **run-off rates**: the assumed fraction of each funding type that flees during the 30-day stress.

Let us build the intuition with the analogy. Picture a coat-check at a theatre. On a normal night, only a trickle of patrons retrieve their coats early, and the attendant copes easily. But the coats are not all alike. Regulars who come every week and trust the place will wait patiently even if there is a small fire scare — they are your *stable* coats. Tourists passing through, who have heard a rumour, will all rush the counter at once — they are your *flighty* coats. If you only keep enough hands at the counter for a normal trickle, the flighty rush will overwhelm you. The run-off rate is just the regulator's estimate of how flighty each type of coat is.

Here is the ladder of stickiness the LCR assumes, from calmest to most panic-prone:

- **Stable retail deposits — about 5% run-off.** Insured (covered by deposit insurance up to the limit), transactional, from ordinary households. These are the franchise: cheap, sticky, and slow to leave. Deposit insurance is precisely what makes them sticky — see [deposit insurance and the lender of last resort](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard).
- **Less-stable retail — about 10%.** Uninsured retail balances (above the insurance cap), or rate-shopping online savers with no real relationship.
- **Operational corporate deposits — about 25%.** Balances a company keeps because it uses the bank for payroll, payments, and cash management; sticky because moving them is operationally painful.
- **Non-operational corporate — about 40%.** Excess corporate cash parked for yield, with no operational reason to stay.
- **Unsecured wholesale — about 75%.** Short-term borrowing from money-market funds and other large non-bank lenders; will not roll at the first whiff of trouble.
- **Funding from other financial institutions — up to 100%.** The flightiest money on earth. Banks know exactly how runs work, so the moment one suspects another is wobbling, it pulls every cent. The LCR assumes *all* of it leaves.

#### Worked example: applying run-off rates to a deposit base

Take a bank with \$200 billion of funding split as follows, and let us compute the modelled 30-day deposit outflow.

| Funding type | Balance | Run-off rate | Modelled outflow |
|---|---|---|---|
| Stable retail (insured) | \$90bn | 5% | \$4.5bn |
| Less-stable retail | \$30bn | 10% | \$3.0bn |
| Operational corporate | \$40bn | 25% | \$10.0bn |
| Non-operational corporate | \$20bn | 40% | \$8.0bn |
| Unsecured wholesale | \$20bn | 75% | \$15.0bn |
| **Total** | **\$200bn** | — | **\$40.5bn** |

The bank funds itself with \$200 billion, but the LCR only assumes \$40.5 billion of it flees in the stress — about 20% of the total. Now flip the composition. Suppose instead the bank had \$150bn of unsecured wholesale and only \$50bn of stable retail: the modelled outflow would balloon to roughly \$115bn — nearly *three times* larger from the *same total funding*. **Two banks of identical size can carry wildly different run risk; the difference is entirely the stickiness of the funding, which is why a cheap, insured deposit base is the most valuable asset a bank owns.**

The chart below ranks the run-off rates by funding type, colour-coded from safe (low run-off, green) to dangerous (high run-off, red). The visual point is the steep climb from retail to wholesale: the further right you go, the faster the money runs.

![Bar chart of assumed 30-day run-off rates rising from 5 percent stable retail to 100 percent financial institution funding](/imgs/blogs/liquidity-management-lcr-nsfr-and-the-liquidity-buffer-4.png)

This is the deepest lesson of the whole topic, and it is why the previous post in this track on [bank treasury and asset-liability management](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit) spends so long on funding mix. A bank does not manage liquidity mainly by hoarding cash — it manages liquidity by *cultivating sticky funding*. The buffer is the last line of defence; the first line is having depositors who do not run. SVB's fatal flaw was on this exact axis: 94% of its deposits were *uninsured*, concentrated in a tight, well-networked community of venture-backed startups who all talk to each other. That is the flightiest possible retail-shaped deposit base, and when the rumour started, they moved as one.

Here is the controversy the run-off rates carry, and it is worth naming honestly. The rates are *fixed numbers set in advance by regulators* — 5% here, 75% there — applied uniformly across banks regardless of the actual behaviour of a specific bank's depositors. That uniformity is the framework's great strength (it is auditable, comparable, and hard to game) and its great weakness (it is, by construction, an *average* that fits no individual bank perfectly). SVB is the cautionary tale: its deposits were classified largely as ordinary corporate balances attracting moderate run-off assumptions, but their *real-world* behaviour — a synchronised, social-media-accelerated stampede — was off the chart relative to any pre-set rate. The standardised run-off rates were calibrated to the runs of 2008, which unfolded over days through phone calls and branch queues. The runs of 2023 unfolded over hours through smartphones and group chats. The deepest open question in liquidity regulation today is whether *any* fixed set of run-off rates can keep pace with a world where a deposit base can evaporate faster than a board can convene. The rates are a floor built on yesterday's runs; the next run will be faster.

## The 30-day stress: watching the buffer drain

It helps to walk through the stress as a *process*, day by day, rather than as a static ratio. The LCR is not a snapshot of a calm balance sheet; it is a simulation of a month of compounding pressure, and the buffer has to still be standing at the end.

Here is the sequence the LCR effectively models. At day zero, the bank is calm and its buffer is full. Then the stress begins. Retail depositors, spooked by a headline, start pulling money — slowly, at the assumed 5–10% over the month. Wholesale lenders refuse to roll their funding; that money is gone fast, at 75–100%. The bank's own corporate clients, sensing trouble, draw down their committed credit lines (the bank promised them they could borrow, and now they do, all at once). To meet all of this in cash, the bank starts selling or repo-ing its HQLA — converting the buffer into the money that walks out the door. The question the LCR answers is whether, after all of that, the buffer is still above zero.

#### Worked example: a solvent bank running out of cash

This is the example the whole post is built around, so let us make it bite. Consider a bank with a pristine balance sheet:

- **Assets:** \$100 billion — \$60bn of good loans, \$35bn of long-dated Treasuries and mortgage bonds, and only **\$5bn of cash**.
- **Liabilities:** \$92 billion of deposits, payable on demand.
- **Equity:** \$8 billion (assets \$100bn − liabilities \$92bn). The bank is clearly *solvent* — its assets exceed its debts by \$8bn, an 8% cushion, exactly the realistic capital level for a commercial bank.

Now a run hits. On day one, depositors demand **\$30 billion** back. The bank has \$5bn of cash. It can sell some Treasuries — but to raise \$25bn of cash *today*, it must dump bonds into a falling market, taking a fire-sale loss. Say it can only realise \$0.92 on the dollar in the panic; raising \$25bn of cash means selling \$27bn of face value and booking a \$2bn loss. And the loans? They are worth \$60bn on the books but there is *no buyer* for a loan book overnight at anything near that price. The bank cannot meet the \$30bn demand from its liquid assets. It is **solvent — equity is still positive — but illiquid**, and unless someone lends it cash against its good assets, it fails. **A bank does not need to be insolvent to die; it only needs to run out of cash before the panic ends.**

That worked example is the SVB story in miniature, and it is why regulators care so much about the *composition* of the buffer, not just its size. The figure below traces the 30-day drain: the buffer starts full at day zero, leaks across retail, wholesale, and drawn credit lines, and — if the bank has built it properly — survives to day 30 with the LCR still above 100%.

![Pipeline showing a 30-day stress draining the HQLA buffer from full tank to a surviving balance at day 30](/imgs/blogs/liquidity-management-lcr-nsfr-and-the-liquidity-buffer-5.png)

The reason the buffer in our worked LCR example was \$120bn — far more than any calm day requires — is precisely this drain. On a normal Tuesday a bank needs almost no cash; deposits in roughly equal the deposits out. The buffer is not for Tuesdays. It is for the one month in a decade when the funding flips from sticky to flighty all at once. Carrying it is expensive — HQLA earns less than loans, so every dollar of buffer is a dollar not earning the lending spread — and that cost is the price of surviving the tail. A bank that skimps on the buffer to juice its [net interest margin](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) is selling its survival for a few extra basis points of yield.

## Inside the buffer: the HQLA tiers

Not all liquid assets are equally liquid, so the buffer itself is tiered. This matters because a bank cannot simply stuff its buffer with the highest-yielding "liquid-ish" assets and call it a day — the rules deliberately make lower tiers count for less.

**Level 1 assets** are the bedrock: cash, central-bank reserves, and high-grade government bonds (US Treasuries, the bonds of strong sovereigns). They count at 100% of their value — *no haircut* — and there is no limit on how much of the buffer they can make up. In a real panic, these are the only assets you can truly rely on to convert to cash at full value, because the market for them stays deep even when everything else freezes.

**Level 2A assets** are a step down: government-agency mortgage-backed securities, bonds issued by highly-rated public-sector entities, certain top-grade corporate bonds. They get a **15% haircut** (\$100 of them counts as \$85 of HQLA) and the whole of Level 2 is *capped* — Level 2A plus 2B together cannot exceed 40% of the total buffer.

**Level 2B assets** are the bottom rung: lower-rated investment-grade corporate bonds and certain blue-chip equities. They carry steep haircuts of **25–50%**, and are separately capped at **15% of the buffer**. The message is blunt: you may hold a little of this, but do not pretend it is reliable cash in a crisis.

#### Worked example: building a compliant HQLA buffer

Suppose a bank needs \$96bn of usable HQLA (to cover the \$96bn net outflow from our earlier example at exactly 100%). It wants to lean on higher-yielding Level 2 assets to save money. Can it?

It tries to hold \$50bn of Level 1 and \$70bn of Level 2A. First, the haircut: \$70bn of Level 2A counts as \$70bn × 0.85 = \$59.5bn. But then the cap bites: Level 2 cannot exceed 40% of total HQLA. If total HQLA is \$50bn + (capped Level 2), the most Level 2 can contribute is 40% of the total — solving, Level 2 is capped at two-thirds of the Level 1 amount, so \$50bn of Level 1 permits at most about \$33bn of *post-haircut* Level 2. The bank's \$59.5bn of Level 2A is slashed to \$33bn for ratio purposes. **Usable HQLA = \$50bn + \$33bn = \$83bn — short of the \$96bn it needs.** To comply, it must hold more Level 1. **The buffer's *quality* is regulated as tightly as its *size*, because a buffer full of assets that gap in a crisis is no buffer at all.**

The figure below shows the HQLA ladder: Level 1 at the top with no haircut and no cap, Level 2A trimmed by a 15% haircut and a 40% cap, Level 2B thin at the bottom with steep haircuts and a 15% cap.

![Stack of HQLA tiers with Level 1 no haircut Level 2A 15 percent haircut and Level 2B steep haircuts](/imgs/blogs/liquidity-management-lcr-nsfr-and-the-liquidity-buffer-6.png)

There is a deeper point hiding here about *what counts as money in a crisis*. In calm times, the line between "cash" and "very safe bond" feels academic — both are money-good. In a panic, that line is the difference between life and death, because the market for everything except Level 1 can seize up exactly when you need to sell. This is the lesson of 2008's frozen markets and 2019's repo spike (covered in depth in [the repo market](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight)): the assets you *thought* were liquid stopped trading the moment everyone tried to sell at once. The HQLA tiering is the regulator's attempt to bake that hard-won knowledge into the rules.

## The NSFR: stable funding for the long haul

The LCR is a survival test for the next month. But surviving the next month is no good if your *entire business model* depends on rolling flighty short-term money forever — because the next 30-day stress is always coming. The **Net Stable Funding Ratio (NSFR)** closes that gap. It is the structural, one-year companion to the LCR.

Take a concrete case. Suppose you fund a 30-year mortgage by borrowing in the overnight market and rolling the loan every single night. On any given night you are fine. But you are betting that the overnight market will *always* be open to you, at a price you can afford, for thirty years. That is an insane bet for a bank to make at scale — and yet, in the run-up to 2008, banks like Northern Rock and Lehman did roughly that, funding long-dated mortgage assets with short-term wholesale money. When the short-term market closed, they had no stable funding to fall back on, and they died. The NSFR forbids that structure.

The NSFR works by scoring both sides of the balance sheet for *stability*. On the funding side, it computes **Available Stable Funding (ASF)**: each funding source is weighted by how reliably it stays put over a year. Equity and long-term debt count at 100% (they cannot run). Stable retail deposits count at about 90–95%. Less-stable deposits and short-term wholesale count for much less, scaling down to 0% for the flightiest financial-institution funding under a year.

On the asset side, it computes **Required Stable Funding (RSF)**: each asset is weighted by how much stable funding it *needs* — essentially, how illiquid and long-dated it is. Cash and Level 1 HQLA require almost no stable funding (0–5%). A 30-year mortgage requires a lot (65%). Illiquid loans to other banks or businesses require even more (85–100%). The rule: **ASF must be at least RSF.** You must have at least as much stable funding *supplied* as your assets *demand*.

#### Worked example: computing an NSFR

Take a bank and score both sides over a one-year horizon.

**Available Stable Funding (the supply of stable money):**

| Funding | Amount | ASF factor | Stable funding |
|---|---|---|---|
| Equity & long-term debt | \$180bn | 100% | \$180bn |
| Stable retail deposits | \$360bn | ~95% | ~\$342bn |
| Less-stable deposits | \$95bn | ~85% | ~\$81bn |
| Wholesale under 1 year | \$25bn | ~50% | ~\$13bn |

Rounding to the structure, **ASF ≈ \$660 billion** of genuinely stable funding.

**Required Stable Funding (the demand for stable money):**

| Asset | Amount | RSF factor | Required funding |
|---|---|---|---|
| Cash & Level 1 HQLA | \$20bn | ~5% | ~\$1bn |
| Mortgages | \$360bn | ~65% | ~\$234bn |
| Corporate loans | \$150bn | ~85% | ~\$128bn |
| Other illiquid assets | \$70bn | ~100% | ~\$70bn |

The bank's assets *demand* roughly **RSF ≈ \$600 billion** of stable funding. So:

$$\text{NSFR} = \frac{\$660\text{bn}}{\$600\text{bn}} = 1.10 = 110\%$$

That clears the **100% floor**. The bank has \$60bn more stable funding than its assets require — a structural buffer against being forced to lean on flighty money. **The NSFR makes sure a bank's long, illiquid assets are matched by funding that will actually still be there in a year, not by overnight money that disappears at the first scare.**

The figure below shows the two stacks: available stable funding on the left (mostly sticky deposits and equity), required stable funding on the right (mostly long-dated mortgages and loans), with the NSFR of 110% comfortably above its floor.

![NSFR figure comparing available stable funding of 660 billion to required stable funding of 600 billion at 110 percent](/imgs/blogs/liquidity-management-lcr-nsfr-and-the-liquidity-buffer-7.png)

The LCR and NSFR are deliberately complementary. The LCR could be satisfied by a bank that loads up on HQLA but still funds itself with hot money — it would survive 30 days but be structurally fragile beyond that. The NSFR could be satisfied by a bank with stable funding but a thin liquid buffer — structurally sound but unable to meet an acute squeeze. You need *both*: enough liquid assets to outlast the acute panic (LCR) *and* a funding structure stable enough that the panic is unlikely to start (NSFR). Together they regulate the bank's exposure to the maturity-transformation trade from both the 30-day and the one-year angle.

## How treasury actually manages liquidity day to day

The ratios are the regulatory floor. Inside the bank, the treasury function manages liquidity as a living, daily discipline — the topic of [bank treasury and ALM](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit), which this post sits beside. A few mechanics worth knowing, because they show how the abstract ratios turn into action.

**The liquidity buffer is sized to a survival horizon, not just to the LCR.** Good treasuries run their own internal stress tests — often harsher and faster than the regulatory 30-day scenario — and ask: how many days can we survive a severe run *with no new funding at all*? This is the *survival horizon*, and a conservative bank wants it to be many weeks. The LCR is a floor; the internal target is usually higher.

**Funds transfer pricing (FTP) makes liquidity cost visible.** Inside the bank, treasury "charges" the lending desks for the liquidity their loans consume and "pays" the deposit desks for the stable funding they bring in. A 30-year mortgage gets charged a high liquidity premium (it locks up funding for decades); a sticky insured deposit earns a credit. This internal pricing is how a bank stops its lenders from writing long, illiquid loans funded by hot money — the FTP charge makes that combination unprofitable on the desk's own P&L.

**Collateral is liquidity in waiting.** Beyond outright HQLA, a bank holds assets it can *pledge* for cash — bonds it can repo, loans eligible as central-bank collateral at the discount window. A loan that is illiquid in the market may still be pledgeable to the central bank, which is why discount-window access (the lender-of-last-resort facility) is a real part of the buffer even though it never appears in the LCR numerator.

**Concentration is the silent killer.** The ratios are computed on averages and assumed run-off rates, but real runs are about *concentration*: a few huge uninsured depositors, a single sector, a tight social network. A bank can pass the LCR on paper and still be one phone call away from a run if a handful of depositors hold a quarter of its funding. The lesson of SVB is that the *distribution* of your deposits matters as much as the *total*.

This is where liquidity management connects back to the spine one more time. The whole apparatus — buffer, ratios, FTP, collateral, concentration limits — exists to manage the one trade that defines a bank: borrowing short and lending long. The bank cannot stop doing the trade; that *is* the business, and the source of its profit. What it can do is hold enough genuinely liquid assets, and cultivate enough genuinely sticky funding, that the gap between "short" and "long" never becomes the gap between "alive" and "dead."

## The survival horizon and the cost of carrying liquidity

The LCR gives you a binary verdict — above 100% or below — but it hides the question a treasurer actually loses sleep over: *if the worst happened and no new money came in, how many days could we last?* That number is the **survival horizon**, and it is the most honest single measure of a bank's liquidity strength. It reframes the buffer not as a ratio to satisfy but as a clock counting down the days of independence the bank has bought itself.

The mechanics are simple to state and brutal to live with. Lay out the bank's expected net cash outflow day by day under a severe, self-defined stress — not the regulator's stylised 30-day scenario, but the treasury's own worst case, which prudent banks make faster and harsher. Then watch the cumulative outflow eat into the stock of liquid resources (HQLA plus pledgeable collateral). The day the cumulative outflow exceeds the available liquidity is the day the bank fails. The number of days until that point is the survival horizon, and a conservative bank wants it measured in many weeks, not days.

#### Worked example: computing a survival horizon

Take a bank with **\$60 billion of usable liquid resources** — \$40bn of Level 1 HQLA at full value, plus \$20bn of pledgeable collateral that the central bank would lend against after haircuts. Now model a severe but not instant run, where the bank loses funding at a *declining* daily pace as the flightiest money leaves first:

- **Days 1–3:** the hot money goes — wholesale lenders refuse to roll, large uninsured depositors flee. Net outflow of about \$12bn per day = \$36bn over three days.
- **Days 4–10:** the panic broadens to less-stable retail and operational balances, but slows as the flightiest money is already gone. Net outflow of about \$3bn per day = \$21bn over seven days.
- **Day 11 onward:** only the stickiest insured retail remains, leaking slowly at under \$1bn per day.

After three days the bank has burned \$36bn, leaving \$24bn. After ten days it has burned \$36bn + \$21bn = \$57bn, leaving just \$3bn. On day 11, the slow leak finishes the buffer. The **survival horizon is about 11 days** — and notice it is *shorter* than the LCR's 30-day window, because this self-defined stress front-loads the outflows more aggressively than the standardised scenario. **A bank can pass the 30-day LCR and still have a survival horizon of under two weeks if its run risk is concentrated in fast, flighty money — which is exactly why the headline ratio is necessary but not sufficient.**

That gap between the regulatory 30-day window and a real bank's true survival horizon is the single most important nuance in modern liquidity management. The LCR is a floor designed by committee for an average bank facing an average crisis. The treasurer's job is to know how their *specific* balance sheet behaves under a run shaped like *their* depositors, who may move far faster than the model assumes. SVB's survival horizon, measured against the social-media-driven run it actually faced, was measured in hours — the LCR was never going to catch that.

There is also the question nobody likes: **liquidity is expensive, and the buffer is a permanent tax on earnings.** Every dollar of HQLA the bank holds is a dollar not lent out at the higher loan yield. Cash and reserves might earn the central-bank deposit rate; Treasuries earn the risk-free rate; loans earn that plus a credit spread of several percentage points. Carrying a large buffer therefore *drags down* the bank's net interest margin and return on equity. A bank with a \$120bn buffer earning, say, 2 percentage points less than it could earn lending that money is giving up roughly \$2.4 billion of pre-tax income a year for the privilege of surviving a tail event. This is the central tension of liquidity management: the buffer is pure insurance, it costs real money every single day, and the temptation to shrink it — to "reach for yield" by holding less HQLA or by funding with cheaper hot money — is exactly the temptation that has killed bank after bank. Management that quietly trades buffer for margin is selling the franchise's survival to flatter this quarter's earnings, and the bill always comes due in a crisis.

This connects to one more practical layer the ratios barely touch: **intraday liquidity.** The LCR and NSFR work in daily and yearly buckets, but a large bank must also fund its payment obligations *within* the day — settling trillions in payments hour by hour, and it needs enough liquidity *at every moment*, not just at the close. A bank can end every day with a healthy balance and still seize up mid-morning if a big outgoing payment lands before the matching incoming one. This is why payment-system plumbing and liquidity management are joined at the hip: a treasurer manages liquidity across three nested clocks at once — within the day, across the 30-day stress, and over the one-year structure — and a failure on any one of them can be fatal.

## Common misconceptions

**"A bank with positive equity can't fail."** This is the central error the whole post is correcting. Positive equity means *solvent*, not *liquid*. SVB had positive equity right up to the moment it was seized — it failed because it could not find cash, not because its assets were worth less than its debts. A bank can be solvent on Monday and gone on Wednesday. Equity protects against losses; the liquidity buffer protects against runs, and a bank needs both.

**"More deposits always make a bank safer."** It depends entirely on *which* deposits. As the worked example showed, \$200bn of sticky insured retail deposits carries a modelled 30-day outflow of around \$40bn, while \$200bn of hot wholesale money carries an outflow nearly three times larger. Rapid deposit growth from flighty, uninsured, concentrated sources — exactly what SVB had — is a *liquidity risk*, not a comfort. The quality of funding dominates the quantity.

**"The 30-day LCR window means a bank has 30 days to react in a crisis."** No. The 30 days is the *modelling horizon* for the stress, not a guaranteed grace period. Real digital-age runs move in hours. SVB lost \$42bn in a day and faced \$100bn more queued for the next morning. The LCR ensures you hold a buffer *sized* to a 30-day stress, but it cannot slow down the speed at which a modern run actually arrives. A bank running at exactly 100% LCR has no margin for a run faster than the model assumes.

**"HQLA is just a fancy word for cash, so any safe asset counts."** Liquidity in a crisis is not the same as safety in calm times. A AAA-rated corporate bond is *safe* but may stop trading in a panic; that is why it lands in Level 2B with a steep haircut, not Level 1. The tiering exists precisely because assets that look money-good on a normal day can gap when everyone tries to sell at once. Only Level 1 — cash, reserves, top-grade government bonds — counts at full value with no cap.

**"The central bank will always bail out a liquid bank, so the buffer doesn't really matter."** The lender-of-last-resort backstop is real and important, but it has limits. It lends against good collateral, at a rate, and using it signals distress that can *accelerate* a run rather than calm it. Several banks — Credit Suisse among them — tapped emergency central-bank lines and still failed when the loss of confidence outran the liquidity support. The buffer is the first line of defence precisely so the bank never has to make the run worse by visibly reaching for the backstop.

**"The LCR and NSFR are basically the same rule with different numbers."** They share a numerator-over-denominator shape and a 100% floor, but they answer fundamentally different questions on different clocks. The LCR is about *surviving an acute event* — does the liquid buffer cover a 30-day storm? The NSFR is about *not being structurally fragile in the first place* — is the funding stable enough to support the assets for a year? A bank can pass one and fail the other: a hot-money-funded bank stuffed with Treasuries can clear the LCR but fail the NSFR (it would survive the month but is structurally unsound), while a stably-funded bank with a thin buffer can clear the NSFR but fail the LCR. Treating them as interchangeable misses that they guard the maturity-transformation trade from two distinct angles.

**"Holding more liquidity is always prudent, so a huge buffer is unambiguously good."** Liquidity is insurance, and like all insurance it has a price. A buffer that is too large drags the net interest margin and return on equity down to the point where the bank cannot earn its cost of capital — which is its *own* kind of slow death, because an unprofitable bank eventually cannot retain the equity cushion that protects against solvency loss. The art is calibration: enough buffer to survive a realistic run, not so much that the franchise stops earning. "More is always safer" ignores that a bank starved of earnings is fragile too, just on a slower clock.

## How it shows up in real banks

The abstractions become unforgettable when you watch them play out in real failures. Each of these is a liquidity death — a bank killed by running out of cash, not (initially) by running out of value.

**Silicon Valley Bank, March 2023 — the 36-hour digital run.** SVB is the textbook modern case. It had ploughed deposits into long-dated Treasuries and mortgage bonds during the low-rate years, classifying \$91 billion of them as *held-to-maturity* so the paper losses did not hit reported capital. When rates rose sharply, those bonds were worth roughly \$17 billion less than their carrying value — an unrealised loss that became real the instant SVB had to sell them to raise cash. And it had to: 94% of its deposits were uninsured, concentrated in a tight venture-capital community. When SVB announced a capital raise to plug the bond hole, the network panicked and pulled \$42 billion in a single day, with \$100 billion more queued. The bank was solvent on its own marks and dead within 36 hours — the LCR's 30-day window was irrelevant against a run that moved in hours. The deeper diagnosis of the duration trap behind it belongs to the [interest-rate-risk and SVB analysis](/blog/trading/finance/svb-credit-suisse-2023-bank-runs); the liquidity lesson is simpler: the buffer was not liquid enough, fast enough, and the funding was the flightiest kind there is.

**Credit Suisse, March 2023 — the slow death of trust.** Credit Suisse was a different shape of the same disease. It was not killed by a single 30-day shock but by a multi-year erosion of confidence (scandals, losses, leadership churn) that finally tipped into a funding run. In the fourth quarter of 2022 alone, clients pulled about CHF 110 billion. The Swiss National Bank extended a CHF 100 billion liquidity line — a textbook lender-of-last-resort intervention — and it *still was not enough*, because by then the run was about trust, not just cash. Credit Suisse was merged into UBS for CHF 3 billion in a state-brokered shotgun deal, and CHF 16 billion of its AT1 bonds were wiped out entirely. The lesson: liquidity support buys time, but it cannot manufacture confidence, and once depositors stop believing, no buffer is large enough.

**Northern Rock, 2007 — the wholesale-funding trap.** Britain's first bank run since 1866 was not caused by bad mortgages — Northern Rock's loan book was, at the time, performing. It was caused by its *funding structure*: it funded long-dated mortgages overwhelmingly with short-term wholesale money rather than retail deposits. When the wholesale market froze in August 2007, Northern Rock simply could not roll its funding, and the queues outside its branches followed. This is the failure the NSFR was explicitly designed to prevent: a bank whose funding structure relied on the perpetual kindness of the short-term market. Had the NSFR existed in 2007, Northern Rock's structure would have failed the ratio years before the queues formed.

**Lehman Brothers, 2008 — the run on an investment bank.** Lehman, at roughly 30 times leverage with \$639 billion of assets, funded a vast book of illiquid assets in the overnight repo market. When counterparties lost confidence, they refused to roll the repo and demanded more collateral simultaneously — a classic wholesale-funding run. Lehman could not raise the cash, and on 15 September 2008 it filed for bankruptcy, freezing the global financial system. Lehman was arguably *insolvent* by the end (its assets had genuinely soured), but the *trigger* was liquidity: the repo market shut its door before the assets could be sold. The interplay of leverage, repo, and the run is exactly what the post-crisis liquidity rules were written to contain.

**The 2019 repo spike — even healthy banks can be caught short.** Not every liquidity event is a failure. In September 2019, the US overnight repo rate suddenly spiked from around 2% to nearly 10% as a confluence of corporate tax payments and Treasury settlements drained reserves from the system faster than banks would lend them out. No bank failed, but the episode showed that liquidity can tighten violently even among healthy institutions — and it reinforced why banks hold large HQLA buffers and why the central bank stands ready to inject reserves. Liquidity is a *system* property as much as a bank-by-bank one.

Step back and the pattern is unmistakable: in every case, the bank ran out of *cash* before it ran out of *value*. The figure below contrasts the two ways a bank dies — the slow solvency death, where losses grind the equity cushion to zero over quarters, and the fast liquidity death, where the cash runs out in hours even while equity is still positive. The LCR and NSFR are built to guard the second path.

![Graph contrasting solvency death from losses eroding equity with liquidity death from cash running out](/imgs/blogs/liquidity-management-lcr-nsfr-and-the-liquidity-buffer-8.png)

## The takeaway: how to use this

If you remember one thing about bank liquidity, make it this: **solvency and liquidity are two different ways to die, and the fast one is liquidity.** A bank's balance sheet can tell you it is rich — assets comfortably above debts, a healthy equity cushion — and that balance sheet can be entirely true on the morning the bank fails. The accounting measures value; runs measure cash. When you read about a bank "in trouble," your first question should not be "are its assets worth more than its debts?" but "can it find the cash to meet what's due this week, and how sticky is the money funding it?"

That reframing changes how you read a bank. When you look at a bank's disclosures, the liquidity story lives in a few specific places. The **LCR** tells you whether the buffer covers a modelled 30-day storm — but read past the headline number to the *composition*: a 120% LCR built on Level 1 government bonds is far stronger than the same 120% leaning on Level 2B corporate bonds that could gap in a panic. The **NSFR** tells you whether the funding structure is stable for the long haul — a bank scraping 101% is structurally living closer to the edge than one at 130%. And the number neither ratio fully captures — **deposit concentration and the insured/uninsured split** — is often the real tell. SVB passed its ratios; what killed it was 94% uninsured deposits in one tight network. A bank with a fat buffer and flighty, concentrated, uninsured funding is more fragile than its ratios admit.

For anyone trying to spot the next failure, the liquidity dashboard is concrete: rapid deposit growth from hot, uninsured sources; a high share of wholesale funding; large unrealised losses on the bond book (the held-to-maturity gap that becomes real the moment you must sell); an LCR or NSFR hovering just above 100%; and depositor concentration. Any one of these can be managed; together they are how a solvent bank quietly becomes a candidate for a 36-hour death.

And the deepest takeaway loops back to the spine of this whole series. A bank is a leveraged, confidence-funded maturity-transformation machine — it borrows short and lends long, and survives only as long as depositors trust it. Liquidity management does not change that fundamental trade; nothing can, because the trade *is* the business. What liquidity management does is buy the bank enough time and enough genuinely liquid resources to survive the moments when confidence wobbles and the short-term money tries to leave faster than the long-term assets can be sold. The buffer is not there for normal Tuesdays. It is there for the one day in a decade when everyone wants their money back at once — and on that day, the only thing that matters is whether you have the cash, not whether you have the value.

## Further reading & cross-links

- [Bank treasury and asset-liability management: the balance-sheet cockpit](/blog/trading/banking/bank-treasury-and-asset-liability-management-the-balance-sheet-cockpit) — where liquidity management lives inside the bank, alongside rate-risk and funds transfer pricing.
- [The anatomy of a bank run: from whisper to collapse](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse) — the self-fulfilling run dynamic that the liquidity buffer is built to outlast.
- [The repo market and how banks fund overnight](/blog/trading/banking/the-repo-market-and-how-banks-fund-overnight) — the wholesale-funding channel that runs fastest in a stress, and the 2008 and 2019 freezes.
- [Deposit insurance, the lender of last resort, and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard) — why insured deposits are sticky and what the central-bank backstop can and cannot do.
- [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the system-level view of the two liquidity deaths that reshaped how regulators think about speed.
- [Net interest margin and the spread business](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) — the earnings the liquidity buffer costs, and why holding HQLA is a deliberate trade against yield.
