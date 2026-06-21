---
title: "The Four Risks Every Bank Runs: Credit, Market, Liquidity, Operational"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A from-zero guide to the four core risks a bank carries, the cross-cutting risks that run through them, the three lines of defense, and how the risks compound into a failure."
tags: ["banking", "risk-management", "credit-risk", "market-risk", "liquidity-risk", "operational-risk", "three-lines-of-defense", "bank-failures", "risk-appetite"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank is a leveraged, confidence-funded machine, and almost every way it can die fits into four buckets: credit risk (a borrower does not repay), market risk (prices move against it), liquidity risk (it cannot pay cash out when people want it), and operational risk (people, processes, or systems fail). Cross-cutting risks — interest-rate, conduct, model, reputational — run through all four.
>
> - **The four core risks are not independent.** A single loss in one bucket eats the thin equity cushion, the thin cushion sparks a confidence crisis, the crisis becomes a run, and the run becomes a liquidity death — even for a bank that is solvent on paper.
> - **Banks measure each risk with a specific number.** Credit risk is priced as PD x LGD x EAD (expected loss). Market risk is summarized by Value at Risk (VaR). Liquidity risk is governed by the LCR and the liquidity gap. Operational risk is sized from a history of loss events.
> - **The defense is organizational, not just mathematical.** The "three lines of defense" — the business, an independent risk and compliance function, and internal audit — exist so that no single person or team both takes a risk and signs off on it.
> - **The one number to remember:** with about **8% equity** funding the balance sheet, a bank runs at roughly **12.5x leverage**, so a loss of just **8% of assets** wipes out the entire equity base. Every risk control exists to keep losses smaller than that thin cushion, and to keep them from arriving all at once.

In March 2023, Silicon Valley Bank looked, by one important measure, perfectly fine. It had more assets than it owed. It was, on paper, solvent. And then, over roughly 36 hours, depositors tried to pull about \$42 billion out of it in a single day, with another \$100 billion queued for the next morning, and the bank was seized by regulators before that next morning's withdrawals could even be attempted. SVB did not die because it ran out of *value*. It died because it ran out of *cash at the exact moment everyone wanted theirs back* — and the reason everyone wanted theirs back at once was a loss buried in its bond portfolio that made depositors doubt the thin sliver of equity standing between them and a default.

That single sentence contains three of the four risks this post is about, tangled together: a **market risk** (bond prices fell when rates rose), which became a **solvency** worry (the loss looked large next to the equity cushion), which detonated a **liquidity risk** (a run). The fourth risk — **operational** — is the one that produces the rogue traders, the fake accounts, the cyber breaches, and the settlement errors that have ended banks with no market move at all. Understanding how these four risks are defined, measured, defended, and — crucially — how they feed each other is the closest thing there is to a unified theory of why banks fail.

This is the map of that whole territory. The diagram below is the mental model for the entire post: four risks, each with a definition, a typical loss event, a way to measure it, and the control that is supposed to keep it in check. Everything that follows is an unpacking of this one grid.

![Matrix of the four bank risks credit market liquidity operational with definition example measurement and mitigant](/imgs/blogs/the-four-risks-every-bank-runs-credit-market-liquidity-operational-1.png)

## Foundations: what risk even means for a bank

Before we can talk about four kinds of risk, we need to be honest about what "risk" means for a business whose entire job *is* to take risk. A bank is not trying to avoid risk; if it avoided all risk it would earn nothing and have no reason to exist. A bank is in the business of taking *measured, priced, diversified* risk and earning a spread for bearing it. So when a banker says "risk", they mean something specific: **the chance that an outcome turns out worse than expected, by an amount large enough to matter.**

Let us define the building blocks from zero.

A **bank** borrows money short-term — mostly **deposits**, the money you and I keep in checking and savings accounts, which we can withdraw at any time — and lends it out long-term, as **loans** and **securities** (bonds it buys), which it will not get back for months or years. This mismatch, called **maturity transformation**, is the source of all the bank's profit and almost all of its danger. The bank earns the difference between the higher rate it charges on loans and the lower rate it pays on deposits, the **net interest margin**, typically a few percent. (For the full mechanics of that trade, see [what a bank actually does](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread).)

The crucial structural fact is **leverage**. A bank does not fund itself mostly with its owners' money. A realistic large commercial bank funds about **71% of its balance sheet with deposits, about 10% with wholesale and repo borrowing, about 7% with long-term debt, about 4% with other liabilities, and only about 8% with equity** — the owners' own capital, the part that absorbs losses first. That 8% is the entire cushion. With 8% equity supporting 100% of assets, the bank is running at roughly **1 / 0.08 = 12.5x leverage.** Every dollar of equity is holding up about \$12.50 of assets.

This is the single most important number in banking, so let us be concrete about what it implies.

#### Worked example: how thin the cushion really is

Take a simple bank with **\$100 of assets**, funded by **\$92 of liabilities** (mostly deposits) and **\$8 of equity**. The equity is what is left for the owners after everyone else is paid: assets minus liabilities, \$100 − \$92 = \$8.

Now suppose the bank's loans go bad and **\$8 of those assets become worthless.** Assets fall from \$100 to \$92. The bank still owes \$92 to its depositors and lenders — that number does not move just because loans soured. So equity is now \$92 − \$92 = **\$0.** The owners have been wiped out. A loss of just **8% of assets** — eight cents on the dollar — has erased the entire equity base. If the loss is \$9, equity is *negative* \$1, and the bank is now insolvent: it owes more than it owns.

The intuition: **risk in a bank is amplified by leverage by a factor of about 12.5x.** A loss that would barely dent an all-equity business can kill a bank, because the bank has only an 8% buffer against every loss. This is why a bank obsesses over risk in a way an ordinary company does not — its margin for error is paper-thin by design. (We go deeper on this in [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion).)

### The four core risks, defined

With leverage understood, here are the four buckets every loss eventually falls into.

- **Credit risk** is the risk that a borrower does not repay. The bank lent money and the money did not come back. This is the oldest and, for most commercial banks, the biggest risk: roughly half a typical bank's balance sheet is loans, and every one of them carries some chance of default.
- **Market risk** is the risk that the *price* of something the bank holds moves against it: bonds it owns fall when rates rise, a currency position loses on an FX swing, a stock position drops. This risk lives mostly in the **trading book** — the positions a bank holds to make markets and trade — but a version of it (interest-rate risk) lurks in the loan-and-deposit **banking book** too.
- **Liquidity risk** is the risk that the bank cannot pay cash out when it is due, *even if it is solvent.* Its assets are real and valuable, but they are long-dated loans that cannot be sold instantly, while its deposits can be withdrawn instantly. The timing does not match. This is the risk that turns a slow problem into a sudden death.
- **Operational risk** is the risk that the bank's own people, processes, or systems fail: fraud, a rogue trader, a cyber attack, a botched settlement, a mis-keyed trade, a compliance breach. There is no market move and no defaulting borrower — the loss comes from inside.

A useful way to keep them straight: **credit and market risk are about the value of what you hold; liquidity risk is about the timing of your cash; operational risk is about the integrity of your machinery.**

### Risk appetite: deciding how much risk to run on purpose

Because a bank takes risk on purpose, it needs a deliberate answer to the question *how much?* That answer is its **risk appetite** — a formal statement, approved by the board, of how much of each kind of risk the bank is willing to accept in pursuit of its returns. It is usually expressed as hard limits: a maximum loss the trading book may risk in a day, a maximum concentration in any one borrower or sector, a minimum liquidity buffer, a minimum capital ratio.

Risk appetite is the difference between a bank that is *taking* risk and one that is *gambling*. A gambler does not know how much they could lose; a well-run bank has decided in advance, written it down, and built limits to enforce it. When you read that a bank "breached its risk appetite", it means the actual risk on the books exceeded what the board said it would tolerate — a governance failure, not just a bad outcome.

## Credit risk: the borrower who does not pay back

Credit risk is where most banks lose most of their money over time, because lending is most of what they do. The mechanism is simple to state and hard to manage: the bank hands over cash today in exchange for a promise of repayment later, and the borrower may break that promise.

The art is that not every borrower is equally likely to break it, and not every broken promise costs the same. Banks formalize this with three numbers that multiply into a single expected loss. (The full machinery is in [credit risk management: PD, LGD, EAD and expected loss](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss); here is the essence.)

- **PD — probability of default.** The chance, over the next year, that the borrower fails to pay. A blue-chip company might have a PD of 0.02%; a struggling small business might be 5% or more.
- **LGD — loss given default.** If they do default, the fraction of the exposure the bank actually *loses* after seizing and selling any collateral and recovering what it can. A mortgage with a house behind it might have an LGD of 25% (the bank recovers 75% by selling the house); an unsecured credit-card balance might be 55% or higher.
- **EAD — exposure at default.** How much the borrower owes the bank at the moment they default.

Multiply them and you get the **expected loss**: `EL = PD x LGD x EAD`. This is the loss the bank *expects on average* and therefore prices into the loan and sets aside as a reserve. It is not the loss that ruins the bank — that comes from *unexpected* losses, the years far worse than average, which is what capital is for.

The reason PD matters so much is that default probability does not rise gently with risk — it rises *exponentially*. The chart below plots the one-year default probability against credit grade, on a log scale. Notice you need a log scale at all: the gap from a top-grade AAA borrower to a junk-rated CCC borrower is not a factor of ten, it is a factor of *thousands*.

![Bar chart of one-year default probability by credit rating grade on a log scale](/imgs/blogs/the-four-risks-every-bank-runs-credit-market-liquidity-operational-3.png)

#### Worked example: pricing the expected loss on a loan

A bank lends **\$1,000,000** to a mid-grade corporate borrower. Suppose the loan is rated BB, with a one-year PD of **1.2%**. The loan is partly secured, so the LGD is **45%** — if the borrower defaults, the bank expects to lose 45 cents on each dollar outstanding after recoveries. The exposure at default, EAD, is the full **\$1,000,000.**

Expected loss for the year:

$$EL = PD \times LGD \times EAD = 0.012 \times 0.45 \times \$1{,}000{,}000 = \$5{,}400$$

So the bank expects to lose **\$5,400** on this single loan, on average, per year. It must charge at least that much — \$5,400, which is 0.54% of the loan, or 54 **basis points** (a basis point is one hundredth of a percent) — *on top of* its cost of funds and operating costs, just to break even on the credit risk. If it lends at a spread that does not cover the expected loss, it is lending at a guaranteed long-run loss.

Now change one input. If the borrower's grade slips from BB (PD 1.2%) to B (PD 5.5%), the expected loss jumps to `0.055 x 0.45 x \$1,000,000 = \$24,750` — almost **five times** higher — for the same loan. The intuition: a small downgrade in credit quality is a large jump in expected loss, because PD compounds. This is exactly why banks watch their borrowers' grades like hawks and why a wave of downgrades in a recession hits earnings so hard.

### How credit risk behaves through a cycle

Credit risk is **pro-cyclical** — it hides in good times and arrives all at once in bad times. When the economy is booming, defaults are rare, reserves look generous, and banks compete by lending more and to weaker borrowers (this is the credit cycle's seduction). Then a recession hits, defaults spike across the whole book at once, and the losses the bank "expected" turn out to be far larger because they all landed in the same year. A loan book that looked safe at a 1% default rate can see 4% or 5% default in a severe downturn — and on a 12.5x-levered balance sheet, that is the difference between a profitable year and a wiped-out one.

The control for credit risk is layered: **underwriting** (only lend to borrowers who can plausibly repay), **collateral** (secure the loan against something you can seize), **diversification** (never let one borrower, sector, or region be large enough to sink you), and **provisioning** (set aside reserves for the expected loss before it happens). The single deadliest failure is **concentration**: a bank that lent too much to one industry — commercial real estate, oil and gas, crypto — discovers that those borrowers all default together, which is precisely when diversification would have saved it.

There is a subtler point buried in the expected-loss formula that explains a great deal of how banks behave through a cycle. The loss the bank *prices in and reserves for* is the **expected** loss — the average. But the loss that *kills* a bank is the **unexpected** loss, the gap between the average year and a genuinely terrible one. Capital, not reserves, is what stands behind the unexpected loss. So a bank carries two distinct defenses against credit risk: a *reserve* sized to the expected loss, taken as a cost on the income statement, and a *capital cushion* sized to the unexpected loss, sitting in equity. When a recession arrives, the bank first burns through its reserves, then through its capital — and the whole point of the 8% cushion is to be big enough to absorb a once-in-a-cycle unexpected loss without going to zero.

The way a bank books reserves also shapes its reported earnings in a way worth understanding. Modern accounting rules — **CECL** in the United States and **IFRS 9** internationally — require banks to reserve for *expected* losses *forward-looking*, not just losses that have already occurred. The effect is that reserves jump *early* in a downturn, the moment the bank's models forecast worse times, which front-loads the pain. In a recovery, the bank *releases* reserves back into profit. This is why bank earnings are so violently pro-cyclical: a single recession can swing a bank from releasing reserves (boosting profit) to building them aggressively (crushing profit) in a couple of quarters, even before a single extra loan has actually defaulted.

## Market risk: when prices move against you

Market risk is the risk that the *market value* of the bank's positions falls. The bank holds bonds, currencies, equities, and derivatives — partly to serve clients as a market-maker, partly as part of managing its own balance sheet — and the prices of all of those move every day. When they move the wrong way, the bank loses money it has to recognize immediately.

The challenge with market risk is that you cannot express it as a single guaranteed loss, because prices are random. Instead, banks summarize the *distribution* of possible daily gains and losses with a single number: **Value at Risk**, or **VaR**. (The full treatment, including its well-known flaws, is in [market risk: VaR, stressed VaR and the trading limits](/blog/trading/banking/market-risk-var-stressed-var-and-the-trading-limits), and the deeper math lives in the risk-management section.)

Here is VaR in one sentence: **VaR is the loss your portfolio will not exceed on most days, at a chosen confidence level, over a chosen horizon.** A "one-day 99% VaR of \$23 million" means: on 99 days out of 100, your one-day loss will be smaller than \$23 million. It says nothing about how bad the *other* one day in a hundred can be — and that omission is the most important thing to understand about it.

The chart below shows the idea. The bell curve is the distribution of a trading book's daily profit and loss. VaR is just a vertical line drawn at the 1% point in the loss tail. The red shaded sliver to its left — the worst 1% of days — is what VaR explicitly does *not* measure. It tells you where the bad days begin, not how bad they get.

![Loss distribution curve with the 99 percent VaR line marking the loss tail](/imgs/blogs/the-four-risks-every-bank-runs-credit-market-liquidity-operational-4.png)

#### Worked example: computing a one-day VaR

Suppose a trading desk's daily profit and loss has a standard deviation (its day-to-day "wobble", called **volatility**) of **\$10 million.** If we assume the daily P&L is roughly normally distributed — a big "if" we will return to — then the 99% point in the tail sits at **2.33 standard deviations** below the mean. (That 2.33 is a fixed property of the normal distribution: 1% of the area lies beyond 2.33 standard deviations on one side.)

$$\text{99\% one-day VaR} = 2.33 \times \$10\text{m} = \$23.3\text{m}$$

So the desk's one-day 99% VaR is **\$23.3 million.** The interpretation: on about 99 trading days out of 100, the desk should lose *less* than \$23.3 million. On the other one day in a hundred — roughly two or three days a year — it should lose *more*, possibly far more.

Now the catch. VaR says nothing about that worst day. If markets behave normally, the average loss on those tail days might be around \$28 million. But markets do not behave normally in a crisis — real loss tails are "fatter" than the bell curve predicts, so the genuinely bad day can be a multiple of VaR. The intuition: **VaR is a useful daily speed limit, but it is a liar about the crash.** A bank that manages market risk by VaR alone is measuring the calm and ignoring the catastrophe — which is why regulators added **stressed VaR** (VaR calibrated to a historical crisis period) and outright stress tests on top of it.

### Where market risk hides outside the trading book

The trading book is the obvious home of market risk, but the most dangerous version often hides in the **banking book** — the ordinary loans and deposits. This is **interest-rate risk in the banking book (IRRBB)**: the bank funds long-dated fixed-rate assets (a 30-year mortgage, a portfolio of long bonds) with short-dated deposits, so when rates rise, the value of the long assets falls while the cost of the deposits climbs. This is not gambling in markets — it is the core maturity-transformation trade itself turning toxic.

This is exactly what broke SVB. It had parked tens of billions in long-dated bonds when rates were near zero; when the Federal Reserve raised rates sharply in 2022, those bonds lost about \$17 billion of unrealized value — a market-risk loss large relative to its equity, sitting quietly in the banking book until depositors noticed. The control for market risk is a **limit framework**: VaR limits, position limits, sensitivity limits, daily mark-to-market and back-testing of the model against actual results. (The exact duration-gap mechanism is in [interest-rate risk in the banking book](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap).)

The accounting that surrounds market risk is itself a source of danger, and SVB is the cleanest illustration. Banks classify the bonds they own into two buckets: **available-for-sale (AFS)**, which are marked to market with changes flowing through equity, and **held-to-maturity (HTM)**, which are carried at cost on the assumption the bank will hold them to the end and never have to sell. The HTM bucket lets a bank *hide* an unrealized market-risk loss — the bond's price has fallen, but because it is labeled HTM, the loss never hits the headline numbers. SVB had about \$91 billion in its HTM book. The trap is that the "we will hold to maturity" assumption breaks the instant the bank is forced to sell to meet withdrawals — at which point the hidden loss becomes a real, realized, cash loss all at once. Market risk that is invisible in calm times is the most dangerous kind, because the bank, its investors, and its depositors all under-estimate it until the day it can no longer be deferred.

Back-testing deserves a word because it is the discipline that keeps a VaR model honest. If a desk's one-day 99% VaR is correct, the desk should lose more than its VaR on roughly **1 day in 100** — about two or three days a year. Banks count these **exceptions**. If a desk blows through its VaR far more often than once in a hundred days, the model is understating risk and regulators force the bank to hold more capital against the trading book (the so-called VaR "multiplier" rises). Back-testing is, in effect, the market telling the bank whether its own risk math is lying to it.

## Liquidity risk: solvent but unable to pay

Here is the risk that fools people the most, because it can kill a bank that is, by the value of its assets, perfectly healthy. **Liquidity risk is the risk that the bank cannot turn its assets into cash fast enough to meet the cash demands coming at it** — even though those assets are worth more than what the bank owes.

The everyday version of this: a corner shop has \$100,000 of inventory on its shelves and owes \$92,000 to suppliers, so it is "worth" \$8,000. But if all the suppliers demand payment *today* and the shop has only \$2,000 in the till, it cannot pay — not because it is poor, but because its wealth is locked up in shelves of goods it cannot sell in an afternoon. The shop is **solvent** (assets exceed liabilities) but **illiquid** (cannot pay on time). For a bank, the shelves are loans, and the suppliers demanding payment are depositors.

This is the most important distinction in all of bank risk, so let us draw it. The figure below shows a bank that is solvent — \$100 of assets, \$92 of liabilities, \$8 of equity — and then shows what happens when \$30 of deposits try to leave in a single day while the bank holds only \$10 of cash. The bank is still "worth" \$8. It still cannot pay \$20 of what is being demanded right now.

![Before and after showing a solvent bank with a liquidity gap when deposits leave](/imgs/blogs/the-four-risks-every-bank-runs-credit-market-liquidity-operational-5.png)

#### Worked example: the liquidity gap

A bank has **\$100 of assets**: **\$10 in cash and liquid securities** it can turn into money the same day, and **\$90 in loans** it cannot sell quickly without taking a steep loss. It is funded by **\$92 of deposits** and **\$8 of equity.** By every solvency measure it is fine: assets \$100 exceed liabilities \$92, equity is a healthy \$8.

Now **\$30 of deposits decide to leave in one day.** The bank can meet \$10 of that from its cash. The remaining **\$20 is a liquidity gap** — demand for cash the bank cannot satisfy from liquid resources.

$$\text{Liquidity gap} = \text{cash demanded} - \text{cash available} = \$30 - \$10 = \$20$$

To find that \$20, the bank must sell loans it never meant to sell, fast, into a market that knows it is a forced seller. Say it can only get **80 cents on the dollar** in a fire-sale; to raise \$20 it must sell \$25 of loans and *book a \$5 loss* in the process. That \$5 loss eats into the \$8 of equity, which makes depositors *more* nervous, which makes *more* of them leave — and the bank that was solvent this morning is insolvent by evening. The intuition: **liquidity risk is solvency risk with a fuse on it.** A liquid asset buffer is not a luxury; it is the difference between meeting a withdrawal calmly and being forced to destroy your own balance sheet to do it.

### How banks govern liquidity

Because liquidity is what actually kills banks in the moment, regulators wrote two specific rules after 2008, both of which must be at least 100%. The **Liquidity Coverage Ratio (LCR)** requires a bank to hold enough **high-quality liquid assets (HQLA)** — cash and government bonds it can sell instantly — to survive a 30-day stress scenario of deposits fleeing and funding drying up. The **Net Stable Funding Ratio (NSFR)** requires that long-dated assets be funded by stable, long-dated funding, so the bank is not financing 30-year mortgages with overnight money. (The mechanics are in [liquidity management: LCR, NSFR and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer).)

There is a second dimension to liquidity risk that the headline ratios do not fully capture: not all funding is equally *sticky*. A small retail saver with an insured deposit is unlikely to flee — they are protected up to the insurance limit (\$250,000 in the US) and tend not to watch the bank's bond portfolio. But a large corporate treasurer with \$200 million parked uninsured will move the entire balance the moment they smell trouble, and they can do it with a few taps on a screen. This is **funding concentration**, and it is half the reason SVB died so fast: about **94%** of its deposits were above the insured limit, held by a tight, well-connected community of venture-backed startups who all received the same warning at the same time and all moved at once. A bank funded by sticky, insured, diversified retail deposits can survive a scare that would kill a bank funded by flighty, uninsured, concentrated wholesale money — even if the two have identical capital ratios.

The deepest defense against liquidity risk is also the most expensive: holding a large buffer of cash and government bonds that earn almost nothing. Every dollar parked in the liquid buffer is a dollar not earning the loan spread, so there is a permanent, quiet tension between *profitability* (lend it out) and *safety* (keep it liquid). A bank that maximizes profit by running a thin buffer is, in calm times, indistinguishable from a well-run bank — and in a crisis, dead. This is the trade the central bank ultimately backstops: as the **lender of last resort**, a central bank will lend a solvent-but-illiquid bank cash against good collateral, precisely to break the fire-sale loop before it becomes a failure. The existence of that backstop is what lets banks run with smaller buffers than they otherwise could — and the moral hazard it creates is a whole story of its own.

The deeper truth these rules encode is the one SVB proved: **a bank can pass every solvency test and still die in a day if its funding is flighty and its assets are slow.** Liquidity is not solvency, and a bank's risk function must manage both, because the market will kill you for either.

## Operational risk: when the failure comes from inside

The first three risks all involve the *outside world* — borrowers, prices, depositors. Operational risk is the one where the bank does it to itself. The formal definition, from the Basel rules, is **the risk of loss from inadequate or failed internal processes, people and systems, or from external events.** In plain terms: fraud, a rogue trader, a cyber attack, a system outage, a botched settlement, a mis-priced model, a compliance breach, a natural disaster hitting a data center.

What makes operational risk strange is that there is no upside to it. Credit, market, and liquidity risk are risks the bank takes *on purpose* to earn a return — operational risk earns the bank nothing; it is pure downside, a tax on running a complicated machine. And it does not show up as a tidy distribution like market risk. It shows up as rare, enormous, lumpy loss events: a quiet decade, then a single \$1 billion line on the income statement when a trader hides positions or a control fails.

Banks measure it with a **loss-distribution approach**: collect a long history of operational loss events (your own and the industry's), fit a distribution to their frequency and severity, and estimate the capital you need to survive a bad year. Because the worst events are rare and huge, this is more art than science, and supervisors typically force banks to hold a capital add-on for it. (The detailed treatment is in [operational risk: fraud, cyber and the loss events](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events).)

Operational risk has grown, not shrunk, as banks have digitized. A modern bank is a giant software system as much as a financial one, and that creates whole new categories of operational loss: a cyber breach that exfiltrates customer data or freezes the core systems, a cloud-provider outage that takes payments offline, a botched system migration that corrupts the ledger. **Concentration** appears here too — not in borrowers but in vendors and infrastructure. When thousands of banks rely on the same handful of cloud providers, payment networks, and software vendors, an outage at one of them is a *systemic* operational event that hits everyone at once, exactly the way a credit shock hits everyone's loan book at once. The lesson is the same in a different costume: the risk you think is diversified across the industry can turn out to be a single shared point of failure.

A second feature that sets operational risk apart is that it interacts viciously with the other three. A cyber attack that knocks a bank's systems offline is an operational event that *instantly* becomes a liquidity event — if customers cannot access their money and the panic spreads, the operational failure has produced a run. A pricing-model error is an operational failure that shows up as a market-risk loss. A breakdown in loan-monitoring systems is an operational failure that shows up as undetected credit losses. Operational risk is the soil in which the other risks grow when no one is watching the machinery.

The chart below makes the scale concrete. These are real operational and conduct losses — fines, settlements, and outright collapses — and they run into the **billions of dollars each.** A single one of these would erase a year of profit at a large bank; in the case of Barings, one of them erased the entire bank.

![Horizontal bar chart of operational and conduct losses and fines in billions of dollars](/imgs/blogs/the-four-risks-every-bank-runs-credit-market-liquidity-operational-6.png)

#### Worked example: sizing an operational-risk loss

Operational losses are usually framed as frequency times severity. Suppose a bank estimates that a *major* operational loss event — a rogue trade, a large fraud, a serious cyber breach — happens on average **once every 5 years**, so the annual probability is **1 / 5 = 20%.** And suppose that when it happens, the average severity is **\$500 million.**

The *expected* annual operational loss is:

$$\text{Expected loss} = \text{frequency} \times \text{severity} = 0.20 \times \$500\text{m} = \$100\text{m per year}$$

So on average the bank should budget about **\$100 million a year** for operational losses. But the average is the wrong thing to plan for, because operational losses are lumpy: in four years out of five the loss is near zero, and in the fifth year it is \$500 million all at once. And the *tail* is far worse than the average — Barings lost about **\$1.3 billion** from one trader, more than its entire capital, and ceased to exist. The intuition: **operational risk is not managed by pricing it like credit risk; it is managed by controls** — segregation of duties, four-eyes checks, system limits, audit — because the only winning move is to stop the \$1 billion event from happening at all, not to reserve for it after the fact.

### The cross-cutting risks

The four core risks do not exhaust the list. Sitting alongside them are **cross-cutting risks** — risks that are not a separate bucket so much as a force that travels *through* the other four and amplifies them. The figure below names the four most important.

![Tree of cross-cutting risks interest-rate conduct model and reputational that run through all four core risks](/imgs/blogs/the-four-risks-every-bank-runs-credit-market-liquidity-operational-9.png)

- **Interest-rate risk** lives in the banking book (the IRRBB we met earlier). It is a flavor of market risk, but it runs through everything, because rate moves change loan values, deposit costs, and the value of the bond buffer at once. It is what broke SVB.
- **Conduct risk** is the risk that the bank treats its customers or the market unfairly — mis-selling products, manipulating a benchmark, running a sales culture that pushes staff to open fake accounts. It manifests as enormous operational losses (the fines), but its root is culture and incentives.
- **Model risk** is the risk that the math the bank relies on — its VaR model, its credit-scoring model, its pricing models — is simply *wrong*, or is used outside the conditions it was built for. In 2008, banks' VaR models radically understated the risk of mortgage securities because the models had never seen a nationwide house-price fall. The number on the screen was confident and false.
- **Reputational risk** is the risk that the bank loses the *trust* of its depositors, counterparties, and regulators. This one is special because, for a confidence-funded institution, reputation *is* the funding. When trust evaporates, deposits and wholesale funding leave at once, and reputational risk turns instantly into a fatal liquidity risk. This is the channel through which a scandal becomes a run.

## The three lines of defense

Knowing the risks is not the same as controlling them. The hardest problem in bank risk is *organizational*: the people who make money by taking risk are not the right people to decide whether the risk is acceptable, because they are paid to take it. A trading desk will always argue its positions are safe; a lending team will always argue its borrowers will repay. If the only check on risk is the people taking it, the bank is one optimistic quarter away from disaster.

The industry's answer is a structure called the **three lines of defense.** It is a deliberate separation of who *takes* risk, who *challenges* it, and who *audits* both. The figure below shows how they fit together and report up.

![Graph of the three lines of defense business risk and compliance and internal audit reporting to the board](/imgs/blogs/the-four-risks-every-bank-runs-credit-market-liquidity-operational-2.png)

- **The first line is the business.** The traders, lenders, branch managers — the people who actually take the risk in pursuit of profit. They *own* the risk they create. Critically, risk management is *their* job first; it is not something they outsource to a separate department. A first line that thinks risk is "someone else's problem" is a broken first line.
- **The second line is risk and compliance.** An *independent* function — it does not report to the business it polices — that sets the limits, builds the models, monitors the exposures, and challenges the first line. When a desk wants to take a position that breaches a limit, the second line is who says no. It is independent precisely so it *can* say no to a star trader without fearing for its job.
- **The third line is internal audit.** An independent group that checks whether the first two lines are actually doing their jobs — whether the controls exist, work, and are followed. It reports directly to the board's audit committee, not to management, so it can tell the board the unvarnished truth even when that truth is "the risk function is being overruled."

Above all three sit the **board**, which sets the risk appetite, and the **regulator**, which is the external backstop. The whole structure exists for one reason: **so that no single person or team both takes a risk and signs off on it.** Almost every great operational and conduct failure — Barings, the fake-accounts scandal, the rogue trades — is, at bottom, a story of these lines collapsing into one: a trader who controlled his own back-office settlements, a sales culture the second line was too weak to challenge, an audit function that was ignored.

#### Worked example: why independence is not optional

Imagine a star trader generating **\$50 million a year** in profit for the bank. The risk officer responsible for that trader's limits earns, say, **\$300,000 a year** and reports — in a badly designed bank — to the *head of trading*, the same person whose bonus depends on the trader's profit.

Now the trader breaches a position limit. To enforce the limit, the risk officer must tell their own boss to cut the bank's most profitable trader. The math of the situation is brutal: the risk officer is risking their \$300,000 job to police a \$50 million profit center, with their own manager on the wrong side. In that structure, the limit will not be enforced; it will be quietly raised. This is not a hypothetical — it is roughly the structure that let Nick Leeson destroy Barings, where the same person effectively ran both the trading and the settlement of his own positions.

The fix is **independence by reporting line**: the second line reports to a Chief Risk Officer who reports to the board, *not* to the business. The intuition: **a control only works if the controller can afford to enforce it** — independence is not bureaucratic box-ticking, it is the structural condition that lets a junior risk officer say no to a senior moneymaker without ending their own career.

## How the risks interact and compound into a failure

We have treated the four risks as separate buckets because that is how you learn them. But banks do not fail one bucket at a time. They fail because the risks are *connected*, and a loss in one becomes a crisis in another through the thin equity cushion and the confidence-funding that define a bank. Understanding this cascade is the whole point of the post.

The chain runs like this. A **trigger loss** appears — a wave of credit defaults, or a market-risk loss on a bond book, or a giant operational loss from a fraud. Because the bank is levered roughly 12.5x, even a moderate loss looks large next to the 8% equity cushion. That sparks **solvency fear**: depositors, counterparties, and analysts start asking whether the cushion is big enough to absorb it. Fear becomes a **deposit run** — people pull their cash, and in the digital age they pull it in hours, not weeks. The run opens a **liquidity gap** the buffer cannot fill, forcing a **fire-sale** of assets at a loss — which *crystallizes* the very losses everyone feared, deepening the solvency hole, accelerating the run. The loop feeds itself until the bank is seized or rescued.

![Pipeline showing one risk cascading from trigger loss to solvency fear to run to liquidity gap to fire sale to failure](/imgs/blogs/the-four-risks-every-bank-runs-credit-market-liquidity-operational-7.png)

The cruel feature of this cascade is that it is **self-fulfilling.** Each depositor pulling out is acting rationally given that others are pulling out, and the act of pulling out makes the bank weaker, which makes pulling out even more rational for the next person. A bank that could have survived if everyone had stayed calm is doomed once enough people decide it might not survive. This is why **reputational risk is the master risk for a bank** — it is the switch that converts any of the other three into a fatal liquidity run.

#### Worked example: a loss that becomes a failure

Start with our familiar bank: **\$100 assets, \$92 liabilities, \$8 equity** (8% capital, 12.5x leverage). Now walk the cascade with numbers.

1. **Trigger loss.** A market-risk loss of **\$5** appears — the bank's bond portfolio fell as rates rose. Assets drop to \$95, equity to \$3. The bank is still solvent (assets \$95 exceed liabilities \$92), but its cushion has shrunk from 8% of assets to about 3%.
2. **Solvency fear.** Depositors notice the \$5 loss and the \$3 of remaining equity. They reason: "If the next loss is bigger than \$3, I might not be made whole." The bank looks fragile.
3. **The run.** **\$30 of deposits** leave in a day. The bank has only \$10 of cash. It faces a **\$20 liquidity gap.**
4. **Fire-sale.** To raise \$20, the bank dumps loans at 80 cents on the dollar, selling \$25 of loans and booking a further **\$5 loss.** Assets now \$95 − \$25 = \$70 of remaining assets plus the \$20 cash raised, but equity has fallen by the \$5 fire-sale loss to *negative* \$2.
5. **Failure.** The bank is now insolvent — equity below zero — *and* still facing more withdrawals. It is seized.

Trace the sequence: a single \$5 market-risk loss (5% of assets) became, through fear, a run, and a fire-sale, a *failure*. The intuition: **no single risk killed this bank — the interaction did.** Market risk lit the fuse, reputational risk spread the fire, and liquidity risk delivered the death. This is why a bank's risk function cannot manage the four risks in separate silos: the failure mode is the *combination.*

## Common misconceptions

**"A bank fails when it runs out of money."** Not quite — a bank usually fails when it runs out of *cash at a particular moment*, which is a liquidity failure, not a wealth failure. SVB in 2023 had more assets than liabilities right up to the seizure; it failed because it could not pay \$42 billion of withdrawals in a day. Solvency is about whether the numbers add up over time; liquidity is about whether you have cash *now.* A bank can be solvent and die, and the distinction is the single most misunderstood thing in banking.

**"Credit risk is the only risk that really matters for a normal bank."** Credit risk is the *biggest* risk for most commercial banks by sheer size, but it is rarely the risk that produces the dramatic, sudden death. Slow-building credit losses give a bank time to raise capital and cut lending. It is market risk (a sudden bond loss) and liquidity risk (a sudden run) that produce 36-hour collapses, and operational risk that produces the out-of-nowhere \$1 billion line. The risk that matters is the one that can move *faster than you can react* — and that is usually not credit.

**"VaR tells you the most you can lose."** VaR tells you the loss you will *not* exceed on most days — it is explicitly a *threshold*, not a worst case. A one-day 99% VaR of \$23 million says nothing about the one day in a hundred when you lose more; on that day the loss can be a multiple of VaR, because real loss tails are fatter than the normal distribution VaR usually assumes. Treating VaR as a maximum loss is exactly the error that let banks feel safe in 2007.

**"More capital makes a bank safe, full stop."** Capital protects against *solvency* — it absorbs losses. It does almost nothing for *liquidity* — a deposit run can kill a well-capitalized bank just as fast, because the run demands cash, and capital is not cash sitting in a vault, it is an accounting cushion. A bank needs *both* a capital buffer (for losses) and a liquidity buffer (for withdrawals); the two are different defenses against different risks, and 2023 proved that a strong capital ratio is no defense against a fast run.

**"Operational risk is small and boring compared with the market risks."** The fines and losses on the operational/conduct chart run to billions of dollars each — LIBOR rigging cost the industry roughly \$9 billion, the Wells Fargo fake-accounts saga roughly \$4.9 billion, and Barings was destroyed outright by a single trader. Operational risk produces no return and is pure downside, and because it is lumpy and rare, banks chronically under-estimate it until the lumpy, rare, enormous event arrives.

## How it shows up in real banks

**Silicon Valley Bank, March 2023 — three risks in one chain.** SVB is the textbook case of risks compounding. It took on huge **interest-rate risk** in the banking book, parking deposits in long-dated bonds at near-zero yields. When rates rose in 2022, those bonds lost about **\$17 billion** of value — a **market-risk** loss large relative to its capital. When that loss became public and a few prominent venture investors told portfolio companies to pull their money, **reputational** worry detonated a **liquidity** run: about **\$42 billion** demanded on March 9, with **94%** of deposits above the insured limit and therefore prone to flight. SVB was solvent on paper and dead within 36 hours. No single risk; a chain. (The full one-pager on the 2023 episode is in [SVB and Credit Suisse: the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**Barings Bank, 1995 — operational risk with no controls.** A single trader in Singapore, Nick Leeson, hid losing derivative positions in a secret error account. The fatal flaw was operational and organizational: Leeson controlled both the *trading* and the *settlement* of his own trades — the first and second lines of defense collapsed into one person. The losses reached about **\$1.3 billion**, more than the entire capital of a 233-year-old bank, and Barings was sold for one pound. No market crashed and no borrower defaulted; the failure came entirely from inside, from the absence of segregation of duties.

**The 2008 mortgage crisis — model risk meets credit risk.** Banks held vast portfolios of mortgage securities whose risk was measured by **models** that had never seen a nationwide fall in US house prices. The models said the securities were safe and the VaR was small. When house prices fell across the whole country at once, the **credit risk** the models had dismissed became real all at once, and the **market value** of the securities collapsed because everyone tried to sell them simultaneously. **Model risk** had hidden the **credit and market risk**, and the loss tail VaR had ignored turned out to be the entire story.

**LIBOR, exposed from 2012 — conduct risk priced in billions.** For years, traders at multiple banks manipulated LIBOR, the benchmark interest rate underpinning trillions of dollars of contracts, to favor their own positions. This was **conduct risk** — unfair behavior toward the market — and it surfaced as roughly **\$9 billion** of industry-wide fines plus the wholesale replacement of LIBOR with new benchmarks. The losses were operational in their accounting, but their root was culture and incentives: a sales-and-trading environment where gaming the number was tolerated until it wasn't.

**Credit Suisse, 2023 — reputational risk as a slow-motion run.** Where SVB died in 36 hours, Credit Suisse died over years, and it is the cleanest case of reputational risk acting as the master switch. A decade of scandals, fines, and losses steadily eroded the one thing a bank cannot manufacture: trust. The damage rarely showed up as a single market or credit loss large enough to breach the capital ratios — Credit Suisse remained, on paper, well-capitalized almost to the end. What it could not survive was the slow exit of clients: roughly **\$110 billion** of outflows in the fourth quarter of 2022 alone, as wealth-management clients and depositors quietly took their money elsewhere. The Swiss authorities arranged an emergency takeover by UBS for about **\$3 billion** in 2023, and in the process roughly **\$16 billion** of Credit Suisse's AT1 bonds were written to zero. The lesson is that reputational risk does not need a dramatic trigger; the steady loss of trust drains the funding base on its own, and for a confidence-funded machine, a drained funding base is fatal even with the capital ratios intact.

**The 2008-2012 failure wave — risks are correlated across banks.** The chart of FDIC-insured bank failures shows the deepest truth about bank risk: failures do not trickle, they **cluster.** In the calm years of the mid-2000s, almost no banks failed. Then 2009 and 2010 saw **140 and 157** failures respectively, because the same credit cycle hit every bank's loan book at once. Risk that looks diversified *within* one bank is often perfectly correlated *across* all banks, because they all lent into the same boom. This is **systemic risk**, and it is why a bank's own prudence is not always enough to save it.

![Bar chart of FDIC insured bank failures per year from 2005 to 2025 showing the 2010 cluster](/imgs/blogs/the-four-risks-every-bank-runs-credit-market-liquidity-operational-8.png)

## The takeaway: how to read a bank through its risks

If you remember one frame from this post, make it this: **a bank is a leveraged, confidence-funded machine, and the four risks are simply the four ways the leverage and the confidence can break.** Credit and market risk attack the *value* of the assets, eating into the thin 8% equity cushion. Liquidity risk attacks the *timing* of the cash, killing a bank that is still worth more than it owes. Operational risk attacks the *integrity* of the machine, producing losses with no market move at all. And the cross-cutting risks — interest-rate, conduct, model, and especially reputational — are the wires that carry a loss in one bucket into a crisis in another.

When you next read about a bank, you can interrogate it through this grid. Ask: *Where is its biggest concentration of credit risk, and is it diversified or piled into one sector?* Ask: *How much interest-rate risk is hiding in its banking book, and what happens to its bond buffer if rates move?* Ask: *How sticky is its funding, and how big is its liquid buffer against a run — its LCR?* Ask: *What is the history of its operational and conduct losses, and does its three-lines structure actually have the independence to enforce a no?* And ask the integrating question that the four buckets all feed into: *How big is the loss it would take to threaten the equity cushion, and how fast could fear turn that loss into a run?*

The reason these questions matter is the spine of the whole series. A bank survives only as long as its thin equity cushion absorbs losses faster than they arrive, and only as long as depositors trust that it will. Every risk control in this post — underwriting, VaR limits, the liquidity buffer, segregation of duties, the three lines of defense — exists to do exactly two things: **keep each loss smaller than the 8% cushion, and keep the losses from arriving all at once.** When a control fails at the first, the bank becomes insolvent slowly. When it fails at the second, the bank dies in a day. The art of running a bank is the art of never letting both happen at the same time.

## Further reading & cross-links

- [Credit risk management: PD, LGD, EAD and expected loss](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss) — the full credit-risk engine behind the expected-loss formula.
- [Market risk: VaR, stressed VaR and the trading limits](/blog/trading/banking/market-risk-var-stressed-var-and-the-trading-limits) — VaR, its flaws, FRTB, and the limit framework in depth.
- [Liquidity management: LCR, NSFR and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — the Basel liquidity rules and why liquidity is not solvency.
- [Operational risk: fraud, cyber and the loss events](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events) — the loss-distribution approach and the great operational failures.
- [Interest-rate risk in the banking book: IRRBB and the duration gap](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap) — the exact rate-mismatch mechanism that broke SVB.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — the 8% equity / 12.5x leverage math that amplifies every risk.
- [SVB and Credit Suisse: the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the system-level view of the episodes used as cases here.
- [BIS and Basel bank regulation](/blog/trading/finance/bis-and-basel-bank-regulation) — the regulatory framework that turns these risk concepts into binding capital and liquidity rules.

*This is educational, not financial advice. It explains how banks manage and fail at risk; it is not a recommendation about any institution or security.*
