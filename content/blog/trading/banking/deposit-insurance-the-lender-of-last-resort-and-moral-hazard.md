---
title: "Deposit Insurance, the Lender of Last Resort, and Moral Hazard: The Three Public Backstops That Hold Banking Up"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a government guarantee stops a bank run before it starts, how the central bank keeps a solvent bank alive through a panic, and the price we pay for both — the moral hazard that lets banks take risks they could never afford alone."
tags: ["banking", "deposit-insurance", "fdic", "lender-of-last-resort", "discount-window", "moral-hazard", "too-big-to-fail", "bagehots-rule", "bank-runs", "financial-stability"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR** — A bank is a confidence machine, and confidence is the one thing a bank cannot manufacture on its own; the public sector supplies it through three backstops — deposit insurance, the lender of last resort, and, as a last resort, the bailout — each of which buys stability by quietly absorbing risk, and each of which therefore breeds the risk-taking it was meant to contain.
>
> - **Deposit insurance stops a run by removing the reason to run.** If your money is guaranteed up to \$250,000 whether you flee or stay, there is no advantage to being first in the queue — so the queue never forms. It converts flighty money into sticky money.
> - **The lender of last resort keeps a solvent-but-illiquid bank alive.** Bagehot's 150-year-old rule still governs: lend freely, against good collateral, at a penalty rate. A bank short of cash but not short of assets borrows against those assets and survives.
> - **Both backstops create moral hazard.** A protected depositor stops watching the bank, cheap insured funding lets the bank take more risk, and "too big to fail" turns the implicit guarantee into a business model — the upside is the bank's, the downside is the public's.
> - **The one number to remember: \$250,000.** That is the US FDIC limit per depositor per bank. SVB had **94% of its deposits above that line** — almost all of its money had a reason to run, which is exactly why it died in 36 hours.

In the early hours of Monday, March 13, 2023, the United States government did something it had sworn it would never do again. Three days earlier, Silicon Valley Bank had collapsed in the fastest bank run in history — depositors tried to pull \$42 billion in a single day, with another \$100 billion queued up behind them. The bank, with \$209 billion in assets, was the sixteenth largest in the country, and it was gone by Friday afternoon. The problem was that 94% of its deposits were *uninsured* — above the \$250,000 line that the deposit insurance fund guarantees. By the letter of the law, those depositors should have taken losses, waited months, and lined up as creditors in a failed-bank receivership.

Instead, the Treasury, the Federal Reserve, and the Federal Deposit Insurance Corporation invoked a clause called the "systemic risk exception" and announced that *every* depositor at SVB would be made whole — including the uninsured ones, including the venture-capital firms with hundreds of millions on deposit. The official limit, the one printed on every bank's window and website, the \$250,000 that defines who is protected and who is not, was simply waived. The message the system sent that weekend was the one it spends most of its energy trying not to send: in a big enough crisis, the line doesn't really exist.

That single weekend contains the whole story of this post. A bank lives or dies on confidence, and confidence is supplied by guarantees the bank itself cannot make credible. The public sector steps in with three of them — insurance for small depositors, emergency lending for solvent banks, and, in extremis, a bailout for the system. Each backstop genuinely works; each one prevents panics that would otherwise be catastrophic. And each one, by absorbing risk that should belong to the bank, gently encourages the bank to take more of it. That trade-off — stability bought with the seeds of future instability — is called *moral hazard*, and it is the central tension in everything that follows.

![A bank run without deposit insurance leads to collapse while a run with insurance never forms](/imgs/blogs/deposit-insurance-the-lender-of-last-resort-and-moral-hazard-1.png)

The diagram above is the mental model for the entire post. On the left, a bank without deposit insurance: a rumor spreads, the first depositors in line get paid in full while the last get cents on the dollar, so *everyone* rationally rushes to be first, the queue forms, the bank dumps assets at fire-sale prices to raise cash, and it fails. On the right, the same bank, the same rumor — but now every depositor knows their money is guaranteed whether they run or not. There is no advantage to being first. The queue never forms. The deposits stay put, the bank keeps its funding, and it survives. *The guarantee did not make the bank stronger. It made the run pointless.* Hold that distinction; it is the key to understanding why a piece of paper can stop a panic that no amount of capital could.

## Foundations: deposits, runs, and the backstops built to stop them

Before we can talk about insurance and last-resort lending, we have to be precise about what a bank actually is and why it is so prone to the thing the backstops exist to prevent.

**A bank is a maturity-transformation machine.** It takes in *deposits* — money you can withdraw at any time, on demand — and it uses that money to make *loans* and buy *securities* that won't be repaid for years. It borrows short and lends long. This is not an accident or an abuse; it is the entire point of a bank. Society wants its savings to be instantly accessible *and* it wants long-term loans for houses and factories. A bank squares that circle by promising everyone instant access while betting — correctly, almost all the time — that not everyone will ask for their money at once. (We unpack this in depth in [what a bank actually does — maturity transformation and the spread](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread); here we only need the consequence.)

**The consequence is that a bank can never honor all its promises simultaneously.** If you deposited \$1,000 and the bank lent \$900 of it to a homebuyer, then the bank has \$100 in cash and a 30-year loan. If you and every other depositor show up demanding cash at the same time, the bank cannot pay. It is not because the bank is broke — the homebuyer is still paying the mortgage — it is because the money is *tied up*. This is the difference between two words we will use constantly:

- **Solvency** is about *worth*. A bank is solvent if its assets are worth more than what it owes. Solvency is a statement about the balance sheet over the long run.
- **Liquidity** is about *timing*. A bank is liquid if it can lay its hands on enough cash *right now* to meet today's withdrawals. Liquidity is a statement about cash flow this week.

A bank can be perfectly solvent and still die of a liquidity crisis — its assets are good, but it cannot turn them into cash fast enough to satisfy a sudden stampede. (This distinction is the spine of [liquidity management — LCR, NSFR, and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer), which covers the rules banks now follow to hold a cushion of sellable assets.)

**A bank run is the stampede.** It is what happens when depositors collectively decide they would rather have their cash than their deposit. The thing that makes runs so vicious is that they are *self-fulfilling*. A bank that everyone believes is safe is safe — deposits stay, the bank keeps lending. The same bank, if everyone suddenly believes it is in trouble, *becomes* in trouble — because the act of everyone trying to leave at once is precisely what kills it. The belief creates the reality. Two equilibria, one bank, and the only difference between them is what depositors think the others will do. (We go deep on this mechanism — including the Nobel-winning Diamond-Dybvig model — in [the anatomy of a bank run, from whisper to collapse](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse).)

Now we can define the three public backstops, each of which attacks the run from a different angle.

**Deposit insurance** is a government guarantee that, if your bank fails, you will get your money back up to a set limit — \$250,000 per depositor per bank in the United States. The guarantee is paid for not by taxpayers but by the banks themselves, through *premiums* into an *insurance fund*. The fund's job is to make the depositor whole and to clean up the failed bank.

**The insurance fund** is the pot of money those premiums build up. In the US it is the Deposit Insurance Fund (DIF), run by the FDIC. When a bank fails, the FDIC dips into this fund to cover insured deposits and to absorb the loss between what the bank's assets fetch and what it owed. The fund has a target size — currently a *reserve ratio* of at least 1.35% of all insured deposits — and when failures drain it, the FDIC raises premiums or levies *special assessments* to refill it.

**The lender of last resort (LOLR)** is the central bank in its role as the emergency cash machine for the banking system. When a solvent bank is hemorrhaging deposits and can't sell its assets fast enough, it can pledge those assets to the central bank and borrow cash against them. In the US, the facility for this is called the *discount window*. The central bank can create the cash it lends — it is the one institution that never runs out of money in its own currency — which is exactly why it, and only it, can be the lender of last resort.

**The discount window** is the specific Federal Reserve facility where banks borrow short-term cash against collateral. It has existed since the Fed was founded in 1913, precisely to be the LOLR. We will see that it carries a peculiar curse — a *stigma* so strong that banks would rather die than be seen using it.

**Bagehot's rule** is the 150-year-old playbook for how the lender of last resort should behave, written by Walter Bagehot, editor of *The Economist*, in his 1873 book *Lombard Street*. The rule is three clauses: in a panic, the central bank should **lend freely** (don't ration the cash — flood the system), **against good collateral** (only to banks that have real assets to pledge, i.e. the solvent ones), and **at a penalty rate** (charge more than the normal market rate, so banks use the facility only when they truly need it and rush to repay). Lend freely, on good collateral, at a penalty rate. We will return to each clause.

**Too big to fail (TBTF)** is the situation in which a bank has grown so large, so interconnected, that its disorderly failure would crater the whole financial system — so the government cannot credibly let it fail, and everyone knows it. A TBTF bank therefore enjoys an *implicit guarantee*: lenders will fund it cheaply because they assume the state will catch it if it falls. That implicit guarantee is the single biggest source of moral hazard in modern banking.

**Moral hazard** is the heart of the matter. It is the tendency of any party that is protected from a risk to take more of that risk. If your car is insured, you may park it in a slightly worse neighborhood. If a bank's depositors are insured and its failure will be caught by the state, the bank can fund itself cheaply, its depositors stop scrutinizing it, and its managers can chase higher returns knowing the downside is socialized. The backstops that make banking stable also make banks bolder. There is no clean way out of this; the entire architecture of bank regulation — capital rules, stress tests, supervision — exists largely to *offset* the moral hazard the backstops create.

With those definitions in hand, let's build up each backstop from first principles.

## Why a guarantee stops a run before it starts

The most counterintuitive fact about deposit insurance is this: its value lies almost entirely in the runs it *prevents*, not the deposits it *pays out*. The FDIC could insure every deposit in America and, in a calm year, never write a single check — and it would still have done its most important job. To see why, we have to think about the run not as a financial event but as a *coordination game*.

Imagine a bank with 100 depositors, each with \$1,000, so \$100,000 in deposits. The bank has lent most of it out and keeps, say, \$20,000 in cash on hand. On a normal day, a few people withdraw, a few deposit, and the \$20,000 cushion is plenty. But suppose a rumor circulates that the bank made some bad loans. Now each depositor faces a decision: run, or stay.

Here is the trap. If you believe *everyone else* will stay, you should stay too — your money is safe and you keep earning interest. But if you believe *everyone else* will run, you must run *first*, because the bank only has \$20,000 in cash; the first 20 people in line get their full \$1,000, and everyone after them gets whatever the bank can scrape together by fire-selling loans — likely far less. The rational move, when you fear others will run, is to run faster than them. And since *everyone* reasons this way, the fear of a run *is* the run. This is the self-fulfilling prophecy made concrete: the bad outcome happens not because the bank is necessarily insolvent, but because each depositor's best response to everyone else running is to run.

Deposit insurance demolishes this logic in one stroke. If your \$1,000 is guaranteed whether you are first or last in line, then there is *no advantage* to running. You get your money back either way. So you don't bother to queue. And since every depositor reasons identically, *nobody* queues. The bad equilibrium — the panic — is simply eliminated. The government has not put a single dollar into the bank; it has merely changed the payoff of running from "maybe get paid in full, maybe get cents on the dollar" to "get paid in full regardless," and that change alone makes running irrational.

This is why economists say deposit insurance works by *changing beliefs*, not by changing balance sheets. The guarantee is a piece of information that, once credible, makes the thing it guarantees against impossible to trigger. It is the rare safety net whose mere existence means you almost never have to use it.

#### Worked example: insured vs uninsured share and the run incentive

Let's quantify the run incentive. Take two banks, each with \$10 billion in deposits.

**Bank A — well-diversified retail bank.** Suppose 50% of its deposits are *insured* (below the \$250,000 line) and 50% are *uninsured*. So \$5 billion is guaranteed and \$5 billion is exposed.

**Bank B — Silicon Valley Bank, March 2023.** Per the regulators, **94% of its deposits were uninsured** — that is the real number. With \$175 billion in deposits, that means roughly \$164.5 billion was *above* the \$250,000 line, and only about \$10.5 billion was insured.

Now ask: in a rumor of trouble, how much money has a *reason to run*? Insured depositors don't — their money is safe either way. The run risk lives entirely in the uninsured slice.

- Bank A: \$5 billion has a reason to run — half the book. Painful, but survivable; the insured half stays put and gives the bank a stable funding base while it manages the outflow.
- Bank B: \$164.5 billion has a reason to run — almost the entire book. There is no stable insured base to anchor the bank. The instant uninsured depositors fear loss, *nearly all the money in the building heads for the door at once.*

That is the arithmetic of SVB's death. With 94% uninsured, SVB had built a bank almost entirely out of flighty money. The \$250,000 guarantee, which makes a normal bank's deposit base sticky, protected only 6% of SVB's funding. **The takeaway: the share of deposits that is uninsured is the share that can run — and a bank whose deposits are mostly uninsured is a bank with the safety net switched off.**

![Insured versus uninsured deposit share for a typical bank and for SVB](/imgs/blogs/deposit-insurance-the-lender-of-last-resort-and-moral-hazard-3.png)

The chart makes the point visceral. A typical US bank splits roughly half-and-half between insured (green — no reason to run) and uninsured (red — a reason to run). SVB's bar is almost entirely red: 94% of its deposits sat above the insurance line. The green slice — the stable money that anchors a normal bank through a scare — was almost nonexistent. When you hear that SVB suffered "the fastest run in history," this is the structural reason: there was virtually no insured ballast to slow the stampede.

## The mechanics of deposit insurance: the limit, the fund, and the premium

Now let's open up the machinery. Three pieces matter: the *coverage limit* (who is protected and how much), the *insurance fund* (where the money comes from), and the *risk-based premium* (how the banks pay for it — and how that payment is supposed to discipline risk-taking).

### The coverage limit

In the United States, deposit insurance covers **\$250,000 per depositor, per insured bank, per ownership category**. That phrasing matters. It is not \$250,000 per person total; it is \$250,000 *at each bank*, and it stacks across *ownership categories* (a single account, a joint account, a retirement account, and a trust account at the same bank are each separately insured). A married couple can structure accounts at one bank to cover well over a million dollars. And by spreading money across multiple banks, a saver can insure essentially any amount — a practice the industry has productized into "deposit networks" that slice a large deposit into \$250,000 chunks across dozens of banks, each insured.

The \$250,000 figure was not handed down on stone tablets. US deposit insurance began in 1933, in the trough of the Great Depression, with a limit of \$2,500. It rose in steps over the decades — \$5,000, \$10,000, \$20,000, \$40,000, \$100,000 by 1980. It sat at \$100,000 for nearly three decades until the 2008 crisis, when it was raised to \$250,000 (initially as a temporary measure, then made permanent by the Dodd-Frank Act in 2010). Each increase was a response to a crisis; the limit ratchets up when panic demands it and rarely comes back down.

![Deposit insurance coverage limits in the US, euro area, and UK](/imgs/blogs/deposit-insurance-the-lender-of-last-resort-and-moral-hazard-2.png)

Other major systems land in a broadly similar range, as the chart shows. The US guarantees **USD 250,000**; the European Union's harmonized Deposit Guarantee Schemes Directive sets a limit of **EUR 100,000** per depositor per bank; the United Kingdom's Financial Services Compensation Scheme (FSCS) covers **GBP 85,000** (raised from GBP 75,000 partly to keep pace with the euro figure). The exact numbers differ, but the design philosophy is identical everywhere: cover the *ordinary* depositor in full, so that the mass of small savers has no reason to run, while leaving large, sophisticated depositors at least partly exposed so that *somebody* still has an incentive to watch the bank. That last clause — the deliberate decision *not* to insure everyone — is the system's main defense against moral hazard, and it is exactly the clause that the SVB rescue overrode.

### The insurance fund

When a bank fails, the FDIC pays insured depositors from the **Deposit Insurance Fund**. The fund is replenished by premiums the banks pay, not by tax dollars — a crucial design choice. By making the industry collectively pre-fund its own rescues, the system internalizes the cost of failure within banking itself, rather than dumping it on the general public. In principle, banking pays for banking's mistakes.

The fund has a legally mandated target. After Dodd-Frank, the FDIC must maintain a **reserve ratio** — the fund balance divided by total insured deposits — of at least **1.35%**, with a long-run goal of 2%. In quiet times the fund builds up; in a failure wave it gets drawn down, sometimes below zero. During the 2008–2010 crisis the fund actually went *negative* (a deficit of about \$20.9 billion at its worst in late 2009), and the FDIC required banks to *prepay* three years of premiums to refill it without borrowing from the Treasury. The fund is real money, it can be exhausted, and when it is, the surviving banks foot the bill through higher assessments.

![FDIC-insured bank failures per year showing the crisis waves the fund must absorb](/imgs/blogs/deposit-insurance-the-lender-of-last-resort-and-moral-hazard-6.png)

The chart above shows the fund's actual workload: FDIC-insured bank failures per year. For most of the 2000s there were essentially none — zero in 2005 and 2006. Then the 2008 crisis hit: 25 failures in 2008, 140 in 2009, a peak of **157 in 2010**, then a long tail through 2012. Then years of near-silence again. Then 2023 — only five failures, but among them were three of the largest in US history (SVB, Signature, First Republic), so the *dollar* damage dwarfed the *count*. This is the pattern of bank failure everywhere: long stretches of calm punctuated by violent clusters. The insurance fund must be sized not for the average year (which is near zero) but for the rare catastrophic one — which is exactly why pre-funding in the good years is non-negotiable.

### The risk-based premium

Here is where the system tries to claw back some of the moral hazard it creates. If every bank paid the *same* insurance premium regardless of how risky it was, then a reckless bank and a prudent bank would pay identically — and the reckless bank would be effectively subsidized by the prudent one. So the FDIC charges **risk-based premiums**: riskier banks pay more. A bank's assessment rate depends on its capital level, its supervisory rating (the CAMELS exam, covered in [stress testing, CCAR, the supervisory exam, and living wills](/blog/trading/banking/stress-testing-ccar-the-supervisory-exam-and-living-wills)), its asset growth, and other risk indicators. A well-capitalized, well-managed bank might pay just a few basis points of its deposit base; a troubled one pays multiples of that.

The logic is the logic of all good insurance: *price the premium to the risk*, so that taking on more risk costs more, and the bank has a financial incentive to stay safe. In practice the risk-sensitivity is real but blunt — the gap between the cheapest and most expensive assessment rates is measured in tens of basis points, not the orders of magnitude a truly risk-priced premium might demand. The premium nudges; it does not fully offset the subsidy.

#### Worked example: a risk-based premium and what it actually offsets

Let's put numbers on it. Take a bank with **\$10 billion in deposits** (its assessment base is actually total assets minus tangible equity under current FDIC rules, but we'll use deposits to keep the intuition clean).

**Prudent bank.** Well-capitalized, top supervisory rating. Suppose its assessment rate is **3 basis points** (0.03%) per year.

$$
\text{Premium} = \$10{,}000{,}000{,}000 \times 0.0003 = \$3{,}000{,}000 \text{ per year}
$$

**Risky bank.** Thinly capitalized, weak rating, rapid loan growth. Suppose its rate is **30 basis points** (0.30%) — ten times higher.

$$
\text{Premium} = \$10{,}000{,}000{,}000 \times 0.0030 = \$30{,}000{,}000 \text{ per year}
$$

The risky bank pays **\$27 million more** per year for the same insurance. That is the price signal: take more risk, pay more to insure it. Now compare that to what the risky bank *gains* from the guarantee. Because its depositors are insured, the risky bank can attract deposits at, say, 4.0% when an uninsured market would demand 4.5% for the same risk — a 50 basis-point funding subsidy. On \$10 billion, that subsidy is worth **\$50 million a year**. The risk-based premium of \$30 million does not even cover the \$50 million subsidy the guarantee provides. **The takeaway: risk-based premiums lean against moral hazard, but they rarely fully price it — the guarantee is still, on net, a subsidy to risk, which is why insurance alone is never enough and capital rules and supervision must do the rest.**

## How a guarantee converts flighty money into sticky money

We've established that deposit insurance prevents runs. But it does something subtler and more valuable in *normal* times too: it transforms the *behavior* of the money itself. Insured deposits are not merely safer — they are *stickier*, and stickiness is worth real money to a bank.

Think about why money moves. Uninsured money is *credit-sensitive*: the depositor is, in effect, an unsecured lender to the bank, and like any lender they will yank their money at the first sign of trouble, or shop it to a competitor for a slightly better rate, or sweep it into a money-market fund. Insured money is *credit-insensitive*: below \$250,000, the depositor doesn't care whether the bank is healthy, because their money is guaranteed regardless. They leave it in a checking account paying near zero, year after year, because it is convenient and safe. The guarantee severs the link between the bank's health and the depositor's decision to stay.

For the bank, this is the whole franchise. A base of sticky, insured, low-cost deposits — what bankers call *core deposits* — is the most valuable funding a bank can have. It is cheap (you can pay near zero on an insured checking account), it is stable (it doesn't flee in a crisis), and it doesn't reprice instantly when rates move (the *deposit beta* — the share of a rate hike passed through to depositors — is low for sticky core deposits). The entire spread business of banking, the borrow-short-lend-long machine, runs on the assumption that this funding stays put. Deposit insurance is what makes that assumption safe. (The franchise value of cheap, sticky deposits is the subject of [retail deposits — the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise).)

#### Worked example: how a guarantee converts flighty money to sticky money

Let's value the stickiness. Take a bank funding \$10 billion of assets, comparing two funding mixes.

**Bank with sticky insured deposits.** It funds the \$10 billion with core insured deposits paying an average **1.0%**. Annual funding cost:

$$
\$10{,}000{,}000{,}000 \times 0.010 = \$100{,}000{,}000
$$

And critically, in a stress scenario, *almost none* of this money leaves — the run risk is near zero.

**Bank with flighty uninsured/wholesale funding.** It funds the same \$10 billion with large uninsured deposits and short-term wholesale borrowing that, to stay, demands **4.0%**. Annual funding cost:

$$
\$10{,}000{,}000{,}000 \times 0.040 = \$400{,}000{,}000
$$

The flighty-funded bank pays **\$300 million more per year** — a 3-percentage-point gap — for funding that is *also* more likely to disappear when it is most needed. The insured-deposit franchise is worth \$300 million a year in lower funding cost *and* an immeasurable amount in survival probability. Now you understand why banks fight so hard for retail deposits and pay so little for them, and why a bank like SVB — which funded itself with huge, rate-chasing, uninsured corporate balances — was paying for that funding twice: once in a higher rate, and once, fatally, in March 2023. **The takeaway: deposit insurance doesn't just protect the depositor; it gives the bank a cheaper, stickier funding base that is the core of its profitability — flighty money costs more and runs faster.**

## The lender of last resort: keeping a solvent bank alive

Deposit insurance handles the small depositor and the slow-motion run. But it cannot, by itself, save a bank that is fundamentally *sound* yet caught in a sudden cash squeeze — a bank whose assets are good but cannot be turned into cash fast enough to meet a wall of withdrawals. For that, you need a different kind of backstop: someone who will lend the bank cash *today*, against its good assets, so it can pay the people leaving and live to see the panic subside. That someone is the central bank, acting as the **lender of last resort**.

Why can only the central bank play this role? Because the central bank is the one institution that *cannot run out of money in its own currency*. It creates the currency. When the Federal Reserve lends a bank \$10 billion through the discount window, it does not first go find \$10 billion in a vault — it credits the bank's reserve account with newly created central-bank money. This is not reckless money-printing; the loan is fully collateralized and must be repaid, so it nets out. But the *capacity* is unlimited, and that is precisely the point. A private lender might run out of cash mid-panic. The central bank never can. Its bottomless balance sheet is what makes its guarantee credible — and credibility, as we keep seeing, is the whole game. (This connects to how money is created in the first place; the system-level view is in [how money is created — banks, central banks, the money multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).)

### Bagehot's rule, clause by clause

The genius of the lender-of-last-resort idea is that it was essentially solved 150 years ago, and the solution still holds. Walter Bagehot's three-clause rule from 1873 is one of the most durable pieces of practical economics ever written.

**Lend freely.** In a panic, the central bank should not ration its lending or dribble it out cautiously. It should make it overwhelmingly clear that *unlimited* cash is available to any solvent bank that needs it. The reason is psychological as much as financial: a panic is a coordination failure, and the way to kill it is to convince everyone that there is more than enough cash to go around, so nobody needs to grab theirs first. Half-measures fail — a central bank that lends timidly signals that it, too, is worried, which feeds the panic. Flooding the system with the *promise* of cash often means very little of it is actually drawn, exactly as with deposit insurance.

**Against good collateral.** The central bank should lend only to banks that can pledge sound assets — Treasuries, high-quality loans, solid securities. This is the clause that separates *illiquidity* from *insolvency*. A bank with good collateral is, by definition, solvent: its assets are worth what it owes. The collateral requirement ensures the central bank rescues the solvent-but-illiquid and *not* the genuinely broke. It is the test that keeps last-resort lending from becoming a bailout. A bank that has no good collateral left to pledge is a bank whose assets are already worthless — and that bank should fail, because lending to it just throws public money after bad.

**At a penalty rate.** The central bank should charge *more* than the normal market rate. This sounds harsh during a crisis, but it is essential for two reasons. First, it ensures banks use the facility only when they genuinely cannot get cash elsewhere — they won't borrow from the central bank at a penalty if a private lender will do it cheaper. Second, it gives the bank a powerful incentive to repay as fast as possible and return to private funding. The penalty rate keeps the lender of last resort from becoming a cheap permanent funding source — it is emergency liquidity, deliberately priced to be uncomfortable.

Put the three clauses together and you get a precise rescue: flood the system with cash so the panic dies, but only to banks that are actually solvent (good collateral), and at a price that makes them treat it as the emergency it is (penalty rate). Lend freely, against good collateral, at a penalty rate.

![Bagehot's rule shown as a pipeline where a solvent bank pledges collateral and survives a panic](/imgs/blogs/deposit-insurance-the-lender-of-last-resort-and-moral-hazard-4.png)

The pipeline above traces Bagehot's rule in action. A solvent bank is losing deposits fast. It pledges its good assets — Treasuries, quality loans — as collateral. The central bank lends freely against them at a penalty rate. The bank uses that cash to pay out the depositors who want their money now. A collateral *haircut* (lending only, say, 95 cents against a dollar of Treasuries) protects the public from any loss even if the collateral dips in value. The panic fades, the bank repays the loan, and it survives. *No public money was lost; the central bank simply bridged the gap between the bank's good-but-illiquid assets and its immediate cash needs.* That is the lender of last resort doing exactly its job.

#### Worked example: the cost of a bailout vs a resolution

The central distinction in this whole post is between *liquidity support* (Bagehot — fully collateralized, repaid, no loss) and a *bailout* (public money into a failing firm). Let's quantify why the first is so much cheaper than the second.

**Liquidity support (the Bagehot case).** A solvent bank faces \$20 billion of deposit outflows in a week. It pledges \$21 billion of Treasuries to the central bank, which applies a 5% haircut and lends \$20 billion. The bank pays a penalty rate of, say, 5.5% (1 percentage point above the normal market rate of 4.5%) for two weeks before the panic subsides:

$$
\text{Interest} = \$20{,}000{,}000{,}000 \times 0.055 \times \tfrac{2}{52} \approx \$42{,}300{,}000
$$

The bank pays roughly \$42 million in interest, repays the \$20 billion in full, and the central bank ends with a small *profit* on the penalty rate. **Public cost: zero.** This is liquidity support working as designed.

**Bailout (the failing-firm case).** Now suppose the bank is not merely illiquid but *insolvent* — its assets are worth \$90 billion against \$100 billion of liabilities, a \$10 billion hole. No collateral can fix this; the bank is genuinely broke. To keep it alive, the government injects \$15 billion of public capital (enough to fill the \$10 billion hole and leave a cushion). If the bank later recovers, the government may sell its stake and recover some or all of it; if it doesn't, the \$15 billion is gone. **Public cost: up to \$15 billion at risk, with no collateral backing it.**

The two interventions look superficially similar — the state hands a struggling bank money — but they are worlds apart. The Bagehot loan is to a *solvent* bank, fully secured, profitable to the lender, and self-extinguishing. The bailout is to an *insolvent* bank, unsecured, and a direct transfer of public funds to private creditors and shareholders. **The takeaway: the lender of last resort is supposed to provide liquidity to the solvent, never capital to the insolvent — and the entire art of crisis management is telling those two situations apart at 2 a.m. on a Sunday with incomplete information.**

## Too big to fail: the implicit guarantee that became a business model

So far the backstops have been *explicit* and *rule-bound*: deposit insurance has a stated limit, the discount window has stated collateral rules. But the most consequential backstop of all is *implicit* and *unwritten*: the market's belief that the government will not allow a sufficiently large bank to fail. That belief is "too big to fail," and it warps everything.

The logic that creates TBTF is grim but simple. When a giant, interconnected bank fails *disorderly* — as Lehman Brothers did in September 2008 — it does not fail alone. Its collapse freezes the payment system, vaporizes the value of every contract it was a counterparty to, triggers fire sales that crash asset prices for *healthy* banks, and ignites panic that spreads to institutions that were perfectly sound. The failure of one big node takes down the network. Faced with that prospect, governments blink — they rescue the giant rather than let the contagion spread. And because everyone *knows* the government will blink, the giant gets to borrow as if it were government-backed even though no law says it is.

The term itself was born in a specific failure. In 1984, **Continental Illinois** — then the seventh-largest US bank — suffered a wholesale-funding run after bad energy loans soured. The FDIC and the Fed rescued it, guaranteeing *all* depositors and creditors, not just the insured ones, because they feered its failure would topple the many small banks that had deposits with it. A congressman, questioning regulators about the rescue, named a list of eleven banks he said were "too big to fail," and the phrase entered the language. From that day, the market understood that size itself confers protection.

This implicit guarantee has a measurable value. Because lenders assume a TBTF bank will be caught if it falls, they accept a lower interest rate to fund it — a *funding advantage* that researchers have estimated, in stressed periods, at anywhere from a few basis points to over a full percentage point of the bank's borrowing cost. That is a direct subsidy, worth billions a year to the largest banks, paid for by no one in calm times and by the public when the guarantee is finally called. The biggest banks are the ones the surcharge in the capital rules tries to claw back — the G-SIB (global systemically important bank) surcharge demands extra capital precisely from the banks whose implicit guarantee is largest. (We cover that surcharge in [Basel I, II, III and the capital rules that govern every bank](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank); the system-level view of regulation is in [BIS and Basel bank regulation](/blog/trading/finance/bis-and-basel-bank-regulation).)

The corrosive part is that TBTF turns the backstop into a *strategy*. If being big and interconnected earns you cheaper funding and a state guarantee, then growing big and interconnected is rational — even if it makes the system more fragile. The protection meant to contain a crisis becomes an incentive to build the very institutions that cause the next one.

## The moral hazard loop: how a backstop subsidizes the risk it insures against

We have now met all three backstops and can finally close the circle on moral hazard — the thread that runs through every one of them. The uncomfortable truth is that *each backstop, by absorbing a risk, lowers the cost of that risk to the bank, which encourages the bank to take more of it.* This is not a bug in a poorly designed system; it is an unavoidable feature of *any* system that provides a safety net. Insurance always changes the behavior of the insured.

Trace the loop. A public backstop — deposit insurance, the implicit TBTF guarantee — makes the bank's funding cheap and *insensitive to its risk*. Depositors stop watching the bank, because they're protected; bondholders fund it cheaply, because they assume a rescue. With cheap, risk-blind funding, the bank can take on more risk than the market would otherwise let it: the upside of that risk accrues to the bank's shareholders and managers, while the downside, beyond the thin equity cushion, falls on the backstop — the insurance fund, the central bank, the public. *Heads I win, tails you lose.* The bank grows, the risk grows, and when the risk eventually goes wrong, the losses exceed the equity cushion and the backstop pays. And because the bank is now bigger, the next backstop must be bigger too. The loop tightens with each cycle.

![The moral hazard loop where a public backstop leads to more risk taking and a bigger backstop](/imgs/blogs/deposit-insurance-the-lender-of-last-resort-and-moral-hazard-7.png)

The diagram above is the loop in one picture. A public backstop makes funding cheap and depositors stop watching. The bank takes more risk — upside mine, downside theirs. It grows bigger, and "too big to fail" becomes the plan rather than an accident. Eventually the risk goes wrong, losses exceed the thin equity cushion, and a bigger backstop is required — which restarts the loop one notch larger. Every arrow is a real mechanism we've defined: the cheapening of funding (deposit insurance, TBTF), the asymmetry of upside and downside (limited liability plus a guarantee), the growth incentive (the funding subsidy scales with size). This is *the* structural problem of modern banking, and there is no way to abolish it without also abolishing the backstops that keep the system from collapsing into runs.

#### Worked example: the asymmetry that drives the loop

Let's make the "heads I win, tails you lose" asymmetry concrete with a single risky bet. Take a bank with **\$8 of equity** funding **\$100 of assets** — the realistic large-bank structure of roughly 8% equity and about 12.5× leverage (1 ÷ 0.08 ≈ 12.5).

Now the bank can make a risky loan that has:
- a **90% chance** of paying off, earning an extra **\$10** profit, and
- a **10% chance** of going bad, losing **\$20**.

The *expected value* of this bet is:

$$
0.90 \times (+\$10) + 0.10 \times (-\$20) = \$9 - \$2 = +\$7
$$

A positive expected value — so far so good. But look at who bears the *downside*. If the loan goes bad and the bank loses \$20, the bank's \$8 of equity is wiped out (it can only lose what it has), and the remaining \$12 of loss falls on... the depositors, or, since they're insured, on the **insurance fund and ultimately the public**. The bank's shareholders cannot lose more than their \$8; everything beyond that is someone else's problem.

So recompute the bet *from the shareholders' point of view*, where their loss is capped at \$8:

$$
0.90 \times (+\$10) + 0.10 \times (-\$8) = \$9 - \$0.80 = +\$8.20
$$

The bet is worth even *more* to shareholders (\$8.20) than its true economic value (\$7), because they've offloaded \$12 of the bad-case loss onto the backstop. And here is the poison: a bank will take this asymmetry to its logical conclusion. It will accept bets with *negative* true expected value as long as they're positive *for the shareholders after capping the downside*. Suppose the bad case lost \$50 instead of \$20:

- True EV: $0.90 \times \$10 + 0.10 \times (-\$50) = \$9 - \$5 = +\$4$ — still positive but worse.
- A bad case of \$120 loss: True EV $= \$9 - \$12 = -\$3$ — a value-destroying bet. But to shareholders, capped at \$8: $0.90 \times \$10 + 0.10 \times (-\$8) = +\$8.20$ — *still attractive.*

**The takeaway: the thinner the equity cushion and the more loss the backstop absorbs, the more a bank is rewarded for taking bets that destroy value for society but enrich its shareholders — which is exactly why capital requirements (more equity at risk) and risk-based premiums (a price on the bet) exist: to put the bank's own skin back in the game.**

This is why capital is the true antidote to moral hazard. The more equity the bank has at risk before the backstop kicks in, the more the shareholders bear their own downside, and the less the asymmetry distorts their choices. Deposit insurance and capital rules are two halves of one design: the insurance stops the run, the capital makes sure the insured bank still has something to lose. (The capital-as-cushion argument is developed in full in [bank capital and leverage — why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion).)

## Three backstops, three answers to who pays

It's worth pausing to lay the three interventions side by side, because the public conversation constantly confuses them. "Bailout" gets used as a slur for all three, but they differ in exactly the dimensions that matter: who they protect, who pays for them, and how much moral hazard they breed.

![A matrix comparing deposit insurance, lender of last resort, and bailout by who is protected and who pays](/imgs/blogs/deposit-insurance-the-lender-of-last-resort-and-moral-hazard-5.png)

The matrix above lays it out. **Deposit insurance** protects small depositors up to the limit; it is paid for by the banks themselves through risk-based premiums; and if those premiums are priced reasonably, the moral hazard is low because the bank bears the cost. **The lender of last resort** protects a solvent bank that is merely short of cash; if the loan is fully collateralized and repaid, no one bears any cost (the central bank may even profit on the penalty rate); the moral hazard is moderate and capped by that penalty rate. **A bailout** protects the failing firm itself and its creditors; it is paid for by the taxpayer, up front, with no collateral backing; and the moral hazard is the highest of all, because the risk-takers keep their upside while the public absorbs the downside.

The clean version of the system uses the first two and never the third. Insurance handles the small saver and stops the everyday run; the lender of last resort bridges a solvent bank through a liquidity squeeze. A bailout is supposed to be the option that never gets used — the admission that the first two failed and the bank should have been allowed to fail in an orderly way instead. The trouble, as 2008 and 2023 both showed, is that when a big enough bank is about to fail and there's no orderly way to wind it down, the choice collapses to "bail it out or watch the system burn" — and at that point, the moral hazard everyone worried about in calm times gets created in a single weekend.

## Orderly resolution: how to let a bank fail without a bailout

If bailouts are the worst option and runs are the disease, what's the cure for a bank that genuinely *should* fail — one that's insolvent, not just illiquid? The answer the post-2008 world built is **orderly resolution**: a way to let a bank fail that imposes losses on its owners and creditors while keeping its deposits, payments, and critical functions running. Done right, resolution is the thing that makes "too big to fail" no longer true.

The contrast that motivated the whole reform is **Lehman vs. the goal**. When Lehman Brothers failed in September 2008, it failed *disorderly* — there was no plan, no buyer lined up, no mechanism to keep its functions running. It simply stopped, and the system seized. Counterparties couldn't tell what they'd lose, money-market funds "broke the buck," short-term funding markets froze worldwide, and the panic forced a cascade of rescues for everyone *else*. Lehman's disorderly death is the single best argument for why you need an orderly process.

![Disorderly failure versus orderly resolution shown as a before and after comparison](/imgs/blogs/deposit-insurance-the-lender-of-last-resort-and-moral-hazard-8.png)

The figure contrasts the two. On the left, disorderly failure (Lehman, 2008): the bank just stops with no plan and no buyer; payments and trades freeze across the system; panic spreads to healthy banks; and the public is forced to bail out *everyone* to stop the run. On the right, the goal of orderly resolution: a *living will* — a pre-written plan for how the bank can be wound down — is ready in advance, with a bridge bank or a buyer lined up; deposits and payments keep running by Monday morning; shareholders are wiped out and bondholders are *bailed in* (their debt converted to equity or written down to absorb losses), so the owners pay first; and there is no taxpayer money and no system freeze. The failure is absorbed, not socialized.

The key tools of orderly resolution are **living wills** (resolution plans every big bank must file, mapping how it could be dismantled without chaos — covered in [stress testing, CCAR, the supervisory exam, and living wills](/blog/trading/banking/stress-testing-ccar-the-supervisory-exam-and-living-wills)), **bail-in debt** (a layer of bonds explicitly designed to absorb losses in a failure, so the loss falls on investors who were paid to take it rather than on the public), and **bridge banks** (a temporary FDIC-run entity that keeps the failed bank's good operations running while the mess is sorted out). The 2023 sale of First Republic to JPMorgan over a weekend, and the FDIC-brokered sale of SVB's deposits, were resolution in action — imperfect, but vastly better than a Lehman-style freeze.

The reason resolution matters so much for *this* post is that it is the only real answer to moral hazard. If a bank's owners and creditors know they will actually be wiped out in a failure — that the living will is real, the bail-in debt will be bailed in, the bridge bank will take over — then the "tails you lose" half of the asymmetry comes back. Skin returns to the game. Whether resolution is *credible* — whether regulators will really let a giant fail through the process rather than blinking and bailing — is the open question on which the entire moral-hazard problem turns. The SVB systemic-risk exception, where the uninsured were made whole rather than taking losses, was a discouraging data point: when the moment came, the system chose stability over the principle.

## Common misconceptions

**"Deposit insurance means the government pays when a bank fails, so it's a taxpayer bailout."** No — in the normal case, deposit insurance is funded by the *banks*, not the taxpayer. The FDIC's Deposit Insurance Fund is built up from premiums paid by every insured bank, and when failures drain it, the surviving banks are assessed to refill it. During 2008–2010 the fund went into deficit and the FDIC made banks *prepay* three years of premiums rather than tap the Treasury. The public backstops the fund only as an ultimate guarantee that has historically not cost the taxpayer for insured deposits. The taxpayer cost in a crisis comes from *bailouts of failing firms*, which are a different intervention entirely.

**"If my bank fails, I'll lose my money or have to wait months to get it."** For insured deposits, the opposite is true. The FDIC's track record is that insured depositors typically get access to their money within *one business day* — the failed bank is usually seized on a Friday and reopens (often under a new owner) the following Monday, with insured balances fully available. No insured depositor has lost a penny of insured money since the FDIC was founded in 1933. The waiting and the losses apply to *uninsured* deposits above the limit, which become claims in the receivership and may recover only partially and slowly — which is exactly why the insured/uninsured line is the line that matters.

**"The lender of last resort is just a bailout by another name."** They are fundamentally different. The lender of last resort lends *cash* against *good collateral* to a *solvent* bank, at a *penalty rate*, and expects to be repaid — often at a profit. A bailout injects *capital* (which can be lost) into an *insolvent* firm with *no collateral*. The first is a fully secured loan that bridges a timing problem; the second is a transfer of public funds that bridges a *worth* problem. Bagehot's whole point — lend on good collateral — exists precisely to keep last-resort lending from sliding into bailout.

**"Raising deposit insurance to cover everyone would just make banking safer."** It would make runs rarer but moral hazard worse, and the trade-off is the whole problem. If *every* deposit were insured regardless of size, then *no* depositor would ever have a reason to monitor their bank's risk-taking — the last private check on reckless banks would vanish, and banks could fund any gamble at the risk-free rate. The deliberate decision to leave large depositors partly exposed is a feature, not an oversight: it keeps *someone* with skin in the game watching the bank. The SVB rescue, which made all depositors whole, removed that check for one bank in one crisis — and the debate over whether to do it permanently is precisely a debate about how much moral hazard we're willing to buy in exchange for fewer runs.

**"Too big to fail was solved by the post-2008 reforms."** Reduced, not solved. Higher capital requirements, the G-SIB surcharge, living wills, bail-in debt, and resolution authorities have all made a big bank's failure more survivable and shifted more loss onto its owners. But the 2023 crisis showed the implicit guarantee is alive: regulators invoked the systemic-risk exception to protect uninsured depositors at SVB and Signature, and First Republic was rescued by a JPMorgan acquisition rather than wound down through the resolution process. The funding advantage that big banks enjoy from their assumed protection has shrunk but not disappeared. TBTF is a problem that is *managed*, not eliminated — and every crisis tests whether the management holds.

## How it shows up in real banks

**The 2008 backstops: the discount window, then everything else.** When the crisis hit in 2007–2008, the Fed first did exactly what Bagehot prescribed — it lent freely against collateral through the discount window and a series of new facilities (the Term Auction Facility, the Primary Dealer Credit Facility, and more) to keep solvent institutions funded as private markets froze. For genuinely *solvent* but illiquid banks, this was textbook lender-of-last-resort policy, and much of it was repaid at a profit. But the crisis also produced the other kind of intervention: the \$700 billion Troubled Asset Relief Program (TARP) injected *capital* into banks, some of which were arguably insolvent, and the AIG rescue committed over \$180 billion to a failing insurer whose collapse threatened the banks it had insured. The 2008 response was thus a mix of all three backstops — clean Bagehot lending, deposit-guarantee expansions (the FDIC raised the limit to \$250,000 and temporarily guaranteed certain bank debt), and genuine bailouts — and the public's enduring anger conflated them all under one word. The lesson regulators drew was that they lacked a tool to let a big bank fail *orderly*, which is why Dodd-Frank created the resolution authority and the living-will requirement.

**SVB's uninsured run and the systemic-risk exception.** Silicon Valley Bank is the purest modern case study because it shows every concept in this post firing at once. SVB had built a deposit base that was **94% uninsured** — almost entirely large corporate and venture-capital balances far above the \$250,000 line. That meant nearly all of its funding was flighty: credit-sensitive money with every reason to run at the first sign of trouble. When the bank announced a \$1.8 billion loss on a forced bond sale on March 8, 2023, the venture-capital community — a tightly networked group that talks constantly — concluded simultaneously that the bank was in danger. The run was instantaneous: **\$42 billion** attempted on March 9, with **\$100 billion** queued for March 10, against a bank of \$209 billion in assets. There was no insured ballast to slow it. The bank was seized on Friday, March 10. Then, over that weekend, regulators faced the choice the whole system is built to avoid: let the uninsured depositors take losses (as the law says) and risk a contagious run on every other regional bank with a high uninsured share, or invoke the **systemic-risk exception** and guarantee *all* deposits, insured and uninsured alike. They chose the exception. The \$250,000 limit was, for SVB, suspended. It stopped the contagion — but at the cost of teaching every large depositor in America that, in a big enough panic, the limit is negotiable. (The full duration-trap-and-digital-run story is told in [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**The discount window stigma.** Here is one of the strangest and most damaging facts in all of central banking: the lender of last resort exists, it works, and banks would rather *fail* than use it. The discount window carries a *stigma* — a belief that any bank seen borrowing from it must be in trouble, so borrowing from it *signals weakness* and can itself trigger the run it was meant to prevent. During 2008, banks went to extraordinary lengths to avoid the window, borrowing from anywhere else first, because they feared that markets or regulators would interpret discount-window use as a death sentence. The Fed has fought this stigma for decades — renaming the facility, reducing the penalty, encouraging healthy banks to borrow occasionally so that borrowing looks normal — with limited success. The stigma is a vivid example of how the *psychology* of confidence can defeat the *mechanics* of a backstop: a tool that perfectly solves the liquidity problem on paper is useless if no one dares to be seen using it. In 2023, the Fed responded to the SVB crisis by creating a *new* facility — the Bank Term Funding Program — that lent against collateral at *par* (no haircut), partly to give banks a less-stigmatized window to use. The fact that they had to build a new front door because the old one was unusable tells you everything about how powerful stigma is.

**Continental Illinois and the birth of the doctrine.** The 1984 rescue of Continental Illinois is where the modern TBTF problem was effectively created. Faced with a wholesale-funding run on the seventh-largest US bank — a bank whose failure would have hit hundreds of smaller "correspondent" banks that kept deposits with it — regulators guaranteed *all* of Continental's depositors and creditors, not just the insured ones. It worked: the contagion was contained. But it established, in the clearest possible terms, that a large bank's uninsured creditors would be protected, and the market never forgot. Every funding decision for a giant bank since has carried the quiet assumption that the state stands behind it. The phrase "too big to fail," coined in the congressional hearings that followed, named a doctrine that the rescue itself had brought into being. (The full story is in [Continental Illinois 1984 and the birth of too-big-to-fail](/blog/trading/banking/continental-illinois-1984-and-the-birth-of-too-big-to-fail).)

**Credit Suisse and the AT1 wipeout.** The March 2023 collapse of Credit Suisse showed the *bail-in* tool working — brutally and controversially. As the bank's deposits bled out and its share price collapsed, Swiss authorities arranged an emergency takeover by UBS. In the process, they wrote down **CHF 16 billion** of Credit Suisse's Additional Tier 1 (AT1) bonds to *zero* — the loss-absorbing debt that was designed precisely to take the hit in a crisis. Controversially, they did this while shareholders received CHF 3 billion in the UBS deal, inverting the usual hierarchy in which equity is wiped before debt. The AT1 wipeout demonstrated that bail-in debt *can* impose losses on private investors rather than the public — which is the whole point of building it — but the messy, hierarchy-inverting execution also rattled the global AT1 market and showed how hard orderly resolution is to pull off cleanly under pressure. It was moral-hazard medicine and a moral-hazard cautionary tale at the same time.

## The takeaway / How to use this

If you remember one thing from this post, make it this: **the backstops that keep banking standing are the same forces that make it dangerous, and there is no version of banking that has one without the other.** A bank is a confidence machine, and confidence is a public good that no private bank can supply for itself. So the public supplies it — through a deposit guarantee that makes runs pointless, through a lender of last resort that bridges solvent banks across a panic, and, when those fail, through bailouts and resolutions that decide who eats the loss. Every one of those backstops works. And every one of them, by absorbing risk that should belong to the bank, quietly pays the bank to take more.

For reading a bank, this reframes what you look for. Don't just ask "is this bank well-run?" — ask **"how much of its funding has a reason to run?"** The single most predictive number in both SVB and Continental Illinois was the *uninsured share* of deposits: the fraction of the funding base that the \$250,000 guarantee does not cover and that will therefore flee at the first scare. A bank funded by sticky, insured, low-cost core deposits is a fortress; a bank funded by large uninsured balances and short-term wholesale money is a tinderbox, no matter how good its loans look on a calm day. The guarantee turns flighty money into sticky money — so the share of money the guarantee *doesn't* reach is the share that can kill the bank.

For thinking about the system, hold the moral-hazard loop in your head as the permanent tension it is. Every time a crisis is met with a bigger backstop — every systemic-risk exception, every "whatever it takes," every guarantee extended to the uninsured — the immediate panic is calmed and the long-run incentive to take risk is nudged up one more notch. The job of capital requirements, risk-based premiums, stress tests, and credible resolution is to push back against that loop, to keep enough skin in the game that the bank still has something to lose. Whether they push back *hard enough* is the question every banking crisis re-asks, and 2023 answered it ambiguously: the system reached for stability and accepted the moral hazard, as it almost always does when the alternative is watching the machine burn.

That is the deepest expression of this series' spine. A bank borrows short, lends long, earns the spread, and survives only as long as depositors trust it. The public backstops *manufacture* that trust when the bank cannot — and in doing so, they make the fragile trade survivable, profitable, and just a little more reckless every time they're used. Read a bank, then, not by whether it could survive on its own, but by how much it is leaning on a guarantee it didn't pay the full price for. The closer a bank lives to the backstop, the more carefully you should watch the cushion between its losses and the line where the public starts to pay.

## Further reading & cross-links

- [The anatomy of a bank run, from whisper to collapse](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse) — the run mechanism in full, including the Diamond-Dybvig model in plain English; this post is the cure, that one is the disease.
- [Liquidity management — LCR, NSFR, and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — the rules banks now follow to hold their own cushion of sellable assets, so they need the lender of last resort less often.
- [Stress testing, CCAR, the supervisory exam, and living wills](/blog/trading/banking/stress-testing-ccar-the-supervisory-exam-and-living-wills) — the supervisory machinery, the CAMELS exam, and the resolution plans that are supposed to make orderly failure credible.
- [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the system-level account of the two failures that tested every backstop in this post.
- [Bank capital and leverage — why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — why capital is the true antidote to the moral hazard the backstops create.
- [How money is created — banks, central banks, the money multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) — why only the central bank can be the lender of last resort.
