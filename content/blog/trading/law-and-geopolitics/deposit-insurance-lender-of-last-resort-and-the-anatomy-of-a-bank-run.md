---
title: "Deposit insurance, the lender of last resort, and the anatomy of a bank run"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A bank run is a legal and game-theoretic event. Using Silicon Valley Bank's 48-hour collapse, this is how the FDIC's 250,000-dollar cap, the Fed's discount window, and the resolution authority decide whether a wobble becomes a failure."
tags: ["banking", "deposit-insurance", "fdic", "lender-of-last-resort", "bank-run", "svb", "regulation", "financial-stability", "systemic-risk", "duration-risk"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank run is not a panic that happens *to* a balance sheet; it is a rational, game-theoretic stampede that the *rules* — deposit insurance, the lender of last resort, and the resolution authority — are designed to interrupt, and whether they interrupt it in time decides everything.
>
> - Banks borrow short (deposits payable on demand) and lend long (loans and bonds locked up for years). That **maturity transformation** is what banks are *for*, and it is also what makes them inherently run-prone: a bank that is perfectly solvent can still be killed by a run, because it cannot turn long assets into cash fast enough.
> - The legal backstops are three: **FDIC deposit insurance** (which removes the incentive to run, but only up to a **\$250,000 cap**), the **Fed's discount window and emergency facilities** (which lend cash against good collateral so an illiquid-but-solvent bank survives), and the **resolution authority** (receivership and purchase-and-assumption, which wind a failed bank down without contagion).
> - Silicon Valley Bank is the modern textbook case. A \$100bn-plus bond book bought at near-zero rates lost roughly **\$15–20bn of market value** as the Fed hiked; **94% of its deposits sat above the \$250k cap**; and a **digital-speed run pulled \$42bn in one day (March 9, 2023) with another \$100bn queued** for the next morning. The bank was seized before any backstop could be arranged.
> - The one number to remember: **\$250,000.** Above that line, a depositor is not a saver, they are an unsecured creditor — and creditors run first.

On the morning of Thursday, March 9, 2023, Silicon Valley Bank was, by the conventional measures, a perfectly ordinary mid-sized bank: the 16th-largest in the United States, profitable, with more assets than it had liabilities. By the close of business on Friday, March 10, it had been seized by its regulator and handed to the Federal Deposit Insurance Corporation. It was the largest US bank failure since 2008 and the second-largest in history, and it happened in a little under two business days. No fraud was needed. No loan book went bad. What killed SVB was a combination of an interest-rate move, an accounting category, a base of depositors who were almost entirely uninsured, and a smartphone.

That last item is what made March 2023 a *modern* bank run rather than a re-run of the 1930s. In the old pictures, a run is a queue of people on the sidewalk outside a bank with shuttered doors. In 2023 the queue was a Slack channel and a wire-transfer screen. There were no doors to shutter, because the deposits could leave at the speed of an API call. The classic three-day weekend that regulators rely on to arrange a rescue did not exist; the run outran the rulebook.

This post is about that rulebook — the legal architecture that governs what happens when confidence in a bank cracks. It is genuinely a law-and-policy story, not just a finance story, because every one of the forces that saved or doomed SVB is a *rule*: the \$250,000 insurance cap is a number in a statute; the discount window is a power granted to the Federal Reserve by the Federal Reserve Act; the receivership process is a procedure written into the Federal Deposit Insurance Act; and the decision to make *all* of SVB's depositors whole, even the uninsured ones, required invoking a specific legal escape hatch called the systemic-risk exception. Read the rules and you understand why a wobble at one bank becomes a collapse, why it then spread to Signature, First Republic, and Credit Suisse, and how a trader or investor reads the run-risk setup *before* the seizure rather than after.

![Maturity transformation makes a bank run-prone by design, ending in a legal backstop or receivership](/imgs/blogs/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run-1.png)

## Foundations: how a bank works, and why that makes it fragile

Before the SVB timeline makes sense, you need four building blocks, each of which is both a financial mechanism and a legal construct. Take them slowly; everything later is just these four interacting.

### Maturity transformation: borrowing short to lend long

Picture a simple bank. It takes in \$100 of deposits. Those deposits are *payable on demand* — the depositor can ask for the cash back any morning, no notice required. The bank does not leave that \$100 sitting in a vault, because then it would earn nothing. It lends, say, \$90 of it out: a 30-year mortgage, a 5-year business loan, or a 10-year government bond. It keeps \$10 as a reserve for day-to-day withdrawals.

This is **maturity transformation**: the bank has turned short-term money (deposits it owes *now*) into long-term assets (loans and bonds it will be repaid over *years*). It is the single most important thing a bank does, and it is enormously useful to the economy — it lets savers keep their money liquid while borrowers get long, patient funding. The bank earns the difference between the low rate it pays depositors and the higher rate its long assets yield. That spread is the core of banking.

But notice the trap built into the design. The bank owes \$100 on demand and holds only \$10 in cash. If every depositor showed up tomorrow wanting their money, the bank could pay the first \$10 of them and then would have to *sell its long assets* to raise the rest — and selling a 10-year bond or a 30-year mortgage in an afternoon, at a moment when everyone knows you are a forced seller, means selling at a loss. The bank is solvent in the sense that its assets are worth more than its liabilities if held to maturity, but it is **illiquid** in the sense that it cannot turn those assets into cash fast enough to meet a sudden demand. A bank, by construction, can never honour all its demand deposits at once. It is *designed* to be unable to.

### Illiquidity versus insolvency: the distinction that runs ignore

Hold onto this pair, because it is the hinge of the whole story:

- **Insolvency** means the bank's assets are worth *less* than its liabilities. Even given infinite time, it cannot pay everyone back. The equity has been wiped out. This is a solvency problem — bad loans, fraud, a collapsed business model.
- **Illiquidity** means the bank's assets are worth *more* than its liabilities if held to maturity, but it cannot convert them to cash *right now* without taking a loss large enough to *create* insolvency. This is a timing problem.

The cruel feature of a bank run is that it converts illiquidity into insolvency. A solvent-but-illiquid bank forced to dump assets at fire-sale prices realises losses it never needed to realise, and those losses can be large enough to eat through its equity and make it genuinely insolvent. The run is self-fulfilling: the *fear* that the bank cannot pay produces the *fact* that it cannot pay. This is why a bank run is, in the precise sense we will develop later, a coordination game with a bad equilibrium, not merely a stampede of fools.

### The FDIC and the \$250,000 insurance cap

How do you stop rational depositors from running? You remove their reason to run. If a depositor *knows* with certainty that they will get their money back whether or not they are first in line, the incentive to rush the door disappears. That is exactly what **deposit insurance** does. In the United States the insurer is the **Federal Deposit Insurance Corporation (FDIC)**, created by the Banking Act of 1933 in the depths of the Depression precisely to stop the wave of runs that had shuttered thousands of banks.

The FDIC guarantees deposits up to a legal cap. That cap is, as of 2023, **\$250,000 per depositor, per insured bank, per ownership category**. Below the line, your money is as safe as the full faith and credit of the United States; you have no reason to run, because running gains you nothing. Above the line, you are **uninsured** — and an uninsured depositor is, legally, an unsecured creditor of the bank. If the bank fails, you join the queue of claimants on whatever the receiver can recover, and you may wait months and take a haircut.

That cap is the load-bearing number of this entire post.

![The 250,000-dollar FDIC cap splits every account into safe insured and at-risk uninsured](/imgs/blogs/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run-5.png)

For a household with \$80,000 in checking, the cap is irrelevant — every dollar is insured, and they will sleep through a panic. For a tech startup with \$20m of venture funding parked in its operating account, the cap covers \$250,000 and leaves **\$19.75m uninsured**. That company does not behave like a saver. It behaves like a creditor who has just heard a rumour that the borrower is in trouble — and creditors, faced with a possible default, do not wait politely.

### The lender of last resort: the discount window

The second backstop addresses *illiquidity* directly. If a bank is fundamentally solvent but is facing a sudden cash demand it cannot meet by selling assets in an orderly way, someone can lend it the cash against those assets as collateral, and the bank survives the squeeze. That someone is the central bank, acting as **lender of last resort**.

In the US this function lives at the Federal Reserve's **discount window**, a power granted by Section 10B and the broader Federal Reserve Act. The classic doctrine, articulated by Walter Bagehot in 1873, is *lend freely, against good collateral, at a penalty rate*: in a panic the central bank should hand out unlimited cash to solvent banks so the run cannot force fire sales, but charge enough and demand enough collateral that only genuinely illiquid (not insolvent) banks come knocking. If the lender of last resort works, an illiquid bank borrows cash on Friday, the panic subsides over the weekend, and it repays on Monday. The run never converts into insolvency. (For the deeper mechanics of how the Fed sets the rates these facilities charge, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) and the [legal mandate of a central bank](/blog/trading/law-and-geopolitics/the-legal-mandate-of-a-central-bank), which sets the outer walls of what the Fed is even allowed to do.)

There is a catch, and it is a behavioural one: **discount-window stigma**. Borrowing from the discount window is a signal of distress, and because the borrowing was historically reported (with a lag) and watched closely by analysts, banks fear that *using* the window will itself spook depositors and counterparties — the very thing it is meant to prevent. So banks avoid it until it is almost too late. The Fed has tried for years to dismantle this stigma — renaming the rate, broadening eligible collateral, even publicly urging banks to test their access in calm times so that drawing in a crisis looks routine — with limited success, because the stigma is a coordination problem in its own right: no bank wants to be the *first* to borrow, since being first is itself the signal. We will see that stigma in the SVB story: the discount window was *open*, and it was *not enough*, for reasons that are subtle and important.

It helps to name the three backstops side by side, because they address different failures and a real crisis usually needs all three. **Deposit insurance** attacks the *incentive* to run — it makes the small depositor indifferent to being first, so the run never starts on the insured base. The **lender of last resort** attacks *illiquidity* — it hands cash to a solvent bank so it never has to fire-sell. The **resolution authority** is for when the first two fail and the bank must actually be wound down — it decides, by statute, who gets paid and in what order so that one bank's death does not freeze the system. SVB is instructive precisely because the first backstop barely applied (94% uninsured), the second was outrun (the run was faster than collateral could be posted), and so the whole weight fell on the third — and then on an *extraordinary* override of the third.

### AFS versus HTM: the accounting that hid the loss

The last building block is an accounting rule, and it is the one most people miss. When a bank buys a bond, it must classify it into one of two buckets, and the bucket determines how the bond's price changes show up on the bank's books.

- **Available-for-sale (AFS):** the bond is marked to market each quarter. If its market value falls because rates rose, the unrealised loss flows through to a line in equity called accumulated other comprehensive income. The loss is *visible*.
- **Held-to-maturity (HTM):** the bond is carried at amortised cost — essentially the price the bank paid — and is *not* marked to market on the income statement or in regulatory capital. The bank asserts it will hold the bond to maturity and collect the full face value, so (the logic goes) interim price swings do not matter. The unrealised loss is disclosed only in a footnote.

Here is why this matters. A bond's price moves *opposite* to interest rates: when rates rise, the fixed coupons a bond pays become less attractive than the higher coupons on new bonds, so the old bond's market price falls. A bank that bought long-dated bonds when rates were near zero, and then watched rates climb to over 5%, is sitting on a large *unrealised* loss. If those bonds are in the HTM bucket, that loss is invisible in the headline capital ratio. The bank *looks* well-capitalised. But the loss is real the moment the bank has to *sell* — and a run forces it to sell. HTM accounting let SVB's bond losses build up out of sight until the day a run dragged them into the open. (This sits in the wider story of how disclosure and accounting law shape what investors can even see; for that, see [disclosure and accounting law](/blog/trading/law-and-geopolitics/disclosure-and-accounting-law-sox-ifrs-vs-gaap).)

## The trigger: how rising rates sank the bond book

Now we can assemble the SVB collapse, and it begins not with the run but with the macro backdrop that loaded the gun: the fastest Federal Reserve hiking cycle in four decades.

Through 2020 and 2021, in the pandemic's low-rate environment, Silicon Valley Bank's deposits exploded. Its clients were venture-funded technology and life-sciences startups, awash in cash from a red-hot fundraising market. Deposits roughly tripled, from around \$60bn to over \$180bn in two years. The bank had to do *something* with that money, and lending it out to startups (most of which do not want bank loans) was not an option at that scale. So SVB bought bonds — a great many long-dated US Treasuries and mortgage-backed securities, locking in the low yields available at the time, and parking the bulk of them in the held-to-maturity bucket.

Then inflation arrived, and the Fed responded with the steepest tightening since Paul Volcker. (For the historical rhyme, see [Volcker's 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation).) The federal funds target went from effectively zero in early 2022 to a 5.25–5.50% range by mid-2023 — over five percentage points in roughly sixteen months.

![Fed funds target upper bound rose from near zero to 5.5 percent over 2022 to 2024](/imgs/blogs/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run-3.png)

Every percentage point of that climb pushed down the market value of SVB's long bonds. By the construction above, the loss was invisible in the HTM footnote — until it wasn't.

#### Worked example: a bond book's mark-to-market loss when rates rise

The quick way to estimate how much a bond portfolio loses when rates move is **modified duration**, which measures the percentage change in a bond's price for a one-percentage-point change in yield. The rule of thumb:

```
price change  =  - duration  x  (change in yield)  x  market value
```

Take an SVB-style book to keep the arithmetic clean: a portfolio worth \$100bn with a modified duration of about 6 years (long-dated Treasuries and mortgage securities sit roughly here). Now run rates up by the 5 percentage points (0.05 in decimal) the Fed delivered:

```
loss  =  - 6  x  0.05  x  $100bn  =  - $30bn
```

A \$30bn unrealised loss on a \$100bn book. SVB's actual reported unrealised loss on its held-to-maturity securities reached roughly \$15bn by end-2022 (its real book and duration were a bit different from this round example), and that figure was larger than the bank's entire tangible common equity. In other words: marked honestly, the bond losses had already eaten the bank's capital cushion. The HTM accounting simply meant nobody had to *say* so on the headline ratio.

The intuition: duration turns a rate move into a balance-sheet hole, and a long-duration book held against demandable deposits is a maturity mismatch dressed up as a conservative bond portfolio.

![Mark-to-market loss on a 6-year 100bn-dollar bond book grows as the Fed funds rate rose](/imgs/blogs/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run-8.png)

This is the difference between SVB and the 2008 banks worth dwelling on. In 2008, banks failed because their *assets were bad* — subprime mortgages that would never be repaid. That is insolvency from credit risk. SVB's assets were not bad at all; they were US Treasuries and agency mortgage bonds, about as safe a credit as exists, certain to pay their face value at maturity. SVB's problem was **duration risk** (the price fell as rates rose) colliding with a **funding structure** (demandable, uninsured deposits) that did not give the bonds time to mature. It was a 1930s-style liquidity failure wearing a 2020s costume, not a 2008-style credit failure.

## The run: 94% uninsured and a digital-speed exit

A bank can carry a big unrealised bond loss for years if its depositors stay put. The losses accrete to maturity and unwind harmlessly as the bonds roll off. What turns the latent loss into a fatal one is the *funding side* — and this is where SVB's depositor base was uniquely combustible.

Recall the cap: \$250,000. Recall SVB's clients: venture-backed companies with operating balances in the millions or tens of millions. The result was that an extraordinary share of SVB's deposits sat *above* the insurance line. At end-2022, roughly **94% of SVB's domestic deposits were uninsured** — among the very highest ratios of any large US bank, where 40–60% is more typical. Almost the entire deposit base was, legally, a base of unsecured creditors, every one of whom had both the incentive and the means to flee at the first sign of trouble.

The spark came on Wednesday, March 8, 2023. To raise liquidity, SVB sold a chunk of its *available-for-sale* bonds — crystallising a real, realised loss of about **\$1.8bn** — and announced a plan to raise \$2.25bn of fresh capital to plug the hole. The disclosure was meant to reassure. It did the opposite. It told the market, in black and white, that the bond losses were real and that the bank needed capital to cover them. The tightly networked venture-capital community — a small number of firms who talk to each other constantly and who collectively advise thousands of the startups that banked at SVB — read the filing and, over the following hours, began advising their portfolio companies to pull their money.

What followed was the fastest bank run in history.

![SVB went from solvent to seized in 48 hours across a five-day timeline](/imgs/blogs/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run-2.png)

On **Thursday, March 9**, depositors initiated withdrawals of about **\$42bn in a single day** — roughly a quarter of the entire bank — by wire transfer and app. By the close, SVB had a negative cash balance of nearly \$1bn. Overnight, with the panic now public, customers queued another **\$100bn-plus of withdrawals for Friday morning**, March 10. No bank on earth can meet a demand for \$142bn over two days against a balance sheet of long bonds. On Friday, before the market opened, the California Department of Financial Protection and Innovation closed Silicon Valley Bank and appointed the FDIC as receiver.

![SVB deposit base and the two-day run in March 2023 shown as four bars](/imgs/blogs/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run-4.png)

#### Worked example: the run arithmetic — how much was actually at risk

Start from the curated figures. SVB held about **\$173bn** of total deposits, of which **94%** were uninsured. The dollar amount with a *rational reason to run*:

```
uninsured deposits  =  0.94  x  $173bn  =  $162.6bn
```

So roughly **\$163bn** of SVB's funding base was held by depositors who would lose money — or at least face a frozen, haircut, months-long claim — if the bank failed. Compare that to the **\$42bn** that actually left on March 9: only about a quarter of the uninsured base had moved before the seizure. The remaining three-quarters were the **\$100bn-plus queued** for the next morning. The run was not nearly finished when the regulators stepped in; it was accelerating. Had it run to completion, essentially the entire uninsured base — \$163bn — would have tried to exit.

The intuition: the size of a bank's *run risk* is not its deposit total, it is the slice of deposits sitting above the insurance cap, because only that slice has a financial reason to flee. SVB's run-risk slice was almost the whole bank.

There is a deeper, structural reason the speed mattered, and it is legal as much as technological. The entire US bank-resolution apparatus is built around the assumption of a **three-day weekend**: a regulator seizes a bank after the Friday close, the FDIC works through Saturday and Sunday to find a buyer or arrange a payout, and the bank reopens — often under a new owner — by Monday morning. That cadence is baked into the receivership machinery, the bidding process for purchase-and-assumption, and the operational reality of moving a bank's systems. A pre-2010 run, conducted by people physically queueing at branches during business hours, gave regulators that weekend. SVB's run, conducted by treasurers clicking "send wire" on Thursday and queuing \$100bn more for Friday, did not. The run threatened to empty the bank *before the weekend the law assumes for the rescue even began.* The digital deposit base did not just make the run faster; it broke the timing assumption embedded in the legal process for stopping runs. Regulators have since openly worried that the next run could be faster still, and that the rulebook needs a continuous (not weekend-batch) resolution capability to match.

This is why the discount window — the lender of last resort — did not, and arguably could not, save SVB. The window was open; the Fed was ready to lend against collateral. But two things broke the rescue. First, **speed**: pledging collateral to the Fed and drawing cash is an operational process measured in hours and days, and SVB's run was measured in the same units; the bank simply ran out of time over a single Thursday-into-Friday. Second, **scale and collateral**: even lending freely against good collateral, the Fed lends against the *market value* of the bonds, which had fallen — so a bank trying to borrow against \$100bn of bonds now worth \$85bn could raise only \$85bn-ish, not the \$142bn it suddenly needed. The discount window can rescue a bank facing a liquidity squeeze of a few percent of its balance sheet. It cannot rescue a bank from which nearly the entire deposit base is sprinting for the exit faster than collateral can be posted. The backstop was real; it was simply outrun.

## Why the run was rational: the depositor's game

It is tempting to call a bank run irrational — a herd panic, a failure of nerve. It is not. For an uninsured depositor, running is the *rational* choice, and that is precisely what makes runs so dangerous: they are a stable equilibrium of self-interested behaviour, not a temporary madness that calm heads can talk down. This is the famous Diamond–Dybvig model of bank runs, and you can capture its core in a simple payoff matrix.

![A bank run is a coordination game where running first weakly dominates waiting](/imgs/blogs/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run-7.png)

Put yourself in the seat of an uninsured depositor — the startup CFO with \$20m at SVB. You have two choices, *wait* or *run*, and your payoff depends on what everyone else does.

- If **everyone waits**, the bank survives, you keep 100 cents on the dollar, and life goes on. Waiting was fine.
- If **everyone runs** and you *wait*, you are last in the queue; the bank fails, its assets are fire-sold, and you recover whatever the receiver eventually scrapes together — say 60 cents on the dollar, paid months later. Waiting was a disaster.
- If **everyone runs** and you *also run early*, you get your wire out before the bank is frozen, and you keep close to 100 cents. Running was smart.

The structure is the trap. *Running is never worse than waiting, and is sometimes much better.* It weakly dominates. And critically, the worst outcome — being a patient depositor in a bank everyone else is fleeing — is one you can only avoid by running yourself. So the moment you assign any meaningful probability to *others* running, your best response is to run too. Everyone reasons identically, and the run becomes self-fulfilling. The bank does not fail because it is insolvent; it fails because each depositor, acting rationally on the fear that others will act rationally, makes the collective outcome inevitable.

#### Worked example: a depositor's run-versus-wait expected value

Make the game concrete with the CFO's \$20m. Suppose, after the March 8 disclosure, the CFO assesses a 50% chance the bank survives and a 50% chance it fails. Compare the two strategies.

**If you WAIT:**
- Bank survives (50%): recover 100% → \$20.0m
- Bank fails and you were slow (50%): recover ~60% as a receiver claim, paid later → \$12.0m

```
E[wait]  =  0.5 x $20.0m  +  0.5 x $12.0m  =  $16.0m
```

**If you RUN (wire out Thursday morning):**
- You get your money out before any freeze (say 95% likely you beat the queue): \$20.0m
- You are caught by the freeze anyway (5%): \$12.0m

```
E[run]  =  0.95 x $20.0m  +  0.05 x $12.0m  =  $19.6m
```

Running is worth **\$19.6m versus \$16.0m for waiting** — a \$3.6m edge — and it *removes* the catastrophic tail of being last in line. No rational treasurer chooses to leave \$3.6m of expected value on the table to be a good citizen. And because every treasurer runs the same calculation, the 50% failure probability they assumed becomes nearer 100% — the assessment is self-fulfilling.

The intuition: deposit insurance works not by reimbursing losses after the fact but by *changing this payoff matrix in advance* — below the \$250k cap, waiting and running both pay 100 cents for certain, so the dominant strategy to run simply disappears. SVB's tragedy is that 94% of its depositors were playing the uninsured game.

## The policy response: a systemic-risk exception and a new facility

By the weekend of March 11–12, 2023, the authorities faced a fast-spreading problem. SVB's uninsured depositors — thousands of startups and the venture funds behind them — were facing the prospect of frozen operating accounts, meaning they could not make payroll on Monday. Worse, every *other* regional bank with a high uninsured-deposit ratio now looked like the next SVB, and depositors at those banks had every incentive to pre-emptively run. A second bank, **Signature Bank** of New York, was already failing the same way. The run was threatening to jump from one bank to the whole regional-banking system. (This is the same contagion logic that animates the [shadow-banking and repo](/blog/trading/finance/shadow-banking-and-the-repo-market) plumbing, where one institution's distress freezes funding for all.)

The legal default would have been brutal: the FDIC insures up to \$250k, full stop, so SVB's uninsured 94% would get a partial advance and then a receiver's certificate — an IOU against future asset recoveries. To avoid the cascade, the authorities reached for a specific legal escape hatch.

### The systemic-risk exception

Ordinarily the FDIC is bound by a **least-cost requirement** written into the Federal Deposit Insurance Act after the 1991 reforms: when resolving a failed bank, it must choose the option that costs the *insurance fund* the least, which generally means protecting insured depositors only and letting uninsured creditors take their lumps. Protecting *uninsured* depositors is normally illegal precisely because it would cost the fund more.

But the statute contains an override: the **systemic-risk exception**. If the Secretary of the Treasury, in consultation with the President, and upon the recommendation of two-thirds of both the FDIC board and the Federal Reserve board, determines that complying with the least-cost rule "would have serious adverse effects on economic conditions or financial stability," the FDIC may take actions — including guaranteeing *all* deposits — that depart from least-cost. This is the legal lever pulled in 2008 and again, on Sunday March 12, 2023, for SVB and Signature. The announcement: **all depositors, insured and uninsured, would be made whole.** The \$250k cap was, for these two banks, effectively suspended.

This is the cleanest illustration in the whole post of the series' thesis — that a *rule*, and the legal procedure to invoke its exception, is what moved markets. The moment the systemic-risk exception was announced, the run-risk equation at every regional bank changed: depositors no longer had to fear being above the cap, because the precedent now suggested the government would cover them. The announcement was the backstop.

### The Bank Term Funding Program

Alongside the deposit guarantee, the Fed created a brand-new lending facility on March 12, the **Bank Term Funding Program (BTFP)**, to fix the *collateral-haircut* problem that had hamstrung the discount window. Recall the issue: the discount window lends against the *market value* of pledged bonds, which had fallen. The BTFP changed the terms in one decisive way — it lent against the bonds' **par (face) value**, not their depressed market value, for up to one year.

This is a remarkable concession. It says, in effect: *we will pretend your underwater bonds are worth their full face value for the purpose of lending you cash.* Any bank with a portfolio of HTM Treasuries and agency bonds sitting on unrealised losses could now post them at par and borrow the cash it needed, without crystallising the loss by selling. The maturity-mismatch trap — being forced to sell long bonds at a loss to meet deposit outflows — was, for any bank that could survive long enough to use the facility, defused.

#### Worked example: the BTFP par-vs-market arbitrage

The BTFP created an almost literal arbitrage, which is why it both worked and drew criticism. Consider a bank holding a \$100m (face value) bond that, because rates rose, now trades at \$85m in the market.

**Under the old discount window** (lend against market value):
```
cash the bank can raise  =  $85m
```
The bank is \$15m short of its bonds' face value; selling to raise \$100m would crystallise a \$15m loss.

**Under the BTFP** (lend against par):
```
cash the bank can raise  =  $100m  (full face value)
```

Now add the cherry. The BTFP charged roughly the one-year overnight index swap rate plus 10 basis points — call it about **4.9%** in early 2023. But the cash, once raised, could be parked back at the Fed in the reserve-balances rate or rolled into T-bills yielding around **5.3–5.4%**. So:

```
borrow at  ~4.9%  against bonds at par
reinvest at  ~5.4%  in risk-free bills
net spread  =  5.4%  -  4.9%  =  ~0.5%  =  ~50 bps, risk-free
```

On \$100m, that is roughly **\$500,000 a year of riskless carry**, *plus* the elimination of the \$15m fire-sale loss. Banks noticed: BTFP balances climbed to over \$160bn by early 2024, partly from genuine liquidity need and partly from this carry trade, which is why the Fed adjusted the rate and closed the facility to new loans in March 2024.

The intuition: by lending at par against bonds worth less, the BTFP transferred the duration loss off the banks' immediate problem and onto the Fed's balance sheet for a year — the rule did not make the losses vanish, it just bought time and quietly subsidised the banks while it did.

## The moral-hazard debate and the deposit-insurance question

The systemic-risk exception and the BTFP stopped the run, but they reopened the oldest argument in banking regulation: **moral hazard.** The term means that protecting people from the consequences of risk makes them take more of it. If the government will make uninsured depositors whole whenever a failure looks "systemic," then large depositors have no reason to monitor their bank's risk-taking — and banks, knowing their depositors will not flee, have less reason to manage duration and liquidity prudently. The 2023 response, critics argued, effectively extended an *implicit, unlimited* deposit guarantee to the whole system while charging premiums calibrated to a \$250k *explicit* one. You get the discipline-free funding of a guarantee without anyone having paid for the guarantee.

The defenders' reply is the one that always wins in the moment: a run is a *collective* failure, and letting thousands of blameless startups miss payroll to teach their treasurers a lesson about counterparty risk would have caused more damage than the moral hazard it deterred. Both sides are correct, which is why the debate never resolves — it just oscillates between "never again will we bail out the reckless" in calm times and "we have no choice, the system is at stake" in panics. The 2008 crisis, the 2023 episode, and every banking panic before them have run this same loop. (The post-2008 leg of the loop — the stress tests and resolution rules meant to make bailouts unnecessary — is the subject of [Dodd-Frank: the post-2008 rulebook](/blog/trading/law-and-geopolitics/dodd-frank-the-post-2008-rulebook).)

SVB also exposed a specific regulatory gap that is pure law-and-policy. After 2008, the Dodd-Frank Act subjected the largest banks to tough liquidity rules (the Liquidity Coverage Ratio) and annual stress tests, and required them to hold capital against unrealised AFS losses. But a **2018 amendment** raised the asset threshold for the strictest oversight from \$50bn to \$250bn — and SVB, at roughly \$210bn of assets, sat just *under* the new line. It was therefore exempted from the full liquidity-coverage requirement, from the most stringent stress tests, and (as a sub-threshold bank) was allowed to *opt out* of recognising AFS losses in its regulatory capital. A bank one tier larger would have been forced to hold liquid assets against exactly the run that killed SVB, and to show its bond losses in its capital ratio. The failure was, in part, a *direct consequence of where the legal threshold was drawn* — a clean example of the series thesis that the precise wording of a statute moves markets.

That has driven a live **deposit-insurance reform** debate with three broad options on the table, each a different rule with different market consequences:

1. **Raise or remove the \$250k cap** — simplest, but the most moral hazard, since it guarantees even the largest, most sophisticated depositors.
2. **Unlimited insurance for business *transaction* accounts only** — the FDIC's own preferred option in its 2023 review, on the logic that payroll and operating accounts are the systemic pressure point (a company *must* keep its payroll somewhere) while investment balances can be expected to bear risk.
3. **Keep the cap but improve the plumbing** — faster resolution, pre-positioned collateral at the discount window, and tighter liquidity rules for mid-sized banks, so that an illiquid-but-solvent bank can actually be saved before a run completes.

Whichever path the law takes will reprice bank funding: a higher guarantee makes deposits stickier and bank equity safer but raises the implicit subsidy; keeping the cap keeps the run risk that the market now prices into uninsured-deposit-heavy names. An investor reads the *direction* of this reform debate as a slow-moving input into how much of a discount to apply to flighty-funded banks.

## The contagion: Signature, First Republic, and Credit Suisse

A run does not respect the boundaries of one bank. Once depositors learn that uninsured money can vanish, they re-examine *their own* bank, and the run jumps to whichever institution looks most like the one that just failed.

**Signature Bank** failed the same weekend as SVB, seized on Sunday March 12. It had a similar profile: a large uninsured-deposit base, this time heavily concentrated in crypto-industry clients, and it experienced its own digital run.

**First Republic Bank** was the next domino and the most revealing, because it died slowly over seven weeks rather than in two days. First Republic served wealthy households on the US coasts — again, an unusually high share of uninsured deposits, and a balance sheet stuffed with low-rate jumbo mortgages whose value had fallen as rates rose: the *same* duration-plus-uninsured-funding setup as SVB. After SVB failed, eleven large banks injected \$30bn of deposits into First Republic in a show of confidence, but the deposit flight continued. On May 1, 2023, the FDIC seized First Republic and sold it to JPMorgan in a purchase-and-assumption deal — the largest US bank failure since Washington Mutual in 2008.

**Credit Suisse** showed the contagion was not even confined to the US. The Swiss bank had its own long-running troubles, but the March 2023 loss of confidence accelerated a deposit and client-asset flight that forced a Swiss-authority-brokered emergency takeover by UBS over the weekend of March 18–19, complete with its own controversial legal manoeuvre — the writedown of \$17bn of Credit Suisse's Additional Tier 1 (AT1) bonds to zero *while shareholders still received some value*, inverting the normal creditor hierarchy and roiling the global AT1 market for months. (Why bank capital is layered this way, and why AT1 sits where it does, is the subject of [Basel III/IV bank capital rules](/blog/trading/law-and-geopolitics/basel-iii-iv-bank-capital-rules-and-the-price-of-credit).)

The throughline: every one of these failures was the same rule-driven mechanism — a duration-impaired asset book funded by runnable, largely uninsured deposits — and each was resolved by a different legal tool (systemic-risk exception, purchase-and-assumption, a state-brokered merger with a creditor-bail-in).

## The bank-resolution process: what the FDIC actually does

Step back from the firefight and look at the orderly machinery the law provides for winding down a failed bank, because reading it tells you what your money is worth in each scenario.

![How the FDIC resolves a failed bank over a weekend, from seizure to receivership to purchase and assumption](/imgs/blogs/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run-6.png)

When a bank fails, its chartering authority (a state regulator or the federal OCC) pulls the charter and appoints the **FDIC as receiver**. This typically happens after the Friday close so the FDIC has the weekend. As receiver, the FDIC steps into the bank's shoes: it takes control of the assets, freezes the books, and chooses a resolution path under the least-cost test:

1. **Purchase and assumption (P&A):** the FDIC finds a healthy acquiring bank to *assume* the failed bank's deposits and *purchase* its good assets, often with the FDIC absorbing some losses or sharing them. Depositors barely notice — their accounts simply reopen under the new bank's name on Monday. This is the FDIC's preferred outcome (First Republic → JPMorgan was a P&A). Insured *and* (in a P&A of the whole deposit book) sometimes uninsured depositors are protected, because the buyer takes the deposits whole.
2. **Deposit payout:** if no buyer can be found, the FDIC pays insured depositors directly, up to the \$250k cap, usually within a day or two. Uninsured depositors receive an advance dividend plus a **receiver's certificate** — a claim on the proceeds as the receiver liquidates the remaining assets over months or years, recovering some fraction.
3. **Systemic-risk exception:** as described above, the override that lets the FDIC protect *all* depositors when a least-cost resolution would threaten the system. SVB and Signature were ultimately resolved with all deposits protected via bridge banks and this exception.

The **creditor hierarchy** in a receivership is fixed by statute and is worth memorising, because it determines who gets paid in what order from the recovered assets: depositors (insured first, then uninsured) rank *ahead* of general unsecured creditors and bondholders, who rank ahead of equity holders, who are usually wiped out entirely. This is why even an uninsured depositor at a failed bank often eventually recovers most of their money — they sit near the top of the queue — but the key word is *eventually*, and "I'll get most of it back in eighteen months" is no comfort to a company that needs to make payroll on Monday. The *timing*, not the ultimate recovery, is what makes a run rational.

## Common misconceptions

**"SVB failed because it made bad loans, like the 2008 banks."** No — and the numbers make the distinction sharp. SVB's losses came from US Treasuries and agency mortgage-backed securities, among the safest credits in existence, certain to repay face value at maturity. There was essentially *zero credit loss*. The roughly \$15bn of unrealised loss was pure **duration risk** — the bonds' market price fell as the Fed hiked rates by over 5 points — colliding with a funding base that gave the bonds no time to mature. 2008 was an insolvency-from-bad-assets story; 2023 was a liquidity-from-good-assets story. Confusing the two leads you to screen for the wrong risk entirely.

**"Deposit insurance means my money is always safe."** Only up to **\$250,000 per depositor, per bank, per ownership category.** SVB is the proof: **94% of its deposits — about \$163bn — sat above that line and were, at the moment of failure, legally uninsured.** Those depositors were only made whole because the government invoked the *extraordinary* systemic-risk exception, which is discretionary, not guaranteed, and was a political and legal decision made over a single weekend. Counting on it is counting on a bailout that the law explicitly frames as an exception, not a right. If your balance exceeds the cap, you are an unsecured creditor, full stop.

**"The Fed's discount window would have saved SVB if they'd just used it."** This is half-true and dangerously so. The window was open and the Fed was willing to lend. But the discount window lends against *market value* and at operational *speed* — and SVB's run was both too fast (\$42bn out in a day, \$100bn-plus queued overnight) and too large relative to the cash its depreciated collateral could raise. A bank trying to borrow against \$100bn of bonds now worth \$85bn cannot conjure the \$142bn it suddenly owes. The lender of last resort defuses a *liquidity squeeze of a few percent*; it cannot outrun a near-total deposit flight. The proof is that the authorities had to *invent* the BTFP (lending at *par*, not market) the very next weekend, precisely because the existing window was structurally insufficient for the problem.

**"A bank run is irrational panic."** The opposite. For an uninsured depositor, running weakly dominates waiting — running is never worse and is sometimes much better, as the \$19.6m-versus-\$16.0m expected-value example showed. A run is a *rational* coordination outcome, which is exactly why it is so hard to stop with reassurance: you cannot talk depositors out of acting in their own interest. You can only change the incentives, which is what insurance (below the cap) and the systemic-risk guarantee (above it) do.

## How it shows up in real markets

The SVB episode is not just history; it left a set of repricings that show how a banking-rule event transmits into asset prices — the series' core spine of law/policy → macro → prices.

**The regional-bank selloff (KRE).** The clearest market expression was the collapse of the regional-bank equity complex. The SPDR S&P Regional Banking ETF (ticker KRE), a basket of US regional and mid-sized banks, fell roughly 30% in the two-and-a-half weeks around the SVB and Signature failures, with the steepest single-day drops on March 10 and March 13, 2023. The market was repricing *the entire category* — not because every regional bank had SVB's exact problem, but because investors suddenly could not tell which ones did, and re-rated all of them for the newly visible risk. Within the index, the banks that fell most were precisely those with the highest uninsured-deposit ratios and the largest HTM unrealised losses — the market was, in real time, running the screen we will build in the playbook.

**The flight to T-bills and money-market funds.** Where did the deposits go? Much of the money that left regional banks did not sit idle; it moved into **government money-market funds and Treasury bills**, which offered yields comparable to or better than bank deposits *with* the safety of direct or near-direct government backing and no \$250k cap to worry about. Money-market fund assets surged by hundreds of billions of dollars in the weeks after SVB. This is a textbook flight to quality, and it tightened funding for banks further (deposits are a bank's cheap funding; losing them to MMFs raises the bank's cost of money), creating a second-order squeeze on bank profitability that persisted long after the acute panic faded.

**A contained volatility spike, not a 2008-scale one.** It is worth noting what *didn't* happen, because it tells you the backstop worked. The CBOE Volatility Index (VIX) — the market's "fear gauge" — rose to about **26.5 on March 13, 2023**, an elevated but not extreme reading. Compare that to **82.7 at the COVID crash** in March 2020 or **37.3 in the 2018 "Volmageddon"** episode: the SVB panic registered as a sharp, sector-specific stress, not a system-wide seizure. The reason is exactly the policy response — the weekend guarantee and the BTFP capped the tail before it could metastasise into a broad credit crunch. For a trader, the *muted* VIX print alongside a *violent* regional-bank selloff was itself a signal: the market judged the problem ring-fenced to a category, not a repeat of 2008, which is precisely why the regional-bank short was a *category* trade rather than an everything-down trade.

**The repricing of uninsured-deposit-heavy banks.** Even after the systemic-risk exception calmed the immediate run, the equity market durably re-rated banks by their deposit *quality*. A bank funded by sticky, insured, retail deposits earned a premium; a bank funded by flighty, uninsured, concentrated deposits earned a discount, reflected in a lower price-to-book multiple. "Deposit beta" and "uninsured-deposit percentage" went from footnote metrics that few equity analysts tracked to front-page screening criteria. The rule-driven event permanently changed *which fundamental the market prices*.

## How to trade it: the bank-run playbook

The whole point of reading the rules is to read the setup before the seizure. Here is how a practitioner turns the SVB anatomy into a repeatable process.

**1. Screen for the two ingredients.** A run-prone bank needs *both* a runnable funding base and an impaired asset book. Screen the regulatory filings (US bank call reports and 10-Ks disclose both) for:
- **Uninsured-deposit percentage.** This is the runnable slice. SVB's 94% was extreme; anything well above the ~50% norm is a flag, especially if the deposits are *concentrated* in one industry or a small number of large accounts (concentration means the depositors talk to each other and run together).
- **HTM unrealised losses relative to tangible common equity.** Pull the held-to-maturity footnote and compare the unrealised loss to the bank's equity. If marking the HTM book to market would wipe out a large share of capital, the bank is solvent *only because the accounting says so* — and a run would force the mark.

A bank that scores high on *both* is the SVB setup. A bank with high uninsured deposits but a short-duration, well-hedged asset book is far safer; so is a bank with big HTM losses but a sticky insured retail base that will not run.

#### Worked example: scoring a bank for run risk

Take a hypothetical regional bank: \$120bn deposits, 70% uninsured, and an HTM book whose footnote discloses a \$9bn unrealised loss against \$10bn of tangible common equity.

```
runnable base   =  0.70  x  $120bn   =  $84bn  uninsured
capital-at-risk =  $9bn loss / $10bn equity  =  90% of equity
```

Both numbers are flashing. \$84bn could rationally flee, and marking the bond book to market would erase 90% of the equity cushion. This bank does not need bad loans to fail; it needs a *headline* — a peer failure, a downgrade, a capital-raise announcement — that makes its uninsured base recalculate the run-versus-wait game. The screen tells you the powder is dry; it does not tell you the date of the spark.

The intuition: run risk is the product of *how much money can rationally leave* and *how little it would take to break the bank if it tries*, and SVB scored near the maximum on both.

**2. Read the spark.** The fuel sits for months; the ignition is usually a *disclosure or an action that forces the latent loss into view.* SVB's spark was a capital-raise announcement and a realised \$1.8bn loss. Watch for: a surprise equity raise (it screams "we have a hole"), a forced AFS bond sale, a credit-rating downgrade, a large-depositor exit becoming public, or — the most dangerous in the digital age — a peer failure that makes every similar bank's depositors recalculate at once. The lesson of the 36-hour SVB run is that in the smartphone era the gap between spark and seizure can be a single day.

**3. Position for the contagion, not just the name.** The tradeable move is rarely the single failing bank (which often gaps down or is halted before you can act); it is the *re-rating of the category*. The KRE selloff was the trade. A practitioner who had pre-screened the regional-bank complex for the two ingredients could, on the SVB spark, short or buy puts on the basket (or the worst-scoring individual names within it) and be long the obvious beneficiaries — the largest "too-big-to-fail" banks that *gain* deposits in a flight to safety, government money-market funds, and short-dated Treasuries. The structural pair trade of a banking panic is *short the flighty-funded regionals, long the deposit-magnet megabanks and T-bills*.

**4. Know what the backstop does to the trade.** The single biggest risk to a short-the-regionals position is a *policy response*: the moment authorities invoke a systemic-risk exception, stand up a facility like the BTFP, or signal an all-deposits guarantee, the run-risk premium collapses and the regional-bank selloff can violently reverse. After March 12, 2023, KRE bounced hard off its lows as the guarantee and BTFP took the tail risk off the table. So the playbook is asymmetric and time-sensitive: the short works in the *window between spark and backstop*, which in a fast modern run may be only a weekend. You are trading the *gap*, and the gap closes when the rulebook's emergency powers are deployed.

**5. What invalidates the thesis.** Be honest about what would make you wrong:
- **A credible, pre-emptive backstop.** If regulators signal early and credibly that all deposits are safe, the run dynamic dies and the short fails. Watch the Treasury, Fed, and FDIC statements minute by minute during a panic.
- **The bank's funding turns out to be sticky.** If a high-uninsured-ratio bank's depositors are operationally captive (their payroll, their treasury system, their lending relationship are all locked in), they may not run despite the incentive. SVB's depositors were unusually mobile *and* unusually networked; not every bank's are.
- **The asset book is hedged.** A bank with a big *nominal* HTM loss that has hedged its duration with interest-rate swaps is far less fragile than the raw footnote suggests. Read the hedging disclosures, not just the unrealised-loss line.
- **Rates fall.** The entire setup is a duration story. If the Fed pivots and long rates drop, HTM losses shrink, the capital hole closes, and the most fragile banks heal without ever being tested. The run risk you screened for can evaporate with the yield curve. (For how the curve itself moves, see the [fixed income](/blog/trading/fixed-income) primers on duration and rates.)

The deepest lesson sits underneath all five steps and ties back to the series spine. A bank run looks like a market event — falling stock prices, fleeing deposits, a category re-rating — but it is *generated by rules*: the \$250k cap that defines who runs, the discount window and BTFP that determine whether an illiquid bank survives, the least-cost test and its systemic-risk override that decide who gets made whole, and the creditor hierarchy that sets recovery values. Read the rules first, and the market moves stop looking like panic and start looking like the predictable output of a game whose payoffs the law has written. (For the broader machinery of how a rule becomes a price, the spine post is [how law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain).)

## Further reading & cross-links

Within this series:
- [Basel III/IV: bank capital rules and the price of credit](/blog/trading/law-and-geopolitics/basel-iii-iv-bank-capital-rules-and-the-price-of-credit) — why bank capital is layered, and why the Credit Suisse AT1 writedown was so shocking.
- [The legal mandate of a central bank](/blog/trading/law-and-geopolitics/the-legal-mandate-of-a-central-bank) — the Federal Reserve Act, the discount window power, and the emergency 13(3) authority that frames the lender-of-last-resort function.
- [Dodd-Frank: the post-2008 rulebook](/blog/trading/law-and-geopolitics/dodd-frank-the-post-2008-rulebook) — the stress-test and resolution regime built after 2008, and the 2018 rollback that exempted mid-sized banks like SVB from the toughest liquidity rules.
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the spine: how a rule change becomes a price.
- [Disclosure and accounting law: SOX, IFRS vs GAAP](/blog/trading/law-and-geopolitics/disclosure-and-accounting-law-sox-ifrs-vs-gaap) — why the AFS-versus-HTM accounting choice that hid SVB's losses exists.

Cross-asset and mechanism:
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — how funding panics propagate through the non-bank plumbing.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the rate machinery whose 5-point move sank SVB's bond book.
- [Paul Volcker's 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation) — the historical rhyme of a fast hiking cycle breaking something.
- [Fixed income](/blog/trading/fixed-income) — duration, convexity, and the yield curve that drive a bank's bond-book risk.
