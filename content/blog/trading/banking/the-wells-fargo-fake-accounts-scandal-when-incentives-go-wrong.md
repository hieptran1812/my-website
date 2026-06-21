---
title: "The Wells Fargo Fake-Accounts Scandal: When Incentives Go Wrong"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a cross-sell quota culture pushed Wells Fargo employees to open millions of unauthorized accounts, why every control missed it, and what about 4.9 billion dollars in fines and an asset cap teach about conduct risk."
tags: ["banking", "conduct-risk", "wells-fargo", "incentives", "governance", "cross-selling", "three-lines-of-defense", "operational-risk", "compliance", "bank-failures"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR** — Wells Fargo did not lose money on a bad loan or a rogue trader; it broke the most valuable thing a bank owns, customer trust, by paying its own staff to commit fraud one quota at a time.
>
> - A relentless cross-sell target ("Eight is Great" — eight products per household) turned into daily branch quotas that staff could only hit by opening accounts customers never asked for; about **3.5 million** unauthorized accounts were eventually identified.
> - Every layer that was supposed to catch it missed it: the business that owned the risk created the pressure, the second-line risk and compliance function had no real power to challenge sales, and internal audit fired the symptoms (employees) instead of the disease (the incentive).
> - The bill came to roughly **\$4.9 billion** in fines (\$185M in 2016, \$3bn in 2020, \$1.7bn in 2022) plus an unprecedented Federal Reserve **asset cap** that froze the bank near **\$1.95 trillion** for years, costing far more than every fake account ever earned.
> - The one number to remember: a single \$25 monthly maintenance fee on an account a customer never wanted is worth almost nothing to the bank, but the conduct failure behind it cost about **\$4.9 billion** — a roughly *million-to-one* loss ratio. That asymmetry is the whole lesson of conduct risk.

In September 2016, a regulator most Americans had never heard of, the Consumer Financial Protection Bureau, announced a \$185 million settlement with what was then the most admired bank in the United States. The number was almost a footnote next to the crisis-era fines of the previous decade. The detail underneath it was not. Wells Fargo, the bank Warren Buffett had called the best-run in the country, had been opening accounts for its own customers without their knowledge — checking accounts, savings accounts, credit cards, online bill-pay — and it had been doing so on an industrial scale, for years, with the full knowledge of managers who were measured and paid on exactly the behavior that produced it.

Within weeks the chief executive was hauled in front of a Senate committee and told, on live television, "You should resign." He was gone within a month. The stock fell. The board clawed back tens of millions in pay. And the number kept growing: an outside review eventually put the count of potentially unauthorized accounts at about 3.5 million, the fines climbed past \$4.9 billion, and in February 2018 the Federal Reserve did something it had never done to a major bank before — it capped Wells Fargo's total assets, forbidding it to grow until it fixed its culture. That cap stayed in place for over six years.

This is not a story about a clever fraud or a sophisticated financial weapon. It is a story about a quota. The diagram above is the mental model for the whole post: a single number set in a boardroom, pushed down an org chart, multiplied by fear of being fired, and turned into millions of small crimes at the teller window. That is **conduct risk** — and it is the one risk that can be created entirely by the bank's own management, sitting in plain sight, while every dashboard glows green.

![Org chart flow from a cross-sell quota down to mass account fraud](/imgs/blogs/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong-1.png)

## Foundations: conduct risk, cross-selling, and the lines of defense

Before we walk through what happened, we need a small vocabulary. Banking has a habit of giving plain ideas intimidating names, so let us define each one from zero, in order.

**Conduct risk.** Most of this series has been about risks that arrive from *outside* the bank: borrowers who do not repay (credit risk), markets that move against a position (market risk), depositors who all want their money back at once (liquidity risk). Conduct risk is different. It is the risk that the bank's *own people*, behaving the way the bank's *own systems* reward them to behave, harm customers, markets, or the bank itself. It is a sub-species of **operational risk** — the catch-all for losses from people, processes, and systems rather than from lending or trading. If you have read [operational risk: fraud, cyber and the loss events](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events), conduct risk is the corner of that map where the "people" failure is not a rogue individual but a designed-in incentive. The defining feature of conduct risk is that it is *self-inflicted*: no recession, no rate shock, no counterparty triggered it. The bank did it to itself.

**Cross-selling.** When a bank sells you a second product, that is cross-selling. You came in for a checking account; the bank would also like you to have its savings account, its credit card, its mortgage, its auto loan, its investment account. There is nothing wrong with this in principle — quite the opposite. A customer with eight products is "sticky": they rarely leave, they generate fees and balances across many lines, and the bank earns more from the relationship it already paid to acquire. Industry research has long shown that the *cost of acquiring* a customer is far higher than the *cost of selling more* to one you already have. Cross-selling is, genuinely, good banking. The problem is never the goal. The problem is what happens when the goal becomes a quota.

**Sales incentives and quotas.** An incentive is anything that rewards a behavior: a bonus, a commission, a public ranking, or simply keeping your job. A quota is an incentive with a hard floor — a *minimum* you must hit, where missing it is punished. The distinction matters enormously. A bonus for exceeding a target makes you try harder at a legitimate sale. A quota that fires you for missing it makes you do *whatever it takes*, because the alternative is unemployment. Wells Fargo's system was the second kind, set absurdly high, measured by the hour, and enforced by the threat of termination.

**The principal-agent problem.** This is the deepest idea in the post, so we will build it slowly. A *principal* is someone who wants a job done; an *agent* is someone hired to do it. The problem is that the agent has their own interests, which need not match the principal's. The bank's owners (shareholders) want long-term, profitable, *real* relationships. The branch employee wants to keep their job this week. When the bank measures the employee on *account openings* rather than on *real, funded, used accounts*, it has handed the agent an incentive to do something the principal never wanted. Economists call the fix "aligning incentives." Wells Fargo did the opposite — it built an incentive that pointed the agent directly at fraud — and then acted surprised when the agent followed it.

#### Worked example: how cheap a fake account was to the bank

> [!note]
> Here is the asymmetry at the heart of the scandal, in numbers. Suppose a fake checking account is opened with no money in it. The bank earns essentially nothing — many such accounts had a balance of \$0 and were quietly closed. Even an account that triggers a \$25 monthly maintenance fee for a few months before the customer notices earns the bank, say, \$25 × 3 = **\$75**, gross, before the cost of refunding it.
>
> Now put that next to the total cost of the scandal. The fines alone came to about **\$4,900,000,000**. Divide: \$4,900,000,000 ÷ \$75 ≈ **65 million** fake accounts' worth of fees to "pay back" the fines — against roughly 3.5 million accounts actually opened, most of which earned far less than \$75.
>
> The intuition: the upside of a single gamed account was a rounding error; the downside, aggregated and discovered, was company-threatening. Conduct risk is the business of measuring exactly that asymmetry — and Wells Fargo's incentive system was blind to it.

**The three lines of defense.** This is the standard mental model for how a bank is *supposed* to control itself, and it will be the spine of our "why nobody caught it" section, so learn it now. Picture three concentric rings of responsibility:

- **First line: the business.** The people who actually take the risk — here, the branch managers and bankers who sell products. They own the risk they create and are supposed to control it day to day.
- **Second line: risk and compliance.** Independent functions whose entire job is to *challenge* the first line — to set limits, question incentives, and say "no" when the business wants to do something dangerous. Their power comes from independence: they must not report to the people they police.
- **Third line: internal audit.** A separate team that checks whether the first two lines are actually working, and reports straight to the board's audit committee, not to management.

The whole point of three independent lines is redundancy: for a problem to go uncaught, *all three* have to fail. At Wells Fargo, all three did. We will see exactly how.

**Consent order.** Finally, a piece of regulator vocabulary. When a US bank regulator (the CFPB, the OCC, the Federal Reserve) finds wrongdoing, it does not usually go to trial. It issues a **consent order** — a binding legal agreement the bank signs, admitting nothing or something, paying a fine, and promising specific fixes (replace these executives, build these controls, hire this monitor). A consent order is enforceable: violate it and the penalties escalate. Wells Fargo collected a stack of them, and the Fed's 2018 asset cap was attached to one. They are the legal machinery through which a culture failure becomes a balance-sheet event.

A useful way to hold these three regulators apart, since they all appear in the story: the **CFPB** (Consumer Financial Protection Bureau) protects retail customers and brings the consumer-harm cases; the **OCC** (Office of the Comptroller of the Currency) is the prudential supervisor of national banks and cares about safety, soundness, and governance; and the **Federal Reserve** sits above the holding company and worries about systemic risk and the quality of the board and senior management. Wells Fargo managed to draw the anger of all three at once — consumer harm, governance failure, and a board that did not stop it — which is precisely why the penalties stacked up across so many years and why the response escalated all the way to a growth cap. A scandal that offends only one regulator is a fine; one that offends all three is an existential event.

With that vocabulary in hand, we can tell the story properly — and see at each step how the bank's own design turned an ordinary good idea (sell customers more products) into the largest consumer-banking conduct failure in American history.

## The quota culture: "Eight is Great" and the daily number

Every conduct scandal has a slogan, and Wells Fargo's was "Eight is Great." The goal, set at the top of the Community Bank division, was eight products per household — eight separate accounts, cards, or services for the average customer relationship. Executives liked the number partly because it rhymed with "great" and partly because the chairman reportedly joked that "eight rhymes with great." That a target this consequential was anchored to a rhyme tells you something about how seriously the *quality* of the goal was examined.

Why eight? Not because customer research showed households needed eight products. The honest answer is that cross-sell ratio was the metric Wall Street had learned to reward. Wells Fargo's roughly six products per household was already the best in the industry, and management presented it, quarter after quarter, as proof of a superior franchise. Analysts cheered. The stock carried a premium valuation *because* of the cross-sell story. So the pressure to keep the number climbing was not a side effect; it was load-bearing for how the market valued the company. The board wanted eight because the share price wanted eight.

A target at the top is harmless. The damage happens in the translation downward, and Wells Fargo translated with brutal precision. The annual goal became a quarterly goal, the quarterly goal a monthly goal, the monthly goal a *daily* goal, and in many branches the daily goal was tracked *by the hour*. Bankers were ranked against each other on leaderboards. Managers held morning huddles and afternoon check-ins. A branch that fell behind got a call from the regional manager; a branch that fell behind repeatedly got a visit. The message, delivered relentlessly, was that the number was not negotiable.

Notice the structural feature here, because it is what made the system so dangerous: the pressure compounded as it descended. The board wanted an annual cross-sell ratio; that is an abstract, year-long, statistical goal. By the time it reached a teller, it had been compressed into a concrete, immediate, personal demand — *make two more sales before lunch or your manager will know.* Each layer of management, measured on its own slice, had every reason to pass the pressure down undiminished and no reason to absorb any of it. A long target horizon at the top became a short target horizon at the bottom, and short horizons are where ethics go to die: it is far easier to rationalize one small wrong thing to survive today than to weigh the cumulative harm of doing it every day for years. The org chart did not just transmit the quota; it *amplified* it.

#### Worked example: the per-employee quota arithmetic

> [!note]
> Let us make the pressure concrete. Suppose a banker is expected to sell about **15 products a day** (a figure consistent with the daily "solutions" targets reported across many branches). A typical branch is open roughly 8 hours, so that is **about 2 sales per working hour**, every hour, all day, every day.
>
> Now think about the honest funnel. To make 2 *real* sales an hour you might need to have a substantive conversation with 6–8 customers an hour — and most retail branches simply do not see that many customers who genuinely need a new product. On a slow Tuesday a branch might serve 30 customers all day, many just depositing a check. If even half already have what they need, the honest ceiling might be 5–6 real sales for the day, against a quota of 15.
>
> The gap — roughly **15 required minus 6 honest = 9 sales a day with no honest source** — is the fraud, expressed as a number. Multiply 9 missing sales × thousands of bankers × hundreds of working days and you arrive, mechanically, at millions of fake accounts. The intuition: the quota was not set against what customers needed; it was set against what Wall Street wanted, and the difference had to come from somewhere.

There is one more ingredient, and it is the cruelest. Missing the quota was not merely embarrassing; it was a path to losing your job. Employees described being "coached" for falling short, written up, put on performance plans, and eventually fired. So the choice the system presented to a banker who could not honestly hit the number was not "try harder versus lose a bonus." It was "commit fraud versus lose your livelihood." Faced with that, a large minority of people — not because they were unusually dishonest, but because the incentive was unusually vicious — chose to game. The remarkable thing is not that some employees cheated. It is that the system was *designed* to make cheating the rational survival strategy, and then punished the employees for being rational.

## The gaming: how an empty account got opened

So how do you "sell" a product to someone who does not want it? The methods that came out in regulatory findings and reporting are mundane, which is exactly what makes them chilling. There was no master forger. There were thousands of ordinary people inventing small workarounds under pressure.

The most common technique was simply to open an account a customer never authorized — usually with no money in it, or funded briefly by moving a few dollars from the customer's real account so the new one looked "active," then sometimes moved back. Bankers opened unauthorized **debit cards** and **online banking** enrollments the customer had not asked for. They issued **credit cards** without consent, which is the most damaging variant because a new credit line and a hard credit inquiry can lower a customer's credit score and trigger annual fees. The term that surfaced internally for one tactic was **"pinning"** — assigning a PIN to a customer's account, or to a new debit card, without telling them, so the banker could enroll the customer in online banking and claim the "solution." Some bankers used their own email addresses or fake phone numbers so the customer never received a notification.

The before-and-after below shows the difference between the legitimate version of the exact same product sale and the gamed one. The product is identical; only the *intent and the consent* differ — and that difference is the entire crime.

![Two-column comparison of a real cross-sell versus a coerced fake account](/imgs/blogs/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong-3.png)

What made the gaming so durable was that, on the bank's dashboards, a fake account looked almost exactly like a real one. The metric was "accounts opened." A gamed account opened. The metric ticked up. The branch hit its number. The regional manager reported success. The division reported a rising cross-sell ratio. The CEO reported it to analysts. The fraud did not *break* the reporting system; it *fed* it. Every level above the teller was being told the truth by its own numbers — that sales were strong — precisely because the numbers measured the wrong thing.

This is the most important technical point in the post, so let us state it plainly. The system did not fail because anyone lacked data. It failed because the data measured *activity* (accounts opened) rather than *outcome* (accounts the customer wanted and used). When you reward activity, you get activity, including activity no one valued. A metric you can hit by harming a customer is not a performance metric; it is a fraud incentive wearing a performance metric's clothes.

There is a name for this from the world of economics, and it is worth knowing because it predicts conduct scandals before they happen. **Goodhart's law** states that *when a measure becomes a target, it ceases to be a good measure.* The cross-sell ratio was, originally, an honest signal of a healthy franchise: customers with more products really were stickier and more profitable, so a rising ratio genuinely meant the bank was doing well. The moment Wells Fargo turned that signal into a hard target enforced by firing, the signal detached from reality. The ratio kept rising, but it no longer meant what it used to mean, because it was now being produced by gaming rather than by good service. The bank was steering by an instrument it had itself broken — and the instrument still read "full speed ahead" all the way into the iceberg.

This is why the fix, when it finally came, was not "watch the cross-sell ratio more carefully." You cannot fix a broken instrument by reading it harder. The fix was to *change what you measure* — to reward outcomes (customers who use their accounts, who stay, who do not complain) that are far harder to fake than a raw count of openings. The harder a metric is to game, the closer it stays to the thing you actually care about. A count of accounts opened is trivially gameable; a measure of accounts still actively used by satisfied customers twelve months later is not. The entire post-scandal redesign of the retail bank's incentives is, at heart, the replacement of an easily-gamed activity metric with hard-to-game outcome metrics.

It is worth pausing on *why* a count is so much easier to game than an outcome. A count is something an employee controls directly and instantly — opening an account is a single keystroke that the banker can perform unilaterally, today, with no cooperation from the customer. An outcome requires the customer to actually want, fund, and keep using the product over time, which no banker can fake into existence at the keyboard. The deeper principle for any incentive design, in banking or anywhere else: reward the things your people *cannot* produce without the result you actually want. Wells Fargo rewarded exactly the thing they *could* produce without it.

#### Worked example: when a fake account costs the customer real money

> [!note]
> Most fake checking accounts earned the bank nothing. But the credit cards were genuinely harmful, and the math shows why. Suppose a banker opens an unauthorized credit card for a customer. Three things happen automatically:
>
> 1. A **hard inquiry** hits the customer's credit file, typically knocking **5 points** off a credit score.
> 2. The new account *lowers the average age* of the customer's credit history, another small ding.
> 3. If the card carries a \$45 annual fee and goes unpaid because the customer never knew it existed, it can generate **late fees** and eventually a **delinquency** mark — which can cost **50 to 100 points**.
>
> A customer applying for a mortgage at, say, a 720 score versus a 670 score might face a rate difference of roughly **0.5%**. On a \$300,000 30-year mortgage, 0.5% is about **\$90 a month**, or roughly **\$32,000** over the life of the loan. So a fake credit card that earned Wells Fargo a few dollars of fees could, in the worst case, cost a single customer tens of thousands of dollars.
>
> The intuition: the bank's gain and the customer's potential loss were not just unequal — they pointed in opposite directions. That is the signature of a conduct failure, and it is why regulators treat consumer harm, not bank profit, as the thing to measure.

## Why every control failed: the three lines that did not defend

Here is the question that makes Wells Fargo a *teaching* case rather than just a scandal: the bank had a risk function, a compliance function, an internal audit team, a board with a risk committee, and a federal regulator inside the building. How did a fraud involving millions of accounts and thousands of employees run for *years* without being stopped? The answer is that the three lines of defense did not fail randomly. They failed in a specific, instructive sequence, each for a reason that recurs in conduct scandals everywhere.

![Pipeline showing the first, second, and third lines of defense each failing](/imgs/blogs/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong-4.png)

**The first line created the risk it was supposed to control.** Recall that the first line of defense is the business — here, the Community Bank's sales management. The whole theory of the first line is that the people closest to the activity own its risks. But at Wells Fargo, the first line was not a neutral risk owner; it was the *source* of the pressure. The regional and branch managers being asked to police gaming were the same people being measured and paid on the sales the gaming produced. Asking them to crack down on fake accounts was asking them to lower their own numbers and risk their own jobs. So when complaints came in, the first line did the rational thing for the first line: it fired individual employees for "ethics violations" — which removed witnesses and protected the quota — and kept the system running. Over several years the bank dismissed about **5,300 employees** for sales-practices violations, a number it later cited as evidence it was *policing* the problem. In fact it was evidence of the opposite: the firings treated the people as the disease, when the people were the symptom.

**The second line had no real power to challenge.** The second line — risk and compliance — exists to be the independent brake. Its only source of authority is independence and a mandate to say "no." At Wells Fargo, the second line was thin, under-resourced relative to the size of the Community Bank, and structurally deferential to a division that was the company's crown jewel and biggest profit center. There was no effective, empowered challenge to the central question — *is this incentive system itself the hazard?* When a second line cannot challenge the business model that funds the bank, it is not a second line of defense; it is decoration.

**The third line audited the wrong thing.** Internal audit, the third line, did look at sales practices. But it framed what it found as a *people-and-process* problem — employees behaving badly, controls to be tightened — rather than as an *incentive* problem. It is the difference between auditing whether the locks work and asking whether you have handed every employee a reason to pick them. Audit reported individual misconduct up the chain; it did not force the board to confront that the misconduct was the *predictable output* of the compensation design. So the board received a stream of "we found some bad apples and dealt with them" reports, never a single "our incentive system is manufacturing fraud" report.

Above all three sat a governance structure that saw the problem in fragments. Complaints went to one place, firings to another, ethics-line reports to a third, and each was small enough, on its own, to be managed and closed. No one was charged with assembling the fragments into the whole. The graph below shows the shape of that failure: signal entering the organization in pieces, each channel handling its slice, and the board committees seeing a scattering of small issues rather than one systemic fraud.

![Graph of fragmented governance signals reaching the board without a clear owner](/imgs/blogs/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong-8.png)

#### Worked example: the controls budget versus the fine

> [!note]
> Imagine the bank had instead invested seriously in the second line. Suppose it had hired **200 additional risk and compliance staff** dedicated to sales-practice oversight, at a fully loaded cost of **\$200,000** each per year. That is 200 × \$200,000 = **\$40 million a year**. Run that for the roughly 5 years the fraud was active and you spend **\$200 million** on prevention.
>
> Against that, the scandal cost about **\$4,900 million** in fines alone — plus the asset cap, the lost market value, and the management churn. So a preventive control budget of \$200 million stood against a realized loss more than **24 times larger**, even ignoring the non-fine costs.
>
> The intuition: a strong, empowered second line *looks* like pure cost on the income statement, because you only ever see what it spends, never what it prevented. Wells Fargo's mistake was treating the cheap insurance as overhead. Conduct risk is the one place where under-spending on control is almost always the most expensive choice.

## The discovery: from a local newspaper to a Senate hearing

Before the public discovery, there were years of private warnings — three of them, each pointing at the same root cause, each handled in a way that let the quota survive. The matrix below lays them out: customer complaints, the wave of firings, and the whistleblower reports. Read across the rows and the failure is the same every time: the signal was real, it reached someone, and it was framed as a small, local problem rather than a systemic one.

![Matrix of complaints firings and whistleblower warnings that were ignored](/imgs/blogs/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong-6.png)

The first signal was **customer complaints**. Customers found accounts they had not opened, fees they did not recognize, and credit cards they had never applied for. These complaints flowed into branches and call centers and were resolved — one at a time, as isolated service issues. No system aggregated them into the question that mattered: *why are so many customers reporting accounts they never asked for?* A complaint resolved case by case is a closed ticket; the same complaints counted together are a pattern. The bank had the tickets and never did the counting.

The second signal was the **firings themselves**. Over several years the bank dismissed about 5,300 employees for sales-practices violations. That number was later offered as proof the bank was policing the problem, but turn it around: a company firing thousands of people for the *same* behavior, year after year, is staring at evidence that the behavior is structural, not individual. High, persistent, single-cause turnover in one division is one of the loudest red flags in all of conduct surveillance, and it was read as housekeeping.

The third signal was the **whistleblowers**. Employees called the ethics hotline, emailed managers, and in some cases warned explicitly that they were being pushed to commit fraud. Some of those who raised concerns were themselves disciplined or fired — the most corrosive possible outcome, because it teaches everyone watching that reporting the fraud is more dangerous than committing it. A whistleblower channel that punishes the whistleblower is worse than no channel at all, because it converts the bravest employees into silent ones.

For all the internal data pointing at the problem, the scandal broke into public view through ordinary journalism. In December 2013, the *Los Angeles Times* published an investigation describing the relentless sales pressure inside Wells Fargo branches and the gaming it produced. That reporting helped prompt the Los Angeles City Attorney to sue the bank in 2015. The federal regulators — the CFPB and the Office of the Comptroller of the Currency (the OCC, the national-bank supervisor) — joined, and in September 2016 the three together announced the \$185 million settlement: \$100 million to the CFPB (its largest penalty to that point), \$35 million to the OCC, and \$50 million to the City and County of Los Angeles.

The fine was small. The disclosure was not. The settlement documents stated that employees had opened roughly **1.5 million** unauthorized deposit accounts and applied for about **565,000** credit cards without authorization, and that the bank had fired some **5,300 employees** in connection with the practices. The combination — millions of fake accounts, thousands of fired staff, the most admired bank in America — turned a regulatory footnote into a national story overnight.

What happened next is a case study in how *not* to respond to a conduct crisis, and it added a second layer of damage on top of the first. The initial corporate framing blamed individual employees — the "bad apples" defense — and emphasized the 5,300 firings as evidence of accountability. At a Senate Banking Committee hearing later that month, that framing collapsed under questioning. If 5,300 people in different states independently invented the same fraud, senators asked, was that really 5,300 bad apples, or was it the barrel? The chief executive's insistence that this was not a systemic or incentive problem, in the face of its obvious scale, is what turned a fixable story into a referendum on the bank's leadership. He retired in October 2016; the head of the Community Bank had already departed.

The board then did something banks rarely do: it commissioned an independent investigation by an outside law firm and published the report in 2017. The report was unsparing about the incentive system and the centralized, defensive culture of the Community Bank, and it justified one of the largest executive pay clawbacks in corporate history — the bank reclaimed tens of millions of dollars of compensation from the former CEO and the former head of the Community Bank. The clawback mattered as a precedent: it established that the people who *designed and defended* the incentive, not just the tellers who acted on it, would bear financial consequences.

## The scale: from 1.5 million to 3.5 million accounts

The 2016 numbers were not the end of the counting. In 2017, an expanded, third-party review covering a longer period (mid-2009 through mid-2016) raised the estimate of *potentially* unauthorized accounts to about **3.5 million** — roughly 3.5 million deposit and credit-card accounts that the bank could not confirm the customer had wanted. The count rose because the review widened: more years, more product types, and a lower bar for "potentially unauthorized." The figure below shows that escalation.

![Bar chart of fake-account estimate rising from 2.1 million to 3.5 million](/imgs/blogs/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong-7.png)

It is worth being precise about what "3.5 million" does and does not mean, because the number is often quoted loosely. It is an estimate of accounts that *may* have been opened without proper authorization, identified by a backward-looking statistical and account-level review, not a list of 3.5 million confirmed individual victims who each lost money. Many of the accounts were empty and harmless in dollar terms. The harm was concentrated in the credit cards (the credit-score damage) and the fees that some accounts generated before discovery. But even read conservatively, the number is staggering: it implies that over the review period, the gaming was not an occasional aberration in a few rogue branches but a steady, system-wide background hum, the predictable output of the quota machine running day after day.

#### Worked example: turning 3.5 million accounts into a daily rate

> [!note]
> Let us sanity-check that 3.5 million is even possible without a conspiracy of supervillains. The review period ran roughly **7 years**, or about **7 × 250 = 1,750 working days**. Wells Fargo's Community Bank had on the order of **6,000 branches** and well over **100,000 customer-facing employees**.
>
> Spread 3.5 million accounts across 1,750 working days and you get **about 2,000 fake accounts a day**, system-wide. Across 6,000 branches, that is **one-third of one fake account per branch per day** — roughly one every three days. Across 100,000 bankers, it is **one fake account per banker every 50 days**.
>
> The intuition: no one had to be a prolific fraudster. A scandal of historic scale was assembled out of each employee gaming the system *occasionally*, when the daily number was just out of reach. That is what makes incentive-driven conduct risk so dangerous — it does not require bad people, only a bad number multiplied by a large enough headcount.

## The fines: \$185M, then \$3bn, then \$1.7bn

The 2016 settlement was the first installment, not the total. The financial reckoning arrived in waves over six years as different harms were quantified and different regulators weighed in. The chart below stacks the three headline settlements and their sum.

![Bar chart of Wells Fargo fines summing to about 4.9 billion dollars](/imgs/blogs/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong-2.png)

**2016 — \$185 million (CFPB, OCC, Los Angeles).** The opening settlement for the unauthorized accounts themselves. Large by the standard of a single consumer-practices case, small relative to what followed.

**2020 — \$3 billion (Department of Justice and SEC).** This was the criminal and securities reckoning. The Department of Justice and the Securities and Exchange Commission resolved investigations into the bank's conduct *and its disclosures* — the question of whether Wells Fargo had misled investors about the health of its vaunted cross-sell metric, which it had been touting to the market even as it knew the metric was inflated by fraud. The \$3 billion resolution covered both the consumer harm and the investor-disclosure dimension.

**2022 — \$1.7 billion (CFPB).** The largest single penalty, and notably *not* about the fake accounts at all. By 2022, the CFPB's findings had widened to a broader pattern of consumer mismanagement across the bank — improper auto-loan repossessions, wrongful home foreclosures, and incorrectly applied fees and charges affecting millions of accounts — with \$1.7 billion in penalties plus more than \$2 billion ordered in customer redress. This installment is the most damning in one specific way: it came years *after* the fake-accounts scandal supposedly triggered a wholesale cleanup, which tells you how deep the cultural problems ran and how long they took to fix.

#### Worked example: adding up the headline fines

> [!note]
> The arithmetic the post keeps returning to, done explicitly. Convert everything to billions:
>
> - 2016: \$185 million = **\$0.185 billion**
> - 2020: **\$3.0 billion**
> - 2022: **\$1.7 billion**
>
> Sum: \$0.185 + \$3.0 + \$1.7 = **\$4.885 billion ≈ \$4.9 billion** in headline fines.
>
> And that is *only* the fines. It excludes the more than \$2 billion in customer redress ordered in 2022, the tens of millions clawed back from executives, the legal and remediation costs, and — by far the largest number of all — the opportunity cost of the asset cap, which we turn to next. The intuition: when people quote "Wells Fargo paid about \$4.9 billion," they are quoting the *visible* bill. The true economic cost was a multiple of it.

Put Wells Fargo's \$4.9 billion next to the other great conduct disasters of the era and its place becomes clear. It sits in the same weight class as the LIBOR rate-rigging fines (around \$9 billion across the industry) and Goldman's 1MDB resolution (about \$5 billion), and well above the losses that *destroyed* banks like Barings (about \$1.3 billion) and Wirecard (€1.9 billion of cash that did not exist). The difference is that Wells Fargo *survived* its number — which is part of why it is the cleanest teaching case of all of them.

![Horizontal bar chart comparing Wells Fargo fines to other conduct cases](/imgs/blogs/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong-5.png)

## The asset cap: the punishment that cost more than the crime

In February 2018, on its way out the door, the Federal Reserve under Chair Janet Yellen did something it had never done to a large US bank. It issued a consent order capping Wells Fargo's **total assets** at their end-2017 level — roughly **\$1.95 trillion** — and forbade the bank from growing past that line until it satisfied the Fed that it had fixed its governance and risk controls. The cap was not lifted until 2025, after more than six years.

To see why this was the most expensive part of the entire affair, recall the spine of this whole series: **a bank is a leveraged, confidence-funded machine that earns the spread between what it pays for deposits and what it earns on assets.** A bank grows its profits primarily by growing its balance sheet — taking in more deposits, making more loans, holding more securities, each earning that spread. Cap the assets and you cap the engine. Wells Fargo could still operate, but it could not *grow*, and in a period when its peers were expanding aggressively, standing still meant falling behind every year the cap held.

The figure below makes the opportunity cost visceral. While Wells Fargo was frozen near \$1.95 trillion, JPMorgan, Bank of America, and Citigroup kept compounding. The bank that had been neck-and-neck with the largest US institutions watched the field pull away.

![Bar chart of US bank assets showing Wells Fargo frozen below peers](/imgs/blogs/the-wells-fargo-fake-accounts-scandal-when-incentives-go-wrong-9.png)

#### Worked example: the opportunity cost of the cap

> [!note]
> Let us put a dollar figure on the cap. Suppose, absent the cap, Wells Fargo could have grown its balance sheet at a modest **5% a year**, in line with the industry, from its \$1.95 trillion base. That is roughly **\$98 billion of new assets a year** it was forbidden to add.
>
> A commercial bank earns a return on assets (ROA) of roughly **1%** in good times. So \$98 billion of forgone growth in year one is about **\$1 billion** of forgone annual profit — and the forgone base compounds, so by year three the bank is missing the profit on roughly \$300 billion of assets it never got to add, on the order of **\$3 billion a year** in lost earnings.
>
> Run that for six years and the cumulative forgone profit plausibly reaches **\$10–20 billion** — *several times the \$4.9 billion in fines*. The intuition: the headline penalty was the fines, but the real punishment was the growth the bank was not allowed to have. Regulators understood that a fine is a one-time cost a giant bank can absorb; a cap on growth is a recurring tax on the franchise itself, and it forced the board to treat fixing the culture as an existential priority rather than a line item.

The asset cap is the single most instructive regulatory move in the story because it changed the *price* of conduct risk. Before 2018, a bank could reasonably treat conduct fines as a cost of doing business — painful, but survivable and, crucially, *finite*. The asset cap made the cost open-ended and tied directly to the thing management cared about most: growth and the share price. Few regulatory tools are aimed more precisely. The Fed answered a broken incentive with a corrective one.

## The cleanup: rebuilding a franchise on trust

What does it take to escape a cap like that? The list is long and tells you what a serious conduct remediation actually involves. Wells Fargo replaced its CEO twice more after 2016, brought in outside leadership, and reconstituted much of its board — several directors did not stand for re-election after a 2017 shareholder revolt in which a striking share of votes were withheld from incumbent directors. It dismantled the product-sales-goals system in the retail bank entirely, replacing it with metrics based on customer usage, relationship quality, and complaint resolution rather than raw account openings. It rebuilt the second line, materially increasing risk and compliance headcount and elevating their authority. It spent years and billions of dollars on remediation programs, customer redress, and the consent-order requirements imposed by the OCC, CFPB, and Fed.

The deepest change was the hardest to measure and the slowest to land: the metric. As long as a bank rewards *accounts opened*, it will get accounts opened, by any means. The fix was to reward what the bank actually wanted in the first place — customers who are well served and stay. That sounds obvious. It is obvious. The whole scandal happened because, for years, a metric that was easy to measure (openings) crowded out the outcome that was hard to measure (genuine relationships), and no one with the power to fix it was incentivized to.

It is worth being concrete about what "rebuilding the second line" actually means, because it is the part outsiders most often wave at vaguely. An empowered second line is not just more compliance headcount; it is a set of structural changes that give risk and compliance the *power to win an argument* with the business. In practice that meant: risk officers who report to the chief risk officer and ultimately the board, not to the sales executive whose numbers they question; a formal mandate that risk can *veto* an incentive scheme before it launches, not merely comment on it afterward; whistleblower channels that are independent, anonymous, and audited so that a report cannot be quietly routed back to the manager being reported; and compensation for senior risk staff that does *not* rise and fall with the division's sales. Each of these attacks the same flaw — the original second line could be overruled by the very business it policed — and each is expensive, slow, and invisible on a good day. That invisibility is exactly why it was neglected for so long, and exactly why a regulator had to force it.

There is a sobering coda. The 2022 \$1.7 billion penalty was *not* about fake accounts — it covered improper auto-loan repossessions, wrongful foreclosures, and mis-applied fees — which means the underlying cultural and control problems were still producing fresh consumer harm *six years* after the scandal broke and the cleanup began. That is the single most humbling fact in the whole episode. It tells you that culture is not a memo or a slogan or a single firing; it is the accumulated set of incentives, habits, and structures that take years to install and years to remove. A bank can change its CEO in a day and its incentive *plan* in a quarter, but the *behavior* the old incentives trained takes far longer to unwind. Conduct risk is sticky in a way that credit and market risk are not.

The cap's removal in 2025 was the formal end of the episode, but the lesson outlasts it. A bank can recover from a credit loss in a quarter and a trading loss in a year. Recovering from a *trust* loss took the better part of a decade, cost a low-double-digit multiple of the visible fines, and required tearing out and rebuilding the management, the board, the controls, and above all the incentive system that caused it. That is the true price of conduct risk: not the fine, but the years.

## Common misconceptions

**"It was just a few thousand bad employees."** This was the original corporate defense and it is wrong on the arithmetic. The bank fired about 5,300 people, but a fraud distributed across thousands of employees, hundreds of branches, and several years is not a coincidence of bad character — it is a system producing its designed output. The worked example above showed it took only *one fake account per banker every 50 days* to reach 3.5 million. The barrel, not the apples, was the story; the independent board report said as much.

**"Customers lost billions, so the bank made billions."** No — and this is the most counterintuitive point. The bank made *almost nothing* on the fake accounts themselves; most were empty. The harm to customers came through credit-score damage and stray fees, and the cost to the *bank* came through fines, remediation, and the asset cap. Conduct risk is the rare case where everyone loses: the customer is harmed, and the bank loses far more than the customer, with the gamed "profit" being a rounding error in between. There was no pot of fraud profits; there was only destruction on both sides.

**"The regulators caught it."** Largely not. The decisive early disclosure came from a newspaper investigation (the *Los Angeles Times*, 2013) and a city attorney's lawsuit (2015), with the federal regulators formalizing and escalating afterward. Internal data had pointed at the problem for years. The lesson is humbling: a problem can be visible in a bank's own complaint logs and turnover statistics and *still* not be stopped, because the people positioned to see it were incentivized not to.

**"The \$185 million fine was the punishment."** The 2016 fine was about 4% of the eventual financial reckoning, and the financial reckoning was itself dwarfed by the asset cap's opportunity cost. Quoting the first fine as "the punishment" understates the real cost by more than an order of magnitude. The true penalty was \$4.9 billion in fines *plus* something like \$10–20 billion in forgone growth *plus* a decade of rebuilding.

**"Cross-selling is inherently predatory."** Cross-selling is normal, valuable banking — a customer with several products they actually use is well served and profitable, which is a genuine win-win. What was predatory was the *quota*: a target divorced from customer need and enforced by fear of firing. The same product sold with consent is good banking; sold under coercion it is fraud. The villain was never the cross-sell; it was the metric.

## How it shows up in real banks: incentive-driven conduct failures

Wells Fargo is the cleanest example, but incentive-driven conduct risk is a recurring genre. A handful of other episodes show the same machinery with different products.

**Payment protection insurance (PPI) in the UK.** Through the 2000s, British banks sold payment-protection insurance — coverage meant to repay a loan if the borrower lost income — bundled onto loans, credit cards, and mortgages, often to people who could never claim on it (the self-employed, the already-insured) and frequently without clearly disclosing it was optional or even that it had been added. The driver was the same: high-commission sales targets. The eventual redress bill across the UK banking industry exceeded **£38 billion** (well over \$50 billion), the largest consumer-redress event in British financial history — an order of magnitude larger than Wells Fargo. Different product, identical root cause: an incentive that rewarded the sale over the suitability.

**The 2008 mortgage origination machine.** In the run-up to the financial crisis, mortgage brokers and loan officers were paid on *volume of loans originated*, not on loans that *performed*. Originate the loan, collect the fee, sell the loan onward, repeat — the originator bore none of the downstream default risk. The result was millions of loans made to borrowers who could not repay, because the incentive rewarded origination and was blind to outcome. It is the Wells Fargo metric problem (reward activity, ignore outcome) operating at the scale of the entire housing market. The mechanism connects directly to the system-level view in [how money is created: banks, central banks, and the money multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).

**Investment-bank "selling against the franchise."** The sell-side has its own version: bankers paid on deal fees who push clients into transactions that are good for the bank and bad for the client, or research analysts whose ratings drift toward the firms their bank wants as investment-banking clients. The structural conflict — the person advising you is paid by the volume of what they sell you — is exactly the principal-agent problem from our Foundations section, and it is why the post-2000 reforms forced a wall between research and banking. The economics of those conflicts are laid out in [inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).

**Cross-sell scandals beyond banking.** The pattern is not unique to finance. Wherever frontline staff are measured on a single easily-gamed activity metric and punished for missing it — sales of warranties, of phone plans, of subscriptions — the same fraud-by-quota dynamic appears. Banking just happens to have the regulators, the disclosure requirements, and the public balance sheets that turn the failure into a measurable \$4.9 billion case study instead of a quiet internal write-off.

Across all of these, the common thread is the one this post has hammered: an incentive that rewards an *activity* the firm can measure rather than an *outcome* the firm actually wants, enforced hard enough that gaming becomes the rational choice. That is the definition of a conduct hazard, and it sits at the intersection of the people-process-systems map in [the four risks every bank runs: credit, market, liquidity, operational](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational) and the funding-franchise logic in [retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — because the trust Wells Fargo broke was the very thing that made its deposit franchise cheap.

## The takeaway: read the incentive, not the dashboard

If you take one idea from this case, let it be this: **the most dangerous risk on a bank's books is often the one its own management put there, and it will not show up on any risk dashboard, because the dashboard is measuring the activity the incentive was designed to produce.** Wells Fargo's sales metrics were *strong* the entire time the fraud ran. The numbers were not lying about activity; they were lying about value, and no one had built a control that knew the difference.

So when you look at a bank — as an investor, a regulator, an employee, or a customer — the conduct-risk question is not "are the controls in place?" It is the deeper, more uncomfortable question: **what is this bank's frontline paid to do, and what would a rational, frightened employee do to hit that number?** If the honest answer involves harming a customer, you have found the hazard, no matter how green the dashboard glows. Look at the incentive, trace it down the org chart, and ask where the gap between the quota and the honest funnel has to go. At Wells Fargo, that gap had to go into 3.5 million fake accounts, and it cost about \$4.9 billion in fines, an asset cap that froze the bank for over six years, and the better part of a decade to rebuild the one asset a bank cannot survive without: the trust of the people whose money it holds.

The spine of this series says a bank is a leveraged, confidence-funded machine that lives only as long as depositors trust it. Wells Fargo is the proof in the negative. It did not run out of capital or liquidity. It ran out of trust — and it ran out of trust because it paid its own people, one quota at a time, to spend it.

## Further reading & cross-links

- [Operational risk: fraud, cyber and the loss events](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events) — the broader category that conduct risk sits inside, with the people-process-systems loss taxonomy.
- [The four risks every bank runs: credit, market, liquidity, operational](/blog/trading/banking/the-four-risks-every-bank-runs-credit-market-liquidity-operational) — where conduct risk fits among the risks a bank must manage, and why it is the self-inflicted one.
- [Retail deposits: the funding base and why cheap money is the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise) — why customer trust is not a soft virtue but the thing that makes a deposit franchise valuable.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the sell-side version of the principal-agent conflict and how fee incentives shape behavior.
- [How money is created: banks, central banks, and the money multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) — the system-level mechanics behind volume-driven origination incentives.

*This article is educational, not financial or legal advice. Figures for fines, account counts, and bank sizes are drawn from regulatory consent orders, company filings, and contemporaneous reporting as cited; conduct-case details evolve as litigation and remediation continue.*
