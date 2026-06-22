---
title: "Spotting the Next Bank Failure: The Early-Warning Signs"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The red-flag dashboard for reading a bank in trouble before it fails — growth, funding concentration, hidden bond losses, deposit flight, and the CAMELS tells, run over SVB, Credit Suisse, and WaMu."
tags: ["banking", "bank-failure", "early-warning-signs", "bank-run", "uninsured-deposits", "camels", "risk", "credit-cycle", "liquidity", "financial-stability"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank almost never fails out of a clear sky. It fails after a quiet build-up of red flags that, read together, spell out a single story: it grew too fast, funded itself with money that could leave overnight, took a hidden loss it could not absorb, and then lost the one thing it cannot survive without — confidence.
>
> - **No single number kills a bank.** Each red flag — rapid growth, concentrated flighty funding, large unrealized bond losses, a shrinking margin, rising bad loans, widening credit spreads, deposit outflows, a collapsing share price — is survivable alone. They become fatal when they *line up*. The skill is reading the column, not the cell.
> - **The chain is always the same shape:** fast growth → funding concentration → a rate or credit shock that opens a hidden loss → cracking trust → a run → forced sale or seizure. Spotting the failure early means catching it *upstream*, before the run.
> - **The single most predictive number is the uninsured-deposit share.** Silicon Valley Bank had **94%** of its deposits above the \$250,000 insurance limit — money with every incentive to flee at the first rumor. A diversified retail bank sits well under half.
> - The one fact to remember: failures are *rare but clustered*. In a normal year the US sees near-zero bank failures; in 2010 it saw **157**. The flags don't tell you a bank will fail tomorrow — they tell you it has lost its margin for error when the cycle turns.

In the first week of March 2023, Silicon Valley Bank looked, by most casual measures, fine. It was the 16th-largest bank in the United States, profitable, well above its regulatory capital minimums, audited clean, with a stock that had quadrupled over the prior decade. Ten days later it was gone — seized by regulators after depositors tried to pull \$42 billion in a single day, with another \$100 billion queued to leave the next morning. The fastest bank run in history collapsed a \$209 billion institution in roughly 36 hours.

Here is the uncomfortable part: every ingredient of that collapse was visible in public filings *months* in advance. The bank had doubled in size in two years. Almost all of its deposits were uninsured. It had parked the flood of incoming money into long-dated bonds at the bottom of the interest-rate cycle, and when rates rose those bonds developed a paper loss larger than the bank's entire equity cushion. None of that was secret. It was sitting in the 10-K. The signs were there — they were simply not *assembled* into a single dashboard by enough people in time.

That is what this post is about. We are going to build the dashboard you would use to read a bank for trouble *before* the headline — the same set of metrics a bank examiner watches, translated into things you can pull from a public filing or a market screen. This is the synthesis post for the whole "great failures" arc of this series: we will pull the bank run, the duration trap, the credit cycle, and the conduct blow-ups together into one usable checklist, and then run that checklist over three real banks that all flashed red — Silicon Valley Bank, Credit Suisse, and Washington Mutual — to see how the warning system would have performed.

![Matrix of bank red flags with healthy, danger, and the bank that showed each](/imgs/blogs/spotting-the-next-bank-failure-the-early-warning-signs-1.png)

The diagram above is the mental model for the entire post, and it is worth sitting with for a moment. Each row is one red flag. The green column is what the metric looks like on a healthy bank; the red column is the danger zone; the amber column names a real bank that actually showed that flag before it failed. The whole skill of reading a bank is learning to scan down that red column and notice when the reds start to stack up in one institution. A bank with one red flag is normal. A bank with five is a problem. A bank with eight is a press release waiting to happen.

This is educational, not investment advice — we are learning to read mechanisms and history, not to time a trade or tell you to short anything.

## Foundations: how a bank actually dies

Before we can spot a failing bank, we need to be clear about what failing *means* for a bank, because it is not the same as for a normal company. A factory fails when it runs out of customers. A bank fails for one of two reasons, and the early-warning signs all map onto one of them.

A bank, at its core, is a **maturity-transformation machine.** It borrows short and lends long. It takes your deposit — money you can demand back today, at any moment, in full — and turns it into a 30-year mortgage or a 5-year business loan that it cannot get back early. It earns the gap between the low rate it pays you and the higher rate it charges the borrower. That gap is the *net interest margin*, and it is the bank's reason to exist. (A *net interest margin*, or NIM, is simply net interest income divided by the bank's interest-earning assets, expressed as a percent — roughly, the spread the bank keeps after funding costs.)

The trouble is that this trade is structurally fragile, and it can break in exactly two ways.

**The first way is insolvency.** A bank is *insolvent* when its assets are worth less than what it owes — when the loans and bonds it holds have lost so much value that they no longer cover the deposits and debts it must repay. The buffer between "assets" and "what we owe" is **equity** (also called capital): the owners' stake, the part of the bank funded by shareholders rather than borrowed. Equity is the shock absorber. When a loan goes bad or a bond loses value, the loss eats into equity first. If losses exceed equity, the bank is insolvent — there is no longer enough to make everyone whole. The everyday image: a corner shop that lent out the cash in its till, then discovered half the borrowers won't repay. If the shortfall is bigger than the owner's own savings in the business, the shop is broke.

**The second way is illiquidity.** A bank can be perfectly solvent — its assets genuinely worth more than it owes — and *still* fail, because it cannot turn those assets into cash fast enough to pay depositors who all show up at once. This is the **bank run**, the oldest failure in finance. The image here is a coat-check that has more coats than tickets, but cannot physically hand back every coat in the same five minutes. Even if every coat is present, a stampede at the counter breaks the system. A bank's loans are long and illiquid; its deposits are short and on-demand. If enough depositors demand their money simultaneously, the bank must sell good assets at fire-sale prices to raise cash — and those fire-sale losses can turn an illiquid bank into an insolvent one. Illiquidity and insolvency are not separate diseases so much as two doors into the same grave.

Layered on top of both is the thing that makes banking unique: **confidence.** A bank funds long-term lending with money that depositors *believe* they can get back any time. That belief is the whole franchise. The moment depositors stop believing — because of a rumor, a loss, a falling share price, a tweet — they rush for the exit, and the rush itself causes the failure they feared. Banking is the rare business where the fear of failure is sufficient to cause it. We covered this dynamic in depth in [the anatomy of a bank run](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse); here it is enough to hold the core idea: **a bank lives on borrowed trust, and trust is the first thing the warning signs are measuring.**

### The metrics, defined from zero

Almost every red flag in this post is a way of measuring one of three things: *how exposed* the bank is (how thin its cushion, how flighty its funding), *how stressed* it is right now (losses building, margin shrinking), or *what the market thinks* (spreads, share price). Let's define the recurring terms once, plainly, so the rest reads smoothly.

- **Capital ratio.** Equity as a percentage of (risk-weighted) assets. The standard regulatory measure is the *CET1 ratio* — the highest-quality equity over risk-weighted assets. Higher is safer; near the regulatory minimum is a warning. We built this up properly in [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) and [risk-weighted assets](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work).
- **Uninsured-deposit share.** The fraction of deposits above the government insurance limit (\$250,000 per depositor in the US). Insured money is *sticky* — it has no reason to run, because it is guaranteed. Uninsured money is *flighty* — it runs at the first sign of trouble. A high uninsured share is a high run risk.
- **HTM / AFS unrealized losses.** Banks hold bonds in two buckets. *Available-for-sale* (AFS) bonds are marked to market — their losses show up. *Held-to-maturity* (HTM) bonds are carried at cost, so their losses are *hidden* from the headline capital ratio — but they are real, and they appear as an "unrealized loss" footnote. A large HTM unrealized loss is a loss the bank is pretending it doesn't have.
- **Non-performing loans (NPLs) and coverage.** NPLs are loans the borrower has stopped paying (typically 90+ days past due). *Coverage* is the bank's loan-loss reserve divided by its NPLs — the cushion it has set aside against those bad loans. Rising NPLs are bad; rising NPLs with *falling* coverage are worse, because it means the bank is under-reserving against a growing problem.
- **AT1 / CDS spreads.** *AT1* (Additional Tier 1) bonds are a bank's riskiest debt — they get wiped out before depositors if the bank fails. A *CDS*, or credit default swap, is insurance against the bank defaulting; its price (the spread) is the market's real-time judgment of failure risk. When AT1 yields and CDS spreads blow out, the market is pricing failure.

With those defined, we can walk the dashboard one red-flag family at a time. The order is deliberate: it follows the chain of how a bank actually gets into trouble.

![Graph showing how fast growth funding concentration and a rate shock chain into a deposit run and failure](/imgs/blogs/spotting-the-next-bank-failure-the-early-warning-signs-4.png)

The figure above is the spine of the whole post: the red flags are not an unordered checklist, they are a *causal chain*. Fast growth feeds funding concentration. Concentration plus a rate shock opens a hidden loss. The loss cracks trust. Cracked trust becomes a run. The run forces a sale or a seizure. Spotting the failure early means recognizing the chain *upstream* — at growth and concentration — long before the run that makes the evening news. Each section below is one link in that chain.

## Red flag #1: rapid growth — outgrowing your peers

The first warning sign is the most counterintuitive, because it looks like success: a bank that is growing much faster than everyone else.

Why is fast growth dangerous? Because a bank's growth is the growth of its *loan book* and its *deposit base*, and both have natural speed limits. Good loans are scarce. If a bank doubles its lending in two years while its peers grow 10%, it is almost certainly doing one of three things: lending to borrowers everyone else passed on (worse credit), cutting prices to win business (thinner margin), or riding a single hot sector that will eventually cool (concentration). All three plant the seeds of the next loss. Meanwhile, deposits that arrive in a flood tend to be the flightiest kind — hot money chasing a high rate or a trendy franchise, not loyal households who keep their paychecks there for decades.

The historical pattern is brutally consistent. The banks that failed in 2008–2010 were disproportionately the ones that had grown fastest in 2004–2006, riding the mortgage boom. Washington Mutual grew its assets aggressively through the housing bubble by piling into option-ARM mortgages — the riskiest, most exotic home loans on offer. SVB roughly doubled in size from 2019 to 2021 as venture-capital money flooded into tech startups, who parked their fundraising at SVB. Fast growth is not failure. But fast growth is the bank acquiring the *exposure* that a later shock will detonate.

#### Worked example: loan growth vs peers

Suppose the banking industry is growing its loan book at about **5% a year** — roughly nominal GDP. You are looking at two banks.

- **Bank A** grew its loans from \$50 billion to \$55 billion over a year: 10% growth. Twice the industry, but not extreme — a fast-growing regional. Worth a second look, not an alarm.
- **Bank B** grew its loans from \$50 billion to \$70 billion: **40% growth**, eight times the industry rate. There is no honest way to originate \$20 billion of *good* new loans in twelve months when your starting book is \$50 billion. Either the underwriting standards collapsed, or the bank bought a portfolio, or it chased one booming sector. Whatever the answer, Bank B now carries a year's worth of loans made at the *top* of a cycle, exactly the vintage that performs worst when the cycle turns.

The number to compute is simple: **(this year's loans − last year's loans) / last year's loans**, compared to the industry's ~5%. For Bank B that is (70 − 50)/50 = **40%**. The intuition: a bank growing many times faster than its peers is not winning — it is borrowing future losses and booking them as today's growth.

## Red flag #2: funding concentration and the uninsured-deposit share

If growth is the seed, funding is the fuse. This is the single most predictive red flag in the whole dashboard, and the one the SVB collapse burned into every regulator's brain.

A bank's deposits are not all the same. The crucial split is **insured vs uninsured.** In the US, the FDIC guarantees deposits up to **\$250,000 per depositor per bank.** Below that line, your money is safe no matter what happens to the bank — so you have no reason to run. Above that line, you are an unsecured creditor of the bank, and if it fails you may lose everything over \$250,000. So uninsured depositors have every incentive to be the first out the door at the first whiff of trouble. Insured money is *sticky*; uninsured money is *flighty*.

Now stack on a second dimension: **who** the depositors are. A bank funded by millions of small household checking accounts has a granular, diversified, sticky base — no single depositor matters, and most balances sit below the insurance line. A bank funded by a few thousand venture-backed startups, all in the same industry, all talking to each other on the same Slack channels and group chats, all with balances of tens of millions — that bank has a *concentrated, correlated, uninsured* base. When one of them gets nervous, they all get nervous, at the same moment, and there is no insurance line stopping them.

![Stacked bars comparing uninsured deposit share of SVB at 94 percent versus a safe bank at 40 percent](/imgs/blogs/spotting-the-next-bank-failure-the-early-warning-signs-2.png)

That chart is the whole SVB story in one chart. At the start of March 2023, roughly **94% of SVB's deposits were uninsured** — above the \$250,000 line, with every incentive to flee. A diversified retail bank typically runs under half its deposits uninsured. SVB had built a deposit base that was almost entirely flight risk, and it had done so quickly, during its growth spurt. The red column and the previous red flag are the same flag seen from two angles.

#### Worked example: the uninsured-deposit share and run risk

Let's put numbers on it. A bank has \$175 billion of deposits (SVB's actual figure). 

- If **40%** are uninsured (a safe-bank profile), then \$70 billion is flighty and \$105 billion is sticky, insured money that will not run. Even a severe run drains the \$70 billion; the bank can survive on the \$105 billion that stays.
- If **94%** are uninsured (SVB's actual profile), then **\$164.5 billion** is flighty and only \$10.5 billion is sticky. A run can, in principle, take almost the entire deposit base. There is no anchor.

Now layer on the speed. On March 9, 2023, depositors actually tried to withdraw **\$42 billion in one day** — about a quarter of all deposits, in hours, through a banking app, with no need to queue at a branch. Another ~\$100 billion was lined up for the next morning. A bank that holds long-dated bonds cannot raise \$142 billion in 24 hours without selling everything at a loss. The intuition: **uninsured-deposit share is the size of the gasoline puddle, and a digital banking app is the lit match.** A bank can survive a high uninsured share *or* a slow-moving depositor base, but the combination of 94% uninsured *and* a wired, herd-like clientele is the most flammable funding structure ever built.

This is why the funding mix matters more than the headline "we are deposit-funded." Two banks can both be 71% deposit-funded and have completely different fragility.

![Stacked bars showing a resilient bank with mostly insured deposits and a fragile bank with mostly uninsured deposits](/imgs/blogs/spotting-the-next-bank-failure-the-early-warning-signs-7.png)

The figure above makes the point precise. Both banks fund themselves the same way at the headline level — about 71% from deposits, the rest from wholesale borrowing, long-term debt, and equity, which is the typical large-bank mix. But the *resilient* bank's deposits are mostly insured and sticky (the green block dominates), while the *fragile* bank's deposits are mostly uninsured and flighty (the red block dominates). Same label, opposite run risk. When you read a bank's filing, do not stop at "deposit-funded" — drill into the insured/uninsured split and the depositor concentration, because that is where the run risk actually lives. For the full anatomy of how a funding base is built and why cheap, sticky deposits are the prize, see [the funding stack](/blog/trading/banking/the-funding-stack-deposits-wholesale-funding-bonds-and-covered-bonds) and [retail deposits as the franchise](/blog/trading/banking/retail-deposits-the-funding-base-and-why-cheap-money-is-the-franchise).

## Red flag #3: the rate shock and hidden bond losses

Now we reach the link in the chain that turns an exposed bank into a wounded one: a shock that opens a loss the bank can't easily absorb. For SVB, the shock was rising interest rates, and the loss was hiding in plain sight in the bond portfolio.

Here is the mechanism, built from zero. When a bank takes in more deposits than it can lend out — exactly what happens during a fast-growth deposit flood — it parks the excess in bonds, usually government and agency bonds, which are safe from *credit* risk (the US Treasury will not default). But bonds carry a different risk: **interest-rate risk.** A bond pays a fixed coupon. If you buy a 10-year bond yielding 1.5% and market rates then rise to 5%, your bond is now worth far less, because nobody will pay full price for a 1.5% bond when new ones pay 5%. The price falls until the yield matches. The longer the bond's *duration* — its sensitivity to rate moves — the bigger the price drop. (A bond with a duration of 5 years loses roughly 5% of its value for every 1 percentage point rise in rates.) We unpacked this in detail in [interest-rate risk in the banking book](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap).

The trap is in the accounting. Banks can label bonds **held-to-maturity (HTM)** if they intend to hold them to the end. HTM bonds are carried at *cost* on the balance sheet — their market losses do not hit the headline capital ratio. The loss is real, but it is hidden in a footnote as an "unrealized loss." The bank can keep the fiction going *as long as it never has to sell.* But a run forces selling. The moment a bank is forced to sell its HTM bonds to raise cash for fleeing depositors, the hidden loss becomes a real, realized loss that crashes through its equity. The duration trap and the run are the same event from two sides.

![Bar chart comparing SVB tangible equity of 16 billion against an unrealized securities loss of 17 billion](/imgs/blogs/spotting-the-next-bank-failure-the-early-warning-signs-3.png)

That is the single most important chart in the SVB post-mortem. By the end of 2022, SVB's available-for-sale and held-to-maturity portfolios carried roughly **\$17 billion in unrealized losses** — losses that existed but had not been recognized in the headline numbers. SVB's tangible common equity at the same point was about **\$16 billion.** Read those two bars together: the paper loss the bank was sitting on was *larger than its entire equity cushion.* On a fully marked basis, SVB was already economically insolvent. The HTM accounting let it report healthy capital ratios right up until the run forced the losses into the open.

#### Worked example: HTM loss versus equity

The test you run is one subtraction. Take the bank's tangible common equity, and subtract its total unrealized securities losses (AFS + HTM, found in the footnotes).

- **Tangible equity:** \$16 billion.
- **Unrealized securities loss:** \$17 billion.
- **Equity, fully marked:** 16 − 17 = **−\$1 billion.**

A negative number means the bank's *real* equity, if it had to mark everything to market, is already gone. Even a positive-but-small number is a warning: a bank with \$16 billion of equity and \$10 billion of hidden losses has only \$6 billion of true cushion, not \$16 billion. The intuition: **a held-to-maturity bond loss is a loss the bank is pretending it doesn't have, and a run is the event that ends the pretending.** When you read a bank in a rising-rate environment, always pull the AFS+HTM unrealized-loss footnote and subtract it from equity. If the result is uncomfortably small, the bank is one forced sale away from insolvency.

This is exactly why liquidity rules exist — to make sure a bank can meet outflows *without* dumping its bond portfolio at a loss. The Liquidity Coverage Ratio (LCR) requires a buffer of high-quality liquid assets sized to a 30-day stress; a low or falling LCR is its own red flag. We cover the mechanics in [liquidity management: LCR, NSFR and the buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer).

## Red flag #4: a deteriorating net interest margin

The next family of flags is about *earnings quality* — whether the bank's core profit engine is healthy or quietly seizing up. The headline gauge is the net interest margin.

Recall the NIM is the spread the bank earns: the rate it charges on loans and bonds, minus the rate it pays on deposits and borrowings, as a percentage of earning assets. A healthy bank's NIM is stable or gently rising. A *falling* NIM, quarter after quarter, is a sign the bank's engine is losing compression. There are several reasons a NIM falls, and each is its own small warning:

- **Funding costs are rising faster than asset yields.** When rates rise, the bank has to pay up to keep depositors — and depositors who can move money with a tap force the bank to pass through more of the rate hike. The fraction of a rate hike the bank passes to depositors is the *deposit beta.* A bank whose deposit beta is climbing fast is one whose cheap-funding advantage is eroding.
- **The bank is locked into low-yielding old assets.** A bank that loaded up on 1.5% long-dated bonds (see the previous flag) is stuck earning 1.5% on a big chunk of assets while paying 4–5% for new deposits. That is a negative spread on the margin, and it bleeds NIM directly.
- **Competition is compressing loan pricing.** A bank chasing growth (flag #1) often does it by cutting loan rates, which directly shrinks the spread.

A deteriorating NIM rarely kills a bank on its own — it is a slow bleed, not a heart attack. But it is an early tell that the franchise economics are weakening, and a weakening franchise has less cushion to absorb the *next* shock. We built up the spread business in full in [net interest margin and the spread business](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) and [the income statement of a bank](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions).

#### Worked example: a NIM squeeze in numbers

Take a bank with \$100 billion of earning assets.

- **Year 1:** it earns 4.0% on assets (\$4.0 billion of interest income) and pays 1.0% on funding (\$1.0 billion of interest expense). Net interest income = \$3.0 billion, so NIM = 3.0 / 100 = **3.0%.**
- **Year 2:** rates rise. Asset yield creeps up to 4.3% (it is slow, because half the assets are locked-in old bonds), but funding cost jumps to 2.5% (deposit beta is high; depositors demand more). Now interest income = \$4.3 billion, expense = \$2.5 billion, net = \$1.8 billion, and NIM = 1.8 / 100 = **1.8%.**

The NIM fell from 3.0% to 1.8% — a **40% collapse in the core profit margin** — purely because funding costs outran asset yields. On \$100 billion of assets, that is \$1.2 billion of annual profit evaporating. The intuition: **a falling NIM means the bank's borrow-short-lend-long trade is working against it instead of for it, and a bank that is not earning is a bank that is not rebuilding its cushion.**

## Red flag #5: rising bad loans with falling coverage

The previous flags are mostly *market and funding* risks. This one is the classic *credit* risk — the risk that borrowers stop paying — and it is the failure mode of the slow-burn collapses like Washington Mutual, as opposed to the lightning-fast liquidity collapses like SVB.

The metric is the **non-performing loan (NPL) ratio**: loans where the borrower has stopped paying (usually 90+ days past due) as a fraction of total loans. Rising NPLs mean the loan book is going bad. But the *single most revealing* version of this flag is rising NPLs combined with *falling coverage.*

**Coverage** is the bank's loan-loss reserve divided by its NPLs. The reserve is the pile of money the bank has set aside in advance to absorb expected loan losses — under modern accounting (IFRS 9 and CECL), banks must reserve for expected losses up front. If a bank has \$1 billion of bad loans and \$1 billion of reserves, its coverage is 100% — it has fully provided for the problem. If it has \$2 billion of bad loans and only \$1 billion of reserves, coverage has fallen to 50% — the problem is growing faster than the cushion. We cover the provisioning machinery in [collateral, security and loan-loss provisioning](/blog/trading/banking/collateral-security-and-loan-loss-provisioning-ifrs9-and-cecl) and the underlying loss math in [credit risk management: PD, LGD, EAD](/blog/trading/banking/credit-risk-management-pd-lgd-ead-and-expected-loss).

Falling coverage as NPLs rise is a red flag with two interpretations, both bad. Either the bank genuinely believes the new bad loans will recover (optimistic, and often wrong at the top of a cycle), or the bank is *deliberately under-reserving* to flatter its current earnings — because every dollar added to reserves is a dollar subtracted from this quarter's profit. A management team fighting to hit earnings targets has a powerful incentive to under-provision, which is exactly why falling coverage in a rising-NPL environment is a governance tell as much as a credit tell.

#### Worked example: coverage falling as NPLs rise

Watch a bank over three years.

- **Year 1:** NPLs = \$1.0 billion, reserves = \$1.2 billion. Coverage = 1.2 / 1.0 = **120%.** Healthy — more than fully reserved.
- **Year 2:** NPLs rise to \$2.0 billion, but reserves only rise to \$1.4 billion. Coverage = 1.4 / 2.0 = **70%.** The problem doubled; the cushion barely moved.
- **Year 3:** NPLs hit \$3.5 billion, reserves \$1.5 billion. Coverage = 1.5 / 3.5 = **43%.** The bank is now reserved for less than half of its acknowledged bad loans.

The trajectory is the tell. Coverage went 120% → 70% → 43% while NPLs went \$1bn → \$2bn → \$3.5bn. The bank is being swamped — and worse, the slow growth in reserves suggests it has been *propping up reported earnings* by under-provisioning, which means the true losses are even larger than the reserves admit. The intuition: **rising NPLs tell you the loan book is sick; falling coverage tells you management is either in denial or hiding it.** When that gap opens, the eventual write-down that closes it often takes the bank's equity with it. This pro-cyclical pattern — earnings looking great at the top, then collapsing as provisions catch up — is the heart of the credit cycle, the topic we develop fully alongside this post in the playbook.

## Red flag #6: market signals — AT1 spreads, CDS, and the share price

The flags so far come from the bank's own filings. This family comes from the *market* — and the market is often faster and more honest than the filings, because it is real-time and it is people betting their own money. When the market starts pricing failure, it is telling you something the quarterly report hasn't admitted yet.

There are three market tells, in roughly increasing order of severity.

**The share price and price-to-book ratio.** A bank's *book value* is its equity per share. A bank's *price-to-book (P/B) ratio* is its share price divided by that book value. A healthy bank trades around or above 1.0× book — the market believes the equity is real and will earn a decent return. When a bank's P/B falls *deep* below 1.0× — to 0.5×, 0.3× — the market is saying, in effect, "we do not believe your stated equity is really worth what you claim; we think there are losses you haven't taken yet." A collapsing P/B is the market front-running the write-down. We cover the valuation mechanics in the companion post on valuing a bank.

**CDS spreads.** A credit default swap is insurance on the bank's debt. Its spread (the annual premium, in basis points) is the market's price for the risk that the bank defaults. When a bank's CDS spread blows out from, say, 100 basis points to 1,000+ basis points, sophisticated creditors are paying up enormously to insure against its failure. CDS often moves days or weeks before depositors react, because the institutions trading it watch the bank closely.

**AT1 bond prices.** AT1 (Additional Tier 1) bonds are a bank's riskiest debt — designed to absorb losses (convert to equity or be written off entirely) if the bank's capital falls too far, *before* depositors lose a cent. When AT1 bonds trade down to distressed prices and their yields spike, the market is pricing a real chance those bonds get wiped. This is the most senior market warning short of an actual run.

The deep lesson is that these signals form a *sequence.* The share price and CDS usually move first, as the most-informed money repositions. Then the AT1 market cracks. Then — and only then — do the depositors notice, and the run begins. By the time depositors are queuing, the market has been screaming for weeks. The early-warning value of these signals is enormous precisely because they lead the run, not lag it.

#### Worked example: reading the market's failure probability

A rough but useful trick: a CDS spread, divided by an assumed loss-given-default, approximates the market's implied annual default probability. Suppose a bank's 5-year CDS trades at **1,000 basis points** (10% per year) and the market assumes that if it defaults, creditors lose about **60%** (a loss-given-default, or LGD, of 0.6).

Implied annual default probability ≈ spread / LGD = 0.10 / 0.6 ≈ **16.7% per year.**

Compare that to a healthy bank trading at 60 basis points: 0.006 / 0.6 ≈ **1% per year.** The market is pricing the stressed bank as roughly *seventeen times* more likely to default. The intuition: **the market's failure probability is computable, and when it jumps from 1% to 16%, the smart money has already concluded the bank is in trouble — long before the press release.** You do not need to be a CDS trader to use this; the *direction and speed* of the move is the signal, and it is public.

## Red flag #7: the run itself — deposit outflows accelerating

This is the last link before failure, and by the time it is clearly visible, the early-warning window has mostly closed. But understanding its dynamics matters, because the *shape* of the outflow tells you whether you are watching a wobble or a death spiral.

Deposit outflows are normal — money moves in and out of a bank constantly. The warning is not outflow, it is *accelerating* outflow, especially uninsured outflow, especially after a piece of bad news. A healthy bank's deposits drift gently. A failing bank's deposits fall, then fall faster, then collapse — because each departing depositor is a signal to the next one, and modern depositors can act on that signal in seconds.

Credit Suisse is the textbook case of the *slow* version. Through the fourth quarter of 2022, after a year of scandals and losses, clients pulled roughly **110 billion Swiss francs** — over a tenth of the bank's assets — in a single quarter. That was not yet a fatal run; the bank had a huge liquidity buffer and even tapped a 100-billion-franc liquidity line from the Swiss National Bank. But the outflow never reversed. Confidence, once cracked, did not heal. The slow bleed continued until, in March 2023, a final loss of confidence forced the Swiss authorities to engineer an emergency sale to UBS for about 3 billion francs over a weekend. The run was slow, then sudden — which is how most of them go.

SVB is the *fast* version: \$42 billion in a day, \$100 billion queued for the next morning, the whole thing over in 36 hours. The difference between the two is mostly the funding structure (flag #2): Credit Suisse had a more diversified, partly insured base that bled slowly; SVB had a concentrated, uninsured, wired base that hemorrhaged.

#### Worked example: an accelerating run

Track a bank's daily deposit balance during a stress.

- **Day 0:** \$175 billion. A bad earnings release and a falling share price the evening before.
- **Day 1:** \$133 billion — a \$42 billion outflow, 24% of deposits, in one day (SVB's actual Day 1).
- **Day 2 (projected):** another ~\$100 billion lined up to leave at the open.

To meet Day 1's \$42 billion, the bank must raise \$42 billion in cash *immediately.* If its liquid buffer is only \$30 billion, it must sell \$12 billion of bonds — at a loss, because rates rose — realizing the hidden HTM loss from flag #3 and crashing it through equity. Now it is visibly insolvent, which accelerates Day 2's outflow, which forces more fire sales, and the spiral closes. The intuition: **a run is not a single event but a feedback loop — outflows force losses, losses destroy confidence, lost confidence causes more outflows — and once the loop is spinning, only an outside rescue (a buyer, a central bank, a government guarantee) can stop it.** Which is exactly why regulators stress liquidity buffers sized to survive the *first* day, before the loop can start.

## Red flag #8: the CAMELS tells — what the examiner sees

Everything above maps onto the framework bank examiners have used for decades: **CAMELS.** It is an acronym for the six things a supervisor scores when rating a bank, each on a 1 (best) to 5 (worst) scale. The reason CAMELS is worth knowing is that it is a *complete* checklist — it forces you to look at all six dimensions, so you don't get mesmerized by one healthy number while another is screaming.

![Matrix mapping each CAMELS letter to what it measures and its warning tell](/imgs/blogs/spotting-the-next-bank-failure-the-early-warning-signs-8.png)

Walking the figure letter by letter:

- **C — Capital adequacy.** Is the equity cushion thick enough for the risks? Warning tell: capital ratios sitting near the regulatory minimum with no buffer, or capital that looks fine on a headline basis but evaporates once you subtract HTM losses (flag #3).
- **A — Asset quality.** How good is the loan book? Warning tell: rising NPLs, falling coverage, concentration in one risky sector (flags #1 and #5).
- **M — Management.** Are the people running the bank competent and honest? This is the soft, qualitative one — and the one that catches the *conduct* failures. Warning tells: accounting restatements, sudden senior-executive departures, auditor changes, aggressive earnings management, a CEO who attacks short-sellers instead of answering questions. Bad management is upstream of every other flag; it is the common cause behind reckless growth, under-reserving, and hidden losses.
- **E — Earnings.** Is the bank making money in a sustainable way? Warning tell: a falling NIM (flag #4), profits propped up by one-off gains or by under-provisioning, a return on assets bleeding toward zero.
- **L — Liquidity.** Can the bank survive a stress without fire sales? Warning tell: flighty, uninsured, concentrated funding (flag #2); a low or falling LCR; heavy reliance on overnight wholesale borrowing that can vanish.
- **S — Sensitivity to market risk.** How exposed is the bank to rate moves? Warning tell: a large duration gap — long-dated assets funded by overnight money — exactly the structure that produced SVB's hidden bond loss.

The discipline of CAMELS is that it is *and*, not *or*. A bank with a pristine capital ratio (C looks great) but a 94% uninsured deposit base (L is a disaster) is not a safe bank — it is SVB. The examiner's habit, and the one this whole post is teaching, is to refuse to be reassured by any single green light while a red one is on. The supervisory machinery that runs this scoring — stress tests, CCAR, living wills — is its own subject; we cover it in [stress testing and the supervisory exam](/blog/trading/banking/stress-testing-ccar-the-supervisory-exam-and-living-wills).

## Common misconceptions

**"A bank with healthy capital ratios is safe."** This is the single most expensive misconception in banking, and SVB is its tombstone. SVB reported capital ratios *above* its regulatory minimums right up until the week it failed, because its bond losses were hidden in the held-to-maturity bucket and never touched the headline ratio. Capital adequacy is necessary but not sufficient. A bank can be well-capitalized on paper and economically insolvent in reality, and it can be perfectly solvent and still die of a liquidity run. Capital answers "can it absorb losses?" — it says nothing about "can it survive everyone asking for their money on Tuesday?"

**"Profitable banks don't fail."** SVB was profitable. Credit Suisse had profitable divisions. Washington Mutual reported profits through much of the housing boom. Profitability tells you the bank made money *in the recent past, under the recent conditions.* It says nothing about the losses embedded in the balance sheet that haven't surfaced yet, or about whether the funding will stay put when conditions change. Profit is a backward-looking flow; failure is about the forward-looking stock of risk.

**"If a bank passed its audit and its stress test, it's fine."** Audits confirm the numbers are reported correctly under the accounting rules — including the rules that let HTM losses hide. Stress tests check specific scenarios; SVB's specific failure mode (a rate-shock-driven run on an ultra-concentrated uninsured base) was not the scenario its supervision was focused on. A clean audit and a passed stress test reduce the odds of *some* failure modes; they do not certify safety, and they explicitly do not catch the risks the rules let banks omit.

**"Deposit insurance means depositors won't run, so runs are a solved problem."** Insurance solves the run *for insured deposits.* It does nothing for the uninsured ones — and at a bank like SVB, 94% of deposits were uninsured. Insurance turned the small-saver run of the 1930s into a non-event; it did not touch the institutional, uninsured, digital run of 2023. The protection only works below the \$250,000 line, and the modern danger lives entirely above it. We cover the mechanism and its limits in [deposit insurance and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard).

**"Bank failures are random black swans you can't see coming."** This is the comforting lie this whole post exists to refute. Failures are clustered, structured, and *preceded by a consistent pattern of public red flags.* The flags don't give you a date, and a bank can carry red flags for years without failing if the shock that would detonate them never arrives. But the failures that do happen are, almost without exception, legible in advance to anyone who assembles the dashboard. The signs were there for SVB, for Credit Suisse, for WaMu, for Continental Illinois in 1984. The failure is rarely a surprise to those who were reading the right page.

## How it shows up in real banks: running the checklist

The real test of a warning system is whether it would have flagged the actual failures. Let's run the dashboard over three banks that died in three different ways — a fast liquidity collapse, a slow erosion of trust, and a classic credit blow-up — and see how many reds each one was flashing before the end.

### Silicon Valley Bank, March 2023 — the fast collapse

SVB is the cleanest possible test because it failed on liquidity and rate risk, almost untouched by credit losses. Run the checklist as of late 2022 / early 2023:

- **Growth (flag #1):** roughly doubled in size 2019–2021. **RED.**
- **Funding concentration (flag #2):** 94% uninsured, a tightly correlated venture-capital clientele. **DEEP RED** — the most concentrated, flighty base imaginable.
- **Rate shock / HTM loss (flag #3):** ~\$17 billion of unrealized securities losses against ~\$16 billion of tangible equity — fully marked, the equity was already gone. **DEEP RED.**
- **NIM (flag #4):** under pressure as funding costs rose and assets were locked into low-yield bonds. **AMBER.**
- **Credit / NPLs (flag #5):** loan book was reasonable; this was not a credit story. **GREEN.**
- **Market signals (flag #6):** the share price cracked after the bank announced a loss-making bond sale and a capital raise on March 8 — the spark. **RED, and the immediate trigger.**
- **The run (flag #7):** \$42 billion out on March 9, \$100 billion queued for March 10. **TERMINAL.**

Six of eight flags red, three of them deep red, and the one green flag (credit) was irrelevant to how this bank actually died. A dashboard reader looking at SVB in February 2023 would have seen one of the most fragile funding structures in the entire US banking system, sitting on a hidden loss bigger than its cushion. The collapse was fast, but it was not unforeseeable. We walk the full 36 hours in [Silicon Valley Bank 2023: the duration trap and the digital run](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run).

### Credit Suisse, 2022–2023 — the slow death of trust

Credit Suisse is the opposite shape: a slow, multi-year erosion driven mostly by the **M** in CAMELS — management and conduct — rather than a single market shock.

- **Growth / asset quality:** not a fast-growth story, but a long string of losses and scandals (the Archegos blow-up, the Greensill collapse, spying scandals, repeated leadership churn). **Management RED** — the foundational flag.
- **Earnings (flag #4):** chronically weak; the investment bank lost money for years. **RED.**
- **The run (flag #7):** ~CHF 110 billion of outflows in Q4 2022 alone — a slow-motion run, over 10% of assets in a quarter, never reversing. **RED.**
- **Market signals (flag #6):** the share price ground relentlessly lower, P/B fell to a small fraction of book value (deep below 1.0×), and CDS spreads widened to distressed levels through 2022 and into 2023. The market had been pricing failure for months. **DEEP RED.**
- **The AT1 wipeout:** when UBS bought Credit Suisse in March 2023 for about CHF 3 billion, the Swiss regulator wrote **CHF 16 billion of AT1 bonds to zero** — the loss-absorbing debt did exactly what it was designed to do, ahead of depositors.

Credit Suisse shows that the warning system works even when there is no single dramatic loss. The flags accumulated slowly — bad management, weak earnings, persistent outflows, a market that had given up — and the dashboard would have read RED for well over a year before the forced sale. The lesson: the *slow* failures give you the longest warning of all, if you are watching the trend rather than waiting for an event. The full story is in [Credit Suisse 2023: the slow death of trust and the AT1 wipeout](/blog/trading/banking/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout).

### Washington Mutual, 2008 — the classic credit blow-up

WaMu was the largest bank failure in US history (about \$307 billion in assets at failure), and it failed the old-fashioned way: bad loans, in volume, into a housing bust.

- **Growth (flag #1):** grew aggressively through the housing boom, piling into option-ARM and subprime mortgages — the riskiest products in the market. **DEEP RED.**
- **Asset quality / NPLs (flag #5):** as house prices fell in 2007–2008, defaults surged across exactly the loan types WaMu had concentrated in. NPLs rose sharply while the reserves built during the boom proved far too thin for the scale of the losses — coverage collapsed. **DEEP RED.**
- **Funding (flag #2):** as the losses became visible, depositors pulled roughly \$16–17 billion over a couple of weeks in September 2008, draining the funding base. **RED.**
- **Market signals (flag #6):** the share price had been in freefall for a year; by September 2008 it was trading at a tiny fraction of its peak. **RED.**

WaMu is the template for the *credit-cycle* failure: fast growth into a hot sector (flag #1) sows the loan losses that bloom when the cycle turns (flag #5), and the visible losses then trigger the funding run (flags #2 and #7) that finishes the bank. It is slower than a pure liquidity collapse — it played out over a year of rising defaults — which means the warning window was *long.* Anyone watching WaMu's NPL ratio climb and its coverage fall through 2007 was watching the failure arrive in slow motion. The detail is in [Washington Mutual and the 2008 mortgage-bank failures](/blog/trading/banking/washington-mutual-and-the-2008-mortgage-bank-failures).

### The base rate: failures are rare but clustered

One more piece of context, because it guards against false alarms. Most banks, in most years, do not fail.

![Bar chart of US bank failures per year from 2005 to 2025 with peaks in 2010 and a small spike in 2023](/imgs/blogs/spotting-the-next-bank-failure-the-early-warning-signs-5.png)

The chart shows FDIC-insured bank failures per year. In the calm years — 2005, 2006, 2018, 2021, 2022 — the count is *zero or near zero.* Then the cycle turns and they cluster: the 2008–2012 wave peaked at **157 failures in 2010**, and the 2023 episode produced a small but high-profile cluster of **5** (SVB, Signature, First Republic, and two others). This base rate is the crucial calibration for the dashboard. Red flags do not mean a bank fails next quarter — in a benign environment, a flag-heavy bank can carry its risks for years. What the flags really measure is *fragility*: how little margin for error the bank has left when the cycle finally turns. The flags tell you which banks will be in the next cluster, not when the cluster arrives. The macro trigger — a rate shock, a recession, a credit event — sets the timing; the flags determine who gets swept up. For why bank earnings and failures move with the cycle, see the companion post on the credit cycle.

## The takeaway: how to use the dashboard

So how do you actually use this? Not as a crystal ball, but as a disciplined reading order. Here is the dashboard distilled into a usable process.

**Read the column, not the cell.** The whole skill is resisting the comfort of a single green number. SVB's capital ratio was green; it did not matter. WaMu's reported profits were fine for years; they did not matter. Pull all eight flags and look at how many are red *at the same time, in the same bank.* One or two reds is a normal bank with normal risks. Five or more reds, especially if they include the funding and hidden-loss flags, is a bank that has lost its margin for error.

![Two column comparison of a fragile bank profile in red and a resilient bank profile in green across six metrics](/imgs/blogs/spotting-the-next-bank-failure-the-early-warning-signs-6.png)

The figure above is the dashboard read as a *profile* rather than a single score. The fragile bank reads red on every line — loan growth at twice its peers, 94% uninsured deposits, a hidden loss that wipes out equity, a falling margin against rising bad loans, a credit spread blowing out, and a price below a third of book value. The resilient bank reads green on the same six lines: growth near the peer pace, deposits mostly insured, hidden losses small against a thick cushion, a stable margin with rising coverage, a tight credit spread, and a price near book value. Resilience is not one heroic ratio; it is the *absence of a stacked red column.* When you finish reading a bank, the question is not "is any single number alarming?" but "does this institution's whole profile lean red or lean green?"

**Follow the chain upstream.** The flags are causally ordered: growth feeds concentration, concentration plus a shock opens a hidden loss, the loss cracks trust, trust becomes a run. By the time you see the run (flag #7), the early-warning game is over. The value of the dashboard is at the *top* of the chain — at rapid growth and funding concentration, which are visible years before the run. The earlier you read, the more the dashboard is worth.

**Weight the funding and hidden-loss flags most heavily.** If you only had time to check two numbers, check the **uninsured-deposit share** (flag #2) and the **HTM unrealized loss versus equity** (flag #3). Those two were the entire SVB story, and they are pullable straight from a public filing. A high uninsured share is the gasoline; a hidden loss bigger than the cushion is the bomb; a falling share price is the lit match.

**Use the market as your early radar.** The CDS spread, the AT1 price, and the price-to-book ratio move *before* depositors react, because they are priced by the most-informed money in real time. A bank whose P/B has collapsed to 0.3× and whose CDS has blown out is a bank the market has already judged. You do not have to agree with the market, but when it is screaming, the burden of proof is on the bank, not on the skeptic.

**Respect the base rate, and don't cry wolf.** Failures are rare. A red-flagged bank in a calm macro environment may carry its risks for years without dying. The dashboard does not let you predict the date — it lets you identify *fragility*, and fragility only becomes failure when a shock arrives. Treat the flags as a measure of how much can go wrong before this particular bank breaks, not as a countdown timer.

Step back and the whole thing collapses to the spine of this series. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short and lends long, earns the spread, and survives only as long as depositors trust it and its thin equity cushion absorbs losses faster than they arrive. Every red flag on the dashboard is just a measurement of that one fragile trade going wrong somewhere. Rapid growth is the machine taking on too much. Funding concentration is the borrowed-short side getting flightier. HTM losses are the lent-long side losing value. A falling NIM is the spread compressing. Rising NPLs are the loans going bad. Widening spreads and a falling share price are the market noticing. And the run is the confidence — the thing the whole structure is built on — finally giving way. Read the dashboard and you are not really reading eight metrics. You are reading the health of one trade, the trade every bank lives or dies on.

## Further reading & cross-links

- [The anatomy of a bank run: from whisper to collapse](/blog/trading/banking/the-anatomy-of-a-bank-run-from-whisper-to-collapse) — the dynamics of the run that sits at the end of the red-flag chain.
- [Silicon Valley Bank 2023: the duration trap and the 36-hour digital run](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run) — the canonical worked example of flags #2 and #3 detonating together.
- [Credit Suisse 2023: the slow death of trust and the AT1 wipeout](/blog/trading/banking/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout) — the slow-burn failure driven by management and market signals.
- [Washington Mutual and the 2008 mortgage-bank failures](/blog/trading/banking/washington-mutual-and-the-2008-mortgage-bank-failures) — the classic credit-cycle blow-up: fast growth, rising NPLs, falling coverage.
- [Interest-rate risk in the banking book (IRRBB) and the duration gap](/blog/trading/banking/interest-rate-risk-in-the-banking-book-irrbb-and-the-duration-gap) — the mechanics behind the hidden bond loss.
- [Liquidity management: LCR, NSFR and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — the buffer that is supposed to survive the first day of a run.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) and [risk-weighted assets](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work) — what the capital flag is really measuring.
- [Stress testing, CCAR, the supervisory exam and living wills](/blog/trading/banking/stress-testing-ccar-the-supervisory-exam-and-living-wills) — the official version of the CAMELS dashboard.
- [Deposit insurance, the lender of last resort and moral hazard](/blog/trading/banking/deposit-insurance-the-lender-of-last-resort-and-moral-hazard) — why insured money is sticky and uninsured money runs.
