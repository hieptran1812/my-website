---
title: "How to Read a Bank's Annual Report: The Analyst's Checklist"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A repeatable workflow for reading a bank's 10-K and Pillar 3 disclosure: the ratios that matter, the footnotes that bite, and where the truth hides."
tags: ["banking", "bank-analysis", "10-k", "financial-statements", "capital-ratios", "cet1", "net-interest-margin", "liquidity", "pillar-3", "credit-risk"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bank's annual report is written to reassure you; your job is to read it in the order a bank actually dies in, so the reassurance can't hide the fracture.
>
> - Read in failure order, not the report's order: **funding → assets → capital → risk → footnotes**. A bank breaks on the right side of its balance sheet (its funding) long before the income statement looks sick.
> - Six ratios carry most of the signal: **NIM** (the spread), **efficiency ratio** (cost discipline), **CET1** (the loss-absorbing cushion), **NPL and coverage** (asset quality), **LCR** (the liquidity buffer), and **ROE/ROA** (how the franchise is judged). Read them together; any one alone lies.
> - The danger almost never sits on the face of the statements. It sits in a **footnote** — the held-to-maturity (HTM) securities note, the uninsured-deposit disclosure, the loan-concentration table, the off-balance-sheet commitments.
> - The one number to remember: a typical commercial bank funds itself with about **8% equity** and **92% other people's money** — roughly **12.5× leverage** — so a loss of just **8% of assets** wipes out the owners. Everything you read is a hunt for the loss that gets there first.

In March 2023, Silicon Valley Bank's last annual report was sitting on the internet, fully public, audited, and clean. Its reported common equity tier 1 (CET1) ratio — the core measure of how much loss-absorbing capital a bank holds against its risk — looked comfortable. Its reported equity was about \$16 billion. By every headline number on the face of the statements, here was a well-capitalised bank.

Then a few analysts and a lot of venture capitalists did something the headline numbers didn't require: they turned to a footnote. In the note on the bank's securities portfolio sat a single, devastating line — the *unrealised loss* on its held-to-maturity bonds, the bonds it carried at original cost rather than at today's market price. That number was about \$17 billion. Mark it against the \$16 billion of reported equity and the bank wasn't well-capitalised at all. It was, on a fair-value basis, insolvent. Within 36 hours, \$42 billion of deposits had tried to leave, and the bank was gone.

The lesson of SVB is not that banks are unknowable. It is the opposite. Everything that killed SVB was *disclosed* — in the annual report, in the right footnote, months ahead, for anyone who knew where to look and read in the right order. This post is that order. The diagram below is the mental model we'll build toward: the analyst's ratio dashboard, the six numbers that summarise a bank's health and the red flag that each one throws when it goes wrong. By the end you'll have a repeatable checklist you can run on any bank's 10-K, annual report, or Basel Pillar 3 disclosure — and you'll know which footnotes to open before you trust a single headline.

![bank analyst ratio dashboard NIM efficiency CET1 NPL coverage LCR ROE healthy ranges and red flags](/imgs/blogs/how-to-read-a-banks-annual-report-the-analysts-checklist-1.png)

This is a synthesis post. The whole "How a Bank Actually Works" series has built up the machinery — what a balance sheet is, how net interest income forms, how capital ratios are computed, how liquidity rules work, how banks fail. Here we assemble it into a single workflow. Where a concept has a dedicated post, I'll point you there rather than re-derive it. This is educational, not investment advice; the point is to make a bank's report *legible*, not to tell you what to buy.

## Foundations: what a bank's annual report actually is

Before we read anything, we need the vocabulary. A reader with no banking background can be completely lost in a 10-K not because the ideas are hard but because the documents have specific names and specific jobs. Let's define them from zero.

### The documents: 10-K, annual report, and Pillar 3

A **10-K** is the comprehensive annual filing a US-listed company submits to the Securities and Exchange Commission (SEC). It is the legally exhaustive version — the audited financial statements plus a long discussion of the business and its risks. Outside the US the equivalent is the **annual report** (and, in the EU and UK, a separate set of regulatory disclosures). When people say "read the 10-K", they mean: read the audited truth, not the glossy marketing brochure that often wraps around it.

A bank files a second document most companies don't: the **Basel Pillar 3 disclosure**. Basel is the global framework of bank-capital rules set by the Basel Committee at the Bank for International Settlements; "Pillar 3" is its market-discipline pillar, which forces banks to publish standardised tables on capital, risk-weighted assets, leverage, and liquidity. (The [BIS and Basel one-pager](/blog/trading/finance/bis-and-basel-bank-regulation) covers the framework's history; here we just use its output.) The Pillar 3 report is where a bank shows its work on the regulatory ratios — the CET1 ratio, the leverage ratio, the liquidity coverage ratio. If the 10-K is the story, Pillar 3 is the spreadsheet behind the story.

A *footnote* (or "note to the accounts") is a numbered disclosure attached to the financial statements that explains, breaks down, or qualifies a line on the face of the statements. The face of the balance sheet might say "Securities: \$120 billion". The footnote tells you how much of that is carried at cost versus market, and what the difference would be if you marked it. **The face of the statements is summary; the footnotes are where the summary is true or false.**

A practical note on geography, because it changes where you look. A US bank files a 10-K with the SEC, an 8-K for material events, and quarterly 10-Qs; its Pillar 3 tables appear in a standalone regulatory disclosure and in the Federal Reserve's FR Y-9C call report. A European bank files an annual report under IFRS accounting and a separate Pillar 3 report under the EU's Capital Requirements Regulation. The *names* differ, but the *content* an analyst needs — capital ratios, RWA, liquidity ratios, deposit composition, loan quality, fair-value disclosures — is the same everywhere, because Basel standardised it globally. So the checklist in this post travels: you run the same steps on JPMorgan's 10-K, HSBC's annual report, or any mid-sized bank's filing; only the page numbers move.

One more distinction that matters for the footnotes later: the difference between *recognition* and *disclosure*. A number is **recognised** when it appears on the face of the statements and flows into equity or profit — a marked-to-market AFS loss is recognised. A number is **disclosed** when it appears only in a footnote and does *not* touch the headline equity — an HTM unrealised loss is merely disclosed. The entire art of reading a bank is knowing which dangers are recognised (you can see them in the headline) and which are only disclosed (you must dig). Almost every bank that surprised the market did so with a *disclosed-but-not-recognised* loss that the headline ratios were free to ignore.

### The four parts and what each tells you

A bank's annual report has four parts an analyst cares about, and they answer different questions. The diagram below maps them.

![parts of a bank 10-K and what each one tells the analyst](/imgs/blogs/how-to-read-a-banks-annual-report-the-analysts-checklist-2.png)

- **The CEO letter / chairman's statement** — the narrative. This is management telling you what they want you to believe. Read it last and read it skeptically; it is the only part written to persuade rather than to disclose. Useful for tone and strategy, useless for facts.
- **The financial statements** — the audited numbers: the balance sheet (a snapshot of assets, liabilities, and equity), the income statement (a year of revenue and cost), and the cash-flow statement. This is the spine.
- **The MD&A** (Management's Discussion and Analysis) — management's own walk through the numbers, segment by segment, with the ratios and the year-over-year movements. Often the fastest way to find the metrics, though management chooses which to emphasise.
- **The notes and Pillar 3** — the footnotes and the regulatory tables. This is where the risk lives: fair-value disclosures, deposit composition, loan concentrations, off-balance-sheet commitments, capital ratios.

### The headline ratios, defined once

Everything downstream depends on six ratios. Let's define each in plain English now, so the deep sections can use them freely. (Each has a fuller treatment elsewhere in the series; I'll link as we go.)

- **Net interest margin (NIM)** — the spread the bank earns on its assets, expressed as a percent. It's net interest income (interest earned on loans and securities minus interest paid on deposits and borrowings) divided by average interest-earning assets. A bank is fundamentally a spread business; NIM is the spread. (See [the income statement of a bank](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions).)
- **Efficiency ratio** — non-interest expense (salaries, premises, technology) divided by total revenue. It answers: how many cents does this bank spend to earn a dollar? Lower is better; under 60% is good, over 70% is a bloated or struggling bank.
- **CET1 ratio** — common equity tier 1 capital (the highest-quality, first-loss equity) divided by risk-weighted assets (RWA, the bank's assets re-weighted by how risky each is). It's the single most important solvency number. (See [risk-weighted assets and how capital ratios really work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work).)
- **NPL ratio and coverage** — non-performing loans (loans where the borrower has stopped paying, usually 90+ days overdue) as a share of total loans, and the coverage ratio, which is loan-loss reserves divided by NPLs. NPL tells you how much has gone bad; coverage tells you whether the bank has already set aside money for it.
- **LCR** (liquidity coverage ratio) — high-quality liquid assets divided by expected net cash outflows over a 30-day stress scenario. It must be at least 100%. It answers: if a run started today, could the bank survive a month without help? (See [liquidity management: LCR, NSFR and the buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer).)
- **ROE and ROA** — return on equity (net income / shareholders' equity) and return on assets (net income / total assets). These are how investors judge the franchise. The rule of thumb: a healthy bank earns about 1% ROA and 10–15% ROE. (See [ROE, ROA and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged).)

### The one fact under everything: the thin cushion

Here is the spine of this whole series, and the reason the checklist is shaped the way it is. A bank is a leveraged, confidence-funded maturity-transformation machine. It borrows short — mostly deposits — and lends long — mortgages, business loans, bonds — and earns the spread. It survives only as long as depositors keep their trust and its thin sliver of equity absorbs losses faster than they arrive.

How thin is the sliver? The chart below shows how a typical commercial bank funds itself.

![how a typical commercial bank funds itself deposits wholesale debt equity share of funding](/imgs/blogs/how-to-read-a-banks-annual-report-the-analysts-checklist-6.png)

About 71% of the funding is deposits, 10% wholesale and repo borrowing, 7% long-term debt, 4% other liabilities — and just **8% equity**. That 8% is the owners' money, the cushion that absorbs losses before depositors lose a cent. Equity of 8% means leverage of roughly 1 / 0.08 = **12.5×**: every dollar of equity supports about \$12.50 of assets. Flip it around and you get the number that should haunt every page you read: **a loss equal to just 8% of assets wipes out the owners entirely.** A bank does not need a 50% catastrophe to fail. It needs an 8% one. Everything in the annual report is, at bottom, a hunt for the loss that gets to 8% first.

#### Worked example: turning the funding mix into leverage

Take a stylised bank with \$100 billion of assets, funded exactly per the chart. Equity is 8% × \$100 billion = \$8 billion. Deposits are 71% × \$100 billion = \$71 billion. Now suppose the bank's loan book sours and it must write down \$5 billion of loans — that's 5% of assets. Equity falls from \$8 billion to \$8 − \$5 = \$3 billion. The bank is still solvent, but its equity has dropped 62.5% (from \$8 billion to \$3 billion). A *5% asset loss* became a *62.5% equity loss*. That is leverage working in reverse: it multiplies the percentage hit to the owners by about 12.5×. The intuition: in a bank, small asset losses are large equity losses, which is exactly why the capital ratio is the number you check first when anything goes wrong.

## Read the funding side first — a bank dies on the right

Most people open a company's report and go straight to profit. With a bank, that's backwards. A bank almost never fails because it stopped being profitable; it fails because it couldn't fund itself one Friday afternoon. So we start on the right-hand side of the balance sheet — the funding — and only then move to the assets.

The order matters enough to draw it. The pipeline below is the read sequence we'll follow for the rest of the post.

![the order to read a bank funding assets capital risk footnotes](/imgs/blogs/how-to-read-a-banks-annual-report-the-analysts-checklist-4.png)

### What to look for in the funding mix

Pull up the liabilities section of the balance sheet and the deposit footnote. You want three things.

First, **how much is deposits versus wholesale funding.** Deposits — especially retail current and savings accounts — are the cheap, sticky franchise. Wholesale funding (interbank borrowing, repo, commercial paper) is fast money that flees the instant the bank looks shaky. A bank that funds 71% with deposits is far more robust than one funding 50% with deposits and 40% with overnight wholesale money. Northern Rock in 2007 funded itself heavily in the wholesale market; when that market froze, so did the bank.

Second, **the composition of the deposits themselves.** Non-interest-bearing current accounts (operating cash that businesses and households leave because they need it for transactions) are the stickiest and cheapest. Hot, rate-shopping term deposits are the least sticky. The CASA ratio — current and savings accounts as a share of total deposits — is the quick read here. (We'll come back to the *insured versus uninsured* split, which is the single most important deposit number, in the footnotes section.)

Third, **the cost of that funding.** This connects straight to the income statement, because cheap funding is what makes the spread. There's a specific number to hunt for here, called the *deposit beta* — the fraction of a central-bank rate increase that the bank ends up passing through to its depositors. A bank with a low deposit beta keeps paying savers very little even as rates rise, so its funding stays cheap and its margin widens; a bank with a high beta has to raise deposit rates almost in lockstep with the central bank, and its margin barely moves. Through the 2022–23 hiking cycle, industry cumulative deposit beta climbed from roughly 0.10 early on to about 0.55 by mid-2024 — meaning banks eventually passed a bit more than half of each rate rise to depositors. A bank disclosing a beta well below peers is signalling either a genuinely loyal, rate-insensitive deposit base (a real franchise) or a base that is about to wake up and demand higher rates (a margin squeeze waiting to happen). The cost-of-funding trend across two or three years tells you which.

Fourth, while you're on the funding side, glance at the **NSFR** — the net stable funding ratio, the LCR's slower cousin. Where the LCR asks "can you survive a 30-day run?", the NSFR asks "is your *long-term* funding stable enough to support your *long-term* assets?" It's available stable funding divided by required stable funding, and like the LCR it must be at least 100%. A bank that funds long-dated loans and bonds with stable deposits and term debt scores well; a bank that funds them with overnight wholesale money scores badly — and that structural mismatch is precisely what broke Northern Rock and the savings-and-loans. The NSFR is the funding-stability number; the LCR is the funding-survival number. Read both. (Both are detailed in [liquidity management: LCR, NSFR and the buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer).)

All of this connects straight back to the spread, because the cheapness and stickiness of the funding base is half of the net interest margin. Which brings us to the spread itself.

### NIM: is the spread real, or is it reaching?

Net interest margin is the bank's core engine. But a number alone is meaningless without a benchmark. The chart below shows the US commercial-banking industry's NIM through a full rate cycle, so you can place any single bank against the backdrop.

![US commercial bank net interest margin 2010 to 2024 industry aggregate](/imgs/blogs/how-to-read-a-banks-annual-report-the-analysts-checklist-9.png)

NIM ran around 3.4–3.8% in the early 2010s, got crushed to a trough of 2.56% in 2021 when the Federal Reserve held rates near zero, then jumped back above 3% as the Fed hiked in 2022–23. So a bank reporting a 3.2% NIM in 2024 is roughly in line with the industry; one reporting 4.5% is earning a much wider spread than peers — which prompts the analyst's real question: *why?*

A high NIM can be a sign of strength (a cheap deposit base, pricing power) or a sign of danger (the bank is lending to riskier borrowers at higher rates, which means higher losses are coming). A NIM that's high *and* paired with a rising NPL ratio is a red flag, not a strength. This is why no ratio is read alone.

#### Worked example: computing NIM from the disclosed line items

A bank's income statement and balance sheet give you everything. Say the bank reports:

- Interest income (on loans and securities): \$4.2 billion
- Interest expense (on deposits and borrowings): \$1.4 billion
- Average interest-earning assets: \$90 billion

Net interest income = \$4.2 billion − \$1.4 billion = \$2.8 billion. NIM = net interest income / average interest-earning assets = \$2.8 billion / \$90 billion = 0.0311 = **3.11%**. That sits right on the industry line, so this bank's spread is unremarkable — which is reassuring. The intuition: NIM converts the bank's profit-and-loss into a single comparable spread, so a bank with \$90 billion of assets and one with \$900 billion can be judged on the same footing.

### The efficiency ratio: cost discipline in one number

While we're on the income statement, the efficiency ratio is the fastest read on management quality. It's non-interest expense divided by total revenue (net interest income plus fee income).

#### Worked example: computing the efficiency ratio

Continue the bank above. Net interest income is \$2.8 billion. Add fee income of \$1.2 billion, so total revenue is \$2.8 + \$1.2 = \$4.0 billion. Non-interest expense (staff, buildings, technology) is \$2.3 billion. Efficiency ratio = \$2.3 billion / \$4.0 billion = 0.575 = **57.5%**. The bank spends about 58 cents to earn each dollar of revenue — comfortably under the 60% threshold, so cost discipline looks good. The intuition: the efficiency ratio is the one number that tells you whether a bank is run lean or bloated, independent of how big its spread is.

## Read the asset side — what is the bank actually holding?

Now flip to the left side of the balance sheet: the assets. A bank's assets are mostly loans and securities, and the analyst's job is to ask what's hiding inside each pile. (The full anatomy is in [reading a bank balance sheet](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity); here we read it as a checklist.)

### Loans: concentration and quality

Loans are the core asset. The loan footnote breaks the book down by type — commercial real estate, residential mortgages, consumer, corporate — and by geography. You're hunting for **concentration**: a bank with 40% of its loans in one sector (say, office commercial real estate) is one bad sector away from a capital problem. Diversification is a defence; concentration is a fuse. The S&L crisis, the 2008 mortgage-bank failures, and a good share of the 2023 regional-bank stress all trace back to one over-weighted exposure.

The loan footnote also discloses how the bank *reserves* against those loans, under one of two expected-credit-loss regimes: IFRS 9 (most of the world) or CECL (the US). Both force the bank to set aside reserves for losses it *expects* in the future, not just losses that have already happened. IFRS 9 sorts loans into three "stages": Stage 1 (performing, reserve for 12-month expected loss), Stage 2 (significant increase in credit risk, reserve for lifetime expected loss), Stage 3 (credit-impaired, the non-performing bucket). The analyst's tell is the *migration* between stages: a bank where loans are sliding from Stage 1 into Stage 2 quarter over quarter is watching its book deteriorate before any of it has formally defaulted. That migration is a leading indicator the NPL ratio — which only captures Stage 3 — won't show you for another year.

### Securities: the HTM/AFS distinction that broke SVB

A bank's securities are split, on the balance sheet, into two buckets, and the difference between them is the most important accounting subtlety in all of bank analysis.

- **Available-for-sale (AFS)** securities are marked to market: their value on the balance sheet moves with the market, and unrealised gains or losses flow through equity (specifically, through "accumulated other comprehensive income").
- **Held-to-maturity (HTM)** securities are carried at *amortised cost* — essentially the price the bank paid — on the assumption it will hold them to maturity and collect par. Their market value is *not* on the face of the balance sheet. If rates rise and the bonds lose value, that loss does not show up in reported equity at all. It is disclosed only in a footnote.

This is the trap. When the Fed hiked rates in 2022–23, the market value of long-dated bonds fell sharply. A bank with a big HTM book showed no loss in its headline equity — but the loss was real, sitting in the securities footnote, waiting. If the bank were ever forced to sell those bonds (to meet deposit withdrawals, say), the paper loss would become a realised loss and crash through its capital.

The before-and-after below is the SVB lesson in one figure.

![book value versus marked to market value once you read the HTM footnote SVB](/imgs/blogs/how-to-read-a-banks-annual-report-the-analysts-checklist-5.png)

#### Worked example: marking the HTM footnote to market

Take SVB's actual disclosed numbers. The HTM book was about \$91 billion, carried at cost. The footnote disclosed an unrealised loss across the AFS and HTM portfolios of about \$17 billion. Reported equity was about \$16 billion.

Now do the mark: true economic equity = reported equity − the unrealised loss the headline ignores = \$16 billion − \$17 billion = **−\$1 billion**. On a fair-value basis, the bank had negative equity. The CET1 ratio that looked fine was computed on the *unmarked* book; mark the book and the cushion was gone. The intuition: HTM accounting lets a bank hide a bond loss from its headline capital, so the securities footnote is the first place you check whenever rates have moved against a bank.

This single distinction — AFS marked, HTM not — is why "read the footnotes" is not generic advice. The footnote held the whole story months before the run. (The full case is in [the SVB 2023 deep dive](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run).)

## Read the capital stack — can it absorb the loss?

Having seen what the bank holds and how it's funded, we ask the solvency question: if losses come, is there enough equity to absorb them? This is the capital section, and it's where the Pillar 3 tables earn their keep.

### CET1, RWA, and why the denominator matters

The CET1 ratio is CET1 capital divided by risk-weighted assets. The numerator is the highest-quality equity. The denominator, RWA, is the bank's assets re-weighted by riskiness: cash and government bonds get a 0% or low weight, a residential mortgage maybe 35%, an unsecured corporate loan 100% or more. So \$100 billion of assets might be only \$60 billion of RWA, or \$90 billion, depending on the mix.

This is also where banks play games: a bank using its own internal models (the IRB approach) can assign lower risk weights than one using the standardised approach, flattering its ratio. That's why regulators added the **leverage ratio** — Tier 1 capital divided by *total* unweighted assets — as a backstop that ignores the risk-weighting entirely. If a bank's CET1 ratio looks healthy but its leverage ratio is near the 3% floor, the risk-weighting is doing suspicious work. (The full mechanics, including the denominator games, are in [risk-weighted assets and how capital ratios really work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work).)

The chart below puts a bank's CET1 against what the regulator actually demands. The demand is built up from the 4.5% minimum, the 2.5% capital-conservation buffer, and (for a systemic bank) a surcharge — here illustrated at 1.5%, for an effective demand of 8.5%.

![CET1 ratio versus the regulatory demand minimum buffer surcharge](/imgs/blogs/how-to-read-a-banks-annual-report-the-analysts-checklist-8.png)

A "healthy bank" at 13.0% CET1 has 4.5 percentage points of headroom above the 8.5% demand — a real cushion. A "thin bank" at 9.0% is barely above the line, and crucially, only just above the buffer: dip into the conservation buffer and the regulator restricts dividends and bonuses. A bank running close to the demand line has no room for a bad year.

#### Worked example: computing the CET1 ratio from disclosed items

A bank's Pillar 3 disclosure gives you the pieces directly. Say it reports:

- CET1 capital: \$12 billion
- Risk-weighted assets: \$100 billion

CET1 ratio = \$12 billion / \$100 billion = 0.12 = **12.0%**. Against an 8.5% regulatory demand, that's 3.5 points of headroom. Now stress it: suppose a recession forces \$4 billion of loan losses, which come straight out of CET1. New CET1 = \$12 − \$4 = \$8 billion; ratio = \$8 billion / \$100 billion = **8.0%** — now *below* the 8.5% demand, into the buffer, dividends frozen. The intuition: the CET1 ratio isn't just a snapshot; the analyst's real use of it is to ask how big a loss the bank can take before the regulator steps in, and 3.5 points of headroom on \$100 billion of RWA is \$3.5 billion of loss capacity.

### The capital stack: what sits above CET1

CET1 is the bottom of the loss-absorbing stack — the first money in to take a hit. Above it sit additional tier 1 (AT1) instruments and tier 2 capital. AT1 includes "contingent convertible" bonds (CoCos), which are designed to convert to equity or be written off entirely when the bank's CET1 ratio falls below a trigger. The point of the stack is that losses climb it from the bottom: common equity absorbs first, then AT1, then tier 2, and only then do senior depositors and bondholders take a loss. When you read the capital section, note how much of the "total capital" is genuine common equity (CET1) versus these higher-up instruments — a bank leaning heavily on AT1 to hit its total-capital target has less true first-loss cushion than the headline total suggests.

The 2023 collapse of Credit Suisse delivered a brutal lesson here: when UBS absorbed it in a regulator-brokered deal, the Swiss authorities wrote off about CHF 16 billion of AT1 bonds to zero *while* shareholders still received some value — inverting the order investors had assumed. The footnote-level point for an analyst: read the *terms* of a bank's AT1 instruments (the trigger level, the write-down mechanics), because in a crisis they behave very differently from ordinary bonds, and the capital "cushion" they provide is conditional on those terms.

### The buffers and the dividend brake

Above the 4.5% CET1 minimum sits the capital-conservation buffer (2.5%), and on top of that, for systemic banks, a G-SIB surcharge and a countercyclical buffer. These buffers aren't a hard floor in the sense the minimum is — a bank can technically operate inside them — but doing so triggers automatic restrictions on dividends and bonuses (the "maximum distributable amount" brake). So the analyst's real question isn't "is the bank above the 4.5% minimum?" but "how close is it to the *buffer* line where the regulator takes the dividend away?" A bank running with only half a point of headroom above its buffer requirement is one bad quarter from cutting its dividend, which is itself a signal that often triggers a share-price slide and a loss of confidence — the start of the doom loop.

### The leverage ratio backstop

Run the same numbers through the leverage ratio. If that bank's *total* assets (unweighted) are \$160 billion and its Tier 1 capital is \$13 billion, its leverage ratio is \$13 billion / \$160 billion = 8.1%. Comfortably above the 3% Basel floor (and the 5–6% enhanced floor for the largest US banks). If instead the leverage ratio came out at 3.2% while the CET1 ratio read 14%, you'd know the risk-weighting was flattering the picture — the bank holds a lot of assets the models call "low risk". Always read the two ratios together: the CET1 ratio is the bank's view of its own risk, and the leverage ratio is the regulator's refusal to take that view on trust.

## Read the risk disclosures — what can go wrong, and how much?

The capital section tells you the size of the cushion. The risk section tells you the size of the punch coming at it. The four risks every bank runs are credit, market, liquidity, and operational; the report discloses all four, and the analyst weights credit and liquidity most heavily for a commercial bank.

### Asset quality: NPLs and coverage

The asset-quality disclosure is where you find out how much of the loan book has already gone bad. Two numbers:

- **The NPL ratio** — non-performing loans / total loans. Rising is bad; the *trend* matters as much as the level. Under 2% is healthy for a developed-market commercial bank; rising quarter on quarter is a flashing light.
- **The coverage ratio** — loan-loss reserves / NPLs. This tells you whether the bank has already set money aside against the bad loans. Coverage above 100% means reserves exceed the face of the bad loans (prudent); coverage below 60% means the bank is under-reserved and future write-offs will hit earnings and capital that haven't been booked yet.

A bank with a low NPL ratio and high coverage is conservatively run. A bank with a *rising* NPL ratio and *falling* coverage is releasing reserves to flatter earnings just as losses build — the classic late-cycle tell. (The mechanics of bad-debt management are in [non-performing loans and the workout process](/blog/trading/banking/non-performing-loans-and-the-workout-process).)

There's a subtlety worth flagging: the *provision* for credit losses on the income statement and the *reserve* (allowance) on the balance sheet are connected but not identical. The provision is the expense the bank books *this period* to top up (or release from) the reserve. So a bank can post a great quarter simply by *releasing* reserves — booking a negative provision because it now expects fewer losses — which flatters profit without anything real improving. Conversely, a bank "building" reserves takes an earnings hit now for losses it expects later. When you read the income statement, separate the bank's *operating* profit (pre-provision) from its *reported* profit (post-provision), because provisions are management's most discretionary lever and the one most used to smooth — or manufacture — an earnings trend.

### Operational risk: the loss that comes from inside

Credit and liquidity dominate a commercial bank's risk, but the report also discloses *operational* risk — the danger of loss from failed processes, fraud, cyber-attacks, or misconduct. This is the risk that doesn't show up in any ratio until it detonates: a rogue trader, a control failure, a money-laundering breach, a mis-selling scandal. The operational-risk section and the "legal proceedings" footnote disclose pending litigation, regulatory investigations, and provisions for fines. A bank with a long and growing list of regulatory matters is telling you something about its *culture* that no capital ratio captures. The biggest conduct failures — fake accounts, benchmark rigging, laundering — each cost their banks billions in fines and far more in lost trust, and every one of them was a culture problem that the numbers caught only after the fact. Read the litigation footnote the way you'd read a doctor's note: a clean one is reassuring, a thick one is a symptom.

#### Worked example: computing NPL and coverage, and the hit to capital

A bank reports:

- Total loans: \$80 billion
- Non-performing loans: \$1.6 billion
- Loan-loss reserves: \$1.3 billion

NPL ratio = \$1.6 billion / \$80 billion = 2.0% — right at the edge of "healthy". Coverage ratio = reserves / NPLs = \$1.3 billion / \$1.6 billion = 0.81 = **81%**. So the bank has reserved 81 cents for every dollar of bad loans; if those loans are worth, say, only 50 cents on the dollar in a workout, the bank is *under*-reserved and faces a further loss. Quantify it: if recoveries are 50%, the true loss on \$1.6 billion of NPLs is \$0.8 billion, but only — wait, reserves of \$1.3 billion already exceed that \$0.8 billion, so the bank is actually *over*-reserved against a 50%-recovery scenario by \$0.5 billion. Coverage below 100% isn't automatically a problem; it depends on the expected recovery. The intuition: NPL tells you the size of the problem and coverage tells you how much is already paid for, but you must pair coverage with an assumed recovery rate to know if the reserve is enough.

### Liquidity: solvent is not the same as funded

The most important thing the risk section teaches is that **liquidity is not solvency**. A bank can be solvent — assets worth more than liabilities — and still die, because its assets are long-dated (loans, bonds) while its funding can leave tomorrow (deposits). If everyone asks for their money at once, no solvent bank can pay, because the money is lent out.

The LCR is the regulatory answer: high-quality liquid assets (HQLA — cash, central-bank reserves, government bonds) divided by the net cash outflows expected in a 30-day stress. It must be at least 100%.

#### Worked example: computing the LCR

A bank's Pillar 3 liquidity table gives:

- High-quality liquid assets (HQLA): \$30 billion
- Expected net cash outflows over a 30-day stress: \$24 billion

LCR = HQLA / net outflows = \$30 billion / \$24 billion = 1.25 = **125%**. The bank holds 25% more liquid assets than it would need to survive a 30-day run, which is a comfortable buffer. Now ask the SVB question: what if the deposit base is mostly *uninsured*, and a digital run pulls money far faster than the regulatory 30-day stress assumes? Then the modelled outflow is too low, and a 125% LCR overstates the bank's safety. The intuition: the LCR is a calibrated guess about how fast money leaves, and a bank with concentrated, uninsured, tech-savvy depositors can lose money faster than any standard stress scenario assumes — which is precisely why you read the deposit footnote alongside the LCR. (More in [liquidity management: LCR, NSFR and the buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer).)

## The footnotes that bite — where the danger actually lives

We've now read the four major sections. But the recurring theme — SVB's HTM loss, the deposit composition, the loan concentration — is that the danger was in a *footnote*, not on the face of the statements. So the most senior move in bank analysis is to go straight to the four footnotes that bite. The matrix below is the map.

![footnotes that bite HTM losses uninsured deposits concentrations off-balance-sheet](/imgs/blogs/how-to-read-a-banks-annual-report-the-analysts-checklist-7.png)

### 1. The HTM unrealised-loss footnote

Covered above: the securities note discloses the fair value of held-to-maturity bonds, which the balance sheet carries at cost. Mark it to market and subtract from equity. In a rising-rate environment, this is the first footnote to open, every time.

### 2. The uninsured-deposit share

Deposit insurance (the FDIC's \$250,000 per depositor per bank in the US, €100,000 in the EU, £85,000 in the UK) is what stops ordinary savers from running: they don't need to, because the government guarantees their money. *Uninsured* deposits — balances above the cap, typically held by businesses and the wealthy — have every incentive to flee at the first whiff of trouble. They are the hot money. The deposit footnote (and Pillar 3 funding tables) disclose the uninsured share.

SVB's uninsured-deposit share was about **94%** — almost the entire deposit base was above the insured cap, held by venture-backed startups that talked to each other constantly. That is the most dangerous funding profile imaginable: a huge pool of money with no reason to stay and every channel to coordinate a run. The 94% figure, sitting in the footnotes, was a five-alarm fire on its own.

#### Worked example: estimating run risk from the uninsured share

Take a bank with \$175 billion of deposits and a 94% uninsured share (SVB's actual profile). Uninsured deposits = 0.94 × \$175 billion = **\$164.5 billion**. That's the pool with an incentive to run. Compare it to the bank's liquid assets: if HQLA is, say, \$40 billion, then even a partial run — 30% of the uninsured pool, or about \$49 billion — exceeds the entire liquid buffer. The bank would be forced to sell its HTM bonds at a loss to meet withdrawals, realising the very loss the footnote disclosed. The intuition: the uninsured-deposit share, multiplied by total deposits and compared to liquid assets, tells you whether the bank could survive even a partial run — and for SVB the answer, visible in the footnotes, was no.

### 3. Loan concentrations

The loan footnote breaks the book down by sector, geography, and single-name exposure. A bank with a quarter of its book in one segment is exposed to that segment's cycle. The 2023 regional-bank stress put a spotlight on commercial-real-estate concentration; the 2008 failures were mortgage concentration; the S&L crisis was a duration-mismatched concentration in long mortgages funded by short deposits. Read the concentration table and ask: what single shock takes out a big slice of this book?

### 4. Off-balance-sheet commitments

The most easily missed footnote. Banks make *commitments* — undrawn credit lines, loan guarantees, letters of credit — that don't appear as assets or liabilities until they're drawn. In a crisis, corporate borrowers draw their committed lines all at once (to hoard cash), and the bank must fund billions it never had on its balance sheet, exactly when funding is hardest to find. The commitments-and-contingencies footnote tells you the size of this hidden call on the bank's liquidity. In March 2020, US banks saw a wave of revolving-credit drawdowns; banks that had under-appreciated their off-balance-sheet commitments scrambled.

## The ratio dashboard — putting the six numbers together

We can now return to the cover figure and read it as a working dashboard rather than a definition. The discipline is: compute all six, place each against its healthy range, and — crucially — read the *pattern across them*, because the lies hide in the combinations.

- A high NIM **with** rising NPLs = reaching for yield, not strength.
- A strong CET1 ratio **with** a thin leverage ratio = risk-weighting games.
- A healthy LCR **with** a 94% uninsured-deposit base = a liquidity buffer calibrated for the wrong run.
- A rising ROE **with** rising leverage and falling coverage = earnings borrowed from the future.

The benchmark for the whole package is ROE, the number investors ultimately judge. The chart below shows the industry's ROE through the cycle, so any single bank can be placed against it.

![US banking industry return on equity 2010 to 2024 with healthy band](/imgs/blogs/how-to-read-a-banks-annual-report-the-analysts-checklist-3.png)

Industry ROE sits in the 10–15% "healthy franchise" band in normal years, collapses in crises (5.85% in 2010, 6.65% in 2020), and recovers afterward. Notice the shape: the troughs line up precisely with the years the banking system was absorbing losses — the aftermath of 2008 and the pandemic shock of 2020 — which is the visible fingerprint of the credit cycle on bank earnings. So a bank earning 11% ROE in a normal year is a typical, healthy franchise. One earning 22% deserves the analyst's hardest question: is this a genuinely superior franchise (a cheap deposit base, real pricing power, a fee business), or is the bank simply taking more risk and more leverage to manufacture a return that will reverse violently in the next downturn? Through the leverage identity, ROE = ROA × leverage, so an outsized ROE built on outsized leverage is fragility wearing the mask of excellence. (See [ROE, ROA and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged).)

#### Worked example: decomposing ROE with the leverage identity

Two banks both report 14% ROE. Bank A has 1.4% ROA and 10× leverage (equity = 10% of assets): 1.4% × 10 = 14%. Bank B has 0.7% ROA and 20× leverage (equity = 5% of assets): 0.7% × 20 = 14%. Same headline ROE, completely different banks. Bank A earns a high return on its assets and is modestly levered — a strong, safe franchise. Bank B earns a mediocre return and doubles it with leverage — and recall that 5% equity means a 5% asset loss wipes out the owners, versus 10% for Bank A. The intuition: identical ROEs can hide opposite risk profiles, so you must always split ROE into its ROA and leverage components before you call a bank "good".

## Common misconceptions

A few beliefs that trip up new bank analysts, each corrected with a number.

**"A profitable bank is a safe bank."** No. SVB was profitable right up to its collapse — its income statement looked fine. Banks die of *funding* and *capital*, not of unprofitability. A bank can post a record year and fail the next quarter when an unrealised bond loss meets a deposit run. Read the funding and the footnotes before you read the profit.

**"A high CET1 ratio means the bank is well-capitalised."** Not necessarily. CET1 is divided by *risk-weighted* assets, and the risk-weighting can be flattering. SVB's reported CET1 looked adequate; marked to market, true equity was about −\$1 billion (\$16 billion reported equity minus a \$17 billion HTM loss). Always cross-check CET1 against the *leverage* ratio, which ignores risk-weighting, and against the HTM footnote.

**"Deposits are stable funding, so a deposit-funded bank is safe."** It depends entirely on *which* deposits. Insured retail deposits are sticky; uninsured, concentrated, digitally-connected deposits are the fastest money in the world. SVB's 94% uninsured base lost \$42 billion in a single day. The deposit footnote, not the deposit total, tells you the truth.

**"A high net interest margin is always good."** A wide spread can mean a cheap deposit franchise (good) or lending to risky borrowers at high rates (bad). A NIM well above the ~3% industry line, paired with a rising NPL ratio, usually means the bank is being paid more because it's taking more credit risk — and the losses arrive a year or two later. NIM must be read against asset quality.

**"The audited statements contain everything important."** The audited *statements* are summary; the important detail is in the *notes*, which are also audited but easy to skip. The HTM loss, the uninsured share, the loan concentration, the off-balance-sheet commitments — none of these jump out on the face of the balance sheet. The face of the statements is where you start; the footnotes are where you find out whether to trust them.

**"If the auditor signed off, the numbers must be sound."** An audit opinion confirms the statements comply with the accounting rules — *not* that the bank is safe. HTM accounting at cost is perfectly compliant; SVB's auditor signed a clean opinion, and the \$17 billion unrealised loss was disclosed exactly as the rules require, in the footnote. Compliant and safe are different things. And in the rare cases where the numbers are outright fabricated — Wirecard's roughly €1.9 billion of cash that simply did not exist passed audit for years — the lesson is sharper still: an audit is a floor, not a guarantee. Read the report as if you, not the auditor, are the last line of defence.

**"A single year's report tells you what you need."** A snapshot can hide a trajectory. Credit Suisse cleared its regulatory ratios right up to the end while bleeding deposits and assets quarter after quarter; the danger was the *trend*, visible only across several years of reports. Always pull three to five years and read the direction of travel — deposits, NIM, NPLs, coverage, capital headroom — not just the latest level.

## How it shows up in real banks: what SVB's last 10-K would have told you

Let's run the checklist, in order, on the one report everyone wishes they'd read: Silicon Valley Bank's final annual disclosure, as of late 2022. Every number below was public months before the March 2023 collapse.

**Step 1 — Funding.** Total deposits about \$175 billion. The deposit footnote disclosed an uninsured share of roughly **94%**. That alone is the single most alarming funding profile a US bank could present: \$164.5 billion of deposits with every incentive and ability to run. The base was also wildly concentrated — venture-backed technology and life-sciences startups, a tightly networked community that would coordinate a run by group chat. A reader who stopped at the funding section already had the thesis.

**Step 2 — Assets.** Total assets about \$209 billion. The securities book was enormous relative to loans, and a huge slice — about **\$91 billion** — was classified as held-to-maturity, carried at amortised cost. Those were long-dated, low-coupon bonds bought when rates were near zero. By late 2022 the Fed had hiked aggressively, and long bonds had fallen hard. The asset side was a duration trap: long, fixed-rate, rate-sensitive assets funded by deposits that could leave overnight.

**Step 3 — Capital.** On the face of it, fine. Reported equity about \$16 billion; the regulatory CET1 ratio, computed on the unmarked book, looked adequate. This is exactly where the headline reassures and the footnote betrays.

**Step 4 — Risk.** The asset-quality numbers weren't the issue — SVB's loans weren't defaulting. The risk was *interest-rate risk in the banking book*, the mismatch between long fixed-rate assets and short flighty funding. That risk doesn't show up in the NPL ratio. It shows up in the securities footnote.

**Step 5 — The footnote that bit.** The securities note disclosed an unrealised loss across AFS and HTM of about **\$17 billion**. Mark the HTM book to market and subtract from the \$16 billion of reported equity: true economic equity was about **−\$1 billion**. The bank was, on a fair-value basis, insolvent. The instant it was forced to sell securities to meet withdrawals — which is exactly what happened, triggering a \$1.8 billion realised loss on a securities sale that spooked the market — the paper loss became real and the run began. On March 9, 2023, \$42 billion of deposits tried to leave in a single day; roughly another \$100 billion was queued for March 10 before regulators seized the bank.

The checklist, run in order, would have flagged SVB twice — at Step 1 (the 94% uninsured base) and at Step 5 (the HTM loss against thin equity) — both from disclosures that had been public for months. (The full narrative, including the contagion to Credit Suisse, is in [the SVB/Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**A second pattern: the slow-trust failure.** Not every bank dies of one footnote. Credit Suisse, in the same month, died of a decade of eroding trust — scandals, losses, leadership churn — that culminated in clients pulling about CHF 110 billion in a single quarter. There the tell wasn't one number but a *trajectory*: deposits and assets-under-management shrinking quarter after quarter, the franchise bleeding even as the capital ratios stayed nominally above the minimums. The lesson for the checklist: read the *trend* in funding and franchise metrics across several years, not just the latest snapshot. A bank can be above every regulatory line and still be dying, if the line on the deposit chart points down.

**A third pattern: the concentration failure.** Washington Mutual in 2008 — the largest bank failure in US history, \$307 billion in assets — was a concentration story. Its loan book was packed with risky mortgages just as housing collapsed. The loan footnote, read for concentration, would have shown the over-weighting years before the seizure. Same shape as the S&L crisis and the regional-bank commercial-real-estate worries of 2023: one over-weighted exposure, one cycle, one capital wipe.

Across all three, the spine holds: a leveraged, confidence-funded machine fails when losses (SVB's bond loss, WaMu's mortgage loss) breach the thin equity cushion faster than the bank can raise capital, *or* when confidence (Credit Suisse's depositors) evaporates faster than the liquidity buffer can pay. The annual report discloses both the loss potential and the funding fragility — you just have to read it in the order the failure happens.

## The takeaway: your repeatable bank-analysis checklist

Here is the whole post compressed into a checklist you can run on any bank's 10-K, annual report, or Pillar 3 disclosure. Work top to bottom — it's ordered the way a bank actually dies, so the early steps catch the fastest failures.

**1. Funding (read this first).**
- Deposits as a share of total funding — higher and stickier is safer. Wholesale/repo reliance is a fragility.
- CASA ratio and deposit composition — non-interest current accounts are the franchise.
- *Open the deposit footnote: what is the uninsured-deposit share?* Above ~50% with concentration is a flashing light; 94% is a fire.

**2. Assets.**
- Loan book by sector and geography — hunt for concentration. What single shock takes out a big slice?
- *Open the securities footnote: how big is the HTM book, and what is the unrealised loss?* Mark it to market and subtract from equity.

**3. Capital.**
- CET1 ratio versus the regulatory demand (minimum + buffers + any surcharge). How many points of headroom?
- Leverage ratio as the backstop — if it's thin while CET1 looks fat, the risk-weighting is doing suspicious work.
- Stress it: how big a loss before the bank drops into the buffer (dividends frozen) or below the minimum (resolution)?

**4. Risk.**
- NPL ratio and its *trend*. Rising is the tell.
- Coverage ratio, paired with an assumed recovery rate — is the reserve actually enough?
- LCR and NSFR — but cross-check the LCR against the uninsured-deposit footnote, because a standard 30-day stress understates a digital run.

**5. The footnotes that bite.**
- HTM unrealised losses (Step 2).
- Uninsured-deposit share (Step 1).
- Loan concentrations (Step 2).
- Off-balance-sheet commitments — the hidden call on liquidity when borrowers draw their lines.

**6. The dashboard, read as a pattern.** Compute NIM, efficiency, CET1, NPL/coverage, LCR, and ROE; place each against its healthy range; then read the *combinations*. High NIM + rising NPLs = reaching. Fat CET1 + thin leverage = games. Strong LCR + uninsured base = wrong run. High ROE + high leverage + falling coverage = earnings borrowed from the future. Decompose ROE into ROA × leverage before you call any bank "good".

**7. Read the narrative last, and skeptically.** The CEO letter is the only part written to persuade. By the time you reach it, you should already know whether to believe it. A useful tell: compare what the letter emphasises against what the footnotes contain. A management that talks at length about growth and digital transformation while the deposit footnote shows a flighty uninsured base and the securities footnote shows a buried bond loss is steering your eyes away from exactly where you should be looking.

**8. Read across years, not just the latest one.** Pull three to five years of the same bank's reports and chart the direction of travel: deposits, NIM, NPLs, coverage, capital headroom, the litigation footnote. A bank can be above every regulatory line in a single snapshot and still be dying if those lines point the wrong way. The snapshot tells you where the bank stands; the trend tells you where it's going.

The deeper point — the one that ties this whole series together — is that a bank is a leveraged, confidence-funded maturity-transformation machine, and its annual report is a complete blueprint of that machine's fault lines, written in a language that conceals the danger in the footnotes. Reading it well is not about memorising ratios. It's about reading in failure order, distrusting any single number, and always, always opening the footnote that the headline is hoping you'll skip. Run that checklist, and the report stops being a brochure and becomes what it really is: a confession, if you read it in the right order.

## Further reading & cross-links

- [Reading a bank balance sheet: assets, liabilities and equity](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity) — the structure behind every number on the checklist.
- [The income statement of a bank: net interest income, fees and provisions](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions) — where NIM and the efficiency ratio come from.
- [Risk-weighted assets and how capital ratios really work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work) — the CET1 denominator and the leverage-ratio backstop in detail.
- [Liquidity management: LCR, NSFR and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — why a solvent bank can still die.
- [ROE, ROA and the leverage identity: how a bank is judged](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged) — decomposing the return investors care about.
- [Silicon Valley Bank 2023: the duration trap and the 36-hour digital run](/blog/trading/banking/silicon-valley-bank-2023-the-duration-trap-and-the-36-hour-digital-run) — the case the checklist would have caught.
- [SVB and Credit Suisse, 2023: the bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) — the system-level view of both failures.
- [BIS and Basel: the architecture of bank regulation](/blog/trading/finance/bis-and-basel-bank-regulation) — where Pillar 3 and the capital rules come from.
